"""CPU tests of the actual GLM-5.3 prefill scorer and causal pool contract.

Run: python -B test/manual/ascend/test_glm53_pooled_score_prefill_cpu.py -v
The production method is extracted because importing SGLang's NPU package on
CPU requires torch_npu. These tests do not claim Triton-Ascend kernel parity.
"""

import ast
import copy
import os
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F


SOURCE = (
    Path(__file__).resolve().parents[3]
    / "python/sglang/srt/hardware_backend/npu/attention/glm53/kpool_indexer.py"
)
KERNEL_SOURCE = SOURCE.with_name("pooled_score_prefill_npu.py")
TREE = ast.parse(SOURCE.read_text())


def actual_function(name, source=SOURCE):
    tree = TREE if source == SOURCE else ast.parse(source.read_text())
    node = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            copy.deepcopy(node),
        ],
        type_ignores=[],
    )
    scope = {"torch": torch, "F": F, "os": os, "get_server_args": lambda: None}
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), scope)
    return scope[name]


TILE_ROWS = actual_function("_prefill_topk_tile_rows")
GUARD = actual_function("_prefill_pooled_score_enabled")
SCORER = actual_function("_prefill_pooled_topk")
SCORER.__globals__["_prefill_pooled_postop_min_keys"] = actual_function(
    "_prefill_pooled_postop_min_keys"
)
SCORER.__globals__["_prefill_topk_tile_rows"] = TILE_ROWS
SCORER.__globals__["_prefill_pooled_score_enabled"] = GUARD
SUPPORTED = actual_function("supported", KERNEL_SOURCE)
LOG_ONCE = actual_function("_log_prefill_pooled_score_once")


def fake_npu_tensor(shape, dtype, device=None):
    if device is None:
        device = types.SimpleNamespace(type="npu", index=0)
    return types.SimpleNamespace(
        shape=shape,
        ndim=len(shape),
        dtype=dtype,
        device=device,
        is_contiguous=lambda: True,
    )


def oracle(q, weights, histories, lengths, ends, topk=8, pool=4):
    """Per-token oracle with no 128-tile assumption or future pool reads."""
    rows = []
    offset = 0
    for length, seq_end, keys in zip(lengths, ends, histories):
        first = seq_end - length
        for local in range(length):
            visible = min((first + local + 1) // pool, len(keys))
            scored = []
            for key_id in range(visible):
                # BF16 dot, BF16 in-place gate product and BF16 head sum match
                # the production tensor dtype boundaries.
                heads = torch.einsum("hd,d->h", q[offset + local], keys[key_id])
                heads.relu_()
                heads.mul_(weights[offset + local])
                scored.append((float(heads.sum()), key_id))
            scored.sort(key=lambda item: item[0], reverse=True)
            chosen = [key_id for _, key_id in scored[:topk]]
            rows.append(chosen + [-1] * (topk - len(chosen)))
        offset += length
    return torch.tensor(rows, dtype=torch.int32)


class PooledScorePrefillCPU(unittest.TestCase):
    def test_auto_score_branch_uses_legacy_tile8_and_preserves_causal_topk(self):
        # Tag only the outer query as NPU. Slices and all arithmetic remain on
        # CPU, allowing the production control flow to run without torch_npu.
        class TaggedQuery:
            device = types.SimpleNamespace(type="npu")

            def __init__(self, data):
                self.data = data

            def __getitem__(self, key):
                return self.data[key]

        class CPUTorch:
            def __getattr__(self, name):
                return getattr(torch, name)

            def arange(self, *args, **kwargs):
                kwargs.pop("device", None)
                return torch.arange(*args, **kwargs)

        q = torch.zeros((4, 32, 128), dtype=torch.bfloat16)
        q[:, 0, 0], q[:, 1, 1] = 1, 1
        keys = torch.zeros((4, 128), dtype=torch.bfloat16)
        keys[0, 0], keys[0, 1] = 8, -8
        keys[1, 0], keys[1, 1] = -8, 4
        keys[2, 0], keys[3, 1] = 2, 1
        weights = torch.zeros((4, 32), dtype=torch.float32)
        weights[:, 0], weights[:, 1] = -1, 1
        expected = oracle(q, weights, [keys], [4], [16])
        batch = types.SimpleNamespace(
            batch_size=1, extend_seq_lens_cpu=[4], seq_lens_cpu=[16]
        )
        indexer = types.SimpleNamespace(index_topk=32, index_kpool=4)
        calls = []

        def cpu_fused(query, gates, history):
            calls.append(tuple(query.shape))
            scores = torch.einsum("qhd,kd->qhk", query, history)
            scores.relu_()
            scores.mul_(gates.unsqueeze(-1))
            return scores.sum(1)

        globals_override = {
            "torch": CPUTorch(),
            "_prefill_pooled_score_backend": lambda: (cpu_fused, lambda *_: True),
            "_log_prefill_pooled_score_once": lambda *_: None,
        }
        with (
            patch.dict(SCORER.__globals__, globals_override),
            patch.dict(
                GUARD.__globals__,
                {
                    "get_server_args": lambda: types.SimpleNamespace(
                        disaggregation_mode="prefill"
                    )
                },
            ),
        ):
            for legacy_tile, fused_calls in ((8, 1), (32, 0)):
                calls.clear()
                with (
                    self.subTest(legacy_tile=legacy_tile),
                    patch.dict(
                        os.environ,
                        {
                            "SGLANG_GLM53_PREFILL_POOLED_SCORE": "auto",
                            "SGLANG_GLM53_PREFILL_TOPK_TILE_ROWS": str(legacy_tile),
                        },
                    ),
                ):
                    actual = SCORER(indexer, TaggedQuery(q), weights, [keys], batch)
                    self.assertTrue(torch.equal(actual, expected))
                    self.assertEqual(len(calls), fused_calls)

    def test_auto_requires_legacy_tile8_and_p_mode(self):
        npu = types.SimpleNamespace(type="npu")
        cpu = types.SimpleNamespace(type="cpu")
        GUARD.__globals__["get_server_args"] = lambda: types.SimpleNamespace(
            disaggregation_mode="prefill"
        )
        for setting in ("1", "auto"):
            with (
                self.subTest(setting=setting),
                patch.dict(os.environ, {"SGLANG_GLM53_PREFILL_POOLED_SCORE": setting}),
            ):
                self.assertTrue(GUARD(npu, 8))
                self.assertFalse(GUARD(npu, 32))
                self.assertFalse(GUARD(cpu, 8))
        with patch.dict(os.environ, {"SGLANG_GLM53_PREFILL_POOLED_SCORE": "always"}):
            self.assertTrue(GUARD(npu, 32))
            GUARD.__globals__["get_server_args"] = lambda: types.SimpleNamespace(
                disaggregation_mode="decode"
            )
            self.assertFalse(GUARD(npu, 8))
        with patch.dict(os.environ, {"SGLANG_GLM53_PREFILL_POOLED_SCORE": "0"}):
            self.assertFalse(GUARD(npu, 8))
        with patch.dict(os.environ):
            os.environ.pop("SGLANG_GLM53_PREFILL_POOLED_SCORE", None)
            self.assertFalse(GUARD(npu, 8))
        with patch.dict(os.environ, {"SGLANG_GLM53_PREFILL_POOLED_SCORE": "invalid"}):
            with self.assertRaisesRegex(ValueError, "PREFILL_POOLED_SCORE"):
                GUARD(npu, 8)

    def test_supported_device_and_shape_contract(self):
        q = fake_npu_tensor((2, 32, 128), torch.bfloat16)
        weights = fake_npu_tensor((2, 32), torch.float32, q.device)
        key = fake_npu_tensor((9, 128), torch.bfloat16, q.device)
        self.assertTrue(SUPPORTED(q, weights, [key]))

        other_npu = types.SimpleNamespace(type="npu", index=1)
        for bad_weights in (
            fake_npu_tensor((2, 32), torch.float32, other_npu),
            fake_npu_tensor((3, 32), torch.float32, q.device),
            fake_npu_tensor((2, 31), torch.float32, q.device),
        ):
            with self.subTest(weights=bad_weights):
                self.assertFalse(SUPPORTED(q, bad_weights, [key]))
        self.assertFalse(
            SUPPORTED(
                q, weights, [fake_npu_tensor((9, 128), torch.bfloat16, other_npu)]
            )
        )
        self.assertFalse(SUPPORTED(torch.empty((2, 32, 128)), weights, [key]))

    def test_activation_log_records_first_real_shape_once(self):
        q = fake_npu_tensor((2, 32, 128), torch.bfloat16)
        weights = fake_npu_tensor((2, 32), torch.float32, q.device)
        key = fake_npu_tensor((9, 128), torch.bfloat16, q.device)
        logger = Mock()
        LOG_ONCE.__globals__["logger"] = logger
        LOG_ONCE.__globals__["_reported_prefill_pooled_score"] = False
        LOG_ONCE(q, weights, key, 32)
        LOG_ONCE(q, weights, key, 32)
        logger.info.assert_called_once()
        message = logger.info.call_args.args[0] % logger.info.call_args.args[1:]
        self.assertIn("q_shape=(2, 32, 128)", message)
        self.assertIn("weights_dtype=torch.float32", message)
        self.assertIn("key_shape=(9, 128)", message)
        self.assertIn("tile_rows=32", message)

    def call(self, q, weights, histories, lengths, ends, *, rows=8, original=None):
        batch = types.SimpleNamespace(
            batch_size=len(lengths) if original is None else len(lengths) + 1,
            _original_batch_size=original,
            extend_seq_lens_cpu=lengths,
            seq_lens_cpu=ends,
        )
        indexer = types.SimpleNamespace(index_topk=32, index_kpool=4)
        with patch.dict(
            os.environ,
            {
                "SGLANG_GLM53_PREFILL_TOPK_TILE_ROWS": str(rows),
                "SGLANG_GLM53_PREFILL_POOLED_SCORE": "1",
            },
        ):
            return SCORER(indexer, q, weights, histories, batch)

    def test_signed_weights_and_relu_before_head_sum(self):
        # Head 0 votes for pool 0; a negative gate reverses it. Head 1 votes
        # for pool 1. Moving ReLU after the weighted head sum changes ranking.
        q = torch.zeros((1, 32, 128), dtype=torch.bfloat16)
        q[0, 0, 0], q[0, 1, 1] = 1, 1
        keys = torch.zeros((4, 128), dtype=torch.bfloat16)
        keys[0, 0], keys[0, 1] = 8, -8
        keys[1, 0], keys[1, 1] = -8, 4
        keys[2, 0], keys[3, 1] = 2, 1
        weights = torch.zeros((1, 32), dtype=torch.float32)
        weights[0, 0], weights[0, 1] = -1, 1
        result = self.call(q, weights, [keys], [1], [16])
        self.assertEqual(result[0, :4].tolist(), [1, 3, 2, 0])
        self.assertTrue(torch.equal(result, oracle(q, weights, [keys], [1], [16])))

    def test_prefix_varlen_pool_boundaries_and_tile_rows(self):
        lengths, ends = [9, 7, 6], [132, 263, 7]
        q = torch.zeros((sum(lengths), 32, 128), dtype=torch.bfloat16)
        q[:, 0, 0] = 1
        weights = torch.zeros((sum(lengths), 32), dtype=torch.float32)
        weights[:, 0] = 1
        weights[4:9, 0] = -1  # historical prefix pools win on these rows
        keys = [torch.zeros((end // 4, 128), dtype=torch.bfloat16) for end in ends]
        for key in keys:
            key[:, 0] = torch.arange(1, len(key) + 1, dtype=torch.bfloat16)
        expected = oracle(q, weights, keys, lengths, ends)
        for tile_rows in (1, 8, 32):
            with self.subTest(tile_rows=tile_rows):
                result = self.call(q, weights, keys, lengths, ends, rows=tile_rows)
                self.assertTrue(torch.equal(result, expected))
        # First request begins at position 123: the boundary at 128 lies
        # inside its extend, and its historical prefix remains searchable.
        self.assertTrue((expected[:9] >= 0).all())
        # Request 3 begins at position 1: no pool closes until position 3.
        self.assertEqual(expected[16].tolist(), [-1] * 8)
        self.assertEqual(expected[18].tolist(), [0] + [-1] * 7)

    def test_original_batch_size_ignores_padding_request(self):
        q = torch.ones((1, 32, 128), dtype=torch.bfloat16)
        weights = torch.ones((1, 32), dtype=torch.float32)
        keys = [torch.ones((1, 128), dtype=torch.bfloat16)]
        result = self.call(q, weights, keys, [1], [4], original=1)
        self.assertEqual(result[0].tolist(), [0] + [-1] * 7)


if __name__ == "__main__":
    unittest.main()
