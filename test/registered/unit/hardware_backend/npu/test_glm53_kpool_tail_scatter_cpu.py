"""CPU interpreter checks for the opt-in GLM-5.3 D tail scatter.

Load isolated production kernels and indexer methods without importing SGLang
or initializing an accelerator. CANN LightningIndexer is a recording stub;
these tests establish copy/state equivalence, not NPU compilation or latency.
"""

import ast
import importlib.util
import os
import unittest
from copy import copy
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Optional
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[5]
SOURCE = ROOT / "python/sglang/srt/hardware_backend/npu/attention/glm53"
FLAG = "SGLANG_GLM53_FUSED_KPOOL_TAIL_SCATTER"


def load_kernel(name):
    spec = importlib.util.spec_from_file_location(f"_cpu_{name}", SOURCE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    # The decorator captures interpreter mode; do not affect other test modules.
    with patch.dict(os.environ, {"TRITON_INTERPRET": "1"}):
        spec.loader.exec_module(module)
    return module


fused = load_kernel("kpool_tail_scatter_npu")
separate = load_kernel("graph_ops")
metadata_kernel = load_kernel("kpool_metadata_npu")


def load_indexer(namespace):
    tree = ast.parse((SOURCE / "kpool_indexer.py").read_text())
    methods = {
        "_init_ascend_kpool",
        "_project_q_key_weights",
        "_project_head_weights",
        "_compress",
        "_decode_topk",
        "_verify_topk",
        "_draft_extend_topk",
        "_expand_with_tail",
        "forward_npu",
    }
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {
            "_ascend_lightning_weights",
            "_as_ascend_sparse_indices",
        }:
            nodes.append(node)
        if isinstance(node, ast.ClassDef) and node.name == "AscendIndexerKPoolMixin":
            node.body = [
                n
                for n in node.body
                if isinstance(n, ast.FunctionDef) and n.name in methods
            ]
            nodes.append(node)

    class IsolateImports(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            if node.module in {"kpool_metadata_npu", "compact_index"}:
                return ast.Pass()
            return node

    module = IsolateImports().visit(ast.Module(body=nodes, type_ignores=[]))
    namespace.update(
        torch=torch, F=F, os=os, copy=copy, Optional=Optional, ForwardBatch=object
    )
    exec(
        compile(
            ast.fix_missing_locations(module), str(SOURCE / "kpool_indexer.py"), "exec"
        ),
        namespace,
    )
    return type(
        "CPUIndexer", (namespace["AscendIndexerKPoolMixin"], torch.nn.Module), {}
    )


def make_indexer(enabled, mode="decode"):
    calls = []

    def lightning(**kwargs):
        calls.append(
            {
                k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in kwargs.items()
            }
        )
        bs = kwargs["query"].shape[0]
        ids = torch.arange(kwargs["sparse_count"], dtype=torch.int32).repeat(bs, 1, 1)
        return ids, None

    namespace = {
        "get_server_args": lambda: NS(
            disaggregation_mode=mode, max_running_requests=16
        ),
        "get_device": lambda: NS(device=torch.device("cpu")),
        "get_spec": lambda: NS(speculative_num_draft_tokens=None),
        "_get_full_attn_metadata": lambda batch: batch._ascend_kpool_metadata,
        "_get_index_k_buffer": lambda batch, _: batch.cache,
        "index_block_table": lambda batch, table: table,
        "decode_metadata": metadata_kernel.decode_metadata,
        "scatter_rows_": Mock(wraps=separate.scatter_rows_),
        "scatter_kpool_tail_": Mock(wraps=fused.scatter_kpool_tail_),
        "torch_npu": NS(npu_lightning_indexer=lightning),
    }
    indexer = load_indexer(namespace)()
    indexer.index_kpool = 4
    indexer.index_topk = 8
    indexer.head_dim = 128
    indexer.n_heads = 2
    indexer.hidden_size = 16
    indexer.softmax_scale = 128**-0.5
    indexer.index_kpool_compress_ape = torch.randn(4, 128)
    with patch.dict(os.environ, {} if enabled is None else {FLAG: str(int(enabled))}):
        if enabled is None:
            os.environ.pop(FLAG, None)
        indexer._init_ascend_kpool()
    indexer._glm53_draft_graph_steps = 4
    indexer._kpool_mtp_tail_k = torch.zeros(16, 4, 4, 128, dtype=torch.bfloat16)
    indexer._kpool_mtp_tail_score = torch.zeros(16, 4, 4, 128)
    return indexer, namespace, calls


def assert_bits_equal(test, actual, expected):
    test.assertTrue(
        torch.equal(
            actual.contiguous().view(torch.uint8),
            expected.contiguous().view(torch.uint8),
        )
    )


class TailScatterCPU(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(53)
        self.env = patch.dict(
            os.environ, {"SGLANG_GLM53_KPOOL_EXPAND": "0", "TRITON_INTERPRET": "1"}
        )
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_only_decode_with_explicit_opt_in(self):
        for mode in ("prefill", "decode", "null"):
            for enabled in (None, False, True):
                with self.subTest(mode=mode, enabled=enabled):
                    indexer, _, _ = make_indexer(enabled, mode)
                    self.assertEqual(
                        indexer._fuse_kpool_tail_scatter,
                        mode == "decode" and enabled is True,
                    )

    def test_interpreted_kernel_bitwise_matches_separate_scatters(self):
        # Global C4/C8, local DP4 batches 1/2, and padded graph buckets.
        for bs in (0, 1, 2, 4, 8, 16):
            for width in (128, 129):
                with self.subTest(bs=bs, width=width):
                    old_k = torch.randn(18, 4, width, dtype=torch.bfloat16)
                    old_s = torch.randn(18, 4, width)
                    new_k, new_s = old_k.clone(), old_s.clone()
                    addresses = new_k.data_ptr(), new_s.data_ptr()
                    # Noncontiguous inputs exercise the same packing as graph_ops.
                    key = torch.randn(bs, width * 2, dtype=torch.bfloat16)[:, ::2]
                    score = torch.randn(bs, width * 2)[:, ::2]
                    rows = torch.arange(1, bs + 1, dtype=torch.int64) * 4
                    for step in range(4):
                        rows.add_(step > 0)  # Mutate the same row tensor between calls.
                        if bs:
                            separate.scatter_rows_(old_k, rows, key)
                            separate.scatter_rows_(old_s, rows, score)
                        fused.scatter_kpool_tail_(new_k, new_s, rows, key, score)
                        assert_bits_equal(self, new_k, old_k)
                        assert_bits_equal(self, new_s, old_s)
                        self.assertEqual(
                            addresses, (new_k.data_ptr(), new_s.data_ptr())
                        )

    def test_copy_preserves_special_value_bits(self):
        # Includes signed zero, infinities, subnormals and NaN payloads.
        key = torch.tensor(
            [0, -32768, 1, 32640, -128, 32705, 32767, -1], dtype=torch.int16
        ).view(torch.bfloat16)[None]
        score = torch.tensor(
            [0, -2147483648, 1, 2139095040, -8388608, 2143289345, 2147483647, -1],
            dtype=torch.int32,
        ).view(torch.float32)[None]
        k, s = torch.zeros(2, 4, 8, dtype=torch.bfloat16), torch.zeros(2, 4, 8)
        fused.scatter_kpool_tail_(
            k, s, torch.tensor([5], dtype=torch.int32), key, score
        )
        assert_bits_equal(self, k[1, 1], key[0])
        assert_bits_equal(self, s[1, 1], score[0])

    def test_padding_aliases_do_not_modify_real_requests(self):
        k, s = torch.ones(4, 4, 128, dtype=torch.bfloat16), torch.ones(4, 4, 128)
        key, score = torch.randn(4, 128, dtype=torch.bfloat16), torch.randn(4, 128)
        fused.scatter_kpool_tail_(k, s, torch.tensor([0, 0, 5, 10]), key, score)
        assert_bits_equal(self, k[1, 1], key[2])
        assert_bits_equal(self, s[1, 1], score[2])
        assert_bits_equal(self, k[2, 2], key[3])
        assert_bits_equal(self, s[2, 2], score[3])
        self.assertTrue(torch.equal(k[3], torch.ones_like(k[3])))
        self.assertTrue(torch.equal(s[3], torch.ones_like(s[3])))

    def test_rejects_lossy_or_invalid_layouts(self):
        k, s = torch.zeros(4, 4, 128, dtype=torch.bfloat16), torch.zeros(4, 4, 128)
        key, score = k[0, :2], s[0, :2]
        cases = (
            (k, s, torch.tensor([1]), key, score),
            (k, s, torch.tensor([1, 2]), key.float(), score),
            (k.transpose(0, 1), s, torch.tensor([1, 2]), key, score),
        )
        for args in cases:
            with self.assertRaises(ValueError):
                fused.scatter_kpool_tail_(*args)

    def test_decode_long_positions_and_mtp_state_match_baseline(self):
        for context, bs in ((256 * 1024, 8), (512 * 1024, 4)):
            for mode in ("decode", "verify", "draft"):
                with self.subTest(context=context, bs=bs, mode=mode):
                    old, old_ns, old_calls = make_indexer(False)
                    new, new_ns, new_calls = make_indexer(True)
                    new.index_kpool_compress_ape.copy_(old.index_kpool_compress_ape)
                    old._kpool_tail_k.normal_()
                    old._kpool_tail_score.normal_()
                    new._kpool_tail_k.copy_(old._kpool_tail_k)
                    new._kpool_tail_score.copy_(old._kpool_tail_score)
                    # Long logical positions with non-identity physical page IDs.
                    table = torch.arange((context + 1024) // 64, dtype=torch.int32)[
                        None
                    ].repeat(bs, 1)
                    table = table.remainder(7) + 1
                    table[:, -1] = -1  # Unused page columns must remain safely clamped.
                    req = torch.arange(1, bs + 1, dtype=torch.int64)
                    req[-1] = 0
                    cache = torch.randn(9, 64, 1, 128, dtype=torch.bfloat16)
                    batch = NS(
                        batch_size=bs,
                        req_pool_indices=req,
                        cache=cache,
                        num_token_non_padded=1,
                        _ascend_kpool_metadata=NS(
                            block_tables=table,
                            seq_lens=torch.full((bs,), context, dtype=torch.int32),
                        ),
                    )
                    batch._ascend_kpool_metadata.seq_lens[-1] = 0
                    other = copy(batch)
                    other.cache = cache.clone()
                    tokens = bs if mode == "decode" else bs * 4
                    q = torch.randn(tokens, 2, 128, dtype=torch.bfloat16)
                    key = torch.randn(tokens, 128, dtype=torch.bfloat16)
                    weights = torch.randn(tokens, 2)
                    scores = torch.randn(tokens, 128)
                    if mode == "decode":
                        positions = torch.full((bs,), context - 2, dtype=torch.int64)
                        method = "_decode_topk"
                        repeats = 4
                    else:
                        positions = (torch.arange(4) + context - 2).repeat(bs)
                        batch.spec_info = NS(
                            draft_token_num=4,
                            num_accept_tokens=torch.arange(bs).remainder(5).int(),
                        )
                        batch.extend_seq_lens = torch.full((bs,), 4, dtype=torch.int32)
                        other.spec_info = batch.spec_info
                        other.extend_seq_lens = batch.extend_seq_lens
                        method = (
                            "_verify_topk" if mode == "verify" else "_draft_extend_topk"
                        )
                        repeats = 1
                    for _ in range(repeats):
                        results = [
                            getattr(i, method)(
                                q, key, weights, scores, positions, b, table, 0
                            )
                            for i, b in ((old, batch), (new, other))
                        ]
                        assert_bits_equal(self, results[0], results[1])
                        for attr in (
                            "_kpool_tail_k",
                            "_kpool_tail_score",
                            "_kpool_mtp_tail_k",
                            "_kpool_mtp_tail_score",
                        ):
                            assert_bits_equal(
                                self, getattr(old, attr), getattr(new, attr)
                            )
                        assert_bits_equal(self, batch.cache, other.cache)
                        positions.add_(1)
                    self.assertEqual(len(old_calls), len(new_calls))
                    for a, b in zip(old_calls, new_calls):
                        for name in a:
                            if isinstance(a[name], torch.Tensor):
                                assert_bits_equal(self, a[name], b[name])
                            else:
                                self.assertEqual(a[name], b[name])
                    self.assertEqual(
                        new_ns["scatter_kpool_tail_"].call_count, len(new_calls)
                    )
                    self.assertEqual(old_ns["scatter_kpool_tail_"].call_count, 0)
                    self.assertEqual(
                        old_ns["scatter_rows_"].call_count
                        - new_ns["scatter_rows_"].call_count,
                        2 * len(new_calls),
                    )

    def test_forward_projects_once_before_decode_verify_or_draft(self):
        for mode in ("decode", "verify", "draft"):
            indexer, _, _ = make_indexer(False)
            indexer.wq_b = Mock(
                side_effect=lambda x: (
                    F.linear(x, torch.ones(256, 8, dtype=x.dtype)),
                    None,
                )
            )
            indexer.wk = Mock(
                side_effect=lambda x: (
                    F.linear(x, torch.ones(128, 16, dtype=x.dtype)),
                    None,
                )
            )
            indexer.weights_proj = Mock(
                side_effect=lambda x: (F.linear(x, torch.ones(2, 16)), None)
            )
            indexer.k_norm = lambda x: x
            indexer.index_kpool_compress_gate = torch.ones(
                128, 16, dtype=torch.bfloat16
            )
            output = torch.zeros(8, 11, dtype=torch.int32)
            method = {
                "decode": "_decode_topk",
                "verify": "_verify_topk",
                "draft": "_draft_extend_topk",
            }[mode]
            setattr(indexer, method, Mock(return_value=output))
            batch = NS(
                _ascend_kpool_metadata=NS(
                    block_tables=torch.zeros(2, 8, dtype=torch.int32)
                ),
                forward_mode=NS(
                    is_idle=lambda: False,
                    is_target_verify=lambda: mode == "verify",
                    is_draft_extend_v2=lambda: mode == "draft",
                    is_extend=lambda: False,
                ),
            )
            x, q_lora = torch.randn(8, 16, dtype=torch.bfloat16), torch.randn(
                8, 8, dtype=torch.bfloat16
            )
            result = indexer.forward_npu(x, q_lora, torch.arange(8), batch, 0)
            self.assertEqual(tuple(result.shape), (8, 1, 11))
            for projection in (indexer.wq_b, indexer.wk, indexer.weights_proj):
                projection.assert_called_once()
            args = getattr(indexer, method).call_args.args
            self.assertEqual(args[1].dtype, torch.bfloat16)
            self.assertEqual(args[2].dtype, torch.float32)
            self.assertEqual(
                indexer.weights_proj.call_args.args[0].dtype, torch.float32
            )


if __name__ == "__main__":
    unittest.main()
