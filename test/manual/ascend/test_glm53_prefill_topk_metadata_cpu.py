"""CPU checks of the production P scorer's integer metadata reuse.

CUDA_VISIBLE_DEVICES='' TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 \
    conda run -n sgl-0511 python -B \
    test/manual/ascend/test_glm53_prefill_topk_metadata_cpu.py -v

Only the query's device tag and factory device arguments are redirected. All
score arithmetic and TopK run on CPU. This does not establish NPU parity or
model accuracy; no device runtime or service is imported.
"""

import ast
import os
import unittest
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

SOURCE = (
    Path(__file__).resolve().parents[3]
    / "python/sglang/srt/hardware_backend/npu/attention/glm53/kpool_indexer.py"
)
FLAG = "SGLANG_GLM53_PREFILL_TOPK_METADATA_REUSE"


class TaggedQuery:
    device = SimpleNamespace(type="npu")

    def __init__(self, tensor):
        self.tensor = tensor

    def __getitem__(self, key):
        return self.tensor[key]


class CPUTorch:
    """Count actual metadata calls and redirect only tensor factories to CPU."""

    def __init__(self):
        self.calls = Counter()

    def __getattr__(self, name):
        return getattr(torch, name)

    def arange(self, *args, **kwargs):
        self.calls["arange"] += 1
        kwargs["device"] = "cpu"
        return torch.arange(*args, **kwargs)

    def full(self, *args, **kwargs):
        kwargs["device"] = "cpu"
        return torch.full(*args, **kwargs)

    def div(self, *args, **kwargs):
        self.calls["div"] += 1
        return torch.div(*args, **kwargs)


class ScoreTrace(TorchDispatchMode):
    """Record the unchanged float operations, masks and TopK, including bits."""

    watched = {
        "aten.bmm.default",
        "aten.relu_.default",
        "aten.mul_.Tensor",
        "aten.sum.dim_IntList",
        "aten.masked_fill_.Scalar",
        "aten.topk.default",
    }

    def __init__(self):
        super().__init__()
        self.records = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if str(func) in self.watched:
            outputs = out if isinstance(out, tuple) else (out,)
            signature = tuple(
                (tuple(x.shape), x.dtype) for x in args if isinstance(x, torch.Tensor)
            )
            self.records.append(
                (str(func), signature, [x.detach().clone() for x in outputs])
            )
        return out


def load_scorer(proxy, role="prefill"):
    names = {
        "_prefill_topk_tile_rows",
        "_prefill_pooled_score_enabled",
        "_prefill_pooled_postop_min_keys",
        "_prefill_pooled_topk",
    }
    tree = ast.parse(SOURCE.read_text())
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *functions,
        ],
        type_ignores=[],
    )
    namespace = dict(
        torch=proxy,
        F=F,
        os=os,
        get_server_args=lambda: SimpleNamespace(disaggregation_mode=role),
    )
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    return namespace["_prefill_pooled_topk"]


def inputs(lengths=(512,), ends=(512,), counts=None, heads=2, width=8):
    gen = torch.Generator(device="cpu").manual_seed(1018)
    q = torch.randn(sum(lengths), heads, width, generator=gen).bfloat16()
    weights = torch.randn(sum(lengths), heads, generator=gen)
    if counts is None:
        counts = [end // 4 for end in ends]
    keys = [torch.randn(n, width, generator=gen).bfloat16() for n in counts]
    batch = SimpleNamespace(
        batch_size=len(lengths),
        extend_seq_lens_cpu=list(lengths),
        seq_lens_cpu=list(ends),
    )
    return q, weights, keys, batch


def run_case(
    data,
    enabled,
    tile=32,
    role="prefill",
    tagged=True,
    trace=False,
    backend="native",
    topk=32,
):
    q, weights, keys, batch = data
    proxy = CPUTorch()
    scorer = load_scorer(proxy, role)
    backend_calls = []

    def relu_gate(logits, gates):
        backend_calls.append(tuple(logits.shape))
        logits.relu_()
        logits.mul_(gates.unsqueeze(-1))
        return True

    def fused_scores(query, gates, history):
        logits = torch.einsum("qhd,kd->qhk", query, history)
        relu_gate(logits, gates)
        return logits.sum(1)

    scorer.__globals__.update(
        _prefill_pooled_score_backend=lambda: (fused_scores, lambda *_: True),
        _prefill_pooled_postop_backend=lambda: relu_gate,
        _log_prefill_pooled_score_once=lambda *_: None,
        _log_prefill_pooled_postop_once=lambda *_: None,
    )
    indexer = SimpleNamespace(index_topk=topk, index_kpool=4)
    ctx = ScoreTrace() if trace else nullcontext()
    with patch.dict(
        os.environ,
        {
            FLAG: "1" if enabled else "0",
            "SGLANG_GLM53_PREFILL_TOPK_TILE_ROWS": str(tile),
            "SGLANG_GLM53_PREFILL_POOLED_SCORE": (
                "always" if backend == "fused" else "0"
            ),
            "SGLANG_GLM53_PREFILL_POOLED_POSTOP": (
                "auto" if backend == "postop" else "0"
            ),
            "SGLANG_GLM53_PREFILL_SCORE_VALIDATE": "0",
        },
    ), ctx:
        out = scorer(indexer, TaggedQuery(q) if tagged else q, weights, keys, batch)
    return out, proxy.calls, ctx.records if trace else [], backend_calls


class PrefillTopkMetadataTests(unittest.TestCase):
    def assert_bits_equal(self, actual, expected):
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(
            torch.equal(
                actual.contiguous().view(torch.uint8),
                expected.contiguous().view(torch.uint8),
            )
        )

    def compare(self, data, **kwargs):
        old = run_case(data, False, **kwargs)
        new = run_case(data, True, **kwargs)
        self.assert_bits_equal(new[0], old[0])
        self.assertEqual(len(new[2]), len(old[2]))
        for a, b in zip(new[2], old[2]):
            self.assertEqual(a[:2], b[:2])
            for av, bv in zip(a[2], b[2]):
                self.assert_bits_equal(av, bv)
        self.assertEqual(new[3], old[3])
        return old, new

    def test_float_bits_shapes_mask_and_topk_unchanged(self):
        # Non-aligned prefix crosses several 128-position causal boundaries.
        data = inputs(ends=(639,), heads=32, width=128)
        old, new = self.compare(data, trace=True)
        self.assertGreater(len(new[2]), 0)
        self.assertEqual(new[1], {"arange": 2, "div": 1})
        self.assertGreater(old[1]["arange"], new[1]["arange"])

    def test_all_tile_sizes_and_pool_tails(self):
        data = inputs(lengths=(513,), ends=(640,))
        for tile in (1, 7, 8, 31, 32):
            with self.subTest(tile=tile):
                self.compare(data, tile=tile)

    def test_varlen_short_request_original_batch_and_query_padding(self):
        data = inputs(lengths=(513, 17, 512), ends=(640, 18, 1543))
        q, weights, keys, batch = data
        batch._original_batch_size = 3
        batch.batch_size = 4
        batch.extend_seq_lens_cpu.append(None)
        batch.seq_lens_cpu.append(None)
        data = (F.pad(q, (0, 0, 0, 0, 0, 9)), weights, keys, batch)
        self.compare(data, tile=8)

    def test_empty_or_capped_history_and_topk_padding(self):
        for count in (0, 1, 7, 128):
            with self.subTest(count=count):
                self.compare(inputs(counts=[count]), topk=2048)

    def test_long_prefix_and_tp_fragment_end_before_history_end(self):
        # A TP fragment can end before the complete key history. Only its
        # candidate prefix may be scored, even though more keys are available.
        for end in (131072, 132608):
            with self.subTest(end=end):
                old, new = self.compare(inputs(ends=(end,), counts=[33152]))
                self.assertEqual(new[1], {"arange": 2, "div": 1})
                self.assertEqual(old[1], {"arange": 32, "div": 16})

    def test_ties_and_nonfinite_scores_keep_identical_topk_order(self):
        for special in (0.0, -0.0, float("inf"), float("nan")):
            with self.subTest(special=special):
                data = inputs()
                data[0].zero_()
                data[0][::37, 0, 0] = special
                self.compare(data, trace=True)

    def test_short_cpu_and_decode_paths_keep_old_metadata_calls(self):
        cases = [
            (inputs(lengths=(511,), ends=(511,)), {}),
            (inputs(), {"tagged": False}),
            (inputs(), {"role": "decode"}),
            (inputs(), {"role": "null"}),
        ]
        for data, kwargs in cases:
            with self.subTest(kwargs=kwargs, length=data[0].shape[0]):
                old, new = self.compare(data, **kwargs)
                self.assertEqual(old[1], new[1])

    def test_native_fused_and_postop_control_flow(self):
        # CPU stand-ins test compatibility, not the fused NPU kernels.
        for backend, end in (("native", 640), ("fused", 640), ("postop", 131072)):
            with self.subTest(backend=backend):
                old, new = self.compare(inputs(ends=(end,)), backend=backend)
                if backend != "native":
                    self.assertGreater(len(new[3]), 0)

    def test_profile_geometry_removes_repeated_metadata(self):
        old, new = self.compare(inputs(lengths=(2048,), ends=(2048,)))
        self.assertEqual(old[1], {"arange": 128, "div": 64})
        self.assertEqual(new[1], {"arange": 2, "div": 1})

    def test_default_off_without_server_context(self):
        proxy = CPUTorch()
        scorer = load_scorer(proxy)
        scorer.__globals__["get_server_args"] = lambda: self.fail("context queried")
        q, weights, keys, batch = inputs()
        with patch.dict(
            os.environ,
            {
                "SGLANG_GLM53_PREFILL_TOPK_TILE_ROWS": "32",
                "SGLANG_GLM53_PREFILL_POOLED_SCORE": "0",
                "SGLANG_GLM53_PREFILL_POOLED_POSTOP": "0",
            },
        ):
            os.environ.pop(FLAG, None)
            scorer(
                SimpleNamespace(index_topk=32, index_kpool=4),
                TaggedQuery(q),
                weights,
                keys,
                batch,
            )
        self.assertEqual(proxy.calls, {"arange": 32, "div": 16})


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
