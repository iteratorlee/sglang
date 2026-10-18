"""Small CPU guard/control-flow checks; no Triton interpreter or NPU import."""

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[3]
DIR = ROOT / "python/sglang/srt/hardware_backend/npu/attention/glm53"
BENCH = Path(__file__).with_name("bench_glm53_postop64_ab.py")
spec = importlib.util.spec_from_file_location("postop64_bench", BENCH)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
bench.torch = torch


class Postop64CPU(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(
            os.environ,
            {
                bench.FLAG: "0",
                "SGLANG_GLM53_PREFILL_POOLED_POSTOP": "auto",
                "SGLANG_GLM53_PREFILL_POOLED_SCORE": "0",
            },
        )
        self.environment.start()
        os.environ.pop(bench.MIN_KEYS, None)
        self.addCleanup(self.environment.stop)
        self.role = "prefill"
        self.scope = dict(
            torch=torch,
            os=os,
            triton=SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b),
            get_server_args=lambda: SimpleNamespace(disaggregation_mode=self.role),
        )
        bench.extract_defs(
            DIR / "pooled_score_postop_npu.py", ["supported", "_postop32_group", "relu_gate_"], self.scope
        )
        bench.extract_defs(
            DIR / "kpool_indexer.py", ["_prefill_pooled_postop_min_keys"], self.scope
        )
        self.device = SimpleNamespace(type="npu", index=0)
        self.formats = patch.dict(
            sys.modules, {"torch_npu": SimpleNamespace(get_npu_format=lambda t: t.fmt)}
        )
        self.formats.start()
        self.addCleanup(self.formats.stop)

    def fake(self, shape, dtype, *, fmt=2, contiguous=True):
        return SimpleNamespace(
            shape=shape,
            ndim=len(shape),
            dtype=dtype,
            device=self.device,
            fmt=fmt,
            is_contiguous=lambda: contiguous,
        )

    def test_default32_and_explicit64_guard(self):
        for flag in ("0", "1"):
            os.environ[bench.FLAG] = flag
            for heads in (32, 64, 48):
                scores = self.fake((8, heads, 8192), torch.bfloat16)
                weights = self.fake((8, heads), torch.float32)
                self.assertEqual(
                    self.scope["supported"](scores, weights),
                    heads == 32 or (heads == 64 and flag == "1"),
                )
        os.environ[bench.FLAG] = (
            "auto"  # New flag deliberately accepts only explicit 1.
        )
        self.assertFalse(
            self.scope["supported"](
                self.fake((8, 64, 8), torch.bfloat16), self.fake((8, 64), torch.float32)
            )
        )

    def test_nd_dtype_rows_and_layout_guards(self):
        os.environ[bench.FLAG] = "1"
        supported = self.scope["supported"]
        good = self.fake((8, 64, 8), torch.bfloat16)
        weights = self.fake((8, 64), torch.float32)
        for bad in (
            self.fake((8, 64, 8), torch.bfloat16, fmt=29),
            self.fake((8, 64, 8), torch.bfloat16, contiguous=False),
            self.fake((8, 64, 8), torch.float32),
            self.fake((33, 64, 8), torch.bfloat16),
        ):
            self.assertFalse(supported(bad, weights))
        self.assertFalse(supported(good, self.fake((8, 64), torch.bfloat16)))
        self.assertFalse(supported(good, self.fake((8, 64), torch.float32, fmt=29)))
        self.assertFalse(supported(good, self.fake((8, 32), torch.float32)))

    def test_policy_default_and_64_thresholds(self):
        policy = self.scope["_prefill_pooled_postop_min_keys"]
        for flag in ("0", "1"):
            os.environ[bench.FLAG] = flag
            self.assertEqual(policy(self.device, 32), 32768)
            self.assertEqual(policy(self.device, 8), 98304)
            self.assertIsNone(policy(self.device, 16, 64))
            self.assertEqual(
                policy(self.device, 32, 64), 16384 if flag == "1" else 32768
            )
            self.assertEqual(
                policy(self.device, 8, 64), 49152 if flag == "1" else 98304
            )
        self.assertIsNone(policy(SimpleNamespace(type="cpu"), 32, 64))
        for role in ("decode", "null"):
            self.role = role
            self.assertIsNone(policy(self.device, 32, 64))
        self.role = "prefill"
        os.environ["SGLANG_GLM53_PREFILL_POOLED_POSTOP"] = "0"
        self.assertIsNone(policy(self.device, 32, 64))
        os.environ["SGLANG_GLM53_PREFILL_POOLED_POSTOP"] = "always"
        with self.assertRaises(ValueError):
            policy(self.device, 32, 64)

    def test_cpu_fallback_applies_signed_gate_once(self):
        scores = torch.arange(-192, 192).bfloat16().reshape(2, 64, 3)
        weights = torch.linspace(-2.25, 3.25, 128).reshape(2, 64)
        expected = scores.clone().relu_().mul_(weights.unsqueeze(-1))
        for flag in ("0", "1"):
            os.environ[bench.FLAG] = flag
            actual = scores.clone()
            self.assertFalse(self.scope["relu_gate_"](actual, weights))
            self.assertTrue(
                torch.equal(actual.view(torch.int16), expected.view(torch.int16))
            )

    def test_override_parse_bounds_and_default(self):
        os.environ[bench.FLAG] = "1"
        policy = self.scope["_prefill_pooled_postop_min_keys"]
        for raw in ("1", "8192", "16384", "32768", "49152", " 8192 "):
            os.environ[bench.MIN_KEYS] = raw
            for tile in (8, 32):
                self.assertEqual(policy(self.device, tile, 64), int(raw))
            self.assertIsNone(policy(self.device, 16, 64))
        for raw in (
            "",
            "0",
            "-1",
            "49153",
            "999999999999999999999",
            "1.5",
            "nan",
            "abc",
        ):
            os.environ[bench.MIN_KEYS] = raw
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                policy(self.device, 8, 64)
        os.environ.pop(bench.MIN_KEYS)
        self.assertEqual(policy(self.device, 8, 64), 49152)
        self.assertEqual(policy(self.device, 32, 64), 16384)

    def test_override_is_ignored_outside_opted_in_p64(self):
        os.environ[bench.MIN_KEYS] = "invalid-but-out-of-scope"
        policy = self.scope["_prefill_pooled_postop_min_keys"]
        for flag in ("0", "1"):
            os.environ[bench.FLAG] = flag
            self.assertEqual(policy(self.device, 8, 32), 98304)
            self.assertEqual(policy(self.device, 32, 32), 32768)
        os.environ[bench.FLAG] = "0"
        self.assertEqual(policy(self.device, 8, 64), 98304)
        os.environ.pop(bench.FLAG)
        self.assertEqual(policy(self.device, 32, 64), 32768)
        os.environ[bench.FLAG] = "1"
        self.assertIsNone(policy(SimpleNamespace(type="cpu"), 8, 64))
        for role in ("decode", "null"):
            self.role = role
            self.assertIsNone(policy(self.device, 8, 64))
        self.role = "prefill"
        os.environ["SGLANG_GLM53_PREFILL_POOLED_POSTOP"] = "0"
        self.assertIsNone(policy(self.device, 8, 64))

    def test_harness_clears_and_restores_inherited_override(self):
        os.environ[bench.MIN_KEYS] = "32768"
        with bench.environment(1, 8):
            self.assertNotIn(bench.MIN_KEYS, os.environ)
        self.assertEqual(os.environ[bench.MIN_KEYS], "32768")
        for flag in (0, 1):
            with bench.environment(flag, 8, 8192):
                self.assertEqual(os.environ[bench.MIN_KEYS], "8192")
                expected = 8192 if flag else 98304
                self.assertEqual(
                    self.scope["_prefill_pooled_postop_min_keys"](self.device, 8, 64),
                    expected,
                )
        self.assertEqual(os.environ[bench.MIN_KEYS], "32768")

    def test_forced_caller_rejects_noop_or_helper_fallback(self):
        self.assertEqual(
            bench.validate_caller_activation([[], [{"active": True}]], 8, 8192, 8192),
            [0, 1],
        )
        for calls in (
            [[], []],
            [[], [{"active": False}]],
            [[], [{"active": True}, {"active": True}]],
        ):
            with self.assertRaises(AssertionError):
                bench.validate_caller_activation(calls, 8, 8192, 8192)
        with self.assertRaises(AssertionError):
            bench.validate_caller_activation([[], []], 8, 8192, 16384)
        self.assertEqual(bench.validate_caller_activation([[], []], 8, 8192), [0, 0])

    def test_grid_limit_has_no_large_allocation(self):
        self.scope["supported"] = lambda *_: True
        scores = Mock(shape=(32, 64, 131072))
        weights = Mock()
        self.assertFalse(self.scope["relu_gate_"](scores, weights, block=2048))
        scores.relu_.assert_called_once_with()
        scores.mul_.assert_called_once_with(weights.unsqueeze.return_value)

    def test_64_launch_uses_flat_qhead_and_existing_kernel(self):
        os.environ[bench.FLAG] = "1"
        launch = Mock()

        class Kernel:
            def __getitem__(_, grid):
                self.assertEqual(grid, (32 * 64, 4))
                return launch

        self.scope["_relu_gate_inplace"] = Kernel()
        scores = self.fake((32, 64, 32768), torch.bfloat16)
        weights = self.fake((32, 64), torch.float32)
        self.assertTrue(self.scope["relu_gate_"](scores, weights))
        launch.assert_called_once_with(
            scores, weights, 32768, 8192, num_warps=4, multibuffer=False
        )

    def test_actual_caller_forwards_head_count_and_preserves_fallback(self):
        observed = []

        class Query:
            device = SimpleNamespace(type="npu")

            def __init__(self, data):
                self.data = data

            def __getitem__(self, key):
                return self.data[key]

        class CPUFactories:
            def __getattr__(self, name):
                return getattr(torch, name)

            def arange(self, *args, **kwargs):
                kwargs.pop("device", None)
                return torch.arange(*args, **kwargs)

        def fallback(scores, weights):
            scores.relu_().mul_(weights.unsqueeze(-1))
            return False

        helper = SimpleNamespace(relu_gate_=fallback)
        calls = []
        fn = bench.scorer(DIR / "kpool_indexer.py", helper, calls=calls)
        fn.__globals__["torch"] = CPUFactories()

        def policy(device, tile, heads=32):
            observed.append(heads)
            return 0

        fn.__globals__["_prefill_pooled_postop_min_keys"] = policy
        q = torch.ones((4, 64, 128), dtype=torch.bfloat16)
        keys = torch.arange(1, 5).bfloat16().reshape(4, 1).expand(4, 128).contiguous()
        weights = torch.full((4, 64), -2.0)
        batch = SimpleNamespace(
            batch_size=1, extend_seq_lens_cpu=[4], seq_lens_cpu=[16]
        )
        # Format reporting is a harness observation only, not NPU emulation.
        with patch.object(bench, "formats", return_value=[2, 2]):
            with bench.environment(1, 8):
                result = fn(
                    SimpleNamespace(index_topk=8, index_kpool=4),
                    Query(q),
                    weights,
                    [keys],
                    batch,
                )
        self.assertEqual(observed, [64])
        self.assertEqual([x["active"] for x in calls], [False])
        self.assertEqual(result.dtype, torch.int32)
        # Negative gate makes all finite values negative; applying it twice
        # would reverse the ordering. Compare an unmodified native scorer.
        reference = bench.scorer(DIR / "kpool_indexer.py", helper)
        reference.__globals__["torch"] = CPUFactories()
        with bench.environment(0, 8):
            expected = reference(
                SimpleNamespace(index_topk=8, index_kpool=4),
                Query(q),
                weights,
                [keys],
                batch,
            )
        self.assertTrue(torch.equal(result, expected))

    def test_byte_oracle_rejects_zero_sign_and_nan_payload(self):
        a = torch.tensor([[0, 32704, 16256]], dtype=torch.int16).view(torch.bfloat16)
        for bits in (
            [[-32768, 32704, 16256]],
            [[0, 32705, 16256]],
            [[0, 32704, 16257]],
        ):
            b = torch.tensor(bits, dtype=torch.int16).view(torch.bfloat16)
            with self.assertRaises(AssertionError):
                bench.strict(a, b, "injected mismatch")
        self.assertTrue(bench.strict(a, a, "identical NaN bits")["strict_equal"])


if __name__ == "__main__":
    unittest.main()
