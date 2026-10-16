"""CPU-only guards, callback boundary, and validation of the remote harness.

CUDA_VISIBLE_DEVICES='' TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -B \
    test/manual/test_glm53_mhc_sinkhorn_cpu.py -v
No accelerator module or kernel is imported/executed by these tests.
"""

import ast
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import bench_glm53_mhc_sinkhorn_npu as harness
import test_glm53_large_mhc_pre_cpu as support
import torch

FLAG = "SGLANG_GLM53_PREFILL_MHC_SINKHORN"
HELPER = support.SOURCE.with_name("mhc_sinkhorn_npu.py")


class SinkhornDispatchTests(support.LargeMhcPreTests):
    # Also rerun the baseline large-P guard/pre-mix tests against this candidate.
    def setUp(self):
        super().setUp()
        patch = mock.patch.dict(os.environ)
        patch.start()
        self.addCleanup(patch.stop)
        os.environ.pop(FLAG, None)
        self.mix_module = ModuleType("sglang.kernels.ops.layernorm.mhc_pre_mix_npu")
        self.mix_module.hc_pre_mix_large_prefill = object()
        self.sink_module = ModuleType("sglang.kernels.ops.layernorm.mhc_sinkhorn_npu")
        self.sink_module.hc_sinkhorn_large_prefill = object()
        modules = mock.patch.dict(
            sys.modules,
            {
                self.mix_module.__name__: self.mix_module,
                self.sink_module.__name__: self.sink_module,
            },
        )
        modules.start()
        self.addCleanup(modules.stop)

    def mock_outputs(self, t):
        return (
            torch.empty(t, 4, 1, device="meta"),
            torch.empty(t, 4, 4, device="meta"),
            torch.empty(t, 4096, device="meta"),
        )

    def test_new_flag_default_off_and_exact_opt_in(self):
        for flag in (None, "0", "true", "2", "1"):
            if flag is None:
                os.environ.pop(FLAG, None)
            else:
                os.environ[FLAG] = flag
            native = self.ns["_mhc_pre_torch"] = mock.Mock(
                return_value=self.mock_outputs(8192)
            )
            out = self.ns["hc_pre"](*self.inputs()[:5], 1e-6, 1e-6, 20)
            self.assertIs(
                native.call_args.kwargs["sinkhorn"],
                self.sink_module.hc_sinkhorn_large_prefill if flag == "1" else None,
            )
            self.assertIs(
                native.call_args.kwargs["pre_mix"],
                self.mix_module.hc_pre_mix_large_prefill,
            )
            self.assertIs(out[3], False)

    def test_new_flag_cannot_bypass_existing_guard(self):
        os.environ[FLAG] = "1"
        for case in (
            "pre_flag_off",
            "decode",
            "null",
            "too_large",
            "cpu",
            "float32",
            "format",
        ):
            with self.subTest(case=case):
                self.role = "prefill"
                os.environ[support.FLAG] = "1"
                values = self.inputs(16385 if case == "too_large" else 8192)
                if case == "pre_flag_off":
                    os.environ[support.FLAG] = "0"
                elif case in ("decode", "null"):
                    self.role = case
                elif case == "cpu":
                    values[0].device = SimpleNamespace(type="cpu", name="cpu")
                elif case == "float32":
                    values[0].dtype = torch.float32
                elif case == "format":
                    values[0].fmt = 29
                t = values[0].shape[0]
                values[0].numel = lambda: t * 16384
                self.ns["_mhc_pre_dispatch"] = mock.Mock(
                    return_value=(*self.mock_outputs(t), False)
                )
                self.ns["_mhc_pre_torch"] = mock.Mock(
                    side_effect=AssertionError("unexpected fused path")
                )
                # A forbidden import fails immediately, even if never called.
                with mock.patch.dict(sys.modules, {self.sink_module.__name__: None}):
                    self.ns["hc_pre"](*values[:5], 1e-6, 1e-6, 20)
                self.ns["_mhc_pre_dispatch"].assert_called_once()

    def test_new_flag_does_not_change_small_or_empty_dispatch(self):
        os.environ[FLAG] = "1"
        helper = ModuleType("sglang.kernels.ops.layernorm.mhc_npu")
        helper.hc_pre = mock.Mock(return_value="existing-small-kernel")
        with mock.patch.dict(
            sys.modules, {helper.__name__: helper, self.sink_module.__name__: None}
        ):
            for self.role in ("decode", "prefill"):
                for t in (0, 1, 4096):
                    self.assertEqual(
                        self.ns["hc_pre"](*self.inputs(t)[:5], 1e-6, 1e-6, 20),
                        "existing-small-kernel",
                    )
        self.assertEqual(helper.hc_pre.call_count, 6)

    def test_callback_receives_native_softmax_eps_and_only_replaces_comb(self):
        torch.manual_seed(67)
        fn, scale, base = torch.randn(24, 256), torch.randn(3), torch.randn(24)
        native = self.ns["_mhc_pre_torch"]
        for t in (0, 1, 7, 33):
            residual = torch.randn(t, 4, 64).bfloat16()
            args = (residual, fn, scale, base, 1e-6, 2e-6, 3e-6, 2.0, 20)
            original = native(*args)
            seen = []

            def callback(comb, iters, eps):
                self.assertEqual((iters, eps), (20, 3e-6))
                self.assertEqual(comb.dtype, torch.float32)
                self.assertEqual(comb.shape, (t, 4, 4))
                self.assertTrue(comb.is_contiguous())
                flat = residual.view(t, 256).float()
                inv = torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-6)
                projected = torch.nn.functional.linear(flat, fn) * inv
                logits = projected[:, 8:].view(t, 4, 4) * scale[2] + base[8:].view(4, 4)
                self.assertTrue(torch.equal(comb, logits.softmax(-1) + eps))
                seen.append(comb)
                return harness.native_sinkhorn(comb, iters, eps)

            actual = native(*args, sinkhorn=callback)
            self.assertEqual(len(seen), 1)
            for a, b in zip(original, actual):
                self.assertTrue(torch.equal(a.view(torch.uint8), b.view(torch.uint8)))
            sentinel = torch.full((t, 4, 4), 0.123)
            changed = native(*args, sinkhorn=lambda *unused: sentinel)
            self.assertIs(changed[1], sentinel)
            for i in (0, 2):
                self.assertTrue(
                    torch.equal(
                        original[i].view(torch.uint8), changed[i].view(torch.uint8)
                    )
                )
            with self.assertRaisesRegex(RuntimeError, "sinkhorn failed"):
                native(
                    *args,
                    sinkhorn=mock.Mock(side_effect=RuntimeError("sinkhorn failed"))
                )


class SinkhornHelperTests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse(HELPER.read_text())
        fn = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "hc_sinkhorn_large_prefill"
        )
        self.launch = mock.MagicMock()
        self.allocate = mock.Mock(return_value=object())
        namespace = {
            "torch": SimpleNamespace(float32=torch.float32, empty_like=self.allocate),
            "torch_npu": SimpleNamespace(get_npu_format=lambda t: t.fmt),
            "triton": SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b),
            "_hc_sinkhorn_4x4": self.launch,
        }
        exec(
            compile(ast.Module(body=[fn], type_ignores=[]), str(HELPER), "exec"),
            namespace,
        )
        self.helper = namespace[fn.name]

    def test_empty_tail_and_launch_contract(self):
        for t in (0, 1, 31, 32, 33, 4097, 8192, 16384):
            self.launch.reset_mock()
            x = support.TensorMetadata((t, 4, 4))
            result = self.helper(x, 20, 1e-6)
            self.assertIs(result, self.allocate.return_value)
            if t == 0:
                self.launch.__getitem__.assert_not_called()
            else:
                self.launch.__getitem__.assert_called_once_with(((t + 31) // 32,))
                call = self.launch.__getitem__.return_value.call_args
                self.assertEqual(call.args, (x, result))
                self.assertEqual(
                    call.kwargs,
                    dict(
                        TOKENS=t,
                        ITERS=20,
                        EPS=1e-6,
                        BT=32,
                        num_warps=1,
                        enable_fp_fusion=False,
                        enable_auto_bind_sub_block=False,
                    ),
                )

    def test_invalid_input_fails_before_allocation_or_launch(self):
        for field, value in (
            ("ndim", 2),
            ("shape", (33, 2, 8)),
            ("shape", (16385, 4, 4)),
            ("dtype", torch.bfloat16),
            ("contiguous", False),
            ("fmt", 29),
            ("device", SimpleNamespace(type="cpu")),
        ):
            x = support.TensorMetadata((33, 4, 4))
            setattr(x, field, value)
            with self.assertRaises(ValueError, msg=(field, value)):
                self.helper(x, 20, 1e-6)
        with self.assertRaises(TypeError):
            self.helper(support.TensorMetadata((33, 4, 4)), 20.0, 1e-6)
        self.allocate.assert_not_called()
        self.launch.__getitem__.assert_not_called()


class HarnessTests(unittest.TestCase):
    def test_pre_ast_matches_frozen_baseline_except_normalization_hook(self):
        baseline = subprocess.check_output(
            ["git", "show", "45add04:python/sglang/kernels/ops/layernorm/mhc.py"],
            cwd=support.SOURCE.parents[5],
            text=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "baseline.py"
            reference.write_text(baseline)
            harness.check_only_sinkhorn_changes(reference, support.SOURCE)

    def test_metrics_do_not_hide_failure_or_nonfinite_values(self):
        reference = torch.tensor([0.0, 0.25, 1.0])
        self.assertTrue(harness.comparison(reference, reference)["allclose"])
        bad = harness.comparison(reference + 1e-3, reference)
        self.assertFalse(bad["allclose"])
        self.assertTrue(bad["maxrel_nonfinite"])
        self.assertFalse(
            harness.comparison(torch.full_like(reference, float("nan")), reference)[
                "allclose"
            ]
        )
        self.assertEqual(
            harness.compare_bits(reference, reference)["difference_ratio"], 0
        )
        signed_zero = harness.compare_bits(torch.tensor([0.0]), torch.tensor([-0.0]))
        self.assertFalse(signed_zero["bitwise_equal"])

    def test_fp64_spec_retains_column_first_epsilon_order(self):
        torch.manual_seed(88)
        initial = torch.randn(33, 4, 4).softmax(-1) + harness.EPS
        spec = harness.native_sinkhorn(initial.double())
        fp32 = harness.native_sinkhorn(initial)
        self.assertTrue(harness.comparison(fp32, spec)["allclose"])
        expected_first = initial / (initial.sum(1, keepdim=True) + harness.EPS)
        self.assertTrue(
            torch.equal(harness.native_sinkhorn(initial, 1), expected_first)
        )
        self.assertTrue(
            torch.equal(harness.native_sinkhorn(initial, 0), expected_first)
        )


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
