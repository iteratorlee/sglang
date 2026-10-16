"""CPU guard and reference checks; these do not establish NPU numerics.

CUDA_VISIBLE_DEVICES='' TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -B \
    test/manual/test_glm53_large_mhc_pre_cpu.py -v
"""

import ast
import logging
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Tuple
import unittest
from unittest import mock

import torch


SOURCE = (
    Path(__file__).resolve().parents[2] / "python/sglang/kernels/ops/layernorm/mhc.py"
)
FLAG = "SGLANG_GLM53_PREFILL_MHC_PRE"


def load_functions():
    names = {"_mhc_pre_torch", "_use_large_prefill_hc_pre", "hc_pre"}
    tree = ast.parse(SOURCE.read_text())
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = dict(
        torch=torch,
        os=os,
        Tuple=Tuple,
        logger=logging.getLogger(__name__),
        _large_prefill_hc_pre_logged=False,
    )
    exec(
        compile(ast.Module(body=body, type_ignores=[]), str(SOURCE), "exec"), namespace
    )
    return namespace


class TensorMetadata:
    def __init__(
        self, shape, dtype=torch.float32, device="npu:0", contiguous=True, fmt=2
    ):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = SimpleNamespace(type=device.split(":")[0], name=device)
        self.ndim = len(shape)
        self.contiguous = contiguous
        self.fmt = fmt

    def is_contiguous(self):
        return self.contiguous

    def view(self, *shape):
        return TensorMetadata(
            shape, self.dtype, self.device.name, self.contiguous, self.fmt
        )


class LargeMhcPreTests(unittest.TestCase):
    def setUp(self):
        self.ns = load_functions()
        self.role = "prefill"
        runtime = ModuleType("sglang.srt.runtime_context")
        runtime.get_server_args = lambda: SimpleNamespace(disaggregation_mode=self.role)
        npu = ModuleType("torch_npu")
        npu.get_npu_format = lambda x: x.fmt
        self.module_patch = mock.patch.dict(
            sys.modules, {runtime.__name__: runtime, npu.__name__: npu}
        )
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        self.env_patch = mock.patch.dict(os.environ, {FLAG: "1"})
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)

    def inputs(self, t=8192):
        return [
            TensorMetadata((t, 16384), torch.bfloat16),
            TensorMetadata((24, 16384)),
            TensorMetadata((3,)),
            TensorMetadata((24,)),
            4,
            None,
        ]

    def eligible(self, values=None):
        return self.ns["_use_large_prefill_hc_pre"](*(values or self.inputs()))

    def test_default_off_does_not_inspect_tensors(self):
        del os.environ[FLAG]
        self.assertFalse(self.ns["_use_large_prefill_hc_pre"](*([None] * 6)))

    def test_only_large_bounded_prefill(self):
        for t in [4097, 8192, 16384]:
            self.assertTrue(self.eligible(self.inputs(t)))
        for t in [0, 1, 4096, 16385, 32768]:
            self.assertFalse(self.eligible(self.inputs(t)))
        for role in ["decode", "null", ""]:
            self.role = role
            self.assertFalse(self.eligible())

    def test_rejects_layout_dtype_device_and_parameter_shapes(self):
        cases = [
            (0, "dtype", torch.float32),
            (0, "shape", (8192, 8192)),
            (0, "ndim", 3),
            (1, "shape", (24, 8192)),
            (2, "shape", (1, 3)),
            (3, "shape", (1, 24)),
        ]
        for i, field, value in cases:
            values = self.inputs()
            setattr(values[i], field, value)
            self.assertFalse(self.eligible(values), (i, field))
        for i in range(4):
            for field, value in [
                ("contiguous", False),
                ("fmt", 29),
                ("device", SimpleNamespace(type="npu", name="npu:1")),
            ]:
                values = self.inputs()
                setattr(values[i], field, value)
                self.assertFalse(self.eligible(values), (i, field))
        for i in [1, 2, 3]:
            values = self.inputs()
            values[i].dtype = torch.bfloat16
            self.assertFalse(self.eligible(values))
        values = self.inputs()
        values[5] = TensorMetadata((16384,))
        self.assertFalse(self.eligible(values))
        values = self.inputs()
        values[4] = 2
        self.assertFalse(self.eligible(values))

    def test_cpu_tensor_never_selects_candidate(self):
        values = self.inputs()
        values[0] = torch.empty((8192, 16384), dtype=torch.bfloat16, device="meta")
        self.assertFalse(self.eligible(values))

    def test_small_kernel_dispatch_is_unchanged_even_when_flag_on(self):
        helper = ModuleType("sglang.kernels.ops.layernorm.mhc_npu")
        helper.hc_pre = mock.Mock(return_value="old-small-kernel")
        with mock.patch.dict(sys.modules, {helper.__name__: helper}):
            for self.role in ["prefill", "decode"]:
                vals = self.inputs(4096)
                got = self.ns["hc_pre"](*vals[:5], 1e-6, 1e-6, 20)
                self.assertEqual(got, "old-small-kernel")
        self.assertEqual(helper.hc_pre.call_count, 2)

    def test_large_dispatch_keeps_unfused_norm_and_logs_after_success(self):
        kernel = ModuleType("sglang.kernels.ops.layernorm.mhc_pre_mix_npu")
        kernel.hc_pre_mix_large_prefill = object()
        n = 8192
        result = (
            torch.empty(n, 4, 1),
            torch.empty(n, 4, 4),
            torch.empty(n, 4096, device="meta"),
        )
        native = mock.Mock(return_value=result)
        self.ns["_mhc_pre_torch"] = native
        self.ns["logger"] = mock.Mock()
        with mock.patch.dict(sys.modules, {kernel.__name__: kernel}):
            out = self.ns["hc_pre"](
                *self.inputs()[:5],
                1e-6,
                1e-6,
                20,
                out_norm_weight=object(),
                out_norm_eps=1e-6
            )
            self.assertFalse(out[3])
            self.assertIs(
                native.call_args.kwargs["pre_mix"], kernel.hc_pre_mix_large_prefill
            )
            self.assertEqual(out[1].shape, (n, 16))
            self.ns["hc_pre"](*self.inputs()[:5], 1e-6, 1e-6, 20)
            self.assertEqual(self.ns["logger"].info.call_count, 1)
        self.ns["_large_prefill_hc_pre_logged"] = False
        native.side_effect = RuntimeError("kernel failed")
        with mock.patch.dict(sys.modules, {kernel.__name__: kernel}):
            with self.assertRaisesRegex(RuntimeError, "kernel failed"):
                self.ns["hc_pre"](*self.inputs()[:5], 1e-6, 1e-6, 20)
        self.assertFalse(self.ns["_large_prefill_hc_pre_logged"])

    def test_native_coefficient_and_bf16_boundary_unchanged_by_hook(self):
        torch.manual_seed(44)
        fn = torch.randn(24, 256, dtype=torch.float32) / 16
        scale = torch.tensor([0.5, -0.25, 1.25])
        base = torch.randn(24)

        def mix(residual, pre):
            return (pre.unsqueeze(-1) * residual.float()).sum(1).to(residual.dtype)

        for t in [1, 7, 17]:
            for magnitude in [0.0, 2.0**-24, 1.0, 128.0]:
                residual = (torch.randn(t, 4, 64) * magnitude).bfloat16()
                before = residual.clone()
                args = (residual, fn, scale, base, 1e-6, 1e-6, 1e-6, 2.0, 20)
                old = self.ns["_mhc_pre_torch"](*args)
                new = self.ns["_mhc_pre_torch"](*args, pre_mix=mix)
                for a, b in zip(old, new):
                    self.assertTrue(
                        torch.equal(
                            a.contiguous().view(torch.uint8),
                            b.contiguous().view(torch.uint8),
                        )
                    )
                self.assertTrue(
                    torch.equal(before.view(torch.int16), residual.view(torch.int16))
                )

    def test_reduction_reassociation_can_change_bf16_output(self):
        # Positive valid coefficients do not make arbitrary reduction trees safe.
        terms = torch.tensor([1.0, 2.0**-26, -1.0, 2.0**-26]) / 4
        sequential = (((terms[0] + terms[1]) + terms[2]) + terms[3]).bfloat16()
        paired = ((terms[0] + terms[2]) + (terms[1] + terms[3])).bfloat16()
        self.assertNotEqual(
            sequential.view(torch.int16).item(), paired.view(torch.int16).item()
        )


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
