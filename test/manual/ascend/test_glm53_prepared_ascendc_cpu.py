"""CPU-only dispatch tests for the actual GLM-5.3 prepared prefill wrapper.

Run: python -B test/manual/ascend/test_glm53_prepared_ascendc_cpu.py -v
Extracting only the production functions avoids importing torch_npu or Triton.
The fake tensors expose metadata; these tests make no kernel accuracy claim.
"""

import ast
import builtins
import copy
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

SOURCE = (
    Path(__file__).resolve().parents[3]
    / "python/sglang/srt/hardware_backend/npu/attention/glm53/kda_prefill_npu.py"
)


class Tensor:
    def __init__(self, shape, dtype, device, stride=None, contiguous=True):
        self.shape = tuple(shape)
        self.ndim = len(shape)
        self.dtype = dtype
        self.device = device
        self._contiguous = contiguous
        self._stride = stride or self._default_stride()

    def _default_stride(self):
        result = [1] * self.ndim
        for i in range(self.ndim - 2, -1, -1):
            result[i] = result[i + 1] * self.shape[i + 1]
        return tuple(result)

    def is_contiguous(self):
        return self._contiguous

    def stride(self):
        return self._stride

    def numel(self):
        result = 1
        for size in self.shape:
            result *= size
        return result


class Launch:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))

        return launch


def production_functions(scope):
    tree = ast.parse(SOURCE.read_text())
    names = {
        "_ascendc_prepared_operator",
        "_bv16_b1_layout_ok",
        "run_prepared_prefill",
    }
    nodes = [
        copy.deepcopy(node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    for node in nodes:
        node.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(SOURCE), "exec"), scope)


class PreparedAscendCDispatchCPU(unittest.TestCase):
    def setUp(self):
        self.device = types.SimpleNamespace(type="npu", index=0)
        self.torch = types.ModuleType("torch")
        self.torch.bfloat16 = "bf16"
        self.torch.float32 = "fp32"
        self.torch.int32 = "int32"
        self.torch.int64 = "int64"
        self.torch.ops = types.SimpleNamespace()
        self.torch.empty = lambda shape, dtype, device: Tensor(shape, dtype, device)
        self.torch.empty_like = lambda tensor: Tensor(
            tensor.shape, tensor.dtype, tensor.device
        )
        self.prepare = Launch()
        self.legacy = Launch()
        self.scope = {
            "os": os,
            "logger": Mock(),
            "_ascendc_logged": False,
            "_prepare_inputs": self.prepare,
            "_prepared_recurrent": self.legacy,
        }
        production_functions(self.scope)
        self.torch_modules = patch.dict(sys.modules, {"torch": self.torch})
        self.torch_modules.start()
        self.addCleanup(self.torch_modules.stop)

    def inputs(self, heads=8, tokens=128, state_transposed=False, metadata="int64"):
        qshape = (1, tokens, heads, 128)
        bf16 = lambda shape: Tensor(shape, self.torch.bfloat16, self.device)
        state_stride = (heads * 16384, 16384, 1, 128) if state_transposed else None
        return dict(
            q=bf16(qshape),
            k=bf16(qshape),
            v=bf16(qshape),
            a=bf16(qshape),
            b=bf16(qshape[:-1]),
            A_log=bf16((heads,)),
            dt_bias=bf16((heads, 128)),
            state=Tensor(
                (2, heads, 128, 128),
                self.torch.float32,
                self.device,
                stride=state_stride,
            ),
            indices=Tensor((1,), metadata, self.device),
            starts=Tensor((2,), self.torch.int32, self.device),
            output=bf16(qshape),
            scale=0.5,
            lower_bound=-8.0,
        )

    def invoke(self, tensors, env=None):
        with patch.dict(os.environ, env or {}, clear=True):
            return self.scope["run_prepared_prefill"](**tensors)

    def test_default_off_never_imports_extension_and_uses_prior_kernel(self):
        tensors = self.inputs()
        real_import = builtins.__import__

        def refuse_extension(name, *args, **kwargs):
            if name == "sgl_glm53_ascend_ops":
                self.fail("default-off path imported the extension")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=refuse_extension):
            result = self.invoke(tensors)
        self.assertIs(result, tensors["output"])
        self.assertEqual(len(self.prepare.calls), 1)
        self.assertEqual(
            [grid for grid, _, _ in self.legacy.calls], [(2, 1, 4), (2, 1, 4)]
        )  # H8 split, BV64

    def test_original_h4_only_binary_is_rejected(self):
        tensors = self.inputs(heads=4)
        old_op = Mock()
        self.torch.ops.sgl_glm53_ascend = types.SimpleNamespace(
            prepared_recurrent=old_op
        )
        with patch.dict(
            sys.modules,
            {"sgl_glm53_ascend_ops": types.ModuleType("sgl_glm53_ascend_ops")},
        ):
            with self.assertRaisesRegex(RuntimeError, "prepared_recurrent_v1"):
                self.invoke(tensors, {"SGLANG_GLM53_KDA_PREPARED_ASCENDC": "1"})
        old_op.assert_not_called()
        self.assertFalse(self.legacy.calls)

    def test_valid_layout_calls_versioned_op_with_original_arguments(self):
        versioned = Mock()
        self.torch.ops.sgl_glm53_ascend = types.SimpleNamespace(
            prepared_recurrent=Mock(), prepared_recurrent_v1=versioned
        )
        with patch.dict(
            sys.modules,
            {"sgl_glm53_ascend_ops": types.ModuleType("sgl_glm53_ascend_ops")},
        ):
            for heads, transposed, metadata, tracking in (
                (4, False, "int32", False),
                (8, True, "int64", True),
            ):
                with self.subTest(
                    heads=heads, transposed=transposed, tracking=tracking
                ):
                    versioned.reset_mock()
                    self.prepare.calls.clear()
                    tensors = self.inputs(
                        heads, state_transposed=transposed, metadata=metadata
                    )
                    if tracking:
                        tensors.update(
                            track_indices=Tensor((1,), metadata, self.device),
                            track_lens=Tensor((1,), self.torch.int64, self.device),
                        )
                    result = self.invoke(
                        tensors, {"SGLANG_GLM53_KDA_PREPARED_ASCENDC": "1"}
                    )
                    self.assertIs(result, tensors["output"])
                    self.assertFalse(self.legacy.calls)
                    versioned.assert_called_once()
                    args = versioned.call_args.args
                    prepared_args = self.prepare.calls[0][1]
                    self.assertIs(args[0], prepared_args[6])  # prepared QN
                    self.assertIs(args[1], prepared_args[7])  # prepared KN
                    self.assertIs(args[3], prepared_args[8])  # prepared decay
                    self.assertIs(args[4], prepared_args[9])  # prepared beta
                    for position, name in (
                        (2, "v"),
                        (5, "output"),
                        (6, "state"),
                        (7, "indices"),
                        (8, "starts"),
                    ):
                        self.assertIs(args[position], tensors[name])
                    self.assertIs(args[9], tensors.get("track_indices"))
                    self.assertIs(args[10], tensors.get("track_lens"))
                    self.assertIs(args[11], tracking)
            self.torch.ops.sgl_glm53_ascend.prepared_recurrent.assert_not_called()

    def test_unsupported_shape_or_stride_keeps_old_kernel(self):
        cases = {
            "short_tokens": self.inputs(tokens=64),
            "wrong_heads": self.inputs(heads=16),
            "state_stride": self.inputs(state_transposed=True),
            "noncontiguous_input": self.inputs(),
            "incomplete_tracker": self.inputs(),
        }
        cases["state_stride"]["state"]._stride = (8 * 16384, 16384, 2, 128)
        cases["noncontiguous_input"]["q"]._contiguous = False
        cases["incomplete_tracker"]["track_indices"] = Tensor(
            (1,), self.torch.int32, self.device
        )
        real_import = builtins.__import__

        def refuse_extension(name, *args, **kwargs):
            if name == "sgl_glm53_ascend_ops":
                self.fail("unsupported layout imported the extension")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=refuse_extension):
            for name, tensors in cases.items():
                with self.subTest(case=name):
                    self.legacy.calls.clear()
                    result = self.invoke(
                        tensors,
                        {
                            "SGLANG_GLM53_KDA_PREPARED_ASCENDC": "1",
                            "SGLANG_GLM53_KDA_PREFILL_BV32": "1",
                        },
                    )
                    self.assertIs(result, tensors["output"])
                    self.assertEqual(len(self.legacy.calls), tensors["q"].shape[2] // 4)
                    self.assertEqual(self.legacy.calls[0][0], (4, 1, 4))


if __name__ == "__main__":
    unittest.main()
