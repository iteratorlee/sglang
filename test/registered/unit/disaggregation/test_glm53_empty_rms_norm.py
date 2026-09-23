"""Empty DP domains must not launch a zero-grid NPU gated RMSNorm kernel."""

import ast
import copy
import unittest
from pathlib import Path
from unittest.mock import Mock

import torch


def load_rms_norm(kernel):
    source = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/hardware_backend/npu/attention/glm53/rms_norm_gated_npu.py"
    )
    tree = ast.parse(source.read_text())
    function = copy.deepcopy(
        next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "glm_rms_norm_gated_npu"
        )
    )
    function.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            function,
        ],
        type_ignores=[],
    )
    namespace = {"torch": torch, "_glm_rms_norm_gated_row_kernel": kernel}
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace["glm_rms_norm_gated_npu"]


class EmptyRMSNormTests(unittest.TestCase):
    def test_zero_tokens_skip_kernel(self):
        kernel = Mock()
        function = load_rms_norm(kernel)
        x = torch.empty((0, 4, 64))
        result = function(x, torch.empty_like(x), torch.ones(64), 1e-6)
        self.assertEqual(tuple(result.shape), tuple(x.shape))
        self.assertEqual(result.dtype, x.dtype)
        kernel.assert_not_called()

    def test_empty_input_still_validates_dimensions(self):
        function = load_rms_norm(Mock())
        with self.assertRaisesRegex(ValueError, "head dimension"):
            function(
                torch.empty((0, 4, 64)), torch.empty((0, 4, 64)), torch.ones(32), 1e-6
            )


if __name__ == "__main__":
    unittest.main()
