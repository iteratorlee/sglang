"""CPU-only dispatch checks for the scoped D loop minimum; no NPU launch."""

import os
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from test_glm53_gated_rms_loop_cpu import Launch, h, loop

import torch

MIN_ROWS = "SGLANG_GLM53_D_GATED_RMS_LOOP_MIN_ROWS"
GROUPED = "SGLANG_GLM53_D_GATED_RMS_ROWS"


class LoopMinRowsCPU(unittest.TestCase):
    def setUp(self):
        self.old, _ = h.load_kernels(cpu=True)
        self.calls = []
        self.old._glm_rms_norm_gated_row_kernel = Launch(self.calls, "small")
        self.old._glm_rms_norm_gated_large_row_kernel = Launch(self.calls, "large")
        self.old.torch = NS(empty_like=lambda x: x, bfloat16=torch.bfloat16)
        self.old.rms_norm_gated_rows = Mock(
            side_effect=lambda x, g, w, out, eps, group: out
        )

    @staticmethod
    def tensor(shape, dtype=torch.bfloat16):
        size = 1
        for dim in shape:
            size *= dim
        return NS(
            shape=shape,
            ndim=len(shape),
            dtype=dtype,
            device=NS(type="npu"),
            numel=lambda: size,
            is_contiguous=lambda: True,
        )

    def dispatch(
        self, rows, minimum=None, *, mode="decode", d_loop="8", p_loop="0", grouped="0"
    ):
        # Only tensor metadata and launch recorders: no accelerator allocation.
        x = self.tensor((1, 16384, 8, 128) if rows == 131072 else (rows, 128))
        w = self.tensor((128,), torch.float32)
        self.old.get_server_args = lambda: NS(disaggregation_mode=mode)
        self.calls.clear()
        self.old.rms_norm_gated_rows.reset_mock()
        with patch.dict(
            os.environ,
            {loop.LOOP_FLAG: d_loop, loop.P_LOOP_FLAG: p_loop, GROUPED: grouped},
        ):
            os.environ.pop(MIN_ROWS, None)
            if minimum is not None:
                os.environ[MIN_ROWS] = minimum
            with patch.dict(sys.modules, {"torch_npu": NS(get_npu_format=lambda t: 2)}):
                self.assertIs(self.old.glm_rms_norm_gated_npu(x, x, w, 1e-6), x)

    def check_launch(self, rows, *, loop_size=None):
        name, grid, kwargs = self.calls[0]
        self.assertEqual(len(self.calls), 1)
        expected_grid = (
            (rows + loop_size - 1) // loop_size if loop_size else min(rows, 32768)
        )
        self.assertEqual(name, "large" if loop_size or rows > 65535 else "small")
        self.assertEqual(grid, (expected_grid,))
        for key, value in (
            ("feature_dim", 128),
            ("block_dim", 128),
            ("num_warps", 1),
            ("num_stages", 1),
        ):
            self.assertEqual(kwargs[key], value)
        if name == "large":
            self.assertEqual(kwargs["num_rows"], rows)
            self.assertEqual(kwargs["grid_rows"], expected_grid)
        self.old.rms_norm_gated_rows.assert_not_called()

    def test_default_and_explicit_1024_preserve_original_scope(self):
        for minimum in (None, "1024"):
            for rows in (1023, 1024, 1535, 1536, 1664, 1792, 2048, 2049):
                with self.subTest(minimum=minimum, rows=rows):
                    self.dispatch(rows, minimum)
                    self.check_launch(
                        rows, loop_size=8 if 1024 <= rows <= 2048 else None
                    )

    def test_min_1536_boundary_dispatch_and_d100_loop8(self):
        for group in (4, 8):
            for rows in (1024, 1535, 1536, 1664, 1792, 2048, 2049, 4096, 65536):
                with self.subTest(group=group, rows=rows):
                    self.dispatch(rows, "1536", d_loop=str(group))
                    self.check_launch(
                        rows, loop_size=group if 1536 <= rows <= 2048 else None
                    )

    def test_interior_and_upper_valid_minima(self):
        for minimum in (1025, 1535, 2048):
            for rows in (minimum - 1, minimum, 2048, 2049):
                with self.subTest(minimum=minimum, rows=rows):
                    self.dispatch(rows, str(minimum))
                    self.check_launch(
                        rows, loop_size=8 if minimum <= rows <= 2048 else None
                    )

    def test_invalid_values_raise_before_any_kernel_launch(self):
        for value in ("", "0", "1023", "2049", "-1", "1536.0", "abc", "nan"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, MIN_ROWS + r".*\[1024, 2048\]"):
                    self.dispatch(1664, value)
                self.assertEqual(self.calls, [])
                self.old.rms_norm_gated_rows.assert_not_called()

    def test_prefill_131072_is_independent_of_d_minimum(self):
        for minimum in (None, "1024", "1536", "2048", "invalid-D-only-setting"):
            for p_loop in ("0", "4", "8"):
                with self.subTest(minimum=minimum, p_loop=p_loop):
                    self.dispatch(131072, minimum, mode="prefill", p_loop=p_loop)
                    self.check_launch(
                        131072, loop_size=int(p_loop) if p_loop != "0" else None
                    )

    def test_disabled_loop_and_existing_grouped_opt_in_are_unchanged(self):
        for grouped in ("0", "8"):
            self.dispatch(1024, "invalid-unused-setting", d_loop="0", grouped=grouped)
            if grouped == "0":
                self.check_launch(1024)
            else:
                self.assertEqual(self.calls, [])
                self.old.rms_norm_gated_rows.assert_called_once()
                self.assertEqual(self.old.rms_norm_gated_rows.call_args.args[-1], 8)


if __name__ == "__main__":
    unittest.main()
