"""Small CPU numerical checks and D/P large-loop candidate dispatch."""

import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

os.environ["TRITON_INTERPRET"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import torch

ROOT = Path(__file__).resolve().parents[5]
MANUAL = ROOT / "test/manual/ascend"


def module(name):
    spec = importlib.util.spec_from_file_location(name, MANUAL / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


loop = module("test_glm53_gated_rms_loop_npu")
h = loop.adapt_reference(module("test_glm53_gated_rms_rows_npu"))
v2 = module("test_glm53_gated_rms_contract_npu")


class Launch:
    def __init__(self, calls, name):
        self.calls, self.name = calls, name

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((self.name, grid, kwargs))
            return "compiled-handle"

        return launch


class LoopCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.old, cls.new = h.load_kernels(cpu=True)

    def test_small_shapes_original_vs_loop_exact(self):
        for rows in (0, 1, 7, 8, 9, 31, 32, 33):
            for wdtype in (torch.bfloat16, torch.float32):
                for odtype in (torch.bfloat16, torch.float32):
                    data = h.inputs(rows, wdtype, "wide")
                    expected = torch.empty((rows, 128), dtype=odtype)
                    h.launch(self.old, self.new, 1, *data, expected)
                    for group in (4, 8):
                        backing = torch.full((rows + 8, 128), -19.0, dtype=odtype)
                        h.launch(self.old, self.new, group, *data, backing[:rows])
                        h.exact(expected, backing[:rows], "1D original vs loop")
                        h.exact(
                            backing[rows:],
                            torch.full_like(backing[rows:], -19),
                            "loop bounds",
                        )

    def test_midpoint_and_failed_v2_seed_on_small_cpu_domain(self):
        for data in (
            h.inputs(32, torch.float32, "wide", 101),
            *(
                v2.input_factory(h.inputs, midpoint_direction=d)(
                    32, torch.float32, "random"
                )
                for d in (-1, 0, 1)
            ),
        ):
            original = [t.clone() for t in data]
            for odtype in (torch.bfloat16, torch.float32):
                expected = torch.empty((32, 128), dtype=odtype)
                h.launch(self.old, self.new, 1, *data, expected)
                for group in (4, 8):
                    out = torch.empty_like(expected)
                    h.launch(self.old, self.new, group, *data, out)
                    h.exact(expected, out, "midpoint loop")
            for before, after in zip(original, data):
                h.exact(before, after, "read-only input")

    def test_adapter_returns_actual_large_kernel_and_grid(self):
        calls = []
        kernel = loop.LoopKernel(Launch(calls, "large"))
        for rows in (1024, 1663, 1664, 1665, 1792, 2048, 131072):
            for group in (4, 8):
                grid = (rows + group - 1) // group
                handle = kernel[(grid,)](
                    None,
                    None,
                    None,
                    None,
                    1e-6,
                    num_rows=rows,
                    rows_per_program=group,
                    num_warps=1,
                    num_stages=1,
                )
                self.assertEqual(handle, "compiled-handle")
                self.assertEqual(calls[-1][1], (grid,))
                self.assertEqual(calls[-1][2]["grid_rows"], grid)
                self.assertEqual(calls[-1][2]["block_dim"], 128)
                # Every row belongs to exactly one original grid-stride loop.
                covered = [row for pid in range(grid) for row in range(pid, rows, grid)]
                self.assertEqual(sorted(covered), list(range(rows)))
                self.assertLessEqual((rows + grid - 1) // grid, group)

    def test_prefill_shape_nd_guard_and_baseline_grid(self):
        old, new = h.load_kernels(cpu=True)
        calls = []
        old._glm_rms_norm_gated_large_row_kernel = Launch(calls, "large")
        old.torch = NS(empty_like=lambda x: x, bfloat16=torch.bfloat16)
        old.get_server_args = lambda: NS(disaggregation_mode="prefill")

        def fake(shape, dtype=torch.bfloat16, fmt=2):
            size = 1
            for n in shape:
                size *= n
            return NS(
                shape=shape,
                ndim=len(shape),
                dtype=dtype,
                device=NS(type="npu"),
                numel=lambda: size,
                is_contiguous=lambda: True,
                fmt=fmt,
            )

        x, g, w = (
            fake((1, 16384, 8, 128)),
            fake((16384, 8, 128)),
            fake((128,), torch.float32),
        )
        with patch.dict(sys.modules, {"torch_npu": NS(get_npu_format=lambda t: t.fmt)}):
            self.assertTrue(new.supported_prefill_loop(x, g, w, x))
            for target in (x, g, w):
                target.fmt = 29
                self.assertFalse(new.supported_prefill_loop(x, g, w, x))
                target.fmt = 2
            for flag, grid in (("0", 32768), ("4", 32768), ("8", 16384)):
                calls.clear()
                with patch.dict(
                    os.environ, {loop.P_LOOP_FLAG: flag, loop.LOOP_FLAG: "0"}
                ):
                    old.glm_rms_norm_gated_npu(x, g, w, 1e-6)
                self.assertEqual(calls[0][1], (grid,))
                self.assertEqual(calls[0][2]["num_rows"], 131072)
            # Wrong head layout must retain the original large grid.
            wrong = fake((1, 4096, 32, 128))
            calls.clear()
            with patch.dict(os.environ, {loop.P_LOOP_FLAG: "8", loop.LOOP_FLAG: "0"}):
                old.glm_rms_norm_gated_npu(wrong, wrong, w, 1e-6)
            self.assertEqual(calls[0][1], (32768,))
        # Strict unit's variant 1 must use the true baseline, not grid=131072.
        calls.clear()
        h.launch(old, new, 1, x, g, w, x)
        self.assertEqual(calls[0][1], (32768,))

    def test_wrapper_defaults_role_dtype_bounds_and_precedence(self):
        old, _ = h.load_kernels(cpu=True)
        calls = []
        old._glm_rms_norm_gated_row_kernel = Launch(calls, "small")
        old._glm_rms_norm_gated_large_row_kernel = Launch(calls, "large")
        old.rms_norm_gated_rows = lambda *args: calls.append(("2d", None, {}))
        old.torch = NS(empty_like=lambda x: x, bfloat16=torch.bfloat16)
        old.supported = lambda *args: True
        for flag, mode, rows, dtype, other, want in (
            ("0", "decode", 1664, torch.bfloat16, "0", "small"),
            ("1", "decode", 1664, torch.bfloat16, "0", "small"),
            ("4", "decode", 1664, torch.bfloat16, "0", "large"),
            ("8", "decode", 1663, torch.bfloat16, "0", "large"),
            ("8", "prefill", 1664, torch.bfloat16, "0", "small"),
            ("8", "decode", 1023, torch.bfloat16, "0", "small"),
            ("8", "decode", 2049, torch.bfloat16, "0", "small"),
            ("8", "decode", 1664, torch.float32, "0", "small"),
            ("8", "decode", 1664, torch.bfloat16, "4", "large"),
            ("0", "decode", 1664, torch.bfloat16, "4", "2d"),
            ("8", "prefill", 65536, torch.bfloat16, "0", "large"),
        ):
            x = NS(
                shape=(rows, 128),
                dtype=dtype,
                device=NS(type="npu"),
                numel=lambda: rows * 128,
            )
            old.get_server_args = lambda: NS(disaggregation_mode=mode)
            calls.clear()
            with patch.dict(
                os.environ,
                {loop.LOOP_FLAG: flag, "SGLANG_GLM53_D_GATED_RMS_ROWS": other},
            ):
                old.glm_rms_norm_gated_npu(x, x, NS(numel=lambda: 128), 1e-6)
            self.assertEqual(calls[0][0], want)
            if want == "large" and rows <= 2048:
                self.assertEqual(calls[0][1], ((rows + int(flag) - 1) // int(flag),))

    def test_both_original_kernel_bodies_and_arithmetic_unchanged(self):
        rel = "python/sglang/srt/hardware_backend/npu/attention/glm53/rms_norm_gated_npu.py"
        before = ast.parse(
            subprocess.check_output(
                ["git", "show", f"e3179b6:{rel}"], cwd=ROOT, text=True
            )
        )
        after = ast.parse((ROOT / rel).read_text())

        def kernels(tree):
            return [
                ast.dump(n)
                for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name.startswith("_glm_")
            ]

        self.assertEqual(kernels(before), kernels(after))

        def math(tree, name):
            fn = next(
                n
                for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == name
            )
            return {
                n.targets[0].id: ast.dump(n.value)
                for n in ast.walk(fn)
                if isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id in ("variance", "normalized", "result")
            }

        self.assertEqual(
            math(after, "_glm_rms_norm_gated_row_kernel"),
            math(after, "_glm_rms_norm_gated_large_row_kernel"),
        )


if __name__ == "__main__":
    unittest.main()
