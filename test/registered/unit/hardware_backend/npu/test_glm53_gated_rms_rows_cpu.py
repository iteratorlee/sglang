"""CPU/interpreter and dispatch checks; not NPU numerical/performance evidence."""

import ast
import importlib.util
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

# Triton's language reductions select their interpreter implementation on import.
os.environ["TRITON_INTERPRET"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import torch

ROOT = Path(__file__).resolve().parents[5]
spec = importlib.util.spec_from_file_location(
    "gated_rows_harness", ROOT / "test/manual/ascend/test_glm53_gated_rms_rows_npu.py"
)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


class GatedRowsCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.old, cls.new = h.load_kernels(cpu=True)

    def setUp(self):
        env = patch.dict(os.environ, {"TRITON_INTERPRET": "1"})
        env.start()
        self.addCleanup(env.stop)

    def compare(self, rows, wdtype, pattern, odtype=torch.bfloat16):
        data = h.inputs(rows, wdtype, pattern)
        original = [t.clone() for t in data]
        expected = torch.empty((rows, 128), dtype=odtype)
        h.launch(self.old, self.new, 1, *data, expected)
        for group in (4, 8):
            # Extra output rows are sentinels: masked lanes must not write them.
            backing = torch.full((rows + 8, 128), -19.0, dtype=odtype)
            out = backing[:rows]
            h.launch(self.old, self.new, group, *data, out)
            h.exact(expected, out, f"rows={rows} group={group} pattern={pattern}")
            h.exact(
                backing[rows:], torch.full_like(backing[rows:], -19), "tail store mask"
            )
            for a, b in zip(original, data):
                h.exact(a, b, "read-only inputs")

    def test_short_rows_and_partial_group_mask(self):
        for rows in (0, 1, 3, 4, 7, 8, 9, 31, 32, 33):
            for wdtype in (torch.bfloat16, torch.float32):
                self.compare(rows, wdtype, "random")

    def test_profile_shape_bf16_and_unrounded_fp32_diagnostic(self):
        for wdtype in (torch.bfloat16, torch.float32):
            for odtype in (torch.bfloat16, torch.float32):
                for pattern in ("random", "zero", "signed-zero", "wide"):
                    self.compare(1664, wdtype, pattern, odtype)

    def test_neighbor_graph_shapes_and_group_remainder(self):
        for rows in (1023, 1024, 1663, 1665, 1792, 2048, 2049):
            self.compare(rows, torch.float32, "random")

    def test_helper_supported_types_layouts_and_empty(self):
        x, g, w = h.inputs(7)
        out = torch.empty_like(x)
        h.exact(
            self.new.rms_norm_gated_rows(x, g, w, out, 1e-6, 4), out, "in-place output"
        )
        for args in (
            (x, g, w, out, 1e-6, 2),
            (x.float(), g, w, out, 1e-6, 4),
            (x, g.float(), w, out, 1e-6, 4),
            (x[:, ::2], g[:, ::2], w[:64], out[:, ::2], 1e-6, 4),
            (x, g, w, out.float(), 1e-6, 4),
        ):
            with self.assertRaises(ValueError):
                self.new.rms_norm_gated_rows(*args)
        empty = x[:0]
        result = self.new.rms_norm_gated_rows(empty, g[:0], w, out[:0], 1e-6, 8)
        self.assertEqual(tuple(result.shape), (0, 128))

    def test_wrapper_default_off_and_narrow_decode_guard(self):
        old, new = h.load_kernels(cpu=True)
        calls = []

        class Launch:
            def __init__(self, label):
                self.label = label

            def __getitem__(self, grid):
                return lambda **kwargs: calls.append((self.label, grid))

        def fake(shape, dtype, device, contiguous=True):
            count = 1
            for n in shape:
                count *= n
            return NS(
                shape=shape,
                ndim=len(shape),
                dtype=dtype,
                device=NS(type=device),
                numel=lambda: count,
                is_contiguous=lambda: contiguous,
            )

        old.torch = NS(empty_like=lambda x: x)
        old._glm_rms_norm_gated_row_kernel = Launch("old")
        old._glm_rms_norm_gated_large_row_kernel = Launch("large")
        old.rms_norm_gated_rows = (
            lambda x, g, w, out, eps, group: calls.append(("group", group)) or out
        )
        cases = [
            (None, "decode", "npu", 1664, 128, True, "old"),
            ("0", "decode", "npu", 1664, 128, True, "old"),
            ("1", "decode", "npu", 1664, 128, True, "old"),
            ("4", "decode", "npu", 1664, 128, True, "group"),
            ("8", "decode", "npu", 1664, 128, True, "group"),
            ("8", "prefill", "npu", 1664, 128, True, "old"),
            ("8", "null", "npu", 1664, 128, True, "old"),
            ("8", "decode", "cpu", 1664, 128, True, "old"),
            ("8", "decode", "npu", 1023, 128, True, "old"),
            ("8", "decode", "npu", 1024, 128, True, "group"),
            ("8", "decode", "npu", 2048, 128, True, "group"),
            ("8", "decode", "npu", 2049, 128, True, "old"),
            ("8", "decode", "npu", 65536, 128, True, "large"),
            ("8", "decode", "npu", 1664, 64, True, "old"),
            ("8", "decode", "npu", 1664, 128, False, "old"),
        ]
        for setting, mode, device, rows, width, contig, want in cases:
            with self.subTest(
                setting=setting,
                mode=mode,
                device=device,
                rows=rows,
                width=width,
                contig=contig,
            ):
                x = fake((rows, width), torch.bfloat16, device, contig)
                w = fake((width,), torch.float32, device)
                old.get_server_args = lambda: NS(disaggregation_mode=mode)
                calls.clear()
                with patch.dict(os.environ, {}):
                    os.environ.pop(h.FLAG, None)
                    if setting is not None:
                        os.environ[h.FLAG] = setting
                    old.glm_rms_norm_gated_npu(x, x, w, 1e-6)
                self.assertEqual(calls[0][0], want)
                if want == "group":
                    self.assertEqual(calls[0][1], int(setting))
        # The early empty return still avoids both old and grouped kernels.
        calls.clear()
        x = fake((0, 128), torch.bfloat16, "npu")
        old.glm_rms_norm_gated_npu(x, x, fake((128,), torch.float32, "npu"), 1e-6)
        self.assertEqual(calls, [])

    def test_original_kernel_bodies_unchanged(self):
        rel = "python/sglang/srt/hardware_backend/npu/attention/glm53/rms_norm_gated_npu.py"
        baseline = subprocess.check_output(
            ["git", "show", f"83cfe9668fb98ca674c3a28816dd99016e38b8b6:{rel}"],
            cwd=ROOT,
            text=True,
        )
        current = (ROOT / rel).read_text()

        def kernels(source):
            return {
                n.name: ast.dump(n, include_attributes=False)
                for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name.startswith("_glm_")
            }

        self.assertEqual(kernels(baseline), kernels(current))


if __name__ == "__main__":
    unittest.main()
