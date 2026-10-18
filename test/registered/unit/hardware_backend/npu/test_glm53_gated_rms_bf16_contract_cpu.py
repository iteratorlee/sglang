"""CPU-only explicit output-contract guards and diagnostic fixture checks."""

import ast
import importlib.util
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

os.environ["TRITON_INTERPRET"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import torch

ROOT = Path(__file__).resolve().parents[5]
UNIT = ROOT / "test/manual/ascend/test_glm53_gated_rms_contract_npu.py"
spec = importlib.util.spec_from_file_location("rms_contract", UNIT)
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)
h = v2.load_reference(UNIT.with_name("test_glm53_gated_rms_rows_npu.py"))


class BF16ContractCPU(unittest.TestCase):
    def test_explicit_output_guards(self):
        old, new = h.load_kernels(cpu=True)
        x, gate, weight = h.inputs(4)
        with self.assertRaises(ValueError):
            new.rms_norm_gated_rows(x, gate, weight, x.float(), 1e-6, 4)
        calls = []

        def fake(dtype):
            return NS(
                shape=(1664, 128),
                dtype=dtype,
                device=NS(type="npu"),
                numel=lambda: 1664 * 128,
            )

        class Kernel:
            def __getitem__(self, grid):
                return lambda **kwargs: calls.append("old")

        old._glm_rms_norm_gated_row_kernel = Kernel()
        old.supported = lambda *args: True  # Outer guard must reject independently.
        old.rms_norm_gated_rows = lambda *args: calls.append("group")
        for dtype, expected in ((torch.float32, "old"), (torch.bfloat16, "group")):
            old.torch = NS(empty_like=lambda x: fake(dtype), bfloat16=torch.bfloat16)
            calls.clear()
            with patch.dict(os.environ, {h.FLAG: "4"}):
                old.glm_rms_norm_gated_npu(
                    fake(torch.bfloat16),
                    fake(torch.bfloat16),
                    NS(numel=lambda: 128),
                    1e-6,
                )
            self.assertEqual(calls, [expected])

    def test_difference_preserves_zero_sign_and_ulp(self):
        a = torch.zeros((2, 128), dtype=torch.float32)
        b = a.clone()
        b[1, 3] = -0.0
        result = v2.difference(a, b)
        self.assertFalse(result["exact"])
        self.assertEqual((result["first_row"], result["first_col"]), (1, 3))
        self.assertEqual(result["different_elements"], 1)
        self.assertEqual(result["max_finite_ulp"], 1)
        a.fill_(1)
        b = torch.nextafter(a, torch.full_like(a, float("inf")))
        self.assertEqual(v2.difference(a, b)["max_finite_ulp"], 1)

    def test_midpoint_fixtures_are_finite_and_inputs_unchanged(self):
        for pattern in ("random", "wide", "zero", "signed-zero"):
            original = h.inputs(32, torch.float32, pattern)
            fixtures = [
                v2.input_factory(h.inputs, midpoint_direction=d)(
                    32, torch.float32, pattern
                )
                for d in (-1, 0, 1)
            ]
            for x, g, w in fixtures:
                h.exact(x, original[0], "x")
                h.exact(g, original[1], "gate")
                self.assertTrue(torch.isfinite(w).all())
                self.assertEqual(w.dtype, torch.float32)
            if pattern == "random":
                self.assertFalse(torch.equal(fixtures[0][2], fixtures[2][2]))

    def test_v1_strict_logic_and_kernel_bodies_unchanged(self):
        base = "53718c52f31f8ceb58c97649d6c15d6c26f616dc"
        for rel in (
            "test/manual/ascend/test_glm53_gated_rms_rows_npu.py",
            "python/sglang/srt/hardware_backend/npu/attention/glm53/rms_norm_gated_npu.py",
            "python/sglang/srt/hardware_backend/npu/attention/glm53/rms_norm_gated_rows_npu.py",
        ):
            before = ast.parse(
                subprocess.check_output(
                    ["git", "show", f"{base}:{rel}"], cwd=ROOT, text=True
                )
            )
            after = ast.parse((ROOT / rel).read_text())

            def selected(tree):
                return [
                    ast.dump(n)
                    for n in tree.body
                    if isinstance(n, ast.FunctionDef)
                    and (rel.startswith("test/") or n.name.startswith("_glm_"))
                ]

            self.assertEqual(selected(before), selected(after))


if __name__ == "__main__":
    unittest.main()
