"""CPU guards/launch ownership and body identity; never Triton interpreter."""

import ast
import importlib.util
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("postop_base", HERE / "bench_glm53_postop64_ab.py")
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
ROOT = HERE.resolve().parents[2]
DIR = ROOT / h.REL
GROUP = "SGLANG_GLM53_PREFILL_POOLED_POSTOP32_GROUP"
MIN_KEYS = "SGLANG_GLM53_PREFILL_POOLED_POSTOP32_MIN_KEYS"


class GroupedCPU(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {GROUP: "0", MIN_KEYS: "32768", h.FLAG: "0",
                                          "SGLANG_GLM53_PREFILL_POOLED_POSTOP": "auto"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.role = "prefill"
        self.scope = dict(os=os, torch=torch,
                          triton=SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b),
                          get_server_args=lambda: SimpleNamespace(disaggregation_mode=self.role))
        h.extract_defs(DIR / "pooled_score_postop_npu.py",
                       ["_postop32_group", "relu_gate_"], self.scope)
        h.extract_defs(DIR / "kpool_indexer.py", ["_prefill_pooled_postop_min_keys"], self.scope)
        self.scope["supported"] = lambda *_: True

    def test_policy_scope_defaults_override_and_invalid_inputs(self):
        policy = self.scope["_prefill_pooled_postop_min_keys"]
        device = SimpleNamespace(type="npu")
        self.assertEqual(policy(device, 8, 32), 98304)
        self.assertEqual(policy(device, 32, 32), 32768)
        for group in ("4", "8"):
            os.environ[GROUP] = group
            for tile in (8, 32):
                self.assertEqual(policy(device, tile, 32), 32768)
            for value in ("1", "8192", "16384", "32768"):
                os.environ[MIN_KEYS] = value
                self.assertEqual(policy(device, 8, 32), int(value))
            for value in ("", "0", "-1", "32769", "8k", "1.5"):
                os.environ[MIN_KEYS] = value
                with self.assertRaises(ValueError):
                    policy(device, 32, 32)
            # Bad override must be ignored outside explicit P32 scope.
            self.assertEqual(policy(device, 8, 64), 98304)
            self.assertIsNone(policy(SimpleNamespace(type="cpu"), 8, 32))
            self.role = "decode"
            self.assertIsNone(policy(device, 8, 32))
            self.role = "prefill"
            os.environ[MIN_KEYS] = "32768"
            self.assertIsNone(policy(device, 16, 32))
        os.environ[GROUP] = "0"
        os.environ[MIN_KEYS] = "invalid"
        self.assertEqual(policy(device, 8, 32), 98304)
        os.environ[GROUP] = "2"
        with self.assertRaises(ValueError):
            policy(device, 32, 32)
        with self.assertRaises(ValueError):
            self.scope["_postop32_group"](32)
        self.assertEqual(self.scope["_postop32_group"](64), 0)

    def test_launch_grid_and_legacy64_unchanged(self):
        launches = []

        class Kernel:
            def __init__(self, name):
                self.name = name

            def __getitem__(self, grid):
                def run(*args, **kwargs):
                    launches.append((self.name, grid, args[2:], kwargs))
                return run

        self.scope["_relu_gate_inplace"] = Kernel("legacy")
        self.scope["_relu_gate_grouped"] = Kernel("grouped")
        weights = Mock()
        for group, programs in (("4", 1024), ("8", 512)):
            os.environ[GROUP] = group
            scores = Mock(shape=(32, 32, 32768))
            self.assertTrue(self.scope["relu_gate_"](scores, weights))
            self.assertEqual(launches[-1], ("grouped", (programs,), (32768, 4096, 8192),
                                            dict(num_warps=4, multibuffer=False)))
            scores.relu_.assert_not_called()
        self.assertTrue(self.scope["relu_gate_"](Mock(shape=(32, 64, 32768)), weights))
        self.assertEqual(launches[-1][0:2], ("legacy", (2048, 4)))
        os.environ[GROUP] = "0"
        self.assertTrue(self.scope["relu_gate_"](Mock(shape=(8, 32, 8192)), weights))
        self.assertEqual(launches[-1][0:2], ("legacy", (256, 1)))

    def test_outside_grouped_envelope_falls_back_once(self):
        os.environ[GROUP] = "4"
        for count, block in ((32769, 8192), (8192, 2048), (8192, 4096)):
            scores, weights = Mock(shape=(8, 32, count)), Mock()
            self.assertFalse(self.scope["relu_gate_"](scores, weights, block))
            scores.relu_.assert_called_once_with()
            scores.mul_.assert_called_once_with(weights.unsqueeze.return_value)

    def test_body_identical_and_small_grid_stride_owns_every_element_once(self):
        tree = ast.parse((DIR / "pooled_score_postop_npu.py").read_text())
        nodes = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        old = nodes["_relu_gate_inplace"].body[2:]
        loop = nodes["_relu_gate_grouped"].body[1]
        self.assertIsInstance(loop, ast.For)
        self.assertEqual([ast.dump(n) for n in old], [ast.dump(n) for n in loop.body[2:]])
        # Small CPU simulation exercises tail masks and nondivisible grids.
        for rows, keys in ((1, 17), (3, 33)):
            source = torch.arange(rows * 32 * keys).remainder(31).sub(15).bfloat16().reshape(rows, 32, keys)
            weights = torch.linspace(-2, 2, rows * 32).reshape(rows, 32)
            expected = source.clone().relu_().mul_(weights.unsqueeze(-1))
            block = 8  # Simulate ownership only, never run a Triton interpreter.
            key_blocks = (keys + block - 1) // block
            total = rows * 32 * key_blocks
            for group in (4, 8):
                actual = source.clone().reshape(rows * 32, keys)
                seen = []
                programs = (total + group - 1) // group
                for pid in range(programs):
                    for block_id in range(pid, total, programs):
                        seen.append(block_id)
                        head, tile = divmod(block_id, key_blocks)
                        section = actual[head, tile * block:min((tile + 1) * block, keys)]
                        section.relu_().mul_(weights.reshape(-1)[head])
                self.assertEqual(sorted(seen), list(range(total)))
                self.assertTrue(torch.equal(actual.reshape_as(expected).view(torch.int16),
                                            expected.view(torch.int16)))


if __name__ == "__main__":
    unittest.main()
