"""Small P32 harness checks; CPU tensors only, no Triton or NPU import."""

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

BENCH = Path(__file__).with_name("bench_glm53_postop64_ab.py")
spec = importlib.util.spec_from_file_location("postop32_sweep", BENCH)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
bench.torch = torch
MODULE = Path(__file__).resolve().parents[3] / bench.REL / "kpool_indexer.py"


class Postop32SweepCPU(unittest.TestCase):
    def test_sweep_scope_and_options(self):
        args = SimpleNamespace(heads=32, blocks=[2048, 4096, 8192],
                               keys=[8192, 16384, 32768], force_min_keys=None)
        bench.validate_sweep_options(args)
        for change in ({"heads": 64}, {"force_min_keys": 8192},
                       {"blocks": [2048, 2048]}, {"keys": [8193]}):
            bad = SimpleNamespace(**(vars(args) | change))
            with self.subTest(change=change), self.assertRaises(ValueError):
                bench.validate_sweep_options(bad)

    def test_real_caller_uses_native_reference_and_requested_block(self):
        q = torch.arange(4 * 32 * 128).reshape(4, 32, 128).remainder(13).bfloat16()
        weights = torch.linspace(-1.5, 2.0, 128).reshape(4, 32)
        keys = torch.arange(32 * 128).reshape(32, 128).remainder(7).bfloat16()
        batch = SimpleNamespace(batch_size=1, extend_seq_lens_cpu=[4], seq_lens_cpu=[128])
        indexer = SimpleNamespace(index_topk=32, index_kpool=4)
        seen = []

        def fake_launch(scores, gates, *, block):
            seen.append(block)
            scores.relu_().mul_(gates.unsqueeze(-1))
            return True  # CPU control-flow oracle only; no NPU parity claim.

        helper = SimpleNamespace(relu_gate_=fake_launch)
        for block in (2048, 4096, 8192):
            calls = [[], []]
            outputs = []
            with bench.environment(0, 8), patch.object(bench, "formats", return_value=[2, 2]):
                for enabled in (False, True):
                    fn = bench.forced32_scorer(MODULE, helper, block, enabled,
                                               calls=calls[int(enabled)])
                    outputs.append(fn(indexer, q, weights, [keys], batch))
            self.assertEqual(calls[0], [])
            self.assertEqual(len(calls[1]), 1)
            self.assertIs(calls[1][0]["active"], True)
            self.assertEqual(seen[-1], block)
            self.assertTrue(torch.equal(outputs[0], outputs[1]))
        self.assertEqual(seen, [2048, 4096, 8192])

    def test_candidate_fallback_is_failure(self):
        scores = torch.zeros((1, 32, 2), dtype=torch.bfloat16)
        weights = torch.ones((1, 32))
        helper = SimpleNamespace(relu_gate_=lambda *_, **__: False)
        with self.assertRaisesRegex(AssertionError, "fell back"):
            bench.launch32(helper, scores, weights, 8192)
        with self.assertRaisesRegex(AssertionError, "non-32-head"):
            bench.launch32(helper, scores[:, :16], weights[:, :16], 8192)


if __name__ == "__main__":
    unittest.main()
