"""CPU-only regression of direct A/B extraction, strict checks and CLI guards."""

import builtins
import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

PATH = Path(__file__).with_name("bench_glm53_prefill_topk_metadata_ab.py")
SPEC = importlib.util.spec_from_file_location("direct_topk_ab", PATH)
AB = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AB)
AB.torch = torch


class DirectABHarnessTests(unittest.TestCase):
    def test_plain_and_observed_copies_only_differ_by_probe(self):
        source = AB.SOURCE.read_text()
        plain = AB.extract(source, torch)
        observed = AB.extract(source, torch, lambda *_: None)
        self.assertNotIn("_observer", plain.__code__.co_names)
        self.assertIn("_observer", observed.__code__.co_names)
        self.assertNotIn("sglang", plain.__globals__)

    def test_source_without_candidate_refuses_noop_ab(self):
        source = AB.SOURCE.read_text().replace(AB.FLAG, "DISABLED")
        with self.assertRaisesRegex(ValueError, "no metadata-reuse flag"):
            AB.extract(source, torch)

    def test_observer_compares_bytes_including_signed_zero(self):
        plan = [(0, 0, 0, 2, 4)]
        old = AB.MaskedLogits(plan)
        baseline = torch.zeros((2, 4), dtype=torch.bfloat16)
        old(baseline, *plan[0], False)
        baseline[0, 0] = -0.0
        new = AB.MaskedLogits(plan, old.tiles)
        new(baseline, *plan[0], True)
        result = new.summary()
        self.assertFalse(result["strict_equal"])
        self.assertEqual(result["mismatch_bytes"], 1)
        self.assertTrue(result["failures"][0]["torch_equal"])
        self.assertEqual(old.summary()["baseline_host_bytes"], 16)

    def test_observer_rejects_missing_tiles_or_wrong_branch(self):
        plan = [(0, 0, 0, 2, 4)]
        observer = AB.MaskedLogits(plan)
        with self.assertRaisesRegex(AssertionError, "missing"):
            observer.summary()
        with self.assertRaisesRegex(AssertionError, "branch"):
            observer(torch.zeros((2, 4), dtype=torch.bfloat16), *plan[0], True)

    def test_strict_integer_check_detects_reordering(self):
        a = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        self.assertFalse(AB.compare(a, a.flip(1))["strict_equal"])
        self.assertTrue(AB.compare(a, a.clone())["strict_equal"])

    def test_npu_confirmation_failure_precedes_device_import(self):
        original = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name in ("torch", "torch_npu"):
                self.fail("device library imported before confirmation check")
            return original(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "result.json"
            with patch("builtins.__import__", guarded), contextlib.redirect_stderr(
                io.StringIO()
            ):
                result = AB.main(["--device", "npu:0", "--output", str(output)])
            self.assertEqual(result, 2)
            self.assertEqual(json.loads(output.read_text())["status"], "error")
            previous = output.read_bytes()
            with self.assertRaises(FileExistsError):
                AB.main(["--device", "npu:0", "--output", str(output)])
            self.assertEqual(output.read_bytes(), previous)


if __name__ == "__main__":
    unittest.main()
