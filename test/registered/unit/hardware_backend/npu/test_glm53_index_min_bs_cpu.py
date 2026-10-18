"""CPU checks for conservative batch-LI scope; no CANN or accuracy conclusion."""

import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "test/manual/ascend"))
import glm53_index_batch_fixture as fx
from glm53_index_batch_contract_v2 import compare_snapshots, snapshot

MIN_BS = "SGLANG_GLM53_VERIFY_INDEX_MIN_BS"
NPU_QUERY = NS(device=NS(type="npu"))  # Metadata only; all tensors remain on CPU.


class IndexMinBSCPU(unittest.TestCase):
    def setUp(self):
        env = patch.dict(
            os.environ, {"TRITON_INTERPRET": "1", "SGLANG_GLM53_KPOOL_EXPAND": "1"}
        )
        env.start()
        self.addCleanup(env.stop)
        torch.set_num_threads(1)
        self.capture = True
        self.ns = dict(
            get_server_args=lambda: NS(
                disaggregation_mode="decode", max_running_requests=2
            ),
            get_device=lambda: NS(device="cpu"),
            get_spec=lambda: NS(speculative_num_draft_tokens=None),
            get_is_capture_mode=lambda: self.capture,
        )
        cls = fx.load_indexer(self.ns)

        class Init(cls, torch.nn.Module):
            pass

        self.cls = Init

    def initialized(self, value=None, *, enabled=True):
        obj = self.cls()
        obj.index_kpool, obj.head_dim = 4, 128
        with patch.dict(os.environ, {fx.FLAG: "1" if enabled else "0"}):
            os.environ.pop(MIN_BS, None)
            if value is not None:
                os.environ[MIN_BS] = value
            obj._init_ascend_kpool()
        return obj

    def test_default_and_explicit_one_preserve_capture_eligibility(self):
        for value in (None, "1"):
            obj = self.initialized(value)
            self.assertEqual(obj._batch_kpool_verify_min_bs, 1)
            # Configuration is sampled at init, not per forward/capture.
            with patch.dict(os.environ, {MIN_BS: "12"}):
                for bs in range(18):
                    self.assertEqual(
                        obj._verify_index_batch_enabled(NPU_QUERY, 4, bs), 1 <= bs <= 16
                    )

    def test_min_twelve_full_range_and_non_capture(self):
        obj = self.initialized("12")
        for capture in (True, False):
            self.capture = capture
            for bs in range(18):
                with self.subTest(capture=capture, bs=bs):
                    self.assertEqual(
                        obj._verify_index_batch_enabled(NPU_QUERY, 4, bs),
                        capture and 12 <= bs <= 16,
                    )

    def test_all_valid_minima_and_batch_flag_still_required(self):
        for minimum in range(1, 17):
            obj = self.initialized(str(minimum))
            for bs in (minimum - 1, minimum, 16, 17):
                self.assertEqual(
                    obj._verify_index_batch_enabled(NPU_QUERY, 4, bs),
                    minimum <= bs <= 16,
                )
        obj = self.initialized("12", enabled=False)
        self.assertFalse(obj._verify_index_batch_enabled(NPU_QUERY, 4, 13))

    def test_invalid_values_fail_at_init_even_when_batch_li_is_off(self):
        for value in ("", "0", "-1", "17", "1000", "abc", "1.5", "12x", "nan"):
            for enabled in (False, True):
                with self.subTest(value=value, enabled=enabled):
                    with self.assertRaisesRegex(ValueError, MIN_BS + r".*\[1, 16\]"):
                        self.initialized(value, enabled=enabled)

    def test_real_verify_dispatch_and_state_contract_with_min_twelve(self):
        minimum = self.initialized("12")._batch_kpool_verify_min_bs
        # Use the unchanged fixtures/oracles and production verify method.
        # Only the eligibility query's device metadata is substituted; no
        # tensor, kernel, or graph is created on an accelerator.
        for bs, capture in [(n, True) for n in range(1, 18)] + [(13, False)]:
            with self.subTest(bs=bs, capture=capture):
                case = fx.fixture([252 + row % 4 for row in range(bs)], heads=2)
                a = fx.Runner(case, fx.cpu_lightning, enabled=False, fused=True)
                b = fx.Runner(
                    case,
                    fx.cpu_lightning,
                    enabled=True,
                    batch_expand=True,
                    fused=True,
                    bypass_cpu_guard=False,
                )
                b.obj._batch_kpool_verify_min_bs = minimum
                b.cls._verify_index_batch_enabled.__globals__["get_is_capture_mode"] = (
                    lambda: capture
                )
                b.obj._verify_index_batch_enabled = lambda q, steps, size: (
                    b.cls._verify_index_batch_enabled(b.obj, NPU_QUERY, steps, size)
                )
                x, y = a.run(), b.run()
                batched = capture and 12 <= bs <= 16
                self.assertEqual(len(b.calls), 1 if batched else 4)
                self.assertEqual(
                    [shape[0] for shape, _ in b.expands],
                    [bs * 4] if batched else [bs] * 4,
                )
                compare_snapshots(
                    case,
                    snapshot(a, x),
                    snapshot(b, y),
                    batch_expand_b=batched,
                    diagnostics={},
                )
                if not batched:
                    fx.compare(a, x, b, y)  # Full old-path byte/shape contract.


if __name__ == "__main__":
    unittest.main()
