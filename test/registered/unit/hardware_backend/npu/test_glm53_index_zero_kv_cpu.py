"""CPU fault injection for production zero-key canonicalization and v2 gates."""

from copy import deepcopy
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "test/manual/ascend"))
import glm53_index_batch_fixture as fx
from glm53_index_batch_contract_v2 import compare_snapshots, snapshot


class IndexZeroKVCPU(unittest.TestCase):
    def setUp(self):
        env = patch.dict(
            os.environ, {"TRITON_INTERPRET": "1", "SGLANG_GLM53_KPOOL_EXPAND": "0"}
        )
        env.start()
        self.addCleanup(env.stop)
        torch.set_num_threads(1)

    def pair(self, starts=(0, 1, 2, 3, 60), requests=(1, 2, 3, 4, 0), *, fused=False):
        case = fx.fixture(list(starts), requests=list(requests))
        before = []
        serial = 0

        def dirty_li(**kwargs):
            nonlocal serial
            serial += 1
            ids, _ = fx.cpu_lightning(**kwargs)
            zero = kwargs["actual_seq_lengths_key"] == 0
            # Shape- and invocation-dependent garbage, including valid-looking
            # ids. Real positive key-length outputs remain the reference.
            ids[zero] = torch.arange(ids.shape[-1], dtype=ids.dtype) + serial * 97
            before.append(ids.clone())
            return ids, None

        a, b = [
            fx.Runner(case, dirty_li, enabled=e, fused=fused) for e in (False, True)
        ]
        x, y = a.run(), b.run()
        aa, bb = snapshot(a, x), snapshot(b, y)
        diag = {}
        compare_snapshots(case, aa, bb, diagnostics=diag)
        self.assertEqual(b.events, ["update"] * 4 + ["indexer"] + ["expand"] * 4)
        self.assertEqual((len(a.calls), len(b.calls)), (4, 1))
        positive = b.calls[0]["actual_seq_lengths_key"] > 0
        fx.exact(
            b.pooled[0][positive], before[-1][positive], "positive lengths untouched"
        )
        self.assertGreater(
            diag["dead_raw"]["elements"] + diag["active_zero_raw"]["elements"], 0
        )
        with self.assertRaisesRegex(AssertionError, "not bitwise exact"):
            fx.compare(a, x, b, y)  # Existing strict behavior is not relaxed.
        return case, a, b, aa, bb, diag

    def test_dirty_zero_keys_short_prefixes_mixed_and_dead(self):
        for fused in (False, True):
            with self.subTest(fused=fused):
                self.pair(fused=fused)
                self.pair((252, 253), (0, 0), fused=fused)
        # Exercise the real fused expansion in Triton's CPU interpreter too.
        with patch.dict(os.environ, {"SGLANG_GLM53_KPOOL_EXPAND": "1"}):
            self.pair(fused=True)

    def test_graph_eager_contract_does_not_compare_old_garbage(self):
        case, a, b, aa, bb, _ = self.pair()
        a.reset()
        repeated = snapshot(a, a.run())
        diag = {}
        compare_snapshots(
            case, aa, repeated, canonical_a=False, canonical_b=False, diagnostics=diag
        )
        self.assertGreater(diag["dead_raw"]["elements"], 0)
        self.assertGreater(diag["active_zero_raw"]["elements"], 0)
        b.reset()
        compare_snapshots(
            case,
            bb,
            snapshot(b, b.run()),
            canonical_a=True,
            canonical_b=True,
            diagnostics={},
        )

    def test_negative_output_mutations_are_rejected(self):
        case, _, _, aa, bb, _ = self.pair()
        mutations = (
            ("zero-key canonical", lambda s: s["raw"][0, 0].fill_(9)),
            ("zero-key canonical", lambda s: s["raw"][-1, 0].fill_(9)),
            ("positive-key raw", lambda s: s["raw"][3, 0, 0].add_(1)),
            ("positive-key expanded", lambda s: s["output"][12, -1].add_(1)),
            ("canonical zero-key expansion", lambda s: s["output"][0, 0].fill_(1)),
            ("canonical zero-key expansion", lambda s: s["output"][-3, -1].add_(1)),
            ("shape padding", lambda s: s["output"][-1, 0].fill_(1)),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                broken = deepcopy(bb)
                mutate(broken)
                diag = {}
                with self.assertRaisesRegex(AssertionError, label):
                    compare_snapshots(case, aa, broken, diagnostics=diag)
                self.assertIn("dead_expanded", diag)
        # Candidate graph/eager must check both sides' canonicalization.
        broken = deepcopy(bb)
        broken["raw"][0, 0].fill_(3)
        with self.assertRaisesRegex(AssertionError, "zero-key canonical"):
            compare_snapshots(case, broken, bb, canonical_a=True, diagnostics={})

    def test_every_state_and_cache_byte_including_dead_sink_is_checked(self):
        case, _, _, aa, bb, _ = self.pair()
        for key in (*fx.STATE, "cache"):
            for where in ("snapshot", "final"):
                with self.subTest(key=key, where=where):
                    broken = deepcopy(bb)
                    target = (
                        broken["states"][3][key]
                        if where == "snapshot"
                        else broken["final"][key]
                    )
                    # Flip one bit; this also catches +0/-0, unlike allclose.
                    target.view(torch.uint8).flatten()[0] ^= 128
                    with self.assertRaisesRegex(AssertionError, "all bytes"):
                        compare_snapshots(case, aa, broken, diagnostics={})
        broken = deepcopy(bb)
        broken["states"].pop()
        with self.assertRaisesRegex(AssertionError, "nine"):
            compare_snapshots(case, aa, broken, diagnostics={})

    def test_short_prefix_baseline_consumed_tail_cannot_be_ignored(self):
        case, _, _, aa, bb, _ = self.pair()
        aa["output"][0, 0] = 1
        with self.assertRaisesRegex(AssertionError, "causal tail"):
            compare_snapshots(case, aa, bb, diagnostics={})


if __name__ == "__main__":
    unittest.main()
