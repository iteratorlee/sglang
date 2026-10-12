"""Meaningful contract tests; no torch_npu import or device initialization."""

import ast
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from score_contract import (
    REL,
    Capture,
    boundary_metrics,
    compare_ids,
    compare_scores,
    cpu_compact_emulation,
    extract,
    fixture,
    legacy_scores,
    load_file,
    make_scorer,
)

ROOT = Path(__file__).resolve().parents[2]
PARALLEL = load_file(ROOT / REL / "prefill_index_parallel.py", "actual_partition_cpu")


class Contract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_signed_gates_relu_order(self):
        q, w, k = fixture("signed", 1, 20)
        expected = (
            k[:, 0].float().clamp_min(0) * -1
            + (k[:, 1].float().clamp_min(0) * 0.75).bfloat16().float()
        ).bfloat16()
        self.assertTrue(torch.equal(legacy_scores(q, w, k)[0], expected))
        wrong = (
            (torch.einsum("qhd,kd->qhk", q, k).float() * w[:, :, None]).sum(1).relu()
        )
        self.assertFalse(torch.equal(wrong.bfloat16(), expected[None]))

    def test_dot_rounding_is_observable(self):
        q, w, k = fixture("dot_round", 1, 3)
        no_dot_cast = (
            (torch.einsum("qhd,kd->qhk", q.float(), k.float()).relu() * w[:, :, None])
            .bfloat16()
            .sum(1)
        )
        self.assertFalse(torch.equal(legacy_scores(q, w, k), no_dot_cast))

    def test_product_rounding_is_observable(self):
        q, w, k = fixture("product_round", 1, 3)
        no_product_cast = (
            (torch.einsum("qhd,kd->qhk", q, k).float().relu() * w[:, :, None])
            .sum(1)
            .bfloat16()
        )
        self.assertFalse(torch.equal(legacy_scores(q, w, k), no_product_cast))

    def test_fp32_reduction_order_is_not_proven_by_final_bf16_cast(self):
        a, b, c, d = [
            torch.tensor(x, dtype=torch.float32) for x in (2**30, 1, -(2**30), 2)
        ]
        left = ((a + b) + c) + d
        paired = (a + b) + (c + d)
        exact = (a.double() + b.double() + c.double() + d.double()).bfloat16()
        self.assertEqual(float(left.bfloat16()), 2)
        self.assertEqual(float(paired.bfloat16()), 0)
        self.assertEqual(float(exact), 3)

    def test_contract_raw_bits_rejects_signed_zero(self):
        a, b = Capture(), Capture()
        a(0, 0, 0, torch.tensor([[0.0]], dtype=torch.bfloat16))
        b(0, 0, 0, torch.tensor([[-0.0]], dtype=torch.bfloat16))
        self.assertFalse(compare_scores(a, b)["exact"])

    def test_nonfinite_rejected_even_if_equal(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            a = Capture()
            a(0, 0, 0, torch.tensor([[value]], dtype=torch.bfloat16))
            self.assertFalse(compare_scores(a, a)["exact"])

    def test_tie_set_or_order_change_rejected(self):
        a = torch.tensor([[0, 1]], dtype=torch.int32)
        self.assertFalse(compare_ids(a, a.flip(1), [1], [16])["exact"])
        self.assertFalse(compare_ids(a, torch.tensor([[0, 2]]), [1], [16])["exact"])

    def test_strict_omission_is_distinguished_from_ties(self):
        capture = Capture()
        scores = torch.zeros((1, 514), dtype=torch.bfloat16)
        scores[:, :511] = 4
        scores[:, 511] = 2
        scores[:, 512] = 2.5
        capture(0, 0, 2055, scores)
        ids = torch.arange(512).reshape(1, -1)
        result = boundary_metrics(capture, ids, [1], [2056])
        self.assertEqual(result["strictly_better_old_pools_omitted"], 1)
        self.assertEqual(result["tied_512_513_rows"], 0)
        self.assertEqual(result["min_margin"], 0.5)

    def test_causal_invalid_rejected_even_if_equal(self):
        a = torch.tensor([[0, -1]])
        self.assertFalse(compare_ids(a, a, [1], [3])["exact"])

    def test_no_future_candidates_and_empty_closed_prefix(self):
        for seq, rows in ((3, 3), (7, 6), (132, 9), (263, 7)):
            q, w, k = fixture("signed", rows, seq // 4)
            a, b = Capture(), Capture()
            x = make_scorer(ROOT, False, 8, observer=a)(q, w, [k], [rows], [seq])
            y = make_scorer(ROOT, True, 8, cpu_compact_emulation, b)(
                q, w, [k], [rows], [seq]
            )
            self.assertTrue(compare_scores(a, b)["exact"])
            self.assertTrue(compare_ids(x, y, [rows], [seq])["exact"])
            if seq == 3:
                self.assertTrue((x == -1).all())

    def test_partial_bq_bk_and_tie_boundary(self):
        for rows, pools in ((1, 63), (3, 65), (5, 513), (7, 515)):
            q, w, k = fixture("boundary_tie", rows, pools)
            self.assertTrue(
                torch.equal(legacy_scores(q, w, k), cpu_compact_emulation(q, w, k))
            )

    def test_tp8_partition_preserves_positions_and_pool_dimension(self):
        for chunk in (2048, 4096, 8192):
            for seq in (262144, 524288, 524287):
                parts, capacity = PARALLEL.partition_queries([chunk], [seq], 8)
                seen = []
                for fragments in parts:
                    for f in fragments:
                        seen.extend(range(f.offset, f.offset + f.length))
                        self.assertEqual(
                            f.sequence_end - f.length, seq - chunk + f.offset
                        )
                        first = f.sequence_end - f.length
                        if f.offset:
                            self.assertEqual(first % 128, 0)
                self.assertEqual(seen, list(range(chunk)))
                self.assertGreaterEqual(capacity * 8, chunk)

    def test_tp8_varlen_scoring_reconstructs_exact_order(self):
        lengths, ends = [9, 137, 6], [132, 263, 7]
        inputs = [
            fixture("signed", length, end // 4, 1012 + i)
            for i, (length, end) in enumerate(zip(lengths, ends))
        ]
        q, w = [torch.cat([x[i] for x in inputs]) for i in (0, 1)]
        histories = [x[2] for x in inputs]
        scorer = make_scorer(ROOT, False, 8)
        full = scorer(q, w, histories, lengths, ends)
        parts, _ = PARALLEL.partition_queries(lengths, ends, 8)
        out = []
        for fragments in parts:
            if not fragments:
                continue
            meta = PARALLEL.FragmentBatchMetadata.from_fragments(fragments)
            out.append(
                scorer(
                    torch.cat([q[f.offset : f.offset + f.length] for f in fragments]),
                    torch.cat([w[f.offset : f.offset + f.length] for f in fragments]),
                    [histories[f.request] for f in fragments],
                    meta.extend_seq_lens_cpu,
                    meta.seq_lens_cpu,
                )
            )
        self.assertTrue(torch.equal(full, torch.cat(out)))

    def test_actual_flag_policy_and_unsupported_contract(self):
        env = {"SGLANG_GLM53_PREFILL_POOLED_SCORE": "auto"}
        scope = dict(
            os=SimpleNamespace(environ=env),
            get_server_args=lambda: SimpleNamespace(disaggregation_mode="prefill"),
        )
        guard = extract(
            ROOT / REL / "kpool_indexer.py", "_prefill_pooled_score_enabled", scope
        )
        self.assertTrue(guard(SimpleNamespace(type="npu"), 8))
        self.assertFalse(guard(SimpleNamespace(type="npu"), 32))
        self.assertFalse(guard(SimpleNamespace(type="cpu"), 8))
        env.clear()
        self.assertFalse(guard(SimpleNamespace(type="npu"), 8))
        supported = extract(
            ROOT / REL / "pooled_score_prefill_npu.py", "supported", {"torch": torch}
        )
        q, w, k = fixture("random", 4, 32)
        self.assertFalse(supported(q, w, [k]))

    def test_postop_harness_preserves_legacy_tiles_and_sum(self):
        def cpu_postop(scores, weights):
            scores.relu_().mul_(weights.unsqueeze(-1))
            return True

        for tile in (8, 32):
            for family in ("random", "head_cancel", "rank_flip"):
                q, w, k = fixture(family, 33, 2048)
                old_capture, new_capture = Capture(), Capture()
                old = make_scorer(ROOT, False, tile, observer=old_capture)
                new = make_scorer(
                    ROOT, False, tile, observer=new_capture, postop=cpu_postop
                )
                x = old(q, w, [k], [33], [8192])
                y = new(q, w, [k], [33], [8192])
                self.assertTrue(new.calls)
                self.assertEqual(old_capture.tiles, new_capture.tiles)
                self.assertTrue(compare_scores(old_capture, new_capture)["exact"])
                self.assertTrue(compare_ids(x, y, [33], [8192])["exact"])


if __name__ == "__main__":
    unittest.main()
