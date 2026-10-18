"""CPU state/causal proofs with real kernels in Triton's interpreter.

The explicit stable-tie reference is not evidence of CANN tie stability. The
negative test demonstrates why NPU exact tests remain a release gate.
"""

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[5]
spec = importlib.util.spec_from_file_location(
    "index_batch_fixture", ROOT / "test/manual/ascend/glm53_index_batch_fixture.py"
)
fx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fx)


class IndexBatchCPU(unittest.TestCase):
    def setUp(self):
        env = patch.dict(
            os.environ, {"TRITON_INTERPRET": "1", "SGLANG_GLM53_KPOOL_EXPAND": "0"}
        )
        env.start()
        self.addCleanup(env.stop)
        torch.set_num_threads(1)

    def pair(self, case, fused=False):
        a, b = [
            fx.Runner(case, fx.cpu_lightning, enabled=e, fused=fused)
            for e in (False, True)
        ]
        x, y = a.run(), b.run()
        fx.compare(a, x, b, y)
        self.assertEqual((len(a.calls), len(b.calls)), (4, 1))
        self.assertEqual(b.events, ["update"] * 4 + ["indexer"] + ["expand"] * 4)
        self.assertEqual([s[0][0] for s in b.expands], [case.bs] * 4)
        packed = b.calls[0]
        for field in ("query", "weights", "actual_seq_lengths_key", "block_table"):
            expected = torch.stack([call[field] for call in a.calls], 1).flatten(0, 1)
            fx.exact(packed[field], expected, field)
        fx.exact(
            packed["actual_seq_lengths_query"],
            torch.arange(1, case.bs * 4 + 1, dtype=torch.int32),
            "independent query sequences",
        )
        self.oracle(case, a, x)
        return a, b

    def oracle(self, case, runner, output):
        d = case.data
        state = {k: d[k].clone() for k in (*fx.STATE, "cache")}
        raw = runner.raw_topk()
        for step in range(4):
            for k in state:
                fx.exact(runner.states[step * 2][k], state[k], f"pre-step {step} {k}")
            for row, req in enumerate(d["req"].tolist()):
                idx = row * 4 + step
                pos = int(d["positions"][idx])
                slot = pos % 4
                state["_kpool_tail_k"][req, slot] = d["key"][idx]
                state["_kpool_tail_score"][req, slot] = d["gate"][idx]
            for row, req in enumerate(d["req"].tolist()):
                pos = int(d["positions"][row * 4 + step])
                prob = (state["_kpool_tail_score"][req] + d["ape"]).softmax(0)
                value = (
                    (state["_kpool_tail_k"][req].float() * prob)
                    .sum(0)
                    .to(torch.bfloat16)
                )
                if req > 0 and pos % 4 == 3:
                    pool = pos // 4
                    page = int(d["table"][row, pool // 64 * 4])
                    state["cache"][page, pool % 64, 0] = value
                else:
                    state["cache"].view(-1, 128)[0] = value * 0
            for k in state:
                fx.exact(
                    runner.states[step * 2 + 1][k], state[k], f"post-write {step} {k}"
                )
            for row, req in enumerate(d["req"].tolist()):
                state["_kpool_mtp_tail_k"][row, step] = state["_kpool_tail_k"][req]
                state["_kpool_mtp_tail_score"][row, step] = state["_kpool_tail_score"][
                    req
                ]
                pos = int(d["positions"][row * 4 + step])
                length = (pos + 1) // 4 if req > 0 else 0
                ids = raw[row, step].tolist()
                valid = [i for i in ids if i >= 0]
                self.assertEqual(len(valid), min(length, case.topk // 4))
                self.assertEqual(len(valid), len(set(valid)))
                self.assertTrue(all(i < length for i in valid), "future pool selected")
                expanded = [
                    i * 4 + j if i >= 0 else pos + 1 for i in ids for j in range(4)
                ] + [pos + 1] * 3
                tail_start = (pos + 1) // 4 * 4
                at = min(tail_start, case.topk)
                for j in range(3):
                    expanded[at + j] = (
                        tail_start + j if tail_start + j <= pos else pos + 1
                    )
                fx.exact(
                    output[row * 4 + step],
                    torch.tensor(expanded, dtype=torch.int32),
                    "independent tail expansion",
                )
            # Snapshot for this step is observable before the next step starts.
        for k in state:
            fx.exact(runner.states[-1][k], state[k], f"final oracle {k}")
        self.assertTrue(
            torch.equal(output[case.bs * 4 :], torch.zeros_like(output[case.bs * 4 :]))
        )

    def test_all_closing_offsets_pages_empty_history_and_ties(self):
        # 64-token page, 256-token pooled page, and the next pooled page.
        for start in (0, 1, 2, 3, 60, 61, 62, 63, 252, 253, 254, 255, 256, 511):
            for kind in ("random", "ties", "near-ties", "closing"):
                with self.subTest(start=start, kind=kind):
                    self.pair(fx.fixture([start, start + 1], kind=kind))

    def test_padding_all_padding_and_shape_padding(self):
        for reqs in ([1, 0], [0, 0], [3, 0, 1, 0]):
            # Padding slots use identical positions/payloads, so no WAW winner
            # affects even the reserved sink in the bitwise comparison.
            self.pair(fx.fixture([253] * len(reqs), requests=reqs, kind="ties"))

    def test_signed_zero_sink_is_compared_bitwise(self):
        case = fx.fixture([252])
        case.data["key"].neg_()
        case.data["_kpool_tail_k"].neg_()
        a, _ = self.pair(case)
        self.assertTrue(torch.signbit(a.states[1]["cache"].view(-1, 128)[0]).all())

    def test_bs14_keeps_four_fused_expands_and_snapshot_combinations(self):
        case = fx.fixture([252 + i % 4 for i in range(14)], kind="closing")
        for fused in (False, True):
            with patch.dict(os.environ, {"SGLANG_GLM53_KPOOL_EXPAND": "1"}):
                self.pair(case, fused=fused)

    def test_fused_and_torch_expansion_exact_including_sentinels(self):
        for bs in (1, 2, 14, 16):
            case = fx.fixture([252 + i % 4 for i in range(bs)], kind="ties", topk=2048)
            a = fx.Runner(case, fx.cpu_lightning, enabled=True)
            x = a.run()
            with patch.dict(os.environ, {"SGLANG_GLM53_KPOOL_EXPAND": "1"}):
                b = fx.Runner(case, fx.cpu_lightning, enabled=True)
                y = b.run()
            fx.compare(a, x, b, y)

    def test_dynamic_positions_requests_pages_with_fixed_storage(self):
        a_case = fx.fixture([252, 253], capacity=1024)
        b_case = fx.fixture(
            [511, 510], requests=[1, 2], capacity=1024, seed=77, kind="closing"
        )
        a, b = [fx.Runner(a_case, fx.cpu_lightning, enabled=e) for e in (False, True)]
        ptrs = [[v.data_ptr() for v in r.data.values()] for r in (a, b)]
        for case in (a_case, b_case, a_case):
            for r in (a, b):
                r.reset(case)
            x, y = a.run(), b.run()
            fx.compare(a, x, b, y)
            self.oracle(case, a, x)
            self.assertEqual(
                ptrs, [[v.data_ptr() for v in r.data.values()] for r in (a, b)]
            )

    def test_flag_and_full_eligibility_guard(self):
        ns = dict(get_is_capture_mode=lambda: True)
        cls = fx.load_indexer(ns)
        q = NS(device=NS(type="npu"))
        for mode in ("decode", "prefill", "null"):
            for setting in (None, "0", "1", "true"):

                class Init(cls, torch.nn.Module):
                    pass

                obj = Init()
                obj.index_kpool, obj.head_dim = 4, 128
                ns.update(
                    get_server_args=lambda: NS(
                        disaggregation_mode=mode, max_running_requests=2
                    ),
                    get_device=lambda: NS(device="cpu"),
                    get_spec=lambda: NS(speculative_num_draft_tokens=None),
                )
                with patch.dict(os.environ, {}):
                    os.environ.pop(fx.FLAG, None)
                    if setting is not None:
                        os.environ[fx.FLAG] = setting
                    obj._init_ascend_kpool()
                self.assertEqual(
                    obj._verify_index_batch_enabled(q, 4, 14),
                    mode == "decode" and setting == "1",
                )
        obj._batch_kpool_verify_index = True
        for device, steps, bs, pool, capture in (
            ("cpu", 4, 1, 4, True),
            ("npu", 3, 1, 4, True),
            ("npu", 4, 0, 4, True),
            ("npu", 4, 17, 4, True),
            ("npu", 4, 1, 2, True),
            ("npu", 4, 1, 4, False),
        ):
            obj.index_kpool = pool
            ns["get_is_capture_mode"] = lambda: capture
            self.assertFalse(
                obj._verify_index_batch_enabled(NS(device=NS(type=device)), steps, bs)
            )
        case = fx.fixture([252])
        r = fx.Runner(case, fx.cpu_lightning, enabled=True, bypass_cpu_guard=False)
        r.run()
        self.assertEqual(len(r.calls), 4)

    def test_empty_forward_returns_without_any_state_or_indexer_work(self):
        r = fx.Runner(fx.fixture([], extra=0), fx.cpu_lightning, enabled=True)
        before = r.state()
        r.batch.forward_mode = NS(is_idle=lambda: True)
        out = r.obj.forward_npu(
            torch.empty(0, 128), None, torch.empty(0, dtype=torch.int64), r.batch, 0
        )
        self.assertEqual(out.shape, (0, 1, 19))
        self.assertEqual(out.dtype, torch.int32)
        self.assertEqual(r.calls, [])
        for k in before:
            fx.exact(before[k], r.state()[k], k)

    def test_one_length4_sequence_is_not_equivalent(self):
        _, b = self.pair(fx.fixture([8], kind="closing"))
        kwargs = dict(b.calls[0])
        kwargs["actual_seq_lengths_query"] = torch.tensor([4], dtype=torch.int32)
        kwargs["actual_seq_lengths_key"] = kwargs["actual_seq_lengths_key"][-1:]
        kwargs["block_table"] = kwargs["block_table"][:1]
        wrong, _ = fx.cpu_lightning(**kwargs)
        self.assertFalse(torch.equal(wrong, b.pooled[0]))

    def test_final_length_broadcast_reads_future_closing_pool(self):
        _, b = self.pair(fx.fixture([8], kind="closing"))
        kwargs = dict(b.calls[0])
        kwargs["actual_seq_lengths_key"] = kwargs["actual_seq_lengths_key"][-1:].repeat(
            4
        )
        wrong, _ = fx.cpu_lightning(**kwargs)
        self.assertEqual(int(wrong[0, 0, 0]), 2)
        self.assertNotIn(2, b.pooled[0][0, 0].tolist())

    def test_shape_sensitive_ties_can_defeat_exact_without_changing_scores(self):
        def legal_but_shape_sensitive_ties(**kwargs):
            result, _ = fx.cpu_lightning(**kwargs)
            if len(kwargs["actual_seq_lengths_query"]) > 1:
                # All scores are zero in this fixture. A different tie policy
                # is allowed by a contract specifying only the top-k values.
                for row, length in enumerate(kwargs["actual_seq_lengths_key"].tolist()):
                    count = min(length, kwargs["sparse_count"])
                    result[row, 0, :count] = torch.arange(
                        length - 1, length - count - 1, -1, dtype=torch.int32
                    )
            return result, None

        case = fx.fixture([32], kind="ties")
        a, b = [
            fx.Runner(case, legal_but_shape_sensitive_ties, enabled=e)
            for e in (False, True)
        ]
        x, y = a.run(), b.run()
        with self.assertRaisesRegex(AssertionError, "not bitwise exact"):
            fx.compare(a, x, b, y)
        for sa, sb in zip(a.states, b.states):
            for k in sa:
                fx.exact(sa[k], sb[k], "state is still identical")


if __name__ == "__main__":
    unittest.main()
