"""CPU/interpreter checks for the explicit D verify expand-batch candidate."""

from copy import deepcopy
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "test/manual/ascend"))
import glm53_index_batch_fixture as fx
from glm53_index_batch_contract_v2 import compare_snapshots, snapshot


class VerifyExpandBatchCPU(unittest.TestCase):
    def setUp(self):
        env = patch.dict(
            os.environ, {"TRITON_INTERPRET": "1", "SGLANG_GLM53_KPOOL_EXPAND": "1"}
        )
        env.start()
        self.addCleanup(env.stop)
        torch.set_num_threads(1)

    def test_default_off_requires_decode_and_batch_li(self):
        ns = {}
        cls = fx.load_indexer(ns)

        class Init(cls, torch.nn.Module):
            pass

        for mode in ("decode", "prefill", "null"):
            for li in ("0", "1"):
                for expand in (None, "0", "1", "true"):
                    with self.subTest(mode=mode, li=li, expand=expand):
                        ns.update(
                            get_server_args=lambda: NS(
                                disaggregation_mode=mode, max_running_requests=2
                            ),
                            get_device=lambda: NS(device="cpu"),
                            get_spec=lambda: NS(speculative_num_draft_tokens=None),
                        )
                        obj = Init()
                        obj.index_kpool, obj.head_dim = 4, 128
                        with patch.dict(os.environ, {fx.FLAG: li}):
                            os.environ.pop(fx.EXPAND_FLAG, None)
                            if expand is not None:
                                os.environ[fx.EXPAND_FLAG] = expand
                            obj._init_ascend_kpool()
                        self.assertEqual(
                            obj._batch_kpool_verify_expand,
                            mode == "decode" and li == "1" and expand == "1",
                        )

    def test_fixture_opt_in_and_production_eligibility_still_gate_dispatch(self):
        case = fx.fixture([252, 253])
        with patch.dict(os.environ, {fx.EXPAND_FLAG: "1"}):
            default = fx.Runner(case, fx.cpu_lightning, enabled=True)
            default.run()
            self.assertEqual([s[0] for s in default.expands], [(2, 4)] * 4)
        for enabled, bypass in ((False, True), (True, False)):
            r = fx.Runner(
                case,
                fx.cpu_lightning,
                enabled=enabled,
                batch_expand=True,
                bypass_cpu_guard=bypass,
            )
            r.run()
            self.assertEqual(len(r.calls), 4)
            self.assertEqual([s[0] for s in r.expands], [(2, 4)] * 4)
        # Even if the flag is set, the existing >16-request guard wins.
        r = fx.Runner(
            fx.fixture([3] * 17), fx.cpu_lightning, enabled=True, batch_expand=True
        )
        r.run()
        self.assertEqual(len(r.calls), 4)
        self.assertEqual([s[0] for s in r.expands], [(17, 4)] * 4)

    def test_52_56_row_integer_kernel_oracle_and_global_guard(self):
        ns = {"expand_with_tail": Mock(side_effect=AssertionError("global guard"))}
        obj = fx.load_indexer(ns)()
        obj.index_kpool = 4
        expand = fx.kernel("kpool_expand_npu", True).expand_with_tail
        for rows in (52, 56):
            for dtype in (torch.int32, torch.int64):
                # Strides exercise the existing kernel's row/column addressing.
                ids = torch.arange(rows * 1024, dtype=dtype).remainder(32768)
                ids = ids.reshape(rows, 1024)[:, ::2]
                ids[:, 2::3] = -1
                positions = torch.tensor(
                    ([0, 1, 2, 3, 252, 2044, 131068] * rows)[:rows], dtype=dtype
                )
                expected = []
                for pools, p in zip(ids.tolist(), positions.tolist()):
                    seq = p + 1
                    values = [
                        4 * pool + j if pool >= 0 else -1
                        for pool in pools
                        for j in range(4)
                    ]
                    values += [-1] * 3
                    tail = seq % 4
                    at = min(seq - tail, 2048)
                    values[at : at + 3] = [
                        seq - tail + j if j < tail else -1 for j in range(3)
                    ]
                    expected.append([v if v >= 0 else seq for v in values])
                expected = torch.tensor(expected, dtype=dtype)
                fx.exact(expand(ids, positions), expected, "integer row oracle")
                fx.exact(
                    obj._expand_with_tail(ids, positions),
                    expected,
                    "global >16 fallback",
                )
        ns["expand_with_tail"].assert_not_called()

    def test_production_batch_expand_exact_states_and_request_major_rows(self):
        for bs in (1, 12, 13, 14):
            for fused in (False, True):
                with self.subTest(bs=bs, fused=fused):
                    case = fx.fixture(
                        [0 if row == 0 else 252 + row % 4 for row in range(bs)],
                        topk=2048,
                        heads=2,
                        extra=3,
                    )
                    a, b = [
                        fx.Runner(
                            case,
                            fx.cpu_lightning,
                            enabled=True,
                            batch_expand=e,
                            fused=fused,
                        )
                        for e in (False, True)
                    ]
                    x, y = a.run(), b.run()
                    compare_snapshots(
                        case,
                        snapshot(a, x),
                        snapshot(b, y),
                        canonical_a=True,
                        batch_expand_b=True,
                        diagnostics={},
                    )
                    fx.exact(x, y, "all expanded slots, including canonical padding")
                    fx.exact(a.raw_topk(), b.raw_topk(), "all raw slots")
                    self.assertEqual((len(a.calls), len(b.calls)), (1, 1))
                    self.assertEqual(b.events, ["update"] * 4 + ["indexer", "expand"])
                    self.assertEqual(b.expands[0][0], (bs * 4, 512))
                    fx.exact(
                        b.expands[0][1], case.data["positions"][: bs * 4], "row order"
                    )
                    self.assertEqual(
                        b.expands[0][1].data_ptr(), b.data["positions"].data_ptr()
                    )
                    self.assertEqual(tuple(y.shape), (bs * 4 + 3, 2051))

    def test_v2_explicit_launch_contract_preserves_zero_and_state_checks(self):
        serial = 0

        def dirty_li(**kwargs):
            nonlocal serial
            serial += 1
            ids, _ = fx.cpu_lightning(**kwargs)
            ids[kwargs["actual_seq_lengths_key"] == 0] = serial * 97
            return ids, None

        case = fx.fixture([0, 1, 2, 3, 252], requests=[1, 2, 3, 4, 0])
        a = fx.Runner(case, dirty_li, enabled=False, fused=True)
        b = fx.Runner(case, dirty_li, enabled=True, batch_expand=True, fused=True)
        x, y = a.run(), b.run()
        aa, bb = snapshot(a, x), snapshot(b, y)
        diag = {}
        compare_snapshots(case, aa, bb, batch_expand_b=True, diagnostics=diag)
        self.assertGreater(diag["dead_raw"]["bytes"], 0)
        self.assertGreater(diag["active_zero_raw"]["bytes"], 0)
        with self.assertRaisesRegex(AssertionError, "expand shape/order"):
            compare_snapshots(case, aa, bb, diagnostics={})  # Default stays strict.
        with self.assertRaisesRegex(AssertionError, "not bitwise exact"):
            fx.compare(a, x, b, y)  # v1 still fails on old zero-key garbage.
        for runner, original, canonical in ((a, aa, False), (b, bb, True)):
            runner.reset()
            compare_snapshots(
                case,
                original,
                snapshot(runner, runner.run()),
                canonical_a=canonical,
                canonical_b=canonical,
                batch_expand_a=canonical,
                batch_expand_b=canonical,
                diagnostics={},
            )  # Exercise both comparator roles; this is not an NPU graph test.
        for label, mutate in (
            ("zero-key canonical", lambda s: s["raw"][0, 0].fill_(2)),
            (
                "all bytes",
                lambda s: s["final"]["cache"]
                .view(torch.uint8)
                .flatten()[0]
                .bitwise_xor_(1),
            ),
            ("expand shape/order", lambda s: s.update(expands=[(5, 4)] * 4)),
        ):
            broken = deepcopy(bb)
            mutate(broken)
            with self.assertRaisesRegex(AssertionError, label):
                compare_snapshots(case, aa, broken, batch_expand_b=True, diagnostics={})


if __name__ == "__main__":
    unittest.main()
