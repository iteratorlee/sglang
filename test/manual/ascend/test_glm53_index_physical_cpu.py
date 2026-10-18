"""Small CPU fixture/dispatch/oracle checks; never certifies CANN or NPU replay."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
os.environ["TRITON_INTERPRET"] = "1"
os.environ["SGLANG_GLM53_KPOOL_EXPAND"] = "1"

import unittest
from types import SimpleNamespace as NS

import torch

from glm53_index_batch_contract_v2 import snapshot
from glm53_index_batch_fixture import Runner, cpu_lightning, exact, fixture
from glm53_index_physical_fixture import PhysicalLayout, compact_cases, physical_fixture
from test_glm53_index_physical_npu import MODES, compare_three, inspect_calls


class PhysicalFixtureTests(unittest.TestCase):
    def test_production_layout_and_bounded_matrix_without_allocation(self):
        layout = PhysicalLayout()
        self.assertEqual(layout.shapes(13)["key"], [29049, 64, 1, 128])
        self.assertEqual(layout.shapes(13)["full_block_table"], [13, 2076])
        self.assertEqual(layout.shapes(13)["legacy_li_table"], [13, 519])
        self.assertEqual(layout.shapes(13)["batched_li_table"], [52, 519])
        self.assertEqual(layout.shapes(13)["key_bytes"], 475938816)
        self.assertEqual(
            PhysicalLayout(full_columns=2074).validate(13, [131068] * 13), 514
        )
        specs = compact_cases()
        self.assertEqual(len(specs), 20)
        self.assertEqual({len(s["starts"]) for s in specs}, {1, 2, 4, 8, 13})
        required = {127, 252, 2044, 2048, 2052}
        self.assertTrue(
            required <= {s["starts"][0] for s in specs if len(s["starts"]) == 1}
        )
        for spec in specs:
            layout.validate(len(spec["starts"]), spec["starts"])

    def test_invalid_capacities_fail_before_allocating(self):
        for layout, starts in (
            (PhysicalLayout(2, 4), [252]),
            (PhysicalLayout(30, 1), [252]),
            (PhysicalLayout(30, 4, 12), [0]),
            (PhysicalLayout(), [-1]),
        ):
            with self.subTest(layout=layout, starts=starts), self.assertRaises(
                ValueError
            ):
                physical_fixture(starts, layout=layout)

    def test_remapping_preserves_every_logical_block_and_future_poison(self):
        for kind in ("random", "ties", "near-ties", "closing"):
            kwargs = dict(
                requests=[2, 1], kind=kind, topk=16, heads=2, extra=0, seed=54
            )
            old = fixture([251, 253], **kwargs)
            new = physical_fixture(
                [251, 253], layout=PhysicalLayout(41, 7, 26), **kwargs
            )
            self.assertEqual(new.data["table"].shape, (2, 26))
            self.assertEqual(new.data["table"][:, ::4].shape, (2, 7))
            self.assertEqual(int(new.data["table"][0, 0]), 40)
            referenced = {0}
            for row in range(2):
                for col in range(0, old.data["table"].shape[1], 4):
                    a, b = int(old.data["table"][row, col]), int(
                        new.data["table"][row, col]
                    )
                    exact(old.data["cache"][a], new.data["cache"][b], kind)
                    referenced.add(b)
            exact(old.data["cache"][0], new.data["cache"][0], "reserved sink")
            unused = next(i for i in range(41) if i not in referenced)
            exact(
                new.data["cache"][unused],
                torch.full_like(new.data["cache"][unused], 32),
                "unused poison",
            )

    def run_three(self, case):
        runners, snapshots = [], []
        for mode in MODES:
            runner = Runner(
                case, cpu_lightning, enabled=mode[1], batch_expand=mode[2], fused=True
            )
            out = runner.run()
            inspect_calls(runner, case, mode, None)
            runners.append(runner)
            snapshots.append(snapshot(runner, out))
        compare_three(case, snapshots, {})
        return runners, snapshots

    def test_three_way_real_methods_small_cpu_and_dynamic_slot_reset(self):
        layout = PhysicalLayout(31, 5)
        a = physical_fixture([7, 11], layout=layout, heads=2, topk=16, requests=[2, 1])
        runners, snaps = self.run_three(a)
        pointers = [[x.data_ptr() for x in r.data.values()] for r in runners]
        b = physical_fixture(
            [8, 12], layout=layout, heads=2, topk=16, requests=[1, 2], seed=59
        )
        self.assertFalse(torch.equal(a.data["table"], b.data["table"]))
        next_snaps = []
        for mode, runner in zip(MODES, runners):
            runner.reset(b)
            out = runner.run()
            inspect_calls(runner, b, mode, None)
            next_snaps.append(snapshot(runner, out))
        compare_three(b, next_snaps, {})
        self.assertEqual(
            pointers, [[x.data_ptr() for x in r.data.values()] for r in runners]
        )

    def test_zero_key_and_padding_keep_existing_contract(self):
        case = physical_fixture(
            [0, 252],
            layout=PhysicalLayout(31, 5),
            heads=2,
            topk=16,
            requests=[1, 0],
            extra=2,
        )
        self.run_three(case)

    def test_full_unused_physical_state_and_live_topk_faults_rejected(self):
        case = physical_fixture([7], layout=PhysicalLayout(11, 4), heads=2, topk=16)
        runners, snaps = self.run_three(case)
        cache = snaps[2]["states"][3]["cache"]
        unused = next(
            i
            for i in range(1, 11)
            if i not in set(case.data["table"].flatten().tolist())
        )
        cache[unused, -1, 0, -1] += 1
        with self.assertRaisesRegex(AssertionError, "snapshot 3 cache"):
            compare_three(case, snaps, {})
        cache[unused, -1, 0, -1] -= 1
        raw = snaps[1]["raw"]
        raw[0, 0, 0] += 1
        with self.assertRaisesRegex(AssertionError, "positive-key raw"):
            compare_three(case, snaps, {})
        # Independently reject a silently narrow CANN table argument.
        runners[0].calls[0]["block_table"] = runners[0].calls[0]["block_table"][:, :-1]
        with self.assertRaisesRegex(AssertionError, "physical table width"):
            inspect_calls(runners[0], case, MODES[0], None)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
