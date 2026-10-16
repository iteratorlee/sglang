"""CPU tests of GLM shape-cache collective ordering, including remote misses.

Run with CUDA_VISIBLE_DEVICES='' TORCH_DEVICE_BACKEND_AUTOLOAD=0 and the
sgl-0511 Python. HCCL is mocked; all shape/cache logic is production code.
"""

import os
import unittest
from unittest import mock

import torch

import test_glm53_normal_idle_dp_cpu as idle
from test_glm53_normal_hccl_cpu import dynamic_quant_reference

FLAG = "SGLANG_GLM53_NORMAL_SHAPE_CACHE"


class ShapeCacheCpuTests(unittest.TestCase):
    def setUp(self):
        self.fixture = idle.IdleDpCpuTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.module = self.fixture.module
        self.group = self.fixture.group
        mock.patch.dict(os.environ, {FLAG: "1"}).start()
        self.calls = []
        self.remote_dirty = False
        self.remote_shape = None

        def vote(value, op, group):
            self.assertIs(op, self.module.dist.ReduceOp.MAX)
            self.assertEqual(value.dtype, torch.int32)
            self.assertEqual(value.shape, (1,))
            self.calls.append(("vote", group, value.item()))
            value.fill_(max(value.item(), int(self.remote_dirty)))

        def gather(out, value, group):
            self.calls.append(("audit", group))
            out.copy_(value.repeat(out.numel() // value.numel()))
            if self.remote_shape is not None:
                out.view(-1, 8)[8].copy_(torch.tensor(self.remote_shape))

        mock.patch.object(self.module.dist, "all_reduce", side_effect=vote).start()
        mock.patch.object(
            self.module.dist, "all_gather_into_tensor", side_effect=gather
        ).start()

    def check(self, tokens=3, width=7, topk=8, *, group=None, world=16):
        self.module._check_equal_dispatch_shapes_cached(
            self.group if group is None else group,
            torch.empty(tokens, width, dtype=torch.bfloat16),
            torch.empty(tokens, topk, dtype=torch.int64),
            torch.empty(tokens, topk, dtype=torch.float32),
            world,
        )

    def test_hits_changes_zero_rows_and_returning_geometry(self):
        with self.module.normal_hccl_idle_dp_scope(idle.batch()):
            for geometry in (
                (3, 7, 8),
                (3, 7, 8),
                (0, 7, 8),
                (0, 7, 8),
                (5, 7, 8),
                (5, 9, 8),
                (5, 9, 4),
                (3, 7, 8),
            ):
                self.check(*geometry)
        self.assertEqual(
            [call[2] for call in self.calls if call[0] == "vote"],
            [1, 0, 1, 0, 1, 1, 1, 1],
        )
        self.assertEqual(sum(call[0] == "audit" for call in self.calls), 6)
        self.assertEqual(len(self.module._dispatch_shape_cache), 1)

    def test_remote_only_change_forces_audit_and_failure_invalidates(self):
        with self.module.normal_hccl_idle_dp_scope(idle.batch()):
            self.check()
            self.remote_dirty = True
            self.remote_shape = [2, 7, 2, 2, 8, 2, 2, 8]
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError, "EP16 requires equal"):
                    self.check()
                self.assertNotIn(self.group, self.module._dispatch_shape_cache)
            self.remote_shape = None
            self.remote_dirty = False
            self.check()
            self.check()
        self.assertEqual(
            [call[0] for call in self.calls],
            [
                "vote",
                "audit",
                "vote",
                "audit",
                "vote",
                "audit",
                "vote",
                "audit",
                "vote",
            ],
        )
        self.assertEqual(
            [call[2] for call in self.calls if call[0] == "vote"], [1, 0, 1, 1, 0]
        )

    def test_peer_cache_loss_reaudits_even_when_local_geometry_is_equal(self):
        with self.module.normal_hccl_idle_dp_scope(idle.batch()):
            self.check()
            self.remote_dirty = True
            self.check()
            self.remote_dirty = False
            self.check()
        self.assertEqual(
            [call[0] for call in self.calls], ["vote", "audit", "vote", "audit", "vote"]
        )

    def test_groups_have_independent_cache_entries(self):
        other = object()
        with self.module.normal_hccl_idle_dp_scope(idle.batch()):
            self.check()
            self.check(group=other)
            self.check()
        self.assertEqual(
            [call[2] for call in self.calls if call[0] == "vote"], [1, 1, 0]
        )

    def test_scoping_default_off_and_disable_reenable(self):
        self.check()  # no scope
        for value in (None, "0", "true", "", "2"):
            with mock.patch.dict(os.environ):
                if value is None:
                    os.environ.pop(FLAG, None)
                else:
                    os.environ[FLAG] = value
                with self.module.normal_hccl_idle_dp_scope(idle.batch()):
                    self.check()
        with self.module.normal_hccl_idle_dp_scope(idle.batch()):
            self.check(world=8)
        self.assertTrue(all(call[0] == "audit" for call in self.calls))
        self.calls.clear()
        with self.module.normal_hccl_idle_dp_scope(idle.batch()):
            self.check()
            with mock.patch.dict(os.environ, {FLAG: "0"}):
                self.check()
            self.check()
        self.assertEqual(
            [call[0] for call in self.calls],
            ["vote", "audit", "audit", "vote", "audit"],
        )

    def test_mismatched_ids_weights_are_not_cached(self):
        with self.module.normal_hccl_idle_dp_scope(idle.batch()):
            with self.assertRaisesRegex(RuntimeError, "equal local rows"):
                self.module._check_equal_dispatch_shapes_cached(
                    self.group,
                    torch.empty(3, 7),
                    torch.empty(2, 8),
                    torch.empty(3, 8),
                    16,
                )
        self.assertNotIn(self.group, self.module._dispatch_shape_cache)

    def test_dispatch_keeps_four_payloads_on_hits_and_reaudits_changes(self):
        gathers = []

        def gather(out, value, group):
            gathers.append(
                "shape" if value.ndim == 1 and value.dtype == torch.int64 else "payload"
            )
            out.copy_(value.repeat(16, *([1] * (value.ndim - 1))))

        with (
            mock.patch.object(self.module.dist, "get_world_size", return_value=16),
            mock.patch.object(self.module.dist, "get_rank", return_value=0),
            mock.patch.object(self.module.dist, "all_gather_into_tensor", gather),
            mock.patch.object(
                torch.ops.npu,
                "npu_dynamic_quant",
                side_effect=dynamic_quant_reference,
                create=True,
            ),
        ):
            for tokens in (3, 3, 0, 1):
                with self.module.normal_hccl_idle_dp_scope(idle.batch((tokens * 8, 0))):
                    self.module.collective_dispatch(
                        self.group,
                        torch.ones(tokens, 7, dtype=torch.bfloat16),
                        torch.zeros(tokens, 8, dtype=torch.int64),
                        torch.ones(tokens, 8),
                    )
        self.assertEqual(
            gathers,
            ["shape"]
            + ["payload"] * 8
            + ["shape"]
            + ["payload"] * 4
            + ["shape"]
            + ["payload"] * 4,
        )
        self.assertEqual([call[2] for call in self.calls], [1, 0, 1, 1])


if __name__ == "__main__":
    unittest.main()
