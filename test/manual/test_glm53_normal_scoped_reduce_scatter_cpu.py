"""CPU contracts for scoped EP16 reduce-scatter and its opt-in reference check.

Only mocks HCCL: FP32 local accumulation, slicing, casting, scoping and
failure voting run through the production module. No NPU/GPU is initialized.
"""

import os
import unittest
from unittest import mock

import torch

# Load before the fixture patches sys.modules. PyTorch 2.11 lazily imports
# this from use_deterministic_algorithms; removing/reimporting its registered
# operators between tests would otherwise fail with duplicate TORCH_LIBRARY.
import torch._inductor.config  # noqa: F401

import test_glm53_normal_idle_dp_cpu as idle

FLAG = "SGLANG_GLM53_NORMAL_REDUCE_SCATTER"
CHECK = "SGLANG_GLM53_NORMAL_REDUCE_SCATTER_CHECK"


class ScopedReduceScatterCpuTests(unittest.TestCase):
    def setUp(self):
        self.fixture = idle.IdleDpCpuTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.module = self.fixture.module
        self.group = self.fixture.group
        mock.patch.dict(os.environ, {FLAG: "2node", CHECK: "0"}).start()

    @staticmethod
    def partials(tokens, world, idle_counts):
        partials, values, states = [], [], []
        for rank in range(world):
            rows = []
            if tokens and rank not in (8, 15):
                for dst in range(world):
                    if world != 16 or idle_counts[dst // 8]:
                        rows.extend(
                            [dst * tokens, dst * tokens, (dst + 1) * tokens - 1]
                        )
            rows = torch.tensor(rows, dtype=torch.int64)
            y = (
                (torch.arange(max(1, len(rows)) * 3).reshape(-1, 3) % 13 - 6) * 16
                + rank
            ).to(torch.bfloat16)
            weights = torch.full((len(rows),), 0.125 + (rank + 1) * 2**-20)
            partial = torch.zeros(world * tokens, 3, dtype=torch.float32)
            for row, value, weight in zip(rows, y, weights):
                partial[row] += value.float() * weight
            if not len(rows):
                y.fill_(float("nan"))  # The dummy allocation must never contribute.
            partials.append(partial)
            values.append(y)
            states.append((rows, weights, tokens, world, 3, len(rows)))
        return partials, values, states

    def combine(self, rank, fixture, *, check=False, corrupt=False, peer_failed=False):
        partials, values, states = fixture
        state = states[rank]
        _, _, tokens, world, width, _ = state
        reference = sum(partials)
        calls = []
        deterministic_before = (
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
        )

        def assert_scope_restored():
            self.assertEqual(
                (
                    torch.are_deterministic_algorithms_enabled(),
                    torch.is_deterministic_algorithms_warn_only_enabled(),
                ),
                deterministic_before,
            )

        def reduce(out, group, op=None):
            self.assertIs(group, self.group)
            assert_scope_restored()
            if out.dtype == torch.int32:
                self.assertIs(op, self.module.dist.ReduceOp.MAX)
                self.assertEqual(out.shape, (1,))
                calls.append(("vote", out.item()))
                out.fill_(max(out.item(), int(peer_failed)))
            else:
                self.assertIsNone(op)
                self.assertEqual(out.dtype, torch.float32)
                torch.testing.assert_close(out, partials[rank], rtol=0, atol=0)
                calls.append(("all_reduce",))
                out.copy_(reference)

        def scatter(out, source, op, group):
            self.assertIs(group, self.group)
            self.assertIs(op, self.module.dist.ReduceOp.SUM)
            assert_scope_restored()
            self.assertEqual(out.dtype, torch.float32)
            self.assertEqual(source.dtype, torch.float32)
            self.assertEqual(out.shape, (tokens, width))
            self.assertTrue(out.is_contiguous() and source.is_contiguous())
            self.assertNotEqual(out.data_ptr(), source.data_ptr())
            torch.testing.assert_close(source, partials[rank], rtol=0, atol=0)
            calls.append(("reduce_scatter",))
            out.copy_(reference[rank * tokens : (rank + 1) * tokens])
            if corrupt:
                out.add_(1)

        with (
            mock.patch.dict(os.environ, {CHECK: "1" if check else "0"}),
            mock.patch.object(self.module.dist, "get_rank", return_value=rank),
            mock.patch.object(self.module.dist, "all_reduce", side_effect=reduce),
            mock.patch.object(
                self.module.dist, "reduce_scatter_tensor", side_effect=scatter
            ),
        ):
            result = self.module.collective_combine(self.group, values[rank], state)
        torch.testing.assert_close(
            result,
            reference[rank * tokens : (rank + 1) * tokens].to(torch.bfloat16),
            rtol=0,
            atol=0,
        )
        return calls

    def test_ep16_dynamic_shapes_empty_ranks_and_idle_dp_match_reference(self):
        for tokens in (1, 7, 0, 3):
            for counts in ((32, 32), (32, 0), (0, 32), (0, 0), (1, 32)):
                fixture = self.partials(tokens, 16, counts)
                with self.module.normal_hccl_idle_dp_scope(idle.batch(counts)):
                    for rank in range(16):
                        for check in (False, True):
                            with self.subTest(
                                tokens=tokens, counts=counts, rank=rank, check=check
                            ):
                                calls = self.combine(rank, fixture, check=check)
                                expected = (
                                    [("reduce_scatter",)]
                                    if tokens
                                    else [("all_reduce",)]
                                )
                                if check and tokens:
                                    expected += [("all_reduce",), ("vote", 0)]
                                self.assertEqual(calls, expected)

    def test_scope_and_default_keep_existing_combine_behavior(self):
        fixture = self.partials(2, 16, (32, 32))
        for mode in idle.Mode:
            if mode != idle.Mode.EXTEND:
                with self.module.normal_hccl_idle_dp_scope(idle.batch(mode=mode)):
                    self.assertEqual(
                        self.combine(0, fixture, check=True), [("all_reduce",)]
                    )
        for world in (1, 8):
            with self.module.normal_hccl_idle_dp_scope(idle.batch()):
                self.assertEqual(
                    self.combine(0, self.partials(2, world, (32, 32)), check=True),
                    [("all_reduce",)],
                )
        for value in (None, "0", "true", "2"):
            with mock.patch.dict(os.environ):
                if value is None:
                    os.environ.pop(FLAG, None)
                else:
                    os.environ[FLAG] = value
                with self.module.normal_hccl_idle_dp_scope(idle.batch()):
                    self.assertEqual(
                        self.combine(0, fixture, check=True), [("all_reduce",)]
                    )
        with mock.patch.dict(os.environ, {FLAG: "1"}):
            # Legacy mode remains available without the new scoped guard.
            self.assertEqual(
                self.combine(0, fixture, check=True), [("reduce_scatter",)]
            )

    def test_check_detects_local_or_remote_mismatch_after_both_collectives(self):
        fixture = self.partials(2, 16, (32, 0))
        with self.module.normal_hccl_idle_dp_scope(idle.batch()):
            for corrupt, peer_failed in ((True, False), (False, True)):
                with self.subTest(corrupt=corrupt, peer_failed=peer_failed):
                    with self.assertRaisesRegex(RuntimeError, "at least one EP rank"):
                        self.combine(
                            0,
                            fixture,
                            check=True,
                            corrupt=corrupt,
                            peer_failed=peer_failed,
                        )


if __name__ == "__main__":
    unittest.main()
