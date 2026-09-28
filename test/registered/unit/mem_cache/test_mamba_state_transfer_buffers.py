import unittest
from unittest import mock

import torch

from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

NUM_LAYERS = 2
NUM_SLOTS = 3


def _pool(temporal: torch.Tensor, num_conv: int = 2) -> MambaPool:
    """A MambaPool stub carrying only what the transfer accessors read."""
    pool = object.__new__(MambaPool)
    pool.num_mamba_layers = NUM_LAYERS
    pool.mamba_layer_ids = list(range(NUM_LAYERS))
    pool._slot_siblings = []
    pool.conv_slice_axis = 0
    pool.mamba_cache = MambaPool.State(
        conv=[torch.zeros(NUM_LAYERS, NUM_SLOTS, 4, 5) for _ in range(num_conv)],
        temporal=temporal,
    )
    return pool


class TestMambaStateTransferBuffers(unittest.TestCase):
    def test_npu_clear_slots_matches_advanced_assignment_for_strided_state(self):
        temporal_base = torch.arange(
            NUM_LAYERS * 6 * NUM_SLOTS * 4, dtype=torch.float32
        ).reshape(NUM_LAYERS, 6, NUM_SLOTS, 4)
        temporal = temporal_base.transpose(1, 2)
        pool = _pool(temporal, num_conv=2)
        for i, conv in enumerate(pool.mamba_cache.conv):
            conv.fill_(i + 1)
        expected_conv = [tensor.clone() for tensor in pool.mamba_cache.conv]
        expected_temporal = temporal.clone()
        indices = torch.tensor([0, 2], dtype=torch.int64)
        for tensor in expected_conv:
            tensor[:, indices] = 0
        expected_temporal[:, indices] = 0

        with mock.patch("sglang.srt.mem_cache.memory_pool._is_npu", True):
            pool.clear_slots(indices)

        for actual, expected in zip(pool.mamba_cache.conv, expected_conv):
            self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(pool.mamba_cache.temporal, expected_temporal))

    def test_npu_copy_slots_preserves_snapshot_semantics_for_strided_state(self):
        for src_slots, dst_slots in (([0], [2]), ([0, 1], [1, 2])):
            with self.subTest(src=src_slots, dst=dst_slots):
                temporal = torch.arange(
                    NUM_LAYERS * 6 * NUM_SLOTS * 4, dtype=torch.float32
                ).reshape(NUM_LAYERS, 6, NUM_SLOTS, 4).transpose(1, 2)
                pool = _pool(temporal)
                pool.replayssm_write_pos = None
                pool.debug_memory_pool = False
                for i, conv in enumerate(pool.mamba_cache.conv):
                    conv.copy_(
                        torch.arange(conv.numel(), dtype=conv.dtype).reshape(conv.shape)
                        + i * conv.numel()
                    )
                expected = [
                    tensor.clone()
                    for tensor in (*pool.mamba_cache.conv, pool.mamba_cache.temporal)
                ]
                sources = torch.tensor(src_slots)
                destinations = torch.tensor(dst_slots)
                for tensor in expected:
                    tensor[:, destinations] = tensor[:, sources].clone()

                with mock.patch("sglang.srt.mem_cache.memory_pool._is_npu", True):
                    pool.copy_from(sources, destinations)

                for actual, reference in zip(
                    (*pool.mamba_cache.conv, pool.mamba_cache.temporal), expected
                ):
                    self.assertTrue(torch.equal(actual, reference))

    def test_conv_only_state_advertises_no_empty_buffer(self):
        """A ShortConv layer declares a degenerate temporal shape, so the pool
        allocates an empty tensor for it. The RDMA engine rejects a zero-length
        region and fails the batch registration that carries the real buffers,
        so an empty buffer must never be advertised."""
        pool = _pool(torch.zeros(NUM_LAYERS, NUM_SLOTS, 0, 0, 0))

        _, lens, item_lens = pool.get_contiguous_buf_infos()

        self.assertNotIn(0, lens)
        self.assertNotIn(0, item_lens)
        self.assertEqual(len(lens), 2 * NUM_LAYERS)

    def test_temporal_state_is_still_advertised(self):
        pool = _pool(torch.zeros(NUM_LAYERS, NUM_SLOTS, 6, 7, 8))

        _, lens, _ = pool.get_contiguous_buf_infos()

        self.assertNotIn(0, lens)
        self.assertEqual(len(lens), 3 * NUM_LAYERS)

    def test_dims_stay_aligned_with_buffers(self):
        """The per-tensor lists are parallel-indexed, so dropping a buffer has to
        drop its dim too."""
        pool = _pool(torch.zeros(NUM_LAYERS, NUM_SLOTS, 0, 0, 0))

        _, lens, _ = pool.get_contiguous_buf_infos()

        self.assertEqual(len(pool.get_state_dim_per_tensor()), len(lens))

    def test_sibling_declares_replicated_transfer_without_field_name_coupling(self):
        pool = _pool(torch.zeros(NUM_LAYERS, NUM_SLOTS, 6, 7, 8))

        class ReplicatedSibling:
            def iter_transfer_state_entries(self):
                yield "future_sibling", torch.zeros(NUM_SLOTS, 9), None, 123

        pool._slot_siblings = [ReplicatedSibling()]

        self.assertEqual(pool.get_state_dim_per_tensor()[-1], 0)
        self.assertEqual(pool.get_state_slice_outer_counts()[-1], 1)


if __name__ == "__main__":
    unittest.main()
