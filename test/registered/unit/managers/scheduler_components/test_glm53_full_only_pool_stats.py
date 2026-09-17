"""CPU regression for GLM53 PD's FULL-only tree on a hybrid SSM model.

Exercise native tree insertion, PREBUILT, release, paged allocation, hybrid
request allocation, pool observation and the strict conservation checker.
No model forward, device runtime, or PD transport is required.
"""

import unittest
from array import array
from dataclasses import replace
from types import SimpleNamespace

import torch

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.disaggregation.decode_schedule_batch_mixin import (
    ScheduleBatchDisaggregationDecodeMixin,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler_components.invariant_checker import (
    SchedulerInvariantChecker,
)
from sglang.srt.managers.scheduler_components.pool_stats_observer import (
    SchedulerPoolStatsObserver,
)
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import get_spec
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestGLM53FullOnlyPoolStats(unittest.TestCase):
    TOTAL = 528896
    PROMPT = 8192
    PHYSICAL_PAGE = 64
    TREE_PAGE = 256

    def setUp(self):
        args = ServerArgs(model_path="dummy", device="cpu", page_size=64)
        args._mamba_cache_chunk_size = 64
        set_global_server_args_for_scheduler(args)
        self.enterContext(get_spec().override(speculative_algorithm="EAGLE"))
        self.enterContext(envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override("python"))

        # Small real recurrent tensors suffice: this test covers ownership and
        # accounting, not the GLM53 KDA math or Ascend transfer payload.
        shape = Mamba2StateShape.create(
            tp_world_size=1,
            intermediate_size=16,
            n_groups=1,
            num_heads=2,
            head_dim=8,
            state_size=8,
            conv_kernel=4,
        )
        self.req_pool = HybridReqToTokenPool(
            size=2,
            mamba_size=8,
            mamba_spec_state_size=2,
            max_context_len=self.PROMPT + 256,
            device="cpu",
            enable_memory_saver=False,
            cache_params=Mamba2CacheParams(shape=shape, layers=[1]),
            mamba_layer_ids=[1],
            enable_mamba_extra_buffer=True,
            speculative_num_draft_tokens=4,
            speculative_eagle_topk=1,
        )
        kv_pool = HybridLinearKVPool(
            size=self.TOTAL,
            dtype=torch.bfloat16,
            page_size=self.PHYSICAL_PAGE,
            head_num=1,
            head_dim=1,
            full_attention_layer_ids=[0],
            device="cpu",
            enable_memory_saver=False,
            mamba_pool=self.req_pool.mamba_pool,
        )
        self.allocator = PagedTokenToKVPoolAllocator(
            size=self.TOTAL,
            page_size=self.PHYSICAL_PAGE,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=kv_pool,
            need_sort=False,
        )
        self.tree = UnifiedRadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=self.req_pool,
                token_to_kv_pool_allocator=self.allocator,
                page_size=self.TREE_PAGE,
                is_eagle=True,
                tree_components=(ComponentType.FULL,),
                enable_mamba_extra_buffer=True,
            )
        )
        self.tree.glm53_kpool_share_page_size = self.TREE_PAGE
        self.observer = SchedulerPoolStatsObserver(
            tree_cache=self.tree,
            token_to_kv_pool_allocator=self.allocator,
            req_to_token_pool=self.req_pool,
            session_controller=SimpleNamespace(sessions={}),
            hisparse_coordinator=None,
            is_hybrid_swa=False,
            is_hybrid_ssm=True,
            enable_hisparse=False,
            full_tokens_per_layer=None,
            swa_tokens_per_layer=None,
            max_total_num_tokens=self.TOTAL,
            get_last_batch=lambda: None,
            get_running_batch=lambda: None,
        )
        self.checker = SchedulerInvariantChecker(
            is_hybrid_swa=False,
            is_hybrid_ssm=True,
            disaggregation_mode=DisaggregationMode.DECODE,
            page_size=self.PHYSICAL_PAGE,
            full_tokens_per_layer=None,
            swa_tokens_per_layer=None,
            max_total_num_tokens=self.TOTAL,
            tree_cache=self.tree,
            token_to_kv_pool_allocator=self.allocator,
            req_to_token_pool=self.req_pool,
            pool_stats_observer=self.observer,
            get_last_batch=lambda: None,
            get_running_batch=lambda: None,
            scheduler_stage_metrics=None,
        )

    def _finish_short_decode(self):
        req = Req(
            rid="glm53-cold-8192",
            origin_input_text="",
            origin_input_ids=array("q", range(self.PROMPT)),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=64),
        )
        self.assertIsNotNone(self.req_pool.alloc([req]))
        indices = self.allocator.alloc(self.PROMPT + self.PHYSICAL_PAGE)
        self.assertIsNotNone(indices)
        self.req_pool.write((req.kv.req_pool_idx, slice(0, len(indices))), indices)
        req.kv.kv_allocated_len = len(indices)
        req.kv.kv_committed_len = self.PROMPT
        req.last_node = self.tree.root_node_handle()
        req.output_ids = array("q", [197])
        req._refresh_fill_ids()
        req.set_extend_range(0, self.PROMPT)

        # PREBUILT has no forward pass; only its draft-relay collaborator is
        # inert. Tree/allocator/request lifecycle methods are all production.
        batch = SimpleNamespace(
            reqs=[req],
            tree_cache=self.tree,
            device="cpu",
            spec_algorithm=SimpleNamespace(
                build_disagg_draft_input=lambda *unused: object()
            ),
        )
        ScheduleBatchDisaggregationDecodeMixin.process_prebuilt(batch, None)
        self.assertEqual(self.tree.protected_size(), self.PROMPT - self.TREE_PAGE)
        self.assertEqual(req.kv.cache_protected_len, self.PROMPT - self.TREE_PAGE)

        req.output_ids.extend([197] * 63)
        req.kv.kv_committed_len = self.PROMPT + 63
        release_kv_cache(req, self.tree)

        self.assertFalse(req.kv.holds_kv)
        self.assertFalse(req.kv.holds_mamba)
        self.assertEqual(self.req_pool.mamba_allocator.available_size(), 8)
        self.assertEqual(self.tree.protected_size(), 0)
        self.assertEqual(self.tree.evictable_size(), self.PROMPT)
        self.assertEqual(self.allocator.available_size(), 520704)

    def test_finished_full_only_prefix_is_accounted_for_hybrid_model(self):
        self.assertFalse(self.tree.supports_mamba())
        self.assertTrue(self.tree.is_tree_cache())
        self._finish_short_decode()

        stats = self.observer.get_pool_stats()
        self.assertTrue(stats.is_hybrid_ssm)
        self.assertEqual(stats.full_available_size, 520704)
        self.assertEqual(stats.full_evictable_size, 8192)
        self.assertEqual(stats.full_num_used, 0)
        self.assertEqual(stats.mamba_num_used, 0)
        self.assertEqual(stats.mamba_evictable_size, 0)
        leak, message = self.checker._check_full_pool(stats)
        self.assertFalse(leak, message)

        # Preserve the observed failure signature without modifying production
        # code: the old observer discarded exactly this real tree-owned prefix.
        old_stats = replace(stats, full_evictable_size=0)
        old_leak, old_message = self.checker._check_full_pool(old_stats)
        self.assertTrue(old_leak, old_message)
        self.assertIn("total=528896", old_message)
        self.assertIn("available=520704", old_message)
        self.assertIn("evictable=0", old_message)

        self.tree.evict(EvictParams(num_tokens=8192))
        self.assertEqual(self.allocator.available_size(), self.TOTAL)
        leak, message = self.checker._check_full_pool(self.observer.get_pool_stats())
        self.assertFalse(leak, message)

    def test_genuine_unaccounted_physical_page_still_fails(self):
        self._finish_short_decode()
        orphan = self.allocator.alloc(self.PHYSICAL_PAGE)
        self.assertIsNotNone(orphan)
        try:
            stats = self.observer.get_pool_stats()
            self.assertEqual(stats.full_available_size, 520640)
            self.assertEqual(stats.full_evictable_size, 8192)
            self.assertEqual(stats.full_num_used, self.PHYSICAL_PAGE)
            leak, message = self.checker._check_full_pool(stats)
            self.assertTrue(leak, message)
            self.assertIn("available=520640", message)
            self.assertIn("evictable=8192", message)
        finally:
            self.allocator.free(orphan)
        leak, message = self.checker._check_full_pool(self.observer.get_pool_stats())
        self.assertFalse(leak, message)


if __name__ == "__main__":
    unittest.main()
