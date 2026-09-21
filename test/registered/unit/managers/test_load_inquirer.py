import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_policy import (
    CacheAgnosticPolicy,
    CacheAwarePolicy,
    SchedulePolicy,
)
from sglang.srt.managers.scheduler_components.load_inquirer import (
    SchedulerLoadInquirer,
)
from sglang.srt.managers.scheduler_components.pool_stats_observer import PoolStats
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestSchedulePolicyWaitingQueueMatching(unittest.TestCase):
    def make_policy(self, policy, supports_fast_match_prefix):
        schedule_policy = object.__new__(SchedulePolicy)
        schedule_policy.policy = policy
        schedule_policy.tree_cache = SimpleNamespace(
            supports_fast_match_prefix=lambda: supports_fast_match_prefix
        )
        return schedule_policy

    def test_cache_agnostic_policy_requires_fast_matching(self):
        policy = self.make_policy(CacheAgnosticPolicy.FCFS, False)
        self.assertFalse(policy.waiting_queue_prefix_matched([]))

        policy.tree_cache = SimpleNamespace(supports_fast_match_prefix=lambda: True)
        self.assertTrue(policy.waiting_queue_prefix_matched([]))

    def test_lpm_queue_limit_respects_fast_matching_capability(self):
        policy = self.make_policy(CacheAwarePolicy.LPM, False)
        self.assertTrue(policy.waiting_queue_prefix_matched([None] * 128))
        self.assertFalse(policy.waiting_queue_prefix_matched([None] * 129))

        policy.tree_cache = SimpleNamespace(supports_fast_match_prefix=lambda: True)
        self.assertTrue(policy.waiting_queue_prefix_matched([None] * 129))


class TestSchedulerLoadInquirer(unittest.TestCase):
    def make_inquirer(self, waiting_queue_prefix_matched):
        waiting_req = SimpleNamespace(seqlen=100, num_matched_prefix_tokens=20)
        chunked_req = SimpleNamespace(seqlen=50, prefix_indices=range(10))
        return SimpleNamespace(
            disaggregation_mode=DisaggregationMode.NULL,
            get_waiting_queue=lambda: [waiting_req],
            waiting_queue_prefix_matched=lambda: waiting_queue_prefix_matched,
            get_chunked_req=lambda: chunked_req,
            get_recent_cache_hit_rate=lambda: 0.75,
        )

    def test_waiting_tokens_are_estimated_when_prefix_matching_is_skipped(self):
        inquirer = self.make_inquirer(waiting_queue_prefix_matched=False)

        self.assertEqual(
            SchedulerLoadInquirer.get_num_waiting_uncached_tokens(inquirer),
            65,
        )

    def test_waiting_tokens_use_exact_match_when_prefix_matching_is_done(self):
        inquirer = self.make_inquirer(waiting_queue_prefix_matched=True)

        self.assertEqual(
            SchedulerLoadInquirer.get_num_waiting_uncached_tokens(inquirer),
            120,
        )

    def test_capacity_uses_observer_counters_without_free_space_derivation(self):
        """GLM53 keeps 256-token radix pages over a 64-token allocator.

        The observer has already reconciled those two views.  The load snapshot
        must preserve its logical-token counters instead of deriving free space
        from max_total_num_tokens - num_used_tokens.
        """
        pool_stats = PoolStats(
            full_num_used=64,
            full_token_usage=64 / 528896,
            full_available_size=520640,
            full_evictable_size=8192,
            is_hybrid_ssm=True,
            mamba_num_used=2,
            mamba_usage=0.25,
            mamba_available_size=6,
            mamba_evictable_size=0,
        )
        observer = SimpleNamespace(
            get_pool_stats=Mock(return_value=pool_stats),
            req_to_token_pool=SimpleNamespace(available_size=lambda: 1),
        )
        stats = SimpleNamespace(
            gen_throughput=10.0,
            cache_hit_rate=0.75,
            utilization=0.5,
            kv_transfer_speed_gb_s=0.0,
            kv_transfer_latency_ms=0.0,
            num_grammar_queue_reqs=0,
            num_paused_reqs=0,
            num_retracted_reqs=0,
        )
        inquirer = SchedulerLoadInquirer(
            disaggregation_mode=DisaggregationMode.NULL,
            ps=SimpleNamespace(dp_rank=3),
            server_args=SimpleNamespace(),
            max_total_num_tokens=528896,
            max_running_requests=128,
            pool_stats_observer=observer,
            tp_worker=SimpleNamespace(
                model_runner=SimpleNamespace(weight_load_mem_usage=1.0),
                graph_memory_usage={},
            ),
            token_to_kv_pool_allocator=SimpleNamespace(
                get_kvcache=lambda: SimpleNamespace(mem_usage=2.0)
            ),
            spec_algorithm=SimpleNamespace(is_none=lambda: True),
            get_running_batch=lambda: SimpleNamespace(reqs=[]),
            get_waiting_queue=lambda: [],
            waiting_queue_prefix_matched=lambda: True,
            get_recent_cache_hit_rate=lambda: 0.0,
            get_stats=lambda: stats,
            get_chunked_req=lambda: None,
            get_disagg_prefill_bootstrap_queue=lambda: SimpleNamespace(queue=[]),
            get_disagg_prefill_inflight_queue=lambda: [],
            get_disagg_decode_prealloc_queue=lambda: SimpleNamespace(
                queue=[], retracted_queue=[]
            ),
            get_disagg_decode_transfer_queue=lambda: SimpleNamespace(queue=[]),
            get_spec_total_num_accept_tokens=lambda: 0,
            get_spec_total_num_forward_ct=lambda: 0,
            get_total_prefill_uncached_tokens=lambda: 0,
            get_total_prefill_busy_us=lambda: 0,
            get_decode_moment_totals=lambda: (0, 0, 0, 0, 0, 0),
        )

        with (
            patch(
                "sglang.srt.managers.scheduler_components.load_inquirer.get_lora",
                return_value=SimpleNamespace(enable_lora=False),
            ),
            patch(
                "sglang.srt.managers.scheduler_components.load_inquirer.time.time",
                return_value=1000.0,
            ),
            patch(
                "sglang.srt.managers.scheduler_components.load_inquirer.time.monotonic",
                return_value=123.5,
            ),
        ):
            snapshot = inquirer.get_loads()

        self.assertEqual(snapshot.timestamp, 1000.0)
        self.assertEqual(snapshot.snapshot_monotonic_s, 123.5)
        self.assertEqual(snapshot.num_used_tokens, 64)
        self.assertEqual(snapshot.kv_capacity.full_available_tokens, 520640)
        self.assertEqual(snapshot.kv_capacity.full_evictable_tokens, 8192)
        self.assertNotEqual(
            snapshot.kv_capacity.full_available_tokens,
            snapshot.max_total_num_tokens - snapshot.num_used_tokens,
        )
        self.assertEqual(snapshot.kv_capacity.mamba_available_slots, 6)
        self.assertEqual(snapshot.kv_capacity.mamba_evictable_slots, 0)
        self.assertIsNone(snapshot.kv_capacity.swa_available_tokens)
        self.assertEqual(snapshot.kv_capacity.request_slots_available, 1)
        observer.get_pool_stats.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
