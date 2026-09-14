"""A retained chunk row must not strand the final free request slot."""

import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class CandidateReached(Exception):
    pass


class BatchBuildReached(Exception):
    pass


class TestChunkedRequestAdmission(unittest.TestCase):
    def admission_reaches_candidate(self, free_rows, running_bs, pp_limit=4, pd=False):
        chunk = MagicMock()
        chunk.kv = ReqKvInfo(req_pool_idx=3, kv_allocated_len=4096)
        candidate = MagicMock()
        candidate.beam_group = None
        candidate.init_next_round_input.side_effect = CandidateReached
        batch = SimpleNamespace(reqs=[object()] * running_bs, batch_is_full=False)
        adder = SimpleNamespace(
            can_run_list=[chunk],
            add_chunked_req=lambda req: req,
            preempt_list=[],
            new_chunked_req=None,
        )
        from sglang.srt.disaggregation.utils import DisaggregationMode

        scheduler = SimpleNamespace(
            grammar_manager=MagicMock(),
            enable_hierarchical_cache=False,
            enable_unified_cache_external_linker=False,
            enable_hicache_storage=False,
            enable_priority_preemption=False,
            is_hybrid_swa=False,
            waiting_queue=[candidate],
            chunked_req=chunk,
            min_free_slots_delayer=None,
            policy=MagicMock(),
            processed_tokens_counter=0,
            chunked_prefill_size=4096,
            dynamic_chunk_sizer=None,
            tp_worker=SimpleNamespace(
                model_runner=SimpleNamespace(attn_backend=object())
            ),
            page_size=64,
            tree_cache=object(),
            token_to_kv_pool_allocator=object(),
            new_token_ratio_tracker=SimpleNamespace(current=1.0),
            max_prefill_tokens=8192,
            is_mixed_chunk=False,
            priority_scheduling_preemption_threshold=0,
            max_prefill_bs=4,
            max_running_requests=4,
            dllm_config=None,
            enable_lora=False,
            req_to_token_pool=SimpleNamespace(available_size=lambda: free_rows),
            beam_coordinator=SimpleNamespace(pending_member_rows=lambda batch: 0),
            running_batch=batch,
            disaggregation_mode=(
                DisaggregationMode.PREFILL if pd else DisaggregationMode.NULL
            ),
            truncation_align_size=None,
        )
        scheduler.grammar_manager.has_waiting_grammars.return_value = False
        scheduler.get_num_allocatable_reqs = MethodType(
            Scheduler.get_num_allocatable_reqs, scheduler
        )
        module = "sglang.srt.managers.scheduler"
        with (
            patch(
                module + ".get_memory",
                return_value=SimpleNamespace(enable_flexkv=False),
            ),
            patch(
                module + ".get_parallel",
                return_value=SimpleNamespace(pp_max_micro_batch_size=pp_limit),
            ),
            patch(
                module + ".get_schedule",
                return_value=SimpleNamespace(prefill_max_requests=None),
            ),
            patch(module + ".TEST_RETRACT", False),
            patch(module + ".PrefillAdder", return_value=adder),
            patch(module + ".set_time_batch", side_effect=BatchBuildReached),
        ):
            try:
                Scheduler._get_new_batch_prefill_raw(scheduler, None, batch)
            except CandidateReached:
                return True
            except BatchBuildReached:
                return False
        return False

    def test_last_free_row_is_not_charged_for_retained_chunk(self):
        self.assertTrue(self.admission_reaches_candidate(free_rows=1, running_bs=2))

    def test_no_free_row_still_blocks_new_request(self):
        self.assertFalse(self.admission_reaches_candidate(free_rows=0, running_bs=2))

    def test_reused_row_does_not_expand_pipeline_budget(self):
        self.assertFalse(
            self.admission_reaches_candidate(free_rows=3, running_bs=2, pp_limit=3)
        )

    def test_prefill_disaggregation_also_reuses_chunk_row(self):
        self.assertTrue(
            self.admission_reaches_candidate(free_rows=1, running_bs=2, pd=True)
        )


if __name__ == "__main__":
    unittest.main()
