"""CPU-only semantic contracts; importing this test never imports torch."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

from prometheus_client import CollectorRegistry


SOURCE = Path(__file__).resolve().parents[1] / "native/sglang_pd_monitoring.py"
SPEC = importlib.util.spec_from_file_location("pd_monitoring_test_target", SOURCE)
monitor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(monitor)


class LengthOnly:
    def __init__(self, count):
        self.count = count

    def __len__(self):
        return self.count

    def __iter__(self):
        raise AssertionError("long prompt must not be copied or scanned")


class TensorBomb:
    touched = False

    def __int__(self):
        self.touched = True
        raise AssertionError("device conversion")

    def item(self):
        self.touched = True
        raise AssertionError("device synchronization")


class FINISH_LENGTH:
    pass


class FINISH_ABORT:
    def __init__(self, status_code):
        self.status_code = status_code


def req(*, total=131072, cached=0, parts=(0, 0, 0), finished=True, rid="request"):
    value = NS(
        rid=rid,
        origin_input_ids=LengthOnly(total),
        output_ids=[1, 2, 3],
        cached_tokens=cached,
        cached_tokens_device=parts[0],
        cached_tokens_host=parts[1],
        cached_tokens_storage=parts[2],
        finished_reason=FINISH_LENGTH() if finished else None,
    )
    value.finished = lambda: value.finished_reason is not None
    return value


def batch(*requests):
    return NS(
        reqs=list(requests),
        padded_batch_size=16,
        forward_mode=NS(is_prebuilt=lambda: False, is_idle=lambda: False),
    )


class PDMonitoringTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.registry = CollectorRegistry()
        self.labels = dict(
            cluster="test",
            model_name="glm53",
            role="decode",
            service_id="d0",
            node="node",
            dp_rank="0",
        )

    def observer(self, role="decode", **config):
        self.labels["role"] = role
        return monitor.PDObserver(
            labels=self.labels,
            config=dict(topk=1, **config),
            registry=self.registry,
            clock=lambda: self.now,
        )

    def value(self, name, **extra):
        return self.registry.get_sample_value(name, dict(self.labels, **extra))

    def test_real_requests_exclude_graph_padding_and_health(self):
        m = self.observer()
        a, b = req(finished=False), req(finished=False)
        m.batch_started(batch(a, b, req(rid="HEALTH_CHECK_probe")))
        self.assertEqual(self.value("llm_pd_running_requests"), 2)
        self.assertEqual(self.value("llm_pd_running_context_tokens"), 2 * (131072 + 3))
        a.finished_reason = FINISH_LENGTH()
        m.batch_finished(batch(a, b))
        self.assertEqual(self.value("llm_pd_running_requests"), 1)

    def test_prefill_overlap_is_union_and_completion_removes_only_its_batch(self):
        m = self.observer("prefill")
        a, b, c = req(), req(), req()
        first, second = batch(a, b), batch(b, c)
        m.batch_started(first)
        m.batch_started(second)
        self.assertEqual(self.value("llm_pd_running_requests"), 3)
        m.batch_finished(first)
        self.assertEqual(self.value("llm_pd_running_requests"), 2)
        m.batch_finished(second)
        self.assertEqual(self.value("llm_pd_running_requests"), 0)

    def test_prefill_copied_batches_and_repeated_chunk_requests(self):
        m = self.observer("prefill")
        a, b = req(), req()
        m.batch_started(batch(a, b))
        m.batch_started(batch(a, b))
        self.assertEqual(self.value("llm_pd_running_requests"), 2)
        m.batch_finished(batch(a, b))
        self.assertEqual(self.value("llm_pd_running_requests"), 2)
        m.batch_finished(batch(a, b))
        self.assertEqual(self.value("llm_pd_running_requests"), 0)
        self.assertEqual(self.value("llm_pd_running_context_tokens"), 0)

    def test_prebuilt_is_not_executing_decode(self):
        m = self.observer()
        work = batch(req(finished=False))
        work.forward_mode.is_prebuilt = lambda: True
        m.batch_started(work)
        m.batch_finished(work)
        self.assertEqual(self.value("llm_pd_running_requests"), 0)

    def test_time_residence_and_fast_idle_do_not_write_each_iteration(self):
        m = self.observer("prefill")
        self.now = 1.0
        work = batch(req())
        m.batch_started(work)
        self.now = 3.0
        m.batch_finished(work)
        self.assertEqual(
            self.value("llm_pd_concurrency_seconds_total", requests="1"), 2
        )
        before = self.value("llm_pd_concurrency_seconds_total", requests="0")
        for tick in range(100):
            m._transition(0, 3.0 + tick / 1000)
        self.assertEqual(
            self.value("llm_pd_concurrency_seconds_total", requests="0"), before
        )
        m._transition(0, 5.0)
        self.assertEqual(
            self.value("llm_pd_concurrency_seconds_total", requests="0"), 3
        )

    def test_prefill_cache_conservation_and_finish_dedup(self):
        m = self.observer("prefill")
        r = req(cached=130816, parts=(100000, 20000, 10816))
        m.finished(r)
        m.finished(r)
        self.assertEqual(self.value("llm_pd_requests_total", event="kv_delivered"), 1)
        for source, expected in (
            ("hbm", 100000),
            ("host", 20000),
            ("ssd", 10816),
            ("miss", 256),
        ):
            self.assertEqual(
                self.value("llm_pd_prefix_tokens_total", source=source), expected
            )
        self.assertEqual(
            self.value("llm_pd_metric_available", metric_group="prefix_tiers"), 1
        )

    def test_missing_tier_is_unattributed_not_false_hbm(self):
        m = self.observer("prefill")
        m.finished(req(cached=130816))
        self.assertEqual(
            self.value("llm_pd_prefix_tokens_total", source="unattributed"), 130816
        )
        self.assertEqual(self.value("llm_pd_prefix_tokens_total", source="hbm"), 0)
        self.assertEqual(
            self.value("llm_pd_metric_available", metric_group="prefix_tiers"), 0
        )

    def test_decode_local_prefix_ignores_carried_prefill_cache(self):
        m = self.observer()
        dr = NS(
            req=req(cached=999999),
            glm53_cache_audit={"input_len": 131072, "actual_prefix_len": 130816},
        )
        m.decode_transfer(dr)
        m.decode_transfer(dr)
        self.assertEqual(
            self.value("llm_pd_prefix_tokens_total", source="local_reuse"), 130816
        )
        self.assertEqual(
            self.value("llm_pd_prefix_tokens_total", source="transferred"), 256
        )
        self.assertEqual(
            self.value("llm_pd_metric_available", metric_group="prefix"), 1
        )

    def test_unknown_decode_profile_is_unavailable(self):
        m = self.observer()
        m.decode_transfer(NS(req=req(), glm53_cache_audit=None))
        self.assertEqual(
            self.value("llm_pd_metric_available", metric_group="prefix"), 0
        )

    def test_mtp_correct_drafts_exclude_bonus_and_follow_active_width(self):
        m = self.observer()
        m.spec_verify(3, 6, 9, {"num_draft_tokens": 5})
        m.spec_verify(2, 4, 6, {"num_draft_tokens": 4})
        self.assertEqual(self.value("llm_pd_spec_tokens_total", kind="accepted"), 10)
        self.assertEqual(self.value("llm_pd_spec_tokens_total", kind="proposed"), 18)
        self.assertEqual(self.value("llm_pd_spec_tokens_total", kind="emitted"), 15)
        self.assertEqual(self.value("llm_pd_spec_verify_total"), 5)

    def test_metrics_never_convert_a_device_tensor(self):
        m = self.observer()
        tensor = TensorBomb()
        with self.assertRaises(ValueError):
            m.spec_verify(tensor, 0, 0, {"num_draft_tokens": 5})
        self.assertFalse(tensor.touched)

    def test_abort_is_not_success_or_cache_hit(self):
        m = self.observer("prefill")
        cancelled, failed = req(cached=100), req(cached=100)
        cancelled.finished_reason = FINISH_ABORT(499)
        failed.finished_reason = FINISH_ABORT(500)
        m.finished(cancelled)
        m.finished(failed)
        self.assertEqual(self.value("llm_pd_requests_total", event="kv_delivered"), 0)
        self.assertEqual(self.value("llm_pd_requests_total", event="cancelled"), 1)
        self.assertEqual(self.value("llm_pd_requests_total", event="failed"), 1)

    def test_queued_abort_has_one_terminal_without_finished_reason(self):
        m = self.observer()
        work = req(finished=False)
        abort = NS(finished_reason=None, abort_message=None)
        m.received(work)
        m.aborted(work, abort)
        m.aborted(work, abort)
        self.assertEqual(self.value("llm_pd_requests_total", event="cancelled"), 1)
        self.assertEqual(self.value("llm_pd_requests_total", event="completed"), 0)

    def test_running_abort_source_distinguishes_cancel_from_timeout(self):
        m = self.observer()
        first, second = (
            req(finished=False, rid="client-1"),
            req(finished=False, rid="timeout-1"),
        )
        scheduler = NS(
            waiting_queue=[],
            collect_inflight_reqs=lambda: [first, second],
            chunked_req=None,
            disagg_decode_prealloc_queue=NS(queue=[], retracted_queue=[]),
            disagg_decode_transfer_queue=NS(queue=[]),
        )
        m.abort_requested(
            scheduler,
            NS(
                rid="client-1",
                abort_all=False,
                finished_reason=None,
                abort_message=None,
            ),
        )
        m.abort_requested(
            scheduler,
            NS(
                rid="timeout-1",
                abort_all=False,
                finished_reason=None,
                abort_message="SLA timeout",
            ),
        )
        first.finished_reason = FINISH_ABORT(None)
        second.finished_reason = FINISH_ABORT(503)
        m.finished(first)
        m.finished(second)
        self.assertEqual(self.value("llm_pd_requests_total", event="cancelled"), 1)
        self.assertEqual(self.value("llm_pd_requests_total", event="failed"), 1)

    def test_refresh_uses_host_slots_and_disjoint_decode_queues(self):
        m = self.observer()
        stats = NS(
            full_num_used=100,
            full_available_size=700,
            full_evictable_size=200,
            is_hybrid_ssm=True,
            mamba_num_used=2,
            mamba_available_size=8,
            mamba_evictable_size=0,
        )
        scheduler = NS(
            waiting_queue=[1],
            disagg_decode_prealloc_queue=NS(queue=[1, 2], retracted_queue=[]),
            disagg_decode_transfer_queue=NS(queue=[1, 2, 3]),
            pool_stats_observer=NS(get_pool_stats=lambda: stats),
        )
        self.now = 10
        m.refresh(scheduler)
        self.assertEqual(self.value("llm_pd_queue_requests", stage="decode_ready"), 1)
        self.assertIsNone(self.value("llm_pd_queue_requests", stage="waiting"))
        self.assertEqual(self.value("llm_pd_kv_tokens", state="total"), 1000)
        self.assertEqual(self.value("llm_pd_mamba_slots", state="total"), 10)
        self.assertEqual(self.value("llm_pd_last_snapshot_monotonic_seconds"), 10)

    def test_hook_failure_is_nonfatal_to_inference(self):
        old = monitor._observer
        try:
            monitor._observer = self.observer()
            monitor.finished(NS())
            self.assertEqual(
                self.value("llm_pd_instrumentation_errors_total", operation="finished"),
                1,
            )
        finally:
            monitor._observer = old


if __name__ == "__main__":
    unittest.main()
