"""CPU semantic tests for actual terminal lengths and resolved P/D domains."""
import ast
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from prometheus_client import CollectorRegistry
from test_native_regression import monitor, req, LengthOnly, TensorBomb, FINISH_ABORT

ROOT = Path(__file__).resolve().parents[1]


class LengthTests(unittest.TestCase):
    def observer(self, role="decode", rank=0, registry=None):
        registry = registry or CollectorRegistry()
        labels = dict(cluster="new", model_name="glm", role=role, service_id=role+"-0", node="host", dp_rank=rank)
        return monitor.PDObserver(labels=labels, config={"prefill_dp_size": 4, "decode_dp_size": 8}, registry=registry), registry

    def total(self, registry, kind="output", suffix="sum", **match):
        return sum(s.value for f in registry.collect() for s in f.samples
                   if s.name == f"llm_pd_request_{kind}_tokens_{suffix}"
                   and all(s.labels.get(k) == str(v) for k,v in match.items()))

    def test_once_only_across_success_abort_and_repeated_output(self):
        m, reg = self.observer(); r = req(total=99)
        r.disagg_prefill_dp_rank = 3
        m.finished(r); m.finished(r); m.aborted(r, NS())
        self.assertEqual(self.total(reg, suffix="count"), 1)
        self.assertEqual(self.total(reg, "input"), 99)
        self.assertEqual(self.total(reg, prefill_dp_rank=3, status="completed"), 3)

    def test_cancelled_zero_output_is_an_observation_not_missing(self):
        m, reg = self.observer(); r = req(finished=False)
        r.output_ids = []
        m.aborted(r, NS()); r.finished_reason = FINISH_ABORT(499); m.finished(r)
        self.assertEqual(self.total(reg, suffix="count", status="cancelled"), 1)
        self.assertEqual(self.total(reg, status="cancelled"), 0)
        self.assertEqual(reg.get_sample_value("llm_pd_request_output_tokens_bucket",
                         dict(m.labels, prefill_dp_rank="unknown", status="cancelled", le="0.0")), 1)

    def test_failure_keeps_original_prompt_after_scheduler_dummy_replacement(self):
        m, reg = self.observer(); r = req(total=1)
        r.origin_input_ids_unpadded = LengthOnly(700000)
        r.finished_reason = FINISH_ABORT(500)
        m.finished(r)
        self.assertEqual(self.total(reg, "input", status="failed"), 700000)

    def test_actual_mtp_stop_no_requested_length_or_array_copy(self):
        m, reg = self.observer(); r = req()
        r.output_ids = LengthOnly(42); r.finished_len = 37
        r.sampling_params = NS(max_new_tokens=65536)
        m.finished(r)
        self.assertEqual(self.total(reg), 37)

    def test_abort_origin_distinguishes_cancel_from_internal_error(self):
        for abort, status in [(NS(), "cancelled"), (NS(abort_message="failure"), "failed"),
                              (NS(finished_reason={"status_code": 499}), "cancelled")]:
            m, reg = self.observer(); r = req(finished=False)
            m.aborted(r, abort); m.aborted(r, abort)
            self.assertEqual(self.total(reg, suffix="count", status=status), 1)
            self.assertEqual(self.total(reg, status=status), 3)

    def test_inflight_client_cancellation_uses_recorded_origin(self):
        m, reg = self.observer(); r = req()
        r.finished_reason = FINISH_ABORT(None); r._pd_metrics_abort_event = "cancelled"
        m.finished(r)
        self.assertEqual(self.total(reg, suffix="count", status="cancelled"), 1)

    def test_unknown_rank_and_bounded_labels_do_not_convert_device_objects(self):
        m, reg = self.observer(rank=None)
        self.assertEqual(m.labels["dp_rank"], "unknown")
        for rank in (None, -1, 4, "99999", "a-request-id", TensorBomb(), True):
            r = req(); r.disagg_prefill_dp_rank = rank
            m.finished(r)
        self.assertEqual(self.total(reg, suffix="count", prefill_dp_rank="unknown"), 7)
        self.assertEqual({s.labels["prefill_dp_rank"] for f in reg.collect() for s in f.samples
                          if "prefill_dp_rank" in s.labels}, {"0","1","2","3","unknown"})
        self.assertFalse(TensorBomb.touched)

    def test_resolved_metadata_precedes_request_hint_and_survives_rebootstrap(self):
        m, reg = self.observer(); r = req(); r.disagg_prefill_dp_rank = 1
        m.prefill_resolved(r, 3); m.prefill_resolved(r, 2)
        m.finished(r)
        self.assertEqual(self.total(reg, suffix="count", prefill_dp_rank="3"), 1)

    def test_unknown_resolution_can_be_replaced_and_missing_hints_stay_unknown(self):
        m, reg = self.observer(); r = req()
        m.prefill_resolved(r, None); m.prefill_resolved(r, 2); m.finished(r)
        m.finished(req())
        self.assertEqual(self.total(reg, suffix="count", prefill_dp_rank="2"), 1)
        self.assertEqual(self.total(reg, suffix="count", prefill_dp_rank="unknown"), 1)

    def test_rebootstrap_temporarily_popped_token_is_counted(self):
        m, reg = self.observer(); r = req(); r.pd_rebootstrap_forced_output_id = 123
        m.aborted(r, NS())
        self.assertEqual(self.total(reg, status="cancelled"), 4)

    def test_prefill_stage_never_publishes_full_request_output(self):
        m, reg = self.observer("prefill", 3); r = req(total=1048576); r.output_ids = [1]
        m.finished(r); m.finished(r)
        self.assertEqual(reg.get_sample_value("llm_pd_prefill_stage_input_tokens_sum", dict(m.labels, status="completed")), 1048576)
        self.assertFalse(any(f.name.startswith("llm_pd_request_") for f in reg.collect()))

    def test_health_and_nonterminal_never_observe_lengths(self):
        m, reg = self.observer()
        m.finished(req(finished=False)); m.finished(req(rid="HEALTH_CHECK-x"))
        m.aborted(req(rid="HEALTH_CHECK-x"), NS())
        self.assertEqual(self.total(reg, suffix="count"), 0)

    def test_rank_global_bucket_aggregation_is_request_weighted(self):
        # Highly unequal rank populations: averaging quantiles would be wrong.
        all_samples = []
        for rank, lengths in [(0, [1]*9), (7, [1024])]:
            m, reg = self.observer(rank=rank)
            for length in lengths:
                r = req(); r.output_ids = LengthOnly(length); r.disagg_prefill_dp_rank = rank % 4
                m.finished(r)
            all_samples.extend(s for f in reg.collect() for s in f.samples
                               if s.name == "llm_pd_request_output_tokens_bucket")
        buckets = {}
        for s in all_samples:
            le = float(s.labels["le"]); buckets[le] = buckets.get(le,0) + s.value
        self.assertEqual(buckets[float("inf")], 10)
        self.assertEqual(buckets[1], 9)
        # Standard linear interpolation in the cumulative histogram: p50 <= 1,
        # whereas averaging this rank's p50 with the other rank's is > 256.
        self.assertAlmostEqual((0.5 * buckets[float("inf")]) / buckets[1], 5/9)
        self.assertEqual(sum(s.value for s in all_samples if s.labels["prefill_dp_rank"] == "3" and s.labels["le"] == "+Inf"), 1)

    def test_tp_cp_pp_replicas_do_not_construct_observers(self):
        created = []
        def factory(**kwargs):
            created.append(kwargs); return NS()
        env = {"SGLANG_PD_METRICS": "1", "SGLANG_PD_METRICS_ROLE": "decode",
               "SGLANG_PD_METRICS_SERVICE_ID": "decode-0", "SGLANG_PD_METRICS_NODE": "d0"}
        args = NS(tp_size=16,dp_size=8,ep_size=16,context_length=1048576,
                  speculative_num_draft_tokens=5,speculative_num_steps=4,speculative_eagle_topk=1,
                  kv_cache_dtype="int8",served_model_name="glm",model_path="glm")
        with patch.dict(os.environ,env), patch.object(monitor,"PDObserver",factory):
            for tp,cp,pp in [(0,0,0),(1,0,0),(0,1,0),(0,0,1)]:
                with patch.object(monitor,"_configured",False), patch.object(monitor,"_observer",None):
                    monitor.configure(NS(server_args=args,ps=NS(attn_tp_rank=tp,attn_cp_rank=cp,pp_rank=pp,dp_rank=None)))
        self.assertEqual(len(created),1)
        self.assertEqual(created[0]["labels"]["dp_rank"],"unknown")

    def test_no_device_imports(self):
        self.assertNotIn("torch",sys.modules)
        self.assertNotIn("torch_npu",sys.modules)


if __name__ == "__main__":
    unittest.main()
