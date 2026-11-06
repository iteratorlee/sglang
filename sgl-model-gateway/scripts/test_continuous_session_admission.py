#!/usr/bin/env python3
"""CPU-only regression tests; Python 3.7+, standard library, no network.

Run: python3 -B -m unittest -v test_continuous_session_admission
Network/Docker I/O is mocked; tick tests use tiny temporary JSON files.
"""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
from urllib.parse import parse_qs, urlsplit


# Loading the sibling by path also works with unittest discovery from elsewhere.
_spec = importlib.util.spec_from_file_location(
    "continuous_session_admission_under_test",
    str(Path(__file__).with_name("continuous_session_admission.py")))
admission = importlib.util.module_from_spec(_spec)
_old_bytecode = sys.dont_write_bytecode
try:
    sys.dont_write_bytecode = True
    _spec.loader.exec_module(admission)
finally:
    sys.dont_write_bytecode = _old_bytecode


def session(number, **changes):
    row = dict(session_hash="{:064x}".format(number), destination="legacy",
               in_flight=0, idle_seconds=30, last_body_bytes=1024,
               last_completed_cleanly=True, warmed=False)
    row.update(changes)
    return row


def healthy_workers():
    rows = []
    for name, role, count in [("incumbent", "prefill", 2),
                              ("new-prefill", "prefill", 2),
                              ("decode", "decode", 8)]:
        for rank in range(count):
            rows.append(dict(url="http://{}.invalid:8000@{}".format(name, rank),
                             worker_type=role, is_healthy=True, load=0))
    return dict(stats=dict(prefill_count=4, decode_count=8, regular_count=0),
                workers=rows)


class PolicyFixture(unittest.TestCase):
    def setUp(self):
        self.policy = dict(admission.DEFAULT_POLICY)
        self.migration = dict(
            config=dict(enabled=True, new_sessions_bps=10000,
                        legacy_sessions_bps=0, max_cold_inflight=1,
                        min_interval_secs=60, quiet_secs=0, max_switches=8,
                        session_hashes=[]),
            enrolled=5, migrated=3, cold_inflight=0, tracked_sessions=2,
            active_legacy=0, active_candidate=0,
            sessions=[session(1), session(2)])
        self.handoff = dict(config=dict(candidate_bps=0, bounded_first_turn=True),
                            persistence_failed=False, legacy_pins=2,
                            candidate_pins=8)
        self.workers = healthy_workers()
        self.feedback = dict(fresh=True, candidate_errors=17,
                             old_pressure=2, new_pressure=1, new_waiting=0)
        self.private = set()
        self.baseline = 17

    def evaluate(self, latched=False):
        return admission.plan(self.migration, self.handoff, self.workers,
                              self.feedback, self.private, self.policy,
                              self.baseline, latched)

    def assert_stopped(self, result, reason):
        self.assertEqual(result["reason"], reason)
        self.assertEqual(result["selected"], [])
        self.assertEqual(result["config"]["new_sessions_bps"], 0)
        self.assertEqual(result["config"]["legacy_sessions_bps"], 0)
        self.assertEqual(result["config"]["session_hashes"], [])
        self.assertEqual(result["config"]["max_switches"], result["used"])


class SelectSessionsTests(PolicyFixture):
    def select(self, rows):
        self.migration["sessions"] = rows
        return admission.select_sessions(self.migration, self.private, self.policy)

    def test_recent_window_includes_zero_and_900_seconds(self):
        rows = [session(1, idle_seconds=0), session(2, idle_seconds=900),
                session(3, idle_seconds=900.001), session(4, idle_seconds=-1)]
        self.assertEqual(self.select(rows), [rows[0]["session_hash"],
                                             rows[1]["session_hash"]])

    def test_only_cleanly_completed_sessions_are_eligible(self):
        for value in (False, None, 0, 1, "true"):
            with self.subTest(completed=value):
                self.assertEqual(self.select([
                    session(1, last_completed_cleanly=value)]), [])
        row = session(1)
        del row["last_completed_cleanly"]
        self.assertEqual(self.select([row]), [])

    def test_body_size_accepts_one_byte_through_one_mib(self):
        rows = [session(1, last_body_bytes=1),
                session(2, last_body_bytes=1048576)]
        self.assertEqual(self.select(rows), [row["session_hash"] for row in rows])
        for value in (0, -1, 1048577, 1.0, True, None, "1024"):
            with self.subTest(size=value):
                self.assertEqual(self.select([session(3, last_body_bytes=value)]), [])
        row = session(1)
        del row["last_body_bytes"]
        self.assertEqual(self.select([row]), [])

    def test_private_and_nonlegacy_sessions_are_excluded(self):
        self.private.add(session(1)["session_hash"])
        rows = [session(1), session(2, destination="candidate"),
                session(3, destination=None), session(4)]
        self.assertEqual(self.select(rows), [rows[3]["session_hash"]])

    def test_invalid_or_missing_idle_metadata_is_excluded(self):
        for value in (None, float("nan"), float("inf"), float("-inf"),
                      True, "30"):
            with self.subTest(idle=value):
                self.assertEqual(self.select([session(1, idle_seconds=value)]), [])
        row = session(1)
        del row["idle_seconds"]
        self.assertEqual(self.select([row]), [])

    def test_malformed_hash_is_never_sent_to_gateway(self):
        for value in ("", "a" * 63, "a" * 65, "g" * 64, "../session"):
            with self.subTest(key=value):
                self.assertEqual(self.select([session(1, session_hash=value)]), [])

    def test_recent_cohort_then_short_bodies_then_recency(self):
        rows = [session(1, idle_seconds=121, last_body_bytes=1),
                session(2, idle_seconds=10, last_body_bytes=512),
                session(3, idle_seconds=120, last_body_bytes=128),
                session(4, idle_seconds=5, last_body_bytes=128),
                session(5, idle_seconds=300, last_body_bytes=2)]
        self.assertEqual(self.select(rows),
                         [session(i)["session_hash"] for i in (4, 3, 2, 1)])

    def test_in_flight_session_can_be_selected_without_mutating_its_pin(self):
        # Actual request completion is enforced by the Rust core's lease gate.
        row = session(1, in_flight=2)
        before = copy.deepcopy(row)
        self.assertEqual(self.select([row]), [row["session_hash"]])
        self.assertEqual(row, before)

    def test_policy_overrides_bound_window_body_and_cohort(self):
        self.policy.update(recent_secs=10, max_body_bytes=128, cohort_size=1)
        self.assertEqual(self.select([
            session(1, idle_seconds=11, last_body_bytes=1),
            session(2, idle_seconds=5, last_body_bytes=129),
            session(3, idle_seconds=5, last_body_bytes=128),
            session(4, idle_seconds=5, last_body_bytes=64)]),
            [session(4)["session_hash"]])


class PlanTests(PolicyFixture):
    def test_exhausted_eight_and_nine_credit_caps_are_replenished(self):
        for used, cap in ((8, 8), (9, 9)):
            with self.subTest(used=used, cap=cap):
                self.migration.update(enrolled=used - 3, migrated=3)
                self.migration["config"]["max_switches"] = cap
                result = self.evaluate()
                self.assertEqual(result["reason"], "credit_ready")
                self.assertEqual(result["used"], used)
                self.assertEqual(result["config"]["max_switches"], used + 1)

    def test_unconsumed_credit_does_not_accumulate_over_repeated_plans(self):
        for unused_poll in range(10):
            result = self.evaluate()
            self.assertEqual(result["config"]["max_switches"], 9)
            self.migration["config"] = copy.deepcopy(result["config"])
        self.assertEqual((self.migration["enrolled"], self.migration["migrated"]),
                         (5, 3))

    def test_multiple_consumed_credits_keep_progressing(self):
        for used in range(8, 20):
            self.migration.update(enrolled=used - 3, migrated=3)
            result = self.evaluate()
            self.assertEqual(result["config"]["max_switches"], used + 1)
            self.migration["config"] = copy.deepcopy(result["config"])

    def test_large_existing_cap_cannot_issue_multiple_unused_credits(self):
        self.migration["config"]["max_switches"] = 100
        self.assertEqual(self.evaluate()["config"]["max_switches"], 9)

    def test_cold_session_blocks_new_switches_and_preserves_all_inputs(self):
        self.migration["cold_inflight"] = 1
        self.migration["config"]["max_switches"] = 9
        before = copy.deepcopy((self.migration, self.handoff, self.workers,
                                self.feedback, self.private, self.policy))
        self.assert_stopped(self.evaluate(), "cold_session_inflight")
        self.assertEqual((self.migration, self.handoff, self.workers,
                          self.feedback, self.private, self.policy), before)

    def test_manual_pause_returns_no_configuration(self):
        self.migration["config"]["enabled"] = False
        for latched in (False, True):
            result = self.evaluate(latched)
            self.assertEqual(result["reason"], "manual_paused")
            self.assertIsNone(result["config"])
            self.assertEqual(result["selected"], [])
            self.assertEqual(result["error_latched"], latched)

    def test_handoff_persistence_failure_or_other_admission_path_stops(self):
        for failed, bps in ((True, 0), (False, 1), (False, 10000)):
            with self.subTest(failed=failed, bps=bps):
                self.handoff["persistence_failed"] = failed
                self.handoff["config"]["candidate_bps"] = bps
                self.assert_stopped(self.evaluate(), "handoff_not_ready")

    def test_each_unhealthy_prefill_or_decode_worker_stops_admission(self):
        for index in range(12):
            with self.subTest(worker=index):
                self.workers = healthy_workers()
                self.workers["workers"][index]["is_healthy"] = False
                self.assert_stopped(self.evaluate(), "workers_not_ready")

    def test_worker_count_or_role_count_mismatch_stops_admission(self):
        for role, value in (("prefill_count", 3), ("prefill_count", 5),
                            ("decode_count", 7), ("regular_count", 1)):
            with self.subTest(role=role, value=value):
                self.workers = healthy_workers()
                self.workers["stats"][role] = value
                self.assert_stopped(self.evaluate(), "workers_not_ready")
        for count in (11, 13):
            with self.subTest(count=count):
                self.workers = healthy_workers()
                self.workers["workers"] = (self.workers["workers"] * 2)[:count]
                self.assert_stopped(self.evaluate(), "workers_not_ready")

    def test_stale_or_missing_feedback_stops_admission(self):
        for feedback in ({"fresh": False}, {}):
            self.feedback = feedback
            self.assert_stopped(self.evaluate(), "feedback_unavailable")

    def test_candidate_error_increase_latches_even_after_counter_recovers(self):
        self.feedback["candidate_errors"] = 18
        result = self.evaluate()
        self.assert_stopped(result, "candidate_error_latched")
        self.assertIs(result["error_latched"], True)
        self.feedback["candidate_errors"] = 17
        self.assert_stopped(self.evaluate(result["error_latched"]),
                            "candidate_error_latched")

    def test_existing_error_latch_prevents_new_credit(self):
        self.assert_stopped(self.evaluate(True), "candidate_error_latched")

    def test_error_counter_rollback_or_invalid_value_stops(self):
        for value in (16, -1, None, float("nan"), float("inf"),
                      float("-inf"), True, "17"):
            with self.subTest(errors=value):
                self.feedback["candidate_errors"] = value
                self.assert_stopped(self.evaluate(), "error_counter_unavailable")
        self.feedback.pop("candidate_errors")
        self.assert_stopped(self.evaluate(), "error_counter_unavailable")

    def test_invalid_load_feedback_stops(self):
        original = copy.deepcopy(self.feedback)
        for key in ("new_pressure", "old_pressure", "new_waiting"):
            for value in (-1, None, float("nan"), float("inf"), True, "0"):
                with self.subTest(key=key, value=value):
                    self.feedback = dict(original, **{key: value})
                    self.assert_stopped(self.evaluate(), "invalid_load_feedback")
            self.feedback = dict(original)
            del self.feedback[key]
            self.assert_stopped(self.evaluate(), "invalid_load_feedback")

    def test_load_thresholds_allow_equality(self):
        for new, old, waiting in ((3, 2, 1), (1, 0, 0), (0, 0, 0)):
            with self.subTest(new=new, old=old, waiting=waiting):
                self.feedback.update(new_pressure=new, old_pressure=old,
                                     new_waiting=waiting)
                self.assertEqual(self.evaluate()["reason"], "credit_ready")

    def test_absolute_relative_and_waiting_thresholds_stop_independently(self):
        for new, old, waiting in ((3.001, 100, 0), (2, 0, 0), (0, 100, 1.001)):
            with self.subTest(new=new, old=old, waiting=waiting):
                self.feedback.update(new_pressure=new, old_pressure=old,
                                     new_waiting=waiting)
                self.assert_stopped(self.evaluate(), "new_prefill_pressure")

    def test_truncated_snapshot_reports_incomplete_and_never_selects_unseen(self):
        unseen = session(999)["session_hash"]
        self.migration["tracked_sessions"] = 1000
        self.migration["config"]["session_hashes"] = [unseen]
        self.private.add(session(1)["session_hash"])
        result = self.evaluate()
        self.assertEqual(result["coverage"], "recent_metadata_incomplete")
        self.assertEqual(result["selected"], [session(2)["session_hash"]])
        self.assertEqual(result["config"]["session_hashes"], result["selected"])
        self.assertNotIn(unseen, result["selected"])

    def test_no_eligible_legacy_session_still_allows_new_sessions(self):
        for rows in ([], [session(1, last_completed_cleanly=False)]):
            with self.subTest(rows=rows):
                self.migration["sessions"] = rows
                self.migration["tracked_sessions"] = len(rows)
                result = self.evaluate()
                self.assertEqual(result["reason"], "credit_ready")
                self.assertEqual(result["selected"], [])
                self.assertEqual(result["config"]["session_hashes"], [])
                self.assertEqual(result["config"]["new_sessions_bps"], 10000)
                self.assertEqual(result["config"]["legacy_sessions_bps"], 0)

    def test_lifetime_limit_never_resets_counters_or_wraps_credit(self):
        self.migration.update(enrolled=65532, migrated=3)
        self.assertEqual(self.evaluate()["config"]["max_switches"], 65536)
        self.migration["enrolled"] = 65533
        self.assert_stopped(self.evaluate(), "ledger_lifetime_limit")
        self.assertEqual((self.migration["enrolled"], self.migration["migrated"]),
                         (65533, 3))
        self.migration["enrolled"] = 65534
        before = copy.deepcopy(self.migration)
        with self.assertRaises(ValueError):
            self.evaluate()
        self.assertEqual(self.migration, before)

    def test_ready_plan_is_pure_and_keeps_core_rate_and_cold_gates(self):
        before = copy.deepcopy((self.migration, self.handoff, self.workers,
                                self.feedback, self.private, self.policy))
        result = self.evaluate()
        self.assertEqual((self.migration, self.handoff, self.workers,
                          self.feedback, self.private, self.policy), before)
        self.assertEqual(result["config"]["min_interval_secs"], 60)
        self.assertEqual(result["config"]["max_cold_inflight"], 1)
        self.assertIs(result["config"]["enabled"], True)


class ReadFeedbackTests(unittest.TestCase):
    NOW = 1700000000.0
    STAGES = ("waiting", "bootstrap", "prefill_inflight")

    def setUp(self):
        self.config = dict(
            prometheus="http://prometheus.invalid", cluster="test-cluster",
            expected_nodes=["old-node", "new-node", "decode-node"],
            prefill_rank_locations=[
                dict(service_id=service, node=node, dp_rank=rank)
                for service, node in (("prefill-0", "old-node"),
                                      ("prefill-1", "new-node"))
                for rank in (0, 1)])
        self.rows = []
        for service, count in (("prefill-0", 2), ("prefill-1", 2), ("decode-0", 8)):
            self.add_row("llm_pd:service_complete", count, service_id=service)
        for node in self.config["expected_nodes"]:
            self.add_row("llm_pd_exporter_source_up", 1, node=node)
        for service, count in (("prefill-0", 1), ("prefill-1", 2)):
            self.add_row("llm_pd:prefill_running_by_service", count, service_id=service)
        for location in self.config["prefill_rank_locations"]:
            for stage in self.STAGES:
                labels = dict(location, dp_rank=str(location["dp_rank"]),
                              role="prefill", stage=stage)
                self.add_row("llm_pd_queue_requests", 0, **labels)
        self.add_row("llm_gateway_requests_total", 17,
                     gateway_scope="front", outcome="error")

    def add_row(self, name, value, **labels):
        self.rows.append(dict(metric=dict(labels, __name__=name),
                              value=[self.NOW, str(value)]))

    def read(self):
        response = dict(status="success", data=dict(result=self.rows))
        def respond(_origin, _method, path):
            query = parse_qs(urlsplit(path).query)['query'][0]
            if query.startswith('timestamp('):
                name = query[len('timestamp('):].split('{', 1)[0]
                values = [dict(metric={k: v for k, v in row['metric'].items() if k != '__name__'},
                               value=[self.NOW, row['value'][0]])
                          for row in self.rows if row['metric']['__name__'] == name]
                return dict(status='success', data=dict(result=values))
            return response
        with mock.patch.object(admission, "request", side_effect=respond) as api:
            with mock.patch.object(admission.time, "time", return_value=self.NOW):
                result = admission.read_feedback(self.config)
        self.assertEqual(api.call_count, 6)
        args = api.call_args[0]
        self.assertEqual(args[:2], (self.config["prometheus"], "GET"))
        self.assertTrue(args[2].startswith("/api/v1/query?"))
        return result

    def test_complete_fresh_rank_coverage_produces_feedback(self):
        result = self.read()
        self.assertIs(result["fresh"], True)
        self.assertEqual(result["candidate_errors"], 17)
        self.assertEqual(result["old_pressure"], 1)
        self.assertEqual(result["new_pressure"], 2)
        self.assertEqual(result["new_waiting"], 0)

    def test_all_sample_ages_zero_through_30_and_future_two_seconds_are_allowed(self):
        for age in (0, 30, -2):
            with self.subTest(age=age):
                for row in self.rows:
                    row["value"][0] = self.NOW - age
                self.assertIs(self.read()["fresh"], True)

    def test_one_stale_or_too_far_future_sample_invalidates_feedback(self):
        for index, row in enumerate(self.rows):
            for age in (30.001, -2.001):
                with self.subTest(sample=index, age=age):
                    row["value"][0] = self.NOW - age
                    self.assertIs(self.read()["fresh"], False)
            row["value"][0] = self.NOW

    def test_nonfinite_sample_timestamp_is_unavailable(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(timestamp=value):
                self.rows[0]["value"][0] = value
                self.assertIs(self.read()["fresh"], False)

    def test_any_nonfinite_or_negative_relevant_value_invalidates_feedback(self):
        for index, row in enumerate(self.rows):
            old_value = row["value"][1]
            for value in ("NaN", "+Inf", "-Inf", "-1"):
                with self.subTest(sample=index, value=value):
                    row["value"][1] = value
                    self.assertIs(self.read()["fresh"], False)
            row["value"][1] = old_value

    def test_complete_values_must_be_exactly_two_two_eight(self):
        for index, expected in ((0, 2), (1, 2), (2, 8)):
            for value in (0, 1, expected + 1):
                with self.subTest(service=index, value=value):
                    self.rows[index]["value"][1] = str(value)
                    self.assertIs(self.read()["fresh"], False)
            self.rows[index]["value"][1] = str(expected)

    def test_each_rank_and_each_stage_is_required_even_when_totals_look_healthy(self):
        original = copy.deepcopy(self.rows)
        for index, row in enumerate(original):
            if row["metric"]["__name__"] != "llm_pd_queue_requests":
                continue
            with self.subTest(missing=row["metric"]):
                self.rows = copy.deepcopy(original[:index] + original[index + 1:])
                self.assertIs(self.read()["fresh"], False)

    def test_missing_whole_rank_or_wrong_node_and_rank_cannot_substitute(self):
        original = copy.deepcopy(self.rows)
        for change in ("missing_rank", "wrong_node", "wrong_rank"):
            with self.subTest(change=change):
                self.rows = copy.deepcopy(original)
                selected = [row for row in self.rows
                            if row["metric"].get("service_id") == "prefill-1"
                            and row["metric"].get("dp_rank") == "1"]
                if change == "missing_rank":
                    self.rows = [row for row in self.rows if row not in selected]
                else:
                    key = "node" if change == "wrong_node" else "dp_rank"
                    for row in selected:
                        row["metric"][key] = "unexpected"
                self.assertIs(self.read()["fresh"], False)

    def test_config_requires_four_distinct_prefill_rank_locations(self):
        original = copy.deepcopy(self.config["prefill_rank_locations"])
        for locations in (original[:3], original + [original[0]],
                          original[:3] + [original[0]]):
            with self.subTest(locations=locations):
                self.config["prefill_rank_locations"] = locations
                self.assertIs(self.read()["fresh"], False)

    def test_missing_service_complete_source_or_running_metric_is_unavailable(self):
        original = copy.deepcopy(self.rows)
        for index, row in enumerate(original):
            if row["metric"]["__name__"] not in (
                    "llm_pd:service_complete", "llm_pd_exporter_source_up",
                    "llm_pd:prefill_running_by_service"):
                continue
            with self.subTest(missing=row["metric"]):
                self.rows = copy.deepcopy(original[:index] + original[index + 1:])
                self.assertIs(self.read()["fresh"], False)

    def test_pressure_includes_waiting_and_bootstrap_plus_inflight(self):
        values = {("prefill-0", "0", "waiting"): 1,
                  ("prefill-0", "1", "waiting"): 2,
                  ("prefill-0", "0", "bootstrap"): 4,
                  ("prefill-0", "1", "bootstrap"): 1,
                  ("prefill-0", "0", "prefill_inflight"): 2,
                  ("prefill-1", "0", "waiting"): 1,
                  ("prefill-1", "1", "waiting"): 2,
                  ("prefill-1", "0", "bootstrap"): 1,
                  ("prefill-1", "1", "prefill_inflight"): 1}
        for row in self.rows:
            labels = row["metric"]
            key = (labels.get("service_id"), labels.get("dp_rank"), labels.get("stage"))
            if key in values:
                row["value"][1] = str(values[key])
        result = self.read()
        self.assertIs(result["fresh"], True)
        self.assertEqual(result["old_pressure"], 7)
        self.assertEqual(result["new_pressure"], 5)
        self.assertEqual(result["new_waiting"], 3)

    def test_unrelated_decode_and_nonfront_errors_do_not_poison_prefill_feedback(self):
        self.add_row("llm_pd_queue_requests", "NaN", role="decode", stage="waiting")
        self.rows[-1]["value"][0] = self.NOW - 1000
        self.add_row("llm_gateway_requests_total", 9000,
                     gateway_scope="private", outcome="error")
        result = self.read()
        self.assertIs(result["fresh"], True)
        self.assertEqual(result["candidate_errors"], 17)

    def test_absent_error_counter_is_not_fabricated_as_zero(self):
        self.rows = [row for row in self.rows
                     if row["metric"]["__name__"] != "llm_gateway_requests_total"]
        self.assertIsNone(self.read()["candidate_errors"])

    def test_prometheus_query_failure_is_not_treated_as_healthy(self):
        with mock.patch.object(admission, "request", return_value={"status": "error"}):
            with self.assertRaises(RuntimeError):
                admission.read_feedback(self.config)

    def test_fresh_evaluation_time_does_not_hide_old_stored_samples(self):
        def respond(_origin, _method, path):
            query = parse_qs(urlsplit(path).query)['query'][0]
            if query.startswith('timestamp('):
                name = query[len('timestamp('):].split('{', 1)[0]
                values = [dict(metric={k: v for k, v in row['metric'].items() if k != '__name__'},
                               value=[self.NOW, self.NOW - 3600])
                          for row in self.rows if row['metric']['__name__'] == name]
                return dict(status='success', data=dict(result=values))
            return dict(status='success', data=dict(result=self.rows))
        with mock.patch.object(admission, 'request', side_effect=respond), \
                mock.patch.object(admission.time, 'time', return_value=self.NOW):
            self.assertFalse(admission.read_feedback(self.config)['fresh'])

    def test_absent_underlying_timestamp_cannot_be_assumed_fresh(self):
        def respond(_origin, _method, path):
            query = parse_qs(urlsplit(path).query)['query'][0]
            return dict(status='success', data=dict(
                result=[] if query.startswith('timestamp(') else self.rows))
        with mock.patch.object(admission, 'request', side_effect=respond), \
                mock.patch.object(admission.time, 'time', return_value=self.NOW):
            self.assertFalse(admission.read_feedback(self.config)['fresh'])


class TickTests(PolicyFixture):
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.config = dict(gateway='http://front.invalid:8000',
                           legacy_gateway='http://legacy.invalid:8000',
                           gateway_cid='a' * 64, gateway_started_at='generation-1',
                           ops_lock=str(root / 'ops.lock'),
                           singleton_lock=str(root / 'singleton.lock'),
                           status_file=str(root / 'status.json'),
                           desired_file=str(root / 'desired.json'),
                           private_hashes_file=str(root / 'private.json'),
                           incumbent_urls=['http://incumbent.invalid:8000'],
                           new_prefill_url='http://new-prefill.invalid:8000',
                           expected_workers=[x['url'] for x in self.workers['workers']],
                           error_baseline=17, poll_secs=30)
        Path(self.config['private_hashes_file']).write_text('[]')
        Path(self.config['desired_file']).write_text(json.dumps(
            dict(migration=self.migration['config'], prefill={'preserved': True})))
        self.info = dict(Id='a' * 64, State=dict(Running=True, StartedAt='generation-1'))
        self.puts = []
        self.gets = 0
        self.fail_put = None
        self.pause_on_recheck = False
        self.rollout = dict(new_prefix_bps=10000, candidate_new_only=True,
                            incumbent_urls=self.config['incumbent_urls'])
        self.feedback_mock = mock.patch.object(admission, 'read_feedback',
                                                side_effect=lambda _: copy.deepcopy(self.feedback)).start()
        self.inspect_mock = mock.patch.object(admission, 'inspect_gateway',
                                               side_effect=lambda _: copy.deepcopy(self.info)).start()
        self.api_mock = mock.patch.object(admission, 'request', side_effect=self.api).start()
        self.addCleanup(mock.patch.stopall)

    def api(self, origin, method, path, value=None):
        if path == admission.MIGRATION:
            if method == 'GET':
                self.gets += 1
                if self.pause_on_recheck and self.gets == 2:
                    self.migration['config']['enabled'] = False
                return copy.deepcopy(self.migration)
            self.puts.append(copy.deepcopy(value))
            if self.fail_put == 'before':
                raise TimeoutError('ambiguous request')
            self.migration['config'] = copy.deepcopy(value)
            if self.fail_put == 'after':
                raise TimeoutError('ambiguous response')
            return copy.deepcopy(value)
        if path == admission.HANDOFF:
            return copy.deepcopy(self.handoff)
        if path == '/workers':
            if origin == self.config['legacy_gateway']:
                return dict(workers=copy.deepcopy(self.workers['workers'][:2]))
            return copy.deepcopy(self.workers)
        if path == '/workers/prefill-rollout':
            return copy.deepcopy(self.rollout)
        raise AssertionError((origin, method, path))

    def read_desired(self):
        return json.loads(Path(self.config['desired_file']).read_text())

    def test_tick_replenishes_after_consumption_without_resetting_counters(self):
        self.migration['config']['max_switches'] = 40
        first = admission.tick(self.config)
        self.assertEqual(first['config']['max_switches'], 9)
        self.migration['enrolled'] += 1
        second = admission.tick(self.config)
        self.assertEqual(second['config']['max_switches'], 10)
        self.assertEqual([x['max_switches'] for x in self.puts], [9, 10])
        self.assertEqual(self.read_desired()['migration'], self.migration['config'])
        self.assertNotIn('controller_pending', self.read_desired())

    def test_dry_run_does_not_put_or_persist(self):
        before = Path(self.config['desired_file']).read_bytes()
        result = admission.tick(self.config, dry_run=True)
        self.assertEqual(result['reason'], 'credit_ready')
        self.assertEqual(self.puts, [])
        self.assertEqual(Path(self.config['desired_file']).read_bytes(), before)
        self.assertFalse(Path(self.config['status_file']).exists())

    def test_changed_gateway_generation_never_writes_configuration(self):
        self.info['State']['StartedAt'] = 'generation-2'
        result = admission.tick(self.config)
        self.assertEqual(result['reason'], 'gateway_generation_changed')
        self.assertEqual(self.puts, [])
        self.api_mock.assert_not_called()

    def test_operator_lock_busy_never_reads_or_writes_gateway(self):
        with mock.patch.object(admission.fcntl, 'flock', side_effect=BlockingIOError()):
            result = admission.tick(self.config)
        self.assertEqual(result['reason'], 'operator_lock_busy')
        self.inspect_mock.assert_not_called()
        self.api_mock.assert_not_called()

    def test_manual_pause_is_saved_and_never_reenabled(self):
        self.migration['config']['enabled'] = False
        result = admission.tick(self.config)
        self.assertEqual(result['reason'], 'manual_paused')
        self.assertEqual(self.puts, [])
        self.assertFalse(self.read_desired()['migration']['enabled'])

    def test_direct_operator_change_before_put_is_not_overwritten(self):
        self.pause_on_recheck = True
        result = admission.tick(self.config)
        self.assertEqual(result['reason'], 'operator_config_changed')
        self.assertEqual(self.puts, [])
        self.assertFalse(self.migration['config']['enabled'])

    def test_pending_write_failure_blocks_every_put(self):
        actual = admission.atomic_json
        def fail_desired(path, value):
            if str(path) == self.config['desired_file']:
                raise OSError('persistence unavailable')
            actual(path, value)
        with mock.patch.object(admission, 'atomic_json', side_effect=fail_desired):
            for _ in range(2):
                with self.assertRaises(OSError):
                    admission.tick(self.config)
        self.assertEqual(self.puts, [])

    def test_failed_ack_requires_persistent_reconciliation_before_next_credit(self):
        actual = admission.atomic_json
        def fail_ack(path, value):
            if str(path) == self.config['desired_file'] and 'controller_pending' not in value:
                raise OSError('ack unavailable')
            actual(path, value)
        with mock.patch.object(admission, 'atomic_json', side_effect=fail_ack):
            with self.assertRaises(OSError):
                admission.tick(self.config)
            self.assertEqual(self.migration['config']['max_switches'], 9)
            self.assertIn('controller_pending', self.read_desired())
            self.migration['enrolled'] += 1
            with self.assertRaises(OSError):
                admission.tick(self.config)
            self.assertEqual(len(self.puts), 1)
        result = admission.tick(self.config)
        self.assertEqual(result['config']['max_switches'], 10)
        self.assertEqual([x['max_switches'] for x in self.puts], [9, 10])
        self.assertNotIn('controller_pending', self.read_desired())

    def test_applied_put_timeout_is_read_back_without_duplicate_credit(self):
        self.fail_put = 'after'
        with self.assertRaises(TimeoutError):
            admission.tick(self.config)
        self.assertEqual(len(self.puts), 1)
        self.assertIn('controller_pending', self.read_desired())
        self.fail_put = None
        result = admission.tick(self.config)
        self.assertEqual(result['config']['max_switches'], 9)
        self.assertEqual(len(self.puts), 1)
        self.assertNotIn('controller_pending', self.read_desired())

    def test_unapplied_put_timeout_requires_readback_before_retry(self):
        self.fail_put = 'before'
        with self.assertRaises(TimeoutError):
            admission.tick(self.config)
        self.assertEqual(self.migration['config']['max_switches'], 8)
        self.fail_put = None
        result = admission.tick(self.config)
        self.assertEqual(result['config']['max_switches'], 9)
        self.assertEqual([x['max_switches'] for x in self.puts], [9, 9])

    def test_error_latch_survives_control_status_and_repeated_ticks(self):
        self.feedback['candidate_errors'] = 18
        self.assertTrue(admission.tick(self.config)['error_latched'])
        self.feedback['candidate_errors'] = 17
        result = admission.tick(self.config)
        self.assertEqual(result['reason'], 'candidate_error_latched')
        self.assertTrue(result['error_latched'])
        self.assertEqual(self.migration['config']['max_switches'], 8)

    def test_missing_feedback_revokes_bridge_credit(self):
        self.migration['config']['max_switches'] = 40
        self.feedback_mock.side_effect = RuntimeError('prometheus unavailable')
        result = admission.tick(self.config)
        self.assertEqual(result['reason'], 'feedback_unavailable')
        self.assertEqual(self.migration['config']['max_switches'], 8)
        self.assertEqual(self.migration['config']['new_sessions_bps'], 0)

    def test_incomplete_rollout_cannot_issue_candidate_credit(self):
        self.rollout['candidate_new_only'] = False
        result = admission.tick(self.config)
        self.assertEqual(result['reason'], 'feedback_unavailable')
        self.assertEqual(self.migration['config']['max_switches'], 8)

    def test_second_controller_instance_cannot_run(self):
        path = Path(self.config['singleton_lock'])
        config_file = path.with_name('config.json')
        config_file.write_text(json.dumps(self.config))
        with path.open('a') as lock:
            admission.fcntl.flock(lock, admission.fcntl.LOCK_EX | admission.fcntl.LOCK_NB)
            with mock.patch.object(sys, 'argv', ['controller', '--config', str(config_file), '--once']):
                with self.assertRaises(BlockingIOError):
                    admission.main()
        self.feedback_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
