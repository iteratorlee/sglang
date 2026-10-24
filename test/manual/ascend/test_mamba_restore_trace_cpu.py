"""CPU-only trace contract; imports no model or accelerator runtime."""

import importlib.util
import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

PATH = (
    Path(__file__).resolve().parents[3]
    / "python/sglang/srt/mem_cache/mamba_restore_trace.py"
)


def load_trace(**env):
    spec = importlib.util.spec_from_file_location("restore_trace_test", PATH)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(os.environ, env, clear=True):
        spec.loader.exec_module(module)
    return module


class TraceTests(unittest.TestCase):
    def test_disabled_does_not_consult_request_or_ranks(self):
        trace = load_trace()
        with patch.object(trace, "_ranks", side_effect=AssertionError):
            self.assertFalse(trace.trace_enabled(object()))
            trace.trace_event(object(), "match")

    def test_rid_filter_leader_and_limit(self):
        trace = load_trace(
            SGLANG_MAMBA_RESTORE_TRACE="1",
            SGLANG_MAMBA_RESTORE_TRACE_RID_PREFIX="sample-",
            SGLANG_MAMBA_RESTORE_TRACE_MAX_EVENTS="2",
        )
        with patch.object(trace, "_ranks", return_value=(1, 0, 0, "prefill")):
            with self.assertLogs(trace.logger, "INFO") as logs:
                trace.trace_event(NS(rid="other"), "match")
                for _ in range(3):
                    trace.trace_event(
                        NS(rid='sample-\n"x'), "match", full_kv_depth=10561
                    )
            rows = [json.loads(r.getMessage().split(" ", 1)[1]) for r in logs.records]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["rid"], 'sample-\n"x')
            self.assertEqual(rows[0]["dp"], 1)
            self.assertEqual(rows[1]["seq"], 2)
        trace = load_trace(SGLANG_MAMBA_RESTORE_TRACE="1")
        for ranks in ((1, 1, 0, "prefill"), (1, 0, 1, "prefill")):
            with patch.object(trace, "_ranks", return_value=ranks):
                with self.assertNoLogs(trace.logger, "INFO"):
                    trace.trace_event(NS(rid="sample"), "track")
        self.assertEqual(trace._emitted, 0)

    def test_insert_reports_actual_truncation_and_decline(self):
        trace = load_trace(SGLANG_MAMBA_RESTORE_TRACE="1")
        req = NS(
            rid="r",
            mamba_branching_seqlen=2304,
            kv=NS(mamba_last_track_seqlen=2304, cache_protected_len=0),
        )
        result = NS(rotation_tail_declined=False, mamba_exist=False, last_device_node=7)
        with patch.object(trace, "_ranks", return_value=(0, 0, 0, "prefill")):
            with self.assertLogs(trace.logger, "INFO") as logs:
                trace.trace_insert(
                    req,
                    result,
                    requested_depth=7141,
                    actual_depth=2304,
                    finished=False,
                    chunked=False,
                )
                result.rotation_tail_declined = True
                trace.trace_insert(
                    req,
                    result,
                    requested_depth=7141,
                    actual_depth=2304,
                    finished=True,
                    chunked=False,
                )
        rows = [json.loads(r.getMessage().split(" ", 1)[1]) for r in logs.records]
        self.assertEqual(rows[0]["actual_insert_depth"], 2304)
        self.assertEqual(rows[0]["requested_depth"], 7141)
        self.assertEqual(rows[0]["last_track"], 2304)
        self.assertIsNone(rows[1]["actual_insert_depth"])


if __name__ == "__main__":
    unittest.main()
