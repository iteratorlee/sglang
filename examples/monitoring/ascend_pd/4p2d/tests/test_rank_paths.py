import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import test_lengths as length_tests
from test_native_regression import monitor, req

spec = importlib.util.spec_from_file_location("real_cpu_rank_paths", Path(__file__).parent / "fixtures/decode_rank_paths.py")
paths = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paths)


class ResolvedRankPaths(unittest.TestCase):
    def setUp(self):
        paths._bootstrap_addr = lambda r: "prefill-0"
        paths.envs = NS(SGLANG_DISAGGREGATION_FORCE_QUERY_PREFILL_DP_RANK=NS(get=lambda: False))
        self.queue = paths.DecodeRankPaths()
        self.queue.kv_manager = NS(prefill_info_table={"prefill-0": NS(dp_size=4,follow_bootstrap_room=False)})

    def test_actual_req_field_all_four_global_p_ranks(self):
        for rank in range(4):
            r = req(); r.disagg_prefill_dp_rank = rank
            self.assertEqual(self.queue._resolve_prefill_dp_rank(r), rank)

    def test_real_slow_path_queries_rank_and_preserves_it_on_terminal_req(self):
        helper = length_tests.LengthTests(); m, registry = helper.observer()
        r = req(); r.disagg_prefill_dp_rank = None; r.bootstrap_room = 101
        initialized = []
        entry = NS(req=r, kv_receiver=NS(init=initialized.append))
        self.queue.pending_reqs = [entry]; self.queue._prefill_dp_rank_queries = {}
        self.queue._ensure_prefill_info = lambda groups: (groups, [])
        calls = []
        def query(addr, rooms):
            calls.append((addr, rooms)); return {"101": 3}
        paths.CommonKVReceiver = NS(query_prefill_dp_ranks=query)
        paths._pd_mon = monitor
        with patch.object(monitor,"_observer",m):
            self.queue._resolve_pending_reqs()
        self.assertEqual(calls, [("prefill-0", [101])])
        self.assertEqual(initialized,[3]); self.assertEqual(r._pd_metrics_prefill_dp_rank,"3")
        self.assertIsNone(r.disagg_prefill_dp_rank)  # Hook does not alter routing.
        m.finished(r)
        self.assertEqual(helper.total(registry,suffix="count",prefill_dp_rank="3"),1)

    def test_unresolved_real_path_remains_pending_then_cancel_is_unknown(self):
        helper = length_tests.LengthTests(); m, registry = helper.observer()
        r = req(); r.disagg_prefill_dp_rank = None; r.bootstrap_room = 101
        entry = NS(req=r,kv_receiver=NS(init=lambda rank:self.fail("unresolved request initialized")))
        self.queue.pending_reqs = [entry]; self.queue._prefill_dp_rank_queries = {}
        self.queue._ensure_prefill_info = lambda groups:(groups,[])
        paths.CommonKVReceiver = NS(query_prefill_dp_ranks=lambda *args:{})
        paths._pd_mon = monitor
        with patch.object(monitor,"_observer",m):
            self.queue._resolve_pending_reqs()
        self.assertEqual(self.queue.pending_reqs,[entry])
        m.aborted(r, NS())
        self.assertEqual(helper.total(registry,suffix="count",prefill_dp_rank="unknown",status="cancelled"),1)

    def test_bootstrap_cache_resolution_has_no_invented_default_rank(self):
        r = req(); r.disagg_prefill_dp_rank = None; r.bootstrap_room = 7
        self.assertIsNone(self.queue._resolve_prefill_dp_rank(r))
        self.queue.kv_manager.prefill_info_table["prefill-0"].follow_bootstrap_room=True
        self.assertEqual(self.queue._resolve_prefill_dp_rank(r),3)
