"""CPU-only tests of the actual PD decode admission methods.

Run from the repo root: python test/manual/ascend/test_pd_decode_execution_cap_cpu.py
"""

import ast
import os
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


SOURCE = (
    Path(__file__).resolve().parents[3] / "python/sglang/srt/disaggregation/decode.py"
)
TREE = ast.parse(SOURCE.read_text(), filename=str(SOURCE))


def source_function(name, class_name=None):
    body = TREE.body
    if class_name:
        body = next(
            node.body
            for node in body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
    function = next(
        node for node in body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    function.decorator_list = []
    return function


class FakeBatch:
    def __init__(self, reqs=()):
        self.reqs = list(reqs)

    @classmethod
    def init_new(cls, reqs, *_args):
        return cls(reqs)

    def batch_size(self):
        return len(self.reqs)

    def is_empty(self):
        return not self.reqs

    def filter_batch(self):
        self.reqs = [req for req in self.reqs if not req.finished]

    def merge_batch(self, other):
        self.reqs.extend(other.reqs)

    def prepare_for_prebuilt(self):
        pass

    def process_prebuilt(self, _future_map):
        pass


globals_for_test = {
    "os": os,
    "ScheduleBatch": FakeBatch,
    "get_disagg": lambda: SimpleNamespace(
        disaggregation_decode_enable_radix_cache=False
    ),
    "set_time_batch": lambda *_args: None,
    "set_schedule_time_batch": lambda *_args: None,
    "NextBatchPlan": lambda **kwargs: SimpleNamespace(**kwargs),
}
methods = [
    source_function("get_decode_execution_max_requests_per_dp"),
    source_function("get_new_prebuilt_batch", "SchedulerDisaggregationDecodeMixin"),
    source_function(
        "get_next_disagg_decode_batch_to_run", "SchedulerDisaggregationDecodeMixin"
    ),
]
module = ast.Module(
    body=[
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        ),
        *methods,
    ],
    type_ignores=[],
)
exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), globals_for_test)


def request(name):
    return SimpleNamespace(
        rid=name,
        finished=False,
        kv=SimpleNamespace(kv_committed_len=None),
        init_next_round_input=MagicMock(),
    )


def scheduler(waiting):
    sched = SimpleNamespace(
        waiting_queue=list(waiting),
        grammar_manager=SimpleNamespace(has_waiting_grammars=lambda: False),
        enable_priority_scheduling=False,
        req_to_token_pool=SimpleNamespace(size=2),
        max_running_requests=2,
        tree_cache=object(),
        token_to_kv_pool_allocator=object(),
        model_config=object(),
        enable_overlap=False,
        spec_algorithm=object(),
        future_map={},
        chunked_req=None,
        batch_result_processor=SimpleNamespace(
            process_batch_result_prebuilt=lambda _batch: None
        ),
        dp_attn_adapter=SimpleNamespace(
            maybe_prepare_mlp_sync_batch=lambda batch: batch
        ),
        enable_hisparse=False,
    )
    sched.get_new_prebuilt_batch = types.MethodType(
        globals_for_test["get_new_prebuilt_batch"], sched
    )
    sched.get_next_disagg_decode_batch_to_run = types.MethodType(
        globals_for_test["get_next_disagg_decode_batch_to_run"], sched
    )

    def update_running_batch(batch):
        batch.filter_batch()
        return batch

    sched.update_running_batch = update_running_batch
    return sched


class TestPDDecodeExecutionCap(unittest.TestCase):
    def test_off_and_positive_bounds_keep_physical_capacity(self):
        for setting, expected in [(None, 2), ("0", 2), ("1", 1), ("2", 2), ("8", 2)]:
            with self.subTest(setting=setting), patch.dict(os.environ):
                name = "SGLANG_PD_DECODE_EXEC_MAX_REQUESTS_PER_DP"
                if setting is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = setting
                reqs = [request(str(i)) for i in range(3)]
                sched = scheduler(reqs)
                admitted = sched.get_new_prebuilt_batch(FakeBatch())
                self.assertEqual(admitted.reqs, reqs[:expected])
                self.assertEqual(sched.waiting_queue, reqs[expected:])
                for req in reqs[expected:]:
                    req.init_next_round_input.assert_not_called()

    def test_invalid_values_fail_before_queue_changes(self):
        for setting in ("-1", "abc", "1.5", ""):
            with (
                self.subTest(setting=setting),
                patch.dict(
                    os.environ, {"SGLANG_PD_DECODE_EXEC_MAX_REQUESTS_PER_DP": setting}
                ),
            ):
                req = request("ready")
                sched = scheduler([req])
                with self.assertRaisesRegex(ValueError, "non-negative integer"):
                    sched.get_new_prebuilt_batch(FakeBatch())
                self.assertEqual(sched.waiting_queue, [req])
                req.init_next_round_input.assert_not_called()
                sched.waiting_queue.clear()
                with self.assertRaisesRegex(ValueError, "non-negative integer"):
                    sched.get_new_prebuilt_batch(FakeBatch())

    @patch.dict(os.environ, {"SGLANG_PD_DECODE_EXEC_MAX_REQUESTS_PER_DP": "1"})
    def test_active_batch_at_cap_preserves_ready_queue(self):
        active, ready = request("active"), request("ready")
        sched = scheduler([ready])
        self.assertIsNone(sched.get_new_prebuilt_batch(FakeBatch([active])))
        self.assertEqual(sched.waiting_queue, [ready])
        ready.init_next_round_input.assert_not_called()

    @patch.dict(os.environ, {"SGLANG_PD_DECODE_EXEC_MAX_REQUESTS_PER_DP": "1"})
    def test_finished_request_releases_execution_slot(self):
        active, ready = request("active"), request("ready")
        sched = scheduler([ready])
        first = sched.get_next_disagg_decode_batch_to_run(FakeBatch([active]))
        self.assertEqual(first.running_batch.reqs, [active])
        self.assertEqual(sched.waiting_queue, [ready])

        active.finished = True
        drained = sched.get_next_disagg_decode_batch_to_run(first.running_batch)
        self.assertTrue(drained.running_batch.is_empty())
        self.assertIsNone(drained.batch_to_run)
        self.assertEqual(sched.waiting_queue, [ready])

        next_plan = sched.get_next_disagg_decode_batch_to_run(drained.running_batch)
        self.assertEqual(next_plan.running_batch.reqs, [ready])
        self.assertEqual(sched.waiting_queue, [])
        ready.init_next_round_input.assert_called_once_with(sched.tree_cache)

    @patch.dict(os.environ, {"SGLANG_PD_DECODE_EXEC_MAX_REQUESTS_PER_DP": "2"})
    def test_prebuilt_merge_respects_remaining_slot_and_fifo(self):
        active, first, second = (
            request(name) for name in ("active", "first", "second")
        )
        sched = scheduler([first, second])
        plan = sched.get_next_disagg_decode_batch_to_run(FakeBatch([active]))
        self.assertEqual(plan.running_batch.reqs, [active, first])
        self.assertEqual(sched.waiting_queue, [second])
        second.init_next_round_input.assert_not_called()

    @patch.dict(os.environ, {"SGLANG_PD_DECODE_EXEC_MAX_REQUESTS_PER_DP": "0"})
    def test_off_merges_to_existing_physical_limit(self):
        active, first, second = (
            request(name) for name in ("active", "first", "second")
        )
        sched = scheduler([first, second])
        plan = sched.get_next_disagg_decode_batch_to_run(FakeBatch([active]))
        self.assertEqual(plan.running_batch.reqs, [active, first])
        self.assertEqual(sched.waiting_queue, [second])


if __name__ == "__main__":
    unittest.main()
