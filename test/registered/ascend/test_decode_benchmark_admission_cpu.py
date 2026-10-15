"""CPU tests executing production admission/state methods through AST extraction.

Run directly with Python; no sglang, torch, or device modules are imported.
"""
import ast
import os
from pathlib import Path
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[3]
SRT = ROOT / 'python/sglang/srt'
GATE = 'benchmark_decode_admission_hold'
MODES = SimpleNamespace(DECODE='decode', PREFILL='prefill', NULL='null')


def extract(path, name, namespace):
    tree = ast.parse(path.read_text(), filename=str(path))
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(functions) != 1:
        raise AssertionError('expected one production method: ' + name)
    node = functions[0]
    node.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module='__future__',
                        names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return namespace[name]


class Batch:
    def __init__(self, reqs=()):
        self.reqs = list(reqs)

    def is_empty(self):
        return not self.reqs

    def batch_size(self):
        return len(self.reqs)


class TestDecodeBenchmarkAdmissionCPU(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {'SGLANG_PD_DECODE_EXEC_MAX_REQUESTS_PER_DP': '10'})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.context = SimpleNamespace(
            override=MagicMock(), resolved_server_args_dict=lambda: {'startup_setting': 7})
        self.batch_factory = MagicMock()
        self.time_batch = MagicMock()
        self.ns = {
            'os': os, 'logging': MagicMock(), 'logger': MagicMock(),
            'DisaggregationMode': MODES, 'ScheduleBatch': self.batch_factory,
            'get_context': lambda: self.context,
            'get_disagg': lambda: SimpleNamespace(disaggregation_decode_enable_radix_cache=False),
            'get_parallel': lambda: SimpleNamespace(enable_dp_attention=True, dp_size=8, tp_size=16, pp_size=1),
            'get_exec': lambda: SimpleNamespace(moe=SimpleNamespace(elastic_ep_backend=None)),
            'compute_world_size': lambda **_: 16, 'build_memory_usage': lambda **_: {},
            'msgspec_to_builtins': lambda v: v, 'RECORD_STEP_TIME': False,
            'envs': SimpleNamespace(SGLANG_EXPOSE_OWN_ENV_VARS=SimpleNamespace(get=lambda: False)),
            'SetInternalStateReqOutput': lambda **kw: SimpleNamespace(**kw),
            'GetInternalStateReqOutput': lambda **kw: SimpleNamespace(**kw),
            'set_time_batch': self.time_batch,
        }
        decode = SRT / 'disaggregation/decode.py'
        scheduler = SRT / 'managers/scheduler.py'
        extract(decode, 'get_decode_execution_max_requests_per_dp', self.ns)
        for path, name in ((decode, 'get_new_prebuilt_batch'),
                           (scheduler, 'get_internal_state'), (scheduler, 'set_internal_state')):
            extract(path, name, self.ns)

    def scheduler(self, mode='decode', running=(), waiting=(), dspark=False):
        worker = SimpleNamespace(graph_memory_usage=0,
            set_dspark_forced_budget_frac=MagicMock(), clear_info_records=MagicMock(),
            dump_info_records=lambda: None)
        scheduler = SimpleNamespace(
            disaggregation_mode=mode, running_batch=Batch(running), waiting_queue=list(waiting),
            grammar_manager=SimpleNamespace(has_waiting_grammars=lambda: False),
            enable_priority_scheduling=False, req_to_token_pool=SimpleNamespace(size=11),
            max_running_requests=11, tree_cache=object(), model_config=object(),
            token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: SimpleNamespace(mem_usage=0)),
            enable_overlap=False, future_map={}, ps=SimpleNamespace(pp_size=1),
            spec_algorithm=SimpleNamespace(is_dspark=lambda: dspark, is_none=lambda: not dspark),
            metrics_reporter=SimpleNamespace(spec_total_num_forward_ct=0,
                spec_total_num_accept_tokens=0, last_gen_throughput=0),
            draft_worker=worker, tp_worker=SimpleNamespace(graph_memory_usage=0,
                model_runner=SimpleNamespace(weight_load_mem_usage=0)),
            startup_available_gpu_memory_gb=0, max_total_num_tokens=100,
            swa_tokens_per_layer=None, startup_time=0)
        for name in ('get_new_prebuilt_batch', 'get_internal_state', 'set_internal_state'):
            setattr(scheduler, name, MethodType(self.ns[name], scheduler))
        return scheduler

    @staticmethod
    def ready_request():
        return SimpleNamespace(last_node=None, kv=SimpleNamespace(kv_committed_len=None),
                               init_next_round_input=MagicMock())

    @staticmethod
    def set_values(scheduler, values):
        return scheduler.set_internal_state(SimpleNamespace(server_args=values)).updated

    def test_hold_preserves_ready_queue_and_prevents_prebuilt_mutations(self):
        ready = self.ready_request()
        scheduler = self.scheduler(waiting=[ready])
        self.assertTrue(self.set_values(scheduler, {GATE: 1}))
        original_queue = scheduler.waiting_queue
        self.assertIsNone(scheduler.get_new_prebuilt_batch(Batch()))
        self.assertIs(scheduler.waiting_queue, original_queue)
        self.assertEqual(scheduler.waiting_queue, [ready])
        ready.init_next_round_input.assert_not_called()
        self.batch_factory.init_new.assert_not_called()
        self.time_batch.assert_not_called()

    def test_hold_does_not_enqueue_grammar_ready_requests(self):
        ready, grammar_ready = self.ready_request(), self.ready_request()
        scheduler = self.scheduler(waiting=[ready])
        scheduler.grammar_manager = SimpleNamespace(has_waiting_grammars=lambda: True,
            get_ready_grammar_requests=MagicMock(return_value=[grammar_ready]))
        scheduler._add_request_to_queue = MagicMock(side_effect=scheduler.waiting_queue.append)
        self.assertTrue(self.set_values(scheduler, {GATE: 1}))
        self.assertIsNone(scheduler.get_new_prebuilt_batch(Batch()))
        self.assertEqual(scheduler.waiting_queue, [ready])
        scheduler._add_request_to_queue.assert_not_called()
        self.batch_factory.init_new.assert_not_called()

    def test_default_and_explicit_off_keep_empty_queue_behavior(self):
        for explicit in (False, True):
            with self.subTest(explicit=explicit):
                scheduler = self.scheduler()
                if explicit:
                    self.assertTrue(self.set_values(scheduler, {GATE: 0}))
                else:
                    self.assertFalse(hasattr(scheduler, '_benchmark_decode_admission_hold'))
                self.assertIsNone(scheduler.get_new_prebuilt_batch(Batch()))
                self.assertEqual(scheduler.waiting_queue, [])
        self.batch_factory.init_new.assert_not_called()

    def test_decode_api_accepts_only_integer_zero_or_one(self):
        for invalid in (True, False, 0.0, 1.0, -1, 2, '0', '1', None, [], {}):
            with self.subTest(value=invalid):
                scheduler = self.scheduler()
                self.assertFalse(self.set_values(scheduler, {GATE: invalid}))
                self.assertFalse(hasattr(scheduler, '_benchmark_decode_admission_hold'))
        for mode in (MODES.PREFILL, MODES.NULL):
            for value in (0, 1):
                with self.subTest(mode=mode, value=value):
                    scheduler = self.scheduler(mode=mode)
                    self.assertFalse(self.set_values(scheduler, {GATE: value}))
        self.context.override.assert_not_called()

    def test_enabling_with_running_requests_is_rejected_but_release_succeeds(self):
        scheduler = self.scheduler(running=[object()])
        self.assertFalse(self.set_values(scheduler, {GATE: 1}))
        self.assertFalse(hasattr(scheduler, '_benchmark_decode_admission_hold'))
        scheduler._benchmark_decode_admission_hold = True
        self.assertTrue(self.set_values(scheduler, {GATE: 0}))
        self.assertIs(scheduler._benchmark_decode_admission_hold, False)

    def test_release_admits_requests_that_became_ready_while_held(self):
        scheduler = self.scheduler()
        self.assertTrue(self.set_values(scheduler, {GATE: 1}))
        ready = self.ready_request()
        scheduler.waiting_queue.append(ready)
        self.assertTrue(self.set_values(scheduler, {GATE: 0}))
        admitted = scheduler.get_new_prebuilt_batch(Batch())
        self.assertIs(admitted, self.batch_factory.init_new.return_value)
        self.assertEqual(scheduler.waiting_queue, [])
        ready.init_next_round_input.assert_called_once_with(scheduler.tree_cache)
        self.assertEqual(self.batch_factory.init_new.call_args.args[0], [ready])
        admitted.prepare_for_prebuilt.assert_called_once_with()
        admitted.process_prebuilt.assert_called_once_with(scheduler.future_map)

    def test_unknown_setting_rejects_entire_update_without_hold_mutation(self):
        for initial, value in ((False, 1), (True, 0)):
            for unknown_first in (False, True):
                with self.subTest(initial=initial, unknown_first=unknown_first):
                    scheduler = self.scheduler()
                    scheduler._benchmark_decode_admission_hold = initial
                    values = {'unknown_setting': 3, GATE: value} if unknown_first else {GATE: value, 'unknown_setting': 3}
                    scheduler.metrics_reporter.spec_total_num_forward_ct = 7
                    self.assertFalse(self.set_values(scheduler, values))
                    self.assertIs(scheduler._benchmark_decode_admission_hold, initial)
                    self.assertEqual(scheduler.metrics_reporter.spec_total_num_forward_ct, 7)
        self.context.override.assert_not_called()

    def test_gate_readback_is_runtime_state_and_never_context_override(self):
        scheduler = self.scheduler(waiting=[self.ready_request()])
        self.assertIs(scheduler.get_internal_state(None).internal_state[GATE], False)
        self.assertTrue(self.set_values(scheduler, {GATE: 1, 'speculative_accept_threshold_single': 0.7}))
        self.context.override.assert_called_once_with(source='update_server_args',
                                                     speculative_accept_threshold_single=0.7)
        state = scheduler.get_internal_state(None).internal_state
        self.assertIs(state[GATE], True)
        self.assertEqual(state['benchmark_decode_ready_requests'], 1)
        self.assertEqual(state['effective_decode_execution_max_requests_per_dp'], 10)
        self.assertNotIn(GATE, self.context.resolved_server_args_dict())
        for mode in (MODES.PREFILL, MODES.NULL):
            self.assertNotIn(GATE, self.scheduler(mode=mode).get_internal_state(None).internal_state)

    def test_dspark_fraction_validation_still_rejects_invalid_combined_updates(self):
        for fraction in (-0.1, 0, 1.01, float('inf'), float('nan')):
            for gate_first in (False, True):
                with self.subTest(fraction=fraction, gate_first=gate_first):
                    scheduler = self.scheduler(dspark=True)
                    values = {GATE: 1, 'dspark_force_budget_frac': fraction} if gate_first else {'dspark_force_budget_frac': fraction, GATE: 1}
                    self.assertFalse(self.set_values(scheduler, values))
                    self.assertFalse(hasattr(scheduler, '_benchmark_decode_admission_hold'))
                    scheduler.draft_worker.set_dspark_forced_budget_frac.assert_not_called()
        self.context.override.assert_not_called()

    def test_dspark_fraction_valid_values_and_worker_requirement(self):
        for fraction in (None, 0.25, 1):
            with self.subTest(fraction=fraction):
                scheduler = self.scheduler(dspark=True)
                self.assertTrue(self.set_values(scheduler, {GATE: 1, 'dspark_force_budget_frac': fraction}))
                scheduler.draft_worker.set_dspark_forced_budget_frac.assert_called_once_with(fraction)
                self.assertIs(scheduler._benchmark_decode_admission_hold, True)
        scheduler = self.scheduler(dspark=False)
        self.assertFalse(self.set_values(scheduler, {GATE: 1, 'dspark_force_budget_frac': 0.5}))
        self.assertFalse(hasattr(scheduler, '_benchmark_decode_admission_hold'))
        self.context.override.assert_not_called()


if __name__ == '__main__':
    unittest.main()
