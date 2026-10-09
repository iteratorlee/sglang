"""CPU-only tests for the Mooncake sender's optional transfer measurements."""

import ast
import concurrent.futures
import importlib
import importlib.util
import json
import logging
import tempfile
import threading
import time
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np


# Prefer the canonical module so connection tests share its ContextVar. On a
# minimal CPU host, load this dependency-free file without the full SGLang tree.
try:
    telemetry = importlib.import_module(
        "sglang.srt.disaggregation.mooncake.transfer_telemetry"
    )
    _CAN_IMPORT_SGLANG = True
except ImportError:
    _CAN_IMPORT_SGLANG = False
    _MODULE_PATH = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/disaggregation/mooncake/transfer_telemetry.py"
    )
    _SPEC = importlib.util.spec_from_file_location(
        "mooncake_transfer_telemetry_test", _MODULE_PATH
    )
    telemetry = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(telemetry)


def _load_connection_classes():
    """Use production imports, or compile the exact sender methods on CPU.

    The fallback strips only unrelated server imports; its transfer and Ascend
    override bodies are parsed from the current source tree, never copied here.
    """
    if _CAN_IMPORT_SGLANG:
        try:
            from sglang.srt.disaggregation.ascend.conn import AscendKVManager
            from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager

            return MooncakeKVManager, AscendKVManager
        except ImportError:
            pass
    repo = Path(__file__).resolve().parents[4]

    def method_nodes(path, class_name, names):
        tree = ast.parse(path.read_text(), filename=str(path))
        cls = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        return [
            node
            for node in cls.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in names
        ]

    def function_node(path, name):
        tree = ast.parse(path.read_text(), filename=str(path))
        return next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )

    mooncake_path = repo / "python/sglang/srt/disaggregation/mooncake/conn.py"
    ascend_path = repo / "python/sglang/srt/disaggregation/ascend/conn.py"
    common_path = repo / "python/sglang/srt/disaggregation/common/utils.py"
    utils_path = repo / "python/sglang/srt/disaggregation/utils.py"
    mooncake = ast.ClassDef(
        name="MooncakeKVManager",
        bases=[],
        keywords=[],
        body=method_nodes(
            mooncake_path,
            "MooncakeKVManager",
            {
                "_transfer_data",
                "_send_kvcache_generic",
                "_send_mamba_state",
                "_await_transfer_futures",
                "send_kvcache_slice",
            },
        ),
        decorator_list=[],
    )
    ascend = ast.ClassDef(
        name="AscendKVManager",
        bases=[ast.Name(id="MooncakeKVManager", ctx=ast.Load())],
        keywords=[],
        body=method_nodes(ascend_path, "AscendKVManager", {"send_kvcache"}),
        decorator_list=[],
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            function_node(common_path, "group_concurrent_contiguous"),
            function_node(utils_path, "build_transfer_entry_pairs"),
            mooncake,
            ascend,
        ],
        type_ignores=[],
    )
    scope = {
        "np": np,
        "deque": deque,
        "concurrent": concurrent,
        "logger": logging.getLogger(__name__),
        "current_transfer_scope": telemetry.current_scope,
        "classify_kv_blocks": telemetry.classify_kv_blocks,
        "typed_transfer": telemetry.typed_transfer,
        "typed_engine_transfer": telemetry.typed_engine_transfer,
        "transfer_submitter": telemetry.transfer_submitter,
    }
    exec(
        compile(ast.fix_missing_locations(module), str(mooncake_path), "exec"),
        scope,
    )
    return scope["MooncakeKVManager"], scope["AscendKVManager"]


class MockEngine:
    def __init__(self, results=None):
        self.results = iter(results or [])
        self.calls = []

    def batch_transfer_sync(self, session, src, dst, lengths):
        self.calls.append((session, src, dst, lengths))
        outcome = next(self.results, 0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class TestMooncakeTransferTelemetry(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.writer = telemetry.TransferTelemetry(
            str(Path(self.tmp.name) / "pd"), rank=3
        )
        self.addCleanup(lambda: __import__("os").close(self.writer._fd))

    def rows(self):
        return [
            json.loads(line) for line in Path(self.writer.path).read_text().splitlines()
        ]

    def test_two_chunks_types_executor_and_duplicate_finish(self):
        engine = MockEngine()
        first = self.writer.begin(17, slice(0, 64), False, "req-17")
        first.dequeued()
        token = telemetry.activate(first)
        try:
            self.assertEqual(
                telemetry.typed_engine_transfer(
                    engine, "peer", [100], [200], [10], "kv_target"
                ),
                0,
            )
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future = telemetry.submit_transfer(
                    executor,
                    telemetry.typed_engine_transfer,
                    engine,
                    "peer",
                    [300],
                    [400],
                    [7],
                    "kv_draft",
                )
                self.assertEqual(future.result(), 0)
            first.completed_tcp(time.monotonic_ns(), 3, False)
        finally:
            telemetry.deactivate(token)
        first.finish()
        first.finish(failed=True)  # duplicate notify must not duplicate or flip the row

        second = self.writer.begin(17, slice(64, 96), True)
        second.dequeued()
        second.requeued()
        second.dequeued()
        token = telemetry.activate(second)
        try:
            telemetry.typed_engine_transfer(engine, "peer", [500], [600], [5], "mamba")
        finally:
            telemetry.deactivate(token)
        second.finish()

        rows = self.rows()
        self.assertEqual(len(rows), 2)
        self.assertEqual([r["chunk_id"] for r in rows], [0, 1])
        self.assertEqual([r["bytes"] for r in rows], [20, 5])
        self.assertEqual(rows[0]["by_type"]["kv_target"]["bytes"], 10)
        self.assertEqual(rows[0]["by_type"]["kv_draft"]["bytes"], 7)
        self.assertEqual(rows[0]["by_type"]["aux_tcp"]["bytes"], 3)
        self.assertEqual(rows[1]["by_type"]["mamba"]["blocks"], 1)
        self.assertEqual(rows[0]["status"], "done")
        self.assertEqual(rows[0]["rid"], "req-17")
        self.assertEqual(rows[1]["defer_count"], 1)
        self.assertGreaterEqual(
            rows[0]["submit_ts_mono_ns"], rows[0]["dequeue_ts_mono_ns"]
        )
        self.assertGreaterEqual(rows[0]["submit_ts_wall_ns"], 1)
        self.assertGreaterEqual(rows[0]["engine_call_ms_sum"], 0)

    def test_failed_engine_result_and_exception_keep_attempted_bytes(self):
        engine = MockEngine([0, -1, RuntimeError("engine broke")])
        scope = self.writer.begin(18, slice(0, 8), True)
        scope.dequeued()
        token = telemetry.activate(scope)
        try:
            for length in (4, 6):
                telemetry.typed_engine_transfer(
                    engine, "peer", [100], [200], [length], "dsa"
                )
            with self.assertRaisesRegex(RuntimeError, "engine broke"):
                telemetry.typed_engine_transfer(
                    engine, "peer", [100], [200], [9], "mamba"
                )
        finally:
            telemetry.deactivate(token)
        scope.finish()
        row = self.rows()[0]
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["bytes"], 19)
        self.assertEqual(row["ok_bytes"], 4)
        self.assertEqual(row["blocks"], 3)
        self.assertEqual(row["by_type"]["dsa"]["calls"], 2)
        self.assertEqual(row["by_type"]["mamba"]["ok_bytes"], 0)

    def test_executor_queue_wait_is_separate_from_engine_call(self):
        engine = MockEngine()
        occupied = threading.Event()
        release = threading.Event()

        def blocker():
            occupied.set()
            release.wait(timeout=2)

        scope = self.writer.begin(20, slice(0, 1), False)
        scope.dequeued()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(blocker)
            self.assertTrue(occupied.wait(timeout=2))
            token = telemetry.activate(scope)
            try:
                future = telemetry.submit_transfer(
                    executor,
                    telemetry.typed_engine_transfer,
                    engine,
                    "peer",
                    [1],
                    [2],
                    [5],
                    "kv_target",
                )
            finally:
                telemetry.deactivate(token)
            time.sleep(0.02)
            release.set()
            self.assertEqual(future.result(), 0)
        scope.finish()
        row = self.rows()[0]
        self.assertGreaterEqual(row["executor_wait_ms_sum"], 10)
        self.assertEqual(row["by_type"]["kv_target"]["bytes"], 5)
        self.assertLess(row["engine_call_ms_sum"], row["executor_wait_ms_sum"])

    def test_disabled_path_passes_original_callable_and_starts_no_writer_thread(self):
        engine = MockEngine()
        self.assertIsNone(telemetry.current_scope())
        before = set(threading.enumerate())
        self.assertEqual(
            telemetry.typed_engine_transfer(engine, "peer", [1], [2], [3], "kv_target"),
            0,
        )
        executor = Mock()
        fn = Mock(return_value=0)
        self.assertEqual(telemetry.transfer_submitter(executor), executor.submit)
        telemetry.submit_transfer(executor, fn, 7)
        executor.submit.assert_called_once_with(fn, 7)
        self.assertEqual(set(threading.enumerate()), before)

    def test_connection_uses_real_engine_blocks_for_target_draft_and_mamba(self):
        MooncakeKVManager, _ = _load_connection_classes()

        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager.engine = MockEngine()
        manager.kv_args = SimpleNamespace(
            num_draft_entries=1,
            kv_data_ptrs=[1000, 2000],
        )
        manager.is_mla_backend = True
        manager.is_hybrid_mla_backend = False
        manager.pp_size = 1
        manager.enable_custom_mem_pool = False
        manager.max_transfer_batch_indices = 0
        manager.get_mla_kv_ptrs_with_pp = lambda src, dst, state_type=None: (
            src,
            dst,
            len(src),
        )
        scope = self.writer.begin(19, slice(0, 2), True)
        scope.dequeued()
        token = telemetry.activate(scope)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                self.assertEqual(
                    manager._send_kvcache_generic(
                        "peer",
                        [1000, 2000],
                        [5000, 6000],
                        [10, 4],
                        np.array([0, 1], dtype=np.int32),
                        np.array([10, 11], dtype=np.int32),
                        executor,
                    ),
                    0,
                )
                self.assertEqual(
                    manager._send_kvcache_generic(
                        "peer",
                        [3000],
                        [7000],
                        [3],
                        np.array([0, 1], dtype=np.int32),
                        np.array([10, 11], dtype=np.int32),
                        executor,
                        state_type=SimpleNamespace(value="dsa"),
                    ),
                    0,
                )
            self.assertEqual(
                manager._send_mamba_state(
                    SimpleNamespace(mooncake_session_id="peer"),
                    [0],
                    [3000],
                    [6],
                    [7000],
                    [0],
                ),
                0,
            )
        finally:
            telemetry.deactivate(token)
        scope.finish()
        row = self.rows()[0]
        self.assertEqual(row["by_type"]["kv_target"]["bytes"], 20)
        self.assertEqual(row["by_type"]["kv_draft"]["bytes"], 8)
        self.assertEqual(row["by_type"]["dsa"]["bytes"], 6)
        self.assertEqual(row["by_type"]["mamba"]["bytes"], 6)
        self.assertEqual(row["bytes"], 40)

    def test_ascend_override_tracks_executor_and_source_ranges(self):
        _, AscendKVManager = _load_connection_classes()

        manager = AscendKVManager.__new__(AscendKVManager)
        manager.engine = MockEngine()
        manager.kv_args = SimpleNamespace(
            kv_data_ptrs=[1000, 2000],
            kv_data_lens=[100, 100],
            kv_item_lens=[10, 4],
            num_draft_entries=1,
        )
        manager.pp_size = 1
        manager.enable_custom_mem_pool = True
        manager._validate_envelope_kv_layout = lambda *args: None
        scope = self.writer.begin(21, slice(0, 2), True)
        scope.dequeued()
        token = telemetry.activate(scope)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                proxy = telemetry.TelemetryExecutorProxy(executor)
                self.assertEqual(
                    manager.send_kvcache(
                        "peer",
                        np.array([0, 1], dtype=np.int32),
                        [5000, 6000],
                        np.array([10, 11], dtype=np.int32),
                        proxy,
                    ),
                    0,
                )
                proxy.drain()
        finally:
            telemetry.deactivate(token)
        scope.finish()
        row = self.rows()[0]
        self.assertEqual(row["by_type"]["kv_target"]["bytes"], 20)
        self.assertEqual(row["by_type"]["kv_draft"]["bytes"], 8)
        self.assertEqual(row["blocks"], 2)
        self.assertEqual(len(manager.engine.calls), 2)

    def test_bounded_index_batches_keep_per_type_counts(self):
        MooncakeKVManager, _ = _load_connection_classes()
        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager.engine = MockEngine()
        manager.kv_args = SimpleNamespace(
            num_draft_entries=1,
            kv_data_ptrs=[1000, 2000],
        )
        manager.is_mla_backend = True
        manager.is_hybrid_mla_backend = False
        manager.pp_size = 1
        manager.enable_custom_mem_pool = False
        manager.max_transfer_batch_indices = 1
        manager.get_mla_kv_ptrs_with_pp = lambda src, dst, state_type=None: (
            src,
            dst,
            len(src),
        )
        scope = self.writer.begin(24, slice(0, 2), False)
        scope.dequeued()
        token = telemetry.activate(scope)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                self.assertEqual(
                    manager._send_kvcache_generic(
                        "peer",
                        [1000, 2000],
                        [5000, 6000],
                        [10, 4],
                        np.array([0, 2], dtype=np.int32),
                        np.array([10, 12], dtype=np.int32),
                        executor,
                    ),
                    0,
                )
        finally:
            telemetry.deactivate(token)
        scope.finish()
        row = self.rows()[0]
        self.assertEqual(row["by_type"]["kv_target"]["bytes"], 20)
        self.assertEqual(row["by_type"]["kv_draft"]["bytes"], 8)
        self.assertEqual(row["blocks"], 4)
        self.assertEqual(len(manager.engine.calls), 2)

    def test_ambiguous_kv_ranges_are_reported(self):
        manager = SimpleNamespace(
            kv_args=SimpleNamespace(
                kv_data_ptrs=[1000, 1000],
                kv_data_lens=[100, 100],
                num_draft_entries=1,
            )
        )
        self.assertEqual(
            telemetry.classify_kv_blocks(manager, [1010, 5000], [5, 5]),
            ["kv_unclassified", "kv_unclassified"],
        )

    def test_tp_slice_direct_engine_path_counts_each_block_once(self):
        MooncakeKVManager, _ = _load_connection_classes()
        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager.engine = MockEngine()
        manager.kv_args = SimpleNamespace(
            engine_rank=0,
            kv_item_lens=[8, 8],
            kv_data_ptrs=[1000, 2000],
            kv_layer_ids=[0, 1],
            num_draft_entries=1,
            page_size=1,
            total_kv_head_num=1,
            kv_head_num=1,
        )
        manager.attn_tp_size = 2
        manager.pp_size = 1
        manager.enable_deferred_decode_kv_release = True
        scope = self.writer.begin(22, slice(0, 2), False)
        scope.dequeued()
        token = telemetry.activate(scope)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                self.assertEqual(
                    manager.send_kvcache_slice(
                        "peer",
                        np.array([0, 1], dtype=np.int32),
                        [5000, 6000],
                        np.array([10, 11], dtype=np.int32),
                        dst_tp_rank=0,
                        dst_attn_tp_size=1,
                        dst_kv_item_len=8,
                        executor=executor,
                        dst_layer_ids=[0, 1],
                    ),
                    0,
                )
        finally:
            telemetry.deactivate(token)
        scope.finish()
        row = self.rows()[0]
        self.assertEqual(row["by_type"]["kv_target"]["bytes"], 16)
        self.assertEqual(row["by_type"]["kv_draft"]["bytes"], 16)
        self.assertEqual(row["bytes"], 32)
        self.assertEqual(row["blocks"], 4)
        self.assertEqual(len(manager.engine.calls), 2)

    def test_ascend_early_failure_drains_only_already_submitted_futures(self):
        _, AscendKVManager = _load_connection_classes()
        draft_started = threading.Event()
        release_draft = threading.Event()

        class FailingEngine(MockEngine):
            def batch_transfer_sync(self, session, src, dst, lengths):
                self.calls.append((session, src, dst, lengths))
                if src[0] == 1000:
                    self.assert_draft_started()
                    return -1
                draft_started.set()
                release_draft.wait(timeout=2)
                return 0

            def assert_draft_started(self):
                if not draft_started.wait(timeout=2):
                    raise AssertionError("draft future did not start")

        manager = AscendKVManager.__new__(AscendKVManager)
        manager.engine = FailingEngine()
        manager.kv_args = SimpleNamespace(
            kv_data_ptrs=[1000, 2000],
            kv_data_lens=[100, 100],
            kv_item_lens=[10, 4],
            num_draft_entries=1,
        )
        manager.pp_size = 1
        manager.enable_custom_mem_pool = True
        manager._validate_envelope_kv_layout = lambda *args: None
        scope = self.writer.begin(23, slice(0, 2), True)
        scope.dequeued()
        token = telemetry.activate(scope)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                proxy = telemetry.TelemetryExecutorProxy(executor)
                self.assertEqual(
                    manager.send_kvcache(
                        "peer",
                        np.array([0, 1], dtype=np.int32),
                        [5000, 6000],
                        np.array([10, 11], dtype=np.int32),
                        proxy,
                    ),
                    -1,
                )
                self.assertTrue(any(not future.done() for future in proxy._futures))
                submitted_futures = list(proxy._futures)
                release_draft.set()
                proxy.drain()
                self.assertTrue(all(future.done() for future in submitted_futures))
        finally:
            release_draft.set()
            telemetry.deactivate(token)
        scope.finish()
        row = self.rows()[0]
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["bytes"], 28)
        self.assertEqual(row["ok_bytes"], 8)


if __name__ == "__main__":
    unittest.main()
