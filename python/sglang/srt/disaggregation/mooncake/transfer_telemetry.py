"""Opt-in, per-chunk Mooncake sender measurements. No device synchronization."""

from __future__ import annotations

import contextvars
import concurrent.futures
import functools
import itertools
import json
import logging
import os
import socket
import threading
import time
from collections import defaultdict

logger = logging.getLogger(__name__)

_current = contextvars.ContextVar("mooncake_transfer_scope", default=None)


def current_scope():
    return _current.get()


def activate(scope):
    return _current.set(scope)


def deactivate(token):
    _current.reset(token)


def submit_transfer(executor, fn, *args):
    """Carry the chunk scope into a sender executor and measure its queue wait."""
    if isinstance(executor, TelemetryExecutorProxy):
        return executor.submit(fn, *args)
    scope = current_scope()
    if scope is None:
        return executor.submit(fn, *args)
    queued_ns = time.monotonic_ns()

    def run():
        token = activate(scope)
        try:
            scope.add_executor_wait(time.monotonic_ns() - queued_ns)
            return fn(*args)
        finally:
            deactivate(token)

    return executor.submit(run)


def transfer_submitter(executor):
    """Select the original submit method once per transfer, not per layer."""
    if current_scope() is None:
        return executor.submit
    return functools.partial(submit_transfer, executor)


class TelemetryExecutorProxy:
    """Covers an Ascend sender override that submits directly to its executor."""

    def __init__(self, executor):
        self._executor = executor
        self._futures = []

    def submit(self, fn, *args):
        future = submit_transfer(self._executor, fn, *args)
        self._futures.append(future)
        return future

    def drain(self):
        futures, self._futures = self._futures, []
        if futures:
            concurrent.futures.wait(futures)


def classify_kv_blocks(manager, src_addrs, lengths):
    """Classify blocks from Ascend's KV override, which has no typed callsite.

    A source range matching both target and draft is ambiguous and is reported
    separately instead of guessing. Only called while telemetry is enabled.
    """
    kv_args = manager.kv_args
    ptrs = kv_args.kv_data_ptrs
    lens = kv_args.kv_data_lens
    draft_count = getattr(kv_args, "num_draft_entries", 0)
    first_draft = len(ptrs) - draft_count
    ranges = [
        (
            int(ptr),
            int(ptr) + int(size),
            "kv_draft" if i >= first_draft else "kv_target",
        )
        for i, (ptr, size) in enumerate(zip(ptrs, lens))
    ]
    kinds = []
    for src, length in zip(src_addrs, lengths):
        matched = {
            kind for start, end, kind in ranges if start <= src and src + length <= end
        }
        kinds.append(next(iter(matched)) if len(matched) == 1 else "kv_unclassified")
    return kinds


def typed_transfer(manager, session_id, blocks, kind="kv", kinds=None):
    """Keep the legacy two-argument transfer path untouched when disabled."""
    if current_scope() is None:
        return manager._transfer_data(session_id, blocks)
    return manager._transfer_data(session_id, blocks, kind=kind, kinds=kinds)


def typed_engine_transfer(engine, session_id, src, dst, lengths, kind):
    """Instrument the TP slice path, which calls the engine directly."""
    scope = current_scope()
    if scope is None:
        return engine.batch_transfer_sync(session_id, src, dst, lengths)
    submit_ns = scope.submitted()
    try:
        ret = engine.batch_transfer_sync(session_id, src, dst, lengths)
    except BaseException:
        scope.completed(submit_ns, lengths, -1, [kind] * len(lengths))
        raise
    scope.completed(submit_ns, lengths, ret, [kind] * len(lengths))
    return ret


class TransferTelemetry:
    """A process owns one append-only file; a manager shares it across workers."""

    def __init__(self, path_prefix: str, rank: int):
        host = socket.gethostname().replace("/", "_")
        self.path = f"{path_prefix}.{host}.rank{rank}.pid{os.getpid()}.jsonl"
        self._fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        self._write_lock = threading.Lock()
        self._write_failed = False
        self._ids = itertools.count()
        logger.info("Mooncake transfer telemetry: %s", self.path)

    def begin(self, room, index_slice, is_last_chunk, rid=None):
        return TransferScope(
            self,
            next(self._ids),
            room,
            index_slice,
            is_last_chunk,
            rid,
        )

    def write(self, record):
        line = (
            json.dumps(record, separators=(",", ":"), ensure_ascii=True) + "\n"
        ).encode()
        with self._write_lock:
            if self._write_failed:
                return
            # One write per row. The host/rank/pid suffix prevents cross-process
            # interleaving; the lock prevents worker threads from interleaving.
            try:
                if os.write(self._fd, line) != len(line):
                    self._write_failed = True
                    logger.error(
                        "Mooncake transfer telemetry short write: %s", self.path
                    )
            except OSError:
                self._write_failed = True
                logger.exception(
                    "Mooncake transfer telemetry write failed: %s", self.path
                )


class TransferScope:
    def __init__(self, telemetry, chunk_id, room, index_slice, last, rid):
        self.telemetry = telemetry
        self.chunk_id = chunk_id
        self.room = int(room)
        # TraceNullContext returns itself for absent attributes. Such an
        # object is not a request ID and must not become a misleading string.
        self.rid = rid if isinstance(rid, str) else None
        self.index_start = (
            int(index_slice.start) if index_slice.start is not None else None
        )
        self.index_stop = (
            int(index_slice.stop) if index_slice.stop is not None else None
        )
        self.is_last_chunk = last
        self.enqueue_ns = time.monotonic_ns()
        self.first_enqueue_ns = self.enqueue_ns
        self.first_dequeue_ns = None
        self.defer_count = 0
        self.queue_wait_ns = 0
        self.executor_wait_ns = 0
        self.engine_call_ns = 0
        self.tcp_call_ns = 0
        self.first_submit_ns = None
        self.first_submit_wall_ns = None
        self.last_done_ns = None
        self.failed = False
        self.by_type = defaultdict(
            lambda: {"bytes": 0, "ok_bytes": 0, "blocks": 0, "calls": 0}
        )
        self._lock = threading.Lock()
        self._finished = False

    def dequeued(self):
        now_ns = time.monotonic_ns()
        self.queue_wait_ns += now_ns - self.enqueue_ns
        if self.first_dequeue_ns is None:
            self.first_dequeue_ns = now_ns

    def requeued(self):
        self.defer_count += 1
        self.enqueue_ns = time.monotonic_ns()

    def add_executor_wait(self, ns):
        with self._lock:
            self.executor_wait_ns += ns

    def submitted(self):
        submit_ns = time.monotonic_ns()
        with self._lock:
            if self.first_submit_ns is None:
                self.first_submit_ns = submit_ns
                self.first_submit_wall_ns = time.time_ns()
        return submit_ns

    def completed(self, submit_ns, lengths, ret, kinds):
        done_ns = time.monotonic_ns()
        with self._lock:
            self.engine_call_ns += done_ns - submit_ns
            self.last_done_ns = max(self.last_done_ns or 0, done_ns)
            self.failed |= ret != 0
            for kind, length in zip(kinds, lengths):
                counters = self.by_type[kind]
                counters["bytes"] += int(length)
                counters["blocks"] += 1
                if ret == 0:
                    counters["ok_bytes"] += int(length)
            # Calls count once per type present in the engine invocation.
            for kind in set(kinds):
                self.by_type[kind]["calls"] += 1

    def completed_tcp(self, start_ns, length, failed):
        done_ns = time.monotonic_ns()
        with self._lock:
            self.tcp_call_ns += done_ns - start_ns
            self.failed |= failed
            counters = self.by_type["aux_tcp"]
            counters["bytes"] += length
            counters["blocks"] += 1
            counters["calls"] += 1
            if not failed:
                counters["ok_bytes"] += length

    def finish(self, failed=False, skipped=False):
        worker_done_ns = time.monotonic_ns()
        with self._lock:
            if self._finished:
                return
            self._finished = True
            by_type = dict(self.by_type)
            record = {
                "room": self.room,
                "rid": self.rid,
                "chunk_id": self.chunk_id,
                "index_start": self.index_start,
                "index_stop": self.index_stop,
                "is_last_chunk": self.is_last_chunk,
                "status": "skipped"
                if skipped
                else ("failed" if failed or self.failed else "done"),
                "enqueue_ts_mono_ns": self.first_enqueue_ns,
                "dequeue_ts_mono_ns": self.first_dequeue_ns,
                "worker_done_ts_mono_ns": worker_done_ns,
                "defer_count": self.defer_count,
                "submit_ts_mono_ns": self.first_submit_ns,
                "submit_ts_wall_ns": self.first_submit_wall_ns,
                "done_ts_mono_ns": self.last_done_ns,
                "queue_wait_ms": self.queue_wait_ns / 1e6,
                "worker_elapsed_ms": (
                    (worker_done_ns - self.first_dequeue_ns) / 1e6
                    if self.first_dequeue_ns is not None
                    else 0.0
                ),
                "executor_wait_ms_sum": self.executor_wait_ns / 1e6,
                "engine_call_ms_sum": self.engine_call_ns / 1e6,
                "tcp_call_ms_sum": self.tcp_call_ns / 1e6,
                "engine_span_ms": (
                    (self.last_done_ns - self.first_submit_ns) / 1e6
                    if self.first_submit_ns is not None
                    and self.last_done_ns is not None
                    else 0.0
                ),
                "bytes": sum(x["bytes"] for x in by_type.values()),
                "ok_bytes": sum(x["ok_bytes"] for x in by_type.values()),
                "blocks": sum(x["blocks"] for x in by_type.values()),
                "by_type": by_type,
            }
        self.telemetry.write(record)
