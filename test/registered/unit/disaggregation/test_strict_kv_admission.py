from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.disaggregation.strict_kv_admission import (
    StrictKVBypassGuard,
    projected_full_kv_tokens,
)
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs


def request(prompt: int, max_output: int, emitted: int = 0):
    return SimpleNamespace(
        origin_input_ids=range(prompt),
        output_ids=range(emitted),
        sampling_params=SimpleNamespace(max_new_tokens=max_output),
    )


def make_queue(running=None):
    queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
    queue.pp_size = 1
    queue.queue = []
    queue.pending_reqs = []
    queue.retracted_queue = []
    queue._resolve_pending_reqs = MagicMock()
    queue._update_handshake_waiters = MagicMock()
    queue._uses_swa_tail_prealloc = MagicMock(return_value=False)
    queue._allocatable_token_budgets = MagicMock(return_value=139_264)
    queue._hicache_pending_restore_tokens = MagicMock(return_value=0)
    queue._pre_alloc = MagicMock(return_value=torch.empty(0, dtype=torch.int64))
    queue.req_to_token_pool = SimpleNamespace(
        available_size=lambda: 20,
        req_to_token=torch.arange(100_000, dtype=torch.int64)[None, :],
    )
    queue.req_to_metadata_buffer_idx_allocator = SimpleNamespace(
        available_size=lambda: 20, alloc=lambda: 0
    )
    queue.token_to_kv_pool_allocator = SimpleNamespace(
        page_size=64, translate_kv_indices_for_transfer=lambda indices: indices
    )
    # The NPU payload path checks the concrete pool type even when this
    # fixture carries no model-specific state. Represent an ordinary pool.
    queue.token_to_kv_pool = SimpleNamespace(page_size=64)
    queue.kv_manager = SimpleNamespace(kv_args=SimpleNamespace(state_types=[]))
    queue.transfer_queue = SimpleNamespace(queue=[], enable_staging=False)
    queue.scheduler = SimpleNamespace(
        running_batch=SimpleNamespace(reqs=running if running is not None else []),
        waiting_queue=[],
        last_batch=None,
        enable_priority_scheduling=False,
        enable_hisparse=False,
        enable_decode_hicache=False,
        output_streamer=MagicMock(),
    )
    queue.num_reserved_decode_tokens = 512
    queue.max_total_num_tokens = 139_264
    queue.glm53_decode_prefix_profile = None
    queue.strict_decode_kv_admission = True
    queue.strict_kv_fair_scan = True
    queue.strict_kv_bypass_guard = StrictKVBypassGuard()
    queue._num_published_destinations = 0
    return queue


def entry(rid, prompt, output, *, ready=True):
    req = request(prompt, output)
    req.rid = rid
    req.bootstrap_host = "real"
    req.kv = SimpleNamespace(req_pool_idx=0)
    req.time_stats = MagicMock()
    req.finished_reason = None
    req.return_logprob = False
    receiver = MagicMock()
    return SimpleNamespace(
        req=req,
        kv_receiver=receiver,
        waiting_for_input=ready,
        is_rebootstrap=False,
    )


def test_rejects_transfer_that_would_starve_running_output():
    running = request(31_104, 5_984, emitted=5_755)
    transfer = request(102_400, 256)
    projected = projected_full_kv_tokens(
        [running, transfer], page_size=64, decode_headroom_per_request=512
    )
    assert projected > 139_264


def test_admits_two_long_requests_with_sufficient_final_kv():
    requests = [request(64_000, 1_000), request(64_000, 1_000)]
    projected = projected_full_kv_tokens(
        requests, page_size=64, decode_headroom_per_request=512
    )
    assert projected <= 139_264


def test_deduplicates_a_request_seen_in_multiple_queue_snapshots():
    req = request(31_104, 5_984)
    once = projected_full_kv_tokens(
        [req], page_size=64, decode_headroom_per_request=512
    )
    twice = projected_full_kv_tokens(
        [req, req], page_size=64, decode_headroom_per_request=512
    )
    assert once == twice


def test_full_projection_rounds_each_distinct_request_and_honors_emitted_output():
    first = request(65, 1, emitted=70)
    second = request(1, 1)
    assert (
        projected_full_kv_tokens(
            [first, second, first], page_size=64, decode_headroom_per_request=512
        )
        == 704 + 576
    )


def test_bypass_guard_reserves_old_request_after_bounded_successful_bypasses():
    guard = StrictKVBypassGuard(max_bypasses=2)
    old, short_a, short_b = object(), object(), object()
    guard.blocked(old)
    guard.admitted(short_a)
    assert not guard.at_limit
    guard.admitted(short_b)
    assert guard.at_limit
    guard.reconcile([old])
    assert guard.at_limit
    guard.admitted(old)
    assert not guard.at_limit
    guard.blocked(short_a)
    guard.admitted(short_b)
    guard.reconcile([])
    assert guard.blocked_id is None


def test_fair_scan_admits_short_requests_once_then_reserves_long_request():
    reset_context()
    try:
        publish(ServerArgs(model_path="dummy"), role="tokenizer")
        queue = make_queue(running=[request(90_000, 1_000)])
        unready = entry("unready", 512, 100, ready=False)
        long = entry("long", 50_000, 1_000)
        second_long = entry("second-long", 80_000, 1_000)
        shorts = [entry(f"short-{i}", 1_024, 100) for i in range(9)]
        queue.queue = [unready, long, second_long, *shorts]

        def check_first_send(*_args, **_kwargs):
            assert queue._pre_alloc.call_count == 1
            assert queue._num_published_destinations == 0

        shorts[0].kv_receiver.send_metadata.side_effect = check_first_send

        queue.strict_kv_fair_scan = False
        assert queue.pop_preallocated() == ([], [])
        assert queue.queue == [unready, long, second_long, *shorts]
        queue.strict_kv_fair_scan = True

        admitted, failed = queue.pop_preallocated()
        assert not failed
        assert admitted == shorts[:8]
        assert queue.queue == [unready, long, second_long, shorts[8]]
        assert queue.strict_kv_bypass_guard.at_limit
        assert queue.strict_kv_bypass_guard.bypasses == 8
        assert queue._num_published_destinations == 8
        assert all(s.kv_receiver.send_metadata.call_count == 1 for s in shorts[:8])
        assert shorts[8].kv_receiver.send_metadata.call_count == 0
        assert unready.kv_receiver.send_metadata.call_count == 0
        assert second_long.kv_receiver.send_metadata.call_count == 0

        # A second pass cannot publish a ninth short while the old request is
        # protected. Releasing running KV lets that request go first.
        assert queue.pop_preallocated() == ([], [])
        queue.note_destinations_queued(8)
        queue.transfer_queue.queue = admitted
        queue.scheduler.running_batch.reqs = []
        next_admitted, failed = queue.pop_preallocated()
        assert not failed
        assert next_admitted == [long]
        assert queue.queue == [unready, second_long, shorts[8]]
        assert long.kv_receiver.send_metadata.call_count == 1
        assert queue._num_published_destinations == 1
        assert queue.strict_kv_bypass_guard.blocked_id is None

        queue.note_destinations_queued(1)
        queue.transfer_queue.queue = [*admitted, long]
        last_admitted, failed = queue.pop_preallocated()
        assert not failed
        assert last_admitted == [shorts[8]]
        assert queue.queue == [unready, second_long]
        assert queue._num_published_destinations == 1
        assert queue.strict_kv_bypass_guard.blocked_entry is second_long
        assert queue.strict_kv_bypass_guard.bypasses == 1
        assert all(s.kv_receiver.send_metadata.call_count == 1 for s in shorts)
        assert unready.kv_receiver.send_metadata.call_count == 0

        queue.note_destinations_queued(1)
        queue.transfer_queue.queue = []
        assert queue.pop_preallocated()[0] == [second_long]
        assert second_long.kv_receiver.send_metadata.call_count == 1
        assert queue.queue == [unready]
        assert queue.strict_kv_bypass_guard.blocked_entry is None
        assert queue._num_published_destinations == 1

        # With a larger bypass allowance, a single call still inspects only
        # the first 16 queue positions after the blocked head.
        queue.note_destinations_queued(1)
        queue.transfer_queue.queue = []
        queue.scheduler.running_batch.reqs = [request(90_000, 1_000)]
        scan_long = entry("scan-long", 50_000, 1_000)
        many_shorts = [entry(f"scan-{i}", 1_024, 100) for i in range(17)]
        queue.queue = [unready, scan_long, *many_shorts]
        queue.strict_kv_bypass_guard = StrictKVBypassGuard(max_bypasses=100)
        scanned, failed = queue.pop_preallocated()
        assert not failed
        assert scanned == many_shorts[:16]
        assert queue.queue == [unready, scan_long, many_shorts[16]]
        assert queue._num_published_destinations == 16
        assert unready.kv_receiver.send_metadata.call_count == 0
    finally:
        reset_context()


def test_fair_scan_rejects_single_request_larger_than_full_kv_capacity():
    queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
    queue.strict_kv_fair_scan = True
    queue._check_if_req_exceed_kv_capacity = MagicMock(return_value=False)
    queue.token_to_kv_pool_allocator = SimpleNamespace(page_size=64)
    queue.num_reserved_decode_tokens = 512
    queue.max_total_num_tokens = 139_264
    queue.scheduler = SimpleNamespace(output_streamer=MagicMock())
    queue.queue = []
    req = request(139_000, 1_000)
    req.rid = "too-long"
    req.return_logprob = False
    queue.add(req)
    assert isinstance(req.finished_reason, FINISH_ABORT)
    assert queue.queue == []
    queue.scheduler.output_streamer.stream_output.assert_called_once_with([req], False)


def test_protected_abort_resets_gate_before_filtered_cleanup_and_same_rid_retry():
    reset_context()
    try:
        publish(ServerArgs(model_path="dummy"), role="tokenizer")
        queue = make_queue(running=[request(90_000, 1_000)])
        old = entry("shared-rid", 50_000, 1_000)
        shorts = [entry(f"short-{i}", 1_024, 100) for i in range(9)]
        queue.queue = [old, *shorts]
        queue.pending_reqs = [old]

        admitted, failed = queue.pop_preallocated()
        assert not failed
        assert admitted == shorts[:8]
        assert queue.strict_kv_bypass_guard.at_limit
        queue.note_destinations_queued(8)
        queue.transfer_queue.queue = admitted

        old.req.finished_reason = FINISH_ABORT("expired")
        # A PP-style RID filter can defer receiver cleanup. The expired head
        # must stop reserving capacity immediately, without double publishing.
        ninth, failed = queue.pop_preallocated(rids_to_check=[shorts[8].req.rid])
        assert not failed
        assert ninth == [shorts[8]]
        assert queue.strict_kv_bypass_guard.blocked_entry is None
        assert queue.queue == [old]
        assert old.kv_receiver.send_metadata.call_count == 0
        assert all(s.kv_receiver.send_metadata.call_count == 1 for s in shorts)
        assert queue._num_published_destinations == 1

        _, failed = queue.pop_preallocated()
        assert failed == [old]
        assert queue.queue == []
        assert queue.pending_reqs == []
        assert old.kv_receiver is None
        assert queue._num_published_destinations == 1
        queue.scheduler.output_streamer.stream_output.assert_called_once_with(
            [old.req], False
        )
        queue.note_destinations_queued(1)
        assert queue._num_published_destinations == 0

        replacement = entry("shared-rid", 50_000, 1_000)
        queue.queue = [replacement]
        assert queue.pop_preallocated() == ([], [])
        assert queue.strict_kv_bypass_guard.blocked_entry is replacement
        assert queue.strict_kv_bypass_guard.bypasses == 0
        assert replacement.kv_receiver.send_metadata.call_count == 0
    finally:
        reset_context()


def test_no_longer_ready_protected_request_releases_bypass_gate():
    reset_context()
    try:
        publish(ServerArgs(model_path="dummy"), role="tokenizer")
        queue = make_queue(running=[request(90_000, 1_000)])
        old = entry("old", 50_000, 1_000)
        shorts = [entry(f"short-{i}", 1_024, 100) for i in range(9)]
        queue.queue = [old, *shorts]
        assert queue.pop_preallocated()[0] == shorts[:8]
        assert queue.strict_kv_bypass_guard.at_limit

        old.waiting_for_input = False
        assert queue.pop_preallocated()[0] == [shorts[8]]
        assert queue.strict_kv_bypass_guard.blocked_entry is None
        assert queue.queue == [old]
        assert old.kv_receiver.send_metadata.call_count == 0
    finally:
        reset_context()
