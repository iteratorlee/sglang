"""CPU regression checks for premature PD completion on a non-final send.

Execute production scheduler/sender/worker bodies with device and wire operations
stubbed. This proves the lifecycle invariant, not an observed runtime root cause.
No torch, NPU, network, or background threads are used.
"""

import ast
import enum
import logging
import time
import unittest
from collections import defaultdict, deque
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS


SRT = Path(__file__).resolve().parents[4] / "python/sglang/srt"


def load_production():
    scope = {
        "enum": enum,
        "time": time,
        "logger": logging.getLogger(__name__),
        "_is_npu": False,
        "DisaggregationMode": NS(PREFILL="prefill"),
        "get_schedule": lambda: NS(chunked_prefill_size=256),
        "TraceNullContext": NS,
        "TransferKVChunk": lambda **kw: NS(staging_counted=False, **kw),
    }
    sources = {
        "disaggregation/base/conn.py": {"StateType", "KVPoll"},
        "disaggregation/prefill.py": {"send_kv_chunk"},
        "disaggregation/common/conn.py": {
            "_prepare_send_indices",
            "_record_transfer_indices",
            "should_send_kv_chunk",
        },
        "disaggregation/mooncake/conn.py": {
            "send",
            "add_transfer_request",
            "transfer_worker",
        },
        "disaggregation/common/staging_buffer.py": {
            "staging_grid_tokens",
            "compute_grid_segments",
        },
        "mem_cache/common.py": {"kv_to_page_indices", "kv_to_page_num"},
    }
    for relative, names in sources.items():
        path = SRT / relative
        tree = ast.parse(path.read_text())
        nodes = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
                nodes.append(node)
            elif isinstance(node, ast.ClassDef):
                nodes.extend(
                    n
                    for n in node.body
                    if isinstance(n, ast.FunctionDef) and n.name in names
                )
        assert {n.name for n in nodes} == names, relative
        for node in nodes:
            node.decorator_list = []
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                *nodes,
            ],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    return scope


PRODUCTION = load_production()
POLL = PRODUCTION["KVPoll"]


class CpuTensor:
    def __init__(self, values):
        self.values = list(values)

    def __getitem__(self, index):
        return CpuTensor(self.values[index])

    def __floordiv__(self, divisor):
        return CpuTensor(v // divisor for v in self.values)

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class TokenRows:
    def __getitem__(self, index):
        return CpuTensor(range(index[1].start or 0, index[1].stop))


class QueueDrained(BaseException):
    """End the real worker's infinite loop without taking its error path."""


class TransferQueue:
    def __init__(self):
        self.pending = deque()
        self.chunks = []

    def put(self, chunk):
        self.pending.append(chunk)
        self.chunks.append(chunk)

    def get(self):
        if not self.pending:
            raise QueueDrained
        return self.pending.popleft()


class TransferHarness:
    def __init__(self, length, prefix=0, staging=False, page_size=64):
        self.room = 123
        self.marker = 0
        self.events = []
        self.queue = TransferQueue()
        pages = PRODUCTION["kv_to_page_num"](length - prefix, page_size)
        session = "127.0.0.1:20126"
        target = NS(
            is_dummy=False,
            mooncake_session_id=session,
            dst_kv_indices=list(range(pages)),
            dst_device_kv_indices=None,
            required_dst_info_num=1,
            endpoint="127.0.0.1",
            dst_port=12345,
            room=self.room,
        )
        self.manager = NS(
            disaggregation_mode="prefill",
            request_status={self.room: POLL.WaitingForInput},
            transfer_infos={self.room: {session: target}},
            transfer_queues=[self.queue],
            enable_all_cp_ranks_for_transfer=False,
            is_dummy_cp_rank=False,
            kv_args=NS(state_types=[PRODUCTION["StateType"].MAMBA], kv_data_ptrs=[]),
            enable_trace=False,
            # Exercise scheduler staging segmentation; actual device transport
            # is stubbed, including its staging allocation/scatter protocol.
            enable_staging=False,
            enable_deferred_decode_kv_release=False,
            session_lock=nullcontext(),
            failed_sessions=set(),
            _staging_outstanding=defaultdict(int),
            req_to_decode_prefix_len={self.room: prefix},
            decode_kv_args_table={
                session: NS(requires_dcp_relayout=False, dst_aux_ptrs=[1])
            },
            _prefill_unique_rank=lambda: 0,
            _get_dsa_cache_transfer_skip_flags=lambda info: (False, False),
            bootstrap_port=1,
        )
        mgr = self.manager
        mgr.check_status = lambda room: mgr.request_status[room]
        mgr.add_transfer_request = lambda *a, **kw: PRODUCTION["add_transfer_request"](
            mgr, *a, **kw
        )
        mgr.maybe_send_extra = self.send_state
        mgr.send_aux = self.send_aux
        mgr.conclude_transfer = self.conclude
        self.sender = NS(
            kv_mgr=mgr,
            bootstrap_room=self.room,
            curr_idx=0,
            num_kv_indices=pages,
            aux_index=0,
            trace_ctx=NS(copy_for_thread=NS),
            _transfer_num_kv_indices=0,
            _transfer_num_state_indices=0,
        )
        for name in ("_prepare_send_indices", "_record_transfer_indices", "send"):
            setattr(self.sender, name, PRODUCTION[name].__get__(self.sender))
        self.sender.should_send_kv_chunk = PRODUCTION["should_send_kv_chunk"].__get__(
            self.sender
        )
        self.req = NS(
            rid="final-page-cpu",
            origin_input_ids=range(length),
            start_send_idx=prefix,
            disagg_decode_prefix_len=prefix,
            extend_range=NS(end=length),
            kv=NS(req_pool_idx=0),
            disagg_kv_sender=self.sender,
        )
        self.scheduler = NS(
            enable_staging=staging,
            token_to_kv_pool_allocator=NS(
                page_size=page_size, translate_kv_indices_for_transfer=lambda x: x
            ),
            req_to_token_pool=NS(
                req_to_token=TokenRows(),
                req_index_to_mamba_index_mapping=[9],
                translate_mamba_indices=lambda index: NS(
                    cpu=lambda: NS(numpy=lambda: index)
                ),
            ),
            disagg_prefill_bootstrap_queue=NS(kv_manager=mgr),
            disagg_metadata_buffers=NS(set_buf=self.set_metadata),
            disagg_prefill_pending_chunk_rids=set(),
        )

    def set_metadata(self, req):
        self.marker = self.room
        self.events.append(("metadata", self.room))

    def send_state(self, req, indices, executor, info):
        self.events.append(("state", indices))
        return 0

    def send_aux(self, req, index, pointers):
        self.events.append(("aux", self.marker))
        return 0

    def conclude(self, *, bootstrap_room, status, **kw):
        self.manager.request_status[bootstrap_room] = status
        self.events.append(("status", status))

    def send(self, last=False, end=None):
        PRODUCTION["send_kv_chunk"](
            self.scheduler, self.req, last_chunk=last, end_idx=end
        )

    def drain(self):
        try:
            PRODUCTION["transfer_worker"](self.manager, self.queue, None)
        except QueueDrained:
            pass


class TestPrefillFinalPage(unittest.TestCase):
    def assert_complete(self, harness, prefix, length, page_size=64):
        harness.send(last=True)
        harness.drain()
        chunks = harness.queue.chunks
        self.assertEqual(sum(c.is_last_chunk for c in chunks), 1)
        self.assertTrue(chunks[-1].is_last_chunk)
        self.assertEqual(chunks[-1].state_indices, [[9]])
        self.assertEqual(
            [page for c in chunks for page in c.prefill_kv_indices],
            list(range(prefix // page_size, (length + page_size - 1) // page_size)),
        )
        self.assertEqual(
            harness.events,
            [
                ("metadata", 123),
                ("state", [[9]]),
                ("aux", 123),
                ("status", POLL.Success),
            ],
        )
        self.assertNotIn(harness.room, harness.manager.transfer_infos)

    def test_exact_128k_does_not_complete_or_drop_later_final(self):
        harness = TransferHarness(131072)
        harness.send()
        self.assertEqual(harness.sender.curr_idx, 2047)
        self.assertFalse(harness.queue.chunks[-1].is_last_chunk)
        harness.drain()
        self.assertEqual(harness.events, [])
        self.assertIn(harness.room, harness.manager.transfer_infos)
        self.assertEqual(
            harness.manager.request_status[harness.room], POLL.WaitingForInput
        )
        self.assert_complete(harness, 0, 131072)

    def test_exact_partial_and_single_pages_with_decode_prefix(self):
        for page_size, lengths in (
            (1, (1, 8)),
            (64, (1, 63, 64, 65, 127, 128, 129, 131073)),
        ):
            for length in lengths:
                for prefix in {0, ((length - 1) // page_size) * page_size}:
                    with self.subTest(
                        page_size=page_size, length=length, prefix=prefix
                    ):
                        harness = TransferHarness(length, prefix, page_size=page_size)
                        harness.send()
                        self.assertLess(
                            harness.sender.curr_idx, harness.sender.num_kv_indices
                        )
                        harness.drain()
                        self.assertEqual(harness.events, [])
                        self.assert_complete(harness, prefix, length, page_size)

    def test_repeated_and_overshooting_nonfinal_calls_keep_reserved_page(self):
        for length in (128, 129):
            with self.subTest(length=length):
                harness = TransferHarness(length)
                harness.send(end=length)
                count = len(harness.queue.chunks)
                harness.send(end=length + 1000)
                self.assertEqual(len(harness.queue.chunks), count)
                harness.drain()
                self.assertEqual(harness.events, [])
                self.assert_complete(harness, 0, length)

    def test_staging_alignment_is_relative_to_decode_prefix(self):
        for length, endpoint in ((832, 576), (896, 832), (897, 832)):
            with self.subTest(length=length):
                harness = TransferHarness(length, prefix=64, staging=True)
                harness.send()
                self.assertEqual(harness.req.start_send_idx, endpoint)
                self.assertEqual((endpoint - 64) % 256, 0)
                self.assertTrue(all(not c.is_last_chunk for c in harness.queue.chunks))
                harness.drain()
                self.assertEqual(harness.events, [])
                self.assert_complete(harness, 64, length)

    def test_zero_page_final_still_sends_metadata_and_state(self):
        for length in (0, 128):
            for staging in (False, True):
                with self.subTest(length=length, staging=staging):
                    harness = TransferHarness(length, prefix=length, staging=staging)
                    harness.send()
                    self.assertEqual(harness.queue.chunks, [])
                    self.assertEqual(harness.req.start_send_idx, length)
                    self.assert_complete(harness, length, length)
                    self.assertEqual(harness.queue.chunks[-1].prefill_kv_indices, [])

    def test_earlier_chunk_boundaries_keep_streaming_progress(self):
        harness = TransferHarness(513)
        for endpoint, expected in ((129, 128), (257, 256), (513, 512)):
            harness.send(end=endpoint)
            self.assertEqual(harness.req.start_send_idx, expected)
            harness.drain()
            self.assertEqual(harness.events, [])
        self.assert_complete(harness, 0, 513)


if __name__ == "__main__":
    unittest.main()
