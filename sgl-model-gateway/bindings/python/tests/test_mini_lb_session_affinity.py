"""Offline tests for MiniLB session affinity; no model service is contacted.

The state-machine tests load ``session_affinity.py`` directly so they remain
runnable on development hosts without FastAPI or the native router extension.
The HTTP entrance tests use an in-memory aiohttp backend and are skipped only
when their optional Python dependencies are unavailable.
"""

import asyncio
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest import mock


PYTHON_BINDINGS = Path(__file__).resolve().parents[1]
SESSION_AFFINITY_SOURCE = (
    PYTHON_BINDINGS / "src" / "sglang_router" / "session_affinity.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_mini_lb_session_affinity_under_test", SESSION_AFFINITY_SOURCE
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import machinery
    raise RuntimeError(f"cannot load {SESSION_AFFINITY_SOURCE}")
_STATE_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _STATE_MODULE
_SPEC.loader.exec_module(_STATE_MODULE)
SessionAffinity = _STATE_MODULE.SessionAffinity
SessionReservation = _STATE_MODULE.SessionReservation


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


async def wait_until(predicate, timeout=1.0):
    async def poll():
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(poll(), timeout=timeout)


def finish(reservation, *, successful=True):
    if successful:
        reservation.mark_successful()
    else:
        reservation.mark_failed()
    if not reservation.release():
        raise AssertionError("first reservation release was not effective")


class SessionAffinityStateTests(unittest.IsolatedAsyncioTestCase):
    async def test_sixteen_sessions_fill_dp8_two_each_without_using_future_osl(self):
        state = SessionAffinity([0] * 8, 2, 7200, 1, clock=FakeClock())
        reservations = []
        for session in range(16):
            reservation = await state.reserve(
                f"session-{session}",
                {
                    "input_ids": [session, 2, 3],
                    # Deliberately different future lengths: placement must not
                    # inspect them.
                    "sampling_params": {"max_new_tokens": 1 + session * 1000},
                },
                affinity_rank=0,
                final=True,
            )
            reservations.append(reservation)

        self.assertEqual(state.lease_counts, [2] * 8)
        self.assertEqual(state.inflight, [2] * 8)
        self.assertEqual(state.peak_lease_counts, [2] * 8)
        self.assertEqual(
            reservations[0].rank,
            0,
            "an otherwise exact tie should preserve the original prefix rank",
        )
        self.assertEqual(state.counters["new_sessions"], 16)
        for reservation in reservations:
            finish(reservation)
        self.assertEqual(state.inflight, [0] * 8)
        self.assertEqual(state.lease_counts, [0] * 8)
        self.assertEqual(state.counters["completed_sessions"], 16)

    async def test_prefix_change_does_not_migrate_an_existing_session(self):
        state = SessionAffinity([0] * 8, 2, 7200, 1, clock=FakeClock())
        first = await state.reserve(
            "stable-session", {"input_ids": [1, 2, 3]}, affinity_rank=0
        )
        first_rank = first.rank
        finish(first)

        second = await state.reserve(
            "stable-session",
            {"input_ids": [900, 901, 902]},
            affinity_rank=7,
            final=True,
        )
        self.assertEqual(second.rank, first_rank)
        self.assertEqual(state.counters["new_sessions"], 1)
        self.assertEqual(state.counters["reused_requests"], 1)
        finish(second)
        self.assertEqual(state.counters["completed_sessions"], 1)

    async def test_new_sessions_balance_by_leases_then_observed_lifetime(self):
        clock = FakeClock()
        state = SessionAffinity([0, 0], 2, 7200, 1, clock=clock)

        heavy = await state.reserve("heavy", {"input_ids": [1]}, 0)
        clock.advance(10)
        finish(heavy)
        light = await state.reserve("light", {"input_ids": [2]}, 0)
        self.assertEqual(light.rank, 1, "lease count must dominate prefix tie")
        clock.advance(1)
        finish(light)

        lower_observed_work = await state.reserve(
            "new-low-work", {"input_ids": [3]}, affinity_rank=0, final=True
        )
        self.assertEqual(
            lower_observed_work.rank,
            1,
            "with equal lease counts, observed response lifetime should balance",
        )
        lease_count_first = await state.reserve(
            "new-lease-gap", {"input_ids": [4]}, affinity_rank=1, final=True
        )
        self.assertEqual(
            lease_count_first.rank,
            0,
            "a lower lease count must outrank the lifetime heuristic",
        )

        finish(lower_observed_work)
        finish(lease_count_first)
        for session in ("heavy", "light"):
            final = await state.reserve(
                session, {"input_ids": [99]}, affinity_rank=0, final=True
            )
            finish(final)
        self.assertEqual(state.lease_counts, [0, 0])

    async def test_existing_session_requests_are_serialized_on_the_same_rank(self):
        state = SessionAffinity([0] * 4, 2, 7200, 1)
        first = await state.reserve("serial", {"input_ids": [1]}, 3)
        second_task = asyncio.create_task(
            state.reserve(
                "serial", {"input_ids": [2]}, affinity_rank=0, final=True
            )
        )
        await wait_until(lambda: state.counters["waiting_requests"] == 1)
        self.assertFalse(second_task.done())
        self.assertEqual(sum(state.inflight), 1)

        finish(first)
        second = await asyncio.wait_for(second_task, timeout=1)
        self.assertEqual(second.rank, first.rank)
        self.assertEqual(sum(state.inflight), 1)
        finish(second)
        self.assertEqual(state.inflight, [0] * 4)
        self.assertEqual(state.counters["waiting_requests"], 0)

    async def test_final_response_releases_capacity_for_a_new_session(self):
        state = SessionAffinity([0], 1, 7200, 1)
        first = await state.reserve("first", {"input_ids": [1]}, 0)
        finish(first)
        self.assertEqual(state.lease_counts, [1])

        newcomer_task = asyncio.create_task(
            state.reserve("newcomer", {"input_ids": [2]}, 0, final=True)
        )
        await wait_until(lambda: state.counters["waiting_requests"] == 1)
        self.assertFalse(newcomer_task.done())

        final = await state.reserve(
            "first", {"input_ids": [3]}, affinity_rank=0, final=True
        )
        self.assertFalse(newcomer_task.done())
        finish(final)
        newcomer = await asyncio.wait_for(newcomer_task, timeout=1)
        self.assertEqual(state.counters["completed_sessions"], 1)
        finish(newcomer)
        self.assertEqual(state.lease_counts, [0])

    async def test_short_request_cleanup_wakes_all_capacity_waiters(self):
        state = SessionAffinity([0] * 8, 2, 7200, 2)

        async def short_session(index):
            reservation = await state.reserve(
                f"short-{index}",
                {"input_ids": [index], "sampling_params": {"max_new_tokens": 1}},
                affinity_rank=index % 8,
                final=True,
            )
            await asyncio.sleep(0)
            finish(reservation)

        await asyncio.wait_for(
            asyncio.gather(*(short_session(index) for index in range(128))),
            timeout=5,
        )
        self.assertEqual(state.inflight, [0] * 8)
        self.assertEqual(state.lease_counts, [0] * 8)
        self.assertEqual(state.counters["waiting_requests"], 0)
        self.assertEqual(state.counters["completed_requests"], 128)
        self.assertEqual(state.counters["completed_sessions"], 128)

    async def test_waiting_new_session_times_out_and_cancellation_has_no_residue(self):
        state = SessionAffinity([0], 1, 7200, 0.02)
        holder = await state.reserve("holder", {"input_ids": [1]}, 0)
        finish(holder)
        with self.assertRaisesRegex(TimeoutError, "admission timed out"):
            await asyncio.wait_for(
                state.reserve("timeout", {"input_ids": [2]}, 0), timeout=1
            )
        self.assertEqual(state.counters["waiting_requests"], 0)
        self.assertNotIn(SessionAffinity.key("timeout", {}), state.leases)

        state.request_timeout = 10
        cancelled_task = asyncio.create_task(
            state.reserve("cancelled", {"input_ids": [3]}, 0)
        )
        await wait_until(lambda: state.counters["waiting_requests"] == 1)
        cancelled_task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await cancelled_task
        self.assertEqual(state.counters["waiting_requests"], 0)
        self.assertNotIn(SessionAffinity.key("cancelled", {}), state.leases)

        final = await state.reserve(
            "holder", {"input_ids": [4]}, affinity_rank=0, final=True
        )
        finish(final)

    async def test_failure_aborts_binding_and_idle_expiry_never_evicts_inflight(self):
        clock = FakeClock()
        state = SessionAffinity([0], 1, 10, 1000, clock=clock)
        active = await state.reserve("active", {"input_ids": [1]}, 0)
        active_key = SessionAffinity.key("active", {})
        clock.advance(100)

        waiter = asyncio.create_task(
            state.reserve("waiting", {"input_ids": [2]}, 0)
        )
        await wait_until(lambda: state.counters["waiting_requests"] == 1)
        self.assertIn(active_key, state.leases)
        self.assertEqual(state.counters["expired_sessions"], 0)
        waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiter

        finish(active, successful=False)
        self.assertNotIn(active_key, state.leases)
        self.assertEqual(state.counters["aborted_sessions"], 1)
        self.assertEqual(state.inflight, [0])

        idle = await state.reserve("idle", {"input_ids": [3]}, 0)
        finish(idle)
        clock.advance(11)
        replacement = await state.reserve(
            "replacement", {"input_ids": [4]}, 0, final=True
        )
        self.assertEqual(state.counters["expired_sessions"], 1)
        finish(replacement)

    def test_session_key_validation_and_cache_domain_isolation(self):
        request_a = {"input_ids": [1, 2, 3], "cache_salt": "tenant-a"}
        request_b = {"input_ids": [900], "cache_salt": "tenant-a"}
        self.assertEqual(
            SessionAffinity.key("session", request_a),
            SessionAffinity.key("session", request_b),
            "prefix changes must not alter a session binding",
        )
        self.assertNotEqual(
            SessionAffinity.key("session", request_a),
            SessionAffinity.key("session", {"cache_salt": "tenant-b"}),
        )
        self.assertNotEqual(
            SessionAffinity.key("session", request_a),
            SessionAffinity.key(
                "session", {"cache_salt": "tenant-a", "extra_key": "model-b"}
            ),
        )
        for invalid in (None, "", "contains space", "line\nbreak", "非ASCII", "x" * 257):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                SessionAffinity.key(invalid, {})


NATIVE_IMPORT_ERROR = None
try:
    source_path = str(PYTHON_BINDINGS / "src")
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    import aiohttp
    import httpx
    from sglang_router import mini_lb
    from sglang_router.launch_router import parse_router_args
    from sglang_router.router_args import RouterArgs
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - host-specific
    NATIVE_IMPORT_ERROR = exc
    aiohttp = httpx = mini_lb = parse_router_args = RouterArgs = None


def make_args(**updates):
    values = dict(
        mini_lb=True,
        pd_disaggregation=True,
        policy="random",
        prefill_urls=[("http://prefill:31194", 8998)],
        decode_urls=["http://decode:32195"],
        mini_lb_prefix_affinity=True,
        mini_lb_prefix_affinity_length=4,
        mini_lb_prefix_affinity_decode_capacity=2,
        mini_lb_session_affinity=True,
        mini_lb_session_affinity_idle_timeout_secs=7200,
        request_timeout_secs=2,
    )
    values.update(updates)
    return RouterArgs(**values)


if NATIVE_IMPORT_ERROR is None:

    def sse_event(payload):
        return (
            b"data: "
            + json.dumps(
                payload, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
            + b"\n\n"
        )


    SSE_DONE = b"data: [DONE]\n\n"


    def successful_stream(rank, finish_type="length"):
        return [
            sse_event(
                {
                    "text": "partial",
                    "meta_info": {"dp_rank": rank, "finish_reason": None},
                }
            ),
            sse_event(
                {
                    "text": "complete",
                    "meta_info": {
                        "dp_rank": rank,
                        "finish_reason": {"type": finish_type},
                    },
                }
            ),
            SSE_DONE,
        ]

    class Response:
        def __init__(
            self,
            body,
            status=200,
            backend=None,
            stream_chunks=None,
            block_after_first_chunk=False,
        ):
            self.body = body
            self.status = status
            self.content = self
            self.backend = backend
            self.stream_chunks = stream_chunks
            self.block_after_first_chunk = block_after_first_chunk

        async def __aenter__(self):
            return self

        async def __aexit__(self, *unused):
            return False

        def raise_for_status(self):
            if self.status != 200:
                raise aiohttp.ClientError(f"backend status {self.status}")

        async def json(self):
            await asyncio.sleep(0)
            return self.body

        async def iter_chunked(self, unused_size):
            chunks = self.stream_chunks
            if chunks is None:
                chunks = [sse_event(self.body), SSE_DONE]
            for index, chunk in enumerate(chunks):
                yield chunk
                if index == 0 and self.block_after_first_chunk:
                    self.backend.first_stream_chunk.set()
                    await self.backend.finish_stream.wait()

        def __aiter__(self):
            return self.iter_chunked(64 * 1024)


    class Backend:
        def __init__(self, decode_dp=8):
            self.gets = []
            self.posts = []
            self.fail_decode_rids = set()
            self.stream_error_rids = set()
            self.stream_chunks_by_rid = {}
            self.block_stream_rids = {"disconnect"}
            self.block_rids = set()
            self.started_by_rid = {}
            self.release_blocked = asyncio.Event()
            self.first_stream_chunk = asyncio.Event()
            self.finish_stream = asyncio.Event()
            self.info = {
                "http://prefill:31194/server_info": {
                    "dp_size": 1,
                    "internal_states": [{}],
                },
                "http://decode:32195/server_info": {
                    "dp_size": decode_dp,
                    "internal_states": [{}] * decode_dp,
                },
            }

        def session(self, **unused):
            return self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *unused):
            return False

        def get(self, url):
            self.gets.append(url)
            return Response(self.info[url])

        async def post(self, url, *, json):
            body = copy.deepcopy(json)
            self.posts.append((url, body))
            rid = body.get("rid")
            self.started_by_rid[rid] = self.started_by_rid.get(rid, 0) + 1
            if rid in self.block_rids:
                await self.release_blocked.wait()
            if rid in self.fail_decode_rids and "decode:" in url:
                raise RuntimeError("synthetic decode failure")
            rank = body.get("routed_dp_rank", 0)
            response_body = {"text": "ok", "meta_info": {"dp_rank": rank}}
            if rid in self.stream_error_rids and "decode:" in url:
                response_body = {"error": "synthetic stream error"}
            stream_chunks = None
            if body.get("stream") and "decode:" in url:
                stream_chunks = self.stream_chunks_by_rid.get(rid)
                if stream_chunks is None:
                    if rid in self.stream_error_rids:
                        stream_chunks = [sse_event(response_body), SSE_DONE]
                    else:
                        stream_chunks = successful_stream(rank)
            return Response(
                response_body,
                backend=self,
                stream_chunks=stream_chunks,
                block_after_first_chunk=rid in self.block_stream_rids,
            )


@unittest.skipIf(
    NATIVE_IMPORT_ERROR is not None,
    f"native HTTP dependencies unavailable: {NATIVE_IMPORT_ERROR}",
)
class SessionAffinityConfigTests(unittest.TestCase):
    def test_cli_flags_parse_and_invalid_combinations_fail(self):
        self.assertFalse(RouterArgs().mini_lb_session_affinity)
        args = parse_router_args(
            [
                "--mini-lb",
                "--pd-disaggregation",
                "--mini-lb-prefix-affinity",
                "--mini-lb-prefix-affinity-decode-capacity",
                "2",
                "--mini-lb-session-affinity",
                "--mini-lb-session-affinity-idle-timeout-secs",
                "7200",
                "--prefill",
                "http://prefill:31194",
                "8998",
                "--decode",
                "http://decode:32195",
            ]
        )
        args._validate_router_args()
        self.assertTrue(args.mini_lb_session_affinity)
        self.assertEqual(args.mini_lb_prefix_affinity_decode_capacity, 2)
        self.assertEqual(args.mini_lb_session_affinity_idle_timeout_secs, 7200)

        for updates in (
            {"mini_lb_prefix_affinity": False},
            {"mini_lb_prefix_affinity_decode_capacity": 0},
            {"mini_lb_session_affinity_idle_timeout_secs": 0},
            {"mini_lb_session_affinity_idle_timeout_secs": float("inf")},
            {"mini_lb_session_affinity_idle_timeout_secs": float("nan")},
            {"request_timeout_secs": 0},
        ):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                mini_lb.MiniLoadBalancer(make_args(**updates))


@unittest.skipIf(
    NATIVE_IMPORT_ERROR is not None,
    f"native HTTP dependencies unavailable: {NATIVE_IMPORT_ERROR}",
)
class SessionStreamStatusTests(unittest.TestCase):
    def status_after(self, chunks):
        status = mini_lb._SessionStreamStatus()
        for chunk in chunks:
            status.feed(chunk)
        return status

    def test_both_native_normal_finish_types_require_done(self):
        for finish_type in ("length", "stop"):
            with self.subTest(finish_type=finish_type):
                chunks = successful_stream(3, finish_type)
                self.assertTrue(self.status_after(chunks).successful)
                self.assertFalse(self.status_after(chunks[:-1]).successful)

    def test_abort_missing_and_malformed_finish_reason_are_not_success(self):
        reasons = (
            None,
            {"type": "abort", "message": "cancelled"},
            {"type": "tool_calls"},
            {},
            "length",
        )
        for reason in reasons:
            with self.subTest(reason=reason):
                event = sse_event(
                    {"text": "terminal", "meta_info": {"finish_reason": reason}}
                )
                status = self.status_after([event, SSE_DONE])
                self.assertFalse(status.successful)
                if reason is not None:
                    self.assertTrue(status.failed)

    def test_incremental_crlf_multiline_utf8_and_coalesced_events(self):
        first = sse_event(
            {"text": "共同前缀", "meta_info": {"finish_reason": None}}
        ).replace(b"\n", b"\r\n")
        # SSE joins multiple data lines with a newline. Split at a JSON
        # whitespace boundary so the joined payload remains valid JSON.
        final = (
            b'data: {"text":"done","meta_info":\r\n'
            b'data: {"finish_reason":{"type":"stop"}}}\r\n\r\n'
        )
        whole = first + final + b"data: [DONE]\r\n\r\n"
        utf8 = "共".encode("utf-8")
        cut = whole.index(utf8) + 1
        chunks = [whole[:cut], whole[cut : cut + 7], whole[cut + 7 :]]
        status = self.status_after(chunks)
        self.assertTrue(status.successful)
        self.assertFalse(status.failed)

    def test_error_or_any_data_after_done_invalidates_success(self):
        final = sse_event(
            {"meta_info": {"finish_reason": {"type": "length"}}}
        )
        error = sse_event({"error": "late failure"})
        ordinary = sse_event({"text": "late data", "meta_info": {}})
        for chunks in (
            [final, error, SSE_DONE],
            [final, SSE_DONE, error],
            [final, SSE_DONE, ordinary],
        ):
            with self.subTest(chunks=chunks):
                status = self.status_after(chunks)
                self.assertFalse(status.successful)
                self.assertTrue(status.failed)

    def test_finish_reason_must_be_on_the_final_data_event(self):
        final = sse_event(
            {"meta_info": {"finish_reason": {"type": "length"}}}
        )
        ordinary = sse_event(
            {"text": "unexpected continuation", "meta_info": {"finish_reason": None}}
        )
        second_finish = sse_event(
            {"meta_info": {"finish_reason": {"type": "stop"}}}
        )
        for chunks in ([final, ordinary, SSE_DONE], [final, second_finish, SSE_DONE]):
            with self.subTest(chunks=chunks):
                status = self.status_after(chunks)
                self.assertFalse(status.successful)
                self.assertTrue(status.failed)

    def test_malformed_trailing_frame_and_oversized_frame_fail_closed(self):
        final = sse_event(
            {"meta_info": {"finish_reason": {"type": "length"}}}
        )
        truncated = self.status_after([final, SSE_DONE, b"data: trailing"])
        self.assertFalse(truncated.successful)

        oversized = mini_lb._SessionStreamStatus()
        oversized.feed(b"data: " + b"x" * (8 * 1024 * 1024) + b"\n")
        self.assertTrue(oversized.failed)
        self.assertFalse(oversized.successful)


@unittest.skipIf(
    NATIVE_IMPORT_ERROR is not None,
    f"native HTTP dependencies unavailable: {NATIVE_IMPORT_ERROR}",
)
class SessionAffinityEntranceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.router = mini_lb.MiniLoadBalancer(make_args())
        self.backend = Backend()
        self.enterContext(mock.patch.object(mini_lb, "lb", self.router))
        self.enterContext(
            mock.patch.object(mini_lb.aiohttp, "ClientSession", self.backend.session)
        )
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mini_lb.app), base_url="http://router"
        )
        self.addAsyncCleanup(self.client.aclose)

    async def send(self, payload, session=None, final=False):
        headers = {}
        if session is not None:
            headers["X-SGLang-Session-ID"] = session
            headers["X-SGLang-Session-End"] = "1" if final else "0"
        response = await self.client.post("/generate", json=payload, headers=headers)
        return response

    def decode_bodies(self, rid=None):
        bodies = [body for url, body in self.backend.posts if "decode:" in url]
        return bodies if rid is None else [body for body in bodies if body.get("rid") == rid]

    async def assert_stream_outcome(self, rid, chunks, *, successful):
        state = self.router._session_affinity
        before_completed = 0 if state is None else state.counters["completed_sessions"]
        before_aborted = 0 if state is None else state.counters["aborted_sessions"]
        self.backend.stream_chunks_by_rid[rid] = chunks
        response = await self.send(
            {"input_ids": [1], "rid": rid, "stream": True},
            session=f"session-{rid}",
            final=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.content,
            b"".join(chunks),
            "the observer must not rewrite, drop, or append streamed bytes",
        )
        snapshot = self.router._session_state().snapshot()
        self.assertEqual(snapshot["active_sessions"], 0)
        self.assertEqual(self.router._decode_inflight_reservations, [0] * 8)
        if successful:
            self.assertEqual(snapshot["completed_sessions"], before_completed + 1)
            self.assertEqual(snapshot["aborted_sessions"], before_aborted)
        else:
            self.assertEqual(snapshot["completed_sessions"], before_completed)
            self.assertEqual(snapshot["aborted_sessions"], before_aborted + 1)
        return response

    async def test_headers_are_router_only_and_prefix_changes_keep_decode_rank(self):
        first = await self.send(
            {"input_ids": [1, 2, 3, 4], "rid": "first"}, session="slot-0"
        )
        second = await self.send(
            {"input_ids": [900, 901, 902, 903], "rid": "second"},
            session="slot-0",
            final=True,
        )
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(
            first.headers["x-sglang-decode-dp"],
            second.headers["x-sglang-decode-dp"],
        )
        self.assertEqual(first.headers["x-sglang-session-affinity"], "1")
        self.assertEqual(second.headers["x-sglang-session-affinity"], "1")
        for _, body in self.backend.posts:
            lowered = {key.lower() for key in body}
            self.assertNotIn("x-sglang-session-id", lowered)
            self.assertNotIn("x-sglang-session-end", lowered)
        self.assertEqual(self.router._session_state().snapshot()["active_sessions"], 0)

    async def test_end_releases_lease_only_after_the_full_response(self):
        self.router = mini_lb.MiniLoadBalancer(
            make_args(mini_lb_prefix_affinity_decode_capacity=1)
        )
        self.backend = Backend(decode_dp=1)
        with (
            mock.patch.object(mini_lb, "lb", self.router),
            mock.patch.object(
                mini_lb.aiohttp, "ClientSession", self.backend.session
            ),
        ):
            initial = await self.send(
                {"input_ids": [1], "rid": "initial"}, session="first"
            )
            self.assertEqual(initial.status_code, 200)
            self.backend.block_rids.add("ending")
            ending = asyncio.create_task(
                self.send(
                    {"input_ids": [2], "rid": "ending"},
                    session="first",
                    final=True,
                )
            )
            await wait_until(lambda: self.backend.started_by_rid.get("ending") == 2)
            newcomer = asyncio.create_task(
                self.send(
                    {"input_ids": [3], "rid": "newcomer"},
                    session="newcomer",
                    final=True,
                )
            )
            await wait_until(
                lambda: self.router._session_state().counters["waiting_requests"]
                == 1
            )
            self.assertEqual(self.backend.started_by_rid.get("newcomer", 0), 0)
            self.assertFalse(newcomer.done())

            self.backend.release_blocked.set()
            ending_response, newcomer_response = await asyncio.gather(
                ending, newcomer
            )
            self.assertEqual(ending_response.status_code, 200)
            self.assertEqual(newcomer_response.status_code, 200)
            self.assertEqual(
                self.router._session_state().snapshot()["active_sessions"], 0
            )

    async def test_cache_salt_creates_independent_session_domains(self):
        first = await self.send(
            {"input_ids": [1], "rid": "salt-a", "cache_salt": "tenant-a"},
            session="shared-id",
        )
        second = await self.send(
            {"input_ids": [1], "rid": "salt-b", "cache_salt": "tenant-b"},
            session="shared-id",
        )
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertNotEqual(
            first.headers["x-sglang-decode-dp"],
            second.headers["x-sglang-decode-dp"],
        )
        self.assertEqual(self.router._session_state().snapshot()["active_sessions"], 2)
        for salt in ("tenant-a", "tenant-b"):
            response = await self.send(
                {"input_ids": [2], "rid": f"end-{salt}", "cache_salt": salt},
                session="shared-id",
                final=True,
            )
            self.assertEqual(response.status_code, 200)
        self.assertEqual(self.router._session_state().snapshot()["active_sessions"], 0)

    async def test_invalid_headers_fail_before_backend_and_legacy_headerless_is_cap2(self):
        invalid = (
            {"X-SGLang-Session-End": "1"},
            {
                "X-SGLang-Session-ID": "valid",
                "X-SGLang-Session-End": "yes",
            },
            {"X-SGLang-Session-ID": "contains space"},
        )
        for headers in invalid:
            with self.subTest(headers=headers):
                response = await self.client.post(
                    "/generate", json={"input_ids": [1]}, headers=headers
                )
                self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(self.backend.posts, [])

        response = await self.client.post(
            "/generate", json={"input_ids": [1], "rid": "legacy"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("x-sglang-session-affinity", response.headers)
        self.assertIsNone(self.router._session_affinity)
        self.assertEqual(self.router._decode_inflight_reservations, [0] * 8)
        self.assertEqual(len(self.decode_bodies("legacy")), 1)

    async def test_upstream_exception_aborts_session_and_releases_inflight(self):
        self.backend.fail_decode_rids.add("failure")
        with self.assertRaisesRegex(RuntimeError, "synthetic decode failure"):
            await self.send(
                {"input_ids": [1], "rid": "failure"}, session="failed-session"
            )
        snapshot = self.router._session_state().snapshot()
        self.assertEqual(snapshot["active_sessions"], 0)
        self.assertEqual(snapshot["aborted_sessions"], 1)
        self.assertEqual(self.router._decode_inflight_reservations, [0] * 8)

    async def test_stream_error_event_aborts_session_instead_of_persisting_lease(self):
        self.backend.stream_error_rids.add("stream-error")
        response = await self.send(
            {"input_ids": [1], "rid": "stream-error", "stream": True},
            session="stream-error-session",
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("synthetic stream error", response.text)
        snapshot = self.router._session_state().snapshot()
        self.assertEqual(
            snapshot["active_sessions"],
            0,
            "a 200 SSE carrying an application error must abort the binding",
        )
        self.assertEqual(snapshot["aborted_sessions"], 1)
        self.assertEqual(self.router._decode_inflight_reservations, [0] * 8)

    async def test_normal_length_and_stop_finish_with_done_commit_exact_bytes(self):
        for finish_type in ("length", "stop"):
            with self.subTest(finish_type=finish_type):
                chunks = successful_stream(0, finish_type)
                await self.assert_stream_outcome(
                    f"normal-{finish_type}", chunks, successful=True
                )

    async def test_truncated_or_done_without_normal_finish_aborts_exact_bytes(self):
        intermediate = sse_event(
            {"text": "partial", "meta_info": {"finish_reason": None}}
        )
        final = sse_event(
            {
                "text": "complete",
                "meta_info": {"finish_reason": {"type": "length"}},
            }
        )
        cases = {
            "final-without-done": [intermediate, final],
            "done-without-final": [intermediate, SSE_DONE],
            "truncated-json": [intermediate, b'data: {"text":"cut'],
            "malformed-finish": [
                sse_event(
                    {
                        "text": "complete",
                        "meta_info": {"finish_reason": "length"},
                    }
                ),
                SSE_DONE,
            ],
            "abort-finish": [
                sse_event(
                    {
                        "text": "aborted",
                        "meta_info": {
                            "finish_reason": {
                                "type": "abort",
                                "message": "synthetic abort",
                            }
                        },
                    }
                ),
                SSE_DONE,
            ],
        }
        for rid, chunks in cases.items():
            with self.subTest(rid=rid):
                await self.assert_stream_outcome(rid, chunks, successful=False)

    async def test_error_after_terminal_finish_or_done_aborts_exact_bytes(self):
        final = sse_event(
            {
                "text": "complete",
                "meta_info": {"finish_reason": {"type": "length"}},
            }
        )
        error = sse_event({"error": "late stream failure"})
        cases = {
            "error-after-finish": [final, error, SSE_DONE],
            "error-after-done": [final, SSE_DONE, error],
        }
        for rid, chunks in cases.items():
            with self.subTest(rid=rid):
                await self.assert_stream_outcome(rid, chunks, successful=False)

    async def test_data_after_finish_reason_aborts_and_forwards_exact_bytes(self):
        final = sse_event(
            {
                "text": "complete",
                "meta_info": {"finish_reason": {"type": "length"}},
            }
        )
        cases = {
            "ordinary-after-finish": [
                final,
                sse_event(
                    {
                        "text": "unexpected continuation",
                        "meta_info": {"finish_reason": None},
                    }
                ),
                SSE_DONE,
            ],
            "second-finish": [
                final,
                sse_event(
                    {"meta_info": {"finish_reason": {"type": "stop"}}}
                ),
                SSE_DONE,
            ],
        }
        for rid, chunks in cases.items():
            with self.subTest(rid=rid):
                await self.assert_stream_outcome(rid, chunks, successful=False)

    async def test_split_utf8_frame_and_coalesced_events_forward_exact_bytes(self):
        first = sse_event(
            {"text": "共同前缀", "meta_info": {"finish_reason": None}}
        )
        final = sse_event(
            {
                "text": "完成",
                "meta_info": {"finish_reason": {"type": "stop"}},
            }
        )
        first_newline = first.rfind(b"\n\n")
        utf8_cut = first.index("共".encode("utf-8")) + 1
        chunks = [
            first[:utf8_cut],
            first[utf8_cut : first_newline + 1],
            first[first_newline + 1 :] + final + SSE_DONE,
        ]
        response = await self.assert_stream_outcome(
            "split-valid", chunks, successful=True
        )
        self.assertIn("共同前缀", response.content.decode("utf-8"))
        self.assertIn("完成", response.content.decode("utf-8"))

    async def test_stream_disconnect_aborts_session_and_releases_inflight(self):
        payload = json.dumps(
            {"input_ids": [1, 2, 3], "rid": "disconnect", "stream": True}
        ).encode()
        first_body = asyncio.Event()
        request_sent = False

        async def receive():
            nonlocal request_sent
            if not request_sent:
                request_sent = True
                return {"type": "http.request", "body": payload, "more_body": False}
            await first_body.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.body" and message.get("body"):
                first_body.set()

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/generate",
            "raw_path": b"/generate",
            "query_string": b"",
            "root_path": "",
            "headers": [
                (b"host", b"router"),
                (b"content-type", b"application/json"),
                (b"x-sglang-session-id", b"stream-session"),
                (b"x-sglang-session-end", b"0"),
            ],
            "client": ("127.0.0.1", 12345),
            "server": ("router", 80),
        }
        app_task = asyncio.create_task(mini_lb.app(scope, receive, send))
        try:
            await asyncio.wait_for(self.backend.first_stream_chunk.wait(), timeout=1)
            await asyncio.wait_for(app_task, timeout=1)
        finally:
            self.backend.finish_stream.set()
            if not app_task.done():
                app_task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await app_task

        snapshot = self.router._session_state().snapshot()
        self.assertEqual(snapshot["active_sessions"], 0)
        self.assertEqual(snapshot["aborted_sessions"], 1)
        self.assertEqual(self.router._decode_inflight_reservations, [0] * 8)


if __name__ == "__main__":
    unittest.main()
