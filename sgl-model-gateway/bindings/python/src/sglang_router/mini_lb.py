"""
Minimal HTTP load balancer for prefill and decode servers for testing.
"""

import asyncio
import hashlib
import ipaddress
import logging
import random
import urllib
import warnings
from http import HTTPStatus
from itertools import chain, count
from typing import Optional

import aiohttp
import orjson
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import ORJSONResponse, Response, StreamingResponse
from sglang_router.router_args import RouterArgs
from sglang_router.session_affinity import SessionAffinity, SessionReservation

logger = logging.getLogger(__name__)

AIOHTTP_STREAM_READ_CHUNK_SIZE = (
    1024 * 64
)  # 64KB, to prevent aiohttp's "Chunk too big" error


class _SessionStreamStatus:
    """Observe native SSE completion without changing forwarded bytes.

    HTTP 200 alone does not distinguish application errors or a truncated SSE
    stream. Bound the observer's memory even for a malformed upstream response.
    """

    def __init__(self):
        self.pending = b""
        self.data = []
        self.frame_size = 0
        self.failed = False
        self.done = False
        self.finished = False

    def feed(self, chunk):
        if self.failed:
            return
        self.pending += chunk
        while b"\n" in self.pending:
            line, self.pending = self.pending.split(b"\n", 1)
            line = line.removesuffix(b"\r")
            self.frame_size += len(line)
            if self.frame_size > 8 * 1024 * 1024:
                self.failed = True
                break
            if line.startswith(b"data:"):
                self.data.append(line[5:].lstrip(b" "))
            elif not line:
                if self.data:
                    self._event(b"\n".join(self.data))
                self.data = []
                self.frame_size = 0
        if self.frame_size + len(self.pending) > 8 * 1024 * 1024:
            self.failed = True
        if self.failed:
            self.pending = b""
            self.data = []

    def _event(self, data):
        if self.done:
            self.failed = True
            return
        if data == b"[DONE]":
            self.done = True
            return
        if self.finished:
            self.failed = True
            return
        try:
            obj = orjson.loads(data)
            if not isinstance(obj, dict) or "error" in obj:
                self.failed = True
                return
            reason = obj.get("meta_info", {}).get("finish_reason")
            if reason is not None:
                if not isinstance(reason, dict) or reason.get("type") not in (
                    "length",
                    "stop",
                ):
                    self.failed = True
                else:
                    self.finished = True
        except (ValueError, TypeError, AttributeError):
            self.failed = True

    @property
    def successful(self):
        return (
            self.done
            and self.finished
            and not self.failed
            and not self.pending.strip()
            and not self.data
        )


class _DecodeReservation:
    """An event-loop-local, idempotently released D-rank reservation."""

    __slots__ = ("_owner", "rank", "_released")

    def __init__(self, owner, rank):
        self._owner = owner
        self.rank = rank
        self._released = False

    def release(self):
        if self._released:
            return False
        self._released = True
        self._owner._release_decode_reservation(self.rank)
        return True


class _ReservationStreamingResponse(StreamingResponse):
    """Keep a D reservation until streaming completion or disconnect."""

    def __init__(self, *args, decode_reservation, **kwargs):
        super().__init__(*args, **kwargs)
        self._decode_reservation = decode_reservation

    async def __call__(self, scope, receive, send):
        try:
            return await super().__call__(scope, receive, send)
        except BaseException:
            if isinstance(self._decode_reservation, SessionReservation):
                self._decode_reservation.mark_failed()
            raise
        finally:
            self._decode_reservation.release()


def maybe_wrap_ipv6_address(address: str) -> str:
    try:
        ipaddress.IPv6Address(address)
        return f"[{address}]"
    except ValueError:
        return address


class MiniLoadBalancer:
    def __init__(
        self,
        router_args: RouterArgs,
    ):
        self._validate_router_args(router_args)

        self.host = router_args.host
        self.port = router_args.port
        self.timeout = router_args.request_timeout_secs
        self.prefill_urls = [url[0] for url in router_args.prefill_urls]
        self.prefill_bootstrap_ports = [url[1] for url in router_args.prefill_urls]
        self.decode_urls = router_args.decode_urls
        self.test_external_dp_routing = router_args.test_external_dp_routing
        self.prefix_affinity = router_args.mini_lb_prefix_affinity
        self.affinity_prefix_length = router_args.mini_lb_prefix_affinity_length
        self.affinity_decode_capacity = (
            router_args.mini_lb_prefix_affinity_decode_capacity
        )
        self.session_affinity_enabled = router_args.mini_lb_session_affinity
        self.session_affinity_idle_timeout = (
            router_args.mini_lb_session_affinity_idle_timeout_secs
        )
        self._session_affinity = None
        self.prefill_dp_size = None
        self.decode_dp_size = None
        self._dp_size_lock = asyncio.Lock()
        self._decode_inflight_reservations = []
        self._decode_reservation_total = 0
        self._decode_reservation_overrides = 0
        self._decode_reservation_all_full_fallbacks = 0
        # A fresh random starting point avoids predictable reuse across restarts;
        # the counter guarantees distinct rooms within this router process.
        self._affinity_rooms = (
            count(random.getrandbits(62)) if self.prefix_affinity else None
        )
        if self.affinity_decode_capacity:
            logger.warning(
                "[MiniLB] Bounded decode affinity enabled: capacity=%d, "
                "router-local in-flight reservations only; all-full falls back "
                "to affinity.",
                self.affinity_decode_capacity,
            )

    def _validate_router_args(self, router_args: RouterArgs):
        if (
            router_args.mini_lb_prefix_affinity
            or router_args.mini_lb_prefix_affinity_decode_capacity
            or router_args.mini_lb_session_affinity
        ):
            router_args._validate_router_args()
        if router_args.mini_lb_prefix_affinity:
            logger.warning(
                "[MiniLB] Prefix affinity enabled: deterministic DP ranks, "
                "no cache occupancy or load feedback; hot prefixes can queue."
            )
        else:
            logger.warning(
                "\x1b[33mMiniLB is only for debugging purposes, it only supports random policy!\033[0m"
            )

        # NOTE: too many arguments unsupported, just validate some important ones
        if router_args.policy != "random":
            logger.warning("[MiniLB] Overriding policy to random")
            router_args.policy = "random"

        if not router_args.pd_disaggregation:
            raise ValueError("MiniLB only supports PD disaggregation mode")

        if len(router_args.prefill_urls) == 0 or len(router_args.decode_urls) == 0:
            raise ValueError(
                "MiniLB requires at least one prefill and one decode server"
            )

    def start(self):
        global lb
        lb = self
        uvicorn.run(app, host=self.host, port=self.port)

    async def _ensure_dp_sizes(self):
        if self.prefix_affinity:
            await self._ensure_affinity_dp_sizes()
            return
        if self.prefill_dp_size is not None:
            return
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.prefill_urls[0]}/server_info") as resp:
                info = await resp.json()
                self.prefill_dp_size = len(info.get("internal_states", [1]))
            async with session.get(f"{self.decode_urls[0]}/server_info") as resp:
                info = await resp.json()
                self.decode_dp_size = len(info.get("internal_states", [1]))
        logger.info(
            f"[MiniLB] DP sizes: prefill={self.prefill_dp_size}, decode={self.decode_dp_size}"
        )

    @staticmethod
    def _dp_size_from_info(info):
        if not isinstance(info, dict):
            raise ValueError("server_info must be an object")
        states = info.get("internal_states")
        size = info.get("dp_size")
        if size is None and isinstance(states, list):
            size = len(states)
        if type(size) is not int or size <= 0:
            raise ValueError("server_info has no positive DP size")
        if states is not None and (not isinstance(states, list) or len(states) != size):
            raise ValueError("server_info DP size and internal_states disagree")
        return size

    async def _ensure_affinity_dp_sizes(self):
        async with self._dp_size_lock:
            if self.prefill_dp_size is not None and self.decode_dp_size is not None:
                return
            sizes = []
            try:
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=min(self.timeout, 30))
                ) as session:
                    for url in (self.prefill_urls[0], self.decode_urls[0]):
                        async with session.get(f"{url}/server_info") as response:
                            response.raise_for_status()
                            sizes.append(self._dp_size_from_info(await response.json()))
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
                logger.warning("[MiniLB] Prefix affinity DP discovery failed: %s", exc)
                raise HTTPException(
                    status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                    detail="Prefix affinity DP discovery failed; retry when both workers are ready",
                ) from exc
            # Publish both only after both workers have supplied valid metadata.
            self.prefill_dp_size, self.decode_dp_size = sizes
            if self.affinity_decode_capacity:
                self._decode_inflight_reservations = [0] * self.decode_dp_size
            logger.info(
                "[MiniLB] Prefix affinity DP sizes: prefill=%d, decode=%d",
                self.prefill_dp_size,
                self.decode_dp_size,
            )

    def _affinity_digest(self, request):
        if any(
            request.get(key) is not None
            for key in (
                "routed_dp_rank",
                "data_parallel_rank",
                "disagg_prefill_dp_rank",
            )
        ):
            raise HTTPException(
                400, "Prefix affinity assigns ranks; omit client DP rank fields"
            )
        sampling = request.get("sampling_params")
        if sampling is not None and (
            not isinstance(sampling, dict)
            or sampling.get("n", 1) != 1
            or sampling.get("beam_width", 1) != 1
        ):
            raise HTTPException(
                400, "Prefix affinity requires one sample (n=1, no beam search)"
            )
        for field in ("extra_key", "cache_salt"):
            if request.get(field) is not None and not isinstance(request[field], str):
                raise HTTPException(400, f"Prefix affinity requires scalar {field}")
        ids, text = request.get("input_ids"), request.get("text")
        if (ids is None) == (text is None):
            raise HTTPException(
                400, "Prefix affinity requires exactly one of input_ids or text"
            )
        if ids is not None:
            if (
                not isinstance(ids, list)
                or not ids
                or any(
                    type(token) is not int or not 0 <= token < 2**63 for token in ids
                )
            ):
                raise HTTPException(
                    400, "Prefix affinity requires one nonempty flat list of token IDs"
                )
            kind, prefix = "input_ids", ids[: self.affinity_prefix_length]
        else:
            if not isinstance(text, str) or not text:
                raise HTTPException(
                    400, "Prefix affinity requires one nonempty text string"
                )
            kind, prefix = "text", text[: self.affinity_prefix_length]
        key = orjson.dumps(
            [kind, prefix, request.get("extra_key"), request.get("cache_salt")]
        )
        return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")

    def _reserve_decode_rank(self, affinity_rank, digest):
        """Select and reserve one valid D rank without yielding the event loop.

        The capacity is deliberately soft: when every rank is at capacity, the
        original affinity rank is retained. This preserves liveness and prefix
        behavior during bursts while making the overload visible in the local
        reservation counters.
        """
        counts = self._decode_inflight_reservations
        if len(counts) != self.decode_dp_size:
            raise HTTPException(
                status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                detail="Bounded decode affinity is not initialized",
            )
        if not 0 <= affinity_rank < len(counts):
            raise HTTPException(
                status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                detail="Affinity selected an invalid decode DP rank",
            )

        selected_rank = affinity_rank
        if counts[affinity_rank] >= self.affinity_decode_capacity:
            eligible = [
                rank
                for rank, inflight in enumerate(counts)
                if inflight < self.affinity_decode_capacity
            ]
            if eligible:
                minimum = min(counts[rank] for rank in eligible)
                least_reserved = [rank for rank in eligible if counts[rank] == minimum]
                # Rotate deterministic ties by the affinity key. Repeated hot
                # prefixes still spread because every reservation updates counts
                # atomically before the next backend await.
                selected_rank = least_reserved[digest % len(least_reserved)]
                self._decode_reservation_overrides += 1
            else:
                self._decode_reservation_all_full_fallbacks += 1

        counts[selected_rank] += 1
        self._decode_reservation_total += 1
        return selected_rank, _DecodeReservation(self, selected_rank)

    def _release_decode_reservation(self, rank):
        counts = self._decode_inflight_reservations
        if not 0 <= rank < len(counts) or counts[rank] <= 0:
            logger.error(
                "[MiniLB] Invalid decode reservation release: rank=%s counts=%s",
                rank,
                counts,
            )
            return
        counts[rank] -= 1
        if self._session_affinity is not None:
            self._session_affinity.changed.set()

    def decode_affinity_snapshot(self):
        return {
            "enabled": bool(self.affinity_decode_capacity),
            "capacity": self.affinity_decode_capacity,
            "inflight_by_rank": list(self._decode_inflight_reservations),
            "total_reservations": self._decode_reservation_total,
            "overrides": self._decode_reservation_overrides,
            "all_full_fallbacks": self._decode_reservation_all_full_fallbacks,
        }

    def _session_state(self):
        if self._session_affinity is None:
            self._session_affinity = SessionAffinity(
                self._decode_inflight_reservations,
                self.affinity_decode_capacity,
                self.session_affinity_idle_timeout,
                self.timeout,
            )
        return self._session_affinity

    def session_context(self, headers):
        session_id = headers.get("x-sglang-session-id")
        ending = headers.get("x-sglang-session-end")
        if not self.session_affinity_enabled:
            return None
        if session_id is None:
            if ending is not None:
                raise HTTPException(400, "Session end requires a session ID")
            return None  # Legacy callers retain the original cap2 policy.
        if ending not in (None, "0", "1"):
            raise HTTPException(400, "X-SGLang-Session-End must be 0 or 1")
        try:
            SessionAffinity.key(session_id, {})
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return session_id, ending == "1"

    async def _prepare_affinity_requests(self, request, endpoint, session_context=None):
        if endpoint != "generate":
            raise HTTPException(400, "MiniLB prefix affinity supports only /generate")
        digest = self._affinity_digest(request)
        await self._ensure_dp_sizes()
        p_rank = digest % self.prefill_dp_size
        d_rank = digest % self.decode_dp_size
        decode_reservation = None
        prefill_req, decode_req = request.copy(), request.copy()
        # Remove nullable aliases too, so each body has one authoritative rank.
        for body in (prefill_req, decode_req):
            body.pop("data_parallel_rank", None)
            body.pop("disagg_prefill_dp_rank", None)
        prefill_req["routed_dp_rank"] = p_rank
        if session_context is not None:
            if not self.session_affinity_enabled:
                raise HTTPException(400, "Session affinity is disabled")
            session_id, final = session_context
            try:
                decode_reservation = await self._session_state().reserve(
                    session_id,
                    request,
                    d_rank,
                    final=final,
                )
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            except TimeoutError as exc:
                raise HTTPException(503, str(exc)) from exc
            selected = decode_reservation.rank
            self._decode_reservation_total += 1
            self._decode_reservation_overrides += int(selected != d_rank)
            d_rank = selected
        elif self.affinity_decode_capacity:
            # There is no await between observing counts and incrementing the
            # selected rank, so concurrent asyncio requests cannot overselect a
            # stale minimum within this router process.
            d_rank, decode_reservation = self._reserve_decode_rank(d_rank, digest)
        decode_req["routed_dp_rank"] = d_rank
        decode_req["disagg_prefill_dp_rank"] = p_rank
        return prefill_req, decode_req, decode_reservation

    def new_bootstrap_room(self):
        if not self.prefix_affinity:
            return _generate_bootstrap_room()
        room = next(self._affinity_rooms)
        if room >= 2**63:
            raise HTTPException(503, "Prefix affinity bootstrap room counter exhausted")
        return room

    def _fork_dp_requests(self, request):
        p_rank = random.randint(0, self.prefill_dp_size - 1)
        d_rank = random.randint(0, self.decode_dp_size - 1)

        prefill_req = request.copy()
        decode_req = request.copy()
        prefill_req["routed_dp_rank"] = p_rank
        decode_req["routed_dp_rank"] = d_rank
        decode_req["disagg_prefill_dp_rank"] = p_rank

        return prefill_req, decode_req, d_rank

    def select_pair(self, request=None):
        assert len(self.prefill_urls) > 0, "No prefill servers available"
        assert len(self.decode_urls) > 0, "No decode servers available"
        if self.prefix_affinity and request is not None:
            # Keep a prefix on the same P instance as well as the same P DP
            # rank. Otherwise multiple P workers scatter reusable KV/Mamba
            # state even though their DP ranks have stable affinity.
            pidx = self._affinity_digest(request) % len(self.prefill_urls)
        else:
            pidx = random.randint(0, len(self.prefill_urls) - 1)
        didx = random.randint(0, len(self.decode_urls) - 1)
        return (
            self.prefill_urls[pidx],
            self.prefill_bootstrap_ports[pidx],
            self.decode_urls[didx],
        )

    async def generate(
        self,
        modified_request,
        prefill_server,
        decode_server,
        endpoint,
        session_context=None,
    ) -> ORJSONResponse:
        assert endpoint[0] != "/", f"Endpoint should not start with '/': {endpoint}"

        expected_decode_dp_rank = None
        decode_reservation = None
        if self.prefix_affinity:
            (
                prefill_req,
                decode_req,
                decode_reservation,
            ) = await self._prepare_affinity_requests(
                modified_request, endpoint, session_context
            )
        elif self.test_external_dp_routing:
            await self._ensure_dp_sizes()
            prefill_req, decode_req, expected_decode_dp_rank = self._fork_dp_requests(
                modified_request
            )
        else:
            prefill_req = modified_request
            decode_req = modified_request

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(
                    total=self.timeout
                )  # Add timeout for request reliability
            ) as session:
                tasks = [
                    session.post(f"{prefill_server}/{endpoint}", json=prefill_req),
                    session.post(f"{decode_server}/{endpoint}", json=decode_req),
                ]

                # Wait for both responses to complete. Prefill should end first.
                prefill_response, decode_response = await asyncio.gather(*tasks)

                if "return_logprob" in modified_request:
                    prefill_json = await prefill_response.json()
                    ret_json = await decode_response.json()

                    # merge `meta_info.input_token_logprobs` from prefill to decode
                    if "meta_info" in ret_json:
                        if "input_token_logprobs" in ret_json["meta_info"]:
                            ret_json["meta_info"]["input_token_logprobs"] = (
                                prefill_json["meta_info"]["input_token_logprobs"]
                                + ret_json["meta_info"]["input_token_logprobs"]
                            )
                else:
                    ret_json = await decode_response.json()

                if expected_decode_dp_rank is not None:
                    actual = ret_json.get("meta_info", {}).get("dp_rank")
                    if actual != expected_decode_dp_rank:
                        return ORJSONResponse(
                            content={
                                "error": f"DP rank mismatch: expected {expected_decode_dp_rank}, got {actual}"
                            },
                            status_code=500,
                        )

                headers = None
                if isinstance(decode_reservation, SessionReservation):
                    if (
                        prefill_response.status == decode_response.status == 200
                        and "error" not in ret_json
                    ):
                        decode_reservation.mark_successful()
                    headers = {
                        "X-SGLang-Session-Affinity": "1",
                        "X-SGLang-Decode-DP": str(decode_reservation.rank),
                    }
                return ORJSONResponse(
                    content=ret_json,
                    status_code=decode_response.status,
                    headers=headers,
                )
        finally:
            if decode_reservation is not None:
                decode_reservation.release()

    async def generate_stream(
        self,
        modified_request,
        prefill_server,
        decode_server,
        endpoint="generate",
        session_context=None,
    ):
        if self.test_external_dp_routing:
            warnings.warn("--test-external-dp-routing is not supported with streaming")

        assert endpoint[0] != "/", f"Endpoint should not start with '/': {endpoint}"

        decode_reservation = None
        if self.prefix_affinity:
            (
                prefill_req,
                decode_req,
                decode_reservation,
            ) = await self._prepare_affinity_requests(
                modified_request, endpoint, session_context
            )
        else:
            prefill_req = decode_req = modified_request

        async def stream_results():
            stream_status = (
                _SessionStreamStatus()
                if isinstance(decode_reservation, SessionReservation)
                else None
            )
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(
                    total=self.timeout
                )  # Add timeout for request reliability
            ) as session:
                # Create the tasks for both prefill and decode requests
                tasks = [
                    session.post(f"{prefill_server}/{endpoint}", json=prefill_req),
                    session.post(f"{decode_server}/{endpoint}", json=decode_req),
                ]

                # Wait for both responses to complete. Since this is streaming, they return immediately.
                prefill_response, decode_response = await asyncio.gather(*tasks)

                if modified_request.get("return_logprob", False):
                    prefill_chunks = []
                    async for chunk in prefill_response.content:
                        prefill_chunks.append(chunk)

                    first_prefill_chunk = (
                        prefill_chunks[0].decode("utf-8")[5:].strip("\n")
                    )
                    first_prefill_chunk_json = orjson.loads(first_prefill_chunk)

                    async for chunk in decode_response.content:
                        if stream_status is not None:
                            stream_status.feed(chunk)
                        # Note: This is inefficient
                        # merge prefill input_token_logprobs, output_token_logprobs to decode
                        decoded_chunk = chunk.decode("utf-8")
                        if (
                            decoded_chunk
                            and decoded_chunk.startswith("data:")
                            and "[DONE]" not in decoded_chunk
                        ):
                            ret_json = orjson.loads(decoded_chunk[5:].strip("\n"))
                            ret_json["meta_info"]["input_token_logprobs"] = (
                                first_prefill_chunk_json["meta_info"][
                                    "input_token_logprobs"
                                ]
                                + ret_json["meta_info"]["input_token_logprobs"]
                            )

                            yield b"data: " + orjson.dumps(ret_json) + b"\n\n"
                        else:
                            yield chunk
                else:
                    async for chunk in decode_response.content.iter_chunked(
                        AIOHTTP_STREAM_READ_CHUNK_SIZE
                    ):
                        if stream_status is not None:
                            stream_status.feed(chunk)
                        yield chunk
                if isinstance(decode_reservation, SessionReservation):
                    if (
                        prefill_response.status == decode_response.status == 200
                        and stream_status.successful
                    ):
                        decode_reservation.mark_successful()

        if decode_reservation is None:
            return StreamingResponse(
                stream_results(),
                media_type="text/event-stream",
            )
        try:
            return _ReservationStreamingResponse(
                stream_results(),
                media_type="text/event-stream",
                decode_reservation=decode_reservation,
                headers=(
                    {
                        "X-SGLang-Session-Affinity": "1",
                        "X-SGLang-Decode-DP": str(decode_reservation.rank),
                    }
                    if isinstance(decode_reservation, SessionReservation)
                    else None
                ),
            )
        except BaseException:
            decode_reservation.release()
            raise


app = FastAPI()
lb: Optional[MiniLoadBalancer] = None


@app.get("/health")
async def health_check():
    return Response(status_code=200)


@app.get("/health_generate")
async def health_generate():
    async with aiohttp.ClientSession() as session:
        # Create the tasks
        tasks = []
        for server in chain(lb.prefill_urls, lb.decode_urls):
            tasks.append(session.get(f"{server}/health_generate"))
        for i, response in enumerate(asyncio.as_completed(tasks)):
            await response
    return Response(status_code=200)


@app.post("/flush_cache")
async def flush_cache(timeout: Optional[float] = None):
    # `timeout` must reach the workers. The scheduler treats a missing or
    # non-positive timeout as "flush now, skip the idle check", so dropping it
    # here frees KV buffers while a PD KV transfer is still reading them: the
    # transfer then fails for real and the peer session gets blacklisted.
    # Forwarding it keeps the scheduler on its deferred, drain-first path.
    params = None if timeout is None else {"timeout": timeout}
    async with aiohttp.ClientSession() as session:
        # Create the tasks
        tasks = []
        for server in chain(lb.prefill_urls, lb.decode_urls):
            tasks.append(session.post(f"{server}/flush_cache", params=params))
        for i, response in enumerate(asyncio.as_completed(tasks)):
            await response
    return Response(status_code=200)


# TODO: Remove `/get_server_info` alias after one release-cycle deprecation window.
@app.get("/server_info")
@app.get("/get_server_info")
async def get_server_info():
    prefill_infos = []
    decode_infos = []
    all_internal_states = []

    async with aiohttp.ClientSession() as session:
        for server in lb.prefill_urls:
            server_info = await session.get(f"{server}/server_info")
            prefill_infos.append(await server_info.json())
        for server in lb.decode_urls:
            server_info = await session.get(f"{server}/server_info")
            info_json = await server_info.json()
            decode_infos.append(info_json)
            # Extract internal_states from decode servers
            if "internal_states" in info_json:
                all_internal_states.extend(info_json["internal_states"])

    # Return format expected by bench_one_batch_server.py
    if all_internal_states:
        result = {
            "internal_states": all_internal_states,
            "prefill": prefill_infos,
            "decode": decode_infos,
        }
    else:
        # Fallback with dummy data if no internal states found
        result = {
            "internal_states": [
                {
                    "last_gen_throughput": 0.0,
                    "avg_spec_accept_length": None,
                }
            ],
            "prefill": prefill_infos,
            "decode": decode_infos,
        }
    if lb.affinity_decode_capacity:
        result["mini_lb_decode_affinity"] = lb.decode_affinity_snapshot()
    if lb.session_affinity_enabled:
        await lb._ensure_dp_sizes()
        result["mini_lb_session_affinity"] = lb._session_state().snapshot()
    return result


async def _get_model_info_impl():
    if not lb or not lb.prefill_urls:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail="There is no server registered",
        )

    target_server_url = lb.prefill_urls[0]
    endpoint_url = f"{target_server_url}/model_info"

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(endpoint_url) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise HTTPException(
                        status_code=HTTPStatus.BAD_GATEWAY,
                        detail=(
                            f"Failed to get model info from {target_server_url}"
                            f"Status: {response.status}, Response: {error_text}"
                        ),
                    )

                model_info_json = await response.json()
                return ORJSONResponse(content=model_info_json)

        except aiohttp.ClientError:
            raise HTTPException(
                status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                detail=f"Failed to get model info from backend",
            )


@app.get("/model_info")
async def model_info():
    return await _get_model_info_impl()


@app.get("/get_model_info")
async def get_model_info():
    return await _get_model_info_impl()


@app.post("/generate")
async def handle_generate_request(request_data: dict, request: Request = None):
    session_context = lb.session_context(request.headers if request is not None else {})
    if lb.prefix_affinity:
        # Validate before batch-shape inference and before either PD request.
        lb._affinity_digest(request_data)
    prefill_server, bootstrap_port, decode_server = lb.select_pair(request_data)

    # Parse and transform prefill_server for bootstrap data
    parsed_url = urllib.parse.urlparse(prefill_server)
    hostname = maybe_wrap_ipv6_address(parsed_url.hostname)
    modified_request = request_data.copy()

    batch_size = _get_request_batch_size(modified_request)
    if batch_size is not None:
        modified_request.update(
            {
                "bootstrap_host": [hostname] * batch_size,
                "bootstrap_port": [bootstrap_port] * batch_size,
                "bootstrap_room": [lb.new_bootstrap_room() for _ in range(batch_size)],
            }
        )
    else:
        modified_request.update(
            {
                "bootstrap_host": hostname,
                "bootstrap_port": bootstrap_port,
                "bootstrap_room": lb.new_bootstrap_room(),
            }
        )

    if request_data.get("stream", False):
        return await lb.generate_stream(
            modified_request,
            prefill_server,
            decode_server,
            "generate",
            session_context=session_context,
        )
    else:
        return await lb.generate(
            modified_request,
            prefill_server,
            decode_server,
            "generate",
            session_context=session_context,
        )


async def _forward_to_backend(request_data: dict, endpoint_name: str):
    prefill_server, bootstrap_port, decode_server = lb.select_pair(request_data)

    # Parse and transform prefill_server for bootstrap data
    parsed_url = urllib.parse.urlparse(prefill_server)
    hostname = maybe_wrap_ipv6_address(parsed_url.hostname)
    modified_request = request_data.copy()
    modified_request.update(
        {
            "bootstrap_host": hostname,
            "bootstrap_port": bootstrap_port,
            "bootstrap_room": _generate_bootstrap_room(),
        }
    )

    if request_data.get("stream", False):
        return await lb.generate_stream(
            modified_request,
            prefill_server,
            decode_server,
            endpoint=endpoint_name,
        )
    else:
        return await lb.generate(
            modified_request,
            prefill_server,
            decode_server,
            endpoint=endpoint_name,
        )


@app.post("/v1/chat/completions")
async def handle_chat_completion_request(request_data: dict):
    return await _forward_to_backend(request_data, "v1/chat/completions")


@app.post("/v1/completions")
async def handle_completion_request(request_data: dict):
    return await _forward_to_backend(request_data, "v1/completions")


def _generate_bootstrap_room():
    return random.randint(0, 2**63 - 1)


# We may utilize `GenerateReqInput`'s logic later
def _get_request_batch_size(request):
    if (text := request.get("text")) is not None:
        return None if isinstance(text, str) else len(text)
    if (input_ids := request.get("input_ids")) is not None:
        return None if isinstance(input_ids[0], int) else len(input_ids)
    return None


@app.get("/v1/models")
async def get_models():
    prefill_server = lb.prefill_urls[0]  # Get the first prefill server
    async with aiohttp.ClientSession() as session:
        try:
            response = await session.get(f"{prefill_server}/v1/models")
            if response.status != 200:
                raise HTTPException(
                    status_code=response.status,
                    detail=f"Prefill server error: Status {response.status}",
                )
            return ORJSONResponse(content=await response.json())
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
