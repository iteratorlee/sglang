"""Offline tests for MiniLB's opt-in bounded decode prefix affinity."""

import asyncio
import unittest
from unittest import mock

from sglang_router import mini_lb
from sglang_router.router_args import RouterArgs


def make_lb(capacity=4, prefill_dp=2, decode_dp=8):
    result = mini_lb.MiniLoadBalancer(
        RouterArgs(
            mini_lb=True,
            pd_disaggregation=True,
            policy="random",
            prefill_urls=[("http://prefill:31194", 8998)],
            decode_urls=["http://decode:32195"],
            mini_lb_prefix_affinity=True,
            mini_lb_prefix_affinity_length=4,
            mini_lb_prefix_affinity_decode_capacity=capacity,
        )
    )
    result.prefill_dp_size = prefill_dp
    result.decode_dp_size = decode_dp
    if capacity:
        result._decode_inflight_reservations = [0] * decode_dp
    return result


def request_for_rank(lb, wanted_rank, **extra):
    for token in range(100_000):
        request = {
            "input_ids": [token, 7, 11, 13, 17],
            "bootstrap_host": "prefill",
            "bootstrap_port": 8998,
            "bootstrap_room": 123456,
            **extra,
        }
        if lb._affinity_digest(request) % lb.decode_dp_size == wanted_rank:
            return request
    raise AssertionError(f"could not find an input for rank {wanted_rank}")


class ConfigTests(unittest.TestCase):
    def test_default_is_off_and_invalid_combinations_fail(self):
        self.assertEqual(RouterArgs().mini_lb_prefix_affinity_decode_capacity, 0)
        for updates in (
            {"mini_lb_prefix_affinity_decode_capacity": -1},
            {"mini_lb_prefix_affinity_decode_capacity": 4},
        ):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                mini_lb.MiniLoadBalancer(
                    RouterArgs(
                        mini_lb=True,
                        pd_disaggregation=True,
                        policy="random",
                        prefill_urls=[("http://prefill:31194", 8998)],
                        decode_urls=["http://decode:32195"],
                        **updates,
                    )
                )

    def test_three_prefill_workers_keep_prefix_affinity(self):
        lb = mini_lb.MiniLoadBalancer(
            RouterArgs(
                mini_lb=True,
                pd_disaggregation=True,
                policy="random",
                prefill_urls=[(f"http://prefill-{i}:31116", 18916) for i in range(3)],
                decode_urls=["http://decode:32117"],
                mini_lb_prefix_affinity=True,
                mini_lb_prefix_affinity_length=4,
            )
        )
        counts = [0] * 3
        for family in range(90):
            request = {
                "input_ids": [family, 7, 11, 13, 17],
                "cache_salt": f"session-{family}",
            }
            first = lb.select_pair(request)
            second = lb.select_pair({**request, "bootstrap_room": family + 1000})
            self.assertEqual(first, second)
            self.assertEqual(first[1], 18916)
            counts[lb.prefill_urls.index(first[0])] += 1
        self.assertTrue(all(count >= 15 for count in counts), counts)


class SelectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_affinity_uses_least_reserved_valid_rank(self):
        lb = make_lb(decode_dp=4)
        lb._decode_inflight_reservations[:] = [4, 1, 2, 3]
        request = request_for_rank(lb, 0)
        prefill, decode, reservation = await lb._prepare_affinity_requests(
            request, "generate"
        )

        self.assertEqual(decode["routed_dp_rank"], 1)
        self.assertEqual(prefill["routed_dp_rank"], decode["disagg_prefill_dp_rank"])
        self.assertEqual(prefill["bootstrap_room"], decode["bootstrap_room"])
        self.assertTrue(reservation.release())
        self.assertFalse(reservation.release())
        self.assertEqual(lb._decode_inflight_reservations, [4, 1, 2, 3])
        self.assertEqual(lb.decode_affinity_snapshot()["overrides"], 1)

    async def test_ties_and_all_full_bursts_balance(self):
        lb = make_lb(decode_dp=4)
        lb._decode_inflight_reservations[:] = [4, 0, 0, 0]
        request = request_for_rank(lb, 0)
        reservations = []
        selected = []
        for _ in range(6):
            _, decode, reservation = await lb._prepare_affinity_requests(
                request, "generate"
            )
            selected.append(decode["routed_dp_rank"])
            reservations.append(reservation)
        self.assertEqual(set(selected[:3]), {1, 2, 3})
        self.assertLessEqual(
            max(lb._decode_inflight_reservations[1:])
            - min(lb._decode_inflight_reservations[1:]),
            1,
        )
        for reservation in reservations:
            reservation.release()

        lb._decode_inflight_reservations[:] = [4, 4, 4, 4]
        request = request_for_rank(lb, 2)
        reservations = []
        selected = []
        for _ in range(8):
            _, decode, reservation = await lb._prepare_affinity_requests(
                request, "generate"
            )
            selected.append(decode["routed_dp_rank"])
            reservations.append(reservation)
        self.assertEqual(selected[0], 2)
        self.assertEqual(set(selected[:4]), {0, 1, 2, 3})
        self.assertEqual(lb._decode_inflight_reservations, [6, 6, 6, 6])
        self.assertEqual(lb.decode_affinity_snapshot()["all_full_fallbacks"], 8)
        for reservation in reservations:
            reservation.release()

    async def test_routes_stay_valid_and_cleanup_has_no_residue(self):
        lb = make_lb(prefill_dp=3, decode_dp=8)
        reservations = []
        for family in range(64):
            request = {
                "input_ids": [family, 2, 3, 4],
                "bootstrap_host": "prefill",
                "bootstrap_port": 8998,
                "bootstrap_room": 9000 + family,
            }
            prefill, decode, reservation = await lb._prepare_affinity_requests(
                request, "generate"
            )
            self.assertTrue(0 <= prefill["routed_dp_rank"] < 3)
            self.assertTrue(0 <= decode["routed_dp_rank"] < 8)
            self.assertEqual(
                decode["disagg_prefill_dp_rank"], prefill["routed_dp_rank"]
            )
            self.assertEqual(prefill["bootstrap_room"], decode["bootstrap_room"])
            reservations.append(reservation)
        for reservation in reservations:
            reservation.release()
        self.assertEqual(lb._decode_inflight_reservations, [0] * 8)


class FakeResponse:
    status = 200

    def __init__(self, rank=0):
        self.rank = rank

    async def json(self):
        return {"text": "ok", "meta_info": {"dp_rank": self.rank}}


class BlockingSession:
    def __init__(self, mode):
        self.mode = mode
        self.started = 0
        self.all_started = asyncio.Event()
        self.never = asyncio.Event()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *unused):
        return False

    async def post(self, url, *, json):
        self.started += 1
        if self.started == 2:
            self.all_started.set()
        await asyncio.sleep(0)
        if self.mode == "error" and "decode:" in url:
            raise RuntimeError("synthetic upstream failure")
        if self.mode == "block":
            await self.never.wait()
        return FakeResponse(json.get("routed_dp_rank", 0))


class LifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_error_and_cancellation_release_pending_p_reservation(self):
        for mode in ("error", "block"):
            with self.subTest(mode=mode):
                lb = make_lb(decode_dp=4)
                session = BlockingSession(mode)
                request = request_for_rank(lb, 0)
                with mock.patch.object(
                    mini_lb.aiohttp, "ClientSession", return_value=session
                ):
                    task = asyncio.create_task(
                        lb.generate(
                            request,
                            "http://prefill:31194",
                            "http://decode:32195",
                            "generate",
                        )
                    )
                    await asyncio.wait_for(session.all_started.wait(), timeout=1)
                    self.assertEqual(sum(lb._decode_inflight_reservations), 1)
                    if mode == "block":
                        task.cancel()
                        with self.assertRaises(asyncio.CancelledError):
                            await task
                    else:
                        with self.assertRaisesRegex(RuntimeError, "synthetic upstream"):
                            await task
                self.assertEqual(lb._decode_inflight_reservations, [0, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
