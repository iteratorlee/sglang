"""Offline HTTP entrance tests; backend responses are in-memory, never model calls."""

import asyncio
import copy
import json
import os
import subprocess
import sys
import unittest
from itertools import count
from unittest import mock

import aiohttp
import httpx
from sglang_router import mini_lb
from sglang_router.launch_router import parse_router_args
from sglang_router.router_args import RouterArgs


def make_args(**updates):
    values = dict(
        mini_lb=True,
        pd_disaggregation=True,
        policy="random",
        prefill_urls=[("http://prefill:31194", 8998)],
        decode_urls=["http://decode:32195"],
        mini_lb_prefix_affinity=True,
        mini_lb_prefix_affinity_length=4,
    )
    values.update(updates)
    return RouterArgs(**values)


class Response:
    def __init__(self, body, status=200):
        self.body, self.status = body, status
        self.content = self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *unused):
        return False

    def raise_for_status(self):
        if self.status != 200:
            raise aiohttp.ClientError(f"backend status {self.status}")

    async def json(self):
        # Yield so concurrent first requests exercise the discovery lock.
        await asyncio.sleep(0)
        return self.body

    async def iter_chunked(self, unused_size):
        yield b"data: " + json.dumps(self.body).encode() + b"\n\n"
        yield b"data: [DONE]\n\n"


class Backend:
    def __init__(self):
        self.gets, self.posts = [], []
        self.info = {
            "http://prefill:31194/server_info": {
                "dp_size": 1,
                "internal_states": [{}],
            },
            "http://decode:32195/server_info": {
                "dp_size": 8,
                "internal_states": [{}] * 8,
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
        self.posts.append((url, copy.deepcopy(json)))
        return Response(
            {"text": "ok", "meta_info": {"dp_rank": json.get("routed_dp_rank")}}
        )


class ConfigTests(unittest.TestCase):
    def test_flags_parse_and_default_is_off(self):
        self.assertFalse(RouterArgs().mini_lb_prefix_affinity)
        args = parse_router_args(
            [
                "--mini-lb",
                "--pd-disaggregation",
                "--mini-lb-prefix-affinity",
                "--mini-lb-prefix-affinity-length",
                "256",
                "--prefill",
                "http://prefill:31194",
                "8998",
                "--decode",
                "http://decode:32195",
            ]
        )
        args._validate_router_args()
        self.assertTrue(args.mini_lb_prefix_affinity)
        self.assertEqual(args.mini_lb_prefix_affinity_length, 256)

    def test_invalid_flag_combinations_fail_at_startup(self):
        for updates in (
            dict(mini_lb=False),
            dict(pd_disaggregation=False),
            dict(mini_lb_prefix_affinity_length=0),
            dict(mini_lb_prefix_affinity_length=-1),
            dict(prefill_urls=[]),
            dict(decode_urls=["http://d1", "http://d2"]),
            dict(test_external_dp_routing=True),
        ):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                mini_lb.MiniLoadBalancer(make_args(**updates))

    def test_digest_is_stable_across_python_hash_seeds(self):
        code = (
            "from sglang_router.mini_lb import MiniLoadBalancer; "
            "from sglang_router.router_args import RouterArgs; "
            "lb=MiniLoadBalancer(RouterArgs(mini_lb=True,pd_disaggregation=True,"
            "prefill_urls=[('http://p',8998)],decode_urls=['http://d'],"
            "policy='random',mini_lb_prefix_affinity=True)); "
            "print(lb._affinity_digest({'input_ids':[1,2,3,4]}))"
        )
        results = [
            subprocess.check_output(
                [sys.executable, "-B", "-c", code],
                env=dict(os.environ, PYTHONHASHSEED=seed),
                text=True,
            ).strip()
            for seed in ("12", "93")
        ]
        self.assertEqual(results[0], results[1])


class EntranceTests(unittest.IsolatedAsyncioTestCase):
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

    async def send(self, payload):
        response = await self.client.post("/generate", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        return response, self.backend.posts[-2:]

    def assert_pair(self, pair):
        (p_url, p), (d_url, d) = pair
        self.assertEqual(p_url, "http://prefill:31194/generate")
        self.assertEqual(d_url, "http://decode:32195/generate")
        self.assertEqual(p["routed_dp_rank"], 0)
        self.assertTrue(0 <= d["routed_dp_rank"] < 8)
        self.assertEqual(d["disagg_prefill_dp_rank"], 0)
        self.assertNotIn("disagg_prefill_dp_rank", p)
        self.assertEqual(p["bootstrap_room"], d["bootstrap_room"])
        self.assertEqual(p["bootstrap_host"], "prefill")
        self.assertEqual(d["bootstrap_port"], 8998)
        return d

    async def test_input_ids_common_prefix_and_fresh_rooms(self):
        payload = dict(input_ids=[11, 12, 13, 14, 15], rid="cold", bootstrap_room=7)
        before = copy.deepcopy(payload)
        _, pair = await self.send(payload)
        first = self.assert_pair(pair)
        _, pair = await self.send(
            dict(payload, input_ids=[11, 12, 13, 14, 99], rid="warm")
        )
        second = self.assert_pair(pair)
        self.assertEqual(first["routed_dp_rank"], second["routed_dp_rank"])
        self.assertNotEqual(first["bootstrap_room"], second["bootstrap_room"])
        self.assertNotEqual(first["bootstrap_room"], 7)
        self.assertEqual(payload, before)
        self.assertEqual(len(self.backend.gets), 2)

    async def test_text_unicode_and_streaming_use_same_affinity(self):
        response, pair = await self.send(dict(text="共同前缀冷请求", stream=True))
        first = self.assert_pair(pair)
        self.assertIn("data: [DONE]", response.text)
        response, pair = await self.send(dict(text="共同前缀热请求", stream=False))
        second = self.assert_pair(pair)
        self.assertEqual(first["routed_dp_rank"], second["routed_dp_rank"])
        self.assertEqual(
            response.json()["meta_info"]["dp_rank"], second["routed_dp_rank"]
        )

    async def test_distinct_families_reach_all_eight_decode_ranks(self):
        ranks = set()
        rooms = set()
        for family in range(128):
            _, pair = await self.send(dict(input_ids=[family, 2, 3, 4]))
            d = self.assert_pair(pair)
            ranks.add(d["routed_dp_rank"])
            rooms.add(d["bootstrap_room"])
        self.assertEqual(ranks, set(range(8)))
        self.assertEqual(len(rooms), 128)

    async def test_concurrent_first_requests_discover_once(self):
        responses = await asyncio.gather(
            *[
                self.client.post(
                    "/generate", json={"input_ids": [1, 2, 3, 4], "rid": str(i)}
                )
                for i in range(16)
            ]
        )
        self.assertTrue(all(response.status_code == 200 for response in responses))
        self.assertEqual(len(self.backend.gets), 2)
        decode = [body for url, body in self.backend.posts if "decode:" in url]
        self.assertEqual(len({body["bootstrap_room"] for body in decode}), 16)
        self.assertEqual(len({body["routed_dp_rank"] for body in decode}), 1)

    async def test_invalid_inputs_never_reach_backends(self):
        for payload in (
            {},
            {"input_ids": []},
            {"input_ids": [[1], [2]]},
            {"input_ids": [True]},
            {"input_ids": [-1]},
            {"input_ids": [2**64]},
            {"text": ["a", "b"]},
            {"text": ""},
            {"input_ids": [1], "text": "a"},
            {"input_ids": [1], "routed_dp_rank": 7},
            {"text": "a", "data_parallel_rank": 1},
            {"text": "a", "disagg_prefill_dp_rank": 0},
            {"text": "a", "sampling_params": {"n": 2}},
            {"text": "a", "sampling_params": [{"n": 1}]},
            {"text": "a", "sampling_params": {"beam_width": 2}},
            {"text": "a", "cache_salt": ["a"]},
        ):
            with self.subTest(payload=payload):
                response = await self.client.post("/generate", json=payload)
                self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(self.backend.gets, [])
        self.assertEqual(self.backend.posts, [])

    async def test_dp_discovery_failure_is_atomic_and_retryable(self):
        key = "http://decode:32195/server_info"
        self.backend.info[key] = {"dp_size": 8, "internal_states": [{}]}
        response = await self.client.post("/generate", json={"text": "prefix"})
        self.assertEqual(response.status_code, 503)
        self.assertIsNone(self.router.prefill_dp_size)
        self.assertIsNone(self.router.decode_dp_size)
        self.assertEqual(self.backend.posts, [])
        self.backend.info[key] = {"internal_states": [{}] * 8}
        _, pair = await self.send({"text": "prefix"})
        self.assert_pair(pair)

    async def test_default_keeps_legacy_bodies_and_room_generation(self):
        self.router = mini_lb.MiniLoadBalancer(make_args(mini_lb_prefix_affinity=False))
        with (
            mock.patch.object(mini_lb, "lb", self.router),
            mock.patch.object(
                mini_lb, "_generate_bootstrap_room", return_value=123
            ) as generate_room,
        ):
            for streaming in (False, True):
                _, pair = await self.send(dict(text="hello", stream=streaming))
                p, d = pair[0][1], pair[1][1]
                self.assertEqual(p, d)
                self.assertNotIn("routed_dp_rank", d)
                self.assertEqual(d["bootstrap_room"], 123)
            self.assertEqual(generate_room.call_count, 2)
        self.assertEqual(self.backend.gets, [])

    async def test_non_generate_is_explicitly_unsupported(self):
        response = await self.client.post("/v1/completions", json={"prompt": "hello"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.backend.posts, [])

    async def test_room_overflow_fails_without_reusing_an_id(self):
        self.router._affinity_rooms = count(2**63 - 1)
        _, pair = await self.send({"text": "last"})
        self.assertEqual(pair[0][1]["bootstrap_room"], 2**63 - 1)
        response = await self.client.post("/generate", json={"text": "overflow"})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(len(self.backend.posts), 2)


if __name__ == "__main__":
    unittest.main()
