"""Dependency-free CPU guards: python -m unittest discover -s <this dir> -p test_chat_affinity_cpu.py."""

import asyncio
import importlib.util
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parents[5]
_PATH = _ROOT / "python/sglang/srt/entrypoints/openai/chat_affinity.py"
_spec = importlib.util.spec_from_file_location("chat_affinity_cpu", _PATH)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


class TestChatAffinityCPU(unittest.IsolatedAsyncioTestCase):
    async def test_parallel_budget_and_cancelled_jobs_retain_their_slots(self):
        with patch.dict("os.environ", {"SGLANG_CHAT_AFFINITY_CPU_WORKERS": "2"}):
            service = _module.ChatAffinityTokenizer(None)
        started = [threading.Event(), threading.Event()]
        release = threading.Event()

        def work(index):
            started[index].set()
            release.wait(5)
            return {"supported": True}

        service._tokenize = work
        tasks = [asyncio.create_task(service.tokenize(i)) for i in range(2)]
        try:
            for _ in range(1000):
                if all(event.is_set() for event in started):
                    break
                await asyncio.sleep(0.001)
            self.assertTrue(all(event.is_set() for event in started))
            self.assertEqual(
                await service.tokenize(0), {"supported": False, "reason": "busy"}
            )
            tasks[0].cancel()
            with self.assertRaises(asyncio.CancelledError):
                await tasks[0]
            self.assertEqual((await service.tokenize(0))["reason"], "busy")
            release.set()
            for _ in range(1000):
                if tasks[1].done() and service._slot._value == 2:
                    break
                await asyncio.sleep(0.001)
            self.assertTrue(tasks[1].done())
            await tasks[1]
            self.assertEqual(service._slot._value, 2)
            next_task = asyncio.create_task(service.tokenize(0))
            for _ in range(1000):
                if next_task.done():
                    break
                await asyncio.sleep(0.001)
            self.assertTrue(next_task.done())
            self.assertTrue((await next_task)["supported"])
        finally:
            release.set()
            service._executor.shutdown()

    def test_cpu_worker_configuration_is_bounded_and_invalid_falls_back(self):
        for value, expected in [
            ("invalid", 1),
            ("0", 1),
            ("-2", 1),
            ("4", 4),
            ("99", 8),
        ]:
            with self.subTest(value=value), patch.dict(
                "os.environ", {"SGLANG_CHAT_AFFINITY_CPU_WORKERS": value}
            ):
                service = _module.ChatAffinityTokenizer(None)
            try:
                self.assertEqual(service._executor._max_workers, expected)
                self.assertEqual(service._slot._value, expected)
            finally:
                service._executor.shutdown()

    async def test_cancellation_keeps_cpu_slot_until_worker_finishes(self):
        service = _module.ChatAffinityTokenizer(None)
        entered, release = threading.Event(), threading.Event()

        def work(data):
            entered.set()
            release.wait(5)
            return {"supported": True}

        service._tokenize = work
        try:
            task = asyncio.create_task(service.tokenize({}))
            for _ in range(1000):
                if entered.is_set():
                    break
                await asyncio.sleep(0.001)
            self.assertTrue(entered.is_set())
            # The event loop remains responsive while the CPU worker is blocked.
            self.assertEqual(
                await service.tokenize({}), {"supported": False, "reason": "busy"}
            )
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual((await service.tokenize({}))["reason"], "busy")
            release.set()
            for _ in range(1000):
                if not service._slot.locked():
                    break
                await asyncio.sleep(0.001)
            self.assertFalse(service._slot.locked())
            self.assertTrue((await service.tokenize({}))["supported"])
        finally:
            release.set()
            service._executor.shutdown()

    def test_media_guard_does_not_confuse_tool_argument_urls_with_media(self):
        self.assertTrue(
            _module.supports_text_request(
                {
                    "messages": [
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {"function": {"arguments": '{"image_url":"text"}'}}
                            ],
                        }
                    ]
                }
            )
        )
        for part in [
            {"type": "image_url"},
            {"type": "video_url"},
            {"type": "unknown"},
            {"type": "text", "text": {}},
        ]:
            self.assertFalse(
                _module.supports_text_request({"messages": [{"content": [part]}]})
            )
        self.assertFalse(
            _module.supports_text_request(
                {"messages": [{"content": "x"}], "session_id": "s"}
            )
        )


if __name__ == "__main__":
    unittest.main()
