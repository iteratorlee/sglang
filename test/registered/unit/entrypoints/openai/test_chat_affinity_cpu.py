"""Dependency-free CPU guards: python -m unittest discover -s <this dir> -p test_chat_affinity_cpu.py."""

import asyncio
import importlib.util
import threading
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[5]
_PATH = _ROOT / "python/sglang/srt/entrypoints/openai/chat_affinity.py"
_spec = importlib.util.spec_from_file_location("chat_affinity_cpu", _PATH)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


class TestChatAffinityCPU(unittest.IsolatedAsyncioTestCase):
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
