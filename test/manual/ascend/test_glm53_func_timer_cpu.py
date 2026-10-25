"""CPU regression checks for repeated serving/auxiliary HTTP metric startup.

Run directly with Python; use real Prometheus collectors and no model imports.
"""

import asyncio
import importlib.util
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from prometheus_client import CollectorRegistry, Histogram

ROOT = Path(__file__).resolve().parents[3]
OBSERVABILITY = ROOT / "python/sglang/srt/observability"


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_timer():
    # Avoid package __init__ importing inference dependencies. Both actual
    # source files are loaded unchanged; no timer/metric implementation is mocked.
    name = "sglang.srt.observability.utils"
    utils = load_source(name, OBSERVABILITY / "utils.py")
    with patch.dict(sys.modules, {name: utils}):
        return load_source("_glm53_timer_cpu", OBSERVABILITY / "func_timer.py")


class FuncTimerInitializationTest(unittest.TestCase):
    def setUp(self):
        self.timer = load_timer()
        self.registry = CollectorRegistry()

    def histogram(self, *args, **kwargs):
        return Histogram(*args, **kwargs, registry=self.registry)

    def count(self, name):
        return self.registry.get_sample_value(
            "sglang:func_latency_seconds_count", {"name": name}
        )

    def test_repeated_startup_preserves_collector_and_observations(self):
        with patch("prometheus_client.Histogram", self.histogram):
            self.timer.enable_func_timer()
            metric = self.timer.FUNC_LATENCY
            metric.labels(name="startup").observe(0.25)
            self.timer.enable_func_timer()
            self.assertIs(metric, self.timer.FUNC_LATENCY)
            self.assertTrue(self.timer.enable_metrics)
            self.assertEqual(self.count("startup"), 1)
            self.assertEqual(
                self.registry.get_sample_value(
                    "sglang:func_latency_seconds_sum", {"name": "startup"}
                ),
                0.25,
            )

    def test_concurrent_startup_registers_once(self):
        calls, errors = [], []
        start = threading.Barrier(12)

        def delayed_histogram(*args, **kwargs):
            calls.append(True)
            time.sleep(0.01)
            return self.histogram(*args, **kwargs)

        def enable():
            try:
                start.wait(timeout=3)
                self.timer.enable_func_timer()
            except Exception as exc:
                errors.append(exc)

        with patch("prometheus_client.Histogram", delayed_histogram):
            threads = [threading.Thread(target=enable) for _ in range(12)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=3)
            self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        self.assertEqual(len(calls), 1)
        self.assertTrue(self.timer.enable_metrics)

    def test_constructor_failure_does_not_publish_invalid_timer(self):
        with patch("prometheus_client.Histogram", side_effect=RuntimeError("setup")):
            with self.assertRaisesRegex(RuntimeError, "setup"):
                self.timer.enable_func_timer()
        self.assertFalse(self.timer.enable_metrics)
        self.assertIsNone(self.timer.FUNC_LATENCY)
        with patch("prometheus_client.Histogram", self.histogram):
            self.timer.enable_func_timer()
        self.assertTrue(self.timer.enable_metrics)
        self.assertIsNotNone(self.timer.FUNC_LATENCY)

    def test_sync_async_timing_and_results_survive_reenable(self):
        @self.timer.time_func_latency(name="sync")
        def sync(value):
            return value + 1

        @self.timer.time_func_latency(name="async")
        async def async_fn(value):
            await asyncio.sleep(0)
            return value + 2

        self.assertEqual(sync(3), 4)
        self.assertIsNone(self.count("sync"))
        with patch("prometheus_client.Histogram", self.histogram):
            self.timer.enable_func_timer()
            self.assertEqual(sync(3), 4)
            self.timer.enable_func_timer()
            self.assertEqual(asyncio.run(async_fn(3)), 5)
        self.assertEqual(self.count("sync"), 1)
        self.assertEqual(self.count("async"), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
