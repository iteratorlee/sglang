"""Startup-only pinned allocator cleanup, with no accelerator imports."""

import ast
import logging
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock


SOURCE = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/mem_cache/pool_host/glm53.py"
)
ENV = "SGLANG_GLM53_HICACHE_TRIM_HOST_CACHE"


class TestStartupHostCache(unittest.TestCase):
    def setUp(self):
        # Extract the actual helper to test on a CPU-only host, without importing
        # SGLang's accelerator dependencies or initializing an accelerator.
        tree = ast.parse(SOURCE.read_text())
        function = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef)
            and n.name == "_trim_npu_host_cache_before_hicache"
        )
        self.namespace = {
            "os": os,
            "logger": logging.getLogger(__name__),
            "_npu_host_cache_trimmed": False,
        }
        exec(
            compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"),
            self.namespace,
        )
        self.trim = self.namespace[function.name]
        self.events = []
        self.stats = {"allocated_bytes.current": 4096, "active_bytes.current": 1024}
        self.live_tensor = [3, 1, 4, 1, 5]
        npu = SimpleNamespace(
            synchronize=lambda: self.events.append("synchronize"),
            host_memory_stats=lambda: self.stats.copy(),
            host_empty_cache=self.empty_cache,
        )
        server = ModuleType("sglang.srt.runtime_context")
        self.args = SimpleNamespace(
            disaggregation_mode="prefill", disable_cuda_graph=True
        )
        server.get_disagg = lambda: self.args
        graph = ModuleType("sglang.srt.model_executor.cuda_graph_config")
        graph.cuda_graph_fully_disabled = lambda: self.args.disable_cuda_graph
        backend = ModuleType("torch_npu")
        backend.npu = npu
        self.npu = npu
        self.modules = {
            "torch_npu": backend,
            "sglang.srt.runtime_context": server,
            "sglang.srt.model_executor.cuda_graph_config": graph,
        }

    def test_framework_imports_are_real_exports_in_this_revision(self):
        # A mocked module can conceal a nonexistent framework API. Verify every
        # local import in the actual helper against the real source definitions.
        function = next(
            n
            for n in ast.parse(SOURCE.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == self.trim.__name__
        )
        for node in ast.walk(function):
            if not isinstance(node, ast.ImportFrom) or not node.module.startswith(
                "sglang."
            ):
                continue
            target = (
                SOURCE.parents[5] / "python" / (node.module.replace(".", "/") + ".py")
            )
            exports = {
                n.name
                for n in ast.parse(target.read_text()).body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef))
            }
            for item in node.names:
                self.assertIn(item.name, exports, str(target))

    def empty_cache(self):
        self.events.append("host_empty_cache")
        self.stats["allocated_bytes.current"] = self.stats["active_bytes.current"]

    def test_only_eager_npu_prefill_opt_in_runs_once(self):
        with mock.patch.dict(os.environ, {ENV: "1"}), mock.patch.dict(
            sys.modules, self.modules
        ), mock.patch("gc.collect", side_effect=lambda: self.events.append("collect")):
            self.assertTrue(self.trim("npu:0"))
            self.assertFalse(self.trim("npu:0"))
        self.assertEqual(self.events, ["synchronize", "collect", "host_empty_cache"])
        self.assertEqual(self.stats["allocated_bytes.current"], 1024)
        self.assertEqual(self.live_tensor, [3, 1, 4, 1, 5])

    def test_default_off_and_non_npu_do_not_import_backend(self):
        with mock.patch.dict(os.environ, {ENV: "0"}), mock.patch.dict(
            sys.modules, {"torch_npu": None}
        ):
            self.assertFalse(self.trim("npu"))
        with mock.patch.dict(os.environ, {ENV: "1"}), mock.patch.dict(
            sys.modules, {"torch_npu": None}
        ):
            for device in ("cpu", "cuda:0", "xpu:0"):
                self.assertFalse(self.trim(device))

    def test_decode_and_graph_prefill_do_not_call_backend(self):
        with mock.patch.dict(os.environ, {ENV: "1"}), mock.patch.dict(
            sys.modules, self.modules
        ):
            self.args.disaggregation_mode = "decode"
            self.assertFalse(self.trim("npu"))
            self.args.disaggregation_mode = "prefill"
            self.args.disable_cuda_graph = False
            self.assertFalse(self.trim("npu"))
        self.assertEqual(self.events, [])
        self.assertFalse(self.namespace["_npu_host_cache_trimmed"])

    def test_failed_cleanup_can_retry_instead_of_marking_success(self):
        self.npu.host_empty_cache = mock.Mock(
            side_effect=RuntimeError("allocator failure")
        )
        with mock.patch.dict(os.environ, {ENV: "1"}), mock.patch.dict(
            sys.modules, self.modules
        ):
            with self.assertRaisesRegex(RuntimeError, "allocator failure"):
                self.trim("npu")
            self.assertFalse(self.namespace["_npu_host_cache_trimmed"])
            self.npu.host_empty_cache = self.empty_cache
            self.assertTrue(self.trim("npu"))


if __name__ == "__main__":
    unittest.main()
