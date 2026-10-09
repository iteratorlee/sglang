"""CPU contract tests for the NPU graph pool experiment.

Extract the real backend methods so no torch_npu, model, or NPU is required.
This does not establish graph allocator or numerical safety on hardware.
"""

import ast
import contextlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


SOURCE = (
    Path(__file__).resolve().parents[3]
    / "python/sglang/srt/hardware_backend/npu/graph_runner/npu_cudagraph_backend.py"
)


def load_backend(namespace):
    tree = ast.parse(SOURCE.read_text())
    original = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "NPUCudaGraphBackend"
    )
    methods = [
        node
        for node in original.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("capture_session", "capture_one")
    ]
    assert len(methods) == 2
    extracted = ast.ClassDef(
        name="Backend",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            extracted,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    return namespace["Backend"]


class Device:
    def __init__(self):
        self.pool_count = 0
        self.free = 100 * 2**20
        self.allocated = 0
        self.reserved = 0

    def graph_pool_handle(self):
        self.pool_count += 1
        return (0, self.pool_count)

    def synchronize(self):
        pass

    def mem_get_info(self):
        return self.free, 100 * 2**20

    def memory_allocated(self):
        return self.allocated

    def memory_reserved(self):
        return self.reserved


class Graph:
    def __init__(self):
        self._pool = None

    def pool(self):
        return self._pool


class PoolTests(unittest.TestCase):
    def setUp(self):
        self.device = Device()
        self.flags = {}
        self.pool = None
        self.pool_ids = []
        self.captures = []
        self.logs = Mock()

        def global_pool(device):
            if self.pool is None:
                self.pool = device.graph_pool_handle()
            return self.pool

        @contextlib.contextmanager
        def graph(graph, *, pool, stream, auto_dispatch_capture):
            graph._pool = pool
            self.captures.append((pool, stream, auto_dispatch_capture))
            yield
            self.device.free -= 2**20
            self.device.allocated += 2**20
            self.device.reserved += 2**20

        self.backend_type = load_backend(
            dict(
                contextmanager=contextlib.contextmanager,
                get_bool_env_var=lambda name: self.flags.get(name, False),
                get_or_create_global_graph_memory_pool=global_pool,
                set_graph_pool_id=self.pool_ids.append,
                torch=types.SimpleNamespace(
                    npu=types.SimpleNamespace(NPUGraph=Graph, graph=graph)
                ),
                empty_context=contextlib.nullcontext,
                logger=self.logs,
            )
        )

    def backend(self):
        backend = self.backend_type()
        backend._pool = None
        backend._runner_name = "FakeTargetRunner"
        backend._capture_stream = None
        backend._device_module = self.device
        backend._tp_group = types.SimpleNamespace(barrier=lambda: None)
        backend._enable_torch_compile = False
        backend._memory_saver_adapter = None
        backend._graphs = {}
        backend._outputs = {}
        return backend

    def capture(self, backend, shape, stream):
        calls = []

        def forward():
            calls.append(shape)
            return (shape, len(calls))

        with backend.capture_session(stream):
            with patch.dict(sys.modules, {"torch_npu": types.ModuleType("torch_npu")}):
                backend.capture_one(shape, forward)
        self.assertEqual(len(calls), 3)
        self.assertEqual(backend._outputs[shape], (shape, 3))
        self.assertIsNone(backend._capture_stream)

    def test_default_keeps_distinct_runner_pools(self):
        target, draft = self.backend(), self.backend()
        stream = object()
        self.capture(target, "verify-bs2", stream)
        self.capture(target, "verify-bs1", stream)
        self.capture(draft, "draft-bs2", stream)
        self.capture(draft, "draft-bs1", stream)
        self.assertEqual(self.device.pool_count, 2)
        self.assertNotEqual(target._pool, draft._pool)
        self.assertEqual(target._graphs["verify-bs1"].pool(), target._pool)
        self.assertEqual(target._graphs["verify-bs2"].pool(), target._pool)
        self.assertEqual(draft._graphs["draft-bs1"].pool(), draft._pool)
        self.assertEqual([entry[0] for entry in self.captures], self.pool_ids)
        self.logs.info.assert_not_called()

    def test_trace_reports_one_shape_without_changing_pool_selection(self):
        self.flags["SGLANG_NPU_GRAPH_MEMORY_TRACE"] = True
        target = self.backend()
        stream = object()
        self.capture(target, "verify-bs2", stream)
        self.assertEqual(self.device.pool_count, 1)
        args = self.logs.info.call_args.args
        self.assertIn("NPU graph capture", args[0])
        self.assertEqual(
            args[1:6],
            ("FakeTargetRunner", "verify-bs2", target._pool, target._pool, stream),
        )
        self.assertEqual(args[6:], (-1.0, 1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
