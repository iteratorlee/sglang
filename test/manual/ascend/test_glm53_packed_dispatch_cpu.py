"""CPU transport/scope checks for HEAD or the exact role-source packed overlay.

Set SGLANG_GLM53_PACKED_COLLECTIVES_SOURCE and SGLANG_GLM53_PACKED_HELPER_SOURCE
to check a prepared overlay. No SGLang/Triton/NPU imports or device execution.
The dispatch AST is stopped before routing; full FP32 combine is untouched.
"""

import ast
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest import mock

from test_glm53_packed_gather_cpu import HELPER, payload, raw_bytes, torch

SOURCE = Path(os.environ.get("SGLANG_GLM53_PACKED_COLLECTIVES_SOURCE", HELPER.with_name("glm53_collectives.py")))


class PackedDispatchCpuTests(unittest.TestCase):
    def setUp(self):
        name = "_packed_dispatch_cpu.glm53_packed_gather"
        spec = importlib.util.spec_from_file_location(name, HELPER)
        self.helper = importlib.util.module_from_spec(spec)
        self.modules = mock.patch.dict(sys.modules, {name: self.helper})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        spec.loader.exec_module(self.helper)
        nodes = []
        for node in ast.parse(SOURCE.read_text()).body:
            if isinstance(node, ast.FunctionDef) and node.name == "normal_hccl_idle_dp_scope":
                nodes.append(node)
            if isinstance(node, ast.FunctionDef) and node.name == "collective_dispatch":
                stop = next(i for i, n in enumerate(node.body) if isinstance(n, ast.Assign)
                            and any(isinstance(t, ast.Name) and t.id == "flat" for t in n.targets))
                node.body = node.body[:stop] + [ast.Return(ast.Tuple(
                    [ast.Name(n, ast.Load()) for n in ("all_q", "all_scale", "all_ids", "all_weights")], ast.Load()))]
                nodes.append(node)
        self.assertEqual(len(nodes), 2)
        self.ns = dict(__name__="_packed_dispatch_cpu.dispatch", __package__="_packed_dispatch_cpu",
                       os=os, torch=torch, contextmanager=contextmanager, nullcontext=nullcontext,
                       _NORMAL_PACKED_GATHER=True, _NORMAL_STATIC_GPU_ROUTE_CAPABLE=False,
                       _NORMAL_STATIC_GPU_ROUTE_ENABLED=False,
                       _packed_target_metadata=ContextVar("packed", default=None),
                       _idle_dp_counts=ContextVar("idle", default=None),
                       _active_batch_shape_epoch=lambda: None,
                       _normal_ordered_combine_enabled=lambda *a: True,
                       release_normal_hccl_cached_memory=lambda *a: None,
                       _check_equal_dispatch_shapes_cached=lambda *a: None,
                       normal_hccl_skip_idle_dp_enabled=lambda: True,
                       _normal_dp_shape_metadata=lambda b: (b.original, b.padded),
                       _normal_hccl_layer_shape_scope=lambda *a: nullcontext())
        exec(compile(ast.fix_missing_locations(ast.Module(nodes, [])), str(SOURCE), "exec"), self.ns)

    def test_dispatch_uses_one_packed_or_four_original_collectives(self):
        for world in (16, 32):
            dp = world // 8
            cases = [(False, ((256,) + (0,) * (dp - 1), (256,) * dp), 32, False),
                     (True, ((256,) + (0,) * (dp - 1), (256,) * dp), 32, True),
                     (True, ((4096,) * dp, (4096,) * dp), 512, True),
                     (True, ((4104,) * dp, (4104,) * dp), 513, False),
                     (True, ((0,) * dp, (8,) * dp), 1, False),
                     (True, None, 32, False)]
            for enabled, metadata, rows, expected_packed in cases:
                with self.subTest(world=world, enabled=enabled, metadata=metadata):
                    peers = [payload(rows, r, torch.int64) for r in range(world)]
                    expected = tuple(torch.cat([p[i] for p in peers]) for i in range(4))
                    calls = []

                    def gather(out, value, *, group):
                        i = len(calls)
                        self.assertTrue(torch.equal(raw_bytes(value), raw_bytes(peers[0][i])))
                        out.copy_(expected[i])
                        calls.append(i)

                    self.ns.update(_NORMAL_PACKED_GATHER=enabled,
                                   dist=NS(get_world_size=lambda g: world, get_rank=lambda g: 0,
                                           all_gather_into_tensor=gather))
                    self.ns["_packed_target_metadata"].set(metadata)
                    self.ns["_idle_dp_counts"].set(metadata[0] if metadata else None)
                    with mock.patch.object(torch.ops.npu, "npu_dynamic_quant", return_value=peers[0][:2], create=True), \
                         mock.patch.object(self.helper, "packed_all_gather", return_value=expected) as packed:
                        actual = self.ns["collective_dispatch"](object(), torch.zeros((rows, 4096), dtype=torch.bfloat16),
                                                               peers[0][2], peers[0][3])
                    self.assertEqual(packed.call_count, int(expected_packed))
                    self.assertEqual(len(calls), 0 if expected_packed else 4)
                    if expected_packed:
                        self.assertEqual(packed.call_args.kwargs["expected_rows"], rows)
                    for a, e in zip(actual, expected):
                        self.assertTrue(torch.equal(raw_bytes(a), raw_bytes(e)))

    def test_target_scope_does_not_leak_into_nested_draft_or_after_exception(self):
        module = ModuleType("sglang.srt.model_executor.forward_batch_info")
        module.ForwardMode = NS(EXTEND="extend")
        batch = NS(forward_mode="extend", can_run_tbo=False, original=(256, 0, 0, 0), padded=(256,) * 4)
        metadata = self.ns["_packed_target_metadata"]
        scope = self.ns["normal_hccl_idle_dp_scope"]
        with mock.patch.dict(sys.modules, {module.__name__: module}):
            with self.assertRaisesRegex(RuntimeError, "test cleanup"):
                with scope(batch, layer_id=0):
                    self.assertEqual(metadata.get(), (batch.original, batch.padded))
                    with scope(batch):
                        self.assertIsNone(metadata.get())
                    self.assertEqual(metadata.get(), (batch.original, batch.padded))
                    raise RuntimeError("test cleanup")
            self.assertIsNone(metadata.get())
            self.assertIsNone(self.ns["_idle_dp_counts"].get())
            batch.can_run_tbo = True
            with self.assertRaisesRegex(RuntimeError, "non-overlap"):
                with scope(batch, layer_id=0):
                    self.fail("TBO must not enter packed scope")


if __name__ == "__main__":
    torch.set_num_interop_threads(1)
    unittest.main(verbosity=2)
