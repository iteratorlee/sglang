"""CPU contracts for the actual GLM53 packed arena, without framework imports.

Only base initialization and the allocator are substituted. The GLM53 class,
helpers, and base destroy method come from AST-extracted production source.
CPU allocations exercise byte layout/lifetime, not physical pinning or SDMA.

Run with TORCH_DEVICE_BACKEND_AUTOLOAD=0 CUDA_VISIBLE_DEVICES='' on CPUs 30,31.
"""

import ast
import ctypes
import gc
import hashlib
import json
import logging
import math
import os
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock
import weakref

# Set before importing torch: never auto-load an accelerator extension.
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "python/sglang/srt/mem_cache/pool_host/glm53.py"
BASE_SOURCE = SOURCE.with_name("base.py")
PACKED_ENV = "SGLANG_GLM53_HICACHE_PACKED_HOST_ALLOC"
TRIM_ENV = "SGLANG_GLM53_HICACHE_TRIM_HOST_CACHE"
FRAMEWORK_MODULES = (
    "sglang.srt.model_executor.cuda_graph_config",
    "sglang.srt.runtime_context",
)


def byte_view(tensor):
    return tensor.view(torch.uint8).reshape(-1)


class CPUAllocator:
    """Record allocator requests without retaining ownership of any tensor."""

    def __init__(self):
        self.calls = []

    def __call__(self, shape, *, dtype, device, pin_memory, allocator):
        if str(device) != "cpu":
            raise AssertionError("test must allocate exclusively on CPU")
        self.calls.append((tuple(shape), dtype, device, pin_memory, allocator))
        return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=False)


def extracted_core(allocator):
    class DummyHostKVCache:
        def __init__(
            self,
            device_pool,
            host_to_device_ratio,
            host_size,
            page_size,
            layout,
            pin_memory,
            device,
            allocator_type,
            *,
            pool_label,
        ):
            # Fixed small capacity avoids framework sizing/distributed imports.
            self.device_pool = device_pool
            self.page_size = page_size
            self.page_num = 5
            self.size = self.page_num * self.page_size
            self.layout = layout
            self.pin_memory = pin_memory
            self.device = device
            self.allocator = object()
            self.size_per_token = self.get_size_per_token()
            self.kv_buffer = self.init_kv_buffer()

    namespace = {
        "HostKVCache": DummyHostKVCache,
        "ALLOC_MEMORY_FUNCS": {"npu": allocator, "npu:0": allocator},
        "torch": torch,
        "os": os,
        "json": json,
        "hashlib": hashlib,
        "logger": logging.getLogger(__name__),
        "_npu_host_cache_trimmed": False,
        "_early_host_arena": None,
        "math": math,
        "contextmanager": contextmanager,
        "wraps": wraps,
        "_is_cuda": False,
        "_is_hip": False,
    }
    base = next(
        n
        for n in ast.parse(BASE_SOURCE.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "HostKVCache"
    )
    destroy = next(
        n for n in base.body if isinstance(n, ast.FunctionDef) and n.name == "destroy"
    )
    exec(
        compile(ast.Module(body=[destroy], type_ignores=[]), str(BASE_SOURCE), "exec"),
        namespace,
    )
    DummyHostKVCache.destroy = namespace.pop("destroy")
    tree = ast.parse(SOURCE.read_text())
    names = {
        "_trim_npu_host_cache_before_hicache",
        "is_glm53_kpool",
        "_packed_host_allocation_enabled",
        "GLM53MLAPoolHost",
        "_EarlyHostArena",
        "_reserve_early_host_arena",
        "glm53_early_host_reservation",
    }
    nodes = [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names
    ]
    if {n.name for n in nodes} != names:
        raise AssertionError("required production class/helpers are missing")
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace
    )
    return namespace


def make_pool(layers=3, index_layers=(0, 2), quantized=True):
    dtype, width = (torch.int8, 656) if quantized else (torch.bfloat16, 512)
    return SimpleNamespace(
        device="npu",
        size=512,
        page_size=64,
        layer_num=layers,
        start_layer=0,
        end_layer=layers - 1,
        store_dtype=dtype,
        kpool_use_compress=True,
        index_kpool=4,
        zero_rope=not quantized,
        dsa_kv_cache_store_int8=quantized,
        qk_rope_head_dim=0,
        kv_cache_dim=width,
        indexer_layer_ids=index_layers,
        layer_shard_enabled=False,
        k_buffer=torch.empty((layers, 9, 64, 1, width), dtype=dtype, device="cpu"),
        index_k_buffer=torch.empty(
            (len(index_layers), 9, 64, 1, 128), dtype=torch.bfloat16, device="cpu"
        ),
    )


def fill_bytes(host, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for component in host._components:
        raw = byte_view(component)
        raw.copy_(
            torch.randint(
                0,
                256,
                (raw.numel(),),
                dtype=torch.uint8,
                device="cpu",
                generator=generator,
            )
        )


class TestGLM53PackedHostArenaCPU(unittest.TestCase):
    def setUp(self):
        self.allocator = CPUAllocator()
        self.core = extracted_core(self.allocator)
        self.args = SimpleNamespace(disaggregation_mode="prefill", graphs_disabled=True)
        graph = ModuleType(FRAMEWORK_MODULES[0])
        graph.cuda_graph_fully_disabled = lambda: self.args.graphs_disabled
        runtime = ModuleType(FRAMEWORK_MODULES[1])
        runtime.get_disagg = lambda: self.args
        self.modules = {
            graph.__name__: graph,
            runtime.__name__: runtime,
            "torch_npu": None,
        }
        self.env_patch = mock.patch.dict(
            os.environ,
            {
                PACKED_ENV: "1",
                TRIM_ENV: "0",
                "SGLANG_GLM53_HICACHE_EARLY_HOST_ALLOC": "0",
            },
        )
        self.module_patch = mock.patch.dict(sys.modules, self.modules)
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)

    def make_host(self, pools, *, packed=True, pin_memory=True):
        with mock.patch.dict(os.environ, {PACKED_ENV: "1" if packed else "0"}):
            return self.core["GLM53MLAPoolHost"](
                pools[0],
                1.5,
                0,
                256,
                "page_first_kv_split",
                pin_memory=pin_memory,
                device="cpu",
                mtp_draft_device_pools=pools[1:],
            )

    def make_pair(self, *, mixed_draft=False):
        pools = (make_pool(), make_pool(1, (0,), quantized=not mixed_draft))
        baseline = self.make_host(pools, packed=False)
        self.assertEqual(len(self.allocator.calls), 4)
        self.allocator.calls.clear()
        packed = self.make_host(pools)
        self.assertEqual(len(self.allocator.calls), 1)
        return pools, baseline, packed

    def assert_bytes_equal(self, left, right):
        # Raw comparison deliberately includes BF16 NaN payloads and signed int8.
        self.assertTrue(torch.equal(byte_view(left), byte_view(right)))

    def test_allocation_count_typed_layout_seeded_bytes_and_namespace(self):
        for mixed in (False, True):
            with self.subTest(mixed_draft=mixed):
                self.allocator.calls.clear()
                pools, baseline, packed = self.make_pair(mixed_draft=mixed)
                self.assertIsNone(baseline._packed_host_arena)
                arena = packed._packed_host_arena
                total = sum(b.numel() * b.element_size() for b in baseline._components)
                self.assertEqual(arena.dtype, torch.uint8)
                self.assertEqual(arena.shape, (total,))
                self.assertEqual(
                    self.allocator.calls[0][:4], ((total,), torch.uint8, "cpu", True)
                )
                self.assertEqual(packed.page_bytes, baseline.page_bytes)
                self.assertEqual(total, packed.page_num * packed.page_bytes)
                self.assertEqual(packed.size_per_token, baseline.size_per_token)
                self.assertEqual(
                    packed.get_storage_namespace(), baseline.get_storage_namespace()
                )
                self.assertTrue(
                    packed.get_storage_namespace().startswith("glm53_c8_v1_")
                )
                fill_bytes(baseline, 617)
                fill_bytes(packed, 617)
                offset = 0
                for old, new in zip(baseline._components, packed._components):
                    self.assertEqual(
                        (new.shape, new.dtype, new.stride()),
                        (old.shape, old.dtype, old.stride()),
                    )
                    self.assertTrue(old.is_contiguous() and new.is_contiguous())
                    self.assertEqual(new.device.type, "cpu")
                    self.assertEqual(new.untyped_storage().data_ptr(), arena.data_ptr())
                    self.assertEqual(new.data_ptr(), arena.data_ptr() + offset)
                    self.assertEqual(new.data_ptr() % new.element_size(), 0)
                    self.assertEqual(new.storage_offset() * new.element_size(), offset)
                    self.assert_bytes_equal(old, new)
                    offset += new.numel() * new.element_size()
                self.assertEqual(offset, total)
                for page in (4, 1, 3, 0, 2):
                    before = baseline.get_data_page(page * 256)
                    after = packed.get_data_page(page * 256)
                    self.assertEqual(after.dtype, torch.uint8)
                    self.assertEqual(after.numel(), packed.page_bytes)
                    self.assertEqual(after.numpy().tobytes(), before.numpy().tobytes())
                    self.assert_bytes_equal(
                        after, packed.get_data_page(page * 256, flat=False)
                    )
                for pool in pools:
                    self.assertEqual(pool.k_buffer.device.type, "cpu")
                    self.assertEqual(pool.index_k_buffer.device.type, "cpu")

    def test_arbitrary_logical_pages_cross_restore_both_directions(self):
        _, baseline, packed = self.make_pair()
        sources, destinations = (4, 1, 3, 0, 2), (2, 4, 0, 3, 1)
        for producer, consumer in ((baseline, packed), (packed, baseline)):
            fill_bytes(producer, 811)
            originals = [byte_view(b).clone() for b in producer._components]
            for b in consumer._components:
                byte_view(b).fill_(0xA5)
            for src, dst in zip(sources, destinations):
                page = producer.get_data_page(src * 256)
                # Model SSD flat bytes through a distinct owned CPU byte buffer.
                payload = bytearray(page.numpy().tobytes())
                consumer.set_from_flat_data_page(
                    dst * 256, torch.frombuffer(payload, dtype=torch.uint8)
                )
                self.assert_bytes_equal(consumer.get_data_page(dst * 256), page)
            for b in producer._components:
                b.zero_()
            for src, dst in zip(sources, destinations):
                producer.set_from_flat_data_page(
                    src * 256, consumer.get_data_page(dst * 256)
                )
            for restored, original in zip(producer._components, originals):
                self.assert_bytes_equal(restored, original)

    def test_page_metadata_points_to_exact_nonoverlapping_page_bytes(self):
        _, baseline, packed = self.make_pair()
        pages = (4, 1, 3, 0)
        indices = torch.cat(
            [torch.arange(p * 256, (p + 1) * 256, device="cpu") for p in pages]
        )
        for host in (baseline, packed):
            fill_bytes(host, 149)
            pointers, sizes = host.get_page_buffer_meta(indices)
            self.assertEqual(len(pointers), len(pages) * 4)
            ranges = []
            for i, page in enumerate(pages):
                serialized = bytearray()
                for j, buf in enumerate(host._components):
                    ptr, size = pointers[i * 4 + j], sizes[i * 4 + j]
                    expected = byte_view(buf[page])
                    self.assertEqual(ptr, buf[page].data_ptr())
                    self.assertEqual(size, expected.numel())
                    payload = ctypes.string_at(ptr, size)
                    self.assertEqual(payload, expected.numpy().tobytes())
                    serialized.extend(payload)
                    ranges.append((ptr, ptr + size))
                self.assertEqual(
                    bytes(serialized), host.get_data_page(page * 256).numpy().tobytes()
                )
            ranges.sort()
            for left, right in zip(ranges, ranges[1:]):
                self.assertLessEqual(left[1], right[0])

    def test_mtp_physical_views_alias_components_without_overlap(self):
        pools, baseline, packed = self.make_pair(mixed_draft=True)
        for host in (baseline, packed):
            fill_bytes(host, 199)
            intervals = []
            for pool_index, pool in enumerate(pools):
                for component_index, physical in enumerate(
                    host._pool_buffers[id(pool)]
                ):
                    logical = host._components[pool_index * 2 + component_index]
                    reference = baseline._pool_buffers[id(pool)][component_index]
                    self.assertEqual(
                        (physical.shape, physical.dtype, physical.stride()),
                        (reference.shape, reference.dtype, reference.stride()),
                    )
                    self.assertTrue(physical.is_contiguous())
                    self.assertEqual(physical.shape[0], host.page_num * 4)
                    self.assertEqual(physical.data_ptr(), logical.data_ptr())
                    intervals.append(
                        (physical.data_ptr(), physical.data_ptr() + physical.nbytes)
                    )
                    for page in (4, 0, 2):
                        for subpage in range(4):
                            self.assertEqual(
                                physical[page * 4 + subpage].data_ptr(),
                                logical[page, subpage].data_ptr(),
                            )
                            self.assert_bytes_equal(
                                physical[page * 4 + subpage], logical[page, subpage]
                            )
                    byte_view(physical[7]).fill_(0x5C)
                    self.assertTrue(torch.all(byte_view(logical[1, 3]) == 0x5C).item())
            intervals.sort()
            self.assertTrue(all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:])))

    def test_destroy_preserves_external_target_and_mtp_view_storage(self):
        pools = (make_pool(), make_pool(1, (0,)))
        host = self.make_host(pools)
        fill_bytes(host, 407)
        views = (
            host._components[0][4],
            host._pool_buffers[id(pools[1])][0][7],
            host._pool_buffers[id(pools[1])][1][3],
        )
        expected = [byte_view(v).clone() for v in views]
        pointers = [v.data_ptr() for v in views]
        host_ref = weakref.ref(host)
        host.destroy()
        host.destroy()
        self.assertTrue(host._destroyed)
        self.assertIsNone(host.kv_buffer)
        self.assertIsNone(host._packed_host_arena)
        self.assertEqual(host._components, [])
        self.assertEqual(host._pool_buffers, {})
        del host
        gc.collect()
        self.assertIsNone(host_ref())
        for view, original, ptr in zip(views, expected, pointers):
            self.assertEqual(view.data_ptr(), ptr)
            self.assert_bytes_equal(view, original)
            byte_view(view).fill_(0xE3)
            self.assertTrue(torch.all(byte_view(view) == 0xE3).item())

    def test_framework_mocks_are_actual_source_exports(self):
        functions = [
            n
            for n in ast.parse(SOURCE.read_text()).body
            if isinstance(n, ast.FunctionDef)
            and n.name
            in (
                "_packed_host_allocation_enabled",
                "_trim_npu_host_cache_before_hicache",
            )
        ]
        checked = set()
        for function in functions:
            for node in ast.walk(function):
                if not isinstance(node, ast.ImportFrom) or not node.module.startswith(
                    "sglang."
                ):
                    continue
                path = ROOT / "python" / (node.module.replace(".", "/") + ".py")
                exports = {
                    n.name
                    for n in ast.parse(path.read_text()).body
                    if isinstance(
                        n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                    )
                }
                for item in node.names:
                    self.assertIn(item.name, exports, str(path))
                    checked.add((node.module, item.name))
        self.assertEqual(
            checked,
            {
                (FRAMEWORK_MODULES[0], "cuda_graph_fully_disabled"),
                (FRAMEWORK_MODULES[1], "get_disagg"),
            },
        )

    def test_default_off_non_npu_non_c8_and_no_pin_short_circuit(self):
        enabled = self.core["_packed_host_allocation_enabled"]
        with mock.patch.dict(sys.modules, {name: None for name in FRAMEWORK_MODULES}):
            os.environ.pop(PACKED_ENV, None)
            self.assertFalse(enabled("npu", True, True))
            for value in ("0", "", "true", "01"):
                os.environ[PACKED_ENV] = value
                self.assertFalse(enabled("npu", True, True))
            os.environ[PACKED_ENV] = "1"
            for device in ("cpu", "cuda:0", "xpu:0"):
                self.assertFalse(enabled(device, True, True))
            self.assertFalse(enabled("npu", False, True))
            self.assertFalse(enabled("npu", True, False))
        self.assertTrue(enabled("npu:0", True, True))

    def test_guards_retain_four_allocations(self):
        cases = (
            ("default_off", False, True, True, "prefill", True),
            ("no_pin", True, False, True, "prefill", True),
            ("non_c8", True, True, False, "prefill", True),
            ("decode", True, True, True, "decode", True),
            ("non_pd", True, True, True, None, True),
            ("graphs", True, True, True, "prefill", False),
        )
        for label, packed, pinned, quantized, mode, graphs in cases:
            with self.subTest(guard=label):
                self.allocator.calls.clear()
                self.args.disaggregation_mode, self.args.graphs_disabled = mode, graphs
                pools = (
                    make_pool(quantized=quantized),
                    make_pool(1, (0,), quantized=quantized),
                )
                host = self.make_host(pools, packed=packed, pin_memory=pinned)
                self.assertEqual(len(self.allocator.calls), 4)
                self.assertIsNone(host._packed_host_arena)
                self.assertEqual(
                    [c[1] for c in self.allocator.calls],
                    [b.dtype for p in pools for b in (p.k_buffer, p.index_k_buffer)],
                )
                if not quantized:
                    self.assertEqual(host.get_storage_namespace(), "")

    def test_existing_page_validation_and_dummy_flat_buffer(self):
        _, baseline, packed = self.make_pair()
        for host in (baseline, packed):
            dummy = host.get_dummy_flat_data_page()
            self.assertEqual(
                (dummy.dtype, dummy.device.type, dummy.numel()),
                (torch.uint8, "cpu", host.page_bytes),
            )
            for index in (-256, 64, host.size, host.size + 256):
                with self.assertRaisesRegex(ValueError, "unaligned/out-of-range"):
                    host.get_data_page(index)
                with self.assertRaisesRegex(ValueError, "unaligned/out-of-range"):
                    host.set_from_flat_data_page(index, dummy)
            with self.assertRaisesRegex(ValueError, "incompatible size"):
                host.set_from_flat_data_page(0, dummy[:-1])
            with self.assertRaisesRegex(ValueError, "incomplete"):
                host.get_page_buffer_meta(torch.arange(64, device="cpu"))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
