"""CPU-only contracts for early reservation and the actual ModelRunner hook.

The allocator records pinning requests but returns CPU storage. This verifies
ownership/bytes/order, not physical NPU pinning or full-model startup success.
"""

import ast
import gc
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock
import weakref

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

from test_glm53_packed_host_arena_cpu import (
    CPUAllocator,
    PACKED_ENV,
    ROOT,
    SOURCE,
    TRIM_ENV,
    byte_view,
    extracted_core,
    fill_bytes,
    make_pool,
)

EARLY_ENV = "SGLANG_GLM53_HICACHE_EARLY_HOST_ALLOC"
RUNNER = ROOT / "python/sglang/srt/model_executor/model_runner.py"


class TestGLM53EarlyHostArenaCPU(unittest.TestCase):
    def setUp(self):
        self.allocator = CPUAllocator()
        self.core = extracted_core(self.allocator)
        self.scope = self.core["glm53_early_host_reservation"]
        self.memory = SimpleNamespace(
            enable_hierarchical_cache=True,
            hicache_mem_layout="page_first_kv_split",
            hicache_io_backend="kernel_ascend",
            hicache_storage_backend="file",
            disable_radix_cache=False,
            hicache_size=0.02,  # Actual CPU tensor: 20 MB, never a 64 GB allocation.
        )
        self.model_args = SimpleNamespace(kv_cache_dtype="int8")
        self.disagg = SimpleNamespace(disaggregation_mode="prefill")
        self.parallel = SimpleNamespace(nnodes=4, dcp_enabled=False)
        self.schedule = SimpleNamespace(page_size=64)
        self.ssm_dtype = torch.float32
        self.graphs_disabled = True
        self.quant = {
            "model.layers.0.mlp.weight": "W8A8_DYNAMIC",
            "norm.weight": "FLOAT",
        }
        self.runner = SimpleNamespace(
            device="npu",
            gpu_id=3,
            is_draft_worker=False,
            ps=SimpleNamespace(tp_size=32, attn_dp_size=4, moe_ep_size=32, pp_size=1),
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    architectures=["Glm5NextForConditionalGeneration"]
                ),
                quantization="modelslim",
                _find_quant_modelslim_config=lambda: self.quant,
            ),
            forward=mock.Mock(return_value="forward-result"),
        )
        runtime = ModuleType("sglang.srt.runtime_context")
        for name, value in (
            ("memory", self.memory),
            ("model", self.model_args),
            ("disagg", self.disagg),
            ("parallel", self.parallel),
            ("schedule", self.schedule),
        ):
            setattr(runtime, "get_" + name, lambda value=value: value)
        graph = ModuleType("sglang.srt.model_executor.cuda_graph_config")
        graph.cuda_graph_fully_disabled = lambda: self.graphs_disabled
        mamba = ModuleType("sglang.srt.configs.mamba_utils")
        mamba.mamba2_state_dtype = lambda config: SimpleNamespace(
            temporal=self.ssm_dtype
        )
        self.modules = {x.__name__: x for x in (runtime, graph, mamba)}
        self.modules["torch_npu"] = None
        self.device = SimpleNamespace(current_device=lambda: self.runner.gpu_id)
        patches = (
            mock.patch.dict(
                os.environ, {EARLY_ENV: "1", PACKED_ENV: "1", TRIM_ENV: "0"}
            ),
            mock.patch.dict(sys.modules, self.modules),
            mock.patch.object(torch, "get_device_module", return_value=self.device),
        )
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def make_host(self, *, pools=None, **kw):
        pools = pools or (make_pool(), make_pool(1, (0,)))
        return self.core["GLM53MLAPoolHost"](
            pools[0],
            1.5,
            0,
            256,
            "page_first_kv_split",
            mtp_draft_device_pools=pools[1:],
            **kw,
        )

    def test_reserve_before_weights_then_reuse_exact_views_and_release(self):
        events = []
        original_forward = self.runner.forward
        with self.scope(self.runner):
            self.assertEqual(len(self.allocator.calls), 1)
            events.append("reserve")
            record = self.core["_early_host_arena"]
            self.assertFalse(record.consumed)
            ptr = record.arena.data_ptr()
            arena_ref = weakref.ref(record.arena)
            events.append("weights")
        with self.scope(SimpleNamespace(device="npu", is_draft_worker=True)):
            events.append("draft-weights")
        self.assertEqual(len(self.allocator.calls), 1)
        host = self.make_host()
        self.assertEqual(len(self.allocator.calls), 1)
        self.assertIsNone(self.core["_early_host_arena"])
        self.assertIsNone(record.arena)
        self.assertTrue(record.consumed)
        self.assertEqual(host._packed_host_arena.data_ptr(), ptr)
        self.assertEqual(host._packed_host_arena.untyped_storage().nbytes(), 20_000_000)
        self.assertIsNotNone(arena_ref())
        self.assertEqual(self.runner.forward(1, value=2), "forward-result")
        self.assertIs(self.runner.forward, original_forward)
        original_forward.assert_called_once_with(1, value=2)
        self.assertEqual(events, ["reserve", "weights", "draft-weights"])
        host.destroy()
        gc.collect()
        self.assertIsNone(arena_ref())

    def test_early_and_late_arena_have_identical_layout_and_page_bytes(self):
        pools = (make_pool(), make_pool(1, (0,)))
        with mock.patch.dict(os.environ, {EARLY_ENV: "0"}):
            late = self.make_host(pools=pools)
        with self.scope(self.runner):
            pass
        early = self.make_host(pools=pools)
        self.assertEqual(len(self.allocator.calls), 2)
        fill_bytes(late, 9201)
        fill_bytes(early, 9201)
        for old, new in zip(late._components, early._components):
            self.assertEqual(
                (old.shape, old.dtype, old.stride()),
                (new.shape, new.dtype, new.stride()),
            )
            self.assertTrue(torch.equal(byte_view(old), byte_view(new)))
        self.assertEqual(late.get_storage_namespace(), early.get_storage_namespace())
        self.assertEqual(late.page_bytes, early.page_bytes)
        for index in (1024, 0, 768, 256, 512):
            self.assertTrue(
                torch.equal(late.get_data_page(index), early.get_data_page(index))
            )

    def test_64gb_is_decimal_cli_budget_without_real_large_allocation(self):
        self.memory.hicache_size = 64
        allocator = mock.Mock(return_value=SimpleNamespace())
        self.core["ALLOC_MEMORY_FUNCS"]["npu"] = allocator
        with self.scope(self.runner):
            pass
        allocator.assert_called_once_with(
            (64_000_000_000,),
            dtype=torch.uint8,
            device="cpu",
            pin_memory=True,
            allocator=None,
        )
        self.core["_early_host_arena"].discard()

    def test_default_disabled_decode_non_npu_and_draft_do_not_reserve(self):
        forward = self.runner.forward
        for env, device, role, draft in (
            ("0", "npu", "prefill", False),
            ("1", "npu", "decode", False),
            ("1", "cpu", "prefill", False),
            ("1", "cuda", "prefill", False),
            ("1", "npu", "prefill", True),
        ):
            with self.subTest(env=env, device=device, role=role, draft=draft):
                self.runner.device = device
                self.runner.is_draft_worker = draft
                self.disagg.disaggregation_mode = role
                with mock.patch.dict(os.environ, {EARLY_ENV: env}), self.scope(
                    self.runner
                ):
                    pass
                self.assertIs(self.runner.forward, forward)
                self.assertIsNone(self.core["_early_host_arena"])
        self.assertEqual(self.allocator.calls, [])

    def test_decode_pool_still_uses_old_four_allocations(self):
        self.disagg.disaggregation_mode = "decode"
        with self.scope(self.runner):
            pass
        host = self.make_host()
        self.assertEqual(len(self.allocator.calls), 4)
        self.assertIsNone(host._packed_host_arena)

    def test_budget_and_precision_guards_fail_before_any_allocation(self):
        for budget in (0, -1, float("nan"), float("inf"), 65, 1e-12):
            with self.subTest(budget=budget):
                self.memory.hicache_size = budget
                with self.assertRaises(ValueError), self.scope(self.runner):
                    self.fail("weights must not load")
        self.memory.hicache_size = 0.02
        for obj, field, invalid in (
            (self.memory, "enable_hierarchical_cache", False),
            (self.memory, "disable_radix_cache", True),
            (self.memory, "hicache_mem_layout", "layer_first"),
            (self.memory, "hicache_io_backend", "direct"),
            (self.memory, "hicache_storage_backend", "shm"),
            (self.model_args, "kv_cache_dtype", "bfloat16"),
            (self.parallel, "dcp_enabled", True),
            (self.runner.ps, "attn_dp_size", 8),
            (self.runner.ps, "pp_size", 2),
            (self.runner.model_config, "quantization", None),
        ):
            old = getattr(obj, field)
            with self.subTest(field=field):
                setattr(obj, field, invalid)
                with self.assertRaises(ValueError), self.scope(self.runner):
                    self.fail("weights must not load")
                setattr(obj, field, old)
        self.quant = {"model.weight": "W4A8"}
        with self.assertRaisesRegex(ValueError, "W8A8"), self.scope(self.runner):
            pass
        self.assertEqual(self.allocator.calls, [])

    def test_packed_graph_and_ssm_guards(self):
        for packed, graphs_disabled, dtype in (
            ("0", True, torch.float32),
            ("1", False, torch.float32),
            ("1", True, torch.bfloat16),
        ):
            self.graphs_disabled, self.ssm_dtype = graphs_disabled, dtype
            with mock.patch.dict(os.environ, {PACKED_ENV: packed}):
                with self.assertRaises(ValueError), self.scope(self.runner):
                    pass
        self.assertEqual(self.allocator.calls, [])

    def test_two_node_topology_is_supported(self):
        self.runner.ps.tp_size = self.runner.ps.moe_ep_size = 16
        self.runner.ps.attn_dp_size = self.parallel.nnodes = 2
        with self.scope(self.runner):
            pass
        self.make_host()
        self.assertEqual(len(self.allocator.calls), 1)

    def test_wrong_device_and_allocator_failure_publish_no_reservation(self):
        self.device.current_device = lambda: 7
        with self.assertRaisesRegex(RuntimeError, "assigned NPU"), self.scope(
            self.runner
        ):
            pass
        self.assertEqual(self.allocator.calls, [])
        self.device.current_device = lambda: 3
        self.core["ALLOC_MEMORY_FUNCS"]["npu"] = mock.Mock(
            side_effect=RuntimeError("207001")
        )
        with self.assertRaisesRegex(RuntimeError, "207001"), self.scope(self.runner):
            pass
        self.assertIsNone(self.core["_early_host_arena"])

    def test_model_and_draft_loading_failure_release_pending_reference(self):
        for failure_at_draft in (False, True):
            with self.subTest(draft=failure_at_draft):
                with self.assertRaisesRegex(RuntimeError, "loading failed"):
                    with self.scope(self.runner):
                        ref = weakref.ref(self.core["_early_host_arena"].arena)
                        if not failure_at_draft:
                            raise RuntimeError("loading failed")
                    with self.scope(
                        SimpleNamespace(device="npu", is_draft_worker=True)
                    ):
                        raise RuntimeError("loading failed")
                gc.collect()
                self.assertIsNone(self.core["_early_host_arena"])
                self.assertIsNone(ref())

    def test_unconsumed_is_fatal_before_first_forward(self):
        original_forward = self.runner.forward
        with self.scope(self.runner):
            ref = weakref.ref(self.core["_early_host_arena"].arena)
        with self.assertRaisesRegex(RuntimeError, "not consumed"):
            self.runner.forward()
        original_forward.assert_not_called()
        gc.collect()
        self.assertIsNone(ref())
        self.assertIsNone(self.core["_early_host_arena"])

    def test_missing_reservation_does_not_silently_allocate_late(self):
        with self.assertRaisesRegex(RuntimeError, "reservation missing"):
            self.make_host()
        self.assertEqual(self.allocator.calls, [])

    def test_insufficient_budget_cleans_up_without_second_allocation(self):
        self.memory.hicache_size = 0.000001
        with self.scope(self.runner):
            ref = weakref.ref(self.core["_early_host_arena"].arena)
        with self.assertRaisesRegex(RuntimeError, "insufficient"):
            self.make_host()
        gc.collect()
        self.assertIsNone(ref())
        self.assertIsNone(self.core["_early_host_arena"])
        self.assertEqual(len(self.allocator.calls), 1)

    def test_pool_validation_failure_and_disabled_packing_do_not_leak(self):
        for case in ("bad-page", "no-pin", "disabled-packed", "disabled-early"):
            with self.subTest(case=case):
                self.allocator.calls.clear()
                with self.scope(self.runner):
                    ref = weakref.ref(self.core["_early_host_arena"].arena)
                pools = (make_pool(), make_pool(1, (0,)))
                if case == "bad-page":
                    pools[1].page_size = 16
                env = (
                    {PACKED_ENV: "0"}
                    if case == "disabled-packed"
                    else ({EARLY_ENV: "0"} if case == "disabled-early" else {})
                )
                with mock.patch.dict(os.environ, env), self.assertRaises(
                    (ValueError, RuntimeError)
                ):
                    self.make_host(pools=pools, pin_memory=case != "no-pin")
                gc.collect()
                self.assertIsNone(ref())
                self.assertIsNone(self.core["_early_host_arena"])
                self.assertEqual(len(self.allocator.calls), 1)

    def test_second_claim_or_second_host_pool_never_allocates_again(self):
        with self.scope(self.runner):
            record = self.core["_early_host_arena"]
        record.claim(1024)
        with self.assertRaisesRegex(RuntimeError, "not available"):
            record.claim(1024)
        record.discard()
        with self.scope(self.runner):
            pass
        self.make_host()
        before = len(self.allocator.calls)
        with self.assertRaisesRegex(RuntimeError, "missing"):
            self.make_host()
        self.assertEqual(len(self.allocator.calls), before)

    def test_base_initialization_failure_after_views_releases_ownership(self):
        with self.scope(self.runner):
            record = self.core["_early_host_arena"]
            ref = weakref.ref(record.arena)
        base = self.core["HostKVCache"]
        initialize = base.__init__

        def fail_after_views(pool, *args, **kwargs):
            initialize(pool, *args, **kwargs)
            self.assertTrue(record.claimed)
            raise RuntimeError("host pool initialization failed")

        with mock.patch.object(base, "__init__", fail_after_views):
            with self.assertRaisesRegex(RuntimeError, "host pool initialization failed"):
                self.make_host()
        gc.collect()
        self.assertIsNone(self.core["_early_host_arena"])
        self.assertIsNone(record.arena)
        self.assertFalse(record.consumed)
        self.assertIsNone(ref())
        self.assertEqual(len(self.allocator.calls), 1)

    def test_device_change_after_reservation_fails_without_second_allocation(self):
        with self.scope(self.runner):
            ref = weakref.ref(self.core["_early_host_arena"].arena)
        self.device.current_device = lambda: 6
        with self.assertRaisesRegex(RuntimeError, "another NPU"):
            self.make_host()
        gc.collect()
        self.assertIsNone(ref())
        self.assertIsNone(self.core["_early_host_arena"])
        self.assertEqual(len(self.allocator.calls), 1)

    def test_real_runner_hook_occurs_after_device_binding_before_load_model(self):
        tree = ast.parse(RUNNER.read_text())
        runner = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "ModelRunner"
        )
        init = next(
            n
            for n in runner.body
            if isinstance(n, ast.FunctionDef) and n.name == "__init__"
        )
        device_call = next(
            n
            for n in ast.walk(init)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "set_device"
        )
        hook = next(
            n
            for n in ast.walk(init)
            if isinstance(n, ast.With)
            and any(
                isinstance(i.context_expr, ast.Call)
                and isinstance(i.context_expr.func, ast.Name)
                and i.context_expr.func.id == "glm53_early_host_reservation"
                for i in n.items
            )
        )
        self.assertLess(device_call.lineno, hook.lineno)
        self.assertEqual(ast.unparse(hook.body[0]), "self.initialize()")
        initialize = next(
            n
            for n in runner.body
            if isinstance(n, ast.FunctionDef) and n.name == "initialize"
        )
        self.assertIn("self.load_model()", ast.unparse(initialize))
        # Execute the actual hook block with a simulated weight loader.
        self.runner.initialize = lambda: self.assertIsNotNone(
            self.core["_early_host_arena"]
        )
        exec(
            compile(ast.Module(body=[hook], type_ignores=[]), str(RUNNER), "exec"),
            {"self": self.runner, "glm53_early_host_reservation": self.scope},
        )
        self.assertEqual(len(self.allocator.calls), 1)

    def test_runtime_imports_and_config_fields_exist_in_real_source(self):
        tree = ast.parse(SOURCE.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module.startswith(
                "sglang."
            ):
                continue
            path = ROOT / "python" / (node.module.replace(".", "/") + ".py")
            if not path.exists():
                continue
            exports = {
                n.name
                for n in ast.parse(path.read_text()).body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef))
            }
            if node.module.endswith(
                ("runtime_context", "mamba_utils", "cuda_graph_config")
            ):
                self.assertTrue({n.name for n in node.names} <= exports, str(path))
        for filename, expected in (
            ("memory.py", vars(self.memory)),
            ("model.py", vars(self.model_args)),
            ("schedule.py", vars(self.schedule)),
        ):
            path = ROOT / "python/sglang/srt/arg_groups/fields" / filename
            names = {
                n.target.id
                for n in ast.walk(ast.parse(path.read_text()))
                if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
            }
            self.assertTrue(
                set(expected) <= names, f"{filename}: {set(expected) - names}"
            )


if __name__ == "__main__":
    unittest.main()
