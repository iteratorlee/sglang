"""Portable CPU contracts for the current static-route and Scheduler sources.

Run with a local CPU Python environment:
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  python -B test/manual/ascend/test_glm53_static_gpu_route_cpu.py

Loads leaf files in an isolated package and interprets kernel/Scheduler ASTs.
No serving initialization, Triton import, real Gloo, device query or NPU work.
These tests do not prove device compilation, FFN parity or model precision.
"""

import ast
from concurrent.futures import ThreadPoolExecutor
import copy
import importlib.util
import os
from pathlib import Path
import sys
import threading
import types
import unittest
from unittest.mock import patch

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ""
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"
sys.dont_write_bytecode = True

import torch

torch.set_num_threads(1)

from glm53_static_gpu_route_cpu_adapter import RoutingKernelsCPU

REPO = Path(__file__).resolve().parents[3]
MOE = REPO / "python/sglang/srt/hardware_backend/npu/moe"
SCHEDULER = REPO / "python/sglang/srt/managers/scheduler.py"
PACKAGE = "_glm53_static_route_cpu_test"
CONTROL_KEY = "glm53_normal_static_gpu_route"


def load_leaf(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


package = types.ModuleType(PACKAGE)
package.__path__ = [str(MOE)]
sys.modules[PACKAGE] = package
host = load_leaf(PACKAGE + ".glm53_static_gpu_route", MOE / "glm53_static_gpu_route.py")
with patch.dict(os.environ, {"SGLANG_GLM53_NORMAL_STATIC_GPU_ROUTE": "0"}):
    collectives = load_leaf(PACKAGE + ".glm53_collectives", MOE / "glm53_collectives.py")


def functions(path):
    return {node.name: node for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.FunctionDef)}


def bits(tensor):
    return tensor.reshape(-1).clone(memory_format=torch.contiguous_format).view(torch.uint8)


def ids_fixture(local, dtype, kind, rank):
    n = local * 16 * 8
    values = ((torch.arange(n, dtype=torch.int64) // 2 + 17) % 288).to(dtype)
    if kind == "hot":
        values[:] = rank * 18 + 7
    elif kind == "none":
        values[:] = -1
    elif kind == "extremes":
        low = rank * 18
        pool = [torch.iinfo(dtype).min, torch.iinfo(dtype).max, -1, 288,
                low, low + 17, low, low + 1]
        if dtype == torch.int64:
            pool += [(1 << 32) + low, -(1 << 32) + low]
        values = torch.tensor(pool, dtype=dtype)[torch.arange(n) % len(pool)]
    return values.reshape(local * 16, 8)


def reference_plan(ids, original, rank):
    """Independent stable Python ordering; never narrows full-width expert IDs."""
    values = ids.reshape(-1).tolist()
    local, low = ids.shape[0] // 16, rank * 18
    selected = [offset for offset, value in enumerate(values)
                if low <= value < low + 18 and original[offset // (local * 64)] > 0]
    selected.sort(key=lambda offset: (values[offset], offset))
    counts = [0] * 18
    inverse = torch.full(ids.shape, -1, dtype=torch.int32)
    used = [0] * ids.shape[0]
    for destination, offset in enumerate(selected):
        counts[values[offset] - low] += 1
        row = offset // 8
        inverse[row, used[row]] = destination
        used[row] += 1
    return selected, torch.tensor(counts, dtype=torch.int64), inverse


class RoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kernels = RoutingKernelsCPU(MOE / "glm53_static_gpu_route_triton.py")

    def assert_plan(self, ids, original, geometry, rank):
        selected, counts, inverse, total = self.kernels.plan(ids, geometry, rank)
        expected, expected_counts, expected_inverse = reference_plan(ids, original, rank)
        n = int(total[0])
        self.assertEqual(selected[:n].tolist(), expected)
        self.assertEqual(counts.dtype, torch.int64)
        self.assertTrue(torch.equal(counts, expected_counts))
        self.assertTrue(torch.equal(inverse, expected_inverse))
        self.assertTrue(bool(((inverse >= -1) & (inverse < n)).all()))
        return selected, counts, inverse, total

    def test_560_count_prefix_compact_cases(self):
        cases = 0
        for local in (1, 3, 23, 32, 64):
            padded = local * 8
            originals = [(1, 0), (0, max(1, padded - 3))]
            if padded * 2 <= 512:
                originals.append((max(1, padded - 7), 1))
            for original in originals:
                geometry = host.static_route_geometry(16, (original, (padded, padded)))
                self.assertIsNotNone(geometry)
                for dtype in (torch.int32, torch.int64):
                    for rank in (0, 1, 7, 8, 15):
                        for kind in ("mixed", "hot", "none", "extremes"):
                            with self.subTest(local=local, original=original, dtype=dtype, rank=rank, kind=kind):
                                self.assert_plan(ids_fixture(local, dtype, kind, rank), original, geometry, rank)
                            cases += 1
        self.assertEqual(cases, 560)

    def test_positive_domain_inner_padding_is_kept(self):
        original = (1, 0)
        g = host.static_route_geometry(16, (original, (184, 184)))
        _, counts, inverse, total = self.assert_plan(ids_fixture(23, torch.int64, "hot", 7), original, g, 7)
        self.assertEqual(int(total[0]), 184 * 8)
        self.assertEqual(int(counts[7]), 184 * 8)
        self.assertTrue(bool((inverse[184:] == -1).all()))

    def test_fullwidth_collisions_duplicates_all16_ranks(self):
        generator = torch.Generator(device="cpu").manual_seed(534)
        for dtype in (torch.int32, torch.int64):
            for rank in range(16):
                low = rank * 18
                pool = [low, low, low + 17, low + 1, low + 7, -1, 288,
                        torch.iinfo(dtype).min, torch.iinfo(dtype).max]
                if dtype == torch.int64:
                    pool += [(1 << 32) + low, -(1 << 32) + low,
                             (1 << 32) + low + 17, -(1 << 32) + low + 17]
                ids = torch.tensor(pool, dtype=dtype)[torch.randint(len(pool), (48, 8), generator=generator)]
                ids[0] = torch.tensor([low + 17, low, low + 1, low, low + 7, low + 7, -1, low], dtype=dtype)
                for original in ((1, 0), (0, 1), (1, 1)):
                    g = host.static_route_geometry(16, (original, (24, 24)))
                    self.assert_plan(ids, original, g, rank)

    def test_block_and_domain_boundaries(self):
        ids = torch.full((368, 8), -1, dtype=torch.int64)
        for row in (0, 127, 128, 183, 184, 255, 256, 367):
            ids[row] = torch.tensor([287, 270, 271, 270, 287, 271, 270, 287])
        for original in ((1, 0), (0, 1), (1, 1)):
            self.assert_plan(ids, original, host.static_route_geometry(16, (original, (184, 184))), 15)

    def test_actual_gather_raw_bits_and_zero_tail(self):
        words = torch.tensor([0, -2147483648, 1, -2147483647, 0x7F800000, -8388608,
                              0x7FC01234, 0x7F800001, -4194303, 0x3F000000], dtype=torch.int32)
        q = (torch.arange(48 * 4096, dtype=torch.int32) % 251 - 125).to(torch.int8).reshape(48, 4096)
        scales = words[torch.arange(48) % len(words)].view(torch.float32)
        gates = words[torch.arange(48 * 8) % len(words)].view(torch.float32).reshape(48, 8)
        for dtype in (torch.int32, torch.int64):
            for original in ((1, 0), (0, 9), (9, 1)):
                g = host.static_route_geometry(16, (original, (24, 24)))
                ids = ids_fixture(3, dtype, "extremes", 8)
                selected, _, _, total = self.assert_plan(ids, original, g, 8)
                recv, rs, rows, weights = self.kernels.gather(q, scales, gates, selected, total)
                expected, _, _ = reference_plan(ids, original, 8)
                loc = torch.tensor(expected, dtype=torch.int64)
                n = len(expected)
                for actual, wanted in ((recv[:n], q[loc // 8]), (rs[:n], scales[loc // 8]),
                                       (weights[:n], gates.reshape(-1)[loc]), (rows[:n], loc // 8)):
                    self.assertTrue(torch.equal(bits(actual), bits(wanted)))
                for tail in (recv[n:], rs[n:], weights[n:], rows[n:]):
                    self.assertFalse(bool(bits(tail).any()))

    def test_geometry_bounds_preserve_small_kernel_unit(self):
        for world, metadata in ((8, ((128, 0), (128, 128))), (16, None),
                                (16, ((0, 0), (128, 128))), (16, ((513, 0), (520, 520))),
                                (16, ((512, 1), (512, 512))), (16, ((1, 0), (9, 9))),
                                (16, ((1, 0), (8, 16))), (16, ((True, 0), (8, 8)))):
            self.assertIsNone(host.static_route_geometry(world, metadata))
        # The production minimum belongs to the caller, not this math helper.
        self.assertIsNotNone(host.static_route_geometry(16, ((1, 1), (8, 8))))
        self.assertEqual(host.static_route_geometry(16, ((512, 0), (512, 512))).capacity, 4096)
        self.assertEqual(host.static_route_geometry(16, ((256, 256), (256, 256))).capacity, 4096)


def admission_namespace():
    dispatch = functions(MOE / "glm53_collectives.py")["collective_dispatch"]
    block = next(node for node in dispatch.body if isinstance(node, ast.If) and
                 any(isinstance(n, ast.Name) and n.id == "_NORMAL_STATIC_GPU_ROUTE_ENABLED"
                     for n in ast.walk(node.test)))
    fn = ast.FunctionDef(name="admit", args=ast.arguments(posonlyargs=[], args=[], vararg=None,
                          kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
                         body=[copy.deepcopy(block), ast.Return(ast.Constant("fallback"))],
                         decorator_list=[], returns=None, type_comment=None)
    ns = dict(collectives.__dict__)
    ns.update(_NORMAL_STATIC_GPU_ROUTE_ENABLED=True, ordered=True, num_experts=288,
              group=None, world=16, rank=0, width=4096,
              x=types.SimpleNamespace(dtype="bf16"), q=types.SimpleNamespace(dtype="i8"),
              scale=types.SimpleNamespace(dtype="f32"), all_q=None, all_scale=None,
              all_ids=None, all_weights=None, _static_route_rank_mapping=lambda *args: True)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
                 "<actual_collective_admission>", "exec"), ns)
    return ns


class CallerAndControlFlagTests(unittest.TestCase):
    def run_gate(self, original, padded, expected, *, enabled=True, idle_match=True):
        ns = admission_namespace()
        ns.update(tokens=padded[0] // 8, _NORMAL_STATIC_GPU_ROUTE_ENABLED=enabled)
        a = collectives._packed_target_metadata.set((original, padded))
        b = collectives._idle_dp_counts.set(original if idle_match else (0, 0))
        try:
            with patch.dict(os.environ, {"SGLANG_GLM53_NORMAL_COMBINE_ORDERED": "2node",
                                         "SGLANG_GLM53_NORMAL_SKIP_IDLE_DP": "1"}), \
                 patch.object(host, "payload_supported", return_value=True), \
                 patch.object(host, "static_gpu_dispatch", return_value="static") as invoke:
                for rank in range(16):
                    ns["rank"] = rank
                    self.assertEqual(ns["admit"](), "static" if expected else "fallback")
                self.assertEqual(invoke.call_count, 16 if expected else 0)
        finally:
            collectives._packed_target_metadata.reset(a)
            collectives._idle_dp_counts.reset(b)

    def test_shared_caller_minimum_and_capacity_boundaries(self):
        for original, padded, eligible in (
            ((1, 0), (184, 184), False), ((0, 5), (8, 8), False),
            ((127, 0), (128, 128), False), ((0, 127), (128, 128), False),
            ((127, 127), (128, 128), False), ((128, 0), (128, 128), True),
            ((0, 128), (128, 128), True), ((128, 1), (128, 128), True),
            ((129, 0), (136, 136), True), ((0, 181), (184, 184), True),
            ((177, 1), (184, 184), True), ((256, 256), (256, 256), True),
            ((512, 0), (512, 512), True), ((512, 1), (512, 512), False),
            ((513, 0), (520, 520), False), ((0, 0), (128, 128), False),
        ):
            with self.subTest(original=original, padded=padded):
                self.run_gate(original, padded, eligible)

    def test_off_and_bad_metadata_fall_back(self):
        self.run_gate((180, 0), (184, 184), False, enabled=False)
        self.run_gate((180, 0), (184, 184), False, idle_match=False)
        self.run_gate((128, 0), (120, 120), False)
        self.run_gate((128, 0), (128, 136), False)

    def test_startup_capability_is_immutable_and_getter_tracks_ABBA(self):
        env = "SGLANG_GLM53_NORMAL_STATIC_GPU_ROUTE"
        with patch.object(collectives, "_NORMAL_STATIC_GPU_ROUTE_CAPABLE", False), \
             patch.object(collectives, "_NORMAL_STATIC_GPU_ROUTE_ENABLED", False), \
             patch.dict(os.environ, {env: "1"}):
            with self.assertRaises(RuntimeError):
                collectives.set_normal_static_gpu_route(1)
            collectives.set_normal_static_gpu_route(0)
            self.assertFalse(collectives.get_normal_static_gpu_route_stats()["startup_capable"])
        with patch.object(collectives, "_NORMAL_STATIC_GPU_ROUTE_CAPABLE", True), \
             patch.object(collectives, "_NORMAL_STATIC_GPU_ROUTE_ENABLED", True), \
             patch.dict(os.environ, {env: "0"}):
            for enabled in (0, 1, 1, 0):
                collectives.set_normal_static_gpu_route(enabled)
                stats = collectives.get_normal_static_gpu_route_stats()
                self.assertTrue(stats["startup_capable"])
                self.assertEqual(stats["enabled"], bool(enabled))
                self.assertEqual(stats["min_shared_original_max"], 128)
            for invalid in (-1, 2, "1", None):
                with self.assertRaises(ValueError):
                    collectives.set_normal_static_gpu_route(invalid)

    def test_activation_is_once_per_rank_and_stats_are_copy_safe(self):
        g = host.static_route_geometry(16, ((512, 0), (512, 512)))
        with patch.object(host, "_rank_stats", {}), patch.object(host, "_calls", 0), \
             patch.object(host, "_capacity_rows", 0), patch.object(host, "_max_capacity_rows", 0), \
             patch.object(host.logger, "info") as log:
            for _ in range(2):
                for rank in range(16):
                    host._record_route_enqueue(g, rank)
            self.assertEqual(log.call_count, 16)
            stats = host.get_static_gpu_route_stats()
            self.assertEqual(stats["static_gpu_route_calls"], 32)
            self.assertEqual(stats["static_gpu_route_capacity_rows"], 32 * 4096)
            self.assertEqual([r["ep_rank"] for r in stats["rank_stats"]], list(range(16)))
            stats["rank_stats"][0]["calls"] = -1
            self.assertEqual(host.get_static_gpu_route_stats()["rank_stats"][0]["calls"], 2)


class _Bus:
    """Only a CPU thread barrier. No real distributed backend is imported/run."""
    def __init__(self):
        self.cv, self.rounds, self.local = threading.Condition(), {}, threading.local()

    def all_gather_object(self, output, row, group):
        if group != "full_tp_cpu":
            raise AssertionError("admin used a group other than full TP CPU")
        self.local.step = getattr(self.local, "step", 0) + 1
        with self.cv:
            rows = self.rounds.setdefault(self.local.step, {})
            rank = self.local.rank
            if rank in rows:
                raise AssertionError("duplicate mock participant")
            rows[rank] = copy.deepcopy(row)
            self.cv.notify_all()
            if not self.cv.wait_for(lambda: len(rows) == 16, timeout=5):
                raise RuntimeError("CPU protocol mock stranded a rank")
            output[:] = [copy.deepcopy(rows[k]) for k in range(16)]


def scheduler_methods():
    cls = next(n for n in ast.parse(SCHEDULER.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
    return {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}


class _AdminHarness:
    def __init__(self):
        self.states = [dict(capable=True, enabled=True, idle=True, scope=True,
                            getter_error=False, setter_error=False, minimum=128) for _ in range(16)]
        self.mutations, self.bus = [], _Bus()
        methods = [copy.deepcopy(n) for name, n in scheduler_methods().items()
                   if name.startswith("_glm53_static_route_") or
                   name in ("_set_glm53_static_route_control", "set_internal_state")]
        for method in methods:
            method.returns = None
            for arg in method.args.args:
                arg.annotation = None
        cls = ast.ClassDef(name="CPU_Scheduler", bases=[], keywords=[], body=methods, decorator_list=[])
        self.dist = types.SimpleNamespace(get_world_size=lambda **kw: 16,
                get_backend=lambda **kw: "gloo",
                all_gather_object=lambda *a, **kw: self.bus.all_gather_object(*a, **kw))
        ns = dict(torch=types.SimpleNamespace(distributed=self.dist),
                  DisaggregationMode=types.SimpleNamespace(PREFILL="P", DECODE="D"),
                  SetInternalStateReqOutput=lambda updated: types.SimpleNamespace(updated=updated))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
                     "<actual_scheduler_admin_methods>", "exec"), ns)
        self.workers = []
        for rank in range(16):
            worker = ns["CPU_Scheduler"]()
            worker.ps = types.SimpleNamespace(tp_size=16, moe_ep_size=16, attn_dp_size=2,
                attn_tp_size=8, pp_size=1, attn_cp_size=1, tp_rank=rank,
                moe_ep_rank=rank, attn_dp_rank=rank // 8, attn_tp_rank=rank % 8)
            worker.disaggregation_mode, worker.tp_cpu_group = "P", "full_tp_cpu"
            worker.is_fully_idle = lambda for_health_check=False, rank=rank: (
                not for_health_check and self.states[rank]["idle"])
            self.workers.append(worker)

    def modules(self):
        name = "sglang.srt.hardware_backend.npu.moe.glm53_collectives"
        modules, parts = {}, name.split(".")
        for i in range(1, len(parts) + 1):
            key = ".".join(parts[:i])
            modules[key] = types.ModuleType(key)
            modules[key].__path__ = []
        module = modules[name]

        def getter():
            state = self.states[self.bus.local.rank]
            if state["getter_error"]:
                raise RuntimeError("injected local getter failure")
            return dict(startup_capable=state["capable"], enabled=state["enabled"],
                        min_shared_original_max=state["minimum"], static_gpu_route_calls=0,
                        rank_stats=[])

        def setter(value):
            rank = self.bus.local.rank
            state = self.states[rank]
            if state["setter_error"]:
                raise RuntimeError("injected local setter failure")
            if value and not state["capable"]:
                raise RuntimeError("startup capability missing")
            self.mutations.append((rank, value))
            state["enabled"] = bool(value)

        module.get_normal_static_gpu_route_stats = getter
        module.set_normal_static_gpu_route = setter
        module.normal_hccl_skip_idle_dp_enabled = lambda: self.states[self.bus.local.rank]["scope"]
        return modules

    def run(self, value=None, *, read=False, extra=False, disagreeing_rank=None):
        self.bus = _Bus()

        def invoke(rank):
            self.bus.local.rank, self.bus.local.step = rank, 0
            if read:
                return self.workers[rank]._glm53_static_route_control_readback()
            values = {CONTROL_KEY: 1 - value if rank == disagreeing_rank else value}
            if extra:
                values["pp_max_micro_batch_size"] = 1
            return self.workers[rank].set_internal_state(types.SimpleNamespace(server_args=values))

        with patch.dict(sys.modules, self.modules()), ThreadPoolExecutor(max_workers=16) as pool:
            return list(pool.map(invoke, range(16)))


class SchedulerAdminTests(unittest.TestCase):
    def test_all16_ABBA_and_readback(self):
        h = _AdminHarness()
        for value in (0, 1, 1, 0):
            self.assertTrue(all(r.updated for r in h.run(value)))
            for response in h.run(read=True):
                self.assertTrue(response["complete_rank_receipts"])
                self.assertTrue(response["global_idle"])
                self.assertEqual([r["ep_rank"] for r in response["rank_receipts"]], list(range(16)))
                self.assertTrue(all(r["enabled"] == bool(value) for r in response["rank_receipts"]))
        self.assertEqual(len(h.mutations), 64)

    def test_nonleader_busy_rejects_all_before_mutation(self):
        h = _AdminHarness()
        h.states[13]["idle"] = False
        self.assertFalse(any(r.updated for r in h.run(0)))
        self.assertEqual(h.mutations, [])
        self.assertTrue(all(s["enabled"] for s in h.states))

    def test_startup_scope_minimum_and_getter_failure_reject(self):
        for field, value in (("capable", False), ("scope", False), ("minimum", 0), ("getter_error", True)):
            h = _AdminHarness()
            h.states[7][field] = value
            self.assertFalse(any(r.updated for r in h.run(0)), field)
            self.assertEqual(h.mutations, [], field)

    def test_invalid_types_mixed_keys_and_requests_reject(self):
        for value in (True, 1.0, "1", None, -1, 2):
            h = _AdminHarness()
            self.assertFalse(any(r.updated for r in h.run(value)))
            self.assertEqual(h.mutations, [])
        for kwargs in ({"extra": True}, {"disagreeing_rank": 11}):
            h = _AdminHarness()
            self.assertFalse(any(r.updated for r in h.run(0, **kwargs)))
            self.assertEqual(h.mutations, [])

    def test_duplicate_identity_rejects(self):
        h = _AdminHarness()
        h.workers[15].ps.tp_rank = 0
        self.assertFalse(any(r.updated for r in h.run(0)))
        self.assertEqual(h.mutations, [])

    def test_partial_setter_failure_is_hold_not_success_or_retry(self):
        h = _AdminHarness()
        for state in h.states:
            state["enabled"] = False
        h.states[5]["setter_error"] = True
        self.assertFalse(any(r.updated for r in h.run(1)))
        self.assertEqual(len(h.mutations), 15)
        self.assertFalse(h.states[5]["enabled"])
        self.assertTrue(h.states[4]["enabled"])
        for worker in h.workers:
            self.assertEqual(worker._glm53_static_route_last_control["phase"], "HOLD_partial_or_failed")

    def test_D_never_enters_group_or_mutates(self):
        h = _AdminHarness()
        for worker in h.workers:
            worker.disaggregation_mode = "D"
        self.assertFalse(any(r.updated for r in h.run(0)))
        self.assertEqual(h.bus.rounds, {})
        self.assertEqual(h.mutations, [])


class SourceContractTests(unittest.TestCase):
    def test_helpers_have_no_host_counts_sync_or_extra_collective(self):
        banned = {"cpu", "numpy", "tolist", "item", "nonzero", "masked_select", "argsort",
                  "synchronize", "wait", "all_gather_into_tensor", "all_reduce", "mem_get_info"}
        for filename in ("glm53_static_gpu_route.py", "glm53_static_gpu_route_triton.py"):
            tree = ast.parse((MOE / filename).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
                    self.assertNotIn(name, banned, (filename, name))
        text = (MOE / "glm53_static_gpu_route_triton.py").read_text()
        self.assertNotIn("atomic_", text)
        self.assertNotIn("tl.float", text)
        self.assertIn("triton.reinterpret", text)

    def test_fixed_state_contract_and_admin_early_command(self):
        dispatch = functions(MOE / "glm53_static_gpu_route.py")["static_gpu_dispatch"]
        ret = next(n for n in dispatch.body if isinstance(n, ast.Return))
        self.assertEqual(len(ret.value.elts), 4)
        self.assertEqual([ast.unparse(n) for n in ret.value.elts[3].elts],
                         ["rows", "gates", "geometry.local_rows", "16", "4096", "c", "inverse"])
        methods = scheduler_methods()
        setter = methods["set_internal_state"]
        branch = next((i, n) for i, n in enumerate(setter.body) if isinstance(n, ast.If)
                      and CONTROL_KEY in ast.unparse(n.test))
        self.assertLessEqual(branch[0], 1)
        self.assertIsInstance(branch[1].body[0], ast.Return)
        self.assertIn("_set_glm53_static_route_control", ast.unparse(branch[1]))
        getter = ast.unparse(methods["get_internal_state"])
        self.assertIn("self._glm53_static_route_control_readback()", getter)
        for name, node in methods.items():
            if name.startswith("_glm53_static_route_") or name == "_set_glm53_static_route_control":
                text = ast.unparse(node)
                for banned in ("torch.npu", "torch.cuda", ".synchronize(", ".item(", "os.getenv", "get_context("):
                    self.assertNotIn(banned, text)

    def test_no_accelerator_or_serving_package_imported(self):
        self.assertNotIn("torch_npu", sys.modules)
        self.assertNotIn("triton", sys.modules)
        self.assertNotIn("sglang", sys.modules)
        self.assertEqual(torch.get_num_threads(), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
