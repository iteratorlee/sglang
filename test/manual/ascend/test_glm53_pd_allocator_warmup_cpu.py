"""CPU contracts: exact dtypes/keys, scratch-only ownership and default-off hook."""

import ast
import importlib.util
import os
from pathlib import Path
import random
import sys
import types
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[3]
SOURCE = (
    REPO
    / "python/sglang/srt/hardware_backend/npu/attention/glm53/pd_allocator_warmup.py"
)
spec = importlib.util.spec_from_file_location("pd_allocator_warmup_cpu", SOURCE)
warm = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = warm
spec.loader.exec_module(warm)


class WarmupContracts(unittest.TestCase):
    def test_dense_bs_real_specializations_and_both_free_alignments(self):
        plan = warm.WarmupPlan(4, 132231, 64, 8, "npu:0")
        cases = warm.variants(plan)
        self.assertEqual({v.batch for v in cases if v.kernel == "assign"}, {1, 2, 3, 4})
        self.assertEqual({v.bs_upper for v in cases if v.kernel == "assign"}, {1, 2, 4})
        self.assertEqual({v.free_offset for v in cases if v.kernel == "extend"}, {0, 1})
        self.assertEqual({v.batch for v in cases if v.lens_dtype == "int64"}, {1})
        self.assertIn(warm.Variant("extend", 2, "int32", 1, 256), cases)
        self.assertIn(warm.Variant("extend", 4, "int32", 1, 512), cases)
        self.assertIn(warm.Variant("extend", 1, "int64", 1, 16384), cases)
        for bs in range(1, 5):
            for offset in (0, 1):
                self.assertIn(warm.Variant("extend", bs, "int32", offset, 2048), cases)

    def test_key_coverage_for_independent_page_allocation_states(self):
        rng = random.Random(192)
        for page in (16, 64):
            for reserve in (8, 64, 128):
                plan = warm.WarmupPlan(4, 132231, page, reserve, "npu:0")
                keys = {
                    (v.batch, v.max_extend)
                    for v in warm.variants(plan)
                    if v.kernel == "extend" and v.lens_dtype == "int32"
                }
                for bs in range(1, 5):
                    for _ in range(1000):
                        cur = [rng.randrange(0, page * 8) for _ in range(bs)]
                        committed = [
                            max(0, x - rng.randrange(reserve + 1)) for x in cur
                        ]
                        nxt = [
                            max(c, (s + reserve + page - 1) // page * page)
                            for c, s in zip(cur, committed)
                        ]
                        delta = sum(n - c for c, n in zip(cur, nxt))
                        if delta:
                            self.assertIn((bs, warm.power2(delta)), keys)

    def test_invalid_metadata_is_rejected_before_torch(self):
        cases = [
            warm.WarmupPlan(0, 1000, 64, 8, "npu"),
            warm.WarmupPlan(4, 0, 64, 8, "npu"),
            warm.WarmupPlan(4, 1000, 3, 8, "npu"),
            warm.WarmupPlan(4, 1000, 64, 8, "npu", req_dtype="float16"),
            warm.WarmupPlan(4, 1000, 64, 8, "npu", free_dtype="float16"),
        ]
        for plan in cases:
            with self.assertRaises(ValueError):
                warm.variants(plan)

    def test_default_off_and_ineligible_never_import_torch(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(warm.maybe_warm_pd_allocators(object()))
        for flag in ("0", "true", "yes", "2", " 1"):
            with patch.dict(os.environ, {warm.FLAG: flag}):
                self.assertIsNone(warm.maybe_warm_pd_allocators(object()))
        with patch.dict(os.environ, {warm.FLAG: "1"}):
            self.assertIsNone(
                warm.maybe_warm_pd_allocators(
                    types.SimpleNamespace(is_draft_worker=True)
                )
            )
            self.assertIsNone(
                warm.maybe_warm_pd_allocators(
                    types.SimpleNamespace(is_draft_worker=False, device="cuda")
                )
            )
            for mode in ("null", "prefill"):
                runner = types.SimpleNamespace(
                    is_draft_worker=False,
                    device="npu",
                    server_args=types.SimpleNamespace(disaggregation_mode=mode),
                )
                self.assertIsNone(warm.maybe_warm_pd_allocators(runner))

    def test_hook_uses_only_metadata_and_skips_second_target_call(self):
        class MetadataOnly:
            dtype = "i32"
            shape = (300, 132231)
            device = types.SimpleNamespace(type="npu")

            def is_contiguous(self):
                return True

            def __getattr__(self, name):
                raise AssertionError("live content access: " + name)

        class Allocator:
            page_size = 64
            free_pages = MetadataOnly()

            def __getattr__(self, name):
                raise AssertionError("live allocator operation: " + name)

        Allocator.free_pages.dtype = "i64"
        torch = types.ModuleType("torch")
        torch.int32, torch.int64 = "i32", "i64"
        modules = {"torch": torch}
        for name, attrs in {
            "sglang.srt.hardware_backend.npu.allocator_npu": {
                "NPUPagedTokenToKVPoolAllocator": Allocator
            },
            "sglang.srt.mem_cache.allocation_sizing": {
                "get_alloc_reserve_per_decode": lambda: 8
            },
            "sglang.srt.runtime_context": {
                "get_spec": lambda: types.SimpleNamespace(speculative_algorithm="EAGLE")
            },
            "sglang.srt.mem_cache.allocation": {
                "attention_backends": lambda: ("ascend", "ascend")
            },
        }.items():
            module = types.ModuleType(name)
            module.__dict__.update(attrs)
            modules[name] = module
        table = MetadataOnly()
        table.dtype = "i32"
        table.device = Allocator.free_pages.device
        runner = types.SimpleNamespace(
            is_draft_worker=False,
            device="npu",
            server_args=types.SimpleNamespace(disaggregation_mode="decode"),
            model_config=types.SimpleNamespace(
                hf_config=types.SimpleNamespace(
                    architectures=["Glm5NextForConditionalGeneration"]
                )
            ),
            token_to_kv_pool_allocator=Allocator(),
            req_to_token_pool=types.SimpleNamespace(req_to_token=table),
            max_running_requests=4,
        )
        seen = []

        def scratch(plan):
            seen.append(plan)
            return {"private": True}

        with patch.dict(sys.modules, modules), patch.dict(
            os.environ, {warm.FLAG: "1"}
        ), patch.object(warm, "warm_scratch", scratch):
            self.assertEqual(warm.maybe_warm_pd_allocators(runner), {"private": True})
            self.assertIsNone(warm.maybe_warm_pd_allocators(runner))
        self.assertEqual(len(seen), 1)
        self.assertEqual((seen[0].pool_len, seen[0].max_running), (132231, 4))
        self.assertTrue(
            all(isinstance(v, (str, int)) for v in seen[0].__dict__.values())
        )

    def test_no_live_pool_call_and_only_minimal_model_runner_ast_change(self):
        tree = ast.parse(SOURCE.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(
                    node.func.attr,
                    ("alloc", "alloc_extend", "clear", "clone", "copy_", "empty_like"),
                )
        current = ast.parse(
            (REPO / "python/sglang/srt/model_executor/model_runner.py").read_text()
        )
        baseline = ast.parse(
            (
                REPO.parent
                / "pd-allocator-warmup/baseline/python/sglang/srt/model_executor/model_runner.py"
            ).read_text()
        )
        current.body = [
            n
            for n in current.body
            if not (isinstance(n, ast.Import) and [a.name for a in n.names] == ["os"])
        ]

        class RemoveHook(ast.NodeTransformer):
            def visit_If(self, node):
                if warm.FLAG in ast.unparse(node.test):
                    return None
                return self.generic_visit(node)

        self.assertEqual(ast.dump(RemoveHook().visit(current)), ast.dump(baseline))


if __name__ == "__main__":
    unittest.main(verbosity=2)
