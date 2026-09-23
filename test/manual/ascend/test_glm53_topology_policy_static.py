"""Static CPU policy tests for the bounded GLM53 P topology candidate.

Run from the source repo:
    python -B test/manual/ascend/test_glm53_topology_policy_static.py -v

The source helpers are extracted with AST. This test imports neither SGLang
nor torch and does not initialize a device or distributed process group.
"""

import ast
import math
import types
import unittest
from pathlib import Path

from glm53_topology_test_schema import (
    PARALLEL_DERIVED_FIELDS,
    PARALLEL_INPUT_FIELDS,
    RUNTIME,
    published_parallel,
    server_args_shape,
)


ROOT = Path(__file__).resolve().parents[3]
BUILDER = "python/sglang/srt/mem_cache/kv_cache_builder.py"
SOURCE = (ROOT / BUILDER).read_text()
NS = types.SimpleNamespace


def extract(names, namespace):
    wanted = set(names)
    nodes = []
    for node in ast.parse(SOURCE).body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(
                isinstance(target, ast.Name) and target.id in wanted
                for target in targets
            ):
                nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in wanted:
            nodes.append(node)
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), BUILDER, "exec"), namespace)
    return namespace


POLICY = extract(
    (
        "_GLM53_VALIDATED_PREFILL_DP_TOPOLOGIES",
        "_glm53_npu_prefill_prefix_topology_supported",
    ),
    {},
)["_glm53_npu_prefill_prefix_topology_supported"]
PAGE_SIZE = extract(
    ("_glm53_npu_prefix_page_size",),
    {
        "math": math,
        "glm5_next_config": lambda model_config: model_config,
    },
)["_glm53_npu_prefix_page_size"]


def parallel(**changes):
    values = dict(
        dp_size=1,
        attn_dp_size=1,
        attn_tp_size=16,
        enable_dp_attention=False,
        nnodes=1,
        tp_size=16,
        ep_size=16,
        moe_dp_size=1,
        dwdp_size=1,
        attn_cp_size=1,
        pp_size=1,
    )
    values.update(changes)
    return NS(**values)


class PrefixTopologyPolicyTests(unittest.TestCase):
    def test_m3_fields_exist_on_source_parallel_context(self):
        helper = next(
            node
            for node in ast.parse(SOURCE).body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_glm53_npu_prefill_prefix_topology_supported"
        )
        reads = {
            node.attr
            for node in ast.walk(helper)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "parallel"
        }
        self.assertTrue(reads <= PARALLEL_INPUT_FIELDS | PARALLEL_DERIVED_FIELDS)
        self.assertNotIn("attn_tp_size", PARALLEL_INPUT_FIELDS)
        for name in ("attn_tp_size", "attn_dp_size", "dcp_enabled"):
            self.assertIsInstance(getattr(RUNTIME["ParallelContext"], name), property)
        for dp, attn_tp, supported in (
            (1, 16, True), (2, 8, True), (4, 4, True), (8, 2, False)
        ):
            with self.subTest(dp=dp):
                cfg = server_args_shape(
                    dp_size=dp,
                    tp_size=16,
                    ep_size=16,
                    enable_dp_attention=dp > 1,
                    nnodes=1,
                    pp_size=1,
                    attn_cp_size=1,
                    dcp_size=1,
                    moe_dp_size=1,
                    dwdp_size=1,
                )
                self.assertFalse(hasattr(cfg, "attn_tp_size"))
                context = published_parallel(cfg)
                for name in reads:
                    self.assertTrue(hasattr(context, name), name)
                self.assertEqual(context.attn_tp_size, attn_tp)
                self.assertEqual(context.attn_dp_size, dp)
                self.assertFalse(context.dcp_enabled)
                RUNTIME["_PARALLEL"] = context
                self.assertIs(RUNTIME["get_parallel"](), context)
                self.assertEqual(POLICY(RUNTIME["get_parallel"]()), supported)

    def test_preserves_tp16_and_admits_910b_dp2tp4(self):
        self.assertTrue(POLICY(parallel()))
        self.assertTrue(
            POLICY(
                parallel(
                    dp_size=2,
                    attn_dp_size=2,
                    attn_tp_size=8,
                    enable_dp_attention=True,
                )
            )
        )
        self.assertTrue(
            POLICY(
                parallel(
                    dp_size=4,
                    attn_dp_size=4,
                    attn_tp_size=4,
                    enable_dp_attention=True,
                )
            )
        )
        cfg = server_args_shape(
            dp_size=2,
            tp_size=8,
            ep_size=8,
            enable_dp_attention=True,
            nnodes=1,
            pp_size=1,
            attn_cp_size=1,
            dcp_size=1,
            moe_dp_size=1,
            dwdp_size=1,
        )
        context = published_parallel(cfg)
        self.assertEqual((context.attn_dp_size, context.attn_tp_size), (2, 4))
        self.assertTrue(POLICY(context))

    def test_rejects_unvalidated_or_relaxed_parallel_layouts(self):
        mutations = (
            dict(dp_size=8, attn_dp_size=8, attn_tp_size=2, enable_dp_attention=True),
            dict(
                dp_size=4, attn_dp_size=4, attn_tp_size=2,
                enable_dp_attention=True, tp_size=8, ep_size=8,
            ),
            dict(
                dp_size=2, attn_dp_size=2, attn_tp_size=4,
                enable_dp_attention=True, tp_size=8, ep_size=16,
            ),
            dict(
                dp_size=2, attn_dp_size=2, attn_tp_size=4,
                enable_dp_attention=True, tp_size=8, ep_size=8, nnodes=2,
            ),
            dict(dp_size=2, attn_dp_size=1, attn_tp_size=8, enable_dp_attention=True),
            dict(dp_size=2, attn_dp_size=2, attn_tp_size=8),
            dict(
                dp_size=2,
                attn_dp_size=2,
                attn_tp_size=8,
                enable_dp_attention=True,
                tp_size=8,
            ),
            dict(
                dp_size=2,
                attn_dp_size=2,
                attn_tp_size=8,
                enable_dp_attention=True,
                ep_size=8,
            ),
            dict(
                dp_size=2,
                attn_dp_size=2,
                attn_tp_size=8,
                enable_dp_attention=True,
                nnodes=2,
            ),
            dict(
                dp_size=2,
                attn_dp_size=2,
                attn_tp_size=8,
                enable_dp_attention=True,
                moe_dp_size=2,
            ),
            dict(
                dp_size=2,
                attn_dp_size=2,
                attn_tp_size=8,
                enable_dp_attention=True,
                dwdp_size=2,
            ),
            dict(
                dp_size=2,
                attn_dp_size=2,
                attn_tp_size=8,
                enable_dp_attention=True,
                attn_cp_size=2,
            ),
            dict(
                dp_size=2,
                attn_dp_size=2,
                attn_tp_size=8,
                enable_dp_attention=True,
                pp_size=2,
            ),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.assertFalse(POLICY(parallel(**mutation)))

    def test_build_gate_keeps_existing_scope_and_calls_bounded_policy(self):
        build = next(
            node
            for node in ast.parse(SOURCE).body
            if isinstance(node, ast.FunctionDef) and node.name == "build_kv_cache"
        )
        text = ast.unparse(build)
        for required in (
            "not get_parallel().dcp_enabled",
            "get_parallel().attn_cp_size == 1",
            "get_parallel().pp_size == 1",
            "get_disagg().disaggregation_mode == 'prefill'",
            "get_exec().mamba.enable_mamba_extra_buffer",
            "not get_exec().mamba.enable_mamba_extra_buffer_lazy",
            "_glm53_npu_prefill_prefix_topology_supported(get_parallel())",
        ):
            with self.subTest(required=required):
                self.assertIn(required, text)

    def test_page_size_helper_still_requires_glm_npu_and_physical_page64(self):
        glm = NS(index_kpool=4)
        self.assertEqual(PAGE_SIZE(glm, "npu:0", 64, 64, True), 256)
        self.assertEqual(PAGE_SIZE(glm, "cpu", 64, 64, True), 64)
        self.assertEqual(PAGE_SIZE(None, "npu", 64, 64, True), 64)
        self.assertEqual(PAGE_SIZE(glm, "npu", 64, 64, False), 64)
        with self.assertRaisesRegex(ValueError, "physical page64"):
            PAGE_SIZE(glm, "npu", 128, 64, True)


if __name__ == "__main__":
    unittest.main()
