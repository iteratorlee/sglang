"""Execute P4 topology policies and EP32 routing on CPU, without SGLang imports.

HCCL and quantization are mocks; the real routing, inverse-map kernel body,
FP32 combine, shape certificate and checkpoint selection execute on CPU.
This verifies contracts, not accelerator lowering, liveness or accuracy.
"""

import ast
import enum
import os
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
from unittest import mock

import torch

import test_glm53_normal_idle_dp_cpu as idle
import test_glm53_device_ordered_indices_cpu as inverse
import test_glm53_normal_ordered_2node_cpu as ordered
from test_glm53_normal_hccl_cpu import SRT, dynamic_quant_reference, load_source

sys.path.insert(0, str(Path(__file__).parent / "ascend"))
from test_glm53_topology_policy_static import POLICY
from test_glm53_2node_pd_checkpoint_cpu import CheckpointCase, END
from glm53_topology_test_schema import published_parallel, server_args_shape

FOUR = "SGLANG_GLM53_4NODE_PD"


def four_node(fixture):
    for key, value in dict(nnodes=4, tp_size=32, ep_size=32, dp_size=4).items():
        setattr(fixture.args, key, value)
    p = fixture.parallel
    p.attn_dp_size = 4
    p.moe_ep_group.world_size = 32
    p.moe_ep_group.ranks = list(range(32))
    p.tp_group.ranks = list(range(32))


def batch(counts=(24, 0, 8, 16)):
    fb = idle.batch(counts)
    fb.global_num_tokens_cpu = [max(fb.global_num_tokens_cpu)] * len(counts)
    fb.can_run_tbo = False
    return fb


class FourNodeTopologyCPU(unittest.TestCase):
    def setUp(self):
        self.fixture = idle.IdleDpCpuTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        four_node(self.fixture)
        self.module = self.fixture.module
        self.group = self.fixture.group
        mock.patch.dict(os.environ, {
            FOUR: "1",
            "SGLANG_GLM53_NORMAL_COMBINE_ORDERED": "2node",
            "SGLANG_GLM53_NORMAL_DEVICE_ROUTING": "ordered",
            "SGLANG_GLM53_NORMAL_DEVICE_ROUTING_MIN_TOKENS": "0",
        }).start()

    def test_explicit_opt_in_and_finite_profile(self):
        profile = self.module.normal_hccl_profile
        args = self.fixture.args
        self.assertEqual(profile(args, 10), "prefill_4node_dp4_tp8_ep32")
        self.assertTrue(self.module.normal_hccl_skip_idle_dp_enabled())
        with mock.patch.dict(os.environ, {FOUR: "0"}):
            self.assertIsNone(profile(args, 10))  # Old 2NODE flag remains enabled.
            self.assertFalse(self.module.normal_hccl_skip_idle_dp_enabled())
            with self.module.normal_hccl_idle_dp_scope(batch()):
                self.assertIsNone(self.module._idle_dp_counts.get())
                self.assertFalse(self.module._normal_ordered_combine_enabled(
                    32, 1024, 8, 4096, torch.bfloat16
                ))
        with mock.patch.multiple(args, nnodes=2, tp_size=16, ep_size=16, dp_size=2), \
             mock.patch.dict(os.environ, {"SGLANG_GLM53_2NODE_PD": "0"}):
            self.assertIsNone(profile(args, 10))  # FOUR alone never admits two nodes.
        for key, value in (
            ("nnodes", 3), ("tp_size", 16), ("ep_size", 16), ("dp_size", 8),
            ("pp_size", 2), ("attn_cp_size", 2), ("dcp_size", 2),
            ("moe_dp_size", 2), ("dwdp_size", 2), ("enable_dp_attention", False),
            ("enable_eplb", True), ("ep_num_redundant_experts", 1),
            ("disaggregation_mode", "decode"), ("quantization", "fp8"),
            ("enable_two_batch_overlap", True), ("disable_overlap_schedule", False),
        ):
            with self.subTest(key=key), mock.patch.object(args, key, value):
                self.assertIsNone(profile(args, 10))
        for key, value in (("disable_cuda_graph", False),
                           ("enable_single_batch_overlap", True),
                           ("deepep_mode", "low_latency")):
            with mock.patch.object(args, key, value):
                self.assertFalse(self.module.normal_hccl_skip_idle_dp_enabled())

    def test_four_domain_metadata_validates_every_capacity(self):
        for counts in ((24, 8, 16, 0), (0, 0, 0, 24), (0, 0, 0, 0)):
            with self.module.normal_hccl_idle_dp_scope(batch(counts)):
                self.assertEqual(self.module._idle_dp_counts.get(), counts)
        for name in ("original_global_num_tokens_cpu", "global_num_tokens_cpu"):
            for bad in ([24, 0], [24, 0, 0], [24, 0, 0, True], [24, 24, 24, -1]):
                fb = batch()
                setattr(fb, name, bad)
                with self.assertRaises(ValueError):
                    self.module._normal_dp_shape_metadata(fb)
        for bad in ([24, 24, 16, 24], [24, 24, 24, 16]):
            fb = batch()
            fb.global_num_tokens_cpu = bad
            with self.assertRaisesRegex(ValueError, "MAX_LEN"):
                self.module._normal_dp_shape_metadata(fb)

    def test_max_len_for_unequal_positive_four_domain_chunks(self):
        p = self.fixture.parallel
        for k, v in vars(self.fixture.args).items():
            setattr(p, k, v)
        p.moe_ep_size = 32
        runtime = sys.modules["sglang.srt.runtime_context"]
        runtime.get_disagg = lambda: NS(disaggregation_mode="prefill")
        utils = ModuleType("sglang.srt.layers.moe.utils")
        utils.get_moe_a2a_backend = lambda: NS(is_pplx=lambda: False, is_deepep_v2=lambda: False)
        cls = next(n for n in ast.parse((SRT / "layers/dp_attention.py").read_text()).body
                   if isinstance(n, ast.ClassDef) and n.name == "DpPaddingMode")
        ns = dict(IntEnum=enum.IntEnum, auto=enum.auto, List=list,
                  get_attention_dp_size=lambda: 4,
                  get_flags=lambda: NS(dp=NS(max_len_with_idle=True)))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls], type_ignores=[])), "padding", "exec"), ns)
        mode = ns["DpPaddingMode"]
        with mock.patch.dict(sys.modules, {utils.__name__: utils}):
            self.assertEqual(mode.get_dp_padding_mode(True, [8192, 8, 16, 24]), mode.MAX_LEN)
            with mock.patch.dict(os.environ, {FOUR: "0"}):
                self.assertEqual(mode.get_dp_padding_mode(True, [8192, 8, 16, 24]), mode.SUM_LEN)

    def test_cache_index_and_checkpoint_policies_use_host_local_tp8(self):
        cfg = server_args_shape(**{k: getattr(self.fixture.args, k) for k in (
            "nnodes", "tp_size", "ep_size", "dp_size", "pp_size", "attn_cp_size",
            "dcp_size", "moe_dp_size", "dwdp_size", "enable_dp_attention", "enable_prefill_cp")})
        parallel = published_parallel(cfg)
        self.assertEqual((parallel.attn_dp_size, parallel.attn_tp_size), (4, 8))
        index_ns = {}
        tree = ast.parse((SRT / "hardware_backend/npu/attention/glm53/prefill_index_parallel.py").read_text())
        nodes = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name == "_pd_prefill_topology_supported") or
                 (isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "_VALIDATED_PD_DP_ATTN_TOPOLOGIES")]
        index_ns["os"] = os
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "index-policy", "exec"), index_ns)
        index_policy = index_ns["_pd_prefill_topology_supported"]
        self.assertTrue(POLICY(parallel))
        self.assertTrue(index_policy(cfg))
        with mock.patch.dict(os.environ, {FOUR: "0"}):
            self.assertFalse(POLICY(parallel))
            self.assertFalse(index_policy(cfg))
        for page in (64, 256):
            case = CheckpointCase(page=page)
            for k, v in dict(nnodes=4, tp_size=32, ep_size=32, dp_size=4).items():
                case.set_context("parallel." + k, v)
            entry = case.run()
            self.assertEqual(entry.track_seqlen, END - page + 1)
            self.assertEqual(case.req.kv.mamba_last_track_seqlen, END - page)

    def test_idle_attention_covers_all_four_domains_and_excludes_draft(self):
        fn = sys.modules["sglang.srt.hardware_backend.npu.moe.glm53_idle_attention"].skip_idle_attention
        with mock.patch.dict(os.environ, {"SGLANG_GLM53_NORMAL_SKIP_IDLE_ATTN": "1"}):
            for rank in range(4):
                self.fixture.parallel.attn_dp_rank = rank
                self.assertEqual(fn(batch((24, 0, 0, 8)), is_nextn=False, local_rows=24), rank in (1, 2))
                self.assertFalse(fn(batch((24, 0, 0, 8)), is_nextn=True, local_rows=24))

    def test_ordered_rows_bound_and_cross_world_scope_rejection(self):
        enabled = self.module._normal_ordered_combine_enabled
        with self.module.normal_hccl_idle_dp_scope(batch()):
            self.assertTrue(enabled(32, 4095, 8, 4096, torch.bfloat16))
            with self.assertRaisesRegex(ValueError, "gathered rows"):
                enabled(32, 4096, 8, 4096, torch.bfloat16)
            self.assertFalse(enabled(16, 4095, 8, 4096, torch.bfloat16))

    def test_ep32_dispatch_covers_288_experts_and_all_idle_masks(self):
        tokens = 2
        all_ids = (torch.arange(32 * tokens * 8).reshape(-1, 8) * 17) % 288
        # Duplicates, invalid IDs, and both ends of each 9-expert slice.
        all_ids[0] = torch.tensor([0, 8, 9, 17, 279, 287, -1, 2**62])
        all_x = (torch.arange(32 * tokens * 4096).reshape(-1, 4096) % 17).to(torch.bfloat16)
        all_q, all_scales = dynamic_quant_reference(all_x)
        gates = torch.full(all_ids.shape, 0.125)
        kernel = ModuleType(inverse.KERNEL_NAME)
        kernel.device_ordered_indices = inverse.emulate_indices
        def gather(out, value, group):
            self.assertIs(group, self.group)
            source = {torch.int8: all_q, torch.int64: all_ids}[value.dtype] if value.dtype != torch.float32 else (gates if value.ndim == 2 else all_scales)
            out.copy_(source)
        flat = all_ids.flatten()
        with mock.patch.dict(sys.modules, {kernel.__name__: kernel}), \
             mock.patch.object(self.module, "_check_equal_dispatch_shapes_cached"), \
             mock.patch.object(self.module.dist, "get_world_size", return_value=32), \
             mock.patch.object(self.module.dist, "all_gather_into_tensor", side_effect=gather), \
             mock.patch.object(torch.ops.npu, "npu_dynamic_quant", side_effect=dynamic_quant_reference, create=True):
            for mask in range(16):
                counts = tuple(16 if mask & (1 << dp) else 0 for dp in range(4))
                fb = batch(counts)
                # A zero-token forward has no dispatch; exercise routing's empty result separately.
                fb.global_num_tokens_cpu = [16] * 4
                with self.module.normal_hccl_idle_dp_scope(fb):
                    for rank in range(32):
                        p = self.fixture.parallel
                        p.moe_ep_group.rank_in_group = rank
                        p.attn_dp_rank, p.attn_tp_rank = divmod(rank, 8)
                        low = 9 * rank
                        selected = [i for i, expert in enumerate(flat.tolist())
                                    if low <= expert < low + 9 and counts[i // (8 * tokens * 8)]]
                        selected.sort(key=lambda i: int(flat[i]))
                        selected = torch.tensor(selected, dtype=torch.int64)
                        with mock.patch.object(self.module.dist, "get_rank", return_value=rank):
                            recv, scale, expert_counts, state = self.module.collective_dispatch(
                                self.group, all_x[rank*tokens:(rank+1)*tokens], all_ids[rank*tokens:(rank+1)*tokens], gates[:tokens])
                        expected_counts = torch.bincount(flat[selected] - low, minlength=9)
                        torch.testing.assert_close(expert_counts, expected_counts, rtol=0, atol=0)
                        torch.testing.assert_close(state[0], selected // 8, rtol=0, atol=0)
                        expected_map = self.module._normal_combine_ordered_indices(selected // 8, 64, 8)
                        torch.testing.assert_close(state[6], expected_map, rtol=0, atol=0)
                        if len(selected):
                            torch.testing.assert_close(recv, all_q[selected // 8], rtol=0, atol=0)
                            torch.testing.assert_close(scale, all_scales[selected // 8], rtol=0, atol=0)
                        else:
                            self.assertEqual(recv.shape, (1, 4096))

    def test_device_inverse_wrapper_accepts_exact_nine_expert_slices(self):
        dev = NS(type="npu")
        def tensor(shape, dtype):
            return NS(shape=shape, ndim=len(shape), dtype=dtype, device=dev,
                      is_contiguous=lambda: True, numel=lambda: __import__("math").prod(shape))
        launch = mock.MagicMock()
        ns = dict(torch=NS(int32=torch.int32, int64=torch.int64, full=mock.Mock()),
                  triton=NS(cdiv=lambda n, d: (n+d-1)//d), _ordered_indices=launch)
        exec(inverse.extract("device_ordered_indices"), ns)
        fn = ns["device_ordered_indices"]
        for rank in range(32):
            fn(tensor((2, 8), torch.int64), tensor((1,), torch.int64), rank*9, 9)
        self.assertEqual(launch.__getitem__.return_value.call_count, 32)
        for low, experts in ((1, 9), (288, 9), (-9, 9), (0, 8), (0, 0)):
            with self.assertRaises(ValueError):
                fn(tensor((2, 8), torch.int64), tensor((1,), torch.int64), low, experts)

    def test_ep32_reduce_scatter_keeps_fp32_and_rank_blocks(self):
        kernel_name = self.module.__name__.rsplit(".", 1)[0] + ".glm53_ordered_combine"
        kernel = ModuleType(kernel_name)
        kernel.ordered_combine = lambda y, weights, indices, rows, width, **kw: ordered.emulate_kernel(y, weights, indices, rows, width)
        rows = torch.arange(64)
        weights = torch.full((64,), 0.125)
        values = (rows[:, None].expand(64, 4096) % 13).to(torch.bfloat16)
        mapping = self.module._normal_combine_ordered_indices(rows, 64, 8)
        partial = values.float() * weights[:, None]
        state = (rows, weights, 2, 32, 4096, 64, mapping)
        with self.module.normal_hccl_idle_dp_scope(batch()), \
             mock.patch.dict(os.environ, {"SGLANG_GLM53_NORMAL_REDUCE_SCATTER": "2node"}), \
             mock.patch.dict(sys.modules, {kernel_name: kernel}), \
             mock.patch.object(self.module.dist, "all_reduce") as all_reduce:
            for rank in range(32):
                def scatter(out, source, op, group):
                    self.assertEqual(source.dtype, torch.float32)
                    torch.testing.assert_close(source, partial, rtol=0, atol=0)
                    out.copy_(source[rank*2:(rank+1)*2] * 32)
                with mock.patch.object(self.module.dist, "reduce_scatter_tensor", side_effect=scatter) as rs:
                    actual = self.module.collective_combine(self.group, values, state)
                rs.assert_called_once()
                torch.testing.assert_close(actual, (partial[rank*2:(rank+1)*2]*32).to(torch.bfloat16), rtol=0, atol=0)
            all_reduce.assert_not_called()

    def test_shape_certificate_requires_32_ranks_and_one_vote_per_forward(self):
        path = SRT.parents[2] / "test/registered/unit/hardware_backend/npu/test_glm53_batch_shape_audit_cpu.py"
        harness = load_source("four_node_shape_harness", path).BatchShapeAuditCPU()
        harness.setUp()
        self.addCleanup(harness.doCleanups)
        four_node(harness.cache.fixture)
        fb = batch((4096, 8, 0, 64))
        for _ in range(2):
            with harness.scope(fb):
                for layer in range(3, 45):
                    harness.layer(fb, layer, world=32)
        self.assertEqual(harness.events(), ["vote", "audit", "vote"])
        with self.assertRaisesRegex(RuntimeError, "world"):
            with harness.scope(fb):
                harness.layer(fb, 3, world=16)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
