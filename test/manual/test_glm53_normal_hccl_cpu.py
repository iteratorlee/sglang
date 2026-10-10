"""CPU reference/mocks for the opt-in GLM53 normal HCCL path.

Run: python test/manual/test_glm53_normal_hccl_cpu.py
Imports the complete edited modules, mocking only their NPU/dependency boundary.
No device initialization or distributed process group is required.
"""

import __future__
import ast
import enum
import importlib.util
import os
from contextlib import ExitStack
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest import mock

import torch


REPO = Path(__file__).resolve().parents[2]
SRT = REPO / "python/sglang/srt"
COLLECTIVES_NAME = "sglang.srt.hardware_backend.npu.moe.glm53_collectives"


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def args_for_profile(**overrides):
    values = dict(
        tp_size=8,
        ep_size=8,
        dp_size=4,
        nnodes=1,
        pp_size=1,
        attn_cp_size=1,
        dcp_size=1,
        moe_dp_size=1,
        enable_dp_attention=True,
        enable_prefill_cp=False,
        enable_two_batch_overlap=False,
        enable_eplb=False,
        ep_num_redundant_experts=0,
        quantization="modelslim",
        moe_a2a_backend="deepep",
        disaggregation_mode="prefill",
        disable_overlap_schedule=True,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class FakeDispatcher:
    def __init__(self, normal=True):
        self.config = {}
        if normal:
            self._normal_dispatcher = object()

    def set_quant_config(self, config):
        self.config.update(config)


def layer_weights():
    return SimpleNamespace(
        w13_weight=torch.nn.Parameter(
            torch.arange(16, dtype=torch.int8).view(2, 4, 2), requires_grad=False
        ),
        w2_weight=torch.nn.Parameter(
            torch.arange(8, dtype=torch.int8).view(2, 2, 2), requires_grad=False
        ),
        w13_weight_scale=torch.nn.Parameter(torch.ones(2, 4, 1)),
        w2_weight_scale=torch.nn.Parameter(torch.ones(2, 2, 1)),
        dispatcher=FakeDispatcher(),
    )


def dynamic_quant_reference(x):
    scale = x.float().abs().amax(dim=1).clamp_min(1.0) / 127
    q = (x.float() / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
    return q, scale


def grouped_matmul_reference(**kwargs):
    x, weight, counts = kwargs["x"][0], kwargs["weight"][0], kwargs["group_list"]
    assert x.dtype == weight.dtype == torch.int8
    assert kwargs["output_dtype"] == torch.int32
    assert kwargs["group_list_type"] == 1
    out = torch.zeros(x.shape[0], weight.shape[-1], dtype=torch.int32)
    start = 0
    for expert, count in enumerate(counts):
        count = int(count)
        out[start : start + count] = x[start : start + count].to(torch.int32) @ weight[
            expert
        ].to(torch.int32)
        start += count
    return [out]


class NormalHcclCpuTests(unittest.TestCase):
    def setUp(self):
        self.collectives = load_source(
            COLLECTIVES_NAME, SRT / "hardware_backend/npu/moe/glm53_collectives.py"
        )
        self.args = args_for_profile()
        base_module = ModuleType(
            "sglang.srt.hardware_backend.npu.quantization.moe_methods"
        )

        class BaseMethod:
            @staticmethod
            def _set_dispatcher_output_dtype(layer, dtype):
                layer.dispatcher.set_quant_config({"dispatcher_output_dtype": dtype})

        base_module.NPUW8A8Int8MoEMethod = BaseMethod
        utils = ModuleType("sglang.srt.hardware_backend.npu.utils")
        utils.npu_format_cast = lambda tensor: tensor
        dequant = ModuleType("sglang.srt.hardware_backend.npu.moe.grouped_dequant")
        dequant.dequantize_grouped_int32 = mock.Mock()
        context = ModuleType("sglang.srt.runtime_context")
        context.get_server_args = lambda: self.args
        self.modules = {
            COLLECTIVES_NAME: self.collectives,
            base_module.__name__: base_module,
            utils.__name__: utils,
            dequant.__name__: dequant,
            context.__name__: context,
        }
        with mock.patch.dict(sys.modules, self.modules):
            self.clamped = load_source(
                "test_w8a8_clamped_moe",
                SRT / "hardware_backend/npu/quantization/w8a8_clamped_moe.py",
            )

    def test_finite_profile_guard_and_preserved_tp16(self):
        profile = self.collectives.normal_hccl_profile
        self.assertEqual(profile(self.args, 10), "prefill_dp4_tp2_ep8")
        # Original TP16 guard did not require PD prefill or disabled overlap.
        legacy = args_for_profile(
            tp_size=16,
            ep_size=16,
            enable_dp_attention=False,
            disaggregation_mode="null",
            disable_overlap_schedule=False,
        )
        self.assertEqual(profile(legacy, 10), "tp16_ep16")
        for changes in (
            {"disaggregation_mode": "decode"},
            {"disaggregation_mode": "null"},
            {"dp_size": 2},
            {"dp_size": 8},
            {"enable_dp_attention": False},
            {"tp_size": 16},
            {"ep_size": 16},
            {"nnodes": 2},
            {"pp_size": 2},
            {"attn_cp_size": 2},
            {"dcp_size": 2},
            {"moe_dp_size": 2},
            {"enable_prefill_cp": True},
            {"disable_overlap_schedule": False},
            {"enable_two_batch_overlap": True},
            {"enable_eplb": True},
            {"ep_num_redundant_experts": 1},
            {"quantization": "fp8"},
            {"moe_a2a_backend": "none"},
        ):
            with self.subTest(changes=changes):
                self.assertIsNone(profile(args_for_profile(**changes), 10))
        self.assertIsNone(profile(self.args, 7))

    def test_default_off_and_load_time_profile_guard(self):
        with mock.patch.dict(sys.modules, self.modules), mock.patch.dict(os.environ):
            os.environ.pop("SGLANG_GLM53_NORMAL_HCCL", None)
            self.args = args_for_profile(disaggregation_mode="decode")
            layer = layer_weights()
            self.clamped.NPUW8A8ClampedMoEMethod(10).process_weights_after_loading(
                layer, "w13"
            )
            self.assertNotIn("glm53_normal_hccl", layer.dispatcher.config)
            self.assertEqual(layer.dispatcher.config["dispatcher_output_dtype"], "int8")
            os.environ["SGLANG_GLM53_NORMAL_HCCL"] = "1"
            with self.assertRaisesRegex(ValueError, "P-only"):
                self.clamped.NPUW8A8ClampedMoEMethod(10).process_weights_after_loading(
                    layer_weights(), "w13"
                )
            self.args = args_for_profile()
            layer = layer_weights()
            layer.dispatcher = FakeDispatcher(normal=False)
            with self.assertRaisesRegex(ValueError, "P-only"):
                self.clamped.NPUW8A8ClampedMoEMethod(10).process_weights_after_loading(
                    layer, "w13"
                )

    def test_tp8_prefill_profile_keeps_decode_and_parallel_guards(self):
        args = args_for_profile(enable_dp_attention=False, dp_size=1)
        self.assertEqual(
            self.collectives.normal_hccl_profile(args, 10), "prefill_tp8_ep8"
        )
        for changes in (
            {"disaggregation_mode": "decode"}, {"disaggregation_mode": "null"},
            {"dp_size": 4}, {"enable_dp_attention": True},
            {"disable_overlap_schedule": False}, {"attn_cp_size": 2},
            {"dcp_size": 2}, {"moe_dp_size": 2}, {"nnodes": 2},
            {"enable_two_batch_overlap": True}, {"enable_eplb": True},
        ):
            with self.subTest(changes=changes):
                value = vars(args).copy()
                value.update(changes)
                self.assertIsNone(
                    self.collectives.normal_hccl_profile(SimpleNamespace(**value), 10)
                )
        with mock.patch.dict(sys.modules, self.modules), mock.patch.dict(
            os.environ, {"SGLANG_GLM53_NORMAL_HCCL": "1"}
        ):
            self.args = args
            layer = layer_weights()
            method = self.clamped.NPUW8A8ClampedMoEMethod(10)
            method.process_weights_after_loading(layer, "w13")
            method.process_weights_after_loading(layer, "w2")
            self.assertEqual(method._glm53_normal_hccl_profile, "prefill_tp8_ep8")
            self.assertEqual(layer.w13_weight.dtype, torch.int8)
            self.assertEqual(layer.w2_weight.dtype, torch.int8)
            self.assertEqual(layer.dispatcher.config["dispatcher_output_dtype"], "int8")

    def test_target_and_draft_weights_remain_int8_with_fp32_scales(self):
        with (
            mock.patch.dict(sys.modules, self.modules),
            mock.patch.dict(os.environ, {"SGLANG_GLM53_NORMAL_HCCL": "1"}),
        ):
            for role in ("target", "draft"):
                with self.subTest(role=role):
                    layer = layer_weights()
                    original = layer.w13_weight.detach().clone()
                    method = self.clamped.NPUW8A8ClampedMoEMethod(10)
                    method.process_weights_after_loading(layer, "w13")
                    method.process_weights_after_loading(layer, "w2")
                    self.assertTrue(layer.dispatcher.config["glm53_normal_hccl"])
                    self.assertEqual(
                        method._glm53_normal_hccl_profile, "prefill_dp4_tp2_ep8"
                    )
                    self.assertEqual(layer.w13_weight.dtype, torch.int8)
                    self.assertEqual(layer.w2_weight.dtype, torch.int8)
                    self.assertEqual(layer.w13_weight_scale.dtype, torch.float32)
                    self.assertEqual(layer.w2_weight_scale.dtype, torch.float32)
                    torch.testing.assert_close(
                        layer.w13_weight, original[:, [0, 2, 1, 3]].transpose(1, 2)
                    )
            for prefix in ("w13", "w2"):
                layer = layer_weights()
                setattr(
                    layer,
                    f"{prefix}_weight",
                    torch.nn.Parameter(
                        getattr(layer, f"{prefix}_weight").to(torch.bfloat16),
                        requires_grad=False,
                    ),
                )
                with (
                    self.subTest(prefix=prefix),
                    self.assertRaisesRegex(ValueError, "INT8 weights"),
                ):
                    self.clamped.NPUW8A8ClampedMoEMethod(
                        10
                    ).process_weights_after_loading(layer, prefix)

    def test_actual_prefill_padding_opt_in_and_unchanged_profiles(self):
        # Isolate the existing production enum from its hardware imports.
        tree = ast.parse((SRT / "layers/dp_attention.py").read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "DpPaddingMode"
        )
        namespace = dict(
            IntEnum=enum.IntEnum,
            auto=enum.auto,
            List=list,
            get_attention_dp_size=lambda: parallel.attn_dp_size,
            get_flags=lambda: SimpleNamespace(
                dp=SimpleNamespace(max_len_with_idle=True)
            ),
        )
        backend = SimpleNamespace(is_pplx=lambda: False, is_deepep_v2=lambda: False)
        utils = ModuleType("sglang.srt.layers.moe.utils")
        utils.get_moe_a2a_backend = lambda: backend
        runtime = ModuleType("sglang.srt.runtime_context")
        disagg = SimpleNamespace(disaggregation_mode="prefill")
        runtime.get_disagg = lambda: disagg
        runtime.get_parallel = lambda: parallel

        def make_parallel(**overrides):
            values = dict(
                tp_size=8,
                moe_ep_size=8,
                dp_size=4,
                attn_dp_size=4,
                attn_tp_size=2,
                pp_size=1,
                attn_cp_size=1,
                dcp_size=1,
                nnodes=1,
                enable_dp_attention=True,
            )
            values.update(overrides)
            return SimpleNamespace(**values)

        parallel = make_parallel()
        with (
            mock.patch.dict(
                sys.modules, {utils.__name__: utils, runtime.__name__: runtime}
            ),
            mock.patch.dict(os.environ),
        ):
            exec(
                compile(
                    ast.Module(body=[cls], type_ignores=[]),
                    "dp_padding_enum",
                    "exec",
                    flags=__future__.annotations.compiler_flag,
                ),
                namespace,
            )
            mode = namespace["DpPaddingMode"]
            busy = [1024, 31, 17, 5]
            idle = [1024, 31, 17, 0]
            os.environ.pop("SGLANG_GLM53_NORMAL_HCCL", None)
            self.assertEqual(mode.get_dp_padding_mode(True, busy), mode.SUM_LEN)
            os.environ["SGLANG_GLM53_NORMAL_HCCL"] = "0"
            self.assertEqual(mode.get_dp_padding_mode(True, busy), mode.SUM_LEN)
            self.assertEqual(mode.get_dp_padding_mode(True, idle), mode.MAX_LEN)
            os.environ["SGLANG_GLM53_NORMAL_HCCL"] = "1"
            self.assertEqual(mode.get_dp_padding_mode(True, busy), mode.MAX_LEN)
            self.assertEqual(mode.get_dp_padding_mode(True, idle), mode.MAX_LEN)
            self.assertEqual(mode.get_dp_padding_mode(False, busy), mode.SUM_LEN)
            for role in ("decode", "null"):
                disagg.disaggregation_mode = role
                for flag in ("0", "1"):
                    os.environ["SGLANG_GLM53_NORMAL_HCCL"] = flag
                    self.assertEqual(mode.get_dp_padding_mode(True, busy), mode.SUM_LEN)
                    self.assertEqual(mode.get_dp_padding_mode(True, idle), mode.MAX_LEN)
            disagg.disaggregation_mode = "prefill"
            os.environ["SGLANG_GLM53_NORMAL_HCCL"] = "1"
            for changes in (
                {"nnodes": 2},
                {"tp_size": 16},
                {"moe_ep_size": 16},
                {"dp_size": 2},
                {"attn_dp_size": 2},
                {"attn_tp_size": 4},
                {"enable_dp_attention": False},
                {"pp_size": 2},
                {"attn_cp_size": 2},
                {"dcp_size": 2},
            ):
                with self.subTest(changes=changes):
                    parallel = make_parallel(**changes)
                    self.assertEqual(mode.get_dp_padding_mode(True, busy), mode.SUM_LEN)
                    self.assertEqual(mode.get_dp_padding_mode(True, idle), mode.MAX_LEN)
            parallel = make_parallel(
                tp_size=16,
                moe_ep_size=16,
                dp_size=1,
                attn_dp_size=1,
                attn_tp_size=16,
                enable_dp_attention=False,
            )
            for flag in ("0", "1"):
                os.environ["SGLANG_GLM53_NORMAL_HCCL"] = flag
                self.assertEqual(mode.get_dp_padding_mode(True, [1024]), mode.MAX_LEN)

    def make_peers(self):
        generator = torch.Generator().manual_seed(1009)
        x = [
            torch.randn(4, 3, generator=generator).to(torch.bfloat16) for _ in range(8)
        ]
        ids = [torch.full((4, 3), -1, dtype=torch.int64) for _ in range(8)]
        weights = [torch.full((4, 3), 1000.0) for _ in range(8)]
        for rank in range(7):
            ids[rank][0, :2] = torch.tensor([rank * 2 + 1, rank * 2])
            weights[rank][0, :2] = torch.tensor([0.25, 0.75])
        ids[0][1] = torch.tensor([8, 0, 8])  # repeated valid expert/row too
        weights[0][1] = torch.tensor([0.2, 0.3, 0.5])
        ids[0][3, 0] = 2
        weights[0][3, 0] = 1.0
        return x, ids, weights

    @staticmethod
    def peer_shape(x, ids, weights):
        return torch.tensor(
            [
                x.shape[0],
                x.shape[1],
                ids.ndim,
                *ids.shape,
                weights.ndim,
                *weights.shape,
            ],
            dtype=torch.int64,
        )

    def dispatch_rank(self, rank, peers, shapes=None):
        xs, ids, weights = peers
        quantized = [dynamic_quant_reference(x) for x in xs]
        batches = [
            shapes or [self.peer_shape(x, i, w) for x, i, w in zip(xs, ids, weights)],
            [q for q, _ in quantized],
            [s for _, s in quantized],
            ids,
            weights,
        ]
        call = 0

        def gather(out, local, group):
            nonlocal call
            torch.testing.assert_close(local, batches[call][rank])
            out.copy_(torch.cat(batches[call]))
            call += 1

        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(
                    self.collectives.dist, "get_world_size", return_value=8
                )
            )
            stack.enter_context(
                mock.patch.object(self.collectives.dist, "get_rank", return_value=rank)
            )
            stack.enter_context(
                mock.patch.object(
                    self.collectives.dist, "all_gather_into_tensor", side_effect=gather
                )
            )
            quant = stack.enter_context(
                mock.patch.object(
                    torch.ops.npu,
                    "npu_dynamic_quant",
                    side_effect=dynamic_quant_reference,
                    create=True,
                )
            )
            result = self.collectives.collective_dispatch(
                object(), xs[rank], ids[rank], weights[rank], 16
            )
            self.assertEqual(call, 5)
            self.assertEqual(quant.call_count, 1)
        return result

    def test_unequal_rows_rejected_before_quantization_or_payload(self):
        peers = self.make_peers()
        shapes = [self.peer_shape(x, i, w) for x, i, w in zip(*peers)]
        shapes[3][[0, 3, 6]] = 2

        def gather(out, local, group):
            out.copy_(torch.cat(shapes))

        with (
            mock.patch.object(self.collectives.dist, "get_world_size", return_value=8),
            mock.patch.object(self.collectives.dist, "get_rank", return_value=0),
            mock.patch.object(
                self.collectives.dist, "all_gather_into_tensor", side_effect=gather
            ) as gathered,
            mock.patch.object(torch.ops.npu, "npu_dynamic_quant", create=True) as quant,
        ):
            with self.assertRaisesRegex(RuntimeError, "equal local rows"):
                self.collectives.collective_dispatch(
                    object(), peers[0][0], peers[1][0], peers[2][0], 16
                )
            self.assertEqual(gathered.call_count, 1)
            quant.assert_not_called()

    def test_dispatch_grouped_weights_and_combine_match_reference(self):
        peers = self.make_peers()
        xs, ids, weights = peers
        quantized = [dynamic_quant_reference(x) for x in xs]
        expert_weights = (torch.arange(16 * 9).reshape(16, 3, 3) % 9 - 4).to(torch.int8)
        expected = torch.zeros(32, 3, dtype=torch.float32)
        partials, outputs, states = [], [], []
        for rank in range(8):
            recv, scale, counts, state = self.dispatch_rank(rank, peers)
            entries = sorted(
                [
                    (int(ids[src][row, k]), src * 4 + row, weights[src][row, k])
                    for src in range(8)
                    for row in range(4)
                    for k in range(3)
                    if rank * 2 <= int(ids[src][row, k]) < (rank + 1) * 2
                ],
                key=lambda item: item[0],
            )
            self.assertEqual(sum(counts), len(entries))
            self.assertEqual(state[-1], len(entries))
            self.assertEqual(recv.dtype, torch.int8)
            self.assertEqual(scale.dtype, torch.float32)
            partial = torch.zeros_like(expected)
            for index, (expert, global_row, routing_weight) in enumerate(entries):
                src, row = divmod(global_row, 4)
                torch.testing.assert_close(recv[index], quantized[src][0][row])
                torch.testing.assert_close(scale[index], quantized[src][1][row])
                self.assertEqual(int(state[0][index]), global_row)
                torch.testing.assert_close(state[1][index], routing_weight)
                y = (
                    (
                        quantized[src][0][row].int() @ expert_weights[expert].int()
                    ).float()
                    * quantized[src][1][row]
                ).to(torch.bfloat16)
                partial[global_row] += y.float() * routing_weight
            with mock.patch.object(
                torch.ops.npu,
                "npu_grouped_matmul",
                side_effect=grouped_matmul_reference,
                create=True,
            ):
                accum = self.clamped.NPUW8A8ClampedMoEMethod._int8_gmm(
                    recv, expert_weights[rank * 2 : (rank + 1) * 2], counts
                )
            outputs.append((accum.float() * scale[:, None]).to(torch.bfloat16))
            states.append(state)
            partials.append(partial)
            expected += partial
        self.assertEqual(states[7][-1], 0)
        self.assertEqual(outputs[7].shape, (1, 3))
        self.assertTrue(torch.equal(expected.view(8, 4, 3)[:, 2], torch.zeros(8, 3)))
        for reduce_scatter in (False, True):
            for rank in range(8):

                def all_reduce(out, group):
                    self.assertEqual(out.dtype, torch.float32)
                    torch.testing.assert_close(out, partials[rank])
                    out.copy_(expected)

                def scatter(out, source, op, group):
                    self.assertEqual(source.dtype, torch.float32)
                    torch.testing.assert_close(source, partials[rank])
                    out.copy_(expected[rank * 4 : (rank + 1) * 4])

                with (
                    mock.patch.dict(os.environ),
                    mock.patch.object(
                        self.collectives.dist, "get_rank", return_value=rank
                    ),
                    mock.patch.object(
                        self.collectives.dist, "all_reduce", side_effect=all_reduce
                    ) as reduced,
                    mock.patch.object(
                        self.collectives.dist,
                        "reduce_scatter_tensor",
                        side_effect=scatter,
                    ) as scattered,
                ):
                    os.environ.pop("SGLANG_GLM53_NORMAL_REDUCE_SCATTER", None)
                    if reduce_scatter:
                        os.environ["SGLANG_GLM53_NORMAL_REDUCE_SCATTER"] = "1"
                    result = self.collectives.collective_combine(
                        object(), outputs[rank], states[rank]
                    )
                    torch.testing.assert_close(
                        result, expected[rank * 4 : (rank + 1) * 4].to(torch.bfloat16)
                    )
                    self.assertEqual(reduced.call_count, int(not reduce_scatter))
                    self.assertEqual(scattered.call_count, int(reduce_scatter))

    def test_combine_accumulates_repeated_rows_in_fp32(self):
        state = (torch.zeros(3, dtype=torch.int64), torch.ones(3), 1, 1, 1, 3)
        y = torch.tensor([[256], [1], [-256]], dtype=torch.bfloat16)
        with (
            mock.patch.dict(os.environ, {"SGLANG_GLM53_NORMAL_REDUCE_SCATTER": "0"}),
            mock.patch.object(self.collectives.dist, "all_reduce") as reduced,
            mock.patch.object(self.collectives.dist, "get_rank", return_value=0),
        ):
            out = self.collectives.collective_combine(object(), y, state)
            self.assertEqual(out.item(), 1)
            self.assertEqual(reduced.call_args.args[0].dtype, torch.float32)

    def test_runtime_activation_once_and_int8_gmm_enforcement(self):
        info = layer_weights()
        info.w13_weight = torch.nn.Parameter(
            torch.ones(2, 4, 3, dtype=torch.int8), requires_grad=False
        )
        info.w2_weight = torch.nn.Parameter(
            torch.ones(2, 3, 2, dtype=torch.int8), requires_grad=False
        )
        info.w2_weight_scale = torch.nn.Parameter(torch.ones(2, 3, 1))
        method = self.clamped.NPUW8A8ClampedMoEMethod(10)
        with (
            mock.patch.dict(sys.modules, self.modules),
            mock.patch.dict(os.environ, {"SGLANG_GLM53_NORMAL_HCCL": "1"}),
            mock.patch.object(self.collectives.logger, "info") as logged,
            mock.patch.object(
                torch.ops.npu,
                "npu_grouped_matmul",
                side_effect=grouped_matmul_reference,
                create=True,
            ),
            mock.patch.object(
                torch.ops.npu, "npu_dequant_swiglu_quant", create=True
            ) as swiglu,
        ):
            method.process_weights_after_loading(info, "w13")
            method.process_weights_after_loading(info, "w2")
            # Loading/low-latency work alone must not report normal activation.
            self.collectives.log_normal_hccl_activation(
                info, method._glm53_normal_hccl_profile
            )
            logged.assert_not_called()
            recv, scale, counts, _ = self.dispatch_rank(0, self.make_peers())
            for _ in range(2):
                method.apply_fused_gmm1_swiglu(info, recv, counts, scale, 1)
            self.assertEqual(logged.call_count, 1)
            self.assertEqual(
                swiglu.call_args.kwargs["activation_scale"].dtype, torch.float32
            )
            self.assertEqual(swiglu.call_args.kwargs["clamp_limit"], 10)
            text = logged.call_args.args[0] % logged.call_args.args[1:]
            for field in (
                "rank=0",
                "world=8",
                "local_rows=4",
                "torch.bfloat16",
                "torch.int8",
                "torch.float32",
                "w13=",
                "w2=",
            ):
                self.assertIn(field, text)
        for x_dtype, w_dtype in (
            (torch.bfloat16, torch.int8),
            (torch.int8, torch.bfloat16),
        ):
            with (
                self.subTest(x_dtype=x_dtype, w_dtype=w_dtype),
                self.assertRaisesRegex(RuntimeError, "INT8 x INT8"),
            ):
                self.clamped.NPUW8A8ClampedMoEMethod._int8_gmm(
                    torch.ones(1, 2, dtype=x_dtype),
                    torch.ones(1, 2, 2, dtype=w_dtype),
                    [1],
                )


if __name__ == "__main__":
    unittest.main()
