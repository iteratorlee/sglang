"""INT8 expert GEMMs with FP32 scales and clamped (non-OAI) SwiGLU."""

import os

import torch

from sglang.srt.hardware_backend.npu.moe.grouped_dequant import dequantize_grouped_int32
from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
    NPUW8A8Int8MoEMethod,
)
from sglang.srt.hardware_backend.npu.utils import npu_format_cast


class NPUW8A8ClampedMoEMethod(NPUW8A8Int8MoEMethod):
    def __init__(self, limit: float):
        super().__init__()
        self.limit = float(limit)
        if not self.limit > 0:
            raise ValueError("SwiGLU clamp limit must be positive")

    def process_weights_after_loading(self, layer, weight_prefix):
        weight = getattr(layer, f"{weight_prefix}_weight")
        scale = getattr(layer, f"{weight_prefix}_weight_scale")
        offset = getattr(layer, f"{weight_prefix}_weight_offset", None)
        if weight.dtype != torch.int8:
            raise ValueError("Clamped W8A8 experts require checkpoint INT8 weights")
        if offset is not None and torch.count_nonzero(offset).item():
            raise ValueError("Clamped W8A8 experts require symmetric weights")
        if not torch.isfinite(scale).all().item() or not (scale > 0).all().item():
            raise ValueError("W8A8 channel scales must be finite and positive")
        scale_data = scale.data.squeeze(-1).float()
        if weight_prefix == "w13":
            # The CANN clamped SwiGLU op consumes alternating gate/up lanes.
            # Permute once before NZ conversion, preserving ND tensor metadata.
            half = weight.shape[1] // 2
            idx = torch.arange(half, device=weight.device)
            order = torch.stack((idx, idx + half), dim=1).flatten()
            weight.data = weight.data.index_select(1, order)
            scale_data = scale_data.index_select(1, order)
            self._set_dispatcher_output_dtype(layer, "int8")
            if os.getenv("SGLANG_GLM53_NORMAL_HCCL", "0") == "1":
                from sglang.srt.runtime_context import get_server_args

                args = get_server_args()
                dispatchers = getattr(layer.dispatcher, "_inners", [layer.dispatcher])
                if not (
                    args.tp_size == args.ep_size == 16
                    and args.nnodes == args.pp_size == 1
                    and not args.enable_dp_attention
                    and not args.enable_two_batch_overlap
                    and not args.enable_eplb
                    and args.ep_num_redundant_experts == 0
                    and args.quantization == "modelslim"
                    and self.limit == 10.0
                    and args.moe_a2a_backend == "deepep"
                    and all(hasattr(d, "_normal_dispatcher") for d in dispatchers)
                ):
                    raise ValueError(
                        "GLM HCCL normal routing requires the TP16/EP16 ModelSlim deployment"
                    )
                layer.dispatcher.set_quant_config(
                    {"dispatcher_output_dtype": "int8", "glm53_normal_hccl": True}
                )
        setattr(
            layer,
            f"{weight_prefix}_weight_scale",
            torch.nn.Parameter(scale_data.contiguous(), requires_grad=False),
        )
        weight.data = npu_format_cast(weight.data.transpose(1, 2).contiguous())

    @staticmethod
    def _counts(expert_tokens, group_list_type):
        if group_list_type == 1:
            return expert_tokens
        if group_list_type == 0:
            return torch.diff(expert_tokens, prepend=expert_tokens.new_zeros(1))
        raise ValueError(f"Unsupported grouped token layout: {group_list_type}")

    @staticmethod
    def _int8_gmm(x, weight, counts):
        if x.dtype != torch.int8 or weight.dtype != torch.int8:
            raise RuntimeError("W8A8 expert computation must remain INT8 x INT8")
        return torch.ops.npu.npu_grouped_matmul(
            x=[x],
            weight=[weight],
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=counts,
            output_dtype=torch.int32,
        )[0]

    def apply_fused_gmm1_swiglu(
        self, quant_info, hidden_states, expert_tokens, pertoken_scale, group_list_type
    ):
        counts = self._counts(expert_tokens, group_list_type)
        if pertoken_scale is None:
            hidden_states, pertoken_scale = self.hidden_states_quantizer(hidden_states)
        accum = self._int8_gmm(hidden_states, quant_info.w13_weight, counts)
        return torch.ops.npu.npu_dequant_swiglu_quant(
            accum,
            weight_scale=quant_info.w13_weight_scale,
            activation_scale=pertoken_scale,
            group_index=counts,
            activate_left=True,
            quant_mode=1,
            swiglu_mode=1,
            clamp_limit=self.limit,
            glu_alpha=1.0,
            glu_bias=0.0,
        )

    def apply(
        self,
        quant_info,
        hidden_states,
        expert_tokens,
        pertoken_scale,
        output_dtype,
        weight_prefix,
        group_list_type,
    ):
        if weight_prefix != "w2":
            raise RuntimeError("Clamped W8A8 gate/up requires fused SwiGLU")
        counts = self._counts(expert_tokens, group_list_type)
        if pertoken_scale is None:
            hidden_states, pertoken_scale = self.hidden_states_quantizer(hidden_states)
        accum = self._int8_gmm(hidden_states, quant_info.w2_weight, counts)
        return dequantize_grouped_int32(
            accum, quant_info.w2_weight_scale, pertoken_scale, counts, output_dtype
        )
