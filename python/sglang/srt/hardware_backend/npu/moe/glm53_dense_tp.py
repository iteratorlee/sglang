"""Finite, opt-in dense TP16 within GLM P TP32; no backend imports."""

import os


def dense_tp16_enabled(args, *, is_npu):
    if args.moe_dense_tp_size in (None, 1, args.tp_size):
        return False
    # Worker bootstrap receives the raw record, while CLI validation already
    # reads a view. Graph resolution declares a typed config without changing
    # the raw input; use the same declaration-aware view in both stages.
    from sglang.srt.arg_groups.model_override_base import resolving_view

    args = resolving_view(args)
    expected = dict(
        moe_dense_tp_size=16,
        tp_size=32,
        ep_size=32,
        dp_size=4,
        nnodes=4,
        pp_size=1,
        attn_cp_size=1,
        dcp_size=1,
        moe_dp_size=1,
        dwdp_size=1,
        enable_dp_attention=True,
        disaggregation_mode="prefill",
        quantization="modelslim",
        moe_a2a_backend="deepep",
        deepep_mode="normal",
        disable_overlap_schedule=True,
        disable_cuda_graph=True,
        enable_two_batch_overlap=False,
        enable_single_batch_overlap=False,
        enable_eplb=False,
        ep_num_redundant_experts=0,
        enable_prefill_cp=False,
        enable_layernorm_sp=False,
        enable_dense_mlp_attn_tp=False,
        enable_symm_mem=False,
        enable_torch_symm_mem=False,
        enable_quant_communications=False,
        enable_torch_compile=False,
        enable_pdmux=False,
        elastic_ep_backend=None,
        ep_join_mode=None,
        ep_join_rank_offset=0,
        weight_cache_mode="off",
    )
    invalid = [
        name
        for name, value in expected.items()
        if getattr(args, name, object()) != value
    ]
    for name in (
        "SGLANG_GLM53_DENSE_TP16",
        "SGLANG_GLM53_4NODE_PD",
        "SGLANG_GLM53_NORMAL_HCCL",
    ):
        if os.getenv(name, "0") != "1":
            invalid.append(name)
    graph = getattr(args, "cuda_graph_config", None)
    if any(
        getattr(getattr(graph, phase, None), "backend", None) != "disabled"
        for phase in ("prefill", "decode")
    ):
        invalid.append("cuda_graph_config (both phases must be disabled)")
    if getattr(args, "load_format", None) not in ("auto", "safetensors"):
        invalid.append("load_format")
    if getattr(args, "custom_weight_loader", object()) not in (None, []):
        invalid.append("custom_weight_loader")
    if not is_npu or invalid:
        raise ValueError(
            "dense TP16 requires finite eager Ascend P4 profile: "
            + ", ".join(invalid or ["NPU backend"])
        )
    return True


def validate_dense_tp16_model(model_config):
    if model_config.hf_config.architectures != ["Glm5NextForConditionalGeneration"]:
        raise ValueError("dense TP16 requires the GLM5Next target architecture")
    config = model_config.hf_text_config
    expected = dict(
        hidden_size=4096,
        intermediate_size=12288,
        first_k_dense_replace=3,
        moe_layer_freq=1,
        mhc=True,
        hidden_act="silu",
        swiglu_limit=10.0,
    )
    if any(getattr(config, name, None) != value for name, value in expected.items()):
        raise ValueError("dense TP16 requires the GLM H4096/I12288 dense prefix")
