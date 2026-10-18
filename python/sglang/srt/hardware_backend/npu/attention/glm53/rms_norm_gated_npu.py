"""GLM-5.3 gated RMSNorm with a graph-safe Ascend row kernel."""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from sgl_kernel_npu.fla.utils import input_guard


@triton.jit
def _glm_rms_norm_gated_row_kernel(
    x,
    gate,
    weight,
    output,
    eps,
    feature_dim: tl.constexpr,
    block_dim: tl.constexpr,
):
    """Normalize one logical ``[head, dim]`` row per Triton program."""

    row = tl.program_id(0)
    offsets = tl.arange(0, block_dim)
    mask = offsets < feature_dim
    row_offsets = row * feature_dim + offsets
    values = tl.load(x + row_offsets, mask=mask, other=0.0).to(tl.float32)
    gates = tl.load(gate + row_offsets, mask=mask, other=0.0).to(tl.float32)
    weights = tl.load(weight + offsets, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(values * values, axis=0) / feature_dim
    normalized = values / tl.sqrt(variance + eps)
    result = normalized * weights / (1.0 + tl.exp(-gates))
    tl.store(output + row_offsets, result, mask=mask)


@triton.jit
def _glm_rms_norm_gated_large_row_kernel(
    x,
    gate,
    weight,
    output,
    eps,
    feature_dim: tl.constexpr,
    block_dim: tl.constexpr,
    num_rows: tl.constexpr,
    grid_rows: tl.constexpr,
):
    """Cover large inputs without exceeding Ascend's 65535-program grid cap."""

    offsets = tl.arange(0, block_dim)
    mask = offsets < feature_dim
    weights = tl.load(weight + offsets, mask=mask, other=0.0).to(tl.float32)
    for row in range(tl.program_id(0), num_rows, grid_rows):
        row_offsets = row * feature_dim + offsets
        values = tl.load(x + row_offsets, mask=mask, other=0.0).to(tl.float32)
        gates = tl.load(gate + row_offsets, mask=mask, other=0.0).to(tl.float32)
        variance = tl.sum(values * values, axis=0) / feature_dim
        normalized = values / tl.sqrt(variance + eps)
        result = normalized * weights / (1.0 + tl.exp(-gates))
        tl.store(output + row_offsets, result, mask=mask)


@input_guard
def glm_rms_norm_gated_npu(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Apply GLM's ``RMSNorm(x) * sigmoid(gate)`` on contiguous rows."""

    if x.numel() != gate.numel():
        raise ValueError("GLM gated RMSNorm requires x and gate with equal numel")
    if x.shape[-1] != weight.numel():
        raise ValueError("GLM gated RMSNorm weight must match the head dimension")
    # Idle attention-DP domains still run the MoE/EP collective, but have no
    # local attention tokens. Ascend rejects a Triton launch with grid=(0,).
    if x.numel() == 0:
        return torch.empty_like(x)
    feature_dim = x.shape[-1]
    block_dim = triton.next_power_of_2(feature_dim)
    output = torch.empty_like(x)
    num_rows = x.numel() // feature_dim
    loop_rows = os.getenv("SGLANG_GLM53_D_GATED_RMS_LOOP_ROWS", "0")
    prefill_loop_rows = os.getenv("SGLANG_GLM53_P_GATED_RMS_LOOP_ROWS", "0")
    if (
        (
            (loop_rows in ("4", "8") and 1024 <= num_rows <= 2048)
            or (prefill_loop_rows in ("4", "8") and num_rows == 131072)
        )
        and x.device.type == "npu"
        and feature_dim == 128
        and output.dtype == torch.bfloat16
    ):
        from sglang.srt.runtime_context import get_server_args

        mode = get_server_args().disaggregation_mode
        selected_loop_rows = None
        if mode == "decode" and loop_rows in ("4", "8") and 1024 <= num_rows <= 2048:
            raw_min_rows = os.getenv("SGLANG_GLM53_D_GATED_RMS_LOOP_MIN_ROWS", "1024")
            try:
                min_rows = int(raw_min_rows)
                if not 1024 <= min_rows <= 2048:
                    raise ValueError
            except ValueError as exc:
                raise ValueError(
                    "SGLANG_GLM53_D_GATED_RMS_LOOP_MIN_ROWS must be an integer "
                    f"in [1024, 2048], got {raw_min_rows!r}"
                ) from exc
            from .rms_norm_gated_rows_npu import supported

            if num_rows >= min_rows and supported(x, gate, weight):
                selected_loop_rows = int(loop_rows)
        elif (
            mode == "prefill" and prefill_loop_rows in ("4", "8") and num_rows == 131072
        ):
            from .rms_norm_gated_rows_npu import supported_prefill_loop

            if supported_prefill_loop(x, gate, weight, output):
                selected_loop_rows = int(prefill_loop_rows)
        if selected_loop_rows is not None:
            grid_rows = triton.cdiv(num_rows, selected_loop_rows)
            _glm_rms_norm_gated_large_row_kernel[(grid_rows,)](
                x,
                gate,
                weight,
                output,
                eps,
                feature_dim=128,
                block_dim=128,
                num_rows=num_rows,
                grid_rows=grid_rows,
                num_warps=1,
                num_stages=1,
            )
            return output
    grouped_rows = os.getenv("SGLANG_GLM53_D_GATED_RMS_ROWS", "0")
    if (
        grouped_rows in ("4", "8")
        and x.device.type == "npu"
        and feature_dim == 128
        and 1024 <= num_rows <= 2048
        and output.dtype == torch.bfloat16
    ):
        from sglang.srt.runtime_context import get_server_args

        if get_server_args().disaggregation_mode == "decode":
            from .rms_norm_gated_rows_npu import rms_norm_gated_rows, supported

            if supported(x, gate, weight):
                return rms_norm_gated_rows(
                    x, gate, weight, output, eps, int(grouped_rows)
                )
    if num_rows > 65535:
        # A 2048-token GLM prefill chunk has 65536 head rows. Ascend rejects
        # that one-dimensional launch. Keep the small/decode kernel unchanged.
        grid_rows = min(num_rows, 32768)
        kernel = _glm_rms_norm_gated_large_row_kernel
    else:
        grid_rows = num_rows
        kernel = _glm_rms_norm_gated_row_kernel
    kwargs = dict(
        x=x,
        gate=gate,
        weight=weight,
        output=output,
        eps=eps,
        feature_dim=feature_dim,
        block_dim=block_dim,
        num_warps=1,
        num_stages=1,
    )
    if num_rows > 65535:
        kwargs.update(num_rows=num_rows, grid_rows=grid_rows)
    kernel[(grid_rows,)](**kwargs)
    return output
