"""GLM-5.3 gated RMSNorm with a graph-safe Ascend row kernel."""

from __future__ import annotations


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
    feature_dim = x.shape[-1]
    block_dim = triton.next_power_of_2(feature_dim)
    output = torch.empty_like(x)
    _glm_rms_norm_gated_row_kernel[(x.numel() // feature_dim,)](
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
    return output
