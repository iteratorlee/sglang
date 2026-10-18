"""Experimental row-grouped GLM gated RMSNorm; no numerical rewrite.

The 128-column reduction and pointwise expressions match the original row
kernel. Changing the layout can still change compiler lowering: target NPU
bitwise acceptance is required before enabling this candidate.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _glm_rms_norm_gated_rows_kernel(
    x,
    gate,
    weight,
    output,
    eps,
    num_rows: tl.constexpr,
    rows_per_program: tl.constexpr,
):
    rows = tl.program_id(0) * rows_per_program + tl.arange(0, rows_per_program)
    cols = tl.arange(0, 128)
    offsets = rows[:, None] * 128 + cols[None, :]
    mask = rows[:, None] < num_rows
    values = tl.load(x + offsets, mask=mask, other=0.0).to(tl.float32)
    gates = tl.load(gate + offsets, mask=mask, other=0.0).to(tl.float32)
    weights = tl.load(weight + cols).to(tl.float32)
    variance = tl.sum(values * values, axis=1) / 128
    normalized = values / tl.sqrt(variance + eps)[:, None]
    result = normalized * weights[None, :] / (1.0 + tl.exp(-gates))
    tl.store(output + offsets, result, mask=mask)


def supported(x, gate, weight):
    return (
        x.ndim >= 2
        and x.shape[-1] == 128
        and x.numel() == gate.numel()
        and weight.numel() == 128
        and x.dtype == gate.dtype == torch.bfloat16
        and weight.dtype in (torch.bfloat16, torch.float32)
        and x.device == gate.device == weight.device
        and all(t.is_contiguous() for t in (x, gate, weight))
    )


def supported_prefill_loop(x, gate, weight, output):
    """Only P16384 x 8 heads, contiguous BF16/ND, for the old 1D loop."""
    shapes = ((16384, 8, 128), (1, 16384, 8, 128))
    if not (
        supported(x, gate, weight)
        and x.device.type == "npu"
        and tuple(x.shape) in shapes
        and tuple(gate.shape) in shapes
        and weight.shape == (128,)
        and output.shape == x.shape
        and output.dtype == torch.bfloat16
        and output.device == x.device
        and output.is_contiguous()
    ):
        return False
    import torch_npu

    try:
        return all(torch_npu.get_npu_format(t) == 2 for t in (x, gate, weight, output))
    except (AttributeError, RuntimeError):
        return False


def rms_norm_gated_rows(x, gate, weight, output, eps, rows_per_program):
    """Write contiguous BF16 rows; allow short shapes for isolated validation."""
    if (
        rows_per_program not in (4, 8)
        or not supported(x, gate, weight)
        or output.shape != x.shape
        or output.dtype != x.dtype
        or output.dtype != torch.bfloat16
        or output.device != x.device
        or not output.is_contiguous()
    ):
        raise ValueError("Unsupported GLM gated RMSNorm row-group layout")
    num_rows = x.numel() // 128
    if num_rows:
        _glm_rms_norm_gated_rows_kernel[(triton.cdiv(num_rows, rows_per_program),)](
            x,
            gate,
            weight,
            output,
            eps,
            num_rows=num_rows,
            rows_per_program=rows_per_program,
            num_warps=1,
            num_stages=1,
        )
    return output
