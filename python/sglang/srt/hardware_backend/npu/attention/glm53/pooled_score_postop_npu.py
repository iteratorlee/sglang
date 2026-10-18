"""Opt-in P prefill in-place ReLU/gate only; default disabled by the caller.

Preserve native CANN matmul, BF16 storage, native head sum and query tiling.
Only ND physical format 2 is admitted. No reduction, threshold or approximation.
"""

import os

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["N"])
def _relu_gate_inplace(S, W, N, BLOCK: tl.constexpr):
    qhead = tl.program_id(0)
    key = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(S + qhead * N + key, mask=key < N, other=0).to(tl.float32)
    gate = tl.load(W + qhead).to(tl.float32)
    # Measured CANN BF16 ReLU maps input -0 to +0 before gate multiplication.
    # Ordered <= canonicalizes both zeros but propagates NaN; multiplication
    # then produces the zero sign required by the FP32 gate. Do not use max,
    # whose NaN behavior need not match CANN.
    value = tl.where(value <= 0.0, 0.0, value)
    result = (value * gate).to(tl.bfloat16)
    tl.store(S + qhead * N + key, result, mask=key < N)


def supported(scores, weights):
    if not (
        scores.device.type == "npu"
        and weights.device == scores.device
        and scores.dtype == torch.bfloat16
        and weights.dtype == torch.float32
        and scores.ndim == 3
        and 0 < scores.shape[0] <= 32
        and (
            scores.shape[1] == 32
            or (
                scores.shape[1] == 64
                and os.environ.get("SGLANG_GLM53_PREFILL_POOLED_POSTOP64", "0") == "1"
            )
        )
        and weights.ndim == 2
        and tuple(weights.shape) == tuple(scores.shape[:2])
        and scores.is_contiguous()
        and weights.is_contiguous()
    ):
        return False
    import torch_npu

    try:
        return (
            torch_npu.get_npu_format(scores) == torch_npu.get_npu_format(weights) == 2
        )
    except (AttributeError, RuntimeError):
        return False


def relu_gate_(scores, weights, block=8192):
    """Modify private scores in place. Return True only if the kernel executed."""
    if block not in (2048, 4096, 8192):
        raise ValueError("Unvalidated block: choose 2048, 4096 or 8192 for microtests")
    if not supported(scores, weights):
        scores.relu_()
        scores.mul_(weights.unsqueeze(-1))
        return False
    rows, heads, count = scores.shape
    if not rows or not count:
        return False
    if rows * heads * triton.cdiv(count, block) > 65535:
        # The Ascend launch ABI limits the logical grid, even for vector ops.
        # This is a shape fallback, with the native operations executed once.
        scores.relu_()
        scores.mul_(weights.unsqueeze(-1))
        return False
    _relu_gate_inplace[(rows * heads, triton.cdiv(count, block))](
        scores, weights, count, block, num_warps=4, multibuffer=False
    )
    return True
