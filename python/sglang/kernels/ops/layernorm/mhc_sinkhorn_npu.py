"""Opt-in large-P 4x4 Sinkhorn, consuming native FP32 softmax + epsilon.

No projection, RMS, sigmoid, softmax, pre mixing or dtype conversion here.
The caller owns the P-only guard; invalid direct calls fail explicitly.
"""

import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def _hc_sinkhorn_4x4(
    COMB,
    OUT,
    TOKENS: tl.constexpr,
    ITERS: tl.constexpr,
    EPS: tl.constexpr,
    BT: tl.constexpr,
):
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    i = tl.arange(0, 4)
    offsets = t[:, None, None] * 16 + i[None, :, None] * 4 + i[None, None, :]
    mask = t[:, None, None] < TOKENS
    # Padded tokens are independent of valid tokens in both reductions.
    c = tl.load(COMB + offsets, mask, other=0.0).to(tl.float32)
    c = c / (tl.sum(c, axis=1)[:, None, :] + EPS)
    for _ in range(ITERS - 1):
        c = c / (tl.sum(c, axis=2)[:, :, None] + EPS)
        c = c / (tl.sum(c, axis=1)[:, None, :] + EPS)
    tl.store(OUT + offsets, c, mask)


def hc_sinkhorn_large_prefill(comb, sinkhorn_iters, eps):
    """Normalize contiguous physical ND FP32 [T,4,4]; return new FP32 storage.

    Direct tests may use 0 <= T <= 16384 (including tails). Production dispatch
    additionally requires the existing 4096 < T <= 16384 large-P pre-mix guard.
    Like the native loop, ITERS <= 1 performs only the initial column normalize.
    No tensor values are read back to the host and no input is modified.
    """
    if comb.ndim != 3 or comb.shape[1:] != (4, 4):
        raise ValueError("mHC Sinkhorn requires [T,4,4]")
    if not 0 <= comb.shape[0] <= 16384:
        raise ValueError("mHC Sinkhorn requires 0 <= T <= 16384")
    if comb.device.type != "npu" or comb.dtype != torch.float32:
        raise ValueError("mHC Sinkhorn requires NPU FP32 input")
    if not comb.is_contiguous() or torch_npu.get_npu_format(comb) != 2:
        raise ValueError("mHC Sinkhorn requires contiguous physical ND(2)")
    if not isinstance(sinkhorn_iters, int):
        raise TypeError("sinkhorn_iters must be an integer")
    output = torch.empty_like(comb)
    tokens = comb.shape[0]
    if tokens:
        _hc_sinkhorn_4x4[(triton.cdiv(tokens, 32),)](
            comb,
            output,
            TOKENS=tokens,
            ITERS=sinkhorn_iters,
            EPS=eps,
            BT=32,
            num_warps=1,
            enable_fp_fusion=False,
            enable_auto_bind_sub_block=False,
        )
    return output
