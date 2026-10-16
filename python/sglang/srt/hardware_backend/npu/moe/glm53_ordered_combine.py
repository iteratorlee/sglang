"""Deterministic, bounded-memory local GLM normal-phase expert reduction.

One output tile owns each destination. Products and the fixed expert-order
accumulator are FP32, with no atomic adds or process-wide determinism changes.
The HCCL reduction and final BF16 conversion remain in the caller.
The kernel is independent of EP size: the caller scopes EP8 mode 1 and the
opt-in two-node P EP16 mode, both with the same row/width/topk bounds.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _ordered_combine(Y, W, INDEX, OUT, ROWS, WIDTH: tl.constexpr,
                     TOPK: tl.constexpr, BLOCK_ROWS: tl.constexpr):
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    channel = tl.arange(0, WIDTH)
    acc = tl.full((BLOCK_ROWS, WIDTH), 0, tl.float32)
    for slot in tl.static_range(TOPK):
        index = tl.load(INDEX + row * TOPK + slot, row < ROWS, other=-1)
        valid = (row < ROWS) & (index >= 0)
        safe_index = tl.maximum(index, 0)
        value = tl.load(Y + safe_index[:, None] * WIDTH + channel[None, :],
                        valid[:, None], other=0).to(tl.float32)
        weight = tl.load(W + safe_index, valid, other=0).to(tl.float32)
        # fp32 product rounds before accumulation, like the existing expression.
        product = value * weight[:, None]
        acc = acc + product
    tl.store(OUT + row[:, None] * WIDTH + channel[None, :], acc, (row < ROWS)[:, None])


def ordered_combine(y, weights, indices, rows, width):
    if not (y.device.type == "npu" and y.dtype == torch.bfloat16
            and weights.device == indices.device == y.device
            and weights.dtype == torch.float32 and indices.dtype == torch.int32
            and y.is_contiguous() and weights.is_contiguous() and indices.is_contiguous()
            and y.ndim == 2 and weights.ndim == 1 and weights.numel() == y.shape[0]
            and y.shape[1] == width == 4096 and 0 <= rows <= 131070
            and tuple(indices.shape) == (rows, 8)):
        raise ValueError("GLM ordered combine requires contiguous BF16 H4096, FP32 gates, INT32 [rows,8] map")
    # A rank with no selected experts has empty input buffers. Keep the
    # original zero-contribution path; never pass null pointers to Triton.
    if not y.shape[0]:
        return torch.zeros((rows, width), device=y.device, dtype=torch.float32)
    out = torch.empty((rows, width), device=y.device, dtype=torch.float32)
    if rows:
        _ordered_combine[(triton.cdiv(rows, 2),)](
            y, weights, indices, out, rows, width, 8, 2,
            num_warps=4, multibuffer=False, enable_fp_fusion=False,
        )
    return out
