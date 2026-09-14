"""Expand compressed KPool indices and append causal tail slots in one kernel."""

import torch
import triton
import triton.language as tl


@triton.jit
def _expand(
    POOLS,
    POS,
    OUT,
    POOLS_PER_ROW: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    COL_STRIDE: tl.constexpr,
    POS_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    width = POOLS_PER_ROW * 4
    blocks = tl.cdiv(width + 3, BLOCK)
    row = tl.program_id(0) // blocks
    col = (tl.program_id(0) % blocks) * BLOCK + tl.arange(0, BLOCK)
    seq = tl.load(POS + row * POS_STRIDE).to(tl.int32)
    seq = seq + 1
    tail_count = seq & 3
    tail_start = seq - tail_count
    history = tl.minimum(tail_start, width)
    pool = tl.load(
        POOLS + row * ROW_STRIDE + (col >> 2) * COL_STRIDE, col < width, -1
    ).to(tl.int32)
    expanded = tl.where((col < width) & (pool >= 0), pool * 4 + (col & 3), -1)
    tail_index = col - history
    tail = tl.where(tail_index < tail_count, tail_start + tail_index, -1)
    result = tl.where((tail_index >= 0) & (tail_index < 3), tail, expanded)
    result = tl.where(result >= 0, result, seq)
    tl.store(OUT + row * (width + 3) + col, result, col < width + 3)


def expand_with_tail(pool_indices, positions):
    if (
        pool_indices.ndim != 2
        or positions.ndim != 1
        or pool_indices.shape[0] != positions.numel()
        or pool_indices.dtype not in (torch.int32, torch.int64)
        or positions.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("Invalid KPool expansion metadata")
    # GLM uses at most 1M context tokens. Keep arithmetic in int32; vector
    # int64 promotion makes this integer-only operation much slower on Ascend.
    bs, pools = pool_indices.shape
    out = torch.empty(
        (bs, pools * 4 + 3), device=pool_indices.device, dtype=pool_indices.dtype
    )
    if bs:
        _expand[(bs * triton.cdiv(pools * 4 + 3, 512),)](
            pool_indices,
            positions,
            out,
            POOLS_PER_ROW=pools,
            ROW_STRIDE=pool_indices.stride(0),
            COL_STRIDE=pool_indices.stride(1),
            POS_STRIDE=positions.stride(0),
            BLOCK=512,
            enable_auto_bind_sub_block=False,
        )
    return out
