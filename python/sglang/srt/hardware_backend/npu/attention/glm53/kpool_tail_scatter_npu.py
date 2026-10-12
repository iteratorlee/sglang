"""Copy GLM-5.3 KPool tail keys and scores with one graph-safe launch."""

import torch
import triton
import triton.language as tl


@triton.jit
def _scatter_kpool_tail_kernel(
    tail_k_ptr,
    tail_score_ptr,
    row_ptr,
    key_ptr,
    score_ptr,
    width: tl.constexpr,
    block: tl.constexpr,
):
    src_row = tl.program_id(0).to(tl.int64)
    dst_row = tl.load(row_ptr + src_row).to(tl.int64)
    col = tl.program_id(1) * block + tl.arange(0, block)
    mask = col < width
    src_offset = src_row * width + col
    dst_offset = dst_row * width + col
    # Keep the two storage dtypes: no arithmetic or precision conversion.
    key = tl.load(key_ptr + src_offset, mask=mask)
    score = tl.load(score_ptr + src_offset, mask=mask)
    tl.store(tail_k_ptr + dst_offset, key, mask=mask)
    tl.store(tail_score_ptr + dst_offset, score, mask=mask)


def scatter_kpool_tail_(tail_k, tail_score, rows, key, score):
    """Replace the two tail ``scatter_rows_`` calls without changing values.

    Destination buffers retain their addresses, and row indices are loaded
    from device memory on each graph replay. Real request rows must be unique,
    as in the separate scatters; duplicate padding rows remain scratch only.
    """
    if tail_k.shape != tail_score.shape or not all(
        tensor.is_contiguous() for tensor in (tail_k, tail_score)
    ):
        raise ValueError("KPool tail buffers must have matching contiguous layouts")
    if (
        tail_k.dtype != torch.bfloat16
        or key.dtype != torch.bfloat16
        or tail_score.dtype != torch.float32
        or score.dtype != torch.float32
    ):
        raise ValueError("KPool tails require BF16 keys and FP32 scores")
    key = key.reshape(-1, key.shape[-1]).contiguous()
    score = score.reshape(-1, score.shape[-1]).contiguous()
    rows = rows.reshape(-1).contiguous()
    if (
        key.shape != score.shape
        or key.shape[-1] != tail_k.shape[-1]
        or key.shape[0] != rows.numel()
        or rows.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("KPool tail source shape or row layout mismatch")
    if rows.numel():
        block = 128
        _scatter_kpool_tail_kernel[(rows.numel(), triton.cdiv(key.shape[-1], block))](
            tail_k.view(-1),
            tail_score.view(-1),
            rows,
            key,
            score,
            width=key.shape[-1],
            block=block,
        )
