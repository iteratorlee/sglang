"""Recompute compact index addresses from live request slots during graph replay."""

import torch
import triton
import triton.language as tl


@triton.jit
def _request_block_table(
    REQ,
    OUT,
    COLS: tl.constexpr,
    NREQ: tl.constexpr,
    STRIDE: tl.constexpr,
    PER_REQ: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    req = tl.load(REQ + row * STRIDE, row < NREQ, other=0).to(tl.int32)
    page = 1 + (req - 1) * PER_REQ + tl.minimum(col // 4, PER_REQ - 1)
    page = tl.where((req > 0) & (req <= CAPACITY), page, 0)
    tl.store(OUT + row * COLS + col, page, col < COLS)


def request_block_table(req, original, layout):
    if req.ndim != 1 or original.ndim != 2:
        raise ValueError("Compact index metadata must have one request dimension")
    rows, cols = original.shape
    output = torch.empty((rows, cols), device=original.device, dtype=torch.int32)
    if rows and cols:
        _request_block_table[(rows, triton.cdiv(cols, 256))](
            req,
            output,
            cols,
            req.numel(),
            req.stride(0),
            layout.pages_per_request,
            layout.requests,
            256,
        )
    return output
