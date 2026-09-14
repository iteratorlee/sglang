"""Integer-only KPool decode addressing, with dynamic graph inputs."""

import torch
import triton
import triton.language as tl


@triton.jit
def _decode_metadata(
    REQ,
    SEQ,
    POS,
    TABLE,
    R,
    TAIL,
    LOC,
    CLOSE,
    LENS,
    Q,
    PAGES,
    TABLE_STRIDE: tl.constexpr,
    NCOLS: tl.constexpr,
    PCOLS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    if block == 0:
        req = tl.load(REQ + row)
        seq = tl.load(SEQ + row)
        pos = tl.load(POS + row)
        valid = (req > 0) & (seq > 0)
        req = tl.where(valid, req, 0)
        slot = pos % 4
        pool = pos // 4
        page = tl.maximum(tl.load(TABLE + row * TABLE_STRIDE + (pool // 64) * 4), 0)
        closing = valid & (slot == 3)
        tl.store(R + row, req)
        tl.store(TAIL + row, req * 4 + slot)
        tl.store(LOC + row, tl.where(closing, page * 64 + pool % 64, 0))
        tl.store(CLOSE + row, closing)
        tl.store(LENS + row, seq // 4)
        tl.store(Q + row, row + 1)
    c = block * BLOCK + tl.arange(0, BLOCK)
    page = tl.load(TABLE + row * TABLE_STRIDE + c * 4, c < PCOLS, 0)
    tl.store(PAGES + row * PCOLS + c, tl.maximum(page, 0), c < PCOLS)


def decode_metadata(req, seq, positions, table):
    """Match the kpool=4 integer expressions without host synchronization."""
    bs = req.numel()
    if (
        seq.numel() != bs
        or positions.numel() != bs
        or table.shape[0] != bs
        or table.stride(1) != 1
        or not all(x.is_contiguous() for x in (req, seq, positions))
    ):
        raise ValueError("Invalid KPool decode metadata layout")
    dev = req.device
    request = torch.empty(bs, dtype=torch.int64, device=dev)
    tail = torch.empty_like(request)
    loc = torch.empty_like(request)
    closing = torch.empty(bs, dtype=torch.bool, device=dev)
    lens = torch.empty(bs, dtype=torch.int32, device=dev)
    actual_q = torch.empty_like(lens)
    columns = triton.cdiv(table.shape[1], 4)
    pages = torch.empty((bs, columns), dtype=table.dtype, device=dev)
    if bs:
        _decode_metadata[(bs, triton.cdiv(columns, 128))](
            req,
            seq,
            positions,
            table,
            request,
            tail,
            loc,
            closing,
            lens,
            actual_q,
            pages,
            TABLE_STRIDE=table.stride(0),
            NCOLS=table.shape[1],
            PCOLS=columns,
            BLOCK=128,
            enable_auto_bind_sub_block=False,
        )
    return request, tail, loc, closing, lens, actual_q, pages
