"""Update the existing GLM-5.3 decode graph block table in one launch."""

import logging
import os

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)
_checked_updates = 0


@triton.jit(do_not_specialize=["bs", "max_seq_pages"])
def _update_block_table_kernel(
    req_to_token,
    req_pool_indices,
    block_tables,
    bs,
    max_seq_pages,
    REQ_ROWS: tl.constexpr,
    REQ_ROW_STRIDE: tl.constexpr,
    REQ_COL_STRIDE: tl.constexpr,
    INDEX_STRIDE: tl.constexpr,
    TABLE_ROW_STRIDE: tl.constexpr,
    TABLE_COL_STRIDE: tl.constexpr,
    TABLE_COLS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    active = (row < bs) & (col < max_seq_pages) & (col < TABLE_COLS)
    req = tl.load(
        req_pool_indices + row * INDEX_STRIDE,
        mask=(row < bs) & (tl.program_id(1) * BLOCK < max_seq_pages),
        other=0,
    ).to(tl.int64)
    # Preserve PyTorch's negative indexing and read row 0 normally. Padded
    # requests inside a graph bucket use row 0, whose contents may be nonzero.
    req = tl.where(req < 0, req + REQ_ROWS, req)
    token = tl.load(
        req_to_token
        + req * REQ_ROW_STRIDE
        + col.to(tl.int64) * PAGE_SIZE * REQ_COL_STRIDE,
        mask=active,
        other=0,
    )
    # Triton integer division truncates; torch // floors, including sentinels.
    page = token // PAGE_SIZE
    page -= ((token < 0) & (token % PAGE_SIZE != 0)).to(page.dtype)
    tl.store(
        block_tables + row * TABLE_ROW_STRIDE + col.to(tl.int64) * TABLE_COL_STRIDE,
        page,
        mask=col < TABLE_COLS,
    )


def try_update_block_table_(
    req_to_token, req_pool_indices, block_tables, bs, max_seq_pages, page_size
):
    """Return False without writing if the original torch path is needed.

    The request pool and static graph table are int32; request IDs are int64
    (int32 is also supported). Only host tensor metadata is inspected here:
    there is no allocation, packing, device readback or synchronization.

    Fill the supplied view, not its max_bs backing allocation. Rows >= bs
    and columns >= max_seq_pages become zero, while IDs of zero within bs
    still read the pool's scratch row. Request IDs must be valid pool indices,
    just as for the original advanced indexing. bs/max_seq_pages are runtime
    kernel scalars so changing lengths does not specialize a new kernel.
    """
    if (
        req_to_token.ndim != 2
        or req_pool_indices.ndim != 1
        or block_tables.ndim != 2
        or req_to_token.dtype != torch.int32
        or block_tables.dtype != torch.int32
        or req_pool_indices.dtype not in (torch.int32, torch.int64)
        or req_to_token.device != block_tables.device
        or req_pool_indices.device != block_tables.device
        or page_size <= 0
        or not 0 <= bs <= min(block_tables.shape[0], req_pool_indices.numel())
        or not 0 <= max_seq_pages <= block_tables.shape[1]
        or max_seq_pages > triton.cdiv(req_to_token.shape[1], page_size)
        or block_tables.stride(1) <= 0
        or block_tables.stride(0) < block_tables.shape[1] * block_tables.stride(1)
    ):
        return False
    if block_tables.numel():
        block = 256
        _update_block_table_kernel[
            (block_tables.shape[0], triton.cdiv(block_tables.shape[1], block))
        ](
            req_to_token,
            req_pool_indices,
            block_tables,
            bs,
            max_seq_pages,
            REQ_ROWS=req_to_token.shape[0],
            REQ_ROW_STRIDE=req_to_token.stride(0),
            REQ_COL_STRIDE=req_to_token.stride(1),
            INDEX_STRIDE=req_pool_indices.stride(0),
            TABLE_ROW_STRIDE=block_tables.stride(0),
            TABLE_COL_STRIDE=block_tables.stride(1),
            TABLE_COLS=block_tables.shape[1],
            PAGE_SIZE=page_size,
            BLOCK=block,
        )
        # Diagnostic only: compare live scheduler mappings before graph replay.
        # This allocates and synchronizes, so it must be off for performance runs.
        if os.getenv("SGLANG_GLM53_CHECK_BLOCK_TABLE_UPDATE", "0") == "1":
            expected = torch.zeros_like(block_tables)
            expected[:bs, :max_seq_pages].copy_(
                req_to_token[
                    req_pool_indices[:bs], 0 : max_seq_pages * page_size : page_size
                ]
                // page_size
            )
            if not torch.equal(block_tables, expected):
                raise RuntimeError("GLM53 fused graph block table differs from torch")
            global _checked_updates
            _checked_updates += 1
            if _checked_updates in (1, 10, 100) or _checked_updates % 1000 == 0:
                logger.info(
                    "GLM53 live block table equals torch: updates=%d bs=%d pages=%d",
                    _checked_updates,
                    bs,
                    max_seq_pages,
                )
    return True
