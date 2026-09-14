"""Small Ascend graph-safe data movement kernels used by model patches.

Some CANN versions capture the *value* of the index tensor passed to
``npu_scatter_nd_update_``.  Decode replay then keeps writing the first token's
cache slot.  Triton loads the index from device memory on every replay, which
preserves SGLang's static-address graph contract while keeping locations
dynamic.
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _scatter_rows_kernel(
    dst_ptr,
    src_ptr,
    row_ptr,
    width: tl.constexpr,
    block: tl.constexpr,
):
    src_row = tl.program_id(0)
    width_block = tl.program_id(1)
    dst_row = tl.load(row_ptr + src_row).to(tl.int64)
    col = width_block * block + tl.arange(0, block)
    mask = col < width
    value = tl.load(src_ptr + src_row.to(tl.int64) * width + col, mask=mask)
    tl.store(dst_ptr + dst_row * width + col, value, mask=mask)


def scatter_rows_(dst, rows, src):
    """Write ``src[i]`` into flattened row ``rows[i]`` of ``dst``.

    The final dimension is the row width; all leading destination dimensions
    are flattened.  The operation is intentionally in-place because cache
    buffers keep a stable address across graph capture/replay.
    """

    src_2d = src.reshape(-1, src.shape[-1]).contiguous()
    rows_1d = rows.reshape(-1).contiguous()
    if src_2d.shape[0] != rows_1d.numel():
        raise ValueError(
            f"scatter row count mismatch: {src_2d.shape[0]} != {rows_1d.numel()}"
        )
    width = src_2d.shape[1]
    block = 128
    _scatter_rows_kernel[(src_2d.shape[0], triton.cdiv(width, block))](
        dst.reshape(-1),
        src_2d,
        rows_1d,
        width=width,
        block=block,
    )
    return dst
