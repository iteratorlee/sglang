"""Copy both KPool verify tails directly into a per-step snapshot.

Only addresses are calculated here. Keys and scores are copied as integer
bits, without changing compression, acceptance, or rollback arithmetic.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _snapshot_kpool_tails(
    KEYS, SCORES, REQUESTS, SNAPSHOT_KEYS, SNAPSHOT_SCORES,
    ROW_ELEMENTS: tl.constexpr, STEPS: tl.constexpr, STEP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    request = tl.load(REQUESTS + row).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = col < ROW_ELEMENTS
    source = request * ROW_ELEMENTS + col
    destination = (row * STEPS + STEP) * ROW_ELEMENTS + col
    # Integer views preserve NaN payloads, signed zeros and subnormals too.
    keys = tl.load(KEYS + source, valid, 0)
    scores = tl.load(SCORES + source, valid, 0)
    tl.store(SNAPSHOT_KEYS + destination, keys, valid)
    tl.store(SNAPSHOT_SCORES + destination, scores, valid)


def snapshot_kpool_tails_(keys, scores, requests, snapshot_keys, snapshot_scores, step):
    """Equivalent to two ``index_select(0, requests)`` followed by copies.

    Requests are live device indices and must be valid source rows, including
    reserved padding row zero. Duplicate requests are legal: each batch row
    writes to its own snapshot. Other rows and steps remain untouched.
    """
    tensors = (keys, scores, requests, snapshot_keys, snapshot_scores)
    if (
        keys.ndim != 3
        or snapshot_keys.ndim != 4
        or keys.shape != scores.shape
        or snapshot_keys.shape != snapshot_scores.shape
        or keys.shape[1:] != snapshot_keys.shape[2:]
        or requests.ndim != 1
        or requests.numel() > snapshot_keys.shape[0]
        or keys.dtype != torch.bfloat16
        or snapshot_keys.dtype != keys.dtype
        or scores.dtype != torch.float32
        or snapshot_scores.dtype != scores.dtype
        or requests.dtype not in (torch.int32, torch.int64)
        or not all(t.is_contiguous() for t in tensors)
        or not all(t.device == keys.device for t in tensors)
        or not isinstance(step, int)
        or not 0 <= step < snapshot_keys.shape[1]
    ):
        raise ValueError("Invalid KPool snapshot layout, dtype, device or step")
    if requests.numel() == 0:
        return
    row_elements = keys.shape[1] * keys.shape[2]
    block = min(512, triton.next_power_of_2(row_elements))
    _snapshot_kpool_tails[(requests.numel(), triton.cdiv(row_elements, block))](
        # Metadata-only integer aliases: BiSheng cannot lower an in-kernel
        # pointer reinterpret cast at this store. No numeric cast or copy.
        keys.view(torch.int16), scores.view(torch.int32), requests,
        snapshot_keys.view(torch.int16), snapshot_scores.view(torch.int32),
        ROW_ELEMENTS=row_elements, STEPS=snapshot_keys.shape[1], STEP=step,
        BLOCK=block, enable_auto_bind_sub_block=False,
    )
