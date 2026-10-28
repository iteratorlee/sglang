"""Device inverse map for stable GLM expert routing, without host readback.

Internal contract: selected contains *all* in-range expert routes for each
retained token, sorted stably by expert ID. The dispatcher may drop entire
idle-DP token rows, but cannot drop individual valid routes within a row.
Each original top-k offset occurs at most once (selected comes from nonzero).

For one token, stable expert order is lexicographic (expert ID, top-k slot).
Counting the at most eight preceding local-expert slots therefore identifies
exactly the destination used by the CPU argsort/bincount/scatter reference.
Every store has a unique destination; no atomic or FP operation is involved.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _ordered_indices(
    IDS,
    SELECTED,
    OUT,
    N,
    LOW,
    EXPERTS: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < N
    # The bounded wrapper admits fewer than 2**20 flattened route offsets.
    loc = tl.load(SELECTED + index, valid, other=0).to(tl.int32)
    row = loc // TOPK
    original_slot = loc % TOPK
    expert = tl.load(IDS + loc, valid, other=-1)
    slot = tl.arange(0, TOPK)
    peer = tl.load(IDS + row[:, None] * TOPK + slot[None, :], valid[:, None], other=-1)
    local = (peer >= LOW) & (peer < LOW + EXPERTS)
    earlier = (peer < expert[:, None]) | (
        (peer == expert[:, None]) & (slot[None, :] < original_slot[:, None])
    )
    destination = tl.sum((local & earlier).to(tl.int32), axis=1)
    tl.store(OUT + row * TOPK + destination, index, valid)


def device_ordered_indices(ids, selected, low, experts):
    """Build INT32 [gathered rows,8] indices for the scoped EP16/EP32 dispatcher.

    Metadata checks do not synchronize. Value/order/uniqueness follow from the
    dispatcher's stable selection; this is not a general arbitrary-route API.
    The standalone parity harness checks that contract against the CPU reference.
    """
    if not (
        ids.device.type == "npu"
        and ids.device == selected.device
        and ids.dtype in (torch.int32, torch.int64)
        and selected.dtype == torch.int64
        and ids.ndim == 2
        and ids.shape[1] == 8
        and 0 <= ids.shape[0] <= 131070
        and selected.ndim == 1
        and selected.numel() <= ids.numel()
        and ids.is_contiguous()
        and selected.is_contiguous()
        and experts in (9, 18)
        and 0 <= low <= 288 - experts
        and low % experts == 0
    ):
        raise ValueError(
            "GLM device ordered indices require NPU contiguous integer [rows,8] "
            "IDs, INT64 selected offsets, <=131070 rows and 9 or 18 local experts"
        )
    out = torch.full(ids.shape, -1, dtype=torch.int32, device=ids.device)
    if selected.numel():
        _ordered_indices[(triton.cdiv(selected.numel(), 256),)](
            ids,
            selected,
            out,
            selected.numel(),
            low,
            experts,
            8,
            256,
            num_warps=4,
            multibuffer=False,
        )
    return out
