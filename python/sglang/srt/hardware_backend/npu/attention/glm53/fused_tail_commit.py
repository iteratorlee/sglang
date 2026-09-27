"""Commit GLM-5.3 speculative KPool key and score tails in one launch.

The pointer table is built once before graph capture.  Replay loads request
and accepted-step indices from device tensors, so changing the batch does not
freeze addresses or accepted steps in the captured graph.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _commit_kernel(
    dst_keys,
    src_keys,
    dst_scores,
    src_scores,
    request_indices,
    accepted_steps,
    ROW_ELEMENTS: tl.constexpr,
    NUM_STEPS: tl.constexpr,
    NUM_LAYERS: tl.constexpr,
    TAIL_GRID: tl.constexpr,
    LOGICAL_GRID: tl.constexpr,
    PHYSICAL_GRID: tl.constexpr,
    BLOCK: tl.constexpr,
):
    physical_pid = tl.program_id(0)
    for logical_pid in range(physical_pid, LOGICAL_GRID, PHYSICAL_GRID):
        tail_pid = logical_pid % TAIL_GRID
        layer_pid = (logical_pid // TAIL_GRID) % NUM_LAYERS
        request_pid = logical_pid // (TAIL_GRID * NUM_LAYERS)
        request_idx = tl.load(request_indices + request_pid).to(tl.int64)
        step_idx = tl.load(accepted_steps + request_pid).to(tl.int64)
        offset = tail_pid * BLOCK + tl.arange(0, BLOCK)
        valid = (request_idx >= 0) & (step_idx >= 0) & (offset < ROW_ELEMENTS)
        src_offset = (request_pid * NUM_STEPS + step_idx) * ROW_ELEMENTS + offset
        dst_offset = request_idx * ROW_ELEMENTS + offset

        dst_key = tl.load(dst_keys + layer_pid).to(tl.pointer_type(tl.bfloat16))
        src_key = tl.load(src_keys + layer_pid).to(tl.pointer_type(tl.bfloat16))
        dst_score = tl.load(dst_scores + layer_pid).to(tl.pointer_type(tl.float32))
        src_score = tl.load(src_scores + layer_pid).to(tl.pointer_type(tl.float32))
        key = tl.load(src_key + src_offset, mask=valid, other=0)
        score = tl.load(src_score + src_offset, mask=valid, other=0)
        tl.store(dst_key + dst_offset, key, mask=valid)
        tl.store(dst_score + dst_offset, score, mask=valid)


def make_pointer_tables(indexers):
    """Validate the contiguous layout and make stable device pointer tables."""
    if not indexers:
        raise ValueError("At least one KPool indexer is required")
    first = indexers[0]
    shape = tuple(first._kpool_tail_k.shape)
    source_shape = tuple(first._kpool_mtp_tail_k.shape)
    if source_shape[2:] != shape[1:]:
        raise ValueError("KPool source and destination tail shapes differ")
    for indexer in indexers:
        for suffix, dtype in (("k", torch.bfloat16), ("score", torch.float32)):
            dst = getattr(indexer, "_kpool_tail_" + suffix)
            src = getattr(indexer, "_kpool_mtp_tail_" + suffix)
            if (
                dst.dtype != dtype
                or src.dtype != dtype
                or tuple(dst.shape) != shape
                or tuple(src.shape) != source_shape
                or not dst.is_contiguous()
                or not src.is_contiguous()
            ):
                raise ValueError("Fused KPool commit requires uniform contiguous tails")
    device = first._kpool_tail_k.device
    tables = []
    for prefix, suffix in (
        ("_kpool_tail_", "k"),
        ("_kpool_mtp_tail_", "k"),
        ("_kpool_tail_", "score"),
        ("_kpool_mtp_tail_", "score"),
    ):
        tables.append(
            torch.tensor(
                [getattr(indexer, prefix + suffix).data_ptr() for indexer in indexers],
                dtype=torch.int64,
                device=device,
            )
        )
    return tuple(tables), shape[1] * shape[2], source_shape[1]


def commit_fused(tables, row_elements, num_steps, requests, accepted):
    if requests.numel() != accepted.numel():
        raise ValueError("Request and accepted-step counts differ")
    if not requests.numel():
        return
    block = min(1024, triton.next_power_of_2(row_elements))
    tail_grid = triton.cdiv(row_elements, block)
    layers = tables[0].numel()
    logical_grid = requests.numel() * layers * tail_grid
    physical_grid = min(48, logical_grid)
    _commit_kernel[(physical_grid,)](
        *tables,
        requests,
        accepted,
        ROW_ELEMENTS=row_elements,
        NUM_STEPS=num_steps,
        NUM_LAYERS=layers,
        TAIL_GRID=tail_grid,
        LOGICAL_GRID=logical_grid,
        PHYSICAL_GRID=physical_grid,
        BLOCK=block,
        enable_auto_bind_sub_block=False,
    )
