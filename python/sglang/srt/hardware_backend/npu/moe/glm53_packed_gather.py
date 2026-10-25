"""Bucketed GLM payload transport. No quantization, float math, or routing here.

INT32 is a transport container for raw bytes, never a numeric conversion.
Triton is imported only on an admitted NPU call. CPU helpers are test oracles.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist

WIDTH = 4096
TOPK = 8
WORLD = 16
MAX_ROWS = 512
CAPACITIES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
ALIGN_WORDS = 32  # Each field and each source-rank record starts at 128 bytes.
_executions = 0


def get_packed_gather_stats():
    """CPU enqueue counter; call outside timing windows, after synchronization.

    Process-local, cumulative, not summed across replicas. No device readback.
    The flag-off path neither imports this module nor updates this counter.
    """
    return {"packed_gather_calls": _executions}


def target_metadata_eligible(world, metadata):
    """Only shared CPU metadata may select 1 versus 4 payload collectives.

    Caller supplies metadata only from a target layer's checked idle-DP scope.
    Never pass a rank's own token count. Flags/config must match all EP ranks.
    """
    if world != WORLD or metadata is None:
        return False
    original, padded = metadata
    if any(
        not isinstance(v, (tuple, list)) or len(v) != 2
        or any(type(n) is not int or n < 0 for n in v)
        for v in (original, padded)
    ):
        raise ValueError("packed gather needs two nonnegative global CPU counts")
    if padded[0] != padded[1] or any(a > b for a, b in zip(original, padded)):
        raise ValueError("packed gather needs matching MAX_LEN global padding")
    return 0 < max(original) <= 4096 and 0 < padded[0] <= 4096 and padded[0] % 8 == 0


@dataclass(frozen=True)
class Layout:
    rows: int
    capacity: int
    id_words: int
    offsets: tuple
    lengths: tuple
    capacity_lengths: tuple
    words: int

    @property
    def nbytes(self):
        return self.words * 4


def payload_layout(rows, ids_dtype):
    if type(rows) is not int or not 0 <= rows <= MAX_ROWS:
        raise ValueError("packed GLM payload requires 0..512 local rows")
    if ids_dtype not in (torch.int64, torch.int32):
        raise ValueError("IDs must retain their original INT64 or INT32 dtype")
    id_words = 2 if ids_dtype == torch.int64 else 1
    # Zero has a CPU-oracle-only empty record in capacity1. Production zero
    # admission stays on gather4 and never launches the bucket kernels.
    capacity = 1 << (max(rows, 1) - 1).bit_length()
    lengths = (rows * (WIDTH // 4), rows, rows * TOPK * id_words, rows * TOPK)
    capacity_lengths = (
        capacity * (WIDTH // 4), capacity,
        capacity * TOPK * id_words, capacity * TOPK,
    )
    offsets, cursor = [], 0
    for length in capacity_lengths:
        offsets.append(cursor)
        cursor += length
        cursor = (cursor + ALIGN_WORDS - 1) // ALIGN_WORDS * ALIGN_WORDS
    return Layout(rows, capacity, id_words, tuple(offsets), lengths, capacity_lengths, cursor)


def validate_payload(q, scale, ids, weights, *, expected_rows=None, npu=False):
    # Contract violations abort; never catch these and locally revert to gather4.
    rows = q.shape[0] if q.ndim else -1
    layout = payload_layout(rows, ids.dtype)
    if (
        tuple(q.shape) != (rows, WIDTH) or q.dtype != torch.int8
        or tuple(scale.shape) != (rows,) or scale.dtype != torch.float32
        or tuple(ids.shape) != (rows, TOPK)
        or tuple(weights.shape) != (rows, TOPK) or weights.dtype != torch.float32
        or any(t.device != q.device for t in (scale, ids, weights))
        or any(t.layout != torch.strided for t in (q, scale, ids, weights))
        or (expected_rows is not None and rows != expected_rows)
        or (npu and q.device.type != "npu")
    ):
        raise ValueError("packed GLM payload violates audited H4096/topk8 contract")
    if npu:
        from torch_npu import get_npu_format

        # Raw pointers must describe ordinary ND storage, including strided
        # views. A logical torch.strided layout alone does not exclude an
        # Ascend blocked/fractal physical format. Reject instead of reordering
        # or making a rank-local collective fallback.
        if any(get_npu_format(t) not in (0, 2) for t in (q, scale, ids, weights)):
            raise ValueError("packed GLM payload requires ordinary ND storage")
    return layout


def pack_reference(q, scale, ids, weights):
    """CPU-only raw-word oracle; zero all field/end padding deterministically."""
    layout = validate_payload(q, scale, ids, weights)
    if q.device.type != "cpu":
        raise ValueError("reference is CPU only")
    packed = torch.zeros(layout.words, dtype=torch.int32)
    for t, offset, length in zip((q, scale, ids, weights), layout.offsets, layout.lengths):
        if length:
            flat = t.reshape(-1).clone(memory_format=torch.contiguous_format)
            packed[offset:offset + length].copy_(flat.view(torch.int32))
    return packed


def unpack_reference(gathered, rows, ids_dtype, world=WORLD):
    """CPU oracle for rank-major HCCL output; four independent contiguous tensors."""
    layout = payload_layout(rows, ids_dtype)
    if gathered.device.type != "cpu" or gathered.dtype != torch.int32 or gathered.numel() != world * layout.words:
        raise ValueError("invalid packed reference input")
    rank_records = gathered.reshape(world, layout.words)
    result = []
    shapes = ((world * rows, WIDTH), (world * rows,), (world * rows, TOPK), (world * rows, TOPK))
    for offset, length, dtype, shape in zip(
        layout.offsets, layout.lengths,
        (torch.int8, torch.float32, ids_dtype, torch.float32), shapes,
    ):
        # clone keeps byte interpretation and avoids inter-record field padding.
        result.append(rank_records[:, offset:offset + length].clone().reshape(-1).view(dtype).reshape(shape))
    return tuple(result)


def packed_all_gather(group, q, scale, ids, weights, *, expected_rows):
    """One pack launch, one INT32 HCCL gather, one unpack launch; no fallback.

    Requires rank-uniform admission and the existing shape audit in the caller.
    IDs use two raw words for INT64, one for INT32; there is no narrowing.
    """
    global _executions
    layout = validate_payload(q, scale, ids, weights, expected_rows=expected_rows, npu=True)
    # expected_rows comes from the shared CPU DP metadata in the caller. A
    # local mismatch raises above, never selects another packet size/gather.
    if not layout.rows or dist.get_world_size(group) != WORLD:
        raise ValueError("packed gather requires positive rows and EP16 admission")
    from .glm53_packed_gather_triton import pack, unpack

    send = torch.empty(layout.words, dtype=torch.int32, device=q.device)
    gathered = torch.empty(WORLD * layout.words, dtype=torch.int32, device=q.device)
    pack(q, scale, ids, weights, send, layout)
    dist.all_gather_into_tensor(gathered, send, group=group)
    result = (
        torch.empty((WORLD * layout.rows, WIDTH), dtype=torch.int8, device=q.device),
        torch.empty((WORLD * layout.rows,), dtype=torch.float32, device=q.device),
        torch.empty((WORLD * layout.rows, TOPK), dtype=ids.dtype, device=q.device),
        torch.empty((WORLD * layout.rows, TOPK), dtype=torch.float32, device=q.device),
    )
    unpack(gathered, result, layout, WORLD)
    _executions += 1
    return result
