"""Default-off bounded routing; CPU metadata only, no data-dependent shapes.

Experimental: full installed-NPU clamped-FFN parity is a release prerequisite.
The caller has already performed the unchanged quantization and collectives.
"""
from dataclasses import dataclass
import logging

import torch

_calls = 0
_capacity_rows = 0
_max_capacity_rows = 0
_rank_stats = {}
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Geometry:
    local_rows: int
    world_rows: int
    capacity: int
    active_mask: int
    blocks: int


def static_route_geometry(world, metadata):
    """Return None outside checked target DP2/TP8/EP16 small-tail geometry.

    No live tensor count selects this path. Positive domains retain ALL their
    padded rows, just as _normal_skip_idle_dp_routes does in the baseline.
    """
    if world != 16 or not isinstance(metadata, (tuple, list)) or len(metadata) != 2:
        return None
    original, padded = metadata
    if any(not isinstance(v, (tuple, list)) or len(v) != 2 or
           any(type(n) is not int or n < 0 for n in v) for v in (original, padded)):
        return None
    if (padded[0] != padded[1] or any(a > b for a, b in zip(original, padded))
            or not 0 < max(original) <= 512 or padded[0] % 8):
        return None
    active = int(original[0] > 0) + int(original[1] > 0)
    active_rows = active * padded[0]
    if not 0 < active_rows <= 512:
        return None
    rows = padded[0] // 8
    return Geometry(rows, rows * 16, active_rows * 8,
                    int(original[0] > 0) | (int(original[1] > 0) << 1),
                    (rows * 16 * 8 + 1023) // 1024)


def get_static_gpu_route_stats():
    """Process-local CPU enqueue counters; not device completion evidence."""
    return {"static_gpu_route_calls": _calls,
            "static_gpu_route_capacity_rows": _capacity_rows,
            "static_gpu_route_max_capacity_rows": _max_capacity_rows,
            "rank_stats": [dict(ep_rank=rank, dp_rank=rank // 8,
                                attn_tp_rank=rank % 8, **values)
                           for rank, values in sorted(_rank_stats.items())]}


def _record_route_enqueue(geometry, rank):
    """CPU only; first successful enqueue logs once per EP rank, in prewarm.

    No reset on OFF/ON. Main must establish all 16 activation receipts before
    timing; local stats never pretend to aggregate other worker processes.
    """
    global _calls, _capacity_rows, _max_capacity_rows
    _calls += 1
    _capacity_rows += geometry.capacity
    _max_capacity_rows = max(_max_capacity_rows, geometry.capacity)
    first = rank not in _rank_stats
    row = _rank_stats.setdefault(rank, {"calls": 0, "capacity_rows": 0,
                                       "max_capacity_rows": 0, "activation_logged": False})
    row["calls"] += 1
    row["capacity_rows"] += geometry.capacity
    row["max_capacity_rows"] = max(row["max_capacity_rows"], geometry.capacity)
    if first:
        logger.info("GLM53 static GPU route active: ep_rank=%d dp_rank=%d attn_tp_rank=%d "
                    "capacity=%d local_rows=%d counts_device=True event=enqueue",
                    rank, rank // 8, rank % 8, geometry.capacity, geometry.local_rows)
        row["activation_logged"] = True


def payload_supported(q, scale, ids, weights, geometry, rank):
    """Only layout/dtype/CPU metadata checks. No value reduction or readback."""
    if geometry is None or type(rank) is not int or not 0 <= rank < 16:
        return False
    n = geometry.world_rows
    supported = (q.device.type == "npu" and q.dtype == torch.int8
            and tuple(q.shape) == (n, 4096) and q.storage_offset() % 4 == 0
            and scale.dtype == weights.dtype == torch.float32
            and ids.dtype in (torch.int32, torch.int64)
            and tuple(scale.shape) == (n,)
            and tuple(ids.shape) == tuple(weights.shape) == (n, 8)
            and all(t.device == q.device and t.layout == torch.strided
                    and t.is_contiguous() for t in (q, scale, ids, weights)))
    if not supported:
        return False
    from torch_npu import get_npu_format
    return all(get_npu_format(t) in (0, 2) for t in (q, scale, ids, weights))


def static_gpu_dispatch(q, scale, ids, weights, geometry, rank):
    """Return recv, scales, actual INT64 device counts, seven-field state.

    `state[5]` is capacity, not actual count. Unused rows/gates are zero and
    inverse indices only address actual routes. Counts alone drive both GMMs.
    Scratch is per call on the current stream; no mutable device-buffer cache.
    """
    if not payload_supported(q, scale, ids, weights, geometry, rank):
        raise ValueError("static routing requires the checked contiguous NPU payload")
    from .glm53_static_gpu_route_triton import route

    c, n, blocks = geometry.capacity, geometry.world_rows, geometry.blocks
    partial = torch.empty((18, blocks), dtype=torch.int32, device=q.device)
    offsets = torch.empty_like(partial)
    counts = torch.empty(18, dtype=torch.int64, device=q.device)
    total = torch.empty(1, dtype=torch.int32, device=q.device)
    selected = torch.empty(c, dtype=torch.int32, device=q.device)
    inverse = torch.full((n, 8), -1, dtype=torch.int32, device=q.device)
    recv = torch.empty((c, 4096), dtype=torch.int8, device=q.device)
    recv_scale = torch.empty(c, dtype=torch.float32, device=q.device)
    rows = torch.empty(c, dtype=torch.int64, device=q.device)
    gates = torch.empty(c, dtype=torch.float32, device=q.device)
    route(q, scale, ids, weights, partial, offsets, counts, total, selected,
          inverse, recv, recv_scale, rows, gates, geometry, rank * 18)
    _record_route_enqueue(geometry, rank)
    return recv, recv_scale, counts, (
        rows, gates, geometry.local_rows, 16, 4096, c, inverse)
