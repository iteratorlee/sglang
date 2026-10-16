"""Bounded, opt-in HCCL all-to-all proof of concept for two-node GLM P.

SGLANG_GLM53_NORMAL_ALL_TO_ALL=1 requires the existing SKIP_IDLE_DP scope
(DP2/TP8/EP16 eager normal prefill, ModelSlim W8A8 clamp10). The contract is
288 experts, topk8, BF16 H4096, and at most 1024 local tokens per rank.
The legacy shape/device-routing/ordered/RS switches do not select algorithms
inside this path. D and the default normal path retain their existing code.

Dispatch gathers only 16x288 INT64 counts. Two variable-split all-to-alls
transfer expanded INT8 rows and FP32 scales separately. Expert order is
identical to the legacy CPU dispatch: expert, source rank, original route.
The reverse all-to-all returns BF16 expert outputs to their source. FP32
gates stay at source; the existing ordered kernel sums in global expert
order, with separately rounded FP32 products and no atomics or HCCL sum.

This changes FP32 association: the legacy path sums within each EP owner
and then uses an HCCL reduction tree; here all contributions are summed
sequentially at source. End-to-end numerical validation is required.
Zero splits must be supported by HCCL, including ranks with no local traffic.
Only a globally empty batch skips payload collectives on every rank.
All EP ranks must use the same switch; there is no mid-collective fallback.
At the bound, full expert skew can place 131072 routes on one owner (1 GiB
of BF16 expert outputs plus another 1 GiB for the reverse permutation).
"""

import logging
import os
from typing import List, NamedTuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)
MAX_LOCAL_TOKENS = 1024
_activation_logged = False


class AllToAllState(NamedTuple):
    send_splits: List[int]
    recv_splits: List[int]
    source_order: torch.Tensor
    weights: torch.Tensor
    indices: torch.Tensor
    tokens: int
    width: int
    received: int
    global_routes: int


def normal_all_to_all_enabled(group):
    if os.getenv("SGLANG_GLM53_NORMAL_ALL_TO_ALL", "0") != "1":
        return False
    from .glm53_collectives import _idle_dp_counts

    return _idle_dp_counts.get() is not None and dist.get_world_size(group) == 16


def _local_routes(ids, num_experts, active):
    """CPU offsets in stable global expert order; invalid routes are ignored."""
    if not active:
        return torch.empty(0, dtype=torch.int64), torch.zeros(
            num_experts, dtype=torch.int64
        )
    flat = ids.cpu().flatten()
    selected = ((flat >= 0) & (flat < num_experts)).nonzero().flatten()
    selected = selected[torch.argsort(flat[selected], stable=True)]
    counts = torch.bincount(flat[selected], minlength=num_experts)
    return selected, counts


def _receive_plan(counts, rank):
    """Map concatenated source blocks to stable expert/source order on CPU."""
    world, num_experts = counts.shape
    if (
        counts.device.type != "cpu"
        or counts.dtype != torch.int64
        or num_experts % world
        or not 0 <= rank < world
        or (counts < 0).any()
    ):
        raise ValueError("Invalid GLM all-to-all CPU count matrix")
    experts = num_experts // world
    local = counts[:, rank * experts : (rank + 1) * experts]
    recv_splits = local.sum(1).tolist()
    expert_counts = local.sum(0).tolist()
    starts = local.cumsum(1) - local
    source_start = 0
    for source, size in enumerate(recv_splits):
        starts[source] += source_start
        source_start += size
    pieces = [
        torch.arange(int(starts[source, expert]), int(starts[source, expert] + size))
        for expert in range(experts)
        for source in range(world)
        if (size := int(local[source, expert]))
    ]
    permutation = torch.cat(pieces) if pieces else torch.empty(0, dtype=torch.int64)
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel())
    return recv_splits, expert_counts, permutation, inverse


def all_to_all_dispatch(group, x, ids, weights, num_experts=288):
    global _activation_logged
    from .glm53_collectives import (
        _normal_combine_ordered_indices,
        _normal_skip_idle_dp_routes,
        release_normal_hccl_cached_memory,
    )

    if not normal_all_to_all_enabled(group):
        raise ValueError("GLM all-to-all requires the opt-in two-node P scope")
    world, rank = dist.get_world_size(group), dist.get_rank(group)
    if not (
        num_experts == 288
        and x.ndim == 2
        and x.dtype == torch.bfloat16
        and x.shape[1] == 4096
        and 0 <= x.shape[0] <= MAX_LOCAL_TOKENS
        and tuple(ids.shape) == tuple(weights.shape) == (x.shape[0], 8)
        and ids.dtype in (torch.int32, torch.int64)
        and weights.dtype == torch.float32
        and x.device == ids.device == weights.device
    ):
        raise ValueError(
            "GLM all-to-all requires BF16 H4096/topk8/288 experts, FP32 gates, local tokens <= 1024"
        )
    tokens, width = x.shape
    # Reuse the verified EP/TP/attention-DP rank-mapping guard without copying
    # global IDs or filtering global route offsets.
    _, _, idle_counts = _normal_skip_idle_dp_routes(
        torch.empty(0, dtype=torch.int64), group, world, rank, tokens, 8
    )
    release_normal_hccl_cached_memory(x)
    selected, local_counts = _local_routes(ids, num_experts, idle_counts[rank // 8] > 0)
    counts_device = local_counts.to(x.device)
    gathered = counts_device.new_empty(world * num_experts)
    dist.all_gather_into_tensor(gathered, counts_device, group=group)
    counts = gathered.view(world, num_experts).cpu()
    if (counts < 0).any() or (counts.sum(1) > MAX_LOCAL_TOKENS * 8).any():
        raise ValueError("GLM all-to-all peer route counts exceed the bounded contract")
    global_routes = int(counts.sum())
    send_splits = local_counts.view(world, -1).sum(1).tolist()
    recv_splits, expert_counts, permutation, inverse = _receive_plan(counts, rank)
    received = sum(recv_splits)
    loc = selected.to(x.device)
    if selected.numel():
        # Quantize each local token once, before expanding to its valid routes.
        q, scale = torch.ops.npu.npu_dynamic_quant(x)
        if (
            q.dtype != torch.int8
            or scale.dtype != torch.float32
            or tuple(scale.shape) != (tokens,)
        ):
            raise RuntimeError(
                "GLM all-to-all requires INT8 rows and FP32 token scales"
            )
        route_rows = torch.div(loc, 8, rounding_mode="floor")
        send_q, send_scale = q.index_select(0, route_rows), scale.index_select(
            0, route_rows
        )
    else:
        # Idle/invalid/zero-token sources still receive expert work from peers.
        send_q = torch.empty((0, width), device=x.device, dtype=torch.int8)
        send_scale = torch.empty((0,), device=x.device, dtype=torch.float32)
    raw_q = send_q.new_empty((received, width))
    raw_scale = send_scale.new_empty((received,))
    if global_routes:
        for output, source in ((raw_q, send_q), (raw_scale, send_scale)):
            dist.all_to_all_single(
                output,
                source.contiguous(),
                output_split_sizes=recv_splits,
                input_split_sizes=send_splits,
                group=group,
            )
    if received:
        order = permutation.to(x.device)
        recv, recv_scale = raw_q.index_select(0, order), raw_scale.index_select(
            0, order
        )
    else:
        # Same GMM dummy allocation/count contract as the legacy dispatcher.
        recv, recv_scale = send_q.new_zeros((1, width)), send_scale.new_ones((1,))
    indices = _normal_combine_ordered_indices(selected // 8, tokens, 8)
    state = AllToAllState(
        send_splits,
        recv_splits,
        inverse.to(x.device),
        weights.flatten().index_select(0, loc),
        indices.to(x.device),
        tokens,
        width,
        received,
        global_routes,
    )
    if not _activation_logged:
        logger.info(
            "GLM53 normal HCCL all-to-all active: rank=%d world=%d local_rows=%d "
            "sent=%d received=%d q=INT8 scale=FP32 return=BF16 "
            "combine=source_ordered_FP32 max_local_rows=%d",
            rank,
            world,
            tokens,
            selected.numel(),
            received,
            MAX_LOCAL_TOKENS,
        )
        _activation_logged = True
    return recv, recv_scale, expert_counts, state


def all_to_all_combine(group, y, state):
    from .glm53_ordered_combine import ordered_combine

    if not (
        y.dtype == torch.bfloat16
        and y.ndim == 2
        and tuple(y.shape) == (max(1, state.received), state.width)
        and y.device == state.weights.device == state.indices.device
    ):
        raise ValueError(
            "GLM all-to-all return requires BF16 expert rows on the dispatch device"
        )
    # Undo destination's expert/source permutation, including empty experts.
    # Never read the dummy row for a rank with no received routes.
    source = y[: state.received].index_select(0, state.source_order)
    returned = y.new_empty((sum(state.send_splits), state.width))
    if state.global_routes:
        dist.all_to_all_single(
            returned,
            source.contiguous(),
            output_split_sizes=state.send_splits,
            input_split_sizes=state.recv_splits,
            group=group,
        )
    # Returned destination blocks are already in the source's stable global
    # expert order. Only a local [tokens, width] FP32 output is allocated.
    out = ordered_combine(
        returned, state.weights, state.indices, state.tokens, state.width
    )
    return out.to(y.dtype)
