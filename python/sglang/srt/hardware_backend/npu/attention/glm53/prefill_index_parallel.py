"""Partition replicated prefill index queries at their existing causal tiles.

Only the pooled INT32 indices are exchanged. Each rank keeps its original
query/key projections, compressed-key cache, score arithmetic and decode path.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from types import SimpleNamespace


@dataclass(frozen=True)
class QueryFragment:
    request: int
    offset: int
    length: int
    sequence_end: int


def partition_queries(query_lengths, sequence_lengths, world_size):
    """Balance complete 128-position score tiles without changing their shape."""
    if world_size <= 0 or len(query_lengths) != len(sequence_lengths):
        raise ValueError("Invalid prefill index partition dimensions")
    blocks = []
    request_offsets = []
    total = 0
    for req, (length, end) in enumerate(zip(query_lengths, sequence_lengths)):
        length, end = int(length), int(end)
        if length < 0 or end < length:
            raise ValueError("Invalid prefill query length or sequence end")
        request_offsets.append(total)
        first = end - length
        start = 0
        while start < length:
            stop = min(length, ((first + start) // 128 + 1) * 128 - first)
            blocks.append((req, start, stop, first))
            start = stop
        total += length
    per_rank = math.ceil(len(blocks) / world_size)
    partitions = []
    for rank in range(world_size):
        fragments = []
        for req, start, stop, first in blocks[rank * per_rank : (rank + 1) * per_rank]:
            offset = request_offsets[req] + start
            if fragments and fragments[-1].request == req:
                previous = fragments.pop()
                assert previous.offset + previous.length == offset
                fragments.append(
                    QueryFragment(
                        req,
                        previous.offset,
                        previous.length + stop - start,
                        first + stop,
                    )
                )
            else:
                fragments.append(QueryFragment(req, offset, stop - start, first + stop))
        partitions.append(fragments)
    return partitions, per_rank * 128


def enabled(q, forward_batch):
    import torch

    if (
        os.getenv("SGLANG_GLM53_PREFILL_INDEX_TP", "1") != "1"
        or q.device.type != "npu"
        or q.ndim != 3
        or q.shape[1:] != (32, 128)
        or q.dtype != torch.bfloat16
        or sum(forward_batch.extend_seq_lens_cpu) < 2048
    ):
        return False
    from sglang.srt.runtime_context import get_server_args

    args = get_server_args()
    return (
        args.tp_size == args.ep_size == 16
        and args.nnodes == 1
        and args.pp_size == 1
        and not args.enable_dp_attention
        and not args.enable_two_batch_overlap
        and args.disable_overlap_schedule
        and args.quantization == "modelslim"
        and (
            (args.disable_radix_cache and args.disaggregation_mode == "null")
            or _pd_prefill_enabled(args, forward_batch)
        )
    )


def _pd_prefill_enabled(args, forward_batch):
    """Explicit PD-P experiment; the existing null-mode policy is unchanged.

    Absolute sequence ends preserve the original scorer's causal tiles even
    after a prefix hit. Prefix-enabled experiments additionally require the
    existing per-rank/layer/shape oracle; cache restoration itself is outside
    this row-partition optimization's correctness contract.
    """
    return (
        args.disaggregation_mode == "prefill"
        and os.getenv("SGLANG_GLM53_PD_PREFILL_INDEX_TP", "0") == "1"
        and args.dp_size == args.moe_dp_size == args.dwdp_size == 1
        and args.attn_cp_size == 1
        and not args.enable_prefill_cp
        and forward_batch.forward_mode.is_extend_without_speculative()
        and not forward_batch.forward_mode.is_mixed()
        and (
            args.disable_radix_cache
            or bool(os.getenv("SGLANG_GLM53_PREFILL_INDEX_VERIFY_DIR"))
        )
    )


def parallel_pooled_topk(
    scorer, q, weights, compressed, forward_batch, pooled_topk, group
):
    """Run original score blocks on one owner and restore their request order."""
    import torch
    import torch.distributed as dist

    world, rank = dist.get_world_size(group), dist.get_rank(group)
    batch = (getattr(forward_batch, "_original_batch_size", None) or forward_batch.batch_size)
    lengths = [int(x) for x in forward_batch.extend_seq_lens_cpu[:batch]]
    ends = [int(x) for x in forward_batch.seq_lens_cpu[:batch]]
    partitions, capacity = partition_queries(lengths, ends, world)
    if capacity == 0:
        return torch.empty((0, pooled_topk), device=q.device, dtype=torch.int32)
    fragments = partitions[rank]
    local = torch.full((capacity, pooled_topk), -1, device=q.device, dtype=torch.int32)
    if fragments:
        query_parts = [q[f.offset : f.offset + f.length] for f in fragments]
        weight_parts = [weights[f.offset : f.offset + f.length] for f in fragments]
        query = query_parts[0] if len(query_parts) == 1 else torch.cat(query_parts, 0)
        gates = (
            weight_parts[0] if len(weight_parts) == 1 else torch.cat(weight_parts, 0)
        )
        metadata = SimpleNamespace(
            batch_size=len(fragments),
            extend_seq_lens_cpu=[f.length for f in fragments],
            seq_lens_cpu=[f.sequence_end for f in fragments],
        )
        result = scorer(
            query, gates, [compressed[f.request] for f in fragments], metadata
        )
        local[: result.shape[0]].copy_(result)
    gathered = torch.empty(
        (world * capacity, pooled_topk), device=q.device, dtype=torch.int32
    )
    dist.all_gather_into_tensor(gathered, local, group=group)
    counts = [sum(f.length for f in fragments) for fragments in partitions]
    if all(n == capacity for n in counts):
        return gathered
    return torch.cat(
        [gathered[r * capacity : r * capacity + n] for r, n in enumerate(counts) if n],
        0,
    )


def verify_once(owner, actual, q, weights, compressed, forward_batch, group):
    """Optional warmup-only, per-shape comparison against every rank's scorer."""
    directory = os.getenv("SGLANG_GLM53_PREFILL_INDEX_VERIFY_DIR")
    if not directory:
        return
    import json
    import time
    from pathlib import Path
    import torch
    import torch.distributed as dist

    count = (getattr(forward_batch, "_original_batch_size", None) or forward_batch.batch_size)
    lengths = tuple(int(v) for v in forward_batch.extend_seq_lens_cpu[:count])
    ends = tuple(int(v) for v in forward_batch.seq_lens_cpu[:count])
    key = (lengths, ends)
    checked = getattr(owner, "_glm53_checked_prefill_shapes", set())
    if key in checked:
        return
    expected = owner._prefill_pooled_topk(q, weights, compressed, forward_batch)
    exact = torch.equal(actual, expected)
    record = dict(
        timestamp=time.time(),
        rank=dist.get_rank(group),
        layer_id=owner.layer_id,
        query_lengths=lengths,
        sequence_ends=ends,
        indices_shape=list(actual.shape),
        exact=exact,
    )
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    with (root / f'rank{record["rank"]}.jsonl').open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    if not exact:
        raise RuntimeError(
            "GLM prefill TP index disagrees with the original per-rank scorer"
        )
    checked.add(key)
    owner._glm53_checked_prefill_shapes = checked
