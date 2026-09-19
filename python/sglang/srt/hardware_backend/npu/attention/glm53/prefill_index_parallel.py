"""Partition replicated prefill index queries at their existing causal tiles.

Only the pooled INT32 indices are exchanged. Each rank keeps its original
query/key projections, compressed-key cache, score arithmetic and decode path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
import math
import os


logger = logging.getLogger(__name__)

_PD_MODE_ENV = "SGLANG_GLM53_PD_PREFILL_INDEX_TP_MODE"
_VERIFY_DIR_ENV = "SGLANG_GLM53_PREFILL_INDEX_VERIFY_DIR"
_PD_VERIFY_MODE = "verify"
_PD_PERFORMANCE_MODE = "performance"
_PD_MODES = frozenset((_PD_VERIFY_MODE, _PD_PERFORMANCE_MODE))

# One record per execution mode and process is enough to prove that the
# optimized branch ran. A server process owns only one rank in production.
_reported_branches = set()


@dataclass(frozen=True)
class QueryFragment:
    request: int
    offset: int
    length: int
    sequence_end: int


@dataclass(frozen=True)
class FragmentBatchMetadata:
    """Minimal scorer metadata for a rank's contiguous query fragments."""

    batch_size: int
    extend_seq_lens_cpu: tuple[int, ...]
    seq_lens_cpu: tuple[int, ...]

    @classmethod
    def from_fragments(cls, fragments):
        if not fragments:
            raise ValueError("Fragment metadata requires at least one fragment")
        for fragment in fragments:
            if fragment.length <= 0 or fragment.sequence_end < fragment.length:
                raise ValueError("Invalid prefill index fragment")
        return cls(
            batch_size=len(fragments),
            extend_seq_lens_cpu=tuple(f.length for f in fragments),
            seq_lens_cpu=tuple(f.sequence_end for f in fragments),
        )


@dataclass(frozen=True)
class BatchShape:
    query_lengths: tuple[int, ...]
    sequence_ends: tuple[int, ...]

    @property
    def token_count(self):
        return sum(self.query_lengths)


@dataclass(frozen=True)
class VerificationShape:
    """Geometry that identifies one per-rank/layer correctness oracle."""

    rank: int
    layer_id: int
    world_size: int
    query_shape: tuple[int, ...]
    weights_shape: tuple[int, ...]
    indices_shape: tuple[int, ...]
    query_lengths: tuple[int, ...]
    sequence_ends: tuple[int, ...]
    compressed_shapes: tuple[tuple[int, ...], ...]


def _batch_shape(forward_batch):
    original = getattr(forward_batch, "_original_batch_size", None)
    batch_size = int(forward_batch.batch_size if original is None else original)
    if batch_size < 0:
        raise ValueError("Invalid prefill batch size")
    query_lengths_raw = forward_batch.extend_seq_lens_cpu
    sequence_ends_raw = forward_batch.seq_lens_cpu
    if (
        query_lengths_raw is None
        or sequence_ends_raw is None
        or len(query_lengths_raw) < batch_size
        or len(sequence_ends_raw) < batch_size
    ):
        raise ValueError("Incomplete prefill batch metadata")
    query_lengths = tuple(int(v) for v in query_lengths_raw[:batch_size])
    sequence_ends = tuple(int(v) for v in sequence_ends_raw[:batch_size])
    if any(length < 0 for length in query_lengths) or any(
        end < length for length, end in zip(query_lengths, sequence_ends)
    ):
        raise ValueError("Invalid prefill query length or sequence end")
    return BatchShape(query_lengths, sequence_ends)


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


def _pd_prefill_mode():
    # verify is the compatibility default: cache-on PD remains disabled without
    # VERIFY_DIR and performs the existing first-shape oracle with a directory.
    return os.getenv(_PD_MODE_ENV, _PD_VERIFY_MODE).strip().lower()


def enabled(q, forward_batch):
    import torch

    try:
        shape = _batch_shape(forward_batch)
    except (AttributeError, TypeError, ValueError):
        return False
    if (
        os.getenv("SGLANG_GLM53_PREFILL_INDEX_TP", "1") != "1"
        or q.device.type != "npu"
        or q.ndim != 3
        or q.shape[1:] != (32, 128)
        or q.dtype != torch.bfloat16
        or not shape.query_lengths
        or any(length == 0 for length in shape.query_lengths)
        or q.shape[0] < shape.token_count
        or shape.token_count < 2048
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
    """Bounded PD-P experiment with explicit oracle and timing modes.

    For cache-on PD, ``verify`` is fail-closed without VERIFY_DIR and runs a
    full per-rank scorer once for every layer and geometry. ``performance``
    never runs that scorer. Cache-off PD keeps its previous enablement policy.
    Absolute sequence ends preserve causal tiles after a prefix hit; cache
    restoration itself remains outside this row-partition optimization.
    """
    mode = _pd_prefill_mode()
    return (
        args.disaggregation_mode == "prefill"
        and os.getenv("SGLANG_GLM53_PD_PREFILL_INDEX_TP", "0") == "1"
        and (
            args.disable_radix_cache
            or (
                mode in _PD_MODES
                and (
                    mode == _PD_PERFORMANCE_MODE
                    or bool(os.getenv(_VERIFY_DIR_ENV))
                )
            )
        )
        and args.dp_size == args.moe_dp_size == args.dwdp_size == 1
        and args.attn_cp_size == 1
        and not args.enable_prefill_cp
        and forward_batch.forward_mode.is_extend_without_speculative()
        and not forward_batch.forward_mode.is_mixed()
    )


def parallel_pooled_topk(
    scorer, q, weights, compressed, forward_batch, pooled_topk, group
):
    """Run original score blocks on one owner and restore their request order."""
    import torch
    import torch.distributed as dist

    shape = _batch_shape(forward_batch)
    if (
        q.shape[0] < shape.token_count
        or weights.shape[0] < shape.token_count
        or len(compressed) < len(shape.query_lengths)
    ):
        raise ValueError("Incomplete prefill index tensors")
    world, rank = dist.get_world_size(group), dist.get_rank(group)
    partitions, capacity = partition_queries(
        shape.query_lengths, shape.sequence_ends, world
    )
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
        metadata = FragmentBatchMetadata.from_fragments(fragments)
        result = scorer(
            query, gates, [compressed[f.request] for f in fragments], metadata
        )
        expected_rows = sum(f.length for f in fragments)
        if result.ndim != 2 or result.shape != (expected_rows, pooled_topk):
            raise ValueError("Prefill index scorer returned an unexpected shape")
        local[:expected_rows].copy_(result)
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


def _execution_mode(args):
    if args.disaggregation_mode == "prefill":
        return f"pd-{_pd_prefill_mode()}"
    return "null"


def _record_branch_once(owner, actual, forward_batch, group, args=None):
    """Emit bounded proof that this process executed the optimized branch."""
    if args is None:
        from sglang.srt.runtime_context import get_server_args

        args = get_server_args()
    mode = _execution_mode(args)
    if mode in _reported_branches:
        return
    _reported_branches.add(mode)
    import torch.distributed as dist

    rank = dist.get_rank(group)
    shape = _batch_shape(forward_batch)
    logger.info(
        "GLM53 prefill index TP branch active: mode=%s rank=%s/%s layer=%s "
        "query_lengths=%s sequence_ends=%s indices_shape=%s",
        mode,
        rank,
        dist.get_world_size(group),
        owner.layer_id,
        shape.query_lengths,
        shape.sequence_ends,
        tuple(int(v) for v in actual.shape),
    )


def _verification_directory(args):
    directory = os.getenv(_VERIFY_DIR_ENV)
    if not directory:
        return None
    # A stale VERIFY_DIR must never re-enable the full scorer during timing.
    if (
        args.disaggregation_mode == "prefill"
        and _pd_prefill_mode() == _PD_PERFORMANCE_MODE
    ):
        return None
    return directory


def _verification_shape(owner, actual, q, weights, compressed, forward_batch, group):
    import torch.distributed as dist

    batch = _batch_shape(forward_batch)
    count = len(batch.query_lengths)
    if len(compressed) < count:
        raise ValueError("Incomplete compressed-key metadata for verification")
    return VerificationShape(
        rank=dist.get_rank(group),
        layer_id=int(owner.layer_id),
        world_size=dist.get_world_size(group),
        query_shape=tuple(int(v) for v in q.shape),
        weights_shape=tuple(int(v) for v in weights.shape),
        indices_shape=tuple(int(v) for v in actual.shape),
        query_lengths=batch.query_lengths,
        sequence_ends=batch.sequence_ends,
        compressed_shapes=tuple(
            tuple(int(v) for v in tensor.shape) for tensor in compressed[:count]
        ),
    )


def verify_once(owner, actual, q, weights, compressed, forward_batch, group):
    """Optionally compare one per-rank/layer/shape result with the full scorer."""
    from sglang.srt.runtime_context import get_server_args

    args = get_server_args()
    _record_branch_once(owner, actual, forward_batch, group, args)
    directory = _verification_directory(args)
    if directory is None:
        return
    import json
    import time
    from pathlib import Path
    import torch

    shape = _verification_shape(
        owner, actual, q, weights, compressed, forward_batch, group
    )
    checked = getattr(owner, "_glm53_checked_prefill_shapes", set())
    if shape in checked:
        return
    expected = owner._prefill_pooled_topk(q, weights, compressed, forward_batch)
    exact = torch.equal(actual, expected)
    record = dict(
        timestamp=time.time(),
        mode=_execution_mode(args),
        oracle="full_per_rank_prefill_scorer",
        exact=exact,
        **asdict(shape),
    )
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    with (root / f'rank{shape.rank}.jsonl').open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    if not exact:
        raise RuntimeError(
            "GLM prefill TP index disagrees with the original per-rank scorer"
        )
    checked.add(shape)
    owner._glm53_checked_prefill_shapes = checked
