"""GLM normal-phase INT8 routing through standard HCCL collectives.

Retains the verified fallback for vendor dynamic-receive-size failures. This
path is explicitly enabled for GLM W8A8 layers; decode still uses DeepEP LL.

The following experiments require the existing NORMAL_SKIP_IDLE_DP=1 scope
in eager two-node DP2/TP8/EP16 prefill (all names have the SGLANG_GLM53_ prefix):
NORMAL_SHAPE_CACHE=1 replaces repeated shape audits with an invalidation vote;
NORMAL_REDUCE_SCATTER=2node selects the scoped FP32 combine collective;
NORMAL_REDUCE_SCATTER_CHECK=1 adds an expensive all-reduce reference check;
NORMAL_DEVICE_ROUTING=1 keeps route sorting and expert counts on device.
NORMAL_COMBINE_ORDERED=2node uses fixed-order FP32 Triton accumulation with
CPU routing; it is mutually exclusive with NORMAL_DEVICE_ROUTING=1.
Use identical collective flags on every EP rank. All experiments default off.
"""

import logging
import os
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)
_last_dispatch_metadata = None
_activation_logged = False
_bulk_activation_logged = False
_idle_dp_counts = ContextVar("glm53_normal_hccl_idle_dp_counts", default=None)
_idle_dp_logged = False
_dispatch_shape_cache = {}
_shape_cache_logged = False
_reduce_scatter_check_logged = False
_device_routing_logged = False
_ordered_combine_logged = False


def release_normal_hccl_cached_memory(x):
    """Leave driver headroom for HCCL/CANN allocations outside torch's pool."""
    reserve_mb = int(os.getenv("SGLANG_GLM53_NORMAL_MEMORY_HEADROOM_MB", "0"))
    if reserve_mb < 0:
        raise ValueError("SGLANG_GLM53_NORMAL_MEMORY_HEADROOM_MB must be non-negative")
    if not reserve_mb or x.device.type != "npu":
        return
    device = x.device
    free, _ = torch.npu.mem_get_info(device)
    if free >= reserve_mb * 1024**2:
        return
    reserved = torch.npu.memory_reserved(device)
    allocated = torch.npu.memory_allocated(device)
    if reserved - allocated < 64 * 1024**2:
        return
    # The P profile is non-overlap/eager. Finish outstanding work before
    # returning unused allocator blocks; live weights and KV are retained.
    torch.npu.synchronize(device)
    torch.npu.empty_cache()
    free_after, _ = torch.npu.mem_get_info(device)
    logger.warning(
        "GLM normal HCCL cache release: free=%.1f->%.1f MiB "
        "reserved=%.1f allocated=%.1f MiB headroom=%d MiB",
        free / 1024**2, free_after / 1024**2,
        reserved / 1024**2, allocated / 1024**2, reserve_mb,
    )


def normal_hccl_profile(args, limit):
    """Finite opt-in profiles; preserve the original single-node guards."""
    common = (
        args.pp_size == 1
        and not args.enable_two_batch_overlap
        and not args.enable_eplb
        and args.ep_num_redundant_experts == 0
        and args.quantization == "modelslim"
        and limit == 10.0
        and args.moe_a2a_backend == "deepep"
    )
    if not common:
        return None
    if args.nnodes == 2:
        # Each TP8 attention domain stays on one 8-device host. Only the EP16
        # gather/combine crosses hosts. Keep this experimental profile explicit
        # until hardware validation; no general nnodes/EP allow-list expansion.
        if (
            os.getenv("SGLANG_GLM53_2NODE_PD", "0") == "1"
            and args.tp_size == args.ep_size == 16
            and args.enable_dp_attention
            and args.dp_size == 2
            and args.attn_cp_size == args.dcp_size == 1
            and args.moe_dp_size == args.dwdp_size == 1
            and not args.enable_prefill_cp
            and args.disaggregation_mode == "prefill"
            and args.disable_overlap_schedule
        ):
            return "prefill_2node_dp2_tp8_ep16"
        return None
    if args.nnodes != 1:
        return None
    if args.tp_size == args.ep_size == 16 and not args.enable_dp_attention:
        return "tp16_ep16"
    if (
        args.tp_size == args.ep_size == 8
        and args.attn_cp_size == args.dcp_size == 1
        and args.moe_dp_size == 1
        and not args.enable_prefill_cp
        and args.disaggregation_mode == "prefill"
        and args.disable_overlap_schedule
    ):
        if args.enable_dp_attention and args.dp_size == 4:
            return "prefill_dp4_tp2_ep8"
        if not args.enable_dp_attention and args.dp_size == 1:
            # Single TP8 attention domain already gives equal local rows on
            # every EP rank. The same checked INT8 gather/GMM/combine contract
            # applies without the DP-specific MAX_LEN padding policy.
            return "prefill_tp8_ep8"
    return None


def normal_hccl_skip_idle_dp_enabled():
    """Only the explicitly enabled, eager two-node P normal profile may opt in."""
    if (
        os.getenv("SGLANG_GLM53_NORMAL_SKIP_IDLE_DP", "0") != "1"
        or os.getenv("SGLANG_GLM53_NORMAL_HCCL", "0") != "1"
    ):
        return False
    from sglang.srt.runtime_context import get_server_args

    args = get_server_args()
    return (
        normal_hccl_profile(args, 10.0) == "prefill_2node_dp2_tp8_ep16"
        and args.deepep_mode == "normal"
        and args.disable_cuda_graph
        and not args.enable_single_batch_overlap
    )


@contextmanager
def normal_hccl_idle_dp_scope(forward_batch, mlp_context=None):
    """Publish immutable original CPU counts for just this GLM MLP invocation."""
    counts = None
    if normal_hccl_skip_idle_dp_enabled():
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        if forward_batch.forward_mode == ForwardMode.EXTEND:
            original = getattr(forward_batch, "original_global_num_tokens_cpu", None)
            padded = getattr(forward_batch, "global_num_tokens_cpu", None)
            for name, values in (("original", original), ("padded", padded)):
                if (
                    not isinstance(values, (list, tuple))
                    or len(values) != 2
                    or any(type(value) is not int or value < 0 for value in values)
                ):
                    raise ValueError(
                        f"GLM idle DP requires two nonnegative CPU {name} counts"
                    )
            if (
                not forward_batch.dp_padding_mode.is_max_len()
                or padded[0] != padded[1]
                or any(real > capacity for real, capacity in zip(original, padded))
            ):
                raise ValueError("GLM idle DP requires matching MAX_LEN padded counts")
            counts = tuple(original)
    with nullcontext() if mlp_context is None else mlp_context:
        token = _idle_dp_counts.set(counts)
        try:
            yield
        finally:
            _idle_dp_counts.reset(token)


def _normal_skip_idle_dp_routes(selected, group, world, rank, tokens, topk):
    """Filter route offsets on their device; retain every positive-count domain."""
    counts = _idle_dp_counts.get()
    if counts is None:
        return selected, 0, None
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    ep = parallel.moe_ep_group
    tp_ranks = parallel.tp_group.ranks
    # With PP/CP/DCP/MoE-DP all one, EP and TP have the same ascending rank
    # order. Attention DP d owns eight consecutive source-rank payload blocks.
    # These are CPU group attributes: no device read or new collective.
    if (
        world != 16
        or ep.world_size != 16
        or ep.device_group is not group
        or len(tp_ranks) != 16
        or list(ep.ranks) != list(tp_ranks)
        or list(ep.ranks) != list(range(ep.ranks[0], ep.ranks[0] + 16))
        or ep.rank_in_group != rank
        or parallel.attn_dp_size != 2
        or parallel.attn_tp_size != 8
        or parallel.attn_dp_rank != rank // 8
        or parallel.attn_tp_rank != rank % 8
    ):
        raise RuntimeError("GLM idle DP requires the native DP2/TP8/EP16 rank mapping")
    before = selected.numel()
    span = 8 * tokens * topk
    for domain, count in enumerate(counts):
        if count == 0:
            start = domain * span
            selected = selected[(selected < start) | (selected >= start + span)]
    return selected, before - selected.numel(), counts


def _check_equal_dispatch_shapes(group, x, ids, weights, world):
    # DP4/TP2 and two-node DP2/TP8 force MAX_LEN padding; TP8 has equal rows
    # within its TP domain. Retain this check before fixed-size payload
    # gathers to catch any downstream geometry or routing-contract violation.
    shape = torch.tensor(
        [
            x.shape[0],
            x.shape[1],
            ids.ndim,
            ids.shape[0] if ids.ndim else -1,
            ids.shape[1] if ids.ndim > 1 else -1,
            weights.ndim,
            weights.shape[0] if weights.ndim else -1,
            weights.shape[1] if weights.ndim > 1 else -1,
        ],
        dtype=torch.int64,
        device=x.device,
    )
    gathered = shape.new_empty((world * shape.numel(),))
    dist.all_gather_into_tensor(gathered, shape, group=group)
    shapes = gathered.view(world, -1).cpu().tolist()
    expected = shapes[0]
    if (
        any(peer != expected for peer in shapes)
        or expected[2] != 2
        or expected[5] != 2
        or expected[0] != expected[3]
        or expected[0] != expected[6]
        or expected[4] != expected[7]
    ):
        raise RuntimeError(
            f"GLM HCCL normal EP{world} requires equal local rows/width/topk across "
            f"all ranks (MAX_LEN or equal SUM_LEN); dispatch shapes={shapes}"
        )


def _check_equal_dispatch_shapes_cached(group, x, ids, weights, world):
    """Opt-in shape audit cache with collective invalidation in two-node P.

    A local cache hit alone cannot skip a collective: a remote rank may have
    changed shape or lost its cache. Every rank first votes on invalidation,
    so all ranks either audit or skip together. This retains a scalar host
    synchronization; it saves the full shape gather/readback on cache hits.
    The idle-DP scope provides the existing eager/no-overlap P-only guard.
    """
    global _shape_cache_logged
    if (
        os.getenv("SGLANG_GLM53_NORMAL_SHAPE_CACHE", "0") != "1"
        or world != 16
        or _idle_dp_counts.get() is None
    ):
        _dispatch_shape_cache.pop(group, None)
        return _check_equal_dispatch_shapes(group, x, ids, weights, world)

    signature = (world,) + tuple(
        (tuple(t.shape), t.dtype, t.device) for t in (x, ids, weights)
    )
    dirty = torch.tensor(
        [int(_dispatch_shape_cache.get(group) != signature)],
        dtype=torch.int32,
        device=x.device,
    )
    dist.all_reduce(dirty, op=dist.ReduceOp.MAX, group=group)
    if dirty.item():
        # Never retain a prior success after a failed collective/audit.
        _dispatch_shape_cache.pop(group, None)
        _check_equal_dispatch_shapes(group, x, ids, weights, world)
        _dispatch_shape_cache[group] = signature
    if not _shape_cache_logged:
        logger.info(
            "GLM53 normal HCCL shape cache active: world=%d "
            "invalidation=all_reduce_MAX scope=two_node_prefill",
            world,
        )
        _shape_cache_logged = True


def log_normal_hccl_activation(quant_info, profile):
    """Log actual dispatch and expert storage once per process, after dispatch."""
    global _activation_logged, _bulk_activation_logged
    if _last_dispatch_metadata is None:
        return
    bulk = _last_dispatch_metadata[2] >= 512
    if _activation_logged and (not bulk or _bulk_activation_logged):
        return
    if any(
        weight.dtype != torch.int8
        for weight in (quant_info.w13_weight, quant_info.w2_weight)
    ):
        raise RuntimeError("GLM HCCL normal experts must retain INT8 weights")
    if any(
        scale.dtype != torch.float32
        for scale in (quant_info.w13_weight_scale, quant_info.w2_weight_scale)
    ):
        raise RuntimeError("GLM HCCL normal expert scales must remain FP32")
    rank, world, rows, width, x_dtype, q_dtype, scale_dtype = _last_dispatch_metadata
    logger.info(
        "GLM53 normal HCCL active: profile=%s rank=%d world=%d local_rows=%d gathered_rows=%d width=%d "
        "x_dtype=%s dispatch_dtype=%s token_scale_dtype=%s "
        "w13=%s/%s w2=%s/%s weight_scale_dtype=%s/%s "
        "reduce_scatter=%s deterministic_local_combine=%s ordered_local_combine=%s",
        profile,
        rank,
        world,
        rows,
        rows * world,
        width,
        x_dtype,
        q_dtype,
        scale_dtype,
        tuple(quant_info.w13_weight.shape),
        quant_info.w13_weight.dtype,
        tuple(quant_info.w2_weight.shape),
        quant_info.w2_weight.dtype,
        quant_info.w13_weight_scale.dtype,
        quant_info.w2_weight_scale.dtype,
        os.getenv("SGLANG_GLM53_NORMAL_REDUCE_SCATTER", "0"),
        os.getenv("SGLANG_GLM53_NORMAL_COMBINE_DETERMINISTIC", "0"),
        os.getenv("SGLANG_GLM53_NORMAL_COMBINE_ORDERED", "0"),
    )
    _activation_logged = True
    _bulk_activation_logged |= bulk


def _normal_ordered_combine_enabled(world, tokens, topk, width, dtype):
    """Retain legacy EP8 opt-in and bound the new two-node P-only mode."""
    mode = os.getenv("SGLANG_GLM53_NORMAL_COMBINE_ORDERED", "0")
    if mode == "2node":
        if world != 16 or _idle_dp_counts.get() is None:
            return False
        if os.getenv("SGLANG_GLM53_NORMAL_DEVICE_ROUTING", "0") == "1":
            raise ValueError("GLM ORDERED=2node requires NORMAL_DEVICE_ROUTING=0")
    elif mode == "1":
        if world != 8:
            raise ValueError("Ordered GLM combine mode 1 requires EP8")
    else:
        return False
    # Match ordered_combine's existing kernel bound before payload allocation.
    # EP16 admits at most 8191 local rows (131056 gathered rows), not 8192.
    if not (
        topk == 8
        and width == 4096
        and dtype == torch.bfloat16
        and 0 <= world * tokens <= 131070
    ):
        raise ValueError(
            "Ordered GLM combine requires topk8/BF16 H4096 and "
            "0 <= gathered rows <= 131070"
        )
    return True


def collective_dispatch(group, x, ids, weights, num_experts=288):
    global _last_dispatch_metadata, _idle_dp_logged, _device_routing_logged
    world, rank = dist.get_world_size(group), dist.get_rank(group)
    tokens, width = x.shape
    topk = ids.shape[1]
    ordered = _normal_ordered_combine_enabled(world, tokens, topk, width, x.dtype)
    if world in (8, 16):
        release_normal_hccl_cached_memory(x)
        _check_equal_dispatch_shapes_cached(group, x, ids, weights, world)

    def gather(t):
        out = t.new_empty((world * t.shape[0], *t.shape[1:]))
        dist.all_gather_into_tensor(out, t.contiguous(), group=group)
        return out

    # Row quantization is identical to quantizing the received BF16 rows.
    q, scale = torch.ops.npu.npu_dynamic_quant(x)
    if q.dtype != torch.int8 or scale.dtype != torch.float32:
        raise RuntimeError("GLM HCCL normal dispatch requires INT8 rows/FP32 scales")
    all_q, all_scale, all_ids, all_weights = map(gather, (q, scale, ids, weights))
    # The scope is published only for EXTEND in the eager, non-overlap
    # two-node P profile. Keep all other normal profiles on the CPU reference
    # path, even if the flag is inherited by another worker (including D).
    device_routing = (
        os.getenv("SGLANG_GLM53_NORMAL_DEVICE_ROUTING", "0") == "1"
        and world == 16
        and _idle_dp_counts.get() is not None
    )
    flat = (all_ids if device_routing else all_ids.cpu()).flatten()
    assert num_experts % world == 0
    experts = num_experts // world
    low = rank * experts
    selected = ((flat >= low) & (flat < low + experts)).nonzero().flatten()
    discarded, idle_counts = 0, None
    if os.getenv("SGLANG_GLM53_NORMAL_SKIP_IDLE_DP", "0") == "1":
        selected, discarded, idle_counts = _normal_skip_idle_dp_routes(
            selected, group, world, rank, tokens, topk
        )
    order = torch.argsort(flat[selected], stable=True)
    selected = selected[order]
    counts = torch.bincount(flat[selected] - low, minlength=experts)
    if not device_routing:
        counts = counts.tolist()
    if idle_counts is not None and 0 in idle_counts and not _idle_dp_logged:
        logger.info(
            "GLM53 normal HCCL skip idle DP: rank=%d world=%d valid_dp_counts=%s "
            "discarded_padded_routes=%d countsum=%d",
            rank,
            world,
            idle_counts,
            discarded,
            selected.numel(),
        )
        _idle_dp_logged = True
    loc = selected.to(device=x.device, dtype=torch.int64)
    rows = torch.div(loc, topk, rounding_mode="floor")
    if selected.numel():
        recv = all_q.index_select(0, rows)
        recv_scale = all_scale.index_select(0, rows)
    else:
        # Match DeepEP's dummy allocation for a rank with no received tokens.
        recv = q.new_zeros((1, width))
        recv_scale = scale.new_ones((1,))
    chosen_weights = all_weights.flatten().index_select(0, loc)
    state = (rows, chosen_weights, tokens, world, width, int(selected.numel()))
    if ordered:
        # selected is already CPU-resident in stable expert order. No extra
        # device synchronization: construct a small inverse map during routing.
        indices = _normal_combine_ordered_indices(selected // topk, world * tokens, topk)
        state += (indices.to(device=x.device),)
    _last_dispatch_metadata = (
        rank,
        world,
        tokens,
        width,
        x.dtype,
        q.dtype,
        scale.dtype,
    )
    if device_routing and not _device_routing_logged:
        logger.info(
            "GLM53 normal HCCL device routing active: rank=%d world=%d "
            "stable_expert_order=True counts_device=%s counts_dtype=%s",
            rank,
            world,
            counts.device,
            counts.dtype,
        )
        _device_routing_logged = True
    return recv, recv_scale, counts, state


def _normal_combine_ordered_indices(rows, count, topk):
    """CPU [token, contribution] map, preserving stable expert order exactly."""
    order = torch.argsort(rows, stable=True)
    sorted_rows = rows[order]
    counts = torch.bincount(rows, minlength=count)
    if counts.numel() > count or (counts.numel() and int(counts.max()) > topk):
        raise ValueError("Invalid GLM route multiplicity")
    offsets = counts.cumsum(0) - counts
    slot = torch.arange(rows.numel(), dtype=torch.int64) - offsets[sorted_rows]
    indices = torch.full((count, topk), -1, dtype=torch.int32)
    indices[sorted_rows, slot] = order.to(torch.int32)
    return indices


def _normal_combine_index_add_(out, rows, values):
    """Opt into deterministic local accumulation without changing HCCL mode.

    Repeated destination rows otherwise use nondeterministic FP32 accumulation
    on Ascend. This switch is for the eager, non-overlap normal-phase path;
    decode's DeepEP LL path does not call it. Restore both process flags even
    when the backend raises, before any collective or subsequent model op.
    """
    if os.getenv("SGLANG_GLM53_NORMAL_COMBINE_DETERMINISTIC", "0") != "1":
        return out.index_add_(0, rows, values)
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        return out.index_add_(0, rows, values)
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def _check_normal_reduce_scatter(group, local_out, global_out, tokens):
    """Debug-only FP32 comparison against the original all-reduce/narrow.

    Every EP rank must enable the check together. This deliberately performs
    both collectives and a synchronized failure vote; disable it for timing.
    Reduction trees may differ, so use an explicit FP32 tolerance instead of
    requiring bitwise equality. This is not an end-to-end accuracy check.
    """
    global _reduce_scatter_check_logged
    dist.all_reduce(global_out, group=group)
    rank = dist.get_rank(group)
    reference = global_out.narrow(0, rank * tokens, tokens)
    failed = (~torch.isclose(local_out, reference, rtol=1e-5, atol=1e-5).all()).to(
        dtype=torch.int32
    ).reshape(1)
    dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=group)
    if failed.item():
        max_abs = (local_out - reference).abs().max().item()
        raise RuntimeError(
            "GLM normal reduce_scatter failed FP32 all_reduce reference check "
            f"on at least one EP rank; local rank={rank} max_abs={max_abs} "
            "rtol=1e-5 atol=1e-5"
        )
    if not _reduce_scatter_check_logged:
        logger.info(
            "GLM53 normal reduce_scatter FP32 reference check passed: "
            "rtol=1e-5 atol=1e-5 scope=two_node_prefill"
        )
        _reduce_scatter_check_logged = True


def collective_combine(group, y, state):
    global _ordered_combine_logged
    rows, weights, tokens, world, width, received = state[:6]
    # Sum each rank's expert contributions in FP32, then reduce them across EP.
    # This keeps the workspace bounded by tokens*hidden, rather than topk times
    # that size. The final activation is BF16; expert products remain W8A8.
    if len(state) == 7:
        from .glm53_ordered_combine import ordered_combine

        out = ordered_combine(y[:received], weights, state[6], world * tokens, width)
        if not _ordered_combine_logged:
            logger.info(
                "GLM53 normal ordered combine active: world=%d rows=%d "
                "accumulator=FP32 stable_expert_order=True",
                world,
                world * tokens,
            )
            _ordered_combine_logged = True
    else:
        out = torch.zeros((world * tokens, width), device=y.device, dtype=torch.float32)
        if received:
            _normal_combine_index_add_(out, rows, y[:received].float() * weights[:, None])
    reduce_scatter_mode = os.getenv("SGLANG_GLM53_NORMAL_REDUCE_SCATTER", "0")
    if (
        (
            reduce_scatter_mode == "1"  # Preserve the existing explicit mode.
            or (
                reduce_scatter_mode == "2node"
                and world == 16
                and _idle_dp_counts.get() is not None
            )
        )
        and world > 1
        and tokens > 0
    ):
        # all_gather packed contiguous token blocks in group-rank order. Each
        # rank only needs its block of the FP32 sum. Local accumulation is
        # unchanged; HCCL may use a different reduction tree and rounding order.
        local_out = torch.empty((tokens, width), device=y.device, dtype=torch.float32)
        dist.reduce_scatter_tensor(local_out, out, op=dist.ReduceOp.SUM, group=group)
        if (
            os.getenv("SGLANG_GLM53_NORMAL_REDUCE_SCATTER_CHECK", "0") == "1"
            and world == 16
            and _idle_dp_counts.get() is not None
        ):
            _check_normal_reduce_scatter(group, local_out, out, tokens)
        return local_out.to(y.dtype)
    dist.all_reduce(out, group=group)
    rank = dist.get_rank(group)
    return out.narrow(0, rank * tokens, tokens).to(y.dtype)
