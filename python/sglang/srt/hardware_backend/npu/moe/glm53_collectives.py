"""GLM normal-phase INT8 routing through standard HCCL collectives.

Retains the verified fallback for vendor dynamic-receive-size failures. This
path is explicitly enabled for GLM W8A8 layers; decode still uses DeepEP LL.
"""

import logging
import os

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)
_last_dispatch_metadata = None
_activation_logged = False
_bulk_activation_logged = False


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


def collective_dispatch(group, x, ids, weights, num_experts=288):
    global _last_dispatch_metadata
    world, rank = dist.get_world_size(group), dist.get_rank(group)
    tokens, width = x.shape
    topk = ids.shape[1]
    if world in (8, 16):
        release_normal_hccl_cached_memory(x)
        _check_equal_dispatch_shapes(group, x, ids, weights, world)

    def gather(t):
        out = t.new_empty((world * t.shape[0], *t.shape[1:]))
        dist.all_gather_into_tensor(out, t.contiguous(), group=group)
        return out

    # Row quantization is identical to quantizing the received BF16 rows.
    q, scale = torch.ops.npu.npu_dynamic_quant(x)
    if q.dtype != torch.int8 or scale.dtype != torch.float32:
        raise RuntimeError("GLM HCCL normal dispatch requires INT8 rows/FP32 scales")
    all_q, all_scale, all_ids, all_weights = map(gather, (q, scale, ids, weights))
    flat = all_ids.cpu().flatten()
    assert num_experts % world == 0
    experts = num_experts // world
    low = rank * experts
    selected = ((flat >= low) & (flat < low + experts)).nonzero().flatten()
    order = torch.argsort(flat[selected], stable=True)
    selected = selected[order]
    counts = torch.bincount(flat[selected] - low, minlength=experts).tolist()
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
    if os.getenv("SGLANG_GLM53_NORMAL_COMBINE_ORDERED", "0") == "1":
        if not (world == 8 and topk == 8 and width == 4096 and x.dtype == torch.bfloat16):
            raise ValueError("Ordered GLM combine is validated only for EP8/topk8/BF16 H4096")
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


def collective_combine(group, y, state):
    rows, weights, tokens, world, width, received = state[:6]
    # Sum each rank's expert contributions in FP32, then reduce them across EP.
    # This keeps the workspace bounded by tokens*hidden, rather than topk times
    # that size. The final activation is BF16; expert products remain W8A8.
    if len(state) == 7:
        from .glm53_ordered_combine import ordered_combine

        out = ordered_combine(y[:received], weights, state[6], world * tokens, width)
    else:
        out = torch.zeros((world * tokens, width), device=y.device, dtype=torch.float32)
        if received:
            _normal_combine_index_add_(out, rows, y[:received].float() * weights[:, None])
    if (
        os.getenv("SGLANG_GLM53_NORMAL_REDUCE_SCATTER", "0") == "1"
        and world > 1
        and tokens > 0
    ):
        # all_gather packed contiguous token blocks in group-rank order. Each
        # rank only needs its block of the FP32 sum. Local accumulation is
        # unchanged; HCCL may use a different reduction tree and rounding order.
        local_out = torch.empty((tokens, width), device=y.device, dtype=torch.float32)
        dist.reduce_scatter_tensor(local_out, out, op=dist.ReduceOp.SUM, group=group)
        return local_out.to(y.dtype)
    dist.all_reduce(out, group=group)
    rank = dist.get_rank(group)
    return out.narrow(0, rank * tokens, tokens).to(y.dtype)
