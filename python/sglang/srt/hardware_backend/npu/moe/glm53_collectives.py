"""GLM normal-phase INT8 routing through standard HCCL collectives.

Retains the verified fallback for vendor dynamic-receive-size failures. This
path is explicitly enabled for GLM W8A8 layers; decode still uses DeepEP LL.
"""

import os

import torch
import torch.distributed as dist


def collective_dispatch(group, x, ids, weights, num_experts=288):
    world, rank = dist.get_world_size(group), dist.get_rank(group)
    tokens, width = x.shape
    topk = ids.shape[1]

    def gather(t):
        out = t.new_empty((world * t.shape[0], *t.shape[1:]))
        dist.all_gather_into_tensor(out, t.contiguous(), group=group)
        return out

    # Row quantization is identical to quantizing the received BF16 rows.
    q, scale = torch.ops.npu.npu_dynamic_quant(x)
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
    return recv, recv_scale, counts, state


def collective_combine(group, y, state):
    rows, weights, tokens, world, width, received = state
    # Sum each rank's expert contributions in FP32, then reduce them across EP.
    # This keeps the workspace bounded by tokens*hidden, rather than topk times
    # that size. The final activation is BF16; expert products remain W8A8.
    out = torch.zeros((world * tokens, width), device=y.device, dtype=torch.float32)
    if received:
        out.index_add_(0, rows, y[:received].float() * weights[:, None])
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
