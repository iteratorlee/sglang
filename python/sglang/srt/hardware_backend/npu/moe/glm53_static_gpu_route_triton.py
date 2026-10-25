"""Integer stable expert-major compaction, 18 experts x 1024-route blocks.

Four stages: block counts; device prefix; stable scatter/inverse; raw gather.
No atomics, nonzero, floating sort, host counts, or dynamic output allocation.
NPU compilation and full FFN parity are deliberately NOT presumed.
"""
import triton
import triton.language as tl


@triton.jit
def _count(IDS, PART, ROUTES, DOMAIN_SPAN, LOW, ACTIVE,
           BLOCKS: tl.constexpr, B: tl.constexpr):
    expert, block = tl.program_id(0), tl.program_id(1)
    loc = block * B + tl.arange(0, B)
    present = loc < ROUTES
    # INT64 IDs remain INT64 through comparison. Extreme values cannot wrap.
    value = tl.load(IDS + loc, present, other=-1)
    domain = loc // DOMAIN_SPAN
    active = ((ACTIVE >> domain) & 1) != 0
    match = present & active & (value == LOW + expert)
    tl.store(PART + expert * BLOCKS + block, tl.sum(match.to(tl.int32), 0))


@triton.jit
def _prefix(PART, OFFSET, COUNTS, TOTAL, BLOCKS: tl.constexpr,
            BB: tl.constexpr, EE: tl.constexpr):
    expert = tl.arange(0, EE)
    block = tl.arange(0, BB)
    c = tl.load(PART + expert[:, None] * BLOCKS + block[None, :],
                (expert[:, None] < 18) & (block[None, :] < BLOCKS), 0)
    block_ends = tl.cumsum(c, 1)
    counts = tl.sum(c, 1)
    expert_starts = tl.cumsum(counts, 0) - counts
    offsets = expert_starts[:, None] + block_ends - c
    tl.store(OFFSET + expert[:, None] * BLOCKS + block[None, :], offsets,
             (expert[:, None] < 18) & (block[None, :] < BLOCKS))
    tl.store(COUNTS + expert, counts.to(tl.int64), expert < 18)
    tl.store(TOTAL, tl.sum(counts, 0))


@triton.jit
def _compact(IDS, OFFSET, SELECTED, INVERSE, ROUTES, DOMAIN_SPAN, LOW, ACTIVE,
             BLOCKS: tl.constexpr, B: tl.constexpr):
    expert, block = tl.program_id(0), tl.program_id(1)
    # B=1024 contains 128 whole rows. Load each row's eight full-width IDs once.
    row = block * (B // 8) + tl.arange(0, B // 8)
    slot = tl.arange(0, 8)
    loc = row[:, None] * 8 + slot[None, :]
    present = loc < ROUTES
    value = tl.load(IDS + loc, present, other=-1)
    active = ((ACTIVE >> ((row * 8) // DOMAIN_SPAN)) & 1) != 0
    local = present & active[:, None] & (value >= LOW) & (value < LOW + 18)
    # Select the upper sentinel BEFORE narrowing; INT64 extremes never wrap
    # into local IDs. All selected values lie in [LOW, LOW+18] <=288.
    safe = tl.where(local, value, LOW + 18).to(tl.int32)
    match = safe == LOW + expert
    hits = match.to(tl.int32)
    within_row = tl.cumsum(hits, 1)
    row_count = tl.sum(hits, 1)
    row_start = tl.cumsum(row_count, 0) - row_count
    start = tl.load(OFFSET + expert * BLOCKS + block)
    # Inclusive scans over 8 slots and 128 row totals reproduce the flat1024
    # inclusive scan exactly: earlier rows, then earlier slots in this row.
    dest = start + row_start[:, None] + within_row - 1
    # A route appears once; disjoint expert/block intervals make stores unique.
    tl.store(SELECTED + dest, loc, match)
    # For a matched slot, its value is the expert constant. Count all smaller
    # local IDs, then equal IDs in earlier slots. Sentinel is larger than every
    # local expert, so invalid/idle/past-end lanes contribute neither count.
    less_in_row = tl.sum((safe < LOW + expert).to(tl.int32), 1)
    earlier = less_in_row[:, None] + within_row - 1
    tl.store(INVERSE + row[:, None] * 8 + earlier, dest, match)


@triton.jit
def _gather(Q, S, W, SELECTED, TOTAL, RQ, RS, ROWS, GATES,
            WORDS: tl.constexpr):
    index = tl.program_id(0)
    valid = index < tl.load(TOTAL)
    loc = tl.load(SELECTED + index, valid, other=0)
    row = loc // 8
    col = tl.arange(0, WORDS)
    # Pointers are reinterpreted to INT32 at launch: NaN payloads / signed zero
    # / subnormals are moved as bits, without FP conversion or arithmetic.
    q = tl.load(Q + row * WORDS + col, valid, other=0)
    s = tl.load(S + row, valid, other=0)
    gate = tl.load(W + loc, valid, other=0)
    tl.store(RQ + index * WORDS + col, q)
    tl.store(RS + index, s)
    tl.store(GATES + index, gate)
    tl.store(ROWS + index, tl.where(valid, row, 0).to(tl.int64))


def route(q, scale, ids, weights, partial, offsets, counts, total, selected,
          inverse, recv, recv_scale, rows, gates, g, low):
    options = dict(num_warps=4, multibuffer=False, enable_fp_fusion=False)
    grid = (18, g.blocks)
    args = (g.world_rows * 8, g.local_rows * 8 * 8, low, g.active_mask)
    _count[grid](ids, partial, *args, g.blocks, 1024, **options)
    _prefix[(1,)](partial, offsets, counts, total, g.blocks,
                  triton.next_power_of_2(g.blocks), 32, **options)
    _compact[grid](ids, offsets, selected, inverse, *args, g.blocks, 1024, **options)
    _gather[(g.capacity,)](
        triton.reinterpret(q, tl.int32), triton.reinterpret(scale, tl.int32),
        triton.reinterpret(weights, tl.int32), selected, total,
        triton.reinterpret(recv, tl.int32), triton.reinterpret(recv_scale, tl.int32),
        rows, triton.reinterpret(gates, tl.int32), 1024, **options)
