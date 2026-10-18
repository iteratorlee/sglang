"""Physical-capacity supplement; the existing fixture and oracle are unchanged."""

from dataclasses import dataclass

import torch

from glm53_index_batch_fixture import fixture


def compact_cases():
    """Twenty fixtures: physical capacity, topk threshold, mixed lengths, padding."""
    mixed = {
        2: [127, 2044],
        4: [63, 252, 2048, 2052],
        8: [32, 127, 252, 1023, 2044, 2048, 2052, 4095],
        13: [131068 + row % 4 for row in range(13)],
    }
    specs = [([p], "random") for p in (127, 252, 2044, 2048, 2052)]
    specs += [(s, "random") for s in mixed.values()]
    specs += [(s, "ties") for s in ([2052], mixed[4], mixed[13])]
    specs += [(s, "near-ties") for s in ([2048], mixed[2], mixed[8], mixed[13])]
    specs += [(mixed[13], "closing")]
    specs += [(mixed[b], "mixed-padding") for b in (2, 8, 13)]
    result = []
    for serial, (starts, pattern) in enumerate(specs):
        bs = len(starts)
        requests = list(range(1, bs + 1))
        if serial % 2:
            requests.reverse()
        if pattern == "mixed-padding":
            requests[-1] = 0
        result.append(
            dict(
                id=f"{serial:02d}-bs{bs}-{pattern}",
                starts=starts.copy(),
                requests=requests,
                pattern=pattern,
                seed=53 + serial,
            )
        )
    return result


@dataclass(frozen=True)
class PhysicalLayout:
    key_blocks: int = 29049
    pooled_columns: int = 519
    full_columns: int | None = None

    @property
    def raw_columns(self):
        # Default full width is 4 * LI width; --full-table-width 2074 also gives 519.
        return (
            self.full_columns
            if self.full_columns is not None
            else 4 * self.pooled_columns
        )

    def validate(self, bs, starts):
        if bs < 1 or len(starts) != bs or min(starts) < 0:
            raise ValueError("nonempty, nonnegative positions required")
        if self.key_blocks < 2 or self.pooled_columns < 1:
            raise ValueError("capacity must include a reserved sink and live blocks")
        if self.raw_columns < 1 or (self.raw_columns + 3) // 4 != self.pooled_columns:
            raise ValueError(
                "full table width must ceil-divide by four to pooled width"
            )
        # Match the original fixture's spare pages, without tying physical capacity
        # or the LI table's width to the history length.
        logical_pages = (max(starts) + 4) // 256 + 2
        if logical_pages > self.pooled_columns:
            raise ValueError("logical history plus spare pages exceeds table width")
        if bs * logical_pages >= self.key_blocks:
            raise ValueError("physical cache cannot hold disjoint live request pages")
        return logical_pages

    def shapes(self, bs):
        return dict(
            key=[self.key_blocks, 64, 1, 128],
            full_block_table=[bs, self.raw_columns],
            legacy_li_table=[bs, self.pooled_columns],
            batched_li_table=[bs * 4, self.pooled_columns],
            key_bytes=self.key_blocks * 64 * 128 * 2,
        )


def physical_fixture(
    starts,
    *,
    layout=PhysicalLayout(),
    requests=None,
    kind="random",
    topk=2048,
    heads=32,
    extra=0,
    seed=53,
):
    """Embed unchanged logical data into scattered high/low physical block IDs.

    Unmapped storage is finite poison, not uninitialized memory. Full-cache byte
    comparisons include that storage and sink zero. Positive input keys retain
    the old fixture's deterministic shared-sink contract. This is synthetic data,
    not a reproduction of the failed GSM tensors.
    """
    bs = len(starts)
    logical_pages = layout.validate(bs, starts)
    case = fixture(
        starts,
        requests=requests,
        kind=kind,
        topk=topk,
        heads=heads,
        extra=extra,
        seed=seed,
    )
    logical = case.data["cache"]
    if logical.shape[0] != bs * logical_pages + 1:
        raise AssertionError("upstream logical fixture capacity changed")
    gen = torch.Generator().manual_seed(seed + 100003)
    order = torch.randperm(layout.key_blocks - 1, generator=gen) + 1
    # Always exercise the last valid physical block, even at short GSM lengths.
    last = int((order == layout.key_blocks - 1).nonzero()[0])
    order[0], order[last] = order[last].clone(), order[0].clone()
    remap = torch.cat((torch.zeros(1, dtype=torch.long), order[: len(logical) - 1]))
    cache = torch.full((layout.key_blocks, 64, 1, 128), 32.0, dtype=logical.dtype)
    cache.index_copy_(0, remap, logical)
    table = torch.full((bs, layout.raw_columns), -1, dtype=torch.int32)
    old_table = case.data["table"]
    width = min(table.shape[1], old_table.shape[1])
    old_part = old_table[:, :width]
    table[:, :width] = torch.where(
        old_part >= 0, remap[old_part.clamp_min(0).long()].int(), -1
    )
    # Force one currently visible/closing page onto the capacity endpoint.
    # Swapping physical blocks AND every reference preserves the logical data.
    live = (case.data["req"] > 0).nonzero().flatten()
    if live.numel():
        row = int(live[0])
        first = int(table[row, 0])
        high = layout.key_blocks - 1
        tmp = cache[first].clone()
        cache[first].copy_(cache[high])
        cache[high].copy_(tmp)
        old = table.clone()
        table[old == first] = high
        table[old == high] = first
    case.data["cache"], case.data["table"] = cache, table
    case.physical_layout = layout
    case.physical_summary = dict(
        **layout.shapes(bs),
        logical_pages_per_request=logical_pages,
        page_seed=seed + 100003,
        live_requests=int((case.data["req"] > 0).sum()),
        max_page=int(table.max()),
        unused_cache_fill=32.0,
    )
    return case
