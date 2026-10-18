"""Defined-output contract, opt-in only; the strict v1 comparator is unchanged.

Zero LI key length is not zero FA length for active positions 0..2: expansion
places the causal tail first. FA consumes first min(seq, index_topk + tail)
indices. Dead requests have zero FA KV length. See the accompanying opt note
for source evidence and the limits of this locally imposed contract.
"""

import torch

from glm53_index_batch_fixture import STATE, exact


def differences(a, b):
    a, b = a.cpu().contiguous(), b.cpu().contiguous()
    if a.shape != b.shape or a.dtype != b.dtype:
        raise AssertionError("diagnostic shape/dtype mismatch")
    return dict(
        elements=int((a != b).sum()),
        bytes=int((a.view(torch.uint8) != b.view(torch.uint8)).sum()),
    )


def compare_outputs(
    case,
    raw_a,
    out_a,
    raw_b,
    out_b,
    *,
    canonical_a=False,
    canonical_b=True,
    diagnostics,
):
    """Same masks for A/B and graph/eager; retain counts before assertions."""
    bs, width = case.bs, case.topk
    raw_a, raw_b = raw_a.cpu(), raw_b.cpu()
    out_a, out_b = out_a.cpu(), out_b.cpu()
    if raw_a.shape != (bs, 4, width // 4) or raw_b.shape != raw_a.shape:
        raise AssertionError("raw index shape")
    expected_shape = (case.data["positions"].numel(), width + 3)
    if out_a.shape != expected_shape or out_b.shape != expected_shape:
        raise AssertionError("expanded index shape")
    positions = case.data["positions"][: bs * 4].cpu().reshape(bs, 4)
    live = (case.data["req"].cpu() > 0)[:, None].expand(bs, 4)
    lengths = torch.where(live, (positions + 1) // 4, 0)
    positive, zero, dead = live & (lengths > 0), lengths == 0, ~live
    aa, bb = out_a[: bs * 4].reshape(bs, 4, -1), out_b[: bs * 4].reshape(bs, 4, -1)
    diagnostics.update(
        live_positive_rows=int(positive.sum()),
        active_zero_rows=int((live & zero).sum()),
        dead_rows=int(dead.sum()),
        dead_raw=differences(raw_a[dead], raw_b[dead]),
        dead_expanded=differences(aa[dead], bb[dead]),
        active_zero_raw=differences(raw_a[live & zero], raw_b[live & zero]),
        active_zero_unconsumed=differences(
            aa[live & zero][:, 3:], bb[live & zero][:, 3:]
        ),
    )
    # Require the entire positive-length row, including order and unused slots.
    exact(raw_a[positive], raw_b[positive], "positive-key raw")
    exact(aa[positive], bb[positive], "positive-key expanded")
    for raw, expanded, canonical in (
        (raw_a, aa, canonical_a),
        (raw_b, bb, canonical_b),
    ):
        if canonical:
            exact(raw[zero], torch.full_like(raw[zero], -1), "zero-key canonical -1")
        for row in range(bs):
            for step in range(4):
                seq = int(positions[row, step]) + 1
                length = int(lengths[row, step])
                if live[row, step] and length > 0:
                    ids = raw[row, step]
                    valid = ids[ids >= 0]
                    if (
                        len(valid) != min(length, width // 4)
                        or len(valid.unique()) != len(valid)
                        or (valid >= length).any()
                    ):
                        raise AssertionError("duplicate, future, or missing pool index")
                if zero[row, step] and canonical:
                    # Independent oracle for every new zero-key expanded slot,
                    # including dead rows and unconsumed padding.
                    expected = torch.full_like(expanded[row, step], seq)
                    tail = seq % 4
                    at = min(seq - tail, width)
                    expected[at : at + tail] = torch.arange(
                        seq - tail, seq, dtype=expected.dtype
                    )
                    exact(expanded[row, step], expected, "canonical zero-key expansion")
                if live[row, step]:
                    tail = seq % 4
                    at = min(seq - tail, width)
                    exact(
                        expanded[row, step, at : at + tail],
                        torch.arange(seq - tail, seq, dtype=expanded.dtype),
                        "causal tail",
                    )
    # For active zero-key rows, all three tail slots are defined by expansion;
    # the first seq (1..3) are the indices FA consumes. Do not compare LI garbage.
    exact(
        aa[live & zero][:, :3],
        bb[live & zero][:, :3],
        "defined short-prefix tail slots",
    )
    for out in (out_a, out_b):
        exact(out[bs * 4 :], torch.zeros_like(out[bs * 4 :]), "shape padding")


def snapshot(runner, output):
    """Materialize before resetting/replaying; no device alias survives."""
    return dict(
        raw=runner.raw_topk().detach().cpu().clone(),
        output=output.detach().cpu().clone(),
        states=[
            {k: v.detach().cpu().clone() for k, v in s.items()} for s in runner.states
        ],
        final={k: runner.data[k].detach().cpu().clone() for k in (*STATE, "cache")},
        expands=[shape for shape, _ in runner.expands],
    )


def compare_snapshots(
    case,
    a,
    b,
    *,
    canonical_a=False,
    canonical_b=True,
    batch_expand_a=False,
    batch_expand_b=False,
    diagnostics,
):
    compare_outputs(
        case,
        a["raw"],
        a["output"],
        b["raw"],
        b["output"],
        canonical_a=canonical_a,
        canonical_b=canonical_b,
        diagnostics=diagnostics,
    )
    if len(a["states"]) != 9 or len(b["states"]) != 9:
        raise AssertionError("expected all nine pre/post/final state snapshots")
    for label, left, right in [
        *[
            (f"snapshot {i}", sa, sb)
            for i, (sa, sb) in enumerate(zip(a["states"], b["states"]))
        ],
        ("final", a["final"], b["final"]),
    ]:
        if set(left) != set((*STATE, "cache")) or set(right) != set(left):
            raise AssertionError("missing state/cache buffer")
        for key in left:
            exact(
                left[key], right[key], f"{label} {key} (all bytes, including dead/sink)"
            )
    for side, batched in ((a, batch_expand_a), (b, batch_expand_b)):
        expected = (
            [(case.bs * 4, case.topk // 4)]
            if batched
            else [(case.bs, case.topk // 4)] * 4
        )
        if side["expands"] != expected:
            raise AssertionError("expand shape/order changed")
