#!/usr/bin/env python3
"""P-only experiment harness; replace only legacy relu_ and FP32-gate mul_."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from bench import measure_pair, memory_sample
from score_contract import (
    REL,
    FAMILIES,
    Capture,
    boundary_metrics,
    compare_ids,
    compare_scores,
    fixture,
    load_file,
    make_scorer,
    source_receipt,
)


class WeightedProbe:
    def __init__(self, operation):
        self.operation = operation
        self.elements = self.different = self.calls = 0
        self.max_abs = 0.0

    def __call__(self, scores, weights):
        reference = scores.clone().relu_().mul_(weights.unsqueeze(-1))
        active = self.operation(scores, weights)
        # Compare every stored bit on the NPU. A scalar count avoids transferring
        # GiB per case to the shared host; copy values only to diagnose failures.
        different = int(
            (reference.view(torch.int16) != scores.view(torch.int16)).sum().item()
        )
        self.elements += scores.numel()
        self.different += different
        self.calls += int(active)
        if different:
            x, y = reference.cpu(), scores.cpu()
            finite = torch.isfinite(x) & torch.isfinite(y)
            if finite.any():
                self.max_abs = max(
                    self.max_abs,
                    float((x.float()[finite] - y.float()[finite]).abs().max()),
                )
        return active

    def result(self, expected_calls):
        return dict(
            elements=self.elements,
            different_bits=self.different,
            max_abs=self.max_abs,
            active_calls=self.calls,
            expected_active_calls=expected_calls,
            exact=self.different == 0 and self.calls == expected_calls,
        )


def special_values(operation):
    """IEEE edge patterns, BF16 NaN payloads, partial blocks and signed gates."""
    patterns = torch.tensor(
        [0, -32768, 16256, -16512, 32640, -128, 32704, 32641, -64, 1, -32767, 32639],
        dtype=torch.int16,
    )
    gates = torch.tensor(
        [
            0.0,
            -0.0,
            1.0,
            -1.0,
            1.00390625,
            1.00390637,
            2**-126,
            -(2**-126),
            float("inf"),
            -float("inf"),
            float("nan"),
        ]
    )
    results = []
    for count in (63, 2049, 8193):
        x = patterns.repeat((2 * 32 * count + len(patterns) - 1) // len(patterns))[
            : 2 * 32 * count
        ]
        x = x.view(torch.bfloat16).reshape(2, 32, count).to("npu:0")
        w = gates.repeat(6)[:64].reshape(2, 32).to("npu:0")
        input_host, gate_host = x.cpu(), w.cpu()
        relu_host = x.clone().relu_().cpu()
        reference = x.clone().relu_().mul_(w.unsqueeze(-1))
        active = operation(x, w)
        a, b = reference.cpu(), x.cpu()
        bad = a.view(torch.int16) != b.view(torch.int16)
        nan_both = torch.isnan(a) & torch.isnan(b)
        zero_both = (a == 0) & (b == 0)
        examples = []
        for row, head, key in bad.nonzero()[:12].tolist():
            examples.append(
                dict(
                    row=row,
                    head=head,
                    key=key,
                    input_bits=int(input_host.view(torch.int16)[row, head, key]),
                    gate_bits=int(gate_host.view(torch.int32)[row, head]),
                    relu_bits=int(relu_host.view(torch.int16)[row, head, key]),
                    old_bits=int(a.view(torch.int16)[row, head, key]),
                    new_bits=int(b.view(torch.int16)[row, head, key]),
                )
            )
        results.append(
            dict(
                keys=count,
                active=active,
                different_bits=int(bad.sum()),
                nan_payload_only=int((bad & nan_both).sum()),
                zero_sign_only=int((bad & zero_both).sum()),
                finite_nonzero_differences=int(
                    (bad & torch.isfinite(a) & torch.isfinite(b) & ~zero_both).sum()
                ),
                nan_semantic_differences=int((torch.isnan(a) != torch.isnan(b)).sum()),
                examples=examples,
                exact=active and not bool(bad.any()),
            )
        )
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--baseline-repo", type=Path)
    p.add_argument(
        "--integrated",
        action="store_true",
        help="Execute the actual default-off auto branch without replacing its AST",
    )
    p.add_argument("--case", action="append")
    p.add_argument("--family", choices=FAMILIES, action="append")
    p.add_argument("--seed", type=int, action="append")
    p.add_argument("--tile", type=int, choices=(8, 32), default=32)
    p.add_argument("--block", type=int, choices=(2048, 4096, 8192), action="append")
    p.add_argument("--repeat", type=int, default=9)
    p.add_argument("--special-values", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.repeat < 0:
        p.error("repeat must be nonnegative")
    blocks = args.block or ([8192] if args.integrated else [2048, 4096, 8192])
    if args.integrated and blocks != [8192]:
        p.error("The production auto branch uses exactly BLOCK8192")
    if os.getenv("ASCEND_RT_VISIBLE_DEVICES") != "7":
        p.error("Require authorized private physical device 7")
    import torch_npu
    import triton

    torch.set_num_threads(1)
    torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)
    torch_npu.npu.config.allow_internal_format = True
    source = args.repo / REL / "pooled_score_postop_npu.py"
    module = load_file(source, "isolated_postop")
    metadata = dict(
        torch=torch.__version__,
        torch_npu=torch_npu.__version__,
        triton=triton.__version__,
        visible="7",
        jit_compile=False,
        allow_internal_format_requested=True,
        candidate_hashes=source_receipt(args.repo),
        prototype_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        harness_hashes={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
        },
        scope=(
            "actual opt-in scorer branch"
            if args.integrated
            else "isolated pointwise substitution"
        ),
        baseline_hashes=source_receipt(args.baseline_repo or args.repo),
    )
    failed = False
    with args.output.open("x") as out:
        out.write(json.dumps(dict(metadata=metadata)) + "\n")
        out.flush()
        for block in blocks:
            operation = lambda scores, weights: module.relu_gate_(
                scores, weights, block
            )
            if args.special_values:
                special = dict(block=block, special_values=special_values(operation))
                out.write(json.dumps(special) + "\n")
                out.flush()
                print(json.dumps(special), flush=True)
                failed |= not all(x["exact"] for x in special["special_values"])
            for case in args.case or ["262144:128", "524288:128"]:
                seq, rows = map(int, case.split(":"))
                if not 0 < rows <= seq:
                    raise ValueError("Require 0 < rows <= sequence length")
                # This microspace has ~60 GiB free; use a reserve for the native
                # baseline and validation clone, never allocate on a model worker.
                required = (
                    2**31
                    + rows * 32 * 128 * 2
                    + (seq // 4) * 128 * 2
                    + args.tile * 32 * (seq // 4) * 16
                )
                if torch.npu.mem_get_info()[0] < required:
                    raise RuntimeError("Insufficient free HBM for private validation")
                for family in args.family or ["random"]:
                    for seed in args.seed or [1012]:
                        q, w, k = [
                            x.to("npu:0") for x in fixture(family, rows, seq // 4, seed)
                        ]
                        before, after = Capture(), Capture()
                        probe = WeightedProbe(operation)
                        old = make_scorer(
                            args.baseline_repo or args.repo,
                            False,
                            args.tile,
                            observer=before,
                        )
                        new = make_scorer(
                            args.repo,
                            False,
                            args.tile,
                            observer=after,
                            postop=probe,
                            integrated=args.integrated,
                        )
                        x = old(q, w, [k], [rows], [seq])
                        y = new(q, w, [k], [rows], [seq])
                        raw = compare_scores(before, after)
                        ids = compare_ids(x, y, [rows], [seq])
                        minimum = (
                            new.scope["_prefill_pooled_postop_min_keys"](
                                q.device, args.tile
                            )
                            if args.integrated
                            else 0
                        )
                        expected_calls = sum(
                            minimum is not None and tile[-1] >= minimum
                            for tile in after.tiles
                        )
                        weighted = probe.result(expected_calls)
                        old_plain = make_scorer(
                            args.baseline_repo or args.repo, False, args.tile
                        )
                        new_plain = make_scorer(
                            args.repo,
                            False,
                            args.tile,
                            postop=operation,
                            integrated=args.integrated,
                        )
                        calls = [
                            lambda: old_plain(q, w, [k], [rows], [seq]),
                            lambda: new_plain(q, w, [k], [rows], [seq]),
                        ]
                        repeats = [
                            compare_ids(x, calls[0](), [rows], [seq]),
                            compare_ids(y, calls[1](), [rows], [seq]),
                        ]
                        result = dict(
                            seq=seq,
                            rows=rows,
                            family=family,
                            seed=seed,
                            block=block,
                            old_tile=args.tile,
                            new_tile=args.tile,
                            weighted=weighted,
                            final_scores=raw,
                            ids=ids,
                            repeats=repeats,
                            boundary=boundary_metrics(before, y, [rows], [seq]),
                            accepted=all(
                                z["exact"] for z in (weighted, raw, ids, *repeats)
                            ),
                        )
                        del before, after, old, new, x, y, probe
                        if args.repeat and result["accepted"]:
                            t0, t1 = measure_pair(
                                calls, torch.npu.synchronize, 3, args.repeat
                            )
                            result.update(
                                old=t0,
                                new=t1,
                                speedup=t0["median_ms"] / t1["median_ms"],
                                memory_old=memory_sample(calls[0], True),
                                memory_new=memory_sample(calls[1], True),
                            )
                        out.write(json.dumps(result, allow_nan=False) + "\n")
                        out.flush()
                        print(
                            json.dumps(
                                {
                                    key: result[key]
                                    for key in (
                                        "seq",
                                        "rows",
                                        "family",
                                        "seed",
                                        "block",
                                        "old_tile",
                                        "accepted",
                                        "weighted",
                                        "final_scores",
                                        "ids",
                                        "speedup",
                                    )
                                    if key in result
                                }
                            ),
                            flush=True,
                        )
                        failed |= not result["accepted"]
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
