#!/usr/bin/env python3
"""Matched raw key-scoring scope: CANN dot + postops + native head sum.

Companion to bench_postop_npu.py, which measures the actual caller including
causal masking, top-k, padding and concatenation over 128 query rows.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from bench import measure_pair, memory_sample
from score_contract import REL, fixture, legacy_scores, load_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--block", choices=(4096, 8192), type=int, default=8192)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if os.getenv("ASCEND_RT_VISIBLE_DEVICES") != "7":
        p.error("Require authorized private physical device 7")
    import torch_npu

    torch.set_num_threads(1)
    torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)
    torch_npu.npu.config.allow_internal_format = True
    source = a.repo / REL / "pooled_score_postop_npu.py"
    module = load_file(source, "stage_postop")
    records = []
    for seq in (262144, 524288):
        for rows in (8, 32):
            q, w, k = [v.to("npu:0") for v in fixture("random", rows, seq // 4)]

            def old():
                return legacy_scores(q, w, k)

            def new():
                scores = torch.einsum("qhd,kd->qhk", q, k)
                active = module.relu_gate_(scores, w, a.block)
                assert active
                return scores.sum(1)

            x, y = old(), new()
            exact = bool(
                torch.equal(x.cpu().view(torch.int16), y.cpu().view(torch.int16))
            )
            if not exact:
                raise RuntimeError("Raw key-scoring numerical failure")
            del x, y
            t0, t1 = measure_pair([old, new], torch.npu.synchronize, 3, 21)
            record = dict(
                seq=seq,
                rows=rows,
                block=a.block,
                raw_bit_exact=exact,
                old=t0,
                new=t1,
                speedup=t0["median_ms"] / t1["median_ms"],
                memory_old=memory_sample(old, True),
                memory_new=memory_sample(new, True),
            )
            # Retain the same BF16 input allocation for both post-op scopes.
            # copy_ resets the input before each invocation and is timed on both.
            dot = torch.einsum("qhd,kd->qhk", q, k)
            target = dot.clone()

            def old_post():
                target.copy_(dot)
                return target.relu_().mul_(w.unsqueeze(-1))

            def new_post():
                target.copy_(dot)
                assert module.relu_gate_(target, w, a.block)
                return target

            p0, p1 = measure_pair([old_post, new_post], torch.npu.synchronize, 3, 21)
            record.update(
                reset_plus_postop_old=p0,
                reset_plus_postop_new=p1,
                memory_reset_plus_postop_old=memory_sample(old_post, True),
                memory_reset_plus_postop_new=memory_sample(new_post, True),
                physical_formats={
                    name: torch_npu.get_npu_format(v)
                    for name, v in [("q", q), ("weights", w), ("keys", k), ("dot", dot)]
                },
            )
            records.append(record)
            del dot, target, q, w, k
    result = dict(
        prototype_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        scope="raw key scoring; reset-plus-postop separately labeled",
        records=records,
    )
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
