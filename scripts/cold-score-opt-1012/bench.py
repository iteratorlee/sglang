#!/usr/bin/env python3
"""Strict pooled-score A/B, using the unchanged production scorer and kernel.

CPU mode emulates compact storage; its timing is NOT an Ascend prediction.
NPU mode must run in the authorized private container on physical device 7.
Exit 2 means numerical/ID/repeat failure. No tolerance or tie-set exemption.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import time

import torch

from score_contract import (
    BASE,
    REL,
    FAMILIES,
    Capture,
    boundary_metrics,
    compare_ids,
    compare_scores,
    cpu_compact_emulation,
    fixture,
    load_file,
    make_scorer,
    source_receipt,
)


def memory_estimate(seq, rows, old_tile):
    pools = seq // 4
    old = min(rows, old_tile) * 32 * pools * 2
    return dict(
        old_score_bytes=old,
        fused_score_bytes=min(rows, 32) * pools * 2,
        input_bytes=rows * 32 * 128 * 2 + rows * 32 * 4 + pools * 128 * 2,
        # Conservative allowance, not a proof/bound on external CANN scratch.
        estimated_scratch_bytes=8 * old,
    )


def measure_pair(calls, sync, warmup, repeat):
    for _ in range(warmup):
        for call in calls:
            call()
    sync()
    samples = [[], []]
    for n in range(repeat):
        for side in (0, 1) if n % 2 == 0 else (1, 0):
            sync()
            start = time.perf_counter_ns()
            calls[side]()
            sync()
            samples[side].append((time.perf_counter_ns() - start) / 1e6)
    return [dict(median_ms=statistics.median(s), samples_ms=s) for s in samples]


def memory_sample(call, npu):
    if not npu:
        return None
    gc.collect()
    torch.npu.synchronize()
    torch.npu.empty_cache()
    before = dict(
        allocated=torch.npu.memory_allocated(),
        reserved=torch.npu.memory_reserved(),
        driver_free=torch.npu.mem_get_info()[0],
    )
    torch.npu.reset_peak_memory_stats()
    result = call()
    torch.npu.synchronize()
    after = dict(
        allocated=torch.npu.memory_allocated(),
        reserved=torch.npu.memory_reserved(),
        driver_free=torch.npu.mem_get_info()[0],
        peak_allocated=torch.npu.max_memory_allocated(),
        peak_reserved=torch.npu.max_memory_reserved(),
    )
    del result
    return dict(
        before=before,
        after=after,
        peak_allocated_delta=after["peak_allocated"] - before["allocated"],
        limitation="driver snapshots do not measure transient external CANN peak",
    )


def run_case(args, seq, rows, family, seed, backend, sync, torch_npu):
    estimate = memory_estimate(seq, rows, args.old_tile)
    if args.device == "npu":
        gc.collect()
        torch.npu.empty_cache()
        free = torch.npu.mem_get_info()[0]
        required = (
            estimate["input_bytes"]
            + estimate["estimated_scratch_bytes"]
            + args.reserve_mb * 2**20
        )
        if free < required:
            raise RuntimeError(f"HBM guard: free={free} required={required}")
    q, w, k = fixture(family, rows, seq // 4, seed)
    if args.device == "npu":
        q, w, k = [x.to("npu:0") for x in (q, w, k)]
    formats = [torch_npu.get_npu_format(x) for x in (q, w, k)] if torch_npu else None
    # Fresh contiguous CPU transfers are synthetic ND fixtures, not a claim
    # about NZ/blocked model inputs. Log actual physical formats for every run.
    if torch_npu and not all(f in (0, 2) for f in formats):
        raise RuntimeError(
            f"Synthetic input unexpectedly has blocked physical format: {formats}"
        )
    lengths, ends = [rows], [seq]
    a, b = Capture(), Capture()
    old = make_scorer(args.repo, False, args.old_tile, observer=a)
    new = make_scorer(args.repo, True, args.old_tile, backend, observer=b)
    x = old(q, w, [k], lengths, ends)
    y = new(q, w, [k], lengths, ends)
    raw = compare_scores(a, b)
    ids = compare_ids(x, y, lengths, ends)
    if not new.calls:
        raise AssertionError("No candidate kernel calls")
    old_plain = make_scorer(args.repo, False, args.old_tile)
    new_plain = make_scorer(args.repo, True, args.old_tile, backend)
    calls = [
        lambda: old_plain(q, w, [k], lengths, ends),
        lambda: new_plain(q, w, [k], lengths, ends),
    ]
    old_repeat = compare_ids(x, calls[0](), lengths, ends)
    new_repeat = compare_ids(y, calls[1](), lengths, ends)
    result = dict(
        seq_len=seq,
        rows=rows,
        family=family,
        seed=seed,
        old_tile=args.old_tile,
        new_tile=32,
        kernel="BQ4_BK64",
        comparison=(
            "kernel_only_equal_tile" if args.old_tile == 32 else "kernel_plus_tile"
        ),
        physical_formats=formats,
        estimate=estimate,
        raw_scores=raw,
        ids=ids,
        boundary=boundary_metrics(a, y, lengths, ends),
        old_repeat=old_repeat,
        new_repeat=new_repeat,
        old_tiles=a.tiles,
        new_tiles=b.tiles,
        accepted=all(z["exact"] for z in (raw, ids, old_repeat, new_repeat)),
    )
    # Release host captures before performance measurements. Never average bad
    # candidates into accepted performance evidence.
    del a, b, old, new, x, y
    if args.repeat and (result["accepted"] or args.time_rejected):
        result["diagnostic_timing_not_acceptance"] = not result["accepted"]
        old_time, new_time = measure_pair(calls, sync, args.warmup, args.repeat)
        result.update(
            old=old_time,
            new=new_time,
            speedup=old_time["median_ms"] / new_time["median_ms"],
            memory_old=memory_sample(calls[0], bool(torch_npu)),
            memory_new=memory_sample(calls[1], bool(torch_npu)),
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--device", choices=("cpu", "npu"), default="cpu")
    parser.add_argument("--case", action="append", help="sequence_tokens:query_rows")
    parser.add_argument("--family", choices=FAMILIES, action="append")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--old-tile", type=int, choices=(8, 32), default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=9)
    parser.add_argument(
        "--time-rejected",
        action="store_true",
        help="Explicit diagnostic timing after a failure; acceptance stays false and exit remains 2",
    )
    parser.add_argument("--reserve-mb", type=int, default=2048)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat < 0 or args.reserve_mb < 512:
        parser.error("warmup/repeat must be nonnegative; reserve >= 512 MiB")
    torch.set_num_threads(1)
    npu = None
    sync = lambda: None
    backend = cpu_compact_emulation
    metadata = dict(
        source_base=BASE,
        hashes=source_receipt(args.repo),
        harness_hashes={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
        },
        device=args.device,
        torch=torch.__version__,
        platform=platform.platform(),
        CPU_is_not_NPU_validation=args.device == "cpu",
    )
    if args.device == "npu":
        if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "7":
            parser.error(
                "This task is authorized on isolated physical device 7; require ASCEND_RT_VISIBLE_DEVICES=7"
            )
        import torch_npu as npu
        import triton

        torch.npu.set_device(0)  # logical 0 maps to the only visible physical 7
        torch.npu.set_compile_mode(jit_compile=False)
        npu.npu.config.allow_internal_format = True  # SGLang NPU initialization
        sync = torch.npu.synchronize
        module = load_file(
            args.repo / REL / "pooled_score_prefill_npu.py", "actual_pooled_score"
        )
        backend = module.fused_pooled_scores
        metadata.update(
            torch_npu=npu.__version__,
            triton=triton.__version__,
            visible_devices=os.environ["ASCEND_RT_VISIBLE_DEVICES"],
            jit_compile=False,
            num_threads=torch.get_num_threads(),
            allow_internal_format_requested=True,
            device_name=torch.npu.get_device_name(0),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases = args.case or (
        ["1024:8", "8192:32"]
        if args.device == "cpu"
        else ["1024:8", "8192:32", "262144:128", "524288:128"]
    )
    failed = False
    with args.output.open("x") as out:
        out.write(json.dumps(dict(metadata=metadata)) + "\n")
        out.flush()
        for case in cases:
            seq, rows = map(int, case.split(":"))
            if not 0 < rows <= seq:
                raise ValueError("Require 0 < rows <= sequence length")
            for family in args.family or ["random"]:
                for seed in args.seed or [1012]:
                    result = run_case(args, seq, rows, family, seed, backend, sync, npu)
                    out.write(json.dumps(result, allow_nan=False) + "\n")
                    out.flush()
                    print(
                        json.dumps(
                            {
                                k: result[k]
                                for k in (
                                    "seq_len",
                                    "rows",
                                    "family",
                                    "seed",
                                    "accepted",
                                    "raw_scores",
                                    "ids",
                                )
                            }
                        ),
                        flush=True,
                    )
                    failed |= not result["accepted"]
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
