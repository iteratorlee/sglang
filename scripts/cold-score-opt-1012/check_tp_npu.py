#!/usr/bin/env python3
"""Serial emulation of eight TP query owners on NPU7; no HCCL/new service."""
import argparse
import json
import os
from pathlib import Path

import torch

from score_contract import REL, compare_ids, fixture, load_file, make_scorer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--seq", type=int, default=262144)
    p.add_argument("--rows", type=int, default=2048)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if os.getenv("ASCEND_RT_VISIBLE_DEVICES") != "7":
        p.error("Require authorized private physical device 7")
    import torch_npu

    torch.set_num_threads(1)
    torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)
    torch_npu.npu.config.allow_internal_format = True
    partition = load_file(
        a.repo / REL / "prefill_index_parallel.py", "actual_partition"
    )
    kernel = load_file(a.repo / REL / "pooled_score_prefill_npu.py", "actual_tp_kernel")
    q, w, k = [x.to("npu:0") for x in fixture("random", a.rows, a.seq // 4)]
    parts, capacity = partition.partition_queries([a.rows], [a.seq], 8)
    report = dict(
        seq=a.seq,
        rows=a.rows,
        world=8,
        capacity=capacity,
        collective="not tested: serial owner reconstruction on physical 7",
        fragments=[[vars(f) for f in group] for group in parts],
    )
    full = []
    for name, fused in [("old", False), ("fused", True)]:
        scorer = make_scorer(a.repo, fused, 32, kernel.fused_pooled_scores)
        expected = scorer(q, w, [k], [a.rows], [a.seq])
        outputs = []
        for group in parts:
            if not group:
                continue
            meta = partition.FragmentBatchMetadata.from_fragments(group)
            outputs.append(
                scorer(
                    torch.cat([q[f.offset : f.offset + f.length] for f in group]),
                    torch.cat([w[f.offset : f.offset + f.length] for f in group]),
                    [k for _ in group],
                    meta.extend_seq_lens_cpu,
                    meta.seq_lens_cpu,
                )
            )
        report[name + "_full_vs_tp"] = compare_ids(
            expected, torch.cat(outputs), [a.rows], [a.seq]
        )
        full.append(expected)
    report["old_vs_fused_ids"] = compare_ids(*full, [a.rows], [a.seq])
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return (
        0
        if all(
            report[key]["exact"]
            for key in ("old_full_vs_tp", "fused_full_vs_tp", "old_vs_fused_ids")
        )
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
