#!/usr/bin/env python3
"""Diagnostic stage timing/memory; rejected full fusion stays rejected."""
import argparse
import json
import os
from pathlib import Path

import torch

from bench import measure_pair, memory_sample
from score_contract import REL, fixture, legacy_scores, load_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if os.getenv("ASCEND_RT_VISIBLE_DEVICES") != "7":
        p.error("Require authorized private physical device 7")
    import torch_npu

    torch.set_num_threads(1)
    torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)
    torch_npu.npu.config.allow_internal_format = True
    kernel = load_file(
        a.repo / REL / "pooled_score_prefill_npu.py", "actual_memory_kernel"
    )
    results = []
    for seq in (262144, 524288):
        q, w, k = fixture("random", 128, seq // 4)
        q, w, k = [x.to("npu:0") for x in (q[:32], w[:32], k)]
        old = lambda: legacy_scores(q, w, k)
        new = lambda: kernel.fused_pooled_scores(q, w, k)
        x, y = old(), new()
        raw_exact = bool(
            torch.equal(x.cpu().view(torch.int16), y.cpu().view(torch.int16))
        )
        dot = torch.einsum("qhd,kd->qhk", q, k)
        formats = {
            name: torch_npu.get_npu_format(value)
            for name, value in [
                ("q", q),
                ("w", w),
                ("k", k),
                ("dot", dot),
                ("scores", x),
            ]
        }
        del dot, x, y
        old_time, new_time = measure_pair([old, new], torch.npu.synchronize, 3, 21)
        record = dict(
            seq=seq,
            rows=32,
            raw_bit_exact=raw_exact,
            physical_formats=formats,
            diagnostic_only=True,
            fusion_rejected_globally=True,
            old_raw=old_time,
            new_raw=new_time,
            memory_old_raw=memory_sample(old, True),
            memory_new_raw=memory_sample(new, True),
        )
        scores = old()
        visible = (torch.arange(seq - 128, seq - 96, device=q.device) + 1) // 4
        scores.masked_fill_(
            torch.arange(len(k), device=q.device)[None, :] >= visible[:, None],
            -float("inf"),
        )
        topk = lambda: scores.topk(512, dim=-1).indices.to(torch.int32)
        time_topk, _ = measure_pair([topk, topk], torch.npu.synchronize, 3, 21)
        record.update(topk=time_topk, memory_topk=memory_sample(topk, True))
        results.append(record)
        del scores, q, w, k
    a.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
