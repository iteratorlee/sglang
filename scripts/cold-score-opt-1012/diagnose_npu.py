#!/usr/bin/env python3
"""Stage-tap diagnostic only: deliberately materializes QxHxK on private NPU7.

Derives the tap kernel AST from production; check tapped final scores against
unchanged production before interpreting stage evidence. Never use for timing.
"""
import argparse
import ast
import copy
import json
import os
from pathlib import Path

import torch

from score_contract import REL, fixture, load_file, source_receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--seq", type=int, default=262144)
    p.add_argument("--rows", type=int, default=128)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--tile", type=int, default=32)
    p.add_argument("--seed", type=int, default=1012)
    p.add_argument("--family", default="random")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if os.getenv("ASCEND_RT_VISIBLE_DEVICES") != "7":
        p.error("Require authorized private physical device 7")
    import torch_npu
    import triton

    torch.set_num_threads(1)
    torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)
    torch_npu.npu.config.allow_internal_format = True
    src = args.repo / REL / "pooled_score_prefill_npu.py"
    production = load_file(src, "production_stage_reference")
    tree = ast.parse(src.read_text())
    kernel = copy.deepcopy(
        next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "_score_pool_tile"
        )
    )
    kernel.name = "tap_kernel"
    kernel.args.args[4:4] = [ast.arg(arg="DOT_OUT"), ast.arg(arg="WEIGHTED_OUT")]
    body = []
    for node in kernel.body:
        body.append(node)
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("dot", "weighted"):
                ptr = "DOT_OUT" if name == "dot" else "WEIGHTED_OUT"
                body.extend(
                    ast.parse(
                        f"""tl.store({ptr} + (query_block * BQ * 32 + query_heads[:, None]) * N + keys[None, :], {name}, mask=(rows[:, None] < ROWS) & (keys[None, :] < N))"""
                    ).body
                )
    kernel.body = body
    module = ast.Module(
        body=ast.parse("import triton\nimport triton.language as tl").body + [kernel],
        type_ignores=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generated = args.output.with_suffix(".kernel.py")
    generated.write_text(ast.unparse(ast.fix_missing_locations(module)) + "\n")
    tapped = load_file(generated, "tapped_score_kernel")
    q, w, k = fixture(args.family, args.rows, args.seq // 4, args.seed)
    q, w = (
        q[args.start : args.start + args.tile],
        w[args.start : args.start + args.tile],
    )
    if not len(q):
        raise ValueError("empty query tile")
    first = args.seq - args.rows + args.start
    count = min(args.seq, ((first // 128) + 1) * 128) // 4
    k = k[:count]
    q, w, k = [x.to("npu:0") for x in (q, w, k)]
    rows, count = len(q), len(k)
    old_dot = torch.einsum("qhd,kd->qhk", q, k)
    old_weighted = old_dot.clone().relu_().mul_(w.unsqueeze(-1))
    old_scores = old_weighted.sum(1)
    # Raw-pointer Triton taps require ND allocations even if CANN produced NZ.
    dot = torch.empty(old_dot.shape, dtype=old_dot.dtype, device=q.device)
    weighted = torch.empty(old_dot.shape, dtype=old_dot.dtype, device=q.device)
    score = torch.empty_like(old_scores)
    tapped.tap_kernel[(triton.cdiv(rows, 4), triton.cdiv(count, 64))](
        q,
        w,
        k,
        score,
        dot,
        weighted,
        rows,
        count,
        64,
        4,
        num_warps=4,
        multibuffer=False,
    )
    new = production.fused_pooled_scores(q, w, k)

    def difference(a, b):
        a, b = a.cpu(), b.cpu()
        bits = a.view(torch.int16) != b.view(torch.int16)
        return dict(
            different=int(bits.sum()),
            max_abs=float((a.float() - b.float()).abs().max()),
        )

    report = dict(
        seq=args.seq,
        rows=args.rows,
        start=args.start,
        tile=args.tile,
        family=args.family,
        seed=args.seed,
        hashes=source_receipt(args.repo),
        physical_formats={
            name: torch_npu.get_npu_format(x)
            for name, x in [
                ("q", q),
                ("w", w),
                ("k", k),
                ("old_dot", old_dot),
                ("old_weighted", old_weighted),
                ("old_scores", old_scores),
            ]
        },
        tap_vs_production=difference(score, new),
        dot=difference(old_dot, dot),
        weighted=difference(old_weighted, weighted),
        final=difference(old_scores, new),
    )
    bad = (old_scores != new).nonzero().cpu()[:8]
    examples = []
    for row, pool in bad.tolist():
        values = old_weighted[row, :, pool].float().cpu()
        examples.append(
            dict(
                query_local=row,
                position=first + row,
                pool=pool,
                old=float(old_scores[row, pool]),
                new=float(new[row, pool]),
                weighted_heads=values.tolist(),
                cpu_fp32_sum=float(values.sum()),
                cpu_fp64_sum=float(values.double().sum()),
            )
        )
    report["examples"] = examples
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
