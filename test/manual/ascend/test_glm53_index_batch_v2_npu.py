"""Opt-in v2 NPU acceptance for defined indices and canonical zero-key output.

Standalone, no service/model/distributed setup. The strict v1 harness stays
unchanged. Both A/B and graph/eager use the same defined-output comparator;
every dead-row difference is counted, never relabeled as full-output exact.
"""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import traceback

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

from glm53_index_batch_fixture import BASELINE, REL, ROOT, SOURCE, Runner
from glm53_index_batch_contract_v2 import compare_snapshots, snapshot
from test_glm53_index_batch_npu import benchmark, capture, empty_forward, frames, ints


def group(torch, op, baseline, bs, prefix, args, report):
    initial = next(frames(bs, prefix, args))[2]
    runners = [
        Runner(
            initial,
            op,
            enabled=e,
            device=f"npu:{args.device}",
            source=None if e else baseline,
            fused=args.fused_state,
        )
        for e in (False, True)
    ]
    for runner in runners:
        runner.run()
    graphs_outputs = [capture(torch, runner) for runner in runners]
    pointers = [[v.data_ptr() for v in r.data.values()] for r in runners]
    for kind, offset, case in frames(bs, prefix, args):
        row = dict(
            bs=bs,
            prefix=prefix,
            pattern=kind,
            offset=offset,
            passed=False,
            comparisons={},
        )
        # Keep diagnostics even if a later invariant fails.
        report["cases"].append(row)
        saved = [(r.calls, r.pooled, r.states, r.expands, r.events) for r in runners]
        eager = []
        for runner in runners:
            (
                runner.calls,
                runner.pooled,
                runner.states,
                runner.expands,
                runner.events,
            ) = ([], [], [], [], [])
            runner.reset(case)
            out = runner.run()
            torch.npu.synchronize()
            eager.append(snapshot(runner, out))
        if (len(runners[0].calls), len(runners[1].calls)) != (4, 1):
            raise AssertionError(
                "expected four baseline LI calls and one candidate call"
            )
        if runners[1].events != ["update"] * 4 + ["indexer"] + ["expand"] * 4:
            raise AssertionError("candidate update/indexer/expand order")

        def check(label, left, right, ca=False, cb=True):
            diag = row["comparisons"][label] = {}
            compare_snapshots(
                case, left, right, canonical_a=ca, canonical_b=cb, diagnostics=diag
            )

        check("eager_ab", *eager)
        # Repeat the legacy operation independently, retaining undefined-domain
        # counts even if both runs happen to return the same bytes.
        runners[0].reset(case)
        repeated = runners[0].run()
        torch.npu.synchronize()
        check(
            "old_eager_repeat", eager[0], snapshot(runners[0], repeated), False, False
        )
        for runner, trace in zip(runners, saved):
            (
                runner.calls,
                runner.pooled,
                runner.states,
                runner.expands,
                runner.events,
            ) = trace
            runner.reset(case)
        for graph, _ in graphs_outputs:
            graph.replay()
        torch.npu.synchronize()
        replay = [snapshot(r, out) for r, (_, out) in zip(runners, graphs_outputs)]
        check("graph_ab", *replay)
        for i, name in enumerate(("old_graph_eager", "new_graph_eager")):
            check(name, replay[i], eager[i], bool(i), bool(i))
        if pointers != [[v.data_ptr() for v in r.data.values()] for r in runners]:
            raise AssertionError("input/state addresses changed during replay")
        row.update(
            passed=True,
            defined_outputs_exact=True,
            all_state_bytes_exact=True,
            new_zero_key_canonical=True,
            li_calls=[4, 1],
            expand_rows=[bs] * 4,
        )
        print(json.dumps(row), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--batches", type=ints, default=[1, 12, 13, 14])
    p.add_argument("--prefixes", type=ints, default=[0, 1, 2, 3, 60, 252, 2044, 131068])
    p.add_argument(
        "--topk", type=int, default=2048, help="uncompressed model index_topk"
    )
    p.add_argument(
        "--heads",
        type=int,
        default=32,
        help="use the target model's actual index query heads",
    )
    p.add_argument("--baseline-source", type=Path)
    p.add_argument(
        "--installed-contract",
        type=Path,
        help="installed CANN 9.0 LI API/source evidence to hash",
    )
    p.add_argument(
        "--cann-build",
        default="UNVERIFIED",
        help="exact installed CANN build identification",
    )
    p.add_argument(
        "--fused-state",
        action="store_true",
        help="also test existing fused scatter + snapshot",
    )
    p.add_argument(
        "--benchmark",
        type=int,
        default=0,
        help="optional trace-free graph timing samples",
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("refusing to overwrite an existing receipt; use a fresh v2 output path")
    if not args.batches or any(not 1 <= b <= 16 for b in args.batches):
        p.error("batches must be in [1,16]")
    if args.topk <= 0 or args.topk % 4 or not 1 <= args.heads <= 64:
        p.error("topk must be positive and divisible by 4; heads in [1,64]")
    if args.benchmark < 0 or not args.prefixes or min(args.prefixes) < 0:
        p.error("benchmark/prefixes must be nonnegative")
    # Explicit import only here: --help / CPU compilation never initializes NPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["SGLANG_GLM53_KPOOL_EXPAND"] = "1"
    os.environ["TRITON_INTERPRET"] = "0"
    import torch
    import torch_npu

    torch.npu.set_device(args.device)
    torch.set_num_threads(1)
    baseline = (
        args.baseline_source.read_text()
        if args.baseline_source
        else subprocess.check_output(
            ["git", "show", f"{BASELINE}:{REL}"], cwd=ROOT, text=True
        )
    )
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [
            SOURCE,
            Path(__file__),
            Path(__file__).with_name("glm53_index_batch_contract_v2.py"),
            Path(__file__).with_name("glm53_index_batch_fixture.py"),
            Path(__file__).with_name("test_glm53_index_batch_npu.py"),
            *[
                SOURCE.parent / f"{name}.py"
                for name in (
                    "kpool_metadata_npu",
                    "graph_ops",
                    "kpool_tail_scatter_npu",
                    "kpool_snapshot_npu",
                    "kpool_expand_npu",
                )
            ],
        ]
    }
    report = dict(
        passed=False,
        contract="defined-v2",
        cases=[],
        torch=torch.__version__,
        torch_npu=torch_npu.__version__,
        torch_npu_path=torch_npu.__file__,
        cann_build=args.cann_build,
        device=torch.npu.get_device_name(args.device),
        source_hashes=hashes,
        baseline_sha256=hashlib.sha256(baseline.encode()).hexdigest(),
        topk=args.topk,
        heads=args.heads,
        fused_state=args.fused_state,
        promotion_allowed=False,
        installed_contract_sha256=None,
    )
    if args.installed_contract:
        report["installed_contract_sha256"] = hashlib.sha256(
            args.installed_contract.read_bytes()
        ).hexdigest()
    try:
        with torch.no_grad():
            empty_forward(torch, torch_npu.npu_lightning_indexer, baseline, args)
            report["empty_forward_exact"] = True
            for bs in args.batches:
                for prefix in args.prefixes:
                    group(
                        torch,
                        torch_npu.npu_lightning_indexer,
                        baseline,
                        bs,
                        prefix,
                        args,
                        report,
                    )
                    # Runner's recording closures form Python reference cycles;
                    # release full-cache traces before the next large fixture.
                    gc.collect()
                    torch.npu.empty_cache()
            if args.benchmark:
                report["timings"] = [
                    benchmark(
                        torch, torch_npu.npu_lightning_indexer, baseline, bs, args
                    )
                    for bs in args.batches
                ]
        report["passed"] = True
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
