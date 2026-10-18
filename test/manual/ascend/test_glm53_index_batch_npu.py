"""Opt-in, standalone real-NPU exact A/B; no server, model, distributed or CUDA.

Run ONLY on the main agent's isolated NPU after its baseline/profile completes.
Tests extracted production methods with real CANN LI and real Triton kernels.
The fixture injects capture context for eager diagnostics as well as NPUGraph;
production eligibility itself is tested separately on CPU. No allclose, sorted
indices, set equivalence, or tolerances are accepted. A pass is finite evidence,
not a guarantee about arbitrary ties or another installed CANN build.
"""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import traceback
from types import SimpleNamespace as NS

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

from glm53_index_batch_fixture import (
    BASELINE,
    REL,
    ROOT,
    SOURCE,
    Runner,
    compare,
    exact,
    fixture,
)


def ints(value):
    return [int(x) for x in value.split(",")]


def visibility(torch, runner, output):
    raw = runner.raw_topk().cpu()
    reqs = runner.data["req"].cpu().tolist()
    pos = runner.data["positions"].cpu().tolist()
    for row, req in enumerate(reqs):
        for step in range(4):
            p = pos[row * 4 + step]
            length = (p + 1) // 4 if req > 0 else 0
            ids = raw[row, step]
            valid = ids[ids >= 0]
            if len(valid) != min(length, runner.case.topk // 4):
                raise AssertionError(
                    f"valid topk count at row={row}, step={step}, pool_len={length}"
                )
            if len(valid.unique()) != len(valid) or (valid >= length).any():
                raise AssertionError(
                    f"duplicate or future pool at row={row}, step={step}"
                )
    if not torch.equal(
        output[runner.case.bs * 4 :].cpu(),
        torch.zeros_like(output[runner.case.bs * 4 :].cpu()),
    ):
        raise AssertionError("shape padding must remain exact integer zero")


def capture(torch, runner):
    for _ in range(3):
        runner.reset()
        runner.run()
    torch.npu.synchronize()
    runner.reset()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out = runner.run()
    torch.npu.synchronize()
    return graph, out


def frames(bs, prefix, args):
    for kind in ("random", "ties", "near-ties", "closing"):
        for offset in range(4):
            yield kind, offset, fixture(
                [prefix + (row + offset) % 4 for row in range(bs)],
                requests=list(range(1, bs + 1)) if offset % 2 else None,
                kind=kind,
                topk=args.topk,
                heads=args.heads,
                capacity=prefix + 264,
                seed=53 + offset,
            )
    for padding in ("mixed-padding", "all-padding"):
        reqs = list(range(1, bs + 1)) if padding == "mixed-padding" else [0] * bs
        reqs[-1] = 0
        yield padding, 0, fixture(
            [prefix] * bs,
            requests=reqs,
            kind="ties",
            topk=args.topk,
            heads=args.heads,
            capacity=prefix + 264,
        )
    if bs == 1:
        signed = fixture(
            [prefix], topk=args.topk, heads=args.heads, capacity=prefix + 264
        )
        signed.data["key"].neg_()
        signed.data["_kpool_tail_k"].neg_()
        yield "signed-zero-sink", 0, signed


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
    a, b = runners
    # First eager run compiles kernels. All subsequent frames also compare
    # actual eager and graph outputs, including exact raw index order.
    for runner in runners:
        runner.run()
    graphs_outputs = [capture(torch, runner) for runner in runners]
    pointers = [[v.data_ptr() for v in r.data.values()] for r in runners]
    for kind, offset, case in frames(bs, prefix, args):
        # Eager uses separate trace lists; save captured tensor references so
        # graph replay is compared using graph outputs, not the latest eager.
        saved = [(r.calls, r.pooled, r.states, r.expands, r.events) for r in runners]
        eager_outputs, eager_raw, eager_states = [], [], []
        for runner in runners:
            (
                runner.calls,
                runner.pooled,
                runner.states,
                runner.expands,
                runner.events,
            ) = ([], [], [], [], [])
            runner.reset(case)
            eager_outputs.append(runner.run().clone())
            eager_raw.append(runner.raw_topk().clone())
        torch.npu.synchronize()
        compare(a, eager_outputs[0], b, eager_outputs[1])
        if (len(a.calls), len(b.calls)) != (4, 1):
            raise AssertionError(
                "expected four baseline LI calls and one candidate call"
            )
        for runner, out in zip(runners, eager_outputs):
            visibility(torch, runner, out)
            eager_states.append(
                [
                    {key: value.cpu() for key, value in state.items()}
                    for state in runner.states
                ]
            )
        for runner, saved_trace in zip(runners, saved):
            (
                runner.calls,
                runner.pooled,
                runner.states,
                runner.expands,
                runner.events,
            ) = saved_trace
            runner.reset(case)
        for graph, _ in graphs_outputs:
            graph.replay()
        torch.npu.synchronize()
        compare(a, graphs_outputs[0][1], b, graphs_outputs[1][1])
        for i, runner in enumerate(runners):
            exact(graphs_outputs[i][1], eager_outputs[i], "graph/eager expanded ids")
            exact(runner.raw_topk(), eager_raw[i], "graph/eager pool ids")
            for step, (graph_state, eager_state) in enumerate(
                zip(runner.states, eager_states[i])
            ):
                for key in graph_state:
                    exact(
                        graph_state[key],
                        eager_state[key],
                        f"graph/eager state {step} {key}",
                    )
            visibility(torch, runner, graphs_outputs[i][1])
        if pointers != [[v.data_ptr() for v in r.data.values()] for r in runners]:
            raise AssertionError("input/state addresses changed during replay")
        row = dict(
            bs=bs,
            prefix=prefix,
            pattern=kind,
            offset=offset,
            eager_exact=True,
            graph_exact=True,
            li_calls=[4, 1],
            expand_rows=[bs] * 4,
            step_states_exact=True,
        )
        report["cases"].append(row)
        print(json.dumps(row), flush=True)


def empty_forward(torch, op, baseline, args):
    case = fixture([], extra=0, topk=args.topk, heads=args.heads)
    outputs = []
    for enabled in (False, True):
        runner = Runner(
            case,
            op,
            enabled=enabled,
            device=f"npu:{args.device}",
            source=None if enabled else baseline,
        )
        before = runner.state()
        runner.batch.forward_mode = NS(is_idle=lambda: True)
        out = runner.obj.forward_npu(
            runner.data["key"], None, runner.data["positions"], runner.batch, 0
        )
        if out.shape != (0, 1, args.topk + 3) or runner.calls:
            raise AssertionError("empty forward contract")
        for key, value in before.items():
            exact(value, runner.state()[key], f"empty forward {key}")
        outputs.append(out)
    exact(*outputs, "empty forward output")


def benchmark(torch, op, baseline, bs, args):
    case = fixture(
        [131068 + row % 4 for row in range(bs)], topk=args.topk, heads=args.heads
    )
    results = []
    for enabled in (False, True):
        runner = Runner(
            case,
            op,
            enabled=enabled,
            device=f"npu:{args.device}",
            source=None if enabled else baseline,
            fused=args.fused_state,
            trace=False,
        )  # No per-step clones in measured graphs.
        graph, _ = capture(torch, runner)
        times = []
        for _ in range(args.benchmark):
            runner.reset()
            start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(
                enable_timing=True
            )
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
        results.append(
            dict(
                enabled=enabled,
                bs=bs,
                samples_ms=times,
                median_ms=statistics.median(times),
                li_calls=len(runner.calls),
                expand_rows=[shape[0] for shape, _ in runner.expands],
            )
        )
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--batches", type=ints, default=[1, 2, 14])
    p.add_argument("--prefixes", type=ints, default=[0, 60, 252, 2044, 131068])
    p.add_argument(
        "--topk", type=int, default=2048, help="uncompressed model index_topk"
    )
    p.add_argument(
        "--heads",
        type=int,
        default=64,
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
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
