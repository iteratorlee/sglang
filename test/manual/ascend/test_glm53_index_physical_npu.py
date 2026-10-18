"""Opt-in physical-pool three-way NPU diagnostic. No service or model startup.

All nine full state snapshots use the unchanged v2 byte comparator, including
unused physical blocks and sink zero. Old zero-KV undefined-domain counts stay
visible. A pass does not clear the failed full-GSM gate or authorize deployment.
"""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import traceback

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import torch

from glm53_index_batch_contract_v2 import compare_snapshots, snapshot
from glm53_index_batch_fixture import SOURCE, Runner, exact
from glm53_index_physical_fixture import PhysicalLayout, compact_cases, physical_fixture
from test_glm53_index_batch_npu import capture, ints

MODES = (("legacy", False, False), ("li", True, False), ("li_expand", True, True))
PATTERNS = ("random", "ties", "near-ties", "closing", "mixed-padding", "all-padding")


def frames(bs, args):
    for spec in args.specs:
        if len(spec["starts"]) != bs:
            continue
        yield spec, physical_fixture(
            spec["starts"],
            layout=args.layout,
            requests=spec["requests"],
            kind="ties" if "padding" in spec["pattern"] else spec["pattern"],
            topk=args.topk,
            heads=args.heads,
            extra=args.extra,
            seed=spec["seed"],
        )


def tensor_info(tensor, torch_npu):
    info = dict(
        shape=list(tensor.shape),
        stride=list(tensor.stride()),
        dtype=str(tensor.dtype),
        device=str(tensor.device),
        storage_offset=tensor.storage_offset(),
        contiguous=tensor.is_contiguous(),
    )
    if tensor.device.type == "npu":
        info["npu_format"] = int(torch_npu.get_npu_format(tensor))
    return info


def inspect_calls(runner, case, mode, torch_npu):
    """Run only OUTSIDE capture: check actual CANN inputs, not planned shapes."""
    name, batched, expanded = mode
    bs, layout = case.bs, case.physical_layout
    expected_events = (
        ["update"] * 4 + ["indexer"] + ["expand"] * (1 if expanded else 4)
        if batched
        else ["update", "indexer", "expand"] * 4
    )
    if runner.events != expected_events or len(runner.calls) != (1 if batched else 4):
        raise AssertionError(f"{name}: LI/expand dispatch changed")
    table = case.data["table"][:, ::4].clamp_min(0).contiguous()
    positions = case.data["positions"][: bs * 4].reshape(bs, 4)
    lengths = torch.where(case.data["req"][:, None] > 0, (positions + 1) // 4, 0).int()
    result = []
    for step, call in enumerate(runner.calls):
        rows = bs * 4 if batched else bs
        expected_table = table.repeat_interleave(4, dim=0) if batched else table
        expected_lens = lengths.flatten() if batched else lengths[:, step]
        if tuple(call["key"].shape) != (layout.key_blocks, 64, 1, 128):
            raise AssertionError("physical key shape does not reach CANN")
        if tuple(call["query"].shape) != (rows, case.heads, 128):
            raise AssertionError("query shape does not reach CANN")
        if tuple(call["block_table"].shape) != (rows, layout.pooled_columns):
            raise AssertionError("physical table width does not reach CANN")
        exact(call["block_table"], expected_table, "actual LI dynamic page table")
        exact(
            call["actual_seq_lengths_key"],
            torch.tensor(expected_lens.tolist(), dtype=torch.int32),
            "actual LI key lengths",
        )
        exact(
            call["actual_seq_lengths_query"],
            torch.arange(1, rows + 1, dtype=torch.int32),
            "independent len1 query prefix sums",
        )
        for key, expected in dict(
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_mode=3,
            sparse_count=case.topk // 4,
        ).items():
            if call[key] != expected:
                raise AssertionError(f"LI argument {key} changed")
        result.append(
            dict(
                tensors={
                    k: tensor_info(v, torch_npu)
                    for k, v in call.items()
                    if isinstance(v, torch.Tensor)
                },
                actual_q=call["actual_seq_lengths_query"].cpu().tolist(),
                actual_k=call["actual_seq_lengths_key"].cpu().tolist(),
                page_min=int(expected_table.min()),
                page_max=int(expected_table.max()),
            )
        )
    return result


def compare_three(case, snapshots, diagnostics):
    for a, b in ((0, 1), (0, 2), (1, 2)):
        label = f"{MODES[a][0]}_vs_{MODES[b][0]}"
        compare_snapshots(
            case,
            snapshots[a],
            snapshots[b],
            canonical_a=MODES[a][1],
            canonical_b=MODES[b][1],
            batch_expand_a=MODES[a][2],
            batch_expand_b=MODES[b][2],
            diagnostics=diagnostics.setdefault(label, {}),
        )


def trace(runner):
    return runner.calls, runner.pooled, runner.states, runner.expands, runner.events


def set_trace(runner, values):
    runner.calls, runner.pooled, runner.states, runner.expands, runner.events = values


def group(bs, args, torch_npu, baseline, report):
    initial = next(frames(bs, args))[1]
    runners = [
        Runner(
            initial,
            torch_npu.npu_lightning_indexer,
            enabled=enabled,
            batch_expand=expand,
            device=f"npu:{args.device}",
            fused=True,
            source=None if enabled else baseline,
        )
        for _, enabled, expand in MODES
    ]
    graphs_outputs = [capture(torch, runner) for runner in runners]
    saved = [trace(r) for r in runners]
    pointers = [[v.data_ptr() for v in r.data.values()] for r in runners]
    for spec, case in frames(bs, args):
        row = dict(
            bs=bs,
            **spec,
            passed=False,
            physical=case.physical_summary,
            comparisons={},
            phase="eager",
        )
        report["cases"].append(row)
        eager, replay = [], []
        row["actual_li_inputs"] = {}
        for mode, runner, captured in zip(MODES, runners, saved):
            set_trace(runner, ([], [], [], [], []))
            runner.reset(case)
            out = runner.run()
            torch.npu.synchronize()
            eager.append(snapshot(runner, out))
            row["actual_li_inputs"][mode[0]] = inspect_calls(
                runner, case, mode, torch_npu
            )
            # Release eager device traces; captured traces remain live for replay.
            set_trace(runner, captured)
        compare_three(case, eager, row["comparisons"].setdefault("eager", {}))
        row["phase"] = "legacy_repeat"
        old = runners[0]
        set_trace(old, ([], [], [], [], []))
        old.reset(case)
        out = old.run()
        torch.npu.synchronize()
        repeated = snapshot(old, out)
        compare_snapshots(
            case,
            eager[0],
            repeated,
            canonical_a=False,
            canonical_b=False,
            diagnostics=row["comparisons"].setdefault("old_repeat", {}),
        )
        del repeated
        set_trace(old, saved[0])
        row["phase"] = "graph"
        for i, (mode, runner, (graph, out)) in enumerate(
            zip(MODES, runners, graphs_outputs)
        ):
            runner.reset(case)
            graph.replay()
            torch.npu.synchronize()
            replay.append(snapshot(runner, out))
            inspect_calls(runner, case, mode, torch_npu)
            compare_snapshots(
                case,
                eager[i],
                replay[i],
                canonical_a=mode[1],
                canonical_b=mode[1],
                batch_expand_a=mode[2],
                batch_expand_b=mode[2],
                diagnostics=row["comparisons"].setdefault(mode[0] + "_graph_eager", {}),
            )
            # Four full-cache host snapshots at peak, not six: the isolated
            # runner caps host RAM at 24 GiB. Keep all graph snapshots for the
            # direct three-way comparison; release only already-checked eager.
            eager[i] = None
        compare_three(case, replay, row["comparisons"].setdefault("graph", {}))
        if pointers != [[v.data_ptr() for v in r.data.values()] for r in runners]:
            raise AssertionError("input/state pointer changed across replay")
        row.update(
            passed=True,
            phase="complete",
            defined_outputs_exact=True,
            all_state_bytes_exact=True,
            canonical_candidates_exact=True,
        )
        print(json.dumps(dict(id=spec["id"], bs=bs, passed=True)), flush=True)
        # Full physical-cache snapshots are intentionally not retained per case.
        del eager, replay, case


def source_hashes():
    paths = [Path(__file__), SOURCE]
    paths += [
        Path(__file__).with_name(name)
        for name in (
            "glm53_index_physical_fixture.py",
            "glm53_index_batch_fixture.py",
            "glm53_index_batch_contract_v2.py",
            "test_glm53_index_batch_npu.py",
        )
    ]
    paths += [
        SOURCE.parent / (name + ".py")
        for name in (
            "kpool_metadata_npu",
            "kpool_expand_npu",
            "graph_ops",
            "kpool_tail_scatter_npu",
            "kpool_snapshot_npu",
        )
    ]
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--key-blocks", type=int, default=29049)
    p.add_argument("--pooled-table-width", type=int, default=519)
    p.add_argument(
        "--full-table-width",
        type=int,
        help="default 4 * pooled width; 2074 also yields 519",
    )
    p.add_argument(
        "--batches",
        type=ints,
        default=[1, 2, 4, 8, 13],
        help="filter the fixed 20-fixture matrix",
    )
    p.add_argument(
        "--case-ids",
        type=lambda s: s.split(","),
        help="optional subset of IDs printed by --plan-only",
    )
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--topk", type=int, default=2048)
    p.add_argument(
        "--extra",
        type=int,
        default=0,
        help="extra output-shape padding rows; production default 0",
    )
    p.add_argument(
        "--baseline-source",
        type=Path,
        help="default: same production source, flags off",
    )
    p.add_argument("--cann-build", default="UNVERIFIED")
    p.add_argument("--installed-contract", type=Path)
    p.add_argument(
        "--plan-only",
        action="store_true",
        help="print geometry/counts, no tensor fixture or device initialization",
    )
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if not args.batches or any(not 1 <= b <= 16 for b in args.batches):
        p.error("batches must be in [1,16]")
    if args.topk < 4 or args.topk % 4 or not 1 <= args.heads <= 64 or args.extra < 0:
        p.error("invalid topk/heads/extra")
    args.layout = PhysicalLayout(
        args.key_blocks, args.pooled_table_width, args.full_table_width
    )
    known = compact_cases()
    if args.case_ids and set(args.case_ids) - {s["id"] for s in known}:
        p.error("unknown --case-ids; inspect --plan-only")
    args.specs = [
        s
        for s in known
        if len(s["starts"]) in args.batches
        and (not args.case_ids or s["id"] in args.case_ids)
    ]
    if not args.specs:
        p.error("matrix filter selects no fixtures")
    args.batches = sorted({len(s["starts"]) for s in args.specs})
    try:
        for spec in args.specs:
            args.layout.validate(len(spec["starts"]), spec["starts"])
    except ValueError as exc:
        p.error(str(exc))
    plan = dict(
        cases=len(args.specs),
        matrix=args.specs,
        batches=args.batches,
        shapes={bs: args.layout.shapes(bs) for bs in args.batches},
        modes=[m[0] for m in MODES],
        source_hashes=source_hashes(),
    )
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return
    if args.output is None or args.output.exists():
        p.error(
            "--output must be a fresh path; existing failure receipts are never overwritten"
        )
    report = dict(
        passed=False,
        promotion_allowed=False,
        contract="unchanged-defined-v2",
        plan=plan,
        cases=[],
        cann_build=args.cann_build,
    )
    baseline = (args.baseline_source or SOURCE).read_text()
    report["baseline_sha256"] = hashlib.sha256(baseline.encode()).hexdigest()
    if args.installed_contract:
        report["installed_contract_sha256"] = hashlib.sha256(
            args.installed_contract.read_bytes()
        ).hexdigest()
    try:
        os.environ["SGLANG_GLM53_KPOOL_EXPAND"] = "1"
        os.environ["TRITON_INTERPRET"] = "0"
        import torch_npu

        torch.set_num_threads(1)
        torch.npu.set_device(args.device)
        report.update(
            torch=torch.__version__,
            torch_npu=torch_npu.__version__,
            torch_npu_path=torch_npu.__file__,
            device=torch.npu.get_device_name(args.device),
        )
        with torch.no_grad():
            for bs in args.batches:
                group(bs, args, torch_npu, baseline, report)
                gc.collect()
                torch.npu.empty_cache()
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
