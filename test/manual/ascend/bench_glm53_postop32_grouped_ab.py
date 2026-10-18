"""Bounded P32 native/legacy/group4/group8 NPU test; no SGLang registration.

All eager/graph correctness precedes independent timing. The parent must
already have stopped P. This script never controls services or resources.
"""

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import statistics
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

GROUP = "SGLANG_GLM53_PREFILL_POOLED_POSTOP32_GROUP"
MIN_KEYS = "SGLANG_GLM53_PREFILL_POOLED_POSTOP32_MIN_KEYS"
MODES = ("native", "legacy", "group4", "group8")


@contextmanager
def environment(mode, rows, min_keys=None):
    with h.environment(0, rows):
        os.environ[GROUP] = mode[-1] if mode.startswith("group") else "0"
        if min_keys is not None:
            os.environ[MIN_KEYS] = str(min_keys)
        yield


class Probe:
    def __init__(self, kernel, name, events):
        self.kernel, self.name, self.events = kernel, name, events

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def invoke(*args, **kwargs):
            self.events.append(dict(kernel=self.name, grid=list(grid),
                                    block=args[-1], num_warps=kwargs.get("num_warps")))
            return launch(*args, **kwargs)

        return invoke


@contextmanager
def observe_launches(helper):
    names = ("_relu_gate_inplace", "_relu_gate_grouped")
    originals = {name: getattr(helper, name) for name in names}
    events = []
    for name in names:
        setattr(helper, name, Probe(originals[name], name, events))
    try:
        yield events
    finally:
        for name, original in originals.items():
            setattr(helper, name, original)


def check_launch(events, mode, rows, keys):
    if mode == "native":
        if events:
            raise AssertionError(f"Native unexpectedly launched Triton: {events}")
        return
    key_blocks = (keys + 8191) // 8192
    if mode == "legacy":
        expected = dict(kernel="_relu_gate_inplace", grid=[rows * 32, key_blocks],
                        block=8192, num_warps=4)
    else:
        group = int(mode[-1])
        expected = dict(kernel="_relu_gate_grouped",
                        grid=[(rows * 32 * key_blocks + group - 1) // group],
                        block=8192, num_warps=4)
    if events != [expected]:
        raise AssertionError(f"Wrong actual launch: {events}; expected {[expected]}")


def operation(helper, mode, scores, weights):
    if mode == "native":
        scores.relu_().mul_(weights.unsqueeze(-1))
    else:
        h.launch32(helper, scores, weights, 8192)


def direct_case(helper, mode, rows, keys, pattern, seed):
    scores, weights = h.fixture(rows, keys, pattern, seed, heads=32)
    if h.formats(scores, weights) != [2, 2]:
        raise AssertionError("Direct case requires ND BF16 scores/FP32 weights")
    reference = scores.clone().relu_().mul_(weights.unsqueeze(-1))
    with environment(mode, rows), observe_launches(helper) as events:
        operation(helper, mode, scores, weights)
    check_launch(events, mode, rows, keys)
    old_logits, old_ids = h.finish(reference, keys)
    logits, ids = h.finish(scores, keys)
    return dict(scope="eager_direct", mode=mode, rows=rows, heads=32, keys=keys,
                pattern=pattern, launches=events, physical_formats=h.formats(scores, weights),
                weighted=h.strict(reference, scores, "weighted BF16"),
                masked_logits=h.strict(old_logits, logits, "masked logits"),
                topk_int32=h.strict(old_ids, ids, "ordered INT32 TopK"))


def caller_case(module_file, helper, mode, rows, keys, seed, auto=False, min_keys=None):
    q, weights, history, batch = h.query_fixture(rows, keys, seed, heads=32)
    indexer = SimpleNamespace(index_topk=2048, index_kpool=4)
    captures, calls = [[], []], []
    baseline = h.forced32_scorer(module_file, helper, 8192, False,
                                lambda x: captures[0].append(x.detach().cpu()))
    with environment("native", rows):
        old_ids = baseline(indexer, q, weights, [history], batch)
    observer = lambda x: captures[1].append(x.detach().cpu())
    fn = (h.scorer(module_file, helper, observer, calls) if auto else
          h.forced32_scorer(module_file, helper, 8192, mode != "native", observer, calls))
    with environment(mode, rows, min_keys), observe_launches(helper) as events:
        ids = fn(indexer, q, weights, [history], batch)
    active = mode != "native" and (not auto or keys >= (min_keys or 32768))
    check_launch(events, mode if active else "native", rows, keys)
    if len(calls) != int(active) or any(c["active"] is not True for c in calls):
        raise AssertionError(f"Wrong caller activation: {calls}")
    if active and (calls[0]["shape"] != [rows, 32, keys] or calls[0]["formats"] != [2, 2]):
        raise AssertionError(f"Unexpected real GEMM geometry/format: {calls}")
    if [len(x) for x in captures] != [1, 1]:
        raise AssertionError("Expected exactly one scored caller tile")
    return dict(scope="production_auto" if auto else "harness_forced_caller",
                mode=mode, rows=rows, heads=32, keys=keys, min_keys=min_keys,
                calls=calls, launches=events, actual_active=active,
                masked_logits=h.strict(captures[0][0], captures[1][0], "caller logits"),
                topk_int32=h.strict(old_ids, ids, "caller INT32 TopK"))


def graph_case(helper, mode, rows, keys, seed):
    # Fixed addresses, with changed score AND gate contents after capture.
    scores, weights = h.fixture(rows, keys, "random", seed, heads=32)
    target = scores.clone()

    def run():
        target.copy_(scores)
        operation(helper, mode, target, weights)
        logits, ids = h.finish(target, keys)
        return logits, ids

    with environment(mode, rows):
        for _ in range(2):
            run()
        h.torch.npu.synchronize()
        graph = h.torch.npu.NPUGraph()
        with observe_launches(helper) as events:
            with h.torch.npu.graph(graph):
                logits, ids = run()
        check_launch(events, mode, rows, keys)
        records = []
        for pattern in ("random", "edge_bits"):
            fresh_scores, fresh_weights = h.fixture(rows, keys, pattern, seed + 7, heads=32)
            scores.copy_(fresh_scores)
            weights.copy_(fresh_weights)
            del fresh_scores, fresh_weights
            reference = scores.clone().relu_().mul_(weights.unsqueeze(-1))
            expected_logits, expected_ids = h.finish(reference, keys)
            graph.replay()
            h.torch.npu.synchronize()
            records.append(dict(
                pattern=pattern, weighted=h.strict(reference, target, "graph weighted BF16"),
                masked_logits=h.strict(expected_logits, logits, "graph logits"),
                topk_int32=h.strict(expected_ids, ids, "graph ordered INT32 TopK"),
            ))
            del reference, expected_logits, expected_ids
    return dict(scope="graph_reset_postop_sum_topk", mode=mode, rows=rows, heads=32,
                keys=keys, capture_launches=events, changed_input_replays=records)


def timed_modes(functions, rows, repeat, warmup):
    values = {mode: dict(host_ms=[], event_ms=[]) for mode in MODES}
    for iteration in range(warmup + repeat):
        offset = iteration % len(MODES)
        order = MODES[offset:] + MODES[:offset]
        if iteration % 2:
            order = tuple(reversed(order))
        for mode in order:
            with environment(mode, rows):
                start = h.torch.npu.Event(enable_timing=True)
                end = h.torch.npu.Event(enable_timing=True)
                h.torch.npu.synchronize()
                begin = time.perf_counter()
                start.record()
                output = functions[mode]()
                end.record()
                end.synchronize()
                host_ms = (time.perf_counter() - begin) * 1000
                event_ms = start.elapsed_time(end)
                if iteration >= warmup:
                    values[mode]["host_ms"].append(host_ms)
                    values[mode]["event_ms"].append(event_ms)
                del output
    for value in values.values():
        for clock in ("host_ms", "event_ms"):
            value[clock + "_p50"] = statistics.median(value[clock])
    return values


def timing_case(module_file, helper, rows, keys, seed, repeat, warmup):
    q, weights, history, batch = h.query_fixture(rows, keys, seed, heads=32)
    indexer = SimpleNamespace(index_topk=2048, index_kpool=4)
    scores = h.torch.einsum("qhd,kd->qhk", q, history)
    if h.formats(scores, weights) != [2, 2]:
        raise AssertionError("Timing requires real ND GEMM scores/gates")
    target = scores.clone()

    def direct(mode):
        def run():
            target.copy_(scores)
            operation(helper, mode, target, weights)
            return target
        return run

    def caller(mode):
        fn = h.forced32_scorer(module_file, helper, 8192, mode != "native")
        return lambda: fn(indexer, q, weights, [history], batch)

    return dict(
        rows=rows, heads=32, keys=keys, block=8192, num_warps=4,
        repeat=repeat, warmup=warmup,
        direct_reset_plus_postop=timed_modes({m: direct(m) for m in MODES}, rows, repeat, warmup),
        harness_forced_caller=timed_modes({m: caller(m) for m in MODES}, rows, repeat, warmup),
        scope="Independent eager timings; no comparison/capture/probe; not production auto/TTFT",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module-file", type=Path, required=True)
    parser.add_argument("--helper-file", type=Path)
    parser.add_argument("--base-harness-file", type=Path,
                        default=Path(__file__).with_name("bench_glm53_postop64_ab.py"))
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--confirm-parent-stopped-p", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tiles", type=int, nargs="+", choices=(8, 32), default=[8, 32])
    parser.add_argument("--keys", type=int, nargs="+", choices=(8192, 16384, 32768),
                        default=[8192, 16384, 32768])
    parser.add_argument("--min-keys", type=int, choices=(8192, 16384, 32768),
                        help="Optional real P32 grouped admission override; default unset")
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1018)
    args = parser.parse_args()
    report = dict(schema="glm53-postop32-grouped-v1", status="running", correctness=[],
                  timings=[], correctness_pass=False, timing_used_for_correctness=False,
                  argv=sys.argv[1:], parent_stopped_p_asserted=args.confirm_parent_stopped_p,
                  limits="Finite fixtures; graph tests cover postop/sum/TopK; no model promotion")
    with args.output.open("x") as output:
        def save():
            output.seek(0)
            json.dump(report, output, indent=2, allow_nan=False)
            output.write("\n")
            output.truncate()
            output.flush()
        save()
        try:
            if not args.confirm_parent_stopped_p or not args.device.startswith("npu:"):
                raise ValueError("Explicit isolated NPU and --confirm-parent-stopped-p required")
            if args.repeat < 0 or args.warmup < 0:
                raise ValueError("repeat/warmup must be nonnegative")
            global h
            spec = importlib.util.spec_from_file_location("postop_base", args.base_harness_file)
            h = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(h)
            import torch
            import torch_npu
            h.torch, h.torch_npu, h.DEVICE = torch, torch_npu, args.device
            torch.set_num_threads(1)
            torch.npu.set_device(args.device)
            torch.npu.set_compile_mode(jit_compile=False)
            torch_npu.npu.config.allow_internal_format = True
            helper_file = args.helper_file or args.module_file.with_name("pooled_score_postop_npu.py")
            report.update(torch=torch.__version__, torch_npu=torch_npu.__version__,
                          source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in (args.module_file, helper_file,
                                                   args.base_harness_file, Path(__file__))})
            if GROUP not in args.module_file.read_text() or GROUP not in helper_file.read_text():
                raise ValueError("Missing production grouped opt-in; refusing inactive experiment")
            helper = h.load_helper(helper_file)

            def record(scope, mode, rows, keys, fn):
                report["active_case"] = dict(scope=scope, mode=mode, rows=rows, keys=keys)
                save()
                report["correctness"].append(fn())
                gc.collect()
                save()

            with torch.inference_mode():
                for rows in args.tiles:
                    for keys in args.keys:
                        for mode in MODES:
                            if mode != "native":
                                for pattern in h.PATTERNS:
                                    record("eager_" + pattern, mode, rows, keys,
                                           lambda: direct_case(helper, mode, rows, keys, pattern, args.seed))
                                record("forced_caller", mode, rows, keys,
                                       lambda: caller_case(args.module_file, helper, mode, rows, keys, args.seed))
                            if mode.startswith("group"):
                                record("production_auto", mode, rows, keys,
                                       lambda: caller_case(args.module_file, helper, mode, rows, keys,
                                                           args.seed, auto=True, min_keys=args.min_keys))
                            record("graph", mode, rows, keys,
                                   lambda: graph_case(helper, mode, rows, keys, args.seed))
                report["correctness_pass"] = True
                save()
                if args.repeat:
                    for rows in args.tiles:
                        for keys in args.keys:
                            report["active_case"] = dict(scope="independent_timing", rows=rows, keys=keys)
                            save()
                            report["timings"].append(timing_case(
                                args.module_file, helper, rows, keys, args.seed, args.repeat, args.warmup))
                            gc.collect()
                            save()
            report.update(status="pass", active_case=None)
            save()
            print(json.dumps(dict(status="pass", cases=len(report["correctness"]),
                                  output=str(args.output))), flush=True)
            return 0
        except Exception:
            report.update(status="fail", error=traceback.format_exc())
            save()
            print(report["error"], file=sys.stderr)
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
