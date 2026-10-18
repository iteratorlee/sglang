"""Strict 64-head NPU postop A/B, with no SGLang package registration.

Extends the 1012 postop tests: byte comparison (including NaN payloads and
signed zero), native head sum, causal mask and ordered INT32 TopK. Direct
helper tests force its opt-in at every requested shape; separate extracted
production-caller tests obey the actual auto thresholds. Correctness finishes
before independent event/host timing. No full-query intermediate tape or
tensor files: at most one 32-row score tile per variant, compared row by row.

--confirm-parent-stopped-p asserts an externally managed isolation window.
This script does not inspect/control services. --help imports only stdlib.
"""

import argparse
import ast
import copy
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

FLAG = "SGLANG_GLM53_PREFILL_POOLED_POSTOP64"
REL = "python/sglang/srt/hardware_backend/npu/attention/glm53"
PATTERNS = ("random", "edge_bits", "rank_boundary")


@contextmanager
def environment(flag, tile):
    settings = {
        FLAG: str(flag),
        "SGLANG_GLM53_PREFILL_POOLED_POSTOP": "auto",
        "SGLANG_GLM53_PREFILL_POOLED_SCORE": "0",
        "SGLANG_GLM53_PREFILL_SCORE_VALIDATE": "0",
        "SGLANG_GLM53_PREFILL_TOPK_TILE_ROWS": str(tile),
        "SGLANG_GLM53_PREFILL_TOPK_METADATA_REUSE": "1",
    }
    previous = {k: os.environ.get(k) for k in settings}
    os.environ.update(settings)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def extract_defs(path, names, namespace, observer=None):
    """Extract real functions; optionally observe only the masked logits."""
    tree = ast.parse(Path(path).read_text())
    nodes = []
    for name in names:
        matches = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == name
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one production function: {name}")
        nodes.append(copy.deepcopy(matches[0]))

    class Observe(ast.NodeTransformer):
        count = 0

        def visit_Expr(self, node):
            call = node.value
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "logits"
                and call.func.attr == "masked_fill_"
            ):
                self.count += 1
                probe = ast.parse("_observe(logits)").body[0]
                return [node, ast.copy_location(probe, node)]
            return node

    if observer is not None:
        transform = Observe()
        nodes = [transform.visit(n) for n in nodes]
        if transform.count != 1:
            raise ValueError("Expected one production masked_fill_ observation point")
        namespace["_observe"] = observer
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


def load_helper(path):
    spec = importlib.util.spec_from_file_location("isolated_postop64", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def strict(a, b, label):
    """Stream one row to host; no tolerance, sorting, or hash-only acceptance."""
    if a.shape != b.shape or a.dtype != b.dtype:
        raise AssertionError(f"{label}: shape/dtype mismatch")
    hashes = [hashlib.sha256(), hashlib.sha256()]
    different = 0
    for row in range(a.shape[0]):
        x = a[row].detach().cpu().contiguous().view(torch.uint8)
        y = b[row].detach().cpu().contiguous().view(torch.uint8)
        hashes[0].update(x.numpy().tobytes())
        hashes[1].update(y.numpy().tobytes())
        different += int(torch.count_nonzero(x != y))
    result = dict(
        shape=list(a.shape),
        dtype=str(a.dtype),
        mismatch_bytes=different,
        sha256=[h.hexdigest() for h in hashes],
        strict_equal=different == 0,
    )
    if different:
        raise AssertionError(f"{label}: {different} differing bytes; {result}")
    return result


def formats(*tensors):
    return [torch_npu.get_npu_format(t) for t in tensors]


def nd(tensor):
    return torch_npu.npu_format_cast(tensor.contiguous().to(DEVICE), 2)


def fixture(rows, keys, pattern, seed, heads=64):
    generator = torch.Generator().manual_seed(seed)
    if pattern == "edge_bits":
        # Rotate all BF16 storage patterns across heads/keys. Selected FP32
        # gates follow the 1012 exhaustive test, including both signs and
        # rounding neighbors. This is not all possible FP32 gate patterns.
        positive = [
            0,
            0x3F800000,
            0x3F808000,
            0x3F808001,
            1,
            0x007FFFFF,
            0x00800000,
            0x00800001,
            0x33800000,
            0x4B800000,
            0x7F7FFFFF,
            0x7F800000,
            0x7FC00000,
            0x7F800001,
            0x7FFFFFFF,
            0x3EAAAAAB,
        ]
        unsigned = positive + [x | 0x80000000 for x in positive]
        bits = [x if x < 2**31 else x - 2**32 for x in unsigned]
        gates = torch.tensor(bits, dtype=torch.int32).view(torch.float32)
        weights = gates.repeat((rows * heads + 31) // 32)[: rows * heads].reshape(
            rows, heads
        )
        # Build one head at a time on CPU, then broadcast only the row axis.
        bits = ((torch.arange(heads * keys, dtype=torch.int32) % 65536) - 32768).short()
        scores = nd(
            bits.view(torch.bfloat16).reshape(1, heads, keys).expand(rows, -1, -1)
        )
    elif pattern == "rank_boundary":
        scores = torch.zeros((rows, heads, keys), dtype=torch.bfloat16)
        weights = torch.zeros((rows, heads), dtype=torch.float32)
        weights[:, :4] = torch.tensor([1.0, 1.0, -1.0, 1.0])
        scores[:, 0, :511] = 4
        scores[:, :4, 511] = torch.tensor([256.0, 2.0, 256.0, 0.0])
        scores[:, 0, 512:514] = 2.5  # Rank boundary and a true score tie.
        scores[:, 0, 514:516] = torch.tensor([2.484375, 2.515625])
        scores = nd(scores)
    else:
        scores = nd(torch.randn((rows, heads, keys), generator=generator).bfloat16())
        weights = torch.randn((rows, heads), generator=generator)
        weights[:, :4] = torch.tensor([0.0, -0.0, 1.00390625, -1.0039063692092896])
    return scores, nd(weights)


def finish(weighted, keys):
    logits = weighted.sum(1)  # Same native head reduction and BF16 storage.
    rows = logits.shape[0]
    first_pos = keys * 4 - rows - 1  # TP/page boundary offset; final pool invisible.
    visible = torch.div(
        torch.arange(rows, device=DEVICE) + first_pos + 1, 4, rounding_mode="floor"
    )
    pool_ids = torch.arange(keys, device=DEVICE)
    logits.masked_fill_(pool_ids.unsqueeze(0) >= visible.unsqueeze(1), float("-inf"))
    values, ids = logits.topk(min(512, keys), dim=-1)
    ids = ids.to(torch.int32).masked_fill(torch.isneginf(values), -1)
    return logits, ids


def helper_case(module, rows, keys, pattern, seed, heads=64):
    scores, weights = fixture(rows, keys, pattern, seed, heads)
    if formats(scores, weights) != [2, 2]:
        raise AssertionError("ND input contract not observed")
    reference = scores.clone().relu_().mul_(weights.unsqueeze(-1))
    with environment(0, rows):
        baseline = scores.clone()
        off_active = module.relu_gate_(baseline, weights)
    strict(reference, baseline, "flag0/native weighted scores")
    del reference
    with environment(1, rows):
        candidate = scores.clone()
        on_active = module.relu_gate_(candidate, weights)
    if off_active != (heads == 32) or not on_active:
        raise AssertionError(f"Unexpected helper activation: {off_active}/{on_active}")
    weighted = strict(baseline, candidate, "weighted BF16 scores")
    old_logits, old_ids = finish(baseline, keys)
    new_logits, new_ids = finish(candidate, keys)
    return dict(
        scope="direct_helper",
        rows=rows,
        heads=heads,
        keys=keys,
        pattern=pattern,
        physical_formats=formats(scores, weights),
        active=[off_active, on_active],
        weighted=weighted,
        masked_logits=strict(old_logits, new_logits, "masked logits"),
        topk_int32=strict(old_ids, new_ids, "ordered INT32 TopK"),
    )


def scorer(module_file, helper, observer=None, calls=None):
    def operation(scores, weights):
        active = helper.relu_gate_(scores, weights)
        if calls is not None:
            calls.append(
                dict(
                    shape=list(scores.shape),
                    formats=formats(scores, weights),
                    active=active,
                )
            )
        return active

    scope = dict(
        torch=torch,
        F=torch.nn.functional,
        os=os,
        get_server_args=lambda: SimpleNamespace(disaggregation_mode="prefill"),
        _prefill_pooled_postop_backend=lambda: operation,
        _log_prefill_pooled_postop_once=lambda *_: None,
    )
    extract_defs(
        module_file,
        [
            "_prefill_topk_tile_rows",
            "_prefill_pooled_score_enabled",
            "_prefill_pooled_postop_min_keys",
            "_prefill_pooled_topk",
        ],
        scope,
        observer,
    )
    return scope["_prefill_pooled_topk"]


def query_fixture(rows, keys, seed):
    generator = torch.Generator().manual_seed(seed)
    q = nd(torch.randn((rows, 64, 128), generator=generator).bfloat16())
    weights = nd(torch.randn((rows, 64), generator=generator))
    history = nd(torch.randn((keys, 128), generator=generator).bfloat16())
    batch = SimpleNamespace(
        batch_size=1, extend_seq_lens_cpu=[rows], seq_lens_cpu=[keys * 4]
    )
    return q, weights, history, batch


def caller_case(module_file, helper, rows, keys, seed):
    q, weights, history, batch = query_fixture(rows, keys, seed)
    indexer = SimpleNamespace(index_topk=2048, index_kpool=4)
    calls, logits = [[], []], [[], []]
    outputs = []
    for flag in (0, 1):
        fn = scorer(
            module_file,
            helper,
            lambda x: logits[flag].append(x.detach().cpu()),
            calls[flag],
        )
        with environment(flag, rows):
            outputs.append(fn(indexer, q, weights, [history], batch))
    # One tile terminating exactly at this causal block's end.
    expected_calls = [
        int(keys >= {32: 32768, 8: 98304}[rows]),
        int(keys >= {32: 16384, 8: 49152}[rows]),
    ]
    if [len(x) for x in calls] != expected_calls:
        raise AssertionError(
            f"Wrong production helper calls: {calls}; expected {expected_calls}"
        )
    if any(c["active"] for c in calls[0]) or not all(c["active"] for c in calls[1]):
        raise AssertionError(f"Wrong production activation: {calls}")
    if len(logits[0]) != 1 or len(logits[1]) != 1:
        raise AssertionError("Expected exactly one scored production tile")
    return dict(
        scope="production_auto_caller",
        rows=rows,
        heads=64,
        keys=keys,
        physical_formats=formats(q, weights, history),
        calls=calls,
        expected_calls=expected_calls,
        masked_logits=strict(logits[0][0], logits[1][0], "production masked logits"),
        topk_int32=strict(outputs[0], outputs[1], "production ordered INT32 TopK"),
    )


def fallback_cases(helper):
    records = []
    for kind in ("noncontiguous", "bf16_weights", "format29", "row_limit", "heads48"):
        rows, heads = (33 if kind == "row_limit" else 2), (
            48 if kind == "heads48" else 64
        )
        scores, weights = fixture(rows, 64, "random", 1018, heads)
        if kind == "noncontiguous":
            scores = scores[:, :, ::2]
        elif kind == "bf16_weights":
            weights = weights.bfloat16()
        elif kind == "format29":
            scores = torch_npu.npu_format_cast(scores, 29)
            if torch_npu.get_npu_format(scores) != 29:
                raise AssertionError("Physical format29 fallback was not exercised")
        expected = scores.clone().relu_().mul_(weights.unsqueeze(-1))
        before = formats(scores, weights)
        with environment(1, 32):
            active = helper.relu_gate_(scores, weights)
        if active:
            raise AssertionError(f"Fallback unexpectedly fused: {kind}")
        records.append(
            dict(
                case=kind,
                formats_before=before,
                actual_active=active,
                weighted=strict(expected, scores, f"fallback {kind}"),
            )
        )
        del scores, weights, expected
    return records


def timed_pair(functions, repeat, warmup):
    """No observation/correctness copies. Both clocks include identical scopes."""
    values = [{"host_ms": [], "event_ms": []}, {"host_ms": [], "event_ms": []}]
    for index in range(warmup + repeat):
        for flag in (0, 1) if index % 2 == 0 else (1, 0):
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            torch.npu.synchronize()
            begin = time.perf_counter()
            start.record()
            output = functions[flag]()
            end.record()
            end.synchronize()
            host_ms = (time.perf_counter() - begin) * 1000
            event_ms = start.elapsed_time(end)
            if index >= warmup:
                values[flag]["host_ms"].append(host_ms)
                values[flag]["event_ms"].append(event_ms)
            del output
    for value in values:
        for clock in ("host_ms", "event_ms"):
            value[f"{clock}_p50"] = statistics.median(value[clock])
    return values


def timing_case(module_file, helper, rows, keys, repeat, warmup, seed):
    q, weights, history, batch = query_fixture(rows, keys, seed)
    indexer = SimpleNamespace(index_topk=2048, index_kpool=4)
    scores = torch.einsum("qhd,kd->qhk", q, history)
    physical_formats = formats(scores, weights)
    if physical_formats != [2, 2]:
        raise AssertionError(
            f"Timing requires real ND GEMM scores/gates: {physical_formats}"
        )
    target = scores.clone()
    fn = scorer(module_file, helper)  # Actual method, no observer/call collector.

    def direct(flag):
        def run():
            with environment(flag, rows):
                target.copy_(scores)  # Included symmetrically; clearly labeled.
                active = helper.relu_gate_(target, weights)
                if active != bool(flag):
                    raise AssertionError(
                        "Timing helper activation disagrees with correctness"
                    )
            return target

        return run

    def production(flag):
        def run():
            with environment(flag, rows):
                return fn(indexer, q, weights, [history], batch)

        return run

    return dict(
        rows=rows,
        heads=64,
        keys=keys,
        direct_physical_formats=physical_formats,
        direct_expected_active=[False, True],
        repeat_pairs=repeat,
        warmup_pairs=warmup,
        direct_reset_plus_postop=timed_pair([direct(0), direct(1)], repeat, warmup),
        production_auto_caller=timed_pair(
            [production(0), production(1)], repeat, warmup
        ),
        production_expected_active=[False, keys >= {32: 16384, 8: 49152}[rows]],
        scope="Independent synchronized host/NPU events; not TTFT; timing not used for correctness",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--module-file", type=Path, required=True, help="Production kpool_indexer.py"
    )
    parser.add_argument(
        "--helper-file",
        type=Path,
        help="Defaults to sibling pooled_score_postop_npu.py",
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confirm-parent-stopped-p", action="store_true")
    parser.add_argument(
        "--tiles", type=int, nargs="+", choices=(8, 32), default=[8, 32]
    )
    parser.add_argument("--keys", type=int, nargs="+", default=[8192, 16384, 32768])
    parser.add_argument("--repeat", type=int, default=10, help="0: correctness only")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1018)
    args = parser.parse_args()
    report = dict(
        schema="glm53-postop64-direct-and-production-ab-v1",
        status="running",
        correctness=[],
        fallbacks=[],
        timings=[],
        correctness_pass=False,
        timing_used_for_correctness=False,
        parent_stopped_p_asserted=args.confirm_parent_stopped_p,
        device=args.device,
        argv=sys.argv[1:],
        limits="Finite fixtures only; no model/D100/accuracy or performance promotion",
        failure_history="Old GSM1264/1319<1267; approximate Sinkhorn42/300 rejected",
    )
    # Exclusive create: never overwrite an earlier failed or successful receipt.
    with args.output.open("x") as output:

        def save():
            output.seek(0)
            json.dump(report, output, indent=2, allow_nan=False)
            output.write("\n")
            output.truncate()
            output.flush()

        save()
        try:
            if not args.confirm_parent_stopped_p:
                raise ValueError(
                    "Require --confirm-parent-stopped-p before importing device libraries"
                )
            if not args.device.startswith("npu:") or args.repeat < 0 or args.warmup < 0:
                raise ValueError(
                    "Require an explicit npu device and nonnegative repeat/warmup"
                )
            if not args.keys or any(k < 1024 or k > 49152 for k in args.keys):
                raise ValueError(
                    "Bounded keys range is [1024,49152]; default 8192/16384/32768"
                )
            global torch, torch_npu, DEVICE
            import torch
            import torch_npu

            DEVICE = args.device
            torch.set_num_threads(1)
            torch.npu.set_device(DEVICE)
            torch.npu.set_compile_mode(jit_compile=False)
            torch_npu.npu.config.allow_internal_format = True
            helper_file = args.helper_file or args.module_file.with_name(
                "pooled_score_postop_npu.py"
            )
            report.update(
                torch=torch.__version__,
                torch_npu=torch_npu.__version__,
                source_sha256={
                    str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (args.module_file, helper_file, Path(__file__))
                },
            )
            if (
                FLAG not in helper_file.read_text()
                or FLAG not in args.module_file.read_text()
            ):
                raise ValueError(
                    "Both production files must contain the opt-in; refusing a no-op A/B"
                )
            helper = load_helper(helper_file)
            with torch.inference_mode():
                for rows in args.tiles:
                    for keys in args.keys:
                        for pattern in PATTERNS:
                            report["active_case"] = dict(
                                scope="direct", rows=rows, keys=keys, pattern=pattern
                            )
                            save()
                            report["correctness"].append(
                                helper_case(helper, rows, keys, pattern, args.seed)
                            )
                            gc.collect()
                            save()
                        report["active_case"] = dict(
                            scope="production_auto", rows=rows, keys=keys
                        )
                        save()
                        report["correctness"].append(
                            caller_case(args.module_file, helper, rows, keys, args.seed)
                        )
                        gc.collect()
                        save()
                report["active_case"] = dict(scope="legacy32_unchanged_and_fallbacks")
                save()
                report["legacy32"] = helper_case(
                    helper, 8, 1024, "edge_bits", args.seed, heads=32
                )
                report["fallbacks"] = fallback_cases(helper)
                report["correctness_pass"] = True
                save()
                if args.repeat:
                    for rows in args.tiles:
                        for keys in args.keys:
                            report["active_case"] = dict(
                                scope="independent_timing", rows=rows, keys=keys
                            )
                            save()
                            report["timings"].append(
                                timing_case(
                                    args.module_file,
                                    helper,
                                    rows,
                                    keys,
                                    args.repeat,
                                    args.warmup,
                                    args.seed,
                                )
                            )
                            gc.collect()
                            save()
            report.update(status="pass", active_case=None)
            save()
            print(
                json.dumps(
                    {
                        "status": report["status"],
                        "cases": len(report["correctness"]),
                        "output": str(args.output),
                    }
                ),
                flush=True,
            )
            return 0
        except Exception:
            report.update(status="fail", error=traceback.format_exc())
            save()
            print(report["error"], file=sys.stderr)
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
