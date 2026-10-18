"""Direct production pooled-TopK A/B; no SGLang package import.

NPU execution requires --confirm-parent-stopped-p. This is a caller assertion,
not a service check. The script never controls services. --help is stdlib-only.
Correctness copies only masked [Q,K] logits to CPU, not [Q,H,K] intermediates.
Baseline host storage is at most 128 MiB for T2048/K32768/BF16; candidate tiles
are compared and discarded immediately. Only JSON summaries are written.
Optional timing runs separately, without observers or device-to-host copies.
"""

import argparse
import ast
import copy
import gc
import hashlib
import json
import os
import statistics
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

FLAG = "SGLANG_GLM53_PREFILL_TOPK_METADATA_REUSE"
SOURCE = (
    Path(__file__).resolve().parents[3]
    / "python/sglang/srt/hardware_backend/npu/attention/glm53/kpool_indexer.py"
)
CASE_NAMES = ("aligned_random", "fragment_random", "page_ties")


@contextmanager
def environment(flag, tile):
    settings = {
        FLAG: str(flag),
        "SGLANG_GLM53_PREFILL_TOPK_TILE_ROWS": str(tile),
        "SGLANG_GLM53_PREFILL_POOLED_SCORE": "0",
        "SGLANG_GLM53_PREFILL_POOLED_POSTOP": "0",
        "SGLANG_GLM53_PREFILL_SCORE_VALIDATE": "0",
    }
    previous = {key: os.environ.get(key) for key in settings}
    os.environ.update(settings)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def extract(source, torch_api, observer=None):
    """Compile only the real method and three real guard functions.

    The instrumented copy adds one read-only callback after masked_fill_. The
    uninstrumented copy used for timing has no AST changes to the method.
    """
    tree = ast.parse(source)
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "AscendIndexerKPoolMixin"
    )
    scorer = copy.deepcopy(
        next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "_prefill_pooled_topk"
        )
    )
    if not any(
        isinstance(n, ast.Constant) and n.value == FLAG for n in ast.walk(scorer)
    ):
        raise ValueError("module-file has no metadata-reuse flag; refusing a no-op A/B")
    helpers = []
    for name in (
        "_prefill_topk_tile_rows",
        "_prefill_pooled_score_enabled",
        "_prefill_pooled_postop_min_keys",
    ):
        helpers.append(
            copy.deepcopy(
                next(
                    n
                    for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == name
                )
            )
        )

    class Instrument(ast.NodeTransformer):
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
                probe = ast.parse(
                    "_observer(logits, i, first_pos, start, end, candidate_count, "
                    "request_visible_pools is not None)"
                ).body[0]
                return [node, ast.copy_location(probe, node)]
            return node

    if observer is not None:
        instrument = Instrument()
        scorer = instrument.visit(scorer)
        if instrument.count != 1:
            raise ValueError("expected exactly one masked-logits observation point")
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *helpers,
            scorer,
        ],
        type_ignores=[],
    )
    namespace = dict(
        torch=torch_api,
        F=torch.nn.functional,
        os=os,
        get_server_args=lambda: SimpleNamespace(disaggregation_mode="prefill"),
        _observer=observer,
    )
    exec(
        compile(ast.fix_missing_locations(module), "<production-scorer>", "exec"),
        namespace,
    )
    return namespace["_prefill_pooled_topk"]


class CPUFactories:
    """Explicit CPU smoke only: tag the outer query and redirect factories."""

    def __getattr__(self, name):
        return getattr(torch, name)

    def arange(self, *args, **kwargs):
        kwargs["device"] = "cpu"
        return torch.arange(*args, **kwargs)

    def full(self, *args, **kwargs):
        kwargs["device"] = "cpu"
        return torch.full(*args, **kwargs)


class CPUQuery:
    device = SimpleNamespace(type="npu")

    def __init__(self, tensor):
        self.tensor = tensor

    def __getitem__(self, key):
        return self.tensor[key]


def host_copy(tensor):
    value = tensor.detach().to("cpu").contiguous()
    return value.clone() if tensor.device.type == "cpu" else value


def raw_bytes(tensor):
    return tensor.contiguous().view(torch.uint8).numpy().tobytes()


def compare(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return {"strict_equal": False, "reason": "shape/dtype mismatch"}
    different = a.view(torch.uint8) != b.view(torch.uint8)
    mismatch_bytes = int(different.sum().item())
    return {
        "strict_equal": bool(torch.equal(a, b)) and mismatch_bytes == 0,
        "torch_equal": bool(torch.equal(a, b)),
        "mismatch_bytes": mismatch_bytes,
        "shape": list(a.shape),
        "dtype": str(a.dtype),
    }


def tile_plan(tokens, first_pos, history_pools, tile):
    plan, start = [], 0
    while start < tokens:
        stop = min(first_pos + tokens, ((first_pos + start) // 128 + 1) * 128)
        end = min(stop - first_pos, start + tile)
        count = min(stop // 4, history_pools)
        if count:
            plan.append((0, first_pos, start, end, count))
        start = end
    return plan


class MaskedLogits:
    def __init__(self, plan, baseline=None):
        self.plan, self.baseline = plan, baseline
        self.tiles = [] if baseline is None else None
        self.seen = 0
        self.equal = True
        self.mismatch_bytes = 0
        self.failures = []
        self.digest = hashlib.sha256()
        self.host_bytes = 0

    def __call__(self, logits, request, first, start, end, count, reused):
        metadata = (request, first, start, end, count)
        if self.seen >= len(self.plan) or metadata != self.plan[self.seen]:
            raise AssertionError(
                "observed score tiling differs from expected causal plan"
            )
        if reused != (self.baseline is not None):
            raise AssertionError("flag did not select the expected metadata branch")
        host = host_copy(logits)
        if host.dtype != torch.bfloat16 or tuple(host.shape) != (end - start, count):
            raise AssertionError("expected BF16 [tile_rows, candidate_pools] logits")
        # Negative infinity is the intended causal mask; NaN/+Inf are failures.
        if not bool((torch.isfinite(host) | torch.isneginf(host)).all()):
            raise AssertionError("unexpected NaN/+Inf in masked logits")
        self.digest.update(raw_bytes(host))
        if self.baseline is None:
            self.tiles.append(host)
            self.host_bytes += host.numel() * host.element_size()
        else:
            result = compare(host, self.baseline[self.seen])
            self.equal &= result["strict_equal"]
            self.mismatch_bytes += result.get("mismatch_bytes", 0)
            if not result["strict_equal"] and len(self.failures) < 3:
                self.failures.append(
                    {"tile": self.seen, "metadata": metadata, **result}
                )
        self.seen += 1

    def summary(self):
        if self.seen != len(self.plan):
            raise AssertionError("missing masked-logit observations")
        return {
            "tiles": self.seen,
            "strict_equal": self.equal,
            "sha256": self.digest.hexdigest(),
            "mismatch_bytes": self.mismatch_bytes,
            "failures": self.failures,
            "baseline_host_bytes": self.host_bytes,
        }


def case_inputs(args, tokens, name):
    history = 8192 if args.cpu_smoke else 131072
    heads, width = (2, 8) if args.cpu_smoke else (args.heads, 128)
    first = {
        "aligned_random": history - tokens,
        "fragment_random": history // 2 + 63,
        "page_ties": history - tokens - 1,
    }[name]
    gen = torch.Generator(device="cpu").manual_seed(
        args.seed + tokens + CASE_NAMES.index(name)
    )
    q = (torch.randn(tokens, heads, width, generator=gen) * 0.1).bfloat16()
    weights = torch.randn(tokens, heads, generator=gen)
    keys = (torch.randn(history // 4, width, generator=gen) * 0.1).bfloat16()
    if name == "page_ties":
        q.zero_()  # All visible scores tie; compare the actual ordered IDs.
    fingerprints = {
        key: hashlib.sha256(raw_bytes(value)).hexdigest()
        for key, value in (("q", q), ("weights", weights), ("keys", keys))
    }
    tensors = [value.to(args.device) for value in (q, weights, keys)]
    formats = None
    if args.device.startswith("npu"):
        tensors = [torch_npu.npu_format_cast(value, 2) for value in tensors]
        formats = [torch_npu.get_npu_format(value) for value in tensors]
        if formats != [2, 2, 2]:
            raise AssertionError("inputs must use physical ND format 2")
    q, weights, keys = tensors
    if args.cpu_smoke:
        q = CPUQuery(q)
    batch = SimpleNamespace(
        batch_size=1, extend_seq_lens_cpu=[tokens], seq_lens_cpu=[first + tokens]
    )
    description = {
        "name": name,
        "tokens": tokens,
        "first_pos": first,
        "sequence_end": first + tokens,
        "history_tokens": history,
        "heads": heads,
        "head_dim": width,
        "index_topk": 2048,
        "index_kpool": 4,
        "physical_formats": formats,
        "input_sha256": fingerprints,
    }
    return (q, weights, [keys], batch), description


def correctness(source, api, args, tokens, tile, name):
    data, description = case_inputs(args, tokens, name)
    plan = tile_plan(
        tokens, description["first_pos"], description["history_tokens"] // 4, tile
    )
    old_trace = MaskedLogits(plan)
    old = extract(source, api, old_trace)
    indexer = SimpleNamespace(index_topk=2048, index_kpool=4)
    with environment(0, tile):
        old_ids = host_copy(old(indexer, *data))
    old_summary = old_trace.summary()
    new_trace = MaskedLogits(plan, old_trace.tiles)
    new = extract(source, api, new_trace)
    with environment(1, tile):
        new_ids = host_copy(new(indexer, *data))
    if old_ids.dtype != torch.int32 or new_ids.dtype != torch.int32:
        raise AssertionError("final TopK IDs must be INT32")
    if tuple(old_ids.shape) != (tokens, 512):
        raise AssertionError("unexpected final TopK shape")
    new_summary = new_trace.summary()
    ids = compare(new_ids, old_ids)
    ids.update(
        baseline_sha256=hashlib.sha256(raw_bytes(old_ids)).hexdigest(),
        candidate_sha256=hashlib.sha256(raw_bytes(new_ids)).hexdigest(),
    )
    result = {
        **description,
        "tile_rows": tile,
        "masked_logits": new_summary,
        "baseline_masked_logits": old_summary,
        "topk_int32": ids,
        "branches_verified": [0, 1],
        "pass": new_summary["strict_equal"] and ids["strict_equal"],
    }
    # Only these summaries escape this function. All tensors/traces are freed.
    return result


def sync(args):
    if args.device.startswith("npu"):
        torch.npu.synchronize()


def timing(source, api, args, tokens, tile, name):
    data, description = case_inputs(args, tokens, name)
    scorer = extract(source, api)  # No observation hook or D2H copies.
    indexer = SimpleNamespace(index_topk=2048, index_kpool=4)
    samples = {0: [], 1: []}
    for iteration in range(args.warmup + args.repeat):
        for flag in (0, 1) if iteration % 2 == 0 else (1, 0):
            with environment(flag, tile):
                sync(args)
                start = time.perf_counter()
                out = scorer(indexer, *data)
                sync(args)
                elapsed = (time.perf_counter() - start) * 1000
                del out
            if iteration >= args.warmup:
                samples[flag].append(elapsed)
    return {
        **description,
        "tile_rows": tile,
        "warmup_pairs": args.warmup,
        "repeat_pairs": args.repeat,
        "observer_enabled": False,
        "metric": "synchronized_host_ms_not_TTFT",
        "variants": {
            str(flag): {"samples_ms": values, "median_ms": statistics.median(values)}
            for flag, values in samples.items()
        },
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module-file", type=Path, default=SOURCE)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--confirm-parent-stopped-p",
        action="store_true",
        help="required caller assertion for NPU; does not stop/check services",
    )
    parser.add_argument(
        "--tokens", type=int, nargs="+", choices=(512, 2048), default=[512, 2048]
    )
    parser.add_argument(
        "--tile-rows", type=int, nargs="+", choices=(8, 32), default=[8, 32]
    )
    parser.add_argument(
        "--cases", nargs="+", choices=CASE_NAMES, default=list(CASE_NAMES)
    )
    parser.add_argument("--heads", type=int, choices=(32, 64), default=32)
    parser.add_argument("--seed", type=int, default=1018)
    parser.add_argument(
        "--repeat",
        type=int,
        default=0,
        help="separate timing pairs AFTER all correctness passes; 0 skips timing",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--cpu-smoke",
        action="store_true",
        help="only with --device cpu: reduced geometry and tagged query; NOT NPU evidence",
    )
    args = parser.parse_args(argv)
    if args.repeat < 0 or args.warmup < 0:
        parser.error("repeat and warmup must be nonnegative")
    if args.device != "cpu" and not (
        args.device.startswith("npu:") and args.device[4:].isdigit()
    ):
        parser.error("device must be cpu or npu:<index>; CUDA is unsupported")
    if (args.device == "cpu") != args.cpu_smoke:
        parser.error("CPU execution requires --device cpu --cpu-smoke together")
    return args


def main(argv=None):
    args = parse_args(argv)
    report = {
        "schema": "glm53-prefill-topk-metadata-direct-ab-v1",
        "status": "starting",
        "device": args.device,
        "cpu_smoke_only": args.cpu_smoke,
        "parent_stopped_p_asserted": args.confirm_parent_stopped_p,
        "controls": {"pooled_score": "0", "pooled_postop": "0", "score_validate": "0"},
        "correctness": [],
        "timings": [],
        "timing_used_for_correctness": False,
        "failure_history": "Prior five-change accuracy 1264/1319 <1267; approximate Sinkhorn rejected (42/300).",
    }
    # Exclusive creation before importing any device library; never overwrite a receipt.
    with args.output.open("x") as output:

        def save():
            output.seek(0)
            json.dump(report, output, indent=2, allow_nan=False)
            output.write("\n")
            output.truncate()
            output.flush()

        save()
        try:
            if not args.cpu_smoke and not args.confirm_parent_stopped_p:
                raise ValueError(
                    "NPU run requires --confirm-parent-stopped-p; no device library imported"
                )
            source = args.module_file.read_text()
            report["module_file"] = str(args.module_file.resolve())
            report["module_sha256"] = hashlib.sha256(source.encode()).hexdigest()
            report["script_sha256"] = hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest()
            # No CUDA discovery and no sglang imports, even on CPU smoke.
            os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            global torch, torch_npu
            import torch

            torch.set_num_threads(1)
            report["torch_version"] = str(torch.__version__)
            if not args.cpu_smoke:
                import torch_npu

                torch.npu.set_device(args.device)
                report["torch_npu_version"] = str(torch_npu.__version__)
            api = CPUFactories() if args.cpu_smoke else torch
            matrix = [
                (n, tile, case)
                for n in args.tokens
                for tile in args.tile_rows
                for case in args.cases
            ]
            report["expected_cases"] = len(matrix)
            report["status"] = "validating"
            save()
            with torch.inference_mode():
                for tokens, tile, name in matrix:
                    report["current_case"] = {
                        "tokens": tokens,
                        "tile_rows": tile,
                        "name": name,
                    }
                    save()
                    result = correctness(source, api, args, tokens, tile, name)
                    report["correctness"].append(result)
                    save()
                    print(
                        json.dumps(
                            {
                                "case": name,
                                "tokens": tokens,
                                "tile": tile,
                                "pass": result["pass"],
                            }
                        ),
                        flush=True,
                    )
                    gc.collect()
                    if not result["pass"]:
                        report["status"] = "failed"
                        save()
                        return 1
                report["correctness_pass"] = True
                if args.repeat:
                    report["status"] = "timing"
                    save()
                    for tokens, tile, name in matrix:
                        report["current_case"] = {
                            "tokens": tokens,
                            "tile_rows": tile,
                            "name": name,
                        }
                        save()
                        report["timings"].append(
                            timing(source, api, args, tokens, tile, name)
                        )
                        save()
                        gc.collect()
            report["status"] = "cpu_smoke_pass" if args.cpu_smoke else "pass"
            save()
            return 0
        except Exception:
            report["status"] = "error"
            report["traceback"] = traceback.format_exc()
            save()
            print(report["traceback"], file=sys.stderr)
            return 2


if __name__ == "__main__":
    raise SystemExit(main())
