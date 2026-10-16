"""Standalone single-NPU numerical gate and full mHC-pre benchmark.

Parent must stop P and reserve one NPU before explicitly running this script.
No server import, service operations, SSH or profiler. Checkpoint tensors are
read with safetensors on CPU. Activations are synthetic, not captured traffic.
"""

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import statistics
import time
import traceback
from pathlib import Path
from typing import Tuple

import torch

SEED = 20261016
EPS = 1e-6
ITERS = 20
ROWS = (4097, 8192, 16384)
# Fixed before NPU evaluation: PyTorch's default allclose tolerances, not
# tunable on the command line. Bit equality is required for unaffected outputs.
COMB_RTOL = 1e-5
COMB_ATOL = 1e-8


def function_ast(path, name):
    return next(
        n
        for n in ast.parse(Path(path).read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == name
    )


def load_reference(path):
    fn = function_ast(path, "_mhc_pre_torch")
    env = {"torch": torch, "Tuple": Tuple}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), env)
    return env[fn.name]


def check_only_sinkhorn_changes(baseline, candidate):
    """Restore the optional hook and compare the entire pre AST to 45add04."""
    old, new = [function_ast(p, "_mhc_pre_torch") for p in (baseline, candidate)]
    assert new.args.args[-1].arg == "sinkhorn"
    assert isinstance(new.args.defaults[-1], ast.Constant)
    assert new.args.defaults[-1].value is None
    new.args.args.pop()
    new.args.defaults.pop()
    branches = [
        n
        for n in new.body
        if isinstance(n, ast.If) and ast.unparse(n.test) == "sinkhorn is None"
    ]
    assert len(branches) == 1
    branch = branches[0]
    pos = new.body.index(branch)
    new.body[pos : pos + 1] = branch.body
    assert ast.dump(old, include_attributes=False) == ast.dump(
        new, include_attributes=False
    ), "Changes outside the post-softmax normalization hook"


def native_sinkhorn(comb, iters=ITERS, eps=EPS):
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return comb


def comparison(actual, reference):
    """Metrics in CPU FP64; allclose uses the fixed tolerances above."""
    a, b = [x.detach().cpu().double() for x in (actual, reference)]
    assert a.shape == b.shape
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    diff = (a - b).abs()
    # Relative error uses the reference magnitude; an exact zero with nonzero
    # error is explicitly infinite, never hidden by an arbitrary denominator.
    relative = torch.where(b != 0, diff / b.abs(), torch.where(diff == 0, 0, math.inf))
    maxabs = float(diff.max()) if diff.numel() else 0.0
    maxrel = float(relative.max()) if diff.numel() else 0.0
    return {
        "elements": a.numel(),
        "finite": finite,
        "maxabs": maxabs if math.isfinite(maxabs) else None,
        "maxrel": maxrel if math.isfinite(maxrel) else None,
        "maxrel_nonfinite": not math.isfinite(maxrel),
        "allclose": finite and torch.allclose(a, b, rtol=COMB_RTOL, atol=COMB_ATOL),
        "rtol": COMB_RTOL,
        "atol": COMB_ATOL,
    }


def compare_bits(a, b, row_chunk=128):
    """Bound CPU transfer memory for full [T,4,4096] BF16 post outputs."""
    assert a.shape == b.shape and a.dtype == b.dtype
    integer = torch.int16 if a.dtype == torch.bfloat16 else torch.int32
    mismatches, maxabs, finite = 0, 0.0, True
    for start in range(0, a.shape[0], row_chunk):
        aa, bb = [
            x[start : start + row_chunk].detach().cpu().contiguous() for x in (a, b)
        ]
        mismatches += int((aa.view(integer) != bb.view(integer)).sum())
        good = bool(torch.isfinite(aa).all() and torch.isfinite(bb).all())
        finite &= good
        if good and aa.numel():
            maxabs = max(maxabs, float((aa.float() - bb.float()).abs().max()))
    return {
        "elements": a.numel(),
        "bit_mismatches": mismatches,
        "difference_ratio": mismatches / a.numel() if a.numel() else 0.0,
        "maxabs": maxabs if finite else None,
        "finite": finite,
        "bitwise_equal": mismatches == 0,
    }


def write_result(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def checkpoint_parameters(model):
    from safetensors import safe_open

    index = next(
        (
            model / name
            for name in (
                "quant_model_weights.safetensors.index.json",
                "model.safetensors.index.json",
            )
            if (model / name).is_file()
        ),
        None,
    )
    if index is None:
        raise FileNotFoundError(f"Missing checkpoint weight index in {model}")
    weights = json.loads(index.read_text())["weight_map"]

    def get(suffixes):
        keys = [k for k in weights if any(k.endswith(s) for s in suffixes)]
        if len(keys) != 1:
            raise ValueError(f"Expected one weight for {suffixes}: {keys}")
        key = keys[0]
        with safe_open(str(model / weights[key]), framework="pt", device="cpu") as f:
            tensor = f.get_tensor(key).contiguous()
        assert tensor.dtype in (torch.float32, torch.bfloat16), (key, tensor.dtype)
        receipt = {
            "key": key,
            "checkpoint_dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "sha256": hashlib.sha256(
                tensor.view(torch.uint8).numpy().tobytes()
            ).hexdigest(),
        }
        return tensor, receipt

    for boundary, norm in (
        ("attn", "input_layernorm"),
        ("ffn", "post_attention_layernorm"),
    ):
        values, receipts = [], {}
        for field in ("fn", "scale", "base"):
            tensor, receipt = get(
                [f"layers.0.hc_{boundary}_{field}", f"layers.0.{boundary}_hc.{field}"]
            )
            # Match the existing harness and GLM loader's FP32 mHC parameters.
            values.append(tensor.float().contiguous())
            receipts[field] = {**receipt, "runtime_dtype": "torch.float32"}
        norm_weight, receipts["out_norm"] = get([f"layers.0.{norm}.weight"])
        yield f"checkpoint-layer0-{boundary}", values + [norm_weight], receipts


def load_helper(directory, name):
    spec = importlib.util.spec_from_file_location(name, directory / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def event_benchmark(calls, warmup, repeats):
    for _ in range(warmup):
        for call in calls.values():
            out = call()
            torch.npu.synchronize()
            del out
    start, end = [torch.npu.Event(enable_timing=True) for _ in range(2)]
    # Initialize events outside measurements; outputs remain alive through sync.
    start.record()
    end.record()
    end.synchronize()
    samples = {name: {"event_ms": [], "host_ms": []} for name in calls}
    for repeat in range(repeats):
        order = list(calls) if repeat % 2 == 0 else list(reversed(calls))
        for name in order:
            torch.npu.synchronize()
            before = time.perf_counter_ns()
            start.record()
            out = calls[name]()
            end.record()
            end.synchronize()
            host_ms = (time.perf_counter_ns() - before) / 1e6
            event_ms = float(start.elapsed_time(end))
            if not math.isfinite(event_ms) or event_ms <= 0:
                raise RuntimeError(f"Invalid NPU event duration: {event_ms}")
            samples[name]["event_ms"].append(event_ms)
            samples[name]["host_ms"].append(host_ms)
            del out
    for stats in samples.values():
        stats["median_event_ms"] = statistics.median(stats["event_ms"])
        stats["median_host_ms"] = statistics.median(stats["host_ms"])
    return samples


def run(args, result):
    check_only_sinkhorn_changes(args.reference_file, args.module_dir / "mhc.py")
    # Load both real layer-0 boundaries on CPU before any device work. Missing
    # weights are fatal; there is no silent synthetic-only substitute.
    params = list(checkpoint_parameters(args.checkpoint))
    import torch_npu

    torch.npu.set_device(args.device)
    device = f"npu:{args.device}"
    torch.manual_seed(SEED)
    torch.npu.manual_seed(SEED)
    result.update(
        torch=str(torch.__version__),
        torch_npu=str(torch_npu.__version__),
        device_name=torch.npu.get_device_name(args.device),
        source_contract_pass=True,
    )
    baseline = load_reference(args.reference_file)
    candidate = load_reference(args.module_dir / "mhc.py")
    mix = load_helper(args.module_dir, "mhc_pre_mix_npu").hc_pre_mix_large_prefill
    sinkhorn = load_helper(
        args.module_dir, "mhc_sinkhorn_npu"
    ).hc_sinkhorn_large_prefill
    post = load_helper(args.module_dir, "mhc_prefill_npu").hc_post_large_prefill

    def nd(tensor):
        out = torch_npu.npu_format_cast(tensor.to(device).contiguous(), 2)
        assert torch_npu.get_npu_format(out) == 2
        return out

    passed = True
    with torch.inference_mode():
        for t in (0, 1, 31, 32, 33, *ROWS):
            for kind in ("random", "near_uniform", "spiky"):
                logits = torch.randn(t, 4, 4)
                if kind == "near_uniform":
                    logits *= 1e-4
                elif kind == "spiky":
                    logits *= 20.0
                # The actual input contract starts with native NPU softmax+eps.
                comb = nd(logits).softmax(-1) + EPS if t else nd(logits)
                before = comb.clone()
                expected = native_sinkhorn(comb) if t else comb.clone()
                actual = sinkhorn(comb, ITERS, EPS)
                spec64 = native_sinkhorn(before.cpu().double())
                row = {
                    "t": t,
                    "case": kind,
                    "candidate_vs_native": comparison(actual, expected),
                    "native_vs_fp64": comparison(expected, spec64),
                    "candidate_vs_fp64": comparison(actual, spec64),
                    "input_unchanged": compare_bits(before, comb),
                }
                row["pass"] = (
                    row["candidate_vs_native"]["allclose"]
                    and row["candidate_vs_fp64"]["allclose"]
                    and row["input_unchanged"]["bitwise_equal"]
                )
                passed &= row["pass"]
                result["direct"].append(row)
                write_result(args.output, result)

        for name, values, receipts in params:
            fn, scale, base, norm_weight = map(nd, values)
            assert (
                fn.shape == (24, 16384) and scale.shape == (3,) and base.shape == (24,)
            )
            assert norm_weight.shape == (4096,)
            for t in ROWS:
                residual = nd(
                    torch.randn(t, 4, 4096, dtype=torch.bfloat16, device=device)
                )
                inputs = (residual, fn, scale, base, EPS, EPS, EPS, 2.0, ITERS)
                captured = []

                def capture(comb, iters, eps):
                    captured.append(comb.clone())
                    return sinkhorn(comb, iters, eps)

                # The baseline already includes the validated pre-mix. Both
                # calls execute full-T native GEMM/RMS/sigmoid/softmax.
                old = baseline(*inputs, pre_mix=mix)
                new = candidate(*inputs, pre_mix=mix, sinkhorn=capture)
                assert len(captured) == 1
                spec64 = native_sinkhorn(captured[0].cpu().double())
                row = {
                    "parameters": name,
                    "receipt": receipts,
                    "t": t,
                    "post": compare_bits(old[0], new[0]),
                    "layer_input": compare_bits(old[2], new[2]),
                    "candidate_vs_native": comparison(new[1], old[1]),
                    "native_vs_fp64": comparison(old[1], spec64),
                    "candidate_vs_fp64": comparison(new[1], spec64),
                }
                old_norm = torch_npu.npu_rms_norm(old[2], norm_weight, EPS)[0]
                new_norm = torch_npu.npu_rms_norm(new[2], norm_weight, EPS)[0]
                row["separate_rmsnorm"] = compare_bits(old_norm, new_norm)
                # Same synthetic layer output and existing large-P post kernel
                # on both sides; report propagation through complete BF16 post.
                layer_out = nd(
                    torch.randn(t, 4096, device=device, dtype=torch.bfloat16)
                )
                old_post = post(layer_out, residual, old[0], old[1], 4)
                new_post = post(layer_out, residual, new[0], new[1], 4)
                row["bf16_mhc_post"] = compare_bits(old_post, new_post)
                # Post differences are a diagnostic, not silently accepted as
                # bitexact or converted into a looser numerical score.
                row["pass"] = (
                    all(
                        row[key]["bitwise_equal"] and row[key]["finite"]
                        for key in ("post", "layer_input", "separate_rmsnorm")
                    )
                    and row["candidate_vs_native"]["allclose"]
                    and row["candidate_vs_fp64"]["allclose"]
                    and row["bf16_mhc_post"]["finite"]
                )
                passed &= row["pass"]
                result["whole_pre"].append(row)
                write_result(args.output, result)
                del old, new, old_norm, new_norm, old_post, new_post, layer_out
                del inputs, residual, captured, spec64

        result["numeric_pass"] = passed
        result["bf16_post_review_required"] = any(
            row["bf16_mhc_post"]["bit_mismatches"] for row in result["whole_pre"]
        )
        if not passed:
            result["status"] = "numeric_failed_no_timing"
            write_result(args.output, result)
            raise SystemExit("Numerical gate failed; see JSON; timings not run")
        result["status"] = "numeric_pass_timing_running"
        write_result(args.output, result)
        for name, values, _ in params:
            fn, scale, base, _ = map(nd, values)
            for t in ROWS:
                residual = nd(
                    torch.randn(t, 4, 4096, dtype=torch.bfloat16, device=device)
                )
                inputs = (residual, fn, scale, base, EPS, EPS, EPS, 2.0, ITERS)
                calls = {
                    "baseline": lambda: baseline(*inputs, pre_mix=mix),
                    "candidate": lambda: candidate(
                        *inputs, pre_mix=mix, sinkhorn=sinkhorn
                    ),
                }
                result["full_pre_timing"].append(
                    {
                        "t": t,
                        "parameters": name,
                        **event_benchmark(calls, args.warmup, args.repeats),
                    }
                )
                write_result(args.output, result)
                del calls, inputs, residual
    result["status"] = "complete"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--module-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--reference-file",
        type=Path,
        required=True,
        help="mhc.py extracted from baseline 45add04",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/public-flash/models/GLM-5.3-Flash-w8a8"),
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--confirm-parent-stopped-p", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.module_dir is None:
        args.module_dir = Path(__file__).resolve().parents[2] / "python/sglang/kernels/ops/layernorm"
    if not args.confirm_parent_stopped_p:
        parser.error("Parent must stop P and reserve an idle NPU before running")
    if args.warmup < 2 or args.repeats < 5:
        parser.error("Require warmup >= 2 and repeats >= 5")
    if args.output.exists():
        parser.error("Output exists; retain previous evidence and choose a new path")
    torch.set_num_threads(4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    paths = [
        args.reference_file,
        Path(__file__),
        *(
            args.module_dir / name
            for name in (
                "mhc.py",
                "mhc_sinkhorn_npu.py",
                "mhc_pre_mix_npu.py",
                "mhc_prefill_npu.py",
            )
        ),
    ]
    result = {
        "status": "numeric_running",
        "seed": SEED,
        "rows": ROWS,
        "iters": ITERS,
        "epsilon": EPS,
        "profiler_enabled": False,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "checkpoint": str(args.checkpoint),
        "device": args.device,
        "sources": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "fp64_spec": "CPU FP64 normalization from the identical native FP32 softmax+eps input",
        "timing_scope": "full mhc-pre including native projection/RMS/sigmoid/softmax and existing pre-mix; excludes subsequent independent RMSNorm",
        "input_kind": "fixed-seed synthetic BF16 activations; real checkpoint layer0 attn/ffn parameters",
        "direct": [],
        "whole_pre": [],
        "full_pre_timing": [],
    }
    write_result(args.output, result)
    try:
        run(args, result)
    except Exception:
        result["status"] = "execution_failed"
        result["error"] = traceback.format_exc()
        write_result(args.output, result)
        raise
    write_result(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output),
                "bf16_post_review_required": result["bf16_post_review_required"],
            }
        )
    )


if __name__ == "__main__":
    main()
