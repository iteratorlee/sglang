#!/usr/bin/env python3
"""Standalone mHC pre mix gate/benchmark; run only on a parent-reserved idle NPU.

No SGLang or server import, network access, weight modification or deployment.
Copies of candidate mhc.py/mhc_pre_mix_npu.py and baseline_mhc.py sit in --kit.
Full-T native PyTorch remains the oracle; no one-row shadow reference.
"""

import argparse
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import time
from typing import Tuple

import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


def check_only_mix_changes(baseline, candidate):
    """Reject accidental changes to projection, RMS, Sinkhorn or output ordering."""
    old, new = [function_ast(p, "_mhc_pre_torch") for p in [baseline, candidate]]
    assert new.args.args[-1].arg == "pre_mix"
    assert isinstance(new.args.defaults[-1], ast.Constant)
    assert new.args.defaults[-1].value is None
    new.args.args.pop()
    new.args.defaults.pop()
    branches = [
        n
        for n in new.body
        if isinstance(n, ast.If) and ast.unparse(n.test) == "pre_mix is None"
    ]
    assert len(branches) == 1 and len(branches[0].body) == 1
    new.body[new.body.index(branches[0])] = branches[0].body[0]
    assert ast.dump(old, include_attributes=False) == ast.dump(
        new, include_attributes=False
    )


def compare_bits(a, b, row_chunk=128):
    assert a.shape == b.shape and a.dtype == b.dtype
    integer = torch.int16 if a.dtype == torch.bfloat16 else torch.int32
    mismatches, max_abs, first = 0, 0.0, None
    for start in range(0, a.shape[0], row_chunk):
        aa, bb = [
            x[start : start + row_chunk].detach().cpu().contiguous() for x in (a, b)
        ]
        different = aa.view(integer) != bb.view(integer)
        mismatches += int(different.sum().item())
        finite = torch.isfinite(aa) & torch.isfinite(bb)
        if finite.any():
            max_abs = max(
                max_abs, float((aa.float()[finite] - bb.float()[finite]).abs().max())
            )
        if first is None and different.any():
            index = tuple(int(i) for i in different.nonzero()[0])
            first = {
                "index": [index[0] + start, *index[1:]],
                "old_bits": int(aa.view(integer)[index]),
                "new_bits": int(bb.view(integer)[index]),
            }
    return {
        "elements": a.numel(),
        "bit_mismatches": mismatches,
        "max_finite_abs": max_abs,
        "first_mismatch": first,
        "pass": mismatches == 0,
    }


def percentiles(xs):
    ys = sorted(xs)

    def at(q):
        pos = (len(ys) - 1) * q
        low, high = math.floor(pos), math.ceil(pos)
        return ys[low] + (ys[high] - ys[low]) * (pos - low)

    return {
        "n": len(xs),
        "mean_ms": statistics.mean(xs),
        **{f"p{int(q*100)}_ms": at(q) for q in [0.5, 0.75, 0.9, 0.99]},
        "samples_ms": xs,
    }


def write_result(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def checkpoint_parameters(model, layers):
    from safetensors import safe_open

    index_path = next((model / name for name in ("quant_model_weights.safetensors.index.json", "model.safetensors.index.json") if (model / name).is_file()), None)
    assert index_path is not None, "missing checkpoint weight index"
    weights = json.loads(index_path.read_text())["weight_map"]

    def get(suffixes):
        matches = [key for key in weights if any(key.endswith(s) for s in suffixes)]
        assert len(matches) == 1, (suffixes, matches)
        key = matches[0]
        with safe_open(str(model / weights[key]), framework="pt", device="cpu") as f:
            return key, f.get_tensor(key).contiguous()

    for layer in layers:
        for boundary, norm in [
            ("attn", "input_layernorm"),
            ("ffn", "post_attention_layernorm"),
        ]:
            values, receipt = [], {}
            for field in ["fn", "scale", "base"]:
                key, tensor = get(
                    [
                        f"layers.{layer}.hc_{boundary}_{field}",
                        f"layers.{layer}.{boundary}_hc.{field}",
                    ]
                )
                assert tensor.dtype in (torch.float32, torch.bfloat16), (key, tensor.dtype)
                checkpoint_dtype = str(tensor.dtype)
                # Match explicitly FP32 nn.Parameters in the native GLM loader.
                tensor = tensor.float().contiguous()
                receipt[field] = {
                    "key": key,
                    "checkpoint_dtype": checkpoint_dtype,
                    "runtime_dtype": str(tensor.dtype),
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "sha256": hashlib.sha256(
                        tensor.view(torch.uint8).numpy().tobytes()
                    ).hexdigest(),
                }
                values.append(tensor)
            key, norm_weight = get([f"layers.{layer}.{norm}.weight"])
            assert norm_weight.dtype in (torch.float32, torch.bfloat16)
            receipt["out_norm"] = {
                "key": key,
                "dtype": str(norm_weight.dtype),
                "sha256": hashlib.sha256(
                    norm_weight.view(torch.uint8).numpy().tobytes()
                ).hexdigest(),
            }
            yield f"checkpoint-layer{layer}-{boundary}", values + [norm_weight], receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kit",
        "--module-dir",
        dest="kit",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing candidate mhc.py and mhc_pre_mix_npu.py",
    )
    parser.add_argument(
        "--reference-file",
        type=Path,
        help="Frozen unmodified baseline mhc.py; default: module-dir/baseline_mhc.py",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device", type=int, default=0, help="Visible logical NPU index"
    )
    parser.add_argument("--confirm-idle-npu", action="store_true")
    parser.add_argument("--rows", type=int, nargs="+", default=[8192, 16384])
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-layers", type=int, nargs="+", default=[0, 21, 41])
    parser.add_argument("--suite", choices=["quick", "full"], default="full")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()
    assert (
        args.confirm_idle_npu
    ), "Parent must allocate an idle physical NPU before running this kit"
    assert (
        not args.output.exists()
    ), "Keep prior numerical and timing evidence immutable"
    assert all(4096 < t <= 16384 for t in args.rows)
    assert args.warmup >= 2 and args.repeats >= 5
    args.output.parent.mkdir(parents=True, exist_ok=True)
    reference_file = args.reference_file or args.kit / "baseline_mhc.py"
    check_only_mix_changes(reference_file, args.kit / "mhc.py")
    import torch_npu

    torch.set_num_threads(4)
    torch.npu.set_device(args.device)
    device = f"npu:{args.device}"

    def nd(x):
        out = torch_npu.npu_format_cast(x.to(device).contiguous(), 2)
        assert torch_npu.get_npu_format(out) == 2
        return out

    torch.manual_seed(20261016)
    torch.npu.manual_seed_all(20261016)
    baseline = load_reference(reference_file)
    candidate_ref = load_reference(args.kit / "mhc.py")
    spec = importlib.util.spec_from_file_location(
        "mhc_pre_mix_npu", args.kit / "mhc_pre_mix_npu.py"
    )
    kernel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernel)
    mix = kernel.hc_pre_mix_large_prefill
    result = {
        "status": "numeric_running",
        "device": device,
        "device_name": torch.npu.get_device_name(),
        "torch": torch.__version__,
        "torch_npu": getattr(torch_npu, "__version__", "unknown"),
        "rows": args.rows,
        "suite": args.suite,
        "checkpoint_layers": args.checkpoint_layers if args.checkpoint else [],
        "seed": 20261016,
        "source_contract_pass": True,
        "sources": {
            p.name: digest(p)
            for p in [
                reference_file,
                args.kit / "mhc.py",
                args.kit / "mhc_pre_mix_npu.py",
                Path(__file__),
            ]
        },
        "direct_mix": [],
        "whole_pre": [],
        "timing": [],
        "input_kind": "synthetic activations; real dimensions",
        "metric": "alternating synchronized eager host latency, includes dispatch/allocation; profiler disabled",
    }
    write_result(args.output, result)

    def require_pass(row, field):
        result[field].append(row)
        write_result(args.output, result)
        tests = (
            [row["comparison"]]
            if "comparison" in row
            else list(row["outputs"].values())
        )
        if not all(test["pass"] for test in tests):
            result["status"] = "numeric_failed_no_timing"
            write_result(args.output, result)
            raise SystemExit(
                "Numerical gate failed; output preserved; no performance measurement"
            )

    direct_cases = (
        ["random", "cancellation", "signed_zero"]
        if args.suite == "quick"
        else [
            "random",
            "uniform",
            "near_identity",
            "cancellation",
            "tiny",
            "signed_zero",
            "nonfinite",
        ]
    )
    with torch.inference_mode():
        for t in args.rows:
            for case in direct_cases:
                residual = nd(
                    torch.randn(t, 4, 4096, device=device, dtype=torch.bfloat16)
                )
                pre = nd(torch.rand(t, 4, device=device, dtype=torch.float32) + 1e-6)
                if case in ["uniform", "cancellation"]:
                    pre.fill_(0.25)
                if case == "near_identity":
                    pre.fill_(1e-6)
                    pre[:, 0] = 1.0
                if case == "cancellation":
                    for i, value in enumerate([1.0, 2.0**-26, -1.0, 2.0**-26]):
                        residual[:, i, :] = value
                if case == "tiny":
                    residual.mul_(2.0**-24)
                if case == "signed_zero":
                    residual.zero_()
                    residual[:, 0, :] = -0.0
                    residual[:, 1, :] = -0.0
                if case == "nonfinite":
                    residual[0, 0, 0] = float("nan")
                    residual[0, 1, 1] = float("inf")
                    residual[0, 2, 2] = -float("inf")
                residual_before, pre_before = residual.clone(), pre.clone()
                reference = (
                    (pre.unsqueeze(-1) * residual.float()).sum(1).to(residual.dtype)
                )
                actual = mix(residual, pre)
                comparison = compare_bits(reference, actual)
                unchanged = torch.equal(
                    residual_before.view(torch.int16), residual.view(torch.int16)
                ) and torch.equal(pre_before.view(torch.int32), pre.view(torch.int32))
                comparison["inputs_unchanged"] = unchanged
                comparison["pass"] &= unchanged
                row = {"t": t, "case": case, "comparison": comparison}
                if comparison["first_mismatch"] is not None:
                    token, hidden = comparison["first_mismatch"]["index"]
                    rr, pp = residual[token, :, hidden].float().cpu(), pre[token].cpu()
                    terms = rr * pp

                    def bits(x):
                        return int(x.bfloat16().view(torch.int16))

                    row["first_counterexample"] = {
                        "residual": [
                            None if not math.isfinite(float(v)) else float(v)
                            for v in rr
                        ],
                        "pre": pp.tolist(),
                        "cpu_sequential_bf16_bits": bits(
                            ((terms[0] + terms[1]) + terms[2]) + terms[3]
                        ),
                        "cpu_pair02_bf16_bits": bits(
                            (terms[0] + terms[2]) + (terms[1] + terms[3])
                        ),
                    }
                require_pass(row, "direct_mix")
                del residual, pre, reference, actual, residual_before, pre_before

        synthetic = [
            torch.randn(24, 16384, dtype=torch.float32) / 128,
            torch.tensor([0.5, -0.25, 1.25]),
            torch.randn(24),
            (1 + torch.randn(4096) * 0.1).bfloat16(),
        ]
        params = [("synthetic", synthetic, {"checkpoint": False})]
        if args.checkpoint:
            params.extend(
                checkpoint_parameters(args.checkpoint, args.checkpoint_layers)
            )
        for name, values, receipt in params:
            fn, scale, base, norm_weight = map(nd, values)
            assert (
                fn.shape == (24, 16384) and scale.shape == (3,) and base.shape == (24,)
            )
            for t in args.rows:
                for amplitude in (
                    [0.0, 2.0**-24, 1.0, 128.0]
                    if name == "synthetic" and args.suite == "full"
                    else [1.0]
                ):
                    residual = nd(
                        (torch.randn(t, 4, 4096, device=device) * amplitude).bfloat16()
                    )
                    originals = [
                        v.clone() for v in (residual, fn, scale, base, norm_weight)
                    ]
                    inputs = (residual, fn, scale, base, 1e-6, 1e-6, 1e-6, 2.0, 20)
                    old, new = baseline(*inputs), candidate_ref(*inputs, pre_mix=mix)
                    outputs = {
                        key: compare_bits(a, b)
                        for key, a, b in zip(["post", "comb", "layer_input"], old, new)
                    }
                    old_norm = torch_npu.npu_rms_norm(old[2], norm_weight, 1e-6)[0]
                    new_norm = torch_npu.npu_rms_norm(new[2], norm_weight, 1e-6)[0]
                    outputs["out_norm"] = compare_bits(old_norm, new_norm)
                    outputs["inputs_unchanged"] = {
                        "pass": all(
                            torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                            for a, b in zip(
                                originals, (residual, fn, scale, base, norm_weight)
                            )
                        )
                    }
                    require_pass(
                        {
                            "parameters": name,
                            "receipt": receipt,
                            "t": t,
                            "amplitude": amplitude,
                            "outputs": outputs,
                        },
                        "whole_pre",
                    )
                    del old, new, old_norm, new_norm, inputs, residual, originals
        result["status"] = "numeric_pass_timing_running"
        write_result(args.output, result)
        # Timings use a representative real checkpoint parameter set if supplied.
        fn, scale, base, norm_weight = map(nd, params[-1][1])
        for t in args.rows:
            residual = nd(torch.randn(t, 4, 4096, device=device, dtype=torch.bfloat16))
            inputs = (residual, fn, scale, base, 1e-6, 1e-6, 1e-6, 2.0, 20)
            calls = {
                "baseline": lambda: baseline(*inputs),
                "candidate": lambda: candidate_ref(*inputs, pre_mix=mix),
            }
            for _ in range(args.warmup):
                for call in calls.values():
                    out = call()
                    torch.npu.synchronize()
                    del out
            timings = {name: [] for name in calls}
            for i in range(args.repeats):
                for name in (
                    ["baseline", "candidate"]
                    if i % 2 == 0
                    else ["candidate", "baseline"]
                ):
                    torch.npu.synchronize()
                    begin = time.perf_counter_ns()
                    out = calls[name]()
                    torch.npu.synchronize()
                    elapsed = (time.perf_counter_ns() - begin) / 1e6
                    timings[name].append(elapsed)
                    del out
            stats = {name: percentiles(xs) for name, xs in timings.items()}
            result["timing"].append(
                {
                    "t": t,
                    "parameters": params[-1][0],
                    **stats,
                    "p50_reduction_percent": 100
                    * (1 - stats["candidate"]["p50_ms"] / stats["baseline"]["p50_ms"]),
                }
            )
            write_result(args.output, result)
            del calls, inputs, residual
    result["status"] = "complete"
    write_result(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output),
                "timing": result["timing"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
