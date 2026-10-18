"""Explicit BF16 production acceptance or FP32 failure diagnosis; no service use.

The v1 strict harness remains unchanged. A pass here names only the requested
contract and does not supersede v1's FP32 failure or establish internal equality.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import traceback
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
if __name__ == "__main__":
    os.environ["TRITON_INTERPRET"] = "0"

import torch


def load_reference(path):
    spec = importlib.util.spec_from_file_location("rms_v1_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def difference(a, b):
    """Exact bytes plus diagnostic statistics; no tolerance acceptance."""
    a, b = a.detach().cpu().contiguous(), b.detach().cpu().contiguous()
    if a.shape != b.shape or a.dtype != b.dtype:
        raise AssertionError("shape/dtype mismatch")
    byte_diff = a.view(torch.uint8) != b.view(torch.uint8)
    element_diff = byte_diff.view(*a.shape, a.element_size()).any(dim=-1)
    row_counts = element_diff.reshape(a.shape[0], -1).sum(dim=-1)
    ids = torch.nonzero(byte_diff.flatten()).flatten()
    first = int(ids[0]) if ids.numel() else None
    result = dict(
        exact=not bool(ids.numel()),
        different_bytes=int(ids.numel()),
        different_elements=int(element_diff.sum()),
        first_byte=first,
        different_rows=int((row_counts != 0).sum()),
        row_mismatch_counts=row_counts.tolist(),
    )
    if first is not None:
        flat = first // a.element_size()
        result.update(
            first_row=flat // a.shape[-1],
            first_col=flat % a.shape[-1],
            first_values=[float(a.flatten()[flat]), float(b.flatten()[flat])],
        )
    if a.dtype == torch.float32:
        # Monotonic integer ordering, including signed zero, for finite values.
        def ordered(x):
            bits = x.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
            return torch.where(
                (bits & 0x80000000) != 0, 0xFFFFFFFF - bits, bits + 0x80000000
            )

        finite = torch.isfinite(a) & torch.isfinite(b)
        distance = (ordered(a) - ordered(b)).abs()
        result["max_finite_ulp"] = int(distance[finite].max()) if finite.any() else None
        result["max_finite_abs"] = (
            float((a[finite] - b[finite]).abs().max()) if finite.any() else None
        )
    return result


def input_factory(original, seed_offset=0, midpoint_direction=None):
    def make(rows, weight_dtype=torch.float32, pattern="random", seed=53):
        x, gate, weight = original(rows, weight_dtype, pattern, seed + seed_offset)
        if midpoint_direction is not None:
            if weight_dtype != torch.float32:
                raise ValueError("Midpoint fixtures require FP32 weights")
            # Input generation only: actual acceptance still compares original
            # versus candidate NPU outputs. No CPU formula serves as an oracle.
            row = min(23, rows - 1)
            value, g = x[row].double(), gate[row].double()
            scale = value / torch.sqrt((value * value).sum() / 128 + 1e-6)
            scale = scale / (1.0 + torch.exp(-g))
            valid = (scale != 0) & torch.isfinite(scale)
            # Midpoint between BF16 1.0 and its next representable value.
            target = torch.where(valid, 1.00390625 / scale, weight.double()).float()
            if midpoint_direction:
                toward = torch.full_like(target, float("inf") * midpoint_direction)
                target = torch.nextafter(target, toward)
            weight = torch.where(valid, target, weight)
        return x, gate, weight

    return make


def production(h, old, new, args, report):
    original_inputs, original_exact = h.inputs, h.exact
    artifact = args.output.with_suffix("").with_name(args.output.stem + "-artifacts")

    def checked(a, b, label):
        try:
            original_exact(a, b, label)
        except AssertionError:
            artifact.mkdir(parents=True, exist_ok=True)
            report["failure_detail"] = dict(label=label, **difference(a, b))
            torch.save(
                dict(baseline=a.detach().cpu(), candidate=b.detach().cpu()),
                artifact / "production-failure.pt",
            )
            report["failure_tensors"] = str(artifact / "production-failure.pt")
            raise

    for rows in args.row_list:
        for seed_offset in args.seed_offsets:
            for wdtype in (torch.bfloat16, torch.float32):
                report["fixture_scope"] = dict(
                    family="standard", seed_offset=seed_offset
                )
                start = len(report["cases"])
                with (
                    patch.object(
                        h, "inputs", input_factory(original_inputs, seed_offset)
                    ),
                    patch.object(h, "exact", checked),
                ):
                    h.case(old, new, rows, wdtype, torch.bfloat16, args, report)
                for entry in report["cases"][start:]:
                    entry.update(report["fixture_scope"])
        for direction in (-1, 0, 1):
            report["fixture_scope"] = dict(
                family="bf16-midpoint-target",
                weight_ulp_step=direction,
                anchor_row=min(23, rows - 1),
            )
            start = len(report["cases"])
            # Adversarial correctness fixtures are never used for timing.
            with (
                patch.object(
                    h,
                    "inputs",
                    input_factory(original_inputs, midpoint_direction=direction),
                ),
                patch.object(h, "exact", checked),
                patch.object(args, "timing_repeats", 0),
            ):
                h.case(old, new, rows, torch.float32, torch.bfloat16, args, report)
            for entry in report["cases"][start:]:
                entry.update(report["fixture_scope"])


def dump_compiled(handle, directory, tag):
    artifacts = {}
    for key, content in getattr(handle, "asm", {}).items():
        safe = "".join(c if c.isalnum() else "_" for c in str(key))
        if not isinstance(content, (str, bytes)):
            continue
        blob = content.encode() if isinstance(content, str) else content
        path = directory / f"{tag}.{safe}"
        path.write_bytes(blob)
        artifacts[str(key)] = dict(
            path=str(path), sha256=hashlib.sha256(blob).hexdigest()
        )
    return artifacts


def diagnostic(h, old, new, args, report):
    """Reproduce v1, inspect both R4/R8 and graph, retaining every mismatch."""
    directory = args.output.with_suffix("").with_name(args.output.stem + "-artifacts")
    directory.mkdir(parents=True, exist_ok=True)
    all_exact = True
    for rows in args.row_list:
        host = h.inputs(rows, torch.bfloat16, "random", 53)
        data = [t.to(f"npu:{args.device}") for t in host]
        saved = dict(zip(("x", "gate", "weight"), host))
        for odtype in (torch.bfloat16, torch.float32):
            outputs, compiled = [], {}
            for group in (1, 4, 8):
                out = torch.empty((rows, 128), dtype=odtype, device=data[0].device)
                if group == 1:
                    handle = old._glm_rms_norm_gated_row_kernel[(rows,)](
                        *data,
                        out,
                        1e-6,
                        feature_dim=128,
                        block_dim=128,
                        num_warps=1,
                        num_stages=1,
                    )
                else:
                    handle = new._glm_rms_norm_gated_rows_kernel[
                        ((rows + group - 1) // group,)
                    ](
                        *data,
                        out,
                        1e-6,
                        num_rows=rows,
                        rows_per_program=group,
                        num_warps=1,
                        num_stages=1,
                    )
                outputs.append(out)
                tag = f"rows{rows}-{str(odtype).split('.')[-1]}-R{group}"
                compiled[tag] = dump_compiled(handle, directory, tag)
            torch.npu.synchronize()
            eager = [out.cpu().clone() for out in outputs]
            graphs = [
                h.capture(old, new, group, data, out)
                for group, out in zip((1, 4, 8), outputs)
            ]
            for graph, out in zip(graphs, outputs):
                out.fill_(float("nan"))
                graph.replay()
            torch.npu.synchronize()
            replay = [out.cpu() for out in outputs]
            for index, group in enumerate((1, 4, 8)):
                for mode, a, b in (
                    ("eager", eager[0], eager[index]),
                    ("graph", replay[0], replay[index]),
                    ("graph-vs-eager", eager[index], replay[index]),
                ):
                    entry = dict(
                        rows=rows,
                        output_dtype=str(odtype),
                        group=group,
                        mode=mode,
                        **difference(a, b),
                    )
                    report["diagnostics"].append(entry)
                    all_exact &= entry["exact"]
                saved[f"{str(odtype)}-R{group}-eager"] = eager[index]
                saved[f"{str(odtype)}-R{group}-graph"] = replay[index]
            report["compiled_artifacts"].update(compiled)
        path = directory / f"rows{rows}-seed53.pt"
        torch.save(saved, path)
        report["tensor_artifacts"].append(str(path))
    if not all_exact:
        raise AssertionError(
            "Diagnostic byte differences recorded; FP32 contract remains failed"
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--contract", required=True, choices=("production-bf16", "fp32-diagnostic")
    )
    p.add_argument(
        "--reference-harness",
        type=Path,
        default=Path(__file__).with_name("test_glm53_gated_rms_rows_npu.py"),
    )
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--rows", default="1024,1663,1664,1665,1792,2048")
    p.add_argument("--seed-offsets", default="0,997,2027")
    p.add_argument("--timing-repeats", type=int, default=0)
    p.add_argument("--cann-build", default="not supplied")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.row_list = [int(x) for x in args.rows.split(",")]
    args.seed_offsets = [int(x) for x in args.seed_offsets.split(",")]
    if any(n < 1024 or n > 2048 for n in args.row_list) or args.timing_repeats < 0:
        p.error("rows must be within the D guard [1024,2048], timing nonnegative")
    if args.contract == "fp32-diagnostic" and args.timing_repeats:
        p.error("diagnostic contract does not time kernels")
    h = load_reference(args.reference_harness)
    import torch_npu

    torch.npu.set_device(args.device)
    torch.set_num_threads(1)
    old, new = h.load_kernels()
    report = dict(
        passed=False,
        contract=args.contract,
        cases=[],
        timings=[],
        diagnostics=[],
        compiled_artifacts={},
        tensor_artifacts=[],
        torch=torch.__version__,
        torch_npu=torch_npu.__version__,
        triton=h.triton.__version__,
        cann_build=args.cann_build,
        device=torch.npu.get_device_name(args.device),
        prior_v1_fp32_status="FAILED: 1024 BF16 weight, FP32 output, eager R4, seed53 random",
        source_sha256={
            name: hashlib.sha256((h.SOURCE / name).read_bytes()).hexdigest()
            for name in ("rms_norm_gated_npu.py", "rms_norm_gated_rows_npu.py")
        },
    )
    try:
        with torch.no_grad():
            if args.contract == "production-bf16":
                production(h, old, new, args, report)
            else:
                diagnostic(h, old, new, args, report)
        report["passed"] = True
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
