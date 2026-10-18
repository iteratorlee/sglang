"""Strict D/P original-vs-1D-loop unit, including FP32 outputs.

No device execution occurs on import. The original strict harness's numerical
checks are reused; only kernel dispatch, role/shape and launch metadata change.
"""

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import statistics
from types import SimpleNamespace as NS
import traceback
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
if __name__ == "__main__":
    os.environ["TRITON_INTERPRET"] = "0"

LOOP_FLAG = "SGLANG_GLM53_D_GATED_RMS_LOOP_ROWS"
P_LOOP_FLAG = "SGLANG_GLM53_P_GATED_RMS_LOOP_ROWS"


class LoopKernel:
    def __init__(self, original):
        self.original = original

    def __getitem__(self, grid):
        def launch(
            x,
            gate,
            weight,
            output,
            eps,
            *,
            num_rows,
            rows_per_program,
            num_warps,
            num_stages,
        ):
            if rows_per_program not in (4, 8):
                raise ValueError("Only R4/R8 loop variants are tested")
            expected = (num_rows + rows_per_program - 1) // rows_per_program
            if grid != (expected,):
                raise AssertionError("Unexpected loop grid")
            return self.original[grid](
                x,
                gate,
                weight,
                output,
                eps,
                feature_dim=128,
                block_dim=128,
                num_rows=num_rows,
                grid_rows=expected,
                num_warps=num_warps,
                num_stages=num_stages,
            )

        return launch


def baseline_grid(rows):
    return min(rows, 32768) if rows > 65535 else rows


def adapt_reference(h, role="decode"):
    original_load = h.load_kernels
    original_launch = h.launch

    def load(*args, **kwargs):
        old, new = original_load(*args, **kwargs)
        new._glm_rms_norm_gated_rows_kernel = LoopKernel(
            old._glm_rms_norm_gated_large_row_kernel
        )
        old.supported_prefill_loop = new.supported_prefill_loop
        old.get_server_args = lambda: NS(disaggregation_mode=role)
        return old, new

    def launch(old, new, variant, x, gate, weight, output, eps=1e-6):
        rows = x.numel() // 128
        if variant == 1 and rows > 65535:
            old._glm_rms_norm_gated_large_row_kernel[(baseline_grid(rows),)](
                x,
                gate,
                weight,
                output,
                eps,
                feature_dim=128,
                block_dim=128,
                num_rows=rows,
                grid_rows=baseline_grid(rows),
                num_warps=1,
                num_stages=1,
            )
            return output
        return original_launch(old, new, variant, x, gate, weight, output, eps)

    def capture_wrapper(old, variant, data):
        rows = data[0].numel() // 128
        if role == "prefill":
            data = [
                data[0].view(1, 16384, 8, 128),
                data[1].view(16384, 8, 128),
                data[2],
            ]
        env = {
            LOOP_FLAG: "0",
            P_LOOP_FLAG: "0",
            "SGLANG_GLM53_D_GATED_RMS_ROWS": "0",
            "SGLANG_GLM53_P_GATED_RMS_ROWS": "0",
        }
        env[P_LOOP_FLAG if role == "prefill" else LOOP_FLAG] = (
            str(variant) if variant != 1 else "0"
        )
        observed = []
        original = old._glm_rms_norm_gated_large_row_kernel

        class Trace:
            def __getitem__(self, grid):
                call = original[grid]

                def run(*args, **kwargs):
                    observed.append(grid)
                    return call(*args, **kwargs)

                return run

        with (
            patch.dict(os.environ, env),
            patch.object(old, "_glm_rms_norm_gated_large_row_kernel", Trace()),
        ):
            for _ in range(3):
                old.glm_rms_norm_gated_npu(*data, eps=1e-6)
            h.torch.npu.synchronize()
            graph = h.torch.npu.NPUGraph()
            with h.torch.npu.graph(graph):
                output = old.glm_rms_norm_gated_npu(*data, eps=1e-6)
            h.torch.npu.synchronize()
        grid = baseline_grid(rows) if variant == 1 else (rows + variant - 1) // variant
        expected = [(grid,)] * 4 if variant != 1 or rows > 65535 else []
        if observed != expected:
            raise AssertionError(
                f"wrapper loop dispatch mismatch: {observed} != {expected}"
            )
        return graph, output

    h.load_kernels = load
    h.launch = launch
    h.capture_wrapper = capture_wrapper
    h.FLAG = P_LOOP_FLAG if role == "prefill" else LOOP_FLAG
    h.candidate_kind = "existing-large-kernel-1d-grid-stride"
    return h


def benchmark(h, old, new, rows, wdtype, odtype, args, report):
    data = [x.to(f"npu:{args.device}") for x in h.inputs(rows, wdtype)]
    outputs = [
        h.torch.empty((rows, 128), dtype=odtype, device=data[0].device)
        for _ in range(3)
    ]
    graphs = [
        h.capture(old, new, group, data, out) for group, out in zip((1, 4, 8), outputs)
    ]
    samples = [[], [], []]
    for repetition in range(args.timing_repeats):
        for i in [(repetition + offset) % 3 for offset in range(3)]:
            begin, end = h.torch.npu.Event(enable_timing=True), h.torch.npu.Event(
                enable_timing=True
            )
            begin.record()
            for _ in range(20):
                graphs[i].replay()
            end.record()
            end.synchronize()
            samples[i].append(begin.elapsed_time(end) * 1000 / 20)
    report["timings"].append(
        dict(
            rows=rows,
            weight_dtype=str(wdtype),
            output_dtype=str(odtype),
            variants=[1, 4, 8],
            median_graph_replay_us=[statistics.median(s) for s in samples],
            samples_us=samples,
        )
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--role", choices=("decode", "prefill"), required=True)
    p.add_argument(
        "--reference-harness",
        type=Path,
        default=Path(__file__).with_name("test_glm53_gated_rms_rows_npu.py"),
    )
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--rows", help="D defaults 1024,1663,1664,1665,1792,2048; P 131072")
    p.add_argument("--timing-repeats", type=int, default=0)
    p.add_argument("--cann-build", default="not supplied")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rows = [
        int(x)
        for x in (
            args.rows
            or ("131072" if args.role == "prefill" else "1024,1663,1664,1665,1792,2048")
        ).split(",")
    ]
    valid = (
        rows == [131072]
        if args.role == "prefill"
        else all(1024 <= n <= 2048 for n in rows)
    )
    if not valid or args.timing_repeats < 0:
        p.error(
            "P only supports 131072; D rows must be [1024,2048]; timing nonnegative"
        )
    spec = importlib.util.spec_from_file_location(
        "loop_reference", args.reference_harness
    )
    h = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h)
    adapt_reference(h, args.role)
    import torch_npu

    h.torch.npu.set_device(args.device)
    h.torch.set_num_threads(1)
    old, new = h.load_kernels()
    report = dict(
        passed=False,
        contract="strict-bf16-and-fp32-output",
        candidate_kind=h.candidate_kind,
        role=args.role,
        cases=[],
        timings=[],
        torch=h.torch.__version__,
        torch_npu=torch_npu.__version__,
        triton=h.triton.__version__,
        cann_build=args.cann_build,
        device=h.torch.npu.get_device_name(args.device),
        source_sha256={
            name: hashlib.sha256((h.SOURCE / name).read_bytes()).hexdigest()
            for name in ("rms_norm_gated_npu.py", "rms_norm_gated_rows_npu.py")
        },
        prior_2d_status="v1 FP32 failed; v2 BF16 with FP32 weight failed; not promoted",
    )
    try:
        with h.torch.no_grad():
            for n in rows:
                for wdtype in (h.torch.bfloat16, h.torch.float32):
                    for odtype in (h.torch.bfloat16, h.torch.float32):
                        start = len(report["cases"])
                        # Reuse every original comparison. Correct its small-only
                        # log grid metadata after the unchanged case returns.
                        try:
                            with (
                                patch.object(args, "timing_repeats", 0),
                                contextlib.redirect_stdout(io.StringIO()),
                            ):
                                h.case(old, new, n, wdtype, odtype, args, report)
                        finally:
                            for case in report["cases"][start:]:
                                case.update(
                                    role=args.role,
                                    candidate_kind=h.candidate_kind,
                                    baseline_kernel="large" if n > 65535 else "small",
                                )
                                case["grids"][0] = baseline_grid(n)
                                print(json.dumps(case), flush=True)
                        if args.timing_repeats:
                            report["active_case"].update(
                                phase="timing", pattern="random"
                            )
                            benchmark(h, old, new, n, wdtype, odtype, args, report)
        report["passed"] = True
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
