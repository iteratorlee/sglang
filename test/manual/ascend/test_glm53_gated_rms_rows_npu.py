"""Standalone exact/graph/timing harness; run only on an isolated target NPU.

No model, service or KDA state is needed. Original and grouped kernels use the
working-tree sources. CPU tests import the loader without initializing devices.
Float32 outputs are a stricter diagnostic in addition to production BF16 stores.
"""

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
from types import ModuleType, SimpleNamespace as NS
import traceback
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
if __name__ == "__main__":
    # Select the compiled language implementation before importing Triton.
    os.environ["TRITON_INTERPRET"] = "0"

import torch
import triton

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "python/sglang/srt/hardware_backend/npu/attention/glm53"
FLAG = "SGLANG_GLM53_D_GATED_RMS_ROWS"


def load_kernels(cpu=False):
    """Remove only framework imports/guard; keep both original kernels intact."""
    path = SOURCE / "rms_norm_gated_npu.py"
    tree = ast.parse(path.read_text())

    class Isolate(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            if node.module in (
                "sgl_kernel_npu.fla.utils",
                "sglang.srt.runtime_context",
                "rms_norm_gated_rows_npu",
            ):
                return ast.Pass()
            return node

        def visit_FunctionDef(self, node):
            node.decorator_list = [
                d
                for d in node.decorator_list
                if not (isinstance(d, ast.Name) and d.id == "input_guard")
            ]
            return self.generic_visit(node)

    module = ModuleType("glm_rms_rows_original")
    module.__file__ = str(path)
    new_path = SOURCE / "rms_norm_gated_rows_npu.py"
    spec = importlib.util.spec_from_file_location("glm_rms_rows_grouped", new_path)
    grouped = importlib.util.module_from_spec(spec)
    with patch.dict(os.environ, {"TRITON_INTERPRET": "1" if cpu else "0"}):
        spec.loader.exec_module(grouped)
        exec(
            compile(
                ast.fix_missing_locations(Isolate().visit(tree)), str(path), "exec"
            ),
            module.__dict__,
        )
    module.get_server_args = lambda: NS(disaggregation_mode="decode")
    module.supported = grouped.supported
    module.rms_norm_gated_rows = grouped.rms_norm_gated_rows
    return module, grouped


def inputs(rows, weight_dtype=torch.float32, pattern="random", seed=53):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, 128, generator=gen).to(torch.bfloat16)
    g = torch.randn(rows, 128, generator=gen).to(torch.bfloat16)
    w = torch.randn(128, generator=gen).to(weight_dtype)
    if pattern == "zero":
        x.zero_()
    elif pattern == "signed-zero":
        x.zero_()
        x[:, ::2] = -0.0
        w[::2] = -1
    elif pattern == "wide":
        x[:, ::4] = 0
        x[:, 1::4] *= 128
        x[:, 2::4] *= 2**-12
        g[:, ::4] = 20
        g[:, 1::4] = -20
    elif pattern != "random":
        raise ValueError(pattern)
    return x, g, w


def launch(old, new, variant, x, gate, weight, output, eps=1e-6):
    rows = x.numel() // 128
    if not rows:
        return output
    if variant == 1:
        old._glm_rms_norm_gated_row_kernel[(rows,)](
            x,
            gate,
            weight,
            output,
            eps,
            feature_dim=128,
            block_dim=128,
            num_warps=1,
            num_stages=1,
        )
    else:
        new._glm_rms_norm_gated_rows_kernel[(triton.cdiv(rows, variant),)](
            x,
            gate,
            weight,
            output,
            eps,
            num_rows=rows,
            rows_per_program=variant,
            num_warps=1,
            num_stages=1,
        )
    return output


def exact(a, b, label):
    a, b = a.detach().cpu().contiguous(), b.detach().cpu().contiguous()
    if a.shape != b.shape or a.dtype != b.dtype:
        raise AssertionError(f"shape/dtype mismatch: {label}")
    aa, bb = a.view(torch.uint8).flatten(), b.view(torch.uint8).flatten()
    if not torch.equal(aa, bb):
        mismatch = torch.nonzero(aa != bb).flatten()
        raise AssertionError(
            f"bit mismatch: {label}; bytes={len(mismatch)}, first={int(mismatch[0])}"
        )


def capture(old, new, variant, data, output):
    for _ in range(3):
        launch(old, new, variant, *data, output)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        launch(old, new, variant, *data, output)
    torch.npu.synchronize()
    return graph


def capture_wrapper(old, variant, data):
    # The standalone loader supplies decode runtime context and removes only
    # input_guard. Dispatch and tensor guards are the production wrapper's code.
    with patch.dict(os.environ, {FLAG: str(variant) if variant != 1 else "0"}):
        for _ in range(3):
            old.glm_rms_norm_gated_npu(*data, eps=1e-6)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            output = old.glm_rms_norm_gated_npu(*data, eps=1e-6)
        torch.npu.synchronize()
    return graph, output


def case(old, new, rows, wdtype, odtype, args, report):
    report["active_case"] = dict(
        rows=rows, weight_dtype=str(wdtype), output_dtype=str(odtype), phase="capture"
    )
    data = [x.to(f"npu:{args.device}") for x in inputs(rows, wdtype)]
    backing = [
        torch.full((rows + 8, 128), -19.0, dtype=odtype, device=data[0].device)
        for _ in range(3)
    ]
    outputs = [out[:rows] for out in backing]
    graphs = [
        capture(old, new, variant, data, out)
        for variant, out in zip((1, 4, 8), outputs)
    ]
    wrapped = []
    if odtype == torch.bfloat16:
        # Exercise model-shaped tensors as well as flattened kernel inputs.
        shape = (1, rows // 32, 32, 128) if rows % 32 == 0 else (rows, 128)
        wrapper_data = [data[0].view(shape), data[1].view(shape), data[2]]
        wrapped = [capture_wrapper(old, variant, wrapper_data) for variant in (1, 4, 8)]
    pointers = [x.data_ptr() for x in (*data, *outputs)]
    for seed in (53, 101):
        for pattern in ("random", "zero", "signed-zero", "wide"):
            report["active_case"].update(seed=seed, pattern=pattern, phase="exact")
            host = inputs(rows, wdtype, pattern, seed)
            for target, source in zip(data, host):
                target.copy_(source)
            eager_backing = [torch.full_like(out, -19.0) for out in backing]
            eager = [out[:rows] for out in eager_backing]
            for variant, out in zip((1, 4, 8), eager):
                launch(old, new, variant, *data, out)
            for out in outputs:
                out.fill_(float("nan"))  # Detect incomplete writes on replay.
            for graph in graphs:
                graph.replay()
            torch.npu.synchronize()
            for i, variant in enumerate((1, 4, 8)):
                exact(eager[0], eager[i], f"eager R={variant}")
                exact(outputs[0], outputs[i], f"graph R={variant}")
                exact(eager[i], outputs[i], f"graph/eager R={variant}")
                for buffer in (backing[i], eager_backing[i]):
                    exact(
                        buffer[rows:],
                        torch.full((8, 128), -19.0, dtype=odtype),
                        f"tail store mask R={variant}",
                    )
            for variant, (graph, out) in zip((1, 4, 8), wrapped):
                out.fill_(float("nan"))
                graph.replay()
                exact(eager[0], out.view(rows, 128), f"wrapper graph flag R={variant}")
            for original, actual in zip(host, data):
                exact(original, actual, "input mutated")
            if pointers != [x.data_ptr() for x in (*data, *outputs)]:
                raise AssertionError("graph buffer address changed")
            record = dict(
                rows=rows,
                weight_dtype=str(wdtype),
                output_dtype=str(odtype),
                seed=seed,
                pattern=pattern,
                exact=True,
                grids=[rows, triton.cdiv(rows, 4), triton.cdiv(rows, 8)],
            )
            report["cases"].append(record)
            print(json.dumps(record), flush=True)
    if args.timing_repeats and rows in (1664, 1792) and odtype == torch.bfloat16:
        report["active_case"]["phase"] = "timing"
        # Timings use random finite data, not the final wide/zero fixture.
        for target, source in zip(data, inputs(rows, wdtype)):
            target.copy_(source)
        samples = [[], [], []]
        for repetition in range(args.timing_repeats):
            # Rotate order to reduce systematic warm-cache / clock bias.
            for i in [(repetition + offset) % 3 for offset in range(3)]:
                begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(
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
                variants=[1, 4, 8],
                median_graph_replay_us=[statistics.median(x) for x in samples],
                samples_us=samples,
            )
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--rows", default="1,7,8,9,32,64,1024,1663,1664,1665,1792,2048")
    p.add_argument("--timing-repeats", type=int, default=0)
    p.add_argument("--cann-build", default="not supplied")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rows = [int(x) for x in args.rows.split(",")]
    if not rows or min(rows) < 1 or max(rows) > 2048 or args.timing_repeats < 0:
        p.error("rows must be in [1,2048], timing repeats nonnegative")
    import torch_npu

    torch.npu.set_device(args.device)
    torch.set_num_threads(1)
    old, new = load_kernels()
    report = dict(
        passed=False,
        cases=[],
        timings=[],
        torch=torch.__version__,
        cann_build=args.cann_build,
        torch_npu=torch_npu.__version__,
        triton=triton.__version__,
        device=torch.npu.get_device_name(args.device),
        source_sha256={
            name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
            for name in ("rms_norm_gated_npu.py", "rms_norm_gated_rows_npu.py")
        },
    )
    try:
        with torch.no_grad():
            for n in rows:
                for wdtype in (torch.bfloat16, torch.float32):
                    for odtype in (torch.bfloat16, torch.float32):
                        case(old, new, n, wdtype, odtype, args, report)
        report["passed"] = True
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
