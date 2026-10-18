"""Isolated exact BHV1/2/4 KDA verify unit; no device use on import.

Package this file with kda_recurrent_npu.py and set GLM53_KDA_SOURCE_DIR.
CPU tests use only tiny interpreter inputs and metadata dispatch checks.
"""

import argparse
import ast
import hashlib
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
    os.environ["TRITON_INTERPRET"] = "0"

import torch
import triton

FLAG = "SGLANG_GLM53_D_KDA_BHV"
SOURCE = (
    Path(os.environ["GLM53_KDA_SOURCE_DIR"])
    if os.environ.get("GLM53_KDA_SOURCE_DIR")
    else Path(__file__).resolve().parents[3]
    / "python/sglang/srt/hardware_backend/npu/attention/glm53"
)
PATTERNS = (
    "dense",
    "mixed-slots",
    "all-pad",
    "mixed-empty",
    "all-empty",
    "zero",
    "wide",
    "dense-again",
)


def load_module():
    path = SOURCE / "kda_recurrent_npu.py"

    class Isolate(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            if node.module == "sglang.srt.runtime_context":
                return ast.Pass()
            return node

    module = ModuleType("isolated_glm_kda_bhv")
    module.__file__ = str(path)
    tree = Isolate().visit(ast.parse(path.read_text()))
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), module.__dict__)
    module.get_server_args = lambda: NS(disaggregation_mode="decode")
    return module


def exact(left, right, label):
    left, right = (t.detach().cpu().contiguous() for t in (left, right))
    if left.shape != right.shape or left.dtype != right.dtype:
        raise AssertionError(f"{label}: shape/dtype mismatch")
    a, b = (t.view(torch.uint8).flatten() for t in (left, right))
    if not torch.equal(a, b):
        diff = torch.nonzero(a != b).flatten()
        raise AssertionError(f"{label}: bytes={diff.numel()}, first={int(diff[0])}")


def inputs(batch, pattern="dense", seed=53, heads=32, dim=128, index_dtype=torch.int32):
    gen = torch.Generator().manual_seed(seed)
    tokens = batch * 4
    data = {
        name: (torch.randn(1, tokens, heads, dim, generator=gen) * 0.2).bfloat16()
        for name in ("q", "k", "v", "a")
    }
    data["b"] = torch.randn(1, tokens, heads, generator=gen).bfloat16()
    data["A_log"] = torch.randn(1, 1, heads, 1, generator=gen) * 0.1
    data["dt_bias"] = torch.randn(heads * dim, generator=gen) * 0.1
    # Physical native [V,K] backing, including unused slots and slot zero.
    data["state_backing"] = torch.randn(batch + 3, heads, dim, dim, generator=gen) * 0.1
    data["mid_backing"] = torch.full((batch + 2, 4, heads, dim, dim), -19.0 - seed % 3)
    slots = [1 + (i + seed) % (batch + 1) for i in range(batch)]
    lengths = [4] * batch
    if pattern == "mixed-slots":
        slots = [0 if i % 3 == 0 else -1 if i % 3 == 1 else 1 for i in range(batch)]
    elif pattern == "all-pad":
        slots = [0 if i % 2 else -1 for i in range(batch)]
    elif pattern == "mixed-empty":
        lengths = [0 if i % 2 else 4 for i in range(batch)]
    elif pattern == "all-empty":
        lengths = [0] * batch
    elif pattern == "zero":
        for name in ("q", "k"):
            data[name].zero_()
            data[name][..., ::2] = -0.0
    elif pattern == "wide":
        for name in ("q", "k", "v"):
            data[name][..., ::4] = 0
            data[name][..., 1::4] *= 128
            data[name][..., 2::4] *= 2**-12
        data["a"][..., ::2] = 20
        data["a"][..., 1::2] = -20
        data["b"][..., ::2] = 20
        data["b"][..., 1::2] = -20
    elif pattern not in ("dense", "dense-again"):
        raise ValueError(pattern)
    starts = [0]
    for length in lengths:
        starts.append(starts[-1] + length)
    data["indices"] = torch.tensor(slots, dtype=index_dtype)
    data["starts"] = torch.tensor(starts, dtype=index_dtype)
    return data


class Buffers:
    def __init__(self, host, device):
        self.data = {name: t.to(device).clone() for name, t in host.items()}
        self.state = self.data["state_backing"].transpose(-1, -2)
        self.mid = self.data["mid_backing"][:-1].transpose(-1, -2)
        q = self.data["q"]
        self.output_backing = torch.full(
            (1, q.shape[1] + 2, q.shape[2], q.shape[3]),
            -31.0,
            dtype=q.dtype,
            device=device,
        )
        self.output = self.output_backing[:, : q.shape[1]]

    def reset(self, host):
        for name, tensor in self.data.items():
            tensor.copy_(host[name])
        self.output_backing.fill_(-31)
        self.output.zero_()

    def pointers(self):
        return [t.data_ptr() for t in (*self.data.values(), self.output_backing)]

    def kwargs(self):
        return dict(
            **{
                name: self.data[name]
                for name in ("q", "k", "v", "a", "b", "A_log", "dt_bias")
            },
            initial_state_source=self.state,
            initial_state_indices=self.data["indices"],
            cu_seqlens=self.data["starts"],
            intermediate_state=self.mid,
            lower_bound=-5.0,
            prefill=False,
        )


def direct(module, buf, bhv):
    d = buf.data
    _, tokens, heads, dim = d["q"].shape
    batch = d["indices"].numel()
    bk, bv = triton.next_power_of_2(dim), min(triton.next_power_of_2(dim), 64)
    return module._glm_kda_varlen_recurrent_kernel[
        (triton.cdiv(dim, bv), batch, heads // bhv)
    ](
        **{name: d[name] for name in ("q", "k", "v", "a", "b", "A_log", "dt_bias")},
        o=buf.output,
        h0_source=buf.state,
        h0_indices=d["indices"],
        cu_seqlens=d["starts"],
        scale=dim**-0.5,
        lower_bound=-5.0,
        intermediate=buf.mid,
        track_indices=d["indices"],
        track_lens=d["starts"],
        S0=buf.state.stride(0),
        S1=buf.state.stride(1),
        SK=buf.state.stride(2),
        SV=buf.state.stride(3),
        M0=buf.mid.stride(0),
        M1=buf.mid.stride(1),
        M2=buf.mid.stride(2),
        MK=buf.mid.stride(3),
        MV=buf.mid.stride(4),
        STEPS=4,
        SAVE_INTERMEDIATE=True,
        TRACK_STATE=False,
        H=heads,
        HV=heads,
        K=dim,
        V=dim,
        BK=bk,
        BV=bv,
        BHV=bhv,
        num_warps=1,
        num_stages=3,
        multibuffer=False,
    )


def wrapped(module, buf, bhv):
    with patch.dict(os.environ, {FLAG: str(bhv)}):
        return module.glm_kda_varlen_recurrent_npu(**buf.kwargs())


def capture(call):
    for _ in range(3):
        call()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = call()
    torch.npu.synchronize()
    return graph, output


def host_result(buf, output):
    return dict(
        output=output.detach().cpu(),
        state=buf.data["state_backing"].cpu(),
        snapshots=buf.data["mid_backing"].cpu(),
    )


def validate(buf, output, host, reference, label):
    result = host_result(buf, output)
    for name in ("output", "state", "snapshots"):
        exact(result[name], reference[name], f"{label}/{name}")
    # Verify never commits into the persistent state; every slot must survive.
    exact(result["state"], host["state_backing"], f"{label}/persistent unchanged")
    for name, tensor in buf.data.items():
        if name not in ("state_backing", "mid_backing"):
            exact(tensor, host[name], f"{label}/input {name}")
    exact(
        buf.output_backing[:, -2:],
        torch.full_like(buf.output_backing[:, -2:], -31),
        f"{label}/output canary",
    )
    batch = host["indices"].numel()
    for row in range(batch):
        length = int(host["starts"][row + 1] - host["starts"][row])
        first_unwritten = length if int(host["indices"][row]) > 0 else 0
        exact(
            result["snapshots"][row, first_unwritten:],
            host["mid_backing"][row, first_unwritten:],
            f"{label}/unwritten snapshots row {row}",
        )
    exact(
        result["snapshots"][batch:],
        host["mid_backing"][batch:],
        f"{label}/scratch canaries",
    )
    used = int(host["starts"][-1])
    exact(
        result["output"][:, used:],
        torch.zeros_like(result["output"][:, used:]),
        f"{label}/unused packed output",
    )
    return result


class Trace:
    def __init__(self, kernel, observed):
        self.kernel, self.observed = kernel, observed

    def __getitem__(self, grid):
        call = self.kernel[grid]

        def launch(*args, **kwargs):
            self.observed.append((grid, kwargs["BHV"], kwargs["BK"], kwargs["BV"]))
            return call(*args, **kwargs)

        return launch


def check(module, batch, dtype, device, report, save):
    host = inputs(batch, index_dtype=dtype)
    eager = [Buffers(host, device) for _ in (1, 2, 4)]
    graph_buffers = [Buffers(host, device) for _ in (1, 2, 4)]
    wrapper_buffers = [Buffers(host, device) for _ in (1, 2, 4)]
    report["active_case"] = dict(batch=batch, index_dtype=str(dtype), phase="capture")
    graphs = [
        capture(lambda b=b, v=v: direct(module, b, v))[0]
        for b, v in zip(graph_buffers, (1, 2, 4))
    ]
    wrappers = []
    for buf, bhv in zip(wrapper_buffers, (1, 2, 4)):
        seen = []
        with patch.object(
            module,
            "_glm_kda_varlen_recurrent_kernel",
            Trace(module._glm_kda_varlen_recurrent_kernel, seen),
        ):
            pair = capture(lambda: wrapped(module, buf, bhv))
        effective = bhv if 12 <= batch <= 14 else 1
        expected = ((2, batch, 32 // effective), effective, 128, 64)
        if seen != [expected] * 4:
            raise AssertionError(f"wrapper dispatch {seen} != {[expected] * 4}")
        wrappers.append(pair)
    pointers = [b.pointers() for b in graph_buffers + wrapper_buffers]
    output_pointers = [out.data_ptr() for _, out in wrappers]
    for i, pattern in enumerate(PATTERNS):
        seed = (53, 101)[i % 2]
        host = inputs(batch, pattern, seed, index_dtype=dtype)
        report["active_case"].update(pattern=pattern, seed=seed, phase="exact")
        for b in eager + graph_buffers + wrapper_buffers:
            b.reset(host)
        for buf, bhv in zip(eager, (1, 2, 4)):
            direct(module, buf, bhv)
        reference = host_result(eager[0], eager[0].output)
        # Eager wrapper also exercises the production zeros_like output path.
        for buf, bhv in zip(wrapper_buffers, (1, 2, 4)):
            out = wrapped(module, buf, bhv)
            validate(buf, out, host, reference, f"wrapper eager BHV{bhv}")
            buf.reset(host)
        for graph in graphs:
            graph.replay()
        for graph, out in wrappers:
            out.fill_(float("nan"))
            graph.replay()
        torch.npu.synchronize()
        for bhv, eb, gb, wb, (_, out) in zip(
            (1, 2, 4), eager, graph_buffers, wrapper_buffers, wrappers
        ):
            validate(eb, eb.output, host, reference, f"eager BHV{bhv}")
            validate(gb, gb.output, host, reference, f"graph BHV{bhv}")
            validate(wb, out, host, reference, f"wrapper graph BHV{bhv}")
        if pointers != [b.pointers() for b in graph_buffers + wrapper_buffers]:
            raise AssertionError("captured input/state/scratch addresses changed")
        if output_pointers != [out.data_ptr() for _, out in wrappers]:
            raise AssertionError("captured wrapper output addresses changed")
        record = dict(
            batch=batch,
            index_dtype=str(dtype),
            pattern=pattern,
            seed=seed,
            exact=True,
            direct_bhv=[1, 2, 4],
            wrapper_bhv=[1, 2, 4] if batch >= 12 else [1, 1, 1],
            state_stride=list(eager[0].state.stride()),
            snapshot_stride=list(eager[0].mid.stride()),
        )
        report["cases"].append(record)
        print(json.dumps(record), flush=True)
        save()


def benchmark(module, batch, device, repeats):
    host = inputs(batch)
    bufs = [Buffers(host, device) for _ in (1, 2, 4)]
    graphs = [
        capture(lambda b=b, v=v: direct(module, b, v))[0]
        for b, v in zip(bufs, (1, 2, 4))
    ]
    samples = [[], [], []]
    for repeat in range(repeats):
        for i in [(repeat + j) % 3 for j in range(3)]:
            begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(
                enable_timing=True
            )
            begin.record()
            for _ in range(20):
                graphs[i].replay()
            end.record()
            end.synchronize()
            samples[i].append(begin.elapsed_time(end) * 1000 / 20)
    return dict(
        batch=batch,
        bhv=[1, 2, 4],
        median_graph_replay_us=[statistics.median(s) for s in samples],
        samples_us=samples,
        excludes="wrapper zero allocation, input/state reset and correctness copies",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batches", default="1,12,13,14")
    parser.add_argument("--timing-repeats", type=int, default=0)
    parser.add_argument("--cann-build", default="not supplied")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    batches = [int(x) for x in args.batches.split(",")]
    if any(b not in (1, 12, 13, 14) for b in batches) or args.timing_repeats < 0:
        parser.error("batches must be 1/12/13/14; timing repeats nonnegative")
    import torch_npu

    torch.npu.set_device(args.device)
    torch.set_num_threads(1)
    device = f"npu:{args.device}"
    module = load_module()
    report = dict(
        passed=False,
        cases=[],
        timings=[],
        lower_bound=-5.0,
        scale=128**-0.5,
        output_dtype="torch.bfloat16",
        state_snapshot_dtype="torch.float32",
        source_sha256=hashlib.sha256(
            (SOURCE / "kda_recurrent_npu.py").read_bytes()
        ).hexdigest(),
        torch=torch.__version__,
        torch_npu=torch_npu.__version__,
        triton=triton.__version__,
        cann_build=args.cann_build,
        device=torch.npu.get_device_name(args.device),
    )

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    try:
        with torch.no_grad():
            for batch in batches:
                for dtype in (torch.int32, torch.int64):
                    check(module, batch, dtype, device, report, save)
            # Separate timing phase: every requested exact case must pass first.
            if args.timing_repeats:
                for batch in batches:
                    report["active_case"] = dict(batch=batch, phase="timing")
                    report["timings"].append(
                        benchmark(module, batch, device, args.timing_repeats)
                    )
                    save()
        report["passed"] = True
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
