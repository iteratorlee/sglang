"""Opt-in single-NPU parity/timing harness; never invoked by CPU tests.

Run only on a device reserved by the parent experiment coordinator:
  python test/manual/ascend/test_glm53_device_ordered_indices_npu.py \
    --device npu:0 --suite all --output /tmp/device-ordered-parity.json

The inverse suite executes real device sort/count/map kernels. The dispatch
suite executes production dispatch with real INT8 quantization and simulated
equal EP peers, and compares stable INT8 payload, FP32 scales/gates, expert
counts and inverse map. Neither suite establishes HCCL/full-model performance.
Compilation and warmup are excluded; wall timings include host overhead and
synchronize only at measurement boundaries. No profiler or polling is used.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import sys
import time
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[3]
MOE = ROOT / "python/sglang/srt/hardware_backend/npu/moe"


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def make_ids(rows, case, dtype):
    generator = torch.Generator(device="cpu").manual_seed(101616 + rows)
    if case == "uniform":
        return torch.randint(0, 288, (rows, 8), generator=generator, dtype=dtype)
    ids = torch.randint(-4, 293, (rows, 8), generator=generator, dtype=dtype)
    if case == "duplicate":
        ids[:] = torch.tensor([9, 1, 9, 0, 17, 1, 0, 9], dtype=dtype)
    elif case == "skew":
        ids[:] = 270
    elif case == "empty":
        ids[:] = -1
    else:
        ids[:, :3] = 0
        if rows:
            ids[0] = torch.tensor([-7, -1, 288, 999, 18, 18, 0, 17], dtype=dtype)
            if dtype == torch.int64:
                # The map must not narrow IDs to INT32 and alias invalid IDs.
                ids[0, 0] = 2**40
    return ids


def select_routes(ids, rank, active):
    """Independent reference selection, usable on CPU or NPU."""
    flat, low = ids.flatten(), rank * 18
    selected = ((flat >= low) & (flat < low + 18)).nonzero().flatten()
    half_routes = ids.numel() // 2
    if not active[0]:
        selected = selected[selected >= half_routes]
    if not active[1]:
        selected = selected[selected < half_routes]
    selected = selected[torch.argsort(flat[selected], stable=True)]
    counts = torch.bincount(flat[selected].to(torch.int64) - low, minlength=18)
    return selected, counts


def adversarial_values(rows, width):
    """Cancellation and gate mantissa cases that expose FP32 reassociation."""
    y = ((torch.arange(len(rows) * width).reshape(-1, width) % 31 - 15) / 8).to(
        torch.bfloat16
    )
    gates = 0.125 + (torch.arange(len(rows), dtype=torch.float32) + 1) * 2**-20
    for row in rows.unique():
        locations = (rows == row).nonzero().flatten()
        if locations.numel() < 5:
            continue
        y[locations, :3] = 0
        y[locations[:3], 0] = torch.tensor([2**24, 1, -(2**24)], dtype=torch.bfloat16)
        y[locations[:3], 2] = torch.tensor([256, 1, -256], dtype=torch.bfloat16)
        gates[locations[:3]] = 1
        y[locations[3], 1], gates[locations[3]] = -1.0078125, 1
        y[locations[4], 1] = 1.0078125
        gates[locations[4]] = torch.nextafter(torch.tensor(1.0), torch.tensor(2.0))
    return y, gates


def alternating_timings(baseline, candidate, warmup, repeat):
    functions = {"cpu_reference": baseline, "device_ordered": candidate}
    for _ in range(warmup):
        for function in functions.values():
            function()
    torch.npu.synchronize()
    timings = {name: [] for name in functions}
    for iteration in range(repeat):
        order = tuple(functions) if iteration % 2 == 0 else tuple(reversed(functions))
        for name in order:
            torch.npu.synchronize()
            start = time.perf_counter()
            value = functions[name]()
            torch.npu.synchronize()
            timings[name].append((time.perf_counter() - start) * 1000)
            del value
    return {
        name: {"samples_ms": values, "p50_ms": statistics.median(values)}
        for name, values in timings.items()
    }


def inverse_suite(args, collectives, inverse, combine, reference):
    records = []
    for rows in args.rows:
        for case in ("mixed", "duplicate", "skew", "empty"):
            for dtype in (torch.int32, torch.int64):
                ids = make_ids(rows, case, dtype)
                device_ids = ids.to(args.device)
                for active in (
                    (True, True),
                    (True, False),
                    (False, True),
                    (False, False),
                ):
                    for rank in args.ranks:
                        selected, counts = select_routes(ids, rank, active)
                        device_selected, device_counts = select_routes(
                            device_ids, rank, active
                        )
                        torch.testing.assert_close(
                            device_selected.cpu(), selected, rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            device_counts.cpu(), counts, rtol=0, atol=0
                        )
                        expected = collectives._normal_combine_ordered_indices(
                            selected // 8, rows, 8
                        )
                        actual = inverse.device_ordered_indices(
                            device_ids, device_selected, rank * 18, 18
                        )
                        torch.testing.assert_close(
                            actual.cpu(), expected, rtol=0, atol=0
                        )
                        record = dict(
                            rows=rows,
                            case=case,
                            ids_dtype=str(dtype),
                            rank=rank,
                            active_dp=active,
                            received=len(selected),
                            indices_exact=True,
                            selection_exact=True,
                            counts_exact=True,
                        )
                        if rows <= 256:
                            y, gates = adversarial_values(selected // 8, 4096)
                            expected_output = reference.sequential_reference(
                                y, gates, selected // 8, rows
                            )
                            # This calls the real existing FP32 combine on the new map.
                            actual_output = combine.ordered_combine(
                                y.to(args.device),
                                gates.to(args.device),
                                actual,
                                rows,
                                4096,
                            )
                            torch.testing.assert_close(
                                actual_output.cpu(), expected_output, rtol=0, atol=0
                            )
                            record["fp32_combine_exact"] = True
                        records.append(record)
                del device_ids
    # One representative real geometry for timing. Include CPU ID readback,
    # sorting, map creation and map H2D in the current reference, as in dispatch.
    rows = max(args.rows)
    ids = make_ids(rows, "uniform", torch.int64).to(args.device)
    active, rank = (True, False), 0

    def baseline():
        selected, counts = select_routes(ids.cpu(), rank, active)
        result = collectives._normal_combine_ordered_indices(selected // 8, rows, 8)
        return selected.to(args.device), counts.to(args.device), result.to(args.device)

    def candidate():
        selected, counts = select_routes(ids, rank, active)
        return selected, counts, inverse.device_ordered_indices(ids, selected, 0, 18)

    return dict(
        cases=records,
        route_and_map_timing_rows=rows,
        timing_route_distribution="uniform random expert IDs; first DP active",
        route_and_map_timing=alternating_timings(
            baseline, candidate, args.warmup, args.repeat
        ),
    )


def dispatch_suite(args, collectives):
    """Production entry point, mocking only EP transport and runtime topology."""
    group = object()

    class Peers:
        rank = 0
        ReduceOp = torch.distributed.ReduceOp

        def get_world_size(self, group):
            return 16

        def get_rank(self, group):
            return self.rank

        def all_gather_into_tensor(self, out, value, group):
            out.copy_(value.repeat((16,) + (1,) * (value.ndim - 1)))

        def all_reduce(self, value, op=None, group=None):
            pass

    peers = Peers()
    context = ModuleType("sglang.srt.runtime_context")
    context.get_parallel = lambda: NS(
        moe_ep_group=NS(
            world_size=16,
            device_group=group,
            ranks=list(range(16)),
            rank_in_group=peers.rank,
        ),
        tp_group=NS(ranks=list(range(16))),
        attn_dp_size=2,
        attn_tp_size=8,
        attn_dp_rank=peers.rank // 8,
        attn_tp_rank=peers.rank % 8,
    )
    records = []
    with (
        mock.patch.object(collectives, "dist", peers),
        mock.patch.dict(sys.modules, {context.__name__: context}),
        mock.patch.dict(
            os.environ,
            {
                "SGLANG_GLM53_NORMAL_SKIP_IDLE_DP": "1",
                "SGLANG_GLM53_NORMAL_SHAPE_CACHE": "0",
                "SGLANG_GLM53_NORMAL_MEMORY_HEADROOM_MB": "0",
                "SGLANG_GLM53_NORMAL_COMBINE_ORDERED": "2node",
            },
        ),
    ):
        for tokens in args.dispatch_tokens:
            generator = torch.Generator(device="cpu").manual_seed(18 + tokens)
            x = torch.randn(tokens, 4096, generator=generator).to(
                dtype=torch.bfloat16, device=args.device
            )
            ids = make_ids(tokens, "uniform", torch.int64).to(args.device)
            gates = torch.rand(tokens, 8, generator=generator).to(args.device)
            for rank in args.ranks:
                peers.rank = rank
                for active in (
                    (True, False),
                    (False, True),
                    (True, True),
                    (False, False),
                ):
                    counts = tuple(8 * tokens * int(value) for value in active)
                    scope = collectives._idle_dp_counts.set(counts)

                    def run(mode):
                        os.environ["SGLANG_GLM53_NORMAL_DEVICE_ROUTING"] = mode
                        return collectives.collective_dispatch(group, x, ids, gates)

                    try:
                        baseline, candidate = run("0"), run("ordered")
                        for left, right in zip(baseline[:2], candidate[:2]):
                            torch.testing.assert_close(
                                left.cpu(), right.cpu(), rtol=0, atol=0
                            )
                        assert baseline[2] == candidate[2].cpu().tolist()
                        for index in (0, 1, 6):
                            torch.testing.assert_close(
                                baseline[3][index].cpu(),
                                candidate[3][index].cpu(),
                                rtol=0,
                                atol=0,
                            )
                        assert baseline[3][2:6] == candidate[3][2:6]
                        assert candidate[0].dtype == torch.int8
                        assert (
                            candidate[1].dtype == candidate[3][1].dtype == torch.float32
                        )
                        record = dict(
                            tokens=tokens,
                            rank=rank,
                            active_dp=active,
                            received=candidate[3][5],
                            payload_scales_gates_counts_rows_map_exact=True,
                        )
                        del baseline, candidate
                        # Limit timings to the current one-active-DP setup.
                        if active == (True, False) and rank in (0, 15):
                            record["simulated_peer_dispatch_timing"] = (
                                alternating_timings(
                                    lambda: run("0"),
                                    lambda: run("ordered"),
                                    args.warmup,
                                    args.repeat,
                                )
                            )
                        records.append(record)
                    finally:
                        collectives._idle_dp_counts.reset(scope)
    return dict(cases=records, timing_route_distribution="uniform random expert IDs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--suite", choices=("inverse", "dispatch", "all"), default="all"
    )
    parser.add_argument("--rows", type=int, nargs="+", default=[0, 16, 256, 32768])
    parser.add_argument("--ranks", type=int, nargs="+", default=[0, 7, 8, 15])
    parser.add_argument("--dispatch-tokens", type=int, nargs="+", default=[16, 2048])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; keep parity/timing artifacts immutable")
    if (
        not args.device.startswith("npu:")
        or args.warmup < 1
        or args.repeat < 1
        or any(rows < 0 or rows > 131070 or rows % 16 for rows in args.rows)
        or any(rank not in range(16) for rank in args.ranks)
        or any(tokens < 0 or tokens > 8191 for tokens in args.dispatch_tokens)
    ):
        parser.error("require NPU, positive repeats, bounded EP16 rows/ranks/tokens")
    import torch_npu  # no accelerator initialization on CPU import

    torch.set_num_threads(1)
    torch.npu.set_device(args.device)
    package = ModuleType("glm_device_harness")
    package.__path__ = [str(MOE)]
    sys.modules[package.__name__] = package
    collectives = load_source(
        package.__name__ + ".glm53_collectives", MOE / "glm53_collectives.py"
    )
    inverse = load_source(
        package.__name__ + ".glm53_device_ordered_indices",
        MOE / "glm53_device_ordered_indices.py",
    )
    combine = load_source(
        package.__name__ + ".glm53_ordered_combine", MOE / "glm53_ordered_combine.py"
    )
    reference = load_source(
        "glm_device_reference",
        Path(__file__).with_name("test_glm53_ordered_combine_npu.py"),
    )
    result = dict(
        scope=__doc__,
        torch_version=torch.__version__,
        torch_npu_version=torch_npu.__version__,
        source_sha256={
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                MOE / "glm53_collectives.py",
                MOE / "glm53_device_ordered_indices.py",
                MOE / "glm53_ordered_combine.py",
                Path(__file__),
                Path(__file__).with_name("test_glm53_ordered_combine_npu.py"),
            )
        },
        config={
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        passed=False,
    )
    if args.suite in ("inverse", "all"):
        result["inverse"] = inverse_suite(
            args, collectives, inverse, combine, reference
        )
    if args.suite in ("dispatch", "all"):
        result["dispatch"] = dispatch_suite(args, collectives)
    result["passed"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": True,
                "output": str(args.output),
                "inverse_cases": len(result.get("inverse", {}).get("cases", [])),
                "dispatch_cases": len(result.get("dispatch", {}).get("cases", [])),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
