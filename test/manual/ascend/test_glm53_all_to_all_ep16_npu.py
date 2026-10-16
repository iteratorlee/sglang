"""Real EP16 HCCL comparison of legacy GLM dispatch/combine and A2A.

Launch externally on two hosts, eight processes per host, for example:
  torchrun --nnodes=2 --nproc-per-node=8 --node-rank=$NODE_RANK \
    --master-addr=$MASTER_ADDR --master-port=29616 \
    test_glm53_all_to_all_ep16_npu.py --module-dir /candidate/npu/moe

This script never spawns processes or imports the installed sglang package.
--module-dir must contain glm53_collectives.py, glm53_all_to_all.py and
glm53_ordered_combine.py. ALL collectives use the real world16 HCCL group.
Only runtime_context.get_parallel is supplied by the test (real group, CPU
topology attributes). The validated idle-count ContextVar is set directly.

Dispatch requires exact INT8 rows, FP32 scales/gates, counts and stable maps.
Combine reports FP32 and BF16 differences, not bit-identical equivalence:
legacy sums by EP owner then HCCL; A2A sums global expert order at source.
Recording hooks forward actual kernel/HCCL calls unchanged and retain only
the final FP32 tensor before casting. No fake collective/quantization/kernel.

Default geometries: rows32/1024 x three active-DP patterns x mixed/fullskew,
plus both-idle at each size, and a final zero-row case. Each geometry runs
with empty expert output y1 and y0 (30 cases). Every rank runs both variants,
so real collective order is identical even when only some ranks receive0.
Fullskew uses expert0,
so an idle rank0 still receives active-domain work and other ranks receive0.
At rows1024/fullskew/both-active, BF16 expert output and reverse permutation
each take 1 GiB on rank0. This is a correctness test, not a timing benchmark.
"""

import argparse
from contextlib import contextmanager
from datetime import timedelta
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import traceback

import torch
import torch.distributed as dist

WORLD, WIDTH, EXPERTS, TOPK = 16, 4096, 288, 8


def case_list(sizes):
    cases = []
    for rows in sizes:
        for active in ((rows * 8, 0), (0, rows * 8), (rows * 8, rows * 8)):
            for pattern in ("mixed", "fullskew"):
                cases.append((rows, active, pattern))
        cases.append((rows, (0, 0), "mixed"))
    cases.append((0, (0, 0), "mixed"))
    return cases


def routes_and_gates(rows, source, pattern):
    row = torch.arange(rows, dtype=torch.int64)[:, None]
    slot = torch.arange(TOPK, dtype=torch.int64)[None, :]
    ids = (row * 17 + slot * 43 + source * 19) % EXPERTS
    if rows:
        ids[::29, 1] = ids[::29, 0]  # stable ties / duplicate experts
        ids[0, 3:6] = torch.tensor([-1, -7, EXPERTS])
    if pattern == "fullskew":
        ids.fill_(0)
    gates = 0.125 + source * 2**-20 + (row * TOPK + slot).float() * 2**-24
    gates[(ids < 0) | (ids >= EXPERTS)] = float("nan")
    return ids, gates


def cpu_oracle(rows, active, pattern, rank):
    """Independent route tuples; no candidate routing/planning helpers used."""
    per_source, gate_tables = [], []
    for source in range(WORLD):
        ids, gates = routes_and_gates(rows, source, pattern)
        gate_tables.append(gates)
        entries = []
        if active[source // 8]:
            for row, route in enumerate(ids.tolist()):
                for slot, expert in enumerate(route):
                    if 0 <= expert < EXPERTS:
                        entries.append((expert, source, row, slot))
        entries.sort(key=lambda entry: entry[0])
        per_source.append(entries)
    raw = [
        entry for entries in per_source for entry in entries if entry[0] // 18 == rank
    ]
    expert_order = sorted(raw, key=lambda entry: entry[0])
    positions = {entry: index for index, entry in enumerate(expert_order)}
    inverse = torch.tensor([positions[entry] for entry in raw], dtype=torch.int64)
    counts = [0] * 18
    for expert, _, _, _ in expert_order:
        counts[expert % 18] += 1
    global_rows = torch.tensor(
        [source * rows + row for _, source, row, _ in expert_order], dtype=torch.int64
    )
    recv_gates = torch.tensor(
        [float(gate_tables[src][row, slot]) for _, src, row, slot in expert_order],
        dtype=torch.float32,
    )
    local = per_source[rank]
    indices = torch.full((rows, TOPK), -1, dtype=torch.int32)
    next_slot = [0] * rows
    for index, (_, _, row, _) in enumerate(local):
        indices[row, next_slot[row]] = index
        next_slot[row] += 1
    send_splits = [
        sum(entry[0] // 18 == owner for entry in local) for owner in range(WORLD)
    ]
    recv_splits = [sum(entry[1] == source for entry in raw) for source in range(WORLD)]
    local_gates = torch.tensor(
        [float(gate_tables[rank][row, slot]) for _, _, row, slot in local],
        dtype=torch.float32,
    )
    return NS(
        counts=counts,
        rows=global_rows,
        recv_gates=recv_gates,
        inverse=inverse,
        indices=indices,
        local_gates=local_gates,
        send_splits=send_splits,
        recv_splits=recv_splits,
        global_routes=sum(map(len, per_source)),
    )


def load_candidates(directory):
    name = "_glm53_ep16_candidate"
    package = ModuleType(name)
    package.__path__ = [str(directory)]
    sys.modules[name] = package
    loaded, hashes = {}, {}
    for leaf in ("glm53_collectives", "glm53_ordered_combine", "glm53_all_to_all"):
        path = directory / (leaf + ".py")
        fullname = name + "." + leaf
        spec = importlib.util.spec_from_file_location(fullname, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[fullname] = module
        spec.loader.exec_module(module)
        loaded[leaf] = module
        hashes[leaf] = hashlib.sha256(path.read_bytes()).hexdigest()
    return NS(
        legacy=loaded["glm53_collectives"],
        a2a=loaded["glm53_all_to_all"],
        kernel=loaded["glm53_ordered_combine"],
        hashes=hashes,
    )


def install_topology(group, rank):
    # Do not import installed sglang (or any server/model configuration).
    for name in ("sglang", "sglang.srt"):
        if name not in sys.modules:
            package = ModuleType(name)
            package.__path__ = []
            sys.modules[name] = package
    parallel = NS(
        moe_ep_group=NS(
            device_group=group,
            world_size=WORLD,
            ranks=list(range(WORLD)),
            rank_in_group=rank,
        ),
        tp_group=NS(ranks=list(range(WORLD))),
        attn_dp_size=2,
        attn_tp_size=8,
        attn_dp_rank=rank // 8,
        attn_tp_rank=rank % 8,
    )
    runtime = ModuleType("sglang.srt.runtime_context")
    runtime.get_parallel = lambda: parallel
    sys.modules[runtime.__name__] = runtime


def configure_flags(args):
    # Explicit test settings; do not inherit the running service's experiments.
    settings = dict(
        NORMAL_HCCL="1",
        NORMAL_SKIP_IDLE_DP="1",
        NORMAL_ALL_TO_ALL="1",
        NORMAL_DEVICE_ROUTING="0",
        NORMAL_SHAPE_CACHE="0",
        NORMAL_COMBINE_ORDERED="2node",
        NORMAL_COMBINE_DETERMINISTIC="1",
        NORMAL_REDUCE_SCATTER=(
            "2node" if args.legacy_reduce == "reduce_scatter" else "0"
        ),
        NORMAL_REDUCE_SCATTER_CHECK="0",
        NORMAL_MEMORY_HEADROOM_MB="0",
    )
    for key, value in settings.items():
        os.environ["SGLANG_GLM53_" + key] = value
    return settings


@contextmanager
def capture_final_fp32(kernel, legacy):
    """Record actual outputs, forwarding every kernel/collective unchanged."""
    captured = NS(tensor=None, origin=None)
    real_kernel = kernel.ordered_combine
    real_reduce, real_scatter = dist.all_reduce, dist.reduce_scatter_tensor

    def ordered(*args, **kwargs):
        output = real_kernel(*args, **kwargs)
        if not legacy:
            captured.tensor, captured.origin = output, "source_ordered_kernel"
        return output

    def reduce(tensor, *args, **kwargs):
        result = real_reduce(tensor, *args, **kwargs)
        if legacy and tensor.dtype == torch.float32 and tensor.ndim == 2:
            captured.tensor, captured.origin = tensor, "all_reduce"
        return result

    def scatter(output, source, *args, **kwargs):
        result = real_scatter(output, source, *args, **kwargs)
        if legacy:
            captured.tensor, captured.origin = output, "reduce_scatter"
        return result

    kernel.ordered_combine = ordered
    dist.all_reduce, dist.reduce_scatter_tensor = reduce, scatter
    try:
        yield captured
    finally:
        kernel.ordered_combine = real_kernel
        dist.all_reduce, dist.reduce_scatter_tensor = real_reduce, real_scatter


def exact_tensor(actual, expected, name, chunk_rows):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(
            f"{name}: {actual.shape}/{actual.dtype} != {expected.shape}/{expected.dtype}"
        )
    for start in range(0, actual.shape[0], chunk_rows):
        left, right = (
            actual[start : start + chunk_rows].cpu(),
            expected[start : start + chunk_rows].cpu(),
        )
        if not torch.equal(left, right):
            mismatches = int((left != right).sum())
            raise AssertionError(
                f"{name}: {mismatches} mismatches in rows starting {start}"
            )


def dispatch_checks(old, new, expected, chunk_rows):
    recv, scale, counts, state = old
    q, s, nc, ns = new
    exact_tensor(recv, q, "INT8 recv", chunk_rows)
    exact_tensor(scale, s, "FP32 scale", chunk_rows)
    if recv.dtype != torch.int8 or scale.dtype != torch.float32:
        raise AssertionError("dispatch dtype contract changed")
    if counts != nc or counts != expected.counts:
        raise AssertionError(
            f"expert counts: legacy={counts}, a2a={nc}, oracle={expected.counts}"
        )
    if state[5] != ns.received or ns.received != sum(expected.counts):
        raise AssertionError("received count / dummy allocation contract differs")
    exact_tensor(state[0], expected.rows, "legacy source rows", chunk_rows)
    exact_tensor(state[1], expected.recv_gates, "legacy FP32 gates", chunk_rows)
    exact_tensor(
        ns.source_order, expected.inverse, "A2A reverse source permutation", chunk_rows
    )
    exact_tensor(
        ns.indices, expected.indices, "A2A source stable route map", chunk_rows
    )
    exact_tensor(ns.weights, expected.local_gates, "A2A FP32 source gates", chunk_rows)
    if (
        ns.send_splits != expected.send_splits
        or ns.recv_splits != expected.recv_splits
        or ns.global_routes != expected.global_routes
    ):
        raise AssertionError("A2A split/global-route oracle mismatch")


def simulate_experts(recv, scale, received, rank, chunk_rows, empty_rows=1):
    y = recv.new_empty((received or empty_rows, recv.shape[1]), dtype=torch.bfloat16)
    if not received and empty_rows:
        y.fill_(float("nan"))  # Neither implementation may consume the dummy.
    for start in range(0, received, chunk_rows):
        end = min(start + chunk_rows, received)
        value = (recv[start:end].float() * scale[start:end, None]).to(torch.bfloat16)
        value.add_(rank / 16.0)
        y[start:end].copy_(value)
    return y


def error_metrics(actual, reference):
    actual, reference = actual.float(), reference.float()
    finite = torch.isfinite(actual) & torch.isfinite(reference)
    count = actual.numel()
    delta = (actual[finite] - reference[finite]).double()
    absolute = delta.abs()
    denominator = reference[finite].double().abs().clamp_min(1e-6)
    return dict(
        elements=count,
        nonfinite=int((~finite).sum()),
        different=int((actual != reference).sum()),
        max_abs=float(absolute.max()) if absolute.numel() else 0.0,
        max_rel=float((absolute / denominator).max()) if absolute.numel() else 0.0,
        mean_abs=float(absolute.mean()) if absolute.numel() else 0.0,
        rmse=float(delta.square().mean().sqrt()) if delta.numel() else 0.0,
    )


def emit(record):
    print(
        "GLM53_A2A_EP16 " + json.dumps(record, sort_keys=True, allow_nan=False),
        flush=True,
    )


def checked_together(check, *, group, device, rank, label):
    message = None
    try:
        check()
    except AssertionError as exc:
        message = str(exc)
        emit(dict(event="check_failure", rank=rank, label=label, error=message))
    failed = torch.tensor([int(message is not None)], dtype=torch.int32, device=device)
    dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=group)
    if failed.item():
        raise AssertionError(
            f"{label} failed on at least one EP rank; see check_failure records"
        )


def summarize_metrics(metrics, *, group, device):
    maxima = torch.tensor(
        [metrics[k] for k in ("max_abs", "max_rel")], dtype=torch.float32, device=device
    )
    counters = torch.tensor(
        [metrics[k] for k in ("elements", "nonfinite", "different")],
        dtype=torch.int32,
        device=device,
    )
    sums = torch.tensor(
        [
            metrics["mean_abs"] * metrics["elements"],
            metrics["rmse"] ** 2 * metrics["elements"],
        ],
        dtype=torch.float32,
        device=device,
    )
    dist.all_reduce(maxima, op=dist.ReduceOp.MAX, group=group)
    dist.all_reduce(counters, op=dist.ReduceOp.SUM, group=group)
    dist.all_reduce(sums, op=dist.ReduceOp.SUM, group=group)
    maximum, counts, total = (
        maxima.cpu().tolist(),
        counters.cpu().tolist(),
        sums.cpu().tolist(),
    )
    return dict(
        elements=counts[0],
        nonfinite=counts[1],
        different=counts[2],
        max_abs=maximum[0],
        max_rel=maximum[1],
        mean_abs=total[0] / max(1, counts[0]),
        rmse=(max(0.0, total[1]) / max(1, counts[0])) ** 0.5,
    )


def run_case(
    modules, rows, active, pattern, rank, group, device, args, empty_expert_rows=1
):
    label = f"rows={rows}/active={active}/pattern={pattern}/empty_y={empty_expert_rows}"
    phase = "inputs"
    token = modules.legacy._idle_dp_counts.set(tuple(active))
    try:
        if rank == 0:
            emit(dict(event="case_begin", case=label))
        column = (torch.arange(WIDTH).float() % 127 - 63)[None, :] / 64
        offset = (torch.arange(rows).float() % 23)[:, None] / 32 + rank / 128
        x = (column + offset).to(torch.bfloat16).to(device)
        ids, gates = routes_and_gates(rows, rank, pattern)
        ids, gates = ids.to(device), gates.to(device)
        expected = cpu_oracle(rows, active, pattern, rank)

        phase = "legacy_dispatch"
        old = modules.legacy.collective_dispatch(group, x, ids, gates)
        phase = "a2a_dispatch"
        new = modules.a2a.all_to_all_dispatch(group, x, ids, gates)
        phase = "dispatch_exact_checks"
        checked_together(
            lambda: dispatch_checks(old, new, expected, args.chunk_rows),
            group=group,
            device=device,
            rank=rank,
            label=label + "/dispatch",
        )
        old_state, new_state = old[3], new[3]
        phase = "simulate_experts"
        # Exact dispatch equality permits one shared BF16 expert-output tensor.
        y = simulate_experts(
            old[0], old[1], old_state[5], rank, args.chunk_rows, empty_expert_rows
        )
        del old, new, x, ids, gates, expected

        phase = "legacy_combine"
        with capture_final_fp32(modules.kernel, legacy=True) as captured:
            old_bf16 = modules.legacy.collective_combine(group, y, old_state).cpu()
            if captured.tensor is None:
                raise RuntimeError(
                    "legacy combine produced no captured FP32 collective output"
                )
            fp32 = captured.tensor
            if captured.origin == "all_reduce":
                fp32 = fp32.narrow(0, rank * rows, rows)
            old_fp32 = fp32.cpu()
            origin = captured.origin
        del captured, fp32, old_state
        phase = "a2a_combine"
        with capture_final_fp32(modules.kernel, legacy=False) as captured:
            new_bf16 = modules.a2a.all_to_all_combine(group, y, new_state).cpu()
            if captured.tensor is None:
                raise RuntimeError(
                    "A2A combine produced no captured FP32 ordered output"
                )
            new_fp32 = captured.tensor.cpu()
        del captured, y
        phase = "combine_checks"

        def check_combine():
            for name, bf16, fp32 in (
                ("legacy", old_bf16, old_fp32),
                ("A2A", new_bf16, new_fp32),
            ):
                exact_tensor(
                    bf16, fp32.to(torch.bfloat16), name + " final cast", args.chunk_rows
                )
                if not torch.isfinite(fp32).all():
                    raise AssertionError(
                        name + " nonfinite output / dummy contamination"
                    )
                if not active[rank // 8] and torch.count_nonzero(fp32):
                    raise AssertionError(
                        name + " idle source received a nonzero output"
                    )

        checked_together(
            check_combine,
            group=group,
            device=device,
            rank=rank,
            label=label + "/combine",
        )
        fp32_error, bf16_error = error_metrics(new_fp32, old_fp32), error_metrics(
            new_bf16, old_bf16
        )
        emit(
            dict(
                event="rank_result",
                case=label,
                rank=rank,
                dispatch_exact=True,
                sent=sum(new_state.send_splits),
                received=new_state.received,
                global_routes=new_state.global_routes,
                send_splits=new_state.send_splits,
                recv_splits=new_state.recv_splits,
                legacy_fp32_origin=origin,
                fp32=fp32_error,
                bf16=bf16_error,
            )
        )
        global_fp32 = summarize_metrics(fp32_error, group=group, device=device)
        global_bf16 = summarize_metrics(bf16_error, group=group, device=device)
        if rank == 0:
            emit(
                dict(
                    event="case_result",
                    case=label,
                    dispatch_exact=True,
                    fp32=global_fp32,
                    bf16=global_bf16,
                    association="legacy_owner_HCCL_vs_source_expert_order",
                )
            )
    except Exception as exc:
        emit(
            dict(
                event="exception",
                case=label,
                rank=rank,
                phase=phase,
                error=repr(exc),
                traceback=traceback.format_exc(),
            )
        )
        raise
    finally:
        modules.legacy._idle_dp_counts.reset(token)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--module-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[32, 1024])
    parser.add_argument(
        "--legacy-reduce",
        choices=["all_reduce", "reduce_scatter"],
        default="reduce_scatter",
    )
    parser.add_argument("--chunk-rows", type=int, default=256)
    parser.add_argument(
        "--empty-expert-rows",
        type=int,
        nargs="+",
        choices=[0, 1],
        default=[1, 0],
        help="empty GMM output shapes to test, on every rank (default: 1 0)",
    )
    parser.add_argument("--timeout-seconds", type=int, default=300)
    args = parser.parse_args()
    args.module_dir = args.module_dir.resolve()
    if not args.rows or any(rows <= 0 or rows > 1024 for rows in args.rows):
        parser.error(
            "--rows must be in [1,1024]; zero rows is always appended separately"
        )
    if args.chunk_rows <= 0 or args.timeout_seconds <= 0:
        parser.error("chunk rows and timeout must be positive")
    for leaf in (
        "glm53_collectives.py",
        "glm53_all_to_all.py",
        "glm53_ordered_combine.py",
    ):
        if not (args.module_dir / leaf).is_file():
            parser.error(f"missing candidate module: {args.module_dir / leaf}")
    rank, world = int(os.environ.get("RANK", "-1")), int(
        os.environ.get("WORLD_SIZE", "0")
    )
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if world != WORLD or not 0 <= rank < WORLD or local_rank != rank % 8:
        parser.error(
            "requires external launcher RANK=0..15, WORLD_SIZE=16, LOCAL_RANK=RANK%8"
        )
    if int(os.environ.get("LOCAL_WORLD_SIZE", "8")) != 8:
        parser.error("requires eight local ranks per host")

    import torch_npu  # noqa: F401; never imported by CPU help/source checks

    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")
    torch.set_num_threads(1)
    settings = configure_flags(args)
    dist.init_process_group(
        "hccl", init_method="env://", timeout=timedelta(seconds=args.timeout_seconds)
    )
    group = dist.group.WORLD
    try:
        if (
            dist.get_world_size(group) != WORLD
            or dist.get_rank(group) != rank
            or str(dist.get_backend(group)).lower() != "hccl"
        ):
            raise RuntimeError("test requires a real HCCL world16 group")
        install_topology(group, rank)
        modules = load_candidates(args.module_dir)
        cases = [
            (rows, active, pattern, empty_rows)
            for rows, active, pattern in case_list(args.rows)
            for empty_rows in args.empty_expert_rows
        ]
        emit(
            dict(
                event="start",
                rank=rank,
                world=WORLD,
                backend=str(dist.get_backend(group)),
                module_dir=str(args.module_dir),
                sha256=modules.hashes,
                flags=settings,
                cases=len(cases),
                precision="BF16 experts; FP32 products/accumulators; report differences",
            )
        )
        for rows, active, pattern, empty_rows in cases:
            run_case(
                modules, rows, active, pattern, rank, group, device, args, empty_rows
            )
        if rank == 0:
            emit(
                dict(
                    event="complete",
                    cases=len(cases),
                    dispatch_exact=True,
                    combine="finite; cast-consistent; errors reported without bitidentity assertion",
                )
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
