"""NPU A/B contract entry; never run this on a CPU-only validation host.

Keep the existing prepared_recurrent_v2 package on PYTHONPATH, set the new
SGLANG_GLM53_KDA_VARLEN_ASCENDC_LIB, then run:
  python test/manual/ascend/test_glm53_kda_varlen_ascendc_npu.py \
    --suite all --output /remote/artifacts/kda-ab.json

A uses the configured original kernel (prepared_v2 for eligible long P cases).
B must actually dispatch to the new op. Original reference tolerances from
test_glm53_kda_prefix.py are preserved: BF16 output rtol=.01/atol=.001,
FP32 state/checkpoint rtol=2e-4/atol=2e-5. Untouched state and scratch, and
candidate graph vs candidate eager, remain exact. This is a correctness entry,
not a latency benchmark. Run the original prefix suite and the MTP state-mover
test as well; the latter now accepts --heads 32 --steps 5.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import traceback
from unittest.mock import patch

import torch
import torch_npu  # noqa: F401

import sglang.srt.layers.quantization  # noqa: F401
from sglang.srt.hardware_backend.npu.attention.glm53 import (
    kda_prefill_npu as prepared,
    kda_varlen_ascendc as adapter,
)
from sglang.srt.hardware_backend.npu.attention.glm53.kda_recurrent_npu import (
    glm_kda_varlen_recurrent_npu as recurrent,
)

spec = importlib.util.spec_from_file_location(
    "prefix_oracle", Path(__file__).with_name("test_glm53_kda_prefix.py")
)
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


def close(left, right, state=False, exact=False):
    torch.testing.assert_close(
        left.detach().cpu(),
        right.detach().cpu(),
        rtol=0 if exact else (2e-4 if state else 0.01),
        atol=0 if exact else (2e-5 if state else 0.001),
    )


def error(left, right):
    return float(
        (left.detach().float().cpu() - right.detach().float().cpu()).abs().max()
    )


def make_case(
    heads,
    lengths,
    prefill,
    steps=None,
    dtype=torch.int32,
    cut=None,
    slots=None,
    disabled_track=False,
    padded_mid=False,
):
    batch = len(lengths)
    slots = slots if slots is not None else list(range(1, batch + 1))
    dev = oracle.DEVICE
    tensors = oracle.make_inputs(sum(lengths), heads)
    constants = oracle.make_constants(heads)
    initial = (torch.randn(batch + 5, heads, 128, 128, device=dev) * 0.1).transpose(
        -1, -2
    )
    args = dict(
        **tensors,
        **constants,
        initial_state_indices=torch.tensor(slots, device=dev, dtype=dtype),
        cu_seqlens=oracle.offsets(lengths).to(dtype),
        prefill=prefill,
        intermediate_state=None,
        track_state_indices=None,
        track_state_lens=None,
    )
    if cut is not None:
        args.update(
            track_state_indices=torch.tensor(
                [0 if disabled_track else batch + 2], device=dev, dtype=dtype
            ),
            track_state_lens=torch.tensor(
                [-777 if disabled_track else cut], device=dev, dtype=torch.int64
            ),
        )

    def buffers():
        state = oracle.clone_state(initial)
        mid = backing = None
        if steps is not None:
            if padded_mid:
                backing = torch.full(
                    (batch + 3, steps + 2, heads + 1, 128, 128), -17.0, device=dev
                )
                mid = backing[1 : batch + 2, :steps, :heads].transpose(-1, -2)
            else:
                backing = torch.full(
                    (batch + 1, steps, heads, 128, 128), -17.0, device=dev
                )
                mid = backing.transpose(-1, -2)
        return state, mid, backing

    return args, initial, buffers, slots


def invoke(args, state, mid, candidate):
    with patch.dict(os.environ, {adapter.ENABLE_ENV: str(int(candidate))}):
        return recurrent(
            **(args | dict(initial_state_source=state, intermediate_state=mid))
        )


def check_case(row, **config):
    args, initial, buffers, slots = make_case(**config)
    state_a, mid_a, backing_a = buffers()
    state_b, mid_b, backing_b = buffers()
    call = args | dict(initial_state_source=state_b, intermediate_state=mid_b)
    assert adapter.unsupported_reason(**call) is None, adapter.unsupported_reason(
        **call
    )
    with patch.object(
        prepared,
        "_ascendc_prepared_operator",
        wraps=prepared._ascendc_prepared_operator,
    ) as old_loader:
        output_a = invoke(args, state_a, mid_a, False)
    require_prepared = config["prefill"] and args["q"].shape[1] >= 128
    if require_prepared:
        assert (
            old_loader.called
        ), "A did not execute the existing prepared_recurrent_v2 path"
    with patch.object(adapter, "_operator", wraps=adapter._operator) as new_loader:
        output_b = invoke(args, state_b, mid_b, True)
    assert new_loader.called, "B never dispatched the AscendC candidate"
    torch.npu.synchronize()
    row.update(
        baseline="prepared_recurrent_v2" if require_prepared else "original_generic",
        candidate_dispatched=True,
        output_max_abs=error(output_a, output_b),
        state_max_abs=error(state_a, state_b),
        state_stride=list(state_b.stride()),
        mid_stride=list(mid_b.stride()) if mid_b is not None else None,
        intermediate_max_abs=(
            error(backing_a, backing_b) if backing_b is not None else None
        ),
    )
    close(output_b, output_a)
    close(state_b, state_a, state=True)
    if mid_b is not None:
        close(backing_b, backing_a, state=True)
        close(state_b, initial, exact=True)  # verify must not commit a source
        # All rows/padding gaps not written by a valid request stay sentinel.
        untouched = backing_a == -17.0
        close(backing_b[untouched], backing_a[untouched], exact=True)
    touched = set(slots) - {0, -1} if mid_b is None else set()
    if config.get("cut") is not None and not config.get("disabled_track"):
        touched.add(len(slots) + 2)
    for slot in set(range(initial.shape[0])) - touched:
        close(state_b[slot], initial[slot], exact=True)
    # Short cases include a CPU mathematical reference independent of either
    # implementation. Longer P cases use the existing prepared_v2 A oracle.
    if sum(config["lengths"]) <= 129:
        begin = 0
        for i, (slot, length) in enumerate(zip(slots, config["lengths"])):
            if length == 0:
                continue
            tensors = {
                n: args[n][:, begin : begin + length] for n in ("q", "k", "v", "a", "b")
            }
            constants = {n: args[n] for n in ("A_log", "dt_bias", "lower_bound")}
            seed = initial[slot] if slot > 0 else torch.zeros_like(initial[0])
            expected, states = oracle.cpu_reference(seed, tensors, constants)
            close(output_b[:, begin : begin + length], expected)
            if mid_b is not None and slot > 0:
                for step in range(length):
                    close(mid_b[i, step], states[step + 1], state=True)
            elif mid_b is None and slot > 0:
                close(state_b[slot], states[-1], state=True)
            if config.get("cut") is not None and not config.get("disabled_track"):
                close(state_b[len(slots) + 2], states[config["cut"]], state=True)
            begin += length
        row["independent_cpu_reference"] = True


def check_graph(row, prefill):
    config = dict(
        heads=8 if prefill else 32,
        lengths=[129] if prefill else [5, 5, 5],
        prefill=prefill,
        steps=None if prefill else 5,
        dtype=torch.int32 if prefill else torch.int64,
        cut=64 if prefill else None,
        padded_mid=not prefill,
    )
    args, initial, buffers, _ = make_case(**config)
    eager, eager_mid, eager_backing = buffers()
    captured, captured_mid, captured_backing = buffers()
    for _ in range(3):
        captured.copy_(initial)
        invoke(args, captured, captured_mid, True)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = invoke(args, captured, captured_mid, True)
    for replay in range(8):
        slots = [0 if replay % 3 == 0 else 1] if prefill else [replay % 3, 3, 4]
        args["initial_state_indices"].copy_(
            torch.tensor(
                slots, dtype=args["initial_state_indices"].dtype, device=oracle.DEVICE
            )
        )
        if prefill:
            destination = 0 if replay % 4 == 0 else 3
            args["track_state_indices"].fill_(destination)
            args["track_state_lens"].fill_(
                [0, 64, 129][replay % 3] if destination else -777
            )
        else:
            # Changing an int64 boundary must be observed after replay's
            # captured int64->int32 cast, including an empty request row.
            args["cu_seqlens"].copy_(
                torch.tensor(
                    [0, 0, 5, 10] if replay % 2 else [0, 5, 10, 15],
                    dtype=torch.int64,
                    device=oracle.DEVICE,
                )
            )
        for name in ("q", "k", "v", "a", "b"):
            args[name].copy_(torch.randn_like(args[name]) * 0.2)
        eager.copy_(initial)
        captured.copy_(initial)
        if eager_backing is not None:
            eager_backing.fill_(-17.0)
            captured_backing.fill_(-17.0)
        expected = invoke(args, eager, eager_mid, True).cpu()
        expected_state = eager.cpu()
        expected_backing = eager_backing.cpu() if eager_backing is not None else None
        torch.npu.synchronize()
        graph.replay()
        torch.npu.synchronize()
        close(output, expected, exact=True)
        close(captured, expected_state, exact=True)
        if captured_backing is not None:
            close(captured_backing, expected_backing, exact=True)
    row.update(replays=8, exact=True, int64_dynamic_metadata=not prefill)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("all", "p", "d", "graph"), default="all")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    oracle.DEVICE = f"npu:{args.device}"
    torch.set_num_threads(4)
    torch.manual_seed(1023)
    os.environ["SGLANG_GLM53_KDA_PREFILL_PREPARE"] = "1"
    os.environ["SGLANG_GLM53_KDA_PREPARED_ASCENDC"] = "1"
    cases = []
    if args.suite in ("all", "p"):
        prepared._ascendc_prepared_operator()  # fail rather than silently change A
        for heads in (4, 8):
            for tokens in (1, 17, 129, 513, 4097):
                for cut in (0, tokens // 2, tokens):
                    cases.append(
                        (
                            f"p-h{heads}-t{tokens}-cut{cut}",
                            dict(heads=heads, lengths=[tokens], prefill=True, cut=cut),
                        )
                    )
            cases.append(
                (
                    f"p-h{heads}-slot0",
                    dict(heads=heads, lengths=[17], prefill=True, cut=7, slots=[0]),
                )
            )
            cases.append(
                (
                    f"p-h{heads}-disabled-track",
                    dict(
                        heads=heads,
                        lengths=[129],
                        prefill=True,
                        cut=0,
                        disabled_track=True,
                    ),
                )
            )
    if args.suite in ("all", "d"):
        for heads in (16, 32, 64):
            for dtype in (torch.int32, torch.int64):
                for steps in (1, 4, 5):
                    cases.append(
                        (
                            f"d-h{heads}-s{steps}-{dtype}",
                            dict(
                                heads=heads,
                                lengths=[steps] * 3,
                                prefill=False,
                                steps=steps if steps > 1 else None,
                                dtype=dtype,
                                slots=[3, 0, 1],
                                padded_mid=True,
                            ),
                        )
                    )
            cases.append(
                (
                    f"d-h{heads}-ragged",
                    dict(
                        heads=heads,
                        lengths=[0, 5, 2],
                        prefill=False,
                        steps=5,
                        dtype=torch.int64,
                        slots=[3, 1, 0],
                        padded_mid=True,
                    ),
                )
            )
    if args.suite in ("all", "graph"):
        cases += [("graph-p", True), ("graph-verify5", False)]
    report = dict(
        passed=False,
        complete=False,
        kernel_commit=adapter.KERNEL_COMMIT,
        library=os.environ.get(adapter.LIB_ENV),
        results=[],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        torch_version=torch.__version__,
        torch_npu_version=torch_npu.__version__,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name, config in cases:
        row = dict(name=name, passed=False)
        start = time.monotonic()
        try:
            if isinstance(config, bool):
                check_graph(row, config)
            else:
                check_case(row, **config)
            row["passed"] = True
        except Exception:
            row["traceback"] = traceback.format_exc()
        row["seconds"] = time.monotonic() - start
        report["results"].append(row)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    report.update(complete=True, passed=all(r["passed"] for r in report["results"]))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
