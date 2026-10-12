"""Real NPUGraph acceptance replay of the source-extracted KPool draft method.

Run in the Ascend environment (no model/weights or serving process required):
    python -B test/manual/ascend/test_glm53_draft_kpool_npu_graph.py

The working-tree method is compared with the pinned audited D baseline.
For an uploaded fixture without git, pass --source, --baseline-source
and --graph-ops-source pointing to actual source snapshots. Neither source
method is rewritten. The real Triton scatter_rows_ runs on NPU.

Only _decode_topk is replaced: it records the received request masks and writes
a tiny persistent NPU tail with torch.index_put_. It does NOT read accept counts
or reproduce the production compression/attention kernels. Fixed index_put_
destinations avoid the separate CANN dynamic-index capture issue documented in
graph_ops.py; the masked values remain dynamic. Slot 0 is an untouched sentinel
in this fixture, whereas production is allowed to write its reserved sink.
This checks tensor dependencies after NPUGraph capture, not model correctness,
the production KPool state kernels, or throughput/acceptance gains.
The fixture uses num_front_tokens=0; widened front windows are not covered.
"""

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


INDEX = "python/sglang/srt/hardware_backend/npu/attention/glm53/kpool_indexer.py"
BASELINE = "ee31fcc955791d07431848ceea20e8083dd76435"
# An uploaded standalone script may live outside the repository hierarchy.
ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / INDEX).is_file()), Path.cwd()
)
OWNER = "AscendIndexerKPoolMixin"
METHOD = "_draft_extend_topk"
TOKENS = [101, 102, 103, 104, 201, 202, 203, 204]
INITIAL = [[-900, -901, -902, -903], [40, 41, 42, 43], [60, 61, 62, 63]]
FULL_TAIL = [INITIAL[0], [104, 101, 102, 103], [203, 204, 201, 202]]
# Explicit fixture expectations, not an implementation of accepted-length logic.
CASES = (
    dict(
        name="full",
        accept=[4, 4],
        requests=[1, 2],
        tail=FULL_TAIL,
        masks=[[1, 2]] * 4,
        rows=list(range(8)),
    ),
    dict(
        name="partial_1_3",
        accept=[1, 3],
        requests=[1, 2],
        tail=[INITIAL[0], [40, 101, 42, 43], [203, 61, 201, 202]],
        masks=[[1, 2], [0, 2], [0, 2], [0, 0]],
        rows=[0, 4, 5, 6],
    ),
    dict(
        name="partial_3_1",
        accept=[3, 1],
        requests=[1, 2],
        tail=[INITIAL[0], [40, 101, 102, 103], [60, 61, 201, 63]],
        masks=[[1, 2], [1, 0], [1, 0], [0, 0]],
        rows=[0, 1, 2, 4],
    ),
    dict(
        name="padding_accept4",
        accept=[1, 4],
        requests=[1, 0],
        tail=[INITIAL[0], [40, 101, 42, 43], INITIAL[2]],
        masks=[[1, 0], [0, 0], [0, 0], [0, 0]],
        rows=[0],
    ),
)


def function_node(source, name, owner=None):
    nodes = ast.parse(source).body
    if owner is not None:
        nodes = next(
            n for n in nodes if isinstance(n, ast.ClassDef) and n.name == owner
        ).body
    matches = [n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(matches) != 1:
        raise ValueError((owner, name, len(matches)))
    return matches[0]


def extract(source, name, namespace, filename, owner=None):
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            function_node(source, name, owner),
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), filename, "exec"), namespace)
    return namespace[name]


def fingerprint(source):
    canonical = ast.dump(function_node(source, METHOD, OWNER), include_attributes=False)
    return dict(
        file_sha256=hashlib.sha256(source.encode()).hexdigest(),
        function_ast_sha256=hashlib.sha256(canonical.encode()).hexdigest(),
    )


class TailProbe:
    def __init__(self, torch, source, filename, scatter, device):
        self.torch = torch
        self.device = device
        self.calls = 0
        self.index_topk, self.index_kpool = 4, 4
        self._glm53_draft_graph_steps = 4
        self.initial = torch.tensor(INITIAL, dtype=torch.float32, device=device)
        self.tail = self.initial.clone()
        self.owner_ids = torch.tensor([1, 2], dtype=torch.int64, device=device)
        self.slots = torch.arange(4, dtype=torch.int64, device=device).view(1, 4)
        self.request_trace = torch.empty((4, 2), dtype=torch.int64, device=device)
        self.accept = torch.full((2,), 4, dtype=torch.int32, device=device)
        self.requests = self.owner_ids.clone()
        self.lens = torch.full((2,), 4, dtype=torch.int32, device=device)
        self.q = torch.arange(8, dtype=torch.float32, device=device).view(8, 1)
        self.key = torch.tensor(TOKENS, dtype=torch.float32, device=device).view(8, 1)
        self.weights = torch.ones((8, 1), dtype=torch.float32, device=device)
        self.gate = torch.zeros_like(self.weights)
        self.positions = torch.tensor(
            [5, 6, 7, 8, 6, 7, 8, 9], device=device, dtype=torch.int64
        )
        self.tables = torch.zeros((2, 4), dtype=torch.int32, device=device)
        self.batch = SimpleNamespace(
            batch_size=2,
            req_pool_indices=self.requests,
            extend_seq_lens=self.lens,
            extend_start_loc=torch.tensor([0, 4], dtype=torch.int32, device=device),
            spec_info=SimpleNamespace(
                num_accept_tokens=self.accept, num_front_tokens=0, num_tokens_per_req=4
            ),
            _ascend_kpool_metadata=SimpleNamespace(
                seq_lens=torch.tensor([9, 10], device=device)
            ),
        )
        namespace = dict(torch=torch, copy=copy.copy, scatter_rows_=scatter)
        extract(source, "_get_full_attn_metadata", namespace, filename)
        self.method = extract(source, METHOD, namespace, filename, OWNER)

    def _decode_topk(self, q, key, weights, gate, positions, batch, tables, layer_id):
        # This fixture has fixed owner IDs [1, 2]. The real source method alone
        # decides whether each lane receives its owner ID or the inactive 0.
        torch = self.torch
        req = batch.req_pool_indices
        self.request_trace[self.calls % 4].copy_(req)
        self.calls += 1
        live = (req == self.owner_ids).unsqueeze(1)
        slot = torch.remainder(positions, 4).unsqueeze(1)
        old = self.tail.index_select(0, self.owner_ids)
        values = torch.where(live & (self.slots == slot), key.expand(-1, 4), old)
        # Static destination rows, dynamic values. No writes to sentinel row 0.
        self.tail.index_put_((self.owner_ids,), values, accumulate=False)
        return key.to(torch.int32).expand(-1, self.index_topk + 3).contiguous()

    def run(self):
        return self.method(
            self,
            self.q,
            self.key,
            self.weights,
            self.gate,
            self.positions,
            self.batch,
            self.tables,
            0,
        )

    def feed(self, case):
        # Host/device setup is deliberately outside graph capture/replay.
        torch = self.torch
        self.accept.copy_(
            torch.tensor(case["accept"], dtype=self.accept.dtype, device=self.device)
        )
        self.requests.copy_(
            torch.tensor(
                case["requests"], dtype=self.requests.dtype, device=self.device
            )
        )
        self.tail.copy_(self.initial)
        self.request_trace.fill_(-1)

    def pointers(self):
        return {
            name: getattr(self, name).data_ptr()
            for name in (
                "accept",
                "requests",
                "lens",
                "tail",
                "positions",
                "key",
                "request_trace",
            )
        }


def check(torch, probe, output, case, variant, phase):
    torch.npu.synchronize()
    tail = probe.tail.cpu()
    masks = probe.request_trace.cpu()
    actual_output = output.cpu()
    if variant == "baseline":
        if case["requests"] == [1, 0]:
            expected_tail = [INITIAL[0], FULL_TAIL[1], INITIAL[2]]
            expected_masks, rows = [[1, 0]] * 4, list(range(4))
        else:
            expected_tail = FULL_TAIL
            expected_masks, rows = [[1, 2]] * 4, list(range(8))
    else:
        expected_tail, expected_masks, rows = case["tail"], case["masks"], case["rows"]
    expected_output = torch.zeros((8, 7), dtype=torch.int32)
    for row in rows:
        expected_output[row].fill_(TOKENS[row])
    message = f"{variant}/{phase}/{case['name']}"
    torch.testing.assert_close(
        tail, torch.tensor(expected_tail, dtype=tail.dtype), rtol=0, atol=0, msg=message
    )
    torch.testing.assert_close(
        masks,
        torch.tensor(expected_masks, dtype=masks.dtype),
        rtol=0,
        atol=0,
        msg=message,
    )
    torch.testing.assert_close(
        actual_output, expected_output, rtol=0, atol=0, msg=message
    )
    torch.testing.assert_close(
        tail[0], probe.initial[0].cpu(), rtol=0, atol=0, msg=message
    )
    torch.testing.assert_close(
        probe.lens.cpu(), torch.tensor([4, 4], dtype=torch.int32), rtol=0, atol=0
    )
    if variant == "baseline" and case["name"] == "partial_1_3":
        if torch.equal(tail, torch.tensor(case["tail"], dtype=tail.dtype)):
            raise AssertionError(
                "The baseline did not reproduce the rejected-slot overwrite"
            )
    print(
        json.dumps(
            dict(
                kind="case_pass",
                variant=variant,
                phase=phase,
                case=case["name"],
                tail=tail.tolist(),
                request_masks=masks.tolist(),
            )
        ),
        flush=True,
    )


def run_variant(torch, source, filename, scatter, device, variant):
    probe = TailProbe(torch, source, filename, scatter, device)
    pointers = probe.pointers()
    for case in CASES:
        probe.feed(case)
        check(torch, probe, probe.run(), case, variant, "eager")
    for _ in range(3):
        probe.feed(CASES[0])
        probe.run()
    torch.npu.synchronize()
    probe.feed(CASES[0])  # Capture with accepts [4, 4], not the partial fixture.
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = probe.run()
    torch.npu.synchronize()
    calls_after_capture = probe.calls
    # Partial replay comes first; returning to full and partial again catches
    # stale capture-time values and stale output rows in the same graph.
    for index in (1, 2, 3, 0, 1):
        case = CASES[index]
        probe.feed(case)
        graph.replay()
        check(torch, probe, output, case, variant, "graph")
        if probe.calls != calls_after_capture or probe.pointers() != pointers:
            raise AssertionError(
                "Replay reran Python or replaced an input/state buffer"
            )
    print(
        json.dumps(
            dict(
                kind="variant_pass",
                variant=variant,
                capture_accept=[4, 4],
                same_buffers=True,
                eager_cases=4,
                graph_replays=5,
            )
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--source",
        "--indexer-source",
        dest="indexer_source",
        type=Path,
        default=ROOT / INDEX,
        help="Actual patched kpool_indexer.py (standalone path allowed)",
    )
    parser.add_argument(
        "--baseline-source",
        type=Path,
        help="Actual unpatched source snapshot (otherwise use git)",
    )
    parser.add_argument(
        "--baseline-ref", default=BASELINE, help="Unpatched git revision; default audited D baseline"
    )
    parser.add_argument(
        "--graph-ops-source",
        type=Path,
        help="Actual graph_ops.py; defaults to indexer's sibling",
    )
    args = parser.parse_args()
    source_path = args.indexer_source.resolve()
    source = source_path.read_text()
    if args.baseline_source is not None:
        baseline_path = str(args.baseline_source.resolve())
        baseline = args.baseline_source.read_text()
    else:
        baseline_path = f"git:{args.baseline_ref}:{INDEX}"
        baseline = subprocess.check_output(
            ["git", "-C", str(ROOT), "show", f"{args.baseline_ref}:{INDEX}"],
            text=True,
        )
    source_info, baseline_info = fingerprint(source), fingerprint(baseline)
    if source_info["function_ast_sha256"] == baseline_info["function_ast_sha256"]:
        raise ValueError(
            "Patched and baseline methods are identical; supply the actual unpatched source/revision"
        )
    graph_path = (
        args.graph_ops_source or source_path.with_name("graph_ops.py")
    ).resolve()
    print(
        json.dumps(
            dict(
                kind="sources",
                patched=dict(path=str(source_path), **source_info),
                baseline=dict(path=baseline_path, **baseline_info),
                graph_ops=dict(
                    path=str(graph_path),
                    file_sha256=hashlib.sha256(graph_path.read_bytes()).hexdigest(),
                ),
            )
        ),
        flush=True,
    )

    # Imports stay below argument/source handling so --help works on CPU hosts.
    import torch
    import torch_npu

    if not torch.npu.is_available():
        raise RuntimeError("A real Ascend NPU is required; there is no CPU fallback")
    torch.npu.set_device(args.device)
    device = f"npu:{args.device}"
    spec = importlib.util.spec_from_file_location("_glm53_accept_graph_ops", graph_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # Triton inspects the actual module/source.
    spec.loader.exec_module(module)
    print(
        json.dumps(
            dict(
                kind="runtime",
                torch=torch.__version__,
                torch_npu=torch_npu.__version__,
                device=torch.npu.get_device_name(args.device),
            )
        ),
        flush=True,
    )
    for variant, text, filename in (
        ("baseline", baseline, baseline_path),
        ("patched", source, str(source_path)),
    ):
        run_variant(torch, text, filename, module.scatter_rows_, device, variant)
    print(
        json.dumps(
            dict(
                kind="pass",
                eager_cases=8,
                graph_replays=10,
                baseline_overwrite_reproduced=True,
                patched_partial_masks_dynamic=True,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
