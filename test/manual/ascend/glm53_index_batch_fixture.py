"""Isolated production-method fixture; never imports SGLang or selects a device.

CPU uses Triton's interpreter and a separately supplied LI reference. NPU uses
the actual metadata/scatter/snapshot/expand kernels and torch_npu LI. Only the
device/capture eligibility predicate may be bypassed for CPU exploration.
"""

import ast
from copy import copy
from functools import lru_cache
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
import torch.nn.functional as F

REL = "python/sglang/srt/hardware_backend/npu/attention/glm53/kpool_indexer.py"
if os.environ.get("GLM53_INDEX_BATCH_SOURCE"):
    SOURCE = Path(os.environ["GLM53_INDEX_BATCH_SOURCE"]).resolve()
    ROOT = SOURCE.parent
else:
    ROOT = Path(__file__).resolve().parents[3]
    SOURCE = ROOT / REL
BASELINE = "83cfe9668fb98ca674c3a28816dd99016e38b8b6"
FLAG = "SGLANG_GLM53_VERIFY_INDEX_BATCH"
EXPAND_FLAG = "SGLANG_GLM53_VERIFY_EXPAND_BATCH"
STATE = (
    "_kpool_tail_k",
    "_kpool_tail_score",
    "_kpool_mtp_tail_k",
    "_kpool_mtp_tail_score",
)
METHODS = {
    "_init_ascend_kpool",
    "_compress",
    "_decode_topk",
    "_verify_topk",
    "_verify_index_batch_enabled",
    "_expand_with_tail",
    "forward_npu",
}


@lru_cache(None)
def kernel(name, cpu):
    spec = importlib.util.spec_from_file_location(
        f"_index_batch_{name}_{cpu}", SOURCE.parent / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(os.environ, {"TRITON_INTERPRET": "1" if cpu else "0"}):
        spec.loader.exec_module(module)
    return module


def load_indexer(namespace, source=None):
    tree = ast.parse(source if source is not None else SOURCE.read_text())
    nodes = [
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
    ]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {
            "_ascend_lightning_weights",
            "_as_ascend_sparse_indices",
        }:
            nodes.append(node)
        if isinstance(node, ast.ClassDef) and node.name == "AscendIndexerKPoolMixin":
            node.body = [
                n
                for n in node.body
                if isinstance(n, ast.FunctionDef) and n.name in METHODS
            ]
            nodes.append(node)

    class Imports(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            if node.module in {
                "kpool_metadata_npu",
                "kpool_expand_npu",
                "sglang.srt.model_executor.runner_utils.capture_mode",
            }:
                return ast.Pass()  # Inject real kernels / explicit capture context.
            return node

    module = ast.fix_missing_locations(
        Imports().visit(ast.Module(body=nodes, type_ignores=[]))
    )
    namespace.update(torch=torch, F=F, os=os, copy=copy)
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["AscendIndexerKPoolMixin"]


def fixture(
    starts,
    *,
    requests=None,
    kind="random",
    topk=16,
    heads=8,
    extra=2,
    capacity=None,
    seed=53,
):
    """CPU-generated fixtures include shuffled pages, poisoned future, safe sinks."""
    gen = torch.Generator().manual_seed(seed)
    bs = len(starts)
    requests = list(reversed(range(1, bs + 1))) if requests is None else requests
    slots = max(bs + 3, max(requests, default=0) + 3)
    cap = max(max(starts, default=0) + 4, capacity or 0)
    pool_pages = cap // 256 + 2
    table = torch.full((bs, pool_pages * 4), -1, dtype=torch.int32)
    pages = torch.randperm(bs * pool_pages, generator=gen).reshape(bs, pool_pages) + 1
    table[:, ::4] = pages.int()
    req = torch.tensor(requests, dtype=torch.int64)
    positions = (
        torch.tensor(starts, dtype=torch.int64)[:, None] + torch.arange(4)
    ).flatten()
    positions = F.pad(positions, (0, extra))
    n = bs * 4 + extra
    rand = lambda *shape: torch.randn(*shape, generator=gen)
    data = dict(
        q=rand(n, heads, 128).to(torch.bfloat16),
        key=rand(n, 128).abs().to(torch.bfloat16),
        weights=rand(n, heads),
        gate=rand(n, 128),
        positions=positions,
        req=req,
        table=table,
        cache=rand(bs * pool_pages + 1, 64, 1, 128).to(torch.bfloat16),
        ape=rand(4, 128),
        _kpool_tail_k=rand(slots, 4, 128).abs().to(torch.bfloat16),
        _kpool_tail_score=rand(slots, 4, 128),
        _kpool_mtp_tail_k=torch.full((bs + 2, 4, 4, 128), -13, dtype=torch.bfloat16),
        _kpool_mtp_tail_score=torch.full((bs + 2, 4, 4, 128), -17.0),
    )
    if kind == "ties":
        data["q"].zero_()  # All scores exactly zero, including >topk boundaries.
    elif kind == "near-ties":
        data["cache"].fill_(1)
        flat = data["cache"].view(-1, 128)
        flat[:, 0] = (1 + (torch.arange(len(flat)) % 3) / 128).to(torch.bfloat16)
    elif kind == "closing":
        data["q"].fill_(1)
        data["weights"].fill_(1)
        data["gate"].zero_()
        data["ape"].zero_()
        data["_kpool_tail_score"].zero_()
        data["_kpool_tail_k"].fill_(64)
        data["key"].fill_(64)
        data["cache"].zero_()
        # A stale high-score entry in the not-yet-closed pool must be invisible.
        for row, start in enumerate(starts):
            pool = start // 4
            data["cache"][int(pages[row, pool // 64]), pool % 64].fill_(128)
    elif kind != "random":
        raise ValueError(kind)
    data["cache"][0].zero_()
    # All non-closing rows write cache row 0. Positive compressed keys make
    # every such write +0 (rather than racing +0/-0 bit patterns). Multiple
    # padding tail rows likewise need identical payloads for strict sink bits.
    for row, request in enumerate(requests):
        if request == 0:
            data["key"][row * 4 : (row + 1) * 4].zero_()
            data["gate"][row * 4 : (row + 1) * 4].zero_()
            data["table"][row].fill_(-1)
    return NS(data=data, bs=bs, topk=topk, heads=heads, kind=kind)


class Runner:
    def __init__(
        self,
        case,
        op,
        *,
        enabled,
        batch_expand=False,
        device="cpu",
        source=None,
        fused=False,
        trace=True,
        bypass_cpu_guard=True,
    ):
        self.case, self.device = case, torch.device(device)
        self.data = {k: v.to(device).clone() for k, v in case.data.items()}
        self.trace_enabled = trace
        self.calls, self.pooled, self.states, self.expands, self.events = (
            [],
            [],
            [],
            [],
            [],
        )
        cpu = self.device.type == "cpu"
        real_expand = kernel("kpool_expand_npu", cpu).expand_with_tail
        in_per_step_expand = False

        def record_expand(ids, positions):
            self.expands.append((tuple(ids.shape), positions))
            self.events.append("expand")

        def fused_expand(ids, positions):
            # Direct batched dispatch bypasses _expand_with_tail. Record it
            # here, without double-counting that helper's existing fused path.
            if not in_per_step_expand:
                record_expand(ids, positions)
            return real_expand(ids, positions)

        ns = dict(
            get_server_args=lambda: NS(
                disaggregation_mode="decode", max_running_requests=16
            ),
            get_device=lambda: NS(device=self.device),
            get_spec=lambda: NS(speculative_num_draft_tokens=None),
            get_is_capture_mode=lambda: True,
            _get_full_attn_metadata=lambda b: b._ascend_kpool_metadata,
            _get_index_k_buffer=lambda b, _: b.cache,
            decode_metadata=kernel("kpool_metadata_npu", cpu).decode_metadata,
            scatter_rows_=kernel("graph_ops", cpu).scatter_rows_,
            scatter_kpool_tail_=kernel(
                "kpool_tail_scatter_npu", cpu
            ).scatter_kpool_tail_,
            snapshot_kpool_tails_=kernel(
                "kpool_snapshot_npu", cpu
            ).snapshot_kpool_tails_,
            expand_with_tail=fused_expand,
            torch_npu=NS(npu_lightning_indexer=self.lightning),
        )
        self.op = op
        self.cls = load_indexer(ns, source)
        self.obj = self.cls()
        obj = self.obj
        obj.index_kpool, obj.index_topk, obj.head_dim = 4, case.topk, 128
        obj._batch_kpool_verify_index = enabled
        # Explicit opt-in, independent of the caller's ambient environment.
        obj._batch_kpool_verify_expand = enabled and batch_expand
        obj._fuse_kpool_tail_scatter = fused
        obj._fuse_kpool_verify_snapshot = fused
        obj.index_kpool_compress_ape = self.data["ape"]
        for attr in STATE:
            setattr(obj, attr, self.data[attr])
        if cpu and bypass_cpu_guard and enabled:
            obj._verify_index_batch_enabled = (
                lambda q, steps, bs: steps == 4 and 0 < bs <= 16
            )
        original_decode, original_expand = obj._decode_topk, obj._expand_with_tail

        def decode(*args, **kwargs):
            if trace:
                self.states.append(self.state())  # Includes previous snapshot.
            self.events.append("update")
            result = original_decode(*args, **kwargs)
            if trace:
                self.states.append(self.state())
            return result

        def expand(ids, positions):
            nonlocal in_per_step_expand
            record_expand(ids, positions)
            in_per_step_expand = True
            try:
                return original_expand(ids, positions)
            finally:
                in_per_step_expand = False

        obj._decode_topk, obj._expand_with_tail = decode, expand
        self.batch = NS(
            batch_size=case.bs,
            _original_batch_size=case.bs,
            req_pool_indices=self.data["req"],
            spec_info=NS(draft_token_num=4),
            cache=self.data["cache"],
            _ascend_kpool_metadata=NS(
                seq_lens=torch.zeros(case.bs, dtype=torch.int32, device=device)
            ),
        )

    def lightning(self, **kwargs):
        self.events.append("indexer")
        self.calls.append(kwargs)
        ids, values = self.op(**kwargs)
        self.pooled.append(ids)
        return ids, values

    def state(self):
        return {k: self.data[k].clone() for k in (*STATE, "cache")}

    def reset(self, case=None):
        case = case or self.case
        for k, value in case.data.items():
            self.data[k].copy_(value)

    def run(self):
        self.calls.clear()
        self.pooled.clear()
        self.states.clear()
        self.expands.clear()
        self.events.clear()
        d = self.data
        out = self.obj._verify_topk(
            d["q"],
            d["key"],
            d["weights"],
            d["gate"],
            d["positions"],
            self.batch,
            d["table"],
            0,
        )
        if self.trace_enabled:
            self.states.append(self.state())
        return out

    def raw_topk(self):
        if len(self.pooled) == 1:
            return self.pooled[0].reshape(self.case.bs, 4, -1)
        return torch.stack(self.pooled, 1).reshape(self.case.bs, 4, -1)


def cpu_lightning(
    *,
    query,
    key,
    weights,
    actual_seq_lengths_query,
    actual_seq_lengths_key,
    block_table,
    layout_query,
    layout_key,
    sparse_count,
    sparse_mode,
):
    """Documented visibility + explicit stable ties; NOT a CANN arithmetic model."""
    assert layout_query == "TND" and layout_key == "PA_BSND" and sparse_mode == 3
    result = torch.full((len(query), 1, sparse_count), -1, dtype=torch.int32)
    begin = 0
    for batch, end in enumerate(actual_seq_lengths_query.tolist()):
        length = int(actual_seq_lengths_key[batch])
        qlen = end - begin
        for i in range(qlen):
            visible = max(0, min(length, length - qlen + i + 1))
            ids = torch.arange(visible)
            pages = block_table[batch, ids // 64].long()
            keys = key[pages, ids % 64, 0].float()
            scores = (
                (query[begin + i].float() @ keys.T).relu()
                * weights[begin + i].float()[:, None]
            ).sum(0)
            selected = torch.argsort(scores, descending=True, stable=True)[
                :sparse_count
            ]
            result[begin + i, 0, : len(selected)] = selected.int()
        begin = end
    assert begin == len(query)
    return result, None


def exact(a, b, label):
    a, b = a.detach().cpu().contiguous(), b.detach().cpu().contiguous()
    if (
        a.shape != b.shape
        or a.dtype != b.dtype
        or not torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    ):
        detail = ""
        if a.shape == b.shape and a.dtype == b.dtype:
            aa, bb = a.view(torch.uint8).flatten(), b.view(torch.uint8).flatten()
            where = torch.nonzero(aa != bb).flatten()
            detail = f"; different_bytes={len(where)}, first_byte={int(where[0])}"
        raise AssertionError(
            f"not bitwise exact: {label}; shape={a.shape}/{b.shape}, dtype={a.dtype}/{b.dtype}{detail}"
        )


def compare(a, out_a, b, out_b):
    exact(out_a, out_b, "expanded topk (integer order, no sorting)")
    exact(a.raw_topk(), b.raw_topk(), "raw pool topk (integer order, no sorting)")
    if len(a.states) != len(b.states):
        raise AssertionError("state trace length")
    for i, (sa, sb) in enumerate(zip(a.states, b.states)):
        for k in sa:
            exact(sa[k], sb[k], f"step trace {i} {k}")
    for k in (*STATE, "cache"):
        exact(a.data[k], b.data[k], f"final {k}")
    if [x[0] for x in a.expands] != [x[0] for x in b.expands]:
        raise AssertionError("expand launch shapes changed")
