"""Isolated source harness. CPU emulation is NOT evidence of Ascend parity."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

REL = Path("python/sglang/srt/hardware_backend/npu/attention/glm53")
BASE = "d20cbd26be0abee69f8a0ca1cad184c31dfb7cc4"


def load_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def extract(path, name, scope, observe=False, postop=False):
    """Execute the actual function AST, optionally inserting a read-only tap."""
    node = copy.deepcopy(
        next(
            n
            for n in ast.walk(ast.parse(path.read_text()))
            if isinstance(n, ast.FunctionDef) and n.name == name
        )
    )
    if postop:

        class ReplacePostop(ast.NodeTransformer):
            count = 0

            def visit_Expr(self, n):
                call = n.value
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == "logits"
                    and call.func.attr in ("relu_", "mul_")
                ):
                    self.count += 1
                    if call.func.attr == "relu_":
                        return ast.parse(
                            "_isolated_postop(logits, w_req[start:end])"
                        ).body[0]
                    return None
                return n

        replacement = ReplacePostop()
        node = replacement.visit(node)
        if replacement.count != 2:
            raise RuntimeError(
                "Source changed: expected exactly legacy ReLU and gate product"
            )
    if observe:

        class Tap(ast.NodeTransformer):
            count = 0

            def visit_Assign(self, n):
                if any(
                    isinstance(t, ast.Name) and t.id == "logical_pos" for t in n.targets
                ):
                    self.count += 1
                    return [
                        ast.parse("observer(i, start, first_pos, logits)").body[0],
                        n,
                    ]
                return n

        tap = Tap()
        node = tap.visit(node)
        if tap.count != 1:
            raise RuntimeError(
                "Source changed: expected exactly one pre-mask score tap"
            )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    return scope[name]


def legacy_scores(q, w, k):
    s = torch.einsum("qhd,kd->qhk", q, k)
    s.relu_()
    s.mul_(w.unsqueeze(-1))
    return s.sum(1)


def cpu_compact_emulation(q, w, k):
    """Materialize only 4x32x64 at once; CPU reduction is not Triton reduction."""
    out = torch.empty((len(q), len(k)), dtype=torch.bfloat16, device=q.device)
    for row in range(0, len(q), 4):
        for key in range(0, len(k), 64):
            out[row : row + 4, key : key + 64] = legacy_scores(
                q[row : row + 4], w[row : row + 4], k[key : key + 64]
            )
    return out


def make_scorer(
    repo, fused, old_tile, backend=None, observer=None, postop=None, integrated=False
):
    path = Path(repo) / REL / "kpool_indexer.py"
    env = dict(
        os.environ,
        SGLANG_GLM53_PREFILL_POOLED_SCORE="always" if fused else "0",
        SGLANG_GLM53_PREFILL_TOPK_TILE_ROWS=str(old_tile),
        SGLANG_GLM53_PREFILL_POOLED_POSTOP="auto" if integrated else "0",
        SGLANG_GLM53_PREFILL_SCORE_VALIDATE="0",
    )
    calls = []

    def candidate(q, w, k):
        calls.append((tuple(q.shape), tuple(k.shape)))
        return backend(q, w, k)

    def apply_postop(scores, weights):
        active = postop(scores, weights)
        if active:
            calls.append(tuple(scores.shape))
        return active

    scope = dict(
        torch=torch,
        F=F,
        os=SimpleNamespace(environ=env),
        get_server_args=lambda: SimpleNamespace(disaggregation_mode="prefill"),
        _log_prefill_pooled_score_once=lambda *_: None,
        _prefill_pooled_score_backend=lambda: (candidate, lambda *_: True),
        _prefill_pooled_postop_backend=lambda: apply_postop,
        _log_prefill_pooled_postop_once=lambda *_: None,
        observer=observer,
        _isolated_postop=apply_postop,
    )
    extract(path, "_prefill_topk_tile_rows", scope)
    if "_prefill_pooled_postop_min_keys" in path.read_text():
        extract(path, "_prefill_pooled_postop_min_keys", scope)
    # This bypass is explicit: execute the candidate on CPU, or force an isolated
    # NPU comparison even when production auto would choose the legacy tile32.
    scope["_prefill_pooled_score_enabled"] = lambda *_: fused
    method = extract(
        path,
        "_prefill_pooled_topk",
        scope,
        observe=observer is not None,
        postop=postop is not None and not integrated,
    )
    owner = SimpleNamespace(index_topk=2048, index_kpool=4)

    def run(q, w, histories, lengths, ends):
        calls.clear()
        meta = SimpleNamespace(
            batch_size=len(lengths), extend_seq_lens_cpu=lengths, seq_lens_cpu=ends
        )
        return method(owner, q, w, histories, meta)

    run.calls = calls
    run.scope = scope
    return run


class Capture:
    def __init__(self):
        self.rows = {}
        self.tiles = []

    def __call__(self, request, start, first_pos, logits):
        host = logits.detach().cpu().clone()
        self.tiles.append([request, first_pos + start, *host.shape])
        for i, row in enumerate(host):
            self.rows[(request, first_pos + start + i)] = row


def compare_scores(a, b):
    if a.rows.keys() != b.rows.keys():
        raise AssertionError("Different scored row coverage")
    total = different = nonfinite = 0
    maximum = error_sum = 0.0
    examples = []
    for key, x in a.rows.items():
        y = b.rows[key]
        if x.shape != y.shape:
            raise AssertionError(
                f"Candidate dimension changed at {key}: {x.shape} / {y.shape}"
            )
        finite = torch.isfinite(x) & torch.isfinite(y)
        bad = x.view(torch.int16) != y.view(torch.int16)
        delta = (x.float() - y.float()).abs()
        count = int(bad.sum())
        total += x.numel()
        different += count
        nonfinite += int((~finite).sum())
        if finite.any():
            maximum = max(maximum, float(delta[finite].max()))
            error_sum += float(delta[finite].double().sum())
        if count and len(examples) < 8:
            col = int(bad.nonzero()[0])
            examples.append(
                dict(
                    request=key[0],
                    position=key[1],
                    pool=col,
                    old=float(x[col]),
                    new=float(y[col]),
                )
            )
    return dict(
        elements=total,
        different_bits=different,
        nonfinite=nonfinite,
        max_abs=maximum,
        mean_abs=error_sum / max(1, total),
        examples=examples,
        exact=different == 0 and nonfinite == 0,
    )


def compare_ids(a, b, lengths, ends):
    a, b = a.cpu(), b.cpu()
    if a.shape != b.shape:
        raise AssertionError("Different ID shapes")
    visible = torch.cat(
        [
            torch.arange(end - length + 1, end + 1) // 4
            for length, end in zip(lengths, ends)
        ]
    )
    invalid = (b < -1) | ((b >= visible[:, None]) & (b != -1))
    ordered = (a == b).all(1)
    sets = (a.sort(1).values == b.sort(1).values).all(1)
    return dict(
        rows=len(a),
        ordered_exact_rows=int(ordered.sum()),
        set_exact_rows=int(sets.sum()),
        causal_invalid=int(invalid.sum()),
        exact=bool(ordered.all()) and not bool(invalid.any()),
    )


def boundary_metrics(reference, candidate_ids, lengths, ends):
    """Diagnostics only. A tie or zero strict omissions never relaxes acceptance."""
    candidate_ids = candidate_ids.cpu()
    ties = eligible = omitted = 0
    margins = []
    offset = 0
    for request, (length, end) in enumerate(zip(lengths, ends)):
        for local in range(length):
            pos = end - length + local
            key = (request, pos)
            if key not in reference.rows:
                continue
            scores = reference.rows[key].float()[: (pos + 1) // 4]
            ids = candidate_ids[offset + local]
            valid = ids[(ids >= 0) & (ids < len(scores))].long()
            if len(scores) > 512:
                values = scores.topk(513).values
                margin = float(values[511] - values[512])
                ties += margin == 0
                eligible += 1
                margins.append(margin)
            if len(valid):
                selected = scores[valid]
                omitted += int(
                    (scores > selected.min()).sum() - (selected > selected.min()).sum()
                )
        offset += length
    return dict(
        rows_with_513_visible=eligible,
        tied_512_513_rows=ties,
        min_margin=min(margins) if margins else None,
        strictly_better_old_pools_omitted=omitted,
    )


FAMILIES = (
    "random",
    "zeros",
    "signed",
    "dot_round",
    "product_round",
    "head_cancel",
    "rank_flip",
    "dot_cancel",
    "boundary_tie",
    "tiny",
    "large",
)


def fixture(family, rows, pools, seed=1012):
    """Deterministic CPU inputs shared by CPU and NPU runs, with FP32 gates."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn((rows, 32, 128), generator=gen).bfloat16()
    k = torch.randn((pools, 128), generator=gen).bfloat16()
    w = torch.randn((rows, 32), generator=gen)
    if family == "random":
        return q, w, k
    if family in ("tiny", "large"):
        scale = 2.0 ** (-50 if family == "tiny" else 40)
        return (q.float() * scale).bfloat16(), w, (k.float() * scale).bfloat16()
    q.zero_()
    k.zero_()
    w.zero_()
    if family == "zeros":
        return q, w, k
    if family == "signed":
        q[:, 0, 0] = 1
        q[:, 1, 1] = 1
        k[:, 0] = (torch.arange(pools) % 19 - 9).bfloat16()
        k[:, 1] = (torch.arange(pools) % 13 - 6).bfloat16()
        w[:, 0] = -1
        w[:, 1] = 0.75
    elif family == "dot_round":
        q[:, 0, :2] = torch.tensor([1, 2**-8], dtype=torch.bfloat16)
        k[:, 0] = 1
        k[:, 1] = (1 + (torch.arange(pools) % 3 - 1) * 2**-7).bfloat16()
        w[:, 0] = 1.003
    elif family == "product_round":
        q[:, :, 0] = 1
        k[:, 0] = 1
        w[:, :4] = torch.tensor([1.00390625, -1, 0, 0])
        w[1::2, 0] = 1.00390637
    elif family in ("head_cancel", "rank_flip"):
        q[:, :, 0] = 1
        k[:, 0] = 1
        w[:, :4] = torch.tensor([2**30, 1, -(2**30), 2])
        # All operands finite and BF16-exact; different FP32 reduction orders
        # can produce different small residuals even after final BF16 rounding.
        if family == "rank_flip":
            if pools < 513:
                raise ValueError("rank_flip requires at least 513 pools")
            q[:, 4, 0] = 0
            q[:, 4, 1] = 1
            w[:, 4] = 1
            k.zero_()
            k[:511, 1] = 4
            k[511, 0] = 1  # CANN=2 (or 0 for one row), Triton=3
            k[512, 1] = 2.5  # strictly between 2 and 3
    elif family == "dot_cancel":
        q[:, 0, :4] = torch.tensor([2**30, 1, -(2**30), 2]).bfloat16()
        k[:, :4] = 1
        w[:, 0] = 1
    elif family == "boundary_tie":
        q[:, 0, 0] = 1
        w[:, 0] = 1
        k[:, 0] = 1
        k[: min(511, pools), 0] = 2
        if pools > 514:
            k[514:, 0] = 0.5
    else:
        raise ValueError(family)
    return q, w, k


def source_receipt(repo):
    paths = [
        REL / f
        for f in (
            "kpool_indexer.py",
            "pooled_score_prefill_npu.py",
            "prefill_index_parallel.py",
        )
    ]
    return {
        str(path): hashlib.sha256((Path(repo) / path).read_bytes()).hexdigest()
        for path in paths
    }
