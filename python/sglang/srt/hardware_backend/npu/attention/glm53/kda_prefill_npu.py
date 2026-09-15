"""FP32 KDA input preparation for the validated long TP16 prefill shape.

Normalization and gates are independent across tokens. Compute them once in
parallel, keeping FP32 intermediates and the original reduction/formula order.
Only the state update remains serial. The caller preserves the original path
for decode, speculative state snapshots and other shapes.
"""

import os

import triton
import triton.language as tl


@triton.jit
def _prepare_inputs(
    Q,
    K,
    A,
    B,
    ALOG,
    BIAS,
    QN,
    KN,
    GDECAY,
    BETA,
    scale,
    lower_bound,
    H: tl.constexpr,
    D: tl.constexpr,
):
    row = tl.program_id(0)
    head = row % H
    cols = tl.arange(0, D)
    q = tl.load(Q + row * D + cols).to(tl.float32)
    k = tl.load(K + row * D + cols).to(tl.float32)
    a = tl.load(A + row * D + cols).to(tl.float32)
    b = tl.load(B + row).to(tl.float32)
    alog = tl.load(ALOG + head).to(tl.float32)
    bias = tl.load(BIAS + head * D + cols).to(tl.float32)
    decay = tl.exp(alog)
    x = a + bias
    gate = lower_bound / (1.0 + tl.exp(-(decay * x)))
    beta = 1.0 / (1.0 + tl.exp(-b))
    q = q / tl.sqrt(tl.sum(q * q) + 1e-6)
    k = k / tl.sqrt(tl.sum(k * k) + 1e-6)
    q *= scale
    tl.store(QN + row * D + cols, q)
    tl.store(KN + row * D + cols, k)
    tl.store(GDECAY + row * D + cols, tl.exp(gate))
    tl.store(BETA + row, beta)


@triton.jit
def _prepared_recurrent(
    QN,
    KN,
    V,
    GDECAY,
    BETA,
    OUT,
    STATE,
    INDICES,
    STARTS,
    TRACK_INDICES,
    TRACK_LENS,
    TRACK_STATE: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BV: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    SK: tl.constexpr,
    SV: tl.constexpr,
):
    iv, req, head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    bos = tl.load(STARTS + req).to(tl.int64)
    eos = tl.load(STARTS + req + 1).to(tl.int64)
    ks = tl.arange(0, D)
    vs = iv * BV + tl.arange(0, BV)
    slot = tl.load(INDICES + req)
    pstate = STATE + slot * S0 + head * S1 + ks[None, :] * SK + vs[:, None] * SV
    state = tl.zeros((BV, D), tl.float32)
    if slot > 0:
        state = tl.load(pstate).to(tl.float32)
    if TRACK_STATE:
        track_slot = tl.load(TRACK_INDICES + req)
        track_len = tl.load(TRACK_LENS + req)
        ptrack = (
            STATE + track_slot * S0 + head * S1 + ks[None, :] * SK + vs[:, None] * SV
        )
        if track_slot > 0 and track_len == 0:
            tl.store(ptrack, state)
    for i in range(eos - bos):
        row = (bos + i) * H + head
        q = tl.load(QN + row * D + ks)
        k = tl.load(KN + row * D + ks)
        v = tl.load(V + row * D + vs).to(tl.float32)
        decay = tl.load(GDECAY + row * D + ks)
        beta = tl.load(BETA + row)
        state *= decay[None, :]
        v -= tl.sum(state * k[None, :], axis=1)
        v *= beta
        state += k[None, :] * v[:, None]
        if TRACK_STATE:
            if track_slot > 0 and i + 1 == track_len:
                tl.store(ptrack, state)
        out = tl.sum(state * q[None, :], axis=1)
        tl.store(OUT + row * D + vs, out.to(OUT.dtype.element_ty))
    if slot > 0:
        tl.store(pstate, state)


def run_prepared_prefill(
    *,
    q,
    k,
    v,
    a,
    b,
    A_log,
    dt_bias,
    state,
    indices,
    starts,
    output,
    scale,
    lower_bound,
    track_indices=None,
    track_lens=None,
):
    import torch

    normalized_q = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    normalized_k = torch.empty_like(normalized_q)
    gate_decay = torch.empty_like(normalized_q)
    beta = torch.empty(b.shape, dtype=torch.float32, device=b.device)
    _prepare_inputs[(q.shape[1] * 4,)](
        q,
        k,
        a,
        b,
        A_log,
        dt_bias,
        normalized_q,
        normalized_k,
        gate_decay,
        beta,
        scale,
        lower_bound,
        4,
        128,
        num_warps=1,
        num_stages=3,
        multibuffer=False,
    )
    # BV32 exposes16 independent value/head programs instead of8. Keep the
    # original arithmetic and FP32 state; smaller tiles failed packed oracles.
    value_tile = 32 if os.getenv("SGLANG_GLM53_KDA_PREFILL_BV32", "0") == "1" else 64
    _prepared_recurrent[(128 // value_tile, 1, 4)](
        normalized_q,
        normalized_k,
        v,
        gate_decay,
        beta,
        output,
        state,
        indices,
        starts,
        track_indices if track_indices is not None else indices,
        track_lens if track_lens is not None else starts,
        track_indices is not None,
        4,
        128,
        value_tile,
        *state.stride(),
        num_warps=1,
        num_stages=3,
        multibuffer=False,
    )
    return output
