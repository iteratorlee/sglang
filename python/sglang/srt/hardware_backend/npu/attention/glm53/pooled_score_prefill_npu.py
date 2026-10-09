"""Experimental compact GLM-5.3 prefill pooled scores for Ascend.

One program computes four queries' 32 heads against 64 pooled keys, applies
ReLU and the signed FP32 head gates, then stores only the reduced BF16 scores.
The caller retains its 128-position candidate and per-row causal boundaries,
top-k, and invalid-id padding. This module is imported only with the explicit
prefill experiment switch; Triton-Ascend 3.2 compilation needs NPU validation.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _score_pool_tile(
    Q, W, K, OUT, ROWS: tl.constexpr, N: tl.constexpr, BK: tl.constexpr, BQ: tl.constexpr
):
    query_block = tl.program_id(0)
    key_block = tl.program_id(1)
    query_heads = tl.arange(0, BQ * 32)
    rows = query_block * BQ + query_heads // 32
    dims = tl.arange(0, 128)
    keys = key_block * BK + tl.arange(0, BK)

    q = tl.load(
        Q + (query_block * BQ * 32 + query_heads[:, None]) * 128 + dims[None, :],
        mask=rows[:, None] < ROWS,
        other=0,
    )
    k = tl.load(
        K + keys[None, :] * 128 + dims[:, None],
        mask=keys[None, :] < N,
        other=0,
    )
    # einsum returns BF16. Its in-place product with FP32 gates rounds to
    # BF16 again before the head reduction, whose output is also BF16.
    dot = tl.dot(q, k).to(tl.bfloat16).to(tl.float32)
    gates = tl.load(
        W + query_block * BQ * 32 + query_heads,
        mask=rows < ROWS,
        other=0,
    ).to(tl.float32)
    weighted = (tl.maximum(dot, 0.0) * gates[:, None]).to(tl.bfloat16)
    scores = tl.sum(tl.reshape(weighted.to(tl.float32), (BQ, 32, BK)), axis=1).to(
        tl.bfloat16
    )
    output_rows = query_block * BQ + tl.arange(0, BQ)
    tl.store(
        OUT + output_rows[:, None] * N + keys[None, :],
        scores,
        mask=(output_rows[:, None] < ROWS) & (keys[None, :] < N),
    )


def supported(
    q: torch.Tensor, weights: torch.Tensor, histories: list[torch.Tensor]
) -> bool:
    """Restrict the experiment to the actual 910B NPU prefill tensor contract."""
    return (
        q.device.type == "npu"
        and q.dtype == torch.bfloat16
        and q.ndim == 3
        and q.shape[1:] == (32, 128)
        and q.is_contiguous()
        and weights.device == q.device
        and weights.dtype == torch.float32
        and weights.ndim == 2
        and weights.shape[0] == q.shape[0]
        and weights.shape[1] == 32
        and weights.is_contiguous()
        and all(
            key.device == q.device
            and key.dtype == torch.bfloat16
            and key.ndim == 2
            and key.shape[1] == 128
            and key.is_contiguous()
            for key in histories
        )
    )


def fused_pooled_scores(
    q: torch.Tensor, weights: torch.Tensor, keys: torch.Tensor
) -> torch.Tensor:
    """Return [queries, closed pools] BF16 scores without [Q,H,K] storage."""
    if not supported(q, weights, [keys]):
        raise ValueError("Unsupported GLM-5.3 fused pooled-score tensor contract")
    rows, count = q.shape[0], keys.shape[0]
    scores = torch.empty((rows, count), device=q.device, dtype=torch.bfloat16)
    if rows and count:
        _score_pool_tile[(triton.cdiv(rows, 4), triton.cdiv(count, 64))](
            q, weights, keys, scores, rows, count, 64, 4, num_warps=4, multibuffer=False
        )
    return scores
