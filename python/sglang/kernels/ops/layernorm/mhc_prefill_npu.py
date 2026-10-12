"""Opt-in GLM large-prefill post mixing; separate from the existing D kernel."""

import triton
import triton.language as tl


@triton.jit
def _hc_post_apply_large(
    X, RES, POST, COMB, OUT, H: tl.constexpr, M: tl.constexpr, B: tl.constexpr
):
    tl.static_assert(M == 4)
    t = tl.program_id(0)
    h = tl.program_id(1) * B + tl.arange(0, B)
    x = tl.load(X + t * H + h, h < H, 0).to(tl.float32)
    r0 = tl.load(RES + t * 4 * H + h, h < H, 0).to(tl.float32)
    r1 = tl.load(RES + t * 4 * H + H + h, h < H, 0).to(tl.float32)
    r2 = tl.load(RES + t * 4 * H + 2 * H + h, h < H, 0).to(tl.float32)
    r3 = tl.load(RES + t * 4 * H + 3 * H + h, h < H, 0).to(tl.float32)
    for j in tl.static_range(4):
        c0 = tl.load(COMB + t * 16 + j)
        c1 = tl.load(COMB + t * 16 + 4 + j)
        c2 = tl.load(COMB + t * 16 + 8 + j)
        c3 = tl.load(COMB + t * 16 + 12 + j)
        post = tl.load(POST + t * 4 + j)
        p0, p1, p2, p3 = r0 * c0, r1 * c1, r2 * c2, r3 * c3
        # Full-T native ReduceSum matched this sequential FP32 tree in V2.
        # Keep separate rounded products/adds; no FP fusion or BF16 intermediate.
        mixed = ((p0 + p1) + p2) + p3
        mixed = tl.where(mixed == 0.0, 0.0, mixed)
        out = post * x + mixed
        tl.store(OUT + t * 4 * H + j * H + h, out, h < H)


def hc_post_large_prefill(x, residual, h_post, h_res, hc_mult):
    """Caller guards P/T>4096; direct unit tests also exercise T1 and T4096.

    Requires H4096, four streams, contiguous physical ND, BF16 x/residual,
    FP32 coefficients, and one device. The existing small-batch helper is separate.
    """
    tokens, hidden = x.shape
    assert hidden == 4096 and hc_mult == 4 and 0 <= tokens <= 16384
    out = x.new_empty((tokens, hc_mult * hidden))
    if tokens:
        _hc_post_apply_large[(tokens, triton.cdiv(hidden, 4096))](
            x,
            residual,
            h_post,
            h_res,
            out,
            H=hidden,
            M=hc_mult,
            B=4096,
            num_warps=1,
            enable_fp_fusion=False,
            enable_auto_bind_sub_block=False,
        )
    return out
