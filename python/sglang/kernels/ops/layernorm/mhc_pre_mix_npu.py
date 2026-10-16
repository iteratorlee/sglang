"""Experimental P-only large mHC pre mixing, independent of D graph kernels.

Keep the full-T native FP32 RMS/projection/Sinkhorn in ``_mhc_pre_torch``.
Only avoid materializing its [T,4,4096] FP32 product and [T,4096] FP32 sum.
Large-T native reduction must be validated on the deployment's CANN version;
the small existing NPU kernel uses a different reduction tree.
"""

import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def _hc_pre_mix_large(X, PRE, OUT, H: tl.constexpr, B: tl.constexpr):
    token = tl.program_id(0)
    h = tl.program_id(1) * B + tl.arange(0, B)
    r0 = tl.load(X + token * 4 * H + h, h < H, 0).to(tl.float32)
    r1 = tl.load(X + token * 4 * H + H + h, h < H, 0).to(tl.float32)
    r2 = tl.load(X + token * 4 * H + 2 * H + h, h < H, 0).to(tl.float32)
    r3 = tl.load(X + token * 4 * H + 3 * H + h, h < H, 0).to(tl.float32)
    p0 = r0 * tl.load(PRE + token * 4)
    p1 = r1 * tl.load(PRE + token * 4 + 1)
    p2 = r2 * tl.load(PRE + token * 4 + 2)
    p3 = r3 * tl.load(PRE + token * 4 + 3)
    mixed = ((p0 + p1) + p2) + p3
    # Native reduction starts from +0. Preserve its zero sign before BF16.
    mixed = tl.where(mixed == 0.0, 0.0, mixed)
    tl.store(OUT + token * H + h, mixed, h < H)


def hc_pre_mix_large_prefill(residual, pre):
    """[T,4,4096] BF16 and [T,4] FP32 -> [T,4096] BF16.

    Caller validates role, range, device and physical ND. Direct numerical
    tests also exercise boundary sizes; no full mHC import/server is needed.
    """
    tokens, mult, hidden = residual.shape
    assert mult == 4 and hidden == 4096 and 0 <= tokens <= 16384
    assert residual.dtype == torch.bfloat16 and pre.dtype == torch.float32
    assert residual.device.type == "npu" and pre.device == residual.device
    assert pre.shape == (tokens, 4)
    assert residual.is_contiguous() and pre.is_contiguous()
    assert torch_npu.get_npu_format(residual) == torch_npu.get_npu_format(pre) == 2
    output = residual.new_empty((tokens, hidden))
    if tokens:
        _hc_pre_mix_large[(tokens, 1)](
            residual,
            pre,
            output,
            H=hidden,
            B=4096,
            num_warps=1,
            enable_fp_fusion=False,
            enable_auto_bind_sub_block=False,
        )
    return output
