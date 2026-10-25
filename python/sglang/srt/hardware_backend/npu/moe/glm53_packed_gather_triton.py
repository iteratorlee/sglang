"""Capacity-bucketed integer loads/stores; actual rows/strides stay runtime.

No CPU JIT execution. Compilation and correctness on the installed NPU stack
are mandatory before enabling. Launch options match existing Ascend kernels.
"""

import inspect

import triton
import triton.language as tl


def _runtime_jit(names):
    # The deployed 3.2-module/3.5-wheel API uses do_not_specialize. Newer local
    # Triton APIs split alignment into another option; disable it too when
    # exposed. No probing, compilation, or device operation occurs here.
    options = {"do_not_specialize": names}
    if "do_not_specialize_on_alignment" in inspect.signature(triton.jit).parameters:
        options["do_not_specialize_on_alignment"] = names
    return triton.jit(**options)


# Request suppression of scalar value/alignment and pointer specialization. The
# tensor pointer dtype, capacity layout, ID width, and raw/byte mode still
# distinguish bounded kernels. No actual rows/strides are constexpr arguments.
@_runtime_jit(["Q", "S", "I", "W", "OUT", "ROWS", "Q0", "Q1", "S0", "I0", "I1", "W0", "W1"])
def _pack(Q, S, I, W, OUT, ROWS, ID_WORDS: tl.constexpr,
          S_OFF: tl.constexpr, I_OFF: tl.constexpr, W_OFF: tl.constexpr,
          WORDS: tl.constexpr, Q0, Q1,
          S0, I0, I1, W0, W1, Q_RAW32: tl.constexpr,
          BLOCK: tl.constexpr):
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    qm = lane < ROWS * 1024
    qr, qc = lane // 1024, (lane % 1024) * 4
    if Q_RAW32:
        q = tl.load(Q + qr * (Q0 // 4) + qc // 4, qm, other=0)
    else:
        # Strided/misaligned byte sources: assemble the same little-endian word
        # entirely in integer registers. No .contiguous() staging kernels.
        ptr = Q + qr * Q0 + qc * Q1
        a = tl.load(ptr, qm, other=0).to(tl.uint32)
        b = tl.load(ptr + Q1, qm, other=0).to(tl.uint32)
        c = tl.load(ptr + 2 * Q1, qm, other=0).to(tl.uint32)
        d = tl.load(ptr + 3 * Q1, qm, other=0).to(tl.uint32)
        q = (a | (b << 8) | (c << 16) | (d << 24)).to(tl.int32, bitcast=True)
    si = tl.maximum(lane - S_OFF, 0)
    sm = (lane >= S_OFF) & (lane < S_OFF + ROWS)
    s = tl.load(S + si * S0, sm, other=0)
    ii = tl.maximum(lane - I_OFF, 0)
    im = (lane >= I_OFF) & (lane < I_OFF + ROWS * 8 * ID_WORDS)
    ir, ic, half = ii // (8 * ID_WORDS), (ii // ID_WORDS) % 8, ii % ID_WORDS
    ids = tl.load(I + (ir * I0 + ic * I1) * ID_WORDS + half, im, other=0)
    wi = tl.maximum(lane - W_OFF, 0)
    wm = (lane >= W_OFF) & (lane < W_OFF + ROWS * 8)
    w = tl.load(W + (wi // 8) * W0 + (wi % 8) * W1, wm, other=0)
    bits = tl.where(qm, q, tl.where(sm, s, tl.where(im, ids, tl.where(wm, w, 0))))
    tl.store(OUT + lane, bits, lane < WORDS)


@_runtime_jit(["IN", "Q", "S", "I", "W", "ROWS"])
def _unpack(IN, Q, S, I, W, ROWS, ID_WORDS: tl.constexpr,
            S_OFF: tl.constexpr, I_OFF: tl.constexpr, W_OFF: tl.constexpr,
            WORDS: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    rank = tl.program_id(1)
    bits = tl.load(IN + rank * WORDS + lane, lane < WORDS, other=0)
    tl.store(Q + rank * ROWS * 1024 + lane,
             bits, lane < ROWS * 1024)
    tl.store(S + rank * ROWS + tl.maximum(lane - S_OFF, 0),
             bits, (lane >= S_OFF) & (lane < S_OFF + ROWS))
    tl.store(I + rank * ROWS * 8 * ID_WORDS + tl.maximum(lane - I_OFF, 0),
             bits, (lane >= I_OFF) & (lane < I_OFF + ROWS * 8 * ID_WORDS))
    tl.store(W + rank * ROWS * 8 + tl.maximum(lane - W_OFF, 0),
             bits, (lane >= W_OFF) & (lane < W_OFF + ROWS * 8))


def pack(q, scale, ids, weights, out, layout):
    raw32 = q.stride(1) == 1 and q.stride(0) % 4 == 0 and q.storage_offset() % 4 == 0
    # Ascend's lowering cannot bitcast memref<i8/i64> to memref<i32>.
    # Declare raw pointer types at the launch boundary, without tensor copies
    # or numeric conversions. Explicit runtime strides remain in source units.
    # Offsets/WORDS/grid use capacity; masks/output rank strides use real ROWS.
    block = 2048 if raw32 else 1024
    _pack[(triton.cdiv(layout.words, block),)](
        triton.reinterpret(q, tl.int32 if raw32 else tl.uint8),
        triton.reinterpret(scale, tl.int32), triton.reinterpret(ids, tl.int32),
        triton.reinterpret(weights, tl.int32), out, layout.rows, layout.id_words,
        *layout.offsets[1:], layout.words, *q.stride(), scale.stride(0),
        *ids.stride(), *weights.stride(),
        raw32,
        block, num_warps=4, multibuffer=False, enable_fp_fusion=False,
    )


def unpack(gathered, result, layout, world):
    block = 2048
    _unpack[(triton.cdiv(layout.words, block), world)](
        gathered, *(triton.reinterpret(t, tl.int32) for t in result),
        layout.rows, layout.id_words,
        *layout.offsets[1:], layout.words, block,
        num_warps=4, multibuffer=False, enable_fp_fusion=False,
    )
