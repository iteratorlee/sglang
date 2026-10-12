#!/usr/bin/env python3
"""All BF16 bit patterns against selected FP32 gates, plus real NPU fallbacks.

This is finite test coverage, not proof for every possible FP32 gate. Nonfinite
payload, signed-zero and finite mismatches are reported separately, never waived.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from score_contract import REL, load_file


def differences(a, b):
    a, b = a.cpu(), b.cpu()
    bad = a.view(torch.int16) != b.view(torch.int16)
    zeros = (a == 0) & (b == 0)
    nans = torch.isnan(a) & torch.isnan(b)
    return dict(
        elements=a.numel(),
        different_bits=int(bad.sum()),
        nan_payload_only=int((bad & nans).sum()),
        zero_sign_only=int((bad & zeros).sum()),
        finite_nonzero=int(
            (bad & torch.isfinite(a) & torch.isfinite(b) & ~zeros).sum()
        ),
        nan_semantic=int((torch.isnan(a) != torch.isnan(b)).sum()),
        examples=[
            dict(
                index=i,
                old_bits=int(a.view(torch.int16)[tuple(i)]),
                new_bits=int(b.view(torch.int16)[tuple(i)]),
            )
            for i in bad.nonzero()[:12].tolist()
        ],
        exact=not bool(bad.any()),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if os.getenv("ASCEND_RT_VISIBLE_DEVICES") != "7":
        p.error("Require authorized private physical device 7")
    import torch_npu

    torch.set_num_threads(1)
    torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)
    torch_npu.npu.config.allow_internal_format = True
    source = a.repo / REL / "pooled_score_postop_npu.py"
    module = load_file(source, "edge_postop")
    # Positive control for the faster on-device counter used by the benchmark:
    # include signed zero, NaN payload and finite one-bit differences.
    bit_a = torch.tensor([0, -32768, 32704, 32641, 16256, -16512], dtype=torch.int16)
    bit_b = torch.tensor([-32768, 0, 32705, 32640, 16257, -16512], dtype=torch.int16)
    device_a = bit_a.to("npu:0").view(torch.bfloat16)
    device_b = bit_b.to("npu:0").view(torch.bfloat16)
    device_count = int((device_a.view(torch.int16) != device_b.view(torch.int16)).sum().item())
    expected_count = int((bit_a != bit_b).sum())
    if device_count != expected_count:
        raise RuntimeError("Device bit counter failed its injected-error control")
    # Both signs: zero, 1, BF16 halfway and its neighbor, min subnormal,
    # max subnormal, min normal, very small/large, max finite, inf, NaN payloads.
    positive = [
        0x00000000,
        0x3F800000,
        0x3F808000,
        0x3F808001,
        0x00000001,
        0x007FFFFF,
        0x00800000,
        0x00800001,
        0x33800000,
        0x4B800000,
        0x7F7FFFFF,
        0x7F800000,
        0x7FC00000,
        0x7F800001,
        0x7FFFFFFF,
        0x3EAAAAAB,
    ]
    unsigned = positive + [x | 0x80000000 for x in positive]
    bits = [x if x < 2**31 else x - 2**32 for x in unsigned]
    weights = (
        torch.tensor(bits, dtype=torch.int32)
        .view(torch.float32)
        .reshape(1, 32)
        .to("npu:0")
    )
    values = torch.arange(-32768, 32768, dtype=torch.int32).short().view(torch.bfloat16)
    scores = values.reshape(1, 1, -1).repeat(1, 32, 1).to("npu:0")
    records = []
    for block in (2048, 4096, 8192):
        x = scores.clone()
        reference = x.clone().relu_().mul_(weights.unsqueeze(-1))
        active = module.relu_gate_(x, weights, block)
        records.append(
            dict(
                case="all_bf16_bits_x_32_fp32_gates",
                block=block,
                active=active,
                **differences(reference, x),
            )
        )
    for case in ("noncontiguous", "format29", "bf16_gates", "grid_limit", "row_limit"):
        generator = torch.Generator().manual_seed(1012)
        shape = (
            (32, 32, 131072)
            if case == "grid_limit"
            else (128, 32, 64) if case == "row_limit" else (2, 32, 64)
        )
        x = torch.randn(shape, generator=generator).bfloat16().to("npu:0")
        w = torch.full(shape[:2], -2.25, device="npu:0", dtype=torch.float32)
        block = 2048 if case == "grid_limit" else 8192
        if case == "noncontiguous":
            x = x[:, :, ::2]
        if case == "format29":
            x = torch_npu.npu_format_cast(x, 29)
            if torch_npu.get_npu_format(x) != 29:
                raise RuntimeError("Requested format29 was not actually produced")
        if case == "bf16_gates":
            w = w.bfloat16()
        reference = x.clone().relu_().mul_(w.unsqueeze(-1))
        fmt_before = torch_npu.get_npu_format(x)
        active = module.relu_gate_(x, w, block)
        records.append(
            dict(
                case=case,
                block=block,
                active=active,
                format_before=fmt_before,
                expected_fallback=True,
                **differences(reference, x),
            )
        )
        del x, w, reference
    result = dict(
        torch=torch.__version__,
        torch_npu=torch_npu.__version__,
        prototype_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        device_bit_counter=dict(expected=expected_count, observed=device_count),
        gate_bits_hex=[f"{x:08x}" for x in unsigned],
        records=records,
    )
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return (
        0
        if all(
            r["exact"] and r["active"] != r.get("expected_fallback", False)
            for r in records
        )
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
