"""Validate and optionally record the loaded GLM ModelSlim execution methods."""

import json
import logging
import os
from collections import Counter
from pathlib import Path

import torch


def validate_w8a8_model(model):
    from sglang.srt.hardware_backend.npu.quantization.w8a8_clamped_moe import (
        NPUW8A8ClampedMoEMethod,
    )
    from sglang.srt.layers.quantization.modelslim.schemes.modelslim_w8a8_int8 import (
        ModelSlimW8A8Int8,
    )

    rows = []
    for name, layer in model.named_modules():
        scheme = getattr(layer, "scheme", None)
        if isinstance(scheme, ModelSlimW8A8Int8):
            if layer.weight.dtype != torch.int8:
                raise ValueError(
                    f"{name}: W8A8 linear weight became {layer.weight.dtype}"
                )
            rows.append(
                dict(
                    name=name,
                    kind="linear",
                    weight_dtype=str(layer.weight.dtype),
                    kernel=type(scheme.kernel).__name__,
                )
            )
        for prefix in ("w13", "w2"):
            kernel = getattr(layer, prefix + "_kernel", None)
            if kernel is None:
                continue
            if not isinstance(kernel, NPUW8A8ClampedMoEMethod):
                raise ValueError(
                    f"{name}.{prefix}: GLM ModelSlim requires the clamped W8A8 expert method"
                )
            weight = getattr(layer, prefix + "_weight")
            scale = getattr(layer, prefix + "_weight_scale")
            if weight.dtype != torch.int8 or scale.dtype != torch.float32:
                raise ValueError(
                    f"{name}.{prefix}: invalid W8A8 weight/FP32 scale dtype"
                )
            rows.append(
                dict(
                    name=name + "." + prefix,
                    kind="expert",
                    weight_dtype=str(weight.dtype),
                    scale_dtype=str(scale.dtype),
                    kernel=type(kernel).__name__,
                )
            )
    counts = dict(Counter(row["kind"] for row in rows))
    if not counts.get("linear") or not counts.get("expert"):
        raise ValueError(
            f"GLM ModelSlim loaded no W8A8 linear/expert modules: {counts}"
        )
    logging.getLogger(__name__).info("GLM native W8A8 validation: %s", counts)
    if folder := os.getenv("SGLANG_GLM53_AUDIT_DIR"):
        path = Path(folder)
        path.mkdir(parents=True, exist_ok=True)
        (path / f"{type(model).__name__}-{os.getpid()}.json").write_text(
            json.dumps(
                dict(model=type(model).__name__, counts=counts, modules=rows), indent=2
            )
        )
