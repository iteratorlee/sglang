"""Opt-in adapter for the pinned 910B KDA extension (no package replacement).

The public recurrent wrapper owns its existing validation. This adapter only
selects the narrower external ABI. It never copies/reorders persistent state,
reads device metadata on the host, or catches a failed kernel launch.
"""

import functools
import hashlib
import logging
import os
from pathlib import Path

import torch

KERNEL_COMMIT = "f7a4f2d7f1355b39b921962e0d768f568464e07a"
ENABLE_ENV = "SGLANG_GLM53_KDA_VARLEN_ASCENDC"
LIB_ENV = "SGLANG_GLM53_KDA_VARLEN_ASCENDC_LIB"
NAMESPACE = "sglang_glm53_kda_ascendc"
logger = logging.getLogger(__name__)
_logged = set()


def unsupported_reason(
    *,
    q,
    k,
    v,
    a,
    b,
    A_log,
    dt_bias,
    initial_state_source,
    initial_state_indices,
    cu_seqlens,
    prefill,
    intermediate_state=None,
    track_state_indices=None,
    track_state_lens=None,
    **unused,
):
    """Metadata-only eligibility; device index bounds remain caller-owned.

    Slot <= 0 is padding: no state read/write; verify also leaves its scratch
    untouched. Active slots/checkpoint destinations must be valid and distinct
    as in the original wrapper. Each sequence must fit the scratch step axis.
    The external host narrows int64 metadata to int32 on the current stream;
    our shape bounds cover valid indices/offsets, without a device-to-host sync.
    """
    if q.device.type != "npu":
        return "device"
    heads, dim = q.shape[2:]
    tokens = q.shape[1]
    batch = initial_state_indices.numel()
    if dim != 128 or tokens < 1 or batch < 1:
        return "shape"
    shape = (1, tokens, heads, 128)
    if any(t.shape != shape for t in (q, k, v, a)) or b.shape != shape[:-1]:
        return "shape"
    if any(t.dtype != torch.bfloat16 or not t.is_contiguous() for t in (q, k, v, a, b)):
        return "input_dtype_or_layout"
    state = initial_state_source
    if (
        state.ndim != 4
        or state.shape[0] < 1
        or state.shape[1:] != (heads, 128, 128)
        or state.dtype != torch.float32
        or state.stride() != (heads * 16384, 16384, 1, 128)
    ):
        # Device code hardcodes the *outer* strides too, unlike its host check.
        return "state_layout"
    if (
        A_log.numel() != heads
        or dt_bias.numel() != heads * 128
        or any(
            t.dtype != torch.float32 or not t.is_contiguous() for t in (A_log, dt_bias)
        )
    ):
        return "constant_layout"
    metadata = [initial_state_indices, cu_seqlens]
    if (
        initial_state_indices.ndim != 1
        or cu_seqlens.ndim != 1
        or cu_seqlens.numel() != batch + 1
    ):
        return "metadata_shape"
    if (track_state_indices is None) != (track_state_lens is None):
        return "tracking_pair"
    if track_state_indices is not None:
        metadata += [track_state_indices, track_state_lens]
        if (
            not prefill
            or intermediate_state is not None
            or any(
                t.ndim != 1 or t.numel() != batch
                for t in (track_state_indices, track_state_lens)
            )
        ):
            return "tracking_mode"
    if any(
        t.dtype not in (torch.int32, torch.int64) or not t.is_contiguous()
        for t in metadata
    ):
        return "metadata_dtype_or_layout"
    if max(tokens, batch, state.shape[0]) >= 2**31:
        return "metadata_int32_range"
    tensors = [q, k, v, a, b, A_log, dt_bias, state, *metadata]
    mid = intermediate_state
    if prefill:
        if heads not in (4, 8) or batch != 1 or mid is not None:
            return "prefill_shape"
    else:
        if heads not in (16, 32, 64) or tokens < batch:
            return "decode_shape"
    if mid is not None:
        if (
            mid.ndim != 5
            or mid.dtype != torch.float32
            or mid.shape[0] < batch
            or mid.shape[1] < 1
            or mid.shape[2:] != (heads, 128, 128)
            or mid.shape[1] * batch < tokens
        ):
            return "intermediate_shape"
        strides = mid.stride()
        # Head/step/request padding is supported, but overlap is not. Kernel
        # offsets for these axes are uint32; K/V are hardcoded within a tile.
        if (
            strides[3:] != (1, 128)
            or strides[2] < 16384
            or strides[1] < heads * strides[2]
            or strides[0] < mid.shape[1] * strides[1]
            or any(s >= 2**32 or s % 8 for s in strides[:3])
        ):
            return "intermediate_layout"
        tensors.append(mid)
    if any(t.device != q.device for t in tensors):
        return "mixed_devices"
    return None


@functools.lru_cache(maxsize=1)
def _operator(library_path):
    if not library_path:
        raise RuntimeError(f"{ENABLE_ENV}=1 requires an explicit {LIB_ENV}")
    path = Path(library_path).expanduser().resolve(strict=True)
    # This loads only the version-pinned varlen op in a separate namespace.
    # Never import sgl_glm53_ascend_ops: the existing prepared_recurrent_v2
    # package and its PYTHONPATH must remain usable for the baseline/fallback.
    torch.ops.load_library(str(path))
    try:
        ops = torch.ops.sglang_glm53_kda_ascendc
        if ops.build_revision() != KERNEL_COMMIT:
            raise RuntimeError(f"{path} was not built from {KERNEL_COMMIT}")
        operator = ops.kda_varlen_recurrent
    except AttributeError as exc:
        raise RuntimeError(
            f"{path} lacks {NAMESPACE}.kda_varlen_recurrent; build with "
            "scripts/npu/build_glm53_kda_ascendc.py"
        ) from exc
    logger.info(
        "GLM53 AscendC kda_varlen library loaded: path=%s sha256=%s "
        "kernel_commit=%s namespace=%s",
        path,
        hashlib.sha256(path.read_bytes()).hexdigest(),
        KERNEL_COMMIT,
        NAMESPACE,
    )
    return operator


def try_run(**kwargs):
    """Return None for unsupported metadata; errors on the selected op surface."""
    mode = (
        "prefill"
        if kwargs["prefill"]
        else ("verify" if kwargs["intermediate_state"] is not None else "decode")
    )
    reason = unsupported_reason(**kwargs)
    if reason is not None:
        key = ("fallback", mode, reason)
        if key not in _logged:
            _logged.add(key)
            logger.info(
                "GLM53 AscendC kda_varlen fallback: mode=%s reason=%s; "
                "retaining configured original path",
                mode,
                reason,
            )
        return None
    library = os.environ.get(LIB_ENV, "")
    result = _operator(library)(**kwargs)
    q = kwargs["q"]
    state = kwargs["initial_state_source"]
    mid = kwargs["intermediate_state"]
    key = (
        "active",
        mode,
        q.shape[2],
        mid.shape[1] if mid is not None else 0,
        kwargs["track_state_indices"] is not None,
    )
    if key not in _logged:
        _logged.add(key)
        logger.info(
            "GLM53 AscendC kda_varlen dispatched: kernel_commit=%s library=%s "
            "op=%s.kda_varlen_recurrent mode=%s H=%s T=%s batch=%s "
            "state_stride=%s intermediate_shape=%s intermediate_stride=%s "
            "indices_dtype=%s starts_dtype=%s tracking=%s",
            KERNEL_COMMIT,
            library,
            NAMESPACE,
            mode,
            q.shape[2],
            q.shape[1],
            kwargs["initial_state_indices"].numel(),
            tuple(state.stride()),
            tuple(mid.shape) if mid is not None else None,
            tuple(mid.stride()) if mid is not None else None,
            kwargs["initial_state_indices"].dtype,
            kwargs["cu_seqlens"].dtype,
            kwargs["track_state_indices"] is not None,
        )
    return result
