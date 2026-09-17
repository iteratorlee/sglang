"""Ascend L1/L2 transfers for heterogeneous Mamba state components."""

from __future__ import annotations

from collections.abc import Sequence
from math import prod
from typing import Literal

import torch


MambaTransferDirection = Literal["h2d", "d2h"]


def _validate_indices(device_indices: torch.Tensor, host_indices: torch.Tensor) -> None:
    if device_indices.ndim != 1 or host_indices.ndim != 1:
        raise ValueError("Mamba state transfer indices must be one-dimensional")
    if device_indices.numel() != host_indices.numel():
        raise ValueError(
            "Mamba state transfer index counts differ: "
            f"device={device_indices.numel()}, host={host_indices.numel()}"
        )
    integer_dtypes = (torch.int32, torch.int64)
    if device_indices.dtype not in integer_dtypes:
        raise TypeError(
            f"device_indices must use int32 or int64, got {device_indices.dtype}"
        )
    if host_indices.dtype not in integer_dtypes:
        raise TypeError(
            f"host_indices must use int32 or int64, got {host_indices.dtype}"
        )
    if not device_indices.is_contiguous() or not host_indices.is_contiguous():
        raise ValueError("Mamba state transfer indices must be contiguous")


def _validate_component_layout(
    device_tensor: torch.Tensor, host_tensor: torch.Tensor
) -> None:
    """Validate the layout contract without touching tensor contents.

    Kept separate from the NPU/pinned checks so the geometry and dtype contract
    can be covered by CPU-only unit tests.
    """
    if device_tensor.ndim < 2:
        raise ValueError(
            "Device Mamba state must be [layers, slots, *state_shape], got "
            f"shape={tuple(device_tensor.shape)}"
        )
    if host_tensor.ndim != device_tensor.ndim + 1:
        raise ValueError(
            "Host Mamba state must be [slots, layers, 1, *state_shape], got "
            f"device_shape={tuple(device_tensor.shape)}, "
            f"host_shape={tuple(host_tensor.shape)}"
        )
    if host_tensor.shape[1] != device_tensor.shape[0]:
        raise ValueError(
            "Mamba state layer counts differ: "
            f"device={device_tensor.shape[0]}, host={host_tensor.shape[1]}"
        )
    if host_tensor.shape[2] != 1:
        raise ValueError(
            f"Mamba host state page size must be 1, got {host_tensor.shape[2]}"
        )
    if tuple(host_tensor.shape[3:]) != tuple(device_tensor.shape[2:]):
        raise ValueError(
            "Mamba state payload shapes differ: "
            f"device={tuple(device_tensor.shape[2:])}, "
            f"host={tuple(host_tensor.shape[3:])}"
        )
    if host_tensor.dtype != device_tensor.dtype:
        raise TypeError(
            "Mamba state transfer does not convert dtype: "
            f"device={device_tensor.dtype}, host={host_tensor.dtype}"
        )
    if not host_tensor.is_contiguous():
        raise ValueError("Mamba state transfer requires contiguous host buffers")
    # Ascend MTP exposes recurrent state with its last two axes transposed.
    # The payload is still one dense physical span per slot. SDMA transfers
    # opaque state bytes; it must neither transpose them nor materialize a
    # contiguous copy of the entire device pool.
    payload = prod(device_tensor.shape[2:])
    if (
        device_tensor.stride(1) != payload
        or device_tensor.stride(0) != device_tensor.shape[1] * payload
    ):
        raise ValueError("Mamba state transfer requires packed layer/slot strides")
    span = 1
    for stride, size in sorted(
        (stride, size)
        for stride, size in zip(device_tensor.stride()[2:], device_tensor.shape[2:])
        if size > 1
    ):
        if stride != span:
            raise ValueError(
                "Mamba state payload must be dense without overlapping or gapped axes"
            )
        span *= size


def _physical_transfer_views(device_tensor, host_tensor):
    """Expose dense state bytes in native transfer order, without allocation.

    Host state is an opaque payload. The same physical order is serialized to
    storage and restored into the original device view on H2D.
    """
    if device_tensor.is_contiguous():
        return device_tensor, host_tensor
    layers, slots = device_tensor.shape[:2]
    payload = prod(device_tensor.shape[2:])
    return (
        device_tensor.as_strided(
            (layers, slots, payload), (slots * payload, payload, 1)
        ),
        host_tensor.view(host_tensor.shape[0], layers, 1, payload),
    )


def _native_transfer_objects():
    # Import lazily so CPU-only SGLang imports never require torch_npu or the
    # Ascend extension package.
    from sgl_kernel_npu.kvcacheio import TransferDirection, transfer_mamba_state

    return TransferDirection, transfer_mamba_state


def _validate_runtime_component(
    device_tensor: torch.Tensor, host_tensor: torch.Tensor
) -> None:
    if device_tensor.device.type != "npu":
        raise ValueError(
            "Ascend Mamba state transfer requires an NPU device tensor, "
            f"got {device_tensor.device}"
        )
    if host_tensor.device.type != "cpu":
        raise ValueError(
            "Ascend Mamba state transfer requires a CPU host tensor, "
            f"got {host_tensor.device}"
        )
    if not host_tensor.is_pinned():
        raise RuntimeError(
            "Ascend Mamba state transfer requires pinned host memory for "
            "nonblocking aclrtMemcpy2dAsync"
        )


def transfer_mamba_state_components(
    *,
    device_tensors: Sequence[torch.Tensor],
    host_tensors: Sequence[torch.Tensor],
    device_indices: torch.Tensor,
    host_indices: torch.Tensor,
    direction: MambaTransferDirection,
) -> None:
    """Transfer all non-empty Mamba components on the current NPU stream.

    Each component is submitted separately to preserve its native dtype. The
    GLM KDA recurrent state therefore remains FP32 while conv state remains in
    its configured activation dtype. ``transfer_mamba_state`` uses
    ``aclrtMemcpy2dAsync`` and the surrounding L2TransferEngine records the
    visibility event on this same stream.
    """
    if direction not in ("h2d", "d2h"):
        raise ValueError(f"Unsupported Mamba transfer direction: {direction!r}")
    if len(device_tensors) != len(host_tensors):
        raise ValueError(
            "Mamba component counts differ: "
            f"device={len(device_tensors)}, host={len(host_tensors)}"
        )
    _validate_indices(device_indices, host_indices)
    if device_indices.numel() == 0:
        return

    pairs = [
        (device_tensor, host_tensor)
        for device_tensor, host_tensor in zip(device_tensors, host_tensors)
        if device_tensor.numel() > 0 or host_tensor.numel() > 0
    ]
    for device_tensor, host_tensor in pairs:
        if device_tensor.numel() == 0 or host_tensor.numel() == 0:
            raise ValueError(
                "Mamba state component is empty on only one side: "
                f"device_numel={device_tensor.numel()}, "
                f"host_numel={host_tensor.numel()}"
            )
        _validate_component_layout(device_tensor, host_tensor)
        _validate_runtime_component(device_tensor, host_tensor)

    TransferDirection, native_transfer = _native_transfer_objects()
    native_direction = (
        TransferDirection.H2D if direction == "h2d" else TransferDirection.D2H
    )
    for device_tensor, host_tensor in pairs:
        device_tensor, host_tensor = _physical_transfer_views(
            device_tensor, host_tensor
        )
        native_transfer(
            device_tensor,
            host_tensor,
            device_indices,
            host_indices,
            native_direction,
        )

    # Device index tensors can be created on a producer stream. Keep their
    # storage live until the asynchronous copies on the current stream retire.
    npu_indices = [
        indices
        for indices in (device_indices, host_indices)
        if indices.device.type == "npu"
    ]
    if npu_indices:
        stream = torch.npu.current_stream(device_tensors[0].device)
        for indices in npu_indices:
            indices.record_stream(stream)
