"""Ascend GLM53 HiCache: logical 256-token pages over physical page64.

Each logical page owns four arbitrary physical KV pages. The compressed
index rows are anchored in the first physical page, so the four mappings
must travel together through device eviction, host eviction and storage.
The host buffers retain physical page64 views for the existing SDMA op;
the allocator and file representation expose indivisible logical pages.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os

import torch

from sglang.srt.mem_cache.pool_host.base import HostKVCache
from sglang.srt.mem_cache.pool_host.common import ALLOC_MEMORY_FUNCS

logger = logging.getLogger(__name__)
_npu_host_cache_trimmed = False


def _trim_npu_host_cache_before_hicache(device) -> bool:
    """Release unused weight-loading staging blocks once, before eager P L2.

    TorchNPU caches pinned CPU allocations separately from device allocations.
    Synchronizing before clearing that cache preserves live tensors and pending
    copies. This is a startup operation, never a per-request operation.
    """
    global _npu_host_cache_trimmed
    if (
        _npu_host_cache_trimmed
        or os.environ.get("SGLANG_GLM53_HICACHE_TRIM_HOST_CACHE") != "1"
        or str(device).split(":", 1)[0] != "npu"
    ):
        return False

    from sglang.srt.server_args import get_server_args

    args = get_server_args()
    if args.disaggregation_mode != "prefill" or not args.disable_cuda_graph:
        return False

    import gc

    import torch_npu

    torch_npu.npu.synchronize()
    gc.collect()
    before = dict(torch_npu.npu.host_memory_stats())
    torch_npu.npu.host_empty_cache()
    after = dict(torch_npu.npu.host_memory_stats())
    _npu_host_cache_trimmed = True
    logger.info(
        "GLM53 startup pinned-host cache trim: allocated_bytes=%s -> %s, "
        "active_bytes=%s -> %s, released_bytes=%s",
        before.get("allocated_bytes.current"),
        after.get("allocated_bytes.current"),
        before.get("active_bytes.current"),
        after.get("active_bytes.current"),
        before.get("allocated_bytes.current", 0)
        - after.get("allocated_bytes.current", 0),
    )
    return True


def is_glm53_kpool(pool) -> bool:
    return (
        str(pool.device).split(":", 1)[0] == "npu"
        and getattr(pool, "kpool_use_compress", False)
        and getattr(pool, "index_kpool", None) == 4
        and (
            getattr(pool, "zero_rope", False)
            or (
                getattr(pool, "dsa_kv_cache_store_int8", False)
                and pool.qk_rope_head_dim == 0
                and pool.kv_cache_dim == 656
            )
        )
        and pool.page_size == 64
    )


class GLM53MLAPoolHost(HostKVCache):
    """Cache target and packed MTP pools, including their index-key buffers.

    Zero-RoPE lanes are immutable zeros on device and need no host/storage
    space. Index pages not used as compressed anchors are copied as well:
    this preserves the physical mapping without depending on page adjacency.
    """

    def __init__(
        self,
        device_pool,
        host_to_device_ratio,
        host_size,
        page_size,
        layout,
        pin_memory=True,
        device="cpu",
        allocator_type="default",
        *,
        mtp_draft_device_pools=(),
        pool_label="kv",
        **kwargs,
    ):
        if layout != "page_first_kv_split" or page_size % 256:
            raise ValueError(
                "GLM53 HiCache needs page_first_kv_split and a 256-token cache grid"
            )
        if kwargs.get("dcp_size", 1) != 1:
            raise ValueError("GLM53 HiCache does not support DCP")
        self.mtp_draft_device_pools = tuple(mtp_draft_device_pools)
        self.device_pools = (device_pool, *self.mtp_draft_device_pools)
        self.dsa_kv_cache_store_int8 = any(
            getattr(pool, "dsa_kv_cache_store_int8", False)
            for pool in self.device_pools
        )
        self.dsa_kv_cache_store_quantized = self.dsa_kv_cache_store_int8
        for pool in self.device_pools:
            if (
                not is_glm53_kpool(pool)
                or getattr(pool, "_glm53_index_layout", None) is not None
            ):
                raise ValueError(
                    "GLM53 HiCache requires native, uncompressed physical page64 allocation"
                )
            if getattr(pool, "layer_shard_enabled", False):
                raise ValueError("GLM53 HiCache layer sharding is not implemented")
        self.physical_page_size = 64
        self._components = []
        self._pool_buffers = {}
        super().__init__(
            device_pool,
            host_to_device_ratio,
            host_size,
            page_size,
            layout,
            pin_memory,
            device,
            allocator_type,
            pool_label=pool_label,
        )

    def get_size_per_token(self):
        self.target_layer_num = self.device_pool.layer_num
        self.layer_num = sum(p.layer_num for p in self.device_pools)
        return sum(
            component.shape[0] * component.shape[-1] * component.element_size()
            for p in self.device_pools
            for component in (p.k_buffer, p.index_k_buffer)
        )

    def get_storage_namespace(self):
        if not self.dsa_kv_cache_store_int8:
            return ""
        descriptor = {
            "version": 1,
            "format": "glm53-kpool-logical256-native-c8-656",
            "page_size": self.page_size,
            "components": [
                (str(buf.dtype), list(buf.shape[1:])) for buf in self._components
            ],
            "indexer_layers": [
                list(getattr(p, "indexer_layer_ids", ())) for p in self.device_pools
            ],
        }
        digest = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()
        return f"glm53_c8_v1_{digest[:32]}"

    def init_kv_buffer(self):
        if self.pin_memory and self.dsa_kv_cache_store_int8:
            _trim_npu_host_cache_before_hicache(self.device_pool.device)
        alloc = ALLOC_MEMORY_FUNCS[self.device_pool.device]
        subpages = self.page_size // self.physical_page_size
        for pool in self.device_pools:
            buffers = []
            for component in (pool.k_buffer, pool.index_k_buffer):
                # Physical transfer view: [host physical page, layer, 64, 1, width].
                # Storage view: [logical page, subpage, layer, 64, 1, width].
                host = alloc(
                    (
                        self.page_num,
                        subpages,
                        component.shape[0],
                        64,
                        1,
                        component.shape[-1],
                    ),
                    dtype=component.dtype,
                    device=self.device,
                    pin_memory=self.pin_memory,
                    allocator=self.allocator,
                )
                self._components.append(host)
                buffers.append(host.flatten(0, 1))
            self._pool_buffers[id(pool)] = tuple(buffers)
        self.page_bytes = sum(b[0].numel() * b.element_size() for b in self._components)
        return self._components[0]

    def _transfer(self, pool, host_indices, device_indices, direction):
        from sgl_kernel_npu.kvcacheio import transfer_kv_dim_exchange

        if (
            host_indices.numel() != device_indices.numel()
            or host_indices.numel() % self.page_size
        ):
            raise ValueError(
                "GLM53 HiCache transfers must contain complete logical pages"
            )
        if not host_indices.numel():
            return
        # kernel_ascend's controller supplies CPU indices. The transfer op
        # accepts arbitrary physical pages; no contiguous-device assumption.
        host_indices = host_indices.cpu().contiguous()
        device_indices = device_indices.cpu().contiguous()
        host_k, host_index = self._pool_buffers[id(pool)]
        empty = torch.empty(0)
        transfer_kv_dim_exchange(
            device_indices=device_indices,
            host_indices=host_indices,
            device_k=pool.k_buffer,
            host_k=host_k,
            device_v=empty,
            host_v=empty,
            device_index_k=pool.index_k_buffer,
            host_index_k=host_index,
            page_size=self.physical_page_size,
            direction=direction,
        )

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft=False,
    ):
        if io_backend != "kernel_ascend":
            raise ValueError("GLM53 HiCache requires kernel_ascend IO")
        # The native SDMA operation copies all layers in this pool. Complete
        # it before the first owned attention layer's controller event.
        if not is_draft and layer_id != 0:
            return
        from sgl_kernel_npu.kvcacheio import TransferDirection

        self._transfer(device_pool, host_indices, device_indices, TransferDirection.H2D)

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        if io_backend != "kernel_ascend":
            raise ValueError("GLM53 HiCache requires kernel_ascend IO")
        from sgl_kernel_npu.kvcacheio import TransferDirection

        for pool in self.device_pools:
            self._transfer(pool, host_indices, device_indices, TransferDirection.D2H)

    def _page(self, index):
        index = int(index)
        if index % self.page_size or not 0 <= index < self.size:
            raise ValueError("unaligned/out-of-range GLM53 host page")
        return index // self.page_size

    def get_data_page(self, index, flat=True):
        page = self._page(index)
        return torch.cat(
            [b[page].reshape(-1).view(torch.uint8) for b in self._components]
        )

    def get_dummy_flat_data_page(self):
        return torch.empty(self.page_bytes, dtype=torch.uint8, device="cpu")

    def set_from_flat_data_page(self, index, data_page):
        page = self._page(index)
        data_page = data_page.reshape(-1).view(torch.uint8)
        if data_page.numel() != self.page_bytes:
            raise ValueError("GLM53 storage page has incompatible size")
        offset = 0
        for buf in self._components:
            dst = buf[page].reshape(-1).view(torch.uint8)
            dst.copy_(data_page[offset : offset + dst.numel()])
            offset += dst.numel()

    def get_page_buffer_meta(self, indices):
        if indices.numel() % self.page_size:
            raise ValueError("incomplete GLM53 logical page")
        pointers, sizes = [], []
        for index in indices[:: self.page_size].tolist():
            page = self._page(index)
            for buf in self._components:
                pointers.append(buf[page].data_ptr())
                sizes.append(buf[page].numel() * buf.element_size())
        return pointers, sizes

    def destroy(self):
        super().destroy()
        self._pool_buffers.clear()
        self._components.clear()
