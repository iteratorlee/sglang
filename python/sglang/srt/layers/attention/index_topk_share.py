from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator, Optional

from sglang.srt.layers.cp.utils import cp_gather_after_forward, is_cp_active

if TYPE_CHECKING:
    import torch

    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class IndexTopKShareState:
    def __init__(
        self,
        forward_batch: ForwardBatch,
        topk_indices: Optional[torch.Tensor],
    ):
        self._forward_batch = forward_batch
        self._topk_indices = topk_indices

    @classmethod
    def from_mtp_carry(cls, forward_batch: ForwardBatch) -> IndexTopKShareState:
        topk_indices = (
            forward_batch.spec_info.dsa_topk_indices
            if forward_batch.reuse_dsa_topk_indices
            else None
        )
        return cls(forward_batch, topk_indices)

    @property
    def enabled(self) -> bool:
        return bool(self._forward_batch.reuse_dsa_topk_indices)

    @property
    def _seed_buf(self) -> Optional[torch.Tensor]:
        spec_info = self._forward_batch.spec_info
        if spec_info is not None and self._forward_batch.forward_mode.is_extend(
            include_draft_extend_v2=True
        ):
            return spec_info.dsa_seed_topk_capture
        return None

    @property
    def should_publish(self) -> bool:
        return self.enabled or self._seed_buf is not None

    @property
    def topk_indices(self) -> Optional[torch.Tensor]:
        return self._topk_indices

    def update(self, topk_indices: Optional[torch.Tensor]) -> None:
        if (
            topk_indices is not None
            and self.should_publish
            and is_cp_active(self._forward_batch)
        ):
            topk_indices = cp_gather_after_forward(topk_indices, self._forward_batch)
        self._topk_indices = topk_indices

    @staticmethod
    def _expand_kpool_seed(src, positions, token_width):
        """Restore the token-index MTP seed ABI from GLM prefill block IDs."""
        import torch

        offsets = torch.arange(4, device=src.device, dtype=src.dtype)
        tokens = (src.unsqueeze(-1) * 4 + offsets).flatten(1)[:, :token_width]
        # Match _expand_with_tail exactly, including every invalid slot.
        # Reused decode seeds must not retain the rounded-up block filler.
        future = (positions + 1).to(dtype=src.dtype).view(-1, 1)
        return torch.minimum(tokens, future)

    def publish(self) -> None:
        if self._topk_indices is None or not self.should_publish:
            return
        if self.enabled:
            self._forward_batch.spec_info.dsa_topk_indices = self._topk_indices
        seed_buf = self._seed_buf
        if seed_buf is not None:
            sel = self._forward_batch.spec_info.dsa_seed_topk_select
            src = (
                self._topk_indices[: seed_buf.shape[0]]
                if sel is None
                else self._topk_indices[sel]
            )
            mode = self._forward_batch.forward_mode
            if (
                src.device.type == "npu"
                and src.ndim == seed_buf.ndim == 2
                and src.shape[1] * 4 - 1 == seed_buf.shape[1]
                and mode.is_extend()
                and not mode.is_draft_extend_v2()
                and not mode.is_target_verify()
            ):
                from sglang.srt.hardware_backend.npu.attention.glm53.kpool_indexer import (
                    get_prefill_sparse_block_size,
                )

                if get_prefill_sparse_block_size(self._forward_batch) == 4:
                    positions = self._forward_batch.positions
                    positions = (
                        positions[: src.shape[0]] if sel is None else positions[sel]
                    )
                    src = self._expand_kpool_seed(
                        src, positions, seed_buf.shape[1]
                    )
            seed_buf[: src.shape[0]].copy_(src)

    @classmethod
    @contextmanager
    def mtp_iteration(
        cls,
        forward_batch: ForwardBatch,
        enabled: bool = True,
        keep_carry_seed: bool = False,
    ) -> Iterator[Optional[IndexTopKShareState]]:
        if not enabled:
            yield None
            return
        spec_info = forward_batch.spec_info
        forward_batch.reuse_dsa_topk_indices = True
        # Keep the draft-extend seed so step 0 reuses it; else recompute it.
        if not (keep_carry_seed and spec_info.dsa_topk_indices is not None):
            spec_info.dsa_topk_indices = None
        try:
            yield cls.from_mtp_carry(forward_batch)
        finally:
            spec_info.dsa_topk_indices = None
            forward_batch.reuse_dsa_topk_indices = False
