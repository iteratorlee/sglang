"""Optional request-owned storage for the four-token KPool index cache.

Attention K/V retain their allocator and physical page table. Only compressed
index keys use this layout. Prefix caching and disaggregation are unsupported.
"""

import logging
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class IndexLayout:
    requests: int
    pages_per_request: int
    full_pages: int

    @property
    def pages(self):
        return 1 + self.requests * self.pages_per_request


def make_layout(requests, context, full_tokens, page_size=64, draft_tokens=0):
    if page_size != 64 or min(requests, context, full_tokens) <= 0:
        raise ValueError(
            "Compact GLM index requires positive capacities and page size 64"
        )
    # Reserve the upstream speculative extension plus one compressed guard page.
    per_request = (context + 4 + draft_tokens + 255) // 256 + 1
    layout = IndexLayout(requests, per_request, full_tokens // 64 + 1)
    return layout if layout.pages < layout.full_pages else None


def cache_options(model_config, requests, full_tokens, page_size):
    """Validate optional GLM storage layouts before constructing native pools."""
    from sglang.srt.runtime_context import get_server_args

    arch = model_config.hf_config.architectures or ()
    if not any(a.startswith("Glm5NextForConditionalGeneration") for a in arch):
        return {}
    compact = os.getenv("SGLANG_GLM53_COMPACT_INDEX", "0") == "1"
    shared = os.getenv("SGLANG_GLM53_SHARE_ZERO_ROPE", "0") == "1"
    if not (compact or shared):
        return {}
    cfg = get_server_args()
    if not (
        cfg.quantization == "modelslim"
        and cfg.disable_radix_cache
        and cfg.disaggregation_mode == "null"
        and cfg.tp_size == cfg.ep_size == 16
        and cfg.nnodes == 1
        and cfg.pp_size == 1
        and page_size == 64
        and not cfg.enable_dp_attention
        and not cfg.enable_two_batch_overlap
        and not cfg.enable_unified_memory
    ):
        raise ValueError(
            "GLM compact caches require TP16/EP16 without prefix cache, PD or unified memory"
        )
    layout = (
        make_layout(
            requests,
            model_config.context_len,
            full_tokens,
            page_size,
            cfg.speculative_num_draft_tokens or 0,
        )
        if compact
        else None
    )
    logging.getLogger(__name__).info(
        "GLM native cache layout: context=%s requests=%s compact_index=%s "
        "index_pages=%s full_pages=%s shared_zero_rope=%s",
        model_config.context_len,
        requests,
        layout is not None,
        layout.pages if layout is not None else full_tokens // page_size + 1,
        full_tokens // page_size + 1,
        shared,
    )
    return {
        "index_layout": layout,
        "share_zero_rope": shared,
    }


def index_block_table(batch, original):
    from sglang.srt.model_executor.forward_context import get_token_to_kv_pool

    pool = get_token_to_kv_pool()
    full_pool = getattr(pool, "full_kv_pool", pool)
    layout = getattr(full_pool, "_glm53_index_layout", None)
    if layout is None:
        return original
    from .compact_index_npu import request_block_table

    return request_block_table(batch.req_pool_indices, original, layout)
