"""Bind GLM NPU KPool request state to the existing PD DSA_TAIL protocol.

This is initialization-only wiring for homogeneous TP16/EP16/PP1. Paged K/V
and index buffers already travel in NPU MLA's main KV list; KDA conv/recurrent
state uses the existing MAMBA component. Only live partial KPool groups need
this additional component. MTP rollback scratch is batch-owned, not request
state, and is reconstructed by the next verify/draft invocation.
"""

import logging
from dataclasses import replace

logger = logging.getLogger(__name__)


def prepare_glm53_pd_mamba_state(token_pool, req_pool, *, mode, draft_tokens):
    """Match P's persistent state layout to D without allocating verify scratch.

    The generic PD prefill allocator passes speculative_num_draft_tokens=None.
    On NPU this also removes the conv padding and temporal transpose needed by
    D. Normalize only this GLM P pool before any forward, checkpoint or graph.
    """
    pool = getattr(token_pool, "mamba_pool", None)
    if pool is None:  # NextN is DSA-only and shares the target request pool.
        return
    if pool is not req_pool.mamba_pool:
        raise ValueError("GLM53 PD target KV/request pools must share Mamba state")
    if mode != "prefill" or draft_tokens is None:
        return
    if getattr(pool, "_glm53_pd_draft_tokens", None) is not None:
        if pool._glm53_pd_draft_tokens != draft_tokens:
            raise RuntimeError(
                "Cannot change GLM53 PD Mamba layout after initialization"
            )
        return
    if draft_tokens < 1:
        raise ValueError("GLM53 PD draft-token capacity must be positive")
    state = pool.mamba_cache
    if hasattr(state, "intermediate_ssm"):
        raise ValueError(
            "GLM53 PD prefill expects the upstream non-speculative Mamba pool"
        )
    if not state.temporal.is_contiguous():
        raise ValueError(
            "GLM53 PD prefill temporal state must start in native contiguous storage"
        )
    convs = []
    extra_bytes = 0
    for conv in state.conv:
        # [layers, physical Mamba slots, window, TP-local QKV channels].
        if len(conv.shape) != 4 or not conv.is_contiguous():
            raise ValueError("GLM53 PD requires contiguous NPU conv state")
        if draft_tokens == 1:
            convs.append(conv)
            continue
        shape = (*conv.shape[:2], conv.shape[2] + draft_tokens - 1, conv.shape[3])
        padded = conv.new_zeros(shape)
        padded[:, :, -conv.shape[2] :, :].copy_(conv)
        convs.append(padded)
        extra_bytes += padded.nbytes - conv.nbytes
    # Match the D allocator's view exactly. _glm_recurrent transposes this view
    # back for its [K,V] kernel; sending raw bytes now preserves the same matrix.
    pool.mamba_cache = replace(
        state, conv=convs, temporal=state.temporal.transpose(-1, -2)
    )
    pool.mem_usage += extra_bytes / (1 << 30)
    pool._glm53_pd_draft_tokens = draft_tokens
    logger.info(
        "GLM53 PD prefill Mamba layout: conv=%s temporal=%s stride=%s "
        "extra_conv_bytes=%d (verify scratch remains disabled)",
        [tuple(conv.shape) for conv in convs],
        tuple(pool.mamba_cache.temporal.shape),
        pool.mamba_cache.temporal.stride(),
        extra_bytes,
    )


def grow_kpool_request_state(indexer, num_req_slots):
    """Grow persistent request rows before graph capture, preserving contents."""
    if num_req_slots < 1:
        raise ValueError("GLM KPool request pool must include its padding row")
    for name in ("_kpool_tail_k", "_kpool_tail_score"):
        old = getattr(indexer, name)
        if old.shape[0] < num_req_slots:
            grown = old.new_zeros((num_req_slots, *old.shape[1:]))
            grown[: old.shape[0]].copy_(old)
            # Assigning an existing nn.Module buffer preserves its registration
            # and persistent=False status. No resize may happen after capture.
            setattr(indexer, name, grown)


def register_glm53_kpool_state(model, token_pool, req_pool, *, parallel):
    from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool

    if not (
        parallel.tp_size == parallel.moe_ep_size == parallel.attn_tp_size == 16
        and parallel.pp_size == 1
        and parallel.attn_cp_size == 1
        and not parallel.enable_dp_attention
    ):
        raise ValueError(
            "GLM53 NPU PD state currently requires TP16/EP16/PP1 without DP/CP"
        )
    pool = getattr(token_pool, "full_kv_pool", token_pool)
    if not isinstance(pool, NPUMLATokenToKVPool):
        raise ValueError("GLM53 NPU PD requires the native NPU MLA KV pool")
    if getattr(pool, "_glm53_index_layout", None) is not None or getattr(
        pool, "share_zero_rope", False
    ):
        raise ValueError(
            "GLM53 NPU PD does not support compact index/shared RoPE pools"
        )

    indexers = sorted(
        (m for m in model.modules() if hasattr(m, "_kpool_tail_k")),
        key=lambda m: m.layer_id,
    )
    mapping = getattr(token_pool, "full_attention_layer_id_mapping", None)
    expected_ids = (
        sorted(mapping) if mapping is not None else list(pool.indexer_layer_ids)
    )
    if not indexers or [m.layer_id for m in indexers] != expected_ids:
        raise ValueError("GLM53 PD KPool indexers must cover every index-cache layer")
    num_req_slots = int(req_pool.req_to_token.shape[0])
    for indexer in indexers:
        if indexer.index_kpool != 4 or indexer.head_dim != 128:
            raise ValueError("GLM53 NPU PD supports four-token, 128-wide KPool tails")
        keys, scores = indexer._kpool_tail_k, indexer._kpool_tail_score
        if (
            keys.shape != scores.shape
            or tuple(keys.shape[1:]) != (4, 128)
            or not keys.is_contiguous()
            or not scores.is_contiguous()
            or keys.device != scores.device
        ):
            raise ValueError("GLM53 PD requires matching contiguous KPool tail tensors")

    registered = getattr(pool, "_glm53_kpool_tail_buffers", ())
    if registered:
        current = tuple(m._kpool_tail_k for m in indexers) + tuple(
            m._kpool_tail_score for m in indexers
        )
        if (
            any(a is not b for a, b in zip(current, registered))
            or len(current) != len(registered)
            or pool._glm53_kpool_req_pool is not req_pool
            or any(buf.shape[0] < num_req_slots for buf in current)
        ):
            raise RuntimeError(
                "Cannot replace GLM53 PD tail buffers after registration"
            )
        return

    for indexer in indexers:
        grow_kpool_request_state(indexer, num_req_slots)
    pool._glm53_kpool_tail_buffers = tuple(m._kpool_tail_k for m in indexers) + tuple(
        m._kpool_tail_score for m in indexers
    )
    pool._glm53_kpool_req_pool = req_pool
    # This describes the actual persistent four-slot row. Extra speculative
    # states live in separate batch-indexed rollback tensors, not in this ring.
    pool.kpool_use_compress = True
    pool.index_kpool = 4
    pool.tail_extra_slots = 0
    logger.info(
        "GLM53 PD registered %d KPool tail tensors, %d request slots, %d bytes",
        len(pool._glm53_kpool_tail_buffers),
        num_req_slots,
        sum(buf.nbytes for buf in pool._glm53_kpool_tail_buffers),
    )


def combined_kpool_tail_infos(pool, draft_pool=None):
    """One DSA_TAIL payload addresses target and draft via shared request ids."""
    if not getattr(pool, "_glm53_kpool_tail_buffers", ()):
        raise ValueError(
            "GLM53 PD target KPool state was not registered before transfer setup"
        )
    ptrs, lens, item_lens = pool.get_compress_tail_buf_infos()
    if draft_pool is not None:
        draft_pool = getattr(draft_pool, "full_kv_pool", draft_pool)
        if not getattr(draft_pool, "_glm53_kpool_tail_buffers", ()):
            raise ValueError(
                "GLM53 PD draft KPool state was not registered before transfer setup"
            )
        if (
            draft_pool._glm53_kpool_req_pool is not pool._glm53_kpool_req_pool
            or draft_pool.index_kpool != pool.index_kpool
            or draft_pool.tail_extra_slots != pool.tail_extra_slots
        ):
            raise ValueError(
                "GLM53 PD target/draft tails must share request slots and geometry"
            )
        dp, dl, di = draft_pool.get_compress_tail_buf_infos()
        ptrs, lens, item_lens = ptrs + dp, lens + dl, item_lens + di
    return ptrs, lens, item_lens
