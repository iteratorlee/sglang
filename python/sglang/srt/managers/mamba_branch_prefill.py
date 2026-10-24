"""Opt-in GLM P chunk boundary for retaining both branch and tail states."""

import os


def branch_checkpoint_chunk_limit(
    tree_cache, req, chunk_limit, *, physical_page_size, truncation_align_size=None
):
    """Return a shorter chunk length only when a reusable branch would be lost.

    Use the ordinary chunked-prefill donation/continuation lifecycle. The live
    state finishes at the branch on this pass, while the next pass can track
    the tail in the existing single checkpoint slot. Never relabel a state or
    suppress branch tracking. Unsupported profiles retain the original policy.
    """
    if os.environ.get("SGLANG_GLM53_PD_MAMBA_BRANCH_SPLIT") != "1":
        return None

    # This marker is installed only by the GLM NPU UnifiedRadixCache builder.
    # Keep the experiment on the current physical64 / KPool256 cache geometry.
    if (
        chunk_limit is None
        or chunk_limit <= 0
        or physical_page_size != 64
        or getattr(tree_cache, "glm53_kpool_share_page_size", None) != 256
        or tree_cache.page_size != 256
        or tree_cache.disable
        or not tree_cache.enable_mamba_extra_buffer
        or not tree_cache.supports_mamba()
        or req.output_ids
        or req.return_logprob
        or req.multimodal_inputs is not None
        or req.input_embeds is not None
        or req.mamba_branching_seqlen is None
    ):
        return None

    from sglang.srt.runtime_context import (
        get_disagg,
        get_exec,
        get_parallel,
        get_schedule,
        mamba_cache_chunk_size,
        mamba_checkpoint_grid,
    )

    execution = get_exec()
    parallel = get_parallel()
    if (
        get_disagg().disaggregation_mode != "prefill"
        or not execution.mamba.enable_mamba_extra_buffer
        or execution.mamba.enable_mamba_extra_buffer_lazy
        or not get_schedule().disable_overlap_schedule
        or execution.overlap.enable_two_batch_overlap
        or execution.overlap.enable_single_batch_overlap
        or execution.deterministic.enable_deterministic_inference
        or parallel.pp_size != 1
        or parallel.attn_cp_size != 1
        or parallel.dcp_size != 1
        or parallel.enable_prefill_cp
        or mamba_cache_chunk_size() != 64
        or mamba_checkpoint_grid(tree_cache.page_size) != 256
    ):
        return None

    prefix = len(req.prefix_indices)
    branch = req.mamba_branching_seqlen
    length = branch - prefix
    end = len(req.full_untruncated_fill_ids)
    tail = end // 256 * 256
    if (
        end % 256 == 0
        and os.environ.get("SGLANG_GLM53_PD_PREFILL_REUSABLE_CHECKPOINT") == "1"
    ):
        tail -= 256
    if (
        prefix % 64 != 0
        or branch % 256 != 0
        or length < 256  # The existing tracker masks shorter extends.
        or length >= chunk_limit
        or branch >= tail
        or (truncation_align_size and length % truncation_align_size != 0)
    ):
        return None
    return length
