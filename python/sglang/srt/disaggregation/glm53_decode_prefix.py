"""Narrow GLM-5.3 NPU PD-decode prefix-cache policy.

The generic PD decode radix path intentionally rejects speculative and hybrid
SSM models.  GLM-5.3 is safe only under the configuration described here:

* target and draft KPool entries share the target allocator's page ids;
* radix ownership is on a 256-token logical boundary (four page64 pages);
* recurrent KDA/conv state is request-scoped and is always transferred from P;
* the D radix tree owns token/KPool pages only, while EAGLE uses the transferred
  state and handoff metadata for each request.

Keep this module dependency-light.  Besides making the server-argument gate
auditable, the pure helpers are exercised by CPU-only tests.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

GLM53_ARCH = "Glm5NextForConditionalGeneration"
GLM53_PHYSICAL_PAGE_SIZE = 64
GLM53_PREFIX_SHARE_PAGE_SIZE = 256


@dataclass(frozen=True)
class Glm53PDDecodePrefixProfile:
    """The one native topology for which the exception is supported."""

    physical_page_size: int = GLM53_PHYSICAL_PAGE_SIZE
    prefix_share_page_size: int = GLM53_PREFIX_SHARE_PAGE_SIZE


@dataclass(frozen=True)
class Glm53PDDecodePrefixPlan:
    """Page-safe split between cached KPool data and P-to-D tail transfer."""

    prefix_indices: Any
    prefix_len: int
    transfer_start: int
    transfer_len: int


def _architectures(model_config: Any) -> Sequence[str]:
    hf_config = getattr(model_config, "hf_config", None)
    return tuple(getattr(hf_config, "architectures", None) or ())


def glm53_pd_decode_prefix_profile(
    cfg: Any, model_config: Any
) -> Optional[Glm53PDDecodePrefixProfile]:
    """Return the supported profile, or ``None`` for every generic path.

    This is deliberately an allow-list rather than model-name-only detection.
    Relaxing a topology or MTP parameter requires its own correctness run.
    """

    device = str(getattr(cfg, "device", "")).split(":", 1)[0]
    return (
        Glm53PDDecodePrefixProfile()
        if (
            GLM53_ARCH in _architectures(model_config)
            and device == "npu"
            and getattr(cfg, "disaggregation_mode", None) == "decode"
            and getattr(cfg, "disaggregation_transfer_backend", None) == "ascend"
            and getattr(cfg, "quantization", None) == "modelslim"
            and getattr(cfg, "speculative_draft_model_quantization", None)
            == "modelslim"
            and getattr(cfg, "tp_size", None) == 16
            and getattr(cfg, "dp_size", None) == 8
            and getattr(cfg, "ep_size", None) == 16
            and getattr(cfg, "pp_size", None) == 1
            and getattr(cfg, "dcp_size", None) == 1
            and getattr(cfg, "attn_cp_size", None) == 1
            and getattr(cfg, "nnodes", None) == 1
            and bool(getattr(cfg, "enable_dp_attention", False))
            and getattr(cfg, "page_size", None) == GLM53_PHYSICAL_PAGE_SIZE
            and not bool(getattr(cfg, "enable_hierarchical_cache", False))
            and getattr(cfg, "disaggregation_decode_retraction_backup", None)
            in (None, "cpu_tensor")
            and str(getattr(cfg, "speculative_algorithm", "")).upper() == "EAGLE"
            and getattr(cfg, "speculative_num_steps", None) == 3
            and getattr(cfg, "speculative_eagle_topk", None) == 1
            and getattr(cfg, "speculative_num_draft_tokens", None) == 4
        )
        else None
    )


def validate_glm53_pd_decode_prefix_runtime(
    profile: Glm53PDDecodePrefixProfile,
    *,
    tree_cache: Any,
    token_to_kv_pool_allocator: Any,
    req_to_token_pool: Any,
    state_types: Iterable[Any],
    draft_token_to_kv_pool: Any,
) -> None:
    """Fail before serving if the builder/transport contract is incomplete."""

    physical_page_size = getattr(token_to_kv_pool_allocator, "page_size", None)
    tree_page_size = getattr(tree_cache, "page_size", None)
    share_page_size = getattr(tree_cache, "glm53_kpool_share_page_size", None)
    supports_mamba = getattr(tree_cache, "supports_mamba", lambda: False)()
    state_values = {
        getattr(state_type, "value", state_type) for state_type in state_types
    }

    if physical_page_size != profile.physical_page_size:
        raise RuntimeError(
            "GLM53 PD decode prefix cache requires physical page64, got "
            f"{physical_page_size!r}"
        )
    if (
        tree_page_size != profile.prefix_share_page_size
        or share_page_size != profile.prefix_share_page_size
    ):
        raise RuntimeError(
            "GLM53 PD decode prefix cache requires a 256-token radix/KPool "
            "sharing boundary while retaining physical page64; got "
            f"tree_page_size={tree_page_size!r}, marker={share_page_size!r}"
        )
    if supports_mamba:
        raise RuntimeError(
            "GLM53 PD decode prefix cache requires a token-only radix tree; "
            "D-side Mamba checkpoints must not gate KPool prefix matches"
        )
    for attr in ("mamba_pool", "mamba_allocator", "free_mamba_cache"):
        if not hasattr(req_to_token_pool, attr):
            raise RuntimeError(
                "GLM53 PD decode prefix cache requires an independent active "
                f"Mamba request pool with {attr}"
            )
    if "mamba" not in state_values:
        raise RuntimeError(
            "GLM53 PD decode prefix cache requires authoritative P-to-D KDA/conv "
            "state transfer (StateType.MAMBA)"
        )
    if "dsa_tail" not in state_values:
        raise RuntimeError(
            "GLM53 PD decode prefix cache requires the request-scoped packed "
            "target/draft KPool index tail (StateType.DSA_TAIL)"
        )
    if draft_token_to_kv_pool is None:
        raise RuntimeError(
            "GLM53 PD decode prefix cache with EAGLE requires the draft KV pool "
            "to share transferred target page ids"
        )


def plan_glm53_pd_decode_prefix(
    profile: Glm53PDDecodePrefixProfile,
    prefix_indices: Any,
    fill_len: int,
) -> Glm53PDDecodePrefixPlan:
    """Clamp reuse to complete KPool sharing pages and describe the tail.

    A correctly built tree already returns a 256-aligned match.  Flooring here
    is a final safety net: a partial four-page KPool group must be transferred
    afresh because its compressed rows can contain values from the old owner.
    """

    if fill_len < 0:
        raise ValueError(f"fill_len must be non-negative, got {fill_len}")
    # EAGLE radix keys are bigrams.  The handoff token is sampled from the
    # final prompt token, so that final token cannot be part of the reusable
    # prefix.  Page alignment then leaves exactly one complete 256-token group
    # as the fresh tail for an aligned exact prompt.
    matched_len = min(len(prefix_indices), max(fill_len - 1, 0))
    prefix_len = (
        matched_len // profile.prefix_share_page_size
    ) * profile.prefix_share_page_size
    safe_prefix_indices = prefix_indices[:prefix_len]
    return Glm53PDDecodePrefixPlan(
        prefix_indices=safe_prefix_indices,
        prefix_len=prefix_len,
        transfer_start=prefix_len,
        transfer_len=fill_len - prefix_len,
    )


def mark_glm53_pd_mamba_state_authoritative(req: Any) -> None:
    """Make the incoming P state the sole live recurrent state for ``req``.

    The token-only D radix tree may reuse target/draft KPool pages, while the
    prefill worker has advanced KDA and conv through the complete prompt.  COW
    or zero-clear on the decode worker would overwrite newer state after RDMA.
    """

    if not getattr(req.kv, "holds_mamba", False):
        raise RuntimeError(
            "GLM53 PD decode requires a fresh active Mamba slot before P-to-D "
            "state transfer"
        )

    req.kv.mamba_cow_src_index = None
    req.kv.mamba_needs_clear = False
    req.kv.mamba_last_track_idx = None
    req.kv.mamba_last_track_seqlen = None
    req.mamba_branching_seqlen = None


def prepare_glm53_pd_rebootstrap(req: Any) -> None:
    """Convert a released GLM53 decode request into a P-recompute request.

    Generic CPU retraction does not carry the separate EAGLE draft KV pool.
    Reusing it for this profile could restore target KV and Mamba while leaving
    draft history at recycled page ids.  The existing PD rebootstrap protocol
    reconstructs all three resources together on P.
    """

    req.retraction_mb_id = None
    req.pd_rebootstrap_forced_output_id = (
        req.output_ids.pop() if req.output_ids else None
    )
    req.pd_rebootstrap_in_progress = True


def write_glm53_cache_audit(
    *,
    rid: str,
    dp_rank: Optional[int],
    tp_rank: int,
    actual_prefix_len: int,
    input_len: int,
    metadata_page_count: int,
    transfer_success: bool,
    fresh_mamba: bool,
    draft_present: bool,
) -> Optional[Path]:
    """Append one opt-in record after a real D transfer commits successfully.

    A separate file per scheduler process avoids cross-process buffering or
    locks.  Auditing is disabled unless ``SGLANG_GLM53_CACHE_AUDIT_DIR`` is set.
    """

    audit_dir = os.environ.get("SGLANG_GLM53_CACHE_AUDIT_DIR")
    if not audit_dir:
        return None

    directory = Path(audit_dir)
    directory.mkdir(parents=True, exist_ok=True)
    resolved_dp_rank = 0 if dp_rank is None else int(dp_rank)
    path = directory / (
        f"decode-prefix-dp{resolved_dp_rank}-tp{int(tp_rank)}-pid{os.getpid()}.jsonl"
    )
    record = {
        "timestamp_ns": time.time_ns(),
        "rid": str(rid),
        "dp_rank": resolved_dp_rank,
        "tp_rank": int(tp_rank),
        "actual_prefix_len": int(actual_prefix_len),
        "input_len": int(input_len),
        "metadata_page_count": int(metadata_page_count),
        "transfer_success": bool(transfer_success),
        "fresh_mamba": bool(fresh_mamba),
        "draft_present": bool(draft_present),
    }
    line = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)
    return path
