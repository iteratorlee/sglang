"""Bounded, opt-in CPU metadata for diagnosing Mamba prefix restoration.

Set SGLANG_MAMBA_RESTORE_TRACE=1 before starting the scheduler. Optionally
restrict RIDs with SGLANG_MAMBA_RESTORE_TRACE_RID_PREFIX and change the
per-process event limit with SGLANG_MAMBA_RESTORE_TRACE_MAX_EVENTS (default
2000). Only attention TP/CP leaders log. No tensor values or tokens are read.
"""

import json
import logging
import os

logger = logging.getLogger(__name__)
_ENABLED = os.environ.get("SGLANG_MAMBA_RESTORE_TRACE") == "1"
_RID_PREFIX = os.environ.get("SGLANG_MAMBA_RESTORE_TRACE_RID_PREFIX", "")
_LIMIT = 2000
if _ENABLED:
    _LIMIT = max(
        0, int(os.environ.get("SGLANG_MAMBA_RESTORE_TRACE_MAX_EVENTS", "2000"))
    )
_emitted = 0


def trace_enabled(req) -> bool:
    return (
        _ENABLED
        and req is not None
        and _emitted < _LIMIT
        and req.rid.startswith(_RID_PREFIX)
    )


def _ranks():
    from sglang.srt.runtime_context import get_disagg, get_parallel

    parallel = get_parallel()
    return (
        parallel.attn_dp_rank,
        parallel.attn_tp_rank,
        parallel.attn_cp_rank,
        get_disagg().disaggregation_mode,
    )


def trace_event(req, event: str, **fields) -> None:
    global _emitted
    if not trace_enabled(req):
        return
    dp, tp, cp, role = _ranks()
    if tp != 0 or cp != 0:
        return
    _emitted += 1
    logger.info(
        "MAMBA_RESTORE_TRACE %s",
        json.dumps(
            dict(
                event=event,
                rid=req.rid,
                dp=dp,
                role=role,
                pid=os.getpid(),
                seq=_emitted,
                **fields,
            ),
            separators=(",", ":"),
        ),
    )


def trace_insert(req, result, *, requested_depth, actual_depth, finished, chunked):
    if not trace_enabled(req):
        return
    declined = result.rotation_tail_declined
    trace_event(
        req,
        "insert",
        requested_depth=requested_depth,
        actual_insert_depth=None if declined else actual_depth,
        last_track=req.kv.mamba_last_track_seqlen,
        branching=req.mamba_branching_seqlen,
        protected_depth=req.kv.cache_protected_len,
        finished=finished,
        chunked=chunked,
        declined=declined,
        mamba_exist=result.mamba_exist,
        node=result.last_device_node,
    )
