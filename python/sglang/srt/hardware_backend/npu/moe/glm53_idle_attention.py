"""Skip discarded attention work in an entirely idle, padded P TP8 group.

Only the existing opt-in eager DP2TP8/EP16 prefill profile may use this.
Every TP peer uses the same scheduler-published original count. All ranks
must configure the flag consistently. MAX_LEN and every EP16 collective
are retained; only attention inside a wholly idle attention TP group is
replaced by zero. No active request owns KV or recurrent state there.
"""

import logging
import os

logger = logging.getLogger(__name__)
_logged = False


def skip_idle_attention(batch, *, is_nextn, local_rows):
    if is_nextn or os.getenv("SGLANG_GLM53_NORMAL_SKIP_IDLE_ATTN", "0") != "1":
        return False
    from sglang.srt.hardware_backend.npu.moe.glm53_collectives import (
        normal_hccl_skip_idle_dp_enabled,
    )
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.runtime_context import get_parallel

    if not normal_hccl_skip_idle_dp_enabled() or batch.forward_mode != ForwardMode.EXTEND:
        return False
    if not batch.dp_padding_mode.is_max_len():
        return False
    original = getattr(batch, "original_global_num_tokens_cpu", None)
    padded = getattr(batch, "global_num_tokens_cpu", None)
    if any(
        not isinstance(values, (list, tuple))
        or len(values) != 2
        or any(type(n) is not int or n < 0 for n in values)
        for values in (original, padded)
    ):
        return False
    if not (
        0 < padded[0] == padded[1] == local_rows <= 8192
        and all(real <= capacity for real, capacity in zip(original, padded))
        and sum(original) > 0
    ):
        return False
    parallel = get_parallel()
    if parallel.attn_dp_size != 2 or parallel.attn_tp_size != 8:
        return False
    rank = parallel.attn_dp_rank
    if rank not in (0, 1) or original[rank] != 0:
        return False
    global _logged
    if not _logged:
        logger.info(
            "GLM53 idle P attention skipped: dp=%d original=%s padded=%s "
            "local_rows=%d; MAX_LEN and EP16 experts remain active",
            rank,
            original,
            padded,
            local_rows,
        )
        _logged = True
    return True
