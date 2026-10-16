"""Bridge GLM's P-only NextN prefill to its checked normal MoE scope.

The draft has a Deepseek decoder, rather than the target GLM decoder.
EAGLE_DRAFT_EXTEND belongs to the draft-input family and its idle prefill
peers are fabricated EXTEND batches. Do not admit real verify/decode phases.
"""
from contextlib import contextmanager, nullcontext
import logging
import os

logger = logging.getLogger(__name__)
_logged = False
_small_logged = False


@contextmanager
def draft_prefill_scope(batch, mlp_context=None):
    from sglang.srt.hardware_backend.npu.moe.glm53_collectives import (
        normal_hccl_idle_dp_scope,
    )
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.speculative.spec_info import SpecInputType

    # Constructor admission is restricted to eager two-node GLM P. The
    # scheduler publishes this global flag on active AND fabricated idle peers.
    if not batch.is_extend_in_batch:
        with nullcontext() if mlp_context is None else mlp_context:
            yield
        return
    if (
        batch.spec_info is None
        or batch.spec_info.spec_input_type != SpecInputType.EAGLE_DRAFT_EXTEND
        or batch.spec_info.num_tokens_per_req != 1
    ):
        raise ValueError("GLM P draft scope requires one-token-per-row EAGLE_DRAFT_EXTEND metadata")
    if batch.forward_mode != ForwardMode.EXTEND:
        raise ValueError("GLM P draft scope requires native EXTEND on active and idle peers")
    minimum = int(os.getenv("SGLANG_GLM53_NORMAL_PREFILL_DRAFT_MIN_TOKENS", "0"))
    if minimum < 0:
        raise ValueError("GLM P draft minimum token count must be nonnegative")
    if minimum:
        counts = batch.original_global_num_tokens_cpu
        if not counts or any(count < 0 for count in counts):
            raise ValueError("GLM P draft minimum requires original per-DP token counts")
        # Every EP rank sees the same original DP vector. Use its maximum,
        # never this rank's active count, so idle peers choose the same path.
        # Keep short prompts/cache tails on the original IndexAdd/all-reduce
        # implementation; large prefill is the measured optimization scope.
        if max(counts) < minimum:
            global _small_logged
            if not _small_logged:
                logger.info(
                    "GLM53 P draft reference path selected: original=%s minimum=%d",
                    counts, minimum,
                )
                _small_logged = True
            with nullcontext() if mlp_context is None else mlp_context:
                yield
            return
    with normal_hccl_idle_dp_scope(batch, mlp_context):
        global _logged
        if not _logged:
            logger.info(
                "GLM53 P draft normal MoE scope active: original=%s padded=%s native_mode=%s",
                batch.original_global_num_tokens_cpu, batch.global_num_tokens_cpu,
                batch.forward_mode,
            )
            _logged = True
        yield
