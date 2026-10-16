"""Bridge GLM's P-only NextN prefill to its checked normal MoE scope.

The draft has a Deepseek decoder, rather than the target GLM decoder. An
idle hybrid draft domain can be represented as TARGET_VERIFY while its
peer executes EXTEND. Do not broaden the general target/decode phase guard.
"""
from contextlib import contextmanager, nullcontext
import logging
from types import SimpleNamespace

logger = logging.getLogger(__name__)
_logged = False


@contextmanager
def draft_prefill_scope(batch, mlp_context=None):
    from sglang.srt.hardware_backend.npu.moe.glm53_collectives import (
        normal_hccl_idle_dp_scope,
    )
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.runtime_context import get_parallel

    # Constructor admission is restricted to eager two-node GLM P. The
    # scheduler publishes this global flag on active AND fabricated idle peers.
    if not batch.is_extend_in_batch:
        with nullcontext() if mlp_context is None else mlp_context:
            yield
        return
    if batch.spec_info is None or batch.spec_info.is_draft_input() or batch.spec_info.num_tokens_per_req != 1:
        raise ValueError("GLM P draft scope requires one-token-per-row prefill draft metadata")
    proxy = batch
    if batch.forward_mode != ForwardMode.EXTEND:
        rank = get_parallel().attn_dp_rank
        counts = batch.original_global_num_tokens_cpu
        if (
            batch.forward_mode != ForwardMode.TARGET_VERIFY
            or getattr(batch, "_original_forward_mode", None) != ForwardMode.IDLE
            or rank not in (0, 1)
            or not isinstance(counts, (list, tuple))
            or len(counts) != 2
            or counts[rank] != 0
        ):
            raise ValueError("GLM P draft scope only accepts EXTEND or its fabricated idle peer")
        # Scope metadata only: the model still executes its native forward
        # mode. Never mutate the live ForwardBatch, its tensors or state.
        proxy = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            original_global_num_tokens_cpu=counts,
            global_num_tokens_cpu=batch.global_num_tokens_cpu,
            dp_padding_mode=batch.dp_padding_mode,
        )
    with normal_hccl_idle_dp_scope(proxy, mlp_context):
        global _logged
        if not _logged:
            logger.info(
                "GLM53 P draft normal MoE scope active: original=%s padded=%s native_mode=%s",
                batch.original_global_num_tokens_cpu, batch.global_num_tokens_cpu,
                batch.forward_mode,
            )
            _logged = True
        yield
