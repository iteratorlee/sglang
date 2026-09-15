"""Opt-in startup JIT warmup; launch real allocator kernels on private scratch only.

The live pools supply metadata, never tensors to a kernel or an allocation call.
No arithmetic, allocation policy, graph, speculative decoding or KV state changes.
"""

import logging
import os
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)
FLAG = "SGLANG_GLM53_PD_WARM_ALLOCATOR"
# Conservative extra key requested for the observed allocator stall.  The
# vendor's default BLOCK_SIZE=2048 is not evidence of max_extend=2048; keep
# these two constants conceptually separate and cover the latter explicitly.
MIN_EXTEND_KEY_COVERAGE = 2048


def power2(n):
    if n <= 0:
        raise ValueError("positive specialization size required")
    return 1 << (n - 1).bit_length()


@dataclass(frozen=True)
class WarmupPlan:
    max_running: int
    pool_len: int
    page_size: int
    reserve: int
    device: str
    req_dtype: str = "int32"
    free_dtype: str = "int64"


@dataclass(frozen=True)
class Variant:
    kernel: str
    batch: int
    lens_dtype: str
    free_offset: int = 0
    max_extend: int = 0

    @property
    def bs_upper(self):
        return power2(self.batch)


def variants(plan):
    """Enumerate cache keys, not token lengths from one observed slow request.

    EAGLE cur/nxt are int32, with committed<=allocated; page rounding permits
    at most reserve+page_size-1 extra tokens per request (including partial
    initial PD allocations). Vendor alloc is used only below 200 new pages.
    Initial PD preallocation is one request, int64, with up to 199 new pages
    plus a partial existing page. Its longer path uses torch, not this kernel.
    """
    if not (1 <= plan.max_running <= 256 and plan.pool_len > 0):
        raise ValueError(
            "allocator warmup requires 1..256 running requests and a positive pool_len"
        )
    if plan.page_size < 2 or plan.page_size & (plan.page_size - 1):
        raise ValueError("allocator warmup requires a paged power-of-two page_size")
    if not (0 < plan.reserve < plan.pool_len):
        raise ValueError(
            "allocator warmup requires a positive reserve smaller than the real row"
        )
    if plan.req_dtype != "int32" or plan.free_dtype != "int64":
        raise ValueError(
            "unsupported live pool metadata dtypes; expected req int32/free pages int64"
        )
    if plan.pool_len * 4 > 64 * 1024 * 1024:
        raise ValueError(
            "one private request row exceeds the bounded 64 MiB scratch budget"
        )
    result = []
    for bs in range(1, plan.max_running + 1):
        result.append(Variant("assign", bs, "int32"))
        # Include small values: an initial PD allocation need not be aligned.
        upper = min(
            bs * (plan.reserve + plan.page_size - 1),
            199 * plan.page_size + bs * (plan.page_size - 1),
        )
        upper = max(upper, MIN_EXTEND_KEY_COVERAGE)
        for key in (1 << p for p in range(power2(upper).bit_length())):
            for offset in (0, 1):
                result.append(Variant("extend", bs, "int32", offset, key))
    upper_prealloc = min(plan.pool_len, 199 * plan.page_size + plan.page_size - 1)
    for key in (1 << p for p in range(power2(upper_prealloc).bit_length())):
        for offset in (0, 1):
            result.append(Variant("extend", 1, "int64", offset, key))
    return result


def warm_scratch(plan, *, on_launch=None):
    """Warm exact JIT objects used by allocation.py and NPUPaged's small path.

    All prefix/end lengths are zero, so both kernels perform zero stores and
    allocator's Part 1 returns before touching the private free-page values.
    This safely compiles every constant while keeping scratch O(pool_len+BS).
    Tensor values are runtime loads; they do not specialize away the real code.
    """
    cases = variants(plan)
    import torch

    from sgl_kernel_npu.mem_cache.allocator import alloc_extend_kernel
    from sglang.srt.mem_cache.allocation import assign_req_to_token_pool

    expected_args = [
        "pre_lens_ptr",
        "seq_lens_ptr",
        "last_loc_ptr",
        "free_page_ptr",
        "out_indices",
        "bs_upper",
        "page_size",
        "max_num_extend_tokens",
        "BLOCK_SIZE",
    ]
    if list(alloc_extend_kernel.arg_names) != expected_args:
        raise RuntimeError(
            "vendor allocator signature changed; warmup contract needs review"
        )
    # Allocate independently; never empty_like/clone/view a live pool tensor.
    row = torch.full((1, plan.pool_len), -731, dtype=torch.int32, device=plan.device)
    free_base = torch.zeros(3, dtype=torch.int64, device=plan.device)
    out32 = torch.full((1,), -733, dtype=torch.int32, device=plan.device)
    out64 = torch.full((1,), -737, dtype=torch.int64, device=plan.device)
    buffers = {}
    launched = []
    for case in cases:
        key = (case.batch, case.lens_dtype)
        if key not in buffers:
            dtype = getattr(torch, case.lens_dtype)
            buffers[key] = (
                torch.zeros(case.batch, dtype=dtype, device=plan.device),
                torch.zeros(case.batch, dtype=dtype, device=plan.device),
                torch.full((case.batch,), -1, dtype=dtype, device=plan.device),
                torch.zeros(case.batch, dtype=torch.int64, device=plan.device),
            )
        start, end, last, reqs = buffers[key]
        if case.kernel == "assign":
            binary = assign_req_to_token_pool[(case.batch,)](
                reqs, row, start, end, out32, plan.pool_len, case.bs_upper
            )
        else:
            binary = alloc_extend_kernel[(case.batch,)](
                start,
                end,
                last,
                free_base[case.free_offset : case.free_offset + 1],
                out64,
                case.bs_upper,
                plan.page_size,
                case.max_extend,
            )
        if on_launch is not None:
            on_launch(case, binary)
        launched.append(
            (
                case.kernel,
                case.bs_upper,
                case.lens_dtype,
                case.free_offset,
                case.max_extend,
            )
        )
    torch.npu.synchronize()
    return {
        "launches": len(cases),
        "unique_specializations": len(set(launched)),
        "scratch_row_bytes": plan.pool_len * 4,
    }


def maybe_warm_pd_allocators(runner):
    """Called once by target after final pool construction, before graph setup."""
    if os.getenv(FLAG, "0") != "1":
        return None
    if getattr(runner, "is_draft_worker", True):
        return None
    if str(getattr(runner, "device", "")).split(":", 1)[0] != "npu":
        return None
    if getattr(runner.server_args, "disaggregation_mode", None) != "decode":
        return None
    architectures = getattr(runner.model_config.hf_config, "architectures", ()) or ()
    if "Glm5NextForConditionalGeneration" not in architectures:
        return None
    if getattr(runner, "_glm53_pd_allocator_warmed", False):
        return None

    import torch

    from sglang.srt.hardware_backend.npu.allocator_npu import (
        NPUPagedTokenToKVPoolAllocator,
    )
    from sglang.srt.mem_cache.allocation_sizing import get_alloc_reserve_per_decode
    from sglang.srt.runtime_context import get_spec

    allocator = runner.token_to_kv_pool_allocator
    if type(allocator) is not NPUPagedTokenToKVPoolAllocator:
        raise RuntimeError(
            "GLM53 PD allocator warmup currently supports exact NPUPagedTokenToKVPoolAllocator only"
        )
    if get_spec().speculative_algorithm not in ("EAGLE", "EAGLE3"):
        raise RuntimeError(
            "GLM53 PD allocator warmup currently requires the EAGLE int32 allocation contract"
        )
    # The observed EAGLE last_loc dtype is int32 (get_last_loc_torch), not the
    # int64 promotion from the triton-dispatch get_last_loc_safe path.
    from sglang.srt.mem_cache.allocation import attention_backends

    if not set(attention_backends()).intersection(("ascend", "torch_native")):
        raise RuntimeError(
            "GLM53 PD allocator warmup requires the verified torch last_loc dispatch"
        )
    table = runner.req_to_token_pool.req_to_token
    pages = allocator.free_pages
    if (
        table.dtype != torch.int32
        or pages.dtype != torch.int64
        or not table.is_contiguous()
    ):
        raise RuntimeError("unexpected req/free-page dtype or request-table layout")
    if table.device != pages.device or table.device.type != "npu":
        raise RuntimeError("request table and free-page metadata must use the same NPU")
    # Use the real post-construction row width, not model context_len: MTP adds
    # headroom, and PD has more preallocated slots than max_running_requests.
    plan = WarmupPlan(
        int(runner.max_running_requests),
        int(table.shape[1]),
        int(allocator.page_size),
        int(get_alloc_reserve_per_decode()),
        str(table.device),
    )
    start = time.monotonic()
    summary = warm_scratch(plan)
    runner._glm53_pd_allocator_warmed = True
    logger.info(
        "GLM53 PD scratch allocator warmup done: plan=%s stats=%s elapsed=%.3fs",
        plan,
        summary,
        time.monotonic() - start,
    )
    return summary
