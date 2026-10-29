"""CPU regression for skip-insert chunk continuation and KV/Mamba ownership.

AST-load production methods to avoid accelerator-only module dependencies. Tensor
storage, allocation accounting and recurrent-state fingerprints use CPU Torch;
the fingerprints check state lifetime, not model numerical equivalence.
"""

import ast
from array import array
import dataclasses
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

import torch

ROOT = Path(__file__).resolve().parents[3]
COMMON = "python/sglang/srt/mem_cache/common.py"
BATCH = "python/sglang/srt/managers/schedule_batch.py"
POLICY = "python/sglang/srt/managers/schedule_policy.py"
UNIFIED = "python/sglang/srt/mem_cache/unified_radix_cache.py"
POOL = "python/sglang/srt/mem_cache/memory_pool.py"
MAMBA = "python/sglang/srt/mem_cache/unified_cache/components/mamba.py"


def load(path, name, namespace, owner=None):
    tree = ast.parse((ROOT / path).read_text())
    nodes = (
        tree.body
        if owner is None
        else next(
            node.body
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == owner
        )
    )
    node = next(node for node in nodes if getattr(node, "name", None) == name)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    exec(
        compile(ast.fix_missing_locations(module), str(ROOT / path), "exec"), namespace
    )
    return namespace[name]


N = dict(
    globals(),
    get_serving=lambda: NS(strip_thinking_cache=False),
    get_spec=lambda: NS(speculative_algorithm=None),
    ceil_align=lambda n, p: (n + p - 1) // p * p,
    branch_checkpoint_chunk_limit=lambda *a, **kw: None,
    CLIP_MAX_NEW_TOKENS=4096,
)
Kv = load(BATCH, "ReqKvInfo", N)
maybe_cache = load(COMMON, "maybe_cache_unfinished_req", N)
release = load(COMMON, "release_kv_cache", N)
load(COMMON, "_release_overallocated_kv_indices", N)
free_segments = load(COMMON, "free_kv_row_segments", N)


class Slots:
    """Allocator boundary that rejects duplicate allocation or release."""

    page_size = 64

    def __init__(self, count):
        self.live = torch.zeros(count, dtype=torch.bool, device="cpu")
        self.freed = []

    def reserve(self, indices):
        assert not self.live[indices].any(), "duplicate allocation"
        self.live[indices] = True

    def free(self, indices):
        indices = indices.to(dtype=torch.int64)
        assert indices.unique().numel() == indices.numel()
        assert self.live[indices].all(), "double free or unowned slot"
        self.live[indices] = False
        self.freed.extend(indices.tolist())

    def free_segments(self, segments):
        for indices, _ in segments:
            self.free(indices)


class RequestPool:
    def __init__(self, count):
        self.req_to_token = torch.arange(count, dtype=torch.int64, device="cpu").view(
            1, -1
        )
        self.mamba_allocator = Slots(3)
        self.mamba_allocator.reserve(torch.arange(3, device="cpu"))
        self.enable_mamba_extra_buffer = True
        self.enable_mamba_extra_buffer_lazy = False
        self.mamba_ping_pong_track_buffer_size = 2
        self.req_index_to_mamba_ping_pong_track_buffer_mapping = torch.tensor(
            [[1, 2]], device="cpu"
        )
        self.states = torch.zeros((3, 2), dtype=torch.float64, device="cpu")
        self.released_rows = []

    def free(self, req):
        self.released_rows.append(req.kv.req_pool_idx)
        req.kv.req_pool_idx = None


RequestPool.free_mamba_cache = load(POOL, "free_mamba_cache", N, "HybridReqToTokenPool")
N["HybridReqToTokenPool"] = RequestPool


class Cache:
    def __init__(self, count):
        self.req_to_token_pool = RequestPool(count)
        self.token_to_kv_pool_allocator = Slots(count)
        self.disable = self.enable_session_radix_cache = False
        self.enable_mamba_extra_buffer = True
        self.session = NS(try_cache_finished_req=lambda *a, **kw: False)
        self.insert = Mock(side_effect=AssertionError("unexpected tree insertion"))
        self.cache_unfinished_req = Mock(
            side_effect=AssertionError("skip request entered cache")
        )
        self.dec_lock_ref = Mock()
        component = NS(cache=self, int8_ckpt_pool=None)
        component.cleanup_after_caching_req = load(
            MAMBA, "cleanup_after_caching_req", N, "MambaComponent"
        ).__get__(component)
        self._components_tuple = (component,)

    def supports_mamba(self):
        return True

    def free_kv_row(self, kv, ranges):
        row = self.req_to_token_pool.req_to_token[kv.req_pool_idx]
        free_segments(
            self.token_to_kv_pool_allocator,
            [(row[start:end], start) for start, end in ranges],
            swa_evicted_seqlen=kv.swa_evicted_seqlen,
        )


Cache.cache_finished_req = load(UNIFIED, "cache_finished_req", N, "UnifiedRadixCache")
Cache._dec_req_lock = load(UNIFIED, "_dec_req_lock", N, "UnifiedRadixCache")


class Adder:
    def __init__(self, cache, limit):
        self.tree_cache = cache
        self.page_size = 64
        self.dllm_config = None
        self.rem_chunk_tokens = limit
        self.rem_total_tokens = 1_048_576
        self.is_hybrid_swa = False
        self.prefill_delayer_single_pass = None
        self.can_run_list = []

    def _update_prefill_budget(self, *args, **kwargs):
        pass

    def _mamba_gap_budget_for_req(self, req):
        return 0


Adder.add_chunked_req = load(POLICY, "add_chunked_req", N, "PrefillAdder")


def request(cache, length, protected=0):
    req = NS(
        origin_input_ids=array("q", range(length)),
        output_ids=array("q"),
        full_untruncated_fill_ids=range(length),
        skip_radix_cache_insert=True,
        prefix_indices=torch.arange(protected, device="cpu"),
        last_node=object(),
        lock_receipt=object(),
        swa_prefix_lock_released=False,
        sampling_params=NS(max_new_tokens=1),
        retracted_stain=False,
    )
    req.kv = Kv(
        req_pool_idx=0,
        cache_protected_len=protected,
        kv_committed_len=protected,
        kv_allocated_len=protected,
        mamba_pool_idx=torch.tensor(0, device="cpu"),
        mamba_ping_pong_track_buffer=torch.tensor([1, 2], device="cpu"),
        mamba_next_track_idx=0,
        mamba_last_track_idx=1,
        mamba_last_track_seqlen=protected,
    )
    req.set_extend_range = lambda start, end: setattr(
        req, "extend_range", NS(start=start, end=end, length=end - start)
    )
    req.effective_kv_committed_len = load(
        BATCH, "effective_kv_committed_len", N, "Req"
    ).__get__(req)
    if protected:
        cache.token_to_kv_pool_allocator.reserve(torch.arange(protected, device="cpu"))
    return req


class ChunkProgressTests(unittest.TestCase):
    def lifecycle(self, length, limit, protected=0, last_node=True):
        cache = Cache(length)
        req = request(cache, length, protected)
        if not last_node:
            req.last_node = None
        node, receipt = req.last_node, req.lock_receipt
        state = cache.req_to_token_pool.states[0]
        state[0] = sum(range(protected))
        state[1] = protected
        slot = req.kv.mamba_pool_idx
        ping_pong = req.kv.mamba_ping_pong_track_buffer
        spans = []
        while len(spans) < length // limit + 2:
            remaining = Adder(cache, limit).add_chunked_req(req)
            start, end = req.extend_range.start, req.extend_range.end
            spans.append((start, end))
            self.assertEqual(start, req.kv.kv_committed_len)
            indices = cache.req_to_token_pool.req_to_token[0, start:end]
            cache.token_to_kv_pool_allocator.reserve(indices)
            req.kv.kv_committed_len = req.kv.kv_allocated_len = end
            # Independent CPU recurrent fingerprint lives in the same Mamba slot
            # across every forward and must match one uninterrupted full prefix.
            state[0] += torch.arange(
                start, end, dtype=torch.float64, device="cpu"
            ).sum()
            state[1] += end - start
            maybe_cache(req, cache, chunked=True)
            self.assertEqual(len(req.prefix_indices), end)
            self.assertIs(req.last_node, node)
            self.assertIs(req.lock_receipt, receipt)
            self.assertEqual(req.kv.cache_protected_len, protected)
            self.assertIs(req.kv.mamba_pool_idx, slot)
            self.assertIs(req.kv.mamba_ping_pong_track_buffer, ping_pong)
            self.assertEqual(req.kv.mamba_last_track_seqlen, protected)
            cache.dec_lock_ref.assert_not_called()
            self.assertEqual(cache.token_to_kv_pool_allocator.live.sum().item(), end)
            if remaining is None:
                break
        self.assertEqual(spans[-1][1], length)
        torch.testing.assert_close(
            state,
            torch.tensor(
                [sum(range(length)), length], dtype=torch.float64, device="cpu"
            ),
        )
        maybe_cache(req, cache)  # Final nonchunked skip must preserve ownership.
        release(req, cache)
        self.assertEqual(cache.token_to_kv_pool_allocator.live.sum().item(), protected)
        self.assertEqual(
            cache.token_to_kv_pool_allocator.freed, list(range(protected, length))
        )
        self.assertEqual(cache.req_to_token_pool.mamba_allocator.freed, [0, 1, 2])
        self.assertFalse(cache.req_to_token_pool.mamba_allocator.live.any())
        self.assertEqual(cache.req_to_token_pool.released_rows, [0])
        self.assertTrue(req.kv.is_kv_released)
        self.assertFalse(req.kv.holds_kv or req.kv.holds_mamba)
        self.assertIsNone(req.kv.mamba_ping_pong_track_buffer)
        cache.insert.assert_not_called()
        cache.cache_unfinished_req.assert_not_called()
        if node is None:
            cache.dec_lock_ref.assert_not_called()
        else:
            cache.dec_lock_ref.assert_called_once_with(node, receipt, skip_swa=False)
        return spans

    def test_128k_finishes_in_eight_chunks_without_kv_or_mamba_leak(self):
        spans = self.lifecycle(131072, 16384)
        self.assertEqual(spans, [(i, i + 16384) for i in range(0, 131072, 16384)])

    def test_matched_tree_prefix_is_preserved_and_lock_released_once(self):
        self.assertEqual(
            self.lifecycle(1024, 256, protected=256),
            [(256, 512), (512, 768), (768, 1024)],
        )

    def test_request_without_tree_node_releases_all_slots(self):
        self.lifecycle(512, 256, last_node=False)

    def test_suffix_extends_full_prefix_with_int64_nonaliased_copy(self):
        for dtype in (torch.int32, torch.int64):
            with self.subTest(dtype=dtype):
                mapping = torch.arange(24, dtype=dtype, device="cpu").view(2, 12)
                req = NS(
                    skip_radix_cache_insert=True,
                    kv=NS(req_pool_idx=1),
                    prefix_indices=mapping[1, :3].to(torch.int64, copy=True),
                    extend_range=NS(start=3, end=7),
                )
                cache = NS(
                    req_to_token_pool=NS(req_to_token=mapping),
                    cache_unfinished_req=Mock(),
                )
                maybe_cache(req, cache, chunked=True)
                self.assertEqual(req.prefix_indices.dtype, torch.int64)
                torch.testing.assert_close(
                    req.prefix_indices, mapping[1, :7].to(torch.int64)
                )
                self.assertNotEqual(
                    req.prefix_indices.data_ptr(), mapping[1].data_ptr()
                )
                mapping[1, :7] = -1
                self.assertEqual(req.prefix_indices.tolist(), list(range(12, 19)))
                req.prefix_indices[0] = 1000
                self.assertEqual(mapping[1, 0].item(), -1)
                cache.cache_unfinished_req.assert_not_called()

    def test_nonchunked_skip_does_not_touch_request_or_pool(self):
        for kwargs in ({}, {"chunked": False}):
            req = NS(skip_radix_cache_insert=True, prefix_indices=object())
            cache = NS(cache_unfinished_req=Mock())
            before = vars(req).copy()
            maybe_cache(req, cache, **kwargs)
            self.assertEqual(vars(req), before)
            cache.cache_unfinished_req.assert_not_called()

    def test_ordinary_and_legacy_requests_delegate_unchanged(self):
        for attributes in ({}, {"skip_radix_cache_insert": False}):
            for kwargs in ({}, {"chunked": True, "checkpoint": object()}):
                req = NS(**attributes)
                cache = NS(cache_unfinished_req=Mock())
                maybe_cache(req, cache, **kwargs)
                cache.cache_unfinished_req.assert_called_once_with(req, **kwargs)


if __name__ == "__main__":
    unittest.main()
