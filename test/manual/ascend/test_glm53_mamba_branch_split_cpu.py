"""CPU source-level regressions for GLM P branch/tail checkpoint retention.

Run with a PyTorch + msgspec CPU-capable interpreter:
  CUDA_VISIBLE_DEVICES='' TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -B \
    test/manual/ascend/test_glm53_mamba_branch_split_cpu.py -v

Executes production tree, components, scheduler admission, tracking and donation
methods. AST loading omits SGLang runtime imports; device allocators, events and
DMA are fixture boundaries. All tensors explicitly live on CPU. Synthetic state
fingerprints check depth/ownership, not NPU model numerical equivalence.
"""

import ast
import dataclasses
from dataclasses import dataclass, field
import heapq
import importlib.util
import logging
import math
import os
import sys
import types
import unittest
from abc import ABC, abstractmethod
from array import array
from collections import defaultdict, namedtuple
from contextlib import nullcontext
from enum import Enum, IntEnum, IntFlag, auto
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any, Callable, NamedTuple, Optional, Sequence
from unittest.mock import patch

import msgspec
import torch
from numpy import float64

from test_glm53_pd_aligned_prefix_cpu import Entry, Mode, extract

ROOT = Path(__file__).resolve().parents[3]
CACHE = "python/sglang/srt/mem_cache/"
TREE = CACHE + "unified_cache/"
POLICY = "python/sglang/srt/managers/schedule_policy.py"
SCHEDULE = "python/sglang/srt/managers/schedule_batch.py"
FLAG = "SGLANG_GLM53_PD_MAMBA_BRANCH_SPLIT"
REUSABLE = "SGLANG_GLM53_PD_PREFILL_REUSABLE_CHECKPOINT"


class Flags:
    def __getattr__(self, name):
        return NS(get=lambda: False)


def load_definitions(path, ns, names=None):
    """Keep actual definitions intact; substitute only module dependencies."""
    source = ast.parse((ROOT / path).read_text())
    body = [
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
    ]
    for node in source.body:
        if names is not None:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
                body.append(node)
        elif isinstance(
            node, (ast.ClassDef, ast.FunctionDef, ast.Assign, ast.AnnAssign)
        ):
            body.append(node)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
            path,
            "exec",
        ),
        ns,
    )


def source_namespace():
    ns = dict(globals())
    ns.update(
        envs=Flags(),
        trace_enabled=lambda req: False,
        trace_insert=lambda *a, **kw: None,
        get_eviction_strategy=lambda *a: NS(get_priority=lambda n: n.last_access_time),
        KVCacheEventRecorder=lambda **kw: NS(record_store=lambda *a, **kw: None),
        StorageMedium=NS(GPU="gpu"),
        split_node_hash_value=lambda v, length, page: (
            (None, None) if v is None else (v[: length // page], v[length // page :])
        ),
        UnifiedTreeCoreInterface=object,
    )
    load_definitions(TREE + "component_type.py", ns)
    load_definitions(
        CACHE + "hicache_storage.py", ns, {"PoolName", "PoolHitPolicy", "PoolTransfer"}
    )
    load_definitions(
        CACHE + "base_prefix_cache.py",
        ns,
        {
            "MatchPrefixParams",
            "InsertParams",
            "InsertResult",
            "MatchResult",
            "IncLockRefResult",
            "DecLockRefParams",
            "DecLockRefResult",
            "InitLoadBackParams",
        },
    )
    load_definitions(CACHE + "radix_cache.py", ns, {"RadixKey"})
    load_definitions(TREE + "cache_action.py", ns)
    load_definitions(TREE + "components/base.py", ns)
    load_definitions(TREE + "components/full.py", ns)
    load_definitions(TREE + "components/mamba.py", ns)
    load_definitions(TREE + "unified_tree_core_interface.py", ns, {"InsertStepResult"})
    load_definitions(TREE + "unified_tree_core.py", ns)
    return ns


N = source_namespace()
CT = N["ComponentType"]


class SlotPool:
    """CPU slots retain fingerprints across real ping-pong donation."""

    def __init__(self, size=32768):
        self.next_slot = 30
        self.freed = []
        self.states = {}
        self.mamba_allocator = self
        self.mamba_ckpt_pool = None
        self.mamba_ping_pong_track_buffer_size = 2
        self.req_to_token = torch.arange(size, dtype=torch.int64, device="cpu").view(
            1, -1
        )
        self.req_index_to_mamba_ping_pong_track_buffer_mapping = torch.zeros(
            (1, 2), dtype=torch.int64, device="cpu"
        )

    def alloc(self, n):
        result = torch.arange(
            self.next_slot, self.next_slot + n, dtype=torch.int64, device="cpu"
        )
        self.next_slot += n
        return result

    def free(self, indices):
        self.freed.extend(indices.tolist())

    def write(self, indices, values):
        self.req_to_token[indices] = values


for name in (
    "get_mamba_ping_pong_other_idx",
    "get_mamba_ping_pong_keep_idx",
    "set_mamba_ping_pong_slot",
    "donate_mamba_ping_pong_slot",
):
    setattr(
        SlotPool,
        name,
        extract(CACHE + "memory_pool.py", name, {"_MAMBA_DEBUG_ASSERTS": False}),
    )


class Cache:
    def __init__(self):
        self.page_size = self.glm53_kpool_share_page_size = 256
        self.tree_components = (CT.FULL, CT.MAMBA)
        self.disable = self.enable_session_radix_cache = False
        self.enable_mamba_extra_buffer = True
        self.req_to_token_pool = SlotPool()
        self.session = NS(try_cache_unfinished_req=lambda *a, **kw: False)
        self.params = NS(
            page_size=256,
            is_eagle=False,
            enable_session_radix_cache=False,
            eviction_policy="lru",
            eviction_policy_config=None,
            token_to_kv_pool_allocator=None,
            enable_kv_cache_events=False,
        )
        full = N["FullComponent"](self, self.params)
        mamba = N["MambaComponent"].__new__(N["MambaComponent"])
        N["TreeComponent"].__init__(mamba, self, self.params)
        mamba.mamba_checkpoint_grid = 256
        mamba.mamba_cache_chunk_size = 64
        mamba.mamba_max_states_per_path = -1
        self.tree_core = N["UnifiedTreeCore"](
            self.params, {CT.FULL: full, CT.MAMBA: mamba}
        )
        self.tree_core.enable_hicache = True
        self._components_tuple = (full, mamba)
        self.actions = []
        self.inserts = []
        self.buffer_pipeline = self.linker = None
        self.fail_load = False

    def supports_mamba(self):
        return True

    def insert(self, params):
        step = self.tree_core.begin_insert(params)
        while True:
            self.actions.extend(step.actions)
            if step.result is not None:
                self.inserts.append(
                    (len(params.key), int(params.mamba_value[0]), params.chunked)
                )
                return step.result
            step = self.tree_core.resume_insert()

    def seed(self, tokens, slot=1):
        return self.insert(
            N["InsertParams"](
                key=N["RadixKey"](array("q", tokens)),
                value=torch.arange(len(tokens), dtype=torch.int64, device="cpu"),
                mamba_value=torch.tensor([slot], dtype=torch.int64, device="cpu"),
                chunked=True,
            )
        )

    def match_prefix(self, params):
        return self.tree_core.match_prefix(params)

    def match(self, tokens):
        return self.match_prefix(
            N["MatchPrefixParams"](N["RadixKey"](array("q", tokens)))
        )

    def inc_lock_ref(self, node, **kwargs):
        return self.tree_core.inc_lock_ref(node, **kwargs)

    def _dec_req_lock(self, req, **kwargs):
        if req.lock_receipt is not None:
            self.tree_core.dec_lock_ref(req.last_node, req.lock_receipt)

    def state(self, node):
        return self.tree_core.node_by_id(node).component_data[CT.MAMBA]

    def backup_and_evict(self, node_id, *, full=True):
        core = self.tree_core
        node = core.node_by_id(node_id)
        host = torch.arange(len(node.key), dtype=torch.int64, device="cpu")
        mhost = torch.tensor([900 + node_id], dtype=torch.int64, device="cpu")
        core.commit_backup(
            node_id,
            host,
            {
                CT.MAMBA: [
                    N["PoolTransfer"](name=N["PoolName"].MAMBA, host_indices=mhost)
                ]
            },
        )
        for component in self._components_tuple:
            if not full and component.component_type == CT.FULL:
                continue
            core._evict_component_and_detach_lru(
                node,
                component,
                defaultdict(list),
                defaultdict(list),
                target=N["EvictLayer"].DEVICE,
                tracker=defaultdict(int),
            )

    def load_back(self, node, quota, *, req):
        mamba = self._components_tuple[1]
        prep = mamba.prepare_load_back(node, req=req)
        if self.fail_load:
            mamba.finalize_load_back(req, prep, False)
            return False
        kv, components = self.tree_core.build_load_back_spec(node, req)
        values = torch.arange(len(kv.host_indices), dtype=torch.int64, device="cpu")
        for transfers in components.values():
            for transfer in transfers:
                if transfer.device_indices is None:
                    transfer.device_indices = self.req_to_token_pool.alloc(1)
        self.tree_core.commit_load_back(node, values, kv, components)
        mamba.finalize_load_back(req, prep, True)
        return True


Cache.cache_unfinished_req = extract(
    CACHE + "unified_radix_cache.py", "cache_unfinished_req", N
)
Cache.init_load_back = extract(CACHE + "unified_radix_cache.py", "init_load_back", N)


def context():
    return NS(
        execution=NS(
            mamba=NS(
                enable_mamba_extra_buffer=True, enable_mamba_extra_buffer_lazy=False
            ),
            overlap=NS(
                enable_two_batch_overlap=False, enable_single_batch_overlap=False
            ),
            deterministic=NS(enable_deterministic_inference=False),
        ),
        parallel=NS(
            dp_size=1, pp_size=1, attn_cp_size=1, dcp_size=1, enable_prefill_cp=False
        ),
        schedule=NS(disable_overlap_schedule=True),
        disagg=NS(disaggregation_mode="prefill"),
    )


def runtime(ctx):
    return NS(
        get_exec=lambda: ctx.execution,
        get_parallel=lambda: ctx.parallel,
        get_schedule=lambda: ctx.schedule,
        get_disagg=lambda: ctx.disagg,
        mamba_cache_chunk_size=lambda: 64,
        mamba_checkpoint_grid=lambda p: math.lcm(64, p),
    )


spec = importlib.util.spec_from_file_location(
    "branch_split", ROOT / "python/sglang/srt/managers/mamba_branch_prefill.py"
)
split = importlib.util.module_from_spec(spec)
spec.loader.exec_module(split)


def request(end=7141, prefix=0, branch=2304, tokens=None):
    ids = array("q", range(end)) if tokens is None else array("q", tokens)
    req = NS(
        rid="sample",
        origin_input_ids=ids,
        full_untruncated_fill_ids=ids,
        prefix_indices=torch.arange(prefix, dtype=torch.int64, device="cpu"),
        output_ids=[],
        return_logprob=False,
        logprob_start_len=-1,
        multimodal_inputs=None,
        input_embeds=None,
        mamba_branching_seqlen=branch,
        kv=NS(
            mamba_ping_pong_track_buffer=torch.tensor(
                [10, 11], dtype=torch.int64, device="cpu"
            ),
            mamba_next_track_idx=0,
            mamba_last_track_idx=None,
            mamba_last_track_seqlen=None,
            req_pool_idx=0,
            cache_protected_len=prefix,
        ),
        sampling_params=NS(max_new_tokens=5, ignore_eos=False),
        retracted_stain=False,
        extra_key=None,
        cache_salt=None,
        kv_rotation_base=None,
        swa_prefix_lock_released=False,
        lock_receipt=None,
        host_hit_length=0,
        host_loaded_length=0,
        swa_host_hit_length=0,
        mamba_host_hit_length=0,
        last_node=0,
    )
    req.set_extend_range = lambda start, end: setattr(
        req, "extend_range", NS(start=start, end=end, length=end - start)
    )
    req.set_extend_range(prefix, end)
    req.get_fill_ids = lambda: req.full_untruncated_fill_ids[: req.extend_range.end]
    req.needs_host_load_back = (
        lambda: req.host_hit_length > 0 or req.mamba_host_hit_length > 0
    )
    req._compute_max_prefix_len = types.MethodType(
        extract(SCHEDULE, "_compute_max_prefix_len", {}), req
    )
    req.kv.holds_mamba = False
    req.kv.mamba_pool_idx = None
    return req


def track(cache, req, ctx):
    ns = dict(
        vars(runtime(ctx)), os=os, ForwardMode=Mode, _MambaRadixCacheV2TrackEntry=Entry
    )
    prepare = extract(SCHEDULE, "_mamba_radix_cache_v2_req_prepare_for_extend", ns)
    batch = NS(
        tree_cache=cache,
        req_to_token_pool=cache.req_to_token_pool,
        device="npu",
        forward_mode=Mode.EXTEND,
        model_config=NS(
            hf_text_config=NS(mamba_chunk_size=64),
            hf_config=NS(architectures=["Glm5NextForConditionalGeneration"]),
        ),
    )
    entry = prepare(batch, req)
    if entry.track_mask:
        # A depth + token-prefix fingerprint stands in for computed recurrent state.
        depth = req.kv.mamba_last_track_seqlen
        cache.req_to_token_pool.states[entry.track_index] = tuple(
            req.origin_input_ids[:depth]
        )
    return entry


class AddResult(Enum):
    CONTINUE = 1
    NO_TOKEN = 2
    OTHER = 3


class Adder:
    def __init__(self, cache, limit=8192):
        self.tree_cache = cache
        self.page_size = 64
        self.dllm_config = None
        self.rem_chunk_tokens = limit
        self.rem_total_tokens = self.rem_input_tokens = 1_000_000
        self.is_hybrid_swa = False
        self.prefill_delayer_single_pass = None
        self.prefill_max_requests = None
        self.can_run_list = []
        self.new_chunked_req = None
        self._mamba_branch_chunk_pending = False
        self.budgets = []
        self.hits = []

    def _update_prefill_budget(self, *args, **kwargs):
        self.budgets.append(args)

    def _mamba_gap_budget_for_req(self, req):
        return 0

    def ceil_paged_tokens(self, n):
        return (n + 63) // 64 * 64

    def _lock_node(self, node):
        return nullcontext()

    def _req_inc_lock_ref(self, req):
        pass

    def _check_prefill_tile_budget(self, length):
        return None

    def _account_prefill_cache_admission(self, req, length):
        self.hits.append(length)

    def budget_state(self):
        return AddResult.CONTINUE


for name in ("add_one_req", "add_chunked_req"):
    setattr(
        Adder,
        name,
        extract(
            POLICY,
            name,
            dict(
                branch_checkpoint_chunk_limit=split.branch_checkpoint_chunk_limit,
                CLIP_MAX_NEW_TOKENS=4096,
                AddReqResult=AddResult,
                torch=torch,
                InitLoadBackParams=N["InitLoadBackParams"],
            ),
        ),
    )


class BranchSplitTests(unittest.TestCase):
    def setUp(self):
        self.ctx = context()
        self.env = patch.dict(
            os.environ, {FLAG: "1", REUSABLE: "1", "SGLANG_MAMBA_RESTORE_TRACE": "0"}
        )
        self.modules = patch.dict(
            sys.modules, {"sglang.srt.runtime_context": runtime(self.ctx)}
        )
        self.env.start()
        self.modules.start()
        self.addCleanup(self.env.stop)
        self.addCleanup(self.modules.stop)

    def test_actual_tree_split_retains_child_state_but_not_parent_state(self):
        cache = Cache()
        end = cache.seed(range(4096)).last_device_node
        cache.state(end).host_value = torch.tensor([101], device="cpu")
        hit = cache.match(range(2304))
        self.assertEqual(hit.full_kv_hit_length, 2304)
        self.assertEqual(len(hit.device_indices), 0)
        self.assertEqual(hit.mamba_branching_seqlen, 2304)
        self.assertEqual(cache.state(end).value.tolist(), [1])
        self.assertEqual(cache.state(end).host_value.tolist(), [101])
        parent = cache.tree_core.node_by_id(end).parent
        self.assertIsNone(parent.component_data[CT.MAMBA].value)
        self.assertIsNone(parent.component_data[CT.MAMBA].host_value)

    def test_two_real_admissions_track_and_donate_branch_then_tail(self):
        cache = Cache()
        old = list(range(4096))
        old[2304:] = [x + 100000 for x in old[2304:]]
        old_node = cache.seed(old, 1).last_device_node
        req = request()
        hit = cache.match(req.origin_input_ids)
        self.assertEqual(hit.mamba_branching_seqlen, 2304)
        first = Adder(cache)
        self.assertEqual(first.add_one_req(req, False, None), AddResult.OTHER)
        self.assertEqual(req.extend_range.end, 2304)
        self.assertIs(first.new_chunked_req, req)
        self.assertEqual(first.budgets[0][:3], (0, 2304, 0))
        self.assertEqual(first.hits, [0])  # Computed branch is not a cache hit.
        track(cache, req, self.ctx)
        cache.cache_unfinished_req(req, chunked=True)
        branch_slot = cache.inserts[-1][1]
        self.assertEqual(cache.inserts[-1][0], 2304)
        self.assertIsNone(req.kv.mamba_last_track_seqlen)
        second = Adder(cache)
        self.assertIsNone(second.add_chunked_req(req))
        self.assertEqual((req.extend_range.start, req.extend_range.end), (2304, 7141))
        track(cache, req, self.ctx)
        cache.cache_unfinished_req(req)
        self.assertEqual(cache.inserts[-1][0], 6912)
        tail_slot = cache.inserts[-1][1]
        self.assertNotEqual(branch_slot, tail_slot)
        self.assertEqual(
            cache.req_to_token_pool.states[branch_slot],
            tuple(req.origin_input_ids[:2304]),
        )
        self.assertEqual(
            cache.req_to_token_pool.states[tail_slot],
            tuple(req.origin_input_ids[:6912]),
        )
        self.assertEqual(len(cache.match(req.origin_input_ids).device_indices), 6912)
        self.assertEqual(len(cache.match(old).device_indices), 4096)
        self.assertEqual(cache.state(old_node).value.tolist(), [1])
        other = list(req.origin_input_ids)
        other[2304:] = [x + 200000 for x in other[2304:]]
        self.assertEqual(len(cache.match(other).device_indices), 2304)
        continuation = list(req.origin_input_ids) + list(range(7141, 10662))
        self.assertEqual(len(cache.match(continuation).device_indices), 6912)

    def test_flag_off_reproduces_branch_only_insert(self):
        cache, req = Cache(), request()
        with patch.dict(os.environ, {FLAG: "0"}):
            self.assertEqual(
                Adder(cache).add_one_req(req, False, None), AddResult.CONTINUE
            )
        self.assertEqual(req.extend_range.end, 7141)
        track(cache, req, self.ctx)
        cache.cache_unfinished_req(req)
        self.assertEqual(cache.inserts[-1][0], 2304)
        self.assertEqual(len(cache.match(req.origin_input_ids).device_indices), 2304)

    def test_replay_sized_3853_lcp_retains_3840_branch_and_5888_tail(self):
        primer = list(range(6026))
        q1 = primer[:3853] + list(range(100000, 100000 + 6020 - 3853))
        q3 = q1 + list(range(200000, 200000 + 7397 - 6020))
        for enabled, expected in (("0", 3840), ("1", 5888)):
            with self.subTest(enabled=enabled), patch.dict(os.environ, {FLAG: enabled}):
                cache = Cache()
                cache.seed(primer[:5888])
                hit = cache.match(q1)
                self.assertEqual(hit.full_kv_hit_length, 3840)
                self.assertEqual(len(hit.device_indices), 0)
                self.assertEqual(hit.mamba_branching_seqlen, 3840)
                req = request(end=6020, branch=hit.mamba_branching_seqlen, tokens=q1)
                adder = Adder(cache)
                adder.add_one_req(req, False, None)
                track(cache, req, self.ctx)
                cache.cache_unfinished_req(req, chunked=adder.new_chunked_req is req)
                if adder.new_chunked_req is req:
                    self.assertIsNone(Adder(cache).add_chunked_req(req))
                    track(cache, req, self.ctx)
                    cache.cache_unfinished_req(req)
                self.assertEqual(len(cache.match(q3).device_indices), expected)
                self.assertEqual(len(cache.match(primer).device_indices), 5888)

    def test_existing_continuation_splits_without_second_chunk_owner(self):
        cache, req = Cache(), request(end=20000, prefix=8192, branch=10496)
        adder = Adder(cache)
        self.assertIs(adder.add_chunked_req(req), req)
        self.assertEqual(req.extend_range.end, 10496)
        self.assertEqual(adder.budgets[0][:3], (0, 2304, 0))
        self.assertEqual(adder.add_one_req(request(), True, None), AddResult.OTHER)
        self.assertEqual(adder.can_run_list, [req])
        self.assertIsNone(adder.new_chunked_req)

    def test_host_restore_precedes_split_decision(self):
        for evict_full in (False, True):
            with self.subTest(evict_full=evict_full):
                cache = Cache()
                node = cache.seed(range(6912)).last_device_node
                cache.backup_and_evict(node, full=evict_full)
                req = request()
                result = cache.match(req.origin_input_ids)
                self.assertEqual(
                    len(result.device_indices) + result.host_hit_length, 6912
                )
                self.assertEqual(result.mamba_host_hit_length, 1)
                self.assertIsNone(result.mamba_branching_seqlen)
                req.prefix_indices = result.device_indices
                req.last_node = result.last_device_node
                req.best_match_node = result.best_match_node
                req.host_hit_length = result.host_hit_length
                req.mamba_host_hit_length = result.mamba_host_hit_length
                # Deliberately stale branch hints must not split behind the restored prefix.
                adder = Adder(cache)
                self.assertEqual(
                    adder.add_one_req(req, False, None), AddResult.CONTINUE
                )
                self.assertEqual(req.host_loaded_length, 6912)
                self.assertEqual(req.extend_range.start, 6912)
                self.assertEqual(req.extend_range.end, 7141)
                self.assertIsNotNone(cache.state(node).value)
                self.assertIsNotNone(req.kv.mamba_pool_idx)
                self.assertEqual(adder.hits, [6912])

    def test_failed_host_restore_releases_slot_without_publishing_state(self):
        cache = Cache()
        node = cache.seed(range(6912)).last_device_node
        cache.backup_and_evict(node)
        cache.fail_load = True
        req = request()
        result = cache.match(req.origin_input_ids)
        req.last_node = result.last_device_node
        req.mamba_host_hit_length = result.mamba_host_hit_length
        indices, anchor = cache.init_load_back(
            N["InitLoadBackParams"](
                best_match_node=node, host_hit_length=result.host_hit_length, req=req
            )
        )
        self.assertEqual(len(indices), 0)
        self.assertEqual(anchor, result.last_device_node)
        self.assertIsNone(req.kv.mamba_pool_idx)
        self.assertEqual(cache.req_to_token_pool.freed, [30])
        self.assertIsNone(cache.state(node).value)
        self.assertEqual(len(cache.match(range(6912)).device_indices), 0)

    def test_host_split_and_missing_mamba_never_reuse_child_state(self):
        cache = Cache()
        node = cache.seed(range(4096)).last_device_node
        cache.backup_and_evict(node)
        result = cache.match(range(2304))
        self.assertEqual(result.full_kv_hit_length, 2304)
        self.assertEqual(result.host_hit_length, 0)
        self.assertEqual(result.mamba_host_hit_length, 0)
        parent = cache.tree_core.node_by_id(node).parent
        self.assertIsNone(parent.component_data[CT.MAMBA].host_value)
        self.assertIsNotNone(cache.state(node).host_value)
        cache.state(node).host_value = None
        result = cache.match(range(4096))
        self.assertEqual(result.full_kv_hit_length, 4096)
        self.assertEqual(len(result.device_indices) + result.host_hit_length, 0)

    def test_branch_donation_triggers_incremental_backup_without_counting_hit(self):
        cache = Cache()
        old = list(range(4096))
        old[2304:] = [x + 100000 for x in old[2304:]]
        node = cache.seed(old).last_device_node
        # A prior FULL+MAMBA host backup survives the key split on the child.
        core = cache.tree_core
        core.commit_backup(
            node,
            torch.arange(4096, device="cpu"),
            {
                CT.MAMBA: [
                    N["PoolTransfer"](
                        name=N["PoolName"].MAMBA,
                        host_indices=torch.tensor([901], device="cpu"),
                    )
                ]
            },
        )
        req = request()
        cache.match(req.origin_input_ids)
        adder = Adder(cache)
        adder.add_one_req(req, False, None)
        track(cache, req, self.ctx)
        cache.cache_unfinished_req(req, chunked=True)
        branch = core.node_by_id(req.last_node)
        self.assertEqual(branch.hit_count, 0)
        self.assertTrue(
            any(
                isinstance(a, N["BackupKV"]) and req.last_node in a.node_ids
                for a in cache.actions
            )
        )

    def test_exact_prompt_boundary_and_duplicate_branch_state(self):
        cache, req = Cache(), request(end=4096)
        branch_node = cache.seed(range(2304), slot=5).last_device_node
        adder = Adder(cache)
        adder.add_one_req(req, False, None)
        track(cache, req, self.ctx)
        cache.cache_unfinished_req(req, chunked=True)
        self.assertEqual(cache.state(branch_node).value.tolist(), [5])
        self.assertIn(10, cache.req_to_token_pool.freed)
        Adder(cache).add_chunked_req(req)
        track(cache, req, self.ctx)
        cache.cache_unfinished_req(req)
        self.assertEqual(cache.inserts[-1][0], 3840)
        limit = req._compute_max_prefix_len(4096)
        self.assertEqual(limit, 4095)
        self.assertEqual(
            len(cache.match(req.origin_input_ids[:limit]).device_indices), 3840
        )

    def test_384k_final_chunk_preserves_context_and_tail_depth(self):
        cache = Cache()
        req = request(end=393216, prefix=385024, branch=387328)
        first = Adder(cache)
        first.add_chunked_req(req)
        track(cache, req, self.ctx)
        self.assertEqual(req.kv.mamba_last_track_seqlen, 387328)
        req.prefix_indices = torch.arange(387328, dtype=torch.int64, device="cpu")
        second = Adder(cache)
        self.assertIsNone(second.add_chunked_req(req))
        track(cache, req, self.ctx)
        self.assertEqual(req.kv.mamba_last_track_seqlen, 392960)
        self.assertEqual(req.extend_range.end, 393216)
        self.assertEqual(len(req.origin_input_ids), 393216)
        self.assertEqual(req.output_ids, [])

    def test_natural_chunk_before_branch_keeps_budget_and_progress(self):
        cache, req = Cache(), request(end=20000, branch=10496)
        first = Adder(cache)
        self.assertEqual(first.add_one_req(req, False, None), AddResult.CONTINUE)
        self.assertEqual(req.extend_range.end, 8192)
        track(cache, req, self.ctx)
        cache.cache_unfinished_req(req, chunked=True)
        second = Adder(cache)
        self.assertIs(second.add_chunked_req(req), req)
        self.assertEqual(req.extend_range.end, 10496)
        track(cache, req, self.ctx)
        cache.cache_unfinished_req(req, chunked=True)
        self.assertEqual([x[0] for x in cache.inserts], [8192, 10496])

    def test_path_cap_preserves_shared_fork_and_tail(self):
        cache = Cache()
        old = list(range(4096))
        old[2304:] = [x + 100000 for x in old[2304:]]
        cache.seed(old)
        branch = cache.seed(range(2304), 2).last_device_node
        tail = cache.seed(range(6912), 3).last_device_node
        component = cache._components_tuple[1]
        component.mamba_max_states_per_path = 1
        frees = defaultdict(list)
        component._evict_excess_path_states(
            cache.tree_core.node_by_id(tail), frees, defaultdict(list)
        )
        self.assertEqual(dict(frees), {})
        self.assertEqual(cache.state(branch).value.tolist(), [2])
        self.assertEqual(cache.state(tail).value.tolist(), [3])

    def test_real_match_trace_distinguishes_full_mamba_and_host(self):
        cache = Cache()
        end = cache.seed(range(4096)).last_device_node
        req = request(end=2304)
        rows = []
        with patch.dict(
            N,
            trace_enabled=lambda req: req is not None,
            trace_event=lambda req, event, **fields: rows.append(
                (req.rid, event, fields)
            ),
        ):
            cache.match_prefix(
                N["MatchPrefixParams"](key=N["RadixKey"](req.origin_input_ids), req=req)
            )
            parent = cache.tree_core.node_by_id(end).parent
            cache.tree_core.commit_backup(
                parent.id, torch.arange(2304, device="cpu"), {}
            )
            cache.backup_and_evict(end)
            req = request(end=4096)
            cache.match_prefix(
                N["MatchPrefixParams"](key=N["RadixKey"](req.origin_input_ids), req=req)
            )
        self.assertEqual(rows[0][:2], ("sample", "match"))
        self.assertEqual(rows[0][2]["full_kv_depth"], 2304)
        self.assertEqual(rows[0][2]["best_mamba_depth"], 0)
        self.assertEqual(rows[0][2]["branching"], 2304)
        self.assertEqual(rows[1][2]["full_kv_depth"], 4096)
        self.assertEqual(rows[1][2]["best_mamba_depth"], 4096)
        self.assertTrue(rows[1][2]["mamba_host"])
        self.assertFalse(rows[1][2]["mamba_device"])
        self.assertEqual(rows[1][2]["mamba_host_hit"], 1)

    def test_align_and_no_progress_boundaries(self):
        cache = Cache()
        for prefix, branch, end, limit, expected in (
            (0, 256, 1025, 8192, 256),
            (64, 512, 1025, 8192, 448),
            (64, 256, 1025, 8192, None),
            (1, 512, 1025, 8192, None),
            (256, 256, 1025, 8192, None),
            (512, 256, 1025, 8192, None),
            (0, 257, 1025, 8192, None),
            (0, 512, 1025, 512, None),
            (0, 512, 1025, 256, None),
            (0, 512, 768, 8192, None),
            (0, 512, 769, 8192, 512),
            (0, 256, 1025, None, None),
        ):
            with self.subTest(prefix=prefix, branch=branch, end=end, limit=limit):
                req = request(end=end, prefix=prefix, branch=branch)
                self.assertEqual(
                    split.branch_checkpoint_chunk_limit(
                        cache, req, limit, physical_page_size=64
                    ),
                    expected,
                )
        self.assertIsNone(
            split.branch_checkpoint_chunk_limit(
                cache,
                request(),
                8192,
                physical_page_size=64,
                truncation_align_size=1024,
            )
        )

    def test_ineligible_profiles_and_decode_unchanged(self):
        cache, req = Cache(), request()
        for obj, attr, value in (
            (self.ctx.disagg, "disaggregation_mode", "decode"),
            (self.ctx.execution.mamba, "enable_mamba_extra_buffer_lazy", True),
            (self.ctx.execution.mamba, "enable_mamba_extra_buffer", False),
            (self.ctx.execution.overlap, "enable_two_batch_overlap", True),
            (self.ctx.execution.overlap, "enable_single_batch_overlap", True),
            (self.ctx.execution.deterministic, "enable_deterministic_inference", True),
            (self.ctx.schedule, "disable_overlap_schedule", False),
            (self.ctx.parallel, "pp_size", 2),
            (self.ctx.parallel, "attn_cp_size", 2),
            (self.ctx.parallel, "dcp_size", 2),
            (self.ctx.parallel, "enable_prefill_cp", True),
            (cache, "disable", True),
            (cache, "enable_mamba_extra_buffer", False),
            (cache, "glm53_kpool_share_page_size", 64),
            (cache, "page_size", 64),
            (req, "output_ids", [1]),
            (req, "return_logprob", True),
            (req, "multimodal_inputs", object()),
            (req, "input_embeds", object()),
        ):
            with self.subTest(attr=attr), patch.object(obj, attr, value):
                self.assertIsNone(
                    split.branch_checkpoint_chunk_limit(
                        cache, req, 8192, physical_page_size=64
                    )
                )


if __name__ == "__main__":
    unittest.main()
