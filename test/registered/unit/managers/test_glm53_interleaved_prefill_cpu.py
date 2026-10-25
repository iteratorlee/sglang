"""Commit-ready, standard-library-only standalone-prefill regression suite.

Place this single file at test/registered/unit/managers/test_repo_candidate.py.
Run: python -B test/registered/unit/managers/test_repo_candidate.py -v
No other test fixture, artifact directory, git executable or git revision is used.
For artifact-only validation, SGLANG_INTERLEAVED_TEST_ROOT may select a repository
copy containing the candidate. Otherwise ROOT is this file's parents[4].

Frozen reference methods come from 60afc9088c2742cc8392166a3cd0460a3d5f4518.
They supply negative controls and cold-math invariants, never the live allocation,
COW, donation, output, counter or prepared-KDA implementation under test.
CPU tensor/radix/transport storage remain fixtures; no device math is executed.
"""
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ast
import copy
import json
import hashlib
import os
from pathlib import Path
import sys
import unittest
from contextlib import contextmanager
from enum import Enum, auto
from types import SimpleNamespace as NS
from unittest.mock import patch


_override = os.environ.get("SGLANG_INTERLEAVED_TEST_ROOT")
ROOT = Path(_override).resolve() if _override else Path(__file__).resolve().parents[4]
if not (ROOT / "python/sglang/srt/managers/scheduler.py").is_file():
    raise RuntimeError(
        "Place this test at test/registered/unit/managers/test_repo_candidate.py "
        "or set SGLANG_INTERLEAVED_TEST_ROOT to a candidate repository copy."
    )
MANAGERS = ROOT / "python/sglang/srt/managers"
FLAG = "SGLANG_GLM53_INTERLEAVED_PREFILL"
TRACE = "SGLANG_GLM53_PREFILL_HOT_LANE_TRACE"
BRANCH = "SGLANG_GLM53_PD_MAMBA_BRANCH_SPLIT"
MEM_CACHE = MANAGERS.parent / "mem_cache"
MAMBA_CONSTANTS = {
    node.targets[0].id: ast.literal_eval(node.value)
    for node in ast.parse((MEM_CACHE / "common.py").read_text()).body
    if isinstance(node, ast.Assign)
    and isinstance(node.targets[0], ast.Name)
    and node.targets[0].id.startswith("MAMBA_STATE_PER_REQ_")
}


# Minimal immutable comparison baseline; it remains fixed after the feature commit.
FROZEN_BASELINE_METHODS = {'managers/scheduler.py': {'Scheduler': {'_get_new_batch_prefill_raw': 'def '
                                                                       '_get_new_batch_prefill_raw(self, '
                                                                       'prefill_delayer_single_pass: '
                                                                       'Optional[PrefillDelayerSinglePassExecutor], '
                                                                       'running_batch: '
                                                                       'ScheduleBatch) -> '
                                                                       'Tuple[Optional[ScheduleBatch], '
                                                                       'ScheduleBatch]:\n'
                                                                       '    if '
                                                                       'self.grammar_manager.has_waiting_grammars():\n'
                                                                       '        '
                                                                       'ready_grammar_requests = '
                                                                       'self.grammar_manager.get_ready_grammar_requests()\n'
                                                                       '        for req in '
                                                                       'ready_grammar_requests:\n'
                                                                       '            '
                                                                       'self._add_request_to_queue(req)\n'
                                                                       '    if '
                                                                       'self.enable_hierarchical_cache '
                                                                       'or '
                                                                       'get_memory().enable_flexkv '
                                                                       'or '
                                                                       'self.enable_unified_cache_external_linker:\n'
                                                                       '        '
                                                                       'self.tree_cache.check_hicache_events()\n'
                                                                       '        if '
                                                                       'self.enable_hicache_storage:\n'
                                                                       '            '
                                                                       'self._retry_missed_storage_prefetches()\n'
                                                                       '    if '
                                                                       'self.enable_priority_preemption '
                                                                       'or self.is_hybrid_swa:\n'
                                                                       '        '
                                                                       'running_batch.batch_is_full '
                                                                       '= False\n'
                                                                       '    if '
                                                                       '(running_batch.batch_is_full '
                                                                       'or len(self.waiting_queue) '
                                                                       '== 0) and self.chunked_req '
                                                                       'is None:\n'
                                                                       '        return (None, '
                                                                       'running_batch)\n'
                                                                       '    running_bs = '
                                                                       'len(running_batch.reqs)\n'
                                                                       '    if '
                                                                       'self.min_free_slots_delayer '
                                                                       'is not None and '
                                                                       'self.chunked_req is None '
                                                                       'and '
                                                                       'self.min_free_slots_delayer.should_delay(running_bs=running_bs, '
                                                                       'num_allocatable_reqs=self.get_num_allocatable_reqs(running_bs, '
                                                                       'running_batch=running_batch)):\n'
                                                                       '        return (None, '
                                                                       'running_batch)\n'
                                                                       '    if '
                                                                       'self.get_num_allocatable_reqs(running_bs, '
                                                                       'running_batch=running_batch) '
                                                                       '<= 0 and self.chunked_req '
                                                                       'is None and (not '
                                                                       'self.enable_priority_preemption):\n'
                                                                       '        '
                                                                       'running_batch.batch_is_full '
                                                                       '= True\n'
                                                                       '        return (None, '
                                                                       'running_batch)\n'
                                                                       '    '
                                                                       'self.policy.calc_priority(self.waiting_queue, '
                                                                       'running_batch, '
                                                                       'processed_tokens=self.processed_tokens_counter)\n'
                                                                       '    if TEST_RETRACT and '
                                                                       'running_bs > '
                                                                       'TEST_RETRACT_NO_PREFILL_BS:\n'
                                                                       '        return (None, '
                                                                       'running_batch)\n'
                                                                       '    chunked_prefill_size = '
                                                                       'self.chunked_prefill_size\n'
                                                                       '    if self.chunked_req is '
                                                                       'not None and '
                                                                       'self.dynamic_chunk_sizer '
                                                                       'is not None:\n'
                                                                       '        history_len = '
                                                                       'len(self.chunked_req.prefix_indices)\n'
                                                                       '        dynamic_size = '
                                                                       'self.dynamic_chunk_sizer.predict(history_len)\n'
                                                                       '        if dynamic_size is '
                                                                       'not None:\n'
                                                                       '            '
                                                                       'chunked_prefill_size = '
                                                                       'dynamic_size\n'
                                                                       '    attn_backend = '
                                                                       'self.tp_worker.model_runner.attn_backend\n'
                                                                       '    if '
                                                                       'hasattr(attn_backend, '
                                                                       "'extend_attention_block_m'):\n"
                                                                       '        '
                                                                       'prefill_tile_block_m = '
                                                                       'attn_backend.extend_attention_block_m\n'
                                                                       '    else:\n'
                                                                       '        '
                                                                       'prefill_tile_block_m = 64\n'
                                                                       '    adder = '
                                                                       'PrefillAdder(self.page_size, '
                                                                       'self.tree_cache, '
                                                                       'self.token_to_kv_pool_allocator, '
                                                                       'running_batch, '
                                                                       'self.new_token_ratio_tracker.current, '
                                                                       'self.max_prefill_tokens, '
                                                                       'chunked_prefill_size, '
                                                                       'running_bs if '
                                                                       'self.is_mixed_chunk else '
                                                                       '0, '
                                                                       'self.priority_scheduling_preemption_threshold, '
                                                                       'max_prefill_bs=int(self.max_prefill_bs), '
                                                                       'max_running_requests=self.max_running_requests, '
                                                                       'prefill_max_requests=get_schedule().prefill_max_requests, '
                                                                       'prefill_delayer_single_pass=prefill_delayer_single_pass, '
                                                                       'dllm_config=self.dllm_config, '
                                                                       'waiting_queue_len=len(self.waiting_queue), '
                                                                       'prefill_tile_block_m=prefill_tile_block_m)\n'
                                                                       '    if self.chunked_req is '
                                                                       'not None:\n'
                                                                       '        '
                                                                       'self.chunked_req.init_next_round_input()\n'
                                                                       '        self.chunked_req = '
                                                                       'adder.add_chunked_req(self.chunked_req)\n'
                                                                       '    if self.enable_lora:\n'
                                                                       '        running_loras = '
                                                                       '{req.lora_id for req in '
                                                                       'running_batch.reqs if not '
                                                                       'req.finished()}\n'
                                                                       '        '
                                                                       'running_loras.update((req.lora_id '
                                                                       'for req in '
                                                                       'adder.can_run_list))\n'
                                                                       '        if '
                                                                       'self.lora_drainer:\n'
                                                                       '            '
                                                                       'self.lora_drainer.update_draining_state(self.waiting_queue, '
                                                                       'running_batch.reqs)\n'
                                                                       '    mamba_allocator = '
                                                                       'getattr(self.req_to_token_pool, '
                                                                       "'mamba_allocator', None)\n"
                                                                       '    if mamba_allocator is '
                                                                       'not None:\n'
                                                                       '        '
                                                                       'mamba_allocator.alloc_group_begin(len(self.waiting_queue))\n'
                                                                       '    jit_prefetch = '
                                                                       'self._use_glm53_jit_prefetch()\n'
                                                                       '    jit_prefetch_owner = '
                                                                       'None\n'
                                                                       '    prefill_candidates = '
                                                                       'self.waiting_queue\n'
                                                                       '    if jit_prefetch:\n'
                                                                       '        jit_prefetch_owner '
                                                                       '= next((r for r in '
                                                                       'self.waiting_queue if '
                                                                       'r.pd_prefill_jit_prefetch_issued), '
                                                                       'None)\n'
                                                                       '        if '
                                                                       'jit_prefetch_owner is not '
                                                                       'None:\n'
                                                                       '            '
                                                                       'prefill_candidates = '
                                                                       '[jit_prefetch_owner] + [r '
                                                                       'for r in '
                                                                       'self.waiting_queue if r is '
                                                                       'not jit_prefetch_owner]\n'
                                                                       '    for req in '
                                                                       'prefill_candidates:\n'
                                                                       '        if '
                                                                       'self.enable_lora and (not '
                                                                       'self._can_schedule_lora_req(req, '
                                                                       'running_loras)):\n'
                                                                       '            continue\n'
                                                                       '        running_bs = '
                                                                       'len(running_batch.reqs)\n'
                                                                       '        '
                                                                       'candidate_beam_width = '
                                                                       'req.beam_group.beam_width '
                                                                       'if req.beam_group is not '
                                                                       'None else None\n'
                                                                       '        reusing_req_slots '
                                                                       '= sum((r.kv.holds_kv for r '
                                                                       'in adder.can_run_list))\n'
                                                                       '        if '
                                                                       'len(adder.can_run_list) >= '
                                                                       'self.get_num_allocatable_reqs(running_bs, '
                                                                       'candidate_beam_width, '
                                                                       'running_batch=running_batch, '
                                                                       'reusing_req_slots=reusing_req_slots):\n'
                                                                       '            '
                                                                       'running_batch.batch_is_full '
                                                                       '= True\n'
                                                                       '        if '
                                                                       'self.disaggregation_mode '
                                                                       '== '
                                                                       'DisaggregationMode.PREFILL:\n'
                                                                       '            if '
                                                                       'len(adder.can_run_list) - '
                                                                       'reusing_req_slots >= '
                                                                       'self.req_to_token_pool.available_size():\n'
                                                                       '                '
                                                                       'running_batch.batch_is_full '
                                                                       '= True\n'
                                                                       '        if '
                                                                       'running_batch.batch_is_full:\n'
                                                                       '            if not '
                                                                       'self.enable_priority_preemption '
                                                                       'or not '
                                                                       'adder.preempt_to_schedule(req):\n'
                                                                       '                break\n'
                                                                       '        if jit_prefetch:\n'
                                                                       '            if '
                                                                       'jit_prefetch_owner is not '
                                                                       'None and req is not '
                                                                       'jit_prefetch_owner or '
                                                                       'adder.budget_state() != '
                                                                       'AddReqResult.CONTINUE or '
                                                                       '(adder.prefill_max_requests '
                                                                       'is not None and '
                                                                       'len(adder.can_run_list) >= '
                                                                       'adder.prefill_max_requests):\n'
                                                                       '                break\n'
                                                                       '            if not '
                                                                       'req.pd_prefill_jit_prefetch_issued:\n'
                                                                       '                '
                                                                       'req.pd_prefill_jit_prefetch_issued '
                                                                       '= True\n'
                                                                       '                '
                                                                       'jit_prefetch_owner = req\n'
                                                                       '                '
                                                                       'self._prefetch_kvcache(req, '
                                                                       'at_admission=True)\n'
                                                                       '                if req.rid '
                                                                       'in '
                                                                       'self.tree_cache.ongoing_prefetch:\n'
                                                                       '                    break\n'
                                                                       '        if '
                                                                       'self.enable_hicache_storage:\n'
                                                                       '            prefetch_done '
                                                                       '= '
                                                                       'self.tree_cache.check_prefetch_progress(req.rid)\n'
                                                                       '            if not '
                                                                       'prefetch_done:\n'
                                                                       '                if '
                                                                       'jit_prefetch:\n'
                                                                       '                    break\n'
                                                                       '                continue\n'
                                                                       '            '
                                                                       '(loaded_tokens, '
                                                                       'loaded_start) = '
                                                                       'self.tree_cache.pop_prefetch_loaded_span(req.rid)\n'
                                                                       '            if '
                                                                       'loaded_tokens > 0:\n'
                                                                       '                '
                                                                       'req.storage_hit_length = '
                                                                       'loaded_tokens\n'
                                                                       '                '
                                                                       'req.storage_hit_start = '
                                                                       'loaded_start\n'
                                                                       '                '
                                                                       'req.host_hit_is_storage = '
                                                                       'False\n'
                                                                       '        '
                                                                       'req.init_next_round_input(self.tree_cache)\n'
                                                                       '        if '
                                                                       'self.enable_hicache_storage '
                                                                       'and '
                                                                       'get_memory().hicache_host_memory_mode '
                                                                       "== 'buffer_only':\n"
                                                                       '            (held_tokens, '
                                                                       'held_swa_tokens) = '
                                                                       'self.tree_cache.plan_staged_splice(req.rid, '
                                                                       'len(req.prefix_indices))\n'
                                                                       '            if held_tokens '
                                                                       '> 0:\n'
                                                                       '                '
                                                                       'req.host_hit_length = '
                                                                       'held_tokens\n'
                                                                       '                '
                                                                       'req.swa_host_hit_length = '
                                                                       'held_swa_tokens\n'
                                                                       '                '
                                                                       'req.storage_hit_length = '
                                                                       'held_tokens\n'
                                                                       '                '
                                                                       'req.storage_hit_start = '
                                                                       'len(req.prefix_indices)\n'
                                                                       '                '
                                                                       'req.host_hit_is_storage = '
                                                                       'True\n'
                                                                       '            elif not '
                                                                       '(req.host_hit_is_storage '
                                                                       'and req.host_loaded_length '
                                                                       '> 0):\n'
                                                                       '                '
                                                                       'req.storage_hit_length = '
                                                                       '0\n'
                                                                       '                '
                                                                       'req.storage_hit_start = '
                                                                       'None\n'
                                                                       '                '
                                                                       'req.host_hit_is_storage = '
                                                                       'False\n'
                                                                       '        res = '
                                                                       'adder.add_one_req(req, '
                                                                       'has_chunked_req=self.chunked_req '
                                                                       'is not None, '
                                                                       'truncation_align_size=self.truncation_align_size)\n'
                                                                       '        if jit_prefetch '
                                                                       'and adder.can_run_list and '
                                                                       '(req is '
                                                                       'adder.can_run_list[-1]):\n'
                                                                       '            '
                                                                       'jit_prefetch_owner = None\n'
                                                                       '        if '
                                                                       'self.enable_lora:\n'
                                                                       '            '
                                                                       'running_loras.add(req.lora_id)\n'
                                                                       '        if res != '
                                                                       'AddReqResult.CONTINUE:\n'
                                                                       '            if res == '
                                                                       'AddReqResult.NO_TOKEN:\n'
                                                                       '                if '
                                                                       'self.enable_hierarchical_cache '
                                                                       'or '
                                                                       'self.enable_unified_cache_external_linker:\n'
                                                                       '                    '
                                                                       'running_batch.batch_is_full '
                                                                       '= len(adder.can_run_list) '
                                                                       '> 0 or not '
                                                                       'running_batch.is_empty()\n'
                                                                       '                else:\n'
                                                                       '                    '
                                                                       'running_batch.batch_is_full '
                                                                       '= True\n'
                                                                       '            added = '
                                                                       'len(adder.can_run_list) > '
                                                                       '0 and req is '
                                                                       'adder.can_run_list[-1]\n'
                                                                       '            if not added:\n'
                                                                       '                '
                                                                       'req.kv.mamba_cow_src_index '
                                                                       '= None\n'
                                                                       '                '
                                                                       'req.kv.mamba_needs_clear = '
                                                                       'False\n'
                                                                       '                if '
                                                                       'req.kv.holds_mamba and '
                                                                       '(not getattr(req, '
                                                                       "'session', None)):\n"
                                                                       '                    '
                                                                       'self.tree_cache.req_to_token_pool.mamba_allocator.free(req.kv.mamba_pool_idx.unsqueeze(-1))\n'
                                                                       '                    '
                                                                       'req.kv.mamba_pool_idx = '
                                                                       'None\n'
                                                                       '            break\n'
                                                                       '    if mamba_allocator is '
                                                                       'not None:\n'
                                                                       '        '
                                                                       'mamba_allocator.alloc_group_end()\n'
                                                                       '    can_run_list: '
                                                                       'List[Req] = '
                                                                       'adder.can_run_list\n'
                                                                       '    if len(can_run_list) '
                                                                       '== 0:\n'
                                                                       '        return (None, '
                                                                       'running_batch)\n'
                                                                       '    can_run_set = '
                                                                       'set(can_run_list)\n'
                                                                       '    self.waiting_queue = '
                                                                       '[x for x in '
                                                                       'self.waiting_queue if x '
                                                                       'not in can_run_set]\n'
                                                                       '    if '
                                                                       'adder.preempt_list:\n'
                                                                       '        for req in '
                                                                       'adder.preempt_list:\n'
                                                                       '            '
                                                                       'self._add_request_to_queue(req)\n'
                                                                       '    if '
                                                                       'adder.new_chunked_req is '
                                                                       'not None:\n'
                                                                       '        assert '
                                                                       'self.chunked_req is None\n'
                                                                       '        self.chunked_req = '
                                                                       'adder.new_chunked_req\n'
                                                                       '    if self.chunked_req is '
                                                                       'not None:\n'
                                                                       '        '
                                                                       'self.chunked_req.inflight_middle_chunks '
                                                                       '+= 1\n'
                                                                       '    '
                                                                       'set_time_batch(can_run_list, '
                                                                       "'set_forward_entry_time')\n"
                                                                       '    new_batch = '
                                                                       'ScheduleBatch.init_new(can_run_list, '
                                                                       'self.req_to_token_pool, '
                                                                       'self.token_to_kv_pool_allocator, '
                                                                       'self.tree_cache, '
                                                                       'self.model_config, '
                                                                       'self.enable_overlap, '
                                                                       'self.spec_algorithm, '
                                                                       'chunked_req=self.chunked_req)\n'
                                                                       '    '
                                                                       'new_batch.contains_last_prefill_chunk '
                                                                       '= self.chunked_req is None '
                                                                       'or len(can_run_list) != 1\n'
                                                                       '    if '
                                                                       'self.enable_hierarchical_cache '
                                                                       'or '
                                                                       'self.enable_unified_cache_external_linker:\n'
                                                                       '        '
                                                                       'new_batch.hicache_consumer_index '
                                                                       '= '
                                                                       'self.tree_cache.ready_to_load_host_cache()\n'
                                                                       '    '
                                                                       'new_batch.prepare_for_extend()\n'
                                                                       '    if '
                                                                       'self.tp_worker.model_runner.prefill_aware_swa:\n'
                                                                       '        for req in '
                                                                       'can_run_list:\n'
                                                                       '            '
                                                                       'req.kv.swa_evict_floor = '
                                                                       'req.extend_range.end\n'
                                                                       '    '
                                                                       'new_batch.prefill_stats = '
                                                                       'PrefillStats.from_adder(adder, '
                                                                       'running_batch.reqs, '
                                                                       'self.enable_priority_scheduling, '
                                                                       'num_pending_tokens=self.load_inquirer._get_num_pending_tokens(chunk_deduct=self.chunked_req.extend_range.length '
                                                                       'if self.chunked_req is not '
                                                                       'None else 0))\n'
                                                                       '    if self.is_mixed_chunk '
                                                                       'and (not '
                                                                       'running_batch.is_empty()) '
                                                                       'and (not '
                                                                       '(new_batch.return_logprob '
                                                                       'or '
                                                                       'running_batch.return_logprob)) '
                                                                       'and '
                                                                       '(new_batch.input_embeds is '
                                                                       'None) and '
                                                                       'all((r.beam_group is None '
                                                                       'for r in '
                                                                       'running_batch.reqs)):\n'
                                                                       '        '
                                                                       'running_batch.filter_batch()\n'
                                                                       '        if not '
                                                                       'running_batch.is_empty():\n'
                                                                       '            '
                                                                       'running_batch.prepare_for_decode()\n'
                                                                       '            '
                                                                       'new_batch.mix_with_running(running_batch)\n'
                                                                       '            '
                                                                       'new_batch.decoding_reqs = '
                                                                       'running_batch.reqs\n'
                                                                       '            if not '
                                                                       'self.enable_overlap and '
                                                                       '(not '
                                                                       'self.spec_algorithm.is_none()):\n'
                                                                       '                '
                                                                       'last_tokens = '
                                                                       'torch.tensor([r.output_ids[-1] '
                                                                       'for r in '
                                                                       'running_batch.reqs], '
                                                                       'dtype=torch.int64, '
                                                                       'device=self.device)\n'
                                                                       '                '
                                                                       'self.future_map.stash_bonus_tokens(running_batch.req_pool_indices, '
                                                                       'last_tokens)\n'
                                                                       '            running_batch '
                                                                       '= ScheduleBatch(reqs=[], '
                                                                       'batch_is_full=running_batch.batch_is_full)\n'
                                                                       '    else:\n'
                                                                       '        '
                                                                       'new_batch.decoding_reqs = '
                                                                       'None\n'
                                                                       '    return (new_batch, '
                                                                       'running_batch)'}},
 'managers/schedule_policy.py': {'PrefillAdder': {'add_chunked_req': 'def add_chunked_req(self, '
                                                                     'req: Req):\n'
                                                                     '    if self.dllm_config is '
                                                                     'not None:\n'
                                                                     '        _rem_tokens = '
                                                                     'self._get_dllm_remain_tokens()\n'
                                                                     '    else:\n'
                                                                     '        _rem_tokens = '
                                                                     'min(self.rem_chunk_tokens, '
                                                                     'int(self.rem_total_tokens))\n'
                                                                     '        if '
                                                                     'self.is_hybrid_swa and (not '
                                                                     'self._swa_req_ring):\n'
                                                                     '            _rem_tokens = '
                                                                     'min(_rem_tokens, '
                                                                     'int(self.rem_swa_tokens) - '
                                                                     'self.page_size)\n'
                                                                     '        if _rem_tokens <= '
                                                                     '0:\n'
                                                                     '            if '
                                                                     'self.is_hybrid_swa:\n'
                                                                     '                return req\n'
                                                                     '            _rem_tokens = '
                                                                     'self.rem_chunk_tokens\n'
                                                                     '    if '
                                                                     'self.prefill_delayer_single_pass '
                                                                     'is not None:\n'
                                                                     '        '
                                                                     'self.prefill_delayer_single_pass.negotiate_should_allow_prefill(local_prefillable=True, '
                                                                     'running_batch=self.running_batch.batch_size(), '
                                                                     'max_prefill_bs=self.max_prefill_bs, '
                                                                     'max_running_requests=self.max_running_requests, '
                                                                     'waiting_queue_len=self.waiting_queue_len)\n'
                                                                     '    cand_extend_input_len = '
                                                                     'len(req.full_untruncated_fill_ids) '
                                                                     '- len(req.prefix_indices)\n'
                                                                     '    branch_limit = None\n'
                                                                     '    if self.dllm_config is '
                                                                     'None:\n'
                                                                     '        branch_limit = '
                                                                     'branch_checkpoint_chunk_limit(self.tree_cache, '
                                                                     'req, _rem_tokens, '
                                                                     'physical_page_size=self.page_size)\n'
                                                                     '        if branch_limit is '
                                                                     'not None:\n'
                                                                     '            _rem_tokens = '
                                                                     'branch_limit\n'
                                                                     '    truncated = '
                                                                     'cand_extend_input_len > '
                                                                     '_rem_tokens\n'
                                                                     '    new_len = '
                                                                     'min(cand_extend_input_len, '
                                                                     '_rem_tokens)\n'
                                                                     '    '
                                                                     'req.set_extend_range(len(req.prefix_indices), '
                                                                     'len(req.prefix_indices) + '
                                                                     'new_len)\n'
                                                                     '    '
                                                                     'self.can_run_list.append(req)\n'
                                                                     '    '
                                                                     'self._mamba_branch_chunk_pending '
                                                                     '= branch_limit is not None\n'
                                                                     '    '
                                                                     'self._update_prefill_budget(0, '
                                                                     'req.extend_range.length, '
                                                                     'min(req.sampling_params.max_new_tokens, '
                                                                     'CLIP_MAX_NEW_TOKENS) if not '
                                                                     'truncated else 0, '
                                                                     'req.retracted_stain, '
                                                                     'mamba_gap_reserve=self._mamba_gap_budget_for_req(req), '
                                                                     'is_chunked_continuation=True)\n'
                                                                     '    return req if truncated '
                                                                     'else None',
                                                  'add_one_req': 'def add_one_req(self, req: Req, '
                                                                 'has_chunked_req: bool, '
                                                                 'truncation_align_size: '
                                                                 'Optional[int]):\n'
                                                                 '    if '
                                                                 'self._mamba_branch_chunk_pending:\n'
                                                                 '        return '
                                                                 'AddReqResult.OTHER\n'
                                                                 '    if (x := '
                                                                 'self.prefill_max_requests) is '
                                                                 'not None and '
                                                                 'len(self.can_run_list) >= x:\n'
                                                                 '        return '
                                                                 'AddReqResult.OTHER\n'
                                                                 '    if '
                                                                 'req.sampling_params.ignore_eos '
                                                                 'and getattr(self.tree_cache, '
                                                                 "'disable', True):\n"
                                                                 '        return '
                                                                 'self.add_one_req_ignore_eos(req)\n'
                                                                 '    max_new = '
                                                                 'min(max(req.sampling_params.max_new_tokens '
                                                                 '- len(req.output_ids), 0), '
                                                                 'CLIP_MAX_NEW_TOKENS)\n'
                                                                 '    cand_extend_input_len = '
                                                                 'len(req.full_untruncated_fill_ids) '
                                                                 '- len(req.prefix_indices)\n'
                                                                 '    total_tokens = '
                                                                 'cand_extend_input_len + max_new '
                                                                 '+ self.page_size\n'
                                                                 '    mamba_gap_reserve = '
                                                                 'self._mamba_gap_budget_for_req(req)\n'
                                                                 '    total_tokens += '
                                                                 'mamba_gap_reserve\n'
                                                                 '    real_input_tokens = '
                                                                 'cand_extend_input_len - '
                                                                 'req.host_hit_length\n'
                                                                 '    real_input_tokens = '
                                                                 'self.ceil_paged_tokens(real_input_tokens)\n'
                                                                 '    prefix_len = '
                                                                 'len(req.prefix_indices)\n'
                                                                 '    if total_tokens >= '
                                                                 'self.rem_total_tokens:\n'
                                                                 '        return '
                                                                 'AddReqResult.NO_TOKEN\n'
                                                                 '    chunk_tokens_limit = '
                                                                 'self.rem_chunk_tokens\n'
                                                                 '    if self.is_hybrid_swa:\n'
                                                                 '        swa_needed = '
                                                                 'self._swa_budget_for_req(real_input_tokens, '
                                                                 'self._swa_new_tokens(req), '
                                                                 'swa_host_hit_length=req.swa_host_hit_length)\n'
                                                                 '        if swa_needed > '
                                                                 'self.rem_swa_tokens if '
                                                                 'self._swa_req_ring else '
                                                                 'swa_needed >= '
                                                                 'self.rem_swa_tokens:\n'
                                                                 '            if not '
                                                                 'self._swa_req_never_fits(real_input_tokens, '
                                                                 'self._swa_new_tokens(req), '
                                                                 'req.swa_host_hit_length):\n'
                                                                 '                return '
                                                                 'AddReqResult.NO_TOKEN\n'
                                                                 '            swa_cap = '
                                                                 'self._swa_chunk_cap(self._swa_new_tokens(req), '
                                                                 'req.swa_host_hit_length)\n'
                                                                 '            if '
                                                                 'self.rem_chunk_tokens is None or '
                                                                 'swa_cap <= 0:\n'
                                                                 '                return '
                                                                 'AddReqResult.NO_TOKEN\n'
                                                                 '            chunk_tokens_limit = '
                                                                 'min(self.rem_chunk_tokens, '
                                                                 'swa_cap)\n'
                                                                 '    if self.rem_chunk_tokens is '
                                                                 'None and len(self.can_run_list) '
                                                                 '!= 0 and (real_input_tokens >= '
                                                                 'self.rem_input_tokens):\n'
                                                                 '        return '
                                                                 'AddReqResult.OTHER\n'
                                                                 '    with '
                                                                 'self._lock_node(req.last_node):\n'
                                                                 '        if total_tokens >= '
                                                                 'self.rem_total_tokens:\n'
                                                                 '            return '
                                                                 'AddReqResult.NO_TOKEN\n'
                                                                 '        if self.is_hybrid_swa:\n'
                                                                 '            swa_needed = '
                                                                 'self._swa_budget_for_req(real_input_tokens, '
                                                                 'self._swa_new_tokens(req), '
                                                                 'swa_host_hit_length=req.swa_host_hit_length)\n'
                                                                 '            if swa_needed > '
                                                                 'self.rem_swa_tokens if '
                                                                 'self._swa_req_ring else '
                                                                 'swa_needed >= '
                                                                 'self.rem_swa_tokens:\n'
                                                                 '                if not '
                                                                 'self._swa_req_never_fits(real_input_tokens, '
                                                                 'self._swa_new_tokens(req), '
                                                                 'req.swa_host_hit_length):\n'
                                                                 '                    return '
                                                                 'AddReqResult.NO_TOKEN\n'
                                                                 '                swa_cap = '
                                                                 'self._swa_chunk_cap(self._swa_new_tokens(req), '
                                                                 'req.swa_host_hit_length)\n'
                                                                 '                if '
                                                                 'self.rem_chunk_tokens is None or '
                                                                 'swa_cap <= 0:\n'
                                                                 '                    return '
                                                                 'AddReqResult.NO_TOKEN\n'
                                                                 '                '
                                                                 'chunk_tokens_limit = '
                                                                 'min(self.rem_chunk_tokens, '
                                                                 'swa_cap)\n'
                                                                 '        if '
                                                                 'self.prefill_delayer_single_pass '
                                                                 'is not None and (not '
                                                                 'self.prefill_delayer_single_pass.negotiate_should_allow_prefill(local_prefillable=True, '
                                                                 'running_batch=self.running_batch.batch_size(), '
                                                                 'max_prefill_bs=self.max_prefill_bs, '
                                                                 'max_running_requests=self.max_running_requests, '
                                                                 'waiting_queue_len=self.waiting_queue_len)):\n'
                                                                 '            return '
                                                                 'AddReqResult.OTHER\n'
                                                                 '        if '
                                                                 'req.needs_host_load_back():\n'
                                                                 '            (new_indices, '
                                                                 'req.last_node) = '
                                                                 'self.tree_cache.init_load_back(InitLoadBackParams(best_match_node=req.best_match_node, '
                                                                 'host_hit_length=req.host_hit_length, '
                                                                 'req=req))\n'
                                                                 '            '
                                                                 'req.host_loaded_length = '
                                                                 'len(new_indices)\n'
                                                                 '            req.prefix_indices = '
                                                                 'torch.cat([req.prefix_indices, '
                                                                 'new_indices])\n'
                                                                 '            prefix_len = '
                                                                 'len(req.prefix_indices)\n'
                                                                 '            '
                                                                 'req.kv.cache_protected_len = '
                                                                 'prefix_len\n'
                                                                 '        input_tokens = '
                                                                 'self.ceil_paged_tokens(len(req.full_untruncated_fill_ids) '
                                                                 '- len(req.prefix_indices))\n'
                                                                 '        branch_limit = None\n'
                                                                 '        if self.dllm_config is '
                                                                 'None and (not has_chunked_req):\n'
                                                                 '            branch_limit = '
                                                                 'branch_checkpoint_chunk_limit(self.tree_cache, '
                                                                 'req, chunk_tokens_limit, '
                                                                 'physical_page_size=self.page_size, '
                                                                 'truncation_align_size=truncation_align_size)\n'
                                                                 '            if branch_limit is '
                                                                 'not None:\n'
                                                                 '                '
                                                                 'chunk_tokens_limit = '
                                                                 'branch_limit\n'
                                                                 '        if self.rem_chunk_tokens '
                                                                 'is None and '
                                                                 'len(self.can_run_list) != 0 and '
                                                                 '(input_tokens >= '
                                                                 'self.rem_input_tokens):\n'
                                                                 '            return '
                                                                 'AddReqResult.OTHER\n'
                                                                 '        if self.dllm_config is '
                                                                 'not None:\n'
                                                                 '            if '
                                                                 'self.rem_dllm_tokens <= 0:\n'
                                                                 '                return '
                                                                 'AddReqResult.OTHER\n'
                                                                 '            assert '
                                                                 'truncation_align_size is None, '
                                                                 "'truncation_align_size is not "
                                                                 "supported for dllm prefill'\n"
                                                                 '            if (tile_stop := '
                                                                 'self._check_prefill_tile_budget(input_tokens)) '
                                                                 'is not None:\n'
                                                                 '                return '
                                                                 'tile_stop\n'
                                                                 '            '
                                                                 'self._add_dllm_req(req, '
                                                                 'prefix_len)\n'
                                                                 '            '
                                                                 'self._req_inc_lock_ref(req)\n'
                                                                 '        elif chunk_tokens_limit '
                                                                 'is None or input_tokens <= '
                                                                 'chunk_tokens_limit:\n'
                                                                 '            if (tile_stop := '
                                                                 'self._check_prefill_tile_budget(input_tokens)) '
                                                                 'is not None:\n'
                                                                 '                return '
                                                                 'tile_stop\n'
                                                                 '            '
                                                                 'req.set_extend_range(len(req.prefix_indices), '
                                                                 'len(req.full_untruncated_fill_ids))\n'
                                                                 '            '
                                                                 'self.can_run_list.append(req)\n'
                                                                 '            '
                                                                 'self._req_inc_lock_ref(req)\n'
                                                                 '            '
                                                                 'self._update_prefill_budget(prefix_len, '
                                                                 'req.extend_range.length, '
                                                                 'min(req.sampling_params.max_new_tokens, '
                                                                 'CLIP_MAX_NEW_TOKENS), '
                                                                 'req.retracted_stain, '
                                                                 'mamba_gap_reserve=mamba_gap_reserve)\n'
                                                                 '            '
                                                                 'self._account_prefill_cache_admission(req, '
                                                                 'prefix_len)\n'
                                                                 '        else:\n'
                                                                 '            trunc_len = '
                                                                 'chunk_tokens_limit // '
                                                                 'self.page_size * self.page_size\n'
                                                                 '            if trunc_len <= 0:\n'
                                                                 '                return '
                                                                 'AddReqResult.OTHER\n'
                                                                 '            if '
                                                                 'truncation_align_size is not '
                                                                 'None:\n'
                                                                 '                if trunc_len < '
                                                                 'truncation_align_size:\n'
                                                                 '                    return '
                                                                 'AddReqResult.OTHER\n'
                                                                 '                else:\n'
                                                                 '                    trunc_len = '
                                                                 'truncation_align_size * '
                                                                 '(trunc_len // '
                                                                 'truncation_align_size)\n'
                                                                 '            now_input_len = '
                                                                 'trunc_len + '
                                                                 'len(req.prefix_indices)\n'
                                                                 '            now_input_len = '
                                                                 'now_input_len // self.page_size '
                                                                 '* self.page_size\n'
                                                                 '            trunc_len = '
                                                                 'now_input_len - '
                                                                 'len(req.prefix_indices)\n'
                                                                 '            if trunc_len <= 0:\n'
                                                                 '                return '
                                                                 'AddReqResult.OTHER\n'
                                                                 '            if (tile_stop := '
                                                                 'self._check_prefill_tile_budget(trunc_len)) '
                                                                 'is not None:\n'
                                                                 '                return '
                                                                 'tile_stop\n'
                                                                 '            '
                                                                 'req.set_extend_range(len(req.prefix_indices), '
                                                                 'len(req.prefix_indices) + '
                                                                 'trunc_len)\n'
                                                                 '            '
                                                                 'self.can_run_list.append(req)\n'
                                                                 '            self.new_chunked_req '
                                                                 '= req\n'
                                                                 '            '
                                                                 'self._mamba_branch_chunk_pending '
                                                                 '= branch_limit is not None\n'
                                                                 '            '
                                                                 'self._req_inc_lock_ref(req)\n'
                                                                 '            '
                                                                 'self._update_prefill_budget(prefix_len, '
                                                                 'trunc_len, 0, '
                                                                 'req.retracted_stain, '
                                                                 'mamba_gap_reserve=mamba_gap_reserve)\n'
                                                                 '            '
                                                                 'self._account_prefill_cache_admission(req, '
                                                                 'prefix_len)\n'
                                                                 '    if '
                                                                 'self._mamba_branch_chunk_pending:\n'
                                                                 '        return '
                                                                 'AddReqResult.OTHER\n'
                                                                 '    return self.budget_state()',
                                                  '_update_prefill_budget': 'def '
                                                                            '_update_prefill_budget(self, '
                                                                            'prefix_len: int, '
                                                                            'extend_input_len: '
                                                                            'int, max_new_tokens: '
                                                                            'int, retracted_stain: '
                                                                            'bool, '
                                                                            'mamba_gap_reserve: '
                                                                            'int=0, '
                                                                            'is_chunked_continuation: '
                                                                            'bool=False):\n'
                                                                            '    '
                                                                            'raw_extend_input_len '
                                                                            '= extend_input_len\n'
                                                                            '    extend_input_len '
                                                                            '= '
                                                                            'self.ceil_paged_tokens(extend_input_len)\n'
                                                                            '    page_overhead = '
                                                                            'self.page_size\n'
                                                                            '    '
                                                                            'self.rem_total_token_offset '
                                                                            '+= extend_input_len + '
                                                                            'max_new_tokens + '
                                                                            'page_overhead + '
                                                                            'mamba_gap_reserve\n'
                                                                            '    '
                                                                            'self.cur_rem_token_offset '
                                                                            '+= extend_input_len + '
                                                                            'page_overhead + '
                                                                            'mamba_gap_reserve\n'
                                                                            '    if '
                                                                            'mamba_gap_reserve and '
                                                                            'self.rem_mamba_slots '
                                                                            'is not None:\n'
                                                                            '        '
                                                                            'self.rem_mamba_slots '
                                                                            '-= 1\n'
                                                                            '    '
                                                                            'self.rem_input_tokens '
                                                                            '-= extend_input_len\n'
                                                                            '    if '
                                                                            'self.is_hybrid_swa:\n'
                                                                            '        if not '
                                                                            '(self._swa_req_ring '
                                                                            'and '
                                                                            'is_chunked_continuation):\n'
                                                                            '            '
                                                                            'self.rem_swa_token_offset '
                                                                            '+= '
                                                                            'self._swa_budget_for_req(extend_input_len, '
                                                                            'max_new_tokens)\n'
                                                                            '    if '
                                                                            'self.dllm_config is '
                                                                            'not None:\n'
                                                                            '        '
                                                                            'self.rem_dllm_tokens '
                                                                            '-= extend_input_len\n'
                                                                            '    elif '
                                                                            'self.rem_chunk_tokens '
                                                                            'is not None:\n'
                                                                            '        '
                                                                            'self.rem_chunk_tokens '
                                                                            '-= extend_input_len\n'
                                                                            '    '
                                                                            'self.log_hit_tokens '
                                                                            '+= prefix_len\n'
                                                                            '    '
                                                                            'self.log_input_tokens '
                                                                            '+= '
                                                                            'raw_extend_input_len\n'
                                                                            '    if '
                                                                            'retracted_stain:\n'
                                                                            '        '
                                                                            'self.reprocessed_log_hit_tokens '
                                                                            '+= prefix_len\n'
                                                                            '        '
                                                                            'self.reprocessed_log_input_tokens '
                                                                            '+= '
                                                                            'raw_extend_input_len'}},
 'disaggregation/prefill.py': {'SchedulerDisaggregationPrefillMixin': {'process_prefill_chunk': 'def '
                                                                                                'process_prefill_chunk(self: '
                                                                                                'Scheduler, '
                                                                                                'last_batch: '
                                                                                                'Optional[ScheduleBatch], '
                                                                                                'running_batch: '
                                                                                                'ScheduleBatch) '
                                                                                                '-> '
                                                                                                'None:\n'
                                                                                                '    '
                                                                                                'chunked_req_to_exclude '
                                                                                                '= '
                                                                                                'set()\n'
                                                                                                '    '
                                                                                                'if '
                                                                                                '(req '
                                                                                                ':= '
                                                                                                'self.chunked_req) '
                                                                                                'is '
                                                                                                'not '
                                                                                                'None:\n'
                                                                                                '        '
                                                                                                'chunked_req_to_exclude.add(req)\n'
                                                                                                '        '
                                                                                                'maybe_cache_unfinished_req(req, '
                                                                                                'self.tree_cache, '
                                                                                                'chunked=True)\n'
                                                                                                '        '
                                                                                                'if '
                                                                                                'not '
                                                                                                'self.check_bootstrap(req):\n'
                                                                                                '            '
                                                                                                'if '
                                                                                                'is_aborted(req):\n'
                                                                                                '                '
                                                                                                'self.chunked_req '
                                                                                                '= '
                                                                                                'None\n'
                                                                                                '            '
                                                                                                'elif '
                                                                                                'self.has_bootstrapped_waiting_req():\n'
                                                                                                '                '
                                                                                                'self.chunked_req '
                                                                                                '= '
                                                                                                'None\n'
                                                                                                '                '
                                                                                                'if '
                                                                                                'not '
                                                                                                'self.enable_overlap:\n'
                                                                                                '                    '
                                                                                                'self.optimistic_release_and_requeue(req)\n'
                                                                                                '        '
                                                                                                'elif '
                                                                                                'self.enable_overlap:\n'
                                                                                                '            '
                                                                                                'req.tmp_end_idx '
                                                                                                '= '
                                                                                                'min(req.extend_range.end, '
                                                                                                'len(req.origin_input_ids))\n'
                                                                                                '        '
                                                                                                'else:\n'
                                                                                                '            '
                                                                                                'self.send_kv_chunk(req)\n'
                                                                                                '        '
                                                                                                'if '
                                                                                                'self.chunked_req '
                                                                                                'is '
                                                                                                'not '
                                                                                                'None:\n'
                                                                                                '            '
                                                                                                'running_batch.batch_is_full '
                                                                                                '= '
                                                                                                'False\n'
                                                                                                '    '
                                                                                                'if '
                                                                                                'last_batch '
                                                                                                'and '
                                                                                                'last_batch.forward_mode.is_extend():\n'
                                                                                                '        '
                                                                                                'if '
                                                                                                'last_batch.chunked_req:\n'
                                                                                                '            '
                                                                                                'chunked_req_to_exclude.add(last_batch.chunked_req)\n'
                                                                                                '        '
                                                                                                'last_bs '
                                                                                                '= '
                                                                                                'last_batch.batch_size()\n'
                                                                                                '        '
                                                                                                'last_batch.filter_batch(chunked_req_to_exclude=list(chunked_req_to_exclude))\n'
                                                                                                '        '
                                                                                                'if '
                                                                                                'last_batch.batch_size() '
                                                                                                '< '
                                                                                                'last_bs:\n'
                                                                                                '            '
                                                                                                'running_batch.batch_is_full '
                                                                                                '= '
                                                                                                'False'}}}
FROZEN_KDA_SHA256 = "4805a645f27cc3aab14e50bdca2e95ca0dbf01d6a7ae43ec77e363f9ab905577"


def read_srt_source(relative, baseline=False):
    text = (ROOT / "python/sglang/srt" / relative).read_text()
    if not baseline or relative not in FROZEN_BASELINE_METHODS:
        return text
    tree = ast.parse(text)
    for cls in tree.body:
        methods = FROZEN_BASELINE_METHODS[relative].get(getattr(cls, "name", None))
        if methods is None:
            continue
        for i, node in enumerate(cls.body):
            if getattr(node, "name", None) in methods:
                cls.body[i] = ast.parse(methods[node.name]).body[0]
    return ast.unparse(tree)


def source(name, baseline=False):
    return read_srt_source("managers/" + name, baseline)


def extract(text, names, namespace, methods=None):
    nodes = []
    for node in ast.parse(text).body:
        if getattr(node, "name", None) in names:
            node = copy.deepcopy(node)
            if methods is not None:
                node.bases = []
                node.body = [
                    n for n in node.body if getattr(n, "name", None) in methods
                ]
            nodes.append(node)
    tree = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(
        compile(ast.fix_missing_locations(tree), "<production AST>", "exec"), namespace
    )


class Slot(int):
    def unsqueeze(self, _):
        return [self]


class Kv:
    def __init__(self):
        self.req_pool_idx = None
        self.mamba_pool_idx = None
        self.mamba_ping_pong_track_buffer = None
        self.mamba_cow_src_index = None
        self.mamba_needs_clear = False
        self.cache_protected_len = 0

    @property
    def holds_kv(self):
        return self.req_pool_idx is not None

    @property
    def holds_mamba(self):
        return self.mamba_pool_idx is not None


class MambaPool:
    def __init__(self, capacity=64, cost=128):
        self.capacity, self.cost = capacity, cost
        self.live = set()
        self.group = []
        self.events = []

    def schedulable_available_size(self):
        return self.capacity - len(self.live)

    def alloc(self, count):
        if count == 1 and self.group:
            return [self.group.pop(0)]
        assert self.schedulable_available_size() >= count, "Mamba overflow"
        slots = [Slot(i) for i in range(self.capacity) if i not in self.live][:count]
        self.live.update(slots)
        return slots

    def free(self, slots):
        for slot in slots:
            assert slot in self.live, "double free"
            self.live.remove(slot)

    def alloc_group_begin(self, count):
        assert not self.group
        self.events.append(("begin", count))
        if count <= self.schedulable_available_size():
            self.group = self.alloc(count)

    def alloc_group_end(self):
        self.free(self.group)
        self.group = []
        self.events.append(("end",))


class UnifiedMambaTokenToKVPoolAllocator:
    def __init__(self, pool, tokens=1_000_000):
        self.mamba_allocator = pool
        self.tokens = tokens
        self.allocated_kv = 0

    def available_size(self):
        return (
            self.tokens
            - self.allocated_kv
            - len(self.mamba_allocator.live) * self.mamba_allocator.cost
        )

    def mamba_slot_full_token_cost(self):
        return self.mamba_allocator.cost


class StaticTokenAllocator:
    """Separate fixed KV/Mamba pools, as opposed to shared-gap accounting."""

    def __init__(self, pool, tokens=1_000_000):
        self.tokens = tokens
        self.allocated_kv = 0

    def available_size(self):
        return self.tokens - self.allocated_kv


class UnifiedRadixCache:
    __module__ = "sglang.srt.mem_cache.unified_radix_cache"

    def __init__(self, slots=64, rows=16, tokens=1_000_000):
        self.disable = False
        self.page_size = 256
        self.glm53_kpool_share_page_size = 256
        self.enable_mamba_extra_buffer = True
        self.host_memory_mode = "cache"
        self.mamba = MambaPool(slots)
        self.allocator = StaticTokenAllocator(self.mamba, tokens)
        self.rows = rows
        self.used_rows = 0
        self.req_to_token_pool = NS(
            size=rows,
            mamba_allocator=self.mamba,
            mamba_ping_pong_track_buffer_size=1,
            available_size=lambda: self.rows - self.used_rows,
        )
        self.ongoing_prefetch = {}
        self.prefetch_loaded_tokens_by_reqid = {}
        self.events = []
        self.nodes = []
        self.evictable = 0
        self.mamba_evictable = 0
        self.lock_drop = 0

    def supports_mamba(self):
        return True

    def is_tree_cache(self):
        return True

    def full_evictable_size(self):
        return max(
            0, self.evictable - sum(n.refs > 0 for n in self.nodes) * self.lock_drop
        )

    def mamba_evictable_size(self):
        return max(0, self.mamba_evictable - sum(n.refs > 0 for n in self.nodes))

    def inc_lock_ref(self, node):
        node.refs += 1
        self.events.append(("lock", node.rid))
        return NS(to_dec_params=lambda: None)

    def dec_lock_ref(self, node, *_):
        node.refs -= 1
        assert node.refs >= 0, "unbalanced lock"
        self.events.append(("unlock", node.rid))

    def check_hicache_events(self):
        self.events.append(("hicache_events",))

    def check_prefetch_progress(self, rid):
        self.events.append(("prefetch_poll", rid))
        return rid not in self.ongoing_prefetch

    def pop_prefetch_loaded_span(self, rid):
        self.events.append(("prefetch_pop", rid))
        return self.prefetch_loaded_tokens_by_reqid.pop(rid, 0), None

    def init_load_back(self, params):
        req = params.req
        self.events.append(("restore", req.rid))
        return list(range(req.host_hit_length)), req.last_node

    def finish_storage_prefetch_admission(self, rid, **kwargs):
        self.events.append(("admit", rid, tuple(sorted(kwargs.items()))))

    def discard_storage_prefetch_accounting(self, rid):
        self.events.append(("discard", rid))


class Req:
    def __init__(self, cache, rid="hot", prefix=4096, extend=128):
        self.rid = rid
        self.hit = prefix
        self.prefix_indices = list(range(prefix))
        self.full_untruncated_fill_ids = range(prefix + extend)
        self.output_ids = []
        self.origin_input_ids = self.full_untruncated_fill_ids
        self.pending_bootstrap = False
        self.return_sampling_mask = False
        self.grammar = None
        self.init_calls = 0
        self.sampling_params = NS(max_new_tokens=8, ignore_eos=False)
        self.host_hit_length = self.swa_host_hit_length = self.mamba_host_hit_length = 0
        self.host_loaded_length = self.storage_hit_length = 0
        self.storage_hit_start = None
        self.host_hit_is_storage = False
        self.return_logprob = False
        self.multimodal_inputs = self.input_embeds = self.beam_group = self.session = (
            None
        )
        self.mamba_branching_seqlen = None
        self.retracted_stain = False
        self.kv = Kv()
        self.last_node = NS(refs=0, rid=rid)
        self.best_match_node = self.last_node
        cache.nodes.append(self.last_node)
        self.inflight_middle_chunks = 0
        self.pd_prefill_jit_prefetch_issued = False
        self.after_cow = lambda req, tree: None
        self.set_extend_range(prefix, prefix)

    def set_extend_range(self, start, end):
        self.extend_range = NS(start=start, end=end, length=end - start)

    def init_next_round_input(self, tree=None, cow_mamba=None):
        self.init_calls += 1
        if tree is None:
            return
        tree.events.append(("match", self.rid, cow_mamba))
        self.prefix_indices = list(range(self.hit))
        if cow_mamba is not False and self.hit and not self.mamba_host_hit_length:
            if not self.kv.holds_mamba:
                self.kv.mamba_pool_idx = tree.mamba.alloc(1)[0]
            self.kv.mamba_cow_src_index = 999
            self.after_cow(self, tree)

    def finished(self):
        return getattr(self, "aborted", False)

    def needs_host_load_back(self):
        return bool(
            self.host_hit_length
            or self.swa_host_hit_length
            or self.mamba_host_hit_length
        )

    def materialized_host_hit_len(self):
        return min(self.host_hit_length, self.host_loaded_length)

    def fulfilled_storage_hit_len(self, prefix):
        return min(prefix, self.storage_hit_length, self.host_loaded_length)


class Batch:
    def __init__(self, reqs=None, **kwargs):
        self.reqs = reqs or []
        self.batch_is_full = False
        self.__dict__.update(kwargs)

    def batch_size(self):
        return len(self.reqs)

    def is_empty(self):
        return not self.reqs

    @classmethod
    def init_new(cls, reqs, pool, allocator, tree, *args, **kwargs):
        return cls(reqs, tree=tree, **kwargs)

    def prepare_for_extend(self):
        tree = self.tree
        for req in self.reqs:
            if not req.kv.holds_kv:
                assert tree.req_to_token_pool.available_size() > 0, "row overflow"
                req.kv.req_pool_idx = req.rid
                tree.used_rows += 1
            if not req.kv.holds_mamba:
                req.kv.mamba_pool_idx = tree.mamba.alloc(1)[0]
            if req.kv.mamba_ping_pong_track_buffer is None:
                req.kv.mamba_ping_pong_track_buffer = tree.mamba.alloc(1)
            req.kv.mamba_cow_src_index = None
            req.kv.mamba_needs_clear = False
            charge = (req.extend_range.length + 63) // 64 * 64 + 64
            req.kv.kv_charge = getattr(req.kv, "kv_charge", 0) + charge
            tree.allocator.allocated_kv += charge
            assert tree.allocator.available_size() >= 0, "KV overflow"


def load(baseline=False):
    ns = dict(
        **MAMBA_CONSTANTS,
        os=os,
        EvictParams=lambda **kw: NS(**kw),
        json=json,
        logger=NS(info=lambda *args: None),
        Enum=Enum,
        auto=auto,
        contextmanager=contextmanager,
        CLIP_MAX_NEW_TOKENS=4096,
        _IS_HIP=False,
        _is_npu=True,
        TEST_RETRACT=False,
        _GLM53_PD_PREFILL_JIT_PREFETCH=False,
        is_swa_req_ring=lambda _: False,
        ScheduleBatch=Batch,
        UnifiedMambaTokenToKVPoolAllocator=UnifiedMambaTokenToKVPoolAllocator,
        InitLoadBackParams=lambda **kwargs: NS(**kwargs),
        torch=NS(cat=lambda seqs: sum(seqs, [])),
        set_time_batch=lambda *args: None,
        DisaggregationMode=NS(PREFILL="prefill"),
        PrefillStats=NS(from_adder=lambda adder, *args, **kwargs: adder),
    )
    for name in (
        "SWATokenToKVPoolAllocator",
        "PureSWATokenToKVPoolAllocator",
        "DeepSeekV4HiSparseTokenToKVPoolAllocator",
        "UnifiedMambaSWATokenToKVPoolAllocator",
    ):
        ns[name] = type(name, (), {})
    extract(
        source("mamba_branch_prefill.py", baseline),
        {"branch_checkpoint_chunk_limit"},
        ns,
    )
    extract(source("schedule_batch.py", baseline), {"split_cached_prefix_by_tier"}, ns)
    extract(
        source("schedule_policy.py", baseline), {"AddReqResult", "PrefillAdder"}, ns
    )
    extract(
        source("scheduler.py", baseline),
        {"Scheduler"},
        ns,
        methods={
            "_get_new_batch_prefill_raw",
            "get_num_allocatable_reqs",
            "get_internal_state",
            "_glm53_interleaved_prefill_enabled",
            "_try_glm53_interleaved_prefill",
            
            "_use_glm53_jit_prefetch",
        },
    )
    return ns


CURRENT = load()
BASELINE = load(True)


class Harness:
    def __init__(
        self, namespace=CURRENT, *, chunk=16384, hard=32768, slots=64, rows=16
    ):
        self.ns = namespace
        self.cache = UnifiedRadixCache(slots=slots, rows=rows)
        self.schedule = NS(disable_overlap_schedule=True, prefill_max_requests=None)
        self.execution = NS(
            mamba=NS(
                enable_mamba_extra_buffer=True, enable_mamba_extra_buffer_lazy=False
            ),
            overlap=NS(
                enable_two_batch_overlap=False, enable_single_batch_overlap=False
            ),
            deterministic=NS(enable_deterministic_inference=False),
        )
        self.parallel = NS(
            pp_size=1,
            attn_cp_size=1,
            dcp_size=1,
            enable_prefill_cp=False,
            pp_max_micro_batch_size=16,
        )
        self.runtime = NS(
            get_exec=lambda: self.execution,
            get_parallel=lambda: self.parallel,
            get_schedule=lambda: self.schedule,
            get_disagg=lambda: NS(disaggregation_mode="prefill"),
            mamba_cache_chunk_size=lambda: 64,
            mamba_checkpoint_grid=lambda _: 256,
        )
        namespace.update(vars(self.runtime))
        namespace["get_memory"] = lambda: NS(
            enable_flexkv=False, hicache_host_memory_mode="cache"
        )
        self.scheduler = s = namespace["Scheduler"]()
        s.__dict__.update(
            ps=NS(attn_tp_rank=0, attn_dp_rank=1),
            tree_cache=self.cache,
            page_size=64,
            grammar_manager=NS(has_waiting_grammars=lambda: False),
            enable_hierarchical_cache=False,
            enable_hicache_storage=False,
            enable_unified_cache_external_linker=False,
            enable_priority_scheduling=False,
            enable_priority_preemption=False,
            is_hybrid_swa=False,
            waiting_queue=[],
            chunked_req=None,
            min_free_slots_delayer=None,
            policy=NS(calc_priority=lambda *a, **k: None),
            schedule_policy="fcfs",
            processed_tokens_counter=0,
            chunked_prefill_size=chunk,
            dynamic_chunk_sizer=None,
            tp_worker=NS(
                model_runner=NS(attn_backend=object(), prefill_aware_swa=False)
            ),
            token_to_kv_pool_allocator=self.cache.allocator,
            new_token_ratio_tracker=NS(current=1.0),
            max_prefill_tokens=hard,
            is_mixed_chunk=False,
            priority_scheduling_preemption_threshold=0,
            max_prefill_bs=0,  # Production default: no prefill-delayer cap.
            max_running_requests=16,
            dllm_config=None,
            enable_lora=False,
            req_to_token_pool=self.cache.req_to_token_pool,
            beam_coordinator=NS(pending_member_rows=lambda batch: 0),
            disaggregation_mode="prefill",
            truncation_align_size=None,
            model_config=NS(
                hf_config=NS(architectures=["Glm5NextForConditionalGeneration"])
            ),
            enable_overlap=False,
            spec_algorithm=None,
            load_inquirer=NS(_get_num_pending_tokens=lambda **_: 0),
            _retry_missed_storage_prefetches=lambda: None,
        )
        self.running = Batch()
        s.running_batch = self.running

    def cold(self, length=65536, prefix=16384):
        req = Req(self.cache, "owner", prefix, length - prefix)
        req.kv.req_pool_idx = "owner"
        self.cache.used_rows += 1
        req.kv.mamba_pool_idx = self.cache.mamba.alloc(1)[0]
        req.kv.mamba_ping_pong_track_buffer = self.cache.mamba.alloc(1)
        self.scheduler.chunked_req = req
        req.last_node.refs = 1
        return req

    def hot(self, rid="hot", extend=128, prefix=4096):
        req = Req(self.cache, rid, prefix, extend)
        self.scheduler.waiting_queue.append(req)
        return req

    def step(self, enabled="1", delayer=None, trace=None):
        # Each namespace keeps runtime accessors separate; reinstall for tests
        # which interleave multiple harnesses of the same extracted class.
        self.ns.update(vars(self.runtime))
        environment = {BRANCH: "1"}
        if enabled is not None:
            environment[FLAG] = enabled
        if trace is not None:
            environment[TRACE] = trace
        with (
            patch.dict(os.environ, environment),
            patch.dict(sys.modules, {"sglang.srt.runtime_context": self.runtime}),
        ):
            if enabled is None:
                os.environ.pop(FLAG, None)
            if trace is None:
                os.environ.pop(TRACE, None)
            return self.scheduler._get_new_batch_prefill_raw(delayer, self.running)[0]



class CpuScalar(int):
    def unsqueeze(self, _):
        return CpuTensor([self])


class CpuTensor(list):
    """Only tensor storage/indexing; allocation and donation policy is real AST."""

    dtype = "int64"
    device = "cpu"

    def __getitem__(self, key):
        value = super().__getitem__(key)
        return CpuTensor(value) if isinstance(key, slice) else value

    def clone(self):
        return CpuTensor(self)

    def to(self, **_):
        return self

    def numel(self):
        return len(self)

    def split(self, count):
        return [self[i : i + count] for i in range(0, len(self), count)]


class CpuRows(dict):
    def __setitem__(self, key, value):
        if isinstance(key, list):
            for row, item in zip(key, value, strict=True):
                self[row] = item
        else:
            super().__setitem__(key, copy.copy(value))


def load_lifecycle():
    ns = dict(
        **MAMBA_CONSTANTS,
        _MAMBA_DEBUG_ASSERTS=False,
        EvictParams=lambda **kw: NS(**kw),
        torch=NS(
            int32="int32",
            int64="int64",
            arange=lambda start, end, **_: CpuTensor(map(CpuScalar, range(start, end))),
            full=lambda shape, value, **_: CpuTensor([CpuScalar(value)] * shape[0]),
            stack=lambda values: CpuTensor(values),
            cat=lambda values: CpuTensor(x for value in values for x in value),
        ),
    )
    extract((MEM_CACHE / "allocator/mamba.py").read_text(), {"MambaSlotAllocator"}, ns)
    extract(
        (MEM_CACHE / "memory_pool.py").read_text(),
        {"ReqToTokenPool"},
        ns,
        methods={"alloc", "available_size"},
    )
    extract(
        (MEM_CACHE / "memory_pool.py").read_text(),
        {"HybridReqToTokenPool"},
        ns,
        methods={
            "alloc",
            "_alloc_ping_pong_buffer",
            "get_mamba_ping_pong_other_idx",
            "get_mamba_ping_pong_keep_idx",
            "set_mamba_ping_pong_slot",
            "donate_mamba_ping_pong_slot",
        },
    )
    # Restore the production inheritance needed by HybridReqToTokenPool.alloc.
    ns["CpuHybridPool"] = type(
        "CpuHybridPool", (ns["HybridReqToTokenPool"], ns["ReqToTokenPool"]), {}
    )
    extract((MEM_CACHE / "allocation.py").read_text(), {"alloc_req_slots"}, ns)
    extract(
        (MEM_CACHE / "unified_cache/components/mamba.py").read_text(),
        {"MambaComponent"},
        ns,
        methods={"_alloc_mamba_slot", "int8_ckpt_pool", "prepare_for_caching_req", "finalize_match_result_in_cache"},
    )
    return ns


LIFECYCLE = load_lifecycle()


def lifecycle_harness(free_slots, shared):
    h = Harness()

    class PhysicalPool(LIFECYCLE["MambaSlotAllocator"]):
        cost = 128

        @property
        def live(self):
            return set(range(1, self.size + 1)) - set(self.free_slots)

        def schedulable_available_size(self):
            slots = super().schedulable_available_size()
            if shared:
                slots = min(slots, h.cache.allocator.available_size() // self.cost)
            return slots

    # Two owner slots and two pinned HBM COW sources are already occupied.
    h.cache.mamba = PhysicalPool(free_slots + 4, "cpu")
    allocator_cls = (
        UnifiedMambaTokenToKVPoolAllocator if shared else StaticTokenAllocator
    )
    h.cache.allocator = allocator_cls(h.cache.mamba)
    if shared:
        # Tight bytes for owner KV, two hot suffixes, four live/track states,
        # nine retained headroom states, and strict max-new reservations.
        h.cache.allocator.tokens = 4 * 128 + 16448 + 2 * 320 + 4 * 128 + 9 * 128 + 17
    h.scheduler.token_to_kv_pool_allocator = h.cache.allocator
    pool = LIFECYCLE["CpuHybridPool"]()
    pool.mamba_allocator = h.cache.mamba
    pool.mamba_pool = NS(replayssm_write_pos=None)
    pool.enable_mamba_extra_buffer = True
    pool.enable_mamba_extra_buffer_lazy = False
    pool.mamba_ping_pong_track_buffer_size = 1
    pool.req_index_to_mamba_index_mapping = CpuRows()
    pool.req_index_to_mamba_ping_pong_track_buffer_mapping = CpuRows()
    pool.size = 16
    pool.free_slots = list(range(2, 17))

    def alloc_rows(count):
        if count > len(pool.free_slots):
            return None
        rows = pool.free_slots[:count]
        del pool.free_slots[:count]
        return rows

    pool.alloc_rows = alloc_rows  # CPU row storage, real parent alloc binds Req.
    h.cache.req_to_token_pool = h.scheduler.req_to_token_pool = pool
    owner = h.cold()
    owner.kv.req_pool_idx = 1
    owner.kv.kv_allocated_len = 16384
    hits = [h.hot(rid, extend=256) for rid in ("a", "b")]
    sources = h.cache.mamba.alloc(2)
    for req, src in zip(hits, sources, strict=True):
        req.after_cow = lambda r, _, src=src: setattr(r.kv, "mamba_cow_src_index", src)
    evictions = []
    h.cache.evict_for_alloc = lambda params: evictions.append(params)
    component = LIFECYCLE["MambaComponent"]()
    component.cache = h.cache

    def prepare(batch):
        # Run production blanket-eviction policy, row reuse, live/track allocation
        # and row->state mapping. Only KV tensor allocation is represented by bytes.
        LIFECYCLE["alloc_req_slots"](pool, batch.reqs, h.cache)
        for req in batch.reqs:
            charge = (req.extend_range.length + 63) // 64 * 64 + 64
            h.cache.allocator.allocated_kv += charge
            req.kv.kv_allocated_len = req.extend_range.end
            req.kv.mamba_last_track_idx = 0
            req.kv.mamba_last_track_seqlen = req.extend_range.end
            req.kv.kv_charge = charge

    return h, component, prepare, sources, evictions




def load_output(baseline=False):
    ns = dict(is_aborted=lambda r: getattr(r, 'aborted', False),
              should_force_retry=lambda r: False)
    extract(read_srt_source('mem_cache/common.py'), {'maybe_cache_unfinished_req'}, ns)
    extract(read_srt_source('disaggregation/prefill.py', baseline),
            {'SchedulerDisaggregationPrefillMixin'}, ns,
            methods={'process_prefill_chunk', 'process_batch_result_disagg_prefill'})
    return ns['SchedulerDisaggregationPrefillMixin']

OUTPUT = load_output()


def connect_output(h, component):
    s = h.scheduler
    donations, sends, checks = [], [], []
    def cache(req, **kwargs):
        # Radix storage is a fixture. Donation/allocation and its invocation from
        # P output/chunk lifecycle are the actual, unmodified production methods.
        params = NS(mamba_value=None)
        component.prepare_for_caching_req(req, params, req.extend_range.end, False)
        donations.append((req.rid, params.mamba_value))
        req.prefix_indices = list(range(req.extend_range.end))
    h.cache.cache_unfinished_req = cache
    s.check_bootstrap = lambda req: checks.append(req.rid) or not req.pending_bootstrap
    s.has_bootstrapped_waiting_req = lambda: bool(s.waiting_queue)
    s.optimistic_release_and_requeue = lambda req: (_ for _ in ()).throw(AssertionError('unexpected requeue'))
    s.send_kv_chunk = lambda req, **kw: sends.append((req.rid, kw))
    s.spec_algorithm = NS(is_eagle=lambda: False)
    s.disagg_prefill_inflight_queue = []
    s.batch_result_processor = NS(
        snapshot_auxiliary_output_starts=lambda *a: None,
        move_logprobs_to_cpu=lambda **k: None)
    s.metrics_reporter = NS(report_prefill_stats=lambda **k: None)
    s.maybe_send_health_check_signal = lambda: None
    for req in [s.chunked_req, *s.waiting_queue]:
        req.time_stats = NS(**{k: lambda: None for k in (
            'set_prefill_finished_time', 'set_prefill_transfer_queue_entry_time',
            'set_last_chunked_prefill_finish_time')})
    return donations, sends, checks


def process_output(h, batch):
    batch.spec_info = None
    batch.dp_cooperation_info = None
    result = NS(logits_output=None, next_token_ids=NS(tolist=lambda: [42]*len(batch.reqs)),
        extend_input_len_per_req=None, extend_logprob_start_len_per_req=None,
        copy_done=None, auxiliary_host_output=None, routed_experts_output=None,
        indexer_topk_output=None, next_draft_input=None, can_run_cuda_graph=False)
    OUTPUT.process_batch_result_disagg_prefill(h.scheduler, batch, result)


def physical(free=4, evictable=0):
    h, comp, prepare, sources, events = lifecycle_harness(free+evictable, False)
    # All other slots are pinned except these explicitly evictable checkpoints.
    reclaim = h.cache.mamba.alloc(evictable) if evictable else CpuTensor()
    h.cache.mamba_evictable_size = lambda: len(reclaim)
    def evict(params):
        events.append(params.mamba_num)
        count=min(params.mamba_num, len(reclaim))
        if count:
            h.cache.mamba.free(CpuTensor(reclaim[:count]))
            del reclaim[:count]
    h.cache.evict_for_alloc = evict
    comp.component_type = 'mamba'
    comp.tree_core = NS(get_component_device_value=lambda node, _: node.source)
    for req, src in zip(h.scheduler.waiting_queue, sources):
        req.last_node.source = CpuTensor([src])
        def match(tree=None, cow_mamba=None, req=req):
            req.init_calls += 1
            if tree is not None:
                comp.finalize_match_result_in_cache(
                    NS(cow_mamba=cow_mamba is not False, req=req),
                    NS(best_match_node=req.last_node))
        req.init_next_round_input = match
    return h, comp, prepare, sources, events


class CandidateTests(unittest.TestCase):
    def test_real_allocation_output_donation_and_forced_owner(self):
        for free, evictable in [(4, 0), (0, 4), (1, 3)]:
            with self.subTest(free=free, evictable=evictable):
                h,c,prepare,sources,evictions=physical(free,evictable)
                owner=h.scheduler.chunked_req
                before=(owner.extend_range,owner.init_calls,owner.kv.mamba_pool_idx,
                        owner.kv.mamba_ping_pong_track_buffer[:],owner.last_node.refs)
                donations,sends,checks=connect_output(h,c)
                with patch.object(Batch,'prepare_for_extend',prepare):
                    hot=h.step()
                    self.assertEqual([r.rid for r in hot.reqs],['a'])
                    self.assertIsNone(hot.chunked_req)
                    self.assertTrue(hot.contains_last_prefill_chunk)
                    self.assertIs(h.scheduler.chunked_req,owner)
                    self.assertEqual(owner.inflight_middle_chunks,0)
                    self.assertEqual(before,(owner.extend_range,owner.init_calls,
                        owner.kv.mamba_pool_idx,owner.kv.mamba_ping_pong_track_buffer[:],owner.last_node.refs))
                    self.assertEqual(h.cache.mamba.available_size(),2)
                    process_output(h,hot)
                    self.assertEqual(owner.inflight_middle_chunks,0)
                    self.assertEqual(h.scheduler.disagg_prefill_inflight_queue,hot.reqs)
                    self.assertEqual(donations[0][0],'a')
                    self.assertEqual(sends,[('a',{'last_chunk':True})])
                    self.assertEqual(h.cache.mamba.available_size(),1)
                    # Real process_prefill_chunk must NOT recache/send paused owner.
                    OUTPUT.process_prefill_chunk(h.scheduler,None,h.running)
                    self.assertEqual(len(donations),1)
                    self.assertEqual(checks,['owner'])
                    self.assertEqual(len(sends),1)
                    cold=h.step()
                    self.assertEqual(cold.reqs,[owner])
                    self.assertEqual((owner.extend_range.start,owner.extend_range.end),(16384,32768))
                    self.assertEqual(owner.inflight_middle_chunks,1)
                    self.assertIs(cold.chunked_req,owner)
                    process_output(h,cold)
                    self.assertEqual(owner.inflight_middle_chunks,0)
                    self.assertEqual(len(donations),1)
                    OUTPUT.process_prefill_chunk(h.scheduler,None,h.running)
                    self.assertEqual([d[0] for d in donations],['a','owner'])
                    self.assertEqual(h.cache.mamba.available_size(),0)
                    self.assertEqual(h.scheduler.waiting_queue[0].rid,'b')
                owned=[*sources, owner.kv.mamba_pool_idx, hot.reqs[0].kv.mamba_pool_idx,
                       owner.kv.mamba_ping_pong_track_buffer[0],hot.reqs[0].kv.mamba_ping_pong_track_buffer[0],
                       *(slot for _,slots in donations for slot in slots)]
                self.assertEqual(len(owned),len(set(owned)))
                self.assertTrue(set(owned)<=h.cache.mamba.live)
                pool=h.cache.req_to_token_pool
                for req in [owner,*hot.reqs]:
                    self.assertEqual(pool.req_index_to_mamba_index_mapping[req.kv.req_pool_idx],req.kv.mamba_pool_idx)

    def test_pinned_exhaustion_falls_back_without_losing_queue(self):
        for free in (0,1,2,3):
            h,c,prepare,sources,events=physical(free)
            queued=list(h.scheduler.waiting_queue)
            with patch.object(Batch,'prepare_for_extend',prepare):
                batch=h.step()
            self.assertEqual(batch.reqs,[h.scheduler.chunked_req])
            self.assertEqual(h.scheduler.waiting_queue,queued)
            self.assertTrue(all(not r.kv.holds_mamba for r in queued))
            self.assertEqual(batch.reqs[0].extend_range.length,16384)

    def test_shared_pool_is_fenced(self):
        h=Harness();owner=h.cold();hot=h.hot()
        h.scheduler.token_to_kv_pool_allocator=UnifiedMambaTokenToKVPoolAllocator(h.cache.mamba)
        batch=h.step()
        self.assertEqual(batch.reqs,[owner]);self.assertIn(hot,h.scheduler.waiting_queue)

    def test_no_cold_branch_host_or_special_admitted_in_hot_pass(self):
        h=Harness();owner=h.cold()
        for name,changes in [('cold',{'hit':0,'prefix_indices':[]}),
            ('long',{'full_untruncated_fill_ids':range(10000)}),
            ('branch',{'mamba_branching_seqlen':4352}),
            ('host',{'host_hit_length':256}),('storage',{'storage_hit_length':256}),
            ('session',{'session':object()}),('pending',{'pending_bootstrap':True})]:
            req=h.hot(name);req.__dict__.update(changes)
        hot=h.hot('eligible',extend=512)
        batch=h.step()
        self.assertEqual(batch.reqs,[hot]);self.assertIsNone(batch.chunked_req)
        self.assertEqual(len(h.scheduler.waiting_queue),7)
        self.assertTrue(all(not r.kv.holds_mamba for r in h.scheduler.waiting_queue))
        self.assertEqual(owner.init_calls,0)

    def test_budget_rows_input_prefill_and_cold_reservation(self):
        cases=[lambda h:setattr(h.scheduler,'max_prefill_tokens',64),
               lambda h:setattr(h.cache,'rows',1),
               lambda h:setattr(h.schedule,'prefill_max_requests',0),
               lambda h:setattr(h.cache.allocator,'tokens',16448+320+8)]
        for mutate in cases:
            h=Harness();owner=h.cold();hot=h.hot(extend=256);mutate(h)
            batch=h.step()
            self.assertEqual(batch.reqs,[owner]);self.assertIn(hot,h.scheduler.waiting_queue)

    def test_zero_prefill_batch_limit_is_unlimited_not_zero_capacity(self):
        for cap in (0, 1, 16):
            with self.subTest(cap=cap):
                h=Harness();owner=h.cold();hot=h.hot(extend=256)
                h.scheduler.max_prefill_bs=cap
                batch=h.step()
                self.assertEqual(batch.reqs,[hot])
                self.assertIsNone(batch.chunked_req)
                self.assertIs(h.scheduler.chunked_req,owner)
                self.assertEqual(owner.init_calls,0)

    def test_reject_after_cow_cleans_allocation_and_preserves_owner(self):
        h=Harness();owner=h.cold();hot=h.hot()
        hot.after_cow=lambda req,tree:setattr(tree.allocator,'tokens',16500)
        batch=h.step()
        self.assertEqual(batch.reqs,[owner]);self.assertFalse(hot.kv.holds_mamba)
        self.assertIsNone(hot.kv.mamba_cow_src_index)
        self.assertEqual(hot.last_node.refs,0)
        self.assertEqual(owner.last_node.refs,1)

    def test_default_off_matches_head(self):
        for enabled in (None,'0','garbage'):
            outcomes=[]
            for ns in (CURRENT,BASELINE):
                h=Harness(namespace=ns);owner=h.cold();h.hot();batch=h.step(enabled=enabled)
                outcomes.append(([r.rid for r in batch.reqs],owner.extend_range.start,
                    owner.extend_range.end,owner.inflight_middle_chunks,
                    [r.rid for r in h.scheduler.waiting_queue],h.cache.events))
            self.assertEqual(*outcomes)

    def test_profile_fences(self):
        for field,value in [('enable_overlap',True),('schedule_policy','lpm'),
            ('enable_priority_preemption',True),('enable_lora',True),
            ('is_mixed_chunk',True),('is_hybrid_swa',True),
            ('disaggregation_mode','decode')]:
            h=Harness();h.cold();h.hot();setattr(h.scheduler,field,value)
            # Test the actual profile guard directly; unsupported schedulers
            # have additional runtime dependencies outside this fixture.
            adder=CURRENT['PrefillAdder'](64,h.cache,h.cache.allocator,h.running,1,32768,16384)
            with patch.dict(os.environ,{FLAG:'1'}):
                self.assertFalse(h.scheduler._glm53_interleaved_prefill_enabled(adder,h.running))

    def test_repeated_yields_finish_cold_with_delayed_transfers(self):
        h,c,prepare,sources,events=physical(20)
        owner=h.scheduler.chunked_req
        donations,sends,checks=connect_output(h,c)
        order=[];ranges=[]
        with patch.object(Batch,'prepare_for_extend',prepare):
            while h.scheduler.chunked_req is not None:
                batch=h.step();order.append(batch.reqs[0].rid)
                self.assertEqual(len(batch.reqs),1)
                if batch.reqs[0] is owner:
                    ranges.append((owner.extend_range.start,owner.extend_range.end))
                process_output(h,batch)
                OUTPUT.process_prefill_chunk(h.scheduler,None,h.running)
                self.assertLess(len(order),8)
        self.assertEqual(order,['a','owner','b','owner','owner'])
        self.assertEqual(ranges,[(16384,32768),(32768,49152),(49152,65536)])
        self.assertEqual(owner.inflight_middle_chunks,0)
        self.assertEqual([r.rid for r in h.scheduler.disagg_prefill_inflight_queue],['a','b','owner'])
        self.assertIsNone(h.scheduler._glm53_interleaved_resume_owner)
        self.assertEqual(owner.output_ids,[42])
        self.assertEqual(h.scheduler.waiting_queue,[])

    def test_tensor_prefix_does_not_read_device_boolean(self):
        class DeviceIndices(list):
            def __bool__(self):
                raise AssertionError('device truth-value/synchronization')
        h=Harness();owner=h.cold();hot=h.hot()
        original=hot.init_next_round_input
        def match(*a,**kw):
            original(*a,**kw)
            hot.prefix_indices=DeviceIndices(hot.prefix_indices)
        hot.init_next_round_input=match
        self.assertEqual(h.step().reqs,[hot])

    def test_paused_owner_bootstrap_failure_and_stale_reference(self):
        h,c,prepare,sources,events=physical(8)
        owner=h.scheduler.chunked_req
        donations,sends,checks=connect_output(h,c)
        with patch.object(Batch,'prepare_for_extend',prepare):
            batch=h.step();process_output(h,batch)
        owner.pending_bootstrap=True;owner.aborted=True
        OUTPUT.process_prefill_chunk(h.scheduler,None,h.running)
        self.assertEqual(checks,['owner'])
        self.assertEqual(len(donations),1)
        self.assertIsNone(h.scheduler.chunked_req)
        self.assertIsNone(h.scheduler._glm53_interleaved_resume_owner)
        h.scheduler._glm53_interleaved_resume_owner=owner
        OUTPUT.process_prefill_chunk(h.scheduler,None,h.running)
        self.assertIsNone(h.scheduler._glm53_interleaved_resume_owner)

    def test_original_chunk_hook_would_donate_paused_owner_again(self):
        h,c,prepare,sources,events=physical(8)
        owner=h.scheduler.chunked_req
        donations,sends,checks=connect_output(h,c)
        owner.kv.mamba_last_track_seqlen=16384
        owner.kv.mamba_last_track_idx=0
        with patch.object(Batch,'prepare_for_extend',prepare):
            process_output(h,h.step())
        load_output(baseline=True).process_prefill_chunk(h.scheduler,None,h.running)
        self.assertEqual([d[0] for d in donations],['a','owner'])
        self.assertEqual(owner.init_calls,0)  # donation happened with no forward

    def test_cumulative_counters_and_real_internal_state_accessor(self):
        def read(h):
            s=h.scheduler
            h.parallel.enable_dp_attention=True
            h.parallel.dp_size=2;h.parallel.tp_size=8
            h.execution.moe=NS(elastic_ep_backend=None)
            h.ns.update(get_context=lambda:NS(resolved_server_args_dict=lambda:{}),
                compute_world_size=lambda **kw:16,
                build_memory_usage=lambda **kw:{},
                GetInternalStateReqOutput=lambda **kw:NS(**kw),
                msgspec_to_builtins=lambda value:value,
                RECORD_STEP_TIME=False,
                envs=NS(SGLANG_EXPOSE_OWN_ENV_VARS=NS(get=lambda:False)))
            s.metrics_reporter=NS(last_gen_throughput=0)
            s.draft_worker=None
            s.tp_worker.model_runner.weight_load_mem_usage=0
            s.tp_worker.graph_memory_usage=0
            s.token_to_kv_pool_allocator.get_kvcache=lambda:NS(mem_usage=0)
            s.startup_available_gpu_memory_gb=0
            s.max_total_num_tokens=100000;s.swa_tokens_per_layer=0;s.startup_time=0
            s.spec_algorithm=NS(is_none=lambda:True,is_dspark=lambda:False)
            h.ns['DisaggregationMode']=NS(PREFILL='prefill',DECODE='decode')
            return s.get_internal_state(None).internal_state['glm53_prefill_hot_lane']
        for dp in (0,1):
            h=Harness();owner=h.cold();h.hot(extend=129)
            h.scheduler.ps.attn_dp_rank=dp
            zero=dict(mode='standalone',dp_rank=dp,attn_tp_rank=0,
                eligible_batches=0,admitted_batches=0,admitted_requests=0,admitted_paged_tokens=0)
            self.assertEqual(read(h),zero)
            original=Batch.prepare_for_extend
            def prepare(batch):
                self.assertEqual(read(h),zero)  # Increment only AFTER preparation.
                original(batch)
            with patch.object(Batch,'prepare_for_extend',prepare):
                hot=h.step()
            first=read(h)
            self.assertEqual(first,{**zero,'eligible_batches':1,'admitted_batches':1,
                'admitted_requests':1,'admitted_paged_tokens':192})
            self.assertEqual(hot.reqs[0].extend_range.length,129)
            self.assertEqual(h.step().reqs,[owner])
            self.assertEqual(read(h),{**first,'eligible_batches':2})
            self.assertEqual(first['eligible_batches'],1)  # Snapshot stays stable.
        h=Harness();owner=h.cold();h.hot()
        h.step(enabled='0')
        self.assertEqual(read(h)['eligible_batches'],0)
        h=Harness(rows=1);h.cold();h.hot()
        h.step()
        self.assertEqual(read(h)['eligible_batches'],1)
        self.assertEqual(read(h)['admitted_requests'],0)
        h=Harness();h.cold();h.hot()
        with patch.object(Batch,'prepare_for_extend',side_effect=RuntimeError('prepare failed')):
            with self.assertRaises(RuntimeError):h.step()
        self.assertEqual(read(h)['eligible_batches'],0)

    def test_bounded_scan_and_zero_storage_result(self):
        h=Harness();owner=h.cold()
        cold=[h.hot(str(i),prefix=0,extend=65536) for i in range(32)]
        hot=h.hot('hot')
        self.assertEqual(h.step().reqs,[owner])
        self.assertEqual(hot.init_calls,0)
        self.assertEqual(h.scheduler.waiting_queue,[*cold,hot])
        h=Harness();h.cold();hot=h.hot()
        h.cache.prefetch_loaded_tokens_by_reqid[hot.rid]=0
        self.assertEqual(h.step().reqs,[hot])

    def test_max_running_includes_retained_owner_and_transfers(self):
        h=Harness();owner=h.cold();hot=h.hot()
        h.scheduler.max_running_requests=1
        self.assertEqual(h.step().reqs,[owner])
        self.assertFalse(hot.kv.holds_mamba)

    def test_unchanged_cold_math_and_prepared_dispatch_guard(self):
        def method(text,cls,name):
            c=next(n for n in ast.parse(text).body if getattr(n,'name',None)==cls)
            return ast.dump(next(n for n in c.body if getattr(n,'name',None)==name))
        for name in ('add_chunked_req','add_one_req','_update_prefill_budget'):
            self.assertEqual(method(source('schedule_policy.py'), 'PrefillAdder',name),
                             method(source('schedule_policy.py',True),'PrefillAdder',name))
        path='python/sglang/srt/hardware_backend/npu/attention/glm53/kda_recurrent_npu.py'
        self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(), FROZEN_KDA_SHA256)
        # Evaluate the REAL prepared dispatch condition, not a copy of its rule.
        tree=ast.parse((ROOT/path).read_text())
        guard=next(n.test for n in ast.walk(tree) if isinstance(n,ast.If)
            and 'SGLANG_GLM53_KDA_PREFILL_PREPARE' in ast.unparse(n.test))
        ns=dict(prefill=True,intermediate_state=None,num_q_heads=4,num_value_heads=4,
            key_dim=128,value_dim=128,os=os,torch=NS(bfloat16='bf16',float32='fp32'),
            initial_state_source=NS(dtype='fp32'))
        for name in ('q','k','v','a','b'):ns[name]=NS(dtype='bf16',shape=(1,256),device=NS(type='npu'))
        with patch.dict(os.environ,{'SGLANG_GLM53_KDA_PREFILL_PREPARE':'1'}):
            for nseq,expected in ((1,True),(2,False)):
                ns['initial_state_indices']=NS(numel=lambda:nseq)
                self.assertEqual(eval(compile(ast.Expression(guard),'<real KDA guard>','eval'),ns),expected)

if __name__=='__main__':unittest.main()
