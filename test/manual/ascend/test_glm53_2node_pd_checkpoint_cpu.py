"""CPU regression for the finite DP2/TP8/EP16 P checkpoint opt-in.

Run: python -B test/manual/ascend/test_glm53_2node_pd_checkpoint_cpu.py -v

The existing CPU harness executes the actual production methods extracted by
AST, with NumPy tensor stand-ins at their dependency boundary. No SGLang/NPU
runtime, model, process group, or network is initialized. These tests establish
checkpoint selection, metadata and slot ownership, not NPU recurrence accuracy.
"""

import math
import os
import unittest
from unittest.mock import patch

import numpy as np

from test_glm53_pd_aligned_prefix_cpu import (
    FLAG,
    SCHEDULE,
    Entry,
    Mode,
    NS,
    Tensor,
    extract,
    inserted,
    make_batch,
    make_request,
    metadata,
)

TWO_NODE = "SGLANG_GLM53_2NODE_PD"
NORMAL_HCCL = "SGLANG_GLM53_NORMAL_HCCL"
END = 131072
START = END - 8192


class CheckpointCase:
    def __init__(self, *, page=64):
        self.context = NS(
            parallel=NS(
                dp_size=2,
                nnodes=2,
                tp_size=16,
                ep_size=16,
                enable_dp_attention=True,
                attn_tp_size=8,
                pp_size=1,
                attn_cp_size=1,
                dcp_size=1,
                moe_dp_size=1,
                dwdp_size=1,
                enable_prefill_cp=False,
            ),
            execution=NS(
                mamba=NS(
                    enable_mamba_extra_buffer=True,
                    enable_mamba_extra_buffer_lazy=False,
                ),
                overlap=NS(
                    enable_two_batch_overlap=False,
                    enable_single_batch_overlap=False,
                ),
                moe=NS(
                    moe_a2a_backend="deepep",
                    deepep_mode="normal",
                    enable_eplb=False,
                    ep_num_redundant_experts=0,
                ),
            ),
            schedule=NS(disable_overlap_schedule=True),
            model=NS(quantization="modelslim"),
            disagg=NS(disaggregation_mode="prefill"),
        )
        self.env = {FLAG: "1", TWO_NODE: "1", NORMAL_HCCL: "1"}
        self.cache_chunk = 64
        self.batch = make_batch(page=page)
        if page != 64:
            self.batch.tree_cache.glm53_kpool_share_page_size = page
        self.req = make_request(end=END, start=START)

    def run(self):
        namespace = dict(
            os=os,
            math=math,
            ForwardMode=Mode,
            get_parallel=lambda: self.context.parallel,
            get_exec=lambda: self.context.execution,
            get_schedule=lambda: self.context.schedule,
            get_model=lambda: self.context.model,
            get_disagg=lambda: self.context.disagg,
            mamba_cache_chunk_size=lambda: self.cache_chunk,
            _MambaRadixCacheV2TrackEntry=Entry,
        )
        extract(
            "python/sglang/srt/runtime_context.py", "mamba_checkpoint_grid", namespace
        )
        prepare = extract(
            SCHEDULE, "_mamba_radix_cache_v2_req_prepare_for_extend", namespace
        )
        with patch.dict(os.environ):
            for name, value in self.env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            return prepare(self.batch, self.req)

    def set_context(self, path, value):
        owner = self.context
        *parents, leaf = path.split(".")
        for name in parents:
            owner = getattr(owner, name)
        setattr(owner, leaf, value)


class TwoNodeCheckpointTests(unittest.TestCase):
    def assert_no_backoff(self, case):
        entry = case.run()
        self.assertTrue(entry.track_mask)
        self.assertEqual(entry.track_seqlen, END)
        self.assertEqual(case.req.kv.mamba_last_track_seqlen, END)

    def test_final_128k_chunk_tracks_reusable_state_and_h_sentinel(self):
        for page in (64, 256):
            with self.subTest(page=page):
                case = CheckpointCase(page=page)
                entry = case.run()
                self.assertTrue(entry.track_mask)
                self.assertEqual(case.req.kv.mamba_last_track_seqlen, END - page)
                self.assertEqual(entry.track_seqlen, END - page + 1)
                self.assertEqual(entry.track_index, 10)
                self.assertEqual(case.req.kv.mamba_last_track_idx, 0)
                self.assertEqual(case.req.kv.mamba_next_track_idx, 1)
                self.assertEqual(case.req.kv.mamba_pool_idx.item(), 3)
                self.assertEqual(case.req.extend_range.end, END)
                self.assertEqual(len(case.req.origin_input_ids), END)

    def test_each_opt_in_requires_literal_one(self):
        for flag in (FLAG, TWO_NODE, NORMAL_HCCL):
            for value in (None, "", "0", "true", "True", "2"):
                with self.subTest(flag=flag, value=value):
                    case = CheckpointCase()
                    case.env[flag] = value
                    self.assert_no_backoff(case)

    def test_other_topologies_and_profiles_retain_original_checkpoint(self):
        rejected = {
            "parallel.dp_size": (0, 3, 4, 8),
            "parallel.nnodes": (1, 3, 4),
            "parallel.tp_size": (8, 32),
            "parallel.ep_size": (8, 32),
            "parallel.enable_dp_attention": (False,),
            "parallel.attn_tp_size": (2, 4, 16),
            "parallel.pp_size": (2,),
            "parallel.attn_cp_size": (2,),
            "parallel.dcp_size": (2,),
            "parallel.moe_dp_size": (2,),
            "parallel.dwdp_size": (16,),
            "parallel.enable_prefill_cp": (True,),
            "schedule.disable_overlap_schedule": (False,),
            "execution.overlap.enable_two_batch_overlap": (True,),
            "execution.overlap.enable_single_batch_overlap": (True,),
            "execution.moe.moe_a2a_backend": ("none", "ascend_fuseep"),
            "execution.moe.deepep_mode": ("auto", "low_latency"),
            "execution.moe.enable_eplb": (True,),
            "execution.moe.ep_num_redundant_experts": (1,),
            "model.quantization": (None, "w8a8_int8"),
            "execution.mamba.enable_mamba_extra_buffer": (False,),
            "execution.mamba.enable_mamba_extra_buffer_lazy": (True,),
            "disagg.disaggregation_mode": ("null", "decode"),
        }
        for path, values in rejected.items():
            for value in values:
                with self.subTest(path=path, value=value):
                    case = CheckpointCase()
                    case.set_context(path, value)
                    self.assert_no_backoff(case)

    def test_existing_model_device_and_forward_guards(self):
        for device in ("cpu", "cuda:0", "xpu:0"):
            with self.subTest(device=device):
                case = CheckpointCase()
                case.batch.device = device
                self.assert_no_backoff(case)
        for architectures in (None, [], ["KimiLinearForCausalLM"]):
            with self.subTest(architectures=architectures):
                case = CheckpointCase()
                case.batch.model_config.hf_config.architectures = architectures
                self.assert_no_backoff(case)
        for mode in Mode:
            if mode != Mode.EXTEND:
                with self.subTest(mode=mode):
                    case = CheckpointCase()
                    case.batch.forward_mode = mode
                    self.assert_no_backoff(case)
        case = CheckpointCase()
        case.batch.tree_cache.disable = True
        self.assert_no_backoff(case)

    def test_checkpoint_geometry_guards(self):
        for cache_chunk, state_chunk in ((128, 64), (64, 128)):
            with self.subTest(cache_chunk=cache_chunk, state_chunk=state_chunk):
                case = CheckpointCase()
                case.cache_chunk = cache_chunk
                case.batch.model_config.hf_text_config.mamba_chunk_size = state_chunk
                self.assert_no_backoff(case)
        case = CheckpointCase(page=256)
        del case.batch.tree_cache.glm53_kpool_share_page_size
        self.assert_no_backoff(case)
        case = CheckpointCase()
        case.batch.tree_cache.page_size = 32  # lcm(64, 32) != tree page
        self.assert_no_backoff(case)
        case = CheckpointCase()
        case.req.extend_range.start += 64  # does not name prefix_indices' end
        self.assert_no_backoff(case)

    def test_intermediate_chunks_remain_at_their_actual_end(self):
        for page in (64, 256):
            for end in (8192, 65536, START):
                with self.subTest(page=page, end=end):
                    case = CheckpointCase(page=page)
                    case.req = make_request(end=end, start=end - 8192, origin=END)
                    entry = case.run()
                    self.assertEqual(entry.track_seqlen, end)
                    self.assertEqual(case.req.kv.mamba_last_track_seqlen, end)

    def test_short_unaligned_output_and_logprob_cases_remain_unchanged(self):
        for options in (
            dict(end=END - 1, start=START),
            dict(end=END + 1, start=START),
            dict(end=END, start=END - 64),
            dict(end=END, start=END - 1),
            dict(end=END, start=START + 1),
            dict(end=END, start=START, output=(1,)),
            dict(end=END, start=START, logprob=START),
        ):
            with self.subTest(options=options):
                case, reference = CheckpointCase(), CheckpointCase()
                case.req = make_request(**options)
                reference.req = make_request(**options)
                reference.env[FLAG] = "0"
                self.assertEqual(case.run(), reference.run())
                for field in (
                    "mamba_last_track_seqlen",
                    "mamba_last_track_idx",
                    "mamba_next_track_idx",
                ):
                    self.assertEqual(
                        getattr(case.req.kv, field), getattr(reference.req.kv, field)
                    )

    def test_prefix_limit_accepts_checkpoint_but_rejects_prompt_end(self):
        for page in (64, 256):
            case = CheckpointCase(page=page)
            self.assertEqual(case.req._compute_max_prefix_len(END), END - 1)
            case.run()
            depth = case.req.kv.mamba_last_track_seqlen
            self.assertEqual(depth, (END - 1) // page * page)
            self.assertLess(depth, END)
            # The logprob cap must also cover the candidate checkpoint.
            for limit in (depth - 1, depth):
                with self.subTest(page=page, logprob_limit=limit):
                    capped = CheckpointCase(page=page)
                    capped.req.return_logprob = True
                    capped.req.logprob_start_len = limit
                    capped.run()
                    self.assertEqual(capped.req._compute_max_prefix_len(END), limit)
                    self.assertEqual(
                        capped.req.kv.mamba_last_track_seqlen,
                        END if limit < depth else depth,
                    )

    def test_branch_checkpoint_keeps_existing_priority(self):
        case = CheckpointCase()
        case.req.mamba_branching_seqlen = START + 4096
        entry = case.run()
        self.assertEqual(case.req.kv.mamba_last_track_seqlen, START + 4096)
        self.assertEqual(entry.track_seqlen, START + 4096 + 1)

    def test_two_requests_per_domain_keep_local_slots_and_matching_metadata(self):
        # Reuse the same local slot IDs in two independent domains, as real
        # per-process pools do. No cross-domain token count or slot translation
        # is introduced by the eligibility change.
        for page in (64, 256):
            domains = []
            for _domain in range(2):
                cases, entries = [], []
                for local_req in range(2):
                    case = CheckpointCase(page=page)
                    case.batch.device = "npu:0"
                    case.req.kv.mamba_ping_pong_track_buffer = Tensor(
                        [10 + 2 * local_req, 11 + 2 * local_req], np.int64
                    )
                    case.req.kv.mamba_pool_idx = Tensor(3 + local_req)
                    cases.append(case)
                    entries.append(case.run())
                requests = [case.req for case in cases]
                lengths, conv, ssm = metadata(requests, entries, 6)
                local_depth = 8192 - page
                np.testing.assert_array_equal(lengths, [local_depth] * 2)
                np.testing.assert_array_equal(ssm[0], [local_depth // 64] * 2)
                np.testing.assert_array_equal(
                    ssm[1], [local_depth // 64, 128 + local_depth // 64]
                )
                np.testing.assert_array_equal(ssm[2], [10, 12])
                np.testing.assert_array_equal(ssm[3], [0, 1])
                self.assertEqual(len(ssm[4]), 0)  # no final/live-state source
                for i, case in enumerate(cases):
                    np.testing.assert_array_equal(
                        conv[i],
                        np.arange(i * 8192 + local_depth - 6, i * 8192 + local_depth),
                    )
                    self.assertEqual(case.req.kv.mamba_pool_idx.item(), 3 + i)
                    params = inserted(case.req, unified=True)
                    self.assertEqual(len(params.key), END - page)
                    self.assertEqual(params.mamba_value.item(), 10 + 2 * i)
                    self.assertEqual(case.req.kv.mamba_pool_idx.item(), 3 + i)
                domains.append(cases)
            self.assertFalse(
                np.shares_memory(
                    domains[0][0].req.kv.mamba_ping_pong_track_buffer,
                    domains[1][0].req.kv.mamba_ping_pong_track_buffer,
                )
            )

    def test_dp1_retains_legacy_behavior_without_new_profile_fields(self):
        for page in (64, 256):
            for enabled in (False, True):
                with self.subTest(page=page, enabled=enabled):
                    case = CheckpointCase(page=page)
                    # Missing new fields/accessor bags make any accidental
                    # dependency on the DP2 profile fail this DP1 regression.
                    case.context.parallel = NS(
                        dp_size=1, pp_size=1, attn_cp_size=1, dcp_size=1
                    )
                    del case.context.execution.moe
                    del case.context.execution.overlap
                    del case.context.schedule
                    del case.context.model
                    case.env.update(
                        {
                            FLAG: "1" if enabled else "0",
                            TWO_NODE: None,
                            NORMAL_HCCL: None,
                        }
                    )
                    entry = case.run()
                    depth = END - page if enabled else END
                    self.assertEqual(case.req.kv.mamba_last_track_seqlen, depth)
                    self.assertEqual(entry.track_seqlen, depth + 1 if enabled else END)


if __name__ == "__main__":
    unittest.main()
