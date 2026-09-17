from __future__ import annotations

import ast
import copy
import json
import os
import tempfile
import unittest
from enum import Enum
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

import torch

from sglang.srt.disaggregation.glm53_decode_prefix import (
    GLM53_PREFIX_SHARE_PAGE_SIZE,
    glm53_pd_decode_prefix_profile,
    mark_glm53_pd_mamba_state_authoritative,
    plan_glm53_pd_decode_prefix,
    prepare_glm53_pd_rebootstrap,
    validate_glm53_pd_decode_prefix_runtime,
    write_glm53_cache_audit,
)

ROOT = Path(__file__).resolve().parents[4]


def extract_class_method(relative_path, class_name, method_name, **namespace):
    """Load one class method without importing the complete server stack.

    The host CPU environment need not import the complete server stack (and its
    NPU/xgrammar dependencies), while this still executes the production method.
    """

    path = ROOT / relative_path
    tree = ast.parse(path.read_text())
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = copy.deepcopy(
        next(
            node
            for node in owner.body
            if isinstance(node, ast.FunctionDef) and node.name == method_name
        )
    )
    method.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            method,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[method_name]


def extract_decode_match(match_prefix_for_req):
    return extract_class_method(
        "python/sglang/srt/disaggregation/decode.py",
        "DecodePreallocQueue",
        "_match_prefix_and_lock",
        match_prefix_for_req=match_prefix_for_req,
        plan_glm53_pd_decode_prefix=plan_glm53_pd_decode_prefix,
    )


def extract_module_function(relative_path, function_name, **namespace):
    """Execute one production function without importing the NPU server stack."""

    path = ROOT / relative_path
    tree = ast.parse(path.read_text())
    function = copy.deepcopy(
        next(
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name
        )
    )
    function.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            function,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[function_name]


def target_model():
    return NS(hf_config=NS(architectures=["Glm5NextForConditionalGeneration"]))


def target_cfg(**overrides):
    values = dict(
        device="npu",
        disaggregation_mode="decode",
        disaggregation_transfer_backend="ascend",
        quantization="modelslim",
        speculative_draft_model_quantization="modelslim",
        tp_size=16,
        dp_size=8,
        ep_size=16,
        pp_size=1,
        dcp_size=1,
        attn_cp_size=1,
        nnodes=1,
        enable_dp_attention=True,
        page_size=64,
        enable_hierarchical_cache=False,
        disaggregation_decode_retraction_backup=None,
        speculative_algorithm="EAGLE",
        speculative_num_steps=3,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=4,
    )
    values.update(overrides)
    return NS(**values)


class ProfileTests(unittest.TestCase):
    def test_exact_supported_profile(self):
        profile = glm53_pd_decode_prefix_profile(target_cfg(), target_model())
        self.assertIsNotNone(profile)
        self.assertEqual(profile.physical_page_size, 64)
        self.assertEqual(profile.prefix_share_page_size, 256)

    def test_every_unadapted_dimension_stays_rejected(self):
        cases = {
            "device": "cuda",
            "disaggregation_mode": "prefill",
            "disaggregation_transfer_backend": "mooncake",
            "quantization": None,
            "speculative_draft_model_quantization": None,
            "tp_size": 8,
            "dp_size": 4,
            "ep_size": 8,
            "pp_size": 2,
            "dcp_size": 2,
            "attn_cp_size": 2,
            "nnodes": 2,
            "enable_dp_attention": False,
            "page_size": 128,
            "enable_hierarchical_cache": True,
            "disaggregation_decode_retraction_backup": "host_pool",
            "speculative_algorithm": "EAGLE3",
            "speculative_num_steps": 2,
            "speculative_eagle_topk": 2,
            "speculative_num_draft_tokens": 5,
        }
        for field, value in cases.items():
            with self.subTest(field=field, value=value):
                self.assertIsNone(
                    glm53_pd_decode_prefix_profile(
                        target_cfg(**{field: value}), target_model()
                    )
                )

        other = NS(hf_config=NS(architectures=["KimiK3ForConditionalGeneration"]))
        self.assertIsNone(glm53_pd_decode_prefix_profile(target_cfg(), other))


class TransferPlanTests(unittest.TestCase):
    def setUp(self):
        self.profile = glm53_pd_decode_prefix_profile(target_cfg(), target_model())

    def test_partial_kpool_group_is_always_in_tail(self):
        indices = torch.arange(4096, dtype=torch.int64)
        for residue in (0, 1, 63, 64, 128, 192, 255):
            matched = 3072 + residue
            with self.subTest(residue=residue):
                plan = plan_glm53_pd_decode_prefix(
                    self.profile, indices[:matched], fill_len=4096
                )
                self.assertEqual(plan.prefix_len, 3072)
                self.assertEqual(plan.transfer_start, 3072)
                self.assertEqual(plan.transfer_len, 1024)
                self.assertTrue(torch.equal(plan.prefix_indices, indices[:3072]))

    def test_complete_groups_and_short_fill(self):
        indices = torch.arange(4096, dtype=torch.int64)
        exact = plan_glm53_pd_decode_prefix(self.profile, indices[:3328], fill_len=4096)
        self.assertEqual((exact.prefix_len, exact.transfer_len), (3328, 768))

        capped = plan_glm53_pd_decode_prefix(self.profile, indices, fill_len=3294)
        self.assertEqual((capped.prefix_len, capped.transfer_len), (3072, 222))

    def test_exact_prompt_keeps_one_complete_group_fresh(self):
        for prompt_len in (8192, 65536, 131072):
            with self.subTest(prompt_len=prompt_len):
                indices = torch.arange(prompt_len, dtype=torch.int64)
                plan = plan_glm53_pd_decode_prefix(
                    self.profile, indices, fill_len=prompt_len
                )
                self.assertEqual(plan.prefix_len, prompt_len - 256)
                self.assertEqual(plan.transfer_start, prompt_len - 256)
                self.assertEqual(plan.transfer_len, 256)
                self.assertEqual(plan.transfer_len // 64, 4)

    def test_negative_fill_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            plan_glm53_pd_decode_prefix(self.profile, (), -1)


class RuntimeContractTests(unittest.TestCase):
    class State(Enum):
        MAMBA = "mamba"
        DSA_TAIL = "dsa_tail"

    def setUp(self):
        self.profile = glm53_pd_decode_prefix_profile(target_cfg(), target_model())
        self.tree = NS(
            page_size=GLM53_PREFIX_SHARE_PAGE_SIZE,
            glm53_kpool_share_page_size=GLM53_PREFIX_SHARE_PAGE_SIZE,
            supports_mamba=lambda: False,
        )
        self.allocator = NS(page_size=64)
        self.req_pool = NS(
            mamba_pool=object(),
            mamba_allocator=object(),
            free_mamba_cache=lambda req: None,
        )

    def validate(self, **overrides):
        values = dict(
            tree_cache=self.tree,
            token_to_kv_pool_allocator=self.allocator,
            req_to_token_pool=self.req_pool,
            state_types=[self.State.MAMBA, self.State.DSA_TAIL],
            draft_token_to_kv_pool=object(),
        )
        values.update(overrides)
        validate_glm53_pd_decode_prefix_runtime(self.profile, **values)

    def test_complete_runtime_contract(self):
        self.validate()

    def test_missing_contract_parts_fail_before_serving(self):
        bad_tree = copy.copy(self.tree)
        bad_tree.glm53_kpool_share_page_size = None
        cases = (
            ({"tree_cache": bad_tree}, "256-token"),
            ({"token_to_kv_pool_allocator": NS(page_size=128)}, "page64"),
            (
                {
                    "tree_cache": NS(
                        page_size=256,
                        glm53_kpool_share_page_size=256,
                        supports_mamba=lambda: True,
                    )
                },
                "token-only radix",
            ),
            ({"req_to_token_pool": NS()}, "active Mamba request pool"),
            ({"state_types": []}, "StateType.MAMBA"),
            ({"state_types": [self.State.MAMBA]}, "StateType.DSA_TAIL"),
            ({"draft_token_to_kv_pool": None}, "draft KV pool"),
        )
        for kwargs, message in cases:
            with (
                self.subTest(kwargs=kwargs),
                self.assertRaisesRegex(RuntimeError, message),
            ):
                self.validate(**kwargs)

    def test_incoming_prefill_state_cannot_be_clobbered(self):
        req = NS(
            kv=NS(
                holds_mamba=True,
                mamba_cow_src_index=torch.tensor([7]),
                mamba_needs_clear=True,
                mamba_last_track_idx=1,
                mamba_last_track_seqlen=3072,
            ),
            mamba_branching_seqlen=2048,
        )
        mark_glm53_pd_mamba_state_authoritative(req)
        self.assertIsNone(req.kv.mamba_cow_src_index)
        self.assertFalse(req.kv.mamba_needs_clear)
        self.assertIsNone(req.kv.mamba_last_track_idx)
        self.assertIsNone(req.kv.mamba_last_track_seqlen)
        self.assertIsNone(req.mamba_branching_seqlen)

    def test_authoritative_state_requires_allocated_active_slot(self):
        req = NS(
            kv=NS(
                holds_mamba=False,
                mamba_cow_src_index=None,
                mamba_needs_clear=False,
                mamba_last_track_idx=None,
                mamba_last_track_seqlen=None,
            ),
            mamba_branching_seqlen=None,
        )
        with self.assertRaisesRegex(RuntimeError, "fresh active Mamba slot"):
            mark_glm53_pd_mamba_state_authoritative(req)

    def test_retraction_rebootstraps_all_mtp_state_from_prefill(self):
        req = NS(
            output_ids=[101, 102, 103],
            retraction_mb_id=7,
            pd_rebootstrap_forced_output_id=None,
            pd_rebootstrap_in_progress=False,
        )
        prepare_glm53_pd_rebootstrap(req)
        self.assertEqual(req.output_ids, [101, 102])
        self.assertEqual(req.pd_rebootstrap_forced_output_id, 103)
        self.assertTrue(req.pd_rebootstrap_in_progress)
        self.assertIsNone(req.retraction_mb_id)


class AuditTests(unittest.TestCase):
    def test_jsonl_is_opt_in_and_contains_actual_transfer_contract(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(
                write_glm53_cache_audit(
                    rid="disabled",
                    dp_rank=1,
                    tp_rank=3,
                    actual_prefix_len=7936,
                    input_len=8192,
                    metadata_page_count=4,
                    transfer_success=True,
                    fresh_mamba=True,
                    draft_present=True,
                )
            )

        with tempfile.TemporaryDirectory() as audit_dir:
            with mock.patch.dict(
                os.environ, {"SGLANG_GLM53_CACHE_AUDIT_DIR": audit_dir}, clear=True
            ):
                path = write_glm53_cache_audit(
                    rid="exact-prompt",
                    dp_rank=7,
                    tp_rank=15,
                    actual_prefix_len=7936,
                    input_len=8192,
                    metadata_page_count=4,
                    transfer_success=True,
                    fresh_mamba=True,
                    draft_present=True,
                )

            self.assertIsNotNone(path)
            record = json.loads(path.read_text())
            self.assertEqual(
                {
                    key: record[key]
                    for key in (
                        "rid",
                        "dp_rank",
                        "tp_rank",
                        "actual_prefix_len",
                        "input_len",
                        "metadata_page_count",
                        "transfer_success",
                        "fresh_mamba",
                        "draft_present",
                    )
                },
                {
                    "rid": "exact-prompt",
                    "dp_rank": 7,
                    "tp_rank": 15,
                    "actual_prefix_len": 7936,
                    "input_len": 8192,
                    "metadata_page_count": 4,
                    "transfer_success": True,
                    "fresh_mamba": True,
                    "draft_present": True,
                },
            )
            self.assertIsInstance(record["timestamp_ns"], int)


class MtpAndLifecycleTests(unittest.TestCase):
    def test_live_mamba_commit_is_independent_of_256_track_boundary(self):
        verify_indices = extract_module_function(
            "python/sglang/srt/speculative/spec_utils.py",
            "_verify_commit_step_indices",
            torch=torch,
            mamba_track_grid=lambda tree_page: 256,
        )
        accept_index = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)
        accept_lens = torch.tensor([4], dtype=torch.int64)
        batch = NS(
            tree_cache=NS(page_size=256),
            mamba_track_indices=torch.tensor([11]),
            seq_lens=torch.tensor([8192], dtype=torch.int64),
        )

        live_step, track_step = verify_indices(
            batch=batch,
            accept_index=accept_index,
            accept_lens=accept_lens,
            draft_token_num=4,
        )
        self.assertEqual(live_step.tolist(), [3])
        self.assertEqual(track_step.tolist(), [-1])

        # Crossing 8448 snapshots step 1, while the live state still advances
        # to the final accepted step 3.
        batch.seq_lens = torch.tensor([8446], dtype=torch.int64)
        live_step, track_step = verify_indices(
            batch=batch,
            accept_index=accept_index,
            accept_lens=accept_lens,
            draft_token_num=4,
        )
        self.assertEqual(live_step.tolist(), [3])
        self.assertEqual(track_step.tolist(), [1])

    def test_full_only_finish_and_abort_free_independent_mamba_state(self):
        events = []

        class HybridReqToTokenPool:
            def free_mamba_cache(self, req):
                events.append("free_mamba")
                req.kv.mamba_pool_idx = None

            def free(self, req):
                events.append("free_req")
                req.kv.req_pool_idx = None

        release = extract_module_function(
            "python/sglang/srt/mem_cache/common.py",
            "release_kv_cache",
            HybridReqToTokenPool=HybridReqToTokenPool,
            _release_overallocated_kv_indices=lambda *args: events.append(
                "free_overallocated"
            ),
        )

        for is_insert in (True, False):
            with self.subTest(is_insert=is_insert):
                events.clear()

                class KV:
                    req_pool_idx = 9
                    mamba_pool_idx = torch.tensor(5)
                    kv_allocated_len = 1024
                    is_kv_released = False

                    @property
                    def holds_kv(self):
                        return self.req_pool_idx is not None

                    @property
                    def holds_mamba(self):
                        return self.mamba_pool_idx is not None

                    def mark_kv_released(self):
                        events.append("mark_released")
                        self.kv_allocated_len = 0
                        self.is_kv_released = True

                req = NS(
                    kv=KV(),
                    skip_radix_cache_insert=False,
                    effective_kv_committed_len=lambda: 1024,
                )
                pool = HybridReqToTokenPool()

                class Tree:
                    req_to_token_pool = pool

                    @staticmethod
                    def supports_mamba():
                        return False

                    @staticmethod
                    def cache_finished_req(req, *, is_insert, kv_len_to_handle):
                        events.append(("cache_full", is_insert, kv_len_to_handle))

                release(req, Tree(), is_insert=is_insert)
                self.assertEqual(
                    events,
                    [
                        ("cache_full", is_insert, 1024),
                        "free_overallocated",
                        "free_mamba",
                        "free_req",
                        "mark_released",
                    ],
                )
                self.assertFalse(req.kv.holds_kv)
                self.assertFalse(req.kv.holds_mamba)


class DecodeWiringTests(unittest.TestCase):
    def test_retracted_profile_uses_p_rebootstrap_instead_of_local_restore(self):
        events = []

        def discard(req, tree, backend):
            events.append(("discard", backend))
            req.kv.retraction_backup = None

        add = extract_class_method(
            "python/sglang/srt/disaggregation/decode.py",
            "DecodePreallocQueue",
            "add",
            retraction_discard=discard,
            get_disagg=lambda: NS(disaggregation_decode_retraction_backup="cpu_tensor"),
            prepare_glm53_pd_rebootstrap=prepare_glm53_pd_rebootstrap,
            _is_fake_transfer=lambda req: True,
            logger=NS(debug=lambda *args, **kwargs: None),
        )
        receiver = NS(init=lambda rank: events.append(("init", rank)))
        queue = NS(
            glm53_decode_prefix_profile=object(),
            tree_cache=object(),
            retracted_queue=[],
            pending_reqs=[],
            _check_if_req_exceed_kv_capacity=lambda req: False,
            _create_receiver_and_enqueue=lambda req, is_rebootstrap: (
                events.append(("create", is_rebootstrap))
                or NS(req=req, kv_receiver=receiver)
            ),
        )
        req = NS(
            output_ids=[21, 22],
            kv=NS(retraction_backup=object()),
            retraction_mb_id=4,
            pd_rebootstrap_forced_output_id=None,
            pd_rebootstrap_in_progress=False,
        )

        add(queue, req, is_retracted=True)

        self.assertEqual(queue.retracted_queue, [])
        self.assertIsNone(req.kv.retraction_backup)
        self.assertEqual(req.output_ids, [21])
        self.assertEqual(req.pd_rebootstrap_forced_output_id, 22)
        self.assertEqual(
            events,
            [("discard", "cpu_tensor"), ("create", True), ("init", 0)],
        )

    def test_prebuilt_merges_prefill_and_decode_prefix_metadata_as_union(self):
        sampling = object()
        prepare = extract_class_method(
            "python/sglang/srt/disaggregation/decode_schedule_batch_mixin.py",
            "ScheduleBatchDisaggregationDecodeMixin",
            "prepare_for_prebuilt",
            torch=torch,
            ForwardMode=NS(PREBUILT="prebuilt"),
            SamplingBatchInfo=NS(
                from_schedule_batch=lambda batch, vocab_size: sampling
            ),
        )

        def run(prefill_hit):
            req = NS(
                extend_range=NS(length=256),
                kv=NS(req_pool_idx=0),
                prefix_indices=torch.arange(7936),
                origin_input_ids=list(range(8192)),
                output_ids=[9],
                retracted_stain=False,
                already_computed=prefill_hit,
                cached_tokens=prefill_hit,
                cached_tokens_device=prefill_hit,
                is_retracted=True,
                pd_rebootstrap_in_progress=False,
                multimodal_inputs=None,
            )
            batch = NS(
                reqs=[req],
                device="cpu",
                req_to_token_pool=NS(
                    req_to_token=torch.arange(8192, dtype=torch.int32).reshape(1, -1)
                ),
                return_logprob=False,
                model_config=NS(vocab_size=1),
            )
            prepare(batch)
            return req, batch

        req, batch = run(prefill_hit=4096)
        self.assertEqual(req.cached_tokens, 7936)
        self.assertEqual(req.cached_tokens_device, 7936)
        self.assertEqual(req.already_computed, 8192)
        self.assertEqual(batch.prefix_lens, [7936])
        self.assertEqual(batch.extend_lens, [256])

        req, _ = run(prefill_hit=8000)
        self.assertEqual(req.cached_tokens, 8000)
        self.assertEqual(req.cached_tokens_device, 8000)

    def test_prebuilt_inserts_transferred_prompt_into_full_tree(self):
        cached = []
        process = extract_class_method(
            "python/sglang/srt/disaggregation/decode_schedule_batch_mixin.py",
            "ScheduleBatchDisaggregationDecodeMixin",
            "process_prebuilt",
            torch=torch,
            maybe_cache_unfinished_req=lambda req, tree: cached.append((req.rid, tree)),
        )
        spec_info = object()
        req = NS(
            rid="warm-prompt",
            output_ids=[17],
            swa_branching_seqlen=None,
            grammar=None,
        )
        batch = NS(
            reqs=[req],
            tree_cache="full-only-tree",
            device="cpu",
            spec_algorithm=NS(
                build_disagg_draft_input=lambda batch, tokens, future_map: spec_info
            ),
            req_pool_indices=torch.tensor([3]),
        )

        process(batch, NS())

        self.assertEqual(cached, [("warm-prompt", "full-only-tree")])
        self.assertIs(batch.spec_info, spec_info)

    def test_match_disables_mamba_cow_and_floors_kpool_prefix(self):
        profile = glm53_pd_decode_prefix_profile(target_cfg(), target_model())
        calls = []

        def match_prefix_for_req(*args, **kwargs):
            calls.append((args, kwargs))
            return NS(device_indices=torch.arange(3294), last_device_node="node")

        match = extract_decode_match(match_prefix_for_req)
        queue = NS()
        queue.glm53_decode_prefix_profile = profile
        queue.tree_cache = NS(
            supports_mamba=lambda: False,
            inc_lock_ref=lambda node: NS(to_dec_params=lambda: "receipt"),
        )
        queue._pre_alloc_fill_len = lambda req: 4096
        queue._build_decode_prefix_match = lambda req, result: NS(
            prefix_indices=result.device_indices
        )
        req = NS(origin_input_ids=list(range(4096)), lock_receipt=None)

        prefix_match = match(queue, req)

        self.assertFalse(calls[0][1]["cow_mamba"])
        self.assertEqual(len(prefix_match.prefix_indices), 3072)
        self.assertEqual(req.lock_receipt, "receipt")

    def test_generic_mamba_path_keeps_existing_cow(self):
        calls = []

        def match_prefix_for_req(*args, **kwargs):
            calls.append((args, kwargs))
            return NS(device_indices=torch.empty(0), last_device_node="node")

        match = extract_decode_match(match_prefix_for_req)
        queue = NS()
        queue.glm53_decode_prefix_profile = None
        queue.tree_cache = NS(
            supports_mamba=lambda: True,
            inc_lock_ref=lambda node: NS(to_dec_params=lambda: "receipt"),
        )
        queue._build_decode_prefix_match = lambda req, result: "match"
        req = NS(origin_input_ids=[1], lock_receipt=None)

        self.assertEqual(match(queue, req), "match")
        self.assertTrue(calls[0][1]["cow_mamba"])


if __name__ == "__main__":
    unittest.main()
