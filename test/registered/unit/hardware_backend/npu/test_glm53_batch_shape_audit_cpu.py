"""CPU-only target-forward certificate checks; reuse the normal HCCL fixtures.

The production module and model/decoder forward bodies run without importing
NPU/CUDA runtimes. Collectives are spies, not a distributed liveness test.
"""

import ast
import os
from contextlib import nullcontext
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "manual"))
import test_glm53_normal_idle_dp_cpu as idle
import test_glm53_normal_shape_cache_cpu as cache_tests

FLAG = "SGLANG_GLM53_NORMAL_BATCH_SHAPE_AUDIT"
SMALL = FLAG + "_SMALL"


class BatchShapeAuditCPU(unittest.TestCase):
    def setUp(self):
        self.cache = cache_tests.ShapeCacheCpuTests()
        self.cache.setUp()
        self.addCleanup(self.cache.doCleanups)
        self.module = self.cache.module
        self.group = self.cache.group
        mock.patch.dict(
            os.environ, {FLAG: "1", SMALL: "0", "SGLANG_GLM53_NORMAL_ALL_TO_ALL": "0"}
        ).start()
        communicator = ModuleType("sglang.srt.layers.communicator")
        communicator.ScatterMode = NS(SCATTERED=object())
        mhc = ModuleType("sglang.srt.layers.communicator_mhc")
        mhc.AscendMHCLayerCommunicator = type("AscendMHCLayerCommunicator", (), {})
        mock.patch.dict(
            sys.modules, {communicator.__name__: communicator, mhc.__name__: mhc}
        ).start()
        self.layers = [
            NS(
                layer_id=i,
                is_nextn=False,
                hidden_size=4096,
                is_layer_sparse=i >= 3,
                _normal_skip_idle_dp=i >= 3,
                layer_communicator=mhc.AscendMHCLayerCommunicator(),
                layer_scatter_modes=NS(mlp_mode=communicator.ScatterMode.SCATTERED),
                mlp=NS(top_k=8, num_fused_shared_experts=0),
                config=NS(n_routed_experts=288, num_hidden_layers=45,
                          first_k_dense_replace=3, moe_layer_freq=1),
            )
            for i in range(45)
        ]

    def batch(self, counts=(4096, 0), mode=idle.Mode.EXTEND):
        fb = idle.batch(counts, mode=mode)
        fb.can_run_tbo = False
        return fb

    def scope(self, fb):
        return self.module.normal_hccl_batch_shape_audit_scope(fb, self.layers)

    def check(self, fb, **changes):
        rows = fb.global_num_tokens_cpu[0] // 8
        tensors = [
            torch.empty(rows, 4096, dtype=torch.bfloat16),
            torch.empty(rows, 8, dtype=torch.int64),
            torch.empty(rows, 8, dtype=torch.float32),
        ]
        for index, name in enumerate(("x", "ids", "weights")):
            if name in changes:
                tensors[index] = changes[name]
        self.module._check_equal_dispatch_shapes_cached(
            changes.get("group", self.group), *tensors, changes.get("world", 16)
        )

    def layer(self, fb, layer_id, **changes):
        with self.module.normal_hccl_idle_dp_scope(fb, layer_id=layer_id):
            self.check(fb, **changes)

    def forward(self, fb):
        with self.scope(fb):
            for i in range(3, 45):
                self.layer(fb, i)
        self.assert_clean()

    def events(self):
        return [call[0] for call in self.cache.calls]

    def assert_clean(self):
        self.assertIsNone(self.module._batch_shape_epoch.get())
        self.assertIsNone(self.module._layer_shape_epoch.get())
        self.assertIsNone(self.module._idle_dp_counts.get())

    def test_42_layers_one_vote_and_same_fb_reused_for_two_chunks(self):
        fb = self.batch()
        self.forward(fb)
        self.forward(fb)
        self.assertEqual(self.events(), ["vote", "audit", "vote"])
        self.assertEqual([c[2] for c in self.cache.calls if c[0] == "vote"], [1, 0])
        self.assertFalse(any("shape" in key for key in vars(fb)))

    def test_46_layers_four_dense_scopes_before_first_moe(self):
        template = self.layers[0]
        self.layers.append(NS(**vars(template)))
        for i, layer in enumerate(self.layers):
            layer.layer_id = i
            layer.is_layer_sparse = i >= 4
            layer._normal_skip_idle_dp = True
            layer.config = NS(n_routed_experts=288, num_hidden_layers=46,
                              first_k_dense_replace=4, moe_layer_freq=1)
        fb = self.batch()
        for _ in range(2):
            with self.scope(fb):
                for i in range(4):
                    with self.module.normal_hccl_idle_dp_scope(fb, layer_id=i):
                        self.assertIsNone(self.module._batch_shape_epoch.get().signature)
                for i in range(4, 46):
                    self.layer(fb, i)
        self.assertEqual(self.events(), ["vote", "audit", "vote"])
        self.assert_clean()
        with self.assertRaisesRegex(RuntimeError, "outside target MoE"):
            with self.scope(fb):
                self.layer(fb, 0)
        self.assertEqual(self.events(), ["vote", "audit", "vote"])
        self.assert_clean()

    def test_all_16_rank_control_paths_use_the_shared_raw_vector(self):
        parallel = self.cache.fixture.parallel
        for counts in ((4096, 0), (0, 4096), (4096, 4096)):
            for rank in range(16):
                parallel.moe_ep_group.rank_in_group = rank
                parallel.attn_dp_rank, parallel.attn_tp_rank = divmod(rank, 8)
                self.module._dispatch_shape_cache.clear()
                self.cache.calls.clear()
                self.forward(self.batch(counts))
                self.assertEqual(self.events(), ["vote", "audit"])

    def test_new_forward_shapes_and_dp_ownership_can_change(self):
        fb = self.batch()
        for counts in ((4096, 0), (0, 4096), (8192, 4097), (4096, 0)):
            replacement = self.batch(counts)
            fb.original_global_num_tokens_cpu = (
                replacement.original_global_num_tokens_cpu
            )
            fb.global_num_tokens_cpu = replacement.global_num_tokens_cpu
            self.forward(fb)
        self.assertEqual(
            self.events(), ["vote", "audit", "vote", "vote", "audit", "vote", "audit"]
        )

    def test_remote_invalidation_on_every_new_forward_is_preserved(self):
        fb = self.batch()
        self.forward(fb)
        self.cache.remote_dirty = True
        self.forward(fb)
        self.assertEqual(self.events(), ["vote", "audit", "vote", "audit"])
        self.cache.remote_shape = [511, 4096, 2, 511, 8, 2, 511, 8]
        with self.assertRaisesRegex(RuntimeError, "EP16 requires equal"):
            self.forward(fb)
        self.assertNotIn(self.group, self.module._dispatch_shape_cache)
        self.assert_clean()
        self.cache.remote_dirty = False
        self.cache.remote_shape = None
        self.forward(fb)
        self.assertEqual(self.events()[-2:], ["vote", "audit"])

    def test_raw_threshold_not_padded_threshold_and_both_dp_domains(self):
        for counts, expected_votes in (
            ((4095, 0), 42),
            ((0, 4095), 42),
            ((4096, 0), 1),
            ((0, 4096), 1),
        ):
            with self.subTest(counts=counts):
                self.cache.calls.clear()
                fb = self.batch(counts)
                self.assertEqual(fb.global_num_tokens_cpu, [4096, 4096])
                self.forward(fb)
                self.assertEqual(self.events().count("vote"), expected_votes)

    def test_default_off_and_cache_dependency(self):
        fb = self.batch()
        for value in (None, "0", "true", "2"):
            with mock.patch.dict(os.environ):
                if value is None:
                    os.environ.pop(FLAG, None)
                else:
                    os.environ[FLAG] = value
                self.cache.calls.clear()
                self.forward(fb)
                self.assertEqual(self.events().count("vote"), 42)
        with mock.patch.dict(os.environ, {cache_tests.FLAG: "0"}):
            self.cache.calls.clear()
            self.forward(fb)
            self.assertEqual(self.events(), ["audit"] * 42)

    def test_decode_nonextend_overlap_and_other_profiles_are_excluded(self):
        for name, value in (
            ("disaggregation_mode", "decode"),
            ("nnodes", 1),
            ("enable_single_batch_overlap", True),
            ("disable_cuda_graph", False),
        ):
            with (
                self.subTest(name=name),
                mock.patch.object(self.cache.fixture.args, name, value),
            ):
                self.cache.calls.clear()
                self.forward(self.batch())
                self.assertEqual(self.events(), ["audit"] * 42)
        for mode in idle.Mode:
            if mode != idle.Mode.EXTEND:
                self.cache.calls.clear()
                self.forward(self.batch(mode=mode))
                self.assertEqual(self.events(), ["audit"] * 42)

    def test_nextn_without_layer_id_uses_old_vote_even_inside_target_scope(self):
        fb = self.batch()
        with self.scope(fb):
            self.layer(fb, 3)
            with self.module.normal_hccl_idle_dp_scope(fb):
                self.assertIsNone(self.module._active_batch_shape_epoch())
                self.check(fb)
            for i in range(4, 45):
                self.layer(fb, i)
        with self.module.normal_hccl_idle_dp_scope(fb):
            self.check(fb)
        self.assertEqual(self.events(), ["vote", "audit", "vote", "vote"])
        self.assert_clean()

    def test_asymmetric_local_change_raises_without_fallback_vote_or_payload(self):
        # Model one rank's corruption after peers have already earned a cert.
        # This proves no mismatched *audit* is submitted; it does not claim
        # that healthy peers blocked in payload HCCL can recover without abort.
        changes = (
            {"x": torch.empty(511, 4096, dtype=torch.bfloat16)},
            {"x": torch.empty(512, 4095, dtype=torch.bfloat16)},
            {"x": torch.empty(512, 4096, dtype=torch.float32)},
            {"ids": torch.empty(512, 7, dtype=torch.int64)},
            {"ids": torch.empty(512, 8, dtype=torch.int32)},
            {"weights": torch.empty(512, 8, dtype=torch.bfloat16)},
            {"x": torch.empty(512, 4096, dtype=torch.bfloat16, device="meta")},
            {"group": object()},
            {"world": 8},
        )
        for change in changes:
            with self.subTest(change=list(change)):
                fb = self.batch()
                self.cache.calls.clear()
                with self.assertRaisesRegex(RuntimeError, "certificate violated"):
                    with self.scope(fb):
                        self.layer(fb, 3)
                        before = list(self.cache.calls)
                        self.layer(fb, 4, **change)
                self.assertEqual(self.cache.calls, before)
                self.assertNotIn(self.group, self.module._dispatch_shape_cache)
                self.assert_clean()

    def test_cache_eviction_or_equal_replacement_is_fatal_inside_epoch(self):
        for replace in (False, True):
            fb = self.batch()
            with self.assertRaisesRegex(RuntimeError, "global cache changed"):
                with self.scope(fb):
                    self.layer(fb, 3)
                    old = self.module._dispatch_shape_cache.pop(self.group)
                    if replace:
                        self.module._dispatch_shape_cache[self.group] = tuple(list(old))
                    before = list(self.cache.calls)
                    self.layer(fb, 4)
            self.assertEqual(self.cache.calls, before)
            self.assert_clean()

    def test_missing_layer_scope_cannot_fall_back_to_a_vote(self):
        fb = self.batch()
        with self.assertRaisesRegex(RuntimeError, "outside target MLP scope"):
            with self.scope(fb):
                self.layer(fb, 3)
                before = list(self.cache.calls)
                self.check(fb)
        self.assertEqual(self.cache.calls, before)
        self.assert_clean()

    def test_real_dispatch_rejects_changed_world_before_quantization(self):
        fb = self.batch()
        with (
            mock.patch.object(self.module.dist, "get_world_size", return_value=4),
            mock.patch.object(self.module.dist, "get_rank", return_value=0),
            mock.patch.object(torch.ops.npu, "npu_dynamic_quant", create=True) as quant,
        ):
            with self.assertRaisesRegex(RuntimeError, "dispatch world"):
                with self.scope(fb):
                    with self.module.normal_hccl_idle_dp_scope(fb, layer_id=3):
                        self.module.collective_dispatch(
                            self.group,
                            torch.empty(512, 4096),
                            torch.empty(512, 8),
                            torch.empty(512, 8),
                        )
            quant.assert_not_called()
        self.assertEqual(self.events(), [])
        self.assert_clean()

    def test_rank_mapping_change_cannot_reuse_certificate(self):
        fb = self.batch()
        with self.assertRaisesRegex(RuntimeError, "certificate"):
            with self.scope(fb):
                self.layer(fb, 3)
                before = list(self.cache.calls)
                with mock.patch.object(self.cache.fixture.parallel, "attn_tp_rank", 1):
                    self.layer(fb, 4)
        self.assertEqual(self.cache.calls, before)
        self.assert_clean()

    def test_payload_values_order_and_four_gathers_remain_identical(self):
        fb = self.batch()
        x = torch.ones(512, 4096, dtype=torch.bfloat16)
        ids = torch.arange(512 * 8, dtype=torch.int64).reshape(512, 8) % 288
        weights = torch.ones(512, 8, dtype=torch.float32) / 8
        gathers = []

        def gather(out, value, group):
            gathers.append((value.dtype, tuple(value.shape)))
            out.copy_(value.repeat(16, *([1] * (value.ndim - 1))))

        def dispatch():
            return self.module.collective_dispatch(self.group, x, ids, weights)

        with (
            mock.patch.object(self.module.dist, "get_world_size", return_value=16),
            mock.patch.object(self.module.dist, "get_rank", return_value=0),
            mock.patch.object(
                self.module.dist, "all_gather_into_tensor", side_effect=gather
            ),
            mock.patch.object(
                torch.ops.npu,
                "npu_dynamic_quant",
                side_effect=cache_tests.dynamic_quant_reference,
                create=True,
            ),
            mock.patch.dict(os.environ, {"SGLANG_GLM53_NORMAL_DEVICE_ROUTING": "0"}),
        ):
            with self.scope(fb):
                outputs = []
                for i in (3, 4):
                    with self.module.normal_hccl_idle_dp_scope(fb, layer_id=i):
                        outputs.append(dispatch())
                for i in range(5, 45):
                    self.layer(fb, i)
            self.assertEqual(self.events(), ["vote"])
            payloads = [
                (torch.int8, (512, 4096)),
                (torch.float32, (512,)),
                (torch.int64, (512, 8)),
                (torch.float32, (512, 8)),
            ]
            self.assertEqual(gathers, [(torch.int64, (8,))] + payloads * 2)
            with self.module.normal_hccl_idle_dp_scope(fb):
                reference = dispatch()
            self.assertEqual(gathers[-4:], payloads)
        for output in outputs:
            # Production dispatch includes the exact sorted routing state.
            torch.testing.assert_close(output, reference, rtol=0, atol=0)
        self.assert_clean()

    def test_batch_metadata_epoch_and_layer_order_changes_fail_closed(self):
        for change in (
            "original",
            "padded",
            "batch",
            "skip",
            "repeat",
            "flag",
            "cache_flag",
        ):
            fb = self.batch()
            with self.subTest(change=change), mock.patch.dict(os.environ):
                with self.assertRaisesRegex(RuntimeError, "certificate violated"):
                    with self.scope(fb):
                        self.layer(fb, 3)
                        before = list(self.cache.calls)
                        if change == "original":
                            fb.original_global_num_tokens_cpu[:] = [0, 4096]
                        elif change == "padded":
                            fb.global_num_tokens_cpu[:] = [8192, 8192]
                        elif change == "batch":
                            fb = self.batch()
                        elif change == "flag":
                            os.environ[FLAG] = "0"
                        elif change == "cache_flag":
                            os.environ[cache_tests.FLAG] = "0"
                        self.layer(
                            fb,
                            5 if change == "skip" else 3 if change == "repeat" else 4,
                        )
                self.assertEqual(self.cache.calls, before)
                self.assert_clean()

    def test_missing_duplicate_dispatch_and_partial_loop_are_fatal(self):
        fb = self.batch()
        for kind in ("missing", "duplicate", "partial", "nested"):
            with (
                self.subTest(kind=kind),
                self.assertRaisesRegex(RuntimeError, "certificate"),
            ):
                with self.scope(fb):
                    if kind == "partial":
                        self.layer(fb, 3)
                    elif kind == "nested":
                        with self.scope(fb):
                            self.fail("nested target accepted")
                    else:
                        with self.module.normal_hccl_idle_dp_scope(fb, layer_id=3):
                            if kind == "duplicate":
                                self.check(fb)
                                self.check(fb)
            self.assert_clean()

    def test_exceptions_restore_context_and_poison_caught_failures(self):
        fb = self.batch()
        with self.assertRaisesRegex(ValueError, "MLP failure"):
            with self.scope(fb):
                with self.module.normal_hccl_idle_dp_scope(fb, layer_id=3):
                    self.check(fb)
                    raise ValueError("MLP failure")
        self.assert_clean()
        self.assertNotIn(self.group, self.module._dispatch_shape_cache)
        self.forward(fb)
        with self.assertRaisesRegex(RuntimeError, "certificate"):
            with self.scope(fb):
                with self.module.normal_hccl_idle_dp_scope(fb, layer_id=3):
                    self.check(fb)
                    with self.assertRaisesRegex(RuntimeError, "certificate"):
                        self.check(fb)  # caught errors still poison the epoch
        self.assert_clean()

    def test_model_contract_and_parallel_mapping_fail_before_collectives(self):
        for target, field, value in (
            (self.layers[3], "is_nextn", True),
            (self.layers[3], "hidden_size", 2048),
            (self.layers[3].mlp, "num_fused_shared_experts", 1),
            (self.layers[3], "_normal_skip_idle_dp", False),
            (self.cache.fixture.parallel, "attn_tp_size", 4),
        ):
            with mock.patch.object(target, field, value):
                with self.assertRaisesRegex(RuntimeError, "requires"):
                    self.forward(self.batch())
        self.assertEqual(self.events(), [])

    def test_actual_target_model_forward_owns_fresh_scope_and_unwinds(self):
        tree = ast.parse((idle.SRT / "models/glm5_next.py").read_text())
        owner = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "Glm5NextModel"
        )
        method = next(
            n
            for n in owner.body
            if isinstance(n, ast.FunctionDef) and n.name == "forward"
        )
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                method,
            ],
            type_ignores=[],
        )
        namespace = dict(
            os=os,
            torch=torch,
            nullcontext=nullcontext,
            BumpAllocator=lambda **kw: None,
            check_cuda_graph_backend=lambda *args: False,
            Phase=NS(PREFILL=1),
            Backend=NS(TC_PIECEWISE=2),
            get_global_expert_distribution_recorder=lambda: NS(
                with_current_layer=lambda i: nullcontext()
            ),
        )
        exec(
            compile(
                ast.fix_missing_locations(module), "<target model forward>", "exec"
            ),
            namespace,
        )
        case = self

        class Layer(NS):
            def __call__(self, positions, hidden, fb, residual, *args, **kwargs):
                if self.is_layer_sparse:
                    case.layer(fb, self.layer_id)
                if self.layer_id == getattr(fb, "fail_layer", -1):
                    raise ValueError("layer failure")
                return hidden, residual, None

        model = NS(
            start_layer=0,
            end_layer=45,
            first_k_dense_replace=3,
            pp_group=NS(is_first_rank=True, is_last_rank=True),
            embed_tokens=lambda ids: torch.zeros(1),
            norm=lambda x: x,
            layers=[Layer(**vars(layer)) for layer in self.layers],
            layers_to_capture=[],
            dflash_capture=False,
        )
        fb = self.batch()
        forward = namespace["forward"]
        forward(model, None, None, fb)
        forward(model, None, None, fb)
        self.assertEqual(self.events(), ["vote", "audit", "vote"])
        fb.fail_layer = 4
        with self.assertRaisesRegex(ValueError, "layer failure"):
            forward(model, None, None, fb)
        self.assert_clean()
        del fb.fail_layer
        forward(model, None, None, fb)
        self.assertEqual(self.events()[-2:], ["vote", "audit"])

    def test_actual_decoder_passes_target_layer_id_but_masks_nextn(self):
        fixture = self.cache.fixture
        layer, forward = fixture.decoder_forward()
        seen = []
        original = self.module.normal_hccl_idle_dp_scope

        def scope(fb, ctx, *, layer_id=None):
            seen.append(layer_id)
            return original(fb, ctx, layer_id=layer_id)

        with mock.patch.object(
            self.module, "normal_hccl_idle_dp_scope", side_effect=scope
        ):
            forward(layer, None, torch.zeros(2, 3), self.batch(), None)
            layer.is_nextn = True
            forward(layer, None, torch.zeros(2, 3), self.batch(), None)
        self.assertEqual(seen, [3, None])
        self.assert_clean()


    def test_small_all_16_ranks_boundary_sizes_and_asymmetric_vectors(self):
        parallel = self.cache.fixture.parallel
        certified = 0
        for tokens in (1, 7, 8, 9, 63, 128, 180, 255, 256, 511, 512, 4095, 4096):
            for counts in ((tokens, 0), (0, tokens), (tokens, max(1, tokens // 3))):
                for rank in range(16):
                    parallel.moe_ep_group.rank_in_group = rank
                    parallel.attn_dp_rank, parallel.attn_tp_rank = divmod(rank, 8)
                    for enabled in (False, True):
                        with self.subTest(tokens=tokens, counts=counts, rank=rank, enabled=enabled), mock.patch.dict(os.environ, {SMALL: str(int(enabled))}):
                            self.module._dispatch_shape_cache.clear()
                            self.cache.calls.clear()
                            self.forward(self.batch(counts))
                            want = 1 if tokens >= 4096 or enabled else 42
                            self.assertEqual(self.events().count("vote"), want)
                            self.assertEqual(self.events().count("audit"), 1)
                            self.assertEqual(self.events()[:2], ["vote", "audit"])
                            certified += int(enabled and tokens < 4096)
        self.assertEqual(self.module.get_small_batch_shape_audit_stats(), {
            "short_certified_forwards": certified, "saved_layer_votes": certified * 41})

    def test_small_exact_optin_zero_and_original_flags_remain_required(self):
        for value in (None, "0", "", "true", "2"):
            with mock.patch.dict(os.environ):
                os.environ.pop(SMALL, None) if value is None else os.environ.__setitem__(SMALL, value)
                self.cache.calls.clear()
                self.forward(self.batch((0, 180)))
                self.assertEqual(self.events().count("vote"), 42)
        with mock.patch.dict(os.environ, {SMALL: "1"}):
            self.cache.calls.clear()
            self.forward(self.batch((0, 0)))
            self.assertEqual(self.events().count("vote"), 42)
            for flag in (FLAG, cache_tests.FLAG):
                with mock.patch.dict(os.environ, {flag: "0"}):
                    self.cache.calls.clear()
                    self.forward(self.batch((0, 180)))
                    self.assertEqual(self.events().count("vote" if flag == FLAG else "audit"), 42)
        self.assertEqual(self.module.get_small_batch_shape_audit_stats()["short_certified_forwards"], 0)

    def test_small_new_epoch_remote_dirty_mismatch_and_failure_cleanup(self):
        with mock.patch.dict(os.environ, {SMALL: "1"}):
            fb = self.batch((0, 180))
            self.forward(fb)
            self.forward(fb)
            self.forward(self.batch((180, 0)))
            self.assertEqual(self.events(), ["vote", "audit", "vote", "vote"])
            self.cache.remote_dirty = True
            self.forward(self.batch((180, 1)))
            self.assertEqual(self.events()[-2:], ["vote", "audit"])
            self.cache.remote_shape = [22, 4096, 2, 22, 8, 2, 22, 8]
            with self.assertRaisesRegex(RuntimeError, "EP16 requires equal"):
                self.forward(self.batch((0, 180)))
            self.assertNotIn(self.group, self.module._dispatch_shape_cache)
            self.assert_clean()
            self.assertEqual(self.module.get_small_batch_shape_audit_stats()["short_certified_forwards"], 4)
            self.cache.remote_shape = None
            self.cache.remote_dirty = False
            for failure in ("raise", "metadata", "small_flag", "missing", "repeat"):
                with self.subTest(failure=failure), mock.patch.dict(os.environ):
                    fb = self.batch((0, 180))
                    with self.assertRaises((RuntimeError, ValueError)):
                        with self.scope(fb):
                            self.layer(fb, 3)
                            before = list(self.cache.calls)
                            if failure == "raise":
                                raise ValueError("injected failure")
                            if failure == "metadata":
                                fb.original_global_num_tokens_cpu[:] = [180, 0]
                            if failure == "small_flag":
                                os.environ[SMALL] = "0"
                            self.layer(fb, 5 if failure == "missing" else 3 if failure == "repeat" else 4)
                    self.assertEqual(before, self.cache.calls)
                    self.assert_clean()
                    self.assertNotIn(self.group, self.module._dispatch_shape_cache)
            self.assertEqual(self.module.get_small_batch_shape_audit_stats()["short_certified_forwards"], 4)
            self.forward(self.batch((0, 180)))
            self.assertEqual(self.events()[-2:], ["vote", "audit"])
            self.assertEqual(self.module.get_small_batch_shape_audit_stats()["saved_layer_votes"], 205)

    def test_small_draft_decode_and_nextn_keep_fallback(self):
        with mock.patch.dict(os.environ, {SMALL: "1"}):
            for mode in idle.Mode:
                if mode == idle.Mode.EXTEND:
                    continue
                self.cache.calls.clear()
                self.forward(self.batch((0, 180), mode=mode))
                self.assertEqual(self.events(), ["audit"] * 42)
            # Existing production decoder AST test proves NextN passes None.
            self.test_actual_decoder_passes_target_layer_id_but_masks_nextn()
            fb = self.batch((0, 180))
            self.cache.calls.clear()
            with self.scope(fb):
                self.layer(fb, 3)
                with self.module.normal_hccl_idle_dp_scope(fb):
                    self.assertIsNone(self.module._active_batch_shape_epoch())
                    self.check(fb)
                for i in range(4, 45):
                    self.layer(fb, i)
            with self.module.normal_hccl_idle_dp_scope(fb):
                self.check(fb)
            self.assertEqual(self.events().count("vote"), 3)
            self.assertEqual(self.module.get_small_batch_shape_audit_stats()["short_certified_forwards"], 1)
            nextn = (idle.SRT / "models/deepseek_nextn.py").read_text()
            self.assertNotIn("normal_hccl_batch_shape_audit_scope", nextn)
            glm_nextn = (idle.SRT / "models/glm5_next_nextn.py").read_text()
            self.assertIn("(DeepseekV3ForCausalLMNextN)", glm_nextn)
            # A malformed target with a NextN layer must fail before a vote.
            before = list(self.cache.calls)
            with mock.patch.object(self.layers[3], "is_nextn", True):
                with self.assertRaisesRegex(RuntimeError, "42-MoE"):
                    self.forward(fb)
            self.assertEqual(self.cache.calls, before)
            self.assert_clean()

    def test_small_actual_target_model_scope_and_stats_snapshot(self):
        # Run existing production target-forward AST with short batches.
        old_batch = self.batch
        with mock.patch.dict(os.environ, {SMALL: "1"}), mock.patch.object(
            self, "batch", side_effect=lambda: old_batch((0, 180))
        ):
            self.test_actual_target_model_forward_owns_fresh_scope_and_unwinds()
        stats = self.module.get_small_batch_shape_audit_stats()
        self.assertEqual(stats, {"short_certified_forwards": 3, "saved_layer_votes": 123})
        stats["short_certified_forwards"] = -1
        self.assertEqual(self.module.get_small_batch_shape_audit_stats()["short_certified_forwards"], 3)

    def test_small_dynamic_expert_counts_and_four_payloads_are_not_cached(self):
        fb = self.batch((180, 0))
        rows = 23
        x = torch.ones(rows, 4096, dtype=torch.bfloat16)
        ids = [torch.full((rows, 8), i, dtype=torch.int64) for i in (0, 1)]
        weights = [torch.full((rows, 8), w, dtype=torch.float32) for w in (0.125, 0.0625)]
        gathers = []

        def gather(out, value, group):
            gathers.append((value.dtype, tuple(value.shape)))
            out.copy_(value.repeat(16, *([1] * (value.ndim - 1))))

        def dispatch(i):
            return self.module.collective_dispatch(self.group, x, ids[i], weights[i])

        with mock.patch.dict(os.environ, {SMALL: "1", "SGLANG_GLM53_NORMAL_DEVICE_ROUTING": "0"}), mock.patch.object(self.module.dist, "get_world_size", return_value=16), mock.patch.object(self.module.dist, "get_rank", return_value=0), mock.patch.object(self.module.dist, "all_gather_into_tensor", side_effect=gather), mock.patch.object(torch.ops.npu, "npu_dynamic_quant", side_effect=cache_tests.dynamic_quant_reference, create=True):
            with self.scope(fb):
                outputs = []
                for i in (0, 1):
                    with self.module.normal_hccl_idle_dp_scope(fb, layer_id=3+i):
                        outputs.append(dispatch(i))
                for layer_id in range(5, 45):
                    self.layer(fb, layer_id)
            self.assertEqual(self.events(), ["vote"])
            payloads = [(torch.int8, (rows, 4096)), (torch.float32, (rows,)), (torch.int64, (rows, 8)), (torch.float32, (rows, 8))]
            self.assertEqual(gathers, [(torch.int64, (8,))] + payloads * 2)
            self.assertNotEqual(outputs[0][2], outputs[1][2])
            for i in (0, 1):
                with self.module.normal_hccl_idle_dp_scope(fb):
                    reference = dispatch(i)
                torch.testing.assert_close(outputs[i], reference, rtol=0, atol=0)
            self.assertEqual(gathers[-8:], payloads * 2)
        self.assert_clean()

    def test_small_actual_idle_conversion_is_uniform_on_all_16_peers(self):
        # Execute the complete frozen production padding method on CPU, then
        # feed that exact ForwardBatch to the real candidate scope.
        tree = ast.parse((idle.SRT / "model_executor/forward_batch_info.py").read_text())
        owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ForwardBatch")
        method = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "prepare_mlp_sync_batch")
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method], type_ignores=[])
        parallel = self.cache.fixture.parallel
        max_len = NS(is_max_len=lambda: True)
        namespace = dict(torch=torch, ForwardMode=idle.Mode, _is_cpu=True,
            get_parallel=lambda: parallel,
            get_exec=lambda: NS(graph=NS(cuda_graph_config=NS(prefill=NS(bs=[])))),
            ceil_align=lambda n, unit: (n+unit-1)//unit*unit,
            DpPaddingMode=NS(get_dp_padding_mode=lambda *a: max_len),
            _elastic_should_preserve_local_token_counts=lambda **kw: False,
            mambaish_config=lambda cfg: object(), set_dp_buffer_len=mock.Mock(),
            set_is_extend_in_batch=mock.Mock())
        exec(compile(ast.fix_missing_locations(module), "<frozen actual padding>", "exec"), namespace)
        tbo=ModuleType("sglang.srt.batch_overlap.two_batch_overlap")
        tbo.TboForwardBatchPreparer=NS(prepare=lambda **kw: None)
        with mock.patch.dict(os.environ, {SMALL:"1"}), mock.patch.dict(sys.modules,{tbo.__name__:tbo}), mock.patch.object(idle.Mode, "is_extend", lambda mode: mode == idle.Mode.EXTEND, create=True):
            vectors = [counts for size in (1, 7, 8, 9, 63, 128, 180, 255, 256, 511, 512, 4095, 4096)
                       for counts in ((0, size), (size, 0), (min(7, size), size))]
            for counts in vectors:
                for rank in range(16):
                    parallel.moe_ep_group.rank_in_group=rank
                    parallel.attn_dp_rank,parallel.attn_tp_rank=divmod(rank,8)
                    local=counts[rank//8]
                    fb=self.batch(counts,mode=idle.Mode.EXTEND if local else idle.Mode.IDLE)
                    original=fb.original_global_num_tokens_cpu
                    fb.global_num_tokens_cpu=list(counts)
                    fb.global_num_tokens_for_logprob_cpu=list(counts)
                    fb.global_num_tokens_gpu=torch.tensor(counts)
                    fb.batch_size=int(local>0);fb.is_extend_in_batch=True
                    fb.extend_num_tokens=local
                    fb.can_run_dp_prefill_cuda_graph=False;fb.spec_info=None
                    fb.seq_lens=torch.tensor([local] if local else [],dtype=torch.int64)
                    fb.orig_seq_lens=fb.seq_lens.clone();fb.seq_lens_cpu=None
                    fb.global_num_token_non_padded=None;fb.tbo_children=None
                    fb._pad_inputs_to_size=mock.Mock()
                    runner=NS(model_config=NS(),is_draft_worker=False,attn_tp_sequence_sharded=lambda n:True)
                    namespace[method.name](fb,runner)
                    self.assertEqual(fb.forward_mode,idle.Mode.EXTEND)
                    self.assertIs(fb.original_global_num_tokens_cpu,original)
                    self.assertEqual(original,list(counts))
                    capacity = (max(counts) + 7) // 8 * 8
                    self.assertEqual(fb.global_num_tokens_cpu,[capacity,capacity])
                    if not local:self.assertEqual(fb._original_forward_mode,idle.Mode.IDLE)
                    self.module._dispatch_shape_cache.clear();self.cache.calls.clear()
                    self.forward(fb)
                    self.assertEqual(self.events(),["vote","audit"])


if __name__ == "__main__":
    unittest.main()
