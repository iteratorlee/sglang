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


class BatchShapeAuditCPU(unittest.TestCase):
    def setUp(self):
        self.cache = cache_tests.ShapeCacheCpuTests()
        self.cache.setUp()
        self.addCleanup(self.cache.doCleanups)
        self.module = self.cache.module
        self.group = self.cache.group
        mock.patch.dict(
            os.environ, {FLAG: "1", "SGLANG_GLM53_NORMAL_ALL_TO_ALL": "0"}
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


if __name__ == "__main__":
    unittest.main()
