"""CPU checks of GLM's scoped idle-DP publication and normal HCCL dispatch.

CUDA_VISIBLE_DEVICES='' TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  /home/iterator/anaconda3/envs/sgl-0511/bin/python -B \
  test/manual/test_glm53_normal_idle_dp_cpu.py -v

Loads the production collectives module and executes the production GLM decoder
forward method with dependency stubs. Quantization/collectives are CPU mocks;
the contribution comparison establishes routing identity, not NPU numerics.
"""

import ast
import enum
import os
import sys
import unittest
from contextlib import contextmanager, nullcontext
from types import ModuleType, SimpleNamespace as NS
from unittest import mock

import torch

from test_glm53_normal_hccl_cpu import (
    COLLECTIVES_NAME,
    SRT,
    args_for_profile,
    dynamic_quant_reference,
    load_source,
)

FLAG = "SGLANG_GLM53_NORMAL_SKIP_IDLE_DP"


class Mode(enum.Enum):
    EXTEND = 1
    DECODE = 2
    IDLE = 3
    TARGET_VERIFY = 4
    DRAFT_EXTEND = 5
    MIXED = 6

    def is_decode(self):
        return self == Mode.DECODE

    def is_target_verify(self):
        return self == Mode.TARGET_VERIFY

    def is_draft_extend_v2(self):
        return self == Mode.DRAFT_EXTEND

    def is_idle(self):
        return self == Mode.IDLE


def batch(counts=(32, 0), *, mode=Mode.EXTEND):
    capacity = (max(counts) + 7) // 8 * 8
    return NS(
        original_global_num_tokens_cpu=list(counts),
        global_num_tokens_cpu=[capacity, capacity],
        dp_padding_mode=NS(is_max_len=lambda: True),
        forward_mode=mode,
    )


class IdleDpCpuTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(mock.patch.stopall)
        mock.patch.dict(
            os.environ,
            {
                FLAG: "1",
                "SGLANG_GLM53_NORMAL_HCCL": "1",
                "SGLANG_GLM53_2NODE_PD": "1",
                "SGLANG_GLM53_NORMAL_MEMORY_HEADROOM_MB": "0",
                "SGLANG_GLM53_NORMAL_COMBINE_ORDERED": "0",
                "SGLANG_GLM53_NORMAL_COMBINE_DETERMINISTIC": "1",
                "SGLANG_GLM53_NORMAL_REDUCE_SCATTER": "0",
                "SGLANG_GLM53_NORMAL_SKIP_IDLE_ATTN": "0",
            },
        ).start()
        self.module = load_source(
            COLLECTIVES_NAME, SRT / "hardware_backend/npu/moe/glm53_collectives.py"
        )
        self.args = args_for_profile(
            nnodes=2,
            tp_size=16,
            ep_size=16,
            dp_size=2,
            deepep_mode="normal",
            disable_cuda_graph=True,
            enable_single_batch_overlap=False,
        )
        self.group = object()
        self.parallel = NS(
            moe_ep_group=NS(
                device_group=self.group,
                ranks=list(range(16)),
                world_size=16,
                rank_in_group=0,
            ),
            tp_group=NS(ranks=list(range(16))),
            attn_dp_size=2,
            attn_tp_size=8,
            attn_dp_rank=0,
            attn_tp_rank=0,
        )
        runtime = ModuleType("sglang.srt.runtime_context")
        runtime.get_server_args = lambda: self.args
        runtime.get_parallel = lambda: self.parallel
        forward = ModuleType("sglang.srt.model_executor.forward_batch_info")
        forward.ForwardMode = Mode
        idle_attention = load_source(
            "sglang.srt.hardware_backend.npu.moe.glm53_idle_attention",
            SRT / "hardware_backend/npu/moe/glm53_idle_attention.py",
        )
        mock.patch.dict(
            sys.modules,
            {
                COLLECTIVES_NAME: self.module,
                runtime.__name__: runtime,
                forward.__name__: forward,
                idle_attention.__name__: idle_attention,
            },
        ).start()

    def test_scope_is_immutable_nested_and_resets_on_exception(self):
        outer = batch((32, 0))
        self.assertIsNone(self.module._idle_dp_counts.get())
        with self.module.normal_hccl_idle_dp_scope(outer):
            self.assertEqual(self.module._idle_dp_counts.get(), (32, 0))
            outer.original_global_num_tokens_cpu[:] = [0, 32]
            self.assertEqual(self.module._idle_dp_counts.get(), (32, 0))
            with self.assertRaisesRegex(RuntimeError, "MLP failure"):
                with self.module.normal_hccl_idle_dp_scope(batch((0, 32))):
                    self.assertEqual(self.module._idle_dp_counts.get(), (0, 32))
                    raise RuntimeError("MLP failure")
            self.assertEqual(self.module._idle_dp_counts.get(), (32, 0))
            with self.module.normal_hccl_idle_dp_scope(batch(mode=Mode.DECODE)):
                self.assertIsNone(self.module._idle_dp_counts.get())
            self.assertEqual(self.module._idle_dp_counts.get(), (32, 0))
        self.assertIsNone(self.module._idle_dp_counts.get())

    def test_metadata_errors_do_not_publish_or_reuse_padded_counts(self):
        for name in ("original_global_num_tokens_cpu", "global_num_tokens_cpu"):
            for bad in (
                None,
                [],
                [32],
                [0, 32, 0],
                [-1, 32],
                [True, 32],
                [0, 32.0],
                torch.tensor([0, 32], device="cpu"),
            ):
                with self.subTest(name=name, bad=bad):
                    fb = batch()
                    setattr(fb, name, bad)
                    with self.assertRaisesRegex(ValueError, "CPU .* counts"):
                        with self.module.normal_hccl_idle_dp_scope(fb):
                            self.fail("invalid metadata reached MLP")
                    self.assertIsNone(self.module._idle_dp_counts.get())
        for fb in (
            NS(forward_mode=Mode.EXTEND),
            batch((33, 0)),
            batch(),
        ):
            if hasattr(fb, "global_num_tokens_cpu"):
                fb.global_num_tokens_cpu = [32, 31]
            with self.assertRaises(ValueError):
                with self.module.normal_hccl_idle_dp_scope(fb):
                    self.fail("invalid MAX_LEN geometry reached MLP")
        fb = batch()
        fb.dp_padding_mode.is_max_len = lambda: False
        with self.assertRaisesRegex(ValueError, "MAX_LEN"):
            with self.module.normal_hccl_idle_dp_scope(fb):
                self.fail("SUM_LEN reached MLP")

    def test_default_off_other_profiles_and_decode_do_not_read_metadata(self):
        for value in (None, "0", "", "true", "2"):
            with self.subTest(value=value), mock.patch.dict(os.environ):
                if value is None:
                    os.environ.pop(FLAG, None)
                else:
                    os.environ[FLAG] = value
                self.assertFalse(self.module.normal_hccl_skip_idle_dp_enabled())
                with self.module.normal_hccl_idle_dp_scope(object()):
                    self.assertIsNone(self.module._idle_dp_counts.get())
        rejected = {
            "nnodes": (1, 4),
            "tp_size": (8, 32),
            "ep_size": (8, 32),
            "dp_size": (1, 4, 8),
            "enable_dp_attention": (False,),
            "pp_size": (2,),
            "attn_cp_size": (2,),
            "dcp_size": (2,),
            "moe_dp_size": (2,),
            "dwdp_size": (16,),
            "enable_prefill_cp": (True,),
            "disaggregation_mode": ("null", "decode"),
            "disable_overlap_schedule": (False,),
            "enable_two_batch_overlap": (True,),
            "enable_single_batch_overlap": (True,),
            "enable_eplb": (True,),
            "ep_num_redundant_experts": (1,),
            "quantization": (None, "fp8"),
            "moe_a2a_backend": ("none", "ascend_fuseep"),
            "deepep_mode": ("auto", "low_latency"),
            "disable_cuda_graph": (False,),
        }
        for name, values in rejected.items():
            for value in values:
                with (
                    self.subTest(name=name, value=value),
                    mock.patch.object(self.args, name, value),
                ):
                    self.assertFalse(self.module.normal_hccl_skip_idle_dp_enabled())
                    with self.module.normal_hccl_idle_dp_scope(object()):
                        self.assertIsNone(self.module._idle_dp_counts.get())
        for flag in ("SGLANG_GLM53_NORMAL_HCCL", "SGLANG_GLM53_2NODE_PD"):
            with mock.patch.dict(os.environ, {flag: "0"}):
                self.assertFalse(self.module.normal_hccl_skip_idle_dp_enabled())
        for mode in Mode:
            if mode != Mode.EXTEND:
                with self.subTest(mode=mode):
                    with self.module.normal_hccl_idle_dp_scope(NS(forward_mode=mode)):
                        self.assertIsNone(self.module._idle_dp_counts.get())

    def decoder_forward(self):
        source = ast.parse((SRT / "models/glm5_next.py").read_text())
        owner = next(
            n
            for n in source.body
            if isinstance(n, ast.ClassDef) and n.name == "Glm5NextDecoderLayer"
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
        self.events = []
        self.seen = []
        case = self

        class MoE:
            experts = NS(moe_runner_config=NS(inplace=True))

            def __call__(self, hidden, fb, allocator):
                case.events.append("mlp")
                case.seen.append(case.module._idle_dp_counts.get())
                if getattr(fb, "fail", False):
                    raise RuntimeError("MLP failure")
                return hidden + 1

        def prepare(hidden, residual, fb):
            self.assertIsNone(self.module._idle_dp_counts.get())
            return hidden, residual

        layer = NS(
            is_nextn=False,
            _normal_skip_idle_dp=self.module.normal_hccl_skip_idle_dp_enabled(),
            mlp=MoE(),
            layer_scatter_modes=object(),
            self_attn=lambda **kw: kw["hidden_states"],
            layer_communicator=NS(
                prepare_attn=prepare,
                prepare_mlp=prepare,
                should_fuse_mlp_allreduce_with_next_layer=lambda fb: False,
                should_use_reduce_scatter=lambda fb: True,
                postprocess_layer=lambda hidden, residual, fb: (hidden, residual),
            ),
        )
        namespace = dict(
            torch=torch,
            nullcontext=nullcontext,
            Glm5NextMLP=type("Dense", (), {}),
            Glm5NextMoE=MoE,
            get_attn_tp_context=lambda: NS(
                clear_attn_inputs=lambda: self.events.append("clear")
            ),
            get_forward=lambda: NS(scoped=lambda **kw: nullcontext()),
        )
        exec(
            compile(ast.fix_missing_locations(module), "<GLM decoder forward>", "exec"),
            namespace,
        )
        return layer, namespace["forward"]

    def test_actual_decoder_publishes_only_around_mlp_and_resets_failure(self):
        layer, forward = self.decoder_forward()
        hidden = torch.arange(6, device="cpu", dtype=torch.float32).reshape(2, 3)
        out, residual, _ = forward(layer, None, hidden, batch((0, 32)), None)
        torch.testing.assert_close(out, hidden + 1)
        self.assertIsNone(residual)
        self.assertEqual(self.seen, [(0, 32)])
        self.assertEqual(self.events, ["clear", "mlp"])
        self.assertIsNone(self.module._idle_dp_counts.get())
        fb = batch()
        fb.fail = True
        with self.assertRaisesRegex(RuntimeError, "MLP failure"):
            forward(layer, None, hidden, fb, None)
        self.assertIsNone(self.module._idle_dp_counts.get())
        # Default-off and D layers retain the original MLP context/call path.
        for changes in ({"flag": "0"}, {"role": "decode"}):
            with self.subTest(changes=changes), mock.patch.dict(os.environ):
                os.environ[FLAG] = changes.get("flag", "1")
                with mock.patch.object(
                    self.args, "disaggregation_mode", changes.get("role", "prefill")
                ):
                    layer, forward = self.decoder_forward()
                    self.assertFalse(layer._normal_skip_idle_dp)
                    with mock.patch.object(
                        self.module,
                        "normal_hccl_idle_dp_scope",
                        side_effect=AssertionError("unexpected wrapper"),
                    ):
                        forward(layer, None, hidden, batch(), None)
                    self.assertEqual(self.seen, [None])

    def test_actual_idle_prefill_conversion_filters_on_idle_process_experts(self):
        source = ast.parse((SRT / "model_executor/forward_batch_info.py").read_text())
        owner = next(
            n
            for n in source.body
            if isinstance(n, ast.ClassDef) and n.name == "ForwardBatch"
        )
        method = next(
            n
            for n in owner.body
            if isinstance(n, ast.FunctionDef) and n.name == "prepare_mlp_sync_batch"
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
        max_len = NS(is_max_len=lambda: True)
        namespace = dict(
            torch=torch,
            ForwardMode=Mode,
            _is_cpu=True,
            get_parallel=lambda: self.parallel,
            get_exec=lambda: NS(graph=NS(cuda_graph_config=NS(prefill=NS(bs=[])))),
            ceil_align=lambda n, unit: (n + unit - 1) // unit * unit,
            DpPaddingMode=NS(get_dp_padding_mode=lambda *args: max_len),
            _elastic_should_preserve_local_token_counts=lambda **kwargs: False,
            mambaish_config=lambda cfg: object(),
            set_dp_buffer_len=mock.Mock(),
            set_is_extend_in_batch=mock.Mock(),
        )
        exec(
            compile(
                ast.fix_missing_locations(module),
                "<actual MLP sync preparation>",
                "exec",
            ),
            namespace,
        )
        tbo = ModuleType("sglang.srt.batch_overlap.two_batch_overlap")
        tbo.TboForwardBatchPreparer = NS(prepare=lambda **kwargs: None)
        peers = self.peers()
        for counts, idle_rank in (((0, 32), 0), ((32, 0), 8)):
            with self.subTest(counts=counts, idle_rank=idle_rank):
                self.parallel.attn_dp_rank, self.parallel.attn_tp_rank = divmod(
                    idle_rank, 8
                )
                fb = batch(counts, mode=Mode.IDLE)
                original = fb.original_global_num_tokens_cpu
                fb.global_num_tokens_cpu = list(counts)  # pre-padding CPU values
                fb.global_num_tokens_for_logprob_cpu = list(counts)
                fb.global_num_tokens_gpu = torch.tensor(counts, device="cpu")
                fb.batch_size = 0
                fb.is_extend_in_batch = True
                fb.can_run_dp_prefill_cuda_graph = False
                fb.spec_info = None
                fb.seq_lens = torch.empty(0, dtype=torch.int64, device="cpu")
                fb.orig_seq_lens = fb.seq_lens.clone()
                fb.seq_lens_cpu = None
                fb.global_num_token_non_padded = None
                fb.tbo_children = None
                fb._pad_inputs_to_size = mock.Mock()
                runner = NS(
                    model_config=NS(),
                    is_draft_worker=False,
                    attn_tp_sequence_sharded=lambda n: True,
                )
                with mock.patch.dict(sys.modules, {tbo.__name__: tbo}):
                    namespace[method.name](fb, runner)
                self.assertEqual(fb._original_forward_mode, Mode.IDLE)
                self.assertEqual(fb.forward_mode, Mode.EXTEND)
                self.assertIs(fb.original_global_num_tokens_cpu, original)
                self.assertEqual(original, list(counts))
                self.assertEqual(fb.global_num_tokens_cpu, [32, 32])
                self.assertEqual(fb.extend_seq_lens_cpu, [32])
                fb._pad_inputs_to_size.assert_called_once_with(runner, 32, 1)
                layer, forward = self.decoder_forward()
                case = self

                class DispatchingMoE(type(layer.mlp)):
                    def __call__(self, hidden, forward_batch, allocator):
                        _, _, local_counts, state = case.dispatch(idle_rank, peers)
                        expected_rows = sorted(
                            (int(peers[1][src][row, k]), src * 4 + row)
                            for src in range(16)
                            for row in range(4)
                            for k in range(8)
                            if counts[src // 8] > 0
                            and idle_rank * 18
                            <= int(peers[1][src][row, k])
                            < (idle_rank + 1) * 18
                        )
                        case.assertEqual(sum(local_counts), len(expected_rows))
                        case.assertGreater(
                            sum(local_counts), 0
                        )  # active peer contributions survive
                        case.assertEqual(
                            state[0].tolist(), [row for _, row in expected_rows]
                        )
                        before = sum(
                            int(
                                (
                                    (routes >= idle_rank * 18)
                                    & (routes < (idle_rank + 1) * 18)
                                ).sum()
                            )
                            for routes in peers[1]
                        )
                        case.assertGreater(
                            before, sum(local_counts)
                        )  # idle routes really removed
                        case.assertTrue(
                            all(counts[int(row) // 32] > 0 for row in state[0])
                        )
                        return super().__call__(hidden, forward_batch, allocator)

                layer.mlp = DispatchingMoE()
                forward(layer, None, torch.ones(4, 3, device="cpu"), fb, None)
                self.assertEqual(self.seen, [counts])
                self.assertIsNone(self.module._idle_dp_counts.get())

    def test_existing_mlp_context_nesting_and_exception_order(self):
        events = []

        @contextmanager
        def existing():
            events.append(("enter", self.module._idle_dp_counts.get()))
            try:
                yield
            finally:
                events.append(("exit", self.module._idle_dp_counts.get()))

        with self.assertRaisesRegex(RuntimeError, "body"):
            with self.module.normal_hccl_idle_dp_scope(batch(), existing()):
                events.append(("body", self.module._idle_dp_counts.get()))
                raise RuntimeError("body")
        self.assertEqual(events, [("enter", None), ("body", (32, 0)), ("exit", None)])

    def peers(self):
        generator = torch.Generator(device="cpu").manual_seed(1015)
        xs = [
            torch.randn(4, 3, generator=generator, device="cpu").to(torch.bfloat16)
            for _ in range(16)
        ]
        ids = [
            (torch.arange(32, device="cpu").reshape(4, 8) * 19 + src * 23) % 288
            for src in range(16)
        ]
        weights = [
            torch.rand(4, 8, generator=generator, device="cpu") for _ in range(16)
        ]
        for routes in ids:
            routes[0, 0] = -1
            routes[0, 1] = -7
            routes[0, 2] = 288
            routes[1, :3] = 0  # dummy expert skew and repeated destinations
        return xs, ids, weights

    def dispatch(self, rank, peers):
        xs, ids, weights = peers
        quantized = [dynamic_quant_reference(x) for x in xs]
        shapes = [
            torch.tensor([4, 3, 2, 4, 8, 2, 4, 8], dtype=torch.int64, device="cpu")
        ] * 16
        payloads = [
            shapes,
            [q for q, _ in quantized],
            [s for _, s in quantized],
            ids,
            weights,
        ]
        self.parallel.moe_ep_group.rank_in_group = rank
        self.parallel.attn_dp_rank, self.parallel.attn_tp_rank = divmod(rank, 8)
        calls = []

        def gather(out, local, group):
            self.assertIs(group, self.group)
            torch.testing.assert_close(
                local, payloads[len(calls)][rank], rtol=0, atol=0
            )
            out.copy_(torch.cat(payloads[len(calls)]))
            calls.append(tuple(out.shape))

        cpu_calls = []
        original_cpu = torch.Tensor.cpu

        def cpu(tensor, *args, **kwargs):
            cpu_calls.append(tuple(tensor.shape))
            return original_cpu(tensor, *args, **kwargs)

        with (
            mock.patch.object(self.module.dist, "get_world_size", return_value=16),
            mock.patch.object(self.module.dist, "get_rank", return_value=rank),
            mock.patch.object(
                self.module.dist, "all_gather_into_tensor", side_effect=gather
            ),
            mock.patch.object(
                torch.ops.npu,
                "npu_dynamic_quant",
                side_effect=dynamic_quant_reference,
                create=True,
            ) as quant,
            mock.patch.object(torch.Tensor, "cpu", cpu),
        ):
            result = self.module.collective_dispatch(
                self.group, xs[rank], ids[rank], weights[rank]
            )
        self.assertEqual(len(calls), 5)  # original shape gather + four payload gathers
        self.assertEqual(len(cpu_calls), 2)  # original shape and ID readbacks only
        self.assertEqual(quant.call_count, 1)
        return result

    def test_dispatch_and_combine_preserve_all_active_contributions(self):
        peers = self.peers()
        xs, ids, weights = peers
        expert_weights = (
            torch.arange(288 * 9, device="cpu").reshape(288, 3, 3) % 7 - 3
        ).int()

        def execute(counts, enabled):
            partials, outputs, states = [], [], []
            with mock.patch.dict(os.environ, {FLAG: "1" if enabled else "0"}):
                with self.module.normal_hccl_idle_dp_scope(batch(counts)):
                    for rank in range(16):
                        recv, scale, local_counts, state = self.dispatch(rank, peers)
                        entries = [
                            (int(ids[src][row, k]), src * 4 + row, weights[src][row, k])
                            for src in range(16)
                            for row in range(4)
                            for k in range(8)
                            if (not enabled or counts[src // 8] > 0)
                            and rank * 18 <= int(ids[src][row, k]) < (rank + 1) * 18
                        ]
                        entries.sort(key=lambda item: item[0])
                        self.assertEqual(sum(local_counts), len(entries))
                        self.assertEqual(state[-1], len(entries))
                        self.assertEqual(state[0].tolist(), [e[1] for e in entries])
                        torch.testing.assert_close(
                            state[1],
                            (
                                torch.stack([e[2] for e in entries])
                                if entries
                                else torch.empty(0)
                            ),
                            rtol=0,
                            atol=0,
                        )
                        y = torch.zeros(
                            (recv.shape[0], 3), dtype=torch.bfloat16, device="cpu"
                        )
                        for index, (expert, global_row, _) in enumerate(entries):
                            src, row = divmod(global_row, 4)
                            q, s = dynamic_quant_reference(xs[src])
                            torch.testing.assert_close(
                                recv[index], q[row], rtol=0, atol=0
                            )
                            torch.testing.assert_close(
                                scale[index], s[row], rtol=0, atol=0
                            )
                            y[index] = (
                                (recv[index].int() @ expert_weights[expert]).float()
                                * scale[index]
                            ).to(torch.bfloat16)
                        partial = torch.zeros(64, 3, device="cpu", dtype=torch.float32)
                        if entries:
                            partial.index_add_(
                                0,
                                state[0],
                                y[: len(entries)].float() * state[1][:, None],
                            )
                        partials.append(partial)
                        outputs.append(y)
                        states.append(state)
            expected = sum(partials)
            for rank in range(16):

                def reduce(out, group):
                    self.assertIs(group, self.group)
                    torch.testing.assert_close(out, partials[rank], rtol=0, atol=0)
                    out.copy_(expected)

                with (
                    mock.patch.object(
                        self.module.dist, "all_reduce", side_effect=reduce
                    ) as reduced,
                    mock.patch.object(self.module.dist, "get_rank", return_value=rank),
                ):
                    actual = self.module.collective_combine(
                        self.group, outputs[rank], states[rank]
                    )
                    torch.testing.assert_close(
                        actual,
                        expected[rank * 4 : (rank + 1) * 4].to(torch.bfloat16),
                        rtol=0,
                        atol=0,
                    )
                    self.assertEqual(actual.shape, (4, 3))
                    self.assertEqual(reduced.call_count, 1)
            return expected

        baseline = execute((32, 32), False)
        for counts in ((0, 32), (32, 0), (32, 32), (0, 0), (1, 32), (32, 1)):
            with self.subTest(counts=counts):
                actual = execute(counts, True)
                expected = baseline.clone()
                for domain, count in enumerate(counts):
                    if count == 0:
                        expected[domain * 32 : (domain + 1) * 32] = 0
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_rank_mapping_errors_and_unpublished_context(self):
        offsets = torch.tensor([0, 8 * 4 * 8], device="cpu")
        selected, discarded, counts = self.module._normal_skip_idle_dp_routes(
            offsets, object(), 2, 1, 4, 8
        )
        self.assertIs(selected, offsets)
        self.assertEqual((discarded, counts), (0, None))
        with self.module.normal_hccl_idle_dp_scope(batch()):
            for owner, name, bad in (
                (self.parallel.moe_ep_group, "ranks", list(reversed(range(16)))),
                (self.parallel.moe_ep_group, "rank_in_group", 8),
                (self.parallel.moe_ep_group, "device_group", object()),
                (self.parallel, "attn_dp_rank", 1),
                (self.parallel, "attn_tp_rank", 1),
                (self.parallel, "attn_dp_size", 4),
                (self.parallel, "attn_tp_size", 4),
            ):
                with self.subTest(name=name), mock.patch.object(owner, name, bad):
                    with self.assertRaisesRegex(RuntimeError, "rank mapping"):
                        self.module._normal_skip_idle_dp_routes(
                            offsets, self.group, 16, 0, 4, 8
                        )

    def test_native_evidence_is_once_after_cpu_counts_and_reports_actual_drops(self):
        peers = self.peers()
        with mock.patch.object(self.module.logger, "info") as logged:
            with self.module.normal_hccl_idle_dp_scope(batch((32, 32))):
                self.dispatch(0, peers)
            logged.assert_not_called()
            with self.module.normal_hccl_idle_dp_scope(batch((0, 32))):
                _, _, counts, _ = self.dispatch(0, peers)
                self.dispatch(0, peers)
            logged.assert_called_once()
            _, rank, world, valid, discarded, countsum = logged.call_args.args
            self.assertEqual((rank, world, valid), (0, 16, (0, 32)))
            expected_discarded = sum(
                int(((routes >= 0) & (routes < 18)).sum()) for routes in peers[1][:8]
            )
            self.assertEqual(discarded, expected_discarded)
            self.assertEqual(countsum, sum(counts))


if __name__ == "__main__":
    unittest.main()
