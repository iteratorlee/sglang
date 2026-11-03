"""Dense TP1 normal-HCCL metadata compatibility, without backend imports.

Run with TORCH_DEVICE_BACKEND_AUTOLOAD=0 CUDA_VISIBLE_DEVICES='' on CPUs 30,31.
Execute production AST bodies; only runtime context and collective/device
boundaries are stubbed. This does not test HCCL liveness or model accuracy.
"""

import __future__
import ast
import os
import sys
import unittest
from dataclasses import dataclass
from enum import IntEnum, auto
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
from unittest import mock

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ""

import torch

SRT = Path(__file__).resolve().parents[3] / "python/sglang/srt"
COLLECTIVES = "sglang.srt.hardware_backend.npu.moe.glm53_collectives"


def load_ast(path, names, namespace):
    tree = ast.parse((SRT / path).read_text())
    nodes = []
    for name in names:
        body = tree.body
        for part in name.split("."):
            node = next(n for n in body if getattr(n, "name", None) == part)
            body = node.body
        nodes.append(node)
    exec(
        compile(
            ast.Module(body=nodes, type_ignores=[]),
            str(SRT / path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )


class Backend:
    def __init__(self, name):
        self.name = name

    def __getattr__(self, name):
        if name.startswith("is_"):
            return lambda: self.name == name[3:]
        raise AttributeError(name)


class DenseTPSyncCPU(unittest.TestCase):
    def setUp(self):
        self.args = NS(
            nnodes=4,
            tp_size=32,
            ep_size=32,
            dp_size=4,
            pp_size=1,
            attn_cp_size=1,
            dcp_size=1,
            moe_dp_size=1,
            dwdp_size=1,
            enable_dp_attention=True,
            enable_dp_lm_head=True,
            moe_dense_tp_size=1,
            enable_prefill_cp=False,
            enable_two_batch_overlap=False,
            enable_eplb=False,
            ep_num_redundant_experts=0,
            quantization="modelslim",
            moe_a2a_backend="deepep",
            deepep_mode="normal",
            disaggregation_mode="prefill",
            disable_overlap_schedule=True,
            attn_dp_size=4,
            attn_tp_size=8,
            attn_dp_rank=0,
            moe_ep_size=32,
        )
        self.npu = True
        self.backend = "deepep"
        self.exec_config = NS(
            moe=NS(elastic_ep_backend=None),
            graph=NS(cuda_graph_config=NS(prefill=NS(bs=[]))),
        )
        self.ns = dict(
            __name__=__name__,
            os=os,
            torch=torch,
            dataclass=dataclass,
            IntEnum=IntEnum,
            auto=auto,
            get_parallel=lambda: self.args,
            get_exec=lambda: self.exec_config,
            is_npu=lambda: self.npu,
            get_moe_a2a_backend=lambda: Backend(self.backend),
            get_bool_env_var=lambda name, default: os.getenv(name, default) == "true",
            _ENABLE_METRICS_DP_ATTENTION=False,
            ForwardMode=NS(IDLE=NS(value=0)),
            get_attention_dp_size=lambda: self.args.dp_size,
            ceil_align=lambda n, size: (n + size - 1) // size * size,
            _elastic_should_preserve_local_token_counts=lambda **kw: False,
            set_dp_buffer_len=mock.Mock(),
            set_is_extend_in_batch=mock.Mock(),
            _is_cpu=True,
        )
        self.env = self.enterContext(
            mock.patch.dict(
                os.environ,
                {
                    "SGLANG_GLM53_NORMAL_HCCL": "1",
                    "SGLANG_GLM53_2NODE_PD": "1",
                    "SGLANG_GLM53_4NODE_PD": "1",
                    "SGLANG_MORI_RECV_BOUND": "false",
                },
            )
        )
        load_ast(
            "hardware_backend/npu/moe/glm53_collectives.py",
            ["normal_hccl_profile", "_normal_dp_shape_metadata"],
            self.ns,
        )
        self.profile = mock.Mock(wraps=self.ns["normal_hccl_profile"])
        self.get_args = mock.Mock(side_effect=lambda: self.args)
        modules = {}
        for name, values in {
            "sglang.srt.layers.moe.utils": {
                "get_moe_a2a_backend": self.ns["get_moe_a2a_backend"]
            },
            "sglang.srt.runtime_context": {
                "get_server_args": self.get_args,
                "get_parallel": self.ns["get_parallel"],
                "get_disagg": lambda: self.args,
            },
            COLLECTIVES: {"normal_hccl_profile": self.profile},
            "sglang.srt.batch_overlap.two_batch_overlap": {
                "TboForwardBatchPreparer": NS(prepare=mock.Mock())
            },
        }.items():
            module = ModuleType(name)
            module.__dict__.update(values)
            modules[name] = module
        self.enterContext(mock.patch.dict(sys.modules, modules))
        load_ast(
            "utils/common.py", ["require_mlp_tp_gather", "require_mlp_sync"], self.ns
        )
        load_ast(
            "managers/scheduler_components/dp_attn.py",
            ["MLPSyncBatchInfo", "_update_gather_batch"],
            self.ns,
        )
        load_ast("layers/dp_attention.py", ["DpPaddingMode"], self.ns)
        load_ast(
            "model_executor/forward_batch_info.py",
            ["ForwardBatch.prepare_mlp_sync_batch"],
            self.ns,
        )
        load_ast(
            "layers/communicator.py",
            ["enable_moe_dense_fully_dp", "LayerScatterModes"],
            self.ns,
        )
        load_ast(
            "layers/communicator_mhc.py",
            ["AscendMHCLayerCommunicator.prepare_mlp"],
            self.ns,
        )

    def configure(self, nodes):
        self.args.nnodes = self.args.dp_size = self.args.attn_dp_size = nodes
        self.args.tp_size = self.args.ep_size = self.args.moe_ep_size = nodes * 8

    def decision(self):
        return self.ns["require_mlp_tp_gather"]()

    def test_finite_profiles_require_sync_and_gather(self):
        for nodes in (2, 4):
            with self.subTest(nodes=nodes):
                self.configure(nodes)
                self.assertTrue(self.decision())
                self.assertTrue(self.ns["require_mlp_sync"]())
                self.profile.assert_called_with(self.args, 10)

    def test_guard_cases(self):
        cases = [
            ({}, {"SGLANG_GLM53_NORMAL_HCCL": "0"}),
            ({}, {"SGLANG_GLM53_NORMAL_HCCL": "true"}),
            ({}, {"SGLANG_GLM53_4NODE_PD": "0"}),
            ({"deepep_mode": "low_latency", "disaggregation_mode": "decode"}, {}),
            ({"deepep_mode": "auto"}, {}),
            ({"quantization": "w8a8_int8"}, {}),
            ({"enable_two_batch_overlap": True}, {}),
            ({"enable_eplb": True}, {}),
            ({"ep_num_redundant_experts": 1}, {}),
            ({"disable_overlap_schedule": False}, {}),
            ({"disaggregation_mode": "decode"}, {}),
            ({"pp_size": 2}, {}),
            ({"attn_cp_size": 2}, {}),
            ({"dcp_size": 2}, {}),
            ({"moe_dp_size": 2}, {}),
            ({"dwdp_size": 2}, {}),
            ({"enable_prefill_cp": True}, {}),
            ({"ep_size": 16}, {}),
            ({"moe_a2a_backend": "mori"}, {}),
            ({"nnodes": 1, "tp_size": 8, "ep_size": 8}, {}),
            ({"moe_dense_tp_size": 2}, {}),
        ]
        for changes, env in cases:
            with self.subTest(changes=changes, env=env):
                with mock.patch.dict(vars(self.args), changes), mock.patch.dict(
                    os.environ, env
                ):
                    self.assertFalse(self.decision())
        self.configure(2)
        with mock.patch.dict(os.environ, {"SGLANG_GLM53_2NODE_PD": "0"}):
            self.assertFalse(self.decision())

    def test_gpu_and_other_backend_do_not_load_profile(self):
        self.npu = False
        self.assertFalse(self.decision())
        self.npu = True
        self.backend = "mori"
        self.assertFalse(self.decision())
        self.get_args.assert_not_called()
        self.profile.assert_not_called()

    def test_existing_defaults_and_backends(self):
        for changes, backend, expected in [
            ({"moe_dense_tp_size": None}, "deepep", True),
            ({"moe_dense_tp_size": 32}, "deepep", True),
            ({"moe_dense_tp_size": 8}, "deepep", False),
            ({"enable_dp_lm_head": False}, "deepep", True),
            ({}, "none", True),
            ({}, "flashinfer", True),
            ({"enable_dp_attention": False}, "deepep", False),
        ]:
            with self.subTest(changes=changes, backend=backend):
                self.backend = backend
                with mock.patch.dict(vars(self.args), changes):
                    self.assertEqual(self.decision(), expected)
        self.get_args.assert_not_called()

    def sync_info(self, tokens, logprob):
        return self.ns["MLPSyncBatchInfo"](
            dp_size=self.args.dp_size,
            tp_size=8,
            cp_size=1,
            num_tokens=tokens,
            num_tokens_for_logprob=logprob,
            can_run_decode_cuda_graph=False,
            can_run_prefill_cuda_graph=False,
            is_extend_in_batch=tokens > 0,
            local_can_run_tbo=False,
            local_forward_mode=int(tokens > 0),
        )

    def test_scheduler_keeps_collective_counts_and_eager_padding(self):
        for counts in ([247, 0], [247, 0, 0, 0], [7, 19, 0, 3]):
            self.configure(len(counts))
            peers = [self.sync_info(n, max(n - 1, 0)) for n in counts]
            # Model every TP rank's actual metadata contribution, including idle
            # peers. The production all_gather method derives the DP vectors.
            contributions = torch.stack(
                [peer._get_local_tensor("cpu") for peer in peers for _ in range(8)]
            )
            group = object()
            self.ns["get_tp_group"] = lambda: NS(
                active_ranks_cpu=torch.ones(len(counts) * 8, dtype=torch.int64)
            )
            for rank, info in enumerate(peers):
                with self.subTest(counts=counts, rank=rank):

                    def gather(output, local, group):
                        torch.testing.assert_close(local, contributions[rank * 8])
                        output.copy_(contributions.flatten())

                    with mock.patch.object(
                        torch.distributed, "all_gather_into_tensor", side_effect=gather
                    ) as collective:
                        info.all_gather("cpu", group)
                    collective.assert_called_once()
                    batch = NS()
                    self.ns["_update_gather_batch"](batch, info, self.decision())
                    self.assertEqual(batch.global_num_tokens, counts)
                    self.assertEqual(
                        batch.global_num_tokens_for_logprob,
                        [max(n - 1, 0) for n in counts],
                    )
                    self.assertIs(batch.global_num_tokens, info.global_num_tokens)
                    self.args.attn_dp_rank = rank
                    fb = NS(
                        original_global_num_tokens_cpu=list(batch.global_num_tokens),
                        global_num_tokens_cpu=batch.global_num_tokens,
                        global_num_tokens_for_logprob_cpu=batch.global_num_tokens_for_logprob,
                        global_num_tokens_gpu=torch.zeros(
                            len(counts), dtype=torch.int64
                        ),
                        batch_size=int(counts[rank] > 0),
                        is_extend_in_batch=True,
                        can_run_dp_prefill_cuda_graph=False,
                        tbo_children=None,
                        forward_mode=NS(
                            is_decode=lambda: False,
                            is_target_verify=lambda: False,
                            is_draft_extend_v2=lambda: False,
                            is_idle=lambda: False,
                            is_extend=lambda: True,
                        ),
                        _pad_inputs_to_size=mock.Mock(),
                    )
                    runner = NS(
                        attn_tp_sequence_sharded=lambda n: True, is_draft_worker=False
                    )
                    self.ns["prepare_mlp_sync_batch"](fb, runner)
                    padded = (max(counts) + 7) // 8 * 8
                    self.assertEqual(fb.global_num_tokens_cpu, [padded] * len(counts))
                    self.assertEqual(
                        self.ns["_normal_dp_shape_metadata"](fb),
                        (tuple(counts), (padded,) * len(counts)),
                    )
                    if max(counts) == 247:
                        self.assertEqual(padded // 8, 31)
                    # The old singleton path still fails the unchanged validator.
                    self.ns["_update_gather_batch"](batch, info, False)
                    fb.original_global_num_tokens_cpu = batch.global_num_tokens
                    with self.assertRaisesRegex(
                        ValueError, "nonnegative CPU original counts"
                    ):
                        self.ns["_normal_dp_shape_metadata"](fb)

    def test_ascend_dense_stays_scattered_with_gather_enabled(self):
        self.assertTrue(self.decision())
        scattered = object()
        self.ns["ScatterMode"] = NS(SCATTERED=scattered)
        mode = self.ns["LayerScatterModes"]._compute_mlp_mode(NS(is_layer_sparse=False))
        self.assertIs(mode, scattered)
        reduce = mock.Mock(side_effect=lambda x: x)
        self.ns["attention_tensor_model_parallel_all_reduce"] = reduce
        for rank in range(8):
            hidden = torch.arange(247 * 2).reshape(247, 2)
            residual = torch.zeros_like(hidden)
            communicator = NS(
                layer_scatter_modes=NS(mlp_mode=mode),
                _context=NS(attn_tp_size=8, attn_tp_rank=rank, attn_dp_size=4),
                post_attention_layernorm=None,
                mhc=NS(attn_to_mlp=lambda h, r, **kw: (h, r)),
            )
            out, res = self.ns["prepare_mlp"](communicator, hidden, residual, NS())
            expected = torch.zeros(31, 2, dtype=hidden.dtype)
            chunk = hidden[rank * 31 : (rank + 1) * 31]
            expected[: len(chunk)] = chunk
            torch.testing.assert_close(out, expected)
            self.assertIs(res, residual)
        self.assertEqual(reduce.call_count, 8)


if __name__ == "__main__":
    unittest.main()
