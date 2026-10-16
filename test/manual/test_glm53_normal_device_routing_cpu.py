"""CPU contracts for scoped GLM device routing, with mocked NPU/HCCL ops.

CUDA_VISIBLE_DEVICES='' TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  /home/iterator/anaconda3/envs/sgl-0511/bin/python -B \
  test/manual/test_glm53_normal_device_routing_cpu.py -v

Checks production dispatch and Ascend normal pre-permute. This establishes
route/order/dtype identity, not Ascend operator support or performance.
"""

import ast
import os
import unittest
from types import SimpleNamespace as NS
from unittest import mock

import torch
import torch._inductor.config  # noqa: F401; preload before sys.modules mocks

import test_glm53_normal_idle_dp_cpu as idle
from test_glm53_normal_hccl_cpu import (
    SRT,
    dynamic_quant_reference,
    grouped_matmul_reference,
)

FLAG = "SGLANG_GLM53_NORMAL_DEVICE_ROUTING"


class DeviceRoutingCpuTests(unittest.TestCase):
    def setUp(self):
        self.fixture = idle.IdleDpCpuTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.module = self.fixture.module
        self.group = self.fixture.group
        mock.patch.dict(os.environ, {FLAG: "1"}).start()

    @staticmethod
    def peers(tokens, *, empty=False, ids_dtype=torch.int64):
        generator = torch.Generator(device="cpu").manual_seed(1016 + tokens)
        xs, ids, weights = [], [], []
        for rank in range(16):
            xs.append(torch.randn(tokens, 3, generator=generator).to(torch.bfloat16))
            routes = torch.randint(0, 288, (tokens, 8), generator=generator)
            if tokens:
                routes[:, :3] = 0  # repeated destinations; stable ties matter
                routes[0, 3:7] = torch.tensor([-7, -1, 288, 999])
            if empty:
                routes.fill_(-1)
            ids.append(routes.to(ids_dtype))
            weights.append(torch.rand(tokens, 8, generator=generator))
        return xs, ids, weights

    def dispatch(
        self,
        rank,
        peers,
        *,
        device=True,
        world=16,
        shapes=None,
        cache=False,
        audit=True,
    ):
        xs, ids, weights = peers
        tokens, width = xs[rank].shape
        quantized = [dynamic_quant_reference(x) for x in xs[:world]]
        shape = [tokens, width, 2, tokens, 8, 2, tokens, 8]
        payloads = [
            shapes or [torch.tensor(shape, dtype=torch.int64)] * world,
            [q for q, _ in quantized],
            [s for _, s in quantized],
            ids[:world],
            weights[:world],
        ]
        parallel = self.fixture.parallel
        parallel.moe_ep_group.rank_in_group = rank
        parallel.attn_dp_rank, parallel.attn_tp_rank = divmod(rank, 8)
        gathers, cpu_calls, list_calls = [], [], []

        def gather(out, local, group):
            self.assertIs(group, self.group)
            index = len(gathers) + int(not audit)
            torch.testing.assert_close(local, payloads[index][rank])
            out.copy_(torch.cat(payloads[index]))
            gathers.append(tuple(out.shape))

        original_cpu, original_list = torch.Tensor.cpu, torch.Tensor.tolist

        def cpu(tensor, *args, **kwargs):
            cpu_calls.append(tuple(tensor.shape))
            return original_cpu(tensor, *args, **kwargs)

        def tolist(tensor, *args, **kwargs):
            list_calls.append(tuple(tensor.shape))
            return original_list(tensor, *args, **kwargs)

        with (
            mock.patch.dict(
                os.environ,
                {
                    FLAG: "1" if device else "0",
                    "SGLANG_GLM53_NORMAL_SHAPE_CACHE": "1" if cache else "0",
                },
            ),
            mock.patch.object(self.module.dist, "get_world_size", return_value=world),
            mock.patch.object(self.module.dist, "get_rank", return_value=rank),
            mock.patch.object(self.module.dist, "all_gather_into_tensor", gather),
            mock.patch.object(self.module.dist, "all_reduce"),
            mock.patch.object(torch.Tensor, "cpu", cpu),
            mock.patch.object(torch.Tensor, "tolist", tolist),
            mock.patch.object(
                torch.ops.npu,
                "npu_dynamic_quant",
                side_effect=dynamic_quant_reference,
                create=True,
            ),
        ):
            result = self.module.collective_dispatch(
                self.group, xs[rank], ids[rank], weights[rank]
            )
        self.assertEqual(len(gathers), 4 + int(audit))
        active = (
            device and world == 16 and self.module._idle_dp_counts.get() is not None
        )
        self.assertEqual(
            cpu_calls,
            ([(world, 8)] if audit else []) + ([] if active else [(world * tokens, 8)]),
        )
        self.assertEqual(
            list_calls,
            ([(world, 8)] if audit else []) + ([] if active else [(288 // world,)]),
        )
        self.assertEqual(isinstance(result[2], torch.Tensor), active)
        return result

    def assert_dispatch_equal(self, actual, reference):
        for left, right in zip(actual[:2], reference[:2]):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        self.assertEqual(actual[2].dtype, torch.int64)
        self.assertEqual(actual[2].tolist(), reference[2])
        for left, right in zip(actual[3][:2], reference[3][:2]):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        self.assertEqual(actual[3][2:], reference[3][2:])

    def test_dynamic_geometry_idle_domains_and_all_ranks_match_cpu(self):
        for tokens in (0, 1, 7, 3, 0):
            peers = self.peers(tokens)
            for counts in (
                (0, 0),
                (0, 8 * tokens),
                (8 * tokens, 0),
                (8 * tokens, 8 * tokens),
                (1, max(1, 8 * tokens)),
            ):
                for rank in range(16):
                    with self.subTest(tokens=tokens, counts=counts, rank=rank):
                        with self.module.normal_hccl_idle_dp_scope(idle.batch(counts)):
                            reference = self.dispatch(rank, peers, device=False)
                            actual = self.dispatch(rank, peers)
                            self.assert_dispatch_equal(actual, reference)

    def test_invalid_ids_empty_experts_and_dummy_rank(self):
        for dtype in (torch.int32, torch.int64):
            peers = self.peers(3, empty=True, ids_dtype=dtype)
            # Exactly one nonempty expert; all other ranks need dummy storage.
            peers[1][0][1, :2] = 18
            for rank in range(16):
                with self.subTest(rank=rank, dtype=dtype):
                    with self.module.normal_hccl_idle_dp_scope(idle.batch((24, 0))):
                        actual = self.dispatch(rank, peers)
                        reference = self.dispatch(rank, peers, device=False)
                    self.assert_dispatch_equal(actual, reference)
                    recv, scales, counts, state = actual
                    if rank != 1:
                        self.assertEqual(recv.shape, (1, 3))
                        self.assertTrue(torch.equal(recv, torch.zeros_like(recv)))
                        self.assertTrue(torch.equal(scales, torch.ones_like(scales)))
                        self.assertEqual(counts.tolist(), [0] * 18)
                        self.assertEqual(state[-1], 0)
                    else:
                        self.assertEqual(state[0].tolist(), [1, 1])
                        self.assertEqual(counts.tolist(), [2] + [0] * 17)

    def test_scope_and_strict_flag_keep_compatibility_paths(self):
        peers = self.peers(2)
        # Outside P EXTEND, even flag=1 retains list counts and the CPU path.
        for mode in idle.Mode:
            if mode != idle.Mode.EXTEND:
                with self.module.normal_hccl_idle_dp_scope(NS(forward_mode=mode)):
                    self.dispatch(0, peers)
        with mock.patch.dict(os.environ, {idle.FLAG: "0"}):
            with self.module.normal_hccl_idle_dp_scope(object()):
                self.dispatch(0, peers)
        for value in (None, "", "0", "true", "2"):
            with mock.patch.dict(os.environ):
                if value is None:
                    os.environ.pop(FLAG, None)
                else:
                    os.environ[FLAG] = value
                # Run directly so dispatch's explicit switch does not override it.
                with self.module.normal_hccl_idle_dp_scope(idle.batch()):
                    with (
                        mock.patch.object(
                            self.module.dist, "get_world_size", return_value=16
                        ),
                        mock.patch.object(self.module.dist, "get_rank", return_value=0),
                        mock.patch.object(self.module, "_check_equal_dispatch_shapes"),
                        mock.patch.object(
                            self.module.dist,
                            "all_gather_into_tensor",
                            side_effect=lambda out, src, group: out.copy_(
                                src.repeat(16, *([1] * (src.ndim - 1)))
                            ),
                        ),
                        mock.patch.object(
                            torch.ops.npu,
                            "npu_dynamic_quant",
                            side_effect=dynamic_quant_reference,
                            create=True,
                        ),
                    ):
                        result = self.module.collective_dispatch(
                            self.group, peers[0][0], peers[1][0], peers[2][0]
                        )
                    self.assertIsInstance(result[2], list)
        self.dispatch(0, peers, world=8)

    def test_remote_geometry_failure_still_precedes_payload(self):
        peers = self.peers(3)
        shapes = [torch.tensor([3, 3, 2, 3, 8, 2, 3, 8])] * 16
        shapes[8] = torch.tensor([2, 3, 2, 2, 8, 2, 2, 8])
        with self.module.normal_hccl_idle_dp_scope(idle.batch((24, 0))):
            with self.assertRaisesRegex(RuntimeError, "EP16 requires equal"):
                self.dispatch(0, peers, shapes=shapes)

    def test_normal_runner_accepts_list_and_preserves_device_counts(self):
        source = ast.parse((SRT / "layers/moe/moe_runner/ascend.py").read_text())
        function = next(
            n
            for n in source.body
            if isinstance(n, ast.FunctionDef)
            and n.name == "pre_permute_deepep_normal_to_ascend"
        )
        function.decorator_list = []
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                function,
            ],
            type_ignores=[],
        )
        namespace = dict(torch=torch, AscendRunnerInput=NS)
        exec(
            compile(ast.fix_missing_locations(module), "actual_normal_runner", "exec"),
            namespace,
        )
        pre_permute = namespace[function.name]
        for counts in (
            [2, 0, 3],
            torch.tensor([2, 0, 3]),
            torch.zeros(18, dtype=torch.int64),
        ):
            with self.subTest(counts=counts):
                hidden = torch.zeros(5, 3, dtype=torch.int8)
                scales, ids, weights = torch.ones(5), object(), object()
                running = {}
                out = pre_permute(
                    (hidden, scales, ids, weights, counts), None, None, running
                )
                self.assertEqual(out.group_list_type, 1)
                self.assertEqual(out.expert_tokens.dtype, torch.int64)
                self.assertEqual(out.expert_tokens.device, hidden.device)
                if isinstance(counts, torch.Tensor):
                    self.assertIs(out.expert_tokens, counts)
                else:
                    self.assertEqual(out.expert_tokens.tolist(), counts)
                self.assertIs(running["topk_ids"], ids)
                self.assertIs(running["topk_weights"], weights)

    def test_device_routing_shape_cache_and_scoped_combine_together(self):
        expert_weights = (torch.arange(288 * 9).reshape(288, 3, 3) % 9 - 4).to(
            torch.int8
        )
        for tokens, counts in ((3, (24, 0)), (7, (0, 56)), (1, (8, 8)), (0, (0, 0))):
            with self.subTest(tokens=tokens, counts=counts):
                peers = self.peers(tokens)
                with self.module.normal_hccl_idle_dp_scope(idle.batch(counts)):
                    baseline = [
                        self.dispatch(rank, peers, device=False) for rank in range(16)
                    ]
                    outputs, states, partials = [], [], []
                    for rank in range(16):
                        candidate = self.dispatch(
                            rank, peers, cache=True, audit=rank == 0
                        )
                        self.assert_dispatch_equal(candidate, baseline[rank])
                        recv, scales, expert_counts, state = candidate
                        accum = grouped_matmul_reference(
                            x=[recv],
                            weight=[expert_weights[rank * 18 : (rank + 1) * 18]],
                            group_list=expert_counts,
                            group_list_type=1,
                            output_dtype=torch.int32,
                        )[0]
                        y = (accum.float() * scales[:, None]).to(torch.bfloat16)
                        partial = torch.zeros(16 * tokens, 3, dtype=torch.float32)
                        for row, weight, value in zip(state[0], state[1], y):
                            partial[row] += weight * value.float()
                        outputs.append(y)
                        states.append(state)
                        partials.append(partial)
                    expected = sum(partials)
                    for rank in range(16):

                        def reduce(out, group, op=None):
                            self.assertIs(group, self.group)
                            if out.dtype == torch.int32:
                                self.assertEqual(out.item(), 0)
                            else:
                                torch.testing.assert_close(
                                    out, partials[rank], rtol=0, atol=0
                                )
                                out.copy_(expected)

                        def scatter(out, source, op, group):
                            self.assertIs(group, self.group)
                            torch.testing.assert_close(
                                source, partials[rank], rtol=0, atol=0
                            )
                            out.copy_(expected[rank * tokens : (rank + 1) * tokens])

                        with (
                            mock.patch.dict(
                                os.environ,
                                {
                                    "SGLANG_GLM53_NORMAL_REDUCE_SCATTER": "2node",
                                    "SGLANG_GLM53_NORMAL_REDUCE_SCATTER_CHECK": "1",
                                },
                            ),
                            mock.patch.object(
                                self.module.dist, "get_rank", return_value=rank
                            ),
                            mock.patch.object(
                                self.module.dist,
                                "reduce_scatter_tensor",
                                side_effect=scatter,
                            ) as scattered,
                            mock.patch.object(
                                self.module.dist, "all_reduce", side_effect=reduce
                            ),
                        ):
                            actual = self.module.collective_combine(
                                self.group, outputs[rank], states[rank]
                            )
                        self.assertEqual(scattered.call_count, int(tokens > 0))
                        torch.testing.assert_close(
                            actual,
                            expected[rank * tokens : (rank + 1) * tokens].to(
                                torch.bfloat16
                            ),
                            rtol=0,
                            atol=0,
                        )


if __name__ == "__main__":
    unittest.main()
