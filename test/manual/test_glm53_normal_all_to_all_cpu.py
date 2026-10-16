"""CPU proof of GLM variable-split A2A routing and reverse routing.

HCCL and quantization are CPU mocks. Expert outputs use an independent
per-route mathematical oracle; source combine executes the existing Triton
kernel body through the CPU emulator. No accelerator is initialized.
"""

import ast
import os
import sys
import unittest
from types import ModuleType, SimpleNamespace as NS
from unittest import mock

import torch

import test_glm53_normal_idle_dp_cpu as idle
from test_glm53_normal_hccl_cpu import (
    COLLECTIVES_NAME,
    SRT,
    dynamic_quant_reference,
    load_source,
)
from test_glm53_normal_ordered_2node_cpu import emulate_kernel

NAME = COLLECTIVES_NAME.rsplit(".", 1)[0] + ".glm53_all_to_all"
FLAG = "SGLANG_GLM53_NORMAL_ALL_TO_ALL"


class AllToAllCpuTests(unittest.TestCase):
    def setUp(self):
        self.fixture = idle.IdleDpCpuTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.group = self.fixture.group
        self.collectives = self.fixture.module
        self.module = load_source(
            NAME, SRT / "hardware_backend/npu/moe/glm53_all_to_all.py"
        )
        mock.patch.dict(os.environ, {FLAG: "1"}).start()
        kernel = ModuleType(NAME.rsplit(".", 1)[0] + ".glm53_ordered_combine")
        kernel.ordered_combine = emulate_kernel
        mock.patch.dict(
            sys.modules, {NAME: self.module, kernel.__name__: kernel}
        ).start()

    @staticmethod
    def fixture_data(tokens, idle_counts, pattern="mixed"):
        xs, ids, gates, local_entries = [], [], [], []
        counts = torch.zeros(16, 288, dtype=torch.int64)
        for source, size in enumerate(tokens):
            x = (
                (torch.arange(size * 4096).view(size, 4096) % 19 - 9) / 8 + source / 32
            ).to(torch.bfloat16)
            routes = (torch.arange(size * 8).view(size, 8) * 19 + source * 18) % 270
            if size:
                routes[0, :3] = 0
                routes[0, 3:6] = torch.tensor([-1, -7, 288])
            if pattern == "skew":
                routes.fill_(0)
            if pattern == "invalid":
                routes.fill_(-1)
            weights = (
                0.125
                + source * 2**-20
                + torch.arange(size * 8, dtype=torch.float32).view(size, 8) * 2**-21
            )
            weights[(routes < 0) | (routes >= 288)] = float("nan")
            entries = [
                (int(routes[row, slot]), row, slot)
                for row in range(size)
                for slot in range(8)
                if idle_counts[source // 8] and 0 <= routes[row, slot] < 288
            ]
            entries.sort(key=lambda entry: entry[0])
            for expert, _, _ in entries:
                counts[source, expert] += 1
            xs.append(x)
            ids.append(routes)
            gates.append(weights)
            local_entries.append(entries)
        return xs, ids, gates, local_entries, counts

    def execute(self, tokens, idle_counts, pattern="mixed"):
        xs, ids, gates, local_entries, counts = self.fixture_data(
            tokens, idle_counts, pattern
        )
        quantized = [dynamic_quant_reference(x) for x in xs]
        owner_entries = []
        for owner in range(16):
            entries = [
                (expert, src, row, slot)
                for src in range(16)
                for expert, row, slot in local_entries[src]
                if expert // 18 == owner
            ]
            entries.sort(key=lambda entry: entry[0])
            owner_entries.append(entries)

        def q_rows(entries):
            return (
                torch.stack([quantized[src][0][row] for _, src, row, _ in entries])
                if entries
                else torch.empty(0, 4096, dtype=torch.int8)
            )

        def scales(entries):
            return (
                torch.stack([quantized[src][1][row] for _, src, row, _ in entries])
                if entries
                else torch.empty(0)
            )

        def expert_result(entries):
            if not entries:
                return torch.empty(0, 4096, dtype=torch.bfloat16)
            return torch.stack(
                [
                    (
                        quantized[src][0][row].float() * quantized[src][1][row]
                        + expert / 16
                    ).to(torch.bfloat16)
                    for expert, src, row, _ in entries
                ]
            )

        global_routes = int(counts.sum())
        states, outputs = [], []
        with self.collectives.normal_hccl_idle_dp_scope(idle.batch(idle_counts)):
            for rank in range(16):
                parallel = self.fixture.parallel
                parallel.moe_ep_group.rank_in_group = rank
                parallel.attn_dp_rank, parallel.attn_tp_rank = divmod(rank, 8)
                sent_entries = [
                    (expert, rank, row, slot)
                    for expert, row, slot in local_entries[rank]
                ]
                raw_entries = sorted(owner_entries[rank], key=lambda entry: entry[1])
                send_splits = [
                    sum(expert // 18 == dest for expert, _, _ in local_entries[rank])
                    for dest in range(16)
                ]
                recv_splits = [
                    sum(src == source for _, src, _, _ in owner_entries[rank])
                    for source in range(16)
                ]
                calls = []

                def gather(out, local, group):
                    self.assertIs(group, self.group)
                    self.assertEqual(
                        local.shape, (288,)
                    )  # counts only; no shape/payload gather
                    self.assertTrue(torch.equal(local, counts[rank]))
                    out.copy_(counts.flatten())

                def a2a(out, value, output_split_sizes, input_split_sizes, group):
                    self.assertIs(group, self.group)
                    self.assertEqual(output_split_sizes, recv_splits)
                    self.assertEqual(input_split_sizes, send_splits)
                    self.assertTrue(value.is_contiguous() and out.is_contiguous())
                    expected_send = (
                        q_rows(sent_entries)
                        if value.dtype == torch.int8
                        else scales(sent_entries)
                    )
                    expected_recv = (
                        q_rows(raw_entries)
                        if value.dtype == torch.int8
                        else scales(raw_entries)
                    )
                    torch.testing.assert_close(value, expected_send, rtol=0, atol=0)
                    self.assertEqual(out.shape, expected_recv.shape)
                    out.copy_(expected_recv)
                    calls.append(value.dtype)

                with (
                    mock.patch.object(
                        self.module.dist, "get_world_size", return_value=16
                    ),
                    mock.patch.object(self.module.dist, "get_rank", return_value=rank),
                    mock.patch.object(
                        self.module.dist, "all_gather_into_tensor", side_effect=gather
                    ) as gathered,
                    mock.patch.object(
                        self.module.dist, "all_to_all_single", side_effect=a2a
                    ),
                    mock.patch.object(
                        self.module.dist,
                        "all_reduce",
                        side_effect=AssertionError("global reduce used"),
                    ),
                    mock.patch.object(
                        torch.ops.npu,
                        "npu_dynamic_quant",
                        side_effect=dynamic_quant_reference,
                        create=True,
                    ) as quant,
                ):
                    recv, scale, expert_counts, state = self.module.all_to_all_dispatch(
                        self.group, xs[rank], ids[rank], gates[rank]
                    )
                self.assertEqual(gathered.call_count, 1)
                self.assertEqual(
                    calls, [torch.int8, torch.float32] if global_routes else []
                )
                self.assertEqual(quant.call_count, int(bool(sent_entries)))
                self.assertEqual(
                    expert_counts,
                    counts[:, rank * 18 : (rank + 1) * 18].sum(0).tolist(),
                )
                self.assertEqual(state.received, len(owner_entries[rank]))
                self.assertEqual(state.tokens, tokens[rank])
                self.assertEqual(state.global_routes, global_routes)
                self.assertEqual(state.indices.shape, (tokens[rank], 8))
                if owner_entries[rank]:
                    torch.testing.assert_close(
                        recv, q_rows(owner_entries[rank]), rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        scale, scales(owner_entries[rank]), rtol=0, atol=0
                    )
                    expert_ids = torch.tensor(
                        [expert for expert, _, _, _ in owner_entries[rank]]
                    )
                    y = (recv.float() * scale[:, None] + expert_ids[:, None] / 16).to(
                        torch.bfloat16
                    )
                else:
                    self.assertTrue(
                        torch.equal(recv, torch.zeros(1, 4096, dtype=torch.int8))
                    )
                    self.assertTrue(torch.equal(scale, torch.ones(1)))
                    y = torch.full((1, 4096), float("nan"), dtype=torch.bfloat16)
                states.append(state)
                outputs.append(y)

            for rank in range(16):
                state = states[rank]
                raw_entries = sorted(owner_entries[rank], key=lambda entry: entry[1])
                returned_entries = [
                    (expert, rank, row, slot)
                    for expert, row, slot in local_entries[rank]
                ]
                expected = torch.zeros(tokens[rank], 4096)
                returned_y = expert_result(returned_entries)
                for index, (_, _, row, slot) in enumerate(returned_entries):
                    expected[row] += returned_y[index].float() * gates[rank][row, slot]

                def reverse(out, value, output_split_sizes, input_split_sizes, group):
                    self.assertIs(group, self.group)
                    self.assertEqual(value.dtype, torch.bfloat16)
                    self.assertEqual(input_split_sizes, state.recv_splits)
                    self.assertEqual(output_split_sizes, state.send_splits)
                    torch.testing.assert_close(
                        value, expert_result(raw_entries), rtol=0, atol=0
                    )
                    self.assertEqual(out.shape, returned_y.shape)
                    out.copy_(returned_y)

                with (
                    mock.patch.object(
                        self.module.dist, "all_to_all_single", side_effect=reverse
                    ) as exchanged,
                    mock.patch.object(
                        self.module.dist,
                        "all_reduce",
                        side_effect=AssertionError("global reduce used"),
                    ),
                    mock.patch.object(
                        self.module.dist,
                        "reduce_scatter_tensor",
                        side_effect=AssertionError("global scatter used"),
                    ),
                    mock.patch.object(
                        self.module.dist,
                        "all_gather_into_tensor",
                        side_effect=AssertionError("payload gather used"),
                    ),
                ):
                    variants = [outputs[rank]]
                    if state.received == 0:
                        variants.append(torch.empty(0, 4096, dtype=torch.bfloat16))
                    for y in variants:
                        result = self.module.all_to_all_combine(self.group, y, state)
                        torch.testing.assert_close(
                            result, expected.to(torch.bfloat16), rtol=0, atol=0
                        )
                self.assertEqual(
                    exchanged.call_count, len(variants) * int(global_routes > 0)
                )

    def test_combine_only_allows_zero_or_dummy_for_empty_expert_output(self):
        state = self.module.AllToAllState(
            [0] * 16,
            [0] * 16,
            torch.empty(0, dtype=torch.int64),
            torch.empty(0),
            torch.full((1, 8), -1, dtype=torch.int32),
            1,
            4096,
            0,
            0,
        )
        with mock.patch.object(self.module.dist, "all_to_all_single") as exchanged:
            for rows in (0, 1):
                y = torch.full((rows, 4096), float("nan"), dtype=torch.bfloat16)
                output = self.module.all_to_all_combine(self.group, y, state)
                self.assertTrue(torch.equal(output, torch.zeros_like(output)))
            for received, shape, dtype in (
                (0, (2, 4096), torch.bfloat16),
                (0, (0, 2048), torch.bfloat16),
                (0, (0, 4096), torch.float32),
                (0, (4096,), torch.bfloat16),
                (1, (0, 4096), torch.bfloat16),
                (1, (2, 4096), torch.bfloat16),
                (2, (1, 4096), torch.bfloat16),
            ):
                with self.subTest(received=received, shape=shape, dtype=dtype):
                    with self.assertRaisesRegex(ValueError, "BF16 expert rows"):
                        self.module.all_to_all_combine(
                            self.group,
                            torch.empty(shape, dtype=dtype),
                            state._replace(received=received),
                        )
            exchanged.assert_not_called()

    def test_roundtrip_arbitrary_rank_sizes_idle_domains_and_zero_traffic(self):
        for tokens, counts, pattern in (
            ([1, 3, 0, 2] * 4, (1, 24), "mixed"),
            ([2] * 16, (16, 0), "mixed"),
            ([2] * 16, (0, 16), "skew"),  # idle rank0 still computes every expert route
            ([1] * 16, (0, 0), "mixed"),
            ([1] * 16, (8, 8), "invalid"),
            ([0] * 16, (0, 0), "mixed"),
        ):
            with self.subTest(tokens=tokens, counts=counts, pattern=pattern):
                self.execute(tokens, counts, pattern)

    def test_receive_permutation_matches_expert_source_order_for_arbitrary_counts(self):
        generator = torch.Generator().manual_seed(1016)
        for world, experts in ((2, 4), (4, 12), (16, 288)):
            counts = torch.randint(0, 5, (world, experts), generator=generator)
            counts[0] = 0
            counts[:, -1] = 0
            for rank in range(world):
                raw = [
                    (expert, source, occurrence)
                    for source in range(world)
                    for expert in range(
                        rank * (experts // world), (rank + 1) * (experts // world)
                    )
                    for occurrence in range(int(counts[source, expert]))
                ]
                splits, local_counts, permutation, inverse = self.module._receive_plan(
                    counts, rank
                )
                reordered = [raw[index] for index in permutation.tolist()]
                self.assertEqual(reordered, sorted(raw))
                self.assertEqual([reordered[index] for index in inverse.tolist()], raw)
                self.assertEqual(sum(splits), len(raw))
                self.assertEqual(sum(local_counts), len(raw))

    def test_scope_strict_flag_geometry_and_input_dtype_contracts(self):
        with mock.patch.object(self.module.dist, "get_world_size", return_value=16):
            self.assertFalse(self.module.normal_all_to_all_enabled(self.group))
            with self.collectives.normal_hccl_idle_dp_scope(idle.batch()):
                self.assertTrue(self.module.normal_all_to_all_enabled(self.group))
                for value in ("0", "", "true", "2"):
                    with mock.patch.dict(os.environ, {FLAG: value}):
                        self.assertFalse(
                            self.module.normal_all_to_all_enabled(self.group)
                        )
                with mock.patch.object(
                    self.module.dist, "get_world_size", return_value=8
                ):
                    self.assertFalse(self.module.normal_all_to_all_enabled(self.group))
            for mode in idle.Mode:
                if mode != idle.Mode.EXTEND:
                    with self.collectives.normal_hccl_idle_dp_scope(
                        idle.batch(mode=mode)
                    ):
                        self.assertFalse(
                            self.module.normal_all_to_all_enabled(self.group)
                        )
        with (
            self.collectives.normal_hccl_idle_dp_scope(idle.batch()),
            mock.patch.object(self.module.dist, "get_world_size", return_value=16),
            mock.patch.object(self.module.dist, "get_rank", return_value=0),
            mock.patch.object(self.module.dist, "all_gather_into_tensor") as gathered,
        ):
            for size, width, xdtype, gdtype, experts in (
                (1025, 4096, torch.bfloat16, torch.float32, 288),
                (1, 2048, torch.bfloat16, torch.float32, 288),
                (1, 4096, torch.float32, torch.float32, 288),
                (1, 4096, torch.bfloat16, torch.bfloat16, 288),
                (1, 4096, torch.bfloat16, torch.float32, 256),
            ):
                with self.assertRaisesRegex(ValueError, "local tokens <= 1024"):
                    self.module.all_to_all_dispatch(
                        self.group,
                        torch.empty(size, width, dtype=xdtype),
                        torch.zeros(size, 8, dtype=torch.int64),
                        torch.ones(size, 8, dtype=gdtype),
                        experts,
                    )
            gathered.assert_not_called()

    def test_numerical_association_change_is_explicit(self):
        # Owner0 has expert0; owner1 has experts18 and19. All BF16 values
        # and FP32 unit gates are exact, yet owner grouping changes the sum.
        values = torch.tensor([2**24, 1, -(2**24)], dtype=torch.bfloat16).float()
        legacy = values[0] + (values[1] + values[2])
        source_ordered = (values[0] + values[1]) + values[2]
        self.assertEqual(legacy.item(), 1)
        self.assertEqual(source_ordered.item(), 0)

    def test_maximum_local_routes_skew_and_rejected_peer_count_metadata(self):
        ids = torch.zeros(1024, 8, dtype=torch.int64)
        selected, counts = self.module._local_routes(ids, 288, True)
        self.assertTrue(torch.equal(selected, torch.arange(8192)))
        self.assertEqual(counts.tolist(), [8192] + [0] * 287)
        matrix = counts.repeat(16, 1)
        splits, expert_counts, permutation, inverse = self.module._receive_plan(
            matrix, 0
        )
        self.assertEqual(splits, [8192] * 16)
        self.assertEqual(expert_counts, [131072] + [0] * 17)
        self.assertTrue(torch.equal(permutation[inverse], torch.arange(131072)))
        for source_counts in (-1, 8193):

            def gather(out, local, group):
                out.zero_()
                out[288] = source_counts

            with (
                self.collectives.normal_hccl_idle_dp_scope(idle.batch()),
                mock.patch.object(self.module.dist, "get_world_size", return_value=16),
                mock.patch.object(self.module.dist, "get_rank", return_value=0),
                mock.patch.object(
                    self.module.dist, "all_gather_into_tensor", side_effect=gather
                ),
                mock.patch.object(self.module.dist, "all_to_all_single") as exchanged,
            ):
                with self.assertRaisesRegex(ValueError, "peer route counts"):
                    self.module.all_to_all_dispatch(
                        self.group,
                        torch.ones(1, 4096, dtype=torch.bfloat16),
                        torch.zeros(1, 8, dtype=torch.int64),
                        torch.ones(1, 8),
                    )
                exchanged.assert_not_called()

    def test_normal_dispatcher_opt_in_and_state_selected_combine(self):
        tree = ast.parse((SRT / "layers/moe/token_dispatcher/deepep.py").read_text())
        owner = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "_DeepEPDispatcherImplNormal"
        )
        methods = [
            n
            for n in owner.body
            if isinstance(n, ast.FunctionDef)
            and n.name in ("_dispatch_core", "_combine_core")
        ]
        namespace = {"os": os}
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                )
            ]
            + methods,
            type_ignores=[],
        )
        exec(
            compile(
                ast.fix_missing_locations(module), "actual_normal_dispatcher", "exec"
            ),
            namespace,
        )
        events = ModuleType("deep_ep.utils")
        events.EventOverlap = lambda: None
        dispatcher = NS(
            quant_config={"glm53_normal_hccl": True}, group=self.group, num_experts=288
        )
        x, ids, weights, state = object(), object(), object(), object()
        with (
            mock.patch.dict(sys.modules, {events.__name__: events}),
            mock.patch.object(
                self.module, "normal_all_to_all_enabled", return_value=True
            ),
            mock.patch.object(
                self.module, "all_to_all_dispatch", return_value=(x, x, [1], state)
            ) as a2a_dispatch,
            mock.patch.object(
                self.module, "all_to_all_combine", return_value=x
            ) as a2a_combine,
            mock.patch.object(
                self.collectives, "collective_dispatch", return_value=(x, x, [2], ())
            ) as legacy_dispatch,
            mock.patch.object(
                self.collectives, "collective_combine", return_value=x
            ) as legacy_combine,
        ):
            namespace["_dispatch_core"](dispatcher, x, ids, weights, None)
            self.assertIs(dispatcher._glm53_all_to_all_state, state)
            legacy_dispatch.assert_not_called()
            with mock.patch.dict(os.environ, {FLAG: "0"}):
                namespace["_combine_core"](dispatcher, x, None)
                self.assertIsNone(dispatcher._glm53_all_to_all_state)
                namespace["_dispatch_core"](dispatcher, x, ids, weights, None)
                namespace["_combine_core"](dispatcher, x, None)
            a2a_dispatch.assert_called_once()
            a2a_combine.assert_called_once_with(self.group, x, state)
            legacy_dispatch.assert_called_once()
            legacy_combine.assert_called_once()


if __name__ == "__main__":
    unittest.main()
