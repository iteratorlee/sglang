"""CPU mathematics, scope and dispatch contracts for EP16 ordered combine.

Executes the actual Triton kernel's Python source with CPU tensor operations
and a tiny pointer/load/store emulator. This does not compile or run Triton;
the companion ascend/test_glm53_ordered_combine_npu.py checks real lowering.
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
    SRT,
    COLLECTIVES_NAME,
    dynamic_quant_reference,
    load_source,
)

FLAG = "SGLANG_GLM53_NORMAL_COMBINE_ORDERED"
DEVICE = "SGLANG_GLM53_NORMAL_DEVICE_ROUTING"
KERNEL_PATH = SRT / "hardware_backend/npu/moe/glm53_ordered_combine.py"
reference = load_source(
    "ordered_reference",
    idle.SRT.parents[2] / "test/manual/ascend/test_glm53_ordered_combine_npu.py",
)


class Pointer:
    def __init__(self, tensor, offset=0):
        self.tensor, self.offset = tensor.flatten(), offset

    def __add__(self, offset):
        return Pointer(self.tensor, self.offset + offset)


def emulate_kernel(y, weights, indices, rows, width):
    """Run the unmodified kernel body; no GPU/NPU/Triton import is needed."""
    source = ast.parse(KERNEL_PATH.read_text())
    function = next(
        n
        for n in source.body
        if isinstance(n, ast.FunctionDef) and n.name == "_ordered_combine"
    )
    function.decorator_list = []
    tile = 0

    def load(pointer, mask, other):
        offsets, valid = torch.broadcast_tensors(pointer.offset, mask)
        value = torch.full(offsets.shape, other, dtype=pointer.tensor.dtype)
        value[valid] = pointer.tensor[offsets[valid]]
        return value

    def store(pointer, value, mask):
        offsets, valid = torch.broadcast_tensors(pointer.offset, mask)
        pointer.tensor[offsets[valid]] = value[valid]

    tl = NS(
        constexpr=int,
        float32=torch.float32,
        program_id=lambda dim: tile,
        arange=torch.arange,
        full=lambda shape, value, dtype: torch.full(shape, value, dtype=dtype),
        static_range=range,
        load=load,
        store=store,
        maximum=lambda x, value: x.clamp_min(value),
    )
    namespace = {"tl": tl}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
            str(KERNEL_PATH),
            "exec",
        ),
        namespace,
    )
    out = torch.full((rows, width), float("nan"), dtype=torch.float32)
    for tile in range((rows + 1) // 2):
        namespace[function.name](
            Pointer(y),
            Pointer(weights),
            Pointer(indices),
            Pointer(out),
            rows,
            width,
            8,
            2,
        )
    return out


class OrderedTwoNodeCpuTests(unittest.TestCase):
    def setUp(self):
        self.fixture = idle.IdleDpCpuTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.module = self.fixture.module
        self.group = self.fixture.group
        mock.patch.dict(os.environ, {FLAG: "2node", DEVICE: "0"}).start()

    def test_scoped_mode_geometry_limits_and_device_mutual_exclusion(self):
        enabled = self.module._normal_ordered_combine_enabled
        self.assertFalse(enabled(16, 1024, 8, 4096, torch.bfloat16))
        for mode in idle.Mode:
            if mode != idle.Mode.EXTEND:
                with self.module.normal_hccl_idle_dp_scope(idle.batch(mode=mode)):
                    self.assertFalse(enabled(16, 1024, 8, 4096, torch.bfloat16))
        with self.module.normal_hccl_idle_dp_scope(idle.batch()):
            for tokens in (0, 1, 1024, 8191):
                self.assertTrue(enabled(16, tokens, 8, 4096, torch.bfloat16))
            for args in (
                (16, 8192, 8, 4096, torch.bfloat16),
                (16, 1, 4, 4096, torch.bfloat16),
                (16, 1, 8, 2048, torch.bfloat16),
                (16, 1, 8, 4096, torch.float32),
            ):
                with self.assertRaisesRegex(ValueError, "topk8/BF16"):
                    enabled(*args)
            self.assertFalse(enabled(8, 1, 8, 4096, torch.bfloat16))
            with mock.patch.dict(os.environ, {DEVICE: "1"}):
                with self.assertRaisesRegex(ValueError, "DEVICE_ROUTING=0"):
                    enabled(16, 1, 8, 4096, torch.bfloat16)
            for flag in ("0", "true", "", "2"):
                with mock.patch.dict(os.environ, {FLAG: flag}):
                    self.assertFalse(enabled(16, 1024, 8, 4096, torch.bfloat16))
        with mock.patch.dict(os.environ, {FLAG: "1"}):
            self.assertTrue(enabled(8, 1, 8, 4096, torch.bfloat16))
            with self.assertRaisesRegex(ValueError, "EP8"):
                enabled(16, 1, 8, 4096, torch.bfloat16)

    def test_actual_kernel_body_matches_sequential_fp32_math(self):
        for rows in (0, 1, 3, 16, 33):
            for case in ("mixed", "empty", "idle_first", "idle_second"):
                with self.subTest(rows=rows, case=case):
                    y, weights, source_rows = reference.make_inputs(rows, 4096, case)
                    inverse = self.module._normal_combine_ordered_indices(
                        source_rows, rows, 8
                    )
                    actual = emulate_kernel(y, weights, inverse, rows, 4096)
                    expected = reference.sequential_reference(
                        y, weights, source_rows, rows
                    )
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertEqual(actual.dtype, torch.float32)
                    for row in source_rows.unique():
                        self.assertEqual(expected[row, 0].item(), 0)
                        self.assertEqual(expected[row, 1].item(), 2**-23)
                        self.assertEqual(expected[row, 2].item(), 1)

    def test_actual_wrapper_keeps_fp32_bounds_and_disables_fusion(self):
        source = ast.parse(KERNEL_PATH.read_text())
        wrapper = next(
            n
            for n in source.body
            if isinstance(n, ast.FunctionDef) and n.name == "ordered_combine"
        )
        device = NS(type="npu")

        def tensor(shape, dtype):
            return NS(
                shape=shape,
                ndim=len(shape),
                dtype=dtype,
                device=device,
                is_contiguous=lambda: True,
                numel=lambda: shape[0],
            )

        launches = mock.MagicMock()
        zeros, empty = mock.Mock(), mock.Mock()
        namespace = dict(
            torch=NS(
                bfloat16=torch.bfloat16,
                float32=torch.float32,
                int32=torch.int32,
                zeros=zeros,
                empty=empty,
            ),
            triton=NS(cdiv=lambda n, d: (n + d - 1) // d),
            _ordered_combine=launches,
        )
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])),
                str(KERNEL_PATH),
                "exec",
            ),
            namespace,
        )
        run = namespace[wrapper.name]
        for rows in (1, 16384, 131056, 131070):
            y, weights, indices = (
                tensor((1, 4096), torch.bfloat16),
                tensor((1,), torch.float32),
                tensor((rows, 8), torch.int32),
            )
            run(y, weights, indices, rows, 4096)
            empty.assert_called_with((rows, 4096), device=device, dtype=torch.float32)
            launches.__getitem__.assert_called_with(((rows + 1) // 2,))
            self.assertIs(
                launches.__getitem__.return_value.call_args.kwargs["enable_fp_fusion"],
                False,
            )
        for rows in (0, 16384):
            launches.reset_mock()
            run(
                tensor((0, 4096), torch.bfloat16),
                tensor((0,), torch.float32),
                tensor((rows, 8), torch.int32),
                rows,
                4096,
            )
            launches.__getitem__.assert_not_called()
            zeros.assert_called_with((rows, 4096), device=device, dtype=torch.float32)
        for rows in (-1, 131071):
            with self.assertRaises(ValueError):
                run(
                    tensor((1, 4096), torch.bfloat16),
                    tensor((1,), torch.float32),
                    tensor((rows, 8), torch.int32),
                    rows,
                    4096,
                )

    def test_mutual_exclusion_fails_before_quantization_or_collectives(self):
        with (
            self.module.normal_hccl_idle_dp_scope(idle.batch()),
            mock.patch.dict(os.environ, {DEVICE: "1"}),
            mock.patch.object(self.module.dist, "get_world_size", return_value=16),
            mock.patch.object(self.module.dist, "get_rank", return_value=0),
            mock.patch.object(
                self.module, "_check_equal_dispatch_shapes_cached"
            ) as audit,
            mock.patch.object(self.module.dist, "all_gather_into_tensor") as gather,
            mock.patch.object(torch.ops.npu, "npu_dynamic_quant", create=True) as quant,
        ):
            with self.assertRaisesRegex(ValueError, "DEVICE_ROUTING=0"):
                self.module.collective_dispatch(
                    self.group,
                    torch.empty(1, 4096, dtype=torch.bfloat16),
                    torch.empty(1, 8, dtype=torch.int64),
                    torch.empty(1, 8),
                )
            audit.assert_not_called()
            gather.assert_not_called()
            quant.assert_not_called()

    def test_inverse_map_ep16_boundary_and_eight_contributions(self):
        for rows in (16384, 131056):
            source_rows = torch.tensor([rows - 1, 0] * 8)
            inverse = self.module._normal_combine_ordered_indices(source_rows, rows, 8)
            self.assertEqual(inverse[-1].tolist(), list(range(0, 16, 2)))
            self.assertEqual(inverse[0].tolist(), list(range(1, 16, 2)))
            self.assertTrue(
                torch.equal(inverse[1:-1], torch.full_like(inverse[1:-1], -1))
            )
        with self.assertRaisesRegex(ValueError, "multiplicity"):
            self.module._normal_combine_ordered_indices(
                torch.zeros(9, dtype=torch.int64), 16, 8
            )

    def dispatch(self, rank, tokens):
        xs = [torch.ones(tokens, 4096, dtype=torch.bfloat16)] * 16
        ids = [
            (torch.arange(tokens * 8).view(tokens, 8) * 7 + src * 18) % 270
            for src in range(16)
        ]  # rank15 is empty, individual experts are empty
        for routes in ids:
            if tokens:
                routes[0, :3] = 0
                routes[0, 3:5] = -1
        weights = [torch.full((tokens, 8), 0.125 + src * 2**-20) for src in range(16)]
        quantized = [dynamic_quant_reference(x) for x in xs]
        payloads = [
            [torch.tensor([tokens, 4096, 2, tokens, 8, 2, tokens, 8])] * 16,
            [q for q, _ in quantized],
            [s for _, s in quantized],
            ids,
            weights,
        ]
        calls = 0

        def gather(out, local, group):
            nonlocal calls
            self.assertIs(group, self.group)
            torch.testing.assert_close(local, payloads[calls][rank])
            out.copy_(torch.cat(payloads[calls]))
            calls += 1

        parallel = self.fixture.parallel
        parallel.moe_ep_group.rank_in_group = rank
        parallel.attn_dp_rank, parallel.attn_tp_rank = divmod(rank, 8)
        with (
            mock.patch.object(self.module.dist, "get_world_size", return_value=16),
            mock.patch.object(self.module.dist, "get_rank", return_value=rank),
            mock.patch.object(self.module.dist, "all_gather_into_tensor", gather),
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
        self.assertEqual(calls, 5)
        return result, ids

    def test_dispatch_idle_empty_ranks_and_combine_use_ordered_fp32(self):
        kernel = ModuleType(
            COLLECTIVES_NAME.rsplit(".", 1)[0] + ".glm53_ordered_combine"
        )
        kernel.ordered_combine = mock.Mock(side_effect=emulate_kernel)
        with mock.patch.dict(sys.modules, {kernel.__name__: kernel}):
            for tokens, counts in ((1, (8, 0)), (3, (0, 24)), (1, (8, 8)), (0, (0, 0))):
                with self.module.normal_hccl_idle_dp_scope(idle.batch(counts)):
                    for rank in (0, 7, 8, 15):
                        with self.subTest(tokens=tokens, counts=counts, rank=rank):
                            (recv, scale, expert_counts, state), ids = self.dispatch(
                                rank, tokens
                            )
                            self.assertIsInstance(expert_counts, list)
                            self.assertEqual(len(state), 7)
                            entries = [
                                (int(ids[src][row, slot]), src * tokens + row)
                                for src in range(16)
                                for row in range(tokens)
                                for slot in range(8)
                                if counts[src // 8]
                                and rank * 18 <= ids[src][row, slot] < (rank + 1) * 18
                            ]
                            entries.sort(key=lambda entry: entry[0])
                            expected_rows = torch.tensor(
                                [row for _, row in entries], dtype=torch.int64
                            )
                            self.assertTrue(torch.equal(state[0], expected_rows))
                            y = (
                                torch.arange(recv.numel()).view(recv.shape) % 17 - 8
                            ).to(torch.bfloat16)
                            if not entries:
                                y.fill_(float("nan"))
                            expected = reference.sequential_reference(
                                y[: len(entries)], state[1], expected_rows, 16 * tokens
                            )
                            for mode in ("0", "2node"):

                                def reduce(out, group):
                                    torch.testing.assert_close(
                                        out, expected, rtol=0, atol=0
                                    )

                                def scatter(out, source, op, group):
                                    torch.testing.assert_close(
                                        source, expected, rtol=0, atol=0
                                    )
                                    out.copy_(
                                        expected[rank * tokens : (rank + 1) * tokens]
                                    )

                                with (
                                    mock.patch.dict(
                                        os.environ,
                                        {"SGLANG_GLM53_NORMAL_REDUCE_SCATTER": mode},
                                    ),
                                    mock.patch.object(
                                        self.module.dist, "get_rank", return_value=rank
                                    ),
                                    mock.patch.object(
                                        self.module.dist,
                                        "all_reduce",
                                        side_effect=reduce,
                                    ),
                                    mock.patch.object(
                                        self.module.dist,
                                        "reduce_scatter_tensor",
                                        side_effect=scatter,
                                    ),
                                    mock.patch.object(
                                        self.module,
                                        "_normal_combine_index_add_",
                                        side_effect=AssertionError("index_add used"),
                                    ),
                                ):
                                    out = self.module.collective_combine(
                                        self.group, y, state
                                    )
                                torch.testing.assert_close(
                                    out,
                                    expected[rank * tokens : (rank + 1) * tokens].to(
                                        torch.bfloat16
                                    ),
                                    rtol=0,
                                    atol=0,
                                )
            self.assertGreater(kernel.ordered_combine.call_count, 0)


if __name__ == "__main__":
    unittest.main()
