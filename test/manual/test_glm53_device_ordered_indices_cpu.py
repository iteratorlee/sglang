"""CPU execution of the real map kernel body and scoped dispatch contracts.

No Triton runtime or accelerator initialization. NPU lowering and timing are
checked separately by ascend/test_glm53_device_ordered_indices_npu.py.
"""

import ast
import os
import sys
import unittest
from types import ModuleType
from types import SimpleNamespace as NS
from unittest import mock

import test_glm53_normal_idle_dp_cpu as idle
import test_glm53_normal_ordered_2node_cpu as ordered
import torch
from test_glm53_normal_hccl_cpu import COLLECTIVES_NAME, SRT, load_source

PATH = SRT / "hardware_backend/npu/moe/glm53_device_ordered_indices.py"
FLAG = "SGLANG_GLM53_NORMAL_DEVICE_ROUTING"
KERNEL_NAME = COLLECTIVES_NAME.rsplit(".", 1)[0] + ".glm53_device_ordered_indices"


def extract(name):
    function = next(
        n
        for n in ast.parse(PATH.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    function.decorator_list = []
    return compile(
        ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
        str(PATH),
        "exec",
    )


def emulate_indices(ids, selected, low, experts):
    """Execute the unmodified Triton body with CPU pointer/load/store mocks."""
    pointer = ordered.Pointer
    tile = 0
    written = set()

    def load(p, mask, other):
        offsets, valid = torch.broadcast_tensors(p.offset, mask)
        value = torch.full(offsets.shape, other, dtype=p.tensor.dtype)
        value[valid] = p.tensor[offsets[valid]]
        return value

    def store(p, value, mask):
        offsets, valid = torch.broadcast_tensors(p.offset, mask)
        # Explicitly detect races, including between separate block launches.
        locations = offsets[valid].tolist()
        assert len(set(locations)) == len(locations)
        assert not written.intersection(locations)
        written.update(locations)
        p.tensor[offsets[valid]] = value[valid]

    tl = NS(
        constexpr=int,
        int32=torch.int32,
        program_id=lambda dim: tile,
        arange=lambda start, end: torch.arange(start, end, dtype=torch.int32),
        load=load,
        store=store,
        sum=lambda value, axis: value.sum(dim=axis, dtype=torch.int32),
    )
    namespace = {"tl": tl}
    exec(extract("_ordered_indices"), namespace)
    out = torch.full(ids.shape, -1, dtype=torch.int32)
    for tile in range((selected.numel() + 255) // 256):
        namespace["_ordered_indices"](
            pointer(ids),
            pointer(selected),
            pointer(out),
            selected.numel(),
            low,
            experts,
            8,
            256,
        )
    return out


class DeviceOrderedIndicesCpuTests(unittest.TestCase):
    def setUp(self):
        self.fixture = ordered.OrderedTwoNodeCpuTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.module = self.fixture.module
        self.kernel = ModuleType(KERNEL_NAME)
        self.kernel.device_ordered_indices = mock.Mock(side_effect=emulate_indices)
        mock.patch.dict(sys.modules, {KERNEL_NAME: self.kernel}).start()
        self.helpers = load_source(
            "device_ordered_harness_reference",
            PATH.parents[6]
            / "test/manual/ascend/test_glm53_device_ordered_indices_npu.py",
        )

    def test_exact_inverse_adversarial_duplicates_idle_and_zero_experts(self):
        for rows in (0, 16, 48, 256):
            for case in ("mixed", "duplicate", "skew", "empty"):
                for dtype in (torch.int32, torch.int64):
                    ids = self.helpers.make_ids(rows, case, dtype)
                    for active in (
                        (True, True),
                        (True, False),
                        (False, True),
                        (False, False),
                    ):
                        for rank in range(16):
                            with self.subTest(
                                rows=rows,
                                case=case,
                                dtype=dtype,
                                active=active,
                                rank=rank,
                            ):
                                selected, _ = self.helpers.select_routes(
                                    ids, rank, active
                                )
                                actual = emulate_indices(ids, selected, rank * 18, 18)
                                expected = self.module._normal_combine_ordered_indices(
                                    selected // 8, rows, 8
                                )
                                torch.testing.assert_close(
                                    actual, expected, rtol=0, atol=0
                                )

    def test_current_and_maximum_geometry_sparse_and_dense(self):
        for rows in (32768, 131056):
            for sparse in (True, False):
                if rows == 131056 and not sparse:
                    continue
                ids = self.helpers.make_ids(rows, "mixed", torch.int64)
                if sparse:
                    ids[1:-1] = -1
                for rank in (0, 15):
                    selected, _ = self.helpers.select_routes(ids, rank, (True, False))
                    actual = emulate_indices(ids, selected, rank * 18, 18)
                    expected = self.module._normal_combine_ordered_indices(
                        selected // 8, rows, 8
                    )
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_adversarial_combine_preserves_exact_fp32_association(self):
        ids = self.helpers.make_ids(16, "duplicate", torch.int64)
        selected, _ = self.helpers.select_routes(ids, 0, (True, True))
        inverse = emulate_indices(ids, selected, 0, 18)
        y, gates = self.helpers.adversarial_values(selected // 8, width=4096)
        expected = ordered.reference.sequential_reference(y, gates, selected // 8, 16)
        actual = ordered.emulate_kernel(y, gates, inverse, 16, 4096)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(torch.equal(actual[:, 0], torch.zeros(16)))
        self.assertTrue(torch.equal(actual[:, 1], torch.full((16,), 2**-23)))
        self.assertTrue(torch.equal(actual[:, 2], torch.ones(16)))

    def test_actual_dispatch_exact_payload_gates_counts_and_inverse(self):
        for tokens, counts in ((0, (0, 0)), (1, (8, 8)), (3, (24, 0)), (3, (0, 24))):
            with self.module.normal_hccl_idle_dp_scope(idle.batch(counts)):
                for rank in range(16):
                    with self.subTest(tokens=tokens, counts=counts, rank=rank):
                        with mock.patch.dict(os.environ, {FLAG: "0"}):
                            baseline, _ = self.fixture.dispatch(rank, tokens)
                        # No CPU map may be reached by the new branch.
                        with (
                            mock.patch.dict(os.environ, {FLAG: "ordered"}),
                            mock.patch.object(
                                self.module,
                                "_normal_combine_ordered_indices",
                                side_effect=AssertionError("CPU inverse called"),
                            ),
                        ):
                            candidate, _ = self.fixture.dispatch(rank, tokens)
                        for actual, expected in zip(candidate[:2], baseline[:2]):
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        self.assertEqual(candidate[0].dtype, torch.int8)
                        self.assertEqual(candidate[1].dtype, torch.float32)
                        self.assertEqual(candidate[2].tolist(), baseline[2])
                        for i in (0, 1, 6):
                            torch.testing.assert_close(
                                candidate[3][i], baseline[3][i], rtol=0, atol=0
                            )
                        self.assertEqual(candidate[3][1].dtype, torch.float32)
                        self.assertEqual(candidate[3][2:6], baseline[3][2:6])
        self.assertEqual(self.kernel.device_ordered_indices.call_count, 64)

    def test_wrong_mode_fails_before_payload_and_other_phases_do_not_activate(self):
        with (
            self.module.normal_hccl_idle_dp_scope(idle.batch()),
            mock.patch.dict(os.environ, {FLAG: "ordered", ordered.FLAG: "0"}),
            mock.patch.object(self.module.dist, "get_world_size", return_value=16),
            mock.patch.object(self.module.dist, "get_rank", return_value=0),
            mock.patch.object(
                self.module, "_check_equal_dispatch_shapes_cached"
            ) as audit,
            mock.patch.object(torch.ops.npu, "npu_dynamic_quant", create=True) as quant,
        ):
            with self.assertRaisesRegex(ValueError, "ORDERED=2node"):
                self.module.collective_dispatch(
                    self.fixture.group,
                    torch.empty(1, 4096, dtype=torch.bfloat16),
                    torch.empty(1, 8, dtype=torch.int64),
                    torch.empty(1, 8),
                )
            audit.assert_not_called()
            quant.assert_not_called()
        for mode in idle.Mode:
            if mode == idle.Mode.EXTEND:
                continue
            with (
                self.module.normal_hccl_idle_dp_scope(idle.batch(mode=mode)),
                mock.patch.dict(os.environ, {FLAG: "ordered"}),
            ):
                result, _ = self.fixture.dispatch(0, 1)
                self.assertEqual(len(result[3]), 6)
                self.assertIsInstance(result[2], list)
        self.kernel.device_ordered_indices.assert_not_called()

    def test_routing_has_no_bulk_cpu_readback_and_decode_args_fall_back(self):
        calls = []
        cpu = torch.Tensor.cpu

        def record(tensor, *args, **kwargs):
            calls.append(tuple(tensor.shape))
            return cpu(tensor, *args, **kwargs)

        with (
            self.module.normal_hccl_idle_dp_scope(idle.batch((24, 0))),
            mock.patch.dict(os.environ, {FLAG: "ordered"}),
            mock.patch.object(torch.Tensor, "cpu", record),
        ):
            self.fixture.dispatch(0, 3)
        # The unchanged shape audit is still allowed to read its small vote.
        # In particular, the gathered [48,8] routing IDs are never read back.
        self.assertEqual(calls, [(16, 8)])
        self.kernel.device_ordered_indices.reset_mock()
        for name, value in (
            ("disaggregation_mode", "decode"),
            ("disable_cuda_graph", False),
            ("enable_single_batch_overlap", True),
            ("nnodes", 1),
        ):
            with (
                mock.patch.object(self.fixture.fixture.args, name, value),
                mock.patch.dict(os.environ, {FLAG: "ordered"}),
                self.module.normal_hccl_idle_dp_scope(idle.batch((24, 0))),
            ):
                result, _ = self.fixture.dispatch(0, 3)
                self.assertIsInstance(result[2], list)
                self.assertEqual(len(result[3]), 6)
        self.kernel.device_ordered_indices.assert_not_called()

    def test_wrapper_bounds_empty_no_null_pointer_launch_and_no_value_readback(self):
        device = NS(type="npu")

        def tensor(shape, dtype):
            return NS(
                shape=shape,
                ndim=len(shape),
                dtype=dtype,
                device=device,
                is_contiguous=lambda: True,
                numel=lambda: __import__("math").prod(shape),
            )

        launches, full = mock.MagicMock(), mock.Mock()
        namespace = dict(
            torch=NS(int32=torch.int32, int64=torch.int64, full=full),
            triton=NS(cdiv=lambda n, d: (n + d - 1) // d),
            _ordered_indices=launches,
        )
        exec(extract("device_ordered_indices"), namespace)
        run = namespace["device_ordered_indices"]
        for rows in (0, 1, 32768, 131056, 131070):
            ids = tensor((rows, 8), torch.int64)
            run(ids, tensor((0,), torch.int64), 270, 18)
            launches.__getitem__.assert_not_called()
            full.assert_called_with(ids.shape, -1, dtype=torch.int32, device=device)
        run(tensor((32768, 8), torch.int64), tensor((16413,), torch.int64), 0, 18)
        launches.__getitem__.assert_called_with((65,))
        for ids, selected, low, experts in (
            (tensor((131071, 8), torch.int64), tensor((0,), torch.int64), 0, 18),
            (tensor((1, 4), torch.int64), tensor((0,), torch.int64), 0, 18),
            (tensor((1, 8), torch.int64), tensor((9,), torch.int64), 0, 18),
            (tensor((1, 8), torch.int64), tensor((1,), torch.int32), 0, 18),
            (tensor((1, 8), torch.float32), tensor((1,), torch.int64), 0, 18),
            (tensor((1, 8), torch.int64), tensor((1,), torch.int64), 1, 18),
            (tensor((1, 8), torch.int64), tensor((1,), torch.int64), 288, 18),
            (tensor((1, 8), torch.int64), tensor((1,), torch.int64), 0, 36),
        ):
            with self.assertRaises(ValueError):
                run(ids, selected, low, experts)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
