"""CPU/source checks of the actual GLM combine function; no torch/NPU imports.

NumPy models tensors and a fixed-rank-order collective oracle. This checks
layout, dtype, flag and call contracts, not a real HCCL reduction tree.
"""

import ast
import os
from pathlib import Path
import subprocess
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PATH = "python/sglang/srt/hardware_backend/npu/moe/glm53_collectives.py"
BASE = "3ef9d35f79b62748345234605d7cd16bf9c38811"
FLAG = "SGLANG_GLM53_NORMAL_REDUCE_SCATTER"
BF16 = "bfloat16"
SUM = object()


def bf16_round(value):
    value = np.asarray(value, np.float32)
    bits = value.view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    return np.asarray(rounded, dtype=np.uint32).view(np.float32)


class Tensor:
    def __init__(self, value, dtype=None, trace=None, device="cpu-simulated-npu"):
        self.array = np.asarray(value)
        self.dtype = self.array.dtype if dtype is None else dtype
        self.device = device
        self.trace = trace

    def __getitem__(self, index):
        return Tensor(self.array[index], self.dtype, self.trace, self.device)

    def float(self):
        return Tensor(self.array.astype(np.float32), np.dtype(np.float32), self.trace)

    def __mul__(self, other):
        return Tensor(
            np.multiply(self.array, other.array, dtype=np.float32),
            np.dtype(np.float32),
            self.trace,
        )

    def to(self, dtype):
        if self.trace is not None:
            self.trace.append(("cast", dtype))
        array = bf16_round(self.array) if dtype == BF16 else self.array.astype(dtype)
        return Tensor(array, dtype, self.trace, self.device)

    def index_add_(self, dim, rows, values):
        assert dim == 0 and values.dtype == np.dtype(np.float32)
        if self.trace is not None:
            self.trace.append(("index_add", rows.array.copy(), values.array.copy()))
        np.add.at(self.array, rows.array, values.array)
        return self

    def narrow(self, dim, start, length):
        assert dim == 0
        if self.trace is not None:
            self.trace.append(("narrow", start, length))
        return Tensor(
            self.array[start : start + length], self.dtype, self.trace, self.device
        )


class Collective:
    def __init__(self, reduced, local_input, group, trace):
        self.reduced = reduced
        self.local_input = local_input
        self.group = group
        self.trace = trace
        self.ReduceOp = types.SimpleNamespace(SUM=SUM)

    def check(self, source, group):
        assert group is self.group
        assert source.dtype == np.dtype(np.float32)
        assert source.array.flags.c_contiguous
        np.testing.assert_array_equal(
            source.array.view(np.uint32), self.local_input.view(np.uint32)
        )

    def all_reduce(self, value, group):
        self.check(value, group)
        self.trace.append(("all_reduce",))
        value.array[...] = self.reduced

    def get_rank(self, group):
        assert group is self.group  # No world/global-rank lookup is allowed.
        self.trace.append(("group_rank",))
        return group.rank

    def reduce_scatter_tensor(self, output, source, op, group, async_op=False):
        self.check(source, group)
        assert op is SUM and not async_op
        assert output.dtype == source.dtype and output.device == source.device
        assert output.array.flags.c_contiguous
        assert not np.shares_memory(output.array, source.array)
        assert source.array.shape == (group.world * group.tokens, group.width)
        assert output.array.shape == (group.tokens, group.width)
        self.trace.append(("reduce_scatter_tensor", "SUM"))
        start = group.rank * group.tokens
        output.array[...] = self.reduced[start : start + group.tokens]


def nodes(source):
    return {n.name: n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}


SOURCE = (ROOT / PATH).read_text()
FUNCTIONS = nodes(SOURCE)


class TestNormalReduceScatter(unittest.TestCase):
    def fixture(self, world, tokens, width, empty_rank=None):
        shape = (world * tokens, width)
        reduced = np.zeros(shape, np.float32)
        entries = []
        for sender in range(world):
            rows = []
            if tokens and sender != empty_rank:
                for destination in range(world):
                    for token in sorted({0, tokens // 2, tokens - 1}):
                        rows.extend([destination * tokens + token] * 2)
            rows = np.asarray(rows, np.int64)
            count = len(rows)
            values = bf16_round(
                (np.arange(max(count, 1) * width).reshape(-1, width) % 31 - 15) / 16
                + sender / 32
            )
            if count == 0:
                values.fill(np.nan)  # Dummy rows must not be read.
            # Low mantissa bits detect accidentally casting weights to BF16.
            weights = np.full(count, np.float32(0.125 + (sender + 1) * 2**-20))
            weighted = np.multiply(values[:count], weights[:, None], dtype=np.float32)
            local = np.zeros(shape, np.float32)
            np.add.at(local, rows, weighted)
            reduced = np.add(reduced, local, dtype=np.float32)
            entries.append((rows, values, weights, local))
        return reduced, entries

    def execute(self, fixture, rank, world, tokens, width, flag, source_node=None):
        reduced, entries = fixture
        rows, values, weights, local = entries[rank]
        trace, allocations = [], []
        # Physical/global ranks intentionally differ from group rank.
        group = types.SimpleNamespace(
            rank=rank,
            global_rank=31 - 2 * rank,
            world=world,
            tokens=tokens,
            width=width,
        )

        def allocate(shape, device, dtype, empty=False):
            allocations.append(("empty" if empty else "zeros", tuple(shape), dtype))
            array = np.full(shape, np.nan, dtype) if empty else np.zeros(shape, dtype)
            return Tensor(array, np.dtype(dtype), trace, device)

        torch = types.SimpleNamespace(
            float32=np.dtype(np.float32),
            zeros=lambda shape, device, dtype: allocate(shape, device, dtype),
            empty=lambda shape, device, dtype: allocate(shape, device, dtype, True),
        )
        namespace = {
            "os": os,
            "torch": torch,
            "dist": Collective(reduced, local, group, trace),
        }
        module = ast.Module(
            body=[source_node or FUNCTIONS["collective_combine"]], type_ignores=[]
        )
        exec(
            compile(ast.fix_missing_locations(module), "actual_glm_combine", "exec"),
            namespace,
        )
        state = (Tensor(rows), Tensor(weights), tokens, world, width, len(rows))
        with patch.dict(os.environ, {}, clear=True):
            if flag is not None:
                os.environ[FLAG] = flag
            result = namespace["collective_combine"](group, Tensor(values, BF16), state)
        expected = bf16_round(reduced[rank * tokens : (rank + 1) * tokens])
        np.testing.assert_array_equal(
            result.array.view(np.uint32), expected.view(np.uint32)
        )
        self.assertEqual(result.dtype, BF16)
        self.assertEqual(result.array.shape, (tokens, width))
        return result, trace, allocations

    def test_contiguous_rank_blocks_world16_and_non_aligned_lengths(self):
        # Token counts include the later NPU targets; narrow CPU width keeps
        # this a cheap source contract test, not a real-shape NPU benchmark.
        for world in (2, 4, 16):
            for tokens in (1, 3, 63, 255, 1024, 4096):
                fixture = self.fixture(world, tokens, 7)
                for rank in range(world):
                    _, trace, allocations = self.execute(
                        fixture, rank, world, tokens, 7, "1"
                    )
                    names = [event[0] for event in trace]
                    self.assertEqual(
                        names, ["index_add", "reduce_scatter_tensor", "cast"]
                    )
                    self.assertEqual(
                        allocations,
                        [
                            ("zeros", (world * tokens, 7), np.dtype(np.float32)),
                            ("empty", (tokens, 7), np.dtype(np.float32)),
                        ],
                    )

    def test_actual_hidden_width4096_with_small_token_count(self):
        fixture = self.fixture(16, 3, 4096)
        for rank in (0, 5, 15):
            old, _, _ = self.execute(fixture, rank, 16, 3, 4096, "0")
            new, _, _ = self.execute(fixture, rank, 16, 3, 4096, "1")
            np.testing.assert_array_equal(
                old.array.view(np.uint32), new.array.view(np.uint32)
            )

    def test_exact_opt_in_and_default_off(self):
        fixture = self.fixture(4, 3, 11)
        for flag in (None, "0", "true", "TRUE", "2", "", " 1"):
            _, trace, allocations = self.execute(fixture, 2, 4, 3, 11, flag)
            self.assertEqual(
                [x[0] for x in trace],
                ["index_add", "all_reduce", "group_rank", "narrow", "cast"],
            )
            self.assertEqual(len(allocations), 1)

    def test_empty_local_experts_still_join_sum_and_receive_remote_rows(self):
        fixture = self.fixture(4, 3, 11, empty_rank=0)
        output, trace, _ = self.execute(fixture, 0, 4, 3, 11, "1")
        self.assertEqual([x[0] for x in trace], ["reduce_scatter_tensor", "cast"])
        self.assertTrue(np.isfinite(output.array).all())
        self.assertTrue((output.array != 0).any())

    def test_single_rank_and_zero_tokens_preserve_legacy_collective(self):
        for world, tokens in ((1, 3), (1, 0), (16, 0)):
            fixture = self.fixture(world, tokens, 7)
            _, trace, allocations = self.execute(fixture, 0, world, tokens, 7, "1")
            self.assertIn("all_reduce", [x[0] for x in trace])
            self.assertNotIn("reduce_scatter_tensor", [x[0] for x in trace])
            self.assertEqual(len(allocations), 1)

    def test_local_index_add_order_fp32_products_and_sum_input_unchanged(self):
        fixture = self.fixture(4, 63, 17)
        for rank in range(4):
            old, old_trace, _ = self.execute(fixture, rank, 4, 63, 17, "0")
            new, new_trace, _ = self.execute(fixture, rank, 4, 63, 17, "1")
            np.testing.assert_array_equal(old_trace[0][1], new_trace[0][1])
            np.testing.assert_array_equal(
                old_trace[0][2].view(np.uint32), new_trace[0][2].view(np.uint32)
            )
            np.testing.assert_array_equal(
                old.array.view(np.uint32), new.array.view(np.uint32)
            )

    def test_dispatch_and_off_path_ast_match_frozen_parent(self):
        baseline = subprocess.check_output(
            ["git", "-C", str(ROOT), "show", BASE + ":" + PATH], text=True
        )
        old = nodes(baseline)
        self.assertEqual(
            ast.dump(old["collective_dispatch"]),
            ast.dump(FUNCTIONS["collective_dispatch"]),
        )
        candidate = nodes(SOURCE)["collective_combine"]
        candidate.body = [
            n
            for n in candidate.body
            if not (isinstance(n, ast.If) and FLAG in ast.unparse(n.test))
        ]
        self.assertEqual(ast.dump(old["collective_combine"]), ast.dump(candidate))
        self.assertNotIn("ordered_combine", SOURCE)

    def test_fp32_reduction_reassociation_can_change_final_bf16(self):
        # A mathematical counterexample, not a simulated HCCL implementation.
        values = np.array([2**24, 1, -(2**24), 0], np.float32)

        def reduce(order):
            result = np.float32(0)
            for i in order:
                result = np.add(result, values[i], dtype=np.float32)
            return result

        first, second = reduce([0, 1, 2, 3]), reduce([0, 2, 1, 3])
        self.assertEqual(first, 0)
        self.assertEqual(second, 1)
        self.assertNotEqual(bf16_round(first), bf16_round(second))


if __name__ == "__main__":
    unittest.main()
