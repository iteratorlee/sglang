"""CPU/source regression for the opt-in GLM-5.3 KPool block4 SFA path.

Run from the source repository:

    python -B test/manual/ascend/test_glm53_sfa_kpool_cpu.py -v

The test executes the production layout and seed-publication methods extracted
from this checkout. NumPy tensor stand-ins exercise the NPU-only publication
guard without importing SGLang, PyTorch, torch_npu, or loading a model. The
separate Ascend operator oracle remains responsible for CANN numerical parity.
"""

from __future__ import annotations

import ast
import copy
import os
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
BACKEND_SOURCE = (
    ROOT / "python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py"
)
KPOOL_SOURCE = (
    ROOT
    / "python/sglang/srt/hardware_backend/npu/attention/glm53/kpool_indexer.py"
)
SHARE_SOURCE = ROOT / "python/sglang/srt/layers/attention/index_topk_share.py"
KPOOL_MODULE = (
    "sglang.srt.hardware_backend.npu.attention.glm53.kpool_indexer"
)
NS = types.SimpleNamespace


class Tensor:
    """Small NumPy-backed subset used by the extracted production methods."""

    def __init__(self, value, *, dtype=None, device="cpu"):
        if isinstance(value, Tensor):
            value = value.values
        self.values = np.asarray(value, dtype=dtype)
        self.device = NS(type=device)

    @property
    def dtype(self):
        return self.values.dtype

    @property
    def ndim(self):
        return self.values.ndim

    @property
    def shape(self):
        return self.values.shape

    def __len__(self):
        return len(self.values)

    def _new(self, value):
        return Tensor(value, device=self.device.type)

    @staticmethod
    def _index(index):
        if isinstance(index, Tensor):
            return index.values
        if isinstance(index, tuple):
            return tuple(Tensor._index(value) for value in index)
        return index

    def __getitem__(self, index):
        return self._new(self.values[self._index(index)])

    def __setitem__(self, index, value):
        self.values[self._index(index)] = _array(value)

    def _binary(self, other, operator):
        return self._new(operator(self.values, _array(other)))

    def __add__(self, other):
        return self._binary(other, np.add)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._binary(other, np.subtract)

    def __rsub__(self, other):
        return self._new(np.subtract(_array(other), self.values))

    def __mul__(self, other):
        return self._binary(other, np.multiply)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __eq__(self, other):
        return self._binary(other, np.equal)

    def __ge__(self, other):
        return self._binary(other, np.greater_equal)

    def __gt__(self, other):
        return self._binary(other, np.greater)

    def __le__(self, other):
        return self._binary(other, np.less_equal)

    def __lt__(self, other):
        return self._binary(other, np.less)

    def __and__(self, other):
        return self._binary(other, np.logical_and)

    def unsqueeze(self, axis):
        return self._new(np.expand_dims(self.values, axis))

    def flatten(self, start_dim=0):
        shape = self.shape[:start_dim] + (-1,)
        return self._new(self.values.reshape(shape))

    def view(self, *shape):
        return self._new(self.values.reshape(shape))

    def to(self, dtype=None, **_kwargs):
        return self._new(self.values.astype(dtype or self.dtype, copy=False))

    def copy_(self, other):
        np.copyto(self.values, _array(other))
        return self


def _array(value):
    return value.values if isinstance(value, Tensor) else value


def _device(*values):
    for value in values:
        if isinstance(value, Tensor):
            return value.device.type
    return "cpu"


class FakeTorch(types.ModuleType):
    def __init__(self):
        super().__init__("torch")

    @staticmethod
    def arange(*args, device=None, dtype=None):
        return Tensor(
            np.arange(*args, dtype=dtype),
            device=getattr(device, "type", "cpu"),
        )

    @staticmethod
    def div(value, divisor, rounding_mode=None):
        result = np.divide(_array(value), divisor)
        if rounding_mode == "floor":
            result = np.floor(result)
        return Tensor(result.astype(value.dtype), device=value.device.type)

    @staticmethod
    def remainder(value, divisor):
        return Tensor(np.remainder(_array(value), divisor), device=value.device.type)

    @staticmethod
    def minimum(left, right):
        return Tensor(
            np.minimum(_array(left), _array(right)), device=_device(left, right)
        )

    @staticmethod
    def full_like(value, fill_value):
        return Tensor(
            np.full_like(value.values, fill_value), device=value.device.type
        )

    @staticmethod
    def where(condition, left, right):
        return Tensor(
            np.where(_array(condition), _array(left), _array(right)),
            device=_device(left, right, condition),
        )


TORCH = FakeTorch()
F = NS(
    pad=lambda tensor, widths, value=-1: Tensor(
        np.pad(
            tensor.values,
            ((0, 0), (widths[0], widths[1])),
            constant_values=value,
        ),
        device=tensor.device.type,
    )
)


def extract_method(path: Path, owner: str, name: str, namespace: dict):
    tree = ast.parse(path.read_text())
    owners = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == owner
    ]
    if len(owners) != 1:
        raise AssertionError((path, owner, len(owners)))
    matches = [
        node
        for node in owners[0].body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    ]
    if len(matches) != 1:
        raise AssertionError((path, owner, name, len(matches)))
    definition = copy.deepcopy(matches[0])
    definition.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            definition,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


def extract_class(path: Path, name: str, namespace: dict):
    tree = ast.parse(path.read_text())
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == name
    ]
    if len(matches) != 1:
        raise AssertionError((path, name, len(matches)))
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            copy.deepcopy(matches[0]),
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


EXPAND_TOKENS = extract_method(
    KPOOL_SOURCE,
    "AscendIndexerKPoolMixin",
    "_expand_with_tail",
    {"os": os, "torch": TORCH, "F": F},
)
LAYOUT_BLOCKS = extract_method(
    KPOOL_SOURCE,
    "AscendIndexerKPoolMixin",
    "_pool_blocks_with_tail",
    {"torch": TORCH, "F": F},
)
SHARE = extract_class(
    SHARE_SOURCE,
    "IndexTopKShareState",
    {"contextmanager": contextmanager},
)


class Mode:
    def __init__(self, kind):
        self.kind = kind

    def is_extend(self, include_draft_extend_v2=False):
        return self.kind == "extend" or (
            include_draft_extend_v2 and self.kind == "draft_extend_v2"
        )

    def is_draft_extend_v2(self):
        return self.kind == "draft_extend_v2"

    def is_target_verify(self):
        return self.kind == "target_verify"


OWNER = NS(index_kpool=4)


def pools_for(positions):
    generator = np.random.default_rng(202609211216)
    pools = np.full((len(positions), 512), -1, dtype=np.int32)
    for row, position in enumerate(positions):
        closed = (position + 1) // 4
        count = min(closed, 512)
        if count:
            pools[row, :count] = generator.permutation(closed)[:count]
    return Tensor(pools, device="npu")


def layouts(position_values):
    positions = Tensor(position_values, dtype=np.int64, device="npu")
    pools = pools_for(position_values)
    with patch.dict(os.environ, {"SGLANG_GLM53_KPOOL_EXPAND": "0"}):
        tokens = EXPAND_TOKENS(OWNER, pools, positions)
    blocks = LAYOUT_BLOCKS(OWNER, pools, positions)
    return tokens, blocks, positions


def batch_for(positions, seed, select=None, *, mode="extend", carry=False):
    return NS(
        forward_mode=Mode(mode),
        positions=positions,
        reuse_dsa_topk_indices=carry,
        attn_cp_metadata=None,
        spec_info=NS(
            dsa_seed_topk_capture=seed,
            dsa_seed_topk_select=select,
            dsa_topk_indices=None,
        ),
    )


@contextmanager
def production_imports():
    kpool_module = types.ModuleType(KPOOL_MODULE)
    kpool_module.get_prefill_sparse_block_size = lambda _batch: 4
    with patch.dict(sys.modules, {"torch": TORCH, KPOOL_MODULE: kpool_module}):
        yield


class KPoolBlockLayoutTests(unittest.TestCase):
    def test_boundaries_preserve_visible_tokens_and_every_seed_slot(self):
        position_values = (
            list(range(8))
            + [62, 63, 64, 65]
            + list(range(2046, 2053))
            + [4093, 4094, 4095, 4096, 65535, 65536]
        )
        tokens, blocks, positions = layouts(position_values)
        with production_imports():
            restored = SHARE._expand_kpool_seed(
                blocks, positions, tokens.shape[1]
            )

        self.assertEqual(tokens.shape, (len(position_values), 2051))
        self.assertEqual(blocks.shape, (len(position_values), 513))
        np.testing.assert_array_equal(restored.values, tokens.values)

        offsets = np.arange(4, dtype=np.int64)
        for row, position in enumerate(position_values):
            with self.subTest(position=position):
                token_visible = sorted(
                    tokens.values[row][tokens.values[row] <= position].tolist()
                )
                expanded_blocks = (
                    blocks.values[row, :, None].astype(np.int64) * 4 + offsets
                ).reshape(-1)
                block_visible = sorted(
                    expanded_blocks[expanded_blocks <= position].tolist()
                )
                self.assertEqual(block_visible, token_visible)

                capacity = ((position + 64) // 64 + 1) * 64
                self.assertTrue(np.all(tokens.values[row] < capacity))
                self.assertTrue(np.all(blocks.values[row] * 4 + 3 < capacity))

    def test_block4_is_explicit_opt_in(self):
        tree = ast.parse(BACKEND_SOURCE.read_text())
        defaults = [
            call.args[1].value
            for call in ast.walk(tree)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "get_bool_env_var"
            and len(call.args) >= 2
            and isinstance(call.args[0], ast.Constant)
            and call.args[0].value == "SGLANG_GLM53_SFA_KPOOL_BLOCK"
            and isinstance(call.args[1], ast.Constant)
        ]
        self.assertEqual(defaults, ["False"])


class MTPSeedPublicationTests(unittest.TestCase):
    def assert_publish_exact(self, position_values, select=None, padded_rows=None):
        tokens, blocks, positions = layouts(position_values)
        selected = (
            np.arange(len(position_values))
            if select is None
            else np.asarray(select)
        )
        expected = tokens.values[selected]

        if padded_rows is not None:
            self.assertGreaterEqual(padded_rows, len(position_values))
            block_padding = np.zeros(
                (padded_rows - len(position_values), blocks.shape[1]), dtype=np.int32
            )
            position_padding = np.zeros(
                padded_rows - len(position_values), dtype=np.int64
            )
            blocks = Tensor(
                np.concatenate((blocks.values, block_padding)), device="npu"
            )
            positions = Tensor(
                np.concatenate((positions.values, position_padding)), device="npu"
            )

        selector = (
            None
            if select is None
            else Tensor(select, dtype=np.int64, device="npu")
        )
        seed = Tensor(np.full(expected.shape, -9, dtype=np.int32), device="npu")
        forward_batch = batch_for(positions, seed, selector)
        state = SHARE(forward_batch, blocks)
        with production_imports():
            state.publish()

        np.testing.assert_array_equal(seed.values, expected)
        self.assertIs(state.topk_indices, blocks)
        self.assertIsNone(forward_batch.spec_info.dsa_topk_indices)

    def test_all_rows_publish_maps_all_2051_slots(self):
        self.assert_publish_exact(
            list(range(8))
            + [62, 63, 64, 65]
            + list(range(2046, 2053))
            + [4093, 4094, 4095, 4096, 65535, 65536]
        )

    def test_selected_multi_request_rows_ignore_padding(self):
        self.assert_publish_exact(
            [62, 63, 64, 0, 1, 2, 3, 4, 63, 64, 65, 63],
            select=[2, 7, 10, 11],
            padded_rows=16,
        )

    def test_existing_token_layout_publish_is_unchanged(self):
        tokens, _blocks, positions = layouts([0, 1, 2, 3, 63, 64, 65])
        tokens.device.type = "cpu"
        seed = Tensor(np.full(tokens.shape, -9, dtype=np.int32), device="cpu")
        SHARE(batch_for(positions, seed), tokens).publish()
        np.testing.assert_array_equal(seed.values, tokens.values)

    def test_decode_carry_identity_and_cleanup_are_unchanged(self):
        carry_seed = object()
        forward_batch = batch_for(
            Tensor([64], dtype=np.int64, device="npu"),
            None,
            mode="decode",
        )
        forward_batch.spec_info.dsa_topk_indices = carry_seed

        with SHARE.mtp_iteration(
            forward_batch, keep_carry_seed=True
        ) as carry_state:
            self.assertTrue(forward_batch.reuse_dsa_topk_indices)
            self.assertIs(carry_state.topk_indices, carry_seed)
            carry_state.publish()
            self.assertIs(forward_batch.spec_info.dsa_topk_indices, carry_seed)

        self.assertFalse(forward_batch.reuse_dsa_topk_indices)
        self.assertIsNone(forward_batch.spec_info.dsa_topk_indices)


if __name__ == "__main__":
    unittest.main()
