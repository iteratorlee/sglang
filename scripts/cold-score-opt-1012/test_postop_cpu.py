"""Fallback ownership and physical-format guards, without NPU initialization."""

import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from score_contract import REL, extract, fixture, make_scorer

PATH = Path(__file__).resolve().parents[2] / REL / "pooled_score_postop_npu.py"


class PostopFallback(unittest.TestCase):
    def setUp(self):
        self.scope = dict(
            torch=torch, triton=SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b)
        )
        self.supported = extract(PATH, "supported", self.scope)
        self.operation = extract(PATH, "relu_gate_", self.scope)

    def test_cpu_fallback_executes_exactly_once_and_returns_false(self):
        scores = torch.arange(-96, 96).bfloat16().reshape(2, 32, 3)
        weights = torch.linspace(-2.2, 3.1, 64).reshape(2, 32)
        expected = scores.clone().relu_().mul_(weights.unsqueeze(-1))
        self.assertFalse(self.operation(scores, weights))
        self.assertTrue(
            torch.equal(scores.view(torch.int16), expected.view(torch.int16))
        )
        # Caller always sums after the helper; False is never an instruction
        # to repeat the product (which would square the signed gate).
        self.assertTrue(torch.equal(scores.sum(1), expected.sum(1)))

    def test_noncontiguous_fallback_mutates_only_view(self):
        storage = torch.ones((1, 32, 6), dtype=torch.bfloat16)
        scores = storage[:, :, ::2]
        weights = torch.full((1, 32), -2.0)
        self.assertFalse(self.operation(scores, weights))
        self.assertTrue((storage[:, :, ::2] == -2).all())
        self.assertTrue((storage[:, :, 1::2] == 1).all())

    def test_cpu_zero_semantics_follow_cpu_reference(self):
        # CANN ReLU maps -0 to +0; PyTorch CPU may retain -0. The fallback
        # follows its native platform. This does not emulate the NPU kernel.
        scores = torch.tensor([0, -32768], dtype=torch.int16).view(torch.bfloat16)
        scores = scores.reshape(1, 1, 2).repeat(1, 32, 1)
        weights = torch.tensor([1.0, -1.0]).repeat(16).reshape(1, 32)
        expected = scores.clone().relu_().mul_(weights.unsqueeze(-1))
        self.assertFalse(self.operation(scores, weights))
        self.assertTrue(
            torch.equal(scores.view(torch.int16), expected.view(torch.int16))
        )

    def test_physical_format_and_dtype_guard(self):
        device = SimpleNamespace(type="npu")

        def fake(shape, dtype, contiguous=True):
            return SimpleNamespace(
                shape=shape,
                ndim=len(shape),
                device=device,
                dtype=dtype,
                is_contiguous=lambda: contiguous,
            )

        scores, weights = fake((8, 32, 1024), torch.bfloat16), fake(
            (8, 32), torch.float32
        )
        for fmt, expected in ((2, True), (0, False), (29, False)):
            with patch.dict(
                sys.modules,
                {"torch_npu": SimpleNamespace(get_npu_format=lambda _: fmt)},
            ):
                self.assertEqual(self.supported(scores, weights), expected)
        with patch.dict(
            sys.modules, {"torch_npu": SimpleNamespace(get_npu_format=lambda _: 2)}
        ):
            self.assertFalse(self.supported(scores, fake((8, 32), torch.bfloat16)))
            self.assertFalse(
                self.supported(fake((8, 32, 1024), torch.bfloat16, False), weights)
            )
            self.assertFalse(
                self.supported(
                    fake((128, 32, 131072), torch.bfloat16),
                    fake((128, 32), torch.float32),
                )
            )
        with patch.dict(sys.modules, {"torch_npu": SimpleNamespace()}):
            self.assertFalse(self.supported(scores, weights))

    def test_grid_limit_fallback_cannot_launch_or_double_gate(self):
        self.scope["supported"] = lambda *_: True
        for rows, block in ((32, 2048), (128, 8192)):
            with self.subTest(rows=rows, block=block):
                scores = Mock(shape=(rows, 32, 131072))
                weights = Mock()
                self.assertFalse(self.operation(scores, weights, block=block))
                scores.relu_.assert_called_once_with()
                scores.mul_.assert_called_once_with(weights.unsqueeze.return_value)
                weights.unsqueeze.assert_called_once_with(-1)


class PostopIntegration(unittest.TestCase):
    def test_policy_is_opt_in_p_only_and_preserves_tile_policy(self):
        env = {}
        scope = dict(
            torch=torch,
            os=SimpleNamespace(environ=env),
            get_server_args=lambda: SimpleNamespace(disaggregation_mode="prefill"),
        )
        policy = extract(
            PATH.with_name("kpool_indexer.py"), "_prefill_pooled_postop_min_keys", scope
        )
        npu, cpu = SimpleNamespace(type="npu"), SimpleNamespace(type="cpu")
        self.assertIsNone(policy(npu, 32))
        for setting in ("1", "auto"):
            env["SGLANG_GLM53_PREFILL_POOLED_POSTOP"] = setting
            self.assertEqual(policy(npu, 32), 32768)
            self.assertEqual(policy(npu, 8), 98304)
            self.assertIsNone(policy(npu, 16))
            self.assertIsNone(policy(cpu, 32))
        scope["get_server_args"] = lambda: SimpleNamespace(disaggregation_mode="decode")
        self.assertIsNone(policy(npu, 32))
        env["SGLANG_GLM53_PREFILL_POOLED_POSTOP"] = "always"
        with self.assertRaises(ValueError):
            policy(npu, 32)

    def test_actual_caller_false_return_does_not_double_gate(self):
        repo = PATH.parents[7]
        # Native CPU arithmetic tests control flow only. The helper deliberately
        # returns False after applying the signed gate, as on an NPU fallback.
        q, w, k = fixture("signed", 33, 2048)
        calls = []

        def operation(scores, weights):
            calls.append(tuple(scores.shape))
            scores.relu_().mul_(weights.unsqueeze(-1))
            return False

        old = make_scorer(repo, False, 32)
        new = make_scorer(repo, False, 32, postop=operation, integrated=True)
        new.scope["_prefill_pooled_postop_min_keys"] = lambda *_: 0
        expected = old(q, w, [k], [33], [8192])
        actual = new(q, w, [k], [33], [8192])
        self.assertEqual(calls, [(32, 32, 2048), (1, 32, 2048)])
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(new.calls, [])  # False is activation status only.

    def test_threshold_skips_helper_without_changing_native_output(self):
        repo = PATH.parents[7]
        q, w, k = fixture("signed", 9, 1024)
        operation = Mock(side_effect=AssertionError("helper must not be called"))
        old = make_scorer(repo, False, 8)
        new = make_scorer(repo, False, 8, postop=operation, integrated=True)
        new.scope["_prefill_pooled_postop_min_keys"] = lambda *_: 98304
        self.assertTrue(
            torch.equal(old(q, w, [k], [9], [4096]), new(q, w, [k], [9], [4096]))
        )
        operation.assert_not_called()


if __name__ == "__main__":
    unittest.main()
