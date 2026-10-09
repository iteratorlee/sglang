"""CPU checks of the actual GLM53 P scorer's one-shot debug validation.

Run: python -B test/manual/ascend/test_glm53_prefill_score_validate_cpu.py -v
Reuses the existing AST method loader so torch_npu/SGLang need not import.
"""

import importlib.util
import os
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch


HARNESS_PATH = Path(__file__).with_name("test_glm53_pooled_score_prefill_cpu.py")
SPEC = importlib.util.spec_from_file_location("glm53_pooled_score_harness", HARNESS_PATH)
HARNESS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = HARNESS
SPEC.loader.exec_module(HARNESS)

SCORER = HARNESS.actual_function("_prefill_pooled_topk")
CLAIM = HARNESS.actual_function("_claim_prefill_score_validation")
VALIDATE = HARNESS.actual_function("_validate_prefill_score_tile")


class TaggedQuery:
    device = types.SimpleNamespace(type="npu")

    def __init__(self, data):
        self.data = data

    def __getitem__(self, key):
        return self.data[key]


class CPUTorch:
    def __getattr__(self, name):
        return getattr(torch, name)

    def arange(self, *args, **kwargs):
        kwargs.pop("device", None)
        return torch.arange(*args, **kwargs)


class TestActualPrefillScoreValidate(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        CLAIM.__globals__["_prefill_score_validation_claimed"] = False
        CLAIM.__globals__["_prefill_score_validation_lock"] = threading.Lock()
        self.logger = Mock()
        self.formats = []

        def get_format(tensor):
            self.formats.append((tuple(tensor.shape), tensor.dtype))
            if tensor.ndim == 3:
                return 29
            return 0 if tensor.shape[-1] == 32 else 3

        VALIDATE.__globals__.update(
            torch_npu=types.SimpleNamespace(get_npu_format=get_format),
            logger=self.logger,
        )
        self.bad_score = False
        self.fused_calls = 0

        def fused(query, weights, keys):
            self.fused_calls += 1
            scores = torch.zeros((query.shape[0], keys.shape[0]), dtype=torch.bfloat16)
            if self.bad_score:
                scores[0, 0] = 1
            return scores

        self.fused = fused
        SCORER.__globals__.update(
            torch=CPUTorch(),
            F=torch.nn.functional,
            _prefill_topk_tile_rows=lambda *_: 8,
            _prefill_pooled_score_enabled=lambda *_: True,
            _prefill_pooled_score_backend=lambda: (self.fused, lambda *_: True),
            _log_prefill_pooled_score_once=lambda *_: None,
            _PREFILL_SCORE_VALIDATE_MIN_KEYS=16384,
            _claim_prefill_score_validation=CLAIM,
            _validate_prefill_score_tile=VALIDATE,
        )
        self.indexer = types.SimpleNamespace(index_topk=2048, index_kpool=4)

    def call(self, seq_len=65536, rows=5):
        query = TaggedQuery(torch.zeros((rows, 32, 128), dtype=torch.bfloat16))
        weights = torch.ones((rows, 32), dtype=torch.float32)
        keys = torch.zeros((seq_len // 4, 128), dtype=torch.bfloat16)
        batch = types.SimpleNamespace(
            batch_size=1, extend_seq_lens_cpu=[rows], seq_lens_cpu=[seq_len]
        )
        return SCORER(self.indexer, query, weights, [keys], batch)

    def log_message(self):
        self.logger.info.assert_called_once()
        args = self.logger.info.call_args.args
        return args[0] % args[1:]

    def test_default_off_skips_reference_even_if_fused_score_is_wrong(self):
        self.bad_score = True
        with patch.dict(os.environ, {}, clear=True):
            self.call()
        self.assertEqual(self.fused_calls, 1)
        self.assertFalse(CLAIM.__globals__["_prefill_score_validation_claimed"])
        self.assertEqual(self.formats, [])
        self.logger.info.assert_not_called()

    def test_first_late_tile_checks_only_four_rows_once_and_logs_actual_formats(self):
        with patch.dict(os.environ, {"SGLANG_GLM53_PREFILL_SCORE_VALIDATE": "1"}):
            self.call(seq_len=65532, rows=1)  # 16383 candidate keys: no claim
            self.assertFalse(CLAIM.__globals__["_prefill_score_validation_claimed"])
            self.call(seq_len=65536, rows=5)
            self.bad_score = True
            self.call(seq_len=65536, rows=5)  # already claimed: no reference
        self.assertEqual(self.fused_calls, 3)
        self.assertTrue(CLAIM.__globals__["_prefill_score_validation_claimed"])
        self.assertEqual([shape for shape, _ in self.formats],
                         [(5, 32, 128), (5, 32), (16384, 128)])
        self.assertEqual([dtype for _, dtype in self.formats],
                         [torch.bfloat16, torch.float32, torch.bfloat16])
        message = self.log_message()
        for field in ("q_format=29", "weights_format=0", "key_format=3",
                      "score_max_abs=0.0", "score_mean_abs=0.0",
                      "top512_set_exact_rows=4/4"):
            self.assertIn(field, message)

    def test_nonzero_score_error_raises_and_still_claims_only_once(self):
        self.bad_score = True
        with patch.dict(os.environ, {"SGLANG_GLM53_PREFILL_SCORE_VALIDATE": "1"}):
            with self.assertRaisesRegex(RuntimeError, "score_max_abs=1.0"):
                self.call()
            self.call()  # no second validation after the first failure
        self.assertTrue(CLAIM.__globals__["_prefill_score_validation_claimed"])
        self.assertIn("score_max_abs=1.0", self.log_message())

    def test_absent_format_api_logs_unavailable_and_inactive_fusion_skips(self):
        VALIDATE.__globals__["torch_npu"] = types.SimpleNamespace()
        with patch.dict(os.environ, {"SGLANG_GLM53_PREFILL_SCORE_VALIDATE": "1"}):
            self.call()
        self.assertIn("q_format=unavailable", self.log_message())
        self.assertIn("weights_format=unavailable", self.log_message())
        self.assertIn("key_format=unavailable", self.log_message())

        self.logger.reset_mock()
        CLAIM.__globals__["_prefill_score_validation_claimed"] = False
        SCORER.__globals__["_prefill_pooled_score_enabled"] = lambda *_: False
        with patch.dict(os.environ, {"SGLANG_GLM53_PREFILL_SCORE_VALIDATE": "1"}):
            self.call(seq_len=16, rows=1)
        self.assertFalse(CLAIM.__globals__["_prefill_score_validation_claimed"])
        self.logger.info.assert_not_called()


if __name__ == "__main__":
    unittest.main()
