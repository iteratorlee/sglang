"""CPU contracts for the opt-in local combine mode; no accelerator imports."""
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import torch


SOURCE = Path(__file__).resolve().parents[2] / "python/sglang/srt/hardware_backend/npu/moe/glm53_collectives.py"
spec = importlib.util.spec_from_file_location("normal_combine_under_test", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
FLAG = "SGLANG_GLM53_NORMAL_COMBINE_DETERMINISTIC"


def flags():
    return (torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled())


class CombineDeterminism(unittest.TestCase):
    def setUp(self):
        self.before = flags()

    def tearDown(self):
        torch.use_deterministic_algorithms(self.before[0], warn_only=self.before[1])

    def test_scope_and_exception_restore(self):
        for enabled in (False, True):
            for warn_only in (False, True):
                for active in (False, True):
                    for fail in (False, True):
                        with self.subTest(enabled=enabled, warn_only=warn_only, active=active, fail=fail):
                            torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
                            expected = (True, False) if active else (enabled, warn_only)
                            seen = []
                            class Destination:
                                def index_add_(self, dim, rows, values):
                                    seen.append(flags())
                                    if fail:
                                        raise RuntimeError("native failure")
                                    return self
                            out = Destination()
                            with patch.dict(os.environ, {FLAG: "1" if active else "0"}):
                                if fail:
                                    with self.assertRaisesRegex(RuntimeError, "native failure"):
                                        module._normal_combine_index_add_(out, None, None)
                                else:
                                    self.assertIs(module._normal_combine_index_add_(out, None, None), out)
                            self.assertEqual(seen, [expected])
                            self.assertEqual(flags(), (enabled, warn_only))

    def test_default_does_not_set_global_mode(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(torch, "use_deterministic_algorithms") as change:
            out = torch.zeros(2, 2)
            module._normal_combine_index_add_(out, torch.tensor([0, 0]), torch.ones(2, 2))
            change.assert_not_called()
            self.assertTrue(torch.equal(out, torch.tensor([[2., 2.], [0., 0.]])))

    def test_native_combine_and_communication_scope(self):
        y = torch.tensor([[1., -2.], [3., 4.], [-1., 0.5]], dtype=torch.bfloat16)
        rows = torch.tensor([0, 1, 0])
        weights = torch.tensor([0.5, 0.25, 0.125])
        state = (rows, weights, 2, 2, 2, 3)
        expected = torch.zeros(4, 2).index_add_(0, rows, y.float() * weights[:, None])
        for reduce_scatter in (False, True):
            for enabled, warn_only in ((False, False), (True, True)):
                torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
                calls = []
                def all_reduce(value, **kwargs):
                    calls.append(flags())
                    self.assertTrue(torch.equal(value, expected))
                def scatter(output, value, **kwargs):
                    calls.append(flags())
                    self.assertTrue(torch.equal(value, expected))
                    output.copy_(value[:2])
                with patch.dict(os.environ, {FLAG: "1", "SGLANG_GLM53_NORMAL_REDUCE_SCATTER": str(int(reduce_scatter))}), \
                     patch.object(module.dist, "all_reduce", all_reduce), \
                     patch.object(module.dist, "reduce_scatter_tensor", scatter), \
                     patch.object(module.dist, "get_rank", return_value=0):
                    result = module.collective_combine(None, y, state)
                self.assertEqual(calls, [(enabled, warn_only)])
                self.assertTrue(torch.equal(result, expected[:2].to(y.dtype)))
                self.assertEqual(flags(), (enabled, warn_only))


if __name__ == "__main__":
    unittest.main()
