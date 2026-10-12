"""Verify immutable dense KDA descriptors without importing accelerator code."""
import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

SOURCE = Path(__file__).resolve().parents[5] / (
    "python/sglang/srt/hardware_backend/npu/attention/ascend_kda_backend.py"
)


def descriptor_method():
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "AscendKDAAttnBackend")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                  and n.name == "_forward_target_verify")
    branch = next(n for n in method.body if isinstance(n, ast.If)
                  and "self._glm53_verify_full_steps is not None" in ast.unparse(n.test))
    function = ast.FunctionDef(
        name="descriptors",
        args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=n) for n in (
            "self", "batch_size", "draft_token_num", "num_dense_tokens", "mixed_qkv")],
            kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=[branch, ast.Return(value=ast.Tuple(elts=[ast.Name(id="num_accepted_tokens", ctx=ast.Load()),
            ast.Name(id="dense_query_start_loc", ctx=ast.Load())], ctx=ast.Load()))],
        decorator_list=[])
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),
                 str(SOURCE),"exec"),namespace)
    return namespace["descriptors"]


class VerifyConstantsCPU(unittest.TestCase):
    def test_dense_descriptors_and_addresses_are_reused(self):
        method = descriptor_method()
        for steps in (1, 3, 4, 5):
            capacity = 16
            full = torch.full((capacity,),steps,dtype=torch.int32)
            starts = torch.arange(0,capacity*steps+1,steps,dtype=torch.int32)
            backend = SimpleNamespace(_glm53_verify_full_steps=full,
                _glm53_verify_dense_starts=starts,_glm53_verify_const_steps=steps)
            for batch in (1,2,8,16,2,1):
                with patch.object(torch,"full",side_effect=AssertionError("unexpected allocation")), \
                     patch.object(torch,"arange",side_effect=AssertionError("unexpected allocation")):
                    actual,lengths=method(backend,batch,steps,batch*steps,torch.empty(0))
                self.assertEqual(actual.tolist(),[steps]*batch)
                self.assertEqual(lengths.tolist(),[i*steps for i in range(batch+1)])
                self.assertEqual(actual.data_ptr(),full.data_ptr())
                self.assertEqual(lengths.data_ptr(),starts.data_ptr())
            self.assertEqual(full.tolist(),[steps]*capacity)

    def test_disabled_changed_width_and_capacity_fallback(self):
        method = descriptor_method()
        for enabled,steps,batch in ((False,4,2),(True,3,2),(True,4,5)):
            backend=SimpleNamespace(_glm53_verify_full_steps=torch.full((4,),4,dtype=torch.int32)
                if enabled else None,_glm53_verify_dense_starts=torch.arange(0,17,4,dtype=torch.int32),
                _glm53_verify_const_steps=4)
            actual,lengths=method(backend,batch,steps,batch*steps,torch.empty(0))
            self.assertEqual(actual.tolist(),[steps]*batch)
            self.assertEqual(lengths.tolist(),[i*steps for i in range(batch+1)])


if __name__ == "__main__":
    unittest.main()
