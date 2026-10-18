"""Metadata guards and tiny CPU interpretation; no target-NPU precision claim."""

import ast
import importlib.util
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
os.environ["TRITON_INTERPRET"] = "1"

import torch

ROOT = Path(__file__).resolve().parents[5]
spec = importlib.util.spec_from_file_location(
    "kda_bhv_harness", ROOT / "test/manual/ascend/test_glm53_kda_bhv_npu.py"
)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


class Meta:
    """Real metadata/strides without allocating large state or touching devices."""

    def __init__(self, shape=None, dtype=torch.bfloat16, tensor=None, device=None):
        self.tensor = (
            tensor
            if tensor is not None
            else torch.empty(shape, dtype=dtype, device="meta")
        )
        self.device = device if device is not None else NS(type="npu", index=0)

    @property
    def shape(self):
        return self.tensor.shape

    @property
    def ndim(self):
        return self.tensor.ndim

    @property
    def dtype(self):
        return self.tensor.dtype

    def numel(self):
        return self.tensor.numel()

    def stride(self, axis):
        return self.tensor.stride(axis)

    def is_contiguous(self):
        return self.tensor.is_contiguous()

    def contiguous(self):
        return Meta(tensor=self.tensor.contiguous(), device=self.device)

    def transpose(self, *axes):
        return Meta(tensor=self.tensor.transpose(*axes), device=self.device)


def metadata(batch=13, heads=32, dim=128, steps=4):
    result = {
        name: Meta((1, batch * steps, heads, dim)) for name in ("q", "k", "v", "a")
    }
    result.update(
        b=Meta((1, batch * steps, heads)),
        A_log=Meta((1, 1, heads, 1), torch.float32),
        dt_bias=Meta((heads * dim,), torch.float32),
        initial_state_source=Meta(
            (batch + 3, heads, dim, dim), torch.float32
        ).transpose(-1, -2),
        intermediate_state=Meta(
            (batch + 1, steps, heads, dim, dim), torch.float32
        ).transpose(-1, -2),
        initial_state_indices=Meta((batch,), torch.int32),
        cu_seqlens=Meta((batch + 1,), torch.int32),
        prefill=False,
        lower_bound=-5.0,
    )
    return result


class FakeKernel:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        return lambda **kwargs: self.calls.append((grid, kwargs))


class KdaBHVCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_guard_and_wrapper_dispatch(self):
        module = h.load_module()
        kernel = FakeKernel()
        module._glm_kda_varlen_recurrent_kernel = kernel
        fake_torch = NS(
            bfloat16=torch.bfloat16,
            float32=torch.float32,
            int32=torch.int32,
            int64=torch.int64,
            zeros_like=lambda t: Meta(t.shape, t.dtype),
        )
        with patch.object(module, "torch", fake_torch):
            for batch in (1, 11, 12, 13, 14, 15):
                for flag in ("0", "1", "2", "4", "8", "invalid"):
                    for role in ("decode", "prefill", "null"):
                        module.get_server_args = lambda: NS(disaggregation_mode=role)
                        with patch.dict(os.environ, {h.FLAG: flag}):
                            module.glm_kda_varlen_recurrent_npu(**metadata(batch))
                        expected = (
                            int(flag)
                            if flag in ("2", "4")
                            and role == "decode"
                            and batch in (12, 13, 14)
                            else 1
                        )
                        grid, kw = kernel.calls[-1]
                        self.assertEqual(grid, (2, batch, 32 // expected))
                        self.assertEqual(
                            (kw["BHV"], kw["BK"], kw["BV"]), (expected, 128, 64)
                        )
                        self.assertEqual(
                            (kw["num_warps"], kw["num_stages"], kw["multibuffer"]),
                            (1, 3, False),
                        )

    def test_unsupported_metadata_falls_back(self):
        module = h.load_module()
        kernel = FakeKernel()
        module._glm_kda_varlen_recurrent_kernel = kernel
        base = metadata()
        cases = [
            dict(prefill=True),
            dict(intermediate_state=None),
            dict(
                intermediate_state=Meta((14, 3, 32, 128, 128), torch.float32).transpose(
                    -1, -2
                )
            ),
            dict(initial_state_source=Meta((16, 32, 128, 128), torch.float32)),
            dict(intermediate_state=Meta((14, 4, 32, 128, 128), torch.float32)),
            dict(A_log=Meta((32,), torch.bfloat16)),
            dict(dt_bias=Meta((4096,), torch.bfloat16)),
            dict(A_log=Meta((31,), torch.float32)),
            dict(q=Meta(base["q"].shape, device=NS(type="cpu", index=0))),
            dict(dt_bias=Meta((4096,), torch.float32, device=NS(type="npu", index=1))),
            dict(initial_state_indices=Meta((1, 13), torch.int32)),
            dict(cu_seqlens=Meta((1, 14), torch.int32)),
        ]
        for name in ("q", "k", "v", "a", "b"):
            cases.append({name: Meta(base[name].shape, torch.float32)})
        fake_torch = NS(
            bfloat16=torch.bfloat16,
            float32=torch.float32,
            int32=torch.int32,
            int64=torch.int64,
            zeros_like=lambda t: Meta(t.shape, t.dtype),
        )
        with patch.object(module, "torch", fake_torch), patch.dict(
            os.environ, {h.FLAG: "4"}
        ):
            for changes in cases:
                module.glm_kda_varlen_recurrent_npu(**(base | changes))
                self.assertEqual(kernel.calls[-1][1]["BHV"], 1, changes)
            for heads, dim in ((16, 128), (32, 64)):
                module.glm_kda_varlen_recurrent_npu(**metadata(heads=heads, dim=dim))
                self.assertEqual(kernel.calls[-1][1]["BHV"], 1)
            # Both state and scratch can use BF16 in the old path; reject new dispatch.
            changed = dict(
                initial_state_source=Meta((16, 32, 128, 128)).transpose(-1, -2),
                intermediate_state=Meta((14, 4, 32, 128, 128)).transpose(-1, -2),
            )
            module.glm_kda_varlen_recurrent_npu(**(base | changed))
            self.assertEqual(kernel.calls[-1][1]["BHV"], 1)

    def test_tiny_interpreter_exact_all_snapshots_and_padding(self):
        module = h.load_module()
        # H4 K8 V8: only 1/2/4 programs per sequence, never a production grid.
        for pattern in h.PATTERNS:
            host = h.inputs(2, pattern, heads=4, dim=8)
            bufs = [h.Buffers(host, "cpu") for _ in (1, 2, 4)]
            for buf, bhv in zip(bufs, (1, 2, 4)):
                buf.reset(host)
                h.direct(module, buf, bhv)
            reference = h.host_result(bufs[0], bufs[0].output)
            for buf, bhv in zip(bufs, (1, 2, 4)):
                h.validate(buf, buf.output, host, reference, f"tiny {pattern} BHV{bhv}")

    def test_dynamic_slots_and_lengths_keep_addresses(self):
        module = h.load_module()
        host = h.inputs(2, heads=4, dim=8)
        buf = h.Buffers(host, "cpu")
        pointers = buf.pointers()
        previous_slots = None
        for i, pattern in enumerate(h.PATTERNS):
            host = h.inputs(2, pattern, 53 + i, heads=4, dim=8)
            ref = h.Buffers(host, "cpu")
            ref.reset(host)
            h.direct(module, ref, 1)
            buf.reset(host)
            h.direct(module, buf, 4)
            h.validate(buf, buf.output, host, h.host_result(ref, ref.output), pattern)
            self.assertEqual(pointers, buf.pointers())
            previous_slots = host["indices"]
        self.assertIsNotNone(previous_slots)

    def test_kernel_body_and_real_verify_call_contract(self):
        path = "python/sglang/srt/hardware_backend/npu/attention/glm53/kda_recurrent_npu.py"
        before = ast.parse(
            subprocess.check_output(
                ["git", "show", f"f3260b0:{path}"], cwd=ROOT, text=True
            )
        )
        after = ast.parse((ROOT / path).read_text())
        kernel = lambda tree: next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef)
            and n.name == "_glm_kda_varlen_recurrent_kernel"
        )
        self.assertEqual(ast.dump(kernel(before)), ast.dump(kernel(after)))
        backend = ast.parse(
            (
                ROOT
                / "python/sglang/srt/hardware_backend/npu/attention/ascend_kda_backend.py"
            ).read_text()
        )
        target = next(
            n
            for n in ast.walk(backend)
            if isinstance(n, ast.FunctionDef) and n.name == "_forward_target_verify"
        )
        call = next(
            n
            for n in ast.walk(target)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "_glm_recurrent"
        )
        self.assertIn("intermediate", [kw.arg for kw in call.keywords])
        self.assertNotIn("prefill", [kw.arg for kw in call.keywords])
        recurrent = next(
            n
            for n in ast.walk(backend)
            if isinstance(n, ast.FunctionDef) and n.name == "_glm_recurrent"
        )
        defaults = dict(
            zip((a.arg for a in recurrent.args.kwonlyargs), recurrent.args.kw_defaults)
        )
        self.assertEqual(ast.literal_eval(defaults["prefill"]), False)
        self.assertEqual(
            sum(
                isinstance(n, ast.For)
                and isinstance(n.iter, ast.Call)
                and isinstance(n.iter.func, ast.Name)
                and n.iter.func.id == "range"
                for n in ast.walk(kernel(after))
            ),
            2,
        )


if __name__ == "__main__":
    unittest.main()
