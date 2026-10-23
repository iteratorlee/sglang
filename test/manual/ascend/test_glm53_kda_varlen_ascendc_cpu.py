"""CPU metadata/dispatch/build contracts. No NPU correctness claim.

CUDA_VISIBLE_DEVICES='' TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -B \
  test/manual/ascend/test_glm53_kda_varlen_ascendc_cpu.py -v
"""

import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
import torch

ROOT = Path(__file__).resolve().parents[3]
REL = "python/sglang/srt/hardware_backend/npu/attention/glm53"
BASE = "93636f759906450beaa43b779e24aa2a51909d5b"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adapter = load("kda_adapter", ROOT / REL / "kda_varlen_ascendc.py")
builder = load("kda_builder", ROOT / "scripts/npu/build_glm53_kda_ascendc.py")


class Meta:
    """Torch meta storage gives real strides, with a simulated NPU device."""

    def __init__(self, shape=None, dtype=torch.bfloat16, *, tensor=None, device=None):
        self.tensor = (
            tensor
            if tensor is not None
            else torch.empty(shape, dtype=dtype, device="meta")
        )
        self.device = device or NS(type="npu", index=0)

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

    def stride(self, *args):
        return self.tensor.stride(*args)

    def is_contiguous(self):
        return self.tensor.is_contiguous()

    def contiguous(self):
        return Meta(tensor=self.tensor.contiguous(), device=self.device)

    def transpose(self, *args):
        return Meta(tensor=self.tensor.transpose(*args), device=self.device)


def inputs(heads=8, tokens=129, batch=1, prefill=True, steps=None, dtype=torch.int64):
    data = {n: Meta((1, tokens, heads, 128)) for n in ("q", "k", "v", "a")}
    data.update(
        b=Meta((1, tokens, heads)),
        A_log=Meta((1, 1, heads, 1), torch.float32),
        dt_bias=Meta((heads * 128,), torch.float32),
        initial_state_source=Meta(
            (batch + 5, heads, 128, 128), torch.float32
        ).transpose(-1, -2),
        initial_state_indices=Meta((batch,), dtype),
        cu_seqlens=Meta((batch + 1,), dtype),
        prefill=prefill,
        lower_bound=-5.0,
        scale=None,
        intermediate_state=None,
        track_state_indices=None,
        track_state_lens=None,
    )
    if steps is not None:
        data["intermediate_state"] = Meta(
            (batch + 2, steps, heads, 128, 128), torch.float32
        ).transpose(-1, -2)
    return data


class Launch:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        return lambda **kwargs: self.calls.append((grid, kwargs))


class DispatchCPU(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        adapter._logged.clear()
        # Execute the actual wrapper, with only device launches replaced. No
        # SGLang model import, backend auto-initialization, or Triton JIT.
        tree = ast.parse((ROOT / REL / "kda_recurrent_npu.py").read_text())
        nodes = [
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef)
            and n.name == "glm_kda_varlen_recurrent_npu"
        ]
        scope = dict(
            __package__="kda_contract",
            torch=NS(
                Tensor=torch.Tensor,
                float32=torch.float32,
                bfloat16=torch.bfloat16,
                int32=torch.int32,
                int64=torch.int64,
                zeros_like=lambda t: Meta(t.shape, t.dtype),
            ),
            Optional=__import__("typing").Optional,
            os=os,
            triton=NS(
                next_power_of_2=lambda n: 1 << (n - 1).bit_length(),
                cdiv=lambda a, b: (a + b - 1) // b,
            ),
        )
        self.legacy = Launch()
        scope["_glm_kda_varlen_recurrent_kernel"] = self.legacy
        self.prepared = Mock(return_value=object())
        prepared_module = ModuleType("kda_contract.kda_prefill_npu")
        prepared_module.run_prepared_prefill = self.prepared
        self.modules = patch.dict(
            sys.modules,
            {
                "kda_contract": ModuleType("kda_contract"),
                "kda_contract.kda_varlen_ascendc": adapter,
                "kda_contract.kda_prefill_npu": prepared_module,
            },
        )
        self.modules.start()
        self.addCleanup(self.modules.stop)
        exec(
            compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT / REL), "exec"),
            scope,
        )
        self.run_wrapper = scope["glm_kda_varlen_recurrent_npu"]

    def test_original_checks_unchanged(self):
        path = REL + "/kda_recurrent_npu.py"
        baseline = subprocess.check_output(
            ["git", "-C", str(ROOT), "show", f"{BASE}:{path}"], text=True
        )

        def prefix(source):
            fn = next(
                n
                for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef)
                and n.name == "glm_kda_varlen_recurrent_npu"
            )
            result = []
            for n in fn.body:
                result.append(ast.dump(n))
                if isinstance(n, ast.Assign) and "contiguous" in ast.unparse(n):
                    break
            return result

        self.assertEqual(prefix(baseline), prefix((ROOT / path).read_text()))

    def test_default_off_retains_prepared_branch_and_its_env(self):
        os.environ.update(
            SGLANG_GLM53_KDA_PREFILL_PREPARE="1", SGLANG_GLM53_KDA_PREPARED_ASCENDC="1"
        )
        with patch.object(
            adapter, "_operator", side_effect=AssertionError("loaded new library")
        ):
            self.assertIs(
                self.run_wrapper(**inputs(dtype=torch.int32)),
                self.prepared.return_value,
            )
        self.prepared.assert_called_once()
        self.assertEqual(os.environ["SGLANG_GLM53_KDA_PREPARED_ASCENDC"], "1")
        self.assertFalse(self.legacy.calls)

    def test_new_op_precedes_prepared_path_and_forwards_original_views(self):
        os.environ.update(
            SGLANG_GLM53_KDA_VARLEN_ASCENDC="1",
            SGLANG_GLM53_KDA_PREFILL_PREPARE="1",
            SGLANG_GLM53_KDA_PREPARED_ASCENDC="1",
        )
        data = inputs()
        data.update(
            track_state_indices=Meta((1,), torch.int64),
            track_state_lens=Meta((1,), torch.int32),
        )
        op = Mock(return_value=object())
        with patch.object(adapter, "_operator", return_value=op), self.assertLogs(
            adapter.logger, level="INFO"
        ) as log:
            self.assertIs(self.run_wrapper(**data), op.return_value)
        kw = op.call_args.kwargs
        for name in (
            "initial_state_source",
            "initial_state_indices",
            "cu_seqlens",
            "track_state_indices",
            "track_state_lens",
        ):
            self.assertIs(kw[name], data[name])
        self.assertEqual(kw["scale"], 128**-0.5)
        self.assertIn("dispatched", log.output[0])
        self.assertFalse(self.prepared.called or self.legacy.calls)

    def test_bad_metadata_rejected_before_new_dispatch(self):
        os.environ[adapter.ENABLE_ENV] = "1"
        with patch.object(adapter, "_operator") as op:
            for change in (
                dict(lower_bound=0),
                dict(track_state_indices=Meta((1,), torch.int32)),
                dict(b=Meta((1, 129, 4))),
                dict(a=Meta((1, 129, 8, 64))),
                dict(intermediate_state=Meta((1, 5, 4, 128, 128), torch.float32)),
            ):
                with self.subTest(change=tuple(change)), self.assertRaises(ValueError):
                    self.run_wrapper(**(inputs() | change))
            op.assert_not_called()

    def test_original_state_layout_fallback_still_uses_prepared(self):
        os.environ.update(
            SGLANG_GLM53_KDA_VARLEN_ASCENDC="1",
            SGLANG_GLM53_KDA_PREFILL_PREPARE="1",
            SGLANG_GLM53_KDA_PREPARED_ASCENDC="1",
        )
        data = inputs(dtype=torch.int32)
        data["initial_state_source"] = Meta((6, 8, 128, 128), torch.float32)
        with patch.object(adapter, "_operator") as op:
            self.run_wrapper(**data)
            op.assert_not_called()
        self.prepared.assert_called_once()

    def test_decode_verify_five_steps_and_int32_int64(self):
        os.environ[adapter.ENABLE_ENV] = "1"
        for heads in (16, 32, 64):
            for dtype in (torch.int32, torch.int64):
                for steps in (1, 4, 5):
                    data = inputs(heads, 3 * steps, 3, False, steps, dtype)
                    op = Mock()
                    with self.subTest(
                        heads=heads, dtype=dtype, steps=steps
                    ), patch.object(adapter, "_operator", return_value=op):
                        self.run_wrapper(**data)
                        self.assertIs(
                            op.call_args.kwargs["intermediate_state"],
                            data["intermediate_state"],
                        )
        self.assertFalse(self.legacy.calls)

    def test_unsupported_layouts_and_modes_retain_generic(self):
        os.environ[adapter.ENABLE_ENV] = "1"
        cases = [
            inputs(heads=16),
            inputs(batch=2),
            inputs(tokens=0),
            inputs(heads=8, prefill=False),
        ]
        data = inputs(heads=32, tokens=15, batch=3, prefill=False, steps=5)
        padded_state = Meta(
            tensor=torch.empty((8, 33, 128, 128), device="meta")[:, :32].transpose(
                -1, -2
            )
        )
        cases += [
            data | {"initial_state_source": padded_state},
            data | {"A_log": Meta((1, 1, 32, 1), device=NS(type="npu", index=1))},
            data | {"dt_bias": Meta((4096,), torch.bfloat16)},
            data | {"initial_state_indices": Meta((3,), torch.float32)},
            data | {"intermediate_state": Meta((5, 5, 32, 128, 128), torch.float32)},
        ]
        with patch.object(adapter, "_operator") as op:
            for data in cases:
                before = len(self.legacy.calls)
                self.run_wrapper(**data)
                self.assertEqual(len(self.legacy.calls), before + 1)
            op.assert_not_called()

    def test_padded_intermediate_outer_strides_supported_but_overlap_rejected(self):
        data = inputs(heads=32, tokens=15, batch=3, prefill=False, steps=5)
        backing = torch.empty((6, 7, 33, 128, 128), dtype=torch.float32, device="meta")
        data["intermediate_state"] = Meta(
            tensor=backing[1:5, :5, :32].transpose(-1, -2)
        )
        self.assertIsNone(adapter.unsupported_reason(**data))
        t = data["intermediate_state"].tensor
        data["intermediate_state"] = Meta(
            tensor=t.as_strided(t.shape, (0, *t.stride()[1:]))
        )
        self.assertEqual(adapter.unsupported_reason(**data), "intermediate_layout")

    def test_kernel_exception_not_silently_retried(self):
        os.environ[adapter.ENABLE_ENV] = "1"
        with patch.object(
            adapter,
            "_operator",
            return_value=Mock(side_effect=RuntimeError("launch failed")),
        ):
            with self.assertRaisesRegex(RuntimeError, "launch failed"):
                self.run_wrapper(**inputs())
        self.assertFalse(self.legacy.calls or self.prepared.called)


class BuildCPU(unittest.TestCase):
    def test_export_is_pinned_isolated_and_does_not_touch_checkout(self):
        source = ROOT.parent / "01-glm53-ascend-1023"
        if not source.exists():
            self.skipTest("Pinned external repository is not adjacent")
        before = subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain"]
        )
        # Build artifacts remain inside this isolated SGLang worktree.
        root = ROOT / "build/kda-contract-tests"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as td:
            dst = Path(td) / "export"
            manifest = builder.prepare(source, dst)
            self.assertEqual(manifest["kernel_commit"], adapter.KERNEL_COMMIT)
            self.assertFalse((dst / "src/sgl_glm53_ascend_ops").exists())
            self.assertFalse((dst / "csrc/kda/prepared_recurrent.cpp").exists())
            changed = [
                p
                for p, h in manifest["sources"].items()
                if h["original_sha256"] != h["build_sha256"]
            ]
            self.assertEqual(changed, ["csrc/kda/kda_varlen_recurrent.cpp"])
            host = (dst / changed[0]).read_text()
            self.assertNotIn("TORCH_LIBRARY_FRAGMENT(sgl_glm53_ascend,", host)
            self.assertIn("build_revision() -> str", host)
            self.assertIn(adapter.KERNEL_COMMIT, host)
        self.assertEqual(
            before,
            subprocess.check_output(
                ["git", "-C", str(source), "status", "--porcelain"]
            ),
        )

    def test_loader_requires_explicit_path(self):
        adapter._operator.cache_clear()
        with self.assertRaisesRegex(RuntimeError, adapter.LIB_ENV):
            adapter._operator("")

    def test_loader_checks_revision_without_importing_old_python_package(self):
        adapter._operator.cache_clear()
        self.addCleanup(adapter._operator.cache_clear)
        root = ROOT / "build/kda-contract-tests"
        root.mkdir(parents=True, exist_ok=True)
        namespace = NS(
            build_revision=Mock(return_value=adapter.KERNEL_COMMIT),
            kda_varlen_recurrent=Mock(),
        )
        with tempfile.TemporaryDirectory(dir=root) as td:
            library = Path(td) / "candidate.so"
            library.write_bytes(b"CPU fake library; never dynamically loaded")
            with patch.object(torch.ops, "load_library") as load_library, patch.object(
                torch.ops, adapter.NAMESPACE, namespace
            ), patch(
                "builtins.__import__",
                side_effect=AssertionError("unexpected package import"),
            ):
                op = adapter._operator(str(library))
                self.assertIs(op, namespace.kda_varlen_recurrent)
                self.assertIs(adapter._operator(str(library)), op)
                load_library.assert_called_once_with(str(library))
                namespace.build_revision.assert_called_once()
                adapter._operator.cache_clear()
                namespace.build_revision.return_value = "unversioned"
                with self.assertRaisesRegex(RuntimeError, "was not built from"):
                    adapter._operator(str(library))


if __name__ == "__main__":
    unittest.main()
