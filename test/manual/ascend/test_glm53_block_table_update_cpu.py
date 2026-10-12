"""CPU-only checks of the production graph metadata entry and Triton kernel.

Run with the sgl-0511 Python, CUDA_VISIBLE_DEVICES='' and TRITON_INTERPRET=1.
AST extraction avoids importing SGLang, torch_npu or any accelerator backend.
These tests establish integer/state equivalence, not NPU compilation or speed.
"""

import __future__

import ast
import importlib.util
import logging
import os
import sys
import unittest
from contextlib import nullcontext
from dataclasses import dataclass
from enum import IntEnum, auto
from itertools import product
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[3]
SRT = ROOT / "python/sglang/srt"
BACKEND = SRT / "hardware_backend/npu/attention/ascend_backend.py"
KERNEL_NAME = "sglang.srt.hardware_backend.npu.attention.glm53.block_table_update_npu"
FLAG = "SGLANG_GLM53_FUSED_BLOCK_TABLE_UPDATE"


def load_kernel():
    spec = importlib.util.spec_from_file_location(
        "_cpu_block_table_update", BACKEND.parent / "glm53/block_table_update_npu.py"
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(os.environ, {"TRITON_INTERPRET": "1"}):
        spec.loader.exec_module(module)
    return module


fused = load_kernel()


def extract(path, names, namespace, methods=None):
    nodes = []
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            if isinstance(node, ast.ClassDef) and methods is not None:
                node.bases = []
                node.body = [
                    n
                    for n in node.body
                    if isinstance(n, ast.FunctionDef) and n.name in methods
                ]
            nodes.append(node)
    assert len(nodes) == len(names), (path, names)
    exec(
        compile(
            ast.Module(body=nodes, type_ignores=[]),
            str(path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )


def load_backend():
    ns = dict(
        torch=torch,
        dataclass=dataclass,
        IntEnum=IntEnum,
        auto=auto,
        os=os,
        logger=logging.getLogger(__name__),
        _warned_bool_env_var_keys=set(),
        SpecInputType=NS(DFLASH_VERIFY="dflash"),
    )
    extract(SRT / "utils/common.py", {"get_bool_env_var"}, ns)
    extract(SRT / "model_executor/forward_batch_info.py", {"ForwardMode"}, ns)
    extract(BACKEND, {"ForwardMetadata", "_is_dflash_verify"}, ns)
    extract(
        BACKEND,
        {"AscendAttnBackend"},
        ns,
        {
            "init_cuda_graph_state",
            "_init_dsv4_graph_buffers",
            "_init_cuda_graph_metadata",
            "_apply_cuda_graph_metadata",
            "init_forward_metadata_out_graph",
        },
    )
    return ns["AscendAttnBackend"], ns["ForwardMode"]


Backend, ForwardMode = load_backend()


def reference(req, indices, table, bs, max_len, page_size):
    pages = (max_len + page_size - 1) // page_size
    table[:bs, :pages].copy_(req[indices[:bs], 0:max_len:page_size] // page_size)
    table[:bs, pages:].fill_(0)
    table[bs:, :].fill_(0)


def make_backend(use_mla=True, rope=0, hybrid=False, pool_dtype=torch.int32):
    backend = Backend()
    backend.device = "cpu"
    backend.use_mla = use_mla
    if use_mla:
        backend.qk_rope_head_dim = rope
    backend.is_hybrid_swa = hybrid
    backend.use_sliding_window_kv_pool = False
    backend.enable_sparsity_driven_kv_offload = False
    backend.page_size = 64
    backend.max_context_len = 1024
    backend.speculative_num_draft_tokens = 5
    backend.speculative_step_id = 2
    backend.speculative_step_offset_npu = torch.tensor(3, device="cpu")
    backend.q_head_num_padding = None
    backend.req_to_token = (
        torch.arange(6 * 1152, dtype=pool_dtype).reshape(6, 1152) + 193
    )
    if hybrid:
        backend.full_to_swa_index_mapping = torch.arange(8192, dtype=torch.int32)
        backend.sliding_window_size = 128
    backend.init_cuda_graph_state(max_bs=8, max_num_tokens=40)
    backend.graph_metadata["block_tables"].fill_(-777)
    return backend


def batch(bs=4, lengths=None, mode=None, spec=None):
    if lengths is None:
        lengths = [63, 1] + [0] * (bs - 2)
    return NS(
        batch_size=bs,
        forward_mode=ForwardMode.DECODE if mode is None else mode,
        req_pool_indices=torch.tensor(([3, 1] + [0] * (bs - 2))[:bs]),
        seq_lens=torch.tensor(lengths, dtype=torch.int64),
        seq_lens_cpu=torch.tensor(lengths, dtype=torch.int64),
        spec_info=spec,
        out_cache_loc=None,
    )


class BlockTableCPU(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(53)
        self.enterContext(
            patch.object(
                torch.cuda,
                "_lazy_init",
                side_effect=AssertionError("CPU isolation forbids CUDA initialization"),
            )
        )
        self.enterContext(patch.dict(os.environ, {"TRITON_INTERPRET": "1"}))
        runtime = ModuleType("sglang.srt.runtime_context")
        self.disagg = NS(disaggregation_mode="decode")
        runtime.get_disagg = Mock(return_value=self.disagg)
        self.enterContext(
            patch.dict(
                sys.modules,
                {"sglang.srt.runtime_context": runtime, KERNEL_NAME: fused},
            )
        )
        self.update = self.enterContext(
            patch.object(
                fused, "try_update_block_table_", wraps=fused.try_update_block_table_
            )
        )

    def assert_bits_equal(self, actual, expected):
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(
            torch.equal(
                actual.clone(memory_format=torch.contiguous_format).view(torch.uint8),
                expected.clone(memory_format=torch.contiguous_format).view(torch.uint8),
            )
        )

    def run_entry(self, backend, fb, flag, capture=False):
        with patch.dict(os.environ):
            if flag is None:
                os.environ.pop(FLAG, None)
            else:
                os.environ[FLAG] = flag
            backend.init_forward_metadata_out_graph(fb, in_capture=capture)

    def test_actual_pool_constructors_dtype_stride_and_reserved_row(self):
        for name, path, extra in (
            ("ReqToTokenPool", "mem_cache/memory_pool.py", {}),
            ("DecodeReqToTokenPool", "disaggregation/decode.py", {"pre_alloc_size": 2}),
        ):
            with self.subTest(pool=name):
                ns = dict(
                    torch=torch,
                    TorchMemorySaverAdapter=NS(
                        create=lambda **kw: NS(region=lambda *a, **k: nullcontext())
                    ),
                    GPU_MEMORY_TYPE_KV_CACHE="cpu-test",
                )
                extract(SRT / path, {name}, ns, {"__init__"})
                pool = ns[name](3, 131073, "cpu", False, **extra)
                self.assertEqual(pool.req_to_token.dtype, torch.int32)
                self.assertEqual(pool.req_to_token.stride(), (131073, 1))
                self.assertEqual(
                    pool.req_to_token.shape[0], 4 + extra.get("pre_alloc_size", 0)
                )
                self.assertTrue(pool.req_to_token.is_contiguous())
                self.assertNotIn(0, pool.free_slots)
                self.assertEqual(torch.count_nonzero(pool.req_to_token[0]).item(), 0)

    def test_page_boundaries_padding_signed_values_and_strides(self):
        for page_size, bs, strided in product(
            (1, 3, 64, 128), (0, 1, 3, 4), (False, True)
        ):
            context = 5 * page_size + 3
            cols = (context + page_size - 1) // page_size + 2
            # Nonzero storage offsets and both non-unit source strides.
            storage = torch.randint(
                -(2**31), 2**31 - 1, (10, 2 * context + 1), dtype=torch.int32
            )
            req = storage[1::2, 1::2] if strided else storage[:5, :context].contiguous()
            ids = torch.tensor([99, 2, 99, 0, 99, 2, 99, -1])[1::2][:bs]
            if not strided:
                ids = ids.contiguous()
            req_before, ids_before = req.clone(), ids.clone()
            for max_len in sorted(
                {0, 1, page_size - 1, page_size, page_size + 1, context}
            ):
                with self.subTest(
                    page_size=page_size, bs=bs, strided=strided, max_len=max_len
                ):
                    backing = torch.full((7, cols * 2 + 3), -777, dtype=torch.int32)
                    expected = backing.clone()
                    table = backing[1:6, 1 : 2 * cols + 1 : 2]
                    ref_table = expected[1:6, 1 : 2 * cols + 1 : 2]
                    address = table.data_ptr()
                    reference(req, ids, ref_table, bs, max_len, page_size)
                    self.assertTrue(
                        fused.try_update_block_table_(
                            req,
                            ids,
                            table,
                            bs,
                            (max_len + page_size - 1) // page_size,
                            page_size,
                        )
                    )
                    self.assert_bits_equal(backing, expected)
                    self.assertEqual(table.data_ptr(), address)
            self.assert_bits_equal(req, req_before)
            self.assert_bits_equal(ids, ids_before)

    def test_exact_int32_limits_and_negative_indexing(self):
        values = torch.tensor(
            [
                -(2**31),
                -(2**31) + 1,
                -129,
                -128,
                -127,
                -1,
                0,
                1,
                127,
                128,
                129,
                2**31 - 1,
            ],
            dtype=torch.int32,
        )
        for page_size, id_dtype in product((1, 3, 64, 128), (torch.int32, torch.int64)):
            req = values.repeat_interleave(page_size)[None].repeat(2, 1)
            ids = torch.tensor([0, -1, -2], dtype=id_dtype)
            table = torch.full((4, values.numel() + 1), -777, dtype=torch.int32)
            expected = table.clone()
            reference(req, ids, expected, 3, req.shape[1], page_size)
            self.assertTrue(
                fused.try_update_block_table_(
                    req, ids, table, 3, values.numel(), page_size
                )
            )
            self.assert_bits_equal(table, expected)

    def test_long_context_live_inputs_and_single_launch(self):
        page_size = 64
        context = 131073
        req = torch.randint(0, 2**31 - 1, (6, context), dtype=torch.int32)
        ids = torch.tensor([4, 0, 4, 1], dtype=torch.int64)
        table = torch.full((4, 2051), -777, dtype=torch.int32)
        expected = table.clone()
        addresses = (req.data_ptr(), ids.data_ptr(), table.data_ptr())
        launches = []
        kernel = fused._update_block_table_kernel

        class Recorder:
            def __getitem__(self, grid):
                launches.append(grid)
                return kernel[grid]

        with patch.object(fused, "_update_block_table_kernel", Recorder()):
            for max_len in (131071, 131072, 131073, 65, 0, 16385, 131073):
                req.add_(1)
                ids.copy_(ids.roll(1))
                reference(req, ids, expected, 4, max_len, page_size)
                self.assertTrue(
                    fused.try_update_block_table_(
                        req,
                        ids,
                        table,
                        4,
                        (max_len + page_size - 1) // page_size,
                        page_size,
                    )
                )
                self.assert_bits_equal(table, expected)
                self.assertEqual(
                    addresses, (req.data_ptr(), ids.data_ptr(), table.data_ptr())
                )
        self.assertEqual(launches, [(4, 9)] * 7)

    def test_gate_defaults_and_excluded_paths_use_original_updates(self):
        for mode, flag, config in product(
            ("decode", "prefill", "null"),
            (None, "0", "1"),
            ({}, {"use_mla": False}, {"rope": 64}, {"hybrid": True}),
        ):
            with self.subTest(mode=mode, flag=flag, config=config):
                self.disagg.disaggregation_mode = mode
                baseline, candidate = make_backend(**config), make_backend(**config)
                self.run_entry(baseline, batch(), "0", capture=True)
                self.update.reset_mock()
                self.run_entry(candidate, batch(), flag, capture=True)
                enabled = mode == "decode" and flag == "1" and not config
                self.assertEqual(self.update.call_count, int(enabled))
                self.assert_bits_equal(
                    candidate.graph_metadata["block_tables"],
                    baseline.graph_metadata["block_tables"],
                )
                if config.get("hybrid"):
                    self.assert_bits_equal(
                        candidate.forward_metadata.block_tables_swa,
                        baseline.forward_metadata.block_tables_swa,
                    )
                    self.assert_bits_equal(
                        candidate.forward_metadata.swa_mask,
                        baseline.forward_metadata.swa_mask,
                    )

    def test_public_capture_replay_spec_lengths_and_max_bs_storage(self):
        for mode, spec, increment in (
            (ForwardMode.DECODE, None, 0),
            (ForwardMode.IDLE, None, 0),
            (ForwardMode.DECODE, NS(spec_input_type="eagle"), 3),
            (ForwardMode.IDLE, NS(spec_input_type="eagle"), 3),
            (ForwardMode.TARGET_VERIFY, NS(spec_input_type="eagle"), 5),
            (ForwardMode.TARGET_VERIFY, NS(spec_input_type="dflash"), 0),
            (ForwardMode.DRAFT_EXTEND_V2, NS(spec_input_type="eagle"), 0),
        ):
            with self.subTest(mode=mode, spec=spec):
                old, new = make_backend(), make_backend()
                self.update.reset_mock()
                seen_buckets = set()
                addresses = {}
                # Large bucket -> small bucket -> large bucket, growing/shrinking lengths.
                for bs, max_len in ((4, 63), (2, 128), (4, 1), (2, 1024), (4, 0)):
                    capture = bs not in seen_buckets
                    untouched_rows = new.graph_metadata["block_tables"][bs:].clone()
                    for backend, flag in ((old, "0"), (new, "1")):
                        backend.req_to_token.add_(64)
                        fb = batch(bs, [max_len] + [0] * (bs - 1), mode, spec)
                        self.run_entry(backend, fb, flag, capture=capture)
                    seen_buckets.add(bs)
                    actual = new.forward_metadata
                    self.assertIs(actual, new.graph_metadata[bs])
                    self.assertTrue(new.graph_mode)
                    self.assert_bits_equal(
                        new.graph_metadata["block_tables"],
                        old.graph_metadata["block_tables"],
                    )
                    self.assert_bits_equal(
                        actual.seq_lens, old.forward_metadata.seq_lens
                    )
                    self.assertEqual(
                        actual.seq_lens_cpu_list, old.forward_metadata.seq_lens_cpu_list
                    )
                    self.assert_bits_equal(
                        new.graph_metadata["block_tables"][bs:], untouched_rows
                    )
                    self.assertEqual(actual.block_tables.shape[0], bs)
                    self.assertEqual(actual.block_tables[bs:].numel(), 0)
                    self.assertEqual(
                        actual.block_tables.data_ptr(),
                        new.graph_metadata["block_tables"].data_ptr(),
                    )
                    address = addresses.setdefault(bs, actual.block_tables.data_ptr())
                    self.assertEqual(actual.block_tables.data_ptr(), address)
                    pages = (max_len + increment + 63) // 64
                    self.assertEqual(self.update.call_args.args[4], pages)
                    self.assertEqual(
                        torch.count_nonzero(actual.block_tables[:, pages:]).item(), 0
                    )
                    if bs == 4 and pages:
                        self.assert_bits_equal(
                            actual.block_tables[2, :pages],
                            new.req_to_token[0, 0 : max_len + increment : 64] // 64,
                        )
                self.assertEqual(self.update.call_count, 5)

    def test_hybrid_wrapper_reaches_full_attention_update(self):
        # GLM's Ascend KDA hybrid inherits this public metadata dispatcher.
        ns = {}
        extract(
            SRT / "layers/attention/hybrid_linear_attn_backend.py",
            {"HybridLinearAttnBackend"},
            ns,
            {"init_forward_metadata_out_graph"},
        )
        wrapper = ns["HybridLinearAttnBackend"]()
        backend = make_backend()
        linear = NS(init_forward_metadata_out_graph=Mock())
        wrapper.attn_backend_list = [backend, linear]
        with patch.object(
            backend,
            "_apply_cuda_graph_metadata",
            wraps=backend._apply_cuda_graph_metadata,
        ) as apply:
            self.run_entry(wrapper, batch(), "1", capture=True)
            self.run_entry(wrapper, batch(lengths=[129, 65, 0, 0]), "1")
            self.assertEqual(apply.call_count, 2)
        self.assertEqual(self.update.call_count, 2)
        self.assertEqual(linear.init_forward_metadata_out_graph.call_count, 2)
        expected = backend.graph_metadata["block_tables"].clone()
        reference(
            backend.req_to_token, batch().req_pool_indices, expected[:4], 4, 129, 64
        )
        self.assert_bits_equal(backend.graph_metadata["block_tables"], expected)

    def test_unsupported_dtype_falls_back_without_writing(self):
        old, new = make_backend(pool_dtype=torch.int64), make_backend(
            pool_dtype=torch.int64
        )
        table = new.graph_metadata["block_tables"]
        before = table.clone()
        self.assertFalse(
            fused.try_update_block_table_(
                new.req_to_token, batch().req_pool_indices, table, 4, 1, 64
            )
        )
        self.assert_bits_equal(table, before)
        self.run_entry(old, batch(), "0", capture=True)
        self.run_entry(new, batch(), "1", capture=True)
        self.assert_bits_equal(table, old.graph_metadata["block_tables"])

    def test_out_of_capacity_returns_false_before_launch(self):
        req = torch.zeros((2, 65), dtype=torch.int32)
        ids = torch.tensor([0, 1])
        table = torch.full((2, 4), -777, dtype=torch.int32)
        for bs, pages, page_size in (
            (3, 1, 64),
            (2, 3, 64),
            (2, 5, 64),
            (2, -1, 64),
            (2, 1, 0),
        ):
            with self.subTest(bs=bs, pages=pages, page_size=page_size):
                self.assertFalse(
                    fused.try_update_block_table_(req, ids, table, bs, pages, page_size)
                )
                self.assertTrue(torch.all(table == -777).item())

    def test_opt_in_live_comparison_detects_mismatches(self):
        req = torch.arange(3 * 129, dtype=torch.int32).reshape(3, 129)
        ids = torch.tensor([2, 0])
        table = torch.full((2, 5), -777, dtype=torch.int32)
        with patch.dict(os.environ, {"SGLANG_GLM53_CHECK_BLOCK_TABLE_UPDATE": "1"}):
            self.assertTrue(fused.try_update_block_table_(req, ids, table, 2, 3, 64))
            expected = torch.zeros_like(table)
            reference(req, ids, expected, 2, 129, 64)
            self.assert_bits_equal(table, expected)
            with patch.object(fused.torch, "equal", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "differs from torch"):
                    fused.try_update_block_table_(req, ids, table, 2, 3, 64)

    def test_graph_runner_call_sites_are_present(self):
        # Read production routing, without importing runners or touching NPU APIs.
        for path, cls, methods in (
            (
                SRT / "model_executor/runner/decode_cuda_graph_runner.py",
                "DecodeCudaGraphRunner",
                ("capture_one_shape", "load_batch"),
            ),
            (
                SRT / "hardware_backend/npu/graph_runner/npu_graph_runner.py",
                "NPUGraphRunner",
                ("execute",),
            ),
        ):
            tree = ast.parse(path.read_text())
            klass = next(
                n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls
            )
            for name in methods:
                method = next(
                    n
                    for n in klass.body
                    if isinstance(n, ast.FunctionDef) and n.name == name
                )
                self.assertTrue(
                    any(
                        isinstance(n, ast.Call)
                        and isinstance(n.func, ast.Attribute)
                        and n.func.attr == "init_forward_metadata_out_graph"
                        for n in ast.walk(method)
                    ),
                    (path, name),
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
