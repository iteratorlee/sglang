"""CPU/source tests for GLM53 PD metadata and byte-address remapping.

Run with: python test/manual/ascend/test_glm53_pd_state_cpu.py -v

The local development host has no torch/torch_npu. Execute the actual selected
source functions with small NumPy-backed tensor/pool stand-ins; only framework
imports are stubbed. uint16 represents BF16 *storage*, not BF16 arithmetic.
These tests do not validate NPU kernels, graph replay, SDMA or model accuracy.
"""

import ast
import concurrent.futures
import math
import ctypes
import importlib.util
import logging
import sys
import types
import unittest
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[3]
SRT = ROOT / "python/sglang/srt"


def source_functions(path, names, namespace, class_name=None):
    tree = ast.parse(path.read_text())
    nodes = tree.body
    if class_name is not None:
        nodes = next(
            n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name
        ).body
    found = [n for n in nodes if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in found} != set(names):
        raise AssertionError(f"Missing source function in {path}: {names}")
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *found,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return {name: namespace[name] for name in names}


class Tensor:
    def __init__(self, array):
        self.array = array
        self.device = "cpu"

    @property
    def shape(self):
        return self.array.shape

    @property
    def nbytes(self):
        return self.array.nbytes

    def data_ptr(self):
        return self.array.ctypes.data

    def is_contiguous(self):
        return self.array.flags.c_contiguous

    def new_zeros(self, shape):
        return Tensor(np.zeros(shape, dtype=self.array.dtype))

    def __getitem__(self, item):
        return Tensor(self.array[item])

    def copy_(self, other):
        np.copyto(self.array, other.array)

    def transpose(self, axis1, axis2):
        return Tensor(self.array.swapaxes(axis1, axis2))

    def stride(self):
        return tuple(s // self.array.itemsize for s in self.array.strides)

    def numel(self):
        return self.array.size


class NPUPool:
    def __init__(self, layers, indexed=None, packed=False, scales=False):
        self.start_layer = 0
        self.layer_num = layers
        self.index_head_dim = 128
        self.indexer_layer_ids = list(range(layers)) if indexed is None else indexed
        self.dsa_kv_cache_store_fp8 = packed
        self.k_buffer = [
            Tensor(np.zeros((9, 4, 5), dtype=np.uint16)) for _ in range(layers)
        ]
        self.v_buffer = [
            Tensor(np.zeros((9, 4, 2), dtype=np.uint16)) for _ in range(layers)
        ]
        self.index_k_buffer = [
            Tensor(np.zeros((9, 4, 3), dtype=np.uint16)) for _ in self.indexer_layer_ids
        ]
        self.index_k_scale_buffer = (
            [
                Tensor(np.zeros((9, 4, 1), dtype=np.float32))
                for _ in self.indexer_layer_ids
            ]
            if scales
            else None
        )

    def _raise_if_native_kv_cache_disabled(self):
        pass


class HybridPool:
    use_dsa = True
    use_mla = True

    def __init__(self, pool, global_ids):
        self.full_kv_pool = pool
        self.full_attention_layer_id_mapping = {
            lid: i for i, lid in enumerate(global_ids)
        }

    def get_state_buf_infos(self):
        return [101, 102], [1000, 2000], [100, 200]

    def get_state_layer_ids(self):
        return [0, 0]


for cls, file, class_name, names in (
    (
        NPUPool,
        SRT / "hardware_backend/npu/memory_pool_npu.py",
        "NPUMLATokenToKVPool",
        [
            "get_kv_layer_ids",
            "get_state_layer_ids",
            "get_compress_tail_buf_infos",
            "get_state_buf_infos",
            "get_contiguous_buf_infos",
        ],
    ),
    (
        HybridPool,
        SRT / "mem_cache/memory_pool.py",
        "HybridLinearKVPool",
        ["get_kv_layer_ids"],
    ),
):
    for name, fn in source_functions(
        file, names, {"_is_npu": True}, class_name
    ).items():
        setattr(cls, name, fn)

spec = importlib.util.spec_from_file_location(
    "glm53_pd_state_under_test",
    SRT / "hardware_backend/npu/attention/glm53/pd_state.py",
)
pd_state = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pd_state)

UTILS = {}
source_functions(
    SRT / "disaggregation/utils.py",
    [
        "setup_state_kv_args",
        "append_state_component",
        "build_kv_layer_ids",
        "_draft_entry_layer_ids",
        "get_dsa_tail_state_indices",
        "build_dsa_tail_transfer_blocks",
        "build_transfer_entry_pairs",
        "compute_mamba_state_slice_byte_blocks",
        "compute_mamba_state_slice_blocks",
        "should_send_replicated_state",
    ],
    UTILS,
)
UTILS.update(
    is_npu=lambda: False,
    is_mla_backend=lambda pool: isinstance(pool, NPUPool),
    deque=deque,
)

@dataclass(frozen=True)
class MambaState:
    conv: list
    temporal: Tensor


class MambaPool:
    _slot_siblings = ()
    _NON_TRANSFER_STATE_FIELDS = {"intermediate_ssm", "intermediate_conv_window"}
    conv_slice_axis = 1
    mamba_layer_ids = [0, 2]
    mem_usage = 0

    def __init__(self, rows, window, transpose, conv_channels=6, temporal_heads=2):
        temporal = Tensor(
            np.zeros((2, rows, temporal_heads, 5, 7), dtype=np.float32)
        )
        self.mamba_cache = MambaState(
            [Tensor(np.zeros((2, rows, window, conv_channels), dtype=np.uint16))],
            temporal.transpose(-1, -2) if transpose else temporal,
        )


for name, fn in source_functions(
    SRT / "mem_cache/memory_pool.py",
    [
        "_iter_transfer_state_entries",
        "get_contiguous_buf_infos",
        "get_state_layer_ids",
        "get_state_dim_per_tensor",
        "get_state_slice_outer_counts",
    ],
    {"math": math},
    "MambaPool",
).items():
    setattr(MambaPool, name, fn)

MAMBA_SEND = source_functions(
    SRT / "disaggregation/mooncake/conn.py",
    ["_send_mamba_state"],
    UTILS,
    "MooncakeKVManager",
)["_send_mamba_state"]

MAMBA_SLICE_SEND = source_functions(
    SRT / "disaggregation/mooncake/conn.py",
    ["_send_mamba_state_slice"],
    UTILS | {"logger": types.SimpleNamespace(warning_once=lambda *args: None)},
    "MooncakeKVManager",
)["_send_mamba_state_slice"]

ASCEND_KV_SEND = source_functions(
    SRT / "disaggregation/ascend/conn.py",
    ["send_kvcache"],
    {
        "group_concurrent_contiguous": source_functions(
            SRT / "disaggregation/common/utils.py",
            ["group_concurrent_contiguous"],
            {"np": np},
        )["group_concurrent_contiguous"],
        "concurrent": concurrent,
    },
    "AscendKVManager",
)["send_kvcache"]


def fake_module(**attrs):
    module = types.ModuleType("test_stub")
    module.__dict__.update(attrs)
    return module


def unused(name):
    return type(name, (), {})


StateType = types.SimpleNamespace(
    **{
        name: name
        for name in [
            "MAMBA",
            "DSA",
            "DSA_TAIL",
            "BLOCK_SCALE",
            "SWA",
            "MINIMAX_INDEX_K",
        ]
    }
)
MODULES = {
    "sglang.srt.disaggregation.base.conn": fake_module(StateType=StateType),
    "sglang.srt.hardware_backend.npu.memory_pool_npu": fake_module(
        NPUMLATokenToKVPool=NPUPool
    ),
    "sglang.srt.hardware_backend.npu.attention.glm53.pd_state": pd_state,
    "sglang.srt.mem_cache.memory_pool": fake_module(
        HybridLinearKVPool=HybridPool,
        DSATokenToKVPool=unused("DSA"),
        MHATokenToKVPoolMXFP8=unused("MXFP8"),
        MiniMaxSparseKVPool=unused("MiniMax"),
    ),
    "sglang.srt.mem_cache.base_swa_memory_pool": fake_module(
        BaseSWAKVPool=unused("BaseSWA")
    ),
    "sglang.srt.mem_cache.deepseek_v4_memory_pool": fake_module(
        DeepSeekV4TokenToKVPool=unused("DSV4")
    ),
    "sglang.srt.mem_cache.qsa_kv_pool": fake_module(QSATokenToKVPool=unused("QSA")),
    "sglang.srt.mem_cache.swa_memory_pool": fake_module(SWAKVPool=unused("SWA")),
}
PARALLEL = types.SimpleNamespace(
    tp_size=16,
    nnodes=1,
    moe_ep_size=16,
    attn_tp_size=16,
    pp_size=1,
    attn_cp_size=1,
    enable_dp_attention=False,
    dp_size=1,
    attn_dp_size=1,
)


def model(layer_ids, rows=3):
    indexers = [
        types.SimpleNamespace(
            layer_id=lid,
            index_kpool=4,
            head_dim=128,
            _kpool_tail_k=Tensor(np.full((rows, 4, 128), lid + 1, dtype=np.uint16)),
            _kpool_tail_score=Tensor(
                np.full((rows, 4, 128), lid + 0.25, dtype=np.float32)
            ),
            _kpool_mtp_tail_k=object(),
            _kpool_mtp_tail_score=object(),
        )
        for lid in layer_ids
    ]
    return types.SimpleNamespace(modules=lambda: iter(reversed(indexers))), indexers


def req_pool(rows):
    return types.SimpleNamespace(
        req_to_token=types.SimpleNamespace(shape=(rows, 132160))
    )


def make_registered(rows, parallel=PARALLEL):
    requests = req_pool(rows)
    target = HybridPool(NPUPool(2), [3, 11])
    draft = NPUPool(1)
    target_model, target_indexers = model([3, 11])
    draft_model, draft_indexers = model([0])
    pd_state.register_glm53_kpool_state(
        target_model, target, requests, parallel=parallel
    )
    pd_state.register_glm53_kpool_state(draft_model, draft, requests, parallel=parallel)
    return target, draft, requests, target_indexers + draft_indexers


class TestGlm53PDState(unittest.TestCase):
    def setUp(self):
        self.patcher = patch.dict(sys.modules, MODULES)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_decode_extra_slots_preserve_state_and_scratch(self):
        m, indexers = model([3, 11])
        pool = HybridPool(NPUPool(2), [3, 11])
        snapshots = [
            (
                i._kpool_tail_k.array.copy(),
                i._kpool_tail_score.array.copy(),
                i._kpool_mtp_tail_k,
            )
            for i in indexers
        ]
        pd_state.register_glm53_kpool_state(m, pool, req_pool(49), parallel=PARALLEL)
        for i, (keys, scores, scratch) in zip(indexers, snapshots):
            self.assertEqual(i._kpool_tail_k.shape, (49, 4, 128))
            np.testing.assert_array_equal(i._kpool_tail_k.array[:3], keys)
            np.testing.assert_array_equal(i._kpool_tail_score.array[:3], scores)
            self.assertFalse(i._kpool_tail_k.array[3:].any())
            self.assertFalse(i._kpool_tail_score.array[3:].any())
            self.assertIs(i._kpool_mtp_tail_k, scratch)
        self.assertEqual(pool.full_kv_pool.tail_extra_slots, 0)

    def test_dp_attention_preserves_registered_request_state(self):
        for dp, attn_tp in ((16, 1), (8, 2), (4, 4), (2, 8)):
            with self.subTest(dp=dp, attn_tp=attn_tp):
                parallel = types.SimpleNamespace(**(vars(PARALLEL) | dict(
                    enable_dp_attention=True, dp_size=dp,
                    attn_dp_size=dp, attn_tp_size=attn_tp,
                )))
                m, indexers = model([0])
                pool, requests = NPUPool(1), req_pool(9)
                keys = indexers[0]._kpool_tail_k.array.copy()
                scores = indexers[0]._kpool_tail_score.array.copy()
                pd_state.register_glm53_kpool_state(m, pool, requests, parallel=parallel)
                np.testing.assert_array_equal(indexers[0]._kpool_tail_k.array[:3], keys)
                np.testing.assert_array_equal(indexers[0]._kpool_tail_score.array[:3], scores)
                self.assertEqual(indexers[0]._kpool_tail_k.shape, (9, 4, 128))
                ptrs = pool.get_compress_tail_buf_infos()[0]
                pd_state.register_glm53_kpool_state(m, pool, requests, parallel=parallel)
                self.assertEqual(ptrs, pool.get_compress_tail_buf_infos()[0])

    def test_910b_tp8_dp4tp2_and_dp2tp4_preserve_request_state(self):
        for dp, attn_tp, enabled in ((1, 8, False), (4, 2, True), (2, 4, True)):
            with self.subTest(dp=dp, attn_tp=attn_tp):
                parallel = types.SimpleNamespace(**(vars(PARALLEL) | dict(
                    tp_size=8, moe_ep_size=8, enable_dp_attention=enabled,
                    dp_size=dp, attn_dp_size=dp, attn_tp_size=attn_tp,
                )))
                m, indexers = model([0])
                pool, requests = NPUPool(1), req_pool(9)
                keys = indexers[0]._kpool_tail_k.array.copy()
                scores = indexers[0]._kpool_tail_score.array.copy()
                pd_state.register_glm53_kpool_state(m, pool, requests, parallel=parallel)
                np.testing.assert_array_equal(indexers[0]._kpool_tail_k.array[:3], keys)
                np.testing.assert_array_equal(indexers[0]._kpool_tail_score.array[:3], scores)
                self.assertEqual(indexers[0]._kpool_tail_k.shape, (9, 4, 128))

    def test_910b_dp4tp2_to_dp4tp2_uses_each_prefill_domain_and_tp_rank(self):
        rank_mapping = source_functions(
            SRT / "disaggregation/common/conn.py",
            ["_resolve_rank_mapping"],
            {},
            "CommonKVManager",
        )["_resolve_rank_mapping"]
        resolve_prefill_dp = source_functions(
            SRT / "disaggregation/decode.py",
            ["_resolve_prefill_dp_rank"],
            {"_bootstrap_addr": lambda req: req.bootstrap_addr},
            "DecodePreallocQueue",
        )["_resolve_prefill_dp_rank"]
        skip_policy = source_functions(
            SRT / "disaggregation/mooncake/conn.py",
            ["_get_dsa_cache_transfer_skip_flags"],
            {},
            "MooncakeKVManager",
        )["_get_dsa_cache_transfer_skip_flags"]
        prefill_info = types.SimpleNamespace(dp_size=4, attn_tp_size=2)
        queue = types.SimpleNamespace(
            kv_manager=types.SimpleNamespace(prefill_info_table={"P": prefill_info})
        )
        for dp_rank in range(4):
            req = types.SimpleNamespace(
                bootstrap_addr="P", disagg_prefill_dp_rank=dp_rank
            )
            self.assertEqual(resolve_prefill_dp(queue, req), dp_rank)
            for local_tp_rank in range(2):
                with self.subTest(dp_rank=dp_rank, local_tp_rank=local_tp_rank):
                    manager = types.SimpleNamespace(
                        attn_tp_size=2,
                        kv_args=types.SimpleNamespace(engine_rank=local_tp_rank),
                        is_mla_backend=False,
                        is_hybrid_mla_backend=True,
                        attn_cp_size=1,
                        attn_cp_rank=0,
                        pp_size=1,
                        pp_rank=0,
                    )
                    info = types.SimpleNamespace(
                        attn_tp_size=2, attn_cp_size=1, pp_size=1
                    )
                    rank_mapping(manager, info)
                    self.assertEqual(info.target_tp_ranks, [local_tp_rank])
                    self.assertEqual(info.required_dst_info_num, 1)
                    self.assertEqual(info.required_prefill_response_num, 1)
                    p_manager = types.SimpleNamespace(
                        attn_tp_size=2,
                        kv_args=types.SimpleNamespace(
                            engine_rank=dp_rank * 2 + local_tp_rank
                        ),
                        is_mla_backend=False,
                        is_hybrid_mla_backend=True,
                        _should_skip_cp_replicated_state_transfer=lambda: False,
                    )
                    self.assertEqual(
                        skip_policy(
                            p_manager, types.SimpleNamespace(dst_attn_tp_size=2)
                        ),
                        (False, False),
                    )

    def test_910b_dp4tp2_to_dp4tp2_target_draft_and_mamba_bytes(self):
        parallel = types.SimpleNamespace(**(vars(PARALLEL) | dict(
            tp_size=8, moe_ep_size=8, enable_dp_attention=True,
            dp_size=4, attn_dp_size=4, attn_tp_size=2,
        )))

        def copy_blocks(session, blocks):
            for source, dest, size in blocks:
                ctypes.memmove(dest, source, size)
            return 0

        for dp_rank in range(4):
            for tp_rank in range(2):
                with self.subTest(dp_rank=dp_rank, tp_rank=tp_rank):
                    marker = 100 * (dp_rank * 2 + tp_rank + 1)
                    src_kv, src_draft = NPUPool(2), NPUPool(1)
                    dst_kv, dst_draft = NPUPool(2), NPUPool(1)
                    src_buffers = (
                        src_kv.k_buffer + src_kv.v_buffer + src_kv.index_k_buffer
                        + src_draft.k_buffer + src_draft.v_buffer
                        + src_draft.index_k_buffer
                    )
                    dst_buffers = (
                        dst_kv.k_buffer + dst_kv.v_buffer + dst_kv.index_k_buffer
                        + dst_draft.k_buffer + dst_draft.v_buffer
                        + dst_draft.index_k_buffer
                    )
                    for entry, (source, dest) in enumerate(zip(src_buffers, dst_buffers)):
                        source.array[2] = marker + entry
                        dest.array.fill(7)
                    src_ptrs, _, src_lens = src_kv.get_contiguous_buf_infos()
                    draft_ptrs, _, draft_lens = src_draft.get_contiguous_buf_infos()
                    dst_ptrs = (
                        dst_kv.get_contiguous_buf_infos()[0]
                        + dst_draft.get_contiguous_buf_infos()[0]
                    )
                    sender = types.SimpleNamespace(
                        pp_size=1,
                        enable_custom_mem_pool=False,
                        kv_args=types.SimpleNamespace(
                            kv_data_ptrs=src_ptrs + draft_ptrs,
                            kv_item_lens=src_lens + draft_lens,
                        ),
                        _validate_envelope_kv_layout=lambda *args: None,
                        _transfer_data=copy_blocks,
                    )
                    self.assertEqual(
                        ASCEND_KV_SEND(
                            sender, "equal-tp", np.array([2], np.int32), dst_ptrs,
                            np.array([7], np.int32), None, dst_attn_tp_size=2,
                        ),
                        0,
                    )
                    for source, dest in zip(src_buffers, dst_buffers):
                        np.testing.assert_array_equal(dest.array[7], source.array[2])
                        self.assertTrue((dest.array[:7] == 7).all())
                        self.assertTrue((dest.array[8:] == 7).all())

                    src, src_nextn, _, src_indexers = make_registered(5, parallel)
                    dst, dst_nextn, _, dst_indexers = make_registered(9, parallel)
                    src_args, dst_args = types.SimpleNamespace(), types.SimpleNamespace()
                    UTILS["setup_state_kv_args"](src_args, src, src_nextn)
                    UTILS["setup_state_kv_args"](dst_args, dst, dst_nextn)
                    self.assertEqual(src_args.state_types, ["MAMBA", "DSA_TAIL"])
                    self.assertEqual(src_args.state_types, dst_args.state_types)
                    self.assertEqual(src_args.state_item_lens, dst_args.state_item_lens)
                    src_tail_ptrs, _, src_tail_lens = pd_state.combined_kpool_tail_infos(
                        src.full_kv_pool, src_nextn
                    )
                    dst_tail_ptrs, _, dst_tail_lens = pd_state.combined_kpool_tail_infos(
                        dst.full_kv_pool, dst_nextn
                    )
                    for entry, (source, dest) in enumerate(zip(src_indexers, dst_indexers)):
                        for state_kind, name in enumerate(("_kpool_tail_k", "_kpool_tail_score")):
                            getattr(source, name).array[2] = marker + 10 * entry + state_kind
                            getattr(dest, name).array.fill(7)
                    for src_addr, dst_addr, size in UTILS["build_dsa_tail_transfer_blocks"](
                        src_tail_ptrs, src_tail_lens, dst_tail_ptrs,
                        UTILS["get_dsa_tail_state_indices"](src, 2, 131075),
                        UTILS["get_dsa_tail_state_indices"](dst, 7, 131075),
                        dst_tail_lens,
                    ):
                        ctypes.memmove(dst_addr, src_addr, size)
                    for source, dest in zip(src_indexers, dst_indexers):
                        for name in ("_kpool_tail_k", "_kpool_tail_score"):
                            expected = np.full_like(getattr(dest, name).array, 7)
                            expected[7, :3] = getattr(source, name).array[2, :3]
                            np.testing.assert_array_equal(getattr(dest, name).array, expected)

                    src_mamba, dst_mamba = MambaPool(5, 3, False), MambaPool(9, 6, True)
                    pd_state.prepare_glm53_pd_mamba_state(
                        types.SimpleNamespace(mamba_pool=src_mamba),
                        types.SimpleNamespace(mamba_pool=src_mamba),
                        mode="prefill", draft_tokens=4,
                    )
                    src_mamba.mamba_cache.conv[0].array[:, 2] = (
                        marker + np.arange(2)[:, None, None] * 10
                        + np.arange(6 * 6).reshape(6, 6)
                    )
                    src_mamba.mamba_cache.temporal.array[:, 2] = (
                        marker + 100 + np.arange(2)[:, None, None, None] * 10
                        + np.arange(2 * 7 * 5).reshape(2, 7, 5)
                    )
                    src_state_ptrs, _, src_state_lens = src_mamba.get_contiguous_buf_infos()
                    dst_state_ptrs, _, dst_state_lens = dst_mamba.get_contiguous_buf_infos()
                    self.assertEqual(src_state_lens, dst_state_lens)
                    self.assertEqual(
                        MAMBA_SEND(
                            types.SimpleNamespace(pp_size=1, _transfer_data=copy_blocks),
                            types.SimpleNamespace(mooncake_session_id="equal-tp"),
                            [2], src_state_ptrs, src_state_lens, dst_state_ptrs, [7],
                            src_mamba.get_state_layer_ids(),
                            dst_mamba.get_state_layer_ids(),
                        ),
                        0,
                    )
                    np.testing.assert_array_equal(
                        dst_mamba.mamba_cache.conv[0].array[:, 7],
                        src_mamba.mamba_cache.conv[0].array[:, 2],
                    )
                    np.testing.assert_array_equal(
                        dst_mamba.mamba_cache.temporal.array[:, 7],
                        src_mamba.mamba_cache.temporal.array[:, 2],
                    )
                    self.assertFalse(dst_mamba.mamba_cache.conv[0].array[:, :7].any())
                    self.assertFalse(dst_mamba.mamba_cache.temporal.array[:, :7].any())

    def test_910b_tp4_to_tp2_rank_mapping_and_kv_draft_pages(self):
        rank_mapping = source_functions(
            SRT / "disaggregation/common/conn.py",
            ["_resolve_rank_mapping"],
            {},
            "CommonKVManager",
        )["_resolve_rank_mapping"]
        skip_policy = source_functions(
            SRT / "disaggregation/mooncake/conn.py",
            ["_get_dsa_cache_transfer_skip_flags"],
            {},
            "MooncakeKVManager",
        )["_get_dsa_cache_transfer_skip_flags"]
        for d_rank, expected_p_ranks in ((0, [0, 1]), (1, [2, 3])):
            manager = types.SimpleNamespace(
                attn_tp_size=2,
                kv_args=types.SimpleNamespace(engine_rank=d_rank),
                is_mla_backend=False,
                is_hybrid_mla_backend=True,
                attn_cp_size=1,
                attn_cp_rank=0,
                pp_size=1,
                pp_rank=0,
            )
            info = types.SimpleNamespace(attn_tp_size=4, attn_cp_size=1, pp_size=1)
            rank_mapping(manager, info)
            self.assertEqual(info.target_tp_ranks, expected_p_ranks)
            self.assertEqual(info.required_prefill_response_num, 2)
            self.assertEqual(info.required_dst_info_num, 1)
            for rank, expected_skip in zip(expected_p_ranks, (False, True)):
                p_manager = types.SimpleNamespace(
                    attn_tp_size=4,
                    kv_args=types.SimpleNamespace(engine_rank=rank),
                    is_mla_backend=False,
                    is_hybrid_mla_backend=True,
                    _should_skip_cp_replicated_state_transfer=lambda: False,
                )
                self.assertEqual(
                    skip_policy(
                        p_manager, types.SimpleNamespace(dst_attn_tp_size=2)
                    ),
                    (expected_skip, False),
                )

        # The elected P rank sends every full-attention and draft KV entry by
        # page id; TP changes do not change the NPU MLA per-page item geometry.
        source_target, source_draft = NPUPool(2), NPUPool(1)
        dest_target, dest_draft = NPUPool(2), NPUPool(1)
        source_buffers = (
            source_target.k_buffer + source_target.v_buffer
            + source_target.index_k_buffer + source_draft.k_buffer
            + source_draft.v_buffer + source_draft.index_k_buffer
        )
        dest_buffers = (
            dest_target.k_buffer + dest_target.v_buffer
            + dest_target.index_k_buffer + dest_draft.k_buffer
            + dest_draft.v_buffer + dest_draft.index_k_buffer
        )
        for entry, (source, dest) in enumerate(zip(source_buffers, dest_buffers)):
            source.array[2] = (
                100 * (entry + 1)
                + np.arange(source.array[2].size).reshape(source.array[2].shape)
            )
            dest.array.fill(7)
        src_ptrs, _, src_lens = source_target.get_contiguous_buf_infos()
        draft_ptrs, _, draft_lens = source_draft.get_contiguous_buf_infos()
        dst_ptrs = (
            dest_target.get_contiguous_buf_infos()[0]
            + dest_draft.get_contiguous_buf_infos()[0]
        )

        def copy_blocks(session, blocks):
            for source, dest, size in blocks:
                ctypes.memmove(dest, source, size)
            return 0

        sender = types.SimpleNamespace(
            pp_size=1,
            enable_custom_mem_pool=False,
            kv_args=types.SimpleNamespace(
                kv_data_ptrs=src_ptrs + draft_ptrs,
                kv_item_lens=src_lens + draft_lens,
            ),
            _validate_envelope_kv_layout=lambda *args: None,
            _transfer_data=copy_blocks,
        )
        self.assertEqual(
            ASCEND_KV_SEND(
                sender, "tp4-to-tp2", np.array([2], dtype=np.int32), dst_ptrs,
                np.array([7], dtype=np.int32), None, dst_attn_tp_size=2,
            ),
            0,
        )
        for source, dest in zip(source_buffers, dest_buffers):
            np.testing.assert_array_equal(dest.array[7], source.array[2])
            self.assertTrue((dest.array[:7] == 7).all())
            self.assertTrue((dest.array[8:] == 7).all())

    def test_910b_tp4_to_tp2_target_and_draft_kpool_tail(self):
        p_parallel = types.SimpleNamespace(**(vars(PARALLEL) | dict(
            tp_size=8, moe_ep_size=8, enable_dp_attention=True,
            dp_size=2, attn_dp_size=2, attn_tp_size=4,
        )))
        d_parallel = types.SimpleNamespace(**(vars(PARALLEL) | dict(
            tp_size=8, moe_ep_size=8, enable_dp_attention=True,
            dp_size=4, attn_dp_size=4, attn_tp_size=2,
        )))
        source, source_draft, _, source_buffers = make_registered(5, p_parallel)
        dest, dest_draft, _, dest_buffers = make_registered(9, d_parallel)
        source_ptrs, _, source_lens = pd_state.combined_kpool_tail_infos(
            source.full_kv_pool, source_draft
        )
        dest_ptrs, _, dest_lens = pd_state.combined_kpool_tail_infos(
            dest.full_kv_pool, dest_draft
        )
        for entry, (p, d) in enumerate(zip(source_buffers, dest_buffers)):
            for name in ("_kpool_tail_k", "_kpool_tail_score"):
                getattr(p, name).array[2] = entry + 1
                getattr(d, name).array.fill(7)
        # Both P writers hold replicated KPool tails; their writes must be
        # byte-identical, including the draft's independently registered tail.
        for _ in range(2):
            blocks = UTILS["build_dsa_tail_transfer_blocks"](
                source_ptrs, source_lens, dest_ptrs,
                UTILS["get_dsa_tail_state_indices"](source, 2, 131075),
                UTILS["get_dsa_tail_state_indices"](dest, 7, 131075),
                dest_lens,
            )
            for source_addr, dest_addr, size in blocks:
                ctypes.memmove(dest_addr, source_addr, size)
        for entry, (p, d) in enumerate(zip(source_buffers, dest_buffers)):
            for name in ("_kpool_tail_k", "_kpool_tail_score"):
                expected = np.full_like(getattr(d, name).array, 7)
                expected[7, :3] = getattr(p, name).array[2, :3]
                np.testing.assert_array_equal(getattr(d, name).array, expected)

    def test_registration_idempotent_and_replacement_rejected(self):
        m, indexers = model([0], rows=10)
        pool, requests = NPUPool(1), req_pool(9)
        pd_state.register_glm53_kpool_state(m, pool, requests, parallel=PARALLEL)
        ptrs = pool.get_compress_tail_buf_infos()[0]
        pd_state.register_glm53_kpool_state(m, pool, requests, parallel=PARALLEL)
        self.assertEqual(ptrs, pool.get_compress_tail_buf_infos()[0])
        indexers[0]._kpool_tail_k = indexers[0]._kpool_tail_k.new_zeros((10, 4, 128))
        with self.assertRaisesRegex(RuntimeError, "after registration"):
            pd_state.register_glm53_kpool_state(m, pool, requests, parallel=PARALLEL)

    def test_target_and_draft_registered_once_without_duplicate_index(self):
        target, draft, requests, _ = make_registered(49)
        args = types.SimpleNamespace()
        UTILS["setup_state_kv_args"](args, target, draft, req_to_token_pool=requests)
        self.assertEqual(args.state_types, ["MAMBA", "DSA_TAIL"])
        self.assertTrue(args.is_hybrid_mla_backend)
        self.assertEqual(args.state_data_ptrs[0], [101, 102])
        self.assertEqual(len(args.state_data_ptrs[1]), 6)
        self.assertEqual(args.state_item_lens[1], [1024, 1024, 2048, 2048, 1024, 2048])
        kv_ptrs = (
            target.full_kv_pool.get_contiguous_buf_infos()[0]
            + draft.get_contiguous_buf_infos()[0]
        )
        self.assertFalse(set(kv_ptrs) & set(args.state_data_ptrs[1]))

    def test_missing_draft_or_mismatched_request_pool_fails(self):
        target, draft, _, _ = make_registered(49)
        with self.assertRaisesRegex(ValueError, "draft KPool state was not registered"):
            pd_state.combined_kpool_tail_infos(target.full_kv_pool, NPUPool(1))
        draft._glm53_kpool_req_pool = req_pool(49)
        with self.assertRaisesRegex(ValueError, "share request slots"):
            pd_state.combined_kpool_tail_infos(target.full_kv_pool, draft)

    def test_target_only_without_mtp(self):
        target, _, _, _ = make_registered(9)
        args = types.SimpleNamespace()
        UTILS["setup_state_kv_args"](args, target)
        self.assertEqual(args.state_types, ["MAMBA", "DSA_TAIL"])
        self.assertEqual(len(args.state_data_ptrs[1]), 4)

    def test_exact_byte_transfer_all_remainders_and_reused_decode_slots(self):
        src, src_draft, _, _ = make_registered(5)
        dst, dst_draft, _, _ = make_registered(49)
        sp, _, si = pd_state.combined_kpool_tail_infos(src.full_kv_pool, src_draft)
        dp, _, di = pd_state.combined_kpool_tail_infos(dst.full_kv_pool, dst_draft)
        src_buffers = (
            src.full_kv_pool._glm53_kpool_tail_buffers
            + src_draft._glm53_kpool_tail_buffers
        )
        dst_buffers = (
            dst.full_kv_pool._glm53_kpool_tail_buffers
            + dst_draft._glm53_kpool_tail_buffers
        )
        # All lengths around the two benchmark sizes, and every PD decode row,
        # including rows above max_running_requests=16.
        for length in (65536, 65537, 65538, 65539, 131072, 131073, 131074, 131075):
            for dst_row in range(1, 49):
                with self.subTest(length=length, dst_row=dst_row):
                    for b, buf in enumerate(src_buffers):
                        buf.array[2] = np.arange(512).reshape(4, 128) + 10 * b
                    for buf in dst_buffers:
                        buf.array.fill(7)
                    blocks = UTILS["build_dsa_tail_transfer_blocks"](
                        sp,
                        si,
                        dp,
                        UTILS["get_dsa_tail_state_indices"](src, 2, length),
                        UTILS["get_dsa_tail_state_indices"](dst, dst_row, length),
                        di,
                    )
                    for source, dest, size in blocks:
                        ctypes.memmove(dest, source, size)
                    for source, dest in zip(src_buffers, dst_buffers):
                        expected = np.full_like(dest.array, 7)
                        expected[dst_row, : length % 4] = source.array[2, : length % 4]
                        np.testing.assert_array_equal(dest.array, expected)

    def test_kv_layer_ids_cover_target_and_draft_groups(self):
        target, draft, _, _ = make_registered(9)
        ids = UTILS["build_kv_layer_ids"](
            token_to_kv_pool=target,
            draft_token_to_kv_pool=draft,
            num_draft_entries=3,
            num_hidden_layers=46,
        )
        self.assertEqual(ids, [3, 11, 3, 11, 3, 11, 46, 46, 46])
        self.assertEqual(
            len(ids),
            len(target.full_kv_pool.get_contiguous_buf_infos()[0])
            + len(draft.get_contiguous_buf_infos()[0]),
        )

    def test_kv_layer_ids_packed_and_index_subset(self):
        for packed in (False, True):
            for scales in (False, True):
                pool = NPUPool(3, indexed=[0, 2], packed=packed, scales=scales)
                hybrid = HybridPool(pool, [3, 7, 11])
                ids = hybrid.get_kv_layer_ids()
                expected = [3, 7, 11] * (1 if packed else 2) + [3, 11] * (
                    2 if scales else 1
                )
                self.assertEqual(ids, expected)
                self.assertEqual(len(ids), len(pool.get_contiguous_buf_infos()[0]))

    def test_unsupported_topology_and_layout_rejected_before_growth(self):
        for change in (
            {"tp_size": 8},
            {"moe_ep_size": 8},
            {"attn_tp_size": 4},
            {"tp_size": 8, "moe_ep_size": 8, "enable_dp_attention": True,
             "dp_size": 2, "attn_dp_size": 2, "attn_tp_size": 2},
            {"tp_size": 8, "moe_ep_size": 8, "enable_dp_attention": True,
             "dp_size": 4, "attn_dp_size": 4, "attn_tp_size": 4},
            {"tp_size": 8, "moe_ep_size": 8, "enable_dp_attention": True,
             "dp_size": 2, "attn_dp_size": 2, "attn_tp_size": 4, "nnodes": 2},
            {"pp_size": 2},
            {"attn_cp_size": 2},
            {"enable_dp_attention": True},
        ):
            m, indexers = model([0])
            parallel = types.SimpleNamespace(**(vars(PARALLEL) | change))
            with self.assertRaisesRegex(ValueError, "PP1/CP1"):
                pd_state.register_glm53_kpool_state(
                    m, NPUPool(1), req_pool(49), parallel=parallel
                )
            self.assertEqual(indexers[0]._kpool_tail_k.shape[0], 3)
        for attr, value in (
            ("_glm53_index_layout", object()),
            ("share_zero_rope", True),
        ):
            pool = NPUPool(1)
            setattr(pool, attr, value)
            with self.assertRaisesRegex(ValueError, "compact index/shared RoPE"):
                pd_state.register_glm53_kpool_state(
                    model([0])[0], pool, req_pool(49), parallel=PARALLEL
                )

    def test_missing_layer_fails(self):
        with self.assertRaisesRegex(ValueError, "every index-cache layer"):
            pd_state.register_glm53_kpool_state(
                model([3])[0],
                HybridPool(NPUPool(2), [3, 11]),
                req_pool(49),
                parallel=PARALLEL,
            )

    def test_unregistered_npu_tail_is_empty(self):
        pool = NPUPool(1)
        self.assertEqual(pool.get_compress_tail_buf_infos(), ([], [], []))
        self.assertEqual(UTILS["get_dsa_tail_state_indices"](pool, 1, 65539), [])

    def test_prefill_mamba_layout_matches_decode_without_verify_scratch(self):
        for draft_tokens in (1, 4):
            p, d = MambaPool(5, 3, False), MambaPool(9, 3 + draft_tokens - 1, True)
            requests = types.SimpleNamespace(mamba_pool=p)
            token_pool = types.SimpleNamespace(mamba_pool=p)
            p.mamba_cache.conv[0].array.fill(9)
            pd_state.prepare_glm53_pd_mamba_state(
                token_pool, requests, mode="prefill", draft_tokens=draft_tokens
            )
            self.assertFalse(hasattr(p.mamba_cache, "intermediate_ssm"))
            self.assertEqual(
                p.get_contiguous_buf_infos()[2], d.get_contiguous_buf_infos()[2]
            )
            self.assertEqual(p.get_state_dim_per_tensor(), d.get_state_dim_per_tensor())
            self.assertEqual(
                p.get_state_slice_outer_counts(), d.get_state_slice_outer_counts()
            )
            self.assertEqual(
                p.mamba_cache.temporal.stride()[2:], d.mamba_cache.temporal.stride()[2:]
            )
            np.testing.assert_array_equal(p.mamba_cache.conv[0].array[:, :, -3:], 9)
            self.assertFalse(
                p.mamba_cache.conv[0].array[:, :, : draft_tokens - 1].any()
            )
            # Idempotent; P/D row counts are deliberately different.
            ptrs = p.get_contiguous_buf_infos()[0]
            pd_state.prepare_glm53_pd_mamba_state(
                token_pool, requests, mode="prefill", draft_tokens=draft_tokens
            )
            self.assertEqual(ptrs, p.get_contiguous_buf_infos()[0])
            # Write a non-symmetric matrix AFTER matching the views. A missing
            # P transpose would silently transpose the matrix copied to D.
            p.mamba_cache.temporal.array[:] = np.arange(2 * 5 * 2 * 7 * 5).reshape(
                2, 5, 2, 7, 5
            )

            def copy_blocks(session, blocks):
                for source, dest, size in blocks:
                    ctypes.memmove(dest, source, size)
                return 0

            manager = types.SimpleNamespace(pp_size=1, _transfer_data=copy_blocks)
            MAMBA_SEND(
                manager,
                types.SimpleNamespace(mooncake_session_id="cpu"),
                [2],
                ptrs,
                p.get_contiguous_buf_infos()[2],
                d.get_contiguous_buf_infos()[0],
                [7],
                p.get_state_layer_ids(),
                d.get_state_layer_ids(),
            )
            np.testing.assert_array_equal(
                p.mamba_cache.temporal.array[:, 2], d.mamba_cache.temporal.array[:, 7]
            )
            np.testing.assert_array_equal(
                p.mamba_cache.conv[0].array[:, 2], d.mamba_cache.conv[0].array[:, 7]
            )
            self.assertFalse(d.mamba_cache.temporal.array[:, :7].any())
            self.assertFalse(d.mamba_cache.temporal.array[:, 8:].any())

    def test_910b_tp4_to_tp2_mamba_conv_and_temporal_bytes(self):
        # Each D TP rank receives two TP-sharded P states from its selected P
        # DP domain. Conv has a row before its channel axis; temporal does not.
        for d_tp_rank in (0, 1):
            with self.subTest(d_tp_rank=d_tp_rank):
                dest = MambaPool(
                    9, 6, True, conv_channels=12, temporal_heads=4
                )
                src_states = []

                def copy_blocks(session, blocks):
                    for source, target, size in blocks:
                        ctypes.memmove(target, source, size)
                    return 0

                for writer in (0, 1):
                    p_tp_rank = d_tp_rank * 2 + writer
                    source = MambaPool(5, 3, False)
                    pd_state.prepare_glm53_pd_mamba_state(
                        types.SimpleNamespace(mamba_pool=source),
                        types.SimpleNamespace(mamba_pool=source),
                        mode="prefill", draft_tokens=4,
                    )
                    source.mamba_cache.conv[0].array[:, 2] = (
                        p_tp_rank * 1000
                        + np.arange(2 * 6 * 6).reshape(2, 6, 6)
                    )
                    source.mamba_cache.temporal.array[:, 2] = (
                        p_tp_rank * 1000
                        + np.arange(2 * 2 * 7 * 5).reshape(2, 2, 7, 5)
                    )
                    src_states.append(source)
                    src_ptrs, _, src_lens = source.get_contiguous_buf_infos()
                    dst_ptrs, _, dst_lens = dest.get_contiguous_buf_infos()
                    sender = types.SimpleNamespace(
                        kv_args=types.SimpleNamespace(engine_rank=p_tp_rank),
                        attn_tp_size=4, pp_size=1,
                        _transfer_data=copy_blocks,
                    )
                    self.assertEqual(
                        MAMBA_SLICE_SEND(
                            sender, types.SimpleNamespace(mooncake_session_id="cpu"),
                            [2], src_ptrs, src_lens,
                            source.get_state_dim_per_tensor(),
                            dst_ptrs, [7], dst_lens,
                            dest.get_state_dim_per_tensor(),
                            d_tp_rank, 2, None,
                            source.get_state_slice_outer_counts(),
                            source.get_state_layer_ids(),
                            dest.get_state_layer_ids(),
                        ),
                        0,
                    )

                np.testing.assert_array_equal(
                    dest.mamba_cache.conv[0].array[:, 7],
                    np.concatenate(
                        [p.mamba_cache.conv[0].array[:, 2] for p in src_states],
                        axis=-1,
                    ),
                )
                np.testing.assert_array_equal(
                    dest.mamba_cache.temporal.array[:, 7],
                    np.concatenate(
                        [p.mamba_cache.temporal.array[:, 2] for p in src_states],
                        axis=1,
                    ),
                )
                self.assertFalse(dest.mamba_cache.conv[0].array[:, :7].any())
                self.assertFalse(dest.mamba_cache.conv[0].array[:, 8:].any())
                self.assertFalse(dest.mamba_cache.temporal.array[:, :7].any())
                self.assertFalse(dest.mamba_cache.temporal.array[:, 8:].any())

    def test_mamba_no_mtp_and_decode_do_not_change_layout(self):
        for mode, draft_tokens in (("prefill", None), ("decode", 4)):
            pool = MambaPool(5, 3, False)
            state = pool.mamba_cache
            pd_state.prepare_glm53_pd_mamba_state(
                types.SimpleNamespace(mamba_pool=pool),
                types.SimpleNamespace(mamba_pool=pool),
                mode=mode,
                draft_tokens=draft_tokens,
            )
            self.assertIs(pool.mamba_cache, state)

    def test_unified_runner_does_not_call_registration_hook(self):
        calls = []
        ns = {
            "_is_npu": True,
            "get_disagg": lambda: types.SimpleNamespace(disaggregation_mode="null"),
            "install_canary": lambda **kwargs: None,
        }
        fn = source_functions(
            SRT / "model_executor/model_runner.py",
            ["_init_post_memory_pool_components"],
            ns,
            "ModelRunner",
        )["_init_post_memory_pool_components"]
        runner = types.SimpleNamespace(
            model=types.SimpleNamespace(
                register_kv_pool_state=lambda *args: calls.append(args)
            ),
            token_to_kv_pool=object(),
            req_to_token_pool=object(),
            server_args=None,
            _token_oracle_manager=None,
        )
        for name in [
            "init_kv_index_translator",
            "init_ngram_embedding_manager",
            "maybe_init_hisparse_coordinator",
            "init_routed_experts_capturer",
            "init_indexer_capturer",
        ]:
            setattr(runner, name, lambda: None)
        fn(runner)
        self.assertEqual(calls, [])
        ns["get_disagg"] = lambda: types.SimpleNamespace(disaggregation_mode="decode")
        fn(runner)
        self.assertEqual(calls, [(runner.token_to_kv_pool, runner.req_to_token_pool)])


@unittest.skipIf(torch is None, "real CPU Tensor checks require torch")
class TestTorchPDState(unittest.TestCase):
    def setUp(self):
        self.patcher = patch.dict(sys.modules, MODULES)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_real_module_buffers_grow_without_becoming_checkpoint_weights(self):
        module = torch.nn.Module()
        module.indexers = torch.nn.ModuleList()
        for lid in (3, 11):
            indexer = torch.nn.Module()
            indexer.layer_id, indexer.index_kpool, indexer.head_dim = lid, 4, 128
            indexer.register_buffer(
                "_kpool_tail_k",
                torch.full((5, 4, 128), lid, dtype=torch.bfloat16),
                persistent=False,
            )
            indexer.register_buffer(
                "_kpool_tail_score",
                torch.full((5, 4, 128), lid + 0.5, dtype=torch.float32),
                persistent=False,
            )
            indexer.register_buffer(
                "_kpool_mtp_tail_k",
                torch.zeros((5, 4, 4, 128), dtype=torch.bfloat16),
                persistent=False,
            )
            module.indexers.append(indexer)
        scratch = [m._kpool_mtp_tail_k.data_ptr() for m in module.indexers]
        pool, requests = HybridPool(NPUPool(2), [3, 11]), req_pool(49)
        pd_state.register_glm53_kpool_state(module, pool, requests, parallel=PARALLEL)
        for i, indexer in enumerate(module.indexers):
            self.assertIs(
                indexer.get_buffer("_kpool_tail_k"),
                pool.full_kv_pool._glm53_kpool_tail_buffers[i],
            )
            self.assertEqual(indexer._kpool_tail_k.dtype, torch.bfloat16)
            self.assertEqual(indexer._kpool_tail_score.dtype, torch.float32)
            self.assertEqual(indexer._kpool_tail_k.device.type, "cpu")
            self.assertEqual(tuple(indexer._kpool_tail_k.shape), (49, 4, 128))
            self.assertTrue(torch.all(indexer._kpool_tail_k[:5] == indexer.layer_id))
            self.assertEqual(torch.count_nonzero(indexer._kpool_tail_k[5:]).item(), 0)
            self.assertEqual(indexer._kpool_mtp_tail_k.data_ptr(), scratch[i])
        self.assertEqual(module.state_dict(), {})
        self.assertEqual(len(list(module.named_buffers())), 6)
        pointers = pool.full_kv_pool.get_compress_tail_buf_infos()[0]
        pd_state.register_glm53_kpool_state(module, pool, requests, parallel=PARALLEL)
        self.assertEqual(pointers, pool.full_kv_pool.get_compress_tail_buf_infos()[0])

    def test_real_npu_conv_allocator_and_mamba_bytes_on_cpu(self):
        conv_init = source_functions(
            SRT / "hardware_backend/npu/memory_pool_npu.py",
            ["_init_npu_conv_state"],
            {"torch": torch},
        )["_init_npu_conv_state"]
        p, d = MambaPool(5, 3, False), MambaPool(9, 6, True)

        def state(rows, draft_tokens):
            conv = conv_init(
                torch.zeros((2, rows, 3, 6), dtype=torch.bfloat16),
                [(3, 6)],
                draft_tokens,
                is_kda=True,
            )
            temporal = torch.zeros((2, rows, 2, 5, 7), dtype=torch.float32)
            return MambaState(
                conv,
                temporal.transpose(-1, -2) if draft_tokens is not None else temporal,
            )

        p.mamba_cache, d.mamba_cache = state(5, None), state(9, 4)
        self.assertNotEqual(
            p.get_contiguous_buf_infos()[2], d.get_contiguous_buf_infos()[2]
        )
        pd_state.prepare_glm53_pd_mamba_state(
            types.SimpleNamespace(mamba_pool=p),
            types.SimpleNamespace(mamba_pool=p),
            mode="prefill",
            draft_tokens=4,
        )
        self.assertEqual(
            p.get_contiguous_buf_infos()[2], d.get_contiguous_buf_infos()[2]
        )
        self.assertEqual(
            p.mamba_cache.temporal.stride()[2:], d.mamba_cache.temporal.stride()[2:]
        )
        p.mamba_cache.temporal.copy_(
            torch.arange(p.mamba_cache.temporal.numel(), dtype=torch.float32).reshape(
                p.mamba_cache.temporal.shape
            )
        )
        p.mamba_cache.conv[0][:, 2].fill_(13)
        d.mamba_cache.temporal.fill_(-7)

        def copy_blocks(session, blocks):
            for source, dest, size in blocks:
                ctypes.memmove(dest, source, size)
            return 0

        MAMBA_SEND(
            types.SimpleNamespace(pp_size=1, _transfer_data=copy_blocks),
            types.SimpleNamespace(mooncake_session_id="cpu"),
            [2],
            p.get_contiguous_buf_infos()[0],
            p.get_contiguous_buf_infos()[2],
            d.get_contiguous_buf_infos()[0],
            [8],
            p.get_state_layer_ids(),
            d.get_state_layer_ids(),
        )
        torch.testing.assert_close(
            p.mamba_cache.temporal[:, 2], d.mamba_cache.temporal[:, 8], rtol=0, atol=0
        )
        torch.testing.assert_close(
            p.mamba_cache.conv[0][:, 2], d.mamba_cache.conv[0][:, 8], rtol=0, atol=0
        )
        self.assertTrue(torch.all(d.mamba_cache.temporal[:, :8] == -7))
        self.assertEqual(d.mamba_cache.temporal.device.type, "cpu")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    unittest.main()
