"""CPU contracts for C8 records; native CANN accuracy is tested on 910B."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import torch

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "python/sglang/srt/hardware_backend/npu/memory_pool_npu.py"


def method(name, **extra):
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "NPUMLATokenToKVPool")
    node = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                              and n.name == name))
    node.decorator_list = []
    namespace = dict(torch=torch, **extra)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[
        ast.alias(name="annotations")], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    return namespace[name]


def cpu_dynamic_quant(x):
    scale = x.float().abs().amax(-1).div(127).clamp_min(1e-20)
    q = (x.float() / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
    return q, scale


def cpu_scatter(dst, indices, updates):
    # Assert dtype instead of allowing index_copy_ to obscure a bad cast.
    assert dst.dtype == updates.dtype
    dst.index_copy_(0, indices.reshape(-1).long(), updates)


class TestNpuGlmC8Pool(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(15)
        self.ops = NS(npu_dynamic_quant=cpu_dynamic_quant,
                      npu_scatter_nd_update_=cpu_scatter)
        self.pool = NS(kv_lora_rank=512, packed_rope_dim=64, kv_cache_dim=656,
                       dsa_kv_cache_store_int8=True,
                       dsa_kv_cache_store_quantized=True,
                       dsa_kv_cache_store_fp8=False,
                       index_dtype=torch.bfloat16, index_store_dtype=torch.bfloat16,
                       start_layer=0, layer_num=2, index_head_dim=128,
                       indexer_layer_ids=(0, 1), index_k_scale_buffer=None,
                       indexer_layer_id_to_slot={0: 0, 1: 1},
                       k_buffer=torch.zeros(2, 3, 64, 1, 656, dtype=torch.int8),
                       v_buffer=torch.empty(2, 3, 64, 1, 0, dtype=torch.int8),
                       index_k_buffer=torch.zeros(2, 3, 64, 1, 128, dtype=torch.bfloat16),
                       _raise_if_native_kv_cache_disabled=lambda: None)
        pack = method("_pack_dsa_int8_kv_cache", torch_npu=self.ops)
        self.pool._pack_dsa_int8_kv_cache = lambda x: pack(self.pool, x)
        self.pool._get_indexer_slot = lambda lid: lid
        self.pool.get_state_buf_infos = lambda: method("get_state_buf_infos")(self.pool)
        self.pool.get_state_layer_ids = lambda: method("get_state_layer_ids")(self.pool)

    def test_group_scales_and_zero_rope_are_packed_losslessly(self):
        x = torch.randn(7, 1, 512, dtype=torch.bfloat16)
        x[:, :, :128] *= 100
        packed = self.pool._pack_dsa_int8_kv_cache(x)
        self.assertEqual(packed.dtype, torch.int8)
        self.assertEqual(tuple(packed.shape), (7, 656))
        self.assertEqual(packed[:, 512:640].count_nonzero(), 0)
        scales = packed[:, 640:].contiguous().view(torch.float32)
        reference_q, reference_s = cpu_dynamic_quant(x.reshape(-1, 128))
        torch.testing.assert_close(scales, reference_s.reshape(7, 4), rtol=0, atol=0)
        torch.testing.assert_close(packed[:, :512], reference_q.reshape(7, 512), rtol=0, atol=0)
        restored = (packed[:, :512].reshape(7, 4, 128).float() * scales[..., None])
        error = (restored - x.reshape(7, 4, 128).float()).abs()
        self.assertTrue((error <= scales[..., None] * .50001 + 1e-6).all())

    def test_sparse_scatter_keeps_record_and_original_index_precision(self):
        x = torch.randn(3, 1, 512, dtype=torch.bfloat16)
        loc = torch.tensor([1, 80, 150])
        write = method("set_kv_buffer", torch_npu=self.ops,
                       unwrap_write_loc=lambda x: (x, None, None))
        write(self.pool, NS(layer_id=1), loc, x, None)
        actual = self.pool.k_buffer[1].reshape(-1, 656)[loc]
        torch.testing.assert_close(actual, self.pool._pack_dsa_int8_kv_cache(x), rtol=0, atol=0)
        self.assertEqual(self.pool.k_buffer[0].count_nonzero(), 0)
        index = torch.randn(3, 1, 128, dtype=torch.bfloat16)
        method("set_index_k_buffer", torch_npu=self.ops)(self.pool, 1, loc, index)
        torch.testing.assert_close(self.pool.index_k_buffer[1].reshape(-1, 128)[loc],
                                   index.reshape(3, 128), rtol=0, atol=0)

    def test_transfer_omits_empty_v_and_includes_intact_index(self):
        ptrs, lens, pages = method("get_contiguous_buf_infos")(self.pool)
        self.assertEqual(len(ptrs), 4)
        self.assertNotIn(0, ptrs)
        self.assertEqual(pages, [64 * 656] * 2 + [64 * 128 * 2] * 2)
        self.assertEqual(lens, [3 * 64 * 656] * 2 + [3 * 64 * 128 * 2] * 2)
        self.assertEqual(method("get_kv_layer_ids")(self.pool), [0, 1, 0, 1])

    def test_zero_groups_remain_finite(self):
        packed = self.pool._pack_dsa_int8_kv_cache(torch.zeros(2, 1, 512, dtype=torch.bfloat16))
        scales = packed[:, 640:].contiguous().view(torch.float32)
        self.assertTrue(scales.isfinite().all())
        self.assertEqual(packed[:, :640].count_nonzero(), 0)


if __name__ == "__main__":
    unittest.main()
