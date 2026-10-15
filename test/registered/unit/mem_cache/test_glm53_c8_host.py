"""CPU contracts for GLM53 logical-page Host storage with C8 target + MTP."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig
from sglang.srt.mem_cache.pool_host import glm53
from sglang.srt.mem_cache.pool_host.common import alloc_with_pin_memory
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def make_pool(layers=2, index_layers=(1,), int8=True, seed=0):
    dtype, width = (torch.int8, 656) if int8 else (torch.bfloat16, 512)
    pool = SimpleNamespace(
        device="npu",
        size=512,
        page_size=64,
        layer_num=layers,
        start_layer=0,
        end_layer=layers - 1,
        store_dtype=dtype,
        kpool_use_compress=True,
        index_kpool=4,
        zero_rope=not int8,
        dsa_kv_cache_store_int8=int8,
        qk_rope_head_dim=0,
        kv_cache_dim=width,
        indexer_layer_ids=index_layers,
        layer_shard_enabled=False,
        k_buffer=torch.empty((layers, 9, 64, 1, width), dtype=dtype),
        index_k_buffer=torch.empty(
            (len(index_layers), 9, 64, 1, 128), dtype=torch.bfloat16
        ),
        v_buffer=torch.empty((layers, 9, 64, 1, 0 if int8 else 64), dtype=dtype),
    )
    for component in (pool.k_buffer, pool.index_k_buffer):
        raw = component.view(torch.uint8).reshape(-1)
        raw.copy_(((torch.arange(raw.numel()) * 17 + seed) % 251).to(torch.uint8))
    # Deliberately no data_ptrs or kv_buffer: native NPU pools expose tensors.
    return pool


def make_host(target, draft=(), **kwargs):
    with mock.patch.dict(glm53.ALLOC_MEMORY_FUNCS, {"npu": alloc_with_pin_memory}):
        return glm53.GLM53MLAPoolHost(
            target,
            1.5,
            kwargs.pop("host_size", 0),
            kwargs.pop("page_size", 256),
            "page_first_kv_split",
            pin_memory=False,
            mtp_draft_device_pools=draft,
            **kwargs,
        )


def make_file(directory, host):
    backend = HiCacheFile(
        HiCacheStorageConfig(
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            is_mla_model=True,
            enable_storage_metrics=False,
            is_page_first_layout=False,
            model_name="GLM5Next",
            extra_config={
                "max_size": "0",
                "min_free_space": "0",
                "enable_metadata_cache": True,
            },
        ),
        file_path=directory,
    )
    backend.register_mem_pool_host(host)
    return backend


def indices(pages):
    return torch.cat([torch.arange(page * 64, (page + 1) * 64) for page in pages])


class TestGLM53C8Host(unittest.TestCase):
    def setUp(self):
        self.env = envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR.override("")
        self.env.__enter__()

    def tearDown(self):
        self.env.__exit__(None, None, None)

    def test_dispatch_accepts_c8_without_zero_rope(self):
        target = make_pool()
        self.assertFalse(target.zero_rope)
        self.assertTrue(glm53.is_glm53_kpool(target))
        self.assertTrue(glm53.is_glm53_kpool(make_pool(int8=False)))
        target.kv_cache_dim = 512
        self.assertFalse(glm53.is_glm53_kpool(target))
        with self.assertRaisesRegex(ValueError, "native"):
            make_host(make_pool(), (target,))
        with self.assertRaisesRegex(ValueError, "256-token"):
            make_host(make_pool(), page_size=64)

    def test_target_and_draft_budget_uses_component_itemsize(self):
        target, draft = make_pool(), make_pool(1, (0,), seed=19)
        expected = (2 * 656 + 128 * 2) + (656 + 128 * 2)
        host = make_host(target, (draft,), host_size=expected * 700 / 1e9)
        self.assertEqual(host.size_per_token, 2480)
        self.assertEqual(host.size, 768)
        self.assertEqual(host.target_layer_num, 2)
        self.assertEqual(host.layer_num, 3)
        self.assertEqual(host.page_size, 256)
        self.assertEqual(host.physical_page_size, 64)
        self.assertEqual(
            [buf.dtype for buf in host._components], [torch.int8, torch.bfloat16] * 2
        )
        self.assertEqual(
            sum(buf.nbytes for buf in host._components), host.size * expected
        )
        self.assertEqual(host.page_bytes, 256 * expected)
        self.assertEqual(host.get_dummy_flat_data_page().numel(), host.page_bytes)
        self.assertEqual(host._pool_buffers[id(target)][0].shape[1], 2)
        self.assertEqual(host._pool_buffers[id(draft)][0].shape[1], 1)

    def test_logical_page_metadata_contains_all_four_components(self):
        host = make_host(make_pool(), (make_pool(1, (0,)),))
        ptrs, sizes = host.get_page_buffer_meta(
            torch.cat((torch.arange(512, 768), torch.arange(256)))
        )
        expected_sizes = [256 * 2 * 656, 256 * 128 * 2, 256 * 656, 256 * 128 * 2]
        self.assertEqual(sizes, expected_sizes * 2)
        self.assertEqual(
            ptrs, [buf[page].data_ptr() for page in (2, 0) for buf in host._components]
        )
        with self.assertRaisesRegex(ValueError, "incomplete"):
            host.get_page_buffer_meta(torch.arange(64))
        with self.assertRaisesRegex(ValueError, "unaligned"):
            host.get_data_page(64)
        with self.assertRaisesRegex(ValueError, "incompatible size"):
            host.set_from_flat_data_page(
                0, torch.empty(host.page_bytes - 1, dtype=torch.uint8)
            )

    def test_arbitrary_four_physical_pages_target_draft_file_roundtrip(self):
        target, draft = make_pool(seed=7), make_pool(1, (0,), seed=83)
        host = make_host(target, (draft,))
        pools = (target, draft)
        expected = [(p.k_buffer.clone(), p.index_k_buffer.clone()) for p in pools]
        source_pages, destination_pages = [7, 1, 5, 3], [0, 6, 2, 4]
        calls = []
        module = ModuleType("sgl_kernel_npu.kvcacheio")
        module.TransferDirection = SimpleNamespace(D2H=0, H2D=1)

        def exchange(**kw):
            self.assertEqual(kw["page_size"], 64)
            self.assertEqual(kw["device_k"].dtype, torch.int8)
            self.assertEqual(kw["device_index_k"].dtype, torch.bfloat16)
            self.assertEqual(kw["host_index_k"].dtype, torch.bfloat16)
            self.assertEqual(kw["device_v"].numel(), 0)
            self.assertEqual(kw["host_v"].numel(), 0)
            calls.append((kw["device_k"].data_ptr(), kw["direction"]))
            for name in ("k", "index_k"):
                device, cpu = kw["device_" + name], kw["host_" + name]
                for di, hi in zip(kw["device_indices"][::64], kw["host_indices"][::64]):
                    dev, dst = device[:, di.item() // 64], cpu[hi.item() // 64]
                    (dst if kw["direction"] == 0 else dev).copy_(
                        dev if kw["direction"] == 0 else dst
                    )

        module.transfer_kv_dim_exchange = exchange
        with mock.patch.dict(
            sys.modules, {module.__name__: module}
        ), tempfile.TemporaryDirectory() as directory:
            host.backup_from_device_all_layer(
                target, torch.arange(256, 512), indices(source_pages), "kernel_ascend"
            )
            golden = torch.cat(
                [
                    component[:, source_pages]
                    .transpose(0, 1)
                    .contiguous()
                    .view(torch.uint8)
                    .reshape(-1)
                    for pair in expected
                    for component in pair
                ]
            )
            self.assertTrue(torch.equal(host.get_data_page(256), golden))
            backend = make_file(directory, host)
            self.assertTrue(backend.set("logical-page", host.get_data_page(256)))
            self.assertEqual(
                Path(backend._get_component_path("logical-page")).read_bytes(),
                golden.numpy().tobytes(),
            )
            for component in host._components:
                component.zero_()
            for pool in pools:
                pool.k_buffer.zero_()
                pool.index_k_buffer.zero_()
            host.set_from_flat_data_page(
                512, backend.get("logical-page", host.get_dummy_flat_data_page())
            )
            host.load_to_device_per_layer(
                target,
                torch.arange(512, 768),
                indices(destination_pages),
                0,
                "kernel_ascend",
            )
            host.load_to_device_per_layer(
                target,
                torch.arange(512, 768),
                indices(destination_pages),
                1,
                "kernel_ascend",
            )
            host.load_to_device_per_layer(
                draft,
                torch.arange(512, 768),
                indices(destination_pages),
                2,
                "kernel_ascend",
                is_draft=True,
            )
            for pool, pair in zip(pools, expected):
                for actual, original in zip((pool.k_buffer, pool.index_k_buffer), pair):
                    self.assertTrue(
                        torch.equal(
                            actual[:, destination_pages].contiguous().view(torch.uint8),
                            original[:, source_pages].contiguous().view(torch.uint8),
                        )
                    )
                    self.assertEqual(
                        actual[:, [1, 3, 5, 7, 8]].view(torch.uint8).count_nonzero(), 0
                    )
            self.assertEqual(
                calls,
                [
                    (target.k_buffer.data_ptr(), 0),
                    (draft.k_buffer.data_ptr(), 0),
                    (target.k_buffer.data_ptr(), 1),
                    (draft.k_buffer.data_ptr(), 1),
                ],
            )

    def test_namespace_separates_bf16_and_encodes_draft_and_logical_layout(self):
        target, draft = make_pool(), make_pool(1, (0,))
        host = make_host(target, (draft,))
        bf16 = make_host(make_pool(int8=False), (make_pool(1, (0,), int8=False),))
        self.assertEqual(bf16.get_storage_namespace(), "")
        self.assertNotEqual(host.get_storage_namespace(), bf16.get_storage_namespace())
        self.assertEqual(
            host.get_storage_namespace(),
            make_host(
                make_pool(), (make_pool(1, (0,)),), host_size=2480 * 700 / 1e9
            ).get_storage_namespace(),
        )
        self.assertNotEqual(
            host.get_storage_namespace(), make_host(target).get_storage_namespace()
        )
        self.assertNotEqual(
            host.get_storage_namespace(),
            make_host(target, (draft,), page_size=512).get_storage_namespace(),
        )
        self.assertNotEqual(
            host.get_storage_namespace(),
            make_host(make_pool(index_layers=(0,)), (draft,)).get_storage_namespace(),
        )
        with tempfile.TemporaryDirectory() as directory:
            old, new = make_file(directory, bf16), make_file(directory, host)
            self.assertTrue(old.set("same-hash", bf16.get_data_page(0)))
            self.assertFalse(new.exists("same-hash"))
            self.assertIsNone(new.get("same-hash", host.get_dummy_flat_data_page()))
            self.assertTrue(new.set("same-hash", host.get_data_page(0)))
            self.assertNotEqual(
                old._get_component_path("same-hash"),
                new._get_component_path("same-hash"),
            )


if __name__ == "__main__":
    unittest.main()
