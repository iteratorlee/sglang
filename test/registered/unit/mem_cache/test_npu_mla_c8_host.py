"""CPU contracts for NPU packed MLA Host/file transfers (no accelerator needed)."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    HiCacheStorageConfig,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.pool_host import mla
from sglang.srt.mem_cache.pool_host.common import alloc_with_pin_memory
from sglang.srt.mem_cache.pool_host.group import HostPoolGroup, PoolEntry
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def device_pool(encoding="int8", index_layers=(0, 2)):
    packed = encoding != "bf16"
    dtype = {"int8": torch.int8, "fp8": torch.float8_e4m3fn, "bf16": torch.bfloat16}[
        encoding
    ]
    index_dtype = torch.bfloat16 if encoding == "int8" else dtype
    kdim, vdim = (656, 0) if packed else (512, 64)
    pool = SimpleNamespace(
        size=192,
        page_size=64,
        layer_num=3,
        start_layer=0,
        end_layer=3,
        layer_shard_enabled=False,
        device="cpu",
        dtype=dtype,
        store_dtype=dtype,
        kv_lora_rank=512,
        qk_rope_head_dim=64 if encoding == "fp8" else 0,
        logical_qk_rope_head_dim=64 if encoding == "fp8" else 0,
        kv_cache_dim=kdim,
        kr_cache_dim=vdim,
        index_head_dim=128,
        index_dtype=index_dtype,
        index_store_dtype=index_dtype,
        indexer_layer_ids=index_layers,
        num_indexer_layers=len(index_layers),
        dsa_kv_cache_store_fp8=encoding == "fp8",
        dsa_kv_cache_store_int8=encoding == "int8",
        dsa_kv_cache_store_quantized=packed,
        k_buffer=torch.empty((3, 4, 64, 1, kdim), dtype=dtype),
        v_buffer=torch.empty((3, 4, 64, 1, vdim), dtype=dtype),
        index_k_buffer=torch.empty(
            (len(index_layers), 4, 64, 1, 128), dtype=index_dtype
        ),
        index_k_scale_buffer=(
            torch.empty((len(index_layers), 4, 64, 1), dtype=torch.float32)
            if encoding == "fp8"
            else None
        ),
    )
    # Distinct raw bytes catch accidental numeric casts of the packed scales,
    # signed INT8 values, and non-integer BF16 index keys.
    for name in ("k_buffer", "v_buffer", "index_k_buffer", "index_k_scale_buffer"):
        tensor = getattr(pool, name)
        if tensor is not None:
            raw = tensor.view(torch.uint8).reshape(-1)
            raw.copy_((torch.arange(raw.numel()) % 251).to(torch.uint8))
    if encoding == "int8":
        pool.k_buffer[..., 512:640].zero_()  # physical BF16 zero-RoPE lanes
        scales = torch.tensor([0.125, 0.25, 0.5, 1.0], dtype=torch.float32)
        pool.k_buffer[..., 640:].view(torch.uint8).copy_(scales.view(torch.uint8))
        pool.index_k_buffer.copy_(
            torch.linspace(-1.5, 1.5, pool.index_k_buffer.numel()).reshape_as(
                pool.index_k_buffer
            )
        )
    return pool


def host_pool(pool, ratio=2, host_size=0):
    with mock.patch.dict(mla.ALLOC_MEMORY_FUNCS, {"cpu": alloc_with_pin_memory}):
        return mla.MLATokenToKVPoolHost(
            pool, ratio, host_size, 64, "page_first_kv_split", pin_memory=False
        )


def file_backend(directory, host, **overrides):
    config = dict(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=True,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="GLM5Next",
        extra_config={
            "enable_metadata_cache": True,
            "max_size": "0",
            "min_free_space": "0",
        },
    )
    config.update(overrides)
    backend = HiCacheFile(HiCacheStorageConfig(**config), file_path=directory)
    backend.register_mem_pool_host(host)
    backend.register_mem_host_pool_v2(host, PoolName.KV)
    return backend


class TestNpuMlaC8Host(unittest.TestCase):
    def setUp(self):
        self.env = envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR.override("")
        self.env.__enter__()

    def tearDown(self):
        self.env.__exit__(None, None, None)

    def test_capacity_matches_allocations_and_fixed_budget(self):
        pool = device_pool()
        expected = 3 * 656 + 2 * 128 * 2
        host = host_pool(pool, host_size=expected * 200 / 1e9)
        self.assertEqual(host.size_per_token, expected)
        self.assertEqual(host.size, 256)  # existing extra-page alignment
        self.assertEqual(host.k_buffer.shape[-1], 656)
        self.assertEqual(host.k_buffer.dtype, torch.int8)
        self.assertEqual(host.index_k_buffer.dtype, torch.bfloat16)
        self.assertEqual(host.v_buffer.numel(), 0)
        self.assertIsNone(host.index_k_scale_buffer)
        self.assertEqual(
            sum(t.nbytes for _, t in host._split_page_components()),
            host.size * expected,
        )
        self.assertEqual(host.get_dummy_flat_data_page().numel(), 64 * expected)

    def test_page_metadata_and_ascendc_pitches(self):
        pool = device_pool()
        host = host_pool(pool)
        indices = torch.cat((torch.arange(128, 192), torch.arange(0, 64)))
        ptrs, sizes = host.get_page_buffer_meta(indices)
        self.assertEqual(sizes, [3 * 64 * 656, 2 * 64 * 128 * 2] * 2)
        self.assertEqual(
            ptrs,
            [
                host.k_buffer[2].data_ptr(),
                host.index_k_buffer[2].data_ptr(),
                host.k_buffer[0].data_ptr(),
                host.index_k_buffer[0].data_ptr(),
            ],
        )
        meta = host._build_ascendc_component_meta(pool, 0, -1, 0, -1)
        self.assertEqual(len(meta), 2)
        self.assertEqual([m[6] for m in meta], [64 * 656, 64 * 128 * 2])
        self.assertEqual(
            [m[4] for m in meta],
            [host.k_buffer[0].nbytes, host.index_k_buffer[0].nbytes],
        )
        self.assertEqual(host._indexer_slot_range_for_layer(pool, 1), (0, 0))
        with self.assertRaisesRegex(TypeError, "does not convert dtype"):
            host._ascendc_comp_meta(
                pool.index_k_buffer, host.index_k_buffer.to(torch.int8), 0, 2
            )
        with self.assertRaisesRegex(ValueError, "payload shapes differ"):
            host._ascendc_comp_meta(pool.k_buffer, host.k_buffer[..., :512], 0, 3)

    def test_device_host_file_roundtrip_preserves_packed_bytes_and_bf16_index(self):
        pool = device_pool()
        host = host_pool(pool)
        calls = []
        direction = SimpleNamespace(D2H=0, H2D=1)

        def exchange(**kw):
            self.assertEqual(kw["device_k"].dtype, torch.int8)
            self.assertEqual(kw["host_k"].dtype, torch.int8)
            self.assertEqual(kw["device_index_k"].dtype, torch.bfloat16)
            self.assertEqual(kw["host_index_k"].dtype, torch.bfloat16)
            self.assertEqual(kw["host_v"].numel(), 0)
            self.assertIsNone(kw["host_index_k_scale"])
            calls.append(kw["direction"])
            for component in ("k", "v", "index_k", "index_k_scale"):
                dev, cpu = kw["device_" + component], kw["host_" + component]
                if dev is None or dev.numel() == 0:
                    continue
                for di, hi in zip(kw["device_indices"][::64], kw["host_indices"][::64]):
                    d, h = dev[:, di.item() // 64], cpu[hi.item() // 64]
                    (h if kw["direction"] == 0 else d).copy_(
                        d if kw["direction"] == 0 else h
                    )

        expected_k = pool.k_buffer[:, 2].clone()
        expected_index = pool.index_k_buffer[:, 2].clone()
        with mock.patch.object(
            mla, "TransferDirection", direction, create=True
        ), mock.patch.object(
            mla, "transfer_kv_dim_exchange", exchange, create=True
        ), tempfile.TemporaryDirectory() as directory:
            host.backup_from_device_all_layer(
                pool, torch.arange(64, 128), torch.arange(128, 192), "kernel_ascend"
            )
            backend = file_backend(directory, host)
            self.assertTrue(backend.set("tokenhash", host.get_data_page(64)))
            raw = Path(backend._get_component_path("tokenhash")).read_bytes()
            self.assertEqual(len(raw), 64 * host.size_per_token)
            self.assertEqual(
                raw[: expected_k.nbytes], expected_k.view(torch.uint8).numpy().tobytes()
            )
            self.assertEqual(
                raw[expected_k.nbytes :],
                expected_index.view(torch.uint8).numpy().tobytes(),
            )
            for _, tensor in host._split_page_components():
                tensor.zero_()
            data = backend.get("tokenhash", host.get_dummy_flat_data_page())
            host.set_from_flat_data_page(192, data)
            pool.k_buffer.zero_()
            pool.index_k_buffer.zero_()
            for layer in range(3):
                host.load_to_device_per_layer(
                    pool,
                    torch.arange(192, 256),
                    torch.arange(64, 128),
                    layer,
                    "kernel_ascend",
                )
            self.assertTrue(torch.equal(pool.k_buffer[:, 1], expected_k))
            self.assertTrue(torch.equal(pool.index_k_buffer[:, 1], expected_index))
            self.assertEqual(calls, [0, 1])

    def test_file_v2_roundtrip_and_legacy_namespace_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            host = host_pool(device_pool())
            host.k_buffer.fill_(-7)
            host.index_k_buffer.fill_(0.75)
            writer = file_backend(directory, host, tp_size=16)
            legacy_host = host_pool(device_pool("bf16"))
            legacy = file_backend(directory, legacy_host)
            self.assertTrue(legacy.set("same", legacy_host.get_data_page(0)))
            # Also seed an unversioned, same-sized file. It must never satisfy a hit.
            Path(directory, "same" + writer.config_suffix + ".bin").write_bytes(
                bytes(64 * host.size_per_token)
            )
            self.assertFalse(writer.exists("same"))
            self.assertEqual(writer.batch_exists_v2(["same"]).kv_hit_pages, 0)
            transfer = PoolTransfer(PoolName.KV, torch.arange(64), keys=["same"])
            self.assertEqual(writer.batch_set_v2([transfer])[PoolName.KV], [True])
            reader_host = host_pool(device_pool(), ratio=3)
            reader = file_backend(directory, reader_host, tp_rank=1, tp_size=8)
            self.assertEqual(
                reader._get_suffixed_key("same"), writer._get_suffixed_key("same")
            )
            self.assertEqual(reader.batch_get_v2([transfer])[PoolName.KV], [True])
            self.assertTrue(
                torch.equal(reader_host.get_data_page(0), host.get_data_page(0))
            )
            self.assertNotEqual(
                writer._get_suffixed_key("same"), legacy._get_suffixed_key("same")
            )
            with self.assertRaisesRegex(ValueError, "incompatible"):
                writer.register_mem_pool_host(legacy_host)

    def test_file_rejects_wrong_size_before_mutating_target(self):
        with tempfile.TemporaryDirectory() as directory:
            host = host_pool(device_pool())
            backend = file_backend(directory, host)
            target = host.get_dummy_flat_data_page().fill_(42)
            path = Path(backend._get_component_path("bad"))
            for delta in (-1, 1):
                path.write_bytes(bytes(target.numel() + delta))
                with self.assertRaisesRegex(ValueError, "page size mismatch"):
                    backend.get("bad", target)
                self.assertTrue(torch.all(target == 42))
            with self.assertRaisesRegex(ValueError, "contiguous CPU"):
                backend.get("bad", torch.empty((2, 3)).t())

    def test_malformed_pages_and_index_dtype_are_rejected(self):
        host = host_pool(device_pool())
        with self.assertRaises(TypeError):
            host.set_from_flat_data_page(
                0, torch.empty(64 * host.size_per_token, dtype=torch.int8)
            )
        with self.assertRaises(ValueError):
            host.set_from_flat_data_page(0, torch.empty(1, dtype=torch.uint8))
        with self.assertRaises(ValueError):
            host.get_data_page(1)
        with self.assertRaises(ValueError):
            host.get_page_buffer_meta(torch.arange(64).flip(0))
        pool = device_pool()
        pool.index_k_buffer = pool.index_k_buffer.to(torch.int8)
        with self.assertRaisesRegex(TypeError, "BF16 index"):
            host_pool(pool)
        pool = device_pool()
        pool.kv_cache_dim = 512
        with self.assertRaisesRegex(ValueError, "width 656"):
            host_pool(pool)
        pool = device_pool()
        pool.index_k_scale_buffer = torch.empty((2, 4, 64, 1))
        with self.assertRaisesRegex(ValueError, "must not have"):
            host_pool(pool)

    def test_fp8_index_scale_and_bf16_zero_rope_regressions(self):
        for encoding in ("fp8", "bf16"):
            with self.subTest(encoding=encoding):
                pool = device_pool(encoding)
                host = host_pool(pool)
                expected = (
                    3 * 656 + 2 * (128 + 4)
                    if encoding == "fp8"
                    else 3 * 512 * 2 + 2 * 128 * 2
                )
                self.assertEqual(host.size_per_token, expected)
                for name, target in host._split_page_components():
                    if target.numel() == 0:
                        continue
                    target[0].view(torch.uint8).reshape(-1).copy_(
                        getattr(pool, name)[:, 1]
                        .contiguous()
                        .view(torch.uint8)
                        .reshape(-1)
                    )
                data = host.get_data_page(0)
                host.set_from_flat_data_page(64, data)
                self.assertTrue(torch.equal(host.get_data_page(64), data))
                self.assertEqual(
                    sum(t.nbytes for _, t in host._split_page_components()),
                    host.size * expected,
                )
                meta = host._build_ascendc_component_meta(pool, 0, -1, 0, -1)
                self.assertEqual(len(meta), 3 if encoding == "fp8" else 2)
                if encoding == "fp8":
                    self.assertEqual(host.index_k_scale_buffer.dtype, torch.float32)
                    self.assertEqual(meta[-1][6], 64 * 4)

    def test_group_delegates_format_to_file_backend(self):
        host = host_pool(device_pool())
        group = HostPoolGroup(
            [PoolEntry(PoolName.KV, host, host.device_pool, lambda layer: layer)]
        )
        self.assertTrue(group.dsa_kv_cache_store_int8)
        self.assertTrue(group.dsa_kv_cache_store_quantized)
        self.assertFalse(group.dsa_kv_cache_store_fp8)
        self.assertEqual(group.get_storage_namespace(), host.get_storage_namespace())
        with tempfile.TemporaryDirectory() as directory:
            backend = file_backend(directory, host)
            backend.register_mem_pool_host(group)
            self.assertIn(
                host.get_storage_namespace(), backend._get_suffixed_key("same")
            )

    def test_memcache_component_keys_and_namespace(self):
        from sglang.srt.mem_cache.storage.npu_memcache.npu_memcache_store import (
            NpuMemcacheStore,
        )

        for encoding, count in (("int8", 2), ("fp8", 3), ("bf16", 3)):
            host = host_pool(device_pool(encoding))
            store = NpuMemcacheStore.__new__(NpuMemcacheStore)
            store.mem_pool_host = host
            store.is_mla_backend = True
            store.mla_suffix = "GLM5Next"
            store.extra_backend_tag = None
            tagged = store._tag_keys(["tokenhash"])
            keys, ptrs, sizes = store._get_mla_buffer_meta(tagged, torch.arange(64))
            self.assertEqual(len(keys), count)
            self.assertEqual(store._mla_key_multiplier(), count)
            self.assertIn(host.get_storage_namespace(), keys[0])
            self.assertEqual(sum(sizes), host.page_size * host.size_per_token)
            with mock.patch.object(
                store, "_batch_exist", return_value=[1] * count
            ) as exists:
                self.assertEqual(store.batch_exists(["tokenhash"]), 1)
                self.assertEqual(exists.call_args.args[0], keys)
            if encoding == "int8":
                self.assertTrue(keys[1].endswith("_index_k"))
                store.store = mock.Mock()
                store.register_buffer(host.v_buffer)
                store.store.register_buffer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
