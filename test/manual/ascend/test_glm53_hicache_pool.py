"""Real Ascend page remapping + file roundtrip, without loading model weights.

python test/manual/ascend/test_glm53_hicache_pool.py --pool-module PATH
"""
import argparse
import importlib.util
import tempfile
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-module", required=True)
    args = parser.parse_args()
    torch.npu.set_device(0)
    spec = importlib.util.spec_from_file_location("glm53_test_host_pool", args.pool_module)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def pool(layers, index_layers):
        shape = (layers, 25, 64, 1, 32)
        k = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape)
        k = (k % 251).to(torch.bfloat16).npu()
        ishape = (index_layers, 25, 64, 1, 16)
        index = torch.arange(torch.tensor(ishape).prod().item(), dtype=torch.float32).reshape(ishape)
        return SimpleNamespace(
            size=1536, page_size=64, device="npu", store_dtype=torch.bfloat16,
            kpool_use_compress=True, index_kpool=4, zero_rope=True,
            layer_num=layers, start_layer=0, end_layer=layers,
            k_buffer=k, index_k_buffer=(index % 127).to(torch.bfloat16).npu(),
        )

    target, draft = pool(3, 2), pool(1, 1)
    cache = mod.GLM53MLAPoolHost(
        target, 2, 0, 256, "page_first_kv_split", mtp_draft_device_pools=(draft,),
    )
    offsets = torch.arange(64, dtype=torch.int64)
    src_pages = torch.tensor([2, 6, 3, 8, 9, 4, 13, 5], dtype=torch.int64)
    dst_pages = torch.tensor([11, 1, 17, 7, 14, 12, 10, 16], dtype=torch.int64)
    source = (src_pages[:, None] * 64 + offsets).flatten()
    dest = (dst_pages[:, None] * 64 + offsets).flatten()
    host = torch.cat((torch.arange(512, 768), torch.arange(1280, 1536)))
    expected = [(p.k_buffer[:, src_pages.npu()].cpu(), p.index_k_buffer[:, src_pages.npu()].cpu())
                for p in (target, draft)]
    cache.backup_from_device_all_layer(target, host, source, "kernel_ascend")
    torch.npu.synchronize()
    # Simulate L2 eviction: persist actual bytes, erase every host buffer,
    # then recover at DIFFERENT host locations before device remapping.
    recovered = torch.arange(512, dtype=torch.int64)
    with tempfile.TemporaryDirectory() as directory:
        from pathlib import Path
        for i, start in enumerate(host[::256]):
            Path(directory, str(i)).write_bytes(cache.get_data_page(start).numpy().tobytes())
        for buf in cache._components:
            buf.zero_()
        for i in range(2):
            data = torch.frombuffer(bytearray(Path(directory, str(i)).read_bytes()), dtype=torch.uint8)
            cache.set_from_flat_data_page(i * 256, data)
        for p in (target, draft):
            p.k_buffer.zero_()
            p.index_k_buffer.zero_()
        cache.load_to_device_per_layer(target, recovered, dest, 0, "kernel_ascend")
        cache.load_to_device_per_layer(draft, recovered, dest, 3, "kernel_ascend", is_draft=True)
        torch.npu.synchronize()
    for p, (k, index) in zip((target, draft), expected):
        assert torch.equal(p.k_buffer[:, dst_pages.npu()].cpu(), k)
        assert torch.equal(p.index_k_buffer[:, dst_pages.npu()].cpu(), index)
    print({"result": "PASS", "logical_page": 256, "physical_page": 64,
           "noncontiguous_physical_pages": True, "host_remap": True,
           "file_roundtrip": True, "target_and_draft_exact": True})


if __name__ == "__main__":
    main()
