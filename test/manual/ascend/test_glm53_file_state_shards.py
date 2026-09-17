"""CPU file-backend regression: replicated MLA KV and rank-sharded SSM state."""
import argparse
import importlib.util
import sys
import tempfile

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage-module", required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("glm53_storage_test", args.storage_module)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    with tempfile.TemporaryDirectory() as directory:
        def backend(rank, size=16):
            cfg = mod.HiCacheStorageConfig(
                tp_rank=rank, tp_size=size, pp_rank=0, pp_size=1,
                attn_cp_rank=0, attn_cp_size=1, is_mla_model=True,
                enable_storage_metrics=False, is_page_first_layout=True,
                model_name="glm53",
            )
            return mod.HiCacheFile(cfg, file_path=directory)

        ranks = [backend(i) for i in range(3)]
        kv = torch.arange(64, dtype=torch.uint8)
        states = [torch.full((64,), i + 17, dtype=torch.uint8) for i in range(2)]
        assert ranks[0].set("prefix", kv)
        for rank in (0, 1):
            assert ranks[rank].set("prefix.mamba", states[rank])
        for rank in (0, 1):
            assert torch.equal(ranks[rank].get("prefix", torch.empty_like(kv)), kv)
            assert torch.equal(ranks[rank].get("prefix.mamba", torch.empty_like(kv)), states[rank])
            transfer = mod.PoolTransfer(name=mod.PoolName.MAMBA, hit_policy=mod.PoolHitPolicy.TRAILING_PAGES)
            hit = ranks[rank].batch_exists_v2(["prefix"], [transfer])
            assert hit.kv_hit_pages == 1, hit
        assert ranks[2].get("prefix.mamba", torch.empty_like(kv)) is None
        assert backend(0, 8).get("prefix.mamba", torch.empty_like(kv)) is None
        transfer = mod.PoolTransfer(name=mod.PoolName.MAMBA, hit_policy=mod.PoolHitPolicy.TRAILING_PAGES)
        assert ranks[2].batch_exists_v2(["prefix"], [transfer]).kv_hit_pages == 0
        print({"result": "PASS", "shared_mla_kv": True, "independent_tp_states": True,
               "missing_rank_misses": True, "different_tp_size_misses": True})


if __name__ == "__main__":
    main()
