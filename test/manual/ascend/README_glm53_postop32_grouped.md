# Default-off P32 grouped postop candidate

Based on f630cc8 and the separate P32 BLOCK harness commit. P22 production
logs establish 32 indexer heads. P64 remains off; its prior unit speedups are
not production P speedups.

The new kernel flattens (qhead,keyblock) into block_id. For G=4/8 the physical
grid is ceil(total_blocks/G), and each program loops over
range(pid,total_blocks,grid). Every load, FP32 expression, BF16 cast and masked
store is copied from the old BLOCK8192 kernel. CPU AST tests require these
five statements to be identical. No reduction, GEMM, head sum or TopK changes;
num_warps=4 and multibuffer=False remain fixed. At rows32/heads32/keys32768,
4096 logical blocks become 1024 or 512 physical programs. This is a scheduling
candidate, not an established speedup or universal bitwise proof.

Production admission requires NPU, prefill role, 32 heads and both switches:

```sh
SGLANG_GLM53_PREFILL_POOLED_POSTOP=auto
SGLANG_GLM53_PREFILL_POOLED_SCORE=0
SGLANG_GLM53_PREFILL_POOLED_POSTOP64=0
SGLANG_GLM53_PREFILL_POOLED_POSTOP32_GROUP=4  # or 8; default 0
```

With GROUP=0/unset, all existing P32/P64 policies and legacy launches remain.
Only GROUP=4/8 selects a conservative 32768-pool threshold for tile8/32.
An experiment-only POSTOP32_MIN_KEYS integer in [1,32768] can override that
threshold; it is ignored outside explicitly enabled NPU P32. Do not use a
lower threshold before exactness and measured caller benefit. The helper keeps
ND/BF16 scores, contiguous FP32 gates, rows<=32 guards, and bounds grouped
launches to BLOCK8192 and keys<=32768. Outside that envelope it performs native
ReLU/mul once and returns False. 64-head calls never use the grouped kernel.

## Parent-run isolated NPU acceptance

Bundle these four files into /kit, preserving their basenames:

- python/sglang/srt/hardware_backend/npu/attention/glm53/kpool_indexer.py
- python/sglang/srt/hardware_backend/npu/attention/glm53/pooled_score_postop_npu.py
- test/manual/ascend/bench_glm53_postop64_ab.py
- test/manual/ascend/bench_glm53_postop32_grouped_ab.py

```sh
timeout 900s python3 -B /kit/bench_glm53_postop32_grouped_ab.py \
  --module-file /kit/kpool_indexer.py \
  --helper-file /kit/pooled_score_postop_npu.py \
  --base-harness-file /kit/bench_glm53_postop64_ab.py \
  --device npu:0 --confirm-parent-stopped-p \
  --tiles 8 32 --keys 8192 16384 32768 \
  --repeat 10 --warmup 2 --output /unit/postop32-grouped-v1.json
```

The harness imports no SGLang registry and never controls services/resources.
The exclusive-create JSON includes all four source hashes and flushes the
active case before execution. The complete default correctness phase includes:

- 54 direct cases: legacy/group4/group8 x six geometries x random/edge-bits/ties.
  Native ReLU/mul is the reference. Weighted BF16, native masked head sums and
  ordered INT32 TopK must be byte-identical, including NaN payloads and ±0.
- 18 extracted real-caller cases force admission only in the test namespace,
  retaining actual GEMM/headsum/TopK. Actual helper call count and exact kernel
  identity/grid are checked, so legacy or native fallback cannot pass as grouped.
- 12 real production-policy cases, with the conservative threshold unchanged;
  only keys32768 may launch by default, for tile8 and tile32. Optional --min-keys
  8192/16384 exercises the real 32-only override after a profitable micro result.
- 24 NPUGraph cases across native/legacy/group4/group8 and all six geometries.
  Each captures reset+postop+native sum/mask/TopK; random then edge-bit score AND
  gate contents change after capture. Both replays compare BF16, logits and IDs
  against native eager. These graphs do not claim full-service graph coverage.

No timing starts until all 108 selected correctness records pass. Six timing
records then compare native, legacy, group4 and group8 independently with
rotating order and synchronized host/NPU event clocks, reporting raw samples
and medians. Scopes: identical reset+postop; actual extracted caller with test
admission forced. Timing has no probe, tensor comparison or CPU logit copy;
active=True is still required for every fused call. Graph capture/replay above
is correctness-only. No automatic policy promotion or end-to-end TTFT claim.

Each case frees its tensors; no full-query QHK tape or tensor files are written.
Largest individual score tensor is 64 MiB. Resident clones/graph buffers and
native CANN workspace are additional; unit does not establish service HBM peak.

## CPU verification

```sh
CUDA_VISIBLE_DEVICES='' TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  /home/iterator/anaconda3/envs/sgl-0511/bin/python -B \
  -m unittest discover -s test/manual/ascend -p 'test_glm53_postop*_cpu.py' -v
```

CPU tests cover P-role/head/override scope, default preservation, launch grids,
native fallback ownership, exact arithmetic-body AST identity and small CPU
grid-stride ownership including tails. No Triton interpreter or NPU run here.
NPU compilation, strict equality and performance remain for the parent window.
Old GSM1264/1319<1267 and rejected approximate Sinkhorn42/300 remain failures.
