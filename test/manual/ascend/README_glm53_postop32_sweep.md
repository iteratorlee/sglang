# P32 postop BLOCK sweep (harness only)

Base: f630cc8. Production P22 logs established scores_shape=(32,32,32768)
and the query-TP branch active on ranks0..7. The prior 64-head units establish
only that synthetic geometry; P64 remains off and has no claimed P benefit.

This change touches only harness/tests/documentation. It sweeps the existing
helper's permitted BLOCK=2048/4096/8192, retaining num_warps=4, kernel body,
native GEMM, BF16 storage, native head reduction and TopK. No production P32
threshold or default changes: tile32>=32768, tile8>=98304 pooled keys.

Bundle the following three files from the same source into /kit (the helper
and caller can be the parent's deployed source; JSON binds their SHA256):

- python/sglang/srt/hardware_backend/npu/attention/glm53/kpool_indexer.py
- python/sglang/srt/hardware_backend/npu/attention/glm53/pooled_score_postop_npu.py
- test/manual/ascend/bench_glm53_postop64_ab.py

Only the parent runs this in an isolated window after stopping P:

```sh
timeout 900s python3 -B /kit/bench_glm53_postop64_ab.py \
  --module-file /kit/kpool_indexer.py \
  --helper-file /kit/pooled_score_postop_npu.py \
  --heads 32 --blocks 2048 4096 8192 \
  --tiles 8 32 --keys 8192 16384 32768 \
  --device npu:0 --confirm-parent-stopped-p \
  --repeat 10 --warmup 2 --output /unit/postop32-blocks-v1.json
```

The direct baseline is explicitly native relu_()+mul_(); POSTOP64=0 alone
cannot disable the 32-head helper. Candidate direct calls must return True.
The real production scorer is AST-extracted without SGLang registration.
Only its admission-policy binding is replaced in the harness namespace:
native=None, candidate=1. The scorer's GEMM/reduction body is unchanged.
This allows small-key evaluation without changing a production threshold.
Each candidate caller must invoke the requested BLOCK exactly once, returning
active=True with the expected [rows,32,keys] ND geometry. Fallback fails the
run. This is labeled harness-forced admission, never production auto activation.
P64 stays 0 throughout and any inherited P64 threshold override is cleared.

Default sweep: 54 direct cases (3 BLOCKs x 2 tiles x 3 key counts x 3 patterns)
and 18 real-caller cases. Direct checks compare weighted BF16 bytes, masked
native head sums and ordered INT32 TopK, including signed zero, nonfinite
payloads, rounding neighbors and rank ties. Caller checks use real BF16 GEMM
and FP32 weights, comparing masked logits and final ordered IDs. CPU copies
are row-streamed for QHK; no full-query tape or tensor files. Cases release
their tensors before proceeding; the largest individual score tile is 64 MiB.

All 72 correctness cases complete before 18 independent timing cases. For
each shape/BLOCK, alternating native/fused A/B reports separate synchronized
host and NPU event samples: (1) reset+postop, identical reset copy included;
(2) extracted caller with forced admission. Timed work has no comparison or
logit-copy observer; every fused timed invocation still checks actual launch.
These are operator/caller timings, not TTFT. --repeat 0 runs correctness only.
The output path is exclusive-create and failure retains the active case.

CPU guard command (no Triton interpreter):

```sh
CUDA_VISIBLE_DEVICES='' TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  /home/iterator/anaconda3/envs/sgl-0511/bin/python -B \
  test/manual/ascend/test_glm53_postop32_sweep_cpu.py -v
```

No NPU run or production enablement is claimed by this commit. A lower P32
threshold remains a separate decision requiring actual launch, strict equality
and measured caller break-even. No end-to-end semantic/accuracy result is
implied; old GSM1264/1319<1267 and the rejected approximate Sinkhorn remain.
