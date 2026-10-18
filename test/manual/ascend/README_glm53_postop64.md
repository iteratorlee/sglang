# P 64-head ReLU/gate candidate (default off)

Base: production P source 29c9d0e. This changes only the pooled-postop policy,
helper admission, and its standalone tests. It does not change the Triton
arithmetic kernel, native GEMM/query tiles, BF16 storage boundaries, native head
sum, causal mask or TopK. It does not enable the rejected full fused scorer.

P opt-in requires both existing POSTOP=auto and the new explicit flag:

```sh
SGLANG_GLM53_PREFILL_POOLED_SCORE=0
SGLANG_GLM53_PREFILL_POOLED_POSTOP=auto
SGLANG_GLM53_PREFILL_POOLED_POSTOP64=1
```

Only the P caller selects this policy. The default/unset new flag preserves the
old 32-head thresholds and rejects 64-head helper launches. The helper retains
ND format2, BF16 scores, contiguous FP32 gates, <=32 rows, and launch-grid guards.
Unsupported helper calls execute native ReLU/gate exactly once and return False.

Initial **experimental**, unmeasured 64-head thresholds:

| legacy query tile | 32-head policy (unchanged) | opt-in 64-head policy |
|---|---:|---:|
| 32 rows | 32768 pools | 16384 pools |
| 8 rows | 98304 pools | 49152 pools |

These preserve rows*heads*pools, and therefore score bytes and BLOCK8192 program
count, at the old 1012 empirically selected thresholds. This is a workload-size
heuristic, not evidence that 64-head latency scales linearly or is faster.
Tile8 remains native for a 128K-token history (32768 pools). The direct helper
tests/timings deliberately also evaluate smaller keys for a later informed
decision; they do not relax the production policy or change GEMM shapes.

An optional experiment-only override is now available:

```sh
SGLANG_GLM53_PREFILL_POOLED_POSTOP64_MIN_KEYS=8192
```

It is parsed only for NPU P, 64 heads, POSTOP64=1 and POSTOP=auto/1. It accepts
an integer in [1,49152], applies to tile8/32, and rejects malformed/zero/negative
or over-bound values in that scope. It is ignored for 32 heads, flag0, CPU and
D. Unset preserves the conservative defaults above; unsupported tiles stay
native. The upper bound limits this experimental threshold knob; it does not
replace the helper's independent tensor/physical-format/grid safety guards.
Select 8192/16384/32768 only from measured 64-head micro break-even results.

## Bundle and run in the parent's isolated P window

Copy these three files, preserving only the names, into /kit:

- python/sglang/srt/hardware_backend/npu/attention/glm53/kpool_indexer.py
- python/sglang/srt/hardware_backend/npu/attention/glm53/pooled_score_postop_npu.py
- test/manual/ascend/bench_glm53_postop64_ab.py

No SGLang package registration or full-score module is imported. The harness
loads the real helper and AST-extracts the real production caller/guards. Use
the same Torch/torch_npu/Triton/CANN image as the service. The parent must have
stopped P and isolated the selected NPU before making the explicit assertion:

```sh
python3 -B /kit/bench_glm53_postop64_ab.py \
  --module-file /kit/kpool_indexer.py \
  --helper-file /kit/pooled_score_postop_npu.py \
  --device npu:0 --confirm-parent-stopped-p \
  --force-min-keys 8192 --repeat 10 --warmup 2 --output /unit/postop64-v2.json
```

The output is created exclusively: choose a new path for each attempt. JSON is
flushed before/after each case and on failure; source hashes, actual formats,
helper call counts/activation, byte differences/hashes and timing samples are
retained. The script never starts/stops/checks services. No confirmation means
failure JSON and refusal before any device-library import. A 900s outer timeout
may be used by the parent; process timeout leaves the last active case in JSON.

Default correctness: 18 direct 64-head cases (tile8/32 x keys8192/16384/32768 x
random/edge-bits/rank-boundary), six real-production auto-policy cases, one
unchanged-32-head case, and five actual device fallback cases. Edge inputs cover
signed zero, nonfinite payloads, BF16 storage patterns and selected FP32 rounding
neighbors; rank-boundary cases include ties and cancellation. Every direct case
compares weighted BF16 bytes, native masked logits and ordered INT32 TopK.
The production cases independently execute native GEMM and compare masked
logits/IDs. No tolerances or sorting are allowed, including nonfinite outputs.

The direct helper MUST fuse at flag1 at all six geometries; auto-policy cases
must have exactly the expected call/activation counts. A fallback is reported
as fallback, never as fusion parity. Default auto-positive 64-head cases are
tile32 at 16384 and 32768 pools. Optional --keys 49152 covers the first tile8
auto-positive point; it is outside this task's 128K workload, not a default case.

With --force-min-keys 8192, six additional real-production forced-caller cases
MUST each record exactly one flag1 helper call with active=True, plus exact
masked logits/INT32 IDs. Missing calls or fallback fail the run. The flag0 side
receives the same override but must ignore it. Auto tests always clear inherited
overrides; the three scopes (direct helper, conservative auto, forced caller)
are reported separately. The CLI requires the forced threshold <= every selected
key count, so an accidentally non-activating forced case is rejected before NPU
imports. To test threshold16384, select --keys 16384 32768; threshold32768 uses
--keys 32768. Omitting --force-min-keys retains the original auto-only matrix.

Only after all correctness passes, run independent alternating A/B pairs with
NPU events and synchronized host clocks. Separate scopes are labeled
reset+postop (same reset copy included in both) and real production auto caller.
When requested, a third timing scope measures the real forced production caller
with the same validated override, only after all forced correctness cases pass.
No observation/D2H comparison is timed. --repeat 0 runs correctness only.
These are operator/caller timings, not cold TTFT, and an inactive auto case
does not measure a fusion benefit. The largest resident score is one 32-row
64-head tile (128 MiB at 32768 keys), not an entire T2048 tape. Row-streamed
CPU comparison and per-case reclamation avoid storing full intermediate tapes.

## CPU guard tests

```sh
CUDA_VISIBLE_DEVICES='' TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  /home/iterator/anaconda3/envs/sgl-0511/bin/python -B \
  test/manual/ascend/test_glm53_postop64_cpu.py -v
```

CPU tests only check guards, thresholds, fallback ownership, launch geometry,
real caller head-count plumbing and the byte-comparison oracle. They do not
run Triton/interpreter or establish NPU parity. NPU and end-to-end validation
remain pending. Old GSM1264/1319<1267 and rejected approximate Sinkhorn42/300
remain failures; this candidate does not fix or replace those results.
