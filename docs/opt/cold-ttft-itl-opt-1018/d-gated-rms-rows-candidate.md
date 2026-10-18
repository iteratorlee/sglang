# D gated RMSNorm grouped-row candidate — 2026-10-18

## Status and scope

**Default off; CPU validated only. Target NPU compilation, byte-exact acceptance,
and timing are pending.** No NPU, GPU, remote service, or main worktree was used
or changed. This is independent of the delivered index-batch candidate and the
subsequent padding diagnosis. No KDA body, P optimization, or indexer is edited.

- Worktree: `/home/iterator/workspace/e2e_opt/910b_pd_opt/repos/sglang-d-rms-grid-1018`
- Branch: `glm53-d-rms-grid-1018`
- Parent: `83cfe9668fb98ca674c3a28816dd99016e38b8b6`
- Author: `liyan <yan.l@pku.edu.cn>`
- Opt-in: `SGLANG_GLM53_D_GATED_RMS_ROWS=4` or `8`. All other values are off.
- Dispatch requires NPU, disaggregation mode `decode`, last dimension 128,
  1024–2048 flattened rows, contiguous BF16 x/gate and BF16 or FP32 weight,
  matching devices, and 128 weights. Unsupported cases use the existing path.
- Set the flag before model graph capture. Changing it cannot alter a graph
  already captured; recapture is required.

The original small-row and large-row kernel bodies are unchanged, checked by
AST comparison with the parent. The original empty-input return is preserved.
The RMSNorm file is identical between the running D source `32ffc6a3` and the
parent `83cfe9668`; this is not a whole-model source equivalence claim.

## Proposed change and numerical risk

The supplied profile attributes about 118 us to one original row-kernel call
at 52 tokens × 32 heads × 128 features. Its grid is 1664 programs, each handling
128 elements. The new kernel handles `[R, 128]` with R=4 or R=8, using 416 or
208 programs for the same workload. The final partial group masks all loads
and stores by row. The 128 weights are loaded once per program and broadcast.

Each row retains the original FP32 conversion and expression order:

```python
variance = sum(values * values, axis=features) / 128
normalized = values / sqrt(variance + eps)
result = normalized * weights / (1.0 + exp(-gates))
```

There is no rsqrt substitution, explicit FMA change, sigmoid rewrite, reordered
division, or cross-row reduction. Launch options remain `num_warps=1` and
`num_stages=1`. The new implementation lives in `rms_norm_gated_rows_npu.py`;
the existing wrapper only adds the guarded dispatch.

**Matching source expressions do not prove bitwise NPU equality.** A 2D layout
can change reduction lowering, vectorization, or fusion on Triton Ascend/CANN.
It can also increase register/UB usage or reduce useful parallelism. The
scheduling-overhead hypothesis is plausible, but no speedup is asserted. Any
NPU bit difference rejects the tested candidate; do not substitute allclose,
ignore signed zero, or accept agreement after only BF16 rounding. Both BF16
production stores and unrounded FP32 diagnostic stores are checked.

## CPU validation

Environment: `sgl-0511`, torch `2.11.0+cu130`, Triton `3.6.0`,
`CUDA_VISIBLE_DEVICES='' TORCH_DEVICE_BACKEND_AUTOLOAD=0`. The CPU entrypoint
sets `TRITON_INTERPRET=1` before importing Triton; importing it first in compiled
mode is insufficient for interpreter reductions.

**6 new tests passed (90.894 s), plus 2 existing empty-input tests.** The new
suite exercises the actual original and grouped Triton kernels in the CPU
interpreter across 43 data/shape/dtype combinations, comparing both R=4 and
R=8 byte-for-byte. Coverage includes:

- Zero and short row counts, partial groups, 1664 target rows, 1663/1665
  neighbors, 1792 rows and 1024/2048 dispatch boundaries.
- BF16/FP32 weights and BF16/FP32 stores; random inputs, zeros, signed zeros,
  wide magnitudes and positive/negative saturated sigmoid inputs.
- Output tail canaries and unchanged input bytes.
- Monkeypatched production dispatch: default off, malformed flags, CPU, P,
  non-D, noncontiguous input, wrong feature width, out-of-range rows, old
  large-row fallback and empty domains.
- Original kernel bodies unchanged relative to the parent commit.

```bash
cd /home/iterator/workspace/e2e_opt/910b_pd_opt/repos/sglang-d-rms-grid-1018
env CUDA_VISIBLE_DEVICES='' TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  /home/iterator/anaconda3/envs/sgl-0511/bin/python -B \
  test/registered/unit/hardware_backend/npu/test_glm53_gated_rms_rows_cpu.py -v
env CUDA_VISIBLE_DEVICES='' TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  /home/iterator/anaconda3/envs/sgl-0511/bin/python -B \
  test/registered/unit/disaggregation/test_glm53_empty_rms_norm.py -v
```

The standalone NPU script was syntax/`--help` checked on CPU only. These CPU
results are not evidence of target-device numerical equality or performance.

## Isolated NPU handoff

Run only in the main agent's next free device window, in the target TorchNPU /
Triton Ascend / CANN environment. No server or model startup is required. Keep
the script and the two RMSNorm source files at their repository-relative paths.
The standalone loader removes framework imports and `input_guard`, and supplies
a decode runtime-context stub; production kernel bodies, wrapper dispatch and
tensor guards are executed. This does not exercise service initialization or
the external `input_guard` decorator.

```bash
# First compile + exact-check the observed D shape (no timing by default).
env CUDA_VISIBLE_DEVICES='' TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  python -B test/manual/ascend/test_glm53_gated_rms_rows_npu.py \
  --device 0 --rows 1664 --cann-build 'EXACT_INSTALLED_BUILD' \
  --output /tmp/glm53-gated-rms-rows-smoke.json

# Full boundary matrix and rotating-order graph timing.
env CUDA_VISIBLE_DEVICES='' TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  python -B test/manual/ascend/test_glm53_gated_rms_rows_npu.py \
  --device 0 --cann-build 'EXACT_INSTALLED_BUILD' --timing-repeats 30 \
  --output /tmp/glm53-gated-rms-rows-full.json
```

Default rows are `1,7,8,9,32,64,1024,1663,1664,1665,1792,2048`. Direct kernel
comparisons intentionally include shapes outside production dispatch to test
partial groups. Every shape tests both weight/store dtypes, two seeds and four
patterns: 32 fixture cases per shape, each comparing original/R4/R8 in eager
and NPUGraph replay. Captured inputs are modified in place between fixtures;
pointer stability, unchanged inputs and output tail canaries are checked.
BF16 fixtures additionally capture the real wrapper with flag 0/4/8, including
model-shaped `[1,T,32,128]` tensors where possible. An exception fails the run
and writes the active case and traceback to JSON; no mismatch is skipped.

Optional timing runs only at 1664/1792 rows with BF16 outputs, after that case's
exact checks. It reports device-event time per graph replay, 20 replays per
window, with variant order rotated across windows, raw samples and medians.
There are no state traces, clones or host transfers in the timed interval.
These measurements can include graph replay gaps and are not pure kernel time
or model wall latency. Report target compilation/runtime versions and source
hashes alongside results; profile the integrated model separately before
claiming TTFT/ITL benefit. Overlapping profile categories must not be summed
into a wall-time estimate.
