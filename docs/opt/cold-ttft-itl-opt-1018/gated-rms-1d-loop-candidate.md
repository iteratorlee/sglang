# Gated RMSNorm existing 1D loop candidate — 2026-10-18

## Handoff

**Default off, CPU validated only. No NPU/GPU or remote/service operation.**
This is a separate alternative after the 2D R4/R8 candidate failed both its
FP32 diagnostic and its BF16-output contract. Those failures and source commits
remain intact. P 2D extension `90adc6d` remains research-only, not promoted.

- Worktree: `/home/iterator/workspace/e2e_opt/910b_pd_opt/repos/sglang-d-rms-loop-1018`
- Branch: `glm53-d-rms-loop-1018`, parent `e3179b6`.
- Author: `liyan <yan.l@pku.edu.cn>`.
- D flag: `SGLANG_GLM53_D_GATED_RMS_LOOP_ROWS=4` or `8`.
- P flag: `SGLANG_GLM53_P_GATED_RMS_LOOP_ROWS=4` or `8`.
- Other flag values are off. The loops require the matching `decode`/`prefill`
  role, NPU device, feature width 128, contiguous BF16 x/gate/output, and
  BF16/FP32 weight. D accepts 1024–2048 flattened rows. P accepts only
  `[16384,8,128]` or `[1,16384,8,128]` for each input, weight `[128]`, with
  physical ND format 2 for x/gate/weight/output. Unknown or other formats fall
  back. No storage-format conversion is added.
- The loop branch precedes the retained 2D branch. The test driver explicitly
  disables both rejected 2D flags. Keep them off in any subsequent deployment.
- No KDA/index body changes. Main, D 2D, BF16-v2 and P 2D worktrees are untouched.

## Minimal numerical change

**There is no new numerical kernel.** The candidate calls the existing
`_glm_rms_norm_gated_large_row_kernel` with `grid_rows=ceil(num_rows/R)`.
Both original small/large kernel bodies are unchanged. Each program still
executes a 1D 128-element vector and `sum(..., axis=0)` followed by the same
variance division, sqrt, division, weight multiply, exp and final division.
The existing `range(program_id, num_rows, grid_rows)` remains; it is not replaced
with `static_range`, 2D vectors, a new math function or new compiler options.
`num_warps=1`, `num_stages=1` are unchanged.

| Shape | Original | Loop R4 | Loop R8 |
| --- | --- | --- | --- |
| D1664 | small, grid 1664 | old large, grid 416 | old large, grid 208 |
| D1792 | small, grid 1792 | old large, grid 448 | old large, grid 224 |
| P131072 | old large, grid 32768, loop 4 | **identical original call** | old large, grid 16384, loop 8 |

All grids are legal. For any row r, its unique program is `r % grid_rows` and
loop iteration is `r // grid_rows`; every row is covered exactly once. The
existing loop bound handles non-divisible D shapes. At most R rows belong to
one program. P R4 is a useful identity/control, not a proposed speedup.

This removes the deliberate 2D/axis-1 layout change. It does **not** prove NPU
bitwise equality: the compiler can still transform loops and hoisted weights,
or select different math lowering. The strict NPU unit decides acceptance;
CPU comparisons are not a substitute. Performance is also unknown.

## Preserved failure evidence and cause limits

`d-gated-rms-unit-v1-result.json`: 1024 rows, BF16 weights, FP32 output,
random seed 53, eager R4, 8358 byte differences starting at row 23/column 0.
Eight BF16-weight/BF16-output cases completed before this failure.

`d-gated-rms-bf16-v2-unit-v1-result.json`: after 15 complete cases, 1024 rows,
FP32 weights, BF16 output, standard fixture (not midpoint), seed 101 wide,
eager R4 differed at row 928/column 97 by one byte:
`-3.055902197957039e-10` versus `-3.0377123039215803e-10`. The failing case's
R8/graph checks and subsequent midpoint fixtures were not reached. This is a
real violation of the candidate's advertised BF16-output/FP32-weight domain.

The evidence does not isolate sum versus sqrt/div/exp. At the latter coordinate,
the fixture recipe sets gate=-20. In a local CPU reconstruction, x=27.5 and
weight=-0.3400409519672394; evaluating the expression in double lies about
4.21e-7 relative from the midpoint of the two reported BF16 values. This is an
illustration of rounding-boundary sensitivity, not a target-device oracle or
proof that different Torch versions generated identical source tensors.

Local Triton 3.6 source expresses `sum` as a reduction and sqrt/exp as IR math
operations; source spelling does not fix their backend implementation. The
failed target uses Triton 3.2/CANN 9.0, so local compiler code cannot certify
its lowering. Original target IR/assembly or controlled stage probes are needed
to identify the cause. Instrumenting extra intermediate stores can itself
change optimization. Do not replace sqrt with rsqrt, exp with a different
sigmoid expression, or change fusion switches without a separate exact study.

## Read-only production weight dtype finding

The local v23 d0/d1 launch plans explicitly pass `--dtype bfloat16` and
`--quantization modelslim`, with source `dd2c99d` and default load format.
`DefaultModelLoader.load_model` constructs under
`set_default_torch_dtype(model_config.dtype)`. `Glm5Next` constructs `o_norm`
without an explicit dtype; `FusedRMSNormGated` creates its parameter with that
default. The normal `Glm5Next.load_weights` path uses `default_weight_loader`,
which copies into parameter storage without replacing its dtype. This norm has
neither a custom parameter loader nor a quant_method. The Deepseek post-load
path skips KDA attention without `kv_b_proj`; ModelSlim's ordinary RMSNorm
patch is for a different class.

A CPU extraction of the real constructor/loader confirmed BF16 parameter
creation and retention after loading an FP32 checkpoint tensor. Thus the
current configuration and standard source path support BF16 production weights;
there has been **no read of actual running per-layer parameter metadata**.
This is not an unconditional guarantee for other load paths or later casts.
An explicit BF16-weight-only guard could exclude the known FP32-weight failure,
but its whole admitted domain still needs NPU exact validation. No such 2D
guard change or new 2D measurement is made in this candidate.

## CPU checks

`sgl-0511`, `CUDA_VISIBLE_DEVICES='' TORCH_DEVICE_BACKEND_AUTOLOAD=0` throughout.
Six new tests pass: small shapes 0/1/7/8/9/31/32/33, both weight/store dtypes,
wide and midpoint inputs, unchanged inputs, output canaries, D/P role/flag/dtype
bounds, P ND guard, exact grid coverage and true large-kernel baseline routing.
AST checks prove both original kernel bodies and their per-row arithmetic
expressions unchanged. Four related default-path/empty regression tests pass.
P131072 is checked by metadata/monkeypatch only; no huge interpreter grid runs.
The strict NPU driver's syntax/`--help` is checked on CPU.

## Next main-agent NPU window

Package four files: new `test_glm53_gated_rms_loop_npu.py`, original strict
`test_glm53_gated_rms_rows_npu.py`, and both RMS source modules from this branch.
The BF16-v2 contract script is **not** needed for this strict driver. Do not use
that script as `--reference-harness`; an existing `/kit/run_npu.py` may now be
the BF16-v2 script rather than the original strict one. Use the explicit name
below to avoid mixing them. No changes to the preserved strict harness's
comparison functions or v1/v2 failure records are required.

```bash
# D card: strict BF16 and FP32 stores, including both weight dtypes.
env CUDA_VISIBLE_DEVICES='' TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  GLM53_GATED_RMS_SOURCE_DIR=/kit \
  python -B /kit/test_glm53_gated_rms_loop_npu.py \
  --reference-harness /kit/test_glm53_gated_rms_rows_npu.py \
  --role decode --device 0 --rows 1024,1663,1664,1665,1792,2048 \
  --timing-repeats 30 --cann-build 9.0.0 --output /unit/rms-loop-d.json

# Independent P card/process in the later authorized window.
env CUDA_VISIBLE_DEVICES='' TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  GLM53_GATED_RMS_SOURCE_DIR=/kit \
  python -B /kit/test_glm53_gated_rms_loop_npu.py \
  --reference-harness /kit/test_glm53_gated_rms_rows_npu.py \
  --role prefill --device 0 --rows 131072 \
  --timing-repeats 30 --cann-build 9.0.0 --output /unit/rms-loop-p.json
```

The driver reuses all original bytewise eager/graph, fixed-address replay,
input-immutability and tail-canary checks. Only candidate dispatch, true large
baseline selection, role/model shape and launch metadata are adapted. BF16
wrapper capture observes the actual large kernel/grid used. FP32 output is a
strict diagnostic, not bypassed. There are 32 fixtures per shape, using the
original seed pair and random/zero/signed-zero/wide patterns. FP32 inputs are
not claimed; x/gate remain BF16 as in the production contract and original unit.

Any mismatch fails the run and retains partial results. Optional timing uses
random data after exact checks for each dtype, rotating variant order, 20 graph
replays per sample, raw samples and medians. It excludes tracing and host
copies, but can include replay gaps; it is not a model wall-time prediction.
Choose D/P enablement only after their separate strict acceptance and timing.
