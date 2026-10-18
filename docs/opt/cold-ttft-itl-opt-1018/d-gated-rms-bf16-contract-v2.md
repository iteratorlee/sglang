# D gated RMSNorm v1 failure and scoped v2 — 2026-10-18

## Status

**v1 remains FAILED. Production usability is unproven; keep the D flag off.**
P extension `90adc6d` is paused and is not part of this D-only change.
No NPU/GPU, remote operation or main-worktree edit was performed here.

Worktree: `/home/iterator/workspace/e2e_opt/910b_pd_opt/repos/sglang-d-rms-bf16-1018`.
Branch: `glm53-d-rms-bf16-1018`, parent `53718c5`, author
`liyan <yan.l@pku.edu.cn>`. The known packaged-path fix from `dd2c99d` is
included identically; the original strict harness's functions are unchanged.

## What the receipt establishes

Local evidence:
`results/cold-ttft-itl-opt-1018/d-gated-rms-unit-v1-result.json`.
Ascend910B4-1, torch/torch_npu 2.10.0, Triton 3.2.0, CANN 9.0.0.

- Eight 1024-row fixtures passed with BF16 inputs/weights/output: seeds 53/101,
  random/zero/signed-zero/wide, R4/R8 eager/graph/wrapper comparisons.
- The next case, BF16 weights / FP32 output / random seed 53, failed at
  **eager R4 versus original**: 8358 different bytes, first byte 11776.
- For a `[1024,128]` FP32 tensor, that first byte is row 23, column 0. The
  receipt does not contain values, ULP distances or compiler artifacts.
- That failing case's R4 graph and R8 comparisons were not reached. FP32
  weights, later row counts and timing were not reached either. There is no
  basis to claim the whole production domain passes.

Different FP32 reduction trees, scalar versus vector sqrt/div/exp lowering,
or output-dtype-dependent compilation are plausible. A first mismatch at a
row boundary is compatible with a changed row normalization factor, but does
not identify the operator. Illustratively, on the same CPU-generated BF16
seed-53 inputs, adjacent-pair and half-vector FP32 sum trees produce row-23
sum bits `0x42f1e31d` and `0x42f1e31e`; 329 of 1024 rows differ between those
illustrative trees. **Neither tree is asserted to be the actual NPU lowering.**
Saved outputs and original compiled IR/assembly are needed to refine the cause.
Even differing FP32 results that round identically in these eight BF16 cases
can cross a BF16 rounding midpoint for other inputs or weights.

## Explicit production contract

The existing helper already implied BF16 output: x/gate must be BF16, and
output dtype must equal x dtype. This revision makes the output check explicit
in both D wrapper dispatch and the grouped helper. FP32 output cannot enter
the production helper. This is a contract clarification/defense, **not an
arithmetic fix** or a proof of BF16 equality. All three kernel bodies and their
compile options remain unchanged; direct FP32 diagnostic launches are still
possible, and the old strict harness still fails on a reproduced difference.

## New explicit-contract unit

`test/manual/ascend/test_glm53_gated_rms_contract_npu.py` requires one of two
contracts. Every JSON includes its contract and the prior v1 FP32 failure.
`passed` applies only to the named contract. No tolerance or mismatch skip is
introduced. Both contracts exit nonzero on any within-contract byte mismatch.

`production-bf16` retains the original eager/graph/wrapper, fixed-address,
input-immutability and tail-canary checks, always with BF16 output. It tests
both BF16/FP32 weights and three seed offsets (0,997,2027), in addition to the
v1 seed pair and four patterns. Extra FP32-weight fixtures target the BF16
rounding midpoint between 1.0 and the next BF16 value at row 23, then perturb
the weight by one FP32 ULP in each direction. CPU double arithmetic only
constructs inputs; NPU original-versus-candidate bytes decide acceptance.
Zero rows remain valid zero fixtures. Midpoint fixtures are never timed.
The default six row counts give 432 fixture records if all pass. First failure
is preserved with byte/row statistics and both output tensors.

`fp32-diagnostic` reproduces seed 53 with BF16 weights and both BF16/FP32
outputs. It records all R4/R8 eager and graph differences plus graph/eager
agreement before failing, so an R4 failure cannot hide R8 evidence. It saves
input/output tensors and available compiled-kernel `asm` artifacts with hashes
for both output specializations. Empty artifact maps mean that this backend
did not expose them, not that lowering is equivalent. Diagnostic statistics
include changed rows/elements/bytes, first values and finite FP32 ULP distance;
none is used as a tolerance. No diagnostic timing is performed.

## Main-agent next NPU window

Package **four files**: the new unit, the original strict unit as its reference
harness, and both RMS source modules from this branch. `--reference-harness`
accepts the existing `/kit/run_npu.py` name. Keep the source-dir environment.

```bash
env CUDA_VISIBLE_DEVICES='' TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  GLM53_GATED_RMS_SOURCE_DIR=/kit \
  python -B /kit/test_glm53_gated_rms_contract_npu.py \
  --reference-harness /kit/run_npu.py --contract production-bf16 \
  --rows 1024,1663,1664,1665,1792,2048 --device 0 --timing-repeats 30 \
  --cann-build 9.0.0 --output /unit/d-rms-bf16-v2.json

env CUDA_VISIBLE_DEVICES='' TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  GLM53_GATED_RMS_SOURCE_DIR=/kit \
  python -B /kit/test_glm53_gated_rms_contract_npu.py \
  --reference-harness /kit/run_npu.py --contract fp32-diagnostic \
  --rows 1024 --device 0 --cann-build 9.0.0 \
  --output /unit/d-rms-fp32-diagnostic.json
```

A production-only pass does not supersede v1 or establish FP32/internal exactness.
Any BF16 failure rejects the candidate. No numerical or performance claim is
made until the main agent runs and reviews the target-device results.

## Local checks

CPU uses `sgl-0511` with CUDA visibility disabled and backend autoload disabled.
Ten selected tests passed. Four new tests check explicit output rejection, signed-zero/ULP diagnostics,
finite midpoint fixtures with unchanged x/gate bytes, and unchanged v1 strict
functions/kernel bodies. Existing short-row, helper, dispatch and empty tests
are rerun. No large-shape interpreter or device test is used for this revision.
