# D KDA existing head-loop candidate — 2026-10-18

Default off; CPU checks only. No GPU/NPU or remote/service operation. The main
worktree, RMS, Kpool and pointwise candidates are untouched.

- Parent: `f3260b0281b0c55272fea180092a763958ccefdb`.
- Branch: `glm53-d-kda-bhv-1018`.
- Worktree: `/home/iterator/workspace/e2e_opt/910b_pd_opt/repos/sglang-d-kda-bhv-1018`.
- Author: `liyan <yan.l@pku.edu.cn>`.
- Flag: `SGLANG_GLM53_D_KDA_BHV=2` or `4`. All other values, including unset,
  retain BHV1. This is an experimental dispatch flag, not an enablement recommendation.

## Change and callsite contract

The original `_glm_kda_varlen_recurrent_kernel` already loops over
`range(BHV)`. Only wrapper selection of BHV changes. Its kernel body, all
per-head expressions, BK128/BV64, num_warps=1, num_stages=3 and
multibuffer=False remain unchanged. This is not a new vectorized head layout.

`AscendKDAAttnBackend._forward_target_verify` constructs dense starts
`[0,4,...,bs*4]` and calls `_glm_recurrent(..., intermediate=intermediate_state)`
without a `prefill` argument. `_glm_recurrent` defaults `prefill=False` and
forwards it unchanged. Ordinary extend explicitly supplies `prefill=True`.
Draft/decode calls without scratch do not match this candidate. A CPU AST test
checks the actual verify call and its default, rather than assuming that a
multi-token call means prefill.

The NPU Mamba speculative pool already presents the persistent state as a
transposed logical [K,V] view. `_glm_recurrent` retains that view and transposes
the scratch's last two axes. The candidate accepts those native strides only:
key stride 1, and transposing the last two axes back must be contiguous.
`Glm5Next` explicitly creates A_log and dt_bias as FP32 parameters.

The new host-only guard requires:

- D `disaggregation_mode="decode"`, `prefill=False`, no prefix tracking.
- q/k/v/a `[1,bs*4,32,128]`, b `[1,bs*4,32]`, BF16 contiguous, bs12/13/14.
- Contiguous 1D int32/int64 indices of length bs and starts of length bs+1.
- Native transposed FP32 state `[pool,32,128,128]` and nonempty scratch
  `[capacity>=bs,4,32,128,128]`, with the native strides above.
- FP32 contiguous A_log/dt_bias with 32/4096 elements, all tensors on the same NPU.

Existing q/k/v/a/b contiguous preparation happens before this guard. No new
copy, tensor arithmetic or host read of device values is added. Guard misses
retain the prior wrapper/kernel behavior, including its existing validation.
As before, callers must provide valid slots and monotonic starts bounded by the
token/scratch capacities. The flag does not add runtime scans of these values;
the verified serving call constructs fixed four-token sequences. The isolated
unit additionally covers zero-length packed sequences without exceeding four
steps. It does not exercise malformed metadata that would overrun old scratch.

## Independence and remaining risk

Each program owns `(value tile, sequence, head group)`. For each head h,
`group=h//BHV` and `i_bhv=h%BHV` select exactly one owner. Every head retains
its independent BV64/BK128 tile, four token iterations and per-token snapshot.
There is no cross-head reduction, no head padding (32 is divisible by 2 and 4),
and no reordering within a head. Verify never commits persistent state because
SAVE_INTERMEDIATE=True. Multiple sequences may read the same state slot without
a write race; scratch addresses use the sequence index rather than that slot.

| Batch | BHV1 grid / programs | BHV2 grid / programs | BHV4 grid / programs |
| --- | --- | --- | --- |
| 12 | (2,12,32) / 768 | (2,12,16) / 384 | (2,12,8) / 192 |
| 13 | (2,13,32) / 832 | (2,13,16) / 416 | (2,13,8) / 208 |
| 14 | (2,14,32) / 896 | (2,14,16) / 448 | (2,14,8) / 224 |

These are legal grids, but fewer programs do not imply a speedup. Four steps
already do substantial state/snapshot work; serialization, occupancy, loop
lowering or register pressure can offset scheduling savings. The current KDA
profile attribution (~6.78 ms) is not a predicted wall-time saving. Changing
BHV specializes a constexpr loop; unchanged source expressions and CPU exact
checks do not prove Ascend floating-point lowering identical. A mismatch in
BF16 output or any FP32 snapshot rejects that variant. No benefit is claimed
before isolated exact and timing results.

## Unit coverage

`test/manual/ascend/test_glm53_kda_bhv_npu.py` loads this one source file and
removes only its runtime-context import, supplying D role locally. Kernel code
and production guard/dispatch remain intact. No model, service, RMS, Kpool,
sgl-kernel package or other harness is needed.

Default matrix: bs1/12/13/14, metadata int32/int64, eight successive fixtures
behind the same captured addresses (64 completed fixture records):

- Four tokens per sequence, changing positive slots and inputs.
- Mixed and all padding slots 0/-1, including repeated positive read slots.
- Mixed/all zero-length sequences with fixed allocated token capacity.
- Signed-zero q/k, wide finite inputs/gates, then another live dense replay.

Every fixture directly launches the original kernel with BHV1/2/4, compares
eager and graph results, and compares eager/graph production-wrapper output.
Wrapper tracing requires actual BHV2/4 at supported batches and actual BHV1 for
bs1. It also verifies BK/BV and grid, preventing a silently falling-back unit
from being reported as a candidate pass.

Comparison is raw-byte equality (not allclose) for the complete BF16 output,
entire persistent FP32 pool, and all FP32 snapshots for every step/head. All
input tensors and metadata must remain unchanged. Snapshot slots/steps not
written, excess scratch rows, direct-output canaries and unused packed output
are checked against their defined initial contents. Slot<=0 still has a
computed output for a four-token sequence; no padding outputs are ignored.
Dynamic slot and length updates reuse captured input/state/scratch/output
addresses. A mismatch exits nonzero and saves partial JSON with the active case.

CPU: five tests pass in about five seconds. Numerical interpretation is only
bs2/H4/K8/V8, with the same eight fixture recipes; large production grids use
metadata only. Tests cover role/flag/shape/dtype/stride fallback, complete small
snapshot/state/output equality, dynamic slots, original kernel AST identity and
the real target-verify callsite. `--help` also works without torch_npu installed.
CPU commands use `CUDA_VISIBLE_DEVICES='' TORCH_DEVICE_BACKEND_AUTOLOAD=0`.

## Bundle and next isolated window

Copy just these two files into `/kit` on the isolated test container:

1. `python/sglang/srt/hardware_backend/npu/attention/glm53/kda_recurrent_npu.py`
2. `test/manual/ascend/test_glm53_kda_bhv_npu.py`

```bash
env CUDA_VISIBLE_DEVICES='' TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  GLM53_KDA_SOURCE_DIR=/kit \
  python -B /kit/test_glm53_kda_bhv_npu.py \
  --device 0 --batches 1,12,13,14 --timing-repeats 30 \
  --cann-build 9.0.0 --output /unit/kda-bhv-unit-v1-result.json
```

The script imports torch_npu only in main and records Torch/torch_npu/Triton,
CANN label, device and source hash. It uses lower_bound=-5 and scale=128**-0.5;
the local GLM config/audit copies also specify gate_lower_bound=-5 (no running
model metadata was read). Timing starts only after all requested
exact cases pass. Separate NPU events measure 20 graph replays per sample with
rotating BHV order; raw samples and median microseconds are saved. This is
direct-kernel timing excluding wrapper zero allocation, resets and correctness
copies. Replay gaps can still contribute; do not add this timing to CPU overlap
or equate it with a model wall-time gain. No timing or precision result has been
obtained on NPU by this candidate's author.
