# D verify index batch minimum size

Base: `85340119e0eefcd99a0c22b4931609b9958bc4fe`.
Branch/worktree: `glm53-d-index-min-bs-1019` / `repos/sglang-d-index-min-bs-1019`.
Author: `liyan <yan.l@pku.edu.cn>`. Review commit only; no deployment or push.

The user reports full GSM source357 combined score 1262 and old full score 1264,
both below 1267, with zero structural failures. These observations do not prove
that LI batching caused the failure. This patch only conservatively restricts
the candidate's scope; it makes no new accuracy or performance claim.

`SGLANG_GLM53_VERIFY_INDEX_MIN_BS` defaults to `1`. Initialization parses an
integer and rejects non-integers or values outside [1,16] with a named
`ValueError`, even when batch LI is disabled. The value is stored once before
buffer allocation. Missing configuration retains the prior eligible bs1..16.

The intended future candidate setting is `SGLANG_GLM53_VERIFY_INDEX_MIN_BS=12`:

- bs1..11 take the original four-LI path, including the stated global GSM C8
  sizes below 12.
- bs12..16 retain batch-LI eligibility in capture context, including the reported
  saturated D100 graph13. The existing NPU/kpool/steps/opt-in guards still apply.
- bs17 and non-capture execution stay ineligible.

The threshold uses the existing `bs` argument, derived by `_verify_topk` from
`_original_batch_size` or `forward_batch.batch_size`. No size bucketing changes.
Expand batching remains subordinate to batch LI. No arithmetic, tensor layout,
memory format, state update, zero-key sentinel, or other execution path changes.
The fixture, existing unit oracles, NPU harnesses and sample definitions are
unchanged. No full GSM retry was performed.

## CPU validation

Local sgl-0511 environment, torch CPU operations and Triton interpreter only.
No GPU, NPU, remote connection, service, or main-worktree file edit.

New `test_glm53_index_min_bs_cpu.py`: 5 tests passed, 9.894s. Covers all valid
minima, default/explicit 1, initialization-time caching, malformed/out-of-range
values, batch-LI disabled, bs0..17 and non-capture eligibility. Integration runs
the existing verify method for every bs1..17 plus non-capture bs13, substitutes
only the guard's device metadata, and checks LI/expand counts plus unchanged
raw/expanded/state/cache byte oracles. This does not execute an actual NPU graph.

Existing index-batch, zero-KV and expand-batch CPU suites: all 21 tests passed,
29.142s. Syntax and `git diff --check` passed. An AST comparison confirms only
initialization and `_verify_index_batch_enabled` changed in the production class.

```bash
cd /home/iterator/workspace/e2e_opt/910b_pd_opt/repos/sglang-d-index-min-bs-1019
env -u SGLANG_GLM53_VERIFY_INDEX_MIN_BS \
  CUDA_VISIBLE_DEVICES='' TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  /home/iterator/anaconda3/envs/sgl-0511/bin/python -B \
  test/registered/unit/hardware_backend/npu/test_glm53_index_min_bs_cpu.py -v
```
