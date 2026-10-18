# D one-dimensional gated RMS loop minimum rows

Review-only alternative; not merged or deployed. Base HEAD:
`85e2800d0f1b2e55c541f1eb48e1f351a3a1f339`, whose Python tree matches `25a7d72`.
Branch/worktree: `glm53-d-rms-loop-min-rows-1019` /
`repos/sglang-d-rms-loop-min-rows-1019`.
Author: `liyan <yan.l@pku.edu.cn>`; no push or main-worktree edits.

`SGLANG_GLM53_D_GATED_RMS_LOOP_MIN_ROWS` defaults to `1024` and accepts integers
in [1024,2048]. A non-integer or out-of-range value raises a named `ValueError`
before kernel launch when the existing one-dimensional D loop branch is eligible.
Parsing is local to that branch: P, disabled loops, and other original fallback
paths do not consume this D-only setting. This preserves P even if the unused
D setting is malformed. Existing role/device/dtype/layout/row guards still apply.

The proposed future setting is `SGLANG_GLM53_D_GATED_RMS_LOOP_MIN_ROWS=1536`,
with the existing D loop size 8 and the unselected 2D option remaining off:

| Flattened rows | Dispatch at minimum 1536 |
| --- | --- |
| 1024 | Original row kernel, grid 1024 |
| 1535 | Original row kernel, grid 1535 |
| 1536 | Existing large-row kernel, loop8, grid 192 |
| 1664 | Existing large-row kernel, loop8, grid 208 |
| 1792 | Existing large-row kernel, loop8, grid 224 |
| 2048 | Existing large-row kernel, loop8, grid 256 |
| Above 2048 | Original fallback dispatch |

Thus the stated GSM globalC8/graph8 maximum of 1024 rows falls back, while the
stated D100 graph13/1664 rows remains eligible for loop8. This is conservative
scope restriction, not evidence that RMS caused a GSM failure or that the new
scope passes GSM. Adoption remains the parent agent's decision after native-old
event availability is resolved. No GSM retry or accelerator validation occurred.

Both original kernel function ASTs are identical to the base, as are the P
selection and 2D selection branches. P131072 retains its original baseline/R4
grid32768 and R8 grid16384. The 2D flag/default, kernel math, tensor layouts and
launch options are unchanged. Existing numerical oracles and fixtures are intact.

## CPU validation

Local sgl-0511 CPU environment only; no GPU/NPU allocation, remote access or
service operation. New dispatch tests use tensor metadata and launch recorders.

- New minimum-row suite: **6 tests passed**, 0.040s. Checks default/explicit1024,
  min1536 boundaries with R4/R8, interior/upper valid minima, invalid values,
  P independence, disabled loops and unchanged 2D opt-in behavior.
- Existing **13 tests passed**, 3.314s: all 7 loop tests (including strict CPU
  numerical comparisons and kernel AST checks), all 4 BF16 contract tests,
  plus the 2D dispatch and original-kernel AST tests. No oracle relaxation.
- Direct AST comparison against base, syntax and `git diff --check` passed.

```bash
cd /home/iterator/workspace/e2e_opt/910b_pd_opt/repos/sglang-d-rms-loop-min-rows-1019
env -u SGLANG_GLM53_D_GATED_RMS_LOOP_MIN_ROWS \
  CUDA_VISIBLE_DEVICES='' TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  /home/iterator/anaconda3/envs/sgl-0511/bin/python -B \
  test/registered/unit/hardware_backend/npu/test_glm53_gated_rms_loop_min_rows_cpu.py -v
```
