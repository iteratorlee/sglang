# P RMS loop base-format guard fix

Parent `f630cc8`; branch `glm53-p-rms-format-fix-1018`. Author
`liyan <yan.l@pku.edu.cn>`. No device/remote operation or NPU retry.

## Preserved first failure

`results/cold-ttft-itl-opt-1018/rms-loop-prefill-unit-v1-result.json` records
zero completed numerical cases: first 131072-row BF16-weight/BF16-output
wrapper capture, R8 expected grid 16384 but all four launches used 32768.
It is a dispatch failure, not an NPU precision result. The result, collection
log, bundle specification and unit plan remain unchanged.

The bundle hashes match the delivered source. Its `capture_wrapper` already
reshapes x to `[1,16384,8,128]`, gate to `[16384,8,128]`, and installs prefill
runtime role. The original strict harness's earlier 32-head view is therefore
not the wrapper's final shape. The failing receipt includes an empty_like
base-format warning but does not record individual tensor formats; identifying
the exact rejected tensor from this receipt alone would be speculation.

The ND2-only guard is independently demonstrably too restrictive for a native
contiguous 4D output. The correction is limited to that supported base layout,
and new diagnostics will establish what the installed runtime actually returns.

## Layout evidence and narrow correction

Read-only upstream Ascend/pytorch v2.10.0 source inspection:

- [InferFormat.cpp](https://raw.githubusercontent.com/Ascend/pytorch/v2.10.0/torch_npu/csrc/framework/InferFormat.cpp):
  `GuessBaseFormat` selects NCHW for rank four; `GuessFormatUnit` also maps
  rank-four ND to NCHW.
- [FormatHelper.cpp](https://raw.githubusercontent.com/Ascend/pytorch/v2.10.0/torch_npu/csrc/framework/FormatHelper.cpp):
  ND and NCHW are base, non-padded formats. For rank four, `InferShapeofNCHW`
  preserves all dimensions in order, with no axis permutation or channel
  padding; ND preserves dimension order as well.
- [FormatCastHelper.cpp](https://raw.githubusercontent.com/Ascend/pytorch/v2.10.0/torch_npu/csrc/aten/common/FormatCastHelper.cpp):
  base-to-base conversion keeps sizes/strides and copies memory, without
  rearranging elements.

Thus for an already contiguous `[1,16384,8,128]` tensor, physical offset is
`((token * 8 + head) * 128 + column)` in both ND and NCHW. The existing kernel
uses exactly that flattened row address. Singleton-axis strides do not alter
this formula. This reasoning does not apply to blocked NC1HWC0/NZ or arbitrary
noncontiguous layouts. The inspected upstream branch is not a retrieved copy
of the installed binary's source revision; runtime metadata still needs recording.

`supported_prefill_loop` now accepts format 2, or format 0 **only for rank four**.
Existing P-only shape, dtype, same-device and contiguity checks remain. Three-
dimensional gate and one-dimensional weight still require ND2. Unknown formats,
format-query errors, blocked formats and noncontiguous inputs still fall back.
There is no formatcast, data copy, allocation change, kernel change or expansion
of the P token/head geometry. D dispatch and both original 1D kernels are unchanged.

## Diagnostic and verification changes

The harness adds `wrapper_diagnostics` to result JSON. Before its grid assertion
it records caller shapes, final x/gate/weight/actual-output shapes, strides,
storage offsets, dtype, device, contiguity, each NPU format (or query error),
runtime role, requested flags, supported_prefill_loop result, and observed /
expected grids. These records survive a mismatch even with `cases=[]`.
No extra probe output is allocated; metadata is read from the actual wrapper
output outside capture. A mismatch still fails; no numerical comparisons are
removed or weakened. Format-query failure remains visible rather than being
treated as acceptance.

CPU regression simulates ND2 input plus NCHW0 empty_like output through the
full P capture adapter: R8 reaches grid 16384. A blocked-format output still
uses 32768, raises, and retains its false supported result and metadata.
It also checks rank-limited format acceptance, invalid formats, missing format
API and noncontiguity using meta tensors only. All seven loop CPU tests pass;
no 131072-row interpreter run is used. This is guard/harness validation, not
NPU numerical validation. D's previously reported 192 exact cases are retained
as prior evidence; this patch makes no new target-device claim.

## Next tag

Use a fresh result/tag such as `rms-loop-prefill-unit-v2`. Repackage the same
four files: loop driver (often named `/kit/run_npu.py`), original strict harness,
`rms_norm_gated_npu.py`, and updated `rms_norm_gated_rows_npu.py`. Only the driver
and grouped/helper source change. Keep `GLM53_GATED_RMS_SOURCE_DIR=/kit` and the
existing `--role prefill --rows 131072 --timing-repeats 30` CLI. Do not reuse or
overwrite the v1 result. The next isolated run is owned by the main agent.
