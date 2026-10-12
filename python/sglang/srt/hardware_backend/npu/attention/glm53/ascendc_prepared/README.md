# GLM-5.3 prepared recurrent AscendC sidecar

This opt-in prefill adapter is pinned to external commit
`6fedd85fdf61481437e64b1b635371d9c8201b14` and its
`csrc/kda/prepared_recurrent.cpp` SHA256
`da0c2e1513debf025bc122c47eefba3db547853b905249424c7fbbcecbacbe71`.
The builder refuses a different source and writes only to a new output tree.
The generated C++ SHA256 is
`16eb7df5547fd3461fc43846e5f35c74ccdedcf588dac068165a2bd68e344be7`.

The generated library registers two ops. `prepared_recurrent_v2` accepts the
complete H4 or H8 tensor ABI in one Python call. For H8, its C++ host function
submits the upstream `H_TOTAL=8, HEAD_OFFSET=0` and `HEAD_OFFSET=4` kernels
in that order on the current NPU stream. Each kernel uses the original H4
32-block grid (8 value tiles × 4 heads); there is no H8 64-block launch.
`prepared_recurrent(..., head_offset=0|4)` remains available for explicit
single-group calls; for H4 only offset 0 is valid. The host API change from
the previous sidecar is `prepared_recurrent_v1` → `prepared_recurrent_v2`;
the original op's `head_offset` argument is preserved.

The adapter retains upstream recurrence, vector arithmetic, double buffering,
and the Vector→MTE3 checkpoint synchronization. It additionally accepts the
native FP32 state layout `[slot, head, key, value]` with contiguous or
last-two-dimensions-transposed strides, converts int64 indices/tracker metadata
to int32, and checks device, dtype, shape, and strides before launching.

On the **P** host, use the same Python environment and CANN toolkit as the
running SGLang and `torch_npu`:

```bash
export SG_ROOT=/workspace/sglang
export ASCENDC_WORK=/tmp/glm53-prepared-ascendc
mkdir -p "$ASCENDC_WORK"
git clone https://github.com/kayzee3327/01-glm53-ascend.git "$ASCENDC_WORK/original"
git -C "$ASCENDC_WORK/original" checkout 6fedd85fdf61481437e64b1b635371d9c8201b14
python "$SG_ROOT/python/sglang/srt/hardware_backend/npu/attention/glm53/ascendc_prepared/build_compatible.py" \
  --original "$ASCENDC_WORK/original/sgl-glm53-ascend-ops" \
  --output "$ASCENDC_WORK/compatible-v2"
cmake -S "$ASCENDC_WORK/compatible-v2" -B "$ASCENDC_WORK/compatible-v2/build" \
  -G Ninja -DCMAKE_BUILD_TYPE=Release -DSGL_OPS=prepared_recurrent
ninja -C "$ASCENDC_WORK/compatible-v2/build" sgl_glm53_ascend_ops_lib
ln -sf "$ASCENDC_WORK/compatible-v2/build/sgl_glm53_ascend_ops_lib.so" \
  "$ASCENDC_WORK/compatible-v2/src/sgl_glm53_ascend_ops/sgl_glm53_ascend_ops_lib.so"
export PYTHONPATH="$ASCENDC_WORK/compatible-v2/src:${PYTHONPATH:-}"
export SGLANG_GLM53_KDA_PREFILL_PREPARE=1
export SGLANG_GLM53_KDA_PREPARED_ASCENDC=1
```

The P worker logs `GLM53 AscendC prepared_recurrent_v2 active` on its first
supported B1/H4 or H8/D128 call. Supported token lengths are 128–16384.
Prepared Q/K/decay/beta are FP32, while V/output are contiguous BF16. Indices
and tracker metadata may be int32 or int64; starts must be int32. Unsupported
layouts keep the Triton recurrence path. If the versioned op is missing on a
supported shape, the P worker raises an error.

On the **D** host, keep the path disabled:

```bash
export SGLANG_GLM53_KDA_PREPARED_ASCENDC=0
```

The original package's pip entry point targets a placeholder SGLang hook. Use
`PYTHONPATH` and the linked shared library to register the Torch op directly.
Build with the target worker's PyTorch, `torch_npu`, CANN, and C++ ABI. CPU
dispatch checks run with:

```bash
python -B "$SG_ROOT/test/manual/ascend/test_glm53_prepared_ascendc_cpu.py" -v
```

CPU checks cover dispatch only. Verify output, FP32 final state, tracker
snapshots, and latency on the P hardware before production use.
