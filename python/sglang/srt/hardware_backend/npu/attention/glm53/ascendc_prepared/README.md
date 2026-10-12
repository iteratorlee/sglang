# GLM-5.3 prepared recurrent AscendC sidecar

This is an opt-in, prefill-only build of the pinned external operator. The
production dispatch requires `prepared_recurrent_v1`; the original binary only
supports H4 and cannot be used for the H8 path. The adapter keeps the original
recurrence, vector arithmetic, and double buffering, while adding runtime H4/H8
selection, native FP32 state layout handling, int64 metadata conversion, and
host-side ABI checks. It refuses a different external commit or C++ source SHA.

On the **P** host, in the same Python environment and CANN toolkit used by the
running SGLang and `torch_npu`:

```bash
export SG_ROOT=/workspace/sglang
export ASCENDC_WORK=/tmp/glm53-prepared-ascendc
mkdir -p "$ASCENDC_WORK"
git clone https://github.com/kayzee3327/01-glm53-ascend.git "$ASCENDC_WORK/original"
git -C "$ASCENDC_WORK/original" checkout 4c4cd9f04fab99aa8c75f2590944f71856c23675
python "$SG_ROOT/python/sglang/srt/hardware_backend/npu/attention/glm53/ascendc_prepared/build_compatible.py" \
  --original "$ASCENDC_WORK/original/sgl-glm53-ascend-ops" \
  --output "$ASCENDC_WORK/compatible"
cmake -S "$ASCENDC_WORK/compatible" -B "$ASCENDC_WORK/compatible/build" \
  -G Ninja -DCMAKE_BUILD_TYPE=Release -DSGL_OPS=prepared_recurrent
ninja -C "$ASCENDC_WORK/compatible/build" sgl_glm53_ascend_ops_lib
ln -sf "$ASCENDC_WORK/compatible/build/sgl_glm53_ascend_ops_lib.so" \
  "$ASCENDC_WORK/compatible/src/sgl_glm53_ascend_ops/sgl_glm53_ascend_ops_lib.so"
export PYTHONPATH="$ASCENDC_WORK/compatible/src:${PYTHONPATH:-}"
export SGLANG_GLM53_KDA_PREFILL_PREPARE=1
export SGLANG_GLM53_KDA_PREPARED_ASCENDC=1
```

Launch the P worker from this shell. Its first supported B1/H4 or H8/D128
call logs `GLM53 AscendC prepared_recurrent_v1 active`. Supported token lengths
are 128 through 16384. Q/K/V/A/B/output must be contiguous BF16; state must
be the native FP32 `[slot, head, key, value]` allocation with either contiguous
or last-two-dimensions-transposed strides. Indices and tracker metadata may be
int32 or int64; starts must be int32. Unsupported layouts keep the existing
Triton recurrence path. If the versioned op is missing on a supported shape,
the P worker raises an error rather than silently using the old binary.

On the **D** host, keep the prepared path disabled:

```bash
export SGLANG_GLM53_KDA_PREPARED_ASCENDC=0
```

The original package's pip entry point targets a placeholder SGLang hook. Do
not `pip install` the copied tree; `PYTHONPATH` and the linked shared library
make its import register the Torch op directly. Build with the same PyTorch,
`torch_npu`, CANN, and C++ ABI as the target worker. No FlagGems dependency is
introduced. CPU dispatch checks run with:

```bash
python -B "$SG_ROOT/test/manual/ascend/test_glm53_prepared_ascendc_cpu.py" -v
```

CPU checks do not validate AscendC numerical behavior; verify output, FP32
final state, tracker snapshots, and latency on the actual P hardware before
using the path for cold prefill experiments.
