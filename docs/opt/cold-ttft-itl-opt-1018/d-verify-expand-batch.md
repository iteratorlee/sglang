# D verify request-major expand batch candidate

Review-only, default off. Base: `bb1b8dade5ddbcffc76904c0188ab7007fd55d4d`.
Branch: `glm53-d-expand-batch-1018`. Independent worktree:
`/home/iterator/workspace/e2e_opt/910b_pd_opt/repos/sglang-d-expand-batch-1018`.
Author: `liyan <yan.l@pku.edu.cn>`. Commit only, no push.

## Production change

`kpool_indexer.py` adds `SGLANG_GLM53_VERIFY_EXPAND_BATCH`, default `0`.
Only exact `1`, decode mode, and an enabled `SGLANG_GLM53_VERIFY_INDEX_BATCH`
can initialize the option. Dispatch additionally requires the existing batch
LI eligibility: NPU, capture context, kpool=4, four verify steps, bs in [1,16].

Immediately after batched LI and the existing zero-key `masked_fill_(-1)`, call
the existing `kpool_expand_npu.expand_with_tail(topk.squeeze(1),
positions[:real_shape_tokens])`. LI already produces request-major
`[bs*4,1,512]` indices for index_topk=2048. This removes four step slices and
their contiguous copies, four bs-row expansions, and the expanded-output stack.
The final q shape-padding allocation/concatenation is unchanged.

The existing kernel assigns each program a row and a 512-column tile, reading
only that row's pool IDs and position and writing only its output row/tile.
For 2051 output columns there are five tiles per row: bs13 launches 52*5=260
programs, bs14 launches 56*5=280. Arithmetic and output are integers. No kernel
source changes are needed. Legality on the installed NPU/compiler still needs
the isolated test below; CPU interpretation does not establish NPU performance.

The general `_expand_with_tail` <=16-row guard is unchanged. So are tail
updates, cache writes, snapshots, raw zero-key canonicalization, and all other
attention paths. This does not edit gated norm, P TopK, topology or MTP settings.

## Explicit test contract

`Runner(..., batch_expand=False)` is explicit and ignores ambient expand flags.
The unchanged v1 entrypoint therefore keeps its strict default. The v2 entrypoint
adds `--batch-expand`; without it, the existing four bs-row expectations remain.
The comparator requires the expected shape for each side independently; it does
not accept either launch shape indiscriminately.

With the option, v2 checks legacy vs candidate eager and graph, legacy repeat,
and each side's graph vs eager using the same defined-output contract:

- Every live positive-key row: full raw and expanded indices byte-exact.
- New zero-key raw indices: all -1, including dead rows; full canonical expanded
  rows checked by the existing independent oracle.
- Active zero-key rows: exact causal tail and defined tail slots; old remaining
  padding differences counted. All dead-row differences retain element/byte counts.
- All nine state snapshots and final state/cache: every byte exact, including
  sink/dead regions. Extra q padding remains integer zero.
- Candidate eager and captured dispatch: one LI, one expansion of bs*4 rows,
  after all four state updates. Receipt fields `candidate_li_calls=1`,
  `expand_rows=[bs*4]`, `graph_expand_rows=[bs*4]`; A/B `li_calls=[4,1]`.

The consumer boundary remains the code's first
`min(seq_len, index_topk + tail)` rule in `_expand_with_tail`. Active positions
0..2 have a causal tail even though LI key length is zero. The prior
[zero-key note](d-index-zero-kv-v2.md) records the zero-KV FA early-return source
evidence and its installed-build limitations. This patch makes no new upstream
API guarantee about undefined padding. It leaves the failed v1 receipt intact.

`--benchmark 30 --batch-expand` times three trace-free graphs per batch at
prefix 131068: `legacy`, `batch_li`, `batch_li_expand`. Each graph's LI and expand
dispatch is asserted, and all 30 event timings plus medians are recorded.
Compare the latter two modes for incremental expansion cost; these are unit
graph times, not service ITL. Without `--batch-expand`, v2 uses its old timing.

## Local validation

CPU only, torch 2.11.0+cu130, Triton 3.6.0 interpreter; no GPU/NPU initialization,
remote access, service run, or main-worktree edit.

- New `test_glm53_expand_batch_cpu.py`: 5 tests passed (16.282s).
  Checks default-off/decode/LI gating, default fixture isolation, CPU and bs>16
  fallback, bs1/12/13/14 production paths with fused/unfused state updates,
  canonical short prefixes and q padding, and all state/cache bytes.
  An independent Python integer oracle checks strided 52/56-row, 512-pool expansion
  through position 131068 for int32/int64; the general >16-row fallback is also
  exercised. v2 fault injection rejects canonical, state-byte and launch-shape
  corruption while keeping old garbage counts and v1 strict failures.
- Existing index batch + zero-KV suites: 16 tests passed (12.420s).
- v2 `--help`, syntax checks, and `git diff --check`: passed.

```bash
cd /home/iterator/workspace/e2e_opt/910b_pd_opt/repos/sglang-d-expand-batch-1018
env CUDA_VISIBLE_DEVICES='' TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  /home/iterator/anaconda3/envs/sgl-0511/bin/python -B \
  test/registered/unit/hardware_backend/npu/test_glm53_expand_batch_cpu.py -v
env CUDA_VISIBLE_DEVICES='' TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  /home/iterator/anaconda3/envs/sgl-0511/bin/python -B -m unittest discover \
  -s test/registered/unit/hardware_backend/npu -p 'test_glm53_index*cpu.py' -v
```

## Isolated NPU handoff — not executed locally

Use the deployment's CANN/torch_npu/Triton environment in the next isolated
window. Preserve all old receipts and use a fresh output filename. The harness
directly opts in through its fixture; no service launch is involved.

```bash
# Prepare the same legacy source used by the existing LI v2 harness.
git show 83cfe9668fb98ca674c3a28816dd99016e38b8b6:python/sglang/srt/hardware_backend/npu/attention/glm53/kpool_indexer.py \
  > /tmp/glm53-expand-batch-legacy.py

# Run only on the allocated isolated NPU, from the transferred checkout.
env CUDA_VISIBLE_DEVICES='' TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  python -B test/manual/ascend/test_glm53_index_batch_v2_npu.py \
  --device 0 --batches 1,12,13,14 --prefixes 0,252,131068 \
  --topk 2048 --heads 32 --fused-state --batch-expand --benchmark 30 \
  --baseline-source /tmp/glm53-expand-batch-legacy.py \
  --cann-build 'REPLACE_WITH_EXACT_INSTALLED_BUILD' \
  --output /tmp/glm53-verify-expand-batch-v2-fused.json
```

For a flat bundle, include the current v2/v1 harnesses, fixture, v2 comparator,
indexer and its five sibling kernels listed in the v2 source-hash table. Set
`GLM53_INDEX_BATCH_SOURCE=/path/to/kpool_indexer.py` and supply the legacy source.
Repeat without `--fused-state` using a fresh receipt if the deployment needs
unfused state updates. Actual NPUGraph replay and CANN LI are untested locally.

Promotion remains false. Only after NPU unit and real D100 content validation
both pass should serving enablement be considered, with both
`SGLANG_GLM53_VERIFY_INDEX_BATCH=1` and `SGLANG_GLM53_VERIFY_EXPAND_BATCH=1`.
No latency improvement is claimed from CPU results.
