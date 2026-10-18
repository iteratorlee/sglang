# LI physical-capacity diagnostic, 2026-10-19

Test-only supplement from `8534011`; its production Python tree equals runtime
`357d985`. No production file, original harness, fixture or v2 comparator changes.
The failed full-GSM result remains **1262/1319 < 1267** (raw1259 plus three percent
extraction corrections); this diagnostic neither retries nor clears that gate.

## Confirmed coverage gap

The old v2/v3 `frames` passes `capacity=prefix+264`. At bs13/prefix131068,
`pool_pages=(131068+264)//256+2=515`: key is `[6696,64,1,128]` and LI table
`[13,515]`. At prefix252 it is `[53,64,1,128]` / `[13,4]`. The saved production
profile instead reports `[29049,64,1,128]` / `[13,519]`. This is a real missing
physical-shape condition, not proof that LI batching caused the GSM regression.

`glm53_index_physical_fixture.py` embeds the original logical fixture into a
configurable physical cache (default29049 blocks), with a fixed-width LI table
(default519 columns). The default full table has2076 columns; `--full-table-width
2074` also yields519 via `[:,::4]`, matching the saved full-attention profile.
The NPU harness asserts the **actual op arguments**: legacy `[bs,519]`, batched
`[bs*4,519]`, common key `[29049,64,1,128]`, TND independent len1 prefix sums,
per-query key lengths, dynamic physical page references, sparse mode3/count512.
It records actual tensor strides, dtype, storage offset and NPU format; no format
cast hides a mismatch. Disjoint request pages are shuffled across physical
capacity, the first live logical page is placed at physical block29048, and
unmapped cache blocks contain finite poison. Closing-pool poison is preserved.

## Bounded matrix and unchanged oracle

Exactly20 fixtures, no timing:

- Nine random: bs1 starts127/252/2044/2048/2052; bs2 `[127,2044]`;
  bs4 `[63,252,2048,2052]`; bs8 `[32,127,252,1023,2044,2048,2052,4095]`;
  bs13 `131068 + row%4`.
- Three ties: bs1/2052, bs4 mixed, bs13 long.
- Four near-ties: bs1/2048, bs2 mixed, bs8 mixed, bs13 long.
- One closing poison: bs13 long.
- Three mixed-padding: bs2/8/13, last request becomes slot0.

Each is legacy flags-off vs LI-only vs LI+expand, pairwise in eager and graph,
plus each graph vs its eager output and an independent legacy repeat. One graph
per mode/batch is reused across its fixtures while request slots, positions,
physical pages and all input buffers change in place. Actual input addresses
must remain fixed. The same production source supplies all three modes unless
an explicit `--baseline-source` is provided. Capture-context injection is the
same existing fixture convention; this is **not** a test of the new MIN_BS guard.

The existing `glm53_index_batch_contract_v2.py` is called unchanged: positive-key
raw and expanded indices exact in original order; zero-key candidate canonical
-1 and expansion oracle; defined short-prefix tails; all nine full pre/post/final
state snapshots and final cache byte-exact, **including every unused physical
block, dead row and sink**. No masked state regions, sorted-topk comparison,
allclose or hash-only equality. Undefined old zero-key differences remain counted.
The extra0 default matches production query geometry; optional `--extra` retains
original shape-padding checks. Shared-sink payload control is inherited unchanged
from the old fixture. Synthetic data cannot reproduce arbitrary real GSM tensors.

Full key storage is475938816 bytes. Full-state traces intentionally cost several
GiB per mode; allow64GiB-class isolated NPU memory and ample host RAM. No large
CPU Triton interpreter run was performed. The parent owns device scheduling.

## Local checks and isolated command

CPU conda `sgl-0511`, CUDA hidden, backend autoload disabled: **6 tests passed**.
They cover production geometry arithmetically (no large allocation), small
physical remapping with future poison, the real extracted three paths with
Triton CPU interpreter, pointer-stable dynamic slot reset, zero-key/padding,
and rejection of an unused-cache bit/value fault, live topk corruption and
wrong actual LI width. These do not certify CANN arithmetic, tie order or graph
replay. `--plan-only` performs no fixture allocation or device initialization.

Repository invocation, only by the parent in the isolated NPU window:

```bash
CUDA_VISIBLE_DEVICES='' TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -B \
  test/manual/ascend/test_glm53_index_physical_npu.py \
  --device 0 --key-blocks 29049 --pooled-table-width 519 \
  --full-table-width 2076 --cann-build 9.0.0 \
  --output /results/li-physical-capacity-unit-v1.json
```

Flat bundle invocation: set `GLM53_INDEX_BATCH_SOURCE=/kit/kpool_indexer.py` and
run `/kit/run_npu.py` with the same arguments. Bundle contains the new fixture,
entrypoint, unchanged shared fixture/oracle/capture harness, and six production
files (`kpool_indexer`, metadata, expand, graph_ops, tail_scatter, snapshot).
`SHA256SUMS` identifies every file; the result repeats the executable source
hashes and baseline hash. Existing receipts are never overwritten, and all
results retain `promotion_allowed=false`. `--batches` or `--case-ids` can select
an explicitly reported subset; the default is the full20-case matrix.
