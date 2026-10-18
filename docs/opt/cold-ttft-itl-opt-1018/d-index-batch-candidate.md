# D verify indexer batching candidate — 2026-10-18

## Status and scope

Reviewable, **default off**, not approved for enabling in the serving process.
The independent length-1 sequence causal construction is valid under the
documented LI contract below. **Target CANN 9.0 installed-contract verification,
NPU exact acceptance (including ties), and performance remain outstanding.**
No NPU, local GPU, remote machine, serving process, or remote service was used
for this work. Do not interrupt the main agent's cold baseline / D100 profile.

- Worktree: `/home/iterator/workspace/e2e_opt/910b_pd_opt/repos/sglang-d-index-batch-1018`
- Branch: `glm53-d-index-batch-1018`
- Base: `83cfe9668fb98ca674c3a28816dd99016e38b8b6`
- Requested author: `liyan <yan.l@pku.edu.cn>`
- Runtime change is confined to `kpool_indexer.py` init/decode/verify.
  No prefill method or expansion kernel is changed. Main worktree untouched;
  no push. The adjacent P metadata-reuse commit `35e8d5d` was not incorporated.
- Flag: `SGLANG_GLM53_VERIFY_INDEX_BATCH=1`, read at indexer initialization.
  Unset, `0`, or any value other than literal `1` leaves the candidate disabled.
- Further guards: NPU device, disaggregation mode `decode`, `index_kpool=4`,
  `draft_token_num=4`, local request batch 1–16, and model graph capture mode
  (including its warmup). Replay executes that captured branch. Ordinary eager,
  P, non-four-step, non-four-pool, empty, and larger-batch paths stay original.

## Source comparison with the running D

`git diff 32ffc6a3cfd6753e2fc95c2d0e768466352717a4
83cfe9668fb98ca674c3a28816dd99016e38b8b6 --
python/sglang/srt/hardware_backend/npu/attention/glm53` is empty. In particular
the running D's indexer, metadata, scatter, snapshot, and expand sources match
this candidate's parent. This is **not** a whole-program D equivalence claim:
the full range changes 24 files, including shared model/MoE/mHC code and the
decode startup script (`DEEPEP_HCCL_BUFFSIZE` 1536 → 1280). Several new shared
optimizations are P- or opt-in-gated, but deployment flags were not inspected.
The A/B script therefore records both source hashes and the installed runtime.

## Algorithm and causal argument

The original four `_decode_topk` calls still compute metadata, scatter tail K
and score, compress the current tail, and scatter the closing pool in order.
An internal keyword returns `(pool_lens, pooled_table)` just before LI, only
when the new verify branch asks for deferral. Each MTP snapshot still follows
its corresponding update, before the next step's update.

After all four updates/snapshots, issue one LI call using request-major rows:
`query[r*4+s]`, `weights[r*4+s]`, `actual_query=arange(1, bs*4+1)`,
`actual_key[r*4+s]=pool_lens[r,s]`, and each request's pooled table repeated
four times. The index cache is shared, not copied. Expand the returned indices
in four **bs-row** calls with the original per-step positions, then restore the
existing request-major output and zero shape-padding rows.

The official [TorchNPU LI API](https://www.hiascend.com/document/detail/en/Pytorch/2610/apiref/customapi/docs/en/custom_APIs/torch_npu/torch_npu-npu_lightning_indexer.md)
defines TND query lengths as B-element cumulative endpoints, PA_BSND key lengths
as individual valid lengths, and mode 3 as right-down causal. It allows zero
valid lengths and requires one block-table row per sequence. This is the public
26.1.0 document, **not evidence of the actually installed CANN 9.0 build**.

Under that contract, query `i` of a sequence with lengths `Q,K` sees key `j`
iff `0 <= j < K` and `j <= i + K - Q`. Here `Q=1`, `i=0`, so this reduces
exactly to `0 <= j < K`, independent of its batch index or cumulative endpoint.

For four consecutive positions `a+s`, `s=0..3`, let `g=floor(a/4)` and
`c=3-(a mod 4)`. There is exactly one closing step `c`. Its write changes pool
`g`; before `c`, `K_s=floor((a+s+1)/4)=g`, so that pool is invisible. From `c`
on, `K_s=g+1` and the final cache contains the same completed pool as the
sequential call. All visible historical pools are unchanged. Tail data and
snapshots have already been copied before subsequent tail updates.

This relies on the existing verify/allocator invariants: consecutive positions
per request, distinct active request slots, no aliasing of mutable closing
cache locations across requests, and reserved padding page 0 excluded from
valid live history. Shared immutable prefix pages are compatible. Padding
request 0 gets key length 0 for every step. Unused table columns retain the
existing clamp to safe page 0. "No future read" here means no future pool
participates in selection; LI's existing physical prefetch of safe masked pages
is not prohibited or changed by the patch.

Using endpoints `[4,8,...]` with one length-4 sequence per request is wrong:
causal offsets then advance in *pool* units, while closed pools advance only
once per four tokens. Broadcasting the final key length to earlier queries is
also wrong. Both mistakes have explicit negative CPU tests.

## Exactness and performance limits

The public API does not establish batch-shape-independent tie ordering. B grows
from `bs` to `4*bs`, which can change tiling, floating reduction, and tie choice.
Equal top-k values or equal index sets are insufficient: different tied members
can select different KV tokens; even reordered identical sets fail this task's
integer-order contract. CPU uses an explicitly stable reference tie policy,
not a model of CANN's arithmetic. A separate adversarial, valid top-k stub
demonstrates different indices with exactly identical scores and cache state.

Reject enabling this candidate if the installed contract differs, any raw or
expanded index differs, any step state differs, or graph replay is stale.
Do not relax checks to allclose, sorting, overlap, or output-token agreement.
A finite NPU pass still does not prove arbitrary-input tie stability; keep the
flag off pending target-build review and the main agent's acceptance decision.
No new tie-breaking rule is imposed on the existing path.

This removes three LI operator invocations, **not three logical history scans**:
there remain four query/history scoring problems. Physical history reuse and
speedup depend on CANN scheduling/cache behavior. Repeated tables, length
stacking, and existing per-step input packing add work. Profile the full verify
region, not only LI launch count.

The `<=16` fused-expand guard is unchanged. Local bs=14 produces 56 LI rows but
still four 14-row expand calls. No 56-row torch expansion or expanded guard is
introduced. CPU checks fused vs torch expansion byte-for-byte for bs 1/2/14/16;
the optional NPU benchmark measures complete verify graphs without trace copies.

## Local validation

CPU environment: `sgl-0511`, torch `2.11.0+cu130`, Triton `3.6.0`, always with
`CUDA_VISIBLE_DEVICES='' TORCH_DEVICE_BACKEND_AUTOLOAD=0`. No CUDA use.

**22 tests passed:** new index-batch suite 11, existing snapshot suite 4,
existing tail-scatter suite 7. The new suite uses source-extracted production
methods and actual Triton kernels under the CPU interpreter. It covers all
closing offsets; positions around 64-token and 256-token pooled-page boundaries;
random/tied/near-tied scores; a poisoned future closing pool; mixed/all padding;
zero closed history; empty forward; extra padded token rows; fixed-storage
request/position/page-table changes; all per-step tails, scores, snapshots and
entire pool cache; raw and expanded integer top-k; guards and negative examples.
An independent sequential state/expansion oracle supplements the A/B comparison.

All bytes are compared, including signed zero. Existing non-closing writes all
target cache row 0 and can race between +0/-0 when compressed keys have mixed
signs. Multi-request fixtures use positive tail keys so every sink write is +0;
duplicate padding tails also have identical payloads. A single-request negative
key case explicitly checks -0. This avoids attributing pre-existing sink WAW
nondeterminism to batching; it does not change or mask production sink values.

```bash
cd /home/iterator/workspace/e2e_opt/910b_pd_opt/repos/sglang-d-index-batch-1018
env CUDA_VISIBLE_DEVICES='' TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  /home/iterator/anaconda3/envs/sgl-0511/bin/python -B \
  test/registered/unit/hardware_backend/npu/test_glm53_index_batch_cpu.py -v
env CUDA_VISIBLE_DEVICES='' TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  /home/iterator/anaconda3/envs/sgl-0511/bin/python -B \
  test/registered/unit/hardware_backend/npu/test_glm53_kpool_snapshot_cpu.py -v
env CUDA_VISIBLE_DEVICES='' TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  /home/iterator/anaconda3/envs/sgl-0511/bin/python -B \
  test/registered/unit/hardware_backend/npu/test_glm53_kpool_tail_scatter_cpu.py -v
```

`pytest` is not installed in this environment; these are unittest entrypoints.
NPU script AST/`--help` checked on CPU only. No NPU result or latency is claimed.

## Main-agent isolated NPU handoff

Use the target CANN 9.0 environment after profiling, with the model's actual
`index_topk` and index query head count. `2048/64` below are explicit script
defaults, not inferred deployment facts. Save the installed LI contract/source
and build identification; the newer public web page is insufficient to certify
the target installation. With no git checkout, supply `--baseline-source` and
retain the helper beside the script plus the repo-relative kernel sources.

```bash
# In an isolated target NPU environment only. No server startup is involved.
env CUDA_VISIBLE_DEVICES='' TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  python -B test/manual/ascend/test_glm53_index_batch_npu.py \
  --device 0 --batches 1,2,14 --prefixes 0,60,252,2044,131068 \
  --topk 2048 --heads 64 \
  --cann-build 'REPLACE_WITH_EXACT_INSTALLED_BUILD' \
  --installed-contract /path/to/installed/li-contract-or-source \
  --output /tmp/glm53-index-batch-exact.json
# Repeat with --fused-state when the deployed tail/snapshot flags are enabled.
# Add --benchmark 50 for trace-free graph timing only after exact checks pass.
```

The script compares frozen parent source versus working-tree source; invokes
real LI and real metadata/scatter/snapshot/expand kernels; captures NPUGraph;
then changes query/key/weights/requests/positions/page tables in place. It checks
every step's full state, raw top-k and expanded top-k, candidate vs baseline and
graph vs eager. Cases include exact ties beyond the top-k cutoff, near-ties,
closing before/after page boundaries, mixed/all padding and empty forward.
Failures exit nonzero and record an error; successful evidence includes runtime
versions, device name, source hashes, call counts and expand shapes. Trace mode
clones the full cache and is intentionally memory-heavy at bs=14/128K; its
latency must not be reported as performance. Optional timing captures fresh
graphs with those copies disabled and excludes fixture reset from the event
interval. No benchmark result authorizes enabling the serving flag by itself.
