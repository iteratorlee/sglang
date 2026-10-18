# D index batch zero-key canonicalization / defined contract v2

Review patch only, **default off**. No remote NPU, GPU, model, or serving process
was used. NPU v2 and real D100 content validation are **pending**; both must pass
before considering enablement. No throughput or latency improvement is claimed.

- Actual base HEAD: `d49c0dededd1e6e87c3f4e91b3a3c1ac67001022`.
- Branch: `glm53-index-zero-kv-1018`.
- Worktree: `/home/iterator/workspace/e2e_opt/910b_pd_opt/repos/sglang-index-zero-kv-1018`.
- Author: `liyan <yan.l@pku.edu.cn>`; commit only, no push.
- Only production edit: four lines after candidate batched LI in
  `python/sglang/srt/hardware_backend/npu/attention/glm53/kpool_indexer.py`.
  `pool_lens == 0` masks raw output to `-1` in place before the existing reshape
  and four bs-row expansions. Positive lengths and their ordering are untouched.
  Tail updates, compression, cache writes and snapshots retain their order.
  This adds a captured mask/fill operation; its NPU cost is not measured here.
- `SGLANG_GLM53_VERIFY_INDEX_BATCH` remains default off with the existing guards.
  No gated norm, P TopK method, expansion kernel, or main worktree file was edited.

## Evidence and correctness boundary

The preserved v1 receipt is
`910b_pd_opt/results/cold-ttft-itl-opt-1018/index-batch-unit-v1-receipt.json`,
SHA256 `68cfc3c15bbfb3d5ef0c7f7cd2f1fd828ea1ada980bf897a83f51bffd0e74ec1`.
It still records the strict failure. Nothing in this patch changes that result
or the v1 harness/comparator/default behavior. The supplied P28 diagnostic
reports live-row and all floating state/cache/snapshot byte equality through
bs14/128K, with old-vs-old instability in the all-zero-key domain. That diagnostic
was not rerun here. Those observations alone establish neither a real precision
regression nor a full correctness pass.

Local code in `AscendIndexerKPoolMixin._expand_with_tail` (lines 472 onward)
explicitly states the **first min(seq_len, index_topk + tail)** consumption rule.
Its tail placement is `min(seq - seq % 4, index_topk)`, and invalid slots become
`seq`. The equivalent Triton `_expand` in `kpool_expand_npu.py` uses the same
history/tail placement and sentinel. For active positions 0..2, closed-pool
length is zero but FA length is 1..3: the consumed prefix is exactly the causal
tail `0..seq-1`. All three tail slots are overwritten by expansion, so even the
unused tail sentinels are defined. Remaining raw-LI-derived old slots have no
selected closed pool and are outside this v2 comparison domain.

The public Huawei CANN source
[`sparse_flash_attention_kernel_mla.h`](https://raw.githubusercontent.com/hicann/ops-transformer/refs/heads/master/attention/sparse_flash_attention/op_kernel/arch22/sparse_flash_attention_kernel_mla.h)
provides corroborating implementation evidence: `GetSparseActualSeqLen` bounds
the sparse length by the causal threshold; `UpdateInnerLoopCond` returns early
on zero effective KV/query length; `DealActSeqLenIsZero` initializes zero output.
`CalcSinnerTopKBegin` bounds the valid index count and returns at that bound.
Thus dead requests with FA KV length zero have no consumed attention indices.
This public source is **not a hash match to the installed CANN 9.0 binary**.

Likewise, the public
[`lightning_indexer_kernel_arch22.h`](https://raw.githubusercontent.com/hicann/ops-transformer/refs/heads/master/attention/lightning_indexer/op_kernel/arch22/lightning_indexer_kernel_arch22.h)
has a zero-key early return in `GetS2BaseBlockNumOnMask`, but also explicit
clearing in `ProcessInvalid` and `DealActSeqLenIsZero`. It does **not** prove
that upstream documentation guarantees undefined zero-key output, nor that all
installed versions leave it unwritten. Here “undefined old padding” means
outside the locally defined selected-index/consumer comparison domain, backed
by the reported repeat instability, not an asserted upstream API guarantee.
Installed source/build verification and real D100 remain the deployment gate.

## Explicit v2 contract

The new entrypoint is `test/manual/ascend/test_glm53_index_batch_v2_npu.py`.
It shares fixture generation, capture, empty-forward and optional timing with
v1, and uses the separate `glm53_index_batch_contract_v2.py` comparator.

| Domain | Required checks |
| --- | --- |
| Every live positive-key row | Entire raw and expanded row bitwise exact, in original order; valid-count, uniqueness and causal range checks |
| Every candidate zero-key row, including dead rows | Raw output all `-1`; entire new expanded row matches an independent canonical expansion oracle |
| Active zero-key short prefix | Old/new consumed causal tail and all three explicitly written tail slots exact; old remaining padding is counted |
| Dead rows | Raw/expanded differing element **and byte** counts saved; candidate canonical checks still mandatory |
| All tail K/scores, MTP snapshots, cache | Every byte in all nine pre/post/final snapshots and final buffers exact, including dead/sink/unused regions; no row masks or tolerance |
| Extra shape padding | Exact integer zero |

The same checks apply to eager A/B, graph A/B, old eager repeat, old graph/eager,
and new graph/eager. Both sides of new graph/eager must be canonical; neither
side of old graph/eager is compared against arbitrary old padding. Snapshots
are materialized before reset/replay. Replay retains the input/state addresses
while requests, positions and page contents change. Four-vs-one LI call counts,
update/indexer/expand ordering and four bs-row expand shapes are asserted.

Each case records diagnostic counts **before** assertions, and is added to the
receipt before comparison, so failing cases keep their counts. Reports use
`defined_outputs_exact` rather than claiming full old/new output equality, keep
`promotion_allowed=false`, and refuse to overwrite any existing output file.
Source hashes include the candidate, kernels, both harnesses, fixture and v2
comparator. Defaults cover bs1/12/13/14, prefixes 0/1/2/3/60/252/2044/131068,
random/ties/near-ties/closing, mixed/all padding, all closing offsets, shuffled
pages and signed-zero sink. Full-cache traces require substantial host/NPU RAM.
The fixture explicitly injects capture context for eager diagnostics; CPU
eligibility tests separately check the actual production guard.

## Local validation

Linux CPU only, torch `2.11.0+cu130`, Triton `3.6.0`, Triton interpreter,
`CUDA_VISIBLE_DEVICES='' TORCH_DEVICE_BACKEND_AUTOLOAD=0`:

- New `test_glm53_index_zero_kv_cpu.py`: **5 tests passed**, 1.917s.
- Existing `test_glm53_index_batch_cpu.py`: **11 tests passed**, 10.697s.
- New harness `--help`, Python AST parsing and `git diff --check`: passed.

New fault injection returns different valid-looking garbage for each zero-key
LI invocation. Tests execute extracted production methods and real metadata,
scatter, snapshot and fused expansion kernels in the CPU interpreter. They
verify positive rows are untouched, old strict comparison still fails, zero-key
canonicalization handles active short prefixes/mixed/all dead rows, and repeated
old garbage is counted. Negative tests corrupt live raw/expanded indices,
canonical raw, consumed tail, dead new padding, shape padding, each state/cache
buffer, and a snapshot count; all are rejected. State faults flip a single bit.
The CPU comparison-role test exercises the graph/eager comparator, **not** NPU
capture/replay. No CPU LI reference can certify real CANN arithmetic or ties.

```bash
cd /home/iterator/workspace/e2e_opt/910b_pd_opt/repos/sglang-index-zero-kv-1018
env CUDA_VISIBLE_DEVICES='' TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  /home/iterator/anaconda3/envs/sgl-0511/bin/python -B \
  test/registered/unit/hardware_backend/npu/test_glm53_index_zero_kv_cpu.py -v
env CUDA_VISIBLE_DEVICES='' TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  /home/iterator/anaconda3/envs/sgl-0511/bin/python -B \
  test/registered/unit/hardware_backend/npu/test_glm53_index_batch_cpu.py -v
```

## Main-agent NPU handoff (not executed)

Run on an isolated target Ascend NPU in the deployment's CANN/torch_npu/Triton
NPU environment. No server startup is needed. The 2048 top-k and 32 heads below
match the saved v1 receipt; verify them against the intended D configuration.
Use fresh v2 paths; retain the original v1 receipt. Supply the recorded exact
installed build string, and optionally `--installed-contract /path/to/evidence`
to hash installed source/contract evidence (the hash alone is not validation).

```bash
# From this checkout, prepare the same pre-candidate baseline used by v1.
git show 83cfe9668fb98ca674c3a28816dd99016e38b8b6:python/sglang/srt/hardware_backend/npu/attention/glm53/kpool_indexer.py \
  > /tmp/glm53-index-baseline-zero-kv-v2.py

# On the isolated NPU, from the transferred checkout:
env CUDA_VISIBLE_DEVICES='' TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  python -B test/manual/ascend/test_glm53_index_batch_v2_npu.py \
  --device 0 --batches 1,12,13,14 --prefixes 0,1,2,3,60,252,2044,131068 \
  --topk 2048 --heads 32 --fused-state \
  --baseline-source /tmp/glm53-index-baseline-zero-kv-v2.py \
  --cann-build 'REPLACE_WITH_EXACT_INSTALLED_BUILD' \
  --output /tmp/glm53-index-batch-defined-v2-fused.json
```

Repeat without `--fused-state` and with a fresh output filename for unfused
state updates. For a flat bundle include both NPU harnesses, the fixture and
v2 contract helper, plus the indexer and its five sibling kernel sources; set
`GLM53_INDEX_BATCH_SOURCE=/path/to/kpool_indexer.py` and supply baseline source.
The legacy strict entrypoint remains `test_glm53_index_batch_npu.py` and retains
its original default behavior. Optional `--benchmark 50` is only useful after
correctness. Even a v2 NPU unit pass must be followed by **real D100 content
validation** before considering the candidate flag for serving.
