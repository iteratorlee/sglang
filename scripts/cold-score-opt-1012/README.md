# Cold pooled-score experiment, 2026-10-12

Base `d20cbd26be0abee69f8a0ca1cad184c31dfb7cc4`. This directory contains isolated
operator harnesses, not model launch scripts. The surrounding task's report is
`910b_pd_opt/docs/opt/910b-cold-score-opt-1012.md`; raw receipts are in
`910b_pd_opt/results/cold-score-opt-1012/`.

The original full fused scorer is rejected: a native-head-sum difference changes
non-tied top-k sets on `rank_flip`, and the compiled kernel needs up to 1 GiB
scratch for a 32-row 512K tile. Keep `SGLANG_GLM53_PREFILL_POOLED_SCORE=0`.

The candidate changes only the in-place ReLU/FP32-gate product and retains native
CANN einsum, native head sum, query tiling, causal mask and native top-k. Its
production flag `SGLANG_GLM53_PREFILL_POOLED_POSTOP` defaults to `0`. Opt-in `auto`
(alias `1`) requires P NPU mode and uses BLOCK8192 for legacy tile32 with at least
32768 pools, or tile8 with at least98304 pools. These thresholds select measured
performance cases; they do not approximate scores. Other tiles and smaller
lengths use the original pointwise operations. Decode behavior is unchanged.

The helper requires BF16 contiguous ND format2 scores with at most32 rows and
32 heads, matching FP32 contiguous ND format2 gates and device. It also guards
total launch grid≤65535. Unsupported inputs run native ReLU/multiply exactly once
inside the helper and return False for activation logging only. Never repeat the
legacy product on a False return. Input -0 becomes +0 before multiplication to
match measured CANN semantics. CPU fallback follows its own native semantics.

## Tests

```bash
python -B -m unittest discover -s scripts/cold-score-opt-1012 -p 'test*cpu.py' -v
python -B test/manual/ascend/test_glm53_pooled_score_prefill_cpu.py -v
python -B test/manual/ascend/test_glm53_prefill_score_validate_cpu.py -v
```

CPU checks cover numerical stage sensitivity, exact ordered ties, causality,
partial and TP fragments, flag policies, launch limits and fallback ownership.
They do not prove Ascend numerical equivalence.

All NPU entry points require `ASCEND_RT_VISIBLE_DEVICES=7` and logical device0.
Run only inside the authorized private D27 utility. Do not use a model worker.
`provision.sh` documents that utility's mounts/runtime; it intentionally fails
on another host or an existing container name. No installation is required.

```bash
# Strict forced post-op substitution; replaces precisely two original statements.
python3 -B scripts/cold-score-opt-1012/bench_postop_npu.py \
  --case 262144:128 --case 524288:128 --tile 32 --block 8192 \
  --seed 1012 --seed 1013 --seed 1014 --repeat 21 --output /eval/results/forced.jsonl

# Actual production auto branch, with a separately staged immutable baseline.
python3 -B scripts/cold-score-opt-1012/bench_postop_npu.py \
  --integrated --baseline-repo /eval/source \
  --case 262144:128 --case 393216:128 --case 524288:128 --tile 8 --block 8192 \
  --repeat 21 --output /eval/results/integrated.jsonl

python3 -B scripts/cold-score-opt-1012/validate_postop_npu.py \
  --output /eval/results/edges.json
python3 -B scripts/cold-score-opt-1012/measure_postop_npu.py \
  --output /eval/results/stages.json
```

The benchmark compares every weighted BF16 stored bit, finite final-score bit,
ordered ID, set and self-repeat. The faster harness counts int16-view mismatches
on NPU and transfers values only when diagnosing differences. The exhaustive
edge script independently compares host int16 copies for all65536 BF16 patterns
against32 FP32 gate patterns. IEEE payload/zero differences are never waived.
In integrated mode an ineligible shape must have zero kernel calls, and its final
scores/IDs must still match. Metadata records hashes for both source trees.

Timings alternate A/B order with explicit NPU synchronization. The caller scope
includes mask/top-k/padding/concatenation over the same128 queries and legacy
tiles. `measure_postop_npu.py` separately reports raw key scoring and
reset-plus-postop. Every memory result includes allocator and driver baselines;
driver snapshots are not a transient external-CANN peak measurement. JIT warmup
is excluded. Shared-host contention and private-card memory headroom differ from
the model service. Do not report these timings as cold TTFT improvements.

## Full-fusion diagnostics

- `bench.py`: original full scorer versus native, CPU or NPU. Numerical failure
  exits2. `--time-rejected` records diagnostic timing without changing acceptance.
- `diagnose_npu.py`: AST-generated stage taps for dot/weighted/final differences;
  also checks tapped final output against the unmodified full kernel.
- `measure_workspace_npu.py`: full-fusion raw scorer and standalone top-k scopes.
- `inspect_workspace.py`: compiler workspace metadata and installed runtime
  allocation source, without initializing NPU.
- `check_tp_npu.py`: serial eight-owner reconstruction on one NPU, not HCCL.

No source activation, service restart, model quality claim or full-P acceptance is
performed by these scripts. Main owns the coordinated model A/B and D27 restore.
