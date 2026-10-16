#!/usr/bin/env bash
# Performance deltas for the eager, two-node GLM W8A8/C8 P profile.
# Source in each P container after its network/device configuration.
# Required CLI: TP16 / attention DP2TP8 / EP16, deepep normal, no graphs or
# overlap, int8 primary KV, hierarchical prefix cache, context 132608.
# The measured 16K-per-DP chunk uses:
#   --chunked-prefill-size 32768 --max-prefill-tokens 16384
# Keep the existing max-mamba-cache-size 66 (33 per attention DP domain).
export HCCL_BUFFSIZE=1024
unset DEEPEP_HCCL_BUFFSIZE
unset HCCL_OP_EXPANSION_MODE
export SGLANG_GLM53_2NODE_PD=1
export SGLANG_GLM53_NORMAL_HCCL=1
export SGLANG_GLM53_NORMAL_SKIP_IDLE_DP=1
export SGLANG_GLM53_NORMAL_SKIP_IDLE_ATTN=1
export SGLANG_GLM53_NORMAL_SHAPE_CACHE=1
export SGLANG_GLM53_NORMAL_COMBINE_DETERMINISTIC=1
export SGLANG_GLM53_NORMAL_COMBINE_ORDERED=2node
export SGLANG_GLM53_NORMAL_REDUCE_SCATTER=2node
export SGLANG_GLM53_NORMAL_REDUCE_SCATTER_CHECK=0
# All math-path additions are opt-in and limited to large P chunks. The raw
# per-DP token guard is shared by all EP ranks; do not use padded token counts.
export SGLANG_GLM53_NORMAL_COMBINE_SKIP_EMPTY=1
export SGLANG_GLM53_NORMAL_COMBINE_SKIP_EMPTY_MIN_TOKENS=4096
export SGLANG_GLM53_NORMAL_PREFILL_DRAFT_SCOPE=1
export SGLANG_GLM53_NORMAL_PREFILL_DRAFT_MIN_TOKENS=4096
export SGLANG_GLM53_PREFILL_MHC_PRE=1
export SGLANG_GLM53_NORMAL_DEVICE_ROUTING=ordered
export SGLANG_GLM53_NORMAL_DEVICE_ROUTING_MIN_TOKENS=512
# Fixed target-forward geometry: retain the first global shape audit, then
# check the per-layer CPU contract. NextN keeps its independent audit.
# The 128K/1K cold case and actual D100 content checks passed. Full GSM8K
# scored 1264/1319 versus the fixed 1267 control; this opt-in recipe is not
# a general accuracy sign-off. All failed trials are retained in the report.
export SGLANG_GLM53_NORMAL_BATCH_SHAPE_AUDIT=1
# Experiments below are not part of the recommended configuration.
export SGLANG_GLM53_NORMAL_ALL_TO_ALL=0
export SGLANG_GLM53_NORMAL_SUM_LEN=0
