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
export SGLANG_GLM53_2NODE_PD=1
export SGLANG_GLM53_NORMAL_HCCL=1
export SGLANG_GLM53_NORMAL_SKIP_IDLE_DP=1
export SGLANG_GLM53_NORMAL_SKIP_IDLE_ATTN=1
export SGLANG_GLM53_NORMAL_SHAPE_CACHE=1
export SGLANG_GLM53_NORMAL_COMBINE_DETERMINISTIC=1
export SGLANG_GLM53_NORMAL_COMBINE_ORDERED=2node
export SGLANG_GLM53_NORMAL_REDUCE_SCATTER=2node
export SGLANG_GLM53_NORMAL_REDUCE_SCATTER_CHECK=0
export SGLANG_GLM53_NORMAL_DEVICE_ROUTING=0
# Experiments below are not part of the recommended configuration.
export SGLANG_GLM53_NORMAL_ALL_TO_ALL=0
export SGLANG_GLM53_NORMAL_SUM_LEN=0
