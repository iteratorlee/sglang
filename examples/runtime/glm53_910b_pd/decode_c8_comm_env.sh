#!/usr/bin/env bash
# Source before launching each node of DP8TP2/EP16 decode on two 8-card 910Bs.
# The existing HCCL pg_options keep the large buffer on MoE communicators only.
# This does not change W8A8/C8, MTP, graph sizes or prefix-cache semantics.
export HCCL_BUFFSIZE=128
export DEEPEP_HCCL_BUFFSIZE=1280
export HCCL_INTRA_PCIE_ENABLE=1
export HCCL_INTRA_ROCE_ENABLE=0
unset HCCL_OP_EXPANSION_MODE

# Keep --pre-warm-nccl enabled so actual communication allocations are charged
# before sizing the KV pool. Recompute independent long-context capacity from
# the live target/draft pools and Mamba slots; do not infer it from CLI limits.

# Measured 128K/1K configuration: 14 requests per attention DP domain, with
# request/Mamba slots and target/draft graphs covering that execution limit.
# The measured pool still decides the full independent context capacity.
# For a new process, the existing SGLANG_GLM53_PD_WARM_ALLOCATOR startup hook
# can automate allocator prewarming when its exact-backend guard is eligible.
# The paired experiment validates explicit private-scratch prewarming first.
