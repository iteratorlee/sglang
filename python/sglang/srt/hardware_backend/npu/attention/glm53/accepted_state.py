"""Commit only accepted speculative KPool tails, alongside native KDA rollback."""

import os

import torch


def _copy_tails(indexers, requests, accepted):
    from sgl_kernel_npu.mamba.speculative_state_scatter import (
        speculative_state_scatter_npu,
    )

    source_rows = torch.arange(
        accepted.numel(), device=accepted.device, dtype=torch.int32
    )
    for indexer in indexers:
        for suffix in ("k", "score"):
            destination = getattr(indexer, "_kpool_tail_" + suffix).unsqueeze(0)
            source = getattr(indexer, "_kpool_mtp_tail_" + suffix).unsqueeze(0)
            speculative_state_scatter_npu(
                destination, source, requests, source_rows, accepted
            )


class _TailCommitGraph:
    def __init__(self, indexers, requests, accepted):
        self.requests = requests.clone()
        self.accepted = accepted.clone()
        # Scratch is immutable after verification. Repeated copies are
        # idempotent; native convolution rollback stays outside this graph.
        _copy_tails(indexers, self.requests, self.accepted)
        torch.npu.synchronize()
        self.graph = torch.npu.NPUGraph()
        with torch.npu.graph(self.graph):
            _copy_tails(indexers, self.requests, self.accepted)

    def replay(self, requests, accepted):
        self.requests.copy_(requests)
        self.accepted.copy_(accepted)
        self.graph.replay()


def commit_kpool_tails(backend, model, accepted, requests):
    indexers = [
        getattr(layer.self_attn, "indexer", None) for layer in model.model.layers
    ]
    indexers = [
        indexer
        for indexer in indexers
        if indexer is not None and hasattr(indexer, "_kpool_mtp_tail_k")
    ]
    if not indexers or not accepted.numel():
        return
    if requests is None:
        requests = indexers[0]._ascend_verify_req_indices
    requests = requests[: accepted.numel()].to(torch.int32)
    accepted = accepted.to(torch.int32)
    if os.getenv("SGLANG_GLM53_MTP_COMMIT_GRAPH", "1") != "1":
        return _copy_tails(indexers, requests, accepted)
    graphs = getattr(backend, "_glm53_tail_commit_graphs", None)
    if graphs is None:
        graphs = backend._glm53_tail_commit_graphs = {}
    if accepted.numel() not in graphs:
        graphs[accepted.numel()] = _TailCommitGraph(indexers, requests, accepted)
    graphs[accepted.numel()].replay(requests, accepted)
