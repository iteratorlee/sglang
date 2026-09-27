"""Commit only accepted speculative KPool tails, alongside native KDA rollback."""

import logging
import os
import time

import torch

logger = logging.getLogger(__name__)


def _kpool_indexers(model):
    layers = getattr(getattr(model, "model", None), "layers", ())
    indexers = [
        getattr(getattr(layer, "self_attn", None), "indexer", None) for layer in layers
    ]
    return [
        indexer
        for indexer in indexers
        if indexer is not None and hasattr(indexer, "_kpool_mtp_tail_k")
    ]


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
        self.fused = os.getenv("SGLANG_GLM53_MTP_FUSED_COMMIT", "0") == "1"
        if self.fused:
            from sglang.srt.hardware_backend.npu.attention.glm53.fused_tail_commit import (
                make_pointer_tables,
            )

            self.tables, self.row_elements, self.num_steps = make_pointer_tables(indexers)
        # Scratch is immutable after verification. Repeated copies are
        # idempotent; native convolution rollback stays outside this graph.
        self._commit(indexers)
        torch.npu.synchronize()
        self.graph = torch.npu.NPUGraph()
        with torch.npu.graph(self.graph):
            self._commit(indexers)

    def _commit(self, indexers):
        if self.fused:
            from sglang.srt.hardware_backend.npu.attention.glm53.fused_tail_commit import (
                commit_fused,
            )

            commit_fused(
                self.tables,
                self.row_elements,
                self.num_steps,
                self.requests,
                self.accepted,
            )
        else:
            _copy_tails(indexers, self.requests, self.accepted)

    def replay(self, requests, accepted):
        self.requests.copy_(requests)
        self.accepted.copy_(accepted)
        self.graph.replay()


def prewarm_kpool_tail_commit_graphs(runner):
    """Capture every admissible raw batch size before serving requests.

    The ordinary target/draft graphs do not include the post-verify commit.
    Capturing this auxiliary graph on a DP rank's first smaller batch stalls
    all EP peers. Negative destination and step indices mask every state
    load/store in speculative_state_scatter_npu; only private index tensors
    are changed while these graphs warm and capture.
    """
    if (
        os.getenv("SGLANG_GLM53_MTP_COMMIT_GRAPH", "1") != "1"
        or runner.is_draft_worker
        or runner.decode_cuda_graph_runner is None
    ):
        return None
    indexers = _kpool_indexers(runner.model)
    if not indexers:
        return None
    backend = runner.attn_backend
    if getattr(backend, "_glm53_tail_commit_prewarmed", False):
        return None
    capture_bs = runner.decode_cuda_graph_runner.capture_bs
    max_bs = min(int(runner.max_running_requests), max(capture_bs))
    if max_bs < 1:
        raise ValueError("KPool commit warmup requires a positive local capacity")
    if any(indexer._kpool_mtp_tail_k.shape[0] < max_bs for indexer in indexers):
        raise ValueError("KPool commit warmup exceeds speculative scratch capacity")
    device = indexers[0]._kpool_tail_k.device
    started = time.monotonic()
    graphs = {}
    for batch_size in range(1, max_bs + 1):
        requests = torch.full((batch_size,), -1, dtype=torch.int32, device=device)
        accepted = torch.full_like(requests, -1)
        graphs[batch_size] = _TailCommitGraph(indexers, requests, accepted)
    torch.npu.synchronize()
    backend._glm53_tail_commit_graphs = graphs
    backend._glm53_tail_commit_prewarmed = True
    summary = dict(batch_sizes=list(graphs), elapsed_s=time.monotonic() - started)
    logger.info("GLM53 MTP tail commit graph startup warmup: %s", summary)
    return summary


def commit_kpool_tails(backend, model, accepted, requests):
    indexers = _kpool_indexers(model)
    if not indexers or not accepted.numel():
        return
    if requests is None:
        requests = indexers[0]._ascend_verify_req_indices
    requests = requests[: accepted.numel()].to(torch.int32)
    accepted = accepted.to(torch.int32)
    if os.getenv("SGLANG_GLM53_MTP_COMMIT_GRAPH", "1") != "1":
        return _copy_tails(indexers, requests, accepted)
    graph = getattr(backend, "_glm53_tail_commit_graphs", {}).get(accepted.numel())
    if graph is None:
        # Graph-disabled and previously unseen runtime states remain correct
        # without a device-wide synchronize/capture in the request path.
        return _copy_tails(indexers, requests, accepted)
    graph.replay(requests, accepted)
