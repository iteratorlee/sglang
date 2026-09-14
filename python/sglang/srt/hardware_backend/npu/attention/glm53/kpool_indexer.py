from __future__ import annotations

from typing import Optional
from copy import copy
import os

import torch
import torch.nn.functional as F

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_server_args, get_device, get_spec
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_token_to_kv_pool,
)
from sglang.srt.utils import is_npu

from .graph_ops import scatter_rows_

if is_npu():
    import torch_npu


def _get_full_attn_metadata(forward_batch: ForwardBatch):
    """Resolve MLA metadata through the active hybrid attention backend."""

    override = getattr(forward_batch, "_ascend_kpool_metadata", None)
    if override is not None:
        return override
    backend = get_attn_backend()
    backend = getattr(backend, "full_attn_backend", backend)
    metadata = getattr(backend, "forward_metadata", None)
    if metadata is None:
        raise RuntimeError("GLM-5.3 KPool requires initialized MLA metadata")
    return metadata


def _get_index_k_buffer(forward_batch: ForwardBatch, layer_id: int):
    """Resolve the NPU MLA index cache through hybrid KV pools.

    ``HybridLinearKVPool`` deliberately exposes only the common KV-cache API.
    Its MLA-only index cache lives on ``full_kv_pool`` and uses a compact
    full-attention layer id.  Preserve the hybrid pool's transfer wait before
    mapping the model layer id, exactly as its regular buffer accessors do.
    """

    pool = get_token_to_kv_pool()
    get_buffer = getattr(pool, "get_index_k_buffer", None)
    if get_buffer is not None:
        return get_buffer(layer_id)

    full_pool = getattr(pool, "full_kv_pool", None)
    transfer_id = getattr(pool, "_transfer_full_attention_id", None)
    if full_pool is None or transfer_id is None:
        known_attrs = [
            name
            for name in (
                "full_kv_pool",
                "kv_pool",
                "token_to_kv_pool",
                "get_kv_buffer",
                "get_index_k_buffer",
                "_transfer_full_attention_id",
            )
            if hasattr(pool, name)
        ]
        raise RuntimeError(
            "GLM-5.3 KPool requires an MLA pool with an index-key buffer; "
            f"got type={type(pool).__module__}.{type(pool).__name__}, "
            f"known_attrs={known_attrs}"
        )
    wait_for_layer = getattr(pool, "_wait_for_layer", None)
    if wait_for_layer is not None:
        wait_for_layer(layer_id)
    return full_pool.get_index_k_buffer(transfer_id(layer_id))


def _ascend_lightning_weights(
    weights: torch.Tensor, query: torch.Tensor
) -> torch.Tensor:
    """Adapt FP32 GLM head gates to the Ascend ACL contract.

    Keep the checkpoint projection and its scale in FP32.  The Ascend
    ``aclnnLightningIndexer`` nevertheless requires ``weights`` to have the
    same BF16/FP16 dtype as query and key.  Convert only at this operator
    boundary; this is captured as a regular dynamic cast in NPUGraph and does
    not change the persistent parameter or pre-cast projection contract.
    """

    if weights.dtype != torch.float32:
        raise ValueError(
            "GLM-5.3 index-head projection must produce FP32 weights before "
            "the Ascend LightningIndexer compatibility cast"
        )
    if query.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("Ascend LightningIndexer requires a BF16 or FP16 GLM query")
    return weights.to(query.dtype)


def _as_ascend_sparse_indices(indices: torch.Tensor) -> torch.Tensor:
    """Add CANN's singleton next-token dimension to logical KPool ids.

    KPool expansion is most naturally expressed as ``[tokens, sparse_count]``.
    The Ascend sparse-attention ABI, however, consumes the same
    three-dimensional layout returned by ``npu_lightning_indexer``:
    ``[tokens, next_n, sparse_count]``.  Ordinary decode has ``next_n == 1``.
    Keep the shape conversion explicit so prefill, decode, idle batches and
    graph padding cannot accidentally diverge.
    """

    if indices.ndim != 2:
        raise ValueError(
            "GLM-5.3 logical KPool indices must have shape [tokens, sparse_count]"
        )
    if indices.dtype != torch.int32:
        raise ValueError("Ascend sparse attention requires int32 KPool indices")
    return indices.unsqueeze(1)


class AscendIndexerKPoolMixin:
    """NPU execution for the native KPool model parameters."""

    def _init_ascend_kpool(self):
        max_reqs = getattr(get_server_args(), "max_running_requests", None) or 2048
        # ReqToTokenPool reserves row 0 for graph padding and allocates real
        # requests in [1, max_reqs].  Mirror its +1 layout here.
        num_req_slots = max_reqs + 1
        self._kpool_padding_req = 0
        device = get_device().device
        self.register_buffer(
            "_kpool_tail_k",
            torch.zeros(
                num_req_slots,
                self.index_kpool,
                self.head_dim,
                dtype=torch.bfloat16,
                device=device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_kpool_tail_score",
            torch.zeros(
                num_req_slots,
                self.index_kpool,
                self.head_dim,
                dtype=torch.float32,
                device=device,
            ),
            persistent=False,
        )

        steps = get_spec().speculative_num_draft_tokens
        if steps is not None:
            from sglang.srt.runtime_context import get_exec

            graph_bs = get_exec().graph.cuda_graph_config.decode.bs or [1]
            scratch_requests = max(num_req_slots, max(graph_bs) + 1)
            self.register_buffer(
                "_kpool_mtp_tail_k",
                torch.zeros(
                    scratch_requests,
                    steps,
                    self.index_kpool,
                    self.head_dim,
                    dtype=torch.bfloat16,
                    device=device,
                ),
                persistent=False,
            )
            self.register_buffer(
                "_kpool_mtp_tail_score",
                torch.zeros(
                    scratch_requests,
                    steps,
                    self.index_kpool,
                    self.head_dim,
                    dtype=torch.float32,
                    device=device,
                ),
                persistent=False,
            )

    def _project_q_key_weights(self, x: torch.Tensor, q_lora: torch.Tensor):
        q, _ = self.wq_b(q_lora)
        q = q.view(-1, self.n_heads, self.head_dim)
        key, _ = self.wk(x.view(-1, self.hidden_size))
        key = self.k_norm(key).to(torch.bfloat16)
        weights = self._project_head_weights(x)
        return q, key, weights

    def _project_head_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Return FP32 index-head gates, including their model scale."""

        weights, _ = self.weights_proj(x.view(-1, self.hidden_size).float())
        weights = weights * (self.softmax_scale * self.n_heads**-0.5)
        return weights

    def _compress(self, key: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
        prob = torch.softmax(
            score.float() + self.index_kpool_compress_ape.unsqueeze(0), dim=1
        )
        return (key.float() * prob).sum(dim=1).to(torch.bfloat16)

    def _pooled_write_locs(
        self, block_table: torch.Tensor, pool_ids: torch.Tensor
    ) -> torch.Tensor:
        # A pooled cache page holds 64 groups.  Group page j reuses token page
        # j * kpool, leaving the other token pages untouched for MLA KV.
        page_columns = torch.div(pool_ids, 64, rounding_mode="floor")
        page_columns = page_columns * self.index_kpool
        pages = block_table.gather(0, page_columns.to(torch.long))
        return pages.to(torch.long) * 64 + torch.remainder(pool_ids, 64)

    def _pooled_page_table(self, block_tables: torch.Tensor) -> torch.Tensor:
        # Eager/graph metadata can retain -1 in unused page-table columns.
        # LightningIndexer may prefetch those columns before applying the
        # sequence-length mask, so map them to the safe padding page 0.
        return block_tables[:, :: self.index_kpool].contiguous().clamp_min(0)

    def _expand_with_tail(
        self, pool_indices: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        # The fused small-row kernel is slower for long prefill batches.
        if (
            self.index_kpool == 4
            and pool_indices.shape[0] <= 16
            and os.getenv("SGLANG_GLM53_KPOOL_EXPAND", "1") == "1"
        ):
            from .kpool_expand_npu import expand_with_tail

            return expand_with_tail(pool_indices, positions)
        valid = pool_indices >= 0
        offsets = torch.arange(
            self.index_kpool, device=pool_indices.device, dtype=pool_indices.dtype
        )
        expanded = pool_indices.unsqueeze(-1) * self.index_kpool + offsets
        expanded = torch.where(valid.unsqueeze(-1), expanded, -1).flatten(1)

        # A query at position p can see p + 1 keys. Only the incomplete pool
        # after the last fully compressed group is appended. The tail must be
        # placed immediately after the valid expanded history, not after the
        # fixed index_topk-wide buffer: Ascend sparse attention consumes the
        # first min(seq_len, index_topk + tail) entries and does not accept a
        # -1 hole inside that valid prefix.
        tail_width = self.index_kpool - 1
        seq_lens = positions + 1
        tail_count = torch.remainder(seq_lens, self.index_kpool)
        tail_start = seq_lens - tail_count
        tail_offsets = torch.arange(
            tail_width, device=positions.device, dtype=positions.dtype
        )
        tail_valid = tail_offsets.view(1, -1) < tail_count.view(-1, 1)
        tail = torch.where(
            tail_valid,
            tail_start.view(-1, 1) + tail_offsets.view(1, -1),
            -1,
        ).to(expanded.dtype)

        output = F.pad(expanded, (0, tail_width), value=-1)
        history_len = torch.minimum(
            tail_start,
            torch.full_like(tail_start, expanded.shape[1]),
        )
        # Avoid an out-of-place NPU scatter in the captured decode graph: on
        # this CANN version scatter indices can be frozen at capture.  Three
        # broadcasted selects are equivalent because tail_width is kpool - 1.
        columns = torch.arange(
            output.shape[1], device=output.device, dtype=history_len.dtype
        ).view(1, -1)
        for tail_idx in range(tail_width):
            target = history_len.view(-1, 1) + tail_idx
            output = torch.where(
                columns == target,
                tail[:, tail_idx].view(-1, 1),
                output,
            )

        # CANN SparseFlashAttention eagerly gathers every sparse slot and does
        # not accept the CUDA convention of -1 sentinels. Use the first future
        # logical position instead. sparse_mode=3 masks these entries before
        # softmax, preserving the -1 semantics while keeping every gather
        # index non-negative. AscendBackend supplies one extra safe page-table
        # column for the exact page-boundary case.
        future = seq_lens.to(output.dtype).view(-1, 1)
        return torch.where(output >= 0, output, future)

    def _store_prefill_pools(
        self,
        key: torch.Tensor,
        gate_score: torch.Tensor,
        forward_batch: ForwardBatch,
        block_tables: torch.Tensor,
        layer_id: int,
    ) -> list[torch.Tensor]:
        compressed_by_request: list[torch.Tensor] = []
        offset = 0
        for i in range(
            (getattr(forward_batch, "_original_batch_size", None) or forward_batch.batch_size)
        ):
            q_len = int(forward_batch.extend_seq_lens_cpu[i])
            seq_len = int(forward_batch.seq_lens_cpu[i])
            first_pos = seq_len - q_len
            if first_pos < 0:
                raise ValueError("GLM-5.3 KPool received a negative prefix length")
            key_chunk = key[offset : offset + q_len]
            score_chunk = gate_score[offset : offset + q_len]
            req = forward_batch.req_pool_indices[i].to(torch.long)
            prior_tail = first_pos % self.index_kpool
            if prior_tail:
                key_chunk = torch.cat(
                    (self._kpool_tail_k[req, :prior_tail], key_chunk), dim=0
                )
                score_chunk = torch.cat(
                    (self._kpool_tail_score[req, :prior_tail], score_chunk), dim=0
                )

            n_pools = key_chunk.shape[0] // self.index_kpool
            if n_pools:
                slot_k = key_chunk[: n_pools * self.index_kpool].view(
                    n_pools, self.index_kpool, self.head_dim
                )
                slot_score = score_chunk[: n_pools * self.index_kpool].view(
                    n_pools, self.index_kpool, self.head_dim
                )
                compressed = self._compress(slot_k, slot_score)
                first_pool = first_pos // self.index_kpool
                pool_ids = first_pool + torch.arange(
                    n_pools, device=key.device, dtype=torch.long
                )
                write_locs = self._pooled_write_locs(block_tables[i], pool_ids)
                scatter_rows_(
                    _get_index_k_buffer(forward_batch, layer_id),
                    write_locs,
                    compressed,
                )
            n_drain = n_pools * self.index_kpool
            n_tail = key_chunk.shape[0] - n_drain
            if n_tail:
                self._kpool_tail_k[req, :n_tail] = key_chunk[n_drain:]
                self._kpool_tail_score[req, :n_tail] = score_chunk[n_drain:]

            # Prefill top-k must search all closed historical pools, not only
            # those produced by this scheduler chunk.  The index cache is
            # paged by logical request position, so reconstruct the compact
            # logical pool sequence from the current block table after writes.
            total_pools = seq_len // self.index_kpool
            if total_pools:
                all_pool_ids = torch.arange(
                    total_pools, device=key.device, dtype=torch.long
                )
                read_locs = self._pooled_write_locs(block_tables[i], all_pool_ids)
                cache = _get_index_k_buffer(forward_batch, layer_id)
                compressed_history = cache.reshape(-1, self.head_dim).index_select(
                    0, read_locs
                )
            else:
                compressed_history = key.new_empty((0, self.head_dim))
            compressed_by_request.append(compressed_history)
            offset += q_len
        return compressed_by_request

    def _prefill_pooled_topk(
        self,
        q: torch.Tensor,
        weights: torch.Tensor,
        compressed_by_request: list[torch.Tensor],
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        pooled_topk = self.index_topk // self.index_kpool
        result = []
        offset = 0
        for i in range(
            (getattr(forward_batch, "_original_batch_size", None) or forward_batch.batch_size)
        ):
            q_len = int(forward_batch.extend_seq_lens_cpu[i])
            seq_len = int(forward_batch.seq_lens_cpu[i])
            first_pos = seq_len - q_len
            q_req = q[offset : offset + q_len]
            w_req = weights[offset : offset + q_len]
            k_req = compressed_by_request[i]
            rows = []
            start = 0
            while start < q_len:
                # Keep matmul shapes independent of scheduler chunking.  If
                # every row scores all pools written by the *whole* extend,
                # then a one-shot request and its chunked equivalent use
                # different K dimensions before the causal mask.  On CANN
                # that can perturb near-boundary logits/top-k for early rows.
                # Split at global 128-token boundaries and score only the
                # causally visible pool prefix for this block.
                global_start = first_pos + start
                global_end = min(
                    first_pos + q_len,
                    ((global_start // 128) + 1) * 128,
                )
                end = global_end - first_pos
                candidate_count = min(global_end // self.index_kpool, k_req.shape[0])
                if candidate_count == 0:
                    ids = torch.full(
                        (end - start, pooled_topk),
                        -1,
                        dtype=torch.int32,
                        device=q.device,
                    )
                else:
                    logits = torch.einsum(
                        "qhd,kd->qhk",
                        q_req[start:end],
                        k_req[:candidate_count],
                    )
                    logits = (F.relu(logits) * w_req[start:end].unsqueeze(-1)).sum(1)
                    logical_pos = first_pos + torch.arange(
                        start, end, device=q.device, dtype=torch.long
                    )
                    visible_pools = torch.div(
                        logical_pos + 1, self.index_kpool, rounding_mode="floor"
                    )
                    pool_ids = torch.arange(candidate_count, device=q.device)
                    logits.masked_fill_(
                        pool_ids.view(1, -1) >= visible_pools.view(-1, 1),
                        float("-inf"),
                    )
                    take = min(pooled_topk, candidate_count)
                    ids = logits.topk(take, dim=-1).indices.to(torch.int32)
                    ids = torch.where(ids < visible_pools.view(-1, 1), ids, -1)
                    ids = F.pad(ids, (0, pooled_topk - take), value=-1)
                rows.append(ids)
                start = end
            result.append(torch.cat(rows, dim=0))
            offset += q_len
        return torch.cat(result, dim=0)

    def _prefill_topk(
        self,
        q: torch.Tensor,
        weights: torch.Tensor,
        compressed_by_request: list[torch.Tensor],
        forward_batch: ForwardBatch,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        from .prefill_index_parallel import enabled, parallel_pooled_topk, verify_once

        if enabled(q, forward_batch):
            from sglang.srt.distributed.parallel_state import get_attn_tp_group

            pool_indices = parallel_pooled_topk(
                self._prefill_pooled_topk,
                q,
                weights,
                compressed_by_request,
                forward_batch,
                self.index_topk // self.index_kpool,
                get_attn_tp_group().device_group,
            )
            verify_once(
                self,
                pool_indices,
                q,
                weights,
                compressed_by_request,
                forward_batch,
                get_attn_tp_group().device_group,
            )
        else:
            pool_indices = self._prefill_pooled_topk(
                q, weights, compressed_by_request, forward_batch
            )
        valid_rows = pool_indices.shape[0]
        topk = self._expand_with_tail(pool_indices, positions[:valid_rows])
        # mHC/communication kernels may pad hidden states (e.g. 27 -> 32
        # tokens). Match the regular DSA indexer's graph/eager contract: keep
        # valid rows first. Padded query rows are outside actual_seq_lengths_q,
        # but still use legal zero indices because CANN may prefetch them.
        if valid_rows < q.shape[0]:
            padding = torch.zeros(
                (q.shape[0] - valid_rows, topk.shape[1]),
                dtype=topk.dtype,
                device=topk.device,
            )
            topk = torch.cat([topk, padding], dim=0)
        return topk

    def _decode_topk(
        self,
        q: torch.Tensor,
        key: torch.Tensor,
        weights: torch.Tensor,
        gate_score: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        block_tables: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        bs = q.shape[0]
        metadata = _get_full_attn_metadata(forward_batch)
        if self.index_kpool == 4:
            from .kpool_metadata_npu import decode_metadata

            req, tail_rows, write_locs, closing, pool_lens, actual_q, pooled_table = (
                decode_metadata(
                    forward_batch.req_pool_indices[:bs],
                    metadata.seq_lens[:bs],
                    positions[:bs],
                    block_tables[:bs],
                )
            )
        else:
            raw_req = forward_batch.req_pool_indices[:bs].to(torch.long)
            token_rows = torch.arange(bs, device=q.device, dtype=torch.long)
            metadata = _get_full_attn_metadata(forward_batch)
            seq_lens = metadata.seq_lens[:bs].to(torch.int32)
            # num_token_non_padded is local to the MoE token shard under EP.
            # Every attention TP rank still owns the complete request batch.
            valid_req = (raw_req > 0) & (seq_lens > 0)
            req = torch.where(
                valid_req,
                raw_req,
                torch.full_like(raw_req, self._kpool_padding_req),
            )
            slot = torch.remainder(positions[:bs], self.index_kpool).to(torch.long)
            tail_rows = req * self.index_kpool + slot
        # IndexPutV2 and npu_scatter_nd_update_ can capture the first decode
        # step's indices on this CANN version.  Read request/slot from device
        # memory in Triton so every graph replay updates the current tail.
        scatter_rows_(self._kpool_tail_k, tail_rows, key)
        scatter_rows_(self._kpool_tail_score, tail_rows, gate_score)

        compressed = self._compress(
            self._kpool_tail_k[req], self._kpool_tail_score[req]
        )
        if self.index_kpool != 4:
            pool_ids = torch.div(
                positions[:bs], self.index_kpool, rounding_mode="floor"
            ).to(torch.long)
            row = token_rows
            page_columns = torch.div(pool_ids, 64, rounding_mode="floor")
            page_columns = (page_columns * self.index_kpool).clamp_min(0)
            # Inactive rows can likewise retain -1 in the page table.  Page 0 is
            # reserved as a safe sink and is never exposed by a real sequence.
            pages = block_tables[row, page_columns].clamp_min(0)
            write_locs = pages.to(torch.long) * 64 + torch.remainder(pool_ids, 64)
            closing = (slot == self.index_kpool - 1) & valid_req
            write_locs = torch.where(closing, write_locs, 0)
        compressed = compressed * closing.to(compressed.dtype).unsqueeze(-1)
        scatter_rows_(
            _get_index_k_buffer(forward_batch, layer_id),
            write_locs,
            compressed,
        )

        if self.index_kpool != 4:
            pool_lens = torch.div(seq_lens, self.index_kpool, rounding_mode="floor").to(
                torch.int32
            )
            actual_q = torch.arange(1, bs + 1, dtype=torch.int32, device=q.device)
            pooled_table = self._pooled_page_table(block_tables)
        topk, _ = torch_npu.npu_lightning_indexer(
            query=q,
            key=_get_index_k_buffer(forward_batch, layer_id),
            weights=_ascend_lightning_weights(weights, q),
            actual_seq_lengths_query=actual_q,
            actual_seq_lengths_key=pool_lens,
            block_table=pooled_table,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=self.index_topk // self.index_kpool,
            sparse_mode=3,
        )
        return self._expand_with_tail(topk.squeeze(1), positions[:bs])

    def _verify_topk(
        self,
        q,
        key,
        weights,
        gate_score,
        positions,
        forward_batch,
        block_tables,
        layer_id,
    ):
        self._ascend_verify_req_indices = forward_batch.req_pool_indices
        steps = forward_batch.spec_info.draft_token_num
        bs = (getattr(forward_batch, "_original_batch_size", None) or forward_batch.batch_size)
        real_shape_tokens = bs * steps
        metadata = _get_full_attn_metadata(forward_batch)
        step_batch = copy(forward_batch)
        step_batch.batch_size = bs
        step_batch.req_pool_indices = forward_batch.req_pool_indices[:bs]
        block_tables = block_tables[:bs]
        outputs = []
        req = forward_batch.req_pool_indices[:bs].long()
        for step in range(steps):
            rows = slice(step, real_shape_tokens, steps)
            step_positions = positions[rows].contiguous()
            step_metadata = copy(metadata)
            step_metadata.seq_lens = torch.where(req > 0, step_positions + 1, 0).int()
            step_batch._ascend_kpool_metadata = step_metadata
            result = self._decode_topk(
                q[rows].contiguous(),
                key[rows].contiguous(),
                weights[rows].contiguous(),
                gate_score[rows].contiguous(),
                step_positions,
                step_batch,
                block_tables,
                layer_id,
            )
            outputs.append(result)
            self._kpool_mtp_tail_k[:bs, step].copy_(
                self._kpool_tail_k.index_select(0, req)
            )
            self._kpool_mtp_tail_score[:bs, step].copy_(
                self._kpool_tail_score.index_select(0, req)
            )
        result = torch.stack(outputs, 1).flatten(0, 1)
        if q.shape[0] > real_shape_tokens:
            result = torch.cat(
                (
                    result,
                    result.new_zeros(q.shape[0] - real_shape_tokens, result.shape[1]),
                ),
                0,
            )
        return result

    def _draft_extend_topk(
        self,
        q,
        key,
        weights,
        gate_score,
        positions,
        forward_batch,
        block_tables,
        layer_id,
    ):
        # Accepted tokens are packed by request; rejected draft slots must not
        # write persistent tails. Device lengths change on each graph replay.
        bs = forward_batch.batch_size
        steps = self._glm53_draft_graph_steps
        lens = forward_batch.extend_seq_lens[:bs]
        starts = lens.cumsum(0) - lens
        raw_req = forward_batch.req_pool_indices[:bs]
        metadata = _get_full_attn_metadata(forward_batch)
        metadata.actual_seq_lengths_q = lens.cumsum(0).int()
        output = torch.zeros(
            (q.shape[0] + 1, self.index_topk + self.index_kpool - 1),
            dtype=torch.int32,
            device=q.device,
        )
        for step in range(steps):
            valid = (raw_req > 0) & (step < lens)
            rows = (starts + step).clamp(max=q.shape[0] - 1).long()
            step_batch = copy(forward_batch)
            step_batch.req_pool_indices = torch.where(valid, raw_req, 0)
            step_positions = positions.index_select(0, rows)
            step_meta = copy(metadata)
            step_meta.seq_lens = torch.where(valid, step_positions + 1, 0).int()
            step_batch._ascend_kpool_metadata = step_meta
            result = self._decode_topk(
                q.index_select(0, rows),
                key.index_select(0, rows),
                weights.index_select(0, rows),
                gate_score.index_select(0, rows),
                step_positions,
                step_batch,
                block_tables,
                layer_id,
            )
            dest = torch.where(valid, rows, q.shape[0])
            scatter_rows_(output, dest, result)
        return output[: q.shape[0]]

    def forward_npu(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        layer_scatter_modes=None,
        dynamic_scale: torch.Tensor = None,
        return_indices: bool = True,
    ) -> Optional[torch.Tensor]:
        if not return_indices:
            return None
        if forward_batch.forward_mode.is_idle() or x.shape[0] == 0:
            return _as_ascend_sparse_indices(
                torch.zeros(
                    (x.shape[0], self.index_topk + self.index_kpool - 1),
                    dtype=torch.int32,
                    device=x.device,
                )
            )

        if isinstance(q_lora, tuple):
            q_lora = q_lora[-1] if len(q_lora) == 3 else q_lora[0]
        q, key, weights = self._project_q_key_weights(x, q_lora)
        gate_score = F.linear(
            x.view(-1, self.hidden_size), self.index_kpool_compress_gate
        ).float()
        block_tables = _get_full_attn_metadata(forward_batch).block_tables
        from .compact_index import index_block_table

        block_tables = index_block_table(forward_batch, block_tables)

        if forward_batch.forward_mode.is_target_verify():
            indices = self._verify_topk(
                q,
                key,
                weights,
                gate_score,
                positions,
                forward_batch,
                block_tables,
                layer_id,
            )
        elif (
            forward_batch.forward_mode.is_draft_extend_v2()
            and getattr(get_attn_backend(), "graph_mode", False)
            and hasattr(self, "_glm53_draft_graph_steps")
        ):
            indices = self._draft_extend_topk(
                q,
                key,
                weights,
                gate_score,
                positions,
                forward_batch,
                block_tables,
                layer_id,
            )
        elif forward_batch.forward_mode.is_extend():
            compressed = self._store_prefill_pools(
                key, gate_score, forward_batch, block_tables, layer_id
            )
            indices = self._prefill_topk(
                q, weights, compressed, forward_batch, positions
            )
        else:
            if getattr(self, "_glm53_draft_decode", False):
                # Each draft step advances positions. Match the compressed
                # causal prefix to those positions even when backend metadata
                # retains the target's pre-draft sequence length.
                metadata = copy(_get_full_attn_metadata(forward_batch))
                bs = forward_batch.batch_size
                valid = (forward_batch.req_pool_indices[:bs] > 0) & (
                    metadata.seq_lens[:bs] > 0
                )
                metadata.seq_lens = torch.where(valid, positions[:bs] + 1, 0).int()
                forward_batch = copy(forward_batch)
                forward_batch._ascend_kpool_metadata = metadata
            indices = self._decode_topk(
                q,
                key,
                weights,
                gate_score,
                positions,
                forward_batch,
                block_tables,
                layer_id,
            )
        return _as_ascend_sparse_indices(indices)
