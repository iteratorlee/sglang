# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU-only, per-attention-DP instrumentation for disaggregated serving.

No model/device imports, tensor conversions, synchronization, network, disk
logging, request IDs as labels, or background threads. The native Prometheus
client owns exposition/storage. Enabled explicitly with SGLANG_PD_METRICS=1.
Logical active requests are scheduler observations, not device utilization.
"""

from __future__ import annotations

import logging
import math
from numbers import Integral
import os
import time


LOGGER = logging.getLogger(__name__)
LABELS = ("cluster", "model_name", "role", "service_id", "node", "dp_rank")
INPUT_TOKEN_BUCKETS = (0, 256, 512, 1024, 2048, 4096, 8192, 12288, 16384,
                       24576, 32768, 49152, 65536, 81920, 98304, 114688,
                       131072, 147456, 163840, 196608, 229376, 262144,
                       294912, 327680, 393216, 458752, 524288, 655360,
                       786432, 917504, 1048576)
OUTPUT_TOKEN_BUCKETS = (0, 1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384,
                        512, 768, 1024, 1536, 2048, 3072, 4096, 6144,
                        8192, 12288, 16384, 24576, 32768, 49152, 65536,
                        81920, 131072)
TERMINAL_STATUSES = ("completed", "cancelled", "failed")
_observer = None
_configured = False
_last_warning = 0.0


def _integer(value):
    # int(tensor) and .item() may synchronize a device: do neither.
    if not isinstance(value, Integral) or isinstance(value, bool) or value < 0:
        raise ValueError("expected a nonnegative CPU integer")
    return int(value)


def _health(req):
    return "HEALTH_CHECK" in str(getattr(req, "rid", ""))


def _rank(value, size):
    # No int(tensor), user-controlled strings, negative ranks, or fallback zero.
    if isinstance(value, str) and value.isascii() and value.isdecimal() and len(value) <= 2:
        value = int(value)
    if isinstance(value, Integral) and not isinstance(value, bool) and 0 <= value < size:
        return str(value)
    return "unknown"


class PDObserver:
    """One owner per logical attention-DP rank and process lifecycle."""

    def __init__(self, *, labels, config, registry=None, clock=time.monotonic):
        from prometheus_client import Counter, Gauge, Histogram

        self.labels = {k: str(labels[k]) for k in LABELS}
        self.role = self.labels["role"]
        if self.role not in ("prefill", "decode"):
            raise ValueError("PD observer requires a prefill or decode role")
        self.config = config
        self.prefill_dp_size = _integer(config.get("prefill_dp_size", 2))
        self.decode_dp_size = _integer(config.get("decode_dp_size", 8))
        if not 1 <= self.prefill_dp_size <= 64 or not 1 <= self.decode_dp_size <= 64:
            raise ValueError("metrics DP sizes must be in 1..64")
        self.labels["dp_rank"] = _rank(
            labels["dp_rank"], self.prefill_dp_size if self.role == "prefill" else self.decode_dp_size
        )
        self.clock = clock
        self._last_state = self.clock()
        self._last_refresh = -math.inf
        self._count = 0
        self._batches = {}
        self._prefix_tiers_valid = True
        self._children = {}
        self._metrics = {}

        def metric(name, description, extra=(), counter=False):
            cls = Counter if counter else Gauge
            kwargs = {} if counter else {"multiprocess_mode": "mostrecent"}
            if registry is not None:
                kwargs["registry"] = registry
            self._metrics[name] = cls(
                name, description, LABELS + tuple(extra), **kwargs
            )

        metric(
            "llm_pd_running_requests",
            "Logical executing requests before graph padding.",
        )
        metric(
            "llm_pd_queue_requests",
            "Requests in mutually exclusive PD lifecycle queues.",
            ("stage",),
        )
        metric("llm_pd_kv_tokens", "Full KV pool capacity in token slots.", ("state",))
        metric(
            "llm_pd_mamba_slots",
            "Active recurrent-state pool capacity in slots, not tokens.",
            ("state",),
        )
        metric(
            "llm_pd_running_context_tokens",
            "Input plus generated tokens of logical running requests.",
        )
        metric(
            "llm_pd_requests_total",
            "Per-DP request lifecycle events.",
            ("event",),
            True,
        )
        metric(
            "llm_pd_prefix_tokens_total",
            "Distinct prompt tokens by actual local cache source.",
            ("source",),
            True,
        )
        metric(
            "llm_pd_spec_tokens_total",
            "Draft acceptance/proposal and emitted tokens from existing CPU results.",
            ("kind",),
            True,
        )
        metric(
            "llm_pd_spec_verify_total",
            "Number of request verification forwards.",
            counter=True,
        )
        metric(
            "llm_pd_concurrency_seconds_total",
            "Scheduler wall time at logical request concurrency.",
            ("requests",),
            True,
        )
        metric(
            "llm_pd_last_snapshot_monotonic_seconds",
            "Last successful CPU snapshot on this host monotonic clock.",
        )
        metric(
            "llm_pd_metric_available",
            "Whether the named metric group has valid observations.",
            ("metric_group",),
        )
        metric(
            "llm_pd_instrumentation_errors_total",
            "Rejected/failed observations, never inference failures.",
            ("operation",),
            True,
        )
        metric(
            "llm_pd_build_info",
            "Monitoring deployment/configuration identity.",
            (
                "tp_size",
                "dp_size",
                "ep_size",
                "context_length",
                "draft_tokens",
                "num_steps",
                "topk",
                "kv_dtype",
                "monitor_version",
            ),
        )

        self.child("llm_pd_running_requests").set(0)
        self.child("llm_pd_running_context_tokens").set(0)
        self.child("llm_pd_concurrency_seconds_total", requests="0").inc(0)
        groups = ("running", "queues", "kv", "mamba", "prefix")
        groups += ("prefix_tiers",) if self.role == "prefill" else ("spec",)
        for group in groups:
            self.available(group, group == "running")
        for event in ("received", "kv_delivered", "completed", "failed", "cancelled"):
            self.child("llm_pd_requests_total", event=event).inc(0)
        sources = (
            ("hbm", "host", "ssd", "unattributed", "miss")
            if self.role == "prefill"
            else ("local_reuse", "transferred")
        )
        for source in sources:
            self.child("llm_pd_prefix_tokens_total", source=source).inc(0)
        if self.role == "decode":
            for kind in ("accepted", "proposed", "emitted"):
                self.child("llm_pd_spec_tokens_total", kind=kind).inc(0)
            self.child("llm_pd_spec_verify_total").inc(0)
        values = {
            k: str(config.get(k, "unknown"))
            for k in (
                "tp_size",
                "dp_size",
                "ep_size",
                "context_length",
                "draft_tokens",
                "num_steps",
                "topk",
                "kv_dtype",
                "monitor_version",
            )
        }
        self.child("llm_pd_build_info", **values).set(1)

        def histogram(name, description, extra):
            kwargs = {} if registry is None else {"registry": registry}
            self._metrics[name] = Histogram(
                name, description, LABELS + extra,
                buckets=OUTPUT_TOKEN_BUCKETS if name == "llm_pd_request_output_tokens" else INPUT_TOKEN_BUCKETS, **kwargs
            )

        if self.role == "decode":
            for kind in ("input", "output"):
                histogram(
                    "llm_pd_request_" + kind + "_tokens",
                    "Actual logical request " + kind + " length at D terminal, once per attention TP leader; stop included.",
                    ("prefill_dp_rank", "status"),
                )
                # Zero baselines make the first request observable by rate().
                # Bounded by configured P DP size, independent of request IDs.
                for rank in [str(i) for i in range(self.prefill_dp_size)] + ["unknown"]:
                    for status in TERMINAL_STATUSES:
                        self.child("llm_pd_request_" + kind + "_tokens", prefill_dp_rank=rank, status=status)
        else:
            histogram("llm_pd_prefill_stage_input_tokens",
                      "Actual P stage input length at stage terminal; not a complete request output distribution.",
                      ("status",))
            for status in TERMINAL_STATUSES:
                self.child("llm_pd_prefill_stage_input_tokens", status=status)

    def prefill_resolved(self, req, rank):
        if self.role != "decode" or _health(req):
            return
        # Retain the initial dispatch domain through rebootstrap/retraction.
        # An unresolved/invalid value may be replaced by a later real resolution.
        if getattr(req, "_pd_metrics_prefill_dp_rank", "unknown") == "unknown":
            req._pd_metrics_prefill_dp_rank = _rank(rank, self.prefill_dp_size)

    def request_lengths(self, req, event):
        # Called only at terminal paths, never from the model/token loop.
        if getattr(req, "_pd_metrics_lengths_finished", False):
            return
        status = "completed" if event == "kv_delivered" else event
        if status not in TERMINAL_STATUSES:
            raise ValueError("invalid request terminal status")
        # set_finish_with_abort replaces origin_input_ids by [0]. The original
        # unpadded prompt remains intact and is the logical text-token length.
        prompt = getattr(req, "origin_input_ids_unpadded", None)
        if prompt is None:
            prompt = req.origin_input_ids
        input_len = len(prompt)
        if self.role == "prefill":
            observations = [("llm_pd_prefill_stage_input_tokens", {"status": status}, input_len)]
        else:
            output_len = len(req.output_ids)
            # MTP can append accepted tokens beyond a stop; use the same stop
            # position as output_ids_through_stop without slicing/copying IDs.
            stop = getattr(req, "finished_len", None)
            if stop is not None:
                output_len = min(output_len, _integer(stop))
            elif getattr(req, "pd_rebootstrap_forced_output_id", None) is not None:
                # Rebootstrap temporarily pops one already generated token.
                output_len += 1
            rank = getattr(req, "_pd_metrics_prefill_dp_rank", "unknown")
            if rank == "unknown":
                rank = getattr(req, "disagg_prefill_dp_rank", None)
            extra = {"prefill_dp_rank": _rank(rank, self.prefill_dp_size), "status": status}
            observations = [("llm_pd_request_input_tokens", extra, input_len),
                            ("llm_pd_request_output_tokens", extra, output_len)]
        # Claim once before writes: a rare storage error must not double count
        # on a repeated terminal callback. _call exposes instrumentation errors.
        req._pd_metrics_lengths_finished = True
        for name, extra, count in observations:
            self.child(name, **extra).observe(count)

    def child(self, name, **extra):
        key = (name, tuple(sorted(extra.items())))
        child = self._children.get(key)
        if child is None:
            child = self._metrics[name].labels(**self.labels, **extra)
            self._children[key] = child
        return child

    def available(self, group, value):
        self.child("llm_pd_metric_available", metric_group=group).set(int(bool(value)))

    def _transition(self, count, now=None):
        count = _integer(count)
        now = self.clock() if now is None else now
        elapsed = now - self._last_state
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("non-monotonic observation")
        if count == self._count and elapsed < 1.0:
            return
        if elapsed:
            self.child(
                "llm_pd_concurrency_seconds_total", requests=str(self._count)
            ).inc(elapsed)
        if count != self._count:
            self.child("llm_pd_running_requests").set(count)
        self._count, self._last_state = count, now

    @staticmethod
    def _requests(batch):
        return [req for req in batch.reqs if not _health(req)]

    def batch_started(self, batch):
        if batch.forward_mode.is_prebuilt() or batch.forward_mode.is_idle():
            return
        requests = self._requests(batch)
        if self.role == "prefill":
            # Overlap scheduling passes batch.copy() to the completion path.
            # Request identity survives that copy; batch object identity does not.
            key = tuple(sorted(id(req) for req in requests))
            count, _ = self._batches.get(key, (0, requests))
            self._batches[key] = (count + 1, requests)
            requests = self._prefill_requests()
            self._transition(len(requests))
        else:
            self._transition(len({id(req) for req in requests}))
        self.child("llm_pd_running_context_tokens").set(
            sum(len(req.origin_input_ids) + len(req.output_ids) for req in requests)
        )

    def batch_finished(self, batch):
        if batch.forward_mode.is_prebuilt() or batch.forward_mode.is_idle():
            return
        if self.role == "prefill":
            key = tuple(sorted(id(req) for req in self._requests(batch)))
            count, requests = self._batches.get(key, (0, ()))
            if count > 1:
                self._batches[key] = (count - 1, requests)
            else:
                self._batches.pop(key, None)
            requests = self._prefill_requests()
            self._transition(len(requests))
            self.child("llm_pd_running_context_tokens").set(
                sum(len(req.origin_input_ids) + len(req.output_ids) for req in requests)
            )
        else:
            remaining = [r for r in self._requests(batch) if not r.finished()]
            self._transition(len({id(r) for r in remaining}))
            self.child("llm_pd_running_context_tokens").set(
                sum(len(r.origin_input_ids) + len(r.output_ids) for r in remaining)
            )

    def _prefill_requests(self):
        active = {id(req): req for _, reqs in self._batches.values() for req in reqs}
        return list(active.values())

    def received(self, req):
        if not _health(req):
            self.child("llm_pd_requests_total", event="received").inc()

    @staticmethod
    def _abort_event(abort):
        reason = getattr(abort, "finished_reason", None)
        if getattr(abort, "abort_message", None):
            return "failed"
        if reason is None:
            return "cancelled"
        return (
            "cancelled"
            if isinstance(reason, dict) and reason.get("status_code") == 499
            else "failed"
        )

    def abort_requested(self, scheduler, abort):
        # Rare control path only. Track the actual origin: FINISH_ABORT(None)
        # alone cannot distinguish a client cancellation from a serving error.
        requests = list(scheduler.waiting_queue) + list(
            scheduler.collect_inflight_reqs()
        )
        if scheduler.chunked_req is not None:
            requests.append(scheduler.chunked_req)
        if self.role == "prefill":
            requests += list(scheduler.disagg_prefill_bootstrap_queue.queue)
            requests += list(scheduler.disagg_prefill_inflight_queue)
        else:
            queue = scheduler.disagg_decode_prealloc_queue
            requests += list(queue.queue) + list(queue.retracted_queue)
            requests += list(scheduler.disagg_decode_transfer_queue.queue)
        for entry in requests:
            req = getattr(entry, "req", entry)
            if not req.finished() and (
                abort.abort_all or req.rid.startswith(abort.rid)
            ):
                req._pd_metrics_abort_event = self._abort_event(abort)

    def aborted(self, req, abort):
        # Queued/retracted removals send AbortReq directly, without the normal
        # output streamer. Observe that existing terminal path once.
        if not _health(req) and not getattr(req, "_pd_metrics_finished", False):
            event = self._abort_event(abort)
            self.request_lengths(req, event)
            self.child("llm_pd_requests_total", event=event).inc()
            req._pd_metrics_finished = True

    def finished(self, req):
        if (
            _health(req)
            or not req.finished()
            or getattr(req, "_pd_metrics_finished", False)
        ):
            return
        # The caller has already excluded finished_output duplicates.
        reason = req.finished_reason
        aborted = type(reason).__name__ == "FINISH_ABORT"
        if aborted:
            event = getattr(req, "_pd_metrics_abort_event", None) or (
                "cancelled" if getattr(reason, "status_code", None) == 499 else "failed"
            )
        else:
            event = "kv_delivered" if self.role == "prefill" else "completed"
        self.request_lengths(req, event)
        self.child("llm_pd_requests_total", event=event).inc()
        req._pd_metrics_finished = True
        if self.role == "prefill" and not aborted:
            total = len(req.origin_input_ids)
            cached = _integer(req.cached_tokens)
            if cached > total:
                raise ValueError("cached prompt exceeds input length")
            parts = [
                getattr(req, key, None)
                for key in (
                    "cached_tokens_device",
                    "cached_tokens_host",
                    "cached_tokens_storage",
                )
            ]
            valid = all(
                isinstance(x, Integral) and not isinstance(x, bool) and x >= 0
                for x in parts
            )
            if valid and sum(parts) == cached:
                for source, value in zip(("hbm", "host", "ssd"), parts):
                    self.child("llm_pd_prefix_tokens_total", source=source).inc(
                        int(value)
                    )
            else:
                self.child("llm_pd_prefix_tokens_total", source="unattributed").inc(
                    cached
                )
                self._prefix_tiers_valid = False
            self.child("llm_pd_prefix_tokens_total", source="miss").inc(total - cached)
            self.available("prefix", True)
            self.available("prefix_tiers", self._prefix_tiers_valid)

    def decode_transfer(self, decode_req):
        if self.role != "decode" or _health(decode_req.req):
            return
        if getattr(decode_req, "_pd_metrics_transfer", False):
            return
        audit = decode_req.glm53_cache_audit
        if audit is None:
            self.available("prefix", False)
            return
        total = _integer(audit["input_len"])
        local = _integer(audit["actual_prefix_len"])
        if local > total:
            raise ValueError("decode-local reuse exceeds input length")
        self.child("llm_pd_prefix_tokens_total", source="local_reuse").inc(local)
        self.child("llm_pd_prefix_tokens_total", source="transferred").inc(
            total - local
        )
        self.available("prefix", True)
        decode_req._pd_metrics_transfer = True

    def spec_verify(self, bs, correct, emitted, active_config):
        if self.role != "decode":
            return
        bs, correct, emitted = map(_integer, (bs, correct, emitted))
        width = _integer(active_config["num_draft_tokens"])
        if self.config.get("topk") != 1 or width < 2:
            self.available("spec", False)
            return
        proposed = bs * (width - 1)
        if correct > proposed or emitted < correct:
            raise ValueError("invalid CPU speculative counts")
        for kind, value in (
            ("accepted", correct),
            ("proposed", proposed),
            ("emitted", emitted),
        ):
            self.child("llm_pd_spec_tokens_total", kind=kind).inc(value)
        self.child("llm_pd_spec_verify_total").inc(bs)
        self.available("spec", True)

    def refresh(self, scheduler, now=None, idle=False):
        now = self.clock() if now is None else now
        if idle and not self._batches:
            self._transition(0, now)
        if now - self._last_refresh < 1.0:
            return
        self._last_refresh = now
        self._transition(self._count, now)
        queue_values = {}
        if self.role == "prefill":
            paths = {
                "waiting": ("waiting_queue",),
                "bootstrap": ("disagg_prefill_bootstrap_queue", "queue"),
                "prefill_inflight": ("disagg_prefill_inflight_queue",),
            }
        else:
            paths = {
                "decode_ready": ("waiting_queue",),
                "decode_prealloc": ("disagg_decode_prealloc_queue", "queue"),
                "decode_transfer": ("disagg_decode_transfer_queue", "queue"),
                "retracted": ("disagg_decode_prealloc_queue", "retracted_queue"),
            }
        for stage, path in paths.items():
            queue = scheduler
            for attr in path:
                queue = getattr(queue, attr)
            queue_values[stage] = len(queue)
        for stage, value in queue_values.items():
            self.child("llm_pd_queue_requests", stage=stage).set(value)
        self.available("queues", True)

        # Already-maintained host allocator metadata only, no device memory API.
        stats = scheduler.pool_stats_observer.get_pool_stats()
        full = [
            _integer(getattr(stats, attr))
            for attr in ("full_num_used", "full_available_size", "full_evictable_size")
        ]
        for state, value in zip(
            ("used", "available", "evictable", "total"), full + [sum(full)]
        ):
            self.child("llm_pd_kv_tokens", state=state).set(value)
        self.available("kv", True)
        if stats.is_hybrid_ssm:
            mamba = [
                _integer(getattr(stats, attr))
                for attr in (
                    "mamba_num_used",
                    "mamba_available_size",
                    "mamba_evictable_size",
                )
            ]
            for state, value in zip(
                ("used", "available", "evictable", "total"), mamba + [sum(mamba)]
            ):
                self.child("llm_pd_mamba_slots", state=state).set(value)
            self.available("mamba", True)
        else:
            self.available("mamba", False)
        self.child("llm_pd_last_snapshot_monotonic_seconds").set(now)


def _call(operation, *args, **kwargs):
    global _last_warning
    if _observer is None:
        return
    try:
        getattr(_observer, operation)(*args, **kwargs)
    except Exception:
        # Instrumentation failure must never terminate model serving.
        try:
            _observer.child(
                "llm_pd_instrumentation_errors_total", operation=operation
            ).inc()
        except Exception:
            pass
        now = time.monotonic()
        if now - _last_warning >= 60:
            _last_warning = now
            LOGGER.warning(
                "PD metrics observation rejected: %s", operation, exc_info=True
            )


def configure(scheduler):
    global _observer, _configured
    if _configured or os.environ.get("SGLANG_PD_METRICS") != "1":
        return
    _configured = True
    if (
        scheduler.ps.attn_tp_rank != 0
        or scheduler.ps.pp_rank != 0
        or scheduler.ps.attn_cp_rank != 0
    ):
        return
    try:
        args = scheduler.server_args
        role = os.environ["SGLANG_PD_METRICS_ROLE"]
        config = {
            "tp_size": args.tp_size,
            "dp_size": args.dp_size,
            "ep_size": args.ep_size,
            "context_length": args.context_length,
            "draft_tokens": args.speculative_num_draft_tokens,
            "num_steps": args.speculative_num_steps,
            "topk": args.speculative_eagle_topk,
            "kv_dtype": args.kv_cache_dtype,
            "monitor_version": os.environ.get("SGLANG_PD_METRICS_VERSION", "v1"),
            "prefill_dp_size": int(os.environ.get("SGLANG_PD_METRICS_PREFILL_DP_SIZE", "2")),
            "decode_dp_size": int(os.environ.get("SGLANG_PD_METRICS_DECODE_DP_SIZE", "8")),
        }
        labels = {
            "cluster": os.environ.get("SGLANG_PD_METRICS_CLUSTER", "glm53-910b"),
            "model_name": args.served_model_name or args.model_path,
            "role": role,
            "service_id": os.environ["SGLANG_PD_METRICS_SERVICE_ID"],
            "node": os.environ["SGLANG_PD_METRICS_NODE"],
            "dp_rank": scheduler.ps.dp_rank if scheduler.ps.dp_rank is not None else "unknown",
        }
        _observer = PDObserver(labels=labels, config=config)
        LOGGER.info(
            "PD CPU metrics enabled: role=%s node=%s dp=%s",
            role,
            labels["node"],
            labels["dp_rank"],
        )
    except Exception:
        LOGGER.exception("PD metrics initialization failed; inference remains enabled")


def batch_started(batch):
    _call("batch_started", batch)


def batch_finished(batch):
    _call("batch_finished", batch)


def heartbeat(scheduler, now_ns, idle):
    _call("refresh", scheduler, now=now_ns / 1e9, idle=idle)


def received(req):
    _call("received", req)


def abort_requested(scheduler, abort):
    _call("abort_requested", scheduler, abort)


def aborted(req, abort):
    _call("aborted", req, abort)


def finished(req):
    _call("finished", req)


def decode_transfer(req):
    _call("decode_transfer", req)


def prefill_resolved(req, rank):
    _call("prefill_resolved", req, rank)


def spec_verify(bs, correct, emitted, active_config):
    _call("spec_verify", bs, correct, emitted, active_config)
