"""Event-loop-local session leases for bounded MiniLB decode affinity.

Leases survive request completion, unlike in-flight reservations. New sessions
are placed by active lease count and observed work; existing sessions never
migrate to chase an instantaneous minimum. No future output lengths are used.
"""

import asyncio
from dataclasses import dataclass
import hashlib
import json
import time


@dataclass
class _Lease:
    rank: int
    last_activity: float
    active_since: float | None = None
    duration_ewma: float | None = None


class SessionReservation:
    def __init__(self, owner, key, rank, final, started):
        self._owner, self._key = owner, key
        self.rank, self.final, self.started = rank, final, started
        self._released = False
        self._successful = False

    def mark_successful(self):
        self._successful = True

    def mark_failed(self):
        self._successful = False

    def release(self):
        if self._released:
            return False
        self._released = True
        self._owner.release(self)
        return True


class SessionAffinity:
    """Bound session ownership and serialize requests of an existing session.

    ``inflight`` is the MiniLB's shared request-reservation vector, including
    legacy traffic. Idle leases expire on admission; active ones never expire.
    Elapsed response lifetime is only a placement heuristic, not decode timing.
    """

    def __init__(self, inflight, capacity, idle_timeout, request_timeout, clock=None):
        if not inflight or capacity <= 0 or idle_timeout <= 0 or request_timeout <= 0:
            raise ValueError("session affinity requires positive sizes and timeouts")
        self.inflight = inflight
        self.capacity = capacity
        self.idle_timeout = idle_timeout
        self.request_timeout = request_timeout
        self.clock = clock or time.monotonic
        self.leases = {}
        self.lease_counts = [0] * len(inflight)
        self.peak_lease_counts = [0] * len(inflight)
        self.completed_requests_by_rank = [0] * len(inflight)
        self.completed_lifetime_s_by_rank = [0.0] * len(inflight)
        self.global_duration_ewma = 1.0
        self.changed = asyncio.Event()
        self.counters = dict(
            new_sessions=0,
            reused_requests=0,
            completed_sessions=0,
            aborted_sessions=0,
            expired_sessions=0,
            requests=0,
            completed_requests=0,
            waited_requests=0,
            waiting_requests=0,
        )

    @staticmethod
    def key(session_id, request):
        if (
            not isinstance(session_id, str)
            or not 1 <= len(session_id) <= 256
            or any(ord(c) < 33 or ord(c) > 126 for c in session_id)
        ):
            raise ValueError(
                "session ID must contain 1..256 printable ASCII characters without spaces"
            )
        # Keep independent cache domains independent even if callers reuse an ID.
        raw = json.dumps(
            [session_id, request.get("cache_salt"), request.get("extra_key")],
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(raw.encode()).hexdigest()

    def _remove(self, key, reason):
        lease = self.leases.pop(key)
        self.lease_counts[lease.rank] -= 1
        self.counters[reason] += 1

    def _expire_idle(self, now):
        expired = [
            key
            for key, lease in self.leases.items()
            if lease.active_since is None
            and now - lease.last_activity >= self.idle_timeout
        ]
        for key in expired:
            self._remove(key, "expired_sessions")

    def _observed_work(self, rank, now):
        score = 0.0
        for lease in self.leases.values():
            if lease.rank != rank:
                continue
            expected = lease.duration_ewma or self.global_duration_ewma
            score += (
                0.25 * expected
                if lease.active_since is None
                else max(expected, now - lease.active_since)
            )
        return score

    def _choose(self, key, affinity_rank, eligible, now):
        # Capacity/active ownership comes first. Lifetime EWMA is a secondary,
        # online heuristic; it includes prefill and handoff, not just decode.
        scores = {
            rank: (
                self.lease_counts[rank],
                self._observed_work(rank, now),
                self.inflight[rank],
                self.completed_lifetime_s_by_rank[rank],
            )
            for rank in eligible
        }
        best = min(scores.values())
        tied = [rank for rank in eligible if scores[rank] == best]
        if affinity_rank in tied:
            return affinity_rank
        return tied[int(key[:16], 16) % len(tied)]

    async def reserve(self, session_id, request, affinity_rank, final=False):
        key = self.key(session_id, request)
        deadline = self.clock() + self.request_timeout
        waited = False
        try:
            while True:
                now = self.clock()
                self._expire_idle(now)
                lease = self.leases.get(key)
                if lease is None:
                    eligible = [
                        rank
                        for rank, count in enumerate(self.lease_counts)
                        if count < self.capacity and self.inflight[rank] < self.capacity
                    ]
                    if eligible:
                        rank = self._choose(key, affinity_rank, eligible, now)
                        lease = _Lease(rank, now)
                        self.leases[key] = lease
                        self.lease_counts[rank] += 1
                        self.peak_lease_counts[rank] = max(
                            self.peak_lease_counts[rank], self.lease_counts[rank]
                        )
                        self.counters["new_sessions"] += 1
                    else:
                        lease = None
                else:
                    rank = lease.rank
                    if (
                        lease.active_since is None
                        and self.inflight[rank] < self.capacity
                    ):
                        self.counters["reused_requests"] += 1
                if (
                    lease is not None
                    and lease.active_since is None
                    and self.inflight[lease.rank] < self.capacity
                ):
                    lease.active_since = now
                    lease.last_activity = now
                    self.inflight[lease.rank] += 1
                    self.counters["requests"] += 1
                    return SessionReservation(self, key, lease.rank, final, now)
                remaining = deadline - now
                if remaining <= 0:
                    raise TimeoutError("session affinity admission timed out")
                if not waited:
                    waited = True
                    self.counters["waited_requests"] += 1
                    self.counters["waiting_requests"] += 1
                self.changed.clear()
                try:
                    await asyncio.wait_for(self.changed.wait(), min(1.0, remaining))
                except asyncio.TimeoutError:
                    pass  # Recheck idle expiry; never evict an active session.
        finally:
            if waited:
                self.counters["waiting_requests"] -= 1

    def release(self, reservation):
        lease = self.leases[reservation._key]
        rank = lease.rank
        assert (
            rank == reservation.rank
            and self.inflight[rank] > 0
            and lease.active_since is not None
        )
        self.inflight[rank] -= 1
        now = self.clock()
        duration = max(0.0, now - reservation.started)
        lease.active_since = None
        lease.last_activity = now
        if reservation._successful:
            self.counters["completed_requests"] += 1
            self.completed_requests_by_rank[rank] += 1
            self.completed_lifetime_s_by_rank[rank] += duration
            lease.duration_ewma = (
                duration
                if lease.duration_ewma is None
                else 0.8 * lease.duration_ewma + 0.2 * duration
            )
            self.global_duration_ewma = 0.8 * self.global_duration_ewma + 0.2 * duration
            if reservation.final:
                self._remove(reservation._key, "completed_sessions")
        else:
            self._remove(reservation._key, "aborted_sessions")
        self.changed.set()

    def snapshot(self):
        return dict(
            enabled=True,
            capacity=self.capacity,
            idle_timeout_s=self.idle_timeout,
            active_sessions=len(self.leases),
            sessions_by_rank=list(self.lease_counts),
            peak_sessions_by_rank=list(self.peak_lease_counts),
            completed_requests_by_rank=list(self.completed_requests_by_rank),
            completed_lifetime_s_by_rank=list(self.completed_lifetime_s_by_rank),
            observed_work_semantics="router response lifetime EWMA; not pure decode time",
            **self.counters
        )
