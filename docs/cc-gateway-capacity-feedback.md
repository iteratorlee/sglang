---
title: "CC Gateway KV capacity feedback"
description: "Wire contract and safety rules for routing with scheduler KV capacity snapshots."
---

# CC Gateway KV capacity feedback

This change extends `/v1/loads` with low-cost scheduler capacity and same-host
age signals. It reuses `SchedulerPoolStatsObserver`; it does not query device
tensors, add a collective, or enter the decode kernel path.

## JSON contract

Each item in `loads` has a core scalar:

```json
{
  "snapshot_monotonic_s": 12345.5,
  "kv_capacity": {
    "full_available_tokens": 520640,
    "full_evictable_tokens": 8192,
    "swa_available_tokens": 4096,
    "swa_evictable_tokens": 1024,
    "mamba_available_slots": 6,
    "mamba_evictable_slots": 0,
    "request_slots_available": 1
  }
}
```

The HTTP response envelope has a second monotonic sample:

```json
{
  "server_monotonic_s": 12345.75,
  "loads": []
}
```

`kv_capacity` is an optional section. The full-pool fields are present when a
new scheduler publishes the section. SWA fields appear only for hybrid SWA
pools, Mamba fields appear only for hybrid SSM pools, and
`request_slots_available` appears only when the scheduler observer exposes a
host-side request pool.

## Capacity semantics

- `full_available_tokens` is the observer's scheduler-visible full-pool
  capacity in logical tokens. It is not derived as
  `max_total_num_tokens - num_used_tokens`.
- `full_evictable_tokens` is the radix cache's separate evictable ledger. It is
  a cache-pressure hint. It is not guaranteed admission capacity.
- The SWA pair has the same available-versus-evictable split in SWA token
  units.
- Mamba values count request-scoped state slots, not tokens.
- `request_slots_available` is the current host-side free-row count from
  `ReqToTokenPool`.

The Gateway must never calculate a hard capacity value by adding available and
evictable tokens. In particular, decode
`_materialize_radix_full_allocatable_tokens` can make the FULL evictable ledger
larger than the amount physically releasable for the next admission. Protected
nodes, page granularity, concurrent scheduler changes, and shared hybrid-pool
frontiers further limit what can be reclaimed. A router may use evictable
capacity as a soft preference or cache-pressure signal. Admission should use a
fresh available value with an explicit safety margin and current stage/inflight
load; the snapshot itself is not a reservation.

Unified-memory allocators may report capacity that is realizable after their
existing bounded peer compaction. This is still separate from radix eviction
and comes directly from the observer's allocator view.

## GLM-5.3 units and conservation

The GLM-5.3 decode cache uses 64-token physical allocation pages and a
256-token full radix-tree page. Existing production-path CPU coverage verifies
that a finished 8192-token prefix yields:

| Counter | Value |
| --- | ---: |
| Full capacity | 528896 logical tokens |
| Full available | 520704 logical tokens |
| Full evictable | 8192 logical tokens |
| Full used | 0 logical tokens |

After allocating one uncached 64-token physical page, available becomes
520640, evictable remains 8192, and used becomes 64. The new snapshot copies
these reconciled observer counters exactly. It does not reinterpret the
256-token eviction ledger as immediately free physical capacity.

## Snapshot age

`snapshot_monotonic_s` is sampled by the scheduler while it builds the load
snapshot. `server_monotonic_s` is sampled by the `/v1/loads` HTTP process while
it assembles the response. Both clocks have the same host-local monotonic
origin, so the Gateway can calculate:

```text
snapshot_age_s = max(0, server_monotonic_s - snapshot_monotonic_s)
```

Do not compare either value with the Gateway's local monotonic clock or with a
sample from another P/D host. A value of zero means that the producer did not
publish the new field and age is unknown.

If the Gateway caches a response, the effective age is the source-side age
above plus time elapsed on the Gateway since that response arrived. Gateway
receive time alone misses a snapshot that was already stale inside the D host.

## Compatibility and cost

`LoadSnapshot` is map-shaped msgpack. New readers default missing monotonic and
capacity fields when they receive an old payload. Old readers ignore the new
map keys. The capacity struct omits unavailable optional values. The existing
wall-clock `timestamp`, queue semantics (`running`, `queues.waiting`, and
`prealloc_ready`), and SHM/ZMQ transport remain unchanged.

One existing `get_pool_stats()` call now supplies both legacy usage metrics and
the new capacity section. Request slots use a host list length, and monotonic
age uses one clock read at each producer. No additional device synchronization,
collective, or model kernel is introduced.

## CPU evidence

The following coverage passed in the local `sgl-0511` CPU environment:

- 5 `test_load_inquirer.py` tests, including direct observer counter
  propagation and monotonic sampling.
- 2 targeted SHM/msgpack tests for section filtering, round trip, and old/new
  decoder compatibility.
- 1 `/v1/loads` test for same-host age calculation.
- 1 existing GLM-5.3 full-only pool test covering the real 256-token radix
  ledger and 64-token allocator conservation path.

The installed local `xgrammar` predates this source tree, so the test process
provided aliases for six structural-tag classes that are unrelated to load
snapshots. The full snapshot backend suite also contains ZMQ IPC tests that the
workspace sandbox cannot bind; the two new SHM-only tests passed independently.
Native deployment validation remains required before enabling Gateway
admission decisions.
