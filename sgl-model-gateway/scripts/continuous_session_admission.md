# Continuous session admission for an online prefill expansion

`continuous_session_admission.py` is a Python 3.7+ standard-library CPU control
process for the existing session-handoff HTTP API. It supports the deployed
shape of two independent prefill instances (two attention DP domains each)
sharing one decode instance (eight attention DP domains). It never starts or
restarts model/gateway processes, edits session pins, or changes model inputs.

The gateway's `enrolled + migrated` counter is cumulative. A small fixed
`max_switches` is useful for a canary but eventually disables further admission.
This controller continuously sets the limit to `used + 1` when feedback permits.
An unused credit is not accumulated on each poll. The existing gateway enforces
one cold session and a minimum 60-second assignment interval. Its response leases
continue to prevent movement while a session has an unfinished request.

On each 30-second cycle the controller:

1. Reads the existing Prometheus metrics; it introduces no model/NPU polling.
2. Checks exact gateway container/start identity, the worker set and health,
   handoff persistence and the candidate-only prefill rollout.
3. Requires recent, finite metrics for all serving nodes and all four prefill
   rank/stage combinations; missing feedback never means idle.
4. Stops admission on excessive new-P pressure or a cold session in flight.
   A new candidate error latches admission closed until operator review.
5. Prefers newly observed eligible sessions for the new instance. It also
   selects up to four observed legacy sessions with a clean completion in the
   last 15 minutes and a previous request body no larger than 1 MiB. The list is
   refreshed continuously, excludes private validation hashes, and prefers
   smaller recent histories. Only the next eligible request can move a session.
6. Persists an intent before a configuration PUT, reads it back, then persists
   the acknowledged desired configuration. An ambiguous PUT is reconciled from
   the live API before any further credit is issued. Failed persistence blocks
   further grants until that reconciliation can be persisted successfully.

The supplied configuration is deployment-specific. Required keys are:

```json
{
  "gateway": "http://127.0.0.1:<front-port>",
  "legacy_gateway": "http://127.0.0.1:<original-gateway-port>",
  "prometheus": "http://<prometheus-host>:<port>",
  "cluster": "<metric-cluster-label>",
  "gateway_cid": "<full-container-id>",
  "gateway_started_at": "<exact-Docker-StartedAt>",
  "incumbent_urls": ["http://<original-prefill>:<port>"],
  "new_prefill_url": "http://<new-prefill>:<port>",
  "expected_workers": ["<all-12-exact-virtual-worker-URLs>"],
  "expected_nodes": ["<all-six-source-node-labels>"],
  "prefill_rank_locations": [
    {"service_id": "prefill-0", "node": "<old-P-node-0>", "dp_rank": "0"},
    {"service_id": "prefill-0", "node": "<old-P-node-1>", "dp_rank": "1"},
    {"service_id": "prefill-1", "node": "<new-P-node-0>", "dp_rank": "0"},
    {"service_id": "prefill-1", "node": "<new-P-node-1>", "dp_rank": "1"}
  ],
  "error_baseline": 0,
  "poll_secs": 30,
  "ops_lock": "<shared-operator-lock-path>",
  "singleton_lock": "<controller-process-lock-path>",
  "status_file": "<private-status.json>",
  "desired_file": "<existing-control-desired.json>",
  "private_hashes_file": "<private-validation-session-hashes.json>"
}
```

`error_baseline` must be a reviewed pre-change error count. Starting/restarting
the controller does not reset it. `desired_file` must already exist and contain
the operator's reviewed gateway control state. Only its migration intent is
updated. Model worker roles, virtual URLs and rollout remain externally managed.

```bash
python3 -B -m unittest discover -s . -p test_continuous_session_admission.py
python3 -B continuous_session_admission.py --config <config.json> --dry-run
python3 -B continuous_session_admission.py --config <config.json>
```

A gateway with `enabled=false` is never enabled by the controller. The supported
pause command must use the same `ops_lock` as the controller, or stop the
controller and wait for it to exit before disabling admission. Arbitrary direct
HTTP PUTs cannot provide compare-and-swap guarantees against concurrent writes.
Pausing leaves existing new-P session pins intact. Keep the candidate-only
prefill rollout enabled so their continuations remain serviceable.

The status JSON records fixed reason codes, feedback, cumulative usage and the
current selected hashes. Check its timestamp as well as the controller process;
an old successful status is not a current health check. Hashes never become
Prometheus labels or journal contents. No customer prompt bodies are collected.

Boundaries:

- Admission is a fraction of eligible sessions, additionally limited by credit,
  cold slots and time. It is not a guarantee about request traffic percentages.
- Previous body size is a selection heuristic, not an enforced size limit on the
  next request. Core response-boundary and cold-slot protections still apply.
- The existing API returns only the first 256 observed sessions. If truncated,
  `coverage=recent_metadata_incomplete` is reported; selection remains limited
  to visible recent sessions and does not claim full old-session coverage.
- The core's finite 65,536 lifetime/ledger capacity remains. Reaching it stops
  issuance; counters and pins are never cleared to extend the limit.
- Prometheus/core feedback has propagation delay. At most one already-issued
  credit can be consumed before a later gate change is applied.
- Gateway generation changes require operator review. The controller cannot
  recover a previous process's in-memory prefix-owner history.
