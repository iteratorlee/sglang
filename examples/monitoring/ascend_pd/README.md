# Ascend disaggregated serving: per-DP Prometheus and Grafana

This opt-in example observes **one logical prefill instance across two hosts and one logical decode instance across two hosts**. It does not create four independent model replicas. The reference deployment has P attention DP2/TP8 + EP16 and D attention DP8/TP2 + EP16.

The scheduler only records existing CPU results and allocator metadata. It does not add device synchronization, tensor-to-host transfers, tokenization, profiling, or GPU/NPU utilization polling. HTTP scraping happens in separate CPU-only exporter processes, never in the inference loop.

## Enable native observations

Keep the model launch parameters and add `--enable-metrics`. Set the following before launching **every node** of each model instance:

```bash
export SGLANG_PD_METRICS=1
export SGLANG_PD_METRICS_CLUSTER=glm53-910b
export SGLANG_PD_METRICS_ROLE=prefill       # decode on D
export SGLANG_PD_METRICS_SERVICE_ID=prefill-0 # decode-0 on D
export SGLANG_PD_METRICS_NODE=910b-16       # this host's explicit identity
export SGLANG_PD_METRICS_VERSION=v1
```

Only each attention-TP leader publishes metrics. Global attention DP IDs must be unique within each logical instance, including across hosts. Preserve SGLang's automatically created `PROMETHEUS_MULTIPROC_DIR`; do not reuse one directory across model generations.

On D, append `--stream-response-default-include-usage` so ordinary chat streams expose real final token counts. This adds a terminal usage chunk; sampling and generated tokens are unchanged. Without known final usage, Gateway TPOT is deliberately unavailable.

Build the Gateway from this checkout with the normal project build procedure and retain its existing HTTP PD routing parameters. Add:

```bash
export SGLANG_GATEWAY_PD_METRICS=1
export SGLANG_GATEWAY_PD_METRICS_MODEL=GLM-5.3-Flash-w8a8
```

Its existing `--prometheus-port 29266` exports the new `llm_gateway_*` families. The gate defaults to off. Original request JSON and response bytes are preserved.

## Per-host CPU exporter

`exporter/exporter.py` requires Python 3.10+ and `prometheus_client`; it does not import torch. The reference model image already contains these dependencies. It uses the official `MultiProcessCollector`, filters dead writer processes, checks PID/start-time/container generations, and serves cached results every 5s.

Set `EXPORTER_CODE` to the absolute path of this example's `exporter/` directory. For the reference 910b-16 node:

```bash
docker run -d --name glm53-pd-exporter-910b-16 \
  --restart unless-stopped --runtime runc --network host --pid host \
  --read-only --cap-drop ALL --cap-add SYS_PTRACE \
  --security-opt no-new-privileges --security-opt seccomp=unconfined \
  --cpus 2 --memory 768m --pids-limit 64 \
  --tmpfs /tmp:rw,nosuid,nodev,size=64m \
  --log-driver json-file --log-opt max-size=10m --log-opt max-file=2 \
  --mount type=bind,src=/var/run/docker.sock,dst=/var/run/docker.sock,readonly \
  --mount type=bind,src="$EXPORTER_CODE",dst=/exporter,readonly \
  -e PYTHONNOUSERSITE=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  --entrypoint python3 quay.io/ascend/sglang:main-cann9.0.0-910b \
  -B /exporter/exporter.py --node 910b-16 \
  --container YOUR_EXACT_MODEL_CONTAINER_NAME \
  --listen 0.0.0.0 --port 19100 --refresh-seconds 5
```

Repeat for 910b-28/17/27 with the exact model-container name. `--container` is repeatable for at most four explicit lifecycle candidates; at most one may run. `seccomp=unconfined` is needed by the reference cluster's older Docker configuration; it is unnecessary where the default profile works. No NPU devices or privileged mode are used by the exporter. The read-only socket mount does not restrict Docker API permissions; the implementation only issues allowlisted inspect GETs, and a protocol-filtering socket proxy can enforce that boundary independently.

The reference inventory is deliberately explicit:

| Node | Role / logical service | Global DP ranks |
|---|---|---|
| 910b-16 | prefill / prefill-0 | 0 |
| 910b-28 | prefill / prefill-0 | 1 |
| 910b-17 | decode / decode-0 | 0,1,2,3 |
| 910b-27 | decode / decode-0 | 4,5,6,7 |

For another cluster, update `exporter.py:TARGETS`, `config/metric-contract.json` targets and `dashboard/build.py`'s reference settings consistently, then regenerate configurations. Unknown/missing sources are unavailable, not zero load. `/healthz` tests the HTTP process, `/readyz` requires all expected fresh DP snapshots, and `/status` gives sanitized discovery/collection health.

## Prometheus and Grafana

`config/prometheus.yml` scrapes four node exporters and the existing Gateway endpoint. A metric relabel rule maps `/models/NAME` to `NAME`, aligning native and Gateway model identities for dashboard filters. Adjust that explicit reference-path rule for another model layout. Use `--config.file=/etc/prometheus/prometheus.yml`, keep `alerts.yml` alongside it, and validate with `promtool check config`. Reference Prometheus listens on 19090 with 5s scrape/evaluation intervals. There is no external Alertmanager or notification receiver configured.

Provision Grafana with these read-only mounts:

| Local path | Container target |
|---|---|
| `config/grafana/provisioning` | `/etc/grafana/provisioning` |
| `config/grafana/dashboards` | `/var/lib/grafana/dashboards` |

Set `GF_SERVER_HTTP_PORT=13000`, a private `GF_SECURITY_ADMIN_PASSWORD`, `GF_AUTH_ANONYMOUS_ENABLED=false` and `GF_USERS_ALLOW_SIGN_UP=false`; keep Grafana data in a persistent writable volume. Do not commit credentials. Datasource UID is `glm53-pd-prometheus`; Dashboard UID is `glm53-910b-pd`.

The 42 data panels include Gateway TTFT/TPOT/E2E p50/p75/p90/p99, sample counts, success/error/cancellation rates, actual execution and mutually exclusive queue stages, per-DP load, KV tokens, Mamba slots, local prefix reuse, MTP acceptance, and independent P-delivery/D-completion rates. Complete service aggregates require fresh P2/D8 coverage; missing data is not zero-filled.

TTFT distinguishes first semantic output (including reasoning) and visible output (text/tool arguments). TPOT uses first/last semantic times and real final completion tokens, not SSE frame counts. Nonstream responses have no decode TPOT sample. These are Gateway observations, excluding the last network hop to the client; histogram percentiles are bucket estimates, not exact offline sample percentiles. D prefix reuse is measured before transfer and is independent of carried P cache statistics. Accepted drafts exclude the bonus token.

## CPU validation

```bash
python -m unittest discover -s test/registered/unit/observability -p test_pd_monitoring.py
python -m unittest discover -s examples/monitoring/ascend_pd/exporter/tests
python examples/monitoring/ascend_pd/exporter/native_contract_check.py \
  --module python/sglang/srt/observability/pd_monitoring.py --out /tmp/pd-contract.json
python examples/monitoring/ascend_pd/dashboard/build.py --check
```

The exporter tests use actual subprocess-written Prometheus multiprocess files, cancellation/generation boundaries and HTTP exposition parsing. Bind local test sockets only where allowed. The native observation module fails open: an instrumentation error increments its own bounded error counter and cannot terminate inference. For rollback, unset the two metrics gates and restart in a controlled order; preserve the original model topology and serving parameters.
