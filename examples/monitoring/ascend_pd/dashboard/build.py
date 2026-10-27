#!/usr/bin/env python3
"""Generate only this subtask's Grafana/Prometheus files from the read-only contract."""
import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((ROOT / "config/metric-contract.json").read_text())
CLUSTER = "glm53-910b"
DP = "cluster, model_name, role, service_id, node, dp_rank"
SERVICE = "cluster, model_name, role, service_id"
TOPOLOGY = "cluster, role, service_id"
NODE = "cluster, node, role, service_id"
DS = {"type": "prometheus", "uid": CONTRACT["grafana"]["datasource_uid"]}
FILTER = 'cluster=~"$cluster",model_name=~"$model"'
FRESH = 'llm_pd:dp_fresh{cluster=~"$cluster",model_name=~"$model"} == 1'
EXPECTED = 'llm_pd:expected_dp_ranks{cluster=~"$cluster"}'
WINDOW = "$window"


def select(metric, labels="", gateway=False):
    job = "llm-gateway" if gateway else "llm-pd"
    return f'{metric}{{job="{job}",{FILTER}{"," + labels if labels else ""}}}'


def pd(metric, labels="", dim="", rate=False):
    """Deduplicate TP/EP only after counter rate; retain each real attention DP."""
    source = select(metric, labels)
    if rate:
        source = f"rate({source}[{WINDOW}])"
    keys = DP + (", " + dim if dim else "")
    result = f"(max by ({keys}) ({source})) and on ({DP}) ({FRESH})"
    group = {"llm_pd_running_requests": "running", "llm_pd_running_context_tokens": "running",
             "llm_pd_queue_requests": "queues", "llm_pd_kv_tokens": "kv",
             "llm_pd_mamba_slots": "mamba", "llm_pd_prefix_tokens_total": "prefix",
             "llm_pd_spec_tokens_total": "spec", "llm_pd_spec_verify_total": "spec"}.get(metric)
    if group:
        available = select("llm_pd_metric_available", f'metric_group="{group}"')
        result += f" and on ({DP}) (max by ({DP}) ({available}) == 1)"
    return result


def complete_total(expr, dim=""):
    """Do not present partial rank telemetry as a complete service total."""
    keys = SERVICE + (", " + dim if dim else "")
    return (f"(sum by ({keys}) ({expr})) and on ({keys}) "
            f"(count by ({keys}) ({expr}) == on ({TOPOLOGY}) group_left() ({EXPECTED})) "
            f'and on ({TOPOLOGY}) llm_pd:service_complete{{cluster=~"$cluster"}}')


def gw(metric, labels="", function=None):
    expr = select(metric, labels, gateway=True)
    if function:
        expr = f"{function}({expr}[{WINDOW}])"
    return (f"({expr}) and on ({NODE}) "
            '(up{job="llm-gateway",cluster=~"$cluster"} == 1)')


def gateway_sum(metric, labels="", dim="", function="rate"):
    keys = "cluster, model_name" + (", " + dim if dim else "")
    return f"sum by ({keys}) ({gw(metric, labels, function)})"


def quantile(metric, q, labels="", dim=""):
    keys = "cluster, model_name" + (", " + dim if dim else "")
    buckets = gateway_sum(metric + "_bucket", labels, "le" + (", " + dim if dim else ""))
    count = gateway_sum(metric + "_count", labels, dim, "increase")
    return f"histogram_quantile({q}, {buckets}) and on ({keys}) ({count} > 0)"


def rules():
    records = []

    def record(name, expr):
        records.append({"record": name, "expr": expr})

    # Inventory is intentionally independent of exporter availability and model labels.
    # up remains present with value 0 on a failed scrape. No observed metric is zero-filled.
    expected_terms = [
        f'(count by ({NODE}) (up{{job="llm-pd",cluster="{CLUSTER}",'
        f'node="{target["node"]}"}}) * {target["expected_local_dp_ranks"]})'
        for target in CONTRACT["targets"]
    ]
    record("llm_pd:expected_local_dp_ranks", " or ".join(expected_terms))
    record("llm_pd:expected_dp_ranks", f"sum by ({TOPOLOGY}) (llm_pd:expected_local_dp_ranks)")
    record("llm_pd:dp_snapshot_age_seconds",
           f'max by ({DP}) (llm_pd_snapshot_age_seconds{{job="llm-pd"}}) '
           f'and on ({NODE}) (up{{job="llm-pd"}} == 1)')
    record("llm_pd:dp_fresh",
           "(llm_pd:dp_snapshot_age_seconds < bool 30) "
           "and (llm_pd:dp_snapshot_age_seconds >= 0)")
    record("llm_pd:observed_local_dp_ranks",
           f"count by ({NODE}) (count by ({NODE}, dp_rank) (llm_pd:dp_fresh == 1))")
    record("llm_pd:observed_dp_ranks",
           f"count by ({TOPOLOGY}) (count by ({TOPOLOGY}, dp_rank) (llm_pd:dp_fresh == 1))")
    record("llm_pd:coverage_ratio",
           f"llm_pd:observed_dp_ranks / on ({TOPOLOGY}) llm_pd:expected_dp_ranks")
    record("llm_pd:service_complete",
           f"llm_pd:observed_dp_ranks == on ({TOPOLOGY}) llm_pd:expected_dp_ranks")

    # Rules also need safe canonical pools and queue counts without Grafana variables.
    def canonical(metric, dimension="", labels=""):
        keys = DP + (", " + dimension if dimension else "")
        return (f'max by ({keys}) ({metric}{{job="llm-pd"{"," + labels if labels else ""}}}) '
                f"and on ({DP}) (llm_pd:dp_fresh == 1)")

    record("llm_pd:running_requests", canonical("llm_pd_running_requests"))
    record("llm_pd:queue_requests", canonical("llm_pd_queue_requests", "stage"))
    for pool in ("kv_tokens", "mamba_slots"):
        denominator = canonical("llm_pd_" + pool, labels='state="total"')
        for state in ("available", "evictable"):
            numerator = canonical("llm_pd_" + pool, labels=f'state="{state}"')
            record(f"llm_pd:{pool}_{state}_ratio", f"({numerator}) / (({denominator}) > 0)")

    alerts = []

    def alert(name, expr, duration, summary, description, severity="warning"):
        alerts.append({"alert": name, "expr": expr, "for": duration,
                       "labels": {"severity": severity, "scope": "local-only"},
                       "annotations": {"summary": summary, "description": description}})

    alert("LLMTargetDown", 'up{cluster="glm53-910b",job=~"llm-pd|llm-gateway"} == 0', "30s",
          "Exporter scrape failed on {{ $labels.node }}",
          "Inspect the target and scrape error locally. Missing metrics are unavailable, not idle.")
    alert("LLMDPCoverageIncomplete",
          f"llm_pd:expected_local_dp_ranks unless on ({NODE}) "
          f"(llm_pd:observed_local_dp_ranks == on ({NODE}) llm_pd:expected_local_dp_ranks)", "1m",
          "Fresh DP coverage incomplete on {{ $labels.node }}",
          "Expected P1 or D4 per node (P2/D8 globally). Missing or stale rank snapshots suppress service totals.")
    alert("LLMGlobalDPCoverageIncomplete",
          f"llm_pd:expected_dp_ranks unless on ({TOPOLOGY}) llm_pd:service_complete", "1m",
          "Logical {{ $labels.role }} instance DP coverage incomplete",
          "Check global DP identities: P ranks 0..1 and D ranks 0..7 across both machines.")
    alert("LLMDPSnapshotStale", "llm_pd:dp_snapshot_age_seconds >= 30", "30s",
          "DP snapshot stale on {{ $labels.node }} rank {{ $labels.dp_rank }}",
          "The exporter is reachable but its scheduler snapshot is at least 30 seconds old.")
    alert("LLMCollectorUnavailable", f'(llm_pd_metric_available{{job="llm-pd"}} == 0) and on ({DP}) '
          f'(max by ({DP}) (llm_pd_requests_total{{job="llm-pd",event="received"}}) > 0)', "2m",
          "Collector group unavailable: {{ $labels.metric_group }} on {{ $labels.node }}",
          "Zero is an explicit availability flag. Absent groups remain NA on the dashboard.")
    alert("LLMCollectorHealthMissing",
          f'(up{{job="llm-pd"}} == 1) unless on ({NODE}) llm_pd_metric_available{{job="llm-pd"}}', "2m",
          "Exporter has no collector availability flags on {{ $labels.node }}",
          "HTTP scrape success does not prove source collection is healthy.")
    alert("LLMDuplicateDPTelemetry",
          f'count by ({DP}) (llm_pd_running_requests{{job="llm-pd"}}) > 1', "1m",
          "Repeated TP/EP telemetry for DP {{ $labels.dp_rank }} on {{ $labels.node }}",
          "The exporter must publish once per attention DP leader. Dashboard max deduplicates copies before sum.")
    alert("LLMDPRankIdentityCollision",
          f"count by ({TOPOLOGY}, dp_rank) (count by ({NODE}, dp_rank) "
          "(llm_pd:dp_snapshot_age_seconds)) > 1", "1m",
          "DP rank {{ $labels.dp_rank }} appears on multiple nodes",
          "dp_rank must be global within the logical service, not a node-local index.")
    alert("LLMExpectedDPContractMismatch",
          f'max by ({NODE}) (llm_pd_expected_local_dp_ranks{{job="llm-pd"}}) '
          f"!= on ({NODE}) llm_pd:expected_local_dp_ranks", "1m",
          "Exporter DP expectation disagrees with inventory on {{ $labels.node }}",
          "Read-only contract expects P1/P1/D4/D4. Reconcile inventory before trusting aggregate panels.")
    pending_dp = f"(sum by ({DP}) (llm_pd:queue_requests) > 0)"
    for pool, title in (("kv_tokens", "KV token"), ("mamba_slots", "Mamba state slot")):
        alert("LLM" + ("KV" if pool == "kv_tokens" else "Mamba") + "HeadroomLow",
              f"(llm_pd:{pool}_available_ratio < 0.05) and on ({DP}) "
              f"(llm_pd:{pool}_evictable_ratio < 0.05) and on ({DP}) {pending_dp}", "3m",
              title + " headroom low with pending work on {{ $labels.node }} DP {{ $labels.dp_rank }}",
              "Available and evictable fractions are both below 5% with queued work for 3m. Evictable is not added to available because pool states may overlap. Missing state telemetry does not fire this rule.")
    # Pending + zero running + no service-level progress, not a high-load/latency alarm.
    pending = '(llm_pd:queue_requests{stage=~"decode_prealloc|decode_transfer|decode_ready|waiting"} > 0)'
    completed_dp = (f"max by ({DP}) "
                    '(rate(llm_pd_requests_total{job="llm-pd",role="decode",event="completed"}[5m])) '
                    f"and on ({DP}) (llm_pd:dp_fresh == 1)")
    completed = (f"(sum by ({SERVICE}) ({completed_dp}) == 0) and on ({SERVICE}) "
                 f"(count by ({SERVICE}) ({completed_dp}) == on ({TOPOLOGY}) group_left() llm_pd:expected_dp_ranks)")
    running_complete = (f'count by ({SERVICE}) (llm_pd:running_requests{{role="decode"}}) '
                        f"== on ({TOPOLOGY}) group_left() llm_pd:expected_dp_ranks")
    alert("LLMDecodePendingWithoutProgress",
          f"sum by ({SERVICE}) ({pending}) > 0 and on ({SERVICE}) "
          f'(sum by ({SERVICE}) (llm_pd:running_requests{{role="decode"}}) == 0) '
          f"and on ({SERVICE}) ({completed}) and on ({SERVICE}) ({running_complete}) "
          f"and on ({TOPOLOGY}) llm_pd:service_complete", "5m",
          "Decode has pending work, no running requests and no completions",
          "Sustained for 5m with complete fresh DP coverage. Check PD transfer and scheduling; no fixed TPOT or utilization threshold is assumed.")
    return {"groups": [{"name": "glm53-pd-recording", "interval": "5s", "rules": records},
                       {"name": "glm53-pd-local-alerts", "interval": "5s", "rules": alerts}]}


def prometheus():
    targets = []
    for t in CONTRACT["targets"]:
        targets.append({"targets": [f'{t["host"]}:{t["port"]}'],
                        "labels": {"cluster": CLUSTER, **{k: t[k] for k in ("node", "role", "service_id")}}})
    g = CONTRACT["gateway_target"]
    model_identity = [{"source_labels": ["model_name"], "regex": "/models/([^/]+)",
                       "target_label": "model_name", "replacement": "$1", "action": "replace"}]
    return {"global": {"scrape_interval": "5s", "scrape_timeout": "4s", "evaluation_interval": "5s"},
            "rule_files": ["alerts.yml"],
            "scrape_configs": [
                {"job_name": "llm-pd", "honor_labels": False, "metrics_path": "/metrics", "static_configs": targets,
                 "metric_relabel_configs": model_identity},
                {"job_name": "llm-gateway", "honor_labels": False, "metrics_path": "/metrics",
                 "metric_relabel_configs": model_identity,
                 "static_configs": [{"targets": [f'{g["host"]}:{g["port"]}'],
                                     "labels": {"cluster": CLUSTER, **{k: g[k] for k in ("node", "role", "service_id")}}}]}]}


def dashboard():
    panels = []
    y = 0

    def panel(title, targets, x, width, height=7, kind="timeseries", unit="short", description="", color=None, binary=False):
        defaults = {"unit": unit, "noValue": "NA", "color": {"mode": color or "palette-classic"},
                    "mappings": [{"type": "special", "options": {"match": "null", "result": {"text": "NA", "color": "text"}}}],
                    "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": None}]},
                    "custom": {"drawStyle": "line", "lineInterpolation": "linear", "lineWidth": 2,
                               "fillOpacity": 8, "showPoints": "never", "spanNulls": False,
                               "axisBorderShow": False, "axisCenteredZero": False, "axisColorMode": "text"}}
        options = {"legend": {"displayMode": "list", "placement": "bottom", "calcs": []},
                   "tooltip": {"mode": "multi", "sort": "desc"}}
        if kind == "stat":
            defaults["decimals"] = 2 if unit in ("s", "percentunit") else 0
            defaults["color"] = {"mode": "fixed", "fixedColor": "light-blue"}
            options = {"reduceOptions": {"values": False, "calcs": ["last"], "fields": ""},
                       "orientation": "auto", "textMode": "auto", "colorMode": "value",
                       "graphMode": "none", "justifyMode": "auto", "wideLayout": True}
        if kind == "state-timeline":
            defaults["color"] = {"mode": "continuous-GrYlRd"}
            defaults["custom"] = {"lineWidth": 0, "fillOpacity": 90, "spanNulls": False,
                                  "insertNulls": 15000, "hideFrom": {"legend": False, "tooltip": False, "viz": False}}
            defaults["min"] = 0
            options.update({"showValue": "never", "mergeValues": True, "rowHeight": 0.8,
                            "alignValue": "left", "legend": {"showLegend": False}})
            if binary:
                defaults["color"] = {"mode": "thresholds"}
                defaults["max"] = 1
                defaults["thresholds"]["steps"] = [{"color": "red", "value": None}, {"color": "green", "value": 1}]
                defaults["mappings"].append({"type": "value", "options": {
                    "0": {"text": "Unavailable", "color": "red"}, "1": {"text": "Available", "color": "green"}}})
        p = {"id": len(panels) + 1, "title": title, "type": kind, "datasource": DS,
             "gridPos": {"x": x, "y": y, "w": width, "h": height},
             "fieldConfig": {"defaults": defaults, "overrides": []}, "options": options,
             "description": description,
             "targets": [{"refId": chr(65 + i), "expr": expr, "legendFormat": legend,
                          "datasource": DS, "editorMode": "code", "range": kind != "stat",
                          "instant": kind == "stat", "interval": "5s", "format": "time_series"}
                         for i, (expr, legend) in enumerate(targets)]}
        if unit in ("short", "reqps", "tps"):
            defaults["min"] = 0
        panels.append(p)

    def row(title):
        nonlocal y
        panels.append({"id": len(panels) + 1, "type": "row", "title": title, "collapsed": False,
                       "gridPos": {"x": 0, "y": y, "w": 24, "h": 1}, "panels": []})
        y += 1

    panels.append({"id": 1, "type": "text", "title": "GLM-5.3 · 910B · PD observatory",
                   "gridPos": {"x": 0, "y": 0, "w": 24, "h": 3},
                   "options": {"mode": "markdown", "content":
                       "**P** 910b-16 + 28 · TP16 / DP2 / EP16　 **D** 910b-17 + 27 · TP16 / DP8 / EP16\n\n"
                       "One logical P → one logical D · Gateway timing is observed at ingress, not full client latency · "
                       "5s sampling · NA = missing / unsupported / no qualifying samples · Device timing OFF"}})
    y = 3
    req = gateway_sum("llm_gateway_requests_total")
    err = gateway_sum("llm_gateway_requests_total", 'outcome="error"')
    panel("Gateway · completed RPS", [(req, "{{model_name}} · all outcomes")], 0, 4, 4, "stat", "reqps")
    panel("Gateway · error fraction", [(f"({err}) / (({req}) > 0)", "{{model_name}}")], 4, 4, 4, "stat", "percentunit")
    panel("Gateway · output tokens/s", [(gateway_sum("llm_gateway_output_tokens_total"), "{{model_name}}")], 8, 4, 4, "stat", "tps")
    panel("Gateway · live stream inflight", [(gateway_sum("llm_gateway_inflight_requests", function=None), "{{model_name}}")], 12, 4, 4, "stat")
    panel("P · fresh DP / expected", [('llm_pd:observed_dp_ranks{cluster=~"$cluster",role="prefill"}', "observed P"),
                                     ('llm_pd:expected_dp_ranks{cluster=~"$cluster",role="prefill"}', "expected P")], 16, 4, 4, "stat")
    panel("D · fresh DP / expected", [('llm_pd:observed_dp_ranks{cluster=~"$cluster",role="decode"}', "observed D"),
                                     ('llm_pd:expected_dp_ranks{cluster=~"$cluster",role="decode"}', "expected D")], 20, 4, 4, "stat")
    y += 4

    row("01 / Gateway · request weighted latency & traffic")
    desc = ("Observed at the gateway, not the client. Sum classic histogram bucket rates before quantile. "
            "Window follows $window; no completed qualifying observations yields NA. Samples are window increase estimates.")
    latency = [("llm_gateway_ttft_seconds", 'first_event="semantic"', "Semantic TTFT · reasoning / text / tool arguments"),
               ("llm_gateway_ttft_seconds", 'first_event="visible"', "Visible TTFT · excludes reasoning"),
               ("llm_gateway_request_tpot_seconds", "", "Request TPOT · known final completion usage"),
               ("llm_gateway_e2e_seconds", "", "E2E · gateway body completion")]
    for i, (metric, labels, title) in enumerate(latency):
        panel(title, [(quantile(metric, q, labels), f"{{{{model_name}}}} · p{int(q*100)}")
                      for q in (0.50, 0.75, 0.90, 0.99)], (i % 2) * 12, 12, unit="s", description=desc)
        if i % 2:
            y += 7
    for i, (metric, labels, title) in enumerate(latency):
        panel(title.split(" · ")[0] + " · samples / window",
              [(gateway_sum(metric + "_count", labels, function="increase"), "{{model_name}} · samples")],
              i * 6, 6, 4, "stat", description=desc)
    y += 4
    panel("Gateway · request outcomes / s", [(gateway_sum("llm_gateway_requests_total", dim="outcome"), "{{model_name}} · {{outcome}}")], 0, 8, unit="reqps",
          description="Terminal request rates: success, error and cancellation remain separate. Inflight is measured over the full stream.")
    panel("Gateway · known usage tokens / s", [(gateway_sum("llm_gateway_prompt_tokens_total"), "{{model_name}} · prompt"),
                                             (gateway_sum("llm_gateway_output_tokens_total"), "{{model_name}} · output")], 8, 8, unit="tps")
    panel("Gateway · missing final usage / window", [(gateway_sum("llm_gateway_usage_missing_total", function="increase"), "{{model_name}} · missing usage")], 16, 8,
          description="Unknown usage is not zero tokens. TPOT requires completion_tokens > 1 and real first/last semantic timestamps; it is never inferred from chunks.")
    y += 7

    row("02 / Scheduler · actual concurrency & pending work")
    running = pd("llm_pd_running_requests")
    queue = pd("llm_pd_queue_requests", dim="stage")
    panel("P / D · actual running requests", [(complete_total(running), "{{role}} · {{model_name}}")], 0, 12,
          description="Logical requests before graph padding. One sample per attention DP; service totals require all expected fresh DP ranks. Not NPU utilization.")
    panel("P / D · queues by stage", [(complete_total(queue, "stage"), "{{role}} · {{stage}} · {{model_name}}")], 12, 12,
          description="Queue depths over time, in requests. Stages are separate pipeline populations and are not blindly summed into unique requests. Contract does not expose queue wait duration.")
    y += 7
    for role, x, title in (("prefill", 0, "P"), ("decode", 12, "D")):
        panel(title + " · concurrency by DP / time", [(pd("llm_pd_running_requests", f'role="{role}"'), "{{node}} · DP {{dp_rank}} · {{model_name}}")], x, 12, 8, "state-timeline",
              description="Heatmap: rows are actual attention DP ranks, x is time, color is logical concurrent requests. Gaps remain NA; no TP/EP replica sum.")
    y += 8
    panel("DP · queue depth by stage", [(queue, "{{role}} · {{node}} · DP {{dp_rank}} · {{stage}}")], 0, 16)
    event_targets = [(complete_total(pd("llm_pd_requests_total", f'role="{role}",event="{event}"', rate=True)), label)
                     for role, event, label in (("prefill", "kv_delivered", "P · KV delivered"), ("decode", "completed", "D · completed"))]
    panel("P delivery → D completion / s", event_targets, 16, 8, unit="reqps",
          description="P kv_delivered and D completed are independent event rates; transfer/stream residence makes short-window differences normal.")
    y += 7

    row("03 / Capacity · token pool & Mamba state slots")
    for pool, title, unit in (("kv_tokens", "KV token pool", "short"), ("mamba_slots", "Mamba state slots", "short")):
        all_states = pd("llm_pd_" + pool, dim="state")
        panel(title + " · logical service", [(complete_total(all_states, "state"), "{{role}} · {{state}}")], 0, 12, unit=unit,
              description="Used / available / evictable / total are distinct exporter states. Evictable may overlap used: never add states to derive capacity. Missing rank/state suppresses its total.")
        a = pd("llm_pd_" + pool, 'state="available"')
        t = pd("llm_pd_" + pool, 'state="total"')
        panel(title + " · immediate headroom by DP", [(f"({a}) / (({t}) > 0)", "{{role}} · {{node}} · DP {{dp_rank}}")], 12, 12, unit="percentunit",
              description="Available / total per DP, not a ratio averaged across ranks. Mamba is measured in slots, KV in tokens.")
        y += 7

    row("04 / Cache · P global hit, attribution & independent D reuse")
    p_sources = "hbm|host|ssd|unattributed|miss"
    p_raw = pd("llm_pd_prefix_tokens_total", f'role="prefill",source=~"{p_sources}"', "source", rate=True)
    p_cached = pd("llm_pd_prefix_tokens_total", 'role="prefill",source=~"hbm|host|ssd|unattributed"', "source", rate=True)
    p_miss = pd("llm_pd_prefix_tokens_total", 'role="prefill",source="miss"', rate=True)
    # Unknown tier details can be absent while an authoritative unattributed cached
    # total is present. Require known cached and miss on every DP, not fabricated
    # zero samples for unavailable hbm/host/ssd categories.
    p_cached_dp = f"sum by ({DP}) ({p_cached})"
    p_total_dp = f"(sum by ({DP}) ({p_raw})) and on ({DP}) ({p_miss}) and on ({DP}) ({p_cached_dp})"
    p_denominator = complete_total(p_total_dp)
    panel("P · global prefix hit incl. unattributed",
          [(f"({complete_total(p_cached_dp)}) / (({p_denominator}) > 0)", "{{model_name}} · global cached")], 0, 8, unit="percentunit",
          description="(hbm + host + ssd + unattributed) / all source tokens including miss. All expected DP ranks must have authoritative cached and miss rates. Unknown tier detail is not filled with zero.")
    for role, sources, x, title in (("prefill", p_sources, 8, "P · tiers incl. unattributed (未分层)"),
                                   ("decode", "local_reuse|transferred", 16, "D · independent local reuse / transferred")):
        raw = pd("llm_pd_prefix_tokens_total", f'role="{role}",source=~"{sources}"', "source", rate=True)
        if role == "prefill":
            known = pd("llm_pd_prefix_tokens_total", 'role="prefill",source=~"hbm|host|ssd"', "source", rate=True)
            tiers = select("llm_pd_metric_available", 'role="prefill",metric_group="prefix_tiers"')
            known = f'({known}) and on ({DP}) (max by ({DP}) ({tiers}) == 1)'
            other = pd("llm_pd_prefix_tokens_total", 'role="prefill",source=~"unattributed|miss"', "source", rate=True)
            raw = f'({known}) or ({other})'
        numer = complete_total(raw, "source")
        if role == "prefill":
            denom = p_denominator
        else:
            per_dp_denom = (f"(sum by ({DP}) ({raw})) and on ({DP}) "
                            f"(count by ({DP}) ({raw}) == 2)")
            denom = complete_total(per_dp_denom)
        panel(title, [(f"({numer}) / on ({SERVICE}) group_left() (({denom}) > 0)", "{{source}} · {{model_name}}")], x, 8, unit="percentunit",
              description="Sum per-DP counter rates before division; never average rank ratios. Unattributed = known cached tokens without trustworthy tier attribution (未分层). All categories are disjoint; missing tier details stay NA. D local_reuse excludes inherited P cache hits.")
    y += 7
    p_unattributed = pd("llm_pd_prefix_tokens_total", 'role="prefill",source="unattributed"', rate=True)
    panel("P · unattributed cached tokens / s (未分层)", [(p_unattributed, "{{node}} · DP {{dp_rank}} · unattributed / 未分层")], 0, 24, 5, unit="tps",
          description="Keep the authoritative unattributed cached amount visible even when another DP cannot expose this category. It is included in global hits and the all-source denominator; unavailable tier details are never treated as zero.")
    y += 5

    row("05 / MTP · strict counters, no inferred acceptance")
    accepted = complete_total(pd("llm_pd_spec_tokens_total", 'role="decode",kind="accepted"', rate=True))
    proposed = complete_total(pd("llm_pd_spec_tokens_total", 'role="decode",kind="proposed"', rate=True))
    emitted = complete_total(pd("llm_pd_spec_tokens_total", 'role="decode",kind="emitted"', rate=True))
    verify = complete_total(pd("llm_pd_spec_verify_total", 'role="decode"', rate=True))
    panel("MTP · accepted drafts / proposed drafts", [(f"({accepted}) / (({proposed}) > 0)", "{{model_name}} · strict acceptance")], 0, 8, unit="percentunit",
          description="Accepted counts actual accepted draft tokens only. Never reverse-engineered from acceptance length. Unsupported counters or zero proposed rate produce NA.")
    panel("MTP · acceptance length incl. bonus", [(f"({emitted}) / (({verify}) > 0)", "{{model_name}} · emitted / verify")], 8, 8,
          description="Optional: emitted must count accepted + bonus tokens for the same request verify calls. Uses emitted / verify only if both are available; never infers accepted. Missing metrics remain NA.")
    panel("MTP · supporting token / verify rates", [(accepted, "accepted drafts/s"), (proposed, "proposed drafts/s"),
                                                   (emitted, "emitted tokens/s"), (verify, "request verify calls/s")], 16, 8,
          description="These rates support both ratios. Verify counts request verification calls, not graph iterations or a TP/EP sum.")
    y += 7

    row("06 / Observability · coverage, freshness & collector health")
    panel("Targets · scrape availability", [('up{cluster=~"$cluster",job=~"llm-pd|llm-gateway"}', "{{node}} · {{role}}")], 0, 8, kind="state-timeline",
          description="0 = scrape failed, 1 = reachable. This only proves HTTP scrape availability, not collector correctness.", binary=True)
    panel("DP coverage · global observed vs inventory", [('llm_pd:observed_dp_ranks{cluster=~"$cluster"}', "{{role}} · fresh observed"),
                                                           ('llm_pd:expected_dp_ranks{cluster=~"$cluster"}', "{{role}} · expected")], 8, 8)
    panel("DP coverage · per-node observed vs expected", [('llm_pd:observed_local_dp_ranks{cluster=~"$cluster"}', "{{node}} · fresh observed"),
                                                             ('llm_pd:expected_local_dp_ranks{cluster=~"$cluster"}', "{{node}} · inventory expected"),
                                                             ('llm_pd_expected_local_dp_ranks{job="llm-pd",cluster=~"$cluster"}', "{{node}} · exporter expected")], 16, 8)
    y += 7
    panel("DP snapshot age · seconds", [('llm_pd:dp_snapshot_age_seconds{cluster=~"$cluster",model_name=~"$model"}', "{{role}} · {{node}} · DP {{dp_rank}}")], 0, 12, unit="s",
          description="At 30s or more a DP is stale and its operational values are excluded. A missing snapshot remains NA, never idle.")
    panel("Collector groups · explicit availability", [('llm_pd_metric_available{job="llm-pd",cluster=~"$cluster"}', "{{node}} · DP {{dp_rank}} · {{metric_group}}")], 12, 12, kind="state-timeline",
          description="Original metric_group labels are preserved without guessing their enum. 1 = available, 0 = unavailable, absent = NA; unsupported is never zero load.", binary=True)
    y += 7
    panel("Gateway · scrape health and local rule alerts", [('up{job="llm-gateway",cluster=~"$cluster"}', "Gateway reachable"),
                                                          ('count by (alertname, alertstate) (ALERTS{cluster=~"$cluster",scope="local-only"})', "{{alertstate}} · {{alertname}}")], 0, 12,
          description="Alerts evaluate locally. No Alertmanager, receiver, remote write or external notifications are configured. Empty alert results mean no active alert series.")
    panel("DP · concurrency telemetry availability", [('count by (cluster, role, service_id, node, dp_rank) (' + select("llm_pd_running_requests") + ')', "{{role}} · {{node}} · DP {{dp_rank}} · copies")], 12, 12,
          description="Expected one canonical running_requests series per attention DP leader. >1 indicates duplicated TP/EP or other unexpected labels; alert diagnoses duplication.")
    y += 7

    row("07 / Optional · concurrency residence time")
    residence = pd("llm_pd_concurrency_seconds_total", dim="requests", rate=True)
    resident_dp = f"sum by ({DP}) ({residence})"
    resident_total = complete_total(resident_dp)
    share = (f"(sum by ({SERVICE}, requests) ({residence})) / on ({SERVICE}) group_left() "
             f"(({resident_total}) > 0)")
    panel("Residence fraction · logical requests per DP", [(share, "{{role}} · concurrency {{requests}} · {{model_name}}")], 0, 24, unit="percentunit",
          description="Optional counter: seconds spent at each actual logical concurrency. Rank-seconds are normalized across all expected fresh DP ranks. This is a residence distribution, not a latency histogram; absent support stays NA.")

    return {"id": None, "uid": CONTRACT["grafana"]["dashboard_uid"], "title": "GLM-5.3 | 910B PD",
            "description": "One P (16/28, DP2) → one D (17/27, DP8). Gateway-observed latency, strict MTP, independent cache tiers and explicit telemetry health.",
            "tags": ["glm53", "910b", "PD"], "timezone": "browser", "schemaVersion": 39,
            "version": 1, "refresh": "5s", "editable": False, "graphTooltip": 1,
            "time": {"from": "now-30m", "to": "now"}, "timepicker": {"refresh_intervals": ["5s", "10s", "30s", "1m"]},
            "templating": {"list": [
                {"name": "cluster", "type": "constant", "label": "Cluster", "query": CLUSTER,
                 "current": {"text": CLUSTER, "value": CLUSTER}, "hide": 2},
                {"name": "model", "label": "Model", "type": "query", "datasource": DS,
                 "query": {"query": 'label_values({__name__=~"llm_pd_build_info|llm_gateway_requests_total",cluster="$cluster"}, model_name)', "refId": "model"},
                 "refresh": 1, "sort": 1, "multi": True, "includeAll": True, "allValue": ".*",
                 "current": {"selected": True, "text": "All", "value": "$__all"}},
                {"name": "window", "label": "Rate / quantile window", "type": "custom", "query": "1m,5m,15m,30m",
                 "current": {"selected": True, "text": "5m", "value": "5m"},
                 "options": [{"text": v, "value": v, "selected": v == "5m"} for v in ("1m", "5m", "15m", "30m")]}]},
            "annotations": {"list": []}, "links": [], "panels": panels}


def artifacts():
    datasource = {"apiVersion": 1, "datasources": [{"name": "GLM53 PD Prometheus", "type": "prometheus",
                  "uid": DS["uid"], "access": "proxy", "url": f'http://{CONTRACT["prometheus"]["host"]}:{CONTRACT["prometheus"]["port"]}',
                  "isDefault": True, "editable": False, "version": 1,
                  "jsonData": {"timeInterval": "5s", "httpMethod": "POST", "manageAlerts": False}}]}
    provider = {"apiVersion": 1, "providers": [{"name": "glm53-910b-pd", "orgId": 1, "folder": "GLM53 · 910B",
                "folderUid": "glm53-910b", "type": "file", "disableDeletion": True, "allowUiUpdates": False,
                "updateIntervalSeconds": 10, "options": {"path": "/var/lib/grafana/dashboards"}}]}
    header = "# Generated by monitoring/dashboard/build.py; edit the generator, then rebuild.\n"
    def yaml_text(value):
        return header + yaml.safe_dump(value, allow_unicode=True, sort_keys=False, width=120)
    return {ROOT / "config/prometheus.yml": yaml_text(prometheus()),
            ROOT / "config/alerts.yml": yaml_text(rules()),
            ROOT / "config/grafana/provisioning/datasources/glm53-pd.yml": yaml_text(datasource),
            ROOT / "config/grafana/provisioning/dashboards/glm53-pd.yml": yaml_text(provider),
            ROOT / "config/grafana/dashboards/glm53-910b-pd.json": json.dumps(dashboard(), ensure_ascii=False, indent=2) + "\n"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Compare generated files without writing")
    args = parser.parse_args()
    different = []
    for path, content in artifacts().items():
        if args.check:
            if not path.exists() or path.read_text() != content:
                different.append(str(path.relative_to(ROOT)))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    if different:
        raise SystemExit("Out of date: " + ", ".join(different))
    print("Generated files match." if args.check else "Generated 5 Grafana/Prometheus files.")


if __name__ == "__main__":
    main()
