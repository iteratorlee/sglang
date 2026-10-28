#!/usr/bin/env python3
"""Offline generator: combined old/new telemetry, topology and length histograms."""
import argparse
import copy
import json
from pathlib import Path
import yaml
import legacy

ROOT = Path(__file__).resolve().parents[1]
DP = "cluster, model_name, role, service_id, node, dp_rank"
GROUPS = {"global": "cluster, model_name", "prefill_dp": "cluster, model_name, prefill_dp_rank",
          "decode_dp": "cluster, model_name, node, dp_rank"}
QUANTILES = (0.50, 0.75, 0.90, 0.99)


def topology(new):
    cluster = "glm53-910b-4p2d" if new else "glm53-910b"
    native_cluster = "glm53-flagos-express-4p2d" if new else "glm53-flagos-express"
    prefill = (25, 26, 29, 30) if new else (16, 28)
    decode = (31, 32) if new else (17, 27)
    targets = []
    for role, hosts in (("prefill", prefill), ("decode", decode)):
        for index, host in enumerate(hosts):
            suffix = ("p" if role == "prefill" else "d") + str(index)
            ranks = [str(index)] if role == "prefill" else [str(4 * index + j) for j in range(4)]
            targets.append({"host": f"10.0.0.{host}", "port": 19100, "node": f"910b-{host}",
                            "cluster": cluster, "role": role, "service_id": role + "-0",
                            "source_cluster": native_cluster, "source_node": suffix,
                            "containers": ["glm53-flagos-express-1028-" + ("4p2d-" if new else "abi1-2p2d-") + suffix],
                            "expected_ranks": ranks, "expected_local_dp_ranks": len(ranks)})
    return {"schema": 1, "cluster": cluster, "dp_sizes": {"prefill": len(prefill), "decode": 8},
            "refresh_seconds": 5, "targets": targets,
            "gateway": {"host": f"10.0.0.{25 if new else 16}", "node": f"910b-{25 if new else 16}",
                        "port": 29266, "cluster": cluster, "role": "gateway", "service_id": "gateway-0"}}


def length_rules():
    records = []
    for scope, keys in GROUPS.items():
        for kind in ("input", "output"):
            bucket = f'llm_pd_request_{kind}_tokens_bucket{{job="llm-pd",role="decode"}}'
            # rate FIRST preserves counter resets separately for each D writer.
            expr = (f'(sum by ({keys}, status, le) (rate({bucket}[5m]) '
                    f'and on ({DP}) (llm_pd:dp_fresh == 1))) '
                    'and on (cluster) llm_pd:service_complete{role="decode"}')
            record = f"llm_pd:request_{kind}_tokens_{scope}:bucket_rate5m"
            records.append({"record": record, "expr": expr})
            for q in QUANTILES:
                records.append({"record": f"llm_pd:request_{kind}_tokens_{scope}:p{int(q*100)}_5m",
                                "expr": f"histogram_quantile({q}, {record})"})
    return {"name": "pd-request-lengths", "interval": "5s", "rules": records}


def length_expr(kind, scope, q):
    keys = GROUPS[scope]
    bucket = (f'llm_pd_request_{kind}_tokens_bucket{{job="llm-pd",role="decode",'
              'cluster=~"$cluster",model_name=~"$model",status=~"$status"}')
    return (f'histogram_quantile({q}, sum by ({keys}, le) (rate({bucket}[$window]) '
            f'and on ({DP}) (llm_pd:dp_fresh == 1))) '
            'and on (cluster) llm_pd:service_complete{role="decode",cluster=~"$cluster"}')


def artifacts():
    old, new = topology(False), topology(True)
    contract = copy.deepcopy(legacy.CONTRACT)
    contract.update(deployment="Parallel cluster-isolated old 2p2d and new 4p2d; one logical P and one logical D per cluster",
                    targets=old["targets"] + new["targets"], gateway_target=old["gateway"],
                    gateway_targets=[old["gateway"], new["gateway"]])
    contract["grafana"]["dashboard_uid"] = "glm53-pd-parallel-1028"
    contract["pd_metrics"].update({
        "llm_pd_request_input_tokens": "D terminal histogram: actual original unpadded prompt length, once per logical Req; extra labels prefill_dp_rank,status; dp_rank is D rank",
        "llm_pd_request_output_tokens": "D terminal histogram: actual generated output through stop, includes P handoff; cancelled/failed cohorts are partial; never requested max_new_tokens",
        "llm_pd_prefill_stage_input_tokens": "P stage terminal input histogram with status; no complete request output inference",
        "llm_pd_expected_local_dp_ranks": "gauge per node; aggregate by cluster,role,service_id for old P2/D8 or new P4/D8"})
    contract["notes"] = ["CPU-only validation; no 4p2d inference, live rollout or scrape performed by this subtask.",
        "All quantiles merge bucket rates before histogram_quantile. No per-rank quantile averaging.",
        "D terminal samples describe actual generated lengths, not client-delivered lengths; zero output is valid.",
        "P metadata comes from resolved receiver rank, then disagg_prefill_dp_rank, otherwise unknown.",
        "Missing/stale metrics and no qualifying traffic remain unavailable. Old native code has no new histograms.",
        "Gateway observation endpoints: old 16:29266, new 25:29266. Native/canonical clusters are distinct."]
    legacy.CONTRACT = contract
    config = legacy.prometheus()
    config["rule_files"] = ["alerts.yml", "request-lengths.rules.yml"]
    config["scrape_configs"][1]["static_configs"] = [
        {"targets": [f'{t["gateway"]["host"]}:{t["gateway"]["port"]}'],
         "labels": {k: t["gateway"][k] for k in ("cluster", "node", "role", "service_id")}}
        for t in (old, new)]
    # Source canonical labels are explicit and never derived from hostname suffix.
    for entry, target in zip(config["scrape_configs"][0]["static_configs"], legacy.CONTRACT["targets"]):
        entry["labels"]["cluster"] = target["cluster"]
    alerts = legacy.rules()
    inventory = alerts["groups"][0]["rules"]
    for rule in inventory:
        if rule.get("record") == "llm_pd:expected_local_dp_ranks":
            rule["expr"] = " or ".join(
                f'(count by ({legacy.NODE}) (up{{job="llm-pd",cluster="{t["cluster"]}",node="{t["node"]}"}}) * {len(t["expected_ranks"])})'
                for t in legacy.CONTRACT["targets"])
        if rule.get("record") in ("llm_pd:observed_local_dp_ranks", "llm_pd:observed_dp_ranks"):
            rule["expr"] = rule["expr"].replace("llm_pd:dp_fresh == 1", 'llm_pd:dp_fresh{dp_rank!="unknown"} == 1')
        if rule.get("record") == "llm_pd:service_complete":
            rule["expr"] = ("(" + rule["expr"] + ") unless on (cluster, role, service_id) "
                            '(count by (cluster, role, service_id) (llm_pd_exporter_source_up{job="llm-pd"} == 0))')
    for group in alerts["groups"]:
        for rule in group["rules"]:
            rule["expr"] = rule["expr"].replace('cluster="glm53-910b"', 'cluster=~"glm53-910b|glm53-910b-4p2d"') if "alert" in rule else rule["expr"]
            if "alert" in rule:
                for k, v in rule["annotations"].items():
                    rule["annotations"][k] = v.replace("P2/D8 globally", "P2/D8 old or P4/D8 new").replace("P ranks 0..1 and D ranks 0..7 across both machines", "P ranks 0..1 (old) or 0..3 (new), D ranks 0..7").replace("P1/P1/D4/D4", "one P rank per P host, four D ranks per D host")
    dash = legacy.dashboard()
    dash["uid"] = "glm53-pd-parallel-1028"
    dash["title"] = "GLM-5.3 | 2p2d + 4p2d"
    dash["description"] = "Cluster-isolated PD telemetry; complete-request token length histograms observed once at D terminal."
    dash["templating"]["list"][0] = {"name": "cluster", "label": "Cluster", "type": "custom", "query": old["cluster"] + "," + new["cluster"],
                                        "current": {"text": new["cluster"], "value": new["cluster"]}, "multi": False, "includeAll": False,
                                        "options": [{"text": t["cluster"], "value": t["cluster"], "selected": t is new} for t in (old, new)]}
    dash["templating"]["list"].append({"name": "status", "label": "Length terminal status", "type": "custom",
                                        "query": "completed,cancelled,failed", "multi": True, "includeAll": True, "allValue": ".*",
                                        "current": {"text": "completed", "value": "completed"}})
    dash["panels"][0]["options"] = {"mode": "html", "content":
        "Old: P16/28 DP2, D17/27 DP8. New: P25/26/29/30 DP4 TP8 EP32, D31/32 DP8 TP2 EP16. "
        "ctx1M W8A8+C8 FP32SSM MTP steps4 draft5. Select one cluster. "
        "Token lengths: D terminal actual lengths; stop included; cancelled/failed lengths are partial; unknown P rank remains visible. "
        "New length histograms require the new native package; old unmodified service shows NA. Sampling/cache 5s."}
    y = max(p["gridPos"]["y"] + p["gridPos"]["h"] for p in dash["panels"])
    for scope in GROUPS:
        for kind, x in (("input", 0), ("output", 12)):
            legend = "p{q} · " + ("P{{prefill_dp_rank}}" if scope == "prefill_dp" else "D{{dp_rank}} · {{node}}" if scope == "decode_dp" else "{{cluster}}")
            panel = {"id": len(dash["panels"]) + 1, "type": "timeseries", "title": f"{scope} · actual {kind} tokens · p50/p75/p90/p99",
                     "datasource": legacy.DS, "gridPos": {"x": x, "y": y, "w": 12, "h": 8},
                     "description": "Request-weighted bucket rates merged before quantile; no average of rank quantiles. Complete D coverage required. P grouping follows initial resolved dispatch rank. No traffic gives NA.",
                     "fieldConfig": {"defaults": {"unit": "short", "min": 0}, "overrides": []},
                     "options": {"legend": {"displayMode": "table", "placement": "bottom"}, "tooltip": {"mode": "multi"}},
                     "targets": [{"refId": chr(65+i), "expr": length_expr(kind, scope, q),
                                  "legendFormat": legend.replace("{q}", str(int(q*100)))} for i, q in enumerate(QUANTILES)]}
            dash["panels"].append(panel)
        y += 8
    dash["panels"].append({"id": len(dash["panels"])+1, "type": "timeseries", "title": "P stage input tokens only · p50/p75/p90/p99",
        "datasource": legacy.DS, "gridPos": {"x": 0, "y": y, "w": 24, "h": 8},
        "description": "Stage terminal observations on P; not complete-request output lengths and never added to D counts.",
        "targets": [{"refId": chr(65+i), "legendFormat": f"p{int(q*100)} · P{{{{dp_rank}}}}", "expr":
            f'histogram_quantile({q}, sum by (cluster, model_name, node, dp_rank, le) (rate(llm_pd_prefill_stage_input_tokens_bucket{{job="llm-pd",cluster=~"$cluster",model_name=~"$model",status=~"$status"}}[$window]) and on ({DP}) (llm_pd:dp_fresh == 1)))'} for i,q in enumerate(QUANTILES)]})
    output = {"config/topology-2p2d.json": old, "config/topology-4p2d.json": new, "config/metric-contract.json": contract,
              "config/prometheus.yml": config, "config/alerts.yml": alerts,
              "config/request-lengths.rules.yml": {"groups": [length_rules()]},
              "config/grafana/dashboards/glm53-pd-parallel.json": dash,
              "config/grafana/provisioning/datasources/glm53-pd.yml": {"apiVersion": 1, "datasources": [{"name": "GLM53 PD Prometheus", "uid": legacy.DS["uid"], "type": "prometheus", "access": "proxy", "url": "http://10.0.0.16:19090", "editable": False, "jsonData": {"timeInterval": "5s"}}]},
              "config/grafana/provisioning/dashboards/glm53-pd.yml": {"apiVersion": 1, "providers": [{"name": "glm53-pd-parallel", "folder": "GLM53 · 910B", "type": "file", "disableDeletion": True, "options": {"path": "/var/lib/grafana/dashboards"}}]}}
    for t in new["targets"]:
        output[f'config/native-env/{t["node"]}.json'] = {
            "SGLANG_PD_METRICS": "1", "SGLANG_PD_METRICS_ROLE": t["role"], "SGLANG_PD_METRICS_SERVICE_ID": t["service_id"],
            "SGLANG_PD_METRICS_CLUSTER": t["source_cluster"], "SGLANG_PD_METRICS_NODE": t["source_node"],
            "SGLANG_PD_METRICS_PREFILL_DP_SIZE": "4", "SGLANG_PD_METRICS_DECODE_DP_SIZE": "8", "SGLANG_PD_METRICS_VERSION": "lengths-v2-1028"}
    return {ROOT / name: json.dumps(obj, ensure_ascii=False, indent=2)+"\n" if name.endswith(".json") else yaml.safe_dump(obj, allow_unicode=True, sort_keys=False)
            for name,obj in output.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for path, text in artifacts().items():
        if args.check:
            assert path.read_text() == text, path
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
    print("Static configs verified" if args.check else "Generated old/new configs, 10 collectors, 2 gateways, rules and Grafana")


if __name__ == "__main__":
    main()
