#!/usr/bin/env python3
"""Offline contract, provisioning, layout and real PromQL parser validation.

No HTTP requests, GPU, Prometheus server or Grafana server are used.
Requires existing PyYAML and the small optional promql-parser wheel.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


class UniqueLoader(yaml.SafeLoader):
    pass


def unique_mapping(loader, node, deep=False):
    pairs = loader.construct_pairs(node, deep=deep)
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate YAML key: {key}")
        result[key] = value
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)


def read_yaml(path):
    return yaml.load(path.read_text(), Loader=UniqueLoader)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parser-path", type=Path, help="Temporary directory containing promql_parser")
    args = parser.parse_args()
    if args.parser_path:
        sys.path.insert(0, str(args.parser_path))
    try:
        import promql_parser
    except ImportError:
        raise SystemExit("Need promql-parser for real syntax validation; pass --parser-path (see README).")

    contract = json.loads((ROOT / "config/metric-contract.json").read_text())
    config = read_yaml(ROOT / "config/prometheus.yml")
    rule_file = read_yaml(ROOT / "config/alerts.yml")
    ds_file = read_yaml(ROOT / "config/grafana/provisioning/datasources/glm53-pd.yml")
    provider = read_yaml(ROOT / "config/grafana/provisioning/dashboards/glm53-pd.yml")
    dash = json.loads((ROOT / "config/grafana/dashboards/glm53-910b-pd.json").read_text())

    require(config["global"]["scrape_interval"] == config["global"]["evaluation_interval"] == "5s", "5s scrape/eval")
    require(config["global"]["scrape_timeout"] == "4s", "Timeout must fit scrape interval")
    require(config["rule_files"] == ["alerts.yml"], "Rule file mount must resolve beside prometheus.yml")
    require("alerting" not in config and "remote_write" not in config, "Local-only monitoring")
    jobs = {j["job_name"]: j for j in config["scrape_configs"]}
    require(set(jobs) == {"llm-pd", "llm-gateway"}, "Only known PD/Gateway targets; no guessed NPU ports")
    for job, expected in ((jobs["llm-pd"], contract["targets"]), (jobs["llm-gateway"], [contract["gateway_target"]])):
        require(job["honor_labels"] is False, "Target labels must win conflicts")
        actual = job["static_configs"]
        require(len(actual) == len(expected), "Target count")
        for item, target in zip(actual, expected):
            require(item["targets"] == [f'{target["host"]}:{target["port"]}'], "Contract target address")
            require(item["labels"] == {"cluster": "glm53-910b", **{k: target[k] for k in ("node", "role", "service_id")}}, "Contract target labels; no fabricated dp_rank")
    require({t["service_id"] for t in contract["targets"]} == {"prefill-0", "decode-0"}, "Two logical services")
    require([sum(t["expected_local_dp_ranks"] for t in contract["targets"] if t["role"] == role)
             for role in ("prefill", "decode")] == [2, 8], "Expected P2/D8 attention DP coverage")

    datasource = ds_file["datasources"][0]
    require(datasource["uid"] == contract["grafana"]["datasource_uid"], "Datasource UID")
    require(datasource["url"] == f'http://{contract["prometheus"]["host"]}:{contract["prometheus"]["port"]}', "Prometheus URL")
    require(datasource["jsonData"]["timeInterval"] == "5s" and datasource["access"] == "proxy", "Server-side datasource, 5s floor")
    require(provider["providers"][0]["options"]["path"] == "/var/lib/grafana/dashboards", "Dashboard mount")
    require(dash["uid"] == contract["grafana"]["dashboard_uid"] and dash["refresh"] == "5s", "Dashboard identity")
    require(next(v for v in dash["templating"]["list"] if v["name"] == "window")["current"]["value"] == "5m", "Default rate window")

    queries = []
    recording_names = set()
    for group in rule_file["groups"]:
        require(group["interval"] == "5s", "Rule interval")
        for rule in group["rules"]:
            if "record" in rule:
                recording_names.add(rule["record"])
            else:
                require(rule["labels"]["scope"] == "local-only" and "for" in rule, "Alert debounce and local scope")
            queries.append((rule.get("record", rule.get("alert")), rule["expr"]))
    ids = set()
    rectangles = []
    data_panels = []
    for p in dash["panels"]:
        require(p["id"] not in ids, "Unique panel id")
        ids.add(p["id"])
        g = p["gridPos"]
        require(g["w"] > 0 and g["h"] > 0 and 0 <= g["x"] < g["x"] + g["w"] <= 24, "24-column grid bounds")
        rect = (g["x"], g["y"], g["x"] + g["w"], g["y"] + g["h"])
        for r in rectangles:
            require(not (rect[0] < r[2] and r[0] < rect[2] and rect[1] < r[3] and r[1] < rect[3]), f'Overlapping panel: {p["title"]}')
        rectangles.append(rect)
        if not p.get("targets"):
            continue
        data_panels.append(p)
        require(p["datasource"]["uid"] == datasource["uid"], "Panel datasource UID")
        require(p["fieldConfig"]["defaults"]["noValue"] == "NA", "No-data rendering")
        require(p["fieldConfig"]["defaults"]["custom"].get("spanNulls") is False, "Do not bridge missing data")
        for t in p["targets"]:
            require(t["datasource"]["uid"] == datasource["uid"], "Target datasource UID")
            if p["type"] == "stat":
                require(t["instant"] and not t["range"], "Stat must not reuse an old non-null range value")
            queries.append((p["title"] + "/" + t["refId"], t["expr"]))

    histogram_names = {k for k, v in contract["gateway_metrics"].items() if "histogram" in v}
    allowed_metrics = (set(contract["pd_metrics"]) | set(contract["gateway_metrics"]) | recording_names
                       | {"up", "ALERTS"} | {h + s for h in histogram_names for s in ("_bucket", "_sum", "_count")})
    quantiles = 0
    for name, query in queries:
        require(not re.search(r"\bor\s+(?:on\s*\([^)]*\)\s*)?vector\s*\(\s*0", query), f"No zero fill: {name}")
        require(not re.search(r"\bavg(?:_over_time)?\s*\(", query), f"No averaged rank ratios/percentiles: {name}")
        expanded = query.replace("$window", "5m").replace("$cluster", "glm53-910b").replace("$model", ".*")
        require("$" not in expanded, f"Unresolved variable: {name}")
        try:
            tree = promql_parser.parse(expanded)
        except Exception as e:
            raise ValueError(f"PromQL error in {name}: {e}\n{expanded}") from e
        selectors = []
        histogram_calls = []

        def inspect(node):
            if isinstance(node, promql_parser.VectorSelector):
                selectors.append(node)
            if isinstance(node, promql_parser.AggregateExpr):
                require(str(node.op) != "avg", f"No averaged ratios/percentiles: {name}")
            if isinstance(node, promql_parser.Call) and node.func.name == "histogram_quantile":
                histogram_calls.append(node)

        promql_parser.walk(tree, inspect)
        used = {s.name for s in selectors if s.name}
        require(used <= allowed_metrics, f"Unknown contract metric in {name}: {used - allowed_metrics}")
        require(tree is not None, f"Parser returned no expression: {name}")
        for call in histogram_calls:
            quantiles += 1
            bucket_input = call.args[1]
            require(isinstance(bucket_input, promql_parser.AggregateExpr) and str(bucket_input.op) == "sum"
                    and bucket_input.modifier is not None and "le" in bucket_input.modifier.labels,
                    "Aggregate classic buckets with le retained before quantile")
            require("_count" in query and "> 0" in query, "Quantile requires nonzero qualifying samples")
        if name.startswith("P · global prefix hit"):
            sources = {m.value for s in selectors if s.name == "llm_pd_prefix_tokens_total"
                       for m in s.matchers.matchers if m.name == "source"}
            require("hbm|host|ssd|unattributed" in sources and "hbm|host|ssd|unattributed|miss" in sources,
                    "P global cached numerator and all-source denominator must retain unattributed")
        if name.startswith("MTP · accepted drafts / proposed drafts"):
            kinds = {m.value for s in selectors if s.name == "llm_pd_spec_tokens_total"
                     for m in s.matchers.matchers if m.name == "kind"}
            require(kinds == {"accepted", "proposed"}, "Strict MTP ratio uses accepted/proposed only")
    require(quantiles == 16, "Four p50/75/90/99 histograms: semantic/visible TTFT, request TPOT, E2E")
    require(sum("samples / window" in p["title"] for p in data_panels) == 4, "Each histogram needs a sample-count panel")
    required_alerts = {"LLMTargetDown", "LLMDPCoverageIncomplete", "LLMGlobalDPCoverageIncomplete", "LLMDPSnapshotStale",
                       "LLMCollectorUnavailable", "LLMKVHeadroomLow", "LLMMambaHeadroomLow", "LLMDecodePendingWithoutProgress"}
    require(required_alerts <= {r.get("alert") for g in rule_file["groups"] for r in g["rules"]}, "Expected cautious local alerts")

    # Reproducibility: compare, never regenerate over unreviewed edits during validation.
    subprocess.run([sys.executable, str(ROOT / "dashboard/build.py"), "--check"], check=True)
    promtool = shutil.which("promtool")
    if promtool:
        subprocess.run([promtool, "check", "config", str(ROOT / "config/prometheus.yml")], check=True)
        subprocess.run([promtool, "check", "rules", str(ROOT / "config/alerts.yml")], check=True)
    print(f"PASS: contract, 5 provisioning/config artifacts, {len(data_panels)} data panels, "
          f"{len(queries)} parsed PromQL expressions, 16 histogram quantiles, nonoverlapping grid.")
    print("promtool checks passed." if promtool else "promtool unavailable: parser/structure checks only; no live execution claimed.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error))
