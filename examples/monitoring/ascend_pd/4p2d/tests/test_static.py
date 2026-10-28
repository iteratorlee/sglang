import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import yaml
import promql_parser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exporter"))
sys.path.insert(0, str(ROOT / "dashboard"))
import exporter
import build


class StaticContracts(unittest.TestCase):
    def test_new_topology_exact_six_nodes_global_ranks_and_old_is_unchanged(self):
        new = exporter.load_topology(ROOT / "config/topology-4p2d.json")
        expected = {25:("0",),26:("1",),29:("2",),30:("3",),31:("0","1","2","3"),32:("4","5","6","7")}
        self.assertEqual({int(k.split("-")[1]):v.expected_ranks for k,v in new.items()},expected)
        self.assertTrue(all(v.cluster == "glm53-910b-4p2d" and v.source_cluster == "glm53-flagos-express-4p2d" for v in new.values()))
        old = exporter.load_topology(ROOT / "config/topology-2p2d.json")
        self.assertEqual({k:v.expected_ranks for k,v in old.items()}, {"910b-16":("0",),"910b-28":("1",),"910b-17":("0","1","2","3"),"910b-27":("4","5","6","7")})
        self.assertTrue(all(v.cluster == "glm53-910b" and v.source_cluster == "glm53-flagos-express" for v in old.values()))

    def test_topology_rejects_duplicate_missing_unbounded_ranks_and_wildcards(self):
        original = json.loads((ROOT / "config/topology-4p2d.json").read_text())
        def invalid(change):
            data = copy.deepcopy(original); change(data)
            with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
                path = Path(directory) / "topology.json"; path.write_text(json.dumps(data))
                with self.assertRaises((ValueError,KeyError,TypeError)):
                    exporter.load_topology(path)
        invalid(lambda d:d["targets"][1].update(expected_ranks=["0"]))
        invalid(lambda d:d["targets"].pop())
        invalid(lambda d:d["targets"][0].update(containers=["glm*"]))
        invalid(lambda d:d["dp_sizes"].update(prefill=999))
        invalid(lambda d:d["targets"][0].pop("source_cluster"))

    def test_combined_prometheus_ten_collectors_two_gateways_cluster_isolation(self):
        config = yaml.safe_load((ROOT / "config/prometheus.yml").read_text())
        self.assertEqual(config["global"]["scrape_interval"],"5s")
        jobs = {j["job_name"]:j for j in config["scrape_configs"]}
        collectors = jobs["llm-pd"]["static_configs"]
        self.assertEqual(len(collectors),10)
        self.assertEqual(len({(c["labels"]["cluster"],c["labels"]["node"]) for c in collectors}),10)
        self.assertEqual({c["targets"][0] for c in jobs["llm-gateway"]["static_configs"]}, {"10.0.0.16:29266","10.0.0.25:29266"})
        self.assertEqual({c["targets"][0] for c in collectors if c["labels"]["cluster"]=="glm53-910b-4p2d"},
                         {f"10.0.0.{n}:19100" for n in (25,26,29,30,31,32)})

    def test_real_promql_parser_accepts_all_rules_and_dashboard_queries(self):
        count = 0
        for name in ("alerts.yml","request-lengths.rules.yml"):
            data = yaml.safe_load((ROOT / "config" / name).read_text())
            for group in data["groups"]:
                for rule in group["rules"]:
                    promql_parser.parse(rule["expr"]); count += 1
        dash = json.loads((ROOT / "config/grafana/dashboards/glm53-pd-parallel.json").read_text())
        for panel in dash["panels"]:
            for target in panel.get("targets",[]):
                expr = target.get("expr")
                if expr:
                    expr = expr.replace("$window","5m").replace("$cluster","glm53-910b-4p2d").replace("$model","glm").replace("$status","completed")
                    promql_parser.parse(expr); count += 1
        self.assertGreater(count,100)

    def test_all_quantiles_merge_buckets_first_and_preserve_required_domains(self):
        rules = yaml.safe_load((ROOT / "config/request-lengths.rules.yml").read_text())["groups"][0]["rules"]
        self.assertEqual(len(rules),30)
        for rule in rules:
            expr = rule["expr"]
            self.assertNotIn("avg",expr)
            if "bucket_rate5m" == rule["record"].split(":")[-1]:
                self.assertIn("sum by",expr); self.assertIn("status, le",expr)
                self.assertIn("rate(llm_pd_request_",expr)
                self.assertIn("cluster, model_name",expr)
                self.assertIn('role="decode"',expr)
            else:
                self.assertTrue(expr.startswith("histogram_quantile("))
        for scope in build.GROUPS:
            for kind in ("input","output"):
                for q in build.QUANTILES:
                    expr = build.length_expr(kind,scope,q)
                    self.assertIn("sum by",expr); self.assertIn("rate(",expr)
                    self.assertNotIn("max_new_tokens",expr)
                    self.assertIn(build.GROUPS[scope]+", le",expr)

    def test_generated_static_files_are_reproducible(self):
        for path,text in build.artifacts().items():
            self.assertEqual(path.read_text(),text,str(path))

    def test_native_env_declares_correct_observer_cardinality(self):
        for n in (25,26,29,30,31,32):
            env=json.loads((ROOT/f"config/native-env/910b-{n}.json").read_text())
            self.assertEqual(env["SGLANG_PD_METRICS_PREFILL_DP_SIZE"],"4")
            self.assertEqual(env["SGLANG_PD_METRICS_DECODE_DP_SIZE"],"8")


if __name__ == "__main__":
    unittest.main()
