import dataclasses
import json
import os
from pathlib import Path
import signal
import unittest
from unittest.mock import patch
import test_exporter as base

ROOT = Path(__file__).resolve().parents[2]


class HistogramCollector(unittest.TestCase):
    setUp = base.Integration.setUp
    tearDown = base.Integration.tearDown
    producer = base.Integration.producer
    engine = base.Integration.engine

    def start_native(self, name="new-native", counter=7):
        self.config = dataclasses.replace(self.config, prefill_dp_size=4)
        with patch.dict(os.environ,{"PD_EXPORTER_TEST_NATIVE_MODULE":str(ROOT / "native/sglang_pd_monitoring.py")}):
            return self.producer(name,counter)

    def hist_count(self, result):
        return sum(base.values(result["payload"],"llm_pd_request_output_tokens_count"))

    def test_real_four_process_histograms_count_logical_requests_once(self):
        producer = self.start_native(); result = self.engine(producer).refresh()
        self.assertEqual(self.hist_count(result),4)
        self.assertEqual(sum(base.values(result["payload"],"llm_pd_request_output_tokens_sum")),28)
        self.assertEqual(sum(base.values(result["payload"],"llm_pd_request_input_tokens_sum")),406)
        for rank in range(4):
            series = [s for s in base.samples(result["payload"]) if s.name == "llm_pd_request_output_tokens_count"
                      and s.labels["prefill_dp_rank"] == str(rank) and s.labels["status"] == "completed"]
            self.assertEqual(sum(s.value for s in series),1)
        self.assertLess(result["samples"], self.config.max_samples)
        self.assertLess(len(result["payload"].encode()), self.config.max_payload_bytes)

    def test_histogram_restart_changes_generation_and_excludes_old_counts(self):
        old = self.start_native(counter=77); engine = self.engine(old); first = engine.refresh()
        new = self.start_native("second-native",counter=3)
        self.docker.add("metrics-d0",new.p.pid,generation="two")
        second = engine.refresh()
        self.assertNotEqual(first["generation"],second["generation"])
        self.assertEqual(self.hist_count(second),4)
        self.assertEqual(sum(base.values(second["payload"],"llm_pd_request_output_tokens_sum")),12)

    def test_dead_histogram_writer_does_not_reappear_on_next_collection(self):
        producer = self.start_native(); engine = self.engine(producer)
        self.assertEqual(self.hist_count(engine.refresh()),4)
        os.kill(producer.info["children"][0],signal.SIGTERM)
        # Reap is performed by producer root at teardown; /proc identity sees zombie.
        import time
        time.sleep(0.15)
        result = engine.refresh()
        self.assertEqual(self.hist_count(result),3)
        store = base.exp.SnapshotStore(self.config); store.update(result)
        self.assertFalse(store.render()[1]["source_up"])

    def test_all_six_targets_native_identity_rank_domains_and_canonicalization(self):
        configs = base.exp.load_topology(ROOT / "config/topology-4p2d.json")
        self.assertEqual(set(configs),{f"910b-{n}" for n in (25,26,29,30,31,32)})
        for node,cfg in configs.items():
            self.config = dataclasses.replace(cfg,containers=("metrics-d0",),docker_socket=self.config.docker_socket,
                                             allowed_dir_prefix=str(self.path),timeout_seconds=2)
            source = dataclasses.replace(self.config,node=cfg.source_node)
            with patch.dict(os.environ,{"PD_EXPORTER_TEST_NATIVE_MODULE":str(ROOT / "native/sglang_pd_monitoring.py"),
                                       "PD_EXPORTER_TEST_CLUSTER":cfg.source_cluster}):
                p = self.producer(node,config=source)
            self.docker.add("metrics-d0",p.p.pid,generation=node)
            result = base.exp.Engine(self.config).refresh()
            self.assertEqual(set(result["heartbeats"]),set(cfg.expected_ranks))
            for sample in base.samples(result["payload"]):
                self.assertEqual(sample.labels["cluster"],"glm53-910b-4p2d")
                self.assertEqual(sample.labels["node"],node)
            if cfg.role == "decode":
                self.assertEqual(self.hist_count(result),4)
            else:
                self.assertEqual(sum(base.values(result["payload"],"llm_pd_prefill_stage_input_tokens_count")),1)
                self.assertEqual(base.values(result["payload"],"llm_pd_request_output_tokens_count"),[])
            p.close()


if __name__ == "__main__":
    unittest.main()
