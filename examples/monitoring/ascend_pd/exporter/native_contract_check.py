#!/usr/bin/env python3
"""CPU-only integration of the parent's actual native module with this exporter."""
import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "tests"))
from test_exporter import FakeDocker, Producer, values, samples
import exporter as exp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    module = args.module.resolve()
    os.environ["PD_EXPORTER_TEST_NATIVE_MODULE"] = str(module)
    results = []
    (ROOT / "validation").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="native-cpu-", dir=ROOT / "validation") as tmp:
        path = Path(tmp)
        docker = FakeDocker("\0glm53-native-contract-" + str(os.getpid()))
        try:
            for node, (role, service, ranks, suffix) in exp.TARGETS.items():
                name = "cpu-fixture-" + suffix
                cfg = exp.Config(node, role, service, ranks, (name,),
                                 docker_socket=docker.server.server_address,
                                 allowed_dir_prefix=str(path))
                producer = Producer(path / node, cfg)
                try:
                    docker.add(name, producer.p.pid, generation=node)
                    engine = exp.Engine(cfg)
                    first = engine.refresh()
                    durations = []
                    for _ in range(5):
                        result = engine.refresh()
                        durations.append(result["duration_seconds"])
                    store = exp.SnapshotStore(cfg)
                    store.update(result)
                    payload, status = store.render()
                    assert status["source_up"], status
                    assert len(values(result["payload"], "llm_pd_running_requests")) == len(ranks)
                    assert len(values(result["payload"], "llm_pd_build_info")) == len(ranks)
                    assert set(result["heartbeats"]) == set(ranks)
                    assert sum(values(result["payload"], "llm_pd_requests_total")) == 7 * len(ranks)
                    stages = {s.labels["stage"] for s in samples(result["payload"]) if s.name == "llm_pd_queue_requests"}
                    expected = {"waiting", "bootstrap", "prefill_inflight"} if role == "prefill" else {
                        "decode_ready", "decode_prealloc", "decode_transfer", "retracted"}
                    assert stages == expected, stages
                    groups = {s.labels["metric_group"] for s in samples(result["payload"])
                              if s.name == "llm_pd_metric_available"}
                    expected_groups = {"running", "queues", "kv", "mamba", "prefix"}
                    expected_groups.add("prefix_tiers" if role == "prefill" else "spec")
                    assert groups == expected_groups, groups
                    if role == "prefill":
                        unattributed = [s.value for s in samples(result["payload"]) if s.name == "llm_pd_prefix_tokens_total"
                                        and s.labels["source"] == "unattributed"]
                        assert unattributed == [130000]
                    results.append({"node": node, "ranks": ranks, "ok": True,
                                    "first_collection_seconds": first["duration_seconds"],
                                    "cached_collection_seconds": durations,
                                    "source_bytes": result["source_bytes"], "filtered_bytes": len(payload),
                                    "samples": result["samples"], "queue_stages": sorted(stages),
                                    "availability_groups": sorted(groups)})
                finally:
                    producer.close()
        finally:
            docker.close()
    receipt = {"native_module": str(module), "native_module_sha256": hashlib.sha256(module.read_bytes()).hexdigest(),
               "python": sys.version, "prometheus_client": importlib.metadata.version("prometheus_client"),
               "cpu_affinity": sorted(os.sched_getaffinity(0)),
               "torch_imported": "torch" in sys.modules, "npu_calls": 0, "nodes": results}
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
