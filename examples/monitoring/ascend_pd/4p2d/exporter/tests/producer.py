"""Real multiprocess writers for CPU integration tests; never import torch."""
import json
import os
import signal
import subprocess
import sys
import time


def writer(rank, role, node, service, counter):
    module_path = os.environ.get("PD_EXPORTER_TEST_NATIVE_MODULE")
    if module_path:
        return native_writer(module_path, rank, role, node, service, counter)
    from prometheus_client import Counter, Gauge

    keys = ["cluster", "model_name", "role", "service_id", "node", "dp_rank"]
    values = [os.environ.get("PD_EXPORTER_TEST_CLUSTER", "glm53-910b"), "GLM-5.3-Flash-w8a8", role, service, node, rank]
    running = Gauge("llm_pd_running_requests", "Real requests", keys, multiprocess_mode="livemostrecent")
    heartbeat = Gauge("llm_pd_last_snapshot_monotonic_seconds", "CPU heartbeat", keys, multiprocess_mode="livemostrecent")
    requests = Counter("llm_pd_requests", "Completed requests", keys + ["event"])
    available = Gauge("llm_pd_metric_available", "Observed groups", keys + ["metric_group"], multiprocess_mode="livemostrecent")
    requests.labels(*values, "completed").inc(counter)
    running.labels(*values).set(int(rank) + 1)
    available.labels(*values, "concurrency").set(1)
    # Native metrics from other TP ranks must never leak into exported payload.
    noise = Gauge("sglang_huge_debug", "Unrelated native family", ["key"], multiprocess_mode="sum")
    for i in range(50):
        noise.labels(str(i)).set(i)
    print("ready", flush=True)
    while True:
        heartbeat.labels(*values).set(time.monotonic())
        time.sleep(0.05)


def native_writer(module_path, rank, role, node, service, counter):
    import importlib.util
    from types import SimpleNamespace as NS
    spec = importlib.util.spec_from_file_location("pd_native_fixture", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    observer = module.PDObserver(labels={
        "cluster": os.environ.get("PD_EXPORTER_TEST_CLUSTER", "glm53-910b"), "model_name": "GLM-5.3-Flash-w8a8",
        "role": role, "service_id": service, "node": node, "dp_rank": rank},
        config={"tp_size": 16, "dp_size": 2 if role == "prefill" else 8, "ep_size": 16,
                "context_length": 1048576, "draft_tokens": 5, "num_steps": 4, "topk": 1,
                "kv_dtype": "int8", "monitor_version": "cpu-contract-check", "prefill_dp_size": 4, "decode_dp_size": 8})
    request = NS(rid="cpu-request", origin_input_ids=range(100 + int(rank)), output_ids=range(counter),
                 finished_reason=NS(), finished=lambda: True, disagg_prefill_dp_rank=int(rank) % 4,
                 cached_tokens=0, cached_tokens_device=0, cached_tokens_host=0, cached_tokens_storage=0)
    observer.finished(request)
    observer.finished(request)
    observer.child("llm_pd_running_requests").set(int(rank) + 1)
    observer.child("llm_pd_running_context_tokens").set(131072)
    observer.child("llm_pd_requests_total", event="completed").inc(counter)
    stages = (("waiting", "bootstrap", "prefill_inflight") if role == "prefill" else
              ("decode_ready", "decode_prealloc", "decode_transfer", "retracted"))
    for stage in stages:
        observer.child("llm_pd_queue_requests", stage=stage).set(0)
    if role == "prefill":
        observer.child("llm_pd_prefix_tokens_total", source="unattributed").inc(130000)
        observer.child("llm_pd_prefix_tokens_total", source="miss").inc(1072)
    else:
        observer.spec_verify(1, 3, 4, {"num_draft_tokens": 5})
    assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
    print("ready", flush=True)
    while True:
        observer.child("llm_pd_last_snapshot_monotonic_seconds").set(time.monotonic())
        time.sleep(0.05)


def root():
    data = json.loads(sys.argv[2])
    children = []
    try:
        for rank in data["ranks"]:
            p = subprocess.Popen([sys.executable, "-B", __file__, "writer", rank,
                                  data["role"], data["node"], data["service"], str(data["counter"])],
                                 stdout=subprocess.PIPE, text=True)
            assert p.stdout.readline().strip() == "ready"
            children.append(p)
        print(json.dumps({"root": os.getpid(), "children": [p.pid for p in children]}), flush=True)
        for line in sys.stdin:
            if line.strip() == "stop":
                break
    finally:
        for p in children:
            if p.poll() is None:
                p.terminate()
            p.wait(timeout=5)
            p.stdout.close()


if __name__ == "__main__":
    if sys.argv[1] == "writer":
        writer(*sys.argv[2:6], int(sys.argv[6]))
    else:
        root()
