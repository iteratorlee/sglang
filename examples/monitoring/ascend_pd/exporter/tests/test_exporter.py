import contextlib
import dataclasses
import http.client
import importlib.util
import json
import os
from pathlib import Path
import signal
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler

from prometheus_client.parser import text_string_to_metric_families

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import exporter as exp


def samples(text):
    return [sample for family in text_string_to_metric_families(text) for sample in family.samples]


def values(text, name):
    return [s.value for s in samples(text) if s.name == name]


class FakeDocker:
    def __init__(self, socket_path):
        self.objects, self.calls, self.delay = {}, [], 0
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                owner.calls.append((self.command, self.path))
                if owner.delay:
                    time.sleep(owner.delay)
                name = self.path.removeprefix("/containers/").removesuffix("/json")
                obj = owner.objects.get(name)
                payload = json.dumps(obj or {"message": "not found"}).encode()
                try:
                    self.send_response(200 if obj else 404)
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                except BrokenPipeError:
                    pass

        class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
            daemon_threads = True

        self.server = Server(str(socket_path), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def add(self, name, pid, generation="one", running=True):
        self.objects[name] = {"Id": generation * 8, "Name": "/" + name,
                              "State": {"Running": running, "Pid": pid, "StartedAt": generation}}

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


class Producer:
    def __init__(self, directory, config, counter=7):
        directory.mkdir()
        self.directory = directory
        env = dict(os.environ, PROMETHEUS_MULTIPROC_DIR=str(directory))
        data = {"ranks": config.expected_ranks, "role": config.role, "node": config.node,
                "service": config.service_id, "counter": counter}
        self.p = subprocess.Popen([sys.executable, "-B", str(HERE / "producer.py"), "root", json.dumps(data)],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env)
        self.info = json.loads(self.p.stdout.readline())
        time.sleep(0.08)

    def close(self):
        if self.p.poll() is None:
            self.p.stdin.write("stop\n")
            self.p.stdin.flush()
            self.p.wait(timeout=5)
        self.p.stdin.close()
        self.p.stdout.close()


class Integration(unittest.TestCase):
    def setUp(self):
        # All test files stay within the delegated exporter directory.
        self.temp = tempfile.TemporaryDirectory(prefix="cpu-fixture-", dir=HERE)
        self.path = Path(self.temp.name)
        socket_path = "\0glm53-exporter-cpu-" + str(os.getpid()) + "-" + self.path.name
        self.docker = FakeDocker(socket_path)
        self.config = exp.Config("910b-17", "decode", "decode-0", ("0", "1", "2", "3"),
                                 ("original-d0", "metrics-d0"), docker_socket=socket_path,
                                 allowed_dir_prefix=str(self.path), timeout_seconds=2.0)
        self.producers = []

    def tearDown(self):
        for p in self.producers:
            p.close()
        self.docker.close()
        self.temp.cleanup()

    def producer(self, name="generation-one", counter=7, config=None):
        p = Producer(self.path / name, config or self.config, counter)
        self.producers.append(p)
        return p

    def engine(self, producer=None):
        p = producer or self.producer()
        self.docker.add("metrics-d0", p.p.pid)
        return exp.Engine(self.config)

    def test_official_aggregation_real_four_dp_writers_and_http(self):
        engine = self.engine()
        result = engine.refresh()
        self.assertEqual(sum(values(result["payload"], "llm_pd_running_requests")), 10)
        self.assertEqual(values(result["payload"], "llm_pd_requests_total"), [7] * 4)
        self.assertEqual(len(result["heartbeats"]), 4)
        self.assertNotIn("sglang_", result["payload"])
        self.assertNotIn('pid="', result["payload"])
        store = exp.SnapshotStore(self.config)
        store.update(result)
        server = exp.make_server(("127.0.0.1", 0), store)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            for path, expected in [("/metrics", 200), ("/readyz", 200), ("/healthz", 200)]:
                conn = http.client.HTTPConnection("127.0.0.1", server.server_port)
                conn.request("GET", path)
                response = conn.getresponse()
                text = response.read().decode()
                self.assertEqual(response.status, expected)
                if path == "/metrics":
                    self.assertEqual(values(text, "llm_pd_expected_local_dp_ranks"), [4])
                    self.assertEqual(values(text, "llm_pd_exporter_source_up"), [1])
                    self.assertEqual(len(values(text, "llm_pd_snapshot_age_seconds")), 4)
                conn.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        self.assertTrue(all(method == "GET" and path.startswith("/containers/") for method, path in self.docker.calls))

    def test_generation_directory_rotation_does_not_merge_old_counters(self):
        first = self.producer(counter=9)
        engine = self.engine(first)
        old = engine.refresh()
        second = self.producer("generation-two", counter=2)
        self.docker.add("metrics-d0", second.p.pid, generation="two")
        new = engine.refresh()
        self.assertNotEqual(old["generation"], new["generation"])
        self.assertEqual(sum(values(old["payload"], "llm_pd_requests_total")), 36)
        self.assertEqual(sum(values(new["payload"], "llm_pd_requests_total")), 8)
        self.assertTrue(list(first.directory.glob("*.db")))

    def test_dead_writer_files_are_not_reintroduced(self):
        p = self.producer()
        engine = self.engine(p)
        engine.refresh()
        dead = p.info["children"][0]
        os.kill(dead, signal.SIGKILL)
        time.sleep(0.05)
        result = engine.refresh()
        self.assertGreater(result["ignored_dead_writer_files"], 0)
        self.assertEqual(set(result["heartbeats"]), {"1", "2", "3"})
        self.assertEqual(sum(values(result["payload"], "llm_pd_requests_total")), 21)
        store = exp.SnapshotStore(self.config)
        store.update(result)
        payload, status = store.render()
        self.assertFalse(status["source_up"])
        self.assertEqual(values(payload.decode(), "llm_pd_exporter_observed_local_dp_ranks"), [3])
        self.assertNotIn('dp_rank="0"', payload.decode())

    def test_missing_source_exports_health_without_fabricated_load(self):
        result = exp.safe_refresh(exp.Engine(self.config))
        self.assertEqual(result["error"], "no_container")
        store = exp.SnapshotStore(self.config)
        store.update(result)
        payload, status = store.render()
        self.assertFalse(status["source_up"])
        self.assertEqual(values(payload.decode(), "llm_pd_expected_local_dp_ranks"), [4])
        self.assertEqual(values(payload.decode(), "llm_pd_running_requests"), [])
        self.assertEqual(values(payload.decode(), "llm_pd_snapshot_age_seconds"), [])

    def test_two_running_candidates_fail_closed(self):
        p = self.producer()
        engine = self.engine(p)
        self.docker.add("original-d0", p.p.pid, generation="conflict")
        self.assertEqual(exp.safe_refresh(engine)["error"], "multiple_containers")

    def test_stale_cached_heartbeat_is_not_fresh_http_success(self):
        result = self.engine().refresh()
        for h in result["heartbeats"].values():
            h["monotonic"] = time.monotonic() - 120
        store = exp.SnapshotStore(self.config)
        store.update(result)
        payload, status = store.render()
        self.assertFalse(status["source_up"])
        self.assertEqual(values(payload.decode(), "llm_pd_exporter_source_up"), [0])
        self.assertTrue(all(x >= 120 for x in values(payload.decode(), "llm_pd_snapshot_age_seconds")))
        self.assertEqual(sum(values(payload.decode(), "llm_pd_running_requests")), 10)

    def test_heartbeat_advancing_during_collection_is_not_future_clock_error(self):
        engine = self.engine()
        official = exp.SelectedMultiProcessCollector.collect

        def slow(collector):
            time.sleep(1.15)  # Real writers keep advancing their heartbeat.
            return official(collector)

        with patch.object(exp.SelectedMultiProcessCollector, "collect", slow):
            result = engine.refresh()
        self.assertTrue(result["ok"])

    def test_failed_refresh_drops_previously_cached_native_payload(self):
        store = exp.SnapshotStore(self.config)
        store.update(self.engine().refresh())
        store.update({"ok": False, "error": "identity", "message": "owner changed"})
        payload, _ = store.render()
        self.assertEqual(values(payload.decode(), "llm_pd_running_requests"), [])

    def test_directory_escape_and_symlink_rejected(self):
        p = self.producer()
        proc = exp.ProcReader(self.config)
        owner = proc.identity(p.p.pid)
        for value in ["/etc", str(self.path) + "/../escape", "relative", str(self.path)]:
            with self.assertRaises(exp.SourceError):
                exp.safe_directory(proc, owner, value, str(self.path))
        link = self.path / "symlink-dir"
        link.symlink_to(p.directory, target_is_directory=True)
        with self.assertRaises(OSError):
            exp.safe_directory(proc, owner, str(link), str(self.path))

    def test_database_symlink_rejected(self):
        p = self.producer()
        (p.directory / "counter_999999.db").symlink_to(next(p.directory.glob("counter*.db")))
        self.assertEqual(exp.safe_refresh(self.engine(p))["error"], "path")

    def test_corrupt_real_file_fails_closed(self):
        p = self.producer()
        counter = next(p.directory.glob("counter*.db"))
        counter.write_bytes(b"not a prometheus database")
        result = exp.safe_refresh(self.engine(p))
        self.assertFalse(result["ok"])
        self.assertNotIn("payload", result)

    def test_pid_reuse_detected_by_start_ticks(self):
        p = self.producer()
        proc = exp.ProcReader(self.config)
        identity = proc.identity(p.p.pid)
        self.assertTrue(proc.same(identity))
        self.assertFalse(proc.same(dict(identity, start_ticks=str(int(identity["start_ticks"]) + 1))))

    def test_file_disappears_during_official_collection(self):
        p = self.producer()
        official = exp.SelectedMultiProcessCollector.collect

        def disappear(collector):
            next(p.directory.glob("counter*.db")).unlink()
            return official(collector)

        with patch.object(exp.SelectedMultiProcessCollector, "collect", disappear):
            result = exp.safe_refresh(self.engine(p))
        self.assertFalse(result["ok"])
        self.assertNotIn("payload", result)

    def test_no_prometheus_environment_is_not_zero_load(self):
        # This CPU root has no model multiprocess env. Discovery must fail,
        # rather than inventing 4 idle DP workers.
        self.docker.add("metrics-d0", os.getpid())
        result = exp.safe_refresh(exp.Engine(self.config))
        self.assertEqual(result["error"], "discovery")

    def test_same_container_directory_inode_replacement_fails_closed(self):
        p = self.producer()
        engine = self.engine(p)
        engine.refresh()
        p.directory.rename(self.path / "old-inode")
        p.directory.mkdir()
        result = exp.safe_refresh(engine)
        self.assertEqual(result["error"], "identity")

    def test_partial_dp_native_payload_not_zero_filled(self):
        config = dataclasses.replace(self.config, expected_ranks=("0", "1"))
        p = self.producer(config=config)
        result = self.engine(p).refresh()
        store = exp.SnapshotStore(self.config)
        store.update(result)
        payload, status = store.render()
        self.assertEqual(set(result["heartbeats"]), {"0", "1"})
        self.assertEqual(len(values(payload.decode(), "llm_pd_running_requests")), 2)
        self.assertFalse(status["source_up"])

    def test_wrong_node_labels_rejected(self):
        config = dataclasses.replace(self.config, node="910b-27")
        p = self.producer(config=config)
        self.assertEqual(exp.safe_refresh(self.engine(p))["error"], "labels")

    def test_source_byte_bound(self):
        p = self.producer()
        engine = self.engine(p)
        engine.config.max_source_bytes = 100
        self.assertEqual(exp.safe_refresh(engine)["error"], "limit")

    def test_real_bounded_subprocess_collection_and_timeout(self):
        self.engine()  # Register live producer in fake Unix Docker API.
        worker = exp.BoundedWorker(self.config)
        try:
            result = worker.refresh()
            self.assertTrue(result["ok"], result)
            worker_pid = worker.process.pid
            self.assertTrue(worker.refresh()["ok"])
            self.assertEqual(worker.process.pid, worker_pid)
            self.docker.delay = 1
            worker.config.timeout_seconds = 0.1
            start = time.monotonic()
            result = worker.refresh()
            self.assertEqual(result["error"], "timeout")
            self.assertLess(time.monotonic() - start, 1)
            self.assertIsNone(worker.process)
        finally:
            worker.close()

    def test_node_global_dp_mapping(self):
        self.assertEqual(exp.TARGETS["910b-16"][2], ("0",))
        self.assertEqual(exp.TARGETS["910b-28"][2], ("1",))
        self.assertEqual(exp.TARGETS["910b-17"][2], ("0", "1", "2", "3"))
        self.assertEqual(exp.TARGETS["910b-27"][2], ("4", "5", "6", "7"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
