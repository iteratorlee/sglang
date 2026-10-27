#!/usr/bin/env python3
"""Read-only, CPU-only exporter for one node of a distributed P or D instance.

All mmap interpretation/aggregation belongs to prometheus_client. This program
only discovers a container generation, bounds inputs, selects live writers,
filters metric families and serves a cached snapshot.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import resource
import selectors
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer

from prometheus_client import CollectorRegistry, generate_latest
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily, Metric
from prometheus_client.multiprocess import MultiProcessCollector


PREFIX = "llm_pd_"
HEARTBEAT = "llm_pd_last_snapshot_monotonic_seconds"
BASE_LABELS = ("cluster", "role", "service_id", "node")
ERROR_CODES = ("no_container", "multiple_containers", "docker", "discovery",
               "identity", "path", "limit", "collect", "labels", "timeout", "worker")
TARGETS = {
    "910b-16": ("prefill", "prefill-0", ("0",), "p0"),
    "910b-28": ("prefill", "prefill-0", ("1",), "p1"),
    "910b-17": ("decode", "decode-0", ("0", "1", "2", "3"), "d0"),
    "910b-27": ("decode", "decode-0", ("4", "5", "6", "7"), "d1"),
}


class SourceError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


@dataclass
class Config:
    node: str
    role: str
    service_id: str
    expected_ranks: tuple[str, ...]
    containers: tuple[str, ...]
    cluster: str = "glm53-910b"
    docker_socket: str = "/var/run/docker.sock"
    proc_root: str = "/proc"
    allowed_dir_prefix: str = "/tmp"
    refresh_seconds: float = 5.0
    timeout_seconds: float = 3.0
    stale_seconds: float = 30.0
    rediscover_seconds: float = 60.0
    max_processes: int = 1024
    max_tasks: int = 8192
    max_files: int = 2048
    max_source_bytes: int = 32 * 1024 * 1024
    max_file_bytes: int = 8 * 1024 * 1024
    max_samples: int = 32768
    max_payload_bytes: int = 2 * 1024 * 1024
    worker_memory_mb: int = 512


def bounded_read(path, limit):
    with open(path, "rb") as f:
        data = f.read(limit + 1)
    if len(data) > limit:
        raise SourceError("limit", "input exceeds configured read limit")
    return data


class UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, path, timeout):
        super().__init__("localhost", timeout=timeout)
        self.socket_path = path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.socket_path)


class DockerReader:
    """The sole Docker operation is GET /containers/<allowlisted-name>/json."""
    def __init__(self, config):
        self.config = config

    def inspect(self, name):
        if name not in self.config.containers or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", name):
            raise SourceError("docker", "container name is not explicitly allowed")
        conn = UnixHTTPConnection(self.config.docker_socket, 1.0)
        try:
            conn.request("GET", "/containers/" + name + "/json")
            response = conn.getresponse()
            data = response.read(1024 * 1024 + 1)
            if len(data) > 1024 * 1024:
                raise SourceError("limit", "Docker response too large")
            if response.status == 404:
                return None
            if response.status != 200:
                raise SourceError("docker", "Docker inspect HTTP " + str(response.status))
            obj = json.loads(data)
            # Never retain or print the full inspect object, environment or secrets.
            return {"Id": obj["Id"], "Name": obj["Name"], "State": {
                key: obj["State"].get(key) for key in ("Running", "Pid", "StartedAt")}}
        except (OSError, ValueError, KeyError, http.client.HTTPException) as exc:
            raise SourceError("docker", "Docker inspect failed: " + type(exc).__name__) from exc
        finally:
            conn.close()

    def active(self):
        running = []
        for name in self.config.containers:
            obj = self.inspect(name)
            if obj is not None and obj["State"]["Running"]:
                if obj["Name"].lstrip("/") != name or int(obj["State"]["Pid"] or 0) <= 0:
                    raise SourceError("identity", "Docker name/PID mismatch")
                running.append(obj)
        if not running:
            raise SourceError("no_container", "no allowed container is running")
        if len(running) != 1:
            raise SourceError("multiple_containers", "more than one allowed container is running")
        return running[0]


class ProcReader:
    def __init__(self, config):
        self.config = config
        self.root = Path(config.proc_root)

    def identity(self, pid):
        raw = bounded_read(self.root / str(pid) / "stat", 8192).decode()
        fields = raw[raw.rfind(")") + 2:].split()
        if fields[0] in ("Z", "X"):
            raise SourceError("identity", "source process exited")
        return {"pid": int(pid), "start_ticks": fields[19]}

    def same(self, identity):
        try:
            return self.identity(identity["pid"]) == identity
        except (FileNotFoundError, ProcessLookupError, SourceError):
            return False

    def boot_id(self):
        return bounded_read(self.root / "sys/kernel/random/boot_id", 128).decode().strip()

    def namespace_pid(self, pid):
        raw = bounded_read(self.root / str(pid) / "status", 65536).decode()
        for line in raw.splitlines():
            if line.startswith("NSpid:"):
                return line.split()[-1]
        # Linux namespace PID proof is required; silently using host PID is unsafe.
        raise SourceError("discovery", "NSpid is absent in process status")

    def env_dir(self, pid):
        raw = bounded_read(self.root / str(pid) / "environ", 262144)
        values = set()
        for item in raw.split(b"\0"):
            key, sep, value = item.partition(b"=")
            if sep and key in (b"PROMETHEUS_MULTIPROC_DIR", b"prometheus_multiproc_dir"):
                values.add(value.decode("utf-8", "strict"))
        if len(values) > 1:
            raise SourceError("discovery", "conflicting Prometheus environment values")
        return next(iter(values), None)

    def descendants(self, root_pid):
        queue, seen, processes, tasks = [root_pid], set(), [], 0
        while queue:
            pid = queue.pop()
            if pid in seen:
                continue
            seen.add(pid)
            if len(seen) > self.config.max_processes:
                raise SourceError("limit", "too many model descendants")
            try:
                identity = self.identity(pid)
                ns_pid = self.namespace_pid(pid)
                directory = self.env_dir(pid)
                # Children can be forked by any thread, not just the main thread.
                with os.scandir(self.root / str(pid) / "task") as entries:
                    for entry in entries:
                        if not entry.name.isdecimal():
                            continue
                        tasks += 1
                        if tasks > self.config.max_tasks:
                            raise SourceError("limit", "too many model tasks")
                        try:
                            data = bounded_read(Path(entry.path) / "children", 65536)
                        except (FileNotFoundError, ProcessLookupError):
                            continue
                        queue.extend(int(x) for x in data.split())
                if self.same(identity):
                    processes.append((identity, ns_pid, directory))
            except (FileNotFoundError, ProcessLookupError):
                continue
        return processes


def safe_directory(proc, owner, value, allowed_prefix):
    """Resolve components with openat/O_NOFOLLOW under the proven process root."""
    path, prefix = PurePosixPath(value), PurePosixPath(allowed_prefix)
    if (not path.is_absolute() or ".." in path.parts or "//" in value
            or "." in value.split("/") or path == prefix or prefix not in path.parents):
        raise SourceError("path", "multiprocess directory is outside the allowed subtree")
    root = proc.root / str(owner["pid"]) / "root"
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    fd = os.open(root, flags)
    try:
        for component in path.parts[1:]:
            next_fd = os.open(component, flags | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        st = os.fstat(fd)
        if not proc.same(owner):
            raise SourceError("identity", "owner changed during directory resolution")
        return fd, (st.st_dev, st.st_ino)
    except BaseException:
        os.close(fd)
        raise


class SelectedMultiProcessCollector(MultiProcessCollector):
    """Official merge/aggregation over bounded, active-writer files only."""
    def __init__(self, path, files):
        super().__init__(None, path=path)
        self.files = files

    def collect(self):
        return self.merge(self.files, accumulate=True)


class Families:
    def __init__(self, families):
        self.families = families

    def collect(self):
        yield from self.families


class Engine:
    """Runs in a bounded CPU subprocess; no filesystem work in HTTP handlers."""
    def __init__(self, config, docker=None, proc=None):
        self.config = config
        self.docker = docker or DockerReader(config)
        self.proc = proc or ProcReader(config)
        self.cache = None
        self.file_owners = {}

    def container_identity(self, obj):
        return {"container_id": obj["Id"], "started_at": obj["State"]["StartedAt"],
                "boot_id": self.proc.boot_id(),
                "root": self.proc.identity(int(obj["State"]["Pid"]))}

    def discover(self, cid, now):
        processes = self.proc.descendants(cid["root"]["pid"])
        dirs = {directory for _, _, directory in processes if directory}
        if len(dirs) != 1:
            raise SourceError("discovery", "expected exactly one live multiprocess directory")
        directory = dirs.pop()
        owners = [identity for identity, _, d in processes if d == directory]
        owner = min(owners, key=lambda x: (int(x["start_ticks"]), x["pid"]))
        fd, inode = safe_directory(self.proc, owner, directory, self.config.allowed_dir_prefix)
        os.close(fd)
        writers = {}
        for identity, ns_pid, d in processes:
            if d == directory:
                if ns_pid in writers and writers[ns_pid] != identity:
                    raise SourceError("identity", "ambiguous namespace PID")
                writers[ns_pid] = identity
        cache = {"container": cid, "owner": owner, "directory": directory,
                 "inode": inode, "writers": writers, "discovered_at": now}
        uid = {k: cache[k] for k in ("container", "directory", "inode")}
        cache["generation"] = hashlib.sha256(json.dumps(uid, sort_keys=True).encode()).hexdigest()
        if self.cache and self.cache["generation"] != cache["generation"]:
            self.file_owners.clear()
        self.cache = cache

    def collect_files(self, fd):
        config, cache = self.config, self.cache
        files, entries, total_bytes, ignored, selected_ids = [], 0, 0, 0, {}
        present_names = set()
        root = "/proc/self/fd/" + str(fd)
        with os.scandir(root) as it:
            for entry in it:
                entries += 1
                if entries > config.max_files:
                    raise SourceError("limit", "too many entries in source directory")
                if not entry.name.endswith(".db"):
                    continue
                present_names.add(entry.name)
                if not re.fullmatch(r"(?:counter|summary|histogram|gauge_[a-z]+)_[0-9]+\.db", entry.name):
                    raise SourceError("collect", "unexpected multiprocess file name")
                st = entry.stat(follow_symlinks=False)
                if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
                    raise SourceError("path", "source db must be a regular, non-linked file")
                total_bytes += st.st_size
                if st.st_size > config.max_file_bytes or total_bytes > config.max_source_bytes:
                    raise SourceError("limit", "source file byte limit exceeded")
                ns_pid = entry.name.rsplit("_", 1)[1][:-3]
                writer = cache["writers"].get(ns_pid)
                if writer is None or not self.proc.same(writer):
                    # Never delete/mark_process_dead model files. Exclude dead
                    # writers from BOTH gauges and counters, not just live gauges.
                    ignored += 1
                    continue
                binding = (st.st_dev, st.st_ino, writer["pid"], writer["start_ticks"])
                previous = self.file_owners.get(entry.name)
                if previous and previous[:2] == binding[:2] and previous[2:] != binding[2:]:
                    raise SourceError("identity", "db inode reused by a different writer lifetime")
                self.file_owners[entry.name] = binding
                selected_ids[entry.name] = (st.st_dev, st.st_ino, writer)
                files.append(str(Path(root) / entry.name))
        self.file_owners = {name: binding for name, binding in self.file_owners.items() if name in present_names}
        if not files:
            raise SourceError("collect", "no live-writer Prometheus files")
        return root, files, selected_ids, total_bytes, ignored

    def refresh(self):
        now = time.monotonic()
        cid = self.container_identity(self.docker.active())
        cache = self.cache
        if (cache is None or cache["container"] != cid or not self.proc.same(cache["owner"])
                or now - cache["discovered_at"] >= self.config.rediscover_seconds):
            self.discover(cid, now)
        cache = self.cache
        fd, inode = safe_directory(self.proc, cache["owner"], cache["directory"],
                                   self.config.allowed_dir_prefix)
        try:
            if inode != cache["inode"]:
                self.cache = None
                raise SourceError("identity", "multiprocess directory replaced")
            root, files, identities, source_bytes, ignored = self.collect_files(fd)
            families, heartbeats, count, models = [], {}, 0, set()
            for family in SelectedMultiProcessCollector(root, files).collect():
                if not family.name.startswith(PREFIX):
                    continue
                if (family.name.startswith("llm_pd_exporter_")
                        or family.name in ("llm_pd_snapshot_age_seconds", "llm_pd_expected_local_dp_ranks")):
                    raise SourceError("labels", "native producer used exporter-reserved metric name")
                kept = Metric(family.name, family.documentation, family.type)
                for sample in family.samples:
                    count += 1
                    if count > self.config.max_samples:
                        raise SourceError("limit", "too many llm_pd samples")
                    labels = sample.labels
                    for key in BASE_LABELS:
                        if labels.get(key) != getattr(self.config, key):
                            raise SourceError("labels", "source label does not match this node: " + key)
                    rank = labels.get("dp_rank")
                    if rank not in self.config.expected_ranks or not labels.get("model_name"):
                        raise SourceError("labels", "unexpected DP rank or absent model_name")
                    if "pid" in labels:
                        raise SourceError("labels", "all/liveall gauge mode exposes duplicate writer PID")
                    models.add(labels["model_name"])
                    if not math.isfinite(sample.value):
                        raise SourceError("collect", "non-finite native metric value")
                    if sample.name == HEARTBEAT:
                        if rank in heartbeats or sample.value <= 0 or sample.value > time.monotonic() + 1.0:
                            raise SourceError("identity", "duplicate/invalid monotonic heartbeat")
                        heartbeats[rank] = {"labels": labels, "monotonic": sample.value}
                    kept.add_sample(sample.name, labels, sample.value)
                families.append(kept)
            if len(models) > 1:
                raise SourceError("labels", "multiple model identities in one node source")
            if not heartbeats:
                raise SourceError("collect", "no native llm_pd heartbeat; source is not instrumented")
            # Require all payload ranks to have a heartbeat; missing ranks remain
            # absent, while expected/fresh coverage exposes incomplete startup.
            for family in families:
                family.samples = [s for s in family.samples if s.labels["dp_rank"] in heartbeats]
            registry = CollectorRegistry(auto_describe=False)
            registry.register(Families(families))
            payload = generate_latest(registry)
            if len(payload) > self.config.max_payload_bytes:
                raise SourceError("limit", "filtered exposition exceeds byte limit")
            # An exit, replacement or inode change during collection invalidates
            # the entire sample. No partial old-generation merge is published.
            if self.container_identity(self.docker.active()) != cid or not self.proc.same(cache["owner"]):
                raise SourceError("identity", "container generation changed during collection")
            check_fd, check_inode = safe_directory(self.proc, cache["owner"], cache["directory"],
                                                   self.config.allowed_dir_prefix)
            os.close(check_fd)
            if check_inode != inode:
                self.cache = None
                raise SourceError("identity", "source directory rotated during collection")
            for name, (dev, ino, writer) in identities.items():
                st = os.stat(name, dir_fd=fd, follow_symlinks=False)
                if (st.st_dev, st.st_ino) != (dev, ino) or not self.proc.same(writer):
                    raise SourceError("identity", "writer/file changed during collection")
            return {"ok": True, "payload": payload.decode(), "heartbeats": heartbeats,
                    "generation": cache["generation"], "container_id": cid["container_id"],
                    "owner": cache["owner"], "directory": cache["directory"],
                    "source_bytes": source_bytes, "selected_files": len(files),
                    "ignored_dead_writer_files": ignored, "samples": count,
                    "refreshed_monotonic": time.monotonic(),
                    "duration_seconds": time.monotonic() - now}
        finally:
            os.close(fd)


def safe_refresh(engine):
    try:
        return engine.refresh()
    except SourceError as exc:
        return {"ok": False, "error": exc.code, "message": str(exc)}
    except PermissionError:
        return {"ok": False, "error": "discovery", "message": "read permission denied; check host PID/PTRACE"}
    except (OSError, ValueError, KeyError, IndexError, RuntimeError, MemoryError) as exc:
        return {"ok": False, "error": "collect", "message": "source read failed: " + type(exc).__name__}


def worker_main():
    config = Config(**json.loads(sys.stdin.buffer.readline(65536)))
    mem = config.worker_memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
    resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    engine = Engine(config)
    for line in sys.stdin.buffer:
        if line != b"refresh\n":
            break
        print(json.dumps(safe_refresh(engine), separators=(",", ":")), flush=True)


class BoundedWorker:
    def __init__(self, config):
        self.config, self.process = config, None

    def close(self):
        if self.process is not None:
            self.process.kill()
            self.process.wait(timeout=2)
            self.process.stdin.close()
            self.process.stdout.close()
            self.process = None

    def refresh(self):
        start = time.monotonic()
        try:
            if self.process is None:
                env = {k: v for k, v in os.environ.items()
                       if k.lower() != "prometheus_multiproc_dir"}
                self.process = subprocess.Popen(
                    [sys.executable, "-B", str(Path(__file__).resolve()), "--worker"],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    env=env, bufsize=0, close_fds=True)
                self.process.stdin.write(json.dumps(asdict(self.config)).encode() + b"\n")
            self.process.stdin.write(b"refresh\n")
            output = bytearray()
            with selectors.DefaultSelector() as selector:
                selector.register(self.process.stdout, selectors.EVENT_READ)
                while True:
                    remaining = self.config.timeout_seconds - (time.monotonic() - start)
                    if remaining <= 0 or not selector.select(remaining):
                        raise SourceError("timeout", "collection exceeded wall-clock deadline")
                    block = os.read(self.process.stdout.fileno(), 65536)
                    if not block:
                        raise SourceError("worker", "collector subprocess exited")
                    output.extend(block)
                    # JSON escapes can expand source text; still absolutely bounded.
                    if len(output) > self.config.max_payload_bytes * 3 + 65536:
                        raise SourceError("limit", "collector response exceeded byte bound")
                    if output.endswith(b"\n"):
                        return json.loads(output)
        except (OSError, ValueError, SourceError) as exc:
            self.close()
            return {"ok": False, "error": getattr(exc, "code", "worker"),
                    "message": str(exc) if isinstance(exc, SourceError) else type(exc).__name__}


class SnapshotStore:
    def __init__(self, config):
        self.config = config
        self.lock = threading.Lock()
        self.snapshot = {"ok": False, "error": "no_container", "message": "not refreshed yet"}
        self.errors = dict.fromkeys(ERROR_CODES, 0)
        self.last_attempt = None

    def update(self, snapshot):
        with self.lock:
            self.snapshot = snapshot  # Fail closed: never reuse a failed generation's payload.
            self.last_attempt = time.monotonic()
            if not snapshot.get("ok"):
                self.errors[snapshot.get("error") if snapshot.get("error") in self.errors else "worker"] += 1

    def render(self):
        with self.lock:
            snapshot, errors, last_attempt = self.snapshot, dict(self.errors), self.last_attempt
        config, now = self.config, time.monotonic()
        base = [getattr(config, key) for key in BASE_LABELS]
        families = []

        def gauge(name, help_text, value):
            family = GaugeMetricFamily(name, help_text, labels=BASE_LABELS)
            family.add_metric(base, value)
            families.append(family)

        heartbeats = snapshot.get("heartbeats", {}) if snapshot.get("ok") else {}
        ages = {rank: max(0.0, now - h["monotonic"]) for rank, h in heartbeats.items()}
        fresh = sum(age <= config.stale_seconds for age in ages.values())
        poll_fresh = last_attempt is not None and now - last_attempt <= config.stale_seconds
        up = bool(snapshot.get("ok") and poll_fresh and fresh == len(config.expected_ranks))
        gauge("llm_pd_expected_local_dp_ranks", "Configured local attention DP ranks, not observed load.", len(config.expected_ranks))
        gauge("llm_pd_exporter_source_up", "1 only when all configured DP heartbeats and collection are fresh.", int(up))
        gauge("llm_pd_exporter_observed_local_dp_ranks", "DP ranks with observed native heartbeat.", len(ages))
        gauge("llm_pd_exporter_fresh_local_dp_ranks", "DP ranks whose native heartbeat is within freshness threshold.", fresh)
        if last_attempt is not None:
            gauge("llm_pd_exporter_refresh_age_seconds", "Monotonic age of most recent collection attempt.", now - last_attempt)
        err = CounterMetricFamily("llm_pd_exporter_errors", "Failed refresh attempts by bounded reason.",
                                  labels=(*BASE_LABELS, "reason"))
        for reason, value in errors.items():
            err.add_metric([*base, reason], value)
        families.append(err)
        if snapshot.get("ok"):
            age_family = GaugeMetricFamily("llm_pd_snapshot_age_seconds", "Age of native snapshot, same-host monotonic clock.",
                                           labels=(*BASE_LABELS, "model_name", "dp_rank"))
            for rank, age in ages.items():
                age_family.add_metric([*base, heartbeats[rank]["labels"]["model_name"], rank], age)
            families.append(age_family)
            for key in ("duration_seconds", "source_bytes", "selected_files", "ignored_dead_writer_files"):
                gauge("llm_pd_exporter_" + key, "Last successful collection " + key + ".", snapshot[key])
            gen = GaugeMetricFamily("llm_pd_exporter_generation_info", "Container/root/directory generation digest; no native counter relabeling.",
                                     labels=(*BASE_LABELS, "generation"))
            gen.add_metric([*base, snapshot["generation"]], 1)
            families.append(gen)
        registry = CollectorRegistry(auto_describe=False)
        registry.register(Families(families))
        payload = snapshot.get("payload", "").encode() if snapshot.get("ok") else b""
        status = {k: v for k, v in snapshot.items() if k != "payload"}
        status.update(source_up=up, snapshot_ages=ages, expected_ranks=config.expected_ranks)
        return payload + generate_latest(registry), status


def make_server(address, store):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(2.0)

        def log_message(self, *_):
            pass

        def do_GET(self):
            metrics, status = store.render()  # In-memory only; never Docker/proc/NPU work.
            if self.path == "/metrics":
                code, content_type, body = 200, "text/plain; version=0.0.4; charset=utf-8", metrics
            elif self.path == "/status":
                code, content_type, body = 200, "application/json", json.dumps(status).encode()
            elif self.path == "/healthz":
                code, content_type, body = 200, "text/plain", b"exporter alive\n"
            elif self.path == "/readyz":
                code, content_type = (200 if status["source_up"] else 503), "text/plain"
                body = b"source fresh\n" if code == 200 else b"source unavailable or stale\n"
            else:
                code, content_type, body = 404, "text/plain", b"not found\n"
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, TimeoutError, ConnectionResetError):
                pass

    # A bounded sequential HTTP server is sufficient for one Prometheus scrape
    # every five seconds; a stalled client can hold it for at most two seconds.
    return HTTPServer(address, Handler)


def main():
    if sys.argv[1:] == ["--worker"]:
        worker_main()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", required=True, choices=TARGETS)
    parser.add_argument("--container", action="append", help="Explicit allowed candidate, repeatable; defaults to old and metrics deployment names")
    parser.add_argument("--docker-socket", default="/var/run/docker.sock")
    parser.add_argument("--proc-root", default="/proc")
    parser.add_argument("--listen", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=19100)
    parser.add_argument("--refresh-seconds", type=float, default=5.0)
    parser.add_argument("--timeout-seconds", type=float, default=3.0)
    parser.add_argument("--stale-seconds", type=float, default=30.0)
    parser.add_argument("--rediscover-seconds", type=float, default=60.0)
    parser.add_argument("--allowed-dir-prefix", default="/tmp")
    parser.add_argument("--once", action="store_true", help="One bounded read; stdout exposition, stderr sanitized status")
    args = parser.parse_args()
    role, service, ranks, suffix = TARGETS[args.node]
    containers = tuple(args.container or ["glm53-context1m-1026-" + suffix,
                                          "glm53-context1m-metrics-1027-" + suffix])
    if (not 1 <= len(containers) <= 4 or len(set(containers)) != len(containers)
            or any(not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", n) for n in containers)):
        parser.error("provide one to four unique, explicit Docker names")
    if not (1 <= args.refresh_seconds <= 30 and 0.1 <= args.timeout_seconds < args.refresh_seconds
            and args.stale_seconds >= 2 * args.refresh_seconds and 10 <= args.rediscover_seconds <= 300):
        parser.error("require refresh 1..30s, timeout < refresh, stale >= 2*refresh, rediscovery 10..300s")
    config = Config(args.node, role, service, ranks, containers, docker_socket=args.docker_socket,
                    proc_root=args.proc_root, refresh_seconds=args.refresh_seconds,
                    timeout_seconds=args.timeout_seconds, stale_seconds=args.stale_seconds,
                    rediscover_seconds=args.rediscover_seconds, allowed_dir_prefix=args.allowed_dir_prefix)
    worker, store, stop = BoundedWorker(config), SnapshotStore(config), threading.Event()
    if args.once:
        try:
            store.update(worker.refresh())
            payload, status = store.render()
            sys.stdout.buffer.write(payload)
            print(json.dumps(status, sort_keys=True), file=sys.stderr)
            sys.exit(0 if status["source_up"] else 2)
        finally:
            worker.close()

    def poll():
        while not stop.is_set():
            start = time.monotonic()
            result = worker.refresh()
            store.update(result)
            stop.wait(max(0.1, config.refresh_seconds - (time.monotonic() - start)))

    def shutdown(*_):
        stop.set()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    thread = threading.Thread(target=poll, name="bounded-cpu-collector", daemon=True)
    thread.start()
    server = make_server((args.listen, args.port), store)
    server.timeout = 0.5
    try:
        while not stop.is_set():
            server.handle_request()
    finally:
        stop.set()
        thread.join(config.timeout_seconds + 3)
        worker.close()
        server.server_close()


if __name__ == "__main__":
    main()
