#!/usr/bin/env python3
"""Run the production prefix/Chat affinity unit tests without building the gateway.

Usage: taskset -c 30,31 python tests/run_prefix_hash_cpu.py [cargo test filter]
Requires an offline Cargo cache. Builds only a temporary debug crate, never
starts a server. Includes prefix_hash.rs verbatim; extracts the production
HashRing, snapshot types/freshness methods and SelectWorkerInfo. Also compiles
chat_affinity.rs verbatim, but skips its HTTP listener test. Worker health,
load counters and metrics are test doubles, so this is not an HTTP integration
test or a substitute for the full gateway tests before deployment.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def section(path, start, end):
    source = path.read_text()
    return source.split(start, 1)[1].split(end, 1)[0]


def main():
    gateway = Path(__file__).resolve().parents[1]
    src = gateway / "src"
    build = Path(tempfile.mkdtemp(prefix="prefix-hash-cpu-"))
    (build / "src").mkdir()
    (build / "Cargo.toml").write_text('''[package]
name = "prefix-hash-cpu"
version = "0.0.0"
edition = "2021"
[dependencies]
async-trait = "0.1"
axum = "0.8"
blake3 = "1"
bytemuck = "1"
http = "1"
metrics = "0.24"
rand = "0.9"
reqwest = { version = "0.12", default-features = false, features = ["json", "stream"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tokio = { version = "1", features = ["macros", "rt-multi-thread", "net", "sync"] }
tracing = "0.1"
xxhash-rust = { version = "0.8", features = ["xxh3"] }
[profile.dev]
debug = 0
''')
    snapshots = section(
        src / "core/worker_manager.rs",
        "const MAX_CONCURRENT: usize = 32;",
        "#[derive(Debug, Deserialize)]\nstruct LoadsEnvelope",
    )
    ring = section(
        src / "core/worker_registry.rs",
        "/// Number of virtual nodes per physical worker for even distribution.",
        "/// Unique identifier for a worker",
    )
    info = section(
        src / "policies/mod.rs",
        "/// Information passed to policy for worker selection",
        "#[cfg(test)]",
    )
    api_path = section(src / "routers/http/pd_types.rs",
                       "/// Construct a full API URL from a base URL and path.",
                       "use serde::Serialize;")
    (build / "src/lib.rs").write_text('''#![allow(dead_code)]
mod observability { pub mod metrics {
    pub struct Metrics;
    impl Metrics { pub fn record_worker_prefix_hash_policy_branch(_: &str) {} }
} }
mod core {
    use std::{collections::hash_map::DefaultHasher, hash::{Hash, Hasher},
        sync::{Arc, atomic::{AtomicBool, AtomicUsize, Ordering}}};
    #[derive(Clone, Debug)]
    pub enum WorkerType { Regular, Prefill { bootstrap_port: Option<u16> }, Decode }
    impl WorkerType {
        pub fn as_metric_label(&self) -> &'static str {
            match self { Self::Regular => "regular", Self::Prefill {..} => "prefill", Self::Decode => "decode" }
        }
    }
    pub trait Worker: std::fmt::Debug + Send + Sync {
        fn url(&self) -> &str;
        fn model_id(&self) -> &str;
        fn worker_type(&self) -> &WorkerType;
        fn base_url(&self) -> &str { self.url() }
        fn api_key(&self) -> &Option<String> { &None }
        fn load(&self) -> usize;
        fn increment_load(&self);
        fn is_available(&self) -> bool;
        fn set_healthy(&self, healthy: bool);
    }
    #[derive(Debug)]
    pub struct BasicWorker {
        url: String, model: String, kind: WorkerType,
        load: AtomicUsize, healthy: AtomicBool,
    }
    impl Worker for BasicWorker {
        fn url(&self) -> &str { &self.url }
        fn model_id(&self) -> &str { &self.model }
        fn worker_type(&self) -> &WorkerType { &self.kind }
        fn load(&self) -> usize { self.load.load(Ordering::Relaxed) }
        fn increment_load(&self) { self.load.fetch_add(1, Ordering::Relaxed); }
        fn is_available(&self) -> bool { self.healthy.load(Ordering::Relaxed) }
        fn set_healthy(&self, healthy: bool) { self.healthy.store(healthy, Ordering::Relaxed); }
    }
    pub struct BasicWorkerBuilder(BasicWorker);
    impl BasicWorkerBuilder {
        pub fn new(url: impl Into<String>) -> Self {
            Self(BasicWorker {url: url.into(), model: "unknown".into(), kind: WorkerType::Regular,
                load: AtomicUsize::new(0), healthy: AtomicBool::new(true)})
        }
        pub fn worker_type(mut self, kind: WorkerType) -> Self { self.0.kind = kind; self }
        pub fn label(mut self, key: &str, value: &str) -> Self {
            assert_eq!(key, "model_id"); self.0.model = value.into(); self
        }
        pub fn build(self) -> BasicWorker { self.0 }
    }
    pub mod worker_manager {
        use std::time::{Duration, Instant};
        use serde::Deserialize;
''' + snapshots + '''
    }
    pub use worker_manager::DPLoadSnapshot;
''' + ring + '''
}
mod policies {
    use std::{collections::HashMap, sync::Arc};
    use crate::core::{DPLoadSnapshot, HashRing, Worker};
    #[async_trait::async_trait]
    pub trait LoadBalancingPolicy: Send + Sync + std::fmt::Debug {
        async fn select_worker(&self, workers: &[Arc<dyn Worker>], info: &SelectWorkerInfo<'_>) -> Option<usize>;
        fn name(&self) -> &'static str;
        fn as_any(&self) -> &dyn std::any::Any;
    }
''' + info + f'''
    #[path = {json.dumps(str(src / "policies/prefix_hash.rs"))}]
    mod prefix_hash;
}}
mod routers {{ pub mod http {{
    mod pd_types {{ {api_path} }}
    #[path = {json.dumps(str(src / "routers/http/chat_affinity.rs"))}]
    mod chat_affinity;
}} }}
''')
    # Do not inherit deployment feature flags into deterministic policy fixtures.
    env = {k: v for k, v in os.environ.items() if not k.startswith("SGLANG_GATEWAY_PD_")}
    env["CARGO_BUILD_JOBS"] = "2"
    env["CARGO_TARGET_DIR"] = str(build / "target")
    print(f"CPU harness: {build}", flush=True)
    return subprocess.run(
        ["cargo", "test", "--offline", "--manifest-path", str(build / "Cargo.toml"),
         *sys.argv[1:], "--", "--skip", "transport_keeps_stream_true_and_rejects_malformed_token_responses"], env=env,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
