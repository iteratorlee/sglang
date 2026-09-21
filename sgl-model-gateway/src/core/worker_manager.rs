//! Worker Management Module
//!
//! Provides worker lifecycle operations and fan-out request utilities.

use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};

use axum::response::{IntoResponse, Response};
use futures::{
    future,
    stream::{self, StreamExt},
};
use http::StatusCode;
use serde::Deserialize;
use serde_json::Value;
use tokio::{
    sync::{watch, Mutex},
    task::JoinHandle,
};
use tracing::{debug, info, warn};

use crate::{
    core::{metrics_aggregator::MetricPack, ConnectionMode, Worker, WorkerRegistry, WorkerType},
    policies::PolicyRegistry,
    protocols::worker_spec::{FlushCacheResult, WorkerLoadInfo, WorkerLoadsResult},
};

const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);
const MAX_CONCURRENT: usize = 32;

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct KVCapacitySnapshot {
    pub full_available_tokens: usize,
    pub full_evictable_tokens: usize,
    pub swa_available_tokens: Option<usize>,
    pub swa_evictable_tokens: Option<usize>,
    pub mamba_available_slots: Option<usize>,
    pub mamba_evictable_slots: Option<usize>,
    pub request_slots_available: Option<usize>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct QueueLoadSnapshot {
    pub waiting: usize,
    pub grammar: usize,
    pub paused: usize,
    pub retracted: usize,
    pub prealloc_ready: usize,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct DisaggregationLoadSnapshot {
    pub prefill_bootstrap_queue_reqs: usize,
    pub prefill_inflight_queue_reqs: usize,
    pub decode_prealloc_queue_reqs: usize,
    pub decode_transfer_queue_reqs: usize,
    pub decode_retracted_queue_reqs: usize,
}

fn observed_now() -> Instant {
    Instant::now()
}

fn monotonic_source_age(server: f64, sampled: f64) -> Option<Duration> {
    if !server.is_finite()
        || !sampled.is_finite()
        || server <= 0.0
        || sampled <= 0.0
        || server < sampled
    {
        return None;
    }

    // Validate each absolute value as well as the difference. Otherwise two
    // unrepresentably large, nearly equal floats can subtract to zero and be
    // mistaken for a fresh snapshot.
    Duration::try_from_secs_f64(server).ok()?;
    Duration::try_from_secs_f64(sampled).ok()?;
    Duration::try_from_secs_f64(server - sampled).ok()
}

/// Typed view of one SRT DP rank returned by `/v1/loads`.
///
/// Source age is derived only from the two monotonic samples in one SRT
/// response. `observed_at` then adds gateway residence time; no monotonic
/// values are compared across hosts.
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct DPLoadSnapshot {
    pub timestamp: f64,
    pub snapshot_monotonic_s: Option<f64>,
    pub dp_rank: usize,
    pub num_running_reqs: usize,
    pub num_waiting_reqs: usize,
    pub num_waiting_uncached_tokens: usize,
    pub num_used_tokens: usize,
    pub num_total_tokens: usize,
    pub num_active_tokens: usize,
    pub num_prealloc_ready_tokens: usize,
    pub max_total_num_tokens: usize,
    pub max_running_requests: usize,
    pub cache_hit_rate: f64,
    pub disaggregation: Option<DisaggregationLoadSnapshot>,
    pub queues: Option<QueueLoadSnapshot>,
    pub kv_capacity: Option<KVCapacitySnapshot>,
    #[serde(skip, default = "observed_now")]
    pub observed_at: Instant,
    #[serde(skip, default)]
    pub age_at_observation: Option<Duration>,
}

impl Default for DPLoadSnapshot {
    fn default() -> Self {
        Self {
            timestamp: 0.0,
            snapshot_monotonic_s: None,
            dp_rank: 0,
            num_running_reqs: 0,
            num_waiting_reqs: 0,
            num_waiting_uncached_tokens: 0,
            num_used_tokens: 0,
            num_total_tokens: 0,
            num_active_tokens: 0,
            num_prealloc_ready_tokens: 0,
            max_total_num_tokens: 0,
            max_running_requests: 0,
            cache_hit_rate: 0.0,
            disaggregation: None,
            queues: None,
            kv_capacity: None,
            observed_at: Instant::now(),
            age_at_observation: None,
        }
    }
}

impl DPLoadSnapshot {
    pub fn is_fresh(&self, max_age: Duration) -> bool {
        self.age_at_observation
            .is_some_and(|age| age.saturating_add(self.observed_at.elapsed()) <= max_age)
    }

    pub fn sampled_at_estimate(&self) -> Option<Instant> {
        self.age_at_observation
            .and_then(|age| self.observed_at.checked_sub(age))
    }
}

#[derive(Debug, Deserialize)]
struct LoadsEnvelope {
    #[serde(default)]
    server_monotonic_s: Option<f64>,
    loads: Vec<DPLoadSnapshot>,
}

#[derive(Clone)]
struct PhysicalWorkerGroup {
    base_url: String,
    api_key: Option<String>,
    virtual_workers: Vec<(String, Option<usize>)>,
}

pub const PD_TOKEN_AFFINITY_ENV: &str = "SGLANG_GATEWAY_PD_TOKEN_AFFINITY";

pub fn pd_token_affinity_enabled() -> bool {
    std::env::var(PD_TOKEN_AFFINITY_ENV)
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

/// Result of a fan-out request to a single worker
struct WorkerResponse {
    url: String,
    result: Result<reqwest::Response, reqwest::Error>,
}

/// Fan out requests to workers in parallel
async fn fan_out(
    workers: &[Arc<dyn Worker>],
    client: &reqwest::Client,
    endpoint: &str,
    method: reqwest::Method,
) -> Vec<WorkerResponse> {
    let futures: Vec<_> = workers
        .iter()
        .map(|worker| {
            let client = client.clone();
            let url = worker.url().to_string();
            let full_url = format!("{}/{}", url, endpoint);
            let api_key = worker.api_key().clone();
            let method = method.clone();

            async move {
                let mut req = client.request(method, &full_url).timeout(REQUEST_TIMEOUT);
                if let Some(key) = api_key {
                    req = req.bearer_auth(key);
                }
                WorkerResponse {
                    url,
                    result: req.send().await,
                }
            }
        })
        .collect();

    stream::iter(futures)
        .buffer_unordered(MAX_CONCURRENT)
        .collect()
        .await
}

pub enum EngineMetricsResult {
    Ok(String),
    Err(String),
}

impl IntoResponse for EngineMetricsResult {
    fn into_response(self) -> Response {
        match self {
            Self::Ok(text) => (StatusCode::OK, text).into_response(),
            Self::Err(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg).into_response(),
        }
    }
}

pub struct WorkerManager;

impl WorkerManager {
    pub fn get_worker_urls(registry: &Arc<WorkerRegistry>) -> Vec<String> {
        registry
            .get_all()
            .iter()
            .map(|w| w.url().to_string())
            .collect()
    }

    pub async fn flush_cache_all(
        worker_registry: &WorkerRegistry,
        client: &reqwest::Client,
    ) -> FlushCacheResult {
        let workers = worker_registry.get_all();
        let total_workers = workers.len();

        let http_workers: Vec<_> = workers
            .into_iter()
            .filter(|w| matches!(w.connection_mode(), ConnectionMode::Http))
            .collect();

        if http_workers.is_empty() {
            return FlushCacheResult {
                successful: vec![],
                failed: vec![],
                total_workers,
                http_workers: 0,
                message: "No HTTP workers available for cache flush".to_string(),
            };
        }

        info!(
            "Flushing cache on {} HTTP workers (out of {} total)",
            http_workers.len(),
            total_workers
        );

        let responses = fan_out(&http_workers, client, "flush_cache", reqwest::Method::POST).await;

        let mut successful = Vec::new();
        let mut failed = Vec::new();

        for resp in responses {
            match resp.result {
                Ok(r) if r.status().is_success() => successful.push(resp.url),
                Ok(r) => failed.push((resp.url, format!("HTTP {}", r.status()))),
                Err(e) => failed.push((resp.url, e.to_string())),
            }
        }

        let message = if failed.is_empty() {
            format!(
                "Successfully flushed cache on all {} HTTP workers",
                successful.len()
            )
        } else {
            format!(
                "Cache flush: {} succeeded, {} failed",
                successful.len(),
                failed.len()
            )
        };

        info!("{}", message);

        FlushCacheResult {
            successful,
            failed,
            total_workers,
            http_workers: http_workers.len(),
            message,
        }
    }

    pub async fn get_all_worker_loads(
        worker_registry: &WorkerRegistry,
        client: &reqwest::Client,
    ) -> WorkerLoadsResult {
        let workers = worker_registry.get_all();
        let total_workers = workers.len();
        let snapshots = Self::get_all_worker_load_snapshots(worker_registry, client).await;
        let loads: Vec<_> = workers
            .iter()
            .map(|worker| WorkerLoadInfo {
                worker: worker.url().to_string(),
                worker_type: match worker.worker_type() {
                    WorkerType::Regular => None,
                    WorkerType::Prefill { .. } => Some("prefill".to_string()),
                    WorkerType::Decode => Some("decode".to_string()),
                },
                load: snapshots
                    .get(worker.url())
                    .map(|snapshot| snapshot.num_total_tokens as isize)
                    .unwrap_or(-1),
            })
            .collect();
        let successful = loads.iter().filter(|l| l.load >= 0).count();
        let failed = loads.iter().filter(|l| l.load < 0).count();

        WorkerLoadsResult {
            loads,
            total_workers,
            successful,
            failed,
        }
    }

    fn physical_worker_groups(worker_registry: &WorkerRegistry) -> Vec<PhysicalWorkerGroup> {
        let mut groups: HashMap<String, PhysicalWorkerGroup> = HashMap::new();
        for worker in worker_registry.get_all() {
            if !matches!(worker.connection_mode(), ConnectionMode::Http) {
                continue;
            }
            let base_url = worker.base_url().to_string();
            let group = groups
                .entry(base_url.clone())
                .or_insert_with(|| PhysicalWorkerGroup {
                    base_url,
                    api_key: worker.api_key().clone(),
                    virtual_workers: Vec::new(),
                });
            group
                .virtual_workers
                .push((worker.url().to_string(), worker.dp_rank()));
        }
        groups.into_values().collect()
    }

    fn parse_load_payload(
        payload: Value,
        request_rtt: Duration,
    ) -> Result<Vec<DPLoadSnapshot>, String> {
        let mut envelope: LoadsEnvelope = serde_json::from_value(payload)
            .map_err(|e| format!("invalid /v1/loads payload: {e}"))?;
        let observed_at = Instant::now();
        for snapshot in &mut envelope.loads {
            snapshot.observed_at = observed_at;
            snapshot.age_at_observation = envelope
                .server_monotonic_s
                .zip(snapshot.snapshot_monotonic_s)
                .and_then(|(server, sampled)| monotonic_source_age(server, sampled))
                .map(|source_age| source_age.saturating_add(request_rtt));
        }
        Ok(envelope.loads)
    }

    async fn fetch_physical_loads(
        client: &reqwest::Client,
        base_url: &str,
        api_key: Option<&str>,
    ) -> Result<Vec<DPLoadSnapshot>, String> {
        // `all` is forward-compatible: old SRT does not know the new
        // `kv_capacity` section, while an updated SRT includes it in `all`.
        let load_url = format!("{}/v1/loads?include=all", base_url);
        let mut req = client.get(&load_url).timeout(REQUEST_TIMEOUT);
        if let Some(key) = api_key {
            req = req.bearer_auth(key);
        }
        let request_started = Instant::now();
        let response = req
            .send()
            .await
            .map_err(|e| format!("load request to {base_url} failed: {e}"))?;
        if !response.status().is_success() {
            return Err(format!(
                "load request to {base_url} returned {}",
                response.status()
            ));
        }
        let payload = response
            .json::<Value>()
            .await
            .map_err(|e| format!("load response from {base_url} is not JSON: {e}"))?;
        Self::parse_load_payload(payload, request_started.elapsed())
    }

    pub async fn get_all_worker_load_snapshots(
        worker_registry: &WorkerRegistry,
        client: &reqwest::Client,
    ) -> HashMap<String, DPLoadSnapshot> {
        let groups = Self::physical_worker_groups(worker_registry);
        let futures = groups.into_iter().map(|group| {
            let client = client.clone();
            async move {
                let result =
                    Self::fetch_physical_loads(&client, &group.base_url, group.api_key.as_deref())
                        .await;
                (group, result)
            }
        });

        let mut snapshots = HashMap::new();
        for (group, result) in future::join_all(futures).await {
            let physical_snapshots = match result {
                Ok(loads) => loads,
                Err(error) => {
                    warn!(base_url = %group.base_url, %error, "Failed to fetch worker load snapshot");
                    continue;
                }
            };
            for (worker_url, dp_rank) in group.virtual_workers {
                let rank = dp_rank.unwrap_or(0);
                if let Some(snapshot) = physical_snapshots.iter().find(|s| s.dp_rank == rank) {
                    snapshots.insert(worker_url, snapshot.clone());
                }
            }
        }
        snapshots
    }

    pub async fn get_engine_metrics(
        worker_registry: &WorkerRegistry,
        client: &reqwest::Client,
    ) -> EngineMetricsResult {
        let workers = worker_registry.get_all();

        if workers.is_empty() {
            return EngineMetricsResult::Err("No available workers".to_string());
        }

        let responses = fan_out(&workers, client, "metrics", reqwest::Method::GET).await;

        let mut metric_packs = Vec::new();
        for resp in responses {
            if let Ok(r) = resp.result {
                if r.status().is_success() {
                    if let Ok(text) = r.text().await {
                        metric_packs.push(MetricPack {
                            labels: vec![("worker_addr".into(), resp.url)],
                            metrics_text: text,
                        });
                    }
                }
            }
        }

        if metric_packs.is_empty() {
            return EngineMetricsResult::Err("All backend requests failed".to_string());
        }

        match crate::core::metrics_aggregator::aggregate_metrics(metric_packs) {
            Ok(text) => EngineMetricsResult::Ok(text),
            Err(e) => EngineMetricsResult::Err(format!("Failed to aggregate metrics: {}", e)),
        }
    }
}

/// Load monitoring service that periodically fetches worker loads
pub struct LoadMonitor {
    worker_registry: Arc<WorkerRegistry>,
    policy_registry: Arc<PolicyRegistry>,
    client: reqwest::Client,
    interval: Duration,
    tx: watch::Sender<HashMap<String, isize>>,
    rx: watch::Receiver<HashMap<String, isize>>,
    snapshot_tx: watch::Sender<HashMap<String, DPLoadSnapshot>>,
    snapshot_rx: watch::Receiver<HashMap<String, DPLoadSnapshot>>,
    monitor_handle: Arc<Mutex<Option<JoinHandle<()>>>>,
}

impl LoadMonitor {
    pub fn new(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        client: reqwest::Client,
        interval_secs: u64,
    ) -> Self {
        let (tx, rx) = watch::channel(HashMap::new());
        let (snapshot_tx, snapshot_rx) = watch::channel(HashMap::new());

        Self {
            worker_registry,
            policy_registry,
            client,
            interval: Duration::from_secs(interval_secs),
            tx,
            rx,
            snapshot_tx,
            snapshot_rx,
            monitor_handle: Arc::new(Mutex::new(None)),
        }
    }

    pub async fn start(&self) {
        let mut handle_guard = self.monitor_handle.lock().await;
        if handle_guard.is_some() {
            debug!("Load monitoring already running");
            return;
        }

        info!(
            "Starting load monitoring with interval: {:?}",
            self.interval
        );

        let worker_registry = Arc::clone(&self.worker_registry);
        let policy_registry = Arc::clone(&self.policy_registry);
        let client = self.client.clone();
        let interval = self.interval;
        let tx = self.tx.clone();
        let snapshot_tx = self.snapshot_tx.clone();

        let handle = tokio::spawn(async move {
            Self::monitor_loop(
                worker_registry,
                policy_registry,
                client,
                interval,
                tx,
                snapshot_tx,
            )
            .await;
        });

        *handle_guard = Some(handle);
    }

    pub async fn stop(&self) {
        let mut handle_guard = self.monitor_handle.lock().await;
        if let Some(handle) = handle_guard.take() {
            info!("Stopping load monitoring");
            handle.abort();
            let _ = handle.await; // Wait for task to finish
        }
    }

    pub fn subscribe(&self) -> watch::Receiver<HashMap<String, isize>> {
        self.rx.clone()
    }

    pub fn subscribe_snapshots(&self) -> watch::Receiver<HashMap<String, DPLoadSnapshot>> {
        self.snapshot_rx.clone()
    }

    async fn monitor_loop(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        client: reqwest::Client,
        interval: Duration,
        tx: watch::Sender<HashMap<String, isize>>,
        snapshot_tx: watch::Sender<HashMap<String, DPLoadSnapshot>>,
    ) {
        let mut interval_timer = tokio::time::interval(interval);
        let detailed_feedback_enabled = pd_token_affinity_enabled();

        loop {
            interval_timer.tick().await;

            // Preserve the old zero-overhead behavior unless either an existing
            // power-of-two policy or the opt-in PD feedback path consumes loads.
            let power_of_two_policies = policy_registry.get_all_power_of_two_policies();
            if power_of_two_policies.is_empty() && !detailed_feedback_enabled {
                continue;
            }

            let snapshots =
                WorkerManager::get_all_worker_load_snapshots(&worker_registry, &client).await;
            let loads: HashMap<String, isize> = snapshots
                .iter()
                .map(|(worker, snapshot)| (worker.clone(), snapshot.num_total_tokens as isize))
                .collect();

            if !loads.is_empty() {
                debug!(
                    "Fetched loads from {} workers, updating {} PowerOfTwo policies",
                    loads.len(),
                    power_of_two_policies.len()
                );
                for policy in &power_of_two_policies {
                    policy.update_loads(&loads);
                }
                let _ = tx.send(loads);
                let _ = snapshot_tx.send(snapshots);
            } else {
                warn!("No loads fetched from workers");
            }
        }
    }

    pub async fn is_running(&self) -> bool {
        let handle_guard = self.monitor_handle.lock().await;
        handle_guard.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_current_srt_loads_array_and_capacity() {
        let payload = serde_json::json!({
            "timestamp": "2026-09-21T00:00:00Z",
            "server_monotonic_s": 100.25,
            "loads": [{
                "snapshot_monotonic_s": 100.0,
                "dp_rank": 3,
                "num_running_reqs": 7,
                "num_total_tokens": 131072,
                "num_active_tokens": 120000,
                "max_total_num_tokens": 532736,
                "queues": {"waiting": 2, "prealloc_ready": 1},
                "disaggregation": {
                    "decode_prealloc_queue_reqs": 2,
                    "decode_transfer_queue_reqs": 1
                },
                "kv_capacity": {
                    "full_available_tokens": 390000,
                    "full_evictable_tokens": 4096,
                    "mamba_available_slots": 12,
                    "request_slots_available": 9
                }
            }]
        });

        let loads = WorkerManager::parse_load_payload(payload, Duration::from_millis(10)).unwrap();
        assert_eq!(loads.len(), 1);
        let load = &loads[0];
        assert_eq!(load.dp_rank, 3);
        assert_eq!(load.num_running_reqs, 7);
        assert_eq!(load.queues.as_ref().unwrap().waiting, 2);
        let capacity = load.kv_capacity.as_ref().unwrap();
        assert_eq!(capacity.full_available_tokens, 390000);
        assert_eq!(capacity.request_slots_available, Some(9));
        let age = load.age_at_observation.unwrap();
        assert!(age >= Duration::from_millis(260));
        assert!(age < Duration::from_millis(300));
    }

    #[test]
    fn invalid_or_missing_monotonic_contract_is_stale_without_panicking() {
        assert_eq!(monotonic_source_age(f64::NAN, 1.0), None);
        assert_eq!(monotonic_source_age(2.0, f64::INFINITY), None);
        assert_eq!(monotonic_source_age(1e308, 1e308), None);

        for payload in [
            serde_json::json!({"loads": [{"dp_rank": 0}]}),
            serde_json::json!({
                "server_monotonic_s": 9.0,
                "loads": [{"dp_rank": 0, "snapshot_monotonic_s": 10.0}]
            }),
            serde_json::json!({
                "server_monotonic_s": 1e308,
                "loads": [{"dp_rank": 0, "snapshot_monotonic_s": 1.0}]
            }),
            serde_json::json!({
                "server_monotonic_s": 0.0,
                "loads": [{"dp_rank": 0, "snapshot_monotonic_s": 0.0}]
            }),
            serde_json::json!({
                "server_monotonic_s": -1.0,
                "loads": [{"dp_rank": 0, "snapshot_monotonic_s": -2.0}]
            }),
        ] {
            let loads = WorkerManager::parse_load_payload(payload, Duration::ZERO).unwrap();
            assert_eq!(loads.len(), 1);
            assert_eq!(loads[0].age_at_observation, None);
            assert!(!loads[0].is_fresh(Duration::from_secs(3)));
            assert!(loads[0].sampled_at_estimate().is_none());
        }
    }

    #[test]
    fn groups_virtual_dp_workers_by_physical_url() {
        use crate::core::{DPAwareWorkerBuilder, WorkerType};

        let registry = WorkerRegistry::new();
        for rank in 0..4 {
            registry.register(Arc::new(
                DPAwareWorkerBuilder::new("http://decode:30001", rank, 4)
                    .worker_type(WorkerType::Decode)
                    .build(),
            ));
        }
        registry.register(Arc::new(
            DPAwareWorkerBuilder::new("http://prefill:30000", 0, 1)
                .worker_type(WorkerType::Prefill {
                    bootstrap_port: None,
                })
                .build(),
        ));

        let mut groups = WorkerManager::physical_worker_groups(&registry);
        groups.sort_by(|a, b| a.base_url.cmp(&b.base_url));
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].base_url, "http://decode:30001");
        assert_eq!(groups[0].virtual_workers.len(), 4);
        assert_eq!(groups[1].base_url, "http://prefill:30000");
        assert_eq!(groups[1].virtual_workers.len(), 1);
    }

    #[test]
    fn rejects_obsolete_aggregate_only_shape() {
        let payload = serde_json::json!({"aggregate": {"total_tokens": 10}});
        assert!(WorkerManager::parse_load_payload(payload, Duration::ZERO).is_err());
    }
}

impl Drop for LoadMonitor {
    fn drop(&mut self) {
        if let Ok(mut handle_guard) = self.monitor_handle.try_lock() {
            if let Some(handle) = handle_guard.take() {
                handle.abort();
            }
        }
    }
}
