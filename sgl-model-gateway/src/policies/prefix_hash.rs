//! Prefix Hash routing policy for KV cache-aware load balancing
//!
//! A lightweight alternative to the full radix tree cache_aware policy.
//! Routes requests based on a hash of their prefix tokens to maximize
//! KV cache hits across workers.
//!
//! ## Algorithm
//!
//! 1. Extract first N tokens from the request (configurable prefix length)
//! 2. Hash the token sequence using xxhash for fast, stable hashing
//! 3. Use consistent hash ring to find the target worker
//! 4. If worker is overloaded (load > avg * load_factor), find least loaded
//! 5. Return least loaded worker that passes load check, or initial if all overloaded
//!
//! ## Complexity
//!
//! - Hash computation: O(prefix_length)
//! - Ring lookup: O(log n) binary search
//! - Load balance fallback: O(n) scan for least loaded
//!
//! ## Comparison with cache_aware
//!
//! | Aspect          | prefix_hash       | cache_aware (radix) |
//! |-----------------|-------------------|---------------------|
//! | Lookup          | O(log n)          | O(prefix_len)       |
//! | Memory          | O(workers × vn)   | O(total_tokens)     |
//! | Update          | O(1)              | O(prefix_len)       |
//! | Precision       | Prefix grouping   | Exact matching      |
//!
//! prefix_hash trades optimal cache utilization for predictable O(log n) performance.
//!
//! `SGLANG_GATEWAY_PD_AFFINITY_PREFIX_TOKENS=N` optionally uses the first N
//! exact tokens (longer than the configured short prefix, at most 65536) in
//! homogeneous P/D pools. Long prompts remember the actual selected owner,
//! including load fallbacks, subject to all existing affinity checks. Short
//! prompts and non-PD pools retain the original policy. This is an advisory
//! placement observation, not an LCP index or evidence of resident cache state.

use std::{
    collections::{hash_map::DefaultHasher, HashMap as PlacementMap},
    hash::{Hash, Hasher},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use super::{LoadBalancingPolicy, SelectWorkerInfo};
use crate::{
    core::{DPLoadSnapshot, Worker, WorkerType},
    observability::metrics::Metrics,
};

const LOAD_SNAPSHOT_MAX_AGE: Duration = Duration::from_secs(3);
const PREFILL_PLACEMENT_TTL: Duration = Duration::from_secs(7200);
const PREFILL_PLACEMENT_CAPACITY: usize = 4096;
const MAX_PD_AFFINITY_PREFIX_TOKENS: usize = 65_536;

type PlacementKey = (u64, u64); // (pool fingerprint, prefix hash)

#[derive(Debug)]
struct PlacementEntry {
    owner_url: String,
    observed_at: Instant,
    sequence: u64,
}

#[derive(Debug, Default)]
struct PlacementBook {
    entries: PlacementMap<PlacementKey, PlacementEntry>,
    next_sequence: u64,
}

impl PlacementBook {
    fn owner(&mut self, key: PlacementKey, now: Instant) -> Option<String> {
        let entry = self.entries.get(&key)?;
        if now.duration_since(entry.observed_at) >= PREFILL_PLACEMENT_TTL {
            self.entries.remove(&key);
            return None;
        }
        Some(entry.owner_url.clone())
    }

    fn record(&mut self, key: PlacementKey, owner_url: &str, now: Instant) {
        // Keep a single map entry per prefix. At the hard cap, expire old
        // observations and evict the oldest remaining one without a duplicate
        // queue that can grow on repeated updates to a hot prefix.
        if !self.entries.contains_key(&key) && self.entries.len() >= PREFILL_PLACEMENT_CAPACITY {
            self.entries
                .retain(|_, entry| now.duration_since(entry.observed_at) < PREFILL_PLACEMENT_TTL);
            if self.entries.len() >= PREFILL_PLACEMENT_CAPACITY {
                if let Some(oldest) = self
                    .entries
                    .iter()
                    .min_by_key(|(_, entry)| entry.sequence)
                    .map(|(key, _)| *key)
                {
                    self.entries.remove(&oldest);
                }
            }
        }
        self.next_sequence = self.next_sequence.wrapping_add(1);
        self.entries.insert(
            key,
            PlacementEntry {
                owner_url: owner_url.to_owned(),
                observed_at: now,
                sequence: self.next_sequence,
            },
        );
    }
}

/// Configuration for the PrefixHash load balancing policy
#[derive(Debug, Clone)]
pub struct PrefixHashConfig {
    /// Number of prefix tokens to use for hashing.
    /// Longer prefixes = more precise routing but less grouping.
    /// Shorter prefixes = more requests grouped together.
    /// Default: 256 tokens (~1 paragraph of text)
    pub prefix_token_count: usize,

    /// Load factor threshold for walking the ring.
    /// If a worker's load > (total_load / num_workers) * load_factor,
    /// walk clockwise to the next worker.
    /// Default: 1.25 (125% of average load)
    pub load_factor: f64,
}

impl Default for PrefixHashConfig {
    fn default() -> Self {
        Self {
            prefix_token_count: 256,
            load_factor: 1.25,
        }
    }
}

/// Execution branch for metrics
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Branch {
    NoHealthyWorkers,
    NoTokens,
    RingHit,
    ObservedPlacement,
    LoadBalanceWalk,
    FallbackLeastLoad,
}

impl Branch {
    #[inline]
    const fn as_str(&self) -> &'static str {
        match self {
            Self::NoHealthyWorkers => "no_healthy_workers",
            Self::NoTokens => "no_tokens",
            Self::RingHit => "ring_hit",
            Self::ObservedPlacement => "observed_placement",
            Self::LoadBalanceWalk => "load_balance_walk",
            Self::FallbackLeastLoad => "fallback_least_load",
        }
    }
}

/// Prefix Hash load balancing policy
///
/// Routes requests based on prefix token hash for KV cache locality.
/// Uses consistent hashing with bounded load balancing.
#[derive(Debug)]
pub struct PrefixHashPolicy {
    config: PrefixHashConfig,
    /// A P can reclaim unlocked prefix-cache pages. Keep this opt-in until
    /// paired replay validation; D's full-output budget stays unchanged.
    pd_prefill_cache_capacity: bool,
    pd_prefill_affinity_slack: Option<usize>,
    pd_prefill_stage_reservation: bool,
    pd_prefill_work_tokens: Option<usize>,
    /// Selection observations only; a dispatch can still be in flight or fail.
    pd_prefill_placement: bool,
    placement_book: Mutex<PlacementBook>,
    /// Opt-in longer exact-token key, with independent observations for P/D.
    /// Only full-length keys participate, so appending to a short prompt does
    /// not continually change its observed owner.
    pd_affinity_prefix_tokens: Option<usize>,
    pd_affinity_book: Mutex<PlacementBook>,
}

impl PrefixHashPolicy {
    /// Create a new PrefixHashPolicy with the given configuration
    pub fn new(config: PrefixHashConfig) -> Self {
        let pd_affinity_prefix_tokens = std::env::var("SGLANG_GATEWAY_PD_AFFINITY_PREFIX_TOKENS")
            .ok()
            .and_then(|value| Self::parse_pd_affinity_prefix_tokens(&value, &config));
        Self {
            config,
            pd_prefill_cache_capacity: std::env::var("SGLANG_GATEWAY_PD_PREFILL_CACHE_CAPACITY")
                .is_ok_and(|value| value == "1"),
            pd_prefill_affinity_slack: std::env::var("SGLANG_GATEWAY_PD_PREFILL_AFFINITY_SLACK")
                .ok()
                .and_then(|value| value.parse().ok()),
            pd_prefill_stage_reservation: std::env::var(
                "SGLANG_GATEWAY_PD_PREFILL_STAGE_RESERVATION",
            )
            .is_ok_and(|value| value == "1"),
            pd_prefill_work_tokens: std::env::var("SGLANG_GATEWAY_PD_PREFILL_WORK_TOKENS")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .filter(|value| *value > 0),
            pd_prefill_placement: std::env::var("SGLANG_GATEWAY_PD_PREFILL_PLACEMENT")
                .is_ok_and(|value| value == "1"),
            placement_book: Mutex::new(PlacementBook::default()),
            pd_affinity_prefix_tokens,
            pd_affinity_book: Mutex::new(PlacementBook::default()),
        }
    }

    fn parse_pd_affinity_prefix_tokens(value: &str, config: &PrefixHashConfig) -> Option<usize> {
        value.parse().ok().filter(|count| {
            *count > config.prefix_token_count && *count <= MAX_PD_AFFINITY_PREFIX_TOKENS
        })
    }

    /// Create a new PrefixHashPolicy with default configuration
    pub fn with_defaults() -> Self {
        Self::new(PrefixHashConfig::default())
    }

    /// Compute hash of prefix tokens using xxhash
    #[inline]
    fn compute_prefix_hash(&self, tokens: &[u32]) -> u64 {
        let prefix_len = tokens.len().min(self.config.prefix_token_count);
        let prefix = &tokens[..prefix_len];

        let bytes: &[u8] = bytemuck::cast_slice(prefix);
        xxhash_rust::xxh3::xxh3_64(bytes)
    }

    /// Check if a worker's load is acceptable
    #[inline]
    fn load_ok(&self, worker_load: usize, total_load: usize, num_workers: usize) -> bool {
        if total_load == 0 || num_workers == 0 {
            return true;
        }

        // Average load per worker (with +1 to simulate incoming request)
        let avg_load = (total_load + 1) as f64 / num_workers as f64;
        let threshold = avg_load * self.config.load_factor;

        (worker_load as f64) <= threshold
    }

    fn snapshot_for<'a>(
        worker: &dyn Worker,
        info: &'a SelectWorkerInfo<'_>,
    ) -> Option<&'a DPLoadSnapshot> {
        info.load_snapshots?
            .get(worker.url())
            .filter(|snapshot| snapshot.is_fresh(LOAD_SNAPSHOT_MAX_AGE))
    }

    fn local_reservations(worker: &dyn Worker, info: &SelectWorkerInfo<'_>) -> usize {
        info.local_reservations
            .and_then(|loads| loads.get(worker.url()))
            .copied()
            .unwrap_or(0)
    }

    fn effective_load(&self, worker: &dyn Worker, info: &SelectWorkerInfo<'_>) -> usize {
        let local = worker.load().max(Self::local_reservations(worker, info));
        let Some(snapshot) = Self::snapshot_for(worker, info) else {
            return local;
        };
        let cache_aware_prefill = self.pd_prefill_cache_capacity
            && matches!(worker.worker_type(), WorkerType::Prefill { .. });
        let stage_aware_prefill = self.pd_prefill_stage_reservation
            && matches!(worker.worker_type(), WorkerType::Prefill { .. });
        let runnable_waiting = snapshot
            .queues
            .as_ref()
            .map(|queues| queues.waiting)
            .unwrap_or(snapshot.num_waiting_reqs);
        let stage_pressure = match worker.worker_type() {
            WorkerType::Prefill { .. } => snapshot
                .disaggregation
                .as_ref()
                .map(|load| load.prefill_bootstrap_queue_reqs + load.prefill_inflight_queue_reqs)
                .unwrap_or(0),
            WorkerType::Decode => snapshot
                .disaggregation
                .as_ref()
                .map(|load| {
                    load.decode_prealloc_queue_reqs
                        + load.decode_transfer_queue_reqs
                        + load.decode_retracted_queue_reqs
                })
                .unwrap_or(0),
            WorkerType::Regular => 0,
        };
        let post_snapshot_tokens = info
            .post_snapshot_reserved_tokens
            .and_then(|loads| loads.get(worker.url()))
            .copied()
            .unwrap_or(0);
        let post_snapshot_requests = info
            .post_snapshot_reserved_requests
            .and_then(|loads| loads.get(worker.url()))
            .copied()
            .unwrap_or(0);
        let capacity_pressure = snapshot.kv_capacity.as_ref().map_or(0, |capacity| {
            // P produces one boundary token, whereas estimated_tokens includes
            // D's entire future output. Do not charge that output to P's pool.
            let request_tokens = if cache_aware_prefill {
                info.tokens
                    .map(|tokens| tokens.len().saturating_add(1))
                    .unwrap_or(info.estimated_tokens)
            } else {
                info.estimated_tokens
            };
            let demand = request_tokens.saturating_add(post_snapshot_tokens);
            let available = if cache_aware_prefill {
                capacity
                    .full_available_tokens
                    .saturating_add(capacity.full_evictable_tokens)
            } else {
                capacity.full_available_tokens
            };
            let token_pressure = if demand > available {
                let deficit = demand - available;
                1usize.saturating_add(deficit / request_tokens.max(1))
            } else {
                0
            };
            let slot_pressure = usize::from(
                capacity
                    .request_slots_available
                    .is_some_and(|slots| slots <= post_snapshot_requests),
            );
            let mamba_pressure = if cache_aware_prefill {
                usize::from(capacity.mamba_available_slots.is_some_and(|slots| {
                    slots.saturating_add(capacity.mamba_evictable_slots.unwrap_or(0))
                        <= post_snapshot_requests
                }))
            } else {
                0
            };
            token_pressure
                .saturating_add(slot_pressure)
                .saturating_add(mamba_pressure)
        });

        // For SSE, the gateway's P WorkerLoadGuard survives until D finishes
        // streaming, even though P is done. A fresh P snapshot represents its
        // actual lifecycle; stale/missing snapshots still fall back to local.
        // A P-stage lease ends when P's response has been consumed. Its
        // lifecycle must not depend on opting into reclaimable-cache capacity:
        // the generic WorkerLoadGuard can outlive P through all of D's stream.
        let active = if cache_aware_prefill || stage_aware_prefill {
            Self::local_reservations(worker, info)
        } else {
            local
        };
        if stage_aware_prefill {
            // The local P lease spans dispatch through its consumed response
            // body. These same requests can already occur in SRT's snapshot;
            // take the larger lifecycle count instead of counting them twice.
            let work_pressure = self.pd_prefill_work_tokens.map_or(0, |quantum| {
                // SRT includes the remaining chunked prompt in this gauge;
                // request counts alone hide a long prefill from the policy.
                snapshot.num_waiting_uncached_tokens.div_ceil(quantum)
            });
            return active
                .max(
                    snapshot
                        .num_running_reqs
                        .saturating_add(runnable_waiting)
                        .saturating_add(stage_pressure),
                )
                .saturating_add(capacity_pressure)
                .saturating_add(work_pressure);
        }
        active
            .max(snapshot.num_running_reqs)
            .saturating_add(runnable_waiting)
            .saturating_add(stage_pressure)
            .saturating_add(capacity_pressure)
    }

    fn under_soft_cap(worker: &dyn Worker, info: &SelectWorkerInfo<'_>) -> bool {
        !info
            .soft_cap
            .is_some_and(|cap| Self::local_reservations(worker, info) >= cap)
    }

    fn pool_fingerprint(workers: &[Arc<dyn Worker>]) -> u64 {
        let mut members: Vec<_> = workers
            .iter()
            .map(|worker| (worker.url(), worker.model_id()))
            .collect();
        members.sort_unstable();
        let mut hasher = DefaultHasher::new();
        members.hash(&mut hasher);
        hasher.finish()
    }

    fn pd_pool_fingerprint(workers: &[Arc<dyn Worker>], info: &SelectWorkerInfo) -> u64 {
        let mut hasher = DefaultHasher::new();
        // HTTP PD passes an availability-filtered worker slice. Use the cached
        // registry ring to avoid forgetting a fallback when a capped/unhealthy
        // worker returns. The shared model ring contains both P and D, so also
        // namespace by role and model. Without a ring, use the supplied pool.
        info.hash_ring
            .as_ref()
            .map_or_else(
                || Self::pool_fingerprint(workers),
                |r| r.membership_fingerprint(),
            )
            .hash(&mut hasher);
        let mut models: Vec<_> = workers.iter().map(|w| w.model_id()).collect();
        models.sort_unstable();
        models.dedup();
        models.hash(&mut hasher);
        workers[0].worker_type().as_metric_label().hash(&mut hasher);
        hasher.finish()
    }

    /// Apply the same allowance used for a healthy ring owner.
    fn affinity_allowed(
        &self,
        worker: &dyn Worker,
        info: &SelectWorkerInfo<'_>,
        worker_load: usize,
        least_load: usize,
        total_load: usize,
        num_workers: usize,
    ) -> bool {
        if matches!(worker.worker_type(), WorkerType::Prefill { .. }) {
            if let Some(slack) = self.pd_prefill_affinity_slack {
                if worker_load <= least_load.saturating_add(slack) {
                    return true;
                }
            }
        }
        if info.soft_cap.is_some() && worker_load <= least_load.saturating_add(1) {
            return true;
        }
        self.load_ok(worker_load, total_load, num_workers)
    }

    fn observed_owner_allowed(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
        owner_url: &str,
    ) -> Option<usize> {
        let available: Vec<_> = workers
            .iter()
            .enumerate()
            .filter(|(_, worker)| {
                worker.is_available() && Self::under_soft_cap(worker.as_ref(), info)
            })
            .map(|(index, worker)| (index, worker, self.effective_load(worker.as_ref(), info)))
            .collect();
        let (index, worker, owner_load) = available
            .iter()
            .find(|(_, worker, _)| worker.url() == owner_url)?;
        let total_load = available.iter().map(|(_, _, load)| load).sum();
        let least_load = available
            .iter()
            .map(|(_, _, load)| *load)
            .min()
            .unwrap_or(*owner_load);
        self.affinity_allowed(
            worker.as_ref(),
            info,
            *owner_load,
            least_load,
            total_load,
            available.len(),
        )
        .then_some(*index)
    }

    /// Find worker using consistent hash ring with load balancing
    fn find_worker_with_load_balance(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo,
        prefix_hash: u64,
    ) -> (Option<usize>, Branch) {
        // Build healthy worker URL to index map
        let healthy_workers: Vec<(usize, &Arc<dyn Worker>)> = workers
            .iter()
            .enumerate()
            .filter(|(_, w)| w.is_available() && Self::under_soft_cap(w.as_ref(), info))
            .collect();

        if healthy_workers.is_empty() {
            return (None, Branch::NoHealthyWorkers);
        }

        // Calculate total load for load balancing
        let total_load: usize = healthy_workers
            .iter()
            .map(|(_, w)| self.effective_load(w.as_ref(), info))
            .sum();
        let num_workers = healthy_workers.len();

        // Use pre-computed ring if available
        if let Some(ref ring) = info.hash_ring {
            // Convert prefix hash to a ring key string for lookup
            let key = format!("{:016x}", prefix_hash);

            // Build URL to (index, worker) map for healthy workers
            let healthy_url_map: std::collections::HashMap<&str, (usize, &Arc<dyn Worker>)> =
                healthy_workers
                    .iter()
                    .map(|(idx, w)| (w.url(), (*idx, *w)))
                    .collect();

            // Find initial worker from ring
            if let Some(initial_url) =
                ring.find_healthy_url(&key, |url| healthy_url_map.contains_key(url))
            {
                if let Some(&(idx, worker)) = healthy_url_map.get(initial_url) {
                    let worker_load = self.effective_load(worker.as_ref(), info);

                    let least_load = healthy_workers
                        .iter()
                        .map(|(_, worker)| self.effective_load(worker.as_ref(), info))
                        .min()
                        .unwrap_or(worker_load);
                    if self.affinity_allowed(
                        worker.as_ref(),
                        info,
                        worker_load,
                        least_load,
                        total_load,
                        num_workers,
                    ) {
                        return (Some(idx), Branch::RingHit);
                    }

                    // Initial worker overloaded, find least loaded healthy worker
                    // This is a simpler approach than walking the ring
                    let least_loaded = healthy_workers
                        .iter()
                        .filter(|(_, w)| {
                            self.load_ok(
                                self.effective_load(w.as_ref(), info),
                                total_load,
                                num_workers,
                            )
                        })
                        .min_by_key(|(_, w)| self.effective_load(w.as_ref(), info));

                    if let Some(&(idx, _)) = least_loaded {
                        return (Some(idx), Branch::LoadBalanceWalk);
                    }

                    // All workers overloaded, use initial worker anyway
                    return (Some(idx), Branch::LoadBalanceWalk);
                }
            }
        }

        // Fallback: no ring or ring lookup failed, use least loaded worker
        let least_loaded = healthy_workers
            .iter()
            .min_by_key(|(_, w)| self.effective_load(w.as_ref(), info))
            .map(|(idx, _)| *idx);

        (least_loaded, Branch::FallbackLeastLoad)
    }

    /// HTTP Chat's explicit compatibility fallback. Preserve existing load,
    /// capacity and soft-cap logic without pretending to know prompt token IDs.
    pub(crate) fn select_without_prefix(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo,
    ) -> Option<usize> {
        self.find_worker_with_load_balance(workers, info, rand::random())
            .0
    }

    fn select_worker_impl(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo,
    ) -> (Option<usize>, Branch) {
        if workers.is_empty() {
            return (None, Branch::NoHealthyWorkers);
        }

        // Get tokens from SelectWorkerInfo
        let tokens = match info.tokens {
            Some(t) if !t.is_empty() => t,
            _ => return (None, Branch::NoTokens),
        };

        if let Some(count) = self.pd_affinity_prefix_tokens {
            let pd_pool = workers
                .iter()
                .all(|w| matches!(w.worker_type(), WorkerType::Prefill { .. }))
                || workers
                    .iter()
                    .all(|w| matches!(w.worker_type(), WorkerType::Decode));
            if pd_pool && tokens.len() >= count {
                let hash = xxhash_rust::xxh3::xxh3_64(bytemuck::cast_slice(&tokens[..count]));
                return self.select_with_observed_owner(
                    workers,
                    info,
                    hash,
                    &self.pd_affinity_book,
                    Self::pd_pool_fingerprint(workers, info),
                );
            }
        }

        // Compute prefix hash
        let prefix_hash = self.compute_prefix_hash(tokens);

        // Placement is opt-in for the P-only policy with stage-aware capacity.
        // Never use a selection observation as evidence of a completed prefill.
        if !(self.pd_prefill_placement
            && self.pd_prefill_cache_capacity
            && self.pd_prefill_stage_reservation
            && workers
                .iter()
                .all(|worker| matches!(worker.worker_type(), WorkerType::Prefill { .. })))
        {
            return self.find_worker_with_load_balance(workers, info, prefix_hash);
        }

        self.select_with_observed_owner(
            workers,
            info,
            prefix_hash,
            &self.placement_book,
            Self::pool_fingerprint(workers),
        )
    }

    fn select_with_observed_owner(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo,
        prefix_hash: u64,
        book: &Mutex<PlacementBook>,
        pool_fingerprint: u64,
    ) -> (Option<usize>, Branch) {
        let key = (pool_fingerprint, prefix_hash);
        let owner = book
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .owner(key, Instant::now());
        if let Some(owner_url) = owner {
            if let Some(index) = self.observed_owner_allowed(workers, info, &owner_url) {
                book.lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .record(key, workers[index].url(), Instant::now());
                return (Some(index), Branch::ObservedPlacement);
            }
            book.lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .entries
                .remove(&key);
        }

        // An unknown or unusable owner follows the original selection path.
        // Record the actual selected virtual URL, including a load fallback.
        let (selection, branch) = self.find_worker_with_load_balance(workers, info, prefix_hash);
        if let Some(index) = selection {
            book.lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .record(key, workers[index].url(), Instant::now());
        }
        (selection, branch)
    }
}

#[async_trait::async_trait]
impl LoadBalancingPolicy for PrefixHashPolicy {
    async fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        let (result, branch) = self.select_worker_impl(workers, info);
        Metrics::record_worker_prefix_hash_policy_branch(branch.as_str());
        result
    }

    fn name(&self) -> &'static str {
        "prefix_hash"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorkerBuilder, HashRing, WorkerType};

    fn placement_policy() -> PrefixHashPolicy {
        let mut policy = PrefixHashPolicy::with_defaults();
        policy.pd_prefill_cache_capacity = true;
        policy.pd_prefill_stage_reservation = true;
        policy.pd_prefill_placement = true;
        policy.pd_prefill_affinity_slack = Some(1);
        policy
    }

    fn prefill_workers(urls: &[&str], model_id: &str) -> Vec<Arc<dyn Worker>> {
        urls.iter()
            .map(|url| {
                Arc::new(
                    BasicWorkerBuilder::new(*url)
                        .worker_type(WorkerType::Prefill {
                            bootstrap_port: Some(9000),
                        })
                        .label("model_id", model_id)
                        .build(),
                ) as Arc<dyn Worker>
            })
            .collect()
    }

    fn waiting_snapshot(count: usize) -> DPLoadSnapshot {
        DPLoadSnapshot {
            age_at_observation: Some(Duration::ZERO),
            num_waiting_reqs: count,
            ..Default::default()
        }
    }

    fn create_workers(urls: &[&str]) -> Vec<Arc<dyn Worker>> {
        urls.iter()
            .map(|url| {
                Arc::new(
                    BasicWorkerBuilder::new(*url)
                        .worker_type(WorkerType::Regular)
                        .build(),
                ) as Arc<dyn Worker>
            })
            .collect()
    }

    fn long_prefix_policy() -> PrefixHashPolicy {
        let mut policy = PrefixHashPolicy::with_defaults();
        policy.pd_affinity_prefix_tokens = Some(8192);
        policy.pd_prefill_placement = false;
        policy.pd_prefill_affinity_slack = None;
        policy
    }

    fn pd_workers(worker_type: WorkerType, count: usize) -> Vec<Arc<dyn Worker>> {
        (0..count)
            .map(|i| {
                Arc::new(
                    BasicWorkerBuilder::new(format!("http://worker:8000@{i}"))
                        .worker_type(worker_type.clone())
                        .label("model_id", "model-a")
                        .build(),
                ) as Arc<dyn Worker>
            })
            .collect()
    }

    #[test]
    fn test_long_prefix_opt_in_validation() {
        let config = PrefixHashConfig::default();
        for value in [
            "",
            "0",
            "256",
            "-1",
            "invalid",
            "65537",
            "9999999999999999999999",
        ] {
            assert_eq!(
                PrefixHashPolicy::parse_pd_affinity_prefix_tokens(value, &config),
                None
            );
        }
        for count in [257, 8192, 65536] {
            assert_eq!(
                PrefixHashPolicy::parse_pd_affinity_prefix_tokens(&count.to_string(), &config),
                Some(count)
            );
        }
    }

    #[test]
    fn test_long_prefix_splits_common_system_and_keeps_growing_conversations() {
        for (kind, count) in [
            (
                WorkerType::Prefill {
                    bootstrap_port: None,
                },
                2,
            ),
            (WorkerType::Decode, 8),
        ] {
            let policy = long_prefix_policy();
            let mut legacy = long_prefix_policy();
            legacy.pd_affinity_prefix_tokens = None;
            let workers = pd_workers(kind, count);
            let ring = Arc::new(HashRing::new(&workers));
            let mut distribution = vec![0; count];
            let mut legacy_owner = None;
            for session in 0..128 {
                // Shared system/tools span far beyond the old 256-token key.
                let mut tokens = vec![7; 8192];
                tokens[4096] = 1000 + session;
                let base = SelectWorkerInfo {
                    tokens: Some(&tokens),
                    hash_ring: Some(ring.clone()),
                    soft_cap: Some(14),
                    ..Default::default()
                };
                let old = legacy.select_worker_impl(&workers, &base).0;
                assert_eq!(*legacy_owner.get_or_insert(old), old);
                let owner = policy.select_worker_impl(&workers, &base).0.unwrap();
                distribution[owner] += 1;
                let mut grown = tokens.clone();
                grown.extend(vec![session; 16384]);
                let followup = SelectWorkerInfo {
                    tokens: Some(&grown),
                    ..base
                };
                assert_eq!(
                    policy.select_worker_impl(&workers, &followup),
                    (Some(owner), Branch::ObservedPlacement)
                );
                assert_eq!(tokens[4096], 1000 + session);
            }
            assert!(distribution.iter().all(|n| *n > 0), "{distribution:?}");
            assert!(
                distribution.iter().all(|n| *n < 128 * 3 / count),
                "{distribution:?}"
            );
        }
    }

    #[test]
    fn test_long_prefix_fallback_survives_filtered_pool_and_idle_ring_owner() {
        for kind in [
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            WorkerType::Decode,
        ] {
            let policy = long_prefix_policy();
            let workers = pd_workers(kind, 2);
            let tokens = vec![9; 8192];
            let base = SelectWorkerInfo {
                tokens: Some(&tokens),
                hash_ring: Some(Arc::new(HashRing::new(&workers))),
                soft_cap: Some(2),
                ..Default::default()
            };
            let initial = policy.select_worker_impl(&workers, &base).0.unwrap();
            let other = 1 - initial;
            let reservations = PlacementMap::from([(workers[initial].url().to_owned(), 2)]);
            let capped = SelectWorkerInfo {
                local_reservations: Some(&reservations),
                ..base.clone()
            };
            // Match HTTP PD's prefiltering before invoking the policy.
            let filtered = vec![workers[other].clone()];
            assert_eq!(policy.select_worker_impl(&filtered, &capped).0, Some(0));
            assert_eq!(
                policy.select_worker_impl(&workers, &base),
                (Some(other), Branch::ObservedPlacement)
            );

            let snapshots =
                PlacementMap::from([(workers[other].url().to_owned(), waiting_snapshot(6))]);
            let busy = SelectWorkerInfo {
                load_snapshots: Some(&snapshots),
                ..base.clone()
            };
            assert_eq!(policy.select_worker_impl(&workers, &busy).0, Some(initial));
            workers[initial].set_healthy(false);
            assert_eq!(policy.select_worker_impl(&workers, &base).0, Some(other));
            workers[initial].set_healthy(true);
            let all_capped = workers.iter().map(|w| (w.url().to_owned(), 2)).collect();
            let capped = SelectWorkerInfo {
                local_reservations: Some(&all_capped),
                ..base
            };
            assert_eq!(policy.select_worker_impl(&workers, &capped).0, None);
        }
    }

    #[test]
    fn test_long_prefix_preserves_capacity_and_post_snapshot_pressure() {
        use crate::core::worker_manager::KVCapacitySnapshot;
        for kind in [
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            WorkerType::Decode,
        ] {
            let policy = long_prefix_policy();
            let workers = pd_workers(kind, 2);
            let tokens = vec![9; 8192];
            let base = SelectWorkerInfo {
                tokens: Some(&tokens),
                hash_ring: Some(Arc::new(HashRing::new(&workers))),
                soft_cap: Some(14),
                estimated_tokens: 9000,
                ..Default::default()
            };
            let owner = policy.select_worker_impl(&workers, &base).0.unwrap();
            let snapshots = PlacementMap::from([(
                workers[owner].url().to_owned(),
                DPLoadSnapshot {
                    age_at_observation: Some(Duration::ZERO),
                    kv_capacity: Some(KVCapacitySnapshot {
                        full_available_tokens: 9000,
                        request_slots_available: Some(1),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            )]);
            let reserved_tokens = PlacementMap::from([(workers[owner].url().to_owned(), 18000)]);
            let reserved_reqs = PlacementMap::from([(workers[owner].url().to_owned(), 1)]);
            let pressure = SelectWorkerInfo {
                load_snapshots: Some(&snapshots),
                post_snapshot_reserved_tokens: Some(&reserved_tokens),
                post_snapshot_reserved_requests: Some(&reserved_reqs),
                ..base.clone()
            };
            assert_eq!(
                policy.select_worker_impl(&workers, &pressure).0,
                Some(1 - owner)
            );
            assert_eq!(
                policy.select_worker_impl(&workers, &base),
                (Some(1 - owner), Branch::ObservedPlacement)
            );
        }
    }

    #[test]
    fn test_long_prefix_short_regular_mixed_and_tokenless_paths_are_unchanged() {
        let p = pd_workers(
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            2,
        );
        let regular = pd_workers(WorkerType::Regular, 2);
        let mixed = vec![p[0].clone(), regular[1].clone()];
        for (workers, len) in [(&p, 8191), (&regular, 8192), (&mixed, 8192)] {
            let policy = long_prefix_policy();
            let mut legacy = long_prefix_policy();
            legacy.pd_affinity_prefix_tokens = None;
            let tokens = vec![5; len];
            let info = SelectWorkerInfo {
                tokens: Some(&tokens),
                hash_ring: Some(Arc::new(HashRing::new(workers))),
                ..Default::default()
            };
            assert_eq!(
                policy.select_worker_impl(workers, &info),
                legacy.select_worker_impl(workers, &info)
            );
            assert!(policy.pd_affinity_book.lock().unwrap().entries.is_empty());
            assert_eq!(
                policy
                    .select_worker_impl(workers, &SelectWorkerInfo::default())
                    .0,
                None
            );
            assert!(policy
                .select_without_prefix(workers, &SelectWorkerInfo::default())
                .is_some());
            assert!(policy.pd_affinity_book.lock().unwrap().entries.is_empty());
        }
    }

    #[test]
    fn test_long_prefix_role_model_membership_and_expiry_namespaces() {
        let policy = long_prefix_policy();
        // Even overlapping URLs in the same ring cannot couple P and D state.
        let p = pd_workers(
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            2,
        );
        let d = pd_workers(WorkerType::Decode, 2);
        let tokens = vec![7; 8192];
        let ring = Arc::new(HashRing::new(&p));
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            hash_ring: Some(ring),
            ..Default::default()
        };
        let p_key = PrefixHashPolicy::pd_pool_fingerprint(&p, &info);
        assert_ne!(p_key, PrefixHashPolicy::pd_pool_fingerprint(&d, &info));
        let reversed = vec![p[1].clone(), p[0].clone()];
        let reordered = SelectWorkerInfo {
            hash_ring: Some(Arc::new(HashRing::new(&reversed))),
            ..info.clone()
        };
        assert_eq!(
            p_key,
            PrefixHashPolicy::pd_pool_fingerprint(&reversed, &reordered)
        );
        let another_model =
            prefill_workers(&["http://worker:8000@0", "http://worker:8000@1"], "model-b");
        let changed = SelectWorkerInfo {
            hash_ring: Some(Arc::new(HashRing::new(&another_model))),
            ..info.clone()
        };
        assert_ne!(
            p_key,
            PrefixHashPolicy::pd_pool_fingerprint(&another_model, &changed)
        );
        let removed = SelectWorkerInfo {
            hash_ring: Some(Arc::new(HashRing::new(&p[..1]))),
            ..info.clone()
        };
        assert_ne!(
            p_key,
            PrefixHashPolicy::pd_pool_fingerprint(&p[..1], &removed)
        );
        policy.select_worker_impl(&p, &info);
        assert_eq!(policy.select_worker_impl(&d, &info).1, Branch::RingHit);
        assert_eq!(policy.pd_affinity_book.lock().unwrap().entries.len(), 2);
        for entry in policy.pd_affinity_book.lock().unwrap().entries.values_mut() {
            entry.observed_at = Instant::now() - PREFILL_PLACEMENT_TTL;
        }
        assert_eq!(policy.select_worker_impl(&p, &info).1, Branch::RingHit);
    }

    #[test]
    fn test_prefix_hash_consistent_routing() {
        let policy = PrefixHashPolicy::with_defaults();
        let workers = create_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        let ring = Arc::new(HashRing::new(&workers));

        // Same tokens should always route to same worker
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            hash_ring: Some(ring.clone()),
            ..Default::default()
        };

        let (first_result, _) = policy.select_worker_impl(&workers, &info);
        let first_idx = first_result.unwrap();

        // Verify consistency
        for _ in 0..10 {
            let (result, _) = policy.select_worker_impl(&workers, &info);
            assert_eq!(result, Some(first_idx));
        }
    }

    #[test]
    fn test_different_prefixes_distribute() {
        let policy = PrefixHashPolicy::with_defaults();
        let workers = create_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        let ring = Arc::new(HashRing::new(&workers));

        let mut distribution = std::collections::HashMap::new();

        // Different token sequences should distribute across workers
        for i in 0..100 {
            let tokens: Vec<u32> = vec![i, i + 1, i + 2, i + 3];
            let info = SelectWorkerInfo {
                tokens: Some(&tokens),
                hash_ring: Some(ring.clone()),
                ..Default::default()
            };

            let (result, _) = policy.select_worker_impl(&workers, &info);
            *distribution.entry(result.unwrap()).or_insert(0) += 1;
        }

        assert!(
            distribution.len() > 1,
            "Should distribute across workers, got {:?}",
            distribution
        );
    }

    #[test]
    fn test_shared_prefix_routes_same() {
        let policy = PrefixHashPolicy::new(PrefixHashConfig {
            prefix_token_count: 5, // Only look at first 5 tokens
            ..Default::default()
        });
        let workers = create_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        let ring = Arc::new(HashRing::new(&workers));

        // Two sequences with same first 5 tokens should route to same worker
        let tokens1: Vec<u32> = vec![1, 2, 3, 4, 5, 100, 200, 300];
        let tokens2: Vec<u32> = vec![1, 2, 3, 4, 5, 999, 888, 777];

        let info1 = SelectWorkerInfo {
            tokens: Some(&tokens1),
            hash_ring: Some(ring.clone()),
            ..Default::default()
        };
        let info2 = SelectWorkerInfo {
            tokens: Some(&tokens2),
            hash_ring: Some(ring.clone()),
            ..Default::default()
        };

        let (result1, _) = policy.select_worker_impl(&workers, &info1);
        let (result2, _) = policy.select_worker_impl(&workers, &info2);

        assert_eq!(result1, result2, "Same prefix should route to same worker");
    }

    #[test]
    fn test_hot_256_token_prefix_stays_affine_until_cap_two_then_balances() {
        let policy = PrefixHashPolicy::with_defaults();
        let workers = create_workers(&[
            "http://w1:8000",
            "http://w2:8000",
            "http://w3:8000",
            "http://w4:8000",
        ]);
        let ring = Arc::new(HashRing::new(&workers));
        let mut first = vec![7; 256];
        first.push(101);
        let mut second = vec![7; 256];
        second.push(202);

        let empty = std::collections::HashMap::new();
        let initial_info = SelectWorkerInfo {
            tokens: Some(&first),
            hash_ring: Some(ring.clone()),
            local_reservations: Some(&empty),
            soft_cap: Some(2),
            ..Default::default()
        };
        let (initial, _) = policy.select_worker_impl(&workers, &initial_info);
        let initial = initial.unwrap();

        let one = std::collections::HashMap::from([(workers[initial].url().to_string(), 1)]);
        let second_info = SelectWorkerInfo {
            tokens: Some(&second),
            hash_ring: Some(ring.clone()),
            local_reservations: Some(&one),
            soft_cap: Some(2),
            ..Default::default()
        };
        assert_eq!(
            policy.select_worker_impl(&workers, &second_info).0,
            Some(initial)
        );

        let two = std::collections::HashMap::from([(workers[initial].url().to_string(), 2)]);
        let capped_info = SelectWorkerInfo {
            tokens: Some(&second),
            hash_ring: Some(ring),
            local_reservations: Some(&two),
            soft_cap: Some(2),
            ..Default::default()
        };
        let (after_cap, _) = policy.select_worker_impl(&workers, &capped_info);
        assert_ne!(after_cap, Some(initial));
    }

    #[test]
    fn test_zero_free_kv_is_soft_pressure_and_never_removes_all_candidates() {
        use crate::core::worker_manager::KVCapacitySnapshot;

        let policy = PrefixHashPolicy::with_defaults();
        let workers = create_workers(&["http://w1:8000", "http://w2:8000"]);
        let ring = Arc::new(HashRing::new(&workers));
        let tokens = vec![42; 300];
        let initial_info = SelectWorkerInfo {
            tokens: Some(&tokens),
            hash_ring: Some(ring.clone()),
            soft_cap: Some(2),
            ..Default::default()
        };
        let initial = policy
            .select_worker_impl(&workers, &initial_info)
            .0
            .unwrap();

        let full = DPLoadSnapshot {
            age_at_observation: Some(Duration::ZERO),
            kv_capacity: Some(KVCapacitySnapshot {
                full_available_tokens: 0,
                full_evictable_tokens: 500_000,
                ..Default::default()
            }),
            ..Default::default()
        };
        let one_full =
            std::collections::HashMap::from([(workers[initial].url().to_string(), full.clone())]);
        let pressured_info = SelectWorkerInfo {
            tokens: Some(&tokens),
            hash_ring: Some(ring.clone()),
            load_snapshots: Some(&one_full),
            estimated_tokens: 128_000,
            soft_cap: Some(2),
            ..Default::default()
        };
        assert_ne!(
            policy.select_worker_impl(&workers, &pressured_info).0,
            Some(initial)
        );

        let all_full = workers
            .iter()
            .map(|worker| (worker.url().to_string(), full.clone()))
            .collect();
        let fallback_info = SelectWorkerInfo {
            tokens: Some(&tokens),
            hash_ring: Some(ring),
            load_snapshots: Some(&all_full),
            estimated_tokens: 128_000,
            soft_cap: Some(2),
            ..Default::default()
        };
        assert!(policy
            .select_worker_impl(&workers, &fallback_info)
            .0
            .is_some());
    }

    #[test]
    fn test_prefill_reclaimable_cache_does_not_look_like_active_work() {
        use crate::core::worker_manager::KVCapacitySnapshot;

        let worker = BasicWorkerBuilder::new("http://p:8000")
            .worker_type(WorkerType::Prefill {
                bootstrap_port: Some(9000),
            })
            .build();
        // These stream guards describe D output, after P has already finished.
        for _ in 0..4 {
            worker.increment_load();
        }
        let snapshot = DPLoadSnapshot {
            age_at_observation: Some(Duration::ZERO),
            kv_capacity: Some(KVCapacitySnapshot {
                full_available_tokens: 0,
                full_evictable_tokens: 139_264,
                request_slots_available: Some(1),
                mamba_available_slots: Some(0),
                mamba_evictable_slots: Some(1),
                ..Default::default()
            }),
            ..Default::default()
        };
        let snapshots = std::collections::HashMap::from([(worker.url().to_string(), snapshot)]);
        let tokens = vec![1; 65_536];
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            load_snapshots: Some(&snapshots),
            estimated_tokens: 100_000,
            ..Default::default()
        };
        let mut policy = PrefixHashPolicy::with_defaults();
        policy.pd_prefill_cache_capacity = true;
        assert_eq!(policy.effective_load(&worker, &info), 0);
        policy.pd_prefill_cache_capacity = false;
        assert!(policy.effective_load(&worker, &info) >= 4);
    }

    #[test]
    fn test_prefill_stage_reservation_covers_stale_observation_without_double_counting() {
        use crate::core::worker_manager::DisaggregationLoadSnapshot;

        let worker = BasicWorkerBuilder::new("http://p:8000")
            .worker_type(WorkerType::Prefill {
                bootstrap_port: Some(9000),
            })
            .build();
        let snapshots = std::collections::HashMap::from([(
            worker.url().to_string(),
            DPLoadSnapshot {
                age_at_observation: Some(Duration::ZERO),
                num_waiting_reqs: 2,
                disaggregation: Some(DisaggregationLoadSnapshot {
                    prefill_bootstrap_queue_reqs: 1,
                    ..Default::default()
                }),
                ..Default::default()
            },
        )]);
        let reservations = std::collections::HashMap::from([(worker.url().to_string(), 4)]);
        let info = SelectWorkerInfo {
            load_snapshots: Some(&snapshots),
            local_reservations: Some(&reservations),
            ..Default::default()
        };
        let mut policy = PrefixHashPolicy::with_defaults();
        policy.pd_prefill_cache_capacity = true;
        policy.pd_prefill_stage_reservation = true;
        assert_eq!(policy.effective_load(&worker, &info), 4);

        // SRT can know about more queued requests than the local gateway.
        let one = std::collections::HashMap::from([(worker.url().to_string(), 1)]);
        let info = SelectWorkerInfo {
            local_reservations: Some(&one),
            ..info
        };
        assert_eq!(policy.effective_load(&worker, &info), 3);
    }

    #[test]
    fn test_prefill_stage_load_does_not_require_reclaimable_capacity_opt_in() {
        use crate::core::worker_manager::{DisaggregationLoadSnapshot, KVCapacitySnapshot};

        let worker = BasicWorkerBuilder::new("http://p:8000")
            .worker_type(WorkerType::Prefill {
                bootstrap_port: Some(9000),
            })
            .build();
        for _ in 0..8 {
            worker.increment_load(); // P is finished; D is still streaming.
        }
        let snapshots = std::collections::HashMap::from([(
            worker.url().to_string(),
            DPLoadSnapshot {
                age_at_observation: Some(Duration::ZERO),
                num_waiting_reqs: 1,
                disaggregation: Some(DisaggregationLoadSnapshot {
                    prefill_inflight_queue_reqs: 1,
                    ..Default::default()
                }),
                kv_capacity: Some(KVCapacitySnapshot {
                    full_available_tokens: 100_000,
                    request_slots_available: Some(4),
                    ..Default::default()
                }),
                ..Default::default()
            },
        )]);
        let reservations = std::collections::HashMap::from([(worker.url().to_string(), 1)]);
        let info = SelectWorkerInfo {
            load_snapshots: Some(&snapshots),
            local_reservations: Some(&reservations),
            estimated_tokens: 10_000,
            ..Default::default()
        };
        let mut policy = PrefixHashPolicy::with_defaults();
        policy.pd_prefill_cache_capacity = false;
        policy.pd_prefill_stage_reservation = true;
        assert_eq!(policy.effective_load(&worker, &info), 2);

        // Stage feedback must not silently enable reclaimable KV capacity.
        let pressured = SelectWorkerInfo {
            estimated_tokens: 200_000,
            ..info
        };
        assert_eq!(policy.effective_load(&worker, &pressured), 3);
        policy.pd_prefill_stage_reservation = false;
        assert!(policy.effective_load(&worker, &pressured) >= 8);
    }

    #[test]
    fn test_prefill_work_pressure_observes_chunked_prompt_and_stays_prefill_only() {
        let prefill = BasicWorkerBuilder::new("http://p:8000")
            .worker_type(WorkerType::Prefill {
                bootstrap_port: Some(9000),
            })
            .build();
        let decode = BasicWorkerBuilder::new("http://d:8000")
            .worker_type(WorkerType::Decode)
            .build();
        let snapshot = DPLoadSnapshot {
            age_at_observation: Some(Duration::ZERO),
            num_waiting_uncached_tokens: 65_537,
            ..Default::default()
        };
        let snapshots = std::collections::HashMap::from([
            (prefill.url().to_string(), snapshot.clone()),
            (decode.url().to_string(), snapshot),
        ]);
        let info = SelectWorkerInfo {
            load_snapshots: Some(&snapshots),
            ..Default::default()
        };
        let mut policy = PrefixHashPolicy::with_defaults();
        policy.pd_prefill_cache_capacity = true;
        policy.pd_prefill_stage_reservation = true;
        policy.pd_prefill_work_tokens = Some(16_384);
        assert_eq!(policy.effective_load(&prefill, &info), 5);
        assert_eq!(policy.effective_load(&decode, &info), 0);
        policy.pd_prefill_work_tokens = None;
        assert_eq!(policy.effective_load(&prefill, &info), 0);
    }

    #[test]
    fn test_prefill_does_not_reserve_decode_output_but_decode_still_does() {
        use crate::core::worker_manager::KVCapacitySnapshot;

        let prefill = BasicWorkerBuilder::new("http://p:8000")
            .worker_type(WorkerType::Prefill {
                bootstrap_port: Some(9000),
            })
            .build();
        let decode = BasicWorkerBuilder::new("http://d:8000")
            .worker_type(WorkerType::Decode)
            .build();
        let snapshot = DPLoadSnapshot {
            age_at_observation: Some(Duration::ZERO),
            kv_capacity: Some(KVCapacitySnapshot {
                full_available_tokens: 20_000,
                full_evictable_tokens: 100_000,
                request_slots_available: Some(1),
                ..Default::default()
            }),
            ..Default::default()
        };
        let snapshots = std::collections::HashMap::from([
            (prefill.url().to_string(), snapshot.clone()),
            (decode.url().to_string(), snapshot),
        ]);
        let tokens = vec![1; 16_384];
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            load_snapshots: Some(&snapshots),
            estimated_tokens: 80_000,
            ..Default::default()
        };
        let mut policy = PrefixHashPolicy::with_defaults();
        policy.pd_prefill_cache_capacity = true;
        assert_eq!(policy.effective_load(&prefill, &info), 0);
        assert!(policy.effective_load(&decode, &info) > 0);
    }

    #[test]
    fn test_prefill_stale_snapshot_preserves_local_load_fallback() {
        let worker = BasicWorkerBuilder::new("http://p:8000")
            .worker_type(WorkerType::Prefill {
                bootstrap_port: Some(9000),
            })
            .build();
        for _ in 0..3 {
            worker.increment_load();
        }
        let snapshots = std::collections::HashMap::from([(
            worker.url().to_string(),
            DPLoadSnapshot {
                age_at_observation: Some(Duration::from_secs(4)),
                ..Default::default()
            },
        )]);
        let info = SelectWorkerInfo {
            load_snapshots: Some(&snapshots),
            ..Default::default()
        };
        let mut policy = PrefixHashPolicy::with_defaults();
        policy.pd_prefill_cache_capacity = true;
        policy.pd_prefill_stage_reservation = true;
        assert_eq!(policy.effective_load(&worker, &info), 3);
    }

    #[test]
    fn test_prefill_affinity_allows_bounded_queue_but_balances_large_backlog() {
        let mut policy = PrefixHashPolicy::with_defaults();
        policy.pd_prefill_cache_capacity = true;
        policy.pd_prefill_affinity_slack = Some(2);
        let workers: Vec<Arc<dyn Worker>> = (0..2)
            .map(|i| {
                Arc::new(
                    BasicWorkerBuilder::new(format!("http://p{i}:8000"))
                        .worker_type(WorkerType::Prefill {
                            bootstrap_port: Some(9000),
                        })
                        .build(),
                ) as Arc<dyn Worker>
            })
            .collect();
        let ring = Arc::new(HashRing::new(&workers));
        let tokens = vec![2; 300];
        let base = SelectWorkerInfo {
            tokens: Some(&tokens),
            hash_ring: Some(ring.clone()),
            ..Default::default()
        };
        let target = policy.select_worker_impl(&workers, &base).0.unwrap();
        let mut snapshots = std::collections::HashMap::from([(
            workers[target].url().to_string(),
            DPLoadSnapshot {
                age_at_observation: Some(Duration::ZERO),
                num_waiting_reqs: 2,
                ..Default::default()
            },
        )]);
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            hash_ring: Some(ring.clone()),
            load_snapshots: Some(&snapshots),
            ..Default::default()
        };
        assert_eq!(policy.select_worker_impl(&workers, &info).0, Some(target));
        snapshots
            .get_mut(workers[target].url())
            .unwrap()
            .num_waiting_reqs = 6;
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            hash_ring: Some(ring),
            load_snapshots: Some(&snapshots),
            ..Default::default()
        };
        assert_ne!(policy.select_worker_impl(&workers, &info).0, Some(target));
    }

    #[test]
    fn test_no_tokens_returns_none() {
        let policy = PrefixHashPolicy::with_defaults();
        let workers = create_workers(&["http://w1:8000"]);
        let ring = Arc::new(HashRing::new(&workers));

        // Empty tokens
        let tokens: Vec<u32> = vec![];
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            hash_ring: Some(ring.clone()),
            ..Default::default()
        };

        let (result, branch) = policy.select_worker_impl(&workers, &info);
        assert_eq!(result, None);
        assert_eq!(branch, Branch::NoTokens);

        // No tokens field
        let info_no_tokens = SelectWorkerInfo {
            tokens: None,
            hash_ring: Some(ring),
            ..Default::default()
        };

        let (result2, branch2) = policy.select_worker_impl(&workers, &info_no_tokens);
        assert_eq!(result2, None);
        assert_eq!(branch2, Branch::NoTokens);
    }

    #[test]
    fn test_no_healthy_workers() {
        let policy = PrefixHashPolicy::with_defaults();
        let workers = create_workers(&["http://w1:8000"]);
        workers[0].set_healthy(false);

        let ring = Arc::new(HashRing::new(&workers));
        let tokens: Vec<u32> = vec![1, 2, 3];
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            hash_ring: Some(ring),
            ..Default::default()
        };

        let (result, branch) = policy.select_worker_impl(&workers, &info);
        assert_eq!(result, None);
        assert_eq!(branch, Branch::NoHealthyWorkers);
    }

    #[test]
    fn test_load_ok_calculation() {
        let policy = PrefixHashPolicy::new(PrefixHashConfig {
            load_factor: 1.25,
            ..Default::default()
        });

        // Total load 100, 4 workers -> avg 25, threshold 31.25
        assert!(policy.load_ok(30, 100, 4)); // 30 <= 31.25
        assert!(!policy.load_ok(35, 100, 4)); // 35 > 31.25

        // Edge cases
        assert!(policy.load_ok(0, 0, 4)); // No load = OK
        assert!(policy.load_ok(100, 0, 0)); // No workers = OK (shouldn't happen)
    }

    #[test]
    fn test_policy_name() {
        let policy = PrefixHashPolicy::with_defaults();
        assert_eq!(policy.name(), "prefix_hash");
    }

    #[test]
    fn test_observed_fallback_is_reused_and_overload_refreshes_owner() {
        let policy = placement_policy();
        let workers = prefill_workers(&["http://p0:8000", "http://p1:8000"], "model-a");
        let ring = Arc::new(HashRing::new(&workers));
        let tokens = vec![17; 300];
        let base = SelectWorkerInfo {
            tokens: Some(&tokens),
            hash_ring: Some(ring.clone()),
            ..Default::default()
        };
        let ring_owner = policy.select_worker_impl(&workers, &base).0.unwrap();
        let fallback = 1 - ring_owner;
        let key = (
            PrefixHashPolicy::pool_fingerprint(&workers),
            policy.compute_prefix_hash(&tokens),
        );

        let busy_ring =
            PlacementMap::from([(workers[ring_owner].url().to_owned(), waiting_snapshot(6))]);
        let busy_ring_info = SelectWorkerInfo {
            load_snapshots: Some(&busy_ring),
            ..base.clone()
        };
        assert_eq!(
            policy.select_worker_impl(&workers, &busy_ring_info),
            (Some(fallback), Branch::LoadBalanceWalk)
        );
        assert_eq!(
            policy.select_worker_impl(&workers, &base),
            (Some(fallback), Branch::ObservedPlacement)
        );

        let busy_fallback =
            PlacementMap::from([(workers[fallback].url().to_owned(), waiting_snapshot(6))]);
        let busy_fallback_info = SelectWorkerInfo {
            load_snapshots: Some(&busy_fallback),
            ..base.clone()
        };
        assert_eq!(
            policy.select_worker_impl(&workers, &busy_fallback_info).0,
            Some(ring_owner)
        );
        assert_eq!(
            policy
                .placement_book
                .lock()
                .unwrap()
                .owner(key, Instant::now()),
            Some(workers[ring_owner].url().to_owned())
        );
    }

    #[test]
    fn test_placement_opt_in_and_non_prefill_selection_are_unchanged() {
        let workers = prefill_workers(&["http://p0:8000", "http://p1:8000"], "model-a");
        let ring = Arc::new(HashRing::new(&workers));
        let tokens = vec![18; 300];
        let base = SelectWorkerInfo {
            tokens: Some(&tokens),
            hash_ring: Some(ring.clone()),
            ..Default::default()
        };
        for disabled_gate in 0..3 {
            let mut policy = placement_policy();
            match disabled_gate {
                0 => policy.pd_prefill_placement = false,
                1 => policy.pd_prefill_cache_capacity = false,
                _ => policy.pd_prefill_stage_reservation = false,
            }
            let ring_owner = policy.select_worker_impl(&workers, &base).0.unwrap();
            let busy =
                PlacementMap::from([(workers[ring_owner].url().to_owned(), waiting_snapshot(6))]);
            let busy_info = SelectWorkerInfo {
                load_snapshots: Some(&busy),
                ..base.clone()
            };
            assert_ne!(
                policy.select_worker_impl(&workers, &busy_info).0,
                Some(ring_owner)
            );
            assert_eq!(
                policy.select_worker_impl(&workers, &base).0,
                Some(ring_owner)
            );
            assert!(policy.placement_book.lock().unwrap().entries.is_empty());
        }

        for worker_type in [WorkerType::Decode, WorkerType::Regular] {
            let policy = placement_policy();
            let workers: Vec<Arc<dyn Worker>> = ["http://w0:8000", "http://w1:8000"]
                .iter()
                .map(|url| {
                    Arc::new(
                        BasicWorkerBuilder::new(*url)
                            .worker_type(worker_type.clone())
                            .build(),
                    ) as Arc<dyn Worker>
                })
                .collect();
            let info = SelectWorkerInfo {
                tokens: Some(&tokens),
                hash_ring: Some(Arc::new(HashRing::new(&workers))),
                ..Default::default()
            };
            let ring_owner = policy.select_worker_impl(&workers, &info).0.unwrap();
            let busy =
                PlacementMap::from([(workers[ring_owner].url().to_owned(), waiting_snapshot(6))]);
            let busy_info = SelectWorkerInfo {
                load_snapshots: Some(&busy),
                ..info.clone()
            };
            assert_ne!(
                policy.select_worker_impl(&workers, &busy_info).0,
                Some(ring_owner)
            );
            assert_eq!(
                policy.select_worker_impl(&workers, &info).0,
                Some(ring_owner)
            );
            assert!(policy.placement_book.lock().unwrap().entries.is_empty());
        }
    }

    #[test]
    fn test_unavailable_or_capped_placement_owner_is_replaced() {
        let policy = placement_policy();
        let workers = prefill_workers(&["http://p0:8000", "http://p1:8000"], "model-a");
        let tokens = vec![19; 300];
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            hash_ring: Some(Arc::new(HashRing::new(&workers))),
            ..Default::default()
        };
        let ring_owner = policy.select_worker_impl(&workers, &info).0.unwrap();
        let other = 1 - ring_owner;
        let key = (
            PrefixHashPolicy::pool_fingerprint(&workers),
            policy.compute_prefix_hash(&tokens),
        );
        policy
            .placement_book
            .lock()
            .unwrap()
            .record(key, workers[other].url(), Instant::now());

        workers[other].set_healthy(false);
        assert_eq!(
            policy.select_worker_impl(&workers, &info).0,
            Some(ring_owner)
        );
        workers[other].set_healthy(true);
        assert_eq!(
            policy
                .placement_book
                .lock()
                .unwrap()
                .owner(key, Instant::now()),
            Some(workers[ring_owner].url().to_owned())
        );

        policy
            .placement_book
            .lock()
            .unwrap()
            .record(key, workers[other].url(), Instant::now());
        let reserved = PlacementMap::from([(workers[other].url().to_owned(), 2)]);
        let capped_info = SelectWorkerInfo {
            local_reservations: Some(&reserved),
            soft_cap: Some(2),
            ..info
        };
        assert_eq!(
            policy.select_worker_impl(&workers, &capped_info).0,
            Some(ring_owner)
        );
        assert_eq!(
            policy
                .placement_book
                .lock()
                .unwrap()
                .owner(key, Instant::now()),
            Some(workers[ring_owner].url().to_owned())
        );
    }

    #[test]
    fn test_placement_ttl_pool_namespace_and_hard_cap() {
        let policy = placement_policy();
        let workers = prefill_workers(&["http://p0:8000", "http://p1:8000"], "model-a");
        let tokens = vec![20; 300];
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            hash_ring: Some(Arc::new(HashRing::new(&workers))),
            ..Default::default()
        };
        let ring_owner = policy.select_worker_impl(&workers, &info).0.unwrap();
        let key = (
            PrefixHashPolicy::pool_fingerprint(&workers),
            policy.compute_prefix_hash(&tokens),
        );
        {
            let mut book = policy.placement_book.lock().unwrap();
            let entry = book.entries.get_mut(&key).unwrap();
            entry.owner_url = workers[1 - ring_owner].url().to_owned();
            entry.observed_at = Instant::now() - PREFILL_PLACEMENT_TTL;
        }
        assert_eq!(
            policy.select_worker_impl(&workers, &info),
            (Some(ring_owner), Branch::RingHit)
        );

        let reversed = vec![workers[1].clone(), workers[0].clone()];
        assert_eq!(
            PrefixHashPolicy::pool_fingerprint(&workers),
            PrefixHashPolicy::pool_fingerprint(&reversed)
        );
        let different_model = prefill_workers(&["http://p0:8000", "http://p1:8000"], "model-b");
        let different_pool = prefill_workers(&["http://p0:8000", "http://p2:8000"], "model-a");
        assert_ne!(key.0, PrefixHashPolicy::pool_fingerprint(&different_model));
        assert_ne!(key.0, PrefixHashPolicy::pool_fingerprint(&different_pool));
        policy.placement_book.lock().unwrap().record(
            key,
            workers[1 - ring_owner].url(),
            Instant::now(),
        );
        for other_workers in [&different_model, &different_pool] {
            let other_info = SelectWorkerInfo {
                tokens: Some(&tokens),
                hash_ring: Some(Arc::new(HashRing::new(other_workers))),
                ..Default::default()
            };
            assert_eq!(
                policy.select_worker_impl(other_workers, &other_info).1,
                Branch::RingHit
            );
        }

        let now = Instant::now();
        let mut book = PlacementBook::default();
        for prefix in 0..PREFILL_PLACEMENT_CAPACITY as u64 {
            book.record((key.0, prefix), "http://p0:8000", now);
        }
        for _ in 0..1000 {
            book.record((key.0, 0), "http://p1:8000", now);
        }
        assert_eq!(book.entries.len(), PREFILL_PLACEMENT_CAPACITY);
        book.record(
            (key.0, PREFILL_PLACEMENT_CAPACITY as u64),
            "http://p0:8000",
            now,
        );
        assert_eq!(book.entries.len(), PREFILL_PLACEMENT_CAPACITY);
        assert!(!book.entries.contains_key(&(key.0, 1)));
        assert_eq!(
            book.owner((key.0, 0), now).as_deref(),
            Some("http://p1:8000")
        );
    }
}
