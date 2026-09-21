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

use std::{sync::Arc, time::Duration};

use super::{LoadBalancingPolicy, SelectWorkerInfo};
use crate::{
    core::{DPLoadSnapshot, Worker, WorkerType},
    observability::metrics::Metrics,
};

const LOAD_SNAPSHOT_MAX_AGE: Duration = Duration::from_secs(3);

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
}

impl PrefixHashPolicy {
    /// Create a new PrefixHashPolicy with the given configuration
    pub fn new(config: PrefixHashConfig) -> Self {
        Self { config }
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

    fn effective_load(worker: &dyn Worker, info: &SelectWorkerInfo<'_>) -> usize {
        let local = worker.load().max(Self::local_reservations(worker, info));
        let Some(snapshot) = Self::snapshot_for(worker, info) else {
            return local;
        };
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
            let demand = info.estimated_tokens.saturating_add(post_snapshot_tokens);
            let token_pressure = if demand > capacity.full_available_tokens {
                let deficit = demand - capacity.full_available_tokens;
                1usize.saturating_add(deficit / info.estimated_tokens.max(1))
            } else {
                0
            };
            let slot_pressure = usize::from(
                capacity
                    .request_slots_available
                    .is_some_and(|slots| slots <= post_snapshot_requests),
            );
            token_pressure.saturating_add(slot_pressure)
        });

        local
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
            .map(|(_, w)| Self::effective_load(w.as_ref(), info))
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
                    let worker_load = Self::effective_load(worker.as_ref(), info);

                    // With bounded affinity, keep the first two requests of a
                    // hot prefix together unless the selected rank has more
                    // than one unit of extra stage/capacity pressure. Once the
                    // local cap is reached that rank is absent from the healthy
                    // map and the ring walks to another DP rank.
                    if info.soft_cap.is_some() {
                        let least_load = healthy_workers
                            .iter()
                            .map(|(_, worker)| Self::effective_load(worker.as_ref(), info))
                            .min()
                            .unwrap_or(worker_load);
                        if worker_load <= least_load.saturating_add(1) {
                            return (Some(idx), Branch::RingHit);
                        }
                    }

                    // Check if initial worker has acceptable load
                    if self.load_ok(worker_load, total_load, num_workers) {
                        return (Some(idx), Branch::RingHit);
                    }

                    // Initial worker overloaded, find least loaded healthy worker
                    // This is a simpler approach than walking the ring
                    let least_loaded = healthy_workers
                        .iter()
                        .filter(|(_, w)| {
                            self.load_ok(
                                Self::effective_load(w.as_ref(), info),
                                total_load,
                                num_workers,
                            )
                        })
                        .min_by_key(|(_, w)| Self::effective_load(w.as_ref(), info));

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
            .min_by_key(|(_, w)| Self::effective_load(w.as_ref(), info))
            .map(|(idx, _)| *idx);

        (least_loaded, Branch::FallbackLeastLoad)
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

        // Compute prefix hash
        let prefix_hash = self.compute_prefix_hash(tokens);

        // Find worker using ring with load balancing
        self.find_worker_with_load_balance(workers, info, prefix_hash)
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
}
