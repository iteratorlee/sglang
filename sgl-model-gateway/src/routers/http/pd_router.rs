use std::{
    borrow::Cow,
    collections::HashMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::Instant,
};

use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
};
use dashmap::DashMap;
use futures_util::StreamExt;
use reqwest::Client;
use serde::Serialize;
use serde_json::{json, Value};
use tokio::sync::watch;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{debug, error, warn};

use super::pd_types::api_path;
use crate::{
    config::types::RetryConfig,
    core::{
        is_retryable_status, pd_token_affinity_enabled, DPLoadSnapshot, HashRing, RetryExecutor,
        Worker, WorkerLoadGuard, WorkerRegistry, WorkerType, UNKNOWN_MODEL_ID,
    },
    observability::{
        events::{self, Event},
        metrics::{bool_to_static_str, metrics_labels, Metrics},
        otel_trace::inject_trace_context_http,
    },
    policies::{LoadBalancingPolicy, PolicyRegistry, SelectWorkerInfo},
    protocols::{
        chat::ChatCompletionRequest,
        classify::ClassifyRequest,
        common::{GenerationRequest, InputIds, StringOrArray},
        completion::CompletionRequest,
        embedding::EmbeddingRequest,
        generate::GenerateRequest,
        rerank::RerankRequest,
    },
    routers::{
        error,
        grpc::utils::{error_type_from_status, route_to_endpoint},
        header_utils,
        sse::SseDoneDetector,
        streaming_utils::BreakerTrackedStream,
        RouterTrait,
    },
};

#[derive(Debug)]
pub struct PDRouter {
    pub worker_registry: Arc<WorkerRegistry>,
    pub policy_registry: Arc<PolicyRegistry>,
    pub client: Client,
    pub retry_config: RetryConfig,
    pub api_key: Option<String>,
    pub enable_igw: bool,
    pd_token_affinity: bool,
    pd_prefill_stage_reservation: bool,
    pd_decode_inflight_cap: usize,
    load_snapshots: watch::Receiver<HashMap<String, DPLoadSnapshot>>,
    reservations: Arc<PDReservationTable>,
}

#[derive(Debug)]
struct ReservationRecord {
    id: u64,
    created_at: Instant,
    tokens: usize,
}

#[derive(Debug, Default)]
struct PDReservationTable {
    next_id: AtomicU64,
    entries: DashMap<String, Arc<Mutex<Vec<ReservationRecord>>>>,
}

#[derive(Debug)]
struct PDReservationLease {
    entry: Arc<Mutex<Vec<ReservationRecord>>>,
    id: u64,
}

#[derive(Debug, Default)]
struct PDReservationPair {
    prefill: Option<PDReservationLease>,
    _decode: Option<PDReservationLease>,
}

// The P load counter and optional local reservation have exactly the same
// lifetime: from pair selection until the P response body has been consumed.
struct PrefillStage {
    _lease: Option<PDReservationLease>,
    _load: WorkerLoadGuard,
}

impl PrefillStage {
    fn new(
        lease: Option<PDReservationLease>,
        worker: Arc<dyn Worker>,
        headers: Option<&HeaderMap>,
    ) -> Self {
        Self {
            _lease: lease,
            _load: WorkerLoadGuard::new(worker, headers),
        }
    }
}

impl Drop for PDReservationLease {
    fn drop(&mut self) {
        let mut records = self.entry.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(index) = records.iter().position(|record| record.id == self.id) {
            records.swap_remove(index);
        }
    }
}

impl PDReservationTable {
    fn try_reserve(
        &self,
        worker_url: &str,
        tokens: usize,
        soft_cap: Option<usize>,
    ) -> Option<PDReservationLease> {
        let entry = self
            .entries
            .entry(worker_url.to_string())
            .or_insert_with(|| Arc::new(Mutex::new(Vec::new())))
            .clone();
        let mut records = entry.lock().unwrap_or_else(|e| e.into_inner());
        if soft_cap.is_some_and(|cap| records.len() >= cap) {
            return None;
        }

        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        records.push(ReservationRecord {
            id,
            created_at: Instant::now(),
            tokens,
        });
        drop(records);
        Some(PDReservationLease { entry, id })
    }

    fn views(
        &self,
        snapshots: &HashMap<String, DPLoadSnapshot>,
    ) -> (
        HashMap<String, usize>,
        HashMap<String, usize>,
        HashMap<String, usize>,
    ) {
        let mut active = HashMap::new();
        let mut post_snapshot_tokens = HashMap::new();
        let mut post_snapshot_requests = HashMap::new();
        for item in self.entries.iter() {
            let records = item.value().lock().unwrap_or_else(|e| e.into_inner());
            if records.is_empty() {
                continue;
            }
            active.insert(item.key().clone(), records.len());
            let sampled_at = snapshots
                .get(item.key())
                .and_then(DPLoadSnapshot::sampled_at_estimate);
            let tokens = records
                .iter()
                .filter(|record| sampled_at.is_none_or(|sampled_at| record.created_at > sampled_at))
                .map(|record| record.tokens)
                .sum();
            let requests = records
                .iter()
                .filter(|record| sampled_at.is_none_or(|sampled_at| record.created_at > sampled_at))
                .count();
            post_snapshot_tokens.insert(item.key().clone(), tokens);
            post_snapshot_requests.insert(item.key().clone(), requests);
        }
        (active, post_snapshot_tokens, post_snapshot_requests)
    }
}

const PD_DECODE_INFLIGHT_CAP_ENV: &str = "SGLANG_GATEWAY_PD_DECODE_INFLIGHT_CAP";
const PD_PREFILL_STAGE_RESERVATION_ENV: &str = "SGLANG_GATEWAY_PD_PREFILL_STAGE_RESERVATION";

fn prefill_stage_reservation_from_env() -> bool {
    prefill_stage_reservation_enabled(
        std::env::var(PD_PREFILL_STAGE_RESERVATION_ENV)
            .ok()
            .as_deref(),
    )
}

fn prefill_stage_reservation_enabled(value: Option<&str>) -> bool {
    value == Some("1")
}

// Input token IDs are exact. Text endpoints have no token IDs at the gateway;
// estimate four UTF-8 bytes per token and reserve at least one input token.
fn prefill_input_tokens(input_ids: Option<usize>, text: Option<&str>) -> usize {
    input_ids
        .filter(|count| *count > 0)
        .unwrap_or_else(|| text.map_or(1, |text| text.len().div_ceil(4).max(1)))
}

fn parse_decode_inflight_cap(raw: &str) -> Result<usize, String> {
    raw.parse::<usize>()
        .ok()
        .filter(|cap| *cap > 0)
        .ok_or_else(|| format!("{PD_DECODE_INFLIGHT_CAP_ENV} must be a positive integer"))
}

fn decode_inflight_cap_from_env() -> Result<usize, String> {
    match std::env::var(PD_DECODE_INFLIGHT_CAP_ENV) {
        Ok(raw) => parse_decode_inflight_cap(&raw),
        Err(std::env::VarError::NotPresent) => Ok(2),
        Err(error) => Err(format!("{PD_DECODE_INFLIGHT_CAP_ENV}: {error}")),
    }
}

// Native /generate requests often omit model, while a non-IGW PD router may
// still serve exactly one model. Use that model's ring for token-ID affinity;
// an absent ring makes prefix_hash fall back to the same first idle P worker.
fn hash_ring_model_for_pd<'a>(
    effective_model_id: Option<&'a str>,
    mut prefill_models: impl Iterator<Item = &'a str>,
) -> Option<&'a str> {
    if effective_model_id.is_some() {
        return effective_model_id;
    }
    let first = prefill_models.next()?;
    prefill_models.all(|model| model == first).then_some(first)
}

struct PreparedWorkerRequest<'a> {
    endpoint_url: String,
    body: Cow<'a, Value>,
}

#[derive(Clone)]
struct PDRequestContext<'a> {
    route: &'static str,
    batch_size: Option<usize>,
    is_stream: bool,
    return_logprob: bool,
    request_text: Option<String>,
    tokens: Option<Arc<[u32]>>,
    estimated_tokens: usize,
    prefill_tokens: usize,
    model_id: Option<&'a str>,
    headers: Option<HeaderMap>,
}

/// Marker placed on a `Response` by paths inside
/// `execute_dual_dispatch_internal` that have already recorded prefill and
/// decode breaker outcomes against the workers' actual per-side results
/// (rather than the final response status). The outer dispatcher reads this
/// and skips its own status-based `record_outcome` calls so a decode-only
/// transport failure can't be misattributed to a healthy prefill.
#[derive(Clone, Copy)]
struct BreakerOutcomesRecorded;

impl PDRouter {
    fn worker_endpoint_url(worker: &dyn Worker, endpoint: &str) -> String {
        api_path(worker.base_url(), endpoint)
    }

    async fn proxy_to_first_prefill_worker(
        &self,
        endpoint: &str,
        headers: Option<Vec<(String, String)>>,
    ) -> Response {
        let workers = self.worker_registry.get_prefill_workers();

        if let Some(worker) = workers.first() {
            self.proxy_to_worker(worker.as_ref(), endpoint, headers)
                .await
        } else {
            error::service_unavailable("no_prefill_servers", "No prefill servers available")
        }
    }

    async fn proxy_to_worker(
        &self,
        worker: &dyn Worker,
        endpoint: &str,
        headers: Option<Vec<(String, String)>>,
    ) -> Response {
        let url = Self::worker_endpoint_url(worker, endpoint);
        let mut request_builder = self.client.get(&url);

        if let Some(headers) = headers {
            for (name, value) in headers {
                request_builder = request_builder.header(name, value);
            }
        }

        match request_builder.send().await {
            Ok(res) if res.status().is_success() => {
                let response_headers = header_utils::preserve_response_headers(res.headers());

                match res.bytes().await {
                    Ok(body) => {
                        let mut response = Response::new(Body::from(body));
                        *response.status_mut() = StatusCode::OK;
                        *response.headers_mut() = response_headers;
                        response
                    }
                    Err(e) => {
                        error!("Failed to read response body: {}", e);
                        error::internal_error(
                            "read_response_body_failed",
                            format!("Failed to read response body: {}", e),
                        )
                    }
                }
            }
            Ok(res) => {
                let status = StatusCode::from_u16(res.status().as_u16())
                    .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                // Use the status code to determine which error function to use
                match status {
                    StatusCode::BAD_REQUEST => error::bad_request(
                        "server_bad_request",
                        format!("Server returned status: {}", res.status()),
                    ),
                    StatusCode::NOT_FOUND => error::not_found(
                        "server_not_found",
                        format!("Server returned status: {}", res.status()),
                    ),
                    StatusCode::INTERNAL_SERVER_ERROR => error::internal_error(
                        "server_internal_error",
                        format!("Server returned status: {}", res.status()),
                    ),
                    StatusCode::SERVICE_UNAVAILABLE => error::service_unavailable(
                        "server_unavailable",
                        format!("Server returned status: {}", res.status()),
                    ),
                    StatusCode::BAD_GATEWAY => error::bad_gateway(
                        "server_bad_gateway",
                        format!("Server returned status: {}", res.status()),
                    ),
                    _ => error::internal_error(
                        "server_error",
                        format!("Server returned status: {}", res.status()),
                    ),
                }
            }
            Err(e) => {
                error!("Failed to proxy request server: {}", e);
                error::internal_error(
                    "proxy_request_failed",
                    format!("Failed to proxy request: {}", e),
                )
            }
        }
    }

    pub async fn new(ctx: &Arc<crate::app_context::AppContext>) -> Result<Self, String> {
        let load_snapshots = if let Some(monitor) = ctx.load_monitor.as_ref() {
            monitor.subscribe_snapshots()
        } else {
            let (_tx, rx) = watch::channel(HashMap::new());
            rx
        };
        Ok(PDRouter {
            worker_registry: Arc::clone(&ctx.worker_registry),
            policy_registry: Arc::clone(&ctx.policy_registry),
            client: ctx.client.clone(),
            retry_config: ctx.router_config.effective_retry_config(),
            api_key: ctx.router_config.api_key.clone(),
            enable_igw: ctx.router_config.enable_igw,
            pd_token_affinity: pd_token_affinity_enabled(),
            pd_prefill_stage_reservation: prefill_stage_reservation_from_env(),
            pd_decode_inflight_cap: decode_inflight_cap_from_env()?,
            load_snapshots,
            reservations: Arc::new(PDReservationTable::default()),
        })
    }

    fn handle_server_selection_error(error: String) -> Response {
        error!("Failed to select PD pair error={}", error);
        error::service_unavailable(
            "server_selection_failed",
            format!("No available servers: {}", error),
        )
    }

    fn handle_serialization_error(error: impl std::fmt::Display) -> Response {
        error!("Failed to serialize request error={}", error);
        error::internal_error("serialization_failed", "Failed to serialize request")
    }

    fn get_generate_batch_size(req: &GenerateRequest) -> Option<usize> {
        // GenerateRequest doesn't support batch via arrays, only via input_ids
        if let Some(InputIds::Batch(batches)) = &req.input_ids {
            if !batches.is_empty() {
                return Some(batches.len());
            }
        }
        None
    }

    fn get_generate_affinity_tokens(req: &GenerateRequest) -> Result<Arc<[u32]>, &'static str> {
        if req.text.is_some() || req.input_embeds.is_some() {
            return Err("Native PD token affinity requires input_ids as the only prompt input");
        }

        let ids = match req.input_ids.as_ref() {
            Some(InputIds::Single(ids)) => ids,
            Some(InputIds::Batch(_)) => {
                return Err(
                    "Native PD token affinity supports only non-empty single input_ids; batched input_ids expand to multiple PD requests",
                );
            }
            None => {
                return Err("Native PD token affinity requires non-empty single input_ids");
            }
        };
        if ids.is_empty() {
            return Err("Native PD token affinity requires non-empty single input_ids");
        }
        if req
            .sampling_params
            .as_ref()
            .and_then(|params| params.n)
            .is_some_and(|n| n > 1)
        {
            return Err(
                "Native PD token affinity requires sampling_params.n <= 1; n > 1 expands to multiple PD requests",
            );
        }

        ids.iter()
            .copied()
            .map(u32::try_from)
            .collect::<Result<Vec<_>, _>>()
            .map(Arc::from)
            .map_err(|_| "Native PD token affinity requires nonnegative input_ids")
    }

    fn get_generate_estimated_tokens(req: &GenerateRequest) -> usize {
        let input_tokens = match req.input_ids.as_ref() {
            Some(InputIds::Single(ids)) => ids.len(),
            Some(InputIds::Batch(ids)) => ids.iter().map(Vec::len).sum(),
            None => 0,
        };
        let output_tokens = req
            .sampling_params
            .as_ref()
            .and_then(|params| params.max_new_tokens)
            .unwrap_or(0) as usize;
        input_tokens.saturating_add(output_tokens)
    }

    async fn route_generate_value(
        &self,
        headers: Option<&HeaderMap>,
        body: &GenerateRequest,
        raw_body: &Value,
        model_id: Option<&str>,
    ) -> Response {
        let tokens = if self.pd_token_affinity {
            match Self::get_generate_affinity_tokens(body) {
                Ok(tokens) => Some(tokens),
                Err(message) => {
                    return error::bad_request("invalid_pd_token_affinity_input", message);
                }
            }
        } else {
            None
        };
        let request_text = if self.policies_need_request_text() {
            body.text.clone()
        } else {
            None
        };
        let context = PDRequestContext {
            route: "/generate",
            batch_size: Self::get_generate_batch_size(body),
            is_stream: body.stream,
            return_logprob: body.return_logprob.unwrap_or(false),
            request_text,
            tokens,
            estimated_tokens: Self::get_generate_estimated_tokens(body),
            prefill_tokens: prefill_input_tokens(
                body.input_ids.as_ref().map(|ids| match ids {
                    InputIds::Single(ids) => ids.len(),
                    InputIds::Batch(batches) => batches.iter().map(Vec::len).sum(),
                }),
                body.text.as_deref(),
            )
            .saturating_add(1),
            model_id,
            headers: headers.cloned(),
        };
        self.execute_dual_dispatch(headers, raw_body, context).await
    }

    fn get_chat_batch_size(req: &ChatCompletionRequest) -> Option<usize> {
        if let Some(n) = req.n {
            if n > 1 {
                return Some(n as usize);
            }
        }
        None
    }

    fn get_completion_batch_size(req: &CompletionRequest) -> Option<usize> {
        if let StringOrArray::Array(arr) = &req.prompt {
            if !arr.is_empty() {
                return Some(arr.len());
            }
        }
        None
    }

    // Static key strings to avoid per-request allocations
    const BOOTSTRAP_HOST_KEY: &'static str = "bootstrap_host";
    const BOOTSTRAP_PORT_KEY: &'static str = "bootstrap_port";
    const BOOTSTRAP_ROOM_KEY: &'static str = "bootstrap_room";
    const DISAGG_PREFILL_DP_RANK_KEY: &'static str = "disagg_prefill_dp_rank";
    const DISAGG_DECODE_DP_RANK_KEY: &'static str = "disagg_decode_dp_rank";

    fn inject_bootstrap_into_value(
        mut original: Value,
        prefill_worker: &dyn Worker,
        batch_size: Option<usize>,
    ) -> Result<Value, String> {
        let obj = original
            .as_object_mut()
            .ok_or_else(|| "Request must be a JSON object".to_string())?;

        if let Some(n) = batch_size {
            let mut hosts = Vec::with_capacity(n);
            let mut ports = Vec::with_capacity(n);
            let mut rooms = Vec::with_capacity(n);
            for _ in 0..n {
                hosts.push(prefill_worker.bootstrap_host());
                ports.push(prefill_worker.bootstrap_port());
                rooms.push(super::pd_types::generate_room_id());
            }
            // Use static string keys to avoid per-request allocations
            obj.insert(
                Self::BOOTSTRAP_HOST_KEY.to_string(),
                Value::Array(hosts.into_iter().map(Value::from).collect()),
            );
            obj.insert(
                Self::BOOTSTRAP_PORT_KEY.to_string(),
                Value::Array(
                    ports
                        .into_iter()
                        .map(|p| match p {
                            Some(v) => Value::from(v),
                            None => Value::Null,
                        })
                        .collect(),
                ),
            );
            obj.insert(
                Self::BOOTSTRAP_ROOM_KEY.to_string(),
                Value::Array(rooms.into_iter().map(Value::from).collect()),
            );
        } else {
            // Use static string keys to avoid per-request allocations
            obj.insert(
                Self::BOOTSTRAP_HOST_KEY.to_string(),
                Value::from(prefill_worker.bootstrap_host()),
            );
            obj.insert(
                Self::BOOTSTRAP_PORT_KEY.to_string(),
                match prefill_worker.bootstrap_port() {
                    Some(v) => Value::from(v),
                    None => Value::Null,
                },
            );
            obj.insert(
                Self::BOOTSTRAP_ROOM_KEY.to_string(),
                Value::from(super::pd_types::generate_room_id()),
            );
        }
        Ok(original)
    }

    fn inject_selected_pd_ranks<'a>(
        request: Cow<'a, Value>,
        prefill_dp_rank: Option<usize>,
        decode_dp_rank: Option<usize>,
    ) -> Result<Cow<'a, Value>, String> {
        let has_rank_hints = request.as_object().is_some_and(|obj| {
            obj.contains_key(Self::DISAGG_PREFILL_DP_RANK_KEY)
                || obj.contains_key(Self::DISAGG_DECODE_DP_RANK_KEY)
        });
        if prefill_dp_rank.is_none() && decode_dp_rank.is_none() && !has_rank_hints {
            return Ok(request);
        }

        let mut request = request.into_owned();
        let Some(obj) = request.as_object_mut() else {
            return Err(
                "Failed to insert selected PD ranks because request body is not an object"
                    .to_string(),
            );
        };

        // Selected workers are authoritative over client-provided rank hints.
        obj.remove(Self::DISAGG_PREFILL_DP_RANK_KEY);
        obj.remove(Self::DISAGG_DECODE_DP_RANK_KEY);
        if let Some(prefill_dp_rank) = prefill_dp_rank {
            obj.insert(
                Self::DISAGG_PREFILL_DP_RANK_KEY.to_string(),
                Value::from(prefill_dp_rank as u64),
            );
        }
        if let Some(decode_dp_rank) = decode_dp_rank {
            obj.insert(
                Self::DISAGG_DECODE_DP_RANK_KEY.to_string(),
                Value::from(decode_dp_rank as u64),
            );
        }
        Ok(Cow::Owned(request))
    }

    async fn prepare_worker_request<'a>(
        route: &'static str,
        worker: &dyn Worker,
        json_request: Cow<'a, Value>,
    ) -> Result<PreparedWorkerRequest<'a>, String> {
        let body = if worker.is_dp_aware() {
            Cow::Owned(
                worker
                    .prepare_request(json_request.into_owned())
                    .await
                    .map_err(|err| {
                        format!(
                            "Failed to prepare request for worker {}: {}",
                            worker.url(),
                            err
                        )
                    })?,
            )
        } else {
            json_request
        };

        Ok(PreparedWorkerRequest {
            endpoint_url: Self::worker_endpoint_url(worker, route),
            body,
        })
    }

    async fn prepare_pd_worker_requests<'a>(
        route: &'static str,
        json_request: &'a Value,
        prefill: &dyn Worker,
        decode: &dyn Worker,
    ) -> Result<(PreparedWorkerRequest<'a>, PreparedWorkerRequest<'a>), String> {
        let prefill_json_request = Self::inject_selected_pd_ranks(
            Cow::Borrowed(json_request),
            prefill.dp_rank(),
            decode.dp_rank(),
        )?;
        let prefill_request =
            Self::prepare_worker_request(route, prefill, prefill_json_request).await?;
        let decode_json_request =
            Self::inject_selected_pd_ranks(Cow::Borrowed(json_request), prefill.dp_rank(), None)?;
        let decode_request =
            Self::prepare_worker_request(route, decode, decode_json_request).await?;

        Ok((prefill_request, decode_request))
    }

    async fn execute_dual_dispatch<T: Serialize + Clone>(
        &self,
        headers: Option<&HeaderMap>,
        original_request: &T,
        context: PDRequestContext<'_>,
    ) -> Response {
        let start_time = Instant::now();

        let route = context.route;
        let model = context.model_id.unwrap_or(UNKNOWN_MODEL_ID);
        let endpoint = route_to_endpoint(route);

        // Record request start (Layer 2)
        Metrics::record_router_request(
            metrics_labels::ROUTER_HTTP,
            metrics_labels::BACKEND_PD,
            metrics_labels::CONNECTION_HTTP,
            model,
            endpoint,
            bool_to_static_str(context.is_stream),
        );
        // Clone request once outside the retry loop, then use Arc to share across attempts
        // This avoids O(retries) clones by sharing the same data
        let shared_request = Arc::new(original_request.clone());
        let response = RetryExecutor::execute_response_with_retry(
            &self.retry_config,
            {
                move |attempt: u32| {
                    // Clone Arc (cheap reference count increment) instead of cloning the entire request
                    let shared_request = Arc::clone(&shared_request);
                    let context = context.clone();
                    async move {
                        let (prefill, decode, mut reservations) = match self
                            .select_pd_pair(
                                context.request_text.as_deref(),
                                context.tokens.as_deref(),
                                context.estimated_tokens,
                                context.prefill_tokens,
                                context.model_id,
                                context.headers.as_ref(),
                                true,
                            )
                            .await
                        {
                            Ok(pair) => pair,
                            Err(e) => {
                                return Self::handle_server_selection_error(e);
                            }
                        };

                        debug!(
                            "PD retry attempt {} using prefill={} decode={}",
                            attempt,
                            prefill.url(),
                            decode.url()
                        );
                        let prefill_stage = PrefillStage::new(
                            reservations.prefill.take(),
                            Arc::clone(&prefill),
                            headers,
                        );

                        let mut json_request = match serde_json::to_value(shared_request.as_ref()) {
                            Ok(v) => v,
                            Err(e) => return Self::handle_serialization_error(e),
                        };

                        json_request = match Self::inject_bootstrap_into_value(
                            json_request,
                            prefill.as_ref(),
                            context.batch_size,
                        ) {
                            Ok(v) => v,
                            Err(e) => return Self::handle_serialization_error(e),
                        };

                        let ctx_is_stream = context.is_stream;
                        let mut response = self
                            .execute_dual_dispatch_internal(
                                headers,
                                json_request,
                                context,
                                Arc::clone(&prefill),
                                Arc::clone(&decode),
                                prefill_stage,
                                start_time,
                            )
                            .await;
                        if self.pd_prefill_stage_reservation {
                            // The frozen replay client already records headers.
                            // Preserve the selected virtual DP URLs for phase
                            // attribution without changing the stream payload.
                            for (name, worker) in [
                                ("x-sglang-pd-prefill-worker", &prefill),
                                ("x-sglang-pd-decode-worker", &decode),
                            ] {
                                if let Ok(value) = HeaderValue::from_str(worker.url()) {
                                    response.headers_mut().insert(name, value);
                                }
                            }
                        }
                        let response =
                            crate::core::AttachedBody::wrap_response(response, reservations);

                        let status = response.status();
                        let outcomes_already_recorded = response
                            .extensions()
                            .get::<BreakerOutcomesRecorded>()
                            .is_some();
                        if !outcomes_already_recorded {
                            let not_error = status.is_success() || status.is_client_error();
                            // Prefill is always non-streaming and fully read before
                            // we get here, so its outcome is final.
                            prefill.record_outcome(not_error);
                            // Decode for a streaming request is still mid-flight at
                            // this point; the `BreakerTrackedStream` wrapped around
                            // its byte stream records the outcome on drop. Skip the
                            // eager success record to avoid masking "200-then-broken"
                            // decode workers.
                            if !ctx_is_stream {
                                decode.record_outcome(not_error);
                            }
                        }

                        // Record worker errors for server errors (5xx)
                        if status.is_server_error() {
                            let error_type = error_type_from_status(status);
                            Metrics::record_worker_error(
                                metrics_labels::WORKER_PREFILL,
                                metrics_labels::CONNECTION_HTTP,
                                error_type,
                            );
                            Metrics::record_worker_error(
                                metrics_labels::WORKER_DECODE,
                                metrics_labels::CONNECTION_HTTP,
                                error_type,
                            );
                        }

                        response
                    }
                }
            },
            |res, _attempt| is_retryable_status(res.status()),
            |delay, attempt| {
                // Layer 3 worker metrics (PD mode uses both prefill and decode workers)
                Metrics::record_worker_retry(metrics_labels::WORKER_PREFILL, endpoint);
                Metrics::record_worker_retry(metrics_labels::WORKER_DECODE, endpoint);
                Metrics::record_worker_retry_backoff(attempt, delay);
            },
            || {
                Metrics::record_worker_retries_exhausted(metrics_labels::WORKER_PREFILL, endpoint);
                Metrics::record_worker_retries_exhausted(metrics_labels::WORKER_DECODE, endpoint);
            },
        )
        .await;

        // Record Layer 2 metrics
        let duration = start_time.elapsed();
        if response.status().is_success() {
            Metrics::record_router_duration(
                metrics_labels::ROUTER_HTTP,
                metrics_labels::BACKEND_PD,
                metrics_labels::CONNECTION_HTTP,
                model,
                endpoint,
                duration,
            );
        } else if !is_retryable_status(response.status()) {
            Metrics::record_router_error(
                metrics_labels::ROUTER_HTTP,
                metrics_labels::BACKEND_PD,
                metrics_labels::CONNECTION_HTTP,
                model,
                endpoint,
                error_type_from_status(response.status()),
            );
        }

        response
    }

    async fn handle_decode_error_response(
        &self,
        res: reqwest::Response,
        context: &PDRequestContext<'_>,
        decode: Arc<dyn Worker>,
    ) -> Response {
        let status = res.status();

        if context.is_stream {
            // Handle streaming error response
            let response_headers = header_utils::preserve_response_headers(res.headers());
            let error_payload = match res.bytes().await {
                Ok(error_body) => match serde_json::from_slice::<Value>(&error_body) {
                    Ok(error_json) => {
                        json!({ "message": error_json, "status": status.as_u16() })
                    }
                    Err(parse_err) => {
                        let body_text = String::from_utf8_lossy(&error_body).to_string();
                        let preview: String = body_text.chars().take(256).collect();
                        tracing::warn!(
                            "Failed to parse decode error body as JSON from {}: {} \
                             (status={}, body preview: {:?})",
                            decode.url(),
                            parse_err,
                            status.as_u16(),
                            preview
                        );
                        json!({ "message": body_text, "status": status.as_u16() })
                    }
                },
                Err(e) => {
                    json!({ "message": format!("Decode server error: {}", e), "status": status.as_u16() })
                }
            };

            let sse_data = format!(
                "data: {{'error': {}}}",
                serde_json::to_string(&error_payload).unwrap_or_default()
            );
            let error_stream = tokio_stream::once(Ok(axum::body::Bytes::from(sse_data)));

            self.create_streaming_response(
                error_stream,
                status,
                None,
                context.return_logprob,
                Some(response_headers),
                decode,
            )
        } else {
            // Handle non-streaming error response
            match res.bytes().await {
                Ok(error_body) => {
                    // Try to parse error message from body, fallback to status-based error
                    let error_message = if let Ok(error_json) =
                        serde_json::from_slice::<Value>(&error_body)
                    {
                        if let Some(msg) = error_json
                            .get("error")
                            .and_then(|e| e.get("message"))
                            .and_then(|m| m.as_str())
                        {
                            msg.to_string()
                        } else if let Some(msg) = error_json.get("message").and_then(|m| m.as_str())
                        {
                            msg.to_string()
                        } else {
                            String::from_utf8_lossy(&error_body).to_string()
                        }
                    } else {
                        String::from_utf8_lossy(&error_body).to_string()
                    };

                    let status_code = StatusCode::from_u16(status.as_u16())
                        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                    match status_code {
                        StatusCode::BAD_REQUEST => {
                            error::bad_request("decode_bad_request", error_message)
                        }
                        StatusCode::NOT_FOUND => {
                            error::not_found("decode_not_found", error_message)
                        }
                        StatusCode::INTERNAL_SERVER_ERROR => {
                            error::internal_error("decode_internal_error", error_message)
                        }
                        StatusCode::SERVICE_UNAVAILABLE => {
                            error::service_unavailable("decode_unavailable", error_message)
                        }
                        StatusCode::BAD_GATEWAY => {
                            error::bad_gateway("decode_bad_gateway", error_message)
                        }
                        _ => error::internal_error("decode_error", error_message),
                    }
                }
                Err(e) => {
                    let error_message = format!("Decode server error: {}", e);
                    let status_code = StatusCode::from_u16(status.as_u16())
                        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                    match status_code {
                        StatusCode::BAD_REQUEST => {
                            error::bad_request("decode_read_failed", error_message)
                        }
                        StatusCode::NOT_FOUND => {
                            error::not_found("decode_read_failed", error_message)
                        }
                        StatusCode::INTERNAL_SERVER_ERROR => {
                            error::internal_error("decode_read_failed", error_message)
                        }
                        StatusCode::SERVICE_UNAVAILABLE => {
                            error::service_unavailable("decode_read_failed", error_message)
                        }
                        StatusCode::BAD_GATEWAY => {
                            error::bad_gateway("decode_read_failed", error_message)
                        }
                        _ => error::internal_error("decode_read_failed", error_message),
                    }
                }
            }
        }
    }

    // D may reject before P has returned even its headers. In that case P is
    // cancelled, so its breaker has no observed outcome. Attribute only D's
    // actual status/transport result and preserve the usual response shape.
    async fn handle_decode_before_prefill(
        &self,
        result: Result<reqwest::Response, reqwest::Error>,
        context: &PDRequestContext<'_>,
        decode: Arc<dyn Worker>,
    ) -> Response {
        let mut response = match result {
            Ok(res) => {
                if !context.is_stream {
                    decode.record_outcome(res.status().is_client_error());
                }
                self.handle_decode_error_response(res, context, decode)
                    .await
            }
            Err(e) => {
                decode.record_outcome(false);
                error!(decode_url = %decode.url(), error = %e, "Decode request failed before prefill headers");
                error::bad_gateway("decode_server_error", format!("Decode server error: {}", e))
            }
        };
        response.extensions_mut().insert(BreakerOutcomesRecorded);
        response
    }

    // Internal method that performs the actual dual dispatch (without retry logic)
    async fn execute_dual_dispatch_internal(
        &self,
        headers: Option<&HeaderMap>,
        json_request: Value,
        context: PDRequestContext<'_>,
        prefill: Arc<dyn Worker>,
        decode: Arc<dyn Worker>,
        prefill_stage: PrefillStage,
        _start_time: Instant,
    ) -> Response {
        // P's guard and lease were acquired immediately after selection. They
        // remain live through the P body, including while D runs concurrently.
        let _decode_guard =
            (!context.is_stream).then(|| WorkerLoadGuard::new(decode.clone(), headers));

        let mut headers_with_trace = headers.cloned().unwrap_or_default();
        inject_trace_context_http(&mut headers_with_trace);
        let headers = Some(&headers_with_trace);

        let (prepared_prefill, prepared_decode) = match Self::prepare_pd_worker_requests(
            context.route,
            &json_request,
            prefill.as_ref(),
            decode.as_ref(),
        )
        .await
        {
            Ok(requests) => requests,
            Err(e) => {
                error!("Failed to prepare PD worker requests: {}", e);
                return error::internal_error("pd_request_preparation_failed", e);
            }
        };

        // Build both requests
        let prefill_request = self.build_post_with_headers(
            &self.client,
            &prepared_prefill.endpoint_url,
            &prepared_prefill.body,
            headers,
            false,
        );
        let decode_request = self.build_post_with_headers(
            &self.client,
            &prepared_decode.endpoint_url,
            &prepared_decode.body,
            headers,
            false,
        );

        // Run both in this handler task (not a detached tokio::spawn) so a client
        // disconnect cancels the pending decode request too, keeping the
        // upstream-cancel behavior from #19524.
        events::RequestPDSentEvent {
            prefill_url: prefill.url(),
            decode_url: decode.url(),
        }
        .emit();

        let mut prefill_fut = Box::pin(prefill_request.send());
        let mut decode_fut = Box::pin(decode_request.send());

        // Poll both until prefill resolves; decode normally resolves later, but
        // may resolve first if it rejects the request outright.
        let prefill_result;
        let mut decode_early: Option<Result<reqwest::Response, reqwest::Error>> = None;
        loop {
            tokio::select! {
                biased;
                pr = &mut prefill_fut => {
                    prefill_result = pr;
                    break;
                }
                dr = &mut decode_fut, if decode_early.is_none() => {
                    let failed = match &dr {
                        Ok(res) => !res.status().is_success(),
                        Err(_) => true,
                    };
                    if failed {
                        // `Box::pin` owns the send future: dropping it closes
                        // P's pending request before reading D's error body.
                        drop(prefill_fut);
                        drop(prefill_stage);
                        return self.handle_decode_before_prefill(dr, &context, decode).await;
                    }
                    decode_early = Some(dr);
                }
            }
        }

        // Decode can't generate without prefill's KV, so any prefill failure
        // (non-2xx / transport error) dooms the paired decode request, which would
        // otherwise block in WaitingForInput until the 300s disaggregation
        // timeout. Drop the decode future to close its connection; the decode
        // engine then detects the disconnect and aborts the request in ~4-8s.
        let prefill_failed = match &prefill_result {
            Ok(resp) => !resp.status().is_success(),
            Err(_) => true,
        };
        let prefill_ok = match &prefill_result {
            Ok(resp) => resp.status().is_success() || resp.status().is_client_error(),
            Err(_) => false,
        };

        if prefill_failed {
            warn!(
                "Prefill failed, aborting paired decode request decode_url={} prefill_url={}",
                decode.url(),
                prefill.url()
            );

            // Tick prefill by its real status (4xx = client fault). Don't record
            // decode: it was cancelled due to a prefill fault, not its own, so a
            // prefill error storm can't trip healthy decode breakers.
            prefill.record_outcome(prefill_ok);

            // Status-faithful error shaping (including P's 500; transport -> 502).
            let mut response = match self
                .process_prefill_response(prefill_result, prefill.url(), false)
                .await
            {
                Err(error_response) => error_response,
                Ok(_) => error::bad_gateway(
                    "prefill_server_error",
                    "Prefill reported failure but returned a success response".to_string(),
                ),
            };
            response.extensions_mut().insert(BreakerOutcomesRecorded);
            return response;
        }

        // Finish P before waiting for successful D headers. Keep polling the
        // concurrent D request while reading P. A D rejection cancels P just
        // as before, rather than waiting indefinitely for an orphaned P body.
        let decode_failed_early = decode_early.as_ref().is_some_and(|result| match result {
            Ok(res) => !res.status().is_success(),
            Err(_) => true,
        });
        let prefill_body_result = if decode_failed_early {
            drop(prefill_result);
            None
        } else {
            let prefill_body_fut = self.process_prefill_response(
                prefill_result,
                prefill.url(),
                context.return_logprob,
            );
            tokio::pin!(prefill_body_fut);
            loop {
                tokio::select! {
                    biased;
                    body = &mut prefill_body_fut => break Some(body),
                    dr = &mut decode_fut, if decode_early.is_none() => {
                        let failed = match &dr {
                            Ok(res) => !res.status().is_success(),
                            Err(_) => true,
                        };
                        decode_early = Some(dr);
                        if failed {
                            break None;
                        }
                    }
                }
            }
        };
        let prefill_body = match prefill_body_result {
            Some(Ok((_, body))) => body,
            Some(Err(error_response)) => return error_response,
            None => None,
        };
        drop(prefill_stage);

        // Prefill ok: take decode's result, awaiting it if still pending.
        let decode_result = match decode_early {
            Some(dr) => dr,
            None => (&mut decode_fut).await,
        };

        events::RequestReceivedEvent {}.emit();

        // Process decode response
        match decode_result {
            Ok(res) => {
                let status = StatusCode::from_u16(res.status().as_u16())
                    .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                debug!("Decode response status: {}", status);

                if !status.is_success() {
                    error!(
                        "Decode server returned error status decode_url={} status={}",
                        decode.url(),
                        status
                    );

                    // Per-worker breaker attribution before the synthetic 5xx
                    // response takes over. Prefill ran concurrently in the
                    // `tokio::join!`: tick it based on its actual response
                    // status, not on the decode-driven failure. For
                    // non-streaming the response carries no tracked stream
                    // so record decode's outcome here too — but treat 4xx
                    // as a client fault rather than a worker fault, matching
                    // the legacy outer-dispatcher rule and the streaming
                    // `BreakerTrackedStream` pre-mark in
                    // `create_streaming_response`. For streaming
                    // `handle_decode_error_response` wraps the synthetic
                    // error SSE in a `BreakerTrackedStream` that ticks
                    // decode on drop, so skip to avoid double-counting.
                    // Mark the response so the outer dispatcher skips its
                    // status-derived `record_outcome`.
                    prefill.record_outcome(prefill_ok);
                    if !context.is_stream {
                        let decode_ok = status.is_success() || status.is_client_error();
                        decode.record_outcome(decode_ok);
                    }

                    let mut response = self
                        .handle_decode_error_response(res, &context, decode)
                        .await;
                    response.extensions_mut().insert(BreakerOutcomesRecorded);
                    return response;
                }

                if context.is_stream {
                    // Streaming response
                    let prefill_logprobs = if context.return_logprob {
                        prefill_body
                            .as_ref()
                            .and_then(|body| serde_json::from_slice::<Value>(body).ok())
                            .and_then(|json| {
                                json.pointer("/meta_info/input_token_logprobs").cloned()
                            })
                    } else {
                        None
                    };

                    let response_headers = header_utils::preserve_response_headers(res.headers());

                    self.create_streaming_response(
                        res.bytes_stream(),
                        status,
                        prefill_logprobs,
                        context.return_logprob,
                        Some(response_headers),
                        decode,
                    )
                } else {
                    // Non-streaming response
                    if context.return_logprob {
                        self.process_non_streaming_response(
                            res,
                            status,
                            context.return_logprob,
                            prefill_body,
                        )
                        .await
                    } else {
                        // Direct passthrough when no logprobs needed
                        let response_headers =
                            header_utils::preserve_response_headers(res.headers());

                        match res.bytes().await {
                            Ok(decode_body) => {
                                let mut response = Response::new(Body::from(decode_body));
                                *response.status_mut() = status;
                                *response.headers_mut() = response_headers;
                                response
                            }
                            Err(e) => {
                                error!("Failed to read decode response: {}", e);
                                error::internal_error(
                                    "read_response_failed",
                                    "Failed to read response",
                                )
                            }
                        }
                    }
                }
            }
            Err(e) => {
                error!(
                    decode_url = %decode.url(),
                    error = %e,
                    "Decode request failed"
                );
                // Decode failed at TCP/transport level. No tracked
                // stream will ever wrap a response (streaming path) and
                // we shortcut past the outer non-streaming
                // `record_outcome` too — so record decode failure
                // directly. Prefill ran concurrently in the
                // `tokio::join!`: record its real per-worker outcome
                // (success on a 2xx/4xx send, failure on transport
                // error) so the decode-driven 502 doesn't penalise a
                // healthy prefill. Mark the response so the outer
                // dispatcher skips its status-derived `record_outcome`
                // and we don't double-count.
                decode.record_outcome(false);
                prefill.record_outcome(prefill_ok);

                let mut response = error::bad_gateway(
                    "decode_server_error",
                    format!("Decode server error: {}", e),
                );
                response.extensions_mut().insert(BreakerOutcomesRecorded);
                response
            }
        }
    }

    fn policies_need_request_text(&self) -> bool {
        let prefill_policy = self.policy_registry.get_prefill_policy();
        let decode_policy = self.policy_registry.get_decode_policy();
        prefill_policy.needs_request_text() || decode_policy.needs_request_text()
    }

    /// Builds the text used for cache-aware routing of a chat request.
    ///
    /// This must reflect the *full* conversation (system prompt, prior turns,
    /// the current message and tool context) so that KV-cache prefix matching
    /// routes to the worker that actually shares the most prefix. Using only the
    /// first message ignores the conversation history that drives KV reuse in
    /// multi-turn chats. See https://github.com/sgl-project/sglang/issues/26263.
    ///
    /// Returns `None` when the conversation has no text to route on, preserving
    /// the prior behavior of not feeding an empty key into prefix matching.
    fn build_chat_request_text(body: &ChatCompletionRequest) -> Option<String> {
        // `extract_text_for_routing` walks every message (system, prior turns,
        // current message, tool content) and is the same routing text the regular
        // (non-PD) router uses, keeping cache-aware routing consistent across both.
        let text = body.extract_text_for_routing();
        if text.is_empty() {
            None
        } else {
            Some(text)
        }
    }

    async fn select_pd_pair(
        &self,
        request_text: Option<&str>,
        tokens: Option<&[u32]>,
        estimated_tokens: usize,
        prefill_tokens: usize,
        model_id: Option<&str>,
        headers: Option<&HeaderMap>,
        reserve: bool,
    ) -> Result<(Arc<dyn Worker>, Arc<dyn Worker>, PDReservationPair), String> {
        let effective_model_id = if !self.enable_igw { None } else { model_id };

        debug!(
            "Selecting PD pair: enable_igw={}, model_id={:?}, effective_model_id={:?}",
            self.enable_igw, model_id, effective_model_id
        );

        let prefill_workers = if let Some(model) = effective_model_id {
            self.worker_registry
                .get_by_model(model)
                .iter()
                .filter(|w| matches!(w.worker_type(), WorkerType::Prefill { .. }))
                .cloned()
                .collect()
        } else {
            self.worker_registry.get_prefill_workers()
        };

        let decode_workers = if let Some(model) = effective_model_id {
            self.worker_registry
                .get_by_model(model)
                .iter()
                .filter(|w| matches!(w.worker_type(), WorkerType::Decode))
                .cloned()
                .collect()
        } else {
            self.worker_registry.get_decode_workers()
        };

        let prefill_policy = self.policy_registry.get_prefill_policy();
        let decode_policy = self.policy_registry.get_decode_policy();

        // Get cached hash ring for consistent hashing
        let hash_ring_model = hash_ring_model_for_pd(
            effective_model_id,
            prefill_workers.iter().map(|worker| worker.model_id()),
        );
        let hash_ring = hash_ring_model.and_then(|model| self.worker_registry.get_hash_ring(model));

        let attempts = decode_workers.len().saturating_mul(2).max(1);
        for _ in 0..attempts {
            let snapshots = if self.pd_token_affinity || self.pd_prefill_stage_reservation {
                self.load_snapshots.borrow().clone()
            } else {
                HashMap::new()
            };
            let (
                local_reservations,
                post_snapshot_reserved_tokens,
                post_snapshot_reserved_requests,
            ) = self.reservations.views(&snapshots);
            let policy_tokens = if self.pd_token_affinity { tokens } else { None };

            let prefill = Self::pick_worker_by_policy_arc(
                &prefill_workers,
                &*prefill_policy,
                request_text,
                policy_tokens,
                headers,
                hash_ring.clone(),
                &snapshots,
                &local_reservations,
                &post_snapshot_reserved_tokens,
                &post_snapshot_reserved_requests,
                if self.pd_prefill_stage_reservation {
                    prefill_tokens
                } else {
                    estimated_tokens
                },
                None,
                "prefill",
            )
            .await?;
            let prefill_lease = if reserve && self.pd_prefill_stage_reservation {
                self.reservations
                    .try_reserve(prefill.url(), prefill_tokens, None)
            } else {
                None
            };
            let decode = Self::pick_worker_by_policy_arc(
                &decode_workers,
                &*decode_policy,
                request_text,
                policy_tokens,
                headers,
                hash_ring.clone(),
                &snapshots,
                &local_reservations,
                &post_snapshot_reserved_tokens,
                &post_snapshot_reserved_requests,
                estimated_tokens,
                self.pd_token_affinity
                    .then_some(self.pd_decode_inflight_cap),
                "decode",
            )
            .await?;

            let decode_lease = if reserve && self.pd_token_affinity {
                let Some(decode_lease) = self.reservations.try_reserve(
                    decode.url(),
                    estimated_tokens,
                    Some(self.pd_decode_inflight_cap),
                ) else {
                    continue;
                };
                Some(decode_lease)
            } else {
                None
            };
            let reservations = PDReservationPair {
                prefill: prefill_lease,
                _decode: decode_lease,
            };

            let model = model_id.unwrap_or(UNKNOWN_MODEL_ID);
            Metrics::record_worker_selection(
                metrics_labels::WORKER_PREFILL,
                metrics_labels::CONNECTION_HTTP,
                model,
                prefill_policy.name(),
            );
            Metrics::record_worker_selection(
                metrics_labels::WORKER_DECODE,
                metrics_labels::CONNECTION_HTTP,
                model,
                decode_policy.name(),
            );
            return Ok((prefill, decode, reservations));
        }

        Err("PD capacity changed while reserving a worker pair".to_string())
    }

    async fn pick_worker_by_policy_arc(
        workers: &[Arc<dyn Worker>],
        policy: &dyn LoadBalancingPolicy,
        request_text: Option<&str>,
        tokens: Option<&[u32]>,
        headers: Option<&HeaderMap>,
        hash_ring: Option<Arc<HashRing>>,
        load_snapshots: &HashMap<String, DPLoadSnapshot>,
        local_reservations: &HashMap<String, usize>,
        post_snapshot_reserved_tokens: &HashMap<String, usize>,
        post_snapshot_reserved_requests: &HashMap<String, usize>,
        estimated_tokens: usize,
        soft_cap: Option<usize>,
        worker_type: &str,
    ) -> Result<Arc<dyn Worker>, String> {
        if workers.is_empty() {
            return Err(format!(
                "No {} workers available. Please check if {} servers are configured and healthy.",
                worker_type, worker_type
            ));
        }

        let available_workers: Vec<Arc<dyn Worker>> = workers
            .iter()
            .filter(|worker| {
                if !worker.is_available() {
                    return false;
                }
                let local = local_reservations.get(worker.url()).copied().unwrap_or(0);
                if soft_cap.is_some_and(|cap| local >= cap) {
                    return false;
                }
                true
            })
            .cloned()
            .collect();

        if available_workers.is_empty() {
            return Err(format!(
                "No available {} workers (all circuits open or unhealthy)",
                worker_type
            ));
        }

        let selected_idx = policy
            .select_worker(
                &available_workers,
                &SelectWorkerInfo {
                    request_text,
                    tokens,
                    headers,
                    hash_ring,
                    load_snapshots: Some(load_snapshots),
                    local_reservations: Some(local_reservations),
                    post_snapshot_reserved_tokens: Some(post_snapshot_reserved_tokens),
                    post_snapshot_reserved_requests: Some(post_snapshot_reserved_requests),
                    estimated_tokens,
                    soft_cap,
                },
            )
            .await
            .ok_or_else(|| {
                format!(
                    "Policy {} failed to select a {} worker",
                    policy.name(),
                    worker_type
                )
            })?;

        Ok(available_workers[selected_idx].clone())
    }

    #[allow(clippy::too_many_arguments)]
    fn create_streaming_response(
        &self,
        stream: impl futures_util::Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send + 'static,
        status: StatusCode,
        prefill_logprobs: Option<Value>,
        return_logprob: bool,
        headers: Option<HeaderMap>,
        decode: Arc<dyn Worker>,
    ) -> Response {
        use crate::core::AttachedBody;

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

        // Uses select! to race stream.next() against tx.closed() so that
        // when the client disconnects the upstream HTTP connection is dropped
        // promptly, allowing the engine to abort the request.
        // `biased;` drains a ready upstream chunk before observing client
        // disconnect, so a chunk already produced by reqwest reaches the
        // client (and the logprob merger) before we tear the loop down.
        //
        // The upstream stream is wrapped in `BreakerTrackedStream` so the
        // decode worker's circuit breaker is updated once on drop: success
        // on clean completion (`[DONE]` sentinel or `None`), failure on
        // stream error, neither on client disconnect. PD's pre-PR semantics
        // treated 4xx (client error) as not-a-worker-fault, so we only
        // pre-mark the wrapper as Errored on 5xx — `handle_decode_error_response`
        // synthesizes a single-chunk SSE error envelope that would otherwise
        // stream cleanly to None and record a spurious success.
        let mut tracked =
            BreakerTrackedStream::new(stream, Arc::clone(&decode), decode.url().to_string());
        if !(status.is_success() || status.is_client_error()) {
            tracked.mark_errored();
        }
        let decode_for_log = decode.clone();
        tokio::spawn(async move {
            let mut done_detector = SseDoneDetector::default();
            loop {
                tokio::select! {
                    biased;
                    chunk_result = tracked.next() => {
                        match chunk_result {
                            Some(Ok(chunk)) => {
                                // The sentinel and its terminating blank line may
                                // span chunks. Inspect SSE framing without changing
                                // the bytes passed to the client/logprob merger.
                                let is_done = done_detector.feed(&chunk);

                                let result = if return_logprob && prefill_logprobs.is_some() {
                                    Self::merge_streaming_logprobs(prefill_logprobs.clone(), &chunk)
                                        .unwrap_or(chunk)
                                } else {
                                    chunk
                                };

                                // Mark the wrapper completed before the client
                                // send: upstream finished cleanly regardless of
                                // whether the client is still listening, and
                                // the worker deserves the success tick either
                                // way. `mark_completed` is a no-op once Errored
                                // is set, so the synthetic-error path is unaffected.
                                if is_done {
                                    tracked.mark_completed();
                                }

                                if tx.send(Ok(result)).is_err() {
                                    tracing::debug!(
                                        "Receiver dropped (likely client disconnect), \
                                        cancelling upstream PD stream"
                                    );
                                    break;
                                }

                                if is_done {
                                    break;
                                }
                            }
                            Some(Err(e)) => {
                                // BreakerTrackedStream already logged the error
                                // and marked the terminal state as Errored so
                                // the worker's circuit breaker will tick on drop.
                                let _ = tx.send(Err(format!("Stream error: {}", e)));
                                break;
                            }
                            None => break,
                        }
                    }
                    _ = tx.closed() => {
                        tracing::info!(
                            "Client disconnected, cancelling upstream PD stream from {}",
                            decode_for_log.url()
                        );
                        break;
                    }
                }
            }
        });

        let stream = UnboundedReceiverStream::new(rx);
        let body = Body::from_stream(stream);

        let decode_guard = WorkerLoadGuard::new(decode, headers.as_ref());

        let mut response = Response::new(body);
        *response.status_mut() = status;

        let mut response_headers = headers.unwrap_or_default();
        response_headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
        *response.headers_mut() = response_headers;

        AttachedBody::wrap_response(response, decode_guard)
    }

    // Helper to process non-streaming decode response with logprob merging
    async fn process_non_streaming_response(
        &self,
        res: reqwest::Response,
        status: StatusCode,
        return_logprob: bool,
        prefill_body: Option<bytes::Bytes>,
    ) -> Response {
        let response = res.bytes().await;
        let decode_body = match response {
            Ok(decode_body) => decode_body,
            Err(e) => {
                error!("Failed to read decode response: {}", e);
                return error::internal_error("read_response_failed", "Failed to read response");
            }
        };

        if !return_logprob {
            return (status, decode_body).into_response();
        }

        let Some(prefill_body) = prefill_body else {
            return (status, decode_body).into_response();
        };

        // Merge logprobs from prefill and decode
        let (Ok(prefill_json), Ok(mut decode_json)) = (
            serde_json::from_slice::<Value>(&prefill_body),
            serde_json::from_slice::<Value>(&decode_body),
        ) else {
            warn!("Failed to parse responses for logprob merging");
            return (status, decode_body).into_response();
        };

        Self::merge_logprobs_in_json(&prefill_json, &mut decode_json);

        // Return merged response
        match serde_json::to_vec(&decode_json) {
            Ok(body) => (status, body).into_response(),
            Err(e) => {
                error!("Failed to serialize merged response: {}", e);
                (status, decode_body).into_response()
            }
        }
    }

    // Helper to process prefill response and extract body if needed for logprobs
    async fn process_prefill_response(
        &self,
        prefill_result: Result<reqwest::Response, reqwest::Error>,
        prefill_url: &str,
        return_logprob: bool,
    ) -> Result<(StatusCode, Option<bytes::Bytes>), Response> {
        // Check prefill result first - it's critical for disaggregated mode
        let prefill_response = match prefill_result {
            Ok(response) => response,
            Err(e) => {
                error!(
                    "Prefill server failed (CRITICAL) prefill_url={} error={}. Decode will timeout without prefill KV cache.",
                    prefill_url,
                    e
                );

                // Return error immediately - don't wait for decode to timeout
                return Err(error::bad_gateway(
                    "prefill_server_error",
                    format!(
                        "Prefill server error: {}. This will cause decode timeout.",
                        e
                    ),
                ));
            }
        };

        let prefill_status = StatusCode::from_u16(prefill_response.status().as_u16())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

        // Check if prefill succeeded
        if !prefill_status.is_success() {
            // Get error body from prefill
            let error_msg = prefill_response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown prefill error".to_string());

            error!(
                "Prefill server returned error status prefill_url={} status={} body={}",
                prefill_url, prefill_status, error_msg
            );

            // Map prefill_status to appropriate error function
            let error_response = match prefill_status {
                StatusCode::BAD_REQUEST => error::bad_request(
                    "prefill_bad_request",
                    format!("Prefill server error ({}): {}", prefill_status, error_msg),
                ),
                StatusCode::NOT_FOUND => error::not_found(
                    "prefill_not_found",
                    format!("Prefill server error ({}): {}", prefill_status, error_msg),
                ),
                StatusCode::INTERNAL_SERVER_ERROR => error::internal_error(
                    "prefill_internal_error",
                    format!("Prefill server error ({}): {}", prefill_status, error_msg),
                ),
                StatusCode::SERVICE_UNAVAILABLE => error::service_unavailable(
                    "prefill_unavailable",
                    format!("Prefill server error ({}): {}", prefill_status, error_msg),
                ),
                StatusCode::BAD_GATEWAY => error::bad_gateway(
                    "prefill_bad_gateway",
                    format!("Prefill server error ({}): {}", prefill_status, error_msg),
                ),
                _ => error::internal_error(
                    "prefill_error",
                    format!("Prefill server error ({}): {}", prefill_status, error_msg),
                ),
            };
            return Err(error_response);
        }

        // Read prefill body if needed for logprob merging
        let prefill_body = if return_logprob {
            match prefill_response.bytes().await {
                Ok(body) => Some(body),
                Err(e) => {
                    warn!("Failed to read prefill response body for logprobs: {}", e);
                    None
                }
            }
        } else {
            // For non-logprob requests, just consume the response without storing
            debug!("Consuming prefill response body (non-logprob request)");
            match prefill_response.bytes().await {
                Ok(_) => debug!("Prefill response consumed successfully"),
                Err(e) => warn!("Error consuming prefill response: {}", e),
            }
            None
        };

        Ok((prefill_status, prefill_body))
    }

    fn build_post_with_headers(
        &self,
        client: &Client,
        endpoint_url: &str,
        json_request: &Value,
        headers: Option<&HeaderMap>,
        connection_close: bool,
    ) -> reqwest::RequestBuilder {
        let mut request = client.post(endpoint_url).json(json_request);
        if connection_close {
            request = request.header("Connection", "close");
        }
        if let Some(headers) = headers {
            for (name, value) in headers.iter() {
                if header_utils::should_forward_request_header(name.as_str()) {
                    if let Ok(val) = value.to_str() {
                        request = request.header(name, val);
                    }
                }
            }
        }
        request
    }

    // Helper to merge logprobs from prefill and decode responses
    // Optimized to avoid double cloning by taking ownership of decode array
    fn merge_logprobs_in_json(prefill_json: &Value, decode_json: &mut Value) -> bool {
        if let (Some(prefill_meta), Some(decode_meta)) = (
            prefill_json.get("meta_info"),
            decode_json.get_mut("meta_info"),
        ) {
            if let (Some(prefill_logprobs), Some(decode_logprobs)) = (
                prefill_meta.get("input_token_logprobs"),
                decode_meta.get_mut("input_token_logprobs"),
            ) {
                if let Some(prefill_arr) = prefill_logprobs.as_array() {
                    // Take ownership of decode array to avoid cloning it
                    let decode_arr = std::mem::take(decode_logprobs);
                    if let Value::Array(decode_vec) = decode_arr {
                        // Pre-allocate merged array with exact capacity
                        let mut merged = Vec::with_capacity(prefill_arr.len() + decode_vec.len());
                        merged.extend(prefill_arr.iter().cloned());
                        merged.extend(decode_vec);
                        decode_meta["input_token_logprobs"] = Value::Array(merged);
                        return true;
                    }
                }
            }
        }
        false
    }

    // Simple helper to merge logprobs in streaming responses
    // Optimized to reduce allocations in the merge path
    fn merge_streaming_logprobs(
        prefill_logprobs: Option<Value>,
        decode_chunk: &[u8],
    ) -> Result<bytes::Bytes, ()> {
        // Skip non-data chunks
        let chunk_str = std::str::from_utf8(decode_chunk).map_err(|_| ())?;
        if !chunk_str.starts_with("data: ") || chunk_str.contains("[DONE]") {
            return Err(());
        }

        // Parse JSON from chunk
        let json_str = chunk_str.trim_start_matches("data: ").trim();
        let mut decode_json: Value = serde_json::from_str(json_str).map_err(|_| ())?;

        // Merge prefill logprobs if available
        if let Some(ref p_logprobs) = prefill_logprobs {
            if let Some(meta) = decode_json.get_mut("meta_info") {
                if let Some(d_logprobs) = meta.get_mut("input_token_logprobs") {
                    if let Some(p_arr) = p_logprobs.as_array() {
                        // Take ownership of decode array to avoid cloning it
                        let decode_arr = std::mem::take(d_logprobs);
                        if let Value::Array(d_vec) = decode_arr {
                            // Pre-allocate merged array with exact capacity
                            let mut merged = Vec::with_capacity(p_arr.len() + d_vec.len());
                            merged.extend(p_arr.iter().cloned());
                            merged.extend(d_vec);
                            *d_logprobs = Value::Array(merged);
                        }
                    }
                }
            }
        }

        // Re-serialize
        let merged_str = format!(
            "data: {}\n\n",
            serde_json::to_string(&decode_json).unwrap_or_default()
        );
        Ok(bytes::Bytes::from(merged_str))
    }
}

#[async_trait]
impl RouterTrait for PDRouter {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn health_generate(&self, _req: Request<Body>) -> Response {
        // Note: This endpoint actually causes the model to generate tokens, so we only test one pair

        // Select a random worker pair using the policy
        let (prefill, decode, _reservations) = match self
            .select_pd_pair(None, None, 0, 1, None, None, false)
            .await
        {
            Ok(pair) => pair,
            Err(e) => {
                return error::service_unavailable(
                    "no_healthy_worker_pair",
                    format!("No healthy worker pair available: {}", e),
                );
            }
        };

        let prefill_url = Self::worker_endpoint_url(prefill.as_ref(), "health_generate");
        let decode_url = Self::worker_endpoint_url(decode.as_ref(), "health_generate");
        let (prefill_result, decode_result) = tokio::join!(
            self.client.get(&prefill_url).send(),
            self.client.get(&decode_url).send()
        );

        // Check results
        let mut errors = Vec::new();

        match prefill_result {
            Ok(res) if res.status().is_success() => {
                debug!(
                    "Health generate passed for prefill server: {}",
                    prefill.url()
                );
            }
            Ok(res) => {
                errors.push(format!(
                    "Prefill {} returned status {}",
                    prefill.url(),
                    res.status()
                ));
            }
            Err(e) => {
                errors.push(format!("Prefill {} error: {}", prefill.url(), e));
            }
        }

        match decode_result {
            Ok(res) if res.status().is_success() => {
                debug!("Health generate passed for decode server: {}", decode.url());
            }
            Ok(res) => {
                errors.push(format!(
                    "Decode {} returned status {}",
                    decode.url(),
                    res.status()
                ));
            }
            Err(e) => {
                errors.push(format!("Decode {} error: {}", decode.url(), e));
            }
        }

        if errors.is_empty() {
            (
                StatusCode::OK,
                format!(
                    "Health generate passed on selected pair: prefill={}, decode={}",
                    prefill.url(),
                    decode.url()
                ),
            )
                .into_response()
        } else {
            error::service_unavailable(
                "health_generate_failed",
                format!("Health generate failed: {:?}", errors),
            )
        }
    }

    async fn get_server_info(&self, _req: Request<Body>) -> Response {
        // Get info from the first decode server to match sglang's server info format
        // Note: We use decode workers for server info to match expected format
        self.proxy_to_first_prefill_worker("server_info", None)
            .await
    }

    async fn get_models(&self, req: Request<Body>) -> Response {
        // Extract headers first to avoid Send issues
        let headers = header_utils::copy_request_headers(&req);

        // Proxy to first prefill worker
        self.proxy_to_first_prefill_worker("v1/models", Some(headers))
            .await
    }

    async fn get_model_info(&self, req: Request<Body>) -> Response {
        // Extract headers first to avoid Send issues
        let headers = header_utils::copy_request_headers(&req);

        // Proxy to first prefill worker
        self.proxy_to_first_prefill_worker("model_info", Some(headers))
            .await
    }

    async fn route_generate(
        &self,
        headers: Option<&HeaderMap>,
        body: &GenerateRequest,
        model_id: Option<&str>,
    ) -> Response {
        let raw_body = match serde_json::to_value(body) {
            Ok(body) => body,
            Err(error) => return Self::handle_serialization_error(error),
        };
        self.route_generate_value(headers, body, &raw_body, model_id)
            .await
    }

    async fn route_generate_raw(
        &self,
        headers: Option<&HeaderMap>,
        body: &GenerateRequest,
        raw_body: &Value,
        model_id: Option<&str>,
    ) -> Response {
        self.route_generate_value(headers, body, raw_body, model_id)
            .await
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        let is_stream = body.stream;
        let return_logprob = body.logprobs;

        let request_text = if self.policies_need_request_text() || self.pd_prefill_stage_reservation
        {
            Self::build_chat_request_text(body)
        } else {
            None
        };

        // Calculate batch size
        let batch_size = Self::get_chat_batch_size(body);
        let prefill_tokens = prefill_input_tokens(None, request_text.as_deref()).saturating_add(1);

        let context = PDRequestContext {
            route: "/v1/chat/completions",
            batch_size,
            is_stream,
            return_logprob,
            request_text,
            tokens: None,
            estimated_tokens: 0,
            prefill_tokens,
            model_id,
            headers: headers.cloned(),
        };

        self.execute_dual_dispatch(headers, body, context).await
    }

    async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        body: &CompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        let is_stream = body.stream;
        let return_logprob = body.logprobs.is_some();

        let request_text = if self.policies_need_request_text() || self.pd_prefill_stage_reservation
        {
            match &body.prompt {
                StringOrArray::String(s) => Some(s.clone()),
                StringOrArray::Array(v) => v.first().map(|s| s.to_string()),
            }
        } else {
            None
        };

        // Calculate batch size
        let batch_size = Self::get_completion_batch_size(body);
        let prefill_tokens = match &body.prompt {
            StringOrArray::String(prompt) => prefill_input_tokens(None, Some(prompt)),
            StringOrArray::Array(prompts) => prompts
                .iter()
                .map(|prompt| prefill_input_tokens(None, Some(&prompt.to_string())))
                .fold(0usize, usize::saturating_add)
                .max(1),
        }
        .saturating_add(1);

        let context = PDRequestContext {
            route: "/v1/completions",
            batch_size,
            is_stream,
            return_logprob,
            request_text,
            tokens: None,
            estimated_tokens: 0,
            prefill_tokens,
            model_id,
            headers: headers.cloned(),
        };

        self.execute_dual_dispatch(headers, body, context).await
    }

    async fn route_rerank(
        &self,
        headers: Option<&HeaderMap>,
        body: &RerankRequest,
        model_id: Option<&str>,
    ) -> Response {
        // Extract text for cache-aware routing
        let req_text = if self.policies_need_request_text() || self.pd_prefill_stage_reservation {
            Some(body.query.clone())
        } else {
            None
        };

        let prefill_tokens = prefill_input_tokens(None, req_text.as_deref()).saturating_add(1);
        let context = PDRequestContext {
            route: "/v1/rerank",
            batch_size: None,
            is_stream: false,
            return_logprob: false,
            request_text: req_text,
            tokens: None,
            estimated_tokens: 0,
            prefill_tokens,
            model_id,
            headers: headers.cloned(),
        };

        self.execute_dual_dispatch(headers, body, context).await
    }

    async fn route_embeddings(
        &self,
        headers: Option<&HeaderMap>,
        body: &EmbeddingRequest,
        model_id: Option<&str>,
    ) -> Response {
        let _ = (headers, body, model_id);
        warn!("PD mode does not support /v1/embeddings; returning bad request");
        error::bad_request(
            "pd_unsupported_embeddings",
            "PD mode does not support /v1/embeddings",
        )
    }

    async fn route_classify(
        &self,
        headers: Option<&HeaderMap>,
        body: &ClassifyRequest,
        model_id: Option<&str>,
    ) -> Response {
        let _ = (headers, body, model_id);
        warn!("PD mode does not support /v1/classify; returning bad request");
        error::bad_request(
            "pd_unsupported_classify",
            "PD mode does not support /v1/classify",
        )
    }

    fn router_type(&self) -> &'static str {
        "pd"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorkerBuilder, DPAwareWorkerBuilder, WorkerType};

    async fn mock_worker(status: StatusCode, body: Body) -> String {
        use axum::routing::post;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let body = Arc::new(Mutex::new(Some(body)));
        let app = axum::Router::new().route(
            "/generate",
            post(move || {
                let body = body.lock().unwrap_or_else(|e| e.into_inner()).take();
                async move { (status, body.unwrap_or_else(Body::empty)) }
            }),
        );
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        url
    }

    async fn mock_prefill_without_headers() -> (String, tokio::sync::oneshot::Receiver<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
        let entered = Arc::new(Mutex::new(Some(entered_tx)));
        let app = axum::Router::new().route(
            "/generate",
            axum::routing::post(move || {
                let signal = entered.lock().unwrap_or_else(|e| e.into_inner()).take();
                async move {
                    if let Some(signal) = signal {
                        let _ = signal.send(());
                    }
                    std::future::pending::<Response>().await
                }
            }),
        );
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (url, entered_rx)
    }

    async fn mock_gated_decode(
        status: Option<StatusCode>,
    ) -> (String, tokio::sync::oneshot::Sender<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let (release_tx, release_rx) = tokio::sync::oneshot::channel();
        if let Some(status) = status {
            let release = Arc::new(Mutex::new(Some(release_rx)));
            let app = axum::Router::new().route(
                "/generate",
                axum::routing::post(move || {
                    let release = release.lock().unwrap_or_else(|e| e.into_inner()).take();
                    async move {
                        if let Some(release) = release {
                            let _ = release.await;
                        }
                        (status, "decode failed")
                    }
                }),
            );
            tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        } else {
            // Accept the request, then close the TCP connection without HTTP
            // headers when released to produce a genuine reqwest send error.
            tokio::spawn(async move {
                let (socket, _) = listener.accept().await.unwrap();
                let _ = release_rx.await;
                drop(socket);
            });
        }
        (url, release_tx)
    }

    async fn assert_decode_failure_before_prefill_headers(
        decode_status: Option<StatusCode>,
        stream: bool,
    ) {
        use http_body_util::BodyExt;
        use tokio::time::{timeout, Duration};

        let (p_url, p_entered) = mock_prefill_without_headers().await;
        let (d_url, release_decode) = mock_gated_decode(decode_status).await;
        let mut router = create_test_pd_router();
        router.pd_token_affinity = true;
        router.pd_prefill_stage_reservation = true;
        router.retry_config.max_retries = 1;
        let (prefill, decode) = register_pd_workers(&router, p_url, d_url);
        let router = Arc::new(router);
        let task_router = Arc::clone(&router);
        let raw = json!({"input_ids": [1, 2, 3], "stream": stream});
        let typed: GenerateRequest = serde_json::from_value(raw.clone()).unwrap();
        let request = tokio::spawn(async move {
            task_router
                .route_generate_raw(None, &typed, &raw, None)
                .await
        });

        timeout(Duration::from_secs(3), p_entered)
            .await
            .expect("P request never reached the worker")
            .unwrap();
        assert!(!request.is_finished(), "P headers are still pending");
        assert_eq!(
            router.reservations.views(&HashMap::new()).0[prefill.url()],
            1
        );
        assert_eq!(
            router.reservations.views(&HashMap::new()).0[decode.url()],
            1
        );
        assert_eq!(prefill.load(), 1);
        assert_eq!(prefill.circuit_breaker().total_successes(), 0);
        assert_eq!(prefill.circuit_breaker().total_failures(), 0);

        release_decode.send(()).unwrap();
        let response = timeout(Duration::from_secs(3), request)
            .await
            .expect("D failure must cancel pending P headers")
            .unwrap();
        assert_eq!(
            response.status(),
            decode_status.unwrap_or(StatusCode::BAD_GATEWAY)
        );
        assert!(!router
            .reservations
            .views(&HashMap::new())
            .0
            .contains_key(prefill.url()));
        assert_eq!(prefill.load(), 0);
        assert_eq!(prefill.circuit_breaker().total_successes(), 0);
        assert_eq!(prefill.circuit_breaker().total_failures(), 0);
        // The existing D lease remains attached to the response body.
        assert_eq!(
            router.reservations.views(&HashMap::new()).0[decode.url()],
            1
        );

        timeout(Duration::from_secs(3), response.into_body().collect())
            .await
            .unwrap()
            .unwrap();
        assert!(router.reservations.views(&HashMap::new()).0.is_empty());
        assert_eq!(prefill.circuit_breaker().total_successes(), 0);
        assert_eq!(prefill.circuit_breaker().total_failures(), 0);
        assert_eq!(
            decode.circuit_breaker().total_successes(),
            u64::from(decode_status.is_some_and(|status| status.is_client_error()))
        );
        assert_eq!(
            decode.circuit_breaker().total_failures(),
            u64::from(!decode_status.is_some_and(|status| status.is_client_error()))
        );
    }

    fn register_pd_workers(
        router: &PDRouter,
        prefill_url: String,
        decode_url: String,
    ) -> (Arc<dyn Worker>, Arc<dyn Worker>) {
        let prefill: Arc<dyn Worker> = Arc::from(create_test_worker(
            prefill_url,
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            true,
        ));
        let decode: Arc<dyn Worker> =
            Arc::from(create_test_worker(decode_url, WorkerType::Decode, true));
        router.worker_registry.register(Arc::clone(&prefill));
        router.worker_registry.register(Arc::clone(&decode));
        (prefill, decode)
    }

    async fn wait_for_prefill_lease(router: &PDRouter, url: &str) {
        use tokio::time::{sleep, timeout, Duration};

        timeout(Duration::from_secs(3), async {
            loop {
                if router.reservations.views(&HashMap::new()).0.get(url) == Some(&1) {
                    break;
                }
                sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("prefill lease was never acquired");
    }

    fn create_test_pd_router() -> PDRouter {
        let worker_registry = Arc::new(WorkerRegistry::new());
        let policy_registry =
            Arc::new(PolicyRegistry::new(crate::config::PolicyConfig::RoundRobin));

        let (_load_tx, load_snapshots) = watch::channel(HashMap::new());
        PDRouter {
            worker_registry,
            policy_registry,
            client: Client::new(),
            retry_config: RetryConfig::default(),
            api_key: Some("test_api_key".to_string()),
            enable_igw: false,
            pd_token_affinity: false,
            pd_prefill_stage_reservation: false,
            pd_decode_inflight_cap: 2,
            load_snapshots,
            reservations: Arc::new(PDReservationTable::default()),
        }
    }

    fn create_test_worker(url: String, worker_type: WorkerType, healthy: bool) -> Box<dyn Worker> {
        let worker = BasicWorkerBuilder::new(url)
            .worker_type(worker_type)
            .build();
        worker.set_healthy(healthy);
        Box::new(worker)
    }

    #[test]
    fn test_chat_request_text_uses_full_conversation() {
        // Regression test for https://github.com/sgl-project/sglang/issues/26263
        // Cache-aware routing must build its text from the full conversation, not
        // just the first message, so that KV-cache prefix matching reflects what
        // the worker will actually process in a multi-turn chat.
        let body: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "First question about apples."},
                {"role": "assistant", "content": "Apples are red."},
                {"role": "user", "content": "Follow up question about oranges."}
            ]
        }))
        .expect("valid chat request");

        let text = PDRouter::build_chat_request_text(&body)
            .expect("multi-message chat should produce routing text");

        assert!(
            text.contains("apples"),
            "routing text must include earlier turns, got: {text:?}"
        );
        assert!(
            text.contains("oranges"),
            "routing text must include later turns (not only the first message), got: {text:?}"
        );
    }

    #[test]
    fn test_chat_request_text_none_when_no_text() {
        // When the conversation carries no text content, no routing text should
        // be produced (None) rather than an empty string, preserving the prior
        // PD behavior. See https://github.com/sgl-project/sglang/issues/26263.
        let body: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "test-model",
            "messages": [
                {"role": "user", "content": ""}
            ]
        }))
        .expect("valid chat request");

        assert!(
            PDRouter::build_chat_request_text(&body).is_none(),
            "empty conversation text should produce None, not Some(\"\")"
        );
    }

    #[tokio::test]
    async fn test_select_healthy_prefill_worker() {
        let router = create_test_pd_router();

        let healthy_worker = create_test_worker(
            "http://healthy".to_string(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            true,
        );
        let unhealthy_worker = create_test_worker(
            "http://unhealthy".to_string(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            false,
        );
        let decode_worker =
            create_test_worker("http://decode".to_string(), WorkerType::Decode, true);

        router.worker_registry.register(Arc::from(unhealthy_worker));
        router.worker_registry.register(Arc::from(healthy_worker));
        router.worker_registry.register(Arc::from(decode_worker));

        let result = router
            .select_pd_pair(None, None, 0, 1, None, None, false)
            .await;

        assert!(result.is_ok());
        let (prefill, _decode, _reservations) = result.unwrap();

        assert_eq!(prefill.url(), "http://healthy");
        assert!(prefill.is_healthy());
    }

    #[tokio::test]
    async fn test_empty_worker_lists() {
        let router = create_test_pd_router();

        let result = router
            .select_pd_pair(None, None, 0, 1, None, None, false)
            .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No prefill workers available"));
    }

    #[test]
    fn test_worker_endpoint_url_uses_base_url_for_dp_aware_worker() {
        let worker = DPAwareWorkerBuilder::new("http://prefill:30000", 2, 4)
            .worker_type(WorkerType::Prefill {
                bootstrap_port: Some(8998),
            })
            .build();

        assert_eq!(
            PDRouter::worker_endpoint_url(&worker, "health_generate"),
            "http://prefill:30000/health_generate"
        );
        assert_eq!(
            PDRouter::worker_endpoint_url(&worker, "/v1/models"),
            "http://prefill:30000/v1/models"
        );
    }

    #[tokio::test]
    async fn test_prepare_pd_worker_requests_carries_selected_prefill_rank_to_both_sides() {
        let prefill = DPAwareWorkerBuilder::new("http://prefill:30000", 1, 2)
            .worker_type(WorkerType::Prefill {
                bootstrap_port: Some(8998),
            })
            .build();
        let decode = DPAwareWorkerBuilder::new("http://decode:30001", 6, 8)
            .worker_type(WorkerType::Decode)
            .build();
        let request = json!({
            "prompt": "shared prefix",
            "max_tokens": 8,
            "rid": "cc-request-1",
            "cache_salt": "cc-session-7",
            "routed_dp_rank": 99,
            "data_parallel_rank": 98,
            "disagg_prefill_dp_rank": 97,
            "disagg_decode_dp_rank": 96,
            "bootstrap_host": "prefill",
            "bootstrap_port": 8998,
            // room % P_DP is 0, deliberately different from selected P rank 1.
            "bootstrap_room": 400,
        });

        let (prefill_request, decode_request) =
            PDRouter::prepare_pd_worker_requests("/v1/completions", &request, &prefill, &decode)
                .await
                .unwrap();

        assert_eq!(
            prefill_request.endpoint_url,
            "http://prefill:30000/v1/completions"
        );
        assert_eq!(prefill_request.body["routed_dp_rank"], 1);
        assert_eq!(prefill_request.body["disagg_prefill_dp_rank"], 1);
        assert_eq!(prefill_request.body["disagg_decode_dp_rank"], 6);
        assert!(prefill_request.body.get("data_parallel_rank").is_none());
        assert_eq!(prefill_request.body["rid"], "cc-request-1");
        assert_eq!(prefill_request.body["cache_salt"], "cc-session-7");
        assert_eq!(prefill_request.body["bootstrap_room"], 400);

        assert_eq!(
            decode_request.endpoint_url,
            "http://decode:30001/v1/completions"
        );
        assert_eq!(decode_request.body["routed_dp_rank"], 6);
        assert!(decode_request.body.get("data_parallel_rank").is_none());
        assert_eq!(decode_request.body["disagg_prefill_dp_rank"], 1);
        assert!(decode_request.body.get("disagg_decode_dp_rank").is_none());
        assert_eq!(decode_request.body["rid"], "cc-request-1");
        assert_eq!(decode_request.body["cache_salt"], "cc-session-7");
        assert_eq!(decode_request.body["bootstrap_room"], 400);
        assert!(matches!(prefill_request.body, Cow::Owned(_)));
        assert!(matches!(decode_request.body, Cow::Owned(_)));
    }

    #[tokio::test]
    async fn test_prepare_pd_worker_requests_preserves_non_dp_workers() {
        let prefill = BasicWorkerBuilder::new("http://prefill:30000")
            .worker_type(WorkerType::Prefill {
                bootstrap_port: Some(8998),
            })
            .build();
        let decode = BasicWorkerBuilder::new("http://decode:30001")
            .worker_type(WorkerType::Decode)
            .build();
        let request = json!({
            "prompt": "shared prefix",
            "max_tokens": 8,
            "bootstrap_room": 1234,
        });

        let (prefill_request, decode_request) =
            PDRouter::prepare_pd_worker_requests("/v1/completions", &request, &prefill, &decode)
                .await
                .unwrap();

        assert_eq!(
            prefill_request.endpoint_url,
            "http://prefill:30000/v1/completions"
        );
        assert_eq!(
            decode_request.endpoint_url,
            "http://decode:30001/v1/completions"
        );
        assert!(prefill_request.body.get("data_parallel_rank").is_none());
        assert!(decode_request.body.get("data_parallel_rank").is_none());
        assert!(decode_request.body.get("disagg_prefill_dp_rank").is_none());
        assert!(matches!(prefill_request.body, Cow::Borrowed(_)));
        assert!(matches!(decode_request.body, Cow::Borrowed(_)));
    }

    #[tokio::test]
    async fn test_prepare_pd_worker_requests_removes_forged_ranks_for_non_dp_workers() {
        let prefill = BasicWorkerBuilder::new("http://prefill:30000")
            .worker_type(WorkerType::Prefill {
                bootstrap_port: Some(8998),
            })
            .build();
        let decode = BasicWorkerBuilder::new("http://decode:30001")
            .worker_type(WorkerType::Decode)
            .build();
        let request = json!({
            "prompt": "shared prefix",
            "disagg_prefill_dp_rank": 7,
            "disagg_decode_dp_rank": 6,
            "cache_salt": "tenant-extension",
        });

        let (prefill_request, decode_request) =
            PDRouter::prepare_pd_worker_requests("/v1/completions", &request, &prefill, &decode)
                .await
                .unwrap();

        for body in [&prefill_request.body, &decode_request.body] {
            assert!(body.get("disagg_prefill_dp_rank").is_none());
            assert!(body.get("disagg_decode_dp_rank").is_none());
            assert_eq!(body["cache_salt"], "tenant-extension");
        }
        assert!(matches!(prefill_request.body, Cow::Owned(_)));
        assert!(matches!(decode_request.body, Cow::Owned(_)));
    }

    #[test]
    fn test_worker_load_metrics() {
        let prefill_worker: Arc<dyn Worker> = Arc::from(create_test_worker(
            "http://prefill".to_string(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            true,
        ));
        let decode_worker: Arc<dyn Worker> = Arc::from(create_test_worker(
            "http://decode".to_string(),
            WorkerType::Decode,
            true,
        ));

        let _prefill_guard = WorkerLoadGuard::new(prefill_worker.clone(), None);
        let _decode_guard = WorkerLoadGuard::new(decode_worker.clone(), None);

        assert_eq!(prefill_worker.load(), 1);
        assert_eq!(decode_worker.load(), 1);

        drop(_prefill_guard);
        drop(_decode_guard);

        assert_eq!(prefill_worker.load(), 0);
        assert_eq!(decode_worker.load(), 0);
    }

    #[test]
    fn test_pd_hash_ring_model_without_igw() {
        assert_eq!(
            hash_ring_model_for_pd(None, ["glm", "glm", "glm"].into_iter()),
            Some("glm")
        );
        assert_eq!(
            hash_ring_model_for_pd(None, ["glm", "other"].into_iter()),
            None
        );
        assert_eq!(hash_ring_model_for_pd(None, [].into_iter()), None);
        assert_eq!(
            hash_ring_model_for_pd(Some("explicit"), ["glm", "other"].into_iter()),
            Some("explicit")
        );
    }

    #[test]
    fn test_decode_inflight_cap_config() {
        assert_eq!(parse_decode_inflight_cap("5").unwrap(), 5);
        assert!(parse_decode_inflight_cap("0").is_err());
        assert!(parse_decode_inflight_cap("invalid").is_err());

        let table = PDReservationTable::default();
        let leases: Vec<_> = (0..5)
            .map(|_| {
                table
                    .try_reserve("http://decode:30001@0", 1, Some(5))
                    .unwrap()
            })
            .collect();
        assert!(table
            .try_reserve("http://decode:30001@0", 1, Some(5))
            .is_none());
        drop(leases);
    }

    #[test]
    fn test_decode_reservation_cap_is_atomic_and_raii() {
        let table = PDReservationTable::default();
        let first = table
            .try_reserve("http://decode:30001@0", 129_024, Some(2))
            .unwrap();
        let second = table
            .try_reserve("http://decode:30001@0", 129_024, Some(2))
            .unwrap();
        assert!(table
            .try_reserve("http://decode:30001@0", 129_024, Some(2))
            .is_none());

        let (active, tokens, requests) = table.views(&HashMap::new());
        assert_eq!(active["http://decode:30001@0"], 2);
        assert_eq!(tokens["http://decode:30001@0"], 258_048);
        assert_eq!(requests["http://decode:30001@0"], 2);

        drop(first);
        let replacement = table
            .try_reserve("http://decode:30001@0", 129_024, Some(2))
            .unwrap();
        drop((second, replacement));
        assert!(table.views(&HashMap::new()).0.is_empty());
    }

    #[test]
    fn test_prefill_input_estimate_excludes_decode_output() {
        assert!(!prefill_stage_reservation_enabled(None));
        assert!(!prefill_stage_reservation_enabled(Some("0")));
        assert!(prefill_stage_reservation_enabled(Some("1")));
        assert_eq!(
            prefill_input_tokens(Some(4096), None).saturating_add(1),
            4097
        );
        assert_eq!(
            prefill_input_tokens(None, Some("12345678")).saturating_add(1),
            3
        );
        assert_eq!(prefill_input_tokens(None, None).saturating_add(1), 2);
        assert_eq!(
            prefill_input_tokens(Some(0), Some("1234")).saturating_add(1),
            2
        );
    }

    #[tokio::test]
    async fn test_prefill_reservation_without_token_ids_uses_text_fallback() {
        let mut router = create_test_pd_router();
        router.pd_prefill_stage_reservation = true;
        let (prefill, decode) = register_pd_workers(
            &router,
            "http://prefill:30000".to_string(),
            "http://decode:30001".to_string(),
        );
        let p_tokens = prefill_input_tokens(None, Some("12345678")).saturating_add(1);
        let (_, _, pair) = router
            .select_pd_pair(Some("12345678"), None, 100_000, p_tokens, None, None, true)
            .await
            .unwrap();
        let (active, tokens, requests) = router.reservations.views(&HashMap::new());
        assert_eq!(active[prefill.url()], 1);
        assert_eq!(tokens[prefill.url()], 3);
        assert_eq!(requests[prefill.url()], 1);
        assert!(!active.contains_key(decode.url()));
        drop(pair);
        assert!(router.reservations.views(&HashMap::new()).0.is_empty());
    }

    #[tokio::test]
    async fn test_prefill_reservation_uses_input_only_and_preserves_decode_soft_cap() {
        let mut router = create_test_pd_router();
        router.pd_token_affinity = true;
        router.pd_prefill_stage_reservation = true;
        router.pd_decode_inflight_cap = 1;
        let (prefill, decode) = register_pd_workers(
            &router,
            "http://prefill:30000".to_string(),
            "http://decode:30001".to_string(),
        );

        let (_, _, pair) = router
            .select_pd_pair(None, Some(&[1, 2, 3]), 100_003, 4, None, None, true)
            .await
            .unwrap();
        let (active, tokens, requests) = router.reservations.views(&HashMap::new());
        assert_eq!(active[prefill.url()], 1);
        assert_eq!(tokens[prefill.url()], 4);
        assert_eq!(requests[prefill.url()], 1);
        assert_eq!(active[decode.url()], 1);
        assert_eq!(tokens[decode.url()], 100_003);
        assert!(router
            .select_pd_pair(None, Some(&[1, 2, 3]), 100_003, 4, None, None, true)
            .await
            .is_err());
        assert_eq!(
            router.reservations.views(&HashMap::new()).0[prefill.url()],
            1
        );

        drop(pair);
        assert!(router.reservations.views(&HashMap::new()).0.is_empty());
    }

    #[tokio::test]
    async fn test_prefill_lease_ends_at_body_completion_while_decode_streams() {
        use http_body_util::BodyExt;
        use std::convert::Infallible;

        let (p_tx, p_rx) = tokio::sync::mpsc::unbounded_channel::<bytes::Bytes>();
        let (d_tx, d_rx) = tokio::sync::mpsc::unbounded_channel::<bytes::Bytes>();
        let p_url = mock_worker(
            StatusCode::OK,
            Body::from_stream(UnboundedReceiverStream::new(p_rx).map(Ok::<_, Infallible>)),
        )
        .await;
        let d_url = mock_worker(
            StatusCode::OK,
            Body::from_stream(UnboundedReceiverStream::new(d_rx).map(Ok::<_, Infallible>)),
        )
        .await;
        let mut router = create_test_pd_router();
        router.pd_token_affinity = true;
        router.pd_prefill_stage_reservation = true;
        router.retry_config.max_retries = 0;
        let (prefill, decode) = register_pd_workers(&router, p_url, d_url);
        let router = Arc::new(router);
        let task_router = Arc::clone(&router);
        let raw = json!({"input_ids": [1, 2, 3], "sampling_params": {"max_new_tokens": 100}, "stream": true});
        let typed: GenerateRequest = serde_json::from_value(raw.clone()).unwrap();
        let request = tokio::spawn(async move {
            task_router
                .route_generate_raw(None, &typed, &raw, None)
                .await
        });

        wait_for_prefill_lease(&router, prefill.url()).await;
        assert_eq!(
            router.reservations.views(&HashMap::new()).0[decode.url()],
            1
        );
        assert_eq!(prefill.load(), 1);
        assert!(!request.is_finished(), "P body is still pending");

        p_tx.send(bytes::Bytes::from_static(b"{}")).unwrap();
        drop(p_tx);
        let response = tokio::time::timeout(std::time::Duration::from_secs(3), request)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let (active, _, _) = router.reservations.views(&HashMap::new());
        assert!(!active.contains_key(prefill.url()));
        assert_eq!(active[decode.url()], 1);
        assert_eq!(prefill.load(), 0);
        assert_eq!(decode.load(), 1);

        d_tx.send(bytes::Bytes::from_static(b"data: [DONE]\n\n"))
            .unwrap();
        drop(d_tx);
        let _ = response.into_body().collect().await.unwrap();
        assert!(router.reservations.views(&HashMap::new()).0.is_empty());
        assert_eq!(decode.load(), 0);
    }

    #[tokio::test]
    async fn test_cancel_pending_prefill_releases_both_leases() {
        use std::convert::Infallible;

        let (_p_tx, p_rx) = tokio::sync::mpsc::unbounded_channel::<bytes::Bytes>();
        let (_d_tx, d_rx) = tokio::sync::mpsc::unbounded_channel::<bytes::Bytes>();
        let p_url = mock_worker(
            StatusCode::OK,
            Body::from_stream(UnboundedReceiverStream::new(p_rx).map(Ok::<_, Infallible>)),
        )
        .await;
        let d_url = mock_worker(
            StatusCode::OK,
            Body::from_stream(UnboundedReceiverStream::new(d_rx).map(Ok::<_, Infallible>)),
        )
        .await;
        let mut router = create_test_pd_router();
        router.pd_token_affinity = true;
        router.pd_prefill_stage_reservation = true;
        let (prefill, decode) = register_pd_workers(&router, p_url, d_url);
        let router = Arc::new(router);
        let task_router = Arc::clone(&router);
        let raw = json!({"input_ids": [1, 2, 3], "stream": true});
        let typed: GenerateRequest = serde_json::from_value(raw.clone()).unwrap();
        let request = tokio::spawn(async move {
            task_router
                .route_generate_raw(None, &typed, &raw, None)
                .await
        });

        wait_for_prefill_lease(&router, prefill.url()).await;
        assert_eq!(prefill.load(), 1);
        request.abort();
        let _ = request.await;
        assert!(router.reservations.views(&HashMap::new()).0.is_empty());
        assert_eq!(prefill.load(), 0);
        assert_eq!(decode.load(), 0);
    }

    #[tokio::test]
    async fn test_prefill_error_releases_leases_and_aborts_decode() {
        use std::convert::Infallible;

        let (_d_tx, d_rx) = tokio::sync::mpsc::unbounded_channel::<bytes::Bytes>();
        let p_url = mock_worker(
            StatusCode::INTERNAL_SERVER_ERROR,
            Body::from("prefill failed"),
        )
        .await;
        let d_url = mock_worker(
            StatusCode::OK,
            Body::from_stream(UnboundedReceiverStream::new(d_rx).map(Ok::<_, Infallible>)),
        )
        .await;
        let mut router = create_test_pd_router();
        router.pd_token_affinity = true;
        router.pd_prefill_stage_reservation = true;
        router.retry_config.max_retries = 0;
        let (prefill, decode) = register_pd_workers(&router, p_url, d_url);
        let raw = json!({"input_ids": [1, 2, 3], "stream": true});
        let typed: GenerateRequest = serde_json::from_value(raw.clone()).unwrap();
        let response = tokio::time::timeout(
            std::time::Duration::from_secs(3),
            router.route_generate_raw(None, &typed, &raw, None),
        )
        .await
        .unwrap();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert!(!router
            .reservations
            .views(&HashMap::new())
            .0
            .contains_key(prefill.url()));
        // The existing D lease remains attached to the returned response.
        assert_eq!(
            router.reservations.views(&HashMap::new()).0[decode.url()],
            1
        );
        drop(response);
        assert!(router.reservations.views(&HashMap::new()).0.is_empty());
        assert_eq!(prefill.load(), 0);
        assert_eq!(decode.load(), 0);
    }

    #[tokio::test]
    async fn test_early_decode_error_cancels_pending_prefill_body() {
        use std::convert::Infallible;

        let (_p_tx, p_rx) = tokio::sync::mpsc::unbounded_channel::<bytes::Bytes>();
        let p_url = mock_worker(
            StatusCode::OK,
            Body::from_stream(UnboundedReceiverStream::new(p_rx).map(Ok::<_, Infallible>)),
        )
        .await;
        let d_url = mock_worker(
            StatusCode::INTERNAL_SERVER_ERROR,
            Body::from("decode failed"),
        )
        .await;
        let mut router = create_test_pd_router();
        router.pd_token_affinity = true;
        router.pd_prefill_stage_reservation = true;
        router.retry_config.max_retries = 0;
        let (prefill, decode) = register_pd_workers(&router, p_url, d_url);
        let raw = json!({"input_ids": [1, 2, 3], "stream": true});
        let typed: GenerateRequest = serde_json::from_value(raw.clone()).unwrap();
        let response = tokio::time::timeout(
            std::time::Duration::from_secs(3),
            router.route_generate_raw(None, &typed, &raw, None),
        )
        .await
        .expect("D rejection must not wait for the held P body");
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(prefill.load(), 0);
        assert!(!router
            .reservations
            .views(&HashMap::new())
            .0
            .contains_key(prefill.url()));
        assert_eq!(
            router.reservations.views(&HashMap::new()).0[decode.url()],
            1
        );
        drop(response);
        assert!(router.reservations.views(&HashMap::new()).0.is_empty());
    }

    #[tokio::test]
    async fn test_decode_400_before_prefill_headers_cancels_prefill_without_breaker_tick() {
        assert_decode_failure_before_prefill_headers(Some(StatusCode::BAD_REQUEST), false).await;
    }

    #[tokio::test]
    async fn test_decode_500_before_prefill_headers_cancels_prefill_without_breaker_tick() {
        assert_decode_failure_before_prefill_headers(Some(StatusCode::INTERNAL_SERVER_ERROR), true)
            .await;
    }

    #[tokio::test]
    async fn test_decode_transport_before_prefill_headers_cancels_prefill_without_breaker_tick() {
        assert_decode_failure_before_prefill_headers(None, true).await;
    }

    #[tokio::test]
    async fn test_prefill_retry_releases_previous_attempt_leases() {
        use std::sync::atomic::AtomicUsize;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let p_url = format!("http://{}", listener.local_addr().unwrap());
        let calls = Arc::new(AtomicUsize::new(0));
        let handler_calls = Arc::clone(&calls);
        let app = axum::Router::new().route(
            "/generate",
            axum::routing::post(move || {
                let attempt = handler_calls.fetch_add(1, Ordering::SeqCst);
                async move {
                    if attempt == 0 {
                        (StatusCode::INTERNAL_SERVER_ERROR, "prefill failed")
                    } else {
                        (StatusCode::OK, "{}")
                    }
                }
            }),
        );
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let d_url = mock_worker(StatusCode::OK, Body::from("{}")).await;
        let mut router = create_test_pd_router();
        router.pd_token_affinity = true;
        router.pd_prefill_stage_reservation = true;
        router.retry_config.max_retries = 2;
        router.retry_config.initial_backoff_ms = 0;
        router.retry_config.jitter_factor = 0.0;
        let (prefill, decode) = register_pd_workers(&router, p_url, d_url);
        let raw = json!({"input_ids": [1, 2, 3], "stream": false});
        let typed: GenerateRequest = serde_json::from_value(raw.clone()).unwrap();
        let response = router.route_generate_raw(None, &typed, &raw, None).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert!(!router
            .reservations
            .views(&HashMap::new())
            .0
            .contains_key(prefill.url()));
        assert_eq!(
            router.reservations.views(&HashMap::new()).0[decode.url()],
            1
        );
        drop(response);
        assert!(router.reservations.views(&HashMap::new()).0.is_empty());
        assert_eq!(prefill.load(), 0);
        assert_eq!(decode.load(), 0);
    }

    #[tokio::test]
    async fn test_nonstream_logprobs_still_merge_after_prefill_body_first() {
        use http_body_util::BodyExt;

        let p_url = mock_worker(
            StatusCode::OK,
            Body::from(r#"{"meta_info":{"input_token_logprobs":[1]}}"#),
        )
        .await;
        let d_url = mock_worker(
            StatusCode::OK,
            Body::from(r#"{"meta_info":{"input_token_logprobs":[2]}}"#),
        )
        .await;
        let mut router = create_test_pd_router();
        router.pd_prefill_stage_reservation = true;
        router.retry_config.max_retries = 0;
        register_pd_workers(&router, p_url, d_url);
        let raw = json!({"input_ids": [1, 2, 3], "return_logprob": true, "stream": false});
        let typed: GenerateRequest = serde_json::from_value(raw.clone()).unwrap();
        let response = router.route_generate_raw(None, &typed, &raw, None).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["meta_info"]["input_token_logprobs"], json!([1, 2]));
        assert!(router.reservations.views(&HashMap::new()).0.is_empty());
    }

    #[tokio::test]
    async fn test_streaming_logprobs_still_merge_after_prefill_body_first() {
        use http_body_util::BodyExt;
        use std::convert::Infallible;

        let p_url = mock_worker(
            StatusCode::OK,
            Body::from(r#"{"meta_info":{"input_token_logprobs":[1]}}"#),
        )
        .await;
        let (d_tx, d_rx) = tokio::sync::mpsc::unbounded_channel::<bytes::Bytes>();
        let d_url = mock_worker(
            StatusCode::OK,
            Body::from_stream(UnboundedReceiverStream::new(d_rx).map(Ok::<_, Infallible>)),
        )
        .await;
        let mut router = create_test_pd_router();
        router.pd_token_affinity = true;
        router.pd_prefill_stage_reservation = true;
        router.retry_config.max_retries = 0;
        let (prefill, _) = register_pd_workers(&router, p_url, d_url);
        let raw = json!({"input_ids": [1, 2, 3], "return_logprob": true, "stream": true});
        let typed: GenerateRequest = serde_json::from_value(raw.clone()).unwrap();
        let response = router.route_generate_raw(None, &typed, &raw, None).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert!(!router
            .reservations
            .views(&HashMap::new())
            .0
            .contains_key(prefill.url()));

        d_tx.send(bytes::Bytes::from_static(
            b"data: {\"meta_info\":{\"input_token_logprobs\":[2]}}\n\n",
        ))
        .unwrap();
        let mut body = response.into_body();
        let frame = tokio::time::timeout(std::time::Duration::from_secs(3), body.frame())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        let chunk = frame.into_data().unwrap();
        let text = std::str::from_utf8(&chunk).unwrap();
        let merged: Value = serde_json::from_str(text.trim_start_matches("data: ").trim()).unwrap();
        assert_eq!(merged["meta_info"]["input_token_logprobs"], json!([1, 2]));
        d_tx.send(bytes::Bytes::from_static(b"data: [DONE]\n\n"))
            .unwrap();
        drop(d_tx);
        let _ = body.collect().await.unwrap();
        assert!(router.reservations.views(&HashMap::new()).0.is_empty());
    }

    #[test]
    fn test_generate_token_ids_and_extension_fields_contract() {
        let raw = json!({
            "input_ids": [11, 12, 13, 14],
            "sampling_params": {"max_new_tokens": 1024},
            "rid": "cc-rid",
            "cache_salt": "tenant-a",
            "routed_dp_rank": 7,
            "disagg_prefill_dp_rank": 6,
            "disagg_decode_dp_rank": 5
        });
        let typed: GenerateRequest = serde_json::from_value(raw.clone()).unwrap();
        assert_eq!(
            PDRouter::get_generate_affinity_tokens(&typed)
                .unwrap()
                .as_ref(),
            [11, 12, 13, 14].as_slice()
        );
        assert_eq!(PDRouter::get_generate_estimated_tokens(&typed), 1028);

        // openai-protocol 1.0.0 drops these unknown fields. The /generate
        // raw-body path must therefore remain the source sent to P and D.
        let typed_json = serde_json::to_value(&typed).unwrap();
        assert!(typed_json.get("cache_salt").is_none());
        assert_eq!(raw["cache_salt"], "tenant-a");
        assert_eq!(raw["routed_dp_rank"], 7);
    }

    #[test]
    fn test_generate_token_affinity_rejects_invalid_or_fanout_inputs() {
        let cases = [
            (
                json!({"input_ids": []}),
                "requires non-empty single input_ids",
            ),
            (
                json!({"input_ids": [11, -1, 13]}),
                "requires nonnegative input_ids",
            ),
            (
                json!({"input_ids": [[11, 12], [13, 14]]}),
                "batched input_ids expand to multiple PD requests",
            ),
            (
                json!({"text": "tokenize me"}),
                "requires input_ids as the only prompt input",
            ),
            (
                json!({
                    "input_ids": [11, 12, 13],
                    "sampling_params": {"n": 2}
                }),
                "n > 1 expands to multiple PD requests",
            ),
        ];

        for (raw, expected) in cases {
            let typed: GenerateRequest = serde_json::from_value(raw).unwrap();
            let error = PDRouter::get_generate_affinity_tokens(&typed).unwrap_err();
            assert!(
                error.contains(expected),
                "expected {expected:?} in {error:?}"
            );
        }
    }

    #[tokio::test]
    async fn test_generate_token_affinity_invalid_inputs_return_http_400() {
        use http_body_util::BodyExt;

        let mut router = create_test_pd_router();
        router.pd_token_affinity = true;
        let cases = [
            json!({"input_ids": []}),
            json!({"input_ids": [11, -1]}),
            json!({"input_ids": [[11], [12]]}),
            json!({"text": "tokenize me"}),
            json!({"input_ids": [11, 12], "sampling_params": {"n": 2}}),
        ];

        for raw in cases {
            let typed: GenerateRequest = serde_json::from_value(raw.clone()).unwrap();
            let response = router.route_generate_raw(None, &typed, &raw, None).await;
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            assert_eq!(
                response
                    .headers()
                    .get(error::HEADER_X_SMG_ERROR_CODE)
                    .and_then(|value| value.to_str().ok()),
                Some("invalid_pd_token_affinity_input")
            );
            let body = response.into_body().collect().await.unwrap().to_bytes();
            let body: Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(
                body["error"]["code"],
                Value::String("invalid_pd_token_affinity_input".to_string())
            );
        }
        assert!(router.reservations.views(&HashMap::new()).0.is_empty());
    }

    // Keep the upstream channel open after [DONE]: completion must come from
    // SSE framing, not transport EOF. Check each forwarded chunk before sending
    // the next so coalescing cannot hide a prematurely dropped terminator.
    async fn assert_sse_forwarding(chunks: &[&[u8]], status: StatusCode) {
        use http_body_util::BodyExt;
        use tokio::time::{timeout, Duration};

        let router = create_test_pd_router();
        let decode: Arc<dyn Worker> = Arc::from(create_test_worker(
            "http://decode".into(),
            WorkerType::Decode,
            true,
        ));
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let response = router.create_streaming_response(
            UnboundedReceiverStream::new(rx).map(Ok),
            status,
            None,
            false,
            None,
            decode.clone(),
        );
        assert_eq!(decode.load(), 1);
        let mut body = response.into_body();
        for chunk in chunks {
            tx.send(bytes::Bytes::copy_from_slice(chunk)).unwrap();
            let frame = timeout(Duration::from_secs(3), body.frame())
                .await
                .unwrap()
                .expect("stream ended before the complete SSE event")
                .unwrap();
            assert_eq!(frame.into_data().unwrap().as_ref(), *chunk);
        }
        timeout(Duration::from_secs(3), tx.closed()).await.unwrap();
        // The response body, rather than the producer task, owns the load guard.
        assert_eq!(decode.load(), 1);
        assert!(timeout(Duration::from_secs(3), body.collect())
            .await
            .unwrap()
            .unwrap()
            .to_bytes()
            .is_empty());
        assert_eq!(decode.load(), 0);
        let cb = decode.circuit_breaker();
        let failed = status.is_server_error();
        assert_eq!(cb.total_successes(), u64::from(!failed));
        assert_eq!(cb.total_failures(), u64::from(failed));
    }

    #[tokio::test]
    async fn test_sse_done_all_chunk_boundaries() {
        for done in [&b"data: [DONE]\n\n"[..], &b"data:[DONE]\r\n\r\n"[..]] {
            assert_sse_forwarding(&[done], StatusCode::OK).await;
            for split in 1..done.len() {
                assert_sse_forwarding(&[&done[..split], &done[split..]], StatusCode::OK).await;
            }
        }
    }

    #[tokio::test]
    async fn test_sse_json_literal_and_error_status() {
        let chunks: &[&[u8]] = &[
            b"data: {\"text\":\"data: [DONE] and data:[DONE]\"}\n\n",
            b"data: {\"text\":\"####4\",\"finish_reason\":\"stop\"}\n\n",
            b"data: [DONE]\n",
            b"\n",
        ];
        assert_sse_forwarding(chunks, StatusCode::OK).await;
        assert_sse_forwarding(chunks, StatusCode::BAD_REQUEST).await;
        assert_sse_forwarding(chunks, StatusCode::INTERNAL_SERVER_ERROR).await;
    }

    #[tokio::test]
    async fn test_sse_split_done_preserves_logprob_merge() {
        use http_body_util::BodyExt;
        use tokio::time::{timeout, Duration};

        let router = create_test_pd_router();
        let decode: Arc<dyn Worker> = Arc::from(create_test_worker(
            "http://decode".into(),
            WorkerType::Decode,
            true,
        ));
        let chunks = [
            bytes::Bytes::from_static(b"data: {\"meta_info\":{\"input_token_logprobs\":[2]}}\n\n"),
            bytes::Bytes::from_static(b"data: [DONE]\n"),
            bytes::Bytes::from_static(b"\n"),
        ];
        let stream =
            futures_util::stream::iter(chunks.map(Ok)).chain(futures_util::stream::pending());
        let response = router.create_streaming_response(
            stream,
            StatusCode::OK,
            Some(json!([1])),
            true,
            None,
            decode.clone(),
        );
        let body = timeout(Duration::from_secs(3), response.into_body().collect())
            .await
            .unwrap()
            .unwrap()
            .to_bytes();
        assert_eq!(
            body.as_ref(),
            b"data: {\"meta_info\":{\"input_token_logprobs\":[1,2]}}\n\ndata: [DONE]\n\n"
        );
        assert_eq!(decode.load(), 0);
        assert_eq!(decode.circuit_breaker().total_successes(), 1);
    }

    #[tokio::test]
    async fn test_sse_incomplete_done_error_disconnect_and_eof() {
        use http_body_util::BodyExt;
        use tokio::time::{timeout, Duration};

        for ending in ["error", "disconnect", "eof"] {
            let router = create_test_pd_router();
            let decode: Arc<dyn Worker> = Arc::from(create_test_worker(
                "http://decode".into(),
                WorkerType::Decode,
                true,
            ));
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
            let response = router.create_streaming_response(
                UnboundedReceiverStream::new(rx),
                StatusCode::OK,
                None,
                false,
                None,
                decode.clone(),
            );
            let mut body = response.into_body();
            let partial = bytes::Bytes::from_static(b"data: [DONE]\n");
            tx.send(Ok(partial.clone())).unwrap();
            let frame = timeout(Duration::from_secs(3), body.frame())
                .await
                .unwrap()
                .unwrap()
                .unwrap();
            assert_eq!(frame.into_data().unwrap(), partial);
            assert_eq!(decode.circuit_breaker().total_successes(), 0);
            assert_eq!(decode.load(), 1);

            match ending {
                "error" => {
                    // Construct a reqwest error without opening any connection.
                    let error = Client::new().get("://invalid").build().unwrap_err();
                    tx.send(Err(error)).unwrap();
                    assert!(timeout(Duration::from_secs(3), body.collect())
                        .await
                        .unwrap()
                        .is_err());
                    timeout(Duration::from_secs(3), tx.closed()).await.unwrap();
                }
                "disconnect" => {
                    drop(body);
                    timeout(Duration::from_secs(3), tx.closed()).await.unwrap();
                }
                _ => {
                    drop(tx);
                    assert!(timeout(Duration::from_secs(3), body.collect())
                        .await
                        .unwrap()
                        .unwrap()
                        .to_bytes()
                        .is_empty());
                }
            }
            assert_eq!(decode.load(), 0);
            let cb = decode.circuit_breaker();
            // Preserve transport EOF semantics; do not invent an SSE terminator.
            assert_eq!(cb.total_successes(), u64::from(ending == "eof"));
            assert_eq!(cb.total_failures(), u64::from(ending == "error"));
        }
    }

    #[tokio::test]
    async fn test_sse_ready_completion_wins_disconnect() {
        for complete in [false, true] {
            let router = create_test_pd_router();
            let decode: Arc<dyn Worker> = Arc::from(create_test_worker(
                "http://decode".into(),
                WorkerType::Decode,
                true,
            ));
            let chunk = if complete {
                bytes::Bytes::from_static(b"data: [DONE]\n\n")
            } else {
                bytes::Bytes::from_static(b"data: [DONE]\n")
            };
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
            tx.send(chunk).unwrap();
            let response = router.create_streaming_response(
                UnboundedReceiverStream::new(rx).map(Ok),
                StatusCode::OK,
                None,
                false,
                None,
                decode.clone(),
            );
            // Both branches are ready when the spawned task first runs.
            drop(response);
            tokio::time::timeout(std::time::Duration::from_secs(3), tx.closed())
                .await
                .unwrap();
            assert_eq!(decode.load(), 0);
            assert_eq!(
                decode.circuit_breaker().total_successes(),
                u64::from(complete)
            );
            assert_eq!(decode.circuit_breaker().total_failures(), 0);
        }
    }

    #[tokio::test]
    async fn test_streaming_load_tracking() {
        use futures_util::StreamExt;
        use tokio::time::{sleep, Duration};

        let router = create_test_pd_router();

        let prefill_worker = create_test_worker(
            "http://prefill".to_string(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            true,
        );
        let decode_worker =
            create_test_worker("http://decode".to_string(), WorkerType::Decode, true);

        router.worker_registry.register(Arc::from(prefill_worker));
        router.worker_registry.register(Arc::from(decode_worker));

        let prefill_workers = router.worker_registry.get_prefill_workers();
        let decode_workers = router.worker_registry.get_decode_workers();

        let prefill_ref = prefill_workers[0].clone();
        let decode_ref = decode_workers[0].clone();

        assert_eq!(prefill_ref.load(), 0);
        assert_eq!(decode_ref.load(), 0);

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let stream = UnboundedReceiverStream::new(rx);

        {
            let response = router.create_streaming_response(
                stream.map(Ok),
                StatusCode::OK,
                None,
                false,
                None,
                decode_ref.clone(),
            );

            // Only D is active for the lifetime of its streaming body.
            assert_eq!(prefill_ref.load(), 0);
            assert_eq!(decode_ref.load(), 1);

            tx.send(bytes::Bytes::from("test data")).unwrap();

            sleep(Duration::from_millis(10)).await;

            // Load still 1 while response body exists
            assert_eq!(prefill_ref.load(), 0);
            assert_eq!(decode_ref.load(), 1);

            drop(tx);

            // Response (and its body with guards) dropped here
            drop(response);
        }

        // Guards dropped when response dropped
        assert_eq!(prefill_ref.load(), 0);
        assert_eq!(decode_ref.load(), 0);
    }
}
