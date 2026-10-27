//! Opt-in, CPU-only HTTP PD experience metrics. Never change request JSON or SSE bytes.
//! The ingress guard owns the whole downstream body; dropping either a pending
//! handler or an unfinished body is a cancellation. Model labels come only from
//! deployment configuration, never from untrusted request strings.
use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::Instant,
};

use axum::{body::Body, response::Response};
use bytes::Bytes;
use http_body::{Body as HttpBody, Frame, SizeHint};
use metrics::{Counter, Gauge, Histogram};
use metrics_exporter_prometheus::{Matcher, PrometheusBuilder};
use serde_json::Value;

use crate::{config::RouterConfig, core::ConnectionMode};

pub const LATENCY_BUCKETS: &[f64] = &[
    0.05, 0.1, 0.25, 0.5, 1., 2., 3., 5., 8., 10., 15., 20., 30., 60., 120., 240., 480., 900.,
    1800.,
];
pub const TPOT_BUCKETS: &[f64] = &[
    0.001, 0.002, 0.005, 0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02, 0.022, 0.025, 0.028, 0.03,
    0.035, 0.04, 0.05, 0.075, 0.1, 0.2, 0.5, 1.,
];
// Bound the observer, not the response: large/unrecognized payloads still pass
// through unchanged. Missing usage never falls back to counting SSE frames.
const MAX_OBSERVATION_BYTES: usize = 1024 * 1024;

pub fn configure_buckets(mut builder: PrometheusBuilder) -> PrometheusBuilder {
    for name in ["llm_gateway_ttft_seconds", "llm_gateway_e2e_seconds"] {
        builder = builder
            .set_buckets_for_metric(Matcher::Full(name.into()), LATENCY_BUCKETS)
            .expect("valid PD latency buckets");
    }
    builder
        .set_buckets_for_metric(
            Matcher::Full("llm_gateway_request_tpot_seconds".into()),
            TPOT_BUCKETS,
        )
        .expect("valid PD TPOT buckets")
}

pub fn describe_metrics() {
    metrics::describe_counter!(
        "llm_gateway_requests_total",
        "HTTP PD requests by final downstream outcome"
    );
    metrics::describe_gauge!(
        "llm_gateway_inflight_requests",
        "HTTP PD requests including queued handlers and unconsumed response bodies"
    );
    metrics::describe_histogram!("llm_gateway_ttft_seconds", "Successful request ingress to first nonempty semantic or visible output observed at Gateway");
    metrics::describe_histogram!("llm_gateway_request_tpot_seconds", "Successful request first-to-last semantic interval / (final completion_tokens - 1); requires distinct output observation times");
    metrics::describe_histogram!(
        "llm_gateway_e2e_seconds",
        "Successful HTTP PD request ingress through downstream body completion"
    );
    metrics::describe_counter!(
        "llm_gateway_prompt_tokens_total",
        "Prompt tokens from final successful response usage only"
    );
    metrics::describe_counter!(
        "llm_gateway_output_tokens_total",
        "Completion tokens from final successful response usage only"
    );
    metrics::describe_counter!("llm_gateway_usage_missing_total", "Successful responses missing complete final token usage; TPOT requires known completion tokens");
}

pub struct PdMetrics {
    success: Counter,
    error: Counter,
    cancelled: Counter,
    inflight: Gauge,
    semantic: Histogram,
    visible: Histogram,
    tpot: Histogram,
    e2e: Histogram,
    prompt: Counter,
    output: Counter,
    missing: Counter,
}

impl PdMetrics {
    pub fn from_config(config: &RouterConfig) -> Option<Arc<Self>> {
        if std::env::var("SGLANG_GATEWAY_PD_METRICS").as_deref() != Ok("1")
            || !config.mode.is_pd_mode()
            || config.connection_mode != ConnectionMode::Http
        {
            return None;
        }
        // A single configured model label per deployment, with a bounded length.
        // Unknown/malicious request model IDs cannot create additional series.
        let model = std::env::var("SGLANG_GATEWAY_PD_METRICS_MODEL")
            .ok()
            .filter(|s| !s.trim().is_empty())
            .or_else(|| {
                config
                    .model_path
                    .as_deref()
                    .and_then(|s| s.trim_end_matches('/').rsplit('/').next())
                    .map(str::to_owned)
            })
            .unwrap_or_else(|| "unknown".into());
        Some(Self::new(&model.chars().take(128).collect::<String>()))
    }

    pub(crate) fn new(model: &str) -> Arc<Self> {
        let label = model.to_owned();
        let result = Arc::new(Self {
            success: metrics::counter!("llm_gateway_requests_total", "model_name" => label.clone(), "outcome" => "success"),
            error: metrics::counter!("llm_gateway_requests_total", "model_name" => label.clone(), "outcome" => "error"),
            cancelled: metrics::counter!("llm_gateway_requests_total", "model_name" => label.clone(), "outcome" => "cancelled"),
            inflight: metrics::gauge!("llm_gateway_inflight_requests", "model_name" => label.clone()),
            semantic: metrics::histogram!("llm_gateway_ttft_seconds", "model_name" => label.clone(), "first_event" => "semantic"),
            visible: metrics::histogram!("llm_gateway_ttft_seconds", "model_name" => label.clone(), "first_event" => "visible"),
            tpot: metrics::histogram!("llm_gateway_request_tpot_seconds", "model_name" => label.clone()),
            e2e: metrics::histogram!("llm_gateway_e2e_seconds", "model_name" => label.clone()),
            prompt: metrics::counter!("llm_gateway_prompt_tokens_total", "model_name" => label.clone()),
            output: metrics::counter!("llm_gateway_output_tokens_total", "model_name" => label.clone()),
            missing: metrics::counter!("llm_gateway_usage_missing_total", "model_name" => label),
        });
        result.inflight.set(0.);
        result
    }
}

#[derive(Default)]
struct OutputObservation {
    first_semantic: Option<Instant>,
    first_visible: Option<Instant>,
    last_semantic: Option<Instant>,
    prompt: Option<u64>,
    completion: Option<u64>,
    final_usage: bool,
    timing_incomplete: bool,
    terminal: bool,
    errored: bool,
    native_text: String,
}

fn nonempty(value: Option<&Value>) -> bool {
    value.and_then(Value::as_str).is_some_and(|s| !s.is_empty())
}

impl OutputObservation {
    fn json(&mut self, value: &Value, now: Instant) {
        if value.get("error").is_some_and(|v| !v.is_null())
            || value.get("type").and_then(Value::as_str) == Some("error")
        {
            self.errored = true;
        }
        if let Some(usage) = value.get("usage").filter(|v| v.is_object()) {
            // Replace rather than infer/accumulate: usage is cumulative, and the
            // last real usage object is authoritative, including explicit zero.
            self.final_usage = self.terminal
                || value.get("choices").is_none()
                || value
                    .get("choices")
                    .and_then(Value::as_array)
                    .is_some_and(|c| {
                        c.is_empty()
                            || c.iter()
                                .any(|v| v.get("finish_reason").is_some_and(|r| !r.is_null()))
                    });
            self.prompt = usage.get("prompt_tokens").and_then(Value::as_u64);
            self.completion = usage.get("completion_tokens").and_then(Value::as_u64);
        }
        let mut semantic = false;
        let mut visible = false;
        if let Some(choices) = value.get("choices").and_then(Value::as_array) {
            for choice in choices {
                if let Some(reason) = choice.get("finish_reason").filter(|v| !v.is_null()) {
                    self.terminal = true;
                    if matches!(reason.as_str(), Some("error" | "abort" | "cancelled")) {
                        self.errored = true;
                    }
                }
                let delta = choice.get("delta").or_else(|| choice.get("message"));
                if let Some(delta) = delta {
                    visible |= nonempty(delta.get("content"));
                    semantic |= nonempty(delta.get("reasoning_content"))
                        || nonempty(delta.get("reasoning"));
                    if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
                        visible |= calls
                            .iter()
                            .any(|c| nonempty(c.pointer("/function/arguments")));
                    }
                    visible |= nonempty(delta.pointer("/function_call/arguments"));
                }
                visible |= nonempty(choice.get("text"));
            }
        }
        // Native /generate has cumulative text and real token counts in
        // meta_info. Repeated text/usage-only final frames are not output.
        if let Some(meta) = value.get("meta_info") {
            if meta.get("prompt_tokens").is_some() || meta.get("completion_tokens").is_some() {
                self.final_usage = meta.get("finish_reason").is_some_and(|v| !v.is_null());
                self.prompt = meta.get("prompt_tokens").and_then(Value::as_u64);
                self.completion = meta.get("completion_tokens").and_then(Value::as_u64);
            }
            if let Some(reason) = meta.get("finish_reason").filter(|v| !v.is_null()) {
                self.terminal = true;
                if matches!(
                    reason.get("type").and_then(Value::as_str),
                    Some("abort" | "error")
                ) {
                    self.errored = true;
                }
            }
            if let Some(text) = value.get("text").and_then(Value::as_str) {
                if !text.is_empty() && text != self.native_text {
                    visible = true;
                    self.native_text.clear();
                    self.native_text.push_str(text);
                }
            }
        }
        semantic |= visible;
        if semantic {
            self.first_semantic.get_or_insert(now);
            self.last_semantic = Some(now);
        }
        if visible {
            self.first_visible.get_or_insert(now);
        }
    }

    fn tpot(&self) -> Option<f64> {
        if !self.final_usage || self.timing_incomplete {
            return None;
        }
        let tokens = self.completion.filter(|n| *n >= 2)?;
        let first = self.first_semantic?;
        let last = self.last_semantic?;
        (last > first).then(|| last.duration_since(first).as_secs_f64() / (tokens - 1) as f64)
    }
}

// Streaming parser accepts LF/CRLF and multi-line data; every original byte is
// passed through by ObservedBody, including comments and unknown event fields.
#[derive(Default)]
struct SseObserver {
    line: Vec<u8>,
    data: Vec<u8>,
    oversized_line: bool,
    oversized_event: bool,
}

impl SseObserver {
    fn feed(&mut self, mut bytes: &[u8], now: Instant, output: &mut OutputObservation) {
        while !bytes.is_empty() {
            let end = memchr::memchr(b'\n', bytes)
                .map(|i| i + 1)
                .unwrap_or(bytes.len());
            let segment = &bytes[..end];
            if self.line.len().saturating_add(segment.len()) > MAX_OBSERVATION_BYTES {
                self.line.clear();
                self.oversized_line = true;
                self.oversized_event = true;
                output.timing_incomplete = true;
                output.final_usage = false;
            }
            if !self.oversized_line {
                self.line.extend_from_slice(segment);
            }
            if segment.last() == Some(&b'\n') {
                if !self.oversized_line {
                    while matches!(self.line.last(), Some(b'\n' | b'\r')) {
                        self.line.pop();
                    }
                    if self.line.is_empty() {
                        if !self.oversized_event && !self.data.is_empty() {
                            if self.data.last() == Some(&b'\n') {
                                self.data.pop();
                            }
                            if self.data == b"[DONE]" {
                                output.terminal = true;
                            } else if let Ok(value) = serde_json::from_slice::<Value>(&self.data) {
                                output.json(&value, now);
                            } else if self.data.iter().any(|b| !b.is_ascii_whitespace()) {
                                output.timing_incomplete = true;
                                output.final_usage = false;
                            }
                        }
                        self.data.clear();
                        self.oversized_event = false;
                    } else if let Some(data) = self.line.strip_prefix(b"data:") {
                        let data = data.strip_prefix(b" ").unwrap_or(data);
                        if self.data.len().saturating_add(data.len()) + 1 <= MAX_OBSERVATION_BYTES
                            && !self.oversized_event
                        {
                            self.data.extend_from_slice(data);
                            self.data.push(b'\n');
                        } else {
                            self.data.clear();
                            self.oversized_event = true;
                            output.timing_incomplete = true;
                            output.final_usage = false;
                        }
                    }
                }
                self.line.clear();
                self.oversized_line = false;
            }
            bytes = &bytes[end..];
        }
    }
}

pub struct RequestObservation {
    metrics: Arc<PdMetrics>,
    start: Instant,
    output: OutputObservation,
    finished: bool,
}

impl RequestObservation {
    pub fn new(metrics: Arc<PdMetrics>, start: Instant) -> Self {
        metrics.inflight.increment(1.);
        Self {
            metrics,
            start,
            output: OutputObservation::default(),
            finished: false,
        }
    }

    pub fn fail(&mut self) {
        self.finish("error");
    }

    fn finish(&mut self, outcome: &str) {
        if self.finished {
            return;
        }
        self.finished = true;
        self.metrics.inflight.decrement(1.);
        match outcome {
            "success" => {
                self.metrics.success.increment(1);
                self.metrics.e2e.record(self.start.elapsed().as_secs_f64());
                if let Some(first) = self
                    .output
                    .first_semantic
                    .filter(|_| !self.output.timing_incomplete)
                {
                    self.metrics
                        .semantic
                        .record(first.duration_since(self.start).as_secs_f64());
                }
                if let Some(first) = self
                    .output
                    .first_visible
                    .filter(|_| !self.output.timing_incomplete)
                {
                    self.metrics
                        .visible
                        .record(first.duration_since(self.start).as_secs_f64());
                }
                if let Some(tpot) = self.output.tpot() {
                    self.metrics.tpot.record(tpot);
                }
                if self.output.final_usage {
                    if let Some(tokens) = self.output.prompt {
                        self.metrics.prompt.increment(tokens);
                    }
                    if let Some(tokens) = self.output.completion {
                        self.metrics.output.increment(tokens);
                    }
                }
                if !self.output.final_usage
                    || self.output.prompt.is_none()
                    || self.output.completion.is_none()
                {
                    self.metrics.missing.increment(1);
                }
            }
            "error" => {
                self.metrics.error.increment(1);
            }
            _ => {
                self.metrics.cancelled.increment(1);
            }
        }
    }

    pub fn wrap_response(self, response: Response) -> Response {
        let (parts, inner) = response.into_parts();
        let sse = parts
            .headers
            .get(http::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| v.starts_with("text/event-stream"));
        let mut observed = ObservedBody {
            inner,
            observation: self,
            sse: sse.then(SseObserver::default),
            json: Vec::new(),
            json_oversized: false,
            http_success: parts.status.is_success(),
        };
        // Empty responses may be dropped by HTTP without any poll_frame call.
        if observed.inner.is_end_stream() {
            observed.complete();
        }
        Response::from_parts(parts, Body::new(observed))
    }
}

impl Drop for RequestObservation {
    fn drop(&mut self) {
        self.finish("cancelled");
    }
}

struct ObservedBody {
    inner: Body,
    observation: RequestObservation,
    sse: Option<SseObserver>,
    json: Vec<u8>,
    json_oversized: bool,
    http_success: bool,
}

impl ObservedBody {
    fn complete(&mut self) {
        if self.observation.finished {
            return;
        }
        if self.sse.is_none() && !self.json_oversized {
            if let Ok(value) = serde_json::from_slice::<Value>(&self.json) {
                self.observation.output.json(&value, Instant::now());
                // Non-streaming body EOF makes its real usage authoritative.
                self.observation.output.final_usage =
                    value.get("usage").is_some_and(Value::is_object)
                        || value.get("meta_info").is_some_and(Value::is_object);
            }
        }
        let good = self.http_success
            && !self.observation.output.errored
            && (self.sse.is_none() || self.observation.output.terminal);
        self.observation
            .finish(if good { "success" } else { "error" });
    }
}

impl HttpBody for ObservedBody {
    type Data = Bytes;
    type Error = axum::Error;
    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, axum::Error>>> {
        let this = self.get_mut();
        let frame = Pin::new(&mut this.inner).poll_frame(cx);
        match &frame {
            Poll::Ready(Some(Ok(frame))) => {
                if let Some(data) = frame.data_ref() {
                    if let Some(sse) = &mut this.sse {
                        sse.feed(data, Instant::now(), &mut this.observation.output);
                    } else if !this.json_oversized {
                        if this.json.len().saturating_add(data.len()) <= MAX_OBSERVATION_BYTES {
                            this.json.extend_from_slice(data);
                        } else {
                            this.json.clear();
                            this.json_oversized = true;
                        }
                    }
                }
                if this.inner.is_end_stream() {
                    this.complete();
                }
            }
            Poll::Ready(Some(Err(_))) => {
                this.observation.fail();
            }
            Poll::Ready(None) => {
                this.complete();
            }
            Poll::Pending => {}
        }
        frame
    }
    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }
    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}

#[cfg(test)]
#[path = "pd_metrics/tests.rs"]
mod tests;
