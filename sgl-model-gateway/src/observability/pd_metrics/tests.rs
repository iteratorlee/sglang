use super::*;
use axum::{http::StatusCode, response::IntoResponse};
use futures_util::{stream, StreamExt};
use http_body_util::BodyExt;
use metrics_exporter_prometheus::PrometheusHandle;
use serde_json::json;
use std::{convert::Infallible, time::Duration};

fn recorder() -> (Arc<PdMetrics>, PrometheusHandle) {
    let recorder = configure_buckets(PrometheusBuilder::new()).build_recorder();
    let handle = recorder.handle();
    let metrics = metrics::with_local_recorder(&recorder, || PdMetrics::new("fixture-model"));
    (metrics, handle)
}
fn series(handle: &PrometheusHandle, name: &str, labels: &[&str]) -> f64 {
    let rendered = handle.render();
    rendered
        .lines()
        .find(|l| l.starts_with(&format!("{name}{{")) && labels.iter().all(|v| l.contains(v)))
        .and_then(|l| l.rsplit(' ').next())
        .unwrap_or("0")
        .parse()
        .unwrap()
}
fn observe(output: &mut OutputObservation, json: Value, when: Instant) {
    output.json(&json, when);
}
fn sse_body(chunks: Vec<Vec<u8>>) -> Body {
    Body::from_stream(stream::iter(
        chunks
            .into_iter()
            .map(|v| Ok::<_, Infallible>(Bytes::from(v))),
    ))
}
fn wrap(metrics: Arc<PdMetrics>, body: Body, sse: bool, status: StatusCode) -> Response {
    let mut response = (status, body).into_response();
    if sse {
        response.headers_mut().insert(
            http::header::CONTENT_TYPE,
            "text/event-stream".parse().unwrap(),
        );
    }
    RequestObservation::new(metrics, Instant::now() - Duration::from_secs(2))
        .wrap_response(response)
}

#[test]
fn semantic_visible_and_real_usage_timing() {
    let start = Instant::now();
    let mut output = OutputObservation::default();
    observe(
        &mut output,
        json!({"choices":[{"delta":{"role":"assistant","content":""}}]}),
        start,
    );
    assert!(output.first_semantic.is_none());
    observe(
        &mut output,
        json!({"choices":[{"delta":{"reasoning_content":"think"}}]}),
        start + Duration::from_secs(1),
    );
    assert!(output.first_visible.is_none());
    observe(
        &mut output,
        json!({"choices":[{"delta":{"tool_calls":[{"id":"foo","function":{"name":"bar","arguments":""}}]}}]}),
        start + Duration::from_secs(2),
    );
    assert!(output.first_visible.is_none());
    observe(
        &mut output,
        json!({"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"{}"}}]}}]}),
        start + Duration::from_secs(3),
    );
    observe(
        &mut output,
        json!({"choices":[],"usage":{"prompt_tokens":128000,"completion_tokens":101}}),
        start + Duration::from_secs(4),
    );
    assert_eq!(output.first_visible, Some(start + Duration::from_secs(3)));
    assert_eq!(output.last_semantic, Some(start + Duration::from_secs(3)));
    assert_eq!(output.tpot(), Some(0.02));
}

#[test]
fn no_usage_no_zero_one_or_single_timestamp_tpot() {
    let start = Instant::now();
    let mut out = OutputObservation::default();
    observe(
        &mut out,
        json!({"choices":[{"delta":{"content":"answer"}}]}),
        start,
    );
    for n in [0, 1, 2, 100] {
        observe(
            &mut out,
            json!({"choices":[],"usage":{"prompt_tokens":8,"completion_tokens":n}}),
            start,
        );
        assert_eq!(out.tpot(), None);
    }
    observe(
        &mut out,
        json!({"choices":[{"delta":{"content":"next"}}]}),
        start + Duration::from_secs(1),
    );
    out.completion = None;
    assert_eq!(out.tpot(), None);
}

#[test]
fn cumulative_native_text_final_usage_no_metadata_false_token() {
    let now = Instant::now();
    let mut out = OutputObservation::default();
    observe(
        &mut out,
        json!({"text":"a","meta_info":{"prompt_tokens":10,"completion_tokens":1,"finish_reason":null}}),
        now,
    );
    observe(
        &mut out,
        json!({"text":"ab","meta_info":{"prompt_tokens":10,"completion_tokens":2,"finish_reason":null}}),
        now + Duration::from_millis(20),
    );
    assert_eq!(out.tpot(), None); // provisional counts are not final usage
    observe(
        &mut out,
        json!({"text":"ab","meta_info":{"prompt_tokens":10,"completion_tokens":2,"finish_reason":{"type":"length"}}}),
        now + Duration::from_secs(1),
    );
    assert_eq!(out.tpot(), Some(0.02));
    assert_eq!(out.last_semantic, Some(now + Duration::from_millis(20)));
}

#[test]
fn bytewise_sse_crlf_multiline_and_metadata() {
    let bytes=b":keepalive\r\nevent: message\r\ndata: {\"choices\": [\r\ndata: {\"delta\":{\"content\":\"hi\"}}]}\r\n\r\ndata: [DONE]\r\n\r\n";
    let mut parser = SseObserver::default();
    let mut out = OutputObservation::default();
    let now = Instant::now();
    for b in bytes {
        parser.feed(&[*b], now, &mut out);
    }
    assert!(out.terminal);
    assert_eq!(out.first_visible, Some(now));
    assert_eq!(out.tpot(), None);
}

#[test]
fn coalesced_events_have_one_timestamp_and_no_tpot() {
    let mut parser = SseObserver::default();
    let mut out = OutputObservation::default();
    parser.feed(b"data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"b\"}}]}\n\ndata: {\"choices\":[],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":2}}\n\ndata: [DONE]\n\n",Instant::now(),&mut out);
    assert!(out.terminal);
    assert_eq!(out.tpot(), None);
}

#[tokio::test]
async fn original_bytes_headers_and_inflight_through_eof() {
    let (metrics, handle) = recorder();
    let bytes=b": ping\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: {\"choices\":[],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":2}}\n\ndata: [DONE]\n\n";
    let response = wrap(
        metrics,
        sse_body(bytes.chunks(3).map(|c| c.to_vec()).collect()),
        true,
        StatusCode::OK,
    );
    assert_eq!(series(&handle, "llm_gateway_inflight_requests", &[]), 1.);
    let body = response.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(body.as_ref(), bytes);
    assert_eq!(series(&handle, "llm_gateway_inflight_requests", &[]), 0.);
    assert_eq!(
        series(&handle, "llm_gateway_requests_total", &["success"]),
        1.
    );
    assert_eq!(series(&handle, "llm_gateway_prompt_tokens_total", &[]), 4.);
    assert_eq!(series(&handle, "llm_gateway_output_tokens_total", &[]), 2.);
    assert_eq!(series(&handle, "llm_gateway_usage_missing_total", &[]), 0.);
    assert!(handle.render().contains("le=\"1800\""));
}

#[tokio::test]
async fn errors_and_truncated_stream_are_not_success() {
    for (status, bytes) in [
        (
            StatusCode::OK,
            b"data: {\"error\":{\"message\":\"late failure\"}}\n\ndata: [DONE]\n\n".as_slice(),
        ),
        (StatusCode::BAD_GATEWAY, b"upstream failed".as_slice()),
        (
            StatusCode::OK,
            b"data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n".as_slice(),
        ),
    ] {
        let (m, h) = recorder();
        let response = wrap(m, sse_body(vec![bytes.to_vec()]), true, status);
        response.into_body().collect().await.unwrap();
        assert_eq!(series(&h, "llm_gateway_requests_total", &["error"]), 1.);
        assert_eq!(series(&h, "llm_gateway_requests_total", &["success"]), 0.);
        assert_eq!(series(&h, "llm_gateway_inflight_requests", &[]), 0.);
    }
}

#[tokio::test]
async fn downstream_cancel_and_handler_cancel_are_counted_once() {
    let (m, h) = recorder();
    drop(RequestObservation::new(m.clone(), Instant::now()));
    let stream = stream::once(async {
        Ok::<_, Infallible>(Bytes::from_static(
            b"data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n",
        ))
    })
    .chain(stream::pending());
    let mut body = wrap(m, Body::from_stream(stream), true, StatusCode::OK).into_body();
    body.frame().await.unwrap().unwrap();
    assert_eq!(series(&h, "llm_gateway_inflight_requests", &[]), 1.);
    drop(body);
    assert_eq!(series(&h, "llm_gateway_requests_total", &["cancelled"]), 2.);
    assert_eq!(series(&h, "llm_gateway_requests_total", &["success"]), 0.);
    assert_eq!(series(&h, "llm_gateway_inflight_requests", &[]), 0.);
}

#[tokio::test]
async fn nonstreaming_is_end_to_end_and_no_inferred_token_times() {
    let (m, h) = recorder();
    let raw=serde_json::to_vec(&json!({"choices":[{"message":{"content":"answer"},"finish_reason":"stop"}],"usage":{"prompt_tokens":8,"completion_tokens":3}})).unwrap();
    let response = wrap(m, Body::from(raw.clone()), false, StatusCode::OK);
    assert_eq!(
        response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes()
            .as_ref(),
        raw
    );
    assert_eq!(
        series(&h, "llm_gateway_request_tpot_seconds_count", &[]),
        0.
    );
    assert_eq!(
        series(&h, "llm_gateway_ttft_seconds_count", &["semantic"]),
        1.
    );
    assert_eq!(series(&h, "llm_gateway_e2e_seconds_count", &[]), 1.);
    assert_eq!(series(&h, "llm_gateway_requests_total", &["success"]), 1.);
}

#[tokio::test]
async fn missing_usage_counted_without_forcing_request_usage() {
    let (m, h) = recorder();
    let response=wrap(m,sse_body(vec![b"data: {\"choices\":[{\"delta\":{\"content\":\"answer\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n".to_vec()]),true,StatusCode::OK);
    response.into_body().collect().await.unwrap();
    assert_eq!(series(&h, "llm_gateway_usage_missing_total", &[]), 1.);
    assert_eq!(
        series(&h, "llm_gateway_request_tpot_seconds_count", &[]),
        0.
    );
}

#[tokio::test]
async fn transport_error_is_not_a_cancel_or_success() {
    let (m, h) = recorder();
    let body = Body::from_stream(stream::iter([Err::<Bytes, _>(std::io::Error::other(
        "fixture",
    ))]));
    assert!(wrap(m, body, true, StatusCode::OK)
        .into_body()
        .collect()
        .await
        .is_err());
    assert_eq!(series(&h, "llm_gateway_requests_total", &["error"]), 1.);
    assert_eq!(series(&h, "llm_gateway_requests_total", &["cancelled"]), 0.);
}

#[test]
fn parser_bounded_and_recovers_after_oversized_event() {
    let mut parser = SseObserver::default();
    let mut out = OutputObservation::default();
    parser.feed(
        &vec![b'x'; MAX_OBSERVATION_BYTES + 128],
        Instant::now(),
        &mut out,
    );
    assert!(parser.line.len() <= MAX_OBSERVATION_BYTES);
    parser.feed(b"\n\ndata: [DONE]\n\n", Instant::now(), &mut out);
    assert!(out.terminal);
    assert!(out.completion.is_none());
}

#[test]
fn histograms_match_contract_and_no_per_request_labels() {
    let (m, h) = recorder();
    let start = Instant::now() - Duration::from_secs(3);
    let mut req = RequestObservation::new(m, start);
    observe(
        &mut req.output,
        json!({"choices":[{"delta":{"content":"a"}}]}),
        start + Duration::from_secs(1),
    );
    observe(
        &mut req.output,
        json!({"choices":[{"delta":{"content":"b"}}]}),
        start + Duration::from_millis(1020),
    );
    observe(
        &mut req.output,
        json!({"choices":[],"usage":{"prompt_tokens":128000,"completion_tokens":2}}),
        start + Duration::from_secs(2),
    );
    req.finish("success");
    drop(req);
    assert_eq!(
        series(&h, "llm_gateway_request_tpot_seconds_sum", &[]),
        0.02
    );
    let text = h.render();
    assert!(text.contains("le=\"0.018\""));
    assert!(text.contains("le=\"0.022\""));
    assert!(!text.contains("request_id="));
    assert!(!text.contains("session_id="));
}

#[test]
fn standalone_final_usage_and_malformed_event_do_not_infer_timing() {
    let now = Instant::now();
    let mut out = OutputObservation::default();
    observe(
        &mut out,
        json!({"choices":[{"delta":{"content":"a"}}]}),
        now,
    );
    observe(
        &mut out,
        json!({"choices":[{"delta":{"content":"b"}}]}),
        now + Duration::from_millis(20),
    );
    observe(
        &mut out,
        json!({"usage":{"prompt_tokens":10,"completion_tokens":2}}),
        now + Duration::from_secs(1),
    );
    assert_eq!(out.tpot(), Some(0.02));
    let mut parser = SseObserver::default();
    parser.feed(b"data: {broken\n\n", now + Duration::from_secs(2), &mut out);
    observe(
        &mut out,
        json!({"usage":{"prompt_tokens":10,"completion_tokens":3}}),
        now + Duration::from_secs(3),
    );
    assert!(out.final_usage);
    assert!(out.timing_incomplete);
    assert_eq!(out.tpot(), None);
}
