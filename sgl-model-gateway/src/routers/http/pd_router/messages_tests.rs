// Included in pd_router::tests to reuse its worker and raw-HTTP fixtures.

fn native_request(stream: bool) -> Value {
    json!({"model":"test-model", "max_tokens":7, "stream":stream,
        "messages":[{"role":"user","content":"hello"}]})
}

async fn native_mock(status: StatusCode, sse: bool) -> EarlyChatMock {
    early_protocol_mock(status, sse, "/v1/messages").await
}

fn start_native(
    p: &EarlyChatMock,
    d: &EarlyChatMock,
    timeout: std::time::Duration,
) -> (
    Arc<PDRouter>,
    Arc<dyn Worker>,
    Arc<dyn Worker>,
    tokio::task::JoinHandle<Response>,
) {
    let mut router = create_test_pd_router();
    router.pd_prefill_stage_reservation = true;
    router.pd_chat_early_stream_timeout = timeout;
    router.retry_config.max_retries = 1;
    let (prefill, decode) = register_pd_workers(&router, p.url.clone(), d.url.clone());
    let router = Arc::new(router);
    let task_router = router.clone();
    let request = tokio::spawn(async move {
        task_router
            .route_messages(None, &native_request(true), None)
            .await
    });
    (router, prefill, decode, request)
}

const NATIVE_START: &str = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"role\":\"assistant\",\"content\":[],\"usage\":{\"input_tokens\":8,\"output_tokens\":0}}}\n\n";
const NATIVE_DELTA: &str = "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"path\\\":\\\"a\\\"}\"}}\n\n";
const NATIVE_STOP: &str = "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";

#[tokio::test]
async fn test_messages_native_stream_split_tool_json_and_terminal_waits_for_prefill() {
    use http_body_util::BodyExt;
    let mut p = native_mock(StatusCode::OK, true).await;
    let mut d = native_mock(StatusCode::OK, true).await;
    let (router, prefill, decode, request) =
        start_native(&p, &d, std::time::Duration::from_secs(30));
    early_wait(&mut p.entered).await.unwrap();
    early_wait(&mut d.entered).await.unwrap();
    d.headers();
    let response = early_wait(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let mut stream = response.into_body().into_data_stream();
    // Deliver real native events before P's headers, split at every byte.
    for event in [NATIVE_START, NATIVE_DELTA] {
        for byte in event.as_bytes() {
            d.data(bytes::Bytes::copy_from_slice(&[*byte]));
        }
        assert_eq!(early_wait(stream.next()).await.unwrap().unwrap(), event);
    }
    // Terminal output may be coalesced by the transport but must wait for P ack.
    d.data(NATIVE_STOP);
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(20), stream.next())
            .await
            .is_err()
    );
    p.headers();
    p.data(NATIVE_STOP);
    // Neither upstream sends HTTP EOF: native message_stop must close both.
    assert_eq!(
        early_wait(stream.next()).await.unwrap().unwrap(),
        NATIVE_STOP
    );
    assert!(early_wait(stream.next()).await.is_none());
    drop(stream);
    early_wait(&mut p.closed).await.unwrap();
    early_wait(&mut d.closed).await.unwrap();
    early_load_zero(&prefill).await;
    early_load_zero(&decode).await;
    assert_eq!(prefill.circuit_breaker().total_successes(), 1);
    assert_eq!(decode.circuit_breaker().total_successes(), 1);
    assert!(router.reservations.views(&HashMap::new()).0.is_empty());
}

#[tokio::test]
async fn test_messages_errors_disconnect_and_eof_cancel_both_upstreams() {
    use http_body_util::BodyExt;
    for ending in ["error", "eof", "disconnect", "timeout", "prefill_error"] {
        let mut p = native_mock(StatusCode::OK, true).await;
        let mut d = native_mock(StatusCode::OK, true).await;
        let budget = if ending == "timeout" {
            std::time::Duration::from_millis(100)
        } else {
            std::time::Duration::from_secs(30)
        };
        let (router, prefill, decode, request) = start_native(&p, &d, budget);
        early_wait(&mut p.entered).await.unwrap();
        early_wait(&mut d.entered).await.unwrap();
        d.headers();
        let mut stream = early_wait(request)
            .await
            .unwrap()
            .into_body()
            .into_data_stream();
        d.data(NATIVE_START);
        assert_eq!(
            early_wait(stream.next()).await.unwrap().unwrap(),
            NATIVE_START
        );
        match ending {
            "error" => d.data("event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"busy\"}}\n\n"),
            "eof" => d.end(),
            "prefill_error" => { p.headers(); p.data("event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"api_error\",\"message\":\"P failed\"}}\n\n"); },
            _ => {},
        }
        if ending != "disconnect" {
            let bytes = early_wait(stream.next()).await.unwrap().unwrap();
            let text = std::str::from_utf8(&bytes).unwrap();
            assert!(text.starts_with("event: error\ndata: "), "{text}");
            let value: Value =
                serde_json::from_str(text.trim().split_once("data: ").unwrap().1).unwrap();
            assert_eq!(value["type"], "error");
            assert!(value["error"]["message"].is_string());
            assert!(!text.contains("message_stop"));
            assert!(early_wait(stream.next()).await.is_none());
        }
        drop(stream);
        early_wait(&mut p.closed).await.unwrap();
        early_wait(&mut d.closed).await.unwrap();
        early_load_zero(&prefill).await;
        early_load_zero(&decode).await;
        assert!(router.reservations.views(&HashMap::new()).0.is_empty());
        assert_eq!(
            decode.circuit_breaker().total_failures(),
            u64::from(matches!(ending, "error" | "eof"))
        );
        assert_eq!(
            prefill.circuit_breaker().total_failures(),
            u64::from(ending == "prefill_error")
        );
    }
}

#[tokio::test]
async fn test_messages_http_errors_preserve_native_status_and_cancel_peer() {
    use http_body_util::BodyExt;
    for failed_prefill in [true, false] {
        let mut p = native_mock(
            if failed_prefill {
                StatusCode::TOO_MANY_REQUESTS
            } else {
                StatusCode::OK
            },
            false,
        )
        .await;
        let mut d = native_mock(
            if failed_prefill {
                StatusCode::OK
            } else {
                StatusCode::UNAUTHORIZED
            },
            false,
        )
        .await;
        let (_, prefill, decode, request) =
            start_native(&p, &d, std::time::Duration::from_secs(30));
        early_wait(&mut p.entered).await.unwrap();
        early_wait(&mut d.entered).await.unwrap();
        let failed = if failed_prefill { &mut p } else { &mut d };
        failed.headers();
        failed.data("{\"type\":\"error\",\"error\":{\"type\":\"rate_limit_error\",\"message\":\"try later\"}}");
        failed.end();
        let response = early_wait(request).await.unwrap();
        assert_eq!(
            response.status(),
            if failed_prefill {
                StatusCode::TOO_MANY_REQUESTS
            } else {
                StatusCode::UNAUTHORIZED
            }
        );
        assert_eq!(response.headers()[CONTENT_TYPE], "application/json");
        let value: Value =
            serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes())
                .unwrap();
        assert_eq!(value["error"]["message"], "try later");
        early_wait(&mut p.closed).await.unwrap();
        early_wait(&mut d.closed).await.unwrap();
        early_load_zero(&prefill).await;
        early_load_zero(&decode).await;
        assert_eq!(prefill.circuit_breaker().total_failures(), 0);
        assert_eq!(decode.circuit_breaker().total_failures(), 0);
    }
}

#[tokio::test]
async fn test_messages_raw_fields_metadata_tokenize_and_count_without_inference() {
    use axum::{routing::post, Json};
    use http_body_util::BodyExt;
    for media in [false, true] {
        let captured = Arc::new(Mutex::new(Vec::<(&'static str, Value, HeaderMap)>::new()));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let mut app = axum::Router::new();
        for endpoint in [
            "/v1/messages",
            "/v1/messages/tokenize",
            "/v1/messages/count_tokens",
        ] {
            let captured = captured.clone();
            app = app.route(endpoint, post(move |headers: HeaderMap, Json(body): Json<Value>| {
                captured.lock().unwrap().push((endpoint, body, headers));
                async move { Json(match endpoint {
                    "/v1/messages/tokenize" => json!({"supported":true,"version":1,"tokens":[10,20,30],"count":3}),
                    "/v1/messages/count_tokens" => json!({"input_tokens":3}),
                    _ => json!({"type":"message","content":[{"type":"text","text":"ok"}],"usage":{"input_tokens":3,"output_tokens":1}}),
                }) }
            }));
        }
        let backend = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let _abort = AbortStreamOnDrop(backend.abort_handle());
        let mut router = create_test_pd_router();
        let mut affinity = ChatAffinity::default();
        affinity.reuse_token_ids = true;
        router.chat_affinity = Some(affinity);
        let prefill: Arc<dyn Worker> = Arc::new(
            DPAwareWorkerBuilder::new(url.clone(), 1, 2)
                .worker_type(WorkerType::Prefill {
                    bootstrap_port: Some(9000),
                })
                .build(),
        );
        let decode: Arc<dyn Worker> = Arc::new(
            DPAwareWorkerBuilder::new(url, 0, 2)
                .worker_type(WorkerType::Decode)
                .build(),
        );
        router.worker_registry.register(prefill.clone());
        router.worker_registry.register(decode.clone());
        let mut body = native_request(false);
        body["system"] =
            json!([{"type":"text","text":"system","cache_control":{"type":"ephemeral"}}]);
        body["tools"] = json!([{"name":"read","input_schema":{"type":"object","properties":{"path":{"type":"string"}}}}]);
        body["thinking"] = json!({"type":"enabled","budget_tokens":3});
        body["cache_salt"] = json!("tenant-a");
        body["vendor_extension"] = json!({"keep":true});
        body["input_ids"] = json!([999]); // never trust client pretokenization
        body["routed_dp_rank"] = json!(99);
        body["messages"] = json!([
            {"role":"assistant","content":[{"type":"thinking","thinking":"reason","signature":"sig"},{"type":"tool_use","id":"tool_1","name":"read","input":{"path":"a","type":"image"}}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"tool_1","content":[{"type":"text","text":"result"}]}]}
        ]);
        if media {
            body["messages"] = json!([
                {"role":"assistant","content":[{"type":"thinking","thinking":"reason","signature":"sig"},{"type":"tool_use","id":"tool_1","name":"read","input":{"path":"a"}}]},
                {"role":"user","content":[{"type":"tool_result","tool_use_id":"tool_1","content":[{"type":"text","text":"result"}]},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"aGVsbG8="}}]}
            ]);
        }
        let mut headers = HeaderMap::new();
        headers.insert("anthropic-version", "2023-06-01".parse().unwrap());
        headers.insert("anthropic-beta", "test-beta".parse().unwrap());
        headers.insert("x-api-key", "native-key".parse().unwrap());
        let response = router.route_messages(Some(&headers), &body, None).await;
        assert_eq!(response.status(), StatusCode::OK);
        let result: Value =
            serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes())
                .unwrap();
        assert_eq!(result["type"], "message");
        {
            let requests = captured.lock().unwrap();
            assert_eq!(
                requests
                    .iter()
                    .filter(|(path, _, _)| *path == "/v1/messages/tokenize")
                    .count(),
                usize::from(!media)
            );
            let generations: Vec<_> = requests
                .iter()
                .filter(|(path, _, _)| *path == "/v1/messages")
                .collect();
            assert_eq!(generations.len(), 2);
            for (_, forwarded, hdrs) in &generations {
                for (key, value) in body.as_object().unwrap() {
                    if !matches!(key.as_str(), "input_ids" | "routed_dp_rank") {
                        assert_eq!(&forwarded[key], value, "{key}");
                    }
                }
                assert_eq!(
                    forwarded["input_ids"],
                    if media {
                        Value::Null
                    } else {
                        json!([10, 20, 30])
                    }
                );
                assert_eq!(forwarded["disagg_prefill_dp_rank"], 1);
                assert_eq!(forwarded["bootstrap_port"], 9000);
                assert!(forwarded["bootstrap_host"].is_string());
                assert_eq!(hdrs["anthropic-version"], "2023-06-01");
                assert_eq!(hdrs["anthropic-beta"], "test-beta");
                assert_eq!(hdrs["authorization"], "Bearer native-key");
                if forwarded["routed_dp_rank"] == 1 {
                    assert_eq!(forwarded["disagg_decode_dp_rank"], 0);
                } else {
                    assert!(forwarded.get("disagg_decode_dp_rank").is_none());
                }
            }
            assert_eq!(
                generations[0].1["bootstrap_room"],
                generations[1].1["bootstrap_room"]
            );
            let mut ranks = generations
                .iter()
                .map(|entry| entry.1["routed_dp_rank"].as_u64().unwrap())
                .collect::<Vec<_>>();
            ranks.sort();
            assert_eq!(ranks, [0, 1]);
        }
        captured.lock().unwrap().clear();
        let before = (
            prefill.circuit_breaker().total_successes(),
            decode.circuit_breaker().total_successes(),
        );
        body.as_object_mut().unwrap().remove("max_tokens");
        let response = router
            .route_messages_count_tokens(Some(&headers), &body, None)
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        let result: Value =
            serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes())
                .unwrap();
        assert_eq!(result, json!({"input_tokens":3}));
        let requests = captured.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].0, "/v1/messages/count_tokens");
        assert_eq!(requests[0].1, body);
        assert_eq!(
            before,
            (
                prefill.circuit_breaker().total_successes(),
                decode.circuit_breaker().total_successes()
            )
        );
        assert_eq!(prefill.load(), 0);
        assert_eq!(decode.load(), 0);
        assert!(router.reservations.views(&HashMap::new()).0.is_empty());
    }
}

#[tokio::test]
async fn test_messages_pending_headers_timeout_cancels_both_requests() {
    let mut p = native_mock(StatusCode::OK, true).await;
    let mut d = native_mock(StatusCode::OK, true).await;
    let (router, prefill, decode, request) =
        start_native(&p, &d, std::time::Duration::from_millis(250));
    early_wait(&mut p.entered).await.unwrap();
    early_wait(&mut d.entered).await.unwrap();
    let response = early_wait(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
    drop(response);
    early_wait(&mut p.closed).await.unwrap();
    early_wait(&mut d.closed).await.unwrap();
    early_load_zero(&prefill).await;
    early_load_zero(&decode).await;
    assert!(router.reservations.views(&HashMap::new()).0.is_empty());
    assert_eq!(prefill.circuit_breaker().total_failures(), 0);
    assert_eq!(decode.circuit_breaker().total_failures(), 0);
}
