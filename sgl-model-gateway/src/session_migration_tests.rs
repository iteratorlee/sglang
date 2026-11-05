use super::*;
use axum::{routing::post, Router};
use futures_util::{Stream, StreamExt};
use std::convert::Infallible;
use tower::ServiceExt;

fn front(path: PathBuf) -> Arc<SessionHandoff> {
    SessionHandoff::open(
        "http://127.0.0.1:1".into(),
        path,
        2 * 1024 * 1024,
        AuthConfig { api_key: None },
    )
    .unwrap()
}
fn enable(f: &SessionHandoff, budget: u64) {
    let mut r = f.migration.lock();
    r.config = MigrationConfig {
        enabled: true,
        new_sessions_bps: 10000,
        legacy_sessions_bps: 10000,
        max_switches: budget,
        ..Default::default()
    };
    r.started = Instant::now() - Duration::from_secs(120);
}
fn age(f: &SessionHandoff, key: &str) {
    let mut r = f.migration.lock();
    r.last_assignment = None;
    let activity = r.sessions.get_mut(key).unwrap();
    activity.finished = Some(Instant::now() - Duration::from_secs(120));
    activity.last_cancelled = activity
        .last_cancelled
        .map(|_| Instant::now() - Duration::from_secs(120));
}
async fn drain(response: Response) -> bytes::Bytes {
    to_bytes(response.into_body(), 2 * 1024 * 1024)
        .await
        .unwrap()
}

#[test]
fn rich_full_history_is_supported_without_small_first_turn_heuristic() {
    let rich = json!({"metadata":{"session_id":"s"},"tools":[{"name":"edit"}],"messages":[
        {"role":"user","content":"x".repeat(MAX_FIRST_TURN+1)},
        {"role":"assistant","content":[{"type":"tool_use","id":"t","name":"edit","input":{}}]},
        {"role":"user","content":[{"type":"tool_result","tool_use_id":"t","content":"ok"},{"type":"image","source":{}}]}]});
    let bytes = serde_json::to_vec(&rich).unwrap();
    assert!(!bounded_first_turn(&bytes));
    assert!(full_chat_context(&bytes, "/v1/messages"));
    assert!(!full_chat_context(&bytes, "/generate"));
    for field in ["session_params", "previous_response_id", "conversation"] {
        let mut stateful = rich.clone();
        stateful[field] = json!("state");
        assert!(!full_chat_context(
            &serde_json::to_vec(&stateful).unwrap(),
            "/v1/messages"
        ));
    }
    assert!(!full_chat_context(br#"{"messages":[]}"#, "/v1/messages"));
    assert!(!MigrationConfig {
        max_cold_inflight: 0,
        ..Default::default()
    }
    .valid());
    assert!(MigrationConfig {
        quiet_secs: 0,
        ..Default::default()
    }
    .valid());
}

#[tokio::test]
async fn active_response_and_recent_cancellation_prevent_migration() {
    let dir = tempfile::tempdir().unwrap();
    let f = front(dir.path().join("pins.json"));
    let key = session_key("active-session").unwrap();
    let (_, _, old) = f.choose(Some(key.clone()), None, false, 12).await.unwrap();
    enable(&f, 2);
    let (destination, _, parallel) = f.choose(Some(key.clone()), None, true, 20).await.unwrap();
    assert_eq!(destination, Destination::Legacy);
    assert_eq!(f.migration.lock().sessions[&key].active, 2);
    drop(parallel);
    drop(old);
    let (destination, _, recent) = f.choose(Some(key.clone()), None, true, 20).await.unwrap();
    assert_eq!(destination, Destination::Legacy);
    drop(recent);
    age(&f, &key);
    let (destination, reason, moved) = f.choose(Some(key.clone()), None, true, 30).await.unwrap();
    assert_eq!(destination, Destination::Candidate);
    assert_eq!(reason, "migrated");
    assert_eq!(f.migration.lock().migrated, 1);
    drop(moved);
    assert_eq!(f.store.lock().await.pins[&key], Destination::Candidate);
}

struct PendingBody(bool);
impl Stream for PendingBody {
    type Item = Result<bytes::Bytes, Infallible>;
    fn poll_next(mut self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if !self.0 {
            self.0 = true;
            Poll::Ready(Some(Ok(bytes::Bytes::from_static(b"data: first\n\n"))))
        } else {
            Poll::Pending
        }
    }
}
#[tokio::test]
async fn streaming_lease_survives_headers_and_cancellation_releases_it() {
    let dir = tempfile::tempdir().unwrap();
    let f = front(dir.path().join("pins.json"));
    let key = session_key("streaming-session").unwrap();
    let (_, _, lease) = f.choose(Some(key.clone()), None, false, 7).await.unwrap();
    let response = wrap(Response::new(Body::from_stream(PendingBody(false))), lease);
    assert_eq!(f.migration.lock().active_legacy, 1);
    let mut stream = response.into_body().into_data_stream();
    assert_eq!(stream.next().await.unwrap().unwrap(), "data: first\n\n");
    assert_eq!(f.migration.lock().sessions[&key].active, 1);
    drop(stream);
    assert_eq!(f.migration.lock().active_legacy, 0);
    assert_eq!(f.migration.lock().sessions[&key].active, 0);
    assert!(f.migration.lock().sessions[&key].finished.is_some());
}

#[tokio::test]
async fn cold_budget_rate_and_finite_switch_limit_are_independent() {
    let dir = tempfile::tempdir().unwrap();
    let f = front(dir.path().join("pins.json"));
    enable(&f, 2);
    let a = session_key("cold-one").unwrap();
    let b = session_key("cold-two").unwrap();
    let (_, _, first) = f.choose(Some(a.clone()), None, true, 100).await.unwrap();
    assert_eq!(f.migration.lock().cold_inflight, 1);
    let (same, _, shared) = f.choose(Some(a.clone()), None, true, 100).await.unwrap();
    assert_eq!(same, Destination::Candidate);
    assert_eq!(f.migration.lock().cold_inflight, 1);
    let (d, _, other) = f.choose(Some(b.clone()), None, true, 100).await.unwrap();
    assert_eq!(d, Destination::Legacy);
    drop(other);
    drop(first);
    assert_eq!(f.migration.lock().cold_inflight, 1);
    drop(shared);
    assert_eq!(f.migration.lock().cold_inflight, 0);
    // Freeing cold capacity alone does not bypass the assignment interval.
    age(&f, &b);
    f.migration.lock().last_assignment = Some(Instant::now());
    let (d, _, wait) = f.choose(Some(b.clone()), None, true, 100).await.unwrap();
    assert_eq!(d, Destination::Legacy);
    drop(wait);
    age(&f, &b);
    let (d, _, second) = f.choose(Some(b.clone()), None, true, 100).await.unwrap();
    assert_eq!(d, Destination::Candidate);
    drop(second);
    f.migration.lock().last_assignment = None;
    let (d, _, over) = f
        .choose(session_key("over-budget"), None, true, 100)
        .await
        .unwrap();
    assert_eq!(d, Destination::Legacy);
    drop(over);
    let runtime = f.migration.lock();
    assert_eq!(runtime.enrolled + runtime.migrated, 2);
}

#[tokio::test]
async fn body_eof_marks_candidate_warm_and_pause_never_repins_it() {
    let dir = tempfile::tempdir().unwrap();
    let f = front(dir.path().join("pins.json"));
    enable(&f, 1);
    let key = session_key("warm-session").unwrap();
    let (_, _, lease) = f.choose(Some(key.clone()), None, true, 123).await.unwrap();
    assert_eq!(
        drain(wrap(Response::new(Body::from("done")), lease)).await,
        "done"
    );
    assert!(f.migration.lock().sessions[&key].warmed);
    assert_eq!(f.migration.lock().cold_inflight, 0);
    f.migration.lock().config.enabled = false;
    let (d, _, followup) = f.choose(Some(key.clone()), None, false, 0).await.unwrap();
    assert_eq!(d, Destination::Candidate);
    assert!(!followup.cold);
    drop(followup);
    assert_eq!(f.store.lock().await.pins[&key], Destination::Candidate);
}

#[tokio::test]
async fn imported_legacy_requires_an_observed_completed_request_before_move() {
    let dir = tempfile::tempdir().unwrap();
    let f = front(dir.path().join("pins.json"));
    let key = session_key("imported").unwrap();
    assert_eq!(
        import_pins(
            State(f.clone()),
            Json(BTreeMap::from([(key.clone(), Destination::Legacy)]))
        )
        .await
        .status(),
        StatusCode::OK
    );
    enable(&f, 1);
    let (d, _, initial) = f.choose(Some(key.clone()), None, true, 10).await.unwrap();
    assert_eq!(d, Destination::Legacy);
    drop(initial);
    age(&f, &key);
    let (d, _, moved) = f.choose(Some(key.clone()), None, true, 10).await.unwrap();
    assert_eq!(d, Destination::Candidate);
    drop(moved);
}

#[tokio::test]
async fn import_is_additive_atomic_and_unavailable_during_migration() {
    let dir = tempfile::tempdir().unwrap();
    let f = front(dir.path().join("pins.json"));
    let a = session_key("import-a").unwrap();
    let b = session_key("import-b").unwrap();
    let initial = BTreeMap::from([(a.clone(), Destination::Legacy)]);
    assert_eq!(
        import_pins(State(f.clone()), Json(initial.clone()))
            .await
            .status(),
        StatusCode::OK
    );
    assert_eq!(
        import_pins(State(f.clone()), Json(initial)).await.status(),
        StatusCode::OK
    );
    assert_eq!(
        import_pins(
            State(f.clone()),
            Json(BTreeMap::from([
                (a.clone(), Destination::Candidate),
                (b.clone(), Destination::Legacy)
            ]))
        )
        .await
        .status(),
        StatusCode::CONFLICT
    );
    assert!(!f.store.lock().await.pins.contains_key(&b));
    enable(&f, 1);
    assert_eq!(
        import_pins(State(f.clone()), Json(BTreeMap::new()))
            .await
            .status(),
        StatusCode::CONFLICT
    );
}

#[tokio::test]
async fn complete_rich_payload_is_enrolled_unchanged_by_real_middleware() {
    let dir = tempfile::tempdir().unwrap();
    let f = front(dir.path().join("pins.json"));
    enable(&f, 1);
    let app = Router::new()
        .route(
            "/v1/messages",
            post(|request: Request| async move {
                assert_eq!(request.headers()[CANDIDATE_HEADER], "1");
                let mut response = Response::new(request.into_body());
                response
                    .headers_mut()
                    .insert("x-destination", HeaderValue::from_static("candidate"));
                response
            }),
        )
        .layer(axum::middleware::from_fn_with_state(f.clone(), dispatch));
    let raw=json!({"metadata":{"session_id":"real-rich-session"},"messages":[{"role":"user","content":"large history".repeat(10000)},{"role":"assistant","content":"previous reply"},{"role":"user","content":[{"type":"image","source":{"data":"own-test"}}]}],"tools":[{"name":"edit"}]}).to_string();
    assert!(raw.len() > MAX_FIRST_TURN);
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header("content-type", "application/json")
                .body(Body::from(raw.clone()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.headers()["x-destination"], "candidate");
    assert_eq!(drain(response).await, raw);
    let key = session_key("real-rich-session").unwrap();
    assert_eq!(f.store.lock().await.pins[&key], Destination::Candidate);
    assert_eq!(f.migration.lock().sessions[&key].active, 0);
}

#[tokio::test]
async fn durable_failure_does_not_publish_a_migration_or_reset_live_leases() {
    let dir = tempfile::tempdir().unwrap();
    let f = front(dir.path().join("pins.json"));
    let key = session_key("persist-move").unwrap();
    let (_, _, initial) = f.choose(Some(key.clone()), None, false, 10).await.unwrap();
    drop(initial);
    age(&f, &key);
    enable(&f, 1);
    f.store.lock().await.path = dir.path().join("missing/pins.json");
    assert!(matches!(
        f.choose(Some(key.clone()), None, true, 10).await,
        Err(StatusCode::SERVICE_UNAVAILABLE)
    ));
    assert_eq!(f.store.lock().await.pins[&key], Destination::Legacy);
    assert_eq!(f.migration.lock().cold_inflight, 0);
    assert_eq!(f.migration.lock().migrated, 0);
}

#[tokio::test]
async fn post_rename_failure_quarantines_the_affected_existing_key() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("pins.json");
    let f = front(path.clone());
    let key = session_key("post-rename").unwrap();
    let (_, _, initial) = f.choose(Some(key.clone()), None, false, 1).await.unwrap();
    drop(initial);
    age(&f, &key);
    enable(&f, 1);
    f.store.lock().await.fail_after_rename = true;
    assert!(matches!(
        f.choose(Some(key.clone()), None, true, 1).await,
        Err(StatusCode::SERVICE_UNAVAILABLE)
    ));
    let disk: BTreeMap<String, Destination> =
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    assert_eq!(disk[&key], Destination::Candidate);
    assert_eq!(f.store.lock().await.pins[&key], Destination::Legacy);
    assert!(matches!(
        f.choose(Some(key), None, false, 0).await,
        Err(StatusCode::SERVICE_UNAVAILABLE)
    ));
    assert_eq!(f.migration.lock().active_legacy, 0);
    assert_eq!(f.migration.lock().cold_inflight, 0);
}

#[tokio::test]
async fn encoded_legacy_stream_is_tracked_and_blocks_a_competing_plain_move() {
    let upstream = Router::new().route(
        "/v1/messages",
        post(|| async { Response::new(Body::from_stream(PendingBody(false))) }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, upstream).await.unwrap() });
    let dir = tempfile::tempdir().unwrap();
    let f = SessionHandoff::open(
        origin,
        dir.path().join("pins.json"),
        1024 * 1024,
        AuthConfig { api_key: None },
    )
    .unwrap();
    let key = session_key("encoded-stream").unwrap();
    let (_, _, initial) = f.choose(Some(key.clone()), None, false, 1).await.unwrap();
    drop(initial);
    age(&f, &key);
    enable(&f, 1);
    let app = Router::new()
        .route("/v1/messages", post(|| async { "candidate" }))
        .layer(axum::middleware::from_fn_with_state(f.clone(), dispatch));
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header("content-encoding", "identity")
                .header("x-sglang-session-id", "encoded-stream")
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await
        .unwrap();
    let mut stream = response.into_body().into_data_stream();
    assert!(stream.next().await.unwrap().is_ok());
    assert_eq!(f.migration.lock().sessions[&key].active, 1);
    let (d, _, parallel) = f.choose(Some(key.clone()), None, true, 5).await.unwrap();
    assert_eq!(d, Destination::Legacy);
    drop(parallel);
    drop(stream);
    assert_eq!(f.migration.lock().sessions[&key].active, 0);
    server.abort();
}

#[tokio::test]
async fn clean_eof_allows_next_turn_but_cancelled_body_keeps_minimum_grace() {
    let dir = tempfile::tempdir().unwrap();
    let f = front(dir.path().join("pins.json"));
    let clean = session_key("clean-next-turn").unwrap();
    let cancelled = session_key("cancel-next-turn").unwrap();
    let (_, _, finished) = f
        .choose(Some(clean.clone()), None, false, 10)
        .await
        .unwrap();
    drain(wrap(Response::new(Body::from("complete")), finished)).await;
    let (_, _, unfinished) = f
        .choose(Some(cancelled.clone()), None, false, 10)
        .await
        .unwrap();
    drop(unfinished);
    enable(&f, 2);
    f.migration.lock().config.quiet_secs = 0;
    let (destination, _, blocked) = f.choose(Some(cancelled), None, true, 10).await.unwrap();
    assert_eq!(destination, Destination::Legacy);
    drop(blocked);
    let (destination, reason, moved) = f.choose(Some(clean), None, true, 10).await.unwrap();
    assert_eq!(destination, Destination::Candidate);
    assert_eq!(reason, "migrated");
    drop(moved);
}

#[tokio::test]
async fn unrelated_existing_candidate_waits_for_cold_budget_without_legacy_fallback() {
    let dir = tempfile::tempdir().unwrap();
    let f = front(dir.path().join("pins.json"));
    let one = session_key("imported-cold-one").unwrap();
    let two = session_key("imported-cold-two").unwrap();
    let map = BTreeMap::from([
        (one.clone(), Destination::Candidate),
        (two.clone(), Destination::Candidate),
    ]);
    assert_eq!(
        import_pins(State(f.clone()), Json(map)).await.status(),
        StatusCode::OK
    );
    let (_, _, first) = f.choose(Some(one), None, false, 0).await.unwrap();
    assert!(matches!(
        f.choose(Some(two.clone()), None, false, 0).await,
        Err(StatusCode::TOO_MANY_REQUESTS)
    ));
    assert_eq!(f.store.lock().await.pins[&two], Destination::Candidate);
    drop(first);
    let (d, _, next) = f.choose(Some(two), None, false, 0).await.unwrap();
    assert_eq!(d, Destination::Candidate);
    drop(next);
}

#[tokio::test]
async fn a_later_clean_parallel_response_does_not_erase_cancellation_grace() {
    let dir = tempfile::tempdir().unwrap();
    let f = front(dir.path().join("pins.json"));
    let key = session_key("cancel-and-clean").unwrap();
    let (_, _, cancelled) = f.choose(Some(key.clone()), None, false, 10).await.unwrap();
    let (_, _, clean) = f.choose(Some(key.clone()), None, false, 10).await.unwrap();
    drop(cancelled);
    drain(wrap(Response::new(Body::from("complete")), clean)).await;
    enable(&f, 1);
    f.migration.lock().config.quiet_secs = 0;
    assert!(f.migration.lock().sessions[&key].clean_end);
    let (destination, _, retry) = f.choose(Some(key.clone()), None, true, 10).await.unwrap();
    assert_eq!(destination, Destination::Legacy);
    drop(retry);
    age(&f, &key);
    let (destination, _, safe) = f.choose(Some(key), None, true, 10).await.unwrap();
    assert_eq!(destination, Destination::Candidate);
    drop(safe);
}

#[tokio::test]
async fn real_http_content_length_completion_is_clean_without_an_extra_eof_poll() {
    let dir = tempfile::tempdir().unwrap();
    let f = front(dir.path().join("pins.json"));
    let key = session_key("content-length").unwrap();
    let endpoint_f = f.clone();
    let endpoint_key = key.clone();
    let app = Router::new().route(
        "/json",
        axum::routing::get(move || {
            let f = endpoint_f.clone();
            let key = endpoint_key.clone();
            async move {
                let (_, _, lease) = f.choose(Some(key), None, false, 0).await.unwrap();
                let stream = futures_util::stream::iter([Ok::<_, Infallible>(
                    bytes::Bytes::from_static(b"done"),
                )]);
                let mut response = Response::new(Body::from_stream(stream));
                response
                    .headers_mut()
                    .insert("content-length", HeaderValue::from_static("4"));
                wrap(response, lease)
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/json", listener.local_addr().unwrap());
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    assert_eq!(
        reqwest::Client::builder()
            .no_proxy()
            .build()
            .unwrap()
            .get(url)
            .send()
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap(),
        "done"
    );
    for _ in 0..10 {
        if f.migration.lock().sessions[&key].active == 0 {
            break;
        }
        tokio::task::yield_now().await;
    }
    assert!(f.migration.lock().sessions[&key].clean_end);
    enable(&f, 1);
    f.migration.lock().config.quiet_secs = 0;
    let (d, _, next) = f.choose(Some(key), None, true, 10).await.unwrap();
    assert_eq!(d, Destination::Candidate);
    drop(next);
    task.abort();
}

#[tokio::test]
async fn truncated_content_length_is_not_a_clean_completion() {
    let dir = tempfile::tempdir().unwrap();
    let f = front(dir.path().join("pins.json"));
    let key = session_key("truncated-reply").unwrap();
    let (_, _, lease) = f.choose(Some(key.clone()), None, false, 0).await.unwrap();
    let mut response = Response::new(Body::from_stream(futures_util::stream::iter([Ok::<
        _,
        Infallible,
    >(
        bytes::Bytes::from_static(b"half"),
    )])));
    response
        .headers_mut()
        .insert("content-length", HeaderValue::from_static("8"));
    assert_eq!(drain(wrap(response, lease)).await, "half");
    assert!(!f.migration.lock().sessions[&key].clean_end);
    assert_eq!(f.migration.lock().sessions[&key].active, 0);
}

#[tokio::test]
async fn a_completed_length_seals_the_body_and_never_polls_trailing_frames() {
    let dir = tempfile::tempdir().unwrap();
    let f = front(dir.path().join("pins.json"));
    let key = session_key("sealed-length").unwrap();
    let (_, _, lease) = f.choose(Some(key.clone()), None, false, 0).await.unwrap();
    let frames = vec![
        Ok(bytes::Bytes::from_static(b"done")),
        Err(io::Error::other("outside completed HTTP framing")),
    ];
    let mut response = Response::new(Body::from_stream(futures_util::stream::iter(frames)));
    response
        .headers_mut()
        .insert("content-length", HeaderValue::from_static("4"));
    let mut body = wrap(response, lease).into_body();
    let frame = std::future::poll_fn(|cx| Pin::new(&mut body).poll_frame(cx))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(frame.into_data().unwrap(), "done");
    assert!(body.is_end_stream());
    assert!(
        std::future::poll_fn(|cx| Pin::new(&mut body).poll_frame(cx))
            .await
            .is_none()
    );
    assert!(f.migration.lock().sessions[&key].clean_end);
}

#[tokio::test]
async fn oversized_frame_is_an_error_and_never_a_clean_completion() {
    let dir = tempfile::tempdir().unwrap();
    let f = front(dir.path().join("pins.json"));
    let key = session_key("oversized-length").unwrap();
    let (_, _, lease) = f.choose(Some(key.clone()), None, false, 0).await.unwrap();
    let mut response = Response::new(Body::from_stream(futures_util::stream::iter([Ok::<
        _,
        Infallible,
    >(
        bytes::Bytes::from_static(b"too-long"),
    )])));
    response
        .headers_mut()
        .insert("content-length", HeaderValue::from_static("4"));
    assert!(to_bytes(wrap(response, lease).into_body(), 100)
        .await
        .is_err());
    assert!(!f.migration.lock().sessions[&key].clean_end);
    assert_eq!(f.migration.lock().sessions[&key].active, 0);
}
