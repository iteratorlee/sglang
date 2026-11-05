use super::*;
use axum::{
    routing::{get, post},
    Router,
};
use futures_util::{Stream, StreamExt};
use std::{
    convert::Infallible,
    pin::Pin,
    task::{Context, Poll},
};
use tower::ServiceExt;

fn front(path: PathBuf, origin: String) -> Arc<SessionHandoff> {
    SessionHandoff::open(origin, path, 1024 * 1024, AuthConfig { api_key: None }).unwrap()
}

#[test]
fn session_identity_never_uses_an_account_user_id() {
    assert_eq!(
        body_identity(br#"{"metadata":{"user_id":"account-123"}}"#).0,
        None
    );
    let direct = body_identity(br#"{"metadata":{"session_id":"s-123"}}"#);
    let nested = body_identity(
        br#"{"metadata":{"user_id":"{\"session_id\":\"s-123\",\"account_id\":\"a\"}"}}"#,
    );
    assert_eq!(direct.0, nested.0);
    assert_eq!(direct.0, session_key("s-123"));
    let mut headers = HeaderMap::new();
    headers.insert("x-opencode-session-id", HeaderValue::from_static("s-123"));
    assert_eq!(header_identity(&headers).unwrap(), direct.0);
    headers.insert("x-sglang-session-id", HeaderValue::from_static("another"));
    assert!(header_identity(&headers).is_err());
    assert_eq!(
        body_identity(
            br#"{"metadata":{"session_id":"s-123","user_id":"{\"session_id\":\"another\"}"}}"#
        )
        .0,
        None
    );
    assert!(session_key("").is_none());
    assert!(session_key("not a session").is_none());
    assert!(session_key(&"s".repeat(257)).is_none());
}

#[test]
fn bounded_first_turn_rejects_history_tools_images_and_large_bodies() {
    assert!(bounded_first_turn(
        br#"{"messages":[{"role":"system","content":"sys"},{"role":"user","content":"hi"}]}"#
    ));
    assert!(bounded_first_turn(
        br#"{"system":"sys","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}"#
    ));
    for message in [
        json!([{"role":"assistant","content":"old"},{"role":"user","content":"hi"}]),
        json!([{"role":"tool","content":"result"},{"role":"user","content":"hi"}]),
        json!([{"role":"user","content":"old"},{"role":"user","content":"hi"}]),
        json!([{"role":"user","content":[{"type":"image","source":{}}]}]),
        json!([{"role":"user","content":[{"type":"image_url","image_url":{}}]}]),
        json!([{"role":"user","content":"hi","tool_calls":[]}]),
        json!([{"role":"user","content":"x".repeat(MAX_FIRST_TURN)}]),
    ] {
        assert!(!bounded_first_turn(
            &serde_json::to_vec(&json!({"messages":message})).unwrap()
        ));
    }
    assert!(HandoffConfig {
        candidate_bps: 5001,
        bounded_first_turn: true
    }
    .validate()
    .is_err());
    assert!(HandoffConfig {
        candidate_bps: 5000,
        bounded_first_turn: true
    }
    .validate()
    .is_ok());
}

#[tokio::test]
async fn pins_are_durable_private_monotonic_and_atomic() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("pins.json");
    let f = front(path.clone(), "http://127.0.0.1:1".into());
    let old = session_key("old-private-id");
    assert_eq!(
        f.decide(old.clone(), None).await.unwrap().0,
        Destination::Legacy
    );
    f.store.lock().await.config.candidate_bps = 10_000;
    assert_eq!(
        f.decide(old, Some("explicit")).await.unwrap().0,
        Destination::Legacy
    );
    let key = session_key("new-private-id");
    let (a, b) = tokio::join!(
        f.decide(key.clone(), Some("explicit")),
        f.decide(key.clone(), None)
    );
    assert_eq!(a.unwrap().0, b.unwrap().0);
    assert_eq!(f.store.lock().await.pins.len(), 2);
    let expected = f
        .store
        .lock()
        .await
        .pins
        .get(key.as_ref().unwrap())
        .copied()
        .unwrap();
    f.store.lock().await.config.candidate_bps = 0;
    assert_eq!(f.decide(key.clone(), None).await.unwrap().0, expected);
    assert!(SessionHandoff::open(
        "http://127.0.0.1:1".into(),
        path.clone(),
        1024,
        AuthConfig { api_key: None }
    )
    .is_err());
    let contents = std::fs::read_to_string(&path).unwrap();
    assert!(!contents.contains("private-id"));
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }
    drop(f);
    let restarted = front(path.clone(), "http://127.0.0.1:1".into());
    assert_eq!(restarted.decide(key, None).await.unwrap().0, expected);
    assert_eq!(restarted.store.lock().await.config.candidate_bps, 0);
    drop(restarted);
    std::fs::write(&path, "corrupt").unwrap();
    assert!(SessionHandoff::open(
        "http://127.0.0.1:1".into(),
        path,
        1024,
        AuthConfig { api_key: None }
    )
    .is_err());
}

#[tokio::test]
async fn failed_persistence_never_dispatches_or_forgets_a_known_pin() {
    let directory = tempfile::tempdir().unwrap();
    let f = front(
        directory.path().join("pins.json"),
        "http://127.0.0.1:1".into(),
    );
    f.store.lock().await.config.candidate_bps = 10_000;
    let known = session_key("already-durable");
    assert_eq!(
        f.decide(known.clone(), Some("explicit")).await.unwrap().0,
        Destination::Candidate
    );
    f.store.lock().await.path = directory.path().join("missing/pins.json");
    assert!(f
        .decide(session_key("uncommitted"), Some("explicit"))
        .await
        .is_err());
    assert!(f.store.lock().await.failed);
    assert!(f.decide(session_key("uncommitted"), None).await.is_err());
    assert_eq!(
        f.decide(known, None).await.unwrap().0,
        Destination::Candidate
    );
}

async fn echo(request: Request) -> Response {
    let mut response = Response::new(request.into_body());
    response
        .headers_mut()
        .insert("x-destination", HeaderValue::from_static("candidate"));
    response
}

async fn call(app: &Router, path: &str, body: &str, headers: &[(&str, &str)]) -> Response {
    let mut request = Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json");
    for (name, value) in headers {
        request = request.header(*name, *value);
    }
    app.clone()
        .oneshot(request.body(Body::from(body.to_owned())).unwrap())
        .await
        .unwrap()
}

#[tokio::test]
async fn proxy_preserves_bytes_query_status_and_candidate_pin_over_zero() {
    let upstream = Router::new().fallback(|request: Request| async move {
        assert_eq!(request.uri().query(), Some("q=kept"));
        assert_eq!(request.headers()["authorization"], "Bearer test");
        let mut response = Response::new(request.into_body());
        *response.status_mut() = StatusCode::TEMPORARY_REDIRECT;
        response
            .headers_mut()
            .insert("location", HeaderValue::from_static("/must-not-follow"));
        response
            .headers_mut()
            .append("set-cookie", HeaderValue::from_static("a=1"));
        response
            .headers_mut()
            .append("set-cookie", HeaderValue::from_static("b=2"));
        response
    });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move { axum::serve(listener, upstream).await.unwrap() });
    let directory = tempfile::tempdir().unwrap();
    let f = front(directory.path().join("pins.json"), origin);
    f.store.lock().await.config.candidate_bps = 10_000;
    let app = Router::new()
        .route("/v1/messages", post(echo))
        .route("/v1/chat/completions", post(echo))
        .route("/workers", get(|| async { "local-admin" }))
        .layer(axum::middleware::from_fn_with_state(f.clone(), dispatch));
    let raw = "{ \"metadata\": {\"user_id\":\"account-only\"}, \"messages\": [] }";
    let response = call(
        &app,
        "/v1/messages?q=kept",
        raw,
        &[("authorization", "Bearer test"), (CANDIDATE_HEADER, "1")],
    )
    .await;
    assert_eq!(response.status(), StatusCode::TEMPORARY_REDIRECT);
    assert_eq!(response.headers().get_all("set-cookie").iter().count(), 2);
    assert_eq!(
        to_bytes(response.into_body(), 1024).await.unwrap().as_ref(),
        raw.as_bytes()
    );
    let mm = json!({"messages":[{"role":"user","content":[{"type":"image","source":{"data":"a".repeat(MAX_FIRST_TURN+1)}}]}]}).to_string();
    let response = call(
        &app,
        "/v1/chat/completions",
        &mm,
        &[
            ("x-opencode-session-id", "test-mm"),
            ("x-sglang-new-session", "1"),
        ],
    )
    .await;
    assert_eq!(response.headers()["x-destination"], "candidate");
    assert_eq!(
        to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap()
            .as_ref(),
        mm.as_bytes()
    );
    f.store.lock().await.config.candidate_bps = 0;
    let response = call(
        &app,
        "/v1/messages",
        "unchanged followup",
        &[("x-sglang-session-id", "test-mm")],
    )
    .await;
    assert_eq!(response.headers()["x-destination"], "candidate");
    assert_eq!(
        to_bytes(response.into_body(), 1024).await.unwrap(),
        "unchanged followup"
    );
    let admin = app
        .oneshot(
            Request::builder()
                .uri("/workers")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        to_bytes(admin.into_body(), 1024).await.unwrap(),
        "local-admin"
    );
    task.abort();
}

struct EndlessSse {
    first: bool,
    dropped: Option<tokio::sync::oneshot::Sender<()>>,
}
impl Stream for EndlessSse {
    type Item = Result<bytes::Bytes, Infallible>;
    fn poll_next(mut self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.first {
            self.first = false;
            Poll::Ready(Some(Ok(bytes::Bytes::from_static(b"data: first\n\n"))))
        } else {
            Poll::Pending
        }
    }
}
impl Drop for EndlessSse {
    fn drop(&mut self) {
        if let Some(tx) = self.dropped.take() {
            let _ = tx.send(());
        }
    }
}

#[tokio::test]
async fn legacy_sse_is_incremental_and_downstream_drop_cancels_upstream() {
    let (tx, rx) = tokio::sync::oneshot::channel();
    let sender = Arc::new(std::sync::Mutex::new(Some(tx)));
    let upstream = Router::new().route(
        "/v1/messages",
        post(move || {
            let tx = sender.lock().unwrap().take();
            async {
                Response::new(Body::from_stream(EndlessSse {
                    first: true,
                    dropped: tx,
                }))
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move { axum::serve(listener, upstream).await.unwrap() });
    let directory = tempfile::tempdir().unwrap();
    let f = front(directory.path().join("pins.json"), origin);
    let app = Router::new()
        .route("/v1/messages", post(echo))
        .layer(axum::middleware::from_fn_with_state(f, dispatch));
    let response = tokio::time::timeout(
        Duration::from_secs(2),
        call(&app, "/v1/messages", "{}", &[]),
    )
    .await
    .unwrap();
    let mut stream = response.into_body().into_data_stream();
    let first = tokio::time::timeout(Duration::from_secs(2), stream.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(first, "data: first\n\n");
    drop(stream);
    tokio::time::timeout(Duration::from_secs(2), rx)
        .await
        .expect("upstream body must be dropped after downstream cancellation")
        .unwrap();
    task.abort();
}
