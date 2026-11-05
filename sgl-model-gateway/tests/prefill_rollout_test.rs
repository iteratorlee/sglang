//! Local HTTP control-plane test: POST /workers really runs DP discovery and
//! registration while the same policy instances continue serving selections.
//! This does not validate physical P->D KV/state transfer or cache residency.

use std::{sync::Arc, time::Duration};

use axum::{
    body::{to_bytes, Body},
    http::{Request, StatusCode},
    routing::get,
    Json, Router,
};
use serde_json::{json, Value};
use smg::{
    app_context::AppContext,
    config::{PolicyConfig, RouterConfig},
    core::{
        job_queue::{JobQueue, JobQueueConfig},
        steps::WorkflowEngines,
        DPAwareWorkerBuilder, ModelCard, Worker, WorkerType,
    },
    middleware::AuthConfig,
    policies::SelectWorkerInfo,
    routers::RouterFactory,
    server::{build_app, AppState},
};
use tower::ServiceExt;

async fn control(app: &Router, method: &str, path: &str, body: Value) -> (StatusCode, Value) {
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method(method)
                .uri(path)
                .header("content-type", "application/json")
                .header("authorization", "Bearer test-only")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap();
    let body =
        serde_json::from_slice(&bytes).unwrap_or_else(|_| json!(String::from_utf8_lossy(&bytes)));
    (status, body)
}

#[tokio::test]
async fn post_workers_expands_live_prefill_dp2_to_dp4_with_rollout() {
    // This integration test has its own test binary, with one test, so these
    // startup-only settings cannot race any other policy fixtures.
    std::env::set_var("SGLANG_GATEWAY_PD_AFFINITY_PREFIX_TOKENS", "8192");
    let mock = Router::new()
        .route("/health", get(|| async { StatusCode::OK }))
        .route(
            "/server_info",
            get(|| async { Json(json!({"dp_size":2,"tp_size":8})) }),
        )
        .route(
            "/model_info",
            get(|| async { Json(json!({"served_model_name":"model-a"})) }),
        );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let new_url = format!("http://{}", listener.local_addr().unwrap());
    let mock_task = tokio::spawn(async move { axum::serve(listener, mock).await.unwrap() });
    let old_url = "http://127.0.0.1:31001";
    let d_url = "http://127.0.0.1:31002";
    let mut config = RouterConfig::builder()
        .prefill_decode_mode(vec![(old_url.into(), Some(18994))], vec![d_url.into()])
        .policy(PolicyConfig::PrefixHash {
            prefix_token_count: 256,
            load_factor: 1.25,
        })
        .build_unchecked();
    config.dp_aware = true;
    config.api_key = Some("test-only".into());
    config.worker_startup_timeout_secs = 10;
    let context = Arc::new(AppContext::from_config(config.clone(), 30).await.unwrap());
    context
        .workflow_engines
        .set(WorkflowEngines::new(&config))
        .unwrap();
    context
        .worker_job_queue
        .set(JobQueue::new(
            JobQueueConfig::default(),
            Arc::downgrade(&context),
        ))
        .unwrap();
    for (url, size, kind) in [
        (
            old_url,
            2,
            WorkerType::Prefill {
                bootstrap_port: Some(18994),
            },
        ),
        (d_url, 8, WorkerType::Decode),
    ] {
        for rank in 0..size {
            context.worker_registry.register(Arc::new(
                DPAwareWorkerBuilder::new(url, rank, size)
                    .worker_type(kind.clone())
                    .model(ModelCard::new("model-a"))
                    .build(),
            ));
        }
    }
    let router = RouterFactory::create_router(&context).await.unwrap();
    let app = build_app(
        Arc::new(AppState {
            router: Arc::from(router),
            context: context.clone(),
            concurrency_queue_tx: None,
            router_manager: None,
            mesh_handler: None,
            mesh_sync_manager: None,
        }),
        AuthConfig {
            api_key: Some("test-only".into()),
        },
        None,
        1 << 20,
        vec![],
        vec![],
    );

    let unauthorized = app
        .clone()
        .oneshot(
            Request::builder()
                .method("PUT")
                .uri("/workers/prefill-rollout")
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(unauthorized.status(), StatusCode::UNAUTHORIZED);
    let rollout = |bps| json!({"incumbent_urls":[old_url],"new_prefix_bps":bps,"owner_max_load":2});
    let (status, initial) = control(&app, "PUT", "/workers/prefill-rollout", rollout(0)).await;
    assert_eq!(status, StatusCode::OK, "{initial}");
    assert_eq!(
        control(&app, "GET", "/workers/prefill-rollout", Value::Null)
            .await
            .1,
        initial
    );
    assert_eq!(
        control(&app, "PUT", "/workers/prefill-rollout", rollout(10001))
            .await
            .0,
        StatusCode::BAD_REQUEST
    );

    let p_policy = context.policy_registry.get_prefill_policy();
    let d_policy = context.policy_registry.get_decode_policy();
    let p_before = context.worker_registry.get_prefill_workers();
    let d_before = context.worker_registry.get_decode_workers();
    let tokens = vec![7; 8192];
    let before = SelectWorkerInfo {
        tokens: Some(&tokens),
        hash_ring: context.worker_registry.get_hash_ring("model-a"),
        ..Default::default()
    };
    let p_owner = p_before[p_policy.select_worker(&p_before, &before).await.unwrap()].clone();
    // Force an observed D fallback that cannot be recovered merely by hashing.
    let d_ring = d_policy.select_worker(&d_before, &before).await.unwrap();
    d_before[d_ring].set_healthy(false);
    let d_owner = d_before[d_policy.select_worker(&d_before, &before).await.unwrap()].clone();
    d_before[d_ring].set_healthy(true);
    p_owner.increment_load();

    let (status, accepted) = control(
        &app,
        "POST",
        "/workers",
        json!({
            "url":new_url, "worker_type":"prefill", "bootstrap_port":18995,
            "model_id":"model-a", "health_check_timeout_secs":1,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::ACCEPTED, "{accepted}");
    tokio::time::timeout(Duration::from_secs(15), async {
        loop {
            let p = context.worker_registry.get_prefill_workers();
            if p.len() == 4 && p.iter().all(|w| w.is_available()) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("POST /workers must finish registration, not merely return 202");
    let (status, listed) = control(&app, "GET", "/workers", Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(listed["stats"]["prefill_count"], 4);
    assert_eq!(listed["stats"]["decode_count"], 8);
    assert!(Arc::ptr_eq(
        &p_policy,
        &context.policy_registry.get_prefill_policy()
    ));
    let p_after = context.worker_registry.get_prefill_workers();
    for rank in 0..2 {
        let w = context
            .worker_registry
            .get_by_url(&format!("{new_url}@{rank}"))
            .unwrap();
        assert_eq!(w.dp_rank(), Some(rank));
        assert_eq!(w.dp_size(), Some(2));
        assert_eq!(w.base_url(), new_url);
        assert_eq!(w.bootstrap_port(), Some(18995));
        assert_eq!(w.bootstrap_host(), "127.0.0.1");
    }
    let after = SelectWorkerInfo {
        hash_ring: context.worker_registry.get_hash_ring("model-a"),
        ..before
    };
    assert_eq!(
        p_after[p_policy.select_worker(&p_after, &after).await.unwrap()].url(),
        p_owner.url()
    );
    assert_eq!(
        d_before[d_policy.select_worker(&d_before, &after).await.unwrap()].url(),
        d_owner.url()
    );
    for id in 10..42 {
        let tokens = vec![id; 8192];
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            ..after.clone()
        };
        assert_eq!(
            p_after[p_policy.select_worker(&p_after, &info).await.unwrap()].base_url(),
            old_url
        );
    }
    assert_eq!(
        control(&app, "PUT", "/workers/prefill-rollout", rollout(10000))
            .await
            .0,
        StatusCode::OK
    );
    // A policy control update must not reset worker health/load/circuit objects.
    assert!(Arc::ptr_eq(
        &p_owner,
        &context.worker_registry.get_by_url(p_owner.url()).unwrap()
    ));
    assert_eq!(p_owner.load(), 1);
    let mut reached_new = false;
    for id in 100..164 {
        let tokens = vec![id; 8192];
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            ..after.clone()
        };
        let w = &p_after[p_policy.select_worker(&p_after, &info).await.unwrap()];
        reached_new |= w.base_url() == new_url;
    }
    assert!(
        reached_new,
        "registered new DP workers must actually become selectable"
    );
    mock_task.abort();
}
