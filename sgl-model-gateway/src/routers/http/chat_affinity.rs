//! Advisory, authoritative Chat tokenization. Only the Python backend knows all
//! serving-template normalization rules. No prompt or token-result cache: model
//! updates and clear_thinking must take effect on every request.
use std::{sync::Arc, time::Duration};

use reqwest::Client;
use serde::Deserialize;
use serde_json::Value;
use tokio::sync::Semaphore;

use crate::core::Worker;

const MAX_REQUEST_BYTES: usize = 8 * 1024 * 1024;
const MAX_RESPONSE_BYTES: usize = 12 * 1024 * 1024;
const MAX_TOKENS: usize = 1024 * 1024;

#[derive(Debug)]
pub(super) struct ChatAffinity {
    slots: Semaphore,
}

impl Default for ChatAffinity {
    fn default() -> Self {
        Self {
            slots: Semaphore::new(8),
        }
    }
}

#[derive(Deserialize)]
struct Tokenization {
    supported: bool,
    version: Option<u32>,
    tokens: Option<Vec<u32>>,
    count: Option<usize>,
}

/// Reject media before serializing/sending any image, video or audio payload.
/// Do not inspect tool arguments: arbitrary JSON there is ordinary text.
pub(super) fn supports_text_request(body: &Value) -> bool {
    if body
        .get("n")
        .is_some_and(|n| !n.is_null() && n.as_u64() != Some(1))
    {
        return false;
    }
    for key in [
        "lora_path",
        "session_id",
        "session_params",
        "input_embeds",
        "image_data",
        "video_data",
        "audio_data",
        "mm_hashes",
    ] {
        if body.get(key).is_some_and(|v| !v.is_null()) {
            return false;
        }
    }
    body.get("messages")
        .and_then(Value::as_array)
        .is_some_and(|messages| {
            !messages.is_empty()
                && messages.iter().all(|message| {
                    message.is_object()
                        && match message.get("content") {
                            None | Some(Value::Null) | Some(Value::String(_)) => true,
                            Some(Value::Array(parts)) => parts.iter().all(|part| {
                                part.get("type").and_then(Value::as_str) == Some("text")
                                    && part.get("text").is_some_and(Value::is_string)
                            }),
                            _ => false,
                        }
                })
        })
}

impl ChatAffinity {
    pub(super) async fn tokenize(
        &self,
        client: &Client,
        worker: &dyn Worker,
        body: &Value,
    ) -> Option<Arc<[u32]>> {
        let started = std::time::Instant::now();
        let result = self.try_tokenize(client, worker, body).await;
        let outcome = result.as_ref().err().copied().unwrap_or("exact");
        metrics::counter!("smg_pd_chat_affinity_total", "outcome" => outcome).increment(1);
        metrics::histogram!("smg_pd_chat_tokenize_duration_seconds")
            .record(started.elapsed().as_secs_f64());
        result.ok()
    }

    async fn try_tokenize(
        &self,
        client: &Client,
        worker: &dyn Worker,
        body: &Value,
    ) -> Result<Arc<[u32]>, &'static str> {
        if !supports_text_request(body) {
            return Err("unsupported_request");
        }
        let _permit = self.slots.try_acquire().map_err(|_| "busy")?;
        let payload = serde_json::to_vec(body).map_err(|_| "serialization")?;
        if payload.len() > MAX_REQUEST_BYTES {
            return Err("too_large");
        }
        let mut request = client
            .post(super::pd_types::api_path(
                worker.base_url(),
                "/v1/chat/tokenize",
            ))
            .timeout(Duration::from_secs(2))
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(payload);
        if let Some(api_key) = worker.api_key() {
            request = request.bearer_auth(api_key);
        }
        // Share the gateway's pooled client, but never acquire a PD lease or
        // update inference load/circuit breakers for this CPU-only operation.
        let mut response = request.send().await.map_err(|_| "transport")?;
        if !response.status().is_success() {
            return Err("http_status");
        }
        let mut bytes = Vec::new();
        while let Some(chunk) = response.chunk().await.map_err(|_| "transport")? {
            if bytes.len().saturating_add(chunk.len()) > MAX_RESPONSE_BYTES {
                return Err("response_too_large");
            }
            bytes.extend_from_slice(&chunk);
        }
        let result: Tokenization =
            serde_json::from_slice(&bytes).map_err(|_| "invalid_response")?;
        if !result.supported {
            return Err("backend_unsupported");
        }
        let tokens = result.tokens.ok_or("invalid_response")?;
        if result.version != Some(1)
            || tokens.is_empty()
            || tokens.len() > MAX_TOKENS
            || result.count != Some(tokens.len())
        {
            return Err("invalid_response");
        }
        Ok(tokens.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn transport_keeps_stream_true_and_rejects_malformed_token_responses() {
        use crate::core::BasicWorkerBuilder;
        use axum::{routing::post, Json};
        for result in [
            json!({"supported":true,"version":1,"tokens":[7,8],"count":2}),
            json!({"supported":true,"version":1,"tokens":[7,8],"count":3}),
            json!({"supported":true,"version":2,"tokens":[7,8],"count":2}),
            json!({"supported":true,"version":1,"tokens":[-1],"count":1}),
        ] {
            let expected_ok = result["count"] == 2 && result["version"] == 1;
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let worker =
                BasicWorkerBuilder::new(format!("http://{}", listener.local_addr().unwrap()))
                    .build();
            let raw = json!({"messages":[{"role":"user","content":"hello"}], "stream":true, "return_spec_tokens_details":true});
            let expected = raw.clone();
            let app = axum::Router::new().route(
                "/v1/chat/tokenize",
                post(move |Json(body): Json<Value>| {
                    assert_eq!(body, expected);
                    let result = result.clone();
                    async move { Json(result) }
                }),
            );
            let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
            assert_eq!(
                ChatAffinity::default()
                    .tokenize(&Client::new(), &worker, &raw)
                    .await
                    .is_some(),
                expected_ok
            );
            task.abort();
        }
    }

    #[test]
    fn eligibility_preserves_tools_but_excludes_media_and_sessions() {
        let text = json!({"messages": [{"role":"user", "content":[{"type":"text", "text":"hello"}]}, {"role":"assistant", "content":null, "tool_calls":[{"function":{"arguments":"{\"image_url\":\"ordinary tool argument\"}"}}]}]});
        assert!(supports_text_request(&text));
        for media in [
            "image_url",
            "video_url",
            "input_audio",
            "audio_url",
            "unknown",
        ] {
            let mut body = text.clone();
            body["messages"][0]["content"][0]["type"] = json!(media);
            assert!(!supports_text_request(&body));
        }
        for key in ["session_id", "lora_path", "input_embeds", "session_params"] {
            let mut body = text.clone();
            body[key] = json!("value");
            assert!(!supports_text_request(&body));
        }
        let mut fanout = text;
        fanout["n"] = json!(2);
        assert!(!supports_text_request(&fanout));
    }
}
