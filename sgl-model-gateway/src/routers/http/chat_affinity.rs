//! Advisory, authoritative Chat tokenization. Only the Python backend knows all
//! serving-template normalization rules. No prompt or token-result cache: model
//! updates and clear_thinking must take effect on every request.
use std::{sync::Arc, time::Duration};

use reqwest::Client;
use serde::Deserialize;
use serde_json::Value;
use tokio::sync::Semaphore;

use crate::{core::Worker, routers::sse::SseProtocol};

const MAX_REQUEST_BYTES: usize = 8 * 1024 * 1024;
const MAX_RESPONSE_BYTES: usize = 12 * 1024 * 1024;
const MAX_TOKENS: usize = 1024 * 1024;

const TOKENIZE_TIMEOUT_ENV: &str = "SGLANG_GATEWAY_PD_CHAT_TOKENIZE_TIMEOUT_MS";
const DEFAULT_TOKENIZE_TIMEOUT: Duration = Duration::from_secs(2);

fn parse_tokenize_timeout(raw: &str) -> Option<Duration> {
    // Keep advisory CPU work bounded, including on a stalled tokenizer host.
    raw.parse::<u64>()
        .ok()
        .filter(|ms| (1..=60_000).contains(ms))
        .map(Duration::from_millis)
}

fn tokenize_timeout_from_env() -> Duration {
    match std::env::var(TOKENIZE_TIMEOUT_ENV) {
        Ok(raw) => parse_tokenize_timeout(&raw).unwrap_or_else(|| {
            tracing::warn!(
                env = TOKENIZE_TIMEOUT_ENV,
                "Expected 1..=60000 milliseconds; using 2000"
            );
            DEFAULT_TOKENIZE_TIMEOUT
        }),
        Err(_) => DEFAULT_TOKENIZE_TIMEOUT,
    }
}

#[cfg(test)]
mod timeout_tests {
    use super::*;

    #[test]
    fn tokenize_timeout_is_bounded_and_default_stays_two_seconds() {
        assert_eq!(DEFAULT_TOKENIZE_TIMEOUT, Duration::from_secs(2));
        for invalid in [
            "",
            "0",
            "-1",
            "1.5",
            "60001",
            "invalid",
            "18446744073709551616",
        ] {
            assert_eq!(parse_tokenize_timeout(invalid), None, "{invalid}");
        }
        for ms in [1, 2000, 5000, 60000] {
            assert_eq!(
                parse_tokenize_timeout(&ms.to_string()),
                Some(Duration::from_millis(ms))
            );
        }
    }
}

#[derive(Debug)]
pub(super) struct ChatAffinity {
    slots: Semaphore,
    tokenize_timeout: Duration,
    /// Reuse this request's authoritative final IDs in the native Chat input
    /// extension. Opt-in: deployments must use matching P/D templates/tokenizers.
    pub(super) reuse_token_ids: bool,
}

impl Default for ChatAffinity {
    fn default() -> Self {
        Self {
            slots: Semaphore::new(8),
            tokenize_timeout: tokenize_timeout_from_env(),
            reuse_token_ids: std::env::var("SGLANG_GATEWAY_PD_CHAT_REUSE_TOKEN_IDS")
                .is_ok_and(|value| value == "1"),
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
/// Tool arguments are JSON text, even when their keys look like media blocks.
/// This is an eligibility guard, not a second backend schema/template validator.
pub(super) fn supports_text_request(body: &Value, protocol: SseProtocol) -> bool {
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
    // Anthropic's system field is separate from messages. Only text is
    // normalized there; never ship media or unknown system blocks to tokenize.
    if protocol == SseProtocol::Anthropic && !plain_text_content(body.get("system")) {
        return false;
    }
    body.get("messages")
        .and_then(Value::as_array)
        .is_some_and(|messages| {
            !messages.is_empty()
                && messages.iter().all(|message| {
                    message.is_object()
                        && match protocol {
                            SseProtocol::OpenAi => plain_text_content(message.get("content")),
                            SseProtocol::Anthropic => {
                                anthropic_text_content(message.get("content"), 0)
                            }
                        }
                })
        })
}

// Preserve the existing OpenAI eligibility rules exactly.
fn plain_text_content(content: Option<&Value>) -> bool {
    match content {
        None | Some(Value::Null) | Some(Value::String(_)) => true,
        Some(Value::Array(parts)) => parts.iter().all(|part| {
            part.get("type").and_then(Value::as_str) == Some("text")
                && part.get("text").is_some_and(Value::is_string)
        }),
        _ => false,
    }
}

fn anthropic_text_content(content: Option<&Value>, depth: usize) -> bool {
    // Bound recursive tool_result inspection independently of JSON parser limits.
    if depth > 64 {
        return false;
    }
    match content {
        None | Some(Value::Null) | Some(Value::String(_)) => true,
        Some(Value::Array(parts)) => parts.iter().all(|part| {
            match part.get("type").and_then(Value::as_str) {
                Some("text") => part.get("text").is_some_and(Value::is_string),
                // The backend serializes input as ordinary JSON arguments.
                // Do not recursively mistake {"type":"image", ...} inside
                // tool input for a real image content block.
                Some("tool_use") => true,
                Some("tool_result") => anthropic_text_content(part.get("content"), depth + 1),
                Some("thinking") => part.get("thinking").is_some_and(Value::is_string),
                // Opaque data is textual. Backend support remains authoritative:
                // supported=false / validation failure never yields reusable IDs.
                Some("redacted_thinking") => part.get("data").is_some_and(Value::is_string),
                Some("tool_reference") => ["tool_name", "name"].iter().any(|key| {
                    part.get(*key)
                        .and_then(Value::as_str)
                        .is_some_and(|s| !s.is_empty())
                }),
                Some("search_result") => {
                    let source_is_text = match part.get("source") {
                        None | Some(Value::Null) | Some(Value::String(_)) => true,
                        Some(Value::Object(source)) => source.iter().all(|(key, value)| {
                            matches!(key.as_str(), "url" | "text") && value.is_string()
                        }),
                        _ => false,
                    };
                    source_is_text && plain_text_content(part.get("content"))
                }
                // Includes images nested in tool results, unknown future blocks,
                // and audio/video/document blocks requiring their own processor.
                _ => false,
            }
        }),
        _ => false,
    }
}

impl ChatAffinity {
    pub(super) fn body_with_prompt_ids(
        &self,
        body: &Value,
        tokens: &[u32],
        protocol: SseProtocol,
    ) -> Option<Value> {
        if !self.reuse_token_ids || tokens.is_empty() || !supports_text_request(body, protocol) {
            return None;
        }
        let mut forwarded = body.clone();
        forwarded.as_object_mut()?.insert(
            "input_ids".to_owned(),
            Value::Array(tokens.iter().copied().map(Value::from).collect()),
        );
        Some(forwarded)
    }

    pub(super) async fn tokenize(
        &self,
        client: &Client,
        worker: &dyn Worker,
        body: &Value,
        protocol: SseProtocol,
    ) -> Option<Arc<[u32]>> {
        let started = std::time::Instant::now();
        let result = self.try_tokenize(client, worker, body, protocol).await;
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
        protocol: SseProtocol,
    ) -> Result<Arc<[u32]>, &'static str> {
        if !supports_text_request(body, protocol) {
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
                protocol.tokenize_endpoint(),
            ))
            .timeout(self.tokenize_timeout)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(payload);
        if let Some(api_key) = worker.api_key() {
            request = request.bearer_auth(api_key);
        }
        // Share the gateway's pooled client, but never acquire a PD lease or
        // update inference load/circuit breakers for this CPU-only operation.
        let mut response = request.send().await.map_err(tokenize_transport_error)?;
        if !response.status().is_success() {
            return Err("http_status");
        }
        let mut bytes = Vec::new();
        while let Some(chunk) = response.chunk().await.map_err(tokenize_transport_error)? {
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

fn tokenize_transport_error(error: reqwest::Error) -> &'static str {
    if error.is_timeout() {
        "timeout"
    } else {
        "transport"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn native_tool_history() -> Value {
        json!({
            "model":"glm", "max_tokens":32, "stream":true,
            "system":[{"type":"text","text":"help","cache_control":{"type":"ephemeral"}}],
            "tools":[{"name":"read","input_schema":{"type":"object"}}],
            "messages":[
                {"role":"user","content":"inspect a file"},
                {"role":"assistant","content":[
                    {"type":"thinking","thinking":"read it","signature":"opaque"},
                    {"type":"tool_use","id":"tool_1","name":"read","input":{
                        "path":"a", "arbitrary_json":{"type":"image","source":{"type":"base64","data":"text argument"}}
                    }}
                ]},
                {"role":"user","content":[
                    {"type":"tool_result","tool_use_id":"tool_1","content":[
                        {"type":"text","text":"result"},
                        {"type":"tool_reference","tool_name":"read"},
                        {"type":"search_result","source":{"url":"https://example.test"},"title":"Reference", "content":[{"type":"text","text":"found"}]}
                    ]}
                ]}
            ],
            "thinking":{"type":"enabled","budget_tokens":1024},
            "cache_salt":"tenant", "unknown_extension":{"keep":true}
        })
    }

    #[test]
    fn native_tool_thinking_roundtrip_is_eligible_and_raw_fields_survive_reuse() {
        let raw = native_tool_history();
        assert!(supports_text_request(&raw, SseProtocol::Anthropic));
        assert!(!supports_text_request(&raw, SseProtocol::OpenAi));
        let mut affinity = ChatAffinity::default();
        affinity.reuse_token_ids = true;
        let mut forwarded = affinity
            .body_with_prompt_ids(&raw, &[10, 20, 30], SseProtocol::Anthropic)
            .unwrap();
        assert_eq!(
            forwarded.as_object_mut().unwrap().remove("input_ids"),
            Some(json!([10, 20, 30]))
        );
        assert_eq!(forwarded, raw);
        assert!(raw.get("input_ids").is_none());
        assert!(affinity
            .body_with_prompt_ids(&raw, &[10], SseProtocol::OpenAi)
            .is_none());
        affinity.reuse_token_ids = false;
        assert!(affinity
            .body_with_prompt_ids(&raw, &[10], SseProtocol::Anthropic)
            .is_none());
    }

    #[test]
    fn native_text_blocks_and_recursive_results_share_the_reuse_guard() {
        let mut affinity = ChatAffinity::default();
        affinity.reuse_token_ids = true;
        for block in [
            json!({"type":"text","text":""}),
            json!({"type":"thinking","thinking":"why"}),
            json!({"type":"redacted_thinking","data":"opaque"}),
            json!({"type":"tool_use","input":[{"type":"image","source":{"data":"ordinary JSON"}}]}),
            json!({"type":"tool_reference","name":"read"}),
            json!({"type":"search_result","source":"https://example.test","content":[{"type":"text","text":"found"}]}),
            json!({"type":"tool_result","tool_use_id":"t","content":null}),
            json!({"type":"tool_result","tool_use_id":"t","content":"result"}),
        ] {
            let nested =
                json!({"type":"tool_result", "tool_use_id":"outer", "content":[block.clone()]});
            for content in [block, nested] {
                let body = json!({"messages":[{"role":"user","content":[content]}]});
                assert!(
                    supports_text_request(&body, SseProtocol::Anthropic),
                    "{body}"
                );
                assert!(affinity
                    .body_with_prompt_ids(&body, &[1], SseProtocol::Anthropic)
                    .is_some());
            }
        }
    }

    #[test]
    fn native_images_unknown_blocks_and_sessions_never_reuse_ids() {
        let mut affinity = ChatAffinity::default();
        affinity.reuse_token_ids = true;
        for block in [
            json!({"type":"image","source":{"type":"base64","data":"image bytes"}}),
            json!({"type":"image","source":{"type":"url","url":"https://example.test/image.png"}}),
            json!({"type":"image_url","image_url":{"url":"https://example.test/image.png"}}),
            json!({"type":"document","source":{"data":"document"}}),
            json!({"type":"input_audio","input_audio":{"data":"audio"}}),
            json!({"type":"unknown","text":"looks textual"}),
            json!({"type":"text","text":{}}),
            json!({"type":"search_result","content":[{"type":"image","source":{"data":"image"}}]}),
            json!({"type":"search_result","source":{"type":"base64","data":"image"}}),
        ] {
            let nested = json!({"type":"tool_result","tool_use_id":"t","content":[block.clone()]});
            let deep = json!({"type":"tool_result","tool_use_id":"t2","content":[nested.clone()]});
            for content in [block, nested, deep] {
                let body = json!({"messages":[{"role":"user","content":[content]}]});
                assert!(
                    !supports_text_request(&body, SseProtocol::Anthropic),
                    "{body}"
                );
                assert!(affinity
                    .body_with_prompt_ids(&body, &[1], SseProtocol::Anthropic)
                    .is_none());
            }
        }
        for key in [
            "session_id",
            "session_params",
            "lora_path",
            "input_embeds",
            "image_data",
            "audio_data",
            "video_data",
            "mm_hashes",
            "n",
        ] {
            let mut body = native_tool_history();
            body[key] = json!(2);
            assert!(
                !supports_text_request(&body, SseProtocol::Anthropic),
                "{key}"
            );
            assert!(affinity
                .body_with_prompt_ids(&body, &[1], SseProtocol::Anthropic)
                .is_none());
        }
        let mut body = native_tool_history();
        body["system"] = json!([{"type":"image","source":{"data":"image"}}]);
        assert!(!supports_text_request(&body, SseProtocol::Anthropic));
        let mut block = json!({"type":"text","text":"deep"});
        for _ in 0..66 {
            block = json!({"type":"tool_result","content":[block]});
        }
        body["system"] = json!("system");
        body["messages"] = json!([{"role":"user","content":[block]}]);
        assert!(!supports_text_request(&body, SseProtocol::Anthropic));
    }

    #[tokio::test]
    async fn native_nested_media_is_rejected_before_tokenizer_transport() {
        use crate::core::BasicWorkerBuilder;
        let body = json!({"messages":[{"role":"user","content":[
            {"type":"tool_result","tool_use_id":"t","content":[{"type":"image","source":{"data":"image"}}]}
        ]}]});
        // Invalid URL would produce "transport" if the guard let this through.
        let worker = BasicWorkerBuilder::new("://invalid".to_owned()).build();
        assert_eq!(
            ChatAffinity::default()
                .try_tokenize(&Client::new(), &worker, &body, SseProtocol::Anthropic)
                .await
                .unwrap_err(),
            "unsupported_request"
        );
    }

    #[test]
    fn exact_prompt_reuse_keeps_tools_extensions_and_raw_input_immutable() {
        let raw = json!({
            "model":"glm", "messages":[{"role":"user","content":"code"}],
            "tools":[{"type":"function","function":{"name":"read","parameters":{}}}],
            "tool_choice":"required", "parallel_tool_calls":true, "stream":true,
            "chat_template_kwargs":{"clear_thinking":true},
            "cache_salt":"tenant-a", "unknown_extension":{"preserve":true}
        });
        let mut affinity = ChatAffinity::default();
        affinity.reuse_token_ids = false;
        assert!(affinity
            .body_with_prompt_ids(&raw, &[1, 2], SseProtocol::OpenAi)
            .is_none());
        affinity.reuse_token_ids = true;
        let mut forwarded = affinity
            .body_with_prompt_ids(&raw, &[1, 2], SseProtocol::OpenAi)
            .unwrap();
        assert_eq!(
            forwarded.as_object_mut().unwrap().remove("input_ids"),
            Some(json!([1, 2]))
        );
        assert_eq!(forwarded, raw);
        assert!(raw.get("input_ids").is_none());
        assert!(affinity
            .body_with_prompt_ids(&raw, &[], SseProtocol::OpenAi)
            .is_none());
        let media = json!({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"example"}}]}]});
        assert!(affinity
            .body_with_prompt_ids(&media, &[1, 2], SseProtocol::OpenAi)
            .is_none());
    }

    #[tokio::test]
    async fn transport_keeps_stream_true_and_rejects_malformed_token_responses() {
        use crate::core::BasicWorkerBuilder;
        use axum::{routing::post, Json};
        for protocol in [SseProtocol::OpenAi, SseProtocol::Anthropic] {
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
                let raw = if protocol == SseProtocol::Anthropic {
                    native_tool_history()
                } else {
                    json!({"messages":[{"role":"user","content":"hello"}], "stream":true, "return_spec_tokens_details":true})
                };
                let expected = raw.clone();
                let app = axum::Router::new().route(
                    protocol.tokenize_endpoint(),
                    post(move |Json(body): Json<Value>| {
                        assert_eq!(body, expected);
                        let result = result.clone();
                        async move { Json(result) }
                    }),
                );
                let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
                assert_eq!(
                    ChatAffinity::default()
                        .tokenize(&Client::new(), &worker, &raw, protocol)
                        .await
                        .is_some(),
                    expected_ok
                );
                task.abort();
            }
        }
    }

    #[test]
    fn eligibility_preserves_tools_but_excludes_media_and_sessions() {
        let text = json!({"messages": [{"role":"user", "content":[{"type":"text", "text":"hello"}]}, {"role":"assistant", "content":null, "tool_calls":[{"function":{"arguments":"{\"image_url\":\"ordinary tool argument\"}"}}]}]});
        assert!(supports_text_request(&text, SseProtocol::OpenAi));
        for media in [
            "image_url",
            "video_url",
            "input_audio",
            "audio_url",
            "unknown",
        ] {
            let mut body = text.clone();
            body["messages"][0]["content"][0]["type"] = json!(media);
            assert!(!supports_text_request(&body, SseProtocol::OpenAi));
        }
        for key in ["session_id", "lora_path", "input_embeds", "session_params"] {
            let mut body = text.clone();
            body[key] = json!("value");
            assert!(!supports_text_request(&body, SseProtocol::OpenAi));
        }
        let mut fanout = text;
        fanout["n"] = json!(2);
        assert!(!supports_text_request(&fanout, SseProtocol::OpenAi));
    }
}
