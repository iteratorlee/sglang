//! Native Messages envelope handling. Content/schema validation belongs to the
//! backend; retaining the raw object preserves new blocks and vendor extensions.
use axum::{
    body::{to_bytes, Body},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde_json::{json, Value};

pub(crate) fn validate(body: &Value, count_tokens: bool) -> Result<(), &'static str> {
    if !body.is_object() {
        return Err("Request must be a JSON object");
    }
    if !body
        .get("model")
        .and_then(Value::as_str)
        .is_some_and(|s| !s.is_empty())
    {
        return Err("model must be a nonempty string");
    }
    if !body
        .get("messages")
        .and_then(Value::as_array)
        .is_some_and(|m| !m.is_empty())
    {
        return Err("messages must be a nonempty array");
    }
    if !count_tokens
        && !body
            .get("max_tokens")
            .and_then(Value::as_u64)
            .is_some_and(|n| n > 0)
    {
        return Err("max_tokens must be a positive integer");
    }
    if body.get("stream").is_some_and(|s| !s.is_boolean()) {
        return Err("stream must be a boolean");
    }
    Ok(())
}

fn error_type(status: StatusCode) -> &'static str {
    match status.as_u16() {
        400 | 405 | 422 => "invalid_request_error",
        401 => "authentication_error",
        403 => "permission_error",
        404 => "not_found_error",
        413 => "request_too_large",
        429 => "rate_limit_error",
        503 | 529 => "overloaded_error",
        _ => "api_error",
    }
}

pub(crate) fn error(status: StatusCode, message: impl Into<String>) -> Response {
    (
        status,
        Json(json!({"type":"error", "error": {
            "type":error_type(status), "message":message.into()
        }})),
    )
        .into_response()
}

pub(crate) fn unsupported() -> Response {
    error(
        StatusCode::NOT_IMPLEMENTED,
        "Native Messages requires HTTP prefill/decode routing",
    )
}

/// Only error bodies are collected, with a hard cap. Successful generations and
/// native SSE pass through untouched. Preserve status, retry headers/extensions.
pub(crate) async fn normalize_error(response: Response) -> Response {
    if response.status().is_success() {
        return response;
    }
    let (mut parts, body) = response.into_parts();
    let bytes = to_bytes(body, 1024 * 1024).await.unwrap_or_default();
    let value = serde_json::from_slice::<Value>(&bytes).ok();
    let native = value.as_ref().is_some_and(|v| {
        v["type"] == "error" && v["error"]["type"].is_string() && v["error"]["message"].is_string()
    });
    let body = if native {
        Body::from(bytes)
    } else {
        let message = if parts.status.is_server_error() {
            // Upstream HTML, traceback text and transport diagnostics are not a
            // public API contract. Native, structured Anthropic errors above
            // retain their intended client-facing message.
            parts
                .status
                .canonical_reason()
                .unwrap_or("Upstream service failed")
                .to_owned()
        } else {
            value
                .as_ref()
                .and_then(|v| {
                    v.pointer("/error/message")
                        .or_else(|| v.get("message"))
                        .or_else(|| v.get("error"))
                })
                .and_then(Value::as_str)
                .map(str::to_owned)
                .unwrap_or_else(|| {
                    if bytes.is_empty() {
                        parts
                            .status
                            .canonical_reason()
                            .unwrap_or("Gateway error")
                            .to_owned()
                    } else {
                        String::from_utf8_lossy(&bytes).into_owned()
                    }
                })
        };
        error(parts.status, message).into_body()
    };
    parts.headers.remove(header::CONTENT_LENGTH);
    parts.headers.insert(
        header::CONTENT_TYPE,
        header::HeaderValue::from_static("application/json"),
    );
    Response::from_parts(parts, body)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_does_not_normalize_content_and_count_needs_no_max_tokens() {
        let mut body =
            json!({"model":"m", "messages":[{"role":"user","content":[{"type":"future_block"}]}]});
        assert!(validate(&body, true).is_ok());
        assert!(validate(&body, false).is_err());
        body["max_tokens"] = json!(10);
        assert!(validate(&body, false).is_ok());
        body["stream"] = json!("true");
        assert!(validate(&body, false).is_err());
        assert!(validate(&json!(null), true).is_err());
    }

    #[tokio::test]
    async fn nonnative_server_errors_do_not_expose_backend_text() {
        for body in [
            "Traceback: private path",
            r#"{"error":{"message":"private stack"}}"#,
        ] {
            let response = normalize_error((StatusCode::BAD_GATEWAY, body).into_response()).await;
            let bytes = to_bytes(response.into_body(), 4096).await.unwrap();
            assert!(!String::from_utf8_lossy(&bytes).contains("private"));
        }
    }

    #[tokio::test]
    async fn errors_preserve_status_headers_and_native_body() {
        for status in [
            StatusCode::BAD_REQUEST,
            StatusCode::UNAUTHORIZED,
            StatusCode::TOO_MANY_REQUESTS,
            StatusCode::SERVICE_UNAVAILABLE,
            StatusCode::NOT_IMPLEMENTED,
        ] {
            let mut response = (status, "backend rejected request").into_response();
            response
                .headers_mut()
                .insert("retry-after", "3".parse().unwrap());
            let response = normalize_error(response).await;
            assert_eq!(response.status(), status);
            assert_eq!(response.headers()["retry-after"], "3");
            let bytes = to_bytes(response.into_body(), 4096).await.unwrap();
            let value: Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(value["type"], "error");
            assert_eq!(value["error"]["type"], error_type(status));
            let again = normalize_error((status, bytes.clone()).into_response()).await;
            assert_eq!(to_bytes(again.into_body(), 4096).await.unwrap(), bytes);
        }
    }
}
