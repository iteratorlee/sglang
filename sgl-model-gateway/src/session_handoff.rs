//! Conservative, process-preserving gateway handoff. Legacy traffic never
//! enters the local PD router/experience observer. Only a durable candidate pin
//! can enter it. Pins contain hashed session IDs and destinations, never bodies.
//!
//! Enable with SGLANG_GATEWAY_HANDOFF_LEGACY_ORIGIN=http://127.0.0.1:30266 and
//! SGLANG_GATEWAY_HANDOFF_PIN_FILE=/absolute/persistent/path.json. Initial
//! admission is zero. PUT /workers/session-handoff changes candidate_bps and
//! bounded_first_turn (default false, capped at 5000 bps when enabled).
//! Explicit enrollment: x-sglang-new-session: 1 plus a supported session ID.
//! Bounded first turn describes API history, NOT proof of a historically new
//! session; old short sessions can match this heuristic. Stable IDs must be
//! retained on every subsequent request. Pins have
//! no TTL/eviction; a full ledger stops enrollment. A restart restores pins,
//! but cannot restore this gateway's PD prefix book or in-flight leases.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{File, OpenOptions},
    io::{self, Read, Write},
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

use axum::{
    body::{to_bytes, Body},
    extract::{Request, State},
    http::{HeaderMap, HeaderValue, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use subtle::ConstantTimeEq;
use tokio::sync::Mutex;

use crate::middleware::AuthConfig;

#[path = "session_migration.rs"]
pub(crate) mod migration;

pub(crate) const CANDIDATE_HEADER: &str = "x-sglang-handoff-candidate";
const HOP_HEADER: &str = "x-sglang-handoff-hop";
const MAX_FIRST_TURN: usize = 64 * 1024;
const MAX_PINS: usize = 65_536;
const MAX_PIN_FILE: u64 = 8 * 1024 * 1024;

#[derive(Clone, Copy)]
pub(crate) struct HandoffIngress(pub Instant);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Destination {
    Legacy,
    Candidate,
}

impl Destination {
    fn label(self) -> &'static str {
        match self {
            Self::Legacy => "legacy",
            Self::Candidate => "candidate",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HandoffConfig {
    pub candidate_bps: u16,
    #[serde(default)]
    pub bounded_first_turn: bool,
}

impl HandoffConfig {
    fn validate(self) -> Result<Self, &'static str> {
        if self.candidate_bps > 10_000 || (self.bounded_first_turn && self.candidate_bps > 5000) {
            Err("candidate_bps must be 0..10000, and at most 5000 with bounded_first_turn")
        } else {
            Ok(self)
        }
    }
}

struct PinStore {
    path: PathBuf,
    _lock: File,
    pins: BTreeMap<String, Destination>,
    config: HandoffConfig,
    failed: bool,
    quarantined: BTreeSet<String>,
    #[cfg(test)]
    fail_after_rename: bool,
}

fn private_file(path: &Path, exclusive: bool) -> io::Result<File> {
    let mut options = OpenOptions::new();
    options.read(true).write(true);
    if exclusive {
        options.create_new(true);
    } else {
        options.create(true);
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options.open(path)
}

fn lock_ledger(file: &File) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::{fd::AsRawFd, raw::c_int};
        // std::fs::File::try_lock requires Rust 1.89. The deployment builder
        // uses 1.86; flock is provided by the system libc on Linux/Ascend.
        unsafe extern "C" {
            fn flock(fd: c_int, operation: c_int) -> c_int;
        }
        // SAFETY: File owns a live descriptor for the full lock lifetime.
        // LOCK_EX=2, LOCK_NB=4 on the supported Unix deployment platforms.
        if unsafe { flock(file.as_raw_fd(), 2 | 4) } != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }
    #[cfg(not(unix))]
    {
        let _ = file;
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "handoff requires Unix file locking",
        ))
    }
}

impl PinStore {
    fn open(path: PathBuf) -> io::Result<Self> {
        if !path.is_absolute() || path.file_name().is_none() {
            return Err(io::Error::other("pin file must be an absolute file path"));
        }
        // Lock a separate inode: the ledger inode is replaced by rename. The
        // OS releases this lock on exit/crash. Never run two writers per ledger.
        let lock = private_file(&path.with_extension("handoff.lock"), false)?;
        lock_ledger(&lock)?;
        let pins: BTreeMap<String, Destination> = match File::open(&path) {
            Ok(file) => {
                let mut bytes = Vec::new();
                file.take(MAX_PIN_FILE + 1).read_to_end(&mut bytes)?;
                if bytes.len() as u64 > MAX_PIN_FILE {
                    return Err(io::Error::other("pin file too large"));
                }
                serde_json::from_slice(&bytes).map_err(|_| io::Error::other("invalid pin file"))?
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => BTreeMap::new(),
            Err(e) => return Err(e),
        };
        if pins.len() > MAX_PINS
            || pins
                .keys()
                .any(|key| key.len() != 64 || !key.bytes().all(|b| b.is_ascii_hexdigit()))
        {
            return Err(io::Error::other("invalid pin keys or too many pins"));
        }
        let store = Self {
            path,
            _lock: lock,
            pins,
            config: HandoffConfig::default(),
            failed: false,
            quarantined: BTreeSet::new(),
            #[cfg(test)]
            fail_after_rename: false,
        };
        // Prove directory/file durability is usable before accepting traffic.
        store.persist(&store.pins)?;
        Ok(store)
    }

    fn persist(&self, pins: &BTreeMap<String, Destination>) -> io::Result<()> {
        let temporary = self
            .path
            .with_extension(format!("{}.tmp", uuid::Uuid::new_v4()));
        let result = (|| {
            let mut file = private_file(&temporary, true)?;
            serde_json::to_writer(&mut file, pins)?;
            file.write_all(b"\n")?;
            file.sync_all()?;
            std::fs::rename(&temporary, &self.path)?;
            #[cfg(test)]
            if self.fail_after_rename {
                return Err(io::Error::other("injected post-rename sync failure"));
            }
            File::open(self.path.parent().unwrap())?.sync_all()
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(temporary);
        }
        result
    }
}

pub struct SessionHandoff {
    origin: String,
    client: reqwest::Client,
    store: Arc<Mutex<PinStore>>,
    max_body: usize,
    auth: AuthConfig,
    migration: Arc<parking_lot::Mutex<migration::Runtime>>,
}

impl SessionHandoff {
    pub(crate) fn from_env(
        port: u16,
        max_body: usize,
        auth: AuthConfig,
    ) -> Result<Option<Arc<Self>>, String> {
        let Ok(origin) = std::env::var("SGLANG_GATEWAY_HANDOFF_LEGACY_ORIGIN") else {
            return Ok(None);
        };
        if origin != "http://127.0.0.1:30266" || port == 30266 {
            return Err(
                "handoff requires legacy origin http://127.0.0.1:30266 and a different listen port"
                    .into(),
            );
        }
        let path = std::env::var("SGLANG_GATEWAY_HANDOFF_PIN_FILE")
            .map_err(|_| "handoff requires SGLANG_GATEWAY_HANDOFF_PIN_FILE")?;
        Self::open(origin, PathBuf::from(path), max_body, auth).map(Some)
    }

    fn open(
        origin: String,
        path: PathBuf,
        max_body: usize,
        auth: AuthConfig,
    ) -> Result<Arc<Self>, String> {
        let store = PinStore::open(path)
            .map_err(|e| format!("cannot open handoff ledger: {}", e.kind()))?;
        let client = reqwest::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(Duration::from_secs(5))
            .no_gzip()
            .no_brotli()
            .no_deflate()
            .no_zstd()
            .build()
            .map_err(|_| "cannot build legacy client")?;
        Ok(Arc::new(Self {
            origin,
            client,
            store: Arc::new(Mutex::new(store)),
            max_body,
            auth,
            migration: Arc::new(parking_lot::Mutex::new(migration::Runtime::default())),
        }))
    }

    #[cfg(test)]
    async fn decide(
        &self,
        key: Option<String>,
        enrollment: Option<&'static str>,
    ) -> Result<(Destination, &'static str), ()> {
        self.choose(key, enrollment, false, 0)
            .await
            .map(|(d, r, _)| (d, r))
            .map_err(|_| ())
    }

    async fn choose(
        &self,
        key: Option<String>,
        enrollment: Option<&'static str>,
        full_context: bool,
        body_bytes: usize,
    ) -> Result<(Destination, &'static str, migration::Lease), StatusCode> {
        let mut store = self.store.clone().lock_owned().await;
        let current = key.as_ref().and_then(|k| store.pins.get(k)).copied();
        let (destination, reason, assignment, migration_assignment, lease) = {
            let mut r = self.migration.lock();
            let mut destination = current.unwrap_or(Destination::Legacy);
            let mut reason = if current.is_some() {
                "pinned"
            } else {
                "no_identity"
            };
            let mut assignment = false;
            let mut migration_assignment = false;
            if let Some(k) = key.as_ref() {
                if store.quarantined.contains(k) {
                    return Err(StatusCode::SERVICE_UNAVAILABLE);
                }
                if store.failed && current.is_none() {
                    return Err(StatusCode::SERVICE_UNAVAILABLE);
                }
                if current.is_none() && store.pins.len() >= MAX_PINS {
                    reason = "ledger_full";
                } else {
                    let old_eligible = enrollment
                        .is_some_and(|kind| kind == "explicit" || store.config.bounded_first_turn);
                    let bucket = u64::from_str_radix(&k[..16], 16).unwrap() % 10000;
                    let cohort = current.is_none()
                        && old_eligible
                        && bucket < u64::from(store.config.candidate_bps);
                    let moving = destination == Destination::Legacy
                        && full_context
                        && !store.failed
                        && r.assignment_allowed(k, current.is_some(), Instant::now());
                    if moving || cohort {
                        // A session is never moved while another tracked request/body is live.
                        let quiet = r.sessions.get(k).is_none_or(|a| a.active == 0);
                        if quiet && r.cold_inflight < r.config.max_cold_inflight {
                            destination = Destination::Candidate;
                            assignment = true;
                            migration_assignment = moving;
                            reason = if moving {
                                if current.is_some() {
                                    "migrated"
                                } else {
                                    "new_history"
                                }
                            } else {
                                enrollment.unwrap()
                            };
                        } else {
                            reason = "cold_budget";
                        }
                    } else if current.is_none() {
                        reason = if store.config.candidate_bps == 0 {
                            "disabled"
                        } else if old_eligible {
                            "cohort_legacy"
                        } else {
                            "not_enrolled"
                        };
                    }
                }
            }
            let cold = destination == Destination::Candidate
                && !key
                    .as_ref()
                    .and_then(|k| r.sessions.get(k))
                    .is_some_and(|a| a.warmed);
            if cold && r.cold_inflight >= r.config.max_cold_inflight {
                // Existing candidate pins cannot fall back to legacy when the budget is full.
                return Err(StatusCode::TOO_MANY_REQUESTS);
            }
            let lease = migration::begin(
                &self.migration,
                &mut r,
                key.clone(),
                destination,
                body_bytes,
                cold,
            );
            (destination, reason, assignment, migration_assignment, lease)
        };
        let needs_pin = key
            .as_ref()
            .is_some_and(|k| store.pins.get(k).copied() != Some(destination))
            && (current.is_some() || store.pins.len() < MAX_PINS);
        if needs_pin {
            let key = key.unwrap();
            let runtime = self.migration.clone();
            let known = current.is_some();
            let result=tokio::task::spawn_blocking(move || {
                let mut updated=store.pins.clone();updated.insert(key.clone(),destination);
                if store.persist(&updated).is_err() {store.failed=true;store.quarantined.insert(key);return false;}
                store.pins=updated;
                if assignment {
                    let mut r=runtime.lock();
                    if known {r.migrated+=1;} else {r.enrolled+=1;}
                    if migration_assignment {r.last_assignment=Some(Instant::now());}
                    metrics::counter!("smg_session_migration_assignments_total","kind"=>if known {"migrated"}else{"new"}).increment(1);
                }
                true
            }).await;
            if !matches!(result, Ok(true)) {
                return Err(StatusCode::SERVICE_UNAVAILABLE);
            }
        }
        Ok((destination, reason, lease))
    }

    async fn proxy(&self, request: Request) -> Response {
        let (parts, body) = request.into_parts();
        let path = parts
            .uri
            .path_and_query()
            .map(|v| v.as_str())
            .unwrap_or("/");
        let mut headers = end_to_end_headers(&parts.headers);
        headers.remove("host");
        headers.insert(HOP_HEADER, HeaderValue::from_static("1"));
        let upstream = self
            .client
            .request(parts.method, format!("{}{}", self.origin, path))
            .headers(headers)
            .body(reqwest::Body::wrap_stream(body.into_data_stream()))
            .send()
            .await;
        match upstream {
            Ok(upstream) => {
                let status = upstream.status();
                let headers = end_to_end_headers(upstream.headers());
                let mut response = Response::new(Body::from_stream(upstream.bytes_stream()));
                *response.status_mut() = status;
                *response.headers_mut() = headers;
                response
            }
            Err(_) => (StatusCode::BAD_GATEWAY, "legacy gateway unavailable").into_response(),
        }
    }
}

fn end_to_end_headers(input: &HeaderMap) -> HeaderMap {
    let mut result = input.clone();
    for value in input.get_all("connection") {
        if let Ok(value) = value.to_str() {
            for name in value.split(',').map(str::trim) {
                result.remove(name);
            }
        }
    }
    for name in [
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    ] {
        result.remove(name);
    }
    result
}

fn session_key(id: &str) -> Option<String> {
    if id.is_empty() || id.len() > 256 || !id.bytes().all(|b| b.is_ascii_graphic()) {
        return None;
    }
    Some(
        blake3::Hash::from(blake3::derive_key(
            "sglang gateway handoff session v1",
            id.as_bytes(),
        ))
        .to_hex()
        .to_string(),
    )
}

fn header_identity(headers: &HeaderMap) -> Result<Option<String>, ()> {
    let mut key = None;
    for name in ["x-sglang-session-id", "x-opencode-session-id"] {
        for value in headers.get_all(name) {
            let current = value.to_str().ok().and_then(session_key).ok_or(())?;
            if key.as_ref().is_some_and(|old| old != &current) {
                return Err(());
            }
            key = Some(current);
        }
    }
    Ok(key)
}

#[derive(Deserialize)]
struct SessionMetadata {
    session_id: Option<String>,
    user_id: Option<String>,
}
#[derive(Deserialize)]
struct Envelope {
    metadata: Option<SessionMetadata>,
}

fn body_identity(bytes: &[u8]) -> (Option<String>, &'static str) {
    let Ok(Envelope {
        metadata: Some(metadata),
    }) = serde_json::from_slice(bytes)
    else {
        return (None, "none");
    };
    let direct = metadata.session_id.as_deref().and_then(session_key);
    let nested = metadata
        .user_id
        .as_deref()
        .filter(|s| s.len() <= 4096)
        .and_then(|s| serde_json::from_str::<Value>(s).ok())
        .and_then(|v| {
            v.get("session_id")
                .and_then(Value::as_str)
                .and_then(session_key)
        });
    match (direct, nested) {
        (Some(a), Some(b)) if a != b => (None, "ambiguous"),
        (Some(a), _) => (Some(a), "metadata"),
        (_, Some(b)) => (Some(b), "claude"),
        _ => (None, "none"),
    }
}

fn text_content(value: &Value) -> bool {
    value.as_str().is_some_and(|s| !s.is_empty())
        || value.as_array().is_some_and(|blocks| {
            !blocks.is_empty()
                && blocks.iter().all(|b| {
                    b.get("type").and_then(Value::as_str) == Some("text")
                        && b.get("text")
                            .and_then(Value::as_str)
                            .is_some_and(|s| !s.is_empty())
                })
        })
}

fn bounded_first_turn(bytes: &[u8]) -> bool {
    if bytes.len() > MAX_FIRST_TURN {
        return false;
    }
    let Ok(value) = serde_json::from_slice::<Value>(bytes) else {
        return false;
    };
    if value.get("system").is_some_and(|s| !text_content(s)) {
        return false;
    }
    if [
        "image_data",
        "images",
        "audio",
        "video",
        "documents",
        "input_audio",
    ]
    .iter()
    .any(|k| value.get(k).is_some())
    {
        return false;
    }
    if value
        .get("modalities")
        .is_some_and(|v| v != &json!(["text"]))
    {
        return false;
    }
    let Some(messages) = value.get("messages").and_then(Value::as_array) else {
        return false;
    };
    let mut users = 0;
    for message in messages {
        if !message.get("content").is_some_and(text_content)
            || ["tool_calls", "tool_call_id", "function_call"]
                .iter()
                .any(|k| message.get(k).is_some())
        {
            return false;
        }
        match message.get("role").and_then(Value::as_str) {
            Some("system" | "developer") if users == 0 => (),
            Some("user") => users += 1,
            _ => return false,
        }
    }
    users == 1
}

fn managed_path(method: &str, path: &str) -> bool {
    (method == "POST"
        && matches!(
            path,
            "/v1/messages"
                | "/v1/chat/completions"
                | "/generate"
                | "/v1/completions"
                | "/v1/messages/count_tokens"
        ))
        || (method == "GET" && matches!(path, "/v1/models" | "/model_info" | "/get_model_info"))
}

fn authorized(auth: &AuthConfig, request: &Request) -> bool {
    let Some(expected) = auth.api_key.as_deref() else {
        return true;
    };
    let token = request
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "))
        .or_else(|| {
            request
                .uri()
                .path()
                .starts_with("/v1/messages")
                .then(|| {
                    request
                        .headers()
                        .get("x-api-key")
                        .and_then(|v| v.to_str().ok())
                })
                .flatten()
        });
    token.is_some_and(|token| token.as_bytes().ct_eq(expected.as_bytes()).into())
}

pub(crate) async fn dispatch(
    State(front): State<Arc<SessionHandoff>>,
    mut request: Request,
    next: Next,
) -> Response {
    request.headers_mut().remove(CANDIDATE_HEADER); // never trust an inbound internal marker
    let path = request.uri().path().to_owned();
    let method = request.method().as_str().to_owned();
    if !managed_path(&method, &path) {
        return next.run(request).await;
    }
    let start = Instant::now();
    metrics::counter!("smg_http_requests_total", "scope" => "handoff_front", "method" => method.clone(), "path" => path.clone()).increment(1);
    let response = if request.headers().contains_key(HOP_HEADER) {
        (StatusCode::LOOP_DETECTED, "handoff proxy loop").into_response()
    } else if !authorized(&front.auth, &request) {
        StatusCode::UNAUTHORIZED.into_response()
    } else {
        route(&front, request, next, start).await
    };
    metrics::histogram!("smg_http_request_duration_seconds", "scope" => "handoff_front", "method" => method, "path" => path).record(start.elapsed().as_secs_f64());
    metrics::counter!("smg_http_responses_total", "scope" => "handoff_front", "status_code" => response.status().as_u16().to_string(), "error_code" => "").increment(1);
    response
}

async fn route(
    front: &SessionHandoff,
    mut request: Request,
    next: Next,
    start: Instant,
) -> Response {
    let generation = request.method() == "POST"
        && matches!(
            request.uri().path(),
            "/v1/messages" | "/v1/chat/completions" | "/generate" | "/v1/completions"
        );
    if !generation {
        record(Destination::Legacy, "none", "endpoint");
        return front.proxy(request).await;
    }
    let explicit = request
        .headers()
        .get_all("x-sglang-new-session")
        .iter()
        .count()
        == 1
        && request
            .headers()
            .get("x-sglang-new-session")
            .is_some_and(|v| v == "1");
    let (mut key, mut source) = match header_identity(request.headers()) {
        Ok(Some(key)) => (Some(key), "header"),
        Ok(None) => (None, "none"),
        Err(()) => {
            record(Destination::Legacy, "ambiguous", "no_identity");
            return front.proxy(request).await;
        }
    };
    let existing = if let Some(key) = key.as_ref() {
        front.store.lock().await.pins.get(key).copied()
    } else {
        None
    };
    let known = existing.is_some();
    let inspect_history =
        front.migration.lock().config.enabled && existing != Some(Destination::Candidate);
    let auto = front.store.lock().await.config.bounded_first_turn;
    let mut first_turn = false;
    let mut full_context = false;
    let mut body_bytes = 0;
    if key.is_none() || (!known && !explicit && auto) || inspect_history {
        // A header ID needs no buffering for explicit enrollment or pinned
        // continuations (including MM). Body IDs require reading the envelope;
        // serde ignores messages during identity extraction. Preserve bytes.
        if request.headers().contains_key("content-encoding") {
            // A body envelope cannot be inspected without decoding it. With a
            // stable header, still retain its pin and stream lease; never bypass
            // activity protection or silently redirect an existing candidate.
            if key.is_none() {
                return (
                    StatusCode::BAD_REQUEST,
                    "encoded bodies require a stable session header",
                )
                    .into_response();
            }
            return match front.choose(key, None, false, 0).await {
                Ok((destination, _, lease)) => {
                    record(destination, source, "encoded_body");
                    let response = match destination {
                        Destination::Legacy => front.proxy(request).await,
                        Destination::Candidate => {
                            request
                                .headers_mut()
                                .insert(CANDIDATE_HEADER, HeaderValue::from_static("1"));
                            request.extensions_mut().insert(HandoffIngress(start));
                            next.run(request).await
                        }
                    };
                    migration::wrap(response, lease)
                }
                Err(status) => (status, "encoded handoff unavailable").into_response(),
            };
        }
        let (parts, body) = request.into_parts();
        let bytes = match to_bytes(body, front.max_body).await {
            Ok(bytes) => bytes,
            Err(_) => return StatusCode::PAYLOAD_TOO_LARGE.into_response(),
        };
        if key.is_none() {
            (key, source) = body_identity(&bytes);
        }
        body_bytes = bytes.len();
        if inspect_history {
            full_context = migration::full_chat_context(&bytes, parts.uri.path());
        }
        if auto {
            first_turn = bounded_first_turn(&bytes);
        }
        request = Request::from_parts(parts, Body::from(bytes));
    }
    let enrollment = if explicit {
        Some("explicit")
    } else if first_turn {
        Some("bounded_first_turn")
    } else {
        None
    };
    match front
        .choose(key, enrollment, full_context, body_bytes)
        .await
    {
        Ok((destination, reason, lease)) => {
            record(destination, source, reason);
            let response = match destination {
                Destination::Legacy => front.proxy(request).await,
                Destination::Candidate => {
                    request
                        .headers_mut()
                        .insert(CANDIDATE_HEADER, HeaderValue::from_static("1"));
                    request.extensions_mut().insert(HandoffIngress(start));
                    next.run(request).await
                }
            };
            migration::wrap(response, lease)
        }
        Err(status) => {
            metrics::counter!("smg_session_handoff_requests_total", "destination" => "reject", "source" => source, "reason" => if status==StatusCode::TOO_MANY_REQUESTS {"cold_budget"} else {"persist_failed"}).increment(1);
            let mut response =
                (status, "handoff unavailable or cold-session budget busy").into_response();
            if status == StatusCode::TOO_MANY_REQUESTS {
                response
                    .headers_mut()
                    .insert("retry-after", HeaderValue::from_static("1"));
            }
            response
        }
    }
}

fn record(destination: Destination, source: &'static str, reason: &'static str) {
    metrics::counter!("smg_session_handoff_requests_total", "destination" => destination.label(), "source" => source, "reason" => reason).increment(1);
}

pub(crate) async fn status(State(front): State<Arc<SessionHandoff>>) -> Json<Value> {
    let store = front.store.lock().await;
    Json(
        json!({"config":store.config, "legacy_pins":store.pins.values().filter(|d| **d == Destination::Legacy).count(),
        "candidate_pins":store.pins.values().filter(|d| **d == Destination::Candidate).count(), "persistence_failed":store.failed}),
    )
}

pub(crate) async fn configure(
    State(front): State<Arc<SessionHandoff>>,
    Json(config): Json<HandoffConfig>,
) -> Response {
    match config.validate() {
        Ok(config) => {
            front.store.lock().await.config = config;
            Json(config).into_response()
        }
        Err(message) => (StatusCode::BAD_REQUEST, message).into_response(),
    }
}

#[cfg(test)]
#[path = "session_handoff_tests.rs"]
mod tests;
