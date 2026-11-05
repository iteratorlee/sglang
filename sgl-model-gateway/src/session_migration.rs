//! Opt-in, bounded session movement. All counters are event-driven; no model polling.
use super::*;
use http_body::{Body as HttpBody, Frame, SizeHint};
use std::{
    pin::Pin,
    task::{Context, Poll},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MigrationConfig {
    pub enabled: bool,
    pub new_sessions_bps: u16,
    pub legacy_sessions_bps: u16,
    pub max_cold_inflight: usize,
    pub min_interval_secs: u64,
    pub quiet_secs: u64,
    pub max_switches: u64,
    pub session_hashes: Vec<String>,
}
impl Default for MigrationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            new_sessions_bps: 0,
            legacy_sessions_bps: 0,
            max_cold_inflight: 1,
            min_interval_secs: 30,
            quiet_secs: 10,
            max_switches: 1,
            session_hashes: Vec::new(),
        }
    }
}
impl MigrationConfig {
    fn valid(&self) -> bool {
        self.new_sessions_bps <= 10000
            && self.legacy_sessions_bps <= 10000
            && (1..=4).contains(&self.max_cold_inflight)
            && (1..=3600).contains(&self.min_interval_secs)
            && (5..=3600).contains(&self.quiet_secs)
            && self.max_switches <= MAX_PINS as u64
            && self.session_hashes.len() <= 256
            && self.session_hashes.iter().all(|s| valid_hash(s))
    }
}
pub(super) fn valid_hash(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

#[derive(Default)]
pub(super) struct Activity {
    pub active: usize,
    pub finished: Option<Instant>,
    pub warmed: bool,
    pub body_bytes: usize,
}
pub(super) struct Runtime {
    pub config: MigrationConfig,
    pub sessions: BTreeMap<String, Activity>,
    pub active_legacy: usize,
    pub active_candidate: usize,
    pub cold_inflight: usize,
    pub enrolled: u64,
    pub migrated: u64,
    pub last_assignment: Option<Instant>,
    pub started: Instant,
}
impl Default for Runtime {
    fn default() -> Self {
        Self {
            config: MigrationConfig::default(),
            sessions: BTreeMap::new(),
            active_legacy: 0,
            active_candidate: 0,
            cold_inflight: 0,
            enrolled: 0,
            migrated: 0,
            last_assignment: None,
            started: Instant::now(),
        }
    }
}
impl Runtime {
    pub fn snapshot(&self, pins: &BTreeMap<String, Destination>) -> Value {
        let now = Instant::now();
        let sessions: Vec<_> = self
            .sessions
            .iter()
            .take(256)
            .map(|(key, a)| {
                json!({
            "session_hash":key,"destination":pins.get(key),"in_flight":a.active,
            "idle_seconds":a.finished.map(|t|now.saturating_duration_since(t).as_secs_f64()),
            "warmed":a.warmed,"last_body_bytes":a.body_bytes})
            })
            .collect();
        json!({"config":self.config,"enrolled":self.enrolled,"migrated":self.migrated,
            "cold_inflight":self.cold_inflight,"active_legacy":self.active_legacy,
            "active_candidate":self.active_candidate,"tracked_sessions":self.sessions.len(),
            "sessions":sessions,"sessions_limit":256})
    }
    pub fn assignment_allowed(&self, key: &str, known: bool, now: Instant) -> bool {
        let c = &self.config;
        if !c.enabled
            || self.enrolled + self.migrated >= c.max_switches
            || self.cold_inflight >= c.max_cold_inflight
            || now.saturating_duration_since(self.started) < Duration::from_secs(c.quiet_secs)
            || self.last_assignment.is_some_and(|t| {
                now.saturating_duration_since(t) < Duration::from_secs(c.min_interval_secs)
            })
        {
            return false;
        }
        let bucket = (u64::from_str_radix(&key[..16], 16).unwrap() % 10000) as u16;
        if known {
            let Some(a) = self.sessions.get(key) else {
                return false;
            };
            if a.active != 0
                || !a.finished.is_some_and(|t| {
                    now.saturating_duration_since(t) >= Duration::from_secs(c.quiet_secs)
                })
            {
                return false;
            }
            if !c.session_hashes.is_empty() {
                c.session_hashes.iter().any(|s| s == key)
            } else {
                bucket < c.legacy_sessions_bps
            }
        } else {
            bucket < c.new_sessions_bps
        }
    }
    fn gauges(&self) {
        metrics::gauge!("smg_session_migration_active_requests","destination"=>"legacy")
            .set(self.active_legacy as f64);
        metrics::gauge!("smg_session_migration_active_requests","destination"=>"candidate")
            .set(self.active_candidate as f64);
        metrics::gauge!("smg_session_migration_cold_inflight").set(self.cold_inflight as f64);
    }
}

pub(super) struct Lease {
    pub runtime: Arc<parking_lot::Mutex<Runtime>>,
    pub key: Option<String>,
    pub destination: Destination,
    pub cold: bool,
    pub completed: bool,
}
impl Drop for Lease {
    fn drop(&mut self) {
        let mut r = self.runtime.lock();
        match self.destination {
            Destination::Legacy => r.active_legacy -= 1,
            Destination::Candidate => r.active_candidate -= 1,
        }
        if self.cold {
            r.cold_inflight -= 1;
        }
        if let Some(a) = self.key.as_ref().and_then(|k| r.sessions.get_mut(k)) {
            a.active -= 1;
            a.finished = Some(Instant::now());
            if self.destination == Destination::Candidate && self.completed {
                a.warmed = true;
            }
        }
        r.gauges();
    }
}
pub(super) fn begin(
    runtime: &Arc<parking_lot::Mutex<Runtime>>,
    r: &mut Runtime,
    key: Option<String>,
    destination: Destination,
    body_bytes: usize,
    cold: bool,
) -> Lease {
    let key = key.filter(|k| r.sessions.contains_key(k) || r.sessions.len() < MAX_PINS);
    if let Some(k) = key.as_ref() {
        let a = r.sessions.entry(k.clone()).or_default();
        a.active += 1;
        if body_bytes > 0 {
            a.body_bytes = body_bytes;
        }
    }
    match destination {
        Destination::Legacy => r.active_legacy += 1,
        Destination::Candidate => r.active_candidate += 1,
    }
    if cold {
        r.cold_inflight += 1;
    }
    r.gauges();
    Lease {
        runtime: runtime.clone(),
        key,
        destination,
        cold,
        completed: false,
    }
}

struct GuardedBody {
    inner: Body,
    lease: Option<Lease>,
    successful: bool,
}
impl HttpBody for GuardedBody {
    type Data = bytes::Bytes;
    type Error = axum::Error;
    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        let this = self.get_mut();
        match Pin::new(&mut this.inner).poll_frame(cx) {
            Poll::Ready(None) => {
                if let Some(mut lease) = this.lease.take() {
                    lease.completed = this.successful;
                    drop(lease);
                }
                Poll::Ready(None)
            }
            Poll::Ready(Some(Err(e))) => {
                this.successful = false;
                this.lease.take();
                Poll::Ready(Some(Err(e)))
            }
            Poll::Ready(Some(Ok(frame))) => {
                if this.inner.is_end_stream() {
                    if let Some(mut lease) = this.lease.take() {
                        lease.completed = this.successful;
                        drop(lease);
                    }
                }
                Poll::Ready(Some(Ok(frame)))
            }
            other => other,
        }
    }
    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }
    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}
pub(super) fn wrap(response: Response, lease: Lease) -> Response {
    let successful = response.status().is_success();
    let (parts, body) = response.into_parts();
    Response::from_parts(
        parts,
        Body::new(GuardedBody {
            inner: body,
            lease: Some(lease),
            successful,
        }),
    )
}

pub(crate) async fn status(State(front): State<Arc<SessionHandoff>>) -> Json<Value> {
    let store = front.store.lock().await;
    Json(front.migration.lock().snapshot(&store.pins))
}
pub(crate) async fn configure(
    State(front): State<Arc<SessionHandoff>>,
    Json(config): Json<MigrationConfig>,
) -> Response {
    if !config.valid() {
        return (StatusCode::BAD_REQUEST, "invalid migration bounds").into_response();
    }
    let _store = front.store.lock().await;
    let mut r = front.migration.lock();
    // Lowering a limit pauses new switches; it never moves existing candidates back.
    r.config = config.clone();
    Json(config).into_response()
}
pub(crate) async fn import_pins(
    State(front): State<Arc<SessionHandoff>>,
    Json(incoming): Json<BTreeMap<String, Destination>>,
) -> Response {
    let mut store = front.store.clone().lock_owned().await;
    if store.config.candidate_bps != 0 || front.migration.lock().config.enabled {
        return (
            StatusCode::CONFLICT,
            "pause enrollment and migration before importing",
        )
            .into_response();
    }
    if store.failed || incoming.len() > MAX_PINS || incoming.keys().any(|s| !valid_hash(s)) {
        return (StatusCode::BAD_REQUEST, "invalid pin import").into_response();
    }
    let mut updated = store.pins.clone();
    for (key, value) in incoming {
        if updated.get(&key).is_some_and(|v| *v != value) {
            return (
                StatusCode::CONFLICT,
                "pin import cannot overwrite a destination",
            )
                .into_response();
        }
        updated.insert(key, value);
    }
    if updated.len() > MAX_PINS {
        return (StatusCode::CONFLICT, "pin capacity exceeded").into_response();
    }
    let added = updated.len() - store.pins.len();
    let result = tokio::task::spawn_blocking(move || {
        if store.persist(&updated).is_err() {
            let uncertain: Vec<_> = updated
                .iter()
                .filter(|(k, v)| store.pins.get(*k) != Some(*v))
                .map(|(k, _)| k.clone())
                .collect();
            store.quarantined.extend(uncertain);
            store.failed = true;
            return false;
        }
        store.pins = updated;
        true
    })
    .await;
    if !matches!(result, Ok(true)) {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "pin import persistence unavailable",
        )
            .into_response();
    }
    Json(json!({"added":added})).into_response()
}

/// Rich chat payloads keep their bytes, messages, tools and model parameters unchanged.
/// Stateful continuation APIs are not enrolled by this path.
pub(super) fn full_chat_context(bytes: &[u8], path: &str) -> bool {
    if !matches!(path, "/v1/messages" | "/v1/chat/completions") {
        return false;
    }
    let Ok(v) = serde_json::from_slice::<Value>(bytes) else {
        return false;
    };
    !["session_params", "previous_response_id", "conversation"]
        .iter()
        .any(|s| v.get(s).is_some())
        && v.get("messages")
            .and_then(Value::as_array)
            .is_some_and(|v| !v.is_empty())
}

#[cfg(test)]
#[path = "session_migration_tests.rs"]
mod tests;
