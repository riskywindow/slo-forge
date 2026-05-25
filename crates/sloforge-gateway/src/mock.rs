//! Deterministic HTTP mock inference backend used by the CPU demo and contract tests.
//!
//! This module is intentionally explicit test infrastructure; gateway production paths never
//! silently substitute it for a configured engine.

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sloforge_telemetry::{Labels, MetricsRegistry};
use std::convert::Infallible;
use std::fmt::Write as _;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;
use tokio::sync::Semaphore;

const NO_TOKEN_FAULT: usize = usize::MAX;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MockBackendConfig {
    pub name: String,
    #[serde(default = "default_bind")]
    pub bind: String,
    #[serde(default)]
    pub seed: u64,
    #[serde(default)]
    pub startup_ms: u64,
    #[serde(default)]
    pub startup_jitter_ms: u64,
    #[serde(default)]
    pub startup_every_n_requests: u64,
    #[serde(default = "default_prefill_base_ms")]
    pub prefill_base_ms: u64,
    #[serde(default = "default_prefill_per_token_us")]
    pub prefill_per_token_us: u64,
    #[serde(default = "default_decode_per_token_ms")]
    pub decode_per_token_ms: u64,
    #[serde(default = "default_concurrency")]
    pub max_concurrency: usize,
    #[serde(default = "default_max_output_tokens")]
    pub max_output_tokens: u32,
    #[serde(default)]
    pub price_per_hour_usd: f64,
    #[serde(default)]
    pub failure_rate: f64,
}

#[derive(Debug, thiserror::Error)]
pub enum MockConfigError {
    #[error("failed to read mock config: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid mock JSON config: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid mock config: {0}")]
    Invalid(String),
}

impl MockBackendConfig {
    /// Read and validate a deterministic mock backend configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be read, parsed, or validated.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, MockConfigError> {
        let bytes = std::fs::read(path)?;
        let config: Self = serde_json::from_slice(&bytes)?;
        config.validate()?;
        Ok(config)
    }

    /// Validate limits, probabilities, and pricing metadata.
    ///
    /// # Errors
    ///
    /// Returns an error describing the first invalid field.
    pub fn validate(&self) -> Result<(), MockConfigError> {
        if self.name.trim().is_empty() || self.max_concurrency == 0 || self.max_output_tokens == 0 {
            return Err(MockConfigError::Invalid(
                "name and positive concurrency/output limits are required".to_owned(),
            ));
        }
        if !(0.0..=1.0).contains(&self.failure_rate)
            || !self.price_per_hour_usd.is_finite()
            || self.price_per_hour_usd < 0.0
        {
            return Err(MockConfigError::Invalid(
                "failure_rate must be in [0,1] and price must be non-negative".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug)]
struct FaultState {
    crashed: AtomicBool,
    slowdown_bits: AtomicU64,
    cold_start_next_ms: AtomicU64,
    malformed_after_tokens: AtomicUsize,
    disconnect_after_tokens: AtomicUsize,
    request_error_count: AtomicU64,
}

impl Default for FaultState {
    fn default() -> Self {
        Self {
            crashed: AtomicBool::new(false),
            slowdown_bits: AtomicU64::new(1.0_f64.to_bits()),
            cold_start_next_ms: AtomicU64::new(0),
            malformed_after_tokens: AtomicUsize::new(NO_TOKEN_FAULT),
            disconnect_after_tokens: AtomicUsize::new(NO_TOKEN_FAULT),
            request_error_count: AtomicU64::new(0),
        }
    }
}

pub struct MockBackend {
    config: MockBackendConfig,
    capacity: Arc<Semaphore>,
    sequence: AtomicU64,
    faults: FaultState,
    pub metrics: MetricsRegistry,
}

impl MockBackend {
    /// Construct a deterministic backend with bounded concurrency.
    ///
    /// # Errors
    ///
    /// Returns an error if the mock configuration is invalid.
    pub fn new(config: MockBackendConfig) -> Result<Arc<Self>, MockConfigError> {
        config.validate()?;
        Ok(Arc::new(Self {
            capacity: Arc::new(Semaphore::new(config.max_concurrency)),
            config,
            sequence: AtomicU64::new(0),
            faults: FaultState::default(),
            metrics: MetricsRegistry::default(),
        }))
    }

    pub fn router(self: &Arc<Self>) -> Router {
        Router::new()
            .route("/v1/completions", post(completion))
            .route("/v1/chat/completions", post(chat_completion))
            .route("/health", get(mock_health))
            .route("/metrics", get(mock_metrics))
            .route("/admin/fault", post(inject_fault))
            .with_state(Arc::clone(self))
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct MockRequest {
    #[serde(default = "default_model")]
    model: String,
    #[serde(default)]
    prompt: Option<Value>,
    #[serde(default)]
    messages: Option<Vec<Value>>,
    #[serde(default)]
    stream: bool,
    #[serde(default = "default_request_tokens")]
    max_tokens: u32,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(tag = "fault", rename_all = "snake_case", deny_unknown_fields)]
pub enum FaultCommand {
    Crash { enabled: bool },
    Slowdown { multiplier: f64 },
    ColdStart { next_delay_ms: u64 },
    MalformedSse { after_tokens: usize },
    Disconnect { after_tokens: usize },
    RequestErrors { count: u64 },
    Clear,
}

async fn completion(
    State(backend): State<Arc<MockBackend>>,
    Json(request): Json<MockRequest>,
) -> Response {
    generate(backend, request, false).await
}

async fn chat_completion(
    State(backend): State<Arc<MockBackend>>,
    Json(request): Json<MockRequest>,
) -> Response {
    generate(backend, request, true).await
}

#[allow(clippy::too_many_lines)]
async fn generate(backend: Arc<MockBackend>, request: MockRequest, chat: bool) -> Response {
    if backend.faults.crashed.load(Ordering::Relaxed) {
        return mock_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "backend_crashed",
            &backend.config.name,
        );
    }
    let pending_errors = backend.faults.request_error_count.load(Ordering::Relaxed);
    if pending_errors > 0
        && backend
            .faults
            .request_error_count
            .compare_exchange(
                pending_errors,
                pending_errors - 1,
                Ordering::Relaxed,
                Ordering::Relaxed,
            )
            .is_ok()
    {
        return mock_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "injected_request_error",
            &backend.config.name,
        );
    }
    let sequence = backend.sequence.fetch_add(1, Ordering::Relaxed);
    if deterministic_unit(backend.config.seed, sequence) < backend.config.failure_rate {
        return mock_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "sampled_failure",
            &backend.config.name,
        );
    }
    let Ok(permit) = Arc::clone(&backend.capacity).try_acquire_owned() else {
        return mock_error(
            StatusCode::TOO_MANY_REQUESTS,
            "capacity_exhausted",
            &backend.config.name,
        );
    };
    drop(backend.metrics.increment_counter(
        "sloforge_mock_requests_total",
        Labels::from([("backend".to_owned(), backend.config.name.clone())]),
        1,
    ));
    let prompt_tokens = prompt_tokens(&request);
    let output_tokens = request.max_tokens.min(backend.config.max_output_tokens);
    let slowdown = f64::from_bits(backend.faults.slowdown_bits.load(Ordering::Relaxed));
    let startup = startup_delay(&backend, sequence);
    let prefill_micros = backend
        .config
        .prefill_base_ms
        .saturating_mul(1_000)
        .saturating_add(
            backend
                .config
                .prefill_per_token_us
                .saturating_mul(prompt_tokens),
        );
    let malformed_after = backend
        .faults
        .malformed_after_tokens
        .swap(NO_TOKEN_FAULT, Ordering::Relaxed);
    let disconnect_after = backend
        .faults
        .disconnect_after_tokens
        .swap(NO_TOKEN_FAULT, Ordering::Relaxed);
    let backend_name = backend.config.name.clone();
    let model = request.model.clone();

    if !request.stream {
        tokio::time::sleep(scaled_duration(
            startup.saturating_add(prefill_micros),
            slowdown,
        ))
        .await;
        tokio::time::sleep(scaled_duration(
            backend
                .config
                .decode_per_token_ms
                .saturating_mul(1_000)
                .saturating_mul(u64::from(output_tokens)),
            slowdown,
        ))
        .await;
        let mut text = String::new();
        for index in 0..output_tokens {
            let _ = write!(text, "{backend_name}:{index} ");
        }
        drop(permit);
        return Json(json!({
            "id": format!("mock-{sequence}"),
            "object": if chat {"chat.completion"} else {"text_completion"},
            "model": model,
            "choices": [{"index": 0, "text": text, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]
        }))
        .into_response();
    }

    let output = async_stream::stream! {
        let _permit = permit;
        tokio::time::sleep(scaled_duration(startup.saturating_add(prefill_micros), slowdown)).await;
        for token in 0..output_tokens as usize {
            if token == disconnect_after {
                return;
            }
            if token == malformed_after {
                yield Result::<Bytes, Infallible>::Ok(Bytes::from_static(b"data: {malformed-json}\n\n"));
                return;
            }
            tokio::time::sleep(scaled_duration(
                backend.config.decode_per_token_ms.saturating_mul(1_000),
                slowdown,
            )).await;
            let text = format!("{backend_name}:{token} ");
            let chunk = if chat {
                json!({"id": format!("mock-{sequence}"), "object":"chat.completion.chunk", "model": model, "choices":[{"index":0,"delta":{"content":text},"finish_reason":null}]})
            } else {
                json!({"id": format!("mock-{sequence}"), "object":"text_completion", "model": model, "choices":[{"index":0,"text":text,"finish_reason":null}]})
            };
            match serde_json::to_string(&chunk) {
                Ok(encoded) => yield Result::<Bytes, Infallible>::Ok(Bytes::from(format!("data: {encoded}\n\n"))),
                Err(_) => return,
            }
        }
        yield Result::<Bytes, Infallible>::Ok(Bytes::from_static(b"data: [DONE]\n\n"));
    };
    let mut response = Response::new(Body::from_stream(output));
    response.headers_mut().insert(
        axum::http::header::CONTENT_TYPE,
        HeaderValue::from_static("text/event-stream"),
    );
    response
}

async fn mock_health(State(backend): State<Arc<MockBackend>>) -> impl IntoResponse {
    if backend.faults.crashed.load(Ordering::Relaxed) {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"status":"crashed", "backend": backend.config.name})),
        )
    } else {
        (
            StatusCode::OK,
            Json(json!({
                "status":"ready",
                "backend": backend.config.name,
                "capacity": backend.config.max_concurrency,
                "available": backend.capacity.available_permits(),
                "price_per_hour_usd": backend.config.price_per_hour_usd
            })),
        )
    }
}

async fn mock_metrics(State(backend): State<Arc<MockBackend>>) -> Response {
    match backend.metrics.render_prometheus() {
        Ok(metrics) => (StatusCode::OK, metrics).into_response(),
        Err(error) => mock_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            &error.to_string(),
            &backend.config.name,
        ),
    }
}

async fn inject_fault(
    State(backend): State<Arc<MockBackend>>,
    Json(command): Json<FaultCommand>,
) -> Response {
    let result = apply_fault(&backend.faults, command);
    match result {
        Ok(()) => Json(json!({"status":"applied", "backend":backend.config.name})).into_response(),
        Err(message) => mock_error(StatusCode::BAD_REQUEST, &message, &backend.config.name),
    }
}

fn apply_fault(faults: &FaultState, command: FaultCommand) -> Result<(), String> {
    match command {
        FaultCommand::Crash { enabled } => faults.crashed.store(enabled, Ordering::Relaxed),
        FaultCommand::Slowdown { multiplier } => {
            if !multiplier.is_finite() || !(0.01..=1_000.0).contains(&multiplier) {
                return Err("slowdown multiplier must be finite and in [0.01, 1000]".to_owned());
            }
            faults
                .slowdown_bits
                .store(multiplier.to_bits(), Ordering::Relaxed);
        }
        FaultCommand::ColdStart { next_delay_ms } => {
            faults
                .cold_start_next_ms
                .store(next_delay_ms, Ordering::Relaxed);
        }
        FaultCommand::MalformedSse { after_tokens } => {
            faults
                .malformed_after_tokens
                .store(after_tokens, Ordering::Relaxed);
        }
        FaultCommand::Disconnect { after_tokens } => {
            faults
                .disconnect_after_tokens
                .store(after_tokens, Ordering::Relaxed);
        }
        FaultCommand::RequestErrors { count } => {
            faults.request_error_count.store(count, Ordering::Relaxed);
        }
        FaultCommand::Clear => {
            faults.crashed.store(false, Ordering::Relaxed);
            faults
                .slowdown_bits
                .store(1.0_f64.to_bits(), Ordering::Relaxed);
            faults.cold_start_next_ms.store(0, Ordering::Relaxed);
            faults
                .malformed_after_tokens
                .store(NO_TOKEN_FAULT, Ordering::Relaxed);
            faults
                .disconnect_after_tokens
                .store(NO_TOKEN_FAULT, Ordering::Relaxed);
            faults.request_error_count.store(0, Ordering::Relaxed);
        }
    }
    Ok(())
}

fn startup_delay(backend: &MockBackend, sequence: u64) -> u64 {
    let injected = backend
        .faults
        .cold_start_next_ms
        .swap(0, Ordering::Relaxed)
        .saturating_mul(1_000);
    let scheduled = sequence == 0
        || (backend.config.startup_every_n_requests > 0
            && sequence % backend.config.startup_every_n_requests == 0);
    let base = if scheduled {
        backend.config.startup_ms.saturating_mul(1_000)
    } else {
        0
    };
    let jitter = if scheduled && backend.config.startup_jitter_ms > 0 {
        deterministic_mix(backend.config.seed ^ sequence)
            % backend.config.startup_jitter_ms.saturating_add(1)
    } else {
        0
    };
    injected
        .saturating_add(base)
        .saturating_add(jitter.saturating_mul(1_000))
}

fn prompt_tokens(request: &MockRequest) -> u64 {
    let prompt_chars = request
        .prompt
        .as_ref()
        .map_or(0, |prompt| prompt.to_string().chars().count());
    let message_chars = request.messages.as_ref().map_or(0, |messages| {
        messages
            .iter()
            .map(|message| message.to_string().chars().count())
            .sum()
    });
    u64::try_from((prompt_chars + message_chars).div_ceil(4)).unwrap_or(u64::MAX)
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn scaled_duration(micros: u64, multiplier: f64) -> Duration {
    let scaled = micros as f64 * multiplier;
    Duration::from_micros(if scaled >= u64::MAX as f64 {
        u64::MAX
    } else {
        scaled as u64
    })
}

#[allow(clippy::cast_precision_loss)]
fn deterministic_unit(seed: u64, sequence: u64) -> f64 {
    let mixed = deterministic_mix(seed ^ sequence.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    mixed as f64 / u64::MAX as f64
}

fn deterministic_mix(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn mock_error(status: StatusCode, kind: &str, backend: &str) -> Response {
    (
        status,
        Json(json!({"error":{"type":kind,"backend":backend}})),
    )
        .into_response()
}

fn default_bind() -> String {
    "127.0.0.1:9001".to_owned()
}
fn default_model() -> String {
    "sloforge/mock".to_owned()
}
const fn default_prefill_base_ms() -> u64 {
    5
}
const fn default_prefill_per_token_us() -> u64 {
    100
}
const fn default_decode_per_token_ms() -> u64 {
    5
}
const fn default_concurrency() -> usize {
    1
}
const fn default_max_output_tokens() -> u32 {
    1_024
}
const fn default_request_tokens() -> u32 {
    16
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> MockBackendConfig {
        MockBackendConfig {
            name: "test".to_owned(),
            bind: "127.0.0.1:0".to_owned(),
            seed: 7,
            startup_ms: 0,
            startup_jitter_ms: 0,
            startup_every_n_requests: 0,
            prefill_base_ms: 0,
            prefill_per_token_us: 0,
            decode_per_token_ms: 0,
            max_concurrency: 1,
            max_output_tokens: 10,
            price_per_hour_usd: 0.1,
            failure_rate: 0.0,
        }
    }

    #[test]
    fn deterministic_failures_are_reproducible() {
        let first = (0..100)
            .map(|sequence| deterministic_unit(9, sequence))
            .collect::<Vec<_>>();
        let second = (0..100)
            .map(|sequence| deterministic_unit(9, sequence))
            .collect::<Vec<_>>();
        assert_eq!(first, second);
    }

    #[test]
    fn validates_failure_rate() {
        let mut invalid = config();
        invalid.failure_rate = 1.1;
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn fault_state_clears() {
        let faults = FaultState::default();
        assert!(apply_fault(&faults, FaultCommand::Crash { enabled: true }).is_ok());
        assert!(faults.crashed.load(Ordering::Relaxed));
        assert!(apply_fault(&faults, FaultCommand::Clear).is_ok());
        assert!(!faults.crashed.load(Ordering::Relaxed));
    }
}
