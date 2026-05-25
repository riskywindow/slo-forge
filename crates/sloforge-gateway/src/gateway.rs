use crate::api::{ChatRequest, CompletionRequest, InferenceRequest};
use crate::backend::{Backend, BackendError, BackendEvent, open_stream};
use crate::config::GatewayConfig;
use crate::routing::{RouteCandidate, RoutingState};
use async_stream::stream;
use axum::body::{Body, Bytes};
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::{Stream, StreamExt};
use serde::Serialize;
use serde_json::{Value, json};
use sloforge_telemetry::{Labels, MetricsRegistry, TraceCollector, TraceEvent, TracePhase};
use std::collections::{BTreeMap, BTreeSet};
use std::convert::Infallible;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio::task::JoinHandle;
use tower_http::catch_panic::CatchPanicLayer;
use tower_http::trace::TraceLayer;
use tracing::{info, warn};
use uuid::Uuid;

type ExecutionStream = Pin<Box<dyn Stream<Item = ExecutionEvent> + Send>>;

#[derive(Debug)]
enum ExecutionEvent {
    Data(Value),
    Done,
    Error(PublicError),
}

#[derive(Clone, Debug, Serialize)]
struct PublicError {
    message: String,
    error_type: &'static str,
    request_id: String,
    retryable: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum GatewayBuildError {
    #[error("invalid gateway config: {0}")]
    Config(#[from] crate::config::ConfigError),
    #[error("failed to build HTTP client: {0}")]
    Http(#[from] reqwest::Error),
}

#[derive(Debug)]
struct HandlerError {
    status: StatusCode,
    public: PublicError,
}

impl IntoResponse for HandlerError {
    fn into_response(self) -> Response {
        (self.status, Json(json!({"error": self.public}))).into_response()
    }
}

#[derive(Clone, Debug)]
struct RequestContext {
    request_id: String,
    traceparent: String,
    started: Instant,
    trace_tid: u64,
}

pub struct Gateway {
    config: GatewayConfig,
    client: reqwest::Client,
    backends: Vec<Arc<Backend>>,
    routing: RoutingState,
    admission: Arc<Semaphore>,
    pub metrics: MetricsRegistry,
    pub traces: TraceCollector,
    origin: Instant,
}

impl Gateway {
    /// Build an immutable gateway and its bounded backend state.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid configuration or HTTP client initialization failure.
    pub fn new(mut config: GatewayConfig) -> Result<Arc<Self>, GatewayBuildError> {
        config.validate()?;
        config
            .provenance
            .entry("component".to_owned())
            .or_insert_with(|| "sloforge-gateway".to_owned());
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_millis(config.connect_timeout_ms))
            .pool_idle_timeout(Duration::from_secs(30))
            .tcp_nodelay(true)
            // Engine origins are compiler-selected. Never let an unhealthy or compromised
            // backend redirect requests (and their trace context) to a different origin.
            .redirect(reqwest::redirect::Policy::none())
            .build()?;
        let backends = config
            .backends
            .iter()
            .cloned()
            .map(|backend| {
                Arc::new(Backend::new(
                    backend,
                    config.breaker_failures,
                    Duration::from_millis(config.breaker_cooldown_ms),
                ))
            })
            .collect();
        Ok(Arc::new(Self {
            admission: Arc::new(Semaphore::new(config.admission_capacity)),
            traces: TraceCollector::with_capacity(
                config.provenance.clone(),
                config.max_trace_events,
            ),
            config,
            client,
            backends,
            routing: RoutingState::default(),
            metrics: MetricsRegistry::default(),
            origin: Instant::now(),
        }))
    }

    pub fn router(self: &Arc<Self>) -> Router {
        Router::new()
            .route("/v1/completions", post(completions))
            .route("/v1/chat/completions", post(chat_completions))
            .route("/health", get(health))
            .route("/ready", get(health))
            .route("/metrics", get(metrics))
            .layer(DefaultBodyLimit::max(self.config.max_request_bytes))
            .layer(CatchPanicLayer::new())
            .layer(TraceLayer::new_for_http())
            .with_state(Arc::clone(self))
    }

    #[must_use]
    pub fn spawn_health_checks(self: &Arc<Self>) -> HealthCheckTask {
        let gateway = Arc::clone(self);
        let task = tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(
                gateway.config.health_interval_ms.max(50),
            ));
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                interval.tick().await;
                let client = gateway.client.clone();
                let health_timeout = Duration::from_millis(gateway.config.health_timeout_ms);
                let checks = gateway.backends.iter().map(Arc::clone).map(|backend| {
                    let client = client.clone();
                    async move {
                        let url = format!(
                            "{}{}",
                            backend.config.base_url.trim_end_matches('/'),
                            backend.config.health_path
                        );
                        let healthy = tokio::time::timeout(health_timeout, client.get(url).send())
                            .await
                            .is_ok_and(|result| {
                                result.is_ok_and(|response| response.status().is_success())
                            });
                        (backend, healthy)
                    }
                });
                let results = futures::stream::iter(checks)
                    .buffer_unordered(16)
                    .collect::<Vec<_>>()
                    .await;
                for (backend, healthy) in results {
                    backend.set_healthy(healthy);
                    gateway.set_backend_gauges(&backend);
                }
            }
        });
        HealthCheckTask { task }
    }

    /// Persist the configured Chrome/Perfetto trace artifact, if configured.
    ///
    /// # Errors
    ///
    /// Returns an error when snapshotting or writing the trace fails.
    pub fn write_trace_artifact(&self) -> Result<(), sloforge_telemetry::TelemetryError> {
        if let Some(path) = &self.config.trace_output {
            self.traces.write_chrome_trace(path)?;
        }
        Ok(())
    }

    fn request_context(headers: &HeaderMap) -> RequestContext {
        let request_id = headers
            .get("x-request-id")
            .and_then(|value| value.to_str().ok())
            .filter(|value| !value.is_empty() && value.len() <= 128)
            .map_or_else(|| Uuid::now_v7().to_string(), ToOwned::to_owned);
        let traceparent = headers
            .get("traceparent")
            .and_then(|value| value.to_str().ok())
            .filter(|value| valid_traceparent(value))
            .map_or_else(new_traceparent, ToOwned::to_owned);
        RequestContext {
            trace_tid: stable_hash(&request_id),
            request_id,
            traceparent,
            started: Instant::now(),
        }
    }

    fn admit(&self, context: &RequestContext) -> Result<OwnedSemaphorePermit, HandlerError> {
        Arc::clone(&self.admission)
            .try_acquire_owned()
            .map_err(|_| {
                self.counter(
                    "sloforge_gateway_rejected_requests_total",
                    Labels::from([("reason".to_owned(), "admission_capacity".to_owned())]),
                );
                HandlerError {
                    status: StatusCode::TOO_MANY_REQUESTS,
                    public: PublicError {
                        message: "gateway admission capacity is exhausted".to_owned(),
                        error_type: "queue_saturated",
                        request_id: context.request_id.clone(),
                        retryable: true,
                    },
                }
            })
    }

    #[allow(clippy::too_many_lines)]
    fn execution_stream(
        self: &Arc<Self>,
        request: InferenceRequest,
        context: RequestContext,
        admission: OwnedSemaphorePermit,
    ) -> ExecutionStream {
        let gateway = Arc::clone(self);
        Box::pin(stream! {
            let _admission = admission;
            info!(
                request_id = %context.request_id,
                traceparent = %context.traceparent,
                model = %request.model,
                max_tokens = request.max_tokens,
                "request admitted"
            );
                gateway.counter("sloforge_gateway_requests_total", Labels::new());
                let mut excluded = BTreeSet::new();
                let mut output_started = false;
                let mut attempt = 0_usize;
                let max_attempts = gateway.config.retry_attempts.saturating_add(1).min(gateway.backends.len());
                while attempt < max_attempts {
                    let Some(index) = gateway.select_backend(&request, &excluded, context.started) else {
                        let public = PublicError {
                            message: "no healthy backend is currently eligible".to_owned(),
                            error_type: "backend_unavailable",
                            request_id: context.request_id.clone(),
                            retryable: true,
                        };
                        gateway.record_terminal_error(&context, &public, output_started);
                        yield ExecutionEvent::Error(public);
                        return;
                    };
                    let backend = Arc::clone(&gateway.backends[index]);
                    excluded.insert(index);
                    attempt = attempt.saturating_add(1);
                    gateway.counter(
                        "sloforge_gateway_route_decisions_total",
                        Labels::from([
                            ("backend".to_owned(), backend.config.name.clone()),
                            ("policy".to_owned(), gateway.config.routing_policy.as_str().to_owned()),
                        ]),
                    );
                    gateway.record_route_trace(&context, &backend.config.name);
                    info!(backend = %backend.config.name, attempt, "selected backend");

                    let lease = match backend.acquire(Duration::from_millis(gateway.config.queue_timeout_ms)).await {
                        Ok(lease) => lease,
                        Err(error) => {
                            gateway.backend_error(&backend, &error);
                            if attempt < max_attempts {
                                gateway.counter("sloforge_gateway_retries_total", Labels::from([("phase".to_owned(), "queue".to_owned())]));
                                continue;
                            }
                            let public = Gateway::public_backend_error(&context, &error);
                            gateway.record_terminal_error(&context, &public, false);
                            yield ExecutionEvent::Error(public);
                            return;
                        }
                    };
                    gateway.set_backend_gauges(&backend);
                    let remaining = remaining_timeout(context.started, gateway.config.request_timeout_ms);
                    let opened = tokio::time::timeout(
                        remaining,
                        open_stream(
                            &gateway.client,
                            &backend,
                            request.endpoint,
                            request.payload.clone(),
                            gateway.config.stream_buffer_bytes,
                            &context.request_id,
                            &context.traceparent,
                        ),
                    )
                    .await;
                    let mut backend_stream = match opened {
                        Ok(Ok(stream)) => stream,
                        Ok(Err(error)) => {
                            backend.record_failure();
                            drop(lease);
                            gateway.backend_error(&backend, &error);
                            if attempt < max_attempts {
                                gateway.counter("sloforge_gateway_retries_total", Labels::from([("phase".to_owned(), "before_output".to_owned())]));
                                continue;
                            }
                            let public = Gateway::public_backend_error(&context, &error);
                            gateway.record_terminal_error(&context, &public, false);
                            yield ExecutionEvent::Error(public);
                            return;
                        }
                        Err(_) => {
                            let error = BackendError::RequestTimeout;
                            backend.record_failure();
                            drop(lease);
                            gateway.backend_error(&backend, &error);
                            if attempt < max_attempts {
                                gateway.counter("sloforge_gateway_retries_total", Labels::from([("phase".to_owned(), "before_output".to_owned())]));
                                continue;
                            }
                            let public = Gateway::public_backend_error(&context, &error);
                            gateway.record_terminal_error(&context, &public, false);
                            yield ExecutionEvent::Error(public);
                            return;
                        }
                    };

                    loop {
                        let remaining = remaining_timeout(context.started, gateway.config.request_timeout_ms);
                        let next = tokio::time::timeout(remaining, backend_stream.next()).await;
                        match next {
                            Ok(Some(Ok(BackendEvent::Data(value)))) => {
                                if !output_started {
                                    output_started = true;
                                    gateway.observe_ttft(&context, &backend.config.name);
                                }
                                gateway.counter(
                                    "sloforge_gateway_stream_events_total",
                                    Labels::from([("backend".to_owned(), backend.config.name.clone())]),
                                );
                                yield ExecutionEvent::Data(value);
                            }
                            Ok(Some(Ok(BackendEvent::Done))) => {
                                let elapsed = lease.elapsed();
                                backend.record_success(elapsed);
                                drop(lease);
                                gateway.set_backend_gauges(&backend);
                                gateway.observe_completion(&context, &backend.config.name);
                                yield ExecutionEvent::Done;
                                return;
                            }
                            Ok(Some(Err(error))) => {
                                backend.record_failure();
                                drop(lease);
                                gateway.backend_error(&backend, &error);
                                if !output_started && attempt < max_attempts {
                                    gateway.counter("sloforge_gateway_retries_total", Labels::from([("phase".to_owned(), "before_output".to_owned())]));
                                    break;
                                }
                                let public = Gateway::public_backend_error(&context, &error);
                                gateway.record_terminal_error(&context, &public, output_started);
                                yield ExecutionEvent::Error(public);
                                return;
                            }
                            Ok(None) => {
                                let error = BackendError::Stream("backend stream ended without a terminal event".to_owned());
                                backend.record_failure();
                                drop(lease);
                                gateway.backend_error(&backend, &error);
                                if !output_started && attempt < max_attempts {
                                    gateway.counter("sloforge_gateway_retries_total", Labels::from([("phase".to_owned(), "before_output".to_owned())]));
                                    break;
                                }
                                let public = Gateway::public_backend_error(&context, &error);
                                gateway.record_terminal_error(&context, &public, output_started);
                                yield ExecutionEvent::Error(public);
                                return;
                            }
                            Err(_) => {
                                let error = BackendError::RequestTimeout;
                                backend.record_failure();
                                drop(lease);
                                gateway.backend_error(&backend, &error);
                                if !output_started && attempt < max_attempts {
                                    gateway.counter("sloforge_gateway_retries_total", Labels::from([("phase".to_owned(), "before_output".to_owned())]));
                                    break;
                                }
                                let public = Gateway::public_backend_error(&context, &error);
                                gateway.record_terminal_error(&context, &public, output_started);
                                yield ExecutionEvent::Error(public);
                                return;
                            }
                        }
                    }
                }
        })
    }

    fn select_backend(
        &self,
        request: &InferenceRequest,
        excluded: &BTreeSet<usize>,
        started: Instant,
    ) -> Option<usize> {
        let now = Instant::now();
        let candidates = self
            .backends
            .iter()
            .enumerate()
            .map(|(index, backend)| RouteCandidate {
                index,
                name: backend.config.name.clone(),
                outstanding: backend.outstanding(),
                capacity: backend.config.capacity,
                estimated_service_ms: backend.estimated_service_ms(),
                price_per_hour_usd: backend.config.price_per_hour_usd,
                weight: backend.config.weight,
                available: !excluded.contains(&index) && backend.is_available(now),
            })
            .collect::<Vec<_>>();
        let deadline_slack = request
            .slo
            .as_ref()
            .and_then(|slo| slo.deadline_ms)
            .map(|deadline| {
                Duration::from_millis(deadline).as_secs_f64() * 1_000.0
                    - started.elapsed().as_secs_f64() * 1_000.0
            });
        self.routing.select(
            self.config.routing_policy,
            &candidates,
            deadline_slack,
            request
                .slo
                .as_ref()
                .and_then(|slo| slo.priority)
                .unwrap_or(0),
        )
    }

    fn counter(&self, name: &str, labels: Labels) {
        drop(self.metrics.increment_counter(name, labels, 1));
    }

    fn set_backend_gauges(&self, backend: &Backend) {
        let labels = Labels::from([("backend".to_owned(), backend.config.name.clone())]);
        drop(self.metrics.set_gauge(
            "sloforge_gateway_backend_outstanding",
            labels.clone(),
            f64::from(u32::try_from(backend.outstanding()).unwrap_or(u32::MAX)),
        ));
        drop(self.metrics.set_gauge(
            "sloforge_gateway_backend_healthy",
            labels.clone(),
            f64::from(backend.is_healthy()),
        ));
        drop(self.metrics.set_gauge(
            "sloforge_gateway_circuit_open",
            labels,
            f64::from(backend.breaker_is_open(Instant::now())),
        ));
    }

    fn backend_error(&self, backend: &Backend, error: &BackendError) {
        warn!(
            backend = %backend.config.name,
            kind = backend_error_kind(error),
            status = backend_status(error),
            "backend attempt failed"
        );
        self.counter(
            "sloforge_gateway_backend_errors_total",
            Labels::from([
                ("backend".to_owned(), backend.config.name.clone()),
                ("kind".to_owned(), backend_error_kind(error).to_owned()),
            ]),
        );
        self.set_backend_gauges(backend);
    }

    fn public_backend_error(context: &RequestContext, error: &BackendError) -> PublicError {
        PublicError {
            message: public_backend_message(error),
            error_type: backend_error_kind(error),
            request_id: context.request_id.clone(),
            retryable: true,
        }
    }

    fn observe_ttft(&self, context: &RequestContext, backend: &str) {
        drop(self.metrics.observe_histogram(
            "sloforge_gateway_ttft_seconds",
            Labels::from([("backend".to_owned(), backend.to_owned())]),
            context.started.elapsed().as_secs_f64(),
            &[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        ));
    }

    fn observe_completion(&self, context: &RequestContext, backend: &str) {
        let elapsed = context.started.elapsed();
        drop(self.metrics.observe_histogram(
            "sloforge_gateway_request_duration_seconds",
            Labels::from([("backend".to_owned(), backend.to_owned())]),
            elapsed.as_secs_f64(),
            &[
                0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 120.0,
            ],
        ));
        self.record_trace(
            &context.request_id,
            context.trace_tid,
            "request",
            context.started,
            elapsed,
            BTreeMap::from([
                ("backend".to_owned(), Value::String(backend.to_owned())),
                ("outcome".to_owned(), Value::String("success".to_owned())),
            ]),
        );
    }

    fn record_terminal_error(
        &self,
        context: &RequestContext,
        error: &PublicError,
        output_started: bool,
    ) {
        self.counter(
            "sloforge_gateway_errors_total",
            Labels::from([
                ("kind".to_owned(), error.error_type.to_owned()),
                ("output_started".to_owned(), output_started.to_string()),
            ]),
        );
        self.record_trace(
            &context.request_id,
            context.trace_tid,
            "request",
            context.started,
            context.started.elapsed(),
            BTreeMap::from([
                ("outcome".to_owned(), Value::String("error".to_owned())),
                (
                    "error_type".to_owned(),
                    Value::String(error.error_type.to_owned()),
                ),
                ("output_started".to_owned(), Value::Bool(output_started)),
            ]),
        );
    }

    fn record_route_trace(&self, context: &RequestContext, backend: &str) {
        let now = Instant::now();
        self.record_trace(
            &context.request_id,
            context.trace_tid,
            "routing",
            now,
            Duration::ZERO,
            BTreeMap::from([
                ("backend".to_owned(), Value::String(backend.to_owned())),
                (
                    "policy".to_owned(),
                    Value::String(self.config.routing_policy.as_str().to_owned()),
                ),
            ]),
        );
    }

    fn record_trace(
        &self,
        request_id: &str,
        tid: u64,
        name: &str,
        started: Instant,
        duration: Duration,
        mut args: BTreeMap<String, Value>,
    ) {
        args.insert(
            "request_id".to_owned(),
            Value::String(request_id.to_owned()),
        );
        let timestamp = started
            .checked_duration_since(self.origin)
            .unwrap_or(Duration::ZERO);
        drop(self.traces.record(TraceEvent {
            name: format!("gateway.{name}"),
            cat: "inference".to_owned(),
            ph: TracePhase::Complete,
            ts: duration_micros(timestamp),
            pid: std::process::id(),
            tid,
            dur: Some(duration_micros(duration)),
            args,
        }));
    }
}

pub struct HealthCheckTask {
    task: JoinHandle<()>,
}

impl Drop for HealthCheckTask {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn completions(
    State(gateway): State<Arc<Gateway>>,
    headers: HeaderMap,
    payload: Result<Json<CompletionRequest>, axum::extract::rejection::JsonRejection>,
) -> Result<Response, HandlerError> {
    let context = Gateway::request_context(&headers);
    let request = payload
        .map_err(|error| invalid_request(&context, error.body_text()))?
        .0;
    let inference = InferenceRequest::try_from(request)
        .map_err(|error| invalid_request(&context, error.to_string()))?;
    handle_inference(gateway, inference, context).await
}

async fn chat_completions(
    State(gateway): State<Arc<Gateway>>,
    headers: HeaderMap,
    payload: Result<Json<ChatRequest>, axum::extract::rejection::JsonRejection>,
) -> Result<Response, HandlerError> {
    let context = Gateway::request_context(&headers);
    let request = payload
        .map_err(|error| invalid_request(&context, error.body_text()))?
        .0;
    let inference = InferenceRequest::try_from(request)
        .map_err(|error| invalid_request(&context, error.to_string()))?;
    handle_inference(gateway, inference, context).await
}

async fn handle_inference(
    gateway: Arc<Gateway>,
    request: InferenceRequest,
    context: RequestContext,
) -> Result<Response, HandlerError> {
    let admission = gateway.admit(&context)?;
    let request_id = context.request_id.clone();
    let traceparent = context.traceparent.clone();
    let stream_requested = request.stream;
    let model = request.model.clone();
    let chat_response = request.endpoint == "/v1/chat/completions";
    let execution = gateway.execution_stream(request, context, admission);
    let mut response = if stream_requested {
        streaming_response(execution)
    } else {
        buffered_response(
            execution,
            &model,
            &request_id,
            chat_response,
            gateway.config.stream_buffer_bytes,
        )
        .await?
    };
    insert_header(response.headers_mut(), "x-request-id", &request_id)?;
    insert_header(response.headers_mut(), "traceparent", &traceparent)?;
    Ok(response)
}

fn streaming_response(mut execution: ExecutionStream) -> Response {
    let output = stream! {
        while let Some(event) = execution.next().await {
            let data = match event {
                ExecutionEvent::Data(value) => serde_json::to_string(&value).unwrap_or_else(|_| "{\"error\":{\"message\":\"serialization failure\"}}".to_owned()),
                ExecutionEvent::Done => "[DONE]".to_owned(),
                ExecutionEvent::Error(error) => serde_json::to_string(&json!({"error": error})).unwrap_or_else(|_| "{\"error\":{\"message\":\"serialization failure\"}}".to_owned()),
            };
            yield Result::<Bytes, Infallible>::Ok(Bytes::from(format!("data: {data}\n\n")));
        }
    };
    let mut response = Response::new(Body::from_stream(output));
    response.headers_mut().insert(
        axum::http::header::CONTENT_TYPE,
        HeaderValue::from_static("text/event-stream"),
    );
    response.headers_mut().insert(
        axum::http::header::CACHE_CONTROL,
        HeaderValue::from_static("no-cache, no-transform"),
    );
    response.headers_mut().insert(
        axum::http::header::CONNECTION,
        HeaderValue::from_static("keep-alive"),
    );
    response
}

async fn buffered_response(
    mut execution: ExecutionStream,
    model: &str,
    request_id: &str,
    chat_response: bool,
    max_response_bytes: usize,
) -> Result<Response, HandlerError> {
    let mut text = String::new();
    let mut last_value = None;
    while let Some(event) = execution.next().await {
        match event {
            ExecutionEvent::Data(value) => {
                if !append_text(&mut text, &value, max_response_bytes) {
                    return Err(HandlerError {
                        status: StatusCode::BAD_GATEWAY,
                        public: PublicError {
                            message: format!(
                                "buffered backend response exceeded {max_response_bytes} bytes"
                            ),
                            error_type: "backend_response_too_large",
                            request_id: request_id.to_owned(),
                            retryable: false,
                        },
                    });
                }
                last_value = Some(value);
            }
            ExecutionEvent::Done => {
                let response = if chat_response {
                    json!({
                        "id": request_id,
                        "object": "chat.completion",
                        "model": model,
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                        "sloforge": {"last_backend_event": last_value}
                    })
                } else {
                    json!({
                        "id": request_id,
                        "object": "text_completion",
                        "model": model,
                        "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
                        "sloforge": {"last_backend_event": last_value}
                    })
                };
                return Ok(Json(response).into_response());
            }
            ExecutionEvent::Error(error) => {
                return Err(HandlerError {
                    status: StatusCode::BAD_GATEWAY,
                    public: error,
                });
            }
        }
    }
    Err(HandlerError {
        status: StatusCode::BAD_GATEWAY,
        public: PublicError {
            message: "execution ended without a terminal event".to_owned(),
            error_type: "backend_stream",
            request_id: request_id.to_owned(),
            retryable: true,
        },
    })
}

fn append_text(output: &mut String, value: &Value, max_bytes: usize) -> bool {
    let addition = if let Some(text) = value.pointer("/choices/0/text").and_then(Value::as_str) {
        Some(text)
    } else {
        value
            .pointer("/choices/0/delta/content")
            .and_then(Value::as_str)
    };
    let Some(addition) = addition else {
        return true;
    };
    if output.len().saturating_add(addition.len()) > max_bytes {
        false
    } else {
        output.push_str(addition);
        true
    }
}

async fn health(State(gateway): State<Arc<Gateway>>) -> impl IntoResponse {
    let now = Instant::now();
    let backends = gateway
        .backends
        .iter()
        .map(|backend| {
            json!({
                "name": backend.config.name,
                "healthy": backend.is_healthy(),
                "circuit_open": backend.breaker_is_open(now),
                "outstanding": backend.outstanding(),
                "capacity": backend.config.capacity,
                "estimated_service_ms": backend.estimated_service_ms(),
            })
        })
        .collect::<Vec<_>>();
    let ready = gateway
        .backends
        .iter()
        .any(|backend| backend.is_available(now));
    let status = if ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (
        status,
        Json(json!({"status": if ready {"ready"} else {"unavailable"}, "backends": backends})),
    )
}

async fn metrics(State(gateway): State<Arc<Gateway>>) -> Response {
    match gateway.metrics.render_prometheus() {
        Ok(rendered) => (
            StatusCode::OK,
            [(
                axum::http::header::CONTENT_TYPE,
                "text/plain; version=0.0.4",
            )],
            rendered,
        )
            .into_response(),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": {"message": error.to_string(), "type": "telemetry"}})),
        )
            .into_response(),
    }
}

fn invalid_request(context: &RequestContext, message: String) -> HandlerError {
    HandlerError {
        status: StatusCode::BAD_REQUEST,
        public: PublicError {
            message,
            error_type: "invalid_request",
            request_id: context.request_id.clone(),
            retryable: false,
        },
    }
}

fn insert_header(
    headers: &mut HeaderMap,
    name: &'static str,
    value: &str,
) -> Result<(), HandlerError> {
    let value = HeaderValue::from_str(value).map_err(|_| HandlerError {
        status: StatusCode::INTERNAL_SERVER_ERROR,
        public: PublicError {
            message: "generated response metadata was invalid".to_owned(),
            error_type: "internal",
            request_id: "unknown".to_owned(),
            retryable: false,
        },
    })?;
    headers.insert(name, value);
    Ok(())
}

fn remaining_timeout(started: Instant, total_ms: u64) -> Duration {
    Duration::from_millis(total_ms)
        .checked_sub(started.elapsed())
        .unwrap_or(Duration::from_millis(1))
        .max(Duration::from_millis(1))
}

fn backend_error_kind(error: &BackendError) -> &'static str {
    match error {
        BackendError::Transport(_) => "backend_transport",
        BackendError::Status { .. } => "backend_status",
        BackendError::Stream(_) => "backend_stream",
        BackendError::QueueTimeout => "backend_queue_timeout",
        BackendError::CircuitOpen => "backend_circuit_open",
        BackendError::RequestTimeout => "backend_request_timeout",
    }
}

fn backend_status(error: &BackendError) -> Option<u16> {
    match error {
        BackendError::Status { status, .. } => Some(*status),
        _ => None,
    }
}

fn public_backend_message(error: &BackendError) -> String {
    match error {
        BackendError::Status { status, .. } => format!("backend returned HTTP {status}"),
        BackendError::Transport(_) => "backend transport failed".to_owned(),
        BackendError::Stream(_) => "backend stream was invalid or incomplete".to_owned(),
        BackendError::QueueTimeout => "backend queue timed out".to_owned(),
        BackendError::CircuitOpen => "backend circuit breaker is open".to_owned(),
        BackendError::RequestTimeout => "backend request timed out".to_owned(),
    }
}

fn valid_traceparent(value: &str) -> bool {
    let parts = value.split('-').collect::<Vec<_>>();
    parts.len() == 4
        && parts[0].len() == 2
        && parts[1].len() == 32
        && parts[2].len() == 16
        && parts[3].len() == 2
        && parts
            .iter()
            .all(|part| part.chars().all(|character| character.is_ascii_hexdigit()))
}

fn new_traceparent() -> String {
    let trace_id = Uuid::new_v4().simple().to_string();
    let span_source = Uuid::new_v4().simple().to_string();
    format!("00-{trace_id}-{}-01", &span_source[..16])
}

fn stable_hash(input: &str) -> u64 {
    input.bytes().fold(1_469_598_103_934_665_603, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(1_099_511_628_211)
    })
}

fn duration_micros(duration: Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
}
