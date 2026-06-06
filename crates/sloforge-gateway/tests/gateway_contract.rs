use axum::body::Body;
use http_body_util::BodyExt;
use serde_json::{Value, json};
use sloforge_gateway::Gateway;
use sloforge_gateway::config::{BackendConfig, GatewayConfig};
use sloforge_gateway::mock::{FaultCommand, MockBackend, MockBackendConfig};
use sloforge_gateway::routing::RoutingPolicy;
use std::collections::BTreeMap;
use std::error::Error;
use std::time::Duration;
use tokio::task::JoinHandle;
use tower::ServiceExt;

struct RunningBackend {
    url: String,
    task: JoinHandle<()>,
}

impl Drop for RunningBackend {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn start_backend(name: &str, decode_ms: u64) -> Result<RunningBackend, Box<dyn Error>> {
    let backend = MockBackend::new(MockBackendConfig {
        name: name.to_owned(),
        bind: "127.0.0.1:0".to_owned(),
        seed: 42,
        startup_ms: 0,
        startup_jitter_ms: 0,
        startup_every_n_requests: 0,
        prefill_base_ms: 1,
        prefill_per_token_us: 10,
        decode_per_token_ms: decode_ms,
        max_concurrency: 1,
        max_output_tokens: 32,
        price_per_hour_usd: 0.25,
        failure_rate: 0.0,
        fault_api_enabled: true,
    })?;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    let router = backend.router();
    let task = tokio::spawn(async move {
        drop(axum::serve(listener, router).await);
    });
    Ok(RunningBackend {
        url: format!("http://{address}"),
        task,
    })
}

async fn start_redirect_backend(target: &str) -> Result<RunningBackend, Box<dyn Error>> {
    let destination = format!("{target}/v1/completions");
    let router = axum::Router::new()
        .route("/health", axum::routing::get(|| async { "ok" }))
        .route(
            "/v1/completions",
            axum::routing::post(move || {
                let destination = destination.clone();
                async move { axum::response::Redirect::temporary(&destination) }
            }),
        );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    let task = tokio::spawn(async move {
        drop(axum::serve(listener, router).await);
    });
    Ok(RunningBackend {
        url: format!("http://{address}"),
        task,
    })
}

fn gateway_config(urls: &[(&str, &str)], admission_capacity: usize) -> GatewayConfig {
    GatewayConfig {
        bind: "127.0.0.1:0".to_owned(),
        backends: urls
            .iter()
            .map(|(name, url)| BackendConfig {
                name: (*name).to_owned(),
                base_url: (*url).to_owned(),
                capacity: 1,
                estimated_service_ms: 10.0,
                price_per_hour_usd: 0.1,
                health_path: "/health".to_owned(),
                weight: 1,
            })
            .collect(),
        routing_policy: RoutingPolicy::RoundRobin,
        admission_capacity,
        stream_buffer_bytes: 16 * 1_024,
        max_request_bytes: 16 * 1_024,
        max_output_tokens: 1_024,
        queue_timeout_ms: 100,
        request_timeout_ms: 3_000,
        connect_timeout_ms: 200,
        health_interval_ms: 10_000,
        health_timeout_ms: 100,
        retry_attempts: 1,
        breaker_failures: 2,
        breaker_cooldown_ms: 100,
        trace_output: None,
        max_trace_events: 10_000,
        provenance: BTreeMap::from([("test".to_owned(), "gateway_contract".to_owned())]),
    }
}

fn completion_request(
    stream: bool,
    max_tokens: u32,
) -> Result<axum::http::Request<Body>, http::Error> {
    axum::http::Request::builder()
        .method("POST")
        .uri("/v1/completions")
        .header("content-type", "application/json")
        .header("x-request-id", "contract-request")
        .header(
            "traceparent",
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
        )
        .body(Body::from(
            json!({
                "model": "sloforge/mock",
                "prompt": "hello deterministic world",
                "stream": stream,
                "max_tokens": max_tokens,
                "sloforge": {"deadline_ms": 500, "priority": 1}
            })
            .to_string(),
        ))
}

fn chat_request(stream: bool) -> Result<axum::http::Request<Body>, http::Error> {
    axum::http::Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({
                "model": "sloforge/mock",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": stream,
                "max_tokens": 2
            })
            .to_string(),
        ))
}

async fn response_text(response: axum::response::Response) -> Result<String, Box<dyn Error>> {
    let bytes = response.into_body().collect().await?.to_bytes();
    Ok(String::from_utf8(bytes.to_vec())?)
}

async fn fault(backend: &RunningBackend, command: &FaultCommand) -> Result<(), Box<dyn Error>> {
    let response = reqwest::Client::new()
        .post(format!("{}/admin/fault", backend.url))
        .json(command)
        .send()
        .await?;
    if response.status().is_success() {
        Ok(())
    } else {
        Err(format!("fault injection returned {}", response.status()).into())
    }
}

#[tokio::test]
async fn streaming_contract_preserves_context_and_emits_metrics_and_trace()
-> Result<(), Box<dyn Error>> {
    let backend = start_backend("alpha", 1).await?;
    let gateway = Gateway::new(gateway_config(&[("alpha", &backend.url)], 4))?;
    let response = gateway
        .router()
        .oneshot(completion_request(true, 3)?)
        .await?;
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get("x-request-id")
            .and_then(|value| value.to_str().ok()),
        Some("contract-request")
    );
    assert_eq!(
        response
            .headers()
            .get("traceparent")
            .and_then(|value| value.to_str().ok()),
        Some("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01")
    );
    let body = response_text(response).await?;
    assert_eq!(body.matches("alpha:").count(), 3);
    assert!(body.ends_with("data: [DONE]\n\n"));

    let metrics_response = gateway
        .router()
        .oneshot(
            axum::http::Request::builder()
                .uri("/metrics")
                .body(Body::empty())?,
        )
        .await?;
    let metrics = response_text(metrics_response).await?;
    assert!(metrics.contains("sloforge_gateway_ttft_seconds_count{backend=\"alpha\"} 1"));
    assert!(
        metrics.contains("sloforge_gateway_request_duration_seconds_count{backend=\"alpha\"} 1")
    );
    let trace = gateway.traces.snapshot()?;
    assert!(
        trace
            .trace_events
            .iter()
            .any(|event| event.name == "gateway.routing")
    );
    assert!(
        trace
            .trace_events
            .iter()
            .any(|event| event.name == "gateway.request")
    );
    Ok(())
}

#[tokio::test]
async fn retries_only_before_output_begins() -> Result<(), Box<dyn Error>> {
    let failing = start_backend("failing", 1).await?;
    let fallback = start_backend("fallback", 1).await?;
    fault(&failing, &FaultCommand::RequestErrors { count: 1 }).await?;
    let gateway = Gateway::new(gateway_config(
        &[("failing", &failing.url), ("fallback", &fallback.url)],
        4,
    ))?;
    let response = gateway
        .router()
        .oneshot(completion_request(true, 2)?)
        .await?;
    let body = response_text(response).await?;
    assert!(!body.contains("failing:"));
    assert_eq!(body.matches("fallback:").count(), 2);
    let metrics = gateway.metrics.render_prometheus()?;
    assert!(metrics.contains("sloforge_gateway_retries_total{phase=\"before_output\"} 1"));
    Ok(())
}

#[tokio::test]
async fn malformed_sse_after_partial_output_is_not_retried() -> Result<(), Box<dyn Error>> {
    let malformed = start_backend("malformed", 1).await?;
    let fallback = start_backend("must-not-run", 1).await?;
    fault(&malformed, &FaultCommand::MalformedSse { after_tokens: 1 }).await?;
    let gateway = Gateway::new(gateway_config(
        &[
            ("malformed", &malformed.url),
            ("must-not-run", &fallback.url),
        ],
        4,
    ))?;
    let response = gateway
        .router()
        .oneshot(completion_request(true, 3)?)
        .await?;
    let body = response_text(response).await?;
    assert_eq!(body.matches("malformed:").count(), 1);
    assert!(!body.contains("must-not-run:"));
    assert!(body.contains("backend_stream"));
    let metrics = gateway.metrics.render_prometheus()?;
    assert!(!metrics.contains("sloforge_gateway_retries_total"));
    assert!(metrics.contains("output_started=\"true\""));
    Ok(())
}

#[tokio::test]
async fn disconnect_before_output_retries_but_disconnect_after_output_does_not()
-> Result<(), Box<dyn Error>> {
    let disconnect = start_backend("disconnect", 1).await?;
    let fallback = start_backend("fallback", 1).await?;
    fault(&disconnect, &FaultCommand::Disconnect { after_tokens: 0 }).await?;
    let gateway = Gateway::new(gateway_config(
        &[("disconnect", &disconnect.url), ("fallback", &fallback.url)],
        4,
    ))?;
    let first = response_text(
        gateway
            .router()
            .oneshot(completion_request(true, 2)?)
            .await?,
    )
    .await?;
    assert_eq!(first.matches("fallback:").count(), 2);

    let partial = start_backend("partial", 1).await?;
    let forbidden = start_backend("forbidden", 1).await?;
    fault(&partial, &FaultCommand::Disconnect { after_tokens: 1 }).await?;
    let gateway = Gateway::new(gateway_config(
        &[("partial", &partial.url), ("forbidden", &forbidden.url)],
        4,
    ))?;
    let second = response_text(
        gateway
            .router()
            .oneshot(completion_request(true, 3)?)
            .await?,
    )
    .await?;
    assert_eq!(second.matches("partial:").count(), 1);
    assert!(!second.contains("forbidden:"));
    assert!(second.contains("backend_stream"));
    Ok(())
}

#[tokio::test]
async fn admission_queue_saturates_and_releases_on_client_drop() -> Result<(), Box<dyn Error>> {
    let backend = start_backend("slow", 20).await?;
    let gateway = Gateway::new(gateway_config(&[("slow", &backend.url)], 1))?;
    let first = gateway
        .router()
        .oneshot(completion_request(true, 10)?)
        .await?;
    let saturated = gateway
        .router()
        .oneshot(completion_request(true, 1)?)
        .await?;
    assert_eq!(
        saturated.status(),
        axum::http::StatusCode::TOO_MANY_REQUESTS
    );
    drop(first);
    let recovered = gateway
        .router()
        .oneshot(completion_request(false, 1)?)
        .await?;
    assert_eq!(recovered.status(), axum::http::StatusCode::OK);
    let body: Value = serde_json::from_str(&response_text(recovered).await?)?;
    assert!(
        body.pointer("/choices/0/text")
            .and_then(Value::as_str)
            .is_some()
    );
    Ok(())
}

#[tokio::test]
async fn dropping_partially_consumed_stream_cancels_backend_work() -> Result<(), Box<dyn Error>> {
    let backend = start_backend("cancel", 10).await?;
    let gateway = Gateway::new(gateway_config(&[("cancel", &backend.url)], 1))?;
    let response = gateway
        .router()
        .oneshot(completion_request(true, 20)?)
        .await?;
    let mut body = response.into_body();
    let first_frame = tokio::time::timeout(Duration::from_secs(1), body.frame()).await?;
    assert!(first_frame.is_some());
    drop(body);
    tokio::time::sleep(Duration::from_millis(30)).await;

    let next = gateway
        .router()
        .oneshot(completion_request(false, 1)?)
        .await?;
    assert_eq!(next.status(), axum::http::StatusCode::OK);
    assert!(response_text(next).await?.contains("cancel:0"));
    Ok(())
}

#[tokio::test]
async fn slow_consumer_receives_ordered_bounded_stream() -> Result<(), Box<dyn Error>> {
    let backend = start_backend("paced", 1).await?;
    let gateway = Gateway::new(gateway_config(&[("paced", &backend.url)], 2))?;
    let response = gateway
        .router()
        .oneshot(completion_request(true, 5)?)
        .await?;
    let mut body = response.into_body();
    let mut bytes = Vec::new();
    while let Some(frame) = body.frame().await {
        let frame = frame?;
        if let Some(data) = frame.data_ref() {
            bytes.extend_from_slice(data);
        }
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    let text = String::from_utf8(bytes)?;
    for token in 0..5 {
        assert!(text.contains(&format!("paced:{token}")));
    }
    assert!(text.ends_with("data: [DONE]\n\n"));
    Ok(())
}

#[tokio::test]
async fn buffered_completion_has_a_total_response_bound() -> Result<(), Box<dyn Error>> {
    let backend = start_backend("backend-name-with-enough-width", 1).await?;
    let mut config = gateway_config(&[("wide", &backend.url)], 2);
    config.stream_buffer_bytes = 256;
    let gateway = Gateway::new(config)?;
    let response = gateway
        .router()
        .oneshot(completion_request(false, 12)?)
        .await?;
    assert_eq!(response.status(), axum::http::StatusCode::BAD_GATEWAY);
    let body = response_text(response).await?;
    assert!(body.contains("backend_response_too_large"));
    Ok(())
}

#[tokio::test]
async fn backend_queue_timeout_does_not_trip_the_circuit() -> Result<(), Box<dyn Error>> {
    let backend = start_backend("queue", 20).await?;
    let mut config = gateway_config(&[("queue", &backend.url)], 4);
    config.breaker_failures = 1;
    config.retry_attempts = 0;
    let gateway = Gateway::new(config)?;

    let first = gateway
        .router()
        .oneshot(completion_request(true, 20)?)
        .await?;
    let mut first_body = first.into_body();
    let first_frame = tokio::time::timeout(Duration::from_secs(1), first_body.frame()).await?;
    assert!(first_frame.is_some());

    let queued = gateway
        .router()
        .oneshot(completion_request(false, 1)?)
        .await?;
    assert_eq!(queued.status(), axum::http::StatusCode::BAD_GATEWAY);
    drop(first_body);
    tokio::time::sleep(Duration::from_millis(25)).await;

    let recovered = gateway
        .router()
        .oneshot(completion_request(false, 1)?)
        .await?;
    assert_eq!(recovered.status(), axum::http::StatusCode::OK);
    Ok(())
}

#[tokio::test]
async fn backend_redirect_is_not_followed() -> Result<(), Box<dyn Error>> {
    let target = start_backend("redirect-target", 1).await?;
    let redirect = start_redirect_backend(&target.url).await?;
    let mut config = gateway_config(&[("redirect", &redirect.url)], 2);
    config.retry_attempts = 0;
    let gateway = Gateway::new(config)?;
    let response = gateway
        .router()
        .oneshot(completion_request(true, 2)?)
        .await?;
    let body = response_text(response).await?;
    assert!(body.contains("backend returned HTTP 307"));
    assert!(!body.contains("redirect-target:"));
    Ok(())
}

#[tokio::test]
async fn backend_error_body_is_not_disclosed_to_clients() -> Result<(), Box<dyn Error>> {
    let backend = start_backend("redacted", 1).await?;
    fault(&backend, &FaultCommand::RequestErrors { count: 1 }).await?;
    let mut config = gateway_config(&[("redacted", &backend.url)], 2);
    config.retry_attempts = 0;
    let gateway = Gateway::new(config)?;
    let response = gateway
        .router()
        .oneshot(completion_request(true, 1)?)
        .await?;
    let body = response_text(response).await?;
    assert!(body.contains("backend returned HTTP 500"));
    assert!(!body.contains("injected_request_error"));
    Ok(())
}

#[tokio::test]
async fn malformed_requests_are_structured_and_backend_health_is_visible()
-> Result<(), Box<dyn Error>> {
    let backend = start_backend("alpha", 1).await?;
    let gateway = Gateway::new(gateway_config(&[("alpha", &backend.url)], 2))?;
    let malformed = axum::http::Request::builder()
        .method("POST")
        .uri("/v1/completions")
        .header("content-type", "application/json")
        .body(Body::from("not json"))?;
    let response = gateway.router().oneshot(malformed).await?;
    assert_eq!(response.status(), axum::http::StatusCode::BAD_REQUEST);
    let body: Value = serde_json::from_str(&response_text(response).await?)?;
    assert_eq!(
        body.pointer("/error/error_type").and_then(Value::as_str),
        Some("invalid_request")
    );

    let health = gateway
        .router()
        .oneshot(
            axum::http::Request::builder()
                .uri("/health")
                .body(Body::empty())?,
        )
        .await?;
    assert_eq!(health.status(), axum::http::StatusCode::OK);
    assert!(response_text(health).await?.contains("alpha"));
    Ok(())
}

#[tokio::test]
async fn output_token_limit_is_enforced_before_backend_admission() -> Result<(), Box<dyn Error>> {
    let backend = start_backend("bounded", 1).await?;
    let mut config = gateway_config(&[("bounded", &backend.url)], 2);
    config.max_output_tokens = 4;
    let gateway = Gateway::new(config)?;
    let response = gateway
        .router()
        .oneshot(completion_request(true, 5)?)
        .await?;
    assert_eq!(response.status(), axum::http::StatusCode::BAD_REQUEST);
    let body = response_text(response).await?;
    assert!(body.contains("max_tokens must not exceed 4"));
    assert!(
        !gateway
            .metrics
            .render_prometheus()?
            .contains("sloforge_gateway_requests_total")
    );
    Ok(())
}

#[tokio::test]
async fn buffered_chat_uses_openai_chat_response_shape() -> Result<(), Box<dyn Error>> {
    let backend = start_backend("chat", 1).await?;
    let gateway = Gateway::new(gateway_config(&[("chat", &backend.url)], 2))?;
    let response = gateway.router().oneshot(chat_request(false)?).await?;
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let value: Value = serde_json::from_str(&response_text(response).await?)?;
    assert_eq!(
        value.get("object").and_then(Value::as_str),
        Some("chat.completion")
    );
    assert_eq!(
        value
            .pointer("/choices/0/message/role")
            .and_then(Value::as_str),
        Some("assistant")
    );
    assert!(
        value
            .pointer("/choices/0/message/content")
            .and_then(Value::as_str)
            .is_some_and(|content| content.contains("chat:0"))
    );
    Ok(())
}

#[tokio::test]
async fn active_health_checks_remove_and_restore_crashed_backend() -> Result<(), Box<dyn Error>> {
    let backend = start_backend("health", 1).await?;
    let mut config = gateway_config(&[("health", &backend.url)], 2);
    config.health_interval_ms = 50;
    let gateway = Gateway::new(config)?;
    let health_task = gateway.spawn_health_checks();
    fault(&backend, &FaultCommand::Crash { enabled: true }).await?;
    let mut unavailable = false;
    for _ in 0..20 {
        tokio::time::sleep(Duration::from_millis(15)).await;
        let response = gateway
            .router()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/health")
                    .body(Body::empty())?,
            )
            .await?;
        if response.status() == axum::http::StatusCode::SERVICE_UNAVAILABLE {
            unavailable = true;
            break;
        }
    }
    assert!(unavailable);

    fault(&backend, &FaultCommand::Clear).await?;
    let mut recovered = false;
    for _ in 0..20 {
        tokio::time::sleep(Duration::from_millis(15)).await;
        let response = gateway
            .router()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/health")
                    .body(Body::empty())?,
            )
            .await?;
        if response.status() == axum::http::StatusCode::OK {
            recovered = true;
            break;
        }
    }
    assert!(recovered);
    drop(health_task);
    Ok(())
}

#[tokio::test]
async fn circuit_breaker_opens_after_configured_failure_threshold() -> Result<(), Box<dyn Error>> {
    let backend = start_backend("breaker", 1).await?;
    fault(&backend, &FaultCommand::RequestErrors { count: 1 }).await?;
    let mut config = gateway_config(&[("breaker", &backend.url)], 2);
    config.breaker_failures = 1;
    config.breaker_cooldown_ms = 5_000;
    config.retry_attempts = 0;
    let gateway = Gateway::new(config)?;
    let response = gateway
        .router()
        .oneshot(completion_request(false, 1)?)
        .await?;
    assert_eq!(response.status(), axum::http::StatusCode::BAD_GATEWAY);
    let health = gateway
        .router()
        .oneshot(
            axum::http::Request::builder()
                .uri("/health")
                .body(Body::empty())?,
        )
        .await?;
    assert_eq!(health.status(), axum::http::StatusCode::SERVICE_UNAVAILABLE);
    let metrics = gateway.metrics.render_prometheus()?;
    assert!(metrics.contains("sloforge_gateway_circuit_open{backend=\"breaker\"} 1"));
    assert!(metrics.contains("sloforge_gateway_backend_outstanding{backend=\"breaker\"} 0"));
    Ok(())
}
