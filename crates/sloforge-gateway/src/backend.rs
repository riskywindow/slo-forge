use crate::config::BackendConfig;
use crate::sse::{SseItem, SseParser};
use async_stream::try_stream;
use futures::{Stream, StreamExt};
use reqwest::Client;
use serde_json::Value;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

pub type BackendStream = Pin<Box<dyn Stream<Item = Result<BackendEvent, BackendError>> + Send>>;

#[derive(Debug)]
pub enum BackendEvent {
    Data(Value),
    Done,
}

#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    #[error("backend request failed: {0}")]
    Transport(#[from] reqwest::Error),
    #[error("backend returned HTTP {status}: {body}")]
    Status { status: u16, body: String },
    #[error("backend stream failed: {0}")]
    Stream(String),
    #[error("backend queue timeout")]
    QueueTimeout,
    #[error("backend circuit breaker is open")]
    CircuitOpen,
    #[error("backend request timeout")]
    RequestTimeout,
}

#[derive(Debug)]
struct CircuitBreaker {
    consecutive_failures: u32,
    open_until: Option<Instant>,
    half_open_in_flight: bool,
}

impl CircuitBreaker {
    const fn new() -> Self {
        Self {
            consecutive_failures: 0,
            open_until: None,
            half_open_in_flight: false,
        }
    }
}

#[derive(Debug)]
pub struct Backend {
    pub config: BackendConfig,
    capacity: Arc<Semaphore>,
    outstanding: Arc<AtomicUsize>,
    healthy: AtomicBool,
    estimated_service_micros: AtomicU64,
    breaker: Mutex<CircuitBreaker>,
    breaker_failures: u32,
    breaker_cooldown: Duration,
}

impl Backend {
    pub fn new(config: BackendConfig, breaker_failures: u32, breaker_cooldown: Duration) -> Self {
        let service_micros = millis_to_micros(config.estimated_service_ms);
        Self {
            capacity: Arc::new(Semaphore::new(config.capacity)),
            outstanding: Arc::new(AtomicUsize::new(0)),
            healthy: AtomicBool::new(true),
            estimated_service_micros: AtomicU64::new(service_micros),
            breaker: Mutex::new(CircuitBreaker::new()),
            breaker_failures,
            breaker_cooldown,
            config,
        }
    }

    pub fn outstanding(&self) -> usize {
        self.outstanding.load(Ordering::Relaxed)
    }

    #[allow(clippy::cast_precision_loss)]
    pub fn estimated_service_ms(&self) -> f64 {
        self.estimated_service_micros.load(Ordering::Relaxed) as f64 / 1_000.0
    }

    pub fn is_available(&self, now: Instant) -> bool {
        if !self.healthy.load(Ordering::Relaxed) {
            return false;
        }
        match self.breaker.lock() {
            Ok(breaker) => match breaker.open_until {
                Some(until) if now < until => false,
                // Only one request may reserve an expired breaker's half-open probe.
                Some(_) => !breaker.half_open_in_flight,
                None => true,
            },
            Err(_) => false,
        }
    }

    pub fn breaker_is_open(&self, now: Instant) -> bool {
        self.breaker.lock().map_or(true, |breaker| {
            breaker
                .open_until
                .is_some_and(|until| until > now || breaker.half_open_in_flight)
        })
    }

    pub fn set_healthy(&self, healthy: bool) {
        self.healthy.store(healthy, Ordering::Relaxed);
    }

    pub fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::Relaxed)
    }

    pub async fn acquire(
        self: &Arc<Self>,
        timeout: Duration,
    ) -> Result<BackendLease, BackendError> {
        let permit = tokio::time::timeout(timeout, Arc::clone(&self.capacity).acquire_owned())
            .await
            .map_err(|_| BackendError::QueueTimeout)?
            .map_err(|_| BackendError::QueueTimeout)?;
        let half_open = {
            let mut breaker = self.breaker.lock().map_err(|_| BackendError::CircuitOpen)?;
            match breaker.open_until {
                Some(until) if Instant::now() < until => return Err(BackendError::CircuitOpen),
                Some(_) if breaker.half_open_in_flight => {
                    return Err(BackendError::CircuitOpen);
                }
                Some(_) => {
                    breaker.half_open_in_flight = true;
                    true
                }
                None => false,
            }
        };
        self.outstanding.fetch_add(1, Ordering::Relaxed);
        Ok(BackendLease {
            backend: Arc::clone(self),
            _permit: permit,
            started: Instant::now(),
            half_open,
        })
    }

    pub fn record_failure(&self) {
        if let Ok(mut breaker) = self.breaker.lock() {
            breaker.consecutive_failures = breaker.consecutive_failures.saturating_add(1);
            if breaker.consecutive_failures >= self.breaker_failures.max(1) {
                breaker.open_until = Some(Instant::now() + self.breaker_cooldown);
            }
            breaker.half_open_in_flight = false;
        }
    }

    pub fn record_success(&self, elapsed: Duration) {
        if let Ok(mut breaker) = self.breaker.lock() {
            breaker.consecutive_failures = 0;
            breaker.open_until = None;
            breaker.half_open_in_flight = false;
        }
        let sample = u64::try_from(elapsed.as_micros()).unwrap_or(u64::MAX);
        let previous = self.estimated_service_micros.load(Ordering::Relaxed);
        // Integer EWMA, alpha=0.2, is stable and avoids non-atomic float state.
        let updated = previous
            .saturating_mul(4)
            .saturating_add(sample)
            .saturating_div(5);
        self.estimated_service_micros
            .store(updated.max(1), Ordering::Relaxed);
    }
}

pub struct BackendLease {
    backend: Arc<Backend>,
    _permit: OwnedSemaphorePermit,
    started: Instant,
    half_open: bool,
}

impl BackendLease {
    pub fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }
}

impl Drop for BackendLease {
    fn drop(&mut self) {
        self.backend.outstanding.fetch_sub(1, Ordering::Relaxed);
        if self.half_open {
            if let Ok(mut breaker) = self.backend.breaker.lock() {
                // A cancelled half-open request should release the probe reservation without
                // being misclassified as a backend failure. The next request may probe again.
                breaker.half_open_in_flight = false;
            }
        }
    }
}

pub async fn open_stream(
    client: &Client,
    backend: &Backend,
    endpoint: &str,
    mut payload: Value,
    max_frame_bytes: usize,
    request_id: &str,
    traceparent: &str,
) -> Result<BackendStream, BackendError> {
    if let Some(object) = payload.as_object_mut() {
        object.insert("stream".to_owned(), Value::Bool(true));
    }
    let url = format!(
        "{}{}",
        backend.config.base_url.trim_end_matches('/'),
        endpoint
    );
    let response = client
        .post(url)
        .header("x-request-id", request_id)
        .header("traceparent", traceparent)
        .json(&payload)
        .send()
        .await?;
    let status = response.status();
    if !status.is_success() {
        let body = read_bounded_error_body(response, 1_024).await;
        return Err(BackendError::Status {
            status: status.as_u16(),
            body,
        });
    }
    let stream = try_stream! {
        let mut parser = SseParser::new(max_frame_bytes);
        let mut bytes = response.bytes_stream();
        while let Some(chunk) = bytes.next().await {
            let chunk = chunk?;
            for item in parser.push(&chunk).map_err(|error| BackendError::Stream(error.to_string()))? {
                match item {
                    SseItem::Data(value) => yield BackendEvent::Data(value),
                    SseItem::Done => yield BackendEvent::Done,
                }
            }
        }
        for item in parser.finish().map_err(|error| BackendError::Stream(error.to_string()))? {
            match item {
                SseItem::Data(value) => yield BackendEvent::Data(value),
                SseItem::Done => yield BackendEvent::Done,
            }
        }
    };
    Ok(Box::pin(stream))
}

async fn read_bounded_error_body(response: reqwest::Response, max_bytes: usize) -> String {
    let mut bytes = response.bytes_stream();
    let mut body = Vec::with_capacity(max_bytes.min(1_024));
    let mut truncated = false;
    while let Some(chunk) = bytes.next().await {
        match chunk {
            Ok(chunk) => {
                let remaining = max_bytes.saturating_sub(body.len());
                body.extend_from_slice(&chunk[..chunk.len().min(remaining)]);
                if chunk.len() > remaining || body.len() == max_bytes {
                    truncated = true;
                    break;
                }
            }
            Err(error) => return format!("unreadable error body: {error}"),
        }
    }
    let mut rendered = String::from_utf8_lossy(&body).into_owned();
    if truncated {
        rendered.push_str("...[truncated]");
    }
    rendered
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn millis_to_micros(value: f64) -> u64 {
    if !value.is_finite() || value <= 0.0 {
        return 1;
    }
    let micros = value * 1_000.0;
    if micros >= u64::MAX as f64 {
        u64::MAX
    } else {
        micros as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn backend() -> Arc<Backend> {
        Arc::new(Backend::new(
            BackendConfig {
                name: "half-open".to_owned(),
                base_url: "http://127.0.0.1:1".to_owned(),
                capacity: 2,
                estimated_service_ms: 1.0,
                price_per_hour_usd: 0.0,
                health_path: "/health".to_owned(),
                weight: 1,
            },
            1,
            Duration::ZERO,
        ))
    }

    #[tokio::test]
    async fn expired_breaker_allows_only_one_half_open_probe() -> Result<(), BackendError> {
        let backend = backend();
        backend.record_failure();
        let probe = backend.acquire(Duration::from_millis(10)).await?;
        assert!(matches!(
            backend.acquire(Duration::from_millis(10)).await,
            Err(BackendError::CircuitOpen)
        ));
        drop(probe);
        let replacement_probe = backend.acquire(Duration::from_millis(10)).await?;
        drop(replacement_probe);
        Ok(())
    }
}
