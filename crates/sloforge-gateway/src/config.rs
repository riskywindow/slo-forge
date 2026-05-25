use crate::routing::RoutingPolicy;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GatewayConfig {
    pub bind: String,
    pub backends: Vec<BackendConfig>,
    #[serde(default)]
    pub routing_policy: RoutingPolicy,
    #[serde(default = "default_admission_capacity")]
    pub admission_capacity: usize,
    #[serde(default = "default_stream_buffer_bytes")]
    pub stream_buffer_bytes: usize,
    #[serde(default = "default_request_bytes")]
    pub max_request_bytes: usize,
    #[serde(default = "default_queue_timeout_ms")]
    pub queue_timeout_ms: u64,
    #[serde(default = "default_request_timeout_ms")]
    pub request_timeout_ms: u64,
    #[serde(default = "default_connect_timeout_ms")]
    pub connect_timeout_ms: u64,
    #[serde(default = "default_health_interval_ms")]
    pub health_interval_ms: u64,
    #[serde(default = "default_health_timeout_ms")]
    pub health_timeout_ms: u64,
    #[serde(default = "default_retry_attempts")]
    pub retry_attempts: usize,
    #[serde(default = "default_breaker_failures")]
    pub breaker_failures: u32,
    #[serde(default = "default_breaker_cooldown_ms")]
    pub breaker_cooldown_ms: u64,
    #[serde(default)]
    pub trace_output: Option<String>,
    #[serde(default = "default_trace_events")]
    pub max_trace_events: usize,
    #[serde(default)]
    pub provenance: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BackendConfig {
    pub name: String,
    pub base_url: String,
    #[serde(default = "default_capacity")]
    pub capacity: usize,
    #[serde(default = "default_service_ms")]
    pub estimated_service_ms: f64,
    #[serde(default)]
    pub price_per_hour_usd: f64,
    #[serde(default = "default_health_path")]
    pub health_path: String,
    #[serde(default = "default_weight")]
    pub weight: u32,
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("failed to read config: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid JSON config: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid gateway config: {0}")]
    Invalid(String),
}

impl GatewayConfig {
    /// Read and validate a gateway configuration from JSON.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be read, parsed, or validated.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let bytes = std::fs::read(path)?;
        let config: Self = serde_json::from_slice(&bytes)?;
        config.validate()?;
        Ok(config)
    }

    /// Validate safety bounds and backend identity constraints.
    ///
    /// # Errors
    ///
    /// Returns an error describing the first violated constraint.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.backends.is_empty() {
            return Err(ConfigError::Invalid(
                "at least one backend is required".to_owned(),
            ));
        }
        if self.backends.len() > 256 {
            return Err(ConfigError::Invalid(
                "at most 256 backends may be configured".to_owned(),
            ));
        }
        if self.admission_capacity == 0 {
            return Err(ConfigError::Invalid(
                "admission_capacity must be positive".to_owned(),
            ));
        }
        if self.stream_buffer_bytes < 256 {
            return Err(ConfigError::Invalid(
                "stream_buffer_bytes must be at least 256".to_owned(),
            ));
        }
        if self.stream_buffer_bytes > 64 * 1_024 * 1_024 || self.max_request_bytes == 0 {
            return Err(ConfigError::Invalid(
                "stream_buffer_bytes must not exceed 64 MiB and max_request_bytes must be positive"
                    .to_owned(),
            ));
        }
        if self.max_trace_events == 0 {
            return Err(ConfigError::Invalid(
                "max_trace_events must be positive".to_owned(),
            ));
        }
        if self.request_timeout_ms == 0
            || self.queue_timeout_ms == 0
            || self.connect_timeout_ms == 0
            || self.health_timeout_ms == 0
            || self.health_interval_ms == 0
        {
            return Err(ConfigError::Invalid("timeouts must be positive".to_owned()));
        }
        let mut names = std::collections::BTreeSet::new();
        for backend in &self.backends {
            if backend.name.trim().is_empty()
                || backend.name.len() > 128
                || !names.insert(&backend.name)
            {
                return Err(ConfigError::Invalid(format!(
                    "backend names must be non-empty and unique: {}",
                    backend.name
                )));
            }
            if backend.capacity == 0 || backend.weight == 0 {
                return Err(ConfigError::Invalid(format!(
                    "backend {} capacity and weight must be positive",
                    backend.name
                )));
            }
            let url = reqwest::Url::parse(&backend.base_url).map_err(|error| {
                ConfigError::Invalid(format!(
                    "backend {} has an invalid base_url: {error}",
                    backend.name
                ))
            })?;
            if !matches!(url.scheme(), "http" | "https")
                || url.host_str().is_none()
                || !url.username().is_empty()
                || url.password().is_some()
                || url.query().is_some()
                || url.fragment().is_some()
            {
                return Err(ConfigError::Invalid(format!(
                    "backend {} base_url must be an HTTP(S) origin/path without credentials, query, or fragment",
                    backend.name
                )));
            }
            if !backend.health_path.starts_with('/')
                || backend.health_path.starts_with("//")
                || backend.health_path.contains(['?', '#'])
                || backend.health_path.chars().any(char::is_control)
                || backend
                    .health_path
                    .split('/')
                    .any(|segment| matches!(segment, "." | ".."))
            {
                return Err(ConfigError::Invalid(format!(
                    "backend {} health_path must be an absolute path without query or fragment",
                    backend.name
                )));
            }
            if !backend.estimated_service_ms.is_finite()
                || backend.estimated_service_ms <= 0.0
                || !backend.price_per_hour_usd.is_finite()
                || backend.price_per_hour_usd < 0.0
            {
                return Err(ConfigError::Invalid(format!(
                    "backend {} has invalid performance or price values",
                    backend.name
                )));
            }
        }
        Ok(())
    }
}

const fn default_admission_capacity() -> usize {
    256
}
const fn default_stream_buffer_bytes() -> usize {
    1_048_576
}
const fn default_request_bytes() -> usize {
    1_048_576
}
const fn default_queue_timeout_ms() -> u64 {
    1_000
}
const fn default_request_timeout_ms() -> u64 {
    120_000
}
const fn default_connect_timeout_ms() -> u64 {
    2_000
}
const fn default_health_interval_ms() -> u64 {
    5_000
}
const fn default_health_timeout_ms() -> u64 {
    1_000
}
const fn default_retry_attempts() -> usize {
    1
}
const fn default_breaker_failures() -> u32 {
    3
}
const fn default_breaker_cooldown_ms() -> u64 {
    10_000
}
const fn default_capacity() -> usize {
    1
}
const fn default_service_ms() -> f64 {
    100.0
}
fn default_health_path() -> String {
    "/health".to_owned()
}
const fn default_weight() -> u32 {
    1
}
const fn default_trace_events() -> usize {
    100_000
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_duplicate_backend_names() -> Result<(), serde_json::Error> {
        let json = r#"{
          "bind":"127.0.0.1:0",
          "backends":[
            {"name":"same","base_url":"http://127.0.0.1:1"},
            {"name":"same","base_url":"http://127.0.0.1:2"}
          ]
        }"#;
        let parsed: GatewayConfig = serde_json::from_str(json)?;
        assert!(parsed.validate().is_err());
        Ok(())
    }

    #[test]
    fn rejects_backend_url_credentials_and_health_url_confusion() -> Result<(), serde_json::Error> {
        let with_credentials: GatewayConfig = serde_json::from_str(
            r#"{"bind":"127.0.0.1:0","backends":[{"name":"unsafe","base_url":"http://user:secret@127.0.0.1:1"}]}"#,
        )?;
        assert!(with_credentials.validate().is_err());

        let health_authority: GatewayConfig = serde_json::from_str(
            r#"{"bind":"127.0.0.1:0","backends":[{"name":"unsafe","base_url":"http://127.0.0.1:1","health_path":"//metadata.invalid"}]}"#,
        )?;
        assert!(health_authority.validate().is_err());
        Ok(())
    }
}
