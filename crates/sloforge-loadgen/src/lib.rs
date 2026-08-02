//! Deterministic workload generation and bounded open-loop HTTP replay.

use futures::StreamExt;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use sloforge_sim::RequestSpec;
use std::collections::HashSet;
use std::io::{BufRead, Write};
use std::time::{Duration, Instant};
use tokio::task::JoinSet;

pub const TRACE_SCHEMA_VERSION: &str = "1.0";
const MAX_TRACE_RECORDS: usize = 1_000_000;
const MAX_LINE_BYTES: usize = 1024 * 1024;
const MAX_TOKENS_PER_REQUEST: u32 = 100_000;
const MAX_SSE_BUFFER_BYTES: usize = 1024 * 1024;

#[derive(Debug, thiserror::Error)]
pub enum LoadgenError {
    #[error("invalid workload configuration: {0}")]
    InvalidConfig(String),
    #[error("invalid trace at line {line}: {message}")]
    InvalidTrace { line: usize, message: String },
    #[error("trace exceeds {MAX_TRACE_RECORDS} records")]
    TraceTooLarge,
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("replay task failed: {0}")]
    Task(#[from] tokio::task::JoinError),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum TokenDistribution {
    Constant { value: u32 },
    Uniform { min: u32, max: u32 },
    Empirical { values: Vec<u32> },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequestClass {
    pub name: String,
    pub weight: f64,
    pub prompt_tokens: TokenDistribution,
    pub output_tokens: TokenDistribution,
    pub priority: u8,
    #[serde(default)]
    pub deadline_ms: Option<u64>,
    #[serde(default)]
    pub canary_eligible: bool,
    #[serde(default)]
    pub adapter_ids: Vec<String>,
    #[serde(default)]
    pub prefix_groups: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BurstWindow {
    pub start_ms: u64,
    pub duration_ms: u64,
    pub multiplier: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ArrivalProcess {
    FixedInterval {
        interval_ms: f64,
    },
    BurstyPoisson {
        base_rate_per_second: f64,
        bursts: Vec<BurstWindow>,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkloadConfig {
    #[serde(default = "default_schema")]
    pub schema_version: String,
    pub seed: u64,
    pub request_count: usize,
    #[serde(default)]
    pub max_duration_ms: Option<u64>,
    pub arrival_process: ArrivalProcess,
    pub classes: Vec<RequestClass>,
}

fn default_schema() -> String {
    TRACE_SCHEMA_VERSION.to_owned()
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TraceSummary {
    pub schema_version: String,
    pub record_count: usize,
    pub first_arrival_ms: Option<u64>,
    pub last_arrival_ms: Option<u64>,
    pub classes: Vec<String>,
    pub priorities: Vec<u8>,
}

/// Cross-language JSONL record shared with `python/sloforge/trace/format.py`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalTraceRequest {
    pub request_id: String,
    pub arrival_ms: f64,
    pub prompt_tokens: u32,
    pub output_tokens: u32,
    #[serde(default = "default_canonical_priority")]
    pub priority: u8,
    #[serde(default = "default_canonical_class")]
    pub request_class: String,
    #[serde(default)]
    pub deadline_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix_group: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cancelled_at_ms: Option<f64>,
}

const fn default_canonical_priority() -> u8 {
    1
}

fn default_canonical_class() -> String {
    "interactive".into()
}

#[derive(Deserialize)]
#[serde(untagged)]
enum ParsedTraceRequest {
    Canonical(CanonicalTraceRequest),
    Legacy(RequestSpec),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayConfig {
    pub target: String,
    pub model: String,
    pub time_scale: f64,
    pub max_concurrency: usize,
    pub request_timeout_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplayStatus {
    Completed,
    HttpError,
    TransportError,
    QueueSaturated,
    MalformedStream,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayMeasurement {
    pub request_id: String,
    pub scheduled_arrival_ms: u64,
    pub actual_start_ms: f64,
    pub status: ReplayStatus,
    pub http_status: Option<u16>,
    pub ttft_ms: Option<f64>,
    pub e2e_ms: f64,
    pub stream_events: usize,
    pub error: Option<String>,
}

/// Generate a deterministic mixed trace. Exponential inter-arrival samples are generated directly
/// to keep the dependency surface and random stream stable.
///
/// # Errors
///
/// Returns [`LoadgenError::InvalidConfig`] when the version, arrival process, class mixture, or
/// token distributions are inconsistent or exceed the bounded trace size.
pub fn generate(config: &WorkloadConfig) -> Result<Vec<RequestSpec>, LoadgenError> {
    validate_config(config)?;
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    let total_weight: f64 = config.classes.iter().map(|class| class.weight).sum();
    let mut now = 0.0_f64;
    let mut requests = Vec::with_capacity(config.request_count);
    for idx in 0..config.request_count {
        if idx > 0 {
            now += next_interarrival(&config.arrival_process, now, &mut rng)?;
        }
        let arrival_ms = rounded_millis(now);
        if config
            .max_duration_ms
            .is_some_and(|limit| arrival_ms > limit)
        {
            break;
        }
        let class = weighted_class(&config.classes, total_weight, &mut rng);
        let adapter_id = choose_optional(&class.adapter_ids, &mut rng);
        let prefix_group = choose_optional(&class.prefix_groups, &mut rng);
        requests.push(RequestSpec {
            id: format!("req-{idx:08}"),
            arrival_ms,
            prompt_tokens: sample_tokens(&class.prompt_tokens, &mut rng),
            output_tokens: sample_tokens(&class.output_tokens, &mut rng).max(1),
            priority: class.priority,
            request_class: class.name.clone(),
            deadline_ms: class.deadline_ms,
            cancel_after_ms: None,
            canary_eligible: class.canary_eligible,
            adapter_id,
            prefix_group,
        });
    }
    Ok(requests)
}

fn validate_config(config: &WorkloadConfig) -> Result<(), LoadgenError> {
    if config.schema_version.split('.').next() != Some("1") {
        return Err(LoadgenError::InvalidConfig(
            "only trace schema major version 1 is supported".into(),
        ));
    }
    if config.request_count == 0 || config.request_count > MAX_TRACE_RECORDS {
        return Err(LoadgenError::InvalidConfig(format!(
            "request_count must be in 1..={MAX_TRACE_RECORDS}"
        )));
    }
    if config.classes.is_empty() {
        return Err(LoadgenError::InvalidConfig(
            "at least one request class is required".into(),
        ));
    }
    let mut names = HashSet::new();
    for class in &config.classes {
        if class.name.is_empty() || !names.insert(&class.name) {
            return Err(LoadgenError::InvalidConfig(
                "class names must be non-empty and unique".into(),
            ));
        }
        if !class.weight.is_finite() || class.weight <= 0.0 {
            return Err(LoadgenError::InvalidConfig(format!(
                "class {} has invalid weight",
                class.name
            )));
        }
        if class.priority > 3 {
            return Err(LoadgenError::InvalidConfig(format!(
                "class {} priority must be in the canonical range 0..=3",
                class.name
            )));
        }
        validate_distribution(&class.prompt_tokens)?;
        validate_distribution(&class.output_tokens)?;
    }
    match &config.arrival_process {
        ArrivalProcess::FixedInterval { interval_ms }
            if !interval_ms.is_finite() || *interval_ms <= 0.0 =>
        {
            Err(LoadgenError::InvalidConfig(
                "fixed interval must be finite and positive".into(),
            ))
        }
        ArrivalProcess::BurstyPoisson {
            base_rate_per_second,
            bursts,
        } if !base_rate_per_second.is_finite()
            || *base_rate_per_second <= 0.0
            || bursts.iter().any(|burst| {
                burst.duration_ms == 0 || !burst.multiplier.is_finite() || burst.multiplier <= 0.0
            }) =>
        {
            Err(LoadgenError::InvalidConfig(
                "Poisson rate and burst windows must be positive".into(),
            ))
        }
        _ => Ok(()),
    }
}

fn validate_distribution(distribution: &TokenDistribution) -> Result<(), LoadgenError> {
    match distribution {
        TokenDistribution::Constant { value } if *value == 0 || *value > MAX_TOKENS_PER_REQUEST => {
            Err(LoadgenError::InvalidConfig(
                "token values must be within 1..=100000".into(),
            ))
        }
        TokenDistribution::Uniform { min, max }
            if *min == 0 || min > max || *max > MAX_TOKENS_PER_REQUEST =>
        {
            Err(LoadgenError::InvalidConfig(
                "token uniform range must be within 1..=100000".into(),
            ))
        }
        TokenDistribution::Empirical { values }
            if values.is_empty()
                || values
                    .iter()
                    .any(|value| *value == 0 || *value > MAX_TOKENS_PER_REQUEST) =>
        {
            Err(LoadgenError::InvalidConfig(
                "empirical token values must be non-empty and within 1..=100000".into(),
            ))
        }
        _ => Ok(()),
    }
}

fn next_interarrival(
    process: &ArrivalProcess,
    now_ms: f64,
    rng: &mut ChaCha8Rng,
) -> Result<f64, LoadgenError> {
    match process {
        ArrivalProcess::FixedInterval { interval_ms } => Ok(*interval_ms),
        ArrivalProcess::BurstyPoisson {
            base_rate_per_second,
            bursts,
        } => {
            let multiplier = bursts
                .iter()
                .filter(|burst| {
                    now_ms >= u64_to_f64(burst.start_ms)
                        && now_ms < u64_to_f64(burst.start_ms.saturating_add(burst.duration_ms))
                })
                .map(|burst| burst.multiplier)
                .product::<f64>();
            let rate = base_rate_per_second * multiplier;
            if !rate.is_finite() || rate <= 0.0 {
                return Err(LoadgenError::InvalidConfig(
                    "effective arrival rate is invalid".into(),
                ));
            }
            let uniform = rng.random_range(f64::EPSILON..1.0);
            Ok(-uniform.ln() * 1_000.0 / rate)
        }
    }
}

fn weighted_class<'a>(
    classes: &'a [RequestClass],
    total: f64,
    rng: &mut ChaCha8Rng,
) -> &'a RequestClass {
    let mut pick = rng.random_range(0.0..total);
    for class in classes {
        if pick < class.weight {
            return class;
        }
        pick -= class.weight;
    }
    &classes[classes.len() - 1]
}

fn sample_tokens(distribution: &TokenDistribution, rng: &mut ChaCha8Rng) -> u32 {
    match distribution {
        TokenDistribution::Constant { value } => *value,
        TokenDistribution::Uniform { min, max } => rng.random_range(*min..=*max),
        TokenDistribution::Empirical { values } => values[rng.random_range(0..values.len())],
    }
}

fn choose_optional(values: &[String], rng: &mut ChaCha8Rng) -> Option<String> {
    if values.is_empty() {
        None
    } else {
        Some(values[rng.random_range(0..values.len())].clone())
    }
}

/// Parse and validate bounded JSONL input.
///
/// # Errors
///
/// Returns an I/O error for unreadable input or [`LoadgenError::InvalidTrace`] when a bounded
/// record is malformed, duplicated, unsorted, or contains invalid token counts.
pub fn read_trace<R: BufRead>(mut reader: R) -> Result<Vec<RequestSpec>, LoadgenError> {
    let mut records = Vec::new();
    let mut line = String::new();
    let mut previous_arrival = None;
    let mut ids = HashSet::new();
    let mut line_number = 0;
    loop {
        line.clear();
        let count = reader.read_line(&mut line)?;
        if count == 0 {
            break;
        }
        line_number += 1;
        if count > MAX_LINE_BYTES {
            return Err(LoadgenError::InvalidTrace {
                line: line_number,
                message: format!("record exceeds {MAX_LINE_BYTES} bytes"),
            });
        }
        if line.trim().is_empty() {
            continue;
        }
        if records.len() >= MAX_TRACE_RECORDS {
            return Err(LoadgenError::TraceTooLarge);
        }
        let parsed: ParsedTraceRequest =
            serde_json::from_str(&line).map_err(|error| LoadgenError::InvalidTrace {
                line: line_number,
                message: error.to_string(),
            })?;
        let record = match parsed {
            ParsedTraceRequest::Canonical(record) => {
                canonical_to_internal(record).map_err(|message| LoadgenError::InvalidTrace {
                    line: line_number,
                    message,
                })?
            }
            ParsedTraceRequest::Legacy(record) => record,
        };
        if record.id.is_empty() || !ids.insert(record.id.clone()) {
            return Err(LoadgenError::InvalidTrace {
                line: line_number,
                message: format!("duplicate or empty request id {}", record.id),
            });
        }
        if previous_arrival.is_some_and(|previous| record.arrival_ms < previous) {
            return Err(LoadgenError::InvalidTrace {
                line: line_number,
                message: "arrival times must be non-decreasing".into(),
            });
        }
        if record.prompt_tokens == 0
            || record.output_tokens == 0
            || record.prompt_tokens > MAX_TOKENS_PER_REQUEST
            || record.output_tokens > MAX_TOKENS_PER_REQUEST
        {
            return Err(LoadgenError::InvalidTrace {
                line: line_number,
                message: "token counts must be within 1..=100000".into(),
            });
        }
        previous_arrival = Some(record.arrival_ms);
        records.push(record);
    }
    if records.is_empty() {
        return Err(LoadgenError::InvalidTrace {
            line: 0,
            message: "trace contains no requests".into(),
        });
    }
    Ok(records)
}

/// Write records using the canonical Python/Rust JSONL trace schema.
///
/// # Errors
///
/// Returns an I/O or JSON serialization error from the provided bounded writer.
pub fn write_trace<W: Write>(mut writer: W, records: &[RequestSpec]) -> Result<(), LoadgenError> {
    for record in records {
        serde_json::to_writer(&mut writer, &canonical_from_internal(record))?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}

fn canonical_from_internal(record: &RequestSpec) -> CanonicalTraceRequest {
    CanonicalTraceRequest {
        request_id: record.id.clone(),
        arrival_ms: u64_to_f64(record.arrival_ms),
        prompt_tokens: record.prompt_tokens,
        output_tokens: record.output_tokens,
        priority: record.priority,
        request_class: record.request_class.clone(),
        deadline_ms: record.deadline_ms.map(u64_to_f64),
        adapter_id: record.adapter_id.clone(),
        prefix_group: record.prefix_group.clone(),
        cancelled_at_ms: record
            .cancel_after_ms
            .map(|offset| u64_to_f64(record.arrival_ms.saturating_add(offset))),
    }
}

fn canonical_to_internal(record: CanonicalTraceRequest) -> Result<RequestSpec, String> {
    if record.priority > 3 {
        return Err("priority must be in the canonical range 0..=3".into());
    }
    let arrival_ms = checked_millis("arrival_ms", record.arrival_ms)?;
    let deadline_ms = record
        .deadline_ms
        .map(|value| checked_millis("deadline_ms", value))
        .transpose()?;
    let cancel_after_ms = record
        .cancelled_at_ms
        .map(|cancelled| {
            if cancelled < record.arrival_ms {
                Err("cancelled_at_ms precedes arrival_ms".into())
            } else {
                checked_millis("cancel_after_ms", cancelled - record.arrival_ms)
            }
        })
        .transpose()?;
    Ok(RequestSpec {
        id: record.request_id,
        arrival_ms,
        prompt_tokens: record.prompt_tokens,
        output_tokens: record.output_tokens,
        priority: record.priority,
        request_class: record.request_class,
        deadline_ms,
        cancel_after_ms,
        canary_eligible: true,
        adapter_id: record.adapter_id,
        prefix_group: record.prefix_group,
    })
}

#[must_use]
pub fn summarize_trace(records: &[RequestSpec]) -> TraceSummary {
    let mut classes: Vec<_> = records
        .iter()
        .map(|record| record.request_class.clone())
        .collect();
    classes.sort();
    classes.dedup();
    let mut priorities: Vec<_> = records.iter().map(|record| record.priority).collect();
    priorities.sort_unstable();
    priorities.dedup();
    TraceSummary {
        schema_version: TRACE_SCHEMA_VERSION.into(),
        record_count: records.len(),
        first_arrival_ms: records.first().map(|record| record.arrival_ms),
        last_arrival_ms: records.last().map(|record| record.arrival_ms),
        classes,
        priorities,
    }
}

/// Replay a trace as bounded open-loop streaming completion requests.
///
/// # Errors
///
/// Returns [`LoadgenError::InvalidConfig`] for unsafe replay bounds or a task error if a spawned
/// request worker panics. Per-request HTTP and SSE failures are returned as measurements.
pub async fn replay(
    records: &[RequestSpec],
    config: &ReplayConfig,
) -> Result<Vec<ReplayMeasurement>, LoadgenError> {
    if config.target.is_empty()
        || !config.time_scale.is_finite()
        || config.time_scale <= 0.0
        || config.max_concurrency == 0
        || config.request_timeout_ms == 0
    {
        return Err(LoadgenError::InvalidConfig(
            "invalid replay target, scale, concurrency, or timeout".into(),
        ));
    }
    let client = reqwest::Client::builder()
        .timeout(Duration::from_millis(config.request_timeout_ms))
        .build()
        .map_err(|error| LoadgenError::InvalidConfig(error.to_string()))?;
    let origin = Instant::now();
    let first_arrival = records.first().map_or(0, |record| record.arrival_ms);
    let mut tasks = JoinSet::new();
    let mut measurements = Vec::with_capacity(records.len());
    for record in records {
        let offset_ms =
            u64_to_f64(record.arrival_ms.saturating_sub(first_arrival)) / config.time_scale;
        tokio::time::sleep_until(tokio::time::Instant::from_std(
            origin + Duration::from_secs_f64(offset_ms / 1_000.0),
        ))
        .await;
        while let Some(result) = tasks.try_join_next() {
            measurements.push(result??);
        }
        if tasks.len() >= config.max_concurrency {
            measurements.push(ReplayMeasurement {
                request_id: record.id.clone(),
                scheduled_arrival_ms: record.arrival_ms,
                actual_start_ms: origin.elapsed().as_secs_f64() * 1_000.0,
                status: ReplayStatus::QueueSaturated,
                http_status: None,
                ttft_ms: None,
                e2e_ms: 0.0,
                stream_events: 0,
                error: Some("bounded client concurrency exhausted".into()),
            });
            continue;
        }
        let record = record.clone();
        let client = client.clone();
        let target = config.target.clone();
        let model = config.model.clone();
        tasks.spawn(async move { replay_one(client, target, model, record, origin).await });
    }
    while let Some(result) = tasks.join_next().await {
        measurements.push(result??);
    }
    measurements.sort_by(|left, right| left.request_id.cmp(&right.request_id));
    Ok(measurements)
}

#[allow(clippy::too_many_lines)]
async fn replay_one(
    client: reqwest::Client,
    target: String,
    model: String,
    request: RequestSpec,
    origin: Instant,
) -> Result<ReplayMeasurement, LoadgenError> {
    let started = Instant::now();
    let actual_start_ms = origin.elapsed().as_secs_f64() * 1_000.0;
    let body = serde_json::json!({
        "model": model,
        "stream": true,
        "prompt": "token ".repeat(request.prompt_tokens as usize),
        "max_tokens": request.output_tokens,
        "sloforge": {
            "request_id": request.id,
            "priority": request.priority,
            "deadline_ms": request.deadline_ms,
            "request_class": request.request_class,
        }
    });
    let response = match client.post(target).json(&body).send().await {
        Ok(response) => response,
        Err(error) => {
            return Ok(ReplayMeasurement {
                request_id: request.id,
                scheduled_arrival_ms: request.arrival_ms,
                actual_start_ms,
                status: ReplayStatus::TransportError,
                http_status: None,
                ttft_ms: None,
                e2e_ms: started.elapsed().as_secs_f64() * 1_000.0,
                stream_events: 0,
                error: Some(error.to_string()),
            });
        }
    };
    let status_code = response.status();
    if !status_code.is_success() {
        return Ok(ReplayMeasurement {
            request_id: request.id,
            scheduled_arrival_ms: request.arrival_ms,
            actual_start_ms,
            status: ReplayStatus::HttpError,
            http_status: Some(status_code.as_u16()),
            ttft_ms: None,
            e2e_ms: started.elapsed().as_secs_f64() * 1_000.0,
            stream_events: 0,
            error: Some(format!("HTTP {status_code}")),
        });
    }
    let mut stream = response.bytes_stream();
    let mut buffer = String::new();
    let mut ttft = None;
    let mut events = 0;
    let mut saw_done = false;
    while let Some(chunk) = stream.next().await {
        let chunk = match chunk {
            Ok(chunk) => chunk,
            Err(error) => {
                return Ok(ReplayMeasurement {
                    request_id: request.id,
                    scheduled_arrival_ms: request.arrival_ms,
                    actual_start_ms,
                    status: ReplayStatus::TransportError,
                    http_status: Some(status_code.as_u16()),
                    ttft_ms: ttft,
                    e2e_ms: started.elapsed().as_secs_f64() * 1_000.0,
                    stream_events: events,
                    error: Some(error.to_string()),
                });
            }
        };
        buffer.push_str(&String::from_utf8_lossy(&chunk));
        if buffer.len() > MAX_SSE_BUFFER_BYTES {
            return Ok(ReplayMeasurement {
                request_id: request.id,
                scheduled_arrival_ms: request.arrival_ms,
                actual_start_ms,
                status: ReplayStatus::MalformedStream,
                http_status: Some(status_code.as_u16()),
                ttft_ms: ttft,
                e2e_ms: started.elapsed().as_secs_f64() * 1_000.0,
                stream_events: events,
                error: Some("SSE frame exceeded the 1 MiB parser bound".into()),
            });
        }
        while let Some(boundary) = buffer.find("\n\n") {
            let event = buffer[..boundary].to_owned();
            buffer.drain(..boundary + 2);
            for data in event.lines().filter_map(|line| line.strip_prefix("data:")) {
                let data = data.trim();
                if data == "[DONE]" {
                    saw_done = true;
                } else if !data.is_empty() {
                    if ttft.is_none() {
                        ttft = Some(started.elapsed().as_secs_f64() * 1_000.0);
                    }
                    events += 1;
                }
            }
        }
    }
    let replay_status = if saw_done || events > 0 {
        ReplayStatus::Completed
    } else {
        ReplayStatus::MalformedStream
    };
    Ok(ReplayMeasurement {
        request_id: request.id,
        scheduled_arrival_ms: request.arrival_ms,
        actual_start_ms,
        status: replay_status,
        http_status: Some(status_code.as_u16()),
        ttft_ms: ttft,
        e2e_ms: started.elapsed().as_secs_f64() * 1_000.0,
        stream_events: events,
        error: (!saw_done && events == 0).then(|| "response contained no SSE data events".into()),
    })
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn rounded_millis(value: f64) -> u64 {
    value.round().clamp(0.0, u64::MAX as f64) as u64
}

fn checked_millis(name: &str, value: f64) -> Result<u64, String> {
    if !value.is_finite() || value < 0.0 || value > u64_to_f64(u64::MAX) {
        Err(format!(
            "{name} must be a finite non-negative millisecond value"
        ))
    } else {
        Ok(rounded_millis(value))
    }
}

#[allow(clippy::cast_precision_loss)]
fn u64_to_f64(value: u64) -> f64 {
    value as f64
}
