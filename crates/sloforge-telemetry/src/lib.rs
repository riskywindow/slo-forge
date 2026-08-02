//! Artifact-first telemetry primitives shared by `SLOForge`'s Rust components.
//!
//! The crate deliberately does not start an exporter or background thread.  Callers own
//! persistence and can therefore attach the raw artifact to an `EvidenceBundle`.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard};

/// Labels are ordered so exported metrics are byte-for-byte reproducible.
pub type Labels = BTreeMap<String, String>;

#[derive(Debug, thiserror::Error)]
pub enum TelemetryError {
    #[error("telemetry lock was poisoned")]
    LockPoisoned,
    #[error("invalid metric name: {0}")]
    InvalidMetricName(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct MetricKey {
    name: String,
    labels: Labels,
}

#[derive(Clone, Debug)]
enum MetricValue {
    Counter(u64),
    Gauge(f64),
    Histogram(Histogram),
}

#[derive(Clone, Debug)]
struct Histogram {
    buckets: Vec<f64>,
    counts: Vec<u64>,
    count: u64,
    sum: f64,
}

impl Histogram {
    fn new(mut buckets: Vec<f64>) -> Self {
        buckets.retain(|value| value.is_finite());
        buckets.sort_by(f64::total_cmp);
        buckets.dedup_by(|left, right| left.total_cmp(right).is_eq());
        let counts = vec![0; buckets.len()];
        Self {
            buckets,
            counts,
            count: 0,
            sum: 0.0,
        }
    }

    fn observe(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }
        self.count = self.count.saturating_add(1);
        self.sum += value;
        for (index, upper) in self.buckets.iter().enumerate() {
            if value <= *upper {
                self.counts[index] = self.counts[index].saturating_add(1);
            }
        }
    }
}

/// A small, dependency-light Prometheus registry.
///
/// Metrics are registered on first use. Type changes for an existing name/label pair are ignored,
/// which keeps instrumentation failure out of the request path.
#[derive(Clone, Debug, Default)]
pub struct MetricsRegistry {
    inner: Arc<Mutex<BTreeMap<MetricKey, MetricValue>>>,
}

impl MetricsRegistry {
    fn lock(&self) -> Result<MutexGuard<'_, BTreeMap<MetricKey, MetricValue>>, TelemetryError> {
        self.inner.lock().map_err(|_| TelemetryError::LockPoisoned)
    }

    /// Increment a counter, registering its label series on first use.
    ///
    /// # Errors
    ///
    /// Returns an error when the name is invalid or the registry lock is poisoned.
    pub fn increment_counter(
        &self,
        name: &str,
        labels: Labels,
        amount: u64,
    ) -> Result<(), TelemetryError> {
        validate_metric_name(name)?;
        let key = MetricKey {
            name: name.to_owned(),
            labels,
        };
        let mut metrics = self.lock()?;
        match metrics.entry(key).or_insert(MetricValue::Counter(0)) {
            MetricValue::Counter(value) => *value = value.saturating_add(amount),
            MetricValue::Gauge(_) | MetricValue::Histogram(_) => {}
        }
        Ok(())
    }

    /// Set a gauge, registering its label series on first use.
    ///
    /// # Errors
    ///
    /// Returns an error when the name is invalid or the registry lock is poisoned.
    pub fn set_gauge(&self, name: &str, labels: Labels, value: f64) -> Result<(), TelemetryError> {
        validate_metric_name(name)?;
        if !value.is_finite() {
            return Ok(());
        }
        let key = MetricKey {
            name: name.to_owned(),
            labels,
        };
        let mut metrics = self.lock()?;
        match metrics.entry(key).or_insert(MetricValue::Gauge(value)) {
            MetricValue::Gauge(current) => *current = value,
            MetricValue::Counter(_) | MetricValue::Histogram(_) => {}
        }
        Ok(())
    }

    /// Add an observation to a histogram with fixed cumulative buckets.
    ///
    /// # Errors
    ///
    /// Returns an error when the name is invalid or the registry lock is poisoned.
    pub fn observe_histogram(
        &self,
        name: &str,
        labels: Labels,
        value: f64,
        buckets: &[f64],
    ) -> Result<(), TelemetryError> {
        validate_metric_name(name)?;
        let key = MetricKey {
            name: name.to_owned(),
            labels,
        };
        let mut metrics = self.lock()?;
        match metrics
            .entry(key)
            .or_insert_with(|| MetricValue::Histogram(Histogram::new(buckets.to_vec())))
        {
            MetricValue::Histogram(histogram) => histogram.observe(value),
            MetricValue::Counter(_) | MetricValue::Gauge(_) => {}
        }
        Ok(())
    }

    /// Render the Prometheus text exposition format in deterministic key order.
    ///
    /// # Errors
    ///
    /// Returns an error if the registry lock is poisoned.
    pub fn render_prometheus(&self) -> Result<String, TelemetryError> {
        let metrics = self.lock()?;
        let mut output = String::new();
        let mut emitted_type = BTreeMap::<String, &'static str>::new();
        for (key, value) in metrics.iter() {
            let kind = match value {
                MetricValue::Counter(_) => "counter",
                MetricValue::Gauge(_) => "gauge",
                MetricValue::Histogram(_) => "histogram",
            };
            if !emitted_type.contains_key(&key.name) {
                let _ = writeln!(output, "# TYPE {} {kind}", key.name);
                emitted_type.insert(key.name.clone(), kind);
            }
            match value {
                MetricValue::Counter(counter) => {
                    let _ = writeln!(
                        output,
                        "{}{} {counter}",
                        key.name,
                        render_labels(&key.labels)
                    );
                }
                MetricValue::Gauge(gauge) => {
                    let _ = writeln!(output, "{}{} {gauge}", key.name, render_labels(&key.labels));
                }
                MetricValue::Histogram(histogram) => {
                    for (upper, count) in histogram.buckets.iter().zip(&histogram.counts) {
                        let mut labels = key.labels.clone();
                        labels.insert("le".to_owned(), upper.to_string());
                        let _ = writeln!(
                            output,
                            "{}_bucket{} {count}",
                            key.name,
                            render_labels(&labels)
                        );
                    }
                    let mut labels = key.labels.clone();
                    labels.insert("le".to_owned(), "+Inf".to_owned());
                    let _ = writeln!(
                        output,
                        "{}_bucket{} {}",
                        key.name,
                        render_labels(&labels),
                        histogram.count
                    );
                    let rendered = render_labels(&key.labels);
                    let _ = writeln!(output, "{}_sum{rendered} {}", key.name, histogram.sum);
                    let _ = writeln!(output, "{}_count{rendered} {}", key.name, histogram.count);
                }
            }
        }
        Ok(output)
    }
}

fn validate_metric_name(name: &str) -> Result<(), TelemetryError> {
    let mut characters = name.chars();
    let valid_first = characters.next().is_some_and(|character| {
        character.is_ascii_alphabetic() || character == '_' || character == ':'
    });
    let valid_rest = characters
        .all(|character| character.is_ascii_alphanumeric() || character == '_' || character == ':');
    if valid_first && valid_rest {
        Ok(())
    } else {
        Err(TelemetryError::InvalidMetricName(name.to_owned()))
    }
}

fn render_labels(labels: &Labels) -> String {
    if labels.is_empty() {
        return String::new();
    }
    let pairs = labels
        .iter()
        .map(|(key, value)| format!(r#"{key}="{}""#, escape_label(value)))
        .collect::<Vec<_>>()
        .join(",");
    format!("{{{pairs}}}")
}

fn escape_label(value: &str) -> String {
    value
        .replace('\\', r"\\")
        .replace('\n', r"\n")
        .replace('"', r#"\""#)
}

/// Chrome/Perfetto trace phase names supported by `SLOForge`.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
pub enum TracePhase {
    #[serde(rename = "B")]
    Begin,
    #[serde(rename = "E")]
    End,
    #[serde(rename = "X")]
    Complete,
    #[serde(rename = "i")]
    Instant,
    #[serde(rename = "C")]
    Counter,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct TraceEvent {
    pub name: String,
    pub cat: String,
    pub ph: TracePhase,
    /// Microseconds since the trace's monotonic origin.
    pub ts: u64,
    pub pid: u32,
    pub tid: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dur: Option<u64>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub args: BTreeMap<String, serde_json::Value>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct TraceArtifact {
    #[serde(rename = "traceEvents")]
    pub trace_events: Vec<TraceEvent>,
    /// Provenance is required so traces cannot be confused with fixture data.
    pub provenance: BTreeMap<String, String>,
    /// Events discarded after reaching the configured in-memory retention bound.
    #[serde(default)]
    pub dropped_events: u64,
    #[serde(rename = "displayTimeUnit")]
    pub display_time_unit: String,
}

#[derive(Clone, Debug)]
pub struct TraceCollector {
    inner: Arc<Mutex<TraceArtifact>>,
    max_events: usize,
}

impl TraceCollector {
    #[must_use]
    pub fn new(provenance: BTreeMap<String, String>) -> Self {
        Self::with_capacity(provenance, 100_000)
    }

    /// Create a collector with an explicit in-memory event retention bound.
    #[must_use]
    pub fn with_capacity(provenance: BTreeMap<String, String>, max_events: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(TraceArtifact {
                trace_events: Vec::new(),
                provenance,
                dropped_events: 0,
                display_time_unit: "ms".to_owned(),
            })),
            max_events,
        }
    }

    /// Record one trace event.
    ///
    /// # Errors
    ///
    /// Returns an error if the trace lock is poisoned.
    pub fn record(&self, event: TraceEvent) -> Result<(), TelemetryError> {
        let mut artifact = self
            .inner
            .lock()
            .map_err(|_| TelemetryError::LockPoisoned)?;
        if artifact.trace_events.len() < self.max_events {
            artifact.trace_events.push(event);
        } else {
            artifact.dropped_events = artifact.dropped_events.saturating_add(1);
        }
        Ok(())
    }

    /// Clone a consistent snapshot of the current trace artifact.
    ///
    /// # Errors
    ///
    /// Returns an error if the trace lock is poisoned.
    pub fn snapshot(&self) -> Result<TraceArtifact, TelemetryError> {
        Ok(self
            .inner
            .lock()
            .map_err(|_| TelemetryError::LockPoisoned)?
            .clone())
    }

    /// Persist the current artifact in Chrome/Perfetto JSON format.
    ///
    /// # Errors
    ///
    /// Returns an error for lock, serialization, or file-system failures.
    pub fn write_chrome_trace(&self, path: impl AsRef<Path>) -> Result<(), TelemetryError> {
        let artifact = self.snapshot()?;
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, &artifact)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
        Ok(())
    }

    /// Emit one JSON object per trace event for ingestion by artifact collectors.
    ///
    /// # Errors
    ///
    /// Returns an error for lock, serialization, or file-system failures.
    pub fn write_json_lines(&self, path: impl AsRef<Path>) -> Result<(), TelemetryError> {
        let artifact = self.snapshot()?;
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        for event in artifact.trace_events {
            serde_json::to_writer(&mut writer, &event)?;
            writer.write_all(b"\n")?;
        }
        writer.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prometheus_histograms_are_cumulative_and_deterministic() -> Result<(), TelemetryError> {
        let metrics = MetricsRegistry::default();
        let labels = Labels::from([("route".to_owned(), "a\"b".to_owned())]);
        metrics.observe_histogram("latency_seconds", labels.clone(), 0.4, &[1.0, 0.5])?;
        metrics.observe_histogram("latency_seconds", labels, 0.8, &[1.0, 0.5])?;
        let rendered = metrics.render_prometheus()?;
        assert!(rendered.contains("latency_seconds_bucket{le=\"0.5\",route=\"a\\\"b\"} 1"));
        assert!(rendered.contains("latency_seconds_bucket{le=\"1\",route=\"a\\\"b\"} 2"));
        assert!(rendered.contains("latency_seconds_count{route=\"a\\\"b\"} 2"));
        Ok(())
    }

    #[test]
    fn trace_artifact_round_trips() -> Result<(), TelemetryError> {
        let collector = TraceCollector::new(BTreeMap::from([(
            "raw_measurements".to_owned(),
            "measurements.jsonl".to_owned(),
        )]));
        collector.record(TraceEvent {
            name: "gateway.queue".to_owned(),
            cat: "request".to_owned(),
            ph: TracePhase::Complete,
            ts: 100,
            pid: 1,
            tid: 7,
            dur: Some(25),
            args: BTreeMap::new(),
        })?;
        let snapshot = collector.snapshot()?;
        let encoded = serde_json::to_string(&snapshot)?;
        let decoded: TraceArtifact = serde_json::from_str(&encoded)?;
        assert_eq!(snapshot, decoded);
        Ok(())
    }

    #[test]
    fn trace_retention_is_bounded_and_reports_drops() -> Result<(), TelemetryError> {
        let collector = TraceCollector::with_capacity(BTreeMap::new(), 1);
        let event = TraceEvent {
            name: "event".to_owned(),
            cat: "test".to_owned(),
            ph: TracePhase::Instant,
            ts: 1,
            pid: 1,
            tid: 1,
            dur: None,
            args: BTreeMap::new(),
        };
        collector.record(event.clone())?;
        collector.record(event)?;
        let artifact = collector.snapshot()?;
        assert_eq!(artifact.trace_events.len(), 1);
        assert_eq!(artifact.dropped_events, 1);
        Ok(())
    }

    #[test]
    fn invalid_metric_names_are_rejected() {
        let result = MetricsRegistry::default().increment_counter("bad-name", Labels::new(), 1);
        assert!(matches!(result, Err(TelemetryError::InvalidMetricName(_))));
    }
}
