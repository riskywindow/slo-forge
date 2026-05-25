use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

fn default_schema_version() -> String {
    crate::SIM_SCHEMA_VERSION.to_owned()
}

fn one() -> f64 {
    1.0
}

fn default_max_queue() -> usize {
    1_024
}

fn default_max_active() -> usize {
    8
}

/// A reproducible empirical or synthetic duration distribution.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum DurationDistribution {
    Constant { value_ms: f64 },
    Uniform { min_ms: f64, max_ms: f64 },
    Empirical { samples_ms: Vec<f64> },
}

impl Default for DurationDistribution {
    fn default() -> Self {
        Self::Constant { value_ms: 0.0 }
    }
}

/// A measured service curve. Coefficients are milliseconds and are intentionally explicit.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServiceCurve {
    pub id: String,
    #[serde(default)]
    pub measurement_artifact: String,
    #[serde(default)]
    pub prefill_intercept_ms: f64,
    pub prefill_ms_per_prompt_token: f64,
    #[serde(default)]
    pub prefill_ms_per_batch_item: f64,
    #[serde(default)]
    pub chunk_overhead_ms: f64,
    #[serde(default)]
    pub chunk_size_tokens: Option<u32>,
    #[serde(default)]
    pub decode_intercept_ms: f64,
    pub decode_ms_per_active_sequence: f64,
    #[serde(default)]
    pub decode_ms_per_context_token: f64,
    #[serde(default)]
    pub startup: DurationDistribution,
}

/// A trace request. Times are offsets from scenario start.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequestSpec {
    pub id: String,
    pub arrival_ms: u64,
    pub prompt_tokens: u32,
    pub output_tokens: u32,
    #[serde(default)]
    pub priority: u8,
    #[serde(default)]
    pub request_class: String,
    #[serde(default)]
    pub deadline_ms: Option<u64>,
    #[serde(default)]
    pub cancel_after_ms: Option<u64>,
    #[serde(default = "default_true")]
    pub canary_eligible: bool,
    #[serde(default)]
    pub adapter_id: Option<String>,
    #[serde(default)]
    pub prefix_group: Option<String>,
}

const fn default_true() -> bool {
    true
}

/// A replica present at scenario start or supplied to an add-replica action.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplicaSpec {
    pub id: String,
    #[serde(default = "default_true")]
    pub initially_warm: bool,
    #[serde(default = "default_max_queue")]
    pub max_queue: usize,
    #[serde(default = "default_max_active")]
    pub max_active_sequences: usize,
    #[serde(default = "one")]
    pub service_rate_multiplier: f64,
    #[serde(default)]
    pub hourly_price_usd: f64,
    #[serde(default)]
    pub canary: bool,
}

/// The routing algorithms shared with the gateway.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoutingPolicy {
    #[default]
    RoundRobin,
    LeastOutstanding,
    EarliestFinish,
    SloSlackAware,
}

/// Timed capacity and fault mutations.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ScenarioAction {
    BackendCrash {
        replica_id: String,
    },
    BackendRecover {
        replica_id: String,
        warm: bool,
    },
    BackendSlowdown {
        replica_id: String,
        factor: f64,
    },
    StartupSlowdown {
        replica_id: String,
        factor: f64,
    },
    RequestErrors {
        replica_id: String,
        probability: f64,
    },
    AddReplica {
        replica: ReplicaSpec,
    },
    RemoveReplica {
        replica_id: String,
    },
    CapacityLoss {
        replica_id: String,
        max_active_sequences: usize,
    },
    QueueSaturation {
        replica_id: String,
        max_queue: usize,
    },
    NetworkLatency {
        replica_id: String,
        latency_ms: f64,
        jitter_ms: f64,
    },
    SimulatedOom {
        replica_id: String,
        max_context_tokens: u32,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TimedAction {
    pub at_ms: u64,
    pub action: ScenarioAction,
}

/// Complete deterministic simulator input.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SimulationRequest {
    #[serde(default = "default_schema_version")]
    pub schema_version: String,
    pub seed: u64,
    pub service_curve: ServiceCurve,
    pub replicas: Vec<ReplicaSpec>,
    pub requests: Vec<RequestSpec>,
    #[serde(default)]
    pub actions: Vec<TimedAction>,
    #[serde(default)]
    pub routing_policy: RoutingPolicy,
    #[serde(default)]
    pub canary_weight: f64,
    #[serde(default = "default_max_events")]
    pub max_events: usize,
}

const fn default_max_events() -> usize {
    5_000_000
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutcomeStatus {
    Completed,
    DeadlineExceeded,
    Cancelled,
    Rejected,
    BackendFailed,
    SimulatedOom,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequestOutcome {
    pub request_id: String,
    pub status: OutcomeStatus,
    pub replica_id: Option<String>,
    pub arrival_ms: f64,
    pub first_token_ms: Option<f64>,
    pub terminal_ms: f64,
    pub completed_ms: Option<f64>,
    pub queue_ms: Option<f64>,
    pub prefill_ms: Option<f64>,
    pub decode_ms: Option<f64>,
    pub ttft_ms: Option<f64>,
    pub e2e_ms: Option<f64>,
    pub mean_itl_ms: Option<f64>,
    pub generated_tokens: u32,
    pub deadline_met: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SimulationMetrics {
    pub request_count: usize,
    pub completed_count: usize,
    pub rejected_count: usize,
    pub failed_count: usize,
    pub deadline_miss_count: usize,
    pub p50_ttft_ms: Option<f64>,
    pub p95_ttft_ms: Option<f64>,
    pub p99_ttft_ms: Option<f64>,
    pub p50_e2e_ms: Option<f64>,
    pub p95_e2e_ms: Option<f64>,
    pub p99_e2e_ms: Option<f64>,
    pub p99_itl_ms: Option<f64>,
    pub p95_queue_ms: Option<f64>,
    pub p95_prefill_ms: Option<f64>,
    pub p95_decode_ms: Option<f64>,
    pub throughput_tokens_per_s: f64,
    pub goodput_requests_per_s: f64,
    pub availability: f64,
    pub cost_usd: f64,
    pub simulated_duration_ms: f64,
    pub processed_events: usize,
}

/// Chrome/Perfetto trace-event compatible record (`ts` and `dur` are microseconds).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TraceEvent {
    pub name: String,
    pub cat: String,
    pub ph: String,
    pub ts: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dur: Option<f64>,
    pub pid: u32,
    pub tid: String,
    #[serde(default)]
    pub args: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SimulationProvenance {
    pub simulator_version: String,
    pub input_sha256: String,
    pub service_curve_id: String,
    pub measurement_artifact: String,
    pub seed: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SimulationOutput {
    pub schema_version: String,
    pub provenance: SimulationProvenance,
    pub metrics: SimulationMetrics,
    pub outcomes: Vec<RequestOutcome>,
    pub trace_events: Vec<TraceEvent>,
}
