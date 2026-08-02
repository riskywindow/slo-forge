use std::collections::{BTreeMap, BTreeSet};

use chrono::{DateTime, Utc};
use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

use crate::{Validate, ValidationError};

pub const SCHEMA_VERSION: &str = "1.0.0";
pub const API_VERSION: &str = "sloforge.io/v1";

fn nonempty(path: &str, value: &str) -> Result<(), ValidationError> {
    if value.trim().is_empty() {
        Err(ValidationError::new(path, "must not be empty"))
    } else {
        Ok(())
    }
}

fn semver(path: &str, value: &str) -> Result<(), ValidationError> {
    fn identifiers_are_valid(value: &str) -> bool {
        !value.is_empty()
            && value.split('.').all(|part| {
                !part.is_empty()
                    && part
                        .bytes()
                        .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
            })
    }

    let (without_build, build) = value
        .split_once('+')
        .map_or((value, None), |(core, suffix)| (core, Some(suffix)));
    if build.is_some_and(|suffix| !identifiers_are_valid(suffix)) {
        return Err(ValidationError::new(path, "must be valid semantic version"));
    }
    let (core, prerelease) = without_build
        .split_once('-')
        .map_or((without_build, None), |(base, suffix)| (base, Some(suffix)));
    if prerelease.is_some_and(|suffix| !identifiers_are_valid(suffix)) {
        return Err(ValidationError::new(path, "must be valid semantic version"));
    }
    let parts: Vec<_> = core.split('.').collect();
    let valid_core = parts.len() == 3
        && parts.iter().all(|part| {
            !part.is_empty()
                && part.bytes().all(|byte| byte.is_ascii_digit())
                && (part == &"0" || !part.starts_with('0'))
        });
    if valid_core {
        Ok(())
    } else {
        Err(ValidationError::new(path, "must be valid semantic version"))
    }
}

fn extension_key_is_valid(key: &str) -> bool {
    let Some((namespace, name)) = key.split_once('/') else {
        return false;
    };
    if namespace.is_empty() || name.is_empty() || namespace.contains('/') {
        return false;
    }
    let mut chars = namespace.chars();
    if !chars.next().is_some_and(|ch| ch.is_ascii_lowercase()) {
        return false;
    }
    if !chars.all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || matches!(ch, '.' | '-')) {
        return false;
    }
    let mut name_chars = name.chars();
    name_chars.next().is_some_and(|ch| ch.is_ascii_alphabetic())
        && name_chars.all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.' | '-'))
}

fn validate_json_value(path: &str, value: &Value) -> Result<(), ValidationError> {
    match value {
        Value::Array(items) => {
            for (index, item) in items.iter().enumerate() {
                validate_json_value(&format!("{path}[{index}]"), item)?;
            }
        }
        Value::Object(items) => {
            for (key, item) in items {
                validate_json_value(&format!("{path}.{key}"), item)?;
            }
        }
        _ => {}
    }
    Ok(())
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, JsonSchema)]
#[serde(transparent)]
pub struct Extensions(pub BTreeMap<String, Value>);

impl<'de> Deserialize<'de> for Extensions {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = BTreeMap::<String, Value>::deserialize(deserializer)?;
        let result = Self(value);
        result.validate().map_err(serde::de::Error::custom)?;
        Ok(result)
    }
}

impl Validate for Extensions {
    fn validate(&self) -> Result<(), ValidationError> {
        for (key, value) in &self.0 {
            if !extension_key_is_valid(key) {
                return Err(ValidationError::new(
                    format!("extensions.{key}"),
                    "extension key must be namespace-qualified",
                ));
            }
            validate_json_value(&format!("extensions.{key}"), value)?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum DigestAlgorithm {
    #[default]
    Sha256,
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactDigest {
    #[serde(default)]
    pub algorithm: DigestAlgorithm,
    pub value: String,
}

impl ArtifactDigest {
    fn validate_at(&self, path: &str) -> Result<(), ValidationError> {
        if self.value.len() != 64
            || !self
                .value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(ValidationError::new(
                path,
                "sha256 digest must be 64 lowercase hexadecimal characters",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DocumentMetadata {
    pub name: String,
    pub uid: String,
    #[serde(default = "default_generation")]
    pub generation: u64,
    pub created_at: DateTime<Utc>,
    #[serde(default)]
    pub labels: BTreeMap<String, String>,
}

const fn default_generation() -> u64 {
    1
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LicenseMetadata {
    pub spdx_id: String,
    pub name: String,
    pub url: Option<String>,
    pub redistribution_allowed: bool,
    pub verified_at: Option<DateTime<Utc>>,
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelArchitecture {
    pub family: String,
    pub parameter_count: u64,
    pub hidden_size: u32,
    pub num_layers: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub vocabulary_size: u32,
}

#[derive(
    Clone, Copy, Debug, Deserialize, Eq, JsonSchema, Ord, PartialEq, PartialOrd, Serialize,
)]
#[serde(rename_all = "lowercase")]
pub enum DType {
    Float32,
    Tf32,
    Float16,
    Bfloat16,
    Fp8,
    Int8,
    Int4,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Quantization {
    #[default]
    None,
    Awq,
    Gptq,
    Bitsandbytes,
    Fp8,
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LoraSpec {
    pub adapter_id: String,
    pub revision: String,
    pub checksum: ArtifactDigest,
    pub rank: u32,
    #[serde(default)]
    pub merged: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelSpec {
    pub model_id: String,
    pub revision: String,
    pub checksum: ArtifactDigest,
    pub tokenizer_id: String,
    pub tokenizer_revision: String,
    pub architecture: ModelArchitecture,
    pub allowed_precisions: Vec<DType>,
    pub minimum_precision: DType,
    pub maximum_sequence_length: u32,
    #[serde(default)]
    pub lora: Vec<LoraSpec>,
    pub license: LicenseMetadata,
    #[serde(default)]
    pub extensions: Extensions,
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ChunkedPrefillSpec {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_chunk_tokens")]
    pub chunk_tokens: u32,
}

const fn default_chunk_tokens() -> u32 {
    512
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PrefixCacheSpec {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub capacity_tokens: u64,
    #[serde(default)]
    pub eviction_policy: EvictionPolicy,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EvictionPolicy {
    #[default]
    Lru,
    Clock,
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SpeculativeDecodingSpec {
    #[serde(default)]
    pub enabled: bool,
    pub draft_model_id: Option<String>,
    #[serde(default = "default_draft_tokens")]
    pub maximum_draft_tokens: u32,
}

const fn default_draft_tokens() -> u32 {
    4
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CompilationMode {
    #[default]
    Eager,
    Compile,
    AheadOfTime,
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CompilationSpec {
    #[serde(default)]
    pub mode: CompilationMode,
    #[serde(default)]
    pub cuda_graphs: bool,
    #[serde(default)]
    pub graph_batch_sizes: Vec<u32>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Runtime {
    Transformers,
    Vllm,
    Sglang,
    TensorrtLlm,
    Mock,
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EngineSpec {
    pub runtime: Runtime,
    pub version: String,
    pub dtype: DType,
    #[serde(default)]
    pub quantization: Quantization,
    #[serde(default = "one_u32")]
    pub tensor_parallelism: u32,
    #[serde(default = "one_u32")]
    pub pipeline_parallelism: u32,
    pub maximum_batched_tokens: u32,
    pub maximum_active_sequences: u32,
    pub chunked_prefill: ChunkedPrefillSpec,
    pub prefix_cache: PrefixCacheSpec,
    pub speculative_decoding: SpeculativeDecodingSpec,
    pub compilation: CompilationSpec,
    #[serde(default)]
    pub extensions: Extensions,
}

const fn one_u32() -> u32 {
    1
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CpuSpec {
    pub architecture: String,
    pub model: String,
    pub physical_cores: u32,
    pub logical_cores: u32,
    pub numa_nodes: u32,
    pub measured_memory_bandwidth_gbps: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GpuSpec {
    pub index: u32,
    pub product: String,
    pub architecture: String,
    pub uuid: String,
    pub vram_bytes: u64,
    pub memory_clock_mhz: Option<f64>,
    pub measured_memory_bandwidth_gbps: Option<f64>,
    pub measured_compute_tflops: Option<f64>,
    pub pcie_generation: Option<u32>,
    pub pcie_width: Option<u32>,
    pub ecc_enabled: Option<bool>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TopologyKind {
    Pcie,
    Nvlink,
    SharedMemory,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TopologyLink {
    pub source_gpu: u32,
    pub target_gpu: u32,
    pub kind: TopologyKind,
    pub measured_bandwidth_gbps: Option<f64>,
    pub measured_latency_us: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HardwareSpec {
    pub fingerprint: ArtifactDigest,
    pub cpu: CpuSpec,
    pub system_memory_bytes: u64,
    #[serde(default)]
    pub gpu_count: u32,
    #[serde(default)]
    pub gpus: Vec<GpuSpec>,
    #[serde(default)]
    pub topology: Vec<TopologyLink>,
    pub driver_version: Option<String>,
    pub cuda_version: Option<String>,
    #[serde(default)]
    pub library_versions: BTreeMap<String, String>,
    pub hourly_price_usd: f64,
    pub region: String,
    pub cloud: Option<String>,
    pub instance_type: Option<String>,
    pub container_memory_limit_bytes: Option<u64>,
    #[serde(default)]
    pub extensions: Extensions,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ArrivalKind {
    Poisson,
    Deterministic,
    Trace,
    Bursty,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ArrivalProcess {
    pub kind: ArrivalKind,
    pub rate_per_second: Option<f64>,
    pub burst_factor: Option<f64>,
    pub trace_uri: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DistributionKind {
    Fixed,
    Empirical,
    Lognormal,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WeightedValue {
    pub value: u32,
    pub weight: f64,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DistributionSpec {
    pub kind: DistributionKind,
    pub fixed_value: Option<u32>,
    #[serde(default)]
    pub empirical: Vec<WeightedValue>,
    pub log_mean: Option<f64>,
    pub log_stddev: Option<f64>,
    #[serde(default)]
    pub minimum: u32,
    pub maximum: u32,
}

#[derive(
    Clone, Copy, Debug, Deserialize, Eq, JsonSchema, Ord, PartialEq, PartialOrd, Serialize,
)]
#[serde(rename_all = "snake_case")]
pub enum Priority {
    Critical,
    Interactive,
    Batch,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RequestClass {
    pub name: String,
    pub weight: f64,
    pub priority: Priority,
    pub tenant: Option<String>,
    pub deadline_ms: Option<f64>,
    #[serde(default)]
    pub adapter_ids: Vec<String>,
    pub prefix_group: Option<String>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkloadSpec {
    pub arrival_process: ArrivalProcess,
    pub prompt_tokens: DistributionSpec,
    pub output_tokens: DistributionSpec,
    pub request_classes: Vec<RequestClass>,
    pub duration_seconds: f64,
    pub seed: u64,
    pub trace_digest: Option<ArtifactDigest>,
    #[serde(default)]
    pub extensions: Extensions,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MetricConstraint {
    pub percentile: f64,
    pub maximum_ms: f64,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FluidityConstraint {
    pub token_deadline_ms: f64,
    pub maximum_missed_fraction: f64,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ObjectiveWeights {
    #[serde(default = "one_f64")]
    pub cost: f64,
    #[serde(default)]
    pub latency: f64,
    #[serde(default)]
    pub goodput: f64,
    #[serde(default)]
    pub availability: f64,
}

const fn one_f64() -> f64 {
    1.0
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SLOSpec {
    #[serde(default)]
    pub ttft: Vec<MetricConstraint>,
    #[serde(default)]
    pub inter_token_latency: Vec<MetricConstraint>,
    #[serde(default)]
    pub end_to_end_latency: Vec<MetricConstraint>,
    pub fluidity: Option<FluidityConstraint>,
    pub minimum_goodput_rps: Option<f64>,
    pub minimum_availability: Option<f64>,
    pub maximum_cost_per_million_tokens_usd: Option<f64>,
    pub objective_weights: ObjectiveWeights,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BudgetSpec {
    pub profiling_budget_usd: f64,
    pub profiling_duration_seconds: Option<f64>,
    pub maximum_real_trials: Option<u32>,
    #[serde(default = "default_reserve")]
    pub reserve_fraction: f64,
}

const fn default_reserve() -> f64 {
    0.15
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReplicaTopology {
    pub minimum_replicas: u32,
    pub maximum_replicas: u32,
    pub initial_replicas: u32,
    #[serde(default = "one_u32")]
    pub tensor_parallelism_per_replica: u32,
    pub regions: Vec<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RoutingPolicyKind {
    RoundRobin,
    LeastOutstanding,
    EarliestFinish,
    SloSlack,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RouteTarget {
    pub variant: String,
    pub weight: f64,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RoutingPolicy {
    pub kind: RoutingPolicyKind,
    pub targets: Vec<RouteTarget>,
    #[serde(default = "default_health_penalty")]
    pub health_penalty_ms: f64,
    #[serde(default)]
    pub cold_start_penalty_ms: f64,
}

const fn default_health_penalty() -> f64 {
    1000.0
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AdmissionPolicy {
    pub queue_capacity: u32,
    pub maximum_queue_time_ms: f64,
    pub shed_below_priority: Option<Priority>,
    #[serde(default = "default_true")]
    pub reject_when_predicted_late: bool,
}

const fn default_true() -> bool {
    true
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BatchingPolicy {
    pub maximum_active_sequences: u32,
    pub maximum_batched_tokens: u32,
    pub maximum_batch_delay_ms: f64,
    #[serde(default = "default_true")]
    pub dynamic_batching: bool,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AutoscalingMode {
    Disabled,
    Reactive,
    Predictive,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AutoscalingPolicy {
    pub mode: AutoscalingMode,
    pub target_utilization: f64,
    pub control_interval_seconds: f64,
    pub scale_up_cooldown_seconds: f64,
    pub scale_down_cooldown_seconds: f64,
    pub minimum_samples: u32,
    pub safety_margin: f64,
    pub maximum_change_per_interval: u32,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ColdStartStrategy {
    pub minimum_warm_replicas: u32,
    #[serde(default = "default_true")]
    pub prefetch_model: bool,
    pub readiness_timeout_seconds: f64,
    pub predicted_p95_startup_ms: f64,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CanaryPolicy {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_canary_weight")]
    pub initial_weight: f64,
    #[serde(default = "default_canary_requests")]
    pub minimum_requests: u32,
    #[serde(default = "default_canary_observation")]
    pub observation_seconds: f64,
    #[serde(default = "default_canary_delta")]
    pub maximum_slo_violation_delta: f64,
}

const fn default_canary_weight() -> f64 {
    0.05
}
const fn default_canary_requests() -> u32 {
    100
}
const fn default_canary_observation() -> f64 {
    60.0
}
const fn default_canary_delta() -> f64 {
    0.01
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RollbackPolicy {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_violation_windows")]
    pub violation_windows: u32,
    #[serde(default = "default_rollback_window")]
    pub window_seconds: f64,
    #[serde(default = "default_availability_floor")]
    pub availability_floor: f64,
    #[serde(default = "default_rollback_cooldown")]
    pub cooldown_seconds: f64,
}

const fn default_violation_windows() -> u32 {
    2
}
const fn default_rollback_window() -> f64 {
    30.0
}
const fn default_availability_floor() -> f64 {
    0.99
}
const fn default_rollback_cooldown() -> f64 {
    120.0
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MetricEstimate {
    pub point: f64,
    pub lower: f64,
    pub upper: f64,
    pub confidence: f64,
    pub unit: String,
    pub sample_count: u64,
    pub measurement_ids: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Provenance {
    pub profile_id: String,
    pub optimizer_run_id: String,
    pub workload_digest: ArtifactDigest,
    pub hardware_fingerprint: ArtifactDigest,
    pub evidence_bundle_uri: String,
    pub compiler_version: String,
    pub git_commit: String,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DeploymentPlan {
    pub schema_version: String,
    pub api_version: String,
    pub kind: String,
    pub metadata: DocumentMetadata,
    pub model: ModelSpec,
    pub engine: EngineSpec,
    pub hardware: HardwareSpec,
    pub workload: WorkloadSpec,
    pub slo: SLOSpec,
    pub budget: BudgetSpec,
    pub replica_topology: ReplicaTopology,
    pub routing: RoutingPolicy,
    pub admission: AdmissionPolicy,
    pub batching: BatchingPolicy,
    pub autoscaling: AutoscalingPolicy,
    pub cold_start: ColdStartStrategy,
    pub canary: CanaryPolicy,
    pub rollback: RollbackPolicy,
    pub predicted_metrics: BTreeMap<String, MetricEstimate>,
    pub provenance: Provenance,
    #[serde(default)]
    pub extensions: Extensions,
}

fn positive(path: &str, value: f64) -> Result<(), ValidationError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(ValidationError::new(path, "must be finite and positive"))
    }
}

fn nonnegative(path: &str, value: f64) -> Result<(), ValidationError> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(ValidationError::new(path, "must be finite and nonnegative"))
    }
}

fn positive_integer(path: &str, value: u64) -> Result<(), ValidationError> {
    if value == 0 {
        Err(ValidationError::new(path, "must be positive"))
    } else {
        Ok(())
    }
}

fn probability(path: &str, value: f64) -> Result<(), ValidationError> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(ValidationError::new(path, "must be between zero and one"))
    }
}

fn validate_metric(path: &str, metric: &MetricEstimate) -> Result<(), ValidationError> {
    if !metric.point.is_finite() || !metric.lower.is_finite() || !metric.upper.is_finite() {
        return Err(ValidationError::new(path, "metric values must be finite"));
    }
    if !(metric.lower <= metric.point && metric.point <= metric.upper) {
        return Err(ValidationError::new(path, "point must lie within interval"));
    }
    if !(0.0 < metric.confidence && metric.confidence < 1.0) {
        return Err(ValidationError::new(
            path,
            "confidence must be between zero and one",
        ));
    }
    if metric.sample_count > 0 && metric.measurement_ids.is_empty() {
        return Err(ValidationError::new(
            path,
            "measured estimate requires measurement IDs",
        ));
    }
    Ok(())
}

fn validate_model_engine_capacities(plan: &DeploymentPlan) -> Result<(), ValidationError> {
    for (path, value) in [
        (
            "model.architecture.parameter_count",
            plan.model.architecture.parameter_count,
        ),
        (
            "model.architecture.hidden_size",
            u64::from(plan.model.architecture.hidden_size),
        ),
        (
            "model.architecture.num_layers",
            u64::from(plan.model.architecture.num_layers),
        ),
        (
            "model.architecture.num_attention_heads",
            u64::from(plan.model.architecture.num_attention_heads),
        ),
        (
            "model.architecture.num_key_value_heads",
            u64::from(plan.model.architecture.num_key_value_heads),
        ),
        (
            "model.architecture.vocabulary_size",
            u64::from(plan.model.architecture.vocabulary_size),
        ),
        (
            "model.maximum_sequence_length",
            u64::from(plan.model.maximum_sequence_length),
        ),
        (
            "engine.tensor_parallelism",
            u64::from(plan.engine.tensor_parallelism),
        ),
        (
            "engine.pipeline_parallelism",
            u64::from(plan.engine.pipeline_parallelism),
        ),
        (
            "engine.maximum_batched_tokens",
            u64::from(plan.engine.maximum_batched_tokens),
        ),
        (
            "engine.maximum_active_sequences",
            u64::from(plan.engine.maximum_active_sequences),
        ),
        (
            "engine.chunked_prefill.chunk_tokens",
            u64::from(plan.engine.chunked_prefill.chunk_tokens),
        ),
    ] {
        positive_integer(path, value)?;
    }
    if plan.engine.speculative_decoding.enabled
        && plan.engine.speculative_decoding.draft_model_id.is_none()
    {
        return Err(ValidationError::new(
            "engine.speculative_decoding.draft_model_id",
            "required when speculative decoding is enabled",
        ));
    }
    if !plan.engine.compilation.cuda_graphs && !plan.engine.compilation.graph_batch_sizes.is_empty()
    {
        return Err(ValidationError::new(
            "engine.compilation.graph_batch_sizes",
            "requires CUDA graphs",
        ));
    }
    Ok(())
}

fn validate_plan_header_and_model(plan: &DeploymentPlan) -> Result<(), ValidationError> {
    if plan.schema_version != SCHEMA_VERSION {
        return Err(ValidationError::new(
            "schema_version",
            "expected stable version 1.0.0",
        ));
    }
    if plan.api_version != API_VERSION || plan.kind != "DeploymentPlan" {
        return Err(ValidationError::new(
            "kind",
            "invalid API version or document kind",
        ));
    }
    nonempty("metadata.name", &plan.metadata.name)?;
    nonempty("metadata.uid", &plan.metadata.uid)?;
    if plan.metadata.generation == 0 {
        return Err(ValidationError::new(
            "metadata.generation",
            "must be positive",
        ));
    }
    plan.model.checksum.validate_at("model.checksum")?;
    nonempty("model.model_id", &plan.model.model_id)?;
    semver("engine.version", &plan.engine.version)?;
    semver(
        "provenance.compiler_version",
        &plan.provenance.compiler_version,
    )?;
    if plan.model.allowed_precisions.is_empty()
        || !plan
            .model
            .allowed_precisions
            .contains(&plan.model.minimum_precision)
    {
        return Err(ValidationError::new(
            "model.allowed_precisions",
            "must include minimum_precision",
        ));
    }
    if plan
        .model
        .allowed_precisions
        .iter()
        .collect::<BTreeSet<_>>()
        .len()
        != plan.model.allowed_precisions.len()
    {
        return Err(ValidationError::new(
            "model.allowed_precisions",
            "must not contain duplicates",
        ));
    }
    if !plan.model.allowed_precisions.contains(&plan.engine.dtype) {
        return Err(ValidationError::new("engine.dtype", "forbidden by model"));
    }
    if plan.model.architecture.num_key_value_heads > plan.model.architecture.num_attention_heads {
        return Err(ValidationError::new(
            "model.architecture.num_key_value_heads",
            "cannot exceed attention heads",
        ));
    }
    validate_model_engine_capacities(plan)
}

fn validate_distribution(
    path: &str,
    distribution: &DistributionSpec,
) -> Result<(), ValidationError> {
    positive_integer(&format!("{path}.maximum"), u64::from(distribution.maximum))?;
    if distribution.minimum > distribution.maximum {
        return Err(ValidationError::new(path, "minimum exceeds maximum"));
    }
    match distribution.kind {
        DistributionKind::Fixed => {
            if !distribution
                .fixed_value
                .is_some_and(|value| (distribution.minimum..=distribution.maximum).contains(&value))
            {
                return Err(ValidationError::new(path, "fixed value is outside bounds"));
            }
        }
        DistributionKind::Empirical => {
            if distribution.empirical.is_empty()
                || distribution.empirical.iter().any(|item| {
                    !(distribution.minimum..=distribution.maximum).contains(&item.value)
                        || !item.weight.is_finite()
                        || item.weight <= 0.0
                })
            {
                return Err(ValidationError::new(path, "invalid empirical distribution"));
            }
        }
        DistributionKind::Lognormal => {
            if !distribution.log_mean.is_some_and(f64::is_finite)
                || !distribution
                    .log_stddev
                    .is_some_and(|value| value.is_finite() && value > 0.0)
            {
                return Err(ValidationError::new(path, "invalid lognormal parameters"));
            }
        }
    }
    Ok(())
}

fn validate_workload(workload: &WorkloadSpec) -> Result<(), ValidationError> {
    if workload.request_classes.is_empty() {
        return Err(ValidationError::new(
            "workload.request_classes",
            "must not be empty",
        ));
    }
    let mut class_names = BTreeSet::new();
    let class_weight = workload
        .request_classes
        .iter()
        .try_fold(0.0, |total, request_class| {
            nonempty("workload.request_classes.name", &request_class.name)?;
            if !class_names.insert(&request_class.name) {
                return Err(ValidationError::new(
                    "workload.request_classes",
                    "class names must be unique",
                ));
            }
            positive("workload.request_classes.weight", request_class.weight)?;
            Ok(total + request_class.weight)
        })?;
    if (class_weight - 1.0_f64).abs() > 1e-6 {
        return Err(ValidationError::new(
            "workload.request_classes",
            "weights must sum to one",
        ));
    }
    match workload.arrival_process.kind {
        ArrivalKind::Trace if workload.arrival_process.trace_uri.is_none() => {
            return Err(ValidationError::new(
                "workload.arrival_process.trace_uri",
                "required for trace arrivals",
            ));
        }
        ArrivalKind::Trace => {}
        ArrivalKind::Bursty => {
            positive(
                "workload.arrival_process.rate_per_second",
                workload.arrival_process.rate_per_second.unwrap_or(f64::NAN),
            )?;
            positive(
                "workload.arrival_process.burst_factor",
                workload.arrival_process.burst_factor.unwrap_or(f64::NAN),
            )?;
        }
        ArrivalKind::Poisson | ArrivalKind::Deterministic => positive(
            "workload.arrival_process.rate_per_second",
            workload.arrival_process.rate_per_second.unwrap_or(f64::NAN),
        )?,
    }
    validate_distribution("workload.prompt_tokens", &workload.prompt_tokens)?;
    validate_distribution("workload.output_tokens", &workload.output_tokens)?;
    positive("workload.duration_seconds", workload.duration_seconds)
}

fn validate_plan_hardware_and_workload(plan: &DeploymentPlan) -> Result<(), ValidationError> {
    plan.hardware
        .fingerprint
        .validate_at("hardware.fingerprint")?;
    if plan.hardware.hourly_price_usd < 0.0 || !plan.hardware.hourly_price_usd.is_finite() {
        return Err(ValidationError::new(
            "hardware.hourly_price_usd",
            "must be nonnegative",
        ));
    }
    for (path, value) in [
        (
            "hardware.system_memory_bytes",
            plan.hardware.system_memory_bytes,
        ),
        (
            "hardware.cpu.physical_cores",
            u64::from(plan.hardware.cpu.physical_cores),
        ),
        (
            "hardware.cpu.logical_cores",
            u64::from(plan.hardware.cpu.logical_cores),
        ),
        (
            "hardware.cpu.numa_nodes",
            u64::from(plan.hardware.cpu.numa_nodes),
        ),
    ] {
        positive_integer(path, value)?;
    }
    if plan.hardware.cpu.logical_cores < plan.hardware.cpu.physical_cores {
        return Err(ValidationError::new(
            "hardware.cpu.logical_cores",
            "cannot be below physical cores",
        ));
    }
    let gpu_indices: BTreeSet<_> = plan.hardware.gpus.iter().map(|gpu| gpu.index).collect();
    if usize::try_from(plan.hardware.gpu_count).ok() != Some(plan.hardware.gpus.len()) {
        return Err(ValidationError::new(
            "hardware.gpu_count",
            "must equal the number of GPU specifications",
        ));
    }
    if gpu_indices.len() != plan.hardware.gpus.len() {
        return Err(ValidationError::new(
            "hardware.gpus",
            "GPU indices must be unique",
        ));
    }
    for link in &plan.hardware.topology {
        if link.source_gpu == link.target_gpu
            || !gpu_indices.contains(&link.source_gpu)
            || !gpu_indices.contains(&link.target_gpu)
        {
            return Err(ValidationError::new(
                "hardware.topology",
                "invalid GPU endpoints",
            ));
        }
    }
    for gpu in &plan.hardware.gpus {
        positive_integer("hardware.gpus.vram_bytes", gpu.vram_bytes)?;
    }
    if let Some(limit) = plan.hardware.container_memory_limit_bytes {
        positive_integer("hardware.container_memory_limit_bytes", limit)?;
    }
    validate_workload(&plan.workload)
}

fn validate_policy_metrics(plan: &DeploymentPlan) -> Result<(), ValidationError> {
    probability(
        "autoscaling.target_utilization",
        plan.autoscaling.target_utilization,
    )?;
    probability("autoscaling.safety_margin", plan.autoscaling.safety_margin)?;
    probability(
        "rollback.availability_floor",
        plan.rollback.availability_floor,
    )?;
    probability("budget.reserve_fraction", plan.budget.reserve_fraction)?;
    nonnegative(
        "budget.profiling_budget_usd",
        plan.budget.profiling_budget_usd,
    )?;
    positive(
        "admission.maximum_queue_time_ms",
        plan.admission.maximum_queue_time_ms,
    )?;
    nonnegative(
        "batching.maximum_batch_delay_ms",
        plan.batching.maximum_batch_delay_ms,
    )?;
    positive(
        "autoscaling.control_interval_seconds",
        plan.autoscaling.control_interval_seconds,
    )?;
    positive(
        "cold_start.readiness_timeout_seconds",
        plan.cold_start.readiness_timeout_seconds,
    )?;
    positive(
        "cold_start.predicted_p95_startup_ms",
        plan.cold_start.predicted_p95_startup_ms,
    )?;
    if plan.slo.objective_weights.cost < 0.0
        || plan.slo.objective_weights.latency < 0.0
        || plan.slo.objective_weights.goodput < 0.0
        || plan.slo.objective_weights.availability < 0.0
        || plan.slo.objective_weights.cost
            + plan.slo.objective_weights.latency
            + plan.slo.objective_weights.goodput
            + plan.slo.objective_weights.availability
            <= 0.0
    {
        return Err(ValidationError::new(
            "slo.objective_weights",
            "weights must be nonnegative with a positive sum",
        ));
    }
    for (name, constraints) in [
        ("slo.ttft", &plan.slo.ttft),
        ("slo.inter_token_latency", &plan.slo.inter_token_latency),
        ("slo.end_to_end_latency", &plan.slo.end_to_end_latency),
    ] {
        let mut percentiles = BTreeSet::new();
        for constraint in constraints {
            let valid = constraint.percentile.is_finite()
                && 0.0 < constraint.percentile
                && constraint.percentile <= 100.0
                && percentiles.insert(constraint.percentile.to_bits());
            if !valid {
                return Err(ValidationError::new(
                    name,
                    "invalid or duplicate percentile",
                ));
            }
            positive(name, constraint.maximum_ms)?;
        }
    }
    if let Some(value) = plan.slo.minimum_availability {
        probability("slo.minimum_availability", value)?;
    }
    Ok(())
}

fn validate_plan_policies(plan: &DeploymentPlan) -> Result<(), ValidationError> {
    for (path, value) in [
        (
            "replica_topology.minimum_replicas",
            plan.replica_topology.minimum_replicas,
        ),
        (
            "replica_topology.maximum_replicas",
            plan.replica_topology.maximum_replicas,
        ),
        (
            "replica_topology.initial_replicas",
            plan.replica_topology.initial_replicas,
        ),
        ("admission.queue_capacity", plan.admission.queue_capacity),
        (
            "batching.maximum_active_sequences",
            plan.batching.maximum_active_sequences,
        ),
        (
            "batching.maximum_batched_tokens",
            plan.batching.maximum_batched_tokens,
        ),
    ] {
        positive_integer(path, u64::from(value))?;
    }
    if !(plan.replica_topology.minimum_replicas <= plan.replica_topology.initial_replicas
        && plan.replica_topology.initial_replicas <= plan.replica_topology.maximum_replicas)
    {
        return Err(ValidationError::new(
            "replica_topology",
            "invalid replica range",
        ));
    }
    if plan.replica_topology.regions.is_empty() {
        return Err(ValidationError::new(
            "replica_topology.regions",
            "must not be empty",
        ));
    }
    if plan.replica_topology.tensor_parallelism_per_replica != plan.engine.tensor_parallelism {
        return Err(ValidationError::new(
            "replica_topology.tensor_parallelism_per_replica",
            "must equal engine tensor_parallelism",
        ));
    }
    let mut route_variants = BTreeSet::new();
    let route_weight = plan.routing.targets.iter().try_fold(0.0, |total, target| {
        nonempty("routing.targets.variant", &target.variant)?;
        if !route_variants.insert(&target.variant) {
            return Err(ValidationError::new(
                "routing.targets",
                "target variants must be unique",
            ));
        }
        positive("routing.targets.weight", target.weight)?;
        Ok(total + target.weight)
    })?;
    if plan.routing.targets.is_empty() || (route_weight - 1.0_f64).abs() > 1e-6 {
        return Err(ValidationError::new(
            "routing.targets",
            "weights must sum to one",
        ));
    }
    if plan.engine.maximum_active_sequences != plan.batching.maximum_active_sequences
        || plan.engine.maximum_batched_tokens != plan.batching.maximum_batched_tokens
    {
        return Err(ValidationError::new(
            "batching",
            "must match engine batching limits",
        ));
    }
    if plan.cold_start.minimum_warm_replicas > plan.replica_topology.maximum_replicas {
        return Err(ValidationError::new(
            "cold_start",
            "warm replicas exceed maximum",
        ));
    }
    validate_policy_metrics(plan)
}

impl Validate for DeploymentPlan {
    fn validate(&self) -> Result<(), ValidationError> {
        validate_plan_header_and_model(self)?;
        validate_plan_hardware_and_workload(self)?;
        validate_plan_policies(self)?;
        for (name, metric) in &self.predicted_metrics {
            validate_metric(&format!("predicted_metrics.{name}"), metric)?;
        }
        self.model.extensions.validate()?;
        self.engine.extensions.validate()?;
        self.hardware.extensions.validate()?;
        self.workload.extensions.validate()?;
        self.extensions.validate()?;
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EnvironmentManifest {
    pub os: String,
    pub kernel: String,
    pub architecture: String,
    pub hostname_hash: ArtifactDigest,
    pub container_image: Option<String>,
    pub python_version: Option<String>,
    pub rust_version: Option<String>,
    pub package_versions: BTreeMap<String, String>,
    #[serde(default)]
    pub environment_allowlist: BTreeMap<String, String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MeasurementKind {
    Hardware,
    Startup,
    Prefill,
    Decode,
    Load,
    Fault,
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MeasurementRef {
    pub measurement_id: String,
    pub kind: MeasurementKind,
    pub uri: String,
    pub digest: ArtifactDigest,
    pub sample_count: u64,
    pub warmup_count: u64,
    pub started_at: DateTime<Utc>,
    pub completed_at: DateTime<Utc>,
    pub hardware_fingerprint: ArtifactDigest,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationSplit {
    Train,
    Validation,
    Test,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationMetricKind {
    Mae,
    Mape,
    Rmse,
    Coverage,
    IntervalWidth,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationMetric {
    pub model_name: String,
    pub split: CalibrationSplit,
    pub metric: CalibrationMetricKind,
    pub value: f64,
    pub sample_count: u64,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Fidelity {
    Static,
    Simulated,
    Measured,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OptimizerAction {
    Evaluate,
    Promote,
    Reject,
    Select,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OptimizerDecision {
    pub sequence: u64,
    pub candidate_id: String,
    pub fidelity: Fidelity,
    pub decision: OptimizerAction,
    pub reason_code: String,
    pub predicted_objective: Option<MetricEstimate>,
    pub cost_usd: f64,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RejectionStage {
    Feasibility,
    Simulation,
    Measurement,
    Selection,
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RejectedCandidate {
    pub candidate_id: String,
    pub stage: RejectionStage,
    pub reason_code: String,
    pub explanation: String,
    #[serde(default)]
    pub violated_constraints: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkResult {
    pub benchmark_id: String,
    pub command: Vec<String>,
    pub raw_result_uri: String,
    pub raw_result_digest: ArtifactDigest,
    pub seed: u64,
    pub started_at: DateTime<Utc>,
    pub completed_at: DateTime<Utc>,
    pub metrics: BTreeMap<String, MetricEstimate>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceBundle {
    pub schema_version: String,
    pub api_version: String,
    pub kind: String,
    pub metadata: DocumentMetadata,
    pub plan_digest: ArtifactDigest,
    pub environment: EnvironmentManifest,
    pub model_assumptions: Vec<String>,
    pub measurements: Vec<MeasurementRef>,
    pub calibration_metrics: Vec<CalibrationMetric>,
    pub optimizer_history: Vec<OptimizerDecision>,
    pub rejected_candidates: Vec<RejectedCandidate>,
    pub benchmark_results: Vec<BenchmarkResult>,
    pub artifact_hashes: BTreeMap<String, ArtifactDigest>,
    pub git_commit: String,
    pub generated_at: DateTime<Utc>,
    #[serde(default)]
    pub extensions: Extensions,
}

fn validate_measurements(measurements: &[MeasurementRef]) -> Result<(), ValidationError> {
    let mut ids = BTreeSet::new();
    for measurement in measurements {
        nonempty("measurements.measurement_id", &measurement.measurement_id)?;
        if !ids.insert(&measurement.measurement_id) {
            return Err(ValidationError::new(
                "measurements",
                "measurement IDs must be unique",
            ));
        }
        if measurement.completed_at < measurement.started_at {
            return Err(ValidationError::new(
                "measurements",
                "completion precedes start",
            ));
        }
        measurement.digest.validate_at("measurements.digest")?;
        measurement
            .hardware_fingerprint
            .validate_at("measurements.hardware_fingerprint")?;
    }
    Ok(())
}

fn validate_optimizer_history(history: &[OptimizerDecision]) -> Result<(), ValidationError> {
    if !history
        .windows(2)
        .all(|items| items[0].sequence <= items[1].sequence)
    {
        return Err(ValidationError::new(
            "optimizer_history",
            "must be ordered by sequence",
        ));
    }
    for decision in history {
        nonempty("optimizer_history.candidate_id", &decision.candidate_id)?;
        nonempty("optimizer_history.reason_code", &decision.reason_code)?;
        nonnegative("optimizer_history.cost_usd", decision.cost_usd)?;
        if let Some(metric) = &decision.predicted_objective {
            validate_metric("optimizer_history.predicted_objective", metric)?;
        }
    }
    Ok(())
}

fn validate_benchmarks(benchmarks: &[BenchmarkResult]) -> Result<(), ValidationError> {
    for benchmark in benchmarks {
        if benchmark.command.is_empty() || benchmark.completed_at < benchmark.started_at {
            return Err(ValidationError::new(
                "benchmark_results",
                "invalid command or times",
            ));
        }
        benchmark
            .raw_result_digest
            .validate_at("benchmark_results.raw_result_digest")?;
        for command_part in &benchmark.command {
            nonempty("benchmark_results.command", command_part)?;
        }
        for (name, metric) in &benchmark.metrics {
            validate_metric(&format!("benchmark_results.metrics.{name}"), metric)?;
        }
    }
    Ok(())
}

impl Validate for EvidenceBundle {
    fn validate(&self) -> Result<(), ValidationError> {
        if self.schema_version != SCHEMA_VERSION
            || self.api_version != API_VERSION
            || self.kind != "EvidenceBundle"
        {
            return Err(ValidationError::new(
                "kind",
                "invalid version or document kind",
            ));
        }
        self.plan_digest.validate_at("plan_digest")?;
        self.environment
            .hostname_hash
            .validate_at("environment.hostname_hash")?;
        nonempty("metadata.name", &self.metadata.name)?;
        nonempty("metadata.uid", &self.metadata.uid)?;
        positive_integer("metadata.generation", self.metadata.generation)?;
        nonempty("git_commit", &self.git_commit)?;
        validate_measurements(&self.measurements)?;
        validate_optimizer_history(&self.optimizer_history)?;
        for calibration in &self.calibration_metrics {
            nonempty("calibration_metrics.model_name", &calibration.model_name)?;
            positive_integer("calibration_metrics.sample_count", calibration.sample_count)?;
            nonnegative("calibration_metrics.value", calibration.value)?;
            if calibration.metric == CalibrationMetricKind::Coverage && calibration.value > 1.0 {
                return Err(ValidationError::new(
                    "calibration_metrics.value",
                    "coverage cannot exceed one",
                ));
            }
        }
        for rejected in &self.rejected_candidates {
            nonempty("rejected_candidates.candidate_id", &rejected.candidate_id)?;
            nonempty("rejected_candidates.reason_code", &rejected.reason_code)?;
            nonempty("rejected_candidates.explanation", &rejected.explanation)?;
        }
        validate_benchmarks(&self.benchmark_results)?;
        for (path, digest) in &self.artifact_hashes {
            digest.validate_at(&format!("artifact_hashes.{path}"))?;
        }
        self.extensions.validate()?;
        Ok(())
    }
}
