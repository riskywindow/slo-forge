use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

fn schema_version() -> String {
    crate::FABRIC_SIM_SCHEMA_VERSION.to_owned()
}

const fn default_max_events() -> usize {
    5_000_000
}

const fn default_max_operations() -> usize {
    100_000
}

const fn one() -> f64 {
    1.0
}

/// Whether a calibration point came from hardware or a deterministic fixture.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProvenanceKind {
    Measured,
    Synthetic,
    Analytical,
}

/// Traceable origin for every service curve.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationProvenance {
    pub kind: ProvenanceKind,
    pub artifact_uri: String,
    pub artifact_sha256: String,
    pub environment_fingerprint: String,
    pub collected_at: String,
}

/// One monotonically increasing point on a latency/bandwidth curve.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CurvePoint {
    pub message_bytes: u64,
    pub latency_us: f64,
    pub bandwidth_gbps: f64,
    #[serde(default)]
    pub uncertainty_fraction: f64,
}

/// Piecewise-linear calibrated curve. Durations include latency plus serialization.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServiceCurve {
    pub id: String,
    pub points: Vec<CurvePoint>,
    pub provenance: CalibrationProvenance,
}

/// Physical capacity scheduled by the flow-level simulator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceKind {
    CpuCoreGroup,
    NumaMemory,
    GpuCompute,
    GpuHbm,
    GpuCopyEngine,
    Nvlink,
    Nvswitch,
    Pcie,
    NicQueue,
    NetworkRail,
    StoragePath,
}

/// Exclusive resources serialize work. Fair-share resources admit bounded
/// concurrent flows and divide calibrated capacity proportionally.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SchedulingMode {
    Exclusive,
    FairShare,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhysicalResource {
    pub id: String,
    pub kind: ResourceKind,
    pub scheduling: SchedulingMode,
    #[serde(default = "one")]
    pub capacity_units: f64,
    #[serde(default = "one_usize")]
    pub max_concurrency: usize,
    pub curve: ServiceCurve,
    #[serde(default)]
    pub sharing_group: Option<String>,
    #[serde(default)]
    pub hourly_cost_usd: f64,
}

const fn one_usize() -> usize {
    1
}

/// A shared bottleneck spanning multiple resources, such as a `PCIe` switch or rail.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SharingGroup {
    pub id: String,
    pub capacity_units: f64,
    pub max_concurrency: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceDemand {
    pub resource_id: String,
    #[serde(default = "one")]
    pub units: f64,
}

/// Dependency-graph operation. Communication kinds carry byte volume; compute and
/// startup carry calibrated base duration. All durations are microseconds.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum OperationKind {
    CpuLaunch {
        duration_us: f64,
    },
    GpuCompute {
        duration_us: f64,
    },
    HbmAccess {
        bytes: u64,
    },
    PointToPoint {
        bytes: u64,
    },
    Collective {
        collective_id: String,
        bytes: u64,
        algorithm: String,
        participating_ranks: Vec<String>,
    },
    ExpertDispatch {
        bytes: u64,
        experts: u32,
    },
    ExpertCombine {
        bytes: u64,
        experts: u32,
    },
    KvTransfer {
        bytes: u64,
        chunks: u32,
    },
    StorageFetch {
        bytes: u64,
    },
    Startup {
        duration_us: f64,
    },
    Synchronization,
}

impl OperationKind {
    #[must_use]
    pub const fn bytes(&self) -> u64 {
        match self {
            Self::HbmAccess { bytes }
            | Self::PointToPoint { bytes }
            | Self::Collective { bytes, .. }
            | Self::ExpertDispatch { bytes, .. }
            | Self::ExpertCombine { bytes, .. }
            | Self::KvTransfer { bytes, .. }
            | Self::StorageFetch { bytes } => *bytes,
            Self::CpuLaunch { .. }
            | Self::GpuCompute { .. }
            | Self::Startup { .. }
            | Self::Synchronization => 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhysicalOperation {
    pub id: String,
    pub kind: OperationKind,
    #[serde(default)]
    pub rank_ids: Vec<String>,
    #[serde(default)]
    pub dependencies: Vec<String>,
    #[serde(default)]
    pub demands: Vec<ResourceDemand>,
    #[serde(default)]
    pub earliest_start_us: f64,
    #[serde(default)]
    pub uncertainty_fraction: f64,
    #[serde(default)]
    pub request_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum FaultEffect {
    ResourceRate {
        resource_id: String,
        multiplier: f64,
    },
    ResourceUnavailable {
        resource_id: String,
    },
    RankSlowdown {
        rank_id: String,
        multiplier: f64,
    },
    CollectiveDelay {
        collective_id: String,
        multiplier: f64,
    },
}

/// A half-open interval `[start_us, end_us)`. Omitting the end makes a fault permanent.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TimedFault {
    pub id: String,
    pub start_us: f64,
    #[serde(default)]
    pub end_us: Option<f64>,
    pub effect: FaultEffect,
    pub ground_truth_label: String,
}

/// Counterfactual transformations are applied to a copy of the input before execution.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum CounterfactualModifier {
    RemoveFault {
        fault_id: String,
    },
    ScaleResourceCurve {
        resource_id: String,
        latency_multiplier: f64,
        bandwidth_multiplier: f64,
    },
    ScaleRank {
        rank_id: String,
        duration_multiplier: f64,
    },
    ReplaceResource {
        from_resource_id: String,
        to_resource_id: String,
    },
}

/// Stable JSON subprocess request.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FabricSimulationRequest {
    #[serde(default = "schema_version")]
    pub schema_version: String,
    pub seed: u64,
    pub resources: Vec<PhysicalResource>,
    #[serde(default)]
    pub sharing_groups: Vec<SharingGroup>,
    pub operations: Vec<PhysicalOperation>,
    #[serde(default)]
    pub faults: Vec<TimedFault>,
    #[serde(default)]
    pub counterfactuals: Vec<CounterfactualModifier>,
    #[serde(default = "default_max_events")]
    pub max_events: usize,
    #[serde(default = "default_max_operations")]
    pub max_operations: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperationStatus {
    Completed,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperationOutcome {
    pub operation_id: String,
    pub status: OperationStatus,
    pub start_us: f64,
    pub end_us: f64,
    pub duration_us: f64,
    pub base_duration_us: f64,
    pub wait_us: f64,
    pub transferred_bytes: u64,
    pub uncertainty_us: f64,
    pub rank_ids: Vec<String>,
    pub resource_ids: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceMetrics {
    pub resource_id: String,
    pub busy_time_us: f64,
    pub utilization: f64,
    pub transferred_bytes: u64,
    pub max_concurrent: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FabricSimulationMetrics {
    pub operation_count: usize,
    pub makespan_us: f64,
    pub total_work_us: f64,
    pub total_transferred_bytes: u64,
    pub cost_usd: f64,
    pub processed_events: usize,
    pub overlap_efficiency: f64,
    pub predicted_lower_us: f64,
    pub predicted_upper_us: f64,
    pub resources: Vec<ResourceMetrics>,
}

/// Chrome/Perfetto complete event (`ts` and `dur` are microseconds).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChromeTraceEvent {
    pub name: String,
    pub cat: String,
    pub ph: String,
    pub ts: f64,
    pub dur: f64,
    pub pid: u32,
    pub tid: String,
    pub args: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SimulationProvenance {
    pub simulator_version: String,
    pub input_sha256: String,
    pub seed: u64,
    pub calibration_artifacts: Vec<String>,
    pub calibration_kinds: Vec<ProvenanceKind>,
    pub counterfactual_count: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FabricSimulationOutput {
    pub schema_version: String,
    pub provenance: SimulationProvenance,
    pub metrics: FabricSimulationMetrics,
    pub operations: Vec<OperationOutcome>,
    pub trace_events: Vec<ChromeTraceEvent>,
    pub applied_faults: Vec<String>,
    pub applied_counterfactuals: Vec<CounterfactualModifier>,
}
