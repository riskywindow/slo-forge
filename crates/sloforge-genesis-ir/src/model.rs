//! Canonical Genesis IR model shared with Python over versioned JSON.

use std::collections::{BTreeMap, BTreeSet};

use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

use crate::{Validate, ValidationError};

pub const SCHEMA_VERSION: &str = "1.0.0";
pub const API_VERSION: &str = "sloforge.io/genesis/v1";

fn nonempty(path: &str, value: &str) -> Result<(), ValidationError> {
    if value.trim().is_empty() {
        Err(ValidationError::new(path, "must not be empty"))
    } else {
        Ok(())
    }
}

fn nonempty_values<'a>(
    path: &str,
    values: impl IntoIterator<Item = &'a String>,
) -> Result<(), ValidationError> {
    for (index, value) in values.into_iter().enumerate() {
        nonempty(&format!("{path}[{index}]"), value)?;
    }
    Ok(())
}

fn finite_nonnegative(path: &str, value: f64) -> Result<(), ValidationError> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(ValidationError::new(
            path,
            "must be finite and non-negative",
        ))
    }
}

fn finite_positive(path: &str, value: f64) -> Result<(), ValidationError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(ValidationError::new(path, "must be finite and positive"))
    }
}

fn extension_key_is_valid(key: &str) -> bool {
    let Some((namespace, name)) = key.split_once('/') else {
        return false;
    };
    let mut namespace_chars = namespace.chars();
    let namespace_valid = namespace_chars
        .next()
        .is_some_and(|ch| ch.is_ascii_lowercase())
        && namespace_chars
            .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || matches!(ch, '.' | '-'));
    let mut name_chars = name.chars();
    namespace_valid
        && name_chars.next().is_some_and(|ch| ch.is_ascii_alphabetic())
        && name_chars.all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.' | '-'))
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, JsonSchema)]
#[serde(transparent)]
pub struct Extensions(pub BTreeMap<String, Value>);

impl<'de> Deserialize<'de> for Extensions {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let values = BTreeMap::<String, Value>::deserialize(deserializer)?;
        if let Some(key) = values.keys().find(|key| !extension_key_is_valid(key)) {
            return Err(serde::de::Error::custom(format!(
                "extension key {key:?} must be namespace-qualified"
            )));
        }
        Ok(Self(values))
    }
}

macro_rules! string_enum {
    ($name:ident { $($variant:ident),+ $(,)? }) => {
        #[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
        #[serde(rename_all = "snake_case")]
        pub enum $name { $($variant),+ }
    };
}

string_enum!(SemanticCategory {
    Exact,
    Approximate,
    Policy,
    Resource,
    Implementation,
    Experimental
});
#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
pub enum VerificationLevel {
    #[serde(rename = "level_0_build")]
    Level0Build,
    #[serde(rename = "level_1_differential")]
    Level1Differential,
    #[serde(rename = "level_2_property")]
    Level2Property,
    #[serde(rename = "level_3_bounded_exhaustive")]
    Level3BoundedExhaustive,
    #[serde(rename = "level_4_solver_backed")]
    Level4SolverBacked,
    #[serde(rename = "level_5_hardware_operational")]
    Level5HardwareOperational,
}
string_enum!(HotSwapCategory {
    PolicyOnly,
    RequestBoundary,
    WorkerRestart,
    NewReplica,
    StateCompatibleMigration,
    StateConversionMigration,
    FullDeploymentRebuild,
    OperatorRequired,
});
string_enum!(LineageRelation {
    Parent,
    DerivedFrom,
    Reused,
    ConstrainedBy,
    InvalidatedBy
});
string_enum!(WorkflowStepKind {
    ModelInvocation,
    ToolCall,
    VerificationPass,
    Branch,
    Loop
});
string_enum!(CancellationBehavior {
    Immediate,
    SafePoint,
    Drain,
    Ignore
});
string_enum!(AdmissionControl {
    BoundedFifo,
    DeadlineAware,
    TokenBudget,
    Reject
});
string_enum!(QueueDiscipline {
    Fifo,
    EarliestDeadline,
    ShortestRemaining,
    WeightedFair
});
string_enum!(RoutingPolicy {
    RoundRobin,
    LeastLoaded,
    CacheAffinity,
    WorkflowAffinity
});
string_enum!(StreamingSemantics {
    TokenCommit,
    ChunkCommit,
    AtomicResponse
});
string_enum!(FallbackBehavior {
    Reject,
    LowerQualityTier,
    ReferenceRuntime,
    RetryCompatible
});
string_enum!(ServingTopology {
    Aggregated,
    Disaggregated
});
string_enum!(PrefillPolicy {
    WholePrompt,
    Chunked,
    Incremental
});
string_enum!(DecodeScheduling {
    RoundRobin,
    DeadlineAware,
    SloSlack,
    WorkflowAware
});
string_enum!(StateKind {
    Autoregressive,
    Kv,
    Recurrent,
    Convolutional,
    Speculative,
    Custom,
    Tool,
    Workflow
});
string_enum!(StateOwnership {
    Request,
    Session,
    Worker,
    Replica,
    SharedReplicated
});
string_enum!(StateLayout {
    Contiguous,
    Paged,
    Interleaved,
    Sharded
});
string_enum!(Precision {
    Bool,
    Float64,
    Float32,
    Bfloat16,
    Float16,
    Fp8,
    Int64,
    Int32,
    Int16,
    Int8,
    Uint8,
    Int4
});
string_enum!(RetentionPolicy {
    RequestLifetime,
    Session,
    Lru,
    DeadlineAware
});
string_enum!(ConsistencyModel {
    Exclusive,
    Versioned,
    ReadOnlyReplicated
});
string_enum!(OffloadTier {
    None,
    Host,
    Peer,
    Remote
});
string_enum!(CollectiveKind {
    AllReduce,
    AllGather,
    ReduceScatter,
    AllToAll,
    SendRecv
});
string_enum!(PrefillDecodeTransfer {
    None,
    Host,
    Peer,
    Rdma
});
string_enum!(KernelBackend {
    Pytorch,
    Triton,
    Cuda,
    Cute,
    Cpp
});
string_enum!(TransitionPoint {
    Immediate,
    TokenBoundary,
    RequestBoundary,
    Drained
});
string_enum!(StateTransfer {
    None,
    Copy,
    Move,
    Recompute
});
string_enum!(ActiveStreamBehavior {
    Preserve,
    Drain,
    Restart,
    Reject
});
string_enum!(TransformationFamily {
    AlgebraicRewrite,
    TensorDecomposition,
    OperatorFusion,
    LayoutTransformation,
    PrecisionTransformation,
    QuantizationTransformation,
    SchedulerTransformation,
    BatchingTransformation,
    CachePolicyTransformation,
    StateLayoutTransformation,
    DistributedPlanTransformation,
    CommunicationTransformation,
    KernelTransformation,
    WorkflowTransformation,
    RecoveryTransformation,
    RuntimeCodePatch,
});
string_enum!(TransformationDesignation {
    SemanticsPreserving,
    ApproximateWithinQualityBudget,
    Policy,
    ResourceOnly,
    RuntimeImplementation,
    ExperimentalOperatorReview,
});
string_enum!(CounterexampleScope {
    CandidateSpecific,
    TransformationFamily,
    HardwareSpecific,
    DependencyVersion,
    UniversalPreconditionViolation,
});
string_enum!(GenomeRegion {
    Workflow,
    Request,
    Serving,
    State,
    Distributed,
    Tensor,
    Kernel,
    Recovery
});
string_enum!(RequestEventAction {
    Admit,
    Schedule,
    Prefill,
    Decode,
    Emit,
    Cancel,
    Disconnect,
    Fail,
    Retry
});

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CandidateState {
    Proposed,
    StaticallyValid,
    Compiled,
    ReferenceTested,
    PropertyTested,
    ModelChecked,
    Simulated,
    HardwareBenchmarked,
    ShadowValidated,
    CanaryValidated,
    CapsuleAccepted,
    Promoted,
    StaticRejected,
    CompileRejected,
    SemanticRejected,
    QualityRejected,
    ResourceRejected,
    ModelCheckRejected,
    PerformanceRejected,
    ShadowRejected,
    CanaryRejected,
    SandboxViolation,
    Superseded,
}

impl CandidateState {
    const fn success_rank(self) -> Option<u64> {
        match self {
            Self::Proposed => Some(0),
            Self::StaticallyValid => Some(1),
            Self::Compiled => Some(2),
            Self::ReferenceTested => Some(3),
            Self::PropertyTested => Some(4),
            Self::ModelChecked => Some(5),
            Self::Simulated => Some(6),
            Self::HardwareBenchmarked => Some(7),
            Self::ShadowValidated => Some(8),
            Self::CanaryValidated => Some(9),
            Self::CapsuleAccepted => Some(10),
            Self::Promoted => Some(11),
            Self::StaticRejected
            | Self::CompileRejected
            | Self::SemanticRejected
            | Self::QualityRejected
            | Self::ResourceRejected
            | Self::ModelCheckRejected
            | Self::PerformanceRejected
            | Self::ShadowRejected
            | Self::CanaryRejected
            | Self::SandboxViolation
            | Self::Superseded => None,
        }
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
        if self.value.len() == 64
            && self
                .value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            Ok(())
        } else {
            Err(ValidationError::new(path, "invalid SHA-256 digest"))
        }
    }
}

impl EvidenceReference {
    fn validate_at(&self, path: &str) -> Result<(), ValidationError> {
        nonempty(&format!("{path}.evidence_id"), &self.evidence_id)?;
        nonempty(&format!("{path}.artifact_uri"), &self.artifact_uri)?;
        self.digest.validate_at(&format!("{path}.digest"))?;
        nonempty_values(&format!("{path}.claim_ids"), &self.claim_ids)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceReference {
    pub evidence_id: String,
    pub artifact_uri: String,
    pub digest: ArtifactDigest,
    pub claim_ids: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LineageReference {
    pub lineage_id: String,
    pub relation: LineageRelation,
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProofObligation {
    pub obligation_id: String,
    pub property: String,
    pub minimum_level: VerificationLevel,
    pub scope: String,
    pub assumptions: Vec<String>,
    pub required: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticContract {
    pub contract_id: String,
    pub category: SemanticCategory,
    pub input_domain: Vec<String>,
    pub output_guarantees: Vec<String>,
    pub state_invariants: Vec<String>,
    pub numerical_contract: String,
    pub deterministic: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceRequirements {
    pub peak_device_bytes: u64,
    pub peak_host_bytes: u64,
    pub queue_entries: u64,
    pub worker_processes: u64,
    pub communication_buffer_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HardwarePrecondition {
    pub architecture: Option<String>,
    pub minimum_device_memory_bytes: u64,
    pub required_features: Vec<String>,
    pub forbidden_features: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SoftwareRequirement {
    pub package: String,
    pub version_range: String,
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SoftwarePrecondition {
    pub requirements: Vec<SoftwareRequirement>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QualityImplication {
    pub metric: String,
    pub expected_delta: f64,
    pub maximum_regression: f64,
    pub evaluation_contract_id: String,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PerformanceEstimate {
    pub metric: String,
    pub expected_delta: f64,
    pub unit: String,
    pub model_id: String,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Uncertainty {
    pub method: String,
    pub confidence: f64,
    pub lower: f64,
    pub upper: f64,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GenomeNodeMetadata {
    pub stable_id: String,
    pub semantic_contract: SemanticContract,
    pub resource_requirements: ResourceRequirements,
    pub legal_rewrite_rules: Vec<String>,
    pub proof_obligations: Vec<ProofObligation>,
    pub hardware_preconditions: Vec<HardwarePrecondition>,
    pub software_preconditions: Vec<SoftwarePrecondition>,
    pub quality_implications: Vec<QualityImplication>,
    pub expected_performance: Vec<PerformanceEstimate>,
    pub uncertainty: Uncertainty,
    pub hot_swap_category: HotSwapCategory,
    pub lineage_references: Vec<LineageReference>,
    pub evidence_references: Vec<EvidenceReference>,
    pub frozen: bool,
    pub extensions: Extensions,
}

impl GenomeNodeMetadata {
    // This is the single admission gate for every mutable-node obligation and
    // precondition. Keeping the checks together prevents a newly added field
    // from being validated on only one side of the Rust/Python boundary.
    #[allow(clippy::too_many_lines)]
    fn validate_at(&self, path: &str) -> Result<(), ValidationError> {
        nonempty(&format!("{path}.stable_id"), &self.stable_id)?;
        nonempty(
            &format!("{path}.semantic_contract.contract_id"),
            &self.semantic_contract.contract_id,
        )?;
        nonempty_values(
            &format!("{path}.semantic_contract.input_domain"),
            &self.semantic_contract.input_domain,
        )?;
        nonempty_values(
            &format!("{path}.semantic_contract.output_guarantees"),
            &self.semantic_contract.output_guarantees,
        )?;
        nonempty_values(
            &format!("{path}.semantic_contract.state_invariants"),
            &self.semantic_contract.state_invariants,
        )?;
        nonempty(
            &format!("{path}.semantic_contract.numerical_contract"),
            &self.semantic_contract.numerical_contract,
        )?;
        if self.legal_rewrite_rules.is_empty() && !self.frozen {
            return Err(ValidationError::new(
                format!("{path}.legal_rewrite_rules"),
                "mutable nodes must declare legal rewrite rules",
            ));
        }
        nonempty_values(
            &format!("{path}.legal_rewrite_rules"),
            &self.legal_rewrite_rules,
        )?;
        if self.proof_obligations.is_empty() {
            return Err(ValidationError::new(
                format!("{path}.proof_obligations"),
                "must not be empty",
            ));
        }
        for (index, obligation) in self.proof_obligations.iter().enumerate() {
            let prefix = format!("{path}.proof_obligations[{index}]");
            nonempty(
                &format!("{prefix}.obligation_id"),
                &obligation.obligation_id,
            )?;
            nonempty(&format!("{prefix}.property"), &obligation.property)?;
            nonempty(&format!("{prefix}.scope"), &obligation.scope)?;
            nonempty_values(&format!("{prefix}.assumptions"), &obligation.assumptions)?;
        }
        for (index, precondition) in self.hardware_preconditions.iter().enumerate() {
            let prefix = format!("{path}.hardware_preconditions[{index}]");
            if let Some(architecture) = &precondition.architecture {
                nonempty(&format!("{prefix}.architecture"), architecture)?;
            }
            nonempty_values(
                &format!("{prefix}.required_features"),
                &precondition.required_features,
            )?;
            nonempty_values(
                &format!("{prefix}.forbidden_features"),
                &precondition.forbidden_features,
            )?;
        }
        for (index, precondition) in self.software_preconditions.iter().enumerate() {
            for (requirement_index, requirement) in precondition.requirements.iter().enumerate() {
                let prefix = format!(
                    "{path}.software_preconditions[{index}].requirements[{requirement_index}]"
                );
                nonempty(&format!("{prefix}.package"), &requirement.package)?;
                nonempty(
                    &format!("{prefix}.version_range"),
                    &requirement.version_range,
                )?;
            }
        }
        for (index, implication) in self.quality_implications.iter().enumerate() {
            let prefix = format!("{path}.quality_implications[{index}]");
            nonempty(&format!("{prefix}.metric"), &implication.metric)?;
            if !implication.expected_delta.is_finite() {
                return Err(ValidationError::new(
                    format!("{prefix}.expected_delta"),
                    "must be finite",
                ));
            }
            finite_nonnegative(
                &format!("{prefix}.maximum_regression"),
                implication.maximum_regression,
            )?;
            nonempty(
                &format!("{prefix}.evaluation_contract_id"),
                &implication.evaluation_contract_id,
            )?;
        }
        for (index, estimate) in self.expected_performance.iter().enumerate() {
            let prefix = format!("{path}.expected_performance[{index}]");
            nonempty(&format!("{prefix}.metric"), &estimate.metric)?;
            if !estimate.expected_delta.is_finite() {
                return Err(ValidationError::new(
                    format!("{prefix}.expected_delta"),
                    "must be finite",
                ));
            }
            nonempty(&format!("{prefix}.unit"), &estimate.unit)?;
            nonempty(&format!("{prefix}.model_id"), &estimate.model_id)?;
        }
        nonempty(
            &format!("{path}.uncertainty.method"),
            &self.uncertainty.method,
        )?;
        if !self.uncertainty.confidence.is_finite()
            || !(0.0..=1.0).contains(&self.uncertainty.confidence)
            || !self.uncertainty.lower.is_finite()
            || !self.uncertainty.upper.is_finite()
            || self.uncertainty.lower > self.uncertainty.upper
        {
            return Err(ValidationError::new(
                format!("{path}.uncertainty"),
                "invalid finite confidence interval",
            ));
        }
        for (index, reference) in self.lineage_references.iter().enumerate() {
            nonempty(
                &format!("{path}.lineage_references[{index}].lineage_id"),
                &reference.lineage_id,
            )?;
        }
        for (index, reference) in self.evidence_references.iter().enumerate() {
            reference.validate_at(&format!("{path}.evidence_references[{index}]"))?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkflowStep {
    pub node: GenomeNodeMetadata,
    pub kind: WorkflowStepKind,
    pub target: String,
    pub branch_probability: f64,
    pub expected_latency_ms: f64,
    pub deadline_ms: Option<f64>,
    pub priority: i64,
    pub maximum_iterations: u64,
    pub model_cascade_targets: Vec<String>,
    pub expected_future_requests: f64,
    pub shared_prefix_group: Option<String>,
    pub cancellation_behavior: CancellationBehavior,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkflowEdge {
    pub node: GenomeNodeMetadata,
    pub source_id: String,
    pub target_id: String,
    pub condition: String,
    pub probability: f64,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkflowGenome {
    pub node: GenomeNodeMetadata,
    pub steps: Vec<WorkflowStep>,
    pub edges: Vec<WorkflowEdge>,
    pub entry_step_id: String,
    pub workflow_deadline_ms: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RequestGenome {
    pub node: GenomeNodeMetadata,
    pub admission_control: AdmissionControl,
    pub maximum_queue_depth: u64,
    pub default_priority: i64,
    pub default_deadline_ms: Option<f64>,
    pub batching_eligible: bool,
    pub routing: RoutingPolicy,
    pub queue_discipline: QueueDiscipline,
    pub cancellation_behavior: CancellationBehavior,
    pub maximum_retries: u64,
    pub streaming_semantics: StreamingSemantics,
    pub request_classes: Vec<String>,
    pub tenant_isolation: bool,
    pub workflow_identity_required: bool,
    pub quality_tiers: Vec<String>,
    pub fallback_behavior: FallbackBehavior,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
// These independent wire flags preserve additive compatibility with the Python
// and JSON Schema representations; they do not encode a runtime state machine.
#[allow(clippy::struct_excessive_bools)]
pub struct ServingGenome {
    pub node: GenomeNodeMetadata,
    pub topology: ServingTopology,
    pub prefill_policy: PrefillPolicy,
    pub incremental_prefill: bool,
    pub prefill_chunk_tokens: u64,
    pub decode_scheduling: DecodeScheduling,
    pub continuous_batching: bool,
    pub maximum_batch_tokens: u64,
    pub speculative_decoding: bool,
    pub draft_model_id: Option<String>,
    pub verification_policy: String,
    pub model_cascade: Vec<String>,
    pub decode_chunk_tokens: u64,
    pub request_migration: bool,
    pub worker_roles: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StateSpec {
    pub node: GenomeNodeMetadata,
    pub state_id: String,
    pub kind: StateKind,
    pub cache_key_fields: Vec<String>,
    pub ownership: StateOwnership,
    pub layout: StateLayout,
    pub precision: Precision,
    pub retention: RetentionPolicy,
    pub replication_factor: u64,
    pub migratable: bool,
    pub offload_tier: OffloadTier,
    pub checkpoint_interval_tokens: u64,
    pub eviction_policy: String,
    pub recomputable: bool,
    pub consistency: ConsistencyModel,
    pub recovery_behavior: String,
    pub maximum_bytes_per_request: u64,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StateGenome {
    pub node: GenomeNodeMetadata,
    pub states: Vec<StateSpec>,
    pub migration_chunk_bytes: u64,
    pub prefetch_enabled: bool,
    pub conversion_artifact: Option<EvidenceReference>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ParallelismSpec {
    pub node: GenomeNodeMetadata,
    pub tensor: u64,
    pub pipeline: u64,
    pub data: u64,
    pub expert: u64,
    pub context: u64,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Placement {
    pub node: GenomeNodeMetadata,
    pub logical_rank: u64,
    pub host_id: String,
    pub device_id: String,
    pub numa_domain: Option<String>,
    pub network_rail: Option<String>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExpertPlacement {
    pub node: GenomeNodeMetadata,
    pub expert_id: u64,
    pub logical_ranks: Vec<u64>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CollectiveStep {
    pub node: GenomeNodeMetadata,
    pub step_id: String,
    pub kind: CollectiveKind,
    pub dependencies: Vec<String>,
    pub algorithm: String,
    pub transport: String,
    pub ranks: Vec<u64>,
    pub chunk_bytes: u64,
    pub overlap_group: Option<String>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DistributedGenome {
    pub node: GenomeNodeMetadata,
    pub parallelism: ParallelismSpec,
    pub rank_placement: Vec<Placement>,
    pub expert_placement: Vec<ExpertPlacement>,
    pub collective_dag: Vec<CollectiveStep>,
    pub prefill_decode_transfer: PrefillDecodeTransfer,
    pub failure_domains: Vec<String>,
    pub recovery_variant_ids: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SymbolicDimension {
    pub node: GenomeNodeMetadata,
    pub name: String,
    pub minimum: u64,
    pub maximum: u64,
    pub divisible_by: u64,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TensorValue {
    pub node: GenomeNodeMetadata,
    pub value_id: String,
    pub shape: Vec<String>,
    pub dtype: Precision,
    pub strides: Vec<String>,
    pub layout: String,
    pub alias_group: Option<String>,
    pub state_dependency: Option<String>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TensorOperator {
    pub node: GenomeNodeMetadata,
    pub operator_id: String,
    pub operator: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub fused_operators: Vec<String>,
    pub decomposition: Vec<String>,
    pub quantization: String,
    pub sparse: bool,
    pub numerical_contract: String,
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RewriteRecord {
    pub transformation_id: String,
    pub source_hash: ArtifactDigest,
    pub target_hash: ArtifactDigest,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TensorGenome {
    pub node: GenomeNodeMetadata,
    pub symbolic_dimensions: Vec<SymbolicDimension>,
    pub values: Vec<TensorValue>,
    pub operators: Vec<TensorOperator>,
    pub graph_inputs: Vec<String>,
    pub graph_outputs: Vec<String>,
    pub rewrite_history: Vec<RewriteRecord>,
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LaunchConfiguration {
    pub block_x: u64,
    pub block_y: u64,
    pub block_z: u64,
    pub warps: u64,
    pub pipeline_stages: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ShapeDomain {
    pub constraints: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct KernelSpec {
    pub node: GenomeNodeMetadata,
    pub kernel_id: String,
    pub source_artifact: EvidenceReference,
    pub backend: KernelBackend,
    pub target_architecture: String,
    pub launch: LaunchConfiguration,
    pub tile_shape: Vec<u64>,
    pub warp_strategy: String,
    pub shared_memory_bytes: u64,
    pub register_estimate: u64,
    pub vector_width: u64,
    pub layout_assumptions: Vec<String>,
    pub supported_shapes: ShapeDomain,
    pub supported_dtypes: Vec<Precision>,
    pub deterministic: bool,
    pub numerical_tolerance: f64,
    pub benchmark_evidence: Vec<EvidenceReference>,
    pub fallback_kernel_id: String,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct KernelGenome {
    pub node: GenomeNodeMetadata,
    pub kernels: Vec<KernelSpec>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveryTransition {
    pub node: GenomeNodeMetadata,
    pub transition_id: String,
    pub safe_point: TransitionPoint,
    pub source_state_contract: String,
    pub target_state_contract: String,
    pub state_conversion_artifact: Option<EvidenceReference>,
    pub state_transfer: StateTransfer,
    pub active_stream_behavior: ActiveStreamBehavior,
    pub rollback_transition_id: String,
    pub failure_invariants: Vec<String>,
    pub operator_action_required: bool,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveryGenome {
    pub node: GenomeNodeMetadata,
    pub transitions: Vec<RecoveryTransition>,
    pub shadow_mode: bool,
    pub canary_mode: bool,
    pub degraded_mode_ids: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct InferenceGenome {
    pub schema_version: String,
    pub api_version: String,
    pub kind: String,
    pub genome_id: String,
    pub seed: u64,
    pub source_model: ArtifactDigest,
    pub workflow: WorkflowGenome,
    pub request: RequestGenome,
    pub serving: ServingGenome,
    pub state: StateGenome,
    pub distributed: DistributedGenome,
    pub tensor: TensorGenome,
    pub kernel: KernelGenome,
    pub recovery: RecoveryGenome,
    pub extensions: Extensions,
}

fn validate_document_header(
    schema_version: &str,
    api_version: &str,
    kind: &str,
    expected_kind: &str,
) -> Result<(), ValidationError> {
    if schema_version != SCHEMA_VERSION {
        return Err(ValidationError::new(
            "schema_version",
            format!("expected {SCHEMA_VERSION}"),
        ));
    }
    if api_version != API_VERSION {
        return Err(ValidationError::new(
            "api_version",
            format!("expected {API_VERSION}"),
        ));
    }
    if kind != expected_kind {
        return Err(ValidationError::new(
            "kind",
            format!("expected {expected_kind}"),
        ));
    }
    Ok(())
}

fn validate_acyclic(
    path: &str,
    dependencies: &BTreeMap<String, Vec<String>>,
) -> Result<(), ValidationError> {
    fn visit(
        node: &str,
        path: &str,
        dependencies: &BTreeMap<String, Vec<String>>,
        visiting: &mut BTreeSet<String>,
        visited: &mut BTreeSet<String>,
    ) -> Result<(), ValidationError> {
        if visiting.contains(node) {
            return Err(ValidationError::new(path, "must be acyclic"));
        }
        if visited.contains(node) {
            return Ok(());
        }
        visiting.insert(node.to_owned());
        for dependency in dependencies.get(node).into_iter().flatten() {
            if !dependencies.contains_key(dependency) {
                return Err(ValidationError::new(
                    path,
                    "dependency must reference a declared node",
                ));
            }
            visit(dependency, path, dependencies, visiting, visited)?;
        }
        visiting.remove(node);
        visited.insert(node.to_owned());
        Ok(())
    }

    let mut visiting = BTreeSet::new();
    let mut visited = BTreeSet::new();
    for node in dependencies.keys() {
        visit(node, path, dependencies, &mut visiting, &mut visited)?;
    }
    Ok(())
}

impl Validate for InferenceGenome {
    // Keeping the complete document validation in one auditable admission gate
    // makes it harder for a newly added genome region to bypass validation.
    #[allow(clippy::too_many_lines)]
    fn validate(&self) -> Result<(), ValidationError> {
        validate_document_header(
            &self.schema_version,
            &self.api_version,
            &self.kind,
            "InferenceGenome",
        )?;
        nonempty("genome_id", &self.genome_id)?;
        self.source_model.validate_at("source_model")?;
        for (path, node) in [
            ("workflow.node", &self.workflow.node),
            ("request.node", &self.request.node),
            ("serving.node", &self.serving.node),
            ("state.node", &self.state.node),
            ("distributed.node", &self.distributed.node),
            ("tensor.node", &self.tensor.node),
            ("kernel.node", &self.kernel.node),
            ("recovery.node", &self.recovery.node),
        ] {
            node.validate_at(path)?;
        }
        if self.request.maximum_queue_depth == 0 {
            return Err(ValidationError::new(
                "request.maximum_queue_depth",
                "must be positive",
            ));
        }
        if let Some(deadline) = self.request.default_deadline_ms {
            finite_positive("request.default_deadline_ms", deadline)?;
        }
        nonempty_values("request.request_classes", &self.request.request_classes)?;
        nonempty_values("request.quality_tiers", &self.request.quality_tiers)?;
        if self.serving.prefill_chunk_tokens == 0
            || self.serving.maximum_batch_tokens == 0
            || self.serving.decode_chunk_tokens == 0
        {
            return Err(ValidationError::new(
                "serving",
                "token chunk and batch bounds must be positive",
            ));
        }
        nonempty(
            "serving.verification_policy",
            &self.serving.verification_policy,
        )?;
        nonempty_values("serving.model_cascade", &self.serving.model_cascade)?;
        nonempty_values("serving.worker_roles", &self.serving.worker_roles)?;
        if self.serving.speculative_decoding && self.serving.draft_model_id.is_none() {
            return Err(ValidationError::new(
                "serving.draft_model_id",
                "required when speculative decoding is enabled",
            ));
        }
        if let Some(draft_model_id) = &self.serving.draft_model_id {
            nonempty("serving.draft_model_id", draft_model_id)?;
        }
        if self.state.migration_chunk_bytes == 0 {
            return Err(ValidationError::new(
                "state.migration_chunk_bytes",
                "must be positive",
            ));
        }
        if let Some(reference) = &self.state.conversion_artifact {
            reference.validate_at("state.conversion_artifact")?;
        }
        if let Some(deadline) = self.workflow.workflow_deadline_ms {
            finite_positive("workflow.workflow_deadline_ms", deadline)?;
        }
        let step_ids: BTreeSet<_> = self
            .workflow
            .steps
            .iter()
            .map(|step| step.node.stable_id.as_str())
            .collect();
        if step_ids.len() != self.workflow.steps.len() {
            return Err(ValidationError::new(
                "workflow.steps",
                "stable identifiers must be unique",
            ));
        }
        if !step_ids.contains(self.workflow.entry_step_id.as_str()) {
            return Err(ValidationError::new(
                "workflow.entry_step_id",
                "must reference a declared step",
            ));
        }
        for (index, edge) in self.workflow.edges.iter().enumerate() {
            edge.node
                .validate_at(&format!("workflow.edges[{index}].node"))?;
            if !step_ids.contains(edge.source_id.as_str())
                || !step_ids.contains(edge.target_id.as_str())
            {
                return Err(ValidationError::new(
                    format!("workflow.edges[{index}]"),
                    "must reference declared steps",
                ));
            }
            nonempty(
                &format!("workflow.edges[{index}].condition"),
                &edge.condition,
            )?;
            if !edge.probability.is_finite() || !(0.0..=1.0).contains(&edge.probability) {
                return Err(ValidationError::new(
                    format!("workflow.edges[{index}].probability"),
                    "must be a finite probability",
                ));
            }
        }
        let mut workflow_dependencies: BTreeMap<String, Vec<String>> = self
            .workflow
            .steps
            .iter()
            .map(|step| (step.node.stable_id.clone(), Vec::new()))
            .collect();
        let mut workflow_edges = BTreeSet::new();
        for edge in &self.workflow.edges {
            if !workflow_edges.insert((
                edge.source_id.as_str(),
                edge.target_id.as_str(),
                edge.condition.as_str(),
            )) {
                return Err(ValidationError::new(
                    "workflow.edges",
                    "edges must be unique",
                ));
            }
            workflow_dependencies
                .get_mut(&edge.target_id)
                .ok_or_else(|| ValidationError::new("workflow.edges", "target must be declared"))?
                .push(edge.source_id.clone());
        }
        validate_acyclic("workflow.edges", &workflow_dependencies)?;
        for (index, step) in self.workflow.steps.iter().enumerate() {
            step.node
                .validate_at(&format!("workflow.steps[{index}].node"))?;
            if !step.branch_probability.is_finite()
                || !(0.0..=1.0).contains(&step.branch_probability)
            {
                return Err(ValidationError::new(
                    format!("workflow.steps[{index}].branch_probability"),
                    "must be a finite probability",
                ));
            }
            nonempty(&format!("workflow.steps[{index}].target"), &step.target)?;
            finite_nonnegative(
                &format!("workflow.steps[{index}].expected_latency_ms"),
                step.expected_latency_ms,
            )?;
            if let Some(deadline) = step.deadline_ms {
                finite_positive(&format!("workflow.steps[{index}].deadline_ms"), deadline)?;
            }
            finite_nonnegative(
                &format!("workflow.steps[{index}].expected_future_requests"),
                step.expected_future_requests,
            )?;
            nonempty_values(
                &format!("workflow.steps[{index}].model_cascade_targets"),
                &step.model_cascade_targets,
            )?;
            if let Some(group) = &step.shared_prefix_group {
                nonempty(
                    &format!("workflow.steps[{index}].shared_prefix_group"),
                    group,
                )?;
            }
        }
        for (index, state) in self.state.states.iter().enumerate() {
            state
                .node
                .validate_at(&format!("state.states[{index}].node"))?;
            nonempty(&format!("state.states[{index}].state_id"), &state.state_id)?;
            nonempty_values(
                &format!("state.states[{index}].cache_key_fields"),
                &state.cache_key_fields,
            )?;
            if state.replication_factor == 0 {
                return Err(ValidationError::new(
                    format!("state.states[{index}].replication_factor"),
                    "must be positive",
                ));
            }
            nonempty(
                &format!("state.states[{index}].eviction_policy"),
                &state.eviction_policy,
            )?;
            nonempty(
                &format!("state.states[{index}].recovery_behavior"),
                &state.recovery_behavior,
            )?;
        }
        let state_ids: BTreeSet<_> = self
            .state
            .states
            .iter()
            .map(|state| state.state_id.as_str())
            .collect();
        if state_ids.len() != self.state.states.len() {
            return Err(ValidationError::new(
                "state.states",
                "state identifiers must be unique",
            ));
        }
        self.distributed
            .parallelism
            .node
            .validate_at("distributed.parallelism.node")?;
        if [
            self.distributed.parallelism.tensor,
            self.distributed.parallelism.pipeline,
            self.distributed.parallelism.data,
            self.distributed.parallelism.expert,
            self.distributed.parallelism.context,
        ]
        .contains(&0)
        {
            return Err(ValidationError::new(
                "distributed.parallelism",
                "parallelism degrees must be positive",
            ));
        }
        for (index, placement) in self.distributed.rank_placement.iter().enumerate() {
            placement
                .node
                .validate_at(&format!("distributed.rank_placement[{index}].node"))?;
            nonempty(
                &format!("distributed.rank_placement[{index}].host_id"),
                &placement.host_id,
            )?;
            nonempty(
                &format!("distributed.rank_placement[{index}].device_id"),
                &placement.device_id,
            )?;
            if let Some(domain) = &placement.numa_domain {
                nonempty(
                    &format!("distributed.rank_placement[{index}].numa_domain"),
                    domain,
                )?;
            }
            if let Some(rail) = &placement.network_rail {
                nonempty(
                    &format!("distributed.rank_placement[{index}].network_rail"),
                    rail,
                )?;
            }
        }
        let rank_ids: BTreeSet<_> = self
            .distributed
            .rank_placement
            .iter()
            .map(|placement| placement.logical_rank)
            .collect();
        if rank_ids.len() != self.distributed.rank_placement.len() {
            return Err(ValidationError::new(
                "distributed.rank_placement",
                "logical ranks must be unique",
            ));
        }
        let mut expert_ids = BTreeSet::new();
        for (index, placement) in self.distributed.expert_placement.iter().enumerate() {
            placement
                .node
                .validate_at(&format!("distributed.expert_placement[{index}].node"))?;
            if !expert_ids.insert(placement.expert_id) {
                return Err(ValidationError::new(
                    "distributed.expert_placement",
                    "expert identifiers must be unique",
                ));
            }
            let placement_ranks: BTreeSet<_> = placement.logical_ranks.iter().copied().collect();
            if placement_ranks.len() != placement.logical_ranks.len()
                || !placement_ranks.is_subset(&rank_ids)
            {
                return Err(ValidationError::new(
                    format!("distributed.expert_placement[{index}].logical_ranks"),
                    "must uniquely reference declared logical ranks",
                ));
            }
        }
        let mut collective_dependencies = BTreeMap::new();
        for (index, collective) in self.distributed.collective_dag.iter().enumerate() {
            collective
                .node
                .validate_at(&format!("distributed.collective_dag[{index}].node"))?;
            nonempty(
                &format!("distributed.collective_dag[{index}].step_id"),
                &collective.step_id,
            )?;
            if collective_dependencies
                .insert(collective.step_id.clone(), collective.dependencies.clone())
                .is_some()
            {
                return Err(ValidationError::new(
                    "distributed.collective_dag",
                    "step identifiers must be unique",
                ));
            }
            let dependencies: BTreeSet<_> = collective.dependencies.iter().collect();
            let ranks: BTreeSet<_> = collective.ranks.iter().copied().collect();
            if dependencies.len() != collective.dependencies.len() {
                return Err(ValidationError::new(
                    format!("distributed.collective_dag[{index}].dependencies"),
                    "dependencies must be unique",
                ));
            }
            if ranks.len() != collective.ranks.len() || !ranks.is_subset(&rank_ids) {
                return Err(ValidationError::new(
                    format!("distributed.collective_dag[{index}].ranks"),
                    "must uniquely reference declared logical ranks",
                ));
            }
            if collective.chunk_bytes == 0 {
                return Err(ValidationError::new(
                    format!("distributed.collective_dag[{index}].chunk_bytes"),
                    "must be positive",
                ));
            }
            nonempty(
                &format!("distributed.collective_dag[{index}].algorithm"),
                &collective.algorithm,
            )?;
            nonempty(
                &format!("distributed.collective_dag[{index}].transport"),
                &collective.transport,
            )?;
            if let Some(group) = &collective.overlap_group {
                nonempty(
                    &format!("distributed.collective_dag[{index}].overlap_group"),
                    group,
                )?;
            }
        }
        nonempty_values(
            "distributed.failure_domains",
            &self.distributed.failure_domains,
        )?;
        validate_acyclic("distributed.collective_dag", &collective_dependencies)?;
        let value_ids: BTreeSet<_> = self
            .tensor
            .values
            .iter()
            .map(|value| value.value_id.as_str())
            .collect();
        for (index, dimension) in self.tensor.symbolic_dimensions.iter().enumerate() {
            dimension
                .node
                .validate_at(&format!("tensor.symbolic_dimensions[{index}].node"))?;
            if dimension.minimum == 0
                || dimension.minimum > dimension.maximum
                || dimension.divisible_by == 0
            {
                return Err(ValidationError::new(
                    format!("tensor.symbolic_dimensions[{index}]"),
                    "invalid positive symbolic dimension range",
                ));
            }
            nonempty(
                &format!("tensor.symbolic_dimensions[{index}].name"),
                &dimension.name,
            )?;
        }
        for (index, value) in self.tensor.values.iter().enumerate() {
            value
                .node
                .validate_at(&format!("tensor.values[{index}].node"))?;
            nonempty(&format!("tensor.values[{index}].value_id"), &value.value_id)?;
            nonempty_values(&format!("tensor.values[{index}].shape"), &value.shape)?;
            nonempty_values(&format!("tensor.values[{index}].strides"), &value.strides)?;
            nonempty(&format!("tensor.values[{index}].layout"), &value.layout)?;
            if let Some(group) = &value.alias_group {
                nonempty(&format!("tensor.values[{index}].alias_group"), group)?;
            }
        }
        if value_ids.len() != self.tensor.values.len() {
            return Err(ValidationError::new(
                "tensor.values",
                "value identifiers must be unique",
            ));
        }
        if self
            .tensor
            .graph_inputs
            .iter()
            .chain(&self.tensor.graph_outputs)
            .any(|value_id| !value_ids.contains(value_id.as_str()))
        {
            return Err(ValidationError::new(
                "tensor.graph_inputs",
                "graph boundary must reference declared values",
            ));
        }
        for (index, operator) in self.tensor.operators.iter().enumerate() {
            operator
                .node
                .validate_at(&format!("tensor.operators[{index}].node"))?;
            nonempty(
                &format!("tensor.operators[{index}].operator_id"),
                &operator.operator_id,
            )?;
            if operator
                .inputs
                .iter()
                .chain(&operator.outputs)
                .any(|value_id| !value_ids.contains(value_id.as_str()))
            {
                return Err(ValidationError::new(
                    format!("tensor.operators[{index}]"),
                    "operator must reference declared values",
                ));
            }
            nonempty(
                &format!("tensor.operators[{index}].operator"),
                &operator.operator,
            )?;
            nonempty_values(
                &format!("tensor.operators[{index}].fused_operators"),
                &operator.fused_operators,
            )?;
            nonempty_values(
                &format!("tensor.operators[{index}].decomposition"),
                &operator.decomposition,
            )?;
            nonempty(
                &format!("tensor.operators[{index}].quantization"),
                &operator.quantization,
            )?;
            nonempty(
                &format!("tensor.operators[{index}].numerical_contract"),
                &operator.numerical_contract,
            )?;
        }
        let operator_ids: BTreeSet<_> = self
            .tensor
            .operators
            .iter()
            .map(|operator| operator.operator_id.as_str())
            .collect();
        if operator_ids.len() != self.tensor.operators.len() {
            return Err(ValidationError::new(
                "tensor.operators",
                "operator identifiers must be unique",
            ));
        }
        let mut producers = BTreeMap::new();
        for (index, operator) in self.tensor.operators.iter().enumerate() {
            let outputs: BTreeSet<_> = operator.outputs.iter().collect();
            if outputs.len() != operator.outputs.len() {
                return Err(ValidationError::new(
                    format!("tensor.operators[{index}].outputs"),
                    "outputs must be unique",
                ));
            }
            for output in &operator.outputs {
                if producers
                    .insert(output.as_str(), operator.operator_id.as_str())
                    .is_some()
                {
                    return Err(ValidationError::new(
                        "tensor.operators",
                        "tensor values must have a single producer",
                    ));
                }
            }
        }
        for (index, value) in self.tensor.values.iter().enumerate() {
            if value
                .state_dependency
                .as_ref()
                .is_some_and(|dependency| !state_ids.contains(dependency.as_str()))
            {
                return Err(ValidationError::new(
                    format!("tensor.values[{index}].state_dependency"),
                    "must reference declared state",
                ));
            }
        }
        if self.tensor.graph_outputs.iter().any(|output| {
            !producers.contains_key(output.as_str()) && !self.tensor.graph_inputs.contains(output)
        }) {
            return Err(ValidationError::new(
                "tensor.graph_outputs",
                "must be produced or passed through",
            ));
        }
        let kernel_ids: BTreeSet<_> = self
            .kernel
            .kernels
            .iter()
            .map(|kernel| kernel.kernel_id.as_str())
            .collect();
        if kernel_ids.len() != self.kernel.kernels.len() {
            return Err(ValidationError::new(
                "kernel.kernels",
                "kernel identifiers must be unique",
            ));
        }
        for (index, kernel) in self.kernel.kernels.iter().enumerate() {
            kernel
                .node
                .validate_at(&format!("kernel.kernels[{index}].node"))?;
            nonempty(
                &format!("kernel.kernels[{index}].kernel_id"),
                &kernel.kernel_id,
            )?;
            kernel
                .source_artifact
                .validate_at(&format!("kernel.kernels[{index}].source_artifact"))?;
            nonempty(
                &format!("kernel.kernels[{index}].target_architecture"),
                &kernel.target_architecture,
            )?;
            if kernel.launch.block_x == 0
                || kernel.launch.block_y == 0
                || kernel.launch.block_z == 0
                || kernel.launch.warps == 0
                || kernel.launch.pipeline_stages == 0
                || kernel.tile_shape.contains(&0)
                || kernel.vector_width == 0
            {
                return Err(ValidationError::new(
                    format!("kernel.kernels[{index}]"),
                    "launch, tile and vector dimensions must be positive",
                ));
            }
            nonempty(
                &format!("kernel.kernels[{index}].warp_strategy"),
                &kernel.warp_strategy,
            )?;
            nonempty_values(
                &format!("kernel.kernels[{index}].layout_assumptions"),
                &kernel.layout_assumptions,
            )?;
            nonempty_values(
                &format!("kernel.kernels[{index}].supported_shapes.constraints"),
                &kernel.supported_shapes.constraints,
            )?;
            finite_nonnegative(
                &format!("kernel.kernels[{index}].numerical_tolerance"),
                kernel.numerical_tolerance,
            )?;
            for (evidence_index, reference) in kernel.benchmark_evidence.iter().enumerate() {
                reference.validate_at(&format!(
                    "kernel.kernels[{index}].benchmark_evidence[{evidence_index}]"
                ))?;
            }
            if !kernel_ids.contains(kernel.fallback_kernel_id.as_str()) {
                return Err(ValidationError::new(
                    format!("kernel.kernels[{index}].fallback_kernel_id"),
                    "must reference a declared kernel",
                ));
            }
        }
        let transition_ids: BTreeSet<_> = self
            .recovery
            .transitions
            .iter()
            .map(|transition| transition.transition_id.as_str())
            .collect();
        if transition_ids.len() != self.recovery.transitions.len() {
            return Err(ValidationError::new(
                "recovery.transitions",
                "transition identifiers must be unique",
            ));
        }
        for (index, transition) in self.recovery.transitions.iter().enumerate() {
            transition
                .node
                .validate_at(&format!("recovery.transitions[{index}].node"))?;
            nonempty(
                &format!("recovery.transitions[{index}].transition_id"),
                &transition.transition_id,
            )?;
            nonempty(
                &format!("recovery.transitions[{index}].source_state_contract"),
                &transition.source_state_contract,
            )?;
            nonempty(
                &format!("recovery.transitions[{index}].target_state_contract"),
                &transition.target_state_contract,
            )?;
            if let Some(reference) = &transition.state_conversion_artifact {
                reference.validate_at(&format!(
                    "recovery.transitions[{index}].state_conversion_artifact"
                ))?;
            }
            nonempty_values(
                &format!("recovery.transitions[{index}].failure_invariants"),
                &transition.failure_invariants,
            )?;
            if !transition_ids.contains(transition.rollback_transition_id.as_str()) {
                return Err(ValidationError::new(
                    format!("recovery.transitions[{index}].rollback_transition_id"),
                    "must reference a declared transition",
                ));
            }
        }
        if self
            .distributed
            .recovery_variant_ids
            .iter()
            .any(|identifier| !transition_ids.contains(identifier.as_str()))
        {
            return Err(ValidationError::new(
                "distributed.recovery_variant_ids",
                "must reference declared recovery transitions",
            ));
        }
        nonempty_values(
            "recovery.degraded_mode_ids",
            &self.recovery.degraded_mode_ids,
        )?;
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GenomePattern {
    pub region: GenomeRegion,
    pub node_ids: Vec<String>,
    pub structural_constraints: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EstimatedChange {
    pub metric: String,
    pub lower: f64,
    pub expected: f64,
    pub upper: f64,
    pub unit: String,
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LearnedConstraint {
    pub constraint_id: String,
    pub expression: String,
    pub scope: String,
    pub counterexample_ids: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Transformation {
    pub schema_version: String,
    pub api_version: String,
    pub kind: String,
    pub transformation_id: String,
    pub family: TransformationFamily,
    pub source_pattern: GenomePattern,
    pub target_pattern: GenomePattern,
    pub semantic_category: SemanticCategory,
    pub designation: TransformationDesignation,
    pub preconditions: Vec<String>,
    pub postconditions: Vec<String>,
    pub expected_quality_cost: Vec<EstimatedChange>,
    pub expected_resource_change: Vec<EstimatedChange>,
    pub expected_performance_change: Vec<EstimatedChange>,
    pub affected_regions: Vec<String>,
    pub verification_obligations: Vec<ProofObligation>,
    pub required_verifier_stages: Vec<String>,
    pub required_benchmark_stages: Vec<String>,
    pub rollback_strategy: String,
    pub proposal_source: String,
    pub parent_transformations: Vec<String>,
    pub learned_constraints: Vec<LearnedConstraint>,
    pub counterexample_references: Vec<String>,
    pub lineage_references: Vec<LineageReference>,
    pub extensions: Extensions,
}

impl Validate for Transformation {
    fn validate(&self) -> Result<(), ValidationError> {
        validate_document_header(
            &self.schema_version,
            &self.api_version,
            &self.kind,
            "Transformation",
        )?;
        nonempty("transformation_id", &self.transformation_id)?;
        if self.verification_obligations.is_empty() {
            return Err(ValidationError::new(
                "verification_obligations",
                "must not be empty",
            ));
        }
        if self.designation == TransformationDesignation::ApproximateWithinQualityBudget
            && self.expected_quality_cost.is_empty()
        {
            return Err(ValidationError::new(
                "expected_quality_cost",
                "approximate transformations must declare quality cost",
            ));
        }
        for (index, change) in self
            .expected_quality_cost
            .iter()
            .chain(&self.expected_resource_change)
            .chain(&self.expected_performance_change)
            .enumerate()
        {
            if !(change.lower.is_finite()
                && change.expected.is_finite()
                && change.upper.is_finite()
                && change.lower <= change.expected
                && change.expected <= change.upper)
            {
                return Err(ValidationError::new(
                    format!("estimated_changes[{index}]"),
                    "must be a finite ordered interval",
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SearchBudget {
    pub wall_time_seconds: f64,
    pub cpu_time_seconds: f64,
    pub gpu_time_seconds: f64,
    pub cloud_cost_usd: f64,
    pub external_synthesis_cost_usd: f64,
    pub candidate_count: u64,
    pub compilation_count: u64,
    pub benchmark_count: u64,
    pub verifier_time_seconds: f64,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BudgetUsage {
    pub wall_time_seconds: f64,
    pub cpu_time_seconds: f64,
    pub gpu_time_seconds: f64,
    pub cloud_cost_usd: f64,
    pub external_synthesis_cost_usd: f64,
    pub candidate_count: u64,
    pub compilation_count: u64,
    pub benchmark_count: u64,
    pub verifier_time_seconds: f64,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LifecycleEvent {
    pub sequence: u64,
    pub from_state: Option<CandidateState>,
    pub to_state: CandidateState,
    pub reason: String,
    pub evidence: Vec<EvidenceReference>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Candidate {
    pub schema_version: String,
    pub api_version: String,
    pub kind: String,
    pub candidate_id: String,
    pub seed: u64,
    pub genome_hash: ArtifactDigest,
    pub parent_candidate_ids: Vec<String>,
    pub transformation_ids: Vec<String>,
    pub state: CandidateState,
    pub lifecycle: Vec<LifecycleEvent>,
    pub budget: SearchBudget,
    pub usage: BudgetUsage,
    pub extensions: Extensions,
}

impl Validate for Candidate {
    fn validate(&self) -> Result<(), ValidationError> {
        validate_document_header(
            &self.schema_version,
            &self.api_version,
            &self.kind,
            "Candidate",
        )?;
        nonempty("candidate_id", &self.candidate_id)?;
        self.genome_hash.validate_at("genome_hash")?;
        if self.lifecycle.is_empty() {
            return Err(ValidationError::new("lifecycle", "must not be empty"));
        }
        for (index, event) in self.lifecycle.iter().enumerate() {
            let expected_sequence = u64::try_from(index).map_err(|_| {
                ValidationError::new(
                    "lifecycle",
                    "contains more events than the wire type permits",
                )
            })?;
            if event.sequence != expected_sequence {
                return Err(ValidationError::new(
                    format!("lifecycle[{index}].sequence"),
                    "must be contiguous from zero",
                ));
            }
            if index == 0 {
                if event.from_state.is_some() || event.to_state != CandidateState::Proposed {
                    return Err(ValidationError::new(
                        "lifecycle[0]",
                        "must begin at PROPOSED",
                    ));
                }
            } else if event.from_state != Some(self.lifecycle[index - 1].to_state) {
                return Err(ValidationError::new(
                    format!("lifecycle[{index}].from_state"),
                    "transition is discontinuous",
                ));
            }
            if index > 0 {
                let previous = self.lifecycle[index - 1].to_state;
                let Some(previous_rank) = previous.success_rank() else {
                    return Err(ValidationError::new(
                        format!("lifecycle[{index}]"),
                        "candidate failure states are terminal",
                    ));
                };
                if let Some(next_rank) = event.to_state.success_rank()
                    && next_rank != previous_rank + 1
                {
                    return Err(ValidationError::new(
                        format!("lifecycle[{index}].to_state"),
                        "success stages cannot be skipped or reversed",
                    ));
                }
            }
        }
        if self.lifecycle.last().map(|event| event.to_state) != Some(self.state) {
            return Err(ValidationError::new(
                "state",
                "must equal final lifecycle state",
            ));
        }
        let over_budget = self.usage.wall_time_seconds > self.budget.wall_time_seconds
            || self.usage.cpu_time_seconds > self.budget.cpu_time_seconds
            || self.usage.gpu_time_seconds > self.budget.gpu_time_seconds
            || self.usage.cloud_cost_usd > self.budget.cloud_cost_usd
            || self.usage.external_synthesis_cost_usd > self.budget.external_synthesis_cost_usd
            || self.usage.candidate_count > self.budget.candidate_count
            || self.usage.compilation_count > self.budget.compilation_count
            || self.usage.benchmark_count > self.budget.benchmark_count
            || self.usage.verifier_time_seconds > self.budget.verifier_time_seconds;
        if over_budget {
            return Err(ValidationError::new(
                "usage",
                "exceeds declared search budget",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TensorInputCase {
    pub shape: Vec<u64>,
    pub strides: Vec<i64>,
    pub dtype: Precision,
    pub values_hex: String,
    pub non_contiguous: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RequestEventCase {
    pub at_step: u64,
    pub request_id: String,
    pub action: RequestEventAction,
    pub worker_id: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TopologyCase {
    pub hosts: u64,
    pub devices_per_host: u64,
    pub failed_links: Vec<String>,
    pub degraded_links: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DependencyCase {
    pub package: String,
    pub version: String,
    pub hardware_architecture: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceCase {
    pub device_bytes: u64,
    pub host_bytes: u64,
    pub queue_depth: u64,
    pub process_count: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CounterexamplePayload {
    Tensor { input: TensorInputCase },
    RequestTrace { events: Vec<RequestEventCase> },
    Topology { topology: TopologyCase },
    Dependency { dependency: DependencyCase },
    Resource { resource: ResourceCase },
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReproductionCommand {
    pub executable: String,
    pub arguments: Vec<String>,
    pub timeout_seconds: u64,
    pub seed: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EnvironmentFact {
    pub name: String,
    pub value: String,
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BehaviorObservation {
    pub description: String,
    pub artifact: Option<EvidenceReference>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Counterexample {
    pub schema_version: String,
    pub api_version: String,
    pub kind: String,
    pub counterexample_id: String,
    pub candidate_id: String,
    pub transformation_id: Option<String>,
    pub violated_contract: String,
    pub scope: CounterexampleScope,
    pub payload: CounterexamplePayload,
    pub reproduction: ReproductionCommand,
    pub environment: Vec<EnvironmentFact>,
    pub expected: BehaviorObservation,
    pub observed: BehaviorObservation,
    pub minimized: bool,
    pub parent_counterexample_id: Option<String>,
    pub lineage_references: Vec<LineageReference>,
    pub extensions: Extensions,
}

impl Validate for Counterexample {
    fn validate(&self) -> Result<(), ValidationError> {
        validate_document_header(
            &self.schema_version,
            &self.api_version,
            &self.kind,
            "Counterexample",
        )?;
        nonempty("counterexample_id", &self.counterexample_id)?;
        nonempty("candidate_id", &self.candidate_id)?;
        nonempty("violated_contract", &self.violated_contract)?;
        nonempty("reproduction.executable", &self.reproduction.executable)?;
        if self.reproduction.timeout_seconds == 0 {
            return Err(ValidationError::new(
                "reproduction.timeout_seconds",
                "must be positive",
            ));
        }
        match &self.payload {
            CounterexamplePayload::Tensor { input } if input.shape.is_empty() => Err(
                ValidationError::new("payload.input.shape", "must not be empty"),
            ),
            CounterexamplePayload::RequestTrace { events } if events.is_empty() => {
                Err(ValidationError::new("payload.events", "must not be empty"))
            }
            _ => Ok(()),
        }
    }
}
