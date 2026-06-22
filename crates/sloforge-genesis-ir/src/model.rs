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
    Speculative,
    Tool,
    Workflow
});
string_enum!(StateOwnership {
    Request,
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
    Float32,
    Bfloat16,
    Float16,
    Fp8,
    Int8,
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
string_enum!(CollectiveKind {
    AllReduce,
    AllGather,
    ReduceScatter,
    AllToAll,
    SendRecv
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
    pub relation: String,
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
    fn validate_at(&self, path: &str) -> Result<(), ValidationError> {
        nonempty(&format!("{path}.stable_id"), &self.stable_id)?;
        nonempty(
            &format!("{path}.semantic_contract.contract_id"),
            &self.semantic_contract.contract_id,
        )?;
        if self.legal_rewrite_rules.is_empty() && !self.frozen {
            return Err(ValidationError::new(
                format!("{path}.legal_rewrite_rules"),
                "mutable nodes must declare legal rewrite rules",
            ));
        }
        if self.proof_obligations.is_empty() {
            return Err(ValidationError::new(
                format!("{path}.proof_obligations"),
                "must not be empty",
            ));
        }
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
    pub offload_tier: String,
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
    pub prefill_decode_transfer: String,
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
    pub state_transfer: String,
    pub active_stream_behavior: String,
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
        if self.serving.speculative_decoding && self.serving.draft_model_id.is_none() {
            return Err(ValidationError::new(
                "serving.draft_model_id",
                "required when speculative decoding is enabled",
            ));
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
        }
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
        }
        for (index, state) in self.state.states.iter().enumerate() {
            state
                .node
                .validate_at(&format!("state.states[{index}].node"))?;
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
        for (index, collective) in self.distributed.collective_dag.iter().enumerate() {
            collective
                .node
                .validate_at(&format!("distributed.collective_dag[{index}].node"))?;
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
        }
        for (index, placement) in self.distributed.expert_placement.iter().enumerate() {
            placement
                .node
                .validate_at(&format!("distributed.expert_placement[{index}].node"))?;
        }
        for (index, operator) in self.tensor.operators.iter().enumerate() {
            operator
                .node
                .validate_at(&format!("tensor.operators[{index}].node"))?;
        }
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
        }
        for (index, value) in self.tensor.values.iter().enumerate() {
            value
                .node
                .validate_at(&format!("tensor.values[{index}].node"))?;
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
        }
        for (index, kernel) in self.kernel.kernels.iter().enumerate() {
            kernel
                .node
                .validate_at(&format!("kernel.kernels[{index}].node"))?;
            kernel
                .source_artifact
                .digest
                .validate_at(&format!("kernel.kernels[{index}].source_artifact.digest"))?;
        }
        for (index, transition) in self.recovery.transitions.iter().enumerate() {
            transition
                .node
                .validate_at(&format!("recovery.transitions[{index}].node"))?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GenomePattern {
    pub region: String,
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
    pub action: String,
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
