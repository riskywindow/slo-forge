//! Canonical portable logical, physical, capsule, and protocol wire models.

use std::collections::{BTreeMap, BTreeSet};

use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

use crate::{ProtocolError, Validate, ValidationError, canonical_hash};

pub const SCHEMA_VERSION: &str = "1.0.0";
pub const API_VERSION: &str = "sloforge.io/continuum/v1";
const MAX_COMPONENTS: usize = 16_384;
const MAX_SEGMENTS: usize = 1_000_000;

fn nonempty(path: &str, value: &str) -> Result<(), ValidationError> {
    if value.trim().is_empty() {
        Err(ValidationError::new(path, "must not be empty"))
    } else {
        Ok(())
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn valid_extension_key(key: &str) -> bool {
    let Some((namespace, name)) = key.split_once('/') else {
        return false;
    };
    let mut namespace_chars = namespace.chars();
    let namespace_valid = namespace_chars
        .next()
        .is_some_and(|character| character.is_ascii_lowercase())
        && namespace_chars.all(|character| {
            character.is_ascii_lowercase()
                || character.is_ascii_digit()
                || matches!(character, '.' | '-')
        });
    let mut name_chars = name.chars();
    namespace_valid
        && name_chars
            .next()
            .is_some_and(|character| character.is_ascii_alphabetic())
        && name_chars.all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '_' | '.' | '-')
        })
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, JsonSchema)]
#[serde(transparent)]
pub struct Extensions(pub BTreeMap<String, Value>);

impl<'de> Deserialize<'de> for Extensions {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let values = BTreeMap::<String, Value>::deserialize(deserializer)?;
        if let Some(key) = values.keys().find(|key| !valid_extension_key(key)) {
            return Err(serde::de::Error::custom(format!(
                "extension key {key:?} must be namespace-qualified"
            )));
        }
        Ok(Self(values))
    }
}

macro_rules! string_enum {
    ($name:ident { $($variant:ident),+ $(,)? }) => {
        #[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, Ord, PartialEq, PartialOrd, Serialize)]
        #[serde(rename_all = "snake_case")]
        pub enum $name { $($variant),+ }
    };
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Digest {
    pub algorithm: String,
    pub value: String,
}

impl Digest {
    fn validate_at(&self, path: &str) -> Result<(), ValidationError> {
        if self.algorithm != "sha256" || !is_sha256(&self.value) {
            return Err(ValidationError::new(
                path,
                "must be a lowercase SHA-256 digest",
            ));
        }
        Ok(())
    }

    fn new(value: String) -> Self {
        Self {
            algorithm: "sha256".into(),
            value,
        }
    }
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Provenance {
    pub producer: String,
    pub producer_version: String,
    pub source_uri: Option<String>,
    pub source_digest: Option<Digest>,
    pub captured_at: String,
    pub raw_evidence_uri: Option<String>,
    #[serde(default)]
    pub extensions: Extensions,
}

string_enum!(ExactnessClass {
    ExactBitwise,
    ExactSemantic,
    NumericallyEquivalent,
    QualityBounded,
    RecomputationAssisted,
    Incompatible
});
string_enum!(StateKind {
    TokenHistory,
    AttentionKv,
    Recurrent,
    StateSpace,
    Convolutional,
    Speculative,
    Sampler,
    GuidedDecoding,
    Workflow,
    ClientDelivery,
    Unknown
});
string_enum!(StateLifetime {
    Request,
    Session,
    Workflow,
    Checkpoint
});
string_enum!(OwnershipScope {
    SessionOwner,
    ImmutableShared,
    CopyOnWrite,
    ExternalCoordinator
});
string_enum!(ConversionPermission {
    ExactRelayout,
    DtypeConversion,
    Quantization,
    Recompute,
    OpaqueCopy
});
string_enum!(RecomputationPermission {
    Forbidden,
    FromTokenHistory,
    FromCheckpoint,
    ModelSpecific
});
string_enum!(DTypeSemantics {
    Bool,
    Uint8,
    Int8,
    Int32,
    Int64,
    Float16,
    Bfloat16,
    Float32,
    Float64,
    Fp8,
    Opaque
});

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StateComponentDescriptor {
    pub semantic_id: String,
    pub schema_version: String,
    pub kind: StateKind,
    pub symbolic_shape: Vec<String>,
    pub dtype_semantics: DTypeSemantics,
    pub update_semantics: String,
    pub lifetime: StateLifetime,
    pub ownership: OwnershipScope,
    pub exactness_requirement: ExactnessClass,
    pub conversion_permissions: Vec<ConversionPermission>,
    pub recomputation_permission: RecomputationPermission,
    pub compatibility_fingerprint: Digest,
    pub integrity_hash: Digest,
    pub provenance: Vec<Provenance>,
    #[serde(default)]
    pub extensions: Extensions,
}

impl StateComponentDescriptor {
    fn validate_at(&self, path: &str) -> Result<(), ValidationError> {
        nonempty(&format!("{path}.semantic_id"), &self.semantic_id)?;
        if self.exactness_requirement == ExactnessClass::Incompatible {
            return Err(ValidationError::new(
                format!("{path}.exactness_requirement"),
                "captured state cannot require incompatible exactness",
            ));
        }
        if self.provenance.is_empty() {
            return Err(ValidationError::new(
                format!("{path}.provenance"),
                "must not be empty",
            ));
        }
        let unique: BTreeSet<_> = self.conversion_permissions.iter().copied().collect();
        if unique.len() != self.conversion_permissions.len() {
            return Err(ValidationError::new(
                format!("{path}.conversion_permissions"),
                "contains duplicates",
            ));
        }
        self.compatibility_fingerprint
            .validate_at(&format!("{path}.compatibility_fingerprint"))?;
        self.integrity_hash
            .validate_at(&format!("{path}.integrity_hash"))
    }
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionIdentity {
    pub session_id: String,
    pub request_id: String,
    pub workflow_id: Option<String>,
    pub tenant_id: String,
    pub model_identity: Digest,
    pub tokenizer_identity: Digest,
    pub adapter_identity: Option<Digest>,
    pub creation_epoch: u64,
    pub current_owner_epoch: u64,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TokenHistoryState {
    pub component: StateComponentDescriptor,
    pub input_token_ids: Vec<u64>,
    pub committed_output_token_ids: Vec<u64>,
    #[serde(default)]
    pub uncommitted_speculative_tokens: Vec<u64>,
    #[serde(default)]
    pub token_positions: Vec<u64>,
    #[serde(default)]
    pub position_offset: u64,
    pub attention_mask_semantics: String,
    pub tokenizer_fingerprint: Digest,
    pub normalization_contract: String,
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TokenRange {
    pub start: u64,
    pub end_exclusive: u64,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AttentionLayerState {
    pub layer_identity: String,
    pub logical_k_shape: Vec<u64>,
    pub logical_v_shape: Vec<u64>,
    pub token_range: TokenRange,
    pub head_count: u64,
    pub kv_head_count: u64,
    pub head_dimension: u64,
    pub positional_encoding_semantics: String,
    pub attention_window_semantics: String,
    pub dtype_semantics: DTypeSemantics,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AttentionState {
    pub component: StateComponentDescriptor,
    pub layers: Vec<AttentionLayerState>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RecurrentState {
    pub component: StateComponentDescriptor,
    pub state_identifier: String,
    pub layer_identity: String,
    pub logical_shape: Vec<u64>,
    pub update_semantics: String,
    pub dtype: DTypeSemantics,
    pub sequence_position: u64,
    pub initialization_contract: String,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SpeculativeState {
    pub component: StateComponentDescriptor,
    pub draft_model_identity: Digest,
    pub verifier_model_identity: Digest,
    pub accepted_prefix: Vec<u64>,
    pub pending_draft_tokens: Vec<u64>,
    pub rng_state: String,
    pub verification_cursor: u64,
    pub rollback_boundary: u64,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SamplerState {
    pub component: StateComponentDescriptor,
    pub sampling_algorithm: String,
    pub seed: u64,
    pub rng_algorithm: String,
    pub rng_counter: u64,
    pub temperature: f64,
    pub top_k: u64,
    pub top_p: f64,
    pub repetition_penalty: f64,
    pub frequency_penalty: f64,
    pub presence_penalty: f64,
    pub deterministic_required: bool,
    pub implementation_independent_state: Option<String>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GuidedDecodingState {
    pub component: StateComponentDescriptor,
    pub automaton_identity: Digest,
    pub current_automaton_state: String,
    pub tokenizer_contract: Digest,
    pub accepted_prefix: Vec<u64>,
    pub pending_constraint_state: Option<String>,
}

string_enum!(SideEffectClass {
    None,
    Idempotent,
    AtMostOnceExternal,
    NonReplayable
});

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ToolResult {
    pub call_id: String,
    pub result_digest: Digest,
    pub side_effect_class: SideEffectClass,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PendingToolCall {
    pub call_id: String,
    pub tool_identity: String,
    pub arguments_digest: Digest,
    pub side_effect_class: SideEffectClass,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkflowState {
    pub component: StateComponentDescriptor,
    pub current_node: String,
    pub branch_identity: String,
    #[serde(default)]
    pub completed_tool_results: Vec<ToolResult>,
    #[serde(default)]
    pub pending_tool_calls: Vec<PendingToolCall>,
    pub side_effect_class: SideEffectClass,
    pub workflow_deadline: Option<String>,
    pub continuation_contract: String,
}

string_enum!(TerminalStatus {
    Open,
    Completed,
    Cancelled,
    Errored
});

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ClientDeliveryState {
    pub component: StateComponentDescriptor,
    pub last_generated_token_index: i64,
    pub last_gateway_committed_token_index: i64,
    pub last_client_acknowledged_token_index: Option<i64>,
    pub stream_owner_epoch: u64,
    pub terminal_status: TerminalStatus,
    pub error_state: Option<String>,
}

string_enum!(UnknownStateHandling {
    Reject,
    PreserveOpaque,
    IgnoreReconstructible
});

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UnknownStateComponent {
    pub component: StateComponentDescriptor,
    pub namespace: String,
    pub type_name: String,
    pub type_version: String,
    pub required_for_resume: bool,
    pub portable_opaque: bool,
    pub payload_digest: Option<Digest>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StateDependencyNode {
    pub component_id: String,
    pub state_producing_fingerprint: Digest,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StateDependencyEdge {
    pub upstream_component_id: String,
    pub downstream_component_id: String,
    pub dependency_semantics: String,
    pub invalidated_by_weight_change: bool,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StateDependencyGraph {
    pub nodes: Vec<StateDependencyNode>,
    pub edges: Vec<StateDependencyEdge>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QualityContract {
    pub metric: String,
    pub maximum_loss: f64,
    pub evaluation_contract: String,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LogicalStateSchema {
    pub schema_version: String,
    pub api_version: String,
    pub kind: String,
    pub execution: ExecutionIdentity,
    pub token_history: TokenHistoryState,
    pub attention: Option<AttentionState>,
    #[serde(default)]
    pub recurrent: Vec<RecurrentState>,
    pub speculative: Option<SpeculativeState>,
    pub sampler: SamplerState,
    pub guided_decoding: Option<GuidedDecodingState>,
    pub workflow: Option<WorkflowState>,
    pub client_delivery: ClientDeliveryState,
    pub dependency_graph: StateDependencyGraph,
    pub unknown_state_handling: UnknownStateHandling,
    #[serde(default)]
    pub unknown_components: Vec<UnknownStateComponent>,
    pub exactness_contract: ExactnessClass,
    pub quality_contract: Option<QualityContract>,
    #[serde(default)]
    pub extensions: Extensions,
}

impl LogicalStateSchema {
    fn components(&self) -> Vec<&StateComponentDescriptor> {
        let mut values = vec![
            &self.token_history.component,
            &self.sampler.component,
            &self.client_delivery.component,
        ];
        if let Some(attention) = &self.attention {
            values.push(&attention.component);
        }
        values.extend(self.recurrent.iter().map(|state| &state.component));
        if let Some(speculative) = &self.speculative {
            values.push(&speculative.component);
        }
        if let Some(guided) = &self.guided_decoding {
            values.push(&guided.component);
        }
        if let Some(workflow) = &self.workflow {
            values.push(&workflow.component);
        }
        values.extend(self.unknown_components.iter().map(|state| &state.component));
        values
    }
}

impl Validate for LogicalStateSchema {
    #[allow(clippy::too_many_lines)]
    fn validate(&self) -> Result<(), ValidationError> {
        if self.schema_version != SCHEMA_VERSION
            || self.api_version != API_VERSION
            || self.kind != "LogicalStateSchema"
        {
            return Err(ValidationError::new(
                "logical_state",
                "unsupported identity/version",
            ));
        }
        if self.execution.current_owner_epoch == 0 {
            return Err(ValidationError::new(
                "execution.current_owner_epoch",
                "must be positive",
            ));
        }
        self.execution
            .model_identity
            .validate_at("execution.model_identity")?;
        self.execution
            .tokenizer_identity
            .validate_at("execution.tokenizer_identity")?;
        if self.token_history.component.kind != StateKind::TokenHistory {
            return Err(ValidationError::new(
                "token_history.component.kind",
                "must be token_history",
            ));
        }
        if self.sampler.component.kind != StateKind::Sampler
            || self.client_delivery.component.kind != StateKind::ClientDelivery
            || self
                .attention
                .as_ref()
                .is_some_and(|state| state.component.kind != StateKind::AttentionKv)
            || self.recurrent.iter().any(|state| {
                !matches!(
                    state.component.kind,
                    StateKind::Recurrent | StateKind::StateSpace
                )
            })
            || self
                .speculative
                .as_ref()
                .is_some_and(|state| state.component.kind != StateKind::Speculative)
            || self
                .guided_decoding
                .as_ref()
                .is_some_and(|state| state.component.kind != StateKind::GuidedDecoding)
            || self
                .workflow
                .as_ref()
                .is_some_and(|state| state.component.kind != StateKind::Workflow)
        {
            return Err(ValidationError::new(
                "components.kind",
                "typed state wrapper does not match component kind",
            ));
        }
        if self.token_history.tokenizer_fingerprint != self.execution.tokenizer_identity {
            return Err(ValidationError::new(
                "token_history.tokenizer_fingerprint",
                "does not match execution tokenizer",
            ));
        }
        let expected_positions = self.token_history.input_token_ids.len()
            + self.token_history.committed_output_token_ids.len();
        if !self.token_history.token_positions.is_empty()
            && self.token_history.token_positions.len() != expected_positions
        {
            return Err(ValidationError::new(
                "token_history.token_positions",
                "must cover input and committed output",
            ));
        }
        if self.client_delivery.stream_owner_epoch != self.execution.current_owner_epoch {
            return Err(ValidationError::new(
                "client_delivery.stream_owner_epoch",
                "does not match execution owner epoch",
            ));
        }
        if self.client_delivery.last_gateway_committed_token_index
            > self.client_delivery.last_generated_token_index
        {
            return Err(ValidationError::new(
                "client_delivery.last_gateway_committed_token_index",
                "cannot exceed generated watermark",
            ));
        }
        if matches!(
            self.exactness_contract,
            ExactnessClass::QualityBounded | ExactnessClass::NumericallyEquivalent
        ) && self.quality_contract.is_none()
        {
            return Err(ValidationError::new(
                "quality_contract",
                "required for quality-bounded or numerically-equivalent exactness",
            ));
        }
        let components = self.components();
        if components.len() > MAX_COMPONENTS {
            return Err(ValidationError::new(
                "components",
                "exceeds bounded component count",
            ));
        }
        let mut component_ids = BTreeSet::new();
        for (index, component) in components.iter().enumerate() {
            component.validate_at(&format!("components[{index}]"))?;
            if component.exactness_requirement == ExactnessClass::NumericallyEquivalent
                && self.quality_contract.is_none()
            {
                return Err(ValidationError::new(
                    "quality_contract",
                    "required for numerically-equivalent components",
                ));
            }
            if !component_ids.insert(component.semantic_id.as_str()) {
                return Err(ValidationError::new(
                    "components",
                    "semantic IDs must be unique",
                ));
            }
        }
        let graph_ids: BTreeSet<_> = self
            .dependency_graph
            .nodes
            .iter()
            .map(|node| node.component_id.as_str())
            .collect();
        if graph_ids != component_ids || graph_ids.len() != self.dependency_graph.nodes.len() {
            return Err(ValidationError::new(
                "dependency_graph.nodes",
                "must exactly cover logical state components",
            ));
        }
        for edge in &self.dependency_graph.edges {
            if !graph_ids.contains(edge.upstream_component_id.as_str())
                || !graph_ids.contains(edge.downstream_component_id.as_str())
            {
                return Err(ValidationError::new(
                    "dependency_graph.edges",
                    "references unknown component",
                ));
            }
        }
        let mut indegree: BTreeMap<&str, usize> = graph_ids
            .iter()
            .copied()
            .map(|identifier| (identifier, 0))
            .collect();
        let mut adjacency: BTreeMap<&str, Vec<&str>> = graph_ids
            .iter()
            .copied()
            .map(|identifier| (identifier, Vec::new()))
            .collect();
        for edge in &self.dependency_graph.edges {
            if let Some(value) = indegree.get_mut(edge.downstream_component_id.as_str()) {
                *value += 1;
            }
            adjacency
                .entry(edge.upstream_component_id.as_str())
                .or_default()
                .push(edge.downstream_component_id.as_str());
        }
        let mut ready: Vec<_> = indegree
            .iter()
            .filter_map(|(identifier, degree)| (*degree == 0).then_some(*identifier))
            .collect();
        let mut visited = 0_usize;
        while let Some(identifier) = ready.pop() {
            visited += 1;
            if let Some(targets) = adjacency.get(identifier) {
                for target in targets {
                    if let Some(degree) = indegree.get_mut(target) {
                        *degree -= 1;
                        if *degree == 0 {
                            ready.push(target);
                        }
                    }
                }
            }
        }
        if visited != graph_ids.len() {
            return Err(ValidationError::new("dependency_graph", "must be acyclic"));
        }
        for state in &self.unknown_components {
            if state.component.kind != StateKind::Unknown {
                return Err(ValidationError::new(
                    "unknown_components.component.kind",
                    "must be unknown",
                ));
            }
            if state.required_for_resume
                && (self.unknown_state_handling != UnknownStateHandling::PreserveOpaque
                    || !state.portable_opaque
                    || state.payload_digest.is_none())
            {
                return Err(ValidationError::new(
                    "unknown_components",
                    "required unknown state must be portable opaque state",
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeIdentity {
    pub runtime_name: String,
    pub runtime_version: String,
    pub adapter_version: String,
    pub build_hash: Digest,
    pub dependency_versions: Vec<String>,
    pub target_hardware: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ByteRange {
    pub offset: u64,
    pub length: u64,
}

impl ByteRange {
    fn end(&self) -> Option<u64> {
        self.offset.checked_add(self.length)
    }
}

string_enum!(CompressionKind {
    None,
    Zstd,
    Lz4,
    RuntimeSpecific
});
string_enum!(EncryptionKind {
    None,
    Aes256Gcm,
    Chacha20Poly1305
});

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StorageLocation {
    pub memory_type: String,
    pub host_id: String,
    pub device_id: Option<String>,
    pub numa_domain: Option<u64>,
    pub memory_tier: String,
    pub network_rail: Option<String>,
    pub fault_domain: String,
}

string_enum!(LayoutKind {
    Contiguous,
    Paged,
    Blocked,
    Interleaved,
    Transposed,
    Tiled,
    RuntimeSpecific
});
string_enum!(Ordering {
    TokenMajor,
    HeadMajor,
    LayerMajor,
    BlockMajor,
    RuntimeSpecific
});
string_enum!(KvPacking {
    Separate,
    PackedKv,
    InterleavedKv
});

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LayoutDescriptor {
    pub layout_id: String,
    pub kind: LayoutKind,
    pub page_size_bytes: Option<u64>,
    pub block_size: Option<u64>,
    pub alignment_bytes: u64,
    #[serde(default)]
    pub padding_bytes: u64,
    pub ordering: Ordering,
    pub k_v_packing: KvPacking,
    pub runtime_layout_name: Option<String>,
    #[serde(default)]
    pub extensions: Extensions,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ShardDescriptor {
    pub shard_id: String,
    pub tensor_parallel_degree: u64,
    pub pipeline_stage: u64,
    pub expert_parallel_group: u64,
    pub data_parallel_replica: u64,
    pub rank: u64,
    pub source_logical_slice: ByteRange,
    pub destination_logical_slice: ByteRange,
    pub shard_order: u64,
    #[serde(default)]
    pub replicated: bool,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PlacementDescriptor {
    pub placement_id: String,
    pub location: StorageLocation,
    pub nic_id: Option<String>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QuantizationDescriptor {
    pub quantization_id: String,
    pub format: String,
    pub scale_granularity: Option<String>,
    #[serde(default)]
    pub zero_point: bool,
    pub metadata_layout: Option<String>,
    pub accumulation_semantics: String,
    pub exactness_class: ExactnessClass,
    pub quality_contract: Option<QualityContract>,
}

string_enum!(AccessPatternKind {
    AppendOnly,
    Mutable,
    ReadOnly,
    LayerSequential,
    SlidingWindow,
    RandomAccess,
    SparseAccess
});

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AccessPatternDescriptor {
    pub access_pattern_id: String,
    pub kind: AccessPatternKind,
    pub required_before_resume: bool,
    pub streamable_before_use: bool,
    pub recomputable: bool,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StateSegment {
    pub logical_state_reference: String,
    pub segment_id: String,
    pub logical_byte_range: ByteRange,
    pub physical_byte_range: ByteRange,
    pub tensor_shape: Vec<u64>,
    pub tensor_strides: Vec<u64>,
    pub storage_offset: u64,
    pub allocation_id: String,
    #[serde(default)]
    pub page_ids: Vec<String>,
    pub chunk_ids: Vec<String>,
    pub current_version: u64,
    pub dirty_epoch: u64,
    pub checksum: Digest,
    pub compression: CompressionKind,
    pub encryption: EncryptionKind,
    pub layout_id: String,
    pub shard_id: String,
    pub placement_id: String,
    pub quantization_id: Option<String>,
    pub access_pattern_id: String,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PageTableEntry {
    pub logical_token_range: TokenRange,
    pub physical_page_id: String,
    pub page_version: u64,
    pub owner_epoch: u64,
    pub dirty: bool,
    pub copy_on_write_reference_count: u64,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PageTableDescriptor {
    pub segment_id: String,
    pub entries: Vec<PageTableEntry>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LogicalComponentSize {
    pub component_id: String,
    pub logical_size_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PhysicalStateLayout {
    pub schema_version: String,
    pub api_version: String,
    pub kind: String,
    pub layout_id: String,
    pub runtime: RuntimeIdentity,
    pub physical_plan_hash: Digest,
    pub owner_epoch: u64,
    pub logical_component_sizes: Vec<LogicalComponentSize>,
    pub layout_descriptors: Vec<LayoutDescriptor>,
    pub shard_descriptors: Vec<ShardDescriptor>,
    pub placement_descriptors: Vec<PlacementDescriptor>,
    #[serde(default)]
    pub quantization_descriptors: Vec<QuantizationDescriptor>,
    pub access_patterns: Vec<AccessPatternDescriptor>,
    pub segments: Vec<StateSegment>,
    #[serde(default)]
    pub page_tables: Vec<PageTableDescriptor>,
    #[serde(default)]
    pub reconstructible_runtime_state: Vec<String>,
    #[serde(default)]
    pub extensions: Extensions,
}

fn unique_strings<'a>(values: impl IntoIterator<Item = &'a str>) -> bool {
    let mut unique = BTreeSet::new();
    values.into_iter().all(|value| unique.insert(value))
}

impl Validate for PhysicalStateLayout {
    #[allow(clippy::too_many_lines)]
    fn validate(&self) -> Result<(), ValidationError> {
        if self.schema_version != SCHEMA_VERSION
            || self.api_version != API_VERSION
            || self.kind != "PhysicalStateLayout"
        {
            return Err(ValidationError::new(
                "physical_state",
                "unsupported identity/version",
            ));
        }
        if self.owner_epoch == 0 || self.segments.len() > MAX_SEGMENTS {
            return Err(ValidationError::new(
                "physical_state",
                "invalid owner epoch or segment bound",
            ));
        }
        self.runtime.build_hash.validate_at("runtime.build_hash")?;
        self.physical_plan_hash.validate_at("physical_plan_hash")?;
        if self.shard_descriptors.iter().any(|shard| {
            shard.tensor_parallel_degree == 0 || shard.rank >= shard.tensor_parallel_degree
        }) {
            return Err(ValidationError::new(
                "shard_descriptors.rank",
                "must be smaller than a positive tensor-parallel degree",
            ));
        }
        if self.quantization_descriptors.iter().any(|descriptor| {
            descriptor.exactness_class == ExactnessClass::Incompatible
                || (descriptor.exactness_class == ExactnessClass::ExactBitwise
                    && !matches!(descriptor.format.as_str(), "none" | "identity"))
                || (descriptor.exactness_class == ExactnessClass::QualityBounded
                    && descriptor.quality_contract.is_none())
        }) {
            return Err(ValidationError::new(
                "quantization_descriptors",
                "contains an invalid exactness or quality contract",
            ));
        }
        if !unique_strings(
            self.logical_component_sizes
                .iter()
                .map(|item| item.component_id.as_str()),
        ) || !unique_strings(
            self.layout_descriptors
                .iter()
                .map(|item| item.layout_id.as_str()),
        ) || !unique_strings(
            self.shard_descriptors
                .iter()
                .map(|item| item.shard_id.as_str()),
        ) || !unique_strings(
            self.placement_descriptors
                .iter()
                .map(|item| item.placement_id.as_str()),
        ) || !unique_strings(
            self.quantization_descriptors
                .iter()
                .map(|item| item.quantization_id.as_str()),
        ) || !unique_strings(
            self.access_patterns
                .iter()
                .map(|item| item.access_pattern_id.as_str()),
        ) || !unique_strings(self.segments.iter().map(|item| item.segment_id.as_str()))
            || !unique_strings(self.page_tables.iter().map(|item| item.segment_id.as_str()))
        {
            return Err(ValidationError::new(
                "physical_state",
                "identifiers must be unique",
            ));
        }
        let sizes: BTreeMap<_, _> = self
            .logical_component_sizes
            .iter()
            .map(|item| (item.component_id.as_str(), item.logical_size_bytes))
            .collect();
        let layouts: BTreeSet<_> = self
            .layout_descriptors
            .iter()
            .map(|item| item.layout_id.as_str())
            .collect();
        let shards: BTreeMap<_, _> = self
            .shard_descriptors
            .iter()
            .map(|item| (item.shard_id.as_str(), item))
            .collect();
        let placements: BTreeSet<_> = self
            .placement_descriptors
            .iter()
            .map(|item| item.placement_id.as_str())
            .collect();
        let quantization: BTreeSet<_> = self
            .quantization_descriptors
            .iter()
            .map(|item| item.quantization_id.as_str())
            .collect();
        let access: BTreeSet<_> = self
            .access_patterns
            .iter()
            .map(|item| item.access_pattern_id.as_str())
            .collect();
        let mut coverage: BTreeMap<&str, Vec<&ByteRange>> = sizes
            .keys()
            .copied()
            .map(|component| (component, Vec::new()))
            .collect();
        for segment in &self.segments {
            let Some(shard) = shards.get(segment.shard_id.as_str()) else {
                return Err(ValidationError::new(
                    "segments.shard_id",
                    "references unknown shard",
                ));
            };
            if !sizes.contains_key(segment.logical_state_reference.as_str())
                || !layouts.contains(segment.layout_id.as_str())
                || !placements.contains(segment.placement_id.as_str())
                || !access.contains(segment.access_pattern_id.as_str())
                || segment
                    .quantization_id
                    .as_deref()
                    .is_some_and(|identifier| !quantization.contains(identifier))
            {
                return Err(ValidationError::new(
                    "segments",
                    "references an unknown descriptor",
                ));
            }
            if segment.logical_byte_range != shard.source_logical_slice {
                return Err(ValidationError::new(
                    "segments.logical_byte_range",
                    "must equal shard source slice",
                ));
            }
            if shard.source_logical_slice.length != shard.destination_logical_slice.length {
                return Err(ValidationError::new(
                    "shards.destination_logical_slice",
                    "exact source and destination slices must have equal length",
                ));
            }
            if segment.tensor_shape.len() != segment.tensor_strides.len()
                || !unique_strings(segment.page_ids.iter().map(String::as_str))
                || !unique_strings(segment.chunk_ids.iter().map(String::as_str))
            {
                return Err(ValidationError::new(
                    "segments",
                    "invalid shape or identifiers",
                ));
            }
            segment.checksum.validate_at("segments.checksum")?;
            if shard.data_parallel_replica == 0 {
                coverage
                    .entry(segment.logical_state_reference.as_str())
                    .or_default()
                    .push(&segment.logical_byte_range);
            }
        }
        for (component, ranges) in &mut coverage {
            ranges.sort_by_key(|range| range.offset);
            let mut cursor = 0_u64;
            for byte_range in ranges {
                if byte_range.offset != cursor {
                    return Err(ValidationError::new(
                        "segments.logical_byte_range",
                        format!("primary coverage gap or overlap for {component}"),
                    ));
                }
                cursor = byte_range.end().ok_or_else(|| {
                    ValidationError::new("segments.logical_byte_range", "range overflows u64")
                })?;
            }
            if Some(&cursor) != sizes.get(component) {
                return Err(ValidationError::new(
                    "segments.logical_byte_range",
                    format!("incomplete primary coverage for {component}"),
                ));
            }
        }
        let mut allocation_ranges: BTreeMap<&str, Vec<&ByteRange>> = BTreeMap::new();
        for segment in &self.segments {
            allocation_ranges
                .entry(segment.allocation_id.as_str())
                .or_default()
                .push(&segment.physical_byte_range);
        }
        for ranges in allocation_ranges.values_mut() {
            ranges.sort_by_key(|range| range.offset);
            let mut previous_end = 0_u64;
            for byte_range in ranges {
                if byte_range.length > 0 && byte_range.offset < previous_end {
                    return Err(ValidationError::new(
                        "segments.physical_byte_range",
                        "physical allocation ranges overlap",
                    ));
                }
                previous_end = previous_end.max(byte_range.end().ok_or_else(|| {
                    ValidationError::new("segments.physical_byte_range", "range overflows u64")
                })?);
            }
        }
        let segment_map: BTreeMap<_, _> = self
            .segments
            .iter()
            .map(|segment| (segment.segment_id.as_str(), segment))
            .collect();
        for table in &self.page_tables {
            let Some(segment) = segment_map.get(table.segment_id.as_str()) else {
                return Err(ValidationError::new(
                    "page_tables.segment_id",
                    "unknown segment",
                ));
            };
            let expected: BTreeSet<_> = segment.page_ids.iter().map(String::as_str).collect();
            let actual: BTreeSet<_> = table
                .entries
                .iter()
                .map(|entry| entry.physical_page_id.as_str())
                .collect();
            if expected != actual || actual.len() != table.entries.len() {
                return Err(ValidationError::new(
                    "page_tables.entries",
                    "must cover page IDs exactly",
                ));
            }
            let mut entries: Vec<_> = table.entries.iter().collect();
            entries.sort_by_key(|entry| entry.logical_token_range.start);
            let mut previous_end = None;
            for entry in entries {
                if entry.page_version != segment.current_version {
                    return Err(ValidationError::new(
                        "page_tables.page_version",
                        "stale page version",
                    ));
                }
                if entry.owner_epoch != self.owner_epoch {
                    return Err(ValidationError::new(
                        "page_tables.owner_epoch",
                        "owner epoch mismatch",
                    ));
                }
                if entry.logical_token_range.end_exclusive < entry.logical_token_range.start
                    || previous_end.is_some_and(|end| end != entry.logical_token_range.start)
                {
                    return Err(ValidationError::new("page_tables", "token gap or overlap"));
                }
                previous_end = Some(entry.logical_token_range.end_exclusive);
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalChunkReference {
    pub chunk_id: String,
    pub content_hash: Digest,
    pub size_bytes: u64,
    pub tenant_security_domain: String,
    pub storage_uri: String,
    pub compression: CompressionKind,
    pub encryption: EncryptionKind,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SegmentManifest {
    pub segment_id: String,
    pub segment_hash: Digest,
    pub chunks: Vec<ExternalChunkReference>,
}

string_enum!(CapsuleType {
    Complete,
    Incremental,
    Fork,
    Rollback,
    Migration,
    RecomputationAssisted
});

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CapsuleIdentity {
    pub capsule_id: String,
    pub capsule_type: CapsuleType,
    pub session_id: String,
    pub tenant_id: String,
    pub model_hash: Digest,
    pub tokenizer_hash: Digest,
    pub adapter_hash: Option<Digest>,
    pub source_runtime: RuntimeIdentity,
    pub source_physical_plan: Digest,
    pub owner_epoch: u64,
    pub capture_timestamp: String,
    pub git_commit: String,
    pub continuum_version: String,
    pub parent_capsule_id: Option<String>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CompatibilityConstraints {
    pub source_compatibility_fingerprint: Digest,
    pub required_destination_capabilities: Vec<String>,
    #[serde(default)]
    pub prohibited_conversions: Vec<ConversionPermission>,
    #[serde(default)]
    pub recomputation_permissions: Vec<RecomputationPermission>,
    pub quality_loss_budget: Option<f64>,
    #[serde(default)]
    pub architecture_restrictions: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OwnershipLease {
    pub session_id: String,
    pub owner_runtime: String,
    pub owner_epoch: u64,
    pub fencing_token: u64,
    pub expiration: String,
    pub coordinator_version: u64,
    pub last_committed_state_version: u64,
    pub last_committed_token_index: i64,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CapsuleTransactionBinding {
    pub transaction_id: Option<String>,
    pub ownership_lease: OwnershipLease,
    pub fencing_token: u64,
    pub source_epoch: u64,
    pub destination_epoch: Option<u64>,
    pub commit_watermark: i64,
    pub rollback_boundary: i64,
    pub pending_dirty_log_hash: Option<Digest>,
    pub transaction_journal_hash: Digest,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct VerificationClaim {
    pub claim_id: String,
    pub property: String,
    pub scope: String,
    pub result: String,
    pub evidence_digest: Digest,
    #[serde(default)]
    pub assumptions: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MigrationVerificationEvidence {
    pub schema_version: String,
    pub api_version: String,
    pub kind: String,
    pub evidence_id: String,
    pub transaction_id: Option<String>,
    pub generated_at: String,
    pub capture_consistency: Vec<VerificationClaim>,
    pub segment_integrity: Vec<VerificationClaim>,
    #[serde(default)]
    pub conversion_verification: Vec<VerificationClaim>,
    #[serde(default)]
    pub continuation_verification: Vec<VerificationClaim>,
    #[serde(default)]
    pub protocol_verification: Vec<VerificationClaim>,
    pub model_check_scope: Option<String>,
    #[serde(default)]
    pub benchmark_provenance: Vec<Provenance>,
    #[serde(default)]
    pub known_limitations: Vec<String>,
    #[serde(default)]
    pub extensions: Extensions,
}

impl Validate for MigrationVerificationEvidence {
    fn validate(&self) -> Result<(), ValidationError> {
        if self.schema_version != SCHEMA_VERSION
            || self.api_version != API_VERSION
            || self.kind != "MigrationVerificationEvidence"
        {
            return Err(ValidationError::new(
                "evidence",
                "unsupported identity/version",
            ));
        }
        for claim in self
            .capture_consistency
            .iter()
            .chain(&self.segment_integrity)
            .chain(&self.conversion_verification)
            .chain(&self.continuation_verification)
            .chain(&self.protocol_verification)
        {
            if !matches!(claim.result.as_str(), "pass" | "fail" | "not_exercised") {
                return Err(ValidationError::new(
                    "evidence.claim.result",
                    "invalid result",
                ));
            }
            claim
                .evidence_digest
                .validate_at("evidence.claim.evidence_digest")?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CapsuleIntegrity {
    pub identity_hash: Digest,
    pub logical_state_hash: Digest,
    pub physical_layout_hash: Digest,
    pub segment_manifests_hash: Digest,
    pub compatibility_hash: Digest,
    pub transaction_binding_hash: Digest,
    pub evidence_hash: Digest,
    pub extensions_hash: Digest,
    pub merkle_root: Digest,
}

#[derive(Serialize)]
#[allow(clippy::struct_field_names)]
struct IntegrityInput<'a> {
    compatibility_hash: &'a str,
    evidence_hash: &'a str,
    extensions_hash: &'a str,
    identity_hash: &'a str,
    logical_state_hash: &'a str,
    physical_layout_hash: &'a str,
    segment_manifests_hash: &'a str,
    transaction_binding_hash: &'a str,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionStateCapsule {
    pub schema_version: String,
    pub api_version: String,
    pub kind: String,
    pub identity: CapsuleIdentity,
    pub logical_state: LogicalStateSchema,
    pub physical_state: PhysicalStateLayout,
    pub segment_manifests: Vec<SegmentManifest>,
    pub compatibility: CompatibilityConstraints,
    pub transaction: CapsuleTransactionBinding,
    pub evidence: MigrationVerificationEvidence,
    pub integrity: CapsuleIntegrity,
    #[serde(default)]
    pub extensions: Extensions,
}

impl ExecutionStateCapsule {
    /// Recompute the layered hashes committed by a capsule.
    ///
    /// # Errors
    ///
    /// Returns an error if a capsule component cannot be represented by the
    /// canonical JSON profile.
    pub fn computed_integrity(&self) -> Result<CapsuleIntegrity, ProtocolError> {
        let mut identity = serde_json::to_value(&self.identity)?;
        identity
            .as_object_mut()
            .ok_or_else(|| {
                ProtocolError::Validation(ValidationError::new(
                    "identity",
                    "must serialize as an object",
                ))
            })?
            .remove("capsule_id");
        let identity_hash = canonical_hash(&identity)?;
        let logical_state_hash = canonical_hash(&self.logical_state)?;
        let physical_layout_hash = canonical_hash(&self.physical_state)?;
        let segment_manifests_hash = canonical_hash(&self.segment_manifests)?;
        let compatibility_hash = canonical_hash(&self.compatibility)?;
        let transaction_binding_hash = canonical_hash(&self.transaction)?;
        let evidence_hash = canonical_hash(&self.evidence)?;
        let extensions_hash = canonical_hash(&self.extensions)?;
        let root = canonical_hash(&IntegrityInput {
            compatibility_hash: &compatibility_hash,
            evidence_hash: &evidence_hash,
            extensions_hash: &extensions_hash,
            identity_hash: &identity_hash,
            logical_state_hash: &logical_state_hash,
            physical_layout_hash: &physical_layout_hash,
            segment_manifests_hash: &segment_manifests_hash,
            transaction_binding_hash: &transaction_binding_hash,
        })?;
        Ok(CapsuleIntegrity {
            identity_hash: Digest::new(identity_hash),
            logical_state_hash: Digest::new(logical_state_hash),
            physical_layout_hash: Digest::new(physical_layout_hash),
            segment_manifests_hash: Digest::new(segment_manifests_hash),
            compatibility_hash: Digest::new(compatibility_hash),
            transaction_binding_hash: Digest::new(transaction_binding_hash),
            evidence_hash: Digest::new(evidence_hash),
            extensions_hash: Digest::new(extensions_hash),
            merkle_root: Digest::new(root),
        })
    }
}

impl Validate for ExecutionStateCapsule {
    #[allow(clippy::too_many_lines)]
    fn validate(&self) -> Result<(), ValidationError> {
        if self.schema_version != SCHEMA_VERSION
            || self.api_version != API_VERSION
            || self.kind != "ExecutionStateCapsule"
        {
            return Err(ValidationError::new(
                "capsule",
                "unsupported identity/version",
            ));
        }
        self.logical_state.validate()?;
        self.physical_state.validate()?;
        self.evidence.validate()?;
        if !is_sha256(&self.identity.capsule_id) {
            return Err(ValidationError::new(
                "identity.capsule_id",
                "must be SHA-256",
            ));
        }
        if self.identity.session_id != self.logical_state.execution.session_id
            || self.identity.tenant_id != self.logical_state.execution.tenant_id
            || self.identity.model_hash != self.logical_state.execution.model_identity
            || self.identity.tokenizer_hash != self.logical_state.execution.tokenizer_identity
            || self.identity.adapter_hash != self.logical_state.execution.adapter_identity
            || self.identity.source_runtime != self.physical_state.runtime
            || self.identity.source_physical_plan != self.physical_state.physical_plan_hash
        {
            return Err(ValidationError::new(
                "identity",
                "does not match logical execution",
            ));
        }
        let epochs = [
            self.identity.owner_epoch,
            self.logical_state.execution.current_owner_epoch,
            self.logical_state.client_delivery.stream_owner_epoch,
            self.physical_state.owner_epoch,
            self.transaction.source_epoch,
            self.transaction.ownership_lease.owner_epoch,
        ];
        if epochs.contains(&0) || !epochs.windows(2).all(|pair| pair[0] == pair[1]) {
            return Err(ValidationError::new(
                "identity.owner_epoch",
                "inconsistent owner epochs",
            ));
        }
        if self.transaction.fencing_token != self.transaction.ownership_lease.fencing_token
            || self.transaction.ownership_lease.session_id != self.identity.session_id
            || self.transaction.ownership_lease.owner_runtime
                != self.physical_state.runtime.runtime_name
            || self.transaction.ownership_lease.last_committed_token_index
                != self.transaction.commit_watermark
            || self.transaction.rollback_boundary > self.transaction.commit_watermark
            || self.evidence.transaction_id != self.transaction.transaction_id
        {
            return Err(ValidationError::new(
                "transaction",
                "lease or fencing mismatch",
            ));
        }
        if self.transaction.commit_watermark
            != self
                .logical_state
                .client_delivery
                .last_gateway_committed_token_index
        {
            return Err(ValidationError::new(
                "transaction.commit_watermark",
                "client state mismatch",
            ));
        }
        let segments: BTreeMap<_, _> = self
            .physical_state
            .segments
            .iter()
            .map(|segment| (segment.segment_id.as_str(), segment))
            .collect();
        let logical_components: BTreeSet<_> = self
            .logical_state
            .components()
            .into_iter()
            .map(|component| component.semantic_id.as_str())
            .collect();
        let physical_components: BTreeSet<_> = self
            .physical_state
            .logical_component_sizes
            .iter()
            .map(|component| component.component_id.as_str())
            .collect();
        if logical_components != physical_components {
            return Err(ValidationError::new(
                "physical_state.logical_component_sizes",
                "must exactly cover logical state components",
            ));
        }
        let manifests: BTreeMap<_, _> = self
            .segment_manifests
            .iter()
            .map(|manifest| (manifest.segment_id.as_str(), manifest))
            .collect();
        if segments.len() != self.physical_state.segments.len()
            || manifests.len() != self.segment_manifests.len()
            || segments.keys().ne(manifests.keys())
        {
            return Err(ValidationError::new(
                "segment_manifests",
                "must exactly cover physical segments",
            ));
        }
        for (segment_id, segment) in &segments {
            let Some(manifest) = manifests.get(segment_id) else {
                return Err(ValidationError::new("segment_manifests", "missing segment"));
            };
            if manifest.segment_hash != segment.checksum {
                return Err(ValidationError::new(
                    "segment_manifests.segment_hash",
                    "altered segment",
                ));
            }
            let chunks: BTreeSet<_> = segment.chunk_ids.iter().map(String::as_str).collect();
            let manifest_chunks: BTreeSet<_> = manifest
                .chunks
                .iter()
                .map(|chunk| chunk.chunk_id.as_str())
                .collect();
            if chunks != manifest_chunks || manifest_chunks.len() != manifest.chunks.len() {
                return Err(ValidationError::new(
                    "segment_manifests.chunks",
                    "chunk coverage mismatch",
                ));
            }
        }
        let expected = self
            .computed_integrity()
            .map_err(|error| ValidationError::new("integrity", error.to_string()))?;
        if expected != self.integrity || self.identity.capsule_id != expected.merkle_root.value {
            return Err(ValidationError::new(
                "integrity",
                "capsule Merkle integrity mismatch",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CompatibilityCheck {
    pub check_id: String,
    pub subject: String,
    pub result: String,
    pub explanation: String,
    #[serde(default)]
    pub evidence: Vec<Digest>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RequiredConversion {
    pub component_id: String,
    pub operation: String,
    pub exactness_class: ExactnessClass,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RecomputeRequirement {
    pub component_id: String,
    pub from_component_ids: Vec<String>,
    pub reason: String,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CompatibilityReport {
    pub schema_version: String,
    pub api_version: String,
    pub kind: String,
    pub report_id: String,
    pub source_capsule_id: String,
    pub destination_runtime: RuntimeIdentity,
    pub destination_physical_plan: Digest,
    pub compatibility_class: ExactnessClass,
    pub checks: Vec<CompatibilityCheck>,
    pub rejected_classes: Vec<ExactnessClass>,
    #[serde(default)]
    pub required_conversions: Vec<RequiredConversion>,
    #[serde(default)]
    pub required_recomputation: Vec<RecomputeRequirement>,
    #[serde(default)]
    pub unsupported_state: Vec<String>,
    #[serde(default)]
    pub quality_implications: Vec<QualityContract>,
    #[serde(default)]
    pub verification_obligations: Vec<String>,
    #[serde(default)]
    pub migration_restrictions: Vec<String>,
    #[serde(default)]
    pub extensions: Extensions,
}

impl Validate for CompatibilityReport {
    fn validate(&self) -> Result<(), ValidationError> {
        if self.schema_version != SCHEMA_VERSION
            || self.api_version != API_VERSION
            || self.kind != "CompatibilityReport"
        {
            return Err(ValidationError::new(
                "compatibility",
                "unsupported identity/version",
            ));
        }
        if !is_sha256(&self.source_capsule_id) {
            return Err(ValidationError::new("source_capsule_id", "must be SHA-256"));
        }
        Ok(())
    }
}

string_enum!(TransformationOpKind {
    Slice,
    Concatenate,
    Split,
    Reshape,
    Permute,
    Transpose,
    Pad,
    Unpad,
    Interleave,
    Deinterleave,
    Pack,
    Unpack,
    Shard,
    Reshard,
    Replicate,
    Gather,
    Scatter,
    PageRemap,
    PageCoalesce,
    PageSplit,
    DtypeConvert,
    Quantize,
    Dequantize,
    Compress,
    Decompress,
    Checksum,
    Encrypt,
    Decrypt,
    Copy,
    ZeroFill,
    ReconstructMetadata,
    Recompute,
    Send,
    Receive,
    WriteDestination,
    Validate
});

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TransformCost {
    pub estimated_duration_ms: f64,
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub peak_memory_bytes: u64,
    pub estimate_source: String,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StateContractRef {
    pub component_id: String,
    pub logical_range: ByteRange,
    pub dtype: DTypeSemantics,
    pub shape: Vec<u64>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TransformationAttributes {
    #[serde(default)]
    pub axes: Vec<u64>,
    #[serde(default)]
    pub permutation: Vec<u64>,
    #[serde(default)]
    pub target_shape: Vec<u64>,
    #[serde(default)]
    pub padding_before: Vec<u64>,
    #[serde(default)]
    pub padding_after: Vec<u64>,
    pub page_size_bytes: Option<u64>,
    pub shard_count: Option<u64>,
    pub target_dtype: Option<DTypeSemantics>,
    pub codec: Option<String>,
    pub transport_id: Option<String>,
    pub checksum_algorithm: Option<String>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StateTransformationOperation {
    pub operation_id: String,
    pub kind: TransformationOpKind,
    pub dependencies: Vec<String>,
    pub sources: Vec<StateContractRef>,
    pub destinations: Vec<StateContractRef>,
    pub preconditions: Vec<String>,
    pub postconditions: Vec<String>,
    pub exactness_class: ExactnessClass,
    pub attributes: TransformationAttributes,
    pub ownership_behavior: String,
    pub target_device: String,
    pub estimated_cost: TransformCost,
    pub streamable: bool,
    pub verification_obligations: Vec<String>,
    pub fallback_implementation: String,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryAllocation {
    pub allocation_id: String,
    pub memory_type: String,
    pub size_bytes: u64,
    pub lifetime_start_operation: String,
    pub lifetime_end_operation: String,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StateTransformationIr {
    pub schema_version: String,
    pub api_version: String,
    pub kind: String,
    pub transformation_id: String,
    pub source_layout_hash: Digest,
    pub destination_layout_hash: Digest,
    pub compatibility_report_hash: Digest,
    pub operations: Vec<StateTransformationOperation>,
    pub memory_plan: Vec<MemoryAllocation>,
    pub maximum_buffer_bytes: u64,
    pub chunk_order: Vec<String>,
    pub rollback_behavior: String,
    pub predicted_duration_ms: f64,
    pub uncertainty_ms: f64,
    #[serde(default)]
    pub extensions: Extensions,
}

impl Validate for StateTransformationIr {
    fn validate(&self) -> Result<(), ValidationError> {
        if self.schema_version != SCHEMA_VERSION
            || self.api_version != API_VERSION
            || self.kind != "StateTransformationIR"
        {
            return Err(ValidationError::new(
                "transformation",
                "unsupported identity/version",
            ));
        }
        let mut known = BTreeSet::new();
        for operation in &self.operations {
            if operation
                .dependencies
                .iter()
                .any(|dependency| !known.contains(dependency))
            {
                return Err(ValidationError::new(
                    "operations",
                    "must be topologically ordered",
                ));
            }
            if !known.insert(operation.operation_id.clone()) {
                return Err(ValidationError::new(
                    "operations",
                    "operation IDs must be unique",
                ));
            }
        }
        if self
            .chunk_order
            .iter()
            .any(|operation| !known.contains(operation))
        {
            return Err(ValidationError::new(
                "chunk_order",
                "references unknown operation",
            ));
        }
        let memory: u64 = self
            .memory_plan
            .iter()
            .try_fold(0_u64, |total, allocation| {
                total.checked_add(allocation.size_bytes)
            })
            .ok_or_else(|| ValidationError::new("memory_plan", "size overflow"))?;
        if memory > self.maximum_buffer_bytes {
            return Err(ValidationError::new(
                "memory_plan",
                "exceeds bounded buffer",
            ));
        }
        Ok(())
    }
}

string_enum!(MigrationStrategy {
    StopAndCopy,
    PreCopy,
    HybridPreCopy,
    RecomputationAssisted,
    ConstrainedLazy,
    Fork,
    Clone
});

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TransportSelection {
    pub transport_id: String,
    pub chunk_size_bytes: u64,
    pub concurrency: u64,
    pub bandwidth_limit_bytes_per_second: Option<u64>,
    pub deadline_ms: u64,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MigrationPlan {
    pub schema_version: String,
    pub api_version: String,
    pub kind: String,
    pub plan_id: String,
    pub source_session_id: String,
    pub destination_runtime: RuntimeIdentity,
    pub destination_physical_plan: Digest,
    pub strategy: MigrationStrategy,
    pub exactness_requirement: ExactnessClass,
    pub conversion_plan_hash: Digest,
    pub transport: TransportSelection,
    pub pre_copy_rounds: u64,
    pub cutover_threshold_bytes: u64,
    pub destination_warmup_actions: Vec<String>,
    pub validation_actions: Vec<String>,
    pub rollback_capsule_id: Option<String>,
    pub expected_interruption_ms: f64,
    pub expected_total_time_ms: f64,
    pub expected_source_overhead_ms: f64,
    pub expected_destination_overhead_ms: f64,
    pub expected_bytes: u64,
    pub expected_temporary_memory_bytes: u64,
    pub expected_cost_usd: f64,
    pub failure_probability: f64,
    pub uncertainty_ms: f64,
    pub rejected_alternatives: Vec<String>,
    pub required_before_resume_segments: Vec<String>,
    #[serde(default)]
    pub extensions: Extensions,
}

impl Validate for MigrationPlan {
    fn validate(&self) -> Result<(), ValidationError> {
        if self.schema_version != SCHEMA_VERSION
            || self.api_version != API_VERSION
            || self.kind != "MigrationPlan"
        {
            return Err(ValidationError::new(
                "migration_plan",
                "unsupported identity/version",
            ));
        }
        if self.transport.chunk_size_bytes == 0
            || self.transport.concurrency == 0
            || self.transport.deadline_ms == 0
        {
            return Err(ValidationError::new("transport", "bounds must be positive"));
        }
        if self.strategy == MigrationStrategy::ConstrainedLazy
            && self.required_before_resume_segments.is_empty()
        {
            return Err(ValidationError::new(
                "required_before_resume_segments",
                "required for constrained lazy migration",
            ));
        }
        Ok(())
    }
}

string_enum!(TransactionPhase {
    Proposed,
    CompatibilityValidated,
    DestinationPreparing,
    Precopying,
    DeltaSyncing,
    CutoverRequested,
    SourceQuiescing,
    SourceFrozen,
    FinalDeltaTransferring,
    DestinationImporting,
    DestinationValidating,
    CommitIntentRecorded,
    OwnershipCommitted,
    GatewaySwitching,
    DestinationActive,
    SourceDraining,
    Completed,
    Rejected,
    Aborting,
    RolledBack,
    FailedBeforeCommit,
    FailedAfterCommit,
    DestinationLost,
    SourceLost,
    CoordinatorUnavailable,
    OperatorRequired
});

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TransactionAcknowledgment {
    pub actor: String,
    pub phase: TransactionPhase,
    pub owner_epoch: u64,
    pub state_hash: Digest,
    pub recorded_at: String,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StateTransaction {
    pub schema_version: String,
    pub api_version: String,
    pub kind: String,
    pub transaction_id: String,
    pub source_owner: String,
    pub destination_candidate: String,
    pub source_epoch: u64,
    pub proposed_destination_epoch: u64,
    pub fencing_token: u64,
    pub migration_plan_hash: Digest,
    pub current_phase: TransactionPhase,
    pub commit_watermark: i64,
    pub rollback_watermark: i64,
    pub state_hashes: Vec<Digest>,
    pub acknowledgments: Vec<TransactionAcknowledgment>,
    pub timeout_at: String,
    pub failure_reason: Option<String>,
    pub journal_hash: Digest,
    #[serde(default)]
    pub extensions: Extensions,
}

impl Validate for StateTransaction {
    fn validate(&self) -> Result<(), ValidationError> {
        if self.schema_version != SCHEMA_VERSION
            || self.api_version != API_VERSION
            || self.kind != "StateTransaction"
        {
            return Err(ValidationError::new(
                "transaction",
                "unsupported identity/version",
            ));
        }
        if self.source_epoch == 0
            || self.proposed_destination_epoch <= self.source_epoch
            || self.fencing_token == 0
        {
            return Err(ValidationError::new(
                "transaction",
                "invalid owner epoch or fencing token",
            ));
        }
        if self.rollback_watermark > self.commit_watermark {
            return Err(ValidationError::new(
                "rollback_watermark",
                "exceeds commit watermark",
            ));
        }
        Ok(())
    }
}
