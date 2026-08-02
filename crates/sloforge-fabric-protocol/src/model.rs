use std::collections::{BTreeMap, BTreeSet};

use chrono::{DateTime, Utc};
use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

use crate::{Validate, ValidationError};

pub const FABRIC_SCHEMA_VERSION: &str = "1.0.0";
pub const FABRIC_API_VERSION: &str = "sloforge.io/fabric/v1";

fn nonempty(path: &str, value: &str) -> Result<(), ValidationError> {
    if value.trim().is_empty() {
        Err(ValidationError::new(path, "must not be empty"))
    } else {
        Ok(())
    }
}

fn positive(path: &str, value: u64) -> Result<(), ValidationError> {
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
        Err(ValidationError::new(path, "must be a finite probability"))
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

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DiscoverySource {
    Sysfs,
    Hwloc,
    Nvml,
    Dcgm,
    Cuda,
    NvidiaSmi,
    Nccl,
    Ibverbs,
    Ethtool,
    Cgroup,
    Kubernetes,
    Synthetic,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FactProvenance {
    pub source: DiscoverySource,
    pub observed_at: DateTime<Utc>,
    pub confidence: f64,
    pub source_uri: Option<String>,
    pub field: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HealthState {
    Healthy,
    Degraded,
    Failed,
    Unknown,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MigState {
    Disabled,
    Enabled,
    Instance,
    Unknown,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HostNode {
    pub node_id: String,
    pub name: String,
    pub architecture: String,
    pub operating_system: String,
    pub total_memory_bytes: u64,
    pub visible_memory_bytes: u64,
    pub container_visible: bool,
    pub fault_domain: String,
    pub provenance: Vec<FactProvenance>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CpuSocketNode {
    pub node_id: String,
    pub host_id: String,
    pub socket_index: u64,
    pub model: String,
    pub physical_cores: u64,
    pub logical_cores: u64,
    pub provenance: Vec<FactProvenance>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NumaDomainNode {
    pub node_id: String,
    pub host_id: String,
    pub socket_id: String,
    pub numa_index: u64,
    pub cpu_set: String,
    pub memory_bytes: u64,
    pub provenance: Vec<FactProvenance>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryDomainNode {
    pub node_id: String,
    pub host_id: String,
    pub numa_domain_id: String,
    pub capacity_bytes: u64,
    pub measured_bandwidth_gbps: Option<f64>,
    pub provenance: Vec<FactProvenance>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GpuNode {
    pub node_id: String,
    pub host_id: String,
    pub gpu_index: u64,
    pub uuid: String,
    pub product: String,
    pub architecture: String,
    pub memory_bytes: u64,
    pub compute_capability: Option<String>,
    pub mig_state: MigState,
    pub numa_domain_id: Option<String>,
    pub pci_address: Option<String>,
    pub health: HealthState,
    pub provenance: Vec<FactProvenance>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NvSwitchNode {
    pub node_id: String,
    pub host_id: String,
    pub switch_domain: String,
    pub generation: Option<String>,
    pub health: HealthState,
    pub provenance: Vec<FactProvenance>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PcieNode {
    pub node_id: String,
    pub host_id: String,
    pub pci_address: Option<String>,
    pub generation: Option<u64>,
    pub width: Option<u64>,
    pub provenance: Vec<FactProvenance>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NicNode {
    pub node_id: String,
    pub host_id: String,
    pub interface: String,
    pub pci_address: Option<String>,
    pub speed_gbps: Option<f64>,
    pub transport: NicTransport,
    pub active: bool,
    pub rdma_capable: Option<bool>,
    pub gpu_direct_rdma: Option<bool>,
    pub numa_domain_id: Option<String>,
    pub health: HealthState,
    pub provenance: Vec<FactProvenance>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NicTransport {
    Ethernet,
    Infiniband,
    Roce,
    Unknown,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NetworkRailNode {
    pub node_id: String,
    pub name: String,
    pub transport: RailTransport,
    pub subnet: Option<String>,
    pub health: HealthState,
    pub provenance: Vec<FactProvenance>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RailTransport {
    Ethernet,
    Infiniband,
    Roce,
    Synthetic,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StorageTierNode {
    pub node_id: String,
    pub host_id: Option<String>,
    pub tier: StorageTier,
    pub capacity_bytes: Option<u64>,
    pub provenance: Vec<FactProvenance>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StorageTier {
    Object,
    RemoteFs,
    LocalNvme,
    PageCache,
    Memory,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RemoteMemoryNode {
    pub node_id: String,
    pub host_id: Option<String>,
    pub capacity_bytes: u64,
    pub protocol: String,
    pub provenance: Vec<FactProvenance>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TopologyNode {
    Host(HostNode),
    CpuSocket(CpuSocketNode),
    NumaDomain(NumaDomainNode),
    MemoryDomain(MemoryDomainNode),
    Gpu(GpuNode),
    NvSwitch(NvSwitchNode),
    PcieRootComplex(PcieNode),
    PcieSwitch(PcieNode),
    Nic(NicNode),
    NetworkRail(NetworkRailNode),
    StorageTier(StorageTierNode),
    RemoteMemory(RemoteMemoryNode),
}

impl TopologyNode {
    #[must_use]
    pub fn node_id(&self) -> &str {
        match self {
            Self::Host(node) => &node.node_id,
            Self::CpuSocket(node) => &node.node_id,
            Self::NumaDomain(node) => &node.node_id,
            Self::MemoryDomain(node) => &node.node_id,
            Self::Gpu(node) => &node.node_id,
            Self::NvSwitch(node) => &node.node_id,
            Self::PcieRootComplex(node) | Self::PcieSwitch(node) => &node.node_id,
            Self::Nic(node) => &node.node_id,
            Self::NetworkRail(node) => &node.node_id,
            Self::StorageTier(node) => &node.node_id,
            Self::RemoteMemory(node) => &node.node_id,
        }
    }
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CurvePoint {
    pub message_bytes: u64,
    pub median: f64,
    pub p95: f64,
    pub robust_dispersion: f64,
    pub confidence_low: f64,
    pub confidence_high: f64,
    pub sample_count: u64,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConnectionType {
    CpuMemory,
    CpuGpu,
    GpuGpu,
    Nvlink,
    Nvswitch,
    Pcie,
    GpuNic,
    NicNetwork,
    StorageHost,
    RemoteMemory,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Directionality {
    Unidirectional,
    Bidirectional,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Duplex {
    Half,
    Full,
    Unknown,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TopologyEdge {
    pub edge_id: String,
    pub source_node_id: String,
    pub target_node_id: String,
    pub connection: ConnectionType,
    pub directionality: Directionality,
    pub duplex: Duplex,
    pub theoretical_bandwidth_gbps: Option<f64>,
    #[serde(default)]
    pub bandwidth_curve_gbps: Vec<CurvePoint>,
    #[serde(default)]
    pub latency_curve_us: Vec<CurvePoint>,
    pub sharing_group: Option<String>,
    pub contention_domain: Option<String>,
    pub health: HealthState,
    pub measurement_confidence: Option<f64>,
    pub measured_at: Option<DateTime<Utc>>,
    pub measurement_environment_digest: Option<ArtifactDigest>,
    pub discovery_provenance: Vec<FactProvenance>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SoftwareComponent {
    pub name: String,
    pub version: String,
    pub source: DiscoverySource,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TopologyGraph {
    pub schema_version: String,
    pub api_version: String,
    pub kind: String,
    pub topology_id: String,
    pub discovered_at: DateTime<Utc>,
    pub nodes: Vec<TopologyNode>,
    pub edges: Vec<TopologyEdge>,
    #[serde(default)]
    pub software: Vec<SoftwareComponent>,
    pub container_limited: bool,
    #[serde(default)]
    pub discovery_warnings: Vec<String>,
    #[serde(default)]
    pub extensions: Extensions,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerKind {
    Embedding,
    Attention,
    FeedForward,
    Moe,
    Normalization,
    Output,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExpertSpec {
    pub expert_id: String,
    pub parameter_bytes: u64,
    pub activation_bytes_per_token: u64,
    pub expected_load: f64,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CommunicationOperation {
    AllReduce,
    AllGather,
    ReduceScatter,
    AllToAll,
    Broadcast,
    SendRecv,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ParallelismDimension {
    Tensor,
    Pipeline,
    Data,
    Expert,
    Context,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CommunicationRequirement {
    pub operation: CommunicationOperation,
    pub bytes_per_token: f64,
    pub synchronization_required: bool,
    pub parallelism_dimension: ParallelismDimension,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LayerSpec {
    pub layer_id: String,
    pub ordinal: u64,
    pub kind: LayerKind,
    pub parameter_bytes: u64,
    pub activation_bytes_per_token: u64,
    pub kv_bytes_per_token: u64,
    #[serde(default)]
    pub experts: Vec<ExpertSpec>,
    #[serde(default)]
    pub communication: Vec<CommunicationRequirement>,
    #[serde(default)]
    pub indivisible: bool,
    #[serde(default = "default_true")]
    pub allowed_stage_boundaries_after: bool,
}

const fn default_true() -> bool {
    true
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PrecisionName {
    Float32,
    Float16,
    Bfloat16,
    Fp8,
    Int8,
    Int4,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PrecisionMode {
    pub name: PrecisionName,
    pub weight_bytes: u64,
    #[serde(default)]
    pub runtime_features: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelGraph {
    pub schema_version: String,
    pub api_version: String,
    pub kind: String,
    pub model_id: String,
    pub model_revision: String,
    pub model_digest: ArtifactDigest,
    pub tokenizer_digest: ArtifactDigest,
    pub hidden_size: u64,
    pub attention_heads: u64,
    pub key_value_heads: u64,
    pub maximum_sequence_length: u64,
    pub layers: Vec<LayerSpec>,
    pub precision_modes: Vec<PrecisionMode>,
    #[serde(default)]
    pub runtime_features: Vec<String>,
    #[serde(default)]
    pub extensions: Extensions,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ParallelismKind {
    Tensor,
    Pipeline,
    Data,
    Expert,
    Context,
    Prefill,
    Decode,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ParallelGroup {
    pub group_id: String,
    pub kind: ParallelismKind,
    pub rank_ids: Vec<u64>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ParallelismPlan {
    pub tensor_parallel_degree: u64,
    pub pipeline_parallel_degree: u64,
    pub data_parallel_degree: u64,
    pub expert_parallel_degree: u64,
    #[serde(default = "one")]
    pub context_parallel_degree: u64,
    pub prefill_decode_disaggregated: bool,
    pub groups: Vec<ParallelGroup>,
    pub replica_groups: Vec<ParallelGroup>,
}

const fn one() -> u64 {
    1
}

impl ParallelismPlan {
    #[must_use]
    pub fn expected_rank_count(&self) -> u64 {
        self.tensor_parallel_degree
            .saturating_mul(self.pipeline_parallel_degree)
            .saturating_mul(self.data_parallel_degree)
            .saturating_mul(self.context_parallel_degree)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerRole {
    Aggregated,
    Prefill,
    Decode,
    Expert,
    Coordinator,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RankBinding {
    pub rank_id: u64,
    pub host_id: String,
    pub gpu_id: String,
    pub numa_domain_id: String,
    pub nic_id: Option<String>,
    pub network_rail_id: Option<String>,
    pub process_cpu_affinity: String,
    pub worker_role: WorkerRole,
    pub replica_id: String,
    pub fault_domain: String,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RankPlacement {
    pub bindings: Vec<RankBinding>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExpertAssignment {
    pub expert_id: String,
    pub rank_ids: Vec<u64>,
    pub expected_load: f64,
    pub capacity_factor: f64,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HotExpertStrategy {
    None,
    Replicate,
    Rebalance,
    ReserveCapacity,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExpertPlacement {
    pub assignments: Vec<ExpertAssignment>,
    pub hot_expert_strategy: HotExpertStrategy,
    pub maximum_replicas_per_expert: u64,
    pub rebalance_minimum_interval_seconds: f64,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CollectiveAlgorithm {
    Ring,
    Tree,
    RecursiveDoubling,
    Direct,
    Pairwise,
    Auto,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CollectiveTransport {
    SharedMemory,
    Nvlink,
    Pcie,
    Infiniband,
    Roce,
    Tcp,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CollectiveFallback {
    Serialize,
    Tcp,
    HostStaged,
    Abort,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CollectiveOperation {
    pub operation_id: String,
    pub operation: CommunicationOperation,
    pub participating_ranks: Vec<u64>,
    pub message_size_intercept_bytes: u64,
    pub message_size_bytes_per_token: f64,
    pub algorithm: CollectiveAlgorithm,
    pub transport: CollectiveTransport,
    pub channel_count: u64,
    pub rail_ids: Vec<String>,
    pub rank_order: Vec<u64>,
    pub expected_duration_us: f64,
    pub uncertainty_us: f64,
    pub overlap_window_id: Option<String>,
    #[serde(default)]
    pub depends_on: Vec<String>,
    pub fallback: CollectiveFallback,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CollectivePlan {
    pub operations: Vec<CollectiveOperation>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum KvSerializationFormat {
    Raw,
    Paged,
    Nixl,
    RuntimeNative,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum KvCacheOwner {
    Prefill,
    Decode,
    Shared,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EvictionPolicy {
    Lru,
    Clock,
    Deadline,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum KvFallback {
    HostStaged,
    Recompute,
    Reject,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct KvTransferRoute {
    pub route_id: String,
    pub producer_rank_ids: Vec<u64>,
    pub consumer_rank_ids: Vec<u64>,
    pub edge_path: Vec<String>,
    pub serialization_format: KvSerializationFormat,
    pub chunk_bytes: u64,
    pub maximum_inflight_chunks: u64,
    pub overlap_with_decode: bool,
    pub cache_owner: KvCacheOwner,
    pub eviction_policy: EvictionPolicy,
    pub retry_limit: u64,
    pub fallback: KvFallback,
    pub transport_adapter: String,
    pub expected_latency_us: f64,
    pub expected_cost_usd: f64,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct KvTransferPlan {
    pub routes: Vec<KvTransferRoute>,
    pub backpressure_limit_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RankMemoryAllocation {
    pub rank_id: u64,
    pub capacity_bytes: u64,
    pub weights_bytes: u64,
    pub kv_cache_bytes: u64,
    pub activations_bytes: u64,
    pub cuda_graph_bytes: u64,
    pub runtime_workspace_bytes: u64,
    pub communication_buffers_bytes: u64,
    pub host_pinned_buffers_bytes: u64,
    pub local_nvme_bytes: u64,
    pub remote_artifacts_bytes: u64,
    pub fragmentation_allowance_bytes: u64,
    pub safety_margin_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryPlan {
    pub allocations: Vec<RankMemoryAllocation>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceContention {
    None,
    CopyEngine,
    Hbm,
    Compute,
    Network,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FallbackSerialization {
    ComputeFirst,
    CommunicationFirst,
    CriticalPath,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OverlapWindow {
    pub window_id: String,
    pub compute_operation_id: String,
    pub communication_operation_id: String,
    pub stream: String,
    #[serde(default)]
    pub depends_on: Vec<String>,
    pub expected_overlap_fraction: f64,
    pub resource_contention: ResourceContention,
    pub fallback_serialization: FallbackSerialization,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CommunicationOverlapPlan {
    pub windows: Vec<OverlapWindow>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MetricInterval {
    pub estimate: f64,
    pub lower: f64,
    pub upper: f64,
    pub confidence: f64,
    pub unit: String,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PhysicalMetrics {
    pub p95_ttft_ms: MetricInterval,
    pub p99_tpot_ms: MetricInterval,
    pub p95_end_to_end_ms: MetricInterval,
    pub throughput_tokens_per_second: MetricInterval,
    pub goodput_tokens_per_second: MetricInterval,
    pub cost_usd_per_million_tokens: MetricInterval,
    pub availability: MetricInterval,
    pub communication_overhead_fraction: MetricInterval,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DocumentReference {
    pub kind: String,
    pub api_version: String,
    pub uri: String,
    pub digest: ArtifactDigest,
    pub uid: Option<String>,
    pub generation: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FailureExposure {
    pub fault_domain: String,
    pub affected_rank_ids: Vec<u64>,
    pub probability: f64,
    pub expected_slo_impact_ms: f64,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveryTrigger {
    pub diagnosis_code: String,
    pub minimum_confidence: f64,
    pub minimum_duration_seconds: f64,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveryVariant {
    pub variant_id: String,
    pub triggers: Vec<RecoveryTrigger>,
    pub alternate_parallelism: Option<ParallelismPlan>,
    pub alternate_rank_placement: Option<RankPlacement>,
    pub alternate_collectives: Option<CollectivePlan>,
    pub alternate_kv_transfer: Option<KvTransferPlan>,
    pub alternate_worker_ratio: Option<f64>,
    pub expected_degraded_metrics: PhysicalMetrics,
    pub transition_cost_usd: f64,
    pub transition_seconds: f64,
    pub rebuild_required: bool,
    #[serde(default)]
    pub compatibility_constraints: Vec<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OptimizerPhase {
    Feasibility,
    LowerBound,
    Placement,
    Simulation,
    Refinement,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OptimizerDecision {
    Evaluate,
    Promote,
    Reject,
    Select,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OptimizerTraceEntry {
    pub sequence: u64,
    pub candidate_id: String,
    pub phase: OptimizerPhase,
    pub decision: OptimizerDecision,
    pub reason_code: String,
    pub simulator_calls: u64,
    pub solver_time_ms: f64,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RejectedPhysicalCandidate {
    pub candidate_id: String,
    pub stage: String,
    pub reason_code: String,
    pub explanation: String,
    #[serde(default)]
    pub violated_constraints: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReproducibilityMetadata {
    pub seed: u64,
    pub generated_at: DateTime<Utc>,
    pub environment_digest: ArtifactDigest,
    pub command: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PhysicalExecutionPlan {
    pub schema_version: String,
    pub api_version: String,
    pub kind: String,
    pub plan_id: String,
    pub logical_deployment_plan: DocumentReference,
    pub model_graph_hash: ArtifactDigest,
    pub topology_fingerprint: ArtifactDigest,
    pub fabric_profile_hash: ArtifactDigest,
    pub parallelism: ParallelismPlan,
    pub rank_placement: RankPlacement,
    pub expert_placement: Option<ExpertPlacement>,
    pub collectives: CollectivePlan,
    pub kv_transfer: Option<KvTransferPlan>,
    pub memory: MemoryPlan,
    pub communication_overlap: CommunicationOverlapPlan,
    pub predicted_metrics: PhysicalMetrics,
    pub bottleneck_prediction: String,
    pub failure_exposure: Vec<FailureExposure>,
    pub optimizer_history: Vec<OptimizerTraceEntry>,
    pub rejected_alternatives: Vec<RejectedPhysicalCandidate>,
    pub recovery_variants: Vec<RecoveryVariant>,
    pub evidence: Vec<DocumentReference>,
    pub compiler_version: String,
    pub git_commit: String,
    pub reproducibility: ReproducibilityMetadata,
    #[serde(default)]
    pub extensions: Extensions,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkInvocation {
    pub argv: Vec<String>,
    pub timeout_seconds: f64,
    pub process_placement: String,
    pub environment_digest: ArtifactDigest,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FabricRawSample {
    pub duration_us: f64,
    pub bytes_transferred: u64,
    pub success: bool,
    pub failure_reason: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkPrimitive {
    Launch,
    Synchronize,
    Memory,
    Gemm,
    Copy,
    P2p,
    Collective,
    Expert,
    KvTransfer,
    Startup,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FabricMeasurementSeries {
    pub measurement_id: String,
    pub primitive: BenchmarkPrimitive,
    pub transport: String,
    pub rank_count: u64,
    pub message_bytes: u64,
    pub concurrency: u64,
    pub warmup_count: u64,
    pub samples: Vec<FabricRawSample>,
    pub summary_median_us: f64,
    pub summary_p95_us: f64,
    pub confidence_low_us: f64,
    pub confidence_high_us: f64,
    pub invocation: BenchmarkInvocation,
    pub artifact_digest: ArtifactDigest,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FabricProfile {
    pub schema_version: String,
    pub api_version: String,
    pub kind: String,
    pub profile_id: String,
    pub topology_fingerprint: ArtifactDigest,
    pub created_at: DateTime<Utc>,
    pub hardware_manifest: DocumentReference,
    pub software_manifest: DocumentReference,
    pub measurements: Vec<FabricMeasurementSeries>,
    pub raw_artifacts: Vec<DocumentReference>,
    #[serde(default)]
    pub extensions: Extensions,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RecoveryActionKind {
    StopRouting,
    DrainWorker,
    RestartWorker,
    QuarantineGpu,
    QuarantineNic,
    QuarantineRail,
    ReplaceReplica,
    ChangeRankPlacement,
    ChangeRankOrdering,
    ChangeNicAffinity,
    ChangeNumaAffinity,
    MoveExpertGroup,
    ReplicateHotExperts,
    ChangeWorkerRatio,
    SwitchKvTransport,
    SwitchCollective,
    ReduceCommunicationConcurrency,
    ReduceRequestConcurrency,
    ShedLowPriority,
    SwitchParallelism,
    SwitchAggregation,
    RebuildDeployment,
    DegradedModel,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RecoveryScope {
    RequestPath,
    WorkerLocal,
    ReplicaLocal,
    NewReplica,
    DeploymentRebuild,
    OperatorRequired,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveryAction {
    pub action_id: String,
    pub kind: RecoveryActionKind,
    pub scope: RecoveryScope,
    pub target_ids: Vec<String>,
    pub order: u64,
    pub idempotency_key: String,
    pub timeout_seconds: f64,
    pub rollback_action_id: Option<String>,
    #[serde(default)]
    pub requires_external_mutation: bool,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrafficMigrationPlan {
    pub shadow_fraction: f64,
    pub canary_fraction: f64,
    pub minimum_shadow_samples: u64,
    pub minimum_canary_samples: u64,
    pub maximum_inflight_streams_at_drain: u64,
    pub preserve_started_streams: bool,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Comparator {
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveryCriterion {
    pub metric: String,
    pub comparator: Comparator,
    pub threshold: f64,
    pub window_seconds: f64,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveryPlan {
    pub schema_version: String,
    pub api_version: String,
    pub kind: String,
    pub recovery_id: String,
    pub diagnosis: DocumentReference,
    pub physical_plan: DocumentReference,
    pub actions: Vec<RecoveryAction>,
    pub expected_slo_improvement: PhysicalMetrics,
    pub expected_cost_usd: f64,
    pub expected_disruption_seconds: f64,
    pub expected_build_seconds: f64,
    pub confidence: f64,
    pub compatibility_constraints: Vec<String>,
    pub traffic_migration: TrafficMigrationPlan,
    pub promotion_criteria: Vec<RecoveryCriterion>,
    pub rollback_criteria: Vec<RecoveryCriterion>,
    pub abort_criteria: Vec<RecoveryCriterion>,
    pub evidence: Vec<DocumentReference>,
    #[serde(default)]
    pub external_mutation_authorized: bool,
    #[serde(default)]
    pub extensions: Extensions,
}

fn validate_header(
    schema_version: &str,
    api_version: &str,
    kind: &str,
    expected_kind: &str,
) -> Result<(), ValidationError> {
    if schema_version != FABRIC_SCHEMA_VERSION
        || api_version != FABRIC_API_VERSION
        || kind != expected_kind
    {
        Err(ValidationError::new(
            "kind",
            "invalid Fabric version or document kind",
        ))
    } else {
        Ok(())
    }
}

fn validate_digest_reference(
    path: &str,
    reference: &DocumentReference,
) -> Result<(), ValidationError> {
    nonempty(&format!("{path}.kind"), &reference.kind)?;
    nonempty(&format!("{path}.uri"), &reference.uri)?;
    reference.digest.validate_at(&format!("{path}.digest"))
}

fn validate_metric(path: &str, metric: &MetricInterval) -> Result<(), ValidationError> {
    if ![
        metric.estimate,
        metric.lower,
        metric.upper,
        metric.confidence,
    ]
    .iter()
    .all(|value| value.is_finite())
        || metric.lower > metric.estimate
        || metric.estimate > metric.upper
    {
        return Err(ValidationError::new(path, "invalid metric interval"));
    }
    probability(&format!("{path}.confidence"), metric.confidence)?;
    nonempty(&format!("{path}.unit"), &metric.unit)
}

fn validate_metrics(metrics: &PhysicalMetrics) -> Result<(), ValidationError> {
    for (name, metric) in [
        ("p95_ttft_ms", &metrics.p95_ttft_ms),
        ("p99_tpot_ms", &metrics.p99_tpot_ms),
        ("p95_end_to_end_ms", &metrics.p95_end_to_end_ms),
        (
            "throughput_tokens_per_second",
            &metrics.throughput_tokens_per_second,
        ),
        (
            "goodput_tokens_per_second",
            &metrics.goodput_tokens_per_second,
        ),
        (
            "cost_usd_per_million_tokens",
            &metrics.cost_usd_per_million_tokens,
        ),
        ("availability", &metrics.availability),
        (
            "communication_overhead_fraction",
            &metrics.communication_overhead_fraction,
        ),
    ] {
        validate_metric(name, metric)?;
    }
    Ok(())
}

impl Validate for TopologyGraph {
    fn validate(&self) -> Result<(), ValidationError> {
        validate_header(
            &self.schema_version,
            &self.api_version,
            &self.kind,
            "TopologyGraph",
        )?;
        nonempty("topology_id", &self.topology_id)?;
        if self.nodes.is_empty() {
            return Err(ValidationError::new("nodes", "must not be empty"));
        }
        let node_ids: BTreeSet<_> = self.nodes.iter().map(TopologyNode::node_id).collect();
        if node_ids.len() != self.nodes.len() {
            return Err(ValidationError::new("nodes", "node IDs must be unique"));
        }
        let mut edge_ids = BTreeSet::new();
        for edge in &self.edges {
            if !edge_ids.insert(&edge.edge_id) {
                return Err(ValidationError::new("edges", "edge IDs must be unique"));
            }
            if edge.source_node_id == edge.target_node_id
                || !node_ids.contains(edge.source_node_id.as_str())
                || !node_ids.contains(edge.target_node_id.as_str())
            {
                return Err(ValidationError::new("edges", "edge has invalid endpoints"));
            }
            let measured =
                !edge.bandwidth_curve_gbps.is_empty() || !edge.latency_curve_us.is_empty();
            let metadata = edge.measurement_confidence.is_some()
                && edge.measured_at.is_some()
                && edge.measurement_environment_digest.is_some();
            if measured != metadata {
                return Err(ValidationError::new(
                    "edges",
                    "measurement curves and metadata must appear together",
                ));
            }
            if let Some(confidence) = edge.measurement_confidence {
                probability("edges.measurement_confidence", confidence)?;
            }
        }
        Ok(())
    }
}

impl Validate for ModelGraph {
    fn validate(&self) -> Result<(), ValidationError> {
        validate_header(
            &self.schema_version,
            &self.api_version,
            &self.kind,
            "ModelGraph",
        )?;
        self.model_digest.validate_at("model_digest")?;
        self.tokenizer_digest.validate_at("tokenizer_digest")?;
        if self.layers.is_empty() || self.precision_modes.is_empty() {
            return Err(ValidationError::new(
                "layers",
                "layers and precision modes must not be empty",
            ));
        }
        for (index, layer) in self.layers.iter().enumerate() {
            if layer.ordinal != u64::try_from(index).unwrap_or(u64::MAX) {
                return Err(ValidationError::new(
                    "layers.ordinal",
                    "must be contiguous from zero",
                ));
            }
            if (layer.kind == LayerKind::Moe) == layer.experts.is_empty() {
                return Err(ValidationError::new(
                    "layers.experts",
                    "experts require an MoE layer and MoE layers require experts",
                ));
            }
        }
        Ok(())
    }
}

impl Validate for FabricProfile {
    fn validate(&self) -> Result<(), ValidationError> {
        validate_header(
            &self.schema_version,
            &self.api_version,
            &self.kind,
            "FabricProfile",
        )?;
        self.topology_fingerprint
            .validate_at("topology_fingerprint")?;
        validate_digest_reference("hardware_manifest", &self.hardware_manifest)?;
        validate_digest_reference("software_manifest", &self.software_manifest)?;
        let mut identifiers = BTreeSet::new();
        for measurement in &self.measurements {
            if !identifiers.insert(&measurement.measurement_id) || measurement.samples.is_empty() {
                return Err(ValidationError::new(
                    "measurements",
                    "measurement IDs must be unique and samples non-empty",
                ));
            }
            if measurement.summary_p95_us < measurement.summary_median_us
                || measurement.confidence_low_us > measurement.summary_median_us
                || measurement.confidence_high_us < measurement.summary_median_us
            {
                return Err(ValidationError::new(
                    "measurements",
                    "invalid summary interval",
                ));
            }
            measurement
                .artifact_digest
                .validate_at("measurements.artifact_digest")?;
            for sample in &measurement.samples {
                if sample.duration_us <= 0.0
                    || !sample.duration_us.is_finite()
                    || sample.success == sample.failure_reason.is_some()
                {
                    return Err(ValidationError::new(
                        "measurements.samples",
                        "invalid raw sample",
                    ));
                }
            }
        }
        Ok(())
    }
}

fn validate_parallelism_and_placement(
    plan: &PhysicalExecutionPlan,
) -> Result<(u64, BTreeSet<u64>), ValidationError> {
    for (path, degree) in [
        (
            "tensor_parallel_degree",
            plan.parallelism.tensor_parallel_degree,
        ),
        (
            "pipeline_parallel_degree",
            plan.parallelism.pipeline_parallel_degree,
        ),
        (
            "data_parallel_degree",
            plan.parallelism.data_parallel_degree,
        ),
        (
            "expert_parallel_degree",
            plan.parallelism.expert_parallel_degree,
        ),
        (
            "context_parallel_degree",
            plan.parallelism.context_parallel_degree,
        ),
    ] {
        positive(path, degree)?;
    }
    let rank_count = plan.parallelism.expected_rank_count();
    if u64::try_from(plan.rank_placement.bindings.len()).unwrap_or(u64::MAX) != rank_count {
        return Err(ValidationError::new(
            "rank_placement",
            "rank count does not match parallelism degrees",
        ));
    }
    let ranks: BTreeSet<_> = plan
        .rank_placement
        .bindings
        .iter()
        .map(|binding| binding.rank_id)
        .collect();
    if ranks != (0..rank_count).collect() {
        return Err(ValidationError::new(
            "rank_placement",
            "rank IDs must be contiguous from zero",
        ));
    }
    if plan
        .rank_placement
        .bindings
        .iter()
        .map(|binding| &binding.gpu_id)
        .collect::<BTreeSet<_>>()
        .len()
        != plan.rank_placement.bindings.len()
    {
        return Err(ValidationError::new(
            "rank_placement",
            "GPU assignments must be unique",
        ));
    }
    Ok((rank_count, ranks))
}

fn validate_memory(
    plan: &PhysicalExecutionPlan,
    ranks: &BTreeSet<u64>,
) -> Result<(), ValidationError> {
    let memory_ranks: BTreeSet<_> = plan
        .memory
        .allocations
        .iter()
        .map(|allocation| allocation.rank_id)
        .collect();
    if &memory_ranks != ranks {
        return Err(ValidationError::new(
            "memory",
            "memory plan must cover every rank",
        ));
    }
    for allocation in &plan.memory.allocations {
        let device_total = allocation
            .weights_bytes
            .saturating_add(allocation.kv_cache_bytes)
            .saturating_add(allocation.activations_bytes)
            .saturating_add(allocation.cuda_graph_bytes)
            .saturating_add(allocation.runtime_workspace_bytes)
            .saturating_add(allocation.communication_buffers_bytes)
            .saturating_add(allocation.fragmentation_allowance_bytes)
            .saturating_add(allocation.safety_margin_bytes);
        if device_total > allocation.capacity_bytes {
            return Err(ValidationError::new(
                "memory.allocations",
                "device allocation exceeds capacity",
            ));
        }
    }
    Ok(())
}

fn validate_collectives(
    plan: &PhysicalExecutionPlan,
    ranks: &BTreeSet<u64>,
) -> Result<(), ValidationError> {
    for collective in &plan.collectives.operations {
        let participants: BTreeSet<_> = collective.participating_ranks.iter().copied().collect();
        let order: BTreeSet<_> = collective.rank_order.iter().copied().collect();
        if participants.len() < 2 || participants != order || !participants.is_subset(ranks) {
            return Err(ValidationError::new(
                "collectives",
                "collective rank membership is invalid",
            ));
        }
    }
    Ok(())
}

impl Validate for PhysicalExecutionPlan {
    fn validate(&self) -> Result<(), ValidationError> {
        validate_header(
            &self.schema_version,
            &self.api_version,
            &self.kind,
            "PhysicalExecutionPlan",
        )?;
        validate_digest_reference("logical_deployment_plan", &self.logical_deployment_plan)?;
        self.model_graph_hash.validate_at("model_graph_hash")?;
        self.topology_fingerprint
            .validate_at("topology_fingerprint")?;
        self.fabric_profile_hash
            .validate_at("fabric_profile_hash")?;
        let (_rank_count, ranks) = validate_parallelism_and_placement(self)?;
        validate_memory(self, &ranks)?;
        validate_collectives(self, &ranks)?;
        if !self
            .optimizer_history
            .windows(2)
            .all(|pair| pair[0].sequence <= pair[1].sequence)
        {
            return Err(ValidationError::new(
                "optimizer_history",
                "must be ordered by sequence",
            ));
        }
        validate_metrics(&self.predicted_metrics)
    }
}

impl Validate for RecoveryPlan {
    fn validate(&self) -> Result<(), ValidationError> {
        validate_header(
            &self.schema_version,
            &self.api_version,
            &self.kind,
            "RecoveryPlan",
        )?;
        validate_digest_reference("diagnosis", &self.diagnosis)?;
        validate_digest_reference("physical_plan", &self.physical_plan)?;
        probability("confidence", self.confidence)?;
        probability(
            "traffic_migration.shadow_fraction",
            self.traffic_migration.shadow_fraction,
        )?;
        probability(
            "traffic_migration.canary_fraction",
            self.traffic_migration.canary_fraction,
        )?;
        if self.actions.is_empty() {
            return Err(ValidationError::new("actions", "must not be empty"));
        }
        let ids: BTreeSet<_> = self
            .actions
            .iter()
            .map(|action| &action.action_id)
            .collect();
        if ids.len() != self.actions.len() {
            return Err(ValidationError::new("actions", "action IDs must be unique"));
        }
        if !self
            .actions
            .windows(2)
            .all(|pair| pair[0].order < pair[1].order)
        {
            return Err(ValidationError::new(
                "actions.order",
                "must be unique and ascending",
            ));
        }
        for action in &self.actions {
            if action.requires_external_mutation && !self.external_mutation_authorized {
                return Err(ValidationError::new(
                    "external_mutation_authorized",
                    "external mutation requires explicit authorization",
                ));
            }
            if action
                .rollback_action_id
                .as_ref()
                .is_some_and(|rollback| !ids.contains(rollback))
            {
                return Err(ValidationError::new(
                    "actions.rollback_action_id",
                    "references unknown action",
                ));
            }
        }
        validate_metrics(&self.expected_slo_improvement)
    }
}
