"""Strict, versioned wire models for the SLOForge Fabric physical IR.

The physical IR composes with the logical deployment IR through immutable
content-addressed references.  Every core object rejects unknown fields; the
only extensibility mechanism is the existing namespace-qualified
``Extensions`` type.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Final, Literal, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from sloforge.ir import ArtifactDigest, Extensions

FABRIC_SCHEMA_VERSION: Final = "1.0.0"
FABRIC_API_VERSION: Final = "sloforge.io/fabric/v1"

__all__ = [
    "FABRIC_API_VERSION",
    "FABRIC_SCHEMA_VERSION",
    "BenchmarkInvocation",
    "CollectiveOperation",
    "CollectivePlan",
    "CommunicationOverlapPlan",
    "CommunicationRequirement",
    "ConnectionType",
    "CpuSocketNode",
    "CurvePoint",
    "DiscoverySource",
    "DocumentReference",
    "ExpertAssignment",
    "ExpertPlacement",
    "ExpertSpec",
    "FabricMeasurementSeries",
    "FabricProfile",
    "FabricRawSample",
    "FactProvenance",
    "FailureExposure",
    "GpuNode",
    "HealthState",
    "HostNode",
    "KVTransferPlan",
    "KVTransferRoute",
    "LayerKind",
    "LayerSpec",
    "MemoryDomainNode",
    "MemoryPlan",
    "MetricInterval",
    "MigState",
    "ModelGraph",
    "NetworkRailNode",
    "NicNode",
    "NumaDomainNode",
    "NvSwitchNode",
    "OptimizerTraceEntry",
    "OverlapWindow",
    "ParallelGroup",
    "ParallelismKind",
    "ParallelismPlan",
    "PcieNode",
    "PhysicalExecutionPlan",
    "PhysicalMetrics",
    "PrecisionMode",
    "RankBinding",
    "RankMemoryAllocation",
    "RankPlacement",
    "RecoveryAction",
    "RecoveryActionKind",
    "RecoveryCriterion",
    "RecoveryPlan",
    "RecoveryScope",
    "RecoveryTrigger",
    "RecoveryVariant",
    "RejectedPhysicalCandidate",
    "RemoteMemoryNode",
    "ReproducibilityMetadata",
    "SoftwareComponent",
    "StorageTierNode",
    "TopologyEdge",
    "TopologyGraph",
    "TopologyNode",
    "TrafficMigrationPlan",
    "WorkerRole",
]

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
PositiveFloat = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]
Probability = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
CpuSet = Annotated[str, StringConstraints(pattern=r"^[0-9]+(?:[-,][0-9]+)*$")]


class FabricModel(BaseModel):
    """Immutable strict base for Fabric IR values."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )


class DiscoverySource(StrEnum):
    SYSFS = "sysfs"
    HWLOC = "hwloc"
    NVML = "nvml"
    DCGM = "dcgm"
    CUDA = "cuda"
    NVIDIA_SMI = "nvidia_smi"
    NCCL = "nccl"
    IBVERBS = "ibverbs"
    ETHTOOL = "ethtool"
    CGROUP = "cgroup"
    KUBERNETES = "kubernetes"
    SYNTHETIC = "synthetic"


class FactProvenance(FabricModel):
    source: DiscoverySource
    observed_at: AwareDatetime
    confidence: Probability
    source_uri: NonEmptyString | None = None
    field: NonEmptyString


class HealthState(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    UNKNOWN = "unknown"


class MigState(StrEnum):
    DISABLED = "disabled"
    ENABLED = "enabled"
    INSTANCE = "instance"
    UNKNOWN = "unknown"


class HostNode(FabricModel):
    kind: Literal["host"] = "host"
    node_id: NonEmptyString
    name: NonEmptyString
    architecture: NonEmptyString
    operating_system: NonEmptyString
    total_memory_bytes: PositiveInt
    visible_memory_bytes: PositiveInt
    container_visible: bool
    fault_domain: NonEmptyString
    provenance: tuple[FactProvenance, ...]


class CpuSocketNode(FabricModel):
    kind: Literal["cpu_socket"] = "cpu_socket"
    node_id: NonEmptyString
    host_id: NonEmptyString
    socket_index: NonNegativeInt
    model: NonEmptyString
    physical_cores: PositiveInt
    logical_cores: PositiveInt
    provenance: tuple[FactProvenance, ...]


class NumaDomainNode(FabricModel):
    kind: Literal["numa_domain"] = "numa_domain"
    node_id: NonEmptyString
    host_id: NonEmptyString
    socket_id: NonEmptyString
    numa_index: NonNegativeInt
    cpu_set: CpuSet
    memory_bytes: PositiveInt
    provenance: tuple[FactProvenance, ...]


class MemoryDomainNode(FabricModel):
    kind: Literal["memory_domain"] = "memory_domain"
    node_id: NonEmptyString
    host_id: NonEmptyString
    numa_domain_id: NonEmptyString
    capacity_bytes: PositiveInt
    measured_bandwidth_gbps: PositiveFloat | None = None
    provenance: tuple[FactProvenance, ...]


class GpuNode(FabricModel):
    kind: Literal["gpu"] = "gpu"
    node_id: NonEmptyString
    host_id: NonEmptyString
    gpu_index: NonNegativeInt
    uuid: NonEmptyString
    product: NonEmptyString
    architecture: NonEmptyString
    memory_bytes: PositiveInt
    compute_capability: NonEmptyString | None = None
    mig_state: MigState
    numa_domain_id: NonEmptyString | None = None
    pci_address: NonEmptyString | None = None
    health: HealthState
    provenance: tuple[FactProvenance, ...]


class NvSwitchNode(FabricModel):
    kind: Literal["nv_switch"] = "nv_switch"
    node_id: NonEmptyString
    host_id: NonEmptyString
    switch_domain: NonEmptyString
    generation: NonEmptyString | None = None
    health: HealthState
    provenance: tuple[FactProvenance, ...]


class PcieNode(FabricModel):
    kind: Literal["pcie_root_complex", "pcie_switch"]
    node_id: NonEmptyString
    host_id: NonEmptyString
    pci_address: NonEmptyString | None = None
    generation: PositiveInt | None = None
    width: PositiveInt | None = None
    provenance: tuple[FactProvenance, ...]


class NicNode(FabricModel):
    kind: Literal["nic"] = "nic"
    node_id: NonEmptyString
    host_id: NonEmptyString
    interface: NonEmptyString
    pci_address: NonEmptyString | None = None
    speed_gbps: PositiveFloat | None = None
    transport: Literal["ethernet", "infiniband", "roce", "unknown"]
    active: bool
    rdma_capable: bool | None = None
    gpu_direct_rdma: bool | None = None
    numa_domain_id: NonEmptyString | None = None
    health: HealthState
    provenance: tuple[FactProvenance, ...]


class NetworkRailNode(FabricModel):
    kind: Literal["network_rail"] = "network_rail"
    node_id: NonEmptyString
    name: NonEmptyString
    transport: Literal["ethernet", "infiniband", "roce", "synthetic"]
    subnet: NonEmptyString | None = None
    health: HealthState
    provenance: tuple[FactProvenance, ...]


class StorageTierNode(FabricModel):
    kind: Literal["storage_tier"] = "storage_tier"
    node_id: NonEmptyString
    host_id: NonEmptyString | None = None
    tier: Literal["object", "remote_fs", "local_nvme", "page_cache", "memory"]
    capacity_bytes: PositiveInt | None = None
    provenance: tuple[FactProvenance, ...]


class RemoteMemoryNode(FabricModel):
    kind: Literal["remote_memory"] = "remote_memory"
    node_id: NonEmptyString
    host_id: NonEmptyString | None = None
    capacity_bytes: PositiveInt
    protocol: NonEmptyString
    provenance: tuple[FactProvenance, ...]


TopologyNode = Annotated[
    HostNode
    | CpuSocketNode
    | NumaDomainNode
    | MemoryDomainNode
    | GpuNode
    | NvSwitchNode
    | PcieNode
    | NicNode
    | NetworkRailNode
    | StorageTierNode
    | RemoteMemoryNode,
    Field(discriminator="kind"),
]


class CurvePoint(FabricModel):
    message_bytes: PositiveInt
    median: PositiveFloat
    p95: PositiveFloat
    robust_dispersion: NonNegativeFloat
    confidence_low: NonNegativeFloat
    confidence_high: PositiveFloat
    sample_count: PositiveInt

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        if self.confidence_low > self.median or self.confidence_high < self.median:
            raise ValueError("confidence interval must contain median")
        if self.p95 < self.median:
            raise ValueError("p95 cannot be below median")
        return self


class ConnectionType(StrEnum):
    CPU_MEMORY = "cpu_memory"
    CPU_GPU = "cpu_gpu"
    GPU_GPU = "gpu_gpu"
    NVLINK = "nvlink"
    NVSWITCH = "nvswitch"
    PCIE = "pcie"
    GPU_NIC = "gpu_nic"
    NIC_NETWORK = "nic_network"
    STORAGE_HOST = "storage_host"
    REMOTE_MEMORY = "remote_memory"


class TopologyEdge(FabricModel):
    edge_id: NonEmptyString
    source_node_id: NonEmptyString
    target_node_id: NonEmptyString
    connection: ConnectionType
    directionality: Literal["unidirectional", "bidirectional"]
    duplex: Literal["half", "full", "unknown"]
    theoretical_bandwidth_gbps: PositiveFloat | None = None
    bandwidth_curve_gbps: tuple[CurvePoint, ...] = ()
    latency_curve_us: tuple[CurvePoint, ...] = ()
    sharing_group: NonEmptyString | None = None
    contention_domain: NonEmptyString | None = None
    health: HealthState
    measurement_confidence: Probability | None = None
    measured_at: AwareDatetime | None = None
    measurement_environment_digest: ArtifactDigest | None = None
    discovery_provenance: tuple[FactProvenance, ...]

    @model_validator(mode="after")
    def validate_measurement_metadata(self) -> Self:
        has_measurements = bool(self.bandwidth_curve_gbps or self.latency_curve_us)
        metadata = (
            self.measurement_confidence,
            self.measured_at,
            self.measurement_environment_digest,
        )
        if has_measurements and any(item is None for item in metadata):
            raise ValueError("measured curves require confidence, timestamp, and environment")
        if not has_measurements and any(item is not None for item in metadata):
            raise ValueError("measurement metadata requires a measured curve")
        return self


class SoftwareComponent(FabricModel):
    name: NonEmptyString
    version: NonEmptyString
    source: DiscoverySource


class TopologyGraph(FabricModel):
    schema_version: Literal["1.0.0"] = FABRIC_SCHEMA_VERSION
    api_version: Literal["sloforge.io/fabric/v1"] = FABRIC_API_VERSION
    kind: Literal["TopologyGraph"] = "TopologyGraph"
    topology_id: NonEmptyString
    discovered_at: AwareDatetime
    nodes: tuple[TopologyNode, ...]
    edges: tuple[TopologyEdge, ...]
    software: tuple[SoftwareComponent, ...] = ()
    container_limited: bool
    discovery_warnings: tuple[NonEmptyString, ...] = ()
    extensions: Extensions = Field(default_factory=lambda: Extensions(root={}))

    @model_validator(mode="after")
    def validate_graph(self) -> Self:
        if not self.nodes:
            raise ValueError("topology must contain nodes")
        ids = [node.node_id for node in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("topology node IDs must be unique")
        edge_ids = [edge.edge_id for edge in self.edges]
        if len(edge_ids) != len(set(edge_ids)):
            raise ValueError("topology edge IDs must be unique")
        known = set(ids)
        for edge in self.edges:
            if edge.source_node_id not in known or edge.target_node_id not in known:
                raise ValueError(f"edge {edge.edge_id} references an unknown node")
            if edge.source_node_id == edge.target_node_id:
                raise ValueError(f"edge {edge.edge_id} cannot be a self-loop")
        return self


class LayerKind(StrEnum):
    EMBEDDING = "embedding"
    ATTENTION = "attention"
    FEED_FORWARD = "feed_forward"
    MOE = "moe"
    NORMALIZATION = "normalization"
    OUTPUT = "output"


class ExpertSpec(FabricModel):
    expert_id: NonEmptyString
    parameter_bytes: PositiveInt
    activation_bytes_per_token: PositiveInt
    expected_load: Probability


class CommunicationRequirement(FabricModel):
    operation: Literal[
        "all_reduce", "all_gather", "reduce_scatter", "all_to_all", "broadcast", "send_recv"
    ]
    bytes_per_token: NonNegativeFloat
    synchronization_required: bool
    parallelism_dimension: Literal["tensor", "pipeline", "data", "expert", "context"]


class LayerSpec(FabricModel):
    layer_id: NonEmptyString
    ordinal: NonNegativeInt
    kind: LayerKind
    parameter_bytes: NonNegativeInt
    activation_bytes_per_token: NonNegativeInt
    kv_bytes_per_token: NonNegativeInt
    experts: tuple[ExpertSpec, ...] = ()
    communication: tuple[CommunicationRequirement, ...] = ()
    indivisible: bool = False
    allowed_stage_boundaries_after: bool = True

    @model_validator(mode="after")
    def validate_experts(self) -> Self:
        if self.kind is LayerKind.MOE and not self.experts:
            raise ValueError("MoE layers require experts")
        if self.kind is not LayerKind.MOE and self.experts:
            raise ValueError("only MoE layers may contain experts")
        return self


class PrecisionMode(FabricModel):
    name: Literal["float32", "float16", "bfloat16", "fp8", "int8", "int4"]
    weight_bytes: PositiveInt
    runtime_features: tuple[NonEmptyString, ...] = ()


class ModelGraph(FabricModel):
    schema_version: Literal["1.0.0"] = FABRIC_SCHEMA_VERSION
    api_version: Literal["sloforge.io/fabric/v1"] = FABRIC_API_VERSION
    kind: Literal["ModelGraph"] = "ModelGraph"
    model_id: NonEmptyString
    model_revision: NonEmptyString
    model_digest: ArtifactDigest
    tokenizer_digest: ArtifactDigest
    hidden_size: PositiveInt
    attention_heads: PositiveInt
    key_value_heads: PositiveInt
    maximum_sequence_length: PositiveInt
    layers: tuple[LayerSpec, ...]
    precision_modes: tuple[PrecisionMode, ...]
    runtime_features: tuple[NonEmptyString, ...] = ()
    extensions: Extensions = Field(default_factory=lambda: Extensions(root={}))

    @model_validator(mode="after")
    def validate_layers(self) -> Self:
        if not self.layers:
            raise ValueError("model graph must contain layers")
        ordinals = [layer.ordinal for layer in self.layers]
        if ordinals != list(range(len(ordinals))):
            raise ValueError("layer ordinals must be contiguous and ordered from zero")
        if len({layer.layer_id for layer in self.layers}) != len(self.layers):
            raise ValueError("layer IDs must be unique")
        if not self.precision_modes:
            raise ValueError("at least one precision mode is required")
        return self


class ParallelismKind(StrEnum):
    TENSOR = "tensor"
    PIPELINE = "pipeline"
    DATA = "data"
    EXPERT = "expert"
    CONTEXT = "context"
    PREFILL = "prefill"
    DECODE = "decode"


class ParallelGroup(FabricModel):
    group_id: NonEmptyString
    kind: ParallelismKind
    rank_ids: tuple[NonNegativeInt, ...]

    @model_validator(mode="after")
    def validate_members(self) -> Self:
        if not self.rank_ids or len(set(self.rank_ids)) != len(self.rank_ids):
            raise ValueError("parallel group rank IDs must be non-empty and unique")
        return self


class ParallelismPlan(FabricModel):
    tensor_parallel_degree: PositiveInt
    pipeline_parallel_degree: PositiveInt
    data_parallel_degree: PositiveInt
    expert_parallel_degree: PositiveInt
    context_parallel_degree: PositiveInt = 1
    prefill_decode_disaggregated: bool
    groups: tuple[ParallelGroup, ...]
    replica_groups: tuple[ParallelGroup, ...]

    @property
    def expected_rank_count(self) -> int:
        return (
            self.tensor_parallel_degree
            * self.pipeline_parallel_degree
            * self.data_parallel_degree
            * self.context_parallel_degree
        )

    @model_validator(mode="after")
    def validate_groups(self) -> Self:
        identifiers = [group.group_id for group in (*self.groups, *self.replica_groups)]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("parallel group IDs must be unique")
        ranks = {rank for group in self.groups for rank in group.rank_ids}
        if ranks and (max(ranks) >= self.expected_rank_count):
            raise ValueError("parallel group contains rank outside expected rank count")
        return self


class WorkerRole(StrEnum):
    AGGREGATED = "aggregated"
    PREFILL = "prefill"
    DECODE = "decode"
    EXPERT = "expert"
    COORDINATOR = "coordinator"


class RankBinding(FabricModel):
    rank_id: NonNegativeInt
    host_id: NonEmptyString
    gpu_id: NonEmptyString
    numa_domain_id: NonEmptyString
    nic_id: NonEmptyString | None = None
    network_rail_id: NonEmptyString | None = None
    process_cpu_affinity: CpuSet
    worker_role: WorkerRole
    replica_id: NonEmptyString
    fault_domain: NonEmptyString


class RankPlacement(FabricModel):
    bindings: tuple[RankBinding, ...]

    @model_validator(mode="after")
    def validate_bindings(self) -> Self:
        if not self.bindings:
            raise ValueError("rank placement cannot be empty")
        ranks = [binding.rank_id for binding in self.bindings]
        if sorted(ranks) != list(range(len(ranks))):
            raise ValueError("rank IDs must be contiguous from zero")
        if len({binding.gpu_id for binding in self.bindings}) != len(self.bindings):
            raise ValueError("a GPU may host at most one rank in the canonical plan")
        return self


class ExpertAssignment(FabricModel):
    expert_id: NonEmptyString
    rank_ids: tuple[NonNegativeInt, ...]
    expected_load: Probability
    capacity_factor: PositiveFloat

    @model_validator(mode="after")
    def validate_ranks(self) -> Self:
        if not self.rank_ids or len(set(self.rank_ids)) != len(self.rank_ids):
            raise ValueError("expert rank IDs must be non-empty and unique")
        return self


class ExpertPlacement(FabricModel):
    assignments: tuple[ExpertAssignment, ...]
    hot_expert_strategy: Literal["none", "replicate", "rebalance", "reserve_capacity"]
    maximum_replicas_per_expert: PositiveInt
    rebalance_minimum_interval_seconds: PositiveFloat


class CollectiveOperation(FabricModel):
    operation_id: NonEmptyString
    operation: Literal[
        "all_reduce", "all_gather", "reduce_scatter", "broadcast", "send_recv", "all_to_all"
    ]
    participating_ranks: tuple[NonNegativeInt, ...]
    message_size_intercept_bytes: NonNegativeInt
    message_size_bytes_per_token: NonNegativeFloat
    algorithm: Literal["ring", "tree", "recursive_doubling", "direct", "pairwise", "auto"]
    transport: Literal["shared_memory", "nvlink", "pcie", "infiniband", "roce", "tcp"]
    channel_count: PositiveInt
    rail_ids: tuple[NonEmptyString, ...]
    rank_order: tuple[NonNegativeInt, ...]
    expected_duration_us: PositiveFloat
    uncertainty_us: NonNegativeFloat
    overlap_window_id: NonEmptyString | None = None
    depends_on: tuple[NonEmptyString, ...] = ()
    fallback: Literal["serialize", "tcp", "host_staged", "abort"]

    @model_validator(mode="after")
    def validate_ranks(self) -> Self:
        if len(self.participating_ranks) < 2:
            raise ValueError("collective operation requires at least two ranks")
        if set(self.rank_order) != set(self.participating_ranks):
            raise ValueError("rank order must be a permutation of participating ranks")
        return self


class CollectivePlan(FabricModel):
    operations: tuple[CollectiveOperation, ...]


class KVTransferRoute(FabricModel):
    route_id: NonEmptyString
    producer_rank_ids: tuple[NonNegativeInt, ...]
    consumer_rank_ids: tuple[NonNegativeInt, ...]
    edge_path: tuple[NonEmptyString, ...]
    serialization_format: Literal["raw", "paged", "nixl", "runtime_native"]
    chunk_bytes: PositiveInt
    maximum_inflight_chunks: PositiveInt
    overlap_with_decode: bool
    cache_owner: Literal["prefill", "decode", "shared"]
    eviction_policy: Literal["lru", "clock", "deadline"]
    retry_limit: NonNegativeInt
    fallback: Literal["host_staged", "recompute", "reject"]
    transport_adapter: NonEmptyString
    expected_latency_us: PositiveFloat
    expected_cost_usd: NonNegativeFloat


class KVTransferPlan(FabricModel):
    routes: tuple[KVTransferRoute, ...]
    backpressure_limit_bytes: PositiveInt


class RankMemoryAllocation(FabricModel):
    rank_id: NonNegativeInt
    capacity_bytes: PositiveInt
    weights_bytes: NonNegativeInt
    kv_cache_bytes: NonNegativeInt
    activations_bytes: NonNegativeInt
    cuda_graph_bytes: NonNegativeInt
    runtime_workspace_bytes: NonNegativeInt
    communication_buffers_bytes: NonNegativeInt
    host_pinned_buffers_bytes: NonNegativeInt
    local_nvme_bytes: NonNegativeInt
    remote_artifacts_bytes: NonNegativeInt
    fragmentation_allowance_bytes: NonNegativeInt
    safety_margin_bytes: NonNegativeInt

    @model_validator(mode="after")
    def validate_capacity(self) -> Self:
        device_total = (
            self.weights_bytes
            + self.kv_cache_bytes
            + self.activations_bytes
            + self.cuda_graph_bytes
            + self.runtime_workspace_bytes
            + self.communication_buffers_bytes
            + self.fragmentation_allowance_bytes
            + self.safety_margin_bytes
        )
        if device_total > self.capacity_bytes:
            raise ValueError("device memory allocation exceeds rank capacity")
        return self


class MemoryPlan(FabricModel):
    allocations: tuple[RankMemoryAllocation, ...]

    @model_validator(mode="after")
    def validate_ranks(self) -> Self:
        ranks = [allocation.rank_id for allocation in self.allocations]
        if len(ranks) != len(set(ranks)):
            raise ValueError("rank memory allocations must be unique")
        return self


class OverlapWindow(FabricModel):
    window_id: NonEmptyString
    compute_operation_id: NonEmptyString
    communication_operation_id: NonEmptyString
    stream: NonEmptyString
    depends_on: tuple[NonEmptyString, ...] = ()
    expected_overlap_fraction: Probability
    resource_contention: Literal["none", "copy_engine", "hbm", "compute", "network"]
    fallback_serialization: Literal["compute_first", "communication_first", "critical_path"]


class CommunicationOverlapPlan(FabricModel):
    windows: tuple[OverlapWindow, ...]

    @model_validator(mode="after")
    def validate_windows(self) -> Self:
        identifiers = [window.window_id for window in self.windows]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("overlap window IDs must be unique")
        return self


class MetricInterval(FabricModel):
    estimate: NonNegativeFloat
    lower: NonNegativeFloat
    upper: NonNegativeFloat
    confidence: Probability
    unit: NonEmptyString

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if self.lower > self.estimate or self.upper < self.estimate:
            raise ValueError("metric interval must contain estimate")
        return self


class PhysicalMetrics(FabricModel):
    p95_ttft_ms: MetricInterval
    p99_tpot_ms: MetricInterval
    p95_end_to_end_ms: MetricInterval
    throughput_tokens_per_second: MetricInterval
    goodput_tokens_per_second: MetricInterval
    cost_usd_per_million_tokens: MetricInterval
    availability: MetricInterval
    communication_overhead_fraction: MetricInterval


class DocumentReference(FabricModel):
    kind: NonEmptyString
    api_version: NonEmptyString
    uri: NonEmptyString
    digest: ArtifactDigest
    uid: NonEmptyString | None = None
    generation: PositiveInt | None = None


class FailureExposure(FabricModel):
    fault_domain: NonEmptyString
    affected_rank_ids: tuple[NonNegativeInt, ...]
    probability: Probability
    expected_slo_impact_ms: NonNegativeFloat


class RecoveryTrigger(FabricModel):
    diagnosis_code: NonEmptyString
    minimum_confidence: Probability
    minimum_duration_seconds: NonNegativeFloat


class RecoveryVariant(FabricModel):
    variant_id: NonEmptyString
    triggers: tuple[RecoveryTrigger, ...]
    alternate_parallelism: ParallelismPlan | None = None
    alternate_rank_placement: RankPlacement | None = None
    alternate_collectives: CollectivePlan | None = None
    alternate_kv_transfer: KVTransferPlan | None = None
    alternate_worker_ratio: PositiveFloat | None = None
    expected_degraded_metrics: PhysicalMetrics
    transition_cost_usd: NonNegativeFloat
    transition_seconds: NonNegativeFloat
    rebuild_required: bool
    compatibility_constraints: tuple[NonEmptyString, ...] = ()

    @model_validator(mode="after")
    def validate_change(self) -> Self:
        if all(
            item is None
            for item in (
                self.alternate_parallelism,
                self.alternate_rank_placement,
                self.alternate_collectives,
                self.alternate_kv_transfer,
                self.alternate_worker_ratio,
            )
        ):
            raise ValueError("recovery variant must change at least one physical plan component")
        return self


class OptimizerTraceEntry(FabricModel):
    sequence: NonNegativeInt
    candidate_id: NonEmptyString
    phase: Literal["feasibility", "lower_bound", "placement", "simulation", "refinement"]
    decision: Literal["evaluate", "promote", "reject", "select"]
    reason_code: NonEmptyString
    simulator_calls: NonNegativeInt
    solver_time_ms: NonNegativeFloat


class RejectedPhysicalCandidate(FabricModel):
    candidate_id: NonEmptyString
    stage: NonEmptyString
    reason_code: NonEmptyString
    explanation: NonEmptyString
    violated_constraints: tuple[NonEmptyString, ...] = ()


class ReproducibilityMetadata(FabricModel):
    seed: NonNegativeInt
    generated_at: AwareDatetime
    environment_digest: ArtifactDigest
    command: tuple[NonEmptyString, ...]


class PhysicalExecutionPlan(FabricModel):
    schema_version: Literal["1.0.0"] = FABRIC_SCHEMA_VERSION
    api_version: Literal["sloforge.io/fabric/v1"] = FABRIC_API_VERSION
    kind: Literal["PhysicalExecutionPlan"] = "PhysicalExecutionPlan"
    plan_id: NonEmptyString
    logical_deployment_plan: DocumentReference
    model_graph_hash: ArtifactDigest
    topology_fingerprint: ArtifactDigest
    fabric_profile_hash: ArtifactDigest
    parallelism: ParallelismPlan
    rank_placement: RankPlacement
    expert_placement: ExpertPlacement | None = None
    collectives: CollectivePlan
    kv_transfer: KVTransferPlan | None = None
    memory: MemoryPlan
    communication_overlap: CommunicationOverlapPlan
    predicted_metrics: PhysicalMetrics
    bottleneck_prediction: NonEmptyString
    failure_exposure: tuple[FailureExposure, ...]
    optimizer_history: tuple[OptimizerTraceEntry, ...]
    rejected_alternatives: tuple[RejectedPhysicalCandidate, ...]
    recovery_variants: tuple[RecoveryVariant, ...]
    evidence: tuple[DocumentReference, ...]
    compiler_version: NonEmptyString
    git_commit: NonEmptyString
    reproducibility: ReproducibilityMetadata
    extensions: Extensions = Field(default_factory=lambda: Extensions(root={}))

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        rank_count = len(self.rank_placement.bindings)
        if rank_count != self.parallelism.expected_rank_count:
            raise ValueError("rank placement count does not match parallelism degrees")
        allocated = {allocation.rank_id for allocation in self.memory.allocations}
        if allocated != set(range(rank_count)):
            raise ValueError("memory plan must cover every rank exactly once")
        topology_ranks = set(range(rank_count))
        for operation in self.collectives.operations:
            if not set(operation.participating_ranks) <= topology_ranks:
                raise ValueError("collective references rank outside placement")
        if self.expert_placement is not None:
            for assignment in self.expert_placement.assignments:
                if not set(assignment.rank_ids) <= topology_ranks:
                    raise ValueError("expert placement references rank outside placement")
        sequences = [entry.sequence for entry in self.optimizer_history]
        if sequences != sorted(sequences):
            raise ValueError("optimizer history must be ordered by sequence")
        return self


class BenchmarkInvocation(FabricModel):
    argv: tuple[NonEmptyString, ...]
    timeout_seconds: PositiveFloat
    process_placement: NonEmptyString
    environment_digest: ArtifactDigest


class FabricRawSample(FabricModel):
    duration_us: PositiveFloat
    bytes_transferred: NonNegativeInt
    success: bool
    failure_reason: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_failure(self) -> Self:
        if self.success == (self.failure_reason is not None):
            raise ValueError("exactly failed samples require failure_reason")
        return self


class FabricMeasurementSeries(FabricModel):
    measurement_id: NonEmptyString
    primitive: Literal[
        "launch",
        "synchronize",
        "memory",
        "gemm",
        "copy",
        "p2p",
        "collective",
        "expert",
        "kv_transfer",
        "startup",
    ]
    transport: NonEmptyString
    rank_count: PositiveInt
    message_bytes: NonNegativeInt
    concurrency: PositiveInt
    warmup_count: NonNegativeInt
    samples: tuple[FabricRawSample, ...]
    summary_median_us: PositiveFloat
    summary_p95_us: PositiveFloat
    confidence_low_us: NonNegativeFloat
    confidence_high_us: PositiveFloat
    invocation: BenchmarkInvocation
    artifact_digest: ArtifactDigest

    @model_validator(mode="after")
    def validate_samples(self) -> Self:
        if not self.samples:
            raise ValueError("measurement series requires raw samples")
        if self.summary_p95_us < self.summary_median_us:
            raise ValueError("measurement p95 cannot be below median")
        if not self.confidence_low_us <= self.summary_median_us <= self.confidence_high_us:
            raise ValueError("measurement confidence interval must contain median")
        return self


class FabricProfile(FabricModel):
    schema_version: Literal["1.0.0"] = FABRIC_SCHEMA_VERSION
    api_version: Literal["sloforge.io/fabric/v1"] = FABRIC_API_VERSION
    kind: Literal["FabricProfile"] = "FabricProfile"
    profile_id: NonEmptyString
    topology_fingerprint: ArtifactDigest
    created_at: AwareDatetime
    hardware_manifest: DocumentReference
    software_manifest: DocumentReference
    measurements: tuple[FabricMeasurementSeries, ...]
    raw_artifacts: tuple[DocumentReference, ...]
    extensions: Extensions = Field(default_factory=lambda: Extensions(root={}))

    @model_validator(mode="after")
    def validate_measurement_ids(self) -> Self:
        identifiers = [measurement.measurement_id for measurement in self.measurements]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("fabric measurement IDs must be unique")
        return self


class RecoveryActionKind(StrEnum):
    STOP_ROUTING = "stop_routing"
    DRAIN_WORKER = "drain_worker"
    RESTART_WORKER = "restart_worker"
    QUARANTINE_GPU = "quarantine_gpu"
    QUARANTINE_NIC = "quarantine_nic"
    QUARANTINE_RAIL = "quarantine_rail"
    REPLACE_REPLICA = "replace_replica"
    CHANGE_RANK_PLACEMENT = "change_rank_placement"
    CHANGE_RANK_ORDERING = "change_rank_ordering"
    CHANGE_NIC_AFFINITY = "change_nic_affinity"
    CHANGE_NUMA_AFFINITY = "change_numa_affinity"
    MOVE_EXPERT_GROUP = "move_expert_group"
    REPLICATE_HOT_EXPERTS = "replicate_hot_experts"
    CHANGE_WORKER_RATIO = "change_worker_ratio"
    SWITCH_KV_TRANSPORT = "switch_kv_transport"
    SWITCH_COLLECTIVE = "switch_collective"
    REDUCE_COMMUNICATION_CONCURRENCY = "reduce_communication_concurrency"
    REDUCE_REQUEST_CONCURRENCY = "reduce_request_concurrency"
    SHED_LOW_PRIORITY = "shed_low_priority"
    SWITCH_PARALLELISM = "switch_parallelism"
    SWITCH_AGGREGATION = "switch_aggregation"
    REBUILD_DEPLOYMENT = "rebuild_deployment"
    DEGRADED_MODEL = "degraded_model"


class RecoveryScope(StrEnum):
    REQUEST_PATH = "request_path"
    WORKER_LOCAL = "worker_local"
    REPLICA_LOCAL = "replica_local"
    NEW_REPLICA = "new_replica"
    DEPLOYMENT_REBUILD = "deployment_rebuild"
    OPERATOR_REQUIRED = "operator_required"


class RecoveryAction(FabricModel):
    action_id: NonEmptyString
    kind: RecoveryActionKind
    scope: RecoveryScope
    target_ids: tuple[NonEmptyString, ...]
    order: NonNegativeInt
    idempotency_key: NonEmptyString
    timeout_seconds: PositiveFloat
    rollback_action_id: NonEmptyString | None = None
    requires_external_mutation: bool = False


class TrafficMigrationPlan(FabricModel):
    shadow_fraction: Probability
    canary_fraction: Probability
    minimum_shadow_samples: PositiveInt
    minimum_canary_samples: PositiveInt
    maximum_inflight_streams_at_drain: NonNegativeInt
    preserve_started_streams: bool


class RecoveryCriterion(FabricModel):
    metric: NonEmptyString
    comparator: Literal["lt", "le", "gt", "ge"]
    threshold: float = Field(allow_inf_nan=False)
    window_seconds: PositiveFloat


class RecoveryPlan(FabricModel):
    schema_version: Literal["1.0.0"] = FABRIC_SCHEMA_VERSION
    api_version: Literal["sloforge.io/fabric/v1"] = FABRIC_API_VERSION
    kind: Literal["RecoveryPlan"] = "RecoveryPlan"
    recovery_id: NonEmptyString
    diagnosis: DocumentReference
    physical_plan: DocumentReference
    actions: tuple[RecoveryAction, ...]
    expected_slo_improvement: PhysicalMetrics
    expected_cost_usd: NonNegativeFloat
    expected_disruption_seconds: NonNegativeFloat
    expected_build_seconds: NonNegativeFloat
    confidence: Probability
    compatibility_constraints: tuple[NonEmptyString, ...]
    traffic_migration: TrafficMigrationPlan
    promotion_criteria: tuple[RecoveryCriterion, ...]
    rollback_criteria: tuple[RecoveryCriterion, ...]
    abort_criteria: tuple[RecoveryCriterion, ...]
    evidence: tuple[DocumentReference, ...]
    external_mutation_authorized: bool = False
    extensions: Extensions = Field(default_factory=lambda: Extensions(root={}))

    @model_validator(mode="after")
    def validate_actions(self) -> Self:
        if not self.actions:
            raise ValueError("recovery plan requires actions")
        identifiers = [action.action_id for action in self.actions]
        orders = [action.order for action in self.actions]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("recovery action IDs must be unique")
        if orders != sorted(orders) or len(orders) != len(set(orders)):
            raise ValueError("recovery actions must have unique ascending order")
        known = set(identifiers)
        for action in self.actions:
            if action.rollback_action_id is not None and action.rollback_action_id not in known:
                raise ValueError("rollback action references unknown action")
            if action.requires_external_mutation and not self.external_mutation_authorized:
                raise ValueError("external mutation action requires explicit authorization")
        return self


def assert_finite_document(document: BaseModel) -> None:
    """Defensive finite-number check used before canonical serialization."""

    def visit(value: object, path: str) -> None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"non-finite number at {path}")
        if isinstance(value, list | tuple):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")
        elif isinstance(value, dict):
            for key, item in value.items():
                visit(item, f"{path}.{key}")

    visit(document.model_dump(mode="python"), document.__class__.__name__)
