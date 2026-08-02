"""Canonical trusted wire types for SLOForge Genesis.

The models in this module are deliberately data-only.  Unknown fields are
rejected, core fields have concrete types, and the only extensibility point is
the namespace-qualified :class:`Extensions` object.
"""

from __future__ import annotations

import math
import re
from enum import StrEnum
from typing import Annotated, Final, Literal, Self, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    RootModel,
    StringConstraints,
    field_validator,
    model_validator,
)

SCHEMA_VERSION: Final = "1.0.0"
API_VERSION: Final = "sloforge.io/genesis/v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EXTENSION_KEY_PATTERN = r"^[a-z][a-z0-9]*(?:[.-][a-z0-9]+)*/[A-Za-z][A-Za-z0-9_.-]*$"
_EXTENSION_KEY = re.compile(_EXTENSION_KEY_PATTERN)

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
ExtensionKey = Annotated[str, StringConstraints(pattern=_EXTENSION_KEY_PATTERN)]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
PositiveFloat = Annotated[float, Field(gt=0.0)]
U64_MAX: Final = (1 << 64) - 1
I64_MIN: Final = -(1 << 63)
I64_MAX: Final = (1 << 63) - 1
NonNegativeInt = Annotated[int, Field(ge=0, le=U64_MAX)]
PositiveInt = Annotated[int, Field(gt=0, le=U64_MAX)]
SignedInt = Annotated[int, Field(ge=I64_MIN, le=I64_MAX)]


class GenesisModel(BaseModel):
    """Strict immutable base for trusted Genesis IR values."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


def _validate_json(value: JsonValue, path: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} contains a non-finite number")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json(item, f"{path}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            _validate_json(item, f"{path}.{key}")


class Extensions(RootModel[dict[ExtensionKey, JsonValue]]):
    """The sole extension namespace; keys identify their owning organization."""

    model_config = ConfigDict(frozen=True, strict=True)

    @model_validator(mode="after")
    def validate_entries(self) -> Self:
        for key, value in self.root.items():
            if _EXTENSION_KEY.fullmatch(key) is None:
                raise ValueError(f"extension key {key!r} must be namespace-qualified")
            _validate_json(value, f"extensions.{key}")
        return self


class ArtifactDigest(GenesisModel):
    algorithm: Literal["sha256"] = "sha256"
    value: str

    @field_validator("value")
    @classmethod
    def valid_sha256(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError("SHA-256 must be 64 lowercase hexadecimal characters")
        return value


class EvidenceReference(GenesisModel):
    evidence_id: NonEmptyString
    artifact_uri: NonEmptyString
    digest: ArtifactDigest
    claim_ids: tuple[NonEmptyString, ...] = ()


class LineageReference(GenesisModel):
    lineage_id: NonEmptyString
    relation: Literal["parent", "derived_from", "reused", "constrained_by", "invalidated_by"]


class SemanticCategory(StrEnum):
    EXACT = "exact"
    APPROXIMATE = "approximate"
    POLICY = "policy"
    RESOURCE = "resource"
    IMPLEMENTATION = "implementation"
    EXPERIMENTAL = "experimental"


class VerificationLevel(StrEnum):
    BUILD = "level_0_build"
    DIFFERENTIAL = "level_1_differential"
    PROPERTY = "level_2_property"
    BOUNDED_EXHAUSTIVE = "level_3_bounded_exhaustive"
    SOLVER_BACKED = "level_4_solver_backed"
    HARDWARE_OPERATIONAL = "level_5_hardware_operational"


class ProofObligation(GenesisModel):
    obligation_id: NonEmptyString
    property: NonEmptyString
    minimum_level: VerificationLevel
    scope: NonEmptyString
    assumptions: tuple[NonEmptyString, ...] = ()
    required: bool = True


class SemanticContract(GenesisModel):
    contract_id: NonEmptyString
    category: SemanticCategory
    input_domain: tuple[NonEmptyString, ...]
    output_guarantees: tuple[NonEmptyString, ...]
    state_invariants: tuple[NonEmptyString, ...] = ()
    numerical_contract: NonEmptyString
    deterministic: bool


class ResourceRequirements(GenesisModel):
    peak_device_bytes: NonNegativeInt = 0
    peak_host_bytes: NonNegativeInt = 0
    queue_entries: NonNegativeInt = 0
    worker_processes: NonNegativeInt = 0
    communication_buffer_bytes: NonNegativeInt = 0


class HardwarePrecondition(GenesisModel):
    architecture: NonEmptyString | None = None
    minimum_device_memory_bytes: NonNegativeInt = 0
    required_features: tuple[NonEmptyString, ...] = ()
    forbidden_features: tuple[NonEmptyString, ...] = ()


class SoftwareRequirement(GenesisModel):
    package: NonEmptyString
    version_range: NonEmptyString


class SoftwarePrecondition(GenesisModel):
    requirements: tuple[SoftwareRequirement, ...] = ()


class QualityImplication(GenesisModel):
    metric: NonEmptyString
    expected_delta: float
    maximum_regression: NonNegativeFloat
    evaluation_contract_id: NonEmptyString

    @field_validator("expected_delta")
    @classmethod
    def finite_delta(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("expected_delta must be finite")
        return value


class PerformanceEstimate(GenesisModel):
    metric: NonEmptyString
    expected_delta: float
    unit: NonEmptyString
    model_id: NonEmptyString

    @field_validator("expected_delta")
    @classmethod
    def finite_delta(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("expected_delta must be finite")
        return value


class Uncertainty(GenesisModel):
    method: NonEmptyString
    confidence: Probability
    lower: float
    upper: float

    @model_validator(mode="after")
    def valid_interval(self) -> Self:
        if not (math.isfinite(self.lower) and math.isfinite(self.upper)):
            raise ValueError("uncertainty interval must be finite")
        if self.lower > self.upper:
            raise ValueError("uncertainty lower bound exceeds upper bound")
        return self


class HotSwapCategory(StrEnum):
    POLICY_ONLY = "policy_only"
    REQUEST_BOUNDARY = "request_boundary"
    WORKER_RESTART = "worker_restart"
    NEW_REPLICA = "new_replica"
    STATE_COMPATIBLE = "state_compatible_migration"
    STATE_CONVERSION = "state_conversion_migration"
    FULL_REBUILD = "full_deployment_rebuild"
    OPERATOR_REQUIRED = "operator_required"


class GenomeNodeMetadata(GenesisModel):
    """Obligations and provenance carried by every mutable genome region."""

    stable_id: NonEmptyString
    semantic_contract: SemanticContract
    resource_requirements: ResourceRequirements
    legal_rewrite_rules: tuple[NonEmptyString, ...]
    proof_obligations: tuple[ProofObligation, ...]
    hardware_preconditions: tuple[HardwarePrecondition, ...] = ()
    software_preconditions: tuple[SoftwarePrecondition, ...] = ()
    quality_implications: tuple[QualityImplication, ...] = ()
    expected_performance: tuple[PerformanceEstimate, ...] = ()
    uncertainty: Uncertainty
    hot_swap_category: HotSwapCategory
    lineage_references: tuple[LineageReference, ...] = ()
    evidence_references: tuple[EvidenceReference, ...] = ()
    frozen: bool = False
    extensions: Extensions = Field(default_factory=lambda: Extensions(root={}))

    @model_validator(mode="after")
    def mutable_nodes_are_verifiable(self) -> Self:
        if not self.frozen and not self.legal_rewrite_rules:
            raise ValueError("mutable nodes must declare legal rewrite rules")
        if not self.proof_obligations:
            raise ValueError("genome nodes must declare proof obligations")
        return self


class WorkflowStepKind(StrEnum):
    MODEL_INVOCATION = "model_invocation"
    TOOL_CALL = "tool_call"
    VERIFICATION = "verification_pass"
    BRANCH = "branch"
    LOOP = "loop"


class CancellationBehavior(StrEnum):
    IMMEDIATE = "immediate"
    SAFE_POINT = "safe_point"
    DRAIN = "drain"
    IGNORE = "ignore"


class WorkflowStep(GenesisModel):
    node: GenomeNodeMetadata
    kind: WorkflowStepKind
    target: NonEmptyString
    branch_probability: Probability
    expected_latency_ms: NonNegativeFloat
    deadline_ms: PositiveFloat | None = None
    priority: SignedInt = 0
    maximum_iterations: NonNegativeInt = 0
    model_cascade_targets: tuple[NonEmptyString, ...] = ()
    expected_future_requests: NonNegativeFloat = 0.0
    shared_prefix_group: NonEmptyString | None = None
    cancellation_behavior: CancellationBehavior = CancellationBehavior.SAFE_POINT


class WorkflowEdge(GenesisModel):
    node: GenomeNodeMetadata
    source_id: NonEmptyString
    target_id: NonEmptyString
    condition: NonEmptyString
    probability: Probability


class WorkflowGenome(GenesisModel):
    node: GenomeNodeMetadata
    steps: tuple[WorkflowStep, ...]
    edges: tuple[WorkflowEdge, ...]
    entry_step_id: NonEmptyString
    workflow_deadline_ms: PositiveFloat | None = None


class AdmissionControl(StrEnum):
    BOUNDED_FIFO = "bounded_fifo"
    DEADLINE_AWARE = "deadline_aware"
    TOKEN_BUDGET = "token_budget"
    REJECT = "reject"


class QueueDiscipline(StrEnum):
    FIFO = "fifo"
    EARLIEST_DEADLINE = "earliest_deadline"
    SHORTEST_REMAINING = "shortest_remaining"
    WEIGHTED_FAIR = "weighted_fair"


class RoutingPolicy(StrEnum):
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    CACHE_AFFINITY = "cache_affinity"
    WORKFLOW_AFFINITY = "workflow_affinity"


class StreamingSemantics(StrEnum):
    TOKEN_COMMIT = "token_commit"
    CHUNK_COMMIT = "chunk_commit"
    ATOMIC_RESPONSE = "atomic_response"


class FallbackBehavior(StrEnum):
    REJECT = "reject"
    LOWER_QUALITY_TIER = "lower_quality_tier"
    REFERENCE_RUNTIME = "reference_runtime"
    RETRY_COMPATIBLE = "retry_compatible"


class RequestGenome(GenesisModel):
    node: GenomeNodeMetadata
    admission_control: AdmissionControl
    maximum_queue_depth: PositiveInt
    default_priority: SignedInt
    default_deadline_ms: PositiveFloat | None
    batching_eligible: bool
    routing: RoutingPolicy
    queue_discipline: QueueDiscipline
    cancellation_behavior: CancellationBehavior
    maximum_retries: NonNegativeInt
    streaming_semantics: StreamingSemantics
    request_classes: tuple[NonEmptyString, ...]
    tenant_isolation: bool
    workflow_identity_required: bool
    quality_tiers: tuple[NonEmptyString, ...]
    fallback_behavior: FallbackBehavior


class ServingTopology(StrEnum):
    AGGREGATED = "aggregated"
    DISAGGREGATED = "disaggregated"


class PrefillPolicy(StrEnum):
    WHOLE_PROMPT = "whole_prompt"
    CHUNKED = "chunked"
    INCREMENTAL = "incremental"


class DecodeScheduling(StrEnum):
    ROUND_ROBIN = "round_robin"
    DEADLINE_AWARE = "deadline_aware"
    SLO_SLACK = "slo_slack"
    WORKFLOW_AWARE = "workflow_aware"


class ServingGenome(GenesisModel):
    node: GenomeNodeMetadata
    topology: ServingTopology
    prefill_policy: PrefillPolicy
    incremental_prefill: bool
    prefill_chunk_tokens: PositiveInt
    decode_scheduling: DecodeScheduling
    continuous_batching: bool
    maximum_batch_tokens: PositiveInt
    speculative_decoding: bool
    draft_model_id: NonEmptyString | None
    verification_policy: NonEmptyString
    model_cascade: tuple[NonEmptyString, ...]
    decode_chunk_tokens: PositiveInt
    request_migration: bool
    worker_roles: tuple[NonEmptyString, ...]

    @model_validator(mode="after")
    def draft_is_declared(self) -> Self:
        if self.speculative_decoding and self.draft_model_id is None:
            raise ValueError("speculative decoding requires draft_model_id")
        return self


class StateKind(StrEnum):
    AUTOREGRESSIVE = "autoregressive"
    KV = "kv"
    RECURRENT = "recurrent"
    CONVOLUTIONAL = "convolutional"
    SPECULATIVE = "speculative"
    CUSTOM = "custom"
    TOOL = "tool"
    WORKFLOW = "workflow"


class StateOwnership(StrEnum):
    REQUEST = "request"
    SESSION = "session"
    WORKER = "worker"
    REPLICA = "replica"
    SHARED_REPLICATED = "shared_replicated"


class StateLayout(StrEnum):
    CONTIGUOUS = "contiguous"
    PAGED = "paged"
    INTERLEAVED = "interleaved"
    SHARDED = "sharded"


class Precision(StrEnum):
    BOOL = "bool"
    FLOAT64 = "float64"
    FLOAT32 = "float32"
    BFLOAT16 = "bfloat16"
    FLOAT16 = "float16"
    FP8 = "fp8"
    INT64 = "int64"
    INT32 = "int32"
    INT16 = "int16"
    INT8 = "int8"
    UINT8 = "uint8"
    INT4 = "int4"


class RetentionPolicy(StrEnum):
    REQUEST_LIFETIME = "request_lifetime"
    SESSION = "session"
    LRU = "lru"
    DEADLINE_AWARE = "deadline_aware"


class ConsistencyModel(StrEnum):
    EXCLUSIVE = "exclusive"
    VERSIONED = "versioned"
    READ_ONLY_REPLICATED = "read_only_replicated"


class StateSpec(GenesisModel):
    node: GenomeNodeMetadata
    state_id: NonEmptyString
    kind: StateKind
    cache_key_fields: tuple[NonEmptyString, ...]
    ownership: StateOwnership
    layout: StateLayout
    precision: Precision
    retention: RetentionPolicy
    replication_factor: PositiveInt
    migratable: bool
    offload_tier: Literal["none", "host", "peer", "remote"]
    checkpoint_interval_tokens: NonNegativeInt
    eviction_policy: NonEmptyString
    recomputable: bool
    consistency: ConsistencyModel
    recovery_behavior: NonEmptyString
    maximum_bytes_per_request: NonNegativeInt


class StateGenome(GenesisModel):
    node: GenomeNodeMetadata
    states: tuple[StateSpec, ...]
    migration_chunk_bytes: PositiveInt
    prefetch_enabled: bool
    conversion_artifact: EvidenceReference | None = None


class ParallelismSpec(GenesisModel):
    node: GenomeNodeMetadata
    tensor: PositiveInt = 1
    pipeline: PositiveInt = 1
    data: PositiveInt = 1
    expert: PositiveInt = 1
    context: PositiveInt = 1


class Placement(GenesisModel):
    node: GenomeNodeMetadata
    logical_rank: NonNegativeInt
    host_id: NonEmptyString
    device_id: NonEmptyString
    numa_domain: NonEmptyString | None = None
    network_rail: NonEmptyString | None = None


class ExpertPlacement(GenesisModel):
    node: GenomeNodeMetadata
    expert_id: NonNegativeInt
    logical_ranks: tuple[NonNegativeInt, ...]


class CollectiveKind(StrEnum):
    ALL_REDUCE = "all_reduce"
    ALL_GATHER = "all_gather"
    REDUCE_SCATTER = "reduce_scatter"
    ALL_TO_ALL = "all_to_all"
    SEND_RECV = "send_recv"


class CollectiveStep(GenesisModel):
    node: GenomeNodeMetadata
    step_id: NonEmptyString
    kind: CollectiveKind
    dependencies: tuple[NonEmptyString, ...]
    algorithm: NonEmptyString
    transport: NonEmptyString
    ranks: tuple[NonNegativeInt, ...]
    chunk_bytes: PositiveInt
    overlap_group: NonEmptyString | None = None


class DistributedGenome(GenesisModel):
    node: GenomeNodeMetadata
    parallelism: ParallelismSpec
    rank_placement: tuple[Placement, ...]
    expert_placement: tuple[ExpertPlacement, ...]
    collective_dag: tuple[CollectiveStep, ...]
    prefill_decode_transfer: Literal["none", "host", "peer", "rdma"]
    failure_domains: tuple[NonEmptyString, ...]
    recovery_variant_ids: tuple[NonEmptyString, ...]


class SymbolicDimension(GenesisModel):
    node: GenomeNodeMetadata
    name: NonEmptyString
    minimum: PositiveInt
    maximum: PositiveInt
    divisible_by: PositiveInt = 1

    @model_validator(mode="after")
    def valid_range(self) -> Self:
        if self.minimum > self.maximum:
            raise ValueError("symbolic dimension minimum exceeds maximum")
        return self


class TensorValue(GenesisModel):
    node: GenomeNodeMetadata
    value_id: NonEmptyString
    shape: tuple[NonEmptyString, ...]
    dtype: Precision
    strides: tuple[NonEmptyString, ...]
    layout: NonEmptyString
    alias_group: NonEmptyString | None = None
    state_dependency: NonEmptyString | None = None


class TensorOperator(GenesisModel):
    node: GenomeNodeMetadata
    operator_id: NonEmptyString
    operator: NonEmptyString
    inputs: tuple[NonEmptyString, ...]
    outputs: tuple[NonEmptyString, ...]
    fused_operators: tuple[NonEmptyString, ...] = ()
    decomposition: tuple[NonEmptyString, ...] = ()
    quantization: NonEmptyString = "none"
    sparse: bool = False
    numerical_contract: NonEmptyString


class RewriteRecord(GenesisModel):
    transformation_id: NonEmptyString
    source_hash: ArtifactDigest
    target_hash: ArtifactDigest


class TensorGenome(GenesisModel):
    node: GenomeNodeMetadata
    symbolic_dimensions: tuple[SymbolicDimension, ...]
    values: tuple[TensorValue, ...]
    operators: tuple[TensorOperator, ...]
    graph_inputs: tuple[NonEmptyString, ...]
    graph_outputs: tuple[NonEmptyString, ...]
    rewrite_history: tuple[RewriteRecord, ...] = ()


class KernelBackend(StrEnum):
    PYTORCH = "pytorch"
    TRITON = "triton"
    CUDA = "cuda"
    CUTE = "cute"
    CPP = "cpp"


class LaunchConfiguration(GenesisModel):
    block_x: PositiveInt
    block_y: PositiveInt = 1
    block_z: PositiveInt = 1
    warps: PositiveInt = 1
    pipeline_stages: PositiveInt = 1


class ShapeDomain(GenesisModel):
    constraints: tuple[NonEmptyString, ...]


class KernelSpec(GenesisModel):
    node: GenomeNodeMetadata
    kernel_id: NonEmptyString
    source_artifact: EvidenceReference
    backend: KernelBackend
    target_architecture: NonEmptyString
    launch: LaunchConfiguration
    tile_shape: tuple[PositiveInt, ...]
    warp_strategy: NonEmptyString
    shared_memory_bytes: NonNegativeInt
    register_estimate: NonNegativeInt
    vector_width: PositiveInt
    layout_assumptions: tuple[NonEmptyString, ...]
    supported_shapes: ShapeDomain
    supported_dtypes: tuple[Precision, ...]
    deterministic: bool
    numerical_tolerance: NonNegativeFloat
    benchmark_evidence: tuple[EvidenceReference, ...]
    fallback_kernel_id: NonEmptyString


class KernelGenome(GenesisModel):
    node: GenomeNodeMetadata
    kernels: tuple[KernelSpec, ...]


class TransitionPoint(StrEnum):
    IMMEDIATE = "immediate"
    TOKEN_BOUNDARY = "token_boundary"
    REQUEST_BOUNDARY = "request_boundary"
    DRAINED = "drained"


class RecoveryTransition(GenesisModel):
    node: GenomeNodeMetadata
    transition_id: NonEmptyString
    safe_point: TransitionPoint
    source_state_contract: NonEmptyString
    target_state_contract: NonEmptyString
    state_conversion_artifact: EvidenceReference | None
    state_transfer: Literal["none", "copy", "move", "recompute"]
    active_stream_behavior: Literal["preserve", "drain", "restart", "reject"]
    rollback_transition_id: NonEmptyString
    failure_invariants: tuple[NonEmptyString, ...]
    operator_action_required: bool


class RecoveryGenome(GenesisModel):
    node: GenomeNodeMetadata
    transitions: tuple[RecoveryTransition, ...]
    shadow_mode: bool
    canary_mode: bool
    degraded_mode_ids: tuple[NonEmptyString, ...]


class InferenceGenome(GenesisModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    api_version: Literal["sloforge.io/genesis/v1"] = API_VERSION
    kind: Literal["InferenceGenome"] = "InferenceGenome"
    genome_id: NonEmptyString
    seed: NonNegativeInt
    source_model: ArtifactDigest
    workflow: WorkflowGenome
    request: RequestGenome
    serving: ServingGenome
    state: StateGenome
    distributed: DistributedGenome
    tensor: TensorGenome
    kernel: KernelGenome
    recovery: RecoveryGenome
    extensions: Extensions = Field(default_factory=lambda: Extensions(root={}))

    @model_validator(mode="after")
    def graph_references_are_closed(self) -> Self:
        def require_dag(
            identifiers: set[str], dependencies: dict[str, tuple[str, ...]], label: str
        ) -> None:
            visiting: set[str] = set()
            visited: set[str] = set()

            def visit(identifier: str) -> None:
                if identifier in visiting:
                    raise ValueError(f"{label} must be acyclic")
                if identifier in visited:
                    return
                visiting.add(identifier)
                for dependency in dependencies.get(identifier, ()):
                    if dependency not in identifiers:
                        raise ValueError(f"{label} dependency must reference a declared node")
                    visit(dependency)
                visiting.remove(identifier)
                visited.add(identifier)

            for identifier in identifiers:
                visit(identifier)

        step_ids = [step.node.stable_id for step in self.workflow.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("workflow step stable identifiers must be unique")
        if self.workflow.entry_step_id not in step_ids:
            raise ValueError("workflow entry_step_id must reference a declared step")
        for edge in self.workflow.edges:
            if edge.source_id not in step_ids or edge.target_id not in step_ids:
                raise ValueError("workflow edge must reference declared steps")
        workflow_dependencies: dict[str, list[str]] = {step_id: [] for step_id in step_ids}
        edge_keys: set[tuple[str, str, str]] = set()
        for edge in self.workflow.edges:
            key = (edge.source_id, edge.target_id, edge.condition)
            if key in edge_keys:
                raise ValueError("workflow edges must be unique")
            edge_keys.add(key)
            workflow_dependencies[edge.target_id].append(edge.source_id)
        require_dag(
            set(step_ids),
            {key: tuple(value) for key, value in workflow_dependencies.items()},
            "workflow DAG",
        )
        state_ids = [state.state_id for state in self.state.states]
        if len(state_ids) != len(set(state_ids)):
            raise ValueError("state identifiers must be unique")
        rank_ids = [placement.logical_rank for placement in self.distributed.rank_placement]
        if len(rank_ids) != len(set(rank_ids)):
            raise ValueError("rank placements must have unique logical ranks")
        declared_ranks = set(rank_ids)
        expert_ids = [placement.expert_id for placement in self.distributed.expert_placement]
        if len(expert_ids) != len(set(expert_ids)):
            raise ValueError("expert placements must have unique expert identifiers")
        for placement in self.distributed.expert_placement:
            if len(placement.logical_ranks) != len(set(placement.logical_ranks)):
                raise ValueError("expert placement logical ranks must be unique")
            if any(rank not in declared_ranks for rank in placement.logical_ranks):
                raise ValueError("expert placement must reference a declared logical rank")
        collective_ids = [step.step_id for step in self.distributed.collective_dag]
        if len(collective_ids) != len(set(collective_ids)):
            raise ValueError("collective step identifiers must be unique")
        for step in self.distributed.collective_dag:
            if len(step.dependencies) != len(set(step.dependencies)):
                raise ValueError("collective dependencies must be unique")
            if len(step.ranks) != len(set(step.ranks)):
                raise ValueError("collective ranks must be unique")
            if any(rank not in declared_ranks for rank in step.ranks):
                raise ValueError("collective must reference declared logical ranks")
        require_dag(
            set(collective_ids),
            {step.step_id: step.dependencies for step in self.distributed.collective_dag},
            "collective DAG",
        )
        value_ids = [value.value_id for value in self.tensor.values]
        if len(value_ids) != len(set(value_ids)):
            raise ValueError("tensor value identifiers must be unique")
        if any(value_id not in value_ids for value_id in self.tensor.graph_inputs):
            raise ValueError("tensor graph input must reference a declared value")
        if any(value_id not in value_ids for value_id in self.tensor.graph_outputs):
            raise ValueError("tensor graph output must reference a declared value")
        for operator in self.tensor.operators:
            if any(value_id not in value_ids for value_id in (*operator.inputs, *operator.outputs)):
                raise ValueError("tensor operator must reference declared values")
        operator_ids = [operator.operator_id for operator in self.tensor.operators]
        if len(operator_ids) != len(set(operator_ids)):
            raise ValueError("tensor operator identifiers must be unique")
        producers: dict[str, str] = {}
        for operator in self.tensor.operators:
            if len(operator.outputs) != len(set(operator.outputs)):
                raise ValueError("tensor operator outputs must be unique")
            for output in operator.outputs:
                if output in producers:
                    raise ValueError("tensor values must have a single producer")
                producers[output] = operator.operator_id
        if any(
            value.state_dependency not in set(state_ids)
            for value in self.tensor.values
            if value.state_dependency is not None
        ):
            raise ValueError("tensor state_dependency must reference declared state")
        if any(
            output not in producers and output not in self.tensor.graph_inputs
            for output in self.tensor.graph_outputs
        ):
            raise ValueError("tensor graph output must be produced or passed through")
        kernel_ids = [kernel.kernel_id for kernel in self.kernel.kernels]
        if len(kernel_ids) != len(set(kernel_ids)):
            raise ValueError("kernel identifiers must be unique")
        if any(kernel.fallback_kernel_id not in set(kernel_ids) for kernel in self.kernel.kernels):
            raise ValueError("kernel fallback must reference a declared kernel")
        transition_ids = [transition.transition_id for transition in self.recovery.transitions]
        if len(transition_ids) != len(set(transition_ids)):
            raise ValueError("recovery transition identifiers must be unique")
        declared_transitions = set(transition_ids)
        if any(
            transition.rollback_transition_id not in declared_transitions
            for transition in self.recovery.transitions
        ):
            raise ValueError("rollback transition must reference a declared transition")
        if any(
            identifier not in declared_transitions
            for identifier in self.distributed.recovery_variant_ids
        ):
            raise ValueError("distributed recovery variant must reference a declared transition")
        return self


class TransformationFamily(StrEnum):
    ALGEBRAIC_REWRITE = "algebraic_rewrite"
    TENSOR_DECOMPOSITION = "tensor_decomposition"
    OPERATOR_FUSION = "operator_fusion"
    LAYOUT = "layout_transformation"
    PRECISION = "precision_transformation"
    QUANTIZATION = "quantization_transformation"
    SCHEDULER = "scheduler_transformation"
    BATCHING = "batching_transformation"
    CACHE_POLICY = "cache_policy_transformation"
    STATE_LAYOUT = "state_layout_transformation"
    DISTRIBUTED_PLAN = "distributed_plan_transformation"
    COMMUNICATION = "communication_transformation"
    KERNEL = "kernel_transformation"
    WORKFLOW = "workflow_transformation"
    RECOVERY = "recovery_transformation"
    RUNTIME_CODE_PATCH = "runtime_code_patch"


class TransformationDesignation(StrEnum):
    SEMANTICS_PRESERVING = "semantics_preserving"
    APPROXIMATE_WITHIN_BUDGET = "approximate_within_quality_budget"
    POLICY = "policy"
    RESOURCE_ONLY = "resource_only"
    RUNTIME_IMPLEMENTATION = "runtime_implementation"
    EXPERIMENTAL_OPERATOR_REVIEW = "experimental_operator_review"


class GenomePattern(GenesisModel):
    region: Literal[
        "workflow", "request", "serving", "state", "distributed", "tensor", "kernel", "recovery"
    ]
    node_ids: tuple[NonEmptyString, ...]
    structural_constraints: tuple[NonEmptyString, ...]


class EstimatedChange(GenesisModel):
    metric: NonEmptyString
    lower: float
    expected: float
    upper: float
    unit: NonEmptyString

    @model_validator(mode="after")
    def ordered_finite(self) -> Self:
        values = (self.lower, self.expected, self.upper)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("estimated change values must be finite")
        if not self.lower <= self.expected <= self.upper:
            raise ValueError("estimated change interval is not ordered")
        return self


class LearnedConstraint(GenesisModel):
    constraint_id: NonEmptyString
    expression: NonEmptyString
    scope: Literal["candidate", "family", "hardware", "dependency", "universal_precondition"]
    counterexample_ids: tuple[NonEmptyString, ...]


class Transformation(GenesisModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    api_version: Literal["sloforge.io/genesis/v1"] = API_VERSION
    kind: Literal["Transformation"] = "Transformation"
    transformation_id: NonEmptyString
    family: TransformationFamily
    source_pattern: GenomePattern
    target_pattern: GenomePattern
    semantic_category: SemanticCategory
    designation: TransformationDesignation
    preconditions: tuple[NonEmptyString, ...]
    postconditions: tuple[NonEmptyString, ...]
    expected_quality_cost: tuple[EstimatedChange, ...]
    expected_resource_change: tuple[EstimatedChange, ...]
    expected_performance_change: tuple[EstimatedChange, ...]
    affected_regions: tuple[NonEmptyString, ...]
    verification_obligations: tuple[ProofObligation, ...]
    required_verifier_stages: tuple[NonEmptyString, ...]
    required_benchmark_stages: tuple[NonEmptyString, ...]
    rollback_strategy: NonEmptyString
    proposal_source: NonEmptyString
    parent_transformations: tuple[NonEmptyString, ...] = ()
    learned_constraints: tuple[LearnedConstraint, ...] = ()
    counterexample_references: tuple[NonEmptyString, ...] = ()
    lineage_references: tuple[LineageReference, ...] = ()
    extensions: Extensions = Field(default_factory=lambda: Extensions(root={}))

    @model_validator(mode="after")
    def approximation_has_quality_obligation(self) -> Self:
        if (
            self.designation is TransformationDesignation.APPROXIMATE_WITHIN_BUDGET
            and not self.expected_quality_cost
        ):
            raise ValueError("approximate transformations require an expected quality cost")
        if not self.verification_obligations:
            raise ValueError("every transformation must create verification obligations")
        return self


class CandidateSuccessState(StrEnum):
    PROPOSED = "PROPOSED"
    STATICALLY_VALID = "STATICALLY_VALID"
    COMPILED = "COMPILED"
    REFERENCE_TESTED = "REFERENCE_TESTED"
    PROPERTY_TESTED = "PROPERTY_TESTED"
    MODEL_CHECKED = "MODEL_CHECKED"
    SIMULATED = "SIMULATED"
    HARDWARE_BENCHMARKED = "HARDWARE_BENCHMARKED"
    SHADOW_VALIDATED = "SHADOW_VALIDATED"
    CANARY_VALIDATED = "CANARY_VALIDATED"
    CAPSULE_ACCEPTED = "CAPSULE_ACCEPTED"
    PROMOTED = "PROMOTED"


class CandidateFailureState(StrEnum):
    STATIC_REJECTED = "STATIC_REJECTED"
    COMPILE_REJECTED = "COMPILE_REJECTED"
    SEMANTIC_REJECTED = "SEMANTIC_REJECTED"
    QUALITY_REJECTED = "QUALITY_REJECTED"
    RESOURCE_REJECTED = "RESOURCE_REJECTED"
    MODEL_CHECK_REJECTED = "MODEL_CHECK_REJECTED"
    PERFORMANCE_REJECTED = "PERFORMANCE_REJECTED"
    SHADOW_REJECTED = "SHADOW_REJECTED"
    CANARY_REJECTED = "CANARY_REJECTED"
    SANDBOX_VIOLATION = "SANDBOX_VIOLATION"
    SUPERSEDED = "SUPERSEDED"


CandidateState: TypeAlias = CandidateSuccessState | CandidateFailureState


class SearchBudget(GenesisModel):
    wall_time_seconds: NonNegativeFloat
    cpu_time_seconds: NonNegativeFloat
    gpu_time_seconds: NonNegativeFloat
    cloud_cost_usd: NonNegativeFloat
    external_synthesis_cost_usd: NonNegativeFloat
    candidate_count: NonNegativeInt
    compilation_count: NonNegativeInt
    benchmark_count: NonNegativeInt
    verifier_time_seconds: NonNegativeFloat


class BudgetUsage(GenesisModel):
    wall_time_seconds: NonNegativeFloat = 0.0
    cpu_time_seconds: NonNegativeFloat = 0.0
    gpu_time_seconds: NonNegativeFloat = 0.0
    cloud_cost_usd: NonNegativeFloat = 0.0
    external_synthesis_cost_usd: NonNegativeFloat = 0.0
    candidate_count: NonNegativeInt = 0
    compilation_count: NonNegativeInt = 0
    benchmark_count: NonNegativeInt = 0
    verifier_time_seconds: NonNegativeFloat = 0.0


class LifecycleEvent(GenesisModel):
    sequence: NonNegativeInt
    from_state: CandidateState | None
    to_state: CandidateState
    reason: NonEmptyString
    evidence: tuple[EvidenceReference, ...] = ()


class Candidate(GenesisModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    api_version: Literal["sloforge.io/genesis/v1"] = API_VERSION
    kind: Literal["Candidate"] = "Candidate"
    candidate_id: NonEmptyString
    seed: NonNegativeInt
    genome_hash: ArtifactDigest
    parent_candidate_ids: tuple[NonEmptyString, ...]
    transformation_ids: tuple[NonEmptyString, ...]
    state: CandidateState
    lifecycle: tuple[LifecycleEvent, ...]
    budget: SearchBudget
    usage: BudgetUsage
    extensions: Extensions = Field(default_factory=lambda: Extensions(root={}))

    @model_validator(mode="after")
    def lifecycle_is_audit_log(self) -> Self:
        if not self.lifecycle:
            raise ValueError("candidate lifecycle must not be empty")
        for expected, event in enumerate(self.lifecycle):
            if event.sequence != expected:
                raise ValueError("candidate lifecycle sequence must be contiguous from zero")
            if expected == 0:
                if (
                    event.from_state is not None
                    or event.to_state is not CandidateSuccessState.PROPOSED
                ):
                    raise ValueError("candidate lifecycle must begin at PROPOSED")
            elif event.from_state != self.lifecycle[expected - 1].to_state:
                raise ValueError("candidate lifecycle transition is discontinuous")
            if expected > 0:
                previous = self.lifecycle[expected - 1].to_state
                if isinstance(previous, CandidateFailureState):
                    raise ValueError("candidate failure states are terminal")
                if isinstance(event.to_state, CandidateSuccessState):
                    success_order = tuple(CandidateSuccessState)
                    if success_order.index(event.to_state) != success_order.index(previous) + 1:
                        raise ValueError("candidate success stages cannot be skipped or reversed")
        if self.lifecycle[-1].to_state != self.state:
            raise ValueError("candidate state must match final lifecycle event")
        budget_pairs = (
            (self.usage.wall_time_seconds, self.budget.wall_time_seconds),
            (self.usage.cpu_time_seconds, self.budget.cpu_time_seconds),
            (self.usage.gpu_time_seconds, self.budget.gpu_time_seconds),
            (self.usage.cloud_cost_usd, self.budget.cloud_cost_usd),
            (
                self.usage.external_synthesis_cost_usd,
                self.budget.external_synthesis_cost_usd,
            ),
            (self.usage.candidate_count, self.budget.candidate_count),
            (self.usage.compilation_count, self.budget.compilation_count),
            (self.usage.benchmark_count, self.budget.benchmark_count),
            (self.usage.verifier_time_seconds, self.budget.verifier_time_seconds),
        )
        if any(used > allowed for used, allowed in budget_pairs):
            raise ValueError("candidate usage exceeds declared search budget")
        return self


class TensorInputCase(GenesisModel):
    shape: tuple[PositiveInt, ...]
    strides: tuple[SignedInt, ...]
    dtype: Precision
    values_hex: NonEmptyString
    non_contiguous: bool


class RequestEventCase(GenesisModel):
    at_step: NonNegativeInt
    request_id: NonEmptyString
    action: Literal[
        "admit", "schedule", "prefill", "decode", "emit", "cancel", "disconnect", "fail", "retry"
    ]
    worker_id: NonEmptyString | None = None


class TopologyCase(GenesisModel):
    hosts: PositiveInt
    devices_per_host: PositiveInt
    failed_links: tuple[NonEmptyString, ...]
    degraded_links: tuple[NonEmptyString, ...]


class DependencyCase(GenesisModel):
    package: NonEmptyString
    version: NonEmptyString
    hardware_architecture: NonEmptyString | None = None


class ResourceCase(GenesisModel):
    device_bytes: NonNegativeInt
    host_bytes: NonNegativeInt
    queue_depth: NonNegativeInt
    process_count: NonNegativeInt


class TensorCounterexamplePayload(GenesisModel):
    kind: Literal["tensor"] = "tensor"
    input: TensorInputCase


class RequestTraceCounterexamplePayload(GenesisModel):
    kind: Literal["request_trace"] = "request_trace"
    events: tuple[RequestEventCase, ...]


class TopologyCounterexamplePayload(GenesisModel):
    kind: Literal["topology"] = "topology"
    topology: TopologyCase


class DependencyCounterexamplePayload(GenesisModel):
    kind: Literal["dependency"] = "dependency"
    dependency: DependencyCase


class ResourceCounterexamplePayload(GenesisModel):
    kind: Literal["resource"] = "resource"
    resource: ResourceCase


CounterexamplePayload: TypeAlias = Annotated[
    TensorCounterexamplePayload
    | RequestTraceCounterexamplePayload
    | TopologyCounterexamplePayload
    | DependencyCounterexamplePayload
    | ResourceCounterexamplePayload,
    Field(discriminator="kind"),
]


class ReproductionCommand(GenesisModel):
    executable: NonEmptyString
    arguments: tuple[str, ...]
    timeout_seconds: PositiveInt
    seed: NonNegativeInt


class EnvironmentFact(GenesisModel):
    name: NonEmptyString
    value: NonEmptyString


class BehaviorObservation(GenesisModel):
    description: NonEmptyString
    artifact: EvidenceReference | None = None


class CounterexampleScope(StrEnum):
    CANDIDATE = "candidate_specific"
    TRANSFORMATION_FAMILY = "transformation_family"
    HARDWARE = "hardware_specific"
    DEPENDENCY = "dependency_version"
    UNIVERSAL_PRECONDITION = "universal_precondition_violation"


class Counterexample(GenesisModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    api_version: Literal["sloforge.io/genesis/v1"] = API_VERSION
    kind: Literal["Counterexample"] = "Counterexample"
    counterexample_id: NonEmptyString
    candidate_id: NonEmptyString
    transformation_id: NonEmptyString | None
    violated_contract: NonEmptyString
    scope: CounterexampleScope
    payload: CounterexamplePayload
    reproduction: ReproductionCommand
    environment: tuple[EnvironmentFact, ...]
    expected: BehaviorObservation
    observed: BehaviorObservation
    minimized: bool
    parent_counterexample_id: NonEmptyString | None = None
    lineage_references: tuple[LineageReference, ...] = ()
    extensions: Extensions = Field(default_factory=lambda: Extensions(root={}))


GenesisDocument: TypeAlias = InferenceGenome | Transformation | Candidate | Counterexample


__all__ = [
    "API_VERSION",
    "SCHEMA_VERSION",
    "AdmissionControl",
    "ArtifactDigest",
    "BehaviorObservation",
    "BudgetUsage",
    "CancellationBehavior",
    "Candidate",
    "CandidateFailureState",
    "CandidateState",
    "CandidateSuccessState",
    "CollectiveKind",
    "CollectiveStep",
    "ConsistencyModel",
    "Counterexample",
    "CounterexamplePayload",
    "CounterexampleScope",
    "DecodeScheduling",
    "DependencyCase",
    "DependencyCounterexamplePayload",
    "DistributedGenome",
    "EnvironmentFact",
    "EstimatedChange",
    "EvidenceReference",
    "ExpertPlacement",
    "Extensions",
    "FallbackBehavior",
    "GenesisDocument",
    "GenesisModel",
    "GenomeNodeMetadata",
    "GenomePattern",
    "HardwarePrecondition",
    "HotSwapCategory",
    "InferenceGenome",
    "KernelBackend",
    "KernelGenome",
    "KernelSpec",
    "LaunchConfiguration",
    "LearnedConstraint",
    "LifecycleEvent",
    "LineageReference",
    "ParallelismSpec",
    "PerformanceEstimate",
    "Placement",
    "Precision",
    "PrefillPolicy",
    "ProofObligation",
    "QualityImplication",
    "QueueDiscipline",
    "RecoveryGenome",
    "RecoveryTransition",
    "ReproductionCommand",
    "RequestEventCase",
    "RequestGenome",
    "RequestTraceCounterexamplePayload",
    "ResourceCase",
    "ResourceCounterexamplePayload",
    "ResourceRequirements",
    "RetentionPolicy",
    "RewriteRecord",
    "RoutingPolicy",
    "SearchBudget",
    "SemanticCategory",
    "SemanticContract",
    "ServingGenome",
    "ServingTopology",
    "ShapeDomain",
    "SoftwarePrecondition",
    "SoftwareRequirement",
    "StateGenome",
    "StateKind",
    "StateLayout",
    "StateOwnership",
    "StateSpec",
    "StreamingSemantics",
    "SymbolicDimension",
    "TensorCounterexamplePayload",
    "TensorGenome",
    "TensorInputCase",
    "TensorOperator",
    "TensorValue",
    "TopologyCase",
    "TopologyCounterexamplePayload",
    "Transformation",
    "TransformationDesignation",
    "TransformationFamily",
    "TransitionPoint",
    "Uncertainty",
    "VerificationLevel",
    "WorkflowEdge",
    "WorkflowGenome",
    "WorkflowStep",
    "WorkflowStepKind",
]
