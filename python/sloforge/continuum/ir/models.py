"""Strict, versioned wire types for the Continuum execution-state ABI.

Logical state is deliberately separate from runtime physical layout.  Core
fields are fully typed and reject unknown input.  Vendor/runtime additions are
only legal in namespace-qualified ``Extensions`` maps.
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
API_VERSION: Final = "sloforge.io/continuum/v1"
MAX_COMPONENTS: Final = 16_384
MAX_SEGMENTS: Final = 1_000_000
U64_MAX: Final = (1 << 64) - 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EXTENSION_KEY_PATTERN = r"^[a-z][a-z0-9]*(?:[.-][a-z0-9]+)*/[A-Za-z][A-Za-z0-9_.-]*$"
_EXTENSION_KEY = re.compile(_EXTENSION_KEY_PATTERN)

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256String = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
ExtensionKey = Annotated[str, StringConstraints(pattern=_EXTENSION_KEY_PATTERN)]
NonNegativeInt = Annotated[int, Field(ge=0, le=U64_MAX)]
PositiveInt = Annotated[int, Field(gt=0, le=U64_MAX)]
NonNegativeFloat = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
PositiveFloat = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]
Probability = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]


class ContinuumModel(BaseModel):
    """Immutable strict base for trusted Continuum wire values."""

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
    """The only extension point; each key names its owning compatibility domain."""

    model_config = ConfigDict(frozen=True, strict=True)

    @model_validator(mode="after")
    def validate_entries(self) -> Self:
        for key, value in self.root.items():
            if _EXTENSION_KEY.fullmatch(key) is None:
                raise ValueError(f"extension key {key!r} must be namespace-qualified")
            _validate_json(value, f"extensions.{key}")
        return self


class Digest(ContinuumModel):
    algorithm: Literal["sha256"] = "sha256"
    value: Sha256String


class Provenance(ContinuumModel):
    producer: NonEmptyString
    producer_version: NonEmptyString
    source_uri: NonEmptyString | None = None
    source_digest: Digest | None = None
    captured_at: NonEmptyString
    raw_evidence_uri: NonEmptyString | None = None
    extensions: Extensions = Field(default_factory=lambda: Extensions(root={}))


class ExactnessClass(StrEnum):
    EXACT_BITWISE = "exact_bitwise"
    EXACT_SEMANTIC = "exact_semantic"
    NUMERICALLY_EQUIVALENT = "numerically_equivalent"
    QUALITY_BOUNDED = "quality_bounded"
    RECOMPUTATION_ASSISTED = "recomputation_assisted"
    INCOMPATIBLE = "incompatible"


class StateKind(StrEnum):
    TOKEN_HISTORY = "token_history"
    ATTENTION_KV = "attention_kv"
    RECURRENT = "recurrent"
    STATE_SPACE = "state_space"
    CONVOLUTIONAL = "convolutional"
    SPECULATIVE = "speculative"
    SAMPLER = "sampler"
    GUIDED_DECODING = "guided_decoding"
    WORKFLOW = "workflow"
    CLIENT_DELIVERY = "client_delivery"
    UNKNOWN = "unknown"


class StateLifetime(StrEnum):
    REQUEST = "request"
    SESSION = "session"
    WORKFLOW = "workflow"
    CHECKPOINT = "checkpoint"


class OwnershipScope(StrEnum):
    SESSION_OWNER = "session_owner"
    IMMUTABLE_SHARED = "immutable_shared"
    COPY_ON_WRITE = "copy_on_write"
    EXTERNAL_COORDINATOR = "external_coordinator"


class ConversionPermission(StrEnum):
    EXACT_RELAYOUT = "exact_relayout"
    DTYPE_CONVERSION = "dtype_conversion"
    QUANTIZATION = "quantization"
    RECOMPUTE = "recompute"
    OPAQUE_COPY = "opaque_copy"


class RecomputationPermission(StrEnum):
    FORBIDDEN = "forbidden"
    FROM_TOKEN_HISTORY = "from_token_history"
    FROM_CHECKPOINT = "from_checkpoint"
    MODEL_SPECIFIC = "model_specific"


class DTypeSemantics(StrEnum):
    BOOL = "bool"
    UINT8 = "uint8"
    INT8 = "int8"
    INT32 = "int32"
    INT64 = "int64"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    FP8 = "fp8"
    OPAQUE = "opaque"


class StateComponentDescriptor(ContinuumModel):
    semantic_id: NonEmptyString
    schema_version: NonEmptyString
    kind: StateKind
    symbolic_shape: tuple[NonEmptyString, ...]
    dtype_semantics: DTypeSemantics
    update_semantics: NonEmptyString
    lifetime: StateLifetime
    ownership: OwnershipScope
    exactness_requirement: ExactnessClass
    conversion_permissions: tuple[ConversionPermission, ...]
    recomputation_permission: RecomputationPermission
    compatibility_fingerprint: Digest
    integrity_hash: Digest
    provenance: tuple[Provenance, ...]
    extensions: Extensions = Field(default_factory=lambda: Extensions(root={}))

    @model_validator(mode="after")
    def unique_permissions(self) -> Self:
        if not self.provenance:
            raise ValueError("state component provenance must not be empty")
        if len(set(self.conversion_permissions)) != len(self.conversion_permissions):
            raise ValueError("conversion_permissions contains duplicates")
        if self.exactness_requirement is ExactnessClass.INCOMPATIBLE:
            raise ValueError("a captured component cannot require incompatible exactness")
        return self


class ExecutionIdentity(ContinuumModel):
    session_id: NonEmptyString
    request_id: NonEmptyString
    workflow_id: NonEmptyString | None = None
    tenant_id: NonEmptyString
    model_identity: Digest
    tokenizer_identity: Digest
    adapter_identity: Digest | None = None
    creation_epoch: NonNegativeInt
    current_owner_epoch: PositiveInt


class TokenHistoryState(ContinuumModel):
    component: StateComponentDescriptor
    input_token_ids: tuple[NonNegativeInt, ...]
    committed_output_token_ids: tuple[NonNegativeInt, ...]
    uncommitted_speculative_tokens: tuple[NonNegativeInt, ...] = ()
    token_positions: tuple[NonNegativeInt, ...] = ()
    position_offset: NonNegativeInt = 0
    attention_mask_semantics: NonEmptyString
    tokenizer_fingerprint: Digest
    normalization_contract: NonEmptyString

    @model_validator(mode="after")
    def valid_component(self) -> Self:
        if self.component.kind is not StateKind.TOKEN_HISTORY:
            raise ValueError("token history component must have kind token_history")
        expected = len(self.input_token_ids) + len(self.committed_output_token_ids)
        if self.token_positions and len(self.token_positions) != expected:
            raise ValueError("token_positions must cover input and committed output tokens")
        return self


class TokenRange(ContinuumModel):
    start: NonNegativeInt
    end_exclusive: NonNegativeInt

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.end_exclusive < self.start:
            raise ValueError("token range end must not precede start")
        return self


class AttentionLayerState(ContinuumModel):
    layer_identity: NonEmptyString
    logical_k_shape: tuple[NonNegativeInt, ...]
    logical_v_shape: tuple[NonNegativeInt, ...]
    token_range: TokenRange
    head_count: PositiveInt
    kv_head_count: PositiveInt
    head_dimension: PositiveInt
    positional_encoding_semantics: NonEmptyString
    attention_window_semantics: NonEmptyString
    dtype_semantics: DTypeSemantics

    @model_validator(mode="after")
    def valid_heads(self) -> Self:
        if self.kv_head_count > self.head_count:
            raise ValueError("kv_head_count cannot exceed head_count")
        if self.head_count % self.kv_head_count != 0:
            raise ValueError("head_count must be divisible by kv_head_count")
        return self


class AttentionState(ContinuumModel):
    component: StateComponentDescriptor
    layers: tuple[AttentionLayerState, ...]

    @model_validator(mode="after")
    def valid_component(self) -> Self:
        if self.component.kind is not StateKind.ATTENTION_KV:
            raise ValueError("attention component must have kind attention_kv")
        if not self.layers:
            raise ValueError("attention state must include at least one layer")
        identities = [item.layer_identity for item in self.layers]
        if len(set(identities)) != len(identities):
            raise ValueError("attention layer identities must be unique")
        return self


class RecurrentState(ContinuumModel):
    component: StateComponentDescriptor
    state_identifier: NonEmptyString
    layer_identity: NonEmptyString
    logical_shape: tuple[PositiveInt, ...]
    update_semantics: NonEmptyString
    dtype: DTypeSemantics
    sequence_position: NonNegativeInt
    initialization_contract: NonEmptyString

    @model_validator(mode="after")
    def valid_component(self) -> Self:
        if self.component.kind not in {StateKind.RECURRENT, StateKind.STATE_SPACE}:
            raise ValueError("recurrent state must use recurrent or state_space component kind")
        return self


class SpeculativeState(ContinuumModel):
    component: StateComponentDescriptor
    draft_model_identity: Digest
    verifier_model_identity: Digest
    accepted_prefix: tuple[NonNegativeInt, ...]
    pending_draft_tokens: tuple[NonNegativeInt, ...]
    rng_state: NonEmptyString
    verification_cursor: NonNegativeInt
    rollback_boundary: NonNegativeInt

    @model_validator(mode="after")
    def valid_component(self) -> Self:
        if self.component.kind is not StateKind.SPECULATIVE:
            raise ValueError("speculative component must have kind speculative")
        return self


class SamplerState(ContinuumModel):
    component: StateComponentDescriptor
    sampling_algorithm: NonEmptyString
    seed: NonNegativeInt
    rng_algorithm: NonEmptyString
    rng_counter: NonNegativeInt
    temperature: NonNegativeFloat
    top_k: NonNegativeInt
    top_p: Probability
    repetition_penalty: PositiveFloat
    frequency_penalty: float
    presence_penalty: float
    deterministic_required: bool
    implementation_independent_state: NonEmptyString | None = None

    @field_validator("frequency_penalty", "presence_penalty")
    @classmethod
    def finite_penalty(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("penalty must be finite")
        return value

    @model_validator(mode="after")
    def valid_component(self) -> Self:
        if self.component.kind is not StateKind.SAMPLER:
            raise ValueError("sampler component must have kind sampler")
        return self


class GuidedDecodingState(ContinuumModel):
    component: StateComponentDescriptor
    automaton_identity: Digest
    current_automaton_state: NonEmptyString
    tokenizer_contract: Digest
    accepted_prefix: tuple[NonNegativeInt, ...]
    pending_constraint_state: NonEmptyString | None = None

    @model_validator(mode="after")
    def valid_component(self) -> Self:
        if self.component.kind is not StateKind.GUIDED_DECODING:
            raise ValueError("guided decoding component must have kind guided_decoding")
        return self


class SideEffectClass(StrEnum):
    NONE = "none"
    IDEMPOTENT = "idempotent"
    AT_MOST_ONCE_EXTERNAL = "at_most_once_external"
    NON_REPLAYABLE = "non_replayable"


class ToolResult(ContinuumModel):
    call_id: NonEmptyString
    result_digest: Digest
    side_effect_class: SideEffectClass


class PendingToolCall(ContinuumModel):
    call_id: NonEmptyString
    tool_identity: NonEmptyString
    arguments_digest: Digest
    side_effect_class: SideEffectClass


class WorkflowState(ContinuumModel):
    component: StateComponentDescriptor
    current_node: NonEmptyString
    branch_identity: NonEmptyString
    completed_tool_results: tuple[ToolResult, ...] = ()
    pending_tool_calls: tuple[PendingToolCall, ...] = ()
    side_effect_class: SideEffectClass
    workflow_deadline: NonEmptyString | None = None
    continuation_contract: NonEmptyString

    @model_validator(mode="after")
    def valid_component(self) -> Self:
        if self.component.kind is not StateKind.WORKFLOW:
            raise ValueError("workflow component must have kind workflow")
        return self


class TerminalStatus(StrEnum):
    OPEN = "open"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERRORED = "errored"


class ClientDeliveryState(ContinuumModel):
    component: StateComponentDescriptor
    last_generated_token_index: int
    last_gateway_committed_token_index: int
    last_client_acknowledged_token_index: int | None
    stream_owner_epoch: PositiveInt
    terminal_status: TerminalStatus
    error_state: NonEmptyString | None = None

    @model_validator(mode="after")
    def valid_watermarks(self) -> Self:
        if self.component.kind is not StateKind.CLIENT_DELIVERY:
            raise ValueError("client delivery component must have kind client_delivery")
        if self.last_gateway_committed_token_index > self.last_generated_token_index:
            raise ValueError("gateway watermark cannot exceed generated watermark")
        if (
            self.last_client_acknowledged_token_index is not None
            and self.last_client_acknowledged_token_index > self.last_gateway_committed_token_index
        ):
            raise ValueError("client watermark cannot exceed gateway watermark")
        if min(self.last_generated_token_index, self.last_gateway_committed_token_index) < -1:
            raise ValueError("token watermarks must be -1 or non-negative")
        return self


class UnknownStateHandling(StrEnum):
    REJECT = "reject"
    PRESERVE_OPAQUE = "preserve_opaque"
    IGNORE_RECONSTRUCTIBLE = "ignore_reconstructible"


class UnknownStateComponent(ContinuumModel):
    component: StateComponentDescriptor
    namespace: NonEmptyString
    type_name: NonEmptyString
    type_version: NonEmptyString
    required_for_resume: bool
    portable_opaque: bool
    payload_digest: Digest | None = None

    @model_validator(mode="after")
    def valid_component(self) -> Self:
        if self.component.kind is not StateKind.UNKNOWN:
            raise ValueError("unknown state component must have kind unknown")
        if self.portable_opaque and self.payload_digest is None:
            raise ValueError("portable opaque state requires a payload digest")
        return self


class StateDependencyNode(ContinuumModel):
    component_id: NonEmptyString
    state_producing_fingerprint: Digest


class StateDependencyEdge(ContinuumModel):
    upstream_component_id: NonEmptyString
    downstream_component_id: NonEmptyString
    dependency_semantics: NonEmptyString
    invalidated_by_weight_change: bool


class StateDependencyGraph(ContinuumModel):
    nodes: tuple[StateDependencyNode, ...]
    edges: tuple[StateDependencyEdge, ...]

    @model_validator(mode="after")
    def valid_graph(self) -> Self:
        node_ids = [node.component_id for node in self.nodes]
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("dependency graph node identifiers must be unique")
        known = set(node_ids)
        adjacency: dict[str, list[str]] = {node: [] for node in known}
        for edge in self.edges:
            if edge.upstream_component_id not in known or edge.downstream_component_id not in known:
                raise ValueError("dependency edge references an unknown component")
            adjacency[edge.upstream_component_id].append(edge.downstream_component_id)
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node: str) -> None:
            if node in visiting:
                raise ValueError("state dependency graph must be acyclic")
            if node in visited:
                return
            visiting.add(node)
            for target in adjacency[node]:
                visit(target)
            visiting.remove(node)
            visited.add(node)

        for node in sorted(known):
            visit(node)
        return self


class QualityContract(ContinuumModel):
    metric: NonEmptyString
    maximum_loss: NonNegativeFloat
    evaluation_contract: NonEmptyString


class LogicalStateSchema(ContinuumModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    api_version: Literal["sloforge.io/continuum/v1"] = API_VERSION
    kind: Literal["LogicalStateSchema"] = "LogicalStateSchema"
    execution: ExecutionIdentity
    token_history: TokenHistoryState
    attention: AttentionState | None = None
    recurrent: tuple[RecurrentState, ...] = ()
    speculative: SpeculativeState | None = None
    sampler: SamplerState
    guided_decoding: GuidedDecodingState | None = None
    workflow: WorkflowState | None = None
    client_delivery: ClientDeliveryState
    dependency_graph: StateDependencyGraph
    unknown_state_handling: UnknownStateHandling = UnknownStateHandling.REJECT
    unknown_components: tuple[UnknownStateComponent, ...] = ()
    exactness_contract: ExactnessClass
    quality_contract: QualityContract | None = None
    extensions: Extensions = Field(default_factory=lambda: Extensions(root={}))

    def component_descriptors(self) -> tuple[StateComponentDescriptor, ...]:
        values = [
            self.token_history.component,
            self.sampler.component,
            self.client_delivery.component,
        ]
        if self.attention is not None:
            values.append(self.attention.component)
        values.extend(state.component for state in self.recurrent)
        if self.speculative is not None:
            values.append(self.speculative.component)
        if self.guided_decoding is not None:
            values.append(self.guided_decoding.component)
        if self.workflow is not None:
            values.append(self.workflow.component)
        values.extend(state.component for state in self.unknown_components)
        return tuple(values)

    @model_validator(mode="after")
    def valid_components(self) -> Self:
        components = self.component_descriptors()
        if len(components) > MAX_COMPONENTS:
            raise ValueError("logical state exceeds component bound")
        component_ids = [component.semantic_id for component in components]
        if len(set(component_ids)) != len(component_ids):
            raise ValueError("logical state component semantic identifiers must be unique")
        if set(component_ids) != {node.component_id for node in self.dependency_graph.nodes}:
            raise ValueError("dependency graph nodes must exactly cover logical state components")
        if self.token_history.tokenizer_fingerprint != self.execution.tokenizer_identity:
            raise ValueError("token history tokenizer does not match execution identity")
        if self.client_delivery.stream_owner_epoch != self.execution.current_owner_epoch:
            raise ValueError("client stream owner epoch does not match execution owner epoch")
        if (
            self.exactness_contract is ExactnessClass.QUALITY_BOUNDED
            and self.quality_contract is None
        ):
            raise ValueError("quality_bounded exactness requires a quality contract")
        if (
            self.exactness_contract is ExactnessClass.NUMERICALLY_EQUIVALENT
            and self.quality_contract is None
        ):
            raise ValueError("numerically_equivalent exactness requires a tolerance contract")
        if (
            any(
                component.exactness_requirement is ExactnessClass.NUMERICALLY_EQUIVALENT
                for component in components
            )
            and self.quality_contract is None
        ):
            raise ValueError("numerically equivalent components require a tolerance contract")
        for unknown in self.unknown_components:
            if unknown.required_for_resume:
                if self.unknown_state_handling is not UnknownStateHandling.PRESERVE_OPAQUE:
                    raise ValueError("required unknown state must use preserve_opaque handling")
                if not unknown.portable_opaque:
                    raise ValueError("required unknown state is not portable")
        return self


class RuntimeIdentity(ContinuumModel):
    runtime_name: NonEmptyString
    runtime_version: NonEmptyString
    adapter_version: NonEmptyString
    build_hash: Digest
    dependency_versions: tuple[NonEmptyString, ...]
    target_hardware: tuple[NonEmptyString, ...]


class ByteRange(ContinuumModel):
    offset: NonNegativeInt
    length: NonNegativeInt

    @property
    def end(self) -> int:
        return self.offset + self.length


class CompressionKind(StrEnum):
    NONE = "none"
    ZSTD = "zstd"
    LZ4 = "lz4"
    RUNTIME_SPECIFIC = "runtime_specific"


class EncryptionKind(StrEnum):
    NONE = "none"
    AES_256_GCM = "aes_256_gcm"
    CHACHA20_POLY1305 = "chacha20_poly1305"


class StorageLocation(ContinuumModel):
    memory_type: Literal["host", "gpu", "storage", "remote"]
    host_id: NonEmptyString
    device_id: NonEmptyString | None = None
    numa_domain: NonNegativeInt | None = None
    memory_tier: NonEmptyString
    network_rail: NonEmptyString | None = None
    fault_domain: NonEmptyString


class LayoutKind(StrEnum):
    CONTIGUOUS = "contiguous"
    PAGED = "paged"
    BLOCKED = "blocked"
    INTERLEAVED = "interleaved"
    TRANSPOSED = "transposed"
    TILED = "tiled"
    RUNTIME_SPECIFIC = "runtime_specific"


class Ordering(StrEnum):
    TOKEN_MAJOR = "token_major"
    HEAD_MAJOR = "head_major"
    LAYER_MAJOR = "layer_major"
    BLOCK_MAJOR = "block_major"
    RUNTIME_SPECIFIC = "runtime_specific"


class KVPacking(StrEnum):
    SEPARATE = "separate"
    PACKED_KV = "packed_kv"
    INTERLEAVED_KV = "interleaved_kv"


class LayoutDescriptor(ContinuumModel):
    layout_id: NonEmptyString
    kind: LayoutKind
    page_size_bytes: PositiveInt | None = None
    block_size: PositiveInt | None = None
    alignment_bytes: PositiveInt
    padding_bytes: NonNegativeInt = 0
    ordering: Ordering
    k_v_packing: KVPacking
    runtime_layout_name: NonEmptyString | None = None
    extensions: Extensions = Field(default_factory=lambda: Extensions(root={}))

    @model_validator(mode="after")
    def valid_layout(self) -> Self:
        if self.kind is LayoutKind.PAGED and self.page_size_bytes is None:
            raise ValueError("paged layout requires page_size_bytes")
        if self.kind is LayoutKind.RUNTIME_SPECIFIC and self.runtime_layout_name is None:
            raise ValueError("runtime-specific layout requires runtime_layout_name")
        return self


class ShardDescriptor(ContinuumModel):
    shard_id: NonEmptyString
    tensor_parallel_degree: PositiveInt
    pipeline_stage: NonNegativeInt
    expert_parallel_group: NonNegativeInt
    data_parallel_replica: NonNegativeInt
    rank: NonNegativeInt
    source_logical_slice: ByteRange
    destination_logical_slice: ByteRange
    shard_order: NonNegativeInt
    replicated: bool = False

    @model_validator(mode="after")
    def valid_rank(self) -> Self:
        if self.rank >= self.tensor_parallel_degree:
            raise ValueError("shard rank must be smaller than tensor_parallel_degree")
        return self


class PlacementDescriptor(ContinuumModel):
    placement_id: NonEmptyString
    location: StorageLocation
    nic_id: NonEmptyString | None = None


class QuantizationDescriptor(ContinuumModel):
    quantization_id: NonEmptyString
    format: NonEmptyString
    scale_granularity: NonEmptyString | None = None
    zero_point: bool = False
    metadata_layout: NonEmptyString | None = None
    accumulation_semantics: NonEmptyString
    exactness_class: ExactnessClass
    quality_contract: QualityContract | None = None

    @model_validator(mode="after")
    def valid_quality(self) -> Self:
        if self.exactness_class is ExactnessClass.INCOMPATIBLE:
            raise ValueError("a physical quantization descriptor cannot be incompatible")
        if self.exactness_class is ExactnessClass.EXACT_BITWISE and self.format not in {
            "none",
            "identity",
        }:
            raise ValueError("a non-identity quantized representation cannot be bitwise exact")
        if self.exactness_class is ExactnessClass.QUALITY_BOUNDED and self.quality_contract is None:
            raise ValueError("quality-bounded quantization requires a quality contract")
        return self


class AccessPatternKind(StrEnum):
    APPEND_ONLY = "append_only"
    MUTABLE = "mutable"
    READ_ONLY = "read_only"
    LAYER_SEQUENTIAL = "layer_sequential"
    SLIDING_WINDOW = "sliding_window"
    RANDOM_ACCESS = "random_access"
    SPARSE_ACCESS = "sparse_access"


class AccessPatternDescriptor(ContinuumModel):
    access_pattern_id: NonEmptyString
    kind: AccessPatternKind
    required_before_resume: bool
    streamable_before_use: bool
    recomputable: bool


class StateSegment(ContinuumModel):
    logical_state_reference: NonEmptyString
    segment_id: NonEmptyString
    logical_byte_range: ByteRange
    physical_byte_range: ByteRange
    tensor_shape: tuple[NonNegativeInt, ...]
    tensor_strides: tuple[NonNegativeInt, ...]
    storage_offset: NonNegativeInt
    allocation_id: NonEmptyString
    page_ids: tuple[NonEmptyString, ...] = ()
    chunk_ids: tuple[NonEmptyString, ...]
    current_version: NonNegativeInt
    dirty_epoch: NonNegativeInt
    checksum: Digest
    compression: CompressionKind
    encryption: EncryptionKind
    layout_id: NonEmptyString
    shard_id: NonEmptyString
    placement_id: NonEmptyString
    quantization_id: NonEmptyString | None = None
    access_pattern_id: NonEmptyString

    @model_validator(mode="after")
    def valid_shape(self) -> Self:
        if len(self.tensor_shape) != len(self.tensor_strides):
            raise ValueError("tensor shape and strides must have equal rank")
        if len(set(self.page_ids)) != len(self.page_ids):
            raise ValueError("page identifiers must be unique within a segment")
        if len(set(self.chunk_ids)) != len(self.chunk_ids):
            raise ValueError("chunk identifiers must be unique within a segment")
        return self


class PageTableEntry(ContinuumModel):
    logical_token_range: TokenRange
    physical_page_id: NonEmptyString
    page_version: NonNegativeInt
    owner_epoch: PositiveInt
    dirty: bool
    copy_on_write_reference_count: PositiveInt


class PageTableDescriptor(ContinuumModel):
    segment_id: NonEmptyString
    entries: tuple[PageTableEntry, ...]


class LogicalComponentSize(ContinuumModel):
    component_id: NonEmptyString
    logical_size_bytes: NonNegativeInt


class PhysicalStateLayout(ContinuumModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    api_version: Literal["sloforge.io/continuum/v1"] = API_VERSION
    kind: Literal["PhysicalStateLayout"] = "PhysicalStateLayout"
    layout_id: NonEmptyString
    runtime: RuntimeIdentity
    physical_plan_hash: Digest
    owner_epoch: PositiveInt
    logical_component_sizes: tuple[LogicalComponentSize, ...]
    layout_descriptors: tuple[LayoutDescriptor, ...]
    shard_descriptors: tuple[ShardDescriptor, ...]
    placement_descriptors: tuple[PlacementDescriptor, ...]
    quantization_descriptors: tuple[QuantizationDescriptor, ...] = ()
    access_patterns: tuple[AccessPatternDescriptor, ...]
    segments: tuple[StateSegment, ...]
    page_tables: tuple[PageTableDescriptor, ...] = ()
    reconstructible_runtime_state: tuple[NonEmptyString, ...] = ()
    extensions: Extensions = Field(default_factory=lambda: Extensions(root={}))

    @model_validator(mode="after")
    def validate_layout(self) -> Self:
        if len(self.segments) > MAX_SEGMENTS:
            raise ValueError("physical layout exceeds segment bound")
        collections = (
            ("logical component", [item.component_id for item in self.logical_component_sizes]),
            ("layout", [item.layout_id for item in self.layout_descriptors]),
            ("shard", [item.shard_id for item in self.shard_descriptors]),
            ("placement", [item.placement_id for item in self.placement_descriptors]),
            ("quantization", [item.quantization_id for item in self.quantization_descriptors]),
            ("access pattern", [item.access_pattern_id for item in self.access_patterns]),
            ("segment", [item.segment_id for item in self.segments]),
            ("page table", [item.segment_id for item in self.page_tables]),
        )
        for label, identifiers in collections:
            if len(set(identifiers)) != len(identifiers):
                raise ValueError(f"{label} identifiers must be unique")
        sizes = {
            item.component_id: item.logical_size_bytes for item in self.logical_component_sizes
        }
        layouts = {item.layout_id for item in self.layout_descriptors}
        shards = {item.shard_id: item for item in self.shard_descriptors}
        placements = {item.placement_id for item in self.placement_descriptors}
        quantization = {item.quantization_id for item in self.quantization_descriptors}
        access = {item.access_pattern_id for item in self.access_patterns}
        primary_ranges: dict[str, list[ByteRange]] = {key: [] for key in sizes}
        for segment in self.segments:
            if segment.logical_state_reference not in sizes:
                raise ValueError("segment references unknown logical component")
            if segment.layout_id not in layouts or segment.placement_id not in placements:
                raise ValueError("segment references unknown layout or placement")
            if segment.quantization_id is not None and segment.quantization_id not in quantization:
                raise ValueError("segment references unknown quantization descriptor")
            if segment.access_pattern_id not in access or segment.shard_id not in shards:
                raise ValueError("segment references unknown access pattern or shard")
            shard = shards[segment.shard_id]
            if segment.logical_byte_range != shard.source_logical_slice:
                raise ValueError("segment logical range must equal its shard source slice")
            if shard.source_logical_slice.length != shard.destination_logical_slice.length:
                raise ValueError("exact shard source and destination slices must have equal length")
            if shard.data_parallel_replica == 0:
                primary_ranges[segment.logical_state_reference].append(segment.logical_byte_range)
        for component_id, ranges in primary_ranges.items():
            cursor = 0
            for byte_range in sorted(ranges, key=lambda item: item.offset):
                if byte_range.offset != cursor:
                    raise ValueError(
                        f"primary shard coverage has a gap or overlap for {component_id}"
                    )
                cursor = byte_range.end
            if cursor != sizes[component_id]:
                raise ValueError(f"primary shard coverage is incomplete for {component_id}")
        allocation_ranges: dict[str, list[ByteRange]] = {}
        for segment in self.segments:
            allocation_ranges.setdefault(segment.allocation_id, []).append(
                segment.physical_byte_range
            )
        for allocation_id, ranges in allocation_ranges.items():
            previous_end = 0
            for byte_range in sorted(ranges, key=lambda item: item.offset):
                if byte_range.length > 0 and byte_range.offset < previous_end:
                    raise ValueError(f"physical segments overlap in allocation {allocation_id}")
                previous_end = max(previous_end, byte_range.end)
        segment_by_id = {item.segment_id: item for item in self.segments}
        for page_table in self.page_tables:
            page_segment = segment_by_id.get(page_table.segment_id)
            if page_segment is None:
                raise ValueError("page table references unknown segment")
            if {entry.physical_page_id for entry in page_table.entries} != set(
                page_segment.page_ids
            ):
                raise ValueError("page table entries must exactly cover segment page IDs")
            page_previous_end: int | None = None
            for entry in sorted(
                page_table.entries, key=lambda item: item.logical_token_range.start
            ):
                if entry.page_version != page_segment.current_version:
                    raise ValueError("stale page version")
                if entry.owner_epoch != self.owner_epoch:
                    raise ValueError("page owner epoch does not match layout owner epoch")
                if (
                    page_previous_end is not None
                    and entry.logical_token_range.start != page_previous_end
                ):
                    raise ValueError("page table has a token gap or overlap")
                page_previous_end = entry.logical_token_range.end_exclusive
        return self


class ExternalChunkReference(ContinuumModel):
    chunk_id: NonEmptyString
    content_hash: Digest
    size_bytes: NonNegativeInt
    tenant_security_domain: NonEmptyString
    storage_uri: NonEmptyString
    compression: CompressionKind
    encryption: EncryptionKind


class SegmentManifest(ContinuumModel):
    segment_id: NonEmptyString
    segment_hash: Digest
    chunks: tuple[ExternalChunkReference, ...]

    @model_validator(mode="after")
    def unique_chunks(self) -> Self:
        identifiers = [chunk.chunk_id for chunk in self.chunks]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("segment manifest chunk identifiers must be unique")
        return self


class CapsuleType(StrEnum):
    COMPLETE = "complete"
    INCREMENTAL = "incremental"
    FORK = "fork"
    ROLLBACK = "rollback"
    MIGRATION = "migration"
    RECOMPUTATION_ASSISTED = "recomputation_assisted"


class CapsuleIdentity(ContinuumModel):
    capsule_id: Sha256String
    capsule_type: CapsuleType
    session_id: NonEmptyString
    tenant_id: NonEmptyString
    model_hash: Digest
    tokenizer_hash: Digest
    adapter_hash: Digest | None = None
    source_runtime: RuntimeIdentity
    source_physical_plan: Digest
    owner_epoch: PositiveInt
    capture_timestamp: NonEmptyString
    git_commit: NonEmptyString
    continuum_version: NonEmptyString
    parent_capsule_id: Sha256String | None = None


class CompatibilityConstraints(ContinuumModel):
    source_compatibility_fingerprint: Digest
    required_destination_capabilities: tuple[NonEmptyString, ...]
    prohibited_conversions: tuple[ConversionPermission, ...] = ()
    recomputation_permissions: tuple[RecomputationPermission, ...] = ()
    quality_loss_budget: NonNegativeFloat | None = None
    architecture_restrictions: tuple[NonEmptyString, ...] = ()


class OwnershipLease(ContinuumModel):
    session_id: NonEmptyString
    owner_runtime: NonEmptyString
    owner_epoch: PositiveInt
    fencing_token: PositiveInt
    expiration: NonEmptyString
    coordinator_version: PositiveInt
    last_committed_state_version: NonNegativeInt
    last_committed_token_index: int


class CapsuleTransactionBinding(ContinuumModel):
    transaction_id: NonEmptyString | None = None
    ownership_lease: OwnershipLease
    fencing_token: PositiveInt
    source_epoch: PositiveInt
    destination_epoch: PositiveInt | None = None
    commit_watermark: int
    rollback_boundary: int
    pending_dirty_log_hash: Digest | None = None
    transaction_journal_hash: Digest


class VerificationClaim(ContinuumModel):
    claim_id: NonEmptyString
    property: NonEmptyString
    scope: NonEmptyString
    result: Literal["pass", "fail", "not_exercised"]
    evidence_digest: Digest
    assumptions: tuple[NonEmptyString, ...] = ()


class MigrationVerificationEvidence(ContinuumModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    api_version: Literal["sloforge.io/continuum/v1"] = API_VERSION
    kind: Literal["MigrationVerificationEvidence"] = "MigrationVerificationEvidence"
    evidence_id: NonEmptyString
    transaction_id: NonEmptyString | None
    generated_at: NonEmptyString
    capture_consistency: tuple[VerificationClaim, ...]
    segment_integrity: tuple[VerificationClaim, ...]
    conversion_verification: tuple[VerificationClaim, ...] = ()
    continuation_verification: tuple[VerificationClaim, ...] = ()
    protocol_verification: tuple[VerificationClaim, ...] = ()
    model_check_scope: NonEmptyString | None = None
    benchmark_provenance: tuple[Provenance, ...] = ()
    known_limitations: tuple[NonEmptyString, ...] = ()
    extensions: Extensions = Field(default_factory=lambda: Extensions(root={}))


class CapsuleIntegrity(ContinuumModel):
    identity_hash: Digest
    logical_state_hash: Digest
    physical_layout_hash: Digest
    segment_manifests_hash: Digest
    compatibility_hash: Digest
    transaction_binding_hash: Digest
    evidence_hash: Digest
    extensions_hash: Digest
    merkle_root: Digest


class ExecutionStateCapsule(ContinuumModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    api_version: Literal["sloforge.io/continuum/v1"] = API_VERSION
    kind: Literal["ExecutionStateCapsule"] = "ExecutionStateCapsule"
    identity: CapsuleIdentity
    logical_state: LogicalStateSchema
    physical_state: PhysicalStateLayout
    segment_manifests: tuple[SegmentManifest, ...]
    compatibility: CompatibilityConstraints
    transaction: CapsuleTransactionBinding
    evidence: MigrationVerificationEvidence
    integrity: CapsuleIntegrity
    extensions: Extensions = Field(default_factory=lambda: Extensions(root={}))

    @model_validator(mode="after")
    def structural_consistency(self) -> Self:
        if self.identity.session_id != self.logical_state.execution.session_id:
            raise ValueError("capsule session does not match logical execution")
        if self.identity.tenant_id != self.logical_state.execution.tenant_id:
            raise ValueError("capsule tenant does not match logical execution")
        if self.identity.model_hash != self.logical_state.execution.model_identity:
            raise ValueError("capsule model does not match logical execution")
        if self.identity.tokenizer_hash != self.logical_state.execution.tokenizer_identity:
            raise ValueError("capsule tokenizer does not match logical execution")
        if self.identity.adapter_hash != self.logical_state.execution.adapter_identity:
            raise ValueError("capsule adapter does not match logical execution")
        if self.identity.source_runtime != self.physical_state.runtime:
            raise ValueError("capsule source runtime does not match physical layout")
        if self.identity.source_physical_plan != self.physical_state.physical_plan_hash:
            raise ValueError("capsule source physical plan does not match physical layout")
        if (
            self.transaction.ownership_lease.owner_runtime
            != self.physical_state.runtime.runtime_name
        ):
            raise ValueError("capsule lease owner runtime does not match physical runtime")
        if self.evidence.transaction_id != self.transaction.transaction_id:
            raise ValueError("capsule evidence transaction does not match transaction binding")
        epochs = {
            self.identity.owner_epoch,
            self.logical_state.execution.current_owner_epoch,
            self.logical_state.client_delivery.stream_owner_epoch,
            self.physical_state.owner_epoch,
            self.transaction.source_epoch,
            self.transaction.ownership_lease.owner_epoch,
        }
        if len(epochs) != 1:
            raise ValueError("capsule owner epochs are inconsistent")
        if self.transaction.fencing_token != self.transaction.ownership_lease.fencing_token:
            raise ValueError("capsule fencing token does not match ownership lease")
        segment_ids = {segment.segment_id for segment in self.physical_state.segments}
        logical_component_ids = {
            component.semantic_id for component in self.logical_state.component_descriptors()
        }
        physical_component_ids = {
            component.component_id for component in self.physical_state.logical_component_sizes
        }
        if physical_component_ids != logical_component_ids:
            raise ValueError("physical component sizes must exactly cover logical components")
        manifest_ids = [manifest.segment_id for manifest in self.segment_manifests]
        if len(set(manifest_ids)) != len(manifest_ids) or set(manifest_ids) != segment_ids:
            raise ValueError("segment manifests must exactly cover physical segments")
        manifests = {manifest.segment_id: manifest for manifest in self.segment_manifests}
        for segment in self.physical_state.segments:
            if {chunk.chunk_id for chunk in manifests[segment.segment_id].chunks} != set(
                segment.chunk_ids
            ):
                raise ValueError("manifest chunks must exactly cover segment chunk IDs")
        return self


class CompatibilityCheck(ContinuumModel):
    check_id: NonEmptyString
    subject: NonEmptyString
    result: Literal["pass", "fail", "requires_recomputation", "not_applicable"]
    explanation: NonEmptyString
    evidence: tuple[Digest, ...] = ()


class RequiredConversion(ContinuumModel):
    component_id: NonEmptyString
    operation: NonEmptyString
    exactness_class: ExactnessClass


class RecomputeRequirement(ContinuumModel):
    component_id: NonEmptyString
    from_component_ids: tuple[NonEmptyString, ...]
    reason: NonEmptyString


class CompatibilityReport(ContinuumModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    api_version: Literal["sloforge.io/continuum/v1"] = API_VERSION
    kind: Literal["CompatibilityReport"] = "CompatibilityReport"
    report_id: NonEmptyString
    source_capsule_id: Sha256String
    destination_runtime: RuntimeIdentity
    destination_physical_plan: Digest
    compatibility_class: ExactnessClass
    checks: tuple[CompatibilityCheck, ...]
    rejected_classes: tuple[ExactnessClass, ...]
    required_conversions: tuple[RequiredConversion, ...] = ()
    required_recomputation: tuple[RecomputeRequirement, ...] = ()
    unsupported_state: tuple[NonEmptyString, ...] = ()
    quality_implications: tuple[QualityContract, ...] = ()
    verification_obligations: tuple[NonEmptyString, ...] = ()
    migration_restrictions: tuple[NonEmptyString, ...] = ()
    extensions: Extensions = Field(default_factory=lambda: Extensions(root={}))


class TransformationOpKind(StrEnum):
    SLICE = "slice"
    CONCATENATE = "concatenate"
    SPLIT = "split"
    RESHAPE = "reshape"
    PERMUTE = "permute"
    TRANSPOSE = "transpose"
    PAD = "pad"
    UNPAD = "unpad"
    INTERLEAVE = "interleave"
    DEINTERLEAVE = "deinterleave"
    PACK = "pack"
    UNPACK = "unpack"
    SHARD = "shard"
    RESHARD = "reshard"
    REPLICATE = "replicate"
    GATHER = "gather"
    SCATTER = "scatter"
    PAGE_REMAP = "page_remap"
    PAGE_COALESCE = "page_coalesce"
    PAGE_SPLIT = "page_split"
    DTYPE_CONVERT = "dtype_convert"
    QUANTIZE = "quantize"
    DEQUANTIZE = "dequantize"
    COMPRESS = "compress"
    DECOMPRESS = "decompress"
    CHECKSUM = "checksum"
    ENCRYPT = "encrypt"
    DECRYPT = "decrypt"
    COPY = "copy"
    ZERO_FILL = "zero_fill"
    RECONSTRUCT_METADATA = "reconstruct_metadata"
    RECOMPUTE = "recompute"
    SEND = "send"
    RECEIVE = "receive"
    WRITE_DESTINATION = "write_destination"
    VALIDATE = "validate"


class TransformCost(ContinuumModel):
    estimated_duration_ms: NonNegativeFloat
    bytes_read: NonNegativeInt
    bytes_written: NonNegativeInt
    peak_memory_bytes: NonNegativeInt
    estimate_source: NonEmptyString


class StateContractRef(ContinuumModel):
    component_id: NonEmptyString
    logical_range: ByteRange
    dtype: DTypeSemantics
    shape: tuple[NonNegativeInt, ...]


class TransformationAttributes(ContinuumModel):
    axes: tuple[NonNegativeInt, ...] = ()
    permutation: tuple[NonNegativeInt, ...] = ()
    target_shape: tuple[NonNegativeInt, ...] = ()
    padding_before: tuple[NonNegativeInt, ...] = ()
    padding_after: tuple[NonNegativeInt, ...] = ()
    page_size_bytes: PositiveInt | None = None
    shard_count: PositiveInt | None = None
    target_dtype: DTypeSemantics | None = None
    codec: NonEmptyString | None = None
    transport_id: NonEmptyString | None = None
    checksum_algorithm: Literal["sha256"] | None = None


class StateTransformationOperation(ContinuumModel):
    operation_id: NonEmptyString
    kind: TransformationOpKind
    dependencies: tuple[NonEmptyString, ...]
    sources: tuple[StateContractRef, ...]
    destinations: tuple[StateContractRef, ...]
    preconditions: tuple[NonEmptyString, ...]
    postconditions: tuple[NonEmptyString, ...]
    exactness_class: ExactnessClass
    attributes: TransformationAttributes
    ownership_behavior: NonEmptyString
    target_device: NonEmptyString
    estimated_cost: TransformCost
    streamable: bool
    verification_obligations: tuple[NonEmptyString, ...]
    fallback_implementation: NonEmptyString


class MemoryAllocation(ContinuumModel):
    allocation_id: NonEmptyString
    memory_type: NonEmptyString
    size_bytes: NonNegativeInt
    lifetime_start_operation: NonEmptyString
    lifetime_end_operation: NonEmptyString


class StateTransformationIR(ContinuumModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    api_version: Literal["sloforge.io/continuum/v1"] = API_VERSION
    kind: Literal["StateTransformationIR"] = "StateTransformationIR"
    transformation_id: NonEmptyString
    source_layout_hash: Digest
    destination_layout_hash: Digest
    compatibility_report_hash: Digest
    operations: tuple[StateTransformationOperation, ...]
    memory_plan: tuple[MemoryAllocation, ...]
    maximum_buffer_bytes: NonNegativeInt
    chunk_order: tuple[NonEmptyString, ...]
    rollback_behavior: NonEmptyString
    predicted_duration_ms: NonNegativeFloat
    uncertainty_ms: NonNegativeFloat
    extensions: Extensions = Field(default_factory=lambda: Extensions(root={}))

    @model_validator(mode="after")
    def valid_dag(self) -> Self:
        identifiers = [operation.operation_id for operation in self.operations]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("transformation operation identifiers must be unique")
        known: set[str] = set()
        for operation in self.operations:
            if not set(operation.dependencies).issubset(known):
                raise ValueError("operations must be topologically ordered")
            known.add(operation.operation_id)
        if not set(self.chunk_order).issubset(known):
            raise ValueError("chunk order references unknown operation")
        peak = sum(allocation.size_bytes for allocation in self.memory_plan)
        if peak > self.maximum_buffer_bytes:
            raise ValueError("memory plan exceeds maximum bounded buffer")
        return self


class MigrationStrategy(StrEnum):
    STOP_AND_COPY = "stop_and_copy"
    PRE_COPY = "pre_copy"
    HYBRID_PRE_COPY = "hybrid_pre_copy"
    RECOMPUTATION_ASSISTED = "recomputation_assisted"
    CONSTRAINED_LAZY = "constrained_lazy"
    FORK = "fork"
    CLONE = "clone"


class TransportSelection(ContinuumModel):
    transport_id: NonEmptyString
    chunk_size_bytes: PositiveInt
    concurrency: PositiveInt
    bandwidth_limit_bytes_per_second: PositiveInt | None = None
    deadline_ms: PositiveInt


class MigrationPlan(ContinuumModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    api_version: Literal["sloforge.io/continuum/v1"] = API_VERSION
    kind: Literal["MigrationPlan"] = "MigrationPlan"
    plan_id: NonEmptyString
    source_session_id: NonEmptyString
    destination_runtime: RuntimeIdentity
    destination_physical_plan: Digest
    strategy: MigrationStrategy
    exactness_requirement: ExactnessClass
    conversion_plan_hash: Digest
    transport: TransportSelection
    pre_copy_rounds: NonNegativeInt
    cutover_threshold_bytes: NonNegativeInt
    destination_warmup_actions: tuple[NonEmptyString, ...]
    validation_actions: tuple[NonEmptyString, ...]
    rollback_capsule_id: Sha256String | None
    expected_interruption_ms: NonNegativeFloat
    expected_total_time_ms: NonNegativeFloat
    expected_source_overhead_ms: NonNegativeFloat
    expected_destination_overhead_ms: NonNegativeFloat
    expected_bytes: NonNegativeInt
    expected_temporary_memory_bytes: NonNegativeInt
    expected_cost_usd: NonNegativeFloat
    failure_probability: Probability
    uncertainty_ms: NonNegativeFloat
    rejected_alternatives: tuple[NonEmptyString, ...]
    required_before_resume_segments: tuple[NonEmptyString, ...]
    extensions: Extensions = Field(default_factory=lambda: Extensions(root={}))

    @model_validator(mode="after")
    def valid_lazy_plan(self) -> Self:
        if (
            self.strategy is MigrationStrategy.CONSTRAINED_LAZY
            and not self.required_before_resume_segments
        ):
            raise ValueError(
                "constrained lazy migration must declare required-before-resume segments"
            )
        return self


class TransactionPhase(StrEnum):
    PROPOSED = "proposed"
    COMPATIBILITY_VALIDATED = "compatibility_validated"
    DESTINATION_PREPARING = "destination_preparing"
    PRECOPYING = "precopying"
    DELTA_SYNCING = "delta_syncing"
    CUTOVER_REQUESTED = "cutover_requested"
    SOURCE_QUIESCING = "source_quiescing"
    SOURCE_FROZEN = "source_frozen"
    FINAL_DELTA_TRANSFERRING = "final_delta_transferring"
    DESTINATION_IMPORTING = "destination_importing"
    DESTINATION_VALIDATING = "destination_validating"
    COMMIT_INTENT_RECORDED = "commit_intent_recorded"
    OWNERSHIP_COMMITTED = "ownership_committed"
    GATEWAY_SWITCHING = "gateway_switching"
    DESTINATION_ACTIVE = "destination_active"
    SOURCE_DRAINING = "source_draining"
    COMPLETED = "completed"
    REJECTED = "rejected"
    ABORTING = "aborting"
    ROLLED_BACK = "rolled_back"
    FAILED_BEFORE_COMMIT = "failed_before_commit"
    FAILED_AFTER_COMMIT = "failed_after_commit"
    DESTINATION_LOST = "destination_lost"
    SOURCE_LOST = "source_lost"
    COORDINATOR_UNAVAILABLE = "coordinator_unavailable"
    OPERATOR_REQUIRED = "operator_required"


class TransactionAcknowledgment(ContinuumModel):
    actor: NonEmptyString
    phase: TransactionPhase
    owner_epoch: PositiveInt
    state_hash: Digest
    recorded_at: NonEmptyString


class StateTransaction(ContinuumModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    api_version: Literal["sloforge.io/continuum/v1"] = API_VERSION
    kind: Literal["StateTransaction"] = "StateTransaction"
    transaction_id: NonEmptyString
    source_owner: NonEmptyString
    destination_candidate: NonEmptyString
    source_epoch: PositiveInt
    proposed_destination_epoch: PositiveInt
    fencing_token: PositiveInt
    migration_plan_hash: Digest
    current_phase: TransactionPhase
    commit_watermark: int
    rollback_watermark: int
    state_hashes: tuple[Digest, ...]
    acknowledgments: tuple[TransactionAcknowledgment, ...]
    timeout_at: NonEmptyString
    failure_reason: NonEmptyString | None = None
    journal_hash: Digest
    extensions: Extensions = Field(default_factory=lambda: Extensions(root={}))

    @model_validator(mode="after")
    def valid_epochs(self) -> Self:
        if self.proposed_destination_epoch <= self.source_epoch:
            raise ValueError("destination epoch must advance source epoch")
        if self.rollback_watermark > self.commit_watermark:
            raise ValueError("rollback watermark cannot exceed commit watermark")
        return self


ContinuumDocument: TypeAlias = (
    LogicalStateSchema
    | PhysicalStateLayout
    | ExecutionStateCapsule
    | CompatibilityReport
    | StateTransformationIR
    | MigrationPlan
    | StateTransaction
    | MigrationVerificationEvidence
)
