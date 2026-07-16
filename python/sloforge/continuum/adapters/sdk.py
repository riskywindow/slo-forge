"""Fail-closed runtime boundary for portable Continuum execution state.

The adapter boundary deliberately contains values rather than process-local handles.
It is safe to serialize the records in this module, but callers should place the
payloads in an authenticated :class:`ExecutionStateCapsule` before crossing a trust
boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256
from typing import NoReturn

MAX_SEGMENT_BYTES = 64 * 1024 * 1024


class RuntimeCapability(StrEnum):
    INSPECT = "inspect"
    CONSISTENT_SNAPSHOT = "consistent_snapshot"
    IMPORT = "import"
    DIRTY_TRACKING = "dirty_tracking"
    FINAL_DELTA = "final_delta"
    TOKEN_BOUNDARY_QUIESCE = "token_boundary_quiesce"
    FENCING = "fencing"
    DRY_RUN_VALIDATION = "dry_run_validation"
    PAUSE_RESUME = "pause_resume"
    CANCELLATION = "cancellation"
    STREAMING = "streaming"
    SIMULATED_DEVICES = "simulated_devices"


class StateKind(StrEnum):
    ATTENTION_KEY = "attention_key"
    ATTENTION_VALUE = "attention_value"
    ATTENTION_PACKED_KV = "attention_packed_kv"
    RECURRENT = "recurrent"
    SAMPLER = "sampler"
    GUIDED_DECODING = "guided_decoding"
    TOKEN_HISTORY = "token_history"
    CLIENT_DELIVERY = "client_delivery"


class LayoutKind(StrEnum):
    PAGED_TOKEN_MAJOR_SEPARATE_KV = "paged_token_major_separate_kv"
    PAGED_HEAD_MAJOR_PACKED_KV = "paged_head_major_packed_kv"


class DirtyTrackingStrategy(StrEnum):
    EXPLICIT_SEGMENT_VERSIONING = "explicit_segment_versioning"
    APPEND_LOG = "append_log"
    COPY_ON_WRITE = "copy_on_write"
    HASH_COMPARISON = "hash_comparison"


class SessionLifecycle(StrEnum):
    PREPARED = "prepared"
    ACTIVE = "active"
    PAUSED = "paused"
    FENCED = "fenced"
    CANCELLED = "cancelled"
    TERMINAL = "terminal"


class ClientTerminalStatus(StrEnum):
    OPEN = "open"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERRORED = "errored"


class FailurePoint(StrEnum):
    CREATE_SESSION = "create_session"
    GENERATE = "generate"
    BEGIN_SNAPSHOT = "begin_snapshot"
    READ_SEGMENT = "read_segment"
    START_DIRTY_TRACKING = "start_dirty_tracking"
    READ_DELTA = "read_delta"
    QUIESCE = "quiesce"
    PREPARE_IMPORT = "prepare_import"
    IMPORT = "import"
    APPLY_DELTA = "apply_delta"
    VALIDATE_IMPORT = "validate_import"
    ACTIVATE = "activate"


class AdapterError(RuntimeError):
    """Typed adapter failure safe to expose without state payloads."""

    code = "adapter_error"

    def __init__(self, message: str, *, operation: str, session_id: str | None = None) -> None:
        super().__init__(message)
        self.operation = operation
        self.session_id = session_id


class UnsupportedCapabilityError(AdapterError):
    code = "unsupported_capability"


class AdapterUnavailableError(AdapterError):
    code = "adapter_unavailable"


class SessionNotFoundError(AdapterError):
    code = "session_not_found"


class SessionStateError(AdapterError):
    code = "invalid_session_state"


class ResourceLimitError(AdapterError):
    code = "resource_limit"


class SnapshotConsistencyError(AdapterError):
    code = "snapshot_consistency"


class SegmentIntegrityError(AdapterError):
    code = "segment_integrity"


class ImportValidationError(AdapterError):
    code = "import_validation"


class StaleOwnerEpochError(AdapterError):
    code = "stale_owner_epoch"


class DirtyLogOverflowError(AdapterError):
    code = "dirty_log_overflow"


class StaleDeltaError(AdapterError):
    code = "stale_delta"


class InjectedFailureError(AdapterError):
    code = "injected_failure"

    def __init__(self, point: FailurePoint, *, session_id: str | None) -> None:
        super().__init__(
            f"deterministic failure injected at {point.value}",
            operation=point.value,
            session_id=session_id,
        )
        self.point = point


def checksum_bytes(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _require_non_empty(value: str, field_name: str) -> None:
    if not value or len(value) > 512:
        raise ValueError(f"{field_name} must contain 1..512 characters")


def _require_hash(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase sha256 hex digest")


@dataclass(frozen=True, slots=True)
class RuntimeIdentity:
    runtime_name: str
    runtime_version: str
    adapter_version: str
    build_hash: str
    dependency_versions: tuple[tuple[str, str], ...]
    target_hardware: str

    def __post_init__(self) -> None:
        _require_non_empty(self.runtime_name, "runtime_name")
        _require_non_empty(self.runtime_version, "runtime_version")
        _require_non_empty(self.adapter_version, "adapter_version")
        _require_hash(self.build_hash, "build_hash")
        _require_non_empty(self.target_hardware, "target_hardware")
        if len(self.dependency_versions) > 128:
            raise ValueError("dependency_versions exceeds 128 entries")


@dataclass(frozen=True, slots=True)
class RuntimeLayout:
    kind: LayoutKind
    page_size_tokens: int
    tensor_parallel_degree: int
    ordering: str
    kv_packing: str
    alignment_bytes: int
    simulated_devices: tuple[str, ...]

    def __post_init__(self) -> None:
        if not 1 <= self.page_size_tokens <= 1_048_576:
            raise ValueError("page_size_tokens must be in 1..1048576")
        if not 1 <= self.tensor_parallel_degree <= 1024:
            raise ValueError("tensor_parallel_degree must be in 1..1024")
        if self.alignment_bytes <= 0 or self.alignment_bytes & (self.alignment_bytes - 1):
            raise ValueError("alignment_bytes must be a positive power of two")
        if len(self.simulated_devices) != self.tensor_parallel_degree:
            raise ValueError("simulated device count must equal tensor_parallel_degree")
        if len(set(self.simulated_devices)) != len(self.simulated_devices):
            raise ValueError("simulated device identifiers must be unique")
        expected = {
            LayoutKind.PAGED_TOKEN_MAJOR_SEPARATE_KV: ("token-major", "separate-k-v"),
            LayoutKind.PAGED_HEAD_MAJOR_PACKED_KV: ("head-major", "packed-k-v"),
        }[self.kind]
        if (self.ordering, self.kv_packing) != expected:
            raise ValueError("layout ordering or K/V packing contradicts the layout kind")


@dataclass(frozen=True, slots=True)
class CapabilityMatrix:
    runtime: RuntimeIdentity
    operations: frozenset[RuntimeCapability]
    state_types: frozenset[StateKind]
    layouts: tuple[RuntimeLayout, ...]
    dirty_tracking_strategies: frozenset[DirtyTrackingStrategy]
    max_sessions: int
    max_open_snapshots: int
    max_dirty_events: int
    max_segment_bytes: int = MAX_SEGMENT_BYTES
    max_stream_buffer_tokens: int = 1

    def __post_init__(self) -> None:
        if not self.operations:
            raise ValueError("an adapter must publish at least one operation")
        if not self.layouts:
            raise ValueError("an adapter must publish at least one layout")
        for name, value in (
            ("max_sessions", self.max_sessions),
            ("max_open_snapshots", self.max_open_snapshots),
            ("max_dirty_events", self.max_dirty_events),
            ("max_segment_bytes", self.max_segment_bytes),
            ("max_stream_buffer_tokens", self.max_stream_buffer_tokens),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")

    def require(self, capability: RuntimeCapability) -> None:
        if capability not in self.operations:
            raise UnsupportedCapabilityError(
                f"runtime {self.runtime.runtime_name} does not support {capability.value}",
                operation=capability.value,
            )


@dataclass(frozen=True, slots=True)
class ModelContract:
    model_id: str
    model_hash: str
    tokenizer_hash: str
    adapter_hash: str
    state_producer_hash: str
    recurrent_update_hash: str
    positional_encoding_hash: str
    vocabulary_size: int

    def __post_init__(self) -> None:
        _require_non_empty(self.model_id, "model_id")
        for name, value in (
            ("model_hash", self.model_hash),
            ("tokenizer_hash", self.tokenizer_hash),
            ("adapter_hash", self.adapter_hash),
            ("state_producer_hash", self.state_producer_hash),
            ("recurrent_update_hash", self.recurrent_update_hash),
            ("positional_encoding_hash", self.positional_encoding_hash),
        ):
            _require_hash(value, name)
        if not 8 <= self.vocabulary_size <= 1_000_000:
            raise ValueError("vocabulary_size must be in 8..1000000")


@dataclass(frozen=True, slots=True)
class SamplerSnapshot:
    algorithm: str
    seed: int
    counter: int
    temperature_milli: int
    top_k: int
    top_p_millionths: int

    def __post_init__(self) -> None:
        if self.algorithm != "continuum-counter-v1":
            raise ValueError("unsupported implementation-independent sampler algorithm")
        if self.counter < 0:
            raise ValueError("sampler counter cannot be negative")
        if self.temperature_milli <= 0:
            raise ValueError("temperature_milli must be positive")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if not 1 <= self.top_p_millionths <= 1_000_000:
            raise ValueError("top_p_millionths must be in 1..1000000")


@dataclass(frozen=True, slots=True)
class GuidedDecodingSnapshot:
    automaton_id: str
    automaton_hash: str
    state: int
    accepted_prefix_length: int

    def __post_init__(self) -> None:
        _require_non_empty(self.automaton_id, "automaton_id")
        _require_hash(self.automaton_hash, "automaton_hash")
        if not 0 <= self.state < 4:
            raise ValueError("guided automaton state must be in 0..3")
        if self.accepted_prefix_length < 0:
            raise ValueError("accepted_prefix_length cannot be negative")


@dataclass(frozen=True, slots=True)
class ClientDeliverySnapshot:
    last_generated_token_index: int
    last_gateway_committed_token_index: int
    last_client_acknowledged_token_index: int
    stream_owner_epoch: int
    terminal_status: ClientTerminalStatus

    def __post_init__(self) -> None:
        generated = self.last_generated_token_index
        gateway = self.last_gateway_committed_token_index
        client = self.last_client_acknowledged_token_index
        if not -1 <= client <= gateway <= generated:
            raise ValueError("client <= gateway <= generated watermark invariant violated")
        if self.stream_owner_epoch <= 0:
            raise ValueError("stream_owner_epoch must be positive")


@dataclass(frozen=True, slots=True)
class LogicalStateManifest:
    schema_version: str
    session_id: str
    request_id: str
    tenant_id: str
    model: ModelContract
    input_token_ids: tuple[int, ...]
    committed_output_token_ids: tuple[int, ...]
    uncommitted_speculative_token_ids: tuple[int, ...]
    attention_layer_count: int
    attention_head_count: int
    attention_kv_head_count: int
    attention_head_dimension: int
    positional_encoding_semantics: str
    attention_window_semantics: str
    recurrent_state: tuple[tuple[int, ...], ...]
    sampler: SamplerSnapshot
    guided_decoding: GuidedDecodingSnapshot
    client_delivery: ClientDeliverySnapshot
    owner_epoch: int
    state_version: int
    dirty_epoch: int
    continuation_hash: str

    def __post_init__(self) -> None:
        if self.schema_version != "continuum.logical.runtime.v1":
            raise ValueError("unsupported runtime logical manifest schema")
        for string_field_name, string_value in (
            ("session_id", self.session_id),
            ("request_id", self.request_id),
            ("tenant_id", self.tenant_id),
        ):
            _require_non_empty(string_value, string_field_name)
        if self.owner_epoch <= 0:
            raise ValueError("owner_epoch must be positive")
        for numeric_field_name, numeric_value in (
            ("attention_layer_count", self.attention_layer_count),
            ("attention_head_count", self.attention_head_count),
            ("attention_kv_head_count", self.attention_kv_head_count),
            ("attention_head_dimension", self.attention_head_dimension),
        ):
            if numeric_value <= 0:
                raise ValueError(f"{numeric_field_name} must be positive")
        if self.attention_kv_head_count > self.attention_head_count:
            raise ValueError("KV head count cannot exceed attention head count")
        if self.attention_head_count % self.attention_kv_head_count:
            raise ValueError("attention head count must be divisible by KV head count")
        if len(self.recurrent_state) != self.attention_layer_count:
            raise ValueError("recurrent state must contain exactly one row per layer")
        if not self.recurrent_state or not self.recurrent_state[0]:
            raise ValueError("recurrent state rows must not be empty")
        recurrent_width = len(self.recurrent_state[0])
        if any(len(row) != recurrent_width for row in self.recurrent_state):
            raise ValueError("recurrent state rows must have a uniform width")
        if not 0 <= self.sampler.seed < 1 << 64:
            raise ValueError("sampler seed must be an unsigned 64-bit integer")
        _require_non_empty(self.positional_encoding_semantics, "positional_encoding_semantics")
        _require_non_empty(self.attention_window_semantics, "attention_window_semantics")
        if self.state_version < 0 or self.dirty_epoch < 0:
            raise ValueError("state versions cannot be negative")
        if self.state_version != self.dirty_epoch:
            raise ValueError("reference manifest state_version must equal dirty_epoch")
        _require_hash(self.continuation_hash, "continuation_hash")
        if self.guided_decoding.accepted_prefix_length != len(self.committed_output_token_ids):
            raise ValueError("guided accepted prefix must match committed output length")
        expected_index = len(self.committed_output_token_ids) - 1
        if self.client_delivery.last_generated_token_index != expected_index:
            raise ValueError("generated watermark must match committed output history")
        for token in (*self.input_token_ids, *self.committed_output_token_ids):
            if not 0 <= token < self.model.vocabulary_size:
                raise ValueError("token is outside the model vocabulary")


@dataclass(frozen=True, slots=True)
class SegmentDescriptor:
    segment_id: str
    semantic_id: str
    state_kind: StateKind
    logical_shape: tuple[int, ...]
    dtype: str
    encoding: str
    layer: int | None
    shard_rank: int
    shard_count: int
    token_start: int
    token_end: int
    head_start: int
    head_end: int
    page_id: int | None
    version: int
    dirty_epoch: int
    required_before_resume: bool
    payload_bytes: int
    checksum: str

    def __post_init__(self) -> None:
        _require_non_empty(self.segment_id, "segment_id")
        _require_non_empty(self.semantic_id, "semantic_id")
        _require_hash(self.checksum, "checksum")
        if any(dimension < 0 for dimension in self.logical_shape):
            raise ValueError("logical shape dimensions cannot be negative")
        if not 0 <= self.shard_rank < self.shard_count:
            raise ValueError("invalid shard rank")
        if not 0 <= self.token_start <= self.token_end:
            raise ValueError("invalid token range")
        if not 0 <= self.head_start <= self.head_end:
            raise ValueError("invalid head range")
        if self.version < 0 or self.dirty_epoch < 0:
            raise ValueError("segment versions cannot be negative")
        if not 0 <= self.payload_bytes <= MAX_SEGMENT_BYTES:
            raise ValueError("segment payload exceeds the hard SDK bound")


@dataclass(frozen=True, slots=True)
class StateSegment:
    descriptor: SegmentDescriptor
    payload: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if len(self.payload) != self.descriptor.payload_bytes:
            raise ValueError("segment payload length does not match its descriptor")
        if checksum_bytes(self.payload) != self.descriptor.checksum:
            raise SegmentIntegrityError(
                "segment checksum mismatch",
                operation="construct_segment",
            )


@dataclass(frozen=True, slots=True)
class PageTableEntry:
    logical_state_id: str
    layer: int
    shard_rank: int
    logical_token_start: int
    logical_token_end: int
    physical_page_id: int
    segment_ids: tuple[str, ...]
    page_version: int
    dirty_epoch: int
    owner_epoch: int
    copy_on_write_refs: int = 1

    def __post_init__(self) -> None:
        if self.logical_token_start < 0 or self.logical_token_end < self.logical_token_start:
            raise ValueError("invalid page token range")
        if self.physical_page_id < 0:
            raise ValueError("physical_page_id cannot be negative")
        if not self.segment_ids:
            raise ValueError("page table entry must reference at least one segment")
        if self.page_version < 0 or self.dirty_epoch < 0 or self.owner_epoch <= 0:
            raise ValueError("invalid page or ownership version")
        if self.copy_on_write_refs <= 0:
            raise ValueError("copy_on_write_refs must be positive")


@dataclass(frozen=True, slots=True)
class SnapshotHandle:
    snapshot_id: str
    session_id: str
    owner_epoch: int
    state_version: int
    dirty_epoch: int
    segment_count: int

    def __post_init__(self) -> None:
        _require_hash(self.snapshot_id, "snapshot_id")
        _require_non_empty(self.session_id, "session_id")
        if self.owner_epoch <= 0 or self.state_version < 0 or self.dirty_epoch < 0:
            raise ValueError("invalid snapshot version")
        if self.segment_count <= 0:
            raise ValueError("snapshot must contain at least one segment")


@dataclass(frozen=True, slots=True)
class CapturedState:
    handle: SnapshotHandle
    runtime: RuntimeIdentity
    layout: RuntimeLayout
    logical: LogicalStateManifest
    segments: tuple[StateSegment, ...]
    page_table: tuple[PageTableEntry, ...]

    def verify(self) -> None:
        if self.handle.session_id != self.logical.session_id:
            raise SnapshotConsistencyError(
                "snapshot session identity mismatch",
                operation="verify_snapshot",
                session_id=self.logical.session_id,
            )
        if self.handle.owner_epoch != self.logical.owner_epoch:
            raise SnapshotConsistencyError(
                "snapshot owner epoch mismatch",
                operation="verify_snapshot",
                session_id=self.logical.session_id,
            )
        if self.handle.state_version != self.logical.state_version:
            raise SnapshotConsistencyError(
                "snapshot state version mismatch",
                operation="verify_snapshot",
                session_id=self.logical.session_id,
            )
        if self.handle.dirty_epoch != self.logical.dirty_epoch:
            raise SnapshotConsistencyError(
                "snapshot dirty epoch mismatch",
                operation="verify_snapshot",
                session_id=self.logical.session_id,
            )
        if self.handle.segment_count != len(self.segments):
            raise SnapshotConsistencyError(
                "snapshot segment count mismatch",
                operation="verify_snapshot",
                session_id=self.logical.session_id,
            )
        segment_ids = {segment.descriptor.segment_id for segment in self.segments}
        segments_by_id = {
            segment.descriptor.segment_id: segment.descriptor for segment in self.segments
        }
        if len(segment_ids) != len(self.segments):
            raise SnapshotConsistencyError(
                "snapshot contains duplicate segment identifiers",
                operation="verify_snapshot",
                session_id=self.logical.session_id,
            )
        for segment in self.segments:
            if checksum_bytes(segment.payload) != segment.descriptor.checksum:
                raise SegmentIntegrityError(
                    "snapshot segment checksum mismatch",
                    operation="verify_snapshot",
                    session_id=self.logical.session_id,
                )
        paged_segment_ids = {
            segment.descriptor.segment_id
            for segment in self.segments
            if segment.descriptor.page_id is not None
        }
        referenced_page_segment_ids: list[str] = []
        page_keys: set[tuple[int, int, int]] = set()
        for page in self.page_table:
            page_key = (page.layer, page.shard_rank, page.physical_page_id)
            if page_key in page_keys:
                raise SnapshotConsistencyError(
                    "snapshot contains a duplicate physical page-table entry",
                    operation="verify_snapshot",
                    session_id=self.logical.session_id,
                )
            page_keys.add(page_key)
            if page.owner_epoch != self.logical.owner_epoch:
                raise SnapshotConsistencyError(
                    "page owner epoch mismatch",
                    operation="verify_snapshot",
                    session_id=self.logical.session_id,
                )
            if any(segment_id not in segment_ids for segment_id in page.segment_ids):
                raise SnapshotConsistencyError(
                    "page table references a missing segment",
                    operation="verify_snapshot",
                    session_id=self.logical.session_id,
                )
            if len(set(page.segment_ids)) != len(page.segment_ids):
                raise SnapshotConsistencyError(
                    "page table contains duplicate segment references",
                    operation="verify_snapshot",
                    session_id=self.logical.session_id,
                )
            for segment_id in page.segment_ids:
                referenced_page_segment_ids.append(segment_id)
                descriptor = segments_by_id[segment_id]
                if descriptor.page_id != page.physical_page_id:
                    raise SnapshotConsistencyError(
                        "page table physical page differs from segment",
                        operation="verify_snapshot",
                        session_id=self.logical.session_id,
                    )
                if descriptor.layer != page.layer or descriptor.shard_rank != page.shard_rank:
                    raise SnapshotConsistencyError(
                        "page table placement differs from segment",
                        operation="verify_snapshot",
                        session_id=self.logical.session_id,
                    )
                if (
                    descriptor.token_start != page.logical_token_start
                    or descriptor.token_end != page.logical_token_end
                ):
                    raise SnapshotConsistencyError(
                        "page table token range differs from segment",
                        operation="verify_snapshot",
                        session_id=self.logical.session_id,
                    )
                if (
                    descriptor.version != page.page_version
                    or descriptor.dirty_epoch != page.dirty_epoch
                ):
                    raise SnapshotConsistencyError(
                        "page table version is stale",
                        operation="verify_snapshot",
                        session_id=self.logical.session_id,
                    )
        if set(referenced_page_segment_ids) != paged_segment_ids or len(
            referenced_page_segment_ids
        ) != len(paged_segment_ids):
            raise SnapshotConsistencyError(
                "page table must reference every paged segment exactly once",
                operation="verify_snapshot",
                session_id=self.logical.session_id,
            )


@dataclass(frozen=True, slots=True)
class DirtyTrackingHandle:
    tracking_id: str
    session_id: str
    baseline_epoch: int
    last_exported_epoch: int

    def __post_init__(self) -> None:
        _require_hash(self.tracking_id, "tracking_id")
        if self.baseline_epoch < 0 or self.last_exported_epoch < self.baseline_epoch:
            raise ValueError("invalid dirty tracking range")


@dataclass(frozen=True, slots=True)
class DirtyDelta:
    tracking_id: str
    session_id: str
    from_epoch: int
    to_epoch: int
    owner_epoch: int
    source_layout: RuntimeLayout
    logical: LogicalStateManifest
    changed_segments: tuple[StateSegment, ...]
    page_table: tuple[PageTableEntry, ...]
    final: bool

    def __post_init__(self) -> None:
        _require_hash(self.tracking_id, "tracking_id")
        if self.from_epoch < 0 or self.to_epoch < self.from_epoch:
            raise ValueError("invalid delta epoch range")
        if self.logical.dirty_epoch != self.to_epoch:
            raise ValueError("delta logical manifest does not match to_epoch")
        if self.owner_epoch != self.logical.owner_epoch:
            raise ValueError("delta owner epoch does not match logical manifest")
        if any(
            segment.descriptor.dirty_epoch <= self.from_epoch for segment in self.changed_segments
        ):
            raise ValueError("delta includes a segment not dirtied after from_epoch")


@dataclass(frozen=True, slots=True)
class SessionMetadata:
    session_id: str
    request_id: str
    tenant_id: str
    lifecycle: SessionLifecycle
    owner_epoch: int
    state_version: int
    committed_output_index: int
    client_visible_index: int
    layout: RuntimeLayout
    model: ModelContract


@dataclass(frozen=True, slots=True)
class TokenEvent:
    session_id: str
    owner_epoch: int
    token_index: int
    token_id: int
    state_commit_version: int
    transaction_id: str | None

    def __post_init__(self) -> None:
        if self.owner_epoch <= 0 or self.token_index < 0 or self.state_commit_version <= 0:
            raise ValueError("invalid token event version")


@dataclass(frozen=True, slots=True)
class ImportValidation:
    session_id: str
    source_logical_hash: str
    imported_logical_hash: str
    dry_run_next_token: int
    segment_count: int
    structurally_valid: bool
    continuation_valid: bool

    def __post_init__(self) -> None:
        _require_hash(self.source_logical_hash, "source_logical_hash")
        _require_hash(self.imported_logical_hash, "imported_logical_hash")
        if self.segment_count <= 0:
            raise ValueError("validation must cover at least one segment")


@dataclass(frozen=True, slots=True)
class FailureRule:
    point: FailurePoint
    trigger_on_call: int
    session_id: str | None = None
    repeat: bool = False

    def __post_init__(self) -> None:
        if self.trigger_on_call <= 0:
            raise ValueError("trigger_on_call must be positive")


class ContinuumRuntimeAdapter:
    """Stable, fail-closed adapter SDK.

    Concrete adapters override only operations present in their capability matrix.
    Every default operation returns an actionable typed error.
    """

    @property
    def identity(self) -> RuntimeIdentity:
        return self.capabilities.runtime

    @property
    def capabilities(self) -> CapabilityMatrix:
        self._unsupported(RuntimeCapability.INSPECT)

    def _unsupported(self, capability: RuntimeCapability) -> NoReturn:
        raise UnsupportedCapabilityError(
            f"adapter does not implement {capability.value}",
            operation=capability.value,
        )

    def create_session(
        self,
        *,
        session_id: str,
        request_id: str,
        tenant_id: str,
        input_token_ids: tuple[int, ...],
        seed: int,
        owner_epoch: int = 1,
    ) -> SessionMetadata:
        self._unsupported(RuntimeCapability.IMPORT)

    def inspect_session(self, session_id: str) -> SessionMetadata:
        self._unsupported(RuntimeCapability.INSPECT)

    def list_sessions(self) -> tuple[SessionMetadata, ...]:
        self._unsupported(RuntimeCapability.INSPECT)

    def generate_token(self, session_id: str, *, transaction_id: str | None = None) -> TokenEvent:
        self._unsupported(RuntimeCapability.STREAMING)

    def dry_run_next_token(self, session_id: str) -> int:
        self._unsupported(RuntimeCapability.DRY_RUN_VALIDATION)

    def acknowledge_gateway(
        self,
        session_id: str,
        *,
        token_index: int,
        owner_epoch: int,
    ) -> SessionMetadata:
        self._unsupported(RuntimeCapability.STREAMING)

    def stream_tokens(
        self,
        session_id: str,
        *,
        count: int,
        transaction_id: str | None = None,
    ) -> tuple[TokenEvent, ...]:
        self._unsupported(RuntimeCapability.STREAMING)

    def pause_session(self, session_id: str) -> SessionMetadata:
        self._unsupported(RuntimeCapability.PAUSE_RESUME)

    def resume_session(self, session_id: str, *, expected_owner_epoch: int) -> SessionMetadata:
        self._unsupported(RuntimeCapability.PAUSE_RESUME)

    def cancel_session(self, session_id: str) -> SessionMetadata:
        self._unsupported(RuntimeCapability.CANCELLATION)

    def begin_consistent_snapshot(self, session_id: str) -> SnapshotHandle:
        self._unsupported(RuntimeCapability.CONSISTENT_SNAPSHOT)

    def enumerate_state_segments(self, handle: SnapshotHandle) -> tuple[SegmentDescriptor, ...]:
        self._unsupported(RuntimeCapability.CONSISTENT_SNAPSHOT)

    def read_state_segment(self, handle: SnapshotHandle, segment_id: str) -> StateSegment:
        self._unsupported(RuntimeCapability.CONSISTENT_SNAPSHOT)

    def read_page_table(self, handle: SnapshotHandle) -> tuple[PageTableEntry, ...]:
        self._unsupported(RuntimeCapability.CONSISTENT_SNAPSHOT)

    def read_logical_state(self, handle: SnapshotHandle) -> LogicalStateManifest:
        self._unsupported(RuntimeCapability.CONSISTENT_SNAPSHOT)

    def end_consistent_snapshot(self, handle: SnapshotHandle) -> None:
        self._unsupported(RuntimeCapability.CONSISTENT_SNAPSHOT)

    def capture_consistent(self, session_id: str) -> CapturedState:
        self.capabilities.require(RuntimeCapability.CONSISTENT_SNAPSHOT)
        handle = self.begin_consistent_snapshot(session_id)
        try:
            descriptors = self.enumerate_state_segments(handle)
            segments = tuple(
                self.read_state_segment(handle, descriptor.segment_id) for descriptor in descriptors
            )
            captured = CapturedState(
                handle=handle,
                runtime=self.identity,
                layout=self.inspect_session(session_id).layout,
                logical=self.read_logical_state(handle),
                segments=segments,
                page_table=self.read_page_table(handle),
            )
            captured.verify()
            return captured
        finally:
            self.end_consistent_snapshot(handle)

    def start_dirty_tracking(self, session_id: str) -> DirtyTrackingHandle:
        self._unsupported(RuntimeCapability.DIRTY_TRACKING)

    def obtain_dirty_delta(self, handle: DirtyTrackingHandle) -> DirtyDelta:
        self._unsupported(RuntimeCapability.DIRTY_TRACKING)

    def quiesce_at_token_boundary(self, session_id: str) -> SessionMetadata:
        self._unsupported(RuntimeCapability.TOKEN_BOUNDARY_QUIESCE)

    def export_final_delta(self, handle: DirtyTrackingHandle) -> DirtyDelta:
        self._unsupported(RuntimeCapability.FINAL_DELTA)

    def stop_dirty_tracking(self, handle: DirtyTrackingHandle) -> None:
        self._unsupported(RuntimeCapability.DIRTY_TRACKING)

    def fence_source_writer(self, session_id: str, *, expected_owner_epoch: int) -> SessionMetadata:
        self._unsupported(RuntimeCapability.FENCING)

    def release_source_fence(
        self,
        session_id: str,
        *,
        expected_owner_epoch: int,
        coordinator_owner_epoch: int,
        ownership_committed: bool,
    ) -> SessionMetadata:
        self._unsupported(RuntimeCapability.FENCING)

    def prepare_destination_session(
        self,
        captured: CapturedState,
        *,
        destination_session_id: str,
        proposed_owner_epoch: int,
    ) -> SessionMetadata:
        self._unsupported(RuntimeCapability.IMPORT)

    def import_captured_state(self, destination_session_id: str, captured: CapturedState) -> None:
        self._unsupported(RuntimeCapability.IMPORT)

    def apply_dirty_delta(self, destination_session_id: str, delta: DirtyDelta) -> None:
        self._unsupported(RuntimeCapability.IMPORT)

    def validate_imported_state(self, destination_session_id: str) -> ImportValidation:
        self._unsupported(RuntimeCapability.DRY_RUN_VALIDATION)

    def activate_destination(
        self,
        destination_session_id: str,
        *,
        committed_owner_epoch: int,
        fencing_token: str,
    ) -> SessionMetadata:
        self._unsupported(RuntimeCapability.FENCING)

    def abort_destination(self, destination_session_id: str) -> None:
        self._unsupported(RuntimeCapability.IMPORT)

    def inject_failure(self, rule: FailureRule) -> None:
        raise UnsupportedCapabilityError(
            "adapter does not support deterministic failure injection",
            operation="inject_failure",
            session_id=rule.session_id,
        )

    def clear_failures(self) -> None:
        raise UnsupportedCapabilityError(
            "adapter does not support deterministic failure injection",
            operation="clear_failures",
        )

    def crash(self) -> None:
        raise UnsupportedCapabilityError(
            "adapter does not support deterministic process failure simulation",
            operation="crash",
        )

    def restart(self) -> None:
        raise UnsupportedCapabilityError(
            "adapter does not support deterministic process restart simulation",
            operation="restart",
        )
