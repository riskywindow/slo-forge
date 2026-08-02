"""Typed persistent-state synthesis and transition contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Ownership(StrEnum):
    EXCLUSIVE = "exclusive"
    REPLICATED_READ_ONLY = "replicated_read_only"
    REPLICATED_CONSENSUS = "replicated_consensus"


class Visibility(StrEnum):
    REQUEST = "request"
    TENANT = "tenant"
    WORKFLOW = "workflow"
    GLOBAL = "global"


class StateLayout(StrEnum):
    CONTIGUOUS = "contiguous"
    PAGED = "paged"
    INTERLEAVED = "interleaved"
    COMPRESSED = "compressed"


class StorageTier(StrEnum):
    DEVICE = "device"
    PEER_DEVICE = "peer_device"
    PINNED_HOST = "pinned_host"
    HOST = "host"
    REMOTE = "remote"


class Consistency(StrEnum):
    LINEARIZABLE = "linearizable"
    SNAPSHOT = "snapshot"
    EVENTUAL_READ_ONLY = "eventual_read_only"


class TransformationKind(StrEnum):
    LAYOUT = "layout"
    PRECISION = "precision"
    OFFLOAD = "offload"
    REPLICATION = "replication"
    MIGRATION = "migration"
    PREFETCH = "prefetch"
    EVICTION = "eviction"
    CHECKPOINT = "checkpoint"
    RECOMPUTE = "recompute"


class StatePrecondition(StrEnum):
    REQUEST_BOUNDARY = "request_boundary"
    QUIESCENT_STATE = "quiescent_state"
    EXCLUSIVE_OWNERSHIP = "exclusive_ownership"
    STATE_CONVERSION_VERIFIED = "state_conversion_verified"
    QUALITY_CONTRACT = "quality_contract"
    CHECKPOINT_AVAILABLE = "checkpoint_available"
    OWNER_ALLOWLIST = "owner_allowlist"


class RollbackStrategy(StrEnum):
    RETAIN_SOURCE_UNTIL_COMMIT = "retain_source_until_commit"
    RESTORE_CHECKPOINT = "restore_checkpoint"
    REVERSE_CONVERSION = "reverse_conversion"


class StateAction(StrEnum):
    ALLOCATE = "allocate"
    ACQUIRE = "acquire"
    READ = "read"
    WRITE = "write"
    BEGIN_MIGRATION = "begin_migration"
    COMMIT_MIGRATION = "commit_migration"
    ABORT_MIGRATION = "abort_migration"
    RELEASE = "release"
    CANCEL_REQUEST = "cancel_request"
    CHECKPOINT = "checkpoint"
    ROLLBACK = "rollback"


@dataclass(frozen=True, slots=True)
class StateRegion:
    region_id: str
    semantic_contract: str
    ownership: Ownership
    owners: tuple[str, ...]
    visibility: Visibility
    layout: StateLayout
    storage: StorageTier
    dtype: str
    bytes_per_item: int
    maximum_items: int
    replication_factor: int
    consistency: Consistency
    mutable: bool
    checkpointed: bool
    compatible_genome_hashes: tuple[str, ...]
    migration_target_owners: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class StateTransformation:
    transformation_id: str
    kind: TransformationKind
    source: StateRegion
    target: StateRegion
    exact: bool
    expected_quality_cost: float
    expected_memory_delta_bytes: int
    migration_chunk_bytes: int
    preconditions: tuple[StatePrecondition, ...]
    proof_obligations: tuple[str, ...]
    rollback_strategy: RollbackStrategy
    conversion_evidence: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CompiledStateTransition:
    transformation_id: str
    coexistence_peak_bytes: int
    migration_chunks: int
    active_stream_compatible: bool
    requires_request_boundary: bool
    checked_preconditions: tuple[str, ...]
    proof_obligations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StateEvent:
    sequence: int
    action: StateAction
    region_id: str
    request_id: str
    actor: str
    epoch: int
    item_count: int = 0
    target_actor: str | None = None
    target_genome_hash: str | None = None


@dataclass(frozen=True, slots=True)
class StateViolation:
    sequence: int
    invariant: str
    detail: str


@dataclass(frozen=True, slots=True)
class StateTraceResult:
    valid: bool
    peak_memory_bytes: int
    final_memory_bytes: int
    violations: tuple[StateViolation, ...]


class StateTransformError(ValueError):
    """Raised when a state plan is unsafe or outside its declared domain."""
