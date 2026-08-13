"""Process-local execution-state boundary for real inference runtimes.

This module is intentionally separate from :mod:`sloforge.continuum.adapters.sdk`.
The portable Continuum adapter serializes complete execution state; this boundary
keeps runtime handles private and exposes only bounded, pointer-free observations
needed to validate same-engine GPU KV sharing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from hashlib import sha256
from typing import TYPE_CHECKING, Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

if TYPE_CHECKING:
    from sloforge.helix.characterization.trace import (
        BranchWorkloadEventV1,
        StateOperationEventV1,
    )

MAX_LIVE_BRANCHES = 64
MAX_TOKEN_HISTORY = 1_048_576
MAX_PHYSICAL_BLOCKS = 1_048_576

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=512)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class StrictValueModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class RealRuntimeCapability(StrEnum):
    START_SESSION = "start_session"
    PREFILL_SESSION = "prefill_session"
    PAUSE_AT_SAFE_DECODE_BOUNDARY = "pause_at_safe_decode_boundary"
    INSPECT_LOGICAL_STATE = "inspect_logical_state"
    INSPECT_PHYSICAL_KV_LAYOUT = "inspect_physical_kv_layout"
    INSPECT_GPU_MEMORY_STATE = "inspect_gpu_memory_state"
    IDENTIFY_SHARED_PREFIX_BLOCKS = "identify_shared_prefix_blocks"
    FORK_SAME_POLICY_SESSION = "fork_same_policy_session"
    CREATE_SHARED_ROOT_REFERENCE = "create_shared_root_reference"
    ALLOCATE_PRIVATE_SUFFIX_STATE = "allocate_private_suffix_state"
    RESUME_BRANCH = "resume_branch"
    RUN_CONCURRENT_BRANCHES = "run_concurrent_branches"
    RUN_CONCURRENT_INDEPENDENT_SESSIONS = "run_concurrent_independent_sessions"
    OBSERVE_BLOCK_ALLOCATION = "observe_block_allocation"
    OBSERVE_BLOCK_RELEASE = "observe_block_release"
    REPORT_BLOCK_REFCOUNTS = "report_block_refcounts_or_equivalent"
    CAPTURE_BRANCHPOINT = "capture_branchpoint"
    EMIT_BRANCH_WORKLOAD_TRACE = "emit_BranchWorkloadTrace"
    EMIT_STATE_OPERATION_TRACE = "emit_StateOperationTrace"
    DESTROY_BRANCH = "destroy_branch"
    DESTROY_SESSION = "destroy_session"
    CLEANUP_RUNTIME = "cleanup_runtime"


class LiveSessionPhase(StrEnum):
    CREATED = "created"
    PREFILLED = "prefilled"
    PAUSED = "paused"
    READY = "ready"
    DECODING = "decoding"
    DESTROYED = "destroyed"


class KvStateClass(StrEnum):
    SHARED_PREFIX = "shared_prefix"
    PRIVATE_SUFFIX = "private_suffix"
    ROOT_REFERENCE = "root_reference"
    RUNTIME_RESERVED = "runtime_reserved"


class RefcountEvidence(StrEnum):
    RUNTIME_NATIVE = "runtime_native"
    REQUEST_TABLE_DERIVED = "request_table_derived"


class RuntimeScopeContract(StrictValueModel):
    same_runtime_only: bool = True
    same_policy_only: bool = True
    same_process_or_adapter_scope: NonEmpty
    cross_runtime_portable: bool = False

    @model_validator(mode="after")
    def conservative_scope(self) -> RuntimeScopeContract:
        if not self.same_runtime_only or not self.same_policy_only:
            raise ValueError("live shared-root state must be same-runtime and same-policy")
        if self.cross_runtime_portable:
            raise ValueError("live runtime handles cannot claim cross-runtime portability")
        return self


class RuntimeModelIdentity(StrictValueModel):
    runtime: NonEmpty
    runtime_version: NonEmpty
    adapter_version: NonEmpty
    model_id: NonEmpty
    model_revision: NonEmpty
    tokenizer_id: NonEmpty
    tokenizer_revision: NonEmpty
    dtype: NonEmpty
    device: NonEmpty
    policy_epoch: NonEmpty


class RuntimeCapabilityMatrix(StrictValueModel):
    identity: RuntimeModelIdentity
    capabilities: frozenset[RealRuntimeCapability]
    scope: RuntimeScopeContract
    version_pin: NonEmpty
    internal_interfaces: tuple[NonEmpty, ...]
    max_sessions: int = Field(ge=1, le=MAX_LIVE_BRANCHES + 1)
    max_fanout: int = Field(ge=1, le=MAX_LIVE_BRANCHES)

    @model_validator(mode="after")
    def complete_cleanup_surface(self) -> RuntimeCapabilityMatrix:
        required = {
            RealRuntimeCapability.DESTROY_BRANCH,
            RealRuntimeCapability.DESTROY_SESSION,
            RealRuntimeCapability.CLEANUP_RUNTIME,
        }
        if not required.issubset(self.capabilities):
            raise ValueError("real-runtime adapters must expose complete cleanup operations")
        return self


class RuntimeSessionRef(StrictValueModel):
    session_id: NonEmpty
    adapter_session_id: NonEmpty
    phase: LiveSessionPhase
    seed: int = Field(ge=0, lt=1 << 64)
    branch_id: str | None = None
    parent_session_id: str | None = None
    root_reference_id: str | None = None


class LogicalRuntimeState(StrictValueModel):
    session_id: NonEmpty
    model: RuntimeModelIdentity
    token_ids: tuple[int, ...]
    token_history_sha256: Sha256
    position_start: int = Field(ge=0)
    position_end_exclusive: int = Field(ge=0)
    attention_layer_ids: tuple[NonEmpty, ...]
    logical_kv_token_ranges: tuple[tuple[int, int], ...]

    @model_validator(mode="after")
    def valid_logical_state(self) -> LogicalRuntimeState:
        if len(self.token_ids) > MAX_TOKEN_HISTORY:
            raise ValueError("token history exceeds the bounded live-state limit")
        if any(token < 0 for token in self.token_ids):
            raise ValueError("token IDs must be non-negative")
        encoded = b"".join(token.to_bytes(8, "little") for token in self.token_ids)
        if sha256(encoded).hexdigest() != self.token_history_sha256:
            raise ValueError("token history hash mismatch")
        if self.position_end_exclusive - self.position_start != len(self.token_ids):
            raise ValueError("logical positions do not match token history")
        if not self.attention_layer_ids:
            raise ValueError("logical state requires at least one attention layer")
        if any(start < 0 or end <= start for start, end in self.logical_kv_token_ranges):
            raise ValueError("logical KV token ranges must be positive half-open intervals")
        return self


class PhysicalKvBlock(StrictValueModel):
    runtime_block_id: NonEmpty
    block_index: int = Field(ge=0)
    kv_cache_group: int = Field(ge=0)
    layer_ids: tuple[NonEmpty, ...]
    device: NonEmpty
    dtype: NonEmpty
    bytes: int = Field(gt=0)
    block_size_tokens: int = Field(gt=0)
    logical_token_start: int = Field(ge=0)
    logical_token_end_exclusive: int = Field(gt=0)
    branch_ids: tuple[NonEmpty, ...]
    refcount: int = Field(ge=0)
    refcount_evidence: RefcountEvidence
    state_class: KvStateClass
    allocation_epoch: int = Field(ge=0)
    cache_resident: bool

    @model_validator(mode="after")
    def valid_block(self) -> PhysicalKvBlock:
        if self.logical_token_end_exclusive <= self.logical_token_start:
            raise ValueError("physical block requires a positive logical token range")
        if self.logical_token_end_exclusive - self.logical_token_start > self.block_size_tokens:
            raise ValueError("logical token range exceeds the physical block capacity")
        if len(self.branch_ids) != len(set(self.branch_ids)):
            raise ValueError("physical block branch references must be unique")
        if not self.layer_ids:
            raise ValueError("physical KV block requires a non-empty layer scope")
        return self


class PhysicalKvLayoutSnapshot(StrictValueModel):
    runtime: NonEmpty
    runtime_version: NonEmpty
    snapshot_epoch: int = Field(ge=0)
    observed_at_monotonic_ns: int = Field(ge=0)
    session_ids: tuple[NonEmpty, ...]
    blocks: tuple[PhysicalKvBlock, ...]
    shared_prefix_block_ids: tuple[NonEmpty, ...]
    private_suffix_block_ids: tuple[NonEmpty, ...]
    root_reference_count: int = Field(ge=0)
    physical_assigned_bytes: int = Field(ge=0)
    shared_prefix_bytes: int = Field(ge=0)
    private_suffix_bytes: int = Field(ge=0)
    kv_pool_reserved_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def valid_accounting(self) -> PhysicalKvLayoutSnapshot:
        if len(self.blocks) > MAX_PHYSICAL_BLOCKS:
            raise ValueError("physical block snapshot exceeds its bound")
        block_ids = [block.runtime_block_id for block in self.blocks]
        if len(block_ids) != len(set(block_ids)):
            raise ValueError("physical block identifiers must be unique within a snapshot")
        known = set(block_ids)
        shared = set(self.shared_prefix_block_ids)
        private = set(self.private_suffix_block_ids)
        if not shared.issubset(known) or not private.issubset(known) or shared & private:
            raise ValueError("shared/private block classifications are inconsistent")
        assigned = sum(block.bytes for block in self.blocks)
        if assigned != self.physical_assigned_bytes:
            raise ValueError("physical assigned-byte accounting mismatch")
        if sum(block.bytes for block in self.blocks if block.runtime_block_id in shared) != (
            self.shared_prefix_bytes
        ):
            raise ValueError("shared-prefix byte accounting mismatch")
        if sum(block.bytes for block in self.blocks if block.runtime_block_id in private) != (
            self.private_suffix_bytes
        ):
            raise ValueError("private-suffix byte accounting mismatch")
        if self.physical_assigned_bytes > self.kv_pool_reserved_bytes:
            raise ValueError("assigned KV bytes exceed the reserved runtime pool")
        return self


class PhysicalKvBlockReleaseEvidence(StrictValueModel):
    """Pointer-free native allocator state for one formerly observed KV block."""

    runtime_block_id: NonEmpty
    block_index: int = Field(ge=0)
    allocation_epoch: int = Field(ge=0)
    native_refcount: int = Field(ge=0)
    block_hash_present: bool
    allocator_available: bool
    is_null: bool

    @model_validator(mode="after")
    def valid_native_release_state(self) -> PhysicalKvBlockReleaseEvidence:
        if self.is_null:
            raise ValueError("release evidence must never target the reserved null block")
        if self.allocator_available and self.native_refcount != 0:
            raise ValueError("a referenced KV block cannot be allocator-available")
        return self


class PhysicalKvReleaseEvidence(StrictValueModel):
    """Exact-ID release evidence independent of live request/session selection."""

    runtime: NonEmpty
    runtime_version: NonEmpty
    device: NonEmpty
    observed_at_monotonic_ns: int = Field(ge=0)
    requested_block_ids: tuple[NonEmpty, ...]
    blocks: tuple[PhysicalKvBlockReleaseEvidence, ...]
    pool_free_block_count: int = Field(ge=0)
    pool_usable_block_count: int = Field(ge=0)

    @model_validator(mode="after")
    def valid_release_evidence(self) -> PhysicalKvReleaseEvidence:
        if not self.requested_block_ids:
            raise ValueError("release evidence requires at least one explicit block ID")
        if len(self.requested_block_ids) != len(set(self.requested_block_ids)):
            raise ValueError("requested release block IDs must be unique")
        observed_ids = tuple(block.runtime_block_id for block in self.blocks)
        if set(observed_ids) != set(self.requested_block_ids) or len(observed_ids) != len(
            self.requested_block_ids
        ):
            raise ValueError("release evidence must cover every requested block exactly once")
        if self.pool_free_block_count > self.pool_usable_block_count:
            raise ValueError("free KV block count exceeds usable pool capacity")
        return self


class GpuMemoryState(StrictValueModel):
    device: NonEmpty
    observed_at_monotonic_ns: int = Field(ge=0)
    nvml_process_bytes: int | None = Field(default=None, ge=0)
    nvml_device_used_bytes: int | None = Field(default=None, ge=0)
    torch_allocated_bytes: int | None = Field(default=None, ge=0)
    torch_reserved_bytes: int | None = Field(default=None, ge=0)
    kv_pool_reserved_bytes: int = Field(ge=0)
    kv_assigned_bytes: int = Field(ge=0)
    kv_unassigned_bytes: int = Field(ge=0)
    sample_source: NonEmpty

    @model_validator(mode="after")
    def valid_pool_accounting(self) -> GpuMemoryState:
        if self.kv_assigned_bytes + self.kv_unassigned_bytes != self.kv_pool_reserved_bytes:
            raise ValueError("KV pool assigned/unassigned accounting mismatch")
        return self


class SharedRootReference(StrictValueModel):
    root_reference_id: NonEmpty
    source_session_id: NonEmpty
    prefix_token_count: int = Field(gt=0)
    block_ids: tuple[NonEmpty, ...]
    physical_bytes: int = Field(gt=0)
    adapter_owned_reference_count: int = Field(ge=1)
    created_at_monotonic_ns: int = Field(ge=0)

    @model_validator(mode="after")
    def unique_blocks(self) -> SharedRootReference:
        if not self.block_ids or len(self.block_ids) != len(set(self.block_ids)):
            raise ValueError("shared root requires unique physical block identifiers")
        return self


class LiveBranchPoint(StrictValueModel):
    branchpoint_id: NonEmpty
    root_reference: SharedRootReference
    scope: RuntimeScopeContract
    model: RuntimeModelIdentity
    parent_session_id: NonEmpty
    branch_session_ids: tuple[NonEmpty, ...]
    prefix_token_count: int = Field(gt=0)
    policy_epoch: NonEmpty
    captured_at_monotonic_ns: int = Field(ge=0)

    @model_validator(mode="after")
    def valid_branches(self) -> LiveBranchPoint:
        if not 1 <= len(self.branch_session_ids) <= MAX_LIVE_BRANCHES:
            raise ValueError("branchpoint fanout is outside the bounded adapter scope")
        if len(self.branch_session_ids) != len(set(self.branch_session_ids)):
            raise ValueError("branchpoint session identifiers must be unique")
        if self.prefix_token_count != self.root_reference.prefix_token_count:
            raise ValueError("branchpoint prefix does not match its root reference")
        return self


class ConcurrentDecodeResult(StrictValueModel):
    branchpoint_id: NonEmpty
    branch_output_token_ids: dict[NonEmpty, tuple[int, ...]]
    # First successful runtime-native slot allocation. With chunked prefill,
    # this is admission only and is deliberately not full branch readiness.
    request_admitted_latency_ns: dict[NonEmpty, int]
    # Host-side post-admission lower bound for the engine step proven to have
    # yielded this branch's first decode token.
    first_decode_token_started_latency_ns: dict[NonEmpty, int]
    # Observation when that engine step returned, proving prompt completion.
    branch_ready_latency_ns: dict[NonEmpty, int]
    # Later adapter observation of the emitted output token.
    first_token_latency_ns: dict[NonEmpty, int]
    # ``elapsed_ns`` starts immediately before runtime submission and therefore
    # includes branch readiness/prefill. The active window starts only when all
    # branches have emitted their first token, preventing independent-prefill
    # latency from being mislabeled as decode-throughput interference.
    elapsed_ns: int = Field(gt=0)
    total_output_tokens: int = Field(ge=0)
    throughput_tokens_per_second: float = Field(ge=0.0)
    decode_active_elapsed_ns: int = Field(gt=0)
    decode_active_boundary_output_counts: dict[NonEmpty, int]
    decode_active_tokens: int = Field(ge=0)
    completed_branch_ids: tuple[NonEmpty, ...]
    allocation_snapshots: tuple[PhysicalKvLayoutSnapshot, ...]

    @model_validator(mode="after")
    def valid_result(self) -> ConcurrentDecodeResult:
        branch_ids = set(self.branch_output_token_ids)
        milestone_mappings = (
            self.request_admitted_latency_ns,
            self.first_decode_token_started_latency_ns,
            self.branch_ready_latency_ns,
            self.first_token_latency_ns,
        )
        if any(set(mapping) != branch_ids for mapping in milestone_mappings):
            raise ValueError("every timing milestone must cover every output branch")
        if any(value < 0 for mapping in milestone_mappings for value in mapping.values()):
            raise ValueError("timing milestone latencies must be non-negative")
        for branch_id in branch_ids:
            if (
                self.request_admitted_latency_ns[branch_id]
                > self.first_decode_token_started_latency_ns[branch_id]
                or self.first_decode_token_started_latency_ns[branch_id]
                > self.branch_ready_latency_ns[branch_id]
                or self.branch_ready_latency_ns[branch_id] > self.first_token_latency_ns[branch_id]
            ):
                raise ValueError("branch timing milestones are not causally ordered")
        if self.total_output_tokens != sum(
            len(tokens) for tokens in self.branch_output_token_ids.values()
        ):
            raise ValueError("decode token accounting mismatch")
        if set(self.decode_active_boundary_output_counts) != set(self.branch_output_token_ids):
            raise ValueError("active decode boundary counts must cover every output branch")
        expected_active = 0
        for branch_id, tokens in self.branch_output_token_ids.items():
            boundary_count = self.decode_active_boundary_output_counts[branch_id]
            if not 1 <= boundary_count <= len(tokens):
                raise ValueError(
                    "active decode boundary count must be within the observed branch output"
                )
            expected_active += len(tokens) - boundary_count
        if self.decode_active_tokens != expected_active:
            raise ValueError("active decode token accounting mismatch")
        if self.decode_active_elapsed_ns > self.elapsed_ns:
            raise ValueError("active decode window cannot exceed request elapsed time")
        return self


class RealRuntimeStateAdapter(ABC):
    """Minimal live-state contract; implementations retain all opaque handles."""

    @property
    @abstractmethod
    def capability_matrix(self) -> RuntimeCapabilityMatrix: ...

    @abstractmethod
    def start_session(
        self, session_id: str, *, token_ids: tuple[int, ...], seed: int, timeout_s: float
    ) -> RuntimeSessionRef: ...

    @abstractmethod
    def prefill_session(self, session_id: str, *, timeout_s: float) -> RuntimeSessionRef: ...

    @abstractmethod
    def pause_at_safe_decode_boundary(
        self, session_id: str, *, timeout_s: float
    ) -> RuntimeSessionRef: ...

    @abstractmethod
    def inspect_logical_state(
        self, session_id: str, *, timeout_s: float
    ) -> LogicalRuntimeState: ...

    @abstractmethod
    def inspect_physical_kv_layout(
        self, session_ids: tuple[str, ...], *, timeout_s: float
    ) -> PhysicalKvLayoutSnapshot: ...

    @abstractmethod
    def inspect_gpu_memory_state(self, *, timeout_s: float) -> GpuMemoryState: ...

    @abstractmethod
    def identify_shared_prefix_blocks(
        self, branch_session_ids: tuple[str, ...], *, timeout_s: float
    ) -> tuple[str, ...]: ...

    @abstractmethod
    def create_shared_root_reference(
        self, session_id: str, *, prefix_token_count: int, timeout_s: float
    ) -> SharedRootReference: ...

    @abstractmethod
    def fork_same_policy_session(
        self,
        root_reference_id: str,
        branch_session_id: str,
        *,
        divergent_token_id: int,
        seed: int,
        timeout_s: float,
    ) -> RuntimeSessionRef: ...

    @abstractmethod
    def allocate_private_suffix_state(
        self, branch_session_id: str, *, minimum_tokens: int, timeout_s: float
    ) -> PhysicalKvLayoutSnapshot: ...

    @abstractmethod
    def resume_branch(self, branch_session_id: str, *, timeout_s: float) -> RuntimeSessionRef: ...

    @abstractmethod
    def run_concurrent_branches(
        self,
        branch_session_ids: tuple[str, ...],
        *,
        maximum_new_tokens: int,
        seed: int,
        timeout_s: float,
    ) -> ConcurrentDecodeResult: ...

    @abstractmethod
    def observe_block_allocation(
        self, session_ids: tuple[str, ...], *, timeout_s: float
    ) -> PhysicalKvLayoutSnapshot: ...

    @abstractmethod
    def observe_block_release(
        self, session_ids: tuple[str, ...], *, timeout_s: float
    ) -> PhysicalKvLayoutSnapshot: ...

    @abstractmethod
    def inspect_block_release_evidence(
        self, runtime_block_ids: tuple[str, ...], *, timeout_s: float
    ) -> PhysicalKvReleaseEvidence: ...

    @abstractmethod
    def report_block_refcounts_or_equivalent(
        self, session_ids: tuple[str, ...], *, timeout_s: float
    ) -> dict[str, int]: ...

    @abstractmethod
    def capture_branchpoint(
        self, root_reference_id: str, branch_session_ids: tuple[str, ...], *, timeout_s: float
    ) -> LiveBranchPoint: ...

    @abstractmethod
    def emit_branch_workload_trace(self) -> tuple[BranchWorkloadEventV1, ...]: ...

    @abstractmethod
    def emit_state_operation_trace(self) -> tuple[StateOperationEventV1, ...]: ...

    @abstractmethod
    def destroy_branch(self, branch_session_id: str, *, timeout_s: float) -> None: ...

    @abstractmethod
    def destroy_session(self, session_id: str, *, timeout_s: float) -> None: ...

    @abstractmethod
    def cleanup_runtime(self, *, timeout_s: float) -> None: ...


def token_history_sha256(token_ids: tuple[int, ...]) -> str:
    if len(token_ids) > MAX_TOKEN_HISTORY or any(token < 0 for token in token_ids):
        raise ValueError("invalid bounded token history")
    return sha256(b"".join(token.to_bytes(8, "little") for token in token_ids)).hexdigest()


__all__ = [
    "ConcurrentDecodeResult",
    "GpuMemoryState",
    "KvStateClass",
    "LiveBranchPoint",
    "LiveSessionPhase",
    "LogicalRuntimeState",
    "PhysicalKvBlock",
    "PhysicalKvBlockReleaseEvidence",
    "PhysicalKvLayoutSnapshot",
    "PhysicalKvReleaseEvidence",
    "RealRuntimeCapability",
    "RealRuntimeStateAdapter",
    "RefcountEvidence",
    "RuntimeCapabilityMatrix",
    "RuntimeModelIdentity",
    "RuntimeScopeContract",
    "RuntimeSessionRef",
    "SharedRootReference",
    "token_history_sha256",
]
