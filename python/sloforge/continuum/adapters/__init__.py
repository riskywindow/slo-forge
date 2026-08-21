"""Continuum runtime adapter SDK and deterministic reference adapters."""

from typing import TYPE_CHECKING

from sloforge.continuum.adapters.abi import (
    CapsuleInputs,
    CapturedChunk,
    captured_to_capsule_inputs,
    captured_to_logical_state,
    captured_to_physical_layout,
)
from sloforge.continuum.adapters.real_runtime import (
    ConcurrentDecodeResult,
    GpuMemoryState,
    KvStateClass,
    LiveBranchPoint,
    LiveSessionPhase,
    LogicalRuntimeState,
    PhysicalKvBlock,
    PhysicalKvBlockReleaseEvidence,
    PhysicalKvLayoutSnapshot,
    PhysicalKvReleaseEvidence,
    RealRuntimeCapability,
    RealRuntimeStateAdapter,
    RefcountEvidence,
    RuntimeCapabilityMatrix,
    RuntimeModelIdentity,
    RuntimeScopeContract,
    RuntimeSessionRef,
    SharedRootReference,
    token_history_sha256,
)
from sloforge.continuum.adapters.reference import (
    ReferenceHeadMajorAdapter,
    ReferenceTokenMajorAdapter,
)
from sloforge.continuum.adapters.sdk import (
    AdapterError,
    AdapterUnavailableError,
    CapabilityMatrix,
    CapturedState,
    ClientTerminalStatus,
    ContinuumRuntimeAdapter,
    DirtyDelta,
    DirtyLogOverflowError,
    DirtyTrackingHandle,
    DirtyTrackingStrategy,
    FailurePoint,
    FailureRule,
    ImportValidation,
    ImportValidationError,
    InjectedFailureError,
    LayoutKind,
    LogicalStateManifest,
    ModelContract,
    PageTableEntry,
    ResourceLimitError,
    RuntimeCapability,
    RuntimeIdentity,
    RuntimeLayout,
    SegmentDescriptor,
    SegmentIntegrityError,
    SessionLifecycle,
    SessionMetadata,
    SessionNotFoundError,
    SessionStateError,
    SnapshotConsistencyError,
    SnapshotHandle,
    StaleDeltaError,
    StaleOwnerEpochError,
    StateKind,
    StateSegment,
    TokenEvent,
    UnsupportedCapabilityError,
)

if TYPE_CHECKING:
    from sloforge.continuum.adapters.real_runtime_fixture import (
        FIXTURE_ADAPTER_VERSION,
        FIXTURE_RUNTIME,
        FIXTURE_RUNTIME_VERSION,
        DeterministicRuntimeFixtureAdapter,
        RuntimeFixtureError,
    )

_FIXTURE_EXPORTS = frozenset(
    {
        "DeterministicRuntimeFixtureAdapter",
        "FIXTURE_ADAPTER_VERSION",
        "FIXTURE_RUNTIME",
        "FIXTURE_RUNTIME_VERSION",
        "RuntimeFixtureError",
    }
)


def __getattr__(name: str) -> object:
    """Load the trace-emitting fixture lazily to keep package imports acyclic."""

    if name not in _FIXTURE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from sloforge.continuum.adapters import real_runtime_fixture

    return getattr(real_runtime_fixture, name)


__all__ = [
    "FIXTURE_ADAPTER_VERSION",
    "FIXTURE_RUNTIME",
    "FIXTURE_RUNTIME_VERSION",
    "AdapterError",
    "AdapterUnavailableError",
    "CapabilityMatrix",
    "CapsuleInputs",
    "CapturedChunk",
    "CapturedState",
    "ClientTerminalStatus",
    "ConcurrentDecodeResult",
    "ContinuumRuntimeAdapter",
    "DeterministicRuntimeFixtureAdapter",
    "DirtyDelta",
    "DirtyLogOverflowError",
    "DirtyTrackingHandle",
    "DirtyTrackingStrategy",
    "FailurePoint",
    "FailureRule",
    "GpuMemoryState",
    "ImportValidation",
    "ImportValidationError",
    "InjectedFailureError",
    "KvStateClass",
    "LayoutKind",
    "LiveBranchPoint",
    "LiveSessionPhase",
    "LogicalRuntimeState",
    "LogicalStateManifest",
    "ModelContract",
    "PageTableEntry",
    "PhysicalKvBlock",
    "PhysicalKvBlockReleaseEvidence",
    "PhysicalKvLayoutSnapshot",
    "PhysicalKvReleaseEvidence",
    "RealRuntimeCapability",
    "RealRuntimeStateAdapter",
    "RefcountEvidence",
    "ReferenceHeadMajorAdapter",
    "ReferenceTokenMajorAdapter",
    "ResourceLimitError",
    "RuntimeCapability",
    "RuntimeCapabilityMatrix",
    "RuntimeFixtureError",
    "RuntimeIdentity",
    "RuntimeLayout",
    "RuntimeModelIdentity",
    "RuntimeScopeContract",
    "RuntimeSessionRef",
    "SegmentDescriptor",
    "SegmentIntegrityError",
    "SessionLifecycle",
    "SessionMetadata",
    "SessionNotFoundError",
    "SessionStateError",
    "SharedRootReference",
    "SnapshotConsistencyError",
    "SnapshotHandle",
    "StaleDeltaError",
    "StaleOwnerEpochError",
    "StateKind",
    "StateSegment",
    "TokenEvent",
    "UnsupportedCapabilityError",
    "captured_to_capsule_inputs",
    "captured_to_logical_state",
    "captured_to_physical_layout",
    "token_history_sha256",
]
