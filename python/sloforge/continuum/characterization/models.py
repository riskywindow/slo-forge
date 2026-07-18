"""Measured, schema-neutral observations from Continuum state operations.

The canonical BranchFabric trace schema lives above Continuum.  This module keeps
the instrumentation boundary deliberately small: callers can adapt these frozen
records to Parquet, JSON, or Perfetto without coupling the state runtime to a
storage format.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Protocol


class TraceLevel(StrEnum):
    """Continuum instrumentation controls with explicit disabled behavior."""

    DISABLED = "disabled"
    MINIMAL = "minimal"
    FULL = "full"


class MeasurementKind(StrEnum):
    """Required provenance categories; categories are never silently combined."""

    SYNTHETIC = "SYNTHETIC"
    REPLAYED = "REPLAYED"
    HARDWARE_BACKED_REAL = "HARDWARE_BACKED_REAL"
    SIMULATED_HARDWARE = "SIMULATED_HARDWARE"


@dataclass(frozen=True, slots=True)
class StateOperationObservation:
    """One measured or explicitly attributed Continuum state operation.

    ``duration_ns`` and ``cpu_time_ns`` are host observations.  A modeled
    transport duration, when present, is stored separately in
    ``transfer_time_ns`` and has ``SIMULATED_HARDWARE`` provenance.
    """

    sequence: int
    operation: str
    monotonic_start_ns: int
    duration_ns: int
    cpu_time_ns: int
    logical_state_id: str
    branch_id: str
    tenant_security_domain: str
    source_physical_representation: str
    destination_physical_representation: str
    bytes: int
    alignment: int
    page_size: int
    chunk_size: int
    fanout: int
    dependencies: tuple[int, ...]
    concurrency: int
    queue_delay_ns: int
    transfer_time_ns: int
    result: str
    failure: str | None
    state_epoch: int
    workload_evidence_class: MeasurementKind
    measurement_kind: MeasurementKind
    seed: int
    evidence_references: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("observation sequence must be non-negative")
        if not self.operation.startswith("STATE_"):
            raise ValueError("Continuum characterization operations require STATE_ names")
        if (
            min(
                self.monotonic_start_ns,
                self.duration_ns,
                self.cpu_time_ns,
                self.bytes,
                self.alignment,
                self.page_size,
                self.chunk_size,
                self.queue_delay_ns,
                self.transfer_time_ns,
                self.state_epoch,
            )
            < 0
        ):
            raise ValueError("observation counts and timings must be non-negative")
        if self.fanout < 1 or self.concurrency < 1:
            raise ValueError("fanout and concurrency must be positive")
        if self.workload_evidence_class is not MeasurementKind.SYNTHETIC:
            raise ValueError("the reference Continuum harness workload is synthetic")
        if not self.logical_state_id or not self.branch_id or not self.tenant_security_domain:
            raise ValueError("state, branch, and security-domain identities are required")
        if not self.evidence_references:
            raise ValueError("measurements require at least one source evidence reference")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible document for an upper-layer trace adapter."""

        document = asdict(self)
        document["measurement_kind"] = self.measurement_kind.value
        return document


class StateOperationRecorder(Protocol):
    """Generic sink implemented by in-memory, Parquet, or canonical recorders."""

    def record(self, observation: StateOperationObservation) -> None: ...


@dataclass(slots=True)
class ListRecorder:
    """Bounded in-memory recorder for tests and local characterization."""

    maximum_events: int = 100_000
    events: list[StateOperationObservation] = field(default_factory=list)
    dropped_events: int = 0

    def __post_init__(self) -> None:
        if self.maximum_events <= 0:
            raise ValueError("maximum_events must be positive")

    def record(self, observation: StateOperationObservation) -> None:
        if len(self.events) >= self.maximum_events:
            self.dropped_events += 1
            return
        self.events.append(observation)


@dataclass(frozen=True, slots=True)
class SharingAnalysis:
    """Artifact-derived branch sharing and first-write COW measurements."""

    branch_count: int
    shared_logical_bytes: int
    physical_allocated_bytes: int
    logical_unique_bytes: int
    naive_independent_allocation_bytes: int
    private_bytes_per_branch: tuple[int, ...]
    metadata_bytes_per_branch: tuple[int, ...]
    cow_bytes: int
    cow_page_count: int
    source_page_refcount: int
    physical_amplification: float
    sharing_efficiency: float
    evidence_references: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CharacterizationResult:
    seed: int
    workload_evidence_class: MeasurementKind
    trace_level: TraceLevel
    events: tuple[StateOperationObservation, ...]
    dropped_events: int
    sharing: SharingAnalysis
    source_continuation_hash: str
    divergent_continuation_hash: str
    migration_transaction_id: str
    migration_transport_kind: MeasurementKind


@dataclass(frozen=True, slots=True)
class OverheadSample:
    trace_level: TraceLevel
    repetition: int
    seed: int
    duration_ns: int
    cpu_time_ns: int
    event_count: int
    dropped_events: int


@dataclass(frozen=True, slots=True)
class InstrumentationOverhead:
    seed: int
    repetitions: int
    randomized_order: tuple[str, ...]
    raw_samples: tuple[OverheadSample, ...]
    median_duration_ns: Mapping[str, int]
    duration_overhead_fraction: Mapping[str, float]
