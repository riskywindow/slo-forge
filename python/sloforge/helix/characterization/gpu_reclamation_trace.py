"""Trace-v1 projection for Experiment 004 hardware phase markers."""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from itertools import pairwise

from sloforge.helix.characterization.gpu_reclamation import (
    ExperimentPhase,
    ExperimentPhaseMarker,
    phase_trace_binding,
)
from sloforge.helix.characterization.trace.canonical import seal_event
from sloforge.helix.characterization.trace.models import (
    BranchOperationType,
    BranchWorkloadEventV1,
    ClockSource,
    MemoryLocation,
    OperationResult,
    StateOperationEventV1,
    StateOperationType,
    StateSegment,
    TimingMeasurementClass,
    TraceLevel,
    TransportType,
    WorkloadProvenance,
)


@dataclass(frozen=True, slots=True)
class Experiment004TraceIdentity:
    trace_id: str
    session_id: str
    branch_group_id: str
    logical_state_id: str
    tenant_id: str
    security_domain: str
    device: str
    host: str = ""
    process_id: int = 0

    def __post_init__(self) -> None:
        for name in (
            "trace_id",
            "session_id",
            "branch_group_id",
            "logical_state_id",
            "tenant_id",
            "security_domain",
            "device",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"trace identity {name} must be nonempty")
        if self.process_id < 0:
            raise ValueError("trace identity process_id cannot be negative")


def _locations(phase: ExperimentPhase) -> tuple[MemoryLocation, MemoryLocation, TransportType]:
    if phase in {ExperimentPhase.D2H_BEGIN, ExperimentPhase.D2H_END}:
        return MemoryLocation.GPU_HBM, MemoryLocation.PINNED_MEMORY, TransportType.PCIE
    if phase in {ExperimentPhase.H2D_BEGIN, ExperimentPhase.H2D_END}:
        return MemoryLocation.PINNED_MEMORY, MemoryLocation.GPU_HBM, TransportType.PCIE
    if phase in {
        ExperimentPhase.STATE_TRANSFORM_BEGIN,
        ExperimentPhase.STATE_TRANSFORM_END,
        ExperimentPhase.STATE_IMPORT_BEGIN,
        ExperimentPhase.STATE_IMPORT_END,
        ExperimentPhase.GPU_STATE_RELEASE_BEGIN,
        ExperimentPhase.GPU_STATE_RELEASE_END,
        ExperimentPhase.HBM_RECLAIM_CONFIRMED,
    }:
        return MemoryLocation.GPU_HBM, MemoryLocation.GPU_HBM, TransportType.NONE
    if phase in {
        ExperimentPhase.INTEGRITY_BEGIN,
        ExperimentPhase.INTEGRITY_END,
        ExperimentPhase.STATE_PUBLISH,
    }:
        return MemoryLocation.PINNED_MEMORY, MemoryLocation.HOST_DRAM, TransportType.MEMORY_COPY
    return MemoryLocation.UNKNOWN, MemoryLocation.UNKNOWN, TransportType.NONE


def phase_markers_to_trace_events(
    markers: tuple[ExperimentPhaseMarker, ...],
    *,
    identity: Experiment004TraceIdentity,
) -> tuple[BranchWorkloadEventV1 | StateOperationEventV1, ...]:
    """Seal paired BranchWorkloadTrace/StateOperationTrace events.

    The existing trace-v1 operation enums remain wire compatible.  Exact
    Experiment 004 vocabulary is carried in ``attributes.experiment_phase``.
    """

    if not markers:
        raise ValueError("hardware phase trace requires at least one marker")
    if any(
        left.monotonic_timestamp_ns > right.monotonic_timestamp_ns
        for left, right in pairwise(markers)
    ):
        raise ValueError("hardware phase markers are not monotonic")
    origin_ns = markers[0].monotonic_timestamp_ns
    host = identity.host or socket.gethostname()
    process_id = identity.process_id or os.getpid()
    events: list[BranchWorkloadEventV1 | StateOperationEventV1] = []
    for marker in markers:
        branch_operation, state_operation = phase_trace_binding(marker.phase)
        source, destination, transport = _locations(marker.phase)
        attributes: dict[str, bool | int | float | str | None] = {
            "experiment_phase": marker.phase.value,
            **marker.attributes,
        }
        branch = BranchWorkloadEventV1(
            collection_level=TraceLevel.MINIMAL,
            provenance=WorkloadProvenance.HARDWARE_BACKED_REAL,
            timing_measurement_class=TimingMeasurementClass.HARDWARE_BACKED_REAL,
            trace_id=identity.trace_id,
            session_id=identity.session_id,
            branch_group_id=identity.branch_group_id,
            branch_id=marker.branch_id,
            transaction_id=identity.logical_state_id,
            host=host,
            process_id=process_id,
            device=identity.device,
            monotonic_timestamp_ns=marker.monotonic_timestamp_ns,
            normalized_timestamp_ns=marker.monotonic_timestamp_ns - origin_ns,
            duration_ns=0,
            clock_source=ClockSource.MONOTONIC,
            alignment_confidence=1.0,
            operation_type=BranchOperationType(branch_operation),
            logical_state_id=identity.logical_state_id,
            state_segment=StateSegment.KV,
            logical_bytes=marker.logical_bytes,
            physical_bytes=marker.physical_bytes,
            transferred_bytes=(
                marker.physical_bytes
                if marker.phase
                in {
                    ExperimentPhase.D2H_BEGIN,
                    ExperimentPhase.D2H_END,
                    ExperimentPhase.H2D_BEGIN,
                    ExperimentPhase.H2D_END,
                }
                else 0
            ),
            location=destination,
            source_location=source,
            destination_location=destination,
            transport_type=transport,
            attributes=attributes,
        )
        state = StateOperationEventV1(
            collection_level=TraceLevel.MINIMAL,
            provenance=WorkloadProvenance.HARDWARE_BACKED_REAL,
            timing_measurement_class=TimingMeasurementClass.HARDWARE_BACKED_REAL,
            trace_id=identity.trace_id,
            session_id=identity.session_id,
            branch_group_id=identity.branch_group_id,
            logical_state_id=identity.logical_state_id,
            branch_id=marker.branch_id,
            tenant_id=identity.tenant_id,
            security_domain=identity.security_domain,
            host=host,
            process_id=process_id,
            device=identity.device,
            monotonic_timestamp_ns=marker.monotonic_timestamp_ns,
            normalized_timestamp_ns=marker.monotonic_timestamp_ns - origin_ns,
            duration_ns=0,
            clock_source=ClockSource.MONOTONIC,
            alignment_confidence=1.0,
            operation_type=StateOperationType(state_operation),
            state_segment=StateSegment.KV,
            source_physical_representation=source.value,
            destination_physical_representation=destination.value,
            bytes=marker.physical_bytes or marker.logical_bytes,
            alignment_bytes=0,
            page_size_bytes=0,
            chunk_size_bytes=0,
            fanout=8,
            operation_latency_ns=0,
            result=OperationResult.SUCCESS,
            state_epoch=0,
            source_location=source,
            destination_location=destination,
            transport_type=transport,
            attributes=attributes,
        )
        events.append(
            seal_event(branch, event_sequence=len(events), collection_level=TraceLevel.MINIMAL)
        )
        events.append(
            seal_event(state, event_sequence=len(events), collection_level=TraceLevel.MINIMAL)
        )
    return tuple(events)


__all__ = ["Experiment004TraceIdentity", "phase_markers_to_trace_events"]
