"""Adapters from measured Helix/Continuum observations to canonical trace v1."""

from __future__ import annotations

import json
import os
import socket
from collections.abc import Mapping

from sloforge.continuum.characterization import StateOperationObservation
from sloforge.helix.characterization.lifecycle.recorder import JsonValue, TraceStream

from .buffer import BoundedTraceBuffer
from .models import (
    BranchOperationType,
    BranchWorkloadEventV1,
    ClockSource,
    MemoryLocation,
    OperationResult,
    StateOperationEventV1,
    StateOperationType,
    StateSegment,
    TimingMeasurementClass,
    TransportType,
    WorkloadProvenance,
)

_BRANCH_OPERATION_MAP = {
    "BRANCH_POINT_CAPTURE": BranchOperationType.BRANCH_POINT,
    "ENVIRONMENT_FORK": BranchOperationType.ENVIRONMENT_FORK,
    "BRANCH_FORK": BranchOperationType.BRANCH_FORK,
    "ROLLOUT_COMPLETE": BranchOperationType.ROLLOUT,
    "REWARD_COMPLETE": BranchOperationType.REWARD,
    "TRAINING_COMPLETE": BranchOperationType.TRAIN,
    "PROMOTION_COMPLETE": BranchOperationType.PROMOTE,
    "BRANCH_PRUNE": BranchOperationType.BRANCH_PRUNE,
    "BRANCH_COMPLETE": BranchOperationType.BRANCH_COMPLETE,
    "LEARNING_TRANSACTION_STAGE": BranchOperationType.LEARNING_TRANSACTION_STAGE,
}
_CANONICAL_KEYS = {
    "trace_id",
    "session_id",
    "branch_group_id",
    "branch_id",
    "parent_branch_id",
    "policy_epoch",
    "environment_id",
    "transaction_id",
    "host",
    "process",
    "rank",
    "device",
    "monotonic_timestamp_ns",
    "normalized_timestamp_ns",
    "duration_ns",
    "cpu_time_ns",
    "clock_source",
    "alignment_confidence",
    "operation_type",
    "logical_state_id",
    "physical_state_id",
    "state_segment",
    "logical_bytes",
    "physical_bytes",
    "compressed_bytes",
    "transferred_bytes",
    "metadata_bytes",
    "queue_delay_ns",
    "transfer_latency_ns",
    "transform_latency_ns",
    "wait_latency_ns",
    "gpu_duration_ns",
    "fanout",
    "result",
    "content_hash",
    "event_sequence",
    "schema_version",
    "trace_producer_version",
    "measurement_source",
    "workload_evidence_class",
    "timing_measurement_class",
    "simulated_gpu_state",
    "seed",
    "tenant",
    "bytes",
    "alignment",
    "page_size",
    "chunk_size",
    "source_physical_representation",
    "destination_physical_representation",
    "state_epoch",
}


def _integer(event: Mapping[str, JsonValue], key: str, default: int = 0) -> int:
    value = event.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"lifecycle field {key!r} must be an integer")
    return value


def _number(event: Mapping[str, JsonValue], key: str, default: float = 0.0) -> float:
    value = event.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"lifecycle field {key!r} must be numeric")
    return float(value)


def _optional_string(event: Mapping[str, JsonValue], key: str) -> str | None:
    value = event.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"lifecycle field {key!r} must be a string or null")
    return value


def _required_string(event: Mapping[str, JsonValue], key: str) -> str:
    value = _optional_string(event, key)
    if not value:
        raise ValueError(f"lifecycle field {key!r} must be non-empty")
    return value


def _state_segment(value: JsonValue) -> StateSegment:
    aliases = {"model": StateSegment.MODEL, "filesystem": StateSegment.FILESYSTEM}
    if isinstance(value, str):
        try:
            return aliases.get(value, StateSegment(value))
        except ValueError:
            pass
    return StateSegment.UNKNOWN


def _attributes(event: Mapping[str, JsonValue]) -> dict[str, bool | int | float | str | None]:
    attributes: dict[str, bool | int | float | str | None] = {}
    for key, value in event.items():
        if key in _CANONICAL_KEYS:
            continue
        if value is None or isinstance(value, (bool, int, float, str)):
            attributes[key] = value
    raw_hash = event.get("content_hash")
    if isinstance(raw_hash, str):
        attributes["source_event_content_hash"] = raw_hash
    raw_operation = event.get("operation_type")
    if isinstance(raw_operation, str):
        attributes["source_operation_type"] = raw_operation
    attributes["simulated_gpu_state"] = bool(event.get("simulated_gpu_state", False))
    return attributes


class CanonicalLifecycleRecorder:
    """Validate and enqueue lifecycle observations directly on the hot path."""

    def __init__(self, buffer: BoundedTraceBuffer) -> None:
        self.buffer = buffer

    def record(self, stream: TraceStream, event: Mapping[str, JsonValue]) -> None:
        if stream is TraceStream.BRANCH_WORKLOAD:
            self.buffer.record(self._branch_event(event))
        else:
            self.buffer.record(self._state_event(event))

    def _branch_event(self, event: Mapping[str, JsonValue]) -> BranchWorkloadEventV1:
        raw_operation = _required_string(event, "operation_type")
        operation = _BRANCH_OPERATION_MAP.get(raw_operation)
        if operation is None:
            try:
                operation = BranchOperationType(raw_operation)
            except ValueError as error:
                raise ValueError(
                    f"unsupported branch lifecycle operation {raw_operation!r}"
                ) from error
        return BranchWorkloadEventV1(
            provenance=WorkloadProvenance(_required_string(event, "workload_evidence_class")),
            timing_measurement_class=TimingMeasurementClass(
                _required_string(event, "timing_measurement_class")
            ),
            trace_id=_required_string(event, "trace_id"),
            session_id=_required_string(event, "session_id"),
            branch_group_id=_optional_string(event, "branch_group_id"),
            branch_id=_optional_string(event, "branch_id"),
            parent_branch_id=_optional_string(event, "parent_branch_id"),
            policy_epoch=_optional_string(event, "policy_epoch"),
            environment_id=_optional_string(event, "environment_id"),
            transaction_id=_optional_string(event, "transaction_id"),
            host=_required_string(event, "host"),
            process_id=_integer(event, "process"),
            rank=_integer(event, "rank"),
            device=_optional_string(event, "device"),
            monotonic_timestamp_ns=_integer(event, "monotonic_timestamp_ns"),
            normalized_timestamp_ns=_integer(event, "normalized_timestamp_ns"),
            duration_ns=_integer(event, "duration_ns"),
            clock_source=ClockSource.PERF_COUNTER,
            alignment_confidence=_number(event, "alignment_confidence", 1.0),
            operation_type=operation,
            logical_state_id=_optional_string(event, "logical_state_id"),
            physical_state_id=_optional_string(event, "physical_state_id"),
            state_segment=_state_segment(event.get("state_segment")),
            logical_bytes=_integer(event, "logical_bytes"),
            physical_bytes=_integer(event, "physical_bytes"),
            compressed_bytes=_integer(event, "compressed_bytes"),
            transferred_bytes=_integer(event, "transferred_bytes"),
            metadata_bytes=_integer(event, "metadata_bytes"),
            execution_latency_ns=_integer(event, "duration_ns"),
            queue_delay_ns=_integer(event, "queue_delay_ns"),
            transfer_latency_ns=_integer(event, "transfer_latency_ns"),
            transform_latency_ns=_integer(event, "transform_latency_ns"),
            wait_latency_ns=_integer(event, "wait_latency_ns"),
            cpu_time_ns=_integer(event, "cpu_time_ns"),
            gpu_duration_ns=(
                _integer(event, "gpu_duration_ns")
                if event.get("gpu_duration_ns") is not None
                else None
            ),
            fanout=_integer(event, "fanout", 1),
            attributes=_attributes(event),
        )

    def _state_event(self, event: Mapping[str, JsonValue]) -> StateOperationEventV1:
        operation = StateOperationType(_required_string(event, "operation_type"))
        return StateOperationEventV1(
            provenance=WorkloadProvenance(_required_string(event, "workload_evidence_class")),
            timing_measurement_class=TimingMeasurementClass(
                _required_string(event, "timing_measurement_class")
            ),
            trace_id=_required_string(event, "trace_id"),
            session_id=_required_string(event, "session_id"),
            branch_group_id=_optional_string(event, "branch_group_id"),
            logical_state_id=_required_string(event, "logical_state_id"),
            branch_id=_optional_string(event, "branch_id"),
            tenant_id=_optional_string(event, "tenant") or "not-recorded",
            security_domain=_optional_string(event, "tenant") or "not-recorded",
            host=_required_string(event, "host"),
            process_id=_integer(event, "process"),
            rank=_integer(event, "rank"),
            device=_optional_string(event, "device"),
            monotonic_timestamp_ns=_integer(event, "monotonic_timestamp_ns"),
            normalized_timestamp_ns=_integer(event, "normalized_timestamp_ns"),
            duration_ns=_integer(event, "duration_ns"),
            clock_source=ClockSource.PERF_COUNTER,
            alignment_confidence=_number(event, "alignment_confidence", 1.0),
            operation_type=operation,
            state_segment=_state_segment(event.get("state_segment")),
            source_physical_representation=(
                _optional_string(event, "source_physical_representation") or "not-recorded"
            ),
            destination_physical_representation=(
                _optional_string(event, "destination_physical_representation") or "not-recorded"
            ),
            bytes=_integer(event, "bytes", _integer(event, "logical_bytes")),
            alignment_bytes=_integer(event, "alignment"),
            page_size_bytes=_integer(event, "page_size"),
            chunk_size_bytes=_integer(event, "chunk_size"),
            fanout=_integer(event, "fanout", 1),
            dependency_event_ids=(),
            concurrency=1,
            queue_delay_ns=_integer(event, "queue_delay_ns"),
            operation_latency_ns=_integer(event, "duration_ns"),
            cpu_time_ns=_integer(event, "cpu_time_ns"),
            gpu_time_ns=_integer(event, "gpu_duration_ns"),
            transfer_time_ns=_integer(event, "transfer_latency_ns"),
            result=OperationResult(_required_string(event, "result")),
            state_epoch=_integer(event, "state_epoch"),
            attributes=_attributes(event),
        )


class CanonicalContinuumRecorder:
    """Adapt schema-neutral Continuum observations into StateOperationTrace v1."""

    def __init__(
        self,
        buffer: BoundedTraceBuffer,
        *,
        trace_id: str,
        session_id: str,
        branch_group_id: str | None,
    ) -> None:
        self.buffer = buffer
        self.trace_id = trace_id
        self.session_id = session_id
        self.branch_group_id = branch_group_id
        self._origin_ns: int | None = None

    def record(self, observation: StateOperationObservation) -> None:
        if self._origin_ns is None:
            self._origin_ns = observation.monotonic_start_ns
        metadata = {
            key: value
            for key, value in observation.metadata.items()
            if value is None or isinstance(value, (bool, int, float, str))
        }
        metadata.update(
            {
                "source_observation_sequence": observation.sequence,
                "evidence_references_json": json.dumps(
                    observation.evidence_references, separators=(",", ":")
                ),
            }
        )
        transport = TransportType.NONE
        if observation.operation in {"STATE_SEND", "STATE_RECEIVE"}:
            transport = (
                TransportType.SYNTHETIC
                if observation.measurement_kind.value == "SIMULATED_HARDWARE"
                else TransportType.MEMORY_COPY
            )
        event = StateOperationEventV1(
            provenance=WorkloadProvenance(observation.workload_evidence_class.value),
            timing_measurement_class=TimingMeasurementClass(observation.measurement_kind.value),
            trace_id=self.trace_id,
            session_id=self.session_id,
            branch_group_id=self.branch_group_id,
            logical_state_id=observation.logical_state_id,
            branch_id=observation.branch_id,
            tenant_id=observation.tenant_security_domain,
            security_domain=observation.tenant_security_domain,
            host=socket.gethostname(),
            process_id=os.getpid(),
            rank=0,
            device="cpu",
            monotonic_timestamp_ns=observation.monotonic_start_ns,
            normalized_timestamp_ns=observation.monotonic_start_ns - self._origin_ns,
            duration_ns=observation.duration_ns,
            clock_source=ClockSource.PERF_COUNTER,
            alignment_confidence=1.0,
            operation_type=StateOperationType(observation.operation),
            state_segment=StateSegment(observation.state_segment),
            source_physical_representation=observation.source_physical_representation,
            destination_physical_representation=observation.destination_physical_representation,
            bytes=observation.bytes,
            alignment_bytes=observation.alignment,
            page_size_bytes=observation.page_size,
            chunk_size_bytes=observation.chunk_size,
            fanout=observation.fanout,
            dependency_event_ids=tuple(
                f"{self.trace_id}:{sequence}" for sequence in observation.dependencies
            ),
            concurrency=observation.concurrency,
            queue_delay_ns=observation.queue_delay_ns,
            operation_latency_ns=observation.duration_ns,
            cpu_time_ns=observation.cpu_time_ns,
            transfer_time_ns=observation.transfer_time_ns,
            result=OperationResult(observation.result),
            failure=observation.failure,
            state_epoch=observation.state_epoch,
            source_location=MemoryLocation.HOST_DRAM,
            destination_location=MemoryLocation.HOST_DRAM,
            transport_type=transport,
            attributes=metadata,
        )
        self.buffer.record(event)


__all__ = ["CanonicalContinuumRecorder", "CanonicalLifecycleRecorder"]
