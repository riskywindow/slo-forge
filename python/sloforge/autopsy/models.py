"""Versioned canonical Autopsy evidence models.

All timestamps are integer nanoseconds. Cross-host normalization is explicit and
keeps alignment uncertainty alongside the transformed timestamps.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class SourceClock(StrEnum):
    MONOTONIC = "monotonic"
    MONOTONIC_RAW = "monotonic_raw"
    CUDA_GLOBAL_TIMER = "cuda_global_timer"
    CUPTI = "cupti"
    WALL = "wall"
    SYNTHETIC = "synthetic"


class EventType(StrEnum):
    REQUEST = "request"
    GATEWAY_QUEUE = "gateway_queue"
    BACKEND_QUEUE = "backend_queue"
    SCHEDULER = "scheduler"
    PREFILL = "prefill"
    DECODE = "decode"
    TOKEN_STREAM = "token_stream"
    CPU_LAUNCH = "cpu_launch"
    GPU_COMPUTE = "gpu_compute"
    GPU_MEMORY = "gpu_memory"
    KERNEL = "kernel"
    COLLECTIVE = "collective"
    COLLECTIVE_WAIT = "collective_wait"
    EXPERT_DISPATCH = "expert_dispatch"
    EXPERT_COMBINE = "expert_combine"
    KV_TRANSFER = "kv_transfer"
    NETWORK_TRANSFER = "network_transfer"
    PCIE_TRANSFER = "pcie_transfer"
    NVLINK_TRANSFER = "nvlink_transfer"
    STORAGE_READ = "storage_read"
    STARTUP = "startup"
    READINESS = "readiness"
    CONTROLLER = "controller"
    RECOVERY = "recovery"
    FAULT = "fault"
    HEALTH = "health"


class AlignmentQuality(StrEnum):
    GOOD = "good"
    DEGRADED = "degraded"
    INSUFFICIENT = "insufficient"


class CounterValue(StrictModel):
    name: NonEmpty
    value: float
    unit: NonEmpty

    @model_validator(mode="after")
    def finite(self) -> Self:
        if not math.isfinite(self.value):
            raise ValueError("counter value must be finite")
        return self


class ResourceRef(StrictModel):
    resource_id: NonEmpty
    resource_type: Literal[
        "cpu_core_group",
        "numa_memory",
        "gpu_compute",
        "gpu_hbm",
        "copy_engine",
        "nvlink",
        "nvswitch",
        "pcie",
        "nic_queue",
        "network_rail",
        "storage",
        "process",
        "logical_queue",
    ]
    topology_node_id: str | None = None
    contention_domain: str | None = None


class EvidenceRef(StrictModel):
    source: NonEmpty
    artifact_uri: NonEmpty
    sha256: Sha256
    record_index: int | None = Field(default=None, ge=0)


class AutopsyEvent(StrictModel):
    schema_version: Literal["sloforge.autopsy.event/v1"] = "sloforge.autopsy.event/v1"
    event_id: NonEmpty
    event_type: EventType
    host: NonEmpty
    process_id: int | None = Field(default=None, ge=0)
    thread_id: str | None = None
    rank: int | None = Field(default=None, ge=0)
    replica: str | None = None
    gpu: str | None = None
    nic: str | None = None
    numa_domain: str | None = None
    request_id: str | None = None
    model_id: str | None = None
    operation: str | None = None
    start_ns: int = Field(ge=0)
    end_ns: int = Field(ge=0)
    source_clock: SourceClock
    normalized_start_ns: int | None = Field(default=None, ge=0)
    normalized_end_ns: int | None = Field(default=None, ge=0)
    alignment_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    alignment_uncertainty_ns: int | None = Field(default=None, ge=0)
    parent_event_id: str | None = None
    dependency_event_ids: tuple[str, ...] = ()
    resource: ResourceRef
    counters: tuple[CounterValue, ...] = ()
    evidence: EvidenceRef

    @model_validator(mode="after")
    def valid_interval(self) -> Self:
        if self.end_ns < self.start_ns:
            raise ValueError("event end precedes start")
        normalized = (self.normalized_start_ns, self.normalized_end_ns)
        if (normalized[0] is None) != (normalized[1] is None):
            raise ValueError("normalized timestamps must be both present or both absent")
        if normalized[0] is not None and normalized[1] is not None:
            if normalized[1] < normalized[0]:
                raise ValueError("normalized event end precedes start")
            if self.alignment_uncertainty_ns is None:
                raise ValueError("normalized timestamps require alignment uncertainty")
        if len(self.dependency_event_ids) != len(set(self.dependency_event_ids)):
            raise ValueError("event dependencies must be unique")
        if self.event_id in self.dependency_event_ids or self.parent_event_id == self.event_id:
            raise ValueError("an event cannot depend on itself")
        names = [counter.name for counter in self.counters]
        if len(names) != len(set(names)):
            raise ValueError("counter names must be unique within an event")
        return self

    @property
    def duration_ns(self) -> int:
        start = self.normalized_start_ns if self.normalized_start_ns is not None else self.start_ns
        end = self.normalized_end_ns if self.normalized_end_ns is not None else self.end_ns
        return end - start


class ClockSample(StrictModel):
    host: NonEmpty
    local_monotonic_ns: int = Field(ge=0)
    reference_monotonic_ns: int = Field(ge=0)
    round_trip_ns: int = Field(ge=0)
    captured_wall_ns: int = Field(ge=0)


class AlignmentEstimate(StrictModel):
    host: NonEmpty
    reference_host: NonEmpty
    offset_ns: float
    drift_ppm: float
    reference_local_ns: int = Field(ge=0)
    uncertainty_ns: int = Field(ge=0)
    sample_count: int = Field(ge=1)
    confidence: float = Field(ge=0.0, le=1.0)
    quality: AlignmentQuality
    residual_p95_ns: int = Field(ge=0)

    @model_validator(mode="after")
    def finite_coefficients(self) -> Self:
        if not math.isfinite(self.offset_ns) or not math.isfinite(self.drift_ppm):
            raise ValueError("alignment coefficients must be finite")
        return self


class FaultInterval(StrictModel):
    fault_id: NonEmpty
    fault_type: NonEmpty
    target: NonEmpty
    start_ns: int = Field(ge=0)
    end_ns: int = Field(ge=0)
    ground_truth: bool = True
    parameters: tuple[CounterValue, ...] = ()

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.end_ns < self.start_ns:
            raise ValueError("fault end precedes start")
        return self


class AutopsyRun(StrictModel):
    schema_version: Literal["sloforge.autopsy.run/v1"] = "sloforge.autopsy.run/v1"
    run_id: NonEmpty
    source: Literal["live", "simulator", "synthetic_fixture", "imported_trace"]
    topology_fingerprint: Sha256
    physical_plan_hash: Sha256
    workload_fingerprint: Sha256
    reference_host: NonEmpty
    events: tuple[AutopsyEvent, ...]
    alignments: tuple[AlignmentEstimate, ...] = ()
    fault_intervals: tuple[FaultInterval, ...] = ()
    artifacts: tuple[EvidenceRef, ...]
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_event_graph(self) -> Self:
        ids = [event.event_id for event in self.events]
        if len(ids) != len(set(ids)):
            raise ValueError("event identifiers must be unique")
        known = set(ids)
        for event in self.events:
            dependencies = set(event.dependency_event_ids)
            if event.parent_event_id is not None:
                dependencies.add(event.parent_event_id)
            unknown = dependencies - known
            if unknown:
                raise ValueError(
                    f"event {event.event_id} references unknown dependencies {unknown}"
                )
        hosts = [alignment.host for alignment in self.alignments]
        if len(hosts) != len(set(hosts)):
            raise ValueError("at most one alignment estimate is allowed per host")
        return self
