"""Versioned event models for BranchFabric workload characterization.

These types intentionally do not replace ``sloforge.helix.ir.BranchWorkloadTrace``.
That legacy document describes a scheduled workload.  The records here are the
event-oriented, high-volume measurement boundary used by characterization.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

BRANCH_TRACE_SCHEMA_VERSION = "sloforge.branchfabric.branch-workload-event/v1"
STATE_TRACE_SCHEMA_VERSION = "sloforge.branchfabric.state-operation-event/v1"
TRACE_MANIFEST_SCHEMA_VERSION = "sloforge.branchfabric.trace-manifest/v1"
TRACE_PRODUCER_VERSION = "sloforge-helix-characterization/1"
UNSEALED_CONTENT_HASH = "0" * 64

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
AttributeValue = bool | int | float | str | None
MAX_ATTRIBUTES = 64


def _validate_attributes(attributes: dict[str, AttributeValue]) -> None:
    if len(attributes) > MAX_ATTRIBUTES:
        raise ValueError(f"event attributes exceed the {MAX_ATTRIBUTES}-entry bound")
    for key, value in attributes.items():
        if not key or len(key) > 128:
            raise ValueError("event attribute keys must contain 1..128 characters")
        if isinstance(value, str) and len(value) > 4096:
            raise ValueError("event attribute string values must not exceed 4096 characters")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("event attribute floats must be finite")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class TraceLevel(StrEnum):
    DISABLED = "disabled"
    MINIMAL = "minimal"
    FULL = "full"


class WorkloadProvenance(StrEnum):
    SYNTHETIC = "SYNTHETIC"
    REPLAYED = "REPLAYED"
    HARDWARE_BACKED_REAL = "HARDWARE_BACKED_REAL"
    SIMULATED_HARDWARE = "SIMULATED_HARDWARE"


class TimingMeasurementClass(StrEnum):
    """How event timings were obtained, independent of workload provenance."""

    SYNTHETIC = "SYNTHETIC"
    REPLAYED = "REPLAYED"
    HARDWARE_BACKED_REAL = "HARDWARE_BACKED_REAL"
    SIMULATED_HARDWARE = "SIMULATED_HARDWARE"


class ClockSource(StrEnum):
    MONOTONIC = "monotonic"
    MONOTONIC_RAW = "monotonic_raw"
    PERF_COUNTER = "perf_counter"
    CUDA_GLOBAL_TIMER = "cuda_global_timer"
    CUPTI = "cupti"
    SYNTHETIC = "synthetic"


class MemoryLocation(StrEnum):
    GPU_HBM = "gpu_hbm"
    GPU_PEER_HBM = "gpu_peer_hbm"
    HOST_DRAM = "host_dram"
    PINNED_MEMORY = "pinned_memory"
    LOCAL_NVME = "local_nvme"
    REMOTE_STORAGE = "remote_storage"
    NIC = "nic"
    TRANSPORT_BUFFER = "transport_buffer"
    UNKNOWN = "unknown"


class StateSegment(StrEnum):
    MODEL = "model"
    TOKEN_HISTORY = "token_history"
    KV = "kv"
    RECURRENT = "recurrent"
    SAMPLER = "sampler"
    GUIDED_DECODING = "guided_decoding"
    WORKFLOW = "workflow"
    ENVIRONMENT = "environment"
    FILESYSTEM = "filesystem"
    DATABASE = "database"
    PROCESS_RECONSTRUCTION = "process_reconstruction"
    TRANSACTION = "transaction"
    INTEGRITY = "integrity"
    RUNTIME_RECONSTRUCTIBLE = "runtime_reconstructible"
    UNKNOWN = "unknown"


class TransportType(StrEnum):
    NONE = "none"
    MEMORY_COPY = "memory_copy"
    PCIE = "pcie"
    NVLINK = "nvlink"
    NCCL = "nccl"
    TCP = "tcp"
    RDMA = "rdma"
    SHARED_MEMORY = "shared_memory"
    STORAGE = "storage"
    SYNTHETIC = "synthetic"


class BranchOperationType(StrEnum):
    BRANCH_POINT = "BRANCH_POINT"
    CAPTURE = "CAPTURE"
    BRANCH_FORK = "BRANCH_FORK"
    BRANCH_READY = "BRANCH_READY"
    BRANCH_DIVERGENCE = "BRANCH_DIVERGENCE"
    BRANCH_PRUNE = "BRANCH_PRUNE"
    BRANCH_ABORT = "BRANCH_ABORT"
    BRANCH_COMMIT = "BRANCH_COMMIT"
    BRANCH_COMPLETE = "BRANCH_COMPLETE"
    BRANCH_MIGRATION = "BRANCH_MIGRATION"
    CHECKPOINT = "CHECKPOINT"
    ROLLOUT = "ROLLOUT"
    REWARD = "REWARD"
    TRAIN = "TRAIN"
    EVALUATE = "EVALUATE"
    CANARY = "CANARY"
    PROMOTE = "PROMOTE"
    ROLLBACK = "ROLLBACK"
    STATE_ALLOC = "STATE_ALLOC"
    STATE_MAP = "STATE_MAP"
    STATE_PUBLISH = "STATE_PUBLISH"
    STATE_FORK = "STATE_FORK"
    STATE_COW = "STATE_COW"
    STATE_APPEND = "STATE_APPEND"
    STATE_READ = "STATE_READ"
    STATE_WRITE = "STATE_WRITE"
    STATE_SNAPSHOT = "STATE_SNAPSHOT"
    STATE_DELTA = "STATE_DELTA"
    STATE_RESHARD = "STATE_RESHARD"
    STATE_TRANSPOSE = "STATE_TRANSPOSE"
    STATE_REPACK = "STATE_REPACK"
    STATE_QUANTIZE = "STATE_QUANTIZE"
    STATE_DEQUANTIZE = "STATE_DEQUANTIZE"
    STATE_COMPRESS = "STATE_COMPRESS"
    STATE_DECOMPRESS = "STATE_DECOMPRESS"
    STATE_HASH = "STATE_HASH"
    STATE_CHECKSUM = "STATE_CHECKSUM"
    STATE_ENCRYPT = "STATE_ENCRYPT"
    STATE_DECRYPT = "STATE_DECRYPT"
    STATE_SEND = "STATE_SEND"
    STATE_RECEIVE = "STATE_RECEIVE"
    STATE_MULTICAST = "STATE_MULTICAST"
    STATE_ACK = "STATE_ACK"
    STATE_RETRY = "STATE_RETRY"
    STATE_COMMIT = "STATE_COMMIT"
    STATE_ABORT = "STATE_ABORT"
    STATE_RECLAIM = "STATE_RECLAIM"
    STATE_FREE = "STATE_FREE"


class StateOperationType(StrEnum):
    STATE_ALLOC = "STATE_ALLOC"
    STATE_MAP = "STATE_MAP"
    STATE_PUBLISH = "STATE_PUBLISH"
    STATE_FORK = "STATE_FORK"
    STATE_COW = "STATE_COW"
    STATE_APPEND = "STATE_APPEND"
    STATE_READ = "STATE_READ"
    STATE_WRITE = "STATE_WRITE"
    STATE_SNAPSHOT = "STATE_SNAPSHOT"
    STATE_DELTA = "STATE_DELTA"
    STATE_RESHARD = "STATE_RESHARD"
    STATE_TRANSPOSE = "STATE_TRANSPOSE"
    STATE_REPACK = "STATE_REPACK"
    STATE_QUANTIZE = "STATE_QUANTIZE"
    STATE_DEQUANTIZE = "STATE_DEQUANTIZE"
    STATE_COMPRESS = "STATE_COMPRESS"
    STATE_DECOMPRESS = "STATE_DECOMPRESS"
    STATE_HASH = "STATE_HASH"
    STATE_CHECKSUM = "STATE_CHECKSUM"
    STATE_ENCRYPT = "STATE_ENCRYPT"
    STATE_DECRYPT = "STATE_DECRYPT"
    STATE_SEND = "STATE_SEND"
    STATE_RECEIVE = "STATE_RECEIVE"
    STATE_MULTICAST = "STATE_MULTICAST"
    STATE_ACK = "STATE_ACK"
    STATE_RETRY = "STATE_RETRY"
    STATE_COMMIT = "STATE_COMMIT"
    STATE_ABORT = "STATE_ABORT"
    STATE_RECLAIM = "STATE_RECLAIM"
    STATE_FREE = "STATE_FREE"


class OperationResult(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    RETRY = "retry"
    SKIPPED = "skipped"


class BranchWorkloadEventV1(StrictModel):
    """One canonical high-level branch or state-lifecycle observation."""

    schema_version: Literal["sloforge.branchfabric.branch-workload-event/v1"] = (
        "sloforge.branchfabric.branch-workload-event/v1"
    )
    kind: Literal["BranchWorkloadTraceEvent"] = "BranchWorkloadTraceEvent"
    trace_producer_version: NonEmpty = TRACE_PRODUCER_VERSION
    collection_level: TraceLevel = TraceLevel.FULL
    provenance: WorkloadProvenance
    timing_measurement_class: TimingMeasurementClass

    trace_id: NonEmpty
    session_id: NonEmpty
    branch_group_id: str | None = None
    branch_id: str | None = None
    parent_branch_id: str | None = None
    policy_epoch: str | None = None
    environment_id: str | None = None
    transaction_id: str | None = None
    host: NonEmpty
    process_id: int = Field(ge=0)
    rank: int | None = Field(default=None, ge=0)
    device: str | None = None

    monotonic_timestamp_ns: int = Field(ge=0)
    normalized_timestamp_ns: int = Field(ge=0)
    duration_ns: int = Field(ge=0)
    clock_source: ClockSource
    alignment_confidence: float = Field(ge=0.0, le=1.0)

    operation_type: BranchOperationType
    logical_state_id: str | None = None
    physical_state_id: str | None = None
    state_segment: StateSegment = StateSegment.UNKNOWN
    page: int | None = Field(default=None, ge=0)
    version: int | None = Field(default=None, ge=0)
    source_epoch: int | None = Field(default=None, ge=0)
    destination_epoch: int | None = Field(default=None, ge=0)

    logical_bytes: int = Field(default=0, ge=0)
    physical_bytes: int = Field(default=0, ge=0)
    compressed_bytes: int = Field(default=0, ge=0)
    transferred_bytes: int = Field(default=0, ge=0)
    metadata_bytes: int = Field(default=0, ge=0)

    location: MemoryLocation = MemoryLocation.UNKNOWN
    source_location: MemoryLocation = MemoryLocation.UNKNOWN
    destination_location: MemoryLocation = MemoryLocation.UNKNOWN
    shared_root: bool = False
    private_suffix: bool = False
    cow_allocation: bool = False

    queue_delay_ns: int = Field(default=0, ge=0)
    execution_latency_ns: int = Field(default=0, ge=0)
    transfer_latency_ns: int = Field(default=0, ge=0)
    transform_latency_ns: int = Field(default=0, ge=0)
    wait_latency_ns: int = Field(default=0, ge=0)
    cpu_cycles: int | None = Field(default=None, ge=0)
    cpu_time_ns: int | None = Field(default=None, ge=0)
    gpu_duration_ns: int | None = Field(default=None, ge=0)

    transport_type: TransportType = TransportType.NONE
    transport_source: str | None = None
    transport_destination: str | None = None
    chunk_size_bytes: int | None = Field(default=None, ge=1)
    fanout: int = Field(default=1, ge=1)
    retransmission: bool = False
    error: str | None = None

    gpu_model: str | None = None
    nic: str | None = None
    numa_node: int | None = Field(default=None, ge=0)
    pcie_path: str | None = None
    network_rail: str | None = None
    memory_tier: str | None = None
    attributes: dict[str, AttributeValue] = Field(default_factory=dict)

    event_sequence: int = Field(default=0, ge=0)
    content_hash: Sha256 = UNSEALED_CONTENT_HASH

    @model_validator(mode="after")
    def validate_semantics(self) -> Self:
        if self.parent_branch_id is not None and self.parent_branch_id == self.branch_id:
            raise ValueError("branch cannot be its own parent")
        if self.error is not None and not self.error.strip():
            raise ValueError("error must be non-empty when present")
        if self.cow_allocation and self.operation_type is not BranchOperationType.STATE_COW:
            raise ValueError("cow_allocation is only valid for STATE_COW")
        _validate_attributes(self.attributes)
        return self


class StateOperationEventV1(StrictModel):
    """One lower-level state operation suitable for later architecture replay."""

    schema_version: Literal["sloforge.branchfabric.state-operation-event/v1"] = (
        "sloforge.branchfabric.state-operation-event/v1"
    )
    kind: Literal["StateOperationTraceEvent"] = "StateOperationTraceEvent"
    trace_producer_version: NonEmpty = TRACE_PRODUCER_VERSION
    collection_level: TraceLevel = TraceLevel.FULL
    provenance: WorkloadProvenance
    timing_measurement_class: TimingMeasurementClass

    trace_id: NonEmpty
    session_id: NonEmpty
    branch_group_id: str | None = None
    logical_state_id: NonEmpty
    branch_id: str | None = None
    tenant_id: NonEmpty
    security_domain: NonEmpty
    host: NonEmpty
    process_id: int = Field(ge=0)
    rank: int | None = Field(default=None, ge=0)
    device: str | None = None

    monotonic_timestamp_ns: int = Field(ge=0)
    normalized_timestamp_ns: int = Field(ge=0)
    duration_ns: int = Field(ge=0)
    clock_source: ClockSource
    alignment_confidence: float = Field(ge=0.0, le=1.0)

    operation_type: StateOperationType
    state_segment: StateSegment = StateSegment.UNKNOWN
    source_physical_representation: NonEmpty
    destination_physical_representation: NonEmpty
    bytes: int = Field(ge=0)
    # Zero is the explicit N/A value for nonpaged metadata, transaction, and
    # filesystem operations. It must not be interpreted as a one-byte granularity.
    alignment_bytes: int = Field(ge=0)
    page_size_bytes: int = Field(ge=0)
    chunk_size_bytes: int = Field(ge=0)
    fanout: int = Field(default=1, ge=1)
    dependency_event_ids: tuple[str, ...] = ()
    concurrency: int = Field(default=1, ge=1)
    queue_delay_ns: int = Field(default=0, ge=0)
    operation_latency_ns: int = Field(ge=0)
    cpu_time_ns: int = Field(default=0, ge=0)
    gpu_time_ns: int = Field(default=0, ge=0)
    transfer_time_ns: int = Field(default=0, ge=0)
    result: OperationResult
    failure: str | None = None
    state_epoch: int = Field(ge=0)
    source_location: MemoryLocation = MemoryLocation.UNKNOWN
    destination_location: MemoryLocation = MemoryLocation.UNKNOWN
    transport_type: TransportType = TransportType.NONE
    attributes: dict[str, AttributeValue] = Field(default_factory=dict)

    event_sequence: int = Field(default=0, ge=0)
    content_hash: Sha256 = UNSEALED_CONTENT_HASH

    @model_validator(mode="after")
    def validate_semantics(self) -> Self:
        if len(self.dependency_event_ids) != len(set(self.dependency_event_ids)):
            raise ValueError("event dependencies must be unique")
        if self.result is OperationResult.FAILURE and self.failure is None:
            raise ValueError("failed operations require failure detail")
        if self.result is not OperationResult.FAILURE and self.failure is not None:
            raise ValueError("only failed operations may carry failure detail")
        _validate_attributes(self.attributes)
        return self


TraceEventV1 = BranchWorkloadEventV1 | StateOperationEventV1


class VersionedComponent(StrictModel):
    name: NonEmpty
    version: NonEmpty
    source: NonEmpty


class HardwareDevice(StrictModel):
    kind: NonEmpty
    name: NonEmpty
    identifier: str | None = None
    numa_node: int | None = Field(default=None, ge=0)
    memory_bytes: int | None = Field(default=None, ge=0)
    attributes: dict[str, str | int | float | bool] = Field(default_factory=dict)


class HardwareManifestV1(StrictModel):
    host: NonEmpty
    machine: NonEmpty
    cpu_model: NonEmpty
    logical_cpu_count: int = Field(ge=1)
    memory_bytes: int = Field(ge=0)
    numa_node_count: int = Field(ge=1)
    devices: tuple[HardwareDevice, ...] = ()
    pcie_paths: tuple[str, ...] = ()
    network_rails: tuple[str, ...] = ()
    clock_settings: tuple[str, ...] = ()


class SoftwareManifestV1(StrictModel):
    operating_system: NonEmpty
    kernel: NonEmpty
    python_version: NonEmpty
    rust_version: str | None = None
    components: tuple[VersionedComponent, ...] = ()


class TraceArtifactV1(StrictModel):
    format: Literal["jsonl", "parquet", "perfetto", "manifest"]
    uri: NonEmpty
    byte_length: int = Field(ge=0)
    sha256: Sha256
    event_count: int = Field(ge=0)


class SamplingConfigurationV1(StrictModel):
    mode: Literal["none", "deterministic_stride"] = "none"
    stride: int = Field(default=1, ge=1)
    seed: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def valid_mode(self) -> Self:
        if self.mode == "none" and self.stride != 1:
            raise ValueError("sampling mode 'none' requires stride 1")
        if self.mode == "deterministic_stride" and self.stride == 1:
            raise ValueError("deterministic stride sampling requires stride greater than 1")
        return self


class TraceManifestV1(StrictModel):
    schema_version: Literal["sloforge.branchfabric.trace-manifest/v1"] = (
        "sloforge.branchfabric.trace-manifest/v1"
    )
    kind: Literal["BranchFabricTraceManifest"] = "BranchFabricTraceManifest"
    trace_producer_version: NonEmpty = TRACE_PRODUCER_VERSION
    trace_id: NonEmpty
    session_id: NonEmpty
    created_at: NonEmpty
    seed: int = Field(ge=0)
    provenance: WorkloadProvenance
    collection_level: TraceLevel
    buffer_capacity_events: int = Field(ge=1)
    attempted_events: int = Field(ge=0)
    accepted_events: int = Field(ge=0)
    dropped_events: int = Field(ge=0)
    filtered_events: int = Field(ge=0)
    highest_event_sequence: int | None = Field(default=None, ge=0)
    event_counts: dict[str, int]
    sampling: SamplingConfigurationV1 = SamplingConfigurationV1()
    hardware: HardwareManifestV1
    software: SoftwareManifestV1
    artifacts: tuple[TraceArtifactV1, ...]
    trace_corpus_hash: Sha256

    @model_validator(mode="after")
    def valid_accounting(self) -> Self:
        if self.attempted_events != (
            self.accepted_events + self.dropped_events + self.filtered_events
        ):
            raise ValueError("trace event accounting does not balance")
        if any(count < 0 for count in self.event_counts.values()):
            raise ValueError("event counts cannot be negative")
        if sum(self.event_counts.values()) != self.accepted_events:
            raise ValueError("event counts must sum to accepted events")
        return self


__all__ = [
    "BRANCH_TRACE_SCHEMA_VERSION",
    "MAX_ATTRIBUTES",
    "STATE_TRACE_SCHEMA_VERSION",
    "TRACE_MANIFEST_SCHEMA_VERSION",
    "TRACE_PRODUCER_VERSION",
    "UNSEALED_CONTENT_HASH",
    "BranchOperationType",
    "BranchWorkloadEventV1",
    "ClockSource",
    "HardwareDevice",
    "HardwareManifestV1",
    "MemoryLocation",
    "OperationResult",
    "SamplingConfigurationV1",
    "SoftwareManifestV1",
    "StateOperationEventV1",
    "StateOperationType",
    "StateSegment",
    "TimingMeasurementClass",
    "TraceArtifactV1",
    "TraceEventV1",
    "TraceLevel",
    "TraceManifestV1",
    "TransportType",
    "VersionedComponent",
    "WorkloadProvenance",
]
