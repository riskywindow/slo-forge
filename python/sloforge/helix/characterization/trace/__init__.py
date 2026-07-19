"""BranchWorkloadTrace v1 and StateOperationTrace v1 event infrastructure."""

from .buffer import BoundedTraceBuffer, TraceBufferStats
from .canonical import (
    TraceIntegrityError,
    canonical_hash,
    canonical_json,
    event_content_hash,
    seal_event,
    verify_event,
)
from .conversion import jsonl_to_parquet, jsonl_to_perfetto, parquet_to_jsonl, parquet_to_perfetto
from .io import JsonlTraceWriter, TraceFormatError, iter_jsonl, write_jsonl
from .manifest import artifact_from_file, build_manifest, trace_corpus_hash, write_manifest
from .models import (
    BRANCH_TRACE_SCHEMA_VERSION,
    STATE_TRACE_SCHEMA_VERSION,
    TRACE_MANIFEST_SCHEMA_VERSION,
    TRACE_PRODUCER_VERSION,
    BranchOperationType,
    BranchWorkloadEventV1,
    ClockSource,
    HardwareDevice,
    HardwareManifestV1,
    MemoryLocation,
    OperationResult,
    SamplingConfigurationV1,
    SoftwareManifestV1,
    StateOperationEventV1,
    StateOperationType,
    StateSegment,
    TimingMeasurementClass,
    TraceArtifactV1,
    TraceEventV1,
    TraceLevel,
    TraceManifestV1,
    TransportType,
    VersionedComponent,
    WorkloadProvenance,
)
from .parquet import ParquetUnavailableError, iter_parquet, write_parquet
from .perfetto import write_perfetto

# Explicit names for architecture tooling while retaining the more precise EventV1 names.
BranchWorkloadTraceV1 = BranchWorkloadEventV1
StateOperationTraceV1 = StateOperationEventV1

__all__ = [
    "BRANCH_TRACE_SCHEMA_VERSION",
    "STATE_TRACE_SCHEMA_VERSION",
    "TRACE_MANIFEST_SCHEMA_VERSION",
    "TRACE_PRODUCER_VERSION",
    "BoundedTraceBuffer",
    "BranchOperationType",
    "BranchWorkloadEventV1",
    "BranchWorkloadTraceV1",
    "ClockSource",
    "HardwareDevice",
    "HardwareManifestV1",
    "JsonlTraceWriter",
    "MemoryLocation",
    "OperationResult",
    "ParquetUnavailableError",
    "SamplingConfigurationV1",
    "SoftwareManifestV1",
    "StateOperationEventV1",
    "StateOperationTraceV1",
    "StateOperationType",
    "StateSegment",
    "TimingMeasurementClass",
    "TraceArtifactV1",
    "TraceBufferStats",
    "TraceEventV1",
    "TraceFormatError",
    "TraceIntegrityError",
    "TraceLevel",
    "TraceManifestV1",
    "TransportType",
    "VersionedComponent",
    "WorkloadProvenance",
    "artifact_from_file",
    "build_manifest",
    "canonical_hash",
    "canonical_json",
    "event_content_hash",
    "iter_jsonl",
    "iter_parquet",
    "jsonl_to_parquet",
    "jsonl_to_perfetto",
    "parquet_to_jsonl",
    "parquet_to_perfetto",
    "seal_event",
    "trace_corpus_hash",
    "verify_event",
    "write_jsonl",
    "write_manifest",
    "write_parquet",
    "write_perfetto",
]
