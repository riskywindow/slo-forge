"""Strict evidence-backed BranchFabric requirements schema and compiler.

The compiler does not estimate missing parameters.  Every numeric or enum
recommendation is wrapped with its source experiment and immutable raw-artifact
identity.  ``UNKNOWN`` and ``UNAVAILABLE`` therefore remain explicit evidence
states instead of becoming zeroes or guessed defaults.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from sloforge.helix.characterization.matrix import EvidenceClass, WorkloadClass
from sloforge.helix.characterization.trace.canonical import canonical_json
from sloforge.helix.characterization.trace.manifest import trace_corpus_hash
from sloforge.helix.characterization.trace.models import (
    StateOperationType,
    StateSegment,
    TraceArtifactV1,
)

REQUIREMENTS_SCHEMA_VERSION = "sloforge.branchfabric.requirements/v1"
REQUIREMENTS_COMPILER_VERSION = "sloforge-helix-requirements/1"
MAX_ARTIFACTS = 100_000
MAX_RECOMMENDATIONS = 100_000

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4096)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
NonNegativeNumber = Annotated[int | float, Field(ge=0, allow_inf_nan=False)]


class RequirementsModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class Availability(StrEnum):
    AVAILABLE = "AVAILABLE"
    UNKNOWN = "UNKNOWN"
    UNAVAILABLE = "UNAVAILABLE"


class ConfidenceOrPercentile(StrEnum):
    MINIMUM = "minimum"
    MEAN = "mean"
    MEDIAN = "median"
    P50 = "p50"
    P90 = "p90"
    P95 = "p95"
    P99 = "p99"
    MAXIMUM = "maximum"
    POINT_ESTIMATE = "point_estimate"
    CI95_LOWER = "ci95_lower"
    CI95_UPPER = "ci95_upper"
    COUNTER_ABSENT = "counter_absent"
    CAPABILITY_UNAVAILABLE = "capability_unavailable"


class RequirementUnit(StrEnum):
    BYTES = "bytes"
    BYTES_PER_SECOND = "bytes_per_second"
    BITS_PER_SECOND = "bits_per_second"
    NANOSECONDS = "nanoseconds"
    MICROSECONDS = "microseconds"
    MILLISECONDS = "milliseconds"
    SECONDS = "seconds"
    COUNT = "count"
    RATIO = "ratio"
    PERCENT = "percent"
    EVENTS_PER_SECOND = "events_per_second"
    OPERATIONS_PER_SECOND = "operations_per_second"
    FAULTS_PER_SECOND = "faults_per_second"
    BYTES_PER_EVENT = "bytes_per_event"
    BYTES_PER_BRANCH = "bytes_per_branch"
    TOKENS = "tokens"
    TOKENS_PER_SECOND = "tokens_per_second"
    OPERATIONS_PER_BRANCH = "operations_per_branch"


class EvidenceReference(RequirementsModel):
    source_experiment: NonEmpty
    sample_count: int = Field(ge=0, le=2**63 - 1)
    evidence_class: EvidenceClass
    confidence_or_percentile: ConfidenceOrPercentile
    artifact_reference: NonEmpty
    artifact_sha256: Sha256


class NumericRequirement(RequirementsModel):
    availability: Availability
    value: NonNegativeNumber | None
    unit: RequirementUnit
    evidence: EvidenceReference
    rationale: NonEmpty

    @model_validator(mode="after")
    def value_matches_availability(self) -> Self:
        if (self.availability is Availability.AVAILABLE) != (self.value is not None):
            raise ValueError("numeric value must be present exactly when availability is AVAILABLE")
        if self.availability is Availability.AVAILABLE and self.evidence.sample_count == 0:
            raise ValueError("available numeric requirements require at least one sample")
        if self.availability is Availability.UNKNOWN and (
            self.evidence.confidence_or_percentile is not ConfidenceOrPercentile.COUNTER_ABSENT
        ):
            raise ValueError("UNKNOWN numeric requirements require counter_absent evidence")
        if self.availability is Availability.UNAVAILABLE and (
            self.evidence.confidence_or_percentile
            is not ConfidenceOrPercentile.CAPABILITY_UNAVAILABLE
        ):
            raise ValueError(
                "UNAVAILABLE numeric requirements require capability_unavailable evidence"
            )
        return self


class RecommendationLabel(StrEnum):
    STRICT = "STRICT"
    TRANSACTIONAL = "TRANSACTIONAL"
    EPOCH_FENCED = "EPOCH_FENCED"
    EVENTUAL = "EVENTUAL"
    BLOCK = "BLOCK"
    REJECT = "REJECT"
    THROTTLE = "THROTTLE"
    SPILL = "SPILL"
    RETRY = "RETRY"
    ABORT = "ABORT"
    ROLLBACK = "ROLLBACK"
    FAIL_CLOSED = "FAIL_CLOSED"
    SOFTWARE_MANAGED = "SOFTWARE_MANAGED"


class EnumRequirement(RequirementsModel):
    availability: Availability
    value: RecommendationLabel | None
    evidence: EvidenceReference
    rationale: NonEmpty

    @model_validator(mode="after")
    def value_matches_availability(self) -> Self:
        if (self.availability is Availability.AVAILABLE) != (self.value is not None):
            raise ValueError("enum value must be present exactly when availability is AVAILABLE")
        if self.availability is Availability.AVAILABLE and self.evidence.sample_count == 0:
            raise ValueError("available enum requirements require at least one sample")
        if self.availability is Availability.UNKNOWN and (
            self.evidence.confidence_or_percentile is not ConfidenceOrPercentile.COUNTER_ABSENT
        ):
            raise ValueError("UNKNOWN enum requirements require counter_absent evidence")
        if self.availability is Availability.UNAVAILABLE and (
            self.evidence.confidence_or_percentile
            is not ConfidenceOrPercentile.CAPABILITY_UNAVAILABLE
        ):
            raise ValueError(
                "UNAVAILABLE enum requirements require capability_unavailable evidence"
            )
        return self


class DistributionRequirement(RequirementsModel):
    p50: NumericRequirement
    p95: NumericRequirement
    p99: NumericRequirement
    maximum: NumericRequirement

    @model_validator(mode="after")
    def available_values_are_ordered_and_compatible(self) -> Self:
        metrics = (self.p50, self.p95, self.p99, self.maximum)
        if len({metric.availability for metric in metrics}) != 1:
            raise ValueError("distribution percentiles must have one availability state")
        if len({metric.unit for metric in metrics}) != 1:
            raise ValueError("distribution percentiles must use the same unit")
        available = [metric for metric in metrics if metric.availability is Availability.AVAILABLE]
        if available:
            expected_confidence = (
                ConfidenceOrPercentile.P50,
                ConfidenceOrPercentile.P95,
                ConfidenceOrPercentile.P99,
                ConfidenceOrPercentile.MAXIMUM,
            )
            if tuple(metric.evidence.confidence_or_percentile for metric in metrics) != (
                expected_confidence
            ):
                raise ValueError(
                    "available distribution fields require matching percentile evidence"
                )
            values = [float(metric.value) for metric in metrics if metric.value is not None]
            if values != sorted(values):
                raise ValueError("distribution percentiles must be nondecreasing")
        return self


def _require_numeric_unit(
    value: NumericRequirement, allowed: frozenset[RequirementUnit], field_name: str
) -> None:
    if value.unit not in allowed:
        raise ValueError(f"{field_name} has unsupported unit {value.unit.value}")


def _require_distribution_unit(
    value: DistributionRequirement,
    allowed: frozenset[RequirementUnit],
    field_name: str,
) -> None:
    _require_numeric_unit(value.p50, allowed, field_name)


_COUNT_UNITS = frozenset({RequirementUnit.COUNT})
_BYTE_UNITS = frozenset({RequirementUnit.BYTES})
_RATE_UNITS = frozenset({RequirementUnit.EVENTS_PER_SECOND, RequirementUnit.OPERATIONS_PER_SECOND})
_BANDWIDTH_UNITS = frozenset({RequirementUnit.BYTES_PER_SECOND, RequirementUnit.BITS_PER_SECOND})
_TIME_UNITS = frozenset(
    {
        RequirementUnit.NANOSECONDS,
        RequirementUnit.MICROSECONDS,
        RequirementUnit.MILLISECONDS,
        RequirementUnit.SECONDS,
    }
)


class WorkloadRequirement(RequirementsModel):
    workload_class: WorkloadClass
    evidence: EvidenceReference
    experiment_count: NumericRequirement
    branch_fanout: DistributionRequirement
    prefix_tokens: DistributionRequirement
    suffix_tokens: DistributionRequirement

    @model_validator(mode="after")
    def valid_units(self) -> Self:
        if self.evidence.sample_count == 0:
            raise ValueError("workload requirements require at least one evidence sample")
        _require_numeric_unit(self.experiment_count, _COUNT_UNITS, "experiment_count")
        _require_distribution_unit(self.branch_fanout, _COUNT_UNITS, "branch_fanout")
        token_units = frozenset({RequirementUnit.TOKENS})
        _require_distribution_unit(self.prefix_tokens, token_units, "prefix_tokens")
        _require_distribution_unit(self.suffix_tokens, token_units, "suffix_tokens")
        return self


class StateRequirements(RequirementsModel):
    branch_fanout: DistributionRequirement
    shared_root_bytes: DistributionRequirement
    private_suffix_bytes: DistributionRequirement
    divergence_rate: DistributionRequirement
    branch_lifetime_ms: DistributionRequirement

    @model_validator(mode="after")
    def valid_units(self) -> Self:
        _require_distribution_unit(self.branch_fanout, _COUNT_UNITS, "state.branch_fanout")
        _require_distribution_unit(self.shared_root_bytes, _BYTE_UNITS, "shared_root_bytes")
        _require_distribution_unit(self.private_suffix_bytes, _BYTE_UNITS, "private_suffix_bytes")
        _require_distribution_unit(
            self.divergence_rate, frozenset({RequirementUnit.RATIO}), "divergence_rate"
        )
        _require_distribution_unit(
            self.branch_lifetime_ms,
            frozenset({RequirementUnit.MILLISECONDS}),
            "branch_lifetime_ms",
        )
        return self


class PageSizeRequirement(RequirementsModel):
    state_segment: StateSegment
    recommended_page_size_bytes: NumericRequirement
    physical_amplification: DistributionRequirement
    cow_fault_rate: DistributionRequirement

    @model_validator(mode="after")
    def valid_units(self) -> Self:
        _require_numeric_unit(
            self.recommended_page_size_bytes, _BYTE_UNITS, "recommended_page_size_bytes"
        )
        _require_distribution_unit(
            self.physical_amplification,
            frozenset({RequirementUnit.RATIO}),
            "physical_amplification",
        )
        _require_distribution_unit(
            self.cow_fault_rate,
            frozenset({RequirementUnit.FAULTS_PER_SECOND}),
            "cow_fault_rate",
        )
        return self


class CowRequirements(RequirementsModel):
    recommended_page_sizes: tuple[PageSizeRequirement, ...] = Field(
        min_length=1, max_length=MAX_RECOMMENDATIONS
    )
    fault_rate: DistributionRequirement
    amplification: DistributionRequirement

    @model_validator(mode="after")
    def valid_units(self) -> Self:
        _require_distribution_unit(
            self.fault_rate,
            frozenset({RequirementUnit.FAULTS_PER_SECOND}),
            "cow.fault_rate",
        )
        _require_distribution_unit(
            self.amplification, frozenset({RequirementUnit.RATIO}), "cow.amplification"
        )
        return self


class MetadataRequirements(RequirementsModel):
    operations_per_second: DistributionRequirement
    queue_depth: DistributionRequirement
    working_set_bytes: DistributionRequirement

    @model_validator(mode="after")
    def valid_units(self) -> Self:
        _require_distribution_unit(
            self.operations_per_second, _RATE_UNITS, "metadata.operations_per_second"
        )
        _require_distribution_unit(self.queue_depth, _COUNT_UNITS, "metadata.queue_depth")
        _require_distribution_unit(
            self.working_set_bytes, _BYTE_UNITS, "metadata.working_set_bytes"
        )
        return self


class TransformSequenceRequirement(RequirementsModel):
    operations: tuple[StateOperationType, ...] = Field(min_length=1, max_length=128)
    evidence: EvidenceReference
    frequency: NumericRequirement
    bytes_processed: NumericRequirement
    latency: NumericRequirement
    temporary_memory_bytes: NumericRequirement

    @model_validator(mode="after")
    def valid_units(self) -> Self:
        if self.evidence.sample_count == 0:
            raise ValueError("transform sequences require at least one evidence sample")
        _require_numeric_unit(
            self.frequency,
            frozenset({RequirementUnit.COUNT, *_RATE_UNITS}),
            "transform sequence frequency",
        )
        _require_numeric_unit(self.bytes_processed, _BYTE_UNITS, "transform bytes_processed")
        _require_numeric_unit(self.latency, _TIME_UNITS, "transform latency")
        _require_numeric_unit(
            self.temporary_memory_bytes, _BYTE_UNITS, "transform temporary_memory_bytes"
        )
        return self


class TransformRequirements(RequirementsModel):
    top_sequences: tuple[TransformSequenceRequirement, ...] = Field(max_length=MAX_RECOMMENDATIONS)
    bandwidth_requirement: DistributionRequirement
    temporary_memory: DistributionRequirement

    @model_validator(mode="after")
    def valid_units(self) -> Self:
        _require_distribution_unit(
            self.bandwidth_requirement, _BANDWIDTH_UNITS, "transform.bandwidth_requirement"
        )
        _require_distribution_unit(self.temporary_memory, _BYTE_UNITS, "transform.temporary_memory")
        return self


class NetworkRequirements(RequirementsModel):
    unicast_bytes: DistributionRequirement
    multicast_opportunity_bytes: DistributionRequirement
    fanout: DistributionRequirement
    required_bandwidth: DistributionRequirement

    @model_validator(mode="after")
    def valid_units(self) -> Self:
        _require_distribution_unit(self.unicast_bytes, _BYTE_UNITS, "network.unicast_bytes")
        _require_distribution_unit(
            self.multicast_opportunity_bytes,
            _BYTE_UNITS,
            "network.multicast_opportunity_bytes",
        )
        _require_distribution_unit(self.fanout, _COUNT_UNITS, "network.fanout")
        _require_distribution_unit(
            self.required_bandwidth, _BANDWIDTH_UNITS, "network.required_bandwidth"
        )
        return self


class TransactionRequirements(RequirementsModel):
    commit_rate: DistributionRequirement
    abort_rate: DistributionRequirement
    epoch_checks_per_second: DistributionRequirement

    @model_validator(mode="after")
    def valid_units(self) -> Self:
        _require_distribution_unit(self.commit_rate, _RATE_UNITS, "transactions.commit_rate")
        _require_distribution_unit(self.abort_rate, _RATE_UNITS, "transactions.abort_rate")
        _require_distribution_unit(
            self.epoch_checks_per_second, _RATE_UNITS, "transactions.epoch_checks_per_second"
        )
        return self


class IsaClassification(StrEnum):
    REQUIRED = "REQUIRED"
    HIGH_VALUE = "HIGH_VALUE"
    OPTIONAL = "OPTIONAL"
    SOFTWARE_ONLY = "SOFTWARE_ONLY"
    NOT_JUSTIFIED = "NOT_JUSTIFIED"


class IsaClassificationRequirement(RequirementsModel):
    availability: Availability
    value: IsaClassification | None
    evidence: EvidenceReference
    rationale: NonEmpty

    @model_validator(mode="after")
    def value_matches_availability(self) -> Self:
        if (self.availability is Availability.AVAILABLE) != (self.value is not None):
            raise ValueError("ISA classification must be present exactly when available")
        if self.availability is Availability.AVAILABLE and self.evidence.sample_count == 0:
            raise ValueError("available ISA classifications require at least one sample")
        if self.availability is Availability.UNKNOWN and (
            self.evidence.confidence_or_percentile is not ConfidenceOrPercentile.COUNTER_ABSENT
        ):
            raise ValueError("UNKNOWN ISA classifications require counter_absent evidence")
        if self.availability is Availability.UNAVAILABLE and (
            self.evidence.confidence_or_percentile
            is not ConfidenceOrPercentile.CAPABILITY_UNAVAILABLE
        ):
            raise ValueError(
                "UNAVAILABLE ISA classifications require capability_unavailable evidence"
            )
        return self


class IsaOperationRecommendation(RequirementsModel):
    operation: StateOperationType
    classification: IsaClassificationRequirement
    measured_frequency: NumericRequirement
    size: DistributionRequirement
    latency_target: NumericRequirement
    throughput_target: NumericRequirement
    concurrency: DistributionRequirement
    fanout: DistributionRequirement
    state_types: tuple[StateSegment, ...] = Field(min_length=1, max_length=32)
    consistency: EnumRequirement
    failure_behavior: EnumRequirement
    expected_end_to_end_speedup: NumericRequirement
    dependencies: tuple[StateOperationType, ...] = Field(max_length=64)

    @model_validator(mode="after")
    def valid_units(self) -> Self:
        _require_numeric_unit(self.measured_frequency, _RATE_UNITS, "ISA measured_frequency")
        _require_distribution_unit(self.size, _BYTE_UNITS, "ISA size")
        _require_numeric_unit(self.latency_target, _TIME_UNITS, "ISA latency_target")
        _require_numeric_unit(
            self.throughput_target,
            frozenset({RequirementUnit.OPERATIONS_PER_SECOND}),
            "ISA throughput_target",
        )
        _require_distribution_unit(self.concurrency, _COUNT_UNITS, "ISA concurrency")
        _require_distribution_unit(self.fanout, _COUNT_UNITS, "ISA fanout")
        _require_numeric_unit(
            self.expected_end_to_end_speedup,
            frozenset({RequirementUnit.RATIO}),
            "ISA expected_end_to_end_speedup",
        )
        return self


class MemoryTier(StrEnum):
    LOW_LATENCY_METADATA = "low_latency_metadata"
    HBM = "hbm"
    DDR = "ddr"
    CXL = "cxl"
    HOST_OR_STORAGE = "host_or_storage"


class MemoryCapacityRequirement(RequirementsModel):
    workload_class: WorkloadClass
    branch_count: NumericRequirement
    low_latency_metadata_bytes: NumericRequirement
    hbm_bytes: NumericRequirement
    ddr_bytes: NumericRequirement
    cxl_bytes: NumericRequirement
    host_or_storage_bytes: NumericRequirement

    @model_validator(mode="after")
    def valid_units(self) -> Self:
        _require_numeric_unit(self.branch_count, _COUNT_UNITS, "memory branch_count")
        for field_name in (
            "low_latency_metadata_bytes",
            "hbm_bytes",
            "ddr_bytes",
            "cxl_bytes",
            "host_or_storage_bytes",
        ):
            _require_numeric_unit(getattr(self, field_name), _BYTE_UNITS, f"memory {field_name}")
        return self


class QueueRequirement(RequirementsModel):
    operation: StateOperationType
    minimum_depth: NumericRequirement
    recommended_depth: NumericRequirement
    pathological_maximum: NumericRequirement
    backpressure_policy: EnumRequirement

    @model_validator(mode="after")
    def valid_units(self) -> Self:
        for field_name in ("minimum_depth", "recommended_depth", "pathological_maximum"):
            _require_numeric_unit(getattr(self, field_name), _COUNT_UNITS, f"queue {field_name}")
        return self


class InterfaceKind(StrEnum):
    GPU_TO_BRANCHFABRIC = "gpu_to_branchfabric"
    BRANCHFABRIC_TO_GPU = "branchfabric_to_gpu"
    INTERNAL_MEMORY = "internal_memory"
    NETWORK_INGRESS = "network_ingress"
    NETWORK_EGRESS = "network_egress"
    HOST = "host"
    STORAGE = "storage"


class BandwidthRequirement(RequirementsModel):
    interface: InterfaceKind
    mean: NumericRequirement
    p95: NumericRequirement
    p99: NumericRequirement
    burst_peak: NumericRequirement
    burst_duration: NumericRequirement

    @model_validator(mode="after")
    def valid_units(self) -> Self:
        for field_name in ("mean", "p95", "p99", "burst_peak"):
            _require_numeric_unit(
                getattr(self, field_name), _BANDWIDTH_UNITS, f"bandwidth {field_name}"
            )
        _require_numeric_unit(self.burst_duration, _TIME_UNITS, "bandwidth burst_duration")
        return self


class LatencyRequirement(RequirementsModel):
    operation: StateOperationType
    target_p50: NumericRequirement
    target_p99: NumericRequirement
    maximum_tolerable: NumericRequirement

    @model_validator(mode="after")
    def valid_units(self) -> Self:
        for field_name in ("target_p50", "target_p99", "maximum_tolerable"):
            _require_numeric_unit(getattr(self, field_name), _TIME_UNITS, f"latency {field_name}")
        return self


class RequirementsSections(RequirementsModel):
    workloads: tuple[WorkloadRequirement, ...] = Field(min_length=1, max_length=10_000)
    state: StateRequirements
    cow: CowRequirements
    metadata: MetadataRequirements
    transform: TransformRequirements
    network: NetworkRequirements
    transactions: TransactionRequirements
    recommended_isa: tuple[IsaOperationRecommendation, ...] = Field(max_length=MAX_RECOMMENDATIONS)
    software_only_operations: tuple[IsaOperationRecommendation, ...] = Field(
        max_length=MAX_RECOMMENDATIONS
    )
    not_justified_operations: tuple[IsaOperationRecommendation, ...] = Field(
        max_length=MAX_RECOMMENDATIONS
    )
    unresolved_isa_operations: tuple[IsaOperationRecommendation, ...] = Field(
        default=(), max_length=MAX_RECOMMENDATIONS
    )
    memory_requirements: tuple[MemoryCapacityRequirement, ...] = Field(
        min_length=1, max_length=MAX_RECOMMENDATIONS
    )
    queue_requirements: tuple[QueueRequirement, ...] = Field(
        min_length=1, max_length=MAX_RECOMMENDATIONS
    )
    latency_targets: tuple[LatencyRequirement, ...] = Field(
        min_length=1, max_length=MAX_RECOMMENDATIONS
    )
    bandwidth_targets: tuple[BandwidthRequirement, ...] = Field(
        min_length=1, max_length=MAX_RECOMMENDATIONS
    )

    @model_validator(mode="after")
    def isa_lists_match_classification(self) -> Self:
        recommended = {
            IsaClassification.REQUIRED,
            IsaClassification.HIGH_VALUE,
            IsaClassification.OPTIONAL,
        }
        for item in self.recommended_isa:
            if item.classification.value not in recommended:
                raise ValueError("recommended_isa contains a non-recommended classification")
        for item in self.software_only_operations:
            if item.classification.value is not IsaClassification.SOFTWARE_ONLY:
                raise ValueError("software_only_operations requires SOFTWARE_ONLY classifications")
        for item in self.not_justified_operations:
            if item.classification.value is not IsaClassification.NOT_JUSTIFIED:
                raise ValueError("not_justified_operations requires NOT_JUSTIFIED classifications")
        for item in self.unresolved_isa_operations:
            if item.classification.availability is Availability.AVAILABLE:
                raise ValueError("unresolved_isa_operations requires UNKNOWN or UNAVAILABLE")
        all_operations = [
            item.operation
            for items in (
                self.recommended_isa,
                self.software_only_operations,
                self.not_justified_operations,
                self.unresolved_isa_operations,
            )
            for item in items
        ]
        if len(all_operations) != len(set(all_operations)):
            raise ValueError("an ISA operation may appear in only one recommendation list")
        return self


class RequirementsDraft(RequirementsSections):
    generated_at: NonEmpty
    limitations: tuple[NonEmpty, ...]


class ArtifactBinding(RequirementsModel):
    artifact_reference: NonEmpty
    file_path: NonEmpty
    expected_sha256: Sha256
    trace_format: Literal["jsonl", "parquet", "perfetto", "manifest"] | None = None
    event_count: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def trace_metadata_is_complete(self) -> Self:
        if (self.trace_format is None) != (self.event_count is None):
            raise ValueError("trace format and event count must be supplied together")
        return self


class RequirementsCompilationInput(RequirementsModel):
    trace_id: NonEmpty
    expected_trace_corpus_hash: Sha256
    artifact_bindings: tuple[ArtifactBinding, ...] = Field(min_length=1, max_length=MAX_ARTIFACTS)
    trace_artifact_references: tuple[NonEmpty, ...] = Field(min_length=1, max_length=MAX_ARTIFACTS)
    draft: RequirementsDraft

    @model_validator(mode="after")
    def bindings_are_unique_and_trace_refs_exist(self) -> Self:
        references = [item.artifact_reference for item in self.artifact_bindings]
        if len(references) != len(set(references)):
            raise ValueError("artifact binding references must be unique")
        if len(self.trace_artifact_references) != len(set(self.trace_artifact_references)):
            raise ValueError("trace artifact references must be unique")
        by_reference = {item.artifact_reference: item for item in self.artifact_bindings}
        for reference in self.trace_artifact_references:
            binding = by_reference.get(reference)
            if binding is None:
                raise ValueError(f"trace artifact has no binding: {reference}")
            if binding.trace_format is None:
                raise ValueError(f"trace artifact binding lacks trace metadata: {reference}")
        return self


class VerifiedArtifact(RequirementsModel):
    artifact_reference: NonEmpty
    file_path: NonEmpty
    byte_length: int = Field(ge=0)
    sha256: Sha256
    included_in_trace_corpus: bool


class BranchFabricRequirementsV1(RequirementsSections):
    schema_version: Literal["sloforge.branchfabric.requirements/v1"] = (
        "sloforge.branchfabric.requirements/v1"
    )
    compiler_version: Literal["sloforge-helix-requirements/1"] = "sloforge-helix-requirements/1"
    generated_at: NonEmpty
    trace_corpus_hash: Sha256
    verified_artifacts: tuple[VerifiedArtifact, ...] = Field(min_length=1, max_length=MAX_ARTIFACTS)
    limitations: tuple[NonEmpty, ...]


def _hash_file(path: Path) -> tuple[str, int]:
    if not path.is_file():
        raise ValueError(f"requirements artifact is not a file: {path}")
    digest = hashlib.sha256()
    byte_length = 0
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
            byte_length += len(block)
    return digest.hexdigest(), byte_length


def _walk_evidence(value: object) -> Iterable[EvidenceReference]:
    if isinstance(value, EvidenceReference):
        yield value
        return
    if isinstance(value, BaseModel):
        for field_name in type(value).model_fields:
            yield from _walk_evidence(getattr(value, field_name))
        return
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _walk_evidence(item)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            yield from _walk_evidence(item)


def compile_requirements(compilation: RequirementsCompilationInput) -> BranchFabricRequirementsV1:
    """Verify every evidence artifact and compile a deterministic requirements document."""

    by_reference = {
        binding.artifact_reference: binding for binding in compilation.artifact_bindings
    }
    verified: dict[str, VerifiedArtifact] = {}
    trace_references = set(compilation.trace_artifact_references)
    for binding in compilation.artifact_bindings:
        digest, byte_length = _hash_file(Path(binding.file_path))
        if digest != binding.expected_sha256:
            raise ValueError(
                f"artifact SHA-256 mismatch for {binding.artifact_reference}: "
                f"expected {binding.expected_sha256}, observed {digest}"
            )
        verified[binding.artifact_reference] = VerifiedArtifact(
            artifact_reference=binding.artifact_reference,
            file_path=binding.file_path,
            byte_length=byte_length,
            sha256=digest,
            included_in_trace_corpus=binding.artifact_reference in trace_references,
        )

    evidence_references = tuple(_walk_evidence(compilation.draft))
    if not evidence_references:
        raise ValueError("requirements draft contains no evidence references")
    for evidence in evidence_references:
        evidence_binding = by_reference.get(evidence.artifact_reference)
        if evidence_binding is None:
            raise ValueError(
                f"recommendation references an unbound artifact: {evidence.artifact_reference}"
            )
        if (
            evidence_binding.event_count is not None
            and evidence.sample_count > evidence_binding.event_count
        ):
            raise ValueError(
                f"recommendation sample count exceeds trace event count: "
                f"{evidence.artifact_reference}"
            )
        if evidence.artifact_sha256 != verified[evidence.artifact_reference].sha256:
            raise ValueError(
                f"recommendation artifact SHA-256 mismatch: {evidence.artifact_reference}"
            )

    trace_artifacts: list[TraceArtifactV1] = []
    for reference in compilation.trace_artifact_references:
        binding = by_reference[reference]
        artifact = verified[reference]
        assert binding.trace_format is not None
        assert binding.event_count is not None
        trace_artifacts.append(
            TraceArtifactV1(
                format=binding.trace_format,
                uri=reference,
                byte_length=artifact.byte_length,
                sha256=artifact.sha256,
                event_count=binding.event_count,
            )
        )
    observed_corpus_hash = trace_corpus_hash(compilation.trace_id, tuple(trace_artifacts))
    if observed_corpus_hash != compilation.expected_trace_corpus_hash:
        raise ValueError(
            "trace corpus hash mismatch: "
            f"expected {compilation.expected_trace_corpus_hash}, observed {observed_corpus_hash}"
        )

    payload = compilation.draft.model_dump(mode="python")
    return BranchFabricRequirementsV1.model_validate(
        {
            **payload,
            "schema_version": REQUIREMENTS_SCHEMA_VERSION,
            "compiler_version": REQUIREMENTS_COMPILER_VERSION,
            "trace_corpus_hash": observed_corpus_hash,
            "verified_artifacts": tuple(verified[key] for key in sorted(verified)),
        }
    )


def write_requirements(
    path: Path, requirements: BranchFabricRequirementsV1, *, overwrite: bool = False
) -> None:
    with path.open("wb" if overwrite else "xb") as handle:
        handle.write(canonical_json(requirements))
        handle.write(b"\n")


__all__ = [
    "REQUIREMENTS_COMPILER_VERSION",
    "REQUIREMENTS_SCHEMA_VERSION",
    "ArtifactBinding",
    "Availability",
    "BandwidthRequirement",
    "BranchFabricRequirementsV1",
    "ConfidenceOrPercentile",
    "CowRequirements",
    "DistributionRequirement",
    "EnumRequirement",
    "EvidenceReference",
    "InterfaceKind",
    "IsaClassification",
    "IsaClassificationRequirement",
    "IsaOperationRecommendation",
    "LatencyRequirement",
    "MemoryCapacityRequirement",
    "MetadataRequirements",
    "NetworkRequirements",
    "NumericRequirement",
    "PageSizeRequirement",
    "QueueRequirement",
    "RecommendationLabel",
    "RequirementUnit",
    "RequirementsCompilationInput",
    "RequirementsDraft",
    "StateRequirements",
    "TransactionRequirements",
    "TransformRequirements",
    "TransformSequenceRequirement",
    "VerifiedArtifact",
    "WorkloadRequirement",
    "compile_requirements",
    "write_requirements",
]
