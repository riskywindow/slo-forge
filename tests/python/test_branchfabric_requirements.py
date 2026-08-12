from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from sloforge.helix.characterization.matrix import EvidenceClass, WorkloadClass
from sloforge.helix.characterization.requirements import (
    ArtifactBinding,
    Availability,
    BandwidthRequirement,
    ConfidenceOrPercentile,
    CowRequirements,
    DistributionRequirement,
    EnumRequirement,
    EvidenceReference,
    InterfaceKind,
    IsaClassification,
    IsaClassificationRequirement,
    IsaOperationRecommendation,
    LatencyRequirement,
    MemoryCapacityRequirement,
    MetadataRequirements,
    NetworkRequirements,
    NumericRequirement,
    PageSizeRequirement,
    QueueRequirement,
    RecommendationLabel,
    RequirementsCompilationInput,
    RequirementsDraft,
    RequirementUnit,
    StateRequirements,
    TransactionRequirements,
    TransformRequirements,
    TransformSequenceRequirement,
    WorkloadRequirement,
    compile_requirements,
    write_requirements,
)
from sloforge.helix.characterization.trace.manifest import trace_corpus_hash
from sloforge.helix.characterization.trace.models import (
    StateOperationType,
    StateSegment,
    TraceArtifactV1,
)


def _evidence(
    digest: str,
    *,
    confidence: ConfidenceOrPercentile = ConfidenceOrPercentile.POINT_ESTIMATE,
    samples: int = 1,
) -> EvidenceReference:
    return EvidenceReference(
        source_experiment="requirements-seed-41",
        sample_count=samples,
        evidence_class=EvidenceClass.SYNTHETIC,
        confidence_or_percentile=confidence,
        artifact_reference="trace.jsonl",
        artifact_sha256=digest,
    )


def _number(
    digest: str,
    value: int | float,
    unit: RequirementUnit,
    *,
    confidence: ConfidenceOrPercentile = ConfidenceOrPercentile.POINT_ESTIMATE,
) -> NumericRequirement:
    return NumericRequirement(
        availability=Availability.AVAILABLE,
        value=value,
        unit=unit,
        evidence=_evidence(digest, confidence=confidence),
        rationale="directly derived from the bounded fixture event",
    )


def _unknown(digest: str, unit: RequirementUnit) -> NumericRequirement:
    return NumericRequirement(
        availability=Availability.UNKNOWN,
        value=None,
        unit=unit,
        evidence=_evidence(digest, confidence=ConfidenceOrPercentile.COUNTER_ABSENT),
        rationale="the canonical trace does not contain this counter",
    )


def _unavailable(digest: str, unit: RequirementUnit) -> NumericRequirement:
    return NumericRequirement(
        availability=Availability.UNAVAILABLE,
        value=None,
        unit=unit,
        evidence=_evidence(
            digest,
            confidence=ConfidenceOrPercentile.CAPABILITY_UNAVAILABLE,
            samples=0,
        ),
        rationale="the recorded machine capability manifest has no compatible GPU",
    )


def _distribution(
    digest: str,
    values: tuple[int | float, int | float, int | float, int | float],
    unit: RequirementUnit,
) -> DistributionRequirement:
    return DistributionRequirement(
        p50=_number(digest, values[0], unit, confidence=ConfidenceOrPercentile.P50),
        p95=_number(digest, values[1], unit, confidence=ConfidenceOrPercentile.P95),
        p99=_number(digest, values[2], unit, confidence=ConfidenceOrPercentile.P99),
        maximum=_number(digest, values[3], unit, confidence=ConfidenceOrPercentile.MAXIMUM),
    )


def _unknown_distribution(digest: str, unit: RequirementUnit) -> DistributionRequirement:
    return DistributionRequirement(
        p50=_unknown(digest, unit),
        p95=_unknown(digest, unit),
        p99=_unknown(digest, unit),
        maximum=_unknown(digest, unit),
    )


def _enum(
    digest: str, value: RecommendationLabel = RecommendationLabel.FAIL_CLOSED
) -> EnumRequirement:
    return EnumRequirement(
        availability=Availability.AVAILABLE,
        value=value,
        evidence=_evidence(digest),
        rationale="selected from the measured transaction failure behavior",
    )


def _isa(
    digest: str,
    operation: StateOperationType,
    classification: IsaClassification,
) -> IsaOperationRecommendation:
    return IsaOperationRecommendation(
        operation=operation,
        classification=IsaClassificationRequirement(
            availability=Availability.AVAILABLE,
            value=classification,
            evidence=_evidence(digest),
            rationale="classification follows the measured Amdahl leverage",
        ),
        measured_frequency=_number(digest, 8, RequirementUnit.EVENTS_PER_SECOND),
        size=_distribution(digest, (4096, 8192, 8192, 16384), RequirementUnit.BYTES),
        latency_target=_unknown(digest, RequirementUnit.NANOSECONDS),
        throughput_target=_unknown(digest, RequirementUnit.OPERATIONS_PER_SECOND),
        concurrency=_distribution(digest, (1, 2, 2, 4), RequirementUnit.COUNT),
        fanout=_distribution(digest, (2, 4, 8, 8), RequirementUnit.COUNT),
        state_types=(StateSegment.KV,),
        consistency=_enum(digest, RecommendationLabel.EPOCH_FENCED),
        failure_behavior=_enum(digest),
        expected_end_to_end_speedup=_unknown(digest, RequirementUnit.RATIO),
        dependencies=(),
    )


def _draft(digest: str) -> RequirementsDraft:
    unknown_bytes = _unknown_distribution(digest, RequirementUnit.BYTES)
    unknown_rate = _unknown_distribution(digest, RequirementUnit.EVENTS_PER_SECOND)
    fanout = _distribution(digest, (2, 4, 8, 8), RequirementUnit.COUNT)
    workload = WorkloadRequirement(
        workload_class=WorkloadClass.CODING_AGENT,
        evidence=_evidence(digest),
        experiment_count=_number(digest, 1, RequirementUnit.COUNT),
        branch_fanout=fanout,
        prefix_tokens=_distribution(digest, (1024, 4096, 4096, 8192), RequirementUnit.TOKENS),
        suffix_tokens=_distribution(digest, (16, 64, 256, 256), RequirementUnit.TOKENS),
    )
    return RequirementsDraft(
        generated_at="2026-08-09T00:00:00Z",
        limitations=("CPU fixture only; GPU requirements remain unavailable",),
        workloads=(workload,),
        state=StateRequirements(
            branch_fanout=fanout,
            shared_root_bytes=_distribution(
                digest, (4096, 8192, 16384, 16384), RequirementUnit.BYTES
            ),
            private_suffix_bytes=unknown_bytes,
            divergence_rate=_unknown_distribution(digest, RequirementUnit.RATIO),
            branch_lifetime_ms=_unknown_distribution(digest, RequirementUnit.MILLISECONDS),
        ),
        cow=CowRequirements(
            recommended_page_sizes=(
                PageSizeRequirement(
                    state_segment=StateSegment.KV,
                    recommended_page_size_bytes=_number(digest, 16 * 1024, RequirementUnit.BYTES),
                    physical_amplification=_unknown_distribution(digest, RequirementUnit.RATIO),
                    cow_fault_rate=_unknown_distribution(digest, RequirementUnit.FAULTS_PER_SECOND),
                ),
            ),
            fault_rate=_unknown_distribution(digest, RequirementUnit.FAULTS_PER_SECOND),
            amplification=_unknown_distribution(digest, RequirementUnit.RATIO),
        ),
        metadata=MetadataRequirements(
            operations_per_second=unknown_rate,
            queue_depth=_unknown_distribution(digest, RequirementUnit.COUNT),
            working_set_bytes=unknown_bytes,
        ),
        transform=TransformRequirements(
            top_sequences=(
                TransformSequenceRequirement(
                    operations=(
                        StateOperationType.STATE_REPACK,
                        StateOperationType.STATE_SEND,
                    ),
                    evidence=_evidence(digest),
                    frequency=_number(digest, 1, RequirementUnit.COUNT),
                    bytes_processed=_number(digest, 4096, RequirementUnit.BYTES),
                    latency=_unknown(digest, RequirementUnit.NANOSECONDS),
                    temporary_memory_bytes=_unknown(digest, RequirementUnit.BYTES),
                ),
            ),
            bandwidth_requirement=_unknown_distribution(digest, RequirementUnit.BYTES_PER_SECOND),
            temporary_memory=unknown_bytes,
        ),
        network=NetworkRequirements(
            unicast_bytes=unknown_bytes,
            multicast_opportunity_bytes=unknown_bytes,
            fanout=fanout,
            required_bandwidth=_unknown_distribution(digest, RequirementUnit.BYTES_PER_SECOND),
        ),
        transactions=TransactionRequirements(
            commit_rate=unknown_rate,
            abort_rate=unknown_rate,
            epoch_checks_per_second=unknown_rate,
        ),
        recommended_isa=(
            _isa(digest, StateOperationType.STATE_FORK, IsaClassification.HIGH_VALUE),
        ),
        software_only_operations=(
            _isa(digest, StateOperationType.STATE_HASH, IsaClassification.SOFTWARE_ONLY),
        ),
        not_justified_operations=(
            _isa(digest, StateOperationType.STATE_ENCRYPT, IsaClassification.NOT_JUSTIFIED),
        ),
        memory_requirements=(
            MemoryCapacityRequirement(
                workload_class=WorkloadClass.CODING_AGENT,
                branch_count=_number(digest, 8, RequirementUnit.COUNT),
                low_latency_metadata_bytes=_unknown(digest, RequirementUnit.BYTES),
                hbm_bytes=_unavailable(digest, RequirementUnit.BYTES),
                ddr_bytes=_unknown(digest, RequirementUnit.BYTES),
                cxl_bytes=_unavailable(digest, RequirementUnit.BYTES),
                host_or_storage_bytes=_unknown(digest, RequirementUnit.BYTES),
            ),
        ),
        queue_requirements=(
            QueueRequirement(
                operation=StateOperationType.STATE_FORK,
                minimum_depth=_unknown(digest, RequirementUnit.COUNT),
                recommended_depth=_unknown(digest, RequirementUnit.COUNT),
                pathological_maximum=_unknown(digest, RequirementUnit.COUNT),
                backpressure_policy=_enum(digest, RecommendationLabel.BLOCK),
            ),
        ),
        latency_targets=(
            LatencyRequirement(
                operation=StateOperationType.STATE_FORK,
                target_p50=_unknown(digest, RequirementUnit.NANOSECONDS),
                target_p99=_unknown(digest, RequirementUnit.NANOSECONDS),
                maximum_tolerable=_unknown(digest, RequirementUnit.NANOSECONDS),
            ),
        ),
        bandwidth_targets=(
            BandwidthRequirement(
                interface=InterfaceKind.GPU_TO_BRANCHFABRIC,
                mean=_unavailable(digest, RequirementUnit.BYTES_PER_SECOND),
                p95=_unavailable(digest, RequirementUnit.BYTES_PER_SECOND),
                p99=_unavailable(digest, RequirementUnit.BYTES_PER_SECOND),
                burst_peak=_unavailable(digest, RequirementUnit.BYTES_PER_SECOND),
                burst_duration=_unavailable(digest, RequirementUnit.MILLISECONDS),
            ),
        ),
    )


def _compilation(
    tmp_path: Path, *, evidence_digest: str | None = None
) -> RequirementsCompilationInput:
    trace = tmp_path / "trace.jsonl"
    trace.write_bytes(b'{"event":"fixture"}\n')
    digest = hashlib.sha256(trace.read_bytes()).hexdigest()
    artifact = TraceArtifactV1(
        format="jsonl",
        uri="trace.jsonl",
        byte_length=trace.stat().st_size,
        sha256=digest,
        event_count=1,
    )
    return RequirementsCompilationInput(
        trace_id="trace-requirements-41",
        expected_trace_corpus_hash=trace_corpus_hash("trace-requirements-41", (artifact,)),
        artifact_bindings=(
            ArtifactBinding(
                artifact_reference="trace.jsonl",
                file_path=str(trace),
                expected_sha256=digest,
                trace_format="jsonl",
                event_count=1,
            ),
        ),
        trace_artifact_references=("trace.jsonl",),
        draft=_draft(evidence_digest or digest),
    )


def test_compiler_verifies_artifacts_corpus_and_preserves_unknowns(tmp_path: Path) -> None:
    compilation = _compilation(tmp_path)

    result = compile_requirements(compilation)

    assert result.schema_version == "sloforge.branchfabric.requirements/v1"
    assert result.trace_corpus_hash == compilation.expected_trace_corpus_hash
    assert result.verified_artifacts[0].included_in_trace_corpus is True
    assert result.state.private_suffix_bytes.p50.availability is Availability.UNKNOWN
    assert result.state.private_suffix_bytes.p50.value is None
    assert result.bandwidth_targets[0].mean.availability is Availability.UNAVAILABLE
    assert result.recommended_isa[0].classification.value is IsaClassification.HIGH_VALUE
    output = tmp_path / "requirements.json"
    write_requirements(output, result)
    payload = output.read_text()
    assert result.trace_corpus_hash in payload
    assert "NaN" not in payload


def test_compiler_rejects_artifact_and_trace_corpus_hash_mismatches(tmp_path: Path) -> None:
    compilation = _compilation(tmp_path)
    bad_artifact = compilation.model_copy(
        update={
            "artifact_bindings": (
                compilation.artifact_bindings[0].model_copy(update={"expected_sha256": "b" * 64}),
            )
        }
    )
    with pytest.raises(ValueError, match="artifact SHA-256 mismatch"):
        compile_requirements(bad_artifact)

    bad_corpus = compilation.model_copy(update={"expected_trace_corpus_hash": "c" * 64})
    with pytest.raises(ValueError, match="trace corpus hash mismatch"):
        compile_requirements(bad_corpus)


def test_compiler_rejects_unbound_or_mismatched_recommendation_evidence(
    tmp_path: Path,
) -> None:
    compilation = _compilation(tmp_path, evidence_digest="d" * 64)
    with pytest.raises(ValueError, match="recommendation artifact SHA-256 mismatch"):
        compile_requirements(compilation)

    valid_compilation = _compilation(tmp_path)
    missing_binding = valid_compilation.model_copy(
        update={
            "draft": valid_compilation.draft.model_copy(
                update={
                    "state": valid_compilation.draft.state.model_copy(
                        update={
                            "branch_fanout": valid_compilation.draft.state.branch_fanout.model_copy(
                                update={
                                    "p50": valid_compilation.draft.state.branch_fanout.p50.model_copy(
                                        update={
                                            "evidence": valid_compilation.draft.state.branch_fanout.p50.evidence.model_copy(
                                                update={"artifact_reference": "missing.jsonl"}
                                            )
                                        }
                                    )
                                }
                            )
                        }
                    )
                }
            )
        }
    )
    with pytest.raises(ValueError, match="unbound artifact"):
        compile_requirements(missing_binding)


def test_numeric_requirements_reject_nan_and_implicit_unknown_values() -> None:
    with pytest.raises(ValidationError):
        _number("a" * 64, math.nan, RequirementUnit.RATIO)
    with pytest.raises(ValidationError, match="present exactly"):
        NumericRequirement(
            availability=Availability.UNKNOWN,
            value=0,
            unit=RequirementUnit.COUNT,
            evidence=_evidence("a" * 64, confidence=ConfidenceOrPercentile.COUNTER_ABSENT),
            rationale="zero must not stand in for unknown",
        )
    with pytest.raises(ValidationError, match="counter_absent"):
        NumericRequirement(
            availability=Availability.UNKNOWN,
            value=None,
            unit=RequirementUnit.COUNT,
            evidence=_evidence("a" * 64),
            rationale="unknown must cite absent-counter evidence",
        )
    with pytest.raises(ValidationError, match="at least one sample"):
        NumericRequirement(
            availability=Availability.AVAILABLE,
            value=1,
            unit=RequirementUnit.COUNT,
            evidence=_evidence("a" * 64, samples=0),
            rationale="an available value cannot have zero supporting samples",
        )


def test_schema_rejects_unsupported_claims_and_misclassified_isa_lists() -> None:
    with pytest.raises(ValidationError):
        EnumRequirement.model_validate(
            {
                "availability": "AVAILABLE",
                "value": "MAGIC_POLICY",
                "evidence": _evidence("a" * 64).model_dump(mode="json"),
                "rationale": "unsupported policy",
            }
        )
    draft = _draft("a" * 64)
    with pytest.raises(ValidationError, match="software_only_operations"):
        RequirementsDraft.model_validate(
            {
                **draft.model_dump(mode="python"),
                "software_only_operations": (
                    _isa(
                        "a" * 64,
                        StateOperationType.STATE_HASH,
                        IsaClassification.REQUIRED,
                    ),
                ),
            }
        )


def test_distribution_rejects_nonmonotonic_percentiles() -> None:
    with pytest.raises(ValidationError, match="nondecreasing"):
        _distribution("a" * 64, (8, 4, 16, 32), RequirementUnit.BYTES)


def test_section_units_fail_closed_and_unknown_isa_has_a_distinct_bucket() -> None:
    digest = "a" * 64
    draft = _draft(digest)
    with pytest.raises(ValidationError, match=r"state\.branch_fanout has unsupported unit"):
        StateRequirements(
            branch_fanout=_distribution(digest, (1, 2, 4, 8), RequirementUnit.BYTES),
            shared_root_bytes=draft.state.shared_root_bytes,
            private_suffix_bytes=draft.state.private_suffix_bytes,
            divergence_rate=draft.state.divergence_rate,
            branch_lifetime_ms=draft.state.branch_lifetime_ms,
        )

    unresolved = _isa(
        digest,
        StateOperationType.STATE_COMPRESS,
        IsaClassification.NOT_JUSTIFIED,
    ).model_copy(
        update={
            "classification": IsaClassificationRequirement(
                availability=Availability.UNKNOWN,
                value=None,
                evidence=_evidence(digest, confidence=ConfidenceOrPercentile.COUNTER_ABSENT),
                rationale="no compression events were present in the trace",
            )
        }
    )
    validated = RequirementsDraft.model_validate(
        {
            **draft.model_dump(mode="python"),
            "unresolved_isa_operations": (unresolved,),
        }
    )
    assert validated.unresolved_isa_operations[0].classification.value is None
