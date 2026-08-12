"""Transformation frequency, dependency-chain, and fusion-bound analysis.

Only explicit dependencies are used to join stages.  A canonical event can be
referenced as ``<trace_id>:<event_sequence>`` or by its sealed event hash.  The
analyzer does not guess causality from timestamp adjacency, which makes missing
producer dependency IDs visible rather than turning them into false pipelines.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sloforge.helix.characterization.analysis.transport import (
    MAX_AGGREGATES,
    MAX_ANALYSIS_EVENTS,
    IntegerDistribution,
    TraceAnalysisEvidence,
)
from sloforge.helix.characterization.trace.models import (
    UNSEALED_CONTENT_HASH,
    StateOperationEventV1,
    StateOperationType,
    StateSegment,
    TimingMeasurementClass,
    WorkloadProvenance,
)

MAX_SEQUENCES = 250_000
MAX_TOP_SEQUENCES = 100


class TransformModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class OptionalCounterTotal(TransformModel):
    observation_count: int = Field(ge=0, le=MAX_ANALYSIS_EVENTS)
    total: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def total_matches_count(self) -> Self:
        if (self.observation_count > 0) != (self.total is not None):
            raise ValueError("counter total must be unknown exactly when no observations exist")
        return self


class TransformOperationAggregate(TransformModel):
    operation_type: StateOperationType
    state_segment: StateSegment
    source_physical_representation: str = Field(min_length=1, max_length=4096)
    destination_physical_representation: str = Field(min_length=1, max_length=4096)
    event_count: int = Field(gt=0, le=MAX_ANALYSIS_EVENTS)
    processed_bytes: int = Field(ge=0)
    operation_latency: IntegerDistribution | None
    operation_latency_total_ns: OptionalCounterTotal
    cpu_time_ns: OptionalCounterTotal
    gpu_time_ns: OptionalCounterTotal
    temporary_bytes: None = None
    memory_read_bytes: None = None
    memory_write_bytes: None = None
    workload_provenance: tuple[WorkloadProvenance, ...]
    timing_measurement_classes: tuple[TimingMeasurementClass, ...]


class FusionUpperBound(TransformModel):
    analytical: Literal[True] = True
    occurrence_count: int = Field(gt=0)
    transform_stage_count_per_occurrence: int = Field(gt=0)
    transform_stage_observation_count: int = Field(gt=0)
    observed_materialized_stage_bytes: int = Field(ge=0)
    ideal_single_stream_bytes: int = Field(ge=0)
    maximum_intermediate_bytes_avoided: int = Field(ge=0)
    observed_latency_count: int = Field(ge=0)
    maximum_latency_reduction_ns: int | None = Field(default=None, ge=0)
    temporary_capacity_bytes: None = None
    assumptions: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def latency_bound_is_complete_only(self) -> Self:
        if (self.maximum_latency_reduction_ns is not None) != (
            self.observed_latency_count == self.transform_stage_observation_count
        ):
            raise ValueError("latency fusion bound requires every transform stage latency")
        return self


class TransformStage(TransformModel):
    operation_type: StateOperationType
    source_physical_representation: str = Field(min_length=1, max_length=4096)
    destination_physical_representation: str = Field(min_length=1, max_length=4096)


class TransformSequenceAggregate(TransformModel):
    sequence_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    operations: tuple[StateOperationType, ...] = Field(min_length=1, max_length=128)
    stages: tuple[TransformStage, ...] = Field(min_length=1, max_length=128)
    occurrence_count: int = Field(gt=0, le=MAX_SEQUENCES)
    total_event_count: int = Field(gt=0, le=MAX_ANALYSIS_EVENTS)
    processed_bytes: int = Field(ge=0)
    operation_latency_ns: OptionalCounterTotal
    temporary_bytes: None = None
    fusion_upper_bound: FusionUpperBound
    workload_provenance: tuple[WorkloadProvenance, ...]
    timing_measurement_classes: tuple[TimingMeasurementClass, ...]

    @model_validator(mode="after")
    def operations_match_stages(self) -> Self:
        if self.operations != tuple(stage.operation_type for stage in self.stages):
            raise ValueError("sequence operations must match the physical stage descriptions")
        return self


class TransformSequenceRanking(TransformModel):
    by_frequency: tuple[str, ...] = Field(max_length=MAX_TOP_SEQUENCES)
    by_processed_bytes: tuple[str, ...] = Field(max_length=MAX_TOP_SEQUENCES)
    by_observed_latency: tuple[str, ...] = Field(max_length=MAX_TOP_SEQUENCES)


class DeclaredFusedOperationChain(TransformModel):
    """Producer-declared operations sharing one inseparable measured span."""

    event_reference: str = Field(min_length=1, max_length=4096)
    operations: tuple[StateOperationType, ...] = Field(min_length=2, max_length=128)
    bytes: int = Field(ge=0)
    inclusive_latency_ns: int = Field(ge=0)
    timing_decomposable: Literal[False] = False
    evidence_note: Literal[
        "producer declaration preserved from operation_chain_json; per-operation latency unknown"
    ] = "producer declaration preserved from operation_chain_json; per-operation latency unknown"


class TransformAnalysis(TransformModel):
    schema_version: Literal["sloforge.branchfabric.transform-analysis/v1"]
    evidence: TraceAnalysisEvidence
    transform_event_count: int = Field(ge=0, le=MAX_ANALYSIS_EVENTS)
    operation_aggregates: tuple[TransformOperationAggregate, ...] = Field(max_length=MAX_AGGREGATES)
    sequence_aggregates: tuple[TransformSequenceAggregate, ...] = Field(max_length=MAX_SEQUENCES)
    sequence_ranking: TransformSequenceRanking
    declared_fused_operation_chains: tuple[DeclaredFusedOperationChain, ...] = Field(
        max_length=MAX_SEQUENCES
    )
    unresolved_dependency_count: int = Field(ge=0)
    ambiguous_pipeline_predecessor_count: int = Field(ge=0)
    noncausal_dependency_count: int = Field(ge=0)
    dependency_cycle_count: int = Field(ge=0)
    chain_extraction_method: Literal[
        "explicit <trace_id>:<event_sequence> or sealed-event-hash dependencies"
    ]
    unknown_counters: tuple[str, ...]


TRANSFORM_OPERATIONS = frozenset(
    {
        StateOperationType.STATE_RESHARD,
        StateOperationType.STATE_TRANSPOSE,
        StateOperationType.STATE_REPACK,
        StateOperationType.STATE_QUANTIZE,
        StateOperationType.STATE_DEQUANTIZE,
        StateOperationType.STATE_COMPRESS,
        StateOperationType.STATE_DECOMPRESS,
        StateOperationType.STATE_ENCRYPT,
        StateOperationType.STATE_DECRYPT,
    }
)

PIPELINE_OPERATIONS = frozenset(
    {
        StateOperationType.STATE_READ,
        *TRANSFORM_OPERATIONS,
        StateOperationType.STATE_HASH,
        StateOperationType.STATE_CHECKSUM,
        StateOperationType.STATE_SEND,
        StateOperationType.STATE_MULTICAST,
    }
)


def _reference(event: StateOperationEventV1) -> str:
    return f"{event.trace_id}:{event.event_sequence}"


def _provenance(events: Sequence[StateOperationEventV1]) -> tuple[WorkloadProvenance, ...]:
    return tuple(sorted({event.provenance for event in events}, key=lambda value: value.value))


def _timing_classes(
    events: Sequence[StateOperationEventV1],
) -> tuple[TimingMeasurementClass, ...]:
    return tuple(
        sorted({event.timing_measurement_class for event in events}, key=lambda value: value.value)
    )


def _distribution(values: Sequence[int]) -> IntegerDistribution | None:
    observed = sorted(value for value in values if value > 0)
    if not observed:
        return None

    def nearest_rank(numerator: int) -> int:
        index = max(0, (len(observed) * numerator + 99) // 100 - 1)
        return observed[min(index, len(observed) - 1)]

    return IntegerDistribution(
        observation_count=len(observed),
        minimum=observed[0],
        p50=nearest_rank(50),
        p95=nearest_rank(95),
        p99=nearest_rank(99),
        maximum=observed[-1],
    )


def _optional_total(values: Sequence[int]) -> OptionalCounterTotal:
    observed = [value for value in values if value > 0]
    return OptionalCounterTotal(
        observation_count=len(observed),
        total=sum(observed) if observed else None,
    )


def _same_state(first: StateOperationEventV1, second: StateOperationEventV1) -> bool:
    return (
        first.trace_id == second.trace_id
        and first.session_id == second.session_id
        and first.logical_state_id == second.logical_state_id
        and first.state_epoch == second.state_epoch
        and first.branch_id == second.branch_id
        and first.tenant_id == second.tenant_id
        and first.security_domain == second.security_domain
    )


def _fusion_bound(paths: Sequence[Sequence[StateOperationEventV1]]) -> FusionUpperBound:
    transforms_by_path = [
        [event for event in path if event.operation_type in TRANSFORM_OPERATIONS] for path in paths
    ]
    if not transforms_by_path or not transforms_by_path[0]:
        raise ValueError("fusion bounds require at least one transform stage")
    stage_count = len(transforms_by_path[0])
    if any(len(transforms) != stage_count for transforms in transforms_by_path):
        raise ValueError("aggregated transform sequences must have equal stage counts")
    materialized = sum(event.bytes for transforms in transforms_by_path for event in transforms)
    ideal = sum(max(event.bytes for event in transforms) for transforms in transforms_by_path)
    observed_latency = [
        event.operation_latency_ns
        for transforms in transforms_by_path
        for event in transforms
        if event.operation_latency_ns
    ]
    total_stage_count = stage_count * len(transforms_by_path)
    complete_latency = len(observed_latency) == total_stage_count
    return FusionUpperBound(
        occurrence_count=len(transforms_by_path),
        transform_stage_count_per_occurrence=stage_count,
        transform_stage_observation_count=total_stage_count,
        observed_materialized_stage_bytes=materialized,
        ideal_single_stream_bytes=ideal,
        maximum_intermediate_bytes_avoided=max(0, materialized - ideal),
        observed_latency_count=len(observed_latency),
        maximum_latency_reduction_ns=sum(observed_latency) if complete_latency else None,
        temporary_capacity_bytes=None,
        assumptions=(
            "the byte bound is an analytical single-stream lower bound, not measured fusion",
            "it assumes compatible layouts and no required intermediate materialization",
            "the latency bound makes every observed transform-stage latency free",
            "no end-to-end speedup is implied and transfer or synchronization time is unchanged",
            "StateOperationTrace v1 has no temporary-allocation or memory-traffic counter",
        ),
    )


def _sequence_id(stages: Sequence[TransformStage]) -> str:
    document = "->".join(
        ":".join(
            (
                stage.operation_type.value,
                stage.source_physical_representation,
                stage.destination_physical_representation,
            )
        )
        for stage in stages
    )
    return hashlib.sha256(document.encode("utf-8")).hexdigest()


def _operation_aggregates(
    events: Sequence[StateOperationEventV1],
) -> tuple[TransformOperationAggregate, ...]:
    groups: dict[tuple[StateOperationType, StateSegment, str, str], list[StateOperationEventV1]] = (
        defaultdict(list)
    )
    for event in events:
        if event.operation_type in TRANSFORM_OPERATIONS:
            groups[
                (
                    event.operation_type,
                    event.state_segment,
                    event.source_physical_representation,
                    event.destination_physical_representation,
                )
            ].append(event)
    if len(groups) > MAX_AGGREGATES:
        raise ValueError(f"transform aggregates exceed the {MAX_AGGREGATES} analysis bound")
    aggregates: list[TransformOperationAggregate] = []
    for key in sorted(groups, key=lambda item: (item[0].value, item[1].value, item[2], item[3])):
        group = groups[key]
        aggregates.append(
            TransformOperationAggregate(
                operation_type=key[0],
                state_segment=key[1],
                source_physical_representation=key[2],
                destination_physical_representation=key[3],
                event_count=len(group),
                processed_bytes=sum(event.bytes for event in group),
                operation_latency=_distribution([event.operation_latency_ns for event in group]),
                operation_latency_total_ns=_optional_total(
                    [event.operation_latency_ns for event in group]
                ),
                cpu_time_ns=_optional_total([event.cpu_time_ns for event in group]),
                gpu_time_ns=_optional_total([event.gpu_time_ns for event in group]),
                temporary_bytes=None,
                memory_read_bytes=None,
                memory_write_bytes=None,
                workload_provenance=_provenance(group),
                timing_measurement_classes=_timing_classes(group),
            )
        )
    return tuple(aggregates)


def _extract_paths(
    events: Sequence[StateOperationEventV1],
) -> tuple[list[list[StateOperationEventV1]], int, int, int, int]:
    by_reference = {_reference(event): event for event in events}
    aliases = dict(by_reference)
    for event in events:
        if event.content_hash != UNSEALED_CONTENT_HASH:
            aliases[event.content_hash] = event
    pipeline = [event for event in events if event.operation_type in PIPELINE_OPERATIONS]
    predecessor: dict[str, str] = {}
    unresolved = 0
    ambiguous = 0
    noncausal = 0
    for event in pipeline:
        resolved_pipeline: dict[str, StateOperationEventV1] = {}
        for dependency in event.dependency_event_ids:
            dependency_event = aliases.get(dependency)
            if dependency_event is None:
                unresolved += 1
            elif dependency_event.operation_type in PIPELINE_OPERATIONS:
                resolved_pipeline[_reference(dependency_event)] = dependency_event
        compatible = [
            dependency_event
            for dependency_event in resolved_pipeline.values()
            if _same_state(dependency_event, event)
        ]
        if len(compatible) > 1:
            ambiguous += 1
            continue
        if len(compatible) != 1:
            continue
        dependency_event = compatible[0]
        if (
            dependency_event.normalized_timestamp_ns > event.normalized_timestamp_ns
            or dependency_event.event_sequence >= event.event_sequence
        ):
            noncausal += 1
            continue
        predecessor[_reference(event)] = _reference(dependency_event)

    successor_counts = Counter(predecessor.values())
    event_by_ref = {_reference(event): event for event in pipeline}
    terminal_refs = [reference for reference in event_by_ref if successor_counts[reference] == 0]
    paths: list[list[StateOperationEventV1]] = []
    cycles = 0
    for terminal in sorted(
        terminal_refs,
        key=lambda reference: (
            event_by_ref[reference].trace_id,
            event_by_ref[reference].event_sequence,
        ),
    ):
        reverse_path: list[StateOperationEventV1] = []
        seen: set[str] = set()
        current: str | None = terminal
        while current is not None:
            if current in seen:
                cycles += 1
                reverse_path = []
                break
            seen.add(current)
            reverse_path.append(event_by_ref[current])
            current = predecessor.get(current)
        if reverse_path:
            path = list(reversed(reverse_path))
            if any(event.operation_type in TRANSFORM_OPERATIONS for event in path):
                paths.append(path)
    linked_refs = set(predecessor) | set(predecessor.values())
    if len(linked_refs) > len(event_by_ref):
        raise AssertionError("dependency references escaped the retained pipeline events")
    if len(paths) > MAX_SEQUENCES:
        raise ValueError(f"transform sequences exceed the {MAX_SEQUENCES} analysis bound")
    return paths, unresolved, ambiguous, noncausal, cycles


def _sequence_aggregates(
    paths: Sequence[Sequence[StateOperationEventV1]],
) -> tuple[TransformSequenceAggregate, ...]:
    groups: dict[
        tuple[tuple[StateOperationType, str, str], ...],
        list[Sequence[StateOperationEventV1]],
    ] = defaultdict(list)
    for path in paths:
        groups[
            tuple(
                (
                    event.operation_type,
                    event.source_physical_representation,
                    event.destination_physical_representation,
                )
                for event in path
            )
        ].append(path)
    results: list[TransformSequenceAggregate] = []
    for signature, occurrences in groups.items():
        stages = tuple(
            TransformStage(
                operation_type=operation,
                source_physical_representation=source,
                destination_physical_representation=destination,
            )
            for operation, source, destination in signature
        )
        operations = tuple(stage.operation_type for stage in stages)
        flattened = [event for path in occurrences for event in path]
        observed_latencies = [
            event.operation_latency_ns for event in flattened if event.operation_latency_ns
        ]
        results.append(
            TransformSequenceAggregate(
                sequence_id=_sequence_id(stages),
                operations=operations,
                stages=stages,
                occurrence_count=len(occurrences),
                total_event_count=len(flattened),
                processed_bytes=sum(event.bytes for event in flattened),
                operation_latency_ns=OptionalCounterTotal(
                    observation_count=len(observed_latencies),
                    total=sum(observed_latencies) if observed_latencies else None,
                ),
                temporary_bytes=None,
                fusion_upper_bound=_fusion_bound(occurrences),
                workload_provenance=_provenance(flattened),
                timing_measurement_classes=_timing_classes(flattened),
            )
        )
    return tuple(sorted(results, key=lambda item: item.sequence_id))


def _rank_sequences(
    sequences: Sequence[TransformSequenceAggregate],
) -> TransformSequenceRanking:
    limit = min(MAX_TOP_SEQUENCES, len(sequences))
    by_frequency = sorted(
        sequences,
        key=lambda item: (-item.occurrence_count, -item.processed_bytes, item.sequence_id),
    )[:limit]
    by_bytes = sorted(
        sequences,
        key=lambda item: (-item.processed_bytes, -item.occurrence_count, item.sequence_id),
    )[:limit]
    with_latency = [item for item in sequences if item.operation_latency_ns.total is not None]
    by_latency = sorted(
        with_latency,
        key=lambda item: (
            -(item.operation_latency_ns.total or 0),
            -item.processed_bytes,
            item.sequence_id,
        ),
    )[:limit]
    return TransformSequenceRanking(
        by_frequency=tuple(item.sequence_id for item in by_frequency),
        by_processed_bytes=tuple(item.sequence_id for item in by_bytes),
        by_observed_latency=tuple(item.sequence_id for item in by_latency),
    )


def analyze_transforms(
    events: Iterable[StateOperationEventV1],
    *,
    source_experiment: str,
    artifact_reference: str,
    artifact_sha256: str,
    seed: int,
    repetition: int,
) -> TransformAnalysis:
    """Analyze transform operations and explicitly linked state pipelines."""

    retained: list[StateOperationEventV1] = []
    for event in events:
        if len(retained) >= MAX_ANALYSIS_EVENTS:
            raise ValueError(f"input exceeds the {MAX_ANALYSIS_EVENTS} event analysis bound")
        retained.append(event)
    selected = [event for event in retained if event.operation_type in PIPELINE_OPERATIONS]
    evidence = TraceAnalysisEvidence(
        schema_version="sloforge.branchfabric.trace-analysis-evidence/v1",
        source_experiment=source_experiment,
        artifact_reference=artifact_reference,
        artifact_sha256=artifact_sha256,
        seed=seed,
        repetition=repetition,
        input_event_count=len(retained),
        selected_event_count=len(selected),
        trace_ids=tuple(sorted({event.trace_id for event in retained})),
        workload_provenance=_provenance(retained),
        timing_measurement_classes=_timing_classes(retained),
    )
    paths, unresolved, ambiguous, noncausal, cycles = _extract_paths(retained)
    sequences = _sequence_aggregates(paths)
    transforms = [event for event in retained if event.operation_type in TRANSFORM_OPERATIONS]
    declared_chains: list[DeclaredFusedOperationChain] = []
    for event in retained:
        raw_chain = event.attributes.get("operation_chain_json")
        if not isinstance(raw_chain, str):
            continue
        try:
            decoded = json.loads(raw_chain)
        except json.JSONDecodeError as error:
            raise ValueError("operation_chain_json must contain valid JSON") from error
        if (
            not isinstance(decoded, list)
            or len(decoded) < 2
            or not all(isinstance(item, str) for item in decoded)
        ):
            raise ValueError("operation_chain_json must be a JSON array of operation names")
        declared_chains.append(
            DeclaredFusedOperationChain(
                event_reference=_reference(event),
                operations=tuple(StateOperationType(item) for item in decoded),
                bytes=event.bytes,
                inclusive_latency_ns=event.operation_latency_ns,
            )
        )
    unknown = [
        "temporary allocation bytes are not represented by StateOperationTrace v1",
        "memory read and write counters are not represented by StateOperationTrace v1",
    ]
    if any(event.operation_latency_ns == 0 for event in transforms):
        unknown.append("operation latency for transform events recorded with zero/N/A")
    if any(event.cpu_time_ns == 0 for event in transforms):
        unknown.append("CPU time for transform events recorded with zero/N/A")
    if any(event.gpu_time_ns == 0 for event in transforms):
        unknown.append("GPU time for transform events recorded with zero/N/A")
    return TransformAnalysis(
        schema_version="sloforge.branchfabric.transform-analysis/v1",
        evidence=evidence,
        transform_event_count=len(transforms),
        operation_aggregates=_operation_aggregates(retained),
        sequence_aggregates=sequences,
        sequence_ranking=_rank_sequences(sequences),
        declared_fused_operation_chains=tuple(declared_chains),
        unresolved_dependency_count=unresolved,
        ambiguous_pipeline_predecessor_count=ambiguous,
        noncausal_dependency_count=noncausal,
        dependency_cycle_count=cycles,
        chain_extraction_method=(
            "explicit <trace_id>:<event_sequence> or sealed-event-hash dependencies"
        ),
        unknown_counters=tuple(unknown),
    )


__all__ = [
    "MAX_SEQUENCES",
    "PIPELINE_OPERATIONS",
    "TRANSFORM_OPERATIONS",
    "DeclaredFusedOperationChain",
    "FusionUpperBound",
    "OptionalCounterTotal",
    "TransformAnalysis",
    "TransformOperationAggregate",
    "TransformSequenceAggregate",
    "TransformSequenceRanking",
    "TransformStage",
    "analyze_transforms",
]
