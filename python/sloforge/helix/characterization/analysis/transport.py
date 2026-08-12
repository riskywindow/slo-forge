"""Evidence-preserving transport and multicast analysis.

The analyzer consumes canonical :class:`StateOperationEventV1` records.  It
never treats an event's ``content_hash`` as a state-content hash: that field
protects the event envelope.  Repeated unicast is grouped only when the events
carry the same explicit logical-state identity, epoch, payload description,
and non-empty dependency set.  This deliberately under-counts opportunities
when a producer did not emit enough identity evidence.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Sequence
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from sloforge.helix.characterization.trace.models import (
    MemoryLocation,
    OperationResult,
    StateOperationEventV1,
    StateOperationType,
    TimingMeasurementClass,
    TransportType,
    WorkloadProvenance,
)

MAX_ANALYSIS_EVENTS = 2_000_000
MAX_AGGREGATES = 100_000
MAX_OPPORTUNITIES = 250_000
FANOUT_THRESHOLDS = (2, 4, 8, 16, 32, 64, 128)

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4096)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class AnalysisModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class TraceAnalysisEvidence(AnalysisModel):
    """Raw-artifact and event provenance retained by every analysis report."""

    schema_version: Literal["sloforge.branchfabric.trace-analysis-evidence/v1"]
    source_experiment: NonEmpty
    artifact_reference: NonEmpty
    artifact_sha256: Sha256
    seed: int = Field(ge=0, le=2**64 - 1)
    repetition: int = Field(ge=0, le=1_000_000)
    input_event_count: int = Field(ge=0, le=MAX_ANALYSIS_EVENTS)
    selected_event_count: int = Field(ge=0, le=MAX_ANALYSIS_EVENTS)
    trace_ids: tuple[str, ...] = Field(max_length=100_000)
    workload_provenance: tuple[WorkloadProvenance, ...] = Field(max_length=4)
    timing_measurement_classes: tuple[TimingMeasurementClass, ...] = Field(max_length=4)
    raw_artifact_immutable: Literal[True] = True


class IntegerDistribution(AnalysisModel):
    observation_count: int = Field(gt=0, le=MAX_ANALYSIS_EVENTS)
    minimum: int = Field(ge=0)
    p50: int = Field(ge=0)
    p95: int = Field(ge=0)
    p99: int = Field(ge=0)
    maximum: int = Field(ge=0)
    percentile_method: Literal["nearest-rank"] = "nearest-rank"

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if not self.minimum <= self.p50 <= self.p95 <= self.p99 <= self.maximum:
            raise ValueError("distribution percentiles must be ordered")
        return self


class TransportAggregate(AnalysisModel):
    operation_type: StateOperationType
    transport_type: TransportType
    source_location: MemoryLocation
    destination_location: MemoryLocation
    source_physical_representation: NonEmpty
    destination_physical_representation: NonEmpty
    devices: tuple[str, ...] = Field(max_length=100_000)
    event_count: int = Field(gt=0, le=MAX_ANALYSIS_EVENTS)
    successful_events: int = Field(ge=0)
    failed_events: int = Field(ge=0)
    retry_results: int = Field(ge=0)
    skipped_events: int = Field(ge=0)
    recorded_payload_bytes: int = Field(ge=0)
    logical_delivery_bytes: int = Field(ge=0)
    transfer_time_observation_count: int = Field(ge=0)
    transfer_time_total_ns: int | None = Field(default=None, ge=0)
    observed_throughput_bytes_per_second: float | None = Field(
        default=None, ge=0.0, allow_inf_nan=False
    )
    operation_latency: IntegerDistribution | None = None
    transfer_latency: IntegerDistribution | None = None
    queue_delay: IntegerDistribution | None = None
    cpu_time: IntegerDistribution | None = None
    gpu_time: IntegerDistribution | None = None
    chunk_size: IntegerDistribution | None = None
    concurrency: IntegerDistribution
    fanout: IntegerDistribution
    workload_provenance: tuple[WorkloadProvenance, ...]
    timing_measurement_classes: tuple[TimingMeasurementClass, ...]

    @model_validator(mode="after")
    def consistent_counts(self) -> Self:
        if (
            self.successful_events + self.failed_events + self.retry_results + self.skipped_events
            != self.event_count
        ):
            raise ValueError("transport result counts must equal event count")
        has_time = self.transfer_time_observation_count > 0
        if has_time != (self.transfer_time_total_ns is not None):
            raise ValueError("transfer time total must track the observation count")
        if has_time != (self.observed_throughput_bytes_per_second is not None):
            raise ValueError("throughput must be unknown when transfer time is absent")
        return self


class MulticastEvidenceBasis(StrEnum):
    EXPLICIT_FANOUT = "explicit_event_fanout"
    SHARED_DEPENDENCY = "logical_identity_and_shared_dependency"


class ObservedDeliveryMode(StrEnum):
    MULTICAST = "STATE_MULTICAST"
    REPEATED_UNICAST = "STATE_SEND"


class MulticastOpportunity(AnalysisModel):
    opportunity_id: Sha256
    trace_id: NonEmpty
    logical_state_id: NonEmpty
    state_epoch: int = Field(ge=0)
    evidence_basis: MulticastEvidenceBasis
    observed_delivery_mode: ObservedDeliveryMode
    dependency_event_ids: tuple[str, ...]
    destination_ids: tuple[str, ...]
    destination_representations: tuple[str, ...] = Field(min_length=1)
    fanout: int = Field(ge=2)
    payload_bytes_per_destination: int = Field(ge=0)
    repeated_unicast_source_bytes: int = Field(ge=0)
    exact_payload_layout_match: bool
    exact_payload_multicast_source_bytes: int | None = Field(default=None, ge=0)
    exact_payload_reduction_bytes: int | None = Field(default=None, ge=0)
    tree_root_egress_bytes: int = Field(ge=0)
    tree_root_egress_reduction_bytes: int = Field(ge=0)
    tree_aggregate_link_bytes: None = None
    host_assisted_multicast_source_bytes: int = Field(ge=0)
    host_assisted_multicast_reduction_bytes: int = Field(ge=0)
    network_multicast_source_bytes: int = Field(ge=0)
    network_multicast_reduction_bytes: int = Field(ge=0)
    source_transform_then_multicast_bytes: int | None = Field(default=None, ge=0)
    source_transform_byte_reduction_upper_bound: int | None = Field(default=None, ge=0)
    multicast_then_destination_transform_bytes: int = Field(ge=0)
    destination_transform_byte_reduction_upper_bound: int = Field(ge=0)
    analytical_assumptions: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_strategy_bounds(self) -> Self:
        exact_values_present = self.exact_payload_multicast_source_bytes is not None
        if exact_values_present != (self.exact_payload_reduction_bytes is not None):
            raise ValueError("exact-payload byte fields must be both known or both unknown")
        source_values_present = self.source_transform_then_multicast_bytes is not None
        if source_values_present != (self.source_transform_byte_reduction_upper_bound is not None):
            raise ValueError("source-transform byte fields must be both known or both unknown")
        if self.exact_payload_layout_match != exact_values_present:
            raise ValueError("exact-payload savings require matching destination layouts")
        return self


class FanoutOpportunitySummary(AnalysisModel):
    minimum_fanout: Literal[2, 4, 8, 16, 32, 64, 128]
    opportunity_count: int = Field(ge=0, le=MAX_OPPORTUNITIES)
    repeated_unicast_source_bytes: int = Field(ge=0)
    exact_payload_reduction_bytes: int = Field(ge=0)
    destination_transform_reduction_upper_bound_bytes: int = Field(ge=0)


class MulticastAnalysis(AnalysisModel):
    opportunity_count: int = Field(ge=0, le=MAX_OPPORTUNITIES)
    opportunities: tuple[MulticastOpportunity, ...] = Field(max_length=MAX_OPPORTUNITIES)
    by_minimum_fanout: tuple[FanoutOpportunitySummary, ...] = Field(min_length=7, max_length=7)
    ungrouped_send_event_count: int = Field(ge=0, le=MAX_ANALYSIS_EVENTS)
    event_content_hash_used_as_payload_identity: Literal[False] = False
    topology_dependent_aggregate_link_bytes: None = None
    limitations: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def opportunities_balance(self) -> Self:
        if self.opportunity_count != len(self.opportunities):
            raise ValueError("opportunity count does not match retained opportunities")
        if tuple(item.minimum_fanout for item in self.by_minimum_fanout) != FANOUT_THRESHOLDS:
            raise ValueError("fanout summaries must contain the canonical thresholds")
        return self


class TransportAnalysis(AnalysisModel):
    schema_version: Literal["sloforge.branchfabric.transport-analysis/v1"]
    evidence: TraceAnalysisEvidence
    aggregates: tuple[TransportAggregate, ...] = Field(max_length=MAX_AGGREGATES)
    source_payload_bytes: int = Field(ge=0)
    logical_delivery_bytes: int = Field(ge=0)
    receive_payload_bytes: int = Field(ge=0)
    retry_payload_bytes: int = Field(ge=0)
    multicast: MulticastAnalysis
    unknown_counters: tuple[str, ...]


_TRANSPORT_OPERATIONS = frozenset(
    {
        StateOperationType.STATE_SEND,
        StateOperationType.STATE_RECEIVE,
        StateOperationType.STATE_MULTICAST,
        StateOperationType.STATE_ACK,
        StateOperationType.STATE_RETRY,
    }
)


def _distribution(
    values: Sequence[int], *, include_zero: bool = True
) -> IntegerDistribution | None:
    selected = list(values) if include_zero else [value for value in values if value > 0]
    if not selected:
        return None
    ordered = sorted(selected)

    def rank(probability: float) -> int:
        index = max(0, (len(ordered) * int(probability * 100) + 99) // 100 - 1)
        return ordered[min(index, len(ordered) - 1)]

    return IntegerDistribution(
        observation_count=len(ordered),
        minimum=ordered[0],
        p50=rank(0.50),
        p95=rank(0.95),
        p99=rank(0.99),
        maximum=ordered[-1],
    )


def _provenance(events: Sequence[StateOperationEventV1]) -> tuple[WorkloadProvenance, ...]:
    return tuple(sorted({event.provenance for event in events}, key=lambda value: value.value))


def _timing_classes(
    events: Sequence[StateOperationEventV1],
) -> tuple[TimingMeasurementClass, ...]:
    return tuple(
        sorted({event.timing_measurement_class for event in events}, key=lambda value: value.value)
    )


def _destination_id(event: StateOperationEventV1) -> str | None:
    if event.branch_id is not None:
        suffix = f"@{event.device}" if event.device is not None else ""
        return f"branch:{event.branch_id}{suffix}"
    if event.device is not None:
        return f"device:{event.device}"
    return None


def _opportunity(
    events: Sequence[StateOperationEventV1],
    *,
    fanout: int,
    basis: MulticastEvidenceBasis,
    mode: ObservedDeliveryMode,
    destination_ids: tuple[str, ...],
) -> MulticastOpportunity:
    first = events[0]
    destination_representations = tuple(
        sorted({event.destination_physical_representation for event in events})
    )
    layout_match = len(destination_representations) == 1
    payload_bytes = first.bytes
    repeated_bytes = payload_bytes * fanout
    ideal_bytes = payload_bytes
    reduction = repeated_bytes - ideal_bytes
    identity_document = "|".join(
        (
            first.trace_id,
            first.logical_state_id,
            str(first.state_epoch),
            ",".join(first.dependency_event_ids),
            str(fanout),
            mode.value,
            ",".join(str(event.event_sequence) for event in events),
            ",".join(destination_ids),
            ",".join(destination_representations),
        )
    )
    return MulticastOpportunity(
        opportunity_id=hashlib.sha256(identity_document.encode("utf-8")).hexdigest(),
        trace_id=first.trace_id,
        logical_state_id=first.logical_state_id,
        state_epoch=first.state_epoch,
        evidence_basis=basis,
        observed_delivery_mode=mode,
        dependency_event_ids=tuple(sorted(first.dependency_event_ids)),
        destination_ids=destination_ids,
        destination_representations=destination_representations,
        fanout=fanout,
        payload_bytes_per_destination=payload_bytes,
        repeated_unicast_source_bytes=repeated_bytes,
        exact_payload_layout_match=layout_match,
        exact_payload_multicast_source_bytes=ideal_bytes if layout_match else None,
        exact_payload_reduction_bytes=reduction if layout_match else None,
        tree_root_egress_bytes=ideal_bytes,
        tree_root_egress_reduction_bytes=reduction,
        tree_aggregate_link_bytes=None,
        host_assisted_multicast_source_bytes=ideal_bytes,
        host_assisted_multicast_reduction_bytes=reduction,
        network_multicast_source_bytes=ideal_bytes,
        network_multicast_reduction_bytes=reduction,
        source_transform_then_multicast_bytes=ideal_bytes if layout_match else None,
        source_transform_byte_reduction_upper_bound=reduction if layout_match else None,
        multicast_then_destination_transform_bytes=ideal_bytes,
        destination_transform_byte_reduction_upper_bound=reduction,
        analytical_assumptions=(
            "bytes is treated as payload bytes per destination, as defined by the v1 event",
            "multicast bounds make state movement free after one source copy and exclude protocol headers",
            "tree aggregate-link traffic is unknown without a measured topology and route",
            "transform byte reductions are analytical upper bounds, not measured transform savings",
            "event content_hash is envelope integrity and is not payload identity",
        ),
    )


def _multicast_analysis(events: Sequence[StateOperationEventV1]) -> MulticastAnalysis:
    opportunities: list[MulticastOpportunity] = []
    grouped_sequences: set[tuple[str, int]] = set()
    unicast_groups: dict[tuple[object, ...], list[StateOperationEventV1]] = defaultdict(list)

    for event in events:
        if event.operation_type not in {
            StateOperationType.STATE_SEND,
            StateOperationType.STATE_MULTICAST,
        }:
            continue
        if event.result is not OperationResult.SUCCESS:
            continue
        if event.fanout >= 2:
            opportunities.append(
                _opportunity(
                    (event,),
                    fanout=event.fanout,
                    basis=MulticastEvidenceBasis.EXPLICIT_FANOUT,
                    mode=(
                        ObservedDeliveryMode.MULTICAST
                        if event.operation_type is StateOperationType.STATE_MULTICAST
                        else ObservedDeliveryMode.REPEATED_UNICAST
                    ),
                    destination_ids=(),
                )
            )
            grouped_sequences.add((event.trace_id, event.event_sequence))
            continue
        if event.operation_type is not StateOperationType.STATE_SEND:
            continue
        destination_id = _destination_id(event)
        if destination_id is None or not event.dependency_event_ids:
            continue
        key = (
            event.trace_id,
            event.session_id,
            event.branch_group_id,
            event.logical_state_id,
            event.state_epoch,
            event.bytes,
            event.source_physical_representation,
            event.source_location,
            tuple(sorted(event.dependency_event_ids)),
        )
        unicast_groups[key].append(event)

    for group in unicast_groups.values():
        by_destination: dict[str, StateOperationEventV1] = {}
        for event in group:
            destination_id = _destination_id(event)
            assert destination_id is not None
            by_destination.setdefault(destination_id, event)
        if len(by_destination) < 2:
            continue
        group_events = tuple(by_destination[key] for key in sorted(by_destination))
        opportunities.append(
            _opportunity(
                group_events,
                fanout=len(group_events),
                basis=MulticastEvidenceBasis.SHARED_DEPENDENCY,
                mode=ObservedDeliveryMode.REPEATED_UNICAST,
                destination_ids=tuple(sorted(by_destination)),
            )
        )
        grouped_sequences.update((event.trace_id, event.event_sequence) for event in group_events)

    if len(opportunities) > MAX_OPPORTUNITIES:
        raise ValueError(f"multicast opportunities exceed the {MAX_OPPORTUNITIES} analysis bound")
    opportunities.sort(key=lambda item: (item.trace_id, item.opportunity_id))
    summaries: list[FanoutOpportunitySummary] = []
    for threshold in FANOUT_THRESHOLDS:
        threshold_items = [item for item in opportunities if item.fanout >= threshold]
        summaries.append(
            FanoutOpportunitySummary(
                minimum_fanout=threshold,  # type: ignore[arg-type]
                opportunity_count=len(threshold_items),
                repeated_unicast_source_bytes=sum(
                    item.repeated_unicast_source_bytes for item in threshold_items
                ),
                exact_payload_reduction_bytes=sum(
                    item.exact_payload_reduction_bytes or 0 for item in threshold_items
                ),
                destination_transform_reduction_upper_bound_bytes=sum(
                    item.destination_transform_byte_reduction_upper_bound
                    for item in threshold_items
                ),
            )
        )
    ungrouped = sum(
        1
        for event in events
        if event.operation_type is StateOperationType.STATE_SEND
        and event.result is OperationResult.SUCCESS
        and (event.trace_id, event.event_sequence) not in grouped_sequences
    )
    return MulticastAnalysis(
        opportunity_count=len(opportunities),
        opportunities=tuple(opportunities),
        by_minimum_fanout=tuple(summaries),
        ungrouped_send_event_count=ungrouped,
        event_content_hash_used_as_payload_identity=False,
        topology_dependent_aggregate_link_bytes=None,
        limitations=(
            "repeated sends without a shared non-empty dependency set are not grouped",
            "logical identity and epoch establish content identity; no payload hash exists in v1",
            "mixed destination layouts are excluded from exact-payload multicast savings",
            "source egress bounds do not estimate topology-dependent aggregate link traffic",
        ),
    )


def analyze_transport(
    events: Iterable[StateOperationEventV1],
    *,
    source_experiment: str,
    artifact_reference: str,
    artifact_sha256: str,
    seed: int,
    repetition: int,
) -> TransportAnalysis:
    """Aggregate measured transport counters and conservative multicast bounds."""

    retained: list[StateOperationEventV1] = []
    selected: list[StateOperationEventV1] = []
    for event in events:
        if len(retained) >= MAX_ANALYSIS_EVENTS:
            raise ValueError(f"input exceeds the {MAX_ANALYSIS_EVENTS} event analysis bound")
        retained.append(event)
        if event.operation_type in _TRANSPORT_OPERATIONS:
            selected.append(event)

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
    grouped: dict[
        tuple[StateOperationType, TransportType, MemoryLocation, MemoryLocation, str, str],
        list[StateOperationEventV1],
    ] = defaultdict(list)
    for event in selected:
        grouped[
            (
                event.operation_type,
                event.transport_type,
                event.source_location,
                event.destination_location,
                event.source_physical_representation,
                event.destination_physical_representation,
            )
        ].append(event)
    if len(grouped) > MAX_AGGREGATES:
        raise ValueError(f"transport aggregates exceed the {MAX_AGGREGATES} analysis bound")

    aggregates: list[TransportAggregate] = []
    for key in sorted(
        grouped,
        key=lambda item: (
            item[0].value,
            item[1].value,
            item[2].value,
            item[3].value,
            item[4],
            item[5],
        ),
    ):
        group = grouped[key]
        timed = [event for event in group if event.transfer_time_ns > 0]
        transfer_ns = sum(event.transfer_time_ns for event in timed)
        timed_bytes = sum(event.bytes * event.fanout for event in timed)
        aggregates.append(
            TransportAggregate(
                operation_type=key[0],
                transport_type=key[1],
                source_location=key[2],
                destination_location=key[3],
                source_physical_representation=key[4],
                destination_physical_representation=key[5],
                devices=tuple(
                    sorted({event.device for event in group if event.device is not None})
                ),
                event_count=len(group),
                successful_events=sum(event.result is OperationResult.SUCCESS for event in group),
                failed_events=sum(event.result is OperationResult.FAILURE for event in group),
                retry_results=sum(event.result is OperationResult.RETRY for event in group),
                skipped_events=sum(event.result is OperationResult.SKIPPED for event in group),
                recorded_payload_bytes=sum(event.bytes for event in group),
                logical_delivery_bytes=sum(event.bytes * event.fanout for event in group),
                transfer_time_observation_count=len(timed),
                transfer_time_total_ns=transfer_ns if timed else None,
                observed_throughput_bytes_per_second=(
                    timed_bytes * 1_000_000_000 / transfer_ns if timed else None
                ),
                operation_latency=_distribution(
                    [event.operation_latency_ns for event in group], include_zero=False
                ),
                transfer_latency=_distribution(
                    [event.transfer_time_ns for event in group], include_zero=False
                ),
                queue_delay=_distribution(
                    [event.queue_delay_ns for event in group], include_zero=False
                ),
                cpu_time=_distribution([event.cpu_time_ns for event in group], include_zero=False),
                gpu_time=_distribution([event.gpu_time_ns for event in group], include_zero=False),
                chunk_size=_distribution(
                    [event.chunk_size_bytes for event in group], include_zero=False
                ),
                concurrency=_distribution([event.concurrency for event in group]) or _unreachable(),
                fanout=_distribution([event.fanout for event in group]) or _unreachable(),
                workload_provenance=_provenance(group),
                timing_measurement_classes=_timing_classes(group),
            )
        )

    source_events = [
        event
        for event in selected
        if event.operation_type
        in {StateOperationType.STATE_SEND, StateOperationType.STATE_MULTICAST}
    ]
    unknown: list[str] = []
    if any(event.transfer_time_ns == 0 for event in selected):
        unknown.append("transfer_time_ns for events recorded with zero/N/A")
    if any(event.chunk_size_bytes == 0 for event in selected):
        unknown.append("chunk_size_bytes for unchunked or uninstrumented events")
    if any(event.queue_delay_ns == 0 for event in selected):
        unknown.append("queue_delay_ns for events recorded with zero/N/A")
    if any(event.cpu_time_ns == 0 for event in selected):
        unknown.append("CPU time for transport events recorded with zero/N/A")
    if any(event.gpu_time_ns == 0 for event in selected):
        unknown.append("GPU time for transport events recorded with zero/N/A")
    if selected:
        unknown.append(
            "transport protocol/header bytes are not represented by StateOperationTrace v1"
        )
        unknown.append("topology route and aggregate link bytes are not represented by v1")
    return TransportAnalysis(
        schema_version="sloforge.branchfabric.transport-analysis/v1",
        evidence=evidence,
        aggregates=tuple(aggregates),
        source_payload_bytes=sum(event.bytes for event in source_events),
        logical_delivery_bytes=sum(event.bytes * event.fanout for event in source_events),
        receive_payload_bytes=sum(
            event.bytes
            for event in selected
            if event.operation_type is StateOperationType.STATE_RECEIVE
        ),
        retry_payload_bytes=sum(
            event.bytes
            for event in selected
            if event.operation_type is StateOperationType.STATE_RETRY
            or event.result is OperationResult.RETRY
        ),
        multicast=_multicast_analysis(selected),
        unknown_counters=tuple(unknown),
    )


def _unreachable() -> IntegerDistribution:
    raise AssertionError(
        "non-empty canonical fanout/concurrency values must produce a distribution"
    )


__all__ = [
    "FANOUT_THRESHOLDS",
    "MAX_ANALYSIS_EVENTS",
    "FanoutOpportunitySummary",
    "IntegerDistribution",
    "MulticastAnalysis",
    "MulticastEvidenceBasis",
    "MulticastOpportunity",
    "ObservedDeliveryMode",
    "TraceAnalysisEvidence",
    "TransportAggregate",
    "TransportAnalysis",
    "analyze_transport",
]
