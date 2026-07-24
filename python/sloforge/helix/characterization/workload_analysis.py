"""Evidence-preserving Branch DAG and Helix workload analysis.

All timing values remain inclusive observations.  Explicit timing-span sidecars
may coalesce duplicate instrumentation records, but the analyzer never reports a
sum of nested inclusive spans as exclusive critical-path time.  Token and action
divergence is derived only from verified exact trajectory JSON artifacts.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from sloforge.helix.characterization.trace.models import (
    BranchOperationType,
    BranchWorkloadEventV1,
    MemoryLocation,
    StateOperationEventV1,
    StateOperationType,
    StateSegment,
    TimingMeasurementClass,
    WorkloadProvenance,
)

MAX_EVENTS = 2_000_000
MAX_TRAJECTORIES = 100_000
MAX_TRAJECTORY_BYTES = 64 * 1024 * 1024

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4096)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class WorkloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class ArtifactDescriptor(WorkloadModel):
    source_experiment: NonEmpty
    artifact_reference: NonEmpty
    artifact_sha256: Sha256
    seed: int = Field(ge=0, le=2**64 - 1)
    repetition: int = Field(ge=0, le=1_000_000)


class TrajectoryArtifactInput(ArtifactDescriptor):
    file_path: NonEmpty
    provenance: WorkloadProvenance


class TraceArtifactEvidence(ArtifactDescriptor):
    event_count: int = Field(ge=0, le=MAX_EVENTS)
    trace_ids: tuple[str, ...] = Field(max_length=100_000)
    workload_provenance: tuple[WorkloadProvenance, ...] = Field(max_length=4)
    timing_measurement_classes: tuple[TimingMeasurementClass, ...] = Field(max_length=4)


class TrajectoryEvidence(ArtifactDescriptor):
    branch_id: NonEmpty
    branch_group_id: NonEmpty
    byte_length: int = Field(ge=0, le=MAX_TRAJECTORY_BYTES)
    token_count: int = Field(ge=0)
    action_count: int = Field(ge=0)
    provenance: WorkloadProvenance


class WorkloadEvidence(WorkloadModel):
    branch_trace: TraceArtifactEvidence
    state_trace: TraceArtifactEvidence
    trajectories: tuple[TrajectoryEvidence, ...] = Field(max_length=MAX_TRAJECTORIES)


class ObservedIntegerDistribution(WorkloadModel):
    sample_count: int = Field(gt=0, le=MAX_EVENTS)
    minimum: int = Field(ge=0)
    p50: int = Field(ge=0)
    p95: int = Field(ge=0)
    p99: int = Field(ge=0)
    maximum: int = Field(ge=0)
    percentile_method: Literal["nearest-rank"] = "nearest-rank"

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if not self.minimum <= self.p50 <= self.p95 <= self.p99 <= self.maximum:
            raise ValueError("distribution values must be ordered")
        return self


class BranchOutcome(StrEnum):
    ACTIVE = "ACTIVE"
    PRUNED = "PRUNED"
    ABORTED = "ABORTED"
    SUCCEEDED = "SUCCEEDED"
    UNKNOWN = "UNKNOWN"


class BranchNode(WorkloadModel):
    trace_id: NonEmpty
    branch_group_id: str | None
    branch_id: NonEmpty
    parent_branch_id: str | None
    fork_timestamp_ns: int | None = Field(default=None, ge=0)
    fork_complete_timestamp_ns: int | None = Field(default=None, ge=0)
    first_private_timestamp_ns: int | None = Field(default=None, ge=0)
    fork_to_first_private_ns: int | None = Field(default=None, ge=0)
    terminal_timestamp_ns: int | None = Field(default=None, ge=0)
    lifetime_ns: int | None = Field(default=None, ge=0)
    outcome: BranchOutcome
    pruned: bool
    aborted: bool
    succeeded: bool

    @model_validator(mode="after")
    def timestamps_and_outcome_are_consistent(self) -> Self:
        if (self.fork_timestamp_ns is None) != (self.fork_complete_timestamp_ns is None):
            raise ValueError("fork start and completion must be known together")
        if self.fork_to_first_private_ns is not None and self.first_private_timestamp_ns is None:
            raise ValueError("first-private latency requires its observed timestamp")
        if (
            self.first_private_timestamp_ns is not None
            and self.fork_complete_timestamp_ns is not None
            and self.fork_to_first_private_ns is None
        ):
            raise ValueError("first-private latency is required when the fork is observed")
        if self.lifetime_ns is not None and self.terminal_timestamp_ns is None:
            raise ValueError("branch lifetime requires a terminal timestamp")
        if (
            self.terminal_timestamp_ns is not None
            and self.fork_timestamp_ns is not None
            and self.lifetime_ns is None
        ):
            raise ValueError("branch lifetime is required when fork and terminal are observed")
        flags = sum((self.pruned, self.aborted, self.succeeded))
        if flags > 1:
            raise ValueError("branch terminal outcomes are mutually exclusive")
        expected = {
            BranchOutcome.PRUNED: self.pruned,
            BranchOutcome.ABORTED: self.aborted,
            BranchOutcome.SUCCEEDED: self.succeeded,
        }
        if self.outcome in expected and not expected[self.outcome]:
            raise ValueError("branch outcome does not match its outcome flag")
        return self


class BranchEdge(WorkloadModel):
    trace_id: NonEmpty
    parent_branch_id: NonEmpty
    child_branch_id: NonEmpty


class RefcountPoint(WorkloadModel):
    normalized_timestamp_ns: int = Field(ge=0)
    refcount: int = Field(ge=0)


class ActiveSiblingPoint(WorkloadModel):
    normalized_timestamp_ns: int = Field(ge=0)
    active_siblings: int = Field(ge=0)


class BranchGroupAnalysis(WorkloadModel):
    trace_id: NonEmpty
    branch_group_id: str | None
    parent_branch_id: str | None
    branch_ids: tuple[str, ...] = Field(min_length=1)
    fanout: int = Field(gt=0)
    pruned_count: int = Field(ge=0)
    aborted_count: int = Field(ge=0)
    succeeded_count: int = Field(ge=0)
    active_or_unknown_count: int = Field(ge=0)
    shared_root_creation_timestamp_ns: int | None = Field(default=None, ge=0)
    shared_root_last_reference_timestamp_ns: int | None = Field(default=None, ge=0)
    shared_root_lifetime_ns: int | None = Field(default=None, ge=0)
    shared_root_logical_bytes: int | None = Field(default=None, ge=0)
    peak_refcount: int = Field(ge=0)
    refcount_timeline: tuple[RefcountPoint, ...]
    active_siblings_timeline: tuple[ActiveSiblingPoint, ...]

    @model_validator(mode="after")
    def group_counts_balance(self) -> Self:
        if self.fanout != len(self.branch_ids):
            raise ValueError("group fanout must equal its branch count")
        if (
            self.pruned_count
            + self.aborted_count
            + self.succeeded_count
            + self.active_or_unknown_count
            != self.fanout
        ):
            raise ValueError("branch group outcomes do not balance")
        if self.shared_root_lifetime_ns is not None and (
            self.shared_root_creation_timestamp_ns is None
            or self.shared_root_last_reference_timestamp_ns is None
        ):
            raise ValueError("root lifetime requires creation and last-reference timestamps")
        return self


class BranchDagAnalysis(WorkloadModel):
    nodes: tuple[BranchNode, ...] = Field(max_length=MAX_EVENTS)
    edges: tuple[BranchEdge, ...] = Field(max_length=MAX_EVENTS)
    groups: tuple[BranchGroupAnalysis, ...] = Field(max_length=100_000)
    fanout: ObservedIntegerDistribution | None
    branch_lifetime_ns: ObservedIntegerDistribution | None
    fork_to_first_private_ns: ObservedIntegerDistribution | None

    @model_validator(mode="after")
    def graph_is_acyclic(self) -> Self:
        parents = {
            (edge.trace_id, edge.child_branch_id): (edge.trace_id, edge.parent_branch_id)
            for edge in self.edges
        }
        completed: set[tuple[str, str]] = set()
        for node in self.nodes:
            seen: set[tuple[str, str]] = set()
            current: tuple[str, str] | None = (node.trace_id, node.branch_id)
            while current in parents and current not in completed:
                if current in seen:
                    raise ValueError("branch DAG contains a cycle")
                seen.add(current)
                current = parents[current]
            completed.update(seen)
        return self


class DivergenceCoverage(StrEnum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    UNAVAILABLE = "UNAVAILABLE"


class DivergenceComparison(WorkloadModel):
    branch_group_id: str | None
    coverage: DivergenceCoverage
    expected_branch_ids: tuple[str, ...]
    compared_branch_ids: tuple[str, ...]
    first_divergent_token_index: int | None = Field(default=None, ge=0)
    token_sequences_all_equal: bool | None
    first_divergent_action_index: int | None = Field(default=None, ge=0)
    action_sequences_all_equal: bool | None
    trajectory_artifact_references: tuple[str, ...]


class StateSegmentComposition(WorkloadModel):
    state_segment: StateSegment
    logical_state_count: int = Field(ge=0)
    unique_logical_state_bytes: int = Field(ge=0)
    operation_observed_bytes: int = Field(ge=0)


class StateCompositionAnalysis(WorkloadModel):
    segments: tuple[StateSegmentComposition, ...]
    model_state_bytes: int = Field(ge=0)
    environment_state_bytes: int = Field(ge=0)
    workflow_transaction_and_integrity_bytes: int = Field(ge=0)
    unknown_segment_bytes: int = Field(ge=0)
    byte_semantics: Literal[
        "max bytes per logical_state_id and segment; operation bytes reported separately"
    ]


class OperationQueueAnalysis(WorkloadModel):
    operation_type: StateOperationType
    event_count: int = Field(gt=0)
    concurrency: ObservedIntegerDistribution
    queue_delay_ns: ObservedIntegerDistribution


class WaterfallStage(StrEnum):
    PRODUCTION = "PRODUCTION"
    CAPTURE = "CAPTURE"
    FORK = "FORK"
    BRANCH_READY = "BRANCH_READY"
    ROLLOUT = "ROLLOUT"
    PRUNE = "PRUNE"
    CHECKPOINT = "CHECKPOINT"
    MIGRATE = "MIGRATE"
    RESUME = "RESUME"
    REWARD = "REWARD"
    TRAIN = "TRAIN"
    EVALUATE = "EVALUATE"
    PROMOTE = "PROMOTE"


class WaterfallStageAnalysis(WorkloadModel):
    stage: WaterfallStage
    observed: bool
    event_count: int = Field(ge=0)
    unique_timing_span_count: int = Field(ge=0)
    timing_deduplicated_event_count: int = Field(ge=0)
    inclusive_span_duration_sum_ns: int | None = Field(default=None, ge=0)
    inclusive_wall_union_ns: int | None = Field(default=None, ge=0)
    wall_envelope_ns: int | None = Field(default=None, ge=0)
    cpu_time_ns: int | None = Field(default=None, ge=0)
    gpu_time_ns: int | None = Field(default=None, ge=0)
    network_bytes: int | None = Field(default=None, ge=0)
    storage_bytes: int | None = Field(default=None, ge=0)
    state_operation_bytes: int | None = Field(default=None, ge=0)
    branch_payload_bytes: int | None = Field(default=None, ge=0)
    branch_count: int | None = Field(default=None, ge=0)
    exclusive_critical_path_ns: None = None

    @model_validator(mode="after")
    def missing_stages_have_unknown_metrics(self) -> Self:
        optional = (
            self.inclusive_span_duration_sum_ns,
            self.inclusive_wall_union_ns,
            self.wall_envelope_ns,
            self.cpu_time_ns,
            self.gpu_time_ns,
            self.network_bytes,
            self.storage_bytes,
            self.state_operation_bytes,
            self.branch_payload_bytes,
            self.branch_count,
        )
        if self.observed != any(item is not None for item in optional):
            raise ValueError("waterfall metric availability must match stage observation")
        return self


class HelixWaterfall(WorkloadModel):
    stages: tuple[WaterfallStageAnalysis, ...] = Field(min_length=13, max_length=13)
    end_to_end_wall_envelope_ns: int | None = Field(default=None, ge=0)
    sum_of_stage_inclusive_unions_ns: int = Field(ge=0)
    exclusive_critical_path_ns: None = None
    timing_span_sidecar_entries_used: int = Field(ge=0)
    timing_policy: Literal[
        "inclusive spans; explicit span IDs deduplicated; exclusive critical path unknown"
    ]

    @model_validator(mode="after")
    def all_stages_are_present(self) -> Self:
        if tuple(item.stage for item in self.stages) != tuple(WaterfallStage):
            raise ValueError("waterfall must contain every stage in canonical order")
        return self


class WorkloadAnalysis(WorkloadModel):
    schema_version: Literal["sloforge.branchfabric.workload-analysis/v1"]
    evidence: WorkloadEvidence
    branch_dag: BranchDagAnalysis
    divergence: tuple[DivergenceComparison, ...]
    state_composition: StateCompositionAnalysis
    operation_queues: tuple[OperationQueueAnalysis, ...]
    waterfall: HelixWaterfall
    unknowns: tuple[str, ...]


class _ExactTrajectory(WorkloadModel):
    branch_id: NonEmpty
    branch_group_id: NonEmpty
    tokens: tuple[str, ...]
    actions: tuple[str, ...]
    evidence: TrajectoryEvidence


def canonical_event_reference(event: BranchWorkloadEventV1 | StateOperationEventV1) -> str:
    stream = "branch" if isinstance(event, BranchWorkloadEventV1) else "state"
    return f"{stream}:{event.trace_id}:{event.event_sequence}"


def _distribution(values: Sequence[int]) -> ObservedIntegerDistribution | None:
    if not values:
        return None
    ordered = sorted(values)

    def rank(percent: int) -> int:
        index = max(0, (len(ordered) * percent + 99) // 100 - 1)
        return ordered[min(index, len(ordered) - 1)]

    return ObservedIntegerDistribution(
        sample_count=len(ordered),
        minimum=ordered[0],
        p50=rank(50),
        p95=rank(95),
        p99=rank(99),
        maximum=ordered[-1],
    )


def _provenance(
    events: Sequence[BranchWorkloadEventV1] | Sequence[StateOperationEventV1],
) -> tuple[WorkloadProvenance, ...]:
    return tuple(sorted({event.provenance for event in events}, key=lambda item: item.value))


def _timing_classes(
    events: Sequence[BranchWorkloadEventV1] | Sequence[StateOperationEventV1],
) -> tuple[TimingMeasurementClass, ...]:
    return tuple(
        sorted({event.timing_measurement_class for event in events}, key=lambda item: item.value)
    )


def _load_trajectories(
    inputs: Sequence[TrajectoryArtifactInput],
) -> tuple[_ExactTrajectory, ...]:
    if len(inputs) > MAX_TRAJECTORIES:
        raise ValueError(f"trajectory count exceeds {MAX_TRAJECTORIES}")
    trajectories: list[_ExactTrajectory] = []
    seen_branches: set[str] = set()
    for artifact in inputs:
        path = Path(artifact.file_path)
        if not path.is_file():
            raise ValueError(f"trajectory artifact is not a file: {path}")
        with path.open("rb") as handle:
            payload = handle.read(MAX_TRAJECTORY_BYTES + 1)
        if len(payload) > MAX_TRAJECTORY_BYTES:
            raise ValueError("trajectory artifact exceeds 64 MiB")
        digest = hashlib.sha256(payload).hexdigest()
        if digest != artifact.artifact_sha256:
            raise ValueError(
                f"trajectory SHA-256 mismatch for {artifact.artifact_reference}: "
                f"expected {artifact.artifact_sha256}, observed {digest}"
            )
        document = json.loads(payload)
        if not isinstance(document, dict):
            raise ValueError("trajectory artifact must be a JSON object")
        branch_id = document.get("branch_id")
        branch_group_id = document.get("branch_group_id")
        raw_tokens = document.get("tokens")
        raw_actions = document.get("actions")
        if not isinstance(branch_id, str) or not branch_id:
            raise ValueError("trajectory requires a branch_id")
        if not isinstance(branch_group_id, str) or not branch_group_id:
            raise ValueError("trajectory requires a branch_group_id")
        if branch_id in seen_branches:
            raise ValueError(f"duplicate exact trajectory for branch {branch_id}")
        if not isinstance(raw_tokens, list) or not isinstance(raw_actions, list):
            raise ValueError("trajectory tokens and actions must be arrays")
        tokens: list[str] = []
        for index, item in enumerate(raw_tokens):
            if not isinstance(item, dict) or item.get("token_index") != index:
                raise ValueError("trajectory token indices must be exact and contiguous")
            token = item.get("token")
            if not isinstance(token, str):
                raise ValueError("trajectory token must be a string")
            tokens.append(token)
        actions: list[str] = []
        for index, item in enumerate(raw_actions):
            if not isinstance(item, dict) or item.get("action_index") != index:
                raise ValueError("trajectory action indices must be exact and contiguous")
            action = item.get("action")
            if not isinstance(action, str):
                raise ValueError("trajectory action must be a string")
            actions.append(action)
        seen_branches.add(branch_id)
        evidence = TrajectoryEvidence(
            **artifact.model_dump(mode="python", exclude={"file_path"}),
            branch_id=branch_id,
            branch_group_id=branch_group_id,
            byte_length=len(payload),
            token_count=len(tokens),
            action_count=len(actions),
        )
        trajectories.append(
            _ExactTrajectory(
                branch_id=branch_id,
                branch_group_id=branch_group_id,
                tokens=tuple(tokens),
                actions=tuple(actions),
                evidence=evidence,
            )
        )
    return tuple(sorted(trajectories, key=lambda item: item.branch_id))


def _first_difference(sequences: Sequence[Sequence[str]]) -> tuple[int | None, bool | None]:
    if len(sequences) < 2:
        return None, None
    maximum = max(len(item) for item in sequences)
    for index in range(maximum):
        values = {item[index] if index < len(item) else None for item in sequences}
        if len(values) > 1:
            return index, False
    return None, True


def _timeline(nodes: Sequence[BranchNode]) -> tuple[tuple[RefcountPoint, ...], int]:
    changes: Counter[int] = Counter()
    for node in nodes:
        if node.fork_complete_timestamp_ns is not None:
            changes[node.fork_complete_timestamp_ns] += 1
        if node.fork_complete_timestamp_ns is not None and node.terminal_timestamp_ns is not None:
            changes[node.terminal_timestamp_ns] -= 1
    refcount = 0
    peak = 0
    points: list[RefcountPoint] = []
    for timestamp in sorted(changes):
        refcount += changes[timestamp]
        if refcount < 0:
            raise ValueError("branch terminal precedes its group fork")
        peak = max(peak, refcount)
        points.append(RefcountPoint(normalized_timestamp_ns=timestamp, refcount=refcount))
    return tuple(points), peak


def _analyze_dag(
    branch_events: Sequence[BranchWorkloadEventV1],
    state_events: Sequence[StateOperationEventV1],
) -> BranchDagAnalysis:
    by_branch: dict[tuple[str, str], list[BranchWorkloadEventV1]] = defaultdict(list)
    state_by_branch: dict[tuple[str, str], list[StateOperationEventV1]] = defaultdict(list)
    for branch_event in branch_events:
        if branch_event.branch_id is not None:
            by_branch[(branch_event.trace_id, branch_event.branch_id)].append(branch_event)
    for state_event in state_events:
        if state_event.branch_id is not None:
            state_by_branch[(state_event.trace_id, state_event.branch_id)].append(state_event)
    keys = set(by_branch)
    keys.update(
        (event.trace_id, event.branch_id)
        for event in state_events
        if event.branch_id is not None
        and event.branch_group_id is not None
        and event.operation_type
        in {
            StateOperationType.STATE_FORK,
            StateOperationType.STATE_COW,
            StateOperationType.STATE_APPEND,
            StateOperationType.STATE_WRITE,
            StateOperationType.STATE_FREE,
        }
    )
    nodes: list[BranchNode] = []
    for key in sorted(keys):
        branch = by_branch.get(key, [])
        state = state_by_branch.get(key, [])
        forks = [
            event for event in branch if event.operation_type is BranchOperationType.BRANCH_FORK
        ]
        if len(forks) > 1:
            raise ValueError(f"branch {key[1]} has multiple canonical fork events")
        fork = min(forks, key=lambda event: event.normalized_timestamp_ns) if forks else None
        private_times = [
            event.normalized_timestamp_ns
            for event in branch
            if event.private_suffix or event.operation_type is BranchOperationType.BRANCH_DIVERGENCE
        ]
        private_times.extend(
            event.normalized_timestamp_ns
            for event in state
            if event.operation_type
            in {
                StateOperationType.STATE_COW,
                StateOperationType.STATE_APPEND,
                StateOperationType.STATE_WRITE,
            }
        )
        terminals = [
            event
            for event in branch
            if event.operation_type
            in {
                BranchOperationType.BRANCH_PRUNE,
                BranchOperationType.BRANCH_ABORT,
                BranchOperationType.BRANCH_COMPLETE,
                BranchOperationType.BRANCH_COMMIT,
            }
        ]
        terminal = (
            min(terminals, key=lambda event: event.normalized_timestamp_ns) if terminals else None
        )
        outcome = BranchOutcome.ACTIVE if fork is not None else BranchOutcome.UNKNOWN
        if terminal is not None:
            outcome = {
                BranchOperationType.BRANCH_PRUNE: BranchOutcome.PRUNED,
                BranchOperationType.BRANCH_ABORT: BranchOutcome.ABORTED,
                BranchOperationType.BRANCH_COMPLETE: BranchOutcome.SUCCEEDED,
                BranchOperationType.BRANCH_COMMIT: BranchOutcome.SUCCEEDED,
            }[terminal.operation_type]
        fork_start = fork.normalized_timestamp_ns if fork is not None else None
        fork_complete = fork_start + fork.duration_ns if fork_start is not None and fork else None
        first_private = min(private_times) if private_times else None
        terminal_end = (
            terminal.normalized_timestamp_ns + terminal.duration_ns
            if terminal is not None
            else None
        )
        group_ids = {
            event.branch_group_id for event in branch if event.branch_group_id is not None
        } | {event.branch_group_id for event in state if event.branch_group_id is not None}
        if len(group_ids) > 1:
            raise ValueError(f"branch {key[1]} belongs to multiple branch groups")
        group_id = next(iter(group_ids), None)
        parent_ids = {
            event.parent_branch_id for event in branch if event.parent_branch_id is not None
        }
        if len(parent_ids) > 1:
            raise ValueError(f"branch {key[1]} has conflicting parents")
        parent_id = next(iter(parent_ids), None)
        nodes.append(
            BranchNode(
                trace_id=key[0],
                branch_group_id=group_id,
                branch_id=key[1],
                parent_branch_id=parent_id,
                fork_timestamp_ns=fork_start,
                fork_complete_timestamp_ns=fork_complete,
                first_private_timestamp_ns=first_private,
                fork_to_first_private_ns=(
                    max(0, first_private - fork_complete)
                    if first_private is not None and fork_complete is not None
                    else None
                ),
                terminal_timestamp_ns=terminal_end,
                lifetime_ns=(
                    max(0, terminal_end - fork_start)
                    if terminal_end is not None and fork_start is not None
                    else None
                ),
                outcome=outcome,
                pruned=outcome is BranchOutcome.PRUNED,
                aborted=outcome is BranchOutcome.ABORTED,
                succeeded=outcome is BranchOutcome.SUCCEEDED,
            )
        )

    groups: dict[tuple[str, str | None, str | None], list[BranchNode]] = defaultdict(list)
    for node in nodes:
        groups[(node.trace_id, node.branch_group_id, node.parent_branch_id)].append(node)
    shared_bytes_by_group: dict[tuple[str, str | None], list[int]] = defaultdict(list)
    for event in branch_events:
        if event.shared_root and event.logical_bytes > 0:
            shared_bytes_by_group[(event.trace_id, event.branch_group_id)].append(
                event.logical_bytes
            )
    group_results: list[BranchGroupAnalysis] = []
    for group_key in sorted(groups, key=lambda item: (item[0], item[1] or "", item[2] or "")):
        members = sorted(groups[group_key], key=lambda item: item.branch_id)
        timeline, peak = _timeline(members)
        root_start = min(
            (
                item.fork_complete_timestamp_ns
                for item in members
                if item.fork_complete_timestamp_ns is not None
            ),
            default=None,
        )
        root_end = (
            max(
                item.terminal_timestamp_ns
                for item in members
                if item.terminal_timestamp_ns is not None
            )
            if root_start is not None
            and members
            and all(item.terminal_timestamp_ns is not None for item in members)
            else None
        )
        shared_bytes = shared_bytes_by_group.get((group_key[0], group_key[1]), [])
        group_results.append(
            BranchGroupAnalysis(
                trace_id=group_key[0],
                branch_group_id=group_key[1],
                parent_branch_id=group_key[2],
                branch_ids=tuple(item.branch_id for item in members),
                fanout=len(members),
                pruned_count=sum(item.pruned for item in members),
                aborted_count=sum(item.aborted for item in members),
                succeeded_count=sum(item.succeeded for item in members),
                active_or_unknown_count=sum(
                    item.outcome in {BranchOutcome.ACTIVE, BranchOutcome.UNKNOWN}
                    for item in members
                ),
                shared_root_creation_timestamp_ns=root_start,
                shared_root_last_reference_timestamp_ns=root_end,
                shared_root_lifetime_ns=(
                    max(0, root_end - root_start)
                    if root_start is not None and root_end is not None
                    else None
                ),
                shared_root_logical_bytes=max(shared_bytes) if shared_bytes else None,
                peak_refcount=peak,
                refcount_timeline=timeline,
                active_siblings_timeline=tuple(
                    ActiveSiblingPoint(
                        normalized_timestamp_ns=item.normalized_timestamp_ns,
                        active_siblings=item.refcount,
                    )
                    for item in timeline
                ),
            )
        )
    edges = tuple(
        BranchEdge(
            trace_id=node.trace_id,
            parent_branch_id=node.parent_branch_id,
            child_branch_id=node.branch_id,
        )
        for node in nodes
        if node.parent_branch_id is not None
    )
    return BranchDagAnalysis(
        nodes=tuple(nodes),
        edges=edges,
        groups=tuple(group_results),
        fanout=_distribution([item.fanout for item in group_results]),
        branch_lifetime_ns=_distribution(
            [item.lifetime_ns for item in nodes if item.lifetime_ns is not None]
        ),
        fork_to_first_private_ns=_distribution(
            [
                item.fork_to_first_private_ns
                for item in nodes
                if item.fork_to_first_private_ns is not None
            ]
        ),
    )


def _analyze_divergence(
    groups: Sequence[BranchGroupAnalysis], trajectories: Sequence[_ExactTrajectory]
) -> tuple[DivergenceComparison, ...]:
    by_group: dict[str, list[_ExactTrajectory]] = defaultdict(list)
    for trajectory in trajectories:
        by_group[trajectory.branch_group_id].append(trajectory)
    expected_members = {
        (group.branch_group_id, branch_id) for group in groups for branch_id in group.branch_ids
    }
    unexpected = sorted(
        (trajectory.branch_group_id, trajectory.branch_id)
        for trajectory in trajectories
        if (trajectory.branch_group_id, trajectory.branch_id) not in expected_members
    )
    if unexpected:
        raise ValueError(f"trajectory artifacts do not match traced branches: {unexpected}")
    results: list[DivergenceComparison] = []
    for group in groups:
        provided = sorted(
            by_group.get(group.branch_group_id or "", []), key=lambda item: item.branch_id
        )
        compared = [item for item in provided if item.branch_id in group.branch_ids]
        compared_ids = tuple(item.branch_id for item in compared)
        coverage = DivergenceCoverage.UNAVAILABLE
        if len(compared) >= 2:
            coverage = (
                DivergenceCoverage.COMPLETE
                if set(compared_ids) == set(group.branch_ids)
                else DivergenceCoverage.PARTIAL
            )
        token_index, token_equal = _first_difference([item.tokens for item in compared])
        action_index, action_equal = _first_difference([item.actions for item in compared])
        results.append(
            DivergenceComparison(
                branch_group_id=group.branch_group_id,
                coverage=coverage,
                expected_branch_ids=group.branch_ids,
                compared_branch_ids=compared_ids,
                first_divergent_token_index=token_index,
                token_sequences_all_equal=token_equal,
                first_divergent_action_index=action_index,
                action_sequences_all_equal=action_equal,
                trajectory_artifact_references=tuple(
                    item.evidence.artifact_reference for item in compared
                ),
            )
        )
    return tuple(results)


_MODEL_SEGMENTS = frozenset(
    {
        StateSegment.TOKEN_HISTORY,
        StateSegment.KV,
        StateSegment.RECURRENT,
        StateSegment.SAMPLER,
        StateSegment.GUIDED_DECODING,
        StateSegment.RUNTIME_RECONSTRUCTIBLE,
    }
)
_ENVIRONMENT_SEGMENTS = frozenset(
    {
        StateSegment.ENVIRONMENT,
        StateSegment.FILESYSTEM,
        StateSegment.DATABASE,
        StateSegment.PROCESS_RECONSTRUCTION,
    }
)


def _analyze_composition(events: Sequence[StateOperationEventV1]) -> StateCompositionAnalysis:
    by_segment: dict[StateSegment, list[StateOperationEventV1]] = defaultdict(list)
    for event in events:
        by_segment[event.state_segment].append(event)
    segments: list[StateSegmentComposition] = []
    unique_bytes: dict[StateSegment, int] = {}
    for segment in StateSegment:
        selected = by_segment.get(segment, [])
        by_state: dict[str, int] = {}
        for event in selected:
            by_state[event.logical_state_id] = max(
                by_state.get(event.logical_state_id, 0), event.bytes
            )
        unique_bytes[segment] = sum(by_state.values())
        segments.append(
            StateSegmentComposition(
                state_segment=segment,
                logical_state_count=len(by_state),
                unique_logical_state_bytes=unique_bytes[segment],
                operation_observed_bytes=sum(event.bytes for event in selected),
            )
        )
    return StateCompositionAnalysis(
        segments=tuple(segments),
        model_state_bytes=sum(unique_bytes[item] for item in _MODEL_SEGMENTS),
        environment_state_bytes=sum(unique_bytes[item] for item in _ENVIRONMENT_SEGMENTS),
        workflow_transaction_and_integrity_bytes=(
            unique_bytes[StateSegment.WORKFLOW]
            + unique_bytes[StateSegment.TRANSACTION]
            + unique_bytes[StateSegment.INTEGRITY]
        ),
        unknown_segment_bytes=unique_bytes[StateSegment.UNKNOWN],
        byte_semantics=(
            "max bytes per logical_state_id and segment; operation bytes reported separately"
        ),
    )


def _analyze_queues(
    events: Sequence[StateOperationEventV1],
) -> tuple[OperationQueueAnalysis, ...]:
    grouped: dict[StateOperationType, list[StateOperationEventV1]] = defaultdict(list)
    for event in events:
        grouped[event.operation_type].append(event)
    return tuple(
        OperationQueueAnalysis(
            operation_type=operation,
            event_count=len(group),
            concurrency=_distribution([event.concurrency for event in group]) or _never(),
            queue_delay_ns=_distribution([event.queue_delay_ns for event in group]) or _never(),
        )
        for operation, group in sorted(grouped.items(), key=lambda item: item[0].value)
    )


def _never() -> ObservedIntegerDistribution:
    raise AssertionError("non-empty canonical operation groups always have samples")


_BRANCH_STAGE = {
    BranchOperationType.BRANCH_POINT: WaterfallStage.CAPTURE,
    BranchOperationType.CAPTURE: WaterfallStage.CAPTURE,
    BranchOperationType.BRANCH_FORK: WaterfallStage.FORK,
    BranchOperationType.BRANCH_READY: WaterfallStage.BRANCH_READY,
    BranchOperationType.ROLLOUT: WaterfallStage.ROLLOUT,
    BranchOperationType.BRANCH_PRUNE: WaterfallStage.PRUNE,
    BranchOperationType.BRANCH_ABORT: WaterfallStage.PRUNE,
    BranchOperationType.CHECKPOINT: WaterfallStage.CHECKPOINT,
    BranchOperationType.BRANCH_MIGRATION: WaterfallStage.MIGRATE,
    BranchOperationType.REWARD: WaterfallStage.REWARD,
    BranchOperationType.TRAIN: WaterfallStage.TRAIN,
    BranchOperationType.EVALUATE: WaterfallStage.EVALUATE,
    BranchOperationType.PROMOTE: WaterfallStage.PROMOTE,
}
_STATE_STAGE = {
    StateOperationType.STATE_FORK: WaterfallStage.FORK,
    StateOperationType.STATE_SNAPSHOT: WaterfallStage.CHECKPOINT,
}


def _interval_union(intervals: Sequence[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    ordered = sorted(intervals)
    total = 0
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def _analyze_waterfall(
    branch_events: Sequence[BranchWorkloadEventV1],
    state_events: Sequence[StateOperationEventV1],
    timing_span_ids: Mapping[str, str],
) -> HelixWaterfall:
    if len(timing_span_ids) > MAX_EVENTS:
        raise ValueError(f"timing-span sidecar exceeds the {MAX_EVENTS} entry bound")
    if any(not key or not value for key, value in timing_span_ids.items()):
        raise ValueError("timing-span sidecar keys and values must be non-empty")
    all_references = {canonical_event_reference(event) for event in branch_events} | {
        canonical_event_reference(event) for event in state_events
    }
    unknown_sidecars = set(timing_span_ids) - all_references
    if unknown_sidecars:
        raise ValueError(
            f"timing-span sidecar references unknown events: {sorted(unknown_sidecars)}"
        )
    staged: dict[
        WaterfallStage, list[tuple[BranchWorkloadEventV1 | StateOperationEventV1, str]]
    ] = defaultdict(list)
    for branch_event in branch_events:
        stage = _BRANCH_STAGE.get(branch_event.operation_type)
        if stage is not None:
            reference = canonical_event_reference(branch_event)
            staged[stage].append((branch_event, timing_span_ids.get(reference, reference)))
    for state_event in state_events:
        stage = _STATE_STAGE.get(state_event.operation_type)
        if stage is not None:
            reference = canonical_event_reference(state_event)
            staged[stage].append((state_event, timing_span_ids.get(reference, reference)))
    span_stages: dict[str, set[WaterfallStage]] = defaultdict(set)
    for stage, observations in staged.items():
        for _event, span_id in observations:
            span_stages[span_id].add(stage)
    cross_stage = {
        span_id: sorted(item.value for item in stages)
        for span_id, stages in span_stages.items()
        if len(stages) > 1
    }
    if cross_stage:
        raise ValueError(f"timing spans cross waterfall stage boundaries: {cross_stage}")
    stage_results: list[WaterfallStageAnalysis] = []
    all_intervals: list[tuple[int, int]] = []
    for stage in WaterfallStage:
        observations = staged.get(stage, [])
        if not observations:
            stage_results.append(
                WaterfallStageAnalysis(
                    stage=stage,
                    observed=False,
                    event_count=0,
                    unique_timing_span_count=0,
                    timing_deduplicated_event_count=0,
                    exclusive_critical_path_ns=None,
                )
            )
            continue
        spans: dict[str, list[BranchWorkloadEventV1 | StateOperationEventV1]] = defaultdict(list)
        for observed_event, span_id in observations:
            spans[span_id].append(observed_event)
        span_representatives: list[BranchWorkloadEventV1 | StateOperationEventV1] = []
        for span_id, events in spans.items():
            starts = {event.normalized_timestamp_ns for event in events}
            durations = {event.duration_ns for event in events}
            if len(starts) != 1 or len(durations) != 1:
                raise ValueError(f"timing span {span_id} has inconsistent inclusive timing")
            span_representatives.append(events[0])
        intervals = [
            (
                event.normalized_timestamp_ns,
                event.normalized_timestamp_ns + event.duration_ns,
            )
            for event in span_representatives
        ]
        all_intervals.extend(intervals)
        state_selected = [
            event for event, _span in observations if isinstance(event, StateOperationEventV1)
        ]
        branch_selected = [
            event for event, _span in observations if isinstance(event, BranchWorkloadEventV1)
        ]
        network_bytes = sum(
            event.bytes * event.fanout
            for event in state_selected
            if event.operation_type
            in {StateOperationType.STATE_SEND, StateOperationType.STATE_MULTICAST}
        )
        storage_bytes = sum(
            event.bytes
            for event in state_selected
            if event.destination_location
            in {MemoryLocation.LOCAL_NVME, MemoryLocation.REMOTE_STORAGE}
        )
        cpu_by_span: list[int] = []
        gpu_by_span: list[int] = []
        for span_events in spans.values():
            cpu_values = [
                event.cpu_time_ns
                for event in span_events
                if event.cpu_time_ns is not None and event.cpu_time_ns > 0
            ]
            if cpu_values:
                cpu_by_span.append(max(cpu_values))
            gpu_values: list[int] = []
            for span_event in span_events:
                gpu_value = (
                    span_event.gpu_duration_ns
                    if isinstance(span_event, BranchWorkloadEventV1)
                    else span_event.gpu_time_ns
                )
                if gpu_value is not None and gpu_value > 0:
                    gpu_values.append(gpu_value)
            if gpu_values:
                gpu_by_span.append(max(gpu_values))
        stage_results.append(
            WaterfallStageAnalysis(
                stage=stage,
                observed=True,
                event_count=len(observations),
                unique_timing_span_count=len(spans),
                timing_deduplicated_event_count=len(observations) - len(spans),
                inclusive_span_duration_sum_ns=sum(
                    event.duration_ns for event in span_representatives
                ),
                inclusive_wall_union_ns=_interval_union(intervals),
                wall_envelope_ns=(
                    max(end for _start, end in intervals) - min(start for start, _end in intervals)
                ),
                cpu_time_ns=(sum(cpu_by_span) if len(cpu_by_span) == len(spans) else None),
                gpu_time_ns=(sum(gpu_by_span) if len(gpu_by_span) == len(spans) else None),
                network_bytes=(
                    network_bytes + sum(event.transferred_bytes for event in branch_selected)
                ),
                storage_bytes=storage_bytes,
                state_operation_bytes=sum(event.bytes for event in state_selected),
                branch_payload_bytes=sum(event.logical_bytes for event in branch_selected),
                branch_count=len(
                    {
                        event.branch_id
                        for event, _span in observations
                        if event.branch_id is not None
                    }
                ),
                exclusive_critical_path_ns=None,
            )
        )
    return HelixWaterfall(
        stages=tuple(stage_results),
        end_to_end_wall_envelope_ns=(
            max(end for _start, end in all_intervals) - min(start for start, _end in all_intervals)
            if all_intervals
            else None
        ),
        sum_of_stage_inclusive_unions_ns=sum(
            item.inclusive_wall_union_ns or 0 for item in stage_results
        ),
        exclusive_critical_path_ns=None,
        timing_span_sidecar_entries_used=sum(
            reference in timing_span_ids for reference in all_references
        ),
        timing_policy=(
            "inclusive spans; explicit span IDs deduplicated; exclusive critical path unknown"
        ),
    )


def analyze_workload(
    branch_events: Iterable[BranchWorkloadEventV1],
    state_events: Iterable[StateOperationEventV1],
    *,
    branch_artifact: ArtifactDescriptor,
    state_artifact: ArtifactDescriptor,
    trajectory_artifacts: Sequence[TrajectoryArtifactInput] = (),
    timing_span_ids: Mapping[str, str] | None = None,
) -> WorkloadAnalysis:
    """Derive one conservative workload characterization from canonical traces."""

    branch = tuple(branch_events)
    state = tuple(state_events)
    if len(branch) > MAX_EVENTS or len(state) > MAX_EVENTS:
        raise ValueError(f"canonical input exceeds the {MAX_EVENTS} event bound")
    references = [canonical_event_reference(event) for event in branch] + [
        canonical_event_reference(event) for event in state
    ]
    if len(references) != len(set(references)):
        raise ValueError("canonical workload input contains duplicate event references")
    trajectories = _load_trajectories(trajectory_artifacts)
    dag = _analyze_dag(branch, state)
    divergence = _analyze_divergence(dag.groups, trajectories)
    unknowns = [
        "exclusive critical-path time is unavailable from inclusive event spans",
        "shared-root physical bytes are unavailable in canonical BranchWorkloadTrace v1",
        "shared-root refcounts are derived from fork/terminal boundaries, not metadata counters",
        "state composition is logical-state identity deduplication, not simultaneous residency",
    ]
    if not trajectories:
        unknowns.append("first divergent token/action requires at least two exact trajectories")
    if not timing_span_ids:
        unknowns.append(
            "timing_span_id is absent from canonical v1; no cross-event span deduplication"
        )
    if any(
        event.operation_type
        in {
            StateOperationType.STATE_SEND,
            StateOperationType.STATE_RECEIVE,
            StateOperationType.STATE_MULTICAST,
            StateOperationType.STATE_DELTA,
        }
        for event in state
    ):
        unknowns.append(
            "state transfer events are not assigned to migration without an explicit "
            "BranchWorkloadTrace migration boundary"
        )
    evidence = WorkloadEvidence(
        branch_trace=TraceArtifactEvidence(
            **branch_artifact.model_dump(mode="python"),
            event_count=len(branch),
            trace_ids=tuple(sorted({event.trace_id for event in branch})),
            workload_provenance=_provenance(branch),
            timing_measurement_classes=_timing_classes(branch),
        ),
        state_trace=TraceArtifactEvidence(
            **state_artifact.model_dump(mode="python"),
            event_count=len(state),
            trace_ids=tuple(sorted({event.trace_id for event in state})),
            workload_provenance=_provenance(state),
            timing_measurement_classes=_timing_classes(state),
        ),
        trajectories=tuple(item.evidence for item in trajectories),
    )
    return WorkloadAnalysis(
        schema_version="sloforge.branchfabric.workload-analysis/v1",
        evidence=evidence,
        branch_dag=dag,
        divergence=divergence,
        state_composition=_analyze_composition(state),
        operation_queues=_analyze_queues(state),
        waterfall=_analyze_waterfall(branch, state, timing_span_ids or {}),
        unknowns=tuple(unknowns),
    )


__all__ = [
    "ActiveSiblingPoint",
    "ArtifactDescriptor",
    "BranchDagAnalysis",
    "BranchGroupAnalysis",
    "BranchNode",
    "BranchOutcome",
    "DivergenceComparison",
    "DivergenceCoverage",
    "HelixWaterfall",
    "OperationQueueAnalysis",
    "StateCompositionAnalysis",
    "TraceArtifactEvidence",
    "TrajectoryArtifactInput",
    "TrajectoryEvidence",
    "WaterfallStage",
    "WaterfallStageAnalysis",
    "WorkloadAnalysis",
    "analyze_workload",
    "canonical_event_reference",
]
