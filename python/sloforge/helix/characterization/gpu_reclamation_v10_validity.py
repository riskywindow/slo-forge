"""Fail-closed scientific-validity gates for Experiment 004 v10.

The evaluator is CPU-only.  It consumes immutable request observations and
explicit runtime evidence after a v10 invocation has ended.  It never invents
phase markers from orchestration intent: every timeline marker must name the
raw or derived evidence from which it was obtained.
"""

from __future__ import annotations

import math
import statistics
from enum import StrEnum
from itertools import pairwise
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sloforge.helix.characterization.gpu_reclamation import (
    BranchGroupSemanticEvidence,
    BranchSemanticRecord,
    RuntimeAllocationIdentity,
    RuntimeIncarnation,
)
from sloforge.helix.characterization.gpu_reclamation_serving import (
    IntervalMetrics,
    PhaseInterval,
    ServingMeasurementPlan,
    ServingObservation,
    ServingSLO,
    ServingWorkload,
    evaluate_serving_slo,
    measure_serving_intervals,
)

NS_PER_SECOND = 1_000_000_000
MINIMUM_STABILITY_WINDOW_NS = 5 * NS_PER_SECOND


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class V10Phase(StrEnum):
    CONTROL_STABLE = "CONTROL_STABLE"
    LOAD_SPIKE_BEGIN = "LOAD_SPIKE_BEGIN"
    GPU0_OVERLOAD_CONFIRMED = "GPU0_OVERLOAD_CONFIRMED"
    RECLAIM_TRIGGER = "RECLAIM_TRIGGER"
    ROLLOUT_ADMISSION_STOP = "ROLLOUT_ADMISSION_STOP"
    BRANCH_QUIESCE_BEGIN = "BRANCH_QUIESCE_BEGIN"
    BRANCH_QUIESCE_END = "BRANCH_QUIESCE_END"
    STATE_CAPTURE_BEGIN = "STATE_CAPTURE_BEGIN"
    STATE_CAPTURE_END = "STATE_CAPTURE_END"
    STATE_TRANSFORM_BEGIN = "STATE_TRANSFORM_BEGIN"
    STATE_TRANSFORM_END = "STATE_TRANSFORM_END"
    INTEGRITY_BEGIN = "INTEGRITY_BEGIN"
    INTEGRITY_END = "INTEGRITY_END"
    D2H_BEGIN = "D2H_BEGIN"
    D2H_END = "D2H_END"
    STATE_PUBLISH = "STATE_PUBLISH"
    GPU1_KV_RELEASE_BEGIN = "GPU1_KV_RELEASE_BEGIN"
    GPU1_KV_RELEASE_END = "GPU1_KV_RELEASE_END"
    GPU1_HBM_RECLAIM_CONFIRMED = "GPU1_HBM_RECLAIM_CONFIRMED"
    GPU1_SERVING_ENABLE = "GPU1_SERVING_ENABLE"
    GPU1_FIRST_USEFUL_SERVING_REQUEST = "GPU1_FIRST_USEFUL_SERVING_REQUEST"
    TWO_GPU_SERVICE_STABLE = "TWO_GPU_SERVICE_STABLE"
    SERVING_QUEUE_DRAIN_BEGIN = "SERVING_QUEUE_DRAIN_BEGIN"
    SERVING_QUEUE_DRAIN_END = "SERVING_QUEUE_DRAIN_END"
    SERVING_SLO_RECOVERY_BEGIN = "SERVING_SLO_RECOVERY_BEGIN"
    SERVING_SLO_RESTORED = "SERVING_SLO_RESTORED"
    SERVING_SLO_STABILITY_BEGIN = "SERVING_SLO_STABILITY_BEGIN"
    SERVING_SLO_STABILITY_END = "SERVING_SLO_STABILITY_END"
    RESTORE_ELIGIBLE = "RESTORE_ELIGIBLE"
    RESTORE_TRIGGER = "RESTORE_TRIGGER"
    H2D_BEGIN = "H2D_BEGIN"
    H2D_END = "H2D_END"
    STATE_IMPORT_BEGIN = "STATE_IMPORT_BEGIN"
    STATE_IMPORT_END = "STATE_IMPORT_END"
    STATE_VALIDATE_BEGIN = "STATE_VALIDATE_BEGIN"
    STATE_VALIDATE_END = "STATE_VALIDATE_END"
    BRANCH_RESUME_BEGIN = "BRANCH_RESUME_BEGIN"
    FIRST_RESUMED_TOKEN = "FIRST_RESUMED_TOKEN"
    ALL_BRANCHES_RESUMED = "ALL_BRANCHES_RESUMED"
    ROLLOUT_CONTINUATION_COMPLETE = "ROLLOUT_CONTINUATION_COMPLETE"


REQUIRED_V10_TIMELINE = tuple(V10Phase)

# vLLM 0.23.0 restoration must validate the first imported page subset before
# publishing its prefix for allocation reuse by the remaining branches.  The
# aggregate validation interval is therefore nested inside the import interval;
# placing STATE_IMPORT_END before STATE_VALIDATE_BEGIN would misrepresent the
# real transaction (or require an unnecessary second full validation pass).
# This is the measured causal order of the existing naive path, not enum/list
# presentation order.  Capture copies the canonical payload D2H before host
# integrity construction. During recovery the negative queue trend starts
# when GPU1 first becomes useful; aggregate stability is established only
# after the drain and SLO-stability windows. Restore validation is nested
# inside import because the vLLM adapter validates each disjoint subset before
# publishing it.
CAUSAL_V10_TIMELINE = (
    V10Phase.CONTROL_STABLE,
    V10Phase.LOAD_SPIKE_BEGIN,
    V10Phase.GPU0_OVERLOAD_CONFIRMED,
    V10Phase.RECLAIM_TRIGGER,
    V10Phase.ROLLOUT_ADMISSION_STOP,
    V10Phase.BRANCH_QUIESCE_BEGIN,
    V10Phase.BRANCH_QUIESCE_END,
    V10Phase.STATE_CAPTURE_BEGIN,
    V10Phase.STATE_CAPTURE_END,
    V10Phase.STATE_TRANSFORM_BEGIN,
    V10Phase.STATE_TRANSFORM_END,
    V10Phase.D2H_BEGIN,
    V10Phase.D2H_END,
    V10Phase.INTEGRITY_BEGIN,
    V10Phase.INTEGRITY_END,
    V10Phase.STATE_PUBLISH,
    V10Phase.GPU1_KV_RELEASE_BEGIN,
    V10Phase.GPU1_KV_RELEASE_END,
    V10Phase.GPU1_HBM_RECLAIM_CONFIRMED,
    V10Phase.GPU1_SERVING_ENABLE,
    V10Phase.GPU1_FIRST_USEFUL_SERVING_REQUEST,
    V10Phase.SERVING_QUEUE_DRAIN_BEGIN,
    V10Phase.SERVING_SLO_RECOVERY_BEGIN,
    V10Phase.SERVING_QUEUE_DRAIN_END,
    V10Phase.SERVING_SLO_RESTORED,
    V10Phase.SERVING_SLO_STABILITY_BEGIN,
    V10Phase.SERVING_SLO_STABILITY_END,
    V10Phase.TWO_GPU_SERVICE_STABLE,
    V10Phase.RESTORE_ELIGIBLE,
    V10Phase.RESTORE_TRIGGER,
    V10Phase.H2D_BEGIN,
    V10Phase.H2D_END,
    V10Phase.STATE_IMPORT_BEGIN,
    V10Phase.STATE_VALIDATE_BEGIN,
    V10Phase.STATE_VALIDATE_END,
    V10Phase.STATE_IMPORT_END,
    V10Phase.BRANCH_RESUME_BEGIN,
    V10Phase.FIRST_RESUMED_TOKEN,
    V10Phase.ALL_BRANCHES_RESUMED,
    V10Phase.ROLLOUT_CONTINUATION_COMPLETE,
)


class TimelineEvidenceKind(StrEnum):
    RAW_REQUEST = "raw-request"
    RAW_RUNTIME_STATE = "raw-runtime-state"
    RAW_CUDA_EVENT = "raw-cuda-event"
    RAW_MEMORY_STATE = "raw-memory-state"
    DERIVED_METRIC = "derived-metric"


class V10TimelineEvent(_StrictModel):
    phase: V10Phase
    monotonic_timestamp_ns: int = Field(ge=0)
    evidence_kind: TimelineEvidenceKind
    evidence_reference: str = Field(min_length=1, max_length=512)


class V10Timeline(_StrictModel):
    schema_version: Literal["sloforge.branchfabric.experiment-004-v10-timeline/v1"] = (
        "sloforge.branchfabric.experiment-004-v10-timeline/v1"
    )
    events: tuple[V10TimelineEvent, ...]

    @model_validator(mode="after")
    def exact_complete_causal_order(self) -> Self:
        phases = tuple(event.phase for event in self.events)
        if len(phases) != len(REQUIRED_V10_TIMELINE) or set(phases) != set(REQUIRED_V10_TIMELINE):
            raise ValueError("v10 timeline must contain every required event exactly once")
        timestamps = {event.phase: event.monotonic_timestamp_ns for event in self.events}
        if any(
            timestamps[right] < timestamps[left] for left, right in pairwise(CAUSAL_V10_TIMELINE)
        ):
            raise ValueError("v10 timeline violates required causal event order")
        return self

    def timestamp(self, phase: V10Phase) -> int:
        return next(event.monotonic_timestamp_ns for event in self.events if event.phase is phase)


class V10ValidityConfig(_StrictModel):
    schema_version: Literal["sloforge.branchfabric.experiment-004-v10-validity-config/v1"] = (
        "sloforge.branchfabric.experiment-004-v10-validity-config/v1"
    )
    control_offered_rate_per_second: float = Field(gt=0.0, allow_inf_nan=False)
    spike_offered_rate_per_second: float = Field(gt=0.0, allow_inf_nan=False)
    restore_offered_rate_per_second: float = Field(gt=0.0, allow_inf_nan=False)
    maximum_p95_ttft_ns: int = Field(default=2 * NS_PER_SECOND, gt=0)
    recovery_queue_depth_threshold: int = Field(ge=0)
    evaluation_window_ns: int = Field(default=250_000_000, gt=0)
    stability_window_ns: int = Field(default=MINIMUM_STABILITY_WINDOW_NS, ge=0)
    arrival_rate_tolerance_fraction: float = Field(
        default=0.05, ge=0.0, le=0.25, allow_inf_nan=False
    )

    @model_validator(mode="after")
    def scientifically_bounded(self) -> Self:
        if self.stability_window_ns < MINIMUM_STABILITY_WINDOW_NS:
            raise ValueError("v10 serving stability window must be at least five seconds")
        if self.stability_window_ns % self.evaluation_window_ns:
            raise ValueError("stability window must contain an integer number of evaluations")
        return self


class QueueTrendEvidence(_StrictModel):
    interval: PhaseInterval
    sample_interval_ns: int = Field(gt=0)
    sample_count: int = Field(ge=3)
    initial_depth: int = Field(ge=0)
    final_depth: int = Field(ge=0)
    first_half_mean_depth: float = Field(ge=0.0, allow_inf_nan=False)
    second_half_mean_depth: float = Field(ge=0.0, allow_inf_nan=False)
    slope_requests_per_second: float = Field(allow_inf_nan=False)
    direction: Literal["positive", "negative", "flat"]
    sustained: bool


class GPU0OverloadEvidence(_StrictModel):
    control: IntervalMetrics
    overload: IntervalMetrics
    control_rate_matches_config: bool
    overload_rate_matches_config: bool
    control_p95_ttft_pass: bool
    ttft_exceeded: bool
    queue_positive_trend: bool
    offered_rate_exceeds_completed_rate: bool
    overload_condition_count: int = Field(ge=0, le=3)
    queue_trend: QueueTrendEvidence
    passed: bool

    @model_validator(mode="after")
    def consistent(self) -> Self:
        count = sum(
            (
                self.ttft_exceeded,
                self.queue_positive_trend,
                self.offered_rate_exceeds_completed_rate,
            )
        )
        expected = (
            self.control_rate_matches_config
            and self.overload_rate_matches_config
            and self.control_p95_ttft_pass
            and self.control.completed_requests > 0
            and count >= 1
        )
        if self.overload_condition_count != count or self.passed != expected:
            raise ValueError("GPU0 overload evidence is internally inconsistent")
        return self


class TwoGpuCapacityEvidence(_StrictModel):
    interval: PhaseInterval
    offered_requests: int = Field(ge=0)
    completed_requests: int = Field(ge=0)
    offered_rate_per_second: float = Field(ge=0.0, allow_inf_nan=False)
    completed_rate_per_second: float = Field(ge=0.0, allow_inf_nan=False)
    gpu0_completions: int = Field(ge=0)
    gpu1_completions: int = Field(ge=0)
    offered_rate_matches_config: bool
    passed: bool

    @model_validator(mode="after")
    def consistent(self) -> Self:
        expected = (
            self.offered_rate_matches_config
            and self.completed_rate_per_second > self.offered_rate_per_second
            and self.gpu0_completions > 0
            and self.gpu1_completions > 0
        )
        if self.passed != expected:
            raise ValueError("two-GPU excess-capacity evidence is internally inconsistent")
        return self


class QueueDrainEvidence(_StrictModel):
    trend: QueueTrendEvidence
    passed: bool

    @model_validator(mode="after")
    def consistent(self) -> Self:
        expected = (
            self.trend.direction == "negative"
            and self.trend.sustained
            and self.trend.final_depth < self.trend.initial_depth
        )
        if self.passed != expected:
            raise ValueError("queue-drain evidence is internally inconsistent")
        return self


class SLOStabilityEvidence(_StrictModel):
    interval: PhaseInterval
    duration_ns: int = Field(ge=MINIMUM_STABILITY_WINDOW_NS)
    evaluation_window_ns: int = Field(gt=0)
    evaluation_count: int = Field(gt=0)
    all_windows_have_ttft_samples: bool
    all_windows_p95_ttft_pass: bool
    maximum_queue_depth: int = Field(ge=0)
    queue_depth_threshold: int = Field(ge=0)
    restored_marker_precedes_stability: bool
    restore_trigger_follows_stability: bool
    passed: bool

    @model_validator(mode="after")
    def consistent(self) -> Self:
        expected = (
            self.duration_ns == self.interval.end_ns - self.interval.start_ns
            and self.all_windows_have_ttft_samples
            and self.all_windows_p95_ttft_pass
            and self.maximum_queue_depth <= self.queue_depth_threshold
            and self.restored_marker_precedes_stability
            and self.restore_trigger_follows_stability
        )
        if self.passed != expected:
            raise ValueError("SLO stability evidence is internally inconsistent")
        return self


class GPU0RestoreStageActivityEvidence(_StrictModel):
    """Observed GPU0 work throughout one causally bounded restore stage."""

    stage: Literal["h2d", "import-validation", "resume"]
    interval: PhaseInterval
    expected_arrival_mass: float = Field(ge=0.0, allow_inf_nan=False)
    scheduled_arrival_count: int = Field(ge=0)
    minimum_scheduled_arrivals: int = Field(ge=0)
    eligible_for_interference: bool
    sufficient_sample: bool
    all_scheduled_arrivals_routed_gpu0: bool
    completion_count: int = Field(ge=0)
    emitted_tokens: int = Field(ge=0)
    ttft_sample_count: int = Field(ge=0)
    passed: bool | None

    @model_validator(mode="after")
    def consistent(self) -> Self:
        expected_eligible = self.minimum_scheduled_arrivals > 0
        expected_sufficient = (
            expected_eligible and self.scheduled_arrival_count >= self.minimum_scheduled_arrivals
        )
        expected_pass = (
            expected_sufficient
            and self.all_scheduled_arrivals_routed_gpu0
            and self.completion_count > 0
            and self.emitted_tokens > 0
            and self.ttft_sample_count > 0
        )
        if (
            self.eligible_for_interference != expected_eligible
            or self.sufficient_sample != expected_sufficient
            or self.passed != (expected_pass if expected_eligible else None)
        ):
            raise ValueError("GPU0 restore-stage activity evidence is internally inconsistent")
        return self


class GPU0RestoreActivityEvidence(_StrictModel):
    interval: PhaseInterval
    arrival_count: int = Field(ge=0)
    completion_count: int = Field(ge=0)
    emitted_tokens: int = Field(ge=0)
    ttft_sample_count: int = Field(ge=0)
    early_half_completion_count: int = Field(ge=0)
    late_half_completion_count: int = Field(ge=0)
    activity_spans_restore: bool
    offered_rate_per_second: float = Field(ge=0.0, allow_inf_nan=False)
    offered_rate_matches_config: bool
    all_scheduled_arrivals_routed_gpu0: bool
    stage_activity: tuple[
        GPU0RestoreStageActivityEvidence,
        GPU0RestoreStageActivityEvidence,
        GPU0RestoreStageActivityEvidence,
    ]
    passed: bool

    @model_validator(mode="after")
    def consistent(self) -> Self:
        eligible = tuple(stage for stage in self.stage_activity if stage.eligible_for_interference)
        expected = (
            self.offered_rate_matches_config
            and self.all_scheduled_arrivals_routed_gpu0
            and self.arrival_count > 0
            and self.completion_count > 0
            and self.emitted_tokens > 0
            and self.ttft_sample_count > 0
            and self.activity_spans_restore
            == (self.early_half_completion_count > 0 and self.late_half_completion_count > 0)
            and tuple(stage.stage for stage in self.stage_activity)
            == ("h2d", "import-validation", "resume")
            and bool(eligible)
            and all(stage.passed is True for stage in eligible)
        )
        if self.passed != expected:
            raise ValueError("GPU0 restore-activity evidence is internally inconsistent")
        return self


class StateCorrectnessEvidence(_StrictModel):
    logical_state_bytes: int = Field(gt=0)
    preserved_state_bytes: int = Field(gt=0)
    source_blocks_expected: int = Field(gt=0)
    source_blocks_released: int = Field(ge=0)
    transport_integrity_valid: bool
    fresh_destination_allocations: bool

    @property
    def passed(self) -> bool:
        return (
            self.logical_state_bytes == self.preserved_state_bytes
            and self.source_blocks_released == self.source_blocks_expected
            and self.transport_integrity_valid
            and self.fresh_destination_allocations
        )


class BranchSemanticSnapshot(_StrictModel):
    """Checkpoint semantics without trusting a claimed runtime request ID."""

    logical_branch_id: str = Field(min_length=1)
    parent_logical_branch_id: str = Field(min_length=1)
    policy_epoch: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    tokenizer_id: str = Field(min_length=1)
    tokenizer_revision: str = Field(min_length=1)
    token_count: int = Field(gt=0)
    computed_tokens: int = Field(gt=0)
    token_history_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sampling_params_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RawBranchGroupRestoreEvidence(_StrictModel):
    """Raw identities and page-allocation maps used to prove restore semantics."""

    source: tuple[BranchSemanticSnapshot, ...]
    restored: tuple[BranchSemanticSnapshot, ...]
    source_runtime_request_ids: dict[str, str]
    restored_runtime_request_ids: dict[str, str]
    logical_page_ids_by_branch: dict[str, tuple[str, ...]]
    source_allocations_by_page: dict[str, RuntimeAllocationIdentity]
    restored_allocations_by_page: dict[str, RuntimeAllocationIdentity]
    expected_first_token_ids: dict[str, int]
    observed_first_token_ids: dict[str, int]
    continuation_token_counts: dict[str, int]


def build_branch_group_semantic_evidence(
    raw: RawBranchGroupRestoreEvidence,
) -> BranchGroupSemanticEvidence:
    """Reconcile actual runtime identities and allocation generations fail closed."""

    source = {record.logical_branch_id: record for record in raw.source}
    restored = {record.logical_branch_id: record for record in raw.restored}
    branch_ids = set(source)
    exact_branch_maps = (
        restored,
        raw.source_runtime_request_ids,
        raw.restored_runtime_request_ids,
        raw.logical_page_ids_by_branch,
        raw.expected_first_token_ids,
        raw.observed_first_token_ids,
        raw.continuation_token_counts,
    )
    if (
        len(source) != len(raw.source)
        or len(restored) != len(raw.restored)
        or any(set(mapping) != branch_ids for mapping in exact_branch_maps)
    ):
        raise ValueError("raw restore evidence does not cover one exact logical branch group")
    source_runtime_ids = tuple(raw.source_runtime_request_ids.values())
    restored_runtime_ids = tuple(raw.restored_runtime_request_ids.values())
    if (
        len(set(source_runtime_ids)) != len(source_runtime_ids)
        or len(set(restored_runtime_ids)) != len(restored_runtime_ids)
        or set(source_runtime_ids) & set(restored_runtime_ids)
    ):
        raise ValueError("source/restored runtime request incarnations are not fresh and unique")
    referenced_pages = {
        page_id for pages in raw.logical_page_ids_by_branch.values() for page_id in pages
    }
    if (
        not referenced_pages
        or set(raw.source_allocations_by_page) != referenced_pages
        or set(raw.restored_allocations_by_page) != referenced_pages
    ):
        raise ValueError("raw allocation maps do not exactly cover the branch page tables")
    if any(
        len(set(allocations.values())) != len(allocations)
        for allocations in (
            raw.source_allocations_by_page,
            raw.restored_allocations_by_page,
        )
    ):
        raise ValueError("distinct logical pages alias one physical allocation identity")

    def semantic_record(
        snapshot: BranchSemanticSnapshot, runtime_request_id: str
    ) -> BranchSemanticRecord:
        return BranchSemanticRecord(
            runtime_request_id=runtime_request_id,
            token_boundary_valid=True,
            **snapshot.model_dump(mode="python"),
        )

    ordered = sorted(branch_ids)
    return BranchGroupSemanticEvidence(
        source=tuple(
            semantic_record(source[branch_id], raw.source_runtime_request_ids[branch_id])
            for branch_id in ordered
        ),
        restored=tuple(
            semantic_record(restored[branch_id], raw.restored_runtime_request_ids[branch_id])
            for branch_id in ordered
        ),
        source_incarnations=tuple(
            RuntimeIncarnation(
                logical_branch_id=branch_id,
                runtime_request_id=raw.source_runtime_request_ids[branch_id],
                allocations=tuple(
                    raw.source_allocations_by_page[page_id]
                    for page_id in raw.logical_page_ids_by_branch[branch_id]
                ),
            )
            for branch_id in ordered
        ),
        restored_incarnations=tuple(
            RuntimeIncarnation(
                logical_branch_id=branch_id,
                runtime_request_id=raw.restored_runtime_request_ids[branch_id],
                allocations=tuple(
                    raw.restored_allocations_by_page[page_id]
                    for page_id in raw.logical_page_ids_by_branch[branch_id]
                ),
            )
            for branch_id in ordered
        ),
        expected_first_token_ids=raw.expected_first_token_ids,
        observed_first_token_ids=raw.observed_first_token_ids,
        continuation_token_counts=raw.continuation_token_counts,
    )


class BranchResumeEvidence(_StrictModel):
    expected_first_tokens: dict[str, int]
    observed_first_tokens: dict[str, int]
    continuation_token_counts: dict[str, int]
    restored_into_fresh_allocations: bool
    integrity_valid: bool
    semantic_continuity: BranchGroupSemanticEvidence

    @property
    def passed(self) -> bool:
        branches = set(self.expected_first_tokens)
        return (
            len(branches) == 8
            and set(self.observed_first_tokens) == branches
            and set(self.continuation_token_counts) == branches
            and self.expected_first_tokens == self.observed_first_tokens
            and all(count >= 8 for count in self.continuation_token_counts.values())
            and self.restored_into_fresh_allocations
            and self.integrity_valid
            and self.semantic_continuity.expected_first_token_ids == self.expected_first_tokens
            and self.semantic_continuity.observed_first_token_ids == self.observed_first_tokens
            and self.semantic_continuity.continuation_token_counts == self.continuation_token_counts
        )


class MovementAccountingEvidence(_StrictModel):
    logical_state_bytes: int = Field(gt=0)
    full_physical_touch_bytes: int = Field(gt=0)
    external_movement_bytes: int = Field(ge=0)
    avoidable_movement_bytes: int = Field(ge=0)
    critical_path_movement_bytes: int = Field(ge=0)
    accounting_duplicate_bytes: int = Field(ge=0)
    all_physical_passes_recorded: bool
    formulas_recomputed_from_raw_artifacts: bool

    @property
    def passed(self) -> bool:
        return (
            self.full_physical_touch_bytes >= self.logical_state_bytes
            and self.external_movement_bytes <= self.full_physical_touch_bytes
            and self.avoidable_movement_bytes <= self.full_physical_touch_bytes
            and self.critical_path_movement_bytes <= self.full_physical_touch_bytes
            and self.all_physical_passes_recorded
            and self.formulas_recomputed_from_raw_artifacts
        )


class BudgetEvidence(_StrictModel):
    conservative_gpu_seconds_before: float = Field(ge=0.0, allow_inf_nan=False)
    invocation_gpu_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    conservative_gpu_seconds_after: float = Field(ge=0.0, allow_inf_nan=False)
    hard_ceiling_gpu_seconds: float = Field(gt=0.0, allow_inf_nan=False)
    ledger_updated: bool

    @property
    def passed(self) -> bool:
        return (
            math.isclose(
                self.conservative_gpu_seconds_after,
                self.conservative_gpu_seconds_before + self.invocation_gpu_seconds,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            and self.conservative_gpu_seconds_after <= self.hard_ceiling_gpu_seconds
            and self.ledger_updated
        )


class CleanupEvidence(_StrictModel):
    zero_active_tasks: bool
    zero_running_containers: bool
    zero_endpoints: bool
    zero_reservations: bool
    zero_owned_child_processes: bool
    zero_profilers: bool
    authorized_persistent_volumes_only: bool

    @property
    def passed(self) -> bool:
        return all(
            (
                self.zero_active_tasks,
                self.zero_running_containers,
                self.zero_endpoints,
                self.zero_reservations,
                self.zero_owned_child_processes,
                self.zero_profilers,
                self.authorized_persistent_volumes_only,
            )
        )


class V10AssessmentWindows(_StrictModel):
    control: PhaseInterval
    overload: PhaseInterval
    two_gpu_excess_capacity: PhaseInterval
    queue_drain: PhaseInterval
    slo_stability: PhaseInterval
    restore_activity: PhaseInterval


class V10ScientificValidity(_StrictModel):
    schema_version: Literal["sloforge.branchfabric.experiment-004-v10-scientific-validity/v1"] = (
        "sloforge.branchfabric.experiment-004-v10-scientific-validity/v1"
    )
    state_correctness_pass: bool
    gpu0_overload_pass: bool
    two_gpu_excess_capacity_pass: bool
    queue_drain_pass: bool
    slo_restoration_pass: bool
    pre_restore_stability_pass: bool
    gpu0_active_during_restore_pass: bool
    branch_resume_pass: bool
    movement_accounting_pass: bool
    budget_pass: bool
    cleanup_pass: bool
    scientifically_valid: bool
    invalid_reasons: tuple[str, ...]
    timeline: V10Timeline | None = None
    gpu0_overload: GPU0OverloadEvidence | None = None
    two_gpu_excess_capacity: TwoGpuCapacityEvidence | None = None
    queue_drain: QueueDrainEvidence | None = None
    slo_stability: SLOStabilityEvidence | None = None
    gpu0_restore_activity: GPU0RestoreActivityEvidence | None = None
    state_correctness: StateCorrectnessEvidence | None = None
    branch_resume: BranchResumeEvidence | None = None
    movement_accounting: MovementAccountingEvidence | None = None
    budget: BudgetEvidence | None = None
    cleanup: CleanupEvidence | None = None

    @model_validator(mode="after")
    def all_required_booleans_fail_closed(self) -> Self:
        expected_flags = (
            self.state_correctness is not None and self.state_correctness.passed,
            self.gpu0_overload is not None and self.gpu0_overload.passed,
            self.two_gpu_excess_capacity is not None and self.two_gpu_excess_capacity.passed,
            self.queue_drain is not None and self.queue_drain.passed,
            self.slo_stability is not None and self.slo_stability.passed,
            self.slo_stability is not None and self.slo_stability.passed,
            self.gpu0_restore_activity is not None and self.gpu0_restore_activity.passed,
            self.branch_resume is not None and self.branch_resume.passed,
            self.movement_accounting is not None and self.movement_accounting.passed,
            self.budget is not None and self.budget.passed,
            self.cleanup is not None and self.cleanup.passed,
        )
        observed_flags = (
            self.state_correctness_pass,
            self.gpu0_overload_pass,
            self.two_gpu_excess_capacity_pass,
            self.queue_drain_pass,
            self.slo_restoration_pass,
            self.pre_restore_stability_pass,
            self.gpu0_active_during_restore_pass,
            self.branch_resume_pass,
            self.movement_accounting_pass,
            self.budget_pass,
            self.cleanup_pass,
        )
        if observed_flags != expected_flags:
            raise ValueError("scientific-validity booleans disagree with their raw evidence")
        expected_valid = (
            all(observed_flags) and self.timeline is not None and not self.invalid_reasons
        )
        if self.scientifically_valid != expected_valid:
            raise ValueError("scientifically_valid must be the conjunction of every gate")
        if not self.scientifically_valid and not self.invalid_reasons:
            raise ValueError("an invalid v10 assessment requires at least one reason")
        return self


def _metrics(
    workload: ServingWorkload,
    observations: tuple[ServingObservation, ...],
    interval: PhaseInterval,
) -> IntervalMetrics:
    return measure_serving_intervals(
        workload,
        observations,
        ServingMeasurementPlan(intervals=(interval,)),
    ).intervals[0]


def _validate_window_causality(
    timeline: V10Timeline,
    windows: V10AssessmentWindows,
    *,
    minimum_capacity_window_ns: int,
) -> None:
    expected_bounds = (
        (
            windows.control.end_ns,
            timeline.timestamp(V10Phase.LOAD_SPIKE_BEGIN),
            "control end/load spike",
        ),
        (
            windows.overload.start_ns,
            timeline.timestamp(V10Phase.LOAD_SPIKE_BEGIN),
            "overload start/load spike",
        ),
        (
            windows.overload.end_ns,
            timeline.timestamp(V10Phase.GPU0_OVERLOAD_CONFIRMED),
            "overload end/confirmation",
        ),
        (
            windows.two_gpu_excess_capacity.start_ns,
            timeline.timestamp(V10Phase.GPU1_FIRST_USEFUL_SERVING_REQUEST),
            "capacity start/first useful GPU1 request",
        ),
        (
            windows.two_gpu_excess_capacity.end_ns,
            timeline.timestamp(V10Phase.TWO_GPU_SERVICE_STABLE),
            "capacity end/two-GPU stability",
        ),
        (
            windows.queue_drain.start_ns,
            timeline.timestamp(V10Phase.SERVING_QUEUE_DRAIN_BEGIN),
            "queue drain start",
        ),
        (
            windows.queue_drain.end_ns,
            timeline.timestamp(V10Phase.SERVING_QUEUE_DRAIN_END),
            "queue drain end",
        ),
        (
            windows.slo_stability.start_ns,
            timeline.timestamp(V10Phase.SERVING_SLO_STABILITY_BEGIN),
            "SLO stability start",
        ),
        (
            windows.slo_stability.end_ns,
            timeline.timestamp(V10Phase.SERVING_SLO_STABILITY_END),
            "SLO stability end",
        ),
        (
            windows.restore_activity.start_ns,
            timeline.timestamp(V10Phase.RESTORE_TRIGGER),
            "restore activity start",
        ),
        (
            windows.restore_activity.end_ns,
            timeline.timestamp(V10Phase.ROLLOUT_CONTINUATION_COMPLETE),
            "restore activity end",
        ),
    )
    mismatches = tuple(name for observed, expected, name in expected_bounds if observed != expected)
    if mismatches:
        raise ValueError(f"assessment windows disagree with timeline: {mismatches!r}")
    if (
        windows.two_gpu_excess_capacity.end_ns - windows.two_gpu_excess_capacity.start_ns
        < minimum_capacity_window_ns
    ):
        raise ValueError("two-GPU excess capacity requires a full stability window")
    if timeline.timestamp(V10Phase.GPU0_OVERLOAD_CONFIRMED) > timeline.timestamp(
        V10Phase.RECLAIM_TRIGGER
    ):
        raise ValueError("reclamation preceded the measured GPU0 overload confirmation")


def _rate_matches(observed: float, configured: float, tolerance: float) -> bool:
    return abs(observed - configured) <= configured * tolerance


def outstanding_queue_depth_at(
    workload: ServingWorkload,
    observations: dict[str, ServingObservation],
    timestamp_ns: int,
) -> int:
    depth = 0
    for request in workload.requests:
        completion_ns = observations[request.request_id].completion_ns
        if request.arrival_ns <= timestamp_ns and (
            completion_ns is None or completion_ns > timestamp_ns
        ):
            depth += 1
    return depth


def _queue_trend(
    workload: ServingWorkload,
    observations: tuple[ServingObservation, ...],
    interval: PhaseInterval,
    *,
    sample_interval_ns: int,
) -> QueueTrendEvidence:
    by_request = {observation.request_id: observation for observation in observations}
    if len(by_request) != len(workload.requests) or set(by_request) != {
        request.request_id for request in workload.requests
    }:
        raise ValueError("queue trend requires complete, unique request observations")
    timestamps = list(range(interval.start_ns, interval.end_ns, sample_interval_ns))
    if not timestamps or timestamps[-1] != interval.end_ns:
        timestamps.append(interval.end_ns)
    if len(timestamps) < 3:
        raise ValueError("queue trend requires at least three boundary samples")
    depths = tuple(
        outstanding_queue_depth_at(workload, by_request, timestamp) for timestamp in timestamps
    )
    seconds = tuple((timestamp - timestamps[0]) / NS_PER_SECOND for timestamp in timestamps)
    mean_x = statistics.fmean(seconds)
    mean_y = statistics.fmean(depths)
    denominator = sum((value - mean_x) ** 2 for value in seconds)
    slope = (
        sum(
            (x_value - mean_x) * (y_value - mean_y)
            for x_value, y_value in zip(seconds, depths, strict=True)
        )
        / denominator
    )
    midpoint = len(depths) // 2
    first_mean = statistics.fmean(depths[:midpoint])
    second_mean = statistics.fmean(depths[midpoint:])
    if slope > 0.0 and depths[-1] > depths[0] and second_mean > first_mean:
        direction: Literal["positive", "negative", "flat"] = "positive"
        sustained = True
    elif slope < 0.0 and depths[-1] < depths[0] and second_mean < first_mean:
        direction = "negative"
        sustained = True
    else:
        direction = "flat"
        sustained = False
    return QueueTrendEvidence(
        interval=interval,
        sample_interval_ns=sample_interval_ns,
        sample_count=len(depths),
        initial_depth=depths[0],
        final_depth=depths[-1],
        first_half_mean_depth=first_mean,
        second_half_mean_depth=second_mean,
        slope_requests_per_second=slope,
        direction=direction,
        sustained=sustained,
    )


def _throughput_evidence(
    workload: ServingWorkload,
    observations: tuple[ServingObservation, ...],
    interval: PhaseInterval,
    *,
    gpu0_device: str,
    gpu1_device: str,
    configured_rate: float,
    tolerance: float,
) -> TwoGpuCapacityEvidence:
    duration_seconds = (interval.end_ns - interval.start_ns) / NS_PER_SECOND
    offered = sum(
        interval.start_ns <= request.arrival_ns < interval.end_ns for request in workload.requests
    )
    completed_rows = tuple(
        observation
        for observation in observations
        if interval.start_ns <= observation.completion_ns < interval.end_ns
    )
    offered_rate = offered / duration_seconds
    completed_rate = len(completed_rows) / duration_seconds
    return TwoGpuCapacityEvidence(
        interval=interval,
        offered_requests=offered,
        completed_requests=len(completed_rows),
        offered_rate_per_second=offered_rate,
        completed_rate_per_second=completed_rate,
        gpu0_completions=sum(row.device == gpu0_device for row in completed_rows),
        gpu1_completions=sum(row.device == gpu1_device for row in completed_rows),
        offered_rate_matches_config=_rate_matches(offered_rate, configured_rate, tolerance),
        passed=(
            _rate_matches(offered_rate, configured_rate, tolerance)
            and completed_rate > offered_rate
            and any(row.device == gpu0_device for row in completed_rows)
            and any(row.device == gpu1_device for row in completed_rows)
        ),
    )


def _slo_stability(
    workload: ServingWorkload,
    observations: tuple[ServingObservation, ...],
    interval: PhaseInterval,
    *,
    config: V10ValidityConfig,
    timeline: V10Timeline,
) -> SLOStabilityEvidence:
    duration_ns = interval.end_ns - interval.start_ns
    if duration_ns < config.stability_window_ns:
        raise ValueError("declared SLO interval is shorter than configured stability")
    intervals: list[PhaseInterval] = []
    cursor = interval.start_ns
    while cursor + config.evaluation_window_ns <= interval.end_ns:
        intervals.append(
            PhaseInterval(
                name=f"v10-slo-stability-{len(intervals):04d}",
                start_ns=cursor,
                end_ns=cursor + config.evaluation_window_ns,
            )
        )
        cursor += config.evaluation_window_ns
    if cursor != interval.end_ns:
        raise ValueError("SLO stability interval is not divisible by evaluation window")
    measurement = measure_serving_intervals(
        workload,
        observations,
        ServingMeasurementPlan(intervals=tuple(intervals)),
    )
    slo = ServingSLO(maximum_p95_ttft_ns=config.maximum_p95_ttft_ns)
    evaluations = tuple(evaluate_serving_slo(row, slo) for row in measurement.intervals)
    by_request = {row.request_id: row for row in observations}
    queue_event_times = {
        interval.start_ns,
        interval.end_ns,
        *(
            request.arrival_ns
            for request in workload.requests
            if interval.start_ns <= request.arrival_ns <= interval.end_ns
        ),
        *(
            row.completion_ns
            for row in observations
            if interval.start_ns <= row.completion_ns <= interval.end_ns
        ),
    }
    maximum_outstanding = max(
        outstanding_queue_depth_at(workload, by_request, timestamp)
        for timestamp in queue_event_times
    )
    have_samples = all(row.ttft.sample_count > 0 for row in measurement.intervals)
    p95_pass = all(evaluation.satisfied for evaluation in evaluations)
    restored_before = (
        timeline.timestamp(V10Phase.SERVING_SLO_RESTORED)
        <= timeline.timestamp(V10Phase.SERVING_SLO_STABILITY_BEGIN)
        <= interval.start_ns
    )
    restore_after = (
        interval.end_ns
        <= timeline.timestamp(V10Phase.SERVING_SLO_STABILITY_END)
        <= timeline.timestamp(V10Phase.RESTORE_ELIGIBLE)
        <= timeline.timestamp(V10Phase.RESTORE_TRIGGER)
    )
    passed = (
        have_samples
        and p95_pass
        and maximum_outstanding <= config.recovery_queue_depth_threshold
        and restored_before
        and restore_after
    )
    return SLOStabilityEvidence(
        interval=interval,
        duration_ns=duration_ns,
        evaluation_window_ns=config.evaluation_window_ns,
        evaluation_count=len(evaluations),
        all_windows_have_ttft_samples=have_samples,
        all_windows_p95_ttft_pass=p95_pass,
        maximum_queue_depth=maximum_outstanding,
        queue_depth_threshold=config.recovery_queue_depth_threshold,
        restored_marker_precedes_stability=restored_before,
        restore_trigger_follows_stability=restore_after,
        passed=passed,
    )


def _gpu0_restore_activity(
    workload: ServingWorkload,
    observations: tuple[ServingObservation, ...],
    interval: PhaseInterval,
    *,
    gpu0_device: str,
    configured_rate: float,
    tolerance: float,
    timeline: V10Timeline,
) -> GPU0RestoreActivityEvidence:
    requests = {
        request.request_id: request
        for request in workload.requests
        if interval.start_ns <= request.arrival_ns < interval.end_ns
    }
    by_request = {row.request_id: row for row in observations}
    if len(by_request) != len(observations) or any(
        request_id not in by_request for request_id in requests
    ):
        raise ValueError("restore activity requires complete unique request observations")
    gpu0_rows = tuple(row for row in observations if row.device == gpu0_device)
    scheduled_rows = tuple(by_request[request_id] for request_id in requests)
    duration_seconds = (interval.end_ns - interval.start_ns) / NS_PER_SECOND
    offered_rate = len(requests) / duration_seconds
    completions = sum(interval.start_ns <= row.completion_ns < interval.end_ns for row in gpu0_rows)
    emitted = sum(
        interval.start_ns <= timestamp < interval.end_ns
        for row in gpu0_rows
        for timestamp in row.token_timestamps_ns
    )
    ttft_samples = sum(
        bool(row.token_timestamps_ns)
        and interval.start_ns <= row.token_timestamps_ns[0] < interval.end_ns
        for row in gpu0_rows
    )
    midpoint_ns = interval.start_ns + (interval.end_ns - interval.start_ns) // 2
    early_completions = sum(
        interval.start_ns <= row.completion_ns < midpoint_ns for row in gpu0_rows
    )
    late_completions = sum(midpoint_ns <= row.completion_ns < interval.end_ns for row in gpu0_rows)
    spans_restore = early_completions > 0 and late_completions > 0
    rate_matches = _rate_matches(offered_rate, configured_rate, tolerance)
    stage_bounds: tuple[
        tuple[Literal["h2d", "import-validation", "resume"], V10Phase, V10Phase], ...
    ] = (
        ("h2d", V10Phase.H2D_BEGIN, V10Phase.H2D_END),
        ("import-validation", V10Phase.STATE_IMPORT_BEGIN, V10Phase.STATE_IMPORT_END),
        (
            "resume",
            V10Phase.BRANCH_RESUME_BEGIN,
            V10Phase.ROLLOUT_CONTINUATION_COMPLETE,
        ),
    )
    stage_activity: list[GPU0RestoreStageActivityEvidence] = []
    for stage, start_phase, end_phase in stage_bounds:
        stage_interval = PhaseInterval(
            name=f"gpu0-restore-{stage}",
            start_ns=timeline.timestamp(start_phase),
            end_ns=timeline.timestamp(end_phase),
        )
        stage_request_ids = {
            request.request_id
            for request in workload.requests
            if stage_interval.start_ns <= request.arrival_ns < stage_interval.end_ns
        }
        expected_mass = (
            (stage_interval.end_ns - stage_interval.start_ns) * configured_rate / NS_PER_SECOND
        )
        minimum_arrivals = (
            0 if expected_mass < 1.0 else max(1, math.floor(expected_mass * (1.0 - tolerance)))
        )
        eligible = minimum_arrivals > 0
        sufficient = eligible and len(stage_request_ids) >= minimum_arrivals
        stage_completions = sum(
            stage_interval.start_ns <= row.completion_ns < stage_interval.end_ns
            for row in gpu0_rows
        )
        stage_tokens = sum(
            stage_interval.start_ns <= timestamp < stage_interval.end_ns
            for row in gpu0_rows
            for timestamp in row.token_timestamps_ns
        )
        stage_ttft = sum(
            bool(row.token_timestamps_ns)
            and stage_interval.start_ns <= row.token_timestamps_ns[0] < stage_interval.end_ns
            for row in gpu0_rows
        )
        all_gpu0 = all(
            request_id in by_request and by_request[request_id].device == gpu0_device
            for request_id in stage_request_ids
        )
        stage_passed = (
            sufficient
            and all_gpu0
            and stage_completions > 0
            and stage_tokens > 0
            and stage_ttft > 0
        )
        stage_activity.append(
            GPU0RestoreStageActivityEvidence(
                stage=stage,
                interval=stage_interval,
                expected_arrival_mass=expected_mass,
                scheduled_arrival_count=len(stage_request_ids),
                minimum_scheduled_arrivals=minimum_arrivals,
                eligible_for_interference=eligible,
                sufficient_sample=sufficient,
                all_scheduled_arrivals_routed_gpu0=all_gpu0,
                completion_count=stage_completions,
                emitted_tokens=stage_tokens,
                ttft_sample_count=stage_ttft,
                passed=stage_passed if eligible else None,
            )
        )
    eligible_stages = tuple(stage for stage in stage_activity if stage.eligible_for_interference)
    stages_pass = bool(eligible_stages) and all(stage.passed is True for stage in eligible_stages)
    return GPU0RestoreActivityEvidence(
        interval=interval,
        arrival_count=len(requests),
        completion_count=completions,
        emitted_tokens=emitted,
        ttft_sample_count=ttft_samples,
        early_half_completion_count=early_completions,
        late_half_completion_count=late_completions,
        activity_spans_restore=spans_restore,
        offered_rate_per_second=offered_rate,
        offered_rate_matches_config=rate_matches,
        all_scheduled_arrivals_routed_gpu0=all(row.device == gpu0_device for row in scheduled_rows),
        stage_activity=(stage_activity[0], stage_activity[1], stage_activity[2]),
        passed=(
            rate_matches
            and bool(requests)
            and all(row.device == gpu0_device for row in scheduled_rows)
            and completions > 0
            and emitted > 0
            and ttft_samples > 0
            and spans_restore
            and stages_pass
        ),
    )


def assess_v10_scientific_validity(
    *,
    config: V10ValidityConfig,
    timeline: V10Timeline,
    windows: V10AssessmentWindows,
    workload: ServingWorkload,
    observations: tuple[ServingObservation, ...],
    gpu0_device: str,
    gpu1_device: str,
    state_correctness: StateCorrectnessEvidence,
    branch_resume: BranchResumeEvidence,
    movement_accounting: MovementAccountingEvidence,
    budget: BudgetEvidence,
    cleanup: CleanupEvidence,
) -> V10ScientificValidity:
    """Evaluate every required v10 gate from explicit immutable evidence."""

    if gpu0_device == gpu1_device:
        raise ValueError("v10 serving evidence must identify two distinct GPUs")
    _validate_window_causality(
        timeline,
        windows,
        minimum_capacity_window_ns=config.stability_window_ns,
    )
    if movement_accounting.logical_state_bytes != state_correctness.logical_state_bytes:
        raise ValueError("state and movement logical-byte denominators disagree")
    control = _metrics(workload, observations, windows.control)
    overload = _metrics(workload, observations, windows.overload)
    control_duration = (windows.control.end_ns - windows.control.start_ns) / NS_PER_SECOND
    overload_duration = (windows.overload.end_ns - windows.overload.start_ns) / NS_PER_SECOND
    control_rate = control.offered_requests / control_duration
    overload_rate = overload.offered_requests / overload_duration
    overload_completed_rate = overload.completions_in_interval / overload_duration
    overload_trend = _queue_trend(
        workload,
        observations,
        windows.overload,
        sample_interval_ns=config.evaluation_window_ns,
    )
    control_rate_match = _rate_matches(
        control_rate,
        config.control_offered_rate_per_second,
        config.arrival_rate_tolerance_fraction,
    )
    overload_rate_match = _rate_matches(
        overload_rate,
        config.spike_offered_rate_per_second,
        config.arrival_rate_tolerance_fraction,
    )
    control_ttft_pass = (
        control.ttft.p95 is not None and control.ttft.p95 <= config.maximum_p95_ttft_ns
    )
    ttft_exceeded = overload.ttft.p95 is not None and overload.ttft.p95 > config.maximum_p95_ttft_ns
    positive_queue = overload_trend.direction == "positive" and overload_trend.sustained
    offered_exceeds = overload_rate > overload_completed_rate
    overload_count = sum((ttft_exceeded, positive_queue, offered_exceeds))
    overload_evidence = GPU0OverloadEvidence(
        control=control,
        overload=overload,
        control_rate_matches_config=control_rate_match,
        overload_rate_matches_config=overload_rate_match,
        control_p95_ttft_pass=control_ttft_pass,
        ttft_exceeded=ttft_exceeded,
        queue_positive_trend=positive_queue,
        offered_rate_exceeds_completed_rate=offered_exceeds,
        overload_condition_count=overload_count,
        queue_trend=overload_trend,
        passed=(
            control_rate_match
            and overload_rate_match
            and control_ttft_pass
            and control.completed_requests > 0
            and overload_count >= 1
        ),
    )
    capacity = _throughput_evidence(
        workload,
        observations,
        windows.two_gpu_excess_capacity,
        gpu0_device=gpu0_device,
        gpu1_device=gpu1_device,
        configured_rate=config.spike_offered_rate_per_second,
        tolerance=config.arrival_rate_tolerance_fraction,
    )
    drain_trend = _queue_trend(
        workload,
        observations,
        windows.queue_drain,
        sample_interval_ns=config.evaluation_window_ns,
    )
    queue_drain = QueueDrainEvidence(
        trend=drain_trend,
        passed=(
            drain_trend.direction == "negative"
            and drain_trend.sustained
            and drain_trend.final_depth < drain_trend.initial_depth
        ),
    )
    stability = _slo_stability(
        workload,
        observations,
        windows.slo_stability,
        config=config,
        timeline=timeline,
    )
    restore_activity = _gpu0_restore_activity(
        workload,
        observations,
        windows.restore_activity,
        gpu0_device=gpu0_device,
        configured_rate=config.restore_offered_rate_per_second,
        tolerance=config.arrival_rate_tolerance_fraction,
        timeline=timeline,
    )
    flags = (
        state_correctness.passed,
        overload_evidence.passed,
        capacity.passed,
        queue_drain.passed,
        stability.passed,
        stability.passed,
        restore_activity.passed,
        branch_resume.passed,
        movement_accounting.passed,
        budget.passed,
        cleanup.passed,
    )
    gate_names = (
        "state correctness",
        "GPU0 overload",
        "two-GPU excess capacity",
        "queue drain",
        "SLO restoration",
        "pre-restore stability",
        "GPU0 active during restore",
        "branch resume",
        "movement accounting",
        "budget",
        "cleanup",
    )
    invalid_reasons = tuple(
        f"required gate failed: {name}"
        for name, passed in zip(gate_names, flags, strict=True)
        if not passed
    )
    return V10ScientificValidity(
        state_correctness_pass=flags[0],
        gpu0_overload_pass=flags[1],
        two_gpu_excess_capacity_pass=flags[2],
        queue_drain_pass=flags[3],
        slo_restoration_pass=flags[4],
        pre_restore_stability_pass=flags[5],
        gpu0_active_during_restore_pass=flags[6],
        branch_resume_pass=flags[7],
        movement_accounting_pass=flags[8],
        budget_pass=flags[9],
        cleanup_pass=flags[10],
        scientifically_valid=all(flags),
        invalid_reasons=invalid_reasons,
        timeline=timeline,
        gpu0_overload=overload_evidence,
        two_gpu_excess_capacity=capacity,
        queue_drain=queue_drain,
        slo_stability=stability,
        gpu0_restore_activity=restore_activity,
        state_correctness=state_correctness,
        branch_resume=branch_resume,
        movement_accounting=movement_accounting,
        budget=budget,
        cleanup=cleanup,
    )


def fail_closed_v10_scientific_validity(reason: str) -> V10ScientificValidity:
    """Produce the required all-false artifact when raw assessment cannot run."""

    if not reason:
        raise ValueError("fail-closed scientific validity requires an explicit reason")
    return V10ScientificValidity(
        state_correctness_pass=False,
        gpu0_overload_pass=False,
        two_gpu_excess_capacity_pass=False,
        queue_drain_pass=False,
        slo_restoration_pass=False,
        pre_restore_stability_pass=False,
        gpu0_active_during_restore_pass=False,
        branch_resume_pass=False,
        movement_accounting_pass=False,
        budget_pass=False,
        cleanup_pass=False,
        scientifically_valid=False,
        invalid_reasons=(reason,),
    )


__all__ = [
    "CAUSAL_V10_TIMELINE",
    "MINIMUM_STABILITY_WINDOW_NS",
    "REQUIRED_V10_TIMELINE",
    "BranchResumeEvidence",
    "BranchSemanticSnapshot",
    "BudgetEvidence",
    "CleanupEvidence",
    "GPU0OverloadEvidence",
    "GPU0RestoreActivityEvidence",
    "GPU0RestoreStageActivityEvidence",
    "MovementAccountingEvidence",
    "QueueDrainEvidence",
    "QueueTrendEvidence",
    "RawBranchGroupRestoreEvidence",
    "SLOStabilityEvidence",
    "StateCorrectnessEvidence",
    "TimelineEvidenceKind",
    "TwoGpuCapacityEvidence",
    "V10AssessmentWindows",
    "V10Phase",
    "V10ScientificValidity",
    "V10Timeline",
    "V10TimelineEvent",
    "V10ValidityConfig",
    "assess_v10_scientific_validity",
    "build_branch_group_semantic_evidence",
    "fail_closed_v10_scientific_validity",
    "outstanding_queue_depth_at",
]
