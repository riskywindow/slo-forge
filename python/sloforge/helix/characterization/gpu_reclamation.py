"""Experiment 004 warm GPU capacity-reclamation transaction primitives.

The models in this module are deliberately hardware-neutral.  Synthetic tests
exercise their invariants, while a GPU runner supplies hardware-backed clocks,
runtime allocation identities, and phase markers.  In particular, a warm vLLM
pool reclaim is expressed as allocator capacity becoming reusable; it is never
misreported as driver-visible HBM deallocation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=512)]
GpuUuid = Annotated[str, StringConstraints(pattern=r"^GPU-[0-9A-Za-z-]+$")]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class ReclamationMode(StrEnum):
    KILL_AND_RECOMPUTE = "KILL_AND_RECOMPUTE"
    PRESERVE_NAIVE = "PRESERVE_NAIVE"
    PRESERVE_OPTIMIZED = "PRESERVE_OPTIMIZED"


class ReclamationTransactionState(StrEnum):
    ROLLOUT_ACTIVE = "ROLLOUT_ACTIVE"
    RECLAIM_TRIGGERED = "RECLAIM_TRIGGERED"
    ADMISSIONS_STOPPED = "ADMISSIONS_STOPPED"
    SOURCE_FROZEN = "SOURCE_FROZEN"
    CAPTURED = "CAPTURED"
    TRANSPORT_VALIDATED = "TRANSPORT_VALIDATED"
    TRANSPORT_PUBLISHED = "TRANSPORT_PUBLISHED"
    STATE_DISCARDED = "STATE_DISCARDED"
    SOURCE_RELEASING = "SOURCE_RELEASING"
    CAPACITY_CONFIRMED = "CAPACITY_CONFIRMED"
    SECONDARY_SERVING_ENABLED = "SECONDARY_SERVING_ENABLED"
    SECONDARY_SERVING_ACTIVE = "SECONDARY_SERVING_ACTIVE"
    SECONDARY_SERVING_DRAINED = "SECONDARY_SERVING_DRAINED"
    RESTORE_TRIGGERED = "RESTORE_TRIGGERED"
    DESTINATION_PREPARED = "DESTINATION_PREPARED"
    STATE_IMPORTED = "STATE_IMPORTED"
    RECOMPUTING = "RECOMPUTING"
    STATE_VALIDATED = "STATE_VALIDATED"
    ROLLOUT_ADMITTED = "ROLLOUT_ADMITTED"
    FIRST_RESUMED_TOKEN = "FIRST_RESUMED_TOKEN"
    ROLLOUT_RESUMED = "ROLLOUT_RESUMED"
    COMPLETED = "COMPLETED"
    FAILED_ROLLED_BACK_PRE_RELEASE = "FAILED_ROLLED_BACK_PRE_RELEASE"
    FAILED_CLOSED_POST_RELEASE = "FAILED_CLOSED_POST_RELEASE"
    FAILED_CLOSED_CORRUPT = "FAILED_CLOSED_CORRUPT"


class ExperimentPhase(StrEnum):
    HELIX_RECLAIM_TRIGGER = "HELIX_RECLAIM_TRIGGER"
    ROLLOUT_ADMISSION_STOP = "ROLLOUT_ADMISSION_STOP"
    BRANCH_QUIESCE = "BRANCH_QUIESCE"
    STATE_CAPTURE_BEGIN = "STATE_CAPTURE_BEGIN"
    STATE_CAPTURE_END = "STATE_CAPTURE_END"
    STATE_TRANSFORM_BEGIN = "STATE_TRANSFORM_BEGIN"
    STATE_TRANSFORM_END = "STATE_TRANSFORM_END"
    INTEGRITY_BEGIN = "INTEGRITY_BEGIN"
    INTEGRITY_END = "INTEGRITY_END"
    D2H_BEGIN = "D2H_BEGIN"
    D2H_END = "D2H_END"
    STATE_PUBLISH = "STATE_PUBLISH"
    GPU_STATE_RELEASE_BEGIN = "GPU_STATE_RELEASE_BEGIN"
    GPU_STATE_RELEASE_END = "GPU_STATE_RELEASE_END"
    HBM_RECLAIM_CONFIRMED = "HBM_RECLAIM_CONFIRMED"
    SERVING_SECONDARY_ENABLE = "SERVING_SECONDARY_ENABLE"
    GPU1_FIRST_SERVING_REQUEST = "GPU1_FIRST_SERVING_REQUEST"
    SERVING_SLO_RESTORED = "SERVING_SLO_RESTORED"
    ROLLOUT_RESTORE_TRIGGER = "ROLLOUT_RESTORE_TRIGGER"
    H2D_BEGIN = "H2D_BEGIN"
    H2D_END = "H2D_END"
    STATE_IMPORT_BEGIN = "STATE_IMPORT_BEGIN"
    STATE_IMPORT_END = "STATE_IMPORT_END"
    STATE_VALIDATE_BEGIN = "STATE_VALIDATE_BEGIN"
    STATE_VALIDATE_END = "STATE_VALIDATE_END"
    BRANCH_RESUME_BEGIN = "BRANCH_RESUME_BEGIN"
    FIRST_RESUMED_TOKEN = "FIRST_RESUMED_TOKEN"
    ROLLOUT_RESUME_COMPLETE = "ROLLOUT_RESUME_COMPLETE"


REQUIRED_PRESERVATION_PHASES = frozenset(ExperimentPhase)


class TimelineKind(StrEnum):
    RECLAMATION = "reclamation"
    RESTORE = "restore"


class CriticalStage(StrEnum):
    ADMISSION_STOP = "admission_stop"
    BRANCH_QUIESCE = "branch_quiesce"
    FINAL_STATE_CAPTURE = "final_state_capture"
    DELTA_EXTRACTION = "delta_extraction"
    SOURCE_LAYOUT_READ = "source_layout_read"
    STATE_TRANSFORM = "state_transform"
    INTEGRITY_GENERATION = "integrity_generation"
    DEVICE_TO_HOST = "device_to_host"
    READ_TRANSFORM_INTEGRITY_D2H_PIPELINE = "read_transform_integrity_d2h_pipeline"
    TRANSPORT_PUBLISH = "transport_publish"
    RUNTIME_STATE_RELEASE = "runtime_state_release"
    CAPACITY_RECLAIM_CONFIRMATION = "capacity_reclaim_confirmation"
    SERVING_SECONDARY_ENABLE = "serving_secondary_enable"
    STATE_DISCARD = "state_discard"
    DESTINATION_ALLOCATION = "destination_allocation"
    TRANSPORT_LAYOUT_READ = "transport_layout_read"
    H2D = "h2d"
    DESTINATION_CONVERSION = "destination_conversion"
    BLOCK_PAGE_RECONSTRUCTION = "block_page_reconstruction"
    H2D_UNPACK_REPAGE_PIPELINE = "h2d_unpack_repage_pipeline"
    RUNTIME_IMPORT = "runtime_import"
    RECOMPUTE_PREFILL = "recompute_prefill"
    STATE_VALIDATION = "state_validation"
    SCHEDULER_ADMISSION = "scheduler_admission"
    FIRST_FORWARD = "first_forward"
    FIRST_TOKEN_OBSERVATION = "first_token_observation"


_NAIVE_RECLAMATION = (
    CriticalStage.ADMISSION_STOP,
    CriticalStage.BRANCH_QUIESCE,
    CriticalStage.FINAL_STATE_CAPTURE,
    CriticalStage.DELTA_EXTRACTION,
    CriticalStage.SOURCE_LAYOUT_READ,
    CriticalStage.STATE_TRANSFORM,
    CriticalStage.DEVICE_TO_HOST,
    CriticalStage.INTEGRITY_GENERATION,
    CriticalStage.TRANSPORT_PUBLISH,
    CriticalStage.RUNTIME_STATE_RELEASE,
    CriticalStage.CAPACITY_RECLAIM_CONFIRMATION,
    CriticalStage.SERVING_SECONDARY_ENABLE,
)

_OPTIMIZED_RECLAMATION = (
    CriticalStage.ADMISSION_STOP,
    CriticalStage.BRANCH_QUIESCE,
    CriticalStage.FINAL_STATE_CAPTURE,
    CriticalStage.DELTA_EXTRACTION,
    CriticalStage.READ_TRANSFORM_INTEGRITY_D2H_PIPELINE,
    CriticalStage.TRANSPORT_PUBLISH,
    CriticalStage.RUNTIME_STATE_RELEASE,
    CriticalStage.CAPACITY_RECLAIM_CONFIRMATION,
    CriticalStage.SERVING_SECONDARY_ENABLE,
)

_KILL_RECLAMATION = (
    CriticalStage.ADMISSION_STOP,
    CriticalStage.BRANCH_QUIESCE,
    CriticalStage.STATE_DISCARD,
    CriticalStage.RUNTIME_STATE_RELEASE,
    CriticalStage.CAPACITY_RECLAIM_CONFIRMATION,
    CriticalStage.SERVING_SECONDARY_ENABLE,
)

_NAIVE_RESTORE = (
    CriticalStage.DESTINATION_ALLOCATION,
    CriticalStage.TRANSPORT_LAYOUT_READ,
    CriticalStage.H2D,
    CriticalStage.DESTINATION_CONVERSION,
    CriticalStage.BLOCK_PAGE_RECONSTRUCTION,
    CriticalStage.RUNTIME_IMPORT,
    CriticalStage.STATE_VALIDATION,
    CriticalStage.SCHEDULER_ADMISSION,
    CriticalStage.FIRST_FORWARD,
    CriticalStage.FIRST_TOKEN_OBSERVATION,
)

_OPTIMIZED_RESTORE = (
    CriticalStage.DESTINATION_ALLOCATION,
    CriticalStage.H2D_UNPACK_REPAGE_PIPELINE,
    CriticalStage.RUNTIME_IMPORT,
    CriticalStage.STATE_VALIDATION,
    CriticalStage.SCHEDULER_ADMISSION,
    CriticalStage.FIRST_FORWARD,
    CriticalStage.FIRST_TOKEN_OBSERVATION,
)

_KILL_RESTORE = (
    CriticalStage.DESTINATION_ALLOCATION,
    CriticalStage.RECOMPUTE_PREFILL,
    CriticalStage.STATE_VALIDATION,
    CriticalStage.SCHEDULER_ADMISSION,
    CriticalStage.FIRST_FORWARD,
    CriticalStage.FIRST_TOKEN_OBSERVATION,
)


class CriticalStageRecord(_StrictModel):
    stage: CriticalStage
    start_ns: int = Field(ge=0)
    end_ns: int = Field(ge=0)

    @model_validator(mode="after")
    def positive_half_open_interval(self) -> Self:
        if self.end_ns < self.start_ns:
            raise ValueError("critical stage end precedes start")
        return self

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns


class CriticalTimeline(_StrictModel):
    kind: TimelineKind
    mode: ReclamationMode
    stages: tuple[CriticalStageRecord, ...]

    @model_validator(mode="after")
    def contiguous_complete_decomposition(self) -> Self:
        expected = {
            (TimelineKind.RECLAMATION, ReclamationMode.PRESERVE_NAIVE): _NAIVE_RECLAMATION,
            (TimelineKind.RECLAMATION, ReclamationMode.PRESERVE_OPTIMIZED): (
                _OPTIMIZED_RECLAMATION
            ),
            (TimelineKind.RECLAMATION, ReclamationMode.KILL_AND_RECOMPUTE): (_KILL_RECLAMATION),
            (TimelineKind.RESTORE, ReclamationMode.PRESERVE_NAIVE): _NAIVE_RESTORE,
            (TimelineKind.RESTORE, ReclamationMode.PRESERVE_OPTIMIZED): _OPTIMIZED_RESTORE,
            (TimelineKind.RESTORE, ReclamationMode.KILL_AND_RECOMPUTE): _KILL_RESTORE,
        }[(self.kind, self.mode)]
        if tuple(item.stage for item in self.stages) != expected:
            raise ValueError("critical timeline stages do not match the mode/kind contract")
        if any(
            left.end_ns != right.start_ns
            for left, right in zip(self.stages, self.stages[1:], strict=False)
        ):
            raise ValueError("critical timeline contains a gap or overlap")
        if self.stages:
            elapsed = self.stages[-1].end_ns - self.stages[0].start_ns
            if sum(item.duration_ns for item in self.stages) != elapsed:
                raise ValueError("critical timeline stage durations do not conserve wall time")
        return self

    @property
    def duration_ns(self) -> int:
        if not self.stages:
            return 0
        return self.stages[-1].end_ns - self.stages[0].start_ns


class ExperimentPhaseMarker(_StrictModel):
    phase: ExperimentPhase
    monotonic_timestamp_ns: int = Field(ge=0)
    logical_bytes: int = Field(default=0, ge=0)
    physical_bytes: int = Field(default=0, ge=0)
    branch_id: str | None = None
    attributes: dict[NonEmpty, bool | int | float | str | None] = Field(default_factory=dict)


class RuntimeAllocationIdentity(_StrictModel):
    gpu_uuid: GpuUuid
    block_index: int = Field(ge=0)
    allocation_epoch: int = Field(ge=0)


class RuntimeIncarnation(_StrictModel):
    logical_branch_id: NonEmpty
    runtime_request_id: NonEmpty
    allocations: tuple[RuntimeAllocationIdentity, ...]

    @model_validator(mode="after")
    def unique_allocations(self) -> Self:
        if not self.allocations:
            raise ValueError("one runtime incarnation requires physical allocation evidence")
        slots = {(item.gpu_uuid, item.block_index) for item in self.allocations}
        if len(self.allocations) != len(slots):
            raise ValueError("one runtime incarnation contains duplicate allocation identities")
        return self


class BranchSemanticRecord(_StrictModel):
    logical_branch_id: NonEmpty
    runtime_request_id: NonEmpty
    parent_logical_branch_id: NonEmpty
    policy_epoch: NonEmpty
    model_id: NonEmpty
    model_revision: NonEmpty
    tokenizer_id: NonEmpty
    tokenizer_revision: NonEmpty
    token_count: int = Field(gt=0)
    computed_tokens: int = Field(gt=0)
    token_history_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    sampling_params_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    token_boundary_valid: Literal[True] = True

    @model_validator(mode="after")
    def valid_token_boundary(self) -> Self:
        if not self.computed_tokens <= self.token_count <= self.computed_tokens + 1:
            raise ValueError("branch token history is not at a vLLM decode boundary")
        if self.parent_logical_branch_id == self.logical_branch_id:
            raise ValueError("logical branch cannot be its own parent")
        return self


class BranchGroupSemanticEvidence(_StrictModel):
    """Source-to-restored branch continuity and first-token evidence."""

    source: tuple[BranchSemanticRecord, ...]
    restored: tuple[BranchSemanticRecord, ...]
    source_incarnations: tuple[RuntimeIncarnation, ...]
    restored_incarnations: tuple[RuntimeIncarnation, ...]
    expected_first_token_ids: dict[NonEmpty, int]
    observed_first_token_ids: dict[NonEmpty, int]
    continuation_token_counts: dict[NonEmpty, int]
    minimum_continuation_tokens: int = Field(default=8, ge=2)

    @model_validator(mode="after")
    def exact_semantic_continuity(self) -> Self:
        collections = (
            self.source,
            self.restored,
            self.source_incarnations,
            self.restored_incarnations,
        )
        for records in collections:
            logical_ids = [record.logical_branch_id for record in records]
            if not logical_ids or len(logical_ids) != len(set(logical_ids)):
                raise ValueError("semantic evidence requires unique non-empty branch groups")
            if logical_ids != sorted(logical_ids):
                raise ValueError("semantic branch evidence must be in deterministic order")
        expected_ids = {record.logical_branch_id for record in self.source}
        if any(
            {record.logical_branch_id for record in records} != expected_ids
            for records in collections[1:]
        ):
            raise ValueError("source and restored semantic evidence cover different branches")
        if any(
            set(mapping) != expected_ids
            for mapping in (
                self.expected_first_token_ids,
                self.observed_first_token_ids,
                self.continuation_token_counts,
            )
        ):
            raise ValueError("resume output evidence does not cover every logical branch")
        if self.expected_first_token_ids != self.observed_first_token_ids:
            raise ValueError("first restored tokens differ from checkpoint control tokens")
        if any(not 0 <= token < 1 << 64 for token in self.observed_first_token_ids.values()):
            raise ValueError("restored token IDs must fit unsigned 64-bit integers")
        if any(
            count < self.minimum_continuation_tokens
            for count in self.continuation_token_counts.values()
        ):
            raise ValueError(
                "restored branches did not continue for the declared validation window"
            )

        source_by_id = {record.logical_branch_id: record for record in self.source}
        restored_by_id = {record.logical_branch_id: record for record in self.restored}
        source_incarnations = {
            record.logical_branch_id: record for record in self.source_incarnations
        }
        restored_incarnations = {
            record.logical_branch_id: record for record in self.restored_incarnations
        }
        identity_fields = (
            "parent_logical_branch_id",
            "policy_epoch",
            "model_id",
            "model_revision",
            "tokenizer_id",
            "tokenizer_revision",
            "token_count",
            "computed_tokens",
            "token_history_sha256",
            "sampling_params_sha256",
        )
        if len({record.parent_logical_branch_id for record in self.source}) != 1:
            raise ValueError("semantic branch group does not retain one logical parent")
        group_identity_fields = (
            "policy_epoch",
            "model_id",
            "model_revision",
            "tokenizer_id",
            "tokenizer_revision",
        )
        if any(
            len({getattr(record, field) for record in self.source}) != 1
            for field in group_identity_fields
        ):
            raise ValueError("semantic branch group has inconsistent runtime identity")
        for logical_id in sorted(expected_ids):
            source = source_by_id[logical_id]
            restored = restored_by_id[logical_id]
            if any(getattr(source, field) != getattr(restored, field) for field in identity_fields):
                raise ValueError("restored branch identity or token boundary differs from source")
            if source.runtime_request_id == restored.runtime_request_id:
                raise ValueError("restored branch reused its source runtime request incarnation")
            source_incarnation = source_incarnations[logical_id]
            restored_incarnation = restored_incarnations[logical_id]
            if source_incarnation.runtime_request_id != source.runtime_request_id or (
                restored_incarnation.runtime_request_id != restored.runtime_request_id
            ):
                raise ValueError("semantic record and allocation incarnation IDs disagree")
        source_allocations = {
            allocation
            for incarnation in self.source_incarnations
            for allocation in incarnation.allocations
        }
        restored_allocations = {
            allocation
            for incarnation in self.restored_incarnations
            for allocation in incarnation.allocations
        }
        if source_allocations & restored_allocations:
            raise ValueError("restored group retained a stale physical allocation generation")
        for allocations in (source_allocations, restored_allocations):
            slot_generations: dict[tuple[str, int], set[int]] = {}
            for allocation in allocations:
                slot = (allocation.gpu_uuid, allocation.block_index)
                slot_generations.setdefault(slot, set()).add(allocation.allocation_epoch)
            if any(len(generations) != 1 for generations in slot_generations.values()):
                raise ValueError("one branch group assigns multiple generations to one GPU slot")
        return self


class PilotValidityEvidence(_StrictModel):
    gpu_uuids: tuple[GpuUuid, GpuUuid]
    gpu_models: tuple[NonEmpty, NonEmpty]
    both_models_warm_before_trigger: bool
    gpu0_baseline_valid: bool
    branch_count: int = Field(ge=0, le=64)
    prefix_tokens: int = Field(ge=0)
    suffix_tokens: int = Field(ge=0)
    shared_bytes: int = Field(ge=0)
    private_bytes: int = Field(ge=0)
    logical_bytes: int = Field(ge=0)
    source_assigned_bytes_before: int = Field(ge=0)
    source_assigned_bytes_after: int = Field(ge=0)
    source_pool_reserved_bytes_before: int = Field(ge=0)
    source_pool_reserved_bytes_after: int = Field(ge=0)
    source_blocks_allocator_available: bool
    source_hashes_cleared: bool
    transport_integrity_valid: bool
    source_transport_destination_layouts_distinct: bool
    movement_domains_complete: bool
    required_phase_events: frozenset[ExperimentPhase]
    critical_timelines_valid: bool
    gpu1_served_real_request: bool
    serving_metrics_recorded: bool
    restored_branch_count: int = Field(ge=0, le=64)
    all_restored_allocations_fresh: bool
    branch_semantics: BranchGroupSemanticEvidence
    resumed_continuation_valid: bool
    required_trace_events_dropped: int = Field(ge=0)
    cleanup_passed: bool
    nvml_hbm_release_claimed: bool = False

    @model_validator(mode="after")
    def pilot_contract(self) -> Self:
        failures: list[str] = []
        if len(set(self.gpu_uuids)) != 2:
            failures.append("two distinct GPU UUIDs")
        if any(
            "A100" not in model or "80GB" not in model.replace(" ", "") for model in self.gpu_models
        ):
            failures.append("two A100-80GB models")
        if not self.both_models_warm_before_trigger:
            failures.append("both warm models")
        if not self.gpu0_baseline_valid:
            failures.append("GPU0 serving baseline")
        if self.branch_count != 8 or self.restored_branch_count != 8:
            failures.append("eight source and restored branches")
        if self.prefix_tokens != 16_384 or self.suffix_tokens < 256:
            failures.append("16K prefix and >=256 suffix")
        if min(self.shared_bytes, self.private_bytes, self.logical_bytes) <= 0:
            failures.append("positive exact state bytes")
        if self.logical_bytes != self.shared_bytes + self.private_bytes:
            failures.append("logical byte conservation")
        reclaimed = self.source_assigned_bytes_before - self.source_assigned_bytes_after
        if reclaimed < self.logical_bytes:
            failures.append("allocator-assigned KV reclaim")
        if self.source_pool_reserved_bytes_before != self.source_pool_reserved_bytes_after:
            failures.append("warm runtime fixed KV pool")
        required_true = {
            "source block availability": self.source_blocks_allocator_available,
            "source hashes cleared": self.source_hashes_cleared,
            "transport integrity": self.transport_integrity_valid,
            "cross-layout proof": self.source_transport_destination_layouts_distinct,
            "movement domains": self.movement_domains_complete,
            "critical timelines": self.critical_timelines_valid,
            "GPU1 real serving": self.gpu1_served_real_request,
            "serving metrics": self.serving_metrics_recorded,
            "fresh destination allocations": self.all_restored_allocations_fresh,
            "resumed continuation": self.resumed_continuation_valid,
            "cleanup": self.cleanup_passed,
        }
        failures.extend(name for name, passed in required_true.items() if not passed)
        if not REQUIRED_PRESERVATION_PHASES.issubset(self.required_phase_events):
            failures.append("required phase markers")
        if self.required_trace_events_dropped:
            failures.append("zero required trace drops")
        if self.nvml_hbm_release_claimed:
            failures.append("no false driver-visible HBM release claim")
        if failures:
            raise ValueError("invalid Experiment 004 pilot: " + ", ".join(failures))
        return self


_PRESERVE_TRANSITIONS: Mapping[
    ReclamationTransactionState, tuple[ReclamationTransactionState, ...]
] = {
    ReclamationTransactionState.ROLLOUT_ACTIVE: (ReclamationTransactionState.RECLAIM_TRIGGERED,),
    ReclamationTransactionState.RECLAIM_TRIGGERED: (
        ReclamationTransactionState.ADMISSIONS_STOPPED,
    ),
    ReclamationTransactionState.ADMISSIONS_STOPPED: (ReclamationTransactionState.SOURCE_FROZEN,),
    ReclamationTransactionState.SOURCE_FROZEN: (ReclamationTransactionState.CAPTURED,),
    ReclamationTransactionState.CAPTURED: (ReclamationTransactionState.TRANSPORT_VALIDATED,),
    ReclamationTransactionState.TRANSPORT_VALIDATED: (
        ReclamationTransactionState.TRANSPORT_PUBLISHED,
    ),
    ReclamationTransactionState.TRANSPORT_PUBLISHED: (
        ReclamationTransactionState.SOURCE_RELEASING,
    ),
    ReclamationTransactionState.SOURCE_RELEASING: (ReclamationTransactionState.CAPACITY_CONFIRMED,),
    ReclamationTransactionState.CAPACITY_CONFIRMED: (
        ReclamationTransactionState.SECONDARY_SERVING_ENABLED,
    ),
    ReclamationTransactionState.SECONDARY_SERVING_ENABLED: (
        ReclamationTransactionState.SECONDARY_SERVING_ACTIVE,
    ),
    ReclamationTransactionState.SECONDARY_SERVING_ACTIVE: (
        ReclamationTransactionState.SECONDARY_SERVING_DRAINED,
    ),
    ReclamationTransactionState.SECONDARY_SERVING_DRAINED: (
        ReclamationTransactionState.RESTORE_TRIGGERED,
    ),
    ReclamationTransactionState.RESTORE_TRIGGERED: (
        ReclamationTransactionState.DESTINATION_PREPARED,
    ),
    ReclamationTransactionState.DESTINATION_PREPARED: (ReclamationTransactionState.STATE_IMPORTED,),
    ReclamationTransactionState.STATE_IMPORTED: (ReclamationTransactionState.STATE_VALIDATED,),
    ReclamationTransactionState.STATE_VALIDATED: (ReclamationTransactionState.ROLLOUT_ADMITTED,),
    ReclamationTransactionState.ROLLOUT_ADMITTED: (
        ReclamationTransactionState.FIRST_RESUMED_TOKEN,
    ),
    ReclamationTransactionState.FIRST_RESUMED_TOKEN: (ReclamationTransactionState.ROLLOUT_RESUMED,),
    ReclamationTransactionState.ROLLOUT_RESUMED: (ReclamationTransactionState.COMPLETED,),
}


_KILL_OVERRIDES: Mapping[ReclamationTransactionState, tuple[ReclamationTransactionState, ...]] = {
    ReclamationTransactionState.SOURCE_FROZEN: (ReclamationTransactionState.STATE_DISCARDED,),
    ReclamationTransactionState.STATE_DISCARDED: (ReclamationTransactionState.SOURCE_RELEASING,),
    ReclamationTransactionState.RESTORE_TRIGGERED: (ReclamationTransactionState.RECOMPUTING,),
    ReclamationTransactionState.RECOMPUTING: (ReclamationTransactionState.STATE_VALIDATED,),
}


class ReclamationTransaction:
    """Small fail-closed control machine with an injected monotonic clock."""

    def __init__(
        self,
        mode: ReclamationMode,
        *,
        seed: int,
        clock_ns: Callable[[], int],
    ) -> None:
        if not 0 <= seed < 1 << 63:
            raise ValueError("seed must fit a signed 64-bit integer")
        self.mode = mode
        self.seed = seed
        self._clock_ns = clock_ns
        self.state = ReclamationTransactionState.ROLLOUT_ACTIVE
        initial_time = self._clock_ns()
        if initial_time < 0:
            raise ValueError("transaction clock must be non-negative")
        self._last_timestamp_ns = initial_time
        self.history: list[tuple[ReclamationTransactionState, int]] = [(self.state, initial_time)]
        self.phase_markers: list[ExperimentPhaseMarker] = []
        self.source_released = False

    def _observe_time(self) -> int:
        timestamp = self._clock_ns()
        if timestamp < self._last_timestamp_ns:
            raise RuntimeError("transaction monotonic clock moved backwards")
        self._last_timestamp_ns = timestamp
        return timestamp

    def transition(self, destination: ReclamationTransactionState) -> None:
        transitions = dict(_PRESERVE_TRANSITIONS)
        if self.mode is ReclamationMode.KILL_AND_RECOMPUTE:
            transitions.update(_KILL_OVERRIDES)
        allowed = transitions.get(self.state, ())
        if destination not in allowed:
            raise RuntimeError(f"illegal reclamation transition {self.state} -> {destination}")
        self.state = destination
        if destination is ReclamationTransactionState.SOURCE_RELEASING:
            self.source_released = True
        self.history.append((destination, self._observe_time()))

    def fail(self, *, corruption: bool = False) -> None:
        if self.state in {
            ReclamationTransactionState.COMPLETED,
            ReclamationTransactionState.FAILED_ROLLED_BACK_PRE_RELEASE,
            ReclamationTransactionState.FAILED_CLOSED_POST_RELEASE,
            ReclamationTransactionState.FAILED_CLOSED_CORRUPT,
        }:
            raise RuntimeError("cannot fail an already-terminal reclamation transaction")
        if corruption:
            destination = ReclamationTransactionState.FAILED_CLOSED_CORRUPT
        elif self.source_released:
            destination = ReclamationTransactionState.FAILED_CLOSED_POST_RELEASE
        else:
            destination = ReclamationTransactionState.FAILED_ROLLED_BACK_PRE_RELEASE
        self.state = destination
        self.history.append((destination, self._observe_time()))

    def mark(
        self,
        phase: ExperimentPhase,
        *,
        logical_bytes: int = 0,
        physical_bytes: int = 0,
        branch_id: str | None = None,
        attributes: dict[str, bool | int | float | str | None] | None = None,
    ) -> ExperimentPhaseMarker:
        marker = ExperimentPhaseMarker(
            phase=phase,
            monotonic_timestamp_ns=self._observe_time(),
            logical_bytes=logical_bytes,
            physical_bytes=physical_bytes,
            branch_id=branch_id,
            attributes=attributes or {},
        )
        self.phase_markers.append(marker)
        return marker


def allocation_is_fresh(
    source: RuntimeAllocationIdentity, destination: RuntimeAllocationIdentity
) -> bool:
    """Allocation generations, rather than reusable pool indices, define freshness."""

    return source.gpu_uuid != destination.gpu_uuid or (
        source.block_index != destination.block_index
        or source.allocation_epoch != destination.allocation_epoch
    )


def phase_trace_binding(phase: ExperimentPhase) -> tuple[str, str]:
    """Return existing trace-v1 operation names for one Experiment 004 marker.

    New phase vocabulary is carried in ``attributes['experiment_phase']`` so
    the trace-v1 enum and Rust/Python wire boundary remain backward compatible.
    """

    if phase in {ExperimentPhase.STATE_CAPTURE_BEGIN, ExperimentPhase.STATE_CAPTURE_END}:
        return "CAPTURE", "STATE_SNAPSHOT"
    if phase in {ExperimentPhase.STATE_TRANSFORM_BEGIN, ExperimentPhase.STATE_TRANSFORM_END}:
        return "BRANCH_MIGRATION", "STATE_REPACK"
    if phase in {ExperimentPhase.INTEGRITY_BEGIN, ExperimentPhase.INTEGRITY_END}:
        return "CHECKPOINT", "STATE_CHECKSUM"
    if phase in {ExperimentPhase.D2H_BEGIN, ExperimentPhase.D2H_END}:
        return "BRANCH_MIGRATION", "STATE_SEND"
    if phase in {ExperimentPhase.H2D_BEGIN, ExperimentPhase.H2D_END}:
        return "BRANCH_MIGRATION", "STATE_RECEIVE"
    if phase is ExperimentPhase.STATE_PUBLISH:
        return "STATE_PUBLISH", "STATE_PUBLISH"
    if phase in {ExperimentPhase.GPU_STATE_RELEASE_BEGIN, ExperimentPhase.GPU_STATE_RELEASE_END}:
        return "STATE_FREE", "STATE_FREE"
    if phase is ExperimentPhase.HBM_RECLAIM_CONFIRMED:
        return "STATE_RECLAIM", "STATE_RECLAIM"
    if phase in {ExperimentPhase.STATE_IMPORT_BEGIN, ExperimentPhase.STATE_IMPORT_END}:
        return "STATE_MAP", "STATE_WRITE"
    if phase in {ExperimentPhase.STATE_VALIDATE_BEGIN, ExperimentPhase.STATE_VALIDATE_END}:
        return "CHECKPOINT", "STATE_CHECKSUM"
    return "BRANCH_MIGRATION", "STATE_COMMIT"


__all__ = [
    "REQUIRED_PRESERVATION_PHASES",
    "BranchGroupSemanticEvidence",
    "BranchSemanticRecord",
    "CriticalStage",
    "CriticalStageRecord",
    "CriticalTimeline",
    "ExperimentPhase",
    "ExperimentPhaseMarker",
    "PilotValidityEvidence",
    "ReclamationMode",
    "ReclamationTransaction",
    "ReclamationTransactionState",
    "RuntimeAllocationIdentity",
    "RuntimeIncarnation",
    "TimelineKind",
    "allocation_is_fresh",
    "phase_trace_binding",
]
