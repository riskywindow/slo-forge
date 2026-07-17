"""Strict, versioned wire models for the SLOForge Helix learning loop.

The Helix IR deliberately carries policy identity and evidence provenance at
every boundary where off-policy data can enter training.  The models are
immutable, reject unknown fields, and validate cross-document lineage rather
than allowing runtimes to infer missing relationships.
"""

from __future__ import annotations

import hashlib
import json
import math
from enum import StrEnum
from itertools import pairwise
from typing import Annotated, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

SCHEMA_VERSION: Final = "1.0.0"
API_VERSION: Final = "sloforge.io/helix/v1"
U64_MAX: Final = (1 << 64) - 1

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256String = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
NonNegativeInt = Annotated[int, Field(ge=0, le=U64_MAX)]
PositiveInt = Annotated[int, Field(gt=0, le=U64_MAX)]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
NonNegativeFloat = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
PositiveFloat = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]
BehaviorLogProbability = Annotated[float, Field(le=0.0, allow_inf_nan=False)]


class HelixModel(BaseModel):
    """Immutable base for trusted Helix wire values."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class Digest(HelixModel):
    algorithm: Literal["sha256"] = "sha256"
    value: Sha256String


class EvidencePointer(HelixModel):
    uri: NonEmptyString
    digest: Digest
    media_type: NonEmptyString
    captured_at: NonEmptyString


class LineageRelation(StrEnum):
    SOURCE = "source"
    PARENT = "parent"
    DERIVED_FROM = "derived_from"
    EVIDENCE = "evidence"
    STATE = "state"
    POLICY = "policy"


class LineageReference(HelixModel):
    artifact_id: NonEmptyString
    artifact_kind: NonEmptyString
    relation: LineageRelation
    digest: Digest


def _canonical_model_hash(model: BaseModel) -> str:
    payload = json.dumps(
        model.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _policy_key(policy: PolicyEpoch) -> tuple[str, int, str]:
    return policy.policy_id, policy.epoch, policy.policy_digest.value


def _policy_lineage_id(policy: PolicyEpoch) -> str:
    return f"{policy.policy_id}@{policy.epoch}"


def _lineage_ids(lineage: tuple[LineageReference, ...]) -> set[str]:
    if not lineage:
        raise ValueError("lineage must not be empty")
    return {item.artifact_id for item in lineage}


def _require_lineage(lineage: tuple[LineageReference, ...], *artifact_ids: str) -> None:
    present = _lineage_ids(lineage)
    missing = [artifact_id for artifact_id in artifact_ids if artifact_id not in present]
    if missing:
        raise ValueError(f"lineage is missing required artifacts: {', '.join(missing)}")


class PolicyEpoch(HelixModel):
    api_version: Literal["sloforge.io/helix/v1"] = API_VERSION
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    kind: Literal["PolicyEpoch"] = "PolicyEpoch"
    policy_id: NonEmptyString
    epoch: NonNegativeInt
    policy_digest: Digest
    parent_epoch: NonNegativeInt | None = None
    parent_policy_digest: Digest | None = None
    training_transaction_id: NonEmptyString | None = None
    created_at: NonEmptyString
    lineage: tuple[LineageReference, ...]

    @model_validator(mode="after")
    def valid_parent(self) -> Self:
        _lineage_ids(self.lineage)
        if self.epoch == 0:
            if self.parent_epoch is not None or self.parent_policy_digest is not None:
                raise ValueError("epoch zero cannot declare a parent policy epoch")
        else:
            if self.parent_epoch is None or self.parent_policy_digest is None:
                raise ValueError("nonzero policy epochs require complete parent identity")
            if self.parent_epoch >= self.epoch:
                raise ValueError("parent_epoch must precede epoch")
            _require_lineage(self.lineage, f"{self.policy_id}@{self.parent_epoch}")
        return self


class EnvironmentStateCapsule(HelixModel):
    api_version: Literal["sloforge.io/helix/v1"] = API_VERSION
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    kind: Literal["EnvironmentStateCapsule"] = "EnvironmentStateCapsule"
    capsule_id: NonEmptyString
    environment_id: NonEmptyString
    captured_at: NonEmptyString
    policy_epoch: PolicyEpoch
    state_schema_digest: Digest
    state_digest: Digest
    compatibility_fingerprint: Digest
    payload_uri: NonEmptyString
    payload_media_type: NonEmptyString
    payload_byte_length: NonNegativeInt
    compatible_policy_digests: tuple[Digest, ...]
    lineage: tuple[LineageReference, ...]

    @model_validator(mode="after")
    def complete_compatibility_declaration(self) -> Self:
        if not self.compatible_policy_digests:
            raise ValueError("compatible_policy_digests must not be empty")
        values = [digest.value for digest in self.compatible_policy_digests]
        if len(set(values)) != len(values):
            raise ValueError("compatible_policy_digests contains duplicates")
        if self.policy_epoch.policy_digest.value not in values:
            raise ValueError("capturing policy digest must be declared compatible")
        _require_lineage(self.lineage, _policy_lineage_id(self.policy_epoch))
        return self


class BranchPoint(HelixModel):
    api_version: Literal["sloforge.io/helix/v1"] = API_VERSION
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    kind: Literal["BranchPoint"] = "BranchPoint"
    branch_point_id: NonEmptyString
    source_trajectory_id: NonEmptyString
    event_index: NonNegativeInt
    token_index: NonNegativeInt
    environment_state: EnvironmentStateCapsule
    policy_epoch: PolicyEpoch
    prefix_digest: Digest
    seed: NonNegativeInt
    created_at: NonEmptyString
    reason: NonEmptyString
    candidate_labels: tuple[NonEmptyString, ...]
    lineage: tuple[LineageReference, ...]

    @model_validator(mode="after")
    def consistent_source(self) -> Self:
        if _policy_key(self.policy_epoch) != _policy_key(self.environment_state.policy_epoch):
            raise ValueError("branch policy epoch must match captured environment state")
        if len(set(self.candidate_labels)) != len(self.candidate_labels):
            raise ValueError("candidate_labels contains duplicates")
        _require_lineage(
            self.lineage,
            self.source_trajectory_id,
            self.environment_state.capsule_id,
            _policy_lineage_id(self.policy_epoch),
        )
        return self


class TrajectoryEventKind(StrEnum):
    PROMPT_TOKEN = "prompt_token"
    GENERATED_TOKEN = "generated_token"
    ACTION = "action"
    OBSERVATION = "observation"
    TOOL_RESULT = "tool_result"
    TERMINAL = "terminal"


class TrajectoryTerminalStatus(StrEnum):
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERRORED = "errored"
    TRUNCATED = "truncated"


class PolicyConsistency(StrEnum):
    STRICT = "strict"
    SEGMENTED = "segmented"


class TrajectoryEvent(HelixModel):
    event_id: NonEmptyString
    event_index: NonNegativeInt
    kind: TrajectoryEventKind
    policy_epoch: PolicyEpoch
    payload_digest: Digest
    source_evidence: EvidencePointer


class TokenProvenance(HelixModel):
    token_id: NonEmptyString
    token_index: NonNegativeInt
    event_id: NonEmptyString
    token_value: NonNegativeInt
    policy_epoch: PolicyEpoch
    behavior_log_probability: BehaviorLogProbability
    sampler_seed: NonNegativeInt
    raw_sample: EvidencePointer


class ActionProvenance(HelixModel):
    action_id: NonEmptyString
    action_index: NonNegativeInt
    event_id: NonEmptyString
    action_type: NonEmptyString
    policy_epoch: PolicyEpoch
    behavior_log_probability: BehaviorLogProbability
    arguments_digest: Digest
    raw_sample: EvidencePointer


class TrajectorySegment(HelixModel):
    segment_id: NonEmptyString
    start_event_index: NonNegativeInt
    end_event_index_exclusive: PositiveInt
    policy_epoch: PolicyEpoch
    segment_evidence: EvidencePointer

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.end_event_index_exclusive <= self.start_event_index:
            raise ValueError("trajectory segment must contain at least one event")
        return self


class TrajectoryCapsule(HelixModel):
    api_version: Literal["sloforge.io/helix/v1"] = API_VERSION
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    kind: Literal["TrajectoryCapsule"] = "TrajectoryCapsule"
    trajectory_id: NonEmptyString
    branch_point_id: NonEmptyString | None = None
    parent_trajectory_id: NonEmptyString | None = None
    source_state_capsule_id: NonEmptyString
    environment_id: NonEmptyString
    policy_consistency: PolicyConsistency
    policy_epochs: tuple[PolicyEpoch, ...]
    segments: tuple[TrajectorySegment, ...]
    events: tuple[TrajectoryEvent, ...]
    tokens: tuple[TokenProvenance, ...]
    actions: tuple[ActionProvenance, ...]
    terminal_status: TrajectoryTerminalStatus
    started_at: NonEmptyString
    completed_at: NonEmptyString
    trace_evidence: EvidencePointer
    lineage: tuple[LineageReference, ...]

    @model_validator(mode="after")
    def complete_policy_and_event_provenance(self) -> Self:
        if not self.events:
            raise ValueError("trajectory must contain at least one event")
        event_ids = [event.event_id for event in self.events]
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("trajectory event IDs must be unique")
        if [event.event_index for event in self.events] != list(range(len(self.events))):
            raise ValueError("trajectory event indexes must be contiguous from zero")
        if not self.segments:
            raise ValueError("trajectory segments must not be empty")
        cursor = 0
        for segment in self.segments:
            if segment.start_event_index != cursor:
                raise ValueError("trajectory segments must be contiguous")
            cursor = segment.end_event_index_exclusive
        if cursor != len(self.events):
            raise ValueError("trajectory segments must cover every event exactly once")
        segment_keys = [_policy_key(segment.policy_epoch) for segment in self.segments]
        if self.policy_consistency is PolicyConsistency.STRICT and len(self.segments) != 1:
            raise ValueError("strict trajectories require exactly one policy segment")
        if self.policy_consistency is PolicyConsistency.SEGMENTED:
            for left, right in pairwise(segment_keys):
                if left == right:
                    raise ValueError("adjacent segmented policy epochs must differ")
        ordered_keys = list(dict.fromkeys(segment_keys))
        if [_policy_key(epoch) for epoch in self.policy_epochs] != ordered_keys:
            raise ValueError("policy_epochs must list segment policies in first-use order")

        event_by_id = {event.event_id: event for event in self.events}
        segment_by_event: dict[int, TrajectorySegment] = {}
        for segment in self.segments:
            for index in range(segment.start_event_index, segment.end_event_index_exclusive):
                segment_by_event[index] = segment
        for event in self.events:
            if _policy_key(event.policy_epoch) != _policy_key(
                segment_by_event[event.event_index].policy_epoch
            ):
                raise ValueError(f"event {event.event_id} policy epoch violates its segment")

        token_ids = [token.token_id for token in self.tokens]
        if len(set(token_ids)) != len(token_ids):
            raise ValueError("trajectory token IDs must be unique")
        if [token.token_index for token in self.tokens] != list(range(len(self.tokens))):
            raise ValueError("trajectory token indexes must be contiguous from zero")
        for token in self.tokens:
            token_event = event_by_id.get(token.event_id)
            if token_event is None:
                raise ValueError(f"token {token.token_id} references an unknown event")
            if token_event.kind not in {
                TrajectoryEventKind.PROMPT_TOKEN,
                TrajectoryEventKind.GENERATED_TOKEN,
            }:
                raise ValueError(f"token {token.token_id} must reference a token event")
            if _policy_key(token.policy_epoch) != _policy_key(token_event.policy_epoch):
                raise ValueError(f"token {token.token_id} policy epoch does not match its event")

        action_ids = [action.action_id for action in self.actions]
        if len(set(action_ids)) != len(action_ids):
            raise ValueError("trajectory action IDs must be unique")
        if [action.action_index for action in self.actions] != list(range(len(self.actions))):
            raise ValueError("trajectory action indexes must be contiguous from zero")
        for action in self.actions:
            action_event = event_by_id.get(action.event_id)
            if action_event is None:
                raise ValueError(f"action {action.action_id} references an unknown event")
            if action_event.kind is not TrajectoryEventKind.ACTION:
                raise ValueError(f"action {action.action_id} must reference an action event")
            if _policy_key(action.policy_epoch) != _policy_key(action_event.policy_epoch):
                raise ValueError(f"action {action.action_id} policy epoch does not match its event")
        if not self.tokens and not self.actions:
            raise ValueError("trajectory must contain token or action provenance")

        required = [self.source_state_capsule_id]
        if self.branch_point_id is not None:
            required.append(self.branch_point_id)
        if self.parent_trajectory_id is not None:
            required.append(self.parent_trajectory_id)
        required.extend(_policy_lineage_id(epoch) for epoch in self.policy_epochs)
        _require_lineage(self.lineage, *required)
        return self


class BranchGroup(HelixModel):
    api_version: Literal["sloforge.io/helix/v1"] = API_VERSION
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    kind: Literal["BranchGroup"] = "BranchGroup"
    group_id: NonEmptyString
    branch_point: BranchPoint
    trajectories: tuple[TrajectoryCapsule, ...]
    baseline_trajectory_id: NonEmptyString
    created_at: NonEmptyString
    lineage: tuple[LineageReference, ...]

    @model_validator(mode="after")
    def valid_branches(self) -> Self:
        if len(self.trajectories) < 2:
            raise ValueError("branch groups require at least two trajectories")
        identifiers = [item.trajectory_id for item in self.trajectories]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("branch trajectory IDs must be unique")
        if self.baseline_trajectory_id not in identifiers:
            raise ValueError("baseline_trajectory_id must identify a group trajectory")
        for trajectory in self.trajectories:
            if trajectory.branch_point_id != self.branch_point.branch_point_id:
                raise ValueError("every branch trajectory must reference the group branch point")
            if trajectory.environment_id != self.branch_point.environment_state.environment_id:
                raise ValueError("branch trajectory environment does not match captured state")
            if trajectory.source_state_capsule_id != self.branch_point.environment_state.capsule_id:
                raise ValueError("branch trajectory does not descend from captured state")
        _require_lineage(
            self.lineage,
            self.branch_point.branch_point_id,
            *(trajectory.trajectory_id for trajectory in self.trajectories),
        )
        return self


class RewardComponent(HelixModel):
    component_id: NonEmptyString
    name: NonEmptyString
    value: FiniteFloat
    weight: FiniteFloat
    policy_epoch: PolicyEpoch
    event_ids: tuple[NonEmptyString, ...]
    raw_evidence: EvidencePointer

    @model_validator(mode="after")
    def scoped(self) -> Self:
        if not self.event_ids:
            raise ValueError("reward components must identify source events")
        if len(set(self.event_ids)) != len(self.event_ids):
            raise ValueError("reward component event_ids contains duplicates")
        return self


class RewardEvidence(HelixModel):
    api_version: Literal["sloforge.io/helix/v1"] = API_VERSION
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    kind: Literal["RewardEvidence"] = "RewardEvidence"
    reward_evidence_id: NonEmptyString
    trajectory_id: NonEmptyString
    trajectory_digest: Digest
    policy_epochs: tuple[PolicyEpoch, ...]
    components: tuple[RewardComponent, ...]
    aggregate_reward: FiniteFloat
    evaluator_digest: Digest
    evaluated_at: NonEmptyString
    lineage: tuple[LineageReference, ...]

    @model_validator(mode="after")
    def valid_aggregate(self) -> Self:
        if not self.components:
            raise ValueError("reward evidence must contain components")
        component_ids = [component.component_id for component in self.components]
        if len(set(component_ids)) != len(component_ids):
            raise ValueError("reward component IDs must be unique")
        expected = math.fsum(component.value * component.weight for component in self.components)
        if not math.isclose(self.aggregate_reward, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("aggregate_reward does not equal weighted reward components")
        ordered = list(dict.fromkeys(_policy_key(item.policy_epoch) for item in self.components))
        if [_policy_key(epoch) for epoch in self.policy_epochs] != ordered:
            raise ValueError("reward policy_epochs must match component policies")
        _require_lineage(
            self.lineage,
            self.trajectory_id,
            *(_policy_lineage_id(epoch) for epoch in self.policy_epochs),
        )
        return self


class CreditSubjectKind(StrEnum):
    EVENT = "event"
    TOKEN = "token"
    ACTION = "action"


class CreditAssignment(HelixModel):
    assignment_id: NonEmptyString
    subject_kind: CreditSubjectKind
    subject_id: NonEmptyString
    event_id: NonEmptyString
    reward_component_id: NonEmptyString
    policy_epoch: PolicyEpoch
    behavior_log_probability: BehaviorLogProbability
    credit: FiniteFloat


class CreditAssignmentEvidence(HelixModel):
    api_version: Literal["sloforge.io/helix/v1"] = API_VERSION
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    kind: Literal["CreditAssignmentEvidence"] = "CreditAssignmentEvidence"
    credit_evidence_id: NonEmptyString
    trajectory_id: NonEmptyString
    trajectory_digest: Digest
    reward_evidence_id: NonEmptyString
    reward_evidence_digest: Digest
    method: NonEmptyString
    policy_epochs: tuple[PolicyEpoch, ...]
    assignments: tuple[CreditAssignment, ...]
    total_credit: FiniteFloat
    generated_at: NonEmptyString
    lineage: tuple[LineageReference, ...]

    @model_validator(mode="after")
    def complete_assignments(self) -> Self:
        if not self.assignments:
            raise ValueError("credit assignment evidence must contain assignments")
        identifiers = [assignment.assignment_id for assignment in self.assignments]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("credit assignment IDs must be unique")
        expected = math.fsum(assignment.credit for assignment in self.assignments)
        if not math.isclose(self.total_credit, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("total_credit does not equal assignment credits")
        ordered = list(dict.fromkeys(_policy_key(item.policy_epoch) for item in self.assignments))
        if [_policy_key(epoch) for epoch in self.policy_epochs] != ordered:
            raise ValueError("credit policy_epochs must match assignment policies")
        _require_lineage(
            self.lineage,
            self.trajectory_id,
            self.reward_evidence_id,
            *(_policy_lineage_id(epoch) for epoch in self.policy_epochs),
        )
        return self


class StalenessDisposition(StrEnum):
    ACCEPT = "accept"
    REWEIGHT = "reweight"
    REJECT = "reject"


class StalenessReport(HelixModel):
    api_version: Literal["sloforge.io/helix/v1"] = API_VERSION
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    kind: Literal["StalenessReport"] = "StalenessReport"
    report_id: NonEmptyString
    sample_id: NonEmptyString
    trajectory_id: NonEmptyString
    behavior_policy_epoch: PolicyEpoch
    learner_policy_epoch: PolicyEpoch
    epoch_lag: NonNegativeInt
    maximum_allowed_lag: NonNegativeInt
    stale: bool
    disposition: StalenessDisposition
    importance_sampling_weight: PositiveFloat | None = None
    assessed_at: NonEmptyString
    lineage: tuple[LineageReference, ...]

    @model_validator(mode="after")
    def valid_lag_and_disposition(self) -> Self:
        if self.behavior_policy_epoch.policy_id != self.learner_policy_epoch.policy_id:
            raise ValueError("staleness cannot compare different policy identities")
        if self.behavior_policy_epoch.epoch > self.learner_policy_epoch.epoch:
            raise ValueError("behavior policy epoch cannot be newer than learner policy")
        expected_lag = self.learner_policy_epoch.epoch - self.behavior_policy_epoch.epoch
        if self.epoch_lag != expected_lag:
            raise ValueError("epoch_lag does not match behavior and learner epochs")
        if self.stale != (self.epoch_lag > self.maximum_allowed_lag):
            raise ValueError("stale flag does not match maximum_allowed_lag")
        if self.disposition is StalenessDisposition.ACCEPT:
            if self.stale or self.importance_sampling_weight is not None:
                raise ValueError("accepted fresh samples must not be reweighted")
        elif self.disposition is StalenessDisposition.REWEIGHT:
            if not self.stale or self.importance_sampling_weight is None:
                raise ValueError("reweighted samples require staleness and an importance weight")
        elif not self.stale or self.importance_sampling_weight is not None:
            raise ValueError("rejected samples must be stale and unweighted")
        _require_lineage(
            self.lineage,
            self.sample_id,
            self.trajectory_id,
            _policy_lineage_id(self.behavior_policy_epoch),
            _policy_lineage_id(self.learner_policy_epoch),
        )
        return self


class StateReuseMode(StrEnum):
    EXACT = "exact"
    CONVERTED = "converted"
    RECOMPUTED = "recomputed"
    INCOMPATIBLE = "incompatible"


class StateReuseReport(HelixModel):
    api_version: Literal["sloforge.io/helix/v1"] = API_VERSION
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    kind: Literal["StateReuseReport"] = "StateReuseReport"
    report_id: NonEmptyString
    source_capsule: EnvironmentStateCapsule
    target_environment_id: NonEmptyString
    target_policy_epoch: PolicyEpoch
    target_compatibility_fingerprint: Digest
    mode: StateReuseMode
    compatible: bool
    reused: bool
    conversion_evidence: EvidencePointer | None = None
    reason: NonEmptyString
    assessed_at: NonEmptyString
    lineage: tuple[LineageReference, ...]

    @model_validator(mode="after")
    def no_silent_incompatible_reuse(self) -> Self:
        declared_policy = self.target_policy_epoch.policy_digest.value in {
            item.value for item in self.source_capsule.compatible_policy_digests
        }
        same_fingerprint = (
            self.target_compatibility_fingerprint.value
            == self.source_capsule.compatibility_fingerprint.value
        )
        if self.mode is StateReuseMode.EXACT:
            if not (declared_policy and same_fingerprint and self.compatible and self.reused):
                raise ValueError(
                    "exact state reuse requires declared policy and matching fingerprint"
                )
            if self.conversion_evidence is not None:
                raise ValueError("exact state reuse cannot declare conversion evidence")
        elif self.mode is StateReuseMode.CONVERTED:
            if not (declared_policy and self.compatible and self.reused):
                raise ValueError("converted state reuse requires declared compatible policy")
            if self.conversion_evidence is None:
                raise ValueError("converted state reuse requires conversion evidence")
        else:
            if self.compatible or self.reused:
                raise ValueError("recomputed or incompatible state must not be marked reused")
            if self.conversion_evidence is not None:
                raise ValueError("non-reused state cannot declare conversion evidence")
        _require_lineage(
            self.lineage,
            self.source_capsule.capsule_id,
            _policy_lineage_id(self.target_policy_epoch),
        )
        return self


class TrainingSampleKind(StrEnum):
    TOKEN = "token"
    ACTION = "action"
    TRAJECTORY = "trajectory"


class TrainingSampleProvenance(HelixModel):
    sample_id: NonEmptyString
    sample_kind: TrainingSampleKind
    trajectory_id: NonEmptyString
    trajectory_digest: Digest
    reward_evidence_id: NonEmptyString
    reward_evidence_digest: Digest
    credit_evidence_id: NonEmptyString
    credit_evidence_digest: Digest
    event_ids: tuple[NonEmptyString, ...]
    token_ids: tuple[NonEmptyString, ...]
    action_ids: tuple[NonEmptyString, ...]
    behavior_policy_epoch: PolicyEpoch
    behavior_log_probability: BehaviorLogProbability
    target_policy_epoch: PolicyEpoch
    importance_sampling_weight: PositiveFloat
    raw_sample: EvidencePointer
    lineage: tuple[LineageReference, ...]

    @model_validator(mode="after")
    def complete_sample_source(self) -> Self:
        if not self.event_ids:
            raise ValueError("training samples must identify source events")
        if self.sample_kind is TrainingSampleKind.TOKEN:
            if not self.token_ids or self.action_ids:
                raise ValueError("token samples require tokens and cannot include actions")
        elif self.sample_kind is TrainingSampleKind.ACTION:
            if not self.action_ids or self.token_ids:
                raise ValueError("action samples require actions and cannot include tokens")
        elif self.token_ids or self.action_ids:
            raise ValueError("trajectory samples use event scope rather than token/action IDs")
        for identifiers, label in (
            (self.event_ids, "event_ids"),
            (self.token_ids, "token_ids"),
            (self.action_ids, "action_ids"),
        ):
            if len(set(identifiers)) != len(identifiers):
                raise ValueError(f"training sample {label} contains duplicates")
        _require_lineage(
            self.lineage,
            self.trajectory_id,
            self.reward_evidence_id,
            self.credit_evidence_id,
            _policy_lineage_id(self.behavior_policy_epoch),
            _policy_lineage_id(self.target_policy_epoch),
        )
        return self


class TrainingBatchManifest(HelixModel):
    api_version: Literal["sloforge.io/helix/v1"] = API_VERSION
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    kind: Literal["TrainingBatchManifest"] = "TrainingBatchManifest"
    batch_id: NonEmptyString
    policy_consistency: PolicyConsistency
    learner_policy_epoch: PolicyEpoch
    samples: tuple[TrainingSampleProvenance, ...]
    created_at: NonEmptyString
    lineage: tuple[LineageReference, ...]

    @model_validator(mode="after")
    def complete_batch_provenance(self) -> Self:
        if not self.samples:
            raise ValueError("training batch must contain samples")
        identifiers = [sample.sample_id for sample in self.samples]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("training sample IDs must be unique")
        learner = _policy_key(self.learner_policy_epoch)
        for sample in self.samples:
            if _policy_key(sample.target_policy_epoch) != learner:
                raise ValueError("every sample target epoch must equal the learner policy epoch")
        behavior_keys = {_policy_key(sample.behavior_policy_epoch) for sample in self.samples}
        if self.policy_consistency is PolicyConsistency.STRICT and len(behavior_keys) != 1:
            raise ValueError("strict training batches cannot mix behavior policy epochs")
        _require_lineage(
            self.lineage,
            *(sample.sample_id for sample in self.samples),
            _policy_lineage_id(self.learner_policy_epoch),
        )
        return self


class LearningTransactionState(StrEnum):
    CREATED = "created"
    EVIDENCE_VALIDATED = "evidence_validated"
    BATCH_ASSEMBLED = "batch_assembled"
    TRAINED = "trained"
    EVALUATED = "evaluated"
    COMMITTED = "committed"
    ABORTED = "aborted"


class LearningStateTransition(HelixModel):
    sequence: PositiveInt
    from_state: LearningTransactionState | None
    to_state: LearningTransactionState
    transitioned_at: NonEmptyString
    actor: NonEmptyString
    reason: NonEmptyString
    evidence_digests: tuple[Digest, ...]


class PromotionDecision(StrEnum):
    PROMOTE = "promote"
    HOLD = "hold"
    ROLLBACK = "rollback"


class PolicyPromotionCapsule(HelixModel):
    api_version: Literal["sloforge.io/helix/v1"] = API_VERSION
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    kind: Literal["PolicyPromotionCapsule"] = "PolicyPromotionCapsule"
    promotion_id: NonEmptyString
    transaction_id: NonEmptyString
    from_policy_epoch: PolicyEpoch
    to_policy_epoch: PolicyEpoch
    decision: PromotionDecision
    evaluation_evidence: tuple[EvidencePointer, ...]
    approved_by: NonEmptyString
    promoted_at: NonEmptyString
    rollback_policy_epoch: PolicyEpoch | None = None
    lineage: tuple[LineageReference, ...]

    @model_validator(mode="after")
    def valid_epoch_transition(self) -> Self:
        if not self.evaluation_evidence:
            raise ValueError("promotion decisions require evaluation evidence")
        source = self.from_policy_epoch
        target = self.to_policy_epoch
        if source.policy_id != target.policy_id:
            raise ValueError("promotion cannot change policy identity")
        if self.decision is PromotionDecision.PROMOTE:
            if target.epoch <= source.epoch or self.rollback_policy_epoch is not None:
                raise ValueError("promotion requires a newer target and no rollback epoch")
        elif self.decision is PromotionDecision.HOLD:
            if _policy_key(target) != _policy_key(source) or self.rollback_policy_epoch is not None:
                raise ValueError("hold must retain the source policy epoch")
        else:
            if self.rollback_policy_epoch is None:
                raise ValueError("rollback decisions require rollback_policy_epoch")
            if self.rollback_policy_epoch.epoch >= source.epoch:
                raise ValueError("rollback policy must precede the source epoch")
        _require_lineage(
            self.lineage,
            self.transaction_id,
            _policy_lineage_id(source),
            _policy_lineage_id(target),
        )
        return self


_LEGAL_TRANSITIONS: Final = {
    LearningTransactionState.CREATED: {
        LearningTransactionState.EVIDENCE_VALIDATED,
        LearningTransactionState.ABORTED,
    },
    LearningTransactionState.EVIDENCE_VALIDATED: {
        LearningTransactionState.BATCH_ASSEMBLED,
        LearningTransactionState.ABORTED,
    },
    LearningTransactionState.BATCH_ASSEMBLED: {
        LearningTransactionState.TRAINED,
        LearningTransactionState.ABORTED,
    },
    LearningTransactionState.TRAINED: {
        LearningTransactionState.EVALUATED,
        LearningTransactionState.ABORTED,
    },
    LearningTransactionState.EVALUATED: {
        LearningTransactionState.COMMITTED,
        LearningTransactionState.ABORTED,
    },
    LearningTransactionState.COMMITTED: set(),
    LearningTransactionState.ABORTED: set(),
}


class LearningTransaction(HelixModel):
    api_version: Literal["sloforge.io/helix/v1"] = API_VERSION
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    kind: Literal["LearningTransaction"] = "LearningTransaction"
    transaction_id: NonEmptyString
    state: LearningTransactionState
    previous_state: LearningTransactionState | None
    transitioned_at: NonEmptyString
    transition_sequence: PositiveInt
    transitions: tuple[LearningStateTransition, ...]
    source_policy_epoch: PolicyEpoch
    candidate_policy_epoch: PolicyEpoch
    branch_group: BranchGroup
    reward_evidence: tuple[RewardEvidence, ...]
    credit_assignment_evidence: tuple[CreditAssignmentEvidence, ...]
    staleness_reports: tuple[StalenessReport, ...]
    state_reuse_reports: tuple[StateReuseReport, ...]
    training_batch: TrainingBatchManifest
    promotion: PolicyPromotionCapsule | None
    created_at: NonEmptyString
    lineage: tuple[LineageReference, ...]

    @model_validator(mode="after")
    def validate_complete_transaction(self) -> Self:
        self._validate_transitions()
        trajectories = {item.trajectory_id: item for item in self.branch_group.trajectories}
        rewards = {item.reward_evidence_id: item for item in self.reward_evidence}
        credits = {item.credit_evidence_id: item for item in self.credit_assignment_evidence}
        if len(rewards) != len(self.reward_evidence):
            raise ValueError("reward evidence IDs must be unique")
        if len(credits) != len(self.credit_assignment_evidence):
            raise ValueError("credit evidence IDs must be unique")
        if not rewards or not credits:
            raise ValueError("learning transactions require reward and credit evidence")

        for reward in rewards.values():
            trajectory = trajectories.get(reward.trajectory_id)
            if trajectory is None:
                raise ValueError("reward evidence references a trajectory outside the branch group")
            if reward.trajectory_digest.value != _canonical_model_hash(trajectory):
                raise ValueError("reward trajectory digest does not match embedded trajectory")
            trajectory_events = {item.event_id: item for item in trajectory.events}
            for component in reward.components:
                for event_id in component.event_ids:
                    event = trajectory_events.get(event_id)
                    if event is None:
                        raise ValueError("reward component references an unknown trajectory event")
                    if _policy_key(component.policy_epoch) != _policy_key(event.policy_epoch):
                        raise ValueError(
                            "reward component policy epoch does not match source event"
                        )

        for credit in credits.values():
            trajectory = trajectories.get(credit.trajectory_id)
            credit_reward = rewards.get(credit.reward_evidence_id)
            if trajectory is None or credit_reward is None:
                raise ValueError("credit evidence lineage is incomplete")
            if credit.trajectory_digest.value != _canonical_model_hash(trajectory):
                raise ValueError("credit trajectory digest does not match embedded trajectory")
            if credit.reward_evidence_digest.value != _canonical_model_hash(credit_reward):
                raise ValueError("credit reward digest does not match embedded reward evidence")
            if not math.isclose(
                credit.total_credit,
                credit_reward.aggregate_reward,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError("credit total must conserve aggregate reward")
            self._validate_credit_subjects(credit, trajectory, credit_reward)

        samples = {sample.sample_id: sample for sample in self.training_batch.samples}
        if {report.sample_id for report in self.staleness_reports} != set(samples):
            raise ValueError("every training sample requires exactly one staleness report")
        if len(self.staleness_reports) != len(samples):
            raise ValueError("staleness report sample IDs must be unique")
        for sample in samples.values():
            trajectory = trajectories.get(sample.trajectory_id)
            sample_reward = rewards.get(sample.reward_evidence_id)
            sample_credit = credits.get(sample.credit_evidence_id)
            if trajectory is None or sample_reward is None or sample_credit is None:
                raise ValueError("training sample lineage is incomplete")
            if sample.trajectory_digest.value != _canonical_model_hash(trajectory):
                raise ValueError("sample trajectory digest does not match embedded trajectory")
            if sample.reward_evidence_digest.value != _canonical_model_hash(sample_reward):
                raise ValueError("sample reward digest does not match embedded reward evidence")
            if sample.credit_evidence_digest.value != _canonical_model_hash(sample_credit):
                raise ValueError("sample credit digest does not match embedded credit evidence")
            self._validate_sample_source(sample, trajectory)
        for report in self.staleness_reports:
            sample = samples[report.sample_id]
            if report.trajectory_id != sample.trajectory_id:
                raise ValueError("staleness report trajectory does not match sample")
            if _policy_key(report.behavior_policy_epoch) != _policy_key(
                sample.behavior_policy_epoch
            ):
                raise ValueError("staleness behavior epoch does not match sample")
            if _policy_key(report.learner_policy_epoch) != _policy_key(
                self.training_batch.learner_policy_epoch
            ):
                raise ValueError("staleness learner epoch does not match training batch")

        if _policy_key(self.training_batch.learner_policy_epoch) != _policy_key(
            self.source_policy_epoch
        ):
            raise ValueError("training batch learner epoch must equal source policy epoch")
        if self.source_policy_epoch.policy_id != self.candidate_policy_epoch.policy_id:
            raise ValueError("candidate policy identity must equal source policy identity")
        if self.candidate_policy_epoch.epoch <= self.source_policy_epoch.epoch:
            raise ValueError("candidate policy epoch must be newer than source policy")
        if not self.state_reuse_reports:
            raise ValueError("learning transaction must explicitly report state reuse decisions")
        if self.state is LearningTransactionState.COMMITTED and (
            self.promotion is None or self.promotion.decision is not PromotionDecision.PROMOTE
        ):
            raise ValueError("committed transactions require a promotion decision")
        if self.promotion is not None:
            if self.promotion.transaction_id != self.transaction_id:
                raise ValueError("promotion transaction_id does not match transaction")
            if _policy_key(self.promotion.from_policy_epoch) != _policy_key(
                self.source_policy_epoch
            ) or _policy_key(self.promotion.to_policy_epoch) != _policy_key(
                self.candidate_policy_epoch
            ):
                raise ValueError("promotion epochs do not match transaction epochs")
        _require_lineage(
            self.lineage,
            self.branch_group.group_id,
            self.training_batch.batch_id,
            *(reward.reward_evidence_id for reward in self.reward_evidence),
            *(credit.credit_evidence_id for credit in self.credit_assignment_evidence),
        )
        return self

    def _validate_transitions(self) -> None:
        if not self.transitions:
            raise ValueError("learning transaction transition history must not be empty")
        current: LearningTransactionState | None = None
        for index, transition in enumerate(self.transitions, start=1):
            if transition.sequence != index:
                raise ValueError("transition sequences must be contiguous from one")
            if transition.from_state is not current:
                raise ValueError("transition from_state does not match preceding state")
            if current is None:
                if transition.to_state is not LearningTransactionState.CREATED:
                    raise ValueError("first transaction transition must create the transaction")
            elif transition.to_state not in _LEGAL_TRANSITIONS[current]:
                raise ValueError(f"illegal learning transaction transition: {current}")
            current = transition.to_state
        last = self.transitions[-1]
        if self.state is not last.to_state:
            raise ValueError("transaction state must equal the last transition target")
        if self.previous_state is not last.from_state:
            raise ValueError("previous_state must equal the last transition source")
        if self.transition_sequence != last.sequence:
            raise ValueError("transition_sequence must equal the last transition sequence")
        if self.transitioned_at != last.transitioned_at:
            raise ValueError("transitioned_at must equal the last transition timestamp")

    @staticmethod
    def _validate_credit_subjects(
        credit: CreditAssignmentEvidence,
        trajectory: TrajectoryCapsule,
        reward: RewardEvidence,
    ) -> None:
        events = {item.event_id: item for item in trajectory.events}
        tokens = {item.token_id: item for item in trajectory.tokens}
        actions = {item.action_id: item for item in trajectory.actions}
        component_ids = {item.component_id for item in reward.components}
        for assignment in credit.assignments:
            event = events.get(assignment.event_id)
            if event is None or assignment.reward_component_id not in component_ids:
                raise ValueError("credit assignment references unknown event or reward component")
            if assignment.subject_kind is CreditSubjectKind.EVENT:
                if assignment.subject_id != assignment.event_id:
                    raise ValueError("event credit subject must equal event_id")
                expected_epoch = event.policy_epoch
                expected_log_probability = 0.0
            elif assignment.subject_kind is CreditSubjectKind.TOKEN:
                token = tokens.get(assignment.subject_id)
                if token is None or token.event_id != assignment.event_id:
                    raise ValueError("token credit assignment references unknown token")
                expected_epoch = token.policy_epoch
                expected_log_probability = token.behavior_log_probability
            else:
                action = actions.get(assignment.subject_id)
                if action is None or action.event_id != assignment.event_id:
                    raise ValueError("action credit assignment references unknown action")
                expected_epoch = action.policy_epoch
                expected_log_probability = action.behavior_log_probability
            if _policy_key(assignment.policy_epoch) != _policy_key(expected_epoch):
                raise ValueError("credit assignment policy epoch does not match subject")
            if not math.isclose(
                assignment.behavior_log_probability,
                expected_log_probability,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "credit assignment behavior log probability does not match subject"
                )

    @staticmethod
    def _validate_sample_source(
        sample: TrainingSampleProvenance, trajectory: TrajectoryCapsule
    ) -> None:
        events = {item.event_id: item for item in trajectory.events}
        tokens = {item.token_id: item for item in trajectory.tokens}
        actions = {item.action_id: item for item in trajectory.actions}
        if any(event_id not in events for event_id in sample.event_ids):
            raise ValueError("training sample references an unknown event")
        policy_keys: list[tuple[str, int, str]] = []
        log_probabilities: list[float] = []
        if sample.sample_kind is TrainingSampleKind.TOKEN:
            for token_id in sample.token_ids:
                token = tokens.get(token_id)
                if token is None or token.event_id not in sample.event_ids:
                    raise ValueError("training sample references an unknown or out-of-scope token")
                policy_keys.append(_policy_key(token.policy_epoch))
                log_probabilities.append(token.behavior_log_probability)
        elif sample.sample_kind is TrainingSampleKind.ACTION:
            for action_id in sample.action_ids:
                action = actions.get(action_id)
                if action is None or action.event_id not in sample.event_ids:
                    raise ValueError("training sample references an unknown or out-of-scope action")
                policy_keys.append(_policy_key(action.policy_epoch))
                log_probabilities.append(action.behavior_log_probability)
        else:
            for event_id in sample.event_ids:
                event = events[event_id]
                policy_keys.append(_policy_key(event.policy_epoch))
            for token in trajectory.tokens:
                if token.event_id in sample.event_ids:
                    log_probabilities.append(token.behavior_log_probability)
            for action in trajectory.actions:
                if action.event_id in sample.event_ids:
                    log_probabilities.append(action.behavior_log_probability)
        if not log_probabilities:
            raise ValueError("training sample has no behavior log probability provenance")
        behavior_key = _policy_key(sample.behavior_policy_epoch)
        if any(key != behavior_key for key in policy_keys):
            raise ValueError("one training sample cannot cross behavior policy epochs")
        expected = math.fsum(log_probabilities)
        if not math.isclose(
            sample.behavior_log_probability, expected, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError("sample behavior log probability does not match source provenance")


class BranchWorkloadStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BranchWorkloadRequest(HelixModel):
    request_id: NonEmptyString
    branch_point_id: NonEmptyString
    trajectory_id: NonEmptyString
    ordinal: NonNegativeInt
    scheduled_offset_ms: NonNegativeFloat
    input_digest: Digest
    output_digest: Digest
    status: BranchWorkloadStatus


class BranchWorkloadTrace(HelixModel):
    api_version: Literal["sloforge.io/helix/v1"] = API_VERSION
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    kind: Literal["BranchWorkloadTrace"] = "BranchWorkloadTrace"
    trace_id: NonEmptyString
    branch_group_id: NonEmptyString
    environment_id: NonEmptyString
    seed: NonNegativeInt
    started_at: NonEmptyString
    completed_at: NonEmptyString
    requests: tuple[BranchWorkloadRequest, ...]
    raw_trace_uri: NonEmptyString
    raw_trace_digest: Digest
    lineage: tuple[LineageReference, ...]

    @model_validator(mode="after")
    def complete_trace(self) -> Self:
        if not self.requests:
            raise ValueError("branch workload trace must contain requests")
        identifiers = [request.request_id for request in self.requests]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("workload request IDs must be unique")
        if [request.ordinal for request in self.requests] != list(range(len(self.requests))):
            raise ValueError("workload request ordinals must be contiguous from zero")
        if any(request.branch_point_id == "" for request in self.requests):
            raise ValueError("workload request branch point must not be empty")
        _require_lineage(
            self.lineage,
            self.branch_group_id,
            *(request.trajectory_id for request in self.requests),
        )
        return self


HelixDocument = (
    PolicyEpoch
    | EnvironmentStateCapsule
    | BranchPoint
    | TrajectoryCapsule
    | BranchGroup
    | RewardEvidence
    | CreditAssignmentEvidence
    | StalenessReport
    | StateReuseReport
    | TrainingBatchManifest
    | LearningTransaction
    | PolicyPromotionCapsule
    | BranchWorkloadTrace
)


__all__ = [
    "API_VERSION",
    "SCHEMA_VERSION",
    "ActionProvenance",
    "BranchGroup",
    "BranchPoint",
    "BranchWorkloadRequest",
    "BranchWorkloadStatus",
    "BranchWorkloadTrace",
    "CreditAssignment",
    "CreditAssignmentEvidence",
    "CreditSubjectKind",
    "Digest",
    "EnvironmentStateCapsule",
    "EvidencePointer",
    "HelixDocument",
    "LearningStateTransition",
    "LearningTransaction",
    "LearningTransactionState",
    "LineageReference",
    "LineageRelation",
    "PolicyConsistency",
    "PolicyEpoch",
    "PolicyPromotionCapsule",
    "PromotionDecision",
    "RewardComponent",
    "RewardEvidence",
    "StalenessDisposition",
    "StalenessReport",
    "StateReuseMode",
    "StateReuseReport",
    "TokenProvenance",
    "TrainingBatchManifest",
    "TrainingSampleKind",
    "TrainingSampleProvenance",
    "TrajectoryCapsule",
    "TrajectoryEvent",
    "TrajectoryEventKind",
    "TrajectorySegment",
    "TrajectoryTerminalStatus",
]
