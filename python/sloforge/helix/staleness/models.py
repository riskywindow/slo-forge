"""Strict policy provenance and staleness decision wire models for Helix."""

from __future__ import annotations

import math
from enum import StrEnum
from itertools import pairwise
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from sloforge.continuum.ir import ExactnessClass

MAX_SEGMENTS = 128
MAX_RECORDS = 16_384
U64_MAX = 2**64 - 1

Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:@/-]{0,255}$")]
Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
NonNegativeInt = Annotated[int, Field(ge=0, le=U64_MAX)]
FiniteNonNegative = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
FinitePositive = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]
LogProbability = Annotated[float, Field(le=0.0, allow_inf_nan=False)]


class StalenessModel(BaseModel):
    """Immutable strict value used at the rollout/training trust boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class PolicySemantics(StrEnum):
    STRICT = "strict"
    SEGMENTED = "segmented"


class SampleKind(StrEnum):
    TOKEN = "token"
    ACTION = "action"


class LogProbabilitySource(StrEnum):
    RECORDED = "recorded"
    RECOMPUTED = "recomputed"
    MISSING = "missing"


class LogProbabilityRole(StrEnum):
    BEHAVIOR = "behavior"
    TARGET = "target"


class StalenessDisposition(StrEnum):
    HARD_REJECT = "hard_reject"
    BOUNDED_ACCEPT = "bounded_accept"
    RECOMPUTE_LOGPROBS = "recompute_logprobs"
    IMPORTANCE_CLIP = "importance_clip"
    TRUNCATE = "truncate"
    RESAMPLE = "resample"
    PRIORITY_REDUCTION = "priority_reduction"
    EVALUATION_ONLY = "evaluation_only"


class TrainingEligibility(StrEnum):
    ELIGIBLE = "eligible"
    ELIGIBLE_WITH_CORRECTION = "eligible_with_correction"
    PARTIALLY_ELIGIBLE = "partially_eligible"
    INELIGIBLE = "ineligible"


class ReasonCode(StrEnum):
    MIXED_POLICY = "mixed_policy"
    POLICY_ID_MISMATCH = "policy_id_mismatch"
    FUTURE_BEHAVIOR_POLICY = "future_behavior_policy"
    FUTURE_COLLECTION_TIME = "future_collection_time"
    MISSING_BEHAVIOR_LOGPROB = "missing_behavior_logprob"
    MISSING_TARGET_LOGPROB = "missing_target_logprob"
    TARGET_POLICY_MISMATCH = "target_policy_mismatch"
    INCOMPATIBLE_TRANSITION = "incompatible_transition"
    INCOMPLETE_STATE_RECOMPUTATION = "incomplete_state_recomputation"
    UNDECLARED_STATE_RECOMPUTATION = "undeclared_state_recomputation"
    MISSING_DISTANCE_EVIDENCE = "missing_distance_evidence"
    EPOCH_DISTANCE_EXCEEDED = "epoch_distance_exceeded"
    UPDATE_DISTANCE_EXCEEDED = "update_distance_exceeded"
    WALL_AGE_EXCEEDED = "wall_age_exceeded"
    PARAMETER_DELTA_EXCEEDED = "parameter_delta_exceeded"
    KL_DIVERGENCE_EXCEEDED = "kl_divergence_exceeded"
    IMPORTANCE_RATIO_EXCEEDED = "importance_ratio_exceeded"
    IMPORTANCE_RATIO_NONFINITE = "importance_ratio_nonfinite"
    IMPORTANCE_RATIO_CLIPPED = "importance_ratio_clipped"
    SEGMENT_TRUNCATED = "segment_truncated"
    RESAMPLE_REQUIRED = "resample_required"
    RESAMPLE_BOUND_EXCEEDED = "resample_bound_exceeded"
    PRIORITY_REDUCED = "priority_reduced"
    EVALUATION_ONLY = "evaluation_only"


class PolicyVersion(StalenessModel):
    """A policy version names both logical progress and immutable parameters."""

    policy_id: Identifier
    epoch: NonNegativeInt
    update: NonNegativeInt
    parameter_digest: Digest
    continuum_compatibility_fingerprint: Digest
    published_at_ms: NonNegativeInt

    @property
    def key(self) -> tuple[str, int, int, str, str]:
        return (
            self.policy_id,
            self.epoch,
            self.update,
            self.parameter_digest,
            self.continuum_compatibility_fingerprint,
        )


class IndexRange(StalenessModel):
    """A bounded half-open token or action range."""

    start: Annotated[int, Field(ge=0, le=MAX_RECORDS)]
    end_exclusive: Annotated[int, Field(ge=0, le=MAX_RECORDS)]

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.end_exclusive < self.start:
            raise ValueError("range end must not precede its start")
        return self

    def contains(self, index: int) -> bool:
        return self.start <= index < self.end_exclusive

    @property
    def length(self) -> int:
        return self.end_exclusive - self.start


class PolicySegment(StalenessModel):
    segment_id: Identifier
    segment_index: Annotated[int, Field(ge=0, lt=MAX_SEGMENTS)]
    policy: PolicyVersion
    token_range: IndexRange
    action_range: IndexRange
    collected_at_ms: NonNegativeInt
    completed_at_ms: NonNegativeInt

    @model_validator(mode="after")
    def valid_segment(self) -> Self:
        if self.token_range.length == 0 and self.action_range.length == 0:
            raise ValueError("a policy segment must contain a token or action")
        if self.completed_at_ms < self.collected_at_ms:
            raise ValueError("segment completion cannot precede collection")
        return self


class LogProbabilityRecomputeEvidence(StalenessModel):
    recomputation_id: Identifier
    subject_kind: SampleKind
    subject_index: Annotated[int, Field(ge=0, lt=MAX_RECORDS)]
    policy: PolicyVersion
    token_history_digest: Digest
    implementation_digest: Digest
    result_digest: Digest
    seed: NonNegativeInt
    recomputed_at_ms: NonNegativeInt


class DecisionLogProbability(StalenessModel):
    """Observed action/token probability; missing data remains explicit and typed."""

    sample_kind: SampleKind
    sample_index: Annotated[int, Field(ge=0, lt=MAX_RECORDS)]
    behavior_policy: PolicyVersion
    behavior_log_probability: LogProbability | None = None
    behavior_source: LogProbabilitySource = LogProbabilitySource.MISSING
    behavior_evidence_digest: Digest | None = None
    behavior_recomputation: LogProbabilityRecomputeEvidence | None = None
    target_policy: PolicyVersion | None = None
    target_log_probability: LogProbability | None = None
    target_source: LogProbabilitySource = LogProbabilitySource.MISSING
    target_evidence_digest: Digest | None = None
    target_recomputation: LogProbabilityRecomputeEvidence | None = None

    @model_validator(mode="after")
    def complete_probability_provenance(self) -> Self:
        self._validate_value(
            "behavior",
            self.behavior_policy,
            self.behavior_log_probability,
            self.behavior_source,
            self.behavior_evidence_digest,
            self.behavior_recomputation,
        )
        if self.target_policy is None:
            if (
                any(
                    value is not None
                    for value in (
                        self.target_log_probability,
                        self.target_evidence_digest,
                        self.target_recomputation,
                    )
                )
                or self.target_source is not LogProbabilitySource.MISSING
            ):
                raise ValueError("target log-probability evidence requires a target policy")
        else:
            self._validate_value(
                "target",
                self.target_policy,
                self.target_log_probability,
                self.target_source,
                self.target_evidence_digest,
                self.target_recomputation,
            )
        return self

    def _validate_value(
        self,
        label: str,
        policy: PolicyVersion,
        value: float | None,
        source: LogProbabilitySource,
        evidence_digest: str | None,
        recomputation: LogProbabilityRecomputeEvidence | None,
    ) -> None:
        if source is LogProbabilitySource.MISSING:
            if value is not None or evidence_digest is not None or recomputation is not None:
                raise ValueError(f"missing {label} log probability cannot carry evidence")
            return
        if value is None or evidence_digest is None:
            raise ValueError(f"{label} log probability requires a value and evidence digest")
        if source is LogProbabilitySource.RECORDED:
            if recomputation is not None:
                raise ValueError(f"recorded {label} log probability cannot claim recomputation")
            return
        if recomputation is None:
            raise ValueError(f"recomputed {label} log probability requires explicit evidence")
        if (
            recomputation.subject_kind is not self.sample_kind
            or recomputation.subject_index != self.sample_index
            or recomputation.policy != policy
            or recomputation.result_digest != evidence_digest
        ):
            raise ValueError(f"{label} log-probability recomputation evidence does not match")


class ContinuumCompatibilityEvidence(StalenessModel):
    """Immutable projection of the Continuum compatibility report used at a boundary."""

    report_id: Identifier
    report_digest: Digest
    compatibility_class: ExactnessClass
    safe: bool
    source_compatibility_fingerprint: Digest
    destination_compatibility_fingerprint: Digest
    required_recomputation: Annotated[tuple[Identifier, ...], Field(max_length=256)] = ()
    verification_obligations: Annotated[tuple[Identifier, ...], Field(max_length=256)] = ()

    @model_validator(mode="after")
    def consistent_decision(self) -> Self:
        if self.safe == (self.compatibility_class is ExactnessClass.INCOMPATIBLE):
            raise ValueError("Continuum evidence is safe exactly when it is not incompatible")
        if len(set(self.required_recomputation)) != len(self.required_recomputation):
            raise ValueError("Continuum recomputation requirements contain duplicates")
        return self


class StateRecomputeActionEvidence(StalenessModel):
    action_id: Identifier
    component_ids: Annotated[tuple[Identifier, ...], Field(min_length=1, max_length=256)]
    source: Literal["token_history", "checkpoint"]
    source_digest: Digest
    result_digest: Digest
    implementation_digest: Digest
    seed: NonNegativeInt
    completed_at_ms: NonNegativeInt

    @model_validator(mode="after")
    def unique_components(self) -> Self:
        if len(set(self.component_ids)) != len(self.component_ids):
            raise ValueError("a recompute action cannot name a component twice")
        return self


class TransitionBoundary(StalenessModel):
    boundary_id: Identifier
    from_segment_index: Annotated[int, Field(ge=0, lt=MAX_SEGMENTS)]
    to_segment_index: Annotated[int, Field(ge=0, lt=MAX_SEGMENTS)]
    token_index: Annotated[int, Field(ge=0, le=MAX_RECORDS)]
    action_index: Annotated[int, Field(ge=0, le=MAX_RECORDS)]
    from_policy: PolicyVersion
    to_policy: PolicyVersion
    compatibility: ContinuumCompatibilityEvidence
    recompute_actions: Annotated[
        tuple[StateRecomputeActionEvidence, ...], Field(max_length=256)
    ] = ()

    @model_validator(mode="after")
    def ordered_transition(self) -> Self:
        if self.to_segment_index != self.from_segment_index + 1:
            raise ValueError("a transition boundary must connect adjacent segments")
        identifiers = [action.action_id for action in self.recompute_actions]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("transition recompute action IDs must be unique")
        components = [
            component for action in self.recompute_actions for component in action.component_ids
        ]
        if len(set(components)) != len(components):
            raise ValueError("a transition component cannot be recomputed more than once")
        return self


class TrajectoryPolicyProvenance(StalenessModel):
    schema_version: Literal["sloforge.helix.policy-provenance/v1"] = (
        "sloforge.helix.policy-provenance/v1"
    )
    trajectory_id: Identifier
    semantics: PolicySemantics
    token_count: Annotated[int, Field(ge=0, le=MAX_RECORDS)]
    action_count: Annotated[int, Field(ge=0, le=MAX_RECORDS)]
    segments: Annotated[tuple[PolicySegment, ...], Field(min_length=1, max_length=MAX_SEGMENTS)]
    transitions: Annotated[tuple[TransitionBoundary, ...], Field(max_length=MAX_SEGMENTS - 1)] = ()
    log_probabilities: Annotated[
        tuple[DecisionLogProbability, ...], Field(min_length=1, max_length=MAX_RECORDS)
    ]
    trace_evidence_digest: Digest

    @model_validator(mode="after")
    def complete_ranges_and_policy_provenance(self) -> Self:
        if self.token_count == 0 and self.action_count == 0:
            raise ValueError("trajectory provenance must contain tokens or actions")
        if self.token_count + self.action_count > MAX_RECORDS:
            raise ValueError("combined token and action provenance exceeds the record bound")
        if [item.segment_index for item in self.segments] != list(range(len(self.segments))):
            raise ValueError("policy segment indexes must be contiguous from zero")
        if len({item.segment_id for item in self.segments}) != len(self.segments):
            raise ValueError("policy segment IDs must be unique")
        token_cursor = 0
        action_cursor = 0
        for segment in self.segments:
            if segment.token_range.start != token_cursor:
                raise ValueError("policy token ranges must be contiguous")
            if segment.action_range.start != action_cursor:
                raise ValueError("policy action ranges must be contiguous")
            token_cursor = segment.token_range.end_exclusive
            action_cursor = segment.action_range.end_exclusive
        if token_cursor != self.token_count or action_cursor != self.action_count:
            raise ValueError("policy segments must exactly cover all token and action indexes")

        if self.semantics is PolicySemantics.STRICT:
            if len(self.segments) != 1 or self.transitions:
                raise ValueError("strict policy semantics require one segment and no transition")
        else:
            if len(self.segments) < 2:
                raise ValueError("segmented policy semantics require at least two segments")
            for left, right in pairwise(self.segments):
                if left.policy == right.policy:
                    raise ValueError("adjacent segmented policy versions must differ")
            if len(self.transitions) != len(self.segments) - 1:
                raise ValueError("every segmented policy change requires one transition boundary")
            for index, boundary in enumerate(self.transitions):
                left = self.segments[index]
                right = self.segments[index + 1]
                if (
                    boundary.from_segment_index != index
                    or boundary.to_segment_index != index + 1
                    or boundary.token_index != left.token_range.end_exclusive
                    or boundary.token_index != right.token_range.start
                    or boundary.action_index != left.action_range.end_exclusive
                    or boundary.action_index != right.action_range.start
                    or boundary.from_policy != left.policy
                    or boundary.to_policy != right.policy
                ):
                    raise ValueError("transition boundary does not match adjacent policy ranges")
                if (
                    boundary.compatibility.source_compatibility_fingerprint
                    != left.policy.continuum_compatibility_fingerprint
                    or boundary.compatibility.destination_compatibility_fingerprint
                    != right.policy.continuum_compatibility_fingerprint
                ):
                    raise ValueError("transition compatibility evidence names the wrong policies")

        expected = {
            *((SampleKind.TOKEN, index) for index in range(self.token_count)),
            *((SampleKind.ACTION, index) for index in range(self.action_count)),
        }
        actual = {(item.sample_kind, item.sample_index) for item in self.log_probabilities}
        if len(actual) != len(self.log_probabilities) or actual != expected:
            raise ValueError("log-probability records must exactly cover tokens and actions")
        for record in self.log_probabilities:
            matching = next(
                (
                    segment
                    for segment in self.segments
                    if (
                        segment.token_range.contains(record.sample_index)
                        if record.sample_kind is SampleKind.TOKEN
                        else segment.action_range.contains(record.sample_index)
                    )
                ),
                None,
            )
            if matching is None or record.behavior_policy != matching.policy:
                raise ValueError("log-probability behavior policy violates its explicit range")
        return self


class PolicyDistanceEvidence(StalenessModel):
    behavior_policy: PolicyVersion
    learner_policy: PolicyVersion
    parameter_delta_l2: FiniteNonNegative
    kl_divergence: FiniteNonNegative
    sample_count: Annotated[int, Field(gt=0, le=1_000_000)]
    evidence_digest: Digest


class StalenessPolicy(StalenessModel):
    policy_id: Identifier
    max_epoch_distance: NonNegativeInt = 1
    max_update_distance: NonNegativeInt = 4
    max_wall_age_ms: NonNegativeInt = 300_000
    max_parameter_delta_l2: FiniteNonNegative = 1.0
    max_kl_divergence: FiniteNonNegative = 0.2
    minimum_importance_ratio: FinitePositive = 0.8
    maximum_importance_ratio: FinitePositive = 1.25
    clip_importance_ratio_minimum: FinitePositive = 0.8
    clip_importance_ratio_maximum: FinitePositive = 1.25
    stale_disposition: StalenessDisposition = StalenessDisposition.HARD_REJECT
    importance_disposition: StalenessDisposition = StalenessDisposition.IMPORTANCE_CLIP
    missing_logprob_disposition: StalenessDisposition = StalenessDisposition.RECOMPUTE_LOGPROBS
    missing_metric_disposition: StalenessDisposition = StalenessDisposition.EVALUATION_ONLY
    priority_reduction_factor: Annotated[float, Field(gt=0.0, lt=1.0, allow_inf_nan=False)] = 0.5
    allow_partial_training: bool = True
    max_resample_directives: Annotated[int, Field(ge=1, le=MAX_SEGMENTS)] = 16

    @model_validator(mode="after")
    def valid_actions_and_bounds(self) -> Self:
        if self.minimum_importance_ratio > self.maximum_importance_ratio:
            raise ValueError("importance-ratio acceptance bounds are reversed")
        if self.clip_importance_ratio_minimum > self.clip_importance_ratio_maximum:
            raise ValueError("importance-ratio clip bounds are reversed")
        if not (
            self.clip_importance_ratio_minimum <= self.minimum_importance_ratio
            and self.clip_importance_ratio_maximum >= self.maximum_importance_ratio
        ):
            raise ValueError("clip bounds must contain the acceptance bounds")
        if self.stale_disposition not in {
            StalenessDisposition.HARD_REJECT,
            StalenessDisposition.TRUNCATE,
            StalenessDisposition.RESAMPLE,
            StalenessDisposition.PRIORITY_REDUCTION,
            StalenessDisposition.EVALUATION_ONLY,
        }:
            raise ValueError("invalid exceeded-staleness disposition")
        if self.importance_disposition not in {
            StalenessDisposition.HARD_REJECT,
            StalenessDisposition.IMPORTANCE_CLIP,
            StalenessDisposition.TRUNCATE,
            StalenessDisposition.RESAMPLE,
            StalenessDisposition.PRIORITY_REDUCTION,
            StalenessDisposition.EVALUATION_ONLY,
        }:
            raise ValueError("invalid importance-ratio disposition")
        if self.missing_logprob_disposition not in {
            StalenessDisposition.HARD_REJECT,
            StalenessDisposition.RECOMPUTE_LOGPROBS,
            StalenessDisposition.EVALUATION_ONLY,
        }:
            raise ValueError("missing log probabilities require reject, recompute, or evaluation")
        if self.missing_metric_disposition not in {
            StalenessDisposition.HARD_REJECT,
            StalenessDisposition.EVALUATION_ONLY,
        }:
            raise ValueError("missing distance metrics require reject or evaluation-only")
        return self


class StalenessAssessmentRequest(StalenessModel):
    trajectory: TrajectoryPolicyProvenance
    learner_policy: PolicyVersion
    policy: StalenessPolicy
    distance_evidence: Annotated[
        tuple[PolicyDistanceEvidence, ...], Field(max_length=MAX_SEGMENTS)
    ] = ()
    assessed_at_ms: NonNegativeInt
    seed: NonNegativeInt

    @model_validator(mode="after")
    def unique_distance_evidence(self) -> Self:
        keys = [item.behavior_policy.key for item in self.distance_evidence]
        if len(set(keys)) != len(keys):
            raise ValueError("distance evidence must be unique per behavior policy version")
        if any(item.learner_policy != self.learner_policy for item in self.distance_evidence):
            raise ValueError("distance evidence must name the assessed learner policy")
        return self


class BoundedMetric(StalenessModel):
    value: FiniteNonNegative | None
    maximum: FiniteNonNegative
    available: bool
    exceeded: bool
    evidence_digest: Digest | None = None

    @model_validator(mode="after")
    def consistent_bound(self) -> Self:
        if self.available != (self.value is not None):
            raise ValueError("metric availability does not match its value")
        if self.exceeded != (self.value is not None and self.value > self.maximum):
            raise ValueError("metric exceeded flag does not match its bound")
        if not self.available and self.evidence_digest is not None:
            raise ValueError("an unavailable metric cannot carry evidence")
        return self


class DistributionSummary(StalenessModel):
    count: Annotated[int, Field(gt=0, le=MAX_RECORDS)]
    minimum: FinitePositive
    maximum: FinitePositive
    mean: FinitePositive
    p50: FinitePositive
    p95: FinitePositive
    p99: FinitePositive

    @model_validator(mode="after")
    def ordered_quantiles(self) -> Self:
        if not self.minimum <= self.p50 <= self.p95 <= self.p99 <= self.maximum:
            raise ValueError("importance-ratio quantiles are not ordered")
        return self


class SegmentMetrics(StalenessModel):
    epoch_distance: BoundedMetric
    update_distance: BoundedMetric
    wall_age_ms: BoundedMetric
    parameter_delta_l2: BoundedMetric
    kl_divergence: BoundedMetric
    importance_ratios: DistributionSummary | None = None


class ImportanceWeight(StalenessModel):
    sample_kind: SampleKind
    sample_index: Annotated[int, Field(ge=0, lt=MAX_RECORDS)]
    derivation: Literal["policy_identity", "explicit_log_probabilities"]
    behavior_policy: PolicyVersion
    target_policy: PolicyVersion
    behavior_log_probability: LogProbability
    target_log_probability: LogProbability | None
    behavior_evidence_digest: Digest
    target_evidence_digest: Digest | None
    raw_ratio: FinitePositive
    applied_ratio: FinitePositive
    clipped: bool

    @model_validator(mode="after")
    def valid_clip(self) -> Self:
        if self.clipped == math.isclose(
            self.raw_ratio, self.applied_ratio, rel_tol=0.0, abs_tol=0.0
        ):
            raise ValueError("importance-weight clipped flag does not match applied ratio")
        if self.derivation == "policy_identity":
            if self.behavior_policy != self.target_policy or self.raw_ratio != 1.0:
                raise ValueError(
                    "identity importance weights require one exact policy and ratio one"
                )
            if self.target_log_probability is not None or self.target_evidence_digest is not None:
                raise ValueError("identity importance weights must not invent target evidence")
        elif self.target_log_probability is None or self.target_evidence_digest is None:
            raise ValueError("off-policy importance weights require explicit target evidence")
        return self


class StalenessReason(StalenessModel):
    code: ReasonCode
    message: Annotated[str, StringConstraints(min_length=1, max_length=1024)]
    segment_index: Annotated[int, Field(ge=0, lt=MAX_SEGMENTS)] | None = None
    sample_kind: SampleKind | None = None
    sample_index: Annotated[int, Field(ge=0, lt=MAX_RECORDS)] | None = None

    @model_validator(mode="after")
    def complete_sample_scope(self) -> Self:
        if (self.sample_kind is None) != (self.sample_index is None):
            raise ValueError("reason sample scope requires both kind and index")
        return self


class SegmentStalenessReport(StalenessModel):
    segment_id: Identifier
    segment_index: Annotated[int, Field(ge=0, lt=MAX_SEGMENTS)]
    behavior_policy: PolicyVersion
    learner_policy: PolicyVersion
    on_policy: bool
    stale: bool
    disposition: StalenessDisposition
    collected_at_ms: NonNegativeInt
    completed_at_ms: NonNegativeInt
    metrics: SegmentMetrics
    importance_weights: Annotated[tuple[ImportanceWeight, ...], Field(max_length=MAX_RECORDS)]
    priority: Annotated[float, Field(gt=0.0, le=1.0, allow_inf_nan=False)]
    training_eligible: bool
    token_range: IndexRange
    action_range: IndexRange
    reasons: Annotated[tuple[StalenessReason, ...], Field(max_length=MAX_RECORDS)]

    @model_validator(mode="after")
    def valid_segment_decision(self) -> Self:
        if self.on_policy and self.behavior_policy != self.learner_policy:
            raise ValueError("an on-policy segment must exactly match the learner policy")
        if (
            self.disposition
            in {
                StalenessDisposition.HARD_REJECT,
                StalenessDisposition.RECOMPUTE_LOGPROBS,
                StalenessDisposition.TRUNCATE,
                StalenessDisposition.RESAMPLE,
                StalenessDisposition.EVALUATION_ONLY,
            }
            and self.training_eligible
        ):
            raise ValueError("non-training dispositions cannot be training eligible")
        if self.completed_at_ms < self.collected_at_ms:
            raise ValueError("reported segment completion cannot precede collection")
        if self.metrics.importance_ratios is not None and (
            self.metrics.importance_ratios.count != len(self.importance_weights)
        ):
            raise ValueError("segment importance distribution has incomplete sample provenance")
        return self


class TruncationDirective(StalenessModel):
    segment_id: Identifier
    token_range: IndexRange
    action_range: IndexRange
    reason: Literal["staleness_policy"] = "staleness_policy"


class ResampleDirective(StalenessModel):
    segment_id: Identifier
    token_range: IndexRange
    action_range: IndexRange
    seed: NonNegativeInt
    reason: Literal["staleness_policy"] = "staleness_policy"


class LogProbabilityRecomputeDirective(StalenessModel):
    segment_id: Identifier
    sample_kind: SampleKind
    sample_index: Annotated[int, Field(ge=0, lt=MAX_RECORDS)]
    role: LogProbabilityRole
    policy: PolicyVersion
    trace_evidence_digest: Digest
    seed: NonNegativeInt


class AggregateMetrics(StalenessModel):
    maximum_epoch_distance: FiniteNonNegative | None
    maximum_update_distance: FiniteNonNegative | None
    maximum_wall_age_ms: FiniteNonNegative | None
    maximum_parameter_delta_l2: FiniteNonNegative | None
    maximum_kl_divergence: FiniteNonNegative | None
    importance_ratios: DistributionSummary | None


class TrajectoryStalenessReport(StalenessModel):
    schema_version: Literal["sloforge.helix.staleness-report/v1"] = (
        "sloforge.helix.staleness-report/v1"
    )
    report_id: Digest
    trajectory_id: Identifier
    trajectory_trace_evidence_digest: Digest
    learner_policy: PolicyVersion
    semantics: PolicySemantics
    mixed_policy: bool
    on_policy: bool
    primary_disposition: StalenessDisposition
    applied_dispositions: Annotated[
        tuple[StalenessDisposition, ...], Field(min_length=1, max_length=8)
    ]
    training_eligibility: TrainingEligibility
    training_eligible: bool
    evaluation_eligible: Literal[True] = True
    performed_recomputation: Literal[False] = False
    segment_reports: Annotated[
        tuple[SegmentStalenessReport, ...], Field(min_length=1, max_length=MAX_SEGMENTS)
    ]
    metrics: AggregateMetrics
    eligible_token_ranges: Annotated[tuple[IndexRange, ...], Field(max_length=MAX_SEGMENTS)]
    eligible_action_ranges: Annotated[tuple[IndexRange, ...], Field(max_length=MAX_SEGMENTS)]
    truncations: Annotated[tuple[TruncationDirective, ...], Field(max_length=MAX_SEGMENTS)]
    resamples: Annotated[tuple[ResampleDirective, ...], Field(max_length=MAX_SEGMENTS)]
    logprob_recomputations: Annotated[
        tuple[LogProbabilityRecomputeDirective, ...], Field(max_length=MAX_RECORDS)
    ]
    reasons: Annotated[tuple[StalenessReason, ...], Field(max_length=MAX_RECORDS)]
    policy_id: Identifier
    assessed_at_ms: NonNegativeInt
    seed: NonNegativeInt

    @model_validator(mode="after")
    def valid_report(self) -> Self:
        if self.mixed_policy and self.on_policy:
            raise ValueError("a mixed-policy trajectory can never be on-policy")
        if self.on_policy and self.semantics is not PolicySemantics.STRICT:
            raise ValueError("only strict trajectories can be on-policy")
        if self.training_eligible != (
            self.training_eligibility is not TrainingEligibility.INELIGIBLE
        ):
            raise ValueError("training eligibility flag disagrees with typed eligibility")
        weight_count = sum(len(report.importance_weights) for report in self.segment_reports)
        if self.metrics.importance_ratios is not None and (
            self.metrics.importance_ratios.count != weight_count
        ):
            raise ValueError("aggregate importance distribution has incomplete sample provenance")
        expected = self.model_dump(mode="json", exclude={"report_id"})
        from .engine import canonical_digest

        if canonical_digest(expected) != self.report_id:
            raise ValueError("staleness report identifier does not match its content")
        return self


__all__ = [
    "AggregateMetrics",
    "BoundedMetric",
    "ContinuumCompatibilityEvidence",
    "DecisionLogProbability",
    "DistributionSummary",
    "IndexRange",
    "LogProbabilityRecomputeDirective",
    "LogProbabilityRecomputeEvidence",
    "LogProbabilityRole",
    "LogProbabilitySource",
    "PolicyDistanceEvidence",
    "PolicySegment",
    "PolicySemantics",
    "PolicyVersion",
    "ReasonCode",
    "ResampleDirective",
    "SampleKind",
    "SegmentMetrics",
    "SegmentStalenessReport",
    "StalenessAssessmentRequest",
    "StalenessDisposition",
    "StalenessPolicy",
    "StalenessReason",
    "StateRecomputeActionEvidence",
    "TrainingEligibility",
    "TrajectoryPolicyProvenance",
    "TrajectoryStalenessReport",
    "TransitionBoundary",
    "TruncationDirective",
]
