"""Deterministic, evidence-only staleness assessment for Helix trajectories."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from typing import Any, Literal

from sloforge.continuum.ir import ExactnessClass

from .models import (
    MAX_RECORDS,
    AggregateMetrics,
    BoundedMetric,
    DecisionLogProbability,
    DistributionSummary,
    ImportanceWeight,
    LogProbabilityRecomputeDirective,
    LogProbabilityRole,
    PolicySegment,
    ReasonCode,
    ResampleDirective,
    SampleKind,
    SegmentMetrics,
    SegmentStalenessReport,
    StalenessAssessmentRequest,
    StalenessDisposition,
    StalenessReason,
    TrainingEligibility,
    TrajectoryStalenessReport,
    TruncationDirective,
)


def canonical_digest(value: object) -> str:
    """Return the canonical SHA-256 identity used by staleness reports."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


_DISPOSITION_ORDER = {
    StalenessDisposition.BOUNDED_ACCEPT: 0,
    StalenessDisposition.PRIORITY_REDUCTION: 1,
    StalenessDisposition.IMPORTANCE_CLIP: 2,
    StalenessDisposition.TRUNCATE: 3,
    StalenessDisposition.RESAMPLE: 4,
    StalenessDisposition.RECOMPUTE_LOGPROBS: 5,
    StalenessDisposition.EVALUATION_ONLY: 6,
    StalenessDisposition.HARD_REJECT: 7,
}


def _primary(dispositions: Iterable[StalenessDisposition]) -> StalenessDisposition:
    return max(dispositions, key=_DISPOSITION_ORDER.__getitem__)


def _ordered_unique(
    dispositions: Iterable[StalenessDisposition],
) -> tuple[StalenessDisposition, ...]:
    return tuple(sorted(set(dispositions), key=_DISPOSITION_ORDER.__getitem__))


def _distribution(values: list[float]) -> DistributionSummary | None:
    if not values:
        return None
    ordered = sorted(values)

    def quantile(probability: float) -> float:
        index = max(0, math.ceil(probability * len(ordered)) - 1)
        return ordered[index]

    return DistributionSummary(
        count=len(ordered),
        minimum=ordered[0],
        maximum=ordered[-1],
        mean=math.fsum(ordered) / len(ordered),
        p50=quantile(0.50),
        p95=quantile(0.95),
        p99=quantile(0.99),
    )


def _bounded_metric(
    value: int | float | None,
    maximum: int | float,
    *,
    evidence_digest: str | None = None,
) -> BoundedMetric:
    numeric = float(value) if value is not None else None
    return BoundedMetric(
        value=numeric,
        maximum=float(maximum),
        available=numeric is not None,
        exceeded=numeric is not None and numeric > maximum,
        evidence_digest=evidence_digest,
    )


def _reason(
    code: ReasonCode,
    message: str,
    *,
    segment: PolicySegment | None = None,
    record: DecisionLogProbability | None = None,
) -> StalenessReason:
    return StalenessReason(
        code=code,
        message=message,
        segment_index=segment.segment_index if segment is not None else None,
        sample_kind=record.sample_kind if record is not None else None,
        sample_index=record.sample_index if record is not None else None,
    )


def _append_reason(reasons: list[StalenessReason], reason: StalenessReason) -> None:
    if len(reasons) < MAX_RECORDS:
        reasons.append(reason)


def _records_for_segment(
    request: StalenessAssessmentRequest, segment: PolicySegment
) -> tuple[DecisionLogProbability, ...]:
    records = (
        record
        for record in request.trajectory.log_probabilities
        if (
            segment.token_range.contains(record.sample_index)
            if record.sample_kind is SampleKind.TOKEN
            else segment.action_range.contains(record.sample_index)
        )
    )
    return tuple(sorted(records, key=lambda item: (item.sample_kind.value, item.sample_index)))


def _transition_reasons(request: StalenessAssessmentRequest) -> tuple[StalenessReason, ...]:
    reasons: list[StalenessReason] = []
    for boundary in request.trajectory.transitions:
        segment = request.trajectory.segments[boundary.to_segment_index]
        compatibility = boundary.compatibility
        if (
            not compatibility.safe
            or compatibility.compatibility_class is ExactnessClass.INCOMPATIBLE
        ):
            _append_reason(
                reasons,
                _reason(
                    ReasonCode.INCOMPATIBLE_TRANSITION,
                    f"transition {boundary.boundary_id} is Continuum-incompatible",
                    segment=segment,
                ),
            )
            continue
        required = set(compatibility.required_recomputation)
        provided = {
            component for action in boundary.recompute_actions for component in action.component_ids
        }
        missing = sorted(required - provided)
        unexpected = sorted(provided - required)
        if missing:
            _append_reason(
                reasons,
                _reason(
                    ReasonCode.INCOMPLETE_STATE_RECOMPUTATION,
                    "transition is missing completed recomputation evidence for: "
                    + ", ".join(missing),
                    segment=segment,
                ),
            )
        if unexpected:
            _append_reason(
                reasons,
                _reason(
                    ReasonCode.UNDECLARED_STATE_RECOMPUTATION,
                    "transition contains undeclared recomputation evidence for: "
                    + ", ".join(unexpected),
                    segment=segment,
                ),
            )
        if (
            compatibility.compatibility_class is ExactnessClass.RECOMPUTATION_ASSISTED
            and not required
        ):
            _append_reason(
                reasons,
                _reason(
                    ReasonCode.INCOMPLETE_STATE_RECOMPUTATION,
                    "recomputation-assisted transition has no declared recomputation requirement",
                    segment=segment,
                ),
            )
    return tuple(reasons)


def _segment_report(
    request: StalenessAssessmentRequest,
    segment: PolicySegment,
    *,
    transition_failure: bool,
) -> SegmentStalenessReport:
    policy = request.policy
    learner = request.learner_policy
    reasons: list[StalenessReason] = []
    dispositions: list[StalenessDisposition] = [StalenessDisposition.BOUNDED_ACCEPT]
    same_policy = segment.policy == learner

    epoch_distance: int | None = None
    update_distance: int | None = None
    if segment.policy.policy_id != learner.policy_id:
        _append_reason(
            reasons,
            _reason(
                ReasonCode.POLICY_ID_MISMATCH,
                "behavior and learner policies have different logical identities",
                segment=segment,
            ),
        )
        dispositions.append(StalenessDisposition.HARD_REJECT)
    elif segment.policy.epoch > learner.epoch or segment.policy.update > learner.update:
        _append_reason(
            reasons,
            _reason(
                ReasonCode.FUTURE_BEHAVIOR_POLICY,
                "behavior policy is newer than the learner policy",
                segment=segment,
            ),
        )
        dispositions.append(StalenessDisposition.HARD_REJECT)
    else:
        epoch_distance = learner.epoch - segment.policy.epoch
        update_distance = learner.update - segment.policy.update

    wall_age_ms: int | None = None
    if segment.completed_at_ms > request.assessed_at_ms:
        _append_reason(
            reasons,
            _reason(
                ReasonCode.FUTURE_COLLECTION_TIME,
                "segment completion time is later than assessment time",
                segment=segment,
            ),
        )
        dispositions.append(StalenessDisposition.HARD_REJECT)
    else:
        wall_age_ms = request.assessed_at_ms - segment.completed_at_ms

    distance = next(
        (item for item in request.distance_evidence if item.behavior_policy == segment.policy),
        None,
    )
    if same_policy:
        parameter_delta = 0.0
        kl_divergence = 0.0
        distance_digest = None
    elif distance is None:
        parameter_delta = None
        kl_divergence = None
        distance_digest = None
        _append_reason(
            reasons,
            _reason(
                ReasonCode.MISSING_DISTANCE_EVIDENCE,
                "off-policy segment lacks explicit parameter-delta and KL evidence",
                segment=segment,
            ),
        )
        dispositions.append(policy.missing_metric_disposition)
    else:
        parameter_delta = distance.parameter_delta_l2
        kl_divergence = distance.kl_divergence
        distance_digest = distance.evidence_digest

    epoch_metric = _bounded_metric(epoch_distance, policy.max_epoch_distance)
    update_metric = _bounded_metric(update_distance, policy.max_update_distance)
    wall_metric = _bounded_metric(wall_age_ms, policy.max_wall_age_ms)
    parameter_metric = _bounded_metric(
        parameter_delta,
        policy.max_parameter_delta_l2,
        evidence_digest=distance_digest,
    )
    kl_metric = _bounded_metric(
        kl_divergence,
        policy.max_kl_divergence,
        evidence_digest=distance_digest,
    )
    metric_reasons = (
        (epoch_metric, ReasonCode.EPOCH_DISTANCE_EXCEEDED, "policy epoch distance exceeds bound"),
        (
            update_metric,
            ReasonCode.UPDATE_DISTANCE_EXCEEDED,
            "policy update distance exceeds bound",
        ),
        (wall_metric, ReasonCode.WALL_AGE_EXCEEDED, "trajectory wall age exceeds bound"),
        (
            parameter_metric,
            ReasonCode.PARAMETER_DELTA_EXCEEDED,
            "parameter L2 delta exceeds bound",
        ),
        (kl_metric, ReasonCode.KL_DIVERGENCE_EXCEEDED, "policy KL divergence exceeds bound"),
    )
    for metric, code, message in metric_reasons:
        if metric.exceeded:
            _append_reason(reasons, _reason(code, message, segment=segment))
            dispositions.append(policy.stale_disposition)

    ratios: list[tuple[DecisionLogProbability, float]] = []
    for record in _records_for_segment(request, segment):
        if record.behavior_log_probability is None:
            _append_reason(
                reasons,
                _reason(
                    ReasonCode.MISSING_BEHAVIOR_LOGPROB,
                    "sample has no behavior-policy log probability",
                    segment=segment,
                    record=record,
                ),
            )
            dispositions.append(policy.missing_logprob_disposition)
            continue
        if same_policy:
            ratios.append((record, 1.0))
            continue
        if record.target_log_probability is None:
            _append_reason(
                reasons,
                _reason(
                    ReasonCode.MISSING_TARGET_LOGPROB,
                    "off-policy sample has no explicit learner-policy log probability",
                    segment=segment,
                    record=record,
                ),
            )
            dispositions.append(policy.missing_logprob_disposition)
            continue
        if record.target_policy != learner:
            _append_reason(
                reasons,
                _reason(
                    ReasonCode.TARGET_POLICY_MISMATCH,
                    "target log probability does not name the assessed learner policy",
                    segment=segment,
                    record=record,
                ),
            )
            dispositions.append(StalenessDisposition.HARD_REJECT)
            continue
        log_ratio = record.target_log_probability - record.behavior_log_probability
        try:
            ratio = math.exp(log_ratio)
        except OverflowError:
            ratio = math.inf
        if not math.isfinite(ratio) or ratio <= 0.0:
            _append_reason(
                reasons,
                _reason(
                    ReasonCode.IMPORTANCE_RATIO_NONFINITE,
                    "explicit log probabilities produce a non-finite importance ratio",
                    segment=segment,
                    record=record,
                ),
            )
            dispositions.append(StalenessDisposition.HARD_REJECT)
            continue
        ratios.append((record, ratio))
        if ratio < policy.minimum_importance_ratio or ratio > policy.maximum_importance_ratio:
            _append_reason(
                reasons,
                _reason(
                    ReasonCode.IMPORTANCE_RATIO_EXCEEDED,
                    "importance ratio lies outside the acceptance interval",
                    segment=segment,
                    record=record,
                ),
            )
            dispositions.append(policy.importance_disposition)

    if transition_failure:
        dispositions.append(StalenessDisposition.HARD_REJECT)

    disposition = _primary(dispositions)
    importance_weights: list[ImportanceWeight] = []
    for record, ratio in ratios:
        if record.behavior_log_probability is None or record.behavior_evidence_digest is None:
            raise AssertionError("assessed ratio lost validated behavior provenance")
        derivation: Literal["policy_identity", "explicit_log_probabilities"]
        applied = ratio
        clipped = False
        if disposition is StalenessDisposition.IMPORTANCE_CLIP:
            applied = min(
                policy.clip_importance_ratio_maximum,
                max(policy.clip_importance_ratio_minimum, ratio),
            )
            clipped = applied != ratio
            if clipped:
                _append_reason(
                    reasons,
                    _reason(
                        ReasonCode.IMPORTANCE_RATIO_CLIPPED,
                        "importance ratio was clipped to the configured interval",
                        segment=segment,
                        record=record,
                    ),
                )
        if same_policy:
            derivation = "policy_identity"
            target_log_probability = None
            target_evidence_digest = None
        else:
            if record.target_log_probability is None or record.target_evidence_digest is None:
                raise AssertionError("off-policy ratio lost validated target provenance")
            derivation = "explicit_log_probabilities"
            target_log_probability = record.target_log_probability
            target_evidence_digest = record.target_evidence_digest
        importance_weights.append(
            ImportanceWeight(
                sample_kind=record.sample_kind,
                sample_index=record.sample_index,
                derivation=derivation,
                behavior_policy=record.behavior_policy,
                target_policy=learner,
                behavior_log_probability=record.behavior_log_probability,
                target_log_probability=target_log_probability,
                behavior_evidence_digest=record.behavior_evidence_digest,
                target_evidence_digest=target_evidence_digest,
                raw_ratio=ratio,
                applied_ratio=applied,
                clipped=clipped,
            )
        )

    if disposition is StalenessDisposition.TRUNCATE:
        _append_reason(
            reasons,
            _reason(
                ReasonCode.SEGMENT_TRUNCATED,
                "segment is excluded from the training range",
                segment=segment,
            ),
        )
    elif disposition is StalenessDisposition.RESAMPLE:
        _append_reason(
            reasons,
            _reason(
                ReasonCode.RESAMPLE_REQUIRED,
                "segment requires a new rollout under the learner policy",
                segment=segment,
            ),
        )
    elif disposition is StalenessDisposition.PRIORITY_REDUCTION:
        _append_reason(
            reasons,
            _reason(
                ReasonCode.PRIORITY_REDUCED,
                "segment remains trainable at reduced replay priority",
                segment=segment,
            ),
        )
    elif disposition is StalenessDisposition.EVALUATION_ONLY:
        _append_reason(
            reasons,
            _reason(
                ReasonCode.EVALUATION_ONLY,
                "segment is retained for evaluation but excluded from training",
                segment=segment,
            ),
        )

    training_eligible = disposition in {
        StalenessDisposition.BOUNDED_ACCEPT,
        StalenessDisposition.IMPORTANCE_CLIP,
        StalenessDisposition.PRIORITY_REDUCTION,
    }
    return SegmentStalenessReport(
        segment_id=segment.segment_id,
        segment_index=segment.segment_index,
        behavior_policy=segment.policy,
        learner_policy=learner,
        on_policy=same_policy,
        stale=any(metric.exceeded for metric, _, _ in metric_reasons)
        or any(reason.code is ReasonCode.IMPORTANCE_RATIO_EXCEEDED for reason in reasons),
        disposition=disposition,
        collected_at_ms=segment.collected_at_ms,
        completed_at_ms=segment.completed_at_ms,
        metrics=SegmentMetrics(
            epoch_distance=epoch_metric,
            update_distance=update_metric,
            wall_age_ms=wall_metric,
            parameter_delta_l2=parameter_metric,
            kl_divergence=kl_metric,
            importance_ratios=_distribution([ratio for _, ratio in ratios]),
        ),
        importance_weights=tuple(importance_weights),
        priority=(
            policy.priority_reduction_factor
            if disposition is StalenessDisposition.PRIORITY_REDUCTION
            else 1.0
        ),
        training_eligible=training_eligible,
        token_range=segment.token_range,
        action_range=segment.action_range,
        reasons=tuple(reasons),
    )


def _maximum(values: Iterable[float | None]) -> float | None:
    available = [value for value in values if value is not None]
    return max(available) if available else None


def _resample_seed(seed: int, trajectory_id: str, segment_id: str) -> int:
    payload = f"sloforge.helix.resample/v1\0{seed}\0{trajectory_id}\0{segment_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _logprob_seed(
    seed: int,
    trajectory_id: str,
    segment_id: str,
    sample_kind: SampleKind,
    sample_index: int,
    role: LogProbabilityRole,
) -> int:
    payload = (
        "sloforge.helix.logprob-recompute/v1"
        f"\0{seed}\0{trajectory_id}\0{segment_id}"
        f"\0{sample_kind.value}\0{sample_index}\0{role.value}"
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def assess_staleness(request: StalenessAssessmentRequest) -> TrajectoryStalenessReport:
    """Assess a bounded trajectory without invoking a policy or recomputing evidence."""

    transition_reasons = _transition_reasons(request)
    transition_failed_segments = {
        reason.segment_index
        for reason in transition_reasons
        if reason.code
        in {
            ReasonCode.INCOMPATIBLE_TRANSITION,
            ReasonCode.INCOMPLETE_STATE_RECOMPUTATION,
            ReasonCode.UNDECLARED_STATE_RECOMPUTATION,
        }
    }
    transition_failure = bool(transition_failed_segments)
    segment_reports = tuple(
        _segment_report(
            request,
            segment,
            transition_failure=transition_failure,
        )
        for segment in request.trajectory.segments
    )

    policy_keys = {segment.policy.key for segment in request.trajectory.segments}
    mixed_policy = len(policy_keys) > 1
    on_policy = (
        request.trajectory.semantics.value == "strict"
        and not mixed_policy
        and all(segment.on_policy for segment in segment_reports)
    )

    all_reasons: list[StalenessReason] = []
    if mixed_policy:
        _append_reason(
            all_reasons,
            _reason(
                ReasonCode.MIXED_POLICY,
                "segmented trajectory contains multiple behavior policies and is off-policy",
            ),
        )
    for reason in transition_reasons:
        _append_reason(all_reasons, reason)
    for report in segment_reports:
        for reason in report.reasons:
            _append_reason(all_reasons, reason)

    truncations = tuple(
        TruncationDirective(
            segment_id=report.segment_id,
            token_range=report.token_range,
            action_range=report.action_range,
        )
        for report in segment_reports
        if report.disposition is StalenessDisposition.TRUNCATE
    )
    resamples = tuple(
        ResampleDirective(
            segment_id=report.segment_id,
            token_range=report.token_range,
            action_range=report.action_range,
            seed=_resample_seed(request.seed, request.trajectory.trajectory_id, report.segment_id),
        )
        for report in segment_reports
        if report.disposition is StalenessDisposition.RESAMPLE
    )
    record_by_key = {
        (record.sample_kind, record.sample_index): record
        for record in request.trajectory.log_probabilities
    }
    recompute_directives: list[LogProbabilityRecomputeDirective] = []
    recompute_keys: set[tuple[int, SampleKind, int, LogProbabilityRole]] = set()
    for report in segment_reports:
        for reason in report.reasons:
            if reason.sample_kind is None or reason.sample_index is None:
                continue
            if reason.code is ReasonCode.MISSING_BEHAVIOR_LOGPROB:
                role = LogProbabilityRole.BEHAVIOR
            elif reason.code is ReasonCode.MISSING_TARGET_LOGPROB:
                role = LogProbabilityRole.TARGET
            else:
                continue
            key = (report.segment_index, reason.sample_kind, reason.sample_index, role)
            if key in recompute_keys:
                continue
            recompute_keys.add(key)
            record = record_by_key[(reason.sample_kind, reason.sample_index)]
            target_policy = (
                record.behavior_policy
                if role is LogProbabilityRole.BEHAVIOR
                else request.learner_policy
            )
            recompute_directives.append(
                LogProbabilityRecomputeDirective(
                    segment_id=report.segment_id,
                    sample_kind=reason.sample_kind,
                    sample_index=reason.sample_index,
                    role=role,
                    policy=target_policy,
                    trace_evidence_digest=request.trajectory.trace_evidence_digest,
                    seed=_logprob_seed(
                        request.seed,
                        request.trajectory.trajectory_id,
                        report.segment_id,
                        reason.sample_kind,
                        reason.sample_index,
                        role,
                    ),
                )
            )
    resample_bound_failed = len(resamples) > request.policy.max_resample_directives
    if resample_bound_failed:
        _append_reason(
            all_reasons,
            _reason(
                ReasonCode.RESAMPLE_BOUND_EXCEEDED,
                "resample directive count exceeds the configured bound",
            ),
        )

    eligible_reports = tuple(report for report in segment_reports if report.training_eligible)
    partial_forbidden = not request.policy.allow_partial_training and len(eligible_reports) != len(
        segment_reports
    )
    globally_rejected = transition_failure or resample_bound_failed or partial_forbidden
    if globally_rejected:
        eligible_reports = ()

    if not eligible_reports:
        eligibility = TrainingEligibility.INELIGIBLE
    elif len(eligible_reports) != len(segment_reports):
        eligibility = TrainingEligibility.PARTIALLY_ELIGIBLE
    elif on_policy and all(
        report.disposition is StalenessDisposition.BOUNDED_ACCEPT for report in segment_reports
    ):
        eligibility = TrainingEligibility.ELIGIBLE
    else:
        eligibility = TrainingEligibility.ELIGIBLE_WITH_CORRECTION

    dispositions = [report.disposition for report in segment_reports]
    if globally_rejected:
        dispositions.append(StalenessDisposition.HARD_REJECT)
    applied_dispositions = _ordered_unique(dispositions)
    importance_values = [
        weight.raw_ratio for report in segment_reports for weight in report.importance_weights
    ]
    aggregate = AggregateMetrics(
        maximum_epoch_distance=_maximum(
            report.metrics.epoch_distance.value for report in segment_reports
        ),
        maximum_update_distance=_maximum(
            report.metrics.update_distance.value for report in segment_reports
        ),
        maximum_wall_age_ms=_maximum(
            report.metrics.wall_age_ms.value for report in segment_reports
        ),
        maximum_parameter_delta_l2=_maximum(
            report.metrics.parameter_delta_l2.value for report in segment_reports
        ),
        maximum_kl_divergence=_maximum(
            report.metrics.kl_divergence.value for report in segment_reports
        ),
        importance_ratios=_distribution(importance_values),
    )
    eligible_token_ranges = tuple(
        report.token_range for report in eligible_reports if report.token_range.length > 0
    )
    eligible_action_ranges = tuple(
        report.action_range for report in eligible_reports if report.action_range.length > 0
    )

    body: dict[str, Any] = {
        "trajectory_id": request.trajectory.trajectory_id,
        "trajectory_trace_evidence_digest": request.trajectory.trace_evidence_digest,
        "learner_policy": request.learner_policy,
        "semantics": request.trajectory.semantics,
        "mixed_policy": mixed_policy,
        "on_policy": on_policy,
        "primary_disposition": _primary(applied_dispositions),
        "applied_dispositions": applied_dispositions,
        "training_eligibility": eligibility,
        "training_eligible": eligibility is not TrainingEligibility.INELIGIBLE,
        "segment_reports": segment_reports,
        "metrics": aggregate,
        "eligible_token_ranges": eligible_token_ranges,
        "eligible_action_ranges": eligible_action_ranges,
        "truncations": truncations,
        "resamples": resamples,
        "logprob_recomputations": tuple(recompute_directives),
        "reasons": tuple(all_reasons),
        "policy_id": request.policy.policy_id,
        "assessed_at_ms": request.assessed_at_ms,
        "seed": request.seed,
    }
    draft = TrajectoryStalenessReport.model_construct(report_id="0" * 64, **body)
    report_id = canonical_digest(draft.model_dump(mode="json", exclude={"report_id"}))
    return TrajectoryStalenessReport(report_id=report_id, **body)


class StalenessAssessor:
    """Small stateless facade for callers that prefer an object boundary."""

    def assess(self, request: StalenessAssessmentRequest) -> TrajectoryStalenessReport:
        return assess_staleness(request)


__all__ = ["StalenessAssessor", "assess_staleness", "canonical_digest"]
