from __future__ import annotations

import hashlib
import math

import pytest
from pydantic import ValidationError

from sloforge.continuum.ir import ExactnessClass
from sloforge.helix.staleness import (
    ContinuumCompatibilityEvidence,
    DecisionLogProbability,
    IndexRange,
    LogProbabilityRole,
    LogProbabilitySource,
    PolicyDistanceEvidence,
    PolicySegment,
    PolicySemantics,
    PolicyVersion,
    ReasonCode,
    SampleKind,
    StalenessAssessmentRequest,
    StalenessDisposition,
    StalenessPolicy,
    StateRecomputeActionEvidence,
    TrainingEligibility,
    TrajectoryPolicyProvenance,
    TransitionBoundary,
    assess_staleness,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _policy(epoch: int, *, update: int | None = None) -> PolicyVersion:
    resolved_update = epoch if update is None else update
    return PolicyVersion(
        policy_id="coding-agent",
        epoch=epoch,
        update=resolved_update,
        parameter_digest=_digest(f"parameters:{epoch}:{resolved_update}"),
        continuum_compatibility_fingerprint=_digest(f"compatibility:{epoch}"),
        published_at_ms=epoch * 100,
    )


def _recorded(
    kind: SampleKind,
    index: int,
    behavior: PolicyVersion,
    log_probability: float,
    *,
    target: PolicyVersion | None = None,
    target_log_probability: float | None = None,
) -> DecisionLogProbability:
    return DecisionLogProbability(
        sample_kind=kind,
        sample_index=index,
        behavior_policy=behavior,
        behavior_log_probability=log_probability,
        behavior_source=LogProbabilitySource.RECORDED,
        behavior_evidence_digest=_digest(f"behavior:{kind}:{index}:{behavior.epoch}"),
        target_policy=target,
        target_log_probability=target_log_probability,
        target_source=(
            LogProbabilitySource.RECORDED if target is not None else LogProbabilitySource.MISSING
        ),
        target_evidence_digest=(
            _digest(f"target:{kind}:{index}:{target.epoch}") if target is not None else None
        ),
    )


def _segment(
    index: int,
    policy: PolicyVersion,
    token_range: tuple[int, int],
    action_range: tuple[int, int],
    *,
    completed_at_ms: int = 1_000,
) -> PolicySegment:
    return PolicySegment(
        segment_id=f"segment-{index}",
        segment_index=index,
        policy=policy,
        token_range=IndexRange(start=token_range[0], end_exclusive=token_range[1]),
        action_range=IndexRange(start=action_range[0], end_exclusive=action_range[1]),
        collected_at_ms=900,
        completed_at_ms=completed_at_ms,
    )


def _compatibility(
    source: PolicyVersion,
    destination: PolicyVersion,
    *,
    exactness: ExactnessClass = ExactnessClass.EXACT_SEMANTIC,
    required_recomputation: tuple[str, ...] = (),
) -> ContinuumCompatibilityEvidence:
    return ContinuumCompatibilityEvidence(
        report_id=f"continuum-report-{source.epoch}-{destination.epoch}",
        report_digest=_digest(f"continuum-report:{source.epoch}:{destination.epoch}:{exactness}"),
        compatibility_class=exactness,
        safe=exactness is not ExactnessClass.INCOMPATIBLE,
        source_compatibility_fingerprint=source.continuum_compatibility_fingerprint,
        destination_compatibility_fingerprint=destination.continuum_compatibility_fingerprint,
        required_recomputation=required_recomputation,
    )


def _boundary(
    source: PolicyVersion,
    destination: PolicyVersion,
    *,
    compatibility: ContinuumCompatibilityEvidence | None = None,
    recompute_actions: tuple[StateRecomputeActionEvidence, ...] = (),
) -> TransitionBoundary:
    return TransitionBoundary(
        boundary_id="boundary-0-1",
        from_segment_index=0,
        to_segment_index=1,
        token_index=1,
        action_index=0,
        from_policy=source,
        to_policy=destination,
        compatibility=compatibility or _compatibility(source, destination),
        recompute_actions=recompute_actions,
    )


def _strict_request(
    *,
    behavior: PolicyVersion | None = None,
    learner: PolicyVersion | None = None,
    records: tuple[DecisionLogProbability, ...] | None = None,
    staleness_policy: StalenessPolicy | None = None,
    completed_at_ms: int = 1_000,
    distance: PolicyDistanceEvidence | None = None,
) -> StalenessAssessmentRequest:
    resolved_behavior = behavior or _policy(2)
    resolved_learner = learner or resolved_behavior
    resolved_records = records or (
        _recorded(SampleKind.TOKEN, 0, resolved_behavior, -0.4),
        _recorded(SampleKind.TOKEN, 1, resolved_behavior, -0.6),
        _recorded(SampleKind.ACTION, 0, resolved_behavior, -0.2),
    )
    trajectory = TrajectoryPolicyProvenance(
        trajectory_id="trajectory-strict",
        semantics=PolicySemantics.STRICT,
        token_count=2,
        action_count=1,
        segments=(
            _segment(
                0,
                resolved_behavior,
                (0, 2),
                (0, 1),
                completed_at_ms=completed_at_ms,
            ),
        ),
        log_probabilities=resolved_records,
        trace_evidence_digest=_digest("strict-trace"),
    )
    return StalenessAssessmentRequest(
        trajectory=trajectory,
        learner_policy=resolved_learner,
        policy=staleness_policy or StalenessPolicy(policy_id="default-staleness"),
        distance_evidence=() if distance is None else (distance,),
        assessed_at_ms=2_000,
        seed=73129,
    )


def test_strict_trajectory_is_on_policy_only_for_an_exact_policy_version() -> None:
    request = _strict_request()
    report = assess_staleness(request)

    assert report.on_policy
    assert not report.mixed_policy
    assert report.primary_disposition is StalenessDisposition.BOUNDED_ACCEPT
    assert report.training_eligibility is TrainingEligibility.ELIGIBLE
    assert report.training_eligible
    assert report.metrics.importance_ratios is not None
    assert report.metrics.importance_ratios.minimum == 1.0
    assert report.performed_recomputation is False
    assert report == assess_staleness(request)

    raw = request.trajectory.model_dump()
    raw["semantics"] = PolicySemantics.STRICT
    raw["segments"] = (
        request.trajectory.segments[0].model_copy(
            update={"token_range": IndexRange(start=0, end_exclusive=1)}
        ),
        _segment(1, request.learner_policy, (1, 2), (1, 1)),
    )
    with pytest.raises(ValidationError, match="strict policy semantics"):
        TrajectoryPolicyProvenance.model_validate(raw, strict=True)


def test_segmented_trajectory_has_explicit_ranges_and_is_never_on_policy() -> None:
    old = _policy(1)
    learner = _policy(2)
    trajectory = TrajectoryPolicyProvenance(
        trajectory_id="trajectory-segmented",
        semantics=PolicySemantics.SEGMENTED,
        token_count=2,
        action_count=1,
        segments=(
            _segment(0, old, (0, 1), (0, 0)),
            _segment(1, learner, (1, 2), (0, 1)),
        ),
        transitions=(_boundary(old, learner),),
        log_probabilities=(
            _recorded(
                SampleKind.TOKEN,
                0,
                old,
                -0.5,
                target=learner,
                target_log_probability=-0.45,
            ),
            _recorded(SampleKind.TOKEN, 1, learner, -0.3),
            _recorded(SampleKind.ACTION, 0, learner, -0.2),
        ),
        trace_evidence_digest=_digest("segmented-trace"),
    )
    distance = PolicyDistanceEvidence(
        behavior_policy=old,
        learner_policy=learner,
        parameter_delta_l2=0.1,
        kl_divergence=0.01,
        sample_count=64,
        evidence_digest=_digest("old-to-new-distance"),
    )
    report = assess_staleness(
        StalenessAssessmentRequest(
            trajectory=trajectory,
            learner_policy=learner,
            policy=StalenessPolicy(policy_id="segmented-policy"),
            distance_evidence=(distance,),
            assessed_at_ms=2_000,
            seed=19,
        )
    )

    assert report.mixed_policy
    assert not report.on_policy
    assert report.training_eligibility is TrainingEligibility.ELIGIBLE_WITH_CORRECTION
    assert ReasonCode.MIXED_POLICY in {reason.code for reason in report.reasons}
    assert report.segment_reports[0].importance_weights[0].raw_ratio == pytest.approx(
        math.exp(0.05)
    )
    assert report.segment_reports[1].on_policy


def test_stale_trajectory_reports_every_bounded_distance_and_hard_rejects() -> None:
    behavior = _policy(1, update=2)
    learner = _policy(5, update=20)
    records = (
        _recorded(
            SampleKind.TOKEN,
            0,
            behavior,
            -0.4,
            target=learner,
            target_log_probability=-0.4,
        ),
        _recorded(
            SampleKind.TOKEN,
            1,
            behavior,
            -0.6,
            target=learner,
            target_log_probability=-0.6,
        ),
        _recorded(
            SampleKind.ACTION,
            0,
            behavior,
            -0.2,
            target=learner,
            target_log_probability=-0.2,
        ),
    )
    distance = PolicyDistanceEvidence(
        behavior_policy=behavior,
        learner_policy=learner,
        parameter_delta_l2=3.0,
        kl_divergence=0.8,
        sample_count=128,
        evidence_digest=_digest("stale-distance"),
    )
    policy = StalenessPolicy(
        policy_id="strict-bounds",
        max_epoch_distance=1,
        max_update_distance=4,
        max_wall_age_ms=100,
        max_parameter_delta_l2=1.0,
        max_kl_divergence=0.2,
    )
    report = assess_staleness(
        _strict_request(
            behavior=behavior,
            learner=learner,
            records=records,
            staleness_policy=policy,
            distance=distance,
        )
    )

    segment = report.segment_reports[0]
    assert segment.stale
    assert segment.metrics.epoch_distance.exceeded
    assert segment.metrics.update_distance.exceeded
    assert segment.metrics.wall_age_ms.exceeded
    assert segment.metrics.parameter_delta_l2.exceeded
    assert segment.metrics.kl_divergence.exceeded
    assert report.primary_disposition is StalenessDisposition.HARD_REJECT
    assert report.training_eligibility is TrainingEligibility.INELIGIBLE


def test_missing_behavior_log_probability_is_explicit_and_never_recomputed() -> None:
    policy = _policy(2)
    records = (
        DecisionLogProbability(
            sample_kind=SampleKind.TOKEN,
            sample_index=0,
            behavior_policy=policy,
        ),
        _recorded(SampleKind.TOKEN, 1, policy, -0.6),
        _recorded(SampleKind.ACTION, 0, policy, -0.2),
    )
    report = assess_staleness(_strict_request(behavior=policy, records=records))

    assert report.primary_disposition is StalenessDisposition.RECOMPUTE_LOGPROBS
    assert not report.training_eligible
    assert not report.performed_recomputation
    assert ReasonCode.MISSING_BEHAVIOR_LOGPROB in {reason.code for reason in report.reasons}
    assert len(report.logprob_recomputations) == 1
    assert report.logprob_recomputations[0].role is LogProbabilityRole.BEHAVIOR
    assert report.logprob_recomputations[0].sample_index == 0

    raw = _recorded(SampleKind.TOKEN, 0, policy, -0.4).model_dump()
    del raw["behavior_log_probability"]
    with pytest.raises(ValidationError, match="requires a value"):
        DecisionLogProbability.model_validate(raw, strict=True)


def test_incompatible_or_unproven_continuum_transition_rejects_all_training() -> None:
    old = _policy(1)
    learner = _policy(2)
    incompatible = _compatibility(
        old,
        learner,
        exactness=ExactnessClass.INCOMPATIBLE,
    )
    trajectory = TrajectoryPolicyProvenance(
        trajectory_id="trajectory-incompatible",
        semantics=PolicySemantics.SEGMENTED,
        token_count=2,
        action_count=0,
        segments=(
            _segment(0, old, (0, 1), (0, 0)),
            _segment(1, learner, (1, 2), (0, 0)),
        ),
        transitions=(_boundary(old, learner, compatibility=incompatible),),
        log_probabilities=(
            _recorded(
                SampleKind.TOKEN,
                0,
                old,
                -0.5,
                target=learner,
                target_log_probability=-0.5,
            ),
            _recorded(SampleKind.TOKEN, 1, learner, -0.3),
        ),
        trace_evidence_digest=_digest("incompatible-trace"),
    )
    distance = PolicyDistanceEvidence(
        behavior_policy=old,
        learner_policy=learner,
        parameter_delta_l2=0.1,
        kl_divergence=0.01,
        sample_count=32,
        evidence_digest=_digest("compatible-distance-only"),
    )
    request = StalenessAssessmentRequest(
        trajectory=trajectory,
        learner_policy=learner,
        policy=StalenessPolicy(policy_id="transition-policy"),
        distance_evidence=(distance,),
        assessed_at_ms=2_000,
        seed=7,
    )
    report = assess_staleness(request)

    assert report.primary_disposition is StalenessDisposition.HARD_REJECT
    assert not report.training_eligible
    assert ReasonCode.INCOMPATIBLE_TRANSITION in {reason.code for reason in report.reasons}

    assisted = _compatibility(
        old,
        learner,
        exactness=ExactnessClass.RECOMPUTATION_ASSISTED,
        required_recomputation=("attention_kv",),
    )
    missing_action = request.model_copy(
        update={
            "trajectory": trajectory.model_copy(
                update={"transitions": (_boundary(old, learner, compatibility=assisted),)}
            )
        }
    )
    unproven = assess_staleness(missing_action)
    assert ReasonCode.INCOMPLETE_STATE_RECOMPUTATION in {reason.code for reason in unproven.reasons}

    action = StateRecomputeActionEvidence(
        action_id="recompute-attention",
        component_ids=("attention_kv",),
        source="token_history",
        source_digest=_digest("tokens"),
        result_digest=_digest("new-attention"),
        implementation_digest=_digest("recompute-implementation"),
        seed=7,
        completed_at_ms=1_100,
    )
    proven_request = request.model_copy(
        update={
            "trajectory": trajectory.model_copy(
                update={
                    "transitions": (
                        _boundary(
                            old,
                            learner,
                            compatibility=assisted,
                            recompute_actions=(action,),
                        ),
                    )
                }
            )
        }
    )
    proven = assess_staleness(proven_request)
    assert proven.primary_disposition is StalenessDisposition.BOUNDED_ACCEPT
    assert proven.training_eligible


def test_importance_ratios_are_clipped_only_from_explicit_log_probabilities() -> None:
    behavior = _policy(1)
    learner = _policy(2)
    records = (
        _recorded(
            SampleKind.TOKEN,
            0,
            behavior,
            -2.0,
            target=learner,
            target_log_probability=-0.1,
        ),
        _recorded(
            SampleKind.TOKEN,
            1,
            behavior,
            -0.6,
            target=learner,
            target_log_probability=-0.6,
        ),
        _recorded(
            SampleKind.ACTION,
            0,
            behavior,
            -0.2,
            target=learner,
            target_log_probability=-0.2,
        ),
    )
    distance = PolicyDistanceEvidence(
        behavior_policy=behavior,
        learner_policy=learner,
        parameter_delta_l2=0.1,
        kl_divergence=0.01,
        sample_count=64,
        evidence_digest=_digest("clipping-distance"),
    )
    report = assess_staleness(
        _strict_request(
            behavior=behavior,
            learner=learner,
            records=records,
            distance=distance,
        )
    )

    # Records are deterministically ordered by action/token kind, so find the clipped token.
    weight = next(
        item
        for item in report.segment_reports[0].importance_weights
        if item.sample_kind is SampleKind.TOKEN and item.sample_index == 0
    )
    assert report.primary_disposition is StalenessDisposition.IMPORTANCE_CLIP
    assert report.training_eligibility is TrainingEligibility.ELIGIBLE_WITH_CORRECTION
    assert weight.raw_ratio == pytest.approx(math.exp(1.9))
    assert weight.applied_ratio == 1.25
    assert weight.clipped
    assert ReasonCode.IMPORTANCE_RATIO_CLIPPED in {reason.code for reason in report.reasons}


def test_resampling_is_seeded_and_truncation_retains_only_fresh_ranges() -> None:
    behavior = _policy(1)
    learner = _policy(2)
    records = (
        _recorded(
            SampleKind.TOKEN,
            0,
            behavior,
            -0.4,
            target=learner,
            target_log_probability=-0.4,
        ),
        _recorded(
            SampleKind.TOKEN,
            1,
            behavior,
            -0.6,
            target=learner,
            target_log_probability=-0.6,
        ),
        _recorded(
            SampleKind.ACTION,
            0,
            behavior,
            -0.2,
            target=learner,
            target_log_probability=-0.2,
        ),
    )
    distance = PolicyDistanceEvidence(
        behavior_policy=behavior,
        learner_policy=learner,
        parameter_delta_l2=0.1,
        kl_divergence=0.01,
        sample_count=64,
        evidence_digest=_digest("resample-distance"),
    )
    resample_policy = StalenessPolicy(
        policy_id="resample-policy",
        max_epoch_distance=0,
        stale_disposition=StalenessDisposition.RESAMPLE,
    )
    request = _strict_request(
        behavior=behavior,
        learner=learner,
        records=records,
        staleness_policy=resample_policy,
        distance=distance,
    )
    first = assess_staleness(request)
    second = assess_staleness(request)
    assert first.resamples == second.resamples
    assert len(first.resamples) == 1
    assert first.resamples[0].seed != request.seed
    assert not first.training_eligible

    segmented = TrajectoryPolicyProvenance(
        trajectory_id="trajectory-truncate",
        semantics=PolicySemantics.SEGMENTED,
        token_count=2,
        action_count=0,
        segments=(
            _segment(0, behavior, (0, 1), (0, 0)),
            _segment(1, learner, (1, 2), (0, 0)),
        ),
        transitions=(_boundary(behavior, learner),),
        log_probabilities=(
            _recorded(
                SampleKind.TOKEN,
                0,
                behavior,
                -0.4,
                target=learner,
                target_log_probability=-0.4,
            ),
            _recorded(SampleKind.TOKEN, 1, learner, -0.2),
        ),
        trace_evidence_digest=_digest("truncate-trace"),
    )
    truncate_report = assess_staleness(
        StalenessAssessmentRequest(
            trajectory=segmented,
            learner_policy=learner,
            policy=StalenessPolicy(
                policy_id="truncate-policy",
                max_epoch_distance=0,
                stale_disposition=StalenessDisposition.TRUNCATE,
            ),
            distance_evidence=(distance,),
            assessed_at_ms=2_000,
            seed=88,
        )
    )
    assert truncate_report.primary_disposition is StalenessDisposition.TRUNCATE
    assert truncate_report.training_eligibility is TrainingEligibility.PARTIALLY_ELIGIBLE
    assert truncate_report.eligible_token_ranges == (IndexRange(start=1, end_exclusive=2),)
    assert truncate_report.truncations[0].token_range == IndexRange(start=0, end_exclusive=1)
