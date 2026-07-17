"""Predicted-versus-observed learning-value accounting with raw provenance."""

from __future__ import annotations

import hashlib
import json
import math

from .models import (
    LearningValueComparison,
    LearningValueEvaluation,
    RawLearningValueSample,
    SchedulerPlan,
)
from .statistics import student_t_mean_interval_95


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def evaluate_learning_value(
    plan: SchedulerPlan,
    samples: tuple[RawLearningValueSample, ...],
    *,
    seed: int,
) -> LearningValueEvaluation:
    """Compare plan predictions only where caller-supplied observations exist."""

    if not 0 <= seed < 2**64:
        raise ValueError("evaluation seed must be an unsigned 64-bit integer")
    sample_ids = [sample.sample_id for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("raw learning-value sample identifiers must be unique")
    work_seeds = [(sample.work_id, sample.seed) for sample in samples]
    if len(work_seeds) != len(set(work_seeds)):
        raise ValueError("each work item accepts at most one observation per seed")
    completed = {
        outcome.work_id: outcome
        for outcome in plan.outcomes
        if outcome.work_id in set(plan.completed_work_ids)
    }
    unknown = sorted({sample.work_id for sample in samples} - set(completed))
    if unknown:
        raise ValueError("learning-value samples reference work not completed by this plan")
    by_work: dict[str, list[RawLearningValueSample]] = {}
    for sample in samples:
        by_work.setdefault(sample.work_id, []).append(sample)

    comparisons: list[LearningValueComparison] = []
    for work_id in sorted(by_work):
        work_samples = sorted(by_work[work_id], key=lambda item: (item.seed, item.sample_id))
        if not 2 <= len(work_samples) <= 32:
            raise ValueError("each compared work item requires observations from two to 32 seeds")
        observed, lower, upper, deviation = student_t_mean_interval_95(
            tuple(item.value for item in work_samples)
        )
        predicted = completed[work_id].predicted_learning_value
        error = observed - predicted
        comparisons.append(
            LearningValueComparison(
                work_id=work_id,
                predicted_value=predicted,
                observed_value=observed,
                signed_error=error,
                absolute_error=abs(error),
                observed_mean_lower_95=lower,
                observed_mean_upper_95=upper,
                observed_sample_standard_deviation=deviation,
                sample_seeds=tuple(item.seed for item in work_samples),
                raw_sample_ids=tuple(item.sample_id for item in work_samples),
                evidence_artifacts=tuple(item.evidence for item in work_samples),
            )
        )

    missing = tuple(sorted(set(completed) - set(by_work)))
    predicted_total = math.fsum(item.predicted_value for item in comparisons)
    observed_total = math.fsum(item.observed_value for item in comparisons)
    mean_absolute_error = (
        math.fsum(item.absolute_error for item in comparisons) / len(comparisons)
        if comparisons
        else 0.0
    )
    observation_seeds = tuple(
        sorted({seed_value for item in comparisons for seed_value in item.sample_seeds})
    )
    limitations = (
        "observed values are per-seed raw observations summarized with a two-sided Student-t 95% mean interval",
        "one observation per work item and seed is accepted to avoid treating same-seed replicates as independent",
        "the interval describes sensitivity across supplied deterministic seeds and is not a population guarantee",
        "missing observations are reported and never imputed",
        "observed total is a sum of per-work means and has no aggregate confidence interval",
        "the comparison does not establish causal value of scheduler selection",
    )
    draft = {
        "schema_version": "sloforge.helix.learning-value-evaluation/v1",
        "plan_id": plan.plan_id,
        "seed": seed,
        "comparisons": tuple(item.model_dump(mode="json") for item in comparisons),
        "missing_observation_work_ids": missing,
        "predicted_total_for_compared_work": predicted_total,
        "observed_total": observed_total,
        "signed_error_total": observed_total - predicted_total,
        "mean_absolute_error": mean_absolute_error,
        "observation_seeds": observation_seeds,
        "raw_sample_count": len(samples),
        "limitations": limitations,
    }
    return LearningValueEvaluation(
        evaluation_id=_digest(draft),
        plan_id=plan.plan_id,
        seed=seed,
        comparisons=tuple(comparisons),
        missing_observation_work_ids=missing,
        predicted_total_for_compared_work=predicted_total,
        observed_total=observed_total,
        signed_error_total=observed_total - predicted_total,
        mean_absolute_error=mean_absolute_error,
        observation_seeds=observation_seeds,
        raw_sample_count=len(samples),
        limitations=limitations,
    )


__all__ = ["evaluate_learning_value"]
