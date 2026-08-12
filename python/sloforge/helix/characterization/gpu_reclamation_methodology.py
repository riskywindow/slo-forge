"""Fail-closed paired methodology for BranchFabric Experiment 004.

This module is intentionally local-only.  It plans the bounded three-seed
campaign, decides whether hardware-backed raw trials are admissible, produces
small-sample summaries without inventing tail statistics, and accounts the
two-GPU budget.  A separate coordinator owns Modal and supplies immutable raw
artifact references to these functions.

Experiment 003's statistical helpers remain unchanged.  They are specialized
for two-path same-GPU pairs; Experiment 004 has three paths and a two-GPU
causal transaction, so its typed keys and validity gates live here.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Annotated, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from sloforge.helix.characterization.gpu_reclamation import (
    PilotValidityEvidence,
    ReclamationMode,
)

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=512)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
GpuUuid = Annotated[str, StringConstraints(pattern=r"^GPU-[0-9A-Za-z-]+$")]

PRIMARY_SEEDS: tuple[int, int, int] = (41, 73, 113)
ORDER_ALGORITHM: Literal["position-balanced-cyclic-permutation:sha256-seeded/v1"] = (
    "position-balanced-cyclic-permutation:sha256-seeded/v1"
)
TARGET_ADDITIONAL_GPU_SECONDS = 3_600.0
# Explicitly extended for the user-authorized v9 validation attempt. This
# remains a finite hard ceiling independent of the dollar budget.
HARD_ADDITIONAL_GPU_SECONDS = 7_200.0
HISTORICAL_THROUGH_EXPERIMENT_003_GPU_HOURS = 1.669146
_MODES = tuple(ReclamationMode)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class ArtifactSampleRef(_StrictModel):
    """Exact raw sample provenance for every admitted measurement."""

    artifact_reference: NonEmpty
    artifact_sha256: Sha256
    sample_selector: NonEmpty


class TrialPlan(_StrictModel):
    trial_id: NonEmpty
    seed: int = Field(ge=0, le=2**63 - 1)
    mode: ReclamationMode
    order_position: int = Field(ge=1, le=3)


class FinalCampaignPlan(_StrictModel):
    schema_version: Literal["sloforge.branchfabric.experiment-004-campaign-plan/v1"] = (
        "sloforge.branchfabric.experiment-004-campaign-plan/v1"
    )
    order_seed: int = Field(ge=0, le=2**63 - 1)
    order_algorithm: Literal["position-balanced-cyclic-permutation:sha256-seeded/v1"] = (
        ORDER_ALGORITHM
    )
    pilot_provenance: ArtifactSampleRef
    pilot_digest: Sha256
    trials: tuple[TrialPlan, ...] = Field(min_length=9, max_length=9)

    @model_validator(mode="after")
    def complete_position_balanced_design(self) -> Self:
        keys = tuple((trial.seed, trial.mode) for trial in self.trials)
        if len(set(keys)) != 9:
            raise ValueError("campaign must contain each seed/mode cell exactly once")
        seeds = tuple(dict.fromkeys(trial.seed for trial in self.trials))
        if len(seeds) != 3:
            raise ValueError("campaign requires exactly three seeds")
        for seed in seeds:
            rows = tuple(trial for trial in self.trials if trial.seed == seed)
            if {trial.mode for trial in rows} != set(_MODES):
                raise ValueError("each seed must exercise all three reclamation modes")
            if {trial.order_position for trial in rows} != {1, 2, 3}:
                raise ValueError("each seed row must use positions one through three")
        for position in (1, 2, 3):
            if {trial.mode for trial in self.trials if trial.order_position == position} != set(
                _MODES
            ):
                raise ValueError("each mode must occur once in every order position")
        if len({trial.trial_id for trial in self.trials}) != 9:
            raise ValueError("trial IDs must be unique")
        return self


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def plan_final_campaign(
    *,
    pilot: PilotValidityEvidence,
    pilot_provenance: ArtifactSampleRef,
    order_seed: int,
    seeds: tuple[int, int, int] = PRIMARY_SEEDS,
) -> FinalCampaignPlan:
    """Plan the final campaign only after the real two-GPU pilot validates.

    The design is a deterministically randomized cyclic Latin square.  It is
    randomized in path labels, direction, and assignment of rows to seeds,
    while preserving exact position balance across the three paths.
    """

    if isinstance(order_seed, bool) or not 0 <= order_seed < 2**63:
        raise ValueError("order_seed must be an unsigned signed-64-bit integer")
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("the final campaign requires exactly three unique seeds")
    if any(isinstance(seed, bool) or not 0 <= seed < 2**63 for seed in seeds):
        raise ValueError("campaign seeds must be unsigned signed-64-bit integers")

    generator = random.Random(order_seed)
    labels = list(_MODES)
    generator.shuffle(labels)
    direction = generator.choice((-1, 1))
    offsets = [0, 1, 2]
    generator.shuffle(offsets)
    trials: list[TrialPlan] = []
    for seed, offset in zip(seeds, offsets, strict=True):
        row = tuple(labels[(offset + direction * index) % 3] for index in range(3))
        for position, mode in enumerate(row, start=1):
            trials.append(
                TrialPlan(
                    trial_id=f"exp004-s{seed}-p{position}-{mode.value.lower()}",
                    seed=seed,
                    mode=mode,
                    order_position=position,
                )
            )
    pilot_payload = pilot.model_dump(mode="json")
    return FinalCampaignPlan(
        order_seed=order_seed,
        pilot_provenance=pilot_provenance,
        pilot_digest=_canonical_digest(pilot_payload),
        trials=tuple(trials),
    )


class TraceMetricControl(_StrictModel):
    """One disabled/minimal/full tracing control on an identical workload."""

    metric: NonEmpty
    unit: NonEmpty
    disabled_value: float = Field(allow_inf_nan=False)
    minimal_value: float = Field(allow_inf_nan=False)
    full_value: float = Field(allow_inf_nan=False)
    materiality_threshold_fraction: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    exact_behavior_metric: bool = False
    disabled_provenance: ArtifactSampleRef
    minimal_provenance: ArtifactSampleRef
    full_provenance: ArtifactSampleRef


class TraceMetricEvaluation(_StrictModel):
    metric: NonEmpty
    unit: NonEmpty
    disabled_value: float = Field(allow_inf_nan=False)
    minimal_value: float = Field(allow_inf_nan=False)
    full_value: float = Field(allow_inf_nan=False)
    minimal_relative_change: float | None = Field(allow_inf_nan=False)
    full_relative_change: float | None = Field(allow_inf_nan=False)
    materiality_threshold_fraction: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    exact_behavior_metric: bool
    minimal_material_change: bool
    full_material_change: bool
    provenance: Mapping[str, ArtifactSampleRef]


class TraceControlGate(_StrictModel):
    schema_version: Literal["sloforge.branchfabric.experiment-004-trace-gate/v1"] = (
        "sloforge.branchfabric.experiment-004-trace-gate/v1"
    )
    control_id: NonEmpty
    seed: int = Field(ge=0, le=2**63 - 1)
    workload_digest: Sha256
    gpu_uuids: tuple[GpuUuid, GpuUuid]
    evaluations: tuple[TraceMetricEvaluation, ...] = Field(min_length=1, max_length=64)
    minimal_trace_causal_trials_allowed: bool
    full_trace_diagnostic_only: Literal[True] = True
    rejection_reasons: tuple[NonEmpty, ...]

    @model_validator(mode="after")
    def gate_is_derived(self) -> Self:
        if len(set(self.gpu_uuids)) != 2:
            raise ValueError("trace controls require two distinct GPU UUIDs")
        if len({item.metric for item in self.evaluations}) != len(self.evaluations):
            raise ValueError("trace control metrics must be unique")
        expected_reasons = tuple(
            f"minimal tracing materially changed {item.metric}"
            for item in self.evaluations
            if item.minimal_material_change
        )
        if self.rejection_reasons != expected_reasons:
            raise ValueError("trace rejection reasons are not derived from evaluations")
        if self.minimal_trace_causal_trials_allowed != (not expected_reasons):
            raise ValueError("trace causal gate disagrees with metric evaluations")
        return self


def _relative_change(candidate: float, baseline: float) -> float | None:
    if baseline == 0.0:
        return None
    return (candidate - baseline) / abs(baseline)


def _material_trace_change(
    control: TraceMetricControl,
    *,
    candidate: float,
    relative: float | None,
) -> bool:
    if control.exact_behavior_metric:
        return candidate != control.disabled_value
    if relative is None:
        return candidate != control.disabled_value
    return abs(relative) > control.materiality_threshold_fraction


def evaluate_trace_controls(
    *,
    control_id: str,
    seed: int,
    workload_digest: str,
    gpu_uuids: tuple[str, str],
    controls: Sequence[TraceMetricControl],
) -> TraceControlGate:
    """Reject minimal causal tracing if any declared metric changes materially."""

    if not controls:
        raise ValueError("at least one trace metric control is required")
    evaluations: list[TraceMetricEvaluation] = []
    for control in controls:
        minimal_relative = _relative_change(control.minimal_value, control.disabled_value)
        full_relative = _relative_change(control.full_value, control.disabled_value)

        evaluations.append(
            TraceMetricEvaluation(
                metric=control.metric,
                unit=control.unit,
                disabled_value=control.disabled_value,
                minimal_value=control.minimal_value,
                full_value=control.full_value,
                minimal_relative_change=minimal_relative,
                full_relative_change=full_relative,
                materiality_threshold_fraction=control.materiality_threshold_fraction,
                exact_behavior_metric=control.exact_behavior_metric,
                minimal_material_change=_material_trace_change(
                    control,
                    candidate=control.minimal_value,
                    relative=minimal_relative,
                ),
                full_material_change=_material_trace_change(
                    control,
                    candidate=control.full_value,
                    relative=full_relative,
                ),
                provenance={
                    "disabled": control.disabled_provenance,
                    "minimal": control.minimal_provenance,
                    "full": control.full_provenance,
                },
            )
        )
    reasons = tuple(
        f"minimal tracing materially changed {evaluation.metric}"
        for evaluation in evaluations
        if evaluation.minimal_material_change
    )
    return TraceControlGate(
        control_id=control_id,
        seed=seed,
        workload_digest=workload_digest,
        gpu_uuids=gpu_uuids,
        evaluations=tuple(evaluations),
        minimal_trace_causal_trials_allowed=not reasons,
        rejection_reasons=reasons,
    )


class TrialSemanticEvidence(_StrictModel):
    transaction_completed: bool
    serving_scenario_valid: bool
    hardware_identity_valid: bool
    workload_replay_valid: bool
    critical_timelines_nonoverlapping: bool
    movement_accounting_valid: bool
    allocator_reclaim_valid: bool
    gpu1_useful_serving_capacity_observed: bool
    cleanup_passed: bool
    budget_accounted: bool
    preservation_integrity_valid: bool = False
    fresh_restore_allocations: bool = False
    all_branches_resumed: bool = False
    recompute_valid: bool = False


class MetricSample(_StrictModel):
    metric: NonEmpty
    unit: NonEmpty
    value: float = Field(allow_inf_nan=False)
    provenance: ArtifactSampleRef


class RawTrial(_StrictModel):
    schema_version: Literal["sloforge.branchfabric.experiment-004-raw-trial/v1"] = (
        "sloforge.branchfabric.experiment-004-raw-trial/v1"
    )
    trial_id: NonEmpty
    seed: int = Field(ge=0, le=2**63 - 1)
    mode: ReclamationMode
    order_position: int = Field(ge=1, le=3)
    tracing_level: Literal["disabled", "minimal", "full"]
    pilot_digest: Sha256
    workload_digest: Sha256
    gpu_uuids: tuple[GpuUuid, GpuUuid]
    semantic_evidence: TrialSemanticEvidence
    metrics: tuple[MetricSample, ...] = Field(min_length=1, max_length=256)
    raw_manifest: ArtifactSampleRef

    @model_validator(mode="after")
    def unique_metrics_and_gpus(self) -> Self:
        keys = tuple((metric.metric, metric.unit) for metric in self.metrics)
        if len(keys) != len(set(keys)):
            raise ValueError("raw trial metric/unit keys must be unique")
        if len(set(self.gpu_uuids)) != 2:
            raise ValueError("raw trial requires two distinct GPU UUIDs")
        return self


class TrialValidity(_StrictModel):
    trial_id: NonEmpty
    causal_sample_accepted: bool
    reasons: tuple[NonEmpty, ...]


def evaluate_trial_validity(
    trial: RawTrial,
    *,
    planned: TrialPlan,
    plan_pilot_digest: str,
    trace_gate: TraceControlGate,
) -> TrialValidity:
    """Apply all causal and semantic gates without silently dropping a trial."""

    reasons: list[str] = []
    if (trial.trial_id, trial.seed, trial.mode, trial.order_position) != (
        planned.trial_id,
        planned.seed,
        planned.mode,
        planned.order_position,
    ):
        reasons.append("raw trial identity does not match the immutable campaign plan")
    if trial.pilot_digest != plan_pilot_digest:
        reasons.append("raw trial is not bound to the validated pilot")
    if trial.gpu_uuids != trace_gate.gpu_uuids:
        reasons.append("raw trial GPU pair differs from trace controls")
    if trial.workload_digest != trace_gate.workload_digest:
        reasons.append("raw trial workload differs from trace controls")
    if trial.tracing_level == "full":
        reasons.append("full tracing is diagnostic-only")
    elif trial.tracing_level == "minimal" and not trace_gate.minimal_trace_causal_trials_allowed:
        reasons.append("minimal tracing failed the paired disabled-tracing control")

    common = {
        "transaction did not complete": trial.semantic_evidence.transaction_completed,
        "serving scenario was invalid": trial.semantic_evidence.serving_scenario_valid,
        "hardware identity was invalid": trial.semantic_evidence.hardware_identity_valid,
        "workload replay was invalid": trial.semantic_evidence.workload_replay_valid,
        "critical stages overlapped or were incomplete": (
            trial.semantic_evidence.critical_timelines_nonoverlapping
        ),
        "movement accounting was invalid": trial.semantic_evidence.movement_accounting_valid,
        "allocator reclaim was not confirmed": trial.semantic_evidence.allocator_reclaim_valid,
        "GPU1 did not serve a real eligible request": (
            trial.semantic_evidence.gpu1_useful_serving_capacity_observed
        ),
        "postflight cleanup failed": trial.semantic_evidence.cleanup_passed,
        "GPU use was not entered in the ledger": trial.semantic_evidence.budget_accounted,
    }
    reasons.extend(reason for reason, passed in common.items() if not passed)
    if trial.mode is ReclamationMode.KILL_AND_RECOMPUTE:
        if not trial.semantic_evidence.recompute_valid:
            reasons.append("kill path recomputation did not validate")
    else:
        preservation = {
            "transport integrity did not validate": (
                trial.semantic_evidence.preservation_integrity_valid
            ),
            "destination allocations were not fresh": (
                trial.semantic_evidence.fresh_restore_allocations
            ),
            "not all branches resumed": trial.semantic_evidence.all_branches_resumed,
        }
        reasons.extend(reason for reason, passed in preservation.items() if not passed)
    return TrialValidity(
        trial_id=trial.trial_id,
        causal_sample_accepted=not reasons,
        reasons=tuple(reasons),
    )


def _metric_sample(trial: RawTrial, metric: str, unit: str) -> MetricSample:
    matches = [sample for sample in trial.metrics if (sample.metric, sample.unit) == (metric, unit)]
    if len(matches) != 1:
        raise ValueError(f"trial {trial.trial_id} does not contain exactly one {metric} [{unit}]")
    return matches[0]


def analyze_final_campaign(
    *,
    plan: FinalCampaignPlan,
    trials: Sequence[RawTrial],
    trace_gate: TraceControlGate,
    metric: str,
    unit: str,
) -> dict[str, object]:
    """Summarize the complete 3x3 campaign using raw matched-seed differences.

    With only three matched seeds, p95/p99 and a two-sided distribution-free
    95% confidence interval are not supported.  The output says so explicitly
    instead of presenting a fragile bootstrap interval as population evidence.
    """

    if len(trials) != 9:
        raise ValueError("final paired analysis requires all nine planned trials")
    plan_by_id = {trial.trial_id: trial for trial in plan.trials}
    if set(plan_by_id) != {trial.trial_id for trial in trials}:
        raise ValueError("raw trial IDs do not exactly match the immutable campaign plan")
    validity = tuple(
        evaluate_trial_validity(
            trial,
            planned=plan_by_id[trial.trial_id],
            plan_pilot_digest=plan.pilot_digest,
            trace_gate=trace_gate,
        )
        for trial in trials
    )
    invalid = tuple(item for item in validity if not item.causal_sample_accepted)
    if invalid:
        details = "; ".join(f"{item.trial_id}: {', '.join(item.reasons)}" for item in invalid)
        raise ValueError(f"final campaign contains invalid causal trials: {details}")
    gpu_pairs = {trial.gpu_uuids for trial in trials}
    workload_digests = {trial.workload_digest for trial in trials}
    if len(gpu_pairs) != 1:
        raise ValueError("paired campaign did not use one fixed two-GPU allocation")
    if len(workload_digests) != 1:
        raise ValueError("paired campaign workload digest changed between trials")

    cells: dict[tuple[int, ReclamationMode], tuple[RawTrial, MetricSample]] = {}
    for trial in trials:
        key = (trial.seed, trial.mode)
        if key in cells:
            raise ValueError("duplicate seed/mode cell in raw trials")
        cells[key] = (trial, _metric_sample(trial, metric, unit))

    seed_order = tuple(dict.fromkeys(item.seed for item in plan.trials))
    modes: dict[str, object] = {}
    for mode in _MODES:
        values = [cells[(seed, mode)][1].value for seed in seed_order]
        modes[mode.value] = {
            "values": values,
            "median": statistics.median(values),
            "mean": statistics.fmean(values),
            "minimum": min(values),
            "maximum": max(values),
            "provenance": [
                cells[(seed, mode)][1].provenance.model_dump(mode="json") for seed in seed_order
            ],
        }

    pairwise: list[dict[str, object]] = []
    for baseline_mode, candidate_mode in combinations(_MODES, 2):
        raw: list[dict[str, object]] = []
        differences: list[float] = []
        relative_changes: list[float | None] = []
        for seed in seed_order:
            baseline_trial, baseline = cells[(seed, baseline_mode)]
            candidate_trial, candidate = cells[(seed, candidate_mode)]
            difference = candidate.value - baseline.value
            relative = _relative_change(candidate.value, baseline.value)
            differences.append(difference)
            relative_changes.append(relative)
            raw.append(
                {
                    "seed": seed,
                    "baseline_trial_id": baseline_trial.trial_id,
                    "candidate_trial_id": candidate_trial.trial_id,
                    "baseline_value": baseline.value,
                    "candidate_value": candidate.value,
                    "candidate_minus_baseline": difference,
                    "relative_change": relative,
                    "baseline_provenance": baseline.provenance.model_dump(mode="json"),
                    "candidate_provenance": candidate.provenance.model_dump(mode="json"),
                }
            )
        available_relative = [value for value in relative_changes if value is not None]
        pairwise.append(
            {
                "baseline_mode": baseline_mode.value,
                "candidate_mode": candidate_mode.value,
                "direction": "candidate minus baseline",
                "raw_pairs": raw,
                "differences": differences,
                "summary": {
                    "median_difference": statistics.median(differences),
                    "mean_difference": statistics.fmean(differences),
                    "minimum_difference": min(differences),
                    "maximum_difference": max(differences),
                    "median_relative_change": (
                        statistics.median(available_relative) if available_relative else None
                    ),
                    "outlier_policy": "none removed",
                },
                "confidence_interval": {
                    "status": "UNAVAILABLE",
                    "confidence_level": 0.95,
                    "reason": (
                        "three matched differences cannot support a two-sided "
                        "distribution-free 95% interval; no parametric normality "
                        "assumption or bootstrap population claim is made"
                    ),
                },
                "tail_statistics": {
                    "p95": None,
                    "p99": None,
                    "status": "TAIL_INSUFFICIENT",
                    "reason": "three matched seeds do not support tail-percentile inference",
                },
            }
        )
    return cast(
        dict[str, object],
        json.loads(
            json.dumps(
                {
                    "schema_version": ("sloforge.branchfabric.experiment-004-paired-analysis/v1"),
                    "metric": metric,
                    "unit": unit,
                    "seed_order": seed_order,
                    "gpu_uuids": next(iter(gpu_pairs)),
                    "workload_digest": next(iter(workload_digests)),
                    "trace_control_id": trace_gate.control_id,
                    "accepted_trial_count": 9,
                    "validity": [item.model_dump(mode="json") for item in validity],
                    "modes": modes,
                    "pairwise": pairwise,
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        ),
    )


class GpuInvocationReservation(_StrictModel):
    reservation_id: NonEmpty
    invocation_id: NonEmpty
    requested_gpu: Literal["A100-80GB"] = "A100-80GB"
    gpu_count: Literal[2] = 2
    maximum_wall_seconds: float = Field(gt=0.0, le=3_600.0, allow_inf_nan=False)
    config_sha256: Sha256

    @property
    def maximum_gpu_seconds(self) -> float:
        return self.maximum_wall_seconds * self.gpu_count


class GpuActiveInterval(_StrictModel):
    invocation_id: NonEmpty
    function_call_id: NonEmpty
    requested_gpu: Literal["A100-80GB"] = "A100-80GB"
    actual_gpu_models: tuple[NonEmpty, NonEmpty]
    gpu_uuids: tuple[GpuUuid, GpuUuid]
    gpu_count: Literal[2] = 2
    accounted_wall_seconds: float = Field(gt=0.0, le=3_600.0, allow_inf_nan=False)
    gpu_price_per_hour_usd: float | None = Field(
        default=None,
        gt=0.0,
        allow_inf_nan=False,
    )
    raw_manifest: ArtifactSampleRef

    @model_validator(mode="after")
    def exact_hardware(self) -> Self:
        if len(set(self.gpu_uuids)) != 2:
            raise ValueError("settled invocation requires two distinct GPU UUIDs")
        if any(
            "A100" not in model or "80GB" not in model.replace(" ", "")
            for model in self.actual_gpu_models
        ):
            raise ValueError("settled invocation did not receive two A100-80GB GPUs")
        return self

    @property
    def gpu_seconds(self) -> float:
        return self.accounted_wall_seconds * self.gpu_count

    @property
    def gpu_cost_usd(self) -> float | None:
        if self.gpu_price_per_hour_usd is None:
            return None
        return self.gpu_seconds / 3_600.0 * self.gpu_price_per_hour_usd


class Experiment004GpuHourLedger(_StrictModel):
    schema_version: Literal["sloforge.branchfabric.experiment-004-gpu-hours/v1"] = (
        "sloforge.branchfabric.experiment-004-gpu-hours/v1"
    )
    historical_through_experiment_003_gpu_hours: float = HISTORICAL_THROUGH_EXPERIMENT_003_GPU_HOURS
    target_additional_gpu_seconds: float = TARGET_ADDITIONAL_GPU_SECONDS
    hard_additional_gpu_seconds: float = HARD_ADDITIONAL_GPU_SECONDS
    consumed_additional_gpu_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    intervals: tuple[GpuActiveInterval, ...] = ()
    reservations: tuple[GpuInvocationReservation, ...] = ()

    @model_validator(mode="after")
    def ledger_is_derived_and_bounded(self) -> Self:
        constants = (
            (
                "historical_through_experiment_003_gpu_hours",
                self.historical_through_experiment_003_gpu_hours,
                HISTORICAL_THROUGH_EXPERIMENT_003_GPU_HOURS,
            ),
            (
                "target_additional_gpu_seconds",
                self.target_additional_gpu_seconds,
                TARGET_ADDITIONAL_GPU_SECONDS,
            ),
            (
                "hard_additional_gpu_seconds",
                self.hard_additional_gpu_seconds,
                HARD_ADDITIONAL_GPU_SECONDS,
            ),
        )
        for field, actual, expected in constants:
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"{field} is fixed by the Experiment 004 contract")
        derived = sum(interval.gpu_seconds for interval in self.intervals)
        if not math.isclose(
            self.consumed_additional_gpu_seconds,
            derived,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("consumed additional GPU-seconds must equal settled intervals")
        if self.consumed_additional_gpu_seconds > self.hard_additional_gpu_seconds:
            raise ValueError("settled GPU use exceeds the Experiment 004 hard limit")
        if len({item.function_call_id for item in self.intervals}) != len(self.intervals):
            raise ValueError("function call IDs must be unique in the GPU ledger")
        if len({item.reservation_id for item in self.reservations}) != len(self.reservations):
            raise ValueError("reservation IDs must be unique in the GPU ledger")
        invocation_ids = [item.invocation_id for item in self.intervals]
        invocation_ids.extend(item.invocation_id for item in self.reservations)
        if len(set(invocation_ids)) != len(invocation_ids):
            raise ValueError("invocation IDs cannot be repeated or both settled and reserved")
        committed = self.consumed_additional_gpu_seconds + sum(
            reservation.maximum_gpu_seconds for reservation in self.reservations
        )
        if committed > self.hard_additional_gpu_seconds:
            raise ValueError("settled plus reserved GPU use exceeds the hard limit")
        if len(self.reservations) > 1:
            raise ValueError("the sole GPU coordinator may hold at most one reservation")
        return self

    @property
    def consumed_additional_gpu_hours(self) -> float:
        return self.consumed_additional_gpu_seconds / 3_600.0

    @property
    def committed_additional_gpu_seconds(self) -> float:
        return self.consumed_additional_gpu_seconds + sum(
            reservation.maximum_gpu_seconds for reservation in self.reservations
        )


class GpuBudgetPreflight(_StrictModel):
    invocation_id: NonEmpty
    gpu_count: Literal[2] = 2
    requested_gpu: Literal["A100-80GB"] = "A100-80GB"
    maximum_wall_seconds: float = Field(gt=0.0, le=3_600.0, allow_inf_nan=False)
    proposed_maximum_gpu_seconds: float = Field(gt=0.0, allow_inf_nan=False)
    consumed_before_gpu_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    committed_before_gpu_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    projected_committed_gpu_seconds: float = Field(gt=0.0, allow_inf_nan=False)
    remaining_hard_budget_after_gpu_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    target_exceeded_if_fully_consumed: bool
    hard_limit_passed: Literal[True] = True


def reserve_gpu_invocation(
    ledger: Experiment004GpuHourLedger,
    *,
    reservation_id: str,
    invocation_id: str,
    maximum_wall_seconds: float,
    config_sha256: str,
) -> tuple[Experiment004GpuHourLedger, GpuBudgetPreflight]:
    """Atomically model one explicit two-A100 reservation after budget preflight."""

    if ledger.reservations:
        raise ValueError("another GPU invocation reservation is already active")
    reservation = GpuInvocationReservation(
        reservation_id=reservation_id,
        invocation_id=invocation_id,
        maximum_wall_seconds=maximum_wall_seconds,
        config_sha256=config_sha256,
    )
    committed_before = ledger.committed_additional_gpu_seconds
    projected = committed_before + reservation.maximum_gpu_seconds
    if projected > ledger.hard_additional_gpu_seconds:
        raise ValueError("GPU invocation preflight would exceed 2.0 additional A100-hours")
    preflight = GpuBudgetPreflight(
        invocation_id=invocation_id,
        maximum_wall_seconds=maximum_wall_seconds,
        proposed_maximum_gpu_seconds=reservation.maximum_gpu_seconds,
        consumed_before_gpu_seconds=ledger.consumed_additional_gpu_seconds,
        committed_before_gpu_seconds=committed_before,
        projected_committed_gpu_seconds=projected,
        remaining_hard_budget_after_gpu_seconds=(ledger.hard_additional_gpu_seconds - projected),
        target_exceeded_if_fully_consumed=projected > ledger.target_additional_gpu_seconds,
    )
    updated = ledger.model_copy(update={"reservations": (*ledger.reservations, reservation)})
    return Experiment004GpuHourLedger.model_validate(updated.model_dump(), strict=True), preflight


def settle_gpu_invocation(
    ledger: Experiment004GpuHourLedger,
    *,
    reservation_id: str,
    function_call_id: str,
    actual_gpu_models: tuple[str, str],
    gpu_uuids: tuple[str, str],
    client_elapsed_seconds: float,
    remote_observed_allocation_seconds: float,
    raw_manifest: ArtifactSampleRef,
    gpu_price_per_hour_usd: float | None = None,
) -> Experiment004GpuHourLedger:
    """Settle against the conservative maximum observed allocation duration."""

    matches = [item for item in ledger.reservations if item.reservation_id == reservation_id]
    if len(matches) != 1:
        raise ValueError("GPU reservation is absent or duplicated")
    reservation = matches[0]
    for field, value in (
        ("client_elapsed_seconds", client_elapsed_seconds),
        ("remote_observed_allocation_seconds", remote_observed_allocation_seconds),
    ):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{field} must be finite and positive")
    accounted_wall = max(client_elapsed_seconds, remote_observed_allocation_seconds)
    if accounted_wall > reservation.maximum_wall_seconds:
        raise ValueError("accounted GPU allocation exceeded its preflight wall-time bound")
    interval = GpuActiveInterval(
        invocation_id=reservation.invocation_id,
        function_call_id=function_call_id,
        actual_gpu_models=actual_gpu_models,
        gpu_uuids=gpu_uuids,
        accounted_wall_seconds=accounted_wall,
        gpu_price_per_hour_usd=gpu_price_per_hour_usd,
        raw_manifest=raw_manifest,
    )
    intervals = (*ledger.intervals, interval)
    reservations = tuple(
        item for item in ledger.reservations if item.reservation_id != reservation_id
    )
    return Experiment004GpuHourLedger(
        consumed_additional_gpu_seconds=sum(item.gpu_seconds for item in intervals),
        intervals=intervals,
        reservations=reservations,
    )


__all__ = [
    "HARD_ADDITIONAL_GPU_SECONDS",
    "HISTORICAL_THROUGH_EXPERIMENT_003_GPU_HOURS",
    "ORDER_ALGORITHM",
    "PRIMARY_SEEDS",
    "TARGET_ADDITIONAL_GPU_SECONDS",
    "ArtifactSampleRef",
    "Experiment004GpuHourLedger",
    "FinalCampaignPlan",
    "GpuActiveInterval",
    "GpuBudgetPreflight",
    "GpuInvocationReservation",
    "MetricSample",
    "RawTrial",
    "TraceControlGate",
    "TraceMetricControl",
    "TraceMetricEvaluation",
    "TrialPlan",
    "TrialSemanticEvidence",
    "TrialValidity",
    "analyze_final_campaign",
    "evaluate_trace_controls",
    "evaluate_trial_validity",
    "plan_final_campaign",
    "reserve_gpu_invocation",
    "settle_gpu_invocation",
]
