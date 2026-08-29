"""Evidence-derived GPU budget for Experiment 004 v10.

The v10 reservation is intentionally derived from immutable historical phase
measurements.  This module validates both the arithmetic and every referenced
source hash before the sole Modal coordinator is allowed to reserve GPU time.
It contains no cloud calls.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

PHASE_ORDER = (
    "modal_function_startup",
    "two_engine_readiness_sanity_check",
    "rollout_state_preparation",
    "control_interval",
    "overload_interval",
    "branch_quiesce",
    "state_capture",
    "transformation",
    "d2h",
    "validation_publish",
    "gpu1_hbm_release",
    "serving_recovery",
    "slo_stability_window",
    "restore_preparation",
    "h2d",
    "state_import",
    "destination_write",
    "state_validation",
    "branch_continuation",
    "cleanup",
)
PhaseName = Literal[
    "modal_function_startup",
    "two_engine_readiness_sanity_check",
    "rollout_state_preparation",
    "control_interval",
    "overload_interval",
    "branch_quiesce",
    "state_capture",
    "transformation",
    "d2h",
    "validation_publish",
    "gpu1_hbm_release",
    "serving_recovery",
    "slo_stability_window",
    "restore_preparation",
    "h2d",
    "state_import",
    "destination_write",
    "state_validation",
    "branch_continuation",
    "cleanup",
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class SourceArtifact(_StrictModel):
    artifact_reference: NonEmpty
    artifact_sha256: Sha256
    sample_selector: NonEmpty


class SourceMeasurement(_StrictModel):
    description: NonEmpty
    derivation: NonEmpty
    artifacts: tuple[SourceArtifact, ...] = Field(min_length=1, max_length=8)


class PhaseBound(_StrictModel):
    phase: PhaseName
    source_measurement: SourceMeasurement
    observed_value_seconds: float = Field(gt=0.0, allow_inf_nan=False)
    selected_upper_bound_seconds: float = Field(gt=0.0, allow_inf_nan=False)
    multiplier: float = Field(ge=1.0, allow_inf_nan=False)
    margin_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    reasoning: NonEmpty

    @model_validator(mode="after")
    def arithmetic_is_derived(self) -> Self:
        expected_multiplier = self.selected_upper_bound_seconds / self.observed_value_seconds
        expected_margin = self.selected_upper_bound_seconds - self.observed_value_seconds
        if self.selected_upper_bound_seconds < self.observed_value_seconds:
            raise ValueError("phase upper bound is below its source measurement")
        if not math.isclose(self.multiplier, expected_multiplier, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("phase multiplier is not derived from observed and upper values")
        if not math.isclose(self.margin_seconds, expected_margin, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("phase margin is not derived from observed and upper values")
        return self


class SafetyMargin(_StrictModel):
    source_measurement: NonEmpty
    observed_phase_upper_bound_subtotal_seconds: float = Field(gt=0.0, allow_inf_nan=False)
    selected_margin_fraction: float = Field(gt=0.0, lt=1.0, allow_inf_nan=False)
    multiplier: float = Field(gt=0.0, lt=1.0, allow_inf_nan=False)
    sdk_timeout_rounding_margin_seconds: float = Field(ge=0.0, lt=1.0, allow_inf_nan=False)
    selected_upper_bound_seconds: float = Field(gt=0.0, allow_inf_nan=False)
    reasoning: NonEmpty

    @model_validator(mode="after")
    def arithmetic_is_derived(self) -> Self:
        if not math.isclose(
            self.multiplier,
            self.selected_margin_fraction,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("safety multiplier differs from its selected margin")
        expected = (
            self.observed_phase_upper_bound_subtotal_seconds * self.selected_margin_fraction
            + self.sdk_timeout_rounding_margin_seconds
        )
        if not math.isclose(
            self.selected_upper_bound_seconds,
            expected,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("safety upper bound is not derived from the phase subtotal")
        return self


class BudgetGateSnapshot(_StrictModel):
    ledger_artifact_reference: NonEmpty
    current_conservative_usage_A100_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    configured_hard_ceiling_A100_seconds: float = Field(gt=0.0, allow_inf_nan=False)
    required_A100_seconds: float = Field(gt=0.0, allow_inf_nan=False)
    projected_conservative_usage_A100_seconds: float = Field(gt=0.0, allow_inf_nan=False)
    remaining_A100_seconds_after_reservation: float = Field(ge=0.0, allow_inf_nan=False)
    condition: Literal[
        "current_conservative_usage_A100_seconds + required_A100_seconds <= configured_hard_ceiling_A100_seconds"
    ]
    decision: Literal["AUTHORIZE_V10", "BUDGET_BLOCKER"]

    @model_validator(mode="after")
    def decision_is_derived(self) -> Self:
        projected = self.current_conservative_usage_A100_seconds + self.required_A100_seconds
        if not math.isclose(
            self.projected_conservative_usage_A100_seconds,
            projected,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("budget snapshot projected usage is not derived")
        remaining = self.configured_hard_ceiling_A100_seconds - projected
        passed = remaining >= -1e-9
        if not passed:
            remaining = 0.0
        if not math.isclose(
            self.remaining_A100_seconds_after_reservation,
            remaining,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("budget snapshot remaining usage is not derived")
        if self.decision != ("AUTHORIZE_V10" if passed else "BUDGET_BLOCKER"):
            raise ValueError("budget snapshot decision is not derived")
        return self


class V10PhaseBudget(_StrictModel):
    schema_version: Literal["sloforge.branchfabric.experiment-004-v10-phase-budget/v1"]
    status: Literal["EVIDENCE_DERIVED"]
    gpu_count: Literal[2]
    allocation_semantics: Literal["two A100s remain allocated for the bounded attempt"]
    legacy_fixed_1800_A100_second_reservation_removed: Literal[True]
    phases: tuple[PhaseBound, ...] = Field(min_length=len(PHASE_ORDER), max_length=len(PHASE_ORDER))
    phase_upper_bound_subtotal_wall_seconds: float = Field(gt=0.0, allow_inf_nan=False)
    safety_margin: SafetyMargin
    predicted_wall_clock_seconds: float = Field(gt=0.0, allow_inf_nan=False)
    required_A100_seconds: float = Field(gt=0.0, allow_inf_nan=False)
    required_A100_hours: float = Field(gt=0.0, allow_inf_nan=False)
    formula: Literal["required_A100_seconds = 2 * predicted_wall_clock_seconds"]
    budget_gate_snapshot: BudgetGateSnapshot

    @model_validator(mode="after")
    def complete_and_derived(self) -> Self:
        if tuple(item.phase for item in self.phases) != PHASE_ORDER:
            raise ValueError("phase budget must contain every required phase in canonical order")
        subtotal = sum(item.selected_upper_bound_seconds for item in self.phases)
        if not math.isclose(
            self.phase_upper_bound_subtotal_wall_seconds,
            subtotal,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("phase upper-bound subtotal is not derived")
        if not math.isclose(
            self.safety_margin.observed_phase_upper_bound_subtotal_seconds,
            subtotal,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("safety margin is based on a different phase subtotal")
        predicted = subtotal + self.safety_margin.selected_upper_bound_seconds
        required = predicted * self.gpu_count
        if not math.isclose(
            self.predicted_wall_clock_seconds, predicted, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError("predicted wall clock is not phase subtotal plus safety margin")
        if not math.isclose(self.required_A100_seconds, required, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("required A100-seconds is not two times predicted wall clock")
        if not math.isclose(
            self.required_A100_hours, required / 3_600.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("required A100-hours is not derived")
        if not math.isclose(
            self.budget_gate_snapshot.required_A100_seconds,
            required,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("budget gate snapshot uses a different reservation")
        return self

    @property
    def cleanup_upper_bound_seconds(self) -> float:
        return next(
            phase.selected_upper_bound_seconds for phase in self.phases if phase.phase == "cleanup"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_v10_phase_budget(path: Path, *, repository_root: Path) -> V10PhaseBudget:
    """Load the budget, verify its arithmetic, and hash-check every source."""

    resolved_root = repository_root.resolve(strict=True)
    resolved_path = path.resolve(strict=True)
    if resolved_path.is_symlink() or not resolved_path.is_file():
        raise ValueError("v10 phase budget must be a regular non-symlink file")
    budget = V10PhaseBudget.model_validate_json(resolved_path.read_text(), strict=True)
    checked: set[tuple[str, str]] = set()
    for phase in budget.phases:
        for source in phase.source_measurement.artifacts:
            key = (source.artifact_reference, source.artifact_sha256)
            if key in checked:
                continue
            relative = Path(source.artifact_reference)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("phase-budget source artifact reference is unsafe")
            source_path = (resolved_root / relative).resolve(strict=True)
            if (
                not source_path.is_relative_to(resolved_root)
                or source_path.is_symlink()
                or not source_path.is_file()
            ):
                raise ValueError("phase-budget source artifact escapes the repository")
            if _sha256(source_path) != source.artifact_sha256:
                raise ValueError(f"phase-budget source hash mismatch: {source.artifact_reference}")
            checked.add(key)
    return budget


def evaluate_budget_gate(
    budget: V10PhaseBudget,
    *,
    current_conservative_usage_A100_seconds: float,
    configured_hard_ceiling_A100_seconds: float,
) -> dict[str, float | str | bool]:
    """Evaluate the live ledger without weakening its configured hard ceiling."""

    for field, value in (
        ("current conservative usage", current_conservative_usage_A100_seconds),
        ("configured hard ceiling", configured_hard_ceiling_A100_seconds),
    ):
        if isinstance(value, bool) or not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{field} must be finite and non-negative")
    projected = current_conservative_usage_A100_seconds + budget.required_A100_seconds
    passed = projected <= configured_hard_ceiling_A100_seconds
    return {
        "status": "AUTHORIZE_V10" if passed else "BUDGET_BLOCKER",
        "budget_pass": passed,
        "current_conservative_usage_A100_seconds": current_conservative_usage_A100_seconds,
        "required_A100_seconds": budget.required_A100_seconds,
        "projected_conservative_usage_A100_seconds": projected,
        "configured_hard_ceiling_A100_seconds": configured_hard_ceiling_A100_seconds,
        "remaining_A100_seconds_after_reservation": max(
            0.0, configured_hard_ceiling_A100_seconds - projected
        ),
    }


def phase_budget_sha256(path: Path) -> str:
    return _sha256(path.resolve(strict=True))


__all__ = [
    "PHASE_ORDER",
    "V10PhaseBudget",
    "evaluate_budget_gate",
    "load_v10_phase_budget",
    "phase_budget_sha256",
]
