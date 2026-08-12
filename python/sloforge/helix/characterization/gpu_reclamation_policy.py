"""Measured Helix policy for reclaiming a GPU that owns rollout state.

The policy is intentionally small.  It consumes rates measured by Experiment
004, estimates the three implemented reclamation paths, and makes one
deterministic choice.  It does not turn projections into measurements: all rate
inputs retain references to the samples that produced them and every result
retains the input digest.

Expected future rollout work is reported but is not added to the comparison.
That work is identical after all three successful restore paths and therefore
cancels.  A group with no remaining work is killed instead of checkpointed.
"""

from __future__ import annotations

import hashlib
import json
import math
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sloforge.helix.characterization.matrix import EvidenceClass
from sloforge.helix.scheduler.models import EvidenceRef

MAX_STATE_BYTES = 1 << 40
MAX_BRANCHES = 4096
MAX_TOKENS_PER_FIELD = 1 << 30
MAX_AGGREGATE_TOKENS = 1 << 42
MAX_SECONDS = 24 * 60 * 60
MAX_RATE = float(1 << 60)


class ReclamationPolicyModel(BaseModel):
    """Strict, immutable model used at the measured-policy boundary."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )


class ReclamationAction(StrEnum):
    """Exact resource-compiler choices supported by Experiment 004."""

    KILL_AND_RECOMPUTE = "KILL_AND_RECOMPUTE"
    PRESERVE_NAIVE = "PRESERVE_NAIVE"
    PRESERVE_OPTIMIZED = "PRESERVE_OPTIMIZED"


class BreakEvenDirection(StrEnum):
    """The side of a history-token threshold on which preservation wins."""

    AT_OR_ABOVE = "AT_OR_ABOVE"
    AT_OR_BELOW = "AT_OR_BELOW"
    ALL_HISTORY_SIZES = "ALL_HISTORY_SIZES"
    NO_HISTORY_SIZE = "NO_HISTORY_SIZE"


class ReclamationWorkload(ReclamationPolicyModel):
    """Bounded branch-group and serving-urgency inputs."""

    schema_version: Literal["sloforge.helix.gpu-reclamation-workload/v1"] = (
        "sloforge.helix.gpu-reclamation-workload/v1"
    )
    seed: int = Field(ge=0, le=2**63 - 1)
    logical_state_bytes: int = Field(gt=0, le=MAX_STATE_BYTES)
    prefix_tokens: int = Field(gt=0, le=MAX_TOKENS_PER_FIELD)
    branch_count: int = Field(gt=0, le=MAX_BRANCHES)
    private_tokens_per_branch: int = Field(ge=0, le=MAX_TOKENS_PER_FIELD)
    expected_remaining_tokens_per_branch: int = Field(ge=0, le=MAX_TOKENS_PER_FIELD)
    serving_urgency_seconds: float = Field(
        gt=0.0,
        le=MAX_SECONDS,
        allow_inf_nan=False,
    )
    environment_reconstruction_seconds: float = Field(
        default=0.0,
        ge=0.0,
        le=MAX_SECONDS,
        allow_inf_nan=False,
    )
    gpu_hour_price_usd: float | None = Field(
        default=None,
        gt=0.0,
        le=1_000_000.0,
        allow_inf_nan=False,
    )

    @model_validator(mode="after")
    def validate_aggregate_tokens(self) -> Self:
        if self.history_tokens > MAX_AGGREGATE_TOKENS:
            raise ValueError("aggregate reconstructable history exceeds the policy bound")
        if self.remaining_tokens > MAX_AGGREGATE_TOKENS:
            raise ValueError("aggregate remaining rollout work exceeds the policy bound")
        return self

    @property
    def history_tokens(self) -> int:
        """Tokens that must be replayed after discarding native KV state."""

        return self.prefix_tokens + self.branch_count * self.private_tokens_per_branch

    @property
    def lost_rollout_work_tokens(self) -> int:
        """Generated branch-private work discarded by the kill path."""

        return self.branch_count * self.private_tokens_per_branch

    @property
    def remaining_tokens(self) -> int:
        return self.branch_count * self.expected_remaining_tokens_per_branch


class ReclamationMeasurements(ReclamationPolicyModel):
    """Measured effective rates and fixed costs for one runtime/topology scope.

    Rates are effective end-to-end rates for the named path, not theoretical
    link bandwidth.  Fixed checkpoint/restore costs contain measured work that
    is not proportional to logical state bytes.  All seconds are
    non-overlapping within their respective path.
    """

    schema_version: Literal["sloforge.helix.gpu-reclamation-measurements/v1"] = (
        "sloforge.helix.gpu-reclamation-measurements/v1"
    )
    kill_release_seconds: float = Field(ge=0.0, le=MAX_SECONDS, allow_inf_nan=False)
    first_resumed_token_seconds: float = Field(ge=0.0, le=MAX_SECONDS, allow_inf_nan=False)
    recompute_tokens_per_second: float = Field(ge=1.0, le=MAX_RATE, allow_inf_nan=False)
    naive_checkpoint_bytes_per_second: float = Field(ge=1.0, le=MAX_RATE, allow_inf_nan=False)
    naive_restore_bytes_per_second: float = Field(ge=1.0, le=MAX_RATE, allow_inf_nan=False)
    optimized_checkpoint_bytes_per_second: float = Field(ge=1.0, le=MAX_RATE, allow_inf_nan=False)
    optimized_restore_bytes_per_second: float = Field(ge=1.0, le=MAX_RATE, allow_inf_nan=False)
    naive_checkpoint_fixed_seconds: float = Field(
        default=0.0, ge=0.0, le=MAX_SECONDS, allow_inf_nan=False
    )
    naive_restore_fixed_seconds: float = Field(
        default=0.0, ge=0.0, le=MAX_SECONDS, allow_inf_nan=False
    )
    optimized_checkpoint_fixed_seconds: float = Field(
        default=0.0, ge=0.0, le=MAX_SECONDS, allow_inf_nan=False
    )
    optimized_restore_fixed_seconds: float = Field(
        default=0.0, ge=0.0, le=MAX_SECONDS, allow_inf_nan=False
    )
    evidence_class: EvidenceClass
    evidence: tuple[EvidenceRef, ...] = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def validate_evidence_is_unique(self) -> Self:
        identities = tuple(
            (item.artifact_uri, item.artifact_sha256, item.sample_ids) for item in self.evidence
        )
        if len(identities) != len(set(identities)):
            raise ValueError("measurement evidence references must be unique")
        return self


class ReclamationPathEstimate(ReclamationPolicyModel):
    """Estimated incremental transition cost for one measured path."""

    action: ReclamationAction
    reclamation_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    rollout_resume_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    full_transition_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    meets_serving_urgency: bool
    recompute_tokens: int = Field(ge=0)
    lost_rollout_work_tokens: int = Field(ge=0)
    recompute_gpu_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    transition_gpu_occupancy_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    transition_gpu_cost_usd: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    expected_remaining_tokens: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_decomposition(self) -> Self:
        if not math.isclose(
            self.full_transition_seconds,
            self.reclamation_seconds + self.rollout_resume_seconds,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("full transition must equal reclamation plus rollout resume")
        if self.action is ReclamationAction.KILL_AND_RECOMPUTE:
            if self.recompute_tokens == 0 or self.recompute_gpu_seconds <= 0.0:
                raise ValueError("kill/recompute must account for replayed tokens and GPU time")
        elif self.recompute_tokens != 0 or self.lost_rollout_work_tokens != 0:
            raise ValueError("preservation paths cannot report killed or recomputed tokens")
        return self


class PreservationBreakEven(ReclamationPolicyModel):
    """Analytic parity point under linear measured-rate interpolation."""

    preservation_action: ReclamationAction
    direction: BreakEvenDirection
    current_history_tokens: int = Field(gt=0)
    current_logical_state_bytes: int = Field(gt=0)
    logical_bytes_per_history_token: float = Field(gt=0.0, allow_inf_nan=False)
    minimum_history_tokens_at_parity: int | None = Field(default=None, ge=0)
    maximum_history_tokens_at_parity: int | None = Field(default=None, ge=0)
    minimum_prefix_tokens_if_no_private_state: int | None = Field(default=None, ge=0)
    minimum_rollout_age_tokens_per_branch_at_current_prefix: int | None = Field(default=None, ge=0)
    equivalent_logical_state_bytes_at_parity: int | None = Field(default=None, ge=0)
    recompute_gpu_seconds_at_parity: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    recompute_gpu_cost_usd_at_parity: float | None = Field(
        default=None, ge=0.0, allow_inf_nan=False
    )
    preservation_is_no_more_expensive_for_current_workload: bool
    minimum_remaining_tokens_per_branch_for_preservation: int | None = Field(
        default=None, ge=1, le=1
    )
    reason: str = Field(min_length=1, max_length=2048)

    @model_validator(mode="after")
    def validate_threshold_shape(self) -> Self:
        if self.preservation_action is ReclamationAction.KILL_AND_RECOMPUTE:
            raise ValueError("break-even analysis requires a preservation action")
        if self.direction is BreakEvenDirection.AT_OR_ABOVE:
            if self.minimum_history_tokens_at_parity is None:
                raise ValueError("AT_OR_ABOVE requires a minimum history threshold")
        elif self.direction is BreakEvenDirection.AT_OR_BELOW:
            if self.maximum_history_tokens_at_parity is None:
                raise ValueError("AT_OR_BELOW requires a maximum history threshold")
        elif (
            self.minimum_history_tokens_at_parity is not None
            or self.maximum_history_tokens_at_parity is not None
        ):
            raise ValueError("unbounded break-even directions cannot carry token thresholds")
        return self


class ReclamationDecision(ReclamationPolicyModel):
    """Deterministic compiler decision with complete measured input provenance."""

    schema_version: Literal["sloforge.helix.gpu-reclamation-decision/v1"] = (
        "sloforge.helix.gpu-reclamation-decision/v1"
    )
    selected_action: ReclamationAction
    selection_reason: str = Field(min_length=1, max_length=4096)
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_class: EvidenceClass
    evidence: tuple[EvidenceRef, ...] = Field(min_length=1, max_length=256)
    estimates: tuple[ReclamationPathEstimate, ...] = Field(min_length=3, max_length=3)
    break_even: tuple[PreservationBreakEven, ...] = Field(min_length=2, max_length=2)
    common_future_work_cancels: Literal[True] = True

    @model_validator(mode="after")
    def validate_complete_action_set(self) -> Self:
        actions = tuple(estimate.action for estimate in self.estimates)
        if actions != tuple(ReclamationAction):
            raise ValueError("path estimates must appear once in canonical action order")
        break_even_actions = tuple(item.preservation_action for item in self.break_even)
        if break_even_actions != (
            ReclamationAction.PRESERVE_NAIVE,
            ReclamationAction.PRESERVE_OPTIMIZED,
        ):
            raise ValueError("break-even results must cover naive then optimized preservation")
        return self


def _seconds_to_cost(seconds: float, gpu_hour_price_usd: float | None) -> float | None:
    if gpu_hour_price_usd is None:
        return None
    return seconds * gpu_hour_price_usd / 3600.0


def _integer_parity(value: float, *, upper: bool) -> int:
    """Round a floating analytic root without crossing an exact integer root."""

    nearest = round(value)
    if math.isclose(value, nearest, rel_tol=1e-12, abs_tol=1e-12):
        return int(nearest)
    return math.ceil(value) if upper else math.floor(value)


def _preservation_rates(
    measurements: ReclamationMeasurements,
    action: ReclamationAction,
) -> tuple[float, float, float, float]:
    if action is ReclamationAction.PRESERVE_NAIVE:
        return (
            measurements.naive_checkpoint_bytes_per_second,
            measurements.naive_restore_bytes_per_second,
            measurements.naive_checkpoint_fixed_seconds,
            measurements.naive_restore_fixed_seconds,
        )
    if action is ReclamationAction.PRESERVE_OPTIMIZED:
        return (
            measurements.optimized_checkpoint_bytes_per_second,
            measurements.optimized_restore_bytes_per_second,
            measurements.optimized_checkpoint_fixed_seconds,
            measurements.optimized_restore_fixed_seconds,
        )
    raise ValueError("kill/recompute does not have preservation transfer rates")


def estimate_reclamation_paths(
    workload: ReclamationWorkload,
    measurements: ReclamationMeasurements,
) -> tuple[ReclamationPathEstimate, ...]:
    """Estimate all paths in canonical action order without sampling or I/O."""

    estimates = [estimate_kill_and_recompute(workload, measurements)]
    for action in (
        ReclamationAction.PRESERVE_NAIVE,
        ReclamationAction.PRESERVE_OPTIMIZED,
    ):
        checkpoint_rate, restore_rate, checkpoint_fixed, restore_fixed = _preservation_rates(
            measurements, action
        )
        reclamation = checkpoint_fixed + workload.logical_state_bytes / checkpoint_rate
        resume = (
            restore_fixed
            + workload.logical_state_bytes / restore_rate
            + measurements.first_resumed_token_seconds
        )
        total = reclamation + resume
        estimates.append(
            ReclamationPathEstimate(
                action=action,
                reclamation_seconds=reclamation,
                rollout_resume_seconds=resume,
                full_transition_seconds=total,
                meets_serving_urgency=reclamation <= workload.serving_urgency_seconds,
                recompute_tokens=0,
                lost_rollout_work_tokens=0,
                recompute_gpu_seconds=0.0,
                transition_gpu_occupancy_seconds=total,
                transition_gpu_cost_usd=_seconds_to_cost(total, workload.gpu_hour_price_usd),
                expected_remaining_tokens=workload.remaining_tokens,
            )
        )
    return tuple(estimates)


def estimate_kill_and_recompute(
    workload: ReclamationWorkload,
    measurements: ReclamationMeasurements,
) -> ReclamationPathEstimate:
    """Estimate the measured discard-and-replay baseline.

    The shared prefix is replayed once and every branch-private suffix is
    replayed once.  Prompt prefill is replay cost but is not generated rollout
    progress, so it is not counted as lost rollout work.
    """

    recompute_seconds = workload.history_tokens / measurements.recompute_tokens_per_second
    resume = (
        workload.environment_reconstruction_seconds
        + recompute_seconds
        + measurements.first_resumed_token_seconds
    )
    total = measurements.kill_release_seconds + resume
    return ReclamationPathEstimate(
        action=ReclamationAction.KILL_AND_RECOMPUTE,
        reclamation_seconds=measurements.kill_release_seconds,
        rollout_resume_seconds=resume,
        full_transition_seconds=total,
        meets_serving_urgency=(
            measurements.kill_release_seconds <= workload.serving_urgency_seconds
        ),
        recompute_tokens=workload.history_tokens,
        lost_rollout_work_tokens=workload.lost_rollout_work_tokens,
        recompute_gpu_seconds=recompute_seconds,
        transition_gpu_occupancy_seconds=total,
        transition_gpu_cost_usd=_seconds_to_cost(total, workload.gpu_hour_price_usd),
        expected_remaining_tokens=workload.remaining_tokens,
    )


def preservation_break_even(
    workload: ReclamationWorkload,
    measurements: ReclamationMeasurements,
    action: ReclamationAction,
) -> PreservationBreakEven:
    """Calculate parity against kill/recompute using measured linear rates.

    Logical bytes per replay token are held at the observed workload ratio.  A
    result is therefore an evidence-scoped interpolation, not an extrapolated
    hardware measurement.
    """

    checkpoint_rate, restore_rate, checkpoint_fixed, restore_fixed = _preservation_rates(
        measurements, action
    )
    bytes_per_token = workload.logical_state_bytes / workload.history_tokens
    preserve_fixed = checkpoint_fixed + restore_fixed + measurements.first_resumed_token_seconds
    kill_fixed = (
        measurements.kill_release_seconds
        + workload.environment_reconstruction_seconds
        + measurements.first_resumed_token_seconds
    )
    fixed_difference = preserve_fixed - kill_fixed
    preserve_seconds_per_token = bytes_per_token * (1.0 / checkpoint_rate + 1.0 / restore_rate)
    kill_seconds_per_token = 1.0 / measurements.recompute_tokens_per_second
    recompute_savings_per_token = kill_seconds_per_token - preserve_seconds_per_token

    minimum: int | None = None
    maximum: int | None = None
    rates_are_equal = math.isclose(
        recompute_savings_per_token,
        0.0,
        rel_tol=1e-12,
        abs_tol=1e-18,
    )
    if recompute_savings_per_token > 0.0 and not rates_are_equal:
        minimum = max(
            0,
            _integer_parity(
                fixed_difference / recompute_savings_per_token,
                upper=True,
            ),
        )
        direction = BreakEvenDirection.AT_OR_ABOVE
        reason = (
            "recompute cost grows faster than preservation traffic at the observed "
            "logical-bytes-per-history-token ratio"
        )
    elif recompute_savings_per_token < 0.0 and not rates_are_equal and fixed_difference <= 0.0:
        maximum = max(
            0,
            _integer_parity(
                fixed_difference / recompute_savings_per_token,
                upper=False,
            ),
        )
        direction = BreakEvenDirection.AT_OR_BELOW
        reason = (
            "preservation has a lower fixed cost but its state traffic grows faster than "
            "measured recomputation"
        )
    elif rates_are_equal and fixed_difference <= 0.0:
        direction = BreakEvenDirection.ALL_HISTORY_SIZES
        reason = "preservation is no more expensive at every history size in the linear model"
    else:
        direction = BreakEvenDirection.NO_HISTORY_SIZE
        reason = "preservation is more expensive at every history size in the linear model"

    parity_tokens = minimum if minimum is not None else maximum
    parity_bytes = None if parity_tokens is None else math.ceil(parity_tokens * bytes_per_token)
    parity_gpu_seconds = (
        None if parity_tokens is None else parity_tokens / measurements.recompute_tokens_per_second
    )
    if minimum is None:
        minimum_prefix = None
        minimum_rollout_age = None
    else:
        minimum_prefix = minimum
        minimum_rollout_age = max(
            0,
            math.ceil((minimum - workload.prefix_tokens) / workload.branch_count),
        )

    estimates = estimate_reclamation_paths(workload, measurements)
    estimates_by_action = {estimate.action: estimate for estimate in estimates}
    preservation_current = estimates_by_action[action]
    kill_current = estimates_by_action[ReclamationAction.KILL_AND_RECOMPUTE]
    return PreservationBreakEven(
        preservation_action=action,
        direction=direction,
        current_history_tokens=workload.history_tokens,
        current_logical_state_bytes=workload.logical_state_bytes,
        logical_bytes_per_history_token=bytes_per_token,
        minimum_history_tokens_at_parity=minimum,
        maximum_history_tokens_at_parity=maximum,
        minimum_prefix_tokens_if_no_private_state=minimum_prefix,
        minimum_rollout_age_tokens_per_branch_at_current_prefix=minimum_rollout_age,
        equivalent_logical_state_bytes_at_parity=parity_bytes,
        recompute_gpu_seconds_at_parity=parity_gpu_seconds,
        recompute_gpu_cost_usd_at_parity=_seconds_to_cost(
            parity_gpu_seconds, workload.gpu_hour_price_usd
        )
        if parity_gpu_seconds is not None
        else None,
        preservation_is_no_more_expensive_for_current_workload=(
            preservation_current.full_transition_seconds <= kill_current.full_transition_seconds
        ),
        minimum_remaining_tokens_per_branch_for_preservation=(
            1
            if preservation_current.full_transition_seconds <= kill_current.full_transition_seconds
            else None
        ),
        reason=reason,
    )


def _request_digest(
    workload: ReclamationWorkload,
    measurements: ReclamationMeasurements,
) -> str:
    payload = {
        "measurements": measurements.model_dump(mode="json"),
        "workload": workload.model_dump(mode="json"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def choose_reclamation_policy(
    workload: ReclamationWorkload,
    measurements: ReclamationMeasurements,
) -> ReclamationDecision:
    """Choose a measured path with a hard serving-urgency constraint.

    Among paths that meet the deadline, the policy minimizes the complete
    preserve/restore or kill/recompute transition.  If none meet the deadline,
    it selects the quickest reclamation path and reports the miss.  Exact cost
    ties prefer optimized preservation, then naive preservation, because those
    paths do not discard branch-private work.
    """

    estimates = estimate_reclamation_paths(workload, measurements)
    priority = {
        ReclamationAction.PRESERVE_OPTIMIZED: 0,
        ReclamationAction.PRESERVE_NAIVE: 1,
        ReclamationAction.KILL_AND_RECOMPUTE: 2,
    }
    if workload.remaining_tokens == 0:
        selected = ReclamationAction.KILL_AND_RECOMPUTE
        reason = "no rollout work remains, so preserving state has no future value"
    else:
        urgent_candidates = tuple(
            estimate for estimate in estimates if estimate.meets_serving_urgency
        )
        if urgent_candidates:
            winner = min(
                urgent_candidates,
                key=lambda estimate: (
                    estimate.full_transition_seconds,
                    estimate.reclamation_seconds,
                    priority[estimate.action],
                ),
            )
            selected = winner.action
            reason = (
                "selected the lowest measured full-transition cost among paths meeting "
                "the serving reclamation deadline"
            )
        else:
            winner = min(
                estimates,
                key=lambda estimate: (
                    estimate.reclamation_seconds,
                    estimate.full_transition_seconds,
                    priority[estimate.action],
                ),
            )
            selected = winner.action
            reason = (
                "no path meets the serving reclamation deadline; selected the fastest "
                "measured reclamation path"
            )

    break_even = tuple(
        preservation_break_even(workload, measurements, action)
        for action in (
            ReclamationAction.PRESERVE_NAIVE,
            ReclamationAction.PRESERVE_OPTIMIZED,
        )
    )
    return ReclamationDecision(
        selected_action=selected,
        selection_reason=reason,
        request_sha256=_request_digest(workload, measurements),
        evidence_class=measurements.evidence_class,
        evidence=measurements.evidence,
        estimates=estimates,
        break_even=break_even,
    )


__all__ = [
    "BreakEvenDirection",
    "PreservationBreakEven",
    "ReclamationAction",
    "ReclamationDecision",
    "ReclamationMeasurements",
    "ReclamationPathEstimate",
    "ReclamationWorkload",
    "choose_reclamation_policy",
    "estimate_kill_and_recompute",
    "estimate_reclamation_paths",
    "preservation_break_even",
]
