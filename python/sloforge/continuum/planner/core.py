"""Predictive strategy selection using explicit measured migration inputs."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=512)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class PlannerModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class MigrationStrategy(StrEnum):
    STOP_AND_COPY = "stop_and_copy"
    PRE_COPY = "pre_copy"
    HYBRID_PRE_COPY = "hybrid_pre_copy"
    RECOMPUTATION_ASSISTED = "recomputation_assisted"
    CONSTRAINED_LAZY = "constrained_lazy"


class ExactnessRequirement(StrEnum):
    EXACT_BITWISE = "exact_bitwise"
    EXACT_SEMANTIC = "exact_semantic"
    NUMERICALLY_EQUIVALENT = "numerically_equivalent"
    QUALITY_BOUNDED = "quality_bounded"
    RECOMPUTATION_ASSISTED = "recomputation_assisted"


class MeasuredRate(PlannerModel):
    name: NonEmpty
    bytes_per_second: Annotated[float, Field(gt=0, le=10**15)]
    sample_count: Annotated[int, Field(ge=1, le=1_000_000)]
    coefficient_of_variation: Annotated[float, Field(ge=0, le=10)]
    artifact_uri: NonEmpty
    artifact_sha256: Sha256
    synthetic: bool


class AccessPattern(PlannerModel):
    segment_id: NonEmpty
    size_bytes: Annotated[int, Field(ge=0)]
    state_type: Literal[
        "attention_kv",
        "recurrent",
        "sampler",
        "guided_decoding",
        "workflow",
        "token_history",
        "other",
    ]
    required_before_resume: bool
    streamable_before_use: bool
    recomputable: bool
    dense_full_attention: bool = False


class ObjectiveWeights(PlannerModel):
    total_time: Annotated[float, Field(ge=0)] = 0.05
    source_overhead: Annotated[float, Field(ge=0)] = 1.0
    cost: Annotated[float, Field(ge=0)] = 1.0
    failure: Annotated[float, Field(ge=0)] = 1000.0
    memory: Annotated[float, Field(ge=0)] = 0.000001
    quality: Annotated[float, Field(ge=0)] = 1000.0


class MigrationPlanningInput(PlannerModel):
    seed: Annotated[int, Field(ge=0, le=2**64 - 1)]
    source_runtime: NonEmpty
    destination_runtime: NonEmpty
    state_size_bytes: Annotated[int, Field(ge=0, le=2**50)]
    dirty_rate_bytes_per_second: Annotated[float, Field(ge=0, le=10**15)]
    generation_tokens_per_second: Annotated[float, Field(ge=0, le=10**9)]
    source_load_fraction: Annotated[float, Field(ge=0, le=1)]
    destination_ready_ms: Annotated[float, Field(ge=0, le=86_400_000)]
    transfer_rates: tuple[MeasuredRate, ...]
    conversion_rates: tuple[MeasuredRate, ...]
    memory_limit_bytes: Annotated[int, Field(ge=1)]
    maximum_interruption_ms: Annotated[float, Field(gt=0)]
    exactness: ExactnessRequirement
    compatibility_allows_recomputation: bool
    rollback_required: bool
    failure_probability: Annotated[float, Field(ge=0, le=1)]
    migration_budget_usd: Annotated[float, Field(ge=0)]
    quality_loss: Annotated[float, Field(ge=0, le=1)] = 0.0
    quality_budget: Annotated[float, Field(ge=0, le=1)] = 0.0
    access_patterns: tuple[AccessPattern, ...]
    maximum_precopy_rounds: Annotated[int, Field(ge=1, le=32)] = 6
    convergence_threshold_bytes: Annotated[int, Field(ge=1)] = 1024 * 1024
    validation_ms: Annotated[float, Field(ge=0)] = 1.0
    cost_per_gib_transferred_usd: Annotated[float, Field(ge=0)] = 0.0
    objective_weights: ObjectiveWeights = ObjectiveWeights()

    @model_validator(mode="after")
    def validate_inputs(self) -> Self:
        if not self.transfer_rates or not self.conversion_rates:
            raise ValueError("planner requires measured transfer and conversion rates")
        if sum(item.size_bytes for item in self.access_patterns) != self.state_size_bytes:
            raise ValueError("access-pattern sizes must cover the complete state")
        if self.quality_loss > self.quality_budget:
            raise ValueError("declared quality loss exceeds the migration quality budget")
        if self.exactness is not ExactnessRequirement.QUALITY_BOUNDED and self.quality_loss > 0:
            raise ValueError("quality loss requires quality_bounded exactness")
        return self


class CandidateEstimate(PlannerModel):
    strategy: MigrationStrategy
    legal: bool
    rejection_reason: str | None = None
    transfer_name: str | None = None
    conversion_name: str | None = None
    chunk_size_bytes: Annotated[int, Field(ge=0)]
    pre_copy_rounds: Annotated[int, Field(ge=0)]
    interruption_ms: Annotated[float, Field(ge=0)]
    total_time_ms: Annotated[float, Field(ge=0)]
    source_overhead_ms: Annotated[float, Field(ge=0)]
    transferred_bytes: Annotated[int, Field(ge=0)]
    temporary_memory_bytes: Annotated[int, Field(ge=0)]
    migration_cost_usd: Annotated[float, Field(ge=0)]
    failure_exposure: Annotated[float, Field(ge=0)]
    quality_loss: Annotated[float, Field(ge=0, le=1)]
    objective: Annotated[float, Field(ge=0)]
    predicted_converged: bool


class PlannedMigration(PlannerModel):
    strategy: MigrationStrategy
    selected_transfer: NonEmpty
    selected_conversion: NonEmpty
    chunk_size_bytes: Annotated[int, Field(ge=1)]
    concurrency: Annotated[int, Field(ge=1, le=64)]
    pre_copy_rounds: Annotated[int, Field(ge=0)]
    cutover_threshold_bytes: Annotated[int, Field(ge=0)]
    destination_warmup_required: bool
    rollback_checkpoint_required: bool
    expected_interruption_ms: Annotated[float, Field(ge=0)]
    expected_total_time_ms: Annotated[float, Field(ge=0)]
    expected_transferred_bytes: Annotated[int, Field(ge=0)]
    uncertainty_fraction: Annotated[float, Field(ge=0, le=10)]
    candidates: tuple[CandidateEstimate, ...]
    evidence_hashes: tuple[Sha256, ...]


def _seconds_for_bytes(size_bytes: float, rate: float) -> float:
    return size_bytes / rate


def _objective(
    request: MigrationPlanningInput,
    *,
    interruption_ms: float,
    total_ms: float,
    source_overhead_ms: float,
    transferred_bytes: int,
    temporary_memory_bytes: int,
    quality_loss: float,
) -> tuple[float, float, float]:
    cost = transferred_bytes / (1024**3) * request.cost_per_gib_transferred_usd
    failure = request.failure_probability * total_ms
    weights = request.objective_weights
    objective = (
        interruption_ms
        + weights.total_time * total_ms
        + weights.source_overhead * source_overhead_ms
        + weights.cost * cost
        + weights.failure * failure
        + weights.memory * temporary_memory_bytes
        + weights.quality * quality_loss
    )
    return objective, cost, failure


def _base_rates(request: MigrationPlanningInput) -> tuple[MeasuredRate, MeasuredRate, float]:
    transfer = max(request.transfer_rates, key=lambda item: item.bytes_per_second)
    conversion = max(request.conversion_rates, key=lambda item: item.bytes_per_second)
    uncertainty = min(
        10.0,
        math.sqrt(transfer.coefficient_of_variation**2 + conversion.coefficient_of_variation**2),
    )
    return transfer, conversion, uncertainty


def _candidate(
    request: MigrationPlanningInput,
    *,
    strategy: MigrationStrategy,
    legal: bool,
    reason: str | None,
    transfer: MeasuredRate,
    conversion: MeasuredRate,
    rounds: int,
    interruption_ms: float,
    total_ms: float,
    source_overhead_ms: float,
    transferred_bytes: int,
    temporary_memory_bytes: int,
    converged: bool,
) -> CandidateEstimate:
    objective, cost, failure = _objective(
        request,
        interruption_ms=interruption_ms,
        total_ms=total_ms,
        source_overhead_ms=source_overhead_ms,
        transferred_bytes=transferred_bytes,
        temporary_memory_bytes=temporary_memory_bytes,
        quality_loss=request.quality_loss,
    )
    if interruption_ms > request.maximum_interruption_ms:
        legal = False
        reason = reason or "predicted interruption exceeds the migration SLO"
    if temporary_memory_bytes > request.memory_limit_bytes:
        legal = False
        reason = reason or "temporary memory exceeds the destination bound"
    if cost > request.migration_budget_usd:
        legal = False
        reason = reason or "predicted transfer cost exceeds the migration budget"
    return CandidateEstimate(
        strategy=strategy,
        legal=legal,
        rejection_reason=reason,
        transfer_name=transfer.name,
        conversion_name=conversion.name,
        chunk_size_bytes=min(max(64 * 1024, request.state_size_bytes // 32), 8 * 1024 * 1024),
        pre_copy_rounds=rounds,
        interruption_ms=interruption_ms,
        total_time_ms=total_ms,
        source_overhead_ms=source_overhead_ms,
        transferred_bytes=transferred_bytes,
        temporary_memory_bytes=temporary_memory_bytes,
        migration_cost_usd=cost,
        failure_exposure=failure,
        quality_loss=request.quality_loss,
        objective=max(0.0, objective),
        predicted_converged=converged,
    )


def plan_migration(request: MigrationPlanningInput) -> PlannedMigration:
    """Select among baselines and advanced strategies using measured rates."""

    transfer, conversion, uncertainty = _base_rates(request)
    effective_rate = min(transfer.bytes_per_second, conversion.bytes_per_second)
    full_seconds = _seconds_for_bytes(request.state_size_bytes, effective_rate)
    stop_interruption = request.destination_ready_ms + full_seconds * 1000 + request.validation_ms
    temporary = min(request.state_size_bytes, request.memory_limit_bytes)
    stop = _candidate(
        request,
        strategy=MigrationStrategy.STOP_AND_COPY,
        legal=True,
        reason=None,
        transfer=transfer,
        conversion=conversion,
        rounds=0,
        interruption_ms=stop_interruption,
        total_ms=stop_interruption,
        source_overhead_ms=0.0,
        transferred_bytes=request.state_size_bytes,
        temporary_memory_bytes=temporary,
        converged=True,
    )

    remaining = float(request.state_size_bytes)
    transferred = 0.0
    rounds = 0
    pre_copy_time = 0.0
    converged = request.dirty_rate_bytes_per_second < effective_rate
    if converged:
        for round_index in range(request.maximum_precopy_rounds):
            round_seconds = _seconds_for_bytes(remaining, effective_rate)
            transferred += remaining
            pre_copy_time += round_seconds
            remaining = request.dirty_rate_bytes_per_second * round_seconds
            rounds = round_index + 1
            if remaining <= request.convergence_threshold_bytes:
                break
    else:
        rounds = 1
        transferred = float(request.state_size_bytes)
        pre_copy_time = full_seconds
        remaining = request.dirty_rate_bytes_per_second * full_seconds
    final_seconds = _seconds_for_bytes(remaining, effective_rate)
    precopy_interruption = final_seconds * 1000 + request.validation_ms
    precopy_total = request.destination_ready_ms + (pre_copy_time + final_seconds) * 1000
    source_overhead = pre_copy_time * 1000 * (0.01 + 0.05 * request.source_load_fraction)
    pre_copy = _candidate(
        request,
        strategy=MigrationStrategy.PRE_COPY,
        legal=converged,
        reason=None if converged else "dirty rate is not below measured conversion/transfer rate",
        transfer=transfer,
        conversion=conversion,
        rounds=rounds,
        interruption_ms=precopy_interruption,
        total_ms=precopy_total,
        source_overhead_ms=source_overhead,
        transferred_bytes=math.ceil(transferred + remaining),
        temporary_memory_bytes=temporary,
        converged=converged and remaining <= request.convergence_threshold_bytes,
    )

    hybrid_rounds = min(2, request.maximum_precopy_rounds)
    hybrid_remaining = float(request.state_size_bytes)
    hybrid_transferred = 0.0
    hybrid_time = 0.0
    for _ in range(hybrid_rounds):
        duration = _seconds_for_bytes(hybrid_remaining, effective_rate)
        hybrid_transferred += hybrid_remaining
        hybrid_time += duration
        hybrid_remaining = request.dirty_rate_bytes_per_second * duration
    hybrid_final = _seconds_for_bytes(hybrid_remaining, effective_rate)
    hybrid = _candidate(
        request,
        strategy=MigrationStrategy.HYBRID_PRE_COPY,
        legal=True,
        reason=None,
        transfer=transfer,
        conversion=conversion,
        rounds=hybrid_rounds,
        interruption_ms=hybrid_final * 1000 + request.validation_ms,
        total_ms=request.destination_ready_ms + (hybrid_time + hybrid_final) * 1000,
        source_overhead_ms=hybrid_time * 1000 * (0.01 + 0.05 * request.source_load_fraction),
        transferred_bytes=math.ceil(hybrid_transferred + hybrid_remaining),
        temporary_memory_bytes=temporary,
        converged=True,
    )

    reusable = sum(
        pattern.size_bytes for pattern in request.access_patterns if not pattern.recomputable
    )
    history = sum(
        pattern.size_bytes
        for pattern in request.access_patterns
        if pattern.state_type == "token_history"
    )
    recompute_bytes = reusable + history
    recompute_seconds = _seconds_for_bytes(recompute_bytes, effective_rate)
    recompute_legal = request.compatibility_allows_recomputation and history > 0
    recompute = _candidate(
        request,
        strategy=MigrationStrategy.RECOMPUTATION_ASSISTED,
        legal=recompute_legal,
        reason=(
            None
            if recompute_legal
            else "compatibility evidence or token history does not authorize recomputation"
        ),
        transfer=transfer,
        conversion=conversion,
        rounds=0,
        interruption_ms=(recompute_seconds * 1000 + request.validation_ms),
        total_ms=request.destination_ready_ms + recompute_seconds * 1000,
        source_overhead_ms=0.0,
        transferred_bytes=recompute_bytes,
        temporary_memory_bytes=min(recompute_bytes, request.memory_limit_bytes),
        converged=True,
    )

    lazy_illegal = [
        pattern.segment_id
        for pattern in request.access_patterns
        if pattern.required_before_resume
        and not pattern.streamable_before_use
        and not pattern.recomputable
    ]
    if any(pattern.dense_full_attention for pattern in request.access_patterns):
        lazy_illegal.append("dense_full_attention")
    immediate = sum(
        pattern.size_bytes for pattern in request.access_patterns if pattern.required_before_resume
    )
    lazy_seconds = _seconds_for_bytes(immediate, effective_rate)
    lazy = _candidate(
        request,
        strategy=MigrationStrategy.CONSTRAINED_LAZY,
        legal=not lazy_illegal,
        reason=(
            None
            if not lazy_illegal
            else "required-before-use legality failed for: " + ", ".join(sorted(lazy_illegal))
        ),
        transfer=transfer,
        conversion=conversion,
        rounds=0,
        interruption_ms=lazy_seconds * 1000 + request.validation_ms,
        total_ms=request.destination_ready_ms + full_seconds * 1000,
        source_overhead_ms=0.0,
        transferred_bytes=request.state_size_bytes,
        temporary_memory_bytes=min(immediate, request.memory_limit_bytes),
        converged=True,
    )
    candidates = (stop, pre_copy, hybrid, recompute, lazy)
    legal = [candidate for candidate in candidates if candidate.legal]
    if not legal:
        explanations = "; ".join(
            f"{candidate.strategy}: {candidate.rejection_reason}" for candidate in candidates
        )
        raise ValueError(f"no migration strategy satisfies the declared contract: {explanations}")
    tie_priority = {
        MigrationStrategy.PRE_COPY: 0,
        MigrationStrategy.HYBRID_PRE_COPY: 1,
        MigrationStrategy.STOP_AND_COPY: 2,
        MigrationStrategy.RECOMPUTATION_ASSISTED: 3,
        MigrationStrategy.CONSTRAINED_LAZY: 4,
    }
    selected = min(
        legal, key=lambda candidate: (candidate.objective, tie_priority[candidate.strategy])
    )
    return PlannedMigration(
        strategy=selected.strategy,
        selected_transfer=transfer.name,
        selected_conversion=conversion.name,
        chunk_size_bytes=max(1, selected.chunk_size_bytes),
        concurrency=min(8, max(1, math.ceil(transfer.bytes_per_second / (1024**3)))),
        pre_copy_rounds=selected.pre_copy_rounds,
        cutover_threshold_bytes=(
            request.convergence_threshold_bytes
            if selected.strategy in {MigrationStrategy.PRE_COPY, MigrationStrategy.HYBRID_PRE_COPY}
            else 0
        ),
        destination_warmup_required=request.destination_ready_ms > 0,
        rollback_checkpoint_required=request.rollback_required,
        expected_interruption_ms=selected.interruption_ms,
        expected_total_time_ms=selected.total_time_ms,
        expected_transferred_bytes=selected.transferred_bytes,
        uncertainty_fraction=uncertainty,
        candidates=candidates,
        evidence_hashes=tuple(
            sorted(
                {
                    *(rate.artifact_sha256 for rate in request.transfer_rates),
                    *(rate.artifact_sha256 for rate in request.conversion_rates),
                }
            )
        ),
    )
