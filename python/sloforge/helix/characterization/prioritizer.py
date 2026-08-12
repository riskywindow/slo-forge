"""Transparent active selection for the next characterization experiment."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Identifier = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_.-]{0,127}$")]


class PrioritizerModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ExperimentCandidate(PrioritizerModel):
    experiment_id: Identifier
    question: Literal[
        "page_size",
        "memory_capacity",
        "multicast",
        "queue_depth",
        "transform_pipeline",
        "nic_bandwidth",
        "metadata_engine",
        "placement",
    ]
    uncertainty: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    sensitivity: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    amdahl_leverage: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    measurement_cost_usd: float = Field(ge=0.0, allow_inf_nan=False)
    wall_time_minutes: float = Field(gt=0.0, allow_inf_nan=False)
    accessible_hardware: bool
    evidence_gap: str = Field(min_length=1, max_length=2048)
    expected_artifacts: tuple[str, ...] = Field(min_length=1, max_length=64)


class RankedExperiment(PrioritizerModel):
    rank: int = Field(ge=1)
    experiment_id: Identifier
    question: str
    score: float = Field(ge=0.0, allow_inf_nan=False)
    uncertainty_reduction_value: float = Field(ge=0.0, allow_inf_nan=False)
    normalized_cost: float = Field(gt=0.0, allow_inf_nan=False)
    eligible: bool
    reason: str = Field(min_length=1, max_length=4096)


class ExperimentQueue(PrioritizerModel):
    schema_version: Literal["sloforge.branchfabric.experiment-queue/v1"]
    seed: int = Field(ge=0, le=2**64 - 1)
    gpu_budget_usd: float = Field(ge=0.0, allow_inf_nan=False)
    reserved_rerun_fraction: float = Field(ge=0.15, le=1.0, allow_inf_nan=False)
    spendable_budget_usd: float = Field(ge=0.0, allow_inf_nan=False)
    ranked: tuple[RankedExperiment, ...]

    @model_validator(mode="after")
    def ranks_are_dense(self) -> ExperimentQueue:
        if tuple(item.rank for item in self.ranked) != tuple(range(1, len(self.ranked) + 1)):
            raise ValueError("experiment ranks must be dense and ordered")
        return self


def prioritize_experiments(
    candidates: tuple[ExperimentCandidate, ...],
    *,
    seed: int,
    gpu_budget_usd: float,
    reserved_rerun_fraction: float = 0.15,
) -> ExperimentQueue:
    """Rank candidates using an inspectable uncertainty/value/cost equation.

    The seed is an explicit deterministic tie breaker only. No learned or opaque
    model is used. Inaccessible hardware and experiments above the spendable
    budget stay visible but are ineligible and sorted after eligible work.
    """

    if not 0 <= seed <= 2**64 - 1:
        raise ValueError("seed must fit uint64")
    if gpu_budget_usd < 0:
        raise ValueError("gpu_budget_usd cannot be negative")
    if not 0.15 <= reserved_rerun_fraction <= 1.0:
        raise ValueError("reserved_rerun_fraction must be within 0.15..1")
    identifiers = [candidate.experiment_id for candidate in candidates]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("candidate experiment identifiers must be unique")

    spendable = gpu_budget_usd * (1.0 - reserved_rerun_fraction)
    scored: list[tuple[bool, float, str, str, ExperimentCandidate, float, float]] = []
    for candidate in candidates:
        value = candidate.uncertainty * candidate.sensitivity * candidate.amdahl_leverage
        normalized_cost = (
            1.0
            + candidate.wall_time_minutes / 60.0
            + candidate.measurement_cost_usd / max(1.0, spendable)
        )
        eligible = candidate.accessible_hardware and candidate.measurement_cost_usd <= spendable
        score = value / normalized_cost if eligible else 0.0
        tie = f"{seed:020d}:{candidate.experiment_id}"
        if not candidate.accessible_hardware:
            reason = f"INELIGIBLE: required hardware is unavailable; {candidate.evidence_gap}"
        elif candidate.measurement_cost_usd > spendable:
            reason = (
                "INELIGIBLE: estimated cost exceeds spendable budget after the rerun "
                f"reserve; {candidate.evidence_gap}"
            )
        else:
            reason = (
                "ELIGIBLE: score = uncertainty x sensitivity x Amdahl leverage / "
                f"normalized cost; {candidate.evidence_gap}"
            )
        scored.append((eligible, score, tie, reason, candidate, value, normalized_cost))
    scored.sort(key=lambda item: (not item[0], -item[1], item[2]))
    ranked = tuple(
        RankedExperiment(
            rank=index,
            experiment_id=candidate.experiment_id,
            question=candidate.question,
            score=score,
            uncertainty_reduction_value=value,
            normalized_cost=cost,
            eligible=eligible,
            reason=reason,
        )
        for index, (eligible, score, _tie, reason, candidate, value, cost) in enumerate(
            scored, start=1
        )
    )
    return ExperimentQueue(
        schema_version="sloforge.branchfabric.experiment-queue/v1",
        seed=seed,
        gpu_budget_usd=gpu_budget_usd,
        reserved_rerun_fraction=reserved_rerun_fraction,
        spendable_budget_usd=spendable,
        ranked=ranked,
    )


__all__ = [
    "ExperimentCandidate",
    "ExperimentQueue",
    "RankedExperiment",
    "prioritize_experiments",
]
