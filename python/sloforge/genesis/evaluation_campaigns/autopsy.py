"""Deterministic, artifact-backed H4 Autopsy-guided search campaign.

The campaign evaluates search *efficiency* using the existing Genesis proposal,
budget, lifecycle, Pareto, and Autopsy mutation-guard implementations.  Its
evaluator is a declared deterministic synthetic workload model: no row in this
campaign is hardware evidence and no wall-clock performance claim is made.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from sloforge.autopsy.models import BottleneckKind, DiagnosisRecord
from sloforge.genesis.autopsy_guidance import (
    ALL_REGIONS,
    MutationBudget,
    MutationGuard,
    build_mutation_budget,
)
from sloforge.genesis.ir import (
    ArtifactDigest,
    BudgetUsage,
    CandidateFailureState,
    CandidateState,
    CandidateSuccessState,
    SearchBudget,
    TransformationFamily,
    canonical_json,
)
from sloforge.genesis.search import (
    CandidateDesign,
    FidelityStage,
    MutationChoice,
    ObjectiveVector,
    ParameterValue,
    ProposalPortfolio,
    ProposalRequest,
    SearchConfiguration,
    SearchEngine,
    StageResult,
    StageSpecification,
)
from sloforge.genesis.search.models import Region

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]

_MAXIMUM_SEEDS = 64
_MAXIMUM_CANDIDATES = 64
_MAXIMUM_DIAGNOSIS_BYTES = 8 * 1024 * 1024
_BASE_GENOME_HASH = ArtifactDigest(value=hashlib.sha256(b"genesis-h4-base-genome").hexdigest())


class CampaignValidationError(ValueError):
    """Campaign artifacts are missing, changed, or not derivable from raw rows."""


class SearchStrategy(StrEnum):
    AUTOPSY_GUIDED = "autopsy_guided"
    RANDOM_REGION = "random_region"
    UNRESTRICTED = "unrestricted"


class CampaignModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class CampaignScope(CampaignModel):
    evaluator: Literal["deterministic_synthetic_h4_v1"] = "deterministic_synthetic_h4_v1"
    evidence_scope: Literal["synthetic_cpu_only"] = "synthetic_cpu_only"
    hardware_backed: Literal[False] = False
    allows_hardware_performance_claims: Literal[False] = False
    time_metric: Literal["modeled_evaluator_time"] = "modeled_evaluator_time"
    time_unit: Literal["synthetic_seconds"] = "synthetic_seconds"
    objective_metric: Literal["network_bottleneck_relief"] = "network_bottleneck_relief"
    objective_unit: Literal["normalized_utility_points"] = "normalized_utility_points"


class RawCandidateRecord(CampaignModel):
    schema_version: Literal["sloforge.genesis.h4-candidate/v1"] = "sloforge.genesis.h4-candidate/v1"
    campaign_seed: NonNegativeInt
    run_seed: NonNegativeInt
    strategy: SearchStrategy
    diagnosis_id: NonEmpty
    bottleneck: BottleneckKind
    mutation_surface: tuple[Region, ...]
    candidate_index: NonNegativeInt
    design: CandidateDesign
    final_state: CandidateState
    stage_results: tuple[StageResult, ...]
    budget_exhausted: bool

    @model_validator(mode="after")
    def consistent_candidate(self) -> Self:
        if not self.mutation_surface:
            raise ValueError("candidate mutation surface cannot be empty")
        if len(self.mutation_surface) != len(set(self.mutation_surface)):
            raise ValueError("candidate mutation surface must be unique")
        if not set(self.design.affected_regions).issubset(self.mutation_surface):
            raise ValueError("candidate mutates outside its recorded surface")
        failed = tuple(result for result in self.stage_results if not result.passed)
        if len(failed) > 1:
            raise ValueError("candidate records more than one terminal stage failure")
        if failed:
            if failed[0] is not self.stage_results[-1]:
                raise ValueError("candidate stages continue after terminal failure")
            if failed[0].failure_state is not self.final_state:
                raise ValueError("candidate final state differs from stage failure")
        elif self.final_state is not CandidateSuccessState.SIMULATED:
            raise ValueError("successful synthetic campaign candidate must reach SIMULATED")
        return self


class SeedStrategySummary(CampaignModel):
    run_seed: NonNegativeInt
    strategy: SearchStrategy
    mutation_surface: tuple[Region, ...]
    candidates_evaluated: NonNegativeInt
    invalid_candidates: NonNegativeInt
    actual_hardware_experiments: Literal[0] = 0
    synthetic_high_fidelity_experiments: NonNegativeInt
    synthetic_high_fidelity_experiments_to_improvement: NonNegativeInt | None
    candidates_to_improvement: PositiveInt | None
    time_to_improvement: NonNegativeFloat | None
    final_objective: float | None
    distinct_transformation_families: NonNegativeInt


class StrategyAggregate(CampaignModel):
    strategy: SearchStrategy
    run_count: PositiveInt
    candidates_evaluated: NonNegativeInt
    mean_candidates_evaluated: NonNegativeFloat
    invalid_candidates: NonNegativeInt
    mean_invalid_candidates: NonNegativeFloat
    actual_hardware_experiments: Literal[0] = 0
    synthetic_high_fidelity_experiments: NonNegativeInt
    mean_synthetic_high_fidelity_experiments_to_improvement: NonNegativeFloat | None
    improvement_run_count: NonNegativeInt
    improvement_rate: Annotated[float, Field(ge=0.0, le=1.0)]
    mean_candidates_to_improvement: NonNegativeFloat | None
    mean_time_to_improvement: NonNegativeFloat | None
    mean_final_objective: float | None
    distinct_transformation_families: NonNegativeInt
    per_seed: tuple[SeedStrategySummary, ...]


class StrategyDelta(CampaignModel):
    baseline: SearchStrategy
    candidates_evaluated_reduction: int
    invalid_candidate_reduction: int
    mean_synthetic_high_fidelity_experiments_to_improvement_reduction: float | None
    mean_time_to_improvement_reduction: float | None
    mean_final_objective_delta: float | None


class AutopsyCampaignReport(CampaignModel):
    schema_version: Literal["sloforge.genesis.h4-campaign/v1"] = "sloforge.genesis.h4-campaign/v1"
    hypothesis_id: Literal["H4"] = "H4"
    statement: Literal["Autopsy-guided mutation improves search efficiency."] = (
        "Autopsy-guided mutation improves search efficiency."
    )
    campaign_seed: NonNegativeInt
    run_seeds: tuple[NonNegativeInt, ...]
    maximum_candidates_per_run: PositiveInt
    improvement_threshold: Annotated[float, Field(gt=0.0)]
    scope: CampaignScope
    diagnosis_path: NonEmpty
    diagnosis_sha256: Sha256
    mutation_budget_path: NonEmpty
    mutation_budget_sha256: Sha256
    raw_candidates_path: NonEmpty
    raw_candidates_sha256: Sha256
    aggregates: tuple[StrategyAggregate, ...]
    guided_deltas: tuple[StrategyDelta, ...]
    limitations: tuple[NonEmpty, ...]
    report_path: NonEmpty

    @model_validator(mode="after")
    def complete_campaign(self) -> Self:
        if not self.run_seeds or len(self.run_seeds) != len(set(self.run_seeds)):
            raise ValueError("campaign run seeds must be non-empty and unique")
        if tuple(item.strategy for item in self.aggregates) != tuple(SearchStrategy):
            raise ValueError("campaign must report all strategies in canonical order")
        if tuple(item.baseline for item in self.guided_deltas) != (
            SearchStrategy.RANDOM_REGION,
            SearchStrategy.UNRESTRICTED,
        ):
            raise ValueError("campaign must compare guided search with both baselines")
        return self


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_document(path: Path, document: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical_json(document) + b"\n")


def _write_records(path: Path, records: tuple[RawCandidateRecord, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        for record in records:
            handle.write(canonical_json(record) + b"\n")


def _derived_seed(campaign_seed: int, index: int) -> int:
    digest = hashlib.sha256(f"genesis-h4\0{campaign_seed}\0{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _mutation(
    identifier: str,
    family: TransformationFamily,
    regions: tuple[Region, ...],
    *,
    expected_upside: float,
    invalidity_risk: float,
    feature_slot: int,
) -> MutationChoice:
    features = tuple(1.0 if index == feature_slot else 0.0 for index in range(8))
    return MutationChoice(
        transformation_id=identifier,
        family=family,
        regions=regions,
        parameters=(ParameterValue(key="variant", value=identifier),),
        expected_upside=expected_upside,
        invalidity_risk=invalidity_risk,
        feature_delta=features,
    )


def _mutation_options() -> tuple[MutationChoice, ...]:
    """Fixed transformation grammar; outcomes remain hidden from proposal ranking."""

    return (
        _mutation(
            "eager-collective-overlap",
            TransformationFamily.COMMUNICATION,
            ("distributed", "kernel"),
            expected_upside=0.36,
            invalidity_risk=0.02,
            feature_slot=0,
        ),
        _mutation(
            "rail-aware-chunking",
            TransformationFamily.COMMUNICATION,
            ("distributed",),
            expected_upside=0.28,
            invalidity_risk=0.03,
            feature_slot=0,
        ),
        _mutation(
            "collective-compute-overlap",
            TransformationFamily.COMMUNICATION,
            ("distributed", "kernel"),
            expected_upside=0.24,
            invalidity_risk=0.04,
            feature_slot=1,
        ),
        _mutation(
            "topology-rank-remap",
            TransformationFamily.DISTRIBUTED_PLAN,
            ("distributed",),
            expected_upside=0.21,
            invalidity_risk=0.03,
            feature_slot=2,
        ),
        _mutation(
            "peer-state-staging",
            TransformationFamily.STATE_LAYOUT,
            ("state", "distributed"),
            expected_upside=0.13,
            invalidity_risk=0.03,
            feature_slot=3,
        ),
        _mutation(
            "dispatch-metadata-packing",
            TransformationFamily.KERNEL,
            ("kernel",),
            expected_upside=0.14,
            invalidity_risk=0.04,
            feature_slot=4,
        ),
        _mutation(
            "rail-failure-fallback",
            TransformationFamily.RECOVERY,
            ("recovery", "distributed"),
            expected_upside=0.10,
            invalidity_risk=0.04,
            feature_slot=5,
        ),
        _mutation(
            "unbounded-deadline-batch",
            TransformationFamily.BATCHING,
            ("request", "serving"),
            expected_upside=0.35,
            invalidity_risk=0.01,
            feature_slot=6,
        ),
        _mutation(
            "workflow-lookahead",
            TransformationFamily.WORKFLOW,
            ("workflow", "request"),
            expected_upside=0.31,
            invalidity_risk=0.02,
            feature_slot=6,
        ),
        _mutation(
            "speculative-prefill-window",
            TransformationFamily.SCHEDULER,
            ("serving", "tensor"),
            expected_upside=0.29,
            invalidity_risk=0.02,
            feature_slot=7,
        ),
        _mutation(
            "aggressive-cache-eviction",
            TransformationFamily.CACHE_POLICY,
            ("state",),
            expected_upside=0.27,
            invalidity_risk=0.02,
            feature_slot=7,
        ),
        _mutation(
            "approximate-state-quantization",
            TransformationFamily.QUANTIZATION,
            ("state", "tensor"),
            expected_upside=0.25,
            invalidity_risk=0.03,
            feature_slot=7,
        ),
        _mutation(
            "state-release-on-drain",
            TransformationFamily.STATE_LAYOUT,
            ("state",),
            expected_upside=0.23,
            invalidity_risk=0.02,
            feature_slot=3,
        ),
        _mutation(
            "gateway-priority-aging",
            TransformationFamily.SCHEDULER,
            ("request",),
            expected_upside=0.17,
            invalidity_risk=0.03,
            feature_slot=6,
        ),
        _mutation(
            "tensor-layout-specialization",
            TransformationFamily.LAYOUT,
            ("tensor",),
            expected_upside=0.16,
            invalidity_risk=0.04,
            feature_slot=7,
        ),
        _mutation(
            "request-boundary-recovery",
            TransformationFamily.RECOVERY,
            ("recovery",),
            expected_upside=0.08,
            invalidity_risk=0.03,
            feature_slot=5,
        ),
    )


_ACTUAL_GAIN: dict[str, float] = {
    "eager-collective-overlap": 0.08,
    "rail-aware-chunking": 0.22,
    "collective-compute-overlap": 0.17,
    "topology-rank-remap": 0.14,
    "peer-state-staging": 0.07,
    "dispatch-metadata-packing": 0.05,
    "rail-failure-fallback": 0.03,
    "request-boundary-recovery": 0.01,
    "state-release-on-drain": 0.02,
    "workflow-lookahead": -0.01,
    "speculative-prefill-window": -0.02,
    "aggressive-cache-eviction": -0.04,
    "approximate-state-quantization": -0.03,
    "gateway-priority-aging": 0.0,
    "tensor-layout-specialization": 0.01,
}


def _contains(candidate: CandidateDesign, identifier: str) -> bool:
    return any(item.transformation_id == identifier for item in candidate.mutations)


def _objective_gain(candidate: CandidateDesign, run_seed: int) -> float:
    identifiers = {item.transformation_id for item in candidate.mutations}
    gain = sum(_ACTUAL_GAIN.get(identifier, 0.0) for identifier in identifiers)
    if {"rail-aware-chunking", "collective-compute-overlap"}.issubset(identifiers):
        gain += 0.05
    if {"topology-rank-remap", "peer-state-staging"}.issubset(identifiers):
        gain += 0.025
    signature = "\0".join(sorted(identifiers))
    digest = hashlib.sha256(f"{run_seed}\0{signature}".encode()).digest()
    jitter = (int.from_bytes(digest[:2], "big") / 65535.0 - 0.5) * 0.012
    return round(max(-0.5, min(0.8, gain + jitter)), 9)


_STAGE_COST_SECONDS: dict[FidelityStage, float] = {
    FidelityStage.STATIC_PRUNING: 0.04,
    FidelityStage.ANALYTICAL_BOUND: 0.03,
    FidelityStage.COMPILE: 0.18,
    FidelityStage.DIGITAL_TWIN: 0.11,
    FidelityStage.DETERMINISTIC_TESTS: 0.15,
    FidelityStage.PROPERTY_VERIFICATION: 0.17,
    FidelityStage.MODEL_CHECK: 0.20,
    FidelityStage.SIMULATION: 0.31,
    FidelityStage.END_TO_END_BENCHMARK: 0.53,
}


class _SyntheticH4Evaluator:
    def __init__(self, run_seed: int) -> None:
        self.run_seed = run_seed

    def evaluate(
        self,
        candidate: CandidateDesign,
        stage: FidelityStage,
        *,
        seed: int,
    ) -> StageResult:
        if stage not in _STAGE_COST_SECONDS:
            raise ValueError(f"unsupported H4 campaign stage: {stage}")
        multiplier = 1.0 + 0.08 * (len(candidate.mutations) - 1)
        synthetic_seconds = round(_STAGE_COST_SECONDS[stage] * multiplier, 9)
        usage = BudgetUsage(
            wall_time_seconds=synthetic_seconds,
            cpu_time_seconds=synthetic_seconds,
            compilation_count=1 if stage is FidelityStage.COMPILE else 0,
            benchmark_count=1 if stage is FidelityStage.END_TO_END_BENCHMARK else 0,
            verifier_time_seconds=(
                synthetic_seconds
                if stage in {FidelityStage.PROPERTY_VERIFICATION, FidelityStage.MODEL_CHECK}
                else 0.0
            ),
        )
        evidence_id = hashlib.sha256(
            f"h4\0{self.run_seed}\0{seed}\0{candidate.candidate_id}\0{stage.value}".encode()
        ).hexdigest()
        failure: CandidateFailureState | None = None
        reason = f"deterministic synthetic evaluator passed {stage.value}"
        if stage is FidelityStage.PROPERTY_VERIFICATION and _contains(
            candidate, "unbounded-deadline-batch"
        ):
            failure = CandidateFailureState.SEMANTIC_REJECTED
            reason = "bounded policy property rejected unbounded deadline batching"
        if failure is not None:
            return StageResult(
                stage=stage,
                passed=False,
                reason=reason,
                usage=usage,
                failure_state=failure,
                evidence_ids=(evidence_id,),
            )
        objective: ObjectiveVector | None = None
        utility: float | None = None
        if stage is FidelityStage.SIMULATION:
            utility = _objective_gain(candidate, self.run_seed)
            objective = ObjectiveVector(
                correctness_confidence=1.0,
                quality=1.0,
                ttft_ms=max(1.0, 40.0 * (1.0 - utility * 0.20)),
                token_latency_ms=max(0.5, 8.0 * (1.0 - utility * 0.15)),
                goodput=max(0.0, 100.0 * (1.0 + utility)),
                throughput=max(0.0, 120.0 * (1.0 + utility * 0.8)),
                cost_usd_per_hour=0.0,
                energy_joules_per_token=max(0.0, 0.2 * (1.0 - utility * 0.1)),
                startup_ms=20.0,
                memory_bytes=1_073_741_824.0,
                reliability=0.999,
                implementation_complexity=float(1 + len(candidate.mutations)),
                transition_cost=float(len(candidate.mutations)),
            )
        return StageResult(
            stage=stage,
            passed=True,
            reason=reason,
            usage=usage,
            objective=objective,
            selection_utility=utility,
            hardware_backed=False,
            evidence_ids=(evidence_id,),
        )


def _strategy_surface(
    strategy: SearchStrategy, run_seed: int, budget: MutationBudget
) -> tuple[Region, ...]:
    if strategy is SearchStrategy.AUTOPSY_GUIDED:
        return budget.mutable_regions
    if strategy is SearchStrategy.UNRESTRICTED:
        return ALL_REGIONS
    generator = random.Random(run_seed)
    selected = set(generator.sample(list(ALL_REGIONS), len(budget.mutable_regions)))
    return tuple(region for region in ALL_REGIONS if region in selected)


def _strategy_options(
    strategy: SearchStrategy, surface: tuple[Region, ...], budget: MutationBudget
) -> tuple[MutationChoice, ...]:
    options = tuple(
        option for option in _mutation_options() if set(option.regions).issubset(surface)
    )
    if strategy is SearchStrategy.AUTOPSY_GUIDED:
        options = tuple(option for option in options if option.family in budget.allowed_families)
    if not options:
        raise ValueError(f"{strategy.value} mutation surface produced no legal options")
    return options


def _designs(
    strategy: SearchStrategy,
    run_seed: int,
    budget: MutationBudget,
    maximum_candidates: int,
) -> tuple[CandidateDesign, ...]:
    surface = _strategy_surface(strategy, run_seed, budget)
    request = ProposalRequest(
        base_genome_hash=_BASE_GENOME_HASH,
        seed=run_seed,
        base_features=(0.0,) * 8,
        options=_strategy_options(strategy, surface, budget),
        mutable_regions=surface,
        maximum_proposals=maximum_candidates,
    )
    designs = ProposalPortfolio().propose(request)
    if strategy is SearchStrategy.AUTOPSY_GUIDED:
        guard = MutationGuard(budget)
        for design in designs:
            guard.validate(design)
    return designs


def _stage_plan() -> tuple[StageSpecification, ...]:
    return tuple(
        StageSpecification(
            stage=stage,
            maximum_usage=BudgetUsage(
                wall_time_seconds=1.0,
                cpu_time_seconds=1.0,
                compilation_count=1 if stage is FidelityStage.COMPILE else 0,
                benchmark_count=1 if stage is FidelityStage.END_TO_END_BENCHMARK else 0,
                verifier_time_seconds=(
                    1.0
                    if stage in {FidelityStage.PROPERTY_VERIFICATION, FidelityStage.MODEL_CHECK}
                    else 0.0
                ),
            ),
        )
        for stage in _STAGE_COST_SECONDS
    )


def _search_configuration(run_seed: int, maximum_candidates: int) -> SearchConfiguration:
    return SearchConfiguration(
        seed=run_seed,
        budget=SearchBudget(
            wall_time_seconds=maximum_candidates * 10.0,
            cpu_time_seconds=maximum_candidates * 10.0,
            gpu_time_seconds=0.0,
            cloud_cost_usd=0.0,
            external_synthesis_cost_usd=0.0,
            candidate_count=maximum_candidates,
            compilation_count=maximum_candidates,
            benchmark_count=maximum_candidates,
            verifier_time_seconds=maximum_candidates * 3.0,
        ),
        stages=_stage_plan(),
        maximum_candidates=maximum_candidates,
        maximum_archive_size=maximum_candidates,
        maximum_events=maximum_candidates * 64,
        allow_hardware=False,
    )


def _run_records(
    *,
    campaign_seed: int,
    run_seed: int,
    strategy: SearchStrategy,
    diagnosis: DiagnosisRecord,
    budget: MutationBudget,
    maximum_candidates: int,
    search_directory: Path,
) -> tuple[RawCandidateRecord, ...]:
    surface = _strategy_surface(strategy, run_seed, budget)
    designs = _designs(strategy, run_seed, budget, maximum_candidates)
    result = SearchEngine(
        _search_configuration(run_seed, maximum_candidates),
        _SyntheticH4Evaluator(run_seed),
        output_directory=search_directory,
    ).run(designs)
    return tuple(
        RawCandidateRecord(
            campaign_seed=campaign_seed,
            run_seed=run_seed,
            strategy=strategy,
            diagnosis_id=diagnosis.diagnosis_id,
            bottleneck=diagnosis.top_hypothesis,
            mutation_surface=surface,
            candidate_index=index,
            design=item.design,
            final_state=item.final_state,
            stage_results=item.stage_results,
            budget_exhausted=item.budget_exhausted,
        )
        for index, item in enumerate(result.candidates)
    )


def _record_utility(record: RawCandidateRecord) -> float | None:
    return next(
        (
            result.selection_utility
            for result in record.stage_results
            if result.stage is FidelityStage.SIMULATION
        ),
        None,
    )


def _record_duration(record: RawCandidateRecord) -> float:
    return sum(result.usage.wall_time_seconds for result in record.stage_results)


def _summarize_seed(
    records: tuple[RawCandidateRecord, ...], improvement_threshold: float
) -> SeedStrategySummary:
    if not records:
        raise CampaignValidationError("each strategy and seed must preserve candidate records")
    first = records[0]
    cumulative = 0.0
    candidates_to_improvement: int | None = None
    time_to_improvement: float | None = None
    experiments_to_improvement: int | None = None
    high_fidelity_experiments = 0
    utilities: list[float] = []
    for index, record in enumerate(records):
        if record.candidate_index != index:
            raise CampaignValidationError("candidate indices are not contiguous")
        cumulative += _record_duration(record)
        high_fidelity_experiments += any(
            result.stage is FidelityStage.END_TO_END_BENCHMARK for result in record.stage_results
        )
        utility = _record_utility(record)
        if utility is not None:
            utilities.append(utility)
            if candidates_to_improvement is None and utility >= improvement_threshold:
                candidates_to_improvement = index + 1
                time_to_improvement = cumulative
                experiments_to_improvement = high_fidelity_experiments
    families = {mutation.family for record in records for mutation in record.design.mutations}
    hardware_experiments = sum(
        result.hardware_backed for record in records for result in record.stage_results
    )
    if hardware_experiments:
        raise CampaignValidationError("synthetic campaign contains hardware-backed evidence")
    return SeedStrategySummary(
        run_seed=first.run_seed,
        strategy=first.strategy,
        mutation_surface=first.mutation_surface,
        candidates_evaluated=len(records),
        invalid_candidates=sum(
            isinstance(record.final_state, CandidateFailureState) for record in records
        ),
        actual_hardware_experiments=0,
        synthetic_high_fidelity_experiments=sum(
            any(
                result.stage is FidelityStage.END_TO_END_BENCHMARK
                for result in record.stage_results
            )
            for record in records
        ),
        synthetic_high_fidelity_experiments_to_improvement=experiments_to_improvement,
        candidates_to_improvement=candidates_to_improvement,
        time_to_improvement=time_to_improvement,
        final_objective=max(utilities) if utilities else None,
        distinct_transformation_families=len(families),
    )


def _aggregate(
    strategy: SearchStrategy,
    records: tuple[RawCandidateRecord, ...],
    run_seeds: tuple[int, ...],
    improvement_threshold: float,
) -> StrategyAggregate:
    per_seed = tuple(
        _summarize_seed(
            tuple(
                record
                for record in records
                if record.strategy is strategy and record.run_seed == seed
            ),
            improvement_threshold,
        )
        for seed in run_seeds
    )
    improved = tuple(item for item in per_seed if item.time_to_improvement is not None)
    improved_experiments = tuple(
        item.synthetic_high_fidelity_experiments_to_improvement
        for item in improved
        if item.synthetic_high_fidelity_experiments_to_improvement is not None
    )
    final = tuple(item.final_objective for item in per_seed if item.final_objective is not None)
    families = {
        mutation.family
        for record in records
        if record.strategy is strategy
        for mutation in record.design.mutations
    }
    count = len(per_seed)
    return StrategyAggregate(
        strategy=strategy,
        run_count=count,
        candidates_evaluated=sum(item.candidates_evaluated for item in per_seed),
        mean_candidates_evaluated=(sum(item.candidates_evaluated for item in per_seed) / count),
        invalid_candidates=sum(item.invalid_candidates for item in per_seed),
        mean_invalid_candidates=sum(item.invalid_candidates for item in per_seed) / count,
        actual_hardware_experiments=0,
        synthetic_high_fidelity_experiments=sum(
            item.synthetic_high_fidelity_experiments for item in per_seed
        ),
        mean_synthetic_high_fidelity_experiments_to_improvement=(
            sum(improved_experiments) / len(improved_experiments) if improved_experiments else None
        ),
        improvement_run_count=len(improved),
        improvement_rate=len(improved) / count,
        mean_candidates_to_improvement=(
            sum(item.candidates_to_improvement or 0 for item in improved) / len(improved)
            if improved
            else None
        ),
        mean_time_to_improvement=(
            sum(item.time_to_improvement or 0.0 for item in improved) / len(improved)
            if improved
            else None
        ),
        mean_final_objective=sum(final) / len(final) if final else None,
        distinct_transformation_families=len(families),
        per_seed=per_seed,
    )


def _delta(guided: StrategyAggregate, baseline: StrategyAggregate) -> StrategyDelta:
    return StrategyDelta(
        baseline=baseline.strategy,
        candidates_evaluated_reduction=(
            baseline.candidates_evaluated - guided.candidates_evaluated
        ),
        invalid_candidate_reduction=baseline.invalid_candidates - guided.invalid_candidates,
        mean_synthetic_high_fidelity_experiments_to_improvement_reduction=(
            baseline.mean_synthetic_high_fidelity_experiments_to_improvement
            - guided.mean_synthetic_high_fidelity_experiments_to_improvement
            if baseline.improvement_rate == guided.improvement_rate == 1.0
            and baseline.mean_synthetic_high_fidelity_experiments_to_improvement is not None
            and guided.mean_synthetic_high_fidelity_experiments_to_improvement is not None
            else None
        ),
        mean_time_to_improvement_reduction=(
            baseline.mean_time_to_improvement - guided.mean_time_to_improvement
            if baseline.improvement_rate == guided.improvement_rate == 1.0
            and baseline.mean_time_to_improvement is not None
            and guided.mean_time_to_improvement is not None
            else None
        ),
        mean_final_objective_delta=(
            guided.mean_final_objective - baseline.mean_final_objective
            if guided.mean_final_objective is not None and baseline.mean_final_objective is not None
            else None
        ),
    )


def _load_records(path: Path, *, maximum_records: int) -> tuple[RawCandidateRecord, ...]:
    try:
        lines = path.read_bytes().splitlines()
    except OSError as error:
        raise CampaignValidationError(f"cannot read raw candidate records: {path}") from error
    if not lines:
        raise CampaignValidationError("raw candidate record artifact is empty")
    if len(lines) > maximum_records:
        raise CampaignValidationError("raw candidate record artifact exceeds campaign bounds")
    try:
        return tuple(RawCandidateRecord.model_validate_json(line, strict=True) for line in lines)
    except ValueError as error:
        raise CampaignValidationError("raw candidate record artifact is invalid") from error


def validate_autopsy_guided_campaign(report: AutopsyCampaignReport) -> None:
    """Independently re-open raw artifacts and recompute every reported summary."""

    expected_seeds = tuple(
        _derived_seed(report.campaign_seed, index) for index in range(len(report.run_seeds))
    )
    if report.run_seeds != expected_seeds:
        raise CampaignValidationError("run seeds are not deterministically derived")
    paths = (
        (Path(report.diagnosis_path), report.diagnosis_sha256, "diagnosis"),
        (Path(report.mutation_budget_path), report.mutation_budget_sha256, "mutation budget"),
        (Path(report.raw_candidates_path), report.raw_candidates_sha256, "raw candidates"),
    )
    for path, expected_digest, label in paths:
        if not path.is_file() or path.is_symlink() or _sha256(path) != expected_digest:
            raise CampaignValidationError(f"{label} artifact is missing, symlinked, or changed")
    try:
        diagnosis = DiagnosisRecord.model_validate_json(
            Path(report.diagnosis_path).read_bytes(), strict=True
        )
        budget = MutationBudget.model_validate_json(
            Path(report.mutation_budget_path).read_bytes(), strict=True
        )
    except ValueError as error:
        raise CampaignValidationError("campaign input artifact is invalid") from error
    expected_budget = build_mutation_budget(diagnosis)
    if budget != expected_budget:
        raise CampaignValidationError("mutation budget is not derived from the bound diagnosis")
    records = _load_records(
        Path(report.raw_candidates_path),
        maximum_records=(
            len(report.run_seeds) * len(SearchStrategy) * report.maximum_candidates_per_run
        ),
    )
    expected_keys = {(strategy, seed) for strategy in SearchStrategy for seed in report.run_seeds}
    actual_keys = {(record.strategy, record.run_seed) for record in records}
    if actual_keys != expected_keys:
        raise CampaignValidationError("raw records do not cover every strategy and seed")
    for strategy, seed in sorted(expected_keys, key=lambda item: (item[0].value, item[1])):
        grouped = tuple(
            record for record in records if record.strategy is strategy and record.run_seed == seed
        )
        expected_surface = _strategy_surface(strategy, seed, budget)
        if any(
            record.campaign_seed != report.campaign_seed
            or record.diagnosis_id != diagnosis.diagnosis_id
            or record.bottleneck is not diagnosis.top_hypothesis
            or record.mutation_surface != expected_surface
            for record in grouped
        ):
            raise CampaignValidationError("raw record provenance or mutation surface differs")
        expected_designs = _designs(strategy, seed, budget, report.maximum_candidates_per_run)
        if {record.design for record in grouped} != set(expected_designs):
            raise CampaignValidationError(
                "raw candidate designs differ from deterministic proposals"
            )
        if len(grouped) != len(expected_designs):
            raise CampaignValidationError("raw candidate records contain duplicate designs")
        if strategy is SearchStrategy.AUTOPSY_GUIDED:
            guard = MutationGuard(budget)
            for record in grouped:
                guard.validate(record.design)
        if any(result.hardware_backed for record in grouped for result in record.stage_results):
            raise CampaignValidationError("synthetic campaign contains hardware-backed evidence")
    expected_aggregates = tuple(
        _aggregate(strategy, records, report.run_seeds, report.improvement_threshold)
        for strategy in SearchStrategy
    )
    if report.aggregates != expected_aggregates:
        raise CampaignValidationError("campaign aggregates are not derived from raw candidates")
    guided = expected_aggregates[0]
    expected_deltas = tuple(_delta(guided, item) for item in expected_aggregates[1:])
    if report.guided_deltas != expected_deltas:
        raise CampaignValidationError("guided comparisons are not derived from aggregates")


def _safe_reset(path: Path) -> None:
    if not path.exists():
        return
    if path.is_symlink():
        raise ValueError(f"refusing to reset symlinked campaign path: {path}")
    resolved = path.resolve()
    repository = Path(__file__).resolve().parents[4]
    if (
        resolved in {Path("/").resolve(), Path.home().resolve(), repository}
        or len(resolved.parts) < 4
    ):
        raise ValueError(f"refusing to reset unsafe campaign path: {resolved}")
    shutil.rmtree(resolved)


def run_autopsy_guided_campaign(
    output_directory: Path,
    *,
    diagnosis_path: Path,
    seed: int,
    count: int = 5,
    maximum_candidates: int = 12,
    improvement_threshold: float = 0.12,
    reset: bool = False,
) -> AutopsyCampaignReport:
    """Run all three H4 strategies and publish a self-validating report."""

    if seed < 0:
        raise ValueError("campaign seed must be non-negative")
    if not 1 <= count <= _MAXIMUM_SEEDS:
        raise ValueError(f"campaign count must be in [1, {_MAXIMUM_SEEDS}]")
    if not 1 <= maximum_candidates <= _MAXIMUM_CANDIDATES:
        raise ValueError(f"maximum candidates must be in [1, {_MAXIMUM_CANDIDATES}]")
    if not math.isfinite(improvement_threshold) or improvement_threshold <= 0.0:
        raise ValueError("improvement threshold must be finite and positive")
    if reset:
        _safe_reset(output_directory)
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(f"campaign output is not empty: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)
    if diagnosis_path.is_symlink() or not diagnosis_path.is_file():
        raise ValueError("diagnosis path must be a regular, non-symlinked file")
    diagnosis_payload = diagnosis_path.read_bytes()
    if len(diagnosis_payload) > _MAXIMUM_DIAGNOSIS_BYTES:
        raise ValueError("diagnosis exceeds the campaign input-size bound")
    diagnosis = DiagnosisRecord.model_validate_json(diagnosis_payload, strict=True)
    budget = build_mutation_budget(diagnosis)
    copied_diagnosis = output_directory / "inputs" / "diagnosis.json"
    budget_path = output_directory / "inputs" / "mutation-budget.json"
    _write_document(copied_diagnosis, diagnosis)
    _write_document(budget_path, budget)
    run_seeds = tuple(_derived_seed(seed, index) for index in range(count))
    records = tuple(
        record
        for run_seed in run_seeds
        for strategy in SearchStrategy
        for record in _run_records(
            campaign_seed=seed,
            run_seed=run_seed,
            strategy=strategy,
            diagnosis=diagnosis,
            budget=budget,
            maximum_candidates=maximum_candidates,
            search_directory=(output_directory / "search" / f"seed-{run_seed}" / strategy.value),
        )
    )
    raw_path = output_directory / "raw-candidates.jsonl"
    _write_records(raw_path, records)
    aggregates = tuple(
        _aggregate(strategy, records, run_seeds, improvement_threshold)
        for strategy in SearchStrategy
    )
    report_path = output_directory / "report.json"
    report = AutopsyCampaignReport(
        campaign_seed=seed,
        run_seeds=run_seeds,
        maximum_candidates_per_run=maximum_candidates,
        improvement_threshold=improvement_threshold,
        scope=CampaignScope(),
        diagnosis_path=str(copied_diagnosis.resolve()),
        diagnosis_sha256=_sha256(copied_diagnosis),
        mutation_budget_path=str(budget_path.resolve()),
        mutation_budget_sha256=_sha256(budget_path),
        raw_candidates_path=str(raw_path.resolve()),
        raw_candidates_sha256=_sha256(raw_path),
        aggregates=aggregates,
        guided_deltas=tuple(_delta(aggregates[0], item) for item in aggregates[1:]),
        limitations=(
            "the evaluator is deterministic synthetic evidence, not wall-clock measurement",
            "actual hardware experiment counts are zero and no GPU claim is supported",
            "the campaign evaluates one synthetic network-degradation diagnosis and mutation grammar",
            "final objective values are normalized utility points, not production latency",
        ),
        report_path=str(report_path.resolve()),
    )
    validate_autopsy_guided_campaign(report)
    _write_document(report_path, report)
    persisted = AutopsyCampaignReport.model_validate_json(report_path.read_bytes(), strict=True)
    validate_autopsy_guided_campaign(persisted)
    return persisted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=73129)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--maximum-candidates", type=int, default=12)
    parser.add_argument("--improvement-threshold", type=float, default=0.12)
    parser.add_argument("--reset", action="store_true")
    arguments = parser.parse_args(argv)
    report = run_autopsy_guided_campaign(
        arguments.output,
        diagnosis_path=arguments.diagnosis,
        seed=arguments.seed,
        count=arguments.count,
        maximum_candidates=arguments.maximum_candidates,
        improvement_threshold=arguments.improvement_threshold,
        reset=arguments.reset,
    )
    print(json.dumps(report.model_dump(mode="json"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AutopsyCampaignReport",
    "CampaignScope",
    "CampaignValidationError",
    "RawCandidateRecord",
    "SearchStrategy",
    "SeedStrategySummary",
    "StrategyAggregate",
    "StrategyDelta",
    "main",
    "run_autopsy_guided_campaign",
    "validate_autopsy_guided_campaign",
]
