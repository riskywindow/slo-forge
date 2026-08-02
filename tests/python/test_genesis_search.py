from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from sloforge.genesis.ir import (
    ArtifactDigest,
    BudgetUsage,
    CandidateFailureState,
    CandidateSuccessState,
    SearchBudget,
    TransformationFamily,
    load_candidate,
)
from sloforge.genesis.search import (
    BudgetExceeded,
    BudgetManager,
    CandidateDesign,
    DeterministicSurrogateRanker,
    FidelityStage,
    MutationChoice,
    ObjectiveVector,
    ParameterValue,
    ParetoArchive,
    ProposalPortfolio,
    ProposalRequest,
    SearchConfiguration,
    SearchEngine,
    StageResult,
    StageSpecification,
    SurrogateObservation,
)
from sloforge.genesis.search.lifecycle import CanonicalEventStore
from sloforge.genesis.search.proposals import (
    BeamProposalEngine,
    EvolutionaryProposalEngine,
    LocalProposalEngine,
)


def _budget(**overrides: float | int) -> SearchBudget:
    values: dict[str, float | int] = {
        "wall_time_seconds": 100.0,
        "cpu_time_seconds": 100.0,
        "gpu_time_seconds": 0.0,
        "cloud_cost_usd": 0.0,
        "external_synthesis_cost_usd": 0.0,
        "candidate_count": 8,
        "compilation_count": 8,
        "benchmark_count": 8,
        "verifier_time_seconds": 100.0,
    }
    values.update(overrides)
    return SearchBudget.model_validate(values)


def _usage(**overrides: float | int) -> BudgetUsage:
    values: dict[str, float | int] = {
        "wall_time_seconds": 0.0,
        "cpu_time_seconds": 0.0,
        "gpu_time_seconds": 0.0,
        "cloud_cost_usd": 0.0,
        "external_synthesis_cost_usd": 0.0,
        "candidate_count": 0,
        "compilation_count": 0,
        "benchmark_count": 0,
        "verifier_time_seconds": 0.0,
    }
    values.update(overrides)
    return BudgetUsage.model_validate(values)


def test_stage_result_rejects_lifecycle_failure_from_an_unrelated_stage() -> None:
    with pytest.raises(ValueError, match="incompatible with the fidelity stage"):
        StageResult(
            stage=FidelityStage.STATIC_PRUNING,
            passed=False,
            reason="not a canary observation",
            failure_state=CandidateFailureState.CANARY_REJECTED,
        )

    valid = StageResult(
        stage=FidelityStage.MODEL_CHECK,
        passed=False,
        reason="bounded protocol counterexample",
        failure_state=CandidateFailureState.MODEL_CHECK_REJECTED,
    )
    assert valid.failure_state is CandidateFailureState.MODEL_CHECK_REJECTED


def _mutation(
    identifier: str,
    family: TransformationFamily,
    regions: tuple[str, ...],
    delta: tuple[float, ...],
    upside: float,
) -> MutationChoice:
    return MutationChoice(
        transformation_id=identifier,
        family=family,
        regions=regions,
        parameters=(ParameterValue(key="mode", value=identifier),),
        expected_upside=upside,
        invalidity_risk=0.05,
        feature_delta=delta,
    )


def _options() -> tuple[MutationChoice, ...]:
    return (
        _mutation(
            "deadline-batching",
            TransformationFamily.BATCHING,
            ("request", "serving"),
            (1.0, 0.0, 0.0),
            0.20,
        ),
        _mutation(
            "paged-state",
            TransformationFamily.STATE_LAYOUT,
            ("state",),
            (0.0, 1.0, 0.0),
            0.16,
        ),
        _mutation(
            "fused-kernel",
            TransformationFamily.KERNEL,
            ("kernel",),
            (0.0, 0.0, 1.0),
            0.25,
        ),
        _mutation(
            "cache-aware-routing",
            TransformationFamily.SCHEDULER,
            ("request", "state"),
            (0.5, 0.5, 0.0),
            0.18,
        ),
    )


def _request(*, parents: tuple[CandidateDesign, ...] = ()) -> ProposalRequest:
    return ProposalRequest(
        base_genome_hash=ArtifactDigest(value=hashlib.sha256(b"base-genome").hexdigest()),
        seed=73129,
        base_features=(0.0, 0.0, 0.0),
        options=_options(),
        parents=parents,
        mutable_regions=("request", "serving", "state"),
        maximum_proposals=8,
    )


@pytest.mark.parametrize(
    "dimension",
    [
        "wall_time_seconds",
        "cpu_time_seconds",
        "gpu_time_seconds",
        "cloud_cost_usd",
        "external_synthesis_cost_usd",
        "candidate_count",
        "compilation_count",
        "benchmark_count",
        "verifier_time_seconds",
    ],
)
def test_budget_manager_hard_limits_every_dimension(dimension: str) -> None:
    budget = _budget(**{dimension: 0})
    manager = BudgetManager(budget)
    charge = _usage(**{dimension: 1})

    with pytest.raises(BudgetExceeded) as captured:
        manager.ensure_reservable(charge)

    assert captured.value.dimensions == (dimension,)
    assert manager.usage == BudgetUsage()


def test_budget_rejects_evaluator_over_reservation() -> None:
    manager = BudgetManager(_budget())

    with pytest.raises(ValueError, match="exceeded its reserved budget"):
        manager.consume(
            _usage(cpu_time_seconds=2.0),
            reserved_maximum=_usage(cpu_time_seconds=1.0),
        )

    assert manager.usage == BudgetUsage()


def test_proposal_engines_are_deterministic_diverse_and_respect_frozen_regions() -> None:
    request = _request()
    first = ProposalPortfolio().propose(request)
    second = ProposalPortfolio().propose(request)

    assert first == second
    assert first
    assert len(
        {tuple(item.transformation_id for item in design.mutations) for design in first}
    ) == len(first)
    assert all("kernel" not in design.affected_regions for design in first)
    assert any(design.cross_layer for design in first)
    assert {design.proposal_engine for design in first} >= {"beam", "novelty"}

    parents = BeamProposalEngine().propose(request)[:2]
    evolved = EvolutionaryProposalEngine().propose(_request(parents=parents))
    local = LocalProposalEngine().propose(_request(parents=parents))
    assert evolved and local
    assert evolved == EvolutionaryProposalEngine().propose(_request(parents=parents))
    assert all(item.parent_candidate_ids for item in (*evolved, *local))


def _objective(*, ttft: float, quality: float, goodput: float) -> ObjectiveVector:
    return ObjectiveVector(
        correctness_confidence=1.0,
        quality=quality,
        ttft_ms=ttft,
        token_latency_ms=4.0,
        goodput=goodput,
        throughput=goodput * 1.2,
        cost_usd_per_hour=1.0,
        energy_joules_per_token=0.1,
        startup_ms=20.0,
        memory_bytes=1024.0,
        reliability=0.999,
        implementation_complexity=2.0,
        transition_cost=1.0,
    )


def test_pareto_archive_and_surrogate_preserve_tradeoffs_deterministically() -> None:
    candidates = ProposalPortfolio().propose(_request())[:3]
    archive = ParetoArchive(maximum_size=2)
    assert archive.add(candidates[0], _objective(ttft=10, quality=1.0, goodput=80))
    assert archive.add(candidates[1], _objective(ttft=8, quality=0.99, goodput=90))
    assert not archive.add(candidates[2], _objective(ttft=20, quality=0.9, goodput=40))
    assert len(archive.entries) == 2

    observations = (
        SurrogateObservation(candidates[0].feature_vector, 0.2),
        SurrogateObservation(candidates[1].feature_vector, 0.4),
    )
    ranker = DeterministicSurrogateRanker(exploration_weight=0.3)
    assert ranker.rank(candidates, observations) == ranker.rank(candidates, observations)


class _PassingEvaluator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, FidelityStage, int]] = []

    def evaluate(
        self, candidate: CandidateDesign, stage: FidelityStage, *, seed: int
    ) -> StageResult:
        self.calls.append((candidate.candidate_id, stage, seed))
        usage = _usage(
            wall_time_seconds=0.1,
            cpu_time_seconds=0.1,
            compilation_count=1 if stage is FidelityStage.COMPILE else 0,
            verifier_time_seconds=(
                0.05
                if stage
                in {
                    FidelityStage.DETERMINISTIC_TESTS,
                    FidelityStage.PROPERTY_VERIFICATION,
                    FidelityStage.MODEL_CHECK,
                }
                else 0.0
            ),
        )
        objective = (
            _objective(ttft=10.0, quality=1.0, goodput=80.0)
            if stage is FidelityStage.SIMULATION
            else None
        )
        return StageResult(
            stage=stage,
            passed=True,
            reason=f"deterministic fixture passed {stage.value}",
            usage=usage,
            objective=objective,
            selection_utility=0.5 if objective is not None else None,
            evidence_ids=(f"evidence-{stage.value}",),
        )


def _stage_plan() -> tuple[StageSpecification, ...]:
    stages = (
        FidelityStage.STATIC_PRUNING,
        FidelityStage.ANALYTICAL_BOUND,
        FidelityStage.COMPILE,
        FidelityStage.DIGITAL_TWIN,
        FidelityStage.DETERMINISTIC_TESTS,
        FidelityStage.PROPERTY_VERIFICATION,
        FidelityStage.MODEL_CHECK,
        FidelityStage.SIMULATION,
    )
    return tuple(
        StageSpecification(
            stage=stage,
            maximum_usage=_usage(
                wall_time_seconds=1.0,
                cpu_time_seconds=1.0,
                compilation_count=1 if stage is FidelityStage.COMPILE else 0,
                verifier_time_seconds=(
                    1.0
                    if stage
                    in {
                        FidelityStage.DETERMINISTIC_TESTS,
                        FidelityStage.PROPERTY_VERIFICATION,
                        FidelityStage.MODEL_CHECK,
                    }
                    else 0.0
                ),
            ),
        )
        for stage in stages
    )


def test_search_engine_persists_complete_lifecycle_and_canonical_events(tmp_path: Path) -> None:
    designs = ProposalPortfolio().propose(_request())[:2]
    evaluator = _PassingEvaluator()
    configuration = SearchConfiguration(
        seed=73129,
        budget=_budget(candidate_count=2, compilation_count=2),
        stages=_stage_plan(),
        maximum_candidates=2,
        maximum_archive_size=4,
    )
    engine = SearchEngine(configuration, evaluator, output_directory=tmp_path)

    result = engine.run(designs)

    assert len(result.candidates) == 2
    assert all(item.final_state is CandidateSuccessState.SIMULATED for item in result.candidates)
    assert result.usage.candidate_count == 2
    assert result.usage.compilation_count == 2
    assert result.pareto_candidate_ids
    loaded_events = CanonicalEventStore(
        tmp_path / "events.jsonl", maximum_events=configuration.maximum_events
    )
    assert loaded_events.events == engine.events.events
    for item in result.candidates:
        candidate = load_candidate(engine.candidates.path_for(item.design.candidate_id))
        assert candidate.state is CandidateSuccessState.SIMULATED
        assert tuple(event.sequence for event in candidate.lifecycle) == tuple(
            range(len(candidate.lifecycle))
        )


def test_search_never_invokes_stage_without_budget_reservation(tmp_path: Path) -> None:
    design = ProposalPortfolio().propose(_request())[:1]
    evaluator = _PassingEvaluator()
    configuration = SearchConfiguration(
        seed=11,
        budget=_budget(candidate_count=1, compilation_count=0),
        stages=_stage_plan(),
        maximum_candidates=1,
        maximum_archive_size=2,
    )
    result = SearchEngine(configuration, evaluator, output_directory=tmp_path).run(design)

    called_stages = [stage for _, stage, _ in evaluator.calls]
    assert called_stages == [FidelityStage.STATIC_PRUNING, FidelityStage.ANALYTICAL_BOUND]
    assert result.candidates[0].budget_exhausted
    assert result.candidates[0].final_state is CandidateSuccessState.STATICALLY_VALID


def test_search_rejects_conflicting_candidate_identifier_collision(tmp_path: Path) -> None:
    design = ProposalPortfolio().propose(_request())[0]
    conflicting = design.model_copy(update={"feature_vector": (99.0,)})
    configuration = SearchConfiguration(
        seed=11,
        budget=_budget(candidate_count=2, compilation_count=2),
        stages=_stage_plan(),
        maximum_candidates=2,
        maximum_archive_size=2,
    )

    with pytest.raises(ValueError, match="identifier collision"):
        SearchEngine(configuration, _PassingEvaluator(), output_directory=tmp_path).run(
            (design, conflicting)
        )
