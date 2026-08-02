"""Deterministic multi-fidelity search orchestration with auditable persistence."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

from sloforge.genesis.ir import (
    BudgetUsage,
    Candidate,
    CandidateState,
    CandidateSuccessState,
)

from .budget import BudgetExceeded, BudgetManager
from .lifecycle import (
    CandidateRepository,
    CanonicalEventStore,
    proposed_candidate,
    transition,
    with_usage,
)
from .models import (
    CandidateDesign,
    CandidateSearchResult,
    FidelityStage,
    SearchConfiguration,
    SearchEvent,
    SearchRunResult,
    StageResult,
)
from .pareto import ParetoArchive
from .surrogate import DeterministicSurrogateRanker, SurrogateObservation


class StageEvaluator(Protocol):
    """Independent adapter; implementations must honor their reserved timeout/cost."""

    def evaluate(
        self,
        candidate: CandidateDesign,
        stage: FidelityStage,
        *,
        seed: int,
    ) -> StageResult: ...


_SUCCESS_TRANSITIONS: dict[FidelityStage, CandidateSuccessState] = {
    FidelityStage.STATIC_PRUNING: CandidateSuccessState.STATICALLY_VALID,
    FidelityStage.COMPILE: CandidateSuccessState.COMPILED,
    FidelityStage.DETERMINISTIC_TESTS: CandidateSuccessState.REFERENCE_TESTED,
    FidelityStage.PROPERTY_VERIFICATION: CandidateSuccessState.PROPERTY_TESTED,
    FidelityStage.MODEL_CHECK: CandidateSuccessState.MODEL_CHECKED,
    FidelityStage.SIMULATION: CandidateSuccessState.SIMULATED,
    FidelityStage.HARDWARE_MICROBENCHMARK: CandidateSuccessState.HARDWARE_BENCHMARKED,
    FidelityStage.SHADOW: CandidateSuccessState.SHADOW_VALIDATED,
    FidelityStage.CANARY: CandidateSuccessState.CANARY_VALIDATED,
}


def _stage_seed(run_seed: int, candidate_id: str, stage: FidelityStage) -> int:
    payload = f"{run_seed}\0{candidate_id}\0{stage.value}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


class SearchEngine:
    def __init__(
        self,
        configuration: SearchConfiguration,
        evaluator: StageEvaluator,
        *,
        output_directory: Path,
    ) -> None:
        self.configuration = configuration
        self.evaluator = evaluator
        self.output_directory = output_directory
        self.budget = BudgetManager(configuration.budget)
        self.events = CanonicalEventStore(
            output_directory / "events.jsonl", maximum_events=configuration.maximum_events
        )
        if self.events.events:
            raise ValueError("search output directory already contains an event log")
        self.candidates = CandidateRepository(output_directory / "candidates")
        self.archive = ParetoArchive(configuration.maximum_archive_size)
        self.ranker = DeterministicSurrogateRanker()

    def _event(
        self,
        event_type: str,
        reason: str,
        *,
        candidate_id: str | None = None,
        stage: FidelityStage | None = None,
        from_state: CandidateState | None = None,
        to_state: CandidateState | None = None,
        passed: bool | None = None,
        evidence_ids: tuple[str, ...] = (),
    ) -> None:
        self.events.append(
            SearchEvent(
                sequence=len(self.events.events),
                event_type=event_type,  # type: ignore[arg-type]
                candidate_id=candidate_id,
                stage=stage,
                from_state=from_state,
                to_state=to_state,
                passed=passed,
                reason=reason,
                usage=self.budget.usage,
                evidence_ids=evidence_ids,
            )
        )

    def _consume(
        self,
        actual: BudgetUsage,
        maximum: BudgetUsage,
        candidate: Candidate,
        stage: FidelityStage,
    ) -> Candidate:
        usage = self.budget.consume(actual, reserved_maximum=maximum)
        updated = with_usage(candidate, usage)
        self.candidates.save(updated)
        self._event(
            "budget_consumed",
            f"consumed bounded evaluator usage for {stage.value}",
            candidate_id=candidate.candidate_id,
            stage=stage,
        )
        return updated

    def run(self, designs: tuple[CandidateDesign, ...]) -> SearchRunResult:
        unique: dict[str, CandidateDesign] = {}
        for design in designs:
            unique.setdefault(design.candidate_id, design)
        remaining = tuple(unique.values())[: self.configuration.maximum_candidates]
        observations: list[SurrogateObservation] = []
        results: list[CandidateSearchResult] = []
        while remaining:
            ranked = self.ranker.rank(remaining, tuple(observations))
            design = ranked[0].candidate
            remaining = tuple(
                item for item in remaining if item.candidate_id != design.candidate_id
            )
            try:
                usage = self.budget.consume_candidate()
            except BudgetExceeded as error:
                self._event("budget_exhausted", str(error))
                break
            candidate = proposed_candidate(
                design,
                budget=self.configuration.budget,
                usage=usage,
            )
            self.candidates.save(candidate)
            self._event(
                "candidate_proposed",
                f"candidate selected by {design.proposal_engine}",
                candidate_id=design.candidate_id,
                to_state=candidate.state,
            )
            stage_results: list[StageResult] = []
            exhausted = False
            for specification in self.configuration.stages:
                try:
                    self.budget.ensure_reservable(specification.maximum_usage)
                except BudgetExceeded as error:
                    exhausted = True
                    self._event(
                        "budget_exhausted",
                        str(error),
                        candidate_id=design.candidate_id,
                        stage=specification.stage,
                    )
                    break
                result = self.evaluator.evaluate(
                    design,
                    specification.stage,
                    seed=_stage_seed(
                        self.configuration.seed, design.candidate_id, specification.stage
                    ),
                )
                if result.stage is not specification.stage:
                    raise ValueError("evaluator returned evidence for the wrong fidelity stage")
                if (
                    result.passed
                    and specification.stage is FidelityStage.HARDWARE_MICROBENCHMARK
                    and not result.hardware_backed
                ):
                    raise ValueError(
                        "hardware benchmark stage cannot silently use synthetic evidence"
                    )
                candidate = self._consume(
                    result.usage,
                    specification.maximum_usage,
                    candidate,
                    specification.stage,
                )
                stage_results.append(result)
                self._event(
                    "stage_result",
                    result.reason,
                    candidate_id=design.candidate_id,
                    stage=result.stage,
                    passed=result.passed,
                    evidence_ids=result.evidence_ids,
                )
                if not result.passed:
                    if result.failure_state is None:
                        raise ValueError("failed evaluator result has no terminal failure state")
                    previous = candidate.state
                    candidate = transition(candidate, result.failure_state, result.reason)
                    self.candidates.save(candidate)
                    self._event(
                        "lifecycle_transition",
                        result.reason,
                        candidate_id=design.candidate_id,
                        stage=result.stage,
                        from_state=previous,
                        to_state=candidate.state,
                        passed=False,
                    )
                    break
                target = _SUCCESS_TRANSITIONS.get(result.stage)
                if target is not None:
                    previous = candidate.state
                    candidate = transition(candidate, target, result.reason)
                    self.candidates.save(candidate)
                    self._event(
                        "lifecycle_transition",
                        result.reason,
                        candidate_id=design.candidate_id,
                        stage=result.stage,
                        from_state=previous,
                        to_state=candidate.state,
                        passed=True,
                    )
                if result.objective is not None:
                    added = self.archive.add(design, result.objective)
                    self._event(
                        "pareto_updated",
                        "candidate retained in Pareto archive"
                        if added
                        else "candidate dominated or evicted by diversity bound",
                        candidate_id=design.candidate_id,
                        stage=result.stage,
                        passed=added,
                    )
                if result.selection_utility is not None:
                    observations.append(
                        SurrogateObservation(
                            feature_vector=design.feature_vector,
                            utility=result.selection_utility,
                        )
                    )
            results.append(
                CandidateSearchResult(
                    design=design,
                    final_state=candidate.state,
                    stage_results=tuple(stage_results),
                    budget_exhausted=exhausted,
                )
            )
        return SearchRunResult(
            seed=self.configuration.seed,
            candidates=tuple(results),
            pareto_candidate_ids=tuple(
                entry.candidate.candidate_id for entry in self.archive.entries
            ),
            usage=self.budget.usage,
            events_path=str(self.events.path.resolve()),
        )
