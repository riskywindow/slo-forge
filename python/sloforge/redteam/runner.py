"""Bounded executable red-team runner and regression replay."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Iterable
from typing import Protocol

from sloforge.genesis.ir import RequestEventCase

from .adversaries import (
    generate_resource_cases,
    generate_schedule_cases,
    generate_tensor_cases,
    generate_topology_cases,
)
from .benchmark import audit_benchmark_integrity
from .conversion import build_regression_corpus, to_counterexample
from .fixture import renumber_events
from .minimize import minimize_sequence
from .models import (
    AdversarialCase,
    BenchmarkAuditCase,
    BenchmarkComparison,
    RedTeamConfiguration,
    RedTeamFinding,
    RedTeamReport,
    RedTeamSurface,
    RegressionCorpus,
    RegressionReplayResult,
    ResourceAdversarialCase,
    ResourceAdversaryConfiguration,
    ScheduleAdversarialCase,
    ScheduleAdversaryConfiguration,
    SurfaceCount,
    TargetDescriptor,
    TensorAdversarialCase,
    TensorAdversaryConfiguration,
    TopologyAdversarialCase,
    TopologyAdversaryConfiguration,
    ViolationObservation,
)


class RedTeamTarget(Protocol):
    descriptor: TargetDescriptor

    def evaluate_tensor(self, case: TensorAdversarialCase) -> ViolationObservation | None: ...

    def evaluate_schedule(self, case: ScheduleAdversarialCase) -> ViolationObservation | None: ...

    def evaluate_topology(self, case: TopologyAdversarialCase) -> ViolationObservation | None: ...

    def evaluate_resource(self, case: ResourceAdversarialCase) -> ViolationObservation | None: ...


def _derived_seed(seed: int, surface: RedTeamSurface) -> int:
    digest = hashlib.sha256(f"{seed}:{surface.value}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _minimize_schedule(
    target: RedTeamTarget,
    case: ScheduleAdversarialCase,
    observation: ViolationObservation,
    configuration: RedTeamConfiguration,
    remaining_seconds: float,
) -> tuple[ScheduleAdversarialCase, int]:
    def reproduces(events: tuple[RequestEventCase, ...]) -> bool:
        if not events:
            return False
        candidate = ScheduleAdversarialCase(
            case_id=case.case_id,
            events=renumber_events(tuple(events)),
        )
        result = target.evaluate_schedule(candidate)
        return result is not None and result.violated_contract == observation.violated_contract

    minimized = minimize_sequence(
        case.events,
        reproduces,
        seed=_derived_seed(configuration.seed, RedTeamSurface.PROTOCOL),
        maximum_evaluations=configuration.maximum_minimization_evaluations,
        timeout_seconds=min(
            configuration.minimization_timeout_seconds, max(0.001, remaining_seconds)
        ),
    )
    minimized_case = ScheduleAdversarialCase(
        case_id=f"{case.case_id}-min",
        events=renumber_events(minimized.items),
    )
    return minimized_case, minimized.evaluations


def run_red_team(
    *,
    target: RedTeamTarget,
    configuration: RedTeamConfiguration,
    benchmark_comparison: BenchmarkComparison | None = None,
) -> RedTeamReport:
    """Execute all bounded adversaries and return auditable counterexamples."""

    deadline = time.monotonic() + configuration.run_timeout_seconds
    cases_by_surface: tuple[
        tuple[
            RedTeamSurface,
            Iterable[AdversarialCase],
            Callable[[AdversarialCase], ViolationObservation | None],
        ],
        ...,
    ] = (
        (
            RedTeamSurface.TENSOR,
            generate_tensor_cases(
                TensorAdversaryConfiguration(
                    seed=_derived_seed(configuration.seed, RedTeamSurface.TENSOR),
                    maximum_cases=configuration.tensor_cases,
                )
            ),
            lambda case: target.evaluate_tensor(_as_tensor(case)),
        ),
        (
            RedTeamSurface.PROTOCOL,
            generate_schedule_cases(
                ScheduleAdversaryConfiguration(
                    seed=_derived_seed(configuration.seed, RedTeamSurface.PROTOCOL),
                    maximum_cases=configuration.schedule_cases,
                )
            ),
            lambda case: target.evaluate_schedule(_as_schedule(case)),
        ),
        (
            RedTeamSurface.TOPOLOGY,
            generate_topology_cases(
                TopologyAdversaryConfiguration(
                    seed=_derived_seed(configuration.seed, RedTeamSurface.TOPOLOGY),
                    maximum_cases=configuration.topology_cases,
                )
            ),
            lambda case: target.evaluate_topology(_as_topology(case)),
        ),
        (
            RedTeamSurface.RESOURCE,
            generate_resource_cases(
                ResourceAdversaryConfiguration(
                    seed=_derived_seed(configuration.seed, RedTeamSurface.RESOURCE),
                    maximum_cases=configuration.resource_cases,
                )
            ),
            lambda case: target.evaluate_resource(_as_resource(case)),
        ),
    )
    evaluated = {surface: 0 for surface in RedTeamSurface}
    violations = {surface: 0 for surface in RedTeamSurface}
    findings: list[RedTeamFinding] = []
    seen_contracts: set[tuple[RedTeamSurface, str]] = set()
    timed_out = False

    def capture(
        original_case: AdversarialCase,
        minimized_case: AdversarialCase,
        observation: ViolationObservation,
        minimization_evaluations: int,
    ) -> None:
        key = (observation.surface, observation.violated_contract)
        violations[observation.surface] += 1
        if key in seen_contracts or len(findings) >= configuration.maximum_findings:
            return
        seen_contracts.add(key)
        counterexample, constraint = to_counterexample(
            target=target.descriptor,
            case=minimized_case,
            observation=observation,
            seed=configuration.seed,
        )
        findings.append(
            RedTeamFinding(
                finding_id=f"finding-{counterexample.counterexample_id.removeprefix('cex-')}",
                candidate_id=target.descriptor.candidate_id,
                surface=observation.surface,
                original_case=original_case,
                minimized_case=minimized_case,
                minimization_evaluations=minimization_evaluations,
                observation=observation,
                counterexample=counterexample,
                learned_constraint=constraint,
            )
        )

    for surface, cases, evaluator in cases_by_surface:
        for case in cases:
            if time.monotonic() >= deadline:
                timed_out = True
                break
            evaluated[surface] += 1
            observation = evaluator(case)
            if observation is None:
                continue
            minimized_case = case
            minimization_evaluations = 0
            if isinstance(case, ScheduleAdversarialCase):
                minimized_case, minimization_evaluations = _minimize_schedule(
                    target,
                    case,
                    observation,
                    configuration,
                    deadline - time.monotonic(),
                )
            capture(case, minimized_case, observation, minimization_evaluations)
        if timed_out:
            break

    if not timed_out and benchmark_comparison is not None:
        evaluated[RedTeamSurface.BENCHMARK] = 1
        for issue in audit_benchmark_integrity(benchmark_comparison):
            case = BenchmarkAuditCase(
                case_id=f"benchmark-{issue.code.value}",
                issue_code=issue.code.value,
                baseline_run_id=issue.baseline_run_id,
                candidate_run_id=issue.candidate_run_id,
            )
            observation = ViolationObservation(
                surface=RedTeamSurface.BENCHMARK,
                violated_contract=f"benchmark.integrity.{issue.code.value}",
                expected_behavior=issue.message,
                observed_behavior=f"integrity auditor detected {issue.code.value}",
                learned_precondition=f"benchmark audit must not report {issue.code.value}",
            )
            capture(case, case, observation, 0)

    counts = tuple(
        SurfaceCount(
            surface=surface,
            evaluated=evaluated[surface],
            violations=violations[surface],
        )
        for surface in RedTeamSurface
    )
    return RedTeamReport(
        seed=configuration.seed,
        target=target.descriptor,
        counts=counts,
        findings=tuple(findings),
        timed_out=timed_out,
        evaluation_steps=sum(count.evaluated for count in counts),
    )


def corpus_from_report(report: RedTeamReport) -> RegressionCorpus:
    return build_regression_corpus(
        seed=report.seed,
        findings=tuple(
            (finding.minimized_case, finding.counterexample, finding.learned_constraint)
            for finding in report.findings
        ),
    )


def replay_regression_corpus(
    *,
    target: RedTeamTarget,
    corpus: RegressionCorpus,
    benchmark_comparison: BenchmarkComparison | None = None,
) -> tuple[RegressionReplayResult, ...]:
    benchmark_contracts = (
        {
            f"benchmark.integrity.{issue.code.value}"
            for issue in audit_benchmark_integrity(benchmark_comparison)
        }
        if benchmark_comparison is not None
        else set()
    )
    results: list[RegressionReplayResult] = []
    for regression in corpus.cases:
        case = regression.case
        observation: ViolationObservation | None
        if isinstance(case, TensorAdversarialCase):
            observation = target.evaluate_tensor(case)
        elif isinstance(case, ScheduleAdversarialCase):
            observation = target.evaluate_schedule(case)
        elif isinstance(case, TopologyAdversarialCase):
            observation = target.evaluate_topology(case)
        elif isinstance(case, ResourceAdversarialCase):
            observation = target.evaluate_resource(case)
        else:
            observation = None
        observed_contract = observation.violated_contract if observation is not None else None
        reproduced = observed_contract == regression.violated_contract
        if isinstance(case, BenchmarkAuditCase):
            observed_contract = (
                regression.violated_contract
                if regression.violated_contract in benchmark_contracts
                else None
            )
            reproduced = observed_contract is not None
        results.append(
            RegressionReplayResult(
                regression_id=regression.regression_id,
                reproduced=reproduced,
                observed_contract=observed_contract,
            )
        )
    return tuple(results)


def _as_tensor(case: AdversarialCase) -> TensorAdversarialCase:
    if not isinstance(case, TensorAdversarialCase):
        raise TypeError("expected tensor case")
    return case


def _as_schedule(case: AdversarialCase) -> ScheduleAdversarialCase:
    if not isinstance(case, ScheduleAdversarialCase):
        raise TypeError("expected schedule case")
    return case


def _as_topology(case: AdversarialCase) -> TopologyAdversarialCase:
    if not isinstance(case, TopologyAdversarialCase):
        raise TypeError("expected topology case")
    return case


def _as_resource(case: AdversarialCase) -> ResourceAdversarialCase:
    if not isinstance(case, ResourceAdversarialCase):
        raise TypeError("expected resource case")
    return case


__all__ = [
    "RedTeamTarget",
    "corpus_from_report",
    "replay_regression_corpus",
    "run_red_team",
]
