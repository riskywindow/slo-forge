"""Conversion from executable red-team findings to trusted Genesis evidence."""

from __future__ import annotations

from typing import Literal, cast

from sloforge.genesis.ir import (
    BehaviorObservation,
    Counterexample,
    CounterexamplePayload,
    CounterexampleScope,
    DependencyCase,
    DependencyCounterexamplePayload,
    EnvironmentFact,
    LearnedConstraint,
    ReproductionCommand,
    RequestTraceCounterexamplePayload,
    ResourceCounterexamplePayload,
    TensorCounterexamplePayload,
    TopologyCounterexamplePayload,
    canonical_hash,
)

from .models import (
    AdversarialCase,
    BenchmarkAuditCase,
    RedTeamSurface,
    RegressionCase,
    RegressionCorpus,
    ResourceAdversarialCase,
    ScheduleAdversarialCase,
    TargetDescriptor,
    TensorAdversarialCase,
    TopologyAdversarialCase,
    ViolationObservation,
)


def _scope(surface: RedTeamSurface) -> CounterexampleScope:
    if surface is RedTeamSurface.TOPOLOGY:
        return CounterexampleScope.HARDWARE
    if surface is RedTeamSurface.BENCHMARK:
        return CounterexampleScope.UNIVERSAL_PRECONDITION
    return CounterexampleScope.TRANSFORMATION_FAMILY


def _payload(case: AdversarialCase) -> CounterexamplePayload:
    if isinstance(case, TensorAdversarialCase):
        return TensorCounterexamplePayload(input=case.input)
    if isinstance(case, ScheduleAdversarialCase):
        return RequestTraceCounterexamplePayload(events=case.events)
    if isinstance(case, TopologyAdversarialCase):
        return TopologyCounterexamplePayload(topology=case.topology)
    if isinstance(case, ResourceAdversarialCase):
        return ResourceCounterexamplePayload(resource=case.resource)
    if isinstance(case, BenchmarkAuditCase):
        return DependencyCounterexamplePayload(
            dependency=DependencyCase(
                package="benchmark-harness",
                version=case.issue_code,
            )
        )
    raise TypeError(f"unsupported red-team case {type(case)!r}")


def to_counterexample(
    *,
    target: TargetDescriptor,
    case: AdversarialCase,
    observation: ViolationObservation,
    seed: int,
) -> tuple[Counterexample, LearnedConstraint]:
    identity = canonical_hash(
        {
            "candidate_id": target.candidate_id,
            "case": case.model_dump(mode="json"),
            "observation": observation.model_dump(mode="json"),
            "seed": seed,
        }
    )
    counterexample_id = f"cex-{identity[:24]}"
    counterexample = Counterexample(
        counterexample_id=counterexample_id,
        candidate_id=target.candidate_id,
        transformation_id=target.transformation_id,
        violated_contract=observation.violated_contract,
        scope=_scope(observation.surface),
        payload=_payload(case),
        reproduction=ReproductionCommand(
            executable="python",
            arguments=(
                "-m",
                "sloforge.redteam.demo",
                "--seed",
                str(seed),
                "--candidate",
                target.candidate_id,
            ),
            timeout_seconds=30,
            seed=seed,
        ),
        environment=(
            EnvironmentFact(name="network", value="disabled"),
            EnvironmentFact(name="execution", value="deterministic-local-fixture"),
            EnvironmentFact(name="case_id", value=case.case_id),
        ),
        expected=BehaviorObservation(description=observation.expected_behavior),
        observed=BehaviorObservation(description=observation.observed_behavior),
        minimized=True,
    )
    constraint_scope = cast(
        Literal["candidate", "family", "hardware", "dependency", "universal_precondition"],
        {
            CounterexampleScope.CANDIDATE: "candidate",
            CounterexampleScope.TRANSFORMATION_FAMILY: "family",
            CounterexampleScope.HARDWARE: "hardware",
            CounterexampleScope.DEPENDENCY: "dependency",
            CounterexampleScope.UNIVERSAL_PRECONDITION: "universal_precondition",
        }[counterexample.scope],
    )
    constraint = LearnedConstraint(
        constraint_id=f"constraint-{canonical_hash({'cex': counterexample_id, 'expression': observation.learned_precondition})[:24]}",
        expression=observation.learned_precondition,
        scope=constraint_scope,
        counterexample_ids=(counterexample_id,),
    )
    return counterexample, constraint


def build_regression_corpus(
    *,
    seed: int,
    findings: tuple[tuple[AdversarialCase, Counterexample, LearnedConstraint], ...],
) -> RegressionCorpus:
    cases = tuple(
        RegressionCase(
            regression_id=f"regression-{canonical_hash({'case': case.case_id, 'cex': counterexample.counterexample_id})[:24]}",
            candidate_id=counterexample.candidate_id,
            violated_contract=counterexample.violated_contract,
            case=case,
            counterexample_id=counterexample.counterexample_id,
            seed=seed,
        )
        for case, counterexample, _constraint in findings
    )
    unique_constraints = {
        constraint.constraint_id: constraint for _case, _counterexample, constraint in findings
    }
    constraints = tuple(unique_constraints[key] for key in sorted(unique_constraints))
    digest = canonical_hash(
        {
            "schema_version": "1.0.0",
            "seed": seed,
            "cases": [case.model_dump(mode="json") for case in cases],
            "constraints": [constraint.model_dump(mode="json") for constraint in constraints],
        }
    )
    return RegressionCorpus(
        seed=seed,
        cases=cases,
        constraints=constraints,
        corpus_digest=digest,
    )


__all__ = ["build_regression_corpus", "to_counterexample"]
