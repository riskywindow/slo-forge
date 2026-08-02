"""Real deterministic cancellation-race rejection fixture for the CEGIS loop."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from sloforge.genesis.ir import (
    ArtifactDigest,
    Counterexample,
    CounterexampleScope,
    LearnedConstraint,
    RequestEventCase,
    TransformationFamily,
)
from sloforge.genesis.search import CandidateDesign, MutationChoice, ParameterValue

from .cegis import CegisRunner
from .models import (
    CegisConfiguration,
    CegisRunResult,
    GeneralizedConstraint,
    ProtocolWitness,
    VerificationFailure,
    VerificationOutcome,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _design(identifier: str, *, safe: bool, upside: float, seed: int) -> CandidateDesign:
    transformation_id = f"deadline-batch-{identifier}"
    mutation = MutationChoice(
        transformation_id=transformation_id,
        family=TransformationFamily.BATCHING,
        regions=("request", "serving"),
        parameters=(
            ParameterValue(key="cancel_check_before_emit", value="true" if safe else "false"),
            ParameterValue(key="queue_policy", value="deadline_bucket"),
        ),
        expected_upside=upside,
        invalidity_risk=0.05,
        feature_delta=(upside, 1.0 if safe else 0.0, 1.0),
    )
    digest = _digest(f"{identifier}:{safe}:{upside}:{seed}")
    return CandidateDesign(
        candidate_id=f"candidate-{identifier}-{digest[:12]}",
        seed=seed,
        genome_hash=ArtifactDigest(value=digest),
        parent_candidate_ids=(),
        mutations=(mutation,),
        feature_vector=mutation.feature_delta,
        proposal_engine="fixture",
    )


def cancellation_fixture_candidates(seed: int) -> tuple[CandidateDesign, ...]:
    """Fast invalid, same-family repeat, then corrected cross-layer policy."""

    return (
        _design("fast", safe=False, upside=0.21, seed=seed),
        _design("repeat", safe=False, upside=0.19, seed=seed + 1),
        _design("corrected", safe=True, upside=0.14, seed=seed + 2),
    )


def _initial_witness() -> ProtocolWitness:
    actions = ("admit", "schedule", "prefill", "decode", "cancel", "emit")
    return ProtocolWitness(
        events=tuple(
            RequestEventCase(
                at_step=index,
                request_id="request-a",
                action=action,  # type: ignore[arg-type]
                worker_id="worker-0",
            )
            for index, action in enumerate(actions)
        )
    )


class CancellationPolicyVerifier:
    """Explicit protocol simulator independent from mutation performance scores."""

    def verify(
        self,
        candidate: CandidateDesign,
        witness: ProtocolWitness | None,
        *,
        seed: int,
    ) -> VerificationOutcome:
        trace = witness or _initial_witness()
        check_before_emit = any(
            mutation.parameter("cancel_check_before_emit") == "true"
            for mutation in candidate.mutations
            if mutation.family is TransformationFamily.BATCHING
        )
        admitted: set[str] = set()
        cancelled: set[str] = set()
        violation = False
        for event in trace.events:
            if event.action == "admit":
                admitted.add(event.request_id)
            elif event.action == "cancel" and event.request_id in admitted:
                cancelled.add(event.request_id)
            elif (
                event.action == "emit"
                and event.request_id in admitted
                and event.request_id in cancelled
            ):
                if check_before_emit:
                    continue
                violation = True
                break
        evidence = _digest(
            json.dumps(
                {
                    "candidate": candidate.candidate_id,
                    "check_before_emit": check_before_emit,
                    "seed": seed,
                    "trace": [event.model_dump(mode="json") for event in trace.events],
                    "violation": violation,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        if not violation:
            return VerificationOutcome(passed=True, evidence_id=f"evidence-{evidence[:24]}")
        transformation_id = next(
            mutation.transformation_id
            for mutation in candidate.mutations
            if mutation.family is TransformationFamily.BATCHING
        )
        return VerificationOutcome(
            passed=False,
            evidence_id=f"evidence-{evidence[:24]}",
            failure=VerificationFailure(
                violated_contract="no committed token is emitted after request cancellation",
                scope=CounterexampleScope.TRANSFORMATION_FAMILY,
                transformation_id=transformation_id,
                expected_behavior="cancelled request emits no subsequent committed token",
                observed_behavior="deadline batch emitted a committed token after cancellation",
                witness=trace,
            ),
        )

    def generalize(
        self,
        candidate: CandidateDesign,
        failure: VerificationFailure,
        counterexample: Counterexample,
    ) -> GeneralizedConstraint:
        identity = _digest(
            f"{failure.violated_contract}:{TransformationFamily.BATCHING.value}:cancel-before-emit"
        )
        learned = LearnedConstraint(
            constraint_id=f"constraint-{identity[:24]}",
            expression="batching.cancel_check_before_emit == true",
            scope="family",
            counterexample_ids=(counterexample.counterexample_id,),
        )
        return GeneralizedConstraint(
            learned=learned,
            family=TransformationFamily.BATCHING,
            parameter_key="cancel_check_before_emit",
            required_value="true",
            rationale=(
                "deadline batching must re-check cancellation immediately before token commitment"
            ),
        )


def run_cancellation_cegis(output_directory: Path, *, seed: int) -> CegisRunResult:
    runner = CegisRunner(
        CegisConfiguration(
            seed=seed,
            maximum_candidates=3,
            maximum_minimization_evaluations=64,
        ),
        CancellationPolicyVerifier(),
        output_directory=output_directory,
    )
    return runner.run(cancellation_fixture_candidates(seed))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--seed", type=int, required=True)
    arguments = parser.parse_args()
    candidate = next(
        (
            item
            for item in cancellation_fixture_candidates(arguments.seed)
            if item.candidate_id == arguments.candidate
        ),
        None,
    )
    if candidate is None:
        raise SystemExit(f"unknown fixture candidate: {arguments.candidate}")
    outcome = CancellationPolicyVerifier().verify(candidate, None, seed=arguments.seed)
    print(outcome.model_dump_json())
    return 0 if outcome.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
