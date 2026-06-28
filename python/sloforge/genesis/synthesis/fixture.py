"""Real deterministic cancellation-race rejection fixture for the CEGIS loop."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

from sloforge.genesis.ir import (
    ArtifactDigest,
    Counterexample,
    CounterexampleScope,
    LearnedConstraint,
    RequestEventCase,
    RequestTraceCounterexamplePayload,
    TransformationFamily,
    load_counterexample,
)
from sloforge.genesis.policy_dsl import (
    BytecodeProgram,
    check_policy,
    compile_policy,
    execute_bytecode,
    format_policy,
    parse_policy,
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


_UNSAFE_POLICY = """\
policy deadline_batch
input queue_length int 0 32
input slo_slack_ms int 0 1000
output int 0 4
limit 64
return (clamp (if (lt slo_slack_ms 20) 1 (min queue_length 4)) 0 4)
"""

_CORRECTED_POLICY = """\
policy deadline_cancel_batch
input queue_length int 0 32
input slo_slack_ms int 0 1000
input cancellation_pending bool false true
output int 0 4
limit 64
return (clamp (if cancellation_pending 0 (if (lt slo_slack_ms 20) 1 (min queue_length 4))) 0 4)
"""


def compiled_candidate_policy(candidate: CandidateDesign) -> tuple[str, BytecodeProgram, bytes]:
    """Compile the exact restricted policy represented by a candidate design."""

    safe = any(
        mutation.parameter("cancel_check_before_emit") == "true" for mutation in candidate.mutations
    )
    program = parse_policy(_CORRECTED_POLICY if safe else _UNSAFE_POLICY)
    check_policy(program)
    bytecode = compile_policy(program)
    payload = json.dumps(
        dataclasses.asdict(bytecode),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return format_policy(program), bytecode, payload


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
        _source, bytecode, policy_payload = compiled_candidate_policy(candidate)
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
                available: dict[str, int | bool] = {
                    "queue_length": 1,
                    "slo_slack_ms": 100,
                    "cancellation_pending": event.request_id in cancelled,
                }
                names = {item.name for item in bytecode.inputs}
                scheduling_decision = execute_bytecode(
                    bytecode, {name: available[name] for name in names}
                )
                if type(scheduling_decision) is int and scheduling_decision == 0:
                    continue
                violation = True
                break
        evidence = _digest(
            json.dumps(
                {
                    "candidate": candidate.candidate_id,
                    "policy_bytecode_sha256": hashlib.sha256(policy_payload).hexdigest(),
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
    parser.add_argument("--verification-seed", type=int)
    parser.add_argument("--counterexample", type=Path)
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
    witness: ProtocolWitness | None = None
    if arguments.counterexample is not None:
        counterexample = load_counterexample(arguments.counterexample)
        if counterexample.candidate_id != candidate.candidate_id:
            raise SystemExit("counterexample candidate identity does not match")
        if not isinstance(counterexample.payload, RequestTraceCounterexamplePayload):
            raise SystemExit("counterexample is not a request-trace witness")
        witness = ProtocolWitness(events=counterexample.payload.events)
    verification_seed = (
        arguments.seed if arguments.verification_seed is None else arguments.verification_seed
    )
    outcome = CancellationPolicyVerifier().verify(candidate, witness, seed=verification_seed)
    print(outcome.model_dump_json())
    return 0 if outcome.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
