"""Counterexample-guided candidate rejection, minimization, and correction."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Protocol

from sloforge.genesis.ir import (
    BehaviorObservation,
    Counterexample,
    EnvironmentFact,
    ReproductionCommand,
    RequestTraceCounterexamplePayload,
    canonical_json,
    write_canonical,
)
from sloforge.genesis.search import CandidateDesign

from .constraints import ConstraintStore
from .models import (
    CegisConfiguration,
    CegisEvent,
    CegisRunResult,
    GeneralizedConstraint,
    MinimizationResult,
    ProtocolWitness,
    VerificationFailure,
    VerificationOutcome,
)


class CegisVerifier(Protocol):
    def verify(
        self,
        candidate: CandidateDesign,
        witness: ProtocolWitness | None,
        *,
        seed: int,
    ) -> VerificationOutcome: ...

    def generalize(
        self,
        candidate: CandidateDesign,
        failure: VerificationFailure,
        counterexample: Counterexample,
    ) -> GeneralizedConstraint: ...


def _verification_seed(seed: int, candidate_id: str, ordinal: int) -> int:
    payload = f"{seed}\0{candidate_id}\0{ordinal}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _renumber(events: tuple[object, ...]) -> ProtocolWitness:
    from sloforge.genesis.ir import RequestEventCase

    return ProtocolWitness(
        events=tuple(
            RequestEventCase(
                at_step=index,
                request_id=event.request_id,
                action=event.action,
                worker_id=event.worker_id,
            )
            for index, event in enumerate(events)
            if isinstance(event, RequestEventCase)
        )
    )


def minimize_protocol_failure(
    candidate: CandidateDesign,
    failure: VerificationFailure,
    verifier: CegisVerifier,
    *,
    seed: int,
    maximum_evaluations: int,
) -> MinimizationResult:
    """Run deterministic delta debugging while preserving the same contract failure."""

    if maximum_evaluations <= 0:
        raise ValueError("maximum minimization evaluations must be positive")
    current = failure.witness
    evaluations = 0

    def preserves(witness: ProtocolWitness) -> bool:
        nonlocal evaluations
        if evaluations >= maximum_evaluations:
            return False
        outcome = verifier.verify(candidate, witness, seed=seed)
        evaluations += 1
        return (
            not outcome.passed
            and outcome.failure is not None
            and outcome.failure.violated_contract == failure.violated_contract
        )

    if not preserves(current):
        raise ValueError("minimizer could not reproduce the original verifier failure")

    granularity = 2
    while len(current.events) >= 2 and evaluations < maximum_evaluations:
        chunk_size = max(1, (len(current.events) + granularity - 1) // granularity)
        reduced = False
        for start in range(0, len(current.events), chunk_size):
            retained = current.events[:start] + current.events[start + chunk_size :]
            if not retained:
                continue
            proposal = _renumber(tuple(retained))
            if preserves(proposal):
                current = proposal
                granularity = max(2, granularity - 1)
                reduced = True
                break
        if reduced:
            continue
        if granularity >= len(current.events):
            break
        granularity = min(len(current.events), granularity * 2)
    return MinimizationResult(witness=current, evaluations=evaluations)


class _CegisEventStore:
    def __init__(self, path: Path, maximum_events: int) -> None:
        self.path = path
        self.maximum_events = maximum_events
        self.events: list[CegisEvent] = []
        if path.exists():
            raise FileExistsError(f"CEGIS event log already exists: {path}")

    def append(self, event: CegisEvent) -> None:
        if len(self.events) >= self.maximum_events:
            raise RuntimeError("bounded CEGIS event store is full")
        if event.sequence != len(self.events):
            raise ValueError("CEGIS event sequence must be contiguous")
        self.events.append(event)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                for item in self.events:
                    handle.write(canonical_json(item) + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def _counterexample(
    candidate: CandidateDesign,
    failure: VerificationFailure,
    witness: ProtocolWitness,
    *,
    verification_seed: int,
    population_seed: int,
    minimized: bool,
    parent_id: str | None,
    counterexample_directory: Path,
) -> Counterexample:
    identity = hashlib.sha256(
        canonical_json(
            {
                "candidate_id": candidate.candidate_id,
                "contract": failure.violated_contract,
                "events": [event.model_dump(mode="json") for event in witness.events],
                "minimized": minimized,
                "parent": parent_id,
                "verification_seed": verification_seed,
                "population_seed": population_seed,
            }
        )
    ).hexdigest()
    counterexample_id = f"counterexample-{identity[:24]}"
    return Counterexample(
        counterexample_id=counterexample_id,
        candidate_id=candidate.candidate_id,
        transformation_id=failure.transformation_id,
        violated_contract=failure.violated_contract,
        scope=failure.scope,
        payload=RequestTraceCounterexamplePayload(events=witness.events),
        reproduction=ReproductionCommand(
            executable="python",
            arguments=(
                "-m",
                "sloforge.genesis.synthesis.fixture",
                "--candidate",
                candidate.candidate_id,
                "--seed",
                str(population_seed),
                "--verification-seed",
                str(verification_seed),
                "--counterexample",
                str((counterexample_directory / f"{counterexample_id}.json").resolve()),
            ),
            timeout_seconds=30,
            seed=verification_seed,
        ),
        environment=(
            EnvironmentFact(name="execution_mode", value="deterministic_protocol_model"),
            EnvironmentFact(name="population_seed", value=str(population_seed)),
            EnvironmentFact(name="verification_seed", value=str(verification_seed)),
        ),
        expected=BehaviorObservation(description=failure.expected_behavior),
        observed=BehaviorObservation(description=failure.observed_behavior),
        minimized=minimized,
        parent_counterexample_id=parent_id,
    )


class CegisRunner:
    def __init__(
        self,
        configuration: CegisConfiguration,
        verifier: CegisVerifier,
        *,
        output_directory: Path,
    ) -> None:
        self.configuration = configuration
        self.verifier = verifier
        self.output_directory = output_directory
        self.constraints = ConstraintStore(output_directory / "constraints.json")
        self.events = _CegisEventStore(
            output_directory / "events.jsonl", configuration.maximum_events
        )

    def _event(
        self,
        event_type: str,
        candidate_id: str,
        reason: str,
        *,
        counterexample_id: str | None = None,
        constraint_id: str | None = None,
    ) -> None:
        self.events.append(
            CegisEvent(
                sequence=len(self.events.events),
                event_type=event_type,  # type: ignore[arg-type]
                candidate_id=candidate_id,
                reason=reason,
                counterexample_id=counterexample_id,
                constraint_id=constraint_id,
            )
        )

    def run(self, candidates: tuple[CandidateDesign, ...]) -> CegisRunResult:
        rejected: list[str] = []
        suppressed: list[str] = []
        counterexample_ids: list[str] = []
        constraint_ids: list[str] = []
        invocations = 0
        minimization_evaluations = 0
        accepted: str | None = None
        for ordinal, candidate in enumerate(candidates[: self.configuration.maximum_candidates]):
            constraint = self.constraints.rejecting_constraint(candidate)
            if constraint is not None:
                suppressed.append(candidate.candidate_id)
                self._event(
                    "candidate_suppressed",
                    candidate.candidate_id,
                    constraint.rationale,
                    constraint_id=constraint.learned.constraint_id,
                )
                continue
            seed = _verification_seed(self.configuration.seed, candidate.candidate_id, ordinal)
            outcome = self.verifier.verify(candidate, None, seed=seed)
            invocations += 1
            self._event(
                "candidate_verification",
                candidate.candidate_id,
                f"verifier evidence {outcome.evidence_id}",
            )
            if outcome.passed:
                accepted = candidate.candidate_id
                self._event(
                    "candidate_accepted",
                    candidate.candidate_id,
                    "independent verifier accepted candidate within its declared scope",
                )
                break
            failure = outcome.failure
            if failure is None:
                raise ValueError("failed verifier outcome omitted its failure")
            original = _counterexample(
                candidate,
                failure,
                failure.witness,
                verification_seed=seed,
                population_seed=self.configuration.seed,
                minimized=False,
                parent_id=None,
                counterexample_directory=self.output_directory / "counterexamples",
            )
            counterexample_directory = self.output_directory / "counterexamples"
            write_canonical(
                original, counterexample_directory / f"{original.counterexample_id}.json"
            )
            self._event(
                "counterexample_captured",
                candidate.candidate_id,
                failure.violated_contract,
                counterexample_id=original.counterexample_id,
            )
            minimized = minimize_protocol_failure(
                candidate,
                failure,
                self.verifier,
                seed=seed,
                maximum_evaluations=self.configuration.maximum_minimization_evaluations,
            )
            invocations += minimized.evaluations
            minimization_evaluations += minimized.evaluations
            compact = _counterexample(
                candidate,
                failure,
                minimized.witness,
                verification_seed=seed,
                population_seed=self.configuration.seed,
                minimized=True,
                parent_id=original.counterexample_id,
                counterexample_directory=counterexample_directory,
            )
            write_canonical(compact, counterexample_directory / f"{compact.counterexample_id}.json")
            counterexample_ids.append(compact.counterexample_id)
            self._event(
                "counterexample_minimized",
                candidate.candidate_id,
                f"reduced protocol trace from {len(failure.witness.events)} to {len(minimized.witness.events)} events",
                counterexample_id=compact.counterexample_id,
            )
            learned = self.verifier.generalize(candidate, failure, compact)
            relevant_families = {
                mutation.family
                for mutation in candidate.mutations
                if mutation.transformation_id == failure.transformation_id
            }
            if relevant_families != {learned.family}:
                raise ValueError("generalized constraint family does not match the failure")
            if learned.learned.counterexample_ids != (compact.counterexample_id,):
                raise ValueError("generalized constraint must cite the minimized counterexample")
            if not learned.rejects(candidate):
                raise ValueError("generalized constraint does not reject its failing candidate")
            self.constraints.add(learned)
            constraint_ids.append(learned.learned.constraint_id)
            self._event(
                "constraint_learned",
                candidate.candidate_id,
                learned.rationale,
                counterexample_id=compact.counterexample_id,
                constraint_id=learned.learned.constraint_id,
            )
            rejected.append(candidate.candidate_id)
            self._event(
                "candidate_rejected",
                candidate.candidate_id,
                failure.violated_contract,
                counterexample_id=compact.counterexample_id,
            )
        return CegisRunResult(
            accepted_candidate_id=accepted,
            rejected_candidate_ids=tuple(rejected),
            suppressed_candidate_ids=tuple(suppressed),
            counterexample_ids=tuple(counterexample_ids),
            constraint_ids=tuple(constraint_ids),
            verifier_invocations=invocations,
            minimization_evaluations=minimization_evaluations,
            events_path=str(self.events.path.resolve()),
        )
