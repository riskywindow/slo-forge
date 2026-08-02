from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from sloforge.genesis.ir import load_counterexample, write_canonical
from sloforge.genesis.synthesis import ConstraintStore
from sloforge.genesis.synthesis.cegis import _counterexample, minimize_protocol_failure
from sloforge.genesis.synthesis.fixture import (
    CancellationPolicyVerifier,
    cancellation_fixture_candidates,
    run_cancellation_cegis,
)
from sloforge.genesis.synthesis.local import bounded_candidate_policy_property_document
from sloforge.genesis.synthesis.models import ProtocolWitness, VerificationOutcome

ROOT = Path(__file__).resolve().parents[2]


def test_cegis_rejects_minimizes_learns_suppresses_and_corrects(tmp_path: Path) -> None:
    result = run_cancellation_cegis(tmp_path, seed=73129)
    candidates = cancellation_fixture_candidates(73129)

    assert result.rejected_candidate_ids == (candidates[0].candidate_id,)
    assert result.suppressed_candidate_ids == (candidates[1].candidate_id,)
    assert result.accepted_candidate_id == candidates[2].candidate_id
    assert candidates[0].mutations[0].expected_upside > candidates[2].mutations[0].expected_upside
    assert candidates[2].cross_layer
    assert result.verifier_invocations == result.minimization_evaluations + 2

    counterexamples = sorted((tmp_path / "counterexamples").glob("*.json"))
    assert len(counterexamples) == 2
    original = next(
        load_counterexample(path)
        for path in counterexamples
        if not json.loads(path.read_text())["minimized"]
    )
    minimized = next(
        load_counterexample(path)
        for path in counterexamples
        if json.loads(path.read_text())["minimized"]
    )
    assert len(original.payload.events) == 6  # type: ignore[union-attr]
    assert [event.action for event in minimized.payload.events] == [  # type: ignore[union-attr]
        "admit",
        "cancel",
        "emit",
    ]
    assert minimized.parent_counterexample_id == original.counterexample_id
    assert minimized.counterexample_id in result.counterexample_ids

    constraints = ConstraintStore(tmp_path / "constraints.json")
    assert len(constraints.constraints) == 1
    assert constraints.rejecting_constraint(candidates[1]) is not None
    assert constraints.rejecting_constraint(candidates[2]) is None
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [event["sequence"] for event in events] == list(range(len(events)))
    assert "candidate_suppressed" in {event["event_type"] for event in events}


def test_policy_property_oracle_rejects_missing_cancellation_input() -> None:
    unsafe, _repeated, corrected = cancellation_fixture_candidates(73129)

    unsafe_result = bounded_candidate_policy_property_document(unsafe, seed=73129)
    corrected_result = bounded_candidate_policy_property_document(corrected, seed=73129)

    assert unsafe_result["result"] == "fail"
    assert unsafe_result["counterexample"] == {
        "assignment": {},
        "observed_output": None,
        "violations": ["required boolean cancellation_pending input is absent"],
    }
    assert corrected_result["result"] == "pass"


def test_minimized_trace_is_one_event_minimal_for_same_contract(tmp_path: Path) -> None:
    result = run_cancellation_cegis(tmp_path, seed=17)
    candidate = cancellation_fixture_candidates(17)[0]
    path = tmp_path / "counterexamples" / f"{result.counterexample_ids[0]}.json"
    counterexample = load_counterexample(path)
    events = counterexample.payload.events  # type: ignore[union-attr]
    verifier = CancellationPolicyVerifier()

    for removed in range(len(events)):
        remaining = events[:removed] + events[removed + 1 :]
        witness = ProtocolWitness(
            events=tuple(
                event.model_copy(update={"at_step": index}) for index, event in enumerate(remaining)
            )
        )
        assert verifier.verify(candidate, witness, seed=3).passed


def test_minimization_preserves_the_original_verification_seed() -> None:
    expected_seed = 101
    candidate = cancellation_fixture_candidates(17)[0]
    underlying = CancellationPolicyVerifier()

    class SeedPinnedVerifier:
        def verify(
            self,
            candidate: object,
            witness: ProtocolWitness | None,
            *,
            seed: int,
        ) -> VerificationOutcome:
            if seed != expected_seed:
                return VerificationOutcome(passed=True, evidence_id="wrong-seed")
            return underlying.verify(candidate, witness, seed=seed)  # type: ignore[arg-type]

    initial = underlying.verify(candidate, None, seed=expected_seed)
    assert initial.failure is not None
    minimized = minimize_protocol_failure(
        candidate,
        initial.failure,
        SeedPinnedVerifier(),  # type: ignore[arg-type]
        seed=expected_seed,
        maximum_evaluations=64,
    )

    reproduced = underlying.verify(candidate, minimized.witness, seed=expected_seed)
    assert not reproduced.passed
    assert reproduced.failure is not None
    assert reproduced.failure.violated_contract == initial.failure.violated_contract


def test_counterexample_reproduction_command_executes_real_verifier(tmp_path: Path) -> None:
    result = run_cancellation_cegis(tmp_path, seed=31)
    counterexample = load_counterexample(
        tmp_path / "counterexamples" / f"{result.counterexample_ids[0]}.json"
    )
    command = [counterexample.reproduction.executable, *counterexample.reproduction.arguments]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=counterexample.reproduction.timeout_seconds,
        env={**os.environ, "PYTHONPATH": str(ROOT / "python")},
    )

    assert completed.returncode == 1
    outcome = json.loads(completed.stdout)
    assert outcome["passed"] is False
    assert outcome["failure"]["violated_contract"] == counterexample.violated_contract


def test_counterexample_replay_recovers_non_first_population_candidate(
    tmp_path: Path,
) -> None:
    population_seed = 41
    verification_seed = 997
    candidate = cancellation_fixture_candidates(population_seed)[1]
    verifier = CancellationPolicyVerifier()
    outcome = verifier.verify(candidate, None, seed=verification_seed)
    assert not outcome.passed
    assert outcome.failure is not None
    counterexample = _counterexample(
        candidate,
        outcome.failure,
        outcome.failure.witness,
        verification_seed=verification_seed,
        population_seed=population_seed,
        minimized=False,
        parent_id=None,
        counterexample_directory=tmp_path,
    )
    counterexample_path = tmp_path / f"{counterexample.counterexample_id}.json"
    write_canonical(counterexample, counterexample_path)

    completed = subprocess.run(
        [counterexample.reproduction.executable, *counterexample.reproduction.arguments],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=counterexample.reproduction.timeout_seconds,
        env={**os.environ, "PYTHONPATH": str(ROOT / "python")},
    )

    assert completed.returncode == 1
    replayed = json.loads(completed.stdout)
    assert replayed["passed"] is False
    assert replayed["evidence_id"] == outcome.evidence_id
    arguments = counterexample.reproduction.arguments
    assert arguments[arguments.index("--seed") + 1] == str(population_seed)
    assert arguments[arguments.index("--verification-seed") + 1] == str(verification_seed)


def test_cegis_artifacts_are_deterministic_across_runs(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    left = run_cancellation_cegis(first, seed=99)
    right = run_cancellation_cegis(second, seed=99)

    assert left.model_copy(update={"events_path": "events"}) == right.model_copy(
        update={"events_path": "events"}
    )
    assert (first / "events.jsonl").read_bytes() == (second / "events.jsonl").read_bytes()
    assert (first / "constraints.json").read_bytes() == (second / "constraints.json").read_bytes()
