from __future__ import annotations

import json
from pathlib import Path

from sloforge.genesis.compiler import initialize_genesis_run
from sloforge.genesis.frontend import inspect_reference_package
from sloforge.genesis.ir import (
    CandidateSuccessState,
    RequestTraceCounterexamplePayload,
    load_candidate,
    load_counterexample,
)
from sloforge.genesis.synthesis import synthesize_local_run

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "models/reference_tasks/hybrid_decoder"


def test_local_synthesis_rejects_minimizes_learns_and_corrects(tmp_path: Path) -> None:
    inspection = inspect_reference_package(PACKAGE)
    run = initialize_genesis_run(PACKAGE, inspection, tmp_path / "run", seed=73129)

    result = synthesize_local_run(run.output_directory, seed=73129)

    assert result.runtime_differential_passed
    assert result.sandbox_termination == "success"
    assert result.accepted_candidate_id is not None
    assert result.cross_layer_accepted
    assert len(result.rejected_candidate_ids) == 1
    assert len(result.suppressed_candidate_ids) == 1
    assert len(result.counterexample_ids) == 1
    assert len(result.constraint_ids) == 1

    accepted = load_candidate(
        run.output_directory / "candidates" / result.accepted_candidate_id / "candidate.json"
    )
    assert accepted.state is CandidateSuccessState.PROPERTY_TESTED
    assert accepted.genome_hash.value == result.accepted_genome_hash
    policy = (
        run.output_directory / "candidates" / result.accepted_candidate_id / "policy.slo"
    ).read_text(encoding="utf-8")
    assert "cancellation_pending" in policy

    counterexample_path = next(
        (run.output_directory / "synthesis/cegis/counterexamples").glob(
            f"{result.counterexample_ids[0]}.json"
        )
    )
    counterexample = load_counterexample(counterexample_path)
    assert counterexample.minimized
    assert isinstance(counterexample.payload, RequestTraceCounterexamplePayload)
    assert len(counterexample.payload.events) == 3
    constraints = json.loads(
        (run.output_directory / "synthesis/cegis/constraints.json").read_text(encoding="utf-8")
    )
    assert constraints["constraints"][0]["parameter_key"] == "cancel_check_before_emit"
