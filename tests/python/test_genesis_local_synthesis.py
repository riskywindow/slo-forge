from __future__ import annotations

import json
from pathlib import Path

from sloforge.genesis.compiler import initialize_genesis_run
from sloforge.genesis.frontend import inspect_reference_package
from sloforge.genesis.ir import (
    CandidateSuccessState,
    RequestTraceCounterexamplePayload,
    canonical_hash,
    load_candidate,
    load_counterexample,
    load_inference_genome,
    load_transformation,
)
from sloforge.genesis.policy_dsl import execute_bytecode
from sloforge.genesis.synthesis import (
    cancellation_fixture_candidates,
    compiled_candidate_policy,
    synthesize_local_run,
)

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
    assert accepted.state is CandidateSuccessState.SIMULATED
    assert accepted.genome_hash.value == result.accepted_genome_hash
    policy = (
        run.output_directory / "candidates" / result.accepted_candidate_id / "policy.slo"
    ).read_text(encoding="utf-8")
    assert "cancellation_pending" in policy
    accepted_directory = run.output_directory / "candidates" / result.accepted_candidate_id
    genome = load_inference_genome(accepted_directory / "inference_genome.json")
    assert canonical_hash(genome) == accepted.genome_hash.value
    transformation = load_transformation(
        next((accepted_directory / "transformations").glob("*.json"))
    )
    assert (
        f"source_genome_sha256 == {result.baseline_genome_hash}"
        in transformation.source_pattern.structural_constraints
    )
    assert (
        f"target_genome_sha256 == {accepted.genome_hash.value}"
        in transformation.target_pattern.structural_constraints
    )
    assert transformation.verification_obligations
    runtime_evidence = json.loads(
        (accepted_directory / "evidence/runtime-differential-result.json").read_text(
            encoding="utf-8"
        )
    )
    assert runtime_evidence["candidate_id"] == accepted.candidate_id
    assert runtime_evidence["candidate_genome_hash"] == accepted.genome_hash.value
    assert runtime_evidence["passed"] is True
    assert (
        runtime_evidence["runtime_artifact_hashes"]["policy.bytecode.json"]
        == (runtime_evidence["policy_bytecode_sha256"])
    )
    modelcheck_evidence = json.loads(
        (accepted_directory / "evidence/modelcheck-result.json").read_text(encoding="utf-8")
    )
    assert modelcheck_evidence["result"] == "pass"
    assert modelcheck_evidence["state_count"] > 0
    assert modelcheck_evidence["transition_count"] > 0
    assert modelcheck_evidence["universal_proof"] is False
    simulation_evidence = json.loads(
        (accepted_directory / "evidence/simulation-result.json").read_text(encoding="utf-8")
    )
    assert simulation_evidence["result"] == "pass"
    assert simulation_evidence["comparison_permitted"] is False

    unsafe, _repeat, corrected = cancellation_fixture_candidates(73129)
    _source, unsafe_bytecode, _payload = compiled_candidate_policy(unsafe)
    _source, corrected_bytecode, _payload = compiled_candidate_policy(corrected)
    assert execute_bytecode(unsafe_bytecode, {"queue_length": 1, "slo_slack_ms": 100}) > 0
    assert (
        execute_bytecode(
            corrected_bytecode,
            {"queue_length": 1, "slo_slack_ms": 100, "cancellation_pending": True},
        )
        == 0
    )

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
