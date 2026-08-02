from __future__ import annotations

import json
from pathlib import Path

import pytest

from sloforge.genesis.capsule.builder import _validate_transformation_chain
from sloforge.genesis.compiler import initialize_genesis_run
from sloforge.genesis.frontend import inspect_reference_package
from sloforge.genesis.ir import (
    CandidateSuccessState,
    GenomeNodeMetadata,
    RequestTraceCounterexamplePayload,
    canonical_hash,
    load_candidate,
    load_counterexample,
    load_inference_genome,
    load_transformation,
    write_canonical,
)
from sloforge.genesis.policy_dsl import execute_bytecode
from sloforge.genesis.runtime import load_generated_runtime
from sloforge.genesis.synthesis import (
    cancellation_fixture_candidates,
    compiled_candidate_policy,
    synthesize_local_run,
)
from sloforge.genesis.synthesis.lowering import lower_candidate

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
    transformations = [
        load_transformation(path)
        for path in (accepted_directory / "transformations").glob("*.json")
    ]
    transformation = next(
        item
        for item in transformations
        if item.source_pattern.structural_constraints
        == (f"source_genome_sha256 == {result.baseline_genome_hash}",)
    )
    assert (
        f"source_genome_sha256 == {result.baseline_genome_hash}"
        in transformation.source_pattern.structural_constraints
    )
    assert any(
        f"target_genome_sha256 == {accepted.genome_hash.value}"
        in item.target_pattern.structural_constraints
        for item in transformations
    )
    assert transformation.verification_obligations
    assert {state.layout.value for state in genome.state.states} == {"paged"}
    runtime_config = json.loads(
        (accepted_directory / "generated_runtime/runtime_config.json").read_text(encoding="utf-8")
    )
    baseline_runtime_config = json.loads(
        (run.output_directory / "generated_runtime/runtime_config.json").read_text(
            encoding="utf-8"
        )
    )
    deployment_manifest = json.loads(
        (accepted_directory / "generated_runtime/deployment_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert runtime_config["runtime_id"] != baseline_runtime_config["runtime_id"]
    assert deployment_manifest["runtime_id"] == runtime_config["runtime_id"]
    assert deployment_manifest["candidate_id"] == accepted.candidate_id
    assert deployment_manifest["genome_hash"] == accepted.genome_hash.value
    assert runtime_config["state_allocator"] == {
        "layout": "paged",
        "page_bytes": 64,
        "maximum_bytes_per_request": 73,
        "maximum_total_bytes": 4096,
    }
    runtime = load_generated_runtime(
        accepted_directory / "generated_runtime/runtime_config.json",
        seed=73129,
        allow_untrusted_in_process=True,
    )
    runtime.start()
    try:
        events = list(
            runtime.submit_text(
                request_id="paged-state-proof",
                text="hybrid",
                maximum_new_tokens=2,
                seed=17,
                timeout_seconds=3.0,
            ).events(3.0)
        )
        assert events[-1].kind.value == "completed"
    finally:
        runtime.shutdown()
    assert runtime.health()["state_allocator_layout"] == "paged"
    assert runtime.metrics()["state_pages_peak"] == 2
    assert runtime.metrics()["state_reserved_bytes_peak"] == 128
    runtime_evidence = json.loads(
        (accepted_directory / "evidence/runtime-differential-result.json").read_text(
            encoding="utf-8"
        )
    )
    assert runtime_evidence["candidate_id"] == accepted.candidate_id
    assert runtime_evidence["candidate_genome_hash"] == accepted.genome_hash.value
    assert runtime_evidence["corpus_role"] == "final_evaluation"
    assert runtime_evidence["runtime_seed"] == 73129
    assert runtime_evidence["candidate_seed"] == accepted.seed
    assert runtime_evidence["passed"] is True
    assert runtime_evidence["state_allocator"] == runtime_config["state_allocator"]
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
    property_evidence = json.loads(
        (accepted_directory / "evidence/property-result.json").read_text(encoding="utf-8")
    )
    assert property_evidence["result"] == "pass"
    assert property_evidence["states_checked"] == 66_066
    assert property_evidence["counterexample"] is None
    property_event = next(
        event for event in accepted.lifecycle if event.to_state.value == "PROPERTY_TESTED"
    )
    assert property_event.evidence
    simulation_evidence = json.loads(
        (accepted_directory / "evidence/simulation-result.json").read_text(encoding="utf-8")
    )
    assert simulation_evidence["result"] == "pass"
    assert simulation_evidence["comparison_permitted"] is False
    assert simulation_evidence["candidate_id"] == accepted.candidate_id
    assert simulation_evidence["candidate_genome_hash"] == accepted.genome_hash.value
    assert (
        simulation_evidence["policy_bytecode_sha256"] == runtime_evidence["policy_bytecode_sha256"]
    )
    assert simulation_evidence["queue_policy"] == "deadline_cancel_batch"

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
    rejected_evidence = json.loads(
        (
            run.output_directory
            / "candidates"
            / result.rejected_candidate_ids[0]
            / "evidence/runtime-differential-result.json"
        ).read_text(encoding="utf-8")
    )
    assert rejected_evidence["corpus_role"] == "search"


def test_multi_transformation_lowering_preserves_derivation_chain(tmp_path: Path) -> None:
    baseline = load_inference_genome(ROOT / "tests/fixtures/genesis/inference-genome-v1.json")

    def allow_policy_lowering(node: GenomeNodeMetadata) -> GenomeNodeMetadata:
        return node.model_copy(
            update={
                "legal_rewrite_rules": (
                    *node.legal_rewrite_rules,
                    "genesis.rules/baseline-exact-v1",
                )
            }
        )

    baseline = baseline.model_copy(
        update={
            "request": baseline.request.model_copy(
                update={"node": allow_policy_lowering(baseline.request.node)}
            ),
            "serving": baseline.serving.model_copy(
                update={"node": allow_policy_lowering(baseline.serving.node)}
            ),
            "state": baseline.state.model_copy(
                update={
                    "node": allow_policy_lowering(baseline.state.node),
                    "states": tuple(
                        state.model_copy(update={"node": allow_policy_lowering(state.node)})
                        for state in baseline.state.states
                    ),
                }
            ),
        }
    )
    design = cancellation_fixture_candidates(73129)[2]

    lowered = lower_candidate(baseline, design)

    assert len(lowered.transformations) == 2
    first, second_transformation = lowered.transformations
    first_delta = first.extensions.root["sloforge.dev/applied-delta"]
    second_delta = second_transformation.extensions.root["sloforge.dev/applied-delta"]
    assert isinstance(first_delta, dict)
    assert isinstance(second_delta, dict)
    assert first_delta["source_genome_hash"] == canonical_hash(baseline)
    assert first_delta["target_genome_hash"] == second_delta["source_genome_hash"]
    assert second_delta["target_genome_hash"] == canonical_hash(lowered.genome)
    assert first.parent_transformations == ()
    assert second_transformation.parent_transformations == (first.transformation_id,)
    synthesis_extension = lowered.genome.extensions.root["sloforge.dev/synthesis-candidate"]
    assert isinstance(synthesis_extension, dict)
    assert synthesis_extension["parent_genome_hash"] == canonical_hash(baseline)
    assert synthesis_extension["immediate_parent_genome_hash"] == second_delta["source_genome_hash"]
    assert synthesis_extension["transformation_ids"] == [
        first.transformation_id,
        second_transformation.transformation_id,
    ]

    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    write_canonical(first, first_path)
    write_canonical(second_transformation, second_path)
    ordered = _validate_transformation_chain(
        [second_path, first_path],
        transformation_ids=(
            first.transformation_id,
            second_transformation.transformation_id,
        ),
        baseline_genome_hash=canonical_hash(baseline),
        candidate_genome_hash=canonical_hash(lowered.genome),
        trusted_transformations=lowered.transformations,
    )
    assert [transformation.transformation_id for _path, transformation in ordered] == [
        first.transformation_id,
        second_transformation.transformation_id,
    ]

    broken_path = tmp_path / "broken-second.json"
    write_canonical(
        second_transformation.model_copy(update={"parent_transformations": ()}),
        broken_path,
    )
    with pytest.raises(ValueError, match="trusted lowering derivation"):
        _validate_transformation_chain(
            [first_path, broken_path],
            transformation_ids=(
                first.transformation_id,
                second_transformation.transformation_id,
            ),
            baseline_genome_hash=canonical_hash(baseline),
            candidate_genome_hash=canonical_hash(lowered.genome),
            trusted_transformations=lowered.transformations,
        )


def test_trusted_lowering_rejects_ambiguous_duplicate_family() -> None:
    baseline = load_inference_genome(ROOT / "tests/fixtures/genesis/inference-genome-v1.json")

    def allow_policy_lowering(node: GenomeNodeMetadata) -> GenomeNodeMetadata:
        return node.model_copy(
            update={
                "legal_rewrite_rules": (
                    *node.legal_rewrite_rules,
                    "genesis.rules/baseline-exact-v1",
                )
            }
        )

    baseline = baseline.model_copy(
        update={
            "request": baseline.request.model_copy(
                update={"node": allow_policy_lowering(baseline.request.node)}
            ),
            "serving": baseline.serving.model_copy(
                update={"node": allow_policy_lowering(baseline.serving.node)}
            ),
        }
    )
    design = cancellation_fixture_candidates(73129)[0]
    duplicate = design.mutations[0].model_copy(
        update={"transformation_id": "deadline-batch-second"}
    )
    ambiguous = design.model_copy(update={"mutations": (*design.mutations, duplicate)})

    with pytest.raises(ValueError, match="at most one executable mutation"):
        lower_candidate(baseline, ambiguous)
    with pytest.raises(ValueError, match="exactly one batching mutation"):
        compiled_candidate_policy(ambiguous)
