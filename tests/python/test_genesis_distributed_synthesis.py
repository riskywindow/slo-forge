from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from sloforge.fabric.ir import canonical_hash, load_physical_execution_plan
from sloforge.genesis.distributed_synthesis import (
    INVALIDATED_FABRIC_FIELDS,
    REQUIRED_REVALIDATION_STAGES,
    CollectiveMutation,
    DistributedSynthesisError,
    DistributedSynthesisResult,
    RankPlacementMutation,
    compile_distributed_mutation,
)

ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = ROOT / "artifacts/fabric-demo/physical-plan.json"


def test_collective_mutation_revalidates_through_fabric_ir() -> None:
    source = load_physical_execution_plan(PLAN_PATH)
    operation = source.collectives.operations[0]
    mutation = CollectiveMutation(
        transformation_id="collective-tree-reorder",
        operation_id=operation.operation_id,
        algorithm="tree",
        transport=operation.transport,
        channel_count=4,
        rank_order=tuple(reversed(operation.rank_order)),
    )

    result = compile_distributed_mutation(source, mutation, seed=73129)
    changed = result.candidate_plan.collectives.operations[0]
    assert changed.algorithm == "tree"
    assert changed.rank_order == tuple(reversed(operation.rank_order))
    assert canonical_hash(result.candidate_plan) != canonical_hash(source)
    assert result.fabric_schema_validated
    assert not result.performance_evidence_valid
    assert not result.eligible_for_performance_comparison
    assert result.bounded_model_check_required
    assert result.evidence_state == "invalidated-pending-revalidation"
    assert result.required_verifier_stages == REQUIRED_REVALIDATION_STAGES
    assert result.candidate_plan_hash == canonical_hash(result.candidate_plan)
    assert result.source_plan_hash == canonical_hash(source)
    assert result.seed == 73129


def test_mutation_strips_stale_fabric_evidence_and_simulation_history() -> None:
    source = load_physical_execution_plan(PLAN_PATH)
    operation = source.collectives.operations[0]
    result = compile_distributed_mutation(
        source,
        CollectiveMutation(
            transformation_id="invalidate-source-evidence",
            operation_id=operation.operation_id,
            algorithm="tree",
            transport=operation.transport,
            channel_count=4,
            rank_order=tuple(reversed(operation.rank_order)),
        ),
        seed=73129,
    )

    assert source.evidence
    assert result.candidate_plan.evidence == ()
    assert all(entry.simulator_calls == 0 for entry in result.candidate_plan.optimizer_history)
    assert result.candidate_plan.bottleneck_prediction == (
        "invalidated-pending-distributed-revalidation"
    )
    assert result.candidate_plan.failure_exposure == ()
    assert result.candidate_plan.rejected_alternatives == ()
    assert result.candidate_plan.recovery_variants == ()
    assert all(
        interval["confidence"] == 0.0
        for interval in result.candidate_plan.predicted_metrics.model_dump(mode="json").values()
    )
    assert result.invalidated_fields == INVALIDATED_FABRIC_FIELDS
    assert result.invalidated_evidence_uris == tuple(
        sorted(reference.uri for reference in source.evidence)
    )
    assert {item.digest_sha256 for item in result.invalidated_evidence} == {
        reference.digest.value for reference in source.evidence
    }

    extension = result.candidate_plan.extensions.root["sloforge.dev/genesis-candidate"]
    assert isinstance(extension, dict)
    assert extension["evidence_state"] == "invalidated-pending-revalidation"
    assert extension["required_revalidation_stages"] == list(REQUIRED_REVALIDATION_STAGES)
    assert extension["source_plan_hash"] == canonical_hash(source)
    assert extension["seed"] == 73129


def test_collective_mutation_rejects_duplicate_rank_permutation_and_coerced_seed() -> None:
    source = load_physical_execution_plan(PLAN_PATH)
    operation = source.collectives.operations[0]
    duplicate = CollectiveMutation(
        transformation_id="duplicate-rank",
        operation_id=operation.operation_id,
        algorithm="tree",
        transport=operation.transport,
        channel_count=4,
        rank_order=(operation.rank_order[0], *operation.rank_order),
    )
    with pytest.raises(DistributedSynthesisError, match="must permute"):
        compile_distributed_mutation(source, duplicate, seed=1)
    with pytest.raises(DistributedSynthesisError, match="non-negative integer"):
        compile_distributed_mutation(source, duplicate, seed=True)  # type: ignore[arg-type]


def test_legacy_result_parses_but_remains_source_unbound() -> None:
    source = load_physical_execution_plan(PLAN_PATH)
    operation = source.collectives.operations[0]
    result = compile_distributed_mutation(
        source,
        CollectiveMutation(
            transformation_id="collective-legacy",
            operation_id=operation.operation_id,
            algorithm="tree",
            transport=operation.transport,
            channel_count=2,
            rank_order=tuple(reversed(operation.rank_order)),
        ),
        seed=17,
    )
    payload = result.model_dump(mode="json")
    payload["schema_version"] = "1.0.0"
    payload.pop("source_plan_hash")
    payload.pop("seed")
    extension = payload["candidate_plan"]["extensions"]["sloforge.dev/genesis-candidate"]
    extension.pop("source_plan_hash")
    legacy_plan = result.candidate_plan.__class__.model_validate_json(
        json.dumps(payload["candidate_plan"], sort_keys=True, separators=(",", ":")),
        strict=True,
    )
    payload["candidate_plan_hash"] = canonical_hash(legacy_plan)

    restored = DistributedSynthesisResult.model_validate_json(
        json.dumps(payload, sort_keys=True, separators=(",", ":")), strict=True
    )

    assert restored.schema_version == "1.0.0"
    with pytest.raises(ValueError, match="source-unbound"):
        restored.require_source_binding()


def test_pending_candidate_cannot_be_mutated_again_before_revalidation() -> None:
    source = load_physical_execution_plan(PLAN_PATH)
    operation = source.collectives.operations[0]
    first = compile_distributed_mutation(
        source,
        CollectiveMutation(
            transformation_id="first-mutation",
            operation_id=operation.operation_id,
            algorithm="tree",
            transport=operation.transport,
            channel_count=4,
            rank_order=tuple(reversed(operation.rank_order)),
        ),
        seed=1,
    )

    with pytest.raises(DistributedSynthesisError, match="requires Fabric revalidation"):
        compile_distributed_mutation(
            first.candidate_plan,
            CollectiveMutation(
                transformation_id="unsafe-second-mutation",
                operation_id=operation.operation_id,
                algorithm="ring",
                transport=operation.transport,
                channel_count=2,
                rank_order=operation.rank_order,
            ),
            seed=2,
        )


def test_result_model_rejects_candidate_that_retains_evidence() -> None:
    source = load_physical_execution_plan(PLAN_PATH)
    operation = source.collectives.operations[0]
    result = compile_distributed_mutation(
        source,
        CollectiveMutation(
            transformation_id="result-invariant",
            operation_id=operation.operation_id,
            algorithm="tree",
            transport=operation.transport,
            channel_count=4,
            rank_order=tuple(reversed(operation.rank_order)),
        ),
        seed=3,
    )
    unsafe_candidate = result.candidate_plan.model_copy(update={"evidence": source.evidence})
    result_payload = result.model_dump(mode="python")
    result_payload["candidate_plan"] = unsafe_candidate

    with pytest.raises(ValidationError, match="must not retain source evidence"):
        DistributedSynthesisResult.model_validate(result_payload, strict=True)


def test_result_model_rejects_mismatched_candidate_hash() -> None:
    source = load_physical_execution_plan(PLAN_PATH)
    operation = source.collectives.operations[0]
    result = compile_distributed_mutation(
        source,
        CollectiveMutation(
            transformation_id="candidate-hash-binding",
            operation_id=operation.operation_id,
            algorithm="tree",
            transport=operation.transport,
            channel_count=4,
            rank_order=tuple(reversed(operation.rank_order)),
        ),
        seed=4,
    )
    result_payload = result.model_dump(mode="python")
    result_payload["candidate_plan_hash"] = "0" * 64

    with pytest.raises(ValidationError, match="candidate plan hash does not match"):
        DistributedSynthesisResult.model_validate(result_payload, strict=True)


def test_rank_placement_mutation_reorders_memory_with_devices() -> None:
    source = load_physical_execution_plan(PLAN_PATH)
    source_order = tuple(reversed(range(len(source.rank_placement.bindings))))
    result = compile_distributed_mutation(
        source,
        RankPlacementMutation(
            transformation_id="reverse-rank-placement",
            source_rank_order=source_order,
        ),
        seed=17,
    )

    assert result.candidate_plan.rank_placement.bindings[0].gpu_id == (
        source.rank_placement.bindings[-1].gpu_id
    )
    assert result.candidate_plan.memory.allocations[0].capacity_bytes == (
        source.memory.allocations[-1].capacity_bytes
    )


def test_invalid_collective_rank_order_is_rejected_before_simulation() -> None:
    source = load_physical_execution_plan(PLAN_PATH)
    operation = source.collectives.operations[0]
    mutation = CollectiveMutation(
        transformation_id="invalid-rank-order",
        operation_id=operation.operation_id,
        algorithm=operation.algorithm,
        transport=operation.transport,
        channel_count=operation.channel_count,
        rank_order=operation.rank_order[:-1],
    )

    with pytest.raises(DistributedSynthesisError, match="permute participating ranks"):
        compile_distributed_mutation(source, mutation, seed=1)
