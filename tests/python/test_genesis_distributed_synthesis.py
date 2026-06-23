from __future__ import annotations

from pathlib import Path

import pytest

from sloforge.fabric.ir import canonical_hash, load_physical_execution_plan
from sloforge.genesis.distributed_synthesis import (
    CollectiveMutation,
    DistributedSynthesisError,
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
    assert result.bounded_model_check_required


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
