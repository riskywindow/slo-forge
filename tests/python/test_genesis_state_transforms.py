from __future__ import annotations

from dataclasses import replace

import pytest

from sloforge.genesis.state_transforms import (
    Consistency,
    Ownership,
    StateAction,
    StateEvent,
    StateLayout,
    StateRegion,
    StateTransformation,
    StateTransformError,
    StorageTier,
    TransformationKind,
    Visibility,
    compile_transformation,
    verify_state_trace,
)


def _region() -> StateRegion:
    return StateRegion(
        region_id="decoder-state",
        semantic_contract="autoregressive recurrent state v1",
        ownership=Ownership.EXCLUSIVE,
        owners=("worker-0",),
        visibility=Visibility.REQUEST,
        layout=StateLayout.CONTIGUOUS,
        storage=StorageTier.DEVICE,
        dtype="float32",
        bytes_per_item=64,
        maximum_items=32,
        replication_factor=1,
        consistency=Consistency.LINEARIZABLE,
        mutable=True,
        checkpointed=True,
        compatible_genome_hashes=("genome-b",),
    )


def test_compile_state_transition_checks_coexistence_and_swap_category() -> None:
    source = _region()
    target = replace(source, layout=StateLayout.PAGED, storage=StorageTier.PINNED_HOST)
    transformation = StateTransformation(
        "state/paged-offload/v1",
        TransformationKind.LAYOUT,
        source,
        target,
        exact=True,
        expected_quality_cost=0.0,
        expected_memory_delta_bytes=0,
        migration_chunk_bytes=256,
        preconditions=("request_boundary",),
        proof_obligations=("differential_state", "rollback"),
        rollback_strategy="retain_source_until_commit",
    )
    compiled = compile_transformation(
        transformation, memory_capacity_bytes=8192, safety_margin_fraction=0.25
    )
    assert compiled.coexistence_peak_bytes == 4096
    assert compiled.migration_chunks == 8
    assert compiled.requires_request_boundary
    with pytest.raises(StateTransformError, match="coexistence"):
        compile_transformation(transformation, memory_capacity_bytes=4096)


def test_valid_trace_preserves_ownership_cancellation_and_release() -> None:
    events = (
        StateEvent(0, StateAction.ALLOCATE, "decoder-state", "r1", "worker-0", 1, 4),
        StateEvent(1, StateAction.ACQUIRE, "decoder-state", "r1", "worker-0", 1),
        StateEvent(2, StateAction.WRITE, "decoder-state", "r1", "worker-0", 1),
        StateEvent(3, StateAction.CHECKPOINT, "decoder-state", "r1", "worker-0", 1),
        StateEvent(4, StateAction.CANCEL_REQUEST, "decoder-state", "r1", "worker-0", 1),
        StateEvent(5, StateAction.RELEASE, "decoder-state", "r1", "worker-0", 1),
    )
    result = verify_state_trace(_region(), events, memory_capacity_bytes=1024)
    assert result.valid
    assert result.peak_memory_bytes == 256
    assert result.final_memory_bytes == 0


def test_partial_migration_and_incompatible_state_produce_real_violations() -> None:
    events = (
        StateEvent(0, StateAction.ALLOCATE, "decoder-state", "r1", "worker-0", 1, 2),
        StateEvent(
            1,
            StateAction.BEGIN_MIGRATION,
            "decoder-state",
            "r1",
            "worker-0",
            1,
            target_actor="worker-1",
            target_genome_hash="genome-incompatible",
        ),
        StateEvent(2, StateAction.READ, "decoder-state", "r1", "worker-0", 1),
        StateEvent(3, StateAction.RELEASE, "decoder-state", "r1", "worker-0", 1),
    )
    result = verify_state_trace(_region(), events, memory_capacity_bytes=1024)
    invariants = {item.invariant for item in result.violations}
    assert "state_compatibility" in invariants
    assert "acquire_before_read" in invariants


def test_double_release_use_after_free_and_memory_bound_fail_closed() -> None:
    events = (
        StateEvent(0, StateAction.ALLOCATE, "decoder-state", "r1", "worker-0", 1, 32),
        StateEvent(1, StateAction.RELEASE, "decoder-state", "r1", "worker-0", 1),
        StateEvent(2, StateAction.RELEASE, "decoder-state", "r1", "worker-0", 1),
    )
    result = verify_state_trace(_region(), events, memory_capacity_bytes=1024)
    invariants = {item.invariant for item in result.violations}
    assert "memory_bound" in invariants
    assert "no_use_after_free" in invariants


def test_ambiguous_region_ownership_is_rejected() -> None:
    invalid = replace(_region(), owners=("worker-0", "worker-1"))
    with pytest.raises(StateTransformError, match="exactly one owner"):
        verify_state_trace(invalid, (), memory_capacity_bytes=1024)
