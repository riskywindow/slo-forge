from __future__ import annotations

import pytest

from sloforge.continuum.adapters.vllm_reclamation import (
    RuntimeBranchCaptureInput,
    build_canonical_capture_plan,
)


def _branches(
    *,
    left_blocks: tuple[int, ...] = (91, 92, 11, 71),
    right_blocks: tuple[int, ...] = (91, 92, 12, 72),
) -> tuple[RuntimeBranchCaptureInput, ...]:
    return (
        RuntimeBranchCaptureInput(
            logical_branch_id="branch.0",
            parent_logical_branch_id="root",
            token_ids=(10, 11, 12, 13, 20, 21, 99),
            computed_tokens=6,
            source_block_indices=left_blocks,
        ),
        RuntimeBranchCaptureInput(
            logical_branch_id="branch.1",
            parent_logical_branch_id="root",
            token_ids=(10, 11, 12, 13, 30, 31, 98),
            computed_tokens=6,
            source_block_indices=right_blocks,
        ),
    )


def _plan(
    branches: tuple[RuntimeBranchCaptureInput, ...],
    epochs: dict[int, int],
):
    return build_canonical_capture_plan(
        branches=branches,
        block_size_tokens=2,
        logical_token_bytes=8,
        physical_page_bytes=16,
        gpu_uuid="GPU-fixture-004",
        allocation_epoch_by_block=epochs,
    )


def test_capture_plan_deduplicates_root_and_trims_uncomputed_headroom() -> None:
    plan = _plan(
        _branches(),
        {91: 1, 92: 2, 11: 3, 12: 4, 71: 5, 72: 6},
    )

    assert [item[1] for item in plan.page_order] == [91, 92, 11, 12]
    assert [item[2] for item in plan.page_order] == [2, 2, 2, 2]
    assert [item[3] for item in plan.page_order] == [
        ("branch.0", "branch.1"),
        ("branch.0", "branch.1"),
        ("branch.0",),
        ("branch.1",),
    ]
    assert plan.branch_tables[0].logical_page_ids == (
        "logical-page-000000",
        "logical-page-000001",
        "logical-page-000002",
    )
    assert plan.branch_tables[1].logical_page_ids == (
        "logical-page-000000",
        "logical-page-000001",
        "logical-page-000003",
    )
    assert plan.logical_state_bytes == 64
    assert plan.shared_logical_bytes == 32
    assert plan.private_logical_bytes == 32
    assert plan.physical_source_bytes == 64
    assert {item.source.block_index for item in plan.capture_evidence.bindings} == {
        91,
        92,
        11,
        12,
    }


def test_logical_plan_is_stable_when_source_physical_ids_change() -> None:
    first = _plan(
        _branches(),
        {91: 1, 92: 2, 11: 3, 12: 4},
    )
    second = _plan(
        _branches(
            left_blocks=(201, 202, 211),
            right_blocks=(201, 202, 212),
        ),
        {201: 7, 202: 8, 211: 9, 212: 10},
    )

    assert first.branch_tables == second.branch_tables
    assert [
        (logical_id, valid_tokens, owners)
        for logical_id, _block, valid_tokens, owners in first.page_order
    ] == [
        (logical_id, valid_tokens, owners)
        for logical_id, _block, valid_tokens, owners in second.page_order
    ]
    assert first.capture_evidence != second.capture_evidence


def test_capture_plan_rejects_shared_physical_page_with_divergent_tokens() -> None:
    branches = list(_branches())
    branches[1] = branches[1].model_copy(update={"token_ids": (10, 55, 12, 13, 30, 31, 98)})
    with pytest.raises(ValueError, match="divergent branch token histories"):
        _plan(tuple(branches), {91: 1, 92: 2, 11: 3, 12: 4})


def test_capture_plan_rejects_partial_group_sharing() -> None:
    branches = (
        *_branches(left_blocks=(91, 92, 11), right_blocks=(91, 92, 12)),
        RuntimeBranchCaptureInput(
            logical_branch_id="branch.2",
            parent_logical_branch_id="root",
            token_ids=(10, 11, 12, 13, 40, 41, 97),
            computed_tokens=6,
            source_block_indices=(81, 82, 13),
        ),
    )
    with pytest.raises(ValueError, match="complete branch group"):
        _plan(branches, {91: 1, 92: 2, 11: 3, 12: 4, 81: 5, 82: 6, 13: 7})


def test_capture_plan_requires_all_live_allocation_epochs() -> None:
    with pytest.raises(ValueError, match="lacks allocation epochs"):
        _plan(_branches(), {91: 1, 92: 2, 11: 3})
