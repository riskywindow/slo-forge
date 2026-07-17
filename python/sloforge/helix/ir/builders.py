"""Small construction helpers that retain strict canonical validation."""

from __future__ import annotations

from .models import BranchGroup, BranchPoint, LineageReference, TrajectoryCapsule


def build_branch_group(
    *,
    group_id: str,
    branch_point: BranchPoint,
    trajectories: tuple[TrajectoryCapsule, ...],
    baseline_trajectory_id: str,
    created_at: str,
    lineage: tuple[LineageReference, ...],
) -> BranchGroup:
    """Build and fully validate a canonical branch group."""

    return BranchGroup(
        group_id=group_id,
        branch_point=branch_point,
        trajectories=trajectories,
        baseline_trajectory_id=baseline_trajectory_id,
        created_at=created_at,
        lineage=lineage,
    )
