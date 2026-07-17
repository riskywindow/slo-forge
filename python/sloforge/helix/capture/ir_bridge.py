"""Late binding from coordinated capture evidence to canonical Helix IR."""

from __future__ import annotations

import importlib
from typing import Any

from .models import CoordinatedBranchPoint


def build_ir_branch_point(
    capture: CoordinatedBranchPoint,
    *,
    environment_state: object,
    policy_epoch: object,
    prefix_digest: object,
    candidate_labels: tuple[str, ...],
    lineage: tuple[object, ...] = (),
) -> object:
    """Construct ``helix.ir.BranchPoint`` without coupling capture imports to the IR lane."""

    branch_point_model: Any = importlib.import_module("sloforge.helix.ir").BranchPoint

    document: dict[str, Any] = {
        "branch_point_id": capture.branch_point_id,
        "source_trajectory_id": capture.source_trajectory_id,
        "event_index": capture.boundary.action_watermark,
        "token_index": capture.boundary.model_token_watermark,
        "environment_state": environment_state,
        "policy_epoch": policy_epoch,
        "prefix_digest": prefix_digest,
        "seed": capture.seed,
        "created_at": capture.created_at,
        "reason": capture.reason,
        "candidate_labels": candidate_labels,
        "lineage": lineage,
    }
    return branch_point_model.model_validate(document, strict=True)
