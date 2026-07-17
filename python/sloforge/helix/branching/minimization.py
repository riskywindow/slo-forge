"""Bounded deterministic minimization of counterfactual branch interventions."""

from __future__ import annotations

from collections.abc import Callable
from itertools import combinations

from sloforge.helix.capture.models import canonical_digest

from .models import BranchIntervention, BranchMinimizationResult


def minimize_branch_interventions(
    interventions: tuple[BranchIntervention, ...],
    reproduces: Callable[[tuple[BranchIntervention, ...]], bool],
    *,
    max_evaluations: int = 256,
) -> BranchMinimizationResult:
    """Find a cardinality-minimal witness when the bounded exhaustive search completes."""

    if not 1 <= len(interventions) <= 16:
        raise ValueError("intervention count must be in 1..16")
    if not 1 <= max_evaluations <= 65_536:
        raise ValueError("max_evaluations must be in 1..65536")
    identifiers = [item.intervention_id for item in interventions]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("intervention identifiers must be unique")
    ordered = tuple(sorted(interventions, key=lambda item: item.intervention_id))
    evaluations = 1
    if not reproduces(ordered):
        raise ValueError("the full intervention set does not reproduce the target")
    best = ordered
    complete = True
    found = False
    for size in range(len(ordered)):
        for candidate in combinations(ordered, size):
            if evaluations >= max_evaluations:
                complete = False
                break
            evaluations += 1
            if reproduces(candidate):
                best = candidate
                found = True
                break
        if found or not complete:
            break
    original_ids = tuple(item.intervention_id for item in ordered)
    minimal_ids = tuple(item.intervention_id for item in best)
    payload = {
        "schema": "sloforge.helix.branch-minimization/v1",
        "original": original_ids,
        "minimal": minimal_ids,
        "evaluations": evaluations,
        "search_complete": complete,
    }
    return BranchMinimizationResult(
        original_intervention_ids=original_ids,
        minimal_intervention_ids=minimal_ids,
        evaluations=evaluations,
        search_complete=complete,
        witness_digest=canonical_digest(payload),
    )
