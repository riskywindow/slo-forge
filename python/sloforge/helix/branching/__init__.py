"""Continuum-backed Helix branch coordination and minimization."""

from .coordinator import (
    BranchCleanupError,
    BranchCompatibilityError,
    BranchError,
    BranchPlanError,
    create_branch_group,
)
from .ir_bridge import build_ir_state_reuse_report
from .minimization import minimize_branch_interventions
from .models import (
    BranchGroupExecution,
    BranchIntervention,
    BranchMember,
    BranchMinimizationResult,
    BranchPlan,
    BranchStrategy,
    CrossPolicyBranch,
    ExactCowBranch,
    InterventionKind,
    RngActivationOverride,
    RngMutationBranch,
    StateReuseReport,
)

__all__ = [
    "BranchCleanupError",
    "BranchCompatibilityError",
    "BranchError",
    "BranchGroupExecution",
    "BranchIntervention",
    "BranchMember",
    "BranchMinimizationResult",
    "BranchPlan",
    "BranchPlanError",
    "BranchStrategy",
    "CrossPolicyBranch",
    "ExactCowBranch",
    "InterventionKind",
    "RngActivationOverride",
    "RngMutationBranch",
    "StateReuseReport",
    "build_ir_state_reuse_report",
    "create_branch_group",
    "minimize_branch_interventions",
]
