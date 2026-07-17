"""Deterministic CPU fault campaigns for the Helix learning loop."""

from .io import MAX_FAULT_MATRIX_BYTES, load_fault_plan_request
from .models import (
    ActivationInterval,
    FaultCampaignFailed,
    FaultCampaignResult,
    FaultExecutionResult,
    FaultKind,
    FaultMutation,
    FaultObservation,
    FaultPlan,
    FaultPlanRequest,
    FaultResponse,
    FaultSpec,
    FaultStage,
    InjectedFault,
    MutationOperation,
    canonical_digest,
)
from .runner import (
    DeterministicFaultInjector,
    FaultCallback,
    FaultRunner,
    compile_fault_plan,
    run_fault_plan,
)

__all__ = [
    "MAX_FAULT_MATRIX_BYTES",
    "ActivationInterval",
    "DeterministicFaultInjector",
    "FaultCallback",
    "FaultCampaignFailed",
    "FaultCampaignResult",
    "FaultExecutionResult",
    "FaultKind",
    "FaultMutation",
    "FaultObservation",
    "FaultPlan",
    "FaultPlanRequest",
    "FaultResponse",
    "FaultRunner",
    "FaultSpec",
    "FaultStage",
    "InjectedFault",
    "MutationOperation",
    "canonical_digest",
    "compile_fault_plan",
    "load_fault_plan_request",
    "run_fault_plan",
]
