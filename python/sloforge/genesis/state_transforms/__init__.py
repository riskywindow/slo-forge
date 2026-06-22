"""Persistent-state synthesis, conservative compilation, and trace verification."""

from .compiler import compile_transformation, region_capacity_bytes, validate_region
from .model import (
    CompiledStateTransition,
    Consistency,
    Ownership,
    StateAction,
    StateEvent,
    StateLayout,
    StateRegion,
    StateTraceResult,
    StateTransformation,
    StateTransformError,
    StateViolation,
    StorageTier,
    TransformationKind,
    Visibility,
)
from .verifier import verify_state_trace

__all__ = [
    "CompiledStateTransition",
    "Consistency",
    "Ownership",
    "StateAction",
    "StateEvent",
    "StateLayout",
    "StateRegion",
    "StateTraceResult",
    "StateTransformError",
    "StateTransformation",
    "StateViolation",
    "StorageTier",
    "TransformationKind",
    "Visibility",
    "compile_transformation",
    "region_capacity_bytes",
    "validate_region",
    "verify_state_trace",
]
