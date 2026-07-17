"""Trusted local promotion registry and active-session routing."""

from .capsule import (
    ContinuumCompatibilityBinding,
    PromotionArtifactReference,
    PromotionArtifactSource,
    PromotionCapsuleValidation,
    PromotionCapsuleValidationError,
    TrustedPolicyPromotionCapsule,
    build_policy_promotion_capsule,
    validate_policy_promotion_capsule,
)
from .registry import (
    CompatibilityClass,
    GateEvidence,
    PolicyRegistry,
    PromotionState,
    SessionRoute,
)

__all__ = [
    "CompatibilityClass",
    "ContinuumCompatibilityBinding",
    "GateEvidence",
    "PolicyRegistry",
    "PromotionArtifactReference",
    "PromotionArtifactSource",
    "PromotionCapsuleValidation",
    "PromotionCapsuleValidationError",
    "PromotionState",
    "SessionRoute",
    "TrustedPolicyPromotionCapsule",
    "build_policy_promotion_capsule",
    "validate_policy_promotion_capsule",
]
