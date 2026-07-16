"""Semantic compatibility analysis for Continuum state capsules."""

from .canonical import to_canonical_report
from .engine import analyze_compatibility
from .models import (
    CompatibilityDecision,
    CompatibilityReason,
    CompatibilityRequest,
    ExactnessClass,
    ModelSemantics,
    QualityEvidence,
    ReasonSeverity,
    RecomputationRequirement,
    RejectedCompatibilityClass,
    RuntimeCapabilities,
    StateDependencyEvidence,
    VerificationKind,
    VerificationObligation,
)

__all__ = [
    "CompatibilityDecision",
    "CompatibilityReason",
    "CompatibilityRequest",
    "ExactnessClass",
    "ModelSemantics",
    "QualityEvidence",
    "ReasonSeverity",
    "RecomputationRequirement",
    "RejectedCompatibilityClass",
    "RuntimeCapabilities",
    "StateDependencyEvidence",
    "VerificationKind",
    "VerificationObligation",
    "analyze_compatibility",
    "to_canonical_report",
]
