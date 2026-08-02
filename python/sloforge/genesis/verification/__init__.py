"""Independent semantic, quality, resource, and performance verification."""

from .model import (
    BenchmarkContract,
    EvidenceStatus,
    MetricDirection,
    OperatorContract,
    OperatorCounterexample,
    OperatorVerificationResult,
    PerformanceEvidence,
    QualityContract,
    QualityEvidence,
    ResourceContract,
    ResourceDemand,
    ResourceEvidence,
    ShapeBound,
    VerificationError,
    VerificationLevel,
)
from .operator import verify_operator
from .performance import evaluate_performance
from .quality import evaluate_quality
from .resource import analyze_resources

__all__ = [
    "BenchmarkContract",
    "EvidenceStatus",
    "MetricDirection",
    "OperatorContract",
    "OperatorCounterexample",
    "OperatorVerificationResult",
    "PerformanceEvidence",
    "QualityContract",
    "QualityEvidence",
    "ResourceContract",
    "ResourceDemand",
    "ResourceEvidence",
    "ShapeBound",
    "VerificationError",
    "VerificationLevel",
    "analyze_resources",
    "evaluate_performance",
    "evaluate_quality",
    "verify_operator",
]
