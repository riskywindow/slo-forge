"""Causal performance evidence, alignment, diagnosis, and counterfactual replay."""

from .alignment import align_run, estimate_alignment
from .comparison import compare_runs
from .diagnosis import diagnose, extract_signals
from .models import (
    AlignmentEstimate,
    AlignmentQuality,
    AutopsyEvent,
    AutopsyRun,
    BottleneckKind,
    CausalHypothesis,
    ClockSample,
    CounterValue,
    DiagnosisRecord,
    DifferentialComparison,
    EventType,
    EvidenceRef,
    FaultInterval,
    ResourceRef,
    SourceClock,
)

__all__ = [
    "AlignmentEstimate",
    "AlignmentQuality",
    "AutopsyEvent",
    "AutopsyRun",
    "BottleneckKind",
    "CausalHypothesis",
    "ClockSample",
    "CounterValue",
    "DiagnosisRecord",
    "DifferentialComparison",
    "EventType",
    "EvidenceRef",
    "FaultInterval",
    "ResourceRef",
    "SourceClock",
    "align_run",
    "compare_runs",
    "diagnose",
    "estimate_alignment",
    "extract_signals",
]
