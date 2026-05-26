"""Causal performance evidence, alignment, diagnosis, and counterfactual replay."""

from .alignment import align_run, estimate_alignment
from .comparison import compare_runs
from .counterfactual import (
    CounterfactualReplay,
    CounterfactualScenario,
    RemoveFault,
    ReplaceResource,
    ScaleRank,
    ScaleResourceCurve,
    ScenarioEvaluation,
    replay_counterfactuals,
    run_fabric_simulator,
)
from .diagnosis import diagnose, extract_signals
from .minimize import MinimizationResult, minimize_run
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
    "CounterfactualReplay",
    "CounterfactualScenario",
    "DiagnosisRecord",
    "DifferentialComparison",
    "EventType",
    "EvidenceRef",
    "FaultInterval",
    "MinimizationResult",
    "RemoveFault",
    "ReplaceResource",
    "ResourceRef",
    "ScaleRank",
    "ScaleResourceCurve",
    "ScenarioEvaluation",
    "SourceClock",
    "align_run",
    "compare_runs",
    "diagnose",
    "estimate_alignment",
    "extract_signals",
    "minimize_run",
    "replay_counterfactuals",
    "run_fabric_simulator",
]
