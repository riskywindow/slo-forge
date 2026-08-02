"""Executable adversaries and minimized Genesis counterexamples."""

from pathlib import Path

from . import models as _models
from .adversaries import (
    generate_resource_cases,
    generate_schedule_cases,
    generate_tensor_cases,
    generate_topology_cases,
)
from .benchmark import audit_benchmark_integrity
from .conversion import build_regression_corpus, to_counterexample
from .fixture import UnsafeStreamingCandidate, unsafe_benchmark_comparison
from .minimize import DeltaMinimizationResult, minimize_sequence
from .models import *  # noqa: F403
from .runner import corpus_from_report, replay_regression_corpus, run_red_team


def run_demo(output_directory: Path, *, seed: int = 73129) -> _models.RedTeamDemoResult:
    """Lazily load the demo so module execution stays warning-free."""

    from .demo import run_demo as run_demo_implementation

    return run_demo_implementation(output_directory, seed=seed)


__all__ = [
    *_models.__all__,
    "DeltaMinimizationResult",
    "UnsafeStreamingCandidate",
    "audit_benchmark_integrity",
    "build_regression_corpus",
    "corpus_from_report",
    "generate_resource_cases",
    "generate_schedule_cases",
    "generate_tensor_cases",
    "generate_topology_cases",
    "minimize_sequence",
    "replay_regression_corpus",
    "run_demo",
    "run_red_team",
    "to_counterexample",
    "unsafe_benchmark_comparison",
]
