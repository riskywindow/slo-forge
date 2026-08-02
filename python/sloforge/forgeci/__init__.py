"""ForgeCI reproducible performance regression detection and bisection."""

from sloforge.forgeci.bisect import bisect_regression
from sloforge.forgeci.evaluation import fixture_case, run_fixture_evaluation
from sloforge.forgeci.fixture import create_regression_fixture
from sloforge.forgeci.hardware import RequirementMismatch, observe_hardware, validate_requirements
from sloforge.forgeci.io import load_matrix, write_matrix
from sloforge.forgeci.matrix import run_matrix
from sloforge.forgeci.minimize import minimize_benchmark, reproducer_environment
from sloforge.forgeci.models import (
    BenchmarkInput,
    BenchmarkMatrix,
    BenchmarkSpec,
    BisectResult,
    CommandSpec,
    ComparisonClassification,
    ComparisonRecord,
    EnvironmentSpec,
    HardwareRequirement,
    MatrixCase,
    MatrixRunRecord,
    MetricDirection,
    MetricSpec,
    MinimalReproducer,
    RunRecord,
)
from sloforge.forgeci.report import render_upstream_issue
from sloforge.forgeci.runner import compare_runs, run_case, verify_run_artifact, write_comparison

__all__ = [
    "BenchmarkInput",
    "BenchmarkMatrix",
    "BenchmarkSpec",
    "BisectResult",
    "CommandSpec",
    "ComparisonClassification",
    "ComparisonRecord",
    "EnvironmentSpec",
    "HardwareRequirement",
    "MatrixCase",
    "MatrixRunRecord",
    "MetricDirection",
    "MetricSpec",
    "MinimalReproducer",
    "RequirementMismatch",
    "RunRecord",
    "bisect_regression",
    "compare_runs",
    "create_regression_fixture",
    "fixture_case",
    "load_matrix",
    "minimize_benchmark",
    "observe_hardware",
    "render_upstream_issue",
    "reproducer_environment",
    "run_case",
    "run_fixture_evaluation",
    "run_matrix",
    "validate_requirements",
    "verify_run_artifact",
    "write_comparison",
    "write_matrix",
]
