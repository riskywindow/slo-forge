"""ServingSynthBench randomized task generation and CPU evaluation."""

from .grammar import (
    execute_architecture,
    generate_tasks,
    load_hidden_cases,
    load_task,
    load_workload,
)
from .integrity import audit_raw_samples
from .models import (
    AggregateMetrics,
    ArchitectureSpec,
    BaselineKind,
    BaselineStatus,
    BaselineSummary,
    BlockKind,
    BlockSpec,
    CpuRunConfiguration,
    GrammarConfiguration,
    HiddenCase,
    HiddenCaseResult,
    HiddenEvaluationSummary,
    IntegrityReport,
    RawCpuSample,
    SpecialCaseAudit,
    SpecialCaseFinding,
    SynthBenchReport,
    TaskDescriptor,
    TaskRunReport,
    WorkloadRequest,
)
from .runner import run_cpu_benchmark
from .special_case import audit_special_casing

__all__ = [
    "AggregateMetrics",
    "ArchitectureSpec",
    "BaselineKind",
    "BaselineStatus",
    "BaselineSummary",
    "BlockKind",
    "BlockSpec",
    "CpuRunConfiguration",
    "GrammarConfiguration",
    "HiddenCase",
    "HiddenCaseResult",
    "HiddenEvaluationSummary",
    "IntegrityReport",
    "RawCpuSample",
    "SpecialCaseAudit",
    "SpecialCaseFinding",
    "SynthBenchDemoResult",
    "SynthBenchReport",
    "TaskDescriptor",
    "TaskRunReport",
    "WorkloadRequest",
    "audit_raw_samples",
    "audit_special_casing",
    "execute_architecture",
    "generate_tasks",
    "load_hidden_cases",
    "load_task",
    "load_workload",
    "run_cpu_benchmark",
    "run_synthbench_demo",
]


def __getattr__(name: str) -> object:
    if name in {"SynthBenchDemoResult", "run_synthbench_demo"}:
        from .demo import SynthBenchDemoResult, run_synthbench_demo

        return {
            "SynthBenchDemoResult": SynthBenchDemoResult,
            "run_synthbench_demo": run_synthbench_demo,
        }[name]
    raise AttributeError(name)
