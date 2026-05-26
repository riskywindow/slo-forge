"""Topology-aware physical execution compiler."""

from .core import (
    CompilerAssumptions,
    CompilerConstraints,
    CompilerObjective,
    CompilerRequest,
    OptimizationStrategy,
    PhysicalCompileResult,
    compile_physical_plan,
)

__all__ = [
    "CompilerAssumptions",
    "CompilerConstraints",
    "CompilerObjective",
    "CompilerRequest",
    "OptimizationStrategy",
    "PhysicalCompileResult",
    "compile_physical_plan",
]
