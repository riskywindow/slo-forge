"""Atomic hard enforcement for all nine Genesis search budget dimensions."""

from __future__ import annotations

from dataclasses import dataclass, field

from sloforge.genesis.ir import BudgetUsage, SearchBudget


class BudgetExceeded(RuntimeError):
    def __init__(self, dimensions: tuple[str, ...]) -> None:
        super().__init__(f"search budget exhausted for: {', '.join(dimensions)}")
        self.dimensions = dimensions


_FLOAT_DIMENSIONS = (
    "wall_time_seconds",
    "cpu_time_seconds",
    "gpu_time_seconds",
    "cloud_cost_usd",
    "external_synthesis_cost_usd",
    "verifier_time_seconds",
)
_INTEGER_DIMENSIONS = ("candidate_count", "compilation_count", "benchmark_count")
_DIMENSIONS = (*_FLOAT_DIMENSIONS, *_INTEGER_DIMENSIONS)


def add_usage(left: BudgetUsage, right: BudgetUsage) -> BudgetUsage:
    return BudgetUsage(
        wall_time_seconds=left.wall_time_seconds + right.wall_time_seconds,
        cpu_time_seconds=left.cpu_time_seconds + right.cpu_time_seconds,
        gpu_time_seconds=left.gpu_time_seconds + right.gpu_time_seconds,
        cloud_cost_usd=left.cloud_cost_usd + right.cloud_cost_usd,
        external_synthesis_cost_usd=(
            left.external_synthesis_cost_usd + right.external_synthesis_cost_usd
        ),
        candidate_count=left.candidate_count + right.candidate_count,
        compilation_count=left.compilation_count + right.compilation_count,
        benchmark_count=left.benchmark_count + right.benchmark_count,
        verifier_time_seconds=left.verifier_time_seconds + right.verifier_time_seconds,
    )


def exceeded_dimensions(usage: BudgetUsage, budget: SearchBudget) -> tuple[str, ...]:
    return tuple(name for name in _DIMENSIONS if getattr(usage, name) > getattr(budget, name))


def usage_within(usage: BudgetUsage, maximum: BudgetUsage) -> bool:
    return not any(getattr(usage, name) > getattr(maximum, name) for name in _DIMENSIONS)


@dataclass
class BudgetManager:
    budget: SearchBudget
    usage: BudgetUsage = field(default_factory=BudgetUsage)

    def ensure_reservable(self, maximum_charge: BudgetUsage) -> None:
        dimensions = exceeded_dimensions(add_usage(self.usage, maximum_charge), self.budget)
        if dimensions:
            raise BudgetExceeded(dimensions)

    def consume(self, actual: BudgetUsage, *, reserved_maximum: BudgetUsage) -> BudgetUsage:
        if not usage_within(actual, reserved_maximum):
            over = tuple(
                name
                for name in _DIMENSIONS
                if getattr(actual, name) > getattr(reserved_maximum, name)
            )
            raise ValueError(f"evaluator exceeded its reserved budget: {', '.join(over)}")
        updated = add_usage(self.usage, actual)
        dimensions = exceeded_dimensions(updated, self.budget)
        if dimensions:
            raise BudgetExceeded(dimensions)
        self.usage = updated
        return updated

    def consume_candidate(self) -> BudgetUsage:
        charge = BudgetUsage(candidate_count=1)
        self.ensure_reservable(charge)
        return self.consume(charge, reserved_maximum=charge)
