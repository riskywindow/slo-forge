"""Autopsy bottleneck attribution and frozen-region enforcement."""

from .guidance import (
    FrozenMutationError,
    MutationGuard,
    build_mutation_budget,
    compare_search_efficiency,
    freeze_genome_regions,
)
from .models import (
    ALL_REGIONS,
    MutationBudget,
    SearchEfficiencyComparison,
    SearchEfficiencySummary,
)

__all__ = [
    "ALL_REGIONS",
    "FrozenMutationError",
    "MutationBudget",
    "MutationGuard",
    "SearchEfficiencyComparison",
    "SearchEfficiencySummary",
    "build_mutation_budget",
    "compare_search_efficiency",
    "freeze_genome_regions",
]
