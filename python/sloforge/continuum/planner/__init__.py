"""SLO-aware, evidence-bound Continuum migration planning."""

from .core import (
    AccessPattern,
    CandidateEstimate,
    ExactnessRequirement,
    MeasuredRate,
    MigrationPlanningInput,
    MigrationStrategy,
    PlannedMigration,
    plan_migration,
)
from .integration import (
    ContinuumMigrationExtension,
    IntegratedPlanningResult,
    fabric_transfer_rates,
    plan_with_fabric_and_warmpath,
    with_continuum_extension,
)

__all__ = [
    "AccessPattern",
    "CandidateEstimate",
    "ContinuumMigrationExtension",
    "ExactnessRequirement",
    "IntegratedPlanningResult",
    "MeasuredRate",
    "MigrationPlanningInput",
    "MigrationStrategy",
    "PlannedMigration",
    "fabric_transfer_rates",
    "plan_migration",
    "plan_with_fabric_and_warmpath",
    "with_continuum_extension",
]
