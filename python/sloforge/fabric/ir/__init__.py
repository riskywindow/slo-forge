"""Public API for the SLOForge Fabric physical IR."""

from . import models as _models
from .canonical import canonical_hash, canonical_json
from .io import (
    FabricValidationError,
    load_fabric_profile,
    load_model_graph,
    load_physical_execution_plan,
    load_recovery_plan,
    load_topology_graph,
    save_fabric_profile,
    save_model_graph,
    save_physical_execution_plan,
    save_recovery_plan,
    save_topology_graph,
)
from .migrations import FabricMigrationError, migrate_document
from .models import *  # noqa: F403
from .schema import write_json_schemas

__all__ = [
    *_models.__all__,
    "FabricMigrationError",
    "FabricValidationError",
    "canonical_hash",
    "canonical_json",
    "load_fabric_profile",
    "load_model_graph",
    "load_physical_execution_plan",
    "load_recovery_plan",
    "load_topology_graph",
    "migrate_document",
    "save_fabric_profile",
    "save_model_graph",
    "save_physical_execution_plan",
    "save_recovery_plan",
    "save_topology_graph",
    "write_json_schemas",
]
