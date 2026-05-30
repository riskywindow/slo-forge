"""Topology-aware offline deployment lowering."""

from sloforge.fabric.adapters.compatibility import RUNTIME_RANGES, validate_runtime
from sloforge.fabric.adapters.core import export_physical_plan
from sloforge.fabric.adapters.models import (
    AdapterCapabilities,
    DeploymentTarget,
    DynamoBackend,
    FabricAdapterContext,
    FabricExportResult,
    GangScheduler,
    GeneratedArtifact,
    RuntimeKind,
    UnsupportedCapabilityError,
)

__all__ = [
    "RUNTIME_RANGES",
    "AdapterCapabilities",
    "DeploymentTarget",
    "DynamoBackend",
    "FabricAdapterContext",
    "FabricExportResult",
    "GangScheduler",
    "GeneratedArtifact",
    "RuntimeKind",
    "UnsupportedCapabilityError",
    "export_physical_plan",
    "validate_runtime",
]
