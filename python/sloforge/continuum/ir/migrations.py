"""Lossless migrations for known pre-stable Continuum fixture documents."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .models import API_VERSION, SCHEMA_VERSION


class ContinuumMigrationError(ValueError):
    """A document cannot be migrated without guessing state semantics."""


_KINDS = {
    "LogicalStateSchema",
    "PhysicalStateLayout",
    "ExecutionStateCapsule",
    "CompatibilityReport",
    "StateTransformationIR",
    "MigrationPlan",
    "StateTransaction",
    "MigrationVerificationEvidence",
}


def migrate_document(document: dict[str, Any]) -> dict[str, Any]:
    """Migrate a known alpha document while rejecting ambiguous aliases."""

    result = deepcopy(document)
    version = result.get("schema_version") or result.get("version")
    kind = result.get("kind")
    if version == SCHEMA_VERSION:
        if kind not in _KINDS:
            raise ContinuumMigrationError(f"unsupported Continuum document kind: {kind!r}")
        return result
    if version not in {"v1alpha1", "0.1.0"}:
        raise ContinuumMigrationError(f"unsupported Continuum schema version: {version!r}")
    aliases = {
        "logical_state": "LogicalStateSchema",
        "physical_state": "PhysicalStateLayout",
        "execution_state_capsule": "ExecutionStateCapsule",
        "compatibility_report": "CompatibilityReport",
        "state_transformation": "StateTransformationIR",
        "migration_plan": "MigrationPlan",
        "state_transaction": "StateTransaction",
        "verification_evidence": "MigrationVerificationEvidence",
    }
    stable_kind = aliases.get(str(kind), str(kind))
    if stable_kind not in _KINDS:
        raise ContinuumMigrationError(f"unsupported Continuum document kind: {kind!r}")
    result.pop("version", None)
    result["schema_version"] = SCHEMA_VERSION
    result["api_version"] = API_VERSION
    result["kind"] = stable_kind
    renames = {
        "ExecutionStateCapsule": {"logical": "logical_state", "physical": "physical_state"},
        "PhysicalStateLayout": {"runtime_identity": "runtime"},
        "StateTransformationIR": {"id": "transformation_id"},
        "MigrationPlan": {"id": "plan_id"},
        "StateTransaction": {"phase": "current_phase"},
    }.get(stable_kind, {})
    for old, new in renames.items():
        if old in result:
            if new in result:
                raise ContinuumMigrationError(f"document contains both {old!r} and {new!r}")
            result[new] = result.pop(old)
    return result
