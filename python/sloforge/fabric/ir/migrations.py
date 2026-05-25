"""Lossless migrations from the Fabric v1alpha1 wire format."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


class FabricMigrationError(ValueError):
    """Raised when a Fabric document cannot be migrated without ambiguity."""


def migrate_document(document: dict[str, Any]) -> dict[str, Any]:
    """Return a stable v1 Fabric document without mutating the source."""

    version = document.get("schema_version") or document.get("version")
    if version == "1.0.0":
        return deepcopy(document)
    if version not in {"v1alpha1", "0.1.0"}:
        raise FabricMigrationError(f"unsupported Fabric IR schema version: {version!r}")
    result = deepcopy(document)
    result.pop("version", None)
    result["schema_version"] = "1.0.0"
    result["api_version"] = "sloforge.io/fabric/v1"
    kind = result.get("kind")
    if kind not in {
        "TopologyGraph",
        "ModelGraph",
        "FabricProfile",
        "PhysicalExecutionPlan",
        "RecoveryPlan",
    }:
        raise FabricMigrationError(f"unsupported Fabric IR document kind: {kind!r}")

    if kind == "PhysicalExecutionPlan":
        renames = {
            "deployment_plan": "logical_deployment_plan",
            "placement": "rank_placement",
            "overlap": "communication_overlap",
            "predictions": "predicted_metrics",
            "rejected_candidates": "rejected_alternatives",
        }
    elif kind == "TopologyGraph":
        renames = {"id": "topology_id", "links": "edges"}
    elif kind == "ModelGraph":
        renames = {"revision": "model_revision", "digest": "model_digest"}
    elif kind == "FabricProfile":
        renames = {"id": "profile_id", "series": "measurements"}
    else:
        renames = {"id": "recovery_id", "proposal_actions": "actions"}
    for old, new in renames.items():
        if old in result:
            if new in result:
                raise FabricMigrationError(f"document contains both {old!r} and {new!r}")
            result[new] = result.pop(old)
    return result
