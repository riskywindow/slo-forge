"""Explicit migrations from historical IR wire formats."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .errors import IRValidationError
from .models import API_VERSION, SCHEMA_VERSION


def _migrate_plan_alpha1(source: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(source)
    result.pop("version", None)
    result["schema_version"] = SCHEMA_VERSION
    result["api_version"] = API_VERSION
    result["kind"] = "DeploymentPlan"

    # Alpha documents used concise field names.  These transformations are
    # deliberately lossless and fail when both old and new names are present.
    renames = {
        "replicas": "replica_topology",
        "routing_policy": "routing",
        "admission_policy": "admission",
        "batching_policy": "batching",
        "autoscaling_policy": "autoscaling",
        "cold_start_strategy": "cold_start",
        "canary_policy": "canary",
        "rollback_policy": "rollback",
    }
    for old, new in renames.items():
        if old in result:
            if new in result:
                raise IRValidationError(f"v1alpha1 document contains both {old!r} and {new!r}")
            result[new] = result.pop(old)

    model = result.get("model")
    if isinstance(model, dict) and "id" in model:
        if "model_id" in model:
            raise IRValidationError("v1alpha1 model contains both 'id' and 'model_id'")
        model["model_id"] = model.pop("id")

    engine = result.get("engine")
    if isinstance(engine, dict) and "runtime_name" in engine:
        if "runtime" in engine:
            raise IRValidationError("v1alpha1 engine contains both runtime names")
        engine["runtime"] = engine.pop("runtime_name")
    return result


def _migrate_evidence_alpha1(source: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(source)
    result.pop("version", None)
    result["schema_version"] = SCHEMA_VERSION
    result["api_version"] = API_VERSION
    result["kind"] = "EvidenceBundle"
    if "optimizer_decisions" in result:
        if "optimizer_history" in result:
            raise IRValidationError("v1alpha1 evidence contains duplicate optimizer history")
        result["optimizer_history"] = result.pop("optimizer_decisions")
    return result


def migrate_document(document: dict[str, Any]) -> dict[str, Any]:
    """Migrate a DeploymentPlan or EvidenceBundle to the current version.

    The operation never mutates its input.  Unknown versions are rejected
    rather than optimistically interpreted.
    """

    schema_version = document.get("schema_version")
    if schema_version == SCHEMA_VERSION:
        return deepcopy(document)
    legacy_version = document.get("version") or schema_version
    if legacy_version not in {"v1alpha1", "0.1.0"}:
        raise IRValidationError(f"unsupported IR schema version: {legacy_version!r}")

    kind = document.get("kind")
    if kind in {None, "DeploymentPlan", "deployment_plan"}:
        return _migrate_plan_alpha1(document)
    if kind in {"EvidenceBundle", "evidence_bundle"}:
        return _migrate_evidence_alpha1(document)
    raise IRValidationError(f"unsupported IR document kind: {kind!r}")
