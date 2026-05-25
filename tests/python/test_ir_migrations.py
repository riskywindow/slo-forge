from __future__ import annotations

import json
from pathlib import Path

from sloforge.ir import API_VERSION, SCHEMA_VERSION, load_deployment_plan, migrate_document

FIXTURES = Path(__file__).parents[1] / "fixtures" / "ir"


def test_v1alpha1_plan_field_migration_is_lossless() -> None:
    source = json.loads((FIXTURES / "deployment-plan-v1alpha1-fragment.json").read_text())
    migrated = migrate_document(source)
    assert source["version"] == "v1alpha1"
    assert migrated["schema_version"] == SCHEMA_VERSION
    assert migrated["api_version"] == API_VERSION
    assert migrated["kind"] == "DeploymentPlan"
    assert migrated["model"]["model_id"] == "Qwen/Qwen3-0.6B"
    assert migrated["engine"]["runtime"] == "mock"
    assert migrated["replica_topology"] == {"minimum_replicas": 1}
    assert "replicas" not in migrated


def test_current_migration_returns_an_independent_copy() -> None:
    source = {"schema_version": SCHEMA_VERSION, "kind": "DeploymentPlan", "nested": {"x": 1}}
    migrated = migrate_document(source)
    migrated["nested"]["x"] = 2
    assert source["nested"]["x"] == 1


def test_full_v1alpha1_golden_migrates_to_stable_golden() -> None:
    legacy = json.loads((FIXTURES / "deployment-plan-v1alpha1.json").read_text())
    stable = json.loads((FIXTURES / "deployment-plan-v1.json").read_text())
    assert migrate_document(legacy) == stable
    assert load_deployment_plan(legacy) == load_deployment_plan(stable)
