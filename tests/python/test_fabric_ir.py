from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from sloforge.fabric.ir import (
    FabricMigrationError,
    FabricValidationError,
    PhysicalExecutionPlan,
    canonical_hash,
    canonical_json,
    load_fabric_profile,
    load_model_graph,
    load_physical_execution_plan,
    load_recovery_plan,
    load_topology_graph,
    migrate_document,
    save_physical_execution_plan,
    write_json_schemas,
)

FIXTURES = Path(__file__).parents[1] / "fixtures" / "fabric"
SCHEMAS = Path(__file__).parents[2] / "schemas" / "fabric"

LOADERS = {
    "topology-graph-v1.json": load_topology_graph,
    "model-graph-v1.json": load_model_graph,
    "fabric-profile-v1.json": load_fabric_profile,
    "physical-execution-plan-v1.json": load_physical_execution_plan,
    "recovery-plan-v1.json": load_recovery_plan,
}

EXPECTED_HASHES = {
    "topology-graph-v1.json": "96a833c71bb9c590f5b42a719cfb4aaafa129b9eccde94f29372c783c61c67c9",
    "model-graph-v1.json": "ae4a4aeeb7ec6b1abed551dd6cb73b7583932f50f11af394f0c2f9a3dcc9abe7",
    "fabric-profile-v1.json": "c09263050d9b413d4c41e6d9bcad484058e95bdeac134d6ccab8a2554712ad40",
    "physical-execution-plan-v1.json": "116275c42541dcbefeeb682344ba34897a4238dd1fcb0feb6427505bb6990f03",
    "recovery-plan-v1.json": "ff9dcb36226df98bdd82714a6c312d6ef23aa011648a6718f4a9e98bbe7434f9",
}


@pytest.mark.parametrize("fixture_name", sorted(LOADERS))
def test_golden_documents_round_trip_canonically(fixture_name: str) -> None:
    loader = LOADERS[fixture_name]
    document = loader(FIXTURES / fixture_name)
    reparsed = loader(canonical_json(document))
    assert reparsed == document
    assert canonical_hash(reparsed) == EXPECTED_HASHES[fixture_name]


def test_checked_in_schemas_match_strict_model_generator(tmp_path: Path) -> None:
    generated = write_json_schemas(tmp_path)
    for actual in generated:
        expected = SCHEMAS / actual.name
        assert json.loads(actual.read_text()) == json.loads(expected.read_text())


@pytest.mark.parametrize("fixture_name", sorted(LOADERS))
def test_json_schema_accepts_golden_document(fixture_name: str) -> None:
    schema_name = fixture_name.replace(".json", ".schema.json")
    document = json.loads((FIXTURES / fixture_name).read_text())
    schema = json.loads((SCHEMAS / schema_name).read_text())
    jsonschema.Draft202012Validator(schema).validate(document)


def test_v1alpha1_topology_migrates_without_mutating_source() -> None:
    source = json.loads((FIXTURES / "topology-graph-v1alpha1.json").read_text())
    migrated = migrate_document(source)
    assert source["version"] == "v1alpha1"
    assert migrated["schema_version"] == "1.0.0"
    assert migrated["api_version"] == "sloforge.io/fabric/v1"
    assert migrated["topology_id"] == "alpha-single-host"
    assert "links" not in migrated
    assert load_topology_graph(migrated).topology_id == "alpha-single-host"


def test_v1alpha1_physical_field_names_migrate() -> None:
    source = json.loads((FIXTURES / "physical-execution-plan-v1.json").read_text())
    source["version"] = "v1alpha1"
    source.pop("schema_version")
    source["api_version"] = "sloforge.io/fabric/v1alpha1"
    for stable, alpha in {
        "logical_deployment_plan": "deployment_plan",
        "rank_placement": "placement",
        "communication_overlap": "overlap",
        "predicted_metrics": "predictions",
        "rejected_alternatives": "rejected_candidates",
    }.items():
        source[alpha] = source.pop(stable)
    assert load_physical_execution_plan(source).plan_id == "physical-fixture-v1"


def test_unknown_version_and_ambiguous_migration_fail_closed() -> None:
    with pytest.raises(FabricMigrationError, match="unsupported"):
        migrate_document({"schema_version": "2.0.0", "kind": "TopologyGraph"})
    source = {"version": "v1alpha1", "kind": "TopologyGraph", "id": "a", "topology_id": "b"}
    with pytest.raises(FabricMigrationError, match="both"):
        migrate_document(source)


def test_unknown_core_field_is_rejected() -> None:
    source = json.loads((FIXTURES / "physical-execution-plan-v1.json").read_text())
    source["rank_placement"]["bindings"][0]["surprise"] = True
    with pytest.raises(FabricValidationError, match="extra_forbidden"):
        load_physical_execution_plan(source)


def test_dangling_topology_edge_is_rejected() -> None:
    source = json.loads((FIXTURES / "topology-graph-v1.json").read_text())
    source["edges"][0]["target_node_id"] = "missing-gpu"
    with pytest.raises(FabricValidationError, match="unknown node"):
        load_topology_graph(source)


def test_rank_and_memory_coverage_invariants_are_enforced() -> None:
    source = json.loads((FIXTURES / "physical-execution-plan-v1.json").read_text())
    source["memory"]["allocations"].pop()
    with pytest.raises(FabricValidationError, match="memory plan must cover"):
        load_physical_execution_plan(source)


def test_recovery_external_mutation_requires_explicit_authorization() -> None:
    source = json.loads((FIXTURES / "recovery-plan-v1.json").read_text())
    source["actions"][0]["requires_external_mutation"] = True
    with pytest.raises(FabricValidationError, match="explicit authorization"):
        load_recovery_plan(source)


def test_save_is_atomic_and_reloadable(tmp_path: Path) -> None:
    document = load_physical_execution_plan(FIXTURES / "physical-execution-plan-v1.json")
    output = tmp_path / "nested" / "physical.json"
    save_physical_execution_plan(output, document)
    assert load_physical_execution_plan(output) == document
    assert output.read_bytes().endswith(b"\n")
    assert not tuple(output.parent.glob("*.tmp"))


@given(st.integers(min_value=0, max_value=2**63 - 1))
def test_seed_round_trip_is_canonical(seed: int) -> None:
    document = load_physical_execution_plan(FIXTURES / "physical-execution-plan-v1.json")
    updated = document.model_copy(
        update={"reproducibility": document.reproducibility.model_copy(update={"seed": seed})}
    )
    reparsed = PhysicalExecutionPlan.model_validate_json(canonical_json(updated))
    assert reparsed.reproducibility.seed == seed
    assert canonical_hash(reparsed) == canonical_hash(updated)


def test_strict_construction_rejects_coercion() -> None:
    source = json.loads((FIXTURES / "topology-graph-v1.json").read_text())
    source["container_limited"] = "false"
    with pytest.raises(ValidationError, match="bool_type"):
        # model_validate (unlike JSON decoding) also demonstrates no Python coercion.
        from sloforge.fabric.ir import TopologyGraph

        TopologyGraph.model_validate(source)
