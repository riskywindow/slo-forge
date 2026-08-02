from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from pydantic import BaseModel, ValidationError

from sloforge.autopsy.counterfactual import CounterfactualReplay
from sloforge.autopsy.minimize import MinimizationResult
from sloforge.autopsy.models import (
    AutopsyEvent,
    AutopsyRun,
    DiagnosisRecord,
    DifferentialComparison,
)
from sloforge.util import canonical_json, sha256_bytes
from sloforge.warmpath.models import (
    ArtifactGraph,
    ExecutionRecord,
    StartupProfile,
    WarmPathPlan,
)

ROOT = Path(__file__).parents[2]
AUTOPSY_FIXTURES = ROOT / "tests" / "fixtures" / "autopsy"
WARMPATH_FIXTURES = ROOT / "tests" / "fixtures" / "warmpath"
AUTOPSY_SCHEMAS = ROOT / "schemas" / "autopsy"
WARMPATH_SCHEMAS = ROOT / "schemas" / "warmpath"

SchemaCase = tuple[Path, Path, type[BaseModel], str]

CASES: tuple[SchemaCase, ...] = (
    (
        AUTOPSY_FIXTURES / "event-v1.json",
        AUTOPSY_SCHEMAS / "event-v1.schema.json",
        AutopsyEvent,
        "sloforge.autopsy.event/v1",
    ),
    (
        AUTOPSY_FIXTURES / "run-v1.json",
        AUTOPSY_SCHEMAS / "run-v1.schema.json",
        AutopsyRun,
        "sloforge.autopsy.run/v1",
    ),
    (
        AUTOPSY_FIXTURES / "comparison-v1.json",
        AUTOPSY_SCHEMAS / "comparison-v1.schema.json",
        DifferentialComparison,
        "sloforge.autopsy.comparison/v1",
    ),
    (
        AUTOPSY_FIXTURES / "diagnosis-v1.json",
        AUTOPSY_SCHEMAS / "diagnosis-v1.schema.json",
        DiagnosisRecord,
        "sloforge.autopsy.diagnosis/v1",
    ),
    (
        AUTOPSY_FIXTURES / "counterfactual-v1.json",
        AUTOPSY_SCHEMAS / "counterfactual-v1.schema.json",
        CounterfactualReplay,
        "sloforge.autopsy.counterfactual/v1",
    ),
    (
        AUTOPSY_FIXTURES / "minimization-v1.json",
        AUTOPSY_SCHEMAS / "minimization-v1.schema.json",
        MinimizationResult,
        "sloforge.autopsy.minimization/v1",
    ),
    (
        WARMPATH_FIXTURES / "artifact-graph-v1.json",
        WARMPATH_SCHEMAS / "artifact-graph-v1.schema.json",
        ArtifactGraph,
        "1.0.0",
    ),
    (
        WARMPATH_FIXTURES / "startup-profile-v1.json",
        WARMPATH_SCHEMAS / "startup-profile-v1.schema.json",
        StartupProfile,
        "1.0.0",
    ),
    (
        WARMPATH_FIXTURES / "plan-v1.json",
        WARMPATH_SCHEMAS / "plan-v1.schema.json",
        WarmPathPlan,
        "1.0.0",
    ),
    (
        WARMPATH_FIXTURES / "execution-v1.json",
        WARMPATH_SCHEMAS / "execution-v1.schema.json",
        ExecutionRecord,
        "1.0.0",
    ),
)

EXPECTED_CANONICAL_HASHES = {
    "comparison-v1.json": "d59c43b1f4329e5d85baccc9e3b8eb1518d4881449a3461c01b6fa3caa4cddee",
    "counterfactual-v1.json": "ee738f70f06c4e758b6c8912628d6ff6809102f2025d39e11151a4ea4ad22391",
    "diagnosis-v1.json": "2a9180691b8f4f30e5aca136b6776a857aec8dc1485c4ec796bdc9731a10ce59",
    "event-v1.json": "10dfcbed7efce7c37fa3589947069f63a3749276066a0e340eb59d425ebf955d",
    "minimization-v1.json": "588fee895a639da6190f521ebc9fdd3ae87742e29ddefe038d0a4451fe60b80b",
    "run-v1.json": "9e2cc63215041bd297d2b8d27da6f0098d5429f39fa9c08a1590a2c1ad620184",
    "artifact-graph-v1.json": "388b47de3fd3e27af426e132c25098cc6c5cd2ba3793b518fbd23f838ec5f2c8",
    "execution-v1.json": "a1bc0ca74bf57ed54fb58aab8d7ff3ffa80574cf5a0238d3e8d9bb757cff2ee1",
    "plan-v1.json": "7f992ca9d82ca19b79eb69a7825ad49600d76e2acd9b5b2bffbc304ff19805db",
    "startup-profile-v1.json": "4af9d440860ad820684f209f04552725229bcc89cca1e098a793fce190388157",
}


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _schema_generated_by_model(
    model: type[BaseModel], checked_schema: dict[str, Any]
) -> dict[str, Any]:
    generated = model.model_json_schema(mode="serialization")
    generated = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": checked_schema["$id"],
        **generated,
    }
    if model is MinimizationResult:
        generated["properties"]["schema_version"] = {
            "const": "sloforge.autopsy.minimization/v1",
            "default": "sloforge.autopsy.minimization/v1",
            "title": "Schema Version",
            "type": "string",
        }
    return generated


def _assert_all_typed_objects_fail_closed(node: object) -> None:
    if isinstance(node, dict):
        if node.get("type") == "object" and "properties" in node:
            assert node.get("additionalProperties") is False
        for child in node.values():
            _assert_all_typed_objects_fail_closed(child)
    elif isinstance(node, list):
        for child in node:
            _assert_all_typed_objects_fail_closed(child)


@pytest.mark.parametrize(
    ("fixture_path", "schema_path", "model", "schema_version"),
    CASES,
    ids=lambda value: value.name if isinstance(value, Path) else None,
)
def test_versioned_golden_records_validate_and_round_trip(
    fixture_path: Path,
    schema_path: Path,
    model: type[BaseModel],
    schema_version: str,
) -> None:
    document = _load(fixture_path)
    schema = _load(schema_path)
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(document)

    parsed = model.model_validate_json(fixture_path.read_text(encoding="utf-8"))
    assert parsed.model_dump(mode="json") == document
    assert canonical_json(parsed.model_dump(mode="json")) == canonical_json(document)
    assert document["schema_version"] == schema_version
    assert (
        sha256_bytes(canonical_json(document).encode())
        == EXPECTED_CANONICAL_HASHES[fixture_path.name]
    )


@pytest.mark.parametrize(
    ("fixture_path", "schema_path", "model", "schema_version"),
    CASES,
    ids=lambda value: value.name if isinstance(value, Path) else None,
)
def test_checked_schemas_match_strict_python_models(
    fixture_path: Path,
    schema_path: Path,
    model: type[BaseModel],
    schema_version: str,
) -> None:
    del fixture_path, schema_version
    checked = _load(schema_path)
    assert checked == _schema_generated_by_model(model, checked)
    _assert_all_typed_objects_fail_closed(checked)


@pytest.mark.parametrize(
    ("fixture_path", "schema_path", "model", "schema_version"),
    CASES,
    ids=lambda value: value.name if isinstance(value, Path) else None,
)
def test_evidence_records_reject_unknown_root_fields(
    fixture_path: Path,
    schema_path: Path,
    model: type[BaseModel],
    schema_version: str,
) -> None:
    del schema_version
    document = copy.deepcopy(_load(fixture_path))
    document["unversioned_extension"] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(_load(schema_path)).validate(document)
    with pytest.raises(ValidationError):
        model.model_validate(document, strict=True)


def test_autopsy_and_warmpath_embedded_hashes_match_canonical_sources() -> None:
    minimization = _load(AUTOPSY_FIXTURES / "minimization-v1.json")
    typed_minimization = MinimizationResult.model_validate_json(
        (AUTOPSY_FIXTURES / "minimization-v1.json").read_text(encoding="utf-8")
    )
    assert minimization["bundle_sha256"] == sha256_bytes(
        canonical_json(typed_minimization.minimized_run.model_dump(mode="json")).encode()
    )

    counterfactual = _load(AUTOPSY_FIXTURES / "counterfactual-v1.json")
    simulation_input = {"seed": 2026, "counterfactuals": []}
    assert counterfactual["simulation_input_sha256"] == sha256_bytes(
        canonical_json(simulation_input).encode()
    )

    graph = _load(WARMPATH_FIXTURES / "artifact-graph-v1.json")
    profile = _load(WARMPATH_FIXTURES / "startup-profile-v1.json")
    plan = _load(WARMPATH_FIXTURES / "plan-v1.json")
    assert plan["graph_hash"] == sha256_bytes(canonical_json(graph).encode())
    assert plan["profile_hash"] == sha256_bytes(canonical_json(profile).encode())

    for measurement in profile["measurements"]:
        raw = {
            "artifact_id": measurement["artifact_id"],
            "tier_id": measurement["tier_id"],
            "stage": measurement["stage"],
            "warmups": measurement["warmup_count"],
            "samples_ms": measurement["raw_samples_ms"],
            "environment_fingerprint": measurement["environment_fingerprint"],
            "invocation": measurement["invocation"],
        }
        assert measurement["artifact_hash"] == sha256_bytes(canonical_json(raw).encode())

    execution = _load(WARMPATH_FIXTURES / "execution-v1.json")
    execution_body = {key: value for key, value in execution.items() if key != "artifact_hash"}
    assert execution["artifact_hash"] == sha256_bytes(canonical_json(execution_body).encode())


def test_no_autopsy_or_warmpath_legacy_fixture_without_a_migration() -> None:
    fixture_names = {
        path.name
        for directory in (AUTOPSY_FIXTURES, WARMPATH_FIXTURES)
        for path in directory.glob("*.json")
    }
    assert not any("alpha" in name or "legacy" in name for name in fixture_names)
