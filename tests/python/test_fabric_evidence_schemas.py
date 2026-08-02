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
    "comparison-v1.json": "603a1e0e8f43b0930ac7a5871e7ed4304dc424ebc21cd8ab239d22655ab364ef",
    "counterfactual-v1.json": "0eaf9b00ce2f35d67fa9a9affc5ed64e29e179a3c974efa65d049919e6cbab67",
    "diagnosis-v1.json": "eba9fd2fa39367842d3db99c4203df1a577d6480aaa9898c7c3a1e7338cf4d90",
    "event-v1.json": "7358842c12ce67595e789860a094a9e68ed73b3ff8dc03830a617be44ba7a6aa",
    "minimization-v1.json": "2f63a51455263503296bd4ab75b69d2aab8e52bb029a3e2579e95509b62155a9",
    "run-v1.json": "dba373186e9bedc14113038886ba81526544669e0e1c81366a374d3aa703b053",
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
