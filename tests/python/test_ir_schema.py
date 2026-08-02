from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from sloforge.ir import DeploymentPlan, EvidenceBundle, write_json_schema

FIXTURES = Path(__file__).parents[1] / "fixtures" / "ir"


def test_generated_schemas_validate_golden_documents() -> None:
    plan = json.loads((FIXTURES / "deployment-plan-v1.json").read_text())
    evidence = json.loads((FIXTURES / "evidence-bundle-v1.json").read_text())
    jsonschema.Draft202012Validator(DeploymentPlan.model_json_schema()).validate(plan)
    jsonschema.Draft202012Validator(EvidenceBundle.model_json_schema()).validate(evidence)


def test_checked_in_schemas_match_generator(tmp_path: Path) -> None:
    generated = write_json_schema(tmp_path)
    checked_in = (
        Path(__file__).parents[2] / "schemas" / "deployment-plan-v1.schema.json",
        Path(__file__).parents[2] / "schemas" / "evidence-bundle-v1.schema.json",
    )
    for actual, expected in zip(generated, checked_in, strict=True):
        assert json.loads(actual.read_text()) == json.loads(expected.read_text())


def test_schema_rejects_unqualified_extension_keys() -> None:
    plan = json.loads((FIXTURES / "deployment-plan-v1.json").read_text())
    plan["extensions"] = {"unqualified": True}
    validator = jsonschema.Draft202012Validator(DeploymentPlan.model_json_schema())
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(plan)
