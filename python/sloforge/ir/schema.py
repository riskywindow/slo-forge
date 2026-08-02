"""JSON Schema generation for the checked-in wire contracts."""

from __future__ import annotations

import json
from pathlib import Path

from .models import DeploymentPlan, EvidenceBundle


def _wire_schema(
    model: type[DeploymentPlan] | type[EvidenceBundle], schema_id: str
) -> dict[str, object]:
    schema: dict[str, object] = model.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = schema_id
    return schema


def write_json_schema(output_directory: Path) -> tuple[Path, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    plan_path = output_directory / "deployment-plan-v1.schema.json"
    evidence_path = output_directory / "evidence-bundle-v1.schema.json"
    plan_path.write_text(
        json.dumps(
            _wire_schema(DeploymentPlan, "urn:sloforge:schema:deployment-plan:1"),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    evidence_path.write_text(
        json.dumps(
            _wire_schema(EvidenceBundle, "urn:sloforge:schema:evidence-bundle:1"),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return plan_path, evidence_path
