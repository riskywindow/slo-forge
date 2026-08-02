"""JSON Schema generation for checked-in Fabric IR contracts."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from .models import FabricProfile, ModelGraph, PhysicalExecutionPlan, RecoveryPlan, TopologyGraph

SCHEMA_MODELS: tuple[tuple[str, type[BaseModel]], ...] = (
    ("topology-graph-v1.schema.json", TopologyGraph),
    ("model-graph-v1.schema.json", ModelGraph),
    ("fabric-profile-v1.schema.json", FabricProfile),
    ("physical-execution-plan-v1.schema.json", PhysicalExecutionPlan),
    ("recovery-plan-v1.schema.json", RecoveryPlan),
)


def write_json_schemas(output_directory: Path) -> tuple[Path, ...]:
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for filename, model in SCHEMA_MODELS:
        schema = model.model_json_schema()
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        schema["$id"] = f"urn:sloforge:fabric:{filename.removesuffix('.schema.json')}"
        output = output_directory / filename
        output.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        outputs.append(output)
    return tuple(outputs)
