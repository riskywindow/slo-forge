"""JSON Schema generation from the authoritative Python ABI types."""

from __future__ import annotations

import json
from pathlib import Path

from .models import (
    CompatibilityReport,
    ExecutionStateCapsule,
    LogicalStateSchema,
    MigrationPlan,
    MigrationVerificationEvidence,
    PhysicalStateLayout,
    StateTransaction,
    StateTransformationIR,
)


def schema_documents() -> tuple[tuple[str, dict[str, object]], ...]:
    models = (
        ("logical-state-v1.schema.json", LogicalStateSchema),
        ("physical-state-layout-v1.schema.json", PhysicalStateLayout),
        ("execution-state-capsule-v1.schema.json", ExecutionStateCapsule),
        ("compatibility-report-v1.schema.json", CompatibilityReport),
        ("state-transformation-ir-v1.schema.json", StateTransformationIR),
        ("migration-plan-v1.schema.json", MigrationPlan),
        ("state-transaction-v1.schema.json", StateTransaction),
        ("migration-verification-evidence-v1.schema.json", MigrationVerificationEvidence),
    )
    return tuple((name, model.model_json_schema(mode="serialization")) for name, model in models)


def write_json_schemas(repository_root: Path) -> tuple[Path, ...]:
    target = repository_root / "schemas/continuum"
    target.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, document in schema_documents():
        path = target / name
        path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
        paths.append(path)
    return tuple(paths)
