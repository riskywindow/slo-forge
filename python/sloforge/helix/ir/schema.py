"""JSON Schema generation from authoritative Helix Pydantic models."""

from __future__ import annotations

import json
from pathlib import Path

from .models import (
    BranchGroup,
    BranchPoint,
    BranchWorkloadTrace,
    CreditAssignmentEvidence,
    EnvironmentStateCapsule,
    LearningTransaction,
    PolicyEpoch,
    PolicyPromotionCapsule,
    RewardEvidence,
    StalenessReport,
    StateReuseReport,
    TrainingBatchManifest,
    TrajectoryCapsule,
)


def schema_documents() -> tuple[tuple[str, dict[str, object]], ...]:
    models = (
        ("policy-epoch-v1.schema.json", PolicyEpoch),
        ("environment-state-capsule-v1.schema.json", EnvironmentStateCapsule),
        ("branch-point-v1.schema.json", BranchPoint),
        ("trajectory-capsule-v1.schema.json", TrajectoryCapsule),
        ("branch-group-v1.schema.json", BranchGroup),
        ("reward-evidence-v1.schema.json", RewardEvidence),
        ("credit-assignment-evidence-v1.schema.json", CreditAssignmentEvidence),
        ("staleness-report-v1.schema.json", StalenessReport),
        ("state-reuse-report-v1.schema.json", StateReuseReport),
        ("training-batch-manifest-v1.schema.json", TrainingBatchManifest),
        ("learning-transaction-v1.schema.json", LearningTransaction),
        ("policy-promotion-capsule-v1.schema.json", PolicyPromotionCapsule),
        ("branch-workload-trace-v1.schema.json", BranchWorkloadTrace),
    )
    return tuple((name, model.model_json_schema(mode="serialization")) for name, model in models)


def write_json_schemas(repository_root: Path) -> tuple[Path, ...]:
    target = repository_root / "schemas/helix"
    target.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, document in schema_documents():
        path = target / name
        path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
        paths.append(path)
    return tuple(paths)
