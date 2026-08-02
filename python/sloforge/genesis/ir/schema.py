"""JSON Schema generation from the canonical Python wire models."""

from __future__ import annotations

import json
from pathlib import Path

from .models import Candidate, Counterexample, InferenceGenome, Transformation


def write_json_schemas(repository_root: Path) -> tuple[Path, ...]:
    targets = (
        (
            InferenceGenome,
            repository_root / "schemas/inference_genome/inference-genome-v1.schema.json",
        ),
        (Transformation, repository_root / "schemas/transformation/transformation-v1.schema.json"),
        (Candidate, repository_root / "schemas/candidate/candidate-v1.schema.json"),
        (Counterexample, repository_root / "schemas/counterexample/counterexample-v1.schema.json"),
    )
    paths: list[Path] = []
    for model, path in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(model.model_json_schema(mode="serialization"), indent=2, sort_keys=True)
            + "\n"
        )
        paths.append(path)
    return tuple(paths)
