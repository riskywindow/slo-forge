"""Safe loading and canonical storage for ForgeCI matrices."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from sloforge.forgeci.models import BenchmarkMatrix


def load_matrix(path: Path) -> BenchmarkMatrix:
    """Load JSON or safe YAML through Pydantic's strict JSON conversion path."""

    raw = path.read_text(encoding="utf-8")
    decoded = json.loads(raw) if path.suffix.lower() == ".json" else yaml.safe_load(raw)
    return BenchmarkMatrix.model_validate_json(
        json.dumps(decoded, sort_keys=True, separators=(",", ":"))
    )


def write_matrix(matrix: BenchmarkMatrix, path: Path) -> str:
    """Write canonical JSON and return its SHA-256 digest."""

    payload = matrix.model_dump_json(indent=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload + "\n", encoding="utf-8")
    return hashlib.sha256((payload + "\n").encode()).hexdigest()
