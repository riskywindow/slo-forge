"""Safe loading and canonical storage for ForgeCI matrices."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from yaml.tokens import AliasToken, AnchorToken

from sloforge.forgeci.models import BenchmarkMatrix

MAX_MATRIX_BYTES = 4 * 1024 * 1024


def _bounded_document(path: Path) -> str:
    with path.open("rb") as handle:
        payload = handle.read(MAX_MATRIX_BYTES + 1)
    if len(payload) > MAX_MATRIX_BYTES:
        raise ValueError("ForgeCI matrix exceeds 4 MiB")
    return payload.decode("utf-8")


def _reject_yaml_references(raw: str) -> None:
    if any(isinstance(token, (AliasToken, AnchorToken)) for token in yaml.scan(raw)):
        raise ValueError("ForgeCI matrices do not permit YAML anchors or aliases")


def load_matrix(path: Path) -> BenchmarkMatrix:
    """Load JSON or safe YAML through Pydantic's strict JSON conversion path."""

    raw = _bounded_document(path)
    if path.suffix.lower() == ".json":
        decoded = json.loads(raw)
    else:
        _reject_yaml_references(raw)
        decoded = yaml.safe_load(raw)
    return BenchmarkMatrix.model_validate_json(
        json.dumps(decoded, sort_keys=True, separators=(",", ":"))
    )


def write_matrix(matrix: BenchmarkMatrix, path: Path) -> str:
    """Write canonical JSON and return its SHA-256 digest."""

    payload = matrix.model_dump_json(indent=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload + "\n", encoding="utf-8")
    return hashlib.sha256((payload + "\n").encode()).hexdigest()
