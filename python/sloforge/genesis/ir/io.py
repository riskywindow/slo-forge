"""Bounded, migration-aware readers for trusted Genesis documents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .canonical import write_canonical
from .migrations import GenesisMigrationError, migrate_document
from .models import Candidate, Counterexample, InferenceGenome, Transformation

MAX_GENESIS_IR_BYTES = 64 * 1024 * 1024
T = TypeVar("T", bound=BaseModel)


class GenesisValidationError(ValueError):
    """A bounded parse, migration, or schema validation failure."""


def _read(source: str | bytes | Path | dict[str, Any]) -> dict[str, Any]:
    try:
        if isinstance(source, dict):
            value: Any = source
        elif isinstance(source, Path):
            with source.open("rb") as handle:
                payload = handle.read(MAX_GENESIS_IR_BYTES + 1)
            if len(payload) > MAX_GENESIS_IR_BYTES:
                raise GenesisValidationError("Genesis document exceeds bounded input size")
            value = json.loads(payload)
        else:
            value = json.loads(source)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GenesisValidationError(f"cannot read Genesis document: {error}") from error
    if not isinstance(value, dict):
        raise GenesisValidationError("Genesis document must be a JSON object")
    return value


def _load(model: type[T], source: str | bytes | Path | dict[str, Any], migrate: bool) -> T:
    try:
        raw = _read(source)
        if migrate:
            raw = migrate_document(raw)
        return model.model_validate_json(json.dumps(raw, allow_nan=False))
    except (ValidationError, GenesisMigrationError, ValueError) as error:
        if isinstance(error, GenesisValidationError):
            raise
        raise GenesisValidationError(f"invalid {model.__name__}: {error}") from error


def load_inference_genome(
    source: str | bytes | Path | dict[str, Any], *, migrate: bool = True
) -> InferenceGenome:
    return _load(InferenceGenome, source, migrate)


def load_transformation(
    source: str | bytes | Path | dict[str, Any], *, migrate: bool = True
) -> Transformation:
    return _load(Transformation, source, migrate)


def load_candidate(
    source: str | bytes | Path | dict[str, Any], *, migrate: bool = True
) -> Candidate:
    return _load(Candidate, source, migrate)


def load_counterexample(
    source: str | bytes | Path | dict[str, Any], *, migrate: bool = True
) -> Counterexample:
    return _load(Counterexample, source, migrate)


def save_document(
    path: Path, document: InferenceGenome | Transformation | Candidate | Counterexample
) -> None:
    write_canonical(document, path)
