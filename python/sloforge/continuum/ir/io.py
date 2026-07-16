"""Bounded, migration-aware readers for Continuum ABI documents."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, TypeVar, cast

from pydantic import BaseModel, ValidationError

from .canonical import CapsuleValidationError, canonical_json, validate_capsule
from .migrations import ContinuumMigrationError, migrate_document
from .models import (
    CompatibilityReport,
    ContinuumDocument,
    ExecutionStateCapsule,
    LogicalStateSchema,
    MigrationPlan,
    MigrationVerificationEvidence,
    PhysicalStateLayout,
    StateTransaction,
    StateTransformationIR,
)

MAX_CONTINUUM_IR_BYTES = 128 * 1024 * 1024
T = TypeVar("T", bound=BaseModel)


class ContinuumValidationError(ValueError):
    """A bounded parse, migration, model, or integrity failure."""


_DOCUMENT_MODELS: dict[str, type[BaseModel]] = {
    "LogicalStateSchema": LogicalStateSchema,
    "PhysicalStateLayout": PhysicalStateLayout,
    "ExecutionStateCapsule": ExecutionStateCapsule,
    "CompatibilityReport": CompatibilityReport,
    "StateTransformationIR": StateTransformationIR,
    "MigrationPlan": MigrationPlan,
    "StateTransaction": StateTransaction,
    "MigrationVerificationEvidence": MigrationVerificationEvidence,
}


def _read(source: str | bytes | Path | dict[str, Any]) -> dict[str, Any]:
    try:
        if isinstance(source, dict):
            value: Any = source
        elif isinstance(source, Path):
            with source.open("rb") as handle:
                payload = handle.read(MAX_CONTINUUM_IR_BYTES + 1)
            if len(payload) > MAX_CONTINUUM_IR_BYTES:
                raise ContinuumValidationError("Continuum document exceeds bounded input size")
            value = json.loads(payload)
        else:
            if len(source) > MAX_CONTINUUM_IR_BYTES:
                raise ContinuumValidationError("Continuum document exceeds bounded input size")
            value = json.loads(source)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContinuumValidationError(f"cannot read Continuum document: {error}") from error
    if not isinstance(value, dict):
        raise ContinuumValidationError("Continuum document must be a JSON object")
    return value


def _load(model: type[T], source: str | bytes | Path | dict[str, Any], migrate: bool) -> T:
    try:
        raw = _read(source)
        if migrate:
            raw = migrate_document(raw)
        document = model.model_validate_json(json.dumps(raw, allow_nan=False))
        if isinstance(document, ExecutionStateCapsule):
            validate_capsule(document)
        return document
    except (ValidationError, ContinuumMigrationError, CapsuleValidationError, ValueError) as error:
        if isinstance(error, ContinuumValidationError):
            raise
        raise ContinuumValidationError(f"invalid {model.__name__}: {error}") from error


def load_document(
    source: str | bytes | Path | dict[str, Any], *, migrate: bool = True
) -> ContinuumDocument:
    raw = _read(source)
    if migrate:
        try:
            raw = migrate_document(raw)
        except ContinuumMigrationError as error:
            raise ContinuumValidationError(str(error)) from error
    kind = raw.get("kind")
    model = _DOCUMENT_MODELS.get(str(kind))
    if model is None:
        raise ContinuumValidationError(f"unsupported Continuum document kind: {kind!r}")
    return cast(ContinuumDocument, _load(model, raw, migrate=False))


def load_logical_state(
    source: str | bytes | Path | dict[str, Any], *, migrate: bool = True
) -> LogicalStateSchema:
    return _load(LogicalStateSchema, source, migrate)


def load_physical_layout(
    source: str | bytes | Path | dict[str, Any], *, migrate: bool = True
) -> PhysicalStateLayout:
    return _load(PhysicalStateLayout, source, migrate)


def load_capsule(
    source: str | bytes | Path | dict[str, Any], *, migrate: bool = True
) -> ExecutionStateCapsule:
    return _load(ExecutionStateCapsule, source, migrate)


def save_document(path: Path, document: ContinuumDocument) -> str:
    """Atomically publish canonical JSON and return its SHA-256 identifier."""

    payload = canonical_json(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    from .canonical import canonical_hash

    return canonical_hash(document)
