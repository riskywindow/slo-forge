"""Bounded, migration-aware Fabric IR readers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from sloforge.ir import write_canonical

from .migrations import FabricMigrationError, migrate_document
from .models import FabricProfile, ModelGraph, PhysicalExecutionPlan, RecoveryPlan, TopologyGraph

MAX_FABRIC_IR_BYTES = 64 * 1024 * 1024
FabricDocument = TopologyGraph | ModelGraph | FabricProfile | PhysicalExecutionPlan | RecoveryPlan
T = TypeVar("T", bound=BaseModel)


class FabricValidationError(ValueError):
    """A bounded parse, migration, or model validation failure."""


def _read(source: str | bytes | Path | dict[str, Any]) -> dict[str, Any]:
    try:
        if isinstance(source, dict):
            value: Any = source
        elif isinstance(source, Path):
            with source.open("rb") as handle:
                payload = handle.read(MAX_FABRIC_IR_BYTES + 1)
            if len(payload) > MAX_FABRIC_IR_BYTES:
                raise FabricValidationError("Fabric document exceeds bounded input size")
            value = json.loads(payload)
        else:
            value = json.loads(source)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FabricValidationError(f"cannot read Fabric document: {error}") from error
    if not isinstance(value, dict):
        raise FabricValidationError("Fabric document must be a JSON object")
    return value


def _load(model: type[T], source: str | bytes | Path | dict[str, Any], migrate: bool) -> T:
    try:
        raw = _read(source)
        if migrate:
            raw = migrate_document(raw)
        return model.model_validate_json(json.dumps(raw, allow_nan=False))
    except (ValidationError, FabricMigrationError, ValueError) as error:
        if isinstance(error, FabricValidationError):
            raise
        raise FabricValidationError(f"invalid {model.__name__}: {error}") from error


def load_topology_graph(
    source: str | bytes | Path | dict[str, Any], *, migrate: bool = True
) -> TopologyGraph:
    return _load(TopologyGraph, source, migrate)


def load_model_graph(
    source: str | bytes | Path | dict[str, Any], *, migrate: bool = True
) -> ModelGraph:
    return _load(ModelGraph, source, migrate)


def load_fabric_profile(
    source: str | bytes | Path | dict[str, Any], *, migrate: bool = True
) -> FabricProfile:
    return _load(FabricProfile, source, migrate)


def load_physical_execution_plan(
    source: str | bytes | Path | dict[str, Any], *, migrate: bool = True
) -> PhysicalExecutionPlan:
    return _load(PhysicalExecutionPlan, source, migrate)


def load_recovery_plan(
    source: str | bytes | Path | dict[str, Any], *, migrate: bool = True
) -> RecoveryPlan:
    return _load(RecoveryPlan, source, migrate)


def save_topology_graph(path: Path, document: TopologyGraph) -> None:
    write_canonical(document, path)


def save_model_graph(path: Path, document: ModelGraph) -> None:
    write_canonical(document, path)


def save_fabric_profile(path: Path, document: FabricProfile) -> None:
    write_canonical(document, path)


def save_physical_execution_plan(path: Path, document: PhysicalExecutionPlan) -> None:
    write_canonical(document, path)


def save_recovery_plan(path: Path, document: RecoveryPlan) -> None:
    write_canonical(document, path)
