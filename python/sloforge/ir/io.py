"""Safe load/save helpers for versioned IR documents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .canonical import write_canonical
from .errors import IRValidationError
from .migrations import migrate_document
from .models import DeploymentPlan, EvidenceBundle

MAX_IR_BYTES = 64 * 1024 * 1024


def _read_json(source: str | bytes | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(source, dict):
        return source
    try:
        if isinstance(source, Path):
            with source.open("rb") as handle:
                payload = handle.read(MAX_IR_BYTES + 1)
            if len(payload) > MAX_IR_BYTES:
                raise IRValidationError(
                    f"cannot read IR JSON: document exceeds {MAX_IR_BYTES} byte safety limit"
                )
            value = json.loads(payload)
        elif isinstance(source, bytes):
            value = json.loads(source)
        else:
            value = json.loads(source)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise IRValidationError(f"cannot read IR JSON: {error}") from error
    if not isinstance(value, dict):
        raise IRValidationError("IR document must be a JSON object")
    return value


def load_deployment_plan(
    source: str | bytes | Path | dict[str, Any], *, migrate: bool = True
) -> DeploymentPlan:
    raw = _read_json(source)
    if migrate:
        raw = migrate_document(raw)
    try:
        # Strict models still accept JSON-native representations of datetimes,
        # enums, and immutable sequences when validation starts from JSON.
        return DeploymentPlan.model_validate_json(json.dumps(raw, allow_nan=False))
    except ValidationError as error:
        raise IRValidationError(f"invalid DeploymentPlan: {error}") from error


def load_evidence_bundle(
    source: str | bytes | Path | dict[str, Any], *, migrate: bool = True
) -> EvidenceBundle:
    raw = _read_json(source)
    if migrate:
        raw = migrate_document(raw)
    try:
        return EvidenceBundle.model_validate_json(json.dumps(raw, allow_nan=False))
    except ValidationError as error:
        raise IRValidationError(f"invalid EvidenceBundle: {error}") from error


def save_deployment_plan(path: Path, plan: DeploymentPlan) -> None:
    write_canonical(plan, path)


def save_evidence_bundle(path: Path, bundle: EvidenceBundle) -> None:
    write_canonical(bundle, path)
