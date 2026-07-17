"""Bounded strict readers for versioned Helix IR documents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ValidationError

from .models import (
    BranchGroup,
    BranchPoint,
    BranchWorkloadTrace,
    CreditAssignmentEvidence,
    EnvironmentStateCapsule,
    HelixDocument,
    LearningTransaction,
    PolicyEpoch,
    PolicyPromotionCapsule,
    RewardEvidence,
    StalenessReport,
    StateReuseReport,
    TrainingBatchManifest,
    TrajectoryCapsule,
)

MAX_HELIX_IR_BYTES = 128 * 1024 * 1024


class HelixValidationError(ValueError):
    """A bounded JSON, version, or semantic validation failure."""


_DOCUMENT_MODELS: dict[str, type[BaseModel]] = {
    "PolicyEpoch": PolicyEpoch,
    "EnvironmentStateCapsule": EnvironmentStateCapsule,
    "BranchPoint": BranchPoint,
    "TrajectoryCapsule": TrajectoryCapsule,
    "BranchGroup": BranchGroup,
    "RewardEvidence": RewardEvidence,
    "CreditAssignmentEvidence": CreditAssignmentEvidence,
    "StalenessReport": StalenessReport,
    "StateReuseReport": StateReuseReport,
    "TrainingBatchManifest": TrainingBatchManifest,
    "LearningTransaction": LearningTransaction,
    "PolicyPromotionCapsule": PolicyPromotionCapsule,
    "BranchWorkloadTrace": BranchWorkloadTrace,
}


def _read(source: str | bytes | Path | dict[str, Any]) -> dict[str, Any]:
    try:
        if isinstance(source, dict):
            value: Any = source
        elif isinstance(source, Path):
            with source.open("rb") as handle:
                payload = handle.read(MAX_HELIX_IR_BYTES + 1)
            if len(payload) > MAX_HELIX_IR_BYTES:
                raise HelixValidationError("Helix document exceeds bounded input size")
            value = json.loads(payload)
        else:
            if len(source) > MAX_HELIX_IR_BYTES:
                raise HelixValidationError("Helix document exceeds bounded input size")
            value = json.loads(source)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HelixValidationError(f"cannot read Helix document: {error}") from error
    if not isinstance(value, dict):
        raise HelixValidationError("Helix document must be a JSON object")
    return value


def load_document(source: str | bytes | Path | dict[str, Any]) -> HelixDocument:
    raw = _read(source)
    kind = raw.get("kind")
    model = _DOCUMENT_MODELS.get(str(kind))
    if model is None:
        raise HelixValidationError(f"unsupported Helix document kind: {kind!r}")
    try:
        document = model.model_validate_json(json.dumps(raw, allow_nan=False))
    except (ValidationError, ValueError) as error:
        raise HelixValidationError(f"invalid {model.__name__}: {error}") from error
    return cast(HelixDocument, document)


def load_learning_transaction(
    source: str | bytes | Path | dict[str, Any],
) -> LearningTransaction:
    document = load_document(source)
    if not isinstance(document, LearningTransaction):
        raise HelixValidationError("Helix document is not a LearningTransaction")
    return document
