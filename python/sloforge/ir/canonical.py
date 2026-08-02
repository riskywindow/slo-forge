"""Canonical serialization and content addressing for SLOForge IR documents."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .models import ArtifactDigest


def canonical_json(document: BaseModel | dict[str, Any]) -> bytes:
    """Return deterministic UTF-8 JSON suitable for cross-language hashing.

    This is a deliberately small canonical profile: object keys are sorted,
    insignificant whitespace is removed, UTF-8 is preserved, and non-finite
    values are rejected.  IR validation prevents ambiguous extension values.
    """

    value = document.model_dump(mode="json") if isinstance(document, BaseModel) else document
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_hash(document: BaseModel | dict[str, Any]) -> str:
    """Return the lowercase SHA-256 hex content identifier."""

    return hashlib.sha256(canonical_json(document)).hexdigest()


def canonical_digest(document: BaseModel | dict[str, Any]) -> ArtifactDigest:
    """Return the canonical hash wrapped as a typed artifact digest."""

    return ArtifactDigest(value=canonical_hash(document))


def write_canonical(document: BaseModel, path: Path) -> ArtifactDigest:
    payload = canonical_json(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
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
    return ArtifactDigest(value=hashlib.sha256(payload).hexdigest())
