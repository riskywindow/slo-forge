"""Cross-language canonical JSON profile for Genesis documents."""

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
    """Serialize using sorted keys, UTF-8, no whitespace, and finite numbers."""

    value = document.model_dump(mode="json") if isinstance(document, BaseModel) else document
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def canonical_hash(document: BaseModel | dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(document)).hexdigest()


def canonical_digest(document: BaseModel | dict[str, Any]) -> ArtifactDigest:
    return ArtifactDigest(value=canonical_hash(document))


def write_canonical(document: BaseModel, path: Path) -> ArtifactDigest:
    """Atomically persist a canonical document and return its content digest."""

    payload = canonical_json(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return ArtifactDigest(value=hashlib.sha256(payload).hexdigest())
