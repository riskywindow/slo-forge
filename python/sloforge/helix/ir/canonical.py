"""Cross-language canonical JSON and SHA-256 helpers for Helix IR."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel

from .models import Digest


def canonical_json(document: BaseModel | dict[str, Any]) -> bytes:
    """Encode sorted UTF-8 JSON without insignificant whitespace or NaN values."""

    value = document.model_dump(mode="json") if isinstance(document, BaseModel) else document
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def canonical_hash(document: BaseModel | dict[str, Any]) -> str:
    """Return the lowercase SHA-256 of :func:`canonical_json`."""

    return hashlib.sha256(canonical_json(document)).hexdigest()


def canonical_digest(document: BaseModel | dict[str, Any]) -> Digest:
    return Digest(value=canonical_hash(document))
