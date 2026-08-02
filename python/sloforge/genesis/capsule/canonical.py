"""Canonical serialization and sealing for Genesis capsule manifests."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel

from .models import Digest, GenesisCapsule


def canonical_json(value: BaseModel | dict[str, Any]) -> bytes:
    document = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def manifest_payload(capsule: GenesisCapsule) -> bytes:
    """Canonical bytes covered by ``capsule_digest``.

    The digest field is replaced by null to avoid a self-referential hash while
    retaining an unambiguous serialized location for it.
    """

    return canonical_json(capsule.model_copy(update={"capsule_digest": None}))


def calculate_capsule_digest(capsule: GenesisCapsule) -> Digest:
    return Digest(value=hashlib.sha256(manifest_payload(capsule)).hexdigest())


def seal_capsule(capsule: GenesisCapsule) -> GenesisCapsule:
    """Return an immutable copy bearing its canonical content digest."""

    return capsule.model_copy(update={"capsule_digest": calculate_capsule_digest(capsule)})
