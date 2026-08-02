"""Canonical serialization and stable hashes for Fabric IR documents."""

from __future__ import annotations

from pydantic import BaseModel

from sloforge.ir import canonical_hash as _canonical_hash
from sloforge.ir import canonical_json as _canonical_json

from .models import assert_finite_document


def canonical_json(document: BaseModel) -> bytes:
    assert_finite_document(document)
    return _canonical_json(document)


def canonical_hash(document: BaseModel) -> str:
    assert_finite_document(document)
    return _canonical_hash(document)
