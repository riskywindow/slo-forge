"""Explicit, lossless migrations for the Genesis v1 wire contract."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .models import API_VERSION, SCHEMA_VERSION


class GenesisMigrationError(ValueError):
    """A document cannot be migrated without guessing its meaning."""


_KINDS = {"InferenceGenome", "Transformation", "Candidate", "Counterexample"}


def _rename(result: dict[str, Any], old: str, new: str) -> None:
    if old not in result:
        return
    if new in result:
        raise GenesisMigrationError(f"legacy document contains both {old!r} and {new!r}")
    result[new] = result.pop(old)


def migrate_document(document: dict[str, Any]) -> dict[str, Any]:
    """Return a current v1 document without mutating the source.

    The sole historical format supported by Genesis is the internal v1alpha1
    fixture used before the stable names were frozen. Unknown versions and
    kinds are rejected rather than interpreted optimistically.
    """

    result = deepcopy(document)
    if result.get("schema_version") == SCHEMA_VERSION:
        if result.get("kind") not in _KINDS:
            raise GenesisMigrationError(f"unsupported Genesis kind: {result.get('kind')!r}")
        return result
    legacy = result.get("version") or result.get("schema_version")
    if legacy not in {"v1alpha1", "0.1.0"}:
        raise GenesisMigrationError(f"unsupported Genesis schema version: {legacy!r}")
    result.pop("version", None)
    result["schema_version"] = SCHEMA_VERSION
    result["api_version"] = API_VERSION
    kind = result.get("kind")
    kind_aliases = {
        "inference_genome": "InferenceGenome",
        "transformation": "Transformation",
        "candidate": "Candidate",
        "counterexample": "Counterexample",
    }
    if kind in kind_aliases:
        result["kind"] = kind_aliases[kind]
    if result.get("kind") not in _KINDS:
        raise GenesisMigrationError(f"unsupported Genesis kind: {kind!r}")
    if result["kind"] == "InferenceGenome":
        for region in (
            "workflow",
            "request",
            "serving",
            "state",
            "distributed",
            "tensor",
            "kernel",
            "recovery",
        ):
            _rename(result, f"{region}_genome", region)
    elif result["kind"] == "Transformation":
        _rename(result, "id", "transformation_id")
        _rename(result, "verification", "verification_obligations")
    elif result["kind"] == "Candidate":
        _rename(result, "id", "candidate_id")
        _rename(result, "events", "lifecycle")
    else:
        _rename(result, "id", "counterexample_id")
        _rename(result, "command", "reproduction")
    return result
