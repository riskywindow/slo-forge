"""Bounded, strict persistence for checkpoint operation descriptors.

State bytes remain in the tenant-scoped content store.  This descriptor binds the
canonical capsule to the exact CAS publication and its authenticated ancestry.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, cast

from sloforge.continuum.ir import ExecutionStateCapsule
from sloforge.continuum.storage import ChunkRef, StoredManifest

from .checkpoint import verify_checkpoint_artifact
from .models import AncestryProof, CheckpointArtifact

_SCHEMA = "sloforge.continuum.checkpoint-artifact/v1"
_MAX_ANCESTRY_DEPTH = 64
_MAX_DESCRIPTOR_BYTES = 32 * 1024 * 1024
_KEYS = {
    "schema_version",
    "capsule",
    "store_manifest",
    "chunk_references",
    "ancestry",
    "changed_segment_ids",
    "parent",
}


def _document(artifact: CheckpointArtifact, *, depth: int) -> dict[str, object]:
    if depth > _MAX_ANCESTRY_DEPTH:
        raise ValueError("checkpoint ancestry exceeds the serialization bound")
    return {
        "schema_version": _SCHEMA,
        "capsule": artifact.capsule.model_dump(mode="json"),
        "store_manifest": artifact.store_manifest.model_dump(mode="json"),
        "chunk_references": [item.model_dump(mode="json") for item in artifact.chunk_references],
        "ancestry": artifact.ancestry.model_dump(mode="json"),
        "changed_segment_ids": list(artifact.changed_segment_ids),
        "parent": (
            _document(artifact.parent, depth=depth + 1) if artifact.parent is not None else None
        ),
    }


def checkpoint_artifact_document(artifact: CheckpointArtifact) -> dict[str, object]:
    """Return a JSON-compatible descriptor after verifying every ancestry edge."""

    verify_checkpoint_artifact(artifact)
    return _document(artifact, depth=0)


def _parse(document: object, *, depth: int) -> CheckpointArtifact:
    if depth > _MAX_ANCESTRY_DEPTH:
        raise ValueError("checkpoint ancestry exceeds the deserialization bound")
    if not isinstance(document, dict) or set(document) != _KEYS:
        raise ValueError("checkpoint artifact has missing or unknown fields")
    typed = cast(dict[str, Any], document)
    if typed["schema_version"] != _SCHEMA:
        raise ValueError("unsupported checkpoint artifact schema")
    changed = typed["changed_segment_ids"]
    chunks = typed["chunk_references"]
    if not isinstance(changed, list) or any(not isinstance(item, str) for item in changed):
        raise ValueError("checkpoint changed-segment identifiers are invalid")
    if not isinstance(chunks, list):
        raise ValueError("checkpoint chunk references must be an array")
    parent_document = typed["parent"]
    parent = _parse(parent_document, depth=depth + 1) if parent_document is not None else None
    artifact = CheckpointArtifact(
        capsule=ExecutionStateCapsule.model_validate_json(
            json.dumps(typed["capsule"]), strict=True
        ),
        store_manifest=StoredManifest.model_validate_json(
            json.dumps(typed["store_manifest"]), strict=True
        ),
        chunk_references=tuple(
            ChunkRef.model_validate_json(json.dumps(item), strict=True) for item in chunks
        ),
        ancestry=AncestryProof.model_validate_json(json.dumps(typed["ancestry"]), strict=True),
        changed_segment_ids=tuple(changed),
        parent=parent,
    )
    verify_checkpoint_artifact(artifact)
    return artifact


def load_checkpoint_artifact(path: Path) -> CheckpointArtifact:
    """Load and verify a bounded operation descriptor without reading state bytes."""

    if path.stat().st_size > _MAX_DESCRIPTOR_BYTES:
        raise ValueError("checkpoint artifact descriptor exceeds 32 MiB")
    try:
        document = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("checkpoint artifact is not valid JSON") from error
    return _parse(document, depth=0)


def save_checkpoint_artifact(path: Path, artifact: CheckpointArtifact) -> None:
    """Atomically persist a checkpoint descriptor; payloads remain in the CAS."""

    encoded = json.dumps(
        checkpoint_artifact_document(artifact),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > _MAX_DESCRIPTOR_BYTES:
        raise ValueError("checkpoint artifact descriptor exceeds 32 MiB")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".continuum-checkpoint-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
