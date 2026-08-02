"""Bounded, atomic persistence for the restart-safe evolution controller."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .models import EvolutionSnapshot, PersistedEvolutionState

MAX_STATE_BYTES = 4 * 1024 * 1024


class EvolutionPersistenceError(RuntimeError):
    """Raised for missing, oversized, malformed, or tampered controller state."""


def _canonical_payload(snapshot: EvolutionSnapshot) -> bytes:
    value: Any = snapshot.model_dump(mode="json")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


class EvolutionStore:
    """Single-file atomic state store with a content digest and bounded reads."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @property
    def exists(self) -> bool:
        return self.path.is_file()

    def save(self, snapshot: EvolutionSnapshot) -> None:
        payload = _canonical_payload(snapshot)
        digest = hashlib.sha256(payload).hexdigest()
        envelope = PersistedEvolutionState(payload_sha256=digest, payload=snapshot)
        encoded = envelope.model_dump_json(indent=2).encode()
        if len(encoded) > MAX_STATE_BYTES:
            raise EvolutionPersistenceError("evolution state exceeds the persistence size bound")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        try:
            with temporary.open("wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            directory = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)

    def load(self) -> EvolutionSnapshot:
        try:
            size = self.path.stat().st_size
        except FileNotFoundError as error:
            raise EvolutionPersistenceError("evolution state does not exist") from error
        if size > MAX_STATE_BYTES:
            raise EvolutionPersistenceError("persisted evolution state exceeds the read bound")
        try:
            envelope = PersistedEvolutionState.model_validate_json(
                self.path.read_bytes(), strict=True
            )
        except (OSError, ValueError) as error:
            raise EvolutionPersistenceError("persisted evolution state is invalid") from error
        actual = hashlib.sha256(_canonical_payload(envelope.payload)).hexdigest()
        if actual != envelope.payload_sha256:
            raise EvolutionPersistenceError("persisted evolution state digest mismatch")
        return envelope.payload
