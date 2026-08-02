"""Bounded, atomic persistence for the restart-safe evolution controller."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import stat
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
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

    def __init__(
        self,
        path: Path,
        *,
        lock_timeout_seconds: float = 5.0,
        lock_poll_seconds: float = 0.01,
    ) -> None:
        if (
            not math.isfinite(lock_timeout_seconds)
            or lock_timeout_seconds <= 0
            or not math.isfinite(lock_poll_seconds)
            or lock_poll_seconds <= 0
            or lock_poll_seconds > lock_timeout_seconds
        ):
            raise ValueError("evolution-store lock bounds must be finite, positive, and ordered")
        self.path = path
        self.lock_timeout_seconds = lock_timeout_seconds
        self.lock_poll_seconds = lock_poll_seconds

    @property
    def _lock_path(self) -> Path:
        return self.path.with_name(f".{self.path.name}.lock")

    @contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.parent.is_symlink() or not self.path.parent.is_dir():
            raise EvolutionPersistenceError("evolution state parent must be a real directory")
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self._lock_path, flags, 0o600)
        except OSError as error:
            raise EvolutionPersistenceError("evolution state lock is unavailable") from error
        locked = False
        try:
            deadline = time.monotonic() + self.lock_timeout_seconds
            while True:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    locked = True
                    break
                except BlockingIOError as error:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise EvolutionPersistenceError(
                            "evolution state lock acquisition timed out"
                        ) from error
                    time.sleep(min(self.lock_poll_seconds, remaining))
            yield
        finally:
            if locked:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    @property
    def exists(self) -> bool:
        if self.path.is_symlink():
            raise EvolutionPersistenceError("evolution state path must not be a symlink")
        return self.path.is_file()

    def save(self, snapshot: EvolutionSnapshot, *, expected_sequence: int | None) -> None:
        """Create or compare-and-swap a snapshot under an interprocess lock.

        ``expected_sequence=None`` is the creation operation and fails if state
        already exists. Updates must name the exact persisted sequence they
        replace, preventing two restored controllers from losing each other's
        transitions.
        """

        if self.path.is_symlink():
            raise EvolutionPersistenceError("evolution state path must not be a symlink")
        payload = _canonical_payload(snapshot)
        digest = hashlib.sha256(payload).hexdigest()
        envelope = PersistedEvolutionState(payload_sha256=digest, payload=snapshot)
        encoded = envelope.model_dump_json(indent=2).encode()
        if len(encoded) > MAX_STATE_BYTES:
            raise EvolutionPersistenceError("evolution state exceeds the persistence size bound")
        with self._exclusive_lock():
            if expected_sequence is None:
                if self.path.exists():
                    raise EvolutionPersistenceError(
                        "evolution state compare-and-swap expected an absent file"
                    )
            else:
                current = self._load_unlocked()
                if current.sequence != expected_sequence:
                    raise EvolutionPersistenceError(
                        "evolution state compare-and-swap sequence mismatch"
                    )
            temporary: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    dir=self.path.parent,
                    prefix=f".{self.path.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as handle:
                    temporary = Path(handle.name)
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.chmod(temporary, 0o600)
                os.replace(temporary, self.path)
                temporary = None
                directory = os.open(self.path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)

    def load(self) -> EvolutionSnapshot:
        with self._exclusive_lock():
            return self._load_unlocked()

    def _load_unlocked(self) -> EvolutionSnapshot:
        if self.path.is_symlink():
            raise EvolutionPersistenceError("evolution state path must not be a symlink")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.path, flags)
        except FileNotFoundError as error:
            raise EvolutionPersistenceError("evolution state does not exist") from error
        except OSError as error:
            raise EvolutionPersistenceError("evolution state cannot be opened safely") from error
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise EvolutionPersistenceError("evolution state must be a regular file")
            if metadata.st_size > MAX_STATE_BYTES:
                raise EvolutionPersistenceError("persisted evolution state exceeds the read bound")
            with os.fdopen(descriptor, "rb", closefd=True) as handle:
                descriptor = -1
                encoded = handle.read(MAX_STATE_BYTES + 1)
            if len(encoded) > MAX_STATE_BYTES:
                raise EvolutionPersistenceError(
                    "persisted evolution state exceeded the read bound while reading"
                )
            envelope = PersistedEvolutionState.model_validate_json(encoded, strict=True)
        except EvolutionPersistenceError:
            raise
        except (OSError, ValueError) as error:
            raise EvolutionPersistenceError("persisted evolution state is invalid") from error
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        actual = hashlib.sha256(_canonical_payload(envelope.payload)).hexdigest()
        if actual != envelope.payload_sha256:
            raise EvolutionPersistenceError("persisted evolution state digest mismatch")
        return envelope.payload
