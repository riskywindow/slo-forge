"""Bounded, immutable publication of Genesis capsule manifests."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from pydantic import ValidationError

from .canonical import calculate_capsule_digest, canonical_json
from .models import GenesisCapsule


class CapsuleIOError(RuntimeError):
    """Raised when a capsule manifest cannot be safely loaded or published."""


def load_capsule(path: Path, *, maximum_bytes: int = 8 * 1024 * 1024) -> GenesisCapsule:
    if maximum_bytes <= 0:
        raise ValueError("maximum_bytes must be positive")
    if path.is_symlink() or not path.is_file():
        raise CapsuleIOError("capsule manifest must be a regular non-symlink file")
    if path.stat().st_size > maximum_bytes:
        raise CapsuleIOError("capsule manifest exceeds maximum_bytes")
    try:
        return GenesisCapsule.model_validate_json(path.read_bytes(), strict=True)
    except (OSError, ValidationError) as exc:
        raise CapsuleIOError(f"capsule manifest is invalid: {exc}") from exc


def publish_capsule(capsule: GenesisCapsule, directory: Path) -> Path:
    """Publish a sealed manifest once under its canonical digest name."""

    if capsule.capsule_digest is None:
        raise CapsuleIOError("an unsealed capsule cannot be published")
    if calculate_capsule_digest(capsule) != capsule.capsule_digest:
        raise CapsuleIOError("a tampered capsule cannot be published")
    if directory.exists() and directory.is_symlink():
        raise CapsuleIOError("capsule directory must not be a symlink")
    directory.mkdir(parents=True, exist_ok=True)
    directory = directory.resolve(strict=True)
    target = directory / f"{capsule.capsule_digest.value}.json"
    payload = canonical_json(capsule) + b"\n"
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_file() or target.read_bytes() != payload:
            raise CapsuleIOError("existing capsule object failed immutable-content verification")
        return target
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            if target.is_symlink() or not target.is_file() or target.read_bytes() != payload:
                raise CapsuleIOError("capsule publication collided with different content") from exc
        temporary.unlink(missing_ok=True)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target
