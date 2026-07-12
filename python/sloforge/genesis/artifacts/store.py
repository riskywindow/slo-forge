"""Small, transactional content-addressed store for Genesis artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from sloforge.genesis.capsule.models import Digest


class ArtifactStoreError(RuntimeError):
    """Raised when CAS integrity or containment cannot be guaranteed."""


@dataclass(frozen=True)
class StoredArtifact:
    digest: Digest
    size_bytes: int
    object_path: Path


class ContentAddressedArtifactStore:
    """Immutable SHA-256 object store with atomic publication.

    Object names are derived from content and never from generated-code input.
    Existing objects are re-hashed before reuse, making local store corruption a
    hard error rather than silently blessing it into a capsule.
    """

    def __init__(self, root: Path, *, maximum_object_bytes: int = 1 << 30) -> None:
        if maximum_object_bytes <= 0:
            raise ValueError("maximum_object_bytes must be positive")
        if root.exists() and root.is_symlink():
            raise ArtifactStoreError("artifact-store root must not be a symlink")
        root.mkdir(parents=True, exist_ok=True)
        self.root = root.resolve(strict=True)
        self.maximum_object_bytes = maximum_object_bytes
        self._objects = self.root / "objects" / "sha256"
        self._temporary = self.root / "tmp"
        self._ensure_directory(self._objects)
        self._ensure_directory(self._temporary)

    def _ensure_directory(self, directory: Path) -> None:
        try:
            relative = directory.relative_to(self.root)
        except ValueError as exc:
            raise ArtifactStoreError("CAS internal directory escapes the store root") from exc
        cursor = self.root
        for part in relative.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise ArtifactStoreError("CAS internal directories must not contain symlinks")
            cursor.mkdir(exist_ok=True)
            if not cursor.is_dir():
                raise ArtifactStoreError("CAS internal path is not a directory")
        try:
            cursor.resolve(strict=True).relative_to(self.root)
        except (OSError, ValueError) as exc:
            raise ArtifactStoreError("CAS internal directory escapes the store root") from exc

    def object_path(self, digest: Digest) -> Path:
        return self._objects / digest.value[:2] / digest.value[2:4] / digest.value

    @staticmethod
    def _hash_file(path: Path) -> tuple[str, int]:
        digest = hashlib.sha256()
        size = 0
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            os.close(descriptor)
            raise ArtifactStoreError(f"CAS object is not a regular file: {path}")
        with os.fdopen(descriptor, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
        return digest.hexdigest(), size

    @staticmethod
    def _reject_symlink_components(path: Path) -> None:
        absolute = path if path.is_absolute() else Path.cwd() / path
        cursor = Path(absolute.anchor)
        for component in absolute.parts[1:]:
            cursor = cursor / component
            if cursor.is_symlink():
                raise ArtifactStoreError(
                    f"materialization destination contains a symlink component: {cursor}"
                )
            if not cursor.exists():
                break

    def _verify_existing(self, path: Path, digest: Digest, size_bytes: int | None) -> int:
        if path.is_symlink() or not path.is_file():
            raise ArtifactStoreError(f"CAS object is not a regular file: {path}")
        actual_digest, actual_size = self._hash_file(path)
        if actual_digest != digest.value:
            raise ArtifactStoreError(f"CAS object failed integrity verification: {digest.value}")
        if size_bytes is not None and actual_size != size_bytes:
            raise ArtifactStoreError(f"CAS object has an unexpected size: {digest.value}")
        return actual_size

    def _publish(self, temporary: Path, digest: Digest, size_bytes: int) -> StoredArtifact:
        target = self.object_path(digest)
        self._ensure_directory(target.parent)
        if target.exists() or target.is_symlink():
            actual_size = self._verify_existing(target, digest, size_bytes)
            temporary.unlink(missing_ok=True)
            return StoredArtifact(digest=digest, size_bytes=actual_size, object_path=target)
        os.chmod(temporary, 0o444)
        try:
            os.replace(temporary, target)
        except OSError:
            if not target.exists():
                raise
            self._verify_existing(target, digest, size_bytes)
            temporary.unlink(missing_ok=True)
        return StoredArtifact(digest=digest, size_bytes=size_bytes, object_path=target)

    def put_stream(self, stream: BinaryIO) -> StoredArtifact:
        digest = hashlib.sha256()
        size = 0
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=self._temporary, delete=False) as handle:
                temporary = Path(handle.name)
                while True:
                    chunk = stream.read(1024 * 1024)
                    if not chunk:
                        break
                    if not isinstance(chunk, bytes):
                        raise TypeError("artifact streams must produce bytes")
                    size += len(chunk)
                    if size > self.maximum_object_bytes:
                        raise ArtifactStoreError("artifact exceeds configured maximum_object_bytes")
                    digest.update(chunk)
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            value = Digest(value=digest.hexdigest())
            published = self._publish(temporary, value, size)
            temporary = None
            return published
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def put_bytes(self, data: bytes) -> StoredArtifact:
        from io import BytesIO

        return self.put_stream(BytesIO(data))

    def put_json(self, document: object) -> StoredArtifact:
        payload = json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return self.put_bytes(payload)

    def put_file(self, source: Path) -> StoredArtifact:
        if source.is_symlink() or not source.is_file():
            raise ArtifactStoreError("artifact source must be a regular non-symlink file")
        with source.open("rb") as handle:
            return self.put_stream(handle)

    def verify(self, digest: Digest) -> bool:
        path = self.object_path(digest)
        try:
            self._verify_existing(path, digest, None)
        except (ArtifactStoreError, OSError):
            return False
        return True

    def read_bytes(self, digest: Digest, *, maximum_bytes: int | None = None) -> bytes:
        path = self.object_path(digest)
        size = self._verify_existing(path, digest, None)
        limit = self.maximum_object_bytes if maximum_bytes is None else maximum_bytes
        if limit < 0:
            raise ValueError("maximum_bytes cannot be negative")
        if size > limit:
            raise ArtifactStoreError("artifact exceeds requested read limit")
        return path.read_bytes()

    def materialize(self, digest: Digest, destination: Path) -> Path:
        """Copy a verified object to an explicit, non-existing capsule path."""

        source = self.object_path(digest)
        expected_size = self._verify_existing(source, digest, None)
        self._reject_symlink_components(destination)
        if destination.exists() or destination.is_symlink():
            raise ArtifactStoreError("materialization destination already exists")
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._reject_symlink_components(destination)
        temporary: Path | None = None
        try:
            copied_digest = hashlib.sha256()
            copied_size = 0
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
                temporary = Path(handle.name)
                source_descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                with os.fdopen(source_descriptor, "rb") as source_handle:
                    for chunk in iter(lambda: source_handle.read(1024 * 1024), b""):
                        copied_digest.update(chunk)
                        copied_size += len(chunk)
                        handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            if copied_digest.hexdigest() != digest.value or copied_size != expected_size:
                raise ArtifactStoreError("CAS object changed during materialization")
            os.chmod(temporary, 0o444)
            self._reject_symlink_components(destination)
            if destination.exists() or destination.is_symlink():
                raise ArtifactStoreError("materialization destination already exists")
            os.replace(temporary, destination)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return destination
