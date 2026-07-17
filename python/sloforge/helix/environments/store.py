"""Tenant-scoped content-addressed storage for environment capsules."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .models import StorageAccounting, content_digest

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class ContentMissingError(FileNotFoundError):
    """A capsule references content which is absent from local storage."""


class ContentCorruptionError(OSError):
    """Content-addressed bytes no longer match their authenticated name."""


def validate_identifier(value: str, *, label: str) -> str:
    if not _SAFE_ID.fullmatch(value):
        raise ValueError(f"{label} contains unsafe characters")
    return value


@dataclass(frozen=True, slots=True)
class ObjectReference:
    tenant_id: str
    digest: str
    size_bytes: int


class LocalContentStore:
    """A bounded local CAS. Identical content is shared only inside one tenant."""

    def __init__(self, root: Path | str, *, max_object_bytes: int = 256 * 1024 * 1024) -> None:
        if max_object_bytes < 1:
            raise ValueError("maximum object size must be positive")
        self.root = Path(root).resolve()
        self.max_object_bytes = max_object_bytes
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)

    def _tenant_root(self, tenant_id: str) -> Path:
        return self.root / "tenants" / validate_identifier(tenant_id, label="tenant id")

    def object_path(self, tenant_id: str, digest: str) -> Path:
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("invalid sha256 digest")
        return self._tenant_root(tenant_id) / "objects" / digest[:2] / digest[2:]

    def put(self, tenant_id: str, data: bytes) -> ObjectReference:
        if len(data) > self.max_object_bytes:
            raise ValueError(f"content object exceeds {self.max_object_bytes} bytes")
        digest = content_digest(data)
        target = self.object_path(tenant_id, digest)
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if target.exists():
            self._verify_path(target, expected=digest, expected_size=len(data))
            return ObjectReference(tenant_id=tenant_id, digest=digest, size_bytes=len(data))
        descriptor, temporary_name = tempfile.mkstemp(prefix=".object-", dir=target.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary, 0o600)
            try:
                temporary.replace(target)
            except FileExistsError:
                temporary.unlink(missing_ok=True)
            self._verify_path(target, expected=digest, expected_size=len(data))
        finally:
            temporary.unlink(missing_ok=True)
        return ObjectReference(tenant_id=tenant_id, digest=digest, size_bytes=len(data))

    def read(self, reference: ObjectReference, *, tenant_id: str) -> bytes:
        if reference.tenant_id != tenant_id:
            raise PermissionError("cross-tenant environment content access is disabled")
        path = self.object_path(tenant_id, reference.digest)
        if not path.is_file() or path.is_symlink():
            raise ContentMissingError(reference.digest)
        self._verify_path(path, expected=reference.digest, expected_size=reference.size_bytes)
        return path.read_bytes()

    def read_digest(self, tenant_id: str, digest: str, *, expected_size: int) -> bytes:
        return self.read(
            ObjectReference(tenant_id=tenant_id, digest=digest, size_bytes=expected_size),
            tenant_id=tenant_id,
        )

    @staticmethod
    def _verify_path(path: Path, *, expected: str, expected_size: int) -> None:
        stat_result = path.stat()
        if stat_result.st_size != expected_size:
            raise ContentCorruptionError(f"content object {expected} has the wrong size")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ContentCorruptionError(f"content object {expected} failed digest verification")

    def accounting(
        self,
        tenant_id: str,
        *,
        capsule_logical_bytes: int = 0,
        branch_workspace_bytes: int = 0,
    ) -> StorageAccounting:
        object_root = self._tenant_root(tenant_id) / "objects"
        objects = (
            tuple(
                path for path in object_root.glob("*/*") if path.is_file() and not path.is_symlink()
            )
            if object_root.exists()
            else ()
        )
        return StorageAccounting(
            tenant_id=tenant_id,
            object_count=len(objects),
            stored_bytes=sum(path.stat().st_size for path in objects),
            capsule_logical_bytes=capsule_logical_bytes,
            branch_workspace_bytes=branch_workspace_bytes,
        )
