"""Bounded local state stores with integrity, deduplication, COW, and GC."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import tempfile
import threading
import zlib
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=256)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_MAX_CHUNK_BYTES = 64 * 1024 * 1024
_MAX_STORED_BYTES = _MAX_CHUNK_BYTES + 1024 * 1024
_MAX_MANIFEST_CHUNKS = 65_536


class StoreModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class ChunkRef(StoreModel):
    tenant_id: NonEmpty
    digest: Sha256
    size_bytes: Annotated[int, Field(ge=0, le=_MAX_CHUNK_BYTES)]
    stored_bytes: Annotated[int, Field(ge=0, le=_MAX_STORED_BYTES)]
    compression: Literal["none", "zlib"] = "none"
    encryption: Literal["none"] = "none"
    key_id: None = None


class StoredManifest(StoreModel):
    tenant_id: NonEmpty
    manifest_id: Sha256
    kind: Literal["complete", "incremental", "fork", "rollback", "migration"]
    chunks: tuple[ChunkRef, ...]
    parent_manifest_id: Sha256 | None = None
    published_at_ms: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        if len(self.chunks) > _MAX_MANIFEST_CHUNKS:
            raise ValueError(f"manifest exceeds {_MAX_MANIFEST_CHUNKS} chunks")
        if any(chunk.tenant_id != self.tenant_id for chunk in self.chunks):
            raise ValueError("manifest cannot reference another tenant's state")
        if len({chunk.digest for chunk in self.chunks}) != len(self.chunks):
            raise ValueError("manifest contains duplicate chunk references")
        return self


class ChunkMissing(FileNotFoundError):
    """A referenced content chunk is absent."""


class ChunkCorrupt(OSError):
    """Stored bytes do not reconstruct the authenticated plaintext chunk."""


class ContentStore(Protocol):
    def put(
        self,
        tenant_id: str,
        data: bytes,
        *,
        compression: Literal["none", "zlib"] = "none",
        expires_at_ms: int | None = None,
    ) -> ChunkRef: ...

    def read(
        self, tenant_id: str, reference: ChunkRef, *, offset: int = 0, length: int | None = None
    ) -> bytes: ...

    def publish(
        self,
        *,
        tenant_id: str,
        kind: Literal["complete", "incremental", "fork", "rollback", "migration"],
        chunks: tuple[ChunkRef, ...],
        published_at_ms: int,
        parent_manifest_id: str | None = None,
    ) -> StoredManifest: ...


def _validate_data(data: bytes) -> None:
    if len(data) > _MAX_CHUNK_BYTES:
        raise ValueError(f"state chunk exceeds {_MAX_CHUNK_BYTES} bytes")


def _encode(data: bytes, compression: Literal["none", "zlib"]) -> bytes:
    return data if compression == "none" else zlib.compress(data, level=6)


def _decode(data: bytes, compression: Literal["none", "zlib"]) -> bytes:
    if len(data) > _MAX_STORED_BYTES:
        raise ChunkCorrupt("stored state chunk exceeds its bounded representation")
    if compression == "none":
        return data
    try:
        decoder = zlib.decompressobj()
        decoded = decoder.decompress(data, _MAX_CHUNK_BYTES + 1)
        if len(decoded) > _MAX_CHUNK_BYTES or decoder.unconsumed_tail:
            raise ChunkCorrupt("compressed state chunk exceeds the plaintext bound")
        remaining = _MAX_CHUNK_BYTES + 1 - len(decoded)
        decoded += decoder.flush(remaining)
        if len(decoded) > _MAX_CHUNK_BYTES:
            raise ChunkCorrupt("compressed state chunk exceeds the plaintext bound")
        if not decoder.eof or decoder.unused_data:
            raise ChunkCorrupt("compressed state chunk has an invalid stream boundary")
        return decoded
    except zlib.error as exc:
        raise ChunkCorrupt("compressed state chunk is corrupt") from exc


def _reference(
    *, tenant_id: str, data: bytes, stored: bytes, compression: Literal["none", "zlib"]
) -> ChunkRef:
    return ChunkRef(
        tenant_id=tenant_id,
        digest=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        stored_bytes=len(stored),
        compression=compression,
    )


def _manifest_id(
    *,
    tenant_id: str,
    kind: str,
    chunks: tuple[ChunkRef, ...],
    parent_manifest_id: str | None,
) -> str:
    content = {
        "schema": "sloforge.continuum.store-manifest/v1",
        "tenant_id": tenant_id,
        "kind": kind,
        "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
        "parent_manifest_id": parent_manifest_id,
    }
    payload = json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _slice(data: bytes, *, offset: int, length: int | None) -> bytes:
    if offset < 0 or offset > len(data):
        raise ValueError("chunk read offset is outside the plaintext chunk")
    if length is not None and length < 0:
        raise ValueError("chunk read length must be non-negative")
    return data[offset:] if length is None else data[offset : offset + length]


class MemoryContentStore:
    """Thread-safe deterministic store used by unit tests and local orchestration."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._chunks: dict[tuple[str, str], tuple[ChunkRef, bytes, int, int | None]] = {}
        self._manifests: dict[tuple[str, str], StoredManifest] = {}

    def put(
        self,
        tenant_id: str,
        data: bytes,
        *,
        compression: Literal["none", "zlib"] = "none",
        expires_at_ms: int | None = None,
    ) -> ChunkRef:
        _validate_data(data)
        if expires_at_ms is not None and expires_at_ms < 0:
            raise ValueError("expiration must be non-negative")
        stored = _encode(data, compression)
        reference = _reference(
            tenant_id=tenant_id, data=data, stored=stored, compression=compression
        )
        key = (tenant_id, reference.digest)
        with self._lock:
            existing = self._chunks.get(key)
            if existing is not None:
                expiration = existing[3]
                if expires_at_ms is not None and (expiration is None or expires_at_ms < expiration):
                    self._chunks[key] = (
                        existing[0],
                        existing[1],
                        existing[2],
                        expires_at_ms,
                    )
                return existing[0]
            self._chunks[key] = (reference, stored, 0, expires_at_ms)
        return reference

    def read(
        self, tenant_id: str, reference: ChunkRef, *, offset: int = 0, length: int | None = None
    ) -> bytes:
        if reference.tenant_id != tenant_id:
            raise PermissionError("state chunk tenant authorization failed")
        with self._lock:
            row = self._chunks.get((tenant_id, reference.digest))
            if row is None:
                raise ChunkMissing(reference.digest)
            plaintext = _decode(row[1], row[0].compression)
        self._verify(reference, plaintext)
        return _slice(plaintext, offset=offset, length=length)

    @staticmethod
    def _verify(reference: ChunkRef, plaintext: bytes) -> None:
        if len(plaintext) != reference.size_bytes:
            raise ChunkCorrupt("state chunk size differs from its manifest")
        if hashlib.sha256(plaintext).hexdigest() != reference.digest:
            raise ChunkCorrupt("state chunk digest differs from its manifest")

    def publish(
        self,
        *,
        tenant_id: str,
        kind: Literal["complete", "incremental", "fork", "rollback", "migration"],
        chunks: tuple[ChunkRef, ...],
        published_at_ms: int,
        parent_manifest_id: str | None = None,
    ) -> StoredManifest:
        identifier = _manifest_id(
            tenant_id=tenant_id,
            kind=kind,
            chunks=chunks,
            parent_manifest_id=parent_manifest_id,
        )
        manifest = StoredManifest(
            tenant_id=tenant_id,
            manifest_id=identifier,
            kind=kind,
            chunks=chunks,
            parent_manifest_id=parent_manifest_id,
            published_at_ms=published_at_ms,
        )
        with self._lock:
            existing = self._manifests.get((tenant_id, identifier))
            if existing is not None:
                return existing
            if (
                parent_manifest_id is not None
                and (tenant_id, parent_manifest_id) not in self._manifests
            ):
                raise ChunkMissing("parent manifest is missing")
            for chunk in chunks:
                row = self._chunks.get((tenant_id, chunk.digest))
                if row is None or row[0] != chunk:
                    raise ChunkMissing(chunk.digest)
            for chunk in chunks:
                row = self._chunks[(tenant_id, chunk.digest)]
                self._chunks[(tenant_id, chunk.digest)] = (row[0], row[1], row[2] + 1, row[3])
            self._manifests[(tenant_id, identifier)] = manifest
        return manifest

    def fork(
        self, *, tenant_id: str, parent_manifest_id: str, published_at_ms: int
    ) -> StoredManifest:
        with self._lock:
            parent = self._manifests.get((tenant_id, parent_manifest_id))
            if parent is None:
                raise ChunkMissing("parent manifest is missing")
        return self.publish(
            tenant_id=tenant_id,
            kind="fork",
            chunks=parent.chunks,
            parent_manifest_id=parent_manifest_id,
            published_at_ms=published_at_ms,
        )

    def delete_manifest(self, tenant_id: str, manifest_id: str) -> None:
        with self._lock:
            manifest = self._manifests.pop((tenant_id, manifest_id), None)
            if manifest is None:
                raise ChunkMissing("manifest is missing")
            for chunk in manifest.chunks:
                row = self._chunks[(tenant_id, chunk.digest)]
                self._chunks[(tenant_id, chunk.digest)] = (row[0], row[1], row[2] - 1, row[3])

    def gc(self, *, now_ms: int, maximum_deletions: int = 1024) -> tuple[str, ...]:
        if now_ms < 0 or not 1 <= maximum_deletions <= 10_000:
            raise ValueError("invalid garbage-collection bound")
        deleted: list[str] = []
        with self._lock:
            for key in sorted(self._chunks):
                _reference_value, _stored, references, expiration = self._chunks[key]
                expired = expiration is not None and expiration <= now_ms
                if references == 0 and expired:
                    deleted.append(key[1])
                    del self._chunks[key]
                    if len(deleted) >= maximum_deletions:
                        break
        return tuple(deleted)

    def corrupt_for_test(self, tenant_id: str, digest: str, payload: bytes) -> None:
        """Deterministic corruption hook; deliberately unavailable on adapters."""

        with self._lock:
            row = self._chunks[(tenant_id, digest)]
            self._chunks[(tenant_id, digest)] = (row[0], payload, row[2], row[3])


class FileContentStore:
    """Filesystem chunks plus embedded crash-consistent SQLite metadata."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.chunk_root = root / "chunks"
        self.chunk_root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)
        os.chmod(self.chunk_root, 0o700)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            root / "metadata.db",
            timeout=5.0,
            isolation_level=None,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.execute("PRAGMA busy_timeout=5000")
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        os.chmod(root / "metadata.db", 0o600)
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS chunks (
              tenant_id TEXT NOT NULL,
              digest TEXT NOT NULL,
              payload TEXT NOT NULL,
              path TEXT NOT NULL,
              ref_count INTEGER NOT NULL CHECK(ref_count >= 0),
              expires_at_ms INTEGER,
              PRIMARY KEY(tenant_id, digest)
            );
            CREATE TABLE IF NOT EXISTS manifests (
              tenant_id TEXT NOT NULL,
              manifest_id TEXT NOT NULL,
              payload TEXT NOT NULL,
              PRIMARY KEY(tenant_id, manifest_id)
            );
            """
        )
        self._recover_orphaned_chunk_files()

    def _recover_orphaned_chunk_files(self) -> None:
        """Remove unpublished files left by a crash between rename and metadata commit."""

        published = {
            str((self.root / row["path"]).resolve())
            for row in self._connection.execute("SELECT path FROM chunks").fetchall()
        }
        for path in self.chunk_root.glob("*/*/*"):
            if path.is_file() and str(path.resolve()) not in published:
                path.unlink(missing_ok=True)

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> FileContentStore:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _path(self, tenant_id: str, digest: str) -> Path:
        tenant_scope = hashlib.sha256(f"tenant\0{tenant_id}".encode()).hexdigest()
        return self.chunk_root / tenant_scope / digest[:2] / digest

    def _verified_path(self, tenant_id: str, digest: str, stored_path: str) -> Path:
        """Resolve only the canonical tenant-scoped path recorded by this store."""

        expected = self._path(tenant_id, digest)
        try:
            expected_relative = expected.relative_to(self.root)
        except ValueError as exc:  # pragma: no cover - construction makes this unreachable
            raise ChunkCorrupt("chunk path escapes the content-store root") from exc
        if Path(stored_path).is_absolute() or Path(stored_path) != expected_relative:
            raise ChunkCorrupt("chunk metadata contains a non-canonical storage path")
        try:
            expected.resolve().relative_to(self.chunk_root.resolve())
        except ValueError as exc:
            raise ChunkCorrupt("chunk path escapes its tenant-scoped storage root") from exc
        return expected

    @staticmethod
    def _read_bounded_file(path: Path, expected_size: int) -> bytes:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except FileNotFoundError:
            raise
        except OSError as exc:
            raise ChunkCorrupt("state chunk is not a readable regular file") from exc
        try:
            file_status = os.fstat(descriptor)
            if not stat.S_ISREG(file_status.st_mode) or file_status.st_size != expected_size:
                raise ChunkCorrupt("stored state chunk size differs from its metadata")
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                payload = handle.read(expected_size + 1)
            if len(payload) != expected_size:
                raise ChunkCorrupt("stored state chunk changed during its bounded read")
            return payload
        finally:
            os.close(descriptor)

    def _fsync_chunk_directory_chain(self, leaf: Path) -> None:
        """Persist newly created chunk-directory entries before metadata publish."""

        current = leaf
        while True:
            descriptor = os.open(current, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            if current == self.root:
                return
            current = current.parent

    def put(
        self,
        tenant_id: str,
        data: bytes,
        *,
        compression: Literal["none", "zlib"] = "none",
        expires_at_ms: int | None = None,
    ) -> ChunkRef:
        _validate_data(data)
        if expires_at_ms is not None and expires_at_ms < 0:
            raise ValueError("expiration must be non-negative")
        stored = _encode(data, compression)
        reference = _reference(
            tenant_id=tenant_id, data=data, stored=stored, compression=compression
        )
        with self._lock:
            existing = self._connection.execute(
                "SELECT payload FROM chunks WHERE tenant_id=? AND digest=?",
                (tenant_id, reference.digest),
            ).fetchone()
            if existing is not None:
                if expires_at_ms is not None:
                    self._connection.execute(
                        "UPDATE chunks SET expires_at_ms=CASE "
                        "WHEN expires_at_ms IS NULL OR expires_at_ms>? THEN ? "
                        "ELSE expires_at_ms END WHERE tenant_id=? AND digest=?",
                        (
                            expires_at_ms,
                            expires_at_ms,
                            tenant_id,
                            reference.digest,
                        ),
                    )
                return ChunkRef.model_validate_json(existing["payload"], strict=True)
            path = self._path(tenant_id, reference.digest)
            path.parent.mkdir(parents=True, exist_ok=True)
            os.chmod(path.parent.parent, 0o700)
            os.chmod(path.parent, 0o700)
            descriptor, temporary_name = tempfile.mkstemp(prefix=".continuum-", dir=path.parent)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(stored)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_name, path)
                self._fsync_chunk_directory_chain(path.parent)
                try:
                    self._connection.execute(
                        "INSERT INTO chunks VALUES(?,?,?,?,?,?)",
                        (
                            tenant_id,
                            reference.digest,
                            reference.model_dump_json(),
                            str(path.relative_to(self.root)),
                            0,
                            expires_at_ms,
                        ),
                    )
                except sqlite3.IntegrityError:
                    existing = self._connection.execute(
                        "SELECT payload FROM chunks WHERE tenant_id=? AND digest=?",
                        (tenant_id, reference.digest),
                    ).fetchone()
                    if existing is None:
                        raise
                    return ChunkRef.model_validate_json(existing["payload"], strict=True)
            finally:
                Path(temporary_name).unlink(missing_ok=True)
        return reference

    def read(
        self, tenant_id: str, reference: ChunkRef, *, offset: int = 0, length: int | None = None
    ) -> bytes:
        if reference.tenant_id != tenant_id:
            raise PermissionError("state chunk tenant authorization failed")
        with self._lock:
            row = self._connection.execute(
                "SELECT payload,path FROM chunks WHERE tenant_id=? AND digest=?",
                (tenant_id, reference.digest),
            ).fetchone()
            if row is None:
                raise ChunkMissing(reference.digest)
            stored_reference = ChunkRef.model_validate_json(row["payload"], strict=True)
            if stored_reference != reference:
                raise ChunkCorrupt("chunk reference does not match embedded metadata")
            path = self._verified_path(tenant_id, reference.digest, row["path"])
            try:
                encoded = self._read_bounded_file(path, reference.stored_bytes)
            except FileNotFoundError as exc:
                raise ChunkMissing(reference.digest) from exc
            plaintext = _decode(encoded, reference.compression)
            MemoryContentStore._verify(reference, plaintext)
            return _slice(plaintext, offset=offset, length=length)

    def publish(
        self,
        *,
        tenant_id: str,
        kind: Literal["complete", "incremental", "fork", "rollback", "migration"],
        chunks: tuple[ChunkRef, ...],
        published_at_ms: int,
        parent_manifest_id: str | None = None,
    ) -> StoredManifest:
        identifier = _manifest_id(
            tenant_id=tenant_id,
            kind=kind,
            chunks=chunks,
            parent_manifest_id=parent_manifest_id,
        )
        manifest = StoredManifest(
            tenant_id=tenant_id,
            manifest_id=identifier,
            kind=kind,
            chunks=chunks,
            parent_manifest_id=parent_manifest_id,
            published_at_ms=published_at_ms,
        )
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                existing = self._connection.execute(
                    "SELECT payload FROM manifests WHERE tenant_id=? AND manifest_id=?",
                    (tenant_id, identifier),
                ).fetchone()
                if existing is not None:
                    self._connection.execute("COMMIT")
                    return StoredManifest.model_validate_json(existing["payload"], strict=True)
                if parent_manifest_id is not None:
                    parent = self._connection.execute(
                        "SELECT 1 FROM manifests WHERE tenant_id=? AND manifest_id=?",
                        (tenant_id, parent_manifest_id),
                    ).fetchone()
                    if parent is None:
                        raise ChunkMissing("parent manifest is missing")
                for chunk in chunks:
                    row = self._connection.execute(
                        "SELECT payload FROM chunks WHERE tenant_id=? AND digest=?",
                        (tenant_id, chunk.digest),
                    ).fetchone()
                    if (
                        row is None
                        or ChunkRef.model_validate_json(row["payload"], strict=True) != chunk
                    ):
                        raise ChunkMissing(chunk.digest)
                self._connection.execute(
                    "INSERT INTO manifests VALUES(?,?,?)",
                    (tenant_id, identifier, manifest.model_dump_json()),
                )
                for chunk in chunks:
                    self._connection.execute(
                        "UPDATE chunks SET ref_count=ref_count+1 WHERE tenant_id=? AND digest=?",
                        (tenant_id, chunk.digest),
                    )
                self._connection.execute("COMMIT")
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
        return manifest

    def fork(
        self, *, tenant_id: str, parent_manifest_id: str, published_at_ms: int
    ) -> StoredManifest:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM manifests WHERE tenant_id=? AND manifest_id=?",
                (tenant_id, parent_manifest_id),
            ).fetchone()
            if row is None:
                raise ChunkMissing("parent manifest is missing")
            parent = StoredManifest.model_validate_json(row["payload"], strict=True)
            return self.publish(
                tenant_id=tenant_id,
                kind="fork",
                chunks=parent.chunks,
                parent_manifest_id=parent_manifest_id,
                published_at_ms=published_at_ms,
            )

    def delete_manifest(self, tenant_id: str, manifest_id: str) -> None:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    "SELECT payload FROM manifests WHERE tenant_id=? AND manifest_id=?",
                    (tenant_id, manifest_id),
                ).fetchone()
                if row is None:
                    raise ChunkMissing("manifest is missing")
                manifest = StoredManifest.model_validate_json(row["payload"], strict=True)
                self._connection.execute(
                    "DELETE FROM manifests WHERE tenant_id=? AND manifest_id=?",
                    (tenant_id, manifest_id),
                )
                for chunk in manifest.chunks:
                    self._connection.execute(
                        "UPDATE chunks SET ref_count=ref_count-1 WHERE tenant_id=? AND digest=?",
                        (tenant_id, chunk.digest),
                    )
                self._connection.execute("COMMIT")
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise

    def gc(self, *, now_ms: int, maximum_deletions: int = 1024) -> tuple[str, ...]:
        if now_ms < 0 or not 1 <= maximum_deletions <= 10_000:
            raise ValueError("invalid garbage-collection bound")
        with self._lock:
            rows = self._connection.execute(
                "SELECT tenant_id,digest,path FROM chunks "
                "WHERE ref_count=0 AND expires_at_ms IS NOT NULL AND expires_at_ms<=? "
                "ORDER BY tenant_id,digest LIMIT ?",
                (now_ms, maximum_deletions),
            ).fetchall()
            verified_rows = [
                (
                    row,
                    self._verified_path(row["tenant_id"], row["digest"], row["path"]),
                )
                for row in rows
            ]
            deleted: list[str] = []
            deleted_paths: list[Path] = []
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                for row, verified_path in verified_rows:
                    changed = self._connection.execute(
                        "DELETE FROM chunks WHERE tenant_id=? AND digest=? AND ref_count=0",
                        (row["tenant_id"], row["digest"]),
                    ).rowcount
                    if changed == 1:
                        deleted_paths.append(verified_path)
                        deleted.append(row["digest"])
                self._connection.execute("COMMIT")
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
            for path in deleted_paths:
                path.unlink(missing_ok=True)
            return tuple(deleted)

    def manifest(self, tenant_id: str, manifest_id: str) -> StoredManifest:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM manifests WHERE tenant_id=? AND manifest_id=?",
                (tenant_id, manifest_id),
            ).fetchone()
            if row is None:
                raise ChunkMissing("manifest is missing")
            return StoredManifest.model_validate_json(row["payload"], strict=True)

    def corrupt_for_test(self, tenant_id: str, digest: str, payload: bytes) -> None:
        with self._lock:
            row = self._connection.execute(
                "SELECT path FROM chunks WHERE tenant_id=? AND digest=?", (tenant_id, digest)
            ).fetchone()
            if row is None:
                raise ChunkMissing(digest)
            self._verified_path(tenant_id, digest, row["path"]).write_bytes(payload)


def total_unique_bytes(references: Iterable[ChunkRef]) -> int:
    """Calculate authorized content-addressed bytes without double-counting chunks."""

    unique: dict[tuple[str, str], int] = {}
    for reference in references:
        unique[(reference.tenant_id, reference.digest)] = reference.size_bytes
    return sum(unique.values())
