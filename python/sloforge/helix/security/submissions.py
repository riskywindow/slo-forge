"""Durable, tenant-scoped replay protection for reward and training submissions."""

from __future__ import annotations

import re
import sqlite3
import threading
from enum import StrEnum
from pathlib import Path
from typing import Self

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class SubmissionDisposition(StrEnum):
    ACCEPTED = "accepted"
    DUPLICATE_ID = "duplicate_id"
    DUPLICATE_CONTENT = "duplicate_content"


class DuplicateSubmissionError(ValueError):
    """An exact submission replay was rejected without exposing submission content."""


class SubmissionIdentityConflict(ValueError):
    """One submission identity was rebound to different content."""


class SubmissionRegistryFull(RuntimeError):
    """The configured replay-protection record bound has been reached."""


def _identifier(value: str, *, label: str) -> str:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{label} contains unsafe characters")
    return value


def _digest(value: str) -> str:
    if not _DIGEST.fullmatch(value):
        raise ValueError("submission content digest is not a lowercase SHA-256 digest")
    return value


class SubmissionRegistry:
    """SQLite replay registry that survives worker restarts and content-ID aliases."""

    def __init__(self, database: str | Path, *, max_records: int = 1_000_000) -> None:
        if not 1 <= max_records <= 10_000_000:
            raise ValueError("submission registry bound must be in [1, 10,000,000]")
        self._max_records = max_records
        self._connection = sqlite3.connect(
            str(database), timeout=5.0, isolation_level="IMMEDIATE", check_same_thread=False
        )
        self._lock = threading.RLock()
        with self._connection:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=FULL")
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS security_submissions (
                    tenant_id TEXT NOT NULL,
                    namespace TEXT NOT NULL,
                    submission_id TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    created_at_ms INTEGER NOT NULL CHECK(created_at_ms >= 0),
                    PRIMARY KEY (tenant_id, namespace, submission_id),
                    UNIQUE (tenant_id, namespace, content_sha256)
                )
                """
            )

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def register(
        self,
        *,
        tenant_id: str,
        namespace: str,
        submission_id: str,
        content_sha256: str,
        created_at_ms: int,
    ) -> SubmissionDisposition:
        tenant_id = _identifier(tenant_id, label="tenant id")
        namespace = _identifier(namespace, label="submission namespace")
        submission_id = _identifier(submission_id, label="submission id")
        content_sha256 = _digest(content_sha256)
        if created_at_ms < 0:
            raise ValueError("submission creation time must be non-negative")
        with self._lock:
            existing_id = self._connection.execute(
                """
                SELECT content_sha256 FROM security_submissions
                WHERE tenant_id=? AND namespace=? AND submission_id=?
                """,
                (tenant_id, namespace, submission_id),
            ).fetchone()
            if existing_id is not None:
                if existing_id[0] != content_sha256:
                    raise SubmissionIdentityConflict(
                        "submission identity is already bound to different content"
                    )
                return SubmissionDisposition.DUPLICATE_ID
            existing_content = self._connection.execute(
                """
                SELECT submission_id FROM security_submissions
                WHERE tenant_id=? AND namespace=? AND content_sha256=?
                """,
                (tenant_id, namespace, content_sha256),
            ).fetchone()
            if existing_content is not None:
                return SubmissionDisposition.DUPLICATE_CONTENT
            count = self._connection.execute("SELECT COUNT(*) FROM security_submissions").fetchone()
            assert count is not None
            if int(count[0]) >= self._max_records:
                raise SubmissionRegistryFull("submission replay registry is full")
            with self._connection:
                self._connection.execute(
                    "INSERT INTO security_submissions VALUES (?, ?, ?, ?, ?)",
                    (tenant_id, namespace, submission_id, content_sha256, created_at_ms),
                )
        return SubmissionDisposition.ACCEPTED

    def require_new(self, **submission: str | int) -> None:
        """Register a submission or reject either identity/content form of replay."""

        disposition = self.register(
            tenant_id=str(submission["tenant_id"]),
            namespace=str(submission["namespace"]),
            submission_id=str(submission["submission_id"]),
            content_sha256=str(submission["content_sha256"]),
            created_at_ms=int(submission["created_at_ms"]),
        )
        if disposition is not SubmissionDisposition.ACCEPTED:
            raise DuplicateSubmissionError(f"duplicate submission rejected: {disposition.value}")
