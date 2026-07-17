"""SQLite-backed, fail-closed coordinated capture orchestration."""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, cast

from sloforge.continuum.adapters import ContinuumRuntimeAdapter, SessionLifecycle
from sloforge.continuum.operations import (
    CheckpointArtifact,
    checkpoint_full,
    load_checkpoint_artifact,
    save_checkpoint_artifact,
    verify_checkpoint_artifact,
)
from sloforge.continuum.storage import ContentStore
from sloforge.continuum.transaction import SessionLease

from .models import (
    ArtifactWatermark,
    CaptureBoundary,
    CaptureJournalEntry,
    CapturePhase,
    CaptureStatus,
    CoordinatedBranchPoint,
    CoordinatedCaptureRequest,
    canonical_digest,
    make_branch_point,
)


class CaptureError(RuntimeError):
    code = "helix_capture_error"


class CaptureIdempotencyConflict(CaptureError):
    code = "helix_capture_idempotency_conflict"


class CaptureBoundaryMismatch(CaptureError):
    code = "helix_capture_boundary_mismatch"


class CaptureArtifactIntegrityError(CaptureError):
    code = "helix_capture_artifact_integrity"


class CaptureQuiescenceTimeout(CaptureError):
    code = "helix_capture_quiescence_timeout"


class CaptureAborted(CaptureError):
    code = "helix_capture_aborted"


class CaptureInProgress(CaptureError):
    code = "helix_capture_in_progress"


class CaptureFailed(CaptureError):
    code = "helix_capture_failed"


@dataclass(frozen=True, slots=True)
class VerifiedCaptureArtifact:
    """A staged reference accompanied by the exact bytes authenticated by its digest."""

    reference: ArtifactWatermark
    payload: bytes


@dataclass(frozen=True, slots=True)
class CaptureSources:
    """Live sources used by one capture attempt.

    The environment and effect callbacks intentionally return small Helix-owned
    references. Their concrete contracts are imported by the application that
    owns those sources, avoiding an import-time dependency cycle.
    """

    model: ContinuumRuntimeAdapter
    model_store: ContentStore
    lease: SessionLease
    read_boundary: Callable[[], CaptureBoundary]
    capture_environment: Callable[[int], ArtifactWatermark | VerifiedCaptureArtifact]
    capture_effects: Callable[[], ArtifactWatermark | VerifiedCaptureArtifact]
    expected_tenant_id: str | None = None
    quiescence_poll: Callable[[int], None] | None = None
    release_quiescence: Callable[[], None] | None = None


PhaseHook = Callable[[CapturePhase], None]


class CoordinatedCaptureCoordinator:
    """Persist every transition and publish all BranchPoint references atomically."""

    def __init__(
        self,
        database: str | Path,
        *,
        artifact_directory: Path | None = None,
        require_verified_artifacts: bool = False,
        max_verified_artifact_bytes: int = 512 * 1024 * 1024,
    ) -> None:
        self._database = str(database)
        self._connection = sqlite3.connect(self._database, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._artifact_directory = artifact_directory
        self._require_verified_artifacts = require_verified_artifacts
        if max_verified_artifact_bytes < 1:
            raise ValueError("verified capture artifact bound must be positive")
        self._max_verified_artifact_bytes = max_verified_artifact_bytes
        if artifact_directory is not None:
            artifact_directory.mkdir(parents=True, exist_ok=True)
        with self._connection:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=FULL")
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS capture_transactions (
                    capture_id TEXT PRIMARY KEY,
                    request_json TEXT NOT NULL,
                    request_digest TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    error_code TEXT,
                    error_message TEXT,
                    branch_point_json TEXT
                );
                CREATE TABLE IF NOT EXISTS capture_journal (
                    capture_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    phase TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    detail TEXT NOT NULL,
                    detail_digest TEXT NOT NULL,
                    PRIMARY KEY (capture_id, sequence),
                    FOREIGN KEY (capture_id) REFERENCES capture_transactions(capture_id)
                );
                """
            )

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> CoordinatedCaptureCoordinator:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _artifact_path(self, capture_id: str) -> Path | None:
        if self._artifact_directory is None:
            return None
        return self._artifact_directory / f"{capture_id}.continuum.json"

    def propose(self, request: CoordinatedCaptureRequest) -> CaptureStatus:
        document = request.model_dump_json()
        digest = canonical_digest(request.model_dump(mode="json"))
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT request_digest FROM capture_transactions WHERE capture_id = ?",
                (request.capture_id,),
            ).fetchone()
            if row is not None:
                if row["request_digest"] != digest:
                    raise CaptureIdempotencyConflict(
                        "capture identifier was already used for a different request"
                    )
                return self.status(request.capture_id)
            self._connection.execute(
                """
                INSERT INTO capture_transactions(
                    capture_id, request_json, request_digest, phase, attempt
                ) VALUES (?, ?, ?, ?, 1)
                """,
                (request.capture_id, document, digest, CapturePhase.CAPTURE_PROPOSED.value),
            )
            self._append_journal_locked(
                request.capture_id,
                CapturePhase.CAPTURE_PROPOSED,
                1,
                "capture request persisted",
            )
        return self.status(request.capture_id)

    def _row(self, capture_id: str) -> sqlite3.Row:
        row = self._connection.execute(
            "SELECT * FROM capture_transactions WHERE capture_id = ?", (capture_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown capture {capture_id}")
        return cast(sqlite3.Row, row)

    def status(self, capture_id: str) -> CaptureStatus:
        with self._lock:
            row = self._row(capture_id)
            request = CoordinatedCaptureRequest.model_validate_json(
                row["request_json"], strict=True
            )
            branch = (
                CoordinatedBranchPoint.model_validate_json(row["branch_point_json"], strict=True)
                if row["branch_point_json"] is not None
                else None
            )
            return CaptureStatus(
                request=request,
                phase=CapturePhase(row["phase"]),
                attempt=row["attempt"],
                error_code=row["error_code"],
                error_message=row["error_message"],
                branch_point=branch,
            )

    def journal(self, capture_id: str) -> tuple[CaptureJournalEntry, ...]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM capture_journal WHERE capture_id = ? ORDER BY sequence",
                (capture_id,),
            ).fetchall()
        return tuple(
            CaptureJournalEntry(
                capture_id=capture_id,
                sequence=row["sequence"],
                phase=CapturePhase(row["phase"]),
                attempt=row["attempt"],
                detail=row["detail"],
                detail_digest=row["detail_digest"],
            )
            for row in rows
        )

    def _append_journal_locked(
        self, capture_id: str, phase: CapturePhase, attempt: int, detail: str
    ) -> None:
        row = self._connection.execute(
            "SELECT COALESCE(MAX(sequence), -1) + 1 AS sequence "
            "FROM capture_journal WHERE capture_id = ?",
            (capture_id,),
        ).fetchone()
        self._connection.execute(
            """
            INSERT INTO capture_journal(
                capture_id, sequence, phase, attempt, detail, detail_digest
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                capture_id,
                row["sequence"],
                phase.value,
                attempt,
                detail,
                canonical_digest({"detail": detail}),
            ),
        )

    def _transition(
        self,
        capture_id: str,
        *,
        expected: CapturePhase,
        phase: CapturePhase,
        attempt: int,
        detail: str,
    ) -> None:
        with self._lock, self._connection:
            changed = self._connection.execute(
                "UPDATE capture_transactions SET phase = ? "
                "WHERE capture_id = ? AND phase = ? AND attempt = ?",
                (phase.value, capture_id, expected.value, attempt),
            ).rowcount
            if changed != 1:
                raise CaptureInProgress(
                    f"capture {capture_id} lost its phase/attempt compare-and-swap"
                )
            self._append_journal_locked(capture_id, phase, attempt, detail)

    def _claim_execution(
        self, capture_id: str, *, recover: bool
    ) -> tuple[CoordinatedCaptureRequest, int]:
        """Atomically claim a proposed/retry attempt before touching live sources."""

        with self._lock, self._connection:
            row = self._row(capture_id)
            phase = CapturePhase(row["phase"])
            if phase is CapturePhase.COMPLETED:
                raise CaptureInProgress("capture completed while execution was being claimed")
            if phase is CapturePhase.ABORTED:
                raise CaptureAborted(f"capture {capture_id} was aborted")
            attempt = int(row["attempt"])
            if phase is CapturePhase.FAILED:
                attempt += 1
                detail = "exact retry accepted"
            elif phase is CapturePhase.CAPTURE_PROPOSED:
                detail = "capture barrier engaged"
            elif recover:
                attempt += 1
                detail = f"explicit recovery claimed from {phase.value}"
            else:
                raise CaptureInProgress(
                    f"capture {capture_id} is already executing in phase {phase.value}"
                )
            changed = self._connection.execute(
                """
                UPDATE capture_transactions
                SET phase = ?, attempt = ?, error_code = NULL, error_message = NULL
                WHERE capture_id = ? AND phase = ? AND attempt = ?
                """,
                (
                    CapturePhase.QUIESCING.value,
                    attempt,
                    capture_id,
                    phase.value,
                    int(row["attempt"]),
                ),
            ).rowcount
            if changed != 1:
                raise CaptureInProgress(f"capture {capture_id} is owned by another executor")
            self._append_journal_locked(
                capture_id,
                CapturePhase.QUIESCING,
                attempt,
                detail,
            )
            request = CoordinatedCaptureRequest.model_validate_json(
                row["request_json"], strict=True
            )
            return request, attempt

    @staticmethod
    def _raise_boundary(actual: CaptureBoundary, expected: CaptureBoundary, where: str) -> NoReturn:
        raise CaptureBoundaryMismatch(
            f"{where} boundary mismatch: expected {expected.model_dump(mode='json')}, "
            f"observed {actual.model_dump(mode='json')}"
        )

    def _quiesce(
        self, request: CoordinatedCaptureRequest, sources: CaptureSources
    ) -> CaptureBoundary:
        previous: CaptureBoundary | None = None
        for poll in range(request.max_quiescence_polls):
            current = sources.read_boundary()
            if current == request.boundary and previous == current:
                return current
            if any(
                observed > expected
                for observed, expected in zip(
                    current.model_dump(mode="json").values(),
                    request.boundary.model_dump(mode="json").values(),
                    strict=True,
                )
            ):
                self._raise_boundary(current, request.boundary, "quiescence")
            previous = current
            if sources.quiescence_poll is not None:
                sources.quiescence_poll(poll)
        raise CaptureQuiescenceTimeout(
            f"sources did not stabilize at the requested boundary after "
            f"{request.max_quiescence_polls} polls"
        )

    def _load_or_capture_model(
        self, request: CoordinatedCaptureRequest, sources: CaptureSources
    ) -> CheckpointArtifact:
        path = self._artifact_path(request.capture_id)
        if path is not None and path.exists():
            artifact = load_checkpoint_artifact(path)
            verify_checkpoint_artifact(artifact)
            if artifact.capsule.identity.session_id != request.session_id:
                raise CaptureArtifactIntegrityError(
                    "persisted Continuum artifact belongs to a different session"
                )
            return artifact
        artifact = checkpoint_full(
            sources.model,
            request.session_id,
            store=sources.model_store,
            lease=sources.lease,
            published_at_ms=request.published_at_ms,
            capture_timestamp=request.capture_timestamp,
            git_commit=request.git_commit,
            continuum_version=request.continuum_version,
        )
        if path is not None:
            save_checkpoint_artifact(path, artifact)
        return artifact

    @staticmethod
    def _validate(
        request: CoordinatedCaptureRequest,
        artifact: CheckpointArtifact,
        environment: ArtifactWatermark,
        effects: ArtifactWatermark,
        observed: CaptureBoundary,
        expected_tenant_id: str | None,
    ) -> None:
        expected = request.boundary
        if (
            expected_tenant_id is not None
            and artifact.capsule.identity.tenant_id != expected_tenant_id
        ):
            raise CaptureBoundaryMismatch(
                "Continuum checkpoint tenant differs from the authorized capture tenant"
            )
        if observed != expected:
            CoordinatedCaptureCoordinator._raise_boundary(observed, expected, "validation")
        capsule_watermark = artifact.capsule.transaction.commit_watermark
        logical_watermark = (
            artifact.capsule.logical_state.client_delivery.last_gateway_committed_token_index
        )
        if capsule_watermark != expected.model_token_watermark:
            raise CaptureBoundaryMismatch(
                "Continuum transaction watermark differs from the proposed model boundary"
            )
        if logical_watermark != expected.model_token_watermark:
            raise CaptureBoundaryMismatch(
                "Continuum logical-state watermark differs from the proposed model boundary"
            )
        if environment.watermark != expected.environment_event_watermark:
            raise CaptureBoundaryMismatch(
                "environment capsule watermark differs from the proposed environment boundary"
            )
        if effects.watermark != expected.effect_watermark:
            raise CaptureBoundaryMismatch(
                "effect ledger watermark differs from the proposed effect boundary"
            )

    def _verify_staged_artifact(
        self,
        value: ArtifactWatermark | VerifiedCaptureArtifact,
        *,
        label: str,
    ) -> ArtifactWatermark:
        if isinstance(value, ArtifactWatermark):
            if self._require_verified_artifacts:
                raise CaptureArtifactIntegrityError(
                    f"{label} capture returned a digest claim without authenticated bytes"
                )
            return value
        if len(value.payload) > self._max_verified_artifact_bytes:
            raise CaptureArtifactIntegrityError(
                f"{label} artifact exceeds the configured verification bound"
            )
        if hashlib.sha256(value.payload).hexdigest() != value.reference.digest:
            raise CaptureArtifactIntegrityError(
                f"{label} artifact bytes do not match the staged digest"
            )
        return value.reference

    def execute(
        self,
        capture_id: str,
        sources: CaptureSources,
        *,
        phase_hook: PhaseHook | None = None,
        recover: bool = False,
    ) -> CoordinatedBranchPoint:
        """Run a capture; explicit ``recover`` may reclaim a crash-stopped mid-state attempt."""

        status = self.status(capture_id)
        if status.phase is CapturePhase.COMPLETED:
            assert status.branch_point is not None
            return status.branch_point
        if status.phase is CapturePhase.ABORTED:
            raise CaptureAborted(f"capture {capture_id} was aborted")
        try:
            request, attempt = self._claim_execution(capture_id, recover=recover)
        except CaptureInProgress:
            raced = self.status(capture_id)
            if raced.phase is CapturePhase.COMPLETED:
                assert raced.branch_point is not None
                return raced.branch_point
            raise
        initially_active = False
        try:
            if phase_hook is not None:
                phase_hook(CapturePhase.QUIESCING)
            self._quiesce(request, sources)
            metadata = sources.model.inspect_session(request.session_id)
            if (
                sources.expected_tenant_id is not None
                and metadata.tenant_id != sources.expected_tenant_id
            ):
                raise CaptureError(
                    "model session tenant differs from the authorized capture tenant"
                )
            initially_active = metadata.lifecycle is SessionLifecycle.ACTIVE
            if initially_active:
                sources.model.pause_session(request.session_id)
            elif metadata.lifecycle is not SessionLifecycle.PAUSED:
                raise CaptureError(
                    f"model session cannot be captured from lifecycle {metadata.lifecycle.value}"
                )
            self._transition(
                capture_id,
                expected=CapturePhase.QUIESCING,
                phase=CapturePhase.QUIESCED,
                attempt=attempt,
                detail="all sources quiesced",
            )
            if phase_hook is not None:
                phase_hook(CapturePhase.QUIESCED)

            artifact = self._load_or_capture_model(request, sources)
            self._transition(
                capture_id,
                expected=CapturePhase.QUIESCED,
                phase=CapturePhase.MODEL_CAPTURED,
                attempt=attempt,
                detail=f"Continuum capsule {artifact.capsule.identity.capsule_id} staged",
            )
            if phase_hook is not None:
                phase_hook(CapturePhase.MODEL_CAPTURED)

            environment = self._verify_staged_artifact(
                sources.capture_environment(request.seed), label="environment"
            )
            self._transition(
                capture_id,
                expected=CapturePhase.MODEL_CAPTURED,
                phase=CapturePhase.ENVIRONMENT_CAPTURED,
                attempt=attempt,
                detail=f"environment capsule {environment.artifact_id} staged",
            )
            if phase_hook is not None:
                phase_hook(CapturePhase.ENVIRONMENT_CAPTURED)

            effects = self._verify_staged_artifact(sources.capture_effects(), label="effect ledger")
            self._transition(
                capture_id,
                expected=CapturePhase.ENVIRONMENT_CAPTURED,
                phase=CapturePhase.EFFECTS_CAPTURED,
                attempt=attempt,
                detail=f"effect ledger {effects.artifact_id} staged",
            )
            if phase_hook is not None:
                phase_hook(CapturePhase.EFFECTS_CAPTURED)

            self._transition(
                capture_id,
                expected=CapturePhase.EFFECTS_CAPTURED,
                phase=CapturePhase.VALIDATING,
                attempt=attempt,
                detail="cross-source consistency validation",
            )
            observed = sources.read_boundary()
            self._validate(
                request,
                artifact,
                environment,
                effects,
                observed,
                sources.expected_tenant_id,
            )
            branch_point = make_branch_point(
                request,
                continuum_capsule_id=artifact.capsule.identity.capsule_id,
                environment=environment,
                effects=effects,
            )
            if phase_hook is not None:
                phase_hook(CapturePhase.VALIDATING)

            # Publication and the terminal transition share one SQLite commit.
            with self._lock, self._connection:
                changed = self._connection.execute(
                    """
                    UPDATE capture_transactions
                    SET phase = ?, branch_point_json = ?, error_code = NULL, error_message = NULL
                    WHERE capture_id = ? AND phase = ? AND attempt = ?
                    """,
                    (
                        CapturePhase.COMPLETED.value,
                        branch_point.model_dump_json(),
                        capture_id,
                        CapturePhase.VALIDATING.value,
                        attempt,
                    ),
                ).rowcount
                if changed != 1:
                    raise CaptureInProgress(
                        f"capture {capture_id} lost its publication compare-and-swap"
                    )
                self._append_journal_locked(
                    capture_id,
                    CapturePhase.COMPLETED,
                    attempt,
                    f"BranchPoint {branch_point.branch_point_id} published atomically",
                )
            if phase_hook is not None:
                phase_hook(CapturePhase.COMPLETED)
            return branch_point
        except Exception as error:
            code = error.code if isinstance(error, CaptureError) else type(error).__name__
            with self._lock, self._connection:
                row = self._row(capture_id)
                if CapturePhase(row["phase"]) is not CapturePhase.COMPLETED:
                    changed = self._connection.execute(
                        """
                        UPDATE capture_transactions
                        SET phase = ?, error_code = ?, error_message = ?, branch_point_json = NULL
                        WHERE capture_id = ? AND attempt = ?
                        """,
                        (CapturePhase.FAILED.value, code, str(error), capture_id, attempt),
                    ).rowcount
                    if changed == 1:
                        self._append_journal_locked(
                            capture_id,
                            CapturePhase.FAILED,
                            attempt,
                            f"capture failed closed: {code}",
                        )
            if isinstance(error, CaptureError):
                raise
            raise CaptureFailed(str(error)) from error
        finally:
            self._resume_sources(request, sources, was_active=initially_active)

    @staticmethod
    def _resume_sources(
        request: CoordinatedCaptureRequest,
        sources: CaptureSources,
        *,
        was_active: bool,
    ) -> None:
        current = sources.model.inspect_session(request.session_id)
        if was_active and current.lifecycle is SessionLifecycle.PAUSED:
            sources.model.resume_session(
                request.session_id, expected_owner_epoch=current.owner_epoch
            )
        if sources.release_quiescence is not None:
            sources.release_quiescence()

    def abort(self, capture_id: str, *, reason: str) -> CaptureStatus:
        if not reason.strip():
            raise ValueError("abort reason must not be empty")
        status = self.status(capture_id)
        if status.phase is CapturePhase.COMPLETED:
            raise CaptureError("a published BranchPoint cannot be aborted")
        if status.phase is CapturePhase.ABORTED:
            return status
        if status.phase not in {CapturePhase.CAPTURE_PROPOSED, CapturePhase.FAILED}:
            raise CaptureInProgress(
                "an executing capture cannot be aborted without a cooperative abort protocol"
            )
        self._transition(
            capture_id,
            expected=status.phase,
            phase=CapturePhase.ABORTING,
            attempt=status.attempt,
            detail=reason,
        )
        self._transition(
            capture_id,
            expected=CapturePhase.ABORTING,
            phase=CapturePhase.ABORTED,
            attempt=status.attempt,
            detail=reason,
        )
        return self.status(capture_id)

    def published_branch_point(self, capture_id: str) -> CoordinatedBranchPoint | None:
        """Return only an atomically completed publication; staged pieces remain invisible."""

        status = self.status(capture_id)
        return status.branch_point if status.phase is CapturePhase.COMPLETED else None
