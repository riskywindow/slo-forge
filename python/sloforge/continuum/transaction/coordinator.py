"""Crash-consistent local CAS coordinator backed by SQLite."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from itertools import pairwise
from pathlib import Path
from threading import RLock

from .models import (
    TERMINAL_PHASES,
    CutoverPhase,
    JournalEntry,
    RecoveryEvidence,
    RecoveryResult,
    SessionLease,
    StateTransactionRecord,
    validate_transaction_id,
)

_MAX_JOURNAL_ENTRIES = 256
_MAX_TIMEOUT_MS = 86_400_000
_MAX_TRANSACTIONS_PER_SESSION = 256


class CoordinatorConflict(RuntimeError):
    """A compare-and-swap precondition did not hold."""


class InvalidTransition(RuntimeError):
    """The requested transaction phase edge is illegal."""


class StaleOwner(RuntimeError):
    """A runtime attempted mutation with an obsolete epoch or fencing token."""


_FORWARD: tuple[CutoverPhase, ...] = (
    CutoverPhase.PROPOSED,
    CutoverPhase.COMPATIBILITY_VALIDATED,
    CutoverPhase.DESTINATION_PREPARING,
    CutoverPhase.PRECOPYING,
    CutoverPhase.DELTA_SYNCING,
    CutoverPhase.CUTOVER_REQUESTED,
    CutoverPhase.SOURCE_QUIESCING,
    CutoverPhase.SOURCE_FROZEN,
    CutoverPhase.FINAL_DELTA_TRANSFERRING,
    CutoverPhase.DESTINATION_IMPORTING,
    CutoverPhase.DESTINATION_VALIDATING,
    CutoverPhase.COMMIT_INTENT_RECORDED,
    CutoverPhase.OWNERSHIP_COMMITTED,
    CutoverPhase.GATEWAY_SWITCHING,
    CutoverPhase.DESTINATION_ACTIVE,
    CutoverPhase.SOURCE_DRAINING,
    CutoverPhase.COMPLETED,
)
_NEXT = {source: destination for source, destination in pairwise(_FORWARD)}
_FAILURE_BEFORE_COMMIT = frozenset(
    {
        CutoverPhase.PROPOSED,
        CutoverPhase.COMPATIBILITY_VALIDATED,
        CutoverPhase.DESTINATION_PREPARING,
        CutoverPhase.PRECOPYING,
        CutoverPhase.DELTA_SYNCING,
        CutoverPhase.CUTOVER_REQUESTED,
        CutoverPhase.SOURCE_QUIESCING,
        CutoverPhase.SOURCE_FROZEN,
        CutoverPhase.FINAL_DELTA_TRANSFERRING,
        CutoverPhase.DESTINATION_IMPORTING,
        CutoverPhase.DESTINATION_VALIDATING,
        CutoverPhase.COMMIT_INTENT_RECORDED,
    }
)
_POST_COMMIT = frozenset(
    {
        CutoverPhase.OWNERSHIP_COMMITTED,
        CutoverPhase.GATEWAY_SWITCHING,
        CutoverPhase.DESTINATION_ACTIVE,
        CutoverPhase.SOURCE_DRAINING,
        CutoverPhase.COMPLETED,
        CutoverPhase.FAILED_AFTER_COMMIT,
    }
)
_DEADLINE_RECOVERY_TARGETS = frozenset(
    {
        CutoverPhase.ABORTING,
        CutoverPhase.ROLLED_BACK,
        CutoverPhase.FAILED_BEFORE_COMMIT,
        CutoverPhase.FAILED_AFTER_COMMIT,
        CutoverPhase.OPERATOR_REQUIRED,
    }
)


def transaction_id(
    *, session_id: str, source_epoch: int, destination: str, plan_hash: str, seed: int
) -> str:
    payload = f"continuum/v1\0{session_id}\0{source_epoch}\0{destination}\0{plan_hash}\0{seed}"
    return hashlib.sha256(payload.encode()).hexdigest()


class DurableCoordinator:
    """Embedded coordinator with persisted idempotent transitions and fencing."""

    def __init__(self, path: Path | str) -> None:
        self.path = str(path)
        self._lock = RLock()
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            self.path,
            timeout=5.0,
            isolation_level=None,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.execute("PRAGMA busy_timeout=5000")
        if self.path != ":memory:":
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=FULL")
        self._initialize()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> DurableCoordinator:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield self._connection
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
            else:
                self._connection.execute("COMMIT")

    def _initialize(self) -> None:
        with self._lock:
            self._connection.executescript(
                """
            CREATE TABLE IF NOT EXISTS continuum_leases (
              session_id TEXT PRIMARY KEY,
              payload TEXT NOT NULL,
              version INTEGER NOT NULL CHECK(version >= 1)
            );
            CREATE TABLE IF NOT EXISTS continuum_transactions (
              transaction_id TEXT PRIMARY KEY,
              session_id TEXT NOT NULL,
              payload TEXT NOT NULL,
              version INTEGER NOT NULL CHECK(version >= 1),
              FOREIGN KEY(session_id) REFERENCES continuum_leases(session_id)
            );
            CREATE TABLE IF NOT EXISTS continuum_journal (
              transaction_id TEXT NOT NULL,
              sequence INTEGER NOT NULL,
              event_id TEXT NOT NULL,
              payload TEXT NOT NULL,
              PRIMARY KEY(transaction_id, sequence),
              UNIQUE(transaction_id, event_id),
              FOREIGN KEY(transaction_id) REFERENCES continuum_transactions(transaction_id)
            );
                """
            )

    @staticmethod
    def _json(value: SessionLease | StateTransactionRecord | JournalEntry) -> str:
        return value.model_dump_json()

    def create_lease(
        self,
        *,
        session_id: str,
        owner_runtime: str,
        expiration_ms: int,
        initial_token_index: int = -1,
    ) -> SessionLease:
        lease = SessionLease(
            session_id=session_id,
            owner_runtime=owner_runtime,
            owner_epoch=1,
            fencing_token=1,
            expiration_ms=expiration_ms,
            coordinator_version=1,
            last_committed_token_index=initial_token_index,
        )
        try:
            with self._write() as connection:
                connection.execute(
                    "INSERT INTO continuum_leases(session_id,payload,version) VALUES(?,?,?)",
                    (lease.session_id, self._json(lease), lease.coordinator_version),
                )
        except sqlite3.IntegrityError as exc:
            raise CoordinatorConflict(f"session {session_id!r} already has a lease") from exc
        return lease

    def lease(self, session_id: str) -> SessionLease:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM continuum_leases WHERE session_id=?", (session_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown session {session_id!r}")
        return SessionLease.model_validate_json(row["payload"], strict=True)

    def assert_owner(
        self,
        *,
        session_id: str,
        owner_runtime: str,
        owner_epoch: int,
        fencing_token: int,
        now_ms: int,
    ) -> SessionLease:
        lease = self.lease(session_id)
        if now_ms >= lease.expiration_ms:
            raise StaleOwner(f"lease for session {session_id!r} expired at {lease.expiration_ms}")
        if (
            lease.owner_runtime != owner_runtime
            or lease.owner_epoch != owner_epoch
            or lease.fencing_token != fencing_token
        ):
            raise StaleOwner(
                f"stale owner for session {session_id!r}; current epoch is {lease.owner_epoch}"
            )
        return lease

    def renew_lease(
        self,
        *,
        session_id: str,
        owner_runtime: str,
        owner_epoch: int,
        fencing_token: int,
        now_ms: int,
        expiration_ms: int,
    ) -> SessionLease:
        """CAS-renew a current, unexpired lease to an absolute logical deadline."""

        if expiration_ms <= now_ms:
            raise ValueError("renewed lease expiration must be after now_ms")
        with self._write() as connection:
            row = connection.execute(
                "SELECT payload FROM continuum_leases WHERE session_id=?", (session_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown session {session_id!r}")
            current = SessionLease.model_validate_json(row["payload"], strict=True)
            if now_ms >= current.expiration_ms:
                raise StaleOwner(f"lease for session {session_id!r} already expired")
            if (
                current.owner_runtime != owner_runtime
                or current.owner_epoch != owner_epoch
                or current.fencing_token != fencing_token
            ):
                raise StaleOwner(f"stale owner cannot renew session {session_id!r}")
            if current.expiration_ms == expiration_ms:
                return current
            if expiration_ms < current.expiration_ms:
                raise ValueError("renewed lease expiration cannot move backwards")
            renewed = current.model_copy(
                update={
                    "expiration_ms": expiration_ms,
                    "coordinator_version": current.coordinator_version + 1,
                }
            )
            changed = connection.execute(
                "UPDATE continuum_leases SET payload=?,version=? WHERE session_id=? AND version=?",
                (
                    self._json(renewed),
                    renewed.coordinator_version,
                    session_id,
                    current.coordinator_version,
                ),
            ).rowcount
            if changed != 1:
                raise CoordinatorConflict("lease renewal compare-and-swap failed")
        return renewed

    def record_committed_progress(
        self,
        *,
        session_id: str,
        owner_runtime: str,
        owner_epoch: int,
        fencing_token: int,
        state_version: int,
        token_index: int,
        now_ms: int,
    ) -> SessionLease:
        """CAS-persist client-visible progress under the current owner fence.

        Streaming runtimes call this after the gateway has durably accepted an
        output boundary and before creating a checkpoint.  It does not change
        ownership and refuses regressions, expired leases, and stale writers.
        """

        if state_version < 0 or token_index < -1:
            raise ValueError("committed state and token watermarks are out of range")
        with self._write() as connection:
            row = connection.execute(
                "SELECT payload,version FROM continuum_leases WHERE session_id=?", (session_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown session {session_id!r}")
            current = SessionLease.model_validate_json(row["payload"], strict=True)
            if now_ms >= current.expiration_ms:
                raise StaleOwner(f"lease for session {session_id!r} expired")
            if (
                current.owner_runtime != owner_runtime
                or current.owner_epoch != owner_epoch
                or current.fencing_token != fencing_token
            ):
                raise StaleOwner(f"stale owner cannot record progress for session {session_id!r}")
            if (
                state_version < current.last_committed_state_version
                or token_index < current.last_committed_token_index
            ):
                raise CoordinatorConflict("committed progress cannot regress")
            if (
                state_version == current.last_committed_state_version
                and token_index == current.last_committed_token_index
            ):
                return current
            updated = current.model_copy(
                update={
                    "last_committed_state_version": state_version,
                    "last_committed_token_index": token_index,
                    "coordinator_version": current.coordinator_version + 1,
                }
            )
            changed = connection.execute(
                "UPDATE continuum_leases SET payload=?,version=? WHERE session_id=? AND version=?",
                (
                    self._json(updated),
                    updated.coordinator_version,
                    session_id,
                    int(row["version"]),
                ),
            ).rowcount
            if changed != 1:
                raise CoordinatorConflict("progress compare-and-swap failed")
        return updated

    def begin_transaction(
        self,
        *,
        session_id: str,
        destination_candidate: str,
        migration_plan_hash: str,
        seed: int,
        now_ms: int,
        timeout_ms: int,
    ) -> StateTransactionRecord:
        if not 1 <= timeout_ms <= _MAX_TIMEOUT_MS:
            raise ValueError(f"timeout_ms must be within 1..{_MAX_TIMEOUT_MS}")
        lease = self.lease(session_id)
        if now_ms >= lease.expiration_ms:
            raise StaleOwner(f"lease for session {session_id!r} expired before migration")
        identifier = transaction_id(
            session_id=session_id,
            source_epoch=lease.owner_epoch,
            destination=destination_candidate,
            plan_hash=migration_plan_hash,
            seed=seed,
        )
        record = StateTransactionRecord(
            transaction_id=identifier,
            session_id=session_id,
            source_owner=lease.owner_runtime,
            destination_candidate=destination_candidate,
            source_epoch=lease.owner_epoch,
            proposed_destination_epoch=lease.owner_epoch + 1,
            migration_plan_hash=migration_plan_hash,
            commit_watermark=lease.last_committed_token_index,
            rollback_watermark=lease.last_committed_token_index,
            timeout_at_ms=now_ms + timeout_ms,
        )
        event = JournalEntry(
            transaction_id=identifier,
            sequence=0,
            event_id="transaction-created",
            from_phase=CutoverPhase.PROPOSED,
            to_phase=CutoverPhase.PROPOSED,
            at_ms=now_ms,
            payload_hash=hashlib.sha256(b"transaction-created").hexdigest(),
        )
        with self._write() as connection:
            lease_row = connection.execute(
                "SELECT payload,version FROM continuum_leases WHERE session_id=?", (session_id,)
            ).fetchone()
            if lease_row is None:
                raise KeyError(f"unknown session {session_id!r}")
            locked_lease = SessionLease.model_validate_json(lease_row["payload"], strict=True)
            if (
                locked_lease.coordinator_version != lease.coordinator_version
                or locked_lease.owner_epoch != lease.owner_epoch
                or locked_lease.owner_runtime != lease.owner_runtime
                or int(lease_row["version"]) != lease.coordinator_version
            ):
                raise CoordinatorConflict("session lease changed while migration was proposed")
            if now_ms >= locked_lease.expiration_ms:
                raise StaleOwner(f"lease for session {session_id!r} expired before migration")
            active = connection.execute(
                "SELECT payload FROM continuum_transactions WHERE session_id=?", (session_id,)
            ).fetchall()
            if len(active) >= _MAX_TRANSACTIONS_PER_SESSION:
                raise CoordinatorConflict(
                    "session transaction history reached its durable retention bound"
                )
            if any(
                StateTransactionRecord.model_validate_json(row["payload"], strict=True).phase
                not in TERMINAL_PHASES
                for row in active
            ):
                raise CoordinatorConflict(f"session {session_id!r} already has an active migration")
            try:
                connection.execute(
                    "INSERT INTO continuum_transactions VALUES(?,?,?,?)",
                    (identifier, session_id, self._json(record), record.version),
                )
                connection.execute(
                    "INSERT INTO continuum_journal VALUES(?,?,?,?)",
                    (identifier, 0, event.event_id, self._json(event)),
                )
            except sqlite3.IntegrityError as exc:
                raise CoordinatorConflict(f"transaction {identifier} is not reusable") from exc
        return record

    def transaction(self, identifier: str) -> StateTransactionRecord:
        validate_transaction_id(identifier)
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM continuum_transactions WHERE transaction_id=?", (identifier,)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown transaction {identifier}")
        return StateTransactionRecord.model_validate_json(row["payload"], strict=True)

    def journal(self, identifier: str) -> tuple[JournalEntry, ...]:
        validate_transaction_id(identifier)
        with self._lock:
            rows = self._connection.execute(
                "SELECT payload FROM continuum_journal WHERE transaction_id=? ORDER BY sequence",
                (identifier,),
            ).fetchall()
        return tuple(JournalEntry.model_validate_json(row["payload"], strict=True) for row in rows)

    def recoverable_transactions(self) -> tuple[StateTransactionRecord, ...]:
        """Enumerate persisted nonterminal transactions after a coordinator restart."""

        with self._lock:
            rows = self._connection.execute(
                "SELECT payload FROM continuum_transactions ORDER BY transaction_id"
            ).fetchall()
        records = tuple(
            StateTransactionRecord.model_validate_json(row["payload"], strict=True) for row in rows
        )
        return tuple(record for record in records if record.phase not in TERMINAL_PHASES)

    @staticmethod
    def _recovery_payload_hash(evidence: RecoveryEvidence) -> str:
        return hashlib.sha256(evidence.model_dump_json().encode("utf-8")).hexdigest()

    def recover_transaction(
        self,
        evidence: RecoveryEvidence,
        *,
        at_ms: int,
    ) -> RecoveryResult:
        """Classify an interrupted transaction from explicit runtime/gateway evidence.

        This method never transfers ownership, renews an expired source lease, activates a
        destination, or releases a runtime fence. Runtime cleanup happens before this call and
        is represented by the version-scoped evidence record. The coordinator only validates
        its durable lease/journal and records an auditable terminal classification.
        """

        with self._lock:
            current = self.transaction(evidence.transaction_id)
            lease = self.lease(current.session_id)
            journal = self.journal(current.transaction_id)
            payload_hash = self._recovery_payload_hash(evidence)
            suffix = evidence.evidence_id[:16]
            committed_entries = tuple(
                entry for entry in journal if entry.to_phase is CutoverPhase.OWNERSHIP_COMMITTED
            )
            lease_is_destination = (
                lease.owner_runtime == current.destination_candidate
                and lease.owner_epoch == current.proposed_destination_epoch
            )
            if len(committed_entries) > 1 or bool(committed_entries) != lease_is_destination:
                raise CoordinatorConflict(
                    "ownership journal and durable lease disagree during recovery"
                )
            ownership_committed = lease_is_destination
            if (
                evidence.observed_owner_runtime != lease.owner_runtime
                or evidence.observed_owner_epoch != lease.owner_epoch
                or evidence.observed_fencing_token != lease.fencing_token
            ):
                raise StaleOwner("recovery evidence was observed against a stale lease")

            recovery_entries = tuple(entry for entry in journal if entry.event_id.endswith(suffix))
            if current.phase in TERMINAL_PHASES:
                if (
                    len(recovery_entries) not in {1, 2}
                    or recovery_entries[-1].to_phase is not current.phase
                    or recovery_entries[-1].payload_hash != payload_hash
                ):
                    raise CoordinatorConflict(
                        "terminal transaction does not match this recovery evidence"
                    )
                prior = recovery_entries[0].from_phase
                return RecoveryResult(
                    transaction_id=current.transaction_id,
                    evidence_id=evidence.evidence_id,
                    ownership_committed=ownership_committed,
                    prior_phase=prior,
                    terminal_phase=current.phase,
                    lease_owner_runtime=lease.owner_runtime,
                    lease_owner_epoch=lease.owner_epoch,
                    recovered_at_ms=recovery_entries[-1].at_ms,
                )
            if evidence.gateway_owner_epoch not in {
                current.source_epoch,
                current.proposed_destination_epoch,
            }:
                raise CoordinatorConflict("gateway recovery epoch is outside the transaction")
            prior_phase = current.phase

            if not ownership_committed:
                if (
                    lease.owner_runtime != current.source_owner
                    or lease.owner_epoch != current.source_epoch
                ):
                    raise StaleOwner("pre-commit recovery cannot prove the source lease")
                if current.phase is not CutoverPhase.ABORTING:
                    current = self._transition_locked(
                        current.transaction_id,
                        expected=current.phase,
                        target=CutoverPhase.ABORTING,
                        event_id=f"recovery-aborting-{suffix}",
                        at_ms=at_ms,
                        payload_hash=payload_hash,
                    )
                    final_at_ms = at_ms + 1
                else:
                    final_at_ms = at_ms
                source_is_safe = (
                    at_ms < lease.expiration_ms
                    and evidence.source_available
                    and evidence.source_resumable
                    and not evidence.source_fenced
                    and evidence.destination_aborted
                    and evidence.cleanup_completed
                    and evidence.gateway_owner_epoch == current.source_epoch
                    and evidence.gateway_commit_watermark == evidence.source_commit_watermark
                )
                if source_is_safe:
                    terminal = CutoverPhase.ROLLED_BACK
                    failure_reason = None
                    event_label = "recovery-rolled-back"
                else:
                    terminal = CutoverPhase.FAILED_BEFORE_COMMIT
                    failure_reason = evidence.reason
                    event_label = "recovery-failed-before-commit"
                recovered = self._transition_locked(
                    current.transaction_id,
                    expected=CutoverPhase.ABORTING,
                    target=terminal,
                    event_id=f"{event_label}-{suffix}",
                    at_ms=final_at_ms,
                    payload_hash=payload_hash,
                    failure_reason=failure_reason,
                )
                recovered_at_ms = final_at_ms
            else:
                postcommit_consistent = (
                    evidence.source_fenced
                    and evidence.destination_available
                    and evidence.destination_validated
                    and not evidence.destination_aborted
                    and evidence.cleanup_completed
                    and evidence.gateway_owner_epoch == current.proposed_destination_epoch
                    and evidence.gateway_commit_watermark == current.commit_watermark
                )
                terminal = (
                    CutoverPhase.FAILED_AFTER_COMMIT
                    if postcommit_consistent
                    else CutoverPhase.OPERATOR_REQUIRED
                )
                recovered = self._transition_locked(
                    current.transaction_id,
                    expected=current.phase,
                    target=terminal,
                    event_id=f"recovery-{terminal.value.lower()}-{suffix}",
                    at_ms=at_ms,
                    payload_hash=payload_hash,
                    failure_reason=evidence.reason,
                )
                recovered_at_ms = at_ms
            return RecoveryResult(
                transaction_id=recovered.transaction_id,
                evidence_id=evidence.evidence_id,
                ownership_committed=ownership_committed,
                prior_phase=prior_phase,
                terminal_phase=recovered.phase,
                lease_owner_runtime=lease.owner_runtime,
                lease_owner_epoch=lease.owner_epoch,
                recovered_at_ms=recovered_at_ms,
            )

    @staticmethod
    def _legal_target(
        current: CutoverPhase, target: CutoverPhase, *, ownership_was_committed: bool
    ) -> bool:
        if _NEXT.get(current) is target:
            return True
        if target is CutoverPhase.REJECTED:
            return current in {CutoverPhase.PROPOSED, CutoverPhase.COMPATIBILITY_VALIDATED}
        if target in {
            CutoverPhase.DESTINATION_LOST,
            CutoverPhase.SOURCE_LOST,
            CutoverPhase.COORDINATOR_UNAVAILABLE,
        }:
            return current not in TERMINAL_PHASES
        if target in {CutoverPhase.ABORTING, CutoverPhase.FAILED_BEFORE_COMMIT}:
            return not ownership_was_committed and current not in TERMINAL_PHASES
        if target is CutoverPhase.ROLLED_BACK:
            return not ownership_was_committed and current in _FAILURE_BEFORE_COMMIT | {
                CutoverPhase.ABORTING
            }
        if target in {CutoverPhase.FAILED_AFTER_COMMIT, CutoverPhase.OPERATOR_REQUIRED}:
            return ownership_was_committed and current not in TERMINAL_PHASES
        return False

    def transition(
        self,
        identifier: str,
        *,
        expected: CutoverPhase,
        target: CutoverPhase,
        event_id: str,
        at_ms: int,
        payload_hash: str,
        state_hashes: tuple[str, ...] | None = None,
        commit_watermark: int | None = None,
        rollback_watermark: int | None = None,
        failure_reason: str | None = None,
    ) -> StateTransactionRecord:
        with self._lock:
            return self._transition_locked(
                identifier,
                expected=expected,
                target=target,
                event_id=event_id,
                at_ms=at_ms,
                payload_hash=payload_hash,
                state_hashes=state_hashes,
                commit_watermark=commit_watermark,
                rollback_watermark=rollback_watermark,
                failure_reason=failure_reason,
            )

    def _transition_locked(
        self,
        identifier: str,
        *,
        expected: CutoverPhase,
        target: CutoverPhase,
        event_id: str,
        at_ms: int,
        payload_hash: str,
        state_hashes: tuple[str, ...] | None = None,
        commit_watermark: int | None = None,
        rollback_watermark: int | None = None,
        failure_reason: str | None = None,
    ) -> StateTransactionRecord:
        current = self.transaction(identifier)
        if current.phase is target:
            matching = [entry for entry in self.journal(identifier) if entry.event_id == event_id]
            if len(matching) == 1:
                entry = matching[0]
                replay_matches = (
                    entry.from_phase is expected
                    and entry.to_phase is target
                    and entry.at_ms == at_ms
                    and entry.payload_hash == payload_hash
                    and (state_hashes is None or current.state_hashes == state_hashes)
                    and (commit_watermark is None or current.commit_watermark == commit_watermark)
                    and (
                        rollback_watermark is None
                        or current.rollback_watermark == rollback_watermark
                    )
                    and current.failure_reason == failure_reason
                )
                if replay_matches:
                    return current
                raise CoordinatorConflict("idempotent transition payload does not match journal")
        if current.phase is not expected:
            raise CoordinatorConflict(
                f"expected transaction phase {expected}, found {current.phase}"
            )
        journal = self.journal(identifier)
        if journal and at_ms < journal[-1].at_ms:
            raise CoordinatorConflict("transaction event time cannot move backwards")
        ownership_was_committed = current.phase in _POST_COMMIT or any(
            entry.to_phase is CutoverPhase.OWNERSHIP_COMMITTED for entry in journal
        )
        if not self._legal_target(
            current.phase,
            target,
            ownership_was_committed=ownership_was_committed,
        ):
            raise InvalidTransition(f"illegal Continuum transition {current.phase} -> {target}")
        if at_ms >= current.timeout_at_ms and target not in _DEADLINE_RECOVERY_TARGETS:
            raise CoordinatorConflict("transaction deadline expired")
        if len(journal) >= _MAX_JOURNAL_ENTRIES:
            raise CoordinatorConflict("transaction journal bound reached")
        if commit_watermark is not None:
            if target is not CutoverPhase.COMMIT_INTENT_RECORDED:
                raise InvalidTransition("the token cutover boundary is recorded with commit intent")
            if commit_watermark < current.commit_watermark:
                raise CoordinatorConflict("commit watermark cannot move backwards")
        if rollback_watermark is not None:
            if target is not CutoverPhase.COMMIT_INTENT_RECORDED:
                raise InvalidTransition("the rollback boundary is recorded with commit intent")
            if rollback_watermark < current.rollback_watermark:
                raise CoordinatorConflict("rollback watermark cannot move backwards")
        next_commit_watermark = (
            commit_watermark if commit_watermark is not None else current.commit_watermark
        )
        next_rollback_watermark = (
            rollback_watermark if rollback_watermark is not None else current.rollback_watermark
        )
        if next_rollback_watermark > next_commit_watermark:
            raise CoordinatorConflict("rollback watermark cannot exceed the commit watermark")
        payload = current.model_dump(mode="python")
        payload.update(
            {
                "phase": target,
                "version": current.version + 1,
                "state_hashes": state_hashes if state_hashes is not None else current.state_hashes,
                "commit_watermark": next_commit_watermark,
                "rollback_watermark": next_rollback_watermark,
                "failure_reason": failure_reason,
            }
        )
        updated = StateTransactionRecord.model_validate(payload)
        entry = JournalEntry(
            transaction_id=current.transaction_id,
            sequence=len(journal),
            event_id=event_id,
            from_phase=current.phase,
            to_phase=target,
            at_ms=at_ms,
            payload_hash=payload_hash,
        )
        try:
            with self._write() as connection:
                changed = connection.execute(
                    "UPDATE continuum_transactions SET payload=?,version=? "
                    "WHERE transaction_id=? AND version=?",
                    (self._json(updated), updated.version, identifier, current.version),
                ).rowcount
                if changed != 1:
                    raise CoordinatorConflict("concurrent transaction update")
                connection.execute(
                    "INSERT INTO continuum_journal VALUES(?,?,?,?)",
                    (identifier, entry.sequence, entry.event_id, self._json(entry)),
                )
        except sqlite3.IntegrityError as exc:
            raise CoordinatorConflict("duplicate transaction event") from exc
        except CoordinatorConflict:
            latest = self.transaction(identifier)
            if latest.version != current.version and latest.phase is target:
                return self._transition_locked(
                    identifier,
                    expected=expected,
                    target=target,
                    event_id=event_id,
                    at_ms=at_ms,
                    payload_hash=payload_hash,
                    state_hashes=state_hashes,
                    commit_watermark=commit_watermark,
                    rollback_watermark=rollback_watermark,
                    failure_reason=failure_reason,
                )
            raise
        return updated

    def commit_ownership(
        self, identifier: str, *, event_id: str, at_ms: int, state_version: int
    ) -> tuple[StateTransactionRecord, SessionLease]:
        with self._lock:
            return self._commit_ownership_locked(
                identifier,
                event_id=event_id,
                at_ms=at_ms,
                state_version=state_version,
            )

    def _commit_ownership_locked(
        self, identifier: str, *, event_id: str, at_ms: int, state_version: int
    ) -> tuple[StateTransactionRecord, SessionLease]:
        current = self.transaction(identifier)
        if current.phase is CutoverPhase.OWNERSHIP_COMMITTED:
            lease = self.lease(current.session_id)
            entries = [entry for entry in self.journal(identifier) if entry.event_id == event_id]
            if (
                len(entries) == 1
                and entries[0].from_phase is CutoverPhase.COMMIT_INTENT_RECORDED
                and entries[0].to_phase is CutoverPhase.OWNERSHIP_COMMITTED
                and entries[0].at_ms == at_ms
                and lease.last_committed_state_version == state_version
                and entries[0].payload_hash
                == hashlib.sha256(self._json(lease).encode()).hexdigest()
            ):
                return current, lease
            raise CoordinatorConflict("idempotent ownership commit does not match journal")
        if current.phase is not CutoverPhase.COMMIT_INTENT_RECORDED:
            raise InvalidTransition("ownership commit requires persisted commit intent")
        if at_ms >= current.timeout_at_ms:
            raise CoordinatorConflict("transaction deadline expired before ownership commit")
        lease = self.lease(current.session_id)
        if at_ms >= lease.expiration_ms:
            raise StaleOwner("source lease expired before ownership commit")
        if lease.owner_epoch != current.source_epoch or lease.owner_runtime != current.source_owner:
            raise StaleOwner("source lease changed before ownership commit")
        updated_lease = SessionLease(
            session_id=lease.session_id,
            owner_runtime=current.destination_candidate,
            owner_epoch=current.proposed_destination_epoch,
            fencing_token=lease.fencing_token + 1,
            expiration_ms=lease.expiration_ms,
            coordinator_version=lease.coordinator_version + 1,
            last_committed_state_version=state_version,
            last_committed_token_index=current.commit_watermark,
        )
        updated_transaction = StateTransactionRecord.model_validate(
            {
                **current.model_dump(mode="python"),
                "phase": CutoverPhase.OWNERSHIP_COMMITTED,
                "version": current.version + 1,
            }
        )
        journal = self.journal(identifier)
        if len(journal) >= _MAX_JOURNAL_ENTRIES:
            raise CoordinatorConflict("transaction journal bound reached")
        entry = JournalEntry(
            transaction_id=identifier,
            sequence=len(journal),
            event_id=event_id,
            from_phase=current.phase,
            to_phase=CutoverPhase.OWNERSHIP_COMMITTED,
            at_ms=at_ms,
            payload_hash=hashlib.sha256(self._json(updated_lease).encode()).hexdigest(),
        )
        try:
            with self._write() as connection:
                lease_changed = connection.execute(
                    "UPDATE continuum_leases SET payload=?,version=? "
                    "WHERE session_id=? AND version=?",
                    (
                        self._json(updated_lease),
                        updated_lease.coordinator_version,
                        lease.session_id,
                        lease.coordinator_version,
                    ),
                ).rowcount
                transaction_changed = connection.execute(
                    "UPDATE continuum_transactions SET payload=?,version=? "
                    "WHERE transaction_id=? AND version=?",
                    (
                        self._json(updated_transaction),
                        updated_transaction.version,
                        identifier,
                        current.version,
                    ),
                ).rowcount
                if lease_changed != 1 or transaction_changed != 1:
                    raise CoordinatorConflict("ownership compare-and-swap failed")
                connection.execute(
                    "INSERT INTO continuum_journal VALUES(?,?,?,?)",
                    (identifier, entry.sequence, entry.event_id, self._json(entry)),
                )
        except CoordinatorConflict:
            latest = self.transaction(identifier)
            if (
                latest.version != current.version
                and latest.phase is CutoverPhase.OWNERSHIP_COMMITTED
            ):
                return self._commit_ownership_locked(
                    identifier,
                    event_id=event_id,
                    at_ms=at_ms,
                    state_version=state_version,
                )
            raise
        return updated_transaction, updated_lease
