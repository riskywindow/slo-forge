"""Transactional SQLite store for persistent Genesis optimization lineage."""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from .models import (
    CandidateRecord,
    CounterexampleRecord,
    EvidenceFreshness,
    EvidenceRecord,
    EvidenceTargetKind,
    InvalidationEvent,
    LearnedConstraintRecord,
    LineageSnapshot,
    TaskFeatures,
    TransferRecord,
    TransformationQuery,
    TransformationRecord,
)
from .versioning import version_matches

_Record = TypeVar("_Record", bound=BaseModel)


class LineageError(RuntimeError):
    """Base class for persistent-lineage failures."""


class LineageConflict(LineageError):
    """Raised when an immutable identifier already exists."""


class LineageNotFound(LineageError):
    """Raised when a referenced lineage node does not exist."""


class LineageLimitExceeded(LineageError):
    """Raised rather than silently truncating a supposedly complete operation."""


class LineageQueryTimeout(LineageError):
    """Raised when SQLite exceeds an explicit query deadline."""


class LineageStore:
    def __init__(self, path: Path, *, timeout_seconds: float = 5.0) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.timeout_seconds = timeout_seconds
        self._connection = sqlite3.connect(path, timeout=timeout_seconds)
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute(f"PRAGMA busy_timeout = {int(timeout_seconds * 1000)}")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute("PRAGMA synchronous = FULL")
        self._initialize_schema()

    def __enter__(self) -> LineageStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._connection.close()

    def _initialize_schema(self) -> None:
        version = int(self._connection.execute("PRAGMA user_version").fetchone()[0])
        if version not in {0, 1}:
            raise LineageError(f"unsupported lineage database version {version}")
        if version == 1:
            return
        self._connection.executescript(
            """
            BEGIN IMMEDIATE;
            CREATE TABLE tasks (
                task_id TEXT PRIMARY KEY,
                model_family TEXT NOT NULL,
                hardware_architecture TEXT NOT NULL,
                document TEXT NOT NULL
            );
            CREATE TABLE candidates (
                candidate_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL REFERENCES tasks(task_id),
                disposition TEXT NOT NULL,
                created_at TEXT NOT NULL,
                document TEXT NOT NULL
            );
            CREATE TABLE transformations (
                transformation_id TEXT PRIMARY KEY,
                family TEXT NOT NULL,
                source_candidate_id TEXT NOT NULL REFERENCES candidates(candidate_id),
                outcome TEXT NOT NULL,
                created_at TEXT NOT NULL,
                document TEXT NOT NULL
            );
            CREATE TABLE evidence (
                evidence_id TEXT PRIMARY KEY,
                target_kind TEXT NOT NULL,
                target_id TEXT NOT NULL,
                freshness TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                document TEXT NOT NULL
            );
            CREATE TABLE evidence_dependencies (
                evidence_id TEXT NOT NULL REFERENCES evidence(evidence_id) ON DELETE CASCADE,
                kind TEXT NOT NULL,
                name TEXT NOT NULL,
                version TEXT NOT NULL,
                PRIMARY KEY (evidence_id, kind, name)
            );
            CREATE INDEX evidence_dependency_lookup
                ON evidence_dependencies(kind, name, version, evidence_id);
            CREATE INDEX evidence_target_lookup
                ON evidence(target_kind, target_id, freshness, evidence_id);
            CREATE TABLE counterexamples (
                counterexample_id TEXT PRIMARY KEY,
                candidate_id TEXT NOT NULL REFERENCES candidates(candidate_id),
                transformation_id TEXT REFERENCES transformations(transformation_id),
                created_at TEXT NOT NULL,
                document TEXT NOT NULL
            );
            CREATE TABLE constraints (
                constraint_id TEXT PRIMARY KEY,
                counterexample_id TEXT NOT NULL REFERENCES counterexamples(counterexample_id),
                transformation_family TEXT NOT NULL,
                transformation_id TEXT REFERENCES transformations(transformation_id),
                created_at TEXT NOT NULL,
                document TEXT NOT NULL
            );
            CREATE INDEX constraint_family_lookup
                ON constraints(transformation_family, transformation_id, constraint_id);
            CREATE TABLE invalidations (
                invalidation_id TEXT PRIMARY KEY,
                dependency_kind TEXT NOT NULL,
                dependency_name TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                document TEXT NOT NULL
            );
            CREATE TABLE transfers (
                transfer_id TEXT PRIMARY KEY,
                target_task_id TEXT NOT NULL REFERENCES tasks(task_id),
                transformation_id TEXT NOT NULL REFERENCES transformations(transformation_id),
                outcome TEXT NOT NULL,
                created_at TEXT NOT NULL,
                document TEXT NOT NULL
            );
            PRAGMA user_version = 1;
            COMMIT;
            """
        )

    @staticmethod
    def _document(record: BaseModel) -> str:
        return record.model_dump_json()

    @contextmanager
    def _deadline(self, timeout_seconds: float | None = None) -> Iterator[None]:
        timeout = self.timeout_seconds if timeout_seconds is None else timeout_seconds
        if timeout <= 0:
            raise ValueError("query timeout must be positive")
        deadline = time.monotonic() + timeout
        self._connection.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
        try:
            yield
        except sqlite3.OperationalError as exc:
            if "interrupted" in str(exc).lower():
                raise LineageQueryTimeout("lineage query exceeded its deadline") from exc
            raise
        finally:
            self._connection.set_progress_handler(None, 0)

    @staticmethod
    def _validate_limit(limit: int) -> int:
        if not 1 <= limit <= 100_000:
            raise ValueError("query limit must be between 1 and 100000")
        return limit

    def _exists(self, table: str, column: str, identifier: str) -> bool:
        allowed = {
            ("tasks", "task_id"),
            ("candidates", "candidate_id"),
            ("transformations", "transformation_id"),
            ("evidence", "evidence_id"),
            ("counterexamples", "counterexample_id"),
        }
        if (table, column) not in allowed:
            raise ValueError("unsupported lineage existence query")
        row = self._connection.execute(
            f"SELECT 1 FROM {table} WHERE {column} = ?", (identifier,)
        ).fetchone()
        return row is not None

    def _require(self, table: str, column: str, identifier: str) -> None:
        if not self._exists(table, column, identifier):
            raise LineageNotFound(f"missing {table}.{column}={identifier}")

    def _insert_task(self, task: TaskFeatures) -> None:
        self._connection.execute(
            "INSERT INTO tasks(task_id, model_family, hardware_architecture, document) "
            "VALUES (?, ?, ?, ?)",
            (task.task_id, task.model_family, task.hardware_architecture, self._document(task)),
        )

    def record_task(self, task: TaskFeatures) -> None:
        try:
            with self._connection:
                self._insert_task(task)
        except sqlite3.IntegrityError as exc:
            raise LineageConflict(f"task identifier already exists: {task.task_id}") from exc

    def _insert_candidate(self, candidate: CandidateRecord) -> None:
        self._require("tasks", "task_id", candidate.task_id)
        for parent in candidate.parent_candidate_ids:
            self._require("candidates", "candidate_id", parent)
        self._connection.execute(
            "INSERT INTO candidates(candidate_id, task_id, disposition, created_at, document) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                candidate.candidate_id,
                candidate.task_id,
                candidate.disposition.value,
                candidate.created_at.isoformat(),
                self._document(candidate),
            ),
        )

    def record_candidate(self, candidate: CandidateRecord) -> None:
        try:
            with self._connection:
                self._insert_candidate(candidate)
                self._verify_candidate_links(candidate)
        except sqlite3.IntegrityError as exc:
            raise LineageConflict(
                f"candidate identifier already exists: {candidate.candidate_id}"
            ) from exc

    def _insert_transformation(self, transformation: TransformationRecord) -> None:
        self._require("candidates", "candidate_id", transformation.source_candidate_id)
        if transformation.target_candidate_id is not None:
            self._require("candidates", "candidate_id", transformation.target_candidate_id)
        for parent in transformation.parent_transformation_ids:
            self._require("transformations", "transformation_id", parent)
        self._connection.execute(
            "INSERT INTO transformations("
            "transformation_id, family, source_candidate_id, outcome, created_at, document"
            ") VALUES (?, ?, ?, ?, ?, ?)",
            (
                transformation.transformation_id,
                transformation.family,
                transformation.source_candidate_id,
                transformation.outcome.value,
                transformation.created_at.isoformat(),
                self._document(transformation),
            ),
        )

    def record_transformation(self, transformation: TransformationRecord) -> None:
        try:
            with self._connection:
                self._insert_transformation(transformation)
        except sqlite3.IntegrityError as exc:
            raise LineageConflict(
                f"transformation identifier already exists: {transformation.transformation_id}"
            ) from exc

    def _insert_evidence(self, evidence: EvidenceRecord) -> None:
        if evidence.target_kind is EvidenceTargetKind.CANDIDATE:
            self._require("candidates", "candidate_id", evidence.target_id)
        else:
            self._require("transformations", "transformation_id", evidence.target_id)
        self._connection.execute(
            "INSERT INTO evidence("
            "evidence_id, target_kind, target_id, freshness, observed_at, document"
            ") VALUES (?, ?, ?, ?, ?, ?)",
            (
                evidence.evidence_id,
                evidence.target_kind.value,
                evidence.target_id,
                evidence.freshness.value,
                evidence.observed_at.isoformat(),
                self._document(evidence),
            ),
        )
        self._connection.executemany(
            "INSERT INTO evidence_dependencies(evidence_id, kind, name, version) "
            "VALUES (?, ?, ?, ?)",
            (
                (evidence.evidence_id, item.kind.value, item.name, item.version)
                for item in evidence.dependencies
            ),
        )

    def record_evidence(self, evidence: EvidenceRecord) -> None:
        try:
            with self._connection:
                self._insert_evidence(evidence)
        except sqlite3.IntegrityError as exc:
            raise LineageConflict(
                f"evidence identifier already exists: {evidence.evidence_id}"
            ) from exc

    def record_candidate_bundle(
        self,
        candidate: CandidateRecord,
        transformations: Sequence[TransformationRecord],
        evidence: Sequence[EvidenceRecord],
    ) -> None:
        """Atomically add a candidate and its newly produced lineage records."""

        try:
            with self._connection:
                self._insert_candidate(candidate)
                for transformation in transformations:
                    self._insert_transformation(transformation)
                for record in evidence:
                    self._insert_evidence(record)
                self._verify_candidate_links(candidate)
        except sqlite3.IntegrityError as exc:
            raise LineageConflict("candidate bundle conflicts with immutable lineage") from exc

    def _verify_candidate_links(self, candidate: CandidateRecord) -> None:
        for transformation_id in candidate.transformation_ids:
            self._require("transformations", "transformation_id", transformation_id)
        for objective in candidate.objectives:
            self._require("evidence", "evidence_id", objective.evidence_id)

    def record_counterexample(self, counterexample: CounterexampleRecord) -> None:
        self._require("candidates", "candidate_id", counterexample.candidate_id)
        if counterexample.transformation_id is not None:
            self._require("transformations", "transformation_id", counterexample.transformation_id)
        try:
            with self._connection:
                self._connection.execute(
                    "INSERT INTO counterexamples("
                    "counterexample_id, candidate_id, transformation_id, created_at, document"
                    ") VALUES (?, ?, ?, ?, ?)",
                    (
                        counterexample.counterexample_id,
                        counterexample.candidate_id,
                        counterexample.transformation_id,
                        counterexample.created_at.isoformat(),
                        self._document(counterexample),
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise LineageConflict("counterexample identifier already exists") from exc

    def record_constraint(self, constraint: LearnedConstraintRecord) -> None:
        self._require("counterexamples", "counterexample_id", constraint.counterexample_id)
        if constraint.transformation_id is not None:
            self._require("transformations", "transformation_id", constraint.transformation_id)
        try:
            with self._connection:
                self._connection.execute(
                    "INSERT INTO constraints("
                    "constraint_id, counterexample_id, transformation_family, "
                    "transformation_id, created_at, document"
                    ") VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        constraint.constraint_id,
                        constraint.counterexample_id,
                        constraint.transformation_family,
                        constraint.transformation_id,
                        constraint.created_at.isoformat(),
                        self._document(constraint),
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise LineageConflict("constraint identifier already exists") from exc

    def record_transfer(self, transfer: TransferRecord) -> None:
        self._require("tasks", "task_id", transfer.target_task_id)
        self._require("transformations", "transformation_id", transfer.transformation_id)
        for evidence_id in transfer.source_evidence_ids:
            self._require("evidence", "evidence_id", evidence_id)
        try:
            with self._connection:
                self._connection.execute(
                    "INSERT INTO transfers("
                    "transfer_id, target_task_id, transformation_id, outcome, created_at, document"
                    ") VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        transfer.transfer_id,
                        transfer.target_task_id,
                        transfer.transformation_id,
                        transfer.outcome.value,
                        transfer.created_at.isoformat(),
                        self._document(transfer),
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise LineageConflict("transfer identifier already exists") from exc

    def invalidate_dependency(
        self, event: InvalidationEvent, *, maximum_evidence: int = 10_000
    ) -> int:
        maximum = self._validate_limit(maximum_evidence)
        try:
            with self._connection, self._deadline():
                rows = self._connection.execute(
                    "SELECT e.evidence_id, e.document, d.version "
                    "FROM evidence AS e JOIN evidence_dependencies AS d "
                    "ON e.evidence_id = d.evidence_id "
                    "WHERE e.freshness = ? AND d.kind = ? AND d.name = ? "
                    "ORDER BY e.evidence_id LIMIT ?",
                    (
                        EvidenceFreshness.FRESH.value,
                        event.selector.kind.value,
                        event.selector.name,
                        maximum + 1,
                    ),
                ).fetchall()
                if len(rows) > maximum:
                    raise LineageLimitExceeded("dependency invalidation exceeds maximum_evidence")
                affected: list[EvidenceRecord] = []
                for _, document, version in rows:
                    if version_matches(str(version), event.selector.version_range):
                        record = EvidenceRecord.model_validate_json(document, strict=True)
                        affected.append(
                            record.model_copy(
                                update={
                                    "freshness": EvidenceFreshness.STALE,
                                    "invalidation_event_ids": (
                                        *record.invalidation_event_ids,
                                        event.invalidation_id,
                                    ),
                                }
                            )
                        )
                self._connection.execute(
                    "INSERT INTO invalidations("
                    "invalidation_id, dependency_kind, dependency_name, occurred_at, document"
                    ") VALUES (?, ?, ?, ?, ?)",
                    (
                        event.invalidation_id,
                        event.selector.kind.value,
                        event.selector.name,
                        event.occurred_at.isoformat(),
                        self._document(event),
                    ),
                )
                for record in affected:
                    self._connection.execute(
                        "UPDATE evidence SET freshness = ?, document = ? WHERE evidence_id = ?",
                        (
                            record.freshness.value,
                            self._document(record),
                            record.evidence_id,
                        ),
                    )
                return len(affected)
        except sqlite3.IntegrityError as exc:
            raise LineageConflict("invalidation identifier already exists") from exc

    def _list_records(
        self,
        table: str,
        model: type[_Record],
        *,
        limit: int,
        order_column: str,
    ) -> tuple[_Record, ...]:
        allowed = {
            ("tasks", "task_id"),
            ("candidates", "candidate_id"),
            ("transformations", "transformation_id"),
            ("evidence", "evidence_id"),
            ("counterexamples", "counterexample_id"),
            ("constraints", "constraint_id"),
            ("transfers", "transfer_id"),
            ("invalidations", "invalidation_id"),
        }
        if (table, order_column) not in allowed:
            raise ValueError("unsupported lineage listing")
        maximum = self._validate_limit(limit)
        with self._deadline():
            rows = self._connection.execute(
                f"SELECT document FROM {table} ORDER BY {order_column} LIMIT ?", (maximum,)
            ).fetchall()
        return tuple(model.model_validate_json(row[0], strict=True) for row in rows)

    def list_tasks(self, *, limit: int = 1000) -> tuple[TaskFeatures, ...]:
        return self._list_records("tasks", TaskFeatures, limit=limit, order_column="task_id")

    def list_candidates(self, *, limit: int = 1000) -> tuple[CandidateRecord, ...]:
        return self._list_records(
            "candidates", CandidateRecord, limit=limit, order_column="candidate_id"
        )

    def list_transformations(self, *, limit: int = 1000) -> tuple[TransformationRecord, ...]:
        return self._list_records(
            "transformations",
            TransformationRecord,
            limit=limit,
            order_column="transformation_id",
        )

    def query_transformations(self, query: TransformationQuery) -> tuple[TransformationRecord, ...]:
        matches: list[TransformationRecord] = []
        for item in self.list_transformations(limit=query.scan_limit):
            if query.model_family is not None and query.model_family not in (
                item.applicable_model_families
            ):
                continue
            if query.operation is not None and query.operation not in item.applicable_operations:
                continue
            if query.hardware_architecture is not None and query.hardware_architecture not in (
                item.applicable_hardware
            ):
                continue
            if query.workload_regime is not None and query.workload_regime not in (
                item.applicable_workloads
            ):
                continue
            if query.family is not None and query.family != item.family:
                continue
            if query.outcome is not None and query.outcome is not item.outcome:
                continue
            matches.append(item)
            if len(matches) == query.limit:
                break
        return tuple(matches)

    def list_evidence(self, *, limit: int = 1000) -> tuple[EvidenceRecord, ...]:
        return self._list_records(
            "evidence", EvidenceRecord, limit=limit, order_column="evidence_id"
        )

    def evidence_for_transformation(
        self, transformation_id: str, *, limit: int = 1000
    ) -> tuple[EvidenceRecord, ...]:
        maximum = self._validate_limit(limit)
        with self._deadline():
            rows = self._connection.execute(
                "SELECT document FROM evidence WHERE target_kind = ? AND target_id = ? "
                "ORDER BY evidence_id LIMIT ?",
                (EvidenceTargetKind.TRANSFORMATION.value, transformation_id, maximum),
            ).fetchall()
        return tuple(EvidenceRecord.model_validate_json(row[0], strict=True) for row in rows)

    def list_counterexamples(self, *, limit: int = 1000) -> tuple[CounterexampleRecord, ...]:
        return self._list_records(
            "counterexamples",
            CounterexampleRecord,
            limit=limit,
            order_column="counterexample_id",
        )

    def list_constraints(self, *, limit: int = 1000) -> tuple[LearnedConstraintRecord, ...]:
        return self._list_records(
            "constraints",
            LearnedConstraintRecord,
            limit=limit,
            order_column="constraint_id",
        )

    def list_transfers(self, *, limit: int = 1000) -> tuple[TransferRecord, ...]:
        return self._list_records(
            "transfers", TransferRecord, limit=limit, order_column="transfer_id"
        )

    def list_invalidations(self, *, limit: int = 1000) -> tuple[InvalidationEvent, ...]:
        return self._list_records(
            "invalidations",
            InvalidationEvent,
            limit=limit,
            order_column="invalidation_id",
        )

    def task_for_candidate(self, candidate_id: str) -> TaskFeatures:
        with self._deadline():
            row = self._connection.execute(
                "SELECT t.document FROM tasks AS t JOIN candidates AS c ON t.task_id = c.task_id "
                "WHERE c.candidate_id = ?",
                (candidate_id,),
            ).fetchone()
        if row is None:
            raise LineageNotFound(f"candidate has no task: {candidate_id}")
        return TaskFeatures.model_validate_json(row[0], strict=True)

    def get_task(self, task_id: str) -> TaskFeatures:
        with self._deadline():
            row = self._connection.execute(
                "SELECT document FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
        if row is None:
            raise LineageNotFound(f"task does not exist: {task_id}")
        return TaskFeatures.model_validate_json(row[0], strict=True)

    def snapshot(self, *, exported_at: datetime, maximum_records: int = 100_000) -> LineageSnapshot:
        maximum = self._validate_limit(maximum_records)
        tables = (
            "tasks",
            "candidates",
            "transformations",
            "evidence",
            "counterexamples",
            "constraints",
            "transfers",
            "invalidations",
        )
        with self._deadline():
            total = sum(
                int(self._connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
                for table in tables
            )
        if total > maximum:
            raise LineageLimitExceeded("lineage export exceeds maximum_records")
        return LineageSnapshot(
            exported_at=exported_at,
            tasks=self.list_tasks(limit=maximum),
            candidates=self.list_candidates(limit=maximum),
            transformations=self.list_transformations(limit=maximum),
            evidence=self.list_evidence(limit=maximum),
            counterexamples=self.list_counterexamples(limit=maximum),
            constraints=self.list_constraints(limit=maximum),
            transfers=self.list_transfers(limit=maximum),
            invalidations=self.list_invalidations(limit=maximum),
        )
