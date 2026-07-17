"""SQLite-backed, hash-checked Helix learning transaction state machine."""

from __future__ import annotations

import json
import math
import sqlite3
from enum import StrEnum
from hashlib import sha256
from itertools import pairwise
from pathlib import Path
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _TransactionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class LearningState(StrEnum):
    OBSERVED = "observed"
    CAPTURE_PROPOSED = "capture_proposed"
    CAPTURED = "captured"
    BRANCH_PLAN_CREATED = "branch_plan_created"
    FORKING = "forking"
    ROLLOUTS_RUNNING = "rollouts_running"
    ROLLOUTS_COMPLETED = "rollouts_completed"
    REWARDS_VALIDATING = "rewards_validating"
    CREDIT_ASSIGNING = "credit_assigning"
    TRAINING_BATCH_BUILDING = "training_batch_building"
    TRAINING = "training"
    CANDIDATE_READY = "candidate_ready"
    ALGORITHM_VALIDATING = "algorithm_validating"
    QUALITY_VALIDATING = "quality_validating"
    SERVING_VALIDATING = "serving_validating"
    COMPATIBILITY_VALIDATING = "compatibility_validating"
    PROMOTION_CAPSULE_READY = "promotion_capsule_ready"
    SHADOWING = "shadowing"
    CANARYING = "canarying"
    PROMOTION_INTENT_RECORDED = "promotion_intent_recorded"
    PROMOTING = "promoting"
    ACTIVE = "active"
    POST_PROMOTION_MONITORING = "post_promotion_monitoring"
    COMPLETED = "completed"

    EXPERIENCE_REJECTED = "experience_rejected"
    CAPTURE_FAILED = "capture_failed"
    BRANCH_FAILED = "branch_failed"
    ROLLOUT_FAILED = "rollout_failed"
    REWARD_REJECTED = "reward_rejected"
    CREDIT_REJECTED = "credit_rejected"
    TRAINING_FAILED = "training_failed"
    CANDIDATE_INVALID = "candidate_invalid"
    QUALITY_REGRESSION = "quality_regression"
    SAFETY_REGRESSION = "safety_regression"
    SERVING_REGRESSION = "serving_regression"
    COMPATIBILITY_REJECTED = "compatibility_rejected"
    SHADOW_REJECTED = "shadow_rejected"
    CANARY_REJECTED = "canary_rejected"
    PROMOTION_FAILED = "promotion_failed"
    ROLLED_BACK = "rolled_back"
    OPERATOR_REQUIRED = "operator_required"

    @property
    def terminal(self) -> bool:
        return self in _TERMINAL_STATES


_SUCCESS_PATH = (
    LearningState.OBSERVED,
    LearningState.CAPTURE_PROPOSED,
    LearningState.CAPTURED,
    LearningState.BRANCH_PLAN_CREATED,
    LearningState.FORKING,
    LearningState.ROLLOUTS_RUNNING,
    LearningState.ROLLOUTS_COMPLETED,
    LearningState.REWARDS_VALIDATING,
    LearningState.CREDIT_ASSIGNING,
    LearningState.TRAINING_BATCH_BUILDING,
    LearningState.TRAINING,
    LearningState.CANDIDATE_READY,
    LearningState.ALGORITHM_VALIDATING,
    LearningState.QUALITY_VALIDATING,
    LearningState.SERVING_VALIDATING,
    LearningState.COMPATIBILITY_VALIDATING,
    LearningState.PROMOTION_CAPSULE_READY,
    LearningState.SHADOWING,
    LearningState.CANARYING,
    LearningState.PROMOTION_INTENT_RECORDED,
    LearningState.PROMOTING,
    LearningState.ACTIVE,
    LearningState.POST_PROMOTION_MONITORING,
    LearningState.COMPLETED,
)

_TERMINAL_STATES = frozenset(
    {
        LearningState.COMPLETED,
        LearningState.EXPERIENCE_REJECTED,
        LearningState.CAPTURE_FAILED,
        LearningState.BRANCH_FAILED,
        LearningState.ROLLOUT_FAILED,
        LearningState.REWARD_REJECTED,
        LearningState.CREDIT_REJECTED,
        LearningState.TRAINING_FAILED,
        LearningState.CANDIDATE_INVALID,
        LearningState.QUALITY_REGRESSION,
        LearningState.SAFETY_REGRESSION,
        LearningState.SERVING_REGRESSION,
        LearningState.COMPATIBILITY_REJECTED,
        LearningState.SHADOW_REJECTED,
        LearningState.CANARY_REJECTED,
        LearningState.PROMOTION_FAILED,
        LearningState.ROLLED_BACK,
        LearningState.OPERATOR_REQUIRED,
    }
)

_FAILURE_TRANSITIONS: dict[LearningState, frozenset[LearningState]] = {
    LearningState.OBSERVED: frozenset({LearningState.EXPERIENCE_REJECTED}),
    LearningState.CAPTURE_PROPOSED: frozenset({LearningState.CAPTURE_FAILED}),
    LearningState.CAPTURED: frozenset({LearningState.BRANCH_FAILED}),
    LearningState.BRANCH_PLAN_CREATED: frozenset({LearningState.BRANCH_FAILED}),
    LearningState.FORKING: frozenset({LearningState.BRANCH_FAILED}),
    LearningState.ROLLOUTS_RUNNING: frozenset({LearningState.ROLLOUT_FAILED}),
    LearningState.ROLLOUTS_COMPLETED: frozenset({LearningState.REWARD_REJECTED}),
    LearningState.REWARDS_VALIDATING: frozenset({LearningState.REWARD_REJECTED}),
    LearningState.CREDIT_ASSIGNING: frozenset({LearningState.CREDIT_REJECTED}),
    LearningState.TRAINING_BATCH_BUILDING: frozenset(
        {LearningState.CREDIT_REJECTED, LearningState.TRAINING_FAILED}
    ),
    LearningState.TRAINING: frozenset({LearningState.TRAINING_FAILED}),
    LearningState.CANDIDATE_READY: frozenset({LearningState.CANDIDATE_INVALID}),
    LearningState.ALGORITHM_VALIDATING: frozenset({LearningState.CANDIDATE_INVALID}),
    LearningState.QUALITY_VALIDATING: frozenset(
        {LearningState.QUALITY_REGRESSION, LearningState.SAFETY_REGRESSION}
    ),
    LearningState.SERVING_VALIDATING: frozenset({LearningState.SERVING_REGRESSION}),
    LearningState.COMPATIBILITY_VALIDATING: frozenset({LearningState.COMPATIBILITY_REJECTED}),
    LearningState.PROMOTION_CAPSULE_READY: frozenset({LearningState.CANDIDATE_INVALID}),
    LearningState.SHADOWING: frozenset({LearningState.SHADOW_REJECTED}),
    LearningState.CANARYING: frozenset({LearningState.CANARY_REJECTED}),
    LearningState.PROMOTION_INTENT_RECORDED: frozenset({LearningState.PROMOTION_FAILED}),
    LearningState.PROMOTING: frozenset({LearningState.PROMOTION_FAILED}),
    LearningState.ACTIVE: frozenset({LearningState.ROLLED_BACK}),
    LearningState.POST_PROMOTION_MONITORING: frozenset({LearningState.ROLLED_BACK}),
}

_ALLOWED: dict[LearningState, frozenset[LearningState]] = {}
for _source, _target in pairwise(_SUCCESS_PATH):
    _ALLOWED[_source] = frozenset({_target}) | _FAILURE_TRANSITIONS.get(_source, frozenset())


class ArtifactReference(_TransactionModel):
    artifact_id: Annotated[str, Field(min_length=1, max_length=256)]
    artifact_kind: Annotated[str, Field(min_length=1, max_length=160)]
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    uri: Annotated[str, Field(min_length=1, max_length=4096)]
    immutable: bool = True


class TransactionEvent(_TransactionModel):
    sequence: Annotated[int, Field(ge=0)]
    state_before: LearningState
    state_after: LearningState
    reason: Annotated[str, Field(min_length=1, max_length=2048)]
    evidence_artifact_ids: Annotated[tuple[str, ...], Field(max_length=1024)] = ()
    observed_at_ms: Annotated[int, Field(ge=0)]


class LearningTransactionRecord(_TransactionModel):
    schema_version: str = "sloforge.helix.learning-transaction/v1"
    transaction_id: Annotated[str, Field(min_length=1, max_length=160)]
    deployment: Annotated[str, Field(min_length=1, max_length=160)]
    champion_policy_epoch_id: Annotated[str, Field(min_length=1, max_length=160)]
    candidate_policy_epoch_id: Annotated[str | None, Field(max_length=160)] = None
    trigger_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    seed: Annotated[int, Field(ge=0, le=2**64 - 1)]
    state: LearningState
    sequence: Annotated[int, Field(ge=0)]
    evidence_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    cost_usd: Annotated[float, Field(ge=0.0)]
    artifacts: Annotated[tuple[ArtifactReference, ...], Field(max_length=10_000)]

    @model_validator(mode="after")
    def finite_cost(self) -> Self:
        if not math.isfinite(self.cost_usd):
            raise ValueError("transaction cost must be finite")
        return self


def _canonical_hash(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


class LearningTransactionStore:
    """Durable local coordinator with transactional state/event/artifact publication."""

    def __init__(self, database: str | Path, *, max_artifacts: int = 10_000) -> None:
        if not 1 <= max_artifacts <= 1_000_000:
            raise ValueError("artifact bound must be in 1..1000000")
        self.max_artifacts = max_artifacts
        self._connection = sqlite3.connect(str(database), timeout=5.0, isolation_level="IMMEDIATE")
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS learning_transactions (
                transaction_id TEXT PRIMARY KEY,
                deployment TEXT NOT NULL,
                champion_policy_epoch_id TEXT NOT NULL,
                candidate_policy_epoch_id TEXT,
                trigger_hash TEXT NOT NULL,
                seed INTEGER NOT NULL CHECK(seed >= 0),
                state TEXT NOT NULL,
                sequence INTEGER NOT NULL CHECK(sequence >= 0),
                evidence_hash TEXT NOT NULL,
                created_at_ms INTEGER NOT NULL CHECK(created_at_ms >= 0),
                updated_at_ms INTEGER NOT NULL CHECK(updated_at_ms >= 0)
            );
            CREATE TABLE IF NOT EXISTS learning_events (
                transaction_id TEXT NOT NULL REFERENCES learning_transactions(transaction_id),
                sequence INTEGER NOT NULL CHECK(sequence >= 0),
                state_before TEXT NOT NULL,
                state_after TEXT NOT NULL,
                reason TEXT NOT NULL,
                evidence_artifact_ids_json TEXT NOT NULL,
                observed_at_ms INTEGER NOT NULL CHECK(observed_at_ms >= 0),
                PRIMARY KEY(transaction_id, sequence)
            );
            CREATE TABLE IF NOT EXISTS learning_artifacts (
                transaction_id TEXT NOT NULL REFERENCES learning_transactions(transaction_id),
                artifact_id TEXT NOT NULL,
                artifact_kind TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                uri TEXT NOT NULL,
                immutable INTEGER NOT NULL CHECK(immutable IN (0,1)),
                PRIMARY KEY(transaction_id, artifact_id)
            );
            CREATE TABLE IF NOT EXISTS learning_costs (
                transaction_id TEXT NOT NULL REFERENCES learning_transactions(transaction_id),
                cost_id TEXT NOT NULL,
                amount_usd REAL NOT NULL CHECK(amount_usd >= 0),
                source TEXT NOT NULL,
                PRIMARY KEY(transaction_id, cost_id)
            );
            CREATE TABLE IF NOT EXISTS learning_resources (
                transaction_id TEXT NOT NULL REFERENCES learning_transactions(transaction_id),
                usage_id TEXT NOT NULL,
                resource_kind TEXT NOT NULL,
                quantity REAL NOT NULL CHECK(quantity >= 0),
                unit TEXT NOT NULL,
                PRIMARY KEY(transaction_id, usage_id)
            );
            """
        )

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def create(
        self,
        *,
        transaction_id: str,
        deployment: str,
        champion_policy_epoch_id: str,
        trigger_hash: str,
        seed: int,
        observed_at_ms: int,
    ) -> LearningTransactionRecord:
        if not 0 <= seed <= 2**63 - 1:
            raise ValueError("SQLite-backed transaction seed must fit a signed 64-bit integer")
        if len(trigger_hash) != 64 or any(item not in "0123456789abcdef" for item in trigger_hash):
            raise ValueError("trigger hash must be a lowercase sha256 digest")
        with self._connection:
            existing = self._connection.execute(
                "SELECT deployment, champion_policy_epoch_id, trigger_hash, seed, created_at_ms "
                "FROM learning_transactions WHERE transaction_id=?",
                (transaction_id,),
            ).fetchone()
            if existing is not None:
                identity = (
                    existing["deployment"],
                    existing["champion_policy_epoch_id"],
                    existing["trigger_hash"],
                    existing["seed"],
                    existing["created_at_ms"],
                )
                requested = (
                    deployment,
                    champion_policy_epoch_id,
                    trigger_hash,
                    seed,
                    observed_at_ms,
                )
                if identity != requested:
                    raise ValueError(
                        "transaction identity was reused with different coordinator inputs"
                    )
                return self.transaction(transaction_id)
            self._connection.execute(
                "INSERT INTO learning_transactions VALUES (?, ?, ?, NULL, ?, ?, ?, 0, ?, ?, ?)",
                (
                    transaction_id,
                    deployment,
                    champion_policy_epoch_id,
                    trigger_hash,
                    seed,
                    LearningState.OBSERVED.value,
                    "0" * 64,
                    observed_at_ms,
                    observed_at_ms,
                ),
            )
            self._connection.execute(
                "INSERT INTO learning_events VALUES (?, 0, ?, ?, ?, '[]', ?)",
                (
                    transaction_id,
                    LearningState.OBSERVED.value,
                    LearningState.OBSERVED.value,
                    "production or synthetic experience observed",
                    observed_at_ms,
                ),
            )
            self._refresh_hash(transaction_id)
        return self.transaction(transaction_id)

    def _artifacts(self, transaction_id: str) -> tuple[ArtifactReference, ...]:
        rows = self._connection.execute(
            "SELECT * FROM learning_artifacts WHERE transaction_id=? ORDER BY artifact_id",
            (transaction_id,),
        ).fetchall()
        return tuple(
            ArtifactReference(
                artifact_id=row["artifact_id"],
                artifact_kind=row["artifact_kind"],
                sha256=row["sha256"],
                uri=row["uri"],
                immutable=bool(row["immutable"]),
            )
            for row in rows
        )

    def _cost(self, transaction_id: str) -> float:
        row = self._connection.execute(
            "SELECT COALESCE(SUM(amount_usd), 0.0) AS amount FROM learning_costs "
            "WHERE transaction_id=?",
            (transaction_id,),
        ).fetchone()
        return float(row["amount"])

    def _evidence_document(self, transaction_id: str) -> dict[str, object]:
        row = self._connection.execute(
            "SELECT * FROM learning_transactions WHERE transaction_id=?", (transaction_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown learning transaction {transaction_id!r}")
        events = [
            dict(item)
            for item in self._connection.execute(
                "SELECT * FROM learning_events WHERE transaction_id=? ORDER BY sequence",
                (transaction_id,),
            ).fetchall()
        ]
        resources = [
            dict(item)
            for item in self._connection.execute(
                "SELECT * FROM learning_resources WHERE transaction_id=? ORDER BY usage_id",
                (transaction_id,),
            ).fetchall()
        ]
        costs = [
            dict(item)
            for item in self._connection.execute(
                "SELECT * FROM learning_costs WHERE transaction_id=? ORDER BY cost_id",
                (transaction_id,),
            ).fetchall()
        ]
        return {
            "transaction_id": row["transaction_id"],
            "deployment": row["deployment"],
            "champion_policy_epoch_id": row["champion_policy_epoch_id"],
            "candidate_policy_epoch_id": row["candidate_policy_epoch_id"],
            "trigger_hash": row["trigger_hash"],
            "seed": row["seed"],
            "state": row["state"],
            "sequence": row["sequence"],
            "events": events,
            "artifacts": [item.model_dump(mode="json") for item in self._artifacts(transaction_id)],
            "costs": costs,
            "resources": resources,
        }

    def _refresh_hash(self, transaction_id: str) -> str:
        digest = _canonical_hash(self._evidence_document(transaction_id))
        self._connection.execute(
            "UPDATE learning_transactions SET evidence_hash=? WHERE transaction_id=?",
            (digest, transaction_id),
        )
        return digest

    def transaction(self, transaction_id: str) -> LearningTransactionRecord:
        row = self._connection.execute(
            "SELECT * FROM learning_transactions WHERE transaction_id=?", (transaction_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown learning transaction {transaction_id!r}")
        digest = _canonical_hash(self._evidence_document(transaction_id))
        if digest != row["evidence_hash"]:
            raise ValueError("learning transaction evidence failed integrity validation")
        return LearningTransactionRecord(
            transaction_id=row["transaction_id"],
            deployment=row["deployment"],
            champion_policy_epoch_id=row["champion_policy_epoch_id"],
            candidate_policy_epoch_id=row["candidate_policy_epoch_id"],
            trigger_hash=row["trigger_hash"],
            seed=row["seed"],
            state=LearningState(row["state"]),
            sequence=row["sequence"],
            evidence_hash=row["evidence_hash"],
            cost_usd=self._cost(transaction_id),
            artifacts=self._artifacts(transaction_id),
        )

    def events(self, transaction_id: str) -> tuple[TransactionEvent, ...]:
        self.transaction(transaction_id)
        rows = self._connection.execute(
            "SELECT * FROM learning_events WHERE transaction_id=? ORDER BY sequence",
            (transaction_id,),
        ).fetchall()
        return tuple(
            TransactionEvent(
                sequence=row["sequence"],
                state_before=LearningState(row["state_before"]),
                state_after=LearningState(row["state_after"]),
                reason=row["reason"],
                evidence_artifact_ids=tuple(
                    str(item) for item in json.loads(row["evidence_artifact_ids_json"])
                ),
                observed_at_ms=row["observed_at_ms"],
            )
            for row in rows
        )

    def add_artifact(self, transaction_id: str, artifact: ArtifactReference) -> None:
        if not artifact.immutable:
            raise ValueError("raw learning evidence must be immutable")
        with self._connection:
            existing_row = self._connection.execute(
                "SELECT * FROM learning_artifacts WHERE transaction_id=? AND artifact_id=?",
                (transaction_id, artifact.artifact_id),
            ).fetchone()
            if existing_row is not None:
                existing = ArtifactReference(
                    artifact_id=existing_row["artifact_id"],
                    artifact_kind=existing_row["artifact_kind"],
                    sha256=existing_row["sha256"],
                    uri=existing_row["uri"],
                    immutable=bool(existing_row["immutable"]),
                )
                if existing != artifact:
                    raise ValueError("artifact identity was reused with changed evidence")
                return
            count = self._connection.execute(
                "SELECT COUNT(*) AS count FROM learning_artifacts WHERE transaction_id=?",
                (transaction_id,),
            ).fetchone()["count"]
            if count >= self.max_artifacts:
                raise ValueError("learning transaction artifact bound reached")
            try:
                self._connection.execute(
                    "INSERT INTO learning_artifacts VALUES (?, ?, ?, ?, ?, 1)",
                    (
                        transaction_id,
                        artifact.artifact_id,
                        artifact.artifact_kind,
                        artifact.sha256,
                        artifact.uri,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                conflicting = next(
                    (
                        item
                        for item in self._artifacts(transaction_id)
                        if item.artifact_id == artifact.artifact_id
                    ),
                    None,
                )
                if conflicting != artifact:
                    raise ValueError("artifact identity was reused with changed evidence") from exc
            self._refresh_hash(transaction_id)

    def record_cost(
        self,
        transaction_id: str,
        *,
        cost_id: str,
        amount_usd: float,
        source: str,
    ) -> None:
        if not math.isfinite(amount_usd) or amount_usd < 0.0:
            raise ValueError("cost must be finite and non-negative")
        with self._connection:
            existing = self._connection.execute(
                "SELECT amount_usd, source FROM learning_costs "
                "WHERE transaction_id=? AND cost_id=?",
                (transaction_id, cost_id),
            ).fetchone()
            if existing is not None:
                if float(existing["amount_usd"]) != amount_usd or existing["source"] != source:
                    raise ValueError("cost identity was reused with different accounting")
                return
            self._connection.execute(
                "INSERT INTO learning_costs VALUES (?, ?, ?, ?)",
                (transaction_id, cost_id, amount_usd, source),
            )
            self._refresh_hash(transaction_id)

    def record_resource(
        self,
        transaction_id: str,
        *,
        usage_id: str,
        resource_kind: str,
        quantity: float,
        unit: str,
    ) -> None:
        if not math.isfinite(quantity) or quantity < 0.0:
            raise ValueError("resource quantity must be finite and non-negative")
        with self._connection:
            existing = self._connection.execute(
                "SELECT resource_kind, quantity, unit FROM learning_resources "
                "WHERE transaction_id=? AND usage_id=?",
                (transaction_id, usage_id),
            ).fetchone()
            if existing is not None:
                identity = (
                    existing["resource_kind"],
                    float(existing["quantity"]),
                    existing["unit"],
                )
                if identity != (resource_kind, quantity, unit):
                    raise ValueError("resource identity was reused with different accounting")
                return
            self._connection.execute(
                "INSERT INTO learning_resources VALUES (?, ?, ?, ?, ?)",
                (transaction_id, usage_id, resource_kind, quantity, unit),
            )
            self._refresh_hash(transaction_id)

    def set_candidate(self, transaction_id: str, candidate_policy_epoch_id: str) -> None:
        with self._connection:
            row = self._connection.execute(
                "SELECT candidate_policy_epoch_id, state FROM learning_transactions "
                "WHERE transaction_id=?",
                (transaction_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"unknown learning transaction {transaction_id!r}")
            if LearningState(row["state"]) not in {
                LearningState.TRAINING,
                LearningState.CANDIDATE_READY,
            }:
                raise ValueError("candidate may only be bound during candidate publication")
            if row["candidate_policy_epoch_id"] not in {None, candidate_policy_epoch_id}:
                raise ValueError("learning transaction candidate identity is immutable")
            self._connection.execute(
                "UPDATE learning_transactions SET candidate_policy_epoch_id=? "
                "WHERE transaction_id=?",
                (candidate_policy_epoch_id, transaction_id),
            )
            self._refresh_hash(transaction_id)

    def transition(
        self,
        transaction_id: str,
        *,
        target: LearningState,
        reason: str,
        observed_at_ms: int,
        evidence_artifact_ids: tuple[str, ...] = (),
        fault_after_state_update: bool = False,
    ) -> LearningTransactionRecord:
        if not reason:
            raise ValueError("transaction transition requires a reason")
        with self._connection:
            current = self.transaction(transaction_id)
            if current.state is target:
                event = self._connection.execute(
                    "SELECT reason, evidence_artifact_ids_json, observed_at_ms "
                    "FROM learning_events WHERE transaction_id=? AND sequence=?",
                    (transaction_id, current.sequence),
                ).fetchone()
                assert event is not None
                persisted_evidence = tuple(
                    str(item) for item in json.loads(event["evidence_artifact_ids_json"])
                )
                if (
                    event["reason"] != reason
                    or persisted_evidence != tuple(sorted(evidence_artifact_ids))
                    or event["observed_at_ms"] != observed_at_ms
                ):
                    raise ValueError(
                        "transition retry does not match the atomically committed event"
                    )
                return current
            if current.state.terminal:
                raise ValueError("terminal learning transaction cannot transition")
            allowed = _ALLOWED.get(current.state, frozenset())
            if target not in allowed:
                raise ValueError(
                    f"invalid learning transition {current.state.value}->{target.value}"
                )
            known = {item.artifact_id for item in current.artifacts}
            missing = set(evidence_artifact_ids) - known
            if missing:
                raise ValueError(f"transition references unknown artifacts: {sorted(missing)}")
            sequence = current.sequence + 1
            updated_at_ms = self._connection.execute(
                "SELECT updated_at_ms FROM learning_transactions WHERE transaction_id=?",
                (transaction_id,),
            ).fetchone()["updated_at_ms"]
            if observed_at_ms < updated_at_ms:
                raise ValueError("learning transition time cannot move backwards")
            changed = self._connection.execute(
                "UPDATE learning_transactions SET state=?, sequence=?, updated_at_ms=? "
                "WHERE transaction_id=? AND state=? AND sequence=?",
                (
                    target.value,
                    sequence,
                    observed_at_ms,
                    transaction_id,
                    current.state.value,
                    current.sequence,
                ),
            ).rowcount
            if changed != 1:
                raise ValueError("learning transition lost compare-and-swap")
            if fault_after_state_update:
                raise RuntimeError("injected fault after learning state update")
            self._connection.execute(
                "INSERT INTO learning_events VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    transaction_id,
                    sequence,
                    current.state.value,
                    target.value,
                    reason,
                    json.dumps(sorted(evidence_artifact_ids), separators=(",", ":")),
                    observed_at_ms,
                ),
            )
            self._refresh_hash(transaction_id)
        return self.transaction(transaction_id)
