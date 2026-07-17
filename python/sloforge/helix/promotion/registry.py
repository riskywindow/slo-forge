"""SQLite-backed champion routing, promotion gates, active sessions, and rollback."""

from __future__ import annotations

import json
import math
import re
import sqlite3
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sloforge.helix.policy import DeterministicPolicy


class _PromotionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class CompatibilityClass(StrEnum):
    DIRECT = "direct"
    RECOMPUTE = "recompute"
    REQUEST_BOUNDARY = "request_boundary"
    CHAMPION_PINNED = "champion_pinned"
    INCOMPATIBLE = "incompatible"


class PromotionState(StrEnum):
    INTENT_RECORDED = "intent_recorded"
    SHADOWING = "shadowing"
    SHADOW_PASSED = "shadow_passed"
    CANARYING = "canarying"
    CANARY_PASSED = "canary_passed"
    ACTIVE = "active"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"

    @property
    def terminal(self) -> bool:
        return self in {self.REJECTED, self.ROLLED_BACK}


class GateEvidence(_PromotionModel):
    tenant_id: Annotated[str, Field(min_length=1, max_length=160)] = "default"
    gate: Literal[
        "lineage",
        "reward_integrity",
        "quality",
        "safety",
        "serving",
        "compatibility",
        "shadow",
        "canary",
    ]
    evidence_id: Annotated[str, Field(min_length=1, max_length=160)]
    artifact_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    passed: bool
    sample_count: Annotated[int, Field(ge=0, le=10_000_000)]
    measured_value: float
    threshold: float
    comparator: Literal["le", "ge", "eq"]
    deterministic_seed: Annotated[int, Field(ge=0, le=2**64 - 1)]
    detail: Annotated[str, Field(min_length=1, max_length=2048)]

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if not math.isfinite(self.measured_value) or not math.isfinite(self.threshold):
            raise ValueError("promotion gate measurements must be finite")
        if self.passed and self.sample_count == 0:
            raise ValueError("a passing promotion gate must evaluate at least one sample")
        evaluated = {
            "le": self.measured_value <= self.threshold,
            "ge": self.measured_value >= self.threshold,
            "eq": self.measured_value == self.threshold,
        }[self.comparator]
        if evaluated != self.passed:
            raise ValueError("gate pass flag disagrees with measured value and threshold")
        return self


class SessionRoute(_PromotionModel):
    tenant_id: Annotated[str, Field(min_length=1, max_length=160)] = "default"
    session_id: Annotated[str, Field(min_length=1, max_length=160)]
    deployment: Annotated[str, Field(min_length=1, max_length=160)]
    policy_epoch_id: Annotated[str, Field(min_length=1, max_length=160)]
    compatibility: CompatibilityClass
    pinned: bool
    active: bool


class PromotionRecord(_PromotionModel):
    tenant_id: str = "default"
    transaction_id: str
    deployment: str
    champion_policy_epoch_id: str
    candidate_policy_epoch_id: str
    state: PromotionState
    evidence_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    rollback_parent: str
    sequence: Annotated[int, Field(ge=0)]
    reason: str


_REQUIRED_PRE_PROMOTION_GATES = {
    "lineage",
    "reward_integrity",
    "quality",
    "safety",
    "serving",
    "compatibility",
}
_TENANT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$")


def _canonical_evidence(evidence: tuple[GateEvidence, ...]) -> str:
    payload = json.dumps(
        [item.model_dump(mode="json") for item in sorted(evidence, key=lambda item: item.gate)],
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(payload.encode("utf-8")).hexdigest()


class PolicyRegistry:
    """Local durable registry; distributed use requires an external linearizable CAS."""

    def __init__(self, database: str | Path, *, tenant_id: str = "default") -> None:
        # ``IMMEDIATE`` makes each connection context below a real transaction and
        # obtains the write reservation before mutating the champion pointer.  In
        # particular, an exception after the pointer update must roll the update
        # back rather than expose a partially promoted deployment.
        if not _TENANT.fullmatch(tenant_id):
            raise ValueError("promotion registry tenant identifier is invalid")
        self.tenant_id = tenant_id
        self._connection = sqlite3.connect(str(database), timeout=5.0, isolation_level="IMMEDIATE")
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS registry_metadata (
                singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                tenant_id TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS policies (
                policy_epoch_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                parent_policy_epoch_id TEXT,
                policy_json TEXT NOT NULL,
                policy_hash TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('champion','challenger','retained','rejected')),
                created_at_ms INTEGER NOT NULL CHECK(created_at_ms >= 0)
            );
            CREATE TABLE IF NOT EXISTS deployments (
                deployment TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                champion_policy_epoch_id TEXT NOT NULL REFERENCES policies(policy_epoch_id),
                generation INTEGER NOT NULL CHECK(generation >= 1)
            );
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                deployment TEXT NOT NULL REFERENCES deployments(deployment),
                policy_epoch_id TEXT NOT NULL REFERENCES policies(policy_epoch_id),
                compatibility TEXT NOT NULL,
                pinned INTEGER NOT NULL CHECK(pinned IN (0,1)),
                active INTEGER NOT NULL CHECK(active IN (0,1)),
                opened_at_ms INTEGER NOT NULL CHECK(opened_at_ms >= 0)
            );
            CREATE TABLE IF NOT EXISTS promotions (
                transaction_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                deployment TEXT NOT NULL REFERENCES deployments(deployment),
                champion_policy_epoch_id TEXT NOT NULL REFERENCES policies(policy_epoch_id),
                candidate_policy_epoch_id TEXT NOT NULL REFERENCES policies(policy_epoch_id),
                state TEXT NOT NULL,
                evidence_json TEXT NOT NULL,
                evidence_hash TEXT NOT NULL,
                rollback_parent TEXT NOT NULL REFERENCES policies(policy_epoch_id),
                sequence INTEGER NOT NULL CHECK(sequence >= 0),
                reason TEXT NOT NULL,
                updated_at_ms INTEGER NOT NULL CHECK(updated_at_ms >= 0)
            );
            CREATE TABLE IF NOT EXISTS promotion_events (
                transaction_id TEXT NOT NULL REFERENCES promotions(transaction_id),
                sequence INTEGER NOT NULL CHECK(sequence >= 0),
                state_before TEXT NOT NULL,
                state_after TEXT NOT NULL,
                reason TEXT NOT NULL,
                observed_at_ms INTEGER NOT NULL CHECK(observed_at_ms >= 0),
                PRIMARY KEY(transaction_id, sequence)
            );
            """
        )
        with self._connection:
            metadata = self._connection.execute(
                "SELECT tenant_id FROM registry_metadata WHERE singleton=1"
            ).fetchone()
            if metadata is None:
                self._connection.execute(
                    "INSERT INTO registry_metadata(singleton, tenant_id) VALUES (1, ?)",
                    (tenant_id,),
                )
            elif metadata["tenant_id"] != tenant_id:
                raise PermissionError("promotion registry is already bound to a different tenant")
            for table in ("policies", "deployments", "sessions", "promotions"):
                columns = {
                    row["name"]
                    for row in self._connection.execute(f"PRAGMA table_info({table})").fetchall()
                }
                if "tenant_id" not in columns:
                    self._connection.execute(f"ALTER TABLE {table} ADD COLUMN tenant_id TEXT")
                self._connection.execute(
                    f"UPDATE {table} SET tenant_id=? WHERE tenant_id IS NULL",
                    (tenant_id,),
                )

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    @staticmethod
    def _policy_hash(policy: DeterministicPolicy) -> str:
        return sha256(policy.model_dump_json().encode("utf-8")).hexdigest()

    def register_policy(
        self,
        policy: DeterministicPolicy,
        *,
        parent_policy_epoch_id: str | None,
        status: Literal["champion", "challenger"],
        created_at_ms: int,
    ) -> None:
        if created_at_ms < 0:
            raise ValueError("policy creation time must be non-negative")
        payload = policy.model_dump_json()
        digest = self._policy_hash(policy)
        try:
            with self._connection:
                if parent_policy_epoch_id is not None:
                    parent = self._connection.execute(
                        "SELECT 1 FROM policies WHERE policy_epoch_id=? AND tenant_id=?",
                        (parent_policy_epoch_id, self.tenant_id),
                    ).fetchone()
                    if parent is None:
                        raise ValueError("parent policy is absent from the registry tenant")
                self._connection.execute(
                    "INSERT INTO policies(policy_epoch_id, tenant_id, parent_policy_epoch_id, "
                    "policy_json, policy_hash, status, created_at_ms) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        policy.policy_epoch_id,
                        self.tenant_id,
                        parent_policy_epoch_id,
                        payload,
                        digest,
                        status,
                        created_at_ms,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            existing = self._connection.execute(
                "SELECT tenant_id, parent_policy_epoch_id, policy_hash, status, created_at_ms "
                "FROM policies WHERE policy_epoch_id=?",
                (policy.policy_epoch_id,),
            ).fetchone()
            if (
                existing is None
                or existing["tenant_id"] != self.tenant_id
                or existing["parent_policy_epoch_id"] != parent_policy_epoch_id
                or existing["policy_hash"] != digest
                or existing["status"] != status
                or existing["created_at_ms"] != created_at_ms
            ):
                raise ValueError("policy epoch identity was reused with changed weights") from exc

    def create_deployment(self, deployment: str, champion_policy_epoch_id: str) -> None:
        with self._connection:
            status = self._connection.execute(
                "SELECT status FROM policies WHERE policy_epoch_id=? AND tenant_id=?",
                (champion_policy_epoch_id, self.tenant_id),
            ).fetchone()
            if status is None or status["status"] != "champion":
                raise ValueError("deployment champion must be a registered champion policy")
            self._connection.execute(
                "INSERT INTO deployments(deployment, tenant_id, champion_policy_epoch_id, "
                "generation) VALUES (?, ?, ?, 1)",
                (deployment, self.tenant_id, champion_policy_epoch_id),
            )

    def champion(self, deployment: str) -> DeterministicPolicy:
        row = self._connection.execute(
            """
            SELECT p.policy_json, p.policy_hash
            FROM deployments d JOIN policies p
            ON d.champion_policy_epoch_id=p.policy_epoch_id
            WHERE d.deployment=? AND d.tenant_id=? AND p.tenant_id=?
            """,
            (deployment, self.tenant_id, self.tenant_id),
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown deployment {deployment!r}")
        policy = DeterministicPolicy.model_validate_json(row["policy_json"], strict=True)
        if self._policy_hash(policy) != row["policy_hash"]:
            raise ValueError("stored champion policy failed integrity validation")
        return policy

    def open_session(self, *, deployment: str, session_id: str, opened_at_ms: int) -> SessionRoute:
        try:
            with self._connection:
                # A no-op write obtains the IMMEDIATE reservation before reading the
                # champion, closing the read-old/insert-after-promotion race.
                claimed = self._connection.execute(
                    "UPDATE deployments SET generation=generation "
                    "WHERE deployment=? AND tenant_id=?",
                    (deployment, self.tenant_id),
                ).rowcount
                if claimed != 1:
                    raise ValueError(f"unknown deployment {deployment!r}")
                champion = self.champion(deployment)
                self._connection.execute(
                    "INSERT INTO sessions(session_id, tenant_id, deployment, policy_epoch_id, "
                    "compatibility, pinned, active, opened_at_ms) VALUES (?, ?, ?, ?, ?, 0, 1, ?)",
                    (
                        session_id,
                        self.tenant_id,
                        deployment,
                        champion.policy_epoch_id,
                        CompatibilityClass.REQUEST_BOUNDARY.value,
                        opened_at_ms,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError("session identity already exists") from exc
        return self.session(session_id)

    def session(self, session_id: str) -> SessionRoute:
        row = self._connection.execute(
            "SELECT * FROM sessions WHERE session_id=? AND tenant_id=?",
            (session_id, self.tenant_id),
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown session {session_id!r}")
        return SessionRoute(
            tenant_id=row["tenant_id"],
            session_id=row["session_id"],
            deployment=row["deployment"],
            policy_epoch_id=row["policy_epoch_id"],
            compatibility=CompatibilityClass(row["compatibility"]),
            pinned=bool(row["pinned"]),
            active=bool(row["active"]),
        )

    def classify_session(
        self,
        session_id: str,
        compatibility: CompatibilityClass,
    ) -> SessionRoute:
        pinned = compatibility in {
            CompatibilityClass.CHAMPION_PINNED,
            CompatibilityClass.INCOMPATIBLE,
            CompatibilityClass.REQUEST_BOUNDARY,
        }
        with self._connection:
            changed = self._connection.execute(
                "UPDATE sessions SET compatibility=?, pinned=? "
                "WHERE session_id=? AND tenant_id=? AND active=1",
                (compatibility.value, int(pinned), session_id, self.tenant_id),
            ).rowcount
            if changed != 1:
                raise ValueError(f"unknown active session {session_id!r}")
        return self.session(session_id)

    def create_promotion(
        self,
        *,
        transaction_id: str,
        deployment: str,
        candidate_policy_epoch_id: str,
        evidence: tuple[GateEvidence, ...],
        observed_at_ms: int,
    ) -> PromotionRecord:
        gates = [item.gate for item in evidence]
        if len(gates) != len(set(gates)) or set(gates) != _REQUIRED_PRE_PROMOTION_GATES:
            raise ValueError("promotion intent requires exactly the six pre-promotion gates")
        if any(item.tenant_id != self.tenant_id for item in evidence):
            raise PermissionError("promotion evidence cannot cross tenant boundaries")
        evidence_hash = _canonical_evidence(evidence)
        evidence_json = json.dumps(
            [item.model_dump(mode="json") for item in evidence],
            sort_keys=True,
            separators=(",", ":"),
        )
        passed = all(item.passed for item in evidence if item.gate in _REQUIRED_PRE_PROMOTION_GATES)
        state = PromotionState.INTENT_RECORDED if passed else PromotionState.REJECTED
        reason = "independent pre-promotion gates passed" if passed else "pre-promotion gate failed"
        with self._connection:
            claimed = self._connection.execute(
                "UPDATE deployments SET generation=generation WHERE deployment=? AND tenant_id=?",
                (deployment, self.tenant_id),
            ).rowcount
            if claimed != 1:
                raise ValueError(f"unknown deployment {deployment!r}")
            existing = self._connection.execute(
                "SELECT * FROM promotions WHERE transaction_id=? AND tenant_id=?",
                (transaction_id, self.tenant_id),
            ).fetchone()
            if existing is not None:
                persisted_evidence = tuple(
                    GateEvidence.model_validate(item, strict=True)
                    for item in json.loads(existing["evidence_json"])
                    if item["gate"] in _REQUIRED_PRE_PROMOTION_GATES
                )
                if (
                    existing["deployment"] != deployment
                    or existing["candidate_policy_epoch_id"] != candidate_policy_epoch_id
                    or _canonical_evidence(persisted_evidence) != evidence_hash
                ):
                    raise ValueError(
                        "promotion transaction identity was reused with different inputs"
                    )
                return self.promotion(transaction_id)
            champion = self.champion(deployment)
            candidate = self._connection.execute(
                "SELECT status FROM policies WHERE policy_epoch_id=? AND tenant_id=?",
                (candidate_policy_epoch_id, self.tenant_id),
            ).fetchone()
            if candidate is None or candidate["status"] != "challenger":
                raise ValueError("candidate is not a registered challenger")
            self._connection.execute(
                "INSERT INTO promotions(transaction_id, tenant_id, deployment, "
                "champion_policy_epoch_id, candidate_policy_epoch_id, state, evidence_json, "
                "evidence_hash, rollback_parent, sequence, reason, updated_at_ms) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)",
                (
                    transaction_id,
                    self.tenant_id,
                    deployment,
                    champion.policy_epoch_id,
                    candidate_policy_epoch_id,
                    state.value,
                    evidence_json,
                    evidence_hash,
                    champion.policy_epoch_id,
                    reason,
                    observed_at_ms,
                ),
            )
            self._connection.execute(
                "INSERT INTO promotion_events VALUES (?, 0, ?, ?, ?, ?)",
                (
                    transaction_id,
                    state.value,
                    state.value,
                    reason,
                    observed_at_ms,
                ),
            )
        return self.promotion(transaction_id)

    def promotion(self, transaction_id: str) -> PromotionRecord:
        row = self._connection.execute(
            "SELECT * FROM promotions WHERE transaction_id=? AND tenant_id=?",
            (transaction_id, self.tenant_id),
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown promotion transaction {transaction_id!r}")
        evidence = tuple(
            GateEvidence.model_validate(item, strict=True)
            for item in json.loads(row["evidence_json"])
        )
        if _canonical_evidence(evidence) != row["evidence_hash"]:
            raise ValueError("promotion evidence failed integrity validation")
        return PromotionRecord(
            tenant_id=row["tenant_id"],
            transaction_id=row["transaction_id"],
            deployment=row["deployment"],
            champion_policy_epoch_id=row["champion_policy_epoch_id"],
            candidate_policy_epoch_id=row["candidate_policy_epoch_id"],
            state=PromotionState(row["state"]),
            evidence_hash=row["evidence_hash"],
            rollback_parent=row["rollback_parent"],
            sequence=row["sequence"],
            reason=row["reason"],
        )

    def _transition(
        self,
        transaction_id: str,
        *,
        expected: PromotionState,
        target: PromotionState,
        reason: str,
        observed_at_ms: int,
    ) -> PromotionRecord:
        current = self.promotion(transaction_id)
        if current.state is target:
            return current
        if current.state is not expected:
            raise ValueError(
                f"promotion transition requires {expected.value}, found {current.state.value}"
            )
        sequence = current.sequence + 1
        with self._connection:
            changed = self._connection.execute(
                """
                UPDATE promotions SET state=?, sequence=?, reason=?, updated_at_ms=?
                WHERE transaction_id=? AND tenant_id=? AND state=? AND sequence=?
                """,
                (
                    target.value,
                    sequence,
                    reason,
                    observed_at_ms,
                    transaction_id,
                    self.tenant_id,
                    expected.value,
                    current.sequence,
                ),
            ).rowcount
            if changed != 1:
                raise ValueError("concurrent promotion transition lost compare-and-swap")
            self._connection.execute(
                "INSERT INTO promotion_events VALUES (?, ?, ?, ?, ?, ?)",
                (
                    transaction_id,
                    sequence,
                    expected.value,
                    target.value,
                    reason,
                    observed_at_ms,
                ),
            )
        return self.promotion(transaction_id)

    def start_shadow(self, transaction_id: str, *, observed_at_ms: int) -> PromotionRecord:
        return self._transition(
            transaction_id,
            expected=PromotionState.INTENT_RECORDED,
            target=PromotionState.SHADOWING,
            reason="isolated shadow started; real side effects disabled",
            observed_at_ms=observed_at_ms,
        )

    def _finish_runtime_gate(
        self,
        transaction_id: str,
        evidence: GateEvidence,
        *,
        gate: Literal["shadow", "canary"],
        expected: PromotionState,
        passed_target: PromotionState,
        observed_at_ms: int,
    ) -> PromotionRecord:
        if evidence.gate != gate:
            raise ValueError(f"{gate} transition requires {gate} evidence")
        if evidence.tenant_id != self.tenant_id:
            raise PermissionError("promotion evidence cannot cross tenant boundaries")
        target = passed_target if evidence.passed else PromotionState.REJECTED
        current = self.promotion(transaction_id)
        row = self._connection.execute(
            "SELECT evidence_json, evidence_hash FROM promotions "
            "WHERE transaction_id=? AND tenant_id=?",
            (transaction_id, self.tenant_id),
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown promotion transaction {transaction_id!r}")
        persisted = tuple(
            GateEvidence.model_validate(item, strict=True)
            for item in json.loads(row["evidence_json"])
        )
        if _canonical_evidence(persisted) != row["evidence_hash"]:
            raise ValueError("promotion evidence failed integrity validation")
        existing = next((item for item in persisted if item.gate == gate), None)
        if current.state is target:
            if existing != evidence:
                raise ValueError(f"{gate} retry does not match committed evidence")
            return current
        if current.state is not expected:
            raise ValueError(
                f"promotion transition requires {expected.value}, found {current.state.value}"
            )
        if existing is not None:
            raise ValueError(f"{gate} evidence was already recorded")
        updated_evidence = (*persisted, evidence)
        evidence_json = json.dumps(
            [item.model_dump(mode="json") for item in updated_evidence],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        evidence_hash = _canonical_evidence(updated_evidence)
        sequence = current.sequence + 1
        reason = (
            f"{gate} gate passed: {evidence.evidence_id}"
            if evidence.passed
            else f"{gate} gate rejected candidate: {evidence.evidence_id}"
        )
        with self._connection:
            changed = self._connection.execute(
                """
                UPDATE promotions SET state=?, evidence_json=?, evidence_hash=?, sequence=?,
                    reason=?, updated_at_ms=?
                WHERE transaction_id=? AND tenant_id=? AND state=? AND sequence=?
                """,
                (
                    target.value,
                    evidence_json,
                    evidence_hash,
                    sequence,
                    reason,
                    observed_at_ms,
                    transaction_id,
                    self.tenant_id,
                    expected.value,
                    current.sequence,
                ),
            ).rowcount
            if changed != 1:
                raise ValueError("concurrent promotion gate update lost compare-and-swap")
            self._connection.execute(
                "INSERT INTO promotion_events VALUES (?, ?, ?, ?, ?, ?)",
                (
                    transaction_id,
                    sequence,
                    expected.value,
                    target.value,
                    reason,
                    observed_at_ms,
                ),
            )
        return self.promotion(transaction_id)

    def finish_shadow(
        self, transaction_id: str, evidence: GateEvidence, *, observed_at_ms: int
    ) -> PromotionRecord:
        return self._finish_runtime_gate(
            transaction_id,
            evidence,
            gate="shadow",
            expected=PromotionState.SHADOWING,
            passed_target=PromotionState.SHADOW_PASSED,
            observed_at_ms=observed_at_ms,
        )

    def start_canary(self, transaction_id: str, *, observed_at_ms: int) -> PromotionRecord:
        return self._transition(
            transaction_id,
            expected=PromotionState.SHADOW_PASSED,
            target=PromotionState.CANARYING,
            reason="bounded eligible canary started",
            observed_at_ms=observed_at_ms,
        )

    def finish_canary(
        self, transaction_id: str, evidence: GateEvidence, *, observed_at_ms: int
    ) -> PromotionRecord:
        return self._finish_runtime_gate(
            transaction_id,
            evidence,
            gate="canary",
            expected=PromotionState.CANARYING,
            passed_target=PromotionState.CANARY_PASSED,
            observed_at_ms=observed_at_ms,
        )

    def promote(
        self,
        transaction_id: str,
        *,
        observed_at_ms: int,
        fault_after_pointer_update: bool = False,
    ) -> PromotionRecord:
        current = self.promotion(transaction_id)
        if current.state is PromotionState.ACTIVE:
            return current
        if current.state is not PromotionState.CANARY_PASSED:
            raise ValueError("promotion requires a passed canary")
        with self._connection:
            deployment = self._connection.execute(
                "SELECT champion_policy_epoch_id, generation FROM deployments "
                "WHERE deployment=? AND tenant_id=?",
                (current.deployment, self.tenant_id),
            ).fetchone()
            if (
                deployment is None
                or deployment["champion_policy_epoch_id"] != current.rollback_parent
            ):
                raise ValueError("champion pointer changed since promotion intent")
            changed = self._connection.execute(
                """
                UPDATE deployments
                SET champion_policy_epoch_id=?, generation=generation+1
                WHERE deployment=? AND tenant_id=? AND champion_policy_epoch_id=?
                """,
                (
                    current.candidate_policy_epoch_id,
                    current.deployment,
                    self.tenant_id,
                    current.rollback_parent,
                ),
            ).rowcount
            if changed != 1:
                raise ValueError("atomic champion pointer update failed")
            if fault_after_pointer_update:
                raise RuntimeError("injected fault after champion pointer update")
            self._connection.execute(
                "UPDATE policies SET status='retained' WHERE policy_epoch_id=? AND tenant_id=? "
                "AND NOT EXISTS (SELECT 1 FROM deployments WHERE tenant_id=? "
                "AND champion_policy_epoch_id=?)",
                (
                    current.rollback_parent,
                    self.tenant_id,
                    self.tenant_id,
                    current.rollback_parent,
                ),
            )
            self._connection.execute(
                "UPDATE policies SET status='champion' WHERE policy_epoch_id=? AND tenant_id=?",
                (current.candidate_policy_epoch_id, self.tenant_id),
            )
            sequence = current.sequence + 1
            promotion_changed = self._connection.execute(
                """
                UPDATE promotions SET state=?, sequence=?, reason=?, updated_at_ms=?
                WHERE transaction_id=? AND tenant_id=? AND state=? AND sequence=?
                """,
                (
                    PromotionState.ACTIVE.value,
                    sequence,
                    "atomic champion pointer updated; pinned sessions preserved",
                    observed_at_ms,
                    transaction_id,
                    self.tenant_id,
                    PromotionState.CANARY_PASSED.value,
                    current.sequence,
                ),
            ).rowcount
            if promotion_changed != 1:
                raise ValueError("promotion record update lost compare-and-swap")
            self._connection.execute(
                "INSERT INTO promotion_events VALUES (?, ?, ?, ?, ?, ?)",
                (
                    transaction_id,
                    sequence,
                    PromotionState.CANARY_PASSED.value,
                    PromotionState.ACTIVE.value,
                    "atomic champion pointer updated; pinned sessions preserved",
                    observed_at_ms,
                ),
            )
        return self.promotion(transaction_id)

    def rollback(self, transaction_id: str, *, reason: str, observed_at_ms: int) -> PromotionRecord:
        current = self.promotion(transaction_id)
        if current.state is PromotionState.ROLLED_BACK:
            return current
        if current.state is not PromotionState.ACTIVE:
            raise ValueError("rollback requires an active promoted candidate")
        with self._connection:
            changed = self._connection.execute(
                """
                UPDATE deployments SET champion_policy_epoch_id=?, generation=generation+1
                WHERE deployment=? AND tenant_id=? AND champion_policy_epoch_id=?
                """,
                (
                    current.rollback_parent,
                    current.deployment,
                    self.tenant_id,
                    current.candidate_policy_epoch_id,
                ),
            ).rowcount
            if changed != 1:
                raise ValueError("rollback champion compare-and-swap failed")
            self._connection.execute(
                "UPDATE policies SET status='champion' WHERE policy_epoch_id=? AND tenant_id=?",
                (current.rollback_parent, self.tenant_id),
            )
            self._connection.execute(
                "UPDATE policies SET status='retained' WHERE policy_epoch_id=? AND tenant_id=? "
                "AND NOT EXISTS (SELECT 1 FROM deployments WHERE tenant_id=? "
                "AND champion_policy_epoch_id=?)",
                (
                    current.candidate_policy_epoch_id,
                    self.tenant_id,
                    self.tenant_id,
                    current.candidate_policy_epoch_id,
                ),
            )
            sequence = current.sequence + 1
            rollback_changed = self._connection.execute(
                """
                UPDATE promotions SET state=?, sequence=?, reason=?, updated_at_ms=?
                WHERE transaction_id=? AND tenant_id=? AND state=? AND sequence=?
                """,
                (
                    PromotionState.ROLLED_BACK.value,
                    sequence,
                    reason,
                    observed_at_ms,
                    transaction_id,
                    self.tenant_id,
                    PromotionState.ACTIVE.value,
                    current.sequence,
                ),
            ).rowcount
            if rollback_changed != 1:
                raise ValueError("rollback record update lost compare-and-swap")
            self._connection.execute(
                "INSERT INTO promotion_events VALUES (?, ?, ?, ?, ?, ?)",
                (
                    transaction_id,
                    sequence,
                    PromotionState.ACTIVE.value,
                    PromotionState.ROLLED_BACK.value,
                    reason,
                    observed_at_ms,
                ),
            )
        return self.promotion(transaction_id)
