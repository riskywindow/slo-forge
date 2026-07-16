"""Persisted gateway acceptance ledger for migration-safe token streaming."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Literal, cast

from .coordinator import StaleOwner
from .models import TokenAcceptance, TokenEvent


class TokenGap(RuntimeError):
    """An owner attempted to commit beyond the next required token index."""


class TokenMismatch(RuntimeError):
    """A repeated token index carried different content or state evidence."""


class StateVersionRegression(RuntimeError):
    """An output event referred to an older committed state generation."""


class GatewayLedgerFull(RuntimeError):
    """The configured durable per-session token-record bound was reached."""


class ClientAcknowledgmentError(RuntimeError):
    """A client acknowledgment cursor was invalid or regressed."""


class GatewayCommitLedger:
    """Exactly-once gateway acceptance with explicit weaker network semantics."""

    def __init__(self, path: Path | str, *, max_token_records: int = 65_536) -> None:
        if not 1 <= max_token_records <= 65_536:
            raise ValueError("max_token_records must be within 1..65536")
        self.path = str(path)
        self._max_token_records = max_token_records
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
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS continuum_gateway_sessions (
              session_id TEXT PRIMARY KEY,
              owner_epoch INTEGER NOT NULL,
              next_token_index INTEGER NOT NULL,
              gateway_watermark INTEGER NOT NULL,
              client_ack_watermark INTEGER,
              last_state_commit_version INTEGER NOT NULL DEFAULT 0,
              terminal INTEGER NOT NULL DEFAULT 0,
              version INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS continuum_gateway_tokens (
              session_id TEXT NOT NULL,
              token_index INTEGER NOT NULL,
              payload TEXT NOT NULL,
              PRIMARY KEY(session_id, token_index),
              FOREIGN KEY(session_id) REFERENCES continuum_gateway_sessions(session_id)
            );
            """
        )
        columns = {
            str(row["name"])
            for row in self._connection.execute(
                "PRAGMA table_info(continuum_gateway_sessions)"
            ).fetchall()
        }
        if "last_state_commit_version" not in columns:
            self._connection.execute(
                "ALTER TABLE continuum_gateway_sessions "
                "ADD COLUMN last_state_commit_version INTEGER NOT NULL DEFAULT 0"
            )

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

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> GatewayCommitLedger:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def register(self, *, session_id: str, owner_epoch: int, next_token_index: int = 0) -> None:
        if owner_epoch < 1 or next_token_index < 0:
            raise ValueError("owner epoch and next token index are positive/non-negative")
        try:
            with self._write() as connection:
                connection.execute(
                    "INSERT INTO continuum_gateway_sessions "
                    "(session_id,owner_epoch,next_token_index,gateway_watermark,"
                    "client_ack_watermark,last_state_commit_version,terminal,version) "
                    "VALUES(?,?,?,?,?,?,?,?)",
                    (
                        session_id,
                        owner_epoch,
                        next_token_index,
                        next_token_index - 1,
                        None,
                        0,
                        0,
                        1,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"gateway session {session_id!r} already exists") from exc

    def switch_owner(
        self,
        *,
        session_id: str,
        expected_epoch: int,
        destination_epoch: int,
        expected_watermark: int,
    ) -> None:
        if destination_epoch != expected_epoch + 1:
            raise ValueError("gateway owner epochs must advance exactly once")
        with self._write() as connection:
            row = self._session(session_id)
            if row["owner_epoch"] == destination_epoch:
                if row["gateway_watermark"] != expected_watermark:
                    raise TokenMismatch("idempotent gateway switch watermark differs")
                return
            if row["owner_epoch"] != expected_epoch:
                raise StaleOwner("gateway owner epoch changed before switch")
            if row["gateway_watermark"] != expected_watermark:
                raise CoordinatorWatermarkError(
                    "gateway watermark does not match the transaction commit boundary"
                )
            changed = connection.execute(
                "UPDATE continuum_gateway_sessions SET owner_epoch=?,version=version+1 "
                "WHERE session_id=? AND owner_epoch=? AND gateway_watermark=?",
                (destination_epoch, session_id, expected_epoch, expected_watermark),
            ).rowcount
            if changed != 1:
                raise StaleOwner("gateway compare-and-swap failed")

    def _session(self, session_id: str) -> sqlite3.Row:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM continuum_gateway_sessions WHERE session_id=?", (session_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown gateway session {session_id!r}")
        return cast(sqlite3.Row, row)

    def accept(
        self,
        event: TokenEvent,
        *,
        client_acknowledged: bool = False,
    ) -> TokenAcceptance:
        with self._write() as connection:
            row = self._session(event.session_id)
            if row["owner_epoch"] != event.owner_epoch:
                raise StaleOwner(
                    f"gateway expects epoch {row['owner_epoch']}, received {event.owner_epoch}"
                )
            if event.token_index < row["next_token_index"]:
                previous = connection.execute(
                    "SELECT payload FROM continuum_gateway_tokens "
                    "WHERE session_id=? AND token_index=?",
                    (event.session_id, event.token_index),
                ).fetchone()
                if previous is None or previous["payload"] != event.model_dump_json():
                    raise TokenMismatch("duplicate token index carries different payload")
                return TokenAcceptance(
                    disposition="duplicate",
                    session_id=event.session_id,
                    owner_epoch=event.owner_epoch,
                    token_index=event.token_index,
                    gateway_commit_watermark=row["gateway_watermark"],
                    delivery_semantics="gateway_exactly_once",
                )
            if bool(row["terminal"]):
                raise TokenMismatch("terminal output was already committed")
            if event.token_index > row["next_token_index"]:
                raise TokenGap(
                    f"expected token index {row['next_token_index']}, received {event.token_index}"
                )
            if event.state_commit_version < row["last_state_commit_version"]:
                raise StateVersionRegression("token state commit version cannot move backwards")
            count = connection.execute(
                "SELECT COUNT(*) AS count FROM continuum_gateway_tokens WHERE session_id=?",
                (event.session_id,),
            ).fetchone()
            if count is None or int(count["count"]) >= self._max_token_records:
                raise GatewayLedgerFull(
                    f"gateway token ledger reached {self._max_token_records} records"
                )
            connection.execute(
                "INSERT INTO continuum_gateway_tokens VALUES(?,?,?)",
                (event.session_id, event.token_index, event.model_dump_json()),
            )
            connection.execute(
                "UPDATE continuum_gateway_sessions SET next_token_index=?,gateway_watermark=?,"
                "client_ack_watermark=?,last_state_commit_version=?,terminal=?,"
                "version=version+1 WHERE session_id=?",
                (
                    event.token_index + 1,
                    event.token_index,
                    event.token_index if client_acknowledged else row["client_ack_watermark"],
                    event.state_commit_version,
                    int(event.terminal),
                    event.session_id,
                ),
            )
        semantics: Literal[
            "client_exactly_once_acknowledged",
            "network_at_least_once_sequence_deduplicated",
        ] = (
            "client_exactly_once_acknowledged"
            if client_acknowledged
            else "network_at_least_once_sequence_deduplicated"
        )
        return TokenAcceptance(
            disposition="accepted",
            session_id=event.session_id,
            owner_epoch=event.owner_epoch,
            token_index=event.token_index,
            gateway_commit_watermark=event.token_index,
            delivery_semantics=semantics,
        )

    def acknowledge_client(self, *, session_id: str, token_index: int) -> None:
        """Persist an acknowledgment-capable client's monotonic durable cursor."""

        if token_index < 0:
            raise ClientAcknowledgmentError("client acknowledgment must be non-negative")
        with self._write() as connection:
            row = self._session(session_id)
            prior = row["client_ack_watermark"]
            if token_index > row["gateway_watermark"] or (
                prior is not None and token_index < prior
            ):
                raise ClientAcknowledgmentError(
                    "client acknowledgment must be monotonic and not exceed the gateway"
                )
            if prior == token_index:
                return
            connection.execute(
                "UPDATE continuum_gateway_sessions SET client_ack_watermark=?,"
                "version=version+1 WHERE session_id=?",
                (token_index, session_id),
            )

    def watermark(self, session_id: str) -> int:
        with self._lock:
            return int(self._session(session_id)["gateway_watermark"])

    def accepted_tokens(self, session_id: str) -> tuple[TokenEvent, ...]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT payload FROM continuum_gateway_tokens "
                "WHERE session_id=? ORDER BY token_index",
                (session_id,),
            ).fetchall()
        return tuple(TokenEvent.model_validate_json(row["payload"], strict=True) for row in rows)


class CoordinatorWatermarkError(RuntimeError):
    """Coordinator and gateway disagree on the declared cutover boundary."""
