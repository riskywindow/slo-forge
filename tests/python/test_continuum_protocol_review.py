from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from sloforge.continuum.transaction import (
    ClientAcknowledgmentError,
    CoordinatorConflict,
    CutoverPhase,
    DurableCoordinator,
    GatewayCommitLedger,
    GatewayLedgerFull,
    InvalidTransition,
    StateVersionRegression,
    TokenEvent,
)

PLAN_HASH = hashlib.sha256(b"protocol-review-plan").hexdigest()
PAYLOAD_HASH = hashlib.sha256(b"protocol-review-event").hexdigest()


def _token(
    index: int,
    *,
    epoch: int = 1,
    state_version: int | None = None,
    terminal: bool = False,
) -> TokenEvent:
    return TokenEvent(
        session_id="review-session",
        owner_epoch=epoch,
        token_index=index,
        token_id=100 + index,
        state_commit_version=index + 1 if state_version is None else state_version,
        terminal=terminal,
    )


def _transaction(coordinator: DurableCoordinator, *, timeout_ms: int = 100) -> str:
    coordinator.create_lease(
        session_id="review-session",
        owner_runtime="source",
        expiration_ms=1_000,
    )
    return coordinator.begin_transaction(
        session_id="review-session",
        destination_candidate="destination",
        migration_plan_hash=PLAN_HASH,
        seed=71,
        now_ms=0,
        timeout_ms=timeout_ms,
    ).transaction_id


def _advance_to_commit_intent(coordinator: DurableCoordinator, identifier: str) -> None:
    current = CutoverPhase.PROPOSED
    phases = (
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
    )
    for tick, target in enumerate(phases, start=1):
        coordinator.transition(
            identifier,
            expected=current,
            target=target,
            event_id=f"review-{tick}",
            at_ms=tick,
            payload_hash=PAYLOAD_HASH,
            commit_watermark=0 if target is CutoverPhase.COMMIT_INTENT_RECORDED else None,
            rollback_watermark=0 if target is CutoverPhase.COMMIT_INTENT_RECORDED else None,
            state_hashes=(PAYLOAD_HASH,) if target is CutoverPhase.COMMIT_INTENT_RECORDED else None,
        )
        current = target


def test_postcommit_failure_cannot_enter_precommit_rollback_window() -> None:
    with DurableCoordinator(":memory:") as coordinator:
        identifier = _transaction(coordinator)
        _advance_to_commit_intent(coordinator, identifier)
        coordinator.commit_ownership(
            identifier,
            event_id="review-owner-commit",
            at_ms=12,
            state_version=4,
        )
        coordinator.transition(
            identifier,
            expected=CutoverPhase.OWNERSHIP_COMMITTED,
            target=CutoverPhase.DESTINATION_LOST,
            event_id="review-destination-lost",
            at_ms=13,
            payload_hash=PAYLOAD_HASH,
        )
        with pytest.raises(InvalidTransition):
            coordinator.transition(
                identifier,
                expected=CutoverPhase.DESTINATION_LOST,
                target=CutoverPhase.ABORTING,
                event_id="review-unsafe-abort",
                at_ms=14,
                payload_hash=PAYLOAD_HASH,
            )
        failed = coordinator.transition(
            identifier,
            expected=CutoverPhase.DESTINATION_LOST,
            target=CutoverPhase.FAILED_AFTER_COMMIT,
            event_id="review-postcommit-failure",
            at_ms=14,
            payload_hash=PAYLOAD_HASH,
            failure_reason="destination lost after ownership commit",
        )
        assert failed.phase is CutoverPhase.FAILED_AFTER_COMMIT
        assert coordinator.lease("review-session").owner_runtime == "destination"


def test_transaction_deadline_is_exclusive() -> None:
    with DurableCoordinator(":memory:") as coordinator:
        identifier = _transaction(coordinator, timeout_ms=10)
        with pytest.raises(CoordinatorConflict, match="deadline"):
            coordinator.transition(
                identifier,
                expected=CutoverPhase.PROPOSED,
                target=CutoverPhase.COMPATIBILITY_VALIDATED,
                event_id="at-exclusive-deadline",
                at_ms=10,
                payload_hash=PAYLOAD_HASH,
            )


def test_terminal_event_exact_replay_is_idempotent() -> None:
    with GatewayCommitLedger(":memory:") as ledger:
        ledger.register(session_id="review-session", owner_epoch=1)
        terminal = _token(0, terminal=True)
        assert ledger.accept(terminal).disposition == "accepted"
        assert ledger.accept(terminal).disposition == "duplicate"


def test_gateway_rejects_state_version_regression_and_invalid_client_cursor() -> None:
    with GatewayCommitLedger(":memory:") as ledger:
        ledger.register(session_id="review-session", owner_epoch=1)
        ledger.accept(_token(0, state_version=5))
        with pytest.raises(StateVersionRegression):
            ledger.accept(_token(1, state_version=4))
        ledger.accept(_token(1, state_version=5))
        ledger.acknowledge_client(session_id="review-session", token_index=1)
        with pytest.raises(ClientAcknowledgmentError):
            ledger.acknowledge_client(session_id="review-session", token_index=0)
        with pytest.raises(ClientAcknowledgmentError):
            ledger.acknowledge_client(session_id="review-session", token_index=2)


def test_gateway_storage_bound_fails_closed() -> None:
    with GatewayCommitLedger(":memory:", max_token_records=2) as ledger:
        ledger.register(session_id="review-session", owner_epoch=1)
        ledger.accept(_token(0))
        ledger.accept(_token(1))
        with pytest.raises(GatewayLedgerFull):
            ledger.accept(_token(2))
        assert ledger.watermark("review-session") == 1


def test_concurrent_gateway_replays_linearize_across_connections(tmp_path: Path) -> None:
    path = tmp_path / "gateway.db"
    first = GatewayCommitLedger(path)
    second = GatewayCommitLedger(path)
    try:
        first.register(session_id="review-session", owner_epoch=1)
        event = _token(0)
        with ThreadPoolExecutor(max_workers=2) as pool:
            dispositions = tuple(
                pool.map(lambda ledger: ledger.accept(event).disposition, (first, second))
            )
        assert sorted(dispositions) == ["accepted", "duplicate"]
        assert first.watermark("review-session") == 0
    finally:
        first.close()
        second.close()


def test_concurrent_coordinator_replays_linearize_across_connections(tmp_path: Path) -> None:
    path = tmp_path / "coordinator.db"
    first = DurableCoordinator(path)
    second = DurableCoordinator(path)
    try:
        identifier = _transaction(first)

        def transition(coordinator: DurableCoordinator) -> CutoverPhase:
            return coordinator.transition(
                identifier,
                expected=CutoverPhase.PROPOSED,
                target=CutoverPhase.COMPATIBILITY_VALIDATED,
                event_id="concurrent-exact-replay",
                at_ms=1,
                payload_hash=PAYLOAD_HASH,
            ).phase

        with ThreadPoolExecutor(max_workers=2) as pool:
            phases = tuple(pool.map(transition, (first, second)))
        assert phases == (
            CutoverPhase.COMPATIBILITY_VALIDATED,
            CutoverPhase.COMPATIBILITY_VALIDATED,
        )
        assert len(first.journal(identifier)) == 2
    finally:
        first.close()
        second.close()
