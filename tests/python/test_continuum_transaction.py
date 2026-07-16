from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from sloforge.continuum.transaction import (
    CoordinatorConflict,
    CutoverPhase,
    DurableCoordinator,
    GatewayCommitLedger,
    InvalidTransition,
    StaleOwner,
    TokenEvent,
    TokenGap,
    TokenMismatch,
)

SHA = hashlib.sha256(b"continuum-plan").hexdigest()
EVENT_SHA = hashlib.sha256(b"event").hexdigest()


def _advance_to_commit(
    coordinator: DurableCoordinator, identifier: str, *, commit_watermark: int | None = None
) -> None:
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
    current = CutoverPhase.PROPOSED
    for index, target in enumerate(phases, start=1):
        coordinator.transition(
            identifier,
            expected=current,
            target=target,
            event_id=f"transition-{index}",
            at_ms=index,
            payload_hash=EVENT_SHA,
            commit_watermark=(
                commit_watermark if target is CutoverPhase.COMMIT_INTENT_RECORDED else None
            ),
        )
        current = target


def test_durable_cas_commit_fences_source_and_survives_restart(tmp_path: Path) -> None:
    path = tmp_path / "coordinator.db"
    with DurableCoordinator(path) as coordinator:
        lease = coordinator.create_lease(
            session_id="session-001", owner_runtime="runtime-a", expiration_ms=10_000
        )
        transaction = coordinator.begin_transaction(
            session_id=lease.session_id,
            destination_candidate="runtime-b",
            migration_plan_hash=SHA,
            seed=17,
            now_ms=0,
            timeout_ms=1_000,
        )
        _advance_to_commit(coordinator, transaction.transaction_id, commit_watermark=11)
        record, committed = coordinator.commit_ownership(
            transaction.transaction_id,
            event_id="ownership-commit",
            at_ms=20,
            state_version=7,
        )
        assert record.phase is CutoverPhase.OWNERSHIP_COMMITTED
        assert committed.owner_runtime == "runtime-b"
        assert committed.owner_epoch == 2
        assert committed.fencing_token == 2
        assert committed.last_committed_token_index == 11
        replay_record, replay_lease = coordinator.commit_ownership(
            transaction.transaction_id,
            event_id="ownership-commit",
            at_ms=20,
            state_version=7,
        )
        assert replay_record == record
        assert replay_lease == committed
        with pytest.raises(CoordinatorConflict, match="does not match"):
            coordinator.commit_ownership(
                transaction.transaction_id,
                event_id="different-ownership-event",
                at_ms=20,
                state_version=7,
            )
        with pytest.raises(StaleOwner):
            coordinator.assert_owner(
                session_id="session-001",
                owner_runtime="runtime-a",
                owner_epoch=1,
                fencing_token=1,
                now_ms=20,
            )

    with DurableCoordinator(path) as reopened:
        persisted = reopened.transaction(transaction.transaction_id)
        assert persisted.phase is CutoverPhase.OWNERSHIP_COMMITTED
        assert len(reopened.journal(transaction.transaction_id)) == 13
        reopened.assert_owner(
            session_id="session-001",
            owner_runtime="runtime-b",
            owner_epoch=2,
            fencing_token=2,
            now_ms=21,
        )


def test_transactions_are_single_active_idempotent_and_rollback_scoped() -> None:
    with DurableCoordinator(":memory:") as coordinator:
        coordinator.create_lease(
            session_id="session-001", owner_runtime="runtime-a", expiration_ms=10_000
        )
        transaction = coordinator.begin_transaction(
            session_id="session-001",
            destination_candidate="runtime-b",
            migration_plan_hash=SHA,
            seed=17,
            now_ms=0,
            timeout_ms=1_000,
        )
        with pytest.raises(CoordinatorConflict):
            coordinator.begin_transaction(
                session_id="session-001",
                destination_candidate="runtime-c",
                migration_plan_hash=SHA,
                seed=18,
                now_ms=0,
                timeout_ms=1_000,
            )
        first = coordinator.transition(
            transaction.transaction_id,
            expected=CutoverPhase.PROPOSED,
            target=CutoverPhase.COMPATIBILITY_VALIDATED,
            event_id="compatibility-ok",
            at_ms=1,
            payload_hash=EVENT_SHA,
        )
        replay = coordinator.transition(
            transaction.transaction_id,
            expected=CutoverPhase.PROPOSED,
            target=CutoverPhase.COMPATIBILITY_VALIDATED,
            event_id="compatibility-ok",
            at_ms=1,
            payload_hash=EVENT_SHA,
        )
        assert replay == first
        with pytest.raises(CoordinatorConflict, match="payload"):
            coordinator.transition(
                transaction.transaction_id,
                expected=CutoverPhase.PROPOSED,
                target=CutoverPhase.COMPATIBILITY_VALIDATED,
                event_id="compatibility-ok",
                at_ms=1,
                payload_hash=hashlib.sha256(b"different").hexdigest(),
            )
        with pytest.raises(InvalidTransition):
            coordinator.transition(
                transaction.transaction_id,
                expected=CutoverPhase.COMPATIBILITY_VALIDATED,
                target=CutoverPhase.OWNERSHIP_COMMITTED,
                event_id="skip-commit-intent",
                at_ms=2,
                payload_hash=EVENT_SHA,
            )
        coordinator.transition(
            transaction.transaction_id,
            expected=CutoverPhase.COMPATIBILITY_VALIDATED,
            target=CutoverPhase.ABORTING,
            event_id="abort",
            at_ms=2,
            payload_hash=EVENT_SHA,
        )
        rolled_back = coordinator.transition(
            transaction.transaction_id,
            expected=CutoverPhase.ABORTING,
            target=CutoverPhase.ROLLED_BACK,
            event_id="rollback",
            at_ms=3,
            payload_hash=EVENT_SHA,
        )
        assert rolled_back.phase is CutoverPhase.ROLLED_BACK
        assert coordinator.lease("session-001").owner_runtime == "runtime-a"


def _token(*, epoch: int, index: int, token_id: int | None = None) -> TokenEvent:
    return TokenEvent(
        session_id="session-001",
        owner_epoch=epoch,
        token_index=index,
        token_id=index + 100 if token_id is None else token_id,
        state_commit_version=index + 1,
    )


def test_gateway_deduplicates_replay_and_rejects_gap_stale_and_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "gateway.db"
    with GatewayCommitLedger(path) as ledger:
        ledger.register(session_id="session-001", owner_epoch=1)
        for index in range(3):
            accepted = ledger.accept(_token(epoch=1, index=index))
            assert accepted.disposition == "accepted"
        duplicate = ledger.accept(_token(epoch=1, index=2))
        assert duplicate.disposition == "duplicate"
        assert ledger.watermark("session-001") == 2
        with pytest.raises(TokenMismatch):
            ledger.accept(_token(epoch=1, index=2, token_id=999))
        with pytest.raises(TokenGap):
            ledger.accept(_token(epoch=1, index=4))
        ledger.switch_owner(
            session_id="session-001",
            expected_epoch=1,
            destination_epoch=2,
            expected_watermark=2,
        )
        with pytest.raises(StaleOwner):
            ledger.accept(_token(epoch=1, index=3))
        ledger.accept(_token(epoch=2, index=3), client_acknowledged=True)

    with GatewayCommitLedger(path) as reopened:
        assert [token.token_index for token in reopened.accepted_tokens("session-001")] == [
            0,
            1,
            2,
            3,
        ]
        assert reopened.watermark("session-001") == 3


def test_terminal_token_commits_once() -> None:
    with GatewayCommitLedger(":memory:") as ledger:
        ledger.register(session_id="session-001", owner_epoch=1)
        payload = _token(epoch=1, index=0).model_dump(mode="python")
        payload["terminal"] = True
        terminal = TokenEvent.model_validate(payload)
        ledger.accept(terminal)
        with pytest.raises(TokenMismatch, match="terminal"):
            ledger.accept(_token(epoch=1, index=1))


def test_coordinator_enforces_lease_and_transaction_deadlines() -> None:
    with DurableCoordinator(":memory:") as coordinator:
        lease = coordinator.create_lease(
            session_id="session-expiry", owner_runtime="runtime-a", expiration_ms=20
        )
        renewed = coordinator.renew_lease(
            session_id=lease.session_id,
            owner_runtime=lease.owner_runtime,
            owner_epoch=lease.owner_epoch,
            fencing_token=lease.fencing_token,
            now_ms=5,
            expiration_ms=40,
        )
        assert renewed.expiration_ms == 40
        with pytest.raises(StaleOwner, match="expired"):
            coordinator.assert_owner(
                session_id=lease.session_id,
                owner_runtime=lease.owner_runtime,
                owner_epoch=lease.owner_epoch,
                fencing_token=lease.fencing_token,
                now_ms=40,
            )
        with pytest.raises(StaleOwner, match="expired"):
            coordinator.begin_transaction(
                session_id=lease.session_id,
                destination_candidate="runtime-b",
                migration_plan_hash=SHA,
                seed=5,
                now_ms=40,
                timeout_ms=10,
            )

        active = coordinator.create_lease(
            session_id="session-deadline", owner_runtime="runtime-a", expiration_ms=1_000
        )
        transaction = coordinator.begin_transaction(
            session_id=active.session_id,
            destination_candidate="runtime-b",
            migration_plan_hash=SHA,
            seed=7,
            now_ms=0,
            timeout_ms=20,
        )
        _advance_to_commit(coordinator, transaction.transaction_id)
        with pytest.raises(CoordinatorConflict, match="deadline"):
            coordinator.commit_ownership(
                transaction.transaction_id,
                event_id="late-ownership",
                at_ms=21,
                state_version=3,
            )


def test_concurrent_idempotent_replay_is_serialized() -> None:
    with DurableCoordinator(":memory:") as coordinator:
        coordinator.create_lease(
            session_id="session-concurrent", owner_runtime="runtime-a", expiration_ms=1_000
        )
        transaction = coordinator.begin_transaction(
            session_id="session-concurrent",
            destination_candidate="runtime-b",
            migration_plan_hash=SHA,
            seed=29,
            now_ms=0,
            timeout_ms=100,
        )

        def replay(_: int) -> CutoverPhase:
            return coordinator.transition(
                transaction.transaction_id,
                expected=CutoverPhase.PROPOSED,
                target=CutoverPhase.COMPATIBILITY_VALIDATED,
                event_id="compatibility-concurrent",
                at_ms=1,
                payload_hash=EVENT_SHA,
            ).phase

        with ThreadPoolExecutor(max_workers=8) as pool:
            phases = tuple(pool.map(replay, range(32)))
        assert set(phases) == {CutoverPhase.COMPATIBILITY_VALIDATED}
        assert len(coordinator.journal(transaction.transaction_id)) == 2
