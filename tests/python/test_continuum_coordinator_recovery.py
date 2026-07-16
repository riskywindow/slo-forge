from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from sloforge.continuum.transaction import (
    CoordinatorConflict,
    CutoverPhase,
    DurableCoordinator,
    InvalidTransition,
    RecoveryEvidence,
)

PLAN_HASH = hashlib.sha256(b"coordinator-recovery-plan").hexdigest()
PAYLOAD_HASH = hashlib.sha256(b"coordinator-recovery-event").hexdigest()


def _evidence_id(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _begin(
    coordinator: DurableCoordinator,
    *,
    session_id: str,
    timeout_ms: int = 10,
    lease_expiration_ms: int = 1_000,
) -> str:
    coordinator.create_lease(
        session_id=session_id,
        owner_runtime="source",
        expiration_ms=lease_expiration_ms,
        initial_token_index=4,
    )
    return coordinator.begin_transaction(
        session_id=session_id,
        destination_candidate="destination",
        migration_plan_hash=PLAN_HASH,
        seed=47,
        now_ms=0,
        timeout_ms=timeout_ms,
    ).transaction_id


def _advance_to_commit(coordinator: DurableCoordinator, identifier: str) -> None:
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
            event_id=f"recovery-setup-{tick}",
            at_ms=tick,
            payload_hash=PAYLOAD_HASH,
            commit_watermark=4 if target is CutoverPhase.COMMIT_INTENT_RECORDED else None,
            rollback_watermark=(4 if target is CutoverPhase.COMMIT_INTENT_RECORDED else None),
        )
        current = target
    coordinator.commit_ownership(
        identifier,
        event_id="recovery-setup-ownership",
        at_ms=12,
        state_version=17,
    )


def _precommit_evidence(identifier: str, *, evidence_label: str) -> RecoveryEvidence:
    return RecoveryEvidence(
        evidence_id=_evidence_id(evidence_label),
        transaction_id=identifier,
        observed_owner_runtime="source",
        observed_owner_epoch=1,
        observed_fencing_token=1,
        source_available=True,
        source_fenced=False,
        source_resumable=True,
        source_commit_watermark=7,
        destination_available=False,
        destination_validated=False,
        destination_aborted=True,
        gateway_owner_epoch=1,
        gateway_commit_watermark=7,
        cleanup_completed=True,
        reason="orchestrator process restarted after destination preparation",
    )


def _postcommit_evidence(
    identifier: str,
    *,
    evidence_label: str,
    gateway_owner_epoch: int,
) -> RecoveryEvidence:
    return RecoveryEvidence(
        evidence_id=_evidence_id(evidence_label),
        transaction_id=identifier,
        observed_owner_runtime="destination",
        observed_owner_epoch=2,
        observed_fencing_token=2,
        source_available=True,
        source_fenced=True,
        source_resumable=False,
        source_commit_watermark=4,
        destination_available=True,
        destination_validated=True,
        destination_aborted=False,
        gateway_owner_epoch=gateway_owner_epoch,
        gateway_commit_watermark=4,
        cleanup_completed=True,
        reason="coordinator restarted after ownership commit",
    )


def test_reopen_enumerates_and_rolls_back_precommit_after_deadline(tmp_path: Path) -> None:
    path = tmp_path / "precommit-recovery.db"
    with DurableCoordinator(path) as coordinator:
        identifier = _begin(coordinator, session_id="precommit-session")
        coordinator.transition(
            identifier,
            expected=CutoverPhase.PROPOSED,
            target=CutoverPhase.COMPATIBILITY_VALIDATED,
            event_id="precommit-compatible",
            at_ms=1,
            payload_hash=PAYLOAD_HASH,
        )
        coordinator.transition(
            identifier,
            expected=CutoverPhase.COMPATIBILITY_VALIDATED,
            target=CutoverPhase.DESTINATION_PREPARING,
            event_id="precommit-preparing",
            at_ms=2,
            payload_hash=PAYLOAD_HASH,
        )

    evidence = _precommit_evidence(identifier, evidence_label="precommit-restart")
    with DurableCoordinator(path) as reopened:
        assert tuple(item.transaction_id for item in reopened.recoverable_transactions()) == (
            identifier,
        )
        with pytest.raises(CoordinatorConflict, match="deadline"):
            reopened.transition(
                identifier,
                expected=CutoverPhase.DESTINATION_PREPARING,
                target=CutoverPhase.PRECOPYING,
                event_id="late-forward-work",
                at_ms=20,
                payload_hash=PAYLOAD_HASH,
            )
        recovered = reopened.recover_transaction(evidence, at_ms=20)
        assert recovered.prior_phase is CutoverPhase.DESTINATION_PREPARING
        assert recovered.terminal_phase is CutoverPhase.ROLLED_BACK
        assert not recovered.ownership_committed
        assert recovered.recovered_at_ms == 21
        assert reopened.lease("precommit-session").owner_runtime == "source"
        assert reopened.lease("precommit-session").owner_epoch == 1
        journal = reopened.journal(identifier)
        assert tuple(item.to_phase for item in journal[-2:]) == (
            CutoverPhase.ABORTING,
            CutoverPhase.ROLLED_BACK,
        )
        assert tuple(item.at_ms for item in journal[-2:]) == (20, 21)
        before_replay = len(journal)
        assert reopened.recover_transaction(evidence, at_ms=20) == recovered
        assert len(reopened.journal(identifier)) == before_replay
        assert reopened.recoverable_transactions() == ()
        altered = evidence.model_copy(update={"reason": "different recovery claim"})
        with pytest.raises(CoordinatorConflict, match="evidence"):
            reopened.recover_transaction(altered, at_ms=20)


def test_expired_source_lease_cannot_be_reactivated_by_recovery(tmp_path: Path) -> None:
    path = tmp_path / "expired-source.db"
    with DurableCoordinator(path) as coordinator:
        identifier = _begin(
            coordinator,
            session_id="expired-source-session",
            lease_expiration_ms=15,
        )

    evidence = _precommit_evidence(identifier, evidence_label="expired-source")
    with DurableCoordinator(path) as reopened:
        recovered = reopened.recover_transaction(evidence, at_ms=20)
        assert recovered.terminal_phase is CutoverPhase.FAILED_BEFORE_COMMIT
        lease = reopened.lease("expired-source-session")
        assert lease.owner_runtime == "source"
        assert lease.owner_epoch == 1
        with pytest.raises(InvalidTransition):
            reopened.transition(
                identifier,
                expected=CutoverPhase.FAILED_BEFORE_COMMIT,
                target=CutoverPhase.ROLLED_BACK,
                event_id="unsafe-source-revival",
                at_ms=22,
                payload_hash=PAYLOAD_HASH,
            )


@pytest.mark.parametrize(
    ("gateway_owner_epoch", "expected_terminal"),
    (
        (2, CutoverPhase.FAILED_AFTER_COMMIT),
        (1, CutoverPhase.OPERATOR_REQUIRED),
    ),
)
def test_postcommit_restart_never_revives_source_and_classifies_gateway_state(
    tmp_path: Path,
    gateway_owner_epoch: int,
    expected_terminal: CutoverPhase,
) -> None:
    path = tmp_path / f"postcommit-{gateway_owner_epoch}.db"
    with DurableCoordinator(path) as coordinator:
        identifier = _begin(
            coordinator,
            session_id=f"postcommit-session-{gateway_owner_epoch}",
            timeout_ms=100,
        )
        _advance_to_commit(coordinator, identifier)

    evidence = _postcommit_evidence(
        identifier,
        evidence_label=f"postcommit-{gateway_owner_epoch}",
        gateway_owner_epoch=gateway_owner_epoch,
    )
    with DurableCoordinator(path) as reopened:
        recovered = reopened.recover_transaction(evidence, at_ms=200)
        assert recovered.ownership_committed
        assert recovered.terminal_phase is expected_terminal
        lease = reopened.lease(f"postcommit-session-{gateway_owner_epoch}")
        assert lease.owner_runtime == "destination"
        assert lease.owner_epoch == 2
        assert lease.fencing_token == 2
        assert reopened.recover_transaction(evidence, at_ms=200) == recovered
