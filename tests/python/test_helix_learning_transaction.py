from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest

from sloforge.helix.transactions import (
    ArtifactReference,
    LearningState,
    LearningTransactionStore,
)


def _store(tmp_path: Path) -> LearningTransactionStore:
    store = LearningTransactionStore(tmp_path / "transactions.sqlite")
    store.create(
        transaction_id="tx-1",
        deployment="coding-agent-prod",
        champion_policy_epoch_id="champion-0",
        trigger_hash=sha256(b"actual failure evidence").hexdigest(),
        seed=47,
        observed_at_ms=1,
    )
    return store


def test_transaction_is_durable_hash_checked_and_transition_ordered(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        artifact = ArtifactReference(
            artifact_id="failure-1",
            artifact_kind="production_failure",
            sha256=sha256(b"failure").hexdigest(),
            uri="artifacts/helix/failure-1.json",
        )
        store.add_artifact("tx-1", artifact)
        captured = store.transition(
            "tx-1",
            target=LearningState.CAPTURE_PROPOSED,
            reason="capture authorized for isolated synthetic fixture",
            observed_at_ms=2,
            evidence_artifact_ids=(artifact.artifact_id,),
        )
        assert captured.sequence == 1
        store.record_cost("tx-1", cost_id="cpu-1", amount_usd=0.0, source="local CPU")
        store.record_resource(
            "tx-1", usage_id="cpu-ms-1", resource_kind="cpu", quantity=12.5, unit="ms"
        )
        assert store.transaction("tx-1").cost_usd == 0.0
        assert [event.state_after for event in store.events("tx-1")] == [
            LearningState.OBSERVED,
            LearningState.CAPTURE_PROPOSED,
        ]

    with LearningTransactionStore(tmp_path / "transactions.sqlite") as reopened:
        assert reopened.transaction("tx-1").state is LearningState.CAPTURE_PROPOSED


def test_invalid_skip_and_unknown_evidence_are_rejected(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        with pytest.raises(ValueError, match="invalid learning transition"):
            store.transition(
                "tx-1",
                target=LearningState.TRAINING,
                reason="illegal skip",
                observed_at_ms=2,
            )
        with pytest.raises(ValueError, match="unknown artifacts"):
            store.transition(
                "tx-1",
                target=LearningState.CAPTURE_PROPOSED,
                reason="missing evidence",
                observed_at_ms=2,
                evidence_artifact_ids=("absent",),
            )


def test_injected_state_only_write_rolls_back_atomically(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        with pytest.raises(RuntimeError, match="injected fault"):
            store.transition(
                "tx-1",
                target=LearningState.CAPTURE_PROPOSED,
                reason="fault injection",
                observed_at_ms=2,
                fault_after_state_update=True,
            )
        current = store.transaction("tx-1")
        assert current.state is LearningState.OBSERVED
        assert current.sequence == 0
        assert len(store.events("tx-1")) == 1


def test_failure_terminal_and_raw_evidence_tampering(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        rejected = store.transition(
            "tx-1",
            target=LearningState.EXPERIENCE_REJECTED,
            reason="privacy authorization absent",
            observed_at_ms=2,
        )
        assert rejected.state.terminal
        with pytest.raises(ValueError, match="terminal"):
            store.transition(
                "tx-1",
                target=LearningState.CAPTURE_PROPOSED,
                reason="cannot resume rejected evidence",
                observed_at_ms=3,
            )

    tampered = LearningTransactionStore(tmp_path / "transactions.sqlite")
    tampered._connection.execute(
        "UPDATE learning_events SET reason='tampered' WHERE transaction_id='tx-1' AND sequence=1"
    )
    tampered._connection.commit()
    with pytest.raises(ValueError, match="integrity"):
        tampered.transaction("tx-1")
    tampered.close()


def test_transaction_retries_are_exact_and_accounting_ids_are_idempotent(tmp_path: Path) -> None:
    with LearningTransactionStore(tmp_path / "transactions.sqlite", max_artifacts=1) as store:
        created = store.create(
            transaction_id="tx-1",
            deployment="prod",
            champion_policy_epoch_id="champion-0",
            trigger_hash=sha256(b"trigger").hexdigest(),
            seed=7,
            observed_at_ms=1,
        )
        assert (
            store.create(
                transaction_id="tx-1",
                deployment="prod",
                champion_policy_epoch_id="champion-0",
                trigger_hash=sha256(b"trigger").hexdigest(),
                seed=7,
                observed_at_ms=1,
            )
            == created
        )
        with pytest.raises(ValueError, match="different coordinator inputs"):
            store.create(
                transaction_id="tx-1",
                deployment="other",
                champion_policy_epoch_id="champion-0",
                trigger_hash=sha256(b"trigger").hexdigest(),
                seed=7,
                observed_at_ms=1,
            )
        artifact = ArtifactReference(
            artifact_id="a",
            artifact_kind="evidence",
            sha256=sha256(b"a").hexdigest(),
            uri="artifact/a",
        )
        store.add_artifact("tx-1", artifact)
        store.add_artifact("tx-1", artifact)
        store.record_cost("tx-1", cost_id="cost", amount_usd=1.0, source="fixture")
        store.record_cost("tx-1", cost_id="cost", amount_usd=1.0, source="fixture")
        store.record_resource("tx-1", usage_id="cpu", resource_kind="cpu", quantity=1.0, unit="ms")
        store.record_resource("tx-1", usage_id="cpu", resource_kind="cpu", quantity=1.0, unit="ms")
        store.transition(
            "tx-1",
            target=LearningState.CAPTURE_PROPOSED,
            reason="exact",
            observed_at_ms=2,
            evidence_artifact_ids=("a",),
        )
        store.transition(
            "tx-1",
            target=LearningState.CAPTURE_PROPOSED,
            reason="exact",
            observed_at_ms=2,
            evidence_artifact_ids=("a",),
        )
        with pytest.raises(ValueError, match="does not match"):
            store.transition(
                "tx-1",
                target=LearningState.CAPTURE_PROPOSED,
                reason="changed",
                observed_at_ms=2,
                evidence_artifact_ids=("a",),
            )
