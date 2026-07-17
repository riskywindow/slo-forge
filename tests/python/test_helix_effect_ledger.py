from __future__ import annotations

import pytest

from sloforge.helix.effects import (
    CompensationError,
    Effect,
    EffectClass,
    EffectCommitError,
    EffectLedger,
    EffectStatus,
    IdempotencyConflict,
    IllegalEffectError,
    require_effect_legal,
)


def test_illegal_speculative_effects_and_unstable_reads_are_rejected() -> None:
    irreversible = Effect.build(
        EffectClass.IRREVERSIBLE_WRITE,
        "send-email",
        real_external=True,
        target="mailbox",
    )
    unknown = Effect.build(
        EffectClass.EXTERNAL_UNKNOWN,
        "vendor-hook",
        real_external=True,
    )
    unstable = Effect.build(EffectClass.READ_ONLY, "wall-clock", stable_read=False)
    for effect in (irreversible, unknown, unstable):
        with pytest.raises(IllegalEffectError):
            require_effect_legal(
                effect,
                speculative=True,
                external_side_effects_enabled=True,
            )


def test_idempotency_commit_evidence_and_compensation_boundaries() -> None:
    ledger = EffectLedger(
        tenant_id="tenant-a",
        external_side_effects_enabled=True,
        max_records=8,
    )
    idempotent = Effect.build(
        EffectClass.IDEMPOTENT_WRITE,
        "put-object",
        idempotency_key="request-7",
        real_external=True,
        tenant_id="tenant-a",
    )
    first = ledger.apply(idempotent, evidence=("object receipt",), virtual_time_ns=10)
    assert ledger.apply(idempotent, evidence=("retry receipt",)) == first
    conflict = Effect.build(
        EffectClass.IDEMPOTENT_WRITE,
        "put-different-object",
        idempotency_key="request-7",
        real_external=True,
        tenant_id="tenant-a",
    )
    with pytest.raises(IdempotencyConflict):
        ledger.apply(conflict, evidence=("receipt",))

    compensatable = Effect.build(
        EffectClass.COMPENSATABLE_WRITE,
        "reserve-quota",
        compensation="release-quota",
        real_external=True,
        tenant_id="tenant-a",
    )
    reservation = ledger.apply(compensatable, evidence=("reservation receipt",))
    assert ledger.watermark == reservation.sequence
    compensated = ledger.compensate(reservation, evidence=("release receipt",))
    assert compensated.status is EffectStatus.COMPENSATED
    with pytest.raises(EffectCommitError):
        ledger.commit()


def test_committed_and_noncompensatable_effects_cannot_be_compensated() -> None:
    ledger = EffectLedger(tenant_id="tenant-a")
    pure = ledger.apply(Effect.build(EffectClass.PURE, "calculate", tenant_id="tenant-a"))
    ledger.commit(pure.sequence)
    assert ledger.artifact_watermark().watermark == pure.sequence
    with pytest.raises(CompensationError):
        ledger.compensate(pure, evidence=("not possible",))

    second = EffectLedger(tenant_id="tenant-a")
    write = second.apply(
        Effect.build(
            EffectClass.IDEMPOTENT_WRITE,
            "local-write",
            idempotency_key="write-1",
            tenant_id="tenant-a",
        ),
        evidence=("local digest",),
    )
    with pytest.raises(CompensationError):
        second.compensate(write, evidence=("not declared",))


def test_external_writes_are_disabled_by_default_and_metadata_is_redacted() -> None:
    effect = Effect.build(
        EffectClass.IDEMPOTENT_WRITE,
        "external-write",
        real_external=True,
        idempotency_key="key",
        metadata={"api_token": "secret", "safe": "secret-value"},
        secret_values=("secret",),
    )
    assert "secret" not in str(effect.metadata)
    with pytest.raises(IllegalEffectError):
        EffectLedger().apply(effect, evidence=("receipt",))
