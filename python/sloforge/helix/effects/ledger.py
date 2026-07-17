"""Bounded in-memory effect ledger with explicit commit and compensation boundaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from sloforge.helix.environments.models import canonical_json, content_digest

if TYPE_CHECKING:
    from sloforge.helix.capture import ArtifactWatermark

from .legality import EffectLegalityChecker, EffectLegalityPolicy, IllegalEffectError
from .models import (
    AuditEvidence,
    Effect,
    EffectClass,
    EffectRecord,
    EffectStatus,
    coerce_evidence,
)


class EffectLedgerFull(RuntimeError):
    """The configured audit record bound was reached."""


class EffectCommitError(RuntimeError):
    """A requested commit watermark is invalid or regresses."""


class CompensationError(RuntimeError):
    """An effect cannot be compensated at the requested boundary."""


class IdempotencyConflict(RuntimeError):
    """An idempotency key was reused for a different effect."""


class EffectLedger:
    """Records effect evidence; it never performs an external side effect itself."""

    def __init__(
        self,
        *,
        tenant_id: str = "default",
        external_side_effects_enabled: bool = False,
        max_records: int = 10_000,
    ) -> None:
        if max_records < 1:
            raise ValueError("effect record bound must be positive")
        self.tenant_id = tenant_id
        self.external_side_effects_enabled = external_side_effects_enabled
        self.max_records = max_records
        self._records: list[EffectRecord] = []
        self._idempotency: dict[str, int] = {}
        self._commit_watermark = -1
        self._checker = EffectLegalityChecker()

    @property
    def records(self) -> tuple[EffectRecord, ...]:
        return tuple(self._records)

    @property
    def commit_watermark(self) -> int:
        return self._commit_watermark

    @property
    def watermark(self) -> int:
        """The last applied record, independently of the durable commit boundary."""

        return len(self._records) - 1

    def apply(
        self,
        effect: Effect | object,
        *,
        evidence: Sequence[AuditEvidence | Mapping[str, object] | str] = (),
        speculative: bool = False,
        virtual_time_ns: int = 0,
    ) -> EffectRecord:
        normalized = effect if isinstance(effect, Effect) else Effect.from_ir(effect)
        self._checker.require(
            normalized,
            EffectLegalityPolicy(
                speculative=speculative,
                external_side_effects_enabled=self.external_side_effects_enabled,
                expected_tenant_id=self.tenant_id,
            ),
        )
        if virtual_time_ns < 0:
            raise ValueError("effect virtual time cannot be negative")
        audit = tuple(coerce_evidence(item) for item in evidence)
        if normalized.classification not in {EffectClass.PURE, EffectClass.READ_ONLY} and not audit:
            raise ValueError("write effects require audit evidence")
        if normalized.idempotency_key is not None:
            existing_sequence = self._idempotency.get(normalized.idempotency_key)
            if existing_sequence is not None:
                existing = self._records[existing_sequence]
                if existing.effect.effect_id != normalized.effect_id:
                    raise IdempotencyConflict(
                        "idempotency key is already bound to a different effect"
                    )
                return existing
        if len(self._records) >= self.max_records:
            raise EffectLedgerFull(f"effect ledger reached {self.max_records} records")
        record = EffectRecord(
            sequence=len(self._records),
            effect=normalized,
            status=EffectStatus.APPLIED,
            evidence=audit,
            applied_virtual_time_ns=virtual_time_ns,
        )
        self._records.append(record)
        if normalized.idempotency_key is not None:
            self._idempotency[normalized.idempotency_key] = record.sequence
        return record

    record = apply

    def commit(self, watermark: int | None = None) -> int:
        if not self._records:
            if watermark not in {None, -1}:
                raise EffectCommitError("cannot commit beyond an empty ledger")
            return self._commit_watermark
        next_watermark = len(self._records) - 1 if watermark is None else watermark
        if next_watermark < self._commit_watermark:
            raise EffectCommitError("effect commit watermark cannot regress")
        if next_watermark >= len(self._records):
            raise EffectCommitError("effect commit watermark exceeds the ledger tail")
        for sequence in range(self._commit_watermark + 1, next_watermark + 1):
            record = self._records[sequence]
            if record.status is EffectStatus.COMPENSATED:
                raise EffectCommitError("a compensated effect cannot be committed")
            self._records[sequence] = record.committed(next_watermark)
        self._commit_watermark = next_watermark
        return self._commit_watermark

    def compensate(
        self,
        effect: str | int | EffectRecord,
        *,
        evidence: Sequence[AuditEvidence | Mapping[str, object] | str],
    ) -> EffectRecord:
        sequence = self._resolve(effect)
        record = self._records[sequence]
        if sequence <= self._commit_watermark or record.status is EffectStatus.COMMITTED:
            raise CompensationError("committed effects are beyond the compensation boundary")
        if record.status is EffectStatus.COMPENSATED:
            return record
        if record.effect.classification is not EffectClass.COMPENSATABLE_WRITE:
            raise CompensationError("effect is not declared compensatable")
        if not record.effect.compensation:
            raise CompensationError("effect has no compensation recipe")
        audit = tuple(coerce_evidence(item) for item in evidence)
        if not audit:
            raise CompensationError("compensation requires audit evidence")
        updated = record.compensated(audit)
        self._records[sequence] = updated
        return updated

    def compensate_after(
        self,
        watermark: int,
        *,
        evidence: Sequence[AuditEvidence | Mapping[str, object] | str],
    ) -> tuple[EffectRecord, ...]:
        if watermark < self._commit_watermark:
            raise CompensationError("cannot compensate below the committed watermark")
        if watermark >= len(self._records):
            raise CompensationError("compensation watermark exceeds the ledger tail")
        compensated: list[EffectRecord] = []
        for record in reversed(self._records[watermark + 1 :]):
            if record.status is EffectStatus.COMPENSATED:
                continue
            if record.effect.classification is not EffectClass.COMPENSATABLE_WRITE:
                raise CompensationError(
                    f"effect {record.effect.effect_id} cannot be rolled back by compensation"
                )
            compensated.append(self.compensate(record, evidence=evidence))
        return tuple(compensated)

    def _resolve(self, value: str | int | EffectRecord) -> int:
        if isinstance(value, EffectRecord):
            sequence = value.sequence
        elif isinstance(value, int):
            sequence = value
        else:
            matches = [
                record.sequence for record in self._records if record.effect.effect_id == value
            ]
            if not matches:
                raise KeyError(value)
            sequence = matches[0]
        if sequence < 0 or sequence >= len(self._records):
            raise KeyError(sequence)
        return sequence

    def audit(self) -> tuple[dict[str, object], ...]:
        return tuple(record.to_dict() for record in self._records)

    def artifact_watermark(self, watermark: int | None = None) -> ArtifactWatermark:
        """Late-bind the ledger prefix to coordinated capture's reference model."""

        chosen = self._commit_watermark if watermark is None else watermark
        if chosen < -1 or chosen >= len(self._records):
            raise ValueError("effect artifact watermark is outside the ledger")
        digest = content_digest(self.artifact_payload(chosen))
        from sloforge.helix.capture import ArtifactWatermark

        return ArtifactWatermark(
            artifact_id=f"effects-{digest[:32]}",
            watermark=chosen,
            digest=digest,
        )

    def artifact_payload(self, watermark: int | None = None) -> bytes:
        """Return the exact committed prefix bytes authenticated by its watermark."""

        chosen = self._commit_watermark if watermark is None else watermark
        if chosen < -1 or chosen >= len(self._records):
            raise ValueError("effect artifact watermark is outside the ledger")
        prefix = tuple(record.to_dict() for record in self._records[: chosen + 1])
        return canonical_json(prefix)

    def require_speculatable(self, effects: Sequence[Effect | object]) -> None:
        for value in effects:
            effect = value if isinstance(value, Effect) else Effect.from_ir(value)
            try:
                self._checker.require(
                    effect,
                    EffectLegalityPolicy(
                        speculative=True,
                        external_side_effects_enabled=self.external_side_effects_enabled,
                        expected_tenant_id=self.tenant_id,
                    ),
                )
            except IllegalEffectError as exc:
                raise IllegalEffectError(
                    f"effect {effect.effect_id} is illegal for speculation: {exc}"
                ) from exc
