"""Typed side-effect contracts for Helix speculative execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from typing import Any, cast

from sloforge.helix.environments.models import JsonValue, canonical_json, content_digest
from sloforge.helix.environments.security import redact_mapping


class EffectClass(StrEnum):
    PURE = "PURE"
    READ_ONLY = "READ_ONLY"
    IDEMPOTENT_WRITE = "IDEMPOTENT_WRITE"
    COMPENSATABLE_WRITE = "COMPENSATABLE_WRITE"
    IRREVERSIBLE_WRITE = "IRREVERSIBLE_WRITE"
    EXTERNAL_UNKNOWN = "EXTERNAL_UNKNOWN"


EffectType = EffectClass


class EffectStatus(StrEnum):
    APPLIED = "applied"
    COMMITTED = "committed"
    COMPENSATED = "compensated"


@dataclass(frozen=True, slots=True)
class AuditEvidence:
    evidence_id: str
    kind: str
    digest: str
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        *,
        kind: str,
        payload: bytes | str,
        metadata: Mapping[str, object] | None = None,
        secret_values: tuple[str, ...] = (),
    ) -> AuditEvidence:
        data = payload if isinstance(payload, bytes) else payload.encode()
        digest = content_digest(data)
        body = {
            "kind": kind,
            "digest": digest,
            "metadata": redact_mapping(metadata or {}, secrets=secret_values),
        }
        return cls(
            evidence_id=content_digest(canonical_json(body)),
            kind=kind,
            digest=digest,
            metadata=cast(dict[str, JsonValue], body["metadata"]),
        )


@dataclass(frozen=True, slots=True)
class Effect:
    effect_id: str
    classification: EffectClass
    operation: str
    target: str | None = None
    real_external: bool = False
    stable_read: bool = True
    idempotency_key: str | None = None
    compensation: str | None = None
    tenant_id: str = "default"
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.operation:
            raise ValueError("effect operation cannot be empty")
        if not self.tenant_id:
            raise ValueError("effect tenant cannot be empty")
        if self.classification is EffectClass.PURE and self.real_external:
            raise ValueError("a pure operation cannot be a real external effect")

    @classmethod
    def build(
        cls,
        classification: EffectClass | str,
        operation: str,
        *,
        target: str | None = None,
        real_external: bool = False,
        stable_read: bool = True,
        idempotency_key: str | None = None,
        compensation: str | None = None,
        tenant_id: str = "default",
        metadata: Mapping[str, object] | None = None,
        secret_values: tuple[str, ...] = (),
    ) -> Effect:
        effect_class = EffectClass(classification)
        safe_metadata = redact_mapping(metadata or {}, secrets=secret_values)
        identity = {
            "classification": effect_class.value,
            "operation": operation,
            "target": target,
            "real_external": real_external,
            "stable_read": stable_read,
            "idempotency_key": idempotency_key,
            "compensation": compensation,
            "tenant_id": tenant_id,
            "metadata": safe_metadata,
        }
        return cls(
            effect_id=content_digest(canonical_json(identity)),
            classification=effect_class,
            operation=operation,
            target=target,
            real_external=real_external,
            stable_read=stable_read,
            idempotency_key=idempotency_key,
            compensation=compensation,
            tenant_id=tenant_id,
            metadata=safe_metadata,
        )

    @classmethod
    def from_ir(cls, value: object, *, tenant_id: str = "default") -> Effect:
        """Duck-type an IR effect declaration without importing the IR package."""

        if isinstance(value, Effect):
            return value
        if isinstance(value, Mapping):
            raw = cast(Mapping[str, Any], value)
            getter = raw.get
        else:

            def getter(key: str, default: object = None) -> object:
                return getattr(value, key, default)

        classification = getter("classification", getter("effect_class", getter("kind", None)))
        if classification is None:
            raise ValueError("IR effect declaration has no classification")
        return cls.build(
            cast(str, classification),
            str(getter("operation", getter("name", "effect"))),
            target=cast(str | None, getter("target", None)),
            real_external=bool(getter("real_external", getter("external", False))),
            stable_read=bool(getter("stable_read", getter("stable", True))),
            idempotency_key=cast(str | None, getter("idempotency_key", None)),
            compensation=cast(str | None, getter("compensation", None)),
            tenant_id=str(getter("tenant_id", tenant_id)),
            metadata=cast(Mapping[str, object], getter("metadata", {})),
        )

    def verify_identity(self) -> None:
        expected = Effect.build(
            self.classification,
            self.operation,
            target=self.target,
            real_external=self.real_external,
            stable_read=self.stable_read,
            idempotency_key=self.idempotency_key,
            compensation=self.compensation,
            tenant_id=self.tenant_id,
            metadata=self.metadata,
        )
        if expected.effect_id != self.effect_id:
            raise ValueError("effect identity digest mismatch")


EffectSpec = Effect


@dataclass(frozen=True, slots=True)
class EffectRecord:
    sequence: int
    effect: Effect
    status: EffectStatus
    evidence: tuple[AuditEvidence, ...]
    applied_virtual_time_ns: int
    commit_watermark: int | None = None
    compensation_evidence: tuple[AuditEvidence, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def committed(self, watermark: int) -> EffectRecord:
        return replace(self, status=EffectStatus.COMMITTED, commit_watermark=watermark)

    def compensated(self, evidence: tuple[AuditEvidence, ...]) -> EffectRecord:
        return replace(
            self,
            status=EffectStatus.COMPENSATED,
            compensation_evidence=evidence,
        )


def coerce_evidence(value: AuditEvidence | Mapping[str, object] | str) -> AuditEvidence:
    if isinstance(value, AuditEvidence):
        return value
    if isinstance(value, str):
        return AuditEvidence.build(kind="assertion", payload=value)
    kind = str(value.get("kind", "assertion"))
    payload_value = value.get("payload", value.get("digest", canonical_json(dict(value))))
    payload = (
        payload_value if isinstance(payload_value, bytes | str) else canonical_json(payload_value)
    )
    metadata = value.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise TypeError("audit evidence metadata must be a mapping")
    return AuditEvidence.build(kind=kind, payload=payload, metadata=metadata)
