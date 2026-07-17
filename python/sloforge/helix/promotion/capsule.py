"""Trusted local builder and validator for canonical Helix promotion capsules."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal, TypeAlias, cast

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from sloforge.continuum.compatibility import CompatibilityDecision
from sloforge.continuum.ir import CompatibilityReport, ExactnessClass
from sloforge.helix.ir import (
    Digest,
    EvidencePointer,
    LineageReference,
    PolicyEpoch,
    PolicyPromotionCapsule,
    PromotionDecision,
)
from sloforge.helix.ir.canonical import canonical_hash

from .registry import GateEvidence, PolicyRegistry, PromotionRecord, PromotionState

GateName: TypeAlias = Literal[
    "lineage",
    "reward_integrity",
    "quality",
    "safety",
    "serving",
    "compatibility",
    "shadow",
    "canary",
]
CompatibilityArtifact: TypeAlias = CompatibilityDecision | CompatibilityReport
CapsuleInput: TypeAlias = "TrustedPolicyPromotionCapsule | str | bytes | Mapping[str, object]"

_ALL_GATES: frozenset[str] = frozenset(
    {
        "lineage",
        "reward_integrity",
        "quality",
        "safety",
        "serving",
        "compatibility",
        "shadow",
        "canary",
    }
)
_MAX_ARTIFACT_BYTES = 16 * 1024 * 1024
_MAX_TOTAL_ARTIFACT_BYTES = 64 * 1024 * 1024
_MAX_VALIDITY_MS = 24 * 60 * 60 * 1000
_U64_MAX = 2**64 - 1

Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:@/-]{0,255}$")]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
NonNegativeInt = Annotated[int, Field(ge=0, le=_U64_MAX)]
RelativePath = Annotated[str, StringConstraints(min_length=1, max_length=4096)]
NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2048)]


class PromotionCapsuleValidationError(ValueError):
    """The capsule cannot authorize promotion because trust validation failed."""


class _CapsuleModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class PromotionArtifactSource(_CapsuleModel):
    """Untrusted local source that the builder resolves and hashes independently."""

    gate: GateName
    relative_path: RelativePath
    media_type: NonEmpty = "application/json"
    captured_at: NonEmpty
    captured_at_ms: NonNegativeInt

    @model_validator(mode="after")
    def safe_relative_path(self) -> PromotionArtifactSource:
        path = PurePosixPath(self.relative_path)
        if path.is_absolute() or ".." in path.parts or "." in path.parts:
            raise ValueError("promotion evidence paths must be normalized relative paths")
        if str(path) != self.relative_path or not path.name:
            raise ValueError("promotion evidence path is not normalized")
        return self


class PromotionArtifactReference(_CapsuleModel):
    gate: GateName
    relative_path: RelativePath
    media_type: NonEmpty
    captured_at: NonEmpty
    captured_at_ms: NonNegativeInt
    artifact_hash: Sha256
    byte_length: Annotated[int, Field(gt=0, le=_MAX_ARTIFACT_BYTES)]


class ContinuumCompatibilityBinding(_CapsuleModel):
    artifact_kind: Literal["decision", "report"]
    compatibility_class: ExactnessClass
    safe: Literal[True]
    canonical_digest: Sha256
    report_id: Identifier | None = None
    required_recomputation: Annotated[tuple[Identifier, ...], Field(max_length=256)] = ()

    @model_validator(mode="after")
    def safe_compatibility(self) -> ContinuumCompatibilityBinding:
        if self.compatibility_class is ExactnessClass.INCOMPATIBLE:
            raise ValueError("incompatible Continuum state cannot authorize policy promotion")
        if self.artifact_kind == "report" and self.report_id is None:
            raise ValueError("Continuum compatibility reports require their report identity")
        if self.artifact_kind == "decision" and self.report_id is not None:
            raise ValueError("engine compatibility decisions do not carry report identities")
        if (
            self.compatibility_class is ExactnessClass.RECOMPUTATION_ASSISTED
            and not self.required_recomputation
        ):
            raise ValueError("recomputation-assisted compatibility requires explicit components")
        if len(set(self.required_recomputation)) != len(self.required_recomputation):
            raise ValueError("Continuum recomputation components contain duplicates")
        return self


class TrustedPolicyPromotionCapsule(_CapsuleModel):
    """Content-addressed trust envelope around the canonical Helix IR capsule."""

    schema_version: Literal["sloforge.helix.trusted-policy-promotion/v1"] = (
        "sloforge.helix.trusted-policy-promotion/v1"
    )
    capsule_digest: Sha256
    tenant_id: Identifier = "default"
    ir_capsule: PolicyPromotionCapsule
    ir_digest: Sha256
    deployment: Identifier
    registry_champion_policy_epoch_id: Identifier
    registry_candidate_policy_epoch_id: Identifier
    registry_evidence_hash: Sha256
    gate_evidence: Annotated[tuple[GateEvidence, ...], Field(min_length=8, max_length=8)]
    artifacts: Annotated[tuple[PromotionArtifactReference, ...], Field(min_length=8, max_length=8)]
    continuum_compatibility: ContinuumCompatibilityBinding
    created_at_ms: NonNegativeInt
    expires_at_ms: NonNegativeInt
    maximum_evidence_age_ms: Annotated[int, Field(gt=0, le=_MAX_VALIDITY_MS)]
    seed: NonNegativeInt

    @model_validator(mode="after")
    def verify_seals_and_bindings(self) -> TrustedPolicyPromotionCapsule:
        if self.expires_at_ms <= self.created_at_ms:
            raise ValueError("trusted promotion capsule expiry must follow creation")
        if self.expires_at_ms - self.created_at_ms > _MAX_VALIDITY_MS:
            raise ValueError("trusted promotion capsule validity exceeds the hard bound")
        if canonical_hash(self.ir_capsule) != self.ir_digest:
            raise ValueError("canonical Helix promotion IR digest is invalid")
        if self.ir_capsule.decision is not PromotionDecision.PROMOTE:
            raise ValueError("trusted promotion capsules must carry a promote decision")
        if self.ir_capsule.transaction_id == "":
            raise ValueError("trusted promotion transaction must not be empty")

        gates = [item.gate for item in self.gate_evidence]
        artifact_gates = [item.gate for item in self.artifacts]
        if len(set(gates)) != len(gates) or set(gates) != _ALL_GATES:
            raise ValueError("trusted promotion requires exactly one evidence record per gate")
        if len(set(artifact_gates)) != len(artifact_gates) or set(artifact_gates) != _ALL_GATES:
            raise ValueError("trusted promotion requires exactly one local artifact per gate")
        evidence_by_gate = {item.gate: item for item in self.gate_evidence}
        artifact_by_gate = {item.gate: item for item in self.artifacts}
        for gate in _ALL_GATES:
            evidence = evidence_by_gate[cast(GateName, gate)]
            artifact = artifact_by_gate[cast(GateName, gate)]
            if not evidence.passed:
                raise ValueError(f"promotion gate {gate!r} did not pass")
            if evidence.sample_count <= 0:
                raise ValueError(f"promotion gate {gate!r} has no evaluated samples")
            if not math.isfinite(evidence.measured_value) or not math.isfinite(evidence.threshold):
                raise ValueError(f"promotion gate {gate!r} contains a non-finite measurement")
            if evidence.artifact_hash != artifact.artifact_hash:
                raise ValueError(f"promotion gate {gate!r} is not bound to its local artifact")
            if artifact.captured_at_ms > self.created_at_ms:
                raise ValueError(f"promotion gate {gate!r} was captured after capsule creation")

        pointer_by_uri = {pointer.uri: pointer for pointer in self.ir_capsule.evaluation_evidence}
        expected_uris = {_artifact_uri(item.relative_path) for item in self.artifacts}
        if len(pointer_by_uri) != len(self.ir_capsule.evaluation_evidence):
            raise ValueError("canonical promotion IR contains duplicate evidence URIs")
        if set(pointer_by_uri) != expected_uris:
            raise ValueError("canonical promotion IR evidence does not cover sealed artifacts")
        for artifact in self.artifacts:
            pointer = pointer_by_uri[_artifact_uri(artifact.relative_path)]
            if (
                pointer.digest.value != artifact.artifact_hash
                or pointer.media_type != artifact.media_type
                or pointer.captured_at != artifact.captured_at
            ):
                raise ValueError("canonical promotion IR evidence pointer is not sealed")

        expected = self.model_dump(mode="json", exclude={"capsule_digest"})
        if canonical_hash(expected) != self.capsule_digest:
            raise ValueError("trusted promotion capsule digest is invalid")
        return self


class PromotionCapsuleValidation(_CapsuleModel):
    schema_version: Literal["sloforge.helix.promotion-validation/v1"] = (
        "sloforge.helix.promotion-validation/v1"
    )
    capsule_digest: Sha256
    tenant_id: Identifier = "default"
    transaction_id: Identifier
    deployment: Identifier
    registry_state: Literal[PromotionState.CANARY_PASSED, PromotionState.ACTIVE]
    compatibility_class: ExactnessClass
    rehashed_artifacts: Annotated[
        tuple[PromotionArtifactReference, ...], Field(min_length=8, max_length=8)
    ]
    validated_at_ms: NonNegativeInt
    eligible_for_promotion: Literal[True] = True


def _artifact_uri(relative_path: str) -> str:
    return f"local-promotion-artifact://{relative_path}"


def _registry_evidence_hash(evidence: tuple[GateEvidence, ...]) -> str:
    payload = json.dumps(
        [item.model_dump(mode="json") for item in sorted(evidence, key=lambda item: item.gate)],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _resolve_artifact(root: Path, relative_path: str) -> Path:
    root_resolved = root.resolve(strict=True)
    candidate = root_resolved.joinpath(*PurePosixPath(relative_path).parts)
    if candidate.is_symlink():
        raise PromotionCapsuleValidationError("promotion evidence must not be a symlink")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root_resolved)
    except (FileNotFoundError, ValueError) as exc:
        raise PromotionCapsuleValidationError(
            f"promotion evidence {relative_path!r} is missing or escapes the artifact root"
        ) from exc
    return resolved


def _read_artifact(root: Path, source: PromotionArtifactSource) -> bytes:
    path = _resolve_artifact(root, source.relative_path)
    descriptor = os.open(path, os.O_RDONLY)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise PromotionCapsuleValidationError("promotion evidence must be a regular file")
        if before.st_size <= 0 or before.st_size > _MAX_ARTIFACT_BYTES:
            raise PromotionCapsuleValidationError("promotion evidence violates the file-size bound")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read(_MAX_ARTIFACT_BYTES + 1)
        after = os.fstat(descriptor)
        identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if identity_before != identity_after or len(payload) != before.st_size:
            raise PromotionCapsuleValidationError("promotion evidence changed while it was read")
        if len(payload) > _MAX_ARTIFACT_BYTES:
            raise PromotionCapsuleValidationError("promotion evidence violates the file-size bound")
        return payload
    finally:
        os.close(descriptor)


def _seal_artifacts(
    root: Path,
    sources: tuple[PromotionArtifactSource, ...],
) -> tuple[tuple[PromotionArtifactReference, ...], dict[GateName, bytes]]:
    gates = [source.gate for source in sources]
    if len(sources) != len(_ALL_GATES) or len(set(gates)) != len(gates) or set(gates) != _ALL_GATES:
        raise PromotionCapsuleValidationError("exactly one local source is required for every gate")
    references: list[PromotionArtifactReference] = []
    payloads: dict[GateName, bytes] = {}
    total = 0
    for source in sorted(sources, key=lambda item: item.gate):
        payload = _read_artifact(root, source)
        total += len(payload)
        if total > _MAX_TOTAL_ARTIFACT_BYTES:
            raise PromotionCapsuleValidationError("promotion evidence exceeds the total-size bound")
        digest = hashlib.sha256(payload).hexdigest()
        references.append(
            PromotionArtifactReference(
                gate=source.gate,
                relative_path=source.relative_path,
                media_type=source.media_type,
                captured_at=source.captured_at,
                captured_at_ms=source.captured_at_ms,
                artifact_hash=digest,
                byte_length=len(payload),
            )
        )
        payloads[source.gate] = payload
    return tuple(references), payloads


def _parse_compatibility(
    payload: bytes,
    artifact_kind: Literal["decision", "report"],
) -> CompatibilityArtifact:
    try:
        if artifact_kind == "decision":
            return CompatibilityDecision.model_validate_json(payload, strict=True)
        return CompatibilityReport.model_validate_json(payload, strict=True)
    except ValueError as exc:
        raise PromotionCapsuleValidationError(
            "compatibility gate artifact is not valid typed Continuum evidence"
        ) from exc


def _compatibility_binding(value: CompatibilityArtifact) -> ContinuumCompatibilityBinding:
    if isinstance(value, CompatibilityDecision):
        if not value.safe or value.compatibility_class is ExactnessClass.INCOMPATIBLE:
            raise PromotionCapsuleValidationError("Continuum compatibility decision is unsafe")
        required = tuple(
            component
            for requirement in value.required_recomputation
            for component in requirement.state_components
        )
        return ContinuumCompatibilityBinding(
            artifact_kind="decision",
            compatibility_class=value.compatibility_class,
            safe=True,
            canonical_digest=canonical_hash(value),
            required_recomputation=required,
        )
    if value.compatibility_class is ExactnessClass.INCOMPATIBLE or value.unsupported_state:
        raise PromotionCapsuleValidationError("Continuum compatibility report is unsafe")
    return ContinuumCompatibilityBinding(
        artifact_kind="report",
        compatibility_class=value.compatibility_class,
        safe=True,
        canonical_digest=canonical_hash(value),
        report_id=value.report_id,
        required_recomputation=tuple(item.component_id for item in value.required_recomputation),
    )


def _validate_registry_binding(
    registry: PolicyRegistry,
    record: PromotionRecord,
    *,
    evidence: tuple[GateEvidence, ...],
) -> None:
    if record.tenant_id != registry.tenant_id:
        raise PromotionCapsuleValidationError("registry promotion belongs to a different tenant")
    if any(item.tenant_id != registry.tenant_id for item in evidence):
        raise PromotionCapsuleValidationError(
            "promotion evidence crosses the registry tenant boundary"
        )
    if record.state not in {PromotionState.CANARY_PASSED, PromotionState.ACTIVE}:
        raise PromotionCapsuleValidationError(
            "registry promotion has not passed both shadow and canary gates"
        )
    expected_hash = _registry_evidence_hash(evidence)
    if expected_hash != record.evidence_hash:
        raise PromotionCapsuleValidationError(
            "capsule gate evidence does not match the registry transaction"
        )
    champion = registry.champion(record.deployment).policy_epoch_id
    expected_champion = (
        record.candidate_policy_epoch_id
        if record.state is PromotionState.ACTIVE
        else record.champion_policy_epoch_id
    )
    if champion != expected_champion:
        raise PromotionCapsuleValidationError(
            "registry champion pointer no longer matches the promotion transaction"
        )


def _policy_epoch_id(policy: PolicyEpoch) -> str:
    return f"{policy.policy_id}@{policy.epoch}"


def build_policy_promotion_capsule(
    *,
    registry: PolicyRegistry,
    transaction_id: str,
    promotion_id: str,
    from_policy_epoch: PolicyEpoch,
    to_policy_epoch: PolicyEpoch,
    gate_evidence: tuple[GateEvidence, ...],
    artifact_root: Path,
    artifact_sources: tuple[PromotionArtifactSource, ...],
    continuum_artifact_kind: Literal["decision", "report"],
    approved_by: str,
    promoted_at: str,
    lineage: tuple[LineageReference, ...],
    created_at_ms: int,
    valid_for_ms: int,
    maximum_evidence_age_ms: int,
    seed: int,
) -> TrustedPolicyPromotionCapsule:
    """Build a sealed capsule after independently reading every local artifact."""

    if not 0 <= created_at_ms <= _U64_MAX or not 0 <= seed <= _U64_MAX:
        raise ValueError("capsule timestamps and seed must fit unsigned 64-bit values")
    if not 0 < valid_for_ms <= _MAX_VALIDITY_MS:
        raise ValueError("capsule validity must be positive and no longer than 24 hours")
    if not 0 < maximum_evidence_age_ms <= _MAX_VALIDITY_MS:
        raise ValueError("maximum evidence age must be positive and bounded")
    if created_at_ms + valid_for_ms > _U64_MAX:
        raise ValueError("capsule expiry exceeds the unsigned 64-bit range")
    evidence_by_gate = {item.gate: item for item in gate_evidence}
    if len(gate_evidence) != 8 or len(evidence_by_gate) != 8 or set(evidence_by_gate) != _ALL_GATES:
        raise PromotionCapsuleValidationError("trusted promotion requires all eight unique gates")

    record = registry.promotion(transaction_id)
    if record.champion_policy_epoch_id != _policy_epoch_id(from_policy_epoch):
        raise PromotionCapsuleValidationError("canonical source policy does not match the registry")
    if record.candidate_policy_epoch_id != _policy_epoch_id(to_policy_epoch):
        raise PromotionCapsuleValidationError("canonical target policy does not match the registry")

    artifacts, payloads = _seal_artifacts(artifact_root, artifact_sources)
    artifact_by_gate = {item.gate: item for item in artifacts}
    for gate, evidence in evidence_by_gate.items():
        if evidence.artifact_hash != artifact_by_gate[gate].artifact_hash:
            raise PromotionCapsuleValidationError(
                f"gate {gate!r} hash does not match the re-read local artifact"
            )

    parsed_compatibility = _parse_compatibility(payloads["compatibility"], continuum_artifact_kind)
    compatibility = _compatibility_binding(parsed_compatibility)
    pointers = tuple(
        EvidencePointer(
            uri=_artifact_uri(artifact.relative_path),
            digest=Digest(value=artifact.artifact_hash),
            media_type=artifact.media_type,
            captured_at=artifact.captured_at,
        )
        for artifact in artifacts
    )
    ir_capsule = PolicyPromotionCapsule(
        promotion_id=promotion_id,
        transaction_id=transaction_id,
        from_policy_epoch=from_policy_epoch,
        to_policy_epoch=to_policy_epoch,
        decision=PromotionDecision.PROMOTE,
        evaluation_evidence=pointers,
        approved_by=approved_by,
        promoted_at=promoted_at,
        lineage=lineage,
    )
    body: dict[str, Any] = {
        "tenant_id": registry.tenant_id,
        "ir_capsule": ir_capsule,
        "ir_digest": canonical_hash(ir_capsule),
        "deployment": record.deployment,
        "registry_champion_policy_epoch_id": record.champion_policy_epoch_id,
        "registry_candidate_policy_epoch_id": record.candidate_policy_epoch_id,
        "registry_evidence_hash": record.evidence_hash,
        "gate_evidence": tuple(sorted(gate_evidence, key=lambda item: item.gate)),
        "artifacts": artifacts,
        "continuum_compatibility": compatibility,
        "created_at_ms": created_at_ms,
        "expires_at_ms": created_at_ms + valid_for_ms,
        "maximum_evidence_age_ms": maximum_evidence_age_ms,
        "seed": seed,
    }
    draft = TrustedPolicyPromotionCapsule.model_construct(capsule_digest="0" * 64, **body)
    digest = canonical_hash(draft.model_dump(mode="json", exclude={"capsule_digest"}))
    trusted = TrustedPolicyPromotionCapsule(capsule_digest=digest, **body)
    _validate_registry_binding(registry, record, evidence=gate_evidence)
    return trusted


def _coerce_capsule(capsule: CapsuleInput) -> TrustedPolicyPromotionCapsule:
    if isinstance(capsule, TrustedPolicyPromotionCapsule):
        return TrustedPolicyPromotionCapsule.model_validate(capsule.model_dump(), strict=True)
    if isinstance(capsule, (str, bytes)):
        return TrustedPolicyPromotionCapsule.model_validate_json(capsule, strict=True)
    serialized = json.dumps(dict(capsule), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return TrustedPolicyPromotionCapsule.model_validate_json(serialized, strict=True)


def validate_policy_promotion_capsule(
    capsule: CapsuleInput,
    *,
    registry: PolicyRegistry,
    artifact_root: Path,
    validated_at_ms: int,
) -> PromotionCapsuleValidation:
    """Re-hash artifacts and recheck registry, freshness, IR, and Continuum evidence."""

    if not 0 <= validated_at_ms <= _U64_MAX:
        raise ValueError("validation time must fit an unsigned 64-bit value")
    trusted = _coerce_capsule(capsule)
    if trusted.tenant_id != registry.tenant_id:
        raise PromotionCapsuleValidationError(
            "trusted promotion capsule belongs to a different tenant"
        )
    if validated_at_ms < trusted.created_at_ms:
        raise PromotionCapsuleValidationError("promotion capsule is not yet valid")
    if validated_at_ms >= trusted.expires_at_ms:
        raise PromotionCapsuleValidationError("promotion capsule is stale")
    if any(
        validated_at_ms - artifact.captured_at_ms > trusted.maximum_evidence_age_ms
        for artifact in trusted.artifacts
    ):
        raise PromotionCapsuleValidationError("promotion evidence is stale")

    record = registry.promotion(trusted.ir_capsule.transaction_id)
    _validate_registry_binding(registry, record, evidence=trusted.gate_evidence)
    if (
        record.deployment != trusted.deployment
        or record.champion_policy_epoch_id != trusted.registry_champion_policy_epoch_id
        or record.candidate_policy_epoch_id != trusted.registry_candidate_policy_epoch_id
        or record.evidence_hash != trusted.registry_evidence_hash
    ):
        raise PromotionCapsuleValidationError("registry promotion binding changed after sealing")

    sources = tuple(
        PromotionArtifactSource(
            gate=artifact.gate,
            relative_path=artifact.relative_path,
            media_type=artifact.media_type,
            captured_at=artifact.captured_at,
            captured_at_ms=artifact.captured_at_ms,
        )
        for artifact in trusted.artifacts
    )
    rehashed, payloads = _seal_artifacts(artifact_root, sources)
    if rehashed != trusted.artifacts:
        raise PromotionCapsuleValidationError("promotion artifact content or metadata was tampered")
    parsed = _parse_compatibility(
        payloads["compatibility"], trusted.continuum_compatibility.artifact_kind
    )
    if _compatibility_binding(parsed) != trusted.continuum_compatibility:
        raise PromotionCapsuleValidationError("Continuum compatibility evidence was tampered")

    return PromotionCapsuleValidation(
        capsule_digest=trusted.capsule_digest,
        tenant_id=trusted.tenant_id,
        transaction_id=trusted.ir_capsule.transaction_id,
        deployment=trusted.deployment,
        registry_state=cast(
            Literal[PromotionState.CANARY_PASSED, PromotionState.ACTIVE], record.state
        ),
        compatibility_class=trusted.continuum_compatibility.compatibility_class,
        rehashed_artifacts=rehashed,
        validated_at_ms=validated_at_ms,
    )


__all__ = [
    "ContinuumCompatibilityBinding",
    "PromotionArtifactReference",
    "PromotionArtifactSource",
    "PromotionCapsuleValidation",
    "PromotionCapsuleValidationError",
    "TrustedPolicyPromotionCapsule",
    "build_policy_promotion_capsule",
    "validate_policy_promotion_capsule",
]
