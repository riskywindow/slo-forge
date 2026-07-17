from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, TypeAlias

import pytest
from pydantic import ValidationError

from sloforge.continuum.compatibility import CompatibilityDecision
from sloforge.continuum.ir import ExactnessClass
from sloforge.helix.ir import Digest, LineageReference, LineageRelation, PolicyEpoch
from sloforge.helix.policy import DeterministicPolicy
from sloforge.helix.promotion import (
    GateEvidence,
    PolicyRegistry,
    PromotionArtifactSource,
    PromotionCapsuleValidationError,
    PromotionState,
    TrustedPolicyPromotionCapsule,
    build_policy_promotion_capsule,
    validate_policy_promotion_capsule,
)

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

_GATES: tuple[GateName, ...] = (
    "lineage",
    "reward_integrity",
    "quality",
    "safety",
    "serving",
    "compatibility",
    "shadow",
    "canary",
)


def _hash(value: str | bytes) -> str:
    encoded = value.encode() if isinstance(value, str) else value
    return hashlib.sha256(encoded).hexdigest()


def _policy(epoch: int) -> DeterministicPolicy:
    return DeterministicPolicy(
        policy_epoch_id=f"policy-main@{epoch}",
        actions=("reject", "accept"),
        logits=(float(epoch), -float(epoch)),
    )


def _lineage(artifact_id: str, label: str) -> LineageReference:
    return LineageReference(
        artifact_id=artifact_id,
        artifact_kind="sloforge.test/promotion",
        relation=LineageRelation.DERIVED_FROM,
        digest=Digest(value=_hash(label)),
    )


def _policy_epochs() -> tuple[PolicyEpoch, PolicyEpoch]:
    source_digest = Digest(value=_hash("policy-source"))
    target_digest = Digest(value=_hash("policy-target"))
    source = PolicyEpoch(
        policy_id="policy-main",
        epoch=1,
        policy_digest=source_digest,
        parent_epoch=0,
        parent_policy_digest=Digest(value=_hash("policy-root")),
        created_at="2026-08-03T10:00:00Z",
        lineage=(_lineage("policy-main@0", "root-lineage"),),
    )
    target = PolicyEpoch(
        policy_id="policy-main",
        epoch=2,
        policy_digest=target_digest,
        parent_epoch=1,
        parent_policy_digest=source_digest,
        training_transaction_id="promotion-tx",
        created_at="2026-08-03T11:00:00Z",
        lineage=(_lineage("policy-main@1", "source-lineage"),),
    )
    return source, target


def _safe_compatibility() -> CompatibilityDecision:
    return CompatibilityDecision(
        compatibility_class=ExactnessClass.EXACT_SEMANTIC,
        safe=True,
        reasons=(),
        rejected_classes=(),
        required_conversion=(),
        required_recomputation=(),
        unsupported_state=(),
        quality_implications=(),
        verification_obligations=(),
        migration_restrictions=(),
    )


def _artifacts(
    root: Path,
    compatibility: CompatibilityDecision,
) -> tuple[tuple[PromotionArtifactSource, ...], tuple[GateEvidence, ...]]:
    root.mkdir()
    sources: list[PromotionArtifactSource] = []
    evidence: list[GateEvidence] = []
    for gate in _GATES:
        relative_path = f"{gate}.json"
        payload = (
            compatibility.model_dump_json().encode()
            if gate == "compatibility"
            else json.dumps(
                {"gate": gate, "samples": 32, "passed": True},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        root.joinpath(relative_path).write_bytes(payload)
        digest = _hash(payload)
        sources.append(
            PromotionArtifactSource(
                gate=gate,
                relative_path=relative_path,
                media_type="application/json",
                captured_at="2026-08-03T12:00:00Z",
                captured_at_ms=100,
            )
        )
        evidence.append(
            GateEvidence(
                gate=gate,
                evidence_id=f"evidence-{gate}",
                artifact_hash=digest,
                passed=True,
                sample_count=32,
                measured_value=0.0,
                threshold=0.0,
                comparator="le",
                deterministic_seed=17,
                detail=f"local {gate} evaluation",
            )
        )
    return tuple(sources), tuple(evidence)


def _registry(root: Path, evidence: tuple[GateEvidence, ...]) -> PolicyRegistry:
    root.mkdir(parents=True, exist_ok=True)
    registry = PolicyRegistry(root / "registry.sqlite")
    source = _policy(1)
    target = _policy(2)
    registry.register_policy(
        source,
        parent_policy_epoch_id=None,
        status="champion",
        created_at_ms=1,
    )
    registry.register_policy(
        target,
        parent_policy_epoch_id=source.policy_epoch_id,
        status="challenger",
        created_at_ms=2,
    )
    registry.create_deployment("coding-agent-prod", source.policy_epoch_id)
    pre_gates = tuple(item for item in evidence if item.gate not in {"shadow", "canary"})
    registry.create_promotion(
        transaction_id="promotion-tx",
        deployment="coding-agent-prod",
        candidate_policy_epoch_id=target.policy_epoch_id,
        evidence=pre_gates,
        observed_at_ms=110,
    )
    shadow = next(item for item in evidence if item.gate == "shadow")
    canary = next(item for item in evidence if item.gate == "canary")
    registry.start_shadow("promotion-tx", observed_at_ms=120)
    registry.finish_shadow("promotion-tx", shadow, observed_at_ms=130)
    registry.start_canary("promotion-tx", observed_at_ms=140)
    registry.finish_canary("promotion-tx", canary, observed_at_ms=150)
    return registry


def _build(
    registry: PolicyRegistry,
    artifact_root: Path,
    sources: tuple[PromotionArtifactSource, ...],
    evidence: tuple[GateEvidence, ...],
    *,
    maximum_evidence_age_ms: int = 1_000,
) -> TrustedPolicyPromotionCapsule:
    source, target = _policy_epochs()
    return build_policy_promotion_capsule(
        registry=registry,
        transaction_id="promotion-tx",
        promotion_id="promotion-capsule-1",
        from_policy_epoch=source,
        to_policy_epoch=target,
        gate_evidence=evidence,
        artifact_root=artifact_root,
        artifact_sources=sources,
        continuum_artifact_kind="decision",
        approved_by="helix-promotion-test",
        promoted_at="2026-08-03T12:00:01Z",
        lineage=(
            _lineage("promotion-tx", "transaction"),
            _lineage("policy-main@1", "source-policy"),
            _lineage("policy-main@2", "target-policy"),
        ),
        created_at_ms=200,
        valid_for_ms=1_000,
        maximum_evidence_age_ms=maximum_evidence_age_ms,
        seed=991,
    )


def test_builder_seals_canonical_ir_registry_and_all_eight_evidence_gates(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "evidence"
    sources, evidence = _artifacts(artifact_root, _safe_compatibility())
    with _registry(tmp_path, evidence) as registry:
        capsule = _build(registry, artifact_root, sources, evidence)
        validation = validate_policy_promotion_capsule(
            capsule,
            registry=registry,
            artifact_root=artifact_root,
            validated_at_ms=250,
        )

        assert capsule.ir_capsule.transaction_id == "promotion-tx"
        assert capsule.ir_capsule.decision.value == "promote"
        assert len(capsule.ir_capsule.evaluation_evidence) == 8
        assert {item.gate for item in capsule.gate_evidence} == set(_GATES)
        assert validation.eligible_for_promotion
        assert validation.registry_state is PromotionState.CANARY_PASSED
        assert validation.compatibility_class is ExactnessClass.EXACT_SEMANTIC
        assert len(validation.rehashed_artifacts) == 8


def test_validator_rehashes_local_artifacts_and_rejects_tampering(tmp_path: Path) -> None:
    artifact_root = tmp_path / "evidence"
    sources, evidence = _artifacts(artifact_root, _safe_compatibility())
    with _registry(tmp_path, evidence) as registry:
        capsule = _build(registry, artifact_root, sources, evidence)
        artifact_root.joinpath("quality.json").write_text('{"tampered":true}')

        with pytest.raises(PromotionCapsuleValidationError, match="tampered"):
            validate_policy_promotion_capsule(
                capsule,
                registry=registry,
                artifact_root=artifact_root,
                validated_at_ms=250,
            )


def test_stale_capsule_or_stale_evaluation_evidence_is_rejected(tmp_path: Path) -> None:
    artifact_root = tmp_path / "evidence"
    sources, evidence = _artifacts(artifact_root, _safe_compatibility())
    with _registry(tmp_path, evidence) as registry:
        capsule = _build(registry, artifact_root, sources, evidence)
        with pytest.raises(PromotionCapsuleValidationError, match="capsule is stale"):
            validate_policy_promotion_capsule(
                capsule,
                registry=registry,
                artifact_root=artifact_root,
                validated_at_ms=1_200,
            )

        short_window = _build(
            registry,
            artifact_root,
            sources,
            evidence,
            maximum_evidence_age_ms=100,
        )
        with pytest.raises(PromotionCapsuleValidationError, match="evidence is stale"):
            validate_policy_promotion_capsule(
                short_window,
                registry=registry,
                artifact_root=artifact_root,
                validated_at_ms=250,
            )


def test_missing_failed_or_unsafe_evidence_cannot_build_a_capsule(tmp_path: Path) -> None:
    artifact_root = tmp_path / "evidence"
    sources, evidence = _artifacts(artifact_root, _safe_compatibility())
    with _registry(tmp_path, evidence) as registry:
        with pytest.raises(PromotionCapsuleValidationError, match="all eight"):
            _build(registry, artifact_root, sources[:-1], evidence[:-1])

        shadow = next(item for item in evidence if item.gate == "shadow")
        failed_shadow = shadow.model_copy(update={"passed": False, "measured_value": 1.0})
        failed = tuple(failed_shadow if item.gate == "shadow" else item for item in evidence)
        with pytest.raises(ValidationError, match="did not pass"):
            _build(registry, artifact_root, sources, failed)

    unsafe_root = tmp_path / "unsafe-evidence"
    unsafe = CompatibilityDecision(
        compatibility_class=ExactnessClass.INCOMPATIBLE,
        safe=False,
        reasons=(),
        rejected_classes=(),
        required_conversion=(),
        required_recomputation=(),
        unsupported_state=("attention_kv",),
        quality_implications=(),
        verification_obligations=(),
        migration_restrictions=("activation prohibited",),
    )
    unsafe_sources, unsafe_evidence = _artifacts(unsafe_root, unsafe)
    with (
        _registry(tmp_path / "unsafe-registry", unsafe_evidence) as registry,
        pytest.raises(PromotionCapsuleValidationError, match="unsafe"),
    ):
        _build(registry, unsafe_root, unsafe_sources, unsafe_evidence)


def test_capsule_rejects_serialized_tampering_and_registry_rollback(tmp_path: Path) -> None:
    artifact_root = tmp_path / "evidence"
    sources, evidence = _artifacts(artifact_root, _safe_compatibility())
    with _registry(tmp_path, evidence) as registry:
        capsule = _build(registry, artifact_root, sources, evidence)
        raw = capsule.model_dump(mode="json")
        raw["seed"] = 992
        with pytest.raises(ValidationError, match="capsule digest"):
            validate_policy_promotion_capsule(
                raw,
                registry=registry,
                artifact_root=artifact_root,
                validated_at_ms=250,
            )

        assert registry.promote("promotion-tx", observed_at_ms=300).state is PromotionState.ACTIVE
        registry.rollback(
            "promotion-tx",
            reason="post-promotion safety regression",
            observed_at_ms=350,
        )
        with pytest.raises(PromotionCapsuleValidationError, match="shadow and canary"):
            validate_policy_promotion_capsule(
                capsule,
                registry=registry,
                artifact_root=artifact_root,
                validated_at_ms=400,
            )
