from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Literal, TypeAlias

import pytest

from sloforge.helix.policy import DeterministicPolicy
from sloforge.helix.promotion import (
    CompatibilityClass,
    GateEvidence,
    PolicyRegistry,
    PromotionState,
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


def _policy(epoch: str, logits: tuple[float, ...]) -> DeterministicPolicy:
    return DeterministicPolicy(
        policy_epoch_id=epoch,
        actions=("wrong", "correct"),
        logits=logits,
    )


def _gate(gate: GateName, *, passed: bool = True) -> GateEvidence:
    value = 0.0 if passed else 1.0
    return GateEvidence(
        gate=gate,
        evidence_id=f"evidence-{gate}",
        artifact_hash=sha256(gate.encode()).hexdigest(),
        passed=passed,
        sample_count=20,
        measured_value=value,
        threshold=0.0,
        comparator="le",
        deterministic_seed=41,
        detail=f"actual {gate} evaluation",
    )


def _pre_gates() -> tuple[GateEvidence, ...]:
    gate_names: tuple[GateName, ...] = (
        "lineage",
        "reward_integrity",
        "quality",
        "safety",
        "serving",
        "compatibility",
    )
    return tuple(_gate(gate) for gate in gate_names)


def _registry(tmp_path: Path) -> PolicyRegistry:
    registry = PolicyRegistry(tmp_path / "registry.sqlite")
    champion = _policy("champion-1", (2.0, 0.0))
    challenger = _policy("challenger-1", (0.0, 2.0))
    registry.register_policy(
        champion, parent_policy_epoch_id=None, status="champion", created_at_ms=1
    )
    registry.register_policy(
        challenger,
        parent_policy_epoch_id=champion.policy_epoch_id,
        status="challenger",
        created_at_ms=2,
    )
    registry.create_deployment("coding-agent-prod", champion.policy_epoch_id)
    return registry


def _pass_shadow_canary(registry: PolicyRegistry) -> None:
    registry.start_shadow("tx-1", observed_at_ms=20)
    registry.finish_shadow("tx-1", _gate("shadow"), observed_at_ms=30)
    registry.start_canary("tx-1", observed_at_ms=40)
    registry.finish_canary("tx-1", _gate("canary"), observed_at_ms=50)


def test_atomic_promotion_routes_new_session_and_keeps_incompatible_session_pinned(
    tmp_path: Path,
) -> None:
    with _registry(tmp_path) as registry:
        old = registry.open_session(
            deployment="coding-agent-prod", session_id="active-old", opened_at_ms=3
        )
        pinned = registry.classify_session(old.session_id, CompatibilityClass.INCOMPATIBLE)
        assert pinned.pinned
        registry.create_promotion(
            transaction_id="tx-1",
            deployment="coding-agent-prod",
            candidate_policy_epoch_id="challenger-1",
            evidence=_pre_gates(),
            observed_at_ms=10,
        )
        _pass_shadow_canary(registry)
        active = registry.promote("tx-1", observed_at_ms=60)
        assert active.state is PromotionState.ACTIVE
        assert registry.session("active-old").policy_epoch_id == "champion-1"
        assert registry.session("active-old").pinned
        new = registry.open_session(
            deployment="coding-agent-prod", session_id="new-session", opened_at_ms=70
        )
        assert new.policy_epoch_id == "challenger-1"
        rolled_back = registry.rollback(
            "tx-1", reason="post-promotion quality regression", observed_at_ms=80
        )
        assert rolled_back.state is PromotionState.ROLLED_BACK
        after = registry.open_session(
            deployment="coding-agent-prod", session_id="after-rollback", opened_at_ms=90
        )
        assert after.policy_epoch_id == "champion-1"


def test_gate_failure_rejects_without_changing_champion(tmp_path: Path) -> None:
    gate_names: tuple[GateName, ...] = (
        "lineage",
        "reward_integrity",
        "quality",
        "safety",
        "serving",
        "compatibility",
    )
    with _registry(tmp_path) as registry:
        gates = tuple(_gate(item, passed=item != "serving") for item in gate_names)
        rejected = registry.create_promotion(
            transaction_id="tx-bad",
            deployment="coding-agent-prod",
            candidate_policy_epoch_id="challenger-1",
            evidence=gates,
            observed_at_ms=10,
        )
        assert rejected.state is PromotionState.REJECTED
        assert registry.champion("coding-agent-prod").policy_epoch_id == "champion-1"
        with pytest.raises(ValueError, match="passed canary"):
            registry.promote("tx-bad", observed_at_ms=20)


def test_injected_partial_pointer_update_rolls_back_sql_transaction(
    tmp_path: Path,
) -> None:
    with _registry(tmp_path) as registry:
        registry.create_promotion(
            transaction_id="tx-1",
            deployment="coding-agent-prod",
            candidate_policy_epoch_id="challenger-1",
            evidence=_pre_gates(),
            observed_at_ms=10,
        )
        _pass_shadow_canary(registry)
        with pytest.raises(RuntimeError, match="injected fault"):
            registry.promote("tx-1", observed_at_ms=60, fault_after_pointer_update=True)
        assert registry.champion("coding-agent-prod").policy_epoch_id == "champion-1"
        assert registry.promotion("tx-1").state is PromotionState.CANARY_PASSED
        assert registry.promote("tx-1", observed_at_ms=70).state is PromotionState.ACTIVE


def test_registry_detects_policy_identity_reuse_and_promotion_evidence_tampering(
    tmp_path: Path,
) -> None:
    with _registry(tmp_path) as registry:
        with pytest.raises(ValueError, match="reused"):
            registry.register_policy(
                _policy("challenger-1", (3.0, -3.0)),
                parent_policy_epoch_id="champion-1",
                status="challenger",
                created_at_ms=3,
            )
        registry.create_promotion(
            transaction_id="tx-1",
            deployment="coding-agent-prod",
            candidate_policy_epoch_id="challenger-1",
            evidence=_pre_gates(),
            observed_at_ms=10,
        )
        registry._connection.execute(
            "UPDATE promotions SET evidence_hash=? WHERE transaction_id='tx-1'",
            ("0" * 64,),
        )
        with pytest.raises(ValueError, match="integrity"):
            registry.promotion("tx-1")


def test_registry_is_durably_tenant_bound_and_persists_runtime_gate_evidence(
    tmp_path: Path,
) -> None:
    database = tmp_path / "tenant-registry.sqlite"
    registry = PolicyRegistry(database, tenant_id="tenant-a")
    champion = _policy("champion-a", (2.0, 0.0))
    challenger = _policy("challenger-a", (0.0, 2.0))
    registry.register_policy(
        champion, parent_policy_epoch_id=None, status="champion", created_at_ms=1
    )
    registry.register_policy(
        challenger,
        parent_policy_epoch_id=champion.policy_epoch_id,
        status="challenger",
        created_at_ms=2,
    )
    registry.create_deployment("prod-a", champion.policy_epoch_id)
    tenant_gates = tuple(item.model_copy(update={"tenant_id": "tenant-a"}) for item in _pre_gates())
    with pytest.raises(PermissionError, match="cross tenant"):
        registry.create_promotion(
            transaction_id="wrong-tenant",
            deployment="prod-a",
            candidate_policy_epoch_id=challenger.policy_epoch_id,
            evidence=_pre_gates(),
            observed_at_ms=10,
        )
    record = registry.create_promotion(
        transaction_id="tenant-promotion",
        deployment="prod-a",
        candidate_policy_epoch_id=challenger.policy_epoch_id,
        evidence=tenant_gates,
        observed_at_ms=10,
    )
    initial_hash = record.evidence_hash
    registry.start_shadow("tenant-promotion", observed_at_ms=20)
    shadow = _gate("shadow").model_copy(update={"tenant_id": "tenant-a"})
    after_shadow = registry.finish_shadow("tenant-promotion", shadow, observed_at_ms=30)
    assert after_shadow.evidence_hash != initial_hash
    registry.start_canary("tenant-promotion", observed_at_ms=40)
    canary = _gate("canary").model_copy(update={"tenant_id": "tenant-a"})
    after_canary = registry.finish_canary("tenant-promotion", canary, observed_at_ms=50)
    evidence_json = registry._connection.execute(
        "SELECT evidence_json FROM promotions WHERE transaction_id='tenant-promotion'"
    ).fetchone()[0]
    assert {item["gate"] for item in json.loads(evidence_json)} == {
        "lineage",
        "reward_integrity",
        "quality",
        "safety",
        "serving",
        "compatibility",
        "shadow",
        "canary",
    }
    assert after_canary.tenant_id == "tenant-a"
    registry.close()
    with pytest.raises(PermissionError, match="different tenant"):
        PolicyRegistry(database, tenant_id="tenant-b")
