from __future__ import annotations

from hashlib import sha256

import pytest

from sloforge.helix.replay import (
    ComparisonScope,
    ExactReplayIdentityMismatch,
    ReplayEvent,
    ReplayFrame,
    ReplayIdentity,
    ReplayMode,
    ReplayToken,
    ReplayTolerances,
    ResourceObservation,
    build_replay_trace,
    compare_replay,
    replay_and_compare,
)


def _hash(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def _event(
    event_id: str,
    kind: str,
    payload: str,
    *,
    semantic: str | None = None,
    parent: str | None = None,
) -> ReplayEvent:
    return ReplayEvent(
        event_id=event_id,
        kind=kind,
        payload_digest=_hash(payload),
        semantic_digest=_hash(semantic or payload),
        causal_parent_id=parent,
    )


def _identity(**updates: object) -> ReplayIdentity:
    identity = ReplayIdentity(
        policy_epoch_id="policy-a",
        policy_digest=_hash("policy-a"),
        runtime_name="continuum-reference-token-major",
        runtime_version="1.0.0",
        runtime_build_hash=_hash("runtime-build"),
        model_hash=_hash("model-a"),
        model_state_digest=_hash("model-state-a"),
        environment_capsule_id="environment-a",
        environment_state_digest=_hash("environment-a"),
        rng_algorithm="continuum-counter-v1",
        rng_seed=97,
        rng_counter=4,
        tool_contract_hash=_hash("tools-v1"),
    )
    return identity.model_copy(update=updates)


def _trace(
    *,
    identity: ReplayIdentity | None = None,
    action_payload: str = "open:a.txt",
    action_semantic: str = "open-file",
    token_id: int = 17,
    environment_payload: str = "a.txt:hello",
    environment_semantic: str = "file-read-success",
    reward: float = 1.0,
    outcome: str = "completed",
    cpu: float = 2.0,
) -> object:
    action = _event("action-0", "tool_call", action_payload, semantic=action_semantic)
    environment = _event(
        "environment-0",
        "tool_result",
        environment_payload,
        semantic=environment_semantic,
        parent="action-0",
    )
    frame = ReplayFrame(
        action_index=0,
        action=action,
        model_tokens=(ReplayToken(token_index=0, token_id=token_id),),
        environment_events=(environment,),
        reward=reward,
        outcome=outcome,
        resources=(ResourceObservation(name="cpu_ms", value=cpu, unit="ms"),),
    )
    return build_replay_trace(
        branch_point_id="a" * 64,
        identity=identity or _identity(),
        frames=(frame,),
        terminal_outcome=outcome,
    )


def test_exact_replay_matches_and_fails_closed_on_every_identity_contract() -> None:
    expected = _trace()
    evidence = replay_and_compare(
        expected,
        lambda trace, _mode, _seed: trace,
        mode=ReplayMode.EXACT,
        seed=97,
    )
    assert evidence.matched
    assert evidence.exact_identity_verified
    assert not evidence.transcript_establishes_state_equivalence

    fields = (
        ("policy_digest", _hash("policy-b")),
        ("runtime_build_hash", _hash("runtime-other")),
        ("model_hash", _hash("model-b")),
        ("environment_state_digest", _hash("environment-b")),
        ("rng_seed", 98),
        ("tool_contract_hash", _hash("tools-v2")),
    )
    for field, value in fields:
        observed = _trace(identity=_identity(**{field: value}))
        with pytest.raises(ExactReplayIdentityMismatch, match=field):
            compare_replay(expected, observed, mode=ReplayMode.EXACT)


def test_replay_records_first_divergence_for_every_evidence_axis() -> None:
    expected = _trace()
    observed = _trace(
        action_payload="open:b.txt",
        action_semantic="different-action",
        token_id=18,
        environment_payload="b.txt:missing",
        environment_semantic="file-read-failure",
        reward=-1.0,
        outcome="failed",
        cpu=7.0,
    )
    evidence = compare_replay(expected, observed, mode=ReplayMode.EXACT)
    assert not evidence.matched
    assert evidence.first_token_divergence is not None
    assert evidence.first_action_divergence is not None
    assert evidence.first_environment_divergence is not None
    assert evidence.first_reward_divergence is not None
    assert evidence.first_outcome_divergence is not None
    assert evidence.first_resource_divergence is not None


def test_causal_and_semantic_modes_are_explicit_and_bounded() -> None:
    expected = _trace()
    causal = _trace(action_payload="open:./a.txt", environment_payload="bytes:hello")
    causal_evidence = compare_replay(
        expected,
        causal,
        mode=ReplayMode.CAUSAL,
        tolerances=ReplayTolerances(reward_absolute=0.01, resource_absolute=0.01),
    )
    assert causal_evidence.matched

    different_semantics = _trace(
        action_payload="delete:a.txt",
        action_semantic="delete-file",
        environment_payload="a.txt:deleted",
        environment_semantic="file-delete-success",
    )
    causal_difference = compare_replay(
        expected,
        different_semantics,
        mode=ReplayMode.CAUSAL,
    )
    assert not causal_difference.matched
    assert causal_difference.first_action_divergence is not None
    assert causal_difference.first_environment_divergence is not None

    semantic = _trace(action_payload="read:a.txt", environment_payload="content:hello")
    semantic_evidence = compare_replay(expected, semantic, mode=ReplayMode.SEMANTIC)
    assert semantic_evidence.matched
    with pytest.raises(ValueError, match="finite and non-negative"):
        ReplayTolerances(reward_absolute=float("nan"))


def test_transcript_environment_model_and_joint_evidence_are_not_conflated() -> None:
    expected = _trace()
    changed_environment = _trace(
        environment_payload="a.txt:different bytes",
        environment_semantic="different state",
    )
    transcript = compare_replay(
        expected,
        changed_environment,
        mode=ReplayMode.EXACT,
        scope=ComparisonScope.TRANSCRIPT,
    )
    environment = compare_replay(
        expected,
        changed_environment,
        mode=ReplayMode.EXACT,
        scope=ComparisonScope.ENVIRONMENT_ONLY,
    )
    model = compare_replay(
        expected,
        changed_environment,
        mode=ReplayMode.EXACT,
        scope=ComparisonScope.MODEL_ONLY,
    )
    joint = compare_replay(
        expected,
        changed_environment,
        mode=ReplayMode.EXACT,
        scope=ComparisonScope.JOINT,
    )
    assert transcript.matched
    assert transcript.transcript_evidence_digest is not None
    assert transcript.environment_state_evidence_digest is None
    assert not environment.matched
    assert environment.environment_state_evidence_digest is not None
    assert model.matched
    assert model.model_state_evidence_digest is not None
    assert not joint.matched
    assert joint.joint_evidence_digest is not None
    assert not joint.transcript_establishes_state_equivalence

    changed_model_state = _trace(identity=_identity(model_state_digest=_hash("model-state-b")))
    model_state = compare_replay(
        expected,
        changed_model_state,
        mode=ReplayMode.CAUSAL,
        scope=ComparisonScope.MODEL_ONLY,
    )
    assert not model_state.matched
    assert model_state.first_token_divergence is None
    assert model_state.model_state_equal is False

    raw = model_state.model_dump()
    raw["matched"] = True
    with pytest.raises(ValueError, match="matched flag"):
        type(model_state).model_validate(raw, strict=True)
