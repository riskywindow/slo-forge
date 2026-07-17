from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jsonschema
import pytest
from pydantic import ValidationError

from sloforge.helix.ir import (
    ActionProvenance,
    BranchGroup,
    BranchPoint,
    CreditAssignment,
    CreditAssignmentEvidence,
    CreditSubjectKind,
    Digest,
    EnvironmentStateCapsule,
    EvidencePointer,
    LearningStateTransition,
    LearningTransaction,
    LearningTransactionState,
    LineageReference,
    LineageRelation,
    PolicyConsistency,
    PolicyEpoch,
    PolicyPromotionCapsule,
    PromotionDecision,
    RewardComponent,
    RewardEvidence,
    StalenessDisposition,
    StalenessReport,
    StateReuseMode,
    StateReuseReport,
    TokenProvenance,
    TrainingBatchManifest,
    TrainingSampleKind,
    TrainingSampleProvenance,
    TrajectoryCapsule,
    TrajectoryEvent,
    TrajectoryEventKind,
    TrajectorySegment,
    TrajectoryTerminalStatus,
    canonical_digest,
    canonical_hash,
    canonical_json,
    load_learning_transaction,
    schema_documents,
)

ROOT = Path(__file__).parents[2]
GOLDEN = ROOT / "tests/fixtures/helix/learning-transaction-v1.json"
GOLDEN_HASH = "784a1e9012ac78c8e3bec1a9bf9e7f6f113862fda04d288409d25968b1236a63"


def _digest(text: str) -> Digest:
    return Digest(value=hashlib.sha256(text.encode()).hexdigest())


def _lineage(*artifact_ids: str) -> tuple[LineageReference, ...]:
    return tuple(
        LineageReference(
            artifact_id=artifact_id,
            artifact_kind="helix.test/fixture",
            relation=LineageRelation.DERIVED_FROM,
            digest=_digest(f"lineage:{artifact_id}"),
        )
        for artifact_id in artifact_ids
    )


def _evidence(name: str) -> EvidencePointer:
    return EvidencePointer(
        uri=f"fixture://raw/{name}",
        digest=_digest(f"raw:{name}"),
        media_type="application/json",
        captured_at="2026-08-03T12:00:00Z",
    )


def _epoch(epoch: int, policy_digest: Digest, parent_digest: Digest | None) -> PolicyEpoch:
    return PolicyEpoch(
        policy_id="policy-main",
        epoch=epoch,
        policy_digest=policy_digest,
        parent_epoch=None if epoch == 0 else epoch - 1,
        parent_policy_digest=parent_digest,
        training_transaction_id=None if epoch < 2 else "learning-tx-001",
        created_at=f"2026-08-03T0{epoch}:00:00Z",
        lineage=_lineage("policy-root" if epoch == 0 else f"policy-main@{epoch - 1}"),
    )


def _trajectory(
    identifier: str,
    event_id: str,
    token_id: str,
    policy: PolicyEpoch,
) -> TrajectoryCapsule:
    event = TrajectoryEvent(
        event_id=event_id,
        event_index=0,
        kind=TrajectoryEventKind.GENERATED_TOKEN,
        policy_epoch=policy,
        payload_digest=_digest(f"payload:{event_id}"),
        source_evidence=_evidence(event_id),
    )
    token = TokenProvenance(
        token_id=token_id,
        token_index=0,
        event_id=event_id,
        token_value=17,
        policy_epoch=policy,
        behavior_log_probability=-0.2,
        sampler_seed=73129,
        raw_sample=_evidence(token_id),
    )
    return TrajectoryCapsule(
        trajectory_id=identifier,
        branch_point_id="branch-point-001",
        source_state_capsule_id="state-capsule-001",
        environment_id="env-grid-v1",
        policy_consistency=PolicyConsistency.STRICT,
        policy_epochs=(policy,),
        segments=(
            TrajectorySegment(
                segment_id=f"segment-{identifier}",
                start_event_index=0,
                end_event_index_exclusive=1,
                policy_epoch=policy,
                segment_evidence=_evidence(f"segment-{identifier}"),
            ),
        ),
        events=(event,),
        tokens=(token,),
        actions=(),
        terminal_status=TrajectoryTerminalStatus.COMPLETED,
        started_at="2026-08-03T12:00:00Z",
        completed_at="2026-08-03T12:00:01Z",
        trace_evidence=_evidence(f"trace-{identifier}"),
        lineage=_lineage("state-capsule-001", "branch-point-001", "policy-main@1"),
    )


def build_complete_transaction() -> LearningTransaction:
    root_digest = _digest("policy:root")
    source_digest = _digest("policy:source")
    candidate_digest = _digest("policy:candidate")
    root_policy = _epoch(0, root_digest, None)
    source_policy = _epoch(1, source_digest, root_policy.policy_digest)
    candidate_policy = _epoch(2, candidate_digest, source_policy.policy_digest)
    state = EnvironmentStateCapsule(
        capsule_id="state-capsule-001",
        environment_id="env-grid-v1",
        captured_at="2026-08-03T11:59:59Z",
        policy_epoch=source_policy,
        state_schema_digest=_digest("state-schema:v1"),
        state_digest=_digest("state:payload"),
        compatibility_fingerprint=_digest("state:compatibility"),
        payload_uri="fixture://state/state-capsule-001",
        payload_media_type="application/vnd.sloforge.environment-state+json",
        payload_byte_length=128,
        compatible_policy_digests=(source_digest, candidate_digest),
        lineage=_lineage("policy-main@1"),
    )
    branch_point = BranchPoint(
        branch_point_id="branch-point-001",
        source_trajectory_id="trajectory-root",
        event_index=3,
        token_index=2,
        environment_state=state,
        policy_epoch=source_policy,
        prefix_digest=_digest("prefix:root"),
        seed=73129,
        created_at="2026-08-03T12:00:00Z",
        reason="compare two deterministic continuations",
        candidate_labels=("baseline", "alternate"),
        lineage=_lineage("trajectory-root", "state-capsule-001", "policy-main@1"),
    )
    baseline = _trajectory("trajectory-baseline", "event-baseline", "token-baseline", source_policy)
    alternate = _trajectory(
        "trajectory-alternate", "event-alternate", "token-alternate", source_policy
    )
    branch_group = BranchGroup(
        group_id="branch-group-001",
        branch_point=branch_point,
        trajectories=(baseline, alternate),
        baseline_trajectory_id=baseline.trajectory_id,
        created_at="2026-08-03T12:00:02Z",
        lineage=_lineage("branch-point-001", "trajectory-baseline", "trajectory-alternate"),
    )
    reward = RewardEvidence(
        reward_evidence_id="reward-001",
        trajectory_id=baseline.trajectory_id,
        trajectory_digest=canonical_digest(baseline),
        policy_epochs=(source_policy,),
        components=(
            RewardComponent(
                component_id="reward-component-001",
                name="task_success",
                value=1.0,
                weight=1.0,
                policy_epoch=source_policy,
                event_ids=("event-baseline",),
                raw_evidence=_evidence("reward-component-001"),
            ),
        ),
        aggregate_reward=1.0,
        evaluator_digest=_digest("evaluator:v1"),
        evaluated_at="2026-08-03T12:01:00Z",
        lineage=_lineage("trajectory-baseline", "policy-main@1"),
    )
    credit = CreditAssignmentEvidence(
        credit_evidence_id="credit-001",
        trajectory_id=baseline.trajectory_id,
        trajectory_digest=canonical_digest(baseline),
        reward_evidence_id=reward.reward_evidence_id,
        reward_evidence_digest=canonical_digest(reward),
        method="single-token-return-v1",
        policy_epochs=(source_policy,),
        assignments=(
            CreditAssignment(
                assignment_id="assignment-001",
                subject_kind=CreditSubjectKind.TOKEN,
                subject_id="token-baseline",
                event_id="event-baseline",
                reward_component_id="reward-component-001",
                policy_epoch=source_policy,
                behavior_log_probability=-0.2,
                credit=1.0,
            ),
        ),
        total_credit=1.0,
        generated_at="2026-08-03T12:01:01Z",
        lineage=_lineage("trajectory-baseline", "reward-001", "policy-main@1"),
    )
    sample = TrainingSampleProvenance(
        sample_id="sample-001",
        sample_kind=TrainingSampleKind.TOKEN,
        trajectory_id=baseline.trajectory_id,
        trajectory_digest=canonical_digest(baseline),
        reward_evidence_id=reward.reward_evidence_id,
        reward_evidence_digest=canonical_digest(reward),
        credit_evidence_id=credit.credit_evidence_id,
        credit_evidence_digest=canonical_digest(credit),
        event_ids=("event-baseline",),
        token_ids=("token-baseline",),
        action_ids=(),
        behavior_policy_epoch=source_policy,
        behavior_log_probability=-0.2,
        target_policy_epoch=source_policy,
        importance_sampling_weight=1.0,
        raw_sample=_evidence("sample-001"),
        lineage=_lineage("trajectory-baseline", "reward-001", "credit-001", "policy-main@1"),
    )
    batch = TrainingBatchManifest(
        batch_id="batch-001",
        policy_consistency=PolicyConsistency.STRICT,
        learner_policy_epoch=source_policy,
        samples=(sample,),
        created_at="2026-08-03T12:02:00Z",
        lineage=_lineage("sample-001", "policy-main@1"),
    )
    staleness = StalenessReport(
        report_id="staleness-001",
        sample_id=sample.sample_id,
        trajectory_id=baseline.trajectory_id,
        behavior_policy_epoch=source_policy,
        learner_policy_epoch=source_policy,
        epoch_lag=0,
        maximum_allowed_lag=1,
        stale=False,
        disposition=StalenessDisposition.ACCEPT,
        assessed_at="2026-08-03T12:02:01Z",
        lineage=_lineage("sample-001", "trajectory-baseline", "policy-main@1"),
    )
    reuse = StateReuseReport(
        report_id="state-reuse-001",
        source_capsule=state,
        target_environment_id="env-grid-v1",
        target_policy_epoch=candidate_policy,
        target_compatibility_fingerprint=state.compatibility_fingerprint,
        mode=StateReuseMode.EXACT,
        compatible=True,
        reused=True,
        reason="state ABI and target policy compatibility are exact",
        assessed_at="2026-08-03T12:03:00Z",
        lineage=_lineage("state-capsule-001", "policy-main@2"),
    )
    states = (
        LearningTransactionState.CREATED,
        LearningTransactionState.EVIDENCE_VALIDATED,
        LearningTransactionState.BATCH_ASSEMBLED,
        LearningTransactionState.TRAINED,
        LearningTransactionState.EVALUATED,
        LearningTransactionState.COMMITTED,
    )
    transitions = tuple(
        LearningStateTransition(
            sequence=index + 1,
            from_state=None if index == 0 else states[index - 1],
            to_state=state_value,
            transitioned_at=f"2026-08-03T12:0{index}:00Z",
            actor="helix-test",
            reason=f"advance to {state_value.value}",
            evidence_digests=() if index == 0 else (_digest(f"transition:{index}"),),
        )
        for index, state_value in enumerate(states)
    )
    promotion = PolicyPromotionCapsule(
        promotion_id="promotion-001",
        transaction_id="learning-tx-001",
        from_policy_epoch=source_policy,
        to_policy_epoch=candidate_policy,
        decision=PromotionDecision.PROMOTE,
        evaluation_evidence=(_evidence("promotion-evaluation"),),
        approved_by="helix-test",
        promoted_at="2026-08-03T12:05:00Z",
        lineage=_lineage("learning-tx-001", "policy-main@1", "policy-main@2"),
    )
    return LearningTransaction(
        transaction_id="learning-tx-001",
        state=LearningTransactionState.COMMITTED,
        previous_state=LearningTransactionState.EVALUATED,
        transitioned_at="2026-08-03T12:05:00Z",
        transition_sequence=6,
        transitions=transitions,
        source_policy_epoch=source_policy,
        candidate_policy_epoch=candidate_policy,
        branch_group=branch_group,
        reward_evidence=(reward,),
        credit_assignment_evidence=(credit,),
        staleness_reports=(staleness,),
        state_reuse_reports=(reuse,),
        training_batch=batch,
        promotion=promotion,
        created_at="2026-08-03T12:00:00Z",
        lineage=_lineage("branch-group-001", "batch-001", "reward-001", "credit-001"),
    )


def test_complete_learning_transaction_matches_golden() -> None:
    transaction = build_complete_transaction()
    assert canonical_hash(transaction) == GOLDEN_HASH
    assert canonical_json(transaction) == canonical_json(load_learning_transaction(GOLDEN))


def test_tampering_unknown_fields_and_missing_behavior_evidence_are_rejected() -> None:
    raw = json.loads(GOLDEN.read_text())
    raw["branch_group"]["trajectories"][0]["tokens"][0]["behavior_log_probability"] = -0.3
    with pytest.raises(ValidationError, match="trajectory digest"):
        LearningTransaction.model_validate_json(json.dumps(raw))

    raw = json.loads(GOLDEN.read_text())
    raw["branch_group"]["trajectories"][0]["tokens"][0]["surprise"] = True
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        LearningTransaction.model_validate_json(json.dumps(raw))

    raw = json.loads(GOLDEN.read_text())
    del raw["training_batch"]["samples"][0]["behavior_log_probability"]
    with pytest.raises(ValidationError, match="Field required"):
        LearningTransaction.model_validate_json(json.dumps(raw))


def test_policy_segments_and_incompatible_reuse_cannot_be_silent() -> None:
    raw = json.loads(GOLDEN.read_text())
    trajectory = raw["branch_group"]["trajectories"][0]
    trajectory["policy_consistency"] = "strict"
    second_event = dict(trajectory["events"][0])
    second_event["event_id"] = "event-baseline-terminal"
    second_event["event_index"] = 1
    second_event["kind"] = "terminal"
    trajectory["events"].append(second_event)
    second_segment = dict(trajectory["segments"][0])
    second_segment["segment_id"] = "segment-baseline-terminal"
    second_segment["start_event_index"] = 1
    second_segment["end_event_index_exclusive"] = 2
    trajectory["segments"].append(second_segment)
    with pytest.raises(ValidationError, match="strict trajectories"):
        LearningTransaction.model_validate_json(json.dumps(raw))

    raw = json.loads(GOLDEN.read_text())
    report = raw["state_reuse_reports"][0]
    report["target_compatibility_fingerprint"]["value"] = "f" * 64
    with pytest.raises(ValidationError, match="exact state reuse"):
        LearningTransaction.model_validate_json(json.dumps(raw))


def test_checked_in_schemas_match_models_and_validate_golden() -> None:
    for name, generated in schema_documents():
        checked_in = json.loads((ROOT / "schemas/helix" / name).read_text())
        assert checked_in == generated
    schema = json.loads((ROOT / "schemas/helix/learning-transaction-v1.schema.json").read_text())
    jsonschema.Draft202012Validator(schema).validate(json.loads(GOLDEN.read_text()))


def test_canonical_json_is_order_independent_and_rejects_nonfinite_values() -> None:
    left = {"雪": [1, 2], "a": {"z": -0.0, "b": 1e-7}}
    right = {"a": {"b": 1e-7, "z": -0.0}, "雪": [1, 2]}
    assert canonical_json(left) == canonical_json(right)
    assert canonical_hash(left) == canonical_hash(right)
    with pytest.raises(ValueError, match="Out of range float values"):
        canonical_json({"bad": float("nan")})


def test_action_policy_epoch_and_behavior_log_probability_are_mandatory() -> None:
    transaction = build_complete_transaction()
    action = ActionProvenance(
        action_id="action-001",
        action_index=0,
        event_id="event-action",
        action_type="move",
        policy_epoch=transaction.source_policy_epoch,
        behavior_log_probability=-0.4,
        arguments_digest=_digest("action:args"),
        raw_sample=_evidence("action-001"),
    )
    assert action.behavior_log_probability == -0.4
    raw = action.model_dump(mode="json")
    del raw["policy_epoch"]
    with pytest.raises(ValidationError, match="Field required"):
        ActionProvenance.model_validate(raw)
