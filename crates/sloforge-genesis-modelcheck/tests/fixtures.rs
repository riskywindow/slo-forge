use sloforge_genesis_modelcheck::{
    CheckStatus, InvariantId, ModelCheckRequest, PartialStateRead, QueueOverflow, StateOwnership,
    TokenDelivery, check, replay_counterexample, validate_result,
};

fn load_fixture(contents: &str) -> ModelCheckRequest {
    serde_json::from_str(contents)
        .unwrap_or_else(|error| panic!("model-check fixture did not decode: {error}"))
}

fn assert_rejects(contents: &str, expected: InvariantId) {
    let request = load_fixture(contents);
    let result = check(&request).unwrap_or_else(|errors| panic!("invalid fixture: {errors:?}"));
    validate_result(&request, &result)
        .unwrap_or_else(|errors| panic!("result validation failed: {errors:?}"));
    assert_eq!(result.status, CheckStatus::Failed);
    let outcome = result
        .invariants
        .iter()
        .find(|outcome| outcome.invariant == expected)
        .unwrap_or_else(|| panic!("missing invariant {expected}"));
    assert_eq!(outcome.status, CheckStatus::Failed);
    assert!(!outcome.passed);
    let trace = outcome
        .counterexample
        .as_ref()
        .unwrap_or_else(|| panic!("missing counterexample for {expected}"));
    assert!(trace.minimized);
    replay_counterexample(&request, trace)
        .unwrap_or_else(|error| panic!("counterexample did not replay: {error}"));
}

#[test]
fn safe_streaming_fixture_passes() {
    let request = load_fixture(include_str!(
        "../../../modelcheck/streaming/safe_protocol.json"
    ));
    let result = check(&request).unwrap_or_else(|errors| panic!("invalid fixture: {errors:?}"));
    validate_result(&request, &result)
        .unwrap_or_else(|errors| panic!("result validation failed: {errors:?}"));
    assert_eq!(result.status, CheckStatus::Passed);
    assert!(result.scope.complete_within_bounds);
}

#[test]
fn concurrent_streaming_fixture_checks_two_request_contention_completely() {
    let request = load_fixture(include_str!(
        "../../../modelcheck/streaming/concurrent_safe_protocol.json"
    ));
    let result = check(&request).unwrap_or_else(|errors| panic!("invalid fixture: {errors:?}"));
    assert_eq!(result.status, CheckStatus::Passed);
    assert!(result.scope.complete_within_bounds);
    assert!(result.scope.truncated_by.is_empty());
    assert!(result.state_count > 1);
    assert!(result.transition_count > result.state_count);
    assert!(
        result
            .invariants
            .iter()
            .all(|outcome| outcome.counterexample.is_none())
    );
}

#[test]
fn duplicate_token_fixture_is_rejected() {
    assert_rejects(
        include_str!("../../../modelcheck/streaming/duplicate_token.json"),
        InvariantId::NoDuplicateCommittedToken,
    );
}

#[test]
fn cancellation_leak_fixture_is_rejected() {
    assert_rejects(
        include_str!("../../../modelcheck/cancellation/leaked_resource.json"),
        InvariantId::CancellationEventuallyReleases,
    );
}

#[test]
fn incompatible_migration_fixture_is_rejected() {
    assert_rejects(
        include_str!("../../../modelcheck/state_migration/incompatible_state.json"),
        InvariantId::NoIncompatibleStateMigration,
    );
}

#[test]
fn active_request_orphan_fixture_is_rejected() {
    assert_rejects(
        include_str!("../../../modelcheck/promotion/orphan_active.json"),
        InvariantId::PromotionPreservesActiveRequests,
    );
}

#[test]
fn invalid_rollback_fixture_is_rejected() {
    assert_rejects(
        include_str!("../../../modelcheck/rollback/invalid_previous.json"),
        InvariantId::RollbackRestoresValidChampion,
    );
}

#[test]
fn stalled_recovery_fixture_is_rejected() {
    assert_rejects(
        include_str!("../../../modelcheck/recovery/stalled_recovery.json"),
        InvariantId::RecoveryTerminates,
    );
}

#[test]
fn dropped_commit_and_queue_overflow_are_rejected() {
    let mut dropped = ModelCheckRequest::safe(83);
    dropped.protocol.token_delivery = TokenDelivery::MayDropCommitted;
    dropped.protocol.rollout_enabled = false;
    dropped.protocol.state_transfer_enabled = false;
    dropped.protocol.worker_failure_enabled = false;
    dropped.protocol.controller_crash_enabled = false;
    let result = check(&dropped).unwrap_or_else(|errors| panic!("invalid model: {errors:?}"));
    assert!(result.invariants.iter().any(|outcome| {
        outcome.invariant == InvariantId::NoCommittedTokenDisappears
            && outcome.status == CheckStatus::Failed
    }));

    let mut overflow = dropped;
    overflow.bounds.max_requests = 2;
    overflow.protocol.token_delivery = TokenDelivery::Reliable;
    overflow.protocol.queue_overflow = QueueOverflow::AdmitBeyondBound;
    let result = check(&overflow).unwrap_or_else(|errors| panic!("invalid model: {errors:?}"));
    assert!(result.invariants.iter().any(|outcome| {
        outcome.invariant == InvariantId::BoundedQueues && outcome.status == CheckStatus::Failed
    }));
}

#[test]
fn ownership_and_partial_read_obligations_are_independent() {
    let mut request = ModelCheckRequest::safe(97);
    request.protocol.rollout_enabled = false;
    request.protocol.worker_failure_enabled = false;
    request.protocol.controller_crash_enabled = false;
    request.protocol.state_ownership = StateOwnership::AmbiguousHandoff;
    request.protocol.partial_state_read = PartialStateRead::Allowed;
    let result = check(&request).unwrap_or_else(|errors| panic!("invalid model: {errors:?}"));
    for invariant in [
        InvariantId::UnambiguousStateOwnership,
        InvariantId::NoPartialStateRead,
    ] {
        assert!(result.invariants.iter().any(|outcome| {
            outcome.invariant == invariant && outcome.status == CheckStatus::Failed
        }));
    }

    request.protocol.replicated_state_declared = true;
    request.protocol.partial_state_read = PartialStateRead::Rejected;
    let result = check(&request).unwrap_or_else(|errors| panic!("invalid model: {errors:?}"));
    let ownership = result
        .invariants
        .iter()
        .find(|outcome| outcome.invariant == InvariantId::UnambiguousStateOwnership)
        .unwrap_or_else(|| panic!("missing ownership invariant"));
    assert_eq!(ownership.status, CheckStatus::Passed);
}

#[test]
fn trace_tampering_is_detected() {
    let request = load_fixture(include_str!(
        "../../../modelcheck/streaming/duplicate_token.json"
    ));
    let result = check(&request).unwrap_or_else(|errors| panic!("invalid fixture: {errors:?}"));
    let mut trace = result
        .invariants
        .iter()
        .find(|outcome| outcome.invariant == InvariantId::NoDuplicateCommittedToken)
        .and_then(|outcome| outcome.counterexample.clone())
        .unwrap_or_else(|| panic!("missing duplicate-token trace"));
    trace.steps[0].state_fingerprint = "fnv1a64:0000000000000000".to_owned();
    assert!(replay_counterexample(&request, &trace).is_err());
}
