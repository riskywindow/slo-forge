use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

use serde::{Deserialize, Serialize};

use crate::{
    Action, CancellationRelease, CheckStatus, CounterexampleTrace, DeploymentStage, InvariantId,
    InvariantOutcome, ModelCheckRequest, ModelCheckResult, PartialStateRead, PromotionBehavior,
    QueueOverflow, RESULT_SCHEMA_VERSION, RecoveryBehavior, ReplayError, RequestPhase,
    RequestSummary, RetryAfterOutput, RollbackBehavior, StateCompatibility, StateOwnership,
    StateSummary, StateTransferSummary, TokenDelivery, TraceStep, TruncationReason,
    VerificationScope,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
struct ProtocolState {
    queue: Vec<u8>,
    requests: Vec<RequestState>,
    deployment: DeploymentState,
    controller: ControllerState,
    worker_failures: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
// Each flag is an independently checked protocol invariant; collapsing them
// would obscure counterexample state and weaken replay diagnostics.
#[allow(clippy::struct_excessive_bools)]
struct RequestState {
    phase: RequestPhase,
    generation: Option<u8>,
    next_token: u8,
    committed: Vec<u8>,
    pending: Vec<u8>,
    emitted: Vec<u8>,
    owners: Vec<u8>,
    transfer: Option<TransferState>,
    transfers_completed: u8,
    resource_allocated: bool,
    retry_after_visible_output: bool,
    partial_read_observed: bool,
    incompatible_migration_observed: bool,
    stalled_ticks: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
struct TransferState {
    from: u8,
    to: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
struct DeploymentState {
    stage: DeploymentStage,
    champion_generation: u8,
    previous_generation: Option<u8>,
    orphaned_requests: Vec<u8>,
    rollback_invalid: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
struct ControllerState {
    up: bool,
    crashes: u8,
    recovering: bool,
    recovery_ticks: u8,
}

#[derive(Debug, Clone)]
struct Node {
    predecessor: Option<ProtocolState>,
    action: Option<Action>,
    depth: u16,
}

impl ProtocolState {
    fn initial(request: &ModelCheckRequest) -> Self {
        let requests = (0..request.bounds.max_requests)
            .map(|_| RequestState {
                phase: RequestPhase::Created,
                generation: None,
                next_token: 0,
                committed: Vec::new(),
                pending: Vec::new(),
                emitted: Vec::new(),
                owners: Vec::new(),
                transfer: None,
                transfers_completed: 0,
                resource_allocated: false,
                retry_after_visible_output: false,
                partial_read_observed: false,
                incompatible_migration_observed: false,
                stalled_ticks: 0,
            })
            .collect();
        Self {
            queue: Vec::new(),
            requests,
            deployment: DeploymentState {
                stage: DeploymentStage::Champion,
                champion_generation: 0,
                previous_generation: None,
                orphaned_requests: Vec::new(),
                rollback_invalid: false,
            },
            controller: ControllerState {
                up: true,
                crashes: 0,
                recovering: false,
                recovery_ticks: 0,
            },
            worker_failures: 0,
        }
    }

    fn summary(&self) -> StateSummary {
        StateSummary {
            queue: self.queue.clone(),
            requests: self
                .requests
                .iter()
                .enumerate()
                .map(|(request, state)| RequestSummary {
                    request: bounded_request_id(request),
                    phase: state.phase,
                    generation: state.generation,
                    committed_tokens: state.committed.clone(),
                    pending_tokens: state.pending.clone(),
                    emitted_tokens: state.emitted.clone(),
                    state_owners: state.owners.clone(),
                    state_transfer: state
                        .transfer
                        .as_ref()
                        .map(|transfer| StateTransferSummary {
                            from: transfer.from,
                            to: transfer.to,
                        }),
                    resource_allocated: state.resource_allocated,
                    retry_after_visible_output: state.retry_after_visible_output,
                    partial_state_read_observed: state.partial_read_observed,
                    incompatible_migration_observed: state.incompatible_migration_observed,
                    stalled_ticks: state.stalled_ticks,
                })
                .collect(),
            deployment_stage: self.deployment.stage,
            champion_generation: self.deployment.champion_generation,
            previous_generation: self.deployment.previous_generation,
            orphaned_requests: self.deployment.orphaned_requests.clone(),
            rollback_invalid: self.deployment.rollback_invalid,
            controller_up: self.controller.up,
            recovering: self.controller.recovering,
            recovery_ticks: self.controller.recovery_ticks,
        }
    }

    fn accepted_live_requests(&self) -> impl Iterator<Item = (usize, &RequestState)> {
        self.requests.iter().enumerate().filter(|(_, state)| {
            matches!(
                state.phase,
                RequestPhase::Queued
                    | RequestPhase::Running
                    | RequestPhase::Cancelling
                    | RequestPhase::Failed
            )
        })
    }

    fn has_progress_obligation(&self) -> bool {
        self.accepted_live_requests().next().is_some()
            || !self.controller.up
            || self.controller.recovering
            || self.deployment.stage == DeploymentStage::Promoting
    }
}

/// Explore the entire requested finite state space unless a declared cap is hit.
///
/// # Errors
///
/// Returns stable validation diagnostics if the request cannot be checked.
// Keeping the breadth-first audit loop contiguous makes every bounded-exit
// condition visible at the trusted model-checking boundary.
#[allow(clippy::too_many_lines)]
pub fn check(
    request: &ModelCheckRequest,
) -> Result<ModelCheckResult, Vec<crate::ValidationDiagnostic>> {
    request.validate()?;

    let initial = ProtocolState::initial(request);
    let mut queue = VecDeque::from([initial.clone()]);
    let mut nodes = HashMap::new();
    nodes.insert(
        initial.clone(),
        Node {
            predecessor: None,
            action: None,
            depth: 0,
        },
    );
    let mut first_failures: BTreeMap<InvariantId, (ProtocolState, bool)> = BTreeMap::new();
    let mut transition_count = 0_u64;
    let mut transition_coverage = BTreeMap::new();
    let mut truncated = BTreeSet::new();
    let mut states_checked = 0_u64;

    while let Some(state) = queue.pop_front() {
        states_checked = states_checked.saturating_add(1);
        let node_depth = nodes.get(&state).map_or(0, |node| node.depth);
        let mut successors = successors(request, &state);
        let deadlocked = successors.is_empty() && state.has_progress_obligation();

        for invariant in violated_invariants(request, &state, deadlocked) {
            first_failures
                .entry(invariant)
                .or_insert_with(|| (state.clone(), true));
        }

        if deadlocked {
            successors.push((Action::AdvanceTime, advance_time(request, &state)));
        }

        if node_depth >= request.bounds.max_depth {
            if successors.iter().any(|(_, next)| !nodes.contains_key(next)) {
                truncated.insert(TruncationReason::DepthLimit);
            }
            continue;
        }
        order_successors(request.seed, &mut successors);

        for (action, next) in successors {
            transition_count = transition_count.saturating_add(1);
            *transition_coverage
                .entry(action.kind().to_owned())
                .or_insert(0_u64) += 1;
            if nodes.contains_key(&next) {
                continue;
            }
            if nodes.len() as u64 >= request.bounds.max_states {
                truncated.insert(TruncationReason::StateLimit);
                queue.clear();
                break;
            }
            nodes.insert(
                next.clone(),
                Node {
                    predecessor: Some(state.clone()),
                    action: Some(action),
                    depth: node_depth.saturating_add(1),
                },
            );
            queue.push_back(next);
        }
    }

    let complete = truncated.is_empty();
    let invariants = InvariantId::ALL
        .into_iter()
        .map(|invariant| {
            let counterexample = first_failures
                .get(&invariant)
                .map(|(state, minimal)| trace_for(invariant, state, &nodes, *minimal));
            InvariantOutcome {
                invariant,
                status: if counterexample.is_some() {
                    CheckStatus::Failed
                } else if complete {
                    CheckStatus::Passed
                } else {
                    CheckStatus::Inconclusive
                },
                passed: counterexample.is_none() && complete,
                states_checked,
                verification_level: 3,
                scope_statement: scope_statement(invariant, request),
                counterexample,
            }
        })
        .collect::<Vec<_>>();
    let any_failed = invariants
        .iter()
        .any(|outcome| outcome.status == CheckStatus::Failed);
    let status = if any_failed {
        CheckStatus::Failed
    } else if complete {
        CheckStatus::Passed
    } else {
        CheckStatus::Inconclusive
    };
    let mut assumptions = vec![
        "bounded explicit-state exploration; no claim beyond the recorded bounds".to_owned(),
        "transition selection is weakly fair at modeled progress points".to_owned(),
        "logical timeout ticks occur only when no ordinary progress transition is enabled"
            .to_owned(),
        "token and state effects are atomic at the transition granularity shown in traces"
            .to_owned(),
    ];
    assumptions.extend(request.assumptions.iter().cloned());

    Ok(ModelCheckResult {
        schema_version: RESULT_SCHEMA_VERSION.to_owned(),
        model_version: request.model_version.clone(),
        seed: request.seed,
        status,
        scope: VerificationScope {
            method: "deterministic_bounded_explicit_state_breadth_first".to_owned(),
            verification_level: 3,
            bounds: request.bounds,
            complete_within_bounds: complete,
            truncated_by: truncated.into_iter().collect(),
            universal_proof: false,
        },
        state_count: nodes.len() as u64,
        transition_count,
        assumptions,
        invariants,
        transition_coverage,
    })
}

fn successors(request: &ModelCheckRequest, state: &ProtocolState) -> Vec<(Action, ProtocolState)> {
    let mut transitions = Vec::new();
    request_successors(request, state, &mut transitions);
    rollout_successors(request, state, &mut transitions);
    controller_successors(request, state, &mut transitions);
    transitions
}

fn request_successors(
    request: &ModelCheckRequest,
    state: &ProtocolState,
    transitions: &mut Vec<(Action, ProtocolState)>,
) {
    for request_index in 0..state.requests.len() {
        let request_id = bounded_request_id(request_index);
        let request_state = &state.requests[request_index];
        match request_state.phase {
            RequestPhase::Created if state.controller.up && !state.controller.recovering => {
                let has_room = state.queue.len() < usize::from(request.bounds.queue_capacity);
                if has_room || request.protocol.queue_overflow == QueueOverflow::AdmitBeyondBound {
                    let mut next = state.clone();
                    next.requests[request_index].phase = RequestPhase::Queued;
                    next.requests[request_index].generation =
                        Some(state.deployment.champion_generation);
                    next.queue.push(request_id);
                    transitions.push((
                        Action::Admit {
                            request: request_id,
                        },
                        next,
                    ));
                }
            }
            RequestPhase::Queued => {
                if state.controller.up && !state.controller.recovering {
                    let worker = request_id % request.bounds.worker_count;
                    let mut next = state.clone();
                    next.queue.retain(|queued| *queued != request_id);
                    let next_request = &mut next.requests[request_index];
                    next_request.phase = RequestPhase::Running;
                    next_request.owners = vec![worker];
                    next_request.resource_allocated = true;
                    transitions.push((
                        Action::Start {
                            request: request_id,
                            worker,
                        },
                        next,
                    ));
                }
                let mut cancelled = state.clone();
                cancelled.queue.retain(|queued| *queued != request_id);
                // A queued request owns no runtime state, so cancellation is an
                // atomic queue removal and cannot exercise the release protocol.
                release_request(
                    &mut cancelled.requests[request_index],
                    RequestPhase::Cancelled,
                );
                transitions.push((
                    Action::Cancel {
                        request: request_id,
                    },
                    cancelled,
                ));
            }
            RequestPhase::Running => {
                running_successors(request, state, request_index, transitions);
            }
            RequestPhase::Cancelling => {
                if request.protocol.cancellation_release == CancellationRelease::Guaranteed {
                    let mut next = state.clone();
                    release_request(&mut next.requests[request_index], RequestPhase::Cancelled);
                    transitions.push((
                        Action::ReleaseCancelled {
                            request: request_id,
                        },
                        next,
                    ));
                }
            }
            RequestPhase::Failed => {
                failed_successors(request, state, request_index, transitions);
            }
            RequestPhase::Created | RequestPhase::Completed | RequestPhase::Cancelled => {}
        }
    }
}

fn running_successors(
    request: &ModelCheckRequest,
    state: &ProtocolState,
    request_index: usize,
    transitions: &mut Vec<(Action, ProtocolState)>,
) {
    let request_id = bounded_request_id(request_index);
    let request_state = &state.requests[request_index];
    if request_state.pending.is_empty()
        && request_state.next_token < request.bounds.max_tokens_per_request
    {
        let token = request_state.next_token;
        let mut next = state.clone();
        next.requests[request_index].next_token = token.saturating_add(1);
        next.requests[request_index].committed.push(token);
        next.requests[request_index].pending.push(token);
        transitions.push((
            Action::CommitToken {
                request: request_id,
                token,
            },
            next,
        ));
    }
    if let Some(token) = request_state.pending.first().copied() {
        let mut next = state.clone();
        next.requests[request_index].pending.remove(0);
        next.requests[request_index].emitted.push(token);
        transitions.push((
            Action::EmitToken {
                request: request_id,
                token,
            },
            next,
        ));
        if request.protocol.token_delivery == TokenDelivery::MayDropCommitted {
            let mut dropped = state.clone();
            dropped.requests[request_index].pending.remove(0);
            transitions.push((
                Action::DropCommittedToken {
                    request: request_id,
                    token,
                },
                dropped,
            ));
        }
    }
    if request.protocol.token_delivery == TokenDelivery::MayDuplicateCommitted
        && request_state.emitted.len()
            < usize::from(request.bounds.max_tokens_per_request).saturating_mul(2)
    {
        if let Some(token) = request_state.emitted.last().copied() {
            let mut duplicate = state.clone();
            duplicate.requests[request_index].emitted.push(token);
            transitions.push((
                Action::DuplicateCommittedToken {
                    request: request_id,
                    token,
                },
                duplicate,
            ));
        }
    }
    if request_state.pending.is_empty()
        && request_state.next_token == request.bounds.max_tokens_per_request
        && request_state.emitted.len() == usize::from(request.bounds.max_tokens_per_request)
    {
        let mut next = state.clone();
        release_request(&mut next.requests[request_index], RequestPhase::Completed);
        transitions.push((
            Action::Complete {
                request: request_id,
            },
            next,
        ));
    }
    if request_state.pending.is_empty() {
        let mut cancelled = state.clone();
        cancelled.requests[request_index].phase = RequestPhase::Cancelling;
        transitions.push((
            Action::Cancel {
                request: request_id,
            },
            cancelled,
        ));
    }
    if request.protocol.worker_failure_enabled
        && state.worker_failures < request.bounds.max_worker_failures
        && request_state.transfer.is_none()
    {
        let mut failed = state.clone();
        failed.worker_failures = failed.worker_failures.saturating_add(1);
        failed.requests[request_index].phase = RequestPhase::Failed;
        transitions.push((
            Action::FailWorker {
                request: request_id,
            },
            failed,
        ));
    }
    state_transfer_successors(request, state, request_index, transitions);
}

fn state_transfer_successors(
    request: &ModelCheckRequest,
    state: &ProtocolState,
    request_index: usize,
    transitions: &mut Vec<(Action, ProtocolState)>,
) {
    if !request.protocol.state_transfer_enabled {
        return;
    }
    let request_id = bounded_request_id(request_index);
    let request_state = &state.requests[request_index];
    if let Some(transfer) = &request_state.transfer {
        let mut complete = state.clone();
        let completed_request = &mut complete.requests[request_index];
        completed_request.owners = vec![transfer.to];
        completed_request.transfer = None;
        completed_request.transfers_completed =
            completed_request.transfers_completed.saturating_add(1);
        transitions.push((
            Action::CompleteStateTransfer {
                request: request_id,
                to: transfer.to,
            },
            complete,
        ));
        if request.protocol.partial_state_read == PartialStateRead::Allowed {
            let mut read = state.clone();
            read.requests[request_index].partial_read_observed = true;
            transitions.push((
                Action::ReadPartialState {
                    request: request_id,
                },
                read,
            ));
        }
        return;
    }
    let Some(from) = request_state.owners.first().copied() else {
        return;
    };
    // This finite model checks at most one state migration per request. More
    // migrations are represented by increasing the request bound, rather than
    // introducing an artificial unbounded handoff cycle.
    if request_state.transfers_completed > 0 {
        return;
    }
    let to = (from + 1) % request.bounds.worker_count;
    let compatible = request.protocol.state_compatibility == StateCompatibility::Compatible;
    if !compatible
        && request.protocol.state_compatibility == StateCompatibility::IncompatibleRejected
    {
        return;
    }
    let mut next = state.clone();
    let transferred = &mut next.requests[request_index];
    transferred.transfer = Some(TransferState { from, to });
    if request.protocol.state_ownership == StateOwnership::AmbiguousHandoff {
        transferred.owners.push(to);
        transferred.owners.sort_unstable();
        transferred.owners.dedup();
    }
    if !compatible {
        transferred.incompatible_migration_observed = true;
    }
    transitions.push((
        Action::BeginStateTransfer {
            request: request_id,
            from,
            to,
        },
        next,
    ));
}

fn failed_successors(
    request: &ModelCheckRequest,
    state: &ProtocolState,
    request_index: usize,
    transitions: &mut Vec<(Action, ProtocolState)>,
) {
    let request_id = bounded_request_id(request_index);
    let failed = &state.requests[request_index];
    let has_visible_output = !failed.emitted.is_empty();
    let can_retry =
        !has_visible_output || request.protocol.retry_after_output != RetryAfterOutput::Forbid;
    if can_retry && state.controller.up && !state.controller.recovering {
        let current_worker = failed.owners.first().copied().unwrap_or(0);
        let worker = (current_worker + 1) % request.bounds.worker_count;
        let mut next = state.clone();
        let retried = &mut next.requests[request_index];
        retried.phase = RequestPhase::Running;
        retried.owners = vec![worker];
        if has_visible_output {
            match request.protocol.retry_after_output {
                RetryAfterOutput::Forbid | RetryAfterOutput::IdempotentContinuation => {}
                RetryAfterOutput::Restart => {
                    retried.retry_after_visible_output = true;
                    retried.next_token = 0;
                    retried.committed.clear();
                    retried.pending.clear();
                }
            }
        }
        transitions.push((
            Action::Retry {
                request: request_id,
                worker,
            },
            next,
        ));
    }
    // An internally committed token must be resumed and delivered before an
    // abort can become externally visible.
    if failed.pending.is_empty() {
        let mut aborted = state.clone();
        release_request(
            &mut aborted.requests[request_index],
            RequestPhase::Cancelled,
        );
        transitions.push((
            Action::AbortFailed {
                request: request_id,
            },
            aborted,
        ));
    }
}

fn rollout_successors(
    request: &ModelCheckRequest,
    state: &ProtocolState,
    transitions: &mut Vec<(Action, ProtocolState)>,
) {
    if !request.protocol.rollout_enabled || !state.controller.up || state.controller.recovering {
        return;
    }
    match state.deployment.stage {
        DeploymentStage::Champion => {
            let mut next = state.clone();
            next.deployment.stage = DeploymentStage::Shadow;
            transitions.push((Action::BeginShadow, next));
        }
        DeploymentStage::Shadow => {
            let mut next = state.clone();
            next.deployment.stage = DeploymentStage::Canary;
            transitions.push((Action::BeginCanary, next));
        }
        DeploymentStage::Canary => {
            let mut next = state.clone();
            next.deployment.stage = DeploymentStage::Promoting;
            transitions.push((Action::BeginPromotion, next));
        }
        DeploymentStage::Promoting => {
            let active_old = state
                .accepted_live_requests()
                .filter(|(_, active)| {
                    active.generation == Some(state.deployment.champion_generation)
                })
                .map(|(index, _)| bounded_request_id(index))
                .collect::<Vec<_>>();
            if active_old.is_empty()
                || request.protocol.promotion == PromotionBehavior::OrphanActive
            {
                let mut next = state.clone();
                next.deployment.previous_generation = Some(state.deployment.champion_generation);
                next.deployment.champion_generation =
                    state.deployment.champion_generation.saturating_add(1);
                next.deployment.stage = DeploymentStage::Promoted;
                if !active_old.is_empty() {
                    next.deployment.orphaned_requests = active_old;
                }
                transitions.push((Action::CommitPromotion, next));
            }
        }
        DeploymentStage::Promoted => {
            let mut next = state.clone();
            next.deployment.stage = DeploymentStage::RolledBack;
            match request.protocol.rollback {
                RollbackBehavior::RestorePrevious => {
                    if let Some(previous) = state.deployment.previous_generation {
                        next.deployment.champion_generation = previous;
                    } else {
                        next.deployment.rollback_invalid = true;
                    }
                }
                RollbackBehavior::LosePrevious => {
                    next.deployment.previous_generation = None;
                    next.deployment.rollback_invalid = true;
                }
            }
            transitions.push((Action::Rollback, next));
        }
        DeploymentStage::RolledBack => {}
    }
}

fn controller_successors(
    request: &ModelCheckRequest,
    state: &ProtocolState,
    transitions: &mut Vec<(Action, ProtocolState)>,
) {
    if !request.protocol.controller_crash_enabled {
        return;
    }
    if state.controller.up && !state.controller.recovering && state.controller.crashes == 0 {
        let mut next = state.clone();
        next.controller.up = false;
        next.controller.crashes = 1;
        transitions.push((Action::CrashController, next));
    } else if !state.controller.up {
        let mut next = state.clone();
        next.controller.up = true;
        next.controller.recovering = true;
        transitions.push((Action::RestartController, next));
    } else if state.controller.recovering
        && request.protocol.recovery == RecoveryBehavior::Guaranteed
    {
        let mut next = state.clone();
        next.controller.recovering = false;
        next.controller.recovery_ticks = 0;
        transitions.push((Action::CompleteRecovery, next));
    }
}

fn advance_time(request: &ModelCheckRequest, state: &ProtocolState) -> ProtocolState {
    let mut next = state.clone();
    for request_state in &mut next.requests {
        if matches!(
            request_state.phase,
            RequestPhase::Queued
                | RequestPhase::Running
                | RequestPhase::Cancelling
                | RequestPhase::Failed
        ) {
            request_state.stalled_ticks = request_state
                .stalled_ticks
                .saturating_add(1)
                .min(request.bounds.fairness_window.saturating_add(1));
        }
    }
    if next.controller.recovering {
        next.controller.recovery_ticks = next
            .controller
            .recovery_ticks
            .saturating_add(1)
            .min(request.bounds.fairness_window.saturating_add(1));
    }
    next
}

fn release_request(request: &mut RequestState, terminal: RequestPhase) {
    request.phase = terminal;
    request.owners.clear();
    request.transfer = None;
    request.resource_allocated = false;
    request.stalled_ticks = 0;
}

fn violated_invariants(
    request: &ModelCheckRequest,
    state: &ProtocolState,
    deadlocked: bool,
) -> Vec<InvariantId> {
    let mut violated = BTreeSet::new();
    if state.queue.len() > usize::from(request.bounds.queue_capacity)
        || state.queue.iter().collect::<BTreeSet<_>>().len() != state.queue.len()
        || state
            .queue
            .iter()
            .any(|queued| state.requests[usize::from(*queued)].phase != RequestPhase::Queued)
    {
        violated.insert(InvariantId::BoundedQueues);
    }
    if !state.deployment.orphaned_requests.is_empty() {
        violated.insert(InvariantId::PromotionPreservesActiveRequests);
    }
    if state.deployment.rollback_invalid {
        violated.insert(InvariantId::RollbackRestoresValidChampion);
    }
    if state.controller.recovery_ticks > request.bounds.fairness_window {
        violated.insert(InvariantId::RecoveryTerminates);
    }
    if deadlocked {
        violated.insert(InvariantId::NoBoundedDeadlock);
    }
    for request_state in &state.requests {
        let emitted_unique = request_state.emitted.iter().collect::<BTreeSet<_>>();
        if emitted_unique.len() != request_state.emitted.len() {
            violated.insert(InvariantId::NoDuplicateCommittedToken);
        }
        let terminal = matches!(
            request_state.phase,
            RequestPhase::Completed | RequestPhase::Cancelled
        );
        if request_state.committed.iter().any(|token| {
            let pending = request_state.pending.iter().fold(0_usize, |count, item| {
                count + usize::from(item == token)
            });
            let emitted = request_state.emitted.iter().fold(0_usize, |count, item| {
                count + usize::from(item == token)
            });
            emitted == 0 && (pending == 0 || terminal)
        }) {
            violated.insert(InvariantId::NoCommittedTokenDisappears);
        }
        if request_state.retry_after_visible_output {
            violated.insert(InvariantId::NoUnsafeRetryAfterOutput);
        }
        if request_state.phase == RequestPhase::Cancelling
            && request_state.resource_allocated
            && request_state.stalled_ticks > request.bounds.fairness_window
        {
            violated.insert(InvariantId::CancellationEventuallyReleases);
        }
        if request_state.owners.len() > 1 && !request.protocol.replicated_state_declared {
            violated.insert(InvariantId::UnambiguousStateOwnership);
        }
        if request_state.partial_read_observed {
            violated.insert(InvariantId::NoPartialStateRead);
        }
        if request_state.incompatible_migration_observed {
            violated.insert(InvariantId::NoIncompatibleStateMigration);
        }
        if request_state.stalled_ticks > request.bounds.fairness_window {
            violated.insert(InvariantId::LiveRequestsProgressOrTimeout);
        }
        if terminal && request_state.resource_allocated {
            violated.insert(InvariantId::CancellationEventuallyReleases);
        }
    }
    violated.into_iter().collect()
}

fn order_successors(seed: u64, successors: &mut [(Action, ProtocolState)]) {
    successors.sort_by_cached_key(|(action, _)| {
        let encoded = serde_json::to_vec(action).unwrap_or_default();
        (seeded_hash(seed, &encoded), encoded)
    });
}

fn bounded_request_id(index: usize) -> u8 {
    // Request validation caps this vector at eight entries. The fallback keeps
    // summary construction total if an internal test deliberately violates it.
    u8::try_from(index).unwrap_or(u8::MAX)
}

fn seeded_hash(seed: u64, bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64 ^ seed.rotate_left(17);
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn fingerprint(state: &ProtocolState) -> String {
    let bytes = serde_json::to_vec(state).unwrap_or_default();
    format!("fnv1a64:{:016x}", seeded_hash(0, &bytes))
}

fn trace_for(
    invariant: InvariantId,
    failed: &ProtocolState,
    nodes: &HashMap<ProtocolState, Node>,
    minimized: bool,
) -> CounterexampleTrace {
    let mut reverse = Vec::new();
    let mut cursor = failed.clone();
    while let Some(node) = nodes.get(&cursor) {
        let Some(action) = node.action.clone() else {
            break;
        };
        reverse.push((action, cursor.clone()));
        let Some(predecessor) = node.predecessor.clone() else {
            break;
        };
        cursor = predecessor;
    }
    reverse.reverse();
    let steps = reverse
        .into_iter()
        .enumerate()
        .map(|(index, (action, state))| TraceStep {
            index: u16::try_from(index).unwrap_or(u16::MAX),
            action,
            state_fingerprint: fingerprint(&state),
            state: state.summary(),
        })
        .collect();
    CounterexampleTrace {
        violated_invariant: invariant,
        minimized,
        minimization: "shortest transition count by exhaustive breadth-first exploration"
            .to_owned(),
        initial_state_fingerprint: fingerprint(&cursor),
        steps,
    }
}

fn scope_statement(invariant: InvariantId, request: &ModelCheckRequest) -> String {
    format!(
        "{invariant} checked by explicit enumeration for <= {} requests, <= {} tokens/request, <= {} workers, depth <= {}, and <= {} states",
        request.bounds.max_requests,
        request.bounds.max_tokens_per_request,
        request.bounds.worker_count,
        request.bounds.max_depth,
        request.bounds.max_states
    )
}

/// Replay a returned trace against the same request and verify its final state
/// violates the declared invariant.
///
/// # Errors
///
/// Returns the first unavailable action, fingerprint mismatch, or missing final
/// violation. This makes serialized traces independently tamper-evident within
/// the checker model; capsule-level cryptographic hashing is handled elsewhere.
pub fn replay_counterexample(
    request: &ModelCheckRequest,
    trace: &CounterexampleTrace,
) -> Result<(), ReplayError> {
    request.validate().map_err(|diagnostics| ReplayError {
        step: 0,
        message: format!("invalid request: {diagnostics:?}"),
    })?;
    let mut state = ProtocolState::initial(request);
    if fingerprint(&state) != trace.initial_state_fingerprint {
        return Err(ReplayError {
            step: 0,
            message: "initial state fingerprint mismatch".to_owned(),
        });
    }
    for (index, expected) in trace.steps.iter().enumerate() {
        let next = successors(request, &state)
            .into_iter()
            .find_map(|(action, next)| (action == expected.action).then_some(next))
            .or_else(|| {
                let normal = successors(request, &state);
                let deadlocked = normal.is_empty() && state.has_progress_obligation();
                (deadlocked && expected.action == Action::AdvanceTime)
                    .then(|| advance_time(request, &state))
            })
            .ok_or_else(|| ReplayError {
                step: index,
                message: format!("action {:?} is not enabled", expected.action),
            })?;
        if fingerprint(&next) != expected.state_fingerprint || next.summary() != expected.state {
            return Err(ReplayError {
                step: index,
                message: "state summary or fingerprint mismatch".to_owned(),
            });
        }
        state = next;
    }
    let deadlocked = successors(request, &state).is_empty() && state.has_progress_obligation();
    if violated_invariants(request, &state, deadlocked).contains(&trace.violated_invariant) {
        Ok(())
    } else {
        Err(ReplayError {
            step: trace.steps.len(),
            message: "final state does not violate the declared invariant".to_owned(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_protocol_passes_with_complete_bounded_scope() {
        let mut request = ModelCheckRequest::safe(73129);
        request.bounds.max_depth = 32;
        request.bounds.max_states = 250_000;
        let result = check(&request).unwrap_or_else(|errors| panic!("invalid request: {errors:?}"));
        assert_eq!(result.status, CheckStatus::Passed);
        assert!(result.scope.complete_within_bounds);
        assert!(!result.scope.universal_proof);
        assert!(result.invariants.iter().all(|outcome| outcome.passed));
        for required in [
            "admit",
            "commit_token",
            "cancel",
            "fail_worker",
            "begin_state_transfer",
            "begin_shadow",
            "begin_canary",
            "begin_promotion",
            "commit_promotion",
            "rollback",
            "crash_controller",
            "restart_controller",
        ] {
            assert!(
                result.transition_coverage.contains_key(required),
                "missing transition coverage for {required}"
            );
        }
    }

    #[test]
    fn unsafe_retry_has_shortest_replayable_counterexample() {
        let mut request = ModelCheckRequest::safe(19);
        request.protocol.retry_after_output = RetryAfterOutput::Restart;
        let result = check(&request).unwrap_or_else(|errors| panic!("invalid request: {errors:?}"));
        let outcome = result
            .invariants
            .iter()
            .find(|outcome| outcome.invariant == InvariantId::NoUnsafeRetryAfterOutput)
            .unwrap_or_else(|| panic!("missing invariant"));
        let trace = outcome
            .counterexample
            .as_ref()
            .unwrap_or_else(|| panic!("unsafe retry was not rejected"));
        assert!(trace.minimized);
        assert!(trace.steps.len() <= 6);
        replay_counterexample(&request, trace)
            .unwrap_or_else(|error| panic!("counterexample did not replay: {error}"));
    }

    #[test]
    fn state_limit_is_inconclusive_not_passed() {
        let mut request = ModelCheckRequest::safe(5);
        request.bounds.max_states = 1;
        let result = check(&request).unwrap_or_else(|errors| panic!("invalid request: {errors:?}"));
        assert_eq!(result.status, CheckStatus::Inconclusive);
        assert!(!result.scope.complete_within_bounds);
        assert_eq!(
            result.scope.truncated_by,
            vec![TruncationReason::StateLimit]
        );
    }

    #[test]
    fn seed_is_deterministic() {
        let request = ModelCheckRequest::safe(41);
        let first = check(&request).unwrap_or_else(|errors| panic!("invalid request: {errors:?}"));
        let second = check(&request).unwrap_or_else(|errors| panic!("invalid request: {errors:?}"));
        assert_eq!(first, second);
    }

    #[test]
    fn schemas_are_objects() {
        let request = crate::request_schema();
        let result = crate::result_schema();
        let request_json = serde_json::to_value(request)
            .unwrap_or_else(|error| panic!("request schema serialization failed: {error}"));
        let result_json = serde_json::to_value(result)
            .unwrap_or_else(|error| panic!("result schema serialization failed: {error}"));
        assert_eq!(request_json["type"], "object");
        assert_eq!(result_json["type"], "object");
    }
}
