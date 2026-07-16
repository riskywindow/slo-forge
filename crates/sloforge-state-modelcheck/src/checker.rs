use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

use serde::{Deserialize, Serialize};
use sloforge_state_transaction::CutoverPhase;

use crate::{
    Action, CheckStatus, Component, CounterexampleTrace, InvariantId, InvariantOutcome,
    MessageKind, ModelCheckRequest, ModelCheckResult, RESULT_SCHEMA_VERSION, ReplayError,
    StateSummary, TraceStep, TruncationReason, VerificationScope,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
struct Message {
    kind: MessageKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[allow(clippy::struct_excessive_bools)]
struct ProtocolState {
    phase: CutoverPhase,
    logical_tick: u8,
    messages: Vec<Message>,
    source_up: bool,
    destination_up: bool,
    gateway_up: bool,
    coordinator_up: bool,
    client_connected: bool,
    partitioned: bool,
    source_active: bool,
    destination_active: bool,
    source_fenced: bool,
    source_owns: bool,
    destination_owns: bool,
    state_owner_epoch: u8,
    gateway_owner_epoch: u8,
    highest_owner_epoch: u8,
    destination_prepared: bool,
    initial_received: bool,
    delta_enqueued: bool,
    delta_received: bool,
    final_received: bool,
    destination_validated: bool,
    commit_intent: bool,
    accepted_tokens: Vec<u8>,
    accepted_epochs: Vec<u8>,
    gateway_next_token: u8,
    highest_watermark: Option<u8>,
    client_ack: Option<u8>,
    pending_ack: bool,
    cancelled: bool,
    duplicate_accepted: bool,
    gap_accepted: bool,
    stale_output_accepted: bool,
    epoch_regressed: bool,
    watermark_regressed: bool,
    replay_changed_state: bool,
    message_duplicated: bool,
    message_dropped: bool,
    source_crashed: bool,
    destination_crashed: bool,
    gateway_crashed: bool,
    coordinator_crashed: bool,
    partition_injected: bool,
    stale_attempted: bool,
    cancellation_injected: bool,
    client_disconnected: bool,
    acknowledgment_delayed: bool,
}

impl ProtocolState {
    fn initial() -> Self {
        Self {
            phase: CutoverPhase::Proposed,
            logical_tick: 0,
            messages: Vec::new(),
            source_up: true,
            destination_up: true,
            gateway_up: true,
            coordinator_up: true,
            client_connected: true,
            partitioned: false,
            source_active: true,
            destination_active: false,
            source_fenced: false,
            source_owns: true,
            destination_owns: false,
            state_owner_epoch: 1,
            gateway_owner_epoch: 1,
            highest_owner_epoch: 1,
            destination_prepared: false,
            initial_received: false,
            delta_enqueued: false,
            delta_received: false,
            final_received: false,
            destination_validated: false,
            commit_intent: false,
            accepted_tokens: Vec::new(),
            accepted_epochs: Vec::new(),
            gateway_next_token: 0,
            highest_watermark: None,
            client_ack: None,
            pending_ack: false,
            cancelled: false,
            duplicate_accepted: false,
            gap_accepted: false,
            stale_output_accepted: false,
            epoch_regressed: false,
            watermark_regressed: false,
            replay_changed_state: false,
            message_duplicated: false,
            message_dropped: false,
            source_crashed: false,
            destination_crashed: false,
            gateway_crashed: false,
            coordinator_crashed: false,
            partition_injected: false,
            stale_attempted: false,
            cancellation_injected: false,
            client_disconnected: false,
            acknowledgment_delayed: false,
        }
    }

    fn summary(&self) -> StateSummary {
        StateSummary {
            phase: self.phase,
            logical_tick: self.logical_tick,
            state_owner_epoch: self.state_owner_epoch,
            gateway_owner_epoch: self.gateway_owner_epoch,
            source_active: self.source_active,
            destination_active: self.destination_active,
            source_fenced: self.source_fenced,
            destination_validated: self.destination_validated,
            gateway_next_token: self.gateway_next_token,
            accepted_tokens: self.accepted_tokens.clone(),
            message_count: u8::try_from(self.messages.len()).unwrap_or(u8::MAX),
            coordinator_up: self.coordinator_up,
            gateway_up: self.gateway_up,
            client_connected: self.client_connected,
        }
    }

    fn is_terminal(&self) -> bool {
        self.phase.is_terminal()
    }
}

#[derive(Debug, Clone)]
struct Node {
    predecessor: Option<ProtocolState>,
    action: Option<Action>,
    depth: u16,
}

/// Exhaustively explore the requested finite state space unless a declared cap
/// is reached.
///
/// # Errors
///
/// Returns stable request validation diagnostics.
#[allow(clippy::too_many_lines)]
pub fn check(
    request: &ModelCheckRequest,
) -> Result<ModelCheckResult, Vec<crate::ValidationDiagnostic>> {
    request.validate()?;
    let initial = ProtocolState::initial();
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
    let mut edges: HashMap<ProtocolState, Vec<ProtocolState>> = HashMap::new();
    let mut first_failures: BTreeMap<InvariantId, ProtocolState> = BTreeMap::new();
    let mut transition_coverage = BTreeMap::new();
    let mut transitions_explored = 0_u64;
    let mut states_explored = 0_u64;
    let mut max_depth_reached = 0_u16;
    let mut truncated = BTreeSet::new();

    while let Some(state) = queue.pop_front() {
        states_explored = states_explored.saturating_add(1);
        let depth = nodes.get(&state).map_or(0, |node| node.depth);
        max_depth_reached = max_depth_reached.max(depth);
        let mut successors = successors(request, &state);
        let deadlocked = successors.is_empty() && !state.is_terminal();
        for invariant in violated_invariants(request, &state, deadlocked) {
            first_failures
                .entry(invariant)
                .or_insert_with(|| state.clone());
        }
        if depth >= request.bounds.max_depth {
            if successors.iter().any(|(_, next)| !nodes.contains_key(next)) {
                truncated.insert(TruncationReason::DepthLimit);
            }
            continue;
        }
        order_successors(request.seed, &mut successors);
        for (action, next) in successors {
            transitions_explored = transitions_explored.saturating_add(1);
            *transition_coverage
                .entry(action.kind().to_owned())
                .or_insert(0_u64) += 1;
            edges.entry(state.clone()).or_default().push(next.clone());
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
                    depth: depth.saturating_add(1),
                },
            );
            queue.push_back(next);
        }
    }

    if truncated.is_empty() {
        let terminal_reachable = states_reaching_terminal(&nodes, &edges);
        if let Some(state) = nodes
            .keys()
            .find(|state| !terminal_reachable.contains(*state))
        {
            first_failures
                .entry(InvariantId::ExplicitTerminalOrRecoverable)
                .or_insert_with(|| state.clone());
        }
    }

    let complete = truncated.is_empty();
    let invariants = InvariantId::ALL
        .into_iter()
        .map(|invariant| {
            if let Some(state) = first_failures.get(&invariant) {
                InvariantOutcome {
                    invariant,
                    status: CheckStatus::Failed,
                    passed: false,
                    description: invariant_description(invariant).to_owned(),
                    counterexample: Some(trace_for(invariant, state, &nodes)),
                }
            } else {
                let status = if complete {
                    CheckStatus::Passed
                } else {
                    CheckStatus::Inconclusive
                };
                InvariantOutcome {
                    invariant,
                    status,
                    passed: status == CheckStatus::Passed,
                    description: invariant_description(invariant).to_owned(),
                    counterexample: None,
                }
            }
        })
        .collect::<Vec<_>>();
    let status = if invariants
        .iter()
        .any(|outcome| outcome.status == CheckStatus::Failed)
    {
        CheckStatus::Failed
    } else if complete {
        CheckStatus::Passed
    } else {
        CheckStatus::Inconclusive
    };

    Ok(ModelCheckResult {
        schema_version: RESULT_SCHEMA_VERSION.to_owned(),
        model_version: request.model_version.clone(),
        seed: request.seed,
        status,
        states_explored,
        transitions_explored,
        max_depth_reached,
        transition_coverage,
        invariants,
        scope: VerificationScope {
            claim: "bounded explicit-state exploration; not a universal proof".to_owned(),
            bounded_not_universal_proof: true,
            complete_within_bounds: complete,
            bounds: request.bounds,
            assumptions: request.assumptions.clone(),
            truncated_by: truncated.into_iter().collect(),
        },
    })
}

fn successors(request: &ModelCheckRequest, state: &ProtocolState) -> Vec<(Action, ProtocolState)> {
    if state.is_terminal() {
        return Vec::new();
    }
    let mut actions = enabled_actions(request, state);
    actions.sort();
    actions.dedup();
    actions
        .into_iter()
        .filter_map(|action| apply_action(request, state, &action).map(|next| (action, next)))
        .filter(|(_, next)| next != state)
        .collect()
}

#[allow(clippy::too_many_lines)]
fn enabled_actions(request: &ModelCheckRequest, state: &ProtocolState) -> Vec<Action> {
    if state.logical_tick > 0 && state.logical_tick < request.bounds.timeout_ticks {
        return vec![Action::AdvanceTime];
    }
    let mut actions = Vec::new();
    let fault_budget = injected_fault_count(state) < request.bounds.max_faults_per_execution;
    match state.phase {
        CutoverPhase::Proposed if state.coordinator_up => {
            actions.push(Action::ValidateCompatibility);
        }
        CutoverPhase::CompatibilityValidated if state.destination_up => {
            actions.push(Action::PrepareDestination);
        }
        CutoverPhase::DestinationPreparing if state.destination_prepared => {
            actions.push(Action::BeginPrecopy);
        }
        CutoverPhase::Precopying => {
            if !state.initial_received
                && !state
                    .messages
                    .iter()
                    .any(|message| message.kind == MessageKind::InitialSnapshot)
            {
                actions.push(Action::Enqueue(MessageKind::InitialSnapshot));
            }
            if !state.delta_enqueued {
                actions.push(Action::Enqueue(MessageKind::Delta));
            }
            if state.initial_received && state.delta_received && state.source_up {
                actions.push(Action::QuiesceSource);
            }
        }
        CutoverPhase::DeltaSyncing => {
            if !state.initial_received
                && !state
                    .messages
                    .iter()
                    .any(|message| message.kind == MessageKind::InitialSnapshot)
            {
                actions.push(Action::Enqueue(MessageKind::InitialSnapshot));
            }
            if !state.delta_enqueued {
                actions.push(Action::Enqueue(MessageKind::Delta));
            }
            if state.initial_received && state.delta_received && state.source_up {
                actions.push(Action::QuiesceSource);
            }
        }
        CutoverPhase::SourceFrozen => {
            if !state.final_received
                && !state
                    .messages
                    .iter()
                    .any(|message| message.kind == MessageKind::FinalDelta)
            {
                actions.push(Action::Enqueue(MessageKind::FinalDelta));
            }
        }
        CutoverPhase::DestinationImporting if state.destination_up && state.final_received => {
            actions.push(Action::ValidateDestination);
            if !request.protocol.require_validation_before_activation {
                actions.push(Action::RecordCommitIntent);
            }
        }
        CutoverPhase::DestinationValidating if state.coordinator_up => {
            actions.push(Action::RecordCommitIntent);
        }
        CutoverPhase::CommitIntentRecorded if state.coordinator_up => {
            actions.push(Action::CommitOwnership);
        }
        CutoverPhase::OwnershipCommitted => {
            if !state
                .messages
                .iter()
                .any(|message| message.kind == MessageKind::GatewaySwitch)
            {
                actions.push(Action::Enqueue(MessageKind::GatewaySwitch));
            }
            if !request.protocol.require_validation_before_activation {
                actions.push(Action::ActivateDestination);
            }
        }
        CutoverPhase::GatewaySwitching if state.destination_up => {
            actions.push(Action::ActivateDestination);
        }
        CutoverPhase::DestinationActive | CutoverPhase::SourceDraining => {
            actions.push(Action::DrainSource);
        }
        CutoverPhase::Aborting => actions.push(Action::FinishAbort),
        CutoverPhase::DestinationLost | CutoverPhase::SourceLost => {
            if !state.destination_owns && state.source_up {
                actions.push(Action::Abort);
            } else if state.destination_owns && state.destination_up && state.destination_validated
            {
                if state.gateway_owner_epoch == 1 {
                    actions.push(Action::SwitchGateway);
                } else if state.destination_active {
                    actions.push(Action::DrainSource);
                } else {
                    actions.push(Action::ActivateDestination);
                }
            } else {
                actions.push(Action::RequireOperator);
            }
        }
        CutoverPhase::CoordinatorUnavailable => {
            if !state.coordinator_up {
                actions.push(Action::Restart(Component::Coordinator));
            } else if state.commit_intent {
                actions.push(Action::RecordCommitIntent);
            } else {
                actions.push(Action::Abort);
            }
        }
        _ => {}
    }

    if !state.partitioned && !state.messages.is_empty() {
        let deliverable = if request.faults.message_reordering {
            state.messages.len()
        } else {
            1
        };
        actions.extend((0..deliverable).filter_map(|position| {
            u8::try_from(position)
                .ok()
                .map(|position| Action::Deliver { position })
        }));
    }
    if request.faults.message_duplication
        && fault_budget
        && !state.message_duplicated
        && !state.messages.is_empty()
    {
        actions.push(Action::Duplicate { position: 0 });
    }
    if request.faults.message_loss
        && fault_budget
        && !state.message_dropped
        && !state.messages.is_empty()
    {
        actions.push(Action::Drop { position: 0 });
    }
    if request.faults.network_partition
        && fault_budget
        && !state.partition_injected
        && state.phase == CutoverPhase::Precopying
    {
        actions.push(Action::StartPartition);
    }
    if state.partitioned {
        actions.push(Action::HealPartition);
    }
    if request.faults.source_crash
        && fault_budget
        && !state.source_crashed
        && state.source_up
        && matches!(
            state.phase,
            CutoverPhase::Proposed
                | CutoverPhase::CompatibilityValidated
                | CutoverPhase::DestinationPreparing
                | CutoverPhase::Precopying
                | CutoverPhase::DeltaSyncing
                | CutoverPhase::CutoverRequested
                | CutoverPhase::SourceQuiescing
                | CutoverPhase::SourceFrozen
                | CutoverPhase::FinalDeltaTransferring
                | CutoverPhase::DestinationImporting
                | CutoverPhase::DestinationValidating
                | CutoverPhase::CommitIntentRecorded
                | CutoverPhase::OwnershipCommitted
                | CutoverPhase::GatewaySwitching
                | CutoverPhase::DestinationActive
                | CutoverPhase::SourceDraining
        )
    {
        actions.push(Action::Crash(Component::Source));
    }
    if request.faults.destination_crash
        && fault_budget
        && !state.destination_crashed
        && state.destination_up
        && matches!(
            state.phase,
            CutoverPhase::DestinationPreparing
                | CutoverPhase::Precopying
                | CutoverPhase::DeltaSyncing
                | CutoverPhase::CutoverRequested
                | CutoverPhase::SourceQuiescing
                | CutoverPhase::SourceFrozen
                | CutoverPhase::FinalDeltaTransferring
                | CutoverPhase::DestinationImporting
                | CutoverPhase::DestinationValidating
                | CutoverPhase::CommitIntentRecorded
                | CutoverPhase::OwnershipCommitted
                | CutoverPhase::GatewaySwitching
                | CutoverPhase::DestinationActive
        )
    {
        actions.push(Action::Crash(Component::Destination));
    }
    if request.faults.gateway_crash
        && fault_budget
        && !state.gateway_crashed
        && state.gateway_up
        && matches!(
            state.phase,
            CutoverPhase::Precopying
                | CutoverPhase::DeltaSyncing
                | CutoverPhase::CommitIntentRecorded
                | CutoverPhase::OwnershipCommitted
                | CutoverPhase::GatewaySwitching
                | CutoverPhase::DestinationActive
        )
    {
        actions.push(Action::Crash(Component::Gateway));
    }
    if request.faults.coordinator_crash
        && fault_budget
        && !state.coordinator_crashed
        && state.coordinator_up
        && matches!(
            state.phase,
            CutoverPhase::Proposed
                | CutoverPhase::CompatibilityValidated
                | CutoverPhase::DestinationPreparing
                | CutoverPhase::Precopying
                | CutoverPhase::DeltaSyncing
                | CutoverPhase::CutoverRequested
                | CutoverPhase::SourceQuiescing
                | CutoverPhase::SourceFrozen
                | CutoverPhase::FinalDeltaTransferring
                | CutoverPhase::DestinationImporting
                | CutoverPhase::DestinationValidating
                | CutoverPhase::CommitIntentRecorded
        )
    {
        actions.push(Action::Crash(Component::Coordinator));
    }
    if !state.gateway_up {
        actions.push(Action::Restart(Component::Gateway));
    }
    if !state.source_up {
        actions.push(Action::Restart(Component::Source));
    }
    if !state.destination_up {
        actions.push(Action::Restart(Component::Destination));
    }
    if request.faults.client_disconnect
        && fault_budget
        && !state.client_disconnected
        && state.client_connected
        && state.phase == CutoverPhase::GatewaySwitching
    {
        actions.push(Action::DisconnectClient);
    }
    if !state.client_connected {
        actions.push(Action::ReconnectClient);
    }
    if request.faults.delayed_acknowledgment
        && fault_budget
        && !state.acknowledgment_delayed
        && state.pending_ack
    {
        actions.push(Action::DelayAcknowledgment);
    }
    if state.pending_ack && state.client_connected {
        actions.push(Action::AcknowledgeClient);
    }
    if state.source_active
        && state.source_up
        && state.gateway_up
        && state.accepted_tokens.len() < usize::from(request.bounds.max_tokens)
    {
        actions.push(Action::EmitSourceToken);
    }
    if state.destination_active
        && state.destination_up
        && state.gateway_up
        && state.accepted_tokens.len() < usize::from(request.bounds.max_tokens)
    {
        actions.push(Action::EmitDestinationToken);
    }
    if !state.accepted_tokens.is_empty() && state.gateway_up {
        actions.push(Action::ReplayLastToken);
    }
    if request.faults.stale_owner_output
        && fault_budget
        && !state.stale_attempted
        && state.gateway_owner_epoch == 2
    {
        actions.push(Action::AttemptStaleOutput);
    }
    if state.gateway_up && state.accepted_tokens.len() < usize::from(request.bounds.max_tokens) {
        actions.push(Action::AttemptGapOutput);
    }
    if request.faults.cancellation
        && fault_budget
        && !state.cancellation_injected
        && matches!(
            state.phase,
            CutoverPhase::DeltaSyncing | CutoverPhase::DestinationActive
        )
    {
        actions.push(Action::Cancel);
    }
    if request.faults.timeout
        && fault_budget
        && state.logical_tick == 0
        && matches!(
            state.phase,
            CutoverPhase::Precopying | CutoverPhase::CommitIntentRecorded
        )
    {
        actions.push(Action::AdvanceTime);
    }
    actions
}

fn injected_fault_count(state: &ProtocolState) -> u8 {
    [
        state.message_duplicated,
        state.message_dropped,
        state.source_crashed,
        state.destination_crashed,
        state.gateway_crashed,
        state.coordinator_crashed,
        state.partition_injected,
        state.stale_attempted,
        state.cancellation_injected,
        state.client_disconnected,
        state.acknowledgment_delayed,
        state.logical_tick > 0,
    ]
    .into_iter()
    .fold(0_u8, |count, injected| {
        count.saturating_add(u8::from(injected))
    })
}

#[allow(clippy::too_many_lines)]
fn apply_action(
    request: &ModelCheckRequest,
    state: &ProtocolState,
    action: &Action,
) -> Option<ProtocolState> {
    let mut next = state.clone();
    match action {
        Action::ValidateCompatibility => {
            next.phase = CutoverPhase::CompatibilityValidated;
        }
        Action::PrepareDestination => {
            next.destination_prepared = true;
            next.phase = CutoverPhase::DestinationPreparing;
        }
        Action::BeginPrecopy => next.phase = CutoverPhase::Precopying,
        Action::Enqueue(kind) => {
            if request.protocol.enforce_queue_bound
                && next.messages.len() >= usize::from(request.bounds.max_messages)
            {
                return None;
            }
            next.messages.push(Message { kind: *kind });
            if *kind == MessageKind::Delta {
                next.delta_enqueued = true;
            }
        }
        Action::Deliver { position } => {
            let index = usize::from(*position);
            if index >= next.messages.len() || next.partitioned {
                return None;
            }
            let message = next.messages.remove(index);
            deliver_message(request, &mut next, message);
        }
        Action::Duplicate { position } => {
            let message = *next.messages.get(usize::from(*position))?;
            if request.protocol.enforce_queue_bound
                && next.messages.len() >= usize::from(request.bounds.max_messages)
            {
                return None;
            }
            next.messages.push(message);
            next.message_duplicated = true;
        }
        Action::Drop { position } => {
            let index = usize::from(*position);
            if index >= next.messages.len() {
                return None;
            }
            let dropped = next.messages.remove(index);
            next.message_dropped = true;
            if dropped.kind == MessageKind::Delta {
                next.delta_enqueued = false;
            }
        }
        Action::StartPartition => {
            next.partitioned = true;
            next.partition_injected = true;
        }
        Action::HealPartition => next.partitioned = false,
        Action::QuiesceSource => {
            next.phase = CutoverPhase::SourceFrozen;
            next.source_fenced = request.protocol.fence_source_before_commit;
            if next.source_fenced {
                next.source_active = false;
            }
        }
        Action::ValidateDestination => {
            next.destination_validated = next.initial_received && next.final_received;
            next.phase = CutoverPhase::DestinationValidating;
        }
        Action::RecordCommitIntent => {
            if (request.protocol.require_validation_before_activation
                && !next.destination_validated)
                || (request.protocol.fence_source_before_commit && !next.source_fenced)
            {
                return None;
            }
            next.commit_intent = true;
            next.phase = CutoverPhase::CommitIntentRecorded;
        }
        Action::CommitOwnership => {
            if !next.commit_intent || !next.coordinator_up {
                return None;
            }
            next.destination_owns = true;
            if request.protocol.atomic_owner_cas {
                next.source_owns = false;
            }
            update_owner_epoch(&mut next, 2);
            next.phase = CutoverPhase::OwnershipCommitted;
        }
        Action::SwitchGateway => {
            update_gateway_epoch(&mut next, 2);
            next.phase = CutoverPhase::GatewaySwitching;
        }
        Action::ActivateDestination => {
            next.destination_active = true;
            next.phase = CutoverPhase::DestinationActive;
        }
        Action::DrainSource => {
            if next.phase == CutoverPhase::DestinationActive {
                next.source_active = false;
                next.phase = CutoverPhase::SourceDraining;
            } else {
                next.source_active = false;
                next.phase = CutoverPhase::Completed;
            }
        }
        Action::EmitSourceToken => {
            let index = next.gateway_next_token;
            accept_token(request, &mut next, 1, index);
        }
        Action::EmitDestinationToken => {
            let index = next.gateway_next_token;
            accept_token(request, &mut next, 2, index);
        }
        Action::ReplayLastToken => {
            let before = next.clone();
            let index = next.gateway_next_token.saturating_sub(1);
            let epoch = next.gateway_owner_epoch;
            accept_token(request, &mut next, epoch, index);
            if !request.protocol.idempotent_replay {
                next.gateway_next_token = next.gateway_next_token.saturating_add(1);
            }
            let expected_only_audit_change = next.duplicate_accepted != before.duplicate_accepted;
            if next.gateway_next_token != before.gateway_next_token
                || next.accepted_tokens != before.accepted_tokens
                || next.highest_watermark != before.highest_watermark
            {
                next.replay_changed_state = true;
            } else if expected_only_audit_change {
                next.replay_changed_state = false;
            }
        }
        Action::AttemptStaleOutput => {
            next.stale_attempted = true;
            let index = next.gateway_next_token;
            accept_token(request, &mut next, 1, index);
        }
        Action::AttemptGapOutput => {
            let index = next.gateway_next_token.saturating_add(1);
            let epoch = next.gateway_owner_epoch;
            accept_token(request, &mut next, epoch, index);
        }
        Action::Crash(component) => crash(&mut next, *component),
        Action::Restart(component) => restart(&mut next, *component),
        Action::DisconnectClient => {
            next.client_connected = false;
            next.client_disconnected = true;
        }
        Action::ReconnectClient => next.client_connected = true,
        Action::DelayAcknowledgment => next.acknowledgment_delayed = true,
        Action::AcknowledgeClient => {
            next.client_ack = next.highest_watermark;
            next.pending_ack = false;
        }
        Action::Cancel => {
            next.cancelled = true;
            next.cancellation_injected = true;
            if next.destination_owns {
                next.source_active = false;
                next.destination_active = false;
                next.phase = CutoverPhase::Completed;
            } else {
                next.phase = CutoverPhase::Aborting;
            }
        }
        Action::AdvanceTime => {
            next.logical_tick = next.logical_tick.saturating_add(1);
            if next.logical_tick >= request.bounds.timeout_ticks {
                next.phase = if next.destination_owns || !next.source_up {
                    CutoverPhase::OperatorRequired
                } else {
                    CutoverPhase::Aborting
                };
            }
        }
        Action::Abort => next.phase = CutoverPhase::Aborting,
        Action::FinishAbort => {
            next.messages.clear();
            next.destination_active = false;
            next.destination_owns = false;
            next.source_owns = true;
            next.source_active = next.source_up;
            next.source_fenced = false;
            update_owner_epoch(&mut next, 1);
            update_gateway_epoch(&mut next, 1);
            next.phase = CutoverPhase::RolledBack;
        }
        Action::RequireOperator => next.phase = CutoverPhase::OperatorRequired,
    }
    Some(next)
}

fn deliver_message(request: &ModelCheckRequest, state: &mut ProtocolState, message: Message) {
    match message.kind {
        MessageKind::InitialSnapshot => {
            state.initial_received = true;
            if matches!(
                state.phase,
                CutoverPhase::Precopying | CutoverPhase::DeltaSyncing
            ) {
                state.phase = CutoverPhase::DeltaSyncing;
            }
        }
        MessageKind::Delta => {
            state.delta_received = true;
            if matches!(
                state.phase,
                CutoverPhase::Precopying | CutoverPhase::DeltaSyncing
            ) {
                state.phase = CutoverPhase::DeltaSyncing;
            }
        }
        MessageKind::FinalDelta => {
            state.final_received = true;
            if matches!(
                state.phase,
                CutoverPhase::SourceFrozen | CutoverPhase::FinalDeltaTransferring
            ) {
                state.phase = CutoverPhase::DestinationImporting;
            }
        }
        MessageKind::GatewaySwitch => {
            if state.gateway_up
                && matches!(
                    state.phase,
                    CutoverPhase::OwnershipCommitted | CutoverPhase::GatewaySwitching
                )
            {
                update_gateway_epoch(state, 2);
                state.phase = CutoverPhase::GatewaySwitching;
            } else if !request.protocol.idempotent_replay {
                state.replay_changed_state = true;
            }
        }
    }
}

fn accept_token(request: &ModelCheckRequest, state: &mut ProtocolState, epoch: u8, index: u8) {
    if !state.gateway_up {
        return;
    }
    if epoch != state.gateway_owner_epoch && request.protocol.reject_stale_output {
        return;
    }
    if epoch != state.gateway_owner_epoch {
        state.stale_output_accepted = true;
    }
    if index < state.gateway_next_token {
        if request.protocol.deduplicate_output {
            return;
        }
        state.duplicate_accepted = true;
    }
    if index > state.gateway_next_token {
        if request.protocol.reject_token_gaps {
            return;
        }
        state.gap_accepted = true;
    }
    state.accepted_tokens.push(index);
    state.accepted_epochs.push(epoch);
    let prior = state.highest_watermark;
    state.highest_watermark = Some(prior.map_or(index, |watermark| watermark.max(index)));
    state.watermark_regressed |= prior.is_some_and(|watermark| index < watermark);
    state.gateway_next_token = index.saturating_add(1);
    state.pending_ack = true;
}

fn update_owner_epoch(state: &mut ProtocolState, epoch: u8) {
    state.epoch_regressed |= epoch < state.highest_owner_epoch;
    state.state_owner_epoch = epoch;
    state.highest_owner_epoch = state.highest_owner_epoch.max(epoch);
}

fn update_gateway_epoch(state: &mut ProtocolState, epoch: u8) {
    state.epoch_regressed |= epoch < state.highest_owner_epoch;
    state.gateway_owner_epoch = epoch;
    state.highest_owner_epoch = state.highest_owner_epoch.max(epoch);
}

fn crash(state: &mut ProtocolState, component: Component) {
    match component {
        Component::Source => {
            state.source_crashed = true;
            state.source_up = false;
            state.source_active = false;
            state.phase = CutoverPhase::SourceLost;
        }
        Component::Destination => {
            state.destination_crashed = true;
            state.destination_up = false;
            state.destination_active = false;
            state.phase = CutoverPhase::DestinationLost;
        }
        Component::Gateway => {
            state.gateway_crashed = true;
            state.gateway_up = false;
        }
        Component::Coordinator => {
            state.coordinator_crashed = true;
            state.coordinator_up = false;
            state.phase = CutoverPhase::CoordinatorUnavailable;
        }
    }
}

fn restart(state: &mut ProtocolState, component: Component) {
    match component {
        Component::Gateway => state.gateway_up = true,
        Component::Coordinator => {
            state.coordinator_up = true;
            state.phase = if state.commit_intent {
                CutoverPhase::CommitIntentRecorded
            } else {
                CutoverPhase::Aborting
            };
        }
        Component::Source => state.source_up = true,
        Component::Destination => state.destination_up = true,
    }
}

fn violated_invariants(
    request: &ModelCheckRequest,
    state: &ProtocolState,
    deadlocked: bool,
) -> Vec<InvariantId> {
    let mut violations = Vec::new();
    if state.source_owns && state.destination_owns {
        violations.push(InvariantId::UniqueStateOwner);
    }
    let accepted_source = state.source_active && state.gateway_owner_epoch == 1;
    let accepted_destination = state.destination_active && state.gateway_owner_epoch == 2;
    if accepted_source && accepted_destination {
        violations.push(InvariantId::UniqueAcceptedOutputOwner);
    }
    if state.epoch_regressed {
        violations.push(InvariantId::MonotonicOwnerEpoch);
    }
    if state.watermark_regressed {
        violations.push(InvariantId::MonotonicTokenWatermark);
    }
    if state.stale_output_accepted {
        violations.push(InvariantId::StaleEpochCannotCommitOutput);
    }
    if state.duplicate_accepted {
        violations.push(InvariantId::NoAcceptedDuplicateToken);
    }
    if state.gap_accepted {
        violations.push(InvariantId::NoAcceptedTokenGap);
    }
    if state.destination_active && !state.destination_validated {
        violations.push(InvariantId::DestinationValidatedBeforeActivation);
    }
    if (state.source_fenced && state.source_active)
        || (state.destination_owns && !state.source_fenced)
    {
        violations.push(InvariantId::SourceInactiveAfterFence);
    }
    if state.phase == CutoverPhase::RolledBack
        && (!state.source_up || !state.source_active || !state.source_owns)
    {
        violations.push(InvariantId::PrecommitAbortPreservesSource);
    }
    if state.phase == CutoverPhase::Completed
        && !state.cancelled
        && (!state.destination_owns || !state.destination_active || state.gateway_owner_epoch != 2)
    {
        violations.push(InvariantId::CompletedMigrationPreservesDestination);
    }
    if deadlocked {
        violations.push(InvariantId::NoDeadlock);
    }
    if state.messages.len() > usize::from(request.bounds.max_messages) {
        violations.push(InvariantId::BoundedQueues);
    }
    if state.replay_changed_state {
        violations.push(InvariantId::IdempotentReplay);
    }
    violations
}

fn states_reaching_terminal(
    nodes: &HashMap<ProtocolState, Node>,
    edges: &HashMap<ProtocolState, Vec<ProtocolState>>,
) -> HashSet<ProtocolState> {
    let mut reverse: HashMap<ProtocolState, Vec<ProtocolState>> = HashMap::new();
    for (source, destinations) in edges {
        for destination in destinations {
            reverse
                .entry(destination.clone())
                .or_default()
                .push(source.clone());
        }
    }
    let mut reachable = nodes
        .keys()
        .filter(|state| state.is_terminal())
        .cloned()
        .collect::<HashSet<_>>();
    let mut queue = reachable.iter().cloned().collect::<VecDeque<_>>();
    while let Some(state) = queue.pop_front() {
        for predecessor in reverse.get(&state).into_iter().flatten() {
            if reachable.insert(predecessor.clone()) {
                queue.push_back(predecessor.clone());
            }
        }
    }
    reachable
}

fn trace_for(
    invariant: InvariantId,
    failed: &ProtocolState,
    nodes: &HashMap<ProtocolState, Node>,
) -> CounterexampleTrace {
    let mut states = Vec::new();
    let mut cursor = failed.clone();
    while let Some(node) = nodes.get(&cursor) {
        if let Some(action) = &node.action {
            states.push((action.clone(), cursor.clone()));
        }
        if let Some(predecessor) = &node.predecessor {
            cursor = predecessor.clone();
        } else {
            break;
        }
    }
    states.reverse();
    CounterexampleTrace {
        invariant,
        minimized: true,
        steps: states
            .into_iter()
            .enumerate()
            .map(|(index, (action, state))| TraceStep {
                ordinal: u16::try_from(index).unwrap_or(u16::MAX),
                action,
                state_fingerprint: fingerprint(&state),
                state: state.summary(),
            })
            .collect(),
    }
}

/// Replay and authenticate a counterexample trace.
///
/// # Errors
///
/// Rejects unavailable actions, altered fingerprints, summaries, or ordinals.
pub fn replay_counterexample(
    request: &ModelCheckRequest,
    trace: &CounterexampleTrace,
) -> Result<(), ReplayError> {
    request
        .validate()
        .map_err(|diagnostics| ReplayError(format!("invalid request: {diagnostics:?}")))?;
    let mut state = ProtocolState::initial();
    for (ordinal, step) in trace.steps.iter().enumerate() {
        if step.ordinal != u16::try_from(ordinal).unwrap_or(u16::MAX) {
            return Err(ReplayError("trace ordinal mismatch".to_owned()));
        }
        let available = enabled_actions(request, &state);
        if !available.contains(&step.action) {
            return Err(ReplayError(format!(
                "action is not enabled at step {ordinal}: {:?}",
                step.action
            )));
        }
        state = apply_action(request, &state, &step.action)
            .ok_or_else(|| ReplayError(format!("action failed at step {ordinal}")))?;
        if fingerprint(&state) != step.state_fingerprint || state.summary() != step.state {
            return Err(ReplayError(format!(
                "state evidence mismatch at step {ordinal}"
            )));
        }
    }
    let deadlocked = successors(request, &state).is_empty() && !state.is_terminal();
    if !violated_invariants(request, &state, deadlocked).contains(&trace.invariant) {
        return Err(ReplayError(
            "trace does not end in its declared invariant violation".to_owned(),
        ));
    }
    Ok(())
}

/// Validate result metadata and replay every attached counterexample.
///
/// # Errors
///
/// Returns all result-integrity problems.
pub fn validate_result(
    request: &ModelCheckRequest,
    result: &ModelCheckResult,
) -> Result<(), Vec<ReplayError>> {
    let mut errors = Vec::new();
    if result.schema_version != RESULT_SCHEMA_VERSION
        || result.model_version != request.model_version
        || result.seed != request.seed
        || !result.scope.bounded_not_universal_proof
        || result.scope.bounds != request.bounds
        || result.scope.assumptions != request.assumptions
        || result.scope.claim != "bounded explicit-state exploration; not a universal proof"
    {
        errors.push(ReplayError("result scope metadata mismatch".to_owned()));
    }
    if result.scope.complete_within_bounds != result.scope.truncated_by.is_empty()
        || result.states_explored == 0
        || result.states_explored > request.bounds.max_states
        || result.max_depth_reached > request.bounds.max_depth
    {
        errors.push(ReplayError(
            "result exploration accounting is inconsistent".to_owned(),
        ));
    }
    if result.invariants.len() != InvariantId::ALL.len() {
        errors.push(ReplayError(
            "invariant outcome set is incomplete".to_owned(),
        ));
    }
    let mut seen_invariants = BTreeSet::new();
    for outcome in &result.invariants {
        if !seen_invariants.insert(outcome.invariant)
            || outcome.passed != (outcome.status == CheckStatus::Passed)
            || outcome.description != invariant_description(outcome.invariant)
        {
            errors.push(ReplayError(
                "invariant outcome metadata is inconsistent".to_owned(),
            ));
        }
        match (&outcome.status, &outcome.counterexample) {
            (CheckStatus::Failed, Some(trace)) => {
                if trace.invariant != outcome.invariant {
                    errors.push(ReplayError("counterexample invariant mismatch".to_owned()));
                } else if let Err(error) = replay_counterexample(request, trace) {
                    errors.push(error);
                }
            }
            (CheckStatus::Failed, None) => {
                errors.push(ReplayError("failed invariant lacks a trace".to_owned()));
            }
            (_, Some(_)) => {
                errors.push(ReplayError(
                    "non-failed invariant has a counterexample".to_owned(),
                ));
            }
            (_, None) => {}
        }
    }
    if seen_invariants != InvariantId::ALL.into_iter().collect::<BTreeSet<_>>() {
        errors.push(ReplayError(
            "invariant identifiers are missing or duplicated".to_owned(),
        ));
    }
    let expected_status = if result
        .invariants
        .iter()
        .any(|outcome| outcome.status == CheckStatus::Failed)
    {
        CheckStatus::Failed
    } else if result.scope.complete_within_bounds {
        CheckStatus::Passed
    } else {
        CheckStatus::Inconclusive
    };
    if result.status != expected_status {
        errors.push(ReplayError(
            "aggregate result status does not match invariant outcomes".to_owned(),
        ));
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

fn order_successors(seed: u64, successors: &mut [(Action, ProtocolState)]) {
    successors.sort_by_key(|(action, state)| {
        let action_bytes = serde_json::to_vec(action).unwrap_or_default();
        let state_bytes = serde_json::to_vec(state).unwrap_or_default();
        (fnv1a64(seed, &action_bytes), fnv1a64(seed, &state_bytes))
    });
}

fn fingerprint(state: &ProtocolState) -> String {
    let bytes = serde_json::to_vec(state).unwrap_or_default();
    format!("fnv1a64:{:016x}", fnv1a64(0xcbf2_9ce4_8422_2325, &bytes))
}

fn fnv1a64(seed: u64, bytes: &[u8]) -> u64 {
    bytes.iter().fold(seed, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn invariant_description(invariant: InvariantId) -> &'static str {
    match invariant {
        InvariantId::UniqueStateOwner => "at most one runtime owns mutable session state",
        InvariantId::UniqueAcceptedOutputOwner => {
            "at most one active runtime matches the gateway output epoch"
        }
        InvariantId::MonotonicOwnerEpoch => "state and gateway owner epochs never regress",
        InvariantId::MonotonicTokenWatermark => "accepted token watermarks never regress",
        InvariantId::StaleEpochCannotCommitOutput => {
            "a stale owner epoch cannot cross the gateway acceptance boundary"
        }
        InvariantId::NoAcceptedDuplicateToken => "no token index is accepted twice by the gateway",
        InvariantId::NoAcceptedTokenGap => "gateway acceptance does not skip a token index",
        InvariantId::DestinationValidatedBeforeActivation => {
            "destination activation requires validated imported state"
        }
        InvariantId::SourceInactiveAfterFence => {
            "ownership commit requires a writer fence and the fenced source stays inactive"
        }
        InvariantId::PrecommitAbortPreservesSource => {
            "a completed pre-commit rollback leaves a valid source owner"
        }
        InvariantId::CompletedMigrationPreservesDestination => {
            "a non-cancelled completed migration leaves a valid destination owner"
        }
        InvariantId::ExplicitTerminalOrRecoverable => {
            "every reachable bounded state can reach a declared terminal outcome"
        }
        InvariantId::NoDeadlock => "no non-terminal reachable state is deadlocked",
        InvariantId::BoundedQueues => "transport queues remain within the declared bound",
        InvariantId::IdempotentReplay => "duplicate protocol messages and outputs are idempotent",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guarded_protocol_passes_complete_bounded_fault_exploration() {
        let request = ModelCheckRequest::safe(17);
        let result = check(&request).unwrap_or_else(|errors| panic!("request: {errors:?}"));
        assert_eq!(result.status, CheckStatus::Passed);
        assert!(result.scope.complete_within_bounds);
        assert!(result.scope.bounded_not_universal_proof);
        assert!(result.states_explored > 100);
        assert!(result.transitions_explored > result.states_explored);
        validate_result(&request, &result).unwrap_or_else(|errors| panic!("result: {errors:?}"));
    }

    #[test]
    fn unsafe_owner_handoff_has_minimal_replayable_trace() {
        let mut request = ModelCheckRequest::safe(23);
        request.protocol.atomic_owner_cas = false;
        let result = check(&request).unwrap_or_else(|errors| panic!("request: {errors:?}"));
        let trace = result
            .invariants
            .iter()
            .find(|outcome| outcome.invariant == InvariantId::UniqueStateOwner)
            .and_then(|outcome| outcome.counterexample.as_ref())
            .unwrap_or_else(|| panic!("missing ownership trace"));
        assert!(trace.minimized);
        replay_counterexample(&request, trace).unwrap_or_else(|error| panic!("replay: {error}"));
    }

    #[test]
    fn stale_output_and_duplicate_mutations_are_detected_independently() {
        let mut stale = ModelCheckRequest::safe(29);
        stale.protocol.reject_stale_output = false;
        let stale_result = check(&stale).unwrap_or_else(|errors| panic!("request: {errors:?}"));
        assert!(stale_result.invariants.iter().any(|outcome| {
            outcome.invariant == InvariantId::StaleEpochCannotCommitOutput
                && outcome.status == CheckStatus::Failed
        }));

        let mut duplicate = ModelCheckRequest::safe(31);
        duplicate.protocol.deduplicate_output = false;
        let duplicate_result =
            check(&duplicate).unwrap_or_else(|errors| panic!("request: {errors:?}"));
        assert!(duplicate_result.invariants.iter().any(|outcome| {
            outcome.invariant == InvariantId::NoAcceptedDuplicateToken
                && outcome.status == CheckStatus::Failed
        }));

        let mut gaps = ModelCheckRequest::safe(33);
        gaps.protocol.reject_token_gaps = false;
        let gap_result = check(&gaps).unwrap_or_else(|errors| panic!("request: {errors:?}"));
        assert!(gap_result.invariants.iter().any(|outcome| {
            outcome.invariant == InvariantId::NoAcceptedTokenGap
                && outcome.status == CheckStatus::Failed
        }));

        let mut validation = ModelCheckRequest::safe(35);
        validation.protocol.require_validation_before_activation = false;
        let validation_result =
            check(&validation).unwrap_or_else(|errors| panic!("request: {errors:?}"));
        assert!(validation_result.invariants.iter().any(|outcome| {
            outcome.invariant == InvariantId::DestinationValidatedBeforeActivation
                && outcome.status == CheckStatus::Failed
        }));

        let mut fencing = ModelCheckRequest::safe(36);
        fencing.protocol.fence_source_before_commit = false;
        let fencing_result = check(&fencing).unwrap_or_else(|errors| panic!("request: {errors:?}"));
        assert!(fencing_result.invariants.iter().any(|outcome| {
            outcome.invariant == InvariantId::SourceInactiveAfterFence
                && outcome.status == CheckStatus::Failed
        }));
    }

    #[test]
    fn altered_counterexample_evidence_is_rejected() {
        let mut request = ModelCheckRequest::safe(37);
        request.protocol.atomic_owner_cas = false;
        let result = check(&request).unwrap_or_else(|errors| panic!("request: {errors:?}"));
        let mut trace = result
            .invariants
            .iter()
            .find(|outcome| outcome.invariant == InvariantId::UniqueStateOwner)
            .and_then(|outcome| outcome.counterexample.clone())
            .unwrap_or_else(|| panic!("missing trace"));
        let step = trace
            .steps
            .last_mut()
            .unwrap_or_else(|| panic!("empty trace"));
        step.state_fingerprint = "fnv1a64:0000000000000000".to_owned();
        assert!(replay_counterexample(&request, &trace).is_err());
    }

    #[test]
    fn altered_result_status_and_scope_are_rejected() {
        let request = ModelCheckRequest::safe(41);
        let result = check(&request).unwrap_or_else(|errors| panic!("request: {errors:?}"));

        let mut altered_status = result.clone();
        altered_status.status = if result.status == CheckStatus::Failed {
            CheckStatus::Passed
        } else {
            CheckStatus::Failed
        };
        assert!(validate_result(&request, &altered_status).is_err());

        let mut altered_scope = result;
        altered_scope
            .scope
            .assumptions
            .push("unrecorded assumption".to_owned());
        assert!(validate_result(&request, &altered_scope).is_err());
    }
}
