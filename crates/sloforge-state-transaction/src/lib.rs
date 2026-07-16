//! Canonical ownership and client-visible commit protocol for Continuum.
//!
//! The crate deliberately models compare-and-swap state without implementing a
//! consensus system. A durable deployment persists [`CoordinatorSnapshot`]
//! through an embedded database or a distributed CAS provider before exposing
//! a successful mutation.

use std::collections::{BTreeMap, BTreeSet};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Current transaction wire schema.
pub const TRANSACTION_SCHEMA_VERSION: &str = "continuum.state-transaction.v1";
/// Maximum persisted transitions in one transaction journal.
pub const MAX_JOURNAL_ENTRIES: usize = 512;
/// Maximum hashes referenced directly by one transaction record.
pub const MAX_STATE_HASHES: usize = 8_192;
/// Maximum gateway token records retained by this bounded reference ledger.
pub const MAX_GATEWAY_TOKENS: usize = 65_536;
/// Maximum sessions retained by the bounded reference coordinator.
pub const MAX_COORDINATOR_SESSIONS: usize = 65_536;
/// Maximum transaction identifiers retained for replay protection.
pub const MAX_COORDINATOR_TRANSACTIONS: usize = 262_144;

/// Stable phase names for a migration cutover.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum CutoverPhase {
    Proposed,
    CompatibilityValidated,
    DestinationPreparing,
    Precopying,
    DeltaSyncing,
    CutoverRequested,
    SourceQuiescing,
    SourceFrozen,
    FinalDeltaTransferring,
    DestinationImporting,
    DestinationValidating,
    CommitIntentRecorded,
    OwnershipCommitted,
    GatewaySwitching,
    DestinationActive,
    SourceDraining,
    Completed,
    Rejected,
    Aborting,
    RolledBack,
    FailedBeforeCommit,
    FailedAfterCommit,
    DestinationLost,
    SourceLost,
    CoordinatorUnavailable,
    OperatorRequired,
}

impl CutoverPhase {
    /// Whether this phase ends automatic processing of this transaction.
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Completed
                | Self::Rejected
                | Self::RolledBack
                | Self::FailedBeforeCommit
                | Self::FailedAfterCommit
                | Self::OperatorRequired
        )
    }

    /// Whether ownership has crossed the transaction's irreversible boundary.
    #[must_use]
    pub const fn is_post_commit(self) -> bool {
        matches!(
            self,
            Self::OwnershipCommitted
                | Self::GatewaySwitching
                | Self::DestinationActive
                | Self::SourceDraining
                | Self::Completed
                | Self::FailedAfterCommit
        )
    }

    /// Return whether `next` is a legal direct transition.
    #[must_use]
    pub fn allows(self, next: Self) -> bool {
        match self {
            Self::Proposed => matches!(next, Self::CompatibilityValidated | Self::Rejected),
            Self::CompatibilityValidated => {
                matches!(next, Self::DestinationPreparing | Self::Aborting)
            }
            Self::DestinationPreparing => matches!(
                next,
                Self::Precopying | Self::CutoverRequested | Self::DestinationLost | Self::Aborting
            ),
            Self::Precopying => {
                matches!(
                    next,
                    Self::DeltaSyncing | Self::CutoverRequested | Self::Aborting
                )
            }
            Self::DeltaSyncing => {
                matches!(
                    next,
                    Self::DeltaSyncing | Self::CutoverRequested | Self::Aborting
                )
            }
            Self::CutoverRequested => matches!(next, Self::SourceQuiescing | Self::Aborting),
            Self::SourceQuiescing => {
                matches!(next, Self::SourceFrozen | Self::SourceLost | Self::Aborting)
            }
            Self::SourceFrozen => matches!(
                next,
                Self::FinalDeltaTransferring | Self::DestinationImporting | Self::Aborting
            ),
            Self::FinalDeltaTransferring => {
                matches!(
                    next,
                    Self::DestinationImporting | Self::DestinationLost | Self::Aborting
                )
            }
            Self::DestinationImporting => matches!(
                next,
                Self::DestinationValidating | Self::DestinationLost | Self::Aborting
            ),
            Self::DestinationValidating => matches!(
                next,
                Self::CommitIntentRecorded | Self::DestinationLost | Self::Aborting
            ),
            Self::CommitIntentRecorded => matches!(
                next,
                Self::OwnershipCommitted
                    | Self::CoordinatorUnavailable
                    | Self::DestinationLost
                    | Self::Aborting
            ),
            Self::OwnershipCommitted => matches!(
                next,
                Self::GatewaySwitching
                    | Self::DestinationLost
                    | Self::SourceLost
                    | Self::FailedAfterCommit
            ),
            Self::GatewaySwitching => matches!(
                next,
                Self::DestinationActive | Self::DestinationLost | Self::FailedAfterCommit
            ),
            Self::DestinationActive => matches!(
                next,
                Self::SourceDraining | Self::DestinationLost | Self::FailedAfterCommit
            ),
            Self::SourceDraining => {
                matches!(
                    next,
                    Self::Completed | Self::SourceLost | Self::FailedAfterCommit
                )
            }
            Self::Aborting => matches!(next, Self::RolledBack | Self::FailedBeforeCommit),
            Self::DestinationLost => {
                matches!(
                    next,
                    Self::Aborting
                        | Self::FailedBeforeCommit
                        | Self::FailedAfterCommit
                        | Self::OperatorRequired
                )
            }
            Self::SourceLost => matches!(
                next,
                Self::FailedBeforeCommit | Self::FailedAfterCommit | Self::OperatorRequired
            ),
            Self::CoordinatorUnavailable => matches!(
                next,
                Self::CommitIntentRecorded | Self::Aborting | Self::OperatorRequired
            ),
            Self::Completed
            | Self::Rejected
            | Self::RolledBack
            | Self::FailedBeforeCommit
            | Self::FailedAfterCommit
            | Self::OperatorRequired => false,
        }
    }
}

/// The authoritative lease for one logical session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SessionLease {
    pub session_id: String,
    pub owner_runtime: String,
    pub owner_epoch: u64,
    pub fencing_token: u64,
    pub expires_at_tick: u64,
    pub coordinator_version: u64,
    pub last_committed_state_version: u64,
    pub last_committed_token_index: Option<u64>,
}

impl SessionLease {
    /// Test whether a writer still has authority at `now_tick`.
    #[must_use]
    pub fn authorizes(&self, owner: &str, epoch: u64, fencing_token: u64, now_tick: u64) -> bool {
        self.owner_runtime == owner
            && self.owner_epoch == epoch
            && self.fencing_token == fencing_token
            && now_tick < self.expires_at_tick
    }
}

/// One durable phase transition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JournalEntry {
    pub sequence: u32,
    pub from: CutoverPhase,
    pub to: CutoverPhase,
    pub logical_tick: u64,
    pub reason_code: String,
}

/// Persistent state for one migration attempt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StateTransaction {
    pub schema_version: String,
    pub transaction_id: String,
    pub session_id: String,
    pub source_owner: String,
    pub destination_candidate: String,
    pub source_epoch: u64,
    pub proposed_destination_epoch: u64,
    pub source_fencing_token: u64,
    pub phase: CutoverPhase,
    pub commit_watermark: Option<u64>,
    pub rollback_watermark: Option<u64>,
    pub state_hashes: Vec<String>,
    pub destination_validated: bool,
    pub source_fenced: bool,
    pub cutover_evidence_recorded: bool,
    pub timeout_tick: u64,
    pub failure_reason: Option<String>,
    pub journal: Vec<JournalEntry>,
}

impl StateTransaction {
    /// Apply a legal, idempotent state transition.
    ///
    /// # Errors
    ///
    /// Returns a typed error for illegal order, an expired transaction, or a
    /// journal exceeding the fixed bound.
    pub fn transition(
        &mut self,
        next: CutoverPhase,
        logical_tick: u64,
        reason_code: &str,
    ) -> Result<TransitionEffect, ProtocolError> {
        if reason_code.is_empty() || reason_code.len() > 128 {
            return Err(ProtocolError::InvalidReasonCode);
        }
        if self.phase == next {
            let exact_replay = self.journal.last().is_some_and(|entry| {
                entry.to == next
                    && entry.logical_tick == logical_tick
                    && entry.reason_code == reason_code
            });
            return if exact_replay {
                Ok(TransitionEffect::Replay)
            } else {
                Err(ProtocolError::IdempotencyConflict)
            };
        }
        if logical_tick >= self.timeout_tick {
            return Err(ProtocolError::TransactionExpired {
                transaction_id: self.transaction_id.clone(),
            });
        }
        let ownership_was_committed = self.phase.is_post_commit()
            || self
                .journal
                .iter()
                .any(|entry| entry.to == CutoverPhase::OwnershipCommitted);
        if ownership_was_committed
            && matches!(
                next,
                CutoverPhase::Aborting
                    | CutoverPhase::RolledBack
                    | CutoverPhase::FailedBeforeCommit
            )
        {
            return Err(ProtocolError::PostCommitRollbackForbidden);
        }
        if !ownership_was_committed && next == CutoverPhase::FailedAfterCommit {
            return Err(ProtocolError::IllegalTransition {
                from: self.phase,
                to: next,
            });
        }
        if !self.phase.allows(next) {
            return Err(ProtocolError::IllegalTransition {
                from: self.phase,
                to: next,
            });
        }
        if self.journal.len() >= MAX_JOURNAL_ENTRIES {
            return Err(ProtocolError::JournalFull);
        }
        if next == CutoverPhase::CommitIntentRecorded
            && (!self.destination_validated
                || !self.source_fenced
                || !self.cutover_evidence_recorded)
        {
            return Err(ProtocolError::CommitPrerequisiteMissing);
        }
        let from = self.phase;
        let sequence = u32::try_from(self.journal.len())
            .map_err(|_| ProtocolError::JournalFull)?
            .saturating_add(1);
        self.phase = next;
        if matches!(
            next,
            CutoverPhase::Rejected
                | CutoverPhase::FailedBeforeCommit
                | CutoverPhase::FailedAfterCommit
                | CutoverPhase::OperatorRequired
        ) {
            self.failure_reason = Some(reason_code.to_owned());
        }
        self.journal.push(JournalEntry {
            sequence,
            from,
            to: next,
            logical_tick,
            reason_code: reason_code.to_owned(),
        });
        Ok(TransitionEffect::Applied)
    }
}

/// Result of an idempotent transition application.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TransitionEffect {
    Applied,
    Replay,
}

/// Complete serializable coordinator state for durable persistence.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CoordinatorSnapshot {
    pub version: u64,
    pub leases: BTreeMap<String, SessionLease>,
    pub transactions: BTreeMap<String, StateTransaction>,
    pub used_transaction_ids: BTreeSet<String>,
}

fn journal_is_valid(transaction: &StateTransaction) -> bool {
    let mut expected_phase = CutoverPhase::Proposed;
    let mut prior_tick = 0_u64;
    let mut ownership_was_committed = false;
    for (index, entry) in transaction.journal.iter().enumerate() {
        if entry.sequence != u32::try_from(index).unwrap_or(u32::MAX).saturating_add(1)
            || entry.from != expected_phase
            || !entry.from.allows(entry.to)
            || (ownership_was_committed
                && matches!(
                    entry.to,
                    CutoverPhase::Aborting
                        | CutoverPhase::RolledBack
                        | CutoverPhase::FailedBeforeCommit
                ))
            || (!ownership_was_committed && entry.to == CutoverPhase::FailedAfterCommit)
            || entry.logical_tick < prior_tick
            || entry.logical_tick >= transaction.timeout_tick
            || entry.reason_code.is_empty()
            || entry.reason_code.len() > 128
        {
            return false;
        }
        ownership_was_committed |= entry.to == CutoverPhase::OwnershipCommitted;
        expected_phase = entry.to;
        prior_tick = entry.logical_tick;
    }
    expected_phase == transaction.phase
        && (!matches!(
            transaction.phase,
            CutoverPhase::CommitIntentRecorded
                | CutoverPhase::OwnershipCommitted
                | CutoverPhase::GatewaySwitching
                | CutoverPhase::DestinationActive
                | CutoverPhase::SourceDraining
                | CutoverPhase::Completed
                | CutoverPhase::FailedAfterCommit
        ) || transaction.cutover_evidence_recorded)
}

/// Deterministic reference CAS coordinator.
#[derive(Debug, Clone, Default)]
pub struct OwnershipCoordinator {
    snapshot: CoordinatorSnapshot,
}

impl OwnershipCoordinator {
    /// Recover from a previously persisted snapshot after validating it.
    ///
    /// # Errors
    ///
    /// Rejects inconsistent epochs, transaction IDs, and coordinator versions.
    pub fn recover(snapshot: CoordinatorSnapshot) -> Result<Self, ProtocolError> {
        for (session_id, lease) in &snapshot.leases {
            if session_id != &lease.session_id
                || lease.owner_epoch == 0
                || lease.fencing_token == 0
                || lease.coordinator_version > snapshot.version
            {
                return Err(ProtocolError::InvalidSnapshot);
            }
        }
        let mut active_sessions = BTreeSet::new();
        for (transaction_id, transaction) in &snapshot.transactions {
            if transaction_id != &transaction.transaction_id
                || !snapshot.used_transaction_ids.contains(transaction_id)
                || transaction.schema_version != TRANSACTION_SCHEMA_VERSION
                || transaction.transaction_id.is_empty()
                || transaction.transaction_id.len() > 256
                || transaction.session_id.is_empty()
                || transaction.session_id.len() > 256
                || transaction.source_owner.is_empty()
                || transaction.source_owner.len() > 256
                || transaction.destination_candidate.is_empty()
                || transaction.destination_candidate.len() > 256
                || transaction.source_fencing_token == 0
                || transaction.source_epoch.checked_add(1)
                    != Some(transaction.proposed_destination_epoch)
                || transaction.journal.len() > MAX_JOURNAL_ENTRIES
                || transaction.state_hashes.len() > MAX_STATE_HASHES
                || (transaction.destination_validated && transaction.state_hashes.is_empty())
                || watermark_exceeds(transaction.rollback_watermark, transaction.commit_watermark)
                || !journal_is_valid(transaction)
            {
                return Err(ProtocolError::InvalidSnapshot);
            }
            if !transaction.phase.is_terminal() {
                if !active_sessions.insert(transaction.session_id.clone()) {
                    return Err(ProtocolError::InvalidSnapshot);
                }
                let lease = snapshot
                    .leases
                    .get(&transaction.session_id)
                    .ok_or(ProtocolError::InvalidSnapshot)?;
                let lease_matches = if transaction.phase.is_post_commit()
                    || transaction
                        .journal
                        .iter()
                        .any(|entry| entry.to == CutoverPhase::OwnershipCommitted)
                {
                    lease.owner_runtime == transaction.destination_candidate
                        && lease.owner_epoch == transaction.proposed_destination_epoch
                } else {
                    lease.owner_runtime == transaction.source_owner
                        && lease.owner_epoch == transaction.source_epoch
                        && lease.fencing_token == transaction.source_fencing_token
                };
                if !lease_matches {
                    return Err(ProtocolError::InvalidSnapshot);
                }
            }
        }
        if snapshot.leases.len() > MAX_COORDINATOR_SESSIONS
            || snapshot.transactions.len() > MAX_COORDINATOR_TRANSACTIONS
            || snapshot.used_transaction_ids.len() > MAX_COORDINATOR_TRANSACTIONS
        {
            return Err(ProtocolError::InvalidSnapshot);
        }
        Ok(Self { snapshot })
    }

    /// Return a clone suitable for atomic persistence by the caller.
    #[must_use]
    pub fn snapshot(&self) -> CoordinatorSnapshot {
        self.snapshot.clone()
    }

    /// Register the initial owner of a session.
    ///
    /// # Errors
    ///
    /// Rejects duplicate sessions or invalid identity/lease bounds.
    pub fn register_session(
        &mut self,
        session_id: &str,
        owner_runtime: &str,
        expires_at_tick: u64,
    ) -> Result<SessionLease, ProtocolError> {
        if session_id.is_empty()
            || session_id.len() > 256
            || owner_runtime.is_empty()
            || owner_runtime.len() > 256
            || expires_at_tick == 0
        {
            return Err(ProtocolError::InvalidIdentity);
        }
        if self.snapshot.leases.len() >= MAX_COORDINATOR_SESSIONS {
            return Err(ProtocolError::CoordinatorCapacityReached);
        }
        if self.snapshot.leases.contains_key(session_id) {
            return Err(ProtocolError::SessionExists(session_id.to_owned()));
        }
        self.snapshot.version = self
            .snapshot
            .version
            .checked_add(1)
            .ok_or(ProtocolError::VersionExhausted)?;
        let lease = SessionLease {
            session_id: session_id.to_owned(),
            owner_runtime: owner_runtime.to_owned(),
            owner_epoch: 1,
            fencing_token: 1,
            expires_at_tick,
            coordinator_version: self.snapshot.version,
            last_committed_state_version: 0,
            last_committed_token_index: None,
        };
        self.snapshot
            .leases
            .insert(session_id.to_owned(), lease.clone());
        Ok(lease)
    }

    /// Begin a migration against an exact source lease generation.
    #[allow(clippy::too_many_arguments)]
    ///
    /// # Errors
    ///
    /// Rejects stale leases and any reused transaction ID.
    pub fn begin_transaction(
        &mut self,
        transaction_id: &str,
        session_id: &str,
        source_owner: &str,
        destination_candidate: &str,
        source_epoch: u64,
        source_fencing_token: u64,
        timeout_tick: u64,
        now_tick: u64,
    ) -> Result<StateTransaction, ProtocolError> {
        if transaction_id.is_empty()
            || transaction_id.len() > 256
            || destination_candidate.is_empty()
            || destination_candidate.len() > 256
        {
            return Err(ProtocolError::InvalidIdentity);
        }
        if self.snapshot.used_transaction_ids.contains(transaction_id) {
            return Err(ProtocolError::TransactionIdReused(
                transaction_id.to_owned(),
            ));
        }
        if self.snapshot.transactions.len() >= MAX_COORDINATOR_TRANSACTIONS {
            return Err(ProtocolError::CoordinatorCapacityReached);
        }
        if self.snapshot.transactions.values().any(|transaction| {
            transaction.session_id == session_id && !transaction.phase.is_terminal()
        }) {
            return Err(ProtocolError::ActiveTransaction(session_id.to_owned()));
        }
        let lease = self
            .snapshot
            .leases
            .get(session_id)
            .ok_or_else(|| ProtocolError::UnknownSession(session_id.to_owned()))?;
        if !lease.authorizes(source_owner, source_epoch, source_fencing_token, now_tick) {
            return Err(ProtocolError::StaleFence);
        }
        if timeout_tick <= now_tick {
            return Err(ProtocolError::TransactionExpired {
                transaction_id: transaction_id.to_owned(),
            });
        }
        let proposed_destination_epoch = source_epoch
            .checked_add(1)
            .ok_or(ProtocolError::VersionExhausted)?;
        let new_version = self
            .snapshot
            .version
            .checked_add(1)
            .ok_or(ProtocolError::VersionExhausted)?;
        self.snapshot
            .used_transaction_ids
            .insert(transaction_id.to_owned());
        let transaction = StateTransaction {
            schema_version: TRANSACTION_SCHEMA_VERSION.to_owned(),
            transaction_id: transaction_id.to_owned(),
            session_id: session_id.to_owned(),
            source_owner: source_owner.to_owned(),
            destination_candidate: destination_candidate.to_owned(),
            source_epoch,
            proposed_destination_epoch,
            source_fencing_token,
            phase: CutoverPhase::Proposed,
            commit_watermark: None,
            rollback_watermark: lease.last_committed_token_index,
            state_hashes: Vec::new(),
            destination_validated: false,
            source_fenced: false,
            cutover_evidence_recorded: false,
            timeout_tick,
            failure_reason: None,
            journal: Vec::new(),
        };
        self.snapshot
            .transactions
            .insert(transaction_id.to_owned(), transaction.clone());
        self.snapshot.version = new_version;
        Ok(transaction)
    }

    /// Read a transaction from the current coordinator generation.
    #[must_use]
    pub fn transaction(&self, transaction_id: &str) -> Option<&StateTransaction> {
        self.snapshot.transactions.get(transaction_id)
    }

    /// Persist cutover validation/fencing evidence before commit intent.
    ///
    /// # Errors
    ///
    /// Rejects missing transactions, invalid phases, or unbounded/empty hashes.
    pub fn record_cutover_evidence(
        &mut self,
        transaction_id: &str,
        destination_validated: bool,
        source_fenced: bool,
        commit_watermark: Option<u64>,
        rollback_watermark: Option<u64>,
        state_hashes: Vec<String>,
    ) -> Result<(), ProtocolError> {
        if state_hashes.len() > MAX_STATE_HASHES
            || (destination_validated && state_hashes.is_empty())
            || state_hashes
                .iter()
                .any(|hash| hash.is_empty() || hash.len() > 256)
        {
            return Err(ProtocolError::InvalidEvidence);
        }
        let new_version = self
            .snapshot
            .version
            .checked_add(1)
            .ok_or(ProtocolError::VersionExhausted)?;
        let transaction = self
            .snapshot
            .transactions
            .get_mut(transaction_id)
            .ok_or_else(|| ProtocolError::UnknownTransaction(transaction_id.to_owned()))?;
        if transaction.phase != CutoverPhase::DestinationValidating {
            return Err(ProtocolError::CommitPrerequisiteMissing);
        }
        if transaction.cutover_evidence_recorded {
            if transaction.destination_validated == destination_validated
                && transaction.source_fenced == source_fenced
                && transaction.commit_watermark == commit_watermark
                && transaction.rollback_watermark == rollback_watermark
                && transaction.state_hashes == state_hashes
            {
                return Ok(());
            }
            return Err(ProtocolError::IdempotencyConflict);
        }
        if watermark_regresses(transaction.rollback_watermark, rollback_watermark)
            || watermark_regresses(transaction.rollback_watermark, commit_watermark)
            || watermark_exceeds(rollback_watermark, commit_watermark)
        {
            return Err(ProtocolError::WatermarkRegression);
        }
        transaction.destination_validated = destination_validated;
        transaction.source_fenced = source_fenced;
        transaction.cutover_evidence_recorded = true;
        transaction.commit_watermark = commit_watermark;
        transaction.rollback_watermark = rollback_watermark;
        transaction.state_hashes = state_hashes;
        self.snapshot.version = new_version;
        Ok(())
    }

    /// Apply and persist an idempotent transaction phase transition.
    ///
    /// # Errors
    ///
    /// Propagates typed transition and lookup errors.
    pub fn transition_transaction(
        &mut self,
        transaction_id: &str,
        next: CutoverPhase,
        logical_tick: u64,
        reason_code: &str,
    ) -> Result<TransitionEffect, ProtocolError> {
        let mut transaction = self
            .snapshot
            .transactions
            .get(transaction_id)
            .cloned()
            .ok_or_else(|| ProtocolError::UnknownTransaction(transaction_id.to_owned()))?;
        let effect = transaction.transition(next, logical_tick, reason_code)?;
        if effect == TransitionEffect::Applied {
            let new_version = self
                .snapshot
                .version
                .checked_add(1)
                .ok_or(ProtocolError::VersionExhausted)?;
            self.snapshot
                .transactions
                .insert(transaction_id.to_owned(), transaction);
            self.snapshot.version = new_version;
        }
        Ok(effect)
    }

    /// Atomically change the authoritative owner after commit intent.
    ///
    /// # Errors
    ///
    /// Rejects missing validation/fence, stale source generations, and token
    /// watermark regression.
    pub fn commit_ownership(
        &mut self,
        transaction_id: &str,
        expected_coordinator_version: u64,
        commit_watermark: Option<u64>,
        committed_state_version: u64,
        now_tick: u64,
        lease_duration_ticks: u64,
    ) -> Result<SessionLease, ProtocolError> {
        let transaction = self
            .snapshot
            .transactions
            .get(transaction_id)
            .ok_or_else(|| ProtocolError::UnknownTransaction(transaction_id.to_owned()))?;
        if transaction.phase == CutoverPhase::OwnershipCommitted {
            let lease = self
                .snapshot
                .leases
                .get(&transaction.session_id)
                .ok_or_else(|| ProtocolError::UnknownSession(transaction.session_id.clone()))?;
            let expected_expiration = now_tick
                .checked_add(lease_duration_ticks)
                .ok_or(ProtocolError::VersionExhausted)?;
            if lease.owner_runtime == transaction.destination_candidate
                && lease.owner_epoch == transaction.proposed_destination_epoch
                && lease.last_committed_token_index == commit_watermark
                && lease.last_committed_state_version == committed_state_version
                && lease.expires_at_tick == expected_expiration
            {
                return Ok(lease.clone());
            }
            return Err(ProtocolError::IdempotencyConflict);
        }
        if self.snapshot.version != expected_coordinator_version {
            return Err(ProtocolError::CompareAndSwapFailed);
        }
        if transaction.phase != CutoverPhase::CommitIntentRecorded
            || !transaction.destination_validated
            || !transaction.source_fenced
            || !transaction.cutover_evidence_recorded
        {
            return Err(ProtocolError::CommitPrerequisiteMissing);
        }
        if now_tick >= transaction.timeout_tick || lease_duration_ticks == 0 {
            return Err(ProtocolError::TransactionExpired {
                transaction_id: transaction_id.to_owned(),
            });
        }
        let lease = self
            .snapshot
            .leases
            .get(&transaction.session_id)
            .ok_or_else(|| ProtocolError::UnknownSession(transaction.session_id.clone()))?;
        if !lease.authorizes(
            &transaction.source_owner,
            transaction.source_epoch,
            transaction.source_fencing_token,
            now_tick,
        ) {
            return Err(ProtocolError::StaleFence);
        }
        if transaction.commit_watermark != commit_watermark
            || watermark_regresses(lease.last_committed_token_index, commit_watermark)
            || committed_state_version < lease.last_committed_state_version
        {
            return Err(ProtocolError::WatermarkRegression);
        }
        let new_version = self
            .snapshot
            .version
            .checked_add(1)
            .ok_or(ProtocolError::VersionExhausted)?;
        let destination = transaction.destination_candidate.clone();
        let session_id = transaction.session_id.clone();
        let fencing_token = lease
            .fencing_token
            .checked_add(1)
            .ok_or(ProtocolError::VersionExhausted)?;
        let expires_at_tick = now_tick
            .checked_add(lease_duration_ticks)
            .ok_or(ProtocolError::VersionExhausted)?;
        let mut committed_transaction = transaction.clone();
        committed_transaction.commit_watermark = commit_watermark;
        committed_transaction.transition(
            CutoverPhase::OwnershipCommitted,
            now_tick,
            "ownership_cas",
        )?;
        let new_lease = SessionLease {
            session_id: session_id.clone(),
            owner_runtime: destination,
            owner_epoch: transaction.proposed_destination_epoch,
            fencing_token,
            expires_at_tick,
            coordinator_version: new_version,
            last_committed_state_version: committed_state_version,
            last_committed_token_index: commit_watermark,
        };
        self.snapshot.leases.insert(session_id, new_lease.clone());
        self.snapshot
            .transactions
            .insert(transaction_id.to_owned(), committed_transaction);
        self.snapshot.version = new_version;
        Ok(new_lease)
    }

    /// Check current mutation/output authority.
    #[must_use]
    pub fn authorizes(
        &self,
        session_id: &str,
        owner: &str,
        epoch: u64,
        fencing_token: u64,
        now_tick: u64,
    ) -> bool {
        self.snapshot
            .leases
            .get(session_id)
            .is_some_and(|lease| lease.authorizes(owner, epoch, fencing_token, now_tick))
    }
}

fn watermark_regresses(current: Option<u64>, proposed: Option<u64>) -> bool {
    match (current, proposed) {
        (Some(current), Some(proposed)) => proposed < current,
        (Some(_), None) => true,
        _ => false,
    }
}

fn watermark_exceeds(left: Option<u64>, right: Option<u64>) -> bool {
    matches!((left, right), (Some(left), Some(right)) if left > right)
        || matches!((left, right), (Some(_), None))
}

/// One runtime token candidate presented to the gateway.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TokenEvent {
    pub session_id: String,
    pub owner_epoch: u64,
    pub token_index: u64,
    pub token_id: u32,
    pub state_commit_version: u64,
    pub transaction_id: Option<String>,
    pub terminal: bool,
}

/// Client-visible gateway state for one session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GatewayCommitState {
    pub session_id: String,
    pub expected_owner_epoch: u64,
    pub next_accepted_token_index: u64,
    pub gateway_commit_watermark: Option<u64>,
    pub client_acknowledgment_watermark: Option<u64>,
    pub last_state_commit_version: u64,
    pub terminal: bool,
    accepted: BTreeMap<u64, TokenEvent>,
}

impl GatewayCommitState {
    /// Create a bounded gateway ledger.
    #[must_use]
    pub fn new(session_id: impl Into<String>, owner_epoch: u64, next_index: u64) -> Self {
        Self {
            session_id: session_id.into(),
            expected_owner_epoch: owner_epoch,
            next_accepted_token_index: next_index,
            gateway_commit_watermark: next_index.checked_sub(1),
            client_acknowledgment_watermark: None,
            last_state_commit_version: 0,
            terminal: false,
            accepted: BTreeMap::new(),
        }
    }

    /// Accept, deduplicate, or reject one event.
    ///
    /// # Errors
    ///
    /// Rejects stale epochs, gaps, conflicting duplicates, post-terminal
    /// output, session mismatches, and the ledger's fixed storage bound.
    pub fn accept(&mut self, event: &TokenEvent) -> Result<Acceptance, ProtocolError> {
        if event.session_id != self.session_id {
            return Err(ProtocolError::InvalidIdentity);
        }
        if event.owner_epoch != self.expected_owner_epoch {
            return Err(ProtocolError::StaleOutputEpoch {
                expected: self.expected_owner_epoch,
                received: event.owner_epoch,
            });
        }
        if let Some(existing) = self.accepted.get(&event.token_index) {
            return if existing == event {
                Ok(Acceptance::Duplicate)
            } else {
                Err(ProtocolError::ConflictingDuplicate)
            };
        }
        if self.terminal {
            return Err(ProtocolError::OutputAfterTerminal);
        }
        if event.token_index != self.next_accepted_token_index {
            return Err(ProtocolError::TokenGap {
                expected: self.next_accepted_token_index,
                received: event.token_index,
            });
        }
        if event.state_commit_version < self.last_state_commit_version {
            return Err(ProtocolError::StateVersionRegression);
        }
        if self.accepted.len() >= MAX_GATEWAY_TOKENS {
            return Err(ProtocolError::GatewayLedgerFull);
        }
        let next_index = event
            .token_index
            .checked_add(1)
            .ok_or(ProtocolError::VersionExhausted)?;
        self.accepted.insert(event.token_index, event.clone());
        self.gateway_commit_watermark = Some(event.token_index);
        self.last_state_commit_version = event.state_commit_version;
        self.next_accepted_token_index = next_index;
        self.terminal = event.terminal;
        Ok(Acceptance::Accepted)
    }

    /// CAS the accepted output owner at an exact token boundary.
    ///
    /// # Errors
    ///
    /// Rejects stale owner or mismatched cutover watermarks.
    pub fn switch_owner(
        &mut self,
        expected_epoch: u64,
        new_epoch: u64,
        committed_watermark: Option<u64>,
    ) -> Result<TransitionEffect, ProtocolError> {
        if self.expected_owner_epoch == new_epoch
            && self.gateway_commit_watermark == committed_watermark
        {
            return Ok(TransitionEffect::Replay);
        }
        if self.expected_owner_epoch != expected_epoch
            || expected_epoch.checked_add(1) != Some(new_epoch)
        {
            return Err(ProtocolError::CompareAndSwapFailed);
        }
        if self.gateway_commit_watermark != committed_watermark {
            return Err(ProtocolError::WatermarkRegression);
        }
        self.expected_owner_epoch = new_epoch;
        Ok(TransitionEffect::Applied)
    }

    /// Record an acknowledgment-capable client's durable cursor.
    ///
    /// # Errors
    ///
    /// Rejects acknowledgments beyond the gateway or behind the prior client
    /// cursor.
    pub fn acknowledge_client(&mut self, index: u64) -> Result<(), ProtocolError> {
        if self
            .gateway_commit_watermark
            .is_none_or(|watermark| index > watermark)
            || self
                .client_acknowledgment_watermark
                .is_some_and(|watermark| index < watermark)
        {
            return Err(ProtocolError::WatermarkRegression);
        }
        self.client_acknowledgment_watermark = Some(index);
        Ok(())
    }
}

/// Outcome at the exactly-once gateway acceptance boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Acceptance {
    Accepted,
    Duplicate,
}

/// Stable errors emitted by protocol validation.
#[derive(Debug, Clone, PartialEq, Eq, Error, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "code", content = "detail", rename_all = "snake_case")]
pub enum ProtocolError {
    #[error("invalid empty identity or lease bound")]
    InvalidIdentity,
    #[error("session already exists: {0}")]
    SessionExists(String),
    #[error("unknown session: {0}")]
    UnknownSession(String),
    #[error("unknown transaction: {0}")]
    UnknownTransaction(String),
    #[error("transaction ID cannot be reused: {0}")]
    TransactionIdReused(String),
    #[error("stale owner epoch or fencing token")]
    StaleFence,
    #[error("compare-and-swap precondition failed")]
    CompareAndSwapFailed,
    #[error("another migration is already active for session: {0}")]
    ActiveTransaction(String),
    #[error("idempotent replay payload differs from the persisted event")]
    IdempotencyConflict,
    #[error("post-commit state cannot use a pre-commit rollback transition")]
    PostCommitRollbackForbidden,
    #[error("bounded reference coordinator capacity reached")]
    CoordinatorCapacityReached,
    #[error("illegal transaction transition {from:?} -> {to:?}")]
    IllegalTransition {
        from: CutoverPhase,
        to: CutoverPhase,
    },
    #[error("commit requires destination validation and a fenced source")]
    CommitPrerequisiteMissing,
    #[error("transaction expired: {transaction_id}")]
    TransactionExpired { transaction_id: String },
    #[error("transaction journal reached its fixed bound")]
    JournalFull,
    #[error("reason code must contain 1 to 128 bytes")]
    InvalidReasonCode,
    #[error("cutover evidence exceeds its bound or contains an invalid hash")]
    InvalidEvidence,
    #[error("token watermark would regress")]
    WatermarkRegression,
    #[error("token state commit version would regress")]
    StateVersionRegression,
    #[error("stale output epoch: expected {expected}, received {received}")]
    StaleOutputEpoch { expected: u64, received: u64 },
    #[error("token gap: expected {expected}, received {received}")]
    TokenGap { expected: u64, received: u64 },
    #[error("duplicate token index has a different token ID")]
    ConflictingDuplicate,
    #[error("output arrived after a terminal token")]
    OutputAfterTerminal,
    #[error("gateway token ledger reached its fixed bound")]
    GatewayLedgerFull,
    #[error("persisted coordinator snapshot is inconsistent")]
    InvalidSnapshot,
    #[error("owner epoch, fencing token, coordinator version, or token index is exhausted")]
    VersionExhausted,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ready_transaction(coordinator: &mut OwnershipCoordinator) -> StateTransaction {
        let lease = coordinator
            .register_session("session", "source", 100)
            .unwrap_or_else(|error| panic!("register: {error}"));
        let transaction = coordinator
            .begin_transaction(
                "tx-1",
                "session",
                "source",
                "destination",
                lease.owner_epoch,
                lease.fencing_token,
                90,
                1,
            )
            .unwrap_or_else(|error| panic!("begin: {error}"));
        for phase in [
            CutoverPhase::CompatibilityValidated,
            CutoverPhase::DestinationPreparing,
            CutoverPhase::CutoverRequested,
            CutoverPhase::SourceQuiescing,
            CutoverPhase::SourceFrozen,
            CutoverPhase::DestinationImporting,
            CutoverPhase::DestinationValidating,
        ] {
            coordinator
                .transition_transaction(&transaction.transaction_id, phase, 2, "test")
                .unwrap_or_else(|error| panic!("transition: {error}"));
        }
        coordinator
            .record_cutover_evidence(
                &transaction.transaction_id,
                true,
                true,
                Some(7),
                Some(7),
                vec!["sha256:state".to_owned()],
            )
            .unwrap_or_else(|error| panic!("evidence: {error}"));
        coordinator
            .transition_transaction(
                &transaction.transaction_id,
                CutoverPhase::CommitIntentRecorded,
                3,
                "validated",
            )
            .unwrap_or_else(|error| panic!("commit intent: {error}"));
        coordinator
            .transaction(&transaction.transaction_id)
            .cloned()
            .unwrap_or_else(|| panic!("transaction disappeared"))
    }

    #[test]
    fn transition_is_ordered_and_idempotent() {
        let mut coordinator = OwnershipCoordinator::default();
        let mut transaction = ready_transaction(&mut coordinator);
        assert_eq!(
            transaction
                .transition(CutoverPhase::CommitIntentRecorded, 3, "validated")
                .unwrap_or_else(|error| panic!("replay: {error}")),
            TransitionEffect::Replay
        );
        assert_eq!(
            transaction.transition(CutoverPhase::CommitIntentRecorded, 3, "different"),
            Err(ProtocolError::IdempotencyConflict)
        );
        assert_eq!(
            transaction.transition(CutoverPhase::Completed, 3, "skip"),
            Err(ProtocolError::IllegalTransition {
                from: CutoverPhase::CommitIntentRecorded,
                to: CutoverPhase::Completed,
            })
        );
    }

    #[test]
    fn ownership_commit_fences_the_source_and_survives_recovery() {
        let mut coordinator = OwnershipCoordinator::default();
        let transaction = ready_transaction(&mut coordinator);
        let version = coordinator.snapshot().version;
        let lease = coordinator
            .commit_ownership(&transaction.transaction_id, version, Some(7), 12, 4, 100)
            .unwrap_or_else(|error| panic!("commit: {error}"));
        assert_eq!(lease.owner_epoch, 2);
        assert_eq!(lease.fencing_token, 2);
        assert!(!coordinator.authorizes("session", "source", 1, 1, 4));
        assert!(coordinator.authorizes("session", "destination", 2, 2, 4));
        let replay = coordinator
            .commit_ownership(&transaction.transaction_id, version, Some(7), 12, 4, 100)
            .unwrap_or_else(|error| panic!("commit replay: {error}"));
        assert_eq!(replay, lease);

        let recovered = OwnershipCoordinator::recover(coordinator.snapshot())
            .unwrap_or_else(|error| panic!("recover: {error}"));
        assert!(recovered.authorizes("session", "destination", 2, 2, 4));
    }

    #[test]
    fn stale_cas_and_transaction_reuse_fail() {
        let mut coordinator = OwnershipCoordinator::default();
        let transaction = ready_transaction(&mut coordinator);
        assert_eq!(
            coordinator.commit_ownership(&transaction.transaction_id, 0, Some(7), 1, 4, 10),
            Err(ProtocolError::CompareAndSwapFailed)
        );
        assert!(matches!(
            coordinator.begin_transaction("tx-1", "session", "source", "other", 1, 1, 99, 4),
            Err(ProtocolError::TransactionIdReused(_))
        ));
    }

    #[test]
    fn gateway_deduplicates_and_rejects_gaps_and_stale_owners() {
        let mut gateway = GatewayCommitState::new("session", 1, 0);
        let first = TokenEvent {
            session_id: "session".to_owned(),
            owner_epoch: 1,
            token_index: 0,
            token_id: 10,
            state_commit_version: 1,
            transaction_id: None,
            terminal: false,
        };
        assert_eq!(
            gateway
                .accept(&first)
                .unwrap_or_else(|error| panic!("first: {error}")),
            Acceptance::Accepted
        );
        let mut conflicting_evidence = first.clone();
        conflicting_evidence.state_commit_version = 2;
        assert_eq!(
            gateway.accept(&conflicting_evidence),
            Err(ProtocolError::ConflictingDuplicate)
        );
        assert_eq!(
            gateway
                .accept(&first)
                .unwrap_or_else(|error| panic!("duplicate: {error}")),
            Acceptance::Duplicate
        );
        let mut gap = first.clone();
        gap.token_index = 2;
        assert!(matches!(
            gateway.accept(&gap),
            Err(ProtocolError::TokenGap { .. })
        ));
        gateway
            .switch_owner(1, 2, Some(0))
            .unwrap_or_else(|error| panic!("switch: {error}"));
        assert!(matches!(
            gateway.accept(&first),
            Err(ProtocolError::StaleOutputEpoch { .. })
        ));
    }

    #[test]
    fn failure_windows_are_explicit() {
        let mut before = CutoverPhase::DestinationValidating;
        assert!(before.allows(CutoverPhase::Aborting));
        before = CutoverPhase::Aborting;
        assert!(before.allows(CutoverPhase::RolledBack));
        assert!(!CutoverPhase::DestinationActive.allows(CutoverPhase::RolledBack));
        assert!(CutoverPhase::DestinationActive.allows(CutoverPhase::FailedAfterCommit));
    }

    #[test]
    fn active_transactions_and_postcommit_rollbacks_are_rejected() {
        let mut coordinator = OwnershipCoordinator::default();
        let transaction = ready_transaction(&mut coordinator);
        assert!(matches!(
            coordinator.begin_transaction("tx-2", "session", "source", "other", 1, 1, 95, 4),
            Err(ProtocolError::ActiveTransaction(_))
        ));
        let version = coordinator.snapshot().version;
        coordinator
            .commit_ownership(&transaction.transaction_id, version, Some(7), 12, 4, 100)
            .unwrap_or_else(|error| panic!("commit: {error}"));
        coordinator
            .transition_transaction(
                &transaction.transaction_id,
                CutoverPhase::DestinationLost,
                5,
                "destination_lost",
            )
            .unwrap_or_else(|error| panic!("destination loss: {error}"));
        assert_eq!(
            coordinator.transition_transaction(
                &transaction.transaction_id,
                CutoverPhase::Aborting,
                6,
                "unsafe_rollback",
            ),
            Err(ProtocolError::PostCommitRollbackForbidden)
        );
    }

    #[test]
    fn expiry_boundaries_are_exclusive_and_epoch_switches_are_contiguous() {
        let mut coordinator = OwnershipCoordinator::default();
        let lease = coordinator
            .register_session("expiring", "source", 10)
            .unwrap_or_else(|error| panic!("register: {error}"));
        assert!(lease.authorizes("source", 1, 1, 9));
        assert!(!lease.authorizes("source", 1, 1, 10));

        let mut gateway = GatewayCommitState::new("session", 1, 0);
        assert_eq!(
            gateway.switch_owner(1, 3, None),
            Err(ProtocolError::CompareAndSwapFailed)
        );
    }

    #[test]
    fn evidence_phase_lease_expiry_and_state_versions_fail_closed() {
        let mut coordinator = OwnershipCoordinator::default();
        let lease = coordinator
            .register_session("session", "source", 10)
            .unwrap_or_else(|error| panic!("register: {error}"));
        let transaction = coordinator
            .begin_transaction("tx", "session", "source", "destination", 1, 1, 20, 1)
            .unwrap_or_else(|error| panic!("begin: {error}"));
        assert_eq!(
            coordinator.record_cutover_evidence(
                &transaction.transaction_id,
                true,
                true,
                Some(0),
                Some(0),
                vec!["sha256:state".to_owned()],
            ),
            Err(ProtocolError::CommitPrerequisiteMissing)
        );

        for phase in [
            CutoverPhase::CompatibilityValidated,
            CutoverPhase::DestinationPreparing,
            CutoverPhase::CutoverRequested,
            CutoverPhase::SourceQuiescing,
            CutoverPhase::SourceFrozen,
            CutoverPhase::DestinationImporting,
            CutoverPhase::DestinationValidating,
        ] {
            coordinator
                .transition_transaction(&transaction.transaction_id, phase, 2, "test")
                .unwrap_or_else(|error| panic!("transition: {error}"));
        }
        coordinator
            .record_cutover_evidence(
                &transaction.transaction_id,
                true,
                true,
                Some(0),
                Some(0),
                vec!["sha256:state".to_owned()],
            )
            .unwrap_or_else(|error| panic!("evidence: {error}"));
        coordinator
            .transition_transaction(
                &transaction.transaction_id,
                CutoverPhase::CommitIntentRecorded,
                3,
                "intent",
            )
            .unwrap_or_else(|error| panic!("intent: {error}"));
        let version = coordinator.snapshot().version;
        assert_eq!(
            coordinator
                .commit_ownership(&transaction.transaction_id, version, Some(0), 12, 10, 10,),
            Err(ProtocolError::StaleFence)
        );

        let mut gateway = GatewayCommitState::new("session", lease.owner_epoch, 0);
        let first = TokenEvent {
            session_id: "session".to_owned(),
            owner_epoch: 1,
            token_index: 0,
            token_id: 10,
            state_commit_version: 4,
            transaction_id: None,
            terminal: false,
        };
        gateway
            .accept(&first)
            .unwrap_or_else(|error| panic!("first token: {error}"));
        let mut regressed = first;
        regressed.token_index = 1;
        regressed.token_id = 11;
        regressed.state_commit_version = 3;
        assert_eq!(
            gateway.accept(&regressed),
            Err(ProtocolError::StateVersionRegression)
        );
    }

    #[test]
    fn recovery_rejects_two_active_transactions_for_one_session() {
        let mut coordinator = OwnershipCoordinator::default();
        let lease = coordinator
            .register_session("session", "source", 100)
            .unwrap_or_else(|error| panic!("register: {error}"));
        let transaction = coordinator
            .begin_transaction("tx-1", "session", "source", "destination", 1, 1, 90, 1)
            .unwrap_or_else(|error| panic!("begin: {error}"));
        let mut snapshot = coordinator.snapshot();
        let mut duplicate = transaction;
        duplicate.transaction_id = "tx-2".to_owned();
        snapshot.used_transaction_ids.insert("tx-2".to_owned());
        snapshot.transactions.insert("tx-2".to_owned(), duplicate);
        assert!(matches!(
            OwnershipCoordinator::recover(snapshot),
            Err(ProtocolError::InvalidSnapshot)
        ));
        assert!(lease.authorizes("source", 1, 1, 1));
    }
}
