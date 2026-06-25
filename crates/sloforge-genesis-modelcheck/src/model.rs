use std::collections::BTreeMap;
use std::fmt;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Current request wire schema.
pub const REQUEST_SCHEMA_VERSION: &str = "genesis.modelcheck.request.v1";
/// Current result wire schema.
pub const RESULT_SCHEMA_VERSION: &str = "genesis.modelcheck.result.v1";

/// Input to one bounded explicit-state exploration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ModelCheckRequest {
    pub schema_version: String,
    /// Version of the protocol model being checked, not a proof-system version.
    pub model_version: String,
    /// Explicit seed used only for deterministic successor tie-breaking.
    pub seed: u64,
    pub bounds: ExplorationBounds,
    pub protocol: ProtocolSpec,
    /// Assumptions supplied by the caller in addition to checker assumptions.
    #[serde(default)]
    pub assumptions: Vec<String>,
}

impl ModelCheckRequest {
    /// Construct a safe, deterministic request suitable for a smoke check.
    #[must_use]
    pub fn safe(seed: u64) -> Self {
        Self {
            schema_version: REQUEST_SCHEMA_VERSION.to_owned(),
            model_version: "genesis-streaming-protocol.v1".to_owned(),
            seed,
            bounds: ExplorationBounds::default(),
            protocol: ProtocolSpec::default(),
            assumptions: Vec::new(),
        }
    }

    /// Validate all finite-state and wire-level bounds before exploration.
    ///
    /// # Errors
    ///
    /// Returns every stable validation diagnostic found in the request.
    pub fn validate(&self) -> Result<(), Vec<ValidationDiagnostic>> {
        let mut diagnostics = Vec::new();
        if self.schema_version != REQUEST_SCHEMA_VERSION {
            diagnostics.push(ValidationDiagnostic::new(
                "schema_version",
                format!("expected {REQUEST_SCHEMA_VERSION}"),
            ));
        }
        if self.model_version.trim().is_empty() {
            diagnostics.push(ValidationDiagnostic::new(
                "model_version",
                "must not be empty",
            ));
        }
        if !(1..=8).contains(&self.bounds.max_requests) {
            diagnostics.push(ValidationDiagnostic::new(
                "bounds.max_requests",
                "must be between 1 and 8",
            ));
        }
        if !(1..=8).contains(&self.bounds.queue_capacity) {
            diagnostics.push(ValidationDiagnostic::new(
                "bounds.queue_capacity",
                "must be between 1 and 8",
            ));
        }
        if !(1..=8).contains(&self.bounds.max_tokens_per_request) {
            diagnostics.push(ValidationDiagnostic::new(
                "bounds.max_tokens_per_request",
                "must be between 1 and 8",
            ));
        }
        if !(1..=4).contains(&self.bounds.worker_count) {
            diagnostics.push(ValidationDiagnostic::new(
                "bounds.worker_count",
                "must be between 1 and 4",
            ));
        }
        if !(1..=128).contains(&self.bounds.max_depth) {
            diagnostics.push(ValidationDiagnostic::new(
                "bounds.max_depth",
                "must be between 1 and 128",
            ));
        }
        if !(1..=1_000_000).contains(&self.bounds.max_states) {
            diagnostics.push(ValidationDiagnostic::new(
                "bounds.max_states",
                "must be between 1 and 1000000",
            ));
        }
        if self.bounds.max_worker_failures > 4 {
            diagnostics.push(ValidationDiagnostic::new(
                "bounds.max_worker_failures",
                "must not exceed 4",
            ));
        }
        if !(1..=32).contains(&self.bounds.fairness_window) {
            diagnostics.push(ValidationDiagnostic::new(
                "bounds.fairness_window",
                "must be between 1 and 32",
            ));
        }
        if self.protocol.state_transfer_enabled && self.bounds.worker_count < 2 {
            diagnostics.push(ValidationDiagnostic::new(
                "bounds.worker_count",
                "state transfer requires at least two workers",
            ));
        }
        if self.protocol.partial_state_read == PartialStateRead::Allowed
            && !self.protocol.state_transfer_enabled
        {
            diagnostics.push(ValidationDiagnostic::new(
                "protocol.partial_state_read",
                "partial reads require state transfer to be enabled",
            ));
        }
        if diagnostics.is_empty() {
            Ok(())
        } else {
            Err(diagnostics)
        }
    }
}

/// Hard bounds defining the explored state space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ExplorationBounds {
    pub max_requests: u8,
    pub queue_capacity: u8,
    pub max_tokens_per_request: u8,
    pub worker_count: u8,
    pub max_worker_failures: u8,
    pub max_depth: u16,
    pub max_states: u64,
    /// Continuously blocked progress beyond this many logical ticks fails a
    /// liveness property under the checker's declared weak-fairness model.
    pub fairness_window: u8,
}

impl Default for ExplorationBounds {
    fn default() -> Self {
        Self {
            max_requests: 1,
            queue_capacity: 1,
            max_tokens_per_request: 1,
            worker_count: 2,
            max_worker_failures: 1,
            max_depth: 24,
            max_states: 50_000,
            fairness_window: 2,
        }
    }
}

/// Protocol choices explored by the checker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
// These flags are independent protocol features in the stable JSON wire
// schema; replacing them with bitfields would reduce schema clarity.
#[allow(clippy::struct_excessive_bools)]
pub struct ProtocolSpec {
    pub queue_overflow: QueueOverflow,
    pub token_delivery: TokenDelivery,
    pub retry_after_output: RetryAfterOutput,
    pub cancellation_release: CancellationRelease,
    pub state_transfer_enabled: bool,
    /// Multiple owners satisfy the ownership invariant only when replication is
    /// explicitly declared in the checked protocol.
    pub replicated_state_declared: bool,
    pub state_ownership: StateOwnership,
    pub partial_state_read: PartialStateRead,
    pub state_compatibility: StateCompatibility,
    pub rollout_enabled: bool,
    pub promotion: PromotionBehavior,
    pub rollback: RollbackBehavior,
    pub worker_failure_enabled: bool,
    pub controller_crash_enabled: bool,
    pub recovery: RecoveryBehavior,
}

impl Default for ProtocolSpec {
    fn default() -> Self {
        Self {
            queue_overflow: QueueOverflow::Reject,
            token_delivery: TokenDelivery::Reliable,
            retry_after_output: RetryAfterOutput::IdempotentContinuation,
            cancellation_release: CancellationRelease::Guaranteed,
            state_transfer_enabled: true,
            replicated_state_declared: false,
            state_ownership: StateOwnership::AtomicHandoff,
            partial_state_read: PartialStateRead::Rejected,
            state_compatibility: StateCompatibility::Compatible,
            rollout_enabled: true,
            promotion: PromotionBehavior::DrainActive,
            rollback: RollbackBehavior::RestorePrevious,
            worker_failure_enabled: true,
            controller_crash_enabled: true,
            recovery: RecoveryBehavior::Guaranteed,
        }
    }
}

macro_rules! wire_enum {
    ($(#[$meta:meta])* $visibility:vis enum $name:ident { $($variant:ident),+ $(,)? }) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
        #[serde(rename_all = "snake_case")]
        $visibility enum $name { $($variant),+ }
    };
}

wire_enum! {
    pub enum QueueOverflow { Reject, AdmitBeyondBound }
}
wire_enum! {
    pub enum TokenDelivery { Reliable, MayDropCommitted, MayDuplicateCommitted }
}
wire_enum! {
    pub enum RetryAfterOutput { Forbid, IdempotentContinuation, Restart }
}
wire_enum! {
    pub enum CancellationRelease { Guaranteed, MayLeak }
}
wire_enum! {
    pub enum StateOwnership { AtomicHandoff, AmbiguousHandoff }
}
wire_enum! {
    pub enum PartialStateRead { Rejected, Allowed }
}
wire_enum! {
    pub enum StateCompatibility { Compatible, IncompatibleRejected, IncompatibleAllowed }
}
wire_enum! {
    pub enum PromotionBehavior { DrainActive, OrphanActive }
}
wire_enum! {
    pub enum RollbackBehavior { RestorePrevious, LosePrevious }
}
wire_enum! {
    pub enum RecoveryBehavior { Guaranteed, MayStall }
}

/// Stable validation diagnostic returned instead of partially checking input.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ValidationDiagnostic {
    pub path: String,
    pub message: String,
}

impl ValidationDiagnostic {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
        }
    }
}

/// Overall bounded result. `Inconclusive` means a state or depth cap prevented
/// complete enumeration within the requested bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CheckStatus {
    Passed,
    Failed,
    Inconclusive,
}

/// Machine-readable output from a model-check run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ModelCheckResult {
    pub schema_version: String,
    pub model_version: String,
    pub seed: u64,
    pub status: CheckStatus,
    pub scope: VerificationScope,
    pub state_count: u64,
    pub transition_count: u64,
    pub assumptions: Vec<String>,
    pub invariants: Vec<InvariantOutcome>,
    /// Counts of explored action kinds, useful for detecting vacuous checks.
    pub transition_coverage: BTreeMap<String, u64>,
}

/// Explicit scope attached to every result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct VerificationScope {
    pub method: String,
    pub verification_level: u8,
    pub bounds: ExplorationBounds,
    pub complete_within_bounds: bool,
    pub truncated_by: Vec<TruncationReason>,
    /// Always false: bounded enumeration is not a universal proof.
    pub universal_proof: bool,
}

wire_enum! {
    pub enum TruncationReason { DepthLimit, StateLimit }
}

/// Required runtime-protocol properties, reported independently.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum InvariantId {
    NoDuplicateCommittedToken,
    NoCommittedTokenDisappears,
    NoUnsafeRetryAfterOutput,
    CancellationEventuallyReleases,
    BoundedQueues,
    UnambiguousStateOwnership,
    NoPartialStateRead,
    NoIncompatibleStateMigration,
    PromotionPreservesActiveRequests,
    RollbackRestoresValidChampion,
    RecoveryTerminates,
    NoBoundedDeadlock,
    LiveRequestsProgressOrTimeout,
}

impl InvariantId {
    pub(crate) const ALL: [Self; 13] = [
        Self::NoDuplicateCommittedToken,
        Self::NoCommittedTokenDisappears,
        Self::NoUnsafeRetryAfterOutput,
        Self::CancellationEventuallyReleases,
        Self::BoundedQueues,
        Self::UnambiguousStateOwnership,
        Self::NoPartialStateRead,
        Self::NoIncompatibleStateMigration,
        Self::PromotionPreservesActiveRequests,
        Self::RollbackRestoresValidChampion,
        Self::RecoveryTerminates,
        Self::NoBoundedDeadlock,
        Self::LiveRequestsProgressOrTimeout,
    ];
}

impl fmt::Display for InvariantId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let encoded = serde_json::to_string(self).map_err(|_| fmt::Error)?;
        formatter.write_str(encoded.trim_matches('"'))
    }
}

/// Independent outcome for one declared property.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct InvariantOutcome {
    pub invariant: InvariantId,
    pub status: CheckStatus,
    pub passed: bool,
    pub states_checked: u64,
    pub verification_level: u8,
    pub scope_statement: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub counterexample: Option<CounterexampleTrace>,
}

/// A shortest counterexample under breadth-first transition exploration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CounterexampleTrace {
    pub violated_invariant: InvariantId,
    pub minimized: bool,
    pub minimization: String,
    pub initial_state_fingerprint: String,
    pub steps: Vec<TraceStep>,
}

/// One replayable transition and the state observed immediately afterward.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TraceStep {
    pub index: u16,
    pub action: Action,
    pub state_fingerprint: String,
    pub state: StateSummary,
}

/// Typed transition vocabulary. No generated code executes in this checker.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Action {
    Admit { request: u8 },
    Start { request: u8, worker: u8 },
    CommitAndEmitToken { request: u8, token: u8 },
    CommitToken { request: u8, token: u8 },
    EmitToken { request: u8, token: u8 },
    DropCommittedToken { request: u8, token: u8 },
    DuplicateCommittedToken { request: u8, token: u8 },
    Complete { request: u8 },
    Cancel { request: u8 },
    ReleaseCancelled { request: u8 },
    FailWorker { request: u8 },
    Retry { request: u8, worker: u8 },
    AbortFailed { request: u8 },
    BeginStateTransfer { request: u8, from: u8, to: u8 },
    CompleteStateTransfer { request: u8, to: u8 },
    ReadPartialState { request: u8 },
    BeginShadow,
    BeginCanary,
    BeginPromotion,
    CommitPromotion,
    Rollback,
    CrashController,
    RestartController,
    CompleteRecovery,
    AdvanceTime,
}

impl Action {
    pub(crate) fn kind(&self) -> &'static str {
        match self {
            Self::Admit { .. } => "admit",
            Self::Start { .. } => "start",
            Self::CommitAndEmitToken { .. } => "commit_and_emit_token",
            Self::CommitToken { .. } => "commit_token",
            Self::EmitToken { .. } => "emit_token",
            Self::DropCommittedToken { .. } => "drop_committed_token",
            Self::DuplicateCommittedToken { .. } => "duplicate_committed_token",
            Self::Complete { .. } => "complete",
            Self::Cancel { .. } => "cancel",
            Self::ReleaseCancelled { .. } => "release_cancelled",
            Self::FailWorker { .. } => "fail_worker",
            Self::Retry { .. } => "retry",
            Self::AbortFailed { .. } => "abort_failed",
            Self::BeginStateTransfer { .. } => "begin_state_transfer",
            Self::CompleteStateTransfer { .. } => "complete_state_transfer",
            Self::ReadPartialState { .. } => "read_partial_state",
            Self::BeginShadow => "begin_shadow",
            Self::BeginCanary => "begin_canary",
            Self::BeginPromotion => "begin_promotion",
            Self::CommitPromotion => "commit_promotion",
            Self::Rollback => "rollback",
            Self::CrashController => "crash_controller",
            Self::RestartController => "restart_controller",
            Self::CompleteRecovery => "complete_recovery",
            Self::AdvanceTime => "advance_time",
        }
    }
}

/// Compact public state carried in a counterexample. It intentionally excludes
/// checker internals while retaining enough data to explain the violation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StateSummary {
    pub queue: Vec<u8>,
    pub requests: Vec<RequestSummary>,
    pub deployment_stage: DeploymentStage,
    pub champion_generation: u8,
    pub previous_generation: Option<u8>,
    pub orphaned_requests: Vec<u8>,
    pub rollback_invalid: bool,
    pub controller_up: bool,
    pub recovering: bool,
    pub recovery_ticks: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
// Each flag records a distinct checked invariant in replayable public evidence.
#[allow(clippy::struct_excessive_bools)]
pub struct RequestSummary {
    pub request: u8,
    pub phase: RequestPhase,
    pub generation: Option<u8>,
    pub committed_tokens: Vec<u8>,
    pub pending_tokens: Vec<u8>,
    pub emitted_tokens: Vec<u8>,
    pub state_owners: Vec<u8>,
    pub state_transfer: Option<StateTransferSummary>,
    pub resource_allocated: bool,
    pub retry_after_visible_output: bool,
    pub partial_state_read_observed: bool,
    pub incompatible_migration_observed: bool,
    pub stalled_ticks: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StateTransferSummary {
    pub from: u8,
    pub to: u8,
}

wire_enum! {
    pub enum RequestPhase { Created, Queued, Running, Cancelling, Failed, Completed, Cancelled }
}
wire_enum! {
    pub enum DeploymentStage { Champion, Shadow, Canary, Promoting, Promoted, RolledBack }
}

/// A replay error indicates tampering or a checker-version mismatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayError {
    pub step: usize,
    pub message: String,
}

impl fmt::Display for ReplayError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "counterexample step {}: {}",
            self.step, self.message
        )
    }
}

impl std::error::Error for ReplayError {}
