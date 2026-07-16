use std::collections::BTreeMap;
use std::fmt;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sloforge_state_transaction::CutoverPhase;

/// Current checker request schema.
pub const REQUEST_SCHEMA_VERSION: &str = "continuum.modelcheck.request.v1";
/// Current checker evidence schema.
pub const RESULT_SCHEMA_VERSION: &str = "continuum.modelcheck.result.v1";

/// A finite explicit-state exploration request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ModelCheckRequest {
    pub schema_version: String,
    pub model_version: String,
    /// Used only for deterministic successor ordering.
    pub seed: u64,
    pub bounds: ExplorationBounds,
    pub faults: FaultSpace,
    pub protocol: ProtocolVariant,
    #[serde(default)]
    pub assumptions: Vec<String>,
}

impl ModelCheckRequest {
    /// Construct the guarded protocol with all bounded fault families enabled.
    #[must_use]
    pub fn safe(seed: u64) -> Self {
        Self {
            schema_version: REQUEST_SCHEMA_VERSION.to_owned(),
            model_version: "continuum-cutover-protocol.v1".to_owned(),
            seed,
            bounds: ExplorationBounds::default(),
            faults: FaultSpace::default(),
            protocol: ProtocolVariant::default(),
            assumptions: vec![
                "compare-and-swap coordinator linearizes committed lease updates".to_owned(),
                "authenticated state hashes detect corrupt final state before validation"
                    .to_owned(),
                "logical ticks provide a bounded timeout and weakly fair recovery action"
                    .to_owned(),
                "correlated fault coverage is limited by max_faults_per_execution".to_owned(),
                "source quiesce and fencing drain output through the declared cutover watermark"
                    .to_owned(),
                "gateway and coordinator restart retain their durable ledgers".to_owned(),
                "runtime mutation and emission paths consult the current lease and fence"
                    .to_owned(),
            ],
        }
    }

    /// Validate finite bounds and caller-provided evidence labels.
    ///
    /// # Errors
    ///
    /// Returns all stable diagnostics found.
    pub fn validate(&self) -> Result<(), Vec<ValidationDiagnostic>> {
        let mut diagnostics = Vec::new();
        if self.schema_version != REQUEST_SCHEMA_VERSION {
            diagnostics.push(ValidationDiagnostic::new(
                "schema_version",
                format!("expected {REQUEST_SCHEMA_VERSION}"),
            ));
        }
        if self.model_version.trim().is_empty() || self.model_version.len() > 128 {
            diagnostics.push(ValidationDiagnostic::new(
                "model_version",
                "must contain 1 to 128 bytes",
            ));
        }
        if !(1..=8).contains(&self.bounds.max_messages) {
            diagnostics.push(ValidationDiagnostic::new(
                "bounds.max_messages",
                "must be between 1 and 8",
            ));
        }
        if !(1..=4).contains(&self.bounds.max_tokens) {
            diagnostics.push(ValidationDiagnostic::new(
                "bounds.max_tokens",
                "must be between 1 and 4",
            ));
        }
        if !(1..=2).contains(&self.bounds.max_faults_per_execution) {
            diagnostics.push(ValidationDiagnostic::new(
                "bounds.max_faults_per_execution",
                "must be between 1 and 2",
            ));
        }
        if !(8..=96).contains(&self.bounds.max_depth) {
            diagnostics.push(ValidationDiagnostic::new(
                "bounds.max_depth",
                "must be between 8 and 96",
            ));
        }
        if !(100..=1_000_000).contains(&self.bounds.max_states) {
            diagnostics.push(ValidationDiagnostic::new(
                "bounds.max_states",
                "must be between 100 and 1000000",
            ));
        }
        if !(2..=32).contains(&self.bounds.timeout_ticks) {
            diagnostics.push(ValidationDiagnostic::new(
                "bounds.timeout_ticks",
                "must be between 2 and 32",
            ));
        }
        if self.assumptions.len() > 64 {
            diagnostics.push(ValidationDiagnostic::new(
                "assumptions",
                "must have at most 64 entries",
            ));
        }
        for (index, assumption) in self.assumptions.iter().enumerate() {
            if assumption.trim().is_empty() || assumption.len() > 512 {
                diagnostics.push(ValidationDiagnostic::new(
                    format!("assumptions[{index}]"),
                    "must contain 1 to 512 bytes",
                ));
            }
        }
        if diagnostics.is_empty() {
            Ok(())
        } else {
            Err(diagnostics)
        }
    }
}

/// Finite-state limits included in every result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ExplorationBounds {
    pub max_messages: u8,
    pub max_tokens: u8,
    /// Enabled fault families are all explored, but at most this many may be
    /// combined in one execution.
    pub max_faults_per_execution: u8,
    pub max_depth: u16,
    pub max_states: u64,
    pub timeout_ticks: u8,
}

impl Default for ExplorationBounds {
    fn default() -> Self {
        Self {
            max_messages: 3,
            max_tokens: 1,
            max_faults_per_execution: 1,
            max_depth: 32,
            max_states: 100_000,
            timeout_ticks: 6,
        }
    }
}

/// Adversarial actions explored at most once per family and protocol run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[allow(clippy::struct_excessive_bools)]
pub struct FaultSpace {
    pub message_reordering: bool,
    pub message_duplication: bool,
    pub message_loss: bool,
    pub source_crash: bool,
    pub destination_crash: bool,
    pub gateway_crash: bool,
    pub coordinator_crash: bool,
    pub network_partition: bool,
    pub delayed_acknowledgment: bool,
    pub stale_owner_output: bool,
    pub cancellation: bool,
    pub timeout: bool,
    pub client_disconnect: bool,
}

impl Default for FaultSpace {
    fn default() -> Self {
        Self {
            message_reordering: true,
            message_duplication: true,
            message_loss: true,
            source_crash: true,
            destination_crash: true,
            gateway_crash: true,
            coordinator_crash: true,
            network_partition: true,
            delayed_acknowledgment: true,
            stale_owner_output: true,
            cancellation: true,
            timeout: true,
            client_disconnect: true,
        }
    }
}

/// Deliberate protocol mutations used to ensure invariant sensitivity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[allow(clippy::struct_excessive_bools)]
pub struct ProtocolVariant {
    pub atomic_owner_cas: bool,
    pub fence_source_before_commit: bool,
    pub reject_stale_output: bool,
    pub deduplicate_output: bool,
    pub reject_token_gaps: bool,
    pub require_validation_before_activation: bool,
    pub idempotent_replay: bool,
    pub enforce_queue_bound: bool,
}

impl Default for ProtocolVariant {
    fn default() -> Self {
        Self {
            atomic_owner_cas: true,
            fence_source_before_commit: true,
            reject_stale_output: true,
            deduplicate_output: true,
            reject_token_gaps: true,
            require_validation_before_activation: true,
            idempotent_replay: true,
            enforce_queue_bound: true,
        }
    }
}

/// Overall or per-invariant outcome.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum CheckStatus {
    Passed,
    Failed,
    Inconclusive,
}

/// Checked protocol invariant.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum InvariantId {
    UniqueStateOwner,
    UniqueAcceptedOutputOwner,
    MonotonicOwnerEpoch,
    MonotonicTokenWatermark,
    StaleEpochCannotCommitOutput,
    NoAcceptedDuplicateToken,
    NoAcceptedTokenGap,
    DestinationValidatedBeforeActivation,
    SourceInactiveAfterFence,
    PrecommitAbortPreservesSource,
    CompletedMigrationPreservesDestination,
    ExplicitTerminalOrRecoverable,
    NoDeadlock,
    BoundedQueues,
    IdempotentReplay,
}

impl InvariantId {
    /// Stable complete invariant order.
    pub const ALL: [Self; 15] = [
        Self::UniqueStateOwner,
        Self::UniqueAcceptedOutputOwner,
        Self::MonotonicOwnerEpoch,
        Self::MonotonicTokenWatermark,
        Self::StaleEpochCannotCommitOutput,
        Self::NoAcceptedDuplicateToken,
        Self::NoAcceptedTokenGap,
        Self::DestinationValidatedBeforeActivation,
        Self::SourceInactiveAfterFence,
        Self::PrecommitAbortPreservesSource,
        Self::CompletedMigrationPreservesDestination,
        Self::ExplicitTerminalOrRecoverable,
        Self::NoDeadlock,
        Self::BoundedQueues,
        Self::IdempotentReplay,
    ];
}

/// Why exploration was not exhaustive within the requested state space.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum TruncationReason {
    DepthLimit,
    StateLimit,
}

/// Scope carried by every result so it cannot be mistaken for a proof.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct VerificationScope {
    pub claim: String,
    pub bounded_not_universal_proof: bool,
    pub complete_within_bounds: bool,
    pub bounds: ExplorationBounds,
    pub assumptions: Vec<String>,
    pub truncated_by: Vec<TruncationReason>,
}

/// Machine-readable checker result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ModelCheckResult {
    pub schema_version: String,
    pub model_version: String,
    pub seed: u64,
    pub status: CheckStatus,
    pub states_explored: u64,
    pub transitions_explored: u64,
    pub max_depth_reached: u16,
    pub transition_coverage: BTreeMap<String, u64>,
    pub invariants: Vec<InvariantOutcome>,
    pub scope: VerificationScope,
}

/// Evidence for a single invariant.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct InvariantOutcome {
    pub invariant: InvariantId,
    pub status: CheckStatus,
    pub passed: bool,
    pub description: String,
    pub counterexample: Option<CounterexampleTrace>,
}

/// A breadth-first trace; therefore minimal in number of model actions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CounterexampleTrace {
    pub invariant: InvariantId,
    pub minimized: bool,
    pub steps: Vec<TraceStep>,
}

/// One replayable trace step.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TraceStep {
    pub ordinal: u16,
    pub action: Action,
    pub state_fingerprint: String,
    pub state: StateSummary,
}

/// Externally readable abstract state included in counterexamples.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[allow(clippy::struct_excessive_bools)]
pub struct StateSummary {
    pub phase: CutoverPhase,
    pub logical_tick: u8,
    pub state_owner_epoch: u8,
    pub gateway_owner_epoch: u8,
    pub source_active: bool,
    pub destination_active: bool,
    pub source_fenced: bool,
    pub destination_validated: bool,
    pub gateway_next_token: u8,
    pub accepted_tokens: Vec<u8>,
    pub message_count: u8,
    pub coordinator_up: bool,
    pub gateway_up: bool,
    pub client_connected: bool,
}

/// Model action recorded in evidence.
#[derive(
    Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema,
)]
#[serde(tag = "kind", content = "detail", rename_all = "snake_case")]
pub enum Action {
    ValidateCompatibility,
    PrepareDestination,
    BeginPrecopy,
    Enqueue(MessageKind),
    Deliver { position: u8 },
    Duplicate { position: u8 },
    Drop { position: u8 },
    StartPartition,
    HealPartition,
    QuiesceSource,
    ValidateDestination,
    RecordCommitIntent,
    CommitOwnership,
    SwitchGateway,
    ActivateDestination,
    DrainSource,
    EmitSourceToken,
    EmitDestinationToken,
    ReplayLastToken,
    AttemptStaleOutput,
    AttemptGapOutput,
    Crash(Component),
    Restart(Component),
    DisconnectClient,
    ReconnectClient,
    DelayAcknowledgment,
    AcknowledgeClient,
    Cancel,
    AdvanceTime,
    Abort,
    FinishAbort,
    RequireOperator,
}

impl Action {
    /// Stable transition coverage key.
    #[must_use]
    pub fn kind(&self) -> &'static str {
        match self {
            Self::ValidateCompatibility => "validate_compatibility",
            Self::PrepareDestination => "prepare_destination",
            Self::BeginPrecopy => "begin_precopy",
            Self::Enqueue(_) => "enqueue",
            Self::Deliver { .. } => "deliver",
            Self::Duplicate { .. } => "duplicate",
            Self::Drop { .. } => "drop",
            Self::StartPartition => "start_partition",
            Self::HealPartition => "heal_partition",
            Self::QuiesceSource => "quiesce_source",
            Self::ValidateDestination => "validate_destination",
            Self::RecordCommitIntent => "record_commit_intent",
            Self::CommitOwnership => "commit_ownership",
            Self::SwitchGateway => "switch_gateway",
            Self::ActivateDestination => "activate_destination",
            Self::DrainSource => "drain_source",
            Self::EmitSourceToken => "emit_source_token",
            Self::EmitDestinationToken => "emit_destination_token",
            Self::ReplayLastToken => "replay_last_token",
            Self::AttemptStaleOutput => "attempt_stale_output",
            Self::AttemptGapOutput => "attempt_gap_output",
            Self::Crash(_) => "crash",
            Self::Restart(_) => "restart",
            Self::DisconnectClient => "disconnect_client",
            Self::ReconnectClient => "reconnect_client",
            Self::DelayAcknowledgment => "delay_acknowledgment",
            Self::AcknowledgeClient => "acknowledge_client",
            Self::Cancel => "cancel",
            Self::AdvanceTime => "advance_time",
            Self::Abort => "abort",
            Self::FinishAbort => "finish_abort",
            Self::RequireOperator => "require_operator",
        }
    }
}

/// Abstract transport message.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum MessageKind {
    InitialSnapshot,
    Delta,
    FinalDelta,
    GatewaySwitch,
}

/// Crashable protocol component.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum Component {
    Source,
    Destination,
    Gateway,
    Coordinator,
}

/// Stable validation diagnostic.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ValidationDiagnostic {
    pub path: String,
    pub message: String,
}

impl ValidationDiagnostic {
    pub(crate) fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
        }
    }
}

/// Counterexample replay or result-integrity error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayError(pub String);

impl fmt::Display for ReplayError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ReplayError {}
