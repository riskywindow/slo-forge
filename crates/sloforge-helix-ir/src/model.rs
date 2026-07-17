//! Portable Helix policy, trajectory, evidence, and transaction wire models.

use std::collections::{HashMap, HashSet};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{Validate, ValidationError, canonical_hash};

pub const API_VERSION: &str = "sloforge.io/helix/v1";
pub const SCHEMA_VERSION: &str = "1.0.0";

fn error(path: &str, message: impl Into<String>) -> ValidationError {
    ValidationError::new(path, message)
}

fn nonempty(path: &str, value: &str) -> Result<(), ValidationError> {
    if value.trim().is_empty() {
        Err(error(path, "must not be empty"))
    } else {
        Ok(())
    }
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_digest(path: &str, digest: &Digest) -> Result<(), ValidationError> {
    if digest.algorithm != "sha256" || !valid_sha256(&digest.value) {
        return Err(error(path, "must be a lowercase SHA-256 digest"));
    }
    Ok(())
}

fn validate_header(
    api_version: &str,
    schema_version: &str,
    kind: &str,
    expected_kind: &str,
) -> Result<(), ValidationError> {
    if api_version != API_VERSION {
        return Err(error("api_version", "unsupported Helix API version"));
    }
    if schema_version != SCHEMA_VERSION {
        return Err(error("schema_version", "unsupported Helix schema version"));
    }
    if kind != expected_kind {
        return Err(error("kind", format!("must be {expected_kind}")));
    }
    Ok(())
}

fn policy_key(policy: &PolicyEpoch) -> (&str, u64, &str) {
    (
        policy.policy_id.as_str(),
        policy.epoch,
        policy.policy_digest.value.as_str(),
    )
}

fn policy_lineage_id(policy: &PolicyEpoch) -> String {
    format!("{}@{}", policy.policy_id, policy.epoch)
}

fn validate_lineage(
    path: &str,
    lineage: &[LineageReference],
    required: &[String],
) -> Result<(), ValidationError> {
    if lineage.is_empty() {
        return Err(error(path, "lineage must not be empty"));
    }
    let mut identifiers = HashSet::new();
    for reference in lineage {
        nonempty(path, &reference.artifact_id)?;
        nonempty(path, &reference.artifact_kind)?;
        validate_digest(path, &reference.digest)?;
        identifiers.insert(reference.artifact_id.as_str());
    }
    for identifier in required {
        if !identifiers.contains(identifier.as_str()) {
            return Err(error(
                path,
                format!("missing required artifact {identifier}"),
            ));
        }
    }
    Ok(())
}

fn validate_evidence(path: &str, evidence: &EvidencePointer) -> Result<(), ValidationError> {
    nonempty(path, &evidence.uri)?;
    nonempty(path, &evidence.media_type)?;
    nonempty(path, &evidence.captured_at)?;
    validate_digest(path, &evidence.digest)
}

fn validate_log_probability(path: &str, value: f64) -> Result<(), ValidationError> {
    if !value.is_finite() || value > 0.0 {
        return Err(error(
            path,
            "behavior log probability must be finite and <= 0",
        ));
    }
    Ok(())
}

fn close(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1e-12_f64.max(1e-12 * left.abs().max(right.abs()))
}

macro_rules! string_enum {
    ($name:ident { $($variant:ident),+ $(,)? }) => {
        #[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, JsonSchema, PartialEq, Serialize)]
        #[serde(rename_all = "snake_case")]
        pub enum $name {
            $($variant),+
        }
    };
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Digest {
    pub algorithm: String,
    pub value: String,
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvidencePointer {
    pub uri: String,
    pub digest: Digest,
    pub media_type: String,
    pub captured_at: String,
}

string_enum!(LineageRelation {
    Source,
    Parent,
    DerivedFrom,
    Evidence,
    State,
    Policy
});

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LineageReference {
    pub artifact_id: String,
    pub artifact_kind: String,
    pub relation: LineageRelation,
    pub digest: Digest,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyEpoch {
    pub api_version: String,
    pub schema_version: String,
    pub kind: String,
    pub policy_id: String,
    pub epoch: u64,
    pub policy_digest: Digest,
    pub parent_epoch: Option<u64>,
    pub parent_policy_digest: Option<Digest>,
    pub training_transaction_id: Option<String>,
    pub created_at: String,
    pub lineage: Vec<LineageReference>,
}

impl PolicyEpoch {
    fn validate_at(&self, path: &str) -> Result<(), ValidationError> {
        validate_header(
            &self.api_version,
            &self.schema_version,
            &self.kind,
            "PolicyEpoch",
        )?;
        nonempty(path, &self.policy_id)?;
        nonempty(path, &self.created_at)?;
        validate_digest(path, &self.policy_digest)?;
        let required = if self.epoch == 0 {
            if self.parent_epoch.is_some() || self.parent_policy_digest.is_some() {
                return Err(error(path, "epoch zero cannot declare a parent"));
            }
            Vec::new()
        } else {
            let Some(parent_epoch) = self.parent_epoch else {
                return Err(error(path, "nonzero epoch requires parent_epoch"));
            };
            let Some(parent_digest) = &self.parent_policy_digest else {
                return Err(error(path, "nonzero epoch requires parent_policy_digest"));
            };
            if parent_epoch >= self.epoch {
                return Err(error(path, "parent epoch must precede epoch"));
            }
            validate_digest(path, parent_digest)?;
            vec![format!("{}@{parent_epoch}", self.policy_id)]
        };
        validate_lineage(path, &self.lineage, &required)
    }
}

impl Validate for PolicyEpoch {
    fn validate(&self) -> Result<(), ValidationError> {
        self.validate_at("policy_epoch")
    }
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EnvironmentStateCapsule {
    pub api_version: String,
    pub schema_version: String,
    pub kind: String,
    pub capsule_id: String,
    pub environment_id: String,
    pub captured_at: String,
    pub policy_epoch: PolicyEpoch,
    pub state_schema_digest: Digest,
    pub state_digest: Digest,
    pub compatibility_fingerprint: Digest,
    pub payload_uri: String,
    pub payload_media_type: String,
    pub payload_byte_length: u64,
    pub compatible_policy_digests: Vec<Digest>,
    pub lineage: Vec<LineageReference>,
}

impl EnvironmentStateCapsule {
    fn validate_at(&self, path: &str) -> Result<(), ValidationError> {
        validate_header(
            &self.api_version,
            &self.schema_version,
            &self.kind,
            "EnvironmentStateCapsule",
        )?;
        nonempty(path, &self.capsule_id)?;
        nonempty(path, &self.environment_id)?;
        nonempty(path, &self.captured_at)?;
        nonempty(path, &self.payload_uri)?;
        nonempty(path, &self.payload_media_type)?;
        self.policy_epoch.validate_at(path)?;
        validate_digest(path, &self.state_schema_digest)?;
        validate_digest(path, &self.state_digest)?;
        validate_digest(path, &self.compatibility_fingerprint)?;
        if self.compatible_policy_digests.is_empty() {
            return Err(error(path, "compatible policy digests must not be empty"));
        }
        let mut digests = HashSet::new();
        for digest in &self.compatible_policy_digests {
            validate_digest(path, digest)?;
            if !digests.insert(digest.value.as_str()) {
                return Err(error(path, "compatible policy digests contain duplicates"));
            }
        }
        if !digests.contains(self.policy_epoch.policy_digest.value.as_str()) {
            return Err(error(path, "capturing policy must be declared compatible"));
        }
        validate_lineage(
            path,
            &self.lineage,
            &[policy_lineage_id(&self.policy_epoch)],
        )
    }
}

impl Validate for EnvironmentStateCapsule {
    fn validate(&self) -> Result<(), ValidationError> {
        self.validate_at("environment_state")
    }
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BranchPoint {
    pub api_version: String,
    pub schema_version: String,
    pub kind: String,
    pub branch_point_id: String,
    pub source_trajectory_id: String,
    pub event_index: u64,
    pub token_index: u64,
    pub environment_state: EnvironmentStateCapsule,
    pub policy_epoch: PolicyEpoch,
    pub prefix_digest: Digest,
    pub seed: u64,
    pub created_at: String,
    pub reason: String,
    pub candidate_labels: Vec<String>,
    pub lineage: Vec<LineageReference>,
}

impl BranchPoint {
    fn validate_at(&self, path: &str) -> Result<(), ValidationError> {
        validate_header(
            &self.api_version,
            &self.schema_version,
            &self.kind,
            "BranchPoint",
        )?;
        nonempty(path, &self.branch_point_id)?;
        nonempty(path, &self.source_trajectory_id)?;
        nonempty(path, &self.created_at)?;
        nonempty(path, &self.reason)?;
        self.environment_state.validate_at(path)?;
        self.policy_epoch.validate_at(path)?;
        validate_digest(path, &self.prefix_digest)?;
        if policy_key(&self.policy_epoch) != policy_key(&self.environment_state.policy_epoch) {
            return Err(error(path, "branch policy does not match captured state"));
        }
        let mut labels = HashSet::new();
        for label in &self.candidate_labels {
            nonempty(path, label)?;
            if !labels.insert(label) {
                return Err(error(path, "candidate labels contain duplicates"));
            }
        }
        validate_lineage(
            path,
            &self.lineage,
            &[
                self.source_trajectory_id.clone(),
                self.environment_state.capsule_id.clone(),
                policy_lineage_id(&self.policy_epoch),
            ],
        )
    }
}

impl Validate for BranchPoint {
    fn validate(&self) -> Result<(), ValidationError> {
        self.validate_at("branch_point")
    }
}

string_enum!(TrajectoryEventKind {
    PromptToken,
    GeneratedToken,
    Action,
    Observation,
    ToolResult,
    Terminal
});
string_enum!(TrajectoryTerminalStatus {
    Completed,
    Cancelled,
    Errored,
    Truncated
});
string_enum!(PolicyConsistency { Strict, Segmented });

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrajectoryEvent {
    pub event_id: String,
    pub event_index: u64,
    pub kind: TrajectoryEventKind,
    pub policy_epoch: PolicyEpoch,
    pub payload_digest: Digest,
    pub source_evidence: EvidencePointer,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TokenProvenance {
    pub token_id: String,
    pub token_index: u64,
    pub event_id: String,
    pub token_value: u64,
    pub policy_epoch: PolicyEpoch,
    pub behavior_log_probability: f64,
    pub sampler_seed: u64,
    pub raw_sample: EvidencePointer,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ActionProvenance {
    pub action_id: String,
    pub action_index: u64,
    pub event_id: String,
    pub action_type: String,
    pub policy_epoch: PolicyEpoch,
    pub behavior_log_probability: f64,
    pub arguments_digest: Digest,
    pub raw_sample: EvidencePointer,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrajectorySegment {
    pub segment_id: String,
    pub start_event_index: u64,
    pub end_event_index_exclusive: u64,
    pub policy_epoch: PolicyEpoch,
    pub segment_evidence: EvidencePointer,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrajectoryCapsule {
    pub api_version: String,
    pub schema_version: String,
    pub kind: String,
    pub trajectory_id: String,
    pub branch_point_id: Option<String>,
    pub parent_trajectory_id: Option<String>,
    pub source_state_capsule_id: String,
    pub environment_id: String,
    pub policy_consistency: PolicyConsistency,
    pub policy_epochs: Vec<PolicyEpoch>,
    pub segments: Vec<TrajectorySegment>,
    pub events: Vec<TrajectoryEvent>,
    pub tokens: Vec<TokenProvenance>,
    pub actions: Vec<ActionProvenance>,
    pub terminal_status: TrajectoryTerminalStatus,
    pub started_at: String,
    pub completed_at: String,
    pub trace_evidence: EvidencePointer,
    pub lineage: Vec<LineageReference>,
}

impl TrajectoryCapsule {
    fn validate_at(&self, path: &str) -> Result<(), ValidationError> {
        validate_header(
            &self.api_version,
            &self.schema_version,
            &self.kind,
            "TrajectoryCapsule",
        )?;
        nonempty(path, &self.trajectory_id)?;
        nonempty(path, &self.source_state_capsule_id)?;
        nonempty(path, &self.environment_id)?;
        nonempty(path, &self.started_at)?;
        nonempty(path, &self.completed_at)?;
        validate_evidence(path, &self.trace_evidence)?;
        if self.events.is_empty() || self.segments.is_empty() {
            return Err(error(path, "events and segments must not be empty"));
        }
        let mut event_ids = HashSet::new();
        let mut events = HashMap::new();
        for (index, event) in self.events.iter().enumerate() {
            if event.event_index != index as u64 || !event_ids.insert(event.event_id.as_str()) {
                return Err(error(
                    path,
                    "event IDs and indexes must be unique and contiguous",
                ));
            }
            nonempty(path, &event.event_id)?;
            event.policy_epoch.validate_at(path)?;
            validate_digest(path, &event.payload_digest)?;
            validate_evidence(path, &event.source_evidence)?;
            events.insert(event.event_id.as_str(), event);
        }
        let mut cursor = 0_u64;
        let mut segment_keys = Vec::new();
        let mut segment_for_event = Vec::with_capacity(self.events.len());
        for segment in &self.segments {
            nonempty(path, &segment.segment_id)?;
            segment.policy_epoch.validate_at(path)?;
            validate_evidence(path, &segment.segment_evidence)?;
            if segment.start_event_index != cursor
                || segment.end_event_index_exclusive <= segment.start_event_index
            {
                return Err(error(path, "segments must be nonempty and contiguous"));
            }
            cursor = segment.end_event_index_exclusive;
            segment_keys.push(policy_key(&segment.policy_epoch));
            for _ in segment.start_event_index..segment.end_event_index_exclusive {
                segment_for_event.push(&segment.policy_epoch);
            }
        }
        if cursor != self.events.len() as u64 {
            return Err(error(path, "segments must cover every event exactly once"));
        }
        if self.policy_consistency == PolicyConsistency::Strict && self.segments.len() != 1 {
            return Err(error(
                path,
                "strict trajectories require one policy segment",
            ));
        }
        if self.policy_consistency == PolicyConsistency::Segmented
            && segment_keys.windows(2).any(|items| items[0] == items[1])
        {
            return Err(error(path, "adjacent segmented policy epochs must differ"));
        }
        let mut first_use = Vec::new();
        for key in &segment_keys {
            if !first_use.contains(key) {
                first_use.push(*key);
            }
        }
        let declared: Vec<_> = self.policy_epochs.iter().map(policy_key).collect();
        if declared != first_use {
            return Err(error(path, "policy_epochs do not match segment policies"));
        }
        for (index, event) in self.events.iter().enumerate() {
            if policy_key(&event.policy_epoch) != policy_key(segment_for_event[index]) {
                return Err(error(path, "event policy epoch violates its segment"));
            }
        }
        let mut token_ids = HashSet::new();
        for (index, token) in self.tokens.iter().enumerate() {
            if token.token_index != index as u64 || !token_ids.insert(token.token_id.as_str()) {
                return Err(error(
                    path,
                    "token IDs and indexes must be unique and contiguous",
                ));
            }
            let Some(event) = events.get(token.event_id.as_str()) else {
                return Err(error(path, "token references unknown event"));
            };
            if !matches!(
                event.kind,
                TrajectoryEventKind::PromptToken | TrajectoryEventKind::GeneratedToken
            ) || policy_key(&token.policy_epoch) != policy_key(&event.policy_epoch)
            {
                return Err(error(path, "token provenance does not match its event"));
            }
            token.policy_epoch.validate_at(path)?;
            validate_log_probability(path, token.behavior_log_probability)?;
            validate_evidence(path, &token.raw_sample)?;
        }
        let mut action_ids = HashSet::new();
        for (index, action) in self.actions.iter().enumerate() {
            if action.action_index != index as u64 || !action_ids.insert(action.action_id.as_str())
            {
                return Err(error(
                    path,
                    "action IDs and indexes must be unique and contiguous",
                ));
            }
            let Some(event) = events.get(action.event_id.as_str()) else {
                return Err(error(path, "action references unknown event"));
            };
            if event.kind != TrajectoryEventKind::Action
                || policy_key(&action.policy_epoch) != policy_key(&event.policy_epoch)
            {
                return Err(error(path, "action provenance does not match its event"));
            }
            action.policy_epoch.validate_at(path)?;
            validate_log_probability(path, action.behavior_log_probability)?;
            validate_digest(path, &action.arguments_digest)?;
            validate_evidence(path, &action.raw_sample)?;
        }
        if self.tokens.is_empty() && self.actions.is_empty() {
            return Err(error(
                path,
                "trajectory requires token or action provenance",
            ));
        }
        let mut required = vec![self.source_state_capsule_id.clone()];
        if let Some(branch_point_id) = &self.branch_point_id {
            required.push(branch_point_id.clone());
        }
        if let Some(parent_id) = &self.parent_trajectory_id {
            required.push(parent_id.clone());
        }
        required.extend(self.policy_epochs.iter().map(policy_lineage_id));
        validate_lineage(path, &self.lineage, &required)
    }
}

impl Validate for TrajectoryCapsule {
    fn validate(&self) -> Result<(), ValidationError> {
        self.validate_at("trajectory")
    }
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BranchGroup {
    pub api_version: String,
    pub schema_version: String,
    pub kind: String,
    pub group_id: String,
    pub branch_point: BranchPoint,
    pub trajectories: Vec<TrajectoryCapsule>,
    pub baseline_trajectory_id: String,
    pub created_at: String,
    pub lineage: Vec<LineageReference>,
}

impl BranchGroup {
    fn validate_at(&self, path: &str) -> Result<(), ValidationError> {
        validate_header(
            &self.api_version,
            &self.schema_version,
            &self.kind,
            "BranchGroup",
        )?;
        nonempty(path, &self.group_id)?;
        nonempty(path, &self.created_at)?;
        self.branch_point.validate_at(path)?;
        if self.trajectories.len() < 2 {
            return Err(error(
                path,
                "branch groups require at least two trajectories",
            ));
        }
        let mut identifiers = HashSet::new();
        for trajectory in &self.trajectories {
            trajectory.validate_at(path)?;
            if !identifiers.insert(trajectory.trajectory_id.as_str()) {
                return Err(error(path, "branch trajectory IDs must be unique"));
            }
            if trajectory.branch_point_id.as_deref()
                != Some(self.branch_point.branch_point_id.as_str())
                || trajectory.environment_id != self.branch_point.environment_state.environment_id
                || trajectory.source_state_capsule_id
                    != self.branch_point.environment_state.capsule_id
            {
                return Err(error(
                    path,
                    "branch trajectory has incomplete captured-state lineage",
                ));
            }
        }
        if !identifiers.contains(self.baseline_trajectory_id.as_str()) {
            return Err(error(
                path,
                "baseline trajectory is not in the branch group",
            ));
        }
        let mut required = vec![self.branch_point.branch_point_id.clone()];
        required.extend(
            self.trajectories
                .iter()
                .map(|item| item.trajectory_id.clone()),
        );
        validate_lineage(path, &self.lineage, &required)
    }
}

impl Validate for BranchGroup {
    fn validate(&self) -> Result<(), ValidationError> {
        self.validate_at("branch_group")
    }
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RewardComponent {
    pub component_id: String,
    pub name: String,
    pub value: f64,
    pub weight: f64,
    pub policy_epoch: PolicyEpoch,
    pub event_ids: Vec<String>,
    pub raw_evidence: EvidencePointer,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RewardEvidence {
    pub api_version: String,
    pub schema_version: String,
    pub kind: String,
    pub reward_evidence_id: String,
    pub trajectory_id: String,
    pub trajectory_digest: Digest,
    pub policy_epochs: Vec<PolicyEpoch>,
    pub components: Vec<RewardComponent>,
    pub aggregate_reward: f64,
    pub evaluator_digest: Digest,
    pub evaluated_at: String,
    pub lineage: Vec<LineageReference>,
}

impl RewardEvidence {
    fn validate_at(&self, path: &str) -> Result<(), ValidationError> {
        validate_header(
            &self.api_version,
            &self.schema_version,
            &self.kind,
            "RewardEvidence",
        )?;
        nonempty(path, &self.reward_evidence_id)?;
        nonempty(path, &self.trajectory_id)?;
        nonempty(path, &self.evaluated_at)?;
        validate_digest(path, &self.trajectory_digest)?;
        validate_digest(path, &self.evaluator_digest)?;
        if self.components.is_empty() || !self.aggregate_reward.is_finite() {
            return Err(error(path, "reward evidence requires finite components"));
        }
        let mut component_ids = HashSet::new();
        let mut component_keys = Vec::new();
        let mut expected = 0.0;
        for component in &self.components {
            nonempty(path, &component.component_id)?;
            nonempty(path, &component.name)?;
            if !component_ids.insert(component.component_id.as_str())
                || component.event_ids.is_empty()
                || !component.value.is_finite()
                || !component.weight.is_finite()
            {
                return Err(error(path, "invalid reward component provenance"));
            }
            let event_ids: HashSet<_> = component.event_ids.iter().collect();
            if event_ids.len() != component.event_ids.len() {
                return Err(error(path, "reward component event IDs contain duplicates"));
            }
            component.policy_epoch.validate_at(path)?;
            validate_evidence(path, &component.raw_evidence)?;
            let key = policy_key(&component.policy_epoch);
            if !component_keys.contains(&key) {
                component_keys.push(key);
            }
            expected += component.value * component.weight;
        }
        let declared: Vec<_> = self.policy_epochs.iter().map(policy_key).collect();
        if declared != component_keys || !close(self.aggregate_reward, expected) {
            return Err(error(
                path,
                "reward aggregate or policy epochs do not match components",
            ));
        }
        for policy in &self.policy_epochs {
            policy.validate_at(path)?;
        }
        let mut required = vec![self.trajectory_id.clone()];
        required.extend(self.policy_epochs.iter().map(policy_lineage_id));
        validate_lineage(path, &self.lineage, &required)
    }
}

impl Validate for RewardEvidence {
    fn validate(&self) -> Result<(), ValidationError> {
        self.validate_at("reward_evidence")
    }
}

string_enum!(CreditSubjectKind {
    Event,
    Token,
    Action
});

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CreditAssignment {
    pub assignment_id: String,
    pub subject_kind: CreditSubjectKind,
    pub subject_id: String,
    pub event_id: String,
    pub reward_component_id: String,
    pub policy_epoch: PolicyEpoch,
    pub behavior_log_probability: f64,
    pub credit: f64,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CreditAssignmentEvidence {
    pub api_version: String,
    pub schema_version: String,
    pub kind: String,
    pub credit_evidence_id: String,
    pub trajectory_id: String,
    pub trajectory_digest: Digest,
    pub reward_evidence_id: String,
    pub reward_evidence_digest: Digest,
    pub method: String,
    pub policy_epochs: Vec<PolicyEpoch>,
    pub assignments: Vec<CreditAssignment>,
    pub total_credit: f64,
    pub generated_at: String,
    pub lineage: Vec<LineageReference>,
}

impl CreditAssignmentEvidence {
    fn validate_at(&self, path: &str) -> Result<(), ValidationError> {
        validate_header(
            &self.api_version,
            &self.schema_version,
            &self.kind,
            "CreditAssignmentEvidence",
        )?;
        nonempty(path, &self.credit_evidence_id)?;
        nonempty(path, &self.trajectory_id)?;
        nonempty(path, &self.reward_evidence_id)?;
        nonempty(path, &self.method)?;
        nonempty(path, &self.generated_at)?;
        validate_digest(path, &self.trajectory_digest)?;
        validate_digest(path, &self.reward_evidence_digest)?;
        if self.assignments.is_empty() || !self.total_credit.is_finite() {
            return Err(error(path, "credit evidence requires finite assignments"));
        }
        let mut identifiers = HashSet::new();
        let mut assignment_keys = Vec::new();
        let mut expected = 0.0;
        for assignment in &self.assignments {
            nonempty(path, &assignment.assignment_id)?;
            nonempty(path, &assignment.subject_id)?;
            nonempty(path, &assignment.event_id)?;
            nonempty(path, &assignment.reward_component_id)?;
            if !identifiers.insert(assignment.assignment_id.as_str())
                || !assignment.credit.is_finite()
            {
                return Err(error(path, "invalid credit assignment identity or value"));
            }
            assignment.policy_epoch.validate_at(path)?;
            validate_log_probability(path, assignment.behavior_log_probability)?;
            let key = policy_key(&assignment.policy_epoch);
            if !assignment_keys.contains(&key) {
                assignment_keys.push(key);
            }
            expected += assignment.credit;
        }
        let declared: Vec<_> = self.policy_epochs.iter().map(policy_key).collect();
        if declared != assignment_keys || !close(self.total_credit, expected) {
            return Err(error(
                path,
                "credit total or policy epochs do not match assignments",
            ));
        }
        let mut required = vec![self.trajectory_id.clone(), self.reward_evidence_id.clone()];
        required.extend(self.policy_epochs.iter().map(policy_lineage_id));
        validate_lineage(path, &self.lineage, &required)
    }
}

impl Validate for CreditAssignmentEvidence {
    fn validate(&self) -> Result<(), ValidationError> {
        self.validate_at("credit_evidence")
    }
}

string_enum!(StalenessDisposition {
    Accept,
    Reweight,
    Reject
});

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StalenessReport {
    pub api_version: String,
    pub schema_version: String,
    pub kind: String,
    pub report_id: String,
    pub sample_id: String,
    pub trajectory_id: String,
    pub behavior_policy_epoch: PolicyEpoch,
    pub learner_policy_epoch: PolicyEpoch,
    pub epoch_lag: u64,
    pub maximum_allowed_lag: u64,
    pub stale: bool,
    pub disposition: StalenessDisposition,
    pub importance_sampling_weight: Option<f64>,
    pub assessed_at: String,
    pub lineage: Vec<LineageReference>,
}

impl StalenessReport {
    fn validate_at(&self, path: &str) -> Result<(), ValidationError> {
        validate_header(
            &self.api_version,
            &self.schema_version,
            &self.kind,
            "StalenessReport",
        )?;
        nonempty(path, &self.report_id)?;
        nonempty(path, &self.sample_id)?;
        nonempty(path, &self.trajectory_id)?;
        nonempty(path, &self.assessed_at)?;
        self.behavior_policy_epoch.validate_at(path)?;
        self.learner_policy_epoch.validate_at(path)?;
        if self.behavior_policy_epoch.policy_id != self.learner_policy_epoch.policy_id
            || self.behavior_policy_epoch.epoch > self.learner_policy_epoch.epoch
        {
            return Err(error(path, "invalid staleness policy comparison"));
        }
        let lag = self.learner_policy_epoch.epoch - self.behavior_policy_epoch.epoch;
        if self.epoch_lag != lag || self.stale != (lag > self.maximum_allowed_lag) {
            return Err(error(path, "staleness lag or flag is inconsistent"));
        }
        let positive_weight = self
            .importance_sampling_weight
            .is_some_and(|weight| weight.is_finite() && weight > 0.0);
        match self.disposition {
            StalenessDisposition::Accept
                if !self.stale && self.importance_sampling_weight.is_none() => {}
            StalenessDisposition::Reweight if self.stale && positive_weight => {}
            StalenessDisposition::Reject
                if self.stale && self.importance_sampling_weight.is_none() => {}
            _ => return Err(error(path, "staleness disposition is inconsistent")),
        }
        validate_lineage(
            path,
            &self.lineage,
            &[
                self.sample_id.clone(),
                self.trajectory_id.clone(),
                policy_lineage_id(&self.behavior_policy_epoch),
                policy_lineage_id(&self.learner_policy_epoch),
            ],
        )
    }
}

impl Validate for StalenessReport {
    fn validate(&self) -> Result<(), ValidationError> {
        self.validate_at("staleness_report")
    }
}

string_enum!(StateReuseMode {
    Exact,
    Converted,
    Recomputed,
    Incompatible
});

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StateReuseReport {
    pub api_version: String,
    pub schema_version: String,
    pub kind: String,
    pub report_id: String,
    pub source_capsule: EnvironmentStateCapsule,
    pub target_environment_id: String,
    pub target_policy_epoch: PolicyEpoch,
    pub target_compatibility_fingerprint: Digest,
    pub mode: StateReuseMode,
    pub compatible: bool,
    pub reused: bool,
    pub conversion_evidence: Option<EvidencePointer>,
    pub reason: String,
    pub assessed_at: String,
    pub lineage: Vec<LineageReference>,
}

impl StateReuseReport {
    fn validate_at(&self, path: &str) -> Result<(), ValidationError> {
        validate_header(
            &self.api_version,
            &self.schema_version,
            &self.kind,
            "StateReuseReport",
        )?;
        nonempty(path, &self.report_id)?;
        nonempty(path, &self.target_environment_id)?;
        nonempty(path, &self.reason)?;
        nonempty(path, &self.assessed_at)?;
        self.source_capsule.validate_at(path)?;
        self.target_policy_epoch.validate_at(path)?;
        validate_digest(path, &self.target_compatibility_fingerprint)?;
        let declared_policy = self
            .source_capsule
            .compatible_policy_digests
            .iter()
            .any(|digest| digest.value == self.target_policy_epoch.policy_digest.value);
        let same_fingerprint =
            self.target_compatibility_fingerprint == self.source_capsule.compatibility_fingerprint;
        match self.mode {
            StateReuseMode::Exact
                if declared_policy
                    && same_fingerprint
                    && self.compatible
                    && self.reused
                    && self.conversion_evidence.is_none() => {}
            StateReuseMode::Converted
                if declared_policy
                    && self.compatible
                    && self.reused
                    && self.conversion_evidence.is_some() =>
            {
                if let Some(evidence) = &self.conversion_evidence {
                    validate_evidence(path, evidence)?;
                }
            }
            StateReuseMode::Recomputed | StateReuseMode::Incompatible
                if !self.compatible && !self.reused && self.conversion_evidence.is_none() => {}
            _ => return Err(error(path, "incompatible state reuse cannot be silent")),
        }
        validate_lineage(
            path,
            &self.lineage,
            &[
                self.source_capsule.capsule_id.clone(),
                policy_lineage_id(&self.target_policy_epoch),
            ],
        )
    }
}

impl Validate for StateReuseReport {
    fn validate(&self) -> Result<(), ValidationError> {
        self.validate_at("state_reuse_report")
    }
}

string_enum!(TrainingSampleKind {
    Token,
    Action,
    Trajectory
});

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingSampleProvenance {
    pub sample_id: String,
    pub sample_kind: TrainingSampleKind,
    pub trajectory_id: String,
    pub trajectory_digest: Digest,
    pub reward_evidence_id: String,
    pub reward_evidence_digest: Digest,
    pub credit_evidence_id: String,
    pub credit_evidence_digest: Digest,
    pub event_ids: Vec<String>,
    pub token_ids: Vec<String>,
    pub action_ids: Vec<String>,
    pub behavior_policy_epoch: PolicyEpoch,
    pub behavior_log_probability: f64,
    pub target_policy_epoch: PolicyEpoch,
    pub importance_sampling_weight: f64,
    pub raw_sample: EvidencePointer,
    pub lineage: Vec<LineageReference>,
}

impl TrainingSampleProvenance {
    fn validate_at(&self, path: &str) -> Result<(), ValidationError> {
        nonempty(path, &self.sample_id)?;
        nonempty(path, &self.trajectory_id)?;
        nonempty(path, &self.reward_evidence_id)?;
        nonempty(path, &self.credit_evidence_id)?;
        validate_digest(path, &self.trajectory_digest)?;
        validate_digest(path, &self.reward_evidence_digest)?;
        validate_digest(path, &self.credit_evidence_digest)?;
        self.behavior_policy_epoch.validate_at(path)?;
        self.target_policy_epoch.validate_at(path)?;
        validate_log_probability(path, self.behavior_log_probability)?;
        if !self.importance_sampling_weight.is_finite() || self.importance_sampling_weight <= 0.0 {
            return Err(error(
                path,
                "importance sampling weight must be finite and positive",
            ));
        }
        validate_evidence(path, &self.raw_sample)?;
        if self.event_ids.is_empty() {
            return Err(error(path, "training samples must identify source events"));
        }
        match self.sample_kind {
            TrainingSampleKind::Token
                if !self.token_ids.is_empty() && self.action_ids.is_empty() => {}
            TrainingSampleKind::Action
                if !self.action_ids.is_empty() && self.token_ids.is_empty() => {}
            TrainingSampleKind::Trajectory
                if self.token_ids.is_empty() && self.action_ids.is_empty() => {}
            _ => {
                return Err(error(
                    path,
                    "sample kind does not match token/action provenance",
                ));
            }
        }
        for identifiers in [&self.event_ids, &self.token_ids, &self.action_ids] {
            let unique: HashSet<_> = identifiers.iter().collect();
            if unique.len() != identifiers.len() {
                return Err(error(path, "training sample source IDs contain duplicates"));
            }
        }
        validate_lineage(
            path,
            &self.lineage,
            &[
                self.trajectory_id.clone(),
                self.reward_evidence_id.clone(),
                self.credit_evidence_id.clone(),
                policy_lineage_id(&self.behavior_policy_epoch),
                policy_lineage_id(&self.target_policy_epoch),
            ],
        )
    }
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingBatchManifest {
    pub api_version: String,
    pub schema_version: String,
    pub kind: String,
    pub batch_id: String,
    pub policy_consistency: PolicyConsistency,
    pub learner_policy_epoch: PolicyEpoch,
    pub samples: Vec<TrainingSampleProvenance>,
    pub created_at: String,
    pub lineage: Vec<LineageReference>,
}

impl TrainingBatchManifest {
    fn validate_at(&self, path: &str) -> Result<(), ValidationError> {
        validate_header(
            &self.api_version,
            &self.schema_version,
            &self.kind,
            "TrainingBatchManifest",
        )?;
        nonempty(path, &self.batch_id)?;
        nonempty(path, &self.created_at)?;
        self.learner_policy_epoch.validate_at(path)?;
        if self.samples.is_empty() {
            return Err(error(path, "training batch must contain samples"));
        }
        let mut sample_ids = HashSet::new();
        let mut behavior_keys = HashSet::new();
        for sample in &self.samples {
            sample.validate_at(path)?;
            if !sample_ids.insert(sample.sample_id.as_str()) {
                return Err(error(path, "training sample IDs must be unique"));
            }
            if policy_key(&sample.target_policy_epoch) != policy_key(&self.learner_policy_epoch) {
                return Err(error(
                    path,
                    "sample target policy must equal learner policy",
                ));
            }
            behavior_keys.insert(policy_key(&sample.behavior_policy_epoch));
        }
        if self.policy_consistency == PolicyConsistency::Strict && behavior_keys.len() != 1 {
            return Err(error(
                path,
                "strict batches cannot mix behavior policy epochs",
            ));
        }
        let mut required: Vec<_> = self
            .samples
            .iter()
            .map(|item| item.sample_id.clone())
            .collect();
        required.push(policy_lineage_id(&self.learner_policy_epoch));
        validate_lineage(path, &self.lineage, &required)
    }
}

impl Validate for TrainingBatchManifest {
    fn validate(&self) -> Result<(), ValidationError> {
        self.validate_at("training_batch")
    }
}

string_enum!(LearningTransactionState {
    Created,
    EvidenceValidated,
    BatchAssembled,
    Trained,
    Evaluated,
    Committed,
    Aborted
});

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LearningStateTransition {
    pub sequence: u64,
    pub from_state: Option<LearningTransactionState>,
    pub to_state: LearningTransactionState,
    pub transitioned_at: String,
    pub actor: String,
    pub reason: String,
    pub evidence_digests: Vec<Digest>,
}

string_enum!(PromotionDecision {
    Promote,
    Hold,
    Rollback
});

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyPromotionCapsule {
    pub api_version: String,
    pub schema_version: String,
    pub kind: String,
    pub promotion_id: String,
    pub transaction_id: String,
    pub from_policy_epoch: PolicyEpoch,
    pub to_policy_epoch: PolicyEpoch,
    pub decision: PromotionDecision,
    pub evaluation_evidence: Vec<EvidencePointer>,
    pub approved_by: String,
    pub promoted_at: String,
    pub rollback_policy_epoch: Option<PolicyEpoch>,
    pub lineage: Vec<LineageReference>,
}

impl PolicyPromotionCapsule {
    fn validate_at(&self, path: &str) -> Result<(), ValidationError> {
        validate_header(
            &self.api_version,
            &self.schema_version,
            &self.kind,
            "PolicyPromotionCapsule",
        )?;
        nonempty(path, &self.promotion_id)?;
        nonempty(path, &self.transaction_id)?;
        nonempty(path, &self.approved_by)?;
        nonempty(path, &self.promoted_at)?;
        self.from_policy_epoch.validate_at(path)?;
        self.to_policy_epoch.validate_at(path)?;
        if self.evaluation_evidence.is_empty() {
            return Err(error(path, "promotion requires evaluation evidence"));
        }
        for evidence in &self.evaluation_evidence {
            validate_evidence(path, evidence)?;
        }
        if self.from_policy_epoch.policy_id != self.to_policy_epoch.policy_id {
            return Err(error(path, "promotion cannot change policy identity"));
        }
        match self.decision {
            PromotionDecision::Promote
                if self.to_policy_epoch.epoch > self.from_policy_epoch.epoch
                    && self.rollback_policy_epoch.is_none() => {}
            PromotionDecision::Hold
                if policy_key(&self.to_policy_epoch) == policy_key(&self.from_policy_epoch)
                    && self.rollback_policy_epoch.is_none() => {}
            PromotionDecision::Rollback
                if self
                    .rollback_policy_epoch
                    .as_ref()
                    .is_some_and(|policy| policy.epoch < self.from_policy_epoch.epoch) =>
            {
                if let Some(policy) = &self.rollback_policy_epoch {
                    policy.validate_at(path)?;
                }
            }
            _ => return Err(error(path, "invalid promotion epoch transition")),
        }
        validate_lineage(
            path,
            &self.lineage,
            &[
                self.transaction_id.clone(),
                policy_lineage_id(&self.from_policy_epoch),
                policy_lineage_id(&self.to_policy_epoch),
            ],
        )
    }
}

impl Validate for PolicyPromotionCapsule {
    fn validate(&self) -> Result<(), ValidationError> {
        self.validate_at("promotion")
    }
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LearningTransaction {
    pub api_version: String,
    pub schema_version: String,
    pub kind: String,
    pub transaction_id: String,
    pub state: LearningTransactionState,
    pub previous_state: Option<LearningTransactionState>,
    pub transitioned_at: String,
    pub transition_sequence: u64,
    pub transitions: Vec<LearningStateTransition>,
    pub source_policy_epoch: PolicyEpoch,
    pub candidate_policy_epoch: PolicyEpoch,
    pub branch_group: BranchGroup,
    pub reward_evidence: Vec<RewardEvidence>,
    pub credit_assignment_evidence: Vec<CreditAssignmentEvidence>,
    pub staleness_reports: Vec<StalenessReport>,
    pub state_reuse_reports: Vec<StateReuseReport>,
    pub training_batch: TrainingBatchManifest,
    pub promotion: Option<PolicyPromotionCapsule>,
    pub created_at: String,
    pub lineage: Vec<LineageReference>,
}

impl LearningTransaction {
    fn validate_at(&self, path: &str) -> Result<(), ValidationError> {
        validate_header(
            &self.api_version,
            &self.schema_version,
            &self.kind,
            "LearningTransaction",
        )?;
        nonempty(path, &self.transaction_id)?;
        nonempty(path, &self.transitioned_at)?;
        nonempty(path, &self.created_at)?;
        self.source_policy_epoch.validate_at(path)?;
        self.candidate_policy_epoch.validate_at(path)?;
        self.branch_group.validate_at(path)?;
        self.training_batch.validate_at(path)?;
        self.validate_transitions(path)?;
        if self.reward_evidence.is_empty() || self.credit_assignment_evidence.is_empty() {
            return Err(error(
                path,
                "transaction requires reward and credit evidence",
            ));
        }
        let trajectories: HashMap<_, _> = self
            .branch_group
            .trajectories
            .iter()
            .map(|item| (item.trajectory_id.as_str(), item))
            .collect();
        let mut rewards = HashMap::new();
        for reward in &self.reward_evidence {
            reward.validate_at(path)?;
            if rewards
                .insert(reward.reward_evidence_id.as_str(), reward)
                .is_some()
            {
                return Err(error(path, "reward evidence IDs must be unique"));
            }
            let Some(trajectory) = trajectories.get(reward.trajectory_id.as_str()) else {
                return Err(error(
                    path,
                    "reward references trajectory outside branch group",
                ));
            };
            if reward.trajectory_digest.value != hash_value(trajectory, path)? {
                return Err(error(path, "reward trajectory digest mismatch"));
            }
            Self::validate_reward_events(reward, trajectory, path)?;
        }
        let mut credits = HashMap::new();
        for credit in &self.credit_assignment_evidence {
            credit.validate_at(path)?;
            if credits
                .insert(credit.credit_evidence_id.as_str(), credit)
                .is_some()
            {
                return Err(error(path, "credit evidence IDs must be unique"));
            }
            let Some(trajectory) = trajectories.get(credit.trajectory_id.as_str()) else {
                return Err(error(path, "credit references unknown trajectory"));
            };
            let Some(reward) = rewards.get(credit.reward_evidence_id.as_str()) else {
                return Err(error(path, "credit references unknown reward"));
            };
            if credit.trajectory_digest.value != hash_value(trajectory, path)?
                || credit.reward_evidence_digest.value != hash_value(reward, path)?
                || !close(credit.total_credit, reward.aggregate_reward)
            {
                return Err(error(
                    path,
                    "credit evidence digest or conservation mismatch",
                ));
            }
            Self::validate_credit_subjects(credit, trajectory, reward, path)?;
        }
        let mut samples = HashMap::new();
        for sample in &self.training_batch.samples {
            if samples.insert(sample.sample_id.as_str(), sample).is_some() {
                return Err(error(path, "sample IDs must be unique"));
            }
            let Some(trajectory) = trajectories.get(sample.trajectory_id.as_str()) else {
                return Err(error(path, "sample references unknown trajectory"));
            };
            let Some(reward) = rewards.get(sample.reward_evidence_id.as_str()) else {
                return Err(error(path, "sample references unknown reward"));
            };
            let Some(credit) = credits.get(sample.credit_evidence_id.as_str()) else {
                return Err(error(path, "sample references unknown credit evidence"));
            };
            if sample.trajectory_digest.value != hash_value(trajectory, path)?
                || sample.reward_evidence_digest.value != hash_value(reward, path)?
                || sample.credit_evidence_digest.value != hash_value(credit, path)?
            {
                return Err(error(path, "sample source digest mismatch"));
            }
            Self::validate_sample_source(sample, trajectory, path)?;
        }
        self.validate_staleness(&samples, path)?;
        if policy_key(&self.training_batch.learner_policy_epoch)
            != policy_key(&self.source_policy_epoch)
            || self.source_policy_epoch.policy_id != self.candidate_policy_epoch.policy_id
            || self.candidate_policy_epoch.epoch <= self.source_policy_epoch.epoch
        {
            return Err(error(
                path,
                "transaction source, learner, and candidate policies conflict",
            ));
        }
        if self.state_reuse_reports.is_empty() {
            return Err(error(
                path,
                "transaction must explicitly report state reuse",
            ));
        }
        for report in &self.state_reuse_reports {
            report.validate_at(path)?;
        }
        if self.state == LearningTransactionState::Committed
            && !self
                .promotion
                .as_ref()
                .is_some_and(|promotion| promotion.decision == PromotionDecision::Promote)
        {
            return Err(error(path, "committed transaction requires promotion"));
        }
        if let Some(promotion) = &self.promotion {
            promotion.validate_at(path)?;
            if promotion.transaction_id != self.transaction_id
                || policy_key(&promotion.from_policy_epoch) != policy_key(&self.source_policy_epoch)
                || policy_key(&promotion.to_policy_epoch)
                    != policy_key(&self.candidate_policy_epoch)
            {
                return Err(error(path, "promotion does not match transaction"));
            }
        }
        let mut required = vec![
            self.branch_group.group_id.clone(),
            self.training_batch.batch_id.clone(),
        ];
        required.extend(
            self.reward_evidence
                .iter()
                .map(|item| item.reward_evidence_id.clone()),
        );
        required.extend(
            self.credit_assignment_evidence
                .iter()
                .map(|item| item.credit_evidence_id.clone()),
        );
        validate_lineage(path, &self.lineage, &required)
    }

    fn validate_transitions(&self, path: &str) -> Result<(), ValidationError> {
        if self.transitions.is_empty() {
            return Err(error(path, "transition history must not be empty"));
        }
        let mut current = None;
        for (index, transition) in self.transitions.iter().enumerate() {
            nonempty(path, &transition.transitioned_at)?;
            nonempty(path, &transition.actor)?;
            nonempty(path, &transition.reason)?;
            for digest in &transition.evidence_digests {
                validate_digest(path, digest)?;
            }
            if transition.sequence != index as u64 + 1 || transition.from_state != current {
                return Err(error(
                    path,
                    "transition sequence or source state is invalid",
                ));
            }
            if current.is_none() {
                if transition.to_state != LearningTransactionState::Created {
                    return Err(error(path, "first transition must create transaction"));
                }
            } else if !legal_transition(current, transition.to_state) {
                return Err(error(path, "illegal learning transaction transition"));
            }
            current = Some(transition.to_state);
        }
        let last = self
            .transitions
            .last()
            .ok_or_else(|| error(path, "transition history must not be empty"))?;
        if self.state != last.to_state
            || self.previous_state != last.from_state
            || self.transition_sequence != last.sequence
            || self.transitioned_at != last.transitioned_at
        {
            return Err(error(
                path,
                "transaction transition fields do not match history",
            ));
        }
        Ok(())
    }

    fn validate_reward_events(
        reward: &RewardEvidence,
        trajectory: &TrajectoryCapsule,
        path: &str,
    ) -> Result<(), ValidationError> {
        let events: HashMap<_, _> = trajectory
            .events
            .iter()
            .map(|item| (item.event_id.as_str(), item))
            .collect();
        for component in &reward.components {
            for event_id in &component.event_ids {
                let Some(event) = events.get(event_id.as_str()) else {
                    return Err(error(path, "reward references unknown event"));
                };
                if policy_key(&component.policy_epoch) != policy_key(&event.policy_epoch) {
                    return Err(error(path, "reward policy does not match source event"));
                }
            }
        }
        Ok(())
    }

    fn validate_credit_subjects(
        credit: &CreditAssignmentEvidence,
        trajectory: &TrajectoryCapsule,
        reward: &RewardEvidence,
        path: &str,
    ) -> Result<(), ValidationError> {
        let events: HashMap<_, _> = trajectory
            .events
            .iter()
            .map(|item| (item.event_id.as_str(), item))
            .collect();
        let tokens: HashMap<_, _> = trajectory
            .tokens
            .iter()
            .map(|item| (item.token_id.as_str(), item))
            .collect();
        let actions: HashMap<_, _> = trajectory
            .actions
            .iter()
            .map(|item| (item.action_id.as_str(), item))
            .collect();
        let component_ids: HashSet<_> = reward
            .components
            .iter()
            .map(|item| item.component_id.as_str())
            .collect();
        for assignment in &credit.assignments {
            let Some(event) = events.get(assignment.event_id.as_str()) else {
                return Err(error(path, "credit references unknown event"));
            };
            if !component_ids.contains(assignment.reward_component_id.as_str()) {
                return Err(error(path, "credit references unknown reward component"));
            }
            let (expected_policy, expected_log) = match assignment.subject_kind {
                CreditSubjectKind::Event if assignment.subject_id == assignment.event_id => {
                    (&event.policy_epoch, 0.0)
                }
                CreditSubjectKind::Token => {
                    let Some(token) = tokens.get(assignment.subject_id.as_str()) else {
                        return Err(error(path, "credit references unknown token"));
                    };
                    if token.event_id != assignment.event_id {
                        return Err(error(path, "credit token event mismatch"));
                    }
                    (&token.policy_epoch, token.behavior_log_probability)
                }
                CreditSubjectKind::Action => {
                    let Some(action) = actions.get(assignment.subject_id.as_str()) else {
                        return Err(error(path, "credit references unknown action"));
                    };
                    if action.event_id != assignment.event_id {
                        return Err(error(path, "credit action event mismatch"));
                    }
                    (&action.policy_epoch, action.behavior_log_probability)
                }
                CreditSubjectKind::Event => {
                    return Err(error(path, "credit event subject mismatch"));
                }
            };
            if policy_key(&assignment.policy_epoch) != policy_key(expected_policy)
                || !close(assignment.behavior_log_probability, expected_log)
            {
                return Err(error(
                    path,
                    "credit policy or behavior log probability mismatch",
                ));
            }
        }
        Ok(())
    }

    fn validate_sample_source(
        sample: &TrainingSampleProvenance,
        trajectory: &TrajectoryCapsule,
        path: &str,
    ) -> Result<(), ValidationError> {
        let event_ids: HashSet<_> = trajectory
            .events
            .iter()
            .map(|item| item.event_id.as_str())
            .collect();
        if sample
            .event_ids
            .iter()
            .any(|event_id| !event_ids.contains(event_id.as_str()))
        {
            return Err(error(path, "sample references unknown event"));
        }
        let scoped_events: HashSet<_> = sample.event_ids.iter().map(String::as_str).collect();
        let behavior_key = policy_key(&sample.behavior_policy_epoch);
        let mut logs = Vec::new();
        match sample.sample_kind {
            TrainingSampleKind::Token => {
                for token_id in &sample.token_ids {
                    let Some(token) = trajectory
                        .tokens
                        .iter()
                        .find(|item| &item.token_id == token_id)
                    else {
                        return Err(error(path, "sample references unknown token"));
                    };
                    if !scoped_events.contains(token.event_id.as_str())
                        || policy_key(&token.policy_epoch) != behavior_key
                    {
                        return Err(error(path, "sample token scope or policy mismatch"));
                    }
                    logs.push(token.behavior_log_probability);
                }
            }
            TrainingSampleKind::Action => {
                for action_id in &sample.action_ids {
                    let Some(action) = trajectory
                        .actions
                        .iter()
                        .find(|item| &item.action_id == action_id)
                    else {
                        return Err(error(path, "sample references unknown action"));
                    };
                    if !scoped_events.contains(action.event_id.as_str())
                        || policy_key(&action.policy_epoch) != behavior_key
                    {
                        return Err(error(path, "sample action scope or policy mismatch"));
                    }
                    logs.push(action.behavior_log_probability);
                }
            }
            TrainingSampleKind::Trajectory => {
                for event in &trajectory.events {
                    if scoped_events.contains(event.event_id.as_str())
                        && policy_key(&event.policy_epoch) != behavior_key
                    {
                        return Err(error(path, "trajectory sample crosses policy epochs"));
                    }
                }
                logs.extend(
                    trajectory
                        .tokens
                        .iter()
                        .filter(|item| scoped_events.contains(item.event_id.as_str()))
                        .map(|item| item.behavior_log_probability),
                );
                logs.extend(
                    trajectory
                        .actions
                        .iter()
                        .filter(|item| scoped_events.contains(item.event_id.as_str()))
                        .map(|item| item.behavior_log_probability),
                );
            }
        }
        if logs.is_empty() || !close(sample.behavior_log_probability, logs.iter().sum()) {
            return Err(error(path, "sample behavior log probability mismatch"));
        }
        Ok(())
    }

    fn validate_staleness(
        &self,
        samples: &HashMap<&str, &TrainingSampleProvenance>,
        path: &str,
    ) -> Result<(), ValidationError> {
        if self.staleness_reports.len() != samples.len() {
            return Err(error(path, "every sample requires one staleness report"));
        }
        let mut report_samples = HashSet::new();
        for report in &self.staleness_reports {
            report.validate_at(path)?;
            if !report_samples.insert(report.sample_id.as_str()) {
                return Err(error(path, "staleness sample IDs must be unique"));
            }
            let Some(sample) = samples.get(report.sample_id.as_str()) else {
                return Err(error(path, "staleness references unknown sample"));
            };
            if report.trajectory_id != sample.trajectory_id
                || policy_key(&report.behavior_policy_epoch)
                    != policy_key(&sample.behavior_policy_epoch)
                || policy_key(&report.learner_policy_epoch)
                    != policy_key(&self.training_batch.learner_policy_epoch)
            {
                return Err(error(path, "staleness report does not match sample"));
            }
        }
        Ok(())
    }
}

impl Validate for LearningTransaction {
    fn validate(&self) -> Result<(), ValidationError> {
        self.validate_at("learning_transaction")
    }
}

fn hash_value<T: Serialize>(value: &T, path: &str) -> Result<String, ValidationError> {
    canonical_hash(value)
        .map_err(|problem| error(path, format!("canonical hash failed: {problem}")))
}

fn legal_transition(from: Option<LearningTransactionState>, to: LearningTransactionState) -> bool {
    matches!(
        (from, to),
        (
            Some(LearningTransactionState::Created),
            LearningTransactionState::EvidenceValidated | LearningTransactionState::Aborted
        ) | (
            Some(LearningTransactionState::EvidenceValidated),
            LearningTransactionState::BatchAssembled | LearningTransactionState::Aborted
        ) | (
            Some(LearningTransactionState::BatchAssembled),
            LearningTransactionState::Trained | LearningTransactionState::Aborted
        ) | (
            Some(LearningTransactionState::Trained),
            LearningTransactionState::Evaluated | LearningTransactionState::Aborted
        ) | (
            Some(LearningTransactionState::Evaluated),
            LearningTransactionState::Committed | LearningTransactionState::Aborted
        )
    )
}

string_enum!(BranchWorkloadStatus {
    Completed,
    Failed,
    Cancelled
});

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BranchWorkloadRequest {
    pub request_id: String,
    pub branch_point_id: String,
    pub trajectory_id: String,
    pub ordinal: u64,
    pub scheduled_offset_ms: f64,
    pub input_digest: Digest,
    pub output_digest: Digest,
    pub status: BranchWorkloadStatus,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BranchWorkloadTrace {
    pub api_version: String,
    pub schema_version: String,
    pub kind: String,
    pub trace_id: String,
    pub branch_group_id: String,
    pub environment_id: String,
    pub seed: u64,
    pub started_at: String,
    pub completed_at: String,
    pub requests: Vec<BranchWorkloadRequest>,
    pub raw_trace_uri: String,
    pub raw_trace_digest: Digest,
    pub lineage: Vec<LineageReference>,
}

impl Validate for BranchWorkloadTrace {
    fn validate(&self) -> Result<(), ValidationError> {
        validate_header(
            &self.api_version,
            &self.schema_version,
            &self.kind,
            "BranchWorkloadTrace",
        )?;
        nonempty("branch_workload_trace", &self.trace_id)?;
        nonempty("branch_workload_trace", &self.branch_group_id)?;
        nonempty("branch_workload_trace", &self.environment_id)?;
        nonempty("branch_workload_trace", &self.started_at)?;
        nonempty("branch_workload_trace", &self.completed_at)?;
        nonempty("branch_workload_trace", &self.raw_trace_uri)?;
        validate_digest("branch_workload_trace", &self.raw_trace_digest)?;
        if self.requests.is_empty() {
            return Err(error("branch_workload_trace", "requests must not be empty"));
        }
        let mut identifiers = HashSet::new();
        for (index, request) in self.requests.iter().enumerate() {
            nonempty("branch_workload_trace", &request.request_id)?;
            nonempty("branch_workload_trace", &request.branch_point_id)?;
            nonempty("branch_workload_trace", &request.trajectory_id)?;
            validate_digest("branch_workload_trace", &request.input_digest)?;
            validate_digest("branch_workload_trace", &request.output_digest)?;
            if request.ordinal != index as u64
                || !identifiers.insert(request.request_id.as_str())
                || !request.scheduled_offset_ms.is_finite()
                || request.scheduled_offset_ms < 0.0
            {
                return Err(error(
                    "branch_workload_trace",
                    "request ordinals, IDs, and offsets must be valid",
                ));
            }
        }
        let mut required = vec![self.branch_group_id.clone()];
        required.extend(self.requests.iter().map(|item| item.trajectory_id.clone()));
        validate_lineage("branch_workload_trace", &self.lineage, &required)
    }
}
