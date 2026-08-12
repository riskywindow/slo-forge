//! `BranchFabric` characterization trace wire models.
//!
//! These event records are deliberately separate from the legacy Helix
//! `BranchWorkloadTrace`, which describes a scheduled workload. The records in
//! this module form the high-volume, event-oriented Rust/Python boundary used
//! for measurement and future architecture replay.

use std::collections::{BTreeMap, HashSet};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{Validate, ValidationError};

pub const BRANCH_WORKLOAD_TRACE_SCHEMA_VERSION: &str =
    "sloforge.branchfabric.branch-workload-event/v1";
pub const STATE_OPERATION_TRACE_SCHEMA_VERSION: &str =
    "sloforge.branchfabric.state-operation-event/v1";
pub const BRANCH_WORKLOAD_TRACE_KIND: &str = "BranchWorkloadTraceEvent";
pub const STATE_OPERATION_TRACE_KIND: &str = "StateOperationTraceEvent";
pub const BRANCHFABRIC_TRACE_PRODUCER_VERSION: &str = "sloforge-helix-characterization/1";
pub const MAX_TRACE_ATTRIBUTES: usize = 64;
pub const MAX_TRACE_ATTRIBUTE_KEY_LENGTH: usize = 128;
pub const MAX_TRACE_ATTRIBUTE_STRING_LENGTH: usize = 4096;

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

macro_rules! snake_case_enum {
    ($name:ident { $($variant:ident),+ $(,)? }) => {
        #[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, JsonSchema, PartialEq, Serialize)]
        #[serde(rename_all = "snake_case")]
        pub enum $name {
            $($variant),+
        }
    };
}

macro_rules! screaming_snake_case_enum {
    ($name:ident { $($variant:ident),+ $(,)? }) => {
        #[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, JsonSchema, PartialEq, Serialize)]
        #[serde(rename_all = "SCREAMING_SNAKE_CASE")]
        pub enum $name {
            $($variant),+
        }
    };
}

snake_case_enum!(TraceCollectionLevel {
    Disabled,
    Minimal,
    Full
});

screaming_snake_case_enum!(TraceProvenance {
    Synthetic,
    Replayed,
    HardwareBackedReal,
    SimulatedHardware
});

screaming_snake_case_enum!(TimingMeasurementClass {
    Synthetic,
    Replayed,
    HardwareBackedReal,
    SimulatedHardware
});

snake_case_enum!(TraceClockSource {
    Monotonic,
    MonotonicRaw,
    PerfCounter,
    CudaGlobalTimer,
    Cupti,
    Synthetic
});

snake_case_enum!(TraceMemoryLocation {
    GpuHbm,
    GpuPeerHbm,
    HostDram,
    PinnedMemory,
    LocalNvme,
    RemoteStorage,
    Nic,
    TransportBuffer,
    Unknown
});

snake_case_enum!(TraceStateSegment {
    Model,
    TokenHistory,
    Kv,
    Recurrent,
    Sampler,
    GuidedDecoding,
    Workflow,
    Environment,
    Filesystem,
    Database,
    ProcessReconstruction,
    Transaction,
    Integrity,
    RuntimeReconstructible,
    Unknown
});

snake_case_enum!(TraceTransportType {
    None,
    MemoryCopy,
    Pcie,
    Nvlink,
    Nccl,
    Tcp,
    Rdma,
    SharedMemory,
    Storage,
    Synthetic
});

screaming_snake_case_enum!(BranchTraceOperationType {
    BranchPoint,
    Capture,
    EnvironmentFork,
    BranchFork,
    BranchReady,
    BranchDivergence,
    BranchPrune,
    BranchAbort,
    BranchCommit,
    BranchComplete,
    BranchMigration,
    Checkpoint,
    Rollout,
    Reward,
    Train,
    Evaluate,
    Canary,
    Promote,
    Rollback,
    LearningTransactionStage,
    StateAlloc,
    StateMap,
    StatePublish,
    StateFork,
    StateCow,
    StateAppend,
    StateRead,
    StateWrite,
    StateSnapshot,
    StateDelta,
    StateReshard,
    StateTranspose,
    StateRepack,
    StateQuantize,
    StateDequantize,
    StateCompress,
    StateDecompress,
    StateHash,
    StateChecksum,
    StateEncrypt,
    StateDecrypt,
    StateSend,
    StateReceive,
    StateMulticast,
    StateAck,
    StateRetry,
    StateCommit,
    StateAbort,
    StateReclaim,
    StateFree
});

screaming_snake_case_enum!(StateTraceOperationType {
    StateAlloc,
    StateMap,
    StatePublish,
    StateFork,
    StateCow,
    StateAppend,
    StateRead,
    StateWrite,
    StateSnapshot,
    StateDelta,
    StateReshard,
    StateTranspose,
    StateRepack,
    StateQuantize,
    StateDequantize,
    StateCompress,
    StateDecompress,
    StateHash,
    StateChecksum,
    StateEncrypt,
    StateDecrypt,
    StateSend,
    StateReceive,
    StateMulticast,
    StateAck,
    StateRetry,
    StateCommit,
    StateAbort,
    StateReclaim,
    StateFree
});

snake_case_enum!(TraceOperationResult {
    Success,
    Failure,
    Retry,
    Skipped
});

/// A bounded scalar attached to an event.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(untagged)]
pub enum TraceAttributeValue {
    Bool(bool),
    SignedInteger(i64),
    UnsignedInteger(u64),
    Float(f64),
    String(String),
    Null(()),
}

// Python-compatible public spellings for the shared event vocabulary.
pub type TraceLevel = TraceCollectionLevel;
pub type WorkloadProvenance = TraceProvenance;
pub type ClockSource = TraceClockSource;
pub type MemoryLocation = TraceMemoryLocation;
pub type StateSegment = TraceStateSegment;
pub type TransportType = TraceTransportType;
pub type BranchOperationType = BranchTraceOperationType;
pub type StateOperationType = StateTraceOperationType;
pub type OperationResult = TraceOperationResult;

fn validate_attributes(
    path: &str,
    attributes: &BTreeMap<String, TraceAttributeValue>,
) -> Result<(), ValidationError> {
    if attributes.len() > MAX_TRACE_ATTRIBUTES {
        return Err(error(
            path,
            format!("must contain at most {MAX_TRACE_ATTRIBUTES} entries"),
        ));
    }
    for (key, value) in attributes {
        if key.is_empty() || key.chars().count() > MAX_TRACE_ATTRIBUTE_KEY_LENGTH {
            return Err(error(path, "attribute keys must contain 1..128 characters"));
        }
        match value {
            TraceAttributeValue::String(value)
                if value.chars().count() > MAX_TRACE_ATTRIBUTE_STRING_LENGTH =>
            {
                return Err(error(
                    path,
                    "attribute string values must contain at most 4096 characters",
                ));
            }
            TraceAttributeValue::Float(value) if !value.is_finite() => {
                return Err(error(path, "attribute floats must be finite"));
            }
            _ => {}
        }
    }
    Ok(())
}

struct CommonValidation<'a> {
    path: &'a str,
    trace_producer_version: &'a str,
    trace_id: &'a str,
    session_id: &'a str,
    branch_group_id: Option<&'a str>,
    host: &'a str,
    alignment_confidence: f64,
    fanout: u64,
    attributes: &'a BTreeMap<String, TraceAttributeValue>,
    content_hash: &'a str,
}

fn validate_common(fields: &CommonValidation<'_>) -> Result<(), ValidationError> {
    nonempty(fields.path, fields.trace_producer_version)?;
    nonempty(fields.path, fields.trace_id)?;
    nonempty(fields.path, fields.session_id)?;
    if let Some(branch_group_id) = fields.branch_group_id {
        nonempty(fields.path, branch_group_id)?;
    }
    nonempty(fields.path, fields.host)?;
    if !fields.alignment_confidence.is_finite()
        || !(0.0..=1.0).contains(&fields.alignment_confidence)
    {
        return Err(error(
            fields.path,
            "alignment confidence must be finite and in [0, 1]",
        ));
    }
    if fields.fanout == 0 {
        return Err(error(fields.path, "fanout must be at least one"));
    }
    validate_attributes(fields.path, fields.attributes)?;
    if !valid_sha256(fields.content_hash) {
        return Err(error(fields.path, "content hash must be lowercase SHA-256"));
    }
    Ok(())
}

/// One canonical high-level branch or state-lifecycle observation.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
#[allow(clippy::struct_excessive_bools)]
pub struct BranchWorkloadTraceEventV1 {
    pub schema_version: String,
    pub kind: String,
    pub trace_producer_version: String,
    pub collection_level: TraceCollectionLevel,
    pub provenance: TraceProvenance,
    pub timing_measurement_class: TimingMeasurementClass,
    pub trace_id: String,
    pub session_id: String,
    pub branch_group_id: Option<String>,
    pub branch_id: Option<String>,
    pub parent_branch_id: Option<String>,
    pub policy_epoch: Option<String>,
    pub environment_id: Option<String>,
    pub transaction_id: Option<String>,
    pub host: String,
    pub process_id: u64,
    pub rank: Option<u64>,
    pub device: Option<String>,
    pub monotonic_timestamp_ns: u64,
    pub normalized_timestamp_ns: u64,
    pub duration_ns: u64,
    pub clock_source: TraceClockSource,
    pub alignment_confidence: f64,
    pub operation_type: BranchTraceOperationType,
    pub logical_state_id: Option<String>,
    pub physical_state_id: Option<String>,
    pub state_segment: TraceStateSegment,
    pub page: Option<u64>,
    pub version: Option<u64>,
    pub source_epoch: Option<u64>,
    pub destination_epoch: Option<u64>,
    pub logical_bytes: u64,
    pub physical_bytes: u64,
    pub compressed_bytes: u64,
    pub transferred_bytes: u64,
    pub metadata_bytes: u64,
    pub location: TraceMemoryLocation,
    pub source_location: TraceMemoryLocation,
    pub destination_location: TraceMemoryLocation,
    pub shared_root: bool,
    pub private_suffix: bool,
    pub cow_allocation: bool,
    pub queue_delay_ns: u64,
    pub execution_latency_ns: u64,
    pub transfer_latency_ns: u64,
    pub transform_latency_ns: u64,
    pub wait_latency_ns: u64,
    pub cpu_cycles: Option<u64>,
    pub cpu_time_ns: Option<u64>,
    pub gpu_duration_ns: Option<u64>,
    pub transport_type: TraceTransportType,
    pub transport_source: Option<String>,
    pub transport_destination: Option<String>,
    pub chunk_size_bytes: Option<u64>,
    pub fanout: u64,
    pub retransmission: bool,
    pub error: Option<String>,
    pub gpu_model: Option<String>,
    pub nic: Option<String>,
    pub numa_node: Option<u64>,
    pub pcie_path: Option<String>,
    pub network_rail: Option<String>,
    pub memory_tier: Option<String>,
    #[serde(default)]
    #[schemars(default)]
    pub attributes: BTreeMap<String, TraceAttributeValue>,
    pub event_sequence: u64,
    pub content_hash: String,
}

impl Validate for BranchWorkloadTraceEventV1 {
    fn validate(&self) -> Result<(), ValidationError> {
        if self.schema_version != BRANCH_WORKLOAD_TRACE_SCHEMA_VERSION {
            return Err(error(
                "schema_version",
                "unsupported branch trace schema version",
            ));
        }
        if self.kind != BRANCH_WORKLOAD_TRACE_KIND {
            return Err(error("kind", "must be BranchWorkloadTraceEvent"));
        }
        validate_common(&CommonValidation {
            path: "branch_workload_trace_event",
            trace_producer_version: &self.trace_producer_version,
            trace_id: &self.trace_id,
            session_id: &self.session_id,
            branch_group_id: self.branch_group_id.as_deref(),
            host: &self.host,
            alignment_confidence: self.alignment_confidence,
            fanout: self.fanout,
            attributes: &self.attributes,
            content_hash: &self.content_hash,
        })?;
        if let (Some(branch_id), Some(parent_branch_id)) = (&self.branch_id, &self.parent_branch_id)
            && branch_id == parent_branch_id
        {
            return Err(error("parent_branch_id", "branch cannot be its own parent"));
        }
        if self
            .error
            .as_ref()
            .is_some_and(|error| error.trim().is_empty())
        {
            return Err(error("error", "must be non-empty when present"));
        }
        if self.cow_allocation && self.operation_type != BranchTraceOperationType::StateCow {
            return Err(error(
                "cow_allocation",
                "is only valid for STATE_COW operations",
            ));
        }
        if self.chunk_size_bytes == Some(0) {
            return Err(error(
                "chunk_size_bytes",
                "must be at least one when present",
            ));
        }
        Ok(())
    }
}

/// One lower-level state operation suitable for future architecture replay.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StateOperationTraceEventV1 {
    pub schema_version: String,
    pub kind: String,
    pub trace_producer_version: String,
    pub collection_level: TraceCollectionLevel,
    pub provenance: TraceProvenance,
    pub timing_measurement_class: TimingMeasurementClass,
    pub trace_id: String,
    pub session_id: String,
    pub branch_group_id: Option<String>,
    pub logical_state_id: String,
    pub branch_id: Option<String>,
    pub tenant_id: String,
    pub security_domain: String,
    pub host: String,
    pub process_id: u64,
    pub rank: Option<u64>,
    pub device: Option<String>,
    pub monotonic_timestamp_ns: u64,
    pub normalized_timestamp_ns: u64,
    pub duration_ns: u64,
    pub clock_source: TraceClockSource,
    pub alignment_confidence: f64,
    pub operation_type: StateTraceOperationType,
    pub state_segment: TraceStateSegment,
    pub source_physical_representation: String,
    pub destination_physical_representation: String,
    pub bytes: u64,
    /// Zero means N/A for a nonpaged or unaligned operation.
    pub alignment_bytes: u64,
    /// Zero means N/A for a nonpaged operation.
    pub page_size_bytes: u64,
    /// Zero means N/A for an unchunked operation.
    pub chunk_size_bytes: u64,
    pub fanout: u64,
    pub dependency_event_ids: Vec<String>,
    pub concurrency: u64,
    pub queue_delay_ns: u64,
    pub operation_latency_ns: u64,
    pub cpu_time_ns: u64,
    pub gpu_time_ns: u64,
    pub transfer_time_ns: u64,
    pub result: TraceOperationResult,
    pub failure: Option<String>,
    pub state_epoch: u64,
    pub source_location: TraceMemoryLocation,
    pub destination_location: TraceMemoryLocation,
    pub transport_type: TraceTransportType,
    #[serde(default)]
    #[schemars(default)]
    pub attributes: BTreeMap<String, TraceAttributeValue>,
    pub event_sequence: u64,
    pub content_hash: String,
}

impl Validate for StateOperationTraceEventV1 {
    fn validate(&self) -> Result<(), ValidationError> {
        if self.schema_version != STATE_OPERATION_TRACE_SCHEMA_VERSION {
            return Err(error(
                "schema_version",
                "unsupported state trace schema version",
            ));
        }
        if self.kind != STATE_OPERATION_TRACE_KIND {
            return Err(error("kind", "must be StateOperationTraceEvent"));
        }
        validate_common(&CommonValidation {
            path: "state_operation_trace_event",
            trace_producer_version: &self.trace_producer_version,
            trace_id: &self.trace_id,
            session_id: &self.session_id,
            branch_group_id: self.branch_group_id.as_deref(),
            host: &self.host,
            alignment_confidence: self.alignment_confidence,
            fanout: self.fanout,
            attributes: &self.attributes,
            content_hash: &self.content_hash,
        })?;
        nonempty("logical_state_id", &self.logical_state_id)?;
        nonempty("tenant_id", &self.tenant_id)?;
        nonempty("security_domain", &self.security_domain)?;
        nonempty(
            "source_physical_representation",
            &self.source_physical_representation,
        )?;
        nonempty(
            "destination_physical_representation",
            &self.destination_physical_representation,
        )?;
        if self.concurrency == 0 {
            return Err(error("concurrency", "must be at least one"));
        }
        let mut dependencies = HashSet::new();
        if self
            .dependency_event_ids
            .iter()
            .any(|dependency| !dependencies.insert(dependency))
        {
            return Err(error("dependency_event_ids", "must be unique"));
        }
        match (&self.result, &self.failure) {
            (TraceOperationResult::Failure, None) => {
                return Err(error("failure", "failed operations require failure detail"));
            }
            (TraceOperationResult::Failure, Some(_)) | (_, None) => {}
            (_, Some(_)) => {
                return Err(error(
                    "failure",
                    "only failed operations may carry failure detail",
                ));
            }
        }
        Ok(())
    }
}

/// Either canonical `BranchFabric` characterization event shape.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(untagged)]
pub enum BranchFabricTraceEventV1 {
    Branch(Box<BranchWorkloadTraceEventV1>),
    State(Box<StateOperationTraceEventV1>),
}

impl Validate for BranchFabricTraceEventV1 {
    fn validate(&self) -> Result<(), ValidationError> {
        match self {
            Self::Branch(event) => event.validate(),
            Self::State(event) => event.validate(),
        }
    }
}

pub type BranchWorkloadEventV1 = BranchWorkloadTraceEventV1;
pub type StateOperationEventV1 = StateOperationTraceEventV1;
pub type TraceEventV1 = BranchFabricTraceEventV1;
