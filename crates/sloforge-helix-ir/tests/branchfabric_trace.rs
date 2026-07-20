#![recursion_limit = "512"]

use serde_json::{Value, json};
use sloforge_helix_ir::{
    BranchFabricTraceEventV1, BranchWorkloadTraceEventV1, StateOperationTraceEventV1,
    canonical_json, from_json,
};

fn branch_event() -> Value {
    json!({
        "schema_version": "sloforge.branchfabric.branch-workload-event/v1",
        "kind": "BranchWorkloadTraceEvent",
        "trace_producer_version": "sloforge-helix-characterization/1",
        "collection_level": "full",
        "provenance": "SYNTHETIC",
        "timing_measurement_class": "HARDWARE_BACKED_REAL",
        "trace_id": "trace-seed-41",
        "session_id": "session-seed-41",
        "branch_group_id": null,
        "branch_id": "branch-a",
        "parent_branch_id": "root",
        "policy_epoch": "policy@0",
        "environment_id": "environment-1",
        "transaction_id": null,
        "host": "host-a",
        "process_id": 314,
        "rank": null,
        "device": "cpu",
        "monotonic_timestamp_ns": 1000,
        "normalized_timestamp_ns": 250,
        "duration_ns": 90,
        "clock_source": "perf_counter",
        "alignment_confidence": 1.0,
        "operation_type": "BRANCH_FORK",
        "logical_state_id": "logical-1",
        "physical_state_id": "physical-1",
        "state_segment": "kv",
        "page": null,
        "version": 2,
        "source_epoch": 1,
        "destination_epoch": 2,
        "logical_bytes": 8192,
        "physical_bytes": 4096,
        "compressed_bytes": 0,
        "transferred_bytes": 0,
        "metadata_bytes": 128,
        "location": "host_dram",
        "source_location": "host_dram",
        "destination_location": "host_dram",
        "shared_root": true,
        "private_suffix": false,
        "cow_allocation": false,
        "queue_delay_ns": 3,
        "execution_latency_ns": 80,
        "transfer_latency_ns": 0,
        "transform_latency_ns": 0,
        "wait_latency_ns": 7,
        "cpu_cycles": null,
        "cpu_time_ns": 75,
        "gpu_duration_ns": null,
        "transport_type": "none",
        "transport_source": null,
        "transport_destination": null,
        "chunk_size_bytes": null,
        "fanout": 4,
        "retransmission": false,
        "error": null,
        "gpu_model": null,
        "nic": null,
        "numa_node": null,
        "pcie_path": null,
        "network_rail": null,
        "memory_tier": "unified_dram",
        "attributes": {
            "branch_ready": true,
            "measurement_note": "host timing of a synthetic workload",
            "sample": 41,
            "scale": 0.5,
            "unset": null
        },
        "event_sequence": 7,
        "content_hash": "0".repeat(64)
    })
}

fn state_event() -> Value {
    json!({
        "schema_version": "sloforge.branchfabric.state-operation-event/v1",
        "kind": "StateOperationTraceEvent",
        "trace_producer_version": "sloforge-helix-characterization/1",
        "collection_level": "full",
        "provenance": "REPLAYED",
        "timing_measurement_class": "HARDWARE_BACKED_REAL",
        "trace_id": "trace-seed-41",
        "session_id": "session-seed-41",
        "branch_group_id": "branch-group-1",
        "logical_state_id": "logical-kv-1",
        "branch_id": "branch-a",
        "tenant_id": "tenant-a",
        "security_domain": "local-test",
        "host": "host-a",
        "process_id": 314,
        "rank": null,
        "device": "cpu",
        "monotonic_timestamp_ns": 1090,
        "normalized_timestamp_ns": 340,
        "duration_ns": 45,
        "clock_source": "perf_counter",
        "alignment_confidence": 1.0,
        "operation_type": "STATE_COW",
        "state_segment": "kv",
        "source_physical_representation": "continuum.chunk.v1",
        "destination_physical_representation": "continuum.chunk.v1",
        "bytes": 4096,
        "alignment_bytes": 0,
        "page_size_bytes": 0,
        "chunk_size_bytes": 0,
        "fanout": 1,
        "dependency_event_ids": ["event-6"],
        "concurrency": 2,
        "queue_delay_ns": 2,
        "operation_latency_ns": 43,
        "cpu_time_ns": 40,
        "gpu_time_ns": 0,
        "transfer_time_ns": 0,
        "result": "success",
        "failure": null,
        "state_epoch": 2,
        "source_location": "host_dram",
        "destination_location": "host_dram",
        "transport_type": "memory_copy",
        "attributes": {"source_reused": true},
        "event_sequence": 8,
        "content_hash": "0".repeat(64)
    })
}

fn encoded_value<T: serde::Serialize>(event: &T) -> Value {
    serde_json::from_slice(
        &canonical_json(event)
            .unwrap_or_else(|problem| panic!("canonical serialization failed: {problem}")),
    )
    .unwrap_or_else(|problem| panic!("canonical JSON decode failed: {problem}"))
}

#[test]
fn python_shaped_branch_and_state_events_round_trip() {
    let raw_branch = branch_event();
    let branch: BranchWorkloadTraceEventV1 = from_json(
        &serde_json::to_vec(&raw_branch)
            .unwrap_or_else(|problem| panic!("branch event encode failed: {problem}")),
    )
    .unwrap_or_else(|problem| panic!("Python-shaped branch event must validate: {problem}"));
    assert_eq!(encoded_value(&branch), raw_branch);

    let raw_state = state_event();
    let state: StateOperationTraceEventV1 = from_json(
        &serde_json::to_vec(&raw_state)
            .unwrap_or_else(|problem| panic!("state event encode failed: {problem}")),
    )
    .unwrap_or_else(|problem| panic!("Python-shaped state event must validate: {problem}"));
    assert_eq!(encoded_value(&state), raw_state);
}

#[test]
fn independent_provenance_nullable_group_and_zero_na_granularity_are_valid() {
    let branch = branch_event();
    assert!(
        from_json::<BranchWorkloadTraceEventV1>(&serde_json::to_vec(&branch).unwrap_or_default())
            .is_ok()
    );

    let state = state_event();
    assert!(
        from_json::<StateOperationTraceEventV1>(&serde_json::to_vec(&state).unwrap_or_default())
            .is_ok()
    );

    assert!(
        from_json::<BranchFabricTraceEventV1>(&serde_json::to_vec(&state).unwrap_or_default())
            .is_ok()
    );
}

#[test]
fn unknown_fields_and_invalid_operation_vocabulary_are_rejected() {
    let mut branch = branch_event();
    branch["invented_hardware_measurement"] = Value::Bool(true);
    assert!(
        from_json::<BranchWorkloadTraceEventV1>(&serde_json::to_vec(&branch).unwrap_or_default())
            .is_err()
    );

    let mut state = state_event();
    state["operation_type"] = Value::String("STATE_TELEPORT".to_owned());
    assert!(
        from_json::<StateOperationTraceEventV1>(&serde_json::to_vec(&state).unwrap_or_default())
            .is_err()
    );
}

#[test]
fn semantic_invariants_reject_self_parent_zero_concurrency_and_bad_failure_detail() {
    let mut branch = branch_event();
    branch["parent_branch_id"] = Value::String("branch-a".to_owned());
    assert!(
        from_json::<BranchWorkloadTraceEventV1>(&serde_json::to_vec(&branch).unwrap_or_default())
            .is_err()
    );

    let mut state = state_event();
    state["concurrency"] = Value::from(0);
    assert!(
        from_json::<StateOperationTraceEventV1>(&serde_json::to_vec(&state).unwrap_or_default())
            .is_err()
    );

    let mut state = state_event();
    state["failure"] = Value::String("not allowed on success".to_owned());
    assert!(
        from_json::<StateOperationTraceEventV1>(&serde_json::to_vec(&state).unwrap_or_default())
            .is_err()
    );
}

#[test]
fn attribute_count_and_scalar_shape_are_bounded() {
    let mut state = state_event();
    state["attributes"] = Value::Object(
        (0..65)
            .map(|index| (format!("attribute-{index}"), Value::from(index)))
            .collect(),
    );
    assert!(
        from_json::<StateOperationTraceEventV1>(&serde_json::to_vec(&state).unwrap_or_default())
            .is_err()
    );

    let mut state = state_event();
    state["attributes"]["nested"] = json!({"not": "a scalar"});
    assert!(
        from_json::<StateOperationTraceEventV1>(&serde_json::to_vec(&state).unwrap_or_default())
            .is_err()
    );
}
