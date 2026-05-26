use std::{fs, path::PathBuf};

use proptest::prelude::*;
use schemars::schema_for;
use sloforge_fabric_protocol::{
    FabricProfile, ModelGraph, PhysicalExecutionPlan, RecoveryPlan, TopologyGraph, Validate,
    canonical_hash, canonical_json, from_json, migrate_document,
};

fn fixture(name: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/fabric")
        .join(name);
    fs::read(path).unwrap_or_else(|error| panic!("fixture {name} must be readable: {error}"))
}

#[test]
fn topology_golden_round_trip_and_python_hash_match() {
    let document: TopologyGraph = from_json(&fixture("topology-graph-v1.json"))
        .unwrap_or_else(|error| panic!("topology parses: {error}"));
    assert_eq!(
        canonical_hash(&document)
            .unwrap_or_else(|error| panic!("topology hash: {error}"))
            .value,
        "96a833c71bb9c590f5b42a719cfb4aaafa129b9eccde94f29372c783c61c67c9"
    );
    let encoded = canonical_json(&document).unwrap_or_else(|error| panic!("canonical: {error}"));
    let reparsed: TopologyGraph =
        from_json(&encoded).unwrap_or_else(|error| panic!("reparse: {error}"));
    assert_eq!(document, reparsed);
}

#[test]
fn model_graph_golden_round_trip_and_python_hash_match() {
    let document: ModelGraph = from_json(&fixture("model-graph-v1.json"))
        .unwrap_or_else(|error| panic!("model graph parses: {error}"));
    assert_eq!(
        canonical_hash(&document)
            .unwrap_or_else(|error| panic!("model hash: {error}"))
            .value,
        "ae4a4aeeb7ec6b1abed551dd6cb73b7583932f50f11af394f0c2f9a3dcc9abe7"
    );
}

#[test]
fn fabric_profile_golden_round_trip_and_python_hash_match() {
    let document: FabricProfile = from_json(&fixture("fabric-profile-v1.json"))
        .unwrap_or_else(|error| panic!("profile parses: {error}"));
    assert_eq!(
        canonical_hash(&document)
            .unwrap_or_else(|error| panic!("profile hash: {error}"))
            .value,
        "c09263050d9b413d4c41e6d9bcad484058e95bdeac134d6ccab8a2554712ad40"
    );
}

#[test]
fn physical_plan_golden_round_trip_and_python_hash_match() {
    let document: PhysicalExecutionPlan = from_json(&fixture("physical-execution-plan-v1.json"))
        .unwrap_or_else(|error| panic!("physical plan parses: {error}"));
    assert_eq!(document.rank_placement.bindings.len(), 2);
    assert_eq!(document.parallelism.expected_rank_count(), 2);
    assert_eq!(
        canonical_hash(&document)
            .unwrap_or_else(|error| panic!("physical hash: {error}"))
            .value,
        "116275c42541dcbefeeb682344ba34897a4238dd1fcb0feb6427505bb6990f03"
    );
    let encoded = canonical_json(&document).unwrap_or_else(|error| panic!("canonical: {error}"));
    let reparsed: PhysicalExecutionPlan =
        from_json(&encoded).unwrap_or_else(|error| panic!("reparse: {error}"));
    assert_eq!(document, reparsed);
}

#[test]
fn recovery_plan_golden_round_trip_and_python_hash_match() {
    let document: RecoveryPlan = from_json(&fixture("recovery-plan-v1.json"))
        .unwrap_or_else(|error| panic!("recovery plan parses: {error}"));
    assert_eq!(
        canonical_hash(&document)
            .unwrap_or_else(|error| panic!("recovery hash: {error}"))
            .value,
        "ff9dcb36226df98bdd82714a6c312d6ef23aa011648a6718f4a9e98bbe7434f9"
    );
}

#[test]
fn alpha_topology_migrates_and_validates() {
    let source: serde_json::Value =
        serde_json::from_slice(&fixture("topology-graph-v1alpha1.json"))
            .unwrap_or_else(|error| panic!("alpha fixture JSON: {error}"));
    let migrated =
        migrate_document(&source).unwrap_or_else(|error| panic!("alpha migration: {error}"));
    assert_eq!(migrated["schema_version"], "1.0.0");
    assert_eq!(migrated["topology_id"], "alpha-single-host");
    let document: TopologyGraph = serde_json::from_value(migrated)
        .unwrap_or_else(|error| panic!("migrated topology parses: {error}"));
    document
        .validate()
        .unwrap_or_else(|error| panic!("migrated topology validates: {error}"));
}

#[test]
fn unknown_core_fields_fail_closed() {
    let mut source: serde_json::Value =
        serde_json::from_slice(&fixture("physical-execution-plan-v1.json"))
            .unwrap_or_else(|error| panic!("fixture JSON: {error}"));
    source["rank_placement"]["bindings"][0]["surprise"] = serde_json::Value::Bool(true);
    let payload = serde_json::to_vec(&source).unwrap_or_else(|error| panic!("serialize: {error}"));
    assert!(from_json::<PhysicalExecutionPlan>(&payload).is_err());
}

#[test]
fn semantic_memory_and_graph_errors_fail_closed() {
    let mut plan: serde_json::Value =
        serde_json::from_slice(&fixture("physical-execution-plan-v1.json"))
            .unwrap_or_else(|error| panic!("fixture JSON: {error}"));
    plan["memory"]["allocations"]
        .as_array_mut()
        .unwrap_or_else(|| panic!("allocations array"))
        .pop();
    let payload = serde_json::to_vec(&plan).unwrap_or_else(|error| panic!("serialize: {error}"));
    assert!(from_json::<PhysicalExecutionPlan>(&payload).is_err());

    let mut topology: serde_json::Value =
        serde_json::from_slice(&fixture("topology-graph-v1.json"))
            .unwrap_or_else(|error| panic!("fixture JSON: {error}"));
    topology["edges"][0]["target_node_id"] = serde_json::Value::String("missing".to_owned());
    let payload =
        serde_json::to_vec(&topology).unwrap_or_else(|error| panic!("serialize: {error}"));
    assert!(from_json::<TopologyGraph>(&payload).is_err());
}

#[test]
fn rust_schema_exposes_typed_physical_definitions() {
    let schema = serde_json::to_value(schema_for!(PhysicalExecutionPlan))
        .unwrap_or_else(|error| panic!("schema JSON: {error}"));
    let definitions = schema
        .get("$defs")
        .and_then(serde_json::Value::as_object)
        .unwrap_or_else(|| panic!("schema definitions"));
    for required in [
        "ParallelismPlan",
        "RankPlacement",
        "CollectivePlan",
        "KvTransferPlan",
        "MemoryPlan",
        "RecoveryVariant",
    ] {
        assert!(definitions.contains_key(required), "missing {required}");
    }
}

proptest! {
    #[test]
    fn canonical_json_does_not_depend_on_map_insertion_order(
        entries in proptest::collection::btree_map("[a-z]{1,12}", any::<i64>(), 0..24)
    ) {
        let forward: serde_json::Map<String, serde_json::Value> = entries
            .iter()
            .map(|(key, value)| (key.clone(), serde_json::json!(value)))
            .collect();
        let reverse: serde_json::Map<String, serde_json::Value> = entries
            .iter()
            .rev()
            .map(|(key, value)| (key.clone(), serde_json::json!(value)))
            .collect();
        let left = canonical_json(&serde_json::Value::Object(forward))
            .unwrap_or_else(|error| panic!("canonical JSON: {error}"));
        let right = canonical_json(&serde_json::Value::Object(reverse))
            .unwrap_or_else(|error| panic!("canonical JSON: {error}"));
        prop_assert_eq!(left, right);
    }
}
