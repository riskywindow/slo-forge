use std::{fs, path::PathBuf};

use proptest::prelude::*;
use schemars::schema_for;
use sloforge_protocol::{
    DeploymentPlan, EvidenceBundle, SCHEMA_VERSION, Validate, canonical_hash, canonical_json,
    from_json, migrate_document,
};

fn fixture(name: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/ir")
        .join(name);
    fs::read(path).unwrap_or_else(|error| panic!("fixture {name} must be readable: {error}"))
}

#[test]
fn deployment_plan_golden_round_trip() {
    let plan: DeploymentPlan = from_json(&fixture("deployment-plan-v1.json"))
        .unwrap_or_else(|error| panic!("valid deployment plan: {error}"));
    assert_eq!(plan.schema_version, SCHEMA_VERSION);
    let canonical = canonical_json(&plan).unwrap_or_else(|error| panic!("canonical JSON: {error}"));
    let reparsed: DeploymentPlan =
        from_json(&canonical).unwrap_or_else(|error| panic!("canonical plan parses: {error}"));
    assert_eq!(plan, reparsed);
    assert_eq!(
        canonical_hash(&plan).unwrap_or_else(|error| panic!("plan hash: {error}")),
        canonical_hash(&reparsed).unwrap_or_else(|error| panic!("reparsed hash: {error}"))
    );
    assert_eq!(
        canonical_hash(&plan)
            .unwrap_or_else(|error| panic!("plan hash: {error}"))
            .value,
        "d578ce29d3a2c5f026c581575fa43bfb3763914197d09eea03edd6d14ee42151"
    );
}

#[test]
fn evidence_bundle_golden_round_trip() {
    let evidence: EvidenceBundle = from_json(&fixture("evidence-bundle-v1.json"))
        .unwrap_or_else(|error| panic!("valid evidence bundle: {error}"));
    evidence
        .validate()
        .unwrap_or_else(|error| panic!("evidence validates: {error}"));
    assert_eq!(evidence.measurements.len(), 1);
    assert_eq!(
        canonical_hash(&evidence)
            .unwrap_or_else(|error| panic!("evidence hash: {error}"))
            .value,
        "06407e562984e50d04a4b289c0520462c0bb80c54e34e9ef4e7fb2d2831471dc"
    );
}

#[test]
fn v1alpha1_migration_renames_wire_fields() {
    let source: serde_json::Value =
        serde_json::from_slice(&fixture("deployment-plan-v1alpha1-fragment.json"))
            .unwrap_or_else(|error| panic!("legacy fixture JSON: {error}"));
    let migrated = migrate_document(&source).unwrap_or_else(|error| panic!("migration: {error}"));
    assert_eq!(migrated["schema_version"], SCHEMA_VERSION);
    assert_eq!(migrated["model"]["model_id"], "Qwen/Qwen3-0.6B");
    assert_eq!(migrated["engine"]["runtime"], "mock");
    assert!(migrated.get("replica_topology").is_some());
}

#[test]
fn full_v1alpha1_golden_migrates_to_stable_plan() {
    let source: serde_json::Value =
        serde_json::from_slice(&fixture("deployment-plan-v1alpha1.json"))
            .unwrap_or_else(|error| panic!("legacy fixture JSON: {error}"));
    let migrated = migrate_document(&source).unwrap_or_else(|error| panic!("migration: {error}"));
    let stable: serde_json::Value = serde_json::from_slice(&fixture("deployment-plan-v1.json"))
        .unwrap_or_else(|error| panic!("stable fixture JSON: {error}"));
    assert_eq!(migrated, stable);
    let plan: DeploymentPlan = serde_json::from_value(migrated)
        .unwrap_or_else(|error| panic!("migrated plan parsing: {error}"));
    plan.validate()
        .unwrap_or_else(|error| panic!("migrated plan validation: {error}"));
}

#[test]
fn unknown_core_fields_are_rejected() {
    let mut value: serde_json::Value = serde_json::from_slice(&fixture("deployment-plan-v1.json"))
        .unwrap_or_else(|error| panic!("fixture JSON: {error}"));
    value["engine"]["surprise"] = serde_json::Value::Bool(true);
    let payload = serde_json::to_vec(&value).unwrap_or_else(|error| panic!("serialize: {error}"));
    let result = from_json::<DeploymentPlan>(&payload);
    assert!(result.is_err());
}

#[test]
fn generated_rust_schema_contains_typed_core_definitions() {
    let schema = serde_json::to_value(schema_for!(DeploymentPlan))
        .unwrap_or_else(|error| panic!("schema serialization: {error}"));
    let definitions = schema
        .get("$defs")
        .and_then(serde_json::Value::as_object)
        .unwrap_or_else(|| panic!("schema must contain definitions"));
    assert!(definitions.contains_key("ModelSpec"));
    assert!(definitions.contains_key("EngineSpec"));
    assert!(definitions.contains_key("HardwareSpec"));
    assert!(definitions.contains_key("WorkloadSpec"));
    assert!(definitions.contains_key("SLOSpec"));
}

#[test]
fn edge_number_canonical_hash_matches_python() {
    let document: serde_json::Value = serde_json::from_slice(&fixture("canonical-edge-cases.json"))
        .unwrap_or_else(|error| panic!("edge fixture JSON: {error}"));
    assert_eq!(
        canonical_hash(&document)
            .unwrap_or_else(|error| panic!("edge hash: {error}"))
            .value,
        "82e4b7f7fc6f923704946afe34437a6f5ed6f927addfa46bad6858b551e0b01a"
    );
}

#[test]
fn invalid_semver_and_extensions_fail_closed() {
    let mut value: serde_json::Value = serde_json::from_slice(&fixture("deployment-plan-v1.json"))
        .unwrap_or_else(|error| panic!("fixture JSON: {error}"));
    value["engine"]["version"] = serde_json::Value::String("latest".to_owned());
    let payload = serde_json::to_vec(&value).unwrap_or_else(|error| panic!("serialize: {error}"));
    assert!(from_json::<DeploymentPlan>(&payload).is_err());

    value["engine"]["version"] = serde_json::Value::String("1.0.0".to_owned());
    value["extensions"] = serde_json::json!({"unqualified": true});
    let payload = serde_json::to_vec(&value).unwrap_or_else(|error| panic!("serialize: {error}"));
    assert!(from_json::<DeploymentPlan>(&payload).is_err());
}

proptest! {
    #[test]
    fn canonical_json_is_independent_of_object_insertion_order(
        entries in proptest::collection::btree_map("[a-z]{1,12}", any::<i64>(), 0..32)
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
