use std::collections::BTreeMap;

use proptest::prelude::*;
use serde_json::Value;
use sloforge_continuum_ir::{
    ExecutionStateCapsule, Validate, canonical_hash, canonical_json, from_json, migrate_document,
};

const GOLDEN: &[u8] =
    include_bytes!("../../../schemas/continuum/golden-execution-state-capsule-v1.json");
const GOLDEN_HASH: &str = "18fbe6c114ec019839072c03213c1bf35efc72d650a668dee2addbb89a7215ee";

#[test]
fn python_golden_capsule_round_trips_with_identical_hash() {
    let capsule: ExecutionStateCapsule = from_json(GOLDEN).unwrap_or_else(|error| {
        panic!("shared Python golden capsule must validate in Rust: {error}")
    });
    assert_eq!(
        canonical_hash(&capsule).unwrap_or_else(|error| panic!("hash failed: {error}")),
        GOLDEN_HASH
    );
    let original: Value = serde_json::from_slice(GOLDEN)
        .unwrap_or_else(|error| panic!("golden JSON decode failed: {error}"));
    let encoded: Value = serde_json::from_slice(
        &canonical_json(&capsule)
            .unwrap_or_else(|error| panic!("canonical serialization failed: {error}")),
    )
    .unwrap_or_else(|error| panic!("canonical JSON decode failed: {error}"));
    assert_eq!(encoded, original);
}

#[test]
fn capsule_tampering_and_unknown_fields_are_rejected() {
    let mut raw: Value = serde_json::from_slice(GOLDEN)
        .unwrap_or_else(|error| panic!("golden JSON decode failed: {error}"));
    raw["segment_manifests"][0]["segment_hash"]["value"] = Value::String("f".repeat(64));
    let payload = serde_json::to_vec(&raw)
        .unwrap_or_else(|error| panic!("tampered fixture encode failed: {error}"));
    assert!(from_json::<ExecutionStateCapsule>(&payload).is_err());

    let mut raw: Value = serde_json::from_slice(GOLDEN)
        .unwrap_or_else(|error| panic!("golden JSON decode failed: {error}"));
    raw["logical_state"]["surprise"] = Value::Bool(true);
    let payload = serde_json::to_vec(&raw)
        .unwrap_or_else(|error| panic!("unknown-field fixture encode failed: {error}"));
    assert!(from_json::<ExecutionStateCapsule>(&payload).is_err());

    let mut raw: Value = serde_json::from_slice(GOLDEN)
        .unwrap_or_else(|error| panic!("golden JSON decode failed: {error}"));
    raw["identity"]["capture_timestamp"] = Value::String("2099-01-01T00:00:00Z".into());
    let payload = serde_json::to_vec(&raw)
        .unwrap_or_else(|error| panic!("identity-tampered fixture encode failed: {error}"));
    assert!(from_json::<ExecutionStateCapsule>(&payload).is_err());

    let mut raw: Value = serde_json::from_slice(GOLDEN)
        .unwrap_or_else(|error| panic!("golden JSON decode failed: {error}"));
    raw["extensions"]["sloforge.io/fixture-seed"] = Value::Number(1_u64.into());
    let payload = serde_json::to_vec(&raw)
        .unwrap_or_else(|error| panic!("extension-tampered fixture encode failed: {error}"));
    assert!(from_json::<ExecutionStateCapsule>(&payload).is_err());
}

#[test]
fn stale_page_and_owner_epoch_are_rejected() {
    let mut raw: Value = serde_json::from_slice(GOLDEN)
        .unwrap_or_else(|error| panic!("golden JSON decode failed: {error}"));
    raw["physical_state"]["page_tables"][0]["entries"][0]["page_version"] =
        Value::Number(2_u64.into());
    let capsule: ExecutionStateCapsule = serde_json::from_value(raw)
        .unwrap_or_else(|error| panic!("typed tampered capsule decode failed: {error}"));
    assert!(capsule.validate().is_err());

    let mut raw: Value = serde_json::from_slice(GOLDEN)
        .unwrap_or_else(|error| panic!("golden JSON decode failed: {error}"));
    raw["identity"]["owner_epoch"] = Value::Number(6_u64.into());
    let capsule: ExecutionStateCapsule = serde_json::from_value(raw)
        .unwrap_or_else(|error| panic!("typed tampered capsule decode failed: {error}"));
    assert!(capsule.validate().is_err());

    let mut raw: Value = serde_json::from_slice(GOLDEN)
        .unwrap_or_else(|error| panic!("golden JSON decode failed: {error}"));
    raw["physical_state"]["shard_descriptors"][2]["rank"] = Value::Number(1_u64.into());
    let capsule: ExecutionStateCapsule = serde_json::from_value(raw)
        .unwrap_or_else(|error| panic!("typed invalid-rank capsule decode failed: {error}"));
    assert!(capsule.validate().is_err());
}

#[test]
fn canonical_float_and_unicode_profile_matches_python() {
    let value = serde_json::json!({
        "negative_zero": -0.0,
        "small": 1e-7,
        "threshold": 1e-6,
        "ordinary": 1e-5,
        "large": 1e20,
        "larger": 1e21,
        "unicode": "λ/雪",
        "control": "a\nb"
    });
    let encoded = canonical_json(&value)
        .unwrap_or_else(|error| panic!("canonical serialization failed: {error}"));
    assert_eq!(
        encoded,
        r#"{"control":"a\nb","large":1e+20,"larger":1e+21,"negative_zero":-0.0,"ordinary":1e-05,"small":1e-07,"threshold":1e-06,"unicode":"λ/雪"}"#.as_bytes()
    );
}

#[test]
fn alpha_migration_preserves_payload_and_rejects_unknown_version() {
    let alpha = serde_json::json!({
        "version": "v1alpha1",
        "kind": "state_transaction",
        "phase": "proposed",
        "transaction_id": "tx-1"
    });
    let migrated = migrate_document(&alpha)
        .unwrap_or_else(|error| panic!("known alpha migration failed: {error}"));
    assert_eq!(migrated["schema_version"], "1.0.0");
    assert_eq!(migrated["kind"], "StateTransaction");
    assert_eq!(migrated["current_phase"], "proposed");
    assert_eq!(alpha["phase"], "proposed");
    assert!(migrate_document(&serde_json::json!({"version": "v9", "kind": "x"})).is_err());
}

proptest! {
    #[test]
    fn canonical_hash_is_stable_for_sorted_maps(values in prop::collection::btree_map("[a-z]{1,8}", any::<i64>(), 0..30)) {
        let reordered: BTreeMap<_, _> = values.iter().rev().map(|(key, value)| (key.clone(), *value)).collect();
        prop_assert_eq!(
            canonical_hash(&values).unwrap_or_else(|error| panic!("canonical hash failed: {error}")),
            canonical_hash(&reordered).unwrap_or_else(|error| panic!("canonical hash failed: {error}")),
        );
    }
}
