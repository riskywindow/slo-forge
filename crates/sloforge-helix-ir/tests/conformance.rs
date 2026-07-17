use schemars::schema_for;
use serde_json::Value;
use sloforge_helix_ir::{LearningTransaction, canonical_hash, canonical_json, from_json};

const GOLDEN: &[u8] = include_bytes!("../../../tests/fixtures/helix/learning-transaction-v1.json");
const GOLDEN_HASH: &str = "784a1e9012ac78c8e3bec1a9bf9e7f6f113862fda04d288409d25968b1236a63";

#[test]
fn complete_python_transaction_round_trips_with_golden_hash() {
    let transaction: LearningTransaction = from_json(GOLDEN)
        .unwrap_or_else(|problem| panic!("Python golden transaction must validate: {problem}"));
    assert_eq!(
        canonical_hash(&transaction)
            .unwrap_or_else(|problem| panic!("canonical hash failed: {problem}")),
        GOLDEN_HASH
    );
    let original: Value = serde_json::from_slice(GOLDEN)
        .unwrap_or_else(|problem| panic!("golden JSON decode failed: {problem}"));
    let encoded: Value = serde_json::from_slice(
        &canonical_json(&transaction)
            .unwrap_or_else(|problem| panic!("canonical serialization failed: {problem}")),
    )
    .unwrap_or_else(|problem| panic!("canonical JSON decode failed: {problem}"));
    assert_eq!(encoded, original);
}

#[test]
fn tampering_and_unknown_nested_fields_are_rejected() {
    let mut raw: Value = serde_json::from_slice(GOLDEN)
        .unwrap_or_else(|problem| panic!("golden JSON decode failed: {problem}"));
    raw["branch_group"]["trajectories"][0]["tokens"][0]["behavior_log_probability"] =
        Value::from(-0.3);
    let bytes = serde_json::to_vec(&raw)
        .unwrap_or_else(|problem| panic!("tampered JSON encode failed: {problem}"));
    assert!(from_json::<LearningTransaction>(&bytes).is_err());

    let mut raw: Value = serde_json::from_slice(GOLDEN)
        .unwrap_or_else(|problem| panic!("golden JSON decode failed: {problem}"));
    raw["training_batch"]["samples"][0]["surprise"] = Value::Bool(true);
    let bytes = serde_json::to_vec(&raw)
        .unwrap_or_else(|problem| panic!("unknown-field JSON encode failed: {problem}"));
    assert!(from_json::<LearningTransaction>(&bytes).is_err());
}

#[test]
fn incompatible_state_reuse_and_missing_policy_provenance_are_rejected() {
    let mut raw: Value = serde_json::from_slice(GOLDEN)
        .unwrap_or_else(|problem| panic!("golden JSON decode failed: {problem}"));
    raw["state_reuse_reports"][0]["target_compatibility_fingerprint"]["value"] =
        Value::String("f".repeat(64));
    let bytes = serde_json::to_vec(&raw)
        .unwrap_or_else(|problem| panic!("reuse JSON encode failed: {problem}"));
    assert!(from_json::<LearningTransaction>(&bytes).is_err());

    let mut raw: Value = serde_json::from_slice(GOLDEN)
        .unwrap_or_else(|problem| panic!("golden JSON decode failed: {problem}"));
    raw["training_batch"]["samples"][0]
        .as_object_mut()
        .unwrap_or_else(|| panic!("sample fixture must be an object"))
        .remove("behavior_policy_epoch");
    let bytes = serde_json::to_vec(&raw)
        .unwrap_or_else(|problem| panic!("missing-policy JSON encode failed: {problem}"));
    assert!(from_json::<LearningTransaction>(&bytes).is_err());
}

#[test]
fn schemars_exposes_a_strict_complete_transaction_schema() {
    let schema = serde_json::to_value(schema_for!(LearningTransaction))
        .unwrap_or_else(|problem| panic!("schema serialization failed: {problem}"));
    assert_eq!(schema["type"], "object");
    assert_eq!(schema["additionalProperties"], false);
    assert!(schema["properties"]["training_batch"].is_object());
}

#[test]
fn canonical_float_and_unicode_profile_matches_python() {
    let value = serde_json::json!({
        "negative_zero": -0.0,
        "small": 1e-7,
        "ordinary": 1e-5,
        "unicode": "λ/雪"
    });
    let encoded = canonical_json(&value)
        .unwrap_or_else(|problem| panic!("canonical serialization failed: {problem}"));
    assert_eq!(
        encoded,
        r#"{"negative_zero":-0.0,"ordinary":1e-05,"small":1e-07,"unicode":"λ/雪"}"#.as_bytes()
    );
}
