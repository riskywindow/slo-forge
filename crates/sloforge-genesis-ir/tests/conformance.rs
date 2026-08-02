use std::{collections::BTreeMap, fs, path::PathBuf};

use proptest::prelude::*;
use serde_json::{Value, json};
use sloforge_genesis_ir::{
    Candidate, Counterexample, InferenceGenome, Transformation, canonical_hash, canonical_json,
    from_json, migrate_document,
};

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/genesis")
}

fn fixture(name: &str) -> Vec<u8> {
    fs::read(fixtures().join(name)).unwrap_or_else(|error| panic!("cannot read {name}: {error}"))
}

#[test]
fn canonical_edge_number_profile_matches_python() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/ir/canonical-edge-cases.json");
    let value: Value = serde_json::from_slice(
        &fs::read(path).unwrap_or_else(|error| panic!("cannot read edge fixture: {error}")),
    )
    .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(
        canonical_hash(&value).unwrap_or_else(|error| panic!("{error}")),
        "82e4b7f7fc6f923704946afe34437a6f5ed6f927addfa46bad6858b551e0b01a"
    );
}

#[test]
fn all_golden_documents_validate_and_round_trip() {
    let genome: InferenceGenome =
        from_json(&fixture("inference-genome-v1.json")).unwrap_or_else(|error| panic!("{error}"));
    let transformation: Transformation =
        from_json(&fixture("transformation-v1.json")).unwrap_or_else(|error| panic!("{error}"));
    let candidate: Candidate =
        from_json(&fixture("candidate-v1.json")).unwrap_or_else(|error| panic!("{error}"));
    let counterexample: Counterexample =
        from_json(&fixture("counterexample-v1.json")).unwrap_or_else(|error| panic!("{error}"));

    for (actual, expected) in [
        (
            canonical_json(&genome).unwrap_or_else(|error| panic!("{error}")),
            fixture("inference-genome-v1.json"),
        ),
        (
            canonical_json(&transformation).unwrap_or_else(|error| panic!("{error}")),
            fixture("transformation-v1.json"),
        ),
        (
            canonical_json(&candidate).unwrap_or_else(|error| panic!("{error}")),
            fixture("candidate-v1.json"),
        ),
        (
            canonical_json(&counterexample).unwrap_or_else(|error| panic!("{error}")),
            fixture("counterexample-v1.json"),
        ),
    ] {
        assert_eq!(actual, expected.strip_suffix(b"\n").unwrap_or(&expected));
    }
}

#[test]
fn canonical_hashes_match_python_goldens() {
    let hashes: BTreeMap<String, String> =
        serde_json::from_slice(&fixture("canonical-hashes-v1.json"))
            .unwrap_or_else(|error| panic!("{error}"));
    let genome: InferenceGenome =
        from_json(&fixture("inference-genome-v1.json")).unwrap_or_else(|error| panic!("{error}"));
    let transformation: Transformation =
        from_json(&fixture("transformation-v1.json")).unwrap_or_else(|error| panic!("{error}"));
    let candidate: Candidate =
        from_json(&fixture("candidate-v1.json")).unwrap_or_else(|error| panic!("{error}"));
    let counterexample: Counterexample =
        from_json(&fixture("counterexample-v1.json")).unwrap_or_else(|error| panic!("{error}"));

    for (filename, actual) in [
        (
            "inference-genome-v1.json",
            canonical_hash(&genome).unwrap_or_else(|error| panic!("{error}")),
        ),
        (
            "transformation-v1.json",
            canonical_hash(&transformation).unwrap_or_else(|error| panic!("{error}")),
        ),
        (
            "candidate-v1.json",
            canonical_hash(&candidate).unwrap_or_else(|error| panic!("{error}")),
        ),
        (
            "counterexample-v1.json",
            canonical_hash(&counterexample).unwrap_or_else(|error| panic!("{error}")),
        ),
    ] {
        assert_eq!(hashes.get(filename), Some(&actual));
    }
}

#[test]
fn unknown_fields_and_unnamespaced_extensions_are_rejected() {
    let mut genome: Value = serde_json::from_slice(&fixture("inference-genome-v1.json"))
        .unwrap_or_else(|error| panic!("{error}"));
    genome["untrusted_guess"] = Value::Bool(true);
    assert!(serde_json::from_value::<InferenceGenome>(genome).is_err());

    let mut genome: Value = serde_json::from_slice(&fixture("inference-genome-v1.json"))
        .unwrap_or_else(|error| panic!("{error}"));
    genome["extensions"] = json!({"not-qualified": true});
    assert!(serde_json::from_value::<InferenceGenome>(genome).is_err());
}

#[test]
fn alpha_migration_renames_all_genome_regions_without_mutation() {
    let mut alpha: Value = serde_json::from_slice(&fixture("inference-genome-v1.json"))
        .unwrap_or_else(|error| panic!("{error}"));
    alpha["schema_version"] = Value::String("0.1.0".into());
    alpha
        .as_object_mut()
        .unwrap_or_else(|| panic!("fixture is an object"))
        .remove("api_version");
    alpha["kind"] = Value::String("inference_genome".into());
    for region in [
        "workflow",
        "request",
        "serving",
        "state",
        "distributed",
        "tensor",
        "kernel",
        "recovery",
    ] {
        let value = alpha
            .as_object_mut()
            .unwrap_or_else(|| panic!("fixture is an object"))
            .remove(region)
            .unwrap_or_else(|| panic!("missing {region}"));
        alpha
            .as_object_mut()
            .unwrap_or_else(|| panic!("fixture is an object"))
            .insert(format!("{region}_genome"), value);
    }
    let original = alpha.clone();
    let migrated = migrate_document(&alpha).unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(alpha, original);
    let payload = serde_json::to_vec(&migrated).unwrap_or_else(|error| panic!("{error}"));
    let _: InferenceGenome = from_json(&payload).unwrap_or_else(|error| panic!("{error}"));
}

proptest! {
    #[test]
    fn any_u64_seed_round_trips_canonically(seed in any::<u64>()) {
        let mut value: Value = serde_json::from_slice(&fixture("inference-genome-v1.json"))
            .unwrap_or_else(|error| panic!("{error}"));
        value["seed"] = Value::from(seed);
        let payload = serde_json::to_vec(&value).unwrap_or_else(|error| panic!("{error}"));
        let genome: InferenceGenome = from_json(&payload).unwrap_or_else(|error| panic!("{error}"));
        let reparsed: InferenceGenome = from_json(
            &canonical_json(&genome).unwrap_or_else(|error| panic!("{error}"))
        ).unwrap_or_else(|error| panic!("{error}"));
        prop_assert_eq!(genome.seed, seed);
        prop_assert_eq!(
            canonical_hash(&genome).unwrap_or_else(|error| panic!("{error}")),
            canonical_hash(&reparsed).unwrap_or_else(|error| panic!("{error}")),
        );
    }
}
