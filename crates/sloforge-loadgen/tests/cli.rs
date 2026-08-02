#![allow(clippy::expect_used)]

use serde_json::Value;
use std::path::PathBuf;
use std::process::Command;

#[test]
fn generate_then_validate_cli_round_trip() {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let path = std::env::temp_dir().join(format!(
        "sloforge-loadgen-trace-{}.jsonl",
        std::process::id()
    ));
    let generated = Command::new(env!("CARGO_BIN_EXE_sloforge-loadgen"))
        .arg("generate")
        .arg("--config")
        .arg(crate_dir.join("tests/fixtures/workload-config.json"))
        .arg("--output")
        .arg(&path)
        .output()
        .expect("generate");
    assert!(generated.status.success());
    let summary: Value = serde_json::from_slice(&generated.stdout).expect("summary");
    assert_eq!(summary["record_count"], 200);
    let validated = Command::new(env!("CARGO_BIN_EXE_sloforge-loadgen"))
        .arg("validate")
        .arg(&path)
        .output()
        .expect("validate");
    assert!(validated.status.success());
    let validated_summary: Value = serde_json::from_slice(&validated.stdout).expect("summary");
    assert_eq!(summary, validated_summary);
    std::fs::remove_file(path).expect("remove trace");
}
