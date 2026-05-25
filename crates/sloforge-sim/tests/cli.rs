#![allow(clippy::expect_used)]

use serde_json::Value;
use std::path::PathBuf;
use std::process::Command;

#[test]
fn compatibility_cli_writes_result_and_chrome_trace() {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let nonce = std::process::id();
    let result_path = std::env::temp_dir().join(format!("sloforge-sim-result-{nonce}.json"));
    let trace_path = std::env::temp_dir().join(format!("sloforge-sim-trace-{nonce}.json"));
    let status = Command::new(env!("CARGO_BIN_EXE_sloforge-sim"))
        .arg("simulate")
        .arg("--config")
        .arg(crate_dir.join("tests/fixtures/scenario.json"))
        .arg("--trace")
        .arg(crate_dir.join("tests/fixtures/workload.jsonl"))
        .arg("--output")
        .arg(&result_path)
        .arg("--chrome-trace")
        .arg(&trace_path)
        .status()
        .expect("run simulator");
    assert!(status.success());
    let output: Value =
        serde_json::from_slice(&std::fs::read(&result_path).expect("result")).expect("JSON");
    assert_eq!(output["schema_version"], "1.0");
    assert_eq!(output["metrics"]["request_count"], 2);
    assert_eq!(output["provenance"]["seed"], 42);
    let chrome: Value =
        serde_json::from_slice(&std::fs::read(&trace_path).expect("trace")).expect("JSON");
    assert!(
        chrome["traceEvents"]
            .as_array()
            .is_some_and(|events| !events.is_empty())
    );
    std::fs::remove_file(result_path).expect("remove result");
    std::fs::remove_file(trace_path).expect("remove trace");
}
