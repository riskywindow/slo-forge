#![allow(clippy::expect_used)]

mod common;

use common::{compute, request, resource};
use sloforge_fabric_sim::{FabricSimulationOutput, ResourceKind, SchedulingMode};
use std::io::Write;
use std::process::{Command, Stdio};

#[test]
fn bounded_json_subprocess_round_trip_and_chrome_trace() {
    let request = request(
        vec![resource(
            "gpu",
            ResourceKind::GpuCompute,
            SchedulingMode::Exclusive,
        )],
        vec![compute("decode", "gpu", 25.0)],
    );
    let directory =
        std::env::temp_dir().join(format!("sloforge-fabric-sim-cli-{}", std::process::id()));
    std::fs::create_dir_all(&directory).expect("create temp directory");
    let trace = directory.join("trace.json");
    let mut child = Command::new(env!("CARGO_BIN_EXE_sloforge-fabric-sim"))
        .args(["simulate", "--compact", "--chrome-trace"])
        .arg(&trace)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn simulator");
    child
        .stdin
        .take()
        .expect("piped stdin")
        .write_all(&serde_json::to_vec(&request).expect("serialize request"))
        .expect("write request");
    let result = child.wait_with_output().expect("wait for simulator");
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let output: FabricSimulationOutput =
        serde_json::from_slice(&result.stdout).expect("parse response");
    assert_eq!(output.metrics.operation_count, 1);
    let chrome: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&trace).expect("read trace")).expect("parse trace");
    assert_eq!(chrome["traceEvents"].as_array().expect("events").len(), 1);
    std::fs::remove_dir_all(directory).expect("remove temp directory");
}
