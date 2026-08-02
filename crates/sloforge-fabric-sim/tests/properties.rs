#![allow(clippy::expect_used)]

mod common;

use common::{compute, request, resource};
use proptest::prelude::*;
use sloforge_fabric_sim::{ResourceKind, SchedulingMode, simulate};

proptest! {
    #[test]
    fn exclusive_analytical_sum_and_deterministic_serialization(
        durations in prop::collection::vec(1_u16..10_000_u16, 1..64)
    ) {
        let operations = durations.iter().enumerate().map(|(index, duration)| {
            compute(&format!("op-{index:03}"), "gpu", f64::from(*duration))
        }).collect();
        let input = request(
            vec![resource("gpu", ResourceKind::GpuCompute, SchedulingMode::Exclusive)],
            operations,
        );
        let first = simulate(&input).expect("generated valid input");
        let second = simulate(&input).expect("same input remains valid");
        let expected: f64 = durations.iter().map(|duration| f64::from(*duration)).sum();
        prop_assert!((first.metrics.makespan_us - expected).abs() < 1e-8);
        prop_assert_eq!(
            serde_json::to_vec(&first).expect("serializable"),
            serde_json::to_vec(&second).expect("serializable")
        );
        let valid_durations = first.operations.iter().all(|outcome| {
            outcome.start_us >= 0.0
                && outcome.end_us >= outcome.start_us
                && outcome.duration_us >= 0.0
        });
        prop_assert!(valid_durations);
    }
}

#[test]
fn simulated_centuries_execute_faster_than_wall_clock() {
    let input = request(
        vec![resource(
            "gpu",
            ResourceKind::GpuCompute,
            SchedulingMode::Exclusive,
        )],
        vec![compute("long", "gpu", 1.0e15)],
    );
    let started = std::time::Instant::now();
    let output = simulate(&input).expect("large virtual duration");
    assert!((output.metrics.makespan_us - 1.0e15).abs() < 1.0);
    assert!(started.elapsed() < std::time::Duration::from_secs(1));
}
