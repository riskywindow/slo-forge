#![allow(clippy::expect_used)]

mod common;

use common::{compute, request, resource};
use sloforge_fabric_sim::{
    CounterfactualModifier, FaultEffect, OperationKind, PhysicalOperation, ResourceDemand,
    ResourceKind, SchedulingMode, TimedFault, simulate,
};

#[test]
fn exclusive_gpu_serializes_independent_compute() {
    let input = request(
        vec![resource(
            "gpu-0",
            ResourceKind::GpuCompute,
            SchedulingMode::Exclusive,
        )],
        vec![compute("a", "gpu-0", 100.0), compute("b", "gpu-0", 100.0)],
    );
    let output = simulate(&input).expect("valid analytical case");
    assert!((output.metrics.makespan_us - 200.0).abs() < 1e-9);
    let a = output
        .operations
        .iter()
        .find(|item| item.operation_id == "a")
        .expect("a");
    let b = output
        .operations
        .iter()
        .find(|item| item.operation_id == "b")
        .expect("b");
    assert!(a.end_us <= b.start_us || b.end_us <= a.start_us);
}

#[test]
fn disjoint_gpus_overlap_without_exceeding_capacity() {
    let input = request(
        vec![
            resource("gpu-0", ResourceKind::GpuCompute, SchedulingMode::Exclusive),
            resource("gpu-1", ResourceKind::GpuCompute, SchedulingMode::Exclusive),
        ],
        vec![compute("a", "gpu-0", 100.0), compute("b", "gpu-1", 100.0)],
    );
    let output = simulate(&input).expect("valid overlap case");
    assert!((output.metrics.makespan_us - 100.0).abs() < 1e-9);
    assert!((output.metrics.overlap_efficiency - 0.5).abs() < 1e-9);
    assert!(
        output
            .metrics
            .resources
            .iter()
            .all(|item| item.utilization <= 1.0)
    );
}

#[test]
fn fair_share_contention_conserves_link_capacity() {
    let mut link = resource(
        "rail-0",
        ResourceKind::NetworkRail,
        SchedulingMode::FairShare,
    );
    link.max_concurrency = 2;
    let transfer = |id: &str| PhysicalOperation {
        id: id.into(),
        kind: OperationKind::PointToPoint { bytes: 1_000_000 },
        rank_ids: vec![id.into()],
        dependencies: Vec::new(),
        demands: vec![ResourceDemand {
            resource_id: "rail-0".into(),
            units: 1.0,
        }],
        earliest_start_us: 0.0,
        uncertainty_fraction: 0.0,
        request_id: None,
    };
    let output = simulate(&request(vec![link], vec![transfer("a"), transfer("b")]))
        .expect("valid contention case");
    // Each 1 MB flow is 1 ms at 8 Gb/s; two fair-sharing flows complete in 2 ms.
    assert!((output.metrics.makespan_us - 2_000.0).abs() < 1e-6);
    let rail = &output.metrics.resources[0];
    assert_eq!(rail.max_concurrent, 2);
    assert!((rail.busy_time_us - 2_000.0).abs() < 1e-6);
    assert_eq!(rail.transferred_bytes, 2_000_000);
}

#[test]
fn collective_waits_for_every_rank_dependency() {
    let mut rank0 = compute("rank0-prefill", "gpu-0", 50.0);
    rank0.rank_ids = vec!["rank-0".into()];
    let mut rank1 = compute("rank1-prefill", "gpu-1", 100.0);
    rank1.rank_ids = vec!["rank-1".into()];
    let collective = PhysicalOperation {
        id: "all-reduce".into(),
        kind: OperationKind::Collective {
            collective_id: "decode-ar-0".into(),
            bytes: 1_000,
            algorithm: "direct".into(),
            participating_ranks: vec!["rank-0".into(), "rank-1".into()],
        },
        rank_ids: vec!["rank-0".into(), "rank-1".into()],
        dependencies: vec!["rank0-prefill".into(), "rank1-prefill".into()],
        demands: vec![ResourceDemand {
            resource_id: "nvlink".into(),
            units: 1.0,
        }],
        earliest_start_us: 0.0,
        uncertainty_fraction: 0.0,
        request_id: None,
    };
    let output = simulate(&request(
        vec![
            resource("gpu-0", ResourceKind::GpuCompute, SchedulingMode::Exclusive),
            resource("gpu-1", ResourceKind::GpuCompute, SchedulingMode::Exclusive),
            resource("nvlink", ResourceKind::Nvlink, SchedulingMode::Exclusive),
        ],
        vec![rank0, rank1, collective],
    ))
    .expect("valid collective case");
    let collective = output
        .operations
        .iter()
        .find(|item| item.operation_id == "all-reduce")
        .expect("collective");
    assert!((collective.start_us - 100.0).abs() < 1e-9);
}

#[test]
fn timed_link_and_rank_faults_slow_then_recover() {
    let mut input = request(
        vec![resource(
            "gpu-0",
            ResourceKind::GpuCompute,
            SchedulingMode::Exclusive,
        )],
        vec![compute("decode", "gpu-0", 100.0)],
    );
    input.operations[0].rank_ids = vec!["rank-6".into()];
    input.faults = vec![
        TimedFault {
            id: "clock".into(),
            start_us: 0.0,
            end_us: Some(50.0),
            effect: FaultEffect::ResourceRate {
                resource_id: "gpu-0".into(),
                multiplier: 0.5,
            },
            ground_truth_label: "gpu_clock_throttle".into(),
        },
        TimedFault {
            id: "rank".into(),
            start_us: 50.0,
            end_us: Some(100.0),
            effect: FaultEffect::RankSlowdown {
                rank_id: "rank-6".into(),
                multiplier: 0.5,
            },
            ground_truth_label: "rank_specific_slowdown".into(),
        },
    ];
    let output = simulate(&input).expect("recovering fault");
    assert!((output.metrics.makespan_us - 150.0).abs() < 1e-9);
    assert_eq!(output.applied_faults, vec!["clock", "rank"]);

    input.counterfactuals = vec![
        CounterfactualModifier::RemoveFault {
            fault_id: "clock".into(),
        },
        CounterfactualModifier::RemoveFault {
            fault_id: "rank".into(),
        },
    ];
    let repaired = simulate(&input).expect("counterfactual repair");
    assert!((repaired.metrics.makespan_us - 100.0).abs() < 1e-9);
    assert!(repaired.applied_faults.is_empty());
}

#[test]
fn temporary_rail_loss_stalls_without_fallback() {
    let mut input = request(
        vec![resource(
            "rail",
            ResourceKind::NetworkRail,
            SchedulingMode::Exclusive,
        )],
        vec![PhysicalOperation {
            id: "kv".into(),
            kind: OperationKind::KvTransfer {
                bytes: 50_000,
                chunks: 1,
            },
            rank_ids: vec!["prefill-0".into(), "decode-0".into()],
            dependencies: Vec::new(),
            demands: vec![ResourceDemand {
                resource_id: "rail".into(),
                units: 1.0,
            }],
            earliest_start_us: 0.0,
            uncertainty_fraction: 0.0,
            request_id: None,
        }],
    );
    input.faults.push(TimedFault {
        id: "rail-loss".into(),
        start_us: 0.0,
        end_us: Some(100.0),
        effect: FaultEffect::ResourceUnavailable {
            resource_id: "rail".into(),
        },
        ground_truth_label: "network_rail_loss".into(),
    });
    let output = simulate(&input).expect("rail recovers");
    assert!((output.metrics.makespan_us - 150.0).abs() < 1e-6);
    assert_eq!(output.operations[0].resource_ids, vec!["rail"]);
}
