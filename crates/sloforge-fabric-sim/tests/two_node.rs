#![allow(clippy::expect_used, clippy::too_many_lines)]

mod common;

use common::{compute, request, resource};
use sloforge_fabric_sim::{
    FaultEffect, OperationKind, PhysicalOperation, ResourceDemand, ResourceKind, SchedulingMode,
    SharingGroup, TimedFault, simulate,
};

fn communication(
    id: &str,
    kind: OperationKind,
    ranks: &[&str],
    dependencies: &[&str],
    resources: &[&str],
) -> PhysicalOperation {
    PhysicalOperation {
        id: id.into(),
        kind,
        rank_ids: ranks.iter().map(ToString::to_string).collect(),
        dependencies: dependencies.iter().map(ToString::to_string).collect(),
        demands: resources
            .iter()
            .map(|resource_id| ResourceDemand {
                resource_id: (*resource_id).into(),
                units: 1.0,
            })
            .collect(),
        earliest_start_us: 0.0,
        uncertainty_fraction: 0.02,
        request_id: Some("mixed-0".into()),
    }
}

#[test]
fn two_node_disaggregated_moe_graph_models_shared_rail_and_recovery() {
    let mut gpu0 = resource(
        "host0-gpu0",
        ResourceKind::GpuCompute,
        SchedulingMode::Exclusive,
    );
    let mut gpu1 = resource(
        "host1-gpu0",
        ResourceKind::GpuCompute,
        SchedulingMode::Exclusive,
    );
    let mut nic0 = resource(
        "host0-nic0",
        ResourceKind::NicQueue,
        SchedulingMode::FairShare,
    );
    let mut nic1 = resource(
        "host1-nic0",
        ResourceKind::NicQueue,
        SchedulingMode::FairShare,
    );
    let mut rail = resource(
        "rail-a",
        ResourceKind::NetworkRail,
        SchedulingMode::FairShare,
    );
    nic0.sharing_group = Some("fabric-a".into());
    nic1.sharing_group = Some("fabric-a".into());
    rail.sharing_group = Some("fabric-a".into());
    nic0.max_concurrency = 8;
    nic1.max_concurrency = 8;
    rail.max_concurrency = 8;
    gpu0.hourly_cost_usd = 2.0;
    gpu1.hourly_cost_usd = 2.0;

    let mut prefill = compute("prefill", "host0-gpu0", 200.0);
    prefill.rank_ids = vec!["prefill-rank-0".into()];
    let kv = communication(
        "kv-transfer",
        OperationKind::KvTransfer {
            bytes: 1_000_000,
            chunks: 4,
        },
        &["prefill-rank-0", "decode-rank-1"],
        &["prefill"],
        &["host0-nic0", "rail-a", "host1-nic0"],
    );
    let dispatch = communication(
        "expert-dispatch",
        OperationKind::ExpertDispatch {
            bytes: 500_000,
            experts: 8,
        },
        &["decode-rank-1", "expert-rank-0"],
        &["kv-transfer"],
        &["host1-nic0", "rail-a", "host0-nic0"],
    );
    let mut expert = compute("expert-compute", "host0-gpu0", 100.0);
    expert.rank_ids = vec!["expert-rank-0".into()];
    expert.dependencies = vec!["expert-dispatch".into()];
    let combine = communication(
        "expert-combine",
        OperationKind::ExpertCombine {
            bytes: 500_000,
            experts: 8,
        },
        &["expert-rank-0", "decode-rank-1"],
        &["expert-compute"],
        &["host0-nic0", "rail-a", "host1-nic0"],
    );
    let mut decode = compute("decode", "host1-gpu0", 150.0);
    decode.rank_ids = vec!["decode-rank-1".into()];
    decode.dependencies = vec!["expert-combine".into()];

    let mut input = request(
        vec![gpu0, gpu1, nic0, nic1, rail],
        vec![prefill, kv, dispatch, expert, combine, decode],
    );
    input.sharing_groups.push(SharingGroup {
        id: "fabric-a".into(),
        capacity_units: 3.0,
        max_concurrency: 16,
    });
    let healthy = simulate(&input).expect("healthy two-node plan");
    assert_eq!(healthy.metrics.operation_count, 6);
    assert_eq!(healthy.metrics.total_transferred_bytes, 2_000_000);
    assert!(healthy.metrics.cost_usd > 0.0);

    input.faults = vec![
        TimedFault {
            id: "rail-degradation".into(),
            start_us: 200.0,
            end_us: Some(3_000.0),
            effect: FaultEffect::ResourceRate {
                resource_id: "rail-a".into(),
                multiplier: 0.25,
            },
            ground_truth_label: "network_bandwidth_degradation".into(),
        },
        TimedFault {
            id: "rank-skew".into(),
            start_us: 0.0,
            end_us: Some(10_000.0),
            effect: FaultEffect::RankSlowdown {
                rank_id: "expert-rank-0".into(),
                multiplier: 0.5,
            },
            ground_truth_label: "rank_specific_slowdown".into(),
        },
    ];
    let degraded = simulate(&input).expect("faults recover without fallback");
    assert!(degraded.metrics.makespan_us > healthy.metrics.makespan_us);
    assert_eq!(
        degraded.applied_faults,
        vec!["rail-degradation", "rank-skew"]
    );
    assert!(
        degraded
            .trace_events
            .iter()
            .any(|event| event.cat == "kv_transfer")
    );
    assert!(
        degraded
            .trace_events
            .iter()
            .any(|event| event.cat == "fabric")
    );
}
