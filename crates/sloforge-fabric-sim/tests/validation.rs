#![allow(clippy::expect_used)]

mod common;

use common::{compute, request, resource};
use sloforge_fabric_sim::{
    OperationKind, PhysicalOperation, ResourceDemand, ResourceKind, SchedulingMode, SimError,
    simulate, validate,
};

#[test]
fn rejects_cycles_unknown_resources_and_negative_values() {
    let mut cyclic = request(
        vec![resource(
            "gpu",
            ResourceKind::GpuCompute,
            SchedulingMode::Exclusive,
        )],
        vec![compute("a", "gpu", 10.0), compute("b", "gpu", 10.0)],
    );
    cyclic.operations[0].dependencies = vec!["b".into()];
    cyclic.operations[1].dependencies = vec!["a".into()];
    assert!(
        matches!(validate(&cyclic), Err(SimError::InvalidInput(message)) if message.contains("cycle"))
    );

    let mut unknown = request(
        vec![resource(
            "gpu",
            ResourceKind::GpuCompute,
            SchedulingMode::Exclusive,
        )],
        vec![compute("a", "missing", 10.0)],
    );
    assert!(validate(&unknown).is_err());
    unknown.operations[0].demands[0].resource_id = "gpu".into();
    unknown.operations[0].earliest_start_us = -1.0;
    assert!(validate(&unknown).is_err());
}

#[test]
fn permanent_unavailability_is_an_explicit_deadlock() {
    let mut input = request(
        vec![resource(
            "gpu",
            ResourceKind::GpuCompute,
            SchedulingMode::Exclusive,
        )],
        vec![compute("a", "gpu", 10.0)],
    );
    input.faults.push(sloforge_fabric_sim::TimedFault {
        id: "dead".into(),
        start_us: 0.0,
        end_us: None,
        effect: sloforge_fabric_sim::FaultEffect::ResourceUnavailable {
            resource_id: "gpu".into(),
        },
        ground_truth_label: "gpu_failure".into(),
    });
    assert!(matches!(simulate(&input), Err(SimError::Deadlock(_))));
}

#[test]
fn enforces_event_limit() {
    let mut input = request(
        vec![resource(
            "gpu",
            ResourceKind::GpuCompute,
            SchedulingMode::Exclusive,
        )],
        vec![compute("a", "gpu", 10.0)],
    );
    input.max_events = 1;
    assert!(matches!(
        simulate(&input),
        Err(SimError::EventLimitExceeded(1))
    ));
}

#[test]
fn rejects_demand_above_exclusive_capacity() {
    let mut input = request(
        vec![resource(
            "gpu",
            ResourceKind::GpuCompute,
            SchedulingMode::Exclusive,
        )],
        vec![compute("a", "gpu", 10.0)],
    );
    input.operations[0].demands[0].units = 2.0;
    assert!(
        matches!(validate(&input), Err(SimError::InvalidInput(message)) if message.contains("exceeds exclusive"))
    );
}

#[test]
fn accepts_every_algorithm_in_the_physical_ir_contract() {
    for algorithm in [
        "ring",
        "tree",
        "recursive_doubling",
        "direct",
        "pairwise",
        "all_to_all",
        "auto",
    ] {
        let collective = PhysicalOperation {
            id: format!("collective-{algorithm}"),
            kind: OperationKind::Collective {
                collective_id: "tp-0".into(),
                bytes: 1_000_000,
                algorithm: algorithm.into(),
                participating_ranks: vec!["rank-0".into(), "rank-1".into()],
            },
            rank_ids: vec!["rank-0".into(), "rank-1".into()],
            dependencies: Vec::new(),
            demands: vec![ResourceDemand {
                resource_id: "fabric".into(),
                units: 1.0,
            }],
            earliest_start_us: 0.0,
            uncertainty_fraction: 0.05,
            request_id: Some("request-0".into()),
        };
        let input = request(
            vec![resource(
                "fabric",
                ResourceKind::NetworkRail,
                SchedulingMode::FairShare,
            )],
            vec![collective],
        );
        assert!(validate(&input).is_ok(), "algorithm={algorithm}");
        assert!(simulate(&input).is_ok(), "algorithm={algorithm}");
    }
}

#[test]
fn rejects_unknown_fault_targets_and_invalid_transformed_demands() {
    let mut unknown_rank = request(
        vec![resource(
            "gpu",
            ResourceKind::GpuCompute,
            SchedulingMode::Exclusive,
        )],
        vec![compute("decode", "gpu", 10.0)],
    );
    unknown_rank.faults.push(sloforge_fabric_sim::TimedFault {
        id: "wrong-rank".into(),
        start_us: 0.0,
        end_us: None,
        effect: sloforge_fabric_sim::FaultEffect::RankSlowdown {
            rank_id: "rank-missing".into(),
            multiplier: 0.5,
        },
        ground_truth_label: "rank_specific_slowdown".into(),
    });
    assert!(validate(&unknown_rank).is_err());

    let mut replaced = request(
        vec![
            resource("gpu-a", ResourceKind::GpuCompute, SchedulingMode::Exclusive),
            resource("gpu-b", ResourceKind::GpuCompute, SchedulingMode::Exclusive),
        ],
        vec![sloforge_fabric_sim::PhysicalOperation {
            id: "two-resources".into(),
            kind: OperationKind::GpuCompute { duration_us: 10.0 },
            rank_ids: vec!["rank-0".into()],
            dependencies: Vec::new(),
            demands: vec![
                sloforge_fabric_sim::ResourceDemand {
                    resource_id: "gpu-a".into(),
                    units: 1.0,
                },
                sloforge_fabric_sim::ResourceDemand {
                    resource_id: "gpu-b".into(),
                    units: 1.0,
                },
            ],
            earliest_start_us: 0.0,
            uncertainty_fraction: 0.0,
            request_id: None,
        }],
    );
    replaced.counterfactuals.push(
        sloforge_fabric_sim::CounterfactualModifier::ReplaceResource {
            from_resource_id: "gpu-a".into(),
            to_resource_id: "gpu-b".into(),
        },
    );
    assert!(simulate(&replaced).is_err());

    let mut wrong_kind = request(
        vec![
            resource("gpu", ResourceKind::GpuCompute, SchedulingMode::Exclusive),
            resource("rail", ResourceKind::NetworkRail, SchedulingMode::FairShare),
        ],
        vec![compute("compute", "gpu", 10.0)],
    );
    wrong_kind.counterfactuals.push(
        sloforge_fabric_sim::CounterfactualModifier::ReplaceResource {
            from_resource_id: "gpu".into(),
            to_resource_id: "rail".into(),
        },
    );
    assert!(
        matches!(simulate(&wrong_kind), Err(SimError::InvalidInput(message)) if message.contains("cannot execute"))
    );
}
