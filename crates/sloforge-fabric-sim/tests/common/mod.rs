use sloforge_fabric_sim::{
    CalibrationProvenance, CurvePoint, FabricSimulationRequest, OperationKind, PhysicalOperation,
    PhysicalResource, ProvenanceKind, ResourceDemand, ResourceKind, SchedulingMode, ServiceCurve,
};

pub fn provenance(name: &str) -> CalibrationProvenance {
    CalibrationProvenance {
        kind: ProvenanceKind::Synthetic,
        artifact_uri: format!("fixture://{name}"),
        artifact_sha256: "d".repeat(64),
        environment_fingerprint: "deterministic-ci-fixture-v1".into(),
        collected_at: "2026-08-01T00:00:00Z".into(),
    }
}

pub fn resource(id: &str, kind: ResourceKind, scheduling: SchedulingMode) -> PhysicalResource {
    PhysicalResource {
        id: id.into(),
        kind,
        scheduling,
        capacity_units: 1.0,
        max_concurrency: if scheduling == SchedulingMode::Exclusive {
            1
        } else {
            8
        },
        curve: ServiceCurve {
            id: format!("{id}-curve"),
            points: vec![CurvePoint {
                message_bytes: 1_000_000,
                latency_us: 0.0,
                bandwidth_gbps: 8.0,
                uncertainty_fraction: 0.05,
            }],
            provenance: provenance(id),
        },
        sharing_group: None,
        hourly_cost_usd: 1.0,
    }
}

pub fn compute(id: &str, resource_id: &str, duration_us: f64) -> PhysicalOperation {
    PhysicalOperation {
        id: id.into(),
        kind: OperationKind::GpuCompute { duration_us },
        rank_ids: vec![format!("rank-{id}")],
        dependencies: Vec::new(),
        demands: vec![ResourceDemand {
            resource_id: resource_id.into(),
            units: 1.0,
        }],
        earliest_start_us: 0.0,
        uncertainty_fraction: 0.05,
        request_id: Some("request-0".into()),
    }
}

pub fn request(
    resources: Vec<PhysicalResource>,
    operations: Vec<PhysicalOperation>,
) -> FabricSimulationRequest {
    FabricSimulationRequest {
        schema_version: "1.0".into(),
        seed: 42,
        resources,
        sharing_groups: Vec::new(),
        operations,
        faults: Vec::new(),
        counterfactuals: Vec::new(),
        max_events: 100_000,
        max_operations: 10_000,
    }
}
