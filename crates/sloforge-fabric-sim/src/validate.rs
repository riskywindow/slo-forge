use crate::{
    CounterfactualModifier, FabricSimulationRequest, FaultEffect, OperationKind, PhysicalOperation,
    ResourceKind, SchedulingMode,
};
use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

const HARD_MAX_OPERATIONS: usize = 1_000_000;
const HARD_MAX_EVENTS: usize = 50_000_000;
const MAX_EXACT_F64_INTEGER: u64 = 1_u64 << 53;

#[derive(Debug, thiserror::Error)]
pub enum SimError {
    #[error("unsupported fabric simulator schema {0:?}; expected major version 1")]
    UnsupportedSchema(String),
    #[error("invalid fabric simulation input: {0}")]
    InvalidInput(String),
    #[error("event limit of {0} exceeded")]
    EventLimitExceeded(usize),
    #[error("execution cannot progress: {0}")]
    Deadlock(String),
    #[error("could not serialize simulation input: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// Validate all graph, calibration, resource, fault, and bounded-work invariants.
///
/// # Errors
///
/// Returns a precise input error rather than attempting a device, transport, or
/// resource fallback.
pub fn validate(input: &FabricSimulationRequest) -> Result<(), SimError> {
    if input.schema_version.split('.').next() != Some("1") {
        return Err(SimError::UnsupportedSchema(input.schema_version.clone()));
    }
    if input.operations.is_empty() || input.resources.is_empty() {
        return invalid("at least one operation and physical resource are required");
    }
    if input.max_operations == 0 || input.max_operations > HARD_MAX_OPERATIONS {
        return invalid("max_operations must be in [1, 1000000]");
    }
    if input.max_events == 0 || input.max_events > HARD_MAX_EVENTS {
        return invalid("max_events must be in [1, 50000000]");
    }
    if input.operations.len() > input.max_operations {
        return invalid("operation count exceeds max_operations");
    }

    let mut resource_ids = HashSet::new();
    let mut resource_properties = HashMap::new();
    for resource in &input.resources {
        if resource.id.is_empty() || !resource_ids.insert(resource.id.as_str()) {
            return invalid(format!("empty or duplicate resource id {:?}", resource.id));
        }
        resource_properties.insert(
            resource.id.as_str(),
            (resource.kind, resource.scheduling, resource.capacity_units),
        );
        finite_positive("resource capacity_units", resource.capacity_units)?;
        if resource.max_concurrency == 0 {
            return invalid(format!("resource {} has zero concurrency", resource.id));
        }
        finite_nonnegative("resource hourly_cost_usd", resource.hourly_cost_usd)?;
        validate_curve(resource)?;
    }
    let mut sharing_ids = HashSet::new();
    for group in &input.sharing_groups {
        if group.id.is_empty() || !sharing_ids.insert(group.id.as_str()) {
            return invalid(format!(
                "empty or duplicate sharing group id {:?}",
                group.id
            ));
        }
        finite_positive("sharing group capacity_units", group.capacity_units)?;
        if group.max_concurrency == 0 {
            return invalid(format!("sharing group {} has zero concurrency", group.id));
        }
    }
    for resource in &input.resources {
        if let Some(group) = &resource.sharing_group {
            if !sharing_ids.contains(group.as_str()) {
                return invalid(format!(
                    "resource {} references unknown sharing group {group}",
                    resource.id
                ));
            }
        }
    }

    let mut operation_ids = HashSet::new();
    for operation in &input.operations {
        if operation.id.is_empty() || !operation_ids.insert(operation.id.as_str()) {
            return invalid(format!(
                "empty or duplicate operation id {:?}",
                operation.id
            ));
        }
    }
    for operation in &input.operations {
        validate_operation(operation, &operation_ids, &resource_properties)?;
    }
    validate_acyclic(&input.operations)?;
    let rank_ids: HashSet<_> = input
        .operations
        .iter()
        .flat_map(|operation| operation.rank_ids.iter().map(String::as_str))
        .collect();
    let collective_ids: HashSet<_> = input
        .operations
        .iter()
        .filter_map(|operation| match &operation.kind {
            OperationKind::Collective { collective_id, .. } => Some(collective_id.as_str()),
            _ => None,
        })
        .collect();
    validate_faults(input, &resource_ids, &rank_ids, &collective_ids)?;
    validate_counterfactuals(input, &resource_ids, &rank_ids)?;
    Ok(())
}

fn validate_curve(resource: &crate::PhysicalResource) -> Result<(), SimError> {
    let curve = &resource.curve;
    if curve.id.is_empty() || curve.points.is_empty() {
        return invalid(format!("resource {} has an empty curve", resource.id));
    }
    let provenance = &curve.provenance;
    if provenance.artifact_uri.is_empty()
        || provenance.environment_fingerprint.is_empty()
        || provenance.collected_at.is_empty()
        || provenance.artifact_sha256.len() != 64
        || !provenance
            .artifact_sha256
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        return invalid(format!(
            "resource {} curve has incomplete provenance or artifact hash",
            resource.id
        ));
    }
    let mut previous = None;
    for point in &curve.points {
        if point.message_bytes > MAX_EXACT_F64_INTEGER {
            return invalid(format!(
                "resource {} curve exceeds the exact timing-model range",
                resource.id
            ));
        }
        if previous.is_some_and(|value| point.message_bytes <= value) {
            return invalid(format!(
                "resource {} curve message sizes are not strictly increasing",
                resource.id
            ));
        }
        previous = Some(point.message_bytes);
        finite_nonnegative("curve latency_us", point.latency_us)?;
        finite_positive("curve bandwidth_gbps", point.bandwidth_gbps)?;
        if !point.uncertainty_fraction.is_finite()
            || !(0.0..=1.0).contains(&point.uncertainty_fraction)
        {
            return invalid("curve uncertainty_fraction must be in [0, 1]");
        }
    }
    Ok(())
}

fn validate_operation(
    operation: &PhysicalOperation,
    operation_ids: &HashSet<&str>,
    resource_properties: &HashMap<&str, (ResourceKind, SchedulingMode, f64)>,
) -> Result<(), SimError> {
    finite_nonnegative("operation earliest_start_us", operation.earliest_start_us)?;
    if !operation.uncertainty_fraction.is_finite()
        || !(0.0..=1.0).contains(&operation.uncertainty_fraction)
    {
        return invalid(format!(
            "operation {} uncertainty must be in [0, 1]",
            operation.id
        ));
    }
    let dependencies: BTreeSet<_> = operation.dependencies.iter().collect();
    if dependencies.len() != operation.dependencies.len()
        || dependencies.iter().any(|dependency| {
            dependency.as_str() == operation.id || !operation_ids.contains(dependency.as_str())
        })
    {
        return invalid(format!(
            "operation {} has a duplicate, self, or unknown dependency",
            operation.id
        ));
    }
    let mut demanded = HashSet::new();
    for demand in &operation.demands {
        if !resource_properties.contains_key(demand.resource_id.as_str())
            || !demanded.insert(demand.resource_id.as_str())
        {
            return invalid(format!(
                "operation {} has an unknown or duplicate resource {}",
                operation.id, demand.resource_id
            ));
        }
        finite_positive("resource demand units", demand.units)?;
        let (kind, scheduling, capacity) = resource_properties[demand.resource_id.as_str()];
        if !resource_compatible(&operation.kind, kind) {
            return invalid(format!(
                "operation {} cannot execute on resource {} of kind {kind:?}",
                operation.id, demand.resource_id
            ));
        }
        if scheduling == SchedulingMode::Exclusive && demand.units > capacity {
            return invalid(format!(
                "operation {} demand exceeds exclusive resource {} capacity",
                operation.id, demand.resource_id
            ));
        }
        if matches!(
            kind,
            ResourceKind::CpuCoreGroup
                | ResourceKind::GpuCompute
                | ResourceKind::GpuCopyEngine
                | ResourceKind::NicQueue
        ) && demand.units > 1_000_000.0
        {
            return invalid(format!(
                "operation {} has an implausibly large unit demand",
                operation.id
            ));
        }
    }
    if !matches!(operation.kind, OperationKind::Synchronization) && operation.demands.is_empty() {
        return invalid(format!(
            "operation {} has no physical resource",
            operation.id
        ));
    }
    validate_operation_kind(operation)?;
    Ok(())
}

const fn resource_compatible(operation: &OperationKind, resource: ResourceKind) -> bool {
    match operation {
        OperationKind::CpuLaunch { .. } => matches!(resource, ResourceKind::CpuCoreGroup),
        OperationKind::GpuCompute { .. } => matches!(resource, ResourceKind::GpuCompute),
        OperationKind::HbmAccess { .. } => matches!(resource, ResourceKind::GpuHbm),
        OperationKind::PointToPoint { .. }
        | OperationKind::Collective { .. }
        | OperationKind::ExpertDispatch { .. }
        | OperationKind::ExpertCombine { .. }
        | OperationKind::KvTransfer { .. } => matches!(
            resource,
            ResourceKind::NumaMemory
                | ResourceKind::GpuHbm
                | ResourceKind::GpuCopyEngine
                | ResourceKind::Nvlink
                | ResourceKind::Nvswitch
                | ResourceKind::Pcie
                | ResourceKind::NicQueue
                | ResourceKind::NetworkRail
        ),
        OperationKind::StorageFetch { .. } => matches!(
            resource,
            ResourceKind::NumaMemory | ResourceKind::NetworkRail | ResourceKind::StoragePath
        ),
        OperationKind::Startup { .. } => matches!(
            resource,
            ResourceKind::CpuCoreGroup
                | ResourceKind::NumaMemory
                | ResourceKind::GpuCompute
                | ResourceKind::GpuHbm
                | ResourceKind::NetworkRail
                | ResourceKind::StoragePath
        ),
        OperationKind::Synchronization => true,
    }
}

fn validate_operation_kind(operation: &PhysicalOperation) -> Result<(), SimError> {
    match &operation.kind {
        OperationKind::CpuLaunch { duration_us }
        | OperationKind::GpuCompute { duration_us }
        | OperationKind::Startup { duration_us } => {
            finite_nonnegative("operation duration_us", *duration_us)?;
        }
        OperationKind::Collective {
            collective_id,
            algorithm,
            participating_ranks,
            ..
        } => {
            let participants: BTreeSet<_> = participating_ranks.iter().collect();
            let ranks: BTreeSet<_> = operation.rank_ids.iter().collect();
            if collective_id.is_empty()
                || !matches!(
                    algorithm.as_str(),
                    "ring"
                        | "tree"
                        | "recursive_doubling"
                        | "direct"
                        | "pairwise"
                        | "all_to_all"
                        | "auto"
                )
                || participants.len() < 2
                || participants.len() != participating_ranks.len()
                || participants != ranks
            {
                return invalid(format!(
                    "collective {} has invalid id, algorithm, or rank barrier",
                    operation.id
                ));
            }
        }
        OperationKind::ExpertDispatch { experts, .. }
        | OperationKind::ExpertCombine { experts, .. } => {
            if *experts == 0 {
                return invalid(format!("operation {} has zero experts", operation.id));
            }
        }
        OperationKind::KvTransfer { chunks, .. } => {
            if *chunks == 0 {
                return invalid(format!("operation {} has zero KV chunks", operation.id));
            }
        }
        OperationKind::HbmAccess { .. }
        | OperationKind::PointToPoint { .. }
        | OperationKind::StorageFetch { .. }
        | OperationKind::Synchronization => {}
    }
    if operation.kind.bytes() > MAX_EXACT_F64_INTEGER {
        return invalid(format!(
            "operation {} byte count exceeds the exact timing-model range",
            operation.id
        ));
    }
    Ok(())
}

fn validate_acyclic(operations: &[PhysicalOperation]) -> Result<(), SimError> {
    let by_id: HashMap<_, _> = operations
        .iter()
        .enumerate()
        .map(|(index, operation)| (operation.id.as_str(), index))
        .collect();
    let mut degree: Vec<_> = operations
        .iter()
        .map(|operation| operation.dependencies.len())
        .collect();
    let mut dependents = vec![Vec::new(); operations.len()];
    for (index, operation) in operations.iter().enumerate() {
        for dependency in &operation.dependencies {
            dependents[by_id[dependency.as_str()]].push(index);
        }
    }
    let mut queue: VecDeque<_> = degree
        .iter()
        .enumerate()
        .filter_map(|(index, value)| (*value == 0).then_some(index))
        .collect();
    let mut visited = 0;
    while let Some(index) = queue.pop_front() {
        visited += 1;
        for dependent in &dependents[index] {
            degree[*dependent] -= 1;
            if degree[*dependent] == 0 {
                queue.push_back(*dependent);
            }
        }
    }
    if visited == operations.len() {
        Ok(())
    } else {
        invalid("operation dependency graph contains a cycle")
    }
}

fn validate_faults(
    input: &FabricSimulationRequest,
    resource_ids: &HashSet<&str>,
    rank_ids: &HashSet<&str>,
    collective_ids: &HashSet<&str>,
) -> Result<(), SimError> {
    let mut ids = HashSet::new();
    for fault in &input.faults {
        if fault.id.is_empty() || !ids.insert(fault.id.as_str()) {
            return invalid(format!("empty or duplicate fault id {:?}", fault.id));
        }
        finite_nonnegative("fault start_us", fault.start_us)?;
        if fault
            .end_us
            .is_some_and(|end| !end.is_finite() || end <= fault.start_us)
        {
            return invalid(format!("fault {} has an invalid end", fault.id));
        }
        if fault.ground_truth_label.is_empty() {
            return invalid(format!("fault {} has no ground-truth label", fault.id));
        }
        match &fault.effect {
            FaultEffect::ResourceRate {
                resource_id,
                multiplier,
            } => {
                known_resource(resource_ids, resource_id)?;
                rate_multiplier(*multiplier)?;
            }
            FaultEffect::ResourceUnavailable { resource_id } => {
                known_resource(resource_ids, resource_id)?;
            }
            FaultEffect::RankSlowdown {
                rank_id,
                multiplier,
            } => {
                known_target(rank_ids, rank_id, "rank")?;
                rate_multiplier(*multiplier)?;
            }
            FaultEffect::CollectiveDelay {
                collective_id,
                multiplier,
            } => {
                known_target(collective_ids, collective_id, "collective")?;
                rate_multiplier(*multiplier)?;
            }
        }
    }
    Ok(())
}

fn validate_counterfactuals(
    input: &FabricSimulationRequest,
    resource_ids: &HashSet<&str>,
    rank_ids: &HashSet<&str>,
) -> Result<(), SimError> {
    let fault_ids: HashSet<_> = input.faults.iter().map(|fault| fault.id.as_str()).collect();
    for modifier in &input.counterfactuals {
        match modifier {
            CounterfactualModifier::RemoveFault { fault_id } => {
                if !fault_ids.contains(fault_id.as_str()) {
                    return invalid(format!(
                        "counterfactual references unknown fault {fault_id}"
                    ));
                }
            }
            CounterfactualModifier::ScaleResourceCurve {
                resource_id,
                latency_multiplier,
                bandwidth_multiplier,
            } => {
                known_resource(resource_ids, resource_id)?;
                finite_positive("latency_multiplier", *latency_multiplier)?;
                finite_positive("bandwidth_multiplier", *bandwidth_multiplier)?;
            }
            CounterfactualModifier::ScaleRank {
                rank_id,
                duration_multiplier,
            } => {
                known_target(rank_ids, rank_id, "rank")?;
                finite_positive("duration_multiplier", *duration_multiplier)?;
            }
            CounterfactualModifier::ReplaceResource {
                from_resource_id,
                to_resource_id,
            } => {
                known_resource(resource_ids, from_resource_id)?;
                known_resource(resource_ids, to_resource_id)?;
                if from_resource_id == to_resource_id {
                    return invalid("resource replacement source and target are identical");
                }
            }
        }
    }
    Ok(())
}

fn known_target(targets: &HashSet<&str>, id: &str, kind: &str) -> Result<(), SimError> {
    if !id.is_empty() && targets.contains(id) {
        Ok(())
    } else {
        invalid(format!("unknown {kind} target {id:?}"))
    }
}

fn known_resource(resources: &HashSet<&str>, id: &str) -> Result<(), SimError> {
    if resources.contains(id) {
        Ok(())
    } else {
        invalid(format!("unknown physical resource {id}"))
    }
}

fn rate_multiplier(value: f64) -> Result<(), SimError> {
    if value.is_finite() && value > 0.0 && value <= 1.0 {
        Ok(())
    } else {
        invalid("fault rate multiplier must be in (0, 1]")
    }
}

fn finite_positive(name: &str, value: f64) -> Result<(), SimError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        invalid(format!("{name} must be finite and positive"))
    }
}

fn finite_nonnegative(name: &str, value: f64) -> Result<(), SimError> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        invalid(format!("{name} must be finite and non-negative"))
    }
}

fn invalid<T>(message: impl Into<String>) -> Result<T, SimError> {
    Err(SimError::InvalidInput(message.into()))
}
