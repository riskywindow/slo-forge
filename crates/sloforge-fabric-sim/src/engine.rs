use crate::curve::{apply_curve_modifiers, estimate};
use crate::{
    ChromeTraceEvent, CounterfactualModifier, FabricSimulationMetrics, FabricSimulationOutput,
    FabricSimulationRequest, FaultEffect, OperationKind, OperationOutcome, OperationStatus,
    PhysicalOperation, PhysicalResource, ResourceMetrics, SchedulingMode, SimError,
    SimulationProvenance, validate,
};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap};

const EPSILON_US: f64 = 1e-9;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Pending,
    Active,
    Completed,
}

#[derive(Clone, Debug)]
struct RuntimeOperation {
    operation: PhysicalOperation,
    dependencies: Vec<usize>,
    state: State,
    ready_us: f64,
    start_us: Option<f64>,
    end_us: Option<f64>,
    base_duration_us: f64,
    uncertainty_us: f64,
    remaining_work_us: f64,
}

#[derive(Clone, Debug, Default)]
struct ResourceAccounting {
    busy_time_us: f64,
    transferred_bytes: u64,
    max_concurrent: usize,
}

struct Engine {
    input: FabricSimulationRequest,
    resources_by_id: HashMap<String, usize>,
    groups_by_id: HashMap<String, usize>,
    operations: Vec<RuntimeOperation>,
    now_us: f64,
    processed_events: usize,
    accounting: Vec<ResourceAccounting>,
    applied_faults: BTreeSet<String>,
}

/// Run a calibrated physical execution graph without consulting wall-clock time.
///
/// # Errors
///
/// Returns validation, bounded-work, serialization, or explicit deadlock errors.
/// The simulator never substitutes a missing resource or transport.
pub fn simulate(input: &FabricSimulationRequest) -> Result<FabricSimulationOutput, SimError> {
    validate(input)?;
    let canonical = serde_json::to_vec(input)?;
    let input_sha256 = format!("{:x}", Sha256::digest(&canonical));
    let mut transformed = input.clone();
    apply_counterfactuals(&mut transformed);
    let mut engine = Engine::new(transformed)?;
    engine.run()?;
    engine.output(input_sha256)
}

fn apply_counterfactuals(input: &mut FabricSimulationRequest) {
    apply_curve_modifiers(&mut input.resources, &input.counterfactuals);
    let removed_faults: BTreeSet<_> = input
        .counterfactuals
        .iter()
        .filter_map(|modifier| match modifier {
            CounterfactualModifier::RemoveFault { fault_id } => Some(fault_id.as_str()),
            _ => None,
        })
        .collect();
    input
        .faults
        .retain(|fault| !removed_faults.contains(fault.id.as_str()));
    for modifier in &input.counterfactuals {
        let CounterfactualModifier::ReplaceResource {
            from_resource_id,
            to_resource_id,
        } = modifier
        else {
            continue;
        };
        for operation in &mut input.operations {
            for demand in &mut operation.demands {
                if demand.resource_id == *from_resource_id {
                    demand.resource_id.clone_from(to_resource_id);
                }
            }
            let mut merged = BTreeMap::<String, f64>::new();
            for demand in &operation.demands {
                *merged.entry(demand.resource_id.clone()).or_default() += demand.units;
            }
            operation.demands = merged
                .into_iter()
                .map(|(resource_id, units)| crate::ResourceDemand { resource_id, units })
                .collect();
        }
    }
}

impl Engine {
    fn new(input: FabricSimulationRequest) -> Result<Self, SimError> {
        let resources_by_id: HashMap<_, _> = input
            .resources
            .iter()
            .enumerate()
            .map(|(index, resource)| (resource.id.clone(), index))
            .collect();
        let groups_by_id = input
            .sharing_groups
            .iter()
            .enumerate()
            .map(|(index, group)| (group.id.clone(), index))
            .collect();
        let operations_by_id: HashMap<_, _> = input
            .operations
            .iter()
            .enumerate()
            .map(|(index, operation)| (operation.id.clone(), index))
            .collect();
        let mut operations = Vec::with_capacity(input.operations.len());
        for operation in &input.operations {
            let (base_duration_us, uncertainty_us) =
                operation_duration(operation, &input.resources, &resources_by_id)?;
            let rank_scale = input
                .counterfactuals
                .iter()
                .filter_map(|modifier| match modifier {
                    CounterfactualModifier::ScaleRank {
                        rank_id,
                        duration_multiplier,
                    } if operation.rank_ids.contains(rank_id)
                        && matches!(
                            operation.kind,
                            OperationKind::GpuCompute { .. } | OperationKind::HbmAccess { .. }
                        ) =>
                    {
                        Some(*duration_multiplier)
                    }
                    _ => None,
                })
                .product::<f64>();
            let scaled_duration = base_duration_us * rank_scale;
            operations.push(RuntimeOperation {
                operation: operation.clone(),
                dependencies: operation
                    .dependencies
                    .iter()
                    .map(|dependency| operations_by_id[dependency])
                    .collect(),
                state: State::Pending,
                ready_us: operation.earliest_start_us,
                start_us: None,
                end_us: None,
                base_duration_us: scaled_duration,
                uncertainty_us: uncertainty_us * rank_scale,
                remaining_work_us: scaled_duration,
            });
        }
        let accounting = vec![ResourceAccounting::default(); input.resources.len()];
        Ok(Self {
            input,
            resources_by_id,
            groups_by_id,
            operations,
            now_us: 0.0,
            processed_events: 0,
            accounting,
            applied_faults: BTreeSet::new(),
        })
    }

    fn run(&mut self) -> Result<(), SimError> {
        while self
            .operations
            .iter()
            .any(|operation| operation.state != State::Completed)
        {
            self.complete_finished()?;
            self.start_ready()?;
            if self
                .operations
                .iter()
                .all(|operation| operation.state == State::Completed)
            {
                break;
            }

            let active = self.active_indices();
            if active.is_empty() {
                let Some(next) = self.next_ready_time() else {
                    return Err(SimError::Deadlock(
                        "pending operations have unsatisfied dependencies".into(),
                    ));
                };
                if next <= self.now_us + EPSILON_US {
                    return Err(SimError::Deadlock(
                        "ready operations cannot acquire bounded exclusive resources".into(),
                    ));
                }
                self.now_us = next;
                self.event()?;
                continue;
            }

            let rates = self.progress_rates(&active);
            let completion_delta = active
                .iter()
                .zip(&rates)
                .filter_map(|(index, rate)| {
                    (*rate > 0.0).then_some(self.operations[*index].remaining_work_us / rate)
                })
                .fold(f64::INFINITY, f64::min);
            let next_boundary = self.next_boundary_time();
            let boundary_delta = next_boundary.map_or(f64::INFINITY, |time| time - self.now_us);
            let delta = completion_delta.min(boundary_delta);
            if !delta.is_finite() || delta < -EPSILON_US {
                return Err(SimError::Deadlock(
                    "active work has zero service rate and no recovery boundary".into(),
                ));
            }
            self.advance(&active, &rates, delta.max(0.0));
            self.now_us += delta.max(0.0);
            self.event()?;
        }
        Ok(())
    }

    fn complete_finished(&mut self) -> Result<(), SimError> {
        let mut finished: Vec<_> = self
            .operations
            .iter()
            .enumerate()
            .filter_map(|(index, operation)| {
                (operation.state == State::Active && operation.remaining_work_us <= EPSILON_US)
                    .then_some(index)
            })
            .collect();
        finished.sort_by(|left, right| {
            self.operations[*left]
                .operation
                .id
                .cmp(&self.operations[*right].operation.id)
        });
        for index in finished {
            self.operations[index].state = State::Completed;
            self.operations[index].remaining_work_us = 0.0;
            self.operations[index].end_us = Some(self.now_us);
            let bytes = self.operations[index].operation.kind.bytes();
            for demand in &self.operations[index].operation.demands {
                let resource_index = self.resources_by_id[&demand.resource_id];
                self.accounting[resource_index].transferred_bytes = self.accounting[resource_index]
                    .transferred_bytes
                    .saturating_add(bytes);
            }
            self.event()?;
        }
        Ok(())
    }

    fn start_ready(&mut self) -> Result<(), SimError> {
        let mut candidates: Vec<_> =
            self.operations
                .iter()
                .enumerate()
                .filter_map(|(index, operation)| {
                    (operation.state == State::Pending
                        && operation.earliest_start_us() <= self.now_us + EPSILON_US
                        && operation.dependencies.iter().all(|dependency| {
                            self.operations[*dependency].state == State::Completed
                        }))
                    .then_some(index)
                })
                .collect();
        candidates.sort_by(|left, right| {
            self.operations[*left]
                .operation
                .id
                .cmp(&self.operations[*right].operation.id)
        });
        for index in candidates {
            if !self.can_admit(index) {
                continue;
            }
            let dependency_ready = self.operations[index]
                .dependencies
                .iter()
                .filter_map(|dependency| self.operations[*dependency].end_us)
                .fold(0.0_f64, f64::max);
            let operation = &mut self.operations[index];
            operation.ready_us = operation.operation.earliest_start_us.max(dependency_ready);
            operation.start_us = Some(self.now_us);
            operation.state = State::Active;
            self.event()?;
        }
        Ok(())
    }

    fn can_admit(&self, candidate: usize) -> bool {
        let active = self.active_indices();
        let operation = &self.operations[candidate].operation;
        for demand in &operation.demands {
            let resource_index = self.resources_by_id[&demand.resource_id];
            let resource = &self.input.resources[resource_index];
            let users = active
                .iter()
                .filter(|index| {
                    self.operations[**index]
                        .operation
                        .demands
                        .iter()
                        .any(|item| item.resource_id == demand.resource_id)
                })
                .count();
            match resource.scheduling {
                SchedulingMode::Exclusive if users > 0 => return false,
                SchedulingMode::FairShare if users >= resource.max_concurrency => return false,
                SchedulingMode::Exclusive | SchedulingMode::FairShare => {}
            }
            if let Some(group_id) = &resource.sharing_group {
                let group = &self.input.sharing_groups[self.groups_by_id[group_id]];
                let group_users = active
                    .iter()
                    .filter(|index| self.operation_uses_group(**index, group_id))
                    .count();
                if group_users >= group.max_concurrency {
                    return false;
                }
            }
        }
        true
    }

    fn progress_rates(&mut self, active: &[usize]) -> Vec<f64> {
        let mut resource_units: HashMap<usize, f64> = HashMap::new();
        let mut group_units: HashMap<usize, f64> = HashMap::new();
        let mut resource_users: HashMap<usize, usize> = HashMap::new();
        for index in active {
            for demand in &self.operations[*index].operation.demands {
                let resource_index = self.resources_by_id[&demand.resource_id];
                *resource_units.entry(resource_index).or_default() += demand.units;
                *resource_users.entry(resource_index).or_default() += 1;
                if let Some(group_id) = &self.input.resources[resource_index].sharing_group {
                    *group_units.entry(self.groups_by_id[group_id]).or_default() += demand.units;
                }
            }
        }
        for (index, users) in resource_users {
            self.accounting[index].max_concurrent =
                self.accounting[index].max_concurrent.max(users);
        }

        active
            .iter()
            .map(|index| {
                let operation = &self.operations[*index].operation;
                let mut rate = 1.0_f64;
                for demand in &operation.demands {
                    let resource_index = self.resources_by_id[&demand.resource_id];
                    let resource = &self.input.resources[resource_index];
                    if resource.scheduling == SchedulingMode::FairShare {
                        let total = resource_units[&resource_index];
                        rate = rate.min((resource.capacity_units / total).min(1.0));
                    }
                    if let Some(group_id) = &resource.sharing_group {
                        let group_index = self.groups_by_id[group_id];
                        let group = &self.input.sharing_groups[group_index];
                        rate =
                            rate.min((group.capacity_units / group_units[&group_index]).min(1.0));
                    }
                }
                rate * self.fault_rate(*index)
            })
            .collect()
    }

    fn fault_rate(&mut self, operation_index: usize) -> f64 {
        let operation = &self.operations[operation_index].operation;
        let mut rate = 1.0;
        for fault in &self.input.faults {
            if !fault_active(fault.start_us, fault.end_us, self.now_us) {
                continue;
            }
            let factor = match &fault.effect {
                FaultEffect::ResourceRate {
                    resource_id,
                    multiplier,
                } if operation
                    .demands
                    .iter()
                    .any(|demand| demand.resource_id == *resource_id) =>
                {
                    Some(*multiplier)
                }
                FaultEffect::ResourceUnavailable { resource_id }
                    if operation
                        .demands
                        .iter()
                        .any(|demand| demand.resource_id == *resource_id) =>
                {
                    Some(0.0)
                }
                FaultEffect::RankSlowdown {
                    rank_id,
                    multiplier,
                } if operation.rank_ids.contains(rank_id)
                    && matches!(
                        operation.kind,
                        OperationKind::GpuCompute { .. } | OperationKind::HbmAccess { .. }
                    ) =>
                {
                    // Rank service degradation models GPU execution, not a
                    // silent reduction in calibrated link capacity. Collective
                    // participants still observe the delayed dependency/barrier.
                    Some(*multiplier)
                }
                FaultEffect::CollectiveDelay {
                    collective_id,
                    multiplier,
                } if matches!(
                    &operation.kind,
                    OperationKind::Collective { collective_id: id, .. } if id == collective_id
                ) =>
                {
                    Some(*multiplier)
                }
                _ => None,
            };
            if let Some(factor) = factor {
                rate *= factor;
                self.applied_faults.insert(fault.id.clone());
            }
        }
        rate
    }

    fn advance(&mut self, active: &[usize], rates: &[f64], delta: f64) {
        let mut busy_resources = BTreeSet::new();
        for (index, rate) in active.iter().zip(rates) {
            self.operations[*index].remaining_work_us =
                (self.operations[*index].remaining_work_us - delta * rate).max(0.0);
            for demand in &self.operations[*index].operation.demands {
                busy_resources.insert(self.resources_by_id[&demand.resource_id]);
            }
        }
        for resource_index in busy_resources {
            self.accounting[resource_index].busy_time_us += delta;
        }
    }

    fn next_ready_time(&self) -> Option<f64> {
        self.operations
            .iter()
            .filter(|operation| {
                operation.state == State::Pending
                    && operation
                        .dependencies
                        .iter()
                        .all(|dependency| self.operations[*dependency].state == State::Completed)
            })
            .map(RuntimeOperation::earliest_start_us)
            .min_by(f64::total_cmp)
    }

    fn next_boundary_time(&self) -> Option<f64> {
        let fault_boundaries = self
            .input
            .faults
            .iter()
            .flat_map(|fault| [Some(fault.start_us), fault.end_us].into_iter().flatten());
        let ready_boundaries = self.operations.iter().filter_map(|operation| {
            (operation.state == State::Pending
                && operation
                    .dependencies
                    .iter()
                    .all(|dependency| self.operations[*dependency].state == State::Completed))
            .then_some(operation.earliest_start_us())
        });
        fault_boundaries
            .chain(ready_boundaries)
            .filter(|time| *time > self.now_us + EPSILON_US)
            .min_by(f64::total_cmp)
    }

    fn active_indices(&self) -> Vec<usize> {
        self.operations
            .iter()
            .enumerate()
            .filter_map(|(index, operation)| (operation.state == State::Active).then_some(index))
            .collect()
    }

    fn operation_uses_group(&self, operation_index: usize, group_id: &str) -> bool {
        self.operations[operation_index]
            .operation
            .demands
            .iter()
            .any(|demand| {
                self.input.resources[self.resources_by_id[&demand.resource_id]]
                    .sharing_group
                    .as_deref()
                    == Some(group_id)
            })
    }

    fn event(&mut self) -> Result<(), SimError> {
        self.processed_events += 1;
        if self.processed_events > self.input.max_events {
            Err(SimError::EventLimitExceeded(self.input.max_events))
        } else {
            Ok(())
        }
    }

    fn output(self, input_sha256: String) -> Result<FabricSimulationOutput, SimError> {
        let (mut outcomes, mut traces, uncertainty_squared) = self.outcomes_and_traces()?;
        outcomes.sort_by(|left, right| left.operation_id.cmp(&right.operation_id));
        let makespan_us = outcomes
            .iter()
            .map(|outcome| outcome.end_us)
            .fold(0.0, f64::max);
        traces.extend(self.fault_trace_events(makespan_us)?);
        traces.sort_by(|left, right| {
            left.ts
                .total_cmp(&right.ts)
                .then_with(|| left.name.cmp(&right.name))
        });
        let total_work_us: f64 = outcomes
            .iter()
            .map(|outcome| outcome.base_duration_us)
            .sum();
        let total_transferred_bytes = outcomes
            .iter()
            .map(|outcome| outcome.transferred_bytes)
            .sum();
        let resources = self.resource_metrics(makespan_us);
        let cost_usd = self
            .input
            .resources
            .iter()
            .map(|resource| resource.hourly_cost_usd * makespan_us / 3_600_000_000.0)
            .sum();
        let interval = 1.96 * uncertainty_squared.sqrt();
        let overlap_efficiency = if total_work_us > 0.0 {
            ((total_work_us - makespan_us) / total_work_us).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let calibration_artifacts = self
            .input
            .resources
            .iter()
            .map(|resource| resource.curve.provenance.artifact_uri.clone())
            .collect();
        let calibration_kinds = self
            .input
            .resources
            .iter()
            .map(|resource| resource.curve.provenance.kind.clone())
            .collect();
        Ok(FabricSimulationOutput {
            schema_version: crate::FABRIC_SIM_SCHEMA_VERSION.into(),
            provenance: SimulationProvenance {
                simulator_version: env!("CARGO_PKG_VERSION").into(),
                input_sha256,
                seed: self.input.seed,
                calibration_artifacts,
                calibration_kinds,
                counterfactual_count: self.input.counterfactuals.len(),
            },
            metrics: FabricSimulationMetrics {
                operation_count: outcomes.len(),
                makespan_us,
                total_work_us,
                total_transferred_bytes,
                cost_usd,
                processed_events: self.processed_events,
                overlap_efficiency,
                predicted_lower_us: (makespan_us - interval).max(0.0),
                predicted_upper_us: makespan_us + interval,
                resources,
            },
            operations: outcomes,
            trace_events: traces,
            applied_faults: self.applied_faults.into_iter().collect(),
            applied_counterfactuals: self.input.counterfactuals,
        })
    }

    fn outcomes_and_traces(
        &self,
    ) -> Result<(Vec<OperationOutcome>, Vec<ChromeTraceEvent>, f64), SimError> {
        let mut outcomes = Vec::with_capacity(self.operations.len());
        let mut traces = Vec::with_capacity(self.operations.len());
        let mut uncertainty_squared = 0.0;
        for operation in &self.operations {
            let start = operation.start_us.ok_or_else(|| {
                SimError::Deadlock(format!(
                    "operation {} never started",
                    operation.operation.id
                ))
            })?;
            let end = operation.end_us.ok_or_else(|| {
                SimError::Deadlock(format!(
                    "operation {} never completed",
                    operation.operation.id
                ))
            })?;
            uncertainty_squared += operation.uncertainty_us.powi(2);
            let resources: Vec<_> = operation
                .operation
                .demands
                .iter()
                .map(|demand| demand.resource_id.clone())
                .collect();
            outcomes.push(OperationOutcome {
                operation_id: operation.operation.id.clone(),
                status: OperationStatus::Completed,
                start_us: start,
                end_us: end,
                duration_us: end - start,
                base_duration_us: operation.base_duration_us,
                wait_us: start - operation.ready_us,
                transferred_bytes: operation.operation.kind.bytes(),
                uncertainty_us: operation.uncertainty_us,
                rank_ids: operation.operation.rank_ids.clone(),
                resource_ids: resources.clone(),
            });
            let mut args = BTreeMap::new();
            args.insert("resources".into(), json!(resources));
            args.insert("ranks".into(), json!(operation.operation.rank_ids));
            args.insert("bytes".into(), json!(operation.operation.kind.bytes()));
            args.insert("request_id".into(), json!(operation.operation.request_id));
            traces.push(ChromeTraceEvent {
                name: operation.operation.id.clone(),
                cat: operation_category(&operation.operation.kind).into(),
                ph: "X".into(),
                ts: start,
                dur: end - start,
                pid: 1,
                tid: operation
                    .operation
                    .rank_ids
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "coordinator".into()),
                args,
            });
        }
        Ok((outcomes, traces, uncertainty_squared))
    }

    fn resource_metrics(&self, makespan_us: f64) -> Vec<ResourceMetrics> {
        self.input
            .resources
            .iter()
            .zip(&self.accounting)
            .map(|(resource, accounting)| ResourceMetrics {
                resource_id: resource.id.clone(),
                busy_time_us: accounting.busy_time_us,
                utilization: if makespan_us > 0.0 {
                    (accounting.busy_time_us / makespan_us).clamp(0.0, 1.0)
                } else {
                    0.0
                },
                transferred_bytes: accounting.transferred_bytes,
                max_concurrent: accounting.max_concurrent,
            })
            .collect()
    }

    fn fault_trace_events(&self, makespan_us: f64) -> Result<Vec<ChromeTraceEvent>, SimError> {
        self.input
            .faults
            .iter()
            .filter(|fault| self.applied_faults.contains(&fault.id))
            .map(|fault| {
                let mut args = BTreeMap::new();
                args.insert("ground_truth_label".into(), json!(fault.ground_truth_label));
                args.insert("effect".into(), serde_json::to_value(&fault.effect)?);
                Ok(ChromeTraceEvent {
                    name: fault.id.clone(),
                    cat: "fabric_fault".into(),
                    ph: "X".into(),
                    ts: fault.start_us,
                    dur: fault.end_us.unwrap_or(makespan_us).min(makespan_us)
                        - fault.start_us.min(makespan_us),
                    pid: 1,
                    tid: "fault-injector".into(),
                    args,
                })
            })
            .collect()
    }
}

impl RuntimeOperation {
    fn earliest_start_us(&self) -> f64 {
        self.operation.earliest_start_us
    }
}

fn operation_duration(
    operation: &PhysicalOperation,
    resources: &[PhysicalResource],
    resources_by_id: &HashMap<String, usize>,
) -> Result<(f64, f64), SimError> {
    let (direct_duration, bytes, repetitions, collective_factor) = match &operation.kind {
        OperationKind::CpuLaunch { duration_us }
        | OperationKind::GpuCompute { duration_us }
        | OperationKind::Startup { duration_us } => (Some(*duration_us), 0, 1, 1.0),
        OperationKind::KvTransfer { bytes, chunks } => {
            (None, bytes.div_ceil(u64::from(*chunks)), *chunks, 1.0)
        }
        OperationKind::Collective {
            bytes,
            algorithm,
            participating_ranks,
            ..
        } => {
            let rank_count = u32::try_from(participating_ranks.len())
                .map_err(|_| SimError::InvalidInput("collective rank count exceeds u32".into()))?;
            let ranks = f64::from(rank_count);
            let factor = match algorithm.as_str() {
                // Auto is a deterministic simulator policy, not an implicit
                // transport fallback: model a ring and retain the requested
                // algorithm in the emitted operation trace.
                "ring" | "auto" => 2.0 * (ranks - 1.0) / ranks,
                "tree" | "recursive_doubling" => ranks.log2().ceil(),
                "all_to_all" | "pairwise" => ranks - 1.0,
                "direct" => 1.0,
                _ => {
                    return Err(SimError::InvalidInput(
                        "unknown collective algorithm".into(),
                    ));
                }
            };
            (None, *bytes, 1, factor)
        }
        OperationKind::HbmAccess { bytes }
        | OperationKind::PointToPoint { bytes }
        | OperationKind::ExpertDispatch { bytes, .. }
        | OperationKind::ExpertCombine { bytes, .. }
        | OperationKind::StorageFetch { bytes } => (None, *bytes, 1, 1.0),
        OperationKind::Synchronization => (Some(0.0), 0, 1, 1.0),
    };
    if let Some(duration) = direct_duration {
        return Ok((duration, duration * operation.uncertainty_fraction));
    }
    let mut duration = 0.0;
    let mut variance = 0.0;
    for demand in &operation.demands {
        let resource = &resources[resources_by_id[&demand.resource_id]];
        let estimate = estimate(&resource.curve, bytes);
        let resource_duration = estimate.duration_us * f64::from(repetitions) * collective_factor;
        duration += resource_duration;
        variance += (resource_duration * estimate.uncertainty_fraction).powi(2);
    }
    let uncertainty = variance
        .sqrt()
        .max(duration * operation.uncertainty_fraction);
    Ok((duration, uncertainty))
}

fn fault_active(start: f64, end: Option<f64>, now: f64) -> bool {
    now + EPSILON_US >= start && end.is_none_or(|value| now < value - EPSILON_US)
}

fn operation_category(kind: &OperationKind) -> &'static str {
    match kind {
        OperationKind::CpuLaunch { .. } => "cpu",
        OperationKind::GpuCompute { .. } | OperationKind::HbmAccess { .. } => "gpu",
        OperationKind::PointToPoint { .. }
        | OperationKind::Collective { .. }
        | OperationKind::ExpertDispatch { .. }
        | OperationKind::ExpertCombine { .. } => "fabric",
        OperationKind::KvTransfer { .. } => "kv_transfer",
        OperationKind::StorageFetch { .. } => "storage",
        OperationKind::Startup { .. } => "startup",
        OperationKind::Synchronization => "synchronization",
    }
}
