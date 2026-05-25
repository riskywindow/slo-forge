use crate::model::{
    DurationDistribution, OutcomeStatus, ReplicaSpec, RequestOutcome, RequestSpec, RoutingPolicy,
    ScenarioAction, SimulationMetrics, SimulationOutput, SimulationProvenance, SimulationRequest,
    TraceEvent,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap, HashMap};

const EPSILON_MS: f64 = 0.000_001;

#[derive(Debug, thiserror::Error)]
pub enum SimError {
    #[error("unsupported simulator schema version {0:?}; expected major version 1")]
    UnsupportedSchema(String),
    #[error("invalid simulation input: {0}")]
    InvalidInput(String),
    #[error("event limit of {0} exceeded")]
    EventLimitExceeded(usize),
    #[error("could not serialize simulation input: {0}")]
    Serialization(#[from] serde_json::Error),
}

#[derive(Clone, Debug)]
enum EventKind {
    Arrival(usize),
    Deadline(usize),
    Cancel(usize),
    Action(usize),
    ReplicaReady(usize, u64),
    PrefillDone(usize, usize, u64, f64),
    DecodeTick(usize, u64, f64),
}

#[derive(Clone, Debug)]
struct Event {
    at: f64,
    order: u64,
    kind: EventKind,
}

impl PartialEq for Event {
    fn eq(&self, other: &Self) -> bool {
        self.at.to_bits() == other.at.to_bits() && self.order == other.order
    }
}

impl Eq for Event {}

impl PartialOrd for Event {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Event {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .at
            .total_cmp(&self.at)
            .then_with(|| other.order.cmp(&self.order))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct QueuedRequest {
    priority: u8,
    arrival_order: usize,
    request_idx: usize,
}

impl Ord for QueuedRequest {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority
            .cmp(&other.priority)
            .then_with(|| other.arrival_order.cmp(&self.arrival_order))
    }
}

impl PartialOrd for QueuedRequest {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug)]
struct RequestState {
    spec: RequestSpec,
    assigned: Option<usize>,
    generated: u32,
    prefill_started: Option<f64>,
    prefill_completed: Option<f64>,
    first_token: Option<f64>,
    last_token: Option<f64>,
    itls: Vec<f64>,
    terminal: Option<(OutcomeStatus, f64)>,
}

#[derive(Debug)]
#[allow(clippy::struct_excessive_bools)]
struct ReplicaState {
    spec: ReplicaSpec,
    healthy: bool,
    warm: bool,
    starting: bool,
    start_epoch: u64,
    decode_epoch: u64,
    decode_scheduled: bool,
    prefill_request: Option<usize>,
    queue: BinaryHeap<QueuedRequest>,
    active: Vec<usize>,
    slowdown: f64,
    startup_slowdown: f64,
    request_error_probability: f64,
    network_latency_ms: f64,
    network_jitter_ms: f64,
    max_context_tokens: Option<u32>,
    added_at: f64,
    removed_at: Option<f64>,
}

impl ReplicaState {
    fn outstanding(&self) -> usize {
        self.queue.len() + self.active.len() + usize::from(self.prefill_request.is_some())
    }

    fn accepting(&self) -> bool {
        self.healthy && self.removed_at.is_none() && self.queue.len() < self.spec.max_queue
    }
}

struct Engine<'a> {
    input: &'a SimulationRequest,
    rng: ChaCha8Rng,
    now: f64,
    events: BinaryHeap<Event>,
    next_order: u64,
    processed: usize,
    requests: Vec<RequestState>,
    replicas: Vec<ReplicaState>,
    replica_by_id: HashMap<String, usize>,
    round_robin: usize,
    traces: Vec<TraceEvent>,
    final_time: f64,
}

/// Execute a simulation without sleeping or consulting wall-clock time.
///
/// # Errors
///
/// Returns [`SimError::InvalidInput`] for inconsistent scenarios,
/// [`SimError::UnsupportedSchema`] for incompatible inputs, and
/// [`SimError::EventLimitExceeded`] when the caller's bounded-work guard fires.
pub fn simulate(input: &SimulationRequest) -> Result<SimulationOutput, SimError> {
    validate(input)?;
    let canonical = serde_json::to_vec(input)?;
    let input_sha256 = format!("{:x}", Sha256::digest(&canonical));
    let mut engine = Engine::new(input);
    engine.run()?;
    let (metrics, outcomes) = engine.summarize();
    Ok(SimulationOutput {
        schema_version: crate::SIM_SCHEMA_VERSION.to_owned(),
        provenance: SimulationProvenance {
            simulator_version: env!("CARGO_PKG_VERSION").to_owned(),
            input_sha256,
            service_curve_id: input.service_curve.id.clone(),
            measurement_artifact: input.service_curve.measurement_artifact.clone(),
            seed: input.seed,
        },
        metrics,
        outcomes,
        trace_events: engine.traces,
    })
}

fn validate(input: &SimulationRequest) -> Result<(), SimError> {
    if input.schema_version.split('.').next() != Some("1") {
        return Err(SimError::UnsupportedSchema(input.schema_version.clone()));
    }
    if input.replicas.is_empty() {
        return Err(SimError::InvalidInput(
            "at least one replica is required".into(),
        ));
    }
    if input.max_events == 0 {
        return Err(SimError::InvalidInput("max_events must be positive".into()));
    }
    if !(0.0..=1.0).contains(&input.canary_weight) {
        return Err(SimError::InvalidInput(
            "canary_weight must be in [0, 1]".into(),
        ));
    }
    let curve = &input.service_curve;
    for (name, value) in [
        ("prefill_intercept_ms", curve.prefill_intercept_ms),
        (
            "prefill_ms_per_prompt_token",
            curve.prefill_ms_per_prompt_token,
        ),
        ("prefill_ms_per_batch_item", curve.prefill_ms_per_batch_item),
        ("chunk_overhead_ms", curve.chunk_overhead_ms),
        ("decode_intercept_ms", curve.decode_intercept_ms),
        (
            "decode_ms_per_active_sequence",
            curve.decode_ms_per_active_sequence,
        ),
        (
            "decode_ms_per_context_token",
            curve.decode_ms_per_context_token,
        ),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(SimError::InvalidInput(format!(
                "{name} must be finite and non-negative"
            )));
        }
    }
    validate_distribution(&curve.startup)?;
    if curve.chunk_size_tokens == Some(0) {
        return Err(SimError::InvalidInput(
            "chunk_size_tokens must be positive when present".into(),
        ));
    }
    let mut ids = std::collections::HashSet::new();
    for replica in &input.replicas {
        validate_replica(replica)?;
        if !ids.insert(&replica.id) {
            return Err(SimError::InvalidInput(format!(
                "duplicate replica id {}",
                replica.id
            )));
        }
    }
    let mut request_ids = std::collections::HashSet::new();
    for request in &input.requests {
        if request.id.is_empty() || !request_ids.insert(&request.id) {
            return Err(SimError::InvalidInput(format!(
                "invalid or duplicate request id {}",
                request.id
            )));
        }
        if request.prompt_tokens == 0 || request.output_tokens == 0 {
            return Err(SimError::InvalidInput(format!(
                "request {} has a zero token count",
                request.id
            )));
        }
    }
    Ok(())
}

fn validate_replica(replica: &ReplicaSpec) -> Result<(), SimError> {
    if replica.id.is_empty() || replica.max_active_sequences == 0 {
        return Err(SimError::InvalidInput(
            "replica id and positive capacity are required".into(),
        ));
    }
    if !replica.service_rate_multiplier.is_finite() || replica.service_rate_multiplier <= 0.0 {
        return Err(SimError::InvalidInput(format!(
            "replica {} has invalid service multiplier",
            replica.id
        )));
    }
    if !replica.hourly_price_usd.is_finite() || replica.hourly_price_usd < 0.0 {
        return Err(SimError::InvalidInput(format!(
            "replica {} has invalid hourly price",
            replica.id
        )));
    }
    Ok(())
}

fn validate_distribution(distribution: &DurationDistribution) -> Result<(), SimError> {
    let valid = match distribution {
        DurationDistribution::Constant { value_ms } => value_ms.is_finite() && *value_ms >= 0.0,
        DurationDistribution::Uniform { min_ms, max_ms } => {
            min_ms.is_finite() && max_ms.is_finite() && *min_ms >= 0.0 && min_ms <= max_ms
        }
        DurationDistribution::Empirical { samples_ms } => {
            !samples_ms.is_empty()
                && samples_ms
                    .iter()
                    .all(|sample| sample.is_finite() && *sample >= 0.0)
        }
    };
    if valid {
        Ok(())
    } else {
        Err(SimError::InvalidInput(
            "startup distribution is empty, negative, or non-finite".into(),
        ))
    }
}

impl<'a> Engine<'a> {
    fn new(input: &'a SimulationRequest) -> Self {
        let requests = input
            .requests
            .iter()
            .cloned()
            .map(|spec| RequestState {
                spec,
                assigned: None,
                generated: 0,
                prefill_started: None,
                prefill_completed: None,
                first_token: None,
                last_token: None,
                itls: Vec::new(),
                terminal: None,
            })
            .collect();
        let replicas: Vec<_> = input
            .replicas
            .iter()
            .cloned()
            .map(|spec| Self::replica_state(spec, 0.0))
            .collect();
        let replica_by_id = replicas
            .iter()
            .enumerate()
            .map(|(idx, replica)| (replica.spec.id.clone(), idx))
            .collect();
        let mut engine = Self {
            input,
            rng: ChaCha8Rng::seed_from_u64(input.seed),
            now: 0.0,
            events: BinaryHeap::new(),
            next_order: 0,
            processed: 0,
            requests,
            replicas,
            replica_by_id,
            round_robin: 0,
            traces: Vec::new(),
            final_time: 0.0,
        };
        for (idx, request) in input.requests.iter().enumerate() {
            engine.push(u64_to_f64(request.arrival_ms), EventKind::Arrival(idx));
            if let Some(deadline) = request.deadline_ms {
                engine.push(
                    u64_to_f64(request.arrival_ms.saturating_add(deadline)),
                    EventKind::Deadline(idx),
                );
            }
            if let Some(cancel) = request.cancel_after_ms {
                engine.push(
                    u64_to_f64(request.arrival_ms.saturating_add(cancel)),
                    EventKind::Cancel(idx),
                );
            }
        }
        for (idx, action) in input.actions.iter().enumerate() {
            engine.push(u64_to_f64(action.at_ms), EventKind::Action(idx));
        }
        engine
    }

    fn replica_state(spec: ReplicaSpec, added_at: f64) -> ReplicaState {
        ReplicaState {
            warm: spec.initially_warm,
            spec,
            healthy: true,
            starting: false,
            start_epoch: 0,
            decode_epoch: 0,
            decode_scheduled: false,
            prefill_request: None,
            queue: BinaryHeap::new(),
            active: Vec::new(),
            slowdown: 1.0,
            startup_slowdown: 1.0,
            request_error_probability: 0.0,
            network_latency_ms: 0.0,
            network_jitter_ms: 0.0,
            max_context_tokens: None,
            added_at,
            removed_at: None,
        }
    }

    fn push(&mut self, at: f64, kind: EventKind) {
        self.events.push(Event {
            at: at.max(self.now),
            order: self.next_order,
            kind,
        });
        self.next_order = self.next_order.saturating_add(1);
    }

    fn run(&mut self) -> Result<(), SimError> {
        while let Some(first) = self.events.pop() {
            let at = first.at;
            let mut batch = vec![first];
            while self
                .events
                .peek()
                .is_some_and(|event| event.at.to_bits() == at.to_bits())
            {
                if let Some(event) = self.events.pop() {
                    batch.push(event);
                }
            }
            batch.sort_by_key(|event| (event_phase(&event.kind), event.order));
            for event in batch {
                self.processed += 1;
                if self.processed > self.input.max_events {
                    return Err(SimError::EventLimitExceeded(self.input.max_events));
                }
                if self.is_stale(&event.kind) {
                    continue;
                }
                self.now = event.at;
                self.final_time = self.final_time.max(self.now);
                match event.kind {
                    EventKind::Arrival(idx) => self.arrival(idx),
                    EventKind::Deadline(idx) => {
                        self.terminate(idx, OutcomeStatus::DeadlineExceeded);
                    }
                    EventKind::Cancel(idx) => self.terminate(idx, OutcomeStatus::Cancelled),
                    EventKind::Action(idx) => self.action(idx)?,
                    EventKind::ReplicaReady(idx, epoch) => self.replica_ready(idx, epoch),
                    EventKind::PrefillDone(replica, request, epoch, duration) => {
                        self.prefill_done(replica, request, epoch, duration);
                    }
                    EventKind::DecodeTick(replica, epoch, duration) => {
                        self.decode_tick(replica, epoch, duration);
                    }
                }
            }
            self.dispatch_all();
        }
        Ok(())
    }

    fn is_stale(&self, kind: &EventKind) -> bool {
        match kind {
            EventKind::Deadline(request_idx) | EventKind::Cancel(request_idx) => {
                self.requests[*request_idx].terminal.is_some()
            }
            EventKind::PrefillDone(replica_idx, request_idx, epoch, _) => {
                self.replicas.get(*replica_idx).is_none_or(|replica| {
                    replica.decode_epoch != *epoch || replica.prefill_request != Some(*request_idx)
                })
            }
            EventKind::DecodeTick(replica_idx, epoch, _) => {
                self.replicas.get(*replica_idx).is_none_or(|replica| {
                    replica.decode_epoch != *epoch
                        || !replica.decode_scheduled
                        || replica.active.is_empty()
                })
            }
            EventKind::ReplicaReady(replica_idx, epoch) => self
                .replicas
                .get(*replica_idx)
                .is_none_or(|replica| replica.start_epoch != *epoch || !replica.starting),
            EventKind::Arrival(_) | EventKind::Action(_) => false,
        }
    }

    fn arrival(&mut self, request_idx: usize) {
        self.trace_instant(
            "request_arrival",
            "request",
            "loadgen".into(),
            [("request_id", json!(self.requests[request_idx].spec.id))],
        );
        let Some(replica_idx) = self.route(request_idx) else {
            self.requests[request_idx].terminal = Some((OutcomeStatus::Rejected, self.now));
            return;
        };
        let context = self.requests[request_idx].spec.prompt_tokens;
        if self.replicas[replica_idx]
            .max_context_tokens
            .is_some_and(|limit| context > limit)
        {
            self.requests[request_idx].assigned = Some(replica_idx);
            self.requests[request_idx].terminal = Some((OutcomeStatus::SimulatedOom, self.now));
            return;
        }
        self.requests[request_idx].assigned = Some(replica_idx);
        self.replicas[replica_idx].queue.push(QueuedRequest {
            priority: self.requests[request_idx].spec.priority,
            arrival_order: request_idx,
            request_idx,
        });
        let replica_id = self.replicas[replica_idx].spec.id.clone();
        self.trace_instant(
            "routing_decision",
            "gateway",
            "gateway".into(),
            [
                ("request_id", json!(self.requests[request_idx].spec.id)),
                ("replica_id", json!(replica_id)),
                ("policy", json!(self.input.routing_policy)),
            ],
        );
        if !self.replicas[replica_idx].warm {
            self.start_replica(replica_idx);
        }
    }

    fn route(&mut self, request_idx: usize) -> Option<usize> {
        let eligible: Vec<_> = self
            .replicas
            .iter()
            .enumerate()
            .filter(|(_, replica)| replica.accepting())
            .map(|(idx, _)| idx)
            .collect();
        if eligible.is_empty() {
            return None;
        }
        let canary = self.requests[request_idx].spec.canary_eligible
            && self.rng.random::<f64>() < self.input.canary_weight;
        let preferred: Vec<_> = eligible
            .iter()
            .copied()
            .filter(|idx| self.replicas[*idx].spec.canary == canary)
            .collect();
        let pool = if preferred.is_empty() {
            eligible
        } else {
            preferred
        };
        let selection = match self.input.routing_policy {
            RoutingPolicy::RoundRobin => {
                let selected = pool[self.round_robin % pool.len()];
                self.round_robin = self.round_robin.wrapping_add(1);
                selected
            }
            RoutingPolicy::LeastOutstanding => *pool
                .iter()
                .min_by_key(|idx| (self.replicas[**idx].outstanding(), **idx))
                .unwrap_or(&pool[0]),
            RoutingPolicy::EarliestFinish => *pool
                .iter()
                .min_by(|left, right| {
                    self.estimated_finish(**left, request_idx)
                        .total_cmp(&self.estimated_finish(**right, request_idx))
                        .then_with(|| left.cmp(right))
                })
                .unwrap_or(&pool[0]),
            RoutingPolicy::SloSlackAware => {
                let deadline_at = self.requests[request_idx]
                    .spec
                    .deadline_ms
                    .map_or(f64::INFINITY, |d| self.now + u64_to_f64(d));
                *pool
                    .iter()
                    .min_by(|left, right| {
                        let left_finish = self.estimated_finish(**left, request_idx);
                        let right_finish = self.estimated_finish(**right, request_idx);
                        let left_miss = left_finish > deadline_at;
                        let right_miss = right_finish > deadline_at;
                        left_miss
                            .cmp(&right_miss)
                            .then_with(|| left_finish.total_cmp(&right_finish))
                            .then_with(|| left.cmp(right))
                    })
                    .unwrap_or(&pool[0])
            }
        };
        Some(selection)
    }

    fn estimated_finish(&self, replica_idx: usize, request_idx: usize) -> f64 {
        let replica = &self.replicas[replica_idx];
        let request = &self.requests[request_idx].spec;
        let ahead = usize_to_f64(replica.outstanding());
        let startup = if replica.warm {
            0.0
        } else {
            expected(&self.input.service_curve.startup)
        };
        let prefill = self.prefill_duration(replica_idx, request_idx);
        let decode = f64::from(request.output_tokens)
            * self.decode_duration_for(replica_idx, 1, f64::from(request.prompt_tokens));
        self.now + startup + (ahead + 1.0) * (prefill + decode)
    }

    fn start_replica(&mut self, replica_idx: usize) {
        if self.replicas[replica_idx].starting || !self.replicas[replica_idx].healthy {
            return;
        }
        self.replicas[replica_idx].starting = true;
        self.replicas[replica_idx].start_epoch += 1;
        let epoch = self.replicas[replica_idx].start_epoch;
        let factor = self.replicas[replica_idx].startup_slowdown;
        let duration = sample(&self.input.service_curve.startup, &mut self.rng) * factor;
        self.push(
            self.now + duration.max(EPSILON_MS),
            EventKind::ReplicaReady(replica_idx, epoch),
        );
        let id = self.replicas[replica_idx].spec.id.clone();
        self.trace_future("backend_startup", "backend", id, duration, BTreeMap::new());
    }

    fn replica_ready(&mut self, replica_idx: usize, epoch: u64) {
        let Some(replica) = self.replicas.get_mut(replica_idx) else {
            return;
        };
        if replica.start_epoch != epoch || !replica.healthy || replica.removed_at.is_some() {
            return;
        }
        replica.starting = false;
        replica.warm = true;
    }

    fn dispatch_all(&mut self) {
        for replica_idx in 0..self.replicas.len() {
            self.dispatch(replica_idx);
            self.ensure_decode(replica_idx);
        }
    }

    fn dispatch(&mut self, replica_idx: usize) {
        let can_dispatch = self.replicas.get(replica_idx).is_some_and(|replica| {
            replica.healthy
                && replica.warm
                && replica.removed_at.is_none()
                && replica.prefill_request.is_none()
                && replica.active.len() < replica.spec.max_active_sequences
        });
        if !can_dispatch {
            return;
        }
        let next = loop {
            let Some(queued) = self.replicas[replica_idx].queue.pop() else {
                return;
            };
            if self.requests[queued.request_idx].terminal.is_none() {
                break queued.request_idx;
            }
        };
        if self.rng.random::<f64>() < self.replicas[replica_idx].request_error_probability {
            self.requests[next].terminal = Some((OutcomeStatus::BackendFailed, self.now));
            return;
        }
        let queue_ms = self.now - u64_to_f64(self.requests[next].spec.arrival_ms);
        self.trace_complete(
            "gateway_queue",
            "gateway",
            "gateway".into(),
            queue_ms,
            BTreeMap::from([
                ("request_id".into(), json!(self.requests[next].spec.id)),
                (
                    "replica_id".into(),
                    json!(self.replicas[replica_idx].spec.id),
                ),
            ]),
        );
        self.requests[next].prefill_started = Some(self.now);
        self.replicas[replica_idx].prefill_request = Some(next);
        let epoch = self.replicas[replica_idx].decode_epoch;
        let duration = self.prefill_duration(replica_idx, next);
        self.push(
            self.now + duration.max(EPSILON_MS),
            EventKind::PrefillDone(replica_idx, next, epoch, duration),
        );
    }

    fn prefill_duration(&self, replica_idx: usize, request_idx: usize) -> f64 {
        let curve = &self.input.service_curve;
        let tokens = f64::from(self.requests[request_idx].spec.prompt_tokens);
        let chunk_count = curve.chunk_size_tokens.map_or(1.0, |chunk| {
            (tokens / f64::from(chunk.max(1))).ceil().max(1.0)
        });
        let base = curve.prefill_intercept_ms
            + curve.prefill_ms_per_prompt_token * tokens
            + curve.prefill_ms_per_batch_item
            + curve.chunk_overhead_ms * (chunk_count - 1.0);
        base * self.replicas[replica_idx].slowdown
            / self.replicas[replica_idx].spec.service_rate_multiplier
    }

    fn prefill_done(&mut self, replica_idx: usize, request_idx: usize, epoch: u64, duration: f64) {
        let prompt_tokens = self.requests[request_idx].spec.prompt_tokens;
        let chunks = self
            .input
            .service_curve
            .chunk_size_tokens
            .map_or(1, |size| prompt_tokens.div_ceil(size.max(1)));
        let Some(replica) = self.replicas.get_mut(replica_idx) else {
            return;
        };
        if replica.decode_epoch != epoch || !replica.healthy || replica.removed_at.is_some() {
            return;
        }
        replica.prefill_request = None;
        if self.requests[request_idx].terminal.is_none() {
            self.requests[request_idx].prefill_completed = Some(self.now);
            replica.active.push(request_idx);
            let id = replica.spec.id.clone();
            self.trace_complete(
                "prefill",
                "inference",
                id,
                duration,
                BTreeMap::from([
                    (
                        "request_id".into(),
                        json!(self.requests[request_idx].spec.id),
                    ),
                    ("chunks".into(), json!(chunks)),
                ]),
            );
        }
    }

    fn ensure_decode(&mut self, replica_idx: usize) {
        let Some(replica) = self.replicas.get(replica_idx) else {
            return;
        };
        if replica.decode_scheduled || replica.active.is_empty() || !replica.healthy {
            return;
        }
        let active: Vec<_> = replica
            .active
            .iter()
            .copied()
            .filter(|idx| self.requests[*idx].terminal.is_none())
            .collect();
        if active.is_empty() {
            self.replicas[replica_idx].active.clear();
            return;
        }
        let mean_context = active
            .iter()
            .map(|idx| {
                f64::from(self.requests[*idx].spec.prompt_tokens + self.requests[*idx].generated)
            })
            .sum::<f64>()
            / usize_to_f64(active.len());
        let estimated_duration = self.decode_duration_for(replica_idx, active.len(), mean_context);
        let jitter = self.replicas[replica_idx].network_jitter_ms;
        let duration = if jitter > 0.0 {
            estimated_duration - jitter / 2.0 + self.rng.random_range(0.0..=jitter)
        } else {
            estimated_duration
        };
        let epoch = self.replicas[replica_idx].decode_epoch;
        self.replicas[replica_idx].decode_scheduled = true;
        self.push(
            self.now + duration.max(EPSILON_MS),
            EventKind::DecodeTick(replica_idx, epoch, duration),
        );
    }

    fn decode_duration_for(&self, replica_idx: usize, active: usize, mean_context: f64) -> f64 {
        let curve = &self.input.service_curve;
        let replica = &self.replicas[replica_idx];
        let jitter = if replica.network_jitter_ms == 0.0 {
            0.0
        } else {
            // Estimation intentionally excludes random jitter; scheduled ticks add it below.
            replica.network_jitter_ms / 2.0
        };
        (curve.decode_intercept_ms
            + curve.decode_ms_per_active_sequence * usize_to_f64(active)
            + curve.decode_ms_per_context_token * mean_context)
            * replica.slowdown
            / replica.spec.service_rate_multiplier
            + replica.network_latency_ms
            + jitter
    }

    fn decode_tick(&mut self, replica_idx: usize, epoch: u64, duration: f64) {
        let Some(replica) = self.replicas.get_mut(replica_idx) else {
            return;
        };
        if replica.decode_epoch != epoch || !replica.healthy || replica.removed_at.is_some() {
            return;
        }
        replica.decode_scheduled = false;
        let active = std::mem::take(&mut replica.active);
        let replica_id = replica.spec.id.clone();
        let mut remaining = Vec::with_capacity(active.len());
        for request_idx in active {
            let request = &mut self.requests[request_idx];
            if request.terminal.is_some() {
                continue;
            }
            if let Some(last) = request.last_token {
                request.itls.push(self.now - last);
            } else {
                request.first_token = Some(self.now);
            }
            request.last_token = Some(self.now);
            request.generated += 1;
            if request.generated >= request.spec.output_tokens {
                request.terminal = Some((OutcomeStatus::Completed, self.now));
            } else {
                remaining.push(request_idx);
            }
        }
        self.replicas[replica_idx].active = remaining;
        self.trace_complete(
            "decode_iteration",
            "inference",
            replica_id,
            duration,
            BTreeMap::from([(
                "active_sequences".into(),
                json!(self.replicas[replica_idx].active.len()),
            )]),
        );
    }

    fn terminate(&mut self, request_idx: usize, status: OutcomeStatus) {
        if self.requests[request_idx].terminal.is_none() {
            self.requests[request_idx].terminal = Some((status, self.now));
            if let Some(replica_idx) = self.requests[request_idx].assigned {
                let replica = &mut self.replicas[replica_idx];
                if replica.prefill_request == Some(request_idx) {
                    replica.prefill_request = None;
                }
                replica
                    .queue
                    .retain(|queued| queued.request_idx != request_idx);
                replica.active.retain(|active| *active != request_idx);
                if replica.active.is_empty() && replica.decode_scheduled {
                    replica.decode_epoch += 1;
                    replica.decode_scheduled = false;
                }
            }
        }
    }

    fn action(&mut self, action_idx: usize) -> Result<(), SimError> {
        let action = self.input.actions[action_idx].action.clone();
        self.trace_instant(
            "scenario_action",
            "chaos",
            "controller".into(),
            [(
                "action",
                serde_json::to_value(&action).unwrap_or(Value::Null),
            )],
        );
        match action {
            ScenarioAction::AddReplica { replica } => {
                validate_replica(&replica)?;
                if self.replica_by_id.contains_key(&replica.id) {
                    return Err(SimError::InvalidInput(format!(
                        "duplicate added replica {}",
                        replica.id
                    )));
                }
                let idx = self.replicas.len();
                self.replica_by_id.insert(replica.id.clone(), idx);
                self.replicas.push(Self::replica_state(replica, self.now));
            }
            other => {
                let id = action_replica_id(&other);
                let Some(&idx) = self.replica_by_id.get(id) else {
                    return Err(SimError::InvalidInput(format!(
                        "action references unknown replica {id}"
                    )));
                };
                match other {
                    ScenarioAction::BackendCrash { .. } | ScenarioAction::RemoveReplica { .. } => {
                        let removed = matches!(other, ScenarioAction::RemoveReplica { .. });
                        self.fail_replica(idx, removed);
                    }
                    ScenarioAction::BackendRecover { warm, .. } => {
                        let replica = &mut self.replicas[idx];
                        replica.healthy = true;
                        replica.removed_at = None;
                        replica.warm = warm;
                        if !warm {
                            self.start_replica(idx);
                        }
                    }
                    ScenarioAction::BackendSlowdown { factor, .. } => {
                        require_positive("slowdown factor", factor)?;
                        self.replicas[idx].slowdown = factor;
                    }
                    ScenarioAction::StartupSlowdown { factor, .. } => {
                        require_positive("startup slowdown factor", factor)?;
                        self.replicas[idx].startup_slowdown = factor;
                    }
                    ScenarioAction::RequestErrors { probability, .. } => {
                        if !(0.0..=1.0).contains(&probability) {
                            return Err(SimError::InvalidInput(
                                "error probability outside [0, 1]".into(),
                            ));
                        }
                        self.replicas[idx].request_error_probability = probability;
                    }
                    ScenarioAction::CapacityLoss {
                        max_active_sequences,
                        ..
                    } => {
                        if max_active_sequences == 0 {
                            return Err(SimError::InvalidInput("capacity cannot be zero".into()));
                        }
                        self.replicas[idx].spec.max_active_sequences = max_active_sequences;
                    }
                    ScenarioAction::QueueSaturation { max_queue, .. } => {
                        self.replicas[idx].spec.max_queue = max_queue;
                    }
                    ScenarioAction::NetworkLatency {
                        latency_ms,
                        jitter_ms,
                        ..
                    } => {
                        if latency_ms < 0.0 || jitter_ms < 0.0 {
                            return Err(SimError::InvalidInput(
                                "network latency cannot be negative".into(),
                            ));
                        }
                        self.replicas[idx].network_latency_ms = latency_ms;
                        self.replicas[idx].network_jitter_ms = jitter_ms;
                    }
                    ScenarioAction::SimulatedOom {
                        max_context_tokens, ..
                    } => {
                        self.replicas[idx].max_context_tokens = Some(max_context_tokens);
                    }
                    ScenarioAction::AddReplica { .. } => {
                        return Err(SimError::InvalidInput(
                            "add-replica action reached an inconsistent dispatch state".into(),
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    fn fail_replica(&mut self, replica_idx: usize, removed: bool) {
        let replica = &mut self.replicas[replica_idx];
        replica.healthy = false;
        replica.warm = false;
        replica.starting = false;
        replica.start_epoch += 1;
        replica.decode_epoch += 1;
        replica.decode_scheduled = false;
        if let Some(request_idx) = replica.prefill_request.take()
            && self.requests[request_idx].terminal.is_none()
        {
            self.requests[request_idx].terminal = Some((OutcomeStatus::BackendFailed, self.now));
        }
        if removed {
            replica.removed_at = Some(self.now);
        }
        let active = std::mem::take(&mut replica.active);
        for request_idx in active {
            if self.requests[request_idx].terminal.is_none() {
                self.requests[request_idx].terminal =
                    Some((OutcomeStatus::BackendFailed, self.now));
            }
        }
        let mut queued = Vec::new();
        while let Some(item) = self.replicas[replica_idx].queue.pop() {
            if self.requests[item.request_idx].terminal.is_none() {
                queued.push(item.request_idx);
            }
        }
        for request_idx in queued {
            self.requests[request_idx].assigned = None;
            if let Some(new_replica) = self.route(request_idx) {
                self.requests[request_idx].assigned = Some(new_replica);
                self.replicas[new_replica].queue.push(QueuedRequest {
                    priority: self.requests[request_idx].spec.priority,
                    arrival_order: request_idx,
                    request_idx,
                });
                if !self.replicas[new_replica].warm {
                    self.start_replica(new_replica);
                }
            } else {
                self.requests[request_idx].terminal =
                    Some((OutcomeStatus::BackendFailed, self.now));
            }
        }
    }

    fn trace_instant<const N: usize>(
        &mut self,
        name: &str,
        category: &str,
        tid: String,
        args: [(&str, Value); N],
    ) {
        self.traces.push(TraceEvent {
            name: name.into(),
            cat: category.into(),
            ph: "i".into(),
            ts: self.now * 1_000.0,
            dur: None,
            pid: 1,
            tid,
            args: args
                .into_iter()
                .map(|(key, value)| (key.into(), value))
                .collect(),
        });
    }

    fn trace_complete(
        &mut self,
        name: &str,
        category: &str,
        tid: String,
        duration_ms: f64,
        args: BTreeMap<String, Value>,
    ) {
        self.traces.push(TraceEvent {
            name: name.into(),
            cat: category.into(),
            ph: "X".into(),
            ts: (self.now - duration_ms).max(0.0) * 1_000.0,
            dur: Some(duration_ms * 1_000.0),
            pid: 1,
            tid,
            args,
        });
    }

    fn trace_future(
        &mut self,
        name: &str,
        category: &str,
        tid: String,
        duration_ms: f64,
        args: BTreeMap<String, Value>,
    ) {
        self.traces.push(TraceEvent {
            name: name.into(),
            cat: category.into(),
            ph: "X".into(),
            ts: self.now * 1_000.0,
            dur: Some(duration_ms * 1_000.0),
            pid: 1,
            tid,
            args,
        });
    }

    #[allow(clippy::too_many_lines)]
    fn summarize(&self) -> (SimulationMetrics, Vec<RequestOutcome>) {
        let mut outcomes = Vec::with_capacity(self.requests.len());
        let mut ttfts = Vec::new();
        let mut e2es = Vec::new();
        let mut itls = Vec::new();
        let mut queue_times = Vec::new();
        let mut prefill_times = Vec::new();
        let mut decode_times = Vec::new();
        let mut completed = 0;
        let mut rejected = 0;
        let mut failed = 0;
        let mut misses = 0;
        let mut generated = 0_u64;
        for request in &self.requests {
            let (status, terminal_at) = request
                .terminal
                .unwrap_or((OutcomeStatus::BackendFailed, self.final_time));
            let arrival = u64_to_f64(request.spec.arrival_ms);
            let ttft = request.first_token.map(|at| at - arrival);
            let e2e = if status == OutcomeStatus::Completed {
                Some(terminal_at - arrival)
            } else {
                None
            };
            let queue_ms = request.prefill_started.map(|started| started - arrival);
            let prefill_ms = request
                .prefill_started
                .zip(request.prefill_completed)
                .map(|(started, completed)| completed - started);
            let decode_ms = request
                .first_token
                .filter(|_| status == OutcomeStatus::Completed)
                .map(|first_token| terminal_at - first_token);
            let mean_itl_ms = if request.itls.is_empty() {
                None
            } else {
                Some(request.itls.iter().sum::<f64>() / usize_to_f64(request.itls.len()))
            };
            if let Some(value) = ttft {
                ttfts.push(value);
            }
            if let Some(value) = e2e {
                e2es.push(value);
            }
            itls.extend(request.itls.iter().copied());
            if let Some(value) = queue_ms {
                queue_times.push(value);
            }
            if let Some(value) = prefill_ms {
                prefill_times.push(value);
            }
            if let Some(value) = decode_ms {
                decode_times.push(value);
            }
            completed += usize::from(status == OutcomeStatus::Completed);
            rejected += usize::from(status == OutcomeStatus::Rejected);
            failed += usize::from(matches!(
                status,
                OutcomeStatus::BackendFailed | OutcomeStatus::SimulatedOom
            ));
            let deadline_met = request.spec.deadline_ms.is_none_or(|deadline| {
                status == OutcomeStatus::Completed && terminal_at <= arrival + u64_to_f64(deadline)
            });
            misses += usize::from(!deadline_met && request.spec.deadline_ms.is_some());
            generated += u64::from(request.generated);
            outcomes.push(RequestOutcome {
                request_id: request.spec.id.clone(),
                status,
                replica_id: request
                    .assigned
                    .map(|idx| self.replicas[idx].spec.id.clone()),
                arrival_ms: arrival,
                first_token_ms: request.first_token,
                terminal_ms: terminal_at,
                completed_ms: (status == OutcomeStatus::Completed).then_some(terminal_at),
                queue_ms,
                prefill_ms,
                decode_ms,
                ttft_ms: ttft,
                e2e_ms: e2e,
                mean_itl_ms,
                generated_tokens: request.generated,
                deadline_met,
            });
        }
        outcomes.sort_by(|left, right| left.request_id.cmp(&right.request_id));
        let duration_s = self.final_time / 1_000.0;
        let cost = self
            .replicas
            .iter()
            .map(|replica| {
                let until = replica.removed_at.unwrap_or(self.final_time);
                (until - replica.added_at).max(0.0) / 3_600_000.0 * replica.spec.hourly_price_usd
            })
            .sum();
        let successful_with_deadline = outcomes
            .iter()
            .filter(|outcome| outcome.status == OutcomeStatus::Completed && outcome.deadline_met)
            .count();
        let metrics = SimulationMetrics {
            request_count: outcomes.len(),
            completed_count: completed,
            rejected_count: rejected,
            failed_count: failed,
            deadline_miss_count: misses,
            p50_ttft_ms: quantile(&mut ttfts, 0.50),
            p95_ttft_ms: quantile(&mut ttfts, 0.95),
            p99_ttft_ms: quantile(&mut ttfts, 0.99),
            p50_e2e_ms: quantile(&mut e2es, 0.50),
            p95_e2e_ms: quantile(&mut e2es, 0.95),
            p99_e2e_ms: quantile(&mut e2es, 0.99),
            p99_itl_ms: quantile(&mut itls, 0.99),
            p95_queue_ms: quantile(&mut queue_times, 0.95),
            p95_prefill_ms: quantile(&mut prefill_times, 0.95),
            p95_decode_ms: quantile(&mut decode_times, 0.95),
            throughput_tokens_per_s: if duration_s > 0.0 {
                u64_to_f64(generated) / duration_s
            } else {
                0.0
            },
            goodput_requests_per_s: if duration_s > 0.0 {
                usize_to_f64(successful_with_deadline) / duration_s
            } else {
                0.0
            },
            availability: if outcomes.is_empty() {
                1.0
            } else {
                usize_to_f64(completed) / usize_to_f64(outcomes.len())
            },
            cost_usd: cost,
            simulated_duration_ms: self.final_time,
            processed_events: self.processed,
        };
        (metrics, outcomes)
    }
}

fn action_replica_id(action: &ScenarioAction) -> &str {
    match action {
        ScenarioAction::BackendCrash { replica_id }
        | ScenarioAction::BackendRecover { replica_id, .. }
        | ScenarioAction::BackendSlowdown { replica_id, .. }
        | ScenarioAction::StartupSlowdown { replica_id, .. }
        | ScenarioAction::RequestErrors { replica_id, .. }
        | ScenarioAction::RemoveReplica { replica_id }
        | ScenarioAction::CapacityLoss { replica_id, .. }
        | ScenarioAction::QueueSaturation { replica_id, .. }
        | ScenarioAction::NetworkLatency { replica_id, .. }
        | ScenarioAction::SimulatedOom { replica_id, .. } => replica_id,
        ScenarioAction::AddReplica { replica } => &replica.id,
    }
}

const fn event_phase(kind: &EventKind) -> u8 {
    match kind {
        EventKind::Action(_) => 0,
        EventKind::ReplicaReady(_, _)
        | EventKind::PrefillDone(_, _, _, _)
        | EventKind::DecodeTick(_, _, _) => 1,
        EventKind::Deadline(_) | EventKind::Cancel(_) => 2,
        EventKind::Arrival(_) => 3,
    }
}

fn require_positive(name: &str, value: f64) -> Result<(), SimError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(SimError::InvalidInput(format!("{name} must be positive")))
    }
}

fn sample(distribution: &DurationDistribution, rng: &mut ChaCha8Rng) -> f64 {
    match distribution {
        DurationDistribution::Constant { value_ms } => *value_ms,
        DurationDistribution::Uniform { min_ms, max_ms } => {
            if min_ms >= max_ms {
                *min_ms
            } else {
                rng.random_range(*min_ms..=*max_ms)
            }
        }
        DurationDistribution::Empirical { samples_ms } => {
            if samples_ms.is_empty() {
                0.0
            } else {
                samples_ms[rng.random_range(0..samples_ms.len())]
            }
        }
    }
}

fn expected(distribution: &DurationDistribution) -> f64 {
    match distribution {
        DurationDistribution::Constant { value_ms } => *value_ms,
        DurationDistribution::Uniform { min_ms, max_ms } => (min_ms + max_ms) / 2.0,
        DurationDistribution::Empirical { samples_ms } => {
            if samples_ms.is_empty() {
                0.0
            } else {
                samples_ms.iter().sum::<f64>() / usize_to_f64(samples_ms.len())
            }
        }
    }
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn quantile(values: &mut [f64], probability: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let index = ((values.len() - 1) as f64 * probability).ceil() as usize;
    Some(values[index.min(values.len() - 1)])
}

#[allow(clippy::cast_precision_loss)]
fn usize_to_f64(value: usize) -> f64 {
    value as f64
}

#[allow(clippy::cast_precision_loss)]
fn u64_to_f64(value: u64) -> f64 {
    value as f64
}
