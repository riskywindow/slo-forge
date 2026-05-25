#![allow(clippy::expect_used)]

use proptest::prelude::*;
use sloforge_sim::{
    DurationDistribution, OutcomeStatus, ReplicaSpec, RequestSpec, RoutingPolicy, ScenarioAction,
    ServiceCurve, SimulationRequest, TimedAction, simulate,
};

fn curve() -> ServiceCurve {
    ServiceCurve {
        id: "measured/toy".into(),
        measurement_artifact: "raw/toy.json".into(),
        prefill_intercept_ms: 0.0,
        prefill_ms_per_prompt_token: 0.1,
        prefill_ms_per_batch_item: 0.0,
        chunk_overhead_ms: 0.0,
        chunk_size_tokens: Some(8),
        decode_intercept_ms: 0.0,
        decode_ms_per_active_sequence: 1.0,
        decode_ms_per_context_token: 0.0,
        startup: DurationDistribution::Constant { value_ms: 5.0 },
    }
}

fn replica(id: &str, warm: bool) -> ReplicaSpec {
    ReplicaSpec {
        id: id.into(),
        initially_warm: warm,
        max_queue: 128,
        max_active_sequences: 8,
        service_rate_multiplier: 1.0,
        hourly_price_usd: 3.6,
        canary: false,
    }
}

fn request(id: &str, arrival_ms: u64, priority: u8) -> RequestSpec {
    RequestSpec {
        id: id.into(),
        arrival_ms,
        prompt_tokens: 10,
        output_tokens: 2,
        priority,
        request_class: "interactive".into(),
        deadline_ms: None,
        cancel_after_ms: None,
        canary_eligible: true,
        adapter_id: None,
        prefix_group: None,
    }
}

fn scenario(requests: Vec<RequestSpec>) -> SimulationRequest {
    SimulationRequest {
        schema_version: "1.0".into(),
        seed: 7,
        service_curve: curve(),
        replicas: vec![replica("r0", true)],
        requests,
        actions: Vec::new(),
        routing_policy: RoutingPolicy::RoundRobin,
        canary_weight: 0.0,
        max_events: 100_000,
    }
}

#[test]
fn known_single_server_timing_is_reproduced() {
    let output = simulate(&scenario(vec![request("q0", 0, 0)])).expect("valid simulation");
    let outcome = &output.outcomes[0];
    assert_eq!(outcome.status, OutcomeStatus::Completed);
    assert_eq!(outcome.ttft_ms, Some(2.0));
    assert_eq!(outcome.e2e_ms, Some(3.0));
    assert_eq!(outcome.queue_ms, Some(0.0));
    assert_eq!(outcome.prefill_ms, Some(1.0));
    assert_eq!(outcome.decode_ms, Some(1.0));
    assert_eq!(outcome.mean_itl_ms, Some(1.0));
    assert_eq!(output.metrics.p99_itl_ms, Some(1.0));
    assert_eq!(output.metrics.p95_queue_ms, Some(0.0));
    assert_eq!(output.metrics.p95_prefill_ms, Some(1.0));
    assert!((output.metrics.cost_usd - 0.000_003).abs() < 1e-12);
}

#[test]
fn completed_request_deadline_timer_does_not_inflate_cost_horizon() {
    let mut timed = request("q0", 0, 0);
    timed.deadline_ms = Some(10_000);
    let output = simulate(&scenario(vec![timed])).expect("valid simulation");
    assert!((output.metrics.simulated_duration_ms - 3.0).abs() < f64::EPSILON);
    assert!((output.metrics.cost_usd - 0.000_003).abs() < 1e-12);
}

#[test]
fn same_seed_is_byte_for_byte_deterministic() {
    let mut input = scenario(
        (0..30)
            .map(|idx| request(&format!("q{idx}"), idx / 3, 0))
            .collect(),
    );
    input.service_curve.startup = DurationDistribution::Uniform {
        min_ms: 2.0,
        max_ms: 8.0,
    };
    input.replicas[0].initially_warm = false;
    input.canary_weight = 0.5;
    input.replicas.push(ReplicaSpec {
        canary: true,
        ..replica("canary", false)
    });
    let left = serde_json::to_vec(&simulate(&input).expect("first run")).expect("json");
    let right = serde_json::to_vec(&simulate(&input).expect("second run")).expect("json");
    assert_eq!(left, right);
}

#[test]
fn priority_queue_dispatches_high_priority_first_after_cold_start() {
    let mut input = scenario(vec![request("low", 0, 1), request("high", 1, 9)]);
    input.replicas[0].initially_warm = false;
    let output = simulate(&input).expect("simulation");
    let first_prefill = output
        .trace_events
        .iter()
        .find(|event| event.name == "prefill")
        .expect("prefill event");
    assert_eq!(first_prefill.args["request_id"], "high");
}

#[test]
fn deadline_cancellation_and_queue_rejection_are_terminal() {
    let mut deadline = request("deadline", 0, 0);
    deadline.deadline_ms = Some(1);
    let mut cancelled = request("cancel", 0, 0);
    cancelled.cancel_after_ms = Some(1);
    let mut input = scenario(vec![deadline, cancelled, request("rejected", 0, 0)]);
    input.replicas[0].initially_warm = false;
    input.replicas[0].max_queue = 2;
    let output = simulate(&input).expect("simulation");
    assert_eq!(output.metrics.request_count, 3);
    assert!(
        output
            .outcomes
            .iter()
            .any(|o| o.status == OutcomeStatus::DeadlineExceeded)
    );
    assert!(
        output
            .outcomes
            .iter()
            .any(|o| o.status == OutcomeStatus::Cancelled)
    );
    assert!(
        output
            .outcomes
            .iter()
            .any(|o| o.status == OutcomeStatus::Rejected)
    );
}

#[test]
fn crash_fails_active_work_and_reroutes_queued_work() {
    let mut input = scenario(vec![request("active", 0, 1), request("queued", 0, 1)]);
    input.replicas.push(replica("r1", true));
    input.routing_policy = RoutingPolicy::LeastOutstanding;
    input.actions.push(TimedAction {
        at_ms: 2,
        action: ScenarioAction::BackendCrash {
            replica_id: "r0".into(),
        },
    });
    let output = simulate(&input).expect("simulation");
    assert!(output.metrics.failed_count >= 1);
    assert!(
        output
            .trace_events
            .iter()
            .any(|event| event.name == "scenario_action")
    );
}

#[test]
fn add_remove_slowdown_and_canary_actions_execute() {
    let mut input = scenario(
        (0..12)
            .map(|idx| request(&format!("q{idx}"), idx, 1))
            .collect(),
    );
    input.actions = vec![
        TimedAction {
            at_ms: 1,
            action: ScenarioAction::AddReplica {
                replica: ReplicaSpec {
                    canary: true,
                    ..replica("r1", false)
                },
            },
        },
        TimedAction {
            at_ms: 2,
            action: ScenarioAction::BackendSlowdown {
                replica_id: "r0".into(),
                factor: 4.0,
            },
        },
        TimedAction {
            at_ms: 3,
            action: ScenarioAction::RemoveReplica {
                replica_id: "r0".into(),
            },
        },
    ];
    input.canary_weight = 1.0;
    let output = simulate(&input).expect("simulation");
    assert!(
        output
            .outcomes
            .iter()
            .any(|outcome| outcome.replica_id.as_deref() == Some("r1"))
    );
}

#[test]
fn finish_time_routing_selects_the_faster_measured_replica() {
    for policy in [RoutingPolicy::EarliestFinish, RoutingPolicy::SloSlackAware] {
        let mut input = scenario(vec![request("routed", 0, 1)]);
        input.routing_policy = policy;
        input.replicas.push(ReplicaSpec {
            service_rate_multiplier: 2.0,
            ..replica("fast", true)
        });
        let output = simulate(&input).expect("simulation");
        assert_eq!(output.outcomes[0].replica_id.as_deref(), Some("fast"));
    }
}

#[test]
fn least_outstanding_balances_simultaneous_arrivals() {
    let mut input = scenario(vec![request("q0", 0, 1), request("q1", 0, 1)]);
    input.replicas.push(replica("r1", true));
    input.routing_policy = RoutingPolicy::LeastOutstanding;
    let output = simulate(&input).expect("simulation");
    let assigned: std::collections::HashSet<_> = output
        .outcomes
        .iter()
        .filter_map(|outcome| outcome.replica_id.as_deref())
        .collect();
    assert_eq!(assigned.len(), 2);
}

#[test]
fn discrete_event_execution_is_at_least_ten_times_faster_than_trace_time() {
    let requests = (0..2_000)
        .map(|idx| {
            let mut item = request(&format!("q{idx}"), idx * 50, 1);
            item.prompt_tokens = 16;
            item
        })
        .collect();
    let input = scenario(requests);
    let started = std::time::Instant::now();
    let output = simulate(&input).expect("simulation");
    let wall_ms = started.elapsed().as_secs_f64() * 1_000.0;
    assert!(wall_ms * 10.0 < output.metrics.simulated_duration_ms);
}

proptest! {
    #[test]
    fn completed_and_failed_counts_never_exceed_total(
        arrivals in prop::collection::vec(0_u64..100, 0..100),
        seed in any::<u64>(),
    ) {
        let requests = arrivals
            .iter()
            .enumerate()
            .map(|(idx, arrival)| request(&format!("q{idx}"), *arrival, [0, 1, 2][idx % 3]))
            .collect();
        let mut input = scenario(requests);
        input.seed = seed;
        let output = simulate(&input).expect("valid generated scenario");
        prop_assert!(output.metrics.completed_count + output.metrics.failed_count + output.metrics.rejected_count <= output.metrics.request_count);
        prop_assert!(output.metrics.availability >= 0.0 && output.metrics.availability <= 1.0);
        prop_assert!(output.metrics.cost_usd >= 0.0);
    }
}
