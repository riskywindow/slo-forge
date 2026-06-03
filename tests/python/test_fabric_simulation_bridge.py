from __future__ import annotations

from pathlib import Path

from sloforge.autopsy import BottleneckKind, compare_runs, diagnose
from sloforge.autopsy.capture import _counters as captured_counters
from sloforge.autopsy.capture import capture_simulation_run
from sloforge.autopsy.models import EventType
from sloforge.fabric.ir import (
    FabricProfile,
    PhysicalExecutionPlan,
    TopologyGraph,
    WorkerRole,
    canonical_hash,
    load_fabric_profile,
    load_physical_execution_plan,
    load_topology_graph,
)
from sloforge.fabric.simulation import (
    FabricSimulationRequest,
    OperationOutcome,
    RankSlowdownFault,
    RemoveFault,
    ResourceRateFault,
    SimulationWorkload,
    TimedFault,
    build_simulation_request,
    request_latencies,
    run_simulation,
)
from sloforge.util import write_json

ROOT = Path(__file__).parents[2]
FIXTURES = Path(__file__).parents[1] / "fixtures" / "fabric"


def _inputs() -> tuple[PhysicalExecutionPlan, TopologyGraph, FabricProfile]:
    return (
        load_physical_execution_plan(FIXTURES / "physical-execution-plan-v1.json"),
        load_topology_graph(FIXTURES / "topology-graph-v1.json"),
        load_fabric_profile(FIXTURES / "fabric-profile-v1.json"),
    )


def test_physical_plan_lowers_to_strict_rust_protocol() -> None:
    plan, topology, profile = _inputs()
    request = build_simulation_request(
        plan,
        topology,
        profile,
        SimulationWorkload(
            request_count=2,
            arrival_interval_us=100.0,
            prompt_tokens=128,
            output_tokens=2,
        ),
        seed=11,
    )
    assert request.operations
    assert request.resources
    assert any(operation.kind.type == "collective" for operation in request.operations)
    collective = next(
        operation for operation in request.operations if operation.kind.type == "collective"
    )
    assert tuple(demand.resource_id for demand in collective.demands) == ("nvlink-0-1",)
    assert sum(":prefill" in operation.id for operation in request.operations) == 2
    assert sum(operation.id.endswith(":decode") for operation in request.operations) == 2
    assert FabricSimulationRequest.model_validate_json(request.model_dump_json()) == request


def test_kv_transfer_retains_stage_and_emits_physical_network_counter() -> None:
    outcome = OperationOutcome(
        operation_id="request-0:kv-transfer",
        status="completed",
        start_us=0.0,
        end_us=1_000.0,
        duration_us=1_000.0,
        base_duration_us=1_000.0,
        wait_us=0.0,
        transferred_bytes=1_000_000,
        uncertainty_us=10.0,
        rank_ids=("rank-0", "rank-1"),
        resource_ids=("copy-engine:gpu-0", "nic-network:nic-0:rail-0"),
    )
    counters = captured_counters(outcome, EventType.KV_TRANSFER)
    values = {counter.name: counter.value for counter in counters}
    assert values["network_bandwidth_gbps"] == 8.0


def test_rust_subprocess_runs_and_metrics_are_event_derived() -> None:
    plan, topology, profile = _inputs()
    request = build_simulation_request(
        plan,
        topology,
        profile,
        SimulationWorkload(
            request_count=2,
            arrival_interval_us=1_000.0,
            prompt_tokens=128,
            output_tokens=2,
        ),
        seed=17,
    )
    output = run_simulation(request, repository_root=ROOT)
    latencies = request_latencies(output)
    assert output.metrics.operation_count == len(request.operations)
    assert output.metrics.makespan_us > 0.0
    assert len(latencies) == 2
    assert all(item.end_to_end_us >= item.ttft_us > 0.0 for item in latencies)
    assert output.provenance.input_sha256


def test_counterfactual_removes_labeled_link_fault() -> None:
    plan, topology, profile = _inputs()
    workload = SimulationWorkload(
        request_count=1,
        arrival_interval_us=0.0,
        prompt_tokens=512,
        output_tokens=1,
    )
    fault = TimedFault(
        id="degraded-nvlink",
        start_us=0.0,
        end_us=1_000_000.0,
        effect=ResourceRateFault(resource_id="nvlink-0-1", multiplier=0.25),
        ground_truth_label="nvlink_bandwidth_degradation",
    )
    degraded_request = build_simulation_request(
        plan,
        topology,
        profile,
        workload,
        seed=19,
        faults=(fault,),
    )
    repaired_request = degraded_request.model_copy(
        update={"counterfactuals": (RemoveFault(fault_id=fault.id),)}
    )
    degraded = run_simulation(degraded_request, repository_root=ROOT)
    repaired = run_simulation(repaired_request, repository_root=ROOT)
    assert degraded.applied_faults == (fault.id,)
    assert repaired.applied_faults == ()
    assert repaired.metrics.makespan_us < degraded.metrics.makespan_us


def test_simulation_capture_drives_rank_straggler_diagnosis(tmp_path: Path) -> None:
    plan, topology, profile = _inputs()
    plan = plan.model_copy(
        update={
            "rank_placement": plan.rank_placement.model_copy(
                update={
                    "bindings": tuple(
                        binding.model_copy(update={"worker_role": WorkerRole.AGGREGATED})
                        for binding in plan.rank_placement.bindings
                    )
                }
            )
        }
    )
    workload = SimulationWorkload(
        request_count=2,
        arrival_interval_us=100.0,
        prompt_tokens=256,
        output_tokens=2,
    )
    healthy_request = build_simulation_request(
        plan,
        topology,
        profile,
        workload,
        seed=23,
    )
    degraded_request = healthy_request.model_copy(
        update={
            "faults": (
                TimedFault(
                    id="rank-0-slow",
                    start_us=0.0,
                    end_us=1_000_000.0,
                    effect=RankSlowdownFault(rank_id="rank-0", multiplier=0.4),
                    ground_truth_label="rank_specific_gpu_slowdown",
                ),
            )
        }
    )
    healthy_output = run_simulation(healthy_request, repository_root=ROOT)
    degraded_output = run_simulation(degraded_request, repository_root=ROOT)
    healthy_path = tmp_path / "healthy.json"
    degraded_path = tmp_path / "degraded.json"
    write_json(healthy_path, healthy_output.model_dump(mode="json"))
    write_json(degraded_path, degraded_output.model_dump(mode="json"))
    topology_hash = canonical_hash(topology)
    healthy_run = capture_simulation_run(
        run_id="healthy",
        request=healthy_request,
        output=healthy_output,
        plan=plan,
        topology_fingerprint=topology_hash,
        workload_fingerprint="d" * 64,
        artifact_path=healthy_path,
    )
    degraded_run = capture_simulation_run(
        run_id="degraded",
        request=degraded_request,
        output=degraded_output,
        plan=plan,
        topology_fingerprint=topology_hash,
        workload_fingerprint="d" * 64,
        artifact_path=degraded_path,
    )
    assert degraded_run.fault_intervals[0].fault_type == "rank_specific_gpu_slowdown"
    comparison = compare_runs(healthy_run, degraded_run)
    diagnosis = diagnose(degraded_run, comparison=comparison, baseline=healthy_run)
    assert diagnosis.top_hypothesis is BottleneckKind.RANK_STRAGGLER
    assert comparison.maximum_rank_skew > 1.0
