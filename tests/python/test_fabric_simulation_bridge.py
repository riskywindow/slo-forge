from __future__ import annotations

from pathlib import Path

from sloforge.fabric.ir import (
    FabricProfile,
    PhysicalExecutionPlan,
    TopologyGraph,
    load_fabric_profile,
    load_physical_execution_plan,
    load_topology_graph,
)
from sloforge.fabric.simulation import (
    FabricSimulationRequest,
    RemoveFault,
    ResourceRateFault,
    SimulationWorkload,
    TimedFault,
    build_simulation_request,
    request_latencies,
    run_simulation,
)

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
    assert FabricSimulationRequest.model_validate_json(request.model_dump_json()) == request


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
