from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import sloforge.fabric.simulation as simulation_module
from sloforge.autopsy import BottleneckKind, compare_runs, diagnose
from sloforge.autopsy.capture import _counters as captured_counters
from sloforge.autopsy.capture import capture_simulation_run
from sloforge.autopsy.models import EventType
from sloforge.fabric.ir import (
    CollectiveOperation,
    CollectivePlan,
    CommunicationOverlapPlan,
    FabricProfile,
    ParallelGroup,
    ParallelismKind,
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
    SimulationRequestShape,
    SimulationWorkload,
    TimedFault,
    _lower_collective_before_kv,
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


def _small_request() -> FabricSimulationRequest:
    plan, topology, profile = _inputs()
    return build_simulation_request(
        plan,
        topology,
        profile,
        SimulationWorkload(
            request_count=1,
            arrival_interval_us=0.0,
            prompt_tokens=16,
            output_tokens=1,
        ),
        seed=3,
    )


def test_simulator_boundary_times_out_and_caps_responses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _small_request()

    def timeout(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        del args, kwargs
        raise subprocess.TimeoutExpired(cmd=("sim",), timeout=0.1)

    monkeypatch.setattr(simulation_module.subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match=r"timed out after 0\.1s"):
        run_simulation(request, repository_root=ROOT, timeout_seconds=0.1)

    monkeypatch.setattr(simulation_module, "_MAX_SIMULATOR_OUTPUT_BYTES", 4)
    monkeypatch.setattr(
        simulation_module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=("sim",), returncode=0, stdout=b"12345", stderr=b""
        ),
    )
    with pytest.raises(RuntimeError, match="response exceeds"):
        run_simulation(request, repository_root=ROOT)


def test_simulator_boundary_truncates_failure_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(simulation_module, "_MAX_SIMULATOR_ERROR_BYTES", 8)
    monkeypatch.setattr(
        simulation_module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=("sim",), returncode=9, stdout=b"", stderr=b"0123456789abcdef"
        ),
    )
    with pytest.raises(RuntimeError, match=r"01234567\n\[stderr truncated\]"):
        run_simulation(_small_request(), repository_root=ROOT)


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


def test_collective_stage_filter_does_not_lower_decode_collectives_before_kv() -> None:
    assert _lower_collective_before_kv({0, 1}, {0, 1}, {2, 3})
    assert not _lower_collective_before_kv({2, 3}, {0, 1}, {2, 3})
    # Backward-compatible v1 plans could split a parallel group across P/D
    # roles; lower that legacy operation once instead of silently dropping it.
    assert _lower_collective_before_kv({1, 2}, {0, 1}, {2, 3})


def test_pipeline_stages_are_lowered_as_dependencies_not_impossible_overlap() -> None:
    plan, topology, profile = _inputs()
    plan = plan.model_copy(
        update={
            "parallelism": plan.parallelism.model_copy(
                update={
                    "tensor_parallel_degree": 1,
                    "pipeline_parallel_degree": 2,
                    "expert_parallel_degree": 1,
                    "prefill_decode_disaggregated": False,
                    "groups": (
                        ParallelGroup(
                            group_id="pp-0",
                            kind=ParallelismKind.PIPELINE,
                            rank_ids=(0, 1),
                        ),
                    ),
                }
            ),
            "rank_placement": plan.rank_placement.model_copy(
                update={
                    "bindings": tuple(
                        binding.model_copy(update={"worker_role": WorkerRole.AGGREGATED})
                        for binding in plan.rank_placement.bindings
                    )
                }
            ),
            "collectives": CollectivePlan(operations=()),
            "kv_transfer": None,
            "communication_overlap": CommunicationOverlapPlan(windows=()),
        }
    )
    request = build_simulation_request(
        plan,
        topology,
        profile,
        SimulationWorkload(
            request_count=1,
            arrival_interval_us=0.0,
            prompt_tokens=128,
            output_tokens=2,
        ),
        seed=13,
    )
    prefill_0 = next(
        operation for operation in request.operations if ":rank-0:prefill" in operation.id
    )
    prefill_1 = next(
        operation for operation in request.operations if ":rank-1:prefill" in operation.id
    )
    decode_0 = next(
        operation for operation in request.operations if operation.id.endswith("rank-0:decode")
    )
    decode_1 = next(
        operation for operation in request.operations if operation.id.endswith("rank-1:decode")
    )
    assert prefill_0.id in prefill_1.dependencies
    assert decode_0.id in decode_1.dependencies

    output = run_simulation(request, repository_root=ROOT)
    outcomes = {operation.operation_id: operation for operation in output.operations}
    assert outcomes[prefill_1.id].start_us >= outcomes[prefill_0.id].end_us
    assert outcomes[decode_1.id].start_us >= outcomes[decode_0.id].end_us


def test_expert_skew_scales_only_all_to_all_message_demand() -> None:
    plan, topology, profile = _inputs()
    existing = plan.collectives.operations[0]
    all_to_all = CollectiveOperation(
        operation_id="expert-all-to-all",
        operation="all_to_all",
        participating_ranks=existing.participating_ranks,
        message_size_intercept_bytes=existing.message_size_intercept_bytes,
        message_size_bytes_per_token=existing.message_size_bytes_per_token,
        algorithm="pairwise",
        transport=existing.transport,
        channel_count=existing.channel_count,
        rail_ids=existing.rail_ids,
        rank_order=existing.rank_order,
        expected_duration_us=existing.expected_duration_us,
        uncertainty_us=existing.uncertainty_us,
        fallback=existing.fallback,
    )
    plan = plan.model_copy(
        update={"collectives": CollectivePlan(operations=(existing, all_to_all))}
    )

    def lowered(skew: float) -> FabricSimulationRequest:
        return build_simulation_request(
            plan,
            topology,
            profile,
            SimulationWorkload(
                request_count=1,
                arrival_interval_us=0.0,
                prompt_tokens=128,
                output_tokens=2,
                requests=(
                    SimulationRequestShape(
                        arrival_us=0.0,
                        prompt_tokens=128,
                        output_tokens=2,
                        priority="normal",
                        request_class="expert",
                        expert_skew_factor=skew,
                    ),
                ),
            ),
            seed=11,
        )

    balanced = tuple(
        operation.kind.bytes
        for operation in lowered(1.0).operations
        if operation.kind.type == "collective"
    )
    skewed = tuple(
        operation.kind.bytes
        for operation in lowered(3.0).operations
        if operation.kind.type == "collective"
    )
    assert skewed == (balanced[0], balanced[1] * 3)


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
