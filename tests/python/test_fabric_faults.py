from __future__ import annotations

from pathlib import Path

import pytest

from sloforge.fabric.faults import (
    PhysicalFaultScenario,
    PhysicalFaultSpec,
    PhysicalFaultType,
    bind_physical_faults,
    load_physical_fault_scenario,
)
from sloforge.fabric.ir import (
    load_fabric_profile,
    load_physical_execution_plan,
    load_topology_graph,
)
from sloforge.fabric.simulation import (
    FabricSimulationRequest,
    SimulationWorkload,
    build_simulation_request,
)

ROOT = Path(__file__).parents[2]
FIXTURES = Path(__file__).parents[1] / "fixtures" / "fabric"


def _request() -> FabricSimulationRequest:
    return build_simulation_request(
        load_physical_execution_plan(FIXTURES / "physical-execution-plan-v1.json"),
        load_topology_graph(FIXTURES / "topology-graph-v1.json"),
        load_fabric_profile(FIXTURES / "fabric-profile-v1.json"),
        SimulationWorkload(
            request_count=1,
            arrival_interval_us=0.0,
            prompt_tokens=128,
            output_tokens=1,
        ),
        seed=0,
    )


def test_every_required_physical_fault_type_has_a_typed_mechanism() -> None:
    request = _request()
    resource = request.resources[0].id
    rank = request.operations[0].rank_ids[0]
    collective = next(
        operation.kind.collective_id
        for operation in request.operations
        if operation.kind.type == "collective"
    )
    specifications = []
    for index, kind in enumerate(PhysicalFaultType):
        target = (
            rank
            if kind
            in {
                PhysicalFaultType.RANK_SPECIFIC_SLOWDOWN,
                PhysicalFaultType.RANK_SPECIFIC_GPU_SLOWDOWN,
                PhysicalFaultType.EXPERT_LOAD_SKEW,
                PhysicalFaultType.HOT_EXPERT_CONCENTRATION,
            }
            else collective
            if kind
            in {
                PhysicalFaultType.COLLECTIVE_DELAY,
                PhysicalFaultType.NCCL_INITIALIZATION_DELAY,
            }
            else resource
        )
        unavailable = kind in {
            PhysicalFaultType.GPU_PROCESS_CRASH,
            PhysicalFaultType.WORKER_CRASH,
            PhysicalFaultType.NETWORK_RAIL_LOSS,
            PhysicalFaultType.COLLECTIVE_FAILURE,
            PhysicalFaultType.PREFILL_WORKER_LOSS,
            PhysicalFaultType.DECODE_WORKER_LOSS,
            PhysicalFaultType.SIMULATED_OOM,
            PhysicalFaultType.MEMORY_FRAGMENTATION_REJECTION,
        }
        specifications.append(
            PhysicalFaultSpec(
                fault_id=f"fault-{index}",
                fault_type=kind,
                target_id=target,
                start_us=0.0,
                end_us=1_000.0,
                degradation_multiplier=None if unavailable else 0.5,
            )
        )
    scenario = PhysicalFaultScenario(
        schema_version="sloforge.fabric.faults/v1",
        scenario_id="catalog",
        faults=tuple(specifications),
    )
    bound = bind_physical_faults(scenario, request)
    assert len(bound) == len(PhysicalFaultType)
    assert {fault.ground_truth_label for fault in bound} == {
        kind.value for kind in PhysicalFaultType
    }


def test_loader_is_simulation_only_and_unknown_targets_fail() -> None:
    loaded = load_physical_fault_scenario(ROOT / "scenarios/fabric/dual-fault-demo.yaml")
    assert loaded.execution_mode == "simulation"
    with pytest.raises(ValueError, match="unknown resource"):
        bind_physical_faults(loaded, _request())


def test_loader_bounds_documents_and_rejects_yaml_references(tmp_path: Path) -> None:
    alias = tmp_path / "alias.yaml"
    alias.write_text(
        "schema_version: &version sloforge.fabric.faults/v1\n"
        "scenario_id: fixture\nexecution_mode: simulation\nfaults: *version\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="anchors or aliases"):
        load_physical_fault_scenario(alias)
    oversized = tmp_path / "oversized.yaml"
    oversized.write_bytes(b" " * (4 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match="exceeds 4 MiB"):
        load_physical_fault_scenario(oversized)
