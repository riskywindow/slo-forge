from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from sloforge.fabric.topology import (
    FactState,
    HealthState,
    NodeKind,
    build_canonical_fixture,
    build_discovery_fixture,
    discover_topology,
    discover_topology_records,
    load_topology,
    save_topology,
)
from sloforge.fabric.topology import discovery as discovery_module
from sloforge.fabric.topology.discovery import observed
from sloforge.fabric.topology.fixtures import FIXTURE_TIME
from sloforge.fabric.topology.models import DiscoveryTopologyGraph, Provenance

FIXTURES = Path(__file__).parents[1] / "fixtures" / "topologies"


@pytest.mark.parametrize(
    "name",
    [
        "single_gpu_workstation",
        "eight_gpu_nvlink",
        "multi_numa_pcie",
        "two_node_infiniband",
        "roce_cluster",
        "mig_host",
        "degraded_topology",
        "limited_container",
        "conflicting_sources",
    ],
)
def test_topology_fixtures_are_deterministic_and_valid(name: str) -> None:
    graph = build_discovery_fixture(FIXTURES / f"{name}.json")
    repeated = build_discovery_fixture(name)
    assert graph == repeated
    assert graph.fingerprint == repeated.fingerprint
    assert graph.captured_at == FIXTURE_TIME
    assert graph.nodes
    assert all(edge.source and edge.target for edge in graph.edges)


def test_expected_fixture_shapes_and_capabilities() -> None:
    workstation = build_discovery_fixture("single_gpu_workstation")
    nvlink = build_discovery_fixture("eight_gpu_nvlink")
    multi_node = build_discovery_fixture("two_node_infiniband")
    roce = build_discovery_fixture("roce_cluster")
    mig = build_discovery_fixture("mig_host")
    assert sum(node.kind is NodeKind.GPU for node in workstation.nodes) == 1
    assert sum(node.kind is NodeKind.GPU for node in nvlink.nodes) == 8
    assert {node.host_id for node in multi_node.nodes if node.kind is NodeKind.GPU} == {
        "host-0",
        "host-1",
    }
    assert any(
        fact.name == "roce_capable" and fact.value is True
        for node in roce.nodes
        if node.kind is NodeKind.NIC
        for fact in node.facts
    )
    assert all(
        node.fact("mig_instances") is not None and node.fact("mig_instances").value == 7  # type: ignore[union-attr]
        for node in mig.nodes
        if node.kind is NodeKind.GPU
    )
    pcie = build_discovery_fixture("multi_numa_pcie")
    assert sum(node.kind is NodeKind.PCIE_ROOT for node in pcie.nodes) == 4
    assert sum(node.kind is NodeKind.PCIE_SWITCH for node in pcie.nodes) == 4
    assert any(edge.kind.value == "pcie" and edge.contention_domain for edge in pcie.edges)


def test_degraded_container_and_conflict_fixtures_preserve_epistemic_state() -> None:
    degraded = build_discovery_fixture("degraded_topology")
    limited = build_discovery_fixture("limited_container")
    conflict = build_discovery_fixture("conflicting_sources")
    assert any(edge.health is HealthState.DEGRADED for edge in degraded.edges)
    assert limited.visibility.in_container
    assert limited.visibility.host_devices_visible is False
    assert len(limited.visibility.visible_gpu_ids) == 1
    conflicts = [
        fact for edge in conflict.edges for fact in edge.facts if fact.state is FactState.CONFLICT
    ]
    assert len(conflicts) == 1
    assert conflicts[0].value is None
    assert len(conflicts[0].observations) == 2


def test_observed_records_disagreement_instead_of_choosing_source() -> None:
    left = Provenance(source="sysfs", source_kind="sysfs", captured_at=FIXTURE_TIME, confidence=0.9)
    right = Provenance(
        source="command", source_kind="command", captured_at=FIXTURE_TIME, confidence=0.8
    )
    fact = observed("pcie_width", [(16, left), (8, right)])
    assert fact.state is FactState.CONFLICT
    assert fact.value is None


def test_current_host_discovery_has_provenance_and_explicit_unknowns(tmp_path: Path) -> None:
    graph = discover_topology_records(topology_id="test-current-host")
    assert graph.nodes[0].kind is NodeKind.HOST
    assert all(fact.observations for node in graph.nodes for fact in node.facts)
    assert all(
        observation.provenance.source
        for node in graph.nodes
        for fact in node.facts
        for observation in fact.observations
    )
    assert graph.visibility.facts
    canonical = discover_topology(topology_id="test-current-host-canonical")
    destination = tmp_path / "topology.json"
    save_topology(destination, canonical)
    assert load_topology(destination) == canonical


def test_topology_hash_rejects_tampering() -> None:
    graph = build_discovery_fixture("single_gpu_workstation")
    payload = graph.model_dump(mode="json")
    payload["topology_id"] = "tampered"
    with pytest.raises(ValidationError, match="fingerprint mismatch"):
        DiscoveryTopologyGraph.model_validate_json(json.dumps(payload))


def test_fixture_descriptors_reject_unknown_fields(tmp_path: Path) -> None:
    payload = json.loads((FIXTURES / "single_gpu_workstation.json").read_text(encoding="utf-8"))
    payload["invented_capability"] = True
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValidationError):
        build_discovery_fixture(path)


def test_nvidia_topology_matrix_normalizes_nvlink_and_pcie(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = """        GPU0 GPU1 GPU2 CPU Affinity\nGPU0   X    NV4  PHB  0-31\nGPU1   NV4  X    PIX  0-31\nGPU2   PHB  PIX  X    32-63\n"""

    def fake_run(argv: tuple[str, ...], timeout_s: float = 5.0) -> subprocess.CompletedProcess[str]:
        del argv, timeout_s
        return subprocess.CompletedProcess(["nvidia-smi"], 0, output, "")

    monkeypatch.setattr(discovery_module, "_run", fake_run)
    rows = tuple(
        (
            str(index),
            f"GPU-{index}",
            "Synthetic GPU",
            "81920",
            f"0000:0{index}:00.0",
            "600.0",
            "9.0",
            "Disabled",
            "0",
            "1500",
            "250.0",
        )
        for index in range(3)
    )
    edges = discovery_module._gpu_topology_edges("host", rows, FIXTURE_TIME)
    assert len(edges) == 3
    assert edges[0].fact("connection_type") is not None
    assert {
        edge.fact("connection_type").value for edge in edges if edge.fact("connection_type")
    } == {
        "nvlink",
        "pcie",
    }


@pytest.mark.parametrize(
    "name", ["single_gpu_workstation", "eight_gpu_nvlink", "two_node_infiniband", "roce_cluster"]
)
def test_fixture_conversion_produces_canonical_fabric_ir(name: str) -> None:
    graph = build_canonical_fixture(name)
    assert graph.kind == "TopologyGraph"
    assert graph.schema_version == "1.0.0"
    assert graph.nodes
    assert all(edge.source_node_id != edge.target_node_id for edge in graph.edges)
    assert graph.extensions.root["sloforge.io/discovery-fingerprint"]
