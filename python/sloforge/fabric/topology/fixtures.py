"""Deterministic synthetic topology fixtures used by CPU-only validation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from sloforge.fabric.ir import TopologyGraph as CanonicalTopologyGraph
from sloforge.fabric.topology.models import (
    DiscoveryTopologyGraph,
    EdgeKind,
    FactState,
    HealthState,
    NodeKind,
    Observation,
    ObservedFact,
    Provenance,
    SoftwareComponent,
    TopologyEdge,
    TopologyNode,
    Visibility,
    finalize_graph,
)

FIXTURE_TIME = "2026-01-01T00:00:00+00:00"


class FixtureSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["sloforge.fabric.fixture/v1"]
    name: str
    hosts: int = Field(ge=1)
    gpus_per_host: int = Field(ge=0)
    numa_per_host: int = Field(ge=1)
    network: Literal["ethernet", "infiniband", "roce"]
    rails_per_host: int = Field(ge=1)
    nvlink_group_size: int = Field(ge=0)
    mig_instances_per_gpu: int = Field(ge=0)
    degraded_edge: str | None = None
    container_limited: bool = False
    conflicting_sources: bool = False


def load_fixture_spec(path: Path) -> FixtureSpec:
    return FixtureSpec.model_validate_json(path.read_text(encoding="utf-8"))


def _prov(source: str, *, confidence: float = 1.0) -> Provenance:
    return Provenance(
        source=source,
        source_kind="fixture",
        captured_at=FIXTURE_TIME,
        confidence=confidence,
    )


def _fact(
    name: str,
    value: str | int | float | bool | None,
    *,
    unit: str | None = None,
    source: str = "fixture-ground-truth",
) -> ObservedFact:
    return ObservedFact(
        name=name,
        unit=unit,
        state=FactState.UNKNOWN if value is None else FactState.KNOWN,
        value=value,
        observations=(Observation(value=value, provenance=_prov(source)),),
    )


def _conflicting_fact(name: str, left: int, right: int, *, unit: str) -> ObservedFact:
    return ObservedFact(
        name=name,
        unit=unit,
        state=FactState.CONFLICT,
        observations=(
            Observation(value=left, provenance=_prov("fixture-sysfs", confidence=0.95)),
            Observation(value=right, provenance=_prov("fixture-command", confidence=0.85)),
        ),
    )


def build_discovery_fixture(spec_or_name: FixtureSpec | str | Path) -> DiscoveryTopologyGraph:
    """Build a canonical graph from a checked-in compact fixture descriptor."""
    if isinstance(spec_or_name, FixtureSpec):
        spec = spec_or_name
    else:
        candidate = Path(spec_or_name)
        if candidate.is_file():
            spec = load_fixture_spec(candidate)
        else:
            fixture_root = Path(__file__).parents[4] / "tests" / "fixtures" / "topologies"
            spec = load_fixture_spec(fixture_root / f"{spec_or_name}.json")

    nodes: list[TopologyNode] = []
    edges: list[TopologyEdge] = []
    visible_gpu_ids: list[str] = []
    for host_index in range(spec.hosts):
        host = f"host-{host_index}"
        nodes.append(
            TopologyNode(
                node_id=host,
                kind=NodeKind.HOST,
                host_id=host,
                health=HealthState.HEALTHY,
                facts=(
                    _fact("hostname", host),
                    _fact("architecture", "x86_64"),
                    _fact("operating_system", "synthetic-linux"),
                    _fact("logical_cpu_count", 64, unit="count"),
                    _fact("memory_capacity", 512 * 1024**3, unit="bytes"),
                    _fact("visible_memory_capacity", 512 * 1024**3, unit="bytes"),
                ),
            )
        )
        for numa_index in range(spec.numa_per_host):
            socket_id = f"{host}/socket/{numa_index}"
            numa_id = f"{host}/numa/{numa_index}"
            nodes.extend(
                (
                    TopologyNode(
                        node_id=socket_id,
                        kind=NodeKind.CPU_SOCKET,
                        host_id=host,
                        health=HealthState.HEALTHY,
                        facts=(
                            _fact("socket_index", numa_index),
                            _fact("cpu_model", "Synthetic EPYC"),
                            _fact("physical_core_count", 16, unit="count"),
                            _fact("logical_cpu_count", 32, unit="count"),
                        ),
                    ),
                    TopologyNode(
                        node_id=numa_id,
                        kind=NodeKind.NUMA_DOMAIN,
                        host_id=host,
                        health=HealthState.HEALTHY,
                        facts=(
                            _fact("numa_index", numa_index),
                            _fact("cpu_set", f"{numa_index * 32}-{numa_index * 32 + 31}"),
                            _fact(
                                "memory_capacity",
                                (512 * 1024**3) // spec.numa_per_host,
                                unit="bytes",
                            ),
                        ),
                    ),
                )
            )
            edges.extend(
                (
                    TopologyEdge(
                        edge_id=f"contains:{host}:{socket_id}",
                        source=host,
                        target=socket_id,
                        kind=EdgeKind.CONTAINS,
                        directed=True,
                        health=HealthState.HEALTHY,
                    ),
                    TopologyEdge(
                        edge_id=f"cpu-memory:{socket_id}:{numa_id}",
                        source=socket_id,
                        target=numa_id,
                        kind=EdgeKind.CPU_MEMORY,
                        directed=False,
                        full_duplex=True,
                        sharing_group=f"{host}-numa-{numa_index}",
                        contention_domain=f"{host}-numa-memory-{numa_index}",
                        health=HealthState.HEALTHY,
                        facts=(
                            _fact("measured_bandwidth", 170_000_000_000, unit="bytes_per_second"),
                            _fact("latency", 0.09, unit="microseconds"),
                        ),
                    ),
                )
            )

        for gpu_index in range(spec.gpus_per_host):
            gpu_id = f"{host}/gpu/{gpu_index}"
            visible_gpu_ids.append(gpu_id)
            numa_index = gpu_index % spec.numa_per_host
            nodes.append(
                TopologyNode(
                    node_id=gpu_id,
                    kind=NodeKind.GPU,
                    host_id=host,
                    health=(
                        HealthState.DEGRADED
                        if spec.degraded_edge == "gpu" and gpu_index == spec.gpus_per_host - 1
                        else HealthState.HEALTHY
                    ),
                    facts=(
                        _fact("index", gpu_index),
                        _fact("uuid", f"GPU-{host_index:02d}-{gpu_index:02d}"),
                        _fact("product_name", "Synthetic H100 SXM"),
                        _fact("architecture", "Hopper"),
                        _fact("compute_capability", "9.0"),
                        _fact("pci_bus_id", f"0000:{numa_index + 1:02x}:{gpu_index:02x}.0"),
                        _fact("memory_capacity", 80 * 1024**3, unit="bytes"),
                        _fact("memory_bandwidth", 3_350_000_000_000, unit="bytes_per_second"),
                        _fact("mig_mode", spec.mig_instances_per_gpu > 0),
                        _fact("mig_instances", spec.mig_instances_per_gpu, unit="count"),
                    ),
                )
            )
            numa_id = f"{host}/numa/{numa_index}"
            bandwidth_fact = (
                _conflicting_fact(
                    "theoretical_bandwidth", 64_000_000_000, 32_000_000_000, unit="bytes_per_second"
                )
                if spec.conflicting_sources and gpu_index == 0
                else _fact("theoretical_bandwidth", 64_000_000_000, unit="bytes_per_second")
            )
            edges.append(
                TopologyEdge(
                    edge_id=f"cpu-gpu:{numa_id}:{gpu_id}",
                    source=numa_id,
                    target=gpu_id,
                    kind=EdgeKind.CPU_GPU,
                    directed=False,
                    full_duplex=True,
                    sharing_group=f"{host}-pcie-{numa_index}",
                    contention_domain=f"{host}-pcie-{numa_index}",
                    health=HealthState.HEALTHY,
                    facts=(
                        bandwidth_fact,
                        _fact("measured_bandwidth", 52_000_000_000, unit="bytes_per_second"),
                        _fact("latency", 2.1, unit="microseconds"),
                        _fact("pcie_generation", 5),
                        _fact("pcie_width", 16),
                    ),
                )
            )

        if spec.nvlink_group_size > 1:
            for left in range(spec.gpus_per_host):
                for right in range(left + 1, spec.gpus_per_host):
                    if left // spec.nvlink_group_size != right // spec.nvlink_group_size:
                        continue
                    left_id, right_id = f"{host}/gpu/{left}", f"{host}/gpu/{right}"
                    degraded = spec.degraded_edge == "nvlink" and left == 0 and right == 1
                    edges.append(
                        TopologyEdge(
                            edge_id=f"nvlink:{left_id}:{right_id}",
                            source=left_id,
                            target=right_id,
                            kind=EdgeKind.GPU_GPU,
                            directed=False,
                            full_duplex=True,
                            sharing_group=f"{host}-nvlink-{left // spec.nvlink_group_size}",
                            contention_domain=f"{host}-nvswitch-{left // spec.nvlink_group_size}",
                            health=HealthState.DEGRADED if degraded else HealthState.HEALTHY,
                            facts=(
                                _fact("connection_type", "nvlink"),
                                _fact(
                                    "theoretical_bandwidth",
                                    450_000_000_000,
                                    unit="bytes_per_second",
                                ),
                                _fact(
                                    "measured_bandwidth",
                                    90_000_000_000 if degraded else 405_000_000_000,
                                    unit="bytes_per_second",
                                ),
                                _fact("latency", 1.4 if degraded else 0.7, unit="microseconds"),
                                _fact("measurement_confidence", 0.96),
                            ),
                        )
                    )

        for rail_index in range(spec.rails_per_host):
            nic_id = f"{host}/nic/{rail_index}"
            rail_id = f"rail-{rail_index}"
            if not any(node.node_id == rail_id for node in nodes):
                nodes.append(
                    TopologyNode(
                        node_id=rail_id,
                        kind=NodeKind.NETWORK_RAIL,
                        host_id=host,
                        health=(
                            HealthState.DEGRADED
                            if spec.degraded_edge == "network" and rail_index == 0
                            else HealthState.HEALTHY
                        ),
                        facts=(
                            _fact("name", rail_id),
                            _fact("transport", spec.network),
                        ),
                    )
                )
            nodes.append(
                TopologyNode(
                    node_id=nic_id,
                    kind=NodeKind.NIC,
                    host_id=host,
                    health=HealthState.HEALTHY,
                    facts=(
                        _fact("interface_name", f"fabric{rail_index}"),
                        _fact("transport", spec.network),
                        _fact("link_speed", 400_000, unit="megabits_per_second"),
                        _fact("active_port", True),
                        _fact("rdma_capable", spec.network in {"infiniband", "roce"}),
                        _fact("gpudirect_rdma", spec.network in {"infiniband", "roce"}),
                        _fact("roce_capable", spec.network == "roce"),
                    ),
                )
            )
            edges.append(
                TopologyEdge(
                    edge_id=f"nic-network:{nic_id}:{rail_id}",
                    source=nic_id,
                    target=rail_id,
                    kind=EdgeKind.NIC_NETWORK,
                    directed=False,
                    full_duplex=True,
                    sharing_group=rail_id,
                    contention_domain=rail_id,
                    health=(
                        HealthState.DEGRADED
                        if spec.degraded_edge == "network" and rail_index == 0
                        else HealthState.HEALTHY
                    ),
                    facts=(
                        _fact("theoretical_bandwidth", 50_000_000_000, unit="bytes_per_second"),
                        _fact(
                            "measured_bandwidth",
                            12_000_000_000
                            if spec.degraded_edge == "network" and rail_index == 0
                            else 45_000_000_000,
                            unit="bytes_per_second",
                        ),
                        _fact(
                            "latency",
                            12.0 if spec.degraded_edge == "network" else 3.2,
                            unit="microseconds",
                        ),
                    ),
                )
            )
            for gpu_index in range(spec.gpus_per_host):
                if gpu_index % spec.rails_per_host != rail_index:
                    continue
                gpu_id = f"{host}/gpu/{gpu_index}"
                edges.append(
                    TopologyEdge(
                        edge_id=f"gpu-nic:{gpu_id}:{nic_id}",
                        source=gpu_id,
                        target=nic_id,
                        kind=EdgeKind.GPU_NIC,
                        directed=False,
                        full_duplex=True,
                        sharing_group=f"{host}-pcie-{gpu_index % spec.numa_per_host}",
                        contention_domain=f"{host}-pcie-{gpu_index % spec.numa_per_host}",
                        health=HealthState.HEALTHY,
                        facts=(
                            _fact("measured_bandwidth", 41_000_000_000, unit="bytes_per_second"),
                            _fact("latency", 2.5, unit="microseconds"),
                        ),
                    )
                )

    fixture_prov = _prov("fixture-visibility")
    restrictions = ("visible GPUs restricted to first device",) if spec.container_limited else ()
    visible = tuple(visible_gpu_ids[:1] if spec.container_limited else visible_gpu_ids)
    return finalize_graph(
        schema_version="sloforge.fabric.topology/v1",
        topology_id=spec.name,
        captured_at=FIXTURE_TIME,
        nodes=tuple(nodes),
        edges=tuple(edges),
        visibility=Visibility(
            in_container=spec.container_limited,
            host_devices_visible=not spec.container_limited,
            visible_gpu_ids=visible,
            restrictions=restrictions,
            facts=(
                ObservedFact(
                    name="fixture_visibility",
                    state=FactState.KNOWN,
                    value="container-limited" if spec.container_limited else "complete",
                    observations=(
                        Observation(
                            value="container-limited" if spec.container_limited else "complete",
                            provenance=fixture_prov,
                        ),
                    ),
                ),
            ),
        ),
        software=(
            SoftwareComponent(
                name="synthetic-fabric",
                version="1.0.0",
                state=FactState.KNOWN,
                provenance=_prov("fixture-generator"),
            ),
        ),
        warnings=(
            ("Host topology is intentionally hidden by the fixture container boundary.",)
            if spec.container_limited
            else ()
        ),
    )


def build_canonical_fixture(spec_or_name: FixtureSpec | str | Path) -> CanonicalTopologyGraph:
    """Build and finalize a fixture into the canonical Fabric IR."""
    from sloforge.fabric.topology.conversion import to_canonical_topology

    return to_canonical_topology(build_discovery_fixture(spec_or_name))


def build_fixture(spec_or_name: FixtureSpec | str | Path) -> CanonicalTopologyGraph:
    """Build a compiler-facing canonical fixture graph."""
    return build_canonical_fixture(spec_or_name)
