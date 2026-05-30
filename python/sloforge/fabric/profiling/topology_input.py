"""Normalize canonical topology IR into the richer benchmark discovery view."""

from __future__ import annotations

from sloforge.fabric.ir import (
    ConnectionType,
    CpuSocketNode,
    GpuNode,
    HostNode,
    MemoryDomainNode,
    NetworkRailNode,
    NicNode,
    NumaDomainNode,
    NvSwitchNode,
    PcieNode,
    RemoteMemoryNode,
    StorageTierNode,
    canonical_hash,
)
from sloforge.fabric.ir import (
    HealthState as CanonicalHealth,
)
from sloforge.fabric.ir import (
    TopologyGraph as CanonicalTopologyGraph,
)
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


def _provenance(captured_at: str, source: str = "canonical-fabric-ir") -> Provenance:
    return Provenance(
        source=source,
        source_kind="derived",
        captured_at=captured_at,
        confidence=1.0,
    )


def _fact(
    name: str,
    value: str | int | float | bool | None,
    captured_at: str,
    *,
    unit: str | None = None,
) -> ObservedFact:
    return ObservedFact(
        name=name,
        unit=unit,
        state=FactState.UNKNOWN if value is None else FactState.KNOWN,
        value=value,
        observations=(Observation(value=value, provenance=_provenance(captured_at)),),
    )


def _health(health: CanonicalHealth) -> HealthState:
    return {
        CanonicalHealth.HEALTHY: HealthState.HEALTHY,
        CanonicalHealth.DEGRADED: HealthState.DEGRADED,
        CanonicalHealth.FAILED: HealthState.UNHEALTHY,
        CanonicalHealth.UNKNOWN: HealthState.UNKNOWN,
    }[health]


def _node(node: object, captured_at: str) -> TopologyNode:
    if isinstance(node, HostNode):
        return TopologyNode(
            node_id=node.node_id,
            kind=NodeKind.HOST,
            host_id=node.node_id,
            health=HealthState.HEALTHY,
            facts=(
                _fact("hostname", node.name, captured_at),
                _fact("architecture", node.architecture, captured_at),
                _fact("operating_system", node.operating_system, captured_at),
                _fact("memory_capacity", node.total_memory_bytes, captured_at, unit="bytes"),
                _fact(
                    "visible_memory_capacity", node.visible_memory_bytes, captured_at, unit="bytes"
                ),
            ),
        )
    if isinstance(node, CpuSocketNode):
        return TopologyNode(
            node_id=node.node_id,
            kind=NodeKind.CPU_SOCKET,
            host_id=node.host_id,
            health=HealthState.HEALTHY,
            facts=(
                _fact("socket_index", node.socket_index, captured_at),
                _fact("cpu_model", node.model, captured_at),
                _fact("physical_core_count", node.physical_cores, captured_at, unit="count"),
                _fact("logical_cpu_count", node.logical_cores, captured_at, unit="count"),
            ),
        )
    if isinstance(node, NumaDomainNode):
        return TopologyNode(
            node_id=node.node_id,
            kind=NodeKind.NUMA_DOMAIN,
            host_id=node.host_id,
            health=HealthState.HEALTHY,
            facts=(
                _fact("numa_index", node.numa_index, captured_at),
                _fact("cpu_set", node.cpu_set, captured_at),
                _fact("memory_capacity", node.memory_bytes, captured_at, unit="bytes"),
            ),
        )
    if isinstance(node, MemoryDomainNode):
        return TopologyNode(
            node_id=node.node_id,
            kind=NodeKind.MEMORY_DOMAIN,
            host_id=node.host_id,
            health=HealthState.HEALTHY,
            facts=(
                _fact("memory_capacity", node.capacity_bytes, captured_at, unit="bytes"),
                _fact(
                    "measured_bandwidth",
                    node.measured_bandwidth_gbps * 1e9 / 8
                    if node.measured_bandwidth_gbps
                    else None,
                    captured_at,
                    unit="bytes_per_second",
                ),
            ),
        )
    if isinstance(node, GpuNode):
        return TopologyNode(
            node_id=node.node_id,
            kind=NodeKind.GPU,
            host_id=node.host_id,
            health=_health(node.health),
            facts=(
                _fact("index", node.gpu_index, captured_at),
                _fact("uuid", node.uuid, captured_at),
                _fact("product_name", node.product, captured_at),
                _fact("architecture", node.architecture, captured_at),
                _fact("memory_capacity", node.memory_bytes, captured_at, unit="bytes"),
                _fact("memory_bandwidth", None, captured_at, unit="bytes_per_second"),
                _fact("compute_capability", node.compute_capability, captured_at),
                _fact("pci_bus_id", node.pci_address, captured_at),
                _fact("mig_mode", node.mig_state.value not in {"disabled", "unknown"}, captured_at),
            ),
        )
    if isinstance(node, NvSwitchNode):
        return TopologyNode(
            node_id=node.node_id,
            kind=NodeKind.NVSWITCH,
            host_id=node.host_id,
            health=_health(node.health),
            facts=(
                _fact("switch_domain", node.switch_domain, captured_at),
                _fact("generation", node.generation, captured_at),
            ),
        )
    if isinstance(node, PcieNode):
        return TopologyNode(
            node_id=node.node_id,
            kind=NodeKind.PCIE_ROOT if node.kind == "pcie_root_complex" else NodeKind.PCIE_SWITCH,
            host_id=node.host_id,
            health=HealthState.UNKNOWN,
            facts=(
                _fact("pci_bus_id", node.pci_address, captured_at),
                _fact("pcie_generation", node.generation, captured_at),
                _fact("pcie_width", node.width, captured_at),
            ),
        )
    if isinstance(node, NicNode):
        return TopologyNode(
            node_id=node.node_id,
            kind=NodeKind.NIC,
            host_id=node.host_id,
            health=_health(node.health),
            facts=(
                _fact("interface_name", node.interface, captured_at),
                _fact("pci_bus_id", node.pci_address, captured_at),
                _fact(
                    "link_speed",
                    node.speed_gbps * 1_000 if node.speed_gbps else None,
                    captured_at,
                    unit="megabits_per_second",
                ),
                _fact("transport", node.transport, captured_at),
                _fact("active_port", node.active, captured_at),
                _fact("rdma_capable", node.rdma_capable, captured_at),
                _fact("gpudirect_rdma", node.gpu_direct_rdma, captured_at),
            ),
        )
    if isinstance(node, NetworkRailNode):
        return TopologyNode(
            node_id=node.node_id,
            kind=NodeKind.NETWORK_RAIL,
            host_id="fabric",
            health=_health(node.health),
            facts=(
                _fact("name", node.name, captured_at),
                _fact("transport", node.transport, captured_at),
                _fact("subnet", node.subnet, captured_at),
            ),
        )
    if isinstance(node, StorageTierNode):
        return TopologyNode(
            node_id=node.node_id,
            kind=NodeKind.STORAGE_TIER,
            host_id=node.host_id or "fabric",
            health=HealthState.UNKNOWN,
            facts=(
                _fact("tier", node.tier, captured_at),
                _fact("capacity", node.capacity_bytes, captured_at, unit="bytes"),
            ),
        )
    if isinstance(node, RemoteMemoryNode):
        return TopologyNode(
            node_id=node.node_id,
            kind=NodeKind.REMOTE_MEMORY,
            host_id=node.host_id or "fabric",
            health=HealthState.UNKNOWN,
            facts=(
                _fact("capacity", node.capacity_bytes, captured_at, unit="bytes"),
                _fact("protocol", node.protocol, captured_at),
            ),
        )
    raise TypeError(f"unsupported canonical topology node: {type(node).__name__}")


def _edge_kind(connection: ConnectionType) -> EdgeKind:
    return {
        ConnectionType.CPU_MEMORY: EdgeKind.CPU_MEMORY,
        ConnectionType.CPU_GPU: EdgeKind.CPU_GPU,
        ConnectionType.GPU_GPU: EdgeKind.GPU_GPU,
        ConnectionType.NVLINK: EdgeKind.GPU_GPU,
        ConnectionType.NVSWITCH: EdgeKind.GPU_GPU,
        ConnectionType.PCIE: EdgeKind.PCIE,
        ConnectionType.GPU_NIC: EdgeKind.GPU_NIC,
        ConnectionType.NIC_NETWORK: EdgeKind.NIC_NETWORK,
        ConnectionType.STORAGE_HOST: EdgeKind.STORAGE_HOST,
        ConnectionType.REMOTE_MEMORY: EdgeKind.REMOTE_MEMORY,
    }[connection]


def _edge(edge: object, captured_at: str) -> TopologyEdge:
    from sloforge.fabric.ir import TopologyEdge as CanonicalEdge

    if not isinstance(edge, CanonicalEdge):
        raise TypeError(f"unsupported canonical topology edge: {type(edge).__name__}")
    bandwidth = edge.bandwidth_curve_gbps[0].median * 1e9 / 8 if edge.bandwidth_curve_gbps else None
    latency = edge.latency_curve_us[0].median if edge.latency_curve_us else None
    return TopologyEdge(
        edge_id=edge.edge_id,
        source=edge.source_node_id,
        target=edge.target_node_id,
        kind=_edge_kind(edge.connection),
        directed=edge.directionality == "unidirectional",
        full_duplex=True if edge.duplex == "full" else False if edge.duplex == "half" else None,
        sharing_group=edge.sharing_group,
        contention_domain=edge.contention_domain,
        health=_health(edge.health),
        facts=(
            _fact(
                "connection_type",
                edge.connection.value,
                captured_at,
            ),
            _fact(
                "theoretical_bandwidth",
                edge.theoretical_bandwidth_gbps * 1e9 / 8
                if edge.theoretical_bandwidth_gbps
                else None,
                captured_at,
                unit="bytes_per_second",
            ),
            _fact("measured_bandwidth", bandwidth, captured_at, unit="bytes_per_second"),
            _fact("latency", latency, captured_at, unit="microseconds"),
            _fact("measurement_confidence", edge.measurement_confidence, captured_at),
        ),
    )


def normalize_benchmark_topology(
    graph: DiscoveryTopologyGraph | CanonicalTopologyGraph,
) -> tuple[DiscoveryTopologyGraph, str]:
    """Return a benchmark view and the input graph's stable identity."""
    if isinstance(graph, DiscoveryTopologyGraph):
        return graph, graph.fingerprint
    captured_at = graph.discovered_at.isoformat()
    visible_extension = graph.extensions.root.get("sloforge.io/visible-gpu-ids")
    visible_ids = (
        tuple(str(item) for item in visible_extension)
        if isinstance(visible_extension, list)
        else tuple(node.node_id for node in graph.nodes if isinstance(node, GpuNode))
    )
    raw = finalize_graph(
        schema_version="sloforge.fabric.topology/v1",
        topology_id=graph.topology_id,
        captured_at=captured_at,
        nodes=tuple(_node(node, captured_at) for node in graph.nodes),
        edges=tuple(_edge(edge, captured_at) for edge in graph.edges),
        visibility=Visibility(
            in_container=graph.container_limited,
            host_devices_visible=not graph.container_limited,
            visible_gpu_ids=visible_ids,
            restrictions=graph.discovery_warnings if graph.container_limited else (),
            facts=(
                _fact("canonical_input", True, captured_at),
                _fact("host_device_inventory_complete", not graph.container_limited, captured_at),
            ),
        ),
        software=tuple(
            SoftwareComponent(
                name=item.name,
                version=item.version,
                state=FactState.KNOWN,
                provenance=_provenance(captured_at, item.source.value),
            )
            for item in graph.software
        ),
        warnings=graph.discovery_warnings,
    )
    return raw, canonical_hash(graph)
