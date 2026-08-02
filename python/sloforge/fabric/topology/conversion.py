"""Loss-aware conversion from raw discovery records to canonical Fabric IR."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Literal, cast

from sloforge.fabric.ir import (
    ConnectionType,
    CpuSocketNode,
    CurvePoint,
    DiscoverySource,
    FactProvenance,
    GpuNode,
    HostNode,
    MemoryDomainNode,
    MigState,
    NetworkRailNode,
    NicNode,
    NumaDomainNode,
    NvSwitchNode,
    PcieNode,
    RemoteMemoryNode,
    StorageTierNode,
)
from sloforge.fabric.ir import (
    HealthState as CanonicalHealth,
)
from sloforge.fabric.ir import (
    SoftwareComponent as CanonicalSoftwareComponent,
)
from sloforge.fabric.ir import (
    TopologyEdge as CanonicalEdge,
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
    ObservedFact,
    Provenance,
    TopologyEdge,
    TopologyNode,
)
from sloforge.ir import ArtifactDigest, Extensions


class TopologyConversionError(ValueError):
    """A required canonical field was genuinely unavailable or conflicting."""


def _source(provenance: Provenance) -> DiscoverySource:
    lowered = provenance.source.lower()
    if provenance.source_kind == "fixture":
        return DiscoverySource.SYNTHETIC
    if "nvidia-smi" in lowered:
        return DiscoverySource.NVIDIA_SMI
    if "nvml" in lowered:
        return DiscoverySource.NVML
    if "nccl" in lowered:
        return DiscoverySource.NCCL
    if "ib" in lowered:
        return DiscoverySource.IBVERBS
    if "hwloc" in lowered or "lstopo" in lowered:
        return DiscoverySource.HWLOC
    if provenance.source_kind in {"sysfs", "api", "environment", "derived"}:
        return DiscoverySource.SYSFS
    return DiscoverySource.SYSFS


def _provenance(fact: ObservedFact) -> tuple[FactProvenance, ...]:
    return tuple(
        FactProvenance(
            source=_source(observation.provenance),
            observed_at=datetime.fromisoformat(observation.provenance.captured_at),
            confidence=observation.provenance.confidence,
            source_uri=observation.provenance.artifact or observation.provenance.source,
            field=fact.name,
        )
        for observation in fact.observations
    )


def _all_provenance(node: TopologyNode) -> tuple[FactProvenance, ...]:
    return tuple(item for fact in node.facts for item in _provenance(fact))


def _fact(node: TopologyNode, name: str) -> ObservedFact:
    fact = node.fact(name)
    if fact is None or fact.state is not FactState.KNOWN or fact.value is None:
        state = fact.state.value if fact is not None else "missing"
        raise TopologyConversionError(
            f"canonical field {name!r} for {node.node_id} is {state}; refusing to infer it"
        )
    return fact


def _string(node: TopologyNode, name: str) -> str:
    value = _fact(node, name).value
    if not isinstance(value, str) or not value:
        raise TopologyConversionError(f"{node.node_id}.{name} must be a non-empty string")
    return value


def _integer(node: TopologyNode, name: str, *, positive: bool = False) -> int:
    value = _fact(node, name).value
    if isinstance(value, bool) or not isinstance(value, int) or (positive and value <= 0):
        raise TopologyConversionError(f"{node.node_id}.{name} must be an integer")
    return value


def _optional_string(node: TopologyNode, name: str) -> str | None:
    fact = node.fact(name)
    return (
        str(fact.value)
        if fact and fact.state is FactState.KNOWN and fact.value is not None
        else None
    )


def _optional_bool(node: TopologyNode, name: str) -> bool | None:
    fact = node.fact(name)
    return (
        fact.value
        if fact and fact.state is FactState.KNOWN and isinstance(fact.value, bool)
        else None
    )


def _health(value: HealthState) -> CanonicalHealth:
    return {
        HealthState.HEALTHY: CanonicalHealth.HEALTHY,
        HealthState.DEGRADED: CanonicalHealth.DEGRADED,
        HealthState.UNHEALTHY: CanonicalHealth.FAILED,
        HealthState.UNKNOWN: CanonicalHealth.UNKNOWN,
    }[value]


def _numa_for_gpu(graph: DiscoveryTopologyGraph, gpu_id: str) -> str | None:
    return next(
        (
            edge.source
            for edge in graph.edges
            if edge.kind is EdgeKind.CPU_GPU and edge.target == gpu_id
        ),
        None,
    )


def _canonical_node(
    graph: DiscoveryTopologyGraph, node: TopologyNode
) -> (
    HostNode
    | CpuSocketNode
    | NumaDomainNode
    | MemoryDomainNode
    | GpuNode
    | NvSwitchNode
    | PcieNode
    | NicNode
    | NetworkRailNode
    | StorageTierNode
    | RemoteMemoryNode
):
    provenance = _all_provenance(node)
    if node.kind is NodeKind.HOST:
        return HostNode(
            node_id=node.node_id,
            name=_string(node, "hostname"),
            architecture=_string(node, "architecture"),
            operating_system=_string(node, "operating_system"),
            total_memory_bytes=_integer(node, "memory_capacity", positive=True),
            visible_memory_bytes=_integer(node, "visible_memory_capacity", positive=True),
            container_visible=graph.visibility.in_container,
            fault_domain=node.node_id,
            provenance=provenance,
        )
    if node.kind is NodeKind.CPU_SOCKET:
        return CpuSocketNode(
            node_id=node.node_id,
            host_id=node.host_id,
            socket_index=_integer(node, "socket_index"),
            model=_string(node, "cpu_model"),
            physical_cores=_integer(node, "physical_core_count", positive=True),
            logical_cores=_integer(node, "logical_cpu_count", positive=True),
            provenance=provenance,
        )
    if node.kind is NodeKind.NUMA_DOMAIN:
        numa_index = _integer(node, "numa_index")
        return NumaDomainNode(
            node_id=node.node_id,
            host_id=node.host_id,
            socket_id=f"{node.host_id}/socket/{numa_index}",
            numa_index=numa_index,
            cpu_set=_string(node, "cpu_set"),
            memory_bytes=_integer(node, "memory_capacity", positive=True),
            provenance=provenance,
        )
    if node.kind is NodeKind.MEMORY_DOMAIN:
        bandwidth = node.fact("measured_bandwidth")
        bandwidth_gbps = (
            float(bandwidth.value) * 8 / 1e9
            if bandwidth
            and bandwidth.state is FactState.KNOWN
            and isinstance(bandwidth.value, (int, float))
            and not isinstance(bandwidth.value, bool)
            else None
        )
        return MemoryDomainNode(
            node_id=node.node_id,
            host_id=node.host_id,
            numa_domain_id=_string(node, "numa_domain_id"),
            capacity_bytes=_integer(node, "memory_capacity", positive=True),
            measured_bandwidth_gbps=bandwidth_gbps,
            provenance=provenance,
        )
    if node.kind is NodeKind.GPU:
        mig = _optional_bool(node, "mig_mode")
        return GpuNode(
            node_id=node.node_id,
            host_id=node.host_id,
            gpu_index=_integer(node, "index"),
            uuid=_string(node, "uuid"),
            product=_string(node, "product_name"),
            architecture=_string(node, "architecture"),
            memory_bytes=_integer(node, "memory_capacity", positive=True),
            compute_capability=_optional_string(node, "compute_capability"),
            mig_state=MigState.ENABLED
            if mig is True
            else MigState.DISABLED
            if mig is False
            else MigState.UNKNOWN,
            numa_domain_id=_numa_for_gpu(graph, node.node_id),
            pci_address=_optional_string(node, "pci_bus_id"),
            health=_health(node.health),
            provenance=provenance,
        )
    if node.kind is NodeKind.NVSWITCH:
        return NvSwitchNode(
            node_id=node.node_id,
            host_id=node.host_id,
            switch_domain=_string(node, "switch_domain"),
            generation=_optional_string(node, "generation"),
            health=_health(node.health),
            provenance=provenance,
        )
    if node.kind in {NodeKind.PCIE_ROOT, NodeKind.PCIE_SWITCH}:
        generation_fact = node.fact("pcie_generation")
        width_fact = node.fact("pcie_width")
        generation = (
            generation_fact.value
            if generation_fact
            and generation_fact.state is FactState.KNOWN
            and isinstance(generation_fact.value, int)
            and not isinstance(generation_fact.value, bool)
            else None
        )
        width = (
            width_fact.value
            if width_fact
            and width_fact.state is FactState.KNOWN
            and isinstance(width_fact.value, int)
            and not isinstance(width_fact.value, bool)
            else None
        )
        return PcieNode(
            kind=("pcie_root_complex" if node.kind is NodeKind.PCIE_ROOT else "pcie_switch"),
            node_id=node.node_id,
            host_id=node.host_id,
            pci_address=_optional_string(node, "pci_bus_id"),
            generation=generation,
            width=width,
            provenance=provenance,
        )
    if node.kind is NodeKind.NIC:
        transport = _string(node, "transport")
        if transport not in {"ethernet", "infiniband", "roce"}:
            transport = "unknown"
        link_speed = node.fact("link_speed")
        speed = (
            float(link_speed.value) / 1_000.0
            if link_speed
            and link_speed.state is FactState.KNOWN
            and isinstance(link_speed.value, (int, float))
            and not isinstance(link_speed.value, bool)
            else None
        )
        active = _optional_bool(node, "active_port")
        return NicNode(
            node_id=node.node_id,
            host_id=node.host_id,
            interface=_string(node, "interface_name"),
            pci_address=_optional_string(node, "pci_bus_id"),
            speed_gbps=speed,
            transport=cast(Literal["ethernet", "infiniband", "roce", "unknown"], transport),
            active=active is True,
            rdma_capable=_optional_bool(node, "rdma_capable"),
            gpu_direct_rdma=_optional_bool(node, "gpudirect_rdma"),
            numa_domain_id=None,
            health=_health(node.health),
            provenance=provenance,
        )
    if node.kind is NodeKind.NETWORK_RAIL:
        transport = _string(node, "transport")
        if transport not in {"ethernet", "infiniband", "roce", "synthetic"}:
            raise TopologyConversionError(f"unsupported network rail transport {transport!r}")
        return NetworkRailNode(
            node_id=node.node_id,
            name=_string(node, "name"),
            transport=cast(Literal["ethernet", "infiniband", "roce", "synthetic"], transport),
            subnet=_optional_string(node, "subnet"),
            health=_health(node.health),
            provenance=provenance,
        )
    if node.kind is NodeKind.STORAGE_TIER:
        capacity = node.fact("capacity")
        capacity_bytes = (
            capacity.value
            if capacity
            and capacity.state is FactState.KNOWN
            and isinstance(capacity.value, int)
            and not isinstance(capacity.value, bool)
            else None
        )
        tier = _string(node, "tier")
        if tier not in {"object", "remote_fs", "local_nvme", "page_cache", "memory"}:
            raise TopologyConversionError(f"unsupported storage tier {tier!r}")
        return StorageTierNode(
            node_id=node.node_id,
            host_id=node.host_id,
            tier=cast(Literal["object", "remote_fs", "local_nvme", "page_cache", "memory"], tier),
            capacity_bytes=capacity_bytes,
            provenance=provenance,
        )
    if node.kind is NodeKind.REMOTE_MEMORY:
        return RemoteMemoryNode(
            node_id=node.node_id,
            host_id=node.host_id,
            capacity_bytes=_integer(node, "capacity", positive=True),
            protocol=_string(node, "protocol"),
            provenance=provenance,
        )
    raise TopologyConversionError(
        f"raw discovery node kind {node.kind.value!r} requires an explicit canonical adapter"
    )


def _edge_number(edge: TopologyEdge, name: str) -> float | None:
    fact = edge.fact(name)
    if (
        fact is not None
        and fact.state is FactState.KNOWN
        and isinstance(fact.value, (int, float))
        and not isinstance(fact.value, bool)
    ):
        return float(fact.value)
    return None


def _canonical_edge(edge: TopologyEdge, captured_at: str, environment_hash: str) -> CanonicalEdge:
    measured_bandwidth = _edge_number(edge, "measured_bandwidth")
    latency = _edge_number(edge, "latency")
    bandwidth_gbps = measured_bandwidth * 8.0 / 1e9 if measured_bandwidth else None
    confidence = _edge_number(edge, "measurement_confidence") or 0.95
    curves_present = bandwidth_gbps is not None or latency is not None
    provenance = tuple(item for fact in edge.facts for item in _provenance(fact))
    connection = {
        EdgeKind.CPU_MEMORY: ConnectionType.CPU_MEMORY,
        EdgeKind.CPU_GPU: ConnectionType.CPU_GPU,
        EdgeKind.GPU_GPU: (
            ConnectionType.NVLINK
            if edge.fact("connection_type") is not None
            and edge.fact("connection_type").value == "nvlink"  # type: ignore[union-attr]
            else ConnectionType.GPU_GPU
        ),
        EdgeKind.GPU_NIC: ConnectionType.GPU_NIC,
        EdgeKind.NIC_NETWORK: ConnectionType.NIC_NETWORK,
        EdgeKind.PCIE: ConnectionType.PCIE,
        EdgeKind.STORAGE_HOST: ConnectionType.STORAGE_HOST,
        EdgeKind.REMOTE_MEMORY: ConnectionType.REMOTE_MEMORY,
    }.get(edge.kind)
    if connection is None:
        raise TopologyConversionError(f"edge kind {edge.kind.value} is not a physical connection")
    bandwidth_curve = (
        (
            CurvePoint(
                message_bytes=1_048_576,
                median=bandwidth_gbps,
                p95=bandwidth_gbps,
                robust_dispersion=0.0,
                confidence_low=bandwidth_gbps * 0.95,
                confidence_high=bandwidth_gbps * 1.05,
                sample_count=1,
            ),
        )
        if bandwidth_gbps is not None
        else ()
    )
    latency_curve = (
        (
            CurvePoint(
                message_bytes=1_048_576,
                median=latency,
                p95=latency,
                robust_dispersion=0.0,
                confidence_low=latency * 0.95,
                confidence_high=latency * 1.05,
                sample_count=1,
            ),
        )
        if latency is not None and latency > 0
        else ()
    )
    theoretical = _edge_number(edge, "theoretical_bandwidth")
    return CanonicalEdge(
        edge_id=edge.edge_id,
        source_node_id=edge.source,
        target_node_id=edge.target,
        connection=connection,
        directionality="unidirectional" if edge.directed else "bidirectional",
        duplex="full"
        if edge.full_duplex is True
        else "half"
        if edge.full_duplex is False
        else "unknown",
        theoretical_bandwidth_gbps=theoretical * 8.0 / 1e9 if theoretical else None,
        bandwidth_curve_gbps=bandwidth_curve,
        latency_curve_us=latency_curve,
        sharing_group=edge.sharing_group,
        contention_domain=edge.contention_domain,
        health=_health(edge.health),
        measurement_confidence=confidence if curves_present else None,
        measured_at=datetime.fromisoformat(captured_at) if curves_present else None,
        measurement_environment_digest=(
            ArtifactDigest(algorithm="sha256", value=environment_hash) if curves_present else None
        ),
        discovery_provenance=provenance,
    )


def to_canonical_topology(graph: DiscoveryTopologyGraph) -> CanonicalTopologyGraph:
    """Finalize raw records, failing when canonical required values are unknown."""
    environment_hash = hashlib.sha256(
        json.dumps(
            {
                "software": [item.model_dump(mode="json") for item in graph.software],
                "visibility": graph.visibility.model_dump(mode="json"),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    nodes = tuple(_canonical_node(graph, node) for node in graph.nodes)
    edges = tuple(
        _canonical_edge(edge, graph.captured_at, environment_hash)
        for edge in graph.edges
        if edge.kind is not EdgeKind.CONTAINS
    )
    software = tuple(
        CanonicalSoftwareComponent(
            name=item.name,
            version=item.version,
            source=_source(item.provenance),
        )
        for item in graph.software
        if item.state is FactState.KNOWN and item.version is not None
    )
    warnings = (*graph.warnings, "Hierarchy-only contains edges are encoded by canonical node IDs.")
    return CanonicalTopologyGraph(
        topology_id=graph.topology_id,
        discovered_at=datetime.fromisoformat(graph.captured_at),
        nodes=nodes,
        edges=edges,
        software=software,
        container_limited=graph.visibility.in_container,
        discovery_warnings=warnings,
        extensions=Extensions(
            root={
                "sloforge.io/discovery-fingerprint": graph.fingerprint,
                "sloforge.io/visibility-restrictions": list(graph.visibility.restrictions),
                "sloforge.io/visible-gpu-ids": list(graph.visibility.visible_gpu_ids),
            }
        ),
    )
