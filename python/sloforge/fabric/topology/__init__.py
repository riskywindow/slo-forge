"""Topology discovery public API.

CLI-facing functions return the canonical :mod:`sloforge.fabric.ir` graph.
Raw records remain available under explicit ``Discovery*`` names for auditing
and source-conflict analysis.
"""

from sloforge.fabric.ir import TopologyGraph
from sloforge.fabric.topology.conversion import TopologyConversionError, to_canonical_topology
from sloforge.fabric.topology.discovery import (
    discover_topology,
    discover_topology_records,
    load_discovery_records,
    load_topology,
    save_discovery_records,
    save_topology,
)
from sloforge.fabric.topology.fixtures import (
    build_canonical_fixture,
    build_discovery_fixture,
    build_fixture,
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
    Visibility,
)
from sloforge.fabric.topology.models import (
    TopologyEdge as DiscoveryTopologyEdge,
)
from sloforge.fabric.topology.models import (
    TopologyNode as DiscoveryTopologyNode,
)

__all__ = [
    "DiscoveryTopologyEdge",
    "DiscoveryTopologyGraph",
    "DiscoveryTopologyNode",
    "EdgeKind",
    "FactState",
    "HealthState",
    "NodeKind",
    "Observation",
    "ObservedFact",
    "Provenance",
    "SoftwareComponent",
    "TopologyConversionError",
    "TopologyGraph",
    "Visibility",
    "build_canonical_fixture",
    "build_discovery_fixture",
    "build_fixture",
    "discover_topology",
    "discover_topology_records",
    "load_discovery_records",
    "load_topology",
    "save_discovery_records",
    "save_topology",
    "to_canonical_topology",
]
