"""Strict normalized topology discovery records.

The discovery layer deliberately represents a fact's epistemic state separately
from its value.  In particular, an unavailable datum is never converted into a
zero or an inferred capability.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Annotated, Any, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

Scalar = str | int | float | bool | None
NonEmpty = Annotated[str, Field(min_length=1)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class NodeKind(StrEnum):
    HOST = "host"
    CPU_SOCKET = "cpu_socket"
    NUMA_DOMAIN = "numa_domain"
    MEMORY_DOMAIN = "memory_domain"
    GPU = "gpu"
    NVSWITCH = "nvswitch"
    PCIE_ROOT = "pcie_root_complex"
    PCIE_SWITCH = "pcie_switch"
    NIC = "nic"
    NETWORK_RAIL = "network_rail"
    STORAGE_TIER = "storage_tier"
    REMOTE_MEMORY = "remote_memory_endpoint"


class EdgeKind(StrEnum):
    CPU_MEMORY = "cpu_to_memory"
    CPU_GPU = "cpu_to_gpu"
    GPU_GPU = "gpu_to_gpu"
    GPU_NIC = "gpu_to_nic"
    NIC_NETWORK = "nic_to_network"
    STORAGE_HOST = "storage_to_host"
    REMOTE_MEMORY = "remote_memory_path"
    CONTAINS = "contains"
    PCIE = "pcie"


class FactState(StrEnum):
    KNOWN = "known"
    UNKNOWN = "unknown"
    CONFLICT = "conflict"


class HealthState(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class Provenance(StrictModel):
    source: NonEmpty
    source_kind: Literal["api", "sysfs", "command", "environment", "fixture", "derived"]
    captured_at: NonEmpty
    command: tuple[str, ...] = ()
    confidence: float = Field(ge=0.0, le=1.0)
    artifact: str | None = None


class Observation(StrictModel):
    value: Scalar
    provenance: Provenance


class ObservedFact(StrictModel):
    name: NonEmpty
    unit: str | None = None
    state: FactState
    value: Scalar = None
    observations: tuple[Observation, ...]

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        if not self.observations:
            raise ValueError("a fact requires at least one observation, including unknown facts")
        if self.state is FactState.KNOWN and self.value is None:
            raise ValueError("known facts require a value")
        if self.state is FactState.UNKNOWN and self.value is not None:
            raise ValueError("unknown facts cannot carry an inferred value")
        if self.state is FactState.CONFLICT:
            values = {json.dumps(item.value, sort_keys=True) for item in self.observations}
            if len(values) < 2:
                raise ValueError("conflicting facts require at least two distinct observations")
        return self


class TopologyNode(StrictModel):
    node_id: NonEmpty
    kind: NodeKind
    host_id: NonEmpty
    health: HealthState = HealthState.UNKNOWN
    facts: tuple[ObservedFact, ...] = ()

    @model_validator(mode="after")
    def unique_fact_names(self) -> Self:
        names = [fact.name for fact in self.facts]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate fact names on node {self.node_id}")
        return self

    def fact(self, name: str) -> ObservedFact | None:
        return next((item for item in self.facts if item.name == name), None)


class TopologyEdge(StrictModel):
    edge_id: NonEmpty
    source: NonEmpty
    target: NonEmpty
    kind: EdgeKind
    directed: bool
    full_duplex: bool | None = None
    sharing_group: str | None = None
    contention_domain: str | None = None
    health: HealthState = HealthState.UNKNOWN
    facts: tuple[ObservedFact, ...] = ()

    def fact(self, name: str) -> ObservedFact | None:
        return next((item for item in self.facts if item.name == name), None)


class Visibility(StrictModel):
    in_container: bool
    host_devices_visible: bool | None
    visible_gpu_ids: tuple[str, ...]
    restrictions: tuple[str, ...]
    facts: tuple[ObservedFact, ...]


class SoftwareComponent(StrictModel):
    name: NonEmpty
    version: str | None
    state: FactState
    provenance: Provenance


class DiscoveryTopologyGraph(StrictModel):
    schema_version: Literal["sloforge.fabric.topology/v1"] = "sloforge.fabric.topology/v1"
    topology_id: NonEmpty
    captured_at: NonEmpty
    nodes: tuple[TopologyNode, ...]
    edges: tuple[TopologyEdge, ...]
    visibility: Visibility
    software: tuple[SoftwareComponent, ...]
    warnings: tuple[str, ...] = ()
    fingerprint: str

    @model_validator(mode="after")
    def validate_graph(self) -> Self:
        ids = [node.node_id for node in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("topology node identifiers must be unique")
        edge_ids = [edge.edge_id for edge in self.edges]
        if len(edge_ids) != len(set(edge_ids)):
            raise ValueError("topology edge identifiers must be unique")
        known = set(ids)
        for edge in self.edges:
            if edge.source not in known or edge.target not in known:
                raise ValueError(f"edge {edge.edge_id} references an unknown node")
        expected = topology_fingerprint(self)
        if expected != self.fingerprint:
            raise ValueError(f"topology fingerprint mismatch: expected {expected}")
        return self


def _stable_payload(graph: DiscoveryTopologyGraph) -> dict[str, object]:
    raw: Any = graph.model_dump(mode="json")
    raw.pop("fingerprint", None)
    raw.pop("captured_at", None)
    for node in raw["nodes"]:
        for fact in node["facts"]:
            for observation in fact["observations"]:
                observation["provenance"].pop("captured_at", None)
    for edge in raw["edges"]:
        for fact in edge["facts"]:
            for observation in fact["observations"]:
                observation["provenance"].pop("captured_at", None)
    for fact in raw["visibility"]["facts"]:
        for observation in fact["observations"]:
            observation["provenance"].pop("captured_at", None)
    for component in raw["software"]:
        component["provenance"].pop("captured_at", None)
    return cast(dict[str, object], raw)


def topology_fingerprint(graph: DiscoveryTopologyGraph) -> str:
    payload = _stable_payload(graph)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def finalize_graph(**values: object) -> DiscoveryTopologyGraph:
    provisional = DiscoveryTopologyGraph.model_construct(
        fingerprint="",
        **values,  # type: ignore[arg-type]
    )
    payload = provisional.model_dump(mode="json")
    payload["fingerprint"] = topology_fingerprint(provisional)
    return DiscoveryTopologyGraph.model_validate_json(json.dumps(payload))
