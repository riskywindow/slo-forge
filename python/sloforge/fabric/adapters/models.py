"""Typed contracts for lowering a physical plan into deployment artifacts."""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from sloforge.fabric.ir import (
    GpuNode,
    HostNode,
    NetworkRailNode,
    NicNode,
    NumaDomainNode,
    PhysicalExecutionPlan,
    TopologyGraph,
)

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class AdapterModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class DeploymentTarget(StrEnum):
    LOCAL = "local"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    DYNAMO = "dynamo"
    MODAL = "modal"
    TRUSS = "truss"


class RuntimeKind(StrEnum):
    NATIVE = "native"
    VLLM = "vllm"
    SGLANG = "sglang"
    DYNAMO = "dynamo"


class DynamoBackend(StrEnum):
    VLLM = "vllm"
    SGLANG = "sglang"


class GangScheduler(StrEnum):
    NONE = "none"
    GROVE = "grove"
    LWS = "lws"


class AdapterCapabilities(AdapterModel):
    exact_host_placement: bool
    exact_gpu_uuid_placement: bool
    numa_affinity: bool
    nic_affinity: bool
    network_rail_affinity: bool
    multi_node_gang_scheduling: bool
    prefill_decode_disaggregation: bool
    expert_parallelism: bool
    advisory_only: tuple[NonEmptyString, ...] = ()


class FabricAdapterContext(AdapterModel):
    plan: PhysicalExecutionPlan
    topology: TopologyGraph
    model_id: NonEmptyString
    model_revision: NonEmptyString
    image: NonEmptyString
    runtime: RuntimeKind
    runtime_version: Annotated[str, StringConstraints(pattern=r"^\d+\.\d+\.\d+$")]
    dynamo_backend: DynamoBackend | None = None
    namespace: Annotated[str, StringConstraints(pattern=r"^[a-z0-9](?:[-a-z0-9]*[a-z0-9])?$")] = (
        "default"
    )
    gpu_resource_name: Annotated[
        str, StringConstraints(pattern=r"^[a-z0-9.-]+/[A-Za-z0-9_.-]+$")
    ] = "nvidia.com/gpu"
    rdma_resource_name: (
        Annotated[str, StringConstraints(pattern=r"^[a-z0-9.-]+/[A-Za-z0-9_.-]+$")] | None
    ) = None
    gang_scheduler: GangScheduler = GangScheduler.NONE
    cpu_limit_per_rank: float = Field(default=4.0, gt=0.0, allow_inf_nan=False)
    memory_limit_gib_per_rank: int = Field(default=32, ge=1)
    pids_limit_per_rank: int = Field(default=512, ge=64)
    shutdown_grace_seconds: int = Field(default=120, ge=1, le=3600)
    allow_advisory_cloud_metadata: bool = False

    @model_validator(mode="after")
    def validate_composition(self) -> Self:
        topology_nodes = {node.node_id: node for node in self.topology.nodes}
        for binding in self.plan.rank_placement.bindings:
            host = topology_nodes.get(binding.host_id)
            gpu = topology_nodes.get(binding.gpu_id)
            numa = topology_nodes.get(binding.numa_domain_id)
            if not isinstance(host, HostNode):
                raise ValueError(f"rank {binding.rank_id} host_id is absent or not a host")
            if not isinstance(gpu, GpuNode) or gpu.host_id != binding.host_id:
                raise ValueError(
                    f"rank {binding.rank_id} gpu_id is absent or belongs to another host"
                )
            if not isinstance(numa, NumaDomainNode) or numa.host_id != binding.host_id:
                raise ValueError(
                    f"rank {binding.rank_id} numa_domain_id is absent or belongs to another host"
                )
            if binding.nic_id is not None:
                nic = topology_nodes.get(binding.nic_id)
                if not isinstance(nic, NicNode) or nic.host_id != binding.host_id:
                    raise ValueError(
                        f"rank {binding.rank_id} nic_id is absent or belongs to another host"
                    )
            if binding.network_rail_id is not None and not isinstance(
                topology_nodes.get(binding.network_rail_id), NetworkRailNode
            ):
                raise ValueError(
                    f"rank {binding.rank_id} network_rail_id is absent or not a network rail"
                )
        if self.runtime is RuntimeKind.DYNAMO and self.dynamo_backend is None:
            raise ValueError("Dynamo runtime requires dynamo_backend")
        if self.runtime is not RuntimeKind.DYNAMO and self.dynamo_backend is not None:
            raise ValueError("dynamo_backend is valid only for the Dynamo runtime")
        return self


class GeneratedArtifact(AdapterModel):
    path: NonEmptyString
    sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class FabricExportResult(AdapterModel):
    schema_version: Literal["sloforge.fabric-export/v1"] = "sloforge.fabric-export/v1"
    target: DeploymentTarget
    plan_id: NonEmptyString
    plan_hash: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    output_dir: Path
    deployed: Literal[False] = False
    capabilities: AdapterCapabilities
    artifacts: tuple[GeneratedArtifact, ...]
    validations: tuple[NonEmptyString, ...]


class UnsupportedCapabilityError(ValueError):
    """Raised when lowering would weaken a required physical-plan invariant."""


def parse_version(value: str) -> tuple[int, int, int]:
    if re.fullmatch(r"\d+\.\d+\.\d+", value) is None:
        raise ValueError(f"invalid semantic version: {value}")
    major, minor, patch = value.split(".")
    return int(major), int(minor), int(patch)
