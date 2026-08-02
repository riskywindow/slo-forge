"""Fail-closed runtime compatibility and capability validation."""

from __future__ import annotations

from dataclasses import dataclass

from sloforge.fabric.adapters.models import (
    DeploymentTarget,
    DynamoBackend,
    FabricAdapterContext,
    RuntimeKind,
    UnsupportedCapabilityError,
    parse_version,
)
from sloforge.fabric.ir import ParallelismKind, WorkerRole


@dataclass(frozen=True)
class VersionRange:
    minimum: tuple[int, int, int]
    maximum_exclusive: tuple[int, int, int]
    source: str


RUNTIME_RANGES: dict[RuntimeKind, VersionRange] = {
    RuntimeKind.NATIVE: VersionRange(
        (0, 1, 0),
        (0, 2, 0),
        "SLOForge Fabric internal JSON protocol v1",
    ),
    RuntimeKind.VLLM: VersionRange(
        (0, 26, 0),
        (0, 27, 0),
        "https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/",
    ),
    RuntimeKind.SGLANG: VersionRange(
        (0, 5, 2),
        (0, 6, 0),
        "https://docs.sglang.io/docs/advanced_features/pd_disaggregation",
    ),
    RuntimeKind.DYNAMO: VersionRange(
        (1, 2, 0),
        (1, 4, 0),
        "https://docs.nvidia.com/dynamo/dev/reference/releases/v1-2-0",
    ),
}


def _engine_runtime(context: FabricAdapterContext) -> RuntimeKind:
    if context.runtime is not RuntimeKind.DYNAMO:
        return context.runtime
    if context.dynamo_backend is DynamoBackend.VLLM:
        return RuntimeKind.VLLM
    if context.dynamo_backend is DynamoBackend.SGLANG:
        return RuntimeKind.SGLANG
    raise UnsupportedCapabilityError("Dynamo requires a validated vLLM or SGLang backend")


def _replica_roles(context: FabricAdapterContext) -> dict[str, WorkerRole]:
    roles: dict[str, set[WorkerRole]] = {}
    for binding in context.plan.rank_placement.bindings:
        roles.setdefault(binding.replica_id, set()).add(binding.worker_role)
    split = {replica_id: values for replica_id, values in roles.items() if len(values) != 1}
    if split:
        identifiers = ", ".join(sorted(split))
        raise UnsupportedCapabilityError(
            "runtime adapters require every complete model replica to have one worker role; "
            f"split replica(s): {identifiers}"
        )
    return {replica_id: next(iter(values)) for replica_id, values in roles.items()}


def _validate_worker_roles(context: FabricAdapterContext) -> dict[str, WorkerRole]:
    roles = _replica_roles(context)
    actual = set(roles.values())
    if context.plan.parallelism.prefill_decode_disaggregated:
        if not actual <= {WorkerRole.PREFILL, WorkerRole.DECODE} or actual != {
            WorkerRole.PREFILL,
            WorkerRole.DECODE,
        }:
            raise UnsupportedCapabilityError(
                "disaggregated runtime lowering requires complete prefill and decode replicas"
            )
    elif actual != {WorkerRole.AGGREGATED}:
        raise UnsupportedCapabilityError(
            "aggregated runtime lowering requires every replica to use role=aggregated"
        )
    return roles


def _validate_expert_parallelism(
    context: FabricAdapterContext,
    target: DeploymentTarget,
    engine: RuntimeKind,
    replica_roles: dict[str, WorkerRole],
) -> None:
    plan = context.plan
    degree = plan.parallelism.expert_parallel_degree
    if degree == 1:
        return

    groups = tuple(
        group for group in plan.parallelism.groups if group.kind is ParallelismKind.EXPERT
    )
    if not groups or any(len(group.rank_ids) != degree for group in groups):
        raise UnsupportedCapabilityError(
            "expert groups must explicitly partition ranks at expert_parallel_degree"
        )
    flattened = [rank_id for group in groups for rank_id in group.rank_ids]
    expected_ranks = {binding.rank_id for binding in plan.rank_placement.bindings}
    if len(flattened) != len(set(flattened)) or set(flattened) != expected_ranks:
        raise UnsupportedCapabilityError(
            "expert groups must partition every physical rank exactly once"
        )

    rank_roles = {
        binding.rank_id: replica_roles[binding.replica_id]
        for binding in plan.rank_placement.bindings
    }
    if any(len({rank_roles[rank_id] for rank_id in group.rank_ids}) != 1 for group in groups):
        raise UnsupportedCapabilityError(
            "an expert group cannot cross prefill, decode, or aggregated runtime components"
        )

    if engine is RuntimeKind.VLLM:
        if target is DeploymentTarget.DYNAMO:
            role_replica_counts: dict[WorkerRole, int] = {}
            for role in replica_roles.values():
                role_replica_counts[role] = role_replica_counts.get(role, 0) + 1
            rendered_dp_sizes = set(role_replica_counts.values())
        else:
            # Local, Docker, and generic Kubernetes emit one complete replica
            # per process group, not one cross-replica vLLM DP supervisor.
            rendered_dp_sizes = {1}
        representable = {
            plan.parallelism.tensor_parallel_degree * data_parallel_size
            for data_parallel_size in rendered_dp_sizes
        }
        if representable != {degree}:
            values = ", ".join(str(value) for value in sorted(representable))
            raise UnsupportedCapabilityError(
                "vLLM computes EP_SIZE as TP_SIZE * rendered DP_SIZE; physical plan "
                f"expert_parallel_degree={degree}, representable degree(s)={values}"
            )

    # SGLang exposes --ep-size directly. DP attention is a separate
    # attention-sharding mode and is intentionally not inferred from EP.
    if engine is RuntimeKind.SGLANG and degree > plan.parallelism.tensor_parallel_degree:
        raise UnsupportedCapabilityError(
            "SGLang EP larger than TP requires an explicit MoE-DP topology that this "
            "adapter does not encode"
        )


def sglang_disaggregation_backend(context: FabricAdapterContext) -> str:
    """Return the one SGLang transfer backend represented by every KV route."""

    routes = context.plan.kv_transfer.routes if context.plan.kv_transfer is not None else ()
    selected: set[str] = set()
    for route in routes:
        adapter = route.transport_adapter.casefold()
        matches = {name for name in ("nixl", "mooncake") if name in adapter}
        if len(matches) != 1:
            raise UnsupportedCapabilityError(
                "SGLang disaggregation requires each route to identify exactly one of "
                "NIXL or Mooncake"
            )
        selected.update(matches)
    if len(selected) != 1:
        raise UnsupportedCapabilityError(
            "one SGLang worker command cannot represent mixed NIXL and Mooncake KV routes"
        )
    return next(iter(selected))


def validate_runtime(context: FabricAdapterContext, target: DeploymentTarget) -> None:
    supported = RUNTIME_RANGES[context.runtime]
    actual = parse_version(context.runtime_version)
    if actual < supported.minimum or actual >= supported.maximum_exclusive:
        minimum = ".".join(str(item) for item in supported.minimum)
        maximum = ".".join(str(item) for item in supported.maximum_exclusive)
        raise UnsupportedCapabilityError(
            f"{context.runtime.value} {context.runtime_version} is outside validated range "
            f"[{minimum}, {maximum}); source: {supported.source}"
        )

    plan = context.plan
    hosts = {binding.host_id for binding in plan.rank_placement.bindings}
    engine = _engine_runtime(context)
    if plan.parallelism.context_parallel_degree > 1 and context.runtime is not RuntimeKind.NATIVE:
        raise UnsupportedCapabilityError(
            f"context parallelism is not validated for {context.runtime.value} adapter"
        )

    if target is DeploymentTarget.DYNAMO and context.runtime is not RuntimeKind.DYNAMO:
        raise UnsupportedCapabilityError("Dynamo target requires runtime=dynamo")
    if context.runtime is RuntimeKind.DYNAMO and target is not DeploymentTarget.DYNAMO:
        raise UnsupportedCapabilityError("Dynamo runtime is emitted only through the Dynamo target")

    if (
        target is DeploymentTarget.DYNAMO
        and len(hosts) > 1
        and context.gang_scheduler.value == "none"
    ):
        raise UnsupportedCapabilityError(
            "multi-node Dynamo requires an explicit Grove or LWS gang scheduler contract"
        )

    if target is DeploymentTarget.KUBERNETES and len(hosts) > 1:
        # The generic exporter emits ordinary Deployments and cannot atomically
        # create a multi-node rank group. Merely naming a gang scheduler is not
        # equivalent to generating its PodGroup/LeaderWorkerSet contract.
        raise UnsupportedCapabilityError(
            "generic Kubernetes export cannot enforce multi-node gang startup; use the "
            "validated DynamoGraphDeployment target"
        )

    if (
        target in {DeploymentTarget.MODAL, DeploymentTarget.TRUSS}
        and not context.allow_advisory_cloud_metadata
    ):
        raise UnsupportedCapabilityError(
            f"{target.value} does not expose exact rank/GPU/NIC placement in its supported "
            "schema; set allow_advisory_cloud_metadata=true to emit non-enforcing metadata"
        )

    if (
        target not in {DeploymentTarget.MODAL, DeploymentTarget.TRUSS}
        and engine is not RuntimeKind.NATIVE
    ):
        replica_roles = _validate_worker_roles(context)
        if target is not DeploymentTarget.DYNAMO:
            replica_hosts: dict[str, set[str]] = {}
            for binding in plan.rank_placement.bindings:
                replica_hosts.setdefault(binding.replica_id, set()).add(binding.host_id)
            if any(len(values) > 1 for values in replica_hosts.values()):
                raise UnsupportedCapabilityError(
                    "local, Docker, and generic Kubernetes engine adapters do not emit "
                    "a multi-node rendezvous; use the Dynamo target"
                )
        if engine is RuntimeKind.SGLANG and target is DeploymentTarget.DYNAMO:
            role_counts: dict[WorkerRole, int] = {}
            for role in replica_roles.values():
                role_counts[role] = role_counts.get(role, 0) + 1
            if any(count > 1 for count in role_counts.values()):
                raise UnsupportedCapabilityError(
                    "SGLang --dp-size is DP-attention topology, not independent physical "
                    "replicas; role-level multi-replica lowering is not representable"
                )
        _validate_expert_parallelism(context, target, engine, replica_roles)

    if engine is RuntimeKind.VLLM and plan.parallelism.prefill_decode_disaggregated:
        routes = plan.kv_transfer.routes if plan.kv_transfer is not None else ()
        if not routes or any("nixl" not in route.transport_adapter.casefold() for route in routes):
            raise UnsupportedCapabilityError(
                "vLLM disaggregation requires an explicit NixlConnector KV route"
            )

    if engine is RuntimeKind.SGLANG and plan.parallelism.prefill_decode_disaggregated:
        sglang_disaggregation_backend(context)

    rank_hosts = {binding.rank_id: binding.host_id for binding in plan.rank_placement.bindings}
    cross_host_collective = any(
        operation.transport in {"infiniband", "roce"}
        and len({rank_hosts[rank_id] for rank_id in operation.participating_ranks}) > 1
        for operation in plan.collectives.operations
    )
    cross_host_kv = plan.kv_transfer is not None and any(
        rank_hosts[producer] != rank_hosts[consumer]
        for route in plan.kv_transfer.routes
        for producer in route.producer_rank_ids
        for consumer in route.consumer_rank_ids
    )
    if (cross_host_collective or cross_host_kv) and context.rdma_resource_name is None:
        raise UnsupportedCapabilityError(
            "cross-host fabric plan requires an explicit cluster RDMA extended resource name"
        )
