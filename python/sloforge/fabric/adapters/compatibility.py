"""Fail-closed runtime compatibility and capability validation."""

from __future__ import annotations

from dataclasses import dataclass

from sloforge.fabric.adapters.models import (
    DeploymentTarget,
    FabricAdapterContext,
    GangScheduler,
    RuntimeKind,
    UnsupportedCapabilityError,
    parse_version,
)


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
        (0, 12, 0),
        (1, 0, 0),
        "https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/",
    ),
    RuntimeKind.SGLANG: VersionRange(
        (0, 5, 0),
        (1, 0, 0),
        "https://docs.sglang.ai/backend/pd_disaggregation.html",
    ),
    RuntimeKind.DYNAMO: VersionRange(
        (1, 0, 0),
        (2, 0, 0),
        "https://docs.nvidia.com/dynamo/dev/kubernetes-api/full-api-reference",
    ),
}


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
    if plan.parallelism.context_parallel_degree > 1 and context.runtime is not RuntimeKind.NATIVE:
        raise UnsupportedCapabilityError(
            f"context parallelism is not validated for {context.runtime.value} adapter"
        )

    if target is DeploymentTarget.DYNAMO and context.runtime is not RuntimeKind.DYNAMO:
        raise UnsupportedCapabilityError("Dynamo target requires runtime=dynamo")
    if context.runtime is RuntimeKind.DYNAMO and target is not DeploymentTarget.DYNAMO:
        raise UnsupportedCapabilityError("Dynamo runtime is emitted only through the Dynamo target")

    if target is DeploymentTarget.KUBERNETES and len(hosts) > 1:
        if context.runtime is not RuntimeKind.NATIVE:
            raise UnsupportedCapabilityError(
                "generic Kubernetes cannot preserve multi-node engine rank identity; use Dynamo"
            )
        if context.gang_scheduler is GangScheduler.NONE:
            raise UnsupportedCapabilityError(
                "multi-node Kubernetes plans require an explicit Grove or LWS gang scheduler"
            )

    if (
        target in {DeploymentTarget.MODAL, DeploymentTarget.TRUSS}
        and not context.allow_advisory_cloud_metadata
    ):
        raise UnsupportedCapabilityError(
            f"{target.value} does not expose exact rank/GPU/NIC placement in its supported "
            "schema; set allow_advisory_cloud_metadata=true to emit non-enforcing metadata"
        )

    if context.runtime is RuntimeKind.VLLM and plan.parallelism.prefill_decode_disaggregated:
        routes = plan.kv_transfer.routes if plan.kv_transfer is not None else ()
        if not routes or any(
            route.serialization_format != "nixl" or "nixl" not in route.transport_adapter.casefold()
            for route in routes
        ):
            raise UnsupportedCapabilityError(
                "vLLM disaggregation requires an explicit NixlConnector KV route"
            )

    if context.runtime is RuntimeKind.SGLANG and plan.parallelism.prefill_decode_disaggregated:
        routes = plan.kv_transfer.routes if plan.kv_transfer is not None else ()
        supported_transfers = ("nixl", "mooncake")
        if not routes or any(
            not any(name in route.transport_adapter.casefold() for name in supported_transfers)
            for route in routes
        ):
            raise UnsupportedCapabilityError(
                "SGLang disaggregation requires an explicit NIXL or Mooncake route"
            )

    cross_host_fabric = len(hosts) > 1 and (
        any(
            operation.transport in {"infiniband", "roce"}
            for operation in plan.collectives.operations
        )
        or plan.kv_transfer is not None
    )
    if cross_host_fabric and context.rdma_resource_name is None:
        raise UnsupportedCapabilityError(
            "cross-host fabric plan requires an explicit cluster RDMA extended resource name"
        )
