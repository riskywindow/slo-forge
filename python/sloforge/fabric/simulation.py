"""Typed JSON boundary to the deterministic Rust Fabric simulator."""

from __future__ import annotations

import hashlib
import math
import os
import subprocess
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from sloforge.fabric.ir import (
    CollectiveOperation,
    ConnectionType,
    FabricProfile,
    HealthState,
    ParallelismKind,
    PhysicalExecutionPlan,
    RankBinding,
    TopologyEdge,
    TopologyGraph,
    WorkerRole,
)

_MAX_SIMULATOR_INPUT_BYTES = 64 * 1024 * 1024
_MAX_SIMULATOR_OUTPUT_BYTES = 128 * 1024 * 1024
_MAX_SIMULATOR_ERROR_BYTES = 16 * 1024

PositiveFloat = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]
NonNegativeFloat = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]


class SimulationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class ProvenanceKind(StrEnum):
    MEASURED = "measured"
    SYNTHETIC = "synthetic"
    ANALYTICAL = "analytical"


class CalibrationProvenance(SimulationModel):
    kind: ProvenanceKind
    artifact_uri: str = Field(min_length=1)
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    environment_fingerprint: str = Field(min_length=1)
    collected_at: str = Field(min_length=1)


class ServiceCurvePoint(SimulationModel):
    message_bytes: Annotated[int, Field(ge=0)]
    latency_us: NonNegativeFloat
    bandwidth_gbps: PositiveFloat
    uncertainty_fraction: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0


class ServiceCurve(SimulationModel):
    id: str = Field(min_length=1)
    points: tuple[ServiceCurvePoint, ...]
    provenance: CalibrationProvenance

    @model_validator(mode="after")
    def validate_points(self) -> ServiceCurve:
        sizes = [point.message_bytes for point in self.points]
        if not sizes or sizes != sorted(set(sizes)):
            raise ValueError("service-curve message sizes must be unique and increasing")
        return self


class ResourceKind(StrEnum):
    CPU_CORE_GROUP = "cpu_core_group"
    NUMA_MEMORY = "numa_memory"
    GPU_COMPUTE = "gpu_compute"
    GPU_HBM = "gpu_hbm"
    GPU_COPY_ENGINE = "gpu_copy_engine"
    NVLINK = "nvlink"
    NVSWITCH = "nvswitch"
    PCIE = "pcie"
    NIC_QUEUE = "nic_queue"
    NETWORK_RAIL = "network_rail"
    STORAGE_PATH = "storage_path"


class SchedulingMode(StrEnum):
    EXCLUSIVE = "exclusive"
    FAIR_SHARE = "fair_share"


class PhysicalResource(SimulationModel):
    id: str = Field(min_length=1)
    kind: ResourceKind
    scheduling: SchedulingMode
    capacity_units: PositiveFloat = 1.0
    max_concurrency: Annotated[int, Field(gt=0)] = 1
    curve: ServiceCurve
    sharing_group: str | None = None
    hourly_cost_usd: NonNegativeFloat = 0.0


class SharingGroup(SimulationModel):
    id: str = Field(min_length=1)
    capacity_units: PositiveFloat
    max_concurrency: Annotated[int, Field(gt=0)]


class ResourceDemand(SimulationModel):
    resource_id: str = Field(min_length=1)
    units: PositiveFloat = 1.0


class CpuLaunch(SimulationModel):
    type: Literal["cpu_launch"] = "cpu_launch"
    duration_us: PositiveFloat


class GpuCompute(SimulationModel):
    type: Literal["gpu_compute"] = "gpu_compute"
    duration_us: PositiveFloat


class PointToPoint(SimulationModel):
    type: Literal["point_to_point"] = "point_to_point"
    bytes: Annotated[int, Field(gt=0)]


class Collective(SimulationModel):
    type: Literal["collective"] = "collective"
    collective_id: str = Field(min_length=1)
    bytes: Annotated[int, Field(gt=0)]
    algorithm: str = Field(min_length=1)
    participating_ranks: tuple[str, ...]


class KvTransfer(SimulationModel):
    type: Literal["kv_transfer"] = "kv_transfer"
    bytes: Annotated[int, Field(gt=0)]
    chunks: Annotated[int, Field(gt=0)]


OperationKind = Annotated[
    CpuLaunch | GpuCompute | PointToPoint | Collective | KvTransfer,
    Field(discriminator="type"),
]


class PhysicalOperation(SimulationModel):
    id: str = Field(min_length=1)
    kind: OperationKind
    rank_ids: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    demands: tuple[ResourceDemand, ...] = ()
    earliest_start_us: NonNegativeFloat = 0.0
    uncertainty_fraction: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    request_id: str | None = None


class ResourceRateFault(SimulationModel):
    type: Literal["resource_rate"] = "resource_rate"
    resource_id: str = Field(min_length=1)
    multiplier: Annotated[float, Field(gt=0.0, le=1.0)]


class ResourceUnavailableFault(SimulationModel):
    type: Literal["resource_unavailable"] = "resource_unavailable"
    resource_id: str = Field(min_length=1)


class RankSlowdownFault(SimulationModel):
    type: Literal["rank_slowdown"] = "rank_slowdown"
    rank_id: str = Field(min_length=1)
    multiplier: Annotated[float, Field(gt=0.0, le=1.0)]


class CollectiveDelayFault(SimulationModel):
    type: Literal["collective_delay"] = "collective_delay"
    collective_id: str = Field(min_length=1)
    multiplier: Annotated[float, Field(gt=0.0, le=1.0)]


FaultEffect = Annotated[
    ResourceRateFault | ResourceUnavailableFault | RankSlowdownFault | CollectiveDelayFault,
    Field(discriminator="type"),
]


class TimedFault(SimulationModel):
    id: str = Field(min_length=1)
    start_us: NonNegativeFloat
    end_us: PositiveFloat | None = None
    effect: FaultEffect
    ground_truth_label: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_interval(self) -> TimedFault:
        if self.end_us is not None and self.end_us <= self.start_us:
            raise ValueError("fault end must be after start")
        return self


class RemoveFault(SimulationModel):
    type: Literal["remove_fault"] = "remove_fault"
    fault_id: str = Field(min_length=1)


class ScaleResourceCurve(SimulationModel):
    type: Literal["scale_resource_curve"] = "scale_resource_curve"
    resource_id: str = Field(min_length=1)
    latency_multiplier: PositiveFloat
    bandwidth_multiplier: PositiveFloat


class ScaleRank(SimulationModel):
    type: Literal["scale_rank"] = "scale_rank"
    rank_id: str = Field(min_length=1)
    duration_multiplier: PositiveFloat


class ReplaceResource(SimulationModel):
    type: Literal["replace_resource"] = "replace_resource"
    from_resource_id: str = Field(min_length=1)
    to_resource_id: str = Field(min_length=1)


CounterfactualModifier = Annotated[
    RemoveFault | ScaleResourceCurve | ScaleRank | ReplaceResource,
    Field(discriminator="type"),
]


class FabricSimulationRequest(SimulationModel):
    schema_version: Literal["1.0"] = "1.0"
    seed: Annotated[int, Field(ge=0)]
    resources: tuple[PhysicalResource, ...]
    sharing_groups: tuple[SharingGroup, ...] = ()
    operations: tuple[PhysicalOperation, ...]
    faults: tuple[TimedFault, ...] = ()
    counterfactuals: tuple[CounterfactualModifier, ...] = ()
    max_events: Annotated[int, Field(gt=0)] = 5_000_000
    max_operations: Annotated[int, Field(gt=0)] = 100_000


class ResourceMetrics(SimulationModel):
    resource_id: str
    busy_time_us: NonNegativeFloat
    utilization: Annotated[float, Field(ge=0.0, le=1.0)]
    transferred_bytes: Annotated[int, Field(ge=0)]
    max_concurrent: Annotated[int, Field(ge=0)]


class FabricSimulationMetrics(SimulationModel):
    operation_count: Annotated[int, Field(ge=0)]
    makespan_us: NonNegativeFloat
    total_work_us: NonNegativeFloat
    total_transferred_bytes: Annotated[int, Field(ge=0)]
    cost_usd: NonNegativeFloat
    processed_events: Annotated[int, Field(ge=0)]
    overlap_efficiency: Annotated[float, Field(ge=0.0, le=1.0)]
    predicted_lower_us: NonNegativeFloat
    predicted_upper_us: NonNegativeFloat
    resources: tuple[ResourceMetrics, ...]


class OperationOutcome(SimulationModel):
    operation_id: str
    status: Literal["completed"]
    start_us: NonNegativeFloat
    end_us: NonNegativeFloat
    duration_us: NonNegativeFloat
    base_duration_us: NonNegativeFloat
    wait_us: NonNegativeFloat
    transferred_bytes: Annotated[int, Field(ge=0)]
    uncertainty_us: NonNegativeFloat
    rank_ids: tuple[str, ...]
    resource_ids: tuple[str, ...]


class TraceEvent(SimulationModel):
    name: str
    cat: str
    ph: str
    ts: NonNegativeFloat
    dur: NonNegativeFloat
    pid: Annotated[int, Field(ge=0)]
    tid: str
    args: dict[str, JsonValue]


class SimulationProvenance(SimulationModel):
    simulator_version: str
    input_sha256: str
    seed: Annotated[int, Field(ge=0)]
    calibration_artifacts: tuple[str, ...]
    calibration_kinds: tuple[ProvenanceKind, ...]
    counterfactual_count: Annotated[int, Field(ge=0)]


class FabricSimulationOutput(SimulationModel):
    schema_version: Literal["1.0"]
    provenance: SimulationProvenance
    metrics: FabricSimulationMetrics
    operations: tuple[OperationOutcome, ...]
    trace_events: tuple[TraceEvent, ...]
    applied_faults: tuple[str, ...]
    applied_counterfactuals: tuple[CounterfactualModifier, ...]


class SimulationRequestShape(SimulationModel):
    arrival_us: NonNegativeFloat
    prompt_tokens: Annotated[int, Field(gt=0)]
    output_tokens: Annotated[int, Field(gt=0, le=4_096)]
    priority: Literal["high", "normal", "low"]
    request_class: str = Field(min_length=1)
    expert_skew_factor: Annotated[float, Field(ge=1.0, le=64.0)] = 1.0


class SimulationWorkload(SimulationModel):
    request_count: Annotated[int, Field(gt=0, le=10_000)]
    arrival_interval_us: NonNegativeFloat
    prompt_tokens: Annotated[int, Field(gt=0)]
    output_tokens: Annotated[int, Field(gt=0, le=4_096)]
    cpu_launch_us: PositiveFloat = 5.0
    requests: tuple[SimulationRequestShape, ...] = ()

    @model_validator(mode="after")
    def validate_requests(self) -> SimulationWorkload:
        if self.requests and len(self.requests) != self.request_count:
            raise ValueError("explicit request shapes must match request_count")
        arrivals = [request.arrival_us for request in self.requests]
        if arrivals != sorted(arrivals):
            raise ValueError("explicit request arrivals must be ordered")
        return self


class RequestLatency(SimulationModel):
    request_id: str
    ttft_us: NonNegativeFloat
    end_to_end_us: NonNegativeFloat


def _provenance(profile: FabricProfile, measured: bool) -> CalibrationProvenance:
    digest = hashlib.sha256(profile.model_dump_json().encode()).hexdigest()
    return CalibrationProvenance(
        kind=ProvenanceKind.MEASURED if measured else ProvenanceKind.ANALYTICAL,
        artifact_uri=profile.raw_artifacts[0].uri
        if profile.raw_artifacts
        else "inline://fabric-profile",
        artifact_sha256=digest,
        environment_fingerprint=profile.software_manifest.digest.value,
        collected_at=profile.created_at.isoformat(),
    )


def _default_curve(resource_id: str, profile: FabricProfile) -> ServiceCurve:
    return ServiceCurve(
        id=f"curve:{resource_id}",
        points=(
            ServiceCurvePoint(
                message_bytes=1,
                latency_us=1.0,
                bandwidth_gbps=1.0,
                uncertainty_fraction=0.25,
            ),
        ),
        provenance=_provenance(profile, measured=False),
    )


def _curve_from_edge(edge: TopologyEdge, profile: FabricProfile) -> ServiceCurve:
    latency = {point.message_bytes: point for point in edge.latency_curve_us}
    bandwidth = {point.message_bytes: point for point in edge.bandwidth_curve_gbps}
    sizes = sorted(set(latency) | set(bandwidth))
    measured = bool(sizes)
    if not sizes:
        sizes = [1, 1 << 20]
    points: list[ServiceCurvePoint] = []
    for size in sizes:
        latency_point = latency.get(size)
        bandwidth_point = bandwidth.get(size)
        gbps = (
            bandwidth_point.median
            if bandwidth_point is not None
            else (edge.theoretical_bandwidth_gbps or 1.0)
        )
        latency_us = latency_point.median if latency_point is not None else 2.0
        dispersion = max(
            latency_point.robust_dispersion if latency_point is not None else 0.0,
            bandwidth_point.robust_dispersion if bandwidth_point is not None else 0.0,
        )
        points.append(
            ServiceCurvePoint(
                message_bytes=size,
                latency_us=latency_us,
                bandwidth_gbps=gbps,
                uncertainty_fraction=min(1.0, dispersion / max(latency_us, 1e-9)),
            )
        )
    return ServiceCurve(
        id=f"curve:{edge.edge_id}",
        points=tuple(points),
        provenance=_provenance(profile, measured=measured),
    )


def _resource_kind(edge: TopologyEdge) -> ResourceKind | None:
    mapping = {
        ConnectionType.NVLINK: ResourceKind.NVLINK,
        ConnectionType.NVSWITCH: ResourceKind.NVSWITCH,
        ConnectionType.PCIE: ResourceKind.PCIE,
        ConnectionType.GPU_NIC: ResourceKind.PCIE,
        ConnectionType.NIC_NETWORK: ResourceKind.NETWORK_RAIL,
        ConnectionType.GPU_GPU: ResourceKind.GPU_COPY_ENGINE,
        ConnectionType.CPU_GPU: ResourceKind.PCIE,
        ConnectionType.CPU_MEMORY: ResourceKind.NUMA_MEMORY,
        ConnectionType.STORAGE_HOST: ResourceKind.STORAGE_PATH,
        ConnectionType.REMOTE_MEMORY: ResourceKind.NETWORK_RAIL,
    }
    return mapping.get(edge.connection)


def _link_resources(
    topology: TopologyGraph, profile: FabricProfile
) -> tuple[PhysicalResource, ...]:
    result: list[PhysicalResource] = []
    for edge in topology.edges:
        kind = _resource_kind(edge)
        if kind is None:
            continue
        result.append(
            PhysicalResource(
                id=edge.edge_id,
                kind=kind,
                scheduling=SchedulingMode.FAIR_SHARE,
                max_concurrency=64,
                curve=_curve_from_edge(edge, profile),
                sharing_group=edge.sharing_group,
            )
        )
    return tuple(result)


def _edge_latency_us(edge: TopologyEdge, message_bytes: int) -> float:
    base_latency = (
        min(
            edge.latency_curve_us,
            key=lambda item: abs(item.message_bytes - message_bytes),
        ).median
        if edge.latency_curve_us
        else 2.0
    )
    bandwidth_points = edge.bandwidth_curve_gbps
    bandwidth = (
        min(
            bandwidth_points,
            key=lambda item: abs(item.message_bytes - message_bytes),
        ).median
        if bandwidth_points
        else (edge.theoretical_bandwidth_gbps or 0.001)
    )
    return base_latency + message_bytes * 8.0 / (bandwidth * 1_000.0)


def _shortest_edge_path(
    topology: TopologyGraph, source: str, target: str, message_bytes: int
) -> tuple[tuple[str, ...], float]:
    adjacency: dict[str, list[tuple[str, TopologyEdge]]] = {}
    for edge in topology.edges:
        if edge.health is HealthState.FAILED:
            continue
        adjacency.setdefault(edge.source_node_id, []).append((edge.target_node_id, edge))
        if edge.directionality == "bidirectional":
            adjacency.setdefault(edge.target_node_id, []).append((edge.source_node_id, edge))
    pending: list[tuple[float, str, tuple[str, ...]]] = [(0.0, source, ())]
    best: dict[str, float] = {}
    while pending:
        pending.sort(key=lambda item: (item[0], item[1], item[2]))
        duration, node_id, path = pending.pop(0)
        if duration >= best.get(node_id, math.inf):
            continue
        best[node_id] = duration
        if node_id == target:
            return path, duration
        for neighbor, edge in sorted(adjacency.get(node_id, ()), key=lambda item: item[1].edge_id):
            pending.append(
                (
                    duration + _edge_latency_us(edge, message_bytes),
                    neighbor,
                    (*path, edge.edge_id),
                )
            )
    raise ValueError(f"collective ranks have no physical path from {source} to {target}")


def _collective_resource_ids(
    collective: CollectiveOperation,
    plan: PhysicalExecutionPlan,
    topology: TopologyGraph,
    resources: tuple[PhysicalResource, ...],
    message_bytes: int,
) -> tuple[str, ...]:
    """Lower a collective onto its slowest rank-order path, not the whole fabric.

    Collective algorithms progress at their critical participant path. The Rust
    model applies the ring/tree/pairwise algorithm factor to this serial path and
    then contends its sharing domains with other operations.
    """

    gpu_by_rank = {binding.rank_id: binding.gpu_id for binding in plan.rank_placement.bindings}
    ranks = collective.rank_order
    if collective.algorithm == "pairwise":
        pairs = tuple(
            (ranks[left], ranks[right])
            for left in range(len(ranks))
            for right in range(left + 1, len(ranks))
        )
    elif len(ranks) == 2:
        pairs = ((ranks[0], ranks[1]),)
    else:
        pairs = tuple(zip(ranks, (*ranks[1:], ranks[0]), strict=True))
    paths = tuple(
        _shortest_edge_path(
            topology,
            gpu_by_rank[left],
            gpu_by_rank[right],
            message_bytes,
        )
        for left, right in pairs
    )
    edge_path, _ = max(paths, key=lambda item: (item[1], item[0]))
    available = {resource.id for resource in resources}
    missing = tuple(edge_id for edge_id in edge_path if edge_id not in available)
    if missing:
        raise ValueError(f"collective path has no simulator resources: {missing}")
    return edge_path


def _lower_collective_before_kv(
    participating_ranks: set[int],
    prefill_ranks: set[int],
    decode_ranks: set[int],
) -> bool:
    """Return whether the operation belongs on the explicit prefill critical path.

    New compiler output keeps parallel groups inside a role-specific replica, so
    decode-only collectives remain represented by the calibrated opaque TPOT
    operation. Mixed-role collectives are accepted only for backwards-compatible
    v1 plans whose worker-role split predates whole-replica disaggregation.
    """

    if participating_ranks <= prefill_ranks:
        return True
    return bool(participating_ranks & prefill_ranks) and participating_ranks <= (
        prefill_ranks | decode_ranks
    )


def _pipeline_stage_by_rank(plan: PhysicalExecutionPlan) -> dict[int, int]:
    """Recover stable PP stage indices from the canonical pipeline groups."""

    stage_by_rank: dict[int, int] = {}
    for group in plan.parallelism.groups:
        if group.kind is not ParallelismKind.PIPELINE:
            continue
        for stage, rank_id in enumerate(group.rank_ids):
            existing = stage_by_rank.setdefault(rank_id, stage)
            if existing != stage:
                raise ValueError(f"rank {rank_id} has conflicting pipeline stages")
    if plan.parallelism.pipeline_parallel_degree == 1:
        return {binding.rank_id: 0 for binding in plan.rank_placement.bindings}
    rank_ids = {binding.rank_id for binding in plan.rank_placement.bindings}
    if set(stage_by_rank) != rank_ids:
        missing = tuple(sorted(rank_ids - set(stage_by_rank)))
        raise ValueError(f"pipeline plan does not assign stages for ranks {missing}")
    return stage_by_rank


def build_simulation_request(
    plan: PhysicalExecutionPlan,
    topology: TopologyGraph,
    profile: FabricProfile,
    workload: SimulationWorkload,
    *,
    seed: int,
    faults: tuple[TimedFault, ...] = (),
    counterfactuals: tuple[CounterfactualModifier, ...] = (),
) -> FabricSimulationRequest:
    """Lower a physical plan and workload to the stable Rust simulator protocol."""

    link_resources = _link_resources(topology, profile)
    resources: list[PhysicalResource] = list(link_resources)
    host_ids = sorted({binding.host_id for binding in plan.rank_placement.bindings})
    for host_id in host_ids:
        resources.append(
            PhysicalResource(
                id=f"cpu:{host_id}",
                kind=ResourceKind.CPU_CORE_GROUP,
                scheduling=SchedulingMode.FAIR_SHARE,
                max_concurrency=8,
                curve=_default_curve(f"cpu:{host_id}", profile),
            )
        )
    for binding in plan.rank_placement.bindings:
        resources.append(
            PhysicalResource(
                id=f"compute:{binding.gpu_id}",
                kind=ResourceKind.GPU_COMPUTE,
                scheduling=SchedulingMode.EXCLUSIVE,
                curve=_default_curve(f"compute:{binding.gpu_id}", profile),
            )
        )
    groups_by_id: dict[str, int] = {}
    for resource in resources:
        if resource.sharing_group is not None:
            groups_by_id[resource.sharing_group] = groups_by_id.get(resource.sharing_group, 0) + 1
    sharing_groups = tuple(
        SharingGroup(id=identifier, capacity_units=1.0, max_concurrency=max(1, count * 8))
        for identifier, count in sorted(groups_by_id.items())
    )
    operations: list[PhysicalOperation] = []
    uncertainty = 1.0 - plan.predicted_metrics.p95_ttft_ms.confidence
    pipeline_stage_by_rank = _pipeline_stage_by_rank(plan)
    pipeline_degree = plan.parallelism.pipeline_parallel_degree
    prefill_by_replica: dict[str, list[RankBinding]] = {}
    decode_by_replica: dict[str, list[RankBinding]] = {}
    for binding in plan.rank_placement.bindings:
        if binding.worker_role in {WorkerRole.PREFILL, WorkerRole.AGGREGATED}:
            prefill_by_replica.setdefault(binding.replica_id, []).append(binding)
        if binding.worker_role in {WorkerRole.DECODE, WorkerRole.AGGREGATED}:
            decode_by_replica.setdefault(binding.replica_id, []).append(binding)
    prefill_replicas = tuple(tuple(value) for _, value in sorted(prefill_by_replica.items()))
    decode_replicas = tuple(tuple(value) for _, value in sorted(decode_by_replica.items()))
    if not prefill_replicas or not decode_replicas:
        raise ValueError("physical plan must have eligible prefill and decode workers")
    planned_communication_us = (
        plan.predicted_metrics.communication_overhead_fraction.estimate
        * plan.predicted_metrics.p95_end_to_end_ms.estimate
        * 1_000.0
    )
    # Predicted TTFT is already a plan-level latency after TP/PP lowering. Each
    # participating rank executes its shard for that wall-clock duration; dividing
    # it by rank count here would apply model parallel speedup a second time. The
    # explicit communication DAG below contributes its own duration, so retain only
    # the compute component in the opaque GPU operation.
    baseline_prefill_per_rank_us = max(
        0.001,
        plan.predicted_metrics.p95_ttft_ms.estimate * 1_000.0 - planned_communication_us,
    )
    request_shapes = workload.requests or tuple(
        SimulationRequestShape(
            arrival_us=index * workload.arrival_interval_us,
            prompt_tokens=workload.prompt_tokens,
            output_tokens=workload.output_tokens,
            priority="normal",
            request_class="default",
        )
        for index in range(workload.request_count)
    )
    for request_index, shape in enumerate(request_shapes):
        request_id = f"request-{request_index:06d}"
        arrival_us = shape.arrival_us
        prefill_bindings = prefill_replicas[request_index % len(prefill_replicas)]
        decode_bindings = decode_replicas[request_index % len(decode_replicas)]
        active_prefill_ranks = {binding.rank_id for binding in prefill_bindings}
        active_decode_ranks = {binding.rank_id for binding in decode_bindings}
        prefill_per_stage_us = (
            baseline_prefill_per_rank_us
            * (shape.prompt_tokens / workload.prompt_tokens)
            / pipeline_degree
        )
        decode_per_stage_us = (
            plan.predicted_metrics.p99_tpot_ms.estimate
            * 1_000.0
            * shape.output_tokens
            / pipeline_degree
        )
        launch_by_rank: dict[int, str] = {}
        for binding in prefill_bindings:
            launch_id = f"{request_id}:rank-{binding.rank_id}:launch"
            launch_by_rank[binding.rank_id] = launch_id
            operations.append(
                PhysicalOperation(
                    id=launch_id,
                    kind=CpuLaunch(duration_us=workload.cpu_launch_us),
                    rank_ids=(f"rank-{binding.rank_id}",),
                    demands=(ResourceDemand(resource_id=f"cpu:{binding.host_id}"),),
                    earliest_start_us=arrival_us,
                    uncertainty_fraction=uncertainty,
                    request_id=request_id,
                )
            )
        previous_stage_ids: tuple[str, ...] = ()
        all_prefill_ids: list[str] = []
        for stage in range(pipeline_degree):
            stage_ids: list[str] = []
            for binding in prefill_bindings:
                if pipeline_stage_by_rank[binding.rank_id] != stage:
                    continue
                prefill_id = f"{request_id}:rank-{binding.rank_id}:prefill"
                operations.append(
                    PhysicalOperation(
                        id=prefill_id,
                        kind=GpuCompute(duration_us=max(0.001, prefill_per_stage_us)),
                        rank_ids=(f"rank-{binding.rank_id}",),
                        dependencies=(launch_by_rank[binding.rank_id], *previous_stage_ids),
                        demands=(ResourceDemand(resource_id=f"compute:{binding.gpu_id}"),),
                        earliest_start_us=arrival_us,
                        uncertainty_fraction=uncertainty,
                        request_id=request_id,
                    )
                )
                stage_ids.append(prefill_id)
                all_prefill_ids.append(prefill_id)
            if not stage_ids:
                raise ValueError(f"prefill replica has no rank for pipeline stage {stage}")
            previous_stage_ids = tuple(stage_ids)
        dependency_ids: tuple[str, ...] = tuple(all_prefill_ids)
        for collective_index, collective in enumerate(plan.collectives.operations):
            # Decode-side collectives are already represented by calibrated
            # opaque TPOT. Lowering them here as pre-KV operations would both
            # mis-stage and double-count communication.
            if not _lower_collective_before_kv(
                set(collective.participating_ranks),
                active_prefill_ranks,
                active_decode_ranks,
            ):
                continue
            operation_id = f"{request_id}:collective-{collective_index}"
            message_bytes = max(
                1,
                collective.message_size_intercept_bytes
                + round(collective.message_size_bytes_per_token * shape.prompt_tokens),
            )
            if collective.operation == "all_to_all":
                # Expert hot-spotting increases dispatch/combination traffic on
                # the collective critical path; dense TP/DP collectives retain
                # their calibrated message shape.
                message_bytes = max(1, round(message_bytes * shape.expert_skew_factor))
            resource_ids = _collective_resource_ids(
                collective,
                plan,
                topology,
                tuple(resources),
                message_bytes,
            )
            operations.append(
                PhysicalOperation(
                    id=operation_id,
                    kind=Collective(
                        collective_id=collective.operation_id,
                        bytes=message_bytes,
                        algorithm=collective.algorithm,
                        participating_ranks=tuple(
                            f"rank-{rank}" for rank in collective.participating_ranks
                        ),
                    ),
                    rank_ids=tuple(f"rank-{rank}" for rank in collective.participating_ranks),
                    dependencies=dependency_ids,
                    demands=tuple(ResourceDemand(resource_id=item) for item in resource_ids),
                    earliest_start_us=arrival_us,
                    uncertainty_fraction=uncertainty,
                    request_id=request_id,
                )
            )
            dependency_ids = (operation_id,)
        if plan.kv_transfer is not None:
            for route_index, route in enumerate(plan.kv_transfer.routes):
                if (
                    not set(route.producer_rank_ids) <= active_prefill_ranks
                    or not set(route.consumer_rank_ids) <= active_decode_ranks
                ):
                    continue
                operation_id = f"{request_id}:kv-{route_index}"
                kv_bytes = max(1, route.chunk_bytes * route.maximum_inflight_chunks)
                operations.append(
                    PhysicalOperation(
                        id=operation_id,
                        kind=KvTransfer(
                            bytes=kv_bytes,
                            chunks=route.maximum_inflight_chunks,
                        ),
                        rank_ids=tuple(
                            f"rank-{rank}"
                            for rank in (*route.producer_rank_ids, *route.consumer_rank_ids)
                        ),
                        dependencies=dependency_ids,
                        demands=tuple(
                            ResourceDemand(resource_id=edge_id) for edge_id in route.edge_path
                        ),
                        earliest_start_us=arrival_us,
                        uncertainty_fraction=uncertainty,
                        request_id=request_id,
                    )
                )
                dependency_ids = (operation_id,)
        previous_stage_ids = dependency_ids
        for stage in range(pipeline_degree):
            stage_ids = []
            for binding in decode_bindings:
                if pipeline_stage_by_rank[binding.rank_id] != stage:
                    continue
                operation_id = f"{request_id}:rank-{binding.rank_id}:decode"
                operations.append(
                    PhysicalOperation(
                        id=operation_id,
                        kind=GpuCompute(duration_us=max(0.001, decode_per_stage_us)),
                        rank_ids=(f"rank-{binding.rank_id}",),
                        dependencies=previous_stage_ids,
                        demands=(ResourceDemand(resource_id=f"compute:{binding.gpu_id}"),),
                        earliest_start_us=arrival_us,
                        uncertainty_fraction=uncertainty,
                        request_id=request_id,
                    )
                )
                stage_ids.append(operation_id)
            if not stage_ids:
                raise ValueError(f"decode replica has no rank for pipeline stage {stage}")
            previous_stage_ids = tuple(stage_ids)
    return FabricSimulationRequest(
        seed=seed,
        resources=tuple(resources),
        sharing_groups=sharing_groups,
        operations=tuple(operations),
        faults=faults,
        counterfactuals=counterfactuals,
    )


def _simulator_command(repository_root: Path) -> tuple[str, ...]:
    explicit = os.environ.get("SLOFORGE_FABRIC_SIM_BIN")
    if explicit:
        return (explicit, "simulate", "--compact")
    binary = repository_root / "target" / "debug" / "sloforge-fabric-sim"
    if binary.is_file():
        return (str(binary), "simulate", "--compact")
    return (
        "cargo",
        "run",
        "--quiet",
        "-p",
        "sloforge-fabric-sim",
        "--",
        "simulate",
        "--compact",
    )


def run_simulation(
    request: FabricSimulationRequest,
    *,
    repository_root: Path,
    timeout_seconds: float = 30.0,
) -> FabricSimulationOutput:
    """Execute the bounded Rust subprocess without transport fallback."""

    command = _simulator_command(repository_root)
    payload = request.model_dump_json().encode()
    if len(payload) > _MAX_SIMULATOR_INPUT_BYTES:
        raise RuntimeError("fabric simulator request exceeds the 64 MiB subprocess boundary")
    try:
        completed = subprocess.run(
            command,
            cwd=repository_root,
            input=payload,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"fabric simulator timed out after {timeout_seconds:g}s") from error
    if len(completed.stdout) > _MAX_SIMULATOR_OUTPUT_BYTES:
        raise RuntimeError("fabric simulator response exceeds the 128 MiB boundary")
    if completed.returncode != 0:
        clipped = completed.stderr[:_MAX_SIMULATOR_ERROR_BYTES]
        suffix = (
            "\n[stderr truncated]" if len(completed.stderr) > _MAX_SIMULATOR_ERROR_BYTES else ""
        )
        stderr = clipped.decode(errors="replace").strip() + suffix
        raise RuntimeError(f"fabric simulator failed ({completed.returncode}): {stderr}")
    try:
        return FabricSimulationOutput.model_validate_json(completed.stdout)
    except ValueError as error:
        raise RuntimeError("fabric simulator emitted an invalid response") from error


def request_latencies(output: FabricSimulationOutput) -> tuple[RequestLatency, ...]:
    """Derive per-request TTFT and E2E metrics from actual operation events."""

    grouped: dict[str, list[OperationOutcome]] = {}
    for operation in output.operations:
        request_id, separator, _ = operation.operation_id.partition(":")
        if not separator:
            continue
        grouped.setdefault(request_id, []).append(operation)
    result: list[RequestLatency] = []
    for request_id, operations in sorted(grouped.items()):
        start = min(operation.start_us for operation in operations)
        prefill_end = max(
            (operation.end_us for operation in operations if ":prefill" in operation.operation_id),
            default=start,
        )
        first_decode_start = min(
            (
                operation.start_us
                for operation in operations
                if operation.operation_id.endswith(":decode")
            ),
            default=prefill_end,
        )
        end = max(operation.end_us for operation in operations)
        result.append(
            RequestLatency(
                request_id=request_id,
                # Prefill completion is not a first-token boundary for a
                # disaggregated plan: collectives and KV transfer must finish
                # before decode can start.
                ttft_us=first_decode_start - start,
                end_to_end_us=end - start,
            )
        )
    return tuple(result)
