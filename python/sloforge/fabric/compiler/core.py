"""Deterministic hierarchical compiler for physical inference plans.

The compiler deliberately uses inspectable analytical lower bounds and measured
fabric curves.  It does not hide placement decisions behind an opaque surrogate:
every rejected candidate and every promotion is retained in the physical IR.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import statistics
import time
from collections import defaultdict
from enum import StrEnum
from functools import cache
from pathlib import Path
from typing import Annotated, Final, Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from sloforge.fabric.ir import (
    CollectiveOperation,
    CollectivePlan,
    CommunicationOverlapPlan,
    DocumentReference,
    ExpertAssignment,
    ExpertPlacement,
    FabricProfile,
    FailureExposure,
    GpuNode,
    HealthState,
    KVTransferPlan,
    KVTransferRoute,
    MemoryPlan,
    MetricInterval,
    ModelGraph,
    NicNode,
    NumaDomainNode,
    OptimizerTraceEntry,
    OverlapWindow,
    ParallelGroup,
    ParallelismKind,
    ParallelismPlan,
    PhysicalExecutionPlan,
    PhysicalMetrics,
    RankBinding,
    RankMemoryAllocation,
    RankPlacement,
    RecoveryTrigger,
    RecoveryVariant,
    RejectedPhysicalCandidate,
    ReproducibilityMetadata,
    TopologyEdge,
    TopologyGraph,
    WorkerRole,
    canonical_hash,
)
from sloforge.ir import ArtifactDigest

_COMPILER_VERSION: Final = "0.1.0"

PositiveFinite = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]
NonNegativeFinite = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
PositiveInt = Annotated[int, Field(gt=0)]


class CompilerModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class CompilerObjective(StrEnum):
    MINIMIZE_COST = "minimize_cost"
    MINIMIZE_LATENCY = "minimize_latency"
    MAXIMIZE_GOODPUT = "maximize_goodput"
    ROBUST_BALANCED = "robust_balanced"


class OptimizationStrategy(StrEnum):
    EXHAUSTIVE = "exhaustive"
    RANDOM_PLACEMENT = "random_placement"
    TOPOLOGY_UNAWARE = "topology_unaware"
    GREEDY_TOPOLOGY_AWARE = "greedy_topology_aware"
    HIERARCHICAL = "hierarchical"
    ROBUST_FAILURE = "robust_failure"


class CompilerAssumptions(CompilerModel):
    """Calibrated compute and economic inputs not represented by fabric links."""

    prefill_tokens_per_second_per_gpu: PositiveFinite
    decode_tokens_per_second_per_gpu: PositiveFinite
    gpu_hourly_price_usd: NonNegativeFinite
    base_availability: Annotated[float, Field(gt=0.0, le=1.0, allow_inf_nan=False)]
    cold_start_ms: NonNegativeFinite
    measurement_relative_uncertainty: Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]


class CompilerConstraints(CompilerModel):
    prompt_tokens_p95: PositiveInt
    output_tokens_p95: PositiveInt
    maximum_concurrent_requests: PositiveInt
    p95_ttft_ms: PositiveFinite
    p99_tpot_ms: PositiveFinite
    minimum_goodput_tokens_per_second: NonNegativeFinite = 0.0
    minimum_availability: Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)] = 0.0
    maximum_cost_per_million_tokens: PositiveFinite | None = None
    maximum_ranks: PositiveInt
    tensor_parallel_degree: PositiveInt | None = None
    pipeline_parallel_degree: PositiveInt | None = None
    data_parallel_degree: PositiveInt | None = None
    expert_parallel_degree: PositiveInt | None = None
    memory_safety_fraction: Annotated[float, Field(gt=0.0, lt=0.5, allow_inf_nan=False)] = 0.10
    require_disaggregation: bool = False
    permitted_transports: tuple[
        Literal["shared_memory", "nvlink", "pcie", "infiniband", "roce", "tcp"], ...
    ] = ("shared_memory", "nvlink", "pcie", "infiniband", "roce", "tcp")

    @model_validator(mode="after")
    def validate_fixed_parallelism(self) -> Self:
        fixed_rank_count = math.prod(
            degree
            for degree in (
                self.tensor_parallel_degree,
                self.pipeline_parallel_degree,
                self.data_parallel_degree,
            )
            if degree is not None
        )
        if fixed_rank_count > self.maximum_ranks:
            raise ValueError("fixed TP x PP x DP exceeds maximum_ranks")
        return self


class CompilerRequest(CompilerModel):
    logical_deployment_plan: DocumentReference
    model: ModelGraph
    topology: TopologyGraph
    fabric_profile: FabricProfile
    constraints: CompilerConstraints
    assumptions: CompilerAssumptions
    objective: CompilerObjective
    strategy: OptimizationStrategy = OptimizationStrategy.HIERARCHICAL
    generated_at: AwareDatetime
    seed: Annotated[int, Field(ge=0)] = 0
    git_commit: str = Field(min_length=1)
    environment_digest: ArtifactDigest

    @model_validator(mode="after")
    def validate_profile_topology(self) -> Self:
        # Profiles carry the fingerprint used during measurement. Comparing the
        # content hash closes the class of accidental stale-profile compiles.
        actual = ArtifactDigest(value=canonical_hash(self.topology))
        if self.fabric_profile.topology_fingerprint != actual:
            raise ValueError("fabric profile does not match topology content hash")
        return self


class CandidateSummary(CompilerModel):
    candidate_id: str
    tensor_parallel: PositiveInt
    pipeline_parallel: PositiveInt
    data_parallel: PositiveInt
    expert_parallel: PositiveInt
    disaggregated: bool
    gpu_ids: tuple[str, ...]
    communication_us: NonNegativeFinite
    p95_ttft_ms: NonNegativeFinite
    p99_tpot_ms: NonNegativeFinite
    goodput_tokens_per_second: NonNegativeFinite
    cost_per_million_tokens: NonNegativeFinite
    availability: Annotated[float, Field(ge=0.0, le=1.0)]
    failure_exposure_score: NonNegativeFinite
    objective_score: float = Field(allow_inf_nan=False)
    feasible: bool
    rejection_codes: tuple[str, ...]


class PhysicalCompileResult(CompilerModel):
    selected: PhysicalExecutionPlan
    pareto_frontier: tuple[CandidateSummary, ...]
    all_candidates: tuple[CandidateSummary, ...]
    strategy: OptimizationStrategy
    # This outer diagnostic is intentionally excluded from PhysicalExecutionPlan
    # canonical serialization. It is the actual local elapsed wall time.
    solver_time_ms: NonNegativeFinite
    simulator_calls: Annotated[int, Field(ge=0)]
    simulator_validated_candidate_ids: tuple[str, ...]
    deterministic_solver_work_units: Annotated[int, Field(ge=0)]


class _Candidate(CompilerModel):
    candidate_id: str
    tp: PositiveInt
    pp: PositiveInt
    dp: PositiveInt
    ep: PositiveInt
    disaggregated: bool
    gpu_ids: tuple[str, ...]
    placement: RankPlacement
    memory: MemoryPlan
    collectives: CollectivePlan
    kv_transfer: KVTransferPlan | None
    overlap: CommunicationOverlapPlan
    summary: CandidateSummary


def _digest(text: str) -> ArtifactDigest:
    return ArtifactDigest(value=hashlib.sha256(text.encode()).hexdigest())


def _healthy_gpus(topology: TopologyGraph) -> tuple[GpuNode, ...]:
    return tuple(
        sorted(
            (
                node
                for node in topology.nodes
                if isinstance(node, GpuNode) and node.health is not HealthState.FAILED
            ),
            key=lambda gpu: (gpu.host_id, gpu.gpu_index, gpu.node_id),
        )
    )


def _edge_bandwidth(edge: TopologyEdge) -> float:
    measured = tuple(point.median for point in edge.bandwidth_curve_gbps)
    if measured:
        return min(measured)
    if edge.theoretical_bandwidth_gbps is not None:
        return edge.theoretical_bandwidth_gbps
    return 0.001


def _edge_latency_us(edge: TopologyEdge, message_bytes: int) -> float:
    points = sorted(edge.latency_curve_us, key=lambda point: point.message_bytes)
    base_latency = (
        min(points, key=lambda point: abs(point.message_bytes - message_bytes)).median
        if points
        else 2.0
    )
    return base_latency + message_bytes * 8.0 / (_edge_bandwidth(edge) * 1_000.0)


def _adjacency(topology: TopologyGraph) -> dict[str, list[tuple[str, TopologyEdge]]]:
    result: dict[str, list[tuple[str, TopologyEdge]]] = defaultdict(list)
    for edge in topology.edges:
        if edge.health is HealthState.FAILED:
            continue
        result[edge.source_node_id].append((edge.target_node_id, edge))
        if edge.directionality == "bidirectional":
            result[edge.target_node_id].append((edge.source_node_id, edge))
    return result


def _shortest_path(
    topology: TopologyGraph, source: str, target: str, message_bytes: int
) -> tuple[tuple[str, ...], float, float]:
    """Return edge IDs, additive latency, and bottleneck bandwidth."""

    if source == target:
        return (), 0.0, math.inf
    graph = _adjacency(topology)
    pending: list[tuple[float, str, tuple[str, ...], float]] = [(0.0, source, (), math.inf)]
    best: dict[str, float] = {}
    while pending:
        pending.sort(key=lambda item: (item[0], item[1], item[2]))
        latency, node_id, path, bandwidth = pending.pop(0)
        if latency >= best.get(node_id, math.inf):
            continue
        best[node_id] = latency
        if node_id == target:
            return path, latency, bandwidth
        for neighbor, edge in sorted(graph.get(node_id, ()), key=lambda item: item[1].edge_id):
            edge_latency = _edge_latency_us(edge, message_bytes)
            pending.append(
                (
                    latency + edge_latency,
                    neighbor,
                    (*path, edge.edge_id),
                    min(bandwidth, _edge_bandwidth(edge)),
                )
            )
    return (), math.inf, 0.0


def _pair_score(topology: TopologyGraph, left: GpuNode, right: GpuNode) -> float:
    _, latency, bandwidth = _shortest_path(topology, left.node_id, right.node_id, 1 << 20)
    host_bonus = 100.0 if left.host_id == right.host_id else 0.0
    locality_bonus = 20.0 if left.numa_domain_id == right.numa_domain_id else 0.0
    return host_bonus + locality_bonus + bandwidth - latency


def _ordered_gpus(
    topology: TopologyGraph,
    count: int,
    strategy: OptimizationStrategy,
    excluded: frozenset[str] = frozenset(),
    seed: int = 0,
) -> tuple[GpuNode, ...]:
    available = tuple(gpu for gpu in _healthy_gpus(topology) if gpu.node_id not in excluded)
    if len(available) < count:
        return ()
    if strategy is OptimizationStrategy.TOPOLOGY_UNAWARE:
        return available[:count]
    if strategy is OptimizationStrategy.RANDOM_PLACEMENT:
        # Content-hash ordering gives a reproducible seeded random baseline
        # independent of process hash randomization and Python container order.
        randomized = sorted(
            available,
            key=lambda gpu: hashlib.sha256(f"{seed}:{gpu.node_id}".encode()).digest(),
        )
        return tuple(randomized[:count])
    # Deterministic greedy maximum-affinity expansion. The first GPU is chosen
    # by fault-domain stability; subsequent ranks maximize their weakest link to
    # the already selected group.
    start = max(
        available,
        key=lambda gpu: (
            sum(_pair_score(topology, gpu, other) for other in available if other != gpu),
            tuple(-ord(char) for char in gpu.node_id),
        ),
    )
    selected = [start]
    while len(selected) < count:
        remaining = [gpu for gpu in available if gpu not in selected]
        chosen = max(
            remaining,
            key=lambda gpu: (
                min(_pair_score(topology, gpu, member) for member in selected),
                sum(_pair_score(topology, gpu, member) for member in selected),
                tuple(-ord(char) for char in gpu.node_id),
            ),
        )
        selected.append(chosen)
    return tuple(selected)


def _ordered_candidate_gpus(
    topology: TopologyGraph,
    *,
    tp: int,
    pp: int,
    dp: int,
    strategy: OptimizationStrategy,
    seed: int,
    excluded: frozenset[str] = frozenset(),
) -> tuple[GpuNode, ...]:
    """Place complete replica groups before optimizing order within each group."""

    rank_count = tp * pp * dp
    if strategy in {
        OptimizationStrategy.RANDOM_PLACEMENT,
        OptimizationStrategy.TOPOLOGY_UNAWARE,
    }:
        return _ordered_gpus(topology, rank_count, strategy, excluded, seed)
    replica_size = tp * pp
    selected: list[GpuNode] = []
    excluded_ids = set(excluded)
    host_replica_counts: dict[str, int] = defaultdict(int)
    for _ in range(dp):
        replica: tuple[GpuNode, ...] = ()
        if strategy is OptimizationStrategy.ROBUST_FAILURE:
            # Prefer a complete replica on the least-used host fault domain.
            # This differs deliberately from latency-first greedy placement,
            # while avoiding the worse failure mode of spreading every TP/PP
            # replica across all hosts.
            candidates: list[tuple[int, float, str, tuple[GpuNode, ...]]] = []
            host_ids = sorted({gpu.host_id for gpu in _healthy_gpus(topology)})
            for host_id in host_ids:
                other_host_ids = {
                    gpu.node_id for gpu in _healthy_gpus(topology) if gpu.host_id != host_id
                }
                host_group = _ordered_gpus(
                    topology,
                    replica_size,
                    OptimizationStrategy.GREEDY_TOPOLOGY_AWARE,
                    frozenset(excluded_ids | other_host_ids),
                    seed,
                )
                if len(host_group) != replica_size:
                    continue
                affinity = sum(
                    _pair_score(topology, left, right)
                    for left, right in itertools.combinations(host_group, 2)
                )
                candidates.append((host_replica_counts[host_id], -affinity, host_id, host_group))
            if candidates:
                _, _, host_id, replica = min(candidates)
                host_replica_counts[host_id] += 1
        if not replica:
            replica = _ordered_gpus(
                topology,
                replica_size,
                OptimizationStrategy.GREEDY_TOPOLOGY_AWARE,
                frozenset(excluded_ids),
                seed,
            )
        if len(replica) != replica_size:
            return ()
        selected.extend(replica)
        excluded_ids.update(gpu.node_id for gpu in replica)
    return tuple(selected)


def _nearest_nic(topology: TopologyGraph, gpu: GpuNode) -> tuple[str | None, str | None]:
    nics = tuple(node for node in topology.nodes if isinstance(node, NicNode) and node.active)
    if not nics:
        return None, None
    same_host = tuple(nic for nic in nics if nic.host_id == gpu.host_id)
    candidates = same_host or nics
    nic = max(
        candidates,
        key=lambda item: (
            item.numa_domain_id == gpu.numa_domain_id,
            item.speed_gbps or 0.0,
            tuple(-ord(char) for char in item.node_id),
        ),
    )
    rails: list[tuple[float, str]] = []
    for edge in topology.edges:
        if edge.source_node_id == nic.node_id:
            target = next(
                (node for node in topology.nodes if node.node_id == edge.target_node_id), None
            )
            if target is not None and getattr(target, "kind", None) == "network_rail":
                rails.append((_edge_bandwidth(edge), target.node_id))
        if edge.target_node_id == nic.node_id:
            source = next(
                (node for node in topology.nodes if node.node_id == edge.source_node_id), None
            )
            if source is not None and getattr(source, "kind", None) == "network_rail":
                rails.append((_edge_bandwidth(edge), source.node_id))
    rail = max(rails, default=(0.0, None), key=lambda item: (item[0], item[1] or ""))[1]
    return nic.node_id, rail


def _numa_cpu_set(topology: TopologyGraph, numa_id: str | None) -> str:
    for node in topology.nodes:
        if node.node_id == numa_id and isinstance(node, NumaDomainNode):
            return str(node.cpu_set)
    return "0"


def _placement(
    topology: TopologyGraph,
    gpus: tuple[GpuNode, ...],
    *,
    tp: int,
    pp: int,
    dp: int,
    disaggregated: bool,
) -> RankPlacement:
    bindings: list[RankBinding] = []
    replica_size = tp * pp
    # Prefill/decode disaggregation is a role split between complete model
    # replicas. Splitting a TP or PP group would create ranks that cannot run a
    # complete collective/stage graph in either worker pool.
    prefill_replica_count = max(1, dp // 2) if disaggregated else 0
    for rank, gpu in enumerate(gpus):
        nic_id, rail_id = _nearest_nic(topology, gpu)
        if disaggregated:
            replica = rank // replica_size
            role = WorkerRole.PREFILL if replica < prefill_replica_count else WorkerRole.DECODE
        else:
            role = WorkerRole.AGGREGATED
        bindings.append(
            RankBinding(
                rank_id=rank,
                host_id=gpu.host_id,
                gpu_id=gpu.node_id,
                numa_domain_id=gpu.numa_domain_id or f"unknown-numa-{gpu.host_id}",
                nic_id=nic_id,
                network_rail_id=rail_id,
                process_cpu_affinity=_numa_cpu_set(topology, gpu.numa_domain_id),
                worker_role=role,
                replica_id=f"replica-{rank // replica_size}",
                fault_domain=gpu.host_id,
            )
        )
    return RankPlacement(bindings=tuple(bindings))


def _parallelism(tp: int, pp: int, dp: int, ep: int, disaggregated: bool) -> ParallelismPlan:
    groups: list[ParallelGroup] = []
    replica_size = tp * pp
    for replica in range(dp):
        base = replica * replica_size
        for stage in range(pp):
            ranks = tuple(base + stage * tp + shard for shard in range(tp))
            if len(ranks) > 1:
                groups.append(
                    ParallelGroup(
                        group_id=f"tp-{replica}-{stage}",
                        kind=ParallelismKind.TENSOR,
                        rank_ids=ranks,
                    )
                )
        if pp > 1:
            for shard in range(tp):
                groups.append(
                    ParallelGroup(
                        group_id=f"pp-{replica}-{shard}",
                        kind=ParallelismKind.PIPELINE,
                        rank_ids=tuple(base + stage * tp + shard for stage in range(pp)),
                    )
                )
    if dp > 1:
        for local_rank in range(replica_size):
            groups.append(
                ParallelGroup(
                    group_id=f"dp-{local_rank}",
                    kind=ParallelismKind.DATA,
                    rank_ids=tuple(replica * replica_size + local_rank for replica in range(dp)),
                )
            )
    rank_count = tp * pp * dp
    if ep > 1:
        # EP groups never cross a data-replica boundary. A group spanning two
        # replicas would turn independent fault/routing units into one mandatory
        # collective domain.
        for replica in range(dp):
            base = replica * replica_size
            for local_start in range(0, replica_size, ep):
                ranks = tuple(range(base + local_start, base + local_start + ep))
                groups.append(
                    ParallelGroup(
                        group_id=f"ep-{replica}-{local_start // ep}",
                        kind=ParallelismKind.EXPERT,
                        rank_ids=ranks,
                    )
                )
    if disaggregated:
        split = max(1, dp // 2) * replica_size
        groups.extend(
            (
                ParallelGroup(
                    group_id="prefill-pool",
                    kind=ParallelismKind.PREFILL,
                    rank_ids=tuple(range(split)),
                ),
                ParallelGroup(
                    group_id="decode-pool",
                    kind=ParallelismKind.DECODE,
                    rank_ids=tuple(range(split, rank_count)),
                ),
            )
        )
    replicas = tuple(
        ParallelGroup(
            group_id=f"replica-{replica}",
            kind=ParallelismKind.DATA,
            rank_ids=tuple(range(replica * replica_size, (replica + 1) * replica_size)),
        )
        for replica in range(dp)
    )
    return ParallelismPlan(
        tensor_parallel_degree=tp,
        pipeline_parallel_degree=pp,
        data_parallel_degree=dp,
        expert_parallel_degree=ep,
        prefill_decode_disaggregated=disaggregated,
        groups=tuple(groups),
        replica_groups=replicas,
    )


def _memory_plan(
    model: ModelGraph,
    gpus: tuple[GpuNode, ...],
    constraints: CompilerConstraints,
    *,
    tp: int,
    pp: int,
    dp: int,
) -> MemoryPlan:
    precision = model.precision_modes[0]
    shard_count = tp * pp
    weights = math.ceil(precision.weight_bytes / shard_count)
    kv_per_request = sum(layer.kv_bytes_per_token for layer in model.layers) * (
        constraints.prompt_tokens_p95 + constraints.output_tokens_p95
    )
    # Data replicas duplicate state; they do not shard one replica's KV cache.
    # TP and PP partition heads/layers, respectively, so each rank owns only its
    # corresponding shard of the per-replica cache.
    kv_cache = math.ceil(kv_per_request * constraints.maximum_concurrent_requests / shard_count)
    activations = math.ceil(
        sum(layer.activation_bytes_per_token for layer in model.layers)
        * constraints.maximum_concurrent_requests
        / shard_count
    )
    allocations: list[RankMemoryAllocation] = []
    for rank, gpu in enumerate(gpus):
        runtime = 256 * 1024 * 1024
        communication = 64 * 1024 * 1024
        subtotal = weights + kv_cache + activations + runtime + communication
        fragmentation = math.ceil(subtotal * constraints.memory_safety_fraction / 2.0)
        safety = math.ceil(subtotal * constraints.memory_safety_fraction / 2.0)
        allocations.append(
            RankMemoryAllocation(
                rank_id=rank,
                capacity_bytes=gpu.memory_bytes,
                weights_bytes=weights,
                kv_cache_bytes=kv_cache,
                activations_bytes=activations,
                cuda_graph_bytes=0,
                runtime_workspace_bytes=runtime,
                communication_buffers_bytes=communication,
                host_pinned_buffers_bytes=64 * 1024 * 1024,
                local_nvme_bytes=0,
                remote_artifacts_bytes=0,
                fragmentation_allowance_bytes=fragmentation,
                safety_margin_bytes=safety,
            )
        )
    return MemoryPlan(allocations=tuple(allocations))


def _transport_for_ranks(
    topology: TopologyGraph, placement: RankPlacement, ranks: tuple[int, ...]
) -> tuple[str, tuple[str, ...], tuple[str, ...], float]:
    bindings = {binding.rank_id: binding for binding in placement.bindings}
    edges_by_id = {edge.edge_id: edge for edge in topology.edges}
    paths: list[str] = []
    total_latency = 0.0
    transports: set[str] = set()
    rails: set[str] = set()
    for left, right in zip(ranks, (*ranks[1:], ranks[0]), strict=True):
        left_binding = bindings[left]
        right_binding = bindings[right]
        edge_path, latency, _ = _shortest_path(
            topology, left_binding.gpu_id, right_binding.gpu_id, 1 << 20
        )
        total_latency += latency
        paths.extend(edge_path)
        for edge_id in edge_path:
            edge = edges_by_id[edge_id]
            if edge.connection.value in {"nvlink", "nvswitch"}:
                transports.add("nvlink")
            elif edge.connection.value == "pcie":
                transports.add("pcie")
            elif edge.connection.value == "nic_network":
                rail = left_binding.network_rail_id or right_binding.network_rail_id
                if rail is not None:
                    rails.add(rail)
                nic = next(
                    (
                        node
                        for node in topology.nodes
                        if isinstance(node, NicNode)
                        and node.node_id in {edge.source_node_id, edge.target_node_id}
                    ),
                    None,
                )
                transports.add(nic.transport if nic is not None else "tcp")
    priority = ("infiniband", "roce", "tcp", "pcie", "nvlink", "shared_memory")
    transport = next((name for name in priority if name in transports), "shared_memory")
    return transport, tuple(sorted(rails)), tuple(paths), total_latency


def _profile_duration(
    profile: FabricProfile, primitive: str, transport: str, ranks: int, message_bytes: int
) -> tuple[float, float]:
    matches = tuple(
        series
        for series in profile.measurements
        if series.primitive == primitive
        and series.transport == transport
        and series.rank_count == ranks
    )
    if not matches:
        return 0.0, 1.0
    grouped: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for point in matches:
        grouped[point.message_bytes].append(
            (
                point.summary_median_us,
                max(
                    point.summary_median_us - point.confidence_low_us,
                    point.confidence_high_us - point.summary_median_us,
                ),
            )
        )
    # Several concrete collective primitives share the canonical "collective"
    # category. Collapse repeated sizes before interpolation; using same-size
    # observations as extrapolation points creates a zero byte span and turns
    # ordinary sample variance into an unbounded slope.
    points = tuple(
        (
            size,
            statistics.median(value[0] for value in values),
            statistics.median(value[1] for value in values),
        )
        for size, values in sorted(grouped.items())
    )

    if message_bytes <= points[0][0] or len(points) == 1:
        return points[0][1], points[0][2]
    for index, right in enumerate(points[1:], start=1):
        if message_bytes <= right[0]:
            left = points[index - 1]
            ratio = (message_bytes - left[0]) / (right[0] - left[0])
            return (
                left[1] + (right[1] - left[1]) * ratio,
                left[2] + (right[2] - left[2]) * ratio,
            )
    # Extrapolate the marginal serialization slope, not the entire duration;
    # multiplying a measured median would incorrectly scale fixed launch latency.
    previous = points[-2]
    last = points[-1]
    byte_span = last[0] - previous[0]
    extra_bytes = message_bytes - last[0]
    return (
        last[1] + max(0.0, last[1] - previous[1]) / byte_span * extra_bytes,
        last[2] + max(0.0, last[2] - previous[2]) / byte_span * extra_bytes,
    )


def _collectives(
    request: CompilerRequest,
    parallelism: ParallelismPlan,
    placement: RankPlacement,
) -> tuple[CollectivePlan, CommunicationOverlapPlan, float, str | None]:
    operations: list[CollectiveOperation] = []
    overlap: list[OverlapWindow] = []
    total_us = 0.0
    sequence = 0
    prompt = request.constraints.prompt_tokens_p95
    requirements = tuple(
        requirement for layer in request.model.layers for requirement in layer.communication
    )
    for requirement in requirements:
        kind = ParallelismKind(requirement.parallelism_dimension)
        groups = tuple(group for group in parallelism.groups if group.kind is kind)
        replica_durations_us: list[float] = []
        for group in groups:
            if len(group.rank_ids) < 2:
                continue
            message_bytes = max(1, round(requirement.bytes_per_token * prompt))
            transport, rails, _, path_us = _transport_for_ranks(
                request.topology, placement, group.rank_ids
            )
            if not math.isfinite(path_us):
                return (
                    CollectivePlan(operations=tuple(operations)),
                    CommunicationOverlapPlan(windows=tuple(overlap)),
                    0.0,
                    "collective_path_unreachable",
                )
            if transport not in request.constraints.permitted_transports:
                return (
                    CollectivePlan(operations=tuple(operations)),
                    CommunicationOverlapPlan(windows=tuple(overlap)),
                    0.0,
                    "collective_transport_not_permitted",
                )
            measured_us, uncertainty_us = _profile_duration(
                request.fabric_profile,
                "collective",
                transport,
                len(group.rank_ids),
                message_bytes,
            )
            duration_us = measured_us if measured_us > 0 else path_us
            duration_us = max(duration_us, 0.001)
            operation_id = f"collective-{sequence:04d}"
            algorithm: Literal["ring", "tree", "recursive_doubling", "direct", "pairwise", "auto"]
            if requirement.operation == "all_to_all":
                algorithm = "pairwise"
            elif len(group.rank_ids) == 2:
                algorithm = "direct"
            else:
                algorithm = "ring"
            window_id = f"overlap-{sequence:04d}"
            operations.append(
                CollectiveOperation(
                    operation_id=operation_id,
                    operation=requirement.operation,
                    participating_ranks=group.rank_ids,
                    message_size_intercept_bytes=0,
                    message_size_bytes_per_token=requirement.bytes_per_token,
                    algorithm=algorithm,
                    transport=transport,
                    channel_count=max(1, min(8, len(group.rank_ids))),
                    rail_ids=rails,
                    rank_order=group.rank_ids,
                    expected_duration_us=duration_us,
                    uncertainty_us=uncertainty_us,
                    overlap_window_id=window_id,
                    fallback="host_staged" if transport != "tcp" else "serialize",
                )
            )
            overlap_fraction = 0.35 if not requirement.synchronization_required else 0.10
            overlap.append(
                OverlapWindow(
                    window_id=window_id,
                    compute_operation_id=f"layer-{sequence:04d}",
                    communication_operation_id=operation_id,
                    stream="communication",
                    expected_overlap_fraction=overlap_fraction,
                    resource_contention="network" if rails else "copy_engine",
                    fallback_serialization="critical_path",
                )
            )
            replica_durations_us.append(duration_us * (1.0 - overlap_fraction))
            sequence += 1
        # Data replicas execute independently. The request critical path sees
        # one group's collective (or the slowest group under robust planning),
        # not the sum of collectives across every replica.
        total_us += max(replica_durations_us, default=0.0)
    return (
        CollectivePlan(operations=tuple(operations)),
        CommunicationOverlapPlan(windows=tuple(overlap)),
        total_us,
        None,
    )


def _kv_transfer(
    request: CompilerRequest, placement: RankPlacement, disaggregated: bool
) -> tuple[KVTransferPlan | None, float, str | None]:
    if not disaggregated:
        return None, 0.0, None
    prefill_replicas: dict[str, list[RankBinding]] = defaultdict(list)
    decode_replicas: dict[str, list[RankBinding]] = defaultdict(list)
    for binding in placement.bindings:
        if binding.worker_role is WorkerRole.PREFILL:
            prefill_replicas[binding.replica_id].append(binding)
        elif binding.worker_role is WorkerRole.DECODE:
            decode_replicas[binding.replica_id].append(binding)
    if not prefill_replicas or not decode_replicas:
        return None, 0.0, "kv_transfer_workers_unavailable"
    kv_bytes = max(
        1,
        sum(layer.kv_bytes_per_token for layer in request.model.layers)
        * request.constraints.prompt_tokens_p95,
    )
    prefill_groups = tuple(tuple(value) for _, value in sorted(prefill_replicas.items()))
    decode_groups = tuple(tuple(value) for _, value in sorted(decode_replicas.items()))
    routes: list[KVTransferRoute] = []
    route_latencies: list[float] = []
    chunk_bytes = min(kv_bytes, 2 * 1024 * 1024)
    chunk_count = math.ceil(kv_bytes / chunk_bytes)
    for producer_index, producers in enumerate(prefill_groups):
        for consumer_index, consumers in enumerate(decode_groups):
            producer = producers[0]
            consumer = consumers[0]
            path, path_us, _ = _shortest_path(
                request.topology, producer.gpu_id, consumer.gpu_id, kv_bytes
            )
            if not path or not math.isfinite(path_us):
                return None, 0.0, "kv_transfer_path_unreachable"
            rank_pair = (producer.rank_id, consumer.rank_id)
            transport, _, _, _ = _transport_for_ranks(request.topology, placement, rank_pair)
            if transport not in request.constraints.permitted_transports:
                return None, 0.0, "kv_transfer_transport_not_permitted"
            serialization: Literal["raw", "paged", "nixl", "runtime_native"] = (
                "nixl" if transport in {"infiniband", "roce"} else "paged"
            )
            routes.append(
                KVTransferRoute(
                    route_id=f"kv-prefill-{producer_index}-decode-{consumer_index}",
                    producer_rank_ids=tuple(binding.rank_id for binding in producers),
                    consumer_rank_ids=tuple(binding.rank_id for binding in consumers),
                    edge_path=path,
                    serialization_format=serialization,
                    chunk_bytes=chunk_bytes,
                    # The stable simulator protocol models chunk repetitions.
                    # Advertising every chunk as admissible preserves total KV
                    # byte volume; plan-level backpressure remains the hard cap.
                    maximum_inflight_chunks=chunk_count,
                    overlap_with_decode=True,
                    cache_owner="decode",
                    eviction_policy="deadline",
                    retry_limit=1,
                    fallback="recompute",
                    transport_adapter=f"sloforge.{transport}",
                    expected_latency_us=max(path_us, 0.001),
                    expected_cost_usd=0.0,
                )
            )
            route_latencies.append(path_us)
    return (
        KVTransferPlan(routes=tuple(routes), backpressure_limit_bytes=kv_bytes * 2),
        max(route_latencies),
        None,
    )


def _candidate_id(
    tp: int, pp: int, dp: int, ep: int, disaggregated: bool, gpu_ids: tuple[str, ...]
) -> str:
    payload = f"{tp}:{pp}:{dp}:{ep}:{int(disaggregated)}:{','.join(gpu_ids)}"
    return f"physical-{hashlib.sha256(payload.encode()).hexdigest()[:12]}"


def _service_availability(placement: RankPlacement, base_availability: float) -> float:
    """Evaluate replica/pool availability without assuming shared domains are independent."""

    replica_bindings: dict[str, list[RankBinding]] = defaultdict(list)
    for binding in placement.bindings:
        replica_bindings[binding.replica_id].append(binding)
    replica_roles: dict[str, WorkerRole] = {}
    replica_domains: dict[str, frozenset[str]] = {}
    for replica_id, bindings in replica_bindings.items():
        replica_domains[replica_id] = frozenset(binding.fault_domain for binding in bindings)
        roles = {binding.worker_role for binding in bindings}
        if len(roles) != 1:
            raise ValueError("a compiler-generated replica cannot mix worker roles")
        replica_roles[replica_id] = next(iter(roles))

    pools: tuple[tuple[frozenset[str], ...], ...]
    if all(role is WorkerRole.AGGREGATED for role in replica_roles.values()):
        pools = (
            tuple(
                sorted(
                    (replica_domains[replica_id] for replica_id in replica_bindings),
                    key=lambda domains: tuple(sorted(domains)),
                )
            ),
        )
    else:
        # Disaggregated service requires at least one complete replica in each
        # role pool. Shared host domains remain correlated across both pools.
        pools = tuple(
            tuple(
                sorted(
                    (
                        replica_domains[replica_id]
                        for replica_id, replica_role in replica_roles.items()
                        if replica_role is role
                    ),
                    key=lambda domains: tuple(sorted(domains)),
                )
            )
            for role in (WorkerRole.PREFILL, WorkerRole.DECODE)
        )

    canonical_pools = tuple(tuple(tuple(sorted(domains)) for domains in pool) for pool in pools)

    @cache
    def probability(state: tuple[tuple[tuple[str, ...], ...], ...]) -> float:
        if any(not pool for pool in state):
            return 0.0
        if all(any(not domains for domains in pool) for pool in state):
            return 1.0
        domain = min(name for pool in state for domains in pool for name in domains)
        healthy = tuple(
            tuple(tuple(item for item in domains if item != domain) for domains in pool)
            for pool in state
        )
        failed = tuple(
            tuple(domains for domains in pool if domain not in domains) for pool in state
        )
        return base_availability * probability(healthy) + (1.0 - base_availability) * probability(
            failed
        )

    return probability(canonical_pools)


def _evaluate_candidate(
    request: CompilerRequest,
    *,
    tp: int,
    pp: int,
    dp: int,
    ep: int,
    disaggregated: bool,
    placement_strategy: OptimizationStrategy,
    excluded: frozenset[str] = frozenset(),
    preselected_gpus: tuple[GpuNode, ...] | None = None,
) -> _Candidate | CandidateSummary:
    rank_count = tp * pp * dp
    gpus = (
        preselected_gpus
        if preselected_gpus is not None
        else _ordered_candidate_gpus(
            request.topology,
            tp=tp,
            pp=pp,
            dp=dp,
            strategy=placement_strategy,
            seed=request.seed,
            excluded=excluded,
        )
    )
    gpu_ids = tuple(gpu.node_id for gpu in gpus)
    candidate_id = _candidate_id(tp, pp, dp, ep, disaggregated, gpu_ids)
    rejection: list[str] = []
    if len(gpus) != rank_count:
        rejection.append("insufficient_healthy_gpus")
    if disaggregated and dp < 2:
        rejection.append("disaggregation_requires_two_data_replicas")
    moe_expert_counts = tuple(len(layer.experts) for layer in request.model.layers if layer.experts)
    replica_size = tp * pp
    if ep > 1 and (
        not moe_expert_counts
        or any(count % ep != 0 for count in moe_expert_counts)
        or replica_size % ep != 0
    ):
        rejection.append("expert_parallelism_not_divisible")
    if tp > 1 and (
        request.model.attention_heads % tp != 0 or request.model.key_value_heads % tp != 0
    ):
        rejection.append("tensor_parallelism_not_divisible")
    available_stage_boundaries = 1 + sum(
        layer.allowed_stage_boundaries_after for layer in request.model.layers[:-1]
    )
    if pp > available_stage_boundaries:
        rejection.append("pipeline_stage_boundaries_unavailable")
    if ep > 1 and "expert_parallel" not in request.model.runtime_features:
        rejection.append("expert_parallelism_not_supported")
    if disaggregated and "prefill_decode_disaggregation" not in request.model.runtime_features:
        rejection.append("disaggregation_not_supported")
    if rank_count > request.constraints.maximum_ranks:
        rejection.append("maximum_ranks_exceeded")
    if rejection:
        return CandidateSummary(
            candidate_id=candidate_id,
            tensor_parallel=tp,
            pipeline_parallel=pp,
            data_parallel=dp,
            expert_parallel=ep,
            disaggregated=disaggregated,
            gpu_ids=gpu_ids,
            communication_us=0.0,
            p95_ttft_ms=0.0,
            p99_tpot_ms=0.0,
            goodput_tokens_per_second=0.0,
            cost_per_million_tokens=0.0,
            availability=0.0,
            failure_exposure_score=1.0,
            objective_score=1.0e30,
            feasible=False,
            rejection_codes=tuple(rejection),
        )
    placement = _placement(
        request.topology,
        gpus,
        tp=tp,
        pp=pp,
        dp=dp,
        disaggregated=disaggregated,
    )
    try:
        memory = _memory_plan(request.model, gpus, request.constraints, tp=tp, pp=pp, dp=dp)
    except ValueError:
        rejection.append("gpu_memory_capacity_exceeded")
        return CandidateSummary(
            candidate_id=candidate_id,
            tensor_parallel=tp,
            pipeline_parallel=pp,
            data_parallel=dp,
            expert_parallel=ep,
            disaggregated=disaggregated,
            gpu_ids=gpu_ids,
            communication_us=0.0,
            p95_ttft_ms=0.0,
            p99_tpot_ms=0.0,
            goodput_tokens_per_second=0.0,
            cost_per_million_tokens=0.0,
            availability=0.0,
            failure_exposure_score=1.0,
            objective_score=1.0e30,
            feasible=False,
            rejection_codes=tuple(rejection),
        )
    parallelism = _parallelism(tp, pp, dp, ep, disaggregated)
    collectives, overlap, collective_us, collective_rejection = _collectives(
        request, parallelism, placement
    )
    kv_transfer, kv_us, kv_rejection = _kv_transfer(request, placement, disaggregated)
    rejection.extend(
        reason for reason in (collective_rejection, kv_rejection) if reason is not None
    )
    if rejection:
        return CandidateSummary(
            candidate_id=candidate_id,
            tensor_parallel=tp,
            pipeline_parallel=pp,
            data_parallel=dp,
            expert_parallel=ep,
            disaggregated=disaggregated,
            gpu_ids=gpu_ids,
            communication_us=0.0,
            p95_ttft_ms=0.0,
            p99_tpot_ms=0.0,
            goodput_tokens_per_second=0.0,
            cost_per_million_tokens=0.0,
            availability=0.0,
            failure_exposure_score=1.0,
            objective_score=1.0e30,
            feasible=False,
            rejection_codes=tuple(rejection),
        )
    prefill_replicas = max(1, dp // 2) if disaggregated else dp
    decode_replicas = dp - prefill_replicas if disaggregated else dp
    # TP shortens the per-request compute path. PP partitions memory and enables
    # pipeline occupancy, but its sequential stages do not divide one request's
    # TTFT/TPOT again. DP adds independent throughput replicas, not latency speedup.
    per_request_compute_parallelism = tp
    prefill_ms = (
        request.constraints.prompt_tokens_p95
        / (request.assumptions.prefill_tokens_per_second_per_gpu * per_request_compute_parallelism)
        * 1_000.0
    )
    decode_step_ms = 1_000.0 / (
        request.assumptions.decode_tokens_per_second_per_gpu * per_request_compute_parallelism
    )
    ttft_ms = prefill_ms + collective_us / 1_000.0 + kv_us / 1_000.0
    tpot_ms = (
        decode_step_ms + collective_us / max(1, request.constraints.output_tokens_p95) / 1_000.0
    )
    goodput = (
        request.assumptions.decode_tokens_per_second_per_gpu
        * per_request_compute_parallelism
        * decode_replicas
        * (0.92 if collective_us > 0 else 1.0)
    )
    cost_per_million = (
        rank_count * request.assumptions.gpu_hourly_price_usd / max(goodput, 1e-9) / 3_600.0 * 1e6
    )
    host_counts: dict[str, int] = defaultdict(int)
    for binding in placement.bindings:
        host_counts[binding.host_id] += 1
    largest_domain_fraction = max(host_counts.values(), default=rank_count) / rank_count
    availability = _service_availability(placement, request.assumptions.base_availability)
    failure_exposure = largest_domain_fraction * (1.0 - availability)
    relative_uncertainty = request.assumptions.measurement_relative_uncertainty
    robust_ttft_ms = ttft_ms * (1.0 + relative_uncertainty)
    robust_tpot_ms = tpot_ms * (1.0 + relative_uncertainty)
    robust_goodput = goodput * (1.0 - relative_uncertainty)
    robust_cost = cost_per_million * (1.0 + relative_uncertainty)
    if robust_ttft_ms > request.constraints.p95_ttft_ms:
        rejection.append("p95_ttft_slo")
    if robust_tpot_ms > request.constraints.p99_tpot_ms:
        rejection.append("p99_tpot_slo")
    if robust_goodput < request.constraints.minimum_goodput_tokens_per_second:
        rejection.append("minimum_goodput")
    # Availability is composed from the explicit per-fault-domain assumption,
    # not from latency measurement residuals; applying that relative error to a
    # probability near one would manufacture an unrelated five-point outage.
    if availability < request.constraints.minimum_availability:
        rejection.append("minimum_availability")
    if (
        request.constraints.maximum_cost_per_million_tokens is not None
        and robust_cost > request.constraints.maximum_cost_per_million_tokens
    ):
        rejection.append("maximum_cost")
    feasible = not rejection
    violation = max(0.0, robust_ttft_ms / request.constraints.p95_ttft_ms - 1.0) + max(
        0.0, robust_tpot_ms / request.constraints.p99_tpot_ms - 1.0
    )
    objective_terms = {
        CompilerObjective.MINIMIZE_COST: cost_per_million + violation * 1e6,
        CompilerObjective.MINIMIZE_LATENCY: ttft_ms + tpot_ms * 10.0 + violation * 1e6,
        CompilerObjective.MAXIMIZE_GOODPUT: -goodput + violation * 1e6,
        CompilerObjective.ROBUST_BALANCED: (
            cost_per_million
            + ttft_ms
            + tpot_ms * 10.0
            + collective_us / 1_000.0
            + failure_exposure * 1_000.0
            + violation * 1e6
        ),
    }
    summary = CandidateSummary(
        candidate_id=candidate_id,
        tensor_parallel=tp,
        pipeline_parallel=pp,
        data_parallel=dp,
        expert_parallel=ep,
        disaggregated=disaggregated,
        gpu_ids=gpu_ids,
        communication_us=collective_us + kv_us,
        p95_ttft_ms=ttft_ms,
        p99_tpot_ms=tpot_ms,
        goodput_tokens_per_second=goodput,
        cost_per_million_tokens=cost_per_million,
        availability=availability,
        failure_exposure_score=failure_exposure,
        objective_score=objective_terms[request.objective],
        feasible=feasible,
        rejection_codes=tuple(rejection),
    )
    return _Candidate(
        candidate_id=candidate_id,
        tp=tp,
        pp=pp,
        dp=dp,
        ep=ep,
        disaggregated=disaggregated,
        gpu_ids=gpu_ids,
        placement=placement,
        memory=memory,
        collectives=collectives,
        kv_transfer=kv_transfer,
        overlap=overlap,
        summary=summary,
    )


def _degrees(
    maximum: int, strategy: OptimizationStrategy, fixed: int | None = None
) -> tuple[int, ...]:
    if fixed is not None:
        return (fixed,)
    if strategy is OptimizationStrategy.EXHAUSTIVE:
        return tuple(range(1, maximum + 1))
    values = [1]
    power = 2
    while power <= maximum:
        values.append(power)
        power *= 2
    return tuple(values)


def _pareto(candidates: tuple[CandidateSummary, ...]) -> tuple[CandidateSummary, ...]:
    feasible = tuple(candidate for candidate in candidates if candidate.feasible)
    result: list[CandidateSummary] = []
    for candidate in feasible:
        dominated = any(
            other.candidate_id != candidate.candidate_id
            and other.p95_ttft_ms <= candidate.p95_ttft_ms
            and other.p99_tpot_ms <= candidate.p99_tpot_ms
            and other.cost_per_million_tokens <= candidate.cost_per_million_tokens
            and other.failure_exposure_score <= candidate.failure_exposure_score
            and other.communication_us <= candidate.communication_us
            and other.goodput_tokens_per_second >= candidate.goodput_tokens_per_second
            and other.availability >= candidate.availability
            and (
                other.p95_ttft_ms < candidate.p95_ttft_ms
                or other.p99_tpot_ms < candidate.p99_tpot_ms
                or other.cost_per_million_tokens < candidate.cost_per_million_tokens
                or other.failure_exposure_score < candidate.failure_exposure_score
                or other.communication_us < candidate.communication_us
                or other.goodput_tokens_per_second > candidate.goodput_tokens_per_second
                or other.availability > candidate.availability
            )
            for other in feasible
        )
        if not dominated:
            result.append(candidate)
    return tuple(sorted(result, key=lambda item: (item.objective_score, item.candidate_id)))


def _metric(value: float, uncertainty: float, unit: str) -> MetricInterval:
    spread = abs(value) * uncertainty
    return MetricInterval(
        estimate=max(0.0, value),
        lower=max(0.0, value - spread),
        upper=max(0.0, value + spread),
        confidence=max(0.0, min(1.0, 1.0 - uncertainty)),
        unit=unit,
    )


def _physical_metrics(
    summary: CandidateSummary,
    uncertainty: float,
    output_tokens_p95: int,
) -> PhysicalMetrics:
    e2e = summary.p95_ttft_ms + summary.p99_tpot_ms * output_tokens_p95
    communication_fraction = summary.communication_us / max(1.0, e2e * 1_000.0)
    return PhysicalMetrics(
        p95_ttft_ms=_metric(summary.p95_ttft_ms, uncertainty, "ms"),
        p99_tpot_ms=_metric(summary.p99_tpot_ms, uncertainty, "ms"),
        p95_end_to_end_ms=_metric(e2e, uncertainty, "ms"),
        throughput_tokens_per_second=_metric(
            summary.goodput_tokens_per_second / 0.92, uncertainty, "tokens/s"
        ),
        goodput_tokens_per_second=_metric(
            summary.goodput_tokens_per_second, uncertainty, "tokens/s"
        ),
        cost_usd_per_million_tokens=_metric(
            summary.cost_per_million_tokens, uncertainty, "USD/1M_tokens"
        ),
        availability=_metric(summary.availability, min(uncertainty, 0.05), "probability"),
        communication_overhead_fraction=_metric(communication_fraction, uncertainty, "fraction"),
    )


def _expert_placement(model: ModelGraph, candidate: _Candidate) -> ExpertPlacement | None:
    experts = tuple(expert for layer in model.layers for expert in layer.experts)
    if not experts:
        return None
    ranks_by_replica: dict[str, list[int]] = {}
    for binding in candidate.placement.bindings:
        ranks_by_replica.setdefault(binding.replica_id, []).append(binding.rank_id)
    replica_ranks = tuple(tuple(sorted(ranks)) for _, ranks in sorted(ranks_by_replica.items()))
    assignments = tuple(
        ExpertAssignment(
            expert_id=expert.expert_id,
            # Every independently routable data replica must contain the whole
            # expert set. The tuple represents one owner per replica; choosing
            # the local owner cyclically spreads experts across that replica's
            # EP-capable ranks without coupling separate replica fault domains.
            rank_ids=tuple(ranks[index % len(ranks)] for ranks in replica_ranks),
            expected_load=expert.expected_load,
            capacity_factor=1.25,
        )
        for index, expert in enumerate(experts)
    )
    return ExpertPlacement(
        assignments=assignments,
        hot_expert_strategy="rebalance",
        maximum_replicas_per_expert=len(replica_ranks),
        rebalance_minimum_interval_seconds=30.0,
    )


def _recovery_variants(
    request: CompilerRequest,
    selected: _Candidate,
    candidates: tuple[_Candidate, ...],
) -> tuple[RecoveryVariant, ...]:
    alternatives = tuple(
        item
        for item in sorted(candidates, key=lambda item: item.summary.objective_score)
        if item.candidate_id != selected.candidate_id
        and item.summary.feasible
        and (
            item.gpu_ids != selected.gpu_ids
            or (item.tp, item.pp, item.dp, item.ep, item.disaggregated)
            != (selected.tp, selected.pp, selected.dp, selected.ep, selected.disaggregated)
        )
    )
    result: list[RecoveryVariant] = []
    for index, alternate in enumerate(alternatives[:3]):
        result.append(
            RecoveryVariant(
                variant_id=f"recovery-{index}-{alternate.candidate_id}",
                triggers=(
                    RecoveryTrigger(
                        diagnosis_code="fabric_resource_degradation",
                        minimum_confidence=0.75,
                        minimum_duration_seconds=5.0,
                    ),
                ),
                alternate_parallelism=_parallelism(
                    alternate.tp,
                    alternate.pp,
                    alternate.dp,
                    alternate.ep,
                    alternate.disaggregated,
                ),
                alternate_rank_placement=alternate.placement,
                alternate_collectives=alternate.collectives,
                alternate_kv_transfer=alternate.kv_transfer,
                expected_degraded_metrics=_physical_metrics(
                    alternate.summary,
                    request.assumptions.measurement_relative_uncertainty,
                    request.constraints.output_tokens_p95,
                ),
                transition_cost_usd=(
                    alternate.summary.cost_per_million_tokens / 1_000_000.0 * 1_000.0
                ),
                transition_seconds=request.assumptions.cold_start_ms / 1_000.0,
                rebuild_required=True,
                compatibility_constraints=("same-model-digest", "healthy-topology-path"),
            )
        )
    return tuple(result)


def _failure_exposure(
    request: CompilerRequest, candidate: _Candidate
) -> tuple[FailureExposure, ...]:
    return tuple(
        FailureExposure(
            fault_domain=host_id,
            affected_rank_ids=tuple(
                binding.rank_id
                for binding in candidate.placement.bindings
                if binding.host_id == host_id
            ),
            probability=max(0.0, 1.0 - request.assumptions.base_availability),
            expected_slo_impact_ms=candidate.summary.p95_ttft_ms,
        )
        for host_id in sorted({binding.host_id for binding in candidate.placement.bindings})
    )


def _candidate_plan(
    request: CompilerRequest,
    candidate: _Candidate,
    *,
    optimizer_history: tuple[OptimizerTraceEntry, ...] = (),
    rejected_alternatives: tuple[RejectedPhysicalCandidate, ...] = (),
    recovery_variants: tuple[RecoveryVariant, ...] = (),
) -> PhysicalExecutionPlan:
    """Build the canonical plan shape used by both refinement and final emission."""

    topology_hash = ArtifactDigest(value=canonical_hash(request.topology))
    profile_hash = ArtifactDigest(value=canonical_hash(request.fabric_profile))
    model_hash = ArtifactDigest(value=canonical_hash(request.model))
    plan_seed = (
        f"{request.logical_deployment_plan.digest.value}:{model_hash.value}:"
        f"{topology_hash.value}:{profile_hash.value}:{candidate.candidate_id}:{request.seed}"
    )
    physical_metrics = _physical_metrics(
        candidate.summary,
        request.assumptions.measurement_relative_uncertainty,
        request.constraints.output_tokens_p95,
    )
    return PhysicalExecutionPlan(
        plan_id=f"physical-plan-{hashlib.sha256(plan_seed.encode()).hexdigest()[:16]}",
        logical_deployment_plan=request.logical_deployment_plan,
        model_graph_hash=model_hash,
        topology_fingerprint=topology_hash,
        fabric_profile_hash=profile_hash,
        parallelism=_parallelism(
            candidate.tp,
            candidate.pp,
            candidate.dp,
            candidate.ep,
            candidate.disaggregated,
        ),
        rank_placement=candidate.placement,
        expert_placement=_expert_placement(request.model, candidate),
        collectives=candidate.collectives,
        kv_transfer=candidate.kv_transfer,
        memory=candidate.memory,
        communication_overlap=candidate.overlap,
        predicted_metrics=physical_metrics,
        bottleneck_prediction=(
            "communication"
            if candidate.summary.communication_us / 1_000.0 > candidate.summary.p95_ttft_ms * 0.30
            else (
                "prefill_compute"
                if candidate.summary.p95_ttft_ms > request.constraints.p95_ttft_ms * 0.75
                else "decode_compute"
            )
        ),
        failure_exposure=_failure_exposure(request, candidate),
        optimizer_history=optimizer_history,
        rejected_alternatives=rejected_alternatives,
        recovery_variants=recovery_variants,
        evidence=(
            request.fabric_profile.hardware_manifest,
            request.fabric_profile.software_manifest,
        ),
        compiler_version=_COMPILER_VERSION,
        git_commit=request.git_commit,
        reproducibility=ReproducibilityMetadata(
            seed=request.seed,
            generated_at=request.generated_at,
            environment_digest=request.environment_digest,
            command=("sloforge", "fabric", "compile"),
        ),
    )


def _compiler_repository_root() -> Path:
    root = Path(__file__).resolve().parents[4]
    if not (root / "Cargo.toml").is_file():
        raise RuntimeError("unable to locate the SLOForge Rust workspace")
    return root


def _percentile(values: tuple[float, ...], quantile: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile without samples")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _objective_score(request: CompilerRequest, summary: CandidateSummary) -> float:
    uncertainty = request.assumptions.measurement_relative_uncertainty
    robust_ttft_ms = summary.p95_ttft_ms * (1.0 + uncertainty)
    robust_tpot_ms = summary.p99_tpot_ms * (1.0 + uncertainty)
    violation = max(0.0, robust_ttft_ms / request.constraints.p95_ttft_ms - 1.0) + max(
        0.0, robust_tpot_ms / request.constraints.p99_tpot_ms - 1.0
    )
    terms = {
        CompilerObjective.MINIMIZE_COST: summary.cost_per_million_tokens + violation * 1e6,
        CompilerObjective.MINIMIZE_LATENCY: summary.p95_ttft_ms
        + summary.p99_tpot_ms * 10.0
        + violation * 1e6,
        CompilerObjective.MAXIMIZE_GOODPUT: -summary.goodput_tokens_per_second + violation * 1e6,
        CompilerObjective.ROBUST_BALANCED: (
            summary.cost_per_million_tokens
            + summary.p95_ttft_ms
            + summary.p99_tpot_ms * 10.0
            + summary.communication_us / 1_000.0
            + summary.failure_exposure_score * 1_000.0
            + violation * 1e6
        ),
    }
    return terms[request.objective]


def _simulated_rejection_codes(
    request: CompilerRequest, summary: CandidateSummary
) -> tuple[str, ...]:
    uncertainty = request.assumptions.measurement_relative_uncertainty
    codes: list[str] = []
    if summary.p95_ttft_ms * (1.0 + uncertainty) > request.constraints.p95_ttft_ms:
        codes.append("simulated_p95_ttft_slo")
    if summary.p99_tpot_ms * (1.0 + uncertainty) > request.constraints.p99_tpot_ms:
        codes.append("simulated_p99_tpot_slo")
    if (
        summary.goodput_tokens_per_second * (1.0 - uncertainty)
        < request.constraints.minimum_goodput_tokens_per_second
    ):
        codes.append("simulated_minimum_goodput")
    if (
        request.constraints.maximum_cost_per_million_tokens is not None
        and summary.cost_per_million_tokens * (1.0 + uncertainty)
        > request.constraints.maximum_cost_per_million_tokens
    ):
        codes.append("simulated_maximum_cost")
    return tuple(codes)


def _simulate_candidate(request: CompilerRequest, candidate: _Candidate) -> tuple[_Candidate, int]:
    """Refine one analytical candidate through the deterministic Rust twin."""

    from sloforge.fabric.simulation import (
        SimulationWorkload,
        build_simulation_request,
        request_latencies,
        run_simulation,
    )

    # Candidate refinement isolates one representative request. The physical
    # simulator models exclusive GPU operation resources rather than an
    # engine's continuous-batching scheduler; treating configured concurrency
    # as simultaneous independent GPU kernels would manufacture queue delay.
    # Full workload queueing remains the subsequent `fabric validate` pass.
    request_count = 1
    workload = SimulationWorkload(
        request_count=request_count,
        arrival_interval_us=0.0,
        prompt_tokens=request.constraints.prompt_tokens_p95,
        output_tokens=request.constraints.output_tokens_p95,
    )
    simulation_request = build_simulation_request(
        _candidate_plan(request, candidate),
        request.topology,
        request.fabric_profile,
        workload,
        seed=request.seed,
    )
    output = run_simulation(
        simulation_request,
        repository_root=_compiler_repository_root(),
        timeout_seconds=30.0,
    )
    latencies = request_latencies(output)
    if len(latencies) != request_count:
        raise RuntimeError(
            f"fabric simulator returned {len(latencies)} request latencies; "
            f"expected {request_count}"
        )
    ttft_ms = _percentile(tuple(item.ttft_us / 1_000.0 for item in latencies), 0.95)
    tpot_samples = tuple(
        max(0.0, item.end_to_end_us - item.ttft_us)
        / request.constraints.output_tokens_p95
        / 1_000.0
        for item in latencies
    )
    tpot_ms = _percentile(tpot_samples, 0.99)
    decode_replicas = (
        candidate.dp - max(1, candidate.dp // 2) if candidate.disaggregated else candidate.dp
    )
    simulated_decode_capacity = 1_000.0 / max(tpot_ms, 1e-9) * decode_replicas
    communication_operations = tuple(
        operation
        for operation in output.operations
        if ":collective-" in operation.operation_id or ":kv-" in operation.operation_id
    )
    communication_us = sum(item.duration_us for item in communication_operations) / request_count
    provisional = candidate.summary.model_copy(
        update={
            "communication_us": communication_us,
            "p95_ttft_ms": ttft_ms,
            "p99_tpot_ms": tpot_ms,
            "goodput_tokens_per_second": min(
                candidate.summary.goodput_tokens_per_second,
                simulated_decode_capacity,
            ),
        }
    )
    cost_per_million = (
        candidate.tp
        * candidate.pp
        * candidate.dp
        * request.assumptions.gpu_hourly_price_usd
        / max(provisional.goodput_tokens_per_second, 1e-9)
        / 3_600.0
        * 1e6
    )
    provisional = provisional.model_copy(update={"cost_per_million_tokens": cost_per_million})
    rejection_codes = _simulated_rejection_codes(request, provisional)
    summary = provisional.model_copy(
        update={
            "objective_score": _objective_score(request, provisional),
            "feasible": not rejection_codes,
            "rejection_codes": rejection_codes,
        }
    )
    # The work unit is a deterministic count of processed simulator events. It
    # is suitable for canonical optimizer traces; actual wall time stays in the
    # outer PhysicalCompileResult diagnostic.
    return (
        candidate.model_copy(update={"summary": summary}),
        output.metrics.processed_events,
    )


def _refine_candidates(
    request: CompilerRequest, candidates: tuple[_Candidate, ...]
) -> tuple[tuple[_Candidate, ...], frozenset[str], dict[str, int], int]:
    """Iteratively simulate every candidate that can reach the final frontier."""

    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    validated: set[str] = set()
    work_by_id: dict[str, int] = {}
    simulator_calls = 0
    while True:
        feasible = tuple(item for item in by_id.values() if item.summary.feasible)
        if not feasible:
            break
        frontier_ids = {
            item.candidate_id for item in _pareto(tuple(item.summary for item in feasible))
        }
        ordered = sorted(
            feasible, key=lambda item: (item.summary.objective_score, item.candidate_id)
        )
        # The leading alternatives are the source for recovery variants, so
        # validate them with the same twin rather than publishing static-only
        # recovery promises.
        promoted = frontier_ids | {item.candidate_id for item in ordered[:4]}
        pending = tuple(sorted(promoted - validated))
        if not pending:
            break
        for candidate_id in pending:
            try:
                refined, work_units = _simulate_candidate(request, by_id[candidate_id])
            except ValueError:
                summary = by_id[candidate_id].summary.model_copy(
                    update={
                        "feasible": False,
                        "objective_score": 1.0e30,
                        "rejection_codes": ("simulator_invalid_physical_plan",),
                    }
                )
                refined = by_id[candidate_id].model_copy(update={"summary": summary})
                work_units = 0
            by_id[candidate_id] = refined
            validated.add(candidate_id)
            work_by_id[candidate_id] = work_units
            simulator_calls += 1
    return (
        tuple(sorted(by_id.values(), key=lambda item: item.candidate_id)),
        frozenset(validated),
        work_by_id,
        simulator_calls,
    )


def compile_physical_plan(request: CompilerRequest) -> PhysicalCompileResult:
    """Compile and explain a deterministic physical execution plan."""

    started_ns = time.perf_counter_ns()
    gpus = _healthy_gpus(request.topology)
    if not gpus:
        raise ValueError("topology contains no non-failed GPUs")
    maximum = min(len(gpus), request.constraints.maximum_ranks)
    tp_degrees = _degrees(maximum, request.strategy, request.constraints.tensor_parallel_degree)
    pp_degrees = _degrees(maximum, request.strategy, request.constraints.pipeline_parallel_degree)
    dp_degrees = _degrees(maximum, request.strategy, request.constraints.data_parallel_degree)
    ep_degrees = _degrees(maximum, request.strategy, request.constraints.expert_parallel_degree)
    disaggregation_options = (
        (True,) if request.constraints.require_disaggregation else (False, True)
    )
    candidates: list[_Candidate] = []
    summaries: list[CandidateSummary] = []
    strategy = request.strategy
    placement_strategy = (
        strategy
        if strategy
        in {
            OptimizationStrategy.RANDOM_PLACEMENT,
            OptimizationStrategy.TOPOLOGY_UNAWARE,
            OptimizationStrategy.ROBUST_FAILURE,
        }
        else OptimizationStrategy.GREEDY_TOPOLOGY_AWARE
    )
    # The hierarchical strategy prunes impossible degree products before any
    # placement or curve evaluation. Tiny exhaustive mode intentionally visits
    # the same finite space to serve as a correctness oracle.
    combinations = itertools.product(
        tp_degrees, pp_degrees, dp_degrees, ep_degrees, disaggregation_options
    )
    placement_cache: dict[tuple[int, int, int], tuple[GpuNode, ...]] = {}
    for tp, pp, dp, ep, disaggregated in combinations:
        rank_count = tp * pp * dp
        if rank_count > maximum:
            continue
        placement_key = (tp, pp, dp)
        if placement_key not in placement_cache:
            placement_cache[placement_key] = _ordered_candidate_gpus(
                request.topology,
                tp=tp,
                pp=pp,
                dp=dp,
                strategy=placement_strategy,
                seed=request.seed,
            )
        outcome = _evaluate_candidate(
            request,
            tp=tp,
            pp=pp,
            dp=dp,
            ep=ep,
            disaggregated=disaggregated,
            placement_strategy=placement_strategy,
            preselected_gpus=placement_cache[placement_key],
        )
        if isinstance(outcome, CandidateSummary):
            summaries.append(outcome)
        else:
            candidates.append(outcome)
            summaries.append(outcome.summary)
    candidate_ids = {candidate.candidate_id for candidate in candidates}
    static_rejections = tuple(
        summary for summary in summaries if summary.candidate_id not in candidate_ids
    )
    refined_candidates, validated_ids, work_by_id, simulator_calls = _refine_candidates(
        request, tuple(candidates)
    )
    summaries = [
        *static_rejections,
        *(candidate.summary for candidate in refined_candidates),
    ]
    candidates = list(refined_candidates)
    feasible = tuple(candidate for candidate in candidates if candidate.summary.feasible)
    if not feasible:
        reasons = sorted({code for item in summaries for code in item.rejection_codes})
        raise ValueError(f"no feasible physical plan; rejection codes: {','.join(reasons)}")
    selected = min(feasible, key=lambda item: (item.summary.objective_score, item.candidate_id))
    frontier = _pareto(tuple(summaries))
    frontier_ids = {item.candidate_id for item in frontier}
    history: list[OptimizerTraceEntry] = []
    for sequence, item in enumerate(sorted(summaries, key=lambda value: value.candidate_id)):
        history.append(
            OptimizerTraceEntry(
                sequence=sequence,
                candidate_id=item.candidate_id,
                phase=(
                    "simulation"
                    if item.candidate_id in validated_ids
                    else ("lower_bound" if item.feasible else "feasibility")
                ),
                decision=(
                    "select"
                    if item.candidate_id == selected.candidate_id
                    else ("promote" if item.candidate_id in frontier_ids else "reject")
                ),
                reason_code=(
                    "simulator_validated_minimum_robust_objective"
                    if item.candidate_id == selected.candidate_id
                    else (
                        "simulator_validated_pareto_candidate"
                        if item.candidate_id in frontier_ids
                        else (
                            "dominated_objective"
                            if item.feasible
                            else "+".join(item.rejection_codes)
                        )
                    )
                ),
                simulator_calls=1 if item.candidate_id in validated_ids else 0,
                # Canonical plans cannot include noisy wall time. This is the
                # deterministic equivalent work estimate in reference-event ms;
                # PhysicalCompileResult.solver_time_ms is actual elapsed time.
                solver_time_ms=work_by_id.get(item.candidate_id, 0) / 1_000.0,
            )
        )
    rejected = tuple(
        RejectedPhysicalCandidate(
            candidate_id=item.candidate_id,
            stage=(
                "simulation_validation"
                if any(code.startswith("simulator_") for code in item.rejection_codes)
                else ("constraint_check" if not item.feasible else "objective_ranking")
            ),
            reason_code=(
                "+".join(item.rejection_codes) if item.rejection_codes else "dominated_objective"
            ),
            explanation=(
                "candidate violated hard constraints"
                if item.rejection_codes
                else "candidate was feasible but not selected by the requested objective"
            ),
            violated_constraints=item.rejection_codes,
        )
        for item in summaries
        if item.candidate_id != selected.candidate_id
    )
    validated_candidates = tuple(
        candidate for candidate in candidates if candidate.candidate_id in validated_ids
    )
    plan = _candidate_plan(
        request,
        selected,
        optimizer_history=tuple(history),
        rejected_alternatives=rejected,
        recovery_variants=_recovery_variants(request, selected, validated_candidates),
    )
    elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
    return PhysicalCompileResult(
        selected=plan,
        pareto_frontier=frontier,
        all_candidates=tuple(sorted(summaries, key=lambda item: item.candidate_id)),
        strategy=request.strategy,
        solver_time_ms=elapsed_ms,
        simulator_calls=simulator_calls,
        simulator_validated_candidate_ids=tuple(sorted(validated_ids)),
        deterministic_solver_work_units=sum(work_by_id.values()),
    )
