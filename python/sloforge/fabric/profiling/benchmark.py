"""Reproducible fabric benchmark harnesses for synthetic and current-host paths."""

from __future__ import annotations

import hashlib
import math
import platform
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from sloforge.fabric.ir import TopologyGraph as CanonicalTopologyGraph
from sloforge.fabric.profiling.models import (
    BenchmarkCase,
    BenchmarkResult,
    BenchmarkStatus,
    Direction,
    EnvironmentFact,
    FabricProfile,
    Invocation,
    MeasurementMode,
    Placement,
    Primitive,
    RawSample,
    RawSampleArtifact,
    RobustSummary,
    finalize_profile,
    finalize_result,
)
from sloforge.fabric.profiling.topology_input import normalize_benchmark_topology
from sloforge.fabric.topology.models import (
    DiscoveryTopologyGraph,
    EdgeKind,
    FactState,
    NodeKind,
    ObservedFact,
    TopologyEdge,
)
from sloforge.util import percentile, utc_now, write_json

_FULL_PRIMITIVES = tuple(Primitive)
_QUICK_PRIMITIVES = (
    Primitive.HOST_MEMCPY,
    Primitive.GPU_P2P,
    Primitive.ALL_REDUCE,
    Primitive.ALL_TO_ALL,
    Primitive.EXPERT_DISPATCH,
    Primitive.KV_TRANSFER,
    Primitive.STARTUP,
)
_COLLECTIVES = {
    Primitive.ALL_REDUCE,
    Primitive.ALL_GATHER,
    Primitive.REDUCE_SCATTER,
    Primitive.BROADCAST,
    Primitive.SEND_RECV,
    Primitive.ALL_TO_ALL,
    Primitive.EXPERT_DISPATCH,
    Primitive.EXPERT_COMBINE,
}
_GPU_LOCAL = {
    Primitive.KERNEL_LAUNCH,
    Primitive.DEVICE_SYNCHRONIZE,
    Primitive.DEVICE_MEMORY,
    Primitive.GEMM,
    Primitive.PREFILL,
    Primitive.DECODE,
}
_NO_PAYLOAD = {
    Primitive.KERNEL_LAUNCH,
    Primitive.DEVICE_SYNCHRONIZE,
    Primitive.STARTUP,
    Primitive.GROUP_INITIALIZATION,
}


def _environment() -> tuple[EnvironmentFact, ...]:
    return (
        EnvironmentFact(name="platform", value=platform.platform(), source="python-platform"),
        EnvironmentFact(name="machine", value=platform.machine(), source="python-platform"),
        EnvironmentFact(name="python", value=sys.version.split()[0], source="python-runtime"),
        EnvironmentFact(name="numpy", value=np.__version__, source="python-package"),
    )


def summarize_samples(
    samples: tuple[RawSample, ...],
    *,
    seed: int,
    confidence_level: float = 0.95,
    bootstrap_rounds: int = 500,
) -> RobustSummary:
    if not samples:
        raise ValueError("cannot summarize zero samples")
    if bootstrap_rounds < 100:
        raise ValueError("bootstrap_rounds must be at least 100")
    durations = [sample.duration_microseconds for sample in samples]
    median = statistics.median(durations)
    deviations = [abs(value - median) for value in durations]
    rng = random.Random(seed)
    bootstrapped = sorted(
        statistics.median(rng.choices(durations, k=len(durations))) for _ in range(bootstrap_rounds)
    )
    alpha = (1.0 - confidence_level) / 2.0
    return RobustSummary(
        sample_count=len(samples),
        median_microseconds=median,
        p95_microseconds=percentile(durations, 0.95),
        p99_microseconds=percentile(durations, 0.99),
        median_absolute_deviation_microseconds=statistics.median(deviations),
        confidence_level=confidence_level,
        median_ci_low_microseconds=percentile(bootstrapped, alpha),
        median_ci_high_microseconds=percentile(bootstrapped, 1.0 - alpha),
    )


def _numeric_fact(edge: TopologyEdge, name: str) -> float | None:
    fact = edge.fact(name)
    if fact is None or fact.state is not FactState.KNOWN or isinstance(fact.value, bool):
        return None
    if isinstance(fact.value, (int, float)):
        return float(fact.value)
    return None


def _path_for(graph: DiscoveryTopologyGraph, primitive: Primitive) -> tuple[TopologyEdge, ...]:
    kinds: set[EdgeKind]
    if primitive is Primitive.GPU_P2P:
        kinds = {EdgeKind.GPU_GPU}
    elif primitive in _COLLECTIVES or primitive is Primitive.KV_TRANSFER:
        kinds = {EdgeKind.GPU_GPU, EdgeKind.GPU_NIC, EdgeKind.NIC_NETWORK}
    elif primitive in {Primitive.H2D_PAGEABLE, Primitive.H2D_PINNED, Primitive.D2H}:
        kinds = {EdgeKind.CPU_GPU}
    elif primitive is Primitive.HOST_MEMCPY:
        kinds = {EdgeKind.CPU_MEMORY}
    else:
        kinds = set()
    candidates = tuple(edge for edge in graph.edges if edge.kind in kinds)
    if not candidates:
        return ()
    # Keep the slowest measured edge in the calibrated path so degraded fixture
    # behavior is visible instead of optimistically using the best link.
    ranked = sorted(
        candidates,
        key=lambda edge: (_numeric_fact(edge, "measured_bandwidth") or math.inf, edge.edge_id),
    )
    if primitive in _COLLECTIVES or primitive is Primitive.KV_TRANSFER:
        by_kind: dict[EdgeKind, TopologyEdge] = {}
        for edge in ranked:
            by_kind.setdefault(edge.kind, edge)
        return tuple(by_kind[kind] for kind in sorted(by_kind, key=str))
    return (ranked[0],)


def _resources(graph: DiscoveryTopologyGraph) -> tuple[float, float]:
    gpu_bandwidths: list[float] = []
    for node in graph.nodes:
        if node.kind is not NodeKind.GPU:
            continue
        fact = node.fact("memory_bandwidth")
        if (
            fact
            and fact.state is FactState.KNOWN
            and isinstance(fact.value, (int, float))
            and not isinstance(fact.value, bool)
        ):
            gpu_bandwidths.append(float(fact.value))
    return (min(gpu_bandwidths, default=900_000_000_000.0), 4.0)


def _base_curve(
    graph: DiscoveryTopologyGraph,
    primitive: Primitive,
    message_bytes: int,
    rank_count: int,
    concurrency: int,
) -> tuple[float, float | None, tuple[TopologyEdge, ...]]:
    path = _path_for(graph, primitive)
    gpu_bandwidth, launch_us = _resources(graph)
    bandwidths = [value for edge in path if (value := _numeric_fact(edge, "measured_bandwidth"))]
    latencies = [value for edge in path if (value := _numeric_fact(edge, "latency")) is not None]
    bandwidth = min(bandwidths, default=gpu_bandwidth)
    path_latency = sum(latencies) if latencies else 0.0
    contention = 1.0 + 0.22 * (concurrency - 1)
    if len({edge.contention_domain for edge in path if edge.contention_domain}) < len(path):
        contention += 0.08

    if primitive is Primitive.KERNEL_LAUNCH:
        duration = launch_us
    elif primitive is Primitive.DEVICE_SYNCHRONIZE:
        duration = launch_us * 1.6
    elif primitive is Primitive.STARTUP:
        duration = 420_000.0 + 24_000.0 * rank_count
    elif primitive is Primitive.GROUP_INITIALIZATION:
        duration = 19_000.0 + 7_500.0 * math.log2(max(2, rank_count))
    else:
        volume_factor = 1.0
        if primitive is Primitive.ALL_REDUCE:
            volume_factor = 2.0 * (rank_count - 1) / rank_count
        elif primitive in {Primitive.ALL_GATHER, Primitive.REDUCE_SCATTER}:
            volume_factor = (rank_count - 1) / rank_count
        elif primitive in {
            Primitive.ALL_TO_ALL,
            Primitive.EXPERT_DISPATCH,
            Primitive.EXPERT_COMBINE,
        }:
            volume_factor = (rank_count - 1) / rank_count * 1.18
        elif primitive is Primitive.BROADCAST:
            volume_factor = math.ceil(math.log2(max(2, rank_count)))
        if primitive is Primitive.H2D_PAGEABLE:
            bandwidth *= 0.58
        elif primitive is Primitive.H2D_PINNED:
            bandwidth *= 0.91
        elif primitive in {Primitive.GEMM, Primitive.PREFILL}:
            bandwidth *= 0.72
        elif primitive is Primitive.DECODE:
            bandwidth *= 0.42
        elif primitive in {Primitive.EXPERT_DISPATCH, Primitive.EXPERT_COMBINE}:
            contention *= 1.28  # deterministic hot-expert skew in the synthetic suite
        transfer_us = (message_bytes * volume_factor / max(1.0, bandwidth)) * 1_000_000.0
        duration = launch_us + path_latency + transfer_us
    duration *= contention
    effective_throughput = (
        message_bytes / (duration / 1_000_000.0) if message_bytes and duration > 0 else None
    )
    return duration, effective_throughput, path


def _placement(graph: DiscoveryTopologyGraph, rank_count: int) -> Placement:
    visible_ids = set(graph.visibility.visible_gpu_ids)

    def visible(node_id: str, uuid_fact: object) -> bool:
        uuid = uuid_fact.value if isinstance(uuid_fact, ObservedFact) else None
        return not visible_ids or node_id in visible_ids or str(uuid or "") in visible_ids

    gpu_nodes = [
        node
        for node in graph.nodes
        if node.kind is NodeKind.GPU and visible(node.node_id, node.fact("uuid"))
    ]
    selected = gpu_nodes[:rank_count]
    hosts = tuple(dict.fromkeys(node.host_id for node in selected)) or (graph.topology_id,)
    return Placement(
        hosts=hosts,
        ranks=tuple(range(rank_count)),
        gpu_ids=tuple(node.node_id for node in selected),
        numa_domains=tuple(f"{node.host_id}/numa/0" for node in selected),
        nic_ids=tuple(f"{host}/nic/0" for host in hosts),
    )


def _case_id(primitive: Primitive, message_bytes: int, rank_count: int, concurrency: int) -> str:
    return f"{primitive.value}-b{message_bytes}-r{rank_count}-c{concurrency}"


def benchmark_synthetic_fabric(
    graph: DiscoveryTopologyGraph | CanonicalTopologyGraph,
    *,
    seed: int,
    suite: str = "quick",
    warmup_count: int = 3,
    sample_count: int = 11,
    output_dir: Path | None = None,
) -> FabricProfile:
    """Generate explicitly synthetic samples calibrated from fixture link curves."""
    if suite not in {"quick", "full"}:
        raise ValueError("suite must be quick or full")
    if warmup_count < 0 or sample_count < 3:
        raise ValueError("warmup_count must be non-negative and sample_count at least three")
    benchmark_graph, input_fingerprint = normalize_benchmark_topology(graph)
    primitives = _QUICK_PRIMITIVES if suite == "quick" else _FULL_PRIMITIVES
    visible_gpu_count = len(benchmark_graph.visibility.visible_gpu_ids)
    results: list[BenchmarkResult] = []
    for primitive in primitives:
        sizes = (0,) if primitive in _NO_PAYLOAD else (1_024, 65_536, 1_048_576, 16_777_216)
        if primitive in _COLLECTIVES:
            ranks = tuple(value for value in (2, 4, 8) if value <= visible_gpu_count) or (2,)
        elif primitive in {Primitive.GPU_P2P, Primitive.KV_TRANSFER}:
            ranks = (2,)
        else:
            ranks = (1,)
        concurrencies = (1,) if primitive in _NO_PAYLOAD else (1, 2)
        for message_bytes in sizes:
            for rank_count in ranks:
                for concurrency in concurrencies:
                    case_id = _case_id(primitive, message_bytes, rank_count, concurrency)
                    base_us, _, path = _base_curve(
                        benchmark_graph, primitive, message_bytes, rank_count, concurrency
                    )
                    case_seed = seed ^ int.from_bytes(
                        hashlib.sha256(case_id.encode()).digest()[:8], "big"
                    )
                    topology_path = tuple(edge.edge_id for edge in path)
                    contention_domains = tuple(
                        dict.fromkeys(
                            edge.contention_domain
                            for edge in path
                            if edge.contention_domain is not None
                        )
                    )
                    case = BenchmarkCase(
                        case_id=case_id,
                        primitive=primitive,
                        message_bytes=message_bytes,
                        rank_count=rank_count,
                        concurrency=concurrency,
                        direction=(
                            Direction.BIDIRECTIONAL
                            if primitive in {Primitive.GPU_P2P, Primitive.KV_TRANSFER}
                            else Direction.NOT_APPLICABLE
                        ),
                        topology_path=topology_path,
                        contention_domains=contention_domains,
                        placement=_placement(benchmark_graph, rank_count),
                        warmup_count=warmup_count,
                        sample_count=sample_count,
                        invocation=Invocation(
                            adapter="sloforge-synthetic-calibrated-v1",
                            adapter_version="1.0.0",
                            argv=(
                                "sloforge",
                                "fabric",
                                "benchmark",
                                "--mode",
                                "synthetic",
                                "--seed",
                                str(seed),
                            ),
                            timeout_seconds=30.0,
                        ),
                    )
                    raw_artifact = str(Path("raw") / f"{case_id}.json")
                    required_gpu_count = (
                        0 if primitive is Primitive.HOST_MEMCPY else max(1, rank_count)
                    )
                    path_required = primitive in {
                        Primitive.GPU_P2P,
                        Primitive.H2D_PAGEABLE,
                        Primitive.H2D_PINNED,
                        Primitive.D2H,
                        Primitive.KV_TRANSFER,
                        *_COLLECTIVES,
                    }
                    if visible_gpu_count < required_gpu_count or (path_required and not path):
                        reason = (
                            f"requires {required_gpu_count} visible GPUs, found {visible_gpu_count}"
                            if visible_gpu_count < required_gpu_count
                            else f"no calibrated physical path for {primitive.value}"
                        )
                        results.append(
                            finalize_result(
                                schema_version="sloforge.fabric.benchmark-result/v1",
                                case=case,
                                mode=MeasurementMode.UNAVAILABLE,
                                status=BenchmarkStatus.UNAVAILABLE,
                                raw_samples=(),
                                summary=None,
                                environment=_environment(),
                                failure_reason=reason,
                                raw_artifact=raw_artifact,
                            )
                        )
                        continue
                    rng = random.Random(case_seed)
                    # Warmups affect generator state and are retained in invocation
                    # metadata, but are intentionally excluded from raw measurements.
                    for _ in range(warmup_count):
                        rng.lognormvariate(0.0, 0.025)
                    raw: list[RawSample] = []
                    for sample_index in range(sample_count):
                        duration_us = base_us * rng.lognormvariate(0.0, 0.025)
                        throughput = (
                            message_bytes / (duration_us / 1_000_000.0)
                            if message_bytes and duration_us > 0
                            else None
                        )
                        raw.append(
                            RawSample(
                                sample_index=sample_index,
                                duration_microseconds=duration_us,
                                throughput_bytes_per_second=throughput,
                                synthetic=True,
                                seed=case_seed,
                            )
                        )
                    samples = tuple(raw)
                    results.append(
                        finalize_result(
                            schema_version="sloforge.fabric.benchmark-result/v1",
                            case=case,
                            mode=MeasurementMode.SYNTHETIC_CALIBRATED,
                            status=BenchmarkStatus.SUCCESS,
                            raw_samples=samples,
                            summary=summarize_samples(samples, seed=case_seed),
                            environment=_environment(),
                            failure_reason=None,
                            raw_artifact=raw_artifact,
                        )
                    )
    profile = finalize_profile(
        schema_version="sloforge.fabric.profile/v1",
        profile_id=f"synthetic-{benchmark_graph.topology_id}-{seed}",
        captured_at=utc_now(),
        topology_fingerprint=input_fingerprint,
        seed=seed,
        suite=suite,
        results=tuple(results),
        environment=_environment(),
    )
    if output_dir is not None:
        save_profile(output_dir, profile)
    return profile


def benchmark_host_memory(
    *,
    message_bytes: int,
    warmup_count: int,
    sample_count: int,
    seed: int,
    timeout_seconds: float = 20.0,
) -> BenchmarkResult:
    """Measure an actual CPU memcpy without substituting another device."""
    if not 1 <= message_bytes <= 256 * 1024 * 1024:
        raise ValueError("host-memory benchmark size must be in [1 byte, 256 MiB]")
    if not 0 <= warmup_count <= 1_000 or not 1 <= sample_count <= 10_000:
        raise ValueError("host-memory benchmark iteration count is outside safety bounds")
    source = np.arange(message_bytes, dtype=np.uint8)
    target = np.empty_like(source)
    started_suite = time.monotonic()
    measured: list[RawSample] = []
    status: BenchmarkStatus
    failure_reason: str | None
    samples: tuple[RawSample, ...]
    summary: RobustSummary | None
    try:
        for iteration in range(warmup_count + sample_count):
            if time.monotonic() - started_suite > timeout_seconds:
                raise TimeoutError("host-memory benchmark exceeded its bounded timeout")
            started = time.perf_counter_ns()
            np.copyto(target, source)
            duration_us = max(0.001, (time.perf_counter_ns() - started) / 1_000.0)
            if iteration >= warmup_count:
                measured.append(
                    RawSample(
                        sample_index=iteration - warmup_count,
                        duration_microseconds=duration_us,
                        throughput_bytes_per_second=message_bytes / (duration_us / 1_000_000.0),
                        synthetic=False,
                        seed=None,
                    )
                )
    except TimeoutError as error:
        status = BenchmarkStatus.FAILED
        failure_reason = str(error)
        samples = ()
        summary = None
    else:
        status = BenchmarkStatus.SUCCESS
        failure_reason = None
        samples = tuple(measured)
        summary = summarize_samples(samples, seed=seed)
    case = BenchmarkCase(
        case_id=_case_id(Primitive.HOST_MEMCPY, message_bytes, 1, 1),
        primitive=Primitive.HOST_MEMCPY,
        message_bytes=message_bytes,
        rank_count=1,
        concurrency=1,
        direction=Direction.FORWARD,
        topology_path=(),
        contention_domains=("host-memory",),
        placement=Placement(
            hosts=(platform.node() or "localhost",),
            ranks=(0,),
            gpu_ids=(),
            numa_domains=(),
            nic_ids=(),
        ),
        warmup_count=warmup_count,
        sample_count=sample_count,
        invocation=Invocation(
            adapter="numpy-copyto",
            adapter_version=np.__version__,
            argv=("numpy.copyto",),
            timeout_seconds=timeout_seconds,
        ),
    )
    return finalize_result(
        schema_version="sloforge.fabric.benchmark-result/v1",
        case=case,
        mode=MeasurementMode.MEASURED,
        status=status,
        raw_samples=samples,
        summary=summary,
        environment=_environment(),
        failure_reason=failure_reason,
        raw_artifact=None,
    )


def save_profile(output_dir: Path, profile: FabricProfile) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for result in profile.results:
        if result.raw_artifact is None:
            continue
        artifact_path = output_dir / result.raw_artifact
        resolved_output = output_dir.resolve()
        if resolved_output not in artifact_path.resolve().parents:
            raise ValueError(f"raw artifact escapes output directory: {result.raw_artifact}")
        artifact = RawSampleArtifact(
            case_id=result.case.case_id,
            mode=result.mode,
            samples=result.raw_samples,
            benchmark_artifact_hash=result.artifact_hash,
        )
        write_json(artifact_path, artifact.model_dump(mode="json"))
    write_json(output_dir / "profile.json", profile.model_dump(mode="json"))


def load_profile(path: Path) -> FabricProfile:
    profile_path = path / "profile.json" if path.is_dir() else path
    profile = FabricProfile.model_validate_json(profile_path.read_text(encoding="utf-8"))
    if path.is_dir():
        resolved_output = path.resolve()
        for result in profile.results:
            if result.raw_artifact is None:
                continue
            artifact_path = path / result.raw_artifact
            if resolved_output not in artifact_path.resolve().parents:
                raise ValueError(f"raw artifact escapes profile directory: {result.raw_artifact}")
            artifact = RawSampleArtifact.model_validate_json(
                artifact_path.read_text(encoding="utf-8")
            )
            if (
                artifact.case_id != result.case.case_id
                or artifact.mode is not result.mode
                or artifact.samples != result.raw_samples
                or artifact.benchmark_artifact_hash != result.artifact_hash
            ):
                raise ValueError(f"raw artifact does not match profile: {result.raw_artifact}")
    return profile
