"""Topology-aware Fabric discovery, profiling, compilation, and simulation CLI."""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Annotated, Literal

import typer

from sloforge.fabric.compiler import (
    CompilerAssumptions,
    CompilerConstraints,
    CompilerObjective,
    CompilerRequest,
    OptimizationStrategy,
    compile_physical_plan,
)
from sloforge.fabric.ir import (
    DocumentReference,
    PhysicalExecutionPlan,
    canonical_hash,
    load_fabric_profile,
    load_physical_execution_plan,
    load_topology_graph,
    save_fabric_profile,
    save_model_graph,
    save_physical_execution_plan,
)
from sloforge.fabric.model_graph import inspect_local_model, synthetic_moe_model_graph
from sloforge.fabric.profiling import (
    BenchmarkStatus,
    benchmark_host_memory,
    benchmark_synthetic_fabric,
    to_canonical_profile,
)
from sloforge.fabric.simulation import (
    FabricSimulationOutput,
    SimulationRequestShape,
    SimulationWorkload,
    build_simulation_request,
    request_latencies,
    run_simulation,
)
from sloforge.fabric.topology import (
    build_canonical_fixture,
    discover_topology,
    save_topology,
)
from sloforge.ir import ArtifactDigest, load_deployment_plan
from sloforge.trace.format import load_trace
from sloforge.util import percentile, sha256_file, write_json

from .common import (
    console,
    current_git_commit,
    environment_digest,
    json_result,
    repository_root,
)

fabric_app = typer.Typer(help="Compile and validate topology-aware physical execution plans.")
model_app = typer.Typer(help="Inspect model structure without executing model code.")
fabric_app.add_typer(model_app, name="model")


def _profile_path(path: Path) -> Path:
    return path / "fabric-profile.json" if path.is_dir() else path


def _parse_slo(value: str) -> tuple[float, float]:
    supported = {"p95_ttft_ms", "p99_tpot_ms", "p99_itl_ms"}
    parsed: dict[str, float] = {}
    for expression in value.split(","):
        name, separator, raw = expression.strip().partition("<=")
        if not separator or name not in supported:
            raise typer.BadParameter("SLO entries must be p95_ttft_ms<=N and p99_tpot_ms<=N")
        try:
            threshold = float(raw)
        except ValueError as error:
            raise typer.BadParameter(f"invalid SLO threshold {raw!r}") from error
        if threshold <= 0.0:
            raise typer.BadParameter("SLO thresholds must be positive")
        parsed[name] = threshold
    tpot = parsed.get("p99_tpot_ms", parsed.get("p99_itl_ms"))
    if "p95_ttft_ms" not in parsed or tpot is None:
        raise typer.BadParameter("both p95_ttft_ms and p99_tpot_ms constraints are required")
    return parsed["p95_ttft_ms"], tpot


def _workload_from_trace(path: Path, *, cpu_launch_us: float) -> SimulationWorkload:
    records = load_trace(path)
    first_arrival = records[0].arrival_ms
    shapes = tuple(
        SimulationRequestShape(
            arrival_us=(record.arrival_ms - first_arrival) * 1_000.0,
            prompt_tokens=record.prompt_tokens,
            output_tokens=record.output_tokens,
            priority="high"
            if record.priority == 0
            else "normal"
            if record.priority == 1
            else "low",
            request_class=record.request_class,
        )
        for record in records
    )
    arrivals = [shape.arrival_us for shape in shapes]
    interval = (arrivals[-1] - arrivals[0]) / (len(arrivals) - 1) if len(arrivals) > 1 else 0.0
    return SimulationWorkload(
        request_count=len(shapes),
        arrival_interval_us=interval,
        prompt_tokens=max(1, round(statistics.median(item.prompt_tokens for item in shapes))),
        output_tokens=max(1, round(statistics.median(item.output_tokens for item in shapes))),
        cpu_launch_us=cpu_launch_us,
        requests=shapes,
    )


@fabric_app.command("discover")
def discover_command(
    output: Annotated[Path, typer.Option("--output", "-o")],
    fixture: Annotated[str | None, typer.Option("--fixture")] = None,
) -> None:
    """Discover this host, or explicitly load a deterministic fixture."""

    graph = build_canonical_fixture(fixture) if fixture is not None else discover_topology()
    save_topology(output, graph)
    json_result(
        {
            "output": str(output),
            "topology_id": graph.topology_id,
            "fingerprint": canonical_hash(graph),
            "nodes": len(graph.nodes),
            "edges": len(graph.edges),
            "source": f"fixture:{fixture}" if fixture is not None else "current-host",
        }
    )


@fabric_app.command("benchmark")
def benchmark_command(
    topology: Annotated[Path, typer.Option("--topology", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    suite: Annotated[Literal["quick", "full", "host-memory"], typer.Option("--suite")] = "quick",
    synthetic: Annotated[bool, typer.Option("--synthetic/--measured")] = False,
    seed: Annotated[int, typer.Option("--seed", min=0)] = 41,
    warmups: Annotated[int, typer.Option("--warmups", min=0)] = 3,
    samples: Annotated[int, typer.Option("--samples", min=3)] = 7,
) -> None:
    """Measure host memory or explicitly generate calibrated synthetic fabric curves."""

    graph = load_topology_graph(topology)
    output.mkdir(parents=True, exist_ok=True)
    if not synthetic:
        if suite != "host-memory":
            raise typer.BadParameter(
                "measured quick/full suites require a hardware adapter; use --synthetic "
                "for fixtures or --suite host-memory for the portable measured probe"
            )
        result = benchmark_host_memory(
            message_bytes=1 << 20,
            warmup_count=warmups,
            sample_count=samples,
            seed=seed,
        )
        path = output / "host-memory.json"
        write_json(path, result.model_dump(mode="json"))
        json_result(
            {
                "output": str(path),
                "mode": result.mode.value,
                "status": result.status.value,
                "artifact_hash": result.artifact_hash,
            }
        )
        return
    if suite == "host-memory":
        raise typer.BadParameter("--suite host-memory is measured and cannot use --synthetic")
    raw = benchmark_synthetic_fabric(
        graph,
        seed=seed,
        suite=suite,
        warmup_count=warmups,
        sample_count=samples,
        output_dir=output / "raw",
    )
    unavailable = [
        item.case.case_id for item in raw.results if item.status is not BenchmarkStatus.SUCCESS
    ]
    if unavailable:
        raise typer.BadParameter(
            f"synthetic suite has {len(unavailable)} unavailable cases for this explicit "
            f"topology; raw results remain in {output / 'raw'}"
        )
    profile = to_canonical_profile(raw, topology=graph)
    profile_path = output / "fabric-profile.json"
    save_fabric_profile(profile_path, profile)
    json_result(
        {
            "output": str(profile_path),
            "raw_output": str(output / "raw"),
            "mode": "synthetic_calibrated",
            "measurements": len(profile.measurements),
            "profile_hash": canonical_hash(profile),
        }
    )


@model_app.command("inspect")
def model_inspect_command(
    model: Annotated[str, typer.Option("--model")],
    output: Annotated[Path, typer.Option("--output", "-o")],
    revision: Annotated[str, typer.Option("--revision")] = "local",
    synthetic_moe: Annotated[bool, typer.Option("--synthetic-moe")] = False,
) -> None:
    """Inspect a resolved local model; synthetic input always requires an explicit flag."""

    if synthetic_moe:
        if model not in {"synthetic", "sloforge/synthetic-moe-fabric-v1"}:
            raise typer.BadParameter("--synthetic-moe requires --model synthetic")
        graph = synthetic_moe_model_graph()
        source = "explicit-synthetic-fixture"
    else:
        model_path = Path(model)
        if not model_path.is_dir():
            raise typer.BadParameter(
                "model inspection is offline: --model must be a resolved local directory"
            )
        graph = inspect_local_model(model_path, model_id=model, revision=revision)
        source = str(model_path.resolve())
    save_model_graph(output, graph)
    json_result(
        {
            "output": str(output),
            "model_id": graph.model_id,
            "model_graph_hash": canonical_hash(graph),
            "layers": len(graph.layers),
            "source": source,
        }
    )


@fabric_app.command("compile")
def compile_command(
    deployment_plan: Annotated[
        Path, typer.Option("--deployment-plan", exists=True, dir_okay=False)
    ],
    topology: Annotated[Path, typer.Option("--topology", exists=True, dir_okay=False)],
    fabric_profile: Annotated[Path, typer.Option("--fabric-profile", exists=True)],
    model_graph: Annotated[Path, typer.Option("--model-graph", exists=True, dir_okay=False)],
    slo: Annotated[str, typer.Option("--slo")],
    output: Annotated[Path, typer.Option("--output", "-o")],
    objective: Annotated[CompilerObjective, typer.Option("--objective")] = (
        CompilerObjective.ROBUST_BALANCED
    ),
    strategy: Annotated[OptimizationStrategy, typer.Option("--strategy")] = (
        OptimizationStrategy.HIERARCHICAL
    ),
    prompt_tokens_p95: Annotated[int, typer.Option("--prompt-tokens-p95", min=1)] = 2_048,
    output_tokens_p95: Annotated[int, typer.Option("--output-tokens-p95", min=1)] = 256,
    concurrency: Annotated[int, typer.Option("--concurrency", min=1)] = 8,
    maximum_ranks: Annotated[int, typer.Option("--maximum-ranks", min=1)] = 8,
    prefill_tokens_per_second: Annotated[
        float, typer.Option("--prefill-tokens-per-second", min=0.001)
    ] = 8_000.0,
    decode_tokens_per_second: Annotated[
        float, typer.Option("--decode-tokens-per-second", min=0.001)
    ] = 120.0,
    gpu_hourly_price_usd: Annotated[float, typer.Option("--gpu-hourly-price-usd", min=0.0)] = 2.0,
    require_disaggregation: Annotated[bool, typer.Option("--require-disaggregation")] = False,
    seed: Annotated[int, typer.Option("--seed", min=0)] = 41,
) -> None:
    """Compile a logical plan into a typed physical execution plan."""

    from sloforge.fabric.ir import load_model_graph

    logical = load_deployment_plan(deployment_plan)
    graph = load_topology_graph(topology)
    profile = load_fabric_profile(_profile_path(fabric_profile))
    model = load_model_graph(model_graph)
    ttft, tpot = _parse_slo(slo)
    request = CompilerRequest(
        logical_deployment_plan=DocumentReference(
            kind="DeploymentPlan",
            api_version=logical.api_version,
            uri=str(deployment_plan.resolve()),
            digest=ArtifactDigest(value=sha256_file(deployment_plan)),
            uid=logical.metadata.uid,
            generation=logical.metadata.generation,
        ),
        model=model,
        topology=graph,
        fabric_profile=profile,
        constraints=CompilerConstraints(
            prompt_tokens_p95=prompt_tokens_p95,
            output_tokens_p95=output_tokens_p95,
            maximum_concurrent_requests=concurrency,
            p95_ttft_ms=ttft,
            p99_tpot_ms=tpot,
            maximum_ranks=maximum_ranks,
            require_disaggregation=require_disaggregation,
        ),
        assumptions=CompilerAssumptions(
            prefill_tokens_per_second_per_gpu=prefill_tokens_per_second,
            decode_tokens_per_second_per_gpu=decode_tokens_per_second,
            gpu_hourly_price_usd=gpu_hourly_price_usd,
            base_availability=0.999,
            cold_start_ms=2_000.0,
            measurement_relative_uncertainty=0.10,
        ),
        objective=objective,
        strategy=strategy,
        generated_at=graph.discovered_at,
        seed=seed,
        git_commit=current_git_commit(),
        environment_digest=environment_digest(),
    )
    result = compile_physical_plan(request)
    save_physical_execution_plan(output, result.selected)
    optimization_path = output.with_suffix(".optimization.json")
    write_json(optimization_path, result.model_dump(mode="json"))
    json_result(
        {
            "output": str(output),
            "optimization": str(optimization_path),
            "plan_id": result.selected.plan_id,
            "strategy": result.strategy.value,
            "pareto_candidates": len(result.pareto_frontier),
            "rejected_candidates": len(result.selected.rejected_alternatives),
            "recovery_variants": len(result.selected.recovery_variants),
        }
    )


@fabric_app.command("explain")
def explain_command(
    plan: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    physical = load_physical_execution_plan(plan)
    roles: dict[str, int] = {}
    for binding in physical.rank_placement.bindings:
        roles[binding.worker_role.value] = roles.get(binding.worker_role.value, 0) + 1
    console.print(
        f"Physical plan {physical.plan_id}\n"
        f"  Parallelism: TP={physical.parallelism.tensor_parallel_degree}, "
        f"PP={physical.parallelism.pipeline_parallel_degree}, "
        f"DP={physical.parallelism.data_parallel_degree}, "
        f"EP={physical.parallelism.expert_parallel_degree}\n"
        f"  Ranks: {len(physical.rank_placement.bindings)} ({roles})\n"
        f"  Bottleneck: {physical.bottleneck_prediction}\n"
        f"  Predicted p95 TTFT: {physical.predicted_metrics.p95_ttft_ms.estimate:.3f} ms\n"
        f"  Predicted p99 TPOT: {physical.predicted_metrics.p99_tpot_ms.estimate:.3f} ms\n"
        f"  Recovery variants: {len(physical.recovery_variants)}\n"
        f"  Evidence records: {len(physical.evidence)}"
    )


def _simulate(
    *,
    plan_path: Path,
    topology_path: Path,
    profile_path: Path,
    trace_path: Path,
    output: Path,
    seed: int,
    timeout_seconds: float,
    cpu_launch_us: float,
) -> tuple[PhysicalExecutionPlan, FabricSimulationOutput]:
    physical = load_physical_execution_plan(plan_path)
    topology = load_topology_graph(topology_path)
    profile = load_fabric_profile(_profile_path(profile_path))
    workload = _workload_from_trace(trace_path, cpu_launch_us=cpu_launch_us)
    request = build_simulation_request(
        physical,
        topology,
        profile,
        workload,
        seed=seed,
    )
    result = run_simulation(
        request,
        repository_root=repository_root(),
        timeout_seconds=timeout_seconds,
    )
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "request.json", request.model_dump(mode="json"))
    write_json(output / "result.json", result.model_dump(mode="json"))
    write_json(
        output / "trace.json",
        {
            "traceEvents": [item.model_dump(mode="json") for item in result.trace_events],
            "displayTimeUnit": "us",
        },
    )
    return physical, result


@fabric_app.command("simulate")
def simulate_command(
    plan: Annotated[Path, typer.Option("--plan", exists=True, dir_okay=False)],
    topology: Annotated[Path, typer.Option("--topology", exists=True, dir_okay=False)],
    fabric_profile: Annotated[Path, typer.Option("--fabric-profile", exists=True)],
    trace: Annotated[Path, typer.Option("--trace", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    seed: Annotated[int, typer.Option("--seed", min=0)] = 41,
    timeout_seconds: Annotated[float, typer.Option("--timeout-seconds", min=0.1)] = 30.0,
    cpu_launch_us: Annotated[float, typer.Option("--cpu-launch-us", min=0.001)] = 5.0,
) -> None:
    _, result = _simulate(
        plan_path=plan,
        topology_path=topology,
        profile_path=fabric_profile,
        trace_path=trace,
        output=output,
        seed=seed,
        timeout_seconds=timeout_seconds,
        cpu_launch_us=cpu_launch_us,
    )
    json_result(
        {
            "output": str(output / "result.json"),
            "makespan_ms": result.metrics.makespan_us / 1_000.0,
            "operations": result.metrics.operation_count,
            "cost_usd": result.metrics.cost_usd,
        }
    )


@fabric_app.command("validate")
def validate_command(
    plan: Annotated[Path, typer.Option("--plan", exists=True, dir_okay=False)],
    topology: Annotated[Path, typer.Option("--topology", exists=True, dir_okay=False)],
    fabric_profile: Annotated[Path, typer.Option("--fabric-profile", exists=True)],
    trace: Annotated[Path, typer.Option("--trace", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    seed: Annotated[int, typer.Option("--seed", min=0)] = 41,
    timeout_seconds: Annotated[float, typer.Option("--timeout-seconds", min=0.1)] = 30.0,
) -> None:
    physical, result = _simulate(
        plan_path=plan,
        topology_path=topology,
        profile_path=fabric_profile,
        trace_path=trace,
        output=output,
        seed=seed,
        timeout_seconds=timeout_seconds,
        cpu_launch_us=5.0,
    )
    latencies = request_latencies(result)
    observed_ttft = percentile([item.ttft_us / 1_000.0 for item in latencies], 0.95)
    predicted_ttft = physical.predicted_metrics.p95_ttft_ms.estimate
    validation = {
        "schema_version": "sloforge.fabric.validation/v1",
        "plan_id": physical.plan_id,
        "seed": seed,
        "request_count": len(latencies),
        "predicted_p95_ttft_ms": predicted_ttft,
        "observed_p95_ttft_ms": observed_ttft,
        "absolute_error_ms": abs(observed_ttft - predicted_ttft),
        "relative_error": abs(observed_ttft - predicted_ttft) / max(predicted_ttft, 1e-9),
        "simulation_result_sha256": sha256_file(output / "result.json"),
    }
    write_json(output / "validation.json", validation)
    json_result({"output": str(output / "validation.json"), **validation})
