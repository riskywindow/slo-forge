"""Topology-aware Fabric discovery, profiling, compilation, and simulation CLI."""

from __future__ import annotations

import shutil
import statistics
from pathlib import Path
from typing import Annotated, Literal

import typer

from sloforge.fabric.adapters import (
    DeploymentTarget,
    DynamoBackend,
    FabricAdapterContext,
    GangScheduler,
    RuntimeKind,
    export_physical_plan,
)
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
    GpuNode,
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
    NvidiaInventoryRecord,
    benchmark_host_memory,
    benchmark_synthetic_fabric,
    build_nccl_tests_command,
    build_nvidia_smi_command,
    read_nvidia_inventory,
    run_nccl_tests_profile,
    save_profile,
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
    samples: Annotated[int, typer.Option("--samples", min=3, max=100)] = 7,
    adapter: Annotated[Literal["nccl-tests"] | None, typer.Option("--adapter")] = None,
    transport: Annotated[Literal["nccl-local"] | None, typer.Option("--transport")] = None,
    adapter_executable: Annotated[Path | None, typer.Option("--adapter-executable")] = None,
    visible_devices: Annotated[list[str] | None, typer.Option("--visible-device")] = None,
    operation: Annotated[
        Literal[
            "all_reduce",
            "all_gather",
            "reduce_scatter",
            "broadcast",
            "send_receive",
            "all_to_all",
        ],
        typer.Option("--operation"),
    ] = "all_reduce",
    inner_iterations: Annotated[int, typer.Option("--inner-iterations", min=1, max=10_000)] = 20,
    timeout_seconds: Annotated[
        float, typer.Option("--timeout-seconds", min=0.1, max=3_600.0)
    ] = 120.0,
    maximum_output_bytes: Annotated[
        int, typer.Option("--maximum-output-bytes", min=4_096, max=16 * 1024 * 1024)
    ] = 1 << 20,
    nvidia_smi_executable: Annotated[Path | None, typer.Option("--nvidia-smi-executable")] = None,
    adapter_version: Annotated[str | None, typer.Option("--adapter-version")] = None,
) -> None:
    """Run an explicit measured adapter or generate calibrated synthetic curves."""

    graph = load_topology_graph(topology)
    output.mkdir(parents=True, exist_ok=True)
    if not synthetic:
        if suite == "host-memory":
            if any(
                value is not None
                for value in (adapter, transport, adapter_executable, visible_devices)
            ):
                raise typer.BadParameter("host-memory does not accept GPU adapter options")
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
        if adapter != "nccl-tests" or transport != "nccl-local":
            raise typer.BadParameter(
                "measured quick/full requires explicit --adapter nccl-tests "
                "--transport nccl-local; use --synthetic for fixtures or --suite host-memory "
                "for the portable probe"
            )
        if adapter_executable is None or not visible_devices:
            raise typer.BadParameter(
                "measured NCCL requires --adapter-executable and at least one "
                "--visible-device; ambient GPU visibility is never inherited"
            )
        gpu_nodes = tuple(node for node in graph.nodes if isinstance(node, GpuNode))
        matched = tuple(
            node
            for device in visible_devices
            for node in gpu_nodes
            if device in {node.node_id, node.uuid}
        )
        if len(matched) != len(visible_devices) or len({node.node_id for node in matched}) != len(
            matched
        ):
            raise typer.BadParameter(
                "every explicit device must uniquely match a GPU node ID or UUID in the topology"
            )
        if len({node.host_id for node in matched}) != 1:
            raise typer.BadParameter(
                "the local NCCL runner cannot span hosts; use an external bounded orchestrator"
            )
        minimum_bytes = 1_024
        maximum_bytes = 1 << (20 if suite == "quick" else 24)
        step_factor = 2
        command = build_nccl_tests_command(
            executable=adapter_executable,
            operation=operation,
            minimum_bytes=minimum_bytes,
            maximum_bytes=maximum_bytes,
            step_factor=step_factor,
            gpus_per_process=len(visible_devices),
            visible_devices=tuple(visible_devices),
            iterations=inner_iterations,
            warmups=warmups,
            timeout_seconds=timeout_seconds,
            transport="local",
        )
        inventory: tuple[NvidiaInventoryRecord, ...] = ()
        if nvidia_smi_executable is not None:
            inventory_fields = ("uuid", "name", "memory.total", "clocks.sm", "clocks.mem")
            inventory = tuple(
                read_nvidia_inventory(
                    build_nvidia_smi_command(
                        executable=nvidia_smi_executable,
                        gpu_id=device,
                        fields=inventory_fields,
                        timeout_seconds=min(10.0, timeout_seconds),
                    ),
                    gpu_id=device,
                    fields=inventory_fields,
                )
                for device in visible_devices
            )
        raw_output = output / "raw"
        raw_profile = run_nccl_tests_profile(
            command=command,
            operation=operation,
            topology_fingerprint=canonical_hash(graph),
            suite=suite,
            minimum_bytes=minimum_bytes,
            maximum_bytes=maximum_bytes,
            step_factor=step_factor,
            repetitions=samples,
            warmup_count=warmups,
            seed=seed,
            adapter_version=adapter_version,
            inventory=inventory,
            output_dir=raw_output,
            maximum_output_bytes=maximum_output_bytes,
        )
        save_profile(raw_output, raw_profile)
        failed = [
            result for result in raw_profile.results if result.status is not BenchmarkStatus.SUCCESS
        ]
        if failed:
            raise typer.BadParameter(
                f"measured adapter failed closed; inspect {raw_output / 'profile.json'}: "
                f"{failed[0].failure_reason}"
            )
        profile = to_canonical_profile(raw_profile, topology=graph)
        profile_path = output / "fabric-profile.json"
        save_fabric_profile(profile_path, profile)
        json_result(
            {
                "output": str(profile_path),
                "raw_output": str(raw_output),
                "mode": "measured",
                "adapter": adapter,
                "operation": operation,
                "measurements": len(profile.measurements),
                "profile_hash": canonical_hash(profile),
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


@fabric_app.command("export")
def export_command(
    plan: Annotated[Path, typer.Option("--plan", exists=True, dir_okay=False)],
    topology: Annotated[Path, typer.Option("--topology", exists=True, dir_okay=False)],
    target: Annotated[DeploymentTarget, typer.Option("--target")],
    output: Annotated[Path, typer.Option("--output", "-o")],
    model_id: Annotated[str, typer.Option("--model-id")],
    model_revision: Annotated[str, typer.Option("--model-revision")],
    image: Annotated[str, typer.Option("--image")],
    runtime: Annotated[RuntimeKind, typer.Option("--runtime")],
    runtime_version: Annotated[str, typer.Option("--runtime-version")],
    dynamo_backend: Annotated[DynamoBackend | None, typer.Option("--dynamo-backend")] = None,
    namespace: Annotated[str, typer.Option("--namespace")] = "default",
    gpu_resource_name: Annotated[str, typer.Option("--gpu-resource-name")] = "nvidia.com/gpu",
    rdma_resource_name: Annotated[str | None, typer.Option("--rdma-resource-name")] = None,
    gang_scheduler: Annotated[GangScheduler, typer.Option("--gang-scheduler")] = (
        GangScheduler.NONE
    ),
    cpu_limit_per_rank: Annotated[
        float, typer.Option("--cpu-limit-per-rank", min=0.001, max=1_024.0)
    ] = 4.0,
    memory_limit_gib_per_rank: Annotated[
        int, typer.Option("--memory-limit-gib-per-rank", min=1, max=1_048_576)
    ] = 32,
    pids_limit_per_rank: Annotated[
        int, typer.Option("--pids-limit-per-rank", min=64, max=1_048_576)
    ] = 512,
    shutdown_grace_seconds: Annotated[
        int, typer.Option("--shutdown-grace-seconds", min=1, max=3_600)
    ] = 120,
    allow_advisory_cloud_metadata: Annotated[
        bool,
        typer.Option(
            "--allow-advisory-cloud-metadata",
            help="Permit explicitly non-enforcing physical metadata for Modal or Truss.",
        ),
    ] = False,
) -> None:
    """Lower a physical plan into validated offline deployment artifacts."""

    physical = load_physical_execution_plan(plan)
    graph = load_topology_graph(topology)
    try:
        context = FabricAdapterContext(
            plan=physical,
            topology=graph,
            model_id=model_id,
            model_revision=model_revision,
            image=image,
            runtime=runtime,
            runtime_version=runtime_version,
            dynamo_backend=dynamo_backend,
            namespace=namespace,
            gpu_resource_name=gpu_resource_name,
            rdma_resource_name=rdma_resource_name,
            gang_scheduler=gang_scheduler,
            cpu_limit_per_rank=cpu_limit_per_rank,
            memory_limit_gib_per_rank=memory_limit_gib_per_rank,
            pids_limit_per_rank=pids_limit_per_rank,
            shutdown_grace_seconds=shutdown_grace_seconds,
            allow_advisory_cloud_metadata=allow_advisory_cloud_metadata,
        )
    except ValueError as error:
        raise typer.BadParameter(f"invalid physical adapter context: {error}") from error

    if output.exists() or output.is_symlink():
        raise typer.BadParameter(
            "physical export output must not already exist; choose a new directory"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    try:
        result = export_physical_plan(context=context, target=target, output=output)
    except ValueError as error:
        shutil.rmtree(output)
        raise typer.BadParameter(f"physical export rejected: {error}") from error
    except BaseException:
        # This invocation exclusively created the directory, so removing it
        # prevents a failed lowering from being mistaken for a valid export.
        shutil.rmtree(output)
        raise
    json_result(result.model_dump(mode="json"))


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
    max_relative_error: Annotated[float, typer.Option("--max-relative-error", min=0.0)] = 0.25,
    slo: Annotated[
        str | None,
        typer.Option(
            "--slo",
            help="Optional hard p95 TTFT and p99 TPOT/ITL constraints.",
        ),
    ] = None,
    require_prediction_interval: Annotated[
        bool,
        typer.Option("--require-prediction-interval/--ignore-prediction-interval"),
    ] = True,
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
    trace_records = load_trace(trace)
    output_tokens = {
        f"request-{index:06d}": record.output_tokens for index, record in enumerate(trace_records)
    }
    token_steps_ms = [
        operation.duration_us / output_tokens[operation.operation_id.partition(":")[0]] / 1_000.0
        for operation in result.operations
        if operation.operation_id.endswith(":decode")
    ]
    if not token_steps_ms:
        raise typer.BadParameter("simulation produced no decode operations for TPOT validation")
    observed_tpot = percentile(token_steps_ms, 0.99)
    predicted_ttft_interval = physical.predicted_metrics.p95_ttft_ms
    predicted_tpot_interval = physical.predicted_metrics.p99_tpot_ms
    predicted_ttft = predicted_ttft_interval.estimate
    predicted_tpot = predicted_tpot_interval.estimate
    ttft_relative_error = abs(observed_ttft - predicted_ttft) / max(predicted_ttft, 1e-9)
    tpot_relative_error = abs(observed_tpot - predicted_tpot) / max(predicted_tpot, 1e-9)
    ttft_interval_covered = (
        predicted_ttft_interval.lower <= observed_ttft <= predicted_ttft_interval.upper
    )
    tpot_interval_covered = (
        predicted_tpot_interval.lower <= observed_tpot <= predicted_tpot_interval.upper
    )
    failure_reasons: list[str] = []
    if ttft_relative_error > max_relative_error:
        failure_reasons.append(
            f"p95 TTFT relative_error={ttft_relative_error:.6f} exceeds {max_relative_error:.6f}"
        )
    if tpot_relative_error > max_relative_error:
        failure_reasons.append(
            f"p99 TPOT relative_error={tpot_relative_error:.6f} exceeds {max_relative_error:.6f}"
        )
    if require_prediction_interval and not ttft_interval_covered:
        failure_reasons.append("observed p95 TTFT is outside the plan prediction interval")
    if require_prediction_interval and not tpot_interval_covered:
        failure_reasons.append("observed p99 TPOT is outside the plan prediction interval")
    slo_attained: bool | None = None
    slo_p95_ttft_ms: float | None = None
    slo_p99_tpot_ms: float | None = None
    if slo is not None:
        slo_p95_ttft_ms, slo_p99_tpot_ms = _parse_slo(slo)
        slo_attained = observed_ttft <= slo_p95_ttft_ms and observed_tpot <= slo_p99_tpot_ms
        if observed_ttft > slo_p95_ttft_ms:
            failure_reasons.append(
                f"observed p95 TTFT {observed_ttft:.6f} ms exceeds hard SLO "
                f"{slo_p95_ttft_ms:.6f} ms"
            )
        if observed_tpot > slo_p99_tpot_ms:
            failure_reasons.append(
                f"observed p99 TPOT {observed_tpot:.6f} ms exceeds hard SLO "
                f"{slo_p99_tpot_ms:.6f} ms"
            )
    validation = {
        "schema_version": "sloforge.fabric.validation/v1",
        "plan_id": physical.plan_id,
        "seed": seed,
        "request_count": len(latencies),
        "predicted_p95_ttft_ms": predicted_ttft,
        "predicted_p95_ttft_lower_ms": predicted_ttft_interval.lower,
        "predicted_p95_ttft_upper_ms": predicted_ttft_interval.upper,
        "observed_p95_ttft_ms": observed_ttft,
        # Retain the v1 TTFT-only aliases for existing artifact consumers.
        "absolute_error_ms": abs(observed_ttft - predicted_ttft),
        "relative_error": ttft_relative_error,
        "prediction_interval_covered": ttft_interval_covered,
        "p95_ttft_absolute_error_ms": abs(observed_ttft - predicted_ttft),
        "p95_ttft_relative_error": ttft_relative_error,
        "p95_ttft_prediction_interval_covered": ttft_interval_covered,
        "predicted_p99_tpot_ms": predicted_tpot,
        "predicted_p99_tpot_lower_ms": predicted_tpot_interval.lower,
        "predicted_p99_tpot_upper_ms": predicted_tpot_interval.upper,
        "observed_p99_tpot_ms": observed_tpot,
        "p99_tpot_absolute_error_ms": abs(observed_tpot - predicted_tpot),
        "p99_tpot_relative_error": tpot_relative_error,
        "p99_tpot_prediction_interval_covered": tpot_interval_covered,
        "maximum_relative_error": max_relative_error,
        "prediction_interval_required": require_prediction_interval,
        "hard_slo_evaluated": slo is not None,
        "slo_p95_ttft_ms": slo_p95_ttft_ms,
        "slo_p99_tpot_ms": slo_p99_tpot_ms,
        "slo_attained": slo_attained,
        "valid": not failure_reasons,
        "failure_reasons": failure_reasons,
        "simulation_result_sha256": sha256_file(output / "result.json"),
    }
    write_json(output / "validation.json", validation)
    json_result({"output": str(output / "validation.json"), **validation})
    if failure_reasons:
        raise typer.Exit(code=1)
