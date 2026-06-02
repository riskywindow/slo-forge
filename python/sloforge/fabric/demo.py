"""CPU-only flagship demonstration for SLOForge Fabric."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from sloforge.autopsy import BottleneckKind, DiagnosisRecord, compare_runs, diagnose
from sloforge.autopsy.capture import capture_simulation_run
from sloforge.autopsy.counterfactual import (
    CounterfactualReplay,
    CounterfactualScenario,
    attach_counterfactuals,
    bind_subprocess_runner,
    replay_counterfactuals,
)
from sloforge.autopsy.counterfactual import (
    RemoveFault as AutopsyRemoveFault,
)
from sloforge.autopsy.counterfactual import (
    ReplaceResource as AutopsyReplaceResource,
)
from sloforge.autopsy.counterfactual import (
    ScaleRank as AutopsyScaleRank,
)
from sloforge.autopsy.counterfactual import (
    ScaleResourceCurve as AutopsyScaleResourceCurve,
)
from sloforge.fabric.compiler import (
    CompilerAssumptions,
    CompilerConstraints,
    CompilerObjective,
    CompilerRequest,
    OptimizationStrategy,
    compile_physical_plan,
)
from sloforge.fabric.faults import bind_physical_faults, load_physical_fault_scenario
from sloforge.fabric.ir import (
    DocumentReference,
    FabricProfile,
    ModelGraph,
    PhysicalExecutionPlan,
    RecoveryPlan,
    TopologyGraph,
    canonical_hash,
    save_fabric_profile,
    save_model_graph,
    save_physical_execution_plan,
    save_topology_graph,
)
from sloforge.fabric.model_graph import synthetic_moe_model_graph
from sloforge.fabric.profiling import benchmark_synthetic_fabric, to_canonical_profile
from sloforge.fabric.simulation import (
    CounterfactualModifier,
    FabricSimulationOutput,
    FabricSimulationRequest,
    RemoveFault,
    ReplaceResource,
    ResourceKind,
    ScaleRank,
    ScaleResourceCurve,
    SimulationRequestShape,
    SimulationWorkload,
    build_simulation_request,
    request_latencies,
    run_simulation,
)
from sloforge.fabric.topology import build_canonical_fixture
from sloforge.ir import ArtifactDigest
from sloforge.recovery import (
    DeterministicRecoveryExecutor,
    MetricObservation,
    RecoveryMachineConfig,
    RecoveryObservation,
    RecoveryPolicy,
    RecoverySnapshot,
    RecoveryState,
    plan_recovery,
)
from sloforge.util import (
    environment_manifest,
    git_commit,
    percentile,
    sha256_file,
    write_json,
)


class DemoModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class WorkloadRecord(DemoModel):
    request_id: str
    arrival_us: Annotated[float, Field(ge=0.0)]
    prompt_tokens: Annotated[int, Field(gt=0)]
    output_tokens: Annotated[int, Field(gt=0)]
    priority: Literal["high", "normal", "low"]
    request_class: Literal["interactive", "long_context", "batch"]


class SloMetrics(DemoModel):
    p95_ttft_ms: Annotated[float, Field(ge=0.0)]
    p99_tpot_ms: Annotated[float, Field(ge=0.0)]
    p95_end_to_end_ms: Annotated[float, Field(ge=0.0)]
    makespan_ms: Annotated[float, Field(ge=0.0)]


class TimelineEvent(DemoModel):
    sequence: Annotated[int, Field(ge=0)]
    at_ms: Annotated[float, Field(ge=0.0)]
    event: str
    detail: str
    evidence_uri: str


class DemoArtifact(DemoModel):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class FabricDemoManifest(DemoModel):
    schema_version: Literal["sloforge.fabric.demo/v1"] = "sloforge.fabric.demo/v1"
    seed: Annotated[int, Field(ge=0)]
    synthetic_hardware: bool
    physical_plan_id: str
    baseline_plan_id: str
    topology_fingerprint: str
    healthy: SloMetrics
    degraded: SloMetrics
    restored: SloMetrics
    p95_ttft_slo_ms: Annotated[float, Field(gt=0.0)]
    healthy_slo_attained: bool
    degraded_slo_attained: bool
    restored_slo_attained: bool
    diagnosis: str
    diagnosis_confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    ground_truth_faults: tuple[str, ...]
    counterfactuals_evaluated: Annotated[int, Field(gt=0)]
    selected_counterfactual: str
    recovery_final_state: str
    artifacts: tuple[DemoArtifact, ...]


def _digest(value: str) -> ArtifactDigest:
    return ArtifactDigest(value=hashlib.sha256(value.encode()).hexdigest())


def _reset_directory(path: Path, reset: bool) -> None:
    if path.exists() and reset:
        if path.resolve() in {Path("/").resolve(), Path.home().resolve()}:
            raise ValueError("refusing to reset a broad directory")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _workload(seed: int) -> tuple[tuple[WorkloadRecord, ...], SimulationWorkload]:
    # Deterministic burst gaps: 12 requests arrive in three clusters while
    # retaining interactive, long-context, and batch classes.
    records: list[WorkloadRecord] = []
    arrival = 0.0
    for index in range(12):
        cluster_position = index % 4
        if index and cluster_position == 0:
            arrival += 18_000.0
        else:
            arrival += 80.0 + float((seed + index * 17) % 60)
        request_class: Literal["interactive", "long_context", "batch"]
        priority: Literal["high", "normal", "low"]
        if index % 4 == 0:
            request_class, priority, prompt, output = "long_context", "normal", 8192, 96
        elif index % 3 == 0:
            request_class, priority, prompt, output = "batch", "low", 2048, 128
        else:
            request_class, priority, prompt, output = "interactive", "high", 128, 24
        records.append(
            WorkloadRecord(
                request_id=f"request-{index:06d}",
                arrival_us=arrival,
                prompt_tokens=prompt,
                output_tokens=output,
                priority=priority,
                request_class=request_class,
            )
        )
    shapes = tuple(
        SimulationRequestShape(
            arrival_us=record.arrival_us,
            prompt_tokens=record.prompt_tokens,
            output_tokens=record.output_tokens,
            priority=record.priority,
            request_class=record.request_class,
        )
        for record in records
    )
    return tuple(records), SimulationWorkload(
        request_count=len(records),
        arrival_interval_us=0.0,
        prompt_tokens=8192,
        output_tokens=128,
        requests=shapes,
    )


def _write_workload(path: Path, records: tuple[WorkloadRecord, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(record.model_dump(mode="json"), sort_keys=True) + "\n" for record in records
    )
    path.write_text(payload, encoding="utf-8")


def _compiler_request(
    *,
    logical_reference: DocumentReference,
    model: ModelGraph,
    topology: TopologyGraph,
    profile: FabricProfile,
    strategy: OptimizationStrategy,
    generated_at: datetime,
    seed: int,
    repository_root: Path,
    environment_digest: ArtifactDigest,
) -> CompilerRequest:
    return CompilerRequest(
        logical_deployment_plan=logical_reference,
        model=model,
        topology=topology,
        fabric_profile=profile,
        constraints=CompilerConstraints(
            prompt_tokens_p95=8192,
            output_tokens_p95=128,
            maximum_concurrent_requests=64,
            p95_ttft_ms=5_000.0,
            p99_tpot_ms=45.0,
            minimum_goodput_tokens_per_second=500.0,
            minimum_availability=0.95,
            maximum_ranks=16,
            require_disaggregation=True,
        ),
        assumptions=CompilerAssumptions(
            prefill_tokens_per_second_per_gpu=20_000.0,
            decode_tokens_per_second_per_gpu=100.0,
            gpu_hourly_price_usd=4.0,
            base_availability=0.999,
            cold_start_ms=10_000.0,
            measurement_relative_uncertainty=0.10,
        ),
        objective=CompilerObjective.ROBUST_BALANCED,
        strategy=strategy,
        generated_at=generated_at,
        seed=seed,
        git_commit=git_commit(repository_root),
        environment_digest=environment_digest,
    )


def _metrics(output: FabricSimulationOutput, workload: SimulationWorkload) -> SloMetrics:
    latencies = request_latencies(output)
    ttft_ms = [item.ttft_us / 1_000.0 for item in latencies]
    e2e_ms = [item.end_to_end_us / 1_000.0 for item in latencies]
    shapes = {f"request-{index:06d}": shape for index, shape in enumerate(workload.requests)}
    token_steps: list[float] = []
    for operation in output.operations:
        if not operation.operation_id.endswith(":decode"):
            continue
        request_id = operation.operation_id.partition(":")[0]
        token_steps.append(operation.duration_us / shapes[request_id].output_tokens / 1_000.0)
    return SloMetrics(
        p95_ttft_ms=percentile(ttft_ms, 0.95),
        p99_tpot_ms=percentile(token_steps, 0.99),
        p95_end_to_end_ms=percentile(e2e_ms, 0.95),
        makespan_ms=output.metrics.makespan_us / 1_000.0,
    )


def _simulation_modifier(value: object) -> CounterfactualModifier:
    if isinstance(value, AutopsyRemoveFault):
        return RemoveFault(fault_id=value.fault_id)
    if isinstance(value, AutopsyScaleResourceCurve):
        return ScaleResourceCurve(
            resource_id=value.resource_id,
            latency_multiplier=value.latency_multiplier,
            bandwidth_multiplier=value.bandwidth_multiplier,
        )
    if isinstance(value, AutopsyScaleRank):
        return ScaleRank(
            rank_id=value.rank_id,
            duration_multiplier=value.duration_multiplier,
        )
    if isinstance(value, AutopsyReplaceResource):
        return ReplaceResource(
            from_resource_id=value.from_resource_id,
            to_resource_id=value.to_resource_id,
        )
    raise TypeError(f"unsupported counterfactual modifier {type(value)!r}")


def _scenarios(
    diagnosis: DiagnosisRecord,
    *,
    network_fault_id: str,
    rank_fault_id: str,
    degraded_request: FabricSimulationRequest,
) -> tuple[CounterfactualScenario, ...]:
    by_kind = {hypothesis.kind: hypothesis for hypothesis in diagnosis.hypotheses}
    network = by_kind[BottleneckKind.NETWORK_BANDWIDTH_DEGRADATION]
    rank = by_kind[BottleneckKind.RANK_STRAGGLER]
    rail_resources = tuple(
        resource.id
        for resource in degraded_request.resources
        if resource.kind is ResourceKind.NETWORK_RAIL
    )
    primary_rail = next(
        resource.id
        for resource in degraded_request.resources
        if resource.id == getattr(degraded_request.faults[0].effect, "resource_id", None)
    )
    alternate_rail = next(identifier for identifier in rail_resources if identifier != primary_rail)
    return (
        CounterfactualScenario(
            scenario_id="remove-network-fault",
            hypothesis_id=network.hypothesis_id,
            hypothesis_kind=network.kind,
            rationale="remove the measured network-path degradation",
            modifications=(AutopsyRemoveFault(fault_id=network_fault_id),),
        ),
        CounterfactualScenario(
            scenario_id="remove-rank-fault",
            hypothesis_id=rank.hypothesis_id,
            hypothesis_kind=rank.kind,
            rationale="remove the rank-specific service-rate degradation",
            modifications=(AutopsyRemoveFault(fault_id=rank_fault_id),),
        ),
        CounterfactualScenario(
            scenario_id="remove-both-faults",
            hypothesis_id=network.hypothesis_id,
            hypothesis_kind=network.kind,
            rationale="restore both independently observed physical resources",
            modifications=(
                AutopsyRemoveFault(fault_id=network_fault_id),
                AutopsyRemoveFault(fault_id=rank_fault_id),
            ),
        ),
        CounterfactualScenario(
            scenario_id="restore-network-curve",
            hypothesis_id=network.hypothesis_id,
            hypothesis_kind=network.kind,
            rationale="restore calibrated bandwidth while retaining the fault interval",
            modifications=(
                AutopsyScaleResourceCurve(
                    resource_id=primary_rail,
                    latency_multiplier=1.0,
                    bandwidth_multiplier=20.0,
                ),
            ),
        ),
        CounterfactualScenario(
            scenario_id="move-network-resource",
            hypothesis_id=network.hypothesis_id,
            hypothesis_kind=network.kind,
            rationale="route affected operations onto the alternate rail",
            modifications=(
                AutopsyReplaceResource(
                    from_resource_id=primary_rail,
                    to_resource_id=alternate_rail,
                ),
            ),
        ),
        CounterfactualScenario(
            scenario_id="restore-rank-rate",
            hypothesis_id=rank.hypothesis_id,
            hypothesis_kind=rank.kind,
            rationale="counteract the measured rank service-time multiplier",
            modifications=(AutopsyScaleRank(rank_id="rank-6", duration_multiplier=0.25),),
        ),
        CounterfactualScenario(
            scenario_id="wrong-rank-control",
            hypothesis_id=rank.hypothesis_id,
            hypothesis_kind=rank.kind,
            rationale="negative control: slow an unaffected rank",
            modifications=(AutopsyScaleRank(rank_id="rank-1", duration_multiplier=1.25),),
        ),
    )


def _run_recovery(
    replay: CounterfactualReplay,
    diagnosis: DiagnosisRecord,
    plan: PhysicalExecutionPlan,
    restored: SloMetrics,
) -> tuple[RecoveryPlan, tuple[TimelineEvent, ...], RecoverySnapshot]:
    ttft_target = max(restored.p95_ttft_ms * 1.10, 1.0)
    tpot_target = max(restored.p99_tpot_ms * 1.10, 1.0)
    proposal = plan_recovery(
        diagnosis,
        plan,
        policy=RecoveryPolicy(
            minimum_diagnosis_confidence=0.50,
            minimum_shadow_samples=3,
            minimum_canary_samples=4,
            target_p95_ttft_ms=ttft_target,
            target_p99_tpot_ms=tpot_target,
        ),
    )
    executor = DeterministicRecoveryExecutor(
        proposal,
        now_ms=0,
        config=RecoveryMachineConfig(promotion_cooldown_ms=0),
    )
    metrics = (
        MetricObservation(name="p99_tpot_ms", value=restored.p99_tpot_ms, window_seconds=30.0),
        MetricObservation(name="p95_ttft_ms", value=restored.p95_ttft_ms, window_seconds=30.0),
        MetricObservation(name="error_rate", value=0.0, window_seconds=30.0),
    )
    observations = (
        RecoveryObservation(
            observed_at_ms=1_000,
            idempotency_key="simulation-validated",
            simulation_validated=replay.selected_scenario_id is not None,
        ),
        RecoveryObservation(observed_at_ms=2_000, idempotency_key="replacement-build"),
        RecoveryObservation(
            observed_at_ms=3_000,
            idempotency_key="replacement-ready",
            replacement_ready=True,
        ),
        RecoveryObservation(
            observed_at_ms=4_000,
            idempotency_key="shadow-complete",
            shadow_samples=3,
            metrics=metrics,
        ),
        RecoveryObservation(
            observed_at_ms=5_000,
            idempotency_key="canary-complete",
            canary_samples=4,
            metrics=metrics,
        ),
        RecoveryObservation(
            observed_at_ms=6_000,
            idempotency_key="traffic-migrated",
            traffic_migration_complete=True,
        ),
        RecoveryObservation(
            observed_at_ms=7_000,
            idempotency_key="draining-streams",
            active_started_streams=2,
        ),
        RecoveryObservation(
            observed_at_ms=8_000,
            idempotency_key="drain-complete",
            active_started_streams=0,
        ),
    )
    timeline: list[TimelineEvent] = []
    previous_state = executor.snapshot.state
    for observation in observations:
        snapshot = executor.tick(observation)
        if snapshot.state is not previous_state:
            timeline.append(
                TimelineEvent(
                    sequence=len(timeline),
                    at_ms=float(observation.observed_at_ms),
                    event=snapshot.state.value,
                    detail=snapshot.audit[-1].reason,
                    evidence_uri="recovery/execution.json",
                )
            )
        previous_state = snapshot.state
    if executor.snapshot.state is not RecoveryState.COMPLETED:
        raise RuntimeError(f"recovery demo did not complete: {executor.snapshot.state.value}")
    return (proposal, tuple(timeline), executor.snapshot)


def _write_trace(path: Path, output: FabricSimulationOutput) -> None:
    write_json(
        path,
        {
            "traceEvents": [event.model_dump(mode="json") for event in output.trace_events],
            "displayTimeUnit": "us",
            "provenance": output.provenance.model_dump(mode="json"),
        },
    )


def _write_prometheus(path: Path, metrics: SloMetrics, *, run: str) -> None:
    lines = (
        "# TYPE sloforge_fabric_p95_ttft_ms gauge",
        f'sloforge_fabric_p95_ttft_ms{{run="{run}"}} {metrics.p95_ttft_ms:.9f}',
        "# TYPE sloforge_fabric_p99_tpot_ms gauge",
        f'sloforge_fabric_p99_tpot_ms{{run="{run}"}} {metrics.p99_tpot_ms:.9f}',
        "# TYPE sloforge_fabric_e2e_p95_ms gauge",
        f'sloforge_fabric_e2e_p95_ms{{run="{run}"}} {metrics.p95_end_to_end_ms:.9f}',
        "",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_plot(
    path: Path, healthy: SloMetrics, degraded: SloMetrics, restored: SloMetrics
) -> None:
    values = (healthy.p95_ttft_ms, degraded.p95_ttft_ms, restored.p95_ttft_ms)
    maximum = max(values)
    widths = tuple(600.0 * value / maximum for value in values)
    labels = ("healthy", "degraded", "restored")
    colors = ("#3b82f6", "#ef4444", "#22c55e")
    bars = "".join(
        f'<text x="10" y="{50 + index * 55}">{label}</text>'
        f'<rect x="100" y="{30 + index * 55}" width="{width:.3f}" height="28" '
        f'fill="{color}"/><text x="{110 + width:.3f}" y="{50 + index * 55}">'
        f"{value:.2f} ms</text>"
        for index, (label, color, width, value) in enumerate(
            zip(labels, colors, widths, values, strict=True)
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="850" height="220" '
        'role="img" aria-label="p95 TTFT comparison">'
        '<rect width="100%" height="100%" fill="white"/>'
        '<text x="10" y="20" font-weight="bold">Artifact-derived p95 TTFT</text>'
        f"{bars}</svg>\n",
        encoding="utf-8",
    )


def _artifact(path: Path, root: Path) -> DemoArtifact:
    return DemoArtifact(path=str(path.relative_to(root)), sha256=sha256_file(path))


def _render_report(manifest: FabricDemoManifest, timeline: tuple[TimelineEvent, ...]) -> str:
    lines = [
        "# SLOForge Fabric CPU demonstration",
        "",
        "All values below are loaded from the manifest and raw simulator artifacts.",
        "Synthetic hardware curves are labeled synthetic and are not GPU measurements.",
        "",
        "## Outcome",
        "",
        f"- Physical plan: `{manifest.physical_plan_id}`",
        f"- Diagnosis: `{manifest.diagnosis}` ({manifest.diagnosis_confidence:.3f} confidence)",
        f"- Counterfactuals evaluated: {manifest.counterfactuals_evaluated}",
        f"- Selected repair: `{manifest.selected_counterfactual}`",
        f"- Recovery state: `{manifest.recovery_final_state}`",
        "",
        "| run | p95 TTFT ms | p99 TPOT ms | p95 E2E ms | SLO attained |",
        "|---|---:|---:|---:|:---:|",
    ]
    for name, metrics, attained in (
        ("healthy", manifest.healthy, manifest.healthy_slo_attained),
        ("degraded", manifest.degraded, manifest.degraded_slo_attained),
        ("restored", manifest.restored, manifest.restored_slo_attained),
    ):
        lines.append(
            f"| {name} | {metrics.p95_ttft_ms:.3f} | {metrics.p99_tpot_ms:.3f} | "
            f"{metrics.p95_end_to_end_ms:.3f} | {'yes' if attained else 'no'} |"
        )
    lines.extend(("", "## Artifact-derived timeline", ""))
    for event in timeline:
        lines.append(
            f"- +{event.at_ms / 1_000.0:.3f}s **{event.event}** — {event.detail} "
            f"(`{event.evidence_uri}`)"
        )
    lines.extend(("", "## Artifact integrity", ""))
    lines.extend(f"- `{item.path}` — `{item.sha256}`" for item in manifest.artifacts)
    return "\n".join(lines) + "\n"


def _validate_report(manifest_path: Path, report_path: Path, artifact_root: Path) -> None:
    manifest = FabricDemoManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    report = report_path.read_text(encoding="utf-8")
    if f"{manifest.degraded.p95_ttft_ms:.3f}" not in report:
        raise RuntimeError("report does not contain the manifest-derived degraded TTFT")
    for artifact in manifest.artifacts:
        path = artifact_root / artifact.path
        if not path.is_file() or sha256_file(path) != artifact.sha256:
            raise RuntimeError(f"report artifact hash mismatch: {artifact.path}")


def run_fabric_demo(
    *,
    repository_root: Path,
    artifact_dir: Path,
    report_dir: Path,
    seed: int = 41,
    reset: bool = False,
) -> FabricDemoManifest:
    _reset_directory(artifact_dir, reset)
    _reset_directory(report_dir, reset)
    generated_at = datetime.now(UTC)
    environment_path = artifact_dir / "environment.json"
    write_json(environment_path, environment_manifest())
    environment_digest = ArtifactDigest(value=sha256_file(environment_path))
    topology = build_canonical_fixture("two_node_infiniband")
    topology_path = artifact_dir / "topology.json"
    save_topology_graph(topology_path, topology)
    raw_profile = benchmark_synthetic_fabric(
        topology,
        seed=seed,
        suite="quick",
        warmup_count=3,
        sample_count=7,
        output_dir=artifact_dir / "fabric-profile-raw",
    )
    profile = to_canonical_profile(raw_profile, topology=topology)
    profile_path = artifact_dir / "fabric-profile.json"
    save_fabric_profile(profile_path, profile)
    model = synthetic_moe_model_graph()
    model_path = artifact_dir / "model-graph.json"
    save_model_graph(model_path, model)
    logical_source = repository_root / "tests" / "fixtures" / "ir" / "deployment-plan-v1.json"
    logical_path = artifact_dir / "logical-deployment-plan.json"
    logical_path.write_bytes(logical_source.read_bytes())
    logical_reference = DocumentReference(
        kind="DeploymentPlan",
        api_version="sloforge.io/v1",
        uri=str(logical_path),
        digest=ArtifactDigest(value=sha256_file(logical_path)),
        uid="fabric-demo-logical-plan",
        generation=1,
    )
    aware = compile_physical_plan(
        _compiler_request(
            logical_reference=logical_reference,
            model=model,
            topology=topology,
            profile=profile,
            strategy=OptimizationStrategy.HIERARCHICAL,
            generated_at=generated_at,
            seed=seed,
            repository_root=repository_root,
            environment_digest=environment_digest,
        )
    )
    unaware = compile_physical_plan(
        _compiler_request(
            logical_reference=logical_reference,
            model=model,
            topology=topology,
            profile=profile,
            strategy=OptimizationStrategy.TOPOLOGY_UNAWARE,
            generated_at=generated_at,
            seed=seed,
            repository_root=repository_root,
            environment_digest=environment_digest,
        )
    )
    plan_path = artifact_dir / "physical-plan.json"
    baseline_path = artifact_dir / "physical-plan-topology-unaware.json"
    save_physical_execution_plan(plan_path, aware.selected)
    save_physical_execution_plan(baseline_path, unaware.selected)
    write_json(artifact_dir / "optimizer.json", aware.model_dump(mode="json"))
    records, workload = _workload(seed)
    workload_path = artifact_dir / "mixed-bursty.jsonl"
    _write_workload(workload_path, records)
    workload_fingerprint = sha256_file(workload_path)
    healthy_request = build_simulation_request(
        aware.selected, topology, profile, workload, seed=seed
    )
    scenario = load_physical_fault_scenario(
        repository_root / "scenarios" / "fabric" / "dual-fault-demo.yaml"
    )
    faults = bind_physical_faults(scenario, healthy_request)
    degraded_request = healthy_request.model_copy(update={"faults": faults})
    healthy_output = run_simulation(
        healthy_request, repository_root=repository_root, timeout_seconds=60
    )
    degraded_output = run_simulation(
        degraded_request, repository_root=repository_root, timeout_seconds=60
    )
    healthy_path = artifact_dir / "simulations" / "healthy.json"
    degraded_path = artifact_dir / "simulations" / "degraded.json"
    write_json(healthy_path, healthy_output.model_dump(mode="json"))
    write_json(degraded_path, degraded_output.model_dump(mode="json"))
    healthy_run = capture_simulation_run(
        run_id="healthy-run",
        request=healthy_request,
        output=healthy_output,
        plan=aware.selected,
        topology_fingerprint=canonical_hash(topology),
        workload_fingerprint=workload_fingerprint,
        artifact_path=healthy_path,
    )
    degraded_run = capture_simulation_run(
        run_id="degraded-run",
        request=degraded_request,
        output=degraded_output,
        plan=aware.selected,
        topology_fingerprint=canonical_hash(topology),
        workload_fingerprint=workload_fingerprint,
        artifact_path=degraded_path,
    )
    healthy_autopsy = artifact_dir / "autopsy" / "healthy-run.json"
    degraded_autopsy = artifact_dir / "autopsy" / "degraded-run.json"
    write_json(healthy_autopsy, healthy_run.model_dump(mode="json"))
    write_json(degraded_autopsy, degraded_run.model_dump(mode="json"))
    comparison = compare_runs(healthy_run, degraded_run)
    diagnosis = diagnose(degraded_run, comparison=comparison, baseline=healthy_run)
    scenarios = _scenarios(
        diagnosis,
        network_fault_id=faults[0].id,
        rank_fault_id=faults[1].id,
        degraded_request=degraded_request,
    )
    replay = replay_counterfactuals(
        diagnosis,
        simulation_request=degraded_request.model_dump(mode="json"),
        scenarios=scenarios,
        healthy_reference_us=healthy_output.metrics.makespan_us,
        runner=bind_subprocess_runner(repository_root=repository_root, timeout_s=60.0),
    )
    diagnosis = attach_counterfactuals(diagnosis, replay)
    if replay.selected_scenario_id is None:
        raise RuntimeError("no counterfactual repair had a positive prediction interval")
    selected = next(
        scenario for scenario in scenarios if scenario.scenario_id == replay.selected_scenario_id
    )
    restored_request = degraded_request.model_copy(
        update={
            "counterfactuals": tuple(
                _simulation_modifier(modifier) for modifier in selected.modifications
            )
        }
    )
    restored_output = run_simulation(
        restored_request, repository_root=repository_root, timeout_seconds=60
    )
    restored_path = artifact_dir / "simulations" / "restored.json"
    write_json(restored_path, restored_output.model_dump(mode="json"))
    healthy_metrics = _metrics(healthy_output, workload)
    degraded_metrics = _metrics(degraded_output, workload)
    restored_metrics = _metrics(restored_output, workload)
    proposal, recovery_timeline, recovery_snapshot = _run_recovery(
        replay, diagnosis, aware.selected, restored_metrics
    )
    recovery_plan_path = artifact_dir / "recovery" / "proposal.json"
    recovery_execution_path = artifact_dir / "recovery" / "execution.json"
    write_json(recovery_plan_path, proposal.model_dump(mode="json"))
    write_json(recovery_execution_path, recovery_snapshot.model_dump(mode="json"))
    comparison_path = artifact_dir / "autopsy" / "comparison.json"
    diagnosis_path = artifact_dir / "autopsy" / "diagnosis.json"
    replay_path = artifact_dir / "autopsy" / "counterfactuals.json"
    write_json(comparison_path, comparison.model_dump(mode="json"))
    write_json(diagnosis_path, diagnosis.model_dump(mode="json"))
    write_json(replay_path, replay.model_dump(mode="json"))
    _write_trace(artifact_dir / "traces" / "degraded.perfetto.json", degraded_output)
    write_json(
        artifact_dir / "traces" / "otel.json",
        {
            "resourceSpans": [
                {
                    "trace_id": degraded_output.provenance.input_sha256[:32],
                    "spans": [
                        event.model_dump(mode="json") for event in degraded_output.trace_events
                    ],
                }
            ]
        },
    )
    _write_prometheus(artifact_dir / "metrics" / "healthy.prom", healthy_metrics, run="healthy")
    _write_prometheus(artifact_dir / "metrics" / "degraded.prom", degraded_metrics, run="degraded")
    _write_prometheus(artifact_dir / "metrics" / "restored.prom", restored_metrics, run="restored")
    _write_plot(report_dir / "p95-ttft.svg", healthy_metrics, degraded_metrics, restored_metrics)
    slo_target = max(healthy_metrics.p95_ttft_ms, restored_metrics.p95_ttft_ms) * 1.10
    if degraded_metrics.p95_ttft_ms <= slo_target:
        raise RuntimeError("faults did not produce the required p95 TTFT SLO regression")
    if restored_metrics.p95_ttft_ms > slo_target:
        raise RuntimeError("selected recovery did not restore the p95 TTFT SLO")
    timeline = (
        TimelineEvent(
            sequence=0,
            at_ms=(diagnosis.first_divergence_ns or 0) / 1_000_000.0,
            event="SLO_REGRESSION",
            detail=(f"p95 TTFT {degraded_metrics.p95_ttft_ms:.3f} ms exceeded {slo_target:.3f} ms"),
            evidence_uri="autopsy/comparison.json",
        ),
        TimelineEvent(
            sequence=1,
            at_ms=(diagnosis.first_divergence_ns or 0) / 1_000_000.0 + 1.0,
            event="CAUSAL_DIAGNOSIS",
            detail=f"{diagnosis.top_hypothesis.value} confidence={diagnosis.confidence:.3f}",
            evidence_uri="autopsy/diagnosis.json",
        ),
        TimelineEvent(
            sequence=2,
            at_ms=(diagnosis.first_divergence_ns or 0) / 1_000_000.0 + 2.0,
            event="COUNTERFACTUAL_SELECTION",
            detail=(
                f"evaluated {len(replay.evaluations)} repairs; selected "
                f"{replay.selected_scenario_id}"
            ),
            evidence_uri="autopsy/counterfactuals.json",
        ),
        *tuple(
            event.model_copy(update={"sequence": index + 3})
            for index, event in enumerate(recovery_timeline)
        ),
        TimelineEvent(
            sequence=len(recovery_timeline) + 3,
            at_ms=9_000.0,
            event="SLO_RESTORED",
            detail=f"p95 TTFT restored to {restored_metrics.p95_ttft_ms:.3f} ms",
            evidence_uri="simulations/restored.json",
        ),
    )
    timeline_path = artifact_dir / "timeline.json"
    write_json(timeline_path, [event.model_dump(mode="json") for event in timeline])
    artifact_paths = (
        topology_path,
        profile_path,
        model_path,
        plan_path,
        baseline_path,
        workload_path,
        healthy_path,
        degraded_path,
        restored_path,
        comparison_path,
        diagnosis_path,
        replay_path,
        recovery_plan_path,
        recovery_execution_path,
        artifact_dir / "traces" / "degraded.perfetto.json",
        artifact_dir / "traces" / "otel.json",
        artifact_dir / "metrics" / "degraded.prom",
        timeline_path,
    )
    manifest = FabricDemoManifest(
        seed=seed,
        synthetic_hardware=True,
        physical_plan_id=aware.selected.plan_id,
        baseline_plan_id=unaware.selected.plan_id,
        topology_fingerprint=canonical_hash(topology),
        healthy=healthy_metrics,
        degraded=degraded_metrics,
        restored=restored_metrics,
        p95_ttft_slo_ms=slo_target,
        healthy_slo_attained=healthy_metrics.p95_ttft_ms <= slo_target,
        degraded_slo_attained=degraded_metrics.p95_ttft_ms <= slo_target,
        restored_slo_attained=restored_metrics.p95_ttft_ms <= slo_target,
        diagnosis=diagnosis.top_hypothesis.value,
        diagnosis_confidence=diagnosis.confidence,
        ground_truth_faults=tuple(fault.ground_truth_label for fault in faults),
        counterfactuals_evaluated=len(replay.evaluations),
        selected_counterfactual=replay.selected_scenario_id,
        recovery_final_state=recovery_snapshot.state.value,
        artifacts=tuple(_artifact(path, artifact_dir) for path in artifact_paths),
    )
    manifest_path = artifact_dir / "manifest.json"
    write_json(manifest_path, manifest.model_dump(mode="json"))
    report_path = report_dir / "fabric-demo.md"
    report_text = _render_report(manifest, timeline)
    report_path.write_text(report_text, encoding="utf-8")
    (report_dir / "fabric-demo.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>SLOForge Fabric demo</title>"
        "<style>body{font:16px system-ui;max-width:1000px;margin:2rem auto;white-space:pre-wrap}</style>"
        f"<body>{html.escape(report_text)}</body>\n",
        encoding="utf-8",
    )
    _validate_report(manifest_path, report_path, artifact_dir)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/fabric-demo"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/fabric-demo"))
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[3]
    manifest = run_fabric_demo(
        repository_root=repository_root,
        artifact_dir=args.artifact_dir,
        report_dir=args.report_dir,
        seed=args.seed,
        reset=args.reset,
    )
    print(manifest.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
