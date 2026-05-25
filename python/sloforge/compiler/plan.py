from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from sloforge.hardware.probe import ProbeResult
from sloforge.ir import (
    AdmissionPolicy,
    ArrivalKind,
    ArrivalProcess,
    ArtifactDigest,
    AutoscalingPolicy,
    BatchingPolicy,
    BudgetSpec,
    CalibrationMetric,
    CanaryPolicy,
    ChunkedPrefillSpec,
    ColdStartStrategy,
    CpuSpec,
    DeploymentPlan,
    DistributionKind,
    DistributionSpec,
    DocumentMetadata,
    DType,
    EngineSpec,
    EnvironmentManifest,
    EvidenceBundle,
    Extensions,
    HardwareSpec,
    MeasurementRef,
    MetricConstraint,
    MetricEstimate,
    ModelSpec,
    ObjectiveWeights,
    OptimizerDecision,
    Priority,
    Provenance,
    Quantization,
    RejectedCandidate,
    ReplicaTopology,
    RequestClass,
    RollbackPolicy,
    RouteTarget,
    RoutingPolicy,
    RoutingPolicyKind,
    SLOSpec,
    WeightedValue,
    WorkloadSpec,
    canonical_hash,
    save_deployment_plan,
    save_evidence_bundle,
)
from sloforge.models.service_curve import CalibratedModels
from sloforge.optimizer.core import Constraint, Metrics, OptimizationResult
from sloforge.profiler.core import ProfileBundle
from sloforge.trace import TraceRequest
from sloforge.util import git_commit, sha256_file

from .model_metadata import ModelProfileMetadata


class CompiledArtifacts(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    plan: DeploymentPlan
    evidence: EvidenceBundle
    plan_path: Path
    evidence_path: Path


def _digest(value: str) -> ArtifactDigest:
    return ArtifactDigest(value=value)


def _sha_text(value: str) -> ArtifactDigest:
    return _digest(hashlib.sha256(value.encode()).hexdigest())


def _rust_version() -> str | None:
    try:
        completed = subprocess.run(
            ["rustc", "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _package_versions() -> dict[str, str]:
    names = ("httpx", "jinja2", "jsonschema", "numpy", "pydantic", "pyyaml", "typer")
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _artifact_uri(path: Path, repository_root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repository_root.resolve()))
    except ValueError:
        return str(resolved)


def _weighted(values: list[int], maximum: int) -> tuple[WeightedValue, ...]:
    counts = Counter(values)
    total = sum(counts.values())
    return tuple(
        WeightedValue(value=value, weight=count / total)
        for value, count in sorted(counts.items())
        if value <= maximum
    )


def _dtype(value: str) -> DType:
    return DType(value)


def _metric_estimate(
    name: str,
    metrics: Metrics,
    uncertainty: Metrics,
    measurement_ids: tuple[str, ...],
    sample_count: int,
    empirical_coverage: float,
) -> MetricEstimate:
    point = float(getattr(metrics, name))
    radius = float(getattr(uncertainty, name))
    units = {
        "p95_ttft_ms": "ms",
        "p99_itl_ms": "ms",
        "p95_e2e_ms": "ms",
        "goodput_tokens_s": "tokens/s",
        "throughput_tokens_s": "tokens/s",
        "availability": "ratio",
        "cost_per_million_tokens": "USD/million_tokens",
        "cold_start_p95_ms": "ms",
    }
    upper = point + radius
    if name == "availability":
        upper = min(1.0, upper)
    return MetricEstimate(
        point=point,
        lower=max(0.0, point - radius),
        upper=upper,
        confidence=min(0.999, max(0.001, empirical_coverage)),
        unit=units[name],
        sample_count=sample_count,
        measurement_ids=measurement_ids,
    )


def _constraint_slo(constraints: list[Constraint]) -> SLOSpec:
    ttft: list[MetricConstraint] = []
    itl: list[MetricConstraint] = []
    e2e: list[MetricConstraint] = []
    minimum_goodput: float | None = None
    availability: float | None = None
    cost: float | None = None
    for constraint in constraints:
        if constraint.metric == "p95_ttft_ms" and constraint.operator == "<=":
            ttft.append(MetricConstraint(percentile=95.0, maximum_ms=constraint.value))
        elif constraint.metric == "p99_itl_ms" and constraint.operator == "<=":
            itl.append(MetricConstraint(percentile=99.0, maximum_ms=constraint.value))
        elif constraint.metric == "p95_e2e_ms" and constraint.operator == "<=":
            e2e.append(MetricConstraint(percentile=95.0, maximum_ms=constraint.value))
        elif constraint.metric == "goodput_tokens_s" and constraint.operator == ">=":
            minimum_goodput = constraint.value
        elif constraint.metric == "availability" and constraint.operator == ">=":
            availability = constraint.value
        elif constraint.metric == "cost_per_million_tokens" and constraint.operator == "<=":
            cost = constraint.value
    return SLOSpec(
        ttft=tuple(ttft),
        inter_token_latency=tuple(itl),
        end_to_end_latency=tuple(e2e),
        minimum_goodput_rps=minimum_goodput,
        minimum_availability=availability,
        maximum_cost_per_million_tokens_usd=cost,
        objective_weights=ObjectiveWeights(cost=1.0, latency=0.25, goodput=0.25),
    )


def compile_deployment(
    *,
    optimization: OptimizationResult,
    profile: ProfileBundle,
    models: CalibratedModels,
    hardware: ProbeResult,
    trace: list[TraceRequest],
    trace_path: Path,
    hardware_path: Path,
    profile_dir: Path,
    output_path: Path,
    evidence_dir: Path,
    model_metadata: ModelProfileMetadata,
    repository_root: Path,
) -> CompiledArtifacts:
    selected = optimization.selected
    config = selected.configuration
    backend_profile = next(
        item
        for item in profile.candidates
        if item.candidate.candidate_id == config.backend_candidate_id
    )
    backend = backend_profile.candidate
    if model_metadata.architecture.parameter_count != backend.model_parameter_count:
        raise ValueError(
            "model metadata parameter count disagrees with the profiled backend candidate"
        )
    if model_metadata.maximum_sequence_length != backend.max_sequence_length:
        raise ValueError(
            "model metadata maximum sequence length disagrees with the profiled backend candidate"
        )
    model_fit = next(
        item for item in models.candidates if item.candidate_id == backend.candidate_id
    )
    now = datetime.now(UTC)
    plan_id = optimization.optimization_id
    measurement_ids_by_metric = {
        "p95_ttft_ms": (f"{profile.profile_id}-prefill", f"{profile.profile_id}-load"),
        "p99_itl_ms": (f"{profile.profile_id}-decode", f"{profile.profile_id}-load"),
        "p95_e2e_ms": (f"{profile.profile_id}-load",),
        "goodput_tokens_s": (f"{profile.profile_id}-load",),
        "throughput_tokens_s": (f"{profile.profile_id}-load",),
        "availability": (f"{profile.profile_id}-load",),
        "cost_per_million_tokens": (f"{profile.profile_id}-load",),
        "cold_start_p95_ms": (f"{profile.profile_id}-startup",),
    }
    sample_count_by_stage: dict[str, int] = {
        stage: sum(
            1
            for item in profile.raw_measurements
            if item.candidate_id == backend.candidate_id and not item.warmup and item.stage == stage
        )
        for stage in ("startup", "prefill", "decode", "load")
    }
    selected_metrics = selected.measured or selected.predicted
    coverage_by_metric = {
        "p95_ttft_ms": model_fit.interval_coverage,
        "p99_itl_ms": model_fit.decode_interval_coverage,
        "p95_e2e_ms": min(model_fit.interval_coverage, model_fit.decode_interval_coverage),
        "goodput_tokens_s": min(model_fit.interval_coverage, model_fit.decode_interval_coverage),
        "throughput_tokens_s": min(model_fit.interval_coverage, model_fit.decode_interval_coverage),
        "availability": model_fit.interval_coverage,
        "cost_per_million_tokens": min(
            model_fit.interval_coverage, model_fit.decode_interval_coverage
        ),
        "cold_start_p95_ms": model_fit.interval_coverage,
    }
    predicted_metrics = {
        name: _metric_estimate(
            name,
            selected_metrics,
            selected.uncertainty,
            measurement_ids_by_metric[name],
            sum(
                sample_count_by_stage[measurement_id.rsplit("-", maxsplit=1)[-1]]
                for measurement_id in measurement_ids_by_metric[name]
            ),
            coverage_by_metric[name],
        )
        for name in Metrics.model_fields
    }
    physical_cores = max(1, hardware.hardware.logical_cpu_count // 2)
    bandwidth = next(
        (item.median for item in hardware.benchmarks if item.name == "host_memory_copy_bandwidth"),
        None,
    )
    trace_digest = _digest(sha256_file(trace_path))
    hardware_digest = _digest(hardware.fingerprint)
    arrival_duration = max((trace[-1].arrival_ms - trace[0].arrival_ms) / 1000.0, 0.001)
    class_counts = Counter(item.request_class for item in trace)
    priority_by_class = {
        "interactive": Priority.INTERACTIVE,
        "long-context": Priority.BATCH,
    }
    request_classes = tuple(
        RequestClass(
            name=name,
            weight=count / len(trace),
            priority=priority_by_class.get(name, Priority.INTERACTIVE),
            deadline_ms=min(
                (
                    item.deadline_ms
                    for item in trace
                    if item.request_class == name and item.deadline_ms is not None
                ),
                default=None,
            ),
        )
        for name, count in sorted(class_counts.items())
    )
    runtime: Literal["transformers", "vllm", "sglang", "tensorrt_llm", "mock"]
    runtime = "tensorrt_llm" if backend.runtime == "tensorrt-llm" else backend.runtime  # type: ignore[assignment]
    plan = DeploymentPlan(
        metadata=DocumentMetadata(
            name=f"sloforge-{backend.candidate_id}", uid=plan_id, created_at=now
        ),
        model=ModelSpec(
            model_id=model_metadata.model_id,
            revision=model_metadata.revision,
            checksum=_digest(model_metadata.checksum_sha256),
            tokenizer_id=model_metadata.model_id,
            tokenizer_revision=model_metadata.revision,
            architecture=model_metadata.architecture,
            allowed_precisions=(_dtype(backend.dtype),),
            minimum_precision=_dtype(backend.dtype),
            maximum_sequence_length=backend.max_sequence_length,
            license=model_metadata.license,
            extensions=Extensions(root={"sloforge.dev/mock-model": model_metadata.model_is_mock}),
        ),
        engine=EngineSpec(
            runtime=runtime,
            version=backend.runtime_version,
            dtype=_dtype(backend.dtype),
            quantization=Quantization.NONE,
            maximum_batched_tokens=config.max_batched_tokens,
            maximum_active_sequences=config.concurrency,
            chunked_prefill=ChunkedPrefillSpec(
                enabled=config.chunked_prefill, chunk_tokens=min(512, config.max_batched_tokens)
            ),
        ),
        hardware=HardwareSpec(
            fingerprint=hardware_digest,
            cpu=CpuSpec(
                architecture=hardware.hardware.architecture,
                model=hardware.hardware.cpu_model,
                physical_cores=physical_cores,
                logical_cores=hardware.hardware.logical_cpu_count,
                numa_nodes=1,
                measured_memory_bandwidth_gbps=bandwidth,
            ),
            system_memory_bytes=hardware.hardware.memory_bytes,
            hourly_price_usd=backend.hourly_price_usd,
            region="local",
            container_memory_limit_bytes=hardware.hardware.cgroup_memory_limit_bytes,
        ),
        workload=WorkloadSpec(
            arrival_process=ArrivalProcess(
                kind=ArrivalKind.TRACE,
                trace_uri=_artifact_uri(trace_path, repository_root),
            ),
            prompt_tokens=DistributionSpec(
                kind=DistributionKind.EMPIRICAL,
                empirical=_weighted(
                    [item.prompt_tokens for item in trace], backend.max_sequence_length
                ),
                minimum=min(item.prompt_tokens for item in trace),
                maximum=backend.max_sequence_length,
            ),
            output_tokens=DistributionSpec(
                kind=DistributionKind.EMPIRICAL,
                empirical=_weighted(
                    [item.output_tokens for item in trace], backend.max_sequence_length
                ),
                minimum=min(item.output_tokens for item in trace),
                maximum=backend.max_sequence_length,
            ),
            request_classes=request_classes,
            duration_seconds=arrival_duration,
            seed=profile.seed,
            trace_digest=trace_digest,
        ),
        slo=_constraint_slo(optimization.request.constraints),
        budget=BudgetSpec(
            profiling_budget_usd=profile.budget.max_cost_usd,
            profiling_duration_seconds=profile.budget.max_duration_s,
            maximum_real_trials=optimization.request.trial_budget,
        ),
        replica_topology=ReplicaTopology(
            minimum_replicas=1,
            maximum_replicas=optimization.request.max_replicas,
            initial_replicas=config.replicas,
            regions=("local",),
        ),
        routing=RoutingPolicy(
            kind=RoutingPolicyKind(config.routing_policy),
            targets=(RouteTarget(variant=backend.candidate_id, weight=1.0),),
            cold_start_penalty_ms=model_fit.startup_p95_ms,
        ),
        admission=AdmissionPolicy(
            queue_capacity=256,
            maximum_queue_time_ms=max(50.0, selected_metrics.p95_ttft_ms),
            shed_below_priority=Priority.BATCH,
        ),
        batching=BatchingPolicy(
            maximum_active_sequences=config.concurrency,
            maximum_batched_tokens=config.max_batched_tokens,
            maximum_batch_delay_ms=2.0,
        ),
        autoscaling=AutoscalingPolicy(
            mode="predictive",
            target_utilization=0.72,
            control_interval_seconds=5.0,
            scale_up_cooldown_seconds=10.0,
            scale_down_cooldown_seconds=120.0,
            minimum_samples=8,
            safety_margin=0.15,
            maximum_change_per_interval=1,
        ),
        cold_start=ColdStartStrategy(
            minimum_warm_replicas=config.warm_replicas,
            readiness_timeout_seconds=max(10.0, model_fit.startup_p95_ms / 1000.0 * 3),
            predicted_p95_startup_ms=model_fit.startup_p95_ms,
        ),
        canary=CanaryPolicy(minimum_requests=20, observation_seconds=20.0),
        rollback=RollbackPolicy(availability_floor=0.95),
        predicted_metrics=predicted_metrics,
        provenance=Provenance(
            profile_id=profile.profile_id,
            optimizer_run_id=optimization.optimization_id,
            workload_digest=trace_digest,
            hardware_fingerprint=hardware_digest,
            evidence_bundle_uri=_artifact_uri(evidence_dir, repository_root),
            compiler_version="0.1.0",
            git_commit=git_commit(repository_root),
        ),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_deployment_plan(output_path, plan)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence_models_path = evidence_dir / "models.json"
    evidence_models_path.write_text(models.model_dump_json(indent=2) + "\n", encoding="utf-8")
    measurements_path = profile_dir / "measurements.jsonl"
    profile_time = datetime.fromisoformat(profile.generated_at)
    measurement_stages: tuple[Literal["startup", "prefill", "decode", "load"], ...] = (
        "startup",
        "prefill",
        "decode",
        "load",
    )
    runtime_measurement_refs = tuple(
        MeasurementRef(
            measurement_id=f"{profile.profile_id}-{stage}",
            kind=stage,
            uri=_artifact_uri(measurements_path, repository_root),
            digest=_digest(sha256_file(measurements_path)),
            sample_count=sum(
                item.stage == stage and not item.warmup for item in profile.raw_measurements
            ),
            warmup_count=sum(
                item.stage == stage and item.warmup for item in profile.raw_measurements
            ),
            started_at=profile_time,
            completed_at=profile_time,
            hardware_fingerprint=hardware_digest,
        )
        for stage in measurement_stages
    )
    hardware_time = datetime.fromisoformat(hardware.captured_at)
    hardware_ref = MeasurementRef(
        measurement_id=f"{profile.profile_id}-hardware",
        kind="hardware",
        uri=_artifact_uri(hardware_path, repository_root),
        digest=_digest(sha256_file(hardware_path)),
        sample_count=sum(len(item.samples) for item in hardware.benchmarks),
        warmup_count=sum(item.warmup_count for item in hardware.benchmarks),
        started_at=hardware_time,
        completed_at=hardware_time,
        hardware_fingerprint=hardware_digest,
    )
    measurement_refs = (hardware_ref, *runtime_measurement_refs)
    history = tuple(
        OptimizerDecision(
            sequence=step.step,
            candidate_id=step.config_id,
            fidelity="measured" if step.fidelity == "measured" else "simulated",
            decision="select" if step.config_id == config.config_id else "evaluate",
            reason_code="uncertainty-aware-pareto-improvement",
            cost_usd=profile.budget.spent_cost_usd / max(1, len(optimization.optimizer_history)),
        )
        for step in optimization.optimizer_history
    )
    rejected = tuple(
        RejectedCandidate(
            candidate_id=item.configuration.config_id,
            stage="selection",
            reason_code="slo-constraint",
            explanation="; ".join(item.rejection_reasons),
            violated_constraints=tuple(item.constraint_margins),
        )
        for item in optimization.rejected_candidates
    )
    hostname = hardware.hardware.hostname
    env = EnvironmentManifest(
        os=hardware.hardware.os,
        kernel=platform.release(),
        architecture=hardware.hardware.architecture,
        hostname_hash=_sha_text(hostname),
        python_version=platform.python_version(),
        rust_version=_rust_version(),
        package_versions=_package_versions(),
    )
    evidence_path = evidence_dir / "evidence.json"
    artifact_files = {
        "deployment_plan": output_path,
        "profile": profile_dir / "profile.json",
        "measurements": measurements_path,
        "hardware": hardware_path,
        "workload": trace_path,
        "models": evidence_models_path,
    }
    evidence = EvidenceBundle(
        metadata=DocumentMetadata(
            name=f"evidence-{backend.candidate_id}", uid=f"evidence-{plan_id}", created_at=now
        ),
        plan_digest=_digest(canonical_hash(plan)),
        environment=env,
        model_assumptions=(
            "CPU demo service measurements come from explicit deterministic mock inference backends; they are not GPU measurements."
            if model_metadata.model_is_mock
            else "Model architecture and license metadata were parsed from the resolved immutable model snapshot; safetensors contents supplied the parameter count and weight checksum.",
            "Prediction intervals cover measured configurations only and do not imply cross-hardware generalization.",
        ),
        measurements=measurement_refs,
        calibration_metrics=(
            CalibrationMetric(
                model_name=f"{backend.candidate_id}-prefill",
                split="test",
                metric="mape",
                value=model_fit.held_out_mape,
                sample_count=model_fit.held_out_sample_count,
            ),
            CalibrationMetric(
                model_name=f"{backend.candidate_id}-decode",
                split="test",
                metric="mape",
                value=model_fit.decode_held_out_mape,
                sample_count=model_fit.held_out_sample_count,
            ),
            CalibrationMetric(
                model_name=f"{backend.candidate_id}-decode",
                split="test",
                metric="coverage",
                value=model_fit.decode_interval_coverage,
                sample_count=model_fit.held_out_sample_count,
            ),
            CalibrationMetric(
                model_name=f"{backend.candidate_id}-prefill",
                split="test",
                metric="coverage",
                value=model_fit.interval_coverage,
                sample_count=model_fit.held_out_sample_count,
            ),
        ),
        optimizer_history=history,
        rejected_candidates=rejected,
        benchmark_results=(),
        artifact_hashes={name: _digest(sha256_file(path)) for name, path in artifact_files.items()},
        git_commit=git_commit(repository_root),
        generated_at=now,
    )
    save_evidence_bundle(evidence_path, evidence)
    return CompiledArtifacts(
        plan=plan, evidence=evidence, plan_path=output_path, evidence_path=evidence_path
    )


def explain_plan(plan: DeploymentPlan) -> str:
    metrics = plan.predicted_metrics
    ttft = metrics.get("p95_ttft_ms")
    itl = metrics.get("p99_itl_ms")
    startup = metrics.get("cold_start_p95_ms")
    candidates = {
        "prefill/queue latency": ttft.point if ttft else 0.0,
        "decode latency": itl.point if itl else 0.0,
        "cold-start latency": startup.point if startup else 0.0,
    }
    bottleneck = max(candidates, key=candidates.get)  # type: ignore[arg-type]
    constraints: list[str] = []
    for constraint in plan.slo.ttft:
        constraints.append(f"p{constraint.percentile:g} TTFT <= {constraint.maximum_ms:g} ms")
    for constraint in plan.slo.inter_token_latency:
        constraints.append(f"p{constraint.percentile:g} ITL <= {constraint.maximum_ms:g} ms")
    constraint_text = ", ".join(constraints) if constraints else "the compiled objective"
    return (
        f"SLOForge selected {plan.engine.runtime} on {plan.hardware.region} with "
        f"{plan.replica_topology.initial_replicas} replicas, concurrency "
        f"{plan.batching.maximum_active_sequences}, and {plan.routing.kind}. "
        f"It is the lowest-objective uncertainty-adjusted Pareto candidate satisfying {constraint_text}. "
        f"The dominant predicted bottleneck is {bottleneck} ({candidates[bottleneck]:.1f} ms). "
        f"Evidence is recorded at {plan.provenance.evidence_bundle_uri}; predictions are bounded to the measured hardware fingerprint."
    )
