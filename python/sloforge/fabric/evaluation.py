"""Artifact-derived, CPU-safe evaluation for the SLOForge Fabric extension.

The checked-in evaluation is deliberately synthetic: it exercises the real
compiler, Rust simulator, Autopsy, and recovery planner but never labels the
fixture curves as hardware measurements.  Hardware-backed validation is a
separate status field and remains unexercised when no NVIDIA device is visible.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import random
import shutil
import statistics
import time
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sloforge.autopsy import BottleneckKind, DiagnosisRecord, compare_runs, diagnose
from sloforge.autopsy.capture import capture_simulation_run
from sloforge.fabric.compiler import (
    CompilerAssumptions,
    CompilerConstraints,
    CompilerObjective,
    CompilerRequest,
    OptimizationStrategy,
    PhysicalCompileResult,
    compile_physical_plan,
)
from sloforge.fabric.ir import (
    DocumentReference,
    FabricProfile,
    PhysicalExecutionPlan,
    TopologyGraph,
    canonical_hash,
)
from sloforge.fabric.model_graph import synthetic_moe_model_graph
from sloforge.fabric.profiling import benchmark_synthetic_fabric, to_canonical_profile
from sloforge.fabric.simulation import (
    FabricSimulationOutput,
    FabricSimulationRequest,
    RankSlowdownFault,
    RemoveFault,
    ResourceKind,
    ResourceRateFault,
    SimulationRequestShape,
    SimulationWorkload,
    TimedFault,
    build_simulation_request,
    request_latencies,
    run_simulation,
)
from sloforge.fabric.topology import build_canonical_fixture
from sloforge.fabric.topology.fixtures import FixtureSpec
from sloforge.ir import ArtifactDigest
from sloforge.recovery import RecoveryPolicy, plan_recovery
from sloforge.util import (
    environment_manifest,
    git_commit,
    percentile,
    sha256_file,
    write_json,
)


class EvaluationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class EvaluationMethod(StrEnum):
    RANDOM = "random_placement"
    SEQUENTIAL = "sequential_placement"
    TOPOLOGY_UNAWARE = "topology_unaware_optimizer"
    GREEDY = "topology_aware_greedy"
    HIERARCHICAL = "hierarchical_compiler"


ArtifactProvenance = Literal["synthetic", "environment", "compiler", "simulator", "autopsy"]
RecoveryMethod = Literal[
    "no_recovery",
    "restart_affected_worker",
    "replace_full_deployment",
    "threshold_recovery",
    "diagnosis_driven",
]


class EvaluationConfig(EvaluationModel):
    seeds: tuple[Annotated[int, Field(ge=0)], ...] = (13, 29, 47)
    topology_fixtures: tuple[str, ...] = (
        "two_node_infiniband",
        "two_node_degraded_network",
    )
    methods: tuple[EvaluationMethod, ...] = tuple(EvaluationMethod)
    request_count: Annotated[int, Field(ge=2, le=64)] = 8
    simulator_timeout_seconds: Annotated[float, Field(gt=0.0, le=300.0)] = 60.0
    p95_ttft_slo_ms: Annotated[float, Field(gt=0.0)] = 2_000.0
    p99_tpot_slo_ms: Annotated[float, Field(gt=0.0)] = 45.0
    bootstrap_repetitions: Annotated[int, Field(ge=100, le=20_000)] = 1_000

    @model_validator(mode="after")
    def unique_nonempty_dimensions(self) -> Self:
        for name, values in (
            ("seeds", self.seeds),
            ("topology_fixtures", self.topology_fixtures),
            ("methods", self.methods),
        ):
            if not values or len(values) != len(set(values)):
                raise ValueError(f"{name} must be non-empty and unique")
        return self


class ArtifactRef(EvaluationModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provenance: ArtifactProvenance


class IntervalSummary(EvaluationModel):
    median: Annotated[float, Field(allow_inf_nan=False)]
    confidence_low: Annotated[float, Field(allow_inf_nan=False)]
    confidence_high: Annotated[float, Field(allow_inf_nan=False)]
    sample_count: Annotated[int, Field(gt=0)]


class PlanTrial(EvaluationModel):
    seed: Annotated[int, Field(ge=0)]
    topology: str
    method: EvaluationMethod
    strategy: OptimizationStrategy
    plan_id: str
    tensor_parallel: Annotated[int, Field(gt=0)]
    pipeline_parallel: Annotated[int, Field(gt=0)]
    data_parallel: Annotated[int, Field(gt=0)]
    expert_parallel: Annotated[int, Field(gt=0)]
    predicted_p95_ttft_ms: Annotated[float, Field(ge=0.0)]
    predicted_p95_ttft_lower_ms: Annotated[float, Field(ge=0.0)]
    predicted_p95_ttft_upper_ms: Annotated[float, Field(ge=0.0)]
    isolated_service_ttft_ms: Annotated[float, Field(ge=0.0)]
    observed_p95_ttft_ms: Annotated[float, Field(ge=0.0)]
    observed_p99_tpot_ms: Annotated[float, Field(ge=0.0)]
    observed_p95_e2e_ms: Annotated[float, Field(ge=0.0)]
    throughput_tokens_per_second: Annotated[float, Field(ge=0.0)]
    goodput_tokens_per_second: Annotated[float, Field(ge=0.0)]
    slo_attainment: Annotated[float, Field(ge=0.0, le=1.0)]
    communication_time_ms: Annotated[float, Field(ge=0.0)]
    communication_overhead_fraction: Annotated[float, Field(ge=0.0, le=1.0)]
    cost_usd_per_million_tokens: Annotated[float, Field(ge=0.0)]
    solver_time_ms: Annotated[float, Field(ge=0.0)]
    simulator_calls: Annotated[int, Field(ge=0)]
    simulation_artifact: str
    isolated_simulation_artifact: str
    compiler_artifact: str


class MethodSummary(EvaluationModel):
    method: EvaluationMethod
    p95_ttft_ms: IntervalSummary
    p99_tpot_ms: IntervalSummary
    p95_end_to_end_ms: IntervalSummary
    throughput_tokens_per_second: IntervalSummary
    goodput_tokens_per_second: IntervalSummary
    communication_time_ms: IntervalSummary
    communication_overhead_fraction: IntervalSummary
    predicted_cost_usd_per_million_tokens: IntervalSummary
    slo_attainment: IntervalSummary


class TwinTrial(EvaluationModel):
    seed: Annotated[int, Field(ge=0)]
    topology: str
    method: EvaluationMethod
    predicted_ms: Annotated[float, Field(ge=0.0)]
    isolated_observed_ms: Annotated[float, Field(ge=0.0)]
    loaded_observed_p95_ms: Annotated[float, Field(ge=0.0)]
    workload_queueing_delta_ms: Annotated[float, Field(ge=0.0)]
    absolute_error_ms: Annotated[float, Field(ge=0.0)]
    relative_error: Annotated[float, Field(ge=0.0)]
    interval_covered: bool


class TwinSummary(EvaluationModel):
    mean_absolute_error_ms: Annotated[float, Field(ge=0.0)]
    median_relative_error: Annotated[float, Field(ge=0.0)]
    rank_correlation: Annotated[float, Field(ge=-1.0, le=1.0)]
    interval_coverage: Annotated[float, Field(ge=0.0, le=1.0)]
    median_workload_queueing_delta_ms: Annotated[float, Field(ge=0.0)]
    hierarchical_selection_regret_ms: Annotated[float, Field(ge=0.0)]
    caveat: str


class DiagnosisTrial(EvaluationModel):
    seed: Annotated[int, Field(ge=0)]
    topology: str
    fault_id: str
    ground_truth: BottleneckKind
    top_one: BottleneckKind
    top_three: tuple[BottleneckKind, ...]
    top_one_correct: bool
    top_three_correct: bool
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    diagnosis_latency_ms: Annotated[float, Field(ge=0.0)]
    healthy_p95_ttft_ms: Annotated[float, Field(ge=0.0)]
    degraded_p95_ttft_ms: Annotated[float, Field(ge=0.0)]
    counterfactual_p95_ttft_ms: Annotated[float, Field(ge=0.0)]
    counterfactual_healthy_residual_ms: Annotated[float, Field(ge=0.0)]
    diagnosis_artifact: str


class DiagnosisSummary(EvaluationModel):
    top_one_accuracy: Annotated[float, Field(ge=0.0, le=1.0)]
    top_three_accuracy: Annotated[float, Field(ge=0.0, le=1.0)]
    median_diagnosis_latency_ms: Annotated[float, Field(ge=0.0)]
    median_confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    median_counterfactual_residual_ms: Annotated[float, Field(ge=0.0)]


class RecoveryTrial(EvaluationModel):
    seed: Annotated[int, Field(ge=0)]
    topology: str
    fault_id: str
    method: RecoveryMethod
    restored_slo: bool
    post_recovery_p95_ttft_ms: Annotated[float, Field(ge=0.0)]
    action_seconds: Annotated[float | None, Field(ge=0.0)]
    estimated_action_cost_usd: Annotated[float, Field(ge=0.0)]
    incorrect_recovery: bool


class RecoveryMethodSummary(EvaluationModel):
    method: str
    restoration_rate: Annotated[float, Field(ge=0.0, le=1.0)]
    median_action_seconds: Annotated[float | None, Field(ge=0.0)]
    median_cost_usd: Annotated[float, Field(ge=0.0)]
    incorrect_recovery_rate: Annotated[float, Field(ge=0.0, le=1.0)]


class FabricEvaluationResult(EvaluationModel):
    schema_version: Literal["sloforge.fabric.evaluation/v1"] = "sloforge.fabric.evaluation/v1"
    generated_at: str
    git_commit: str
    command: tuple[str, ...]
    validation_mode: Literal["synthetic_cpu"] = "synthetic_cpu"
    hardware_backed_validation: Literal["not_exercised_no_compatible_hardware"]
    config: EvaluationConfig
    plan_trials: tuple[PlanTrial, ...]
    method_summaries: tuple[MethodSummary, ...]
    twin_trials: tuple[TwinTrial, ...]
    twin_summary: TwinSummary
    diagnosis_trials: tuple[DiagnosisTrial, ...]
    diagnosis_summary: DiagnosisSummary
    recovery_trials: tuple[RecoveryTrial, ...]
    recovery_summaries: tuple[RecoveryMethodSummary, ...]
    limitations: tuple[str, ...]
    artifacts: tuple[ArtifactRef, ...]

    @model_validator(mode="after")
    def complete_matrix_and_unique_artifacts(self) -> Self:
        expected_trials = (
            len(self.config.seeds) * len(self.config.topology_fixtures) * len(self.config.methods)
        )
        if len(self.plan_trials) != expected_trials:
            raise ValueError("plan trials do not cover the configured evaluation matrix")
        methods = {item.method for item in self.method_summaries}
        if methods != set(self.config.methods):
            raise ValueError("method summaries do not cover configured methods")
        paths = [artifact.path for artifact in self.artifacts]
        if len(paths) != len(set(paths)):
            raise ValueError("evaluation artifact references must be unique")
        return self


def _bootstrap_median(values: tuple[float, ...], *, seed: int, repetitions: int) -> IntervalSummary:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    generator = random.Random(seed)
    medians = sorted(
        statistics.median(generator.choice(values) for _ in values) for _ in range(repetitions)
    )
    return IntervalSummary(
        median=statistics.median(values),
        confidence_low=percentile(medians, 0.025),
        confidence_high=percentile(medians, 0.975),
        sample_count=len(values),
    )


def _ranks(values: tuple[float, ...]) -> tuple[float, ...]:
    result = [0.0] * len(values)
    ordered = sorted(range(len(values)), key=lambda index: (values[index], index))
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[cursor]]:
            end += 1
        rank = (cursor + end - 1) / 2.0
        for index in ordered[cursor:end]:
            result[index] = rank
        cursor = end
    return tuple(result)


def _spearman(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_ranks, right_ranks = _ranks(left), _ranks(right)
    left_mean, right_mean = statistics.mean(left_ranks), statistics.mean(right_ranks)
    numerator = sum(
        (first - left_mean) * (second - right_mean)
        for first, second in zip(left_ranks, right_ranks, strict=True)
    )
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left_ranks)
        * sum((value - right_mean) ** 2 for value in right_ranks)
    )
    return numerator / denominator if denominator else 0.0


def _strategy(method: EvaluationMethod) -> OptimizationStrategy:
    return {
        EvaluationMethod.RANDOM: OptimizationStrategy.RANDOM_PLACEMENT,
        EvaluationMethod.SEQUENTIAL: OptimizationStrategy.TOPOLOGY_UNAWARE,
        EvaluationMethod.TOPOLOGY_UNAWARE: OptimizationStrategy.TOPOLOGY_UNAWARE,
        EvaluationMethod.GREEDY: OptimizationStrategy.GREEDY_TOPOLOGY_AWARE,
        EvaluationMethod.HIERARCHICAL: OptimizationStrategy.HIERARCHICAL,
    }[method]


def _topology(name: str) -> TopologyGraph:
    if name != "two_node_degraded_network":
        return build_canonical_fixture(name)
    return build_canonical_fixture(
        FixtureSpec(
            schema_version="sloforge.fabric.fixture/v1",
            name=name,
            hosts=2,
            gpus_per_host=8,
            numa_per_host=2,
            network="infiniband",
            rails_per_host=2,
            nvlink_group_size=4,
            mig_instances_per_gpu=0,
            degraded_edge="network",
        )
    )


def _workload(seed: int, request_count: int) -> SimulationWorkload:
    arrivals = 0.0
    requests: list[SimulationRequestShape] = []
    for index in range(request_count):
        arrivals += 15_000.0 if index and index % 4 == 0 else 80.0 + (seed + index * 11) % 70
        priority: Literal["high", "normal", "low"]
        if index % 4 == 0:
            prompt, output, priority, request_class = 8_192, 64, "normal", "long_context"
        elif index % 3 == 0:
            prompt, output, priority, request_class = 2_048, 96, "low", "batch"
        else:
            prompt, output, priority, request_class = 128, 24, "high", "interactive"
        requests.append(
            SimulationRequestShape(
                arrival_us=arrivals,
                prompt_tokens=prompt,
                output_tokens=output,
                priority=priority,
                request_class=request_class,
            )
        )
    return SimulationWorkload(
        request_count=request_count,
        arrival_interval_us=0.0,
        prompt_tokens=8_192,
        output_tokens=96,
        requests=tuple(requests),
    )


def _isolated_service_workload() -> SimulationWorkload:
    """Representative p95 request shape without workload queueing."""

    return SimulationWorkload(
        request_count=1,
        arrival_interval_us=0.0,
        prompt_tokens=8_192,
        output_tokens=96,
        requests=(
            SimulationRequestShape(
                arrival_us=0.0,
                prompt_tokens=8_192,
                output_tokens=96,
                priority="normal",
                request_class="isolated_p95_shape",
            ),
        ),
    )


def _logical_reference(repository_root: Path) -> DocumentReference:
    path = repository_root / "tests" / "fixtures" / "ir" / "deployment-plan-v1.json"
    return DocumentReference(
        kind="DeploymentPlan",
        api_version="sloforge.io/v1",
        uri=str(path),
        digest=ArtifactDigest(value=sha256_file(path)),
        uid="fabric-evaluation-logical-plan",
        generation=1,
    )


def _compile_request(
    *,
    repository_root: Path,
    topology: TopologyGraph,
    profile: FabricProfile,
    method: EvaluationMethod,
    seed: int,
    environment_digest: ArtifactDigest,
) -> CompilerRequest:
    # Fixed parallelism isolates rank placement for random/sequential/greedy/
    # hierarchical comparisons. The topology-unaware optimizer is additionally
    # allowed to select degrees, matching its intended baseline semantics.
    fixed = method is not EvaluationMethod.TOPOLOGY_UNAWARE
    return CompilerRequest(
        logical_deployment_plan=_logical_reference(repository_root),
        model=synthetic_moe_model_graph(),
        topology=topology,
        fabric_profile=profile,
        constraints=CompilerConstraints(
            prompt_tokens_p95=8_192,
            output_tokens_p95=96,
            maximum_concurrent_requests=32,
            p95_ttft_ms=5_000.0,
            p99_tpot_ms=45.0,
            minimum_goodput_tokens_per_second=100.0,
            minimum_availability=0.95,
            maximum_ranks=16,
            tensor_parallel_degree=8 if fixed else None,
            pipeline_parallel_degree=1 if fixed else None,
            data_parallel_degree=2 if fixed else None,
            expert_parallel_degree=4 if fixed else None,
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
        strategy=_strategy(method),
        generated_at=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=seed),
        seed=seed,
        git_commit=git_commit(repository_root),
        environment_digest=environment_digest,
    )


def _output_metrics(
    output: FabricSimulationOutput,
    workload: SimulationWorkload,
    *,
    ttft_slo_ms: float,
    tpot_slo_ms: float,
) -> tuple[float, float, float, float, float, float, float, float]:
    latencies = request_latencies(output)
    ttft = tuple(item.ttft_us / 1_000.0 for item in latencies)
    e2e = tuple(item.end_to_end_us / 1_000.0 for item in latencies)
    shapes = {f"request-{index:06d}": shape for index, shape in enumerate(workload.requests)}
    token_steps: list[float] = []
    communication_us = 0.0
    for operation in output.operations:
        if ":collective-" in operation.operation_id or ":kv-" in operation.operation_id:
            communication_us += operation.duration_us
        if operation.operation_id.endswith(":decode"):
            request_id = operation.operation_id.partition(":")[0]
            token_steps.append(operation.duration_us / shapes[request_id].output_tokens / 1_000.0)
    makespan_seconds = max(output.metrics.makespan_us / 1_000_000.0, 1e-12)
    total_tokens = sum(shape.output_tokens for shape in workload.requests)
    attained = tuple(
        latency.ttft_us / 1_000.0 <= ttft_slo_ms and max(token_steps, default=0.0) <= tpot_slo_ms
        for latency in latencies
    )
    good_tokens = sum(
        shape.output_tokens
        for shape, passed in zip(workload.requests, attained, strict=True)
        if passed
    )
    return (
        percentile(ttft, 0.95),
        percentile(token_steps, 0.99),
        percentile(e2e, 0.95),
        total_tokens / makespan_seconds,
        good_tokens / makespan_seconds,
        sum(attained) / len(attained),
        communication_us / 1_000.0,
        min(1.0, communication_us / max(output.metrics.total_work_us, 1e-12)),
    )


def _artifact(path: Path, root: Path, provenance: ArtifactProvenance) -> ArtifactRef:
    return ArtifactRef(
        path=str(path.relative_to(root)),
        sha256=sha256_file(path),
        provenance=provenance,
    )


def _write_simulation(path: Path, output: FabricSimulationOutput) -> None:
    write_json(path, output.model_dump(mode="json"))


def _plan_trial(
    *,
    seed: int,
    topology_name: str,
    method: EvaluationMethod,
    compilation: PhysicalCompileResult,
    output: FabricSimulationOutput,
    isolated_output: FabricSimulationOutput,
    workload: SimulationWorkload,
    config: EvaluationConfig,
    simulation_path: Path,
    isolated_simulation_path: Path,
    compiler_path: Path,
) -> PlanTrial:
    plan = compilation.selected
    observed = _output_metrics(
        output,
        workload,
        ttft_slo_ms=config.p95_ttft_slo_ms,
        tpot_slo_ms=config.p99_tpot_slo_ms,
    )
    metrics = plan.predicted_metrics
    parallel = plan.parallelism
    return PlanTrial(
        seed=seed,
        topology=topology_name,
        method=method,
        strategy=compilation.strategy,
        plan_id=plan.plan_id,
        tensor_parallel=parallel.tensor_parallel_degree,
        pipeline_parallel=parallel.pipeline_parallel_degree,
        data_parallel=parallel.data_parallel_degree,
        expert_parallel=parallel.expert_parallel_degree,
        predicted_p95_ttft_ms=metrics.p95_ttft_ms.estimate,
        predicted_p95_ttft_lower_ms=metrics.p95_ttft_ms.lower,
        predicted_p95_ttft_upper_ms=metrics.p95_ttft_ms.upper,
        isolated_service_ttft_ms=request_latencies(isolated_output)[0].ttft_us / 1_000.0,
        observed_p95_ttft_ms=observed[0],
        observed_p99_tpot_ms=observed[1],
        observed_p95_e2e_ms=observed[2],
        throughput_tokens_per_second=observed[3],
        goodput_tokens_per_second=observed[4],
        slo_attainment=observed[5],
        communication_time_ms=observed[6],
        communication_overhead_fraction=observed[7],
        cost_usd_per_million_tokens=metrics.cost_usd_per_million_tokens.estimate,
        solver_time_ms=compilation.solver_time_ms,
        simulator_calls=compilation.simulator_calls,
        simulation_artifact=str(simulation_path),
        isolated_simulation_artifact=str(isolated_simulation_path),
        compiler_artifact=str(compiler_path),
    )


def _method_summaries(
    trials: tuple[PlanTrial, ...], config: EvaluationConfig
) -> tuple[MethodSummary, ...]:
    result: list[MethodSummary] = []
    for index, method in enumerate(config.methods):
        selected = tuple(item for item in trials if item.method is method)
        result.append(
            MethodSummary(
                method=method,
                p95_ttft_ms=_bootstrap_median(
                    tuple(item.observed_p95_ttft_ms for item in selected),
                    seed=10_000 + index,
                    repetitions=config.bootstrap_repetitions,
                ),
                p99_tpot_ms=_bootstrap_median(
                    tuple(item.observed_p99_tpot_ms for item in selected),
                    seed=20_000 + index,
                    repetitions=config.bootstrap_repetitions,
                ),
                p95_end_to_end_ms=_bootstrap_median(
                    tuple(item.observed_p95_e2e_ms for item in selected),
                    seed=25_000 + index,
                    repetitions=config.bootstrap_repetitions,
                ),
                throughput_tokens_per_second=_bootstrap_median(
                    tuple(item.throughput_tokens_per_second for item in selected),
                    seed=30_000 + index,
                    repetitions=config.bootstrap_repetitions,
                ),
                goodput_tokens_per_second=_bootstrap_median(
                    tuple(item.goodput_tokens_per_second for item in selected),
                    seed=35_000 + index,
                    repetitions=config.bootstrap_repetitions,
                ),
                communication_time_ms=_bootstrap_median(
                    tuple(item.communication_time_ms for item in selected),
                    seed=40_000 + index,
                    repetitions=config.bootstrap_repetitions,
                ),
                communication_overhead_fraction=_bootstrap_median(
                    tuple(item.communication_overhead_fraction for item in selected),
                    seed=45_000 + index,
                    repetitions=config.bootstrap_repetitions,
                ),
                predicted_cost_usd_per_million_tokens=_bootstrap_median(
                    tuple(item.cost_usd_per_million_tokens for item in selected),
                    seed=47_000 + index,
                    repetitions=config.bootstrap_repetitions,
                ),
                slo_attainment=_bootstrap_median(
                    tuple(item.slo_attainment for item in selected),
                    seed=50_000 + index,
                    repetitions=config.bootstrap_repetitions,
                ),
            )
        )
    return tuple(result)


def _twin_results(trials: tuple[PlanTrial, ...]) -> tuple[tuple[TwinTrial, ...], TwinSummary]:
    items = tuple(
        TwinTrial(
            seed=trial.seed,
            topology=trial.topology,
            method=trial.method,
            predicted_ms=trial.predicted_p95_ttft_ms,
            isolated_observed_ms=trial.isolated_service_ttft_ms,
            loaded_observed_p95_ms=trial.observed_p95_ttft_ms,
            workload_queueing_delta_ms=max(
                0.0, trial.observed_p95_ttft_ms - trial.isolated_service_ttft_ms
            ),
            absolute_error_ms=abs(trial.predicted_p95_ttft_ms - trial.isolated_service_ttft_ms),
            relative_error=(
                abs(trial.predicted_p95_ttft_ms - trial.isolated_service_ttft_ms)
                / max(trial.isolated_service_ttft_ms, 1e-12)
            ),
            interval_covered=(
                trial.predicted_p95_ttft_lower_ms
                <= trial.isolated_service_ttft_ms
                <= trial.predicted_p95_ttft_upper_ms
            ),
        )
        for trial in trials
    )
    correlations: list[float] = []
    regrets: list[float] = []
    groups = sorted({(item.seed, item.topology) for item in trials})
    for seed, topology in groups:
        group = tuple(item for item in trials if item.seed == seed and item.topology == topology)
        correlations.append(
            _spearman(
                tuple(item.predicted_p95_ttft_ms for item in group),
                tuple(item.isolated_service_ttft_ms for item in group),
            )
        )
        hierarchy = next(
            (item for item in group if item.method is EvaluationMethod.HIERARCHICAL), None
        )
        if hierarchy is not None:
            regrets.append(
                max(
                    0.0,
                    hierarchy.observed_p95_ttft_ms
                    - min(item.observed_p95_ttft_ms for item in group),
                )
            )
    return items, TwinSummary(
        mean_absolute_error_ms=statistics.mean(item.absolute_error_ms for item in items),
        median_relative_error=statistics.median(item.relative_error for item in items),
        rank_correlation=statistics.mean(correlations) if correlations else 0.0,
        interval_coverage=sum(item.interval_covered for item in items) / len(items),
        median_workload_queueing_delta_ms=statistics.median(
            item.workload_queueing_delta_ms for item in items
        ),
        hierarchical_selection_regret_ms=statistics.median(regrets) if regrets else 0.0,
        caveat=(
            "Compiler isolated-service intervals are compared with one representative p95-shape "
            "request in the deterministic Rust simulator; loaded p95 and queueing are reported "
            "separately. This remains synthetic internal validation, not hardware accuracy."
        ),
    )


def _faults(request: FabricSimulationRequest) -> tuple[tuple[TimedFault, BottleneckKind], ...]:
    demanded = {
        demand.resource_id for operation in request.operations for demand in operation.demands
    }
    rail = next(
        resource
        for resource in request.resources
        if resource.kind is ResourceKind.NETWORK_RAIL and resource.id in demanded
    )
    return (
        (
            TimedFault(
                id="network-bandwidth-degradation",
                start_us=0.0,
                end_us=10_000_000_000.0,
                effect=ResourceRateFault(resource_id=rail.id, multiplier=0.15),
                ground_truth_label=BottleneckKind.NETWORK_BANDWIDTH_DEGRADATION.value,
            ),
            BottleneckKind.NETWORK_BANDWIDTH_DEGRADATION,
        ),
        (
            TimedFault(
                id="rank-2-straggler",
                start_us=0.0,
                end_us=10_000_000_000.0,
                effect=RankSlowdownFault(rank_id="rank-2", multiplier=0.35),
                ground_truth_label=BottleneckKind.RANK_STRAGGLER.value,
            ),
            BottleneckKind.RANK_STRAGGLER,
        ),
    )


def _recovery_trials(
    *,
    seed: int,
    topology: str,
    fault: TimedFault,
    expected: BottleneckKind,
    diagnosis_top: BottleneckKind,
    degraded_p95: float,
    restored_p95: float,
    healthy_p95: float,
    plan: PhysicalExecutionPlan,
    diagnosis_record: DiagnosisRecord,
) -> tuple[RecoveryTrial, ...]:
    # Action times and resource costs are policy parameters, not measured
    # latencies. The post-action SLO metric always comes from a simulator run.
    proposal = plan_recovery(
        diagnosis_record,
        plan,
        policy=RecoveryPolicy(
            minimum_diagnosis_confidence=0.0,
            target_p95_ttft_ms=max(healthy_p95 * 1.10, 1.0),
        ),
    )
    target = max(healthy_p95 * 1.10, 1.0)
    gpu_hourly = 4.0
    full_deployment_seconds = 180.0
    threshold_seconds = 60.0
    restart_repairs = expected is BottleneckKind.RANK_STRAGGLER
    diagnosis_repairs = diagnosis_top is expected
    rows: tuple[tuple[RecoveryMethod, float, float | None, float, bool], ...] = (
        ("no_recovery", degraded_p95, None, 0.0, False),
        (
            "restart_affected_worker",
            restored_p95 if restart_repairs else degraded_p95,
            120.0,
            gpu_hourly * 120.0 / 3_600.0,
            restart_repairs,
        ),
        (
            "replace_full_deployment",
            restored_p95,
            full_deployment_seconds,
            gpu_hourly * len(plan.rank_placement.bindings) * full_deployment_seconds / 3_600.0,
            True,
        ),
        (
            "threshold_recovery",
            restored_p95,
            threshold_seconds,
            gpu_hourly * threshold_seconds / 3_600.0,
            True,
        ),
        (
            "diagnosis_driven",
            restored_p95 if diagnosis_repairs else degraded_p95,
            proposal.expected_build_seconds + proposal.expected_disruption_seconds,
            proposal.expected_cost_usd,
            diagnosis_repairs,
        ),
    )
    return tuple(
        RecoveryTrial(
            seed=seed,
            topology=topology,
            fault_id=fault.id,
            method=method,
            restored_slo=post <= target,
            post_recovery_p95_ttft_ms=post,
            action_seconds=action_seconds,
            estimated_action_cost_usd=cost,
            incorrect_recovery=(method != "no_recovery" and not correct_action),
        )
        for method, post, action_seconds, cost, correct_action in rows
    )


def _diagnosis_and_recovery(
    *,
    repository_root: Path,
    artifact_root: Path,
    seed: int,
    topology_name: str,
    topology: TopologyGraph,
    profile: FabricProfile,
    plan: PhysicalExecutionPlan,
    workload: SimulationWorkload,
    healthy_request: FabricSimulationRequest,
    healthy_output: FabricSimulationOutput,
    healthy_path: Path,
    config: EvaluationConfig,
) -> tuple[tuple[DiagnosisTrial, ...], tuple[RecoveryTrial, ...], tuple[Path, ...]]:
    output_dir = artifact_root / "autopsy" / topology_name / f"seed-{seed}"
    workload_hash = hashlib.sha256(workload.model_dump_json().encode()).hexdigest()
    healthy_metrics = _output_metrics(
        healthy_output,
        workload,
        ttft_slo_ms=config.p95_ttft_slo_ms,
        tpot_slo_ms=config.p99_tpot_slo_ms,
    )
    diagnosis_trials: list[DiagnosisTrial] = []
    recovery_trials: list[RecoveryTrial] = []
    artifacts: list[Path] = []
    for fault, expected in _faults(healthy_request):
        degraded_request = healthy_request.model_copy(update={"faults": (fault,)})
        degraded = run_simulation(
            degraded_request,
            repository_root=repository_root,
            timeout_seconds=config.simulator_timeout_seconds,
        )
        restored_request = degraded_request.model_copy(
            update={"counterfactuals": (RemoveFault(fault_id=fault.id),)}
        )
        restored = run_simulation(
            restored_request,
            repository_root=repository_root,
            timeout_seconds=config.simulator_timeout_seconds,
        )
        degraded_path = output_dir / fault.id / "degraded-simulation.json"
        restored_path = output_dir / fault.id / "counterfactual-simulation.json"
        _write_simulation(degraded_path, degraded)
        _write_simulation(restored_path, restored)
        healthy_run = capture_simulation_run(
            run_id=f"healthy-{topology_name}-{seed}-{fault.id}",
            request=healthy_request,
            output=healthy_output,
            plan=plan,
            topology_fingerprint=canonical_hash(topology),
            workload_fingerprint=workload_hash,
            artifact_path=healthy_path,
        )
        degraded_run = capture_simulation_run(
            run_id=f"degraded-{topology_name}-{seed}-{fault.id}",
            request=degraded_request,
            output=degraded,
            plan=plan,
            topology_fingerprint=canonical_hash(topology),
            workload_fingerprint=workload_hash,
            artifact_path=degraded_path,
        )
        started = time.perf_counter_ns()
        comparison = compare_runs(healthy_run, degraded_run)
        diagnosis = diagnose(degraded_run, comparison=comparison, baseline=healthy_run)
        latency_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        degraded_metrics = _output_metrics(
            degraded,
            workload,
            ttft_slo_ms=config.p95_ttft_slo_ms,
            tpot_slo_ms=config.p99_tpot_slo_ms,
        )
        restored_metrics = _output_metrics(
            restored,
            workload,
            ttft_slo_ms=config.p95_ttft_slo_ms,
            tpot_slo_ms=config.p99_tpot_slo_ms,
        )
        comparison_path = output_dir / fault.id / "comparison.json"
        diagnosis_path = output_dir / fault.id / "diagnosis.json"
        write_json(comparison_path, comparison.model_dump(mode="json"))
        write_json(diagnosis_path, diagnosis.model_dump(mode="json"))
        artifacts.extend((degraded_path, restored_path, comparison_path, diagnosis_path))
        diagnosis_trials.append(
            DiagnosisTrial(
                seed=seed,
                topology=topology_name,
                fault_id=fault.id,
                ground_truth=expected,
                top_one=diagnosis.top_hypothesis,
                top_three=diagnosis.top_three,
                top_one_correct=diagnosis.top_hypothesis is expected,
                top_three_correct=expected in diagnosis.top_three,
                confidence=diagnosis.confidence,
                diagnosis_latency_ms=latency_ms,
                healthy_p95_ttft_ms=healthy_metrics[0],
                degraded_p95_ttft_ms=degraded_metrics[0],
                counterfactual_p95_ttft_ms=restored_metrics[0],
                counterfactual_healthy_residual_ms=abs(restored_metrics[0] - healthy_metrics[0]),
                diagnosis_artifact=str(diagnosis_path.relative_to(artifact_root)),
            )
        )
        recovery_trials.extend(
            _recovery_trials(
                seed=seed,
                topology=topology_name,
                fault=fault,
                expected=expected,
                diagnosis_top=diagnosis.top_hypothesis,
                degraded_p95=degraded_metrics[0],
                restored_p95=restored_metrics[0],
                healthy_p95=healthy_metrics[0],
                plan=plan,
                diagnosis_record=diagnosis,
            )
        )
    return tuple(diagnosis_trials), tuple(recovery_trials), tuple(artifacts)


def _diagnosis_summary(trials: tuple[DiagnosisTrial, ...]) -> DiagnosisSummary:
    return DiagnosisSummary(
        top_one_accuracy=sum(item.top_one_correct for item in trials) / len(trials),
        top_three_accuracy=sum(item.top_three_correct for item in trials) / len(trials),
        median_diagnosis_latency_ms=statistics.median(item.diagnosis_latency_ms for item in trials),
        median_confidence=statistics.median(item.confidence for item in trials),
        median_counterfactual_residual_ms=statistics.median(
            item.counterfactual_healthy_residual_ms for item in trials
        ),
    )


def _recovery_summaries(trials: tuple[RecoveryTrial, ...]) -> tuple[RecoveryMethodSummary, ...]:
    methods = tuple(dict.fromkeys(item.method for item in trials))
    result: list[RecoveryMethodSummary] = []
    for method in methods:
        selected = tuple(item for item in trials if item.method == method)
        times = tuple(item.action_seconds for item in selected if item.action_seconds is not None)
        result.append(
            RecoveryMethodSummary(
                method=method,
                restoration_rate=sum(item.restored_slo for item in selected) / len(selected),
                median_action_seconds=statistics.median(times) if times else None,
                median_cost_usd=statistics.median(
                    item.estimated_action_cost_usd for item in selected
                ),
                incorrect_recovery_rate=sum(item.incorrect_recovery for item in selected)
                / len(selected),
            )
        )
    return tuple(result)


def _bar_plot(path: Path, title: str, values: tuple[tuple[str, float], ...], unit: str) -> None:
    maximum = max((value for _, value in values), default=1.0) or 1.0
    bars: list[str] = []
    for index, (label, value) in enumerate(values):
        y = 55 + index * 48
        width = value / maximum * 520.0
        bars.append(
            f'<text x="10" y="{y}">{html.escape(label)}</text>'
            f'<rect x="230" y="{y - 22}" width="{width:.3f}" height="26" fill="#2563eb"/>'
            f'<text x="{240 + width:.3f}" y="{y}">{value:.4g} {html.escape(unit)}</text>'
        )
    height = 80 + len(values) * 48
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="920" height="{height}" '
        f'role="img" aria-label="{html.escape(title)}"><rect width="100%" height="100%" '
        f'fill="white"/><text x="10" y="24" font-weight="bold">{html.escape(title)}</text>'
        + "".join(bars)
        + "</svg>\n",
        encoding="utf-8",
    )


def _render_fabric_report(result: FabricEvaluationResult) -> str:
    fastest = min(result.method_summaries, key=lambda item: item.p95_ttft_ms.median)
    hierarchy = next(
        item for item in result.method_summaries if item.method is EvaluationMethod.HIERARCHICAL
    )
    hierarchy_delta = (
        hierarchy.p95_ttft_ms.median / max(fastest.p95_ttft_ms.median, 1e-12) - 1.0
    ) * 100.0
    lines = [
        "# SLOForge Fabric evaluation",
        "",
        "This report is generated from deterministic CPU simulation artifacts. It contains no "
        "hardware-backed GPU or network measurements.",
        "",
        "## H1 — topology-aware compilation",
        "",
        "| method | p95 TTFT ms (median, 95% bootstrap CI) | communication ms | SLO attainment |",
        "|---|---:|---:|---:|",
    ]
    for item in result.method_summaries:
        lines.append(
            f"| {item.method.value} | {item.p95_ttft_ms.median:.3f} "
            f"[{item.p95_ttft_ms.confidence_low:.3f}, {item.p95_ttft_ms.confidence_high:.3f}] | "
            f"{item.communication_time_ms.median:.3f} | {item.slo_attainment.median:.3f} |"
        )
    lines.extend(
        (
            "",
            f"Fastest median: `{fastest.method.value}` at "
            f"{fastest.p95_ttft_ms.median:.3f} ms. The hierarchical compiler was "
            f"{hierarchy_delta:+.2f}% relative to that baseline. This comparison is reported "
            "even when the primary method loses.",
        )
    )
    lines.extend(
        (
            "",
            "## H2 — digital-twin ranking",
            "",
            f"- Mean absolute TTFT error: {result.twin_summary.mean_absolute_error_ms:.3f} ms",
            f"- Median relative TTFT error: {result.twin_summary.median_relative_error:.3f}",
            f"- Spearman rank correlation: {result.twin_summary.rank_correlation:.3f}",
            f"- Prediction-interval coverage: {result.twin_summary.interval_coverage:.3f}",
            f"- Median loaded-workload queueing delta: "
            f"{result.twin_summary.median_workload_queueing_delta_ms:.3f} ms",
            f"- Hierarchical selection regret: "
            f"{result.twin_summary.hierarchical_selection_regret_ms:.3f} ms",
            "",
            result.twin_summary.caveat,
            "",
            "## Validation provenance",
            "",
            f"- Mode: `{result.validation_mode}`",
            f"- Hardware-backed validation: `{result.hardware_backed_validation}`",
            f"- Git commit: `{result.git_commit}`",
            "- Raw results: `artifacts/fabric/evaluation/result.json`",
            "- Artifact manifest: `artifacts/fabric/evaluation/manifest.json`",
            "",
            "## Limitations",
            "",
        )
    )
    lines.extend(f"- {limitation}" for limitation in result.limitations)
    return "\n".join(lines) + "\n"


def _render_autopsy_report(result: FabricEvaluationResult) -> str:
    diagnosis = result.diagnosis_summary
    lines = [
        "# SLOForge Autopsy and recovery evaluation",
        "",
        "All faults and observations in this report come from deterministic simulator artifacts.",
        "",
        "## H3 — causal diagnosis",
        "",
        f"- Top-1 diagnosis accuracy: {diagnosis.top_one_accuracy:.3f}",
        f"- Top-3 diagnosis accuracy: {diagnosis.top_three_accuracy:.3f}",
        f"- Median diagnosis latency: {diagnosis.median_diagnosis_latency_ms:.3f} ms",
        f"- Median diagnosis confidence: {diagnosis.median_confidence:.3f}",
        f"- Median counterfactual healthy residual: "
        f"{diagnosis.median_counterfactual_residual_ms:.6f} ms",
        "",
        "## H4 — recovery policy",
        "",
        "| method | restoration rate | median action seconds | median estimated cost USD | "
        "incorrect action rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in result.recovery_summaries:
        action = (
            "n/a" if item.median_action_seconds is None else f"{item.median_action_seconds:.3f}"
        )
        lines.append(
            f"| {item.method} | {item.restoration_rate:.3f} | {action} | "
            f"{item.median_cost_usd:.6f} | {item.incorrect_recovery_rate:.3f} |"
        )
    lines.extend(
        (
            "",
            "Action time and cost are declared recovery-policy parameters. Post-recovery TTFT is "
            "derived from counterfactual Rust simulation; this is not a live-cluster failover test.",
            "",
            "Raw evidence is indexed by `artifacts/fabric/evaluation/manifest.json`.",
        )
    )
    return "\n".join(lines) + "\n"


def _write_reports(
    result: FabricEvaluationResult,
    *,
    report_dir: Path,
) -> tuple[Path, ...]:
    fabric_markdown = report_dir / "fabric-evaluation.md"
    fabric_html = report_dir / "fabric-evaluation.html"
    autopsy_markdown = report_dir / "autopsy-evaluation.md"
    fabric_text = _render_fabric_report(result)
    autopsy_text = _render_autopsy_report(result)
    fabric_markdown.parent.mkdir(parents=True, exist_ok=True)
    fabric_markdown.write_text(fabric_text, encoding="utf-8")
    autopsy_markdown.write_text(autopsy_text, encoding="utf-8")
    fabric_html.write_text(
        "<!doctype html><meta charset='utf-8'><title>SLOForge Fabric evaluation</title>"
        "<style>body{font:16px system-ui;max-width:1100px;margin:2rem auto;white-space:pre-wrap}</style>"
        f"<body>{html.escape(fabric_text)}</body>\n",
        encoding="utf-8",
    )
    h1_plot = report_dir / "fabric-evaluation-h1.svg"
    h3_plot = report_dir / "autopsy-evaluation-h3.svg"
    h4_plot = report_dir / "recovery-evaluation-h4.svg"
    _bar_plot(
        h1_plot,
        "H1 median p95 TTFT (synthetic)",
        tuple((item.method.value, item.p95_ttft_ms.median) for item in result.method_summaries),
        "ms",
    )
    _bar_plot(
        h3_plot,
        "H3 diagnosis accuracy (synthetic)",
        (
            ("top-1", result.diagnosis_summary.top_one_accuracy),
            ("top-3", result.diagnosis_summary.top_three_accuracy),
        ),
        "ratio",
    )
    _bar_plot(
        h4_plot,
        "H4 SLO restoration rate (synthetic)",
        tuple((item.method, item.restoration_rate) for item in result.recovery_summaries),
        "ratio",
    )
    return fabric_markdown, fabric_html, autopsy_markdown, h1_plot, h3_plot, h4_plot


def validate_evaluation_artifacts(
    *, artifact_root: Path, report_dir: Path
) -> FabricEvaluationResult:
    result_path = artifact_root / "result.json"
    result = FabricEvaluationResult.model_validate_json(result_path.read_text(encoding="utf-8"))
    for artifact in result.artifacts:
        path = artifact_root / artifact.path
        if not path.is_file() or sha256_file(path) != artifact.sha256:
            raise RuntimeError(f"evaluation artifact hash mismatch: {artifact.path}")
    fabric_report = (report_dir / "fabric-evaluation.md").read_text(encoding="utf-8")
    autopsy_report = (report_dir / "autopsy-evaluation.md").read_text(encoding="utf-8")
    required_values = (
        f"{result.twin_summary.rank_correlation:.3f}",
        f"{result.method_summaries[0].p95_ttft_ms.median:.3f}",
    )
    if not all(value in fabric_report for value in required_values):
        raise RuntimeError("Fabric report is not derived from result.json")
    if f"{result.diagnosis_summary.top_one_accuracy:.3f}" not in autopsy_report:
        raise RuntimeError("Autopsy report is not derived from result.json")
    manifest = json.loads((artifact_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("result_sha256") != sha256_file(result_path):
        raise RuntimeError("evaluation manifest does not match result.json")
    return result


def run_fabric_evaluation(
    *,
    repository_root: Path,
    artifact_root: Path,
    report_dir: Path,
    config: EvaluationConfig | None = None,
    reset: bool = False,
) -> FabricEvaluationResult:
    active = config or EvaluationConfig()
    if artifact_root.exists() and reset:
        if artifact_root.resolve() in {Path("/").resolve(), Path.home().resolve()}:
            raise ValueError("refusing to reset a broad artifact directory")
        shutil.rmtree(artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    environment_path = artifact_root / "environment.json"
    hardware_path = artifact_root / "hardware-manifest.json"
    commands_path = artifact_root / "commands.txt"
    write_json(environment_path, environment_manifest())
    write_json(
        hardware_path,
        {
            "validation_mode": "synthetic_cpu",
            "nvidia_smi_available": shutil.which("nvidia-smi") is not None,
            "gpu_measurements_executed": False,
            "reason": "evaluation command is CPU-safe and does not claim fixture curves as hardware",
        },
    )
    command = (
        "python",
        "-m",
        "sloforge.fabric.evaluation",
        "--artifact-dir",
        str(artifact_root),
        "--report-dir",
        str(report_dir),
    )
    commands_path.write_text(
        "PYTHONPATH=python " + " ".join(command) + " --reset\n", encoding="utf-8"
    )
    environment_digest = ArtifactDigest(value=sha256_file(environment_path))
    plan_trials: list[PlanTrial] = []
    diagnosis_trials: list[DiagnosisTrial] = []
    recovery_trials: list[RecoveryTrial] = []
    raw_artifacts: list[tuple[Path, ArtifactProvenance]] = [
        (environment_path, "environment"),
        (hardware_path, "environment"),
        (commands_path, "environment"),
    ]
    for topology_name in active.topology_fixtures:
        topology = _topology(topology_name)
        for seed in active.seeds:
            case_dir = artifact_root / "trials" / topology_name / f"seed-{seed}"
            raw_profile = benchmark_synthetic_fabric(
                topology,
                seed=seed,
                suite="quick",
                warmup_count=2,
                sample_count=5,
                output_dir=case_dir / "fabric-profile-raw",
            )
            raw_artifacts.extend(
                (path, "synthetic")
                for path in sorted((case_dir / "fabric-profile-raw").rglob("*.json"))
            )
            profile = to_canonical_profile(raw_profile, topology=topology)
            topology_path = case_dir / "topology.json"
            profile_path = case_dir / "fabric-profile.json"
            write_json(topology_path, topology.model_dump(mode="json"))
            write_json(profile_path, profile.model_dump(mode="json"))
            raw_artifacts.extend(((topology_path, "synthetic"), (profile_path, "synthetic")))
            workload = _workload(seed, active.request_count)
            workload_path = case_dir / "workload.json"
            write_json(workload_path, workload.model_dump(mode="json"))
            raw_artifacts.append((workload_path, "synthetic"))
            hierarchy_context: (
                tuple[
                    PhysicalExecutionPlan,
                    FabricSimulationRequest,
                    FabricSimulationOutput,
                    Path,
                ]
                | None
            ) = None
            for method in active.methods:
                compile_result = compile_physical_plan(
                    _compile_request(
                        repository_root=repository_root,
                        topology=topology,
                        profile=profile,
                        method=method,
                        seed=seed,
                        environment_digest=environment_digest,
                    )
                )
                simulation_request = build_simulation_request(
                    compile_result.selected,
                    topology,
                    profile,
                    workload,
                    seed=seed,
                )
                output = run_simulation(
                    simulation_request,
                    repository_root=repository_root,
                    timeout_seconds=active.simulator_timeout_seconds,
                )
                isolated_request = build_simulation_request(
                    compile_result.selected,
                    topology,
                    profile,
                    _isolated_service_workload(),
                    seed=seed,
                )
                isolated_output = run_simulation(
                    isolated_request,
                    repository_root=repository_root,
                    timeout_seconds=active.simulator_timeout_seconds,
                )
                method_dir = case_dir / method.value
                compiler_path = method_dir / "compiler.json"
                simulation_path = method_dir / "simulation.json"
                isolated_simulation_path = method_dir / "isolated-service-simulation.json"
                write_json(compiler_path, compile_result.model_dump(mode="json"))
                _write_simulation(simulation_path, output)
                _write_simulation(isolated_simulation_path, isolated_output)
                raw_artifacts.extend(
                    (
                        (compiler_path, "compiler"),
                        (simulation_path, "simulator"),
                        (isolated_simulation_path, "simulator"),
                    )
                )
                plan_trials.append(
                    _plan_trial(
                        seed=seed,
                        topology_name=topology_name,
                        method=method,
                        compilation=compile_result,
                        output=output,
                        isolated_output=isolated_output,
                        workload=workload,
                        config=active,
                        simulation_path=simulation_path.relative_to(artifact_root),
                        isolated_simulation_path=isolated_simulation_path.relative_to(
                            artifact_root
                        ),
                        compiler_path=compiler_path.relative_to(artifact_root),
                    )
                )
                if method is EvaluationMethod.HIERARCHICAL:
                    hierarchy_context = (
                        compile_result.selected,
                        simulation_request,
                        output,
                        simulation_path,
                    )
            if hierarchy_context is None:
                # H3/H4 require the primary compiler even if a caller requests
                # only baseline methods for a focused H1/H2 test.
                compile_result = compile_physical_plan(
                    _compile_request(
                        repository_root=repository_root,
                        topology=topology,
                        profile=profile,
                        method=EvaluationMethod.HIERARCHICAL,
                        seed=seed,
                        environment_digest=environment_digest,
                    )
                )
                request = build_simulation_request(
                    compile_result.selected, topology, profile, workload, seed=seed
                )
                output = run_simulation(
                    request,
                    repository_root=repository_root,
                    timeout_seconds=active.simulator_timeout_seconds,
                )
                simulation_path = case_dir / "hierarchical-autopsy" / "simulation.json"
                _write_simulation(simulation_path, output)
                raw_artifacts.append((simulation_path, "simulator"))
                hierarchy_context = (compile_result.selected, request, output, simulation_path)
            plan, request, healthy_output, healthy_path = hierarchy_context
            diagnoses, recoveries, autopsy_paths = _diagnosis_and_recovery(
                repository_root=repository_root,
                artifact_root=artifact_root,
                seed=seed,
                topology_name=topology_name,
                topology=topology,
                profile=profile,
                plan=plan,
                workload=workload,
                healthy_request=request,
                healthy_output=healthy_output,
                healthy_path=healthy_path,
                config=active,
            )
            diagnosis_trials.extend(diagnoses)
            recovery_trials.extend(recoveries)
            raw_artifacts.extend((path, "autopsy") for path in autopsy_paths)
    plans = tuple(plan_trials)
    twin_trials, twin_summary = _twin_results(plans)
    result = FabricEvaluationResult(
        generated_at=datetime.now(UTC).isoformat(),
        git_commit=git_commit(repository_root),
        command=command,
        hardware_backed_validation="not_exercised_no_compatible_hardware",
        config=active,
        plan_trials=plans,
        method_summaries=_method_summaries(plans, active),
        twin_trials=twin_trials,
        twin_summary=twin_summary,
        diagnosis_trials=tuple(diagnosis_trials),
        diagnosis_summary=_diagnosis_summary(tuple(diagnosis_trials)),
        recovery_trials=tuple(recovery_trials),
        recovery_summaries=_recovery_summaries(tuple(recovery_trials)),
        limitations=(
            "Synthetic H100, NVLink, PCIe, and InfiniBand curves are deterministic fixtures, not measurements.",
            "Compiler analytical predictions and the Rust simulator share calibration inputs; H2 is internal validation.",
            "Recovery action durations are declared policy parameters; only post-action request metrics are simulated.",
            "No GPU, NCCL, RDMA, multi-node runtime, or privileged fault was exercised by this evaluation.",
        ),
        artifacts=tuple(
            _artifact(path, artifact_root, provenance)
            for path, provenance in sorted(raw_artifacts, key=lambda item: str(item[0]))
        ),
    )
    result_path = artifact_root / "result.json"
    write_json(result_path, result.model_dump(mode="json"))
    report_paths = _write_reports(result, report_dir=report_dir)
    manifest_path = artifact_root / "manifest.json"
    write_json(
        manifest_path,
        {
            "schema_version": "sloforge.fabric.evaluation-manifest/v1",
            "result_sha256": sha256_file(result_path),
            "reports": [{"path": str(path), "sha256": sha256_file(path)} for path in report_paths],
            "artifact_count": len(result.artifacts),
        },
    )
    validate_evaluation_artifacts(artifact_root=artifact_root, report_dir=report_dir)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/fabric/evaluation"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--seeds", default="13,29,47")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    result = run_fabric_evaluation(
        repository_root=Path(__file__).resolve().parents[3],
        artifact_root=args.artifact_dir,
        report_dir=args.report_dir,
        config=EvaluationConfig(seeds=seeds),
        reset=args.reset,
    )
    print(
        json.dumps(
            {
                "result": str(args.artifact_dir / "result.json"),
                "trials": len(result.plan_trials),
                "top1_diagnosis_accuracy": result.diagnosis_summary.top_one_accuracy,
                "validation_mode": result.validation_mode,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
