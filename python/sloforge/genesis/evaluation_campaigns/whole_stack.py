"""Artifact-backed H2/H9 whole-stack evaluation for the local HybridDecoder scope."""

from __future__ import annotations

import hashlib
import random
import statistics
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from sloforge.fabric.ir import load_physical_execution_plan
from sloforge.genesis.compiler import initialize_genesis_run
from sloforge.genesis.distributed_synthesis import OverlapMutation, compile_distributed_mutation
from sloforge.genesis.frontend import inspect_reference_package
from sloforge.genesis.ir import (
    CandidateSuccessState,
    StateLayout,
    TransformationFamily,
    canonical_json,
    load_candidate,
    load_inference_genome,
    load_transformation,
)
from sloforge.genesis.policy_dsl import execute_bytecode, load_bytecode_document
from sloforge.genesis.synthesis import synthesize_local_run
from sloforge.genesis.tensor_rewrites import (
    BUILTIN_RULES,
    DType,
    OperatorParameters,
    TensorGraph,
    TensorNode,
    TensorSpec,
    explore_rewrites,
)

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
NonNegativeInt = Annotated[int, Field(ge=0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
VariantName: TypeAlias = Literal[
    "configuration_only", "policy_only", "state_only", "genesis_two_layer"
]


class WholeStackValidationError(ValueError):
    """Persisted H2/H9 evidence does not independently reproduce."""


class _Model(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
        allow_inf_nan=False,
    )


class WorkloadRequest(_Model):
    request_id: NonEmpty
    arrival_units: NonNegativeFloat
    service_units: Annotated[float, Field(gt=0.0)]
    output_tokens: Annotated[int, Field(gt=0)]
    deadline_units: Annotated[float, Field(gt=0.0)]
    priority: Annotated[int, Field(ge=1, le=4)]


class RequestObservation(_Model):
    request_id: NonEmpty
    batch_ordinal: NonNegativeInt
    completion_units: Annotated[float, Field(gt=0.0)]
    ttft_units: Annotated[float, Field(gt=0.0)]
    token_latency_units: Annotated[float, Field(gt=0.0)]
    tardiness_units: NonNegativeFloat


class VariantObservation(_Model):
    variant: VariantName
    policy_enabled: bool
    paged_state_enabled: bool
    objective_units: Annotated[float, Field(gt=0.0)]
    p95_ttft_units: Annotated[float, Field(gt=0.0)]
    median_token_latency_units: Annotated[float, Field(gt=0.0)]
    state_reservation_bytes: Annotated[int, Field(gt=0)]
    requests: tuple[RequestObservation, ...]


class CategoryEvidence(_Model):
    policy: Literal["accepted_runtime_replayed"]
    state: Literal["accepted_runtime_replayed"]
    distributed: Literal["statically_valid_pending_revalidation"]
    tensor_kernel: Literal["exact_rewrite_verified_not_lowered"]
    distributed_candidate_hash: Sha256
    distributed_performance_comparison_eligible: Literal[False]
    tensor_source_nodes: Literal[2]
    tensor_target_nodes: Literal[1]
    tensor_rule_id: Literal["tensor/redundant-cast/v1"]


class WholeStackSeedResult(_Model):
    seed: NonNegativeInt
    accepted_candidate_id: NonEmpty
    accepted_candidate_hash: Sha256
    candidate_artifact_path: NonEmpty
    candidate_artifact_sha256: Sha256
    genome_artifact_path: NonEmpty
    genome_artifact_sha256: Sha256
    policy_artifact_path: NonEmpty
    policy_artifact_sha256: Sha256
    transformation_families: tuple[TransformationFamily, ...]
    affected_genome_regions: tuple[NonEmpty, ...]
    category_evidence: CategoryEvidence
    workload: tuple[WorkloadRequest, ...]
    variants: tuple[VariantObservation, ...]

    @model_validator(mode="after")
    def complete_result(self) -> WholeStackSeedResult:
        if self.transformation_families != (
            TransformationFamily.BATCHING,
            TransformationFamily.STATE_LAYOUT,
        ):
            raise ValueError("accepted result must contain the trusted policy/state chain")
        if self.affected_genome_regions != ("request", "serving", "state"):
            raise ValueError("accepted result does not span request, serving, and state")
        if tuple(item.variant for item in self.variants) != (
            "configuration_only",
            "policy_only",
            "state_only",
            "genesis_two_layer",
        ):
            raise ValueError("whole-stack variants are incomplete or unordered")
        return self


class PairedEffect(_Model):
    comparison: Literal["configuration_only_minus_genesis", "best_single_layer_minus_genesis"]
    per_seed_differences: tuple[float, ...]
    mean_difference: float
    confidence_low: float
    confidence_high: float
    confidence_level: Annotated[float, Field(ge=0.95, le=0.95)] = 0.95
    bootstrap_resamples: Literal[4000] = 4000


class WholeStackCampaignReport(_Model):
    schema_version: Literal["sloforge.genesis.whole-stack-campaign/v1"] = (
        "sloforge.genesis.whole-stack-campaign/v1"
    )
    hypothesis_ids: Literal["H2,H9"] = "H2,H9"
    seeds: tuple[NonNegativeInt, ...]
    synthesis_seed: Literal[73129] = 73129
    scope: Literal["deterministic_cpu_service_model"] = "deterministic_cpu_service_model"
    hardware_backed_runs: Literal[0] = 0
    raw_results_path: NonEmpty
    raw_results_sha256: Sha256
    fabric_fixture_path: NonEmpty
    fabric_fixture_sha256: Sha256
    results: tuple[WholeStackSeedResult, ...]
    h2_effect: PairedEffect
    h9_effect: PairedEffect
    h2_conclusion: Literal["supported_in_declared_synthetic_scope", "not_supported"]
    h9_conclusion: Literal["supported_in_declared_synthetic_scope", "not_supported"]
    limitations: tuple[NonEmpty, ...]

    @model_validator(mode="after")
    def seeds_are_bound(self) -> WholeStackCampaignReport:
        if len(self.seeds) < 3 or len(self.seeds) != len(set(self.seeds)):
            raise ValueError("whole-stack campaign requires at least three unique seeds")
        if tuple(item.seed for item in self.results) != self.seeds:
            raise ValueError("whole-stack results do not match campaign seeds")
        return self


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite campaign artifact: {path}")
    path.write_bytes(payload)


def _workload(seed: int, *, count: int = 32) -> tuple[WorkloadRequest, ...]:
    generator = random.Random(seed)
    requests: list[WorkloadRequest] = []
    for index in range(count):
        burst = index // 8
        arrival = burst * 2.4 + generator.randrange(0, 4) * 0.05
        short = generator.random() < 0.62
        service = generator.uniform(0.8, 1.7) if short else generator.uniform(5.5, 8.5)
        output_tokens = generator.randint(1, 4) if short else generator.randint(8, 16)
        priority = generator.randint(3, 4) if short else generator.randint(1, 2)
        slack = generator.uniform(1.8, 4.0) if short else generator.uniform(15.0, 25.0)
        requests.append(
            WorkloadRequest(
                request_id=f"request-{seed}-{index:03d}",
                arrival_units=round(arrival, 6),
                service_units=round(service, 6),
                output_tokens=output_tokens,
                deadline_units=round(arrival + slack, 6),
                priority=priority,
            )
        )
    return tuple(requests)


def _percentile(values: tuple[float, ...], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def _simulate(
    workload: tuple[WorkloadRequest, ...],
    *,
    variant: VariantName,
    policy_path: Path,
) -> VariantObservation:
    policy_enabled = variant in {"policy_only", "genesis_two_layer"}
    paged = variant in {"state_only", "genesis_two_layer"}
    policy = load_bytecode_document(policy_path.read_bytes()) if policy_enabled else None
    pending = list(workload)
    now = min(item.arrival_units for item in workload)
    observations: list[RequestObservation] = []
    batch_ordinal = 0
    while pending:
        ready = [item for item in pending if item.arrival_units <= now + 1e-12]
        if not ready:
            now = min(item.arrival_units for item in pending)
            ready = [item for item in pending if item.arrival_units <= now + 1e-12]
        if policy is None:
            ordered = sorted(ready, key=lambda item: (item.arrival_units, item.request_id))
            batch_limit = min(4, len(ordered))
        else:
            ordered = sorted(
                ready,
                key=lambda item: (item.deadline_units, -item.priority, item.request_id),
            )
            first = ordered[0]
            values: dict[str, int | bool] = {
                "queue_length": min(32, len(ordered)),
                "slo_slack_ms": max(0, min(1000, round((first.deadline_units - now) * 1000.0))),
                "cancellation_pending": False,
            }
            names = {item.name for item in policy.inputs}
            decision = execute_bytecode(policy, {name: values[name] for name in names})
            if type(decision) is not int or not 1 <= decision <= 4:
                raise WholeStackValidationError("accepted policy emitted an invalid batch bound")
            batch_limit = min(decision, len(ordered))
        batch = ordered[:batch_limit]
        page_overhead = 0.001 * len(batch) if paged else 0.0
        duration = (
            0.05
            + max(item.service_units for item in batch) * 0.1 * (0.7 + 0.075 * len(batch))
            + page_overhead
        )
        completed_at = now + duration
        for item in batch:
            ttft = (now - item.arrival_units) + min(0.5, item.service_units / 4.0)
            token_latency = duration / item.output_tokens
            observations.append(
                RequestObservation(
                    request_id=item.request_id,
                    batch_ordinal=batch_ordinal,
                    completion_units=round(completed_at, 9),
                    ttft_units=round(ttft, 9),
                    token_latency_units=round(token_latency, 9),
                    tardiness_units=round(max(0.0, completed_at - item.deadline_units), 9),
                )
            )
            pending.remove(item)
        now = completed_at
        batch_ordinal += 1
    by_id = {item.request_id: item for item in workload}
    weighted_tardiness = sum(
        observation.tardiness_units * by_id[observation.request_id].priority
        for observation in observations
    )
    mean_completion = statistics.fmean(
        observation.completion_units - by_id[observation.request_id].arrival_units
        for observation in observations
    )
    objective = weighted_tardiness + mean_completion
    reservation = 80 if paged else 73
    return VariantObservation(
        variant=variant,
        policy_enabled=policy_enabled,
        paged_state_enabled=paged,
        objective_units=round(objective, 9),
        p95_ttft_units=round(_percentile(tuple(item.ttft_units for item in observations), 0.95), 9),
        median_token_latency_units=round(
            statistics.median(item.token_latency_units for item in observations), 9
        ),
        state_reservation_bytes=reservation,
        requests=tuple(sorted(observations, key=lambda item: item.request_id)),
    )


def _tensor_evidence() -> tuple[int, int, str]:
    specification = TensorSpec(shape=("tokens",), dtype=DType.INT32)
    graph = TensorGraph(
        nodes=(
            TensorNode("input", "input", (), specification),
            TensorNode(
                "redundant-cast",
                "cast",
                ("input",),
                specification,
                parameters=OperatorParameters(target_dtype=DType.INT32),
            ),
        ),
        outputs=("redundant-cast",),
    )
    candidates = explore_rewrites(
        graph, BUILTIN_RULES, quality_budget=0.0, maximum_candidates=8, maximum_depth=1
    )
    rewritten = next(
        candidate
        for candidate in candidates
        if candidate.history and candidate.history[-1].rule_id == "tensor/redundant-cast/v1"
    )
    return len(graph.nodes), len(rewritten.graph.nodes), rewritten.history[-1].rule_id


def _category_evidence(fabric_fixture: Path, *, seed: int) -> CategoryEvidence:
    source = load_physical_execution_plan(fabric_fixture)
    distributed = compile_distributed_mutation(
        source,
        OverlapMutation(
            transformation_id=f"whole-stack-overlap-{seed}",
            window_id="overlap-0",
            expected_overlap_fraction=0.75,
            stream="genesis-comm",
            fallback_serialization="critical_path",
        ),
        seed=seed,
    )
    source_nodes, target_nodes, rule_id = _tensor_evidence()
    if (source_nodes, target_nodes, rule_id) != (
        2,
        1,
        "tensor/redundant-cast/v1",
    ):
        raise WholeStackValidationError("exact tensor rewrite did not produce the declared result")
    return CategoryEvidence(
        policy="accepted_runtime_replayed",
        state="accepted_runtime_replayed",
        distributed="statically_valid_pending_revalidation",
        tensor_kernel="exact_rewrite_verified_not_lowered",
        distributed_candidate_hash=distributed.candidate_plan_hash,
        distributed_performance_comparison_eligible=False,
        tensor_source_nodes=2,
        tensor_target_nodes=1,
        tensor_rule_id="tensor/redundant-cast/v1",
    )


def _paired_effect(
    differences: tuple[float, ...],
    *,
    comparison: Literal["configuration_only_minus_genesis", "best_single_layer_minus_genesis"],
    seed: int,
) -> PairedEffect:
    generator = random.Random(seed)
    means = sorted(
        statistics.fmean(generator.choice(differences) for _ in differences) for _ in range(4000)
    )
    return PairedEffect(
        comparison=comparison,
        per_seed_differences=differences,
        mean_difference=round(statistics.fmean(differences), 9),
        confidence_low=round(means[int(0.025 * len(means))], 9),
        confidence_high=round(means[int(0.975 * len(means)) - 1], 9),
    )


def _effects(results: tuple[WholeStackSeedResult, ...]) -> tuple[PairedEffect, PairedEffect]:
    h2: list[float] = []
    h9: list[float] = []
    for result in results:
        by_variant = {item.variant: item.objective_units for item in result.variants}
        full = by_variant["genesis_two_layer"]
        h2.append(round(by_variant["configuration_only"] - full, 9))
        best_single = min(by_variant["policy_only"], by_variant["state_only"])
        h9.append(round(best_single - full, 9))
    return (
        _paired_effect(tuple(h2), comparison="configuration_only_minus_genesis", seed=918273),
        _paired_effect(tuple(h9), comparison="best_single_layer_minus_genesis", seed=918274),
    )


def run_whole_stack_campaign(
    output_root: Path,
    *,
    seeds: tuple[int, ...],
    reference_package: Path,
    fabric_fixture: Path,
) -> WholeStackCampaignReport:
    """Run real compiler/verifier surfaces and a scoped deterministic service model."""

    if len(seeds) < 3 or len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        raise ValueError("campaign requires at least three unique non-negative seeds")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("whole-stack campaign output must be empty")
    output_root.mkdir(parents=True, exist_ok=True)
    fabric_copy = output_root / "inputs/physical-execution-plan-v1.json"
    _write_once(fabric_copy, fabric_fixture.read_bytes())
    inspection = inspect_reference_package(reference_package)
    run = initialize_genesis_run(reference_package, inspection, output_root / "run", seed=73129)
    synthesis = synthesize_local_run(run.output_directory, seed=73129)
    if synthesis.accepted_candidate_id is None:
        raise RuntimeError("whole-stack campaign synthesis produced no accepted candidate")
    candidate_root = run.output_directory / "candidates" / synthesis.accepted_candidate_id
    candidate_path = candidate_root / "candidate.json"
    genome_path = candidate_root / "inference_genome.json"
    policy_path = candidate_root / "policy.bytecode.json"
    candidate = load_candidate(candidate_path)
    genome = load_inference_genome(genome_path)
    if candidate.state is not CandidateSuccessState.SIMULATED:
        raise RuntimeError("campaign candidate did not reach the simulated acceptance state")
    transformations = tuple(
        load_transformation(candidate_root / "transformations" / f"{identifier}.json")
        for identifier in candidate.transformation_ids
    )
    families = tuple(item.family for item in transformations)
    regions = tuple(
        sorted({region for item in transformations for region in item.affected_regions})
    )
    if {state.layout for state in genome.state.states} != {StateLayout.PAGED}:
        raise RuntimeError("accepted whole-stack candidate did not lower paged state")
    results: list[WholeStackSeedResult] = []
    for seed in seeds:
        workload = _workload(seed)
        variant_names: tuple[VariantName, ...] = (
            "configuration_only",
            "policy_only",
            "state_only",
            "genesis_two_layer",
        )
        variants = tuple(
            _simulate(workload, variant=variant, policy_path=policy_path)
            for variant in variant_names
        )
        results.append(
            WholeStackSeedResult(
                seed=seed,
                accepted_candidate_id=candidate.candidate_id,
                accepted_candidate_hash=candidate.genome_hash.value,
                candidate_artifact_path=str(candidate_path.resolve()),
                candidate_artifact_sha256=_sha256(candidate_path),
                genome_artifact_path=str(genome_path.resolve()),
                genome_artifact_sha256=_sha256(genome_path),
                policy_artifact_path=str(policy_path.resolve()),
                policy_artifact_sha256=_sha256(policy_path),
                transformation_families=families,
                affected_genome_regions=regions,
                category_evidence=_category_evidence(fabric_copy, seed=seed),
                workload=workload,
                variants=variants,
            )
        )
    result_tuple = tuple(results)
    raw_path = output_root / "raw/results.jsonl"
    raw_payload = b"".join(canonical_json(item) + b"\n" for item in result_tuple)
    _write_once(raw_path, raw_payload)
    h2, h9 = _effects(result_tuple)
    report = WholeStackCampaignReport(
        seeds=seeds,
        raw_results_path=str(raw_path.resolve()),
        raw_results_sha256=_sha256(raw_path),
        fabric_fixture_path=str(fabric_copy.resolve()),
        fabric_fixture_sha256=_sha256(fabric_copy),
        results=result_tuple,
        h2_effect=h2,
        h9_effect=h9,
        h2_conclusion=(
            "supported_in_declared_synthetic_scope" if h2.confidence_low > 0 else "not_supported"
        ),
        h9_conclusion=(
            "supported_in_declared_synthetic_scope" if h9.confidence_low > 0 else "not_supported"
        ),
        limitations=(
            "service time is a deterministic model unit, not a wall-clock or hardware measurement",
            "the Fabric mutation is deliberately ineligible until its complete revalidation pipeline runs",
            "the exact tensor rewrite is verified but not lowered into the flagship generated runtime",
            "the paged allocator trades bounded fragmentation for transition-friendly state units",
        ),
    )
    report_path = output_root / "report.json"
    _write_once(report_path, canonical_json(report) + b"\n")
    validate_whole_stack_campaign(report)
    return report


def validate_whole_stack_campaign(report: WholeStackCampaignReport | Path) -> None:
    """Reopen every raw artifact and independently reproduce all scoped conclusions."""

    value = (
        WholeStackCampaignReport.model_validate_json(report.read_bytes(), strict=True)
        if isinstance(report, Path)
        else report
    )
    raw_path = Path(value.raw_results_path)
    fabric_path = Path(value.fabric_fixture_path)
    if _sha256(raw_path) != value.raw_results_sha256:
        raise WholeStackValidationError("raw whole-stack results digest mismatch")
    if _sha256(fabric_path) != value.fabric_fixture_sha256:
        raise WholeStackValidationError("Fabric fixture digest mismatch")
    try:
        reopened = tuple(
            WholeStackSeedResult.model_validate_json(line, strict=True)
            for line in raw_path.read_text(encoding="utf-8").splitlines()
            if line
        )
    except ValueError as error:
        raise WholeStackValidationError("raw whole-stack result is invalid") from error
    if reopened != value.results:
        raise WholeStackValidationError("report results differ from canonical raw results")
    for result in reopened:
        artifact_pairs = (
            (result.candidate_artifact_path, result.candidate_artifact_sha256),
            (result.genome_artifact_path, result.genome_artifact_sha256),
            (result.policy_artifact_path, result.policy_artifact_sha256),
        )
        if any(_sha256(Path(path)) != digest for path, digest in artifact_pairs):
            raise WholeStackValidationError("accepted candidate artifact digest mismatch")
        candidate = load_candidate(Path(result.candidate_artifact_path))
        genome = load_inference_genome(Path(result.genome_artifact_path))
        if candidate.genome_hash.value != result.accepted_candidate_hash:
            raise WholeStackValidationError("accepted candidate hash mismatch")
        if {state.layout for state in genome.state.states} != {StateLayout.PAGED}:
            raise WholeStackValidationError("accepted candidate state layout changed")
        reproduced = tuple(
            _simulate(
                result.workload,
                variant=item.variant,
                policy_path=Path(result.policy_artifact_path),
            )
            for item in result.variants
        )
        if reproduced != result.variants:
            raise WholeStackValidationError("variant observations do not replay")
        if _category_evidence(fabric_path, seed=result.seed) != result.category_evidence:
            raise WholeStackValidationError("category evidence does not reproduce")
    h2, h9 = _effects(reopened)
    if h2 != value.h2_effect or h9 != value.h9_effect:
        raise WholeStackValidationError("paired effects do not reproduce")
    expected_h2 = (
        "supported_in_declared_synthetic_scope" if h2.confidence_low > 0 else "not_supported"
    )
    expected_h9 = (
        "supported_in_declared_synthetic_scope" if h9.confidence_low > 0 else "not_supported"
    )
    if value.h2_conclusion != expected_h2 or value.h9_conclusion != expected_h9:
        raise WholeStackValidationError("hypothesis conclusion differs from paired evidence")


__all__ = [
    "WholeStackCampaignReport",
    "WholeStackValidationError",
    "run_whole_stack_campaign",
    "validate_whole_stack_campaign",
]
