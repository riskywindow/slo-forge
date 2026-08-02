"""Typed task grammar, raw evidence, and reports for ServingSynthBench."""

from __future__ import annotations

import math
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4096)]
Identifier = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_.-]*$", max_length=128)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0, le=(1 << 64) - 1)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]


class SynthBenchModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class BlockKind(StrEnum):
    DENSE_ATTENTION = "dense_attention"
    SLIDING_WINDOW_ATTENTION = "sliding_window_attention"
    GROUPED_QUERY_ATTENTION = "grouped_query_attention"
    GATED_MLP = "gated_mlp"
    SPARSE_MOE = "sparse_moe"
    STATE_SPACE = "state_space"
    RECURRENT = "recurrent_state"
    CONVOLUTIONAL_STATE = "convolutional_state"
    CUSTOM_NORMALIZATION = "custom_normalization"
    QUANTIZED_STATE = "quantized_state_transformation"
    RESIDUAL_BRANCH = "residual_branch"
    SPECULATIVE_HEAD = "speculative_head"
    CUSTOM_SAMPLER = "custom_sampler"
    CROSS_ATTENTION = "cross_attention"


class BlockSpec(SynthBenchModel):
    block_id: Identifier
    kind: BlockKind
    hidden_size: PositiveInt
    window_size: PositiveInt = 4
    expert_count: PositiveInt = 2
    top_k: PositiveInt = 1
    group_count: PositiveInt = 1
    kernel_size: PositiveInt = 3
    quantization_bits: Literal[4, 8] = 8
    state_size: PositiveInt = 4

    @model_validator(mode="after")
    def valid_parameters(self) -> Self:
        if self.top_k > self.expert_count:
            raise ValueError("MoE top_k cannot exceed expert_count")
        if self.hidden_size % self.group_count != 0:
            raise ValueError("hidden_size must be divisible by group_count")
        if self.kernel_size % 2 == 0:
            raise ValueError("convolution kernel_size must be odd")
        return self


class ArchitectureSpec(SynthBenchModel):
    architecture_id: Identifier
    seed: NonNegativeInt
    vocabulary_size: Annotated[int, Field(ge=16, le=256)]
    hidden_size: Annotated[int, Field(ge=8, le=256)]
    maximum_sequence_length: Annotated[int, Field(ge=8, le=512)]
    blocks: tuple[BlockSpec, ...]

    @model_validator(mode="after")
    def valid_architecture(self) -> Self:
        if not 2 <= len(self.blocks) <= 32:
            raise ValueError("architecture must contain between 2 and 32 blocks")
        identifiers = [block.block_id for block in self.blocks]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("architecture block identifiers must be unique")
        if any(block.hidden_size != self.hidden_size for block in self.blocks):
            raise ValueError("all affordable grammar blocks must share hidden_size")
        return self


class WorkloadRequest(SynthBenchModel):
    request_id: Identifier
    prompt_tokens: tuple[NonNegativeInt, ...]
    maximum_new_tokens: Annotated[int, Field(ge=1, le=64)]
    seed: NonNegativeInt
    arrival_offset_ms: NonNegativeFloat
    priority: Annotated[int, Field(ge=0, le=7)]
    deadline_ms: Annotated[float, Field(gt=0.0)]
    expected_tokens: tuple[NonNegativeInt, ...]

    @model_validator(mode="after")
    def aligned_output(self) -> Self:
        if not self.prompt_tokens:
            raise ValueError("workload prompt cannot be empty")
        if len(self.prompt_tokens) > 512:
            raise ValueError("workload prompt exceeds the task-grammar bound")
        if len(self.expected_tokens) != self.maximum_new_tokens:
            raise ValueError("expected output count must match maximum_new_tokens")
        return self


class HiddenCase(SynthBenchModel):
    case_id: Identifier
    request: WorkloadRequest
    trap: Literal[
        "minimum_shape",
        "maximum_shape",
        "rare_length",
        "state_reset",
        "sampler_tie",
        "burst_priority",
    ]


class TaskDescriptor(SynthBenchModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    task_id: Identifier
    seed: NonNegativeInt
    architecture: ArchitectureSpec
    public_package_path: NonEmpty
    workload_path: NonEmpty
    hidden_cases_path: NonEmpty
    hidden_commitment: Sha256
    public_package_hash: Sha256

    @model_validator(mode="after")
    def paths_are_task_relative(self) -> Self:
        for value in (
            self.public_package_path,
            self.workload_path,
            self.hidden_cases_path,
        ):
            path = PurePosixPath(value)
            if (
                "\\" in value
                or path.is_absolute()
                or not path.parts
                or path.as_posix() != value
                or any(part in {"", ".", ".."} for part in path.parts)
            ):
                raise ValueError("task artifact paths must be normalized task-relative paths")
        return self


class GrammarConfiguration(SynthBenchModel):
    seed: NonNegativeInt
    count: PositiveInt
    minimum_blocks: Annotated[int, Field(ge=2, le=16)] = 4
    maximum_blocks: Annotated[int, Field(ge=2, le=32)] = 8
    public_cases_per_task: Annotated[int, Field(ge=2, le=64)] = 5
    hidden_cases_per_task: Annotated[int, Field(ge=3, le=64)] = 6

    @model_validator(mode="after")
    def ordered_blocks(self) -> Self:
        if self.maximum_blocks < self.minimum_blocks:
            raise ValueError("maximum_blocks cannot be below minimum_blocks")
        return self


class SpecialCaseFinding(SynthBenchModel):
    finding_id: Identifier
    severity: Literal["warning", "reject"]
    category: Literal["task_identity", "seed_identity", "hidden_commitment", "hidden_shape"]
    message: NonEmpty
    line: PositiveInt


class SpecialCaseAudit(SynthBenchModel):
    passed: bool
    findings: tuple[SpecialCaseFinding, ...]


class BaselineKind(StrEnum):
    PYTHON_EAGER_REFERENCE = "python_eager_reference"
    PYTORCH_EAGER_REFERENCE = "pytorch_eager_reference"
    TORCH_COMPILE = "torch_compile"
    GENERIC_RUNTIME = "generic_runtime_adapter"
    TUNED_STATIC_SLOFORGE = "tuned_static_sloforge"
    PHYSICAL_PLAN_ONLY = "physical_plan_only_sloforge"
    POLICY_ONLY = "policy_only_search"
    KERNEL_ONLY = "kernel_only_search"
    WITHOUT_AUTOPSY = "genesis_without_autopsy"
    WITHOUT_COUNTEREXAMPLE_LEARNING = "genesis_without_counterexample_learning"
    WITHOUT_LINEAGE = "genesis_without_lineage"
    GENESIS_FULL = "genesis_full"
    DETERMINISTIC_SINGLE_SHOT = "deterministic_single_shot"


class BaselineStatus(StrEnum):
    MEASURED = "measured"
    SURROGATE = "surrogate"
    FAILED = "failed"
    NOT_APPLICABLE = "not_applicable"


class RawCpuSample(SynthBenchModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    task_id: Identifier
    task_hash: Sha256
    workload_fingerprint: Sha256
    environment_fingerprint: Sha256
    baseline: BaselineKind
    run_seed: NonNegativeInt
    repetition: NonNegativeInt
    measurement_order_ordinal: NonNegativeInt
    execution_ordinal: NonNegativeInt
    request_id: Identifier
    request_seed: NonNegativeInt
    latency_ns: PositiveInt
    ttft_ns: PositiveInt
    inter_token_ns: tuple[PositiveInt, ...]
    observed_tokens: tuple[NonNegativeInt, ...]
    expected_tokens: tuple[NonNegativeInt, ...]
    exact_match: bool
    execution_surface: Literal[
        "python_eager_reference",
        "reference_order_surrogate",
        "genesis_generated_runtime",
    ]
    execution_evidence_path: NonEmpty | None = None
    execution_evidence_sha256: Sha256 | None = None
    source: Literal["measured_cpu_monotonic_clock", "replayed_cpu_reference_observation"] = (
        "measured_cpu_monotonic_clock"
    )
    precision: Literal["python_float64_reference"] = "python_float64_reference"

    @model_validator(mode="after")
    def generated_execution_has_bound_evidence(self) -> Self:
        if self.execution_evidence_path is None or self.execution_evidence_sha256 is None:
            raise ValueError("every executed CPU sample must bind sandbox execution evidence")
        return self


class BaselineSummary(SynthBenchModel):
    baseline: BaselineKind
    status: BaselineStatus
    reason: NonEmpty
    raw_samples_path: NonEmpty | None
    raw_samples_sha256: Sha256 | None
    execution_surface: Literal[
        "not_applicable",
        "python_eager_reference",
        "reference_order_surrogate",
        "genesis_generated_runtime",
    ]
    sample_count: NonNegativeInt
    valid_request_rate: Annotated[float, Field(ge=0.0, le=1.0)]
    median_latency_ns: NonNegativeFloat
    p95_latency_ns: NonNegativeFloat
    median_ttft_ns: NonNegativeFloat
    median_inter_token_ns: NonNegativeFloat
    observed_request_wall_seconds: NonNegativeFloat
    candidate_count: NonNegativeInt
    human_authored_model_specific_lines: NonNegativeInt

    @model_validator(mode="after")
    def evidence_matches_status(self) -> Self:
        executed = self.status in {BaselineStatus.MEASURED, BaselineStatus.SURROGATE}
        if executed != (self.raw_samples_path is not None):
            raise ValueError("executed baseline status and raw sample path must agree")
        if executed != (self.raw_samples_sha256 is not None):
            raise ValueError("executed baseline status and raw sample digest must agree")
        if executed == (self.execution_surface == "not_applicable"):
            raise ValueError("baseline execution surface does not match its status")
        return self


class IntegrityReport(SynthBenchModel):
    passed: bool
    violations: tuple[NonEmpty, ...]
    workload_fingerprint: Sha256
    environment_fingerprint: Sha256
    expected_repetitions: PositiveInt
    expected_request_ids: tuple[Identifier, ...]


class CpuRunConfiguration(SynthBenchModel):
    seeds: tuple[NonNegativeInt, ...]
    warmup_count: NonNegativeInt = 1
    repetitions: Annotated[int, Field(ge=2, le=100)] = 3
    maximum_tasks: PositiveInt
    maximum_runtime_seconds: Annotated[float, Field(gt=0.0)] = 120.0

    @model_validator(mode="after")
    def distinct_seeds(self) -> Self:
        if len(self.seeds) < 2 or len(self.seeds) != len(set(self.seeds)):
            raise ValueError("CPU runner requires at least two distinct seeds")
        return self


class HiddenCaseResult(SynthBenchModel):
    case_id: Identifier
    baseline: BaselineKind
    request_id: Identifier
    request_seed: NonNegativeInt
    observed_tokens: tuple[NonNegativeInt, ...]
    expected_tokens: tuple[NonNegativeInt, ...]
    exact_match: bool
    execution_surface: Literal[
        "python_eager_reference",
        "reference_order_surrogate",
        "genesis_generated_runtime",
    ]
    execution_evidence_path: NonEmpty | None = None
    execution_evidence_sha256: Sha256 | None = None
    source: Literal["evaluator_only_cpu_execution"] = "evaluator_only_cpu_execution"

    @model_validator(mode="after")
    def generated_execution_has_bound_evidence(self) -> Self:
        if self.execution_evidence_path is None or self.execution_evidence_sha256 is None:
            raise ValueError("every hidden CPU result must bind sandbox execution evidence")
        return self


class HiddenEvaluationSummary(SynthBenchModel):
    baseline: BaselineKind
    status: BaselineStatus
    evidence_path: NonEmpty | None
    evidence_sha256: Sha256 | None
    case_count: NonNegativeInt
    exact_case_rate: Annotated[float, Field(ge=0.0, le=1.0)]
    escaped_regressions: NonNegativeInt

    @model_validator(mode="after")
    def evidence_matches_status(self) -> Self:
        executed = self.status in {BaselineStatus.MEASURED, BaselineStatus.SURROGATE}
        if executed != (self.evidence_path is not None):
            raise ValueError("executed hidden status and evidence path must agree")
        if executed != (self.evidence_sha256 is not None):
            raise ValueError("executed hidden status and evidence digest must agree")
        return self


class TaskRunReport(SynthBenchModel):
    task_id: Identifier
    task_hash: Sha256
    task_descriptor_path: NonEmpty
    task_descriptor_sha256: Sha256
    public_package_hash: Sha256
    reference_package_hash: Sha256
    task_generation_seed: NonNegativeInt
    run_seed: NonNegativeInt
    warmup_count: NonNegativeInt
    repetitions: PositiveInt
    measurement_run_order: tuple[NonEmpty, ...]
    baselines: tuple[BaselineSummary, ...]
    hidden_evaluations: tuple[HiddenEvaluationSummary, ...]
    integrity: IntegrityReport
    genesis_runtime_manifest_path: NonEmpty
    genesis_runtime_manifest_sha256: Sha256


class AggregateMetrics(SynthBenchModel):
    measured_baseline_runs: NonNegativeInt
    surrogate_baseline_runs: NonNegativeInt
    unavailable_baseline_runs: NonNegativeInt
    valid_system_rate: Annotated[float, Field(ge=0.0, le=1.0)]
    exact_request_rate: Annotated[float, Field(ge=0.0, le=1.0)]
    observed_request_wall_seconds: NonNegativeFloat
    task_count: NonNegativeInt
    distinct_task_seeds: NonNegativeInt


class SynthBenchReport(SynthBenchModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    run_seeds: tuple[NonNegativeInt, ...]
    tasks: tuple[TaskRunReport, ...]
    metrics: AggregateMetrics
    report_source: Literal["derived_from_raw_samples"] = "derived_from_raw_samples"

    @model_validator(mode="after")
    def finite_metrics(self) -> Self:
        values = (
            self.metrics.valid_system_rate,
            self.metrics.exact_request_rate,
            self.metrics.observed_request_wall_seconds,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("aggregate metrics must be finite")
        if len(self.run_seeds) < 2 or len(self.run_seeds) != len(set(self.run_seeds)):
            raise ValueError("report requires at least two distinct run seeds")
        if self.metrics.task_count <= 0 or not self.tasks:
            raise ValueError("report must contain executed tasks")
        return self
