"""Strict, versioned models for reproducible ForgeCI experiments."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StringConstraints, model_validator

FORGECI_SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveFloat = Annotated[float, Field(gt=0.0)]


class ForgeModel(BaseModel):
    """Base model that refuses ambiguous or silently ignored fields."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class MetricDirection(StrEnum):
    LOWER_IS_BETTER = "lower_is_better"
    HIGHER_IS_BETTER = "higher_is_better"


class ComparisonClassification(StrEnum):
    REGRESSION = "regression"
    IMPROVEMENT = "improvement"
    UNCHANGED = "unchanged"
    INCONCLUSIVE = "inconclusive"
    FLAKY = "flaky"
    FAILED = "failed"


class TrialStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


class CommandSpec(ForgeModel):
    executable: NonEmpty
    arguments: tuple[str, ...] = ()
    timeout_seconds: PositiveFloat = 60.0


class EnvironmentVariable(ForgeModel):
    name: Annotated[str, StringConstraints(pattern=r"^[A-Z_][A-Z0-9_]*$")]
    value: str
    sensitive: bool = False


class EnvironmentSpec(ForgeModel):
    python_version: NonEmpty | None = None
    cuda_version: NonEmpty | None = None
    pytorch_version: NonEmpty | None = None
    communication_library_version: NonEmpty | None = None
    container_image: NonEmpty | None = None
    variables: tuple[EnvironmentVariable, ...] = ()

    @model_validator(mode="after")
    def unique_variables(self) -> Self:
        names = [entry.name for entry in self.variables]
        if len(names) != len(set(names)):
            raise ValueError("environment variable names must be unique")
        return self


class HardwareRequirement(ForgeModel):
    architecture: NonEmpty
    minimum_cpu_cores: PositiveInt = 1
    minimum_memory_gib: PositiveFloat = 1.0
    gpu_count: NonNegativeInt = 0
    gpu_model: NonEmpty | None = None
    minimum_gpu_memory_gib: PositiveFloat | None = None
    requires_rdma: bool = False
    topology_fingerprint: NonEmpty | None = None

    @model_validator(mode="after")
    def validate_gpu_fields(self) -> Self:
        if self.gpu_count == 0 and (self.gpu_model or self.minimum_gpu_memory_gib):
            raise ValueError("GPU constraints require gpu_count greater than zero")
        return self


class MetricSpec(ForgeModel):
    name: Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_.-]*$")]
    unit: NonEmpty
    direction: MetricDirection
    practical_threshold_percent: Annotated[float, Field(ge=0.0)] = 5.0
    significance_level: Annotated[float, Field(gt=0.0, lt=0.5)] = 0.05
    maximum_noise_percent: Annotated[float, Field(gt=0.0)] = 10.0


class BenchmarkInput(ForgeModel):
    model: NonEmpty = "synthetic"
    workload: NonEmpty = "fixture"
    physical_plan: NonEmpty | None = None
    prompt_tokens: PositiveInt = 128
    output_tokens: PositiveInt = 32
    concurrency: PositiveInt = 1
    runtime_arguments: tuple[str, ...] = ()


class BenchmarkSpec(ForgeModel):
    command: CommandSpec
    metrics: tuple[MetricSpec, ...]
    input: BenchmarkInput = Field(default_factory=BenchmarkInput)
    warmup_trials: NonNegativeInt = 2
    repetitions: Annotated[int, Field(ge=3)] = 7
    maximum_repetitions: Annotated[int, Field(ge=3)] = 35
    failed_trial_retries: Annotated[int, Field(ge=0, le=5)] = 1
    bootstrap_rounds: Annotated[int, Field(ge=200)] = 1_000
    seed: int = 17

    @model_validator(mode="after")
    def validate_benchmark(self) -> Self:
        if not self.metrics:
            raise ValueError("benchmark must define at least one metric")
        names = [metric.name for metric in self.metrics]
        if len(names) != len(set(names)):
            raise ValueError("metric names must be unique")
        if self.maximum_repetitions < self.repetitions:
            raise ValueError("maximum_repetitions cannot be less than repetitions")
        return self


class MatrixCase(ForgeModel):
    case_id: Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9-]*$")]
    repository: NonEmpty
    revision: NonEmpty
    build: tuple[CommandSpec, ...] = ()
    environment: EnvironmentSpec = Field(default_factory=EnvironmentSpec)
    hardware: HardwareRequirement
    benchmark: BenchmarkSpec
    budget_seconds: PositiveFloat = 600.0


class BenchmarkMatrix(ForgeModel):
    schema_version: Literal["1.0.0"] = FORGECI_SCHEMA_VERSION
    matrix_id: NonEmpty
    cases: tuple[MatrixCase, ...]

    @model_validator(mode="after")
    def validate_cases(self) -> Self:
        if not self.cases:
            raise ValueError("benchmark matrix cannot be empty")
        identifiers = [case.case_id for case in self.cases]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("matrix case identifiers must be unique")
        return self


class MetricValue(ForgeModel):
    name: NonEmpty
    value: float


class TrialRecord(ForgeModel):
    trial_index: NonNegativeInt
    attempt: NonNegativeInt
    phase: Literal["warmup", "measurement"]
    seed: int
    status: TrialStatus
    duration_seconds: Annotated[float, Field(ge=0.0)]
    metrics: tuple[MetricValue, ...] = ()
    stdout_path: NonEmpty
    stderr_path: NonEmpty
    exit_code: int | None = None
    error: str | None = None


class MetricSummary(ForgeModel):
    name: NonEmpty
    unit: NonEmpty
    sample_count: PositiveInt
    median: float
    median_absolute_deviation: Annotated[float, Field(ge=0.0)]
    p95: float
    confidence_level: Annotated[float, Field(gt=0.0, lt=1.0)]
    median_ci_low: float
    median_ci_high: float
    raw_samples: tuple[float, ...]

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        if not self.median_ci_low <= self.median <= self.median_ci_high:
            raise ValueError("median confidence interval must contain the median")
        if self.p95 < min(self.raw_samples):
            raise ValueError("p95 is outside the sample range")
        return self


class EnvironmentManifest(ForgeModel):
    platform: NonEmpty
    python_version: NonEmpty
    machine: NonEmpty
    processor: str
    git_commit: NonEmpty
    environment: tuple[EnvironmentVariable, ...]
    hardware_fingerprint: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class HardwareObservation(ForgeModel):
    architecture: NonEmpty
    cpu_cores: PositiveInt
    memory_gib: PositiveFloat | None
    gpu_count: NonNegativeInt
    gpu_models: tuple[NonEmpty, ...]
    gpu_memory_gib: tuple[PositiveFloat, ...]
    rdma_available: bool
    topology_fingerprint: NonEmpty | None
    fingerprint: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class RunRecord(ForgeModel):
    schema_version: Literal["1.0.0"] = FORGECI_SCHEMA_VERSION
    run_id: NonEmpty
    case_id: NonEmpty
    revision: NonEmpty
    started_at: AwareDatetime
    completed_at: AwareDatetime
    success: bool
    warmups: tuple[TrialRecord, ...]
    trials: tuple[TrialRecord, ...]
    summaries: tuple[MetricSummary, ...]
    environment_manifest: EnvironmentManifest
    artifact_directory: NonEmpty
    artifact_hash: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    warnings: tuple[str, ...] = ()


class MatrixRunRecord(ForgeModel):
    schema_version: Literal["1.0.0"] = FORGECI_SCHEMA_VERSION
    matrix_id: NonEmpty
    started_at: AwareDatetime
    completed_at: AwareDatetime
    success: bool
    runs: tuple[RunRecord, ...]
    artifact_directory: NonEmpty
    warnings: tuple[str, ...] = ()


class MetricComparison(ForgeModel):
    metric: NonEmpty
    unit: NonEmpty
    baseline_median: float
    candidate_median: float
    degradation_percent: float
    degradation_ci_low_percent: float
    degradation_ci_high_percent: float
    cliffs_delta: Annotated[float, Field(ge=-1.0, le=1.0)]
    noise_floor_percent: Annotated[float, Field(ge=0.0)]
    corrected_significance_level: Annotated[float, Field(gt=0.0, lt=0.5)]
    classification: ComparisonClassification
    rationale: NonEmpty


class ComparisonRecord(ForgeModel):
    schema_version: Literal["1.0.0"] = FORGECI_SCHEMA_VERSION
    comparison_id: NonEmpty
    baseline_run_id: NonEmpty
    candidate_run_id: NonEmpty
    classification: ComparisonClassification
    metrics: tuple[MetricComparison, ...]
    correction_method: Literal["bonferroni"] = "bonferroni"
    warnings: tuple[str, ...] = ()


class BisectStep(ForgeModel):
    commit: NonEmpty
    classification: ComparisonClassification
    comparison_artifact: NonEmpty
    repetitions: PositiveInt
    attempt: NonNegativeInt


class BisectResult(ForgeModel):
    schema_version: Literal["1.0.0"] = FORGECI_SCHEMA_VERSION
    repository: NonEmpty
    good_commit: NonEmpty
    bad_commit: NonEmpty
    first_regressing_commit: NonEmpty | None
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    steps: tuple[BisectStep, ...]
    inconclusive_commits: tuple[str, ...]
    artifact_directory: NonEmpty
    caveats: tuple[str, ...] = ()


class MinimizationStep(ForgeModel):
    field: NonEmpty
    before: NonEmpty
    after: NonEmpty
    preserved_regression: bool


class MinimalReproducer(ForgeModel):
    schema_version: Literal["1.0.0"] = FORGECI_SCHEMA_VERSION
    good_commit: NonEmpty
    bad_commit: NonEmpty
    benchmark: BenchmarkSpec
    steps: tuple[MinimizationStep, ...]
    reproduction_commands: tuple[NonEmpty, ...]
    expected_regression: NonEmpty
    confidence_interval: NonEmpty
    hardware: HardwareRequirement
    artifact_references: tuple[NonEmpty, ...]


class ForgeCIEvaluation(ForgeModel):
    schema_version: Literal["1.0.0"] = FORGECI_SCHEMA_VERSION
    expected_first_regressing_commit: NonEmpty
    identified_first_regressing_commit: NonEmpty | None
    bisection_correct: bool
    commits_in_range: PositiveInt
    unique_commits_evaluated: PositiveInt
    inconclusive_rate: Annotated[float, Field(ge=0.0, le=1.0)]
    comparison: ComparisonRecord
    bisection: BisectResult
    reproducer: MinimalReproducer
    report_path: NonEmpty
    issue_path: NonEmpty
