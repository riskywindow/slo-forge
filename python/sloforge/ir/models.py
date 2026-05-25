"""Canonical, versioned SLOForge deployment intermediate representation.

Core objects intentionally reject unknown fields.  The sole escape hatch is an
``Extensions`` map whose keys are namespace-qualified, making ownership and
compatibility explicit.
"""

from __future__ import annotations

import math
import re
from enum import StrEnum
from typing import Annotated, Final, Literal, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    RootModel,
    StringConstraints,
    field_validator,
    model_validator,
)

SCHEMA_VERSION: Final = "1.0.0"
API_VERSION: Final = "sloforge.io/v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EXTENSION_KEY_PATTERN = r"^[a-z][a-z0-9]*(?:[.-][a-z0-9]+)*/[A-Za-z][A-Za-z0-9_.-]*$"
_EXTENSION_KEY = re.compile(_EXTENSION_KEY_PATTERN)

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
SemVerString = Annotated[
    str,
    StringConstraints(
        pattern=(
            r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
            r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
            r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
        )
    ),
]
ExtensionKey = Annotated[str, StringConstraints(pattern=_EXTENSION_KEY_PATTERN)]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]
PositiveFloat = Annotated[float, Field(gt=0.0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]


class IRModel(BaseModel):
    """Strict base for every object in the canonical IR."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )


def _validate_json(value: JsonValue, path: str = "extensions") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} contains a non-finite number")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json(item, f"{path}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            _validate_json(item, f"{path}.{key}")


class Extensions(RootModel[dict[ExtensionKey, JsonValue]]):
    """Validated vendor extension fields serialized as a plain JSON object."""

    model_config = ConfigDict(
        frozen=True,
        strict=True,
        json_schema_extra={"additionalProperties": False},
    )

    @model_validator(mode="after")
    def validate_entries(self) -> Self:
        for key, value in self.root.items():
            if _EXTENSION_KEY.fullmatch(key) is None:
                raise ValueError(
                    f"extension key {key!r} must be namespace-qualified (for example acme.io/key)"
                )
            _validate_json(value, f"extensions.{key}")
        return self


class ArtifactDigest(IRModel):
    algorithm: Literal["sha256"] = "sha256"
    value: str

    @field_validator("value")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError("sha256 digest must be 64 lowercase hexadecimal characters")
        return value


class DocumentMetadata(IRModel):
    name: NonEmptyString
    uid: NonEmptyString
    generation: PositiveInt = 1
    created_at: AwareDatetime
    labels: dict[str, str] = Field(default_factory=dict)


class LicenseMetadata(IRModel):
    spdx_id: NonEmptyString
    name: NonEmptyString
    url: str | None = None
    redistribution_allowed: bool
    verified_at: AwareDatetime | None = None


class ModelArchitecture(IRModel):
    family: NonEmptyString
    parameter_count: PositiveInt
    hidden_size: PositiveInt
    num_layers: PositiveInt
    num_attention_heads: PositiveInt
    num_key_value_heads: PositiveInt
    vocabulary_size: PositiveInt

    @model_validator(mode="after")
    def validate_heads(self) -> Self:
        if self.num_key_value_heads > self.num_attention_heads:
            raise ValueError("num_key_value_heads cannot exceed num_attention_heads")
        return self


class DType(StrEnum):
    FLOAT32 = "float32"
    TF32 = "tf32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FP8 = "fp8"
    INT8 = "int8"
    INT4 = "int4"


class Quantization(StrEnum):
    NONE = "none"
    AWQ = "awq"
    GPTQ = "gptq"
    BITSANDBYTES = "bitsandbytes"
    FP8 = "fp8"


class LoraSpec(IRModel):
    adapter_id: NonEmptyString
    revision: NonEmptyString
    checksum: ArtifactDigest
    rank: PositiveInt
    merged: bool = False


class ModelSpec(IRModel):
    model_id: NonEmptyString
    revision: NonEmptyString
    checksum: ArtifactDigest
    tokenizer_id: NonEmptyString
    tokenizer_revision: NonEmptyString
    architecture: ModelArchitecture
    allowed_precisions: tuple[DType, ...]
    minimum_precision: DType
    maximum_sequence_length: PositiveInt
    lora: tuple[LoraSpec, ...] = ()
    license: LicenseMetadata
    extensions: Extensions = Field(default_factory=lambda: Extensions(root={}))

    @model_validator(mode="after")
    def validate_precision(self) -> Self:
        if not self.allowed_precisions:
            raise ValueError("allowed_precisions must not be empty")
        if self.minimum_precision not in self.allowed_precisions:
            raise ValueError("minimum_precision must be included in allowed_precisions")
        if len(set(self.allowed_precisions)) != len(self.allowed_precisions):
            raise ValueError("allowed_precisions cannot contain duplicates")
        return self


class ChunkedPrefillSpec(IRModel):
    enabled: bool = False
    chunk_tokens: PositiveInt = 512


class PrefixCacheSpec(IRModel):
    enabled: bool = False
    capacity_tokens: NonNegativeInt = 0
    eviction_policy: Literal["lru", "clock"] = "lru"


class SpeculativeDecodingSpec(IRModel):
    enabled: bool = False
    draft_model_id: NonEmptyString | None = None
    maximum_draft_tokens: PositiveInt = 4

    @model_validator(mode="after")
    def validate_draft_model(self) -> Self:
        if self.enabled and self.draft_model_id is None:
            raise ValueError("enabled speculative decoding requires draft_model_id")
        return self


class CompilationSpec(IRModel):
    mode: Literal["eager", "compile", "ahead_of_time"] = "eager"
    cuda_graphs: bool = False
    graph_batch_sizes: tuple[PositiveInt, ...] = ()

    @model_validator(mode="after")
    def validate_graph_sizes(self) -> Self:
        if not self.cuda_graphs and self.graph_batch_sizes:
            raise ValueError("graph_batch_sizes require cuda_graphs")
        return self


class EngineSpec(IRModel):
    runtime: Literal["transformers", "vllm", "sglang", "tensorrt_llm", "mock"]
    version: SemVerString
    dtype: DType
    quantization: Quantization = Quantization.NONE
    tensor_parallelism: PositiveInt = 1
    pipeline_parallelism: PositiveInt = 1
    maximum_batched_tokens: PositiveInt
    maximum_active_sequences: PositiveInt
    chunked_prefill: ChunkedPrefillSpec = Field(default_factory=ChunkedPrefillSpec)
    prefix_cache: PrefixCacheSpec = Field(default_factory=PrefixCacheSpec)
    speculative_decoding: SpeculativeDecodingSpec = Field(default_factory=SpeculativeDecodingSpec)
    compilation: CompilationSpec = Field(default_factory=CompilationSpec)
    extensions: Extensions = Field(default_factory=lambda: Extensions(root={}))


class CpuSpec(IRModel):
    architecture: NonEmptyString
    model: NonEmptyString
    physical_cores: PositiveInt
    logical_cores: PositiveInt
    numa_nodes: PositiveInt
    measured_memory_bandwidth_gbps: PositiveFloat | None = None

    @model_validator(mode="after")
    def validate_cores(self) -> Self:
        if self.logical_cores < self.physical_cores:
            raise ValueError("logical_cores cannot be less than physical_cores")
        return self


class GpuSpec(IRModel):
    index: NonNegativeInt
    product: NonEmptyString
    architecture: NonEmptyString
    uuid: NonEmptyString
    vram_bytes: PositiveInt
    memory_clock_mhz: PositiveFloat | None = None
    measured_memory_bandwidth_gbps: PositiveFloat | None = None
    measured_compute_tflops: PositiveFloat | None = None
    pcie_generation: PositiveInt | None = None
    pcie_width: PositiveInt | None = None
    ecc_enabled: bool | None = None


class TopologyLink(IRModel):
    source_gpu: NonNegativeInt
    target_gpu: NonNegativeInt
    kind: Literal["pcie", "nvlink", "shared_memory"]
    measured_bandwidth_gbps: PositiveFloat | None = None
    measured_latency_us: PositiveFloat | None = None

    @model_validator(mode="after")
    def validate_endpoints(self) -> Self:
        if self.source_gpu == self.target_gpu:
            raise ValueError("topology link endpoints must differ")
        return self


class HardwareSpec(IRModel):
    fingerprint: ArtifactDigest
    cpu: CpuSpec
    system_memory_bytes: PositiveInt
    gpu_count: NonNegativeInt = 0
    gpus: tuple[GpuSpec, ...] = ()
    topology: tuple[TopologyLink, ...] = ()
    driver_version: str | None = None
    cuda_version: str | None = None
    library_versions: dict[str, str] = Field(default_factory=dict)
    hourly_price_usd: NonNegativeFloat
    region: NonEmptyString
    cloud: NonEmptyString | None = None
    instance_type: NonEmptyString | None = None
    container_memory_limit_bytes: PositiveInt | None = None
    extensions: Extensions = Field(default_factory=lambda: Extensions(root={}))

    @model_validator(mode="after")
    def validate_topology(self) -> Self:
        if self.gpu_count != len(self.gpus):
            raise ValueError("gpu_count must equal the number of GPU specifications")
        indices = [gpu.index for gpu in self.gpus]
        if len(indices) != len(set(indices)):
            raise ValueError("GPU indices must be unique")
        index_set = set(indices)
        for link in self.topology:
            if link.source_gpu not in index_set or link.target_gpu not in index_set:
                raise ValueError("topology link references an unknown GPU")
        return self


class ArrivalKind(StrEnum):
    POISSON = "poisson"
    DETERMINISTIC = "deterministic"
    TRACE = "trace"
    BURSTY = "bursty"


class ArrivalProcess(IRModel):
    kind: ArrivalKind
    rate_per_second: PositiveFloat | None = None
    burst_factor: PositiveFloat | None = None
    trace_uri: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_parameters(self) -> Self:
        if self.kind is ArrivalKind.TRACE:
            if self.trace_uri is None:
                raise ValueError("trace arrivals require trace_uri")
        elif self.rate_per_second is None:
            raise ValueError(f"{self.kind.value} arrivals require rate_per_second")
        if self.kind is ArrivalKind.BURSTY and self.burst_factor is None:
            raise ValueError("bursty arrivals require burst_factor")
        return self


class DistributionKind(StrEnum):
    FIXED = "fixed"
    EMPIRICAL = "empirical"
    LOGNORMAL = "lognormal"


class WeightedValue(IRModel):
    value: NonNegativeInt
    weight: PositiveFloat


class DistributionSpec(IRModel):
    kind: DistributionKind
    fixed_value: NonNegativeInt | None = None
    empirical: tuple[WeightedValue, ...] = ()
    log_mean: float | None = None
    log_stddev: PositiveFloat | None = None
    minimum: NonNegativeInt = 0
    maximum: PositiveInt

    @model_validator(mode="after")
    def validate_distribution(self) -> Self:
        if self.minimum > self.maximum:
            raise ValueError("distribution minimum cannot exceed maximum")
        if self.kind is DistributionKind.FIXED:
            if self.fixed_value is None or not self.minimum <= self.fixed_value <= self.maximum:
                raise ValueError("fixed distribution value must be within bounds")
        elif self.kind is DistributionKind.EMPIRICAL:
            if not self.empirical:
                raise ValueError("empirical distribution requires weighted values")
            if any(not self.minimum <= item.value <= self.maximum for item in self.empirical):
                raise ValueError("empirical value is outside distribution bounds")
        elif self.log_mean is None or self.log_stddev is None:
            raise ValueError("lognormal distribution requires log_mean and log_stddev")
        return self


class Priority(StrEnum):
    CRITICAL = "critical"
    INTERACTIVE = "interactive"
    BATCH = "batch"


class RequestClass(IRModel):
    name: NonEmptyString
    weight: PositiveFloat
    priority: Priority
    tenant: NonEmptyString | None = None
    deadline_ms: PositiveFloat | None = None
    adapter_ids: tuple[str, ...] = ()
    prefix_group: str | None = None


class WorkloadSpec(IRModel):
    arrival_process: ArrivalProcess
    prompt_tokens: DistributionSpec
    output_tokens: DistributionSpec
    request_classes: tuple[RequestClass, ...]
    duration_seconds: PositiveFloat
    seed: NonNegativeInt
    trace_digest: ArtifactDigest | None = None
    extensions: Extensions = Field(default_factory=lambda: Extensions(root={}))

    @model_validator(mode="after")
    def validate_classes(self) -> Self:
        if not self.request_classes:
            raise ValueError("request_classes must not be empty")
        names = [request_class.name for request_class in self.request_classes]
        if len(names) != len(set(names)):
            raise ValueError("request class names must be unique")
        if abs(sum(request_class.weight for request_class in self.request_classes) - 1.0) > 1e-6:
            raise ValueError("request class weights must sum to one")
        return self


class MetricConstraint(IRModel):
    percentile: Annotated[float, Field(gt=0.0, le=100.0)]
    maximum_ms: PositiveFloat


class FluidityConstraint(IRModel):
    token_deadline_ms: PositiveFloat
    maximum_missed_fraction: Probability


class ObjectiveWeights(IRModel):
    cost: NonNegativeFloat = 1.0
    latency: NonNegativeFloat = 0.0
    goodput: NonNegativeFloat = 0.0
    availability: NonNegativeFloat = 0.0

    @model_validator(mode="after")
    def validate_nonzero(self) -> Self:
        if self.cost + self.latency + self.goodput + self.availability <= 0:
            raise ValueError("at least one objective weight must be positive")
        return self


class SLOSpec(IRModel):
    ttft: tuple[MetricConstraint, ...] = ()
    inter_token_latency: tuple[MetricConstraint, ...] = ()
    end_to_end_latency: tuple[MetricConstraint, ...] = ()
    fluidity: FluidityConstraint | None = None
    minimum_goodput_rps: PositiveFloat | None = None
    minimum_availability: Probability | None = None
    maximum_cost_per_million_tokens_usd: PositiveFloat | None = None
    objective_weights: ObjectiveWeights = Field(default_factory=ObjectiveWeights)

    @model_validator(mode="after")
    def validate_percentiles(self) -> Self:
        for name, constraints in (
            ("ttft", self.ttft),
            ("inter_token_latency", self.inter_token_latency),
            ("end_to_end_latency", self.end_to_end_latency),
        ):
            percentiles = [item.percentile for item in constraints]
            if len(percentiles) != len(set(percentiles)):
                raise ValueError(f"{name} has duplicate percentile constraints")
        return self


class BudgetSpec(IRModel):
    profiling_budget_usd: NonNegativeFloat
    profiling_duration_seconds: PositiveFloat | None = None
    maximum_real_trials: PositiveInt | None = None
    reserve_fraction: Probability = 0.15


class ReplicaTopology(IRModel):
    minimum_replicas: PositiveInt
    maximum_replicas: PositiveInt
    initial_replicas: PositiveInt
    tensor_parallelism_per_replica: PositiveInt = 1
    regions: tuple[NonEmptyString, ...]

    @model_validator(mode="after")
    def validate_replicas(self) -> Self:
        if not self.minimum_replicas <= self.initial_replicas <= self.maximum_replicas:
            raise ValueError("initial_replicas must be between minimum and maximum")
        if not self.regions:
            raise ValueError("regions must not be empty")
        return self


class RoutingPolicyKind(StrEnum):
    ROUND_ROBIN = "round_robin"
    LEAST_OUTSTANDING = "least_outstanding"
    EARLIEST_FINISH = "earliest_finish"
    SLO_SLACK = "slo_slack"


class RouteTarget(IRModel):
    variant: NonEmptyString
    weight: Annotated[float, Field(gt=0.0, le=1.0)]


class RoutingPolicy(IRModel):
    kind: RoutingPolicyKind
    targets: tuple[RouteTarget, ...]
    health_penalty_ms: NonNegativeFloat = 1000.0
    cold_start_penalty_ms: NonNegativeFloat = 0.0

    @model_validator(mode="after")
    def validate_weights(self) -> Self:
        if not self.targets:
            raise ValueError("routing targets must not be empty")
        variants = [target.variant for target in self.targets]
        if len(variants) != len(set(variants)):
            raise ValueError("routing target variants must be unique")
        if abs(sum(target.weight for target in self.targets) - 1.0) > 1e-6:
            raise ValueError("routing target weights must sum to one")
        return self


class AdmissionPolicy(IRModel):
    queue_capacity: PositiveInt
    maximum_queue_time_ms: PositiveFloat
    shed_below_priority: Priority | None = None
    reject_when_predicted_late: bool = True


class BatchingPolicy(IRModel):
    maximum_active_sequences: PositiveInt
    maximum_batched_tokens: PositiveInt
    maximum_batch_delay_ms: NonNegativeFloat
    dynamic_batching: bool = True


class AutoscalingPolicy(IRModel):
    mode: Literal["disabled", "reactive", "predictive"]
    target_utilization: Annotated[float, Field(gt=0.0, le=1.0)]
    control_interval_seconds: PositiveFloat
    scale_up_cooldown_seconds: NonNegativeFloat
    scale_down_cooldown_seconds: NonNegativeFloat
    minimum_samples: PositiveInt
    safety_margin: Annotated[float, Field(ge=0.0, lt=1.0)]
    maximum_change_per_interval: PositiveInt


class ColdStartStrategy(IRModel):
    minimum_warm_replicas: NonNegativeInt
    prefetch_model: bool = True
    readiness_timeout_seconds: PositiveFloat
    predicted_p95_startup_ms: PositiveFloat


class CanaryPolicy(IRModel):
    enabled: bool = True
    initial_weight: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.05
    minimum_requests: PositiveInt = 100
    observation_seconds: PositiveFloat = 60.0
    maximum_slo_violation_delta: Probability = 0.01


class RollbackPolicy(IRModel):
    enabled: bool = True
    violation_windows: PositiveInt = 2
    window_seconds: PositiveFloat = 30.0
    availability_floor: Probability = 0.99
    cooldown_seconds: NonNegativeFloat = 120.0


class MetricEstimate(IRModel):
    point: float
    lower: float
    upper: float
    confidence: Annotated[float, Field(gt=0.0, lt=1.0)]
    unit: NonEmptyString
    sample_count: NonNegativeInt
    measurement_ids: tuple[NonEmptyString, ...]

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        if not self.lower <= self.point <= self.upper:
            raise ValueError("metric estimate point must lie within its interval")
        if not all(math.isfinite(value) for value in (self.point, self.lower, self.upper)):
            raise ValueError("metric estimate values must be finite")
        if self.sample_count > 0 and not self.measurement_ids:
            raise ValueError("measured estimates require measurement_ids")
        return self


class Provenance(IRModel):
    profile_id: NonEmptyString
    optimizer_run_id: NonEmptyString
    workload_digest: ArtifactDigest
    hardware_fingerprint: ArtifactDigest
    evidence_bundle_uri: NonEmptyString
    compiler_version: SemVerString
    git_commit: NonEmptyString


class DeploymentPlan(IRModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    api_version: Literal["sloforge.io/v1"] = API_VERSION
    kind: Literal["DeploymentPlan"] = "DeploymentPlan"
    metadata: DocumentMetadata
    model: ModelSpec
    engine: EngineSpec
    hardware: HardwareSpec
    workload: WorkloadSpec
    slo: SLOSpec
    budget: BudgetSpec
    replica_topology: ReplicaTopology
    routing: RoutingPolicy
    admission: AdmissionPolicy
    batching: BatchingPolicy
    autoscaling: AutoscalingPolicy
    cold_start: ColdStartStrategy
    canary: CanaryPolicy
    rollback: RollbackPolicy
    predicted_metrics: dict[str, MetricEstimate]
    provenance: Provenance
    extensions: Extensions = Field(default_factory=lambda: Extensions(root={}))

    @model_validator(mode="after")
    def validate_cross_fields(self) -> Self:
        if self.engine.dtype not in self.model.allowed_precisions:
            raise ValueError("engine dtype is forbidden by the model precision constraints")
        if self.engine.maximum_active_sequences != self.batching.maximum_active_sequences:
            raise ValueError("engine and batching maximum_active_sequences must match")
        if self.engine.maximum_batched_tokens != self.batching.maximum_batched_tokens:
            raise ValueError("engine and batching maximum_batched_tokens must match")
        if self.replica_topology.tensor_parallelism_per_replica != self.engine.tensor_parallelism:
            raise ValueError("replica and engine tensor parallelism must match")
        if self.cold_start.minimum_warm_replicas > self.replica_topology.maximum_replicas:
            raise ValueError("minimum warm replicas exceeds maximum replicas")
        return self


class EnvironmentManifest(IRModel):
    os: NonEmptyString
    kernel: NonEmptyString
    architecture: NonEmptyString
    hostname_hash: ArtifactDigest
    container_image: NonEmptyString | None = None
    python_version: NonEmptyString | None = None
    rust_version: NonEmptyString | None = None
    package_versions: dict[str, str]
    environment_allowlist: dict[str, str] = Field(default_factory=dict)


class MeasurementRef(IRModel):
    measurement_id: NonEmptyString
    kind: Literal["hardware", "startup", "prefill", "decode", "load", "fault"]
    uri: NonEmptyString
    digest: ArtifactDigest
    sample_count: NonNegativeInt
    warmup_count: NonNegativeInt
    started_at: AwareDatetime
    completed_at: AwareDatetime
    hardware_fingerprint: ArtifactDigest

    @model_validator(mode="after")
    def validate_times(self) -> Self:
        if self.completed_at < self.started_at:
            raise ValueError("measurement completed_at precedes started_at")
        return self


class CalibrationMetric(IRModel):
    model_name: NonEmptyString
    split: Literal["train", "validation", "test"]
    metric: Literal["mae", "mape", "rmse", "coverage", "interval_width"]
    value: NonNegativeFloat
    sample_count: PositiveInt

    @model_validator(mode="after")
    def validate_metric_range(self) -> Self:
        if self.metric == "coverage" and self.value > 1.0:
            raise ValueError("coverage cannot exceed one")
        return self


class OptimizerDecision(IRModel):
    sequence: NonNegativeInt
    candidate_id: NonEmptyString
    fidelity: Literal["static", "simulated", "measured"]
    decision: Literal["evaluate", "promote", "reject", "select"]
    reason_code: NonEmptyString
    predicted_objective: MetricEstimate | None = None
    cost_usd: NonNegativeFloat


class RejectedCandidate(IRModel):
    candidate_id: NonEmptyString
    stage: Literal["feasibility", "simulation", "measurement", "selection"]
    reason_code: NonEmptyString
    explanation: NonEmptyString
    violated_constraints: tuple[NonEmptyString, ...] = ()


class BenchmarkResult(IRModel):
    benchmark_id: NonEmptyString
    command: tuple[NonEmptyString, ...]
    raw_result_uri: NonEmptyString
    raw_result_digest: ArtifactDigest
    seed: NonNegativeInt
    started_at: AwareDatetime
    completed_at: AwareDatetime
    metrics: dict[str, MetricEstimate]

    @model_validator(mode="after")
    def validate_benchmark(self) -> Self:
        if not self.command:
            raise ValueError("benchmark command must not be empty")
        if self.completed_at < self.started_at:
            raise ValueError("benchmark completed_at precedes started_at")
        return self


class EvidenceBundle(IRModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    api_version: Literal["sloforge.io/v1"] = API_VERSION
    kind: Literal["EvidenceBundle"] = "EvidenceBundle"
    metadata: DocumentMetadata
    plan_digest: ArtifactDigest
    environment: EnvironmentManifest
    model_assumptions: tuple[NonEmptyString, ...]
    measurements: tuple[MeasurementRef, ...]
    calibration_metrics: tuple[CalibrationMetric, ...]
    optimizer_history: tuple[OptimizerDecision, ...]
    rejected_candidates: tuple[RejectedCandidate, ...]
    benchmark_results: tuple[BenchmarkResult, ...]
    artifact_hashes: dict[NonEmptyString, ArtifactDigest]
    git_commit: NonEmptyString
    generated_at: AwareDatetime
    extensions: Extensions = Field(default_factory=lambda: Extensions(root={}))

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        measurement_ids = [item.measurement_id for item in self.measurements]
        if len(measurement_ids) != len(set(measurement_ids)):
            raise ValueError("measurement IDs must be unique")
        decision_sequences = [item.sequence for item in self.optimizer_history]
        if decision_sequences != sorted(decision_sequences):
            raise ValueError("optimizer decisions must be ordered by sequence")
        return self
