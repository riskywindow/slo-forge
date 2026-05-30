"""Typed intermediate representation for portable cold-start planning."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

WARMPATH_SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Identifier = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_.-]*$")]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveFloat = Annotated[float, Field(gt=0.0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class WarmPathModel(BaseModel):
    """Strict immutable base for plans and evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class ArtifactKind(StrEnum):
    CONTAINER_LAYER = "container_layer"
    PYTHON_ENVIRONMENT = "python_environment"
    TOKENIZER = "tokenizer"
    MODEL_CONFIG = "model_config"
    MODEL_WEIGHTS = "model_weights"
    QUANTIZED_WEIGHTS = "quantized_weights"
    ADAPTER_WEIGHTS = "adapter_weights"
    COMPILED_KERNEL = "compiled_kernel"
    RUNTIME_CACHE = "runtime_cache"
    GRAPH_CAPTURE = "graph_capture"
    ENGINE_METADATA = "engine_metadata"
    COMMUNICATION_METADATA = "communication_metadata"
    PROCESS_CHECKPOINT = "process_checkpoint"
    CPU_MEMORY_IMAGE = "cpu_memory_image"
    PINNED_HOST_BUFFER = "pinned_host_buffer"
    GPU_MEMORY_IMAGE = "gpu_memory_image"
    READINESS_METADATA = "readiness_metadata"


class StorageKind(StrEnum):
    OBJECT_STORAGE = "object_storage"
    REMOTE_FILESYSTEM = "remote_filesystem"
    REGIONAL_CACHE = "regional_cache"
    ZONAL_CACHE = "zonal_cache"
    PEER_HOST = "peer_host"
    REMOTE_MEMORY = "remote_memory"
    LOCAL_NVME = "local_nvme"
    PAGE_CACHE = "page_cache"
    HOST_MEMORY = "host_memory"
    PINNED_HOST_MEMORY = "pinned_host_memory"
    GPU_HBM = "gpu_hbm"


class StartupStage(StrEnum):
    FETCH = "fetch"
    VERIFY = "verify"
    RESTORE = "restore"
    WEIGHT_LOAD = "weight_load"
    RUNTIME_INITIALIZATION = "runtime_initialization"
    COMMUNICATION_INITIALIZATION = "communication_initialization"
    GRAPH_CAPTURE = "graph_capture"
    READINESS = "readiness"


class MaterializationMode(StrEnum):
    EAGER_RESTORE = "eager_restore"
    LAZY_RESTORE = "lazy_restore"
    REBUILD = "rebuild"
    KEEP_WARM = "keep_warm"


class SecurityClass(StrEnum):
    PUBLIC = "public"
    INTERNAL = "internal"
    RESTRICTED = "restricted"


class CompatibilityConstraint(WarmPathModel):
    """Host constraints for artifacts whose captured state is not portable."""

    portable: bool = True
    operating_systems: tuple[NonEmpty, ...] = ()
    architectures: tuple[NonEmpty, ...] = ()
    runtimes: tuple[NonEmpty, ...] = ()
    runtime_versions: tuple[NonEmpty, ...] = ()
    gpu_architectures: tuple[NonEmpty, ...] = ()
    driver_major_versions: tuple[NonNegativeInt, ...] = ()
    topology_fingerprint: Sha256 | None = None
    captured_host_fingerprint: Sha256 | None = None
    maximum_gpu_count: PositiveInt | None = None

    @model_validator(mode="after")
    def require_identity_for_nonportable_state(self) -> Self:
        if not self.portable and not (
            self.captured_host_fingerprint or self.topology_fingerprint or self.gpu_architectures
        ):
            raise ValueError("non-portable artifacts require an exact host or GPU constraint")
        return self


class HostEnvironment(WarmPathModel):
    operating_system: NonEmpty
    architecture: NonEmpty
    runtime: NonEmpty
    runtime_version: NonEmpty
    gpu_architecture: NonEmpty | None = None
    driver_major_version: NonNegativeInt | None = None
    topology_fingerprint: Sha256 | None = None
    host_fingerprint: Sha256
    gpu_count: NonNegativeInt = 0


class ArtifactNode(WarmPathModel):
    artifact_id: Identifier
    kind: ArtifactKind
    size_bytes: PositiveInt
    sha256: Sha256
    dependencies: tuple[Identifier, ...] = ()
    required_for_readiness: bool = True
    lazy_restore_allowed: bool = False
    rebuild_time_ms: NonNegativeFloat = 0.0
    compatibility: CompatibilityConstraint = Field(default_factory=CompatibilityConstraint)
    security_class: SecurityClass = SecurityClass.INTERNAL
    source_relative_path: NonEmpty

    @model_validator(mode="after")
    def validate_restore_mode(self) -> Self:
        if not self.required_for_readiness and not self.lazy_restore_allowed:
            raise ValueError("non-readiness artifacts must allow lazy restoration")
        if self.kind == ArtifactKind.GPU_MEMORY_IMAGE and self.compatibility.portable:
            raise ValueError("GPU memory images must declare non-portable compatibility")
        if self.source_relative_path.startswith(
            ("/", "\\")
        ) or ".." in self.source_relative_path.split("/"):
            raise ValueError("artifact source path must be relative and cannot traverse parents")
        return self


class ArtifactGraph(WarmPathModel):
    schema_version: Literal["1.0.0"] = WARMPATH_SCHEMA_VERSION
    graph_id: Identifier
    artifacts: tuple[ArtifactNode, ...]

    @model_validator(mode="after")
    def validate_dag(self) -> Self:
        if not self.artifacts:
            raise ValueError("artifact graph cannot be empty")
        identifiers = [artifact.artifact_id for artifact in self.artifacts]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("artifact identifiers must be unique")
        known = set(identifiers)
        dependencies = {item.artifact_id: item.dependencies for item in self.artifacts}
        for artifact_id, edges in dependencies.items():
            if artifact_id in edges:
                raise ValueError(f"artifact {artifact_id} cannot depend on itself")
            unknown = set(edges) - known
            if unknown:
                raise ValueError(
                    f"artifact {artifact_id} has unknown dependencies: {sorted(unknown)}"
                )

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(artifact_id: str) -> None:
            if artifact_id in visiting:
                raise ValueError(f"artifact graph contains a cycle through {artifact_id}")
            if artifact_id in visited:
                return
            visiting.add(artifact_id)
            for dependency in dependencies[artifact_id]:
                visit(dependency)
            visiting.remove(artifact_id)
            visited.add(artifact_id)

        for artifact_id in identifiers:
            visit(artifact_id)
        return self

    def topological_order(self) -> tuple[ArtifactNode, ...]:
        """Return dependencies before consumers using stable artifact ordering."""

        by_id = {artifact.artifact_id: artifact for artifact in self.artifacts}
        ordered: list[ArtifactNode] = []
        visited: set[str] = set()

        def append(artifact_id: str) -> None:
            if artifact_id in visited:
                return
            for dependency in by_id[artifact_id].dependencies:
                append(dependency)
            visited.add(artifact_id)
            ordered.append(by_id[artifact_id])

        for artifact in self.artifacts:
            append(artifact.artifact_id)
        return tuple(ordered)


class StorageTierSpec(WarmPathModel):
    tier_id: Identifier
    kind: StorageKind
    capacity_bytes: PositiveInt
    read_bandwidth_bytes_per_second: PositiveFloat
    base_read_latency_ms: NonNegativeFloat
    maximum_parallel_reads: PositiveInt = 1
    hourly_cost_per_gib: NonNegativeFloat = 0.0
    restore_failure_probability: Probability = 0.0
    persistent: bool = True
    maximum_security_class: SecurityClass = SecurityClass.INTERNAL
    local_path: NonEmpty | None = None

    @model_validator(mode="after")
    def require_local_path_for_local_storage(self) -> Self:
        local = {
            StorageKind.LOCAL_NVME,
            StorageKind.PAGE_CACHE,
            StorageKind.HOST_MEMORY,
            StorageKind.PINNED_HOST_MEMORY,
        }
        if self.kind in local and self.local_path is None:
            raise ValueError(f"{self.kind} requires local_path")
        return self


class StageMeasurement(WarmPathModel):
    artifact_id: Identifier
    tier_id: Identifier
    stage: StartupStage
    warmup_count: NonNegativeInt
    raw_samples_ms: tuple[NonNegativeFloat, ...]
    median_ms: NonNegativeFloat
    p95_ms: NonNegativeFloat
    median_absolute_deviation_ms: NonNegativeFloat
    confidence_level: Annotated[float, Field(gt=0.0, lt=1.0)]
    confidence_interval_low_ms: NonNegativeFloat
    confidence_interval_high_ms: NonNegativeFloat
    source: Literal["measured", "synthetic_fixture"]
    environment_fingerprint: Sha256
    invocation: NonEmpty
    timeout_seconds: PositiveFloat
    artifact_hash: Sha256

    @model_validator(mode="after")
    def validate_samples(self) -> Self:
        if len(self.raw_samples_ms) < 3:
            raise ValueError("stage measurements require at least three samples")
        if (
            not self.confidence_interval_low_ms
            <= self.median_ms
            <= self.confidence_interval_high_ms
        ):
            raise ValueError("confidence interval must contain the median")
        if self.p95_ms < min(self.raw_samples_ms):
            raise ValueError("p95 must be within the raw sample range")
        return self


class StartupProfile(WarmPathModel):
    schema_version: Literal["1.0.0"] = WARMPATH_SCHEMA_VERSION
    profile_id: Identifier
    graph_id: Identifier
    host: HostEnvironment
    tiers: tuple[StorageTierSpec, ...]
    measurements: tuple[StageMeasurement, ...]
    raw_artifact_directory: NonEmpty
    environment_manifest_path: NonEmpty
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        tier_ids = [tier.tier_id for tier in self.tiers]
        if len(tier_ids) != len(set(tier_ids)):
            raise ValueError("storage tier identifiers must be unique")
        unknown = {item.tier_id for item in self.measurements} - set(tier_ids)
        if unknown:
            raise ValueError(f"measurements reference unknown storage tiers: {sorted(unknown)}")
        return self


class WarmPathObjective(WarmPathModel):
    ready_time_weight: NonNegativeFloat = 1.0
    hourly_cost_weight: NonNegativeFloat = 0.0
    failure_risk_weight: NonNegativeFloat = 0.0
    maximum_p95_ready_time_ms: PositiveFloat | None = None
    maximum_hourly_cost: NonNegativeFloat | None = None
    warm_replica_hourly_cost: NonNegativeFloat = 0.0
    maximum_warm_replicas: NonNegativeInt = 0

    @model_validator(mode="after")
    def require_objective(self) -> Self:
        if self.ready_time_weight + self.hourly_cost_weight + self.failure_risk_weight <= 0.0:
            raise ValueError("at least one objective weight must be positive")
        return self


class ArtifactPlacement(WarmPathModel):
    artifact_id: Identifier
    tier_id: Identifier
    mode: MaterializationMode
    prefetch_order: NonNegativeInt
    expected_duration_ms: NonNegativeFloat
    estimate_source: Literal["measured", "theoretical", "warm"]


class StartupStagePrediction(WarmPathModel):
    artifact_id: Identifier
    stage: StartupStage
    start_ms: NonNegativeFloat
    finish_ms: NonNegativeFloat
    resource: Identifier
    estimate_source: Literal["measured", "theoretical", "warm"]


class ColdStartTrial(WarmPathModel):
    trial_index: NonNegativeInt
    ready_time_ms: NonNegativeFloat | None
    restore_failed: bool
    used_rebuild_fallback: bool


class ColdStartSimulation(WarmPathModel):
    seed: int
    trials: tuple[ColdStartTrial, ...]
    p50_ready_time_ms: NonNegativeFloat
    p95_ready_time_ms: NonNegativeFloat
    interval_low_ms: NonNegativeFloat
    interval_high_ms: NonNegativeFloat
    restore_failure_probability: Probability
    stage_predictions: tuple[StartupStagePrediction, ...]
    estimate_sources: tuple[Literal["measured", "theoretical", "warm"], ...]

    @model_validator(mode="after")
    def validate_trials(self) -> Self:
        if len(self.trials) < 3:
            raise ValueError("cold-start simulation requires at least three trials")
        if not self.interval_low_ms <= self.p50_ready_time_ms:
            raise ValueError("simulation interval low bound exceeds p50")
        if not self.p95_ready_time_ms <= self.interval_high_ms:
            raise ValueError("simulation interval high bound is below p95")
        return self


class RejectedWarmPathCandidate(WarmPathModel):
    candidate_id: Identifier
    reason_code: Literal[
        "compatibility",
        "capacity",
        "startup_slo",
        "cost_budget",
        "security",
        "failure",
    ]
    explanation: NonEmpty


class WarmPathPlan(WarmPathModel):
    schema_version: Literal["1.0.0"] = WARMPATH_SCHEMA_VERSION
    plan_id: Identifier
    profile_id: Identifier
    graph_id: Identifier
    host_fingerprint: Sha256
    placements: tuple[ArtifactPlacement, ...]
    warm_replica_count: NonNegativeInt
    predicted_p50_ready_time_ms: NonNegativeFloat
    predicted_p95_ready_time_ms: NonNegativeFloat
    prediction_interval_low_ms: NonNegativeFloat
    prediction_interval_high_ms: NonNegativeFloat
    predicted_hourly_cost: NonNegativeFloat
    predicted_restore_failure_probability: Probability
    objective_value: NonNegativeFloat
    stage_predictions: tuple[StartupStagePrediction, ...]
    rejected_candidates: tuple[RejectedWarmPathCandidate, ...]
    optimizer: Literal["exhaustive", "beam"]
    optimizer_seed: int
    evaluated_candidate_count: PositiveInt
    evidence_references: tuple[NonEmpty, ...]
    compiler_version: NonEmpty
    graph_hash: Sha256
    profile_hash: Sha256

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        identifiers = [placement.artifact_id for placement in self.placements]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("each artifact may have only one placement")
        if self.prediction_interval_low_ms > self.predicted_p50_ready_time_ms:
            raise ValueError("prediction interval low bound exceeds p50")
        if self.predicted_p95_ready_time_ms > self.prediction_interval_high_ms:
            raise ValueError("prediction interval high bound is below p95")
        if not self.evidence_references:
            raise ValueError("WarmPath plans require evidence references")
        return self


class ExecutorArtifactRecord(WarmPathModel):
    artifact_id: Identifier
    tier_id: Identifier
    mode: MaterializationMode
    status: Literal["restored", "rebuilt", "kept_warm", "deferred", "failed"]
    started_ns: NonNegativeInt
    finished_ns: NonNegativeInt
    bytes_materialized: NonNegativeInt
    checksum_verified: bool
    source_path: NonEmpty | None = None
    destination_path: NonEmpty | None = None
    error: str | None = None

    @model_validator(mode="after")
    def validate_timing(self) -> Self:
        if self.finished_ns < self.started_ns:
            raise ValueError("executor finish time cannot precede start time")
        if self.status == "failed" and self.error is None:
            raise ValueError("failed records require an error")
        return self


class ExecutionRecord(WarmPathModel):
    schema_version: Literal["1.0.0"] = WARMPATH_SCHEMA_VERSION
    execution_id: Identifier
    plan_id: Identifier
    success: bool
    records: tuple[ExecutorArtifactRecord, ...]
    ready_time_ms: NonNegativeFloat
    output_directory: NonEmpty
    cache_evictions: tuple[Identifier, ...] = ()
    failure_reason: str | None = None
    artifact_hash: Sha256

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        if self.success == (self.failure_reason is not None):
            raise ValueError("failure_reason must be present exactly when execution failed")
        return self


def compatibility_violations(
    constraint: CompatibilityConstraint, host: HostEnvironment
) -> tuple[str, ...]:
    """Return every incompatibility instead of silently selecting a fallback."""

    failures: list[str] = []
    checks: tuple[tuple[str, object, tuple[object, ...]], ...] = (
        ("operating_system", host.operating_system, constraint.operating_systems),
        ("architecture", host.architecture, constraint.architectures),
        ("runtime", host.runtime, constraint.runtimes),
        ("runtime_version", host.runtime_version, constraint.runtime_versions),
    )
    for name, actual, allowed in checks:
        if allowed and actual not in allowed:
            failures.append(f"{name}={actual!s} is not in {allowed!r}")
    if constraint.gpu_architectures and host.gpu_architecture not in constraint.gpu_architectures:
        failures.append("GPU architecture is incompatible")
    if (
        constraint.driver_major_versions
        and host.driver_major_version not in constraint.driver_major_versions
    ):
        failures.append("driver major version is incompatible")
    if (
        constraint.topology_fingerprint is not None
        and host.topology_fingerprint != constraint.topology_fingerprint
    ):
        failures.append("topology fingerprint differs from capture")
    if (
        constraint.captured_host_fingerprint is not None
        and host.host_fingerprint != constraint.captured_host_fingerprint
    ):
        failures.append("host fingerprint differs from capture")
    if constraint.maximum_gpu_count is not None and host.gpu_count > constraint.maximum_gpu_count:
        failures.append("host GPU count exceeds snapshot compatibility")
    return tuple(failures)


def security_allows(tier: StorageTierSpec, artifact: ArtifactNode) -> bool:
    """Whether the tier's declared security boundary admits the artifact."""

    order = {
        SecurityClass.PUBLIC: 0,
        SecurityClass.INTERNAL: 1,
        SecurityClass.RESTRICTED: 2,
    }
    return order[artifact.security_class] <= order[tier.maximum_security_class]


def safe_identifier(value: str) -> str:
    """Normalize user-facing IDs for plan components without hiding emptiness."""

    normalized = re.sub(r"[^a-z0-9_.-]+", "-", value.lower()).strip("-.")
    if not normalized or not normalized[0].isalpha():
        normalized = f"warm-{normalized}"
    return normalized
