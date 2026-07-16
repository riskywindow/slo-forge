"""Strict artifact and statistical models for Continuum CPU evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4096)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class EvaluationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class EvaluationRequest(EvaluationModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    output_dir: Path
    seeds: tuple[Annotated[int, Field(ge=0, le=2**64 - 1)], ...]
    git_commit: NonEmpty
    continuum_version: NonEmpty = "0.1.0"
    capture_timestamp: NonEmpty = "2026-08-02T00:00:00Z"
    initial_output_tokens: Annotated[int, Field(ge=8, le=128)] = 16
    delta_rounds: tuple[Annotated[int, Field(ge=1, le=64)], ...] = (3, 2)
    resumed_tokens: Annotated[int, Field(ge=2, le=64)] = 3
    converter_repetitions: Annotated[int, Field(ge=3, le=25)] = 5

    @model_validator(mode="after")
    def validate_seed_matrix(self) -> Self:
        if not 3 <= len(self.seeds) <= 30:
            raise ValueError("evaluation requires 3..30 seeds")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("evaluation seeds must be unique")
        return self


class ArtifactReference(EvaluationModel):
    path: NonEmpty
    sha256: Sha256
    size_bytes: Annotated[int, Field(ge=1, le=64 * 1024 * 1024)]
    media_type: Literal["application/json", "text/markdown", "text/html"]


class PackageVersion(EvaluationModel):
    package: NonEmpty
    version: NonEmpty


class SoftwareManifest(EvaluationModel):
    python_version: NonEmpty
    python_implementation: NonEmpty
    platform: NonEmpty
    git_commit: NonEmpty
    continuum_version: NonEmpty
    packages: tuple[PackageVersion, ...]


class HardwareManifest(EvaluationModel):
    mode: Literal["cpu_only"] = "cpu_only"
    machine: NonEmpty
    processor: str
    operating_system: NonEmpty
    logical_cpu_count: Annotated[int, Field(ge=1)]
    nvidia_smi_available: bool
    gpu_opt_in_enabled: bool
    gpu_exercised: Literal[False] = False
    rdma_exercised: Literal[False] = False
    hardware_result_claims: tuple[str, ...] = ()


class AdapterEvaluation(EvaluationModel):
    runtime: NonEmpty
    version: str | None
    adapter_status: NonEmpty
    discovery_exercised: bool
    migration_exercised: bool
    capabilities: tuple[str, ...]
    evidence: tuple[str, ...]
    limitation: NonEmpty


class SeedMeasurement(EvaluationModel):
    schema_version: Literal["sloforge.continuum.seed-measurement/v1"] = (
        "sloforge.continuum.seed-measurement/v1"
    )
    seed: Annotated[int, Field(ge=0, le=2**64 - 1)]
    observed_flagship_wall_ns: Annotated[int, Field(gt=0)]
    observed_conversion_campaign_wall_ns: Annotated[int, Field(gt=0)]
    observed_canonical_conversion_median_ns: Annotated[int, Field(gt=0)]
    observed_direct_conversion_median_ns: Annotated[int, Field(gt=0)]
    observed_precopy_interruption_ns: Annotated[int, Field(gt=0)]
    observed_stop_and_copy_interruption_ns: Annotated[int, Field(gt=0)]
    selected_converter: Literal["canonical_cpu", "direct_cpu"]
    conversion_exact: bool
    conversion_maximum_absolute_error: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    synthetic_transport_elapsed_us: Annotated[int, Field(ge=0)]
    synthetic_transport_bytes_on_wire: Annotated[int, Field(ge=0)]
    synthetic_final_delta_transfer_us: Annotated[int, Field(ge=0)]
    gateway_accepted_tokens: Annotated[int, Field(ge=1)]
    gateway_duplicate_count: Annotated[int, Field(ge=0)]
    gateway_gap_count: Annotated[int, Field(ge=0)]
    failed_transaction_final_phase: NonEmpty
    successful_transaction_final_phase: NonEmpty
    source_owner_epoch: Annotated[int, Field(ge=1)]
    destination_owner_epoch: Annotated[int, Field(ge=2)]
    checkpoint_bytes_deduplicated: Annotated[int, Field(ge=1)]
    cow_divergence_unique_bytes: Annotated[int, Field(ge=1)]
    direct_reuse_class: NonEmpty
    recomputation_class: NonEmpty
    planner_selected_strategy: NonEmpty
    planner_objective: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    planner_oracle_objective: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    planner_regret: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    fixed_stop_objective: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    planner_predicted_interruption_ms: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    planner_observed_interruption_ms: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    planner_interruption_absolute_error_ms: Annotated[float, Field(ge=0, allow_inf_nan=False)]


class SeedEvaluation(SeedMeasurement):
    measurement_artifact: ArtifactReference
    flagship_artifact: ArtifactReference
    conversion_artifact: ArtifactReference
    stop_and_copy_artifact: ArtifactReference
    planner_artifact: ArtifactReference


class StopAndCopyMeasurement(EvaluationModel):
    schema_version: Literal["sloforge.continuum.stop-copy-measurement/v1"] = (
        "sloforge.continuum.stop-copy-measurement/v1"
    )
    seed: Annotated[int, Field(ge=0, le=2**64 - 1)]
    observed_interruption_ns: Annotated[int, Field(gt=0)]
    observed_precopy_interruption_ns: Annotated[int, Field(gt=0)]
    checkpoint_bytes: Annotated[int, Field(gt=0)]
    source_owner_epoch: Annotated[int, Field(ge=1)]
    destination_owner_epoch: Annotated[int, Field(ge=2)]
    resumed_token_index: Annotated[int, Field(ge=0)]
    duplicate_count: Annotated[int, Field(ge=0)]
    gap_count: Annotated[int, Field(ge=0)]
    measurement_scope: Literal["cpu_in_memory_content_store"] = "cpu_in_memory_content_store"


class ConfidenceInterval(EvaluationModel):
    metric: NonEmpty
    unit: NonEmpty
    sample_count: Annotated[int, Field(ge=2)]
    mean: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    sample_standard_deviation: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    lower: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    upper: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    confidence_level: Annotated[float, Field(ge=0.95, le=0.95)] = 0.95
    method: Literal["two_sided_student_t"] = "two_sided_student_t"
    metric_class: Literal["observed_host", "synthetic_protocol", "artifact_derived"]

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if not self.lower <= self.mean <= self.upper:
            raise ValueError("confidence interval must contain its sample mean")
        return self


class HypothesisOutcome(EvaluationModel):
    hypothesis: Literal["H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9"]
    status: Literal["pass", "mixed", "negative", "not_exercised"]
    statement: NonEmpty
    evidence: tuple[NonEmpty, ...]
    limitation: NonEmpty


class EvaluationBundle(EvaluationModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    evaluation_id: Sha256
    generated_at: NonEmpty
    exact_command: NonEmpty
    seeds: tuple[Annotated[int, Field(ge=0, le=2**64 - 1)], ...]
    software: SoftwareManifest
    hardware: HardwareManifest
    adapters: tuple[AdapterEvaluation, ...]
    per_seed: tuple[SeedEvaluation, ...]
    confidence_intervals: tuple[ConfidenceInterval, ...]
    hypotheses: tuple[HypothesisOutcome, ...]
    negative_results: tuple[NonEmpty, ...]


class ReportSet(EvaluationModel):
    evaluation_markdown: ArtifactReference
    evaluation_html: ArtifactReference
    compatibility_markdown: ArtifactReference
    fault_tolerance_markdown: ArtifactReference
    runtime_adapters_markdown: ArtifactReference


class EvaluationCampaignResult(EvaluationModel):
    evaluation: EvaluationBundle
    summary_artifact: ArtifactReference
    reports: ReportSet
