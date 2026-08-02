"""Typed contracts and evidence records for focused Genesis kernel synthesis."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated, Final, Literal, Self, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

KERNEL_LAB_SCHEMA_VERSION: Final = "sloforge.genesis.kernel-lab/v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Identifier = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$",
    ),
]
CaseCategory: TypeAlias = Literal["edge", "random", "stride", "noncontiguous", "nonfinite", "alias"]


class KernelModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
        allow_inf_nan=False,
    )


class KernelBackend(StrEnum):
    PYTHON_CPU = "python_cpu"
    TRITON = "triton"


class EvidenceSource(StrEnum):
    AUTOPSY_MEASURED = "autopsy_measured"
    CPU_PROFILE_MEASURED = "cpu_profile_measured"
    DIGITAL_TWIN = "digital_twin"


class LabStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"
    UNAVAILABLE = "unavailable"


class AcceptanceStatus(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"
    UNAVAILABLE = "unavailable"


class AdapterStatus(StrEnum):
    UNEXERCISED = "unexercised"
    UNAVAILABLE = "unavailable"


class ShapeDimension(KernelModel):
    name: Identifier
    minimum: int = Field(gt=0)
    maximum: int = Field(gt=0)
    multiple_of: int = Field(default=1, gt=0)

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        if self.maximum < self.minimum:
            raise ValueError("shape maximum must be at least its minimum")
        return self


class TensorConstraint(KernelModel):
    name: Identifier
    dtype: NonEmpty
    shape: tuple[ShapeDimension, ...]
    allowed_strides: tuple[int, ...]
    contiguous_required: bool
    mutable: bool

    @model_validator(mode="after")
    def validate_tensor(self) -> Self:
        if not self.shape:
            raise ValueError("tensor shape cannot be empty")
        if not self.allowed_strides or any(stride <= 0 for stride in self.allowed_strides):
            raise ValueError("tensor strides must be positive and explicit")
        if len(self.allowed_strides) != len(set(self.allowed_strides)):
            raise ValueError("tensor strides must be unique")
        if self.contiguous_required and self.allowed_strides != (1,):
            raise ValueError("contiguous tensors must declare only stride one")
        return self


class AliasingConstraint(KernelModel):
    output_may_alias: tuple[Identifier, ...]
    activation_alias_forbidden: bool = True
    exact_index_mapping_required: bool = True


class NumericalContract(KernelModel):
    exact: Literal[True] = True
    rounding: Literal["nearest_ties_to_even"] = "nearest_ties_to_even"
    saturation_minimum: int = -127
    saturation_maximum: int = 127
    maximum_absolute_error: Annotated[float, Field(ge=0.0, le=0.0)] = 0.0
    maximum_relative_error: Annotated[float, Field(ge=0.0, le=0.0)] = 0.0
    nan_behavior: Literal["reject"] = "reject"
    infinity_behavior: Literal["reject"] = "reject"
    deterministic: Literal[True] = True

    @model_validator(mode="after")
    def validate_saturation(self) -> Self:
        if self.saturation_minimum >= self.saturation_maximum:
            raise ValueError("saturation bounds are invalid")
        return self


class ArchitectureConstraint(KernelModel):
    backend: KernelBackend
    architectures: tuple[NonEmpty, ...]
    minimum_python: NonEmpty
    requires_gpu: bool
    hidden_fallback_forbidden: Literal[True] = True


class OperatorSchema(KernelModel):
    schema_version: Literal["sloforge.genesis.kernel-lab/v1"] = KERNEL_LAB_SCHEMA_VERSION
    operator_id: Literal["quantized-state-update"] = "quantized-state-update"
    symbol: Literal["quantized_recurrent_state_update"] = "quantized_recurrent_state_update"
    semantic_contract: NonEmpty
    inputs: tuple[TensorConstraint, ...]
    output: TensorConstraint
    aliasing: AliasingConstraint
    numerical: NumericalContract
    architecture: ArchitectureConstraint
    stateful: Literal[True] = True
    atomicity: Literal["per_token"] = "per_token"

    @model_validator(mode="after")
    def validate_schema(self) -> Self:
        names = [item.name for item in self.inputs]
        if len(names) != len(set(names)):
            raise ValueError("operator inputs must have unique names")
        if self.output.name in names:
            raise ValueError("output must have a distinct schema name")
        if any(name not in names for name in self.aliasing.output_may_alias):
            raise ValueError("aliasing constraint references an unknown input")
        return self


class BottleneckEvidence(KernelModel):
    evidence_id: Identifier
    source: EvidenceSource
    operator_id: Identifier
    genome_region: Identifier
    observed_fraction: float = Field(gt=0.0, le=1.0)
    sample_count: int = Field(gt=0)
    deterministic_seed: int = Field(ge=0)
    raw_evidence_path: NonEmpty
    raw_evidence_sha256: str
    hardware_fingerprint: NonEmpty
    workload_fingerprint: NonEmpty
    synthetic: bool

    @field_validator("raw_evidence_sha256")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError("raw evidence digest must be lowercase sha256")
        return value


class RawBottleneckRecord(KernelModel):
    operator_id: Identifier
    inclusive_cpu_time_ns: tuple[int, ...]
    token_loop_fraction: float = Field(gt=0.0, le=1.0)
    seed: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_samples(self) -> Self:
        if not self.inclusive_cpu_time_ns or any(
            sample <= 0 for sample in self.inclusive_cpu_time_ns
        ):
            raise ValueError("raw bottleneck samples must be non-empty and positive")
        return self


class KernelParameters(KernelModel):
    unroll_factor: Literal[1, 2, 4]
    clamp_style: Literal["builtins", "branches"]
    cache_coefficients: bool


class KernelCandidate(KernelModel):
    candidate_id: Identifier
    operator_schema: OperatorSchema
    backend: KernelBackend
    target_architecture: NonEmpty
    parameters: KernelParameters
    source_sha256: str
    deterministic_seed: int = Field(ge=0)
    fallback_symbol: Literal["reference_quantized_state_update"] = (
        "reference_quantized_state_update"
    )

    @field_validator("source_sha256")
    @classmethod
    def validate_source_digest(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError("candidate source digest must be lowercase sha256")
        return value


class KernelCase(KernelModel):
    case_id: Identifier
    previous_storage: tuple[int, ...]
    activation_storage: tuple[str, ...]
    count: int = Field(gt=0, le=128)
    previous_offset: int = Field(ge=0)
    previous_stride: int = Field(gt=0)
    activation_offset: int = Field(ge=0)
    activation_stride: int = Field(gt=0)
    output_alias_previous: bool
    expected: tuple[int, ...] | None
    expected_error: Literal["ValueError"] | None
    category: CaseCategory

    @model_validator(mode="after")
    def validate_case(self) -> Self:
        if (self.expected is None) == (self.expected_error is None):
            raise ValueError("case must declare exactly one expected outcome")
        if self.expected is not None and len(self.expected) != self.count:
            raise ValueError("expected output length must equal count")
        previous_last = self.previous_offset + (self.count - 1) * self.previous_stride
        activation_last = self.activation_offset + (self.count - 1) * self.activation_stride
        if previous_last >= len(self.previous_storage):
            raise ValueError("previous storage is too small for its view")
        if activation_last >= len(self.activation_storage):
            raise ValueError("activation storage is too small for its view")
        if len(self.previous_storage) > 512 or len(self.activation_storage) > 512:
            raise ValueError("case storage exceeds the bounded operator view")
        return self


class CaseResult(KernelModel):
    case_id: Identifier
    observed: tuple[int, ...] | None
    error_type: str | None
    previous_storage_after: tuple[int, ...]


class CorrectnessMismatch(KernelModel):
    case_id: Identifier
    category: NonEmpty
    expected: NonEmpty
    observed: NonEmpty


class CorrectnessEvidence(KernelModel):
    status: LabStatus
    candidate: KernelCandidate
    deterministic_seed: int = Field(ge=0)
    source_allowlist_valid: bool
    cases_executed: int = Field(ge=0)
    edge_cases: int = Field(ge=0)
    randomized_cases: int = Field(ge=0)
    stride_cases: int = Field(ge=0)
    noncontiguous_cases: int = Field(ge=0)
    nonfinite_cases: int = Field(ge=0)
    alias_cases: int = Field(ge=0)
    mismatches: tuple[CorrectnessMismatch, ...]
    sandbox_termination: NonEmpty
    sandbox_backend: NonEmpty
    assumptions: tuple[NonEmpty, ...]

    @model_validator(mode="after")
    def validate_correctness(self) -> Self:
        if self.status is LabStatus.PASSED and (self.mismatches or self.cases_executed == 0):
            raise ValueError("passed correctness evidence requires executed mismatch-free cases")
        return self


class BenchmarkSample(KernelModel):
    regime: Identifier
    alternative: Literal["reference", "candidate"]
    order_index: int = Field(ge=0)
    trial_index: int = Field(ge=0)
    iterations: int = Field(gt=0)
    duration_ns: int = Field(gt=0)


class BenchmarkRegimeEvidence(KernelModel):
    regime: Identifier
    status: LabStatus
    warmup_count: int = Field(gt=0)
    repetitions: int = Field(ge=7)
    iterations: int = Field(gt=0)
    run_order: tuple[Literal["reference", "candidate"], ...]
    reference_samples_ns: tuple[float, ...]
    candidate_samples_ns: tuple[float, ...]
    reference_median_ns: float = Field(gt=0.0)
    candidate_median_ns: float = Field(gt=0.0)
    improvement_percent: float
    confidence_interval_low_percent: float
    confidence_interval_high_percent: float
    effect_size: float
    rationale: NonEmpty

    @model_validator(mode="after")
    def validate_regime(self) -> Self:
        if len(self.reference_samples_ns) != self.repetitions:
            raise ValueError("reference sample count must match repetitions")
        if len(self.candidate_samples_ns) != self.repetitions:
            raise ValueError("candidate sample count must match repetitions")
        if len(self.run_order) != self.repetitions * 2:
            raise ValueError("run order must contain every reference and candidate trial")
        if self.run_order.count("reference") != self.repetitions:
            raise ValueError("run order has the wrong reference count")
        if self.run_order.count("candidate") != self.repetitions:
            raise ValueError("run order has the wrong candidate count")
        if self.confidence_interval_low_percent > self.confidence_interval_high_percent:
            raise ValueError("confidence interval bounds are reversed")
        return self


class KernelBenchmarkReport(KernelModel):
    status: LabStatus
    candidate: KernelCandidate
    deterministic_seed: int = Field(ge=0)
    hardware_fingerprint: NonEmpty
    software_manifest: tuple[NonEmpty, ...]
    workload_fingerprint: str
    raw_samples: tuple[BenchmarkSample, ...]
    raw_samples_sha256: str | None
    measurement_provenance: Literal["sandboxed_cpu_wall_clock"] = "sandboxed_cpu_wall_clock"
    regimes: tuple[BenchmarkRegimeEvidence, ...]
    sandbox_termination: NonEmpty
    sandbox_backend: NonEmpty

    @field_validator("workload_fingerprint")
    @classmethod
    def validate_workload_digest(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError("workload fingerprint must be lowercase sha256")
        return value

    @field_validator("raw_samples_sha256")
    @classmethod
    def validate_raw_digest(cls, value: str | None) -> str | None:
        if value is not None and _SHA256.fullmatch(value) is None:
            raise ValueError("raw sample digest must be lowercase sha256")
        return value

    @model_validator(mode="after")
    def validate_benchmark(self) -> Self:
        if bool(self.raw_samples) != (self.raw_samples_sha256 is not None):
            raise ValueError("raw samples and their digest must be present together")
        declared = {item.regime for item in self.regimes}
        if any(sample.regime not in declared for sample in self.raw_samples):
            raise ValueError("raw sample references an undeclared benchmark regime")
        return self


class KernelBenchmarkConfig(KernelModel):
    deterministic_seed: int = Field(ge=0)
    warmup_count: int = Field(default=3, gt=0)
    repetitions: int = Field(default=9, ge=7, le=1_000)
    micro_iterations: int = Field(default=100, gt=0, le=1_000_000)
    token_loop_iterations: int = Field(default=20, gt=0, le=100_000)
    token_steps: int = Field(default=8, gt=0, le=256)
    bootstrap_rounds: int = Field(default=500, ge=100, le=100_000)
    confidence: float = Field(default=0.95, gt=0.5, lt=1.0)
    practical_significance_percent: float = Field(default=1.0, ge=0.0)
    noise_floor_percent: float = Field(default=0.5, ge=0.0)
    wall_time_seconds: float = Field(default=15.0, gt=0.0, le=120.0)


class CandidateDecision(KernelModel):
    candidate_id: Identifier
    status: AcceptanceStatus
    correctness_status: LabStatus
    microbenchmark_status: LabStatus
    end_to_end_status: LabStatus
    claim: NonEmpty
    reasons: tuple[NonEmpty, ...]

    @model_validator(mode="after")
    def validate_decision(self) -> Self:
        if self.status is AcceptanceStatus.ACCEPTED and (
            self.correctness_status is not LabStatus.PASSED
            or self.microbenchmark_status is not LabStatus.PASSED
            or self.end_to_end_status is not LabStatus.PASSED
        ):
            raise ValueError("accepted candidates must pass every required gate")
        return self


class TritonAdapterReport(KernelModel):
    status: AdapterStatus
    installed_version: str | None
    exercised: Literal[False] = False
    reason: NonEmpty
    required_opt_in: Literal["SLOFORGE_GENESIS_ALLOW_GPU"] = "SLOFORGE_GENESIS_ALLOW_GPU"


class KernelLabReport(KernelModel):
    schema_version: Literal["sloforge.genesis.kernel-lab/v1"] = KERNEL_LAB_SCHEMA_VERSION
    evidence: BottleneckEvidence
    operator_schema: OperatorSchema
    candidates: tuple[KernelCandidate, ...]
    correctness: tuple[CorrectnessEvidence, ...]
    benchmarks: tuple[KernelBenchmarkReport, ...]
    decisions: tuple[CandidateDecision, ...]
    triton_adapter: TritonAdapterReport
