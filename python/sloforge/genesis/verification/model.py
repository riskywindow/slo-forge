"""Independent, scoped evidence models for Genesis candidate verification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum


class VerificationLevel(IntEnum):
    BUILD = 0
    DIFFERENTIAL = 1
    PROPERTY = 2
    BOUNDED_EXHAUSTIVE = 3
    SOLVER_BACKED = 4
    HARDWARE_OPERATIONAL = 5


class EvidenceStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class ShapeBound:
    minimum: int
    maximum: int
    symbol: str


@dataclass(frozen=True, slots=True)
class OperatorContract:
    operator_id: str
    shape: tuple[ShapeBound, ...]
    dtypes: tuple[str, ...]
    allow_non_contiguous: bool
    allow_aliasing: bool
    exact: bool
    maximum_absolute_error: float
    maximum_relative_error: float
    preserve_nan: bool
    preserve_infinity: bool
    deterministic: bool
    maximum_cases: int


@dataclass(frozen=True, slots=True)
class OperatorCounterexample:
    case_index: int
    seed: int
    shape: tuple[int, ...]
    dtype: str
    stride_variant: str
    violation: str
    expected_digest: str
    observed_digest: str
    minimized_values: tuple[float | int | bool, ...]


@dataclass(frozen=True, slots=True)
class OperatorVerificationResult:
    claim: str
    level: VerificationLevel
    status: EvidenceStatus
    seed: int
    cases_executed: int
    domain: OperatorContract
    counterexample: OperatorCounterexample | None
    assumptions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class QualityContract:
    exact_token_match_minimum: float
    top1_agreement_minimum: float
    topk: int
    topk_agreement_minimum: float
    maximum_kl_divergence: float
    maximum_js_divergence: float
    maximum_absolute_error: float


@dataclass(frozen=True, slots=True)
class QualityEvidence:
    status: EvidenceStatus
    sample_count: int
    exact_token_match: float
    top1_agreement: float
    topk_agreement: float
    kl_divergence: float
    js_divergence: float
    maximum_absolute_error: float
    violations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ResourceContract:
    device_capacity_bytes: int
    host_capacity_bytes: int
    safety_margin_fraction: float
    fragmentation_fraction: float
    maximum_processes: int
    maximum_threads: int
    maximum_file_descriptors: int


@dataclass(frozen=True, slots=True)
class ResourceDemand:
    model_device_bytes: int
    state_device_bytes: int
    queue_device_bytes: int
    communication_device_bytes: int
    workspace_device_bytes: int
    challenger_device_bytes: int
    host_bytes: int
    conversion_overlap_bytes: int
    processes: int
    threads: int
    file_descriptors: int


@dataclass(frozen=True, slots=True)
class ResourceEvidence:
    status: EvidenceStatus
    conservative_peak_device_bytes: int
    usable_device_bytes: int
    conservative_peak_host_bytes: int
    usable_host_bytes: int
    violations: tuple[str, ...]
    assumptions: tuple[str, ...]


class MetricDirection(StrEnum):
    LOWER_IS_BETTER = "lower_is_better"
    HIGHER_IS_BETTER = "higher_is_better"


@dataclass(frozen=True, slots=True)
class BenchmarkContract:
    benchmark_id: str
    metric: str
    unit: str
    direction: MetricDirection
    workload_fingerprint: str
    hardware_fingerprint: str
    software_manifest_hash: str
    warmup_count: int
    practical_significance_percent: float
    noise_floor_percent: float
    bootstrap_rounds: int
    confidence: float


@dataclass(frozen=True, slots=True)
class PerformanceEvidence:
    status: EvidenceStatus
    seed: int
    contract: BenchmarkContract
    baseline_samples: tuple[float, ...]
    candidate_samples: tuple[float, ...]
    run_order: tuple[str, ...]
    baseline_median: float
    candidate_median: float
    improvement_percent: float
    interval_low_percent: float
    interval_high_percent: float
    effect_size: float
    rationale: str


class VerificationError(ValueError):
    """Raised when evidence inputs are malformed or outside declared scope."""
