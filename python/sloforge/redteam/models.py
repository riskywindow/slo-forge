"""Strict executable red-team cases, findings, audit inputs, and reports."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Annotated, Literal, Self, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from sloforge.genesis.ir import (
    Counterexample,
    LearnedConstraint,
    Precision,
    RequestEventCase,
    ResourceCase,
    TensorInputCase,
    TopologyCase,
)

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")]
Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]


class RedTeamModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
        allow_inf_nan=False,
    )


class RedTeamSurface(StrEnum):
    TENSOR = "tensor"
    PROTOCOL = "protocol"
    TOPOLOGY = "topology"
    RESOURCE = "resource"
    BENCHMARK = "benchmark_integrity"


class TensorAdversarialCase(RedTeamModel):
    kind: Literal["tensor"] = "tensor"
    case_id: Identifier
    input: TensorInputCase


class ScheduleAdversarialCase(RedTeamModel):
    kind: Literal["schedule"] = "schedule"
    case_id: Identifier
    events: tuple[RequestEventCase, ...]

    @model_validator(mode="after")
    def bounded_contiguous_schedule(self) -> Self:
        if not self.events or len(self.events) > 256:
            raise ValueError("schedule must contain between 1 and 256 events")
        if tuple(item.at_step for item in self.events) != tuple(range(len(self.events))):
            raise ValueError("schedule steps must be contiguous from zero")
        return self


class TopologyAdversarialCase(RedTeamModel):
    kind: Literal["topology"] = "topology"
    case_id: Identifier
    topology: TopologyCase


class ResourceAdversarialCase(RedTeamModel):
    kind: Literal["resource"] = "resource"
    case_id: Identifier
    resource: ResourceCase


class BenchmarkAuditCase(RedTeamModel):
    kind: Literal["benchmark"] = "benchmark"
    case_id: Identifier
    issue_code: NonEmpty
    baseline_run_id: Identifier
    candidate_run_id: Identifier


AdversarialCase: TypeAlias = Annotated[
    TensorAdversarialCase
    | ScheduleAdversarialCase
    | TopologyAdversarialCase
    | ResourceAdversarialCase
    | BenchmarkAuditCase,
    Field(discriminator="kind"),
]


class TensorAdversaryConfiguration(RedTeamModel):
    seed: NonNegativeInt
    maximum_cases: Annotated[int, Field(ge=1, le=10_000)] = 64
    maximum_rank: Annotated[int, Field(ge=1, le=8)] = 4
    maximum_dimension: Annotated[int, Field(ge=1, le=65_536)] = 257


class ScheduleAdversaryConfiguration(RedTeamModel):
    seed: NonNegativeInt
    maximum_cases: Annotated[int, Field(ge=1, le=10_000)] = 64
    maximum_events: Annotated[int, Field(ge=8, le=256)] = 32
    request_count: Annotated[int, Field(ge=1, le=64)] = 4
    worker_count: Annotated[int, Field(ge=1, le=64)] = 2


class TopologyAdversaryConfiguration(RedTeamModel):
    seed: NonNegativeInt
    maximum_cases: Annotated[int, Field(ge=1, le=10_000)] = 32
    maximum_hosts: Annotated[int, Field(ge=1, le=128)] = 4
    maximum_devices_per_host: Annotated[int, Field(ge=1, le=64)] = 8


class ResourceAdversaryConfiguration(RedTeamModel):
    seed: NonNegativeInt
    maximum_cases: Annotated[int, Field(ge=1, le=10_000)] = 32
    maximum_device_bytes: PositiveInt = 1 << 40
    maximum_host_bytes: PositiveInt = 1 << 42
    maximum_queue_depth: PositiveInt = 1_000_000
    maximum_process_count: PositiveInt = 4096


class TargetDescriptor(RedTeamModel):
    candidate_id: Identifier
    transformation_id: Identifier | None
    description: NonEmpty
    queue_capacity: PositiveInt
    device_capacity_bytes: PositiveInt
    host_capacity_bytes: PositiveInt
    process_limit: PositiveInt
    selected_links: tuple[NonEmpty, ...]


class ViolationObservation(RedTeamModel):
    surface: RedTeamSurface
    violated_contract: NonEmpty
    expected_behavior: NonEmpty
    observed_behavior: NonEmpty
    learned_precondition: NonEmpty


class RedTeamFinding(RedTeamModel):
    finding_id: Identifier
    candidate_id: Identifier
    surface: RedTeamSurface
    original_case: AdversarialCase
    minimized_case: AdversarialCase
    minimization_evaluations: NonNegativeInt
    observation: ViolationObservation
    counterexample: Counterexample
    learned_constraint: LearnedConstraint

    @model_validator(mode="after")
    def matching_surfaces(self) -> Self:
        case_surface = {
            "tensor": RedTeamSurface.TENSOR,
            "schedule": RedTeamSurface.PROTOCOL,
            "topology": RedTeamSurface.TOPOLOGY,
            "resource": RedTeamSurface.RESOURCE,
            "benchmark": RedTeamSurface.BENCHMARK,
        }
        if case_surface[self.original_case.kind] is not self.surface:
            raise ValueError("finding surface does not match original case")
        if self.original_case.kind != self.minimized_case.kind:
            raise ValueError("minimization cannot change case kind")
        return self


class SurfaceCount(RedTeamModel):
    surface: RedTeamSurface
    evaluated: NonNegativeInt
    violations: NonNegativeInt


class RedTeamConfiguration(RedTeamModel):
    seed: NonNegativeInt
    maximum_findings: Annotated[int, Field(ge=1, le=10_000)] = 32
    maximum_minimization_evaluations: Annotated[int, Field(ge=1, le=100_000)] = 256
    minimization_timeout_seconds: Annotated[float, Field(gt=0.0, le=600.0)] = 5.0
    run_timeout_seconds: Annotated[float, Field(gt=0.0, le=3600.0)] = 30.0
    tensor_cases: Annotated[int, Field(ge=1, le=10_000)] = 32
    schedule_cases: Annotated[int, Field(ge=1, le=10_000)] = 32
    topology_cases: Annotated[int, Field(ge=1, le=10_000)] = 16
    resource_cases: Annotated[int, Field(ge=1, le=10_000)] = 16


class RedTeamReport(RedTeamModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    seed: NonNegativeInt
    target: TargetDescriptor
    counts: tuple[SurfaceCount, ...]
    findings: tuple[RedTeamFinding, ...]
    timed_out: bool
    evaluation_steps: NonNegativeInt


class RegressionCase(RedTeamModel):
    regression_id: Identifier
    candidate_id: Identifier
    violated_contract: NonEmpty
    case: AdversarialCase
    counterexample_id: Identifier
    seed: NonNegativeInt


class RegressionCorpus(RedTeamModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    seed: NonNegativeInt
    cases: tuple[RegressionCase, ...]
    constraints: tuple[LearnedConstraint, ...]
    corpus_digest: Digest

    @model_validator(mode="after")
    def content_digest_matches(self) -> Self:
        content = self.model_dump(mode="json", exclude={"corpus_digest"})
        payload = json.dumps(
            content,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
        if hashlib.sha256(payload).hexdigest() != self.corpus_digest:
            raise ValueError("regression corpus digest does not match its content")
        return self


class TimerKind(StrEnum):
    MONOTONIC = "monotonic"
    CUDA_EVENT = "cuda_event"
    WALL_CLOCK = "wall_clock"


class CacheRegime(StrEnum):
    COLD = "cold"
    WARM = "warm"
    MIXED = "mixed"


class BenchmarkRunManifest(RedTeamModel):
    run_id: Identifier
    candidate_id: Identifier
    benchmark_definition_hash: Digest
    input_fingerprints: tuple[Digest, ...]
    synchronized: bool
    timer_kind: TimerKind
    warmup_iterations: NonNegativeInt
    cache_regime: CacheRegime
    cache_reset_between_trials: bool
    fallback_invocations: NonNegativeInt
    precision: Precision
    quality_contract_hash: Digest
    quality_score: Annotated[float, Field(ge=0.0, le=1.0)]
    failures_included: bool
    hardware_clock_mhz: Annotated[int, Field(gt=0)]
    cpu_affinity: tuple[NonNegativeInt, ...]
    background_processes: tuple[NonEmpty, ...]
    raw_samples: tuple[Annotated[float, Field(ge=0.0)], ...]
    discarded_sample_indices: tuple[NonNegativeInt, ...] = ()

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        if not self.input_fingerprints or not self.raw_samples:
            raise ValueError("benchmark manifest requires inputs and raw samples")
        if len(self.cpu_affinity) != len(set(self.cpu_affinity)):
            raise ValueError("CPU affinity entries must be unique")
        if any(index >= len(self.raw_samples) for index in self.discarded_sample_indices):
            raise ValueError("discarded sample index is outside raw samples")
        return self


class BenchmarkComparison(RedTeamModel):
    baseline: BenchmarkRunManifest
    candidate: BenchmarkRunManifest
    required_precision: Precision
    quality_contract_hash: Digest
    minimum_quality_score: Annotated[float, Field(ge=0.0, le=1.0)]
    maximum_clock_delta_mhz: NonNegativeInt = 15
    allowed_background_processes: tuple[NonEmpty, ...] = ()


class BenchmarkIntegrityCode(StrEnum):
    MISSING_SYNCHRONIZATION = "missing_synchronization"
    TIMER_MISUSE = "timer_misuse"
    WARMUP_MISMATCH = "warmup_mismatch"
    INPUT_DISTRIBUTION_MISMATCH = "input_distribution_mismatch"
    CACHE_CONTAMINATION = "cache_contamination"
    HIDDEN_FALLBACK = "hidden_fallback"
    PRECISION_MISMATCH = "precision_mismatch"
    QUALITY_MISMATCH = "quality_mismatch"
    OMITTED_FAILURES = "omitted_failures"
    HARDWARE_CLOCK_CHANGE = "hardware_clock_change"
    CPU_AFFINITY_CHANGE = "cpu_affinity_change"
    BACKGROUND_PROCESS_CHANGE = "background_process_change"
    DISCARDED_SLOW_SAMPLES = "discarded_slow_samples"
    BENCHMARK_DEFINITION_MISMATCH = "benchmark_definition_mismatch"


class BenchmarkIntegrityIssue(RedTeamModel):
    code: BenchmarkIntegrityCode
    message: NonEmpty
    baseline_run_id: Identifier
    candidate_run_id: Identifier


class RegressionReplayResult(RedTeamModel):
    regression_id: Identifier
    reproduced: bool
    observed_contract: NonEmpty | None


class RedTeamDemoResult(RedTeamModel):
    seed: NonNegativeInt
    report_path: NonEmpty
    corpus_path: NonEmpty
    counterexample_paths: tuple[NonEmpty, ...]
    constraint_paths: tuple[NonEmpty, ...]
    finding_count: NonNegativeInt
    reproduced_regressions: NonNegativeInt


__all__ = [
    "AdversarialCase",
    "BenchmarkAuditCase",
    "BenchmarkComparison",
    "BenchmarkIntegrityCode",
    "BenchmarkIntegrityIssue",
    "BenchmarkRunManifest",
    "CacheRegime",
    "RedTeamConfiguration",
    "RedTeamDemoResult",
    "RedTeamFinding",
    "RedTeamModel",
    "RedTeamReport",
    "RedTeamSurface",
    "RegressionCase",
    "RegressionCorpus",
    "RegressionReplayResult",
    "ResourceAdversarialCase",
    "ResourceAdversaryConfiguration",
    "ScheduleAdversarialCase",
    "ScheduleAdversaryConfiguration",
    "SurfaceCount",
    "TargetDescriptor",
    "TensorAdversarialCase",
    "TensorAdversaryConfiguration",
    "TimerKind",
    "TopologyAdversarialCase",
    "TopologyAdversaryConfiguration",
    "ViolationObservation",
]
