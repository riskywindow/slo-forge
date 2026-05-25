"""Typed fabric benchmark artifacts with raw-sample provenance."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

NonEmpty = Annotated[str, Field(min_length=1)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class MeasurementMode(StrEnum):
    MEASURED = "measured"
    SYNTHETIC_CALIBRATED = "synthetic_calibrated"
    UNAVAILABLE = "unavailable"


class BenchmarkStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"


class Primitive(StrEnum):
    KERNEL_LAUNCH = "kernel_launch"
    DEVICE_SYNCHRONIZE = "device_synchronize"
    DEVICE_MEMORY = "device_memory_bandwidth"
    GEMM = "gemm"
    PREFILL = "prefill"
    DECODE = "decode"
    HOST_MEMCPY = "host_memcpy"
    H2D_PAGEABLE = "host_to_device_pageable"
    H2D_PINNED = "host_to_device_pinned"
    D2H = "device_to_host"
    GPU_P2P = "gpu_peer_to_peer"
    ALL_REDUCE = "all_reduce"
    ALL_GATHER = "all_gather"
    REDUCE_SCATTER = "reduce_scatter"
    BROADCAST = "broadcast"
    SEND_RECV = "send_receive"
    ALL_TO_ALL = "all_to_all"
    EXPERT_DISPATCH = "expert_dispatch"
    EXPERT_COMBINE = "expert_combine"
    KV_TRANSFER = "kv_transfer"
    STARTUP = "startup"
    GROUP_INITIALIZATION = "communication_group_initialization"


class Direction(StrEnum):
    FORWARD = "forward"
    REVERSE = "reverse"
    BIDIRECTIONAL = "bidirectional"
    NOT_APPLICABLE = "not_applicable"


class Placement(StrictModel):
    hosts: tuple[str, ...]
    ranks: tuple[int, ...]
    gpu_ids: tuple[str, ...]
    numa_domains: tuple[str, ...]
    nic_ids: tuple[str, ...]
    cpu_affinity: tuple[int, ...] = ()


class Invocation(StrictModel):
    adapter: NonEmpty
    adapter_version: str | None
    argv: tuple[str, ...]
    timeout_seconds: float = Field(gt=0.0)
    environment: tuple[tuple[str, str], ...] = ()
    working_directory: str | None = None


class BenchmarkCase(StrictModel):
    case_id: NonEmpty
    primitive: Primitive
    message_bytes: int = Field(ge=0)
    rank_count: int = Field(ge=1)
    concurrency: int = Field(ge=1)
    direction: Direction
    topology_path: tuple[str, ...]
    contention_domains: tuple[str, ...]
    placement: Placement
    warmup_count: int = Field(ge=0)
    sample_count: int = Field(ge=1)
    invocation: Invocation


class RawSample(StrictModel):
    sample_index: int = Field(ge=0)
    duration_microseconds: float = Field(ge=0.0)
    throughput_bytes_per_second: float | None = Field(default=None, ge=0.0)
    synthetic: bool
    seed: int | None


class RobustSummary(StrictModel):
    sample_count: int = Field(ge=1)
    median_microseconds: float = Field(ge=0.0)
    p95_microseconds: float = Field(ge=0.0)
    p99_microseconds: float = Field(ge=0.0)
    median_absolute_deviation_microseconds: float = Field(ge=0.0)
    confidence_level: float = Field(gt=0.0, lt=1.0)
    median_ci_low_microseconds: float = Field(ge=0.0)
    median_ci_high_microseconds: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if self.median_ci_low_microseconds > self.median_ci_high_microseconds:
            raise ValueError("confidence interval bounds are inverted")
        return self


class EnvironmentFact(StrictModel):
    name: NonEmpty
    value: str | int | float | bool | None
    source: NonEmpty


class BenchmarkResult(StrictModel):
    schema_version: Literal["sloforge.fabric.benchmark-result/v1"] = (
        "sloforge.fabric.benchmark-result/v1"
    )
    case: BenchmarkCase
    mode: MeasurementMode
    status: BenchmarkStatus
    raw_samples: tuple[RawSample, ...]
    summary: RobustSummary | None
    environment: tuple[EnvironmentFact, ...]
    failure_reason: str | None = None
    raw_artifact: str | None = None
    artifact_hash: str

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if self.status is BenchmarkStatus.SUCCESS:
            if len(self.raw_samples) != self.case.sample_count or self.summary is None:
                raise ValueError("successful benchmark must retain every requested raw sample")
            if self.failure_reason is not None:
                raise ValueError("successful benchmark cannot have a failure reason")
        elif self.raw_samples or self.summary is not None or not self.failure_reason:
            raise ValueError("failed/unavailable benchmark requires a reason and no measurements")
        if result_hash(self) != self.artifact_hash:
            raise ValueError("benchmark artifact hash mismatch")
        return self


class FabricProfile(StrictModel):
    schema_version: Literal["sloforge.fabric.profile/v1"] = "sloforge.fabric.profile/v1"
    profile_id: NonEmpty
    captured_at: NonEmpty
    topology_fingerprint: NonEmpty
    seed: int
    suite: NonEmpty
    results: tuple[BenchmarkResult, ...]
    environment: tuple[EnvironmentFact, ...]
    profile_hash: str

    @model_validator(mode="after")
    def validate_hash(self) -> Self:
        if profile_hash(self) != self.profile_hash:
            raise ValueError("fabric profile hash mismatch")
        return self


def _canonical(value: BaseModel, omitted: set[str]) -> bytes:
    payload = value.model_dump(mode="json", exclude=omitted)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def result_hash(result: BenchmarkResult) -> str:
    return hashlib.sha256(_canonical(result, {"artifact_hash", "raw_artifact"})).hexdigest()


def profile_hash(profile: FabricProfile) -> str:
    return hashlib.sha256(_canonical(profile, {"profile_hash", "captured_at"})).hexdigest()


def finalize_result(**values: object) -> BenchmarkResult:
    provisional = BenchmarkResult.model_construct(artifact_hash="", **values)
    payload = provisional.model_dump(mode="json")
    payload["artifact_hash"] = result_hash(provisional)
    return BenchmarkResult.model_validate(payload)


def finalize_profile(**values: object) -> FabricProfile:
    provisional = FabricProfile.model_construct(profile_hash="", **values)
    payload = provisional.model_dump(mode="json")
    payload["profile_hash"] = profile_hash(provisional)
    return FabricProfile.model_validate(payload)
