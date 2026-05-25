from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sloforge.trace import TraceRequest
from sloforge.util import environment_manifest, percentile, sha256_file, utc_now, write_json


class BackendCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(min_length=1)
    runtime: str = Field(min_length=1)
    runtime_version: str
    hardware_id: str
    dtype: Literal["float32", "float16", "bfloat16", "int8", "int4"] = "float16"
    hourly_price_usd: float = Field(ge=0)
    startup_ms: float = Field(gt=0)
    startup_jitter: float = Field(ge=0, le=1)
    prefill_base_ms: float = Field(ge=0)
    prefill_ms_per_token: float = Field(gt=0)
    decode_base_ms: float = Field(ge=0)
    decode_ms_per_active_sequence: float = Field(gt=0)
    max_concurrency: int = Field(ge=1)
    memory_bytes: int = Field(gt=0)
    failure_rate: float = Field(ge=0, lt=1)
    model_parameter_count: int = Field(gt=0)
    max_sequence_length: int = Field(ge=128)


class ProfilingBudget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_duration_s: float = Field(gt=0)
    max_cost_usd: float = Field(ge=0)
    started_monotonic_s: float = Field(default_factory=time.monotonic, exclude=True)
    spent_cost_usd: float = 0.0
    measured_seconds: float = 0.0

    def reserve(self, *, duration_s: float, hourly_price_usd: float) -> None:
        if duration_s < 0 or hourly_price_usd < 0:
            raise ValueError("profiling reservations cannot be negative")
        projected_cost = self.spent_cost_usd + duration_s / 3600.0 * hourly_price_usd
        projected_duration = self.measured_seconds + duration_s
        if projected_duration > self.max_duration_s + 1e-9:
            raise RuntimeError(
                f"profiling duration budget exhausted: {projected_duration:.3f}s > {self.max_duration_s:.3f}s"
            )
        if projected_cost > self.max_cost_usd + 1e-12:
            raise RuntimeError(
                f"profiling cost budget exhausted: ${projected_cost:.6f} > ${self.max_cost_usd:.6f}"
            )
        self.measured_seconds = projected_duration
        self.spent_cost_usd = projected_cost


class RawMeasurement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    measurement_id: str
    candidate_id: str
    stage: Literal["startup", "prefill", "decode", "load"]
    sample_index: int = Field(ge=0)
    warmup: bool
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    batch_size: int | None = None
    active_sequences: int | None = None
    latency_ms: float = Field(ge=0)
    ttft_ms: float | None = Field(default=None, ge=0)
    itl_ms: float | None = Field(default=None, ge=0)
    e2e_ms: float | None = Field(default=None, ge=0)
    peak_memory_bytes: int | None = Field(default=None, ge=0)
    failed: bool = False
    seed: int


class CandidateProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate: BackendCandidate
    feasible: bool
    rejection_reason: str | None = None
    estimated_weight_bytes: int
    estimated_kv_bytes_per_token: int
    raw_measurement_ids: list[str]
    summaries: dict[str, float]

    @model_validator(mode="after")
    def rejection_is_consistent(self) -> CandidateProfile:
        if self.feasible == (self.rejection_reason is not None):
            raise ValueError(
                "feasible profiles cannot have a rejection reason and rejected ones must"
            )
        return self


class ProfileBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["sloforge.profile/v1"] = "sloforge.profile/v1"
    profile_id: str
    generated_at: str
    seed: int
    workload_sha256: str
    hardware_sha256: str
    budget: ProfilingBudget
    candidates: list[CandidateProfile]
    raw_measurements: list[RawMeasurement]
    environment: dict[str, object]


def _memory_estimate(
    candidate: BackendCandidate, *, max_batch_tokens: int = 8192
) -> tuple[int, int]:
    bytes_per_parameter = {"float32": 4, "float16": 2, "bfloat16": 2, "int8": 1, "int4": 1}[
        candidate.dtype
    ]
    weight_bytes = candidate.model_parameter_count * bytes_per_parameter
    # A conservative architecture-independent KV estimate for feasibility pruning.
    kv_bytes_per_token = max(256, int(math.sqrt(candidate.model_parameter_count) * 32))
    return weight_bytes, kv_bytes_per_token * max_batch_tokens


def _jittered(rng: random.Random, center: float, relative_sigma: float = 0.04) -> float:
    return max(0.001, rng.lognormvariate(math.log(max(center, 0.001)), relative_sigma))


def profile_mock_candidates(
    *,
    candidates: list[BackendCandidate],
    trace: list[TraceRequest],
    trace_path: Path,
    hardware_path: Path,
    budget: ProfilingBudget,
    seed: int,
    output_dir: Path,
) -> ProfileBundle:
    """Run deterministic multi-fidelity profiling for explicit mock backends."""
    if not candidates:
        raise ValueError("profiling requires at least one candidate")
    rng = random.Random(seed)
    measurements: list[RawMeasurement] = []
    profiles: list[CandidateProfile] = []
    prompt_grid = sorted(
        {32, 128, 512, 2048, int(percentile([float(r.prompt_tokens) for r in trace], 0.95))}
    )
    active_grid = [1, 2, 4, 8]
    for candidate_index, candidate in enumerate(candidates):
        weight_bytes, kv_bytes = _memory_estimate(candidate)
        required = int((weight_bytes + kv_bytes) * 1.10)
        if required > candidate.memory_bytes:
            profiles.append(
                CandidateProfile(
                    candidate=candidate,
                    feasible=False,
                    rejection_reason=(
                        f"static memory estimate {required} exceeds capacity {candidate.memory_bytes}"
                    ),
                    estimated_weight_bytes=weight_bytes,
                    estimated_kv_bytes_per_token=kv_bytes // 8192,
                    raw_measurement_ids=[],
                    summaries={},
                )
            )
            continue
        candidate_measurements: list[RawMeasurement] = []
        # Stage C: startup. Two warmups are retained and explicitly labelled.
        for sample_index in range(9):
            budget.reserve(duration_s=0.003, hourly_price_usd=candidate.hourly_price_usd)
            latency = _jittered(rng, candidate.startup_ms, max(candidate.startup_jitter, 0.001))
            candidate_measurements.append(
                RawMeasurement(
                    measurement_id=f"{candidate.candidate_id}-startup-{sample_index}",
                    candidate_id=candidate.candidate_id,
                    stage="startup",
                    sample_index=sample_index,
                    warmup=sample_index < 2,
                    latency_ms=latency,
                    seed=seed,
                )
            )
        # Stage D: prefill and decode curves.
        for prompt in prompt_grid:
            for sample_index in range(7):
                budget.reserve(duration_s=0.002, hourly_price_usd=candidate.hourly_price_usd)
                center = candidate.prefill_base_ms + prompt * candidate.prefill_ms_per_token
                latency = _jittered(rng, center, 0.035 + prompt / 150_000)
                candidate_measurements.append(
                    RawMeasurement(
                        measurement_id=f"{candidate.candidate_id}-prefill-{prompt}-{sample_index}",
                        candidate_id=candidate.candidate_id,
                        stage="prefill",
                        sample_index=sample_index,
                        warmup=sample_index == 0,
                        prompt_tokens=prompt,
                        batch_size=1,
                        latency_ms=latency,
                        peak_memory_bytes=weight_bytes + prompt * (kv_bytes // 8192),
                        seed=seed,
                    )
                )
        for active in active_grid:
            if active > candidate.max_concurrency:
                continue
            for sample_index in range(7):
                budget.reserve(duration_s=0.002, hourly_price_usd=candidate.hourly_price_usd)
                center = candidate.decode_base_ms + active * candidate.decode_ms_per_active_sequence
                latency = _jittered(rng, center, 0.045)
                candidate_measurements.append(
                    RawMeasurement(
                        measurement_id=f"{candidate.candidate_id}-decode-{active}-{sample_index}",
                        candidate_id=candidate.candidate_id,
                        stage="decode",
                        sample_index=sample_index,
                        warmup=sample_index == 0,
                        active_sequences=active,
                        batch_size=active,
                        latency_ms=latency,
                        seed=seed,
                    )
                )
        # Stage E: held-out representative load measurements.
        sampled_trace = trace[candidate_index :: max(1, len(candidates))][:36]
        for sample_index, request in enumerate(sampled_trace):
            budget.reserve(duration_s=0.003, hourly_price_usd=candidate.hourly_price_usd)
            active = 1 + sample_index % candidate.max_concurrency
            prefill = (
                candidate.prefill_base_ms + request.prompt_tokens * candidate.prefill_ms_per_token
            )
            itl = candidate.decode_base_ms + active * candidate.decode_ms_per_active_sequence
            queue = max(0.0, (active - candidate.max_concurrency * 0.7) * itl * 0.4)
            failed = rng.random() < candidate.failure_rate
            ttft = _jittered(rng, prefill + queue, 0.06)
            e2e = ttft + itl * max(0, request.output_tokens - 1)
            candidate_measurements.append(
                RawMeasurement(
                    measurement_id=f"{candidate.candidate_id}-load-{sample_index}",
                    candidate_id=candidate.candidate_id,
                    stage="load",
                    sample_index=sample_index,
                    warmup=False,
                    prompt_tokens=request.prompt_tokens,
                    output_tokens=request.output_tokens,
                    active_sequences=active,
                    batch_size=active,
                    latency_ms=e2e,
                    ttft_ms=ttft,
                    itl_ms=itl,
                    e2e_ms=e2e,
                    failed=failed,
                    seed=seed,
                )
            )
        non_warm = [sample for sample in candidate_measurements if not sample.warmup]
        startup = [sample.latency_ms for sample in non_warm if sample.stage == "startup"]
        load = [sample for sample in non_warm if sample.stage == "load" and not sample.failed]
        itls = [sample.itl_ms for sample in load if sample.itl_ms is not None]
        ttfts = [sample.ttft_ms for sample in load if sample.ttft_ms is not None]
        e2es = [sample.e2e_ms for sample in load if sample.e2e_ms is not None]
        duration_s = max((trace[-1].arrival_ms - trace[0].arrival_ms) / 1000.0, 0.001)
        successful_tokens = sum(
            sample.output_tokens or 0
            for sample in candidate_measurements
            if sample.stage == "load" and not sample.failed
        )
        summaries = {
            "startup_p95_ms": percentile(startup, 0.95),
            "ttft_p50_ms": percentile(ttfts, 0.50),
            "ttft_p95_ms": percentile(ttfts, 0.95),
            "itl_p99_ms": percentile(itls, 0.99),
            "e2e_p95_ms": percentile(e2es, 0.95),
            "availability": len(load) / max(1, len([s for s in non_warm if s.stage == "load"])),
            "measured_goodput_tokens_s": successful_tokens / duration_s,
        }
        profiles.append(
            CandidateProfile(
                candidate=candidate,
                feasible=True,
                estimated_weight_bytes=weight_bytes,
                estimated_kv_bytes_per_token=kv_bytes // 8192,
                raw_measurement_ids=[sample.measurement_id for sample in candidate_measurements],
                summaries=summaries,
            )
        )
        measurements.extend(candidate_measurements)
    profile_id = f"profile-{seed}-{int(time.time())}"
    bundle = ProfileBundle(
        profile_id=profile_id,
        generated_at=utc_now(),
        seed=seed,
        workload_sha256=sha256_file(trace_path),
        hardware_sha256=sha256_file(hardware_path),
        budget=budget,
        candidates=profiles,
        raw_measurements=measurements,
        environment=environment_manifest(),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "profile.json", bundle.model_dump(exclude={"raw_measurements"}))
    with (output_dir / "measurements.jsonl").open("w", encoding="utf-8") as handle:
        for measurement in measurements:
            handle.write(measurement.model_dump_json(exclude_none=True) + "\n")
    write_json(output_dir / "environment.json", bundle.environment)
    return bundle


def load_profile(directory: Path) -> ProfileBundle:
    envelope_path = directory / "profile.json"
    with envelope_path.open("rb") as handle:
        envelope_payload = handle.read(64 * 1024 * 1024 + 1)
    if len(envelope_payload) > 64 * 1024 * 1024:
        raise ValueError("profile envelope exceeds 64 MiB safety limit")
    envelope = json.loads(envelope_payload)
    if not isinstance(envelope, dict):
        raise ValueError("profile envelope must be a JSON object")
    measurements: list[RawMeasurement] = []
    with (directory / "measurements.jsonl").open("rb") as handle:
        line_number = 0
        while line := handle.readline(1024 * 1024 + 1):
            line_number += 1
            if len(line) > 1024 * 1024:
                raise ValueError(f"measurement record {line_number} exceeds 1 MiB safety limit")
            if len(measurements) >= 1_000_000:
                raise ValueError("profile exceeds one million measurement records")
            if line.strip():
                measurements.append(RawMeasurement.model_validate_json(line))
    envelope["raw_measurements"] = [item.model_dump() for item in measurements]
    return ProfileBundle.model_validate(envelope)
