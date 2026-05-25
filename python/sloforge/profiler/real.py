from __future__ import annotations

import concurrent.futures
import hashlib
import importlib
import importlib.metadata
import json
import math
import random
import re
import shutil
import statistics
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from sloforge.adapters.engines import get_engine_adapter
from sloforge.hardware.probe import ProbeResult
from sloforge.ir import LicenseMetadata, ModelArchitecture
from sloforge.profiler.core import (
    BackendCandidate,
    CandidateProfile,
    ProfileBundle,
    ProfilingBudget,
    RawMeasurement,
)
from sloforge.profiler.gpu_tools import (
    ManagedEngineServer,
    OpenAIStreamTiming,
    build_nsight_systems_command,
    cuda_subprocess_environment,
    ensure_cuda_requested,
    gpu_environment,
    stream_openai_completion,
    torch_perfetto_trace,
)
from sloforge.trace import TraceRequest
from sloforge.util import (
    canonical_json,
    environment_manifest,
    percentile,
    sha256_bytes,
    sha256_file,
    utc_now,
    write_json,
)

T = TypeVar("T")
ProfileDType = Literal["float32", "float16", "bfloat16", "int8", "int4"]


class RealProbeCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1, max_length=256)
    prompt: str = Field(min_length=1)
    prompt_tokens_hint: int = Field(ge=1)
    max_new_tokens: int = Field(ge=1, le=65_536)


class RealProfilerSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    model_revision: str | None = None
    device: Literal["cuda"] = "cuda"
    device_index: int = Field(default=0, ge=0)
    host: Literal["127.0.0.1", "localhost"] = "127.0.0.1"
    base_port: int = Field(default=18_100, ge=1024, le=65_000)
    readiness_timeout_s: float = Field(default=300.0, gt=0)
    request_timeout_s: float = Field(default=120.0, gt=0)
    shutdown_timeout_s: float = Field(default=10.0, gt=0)
    warmup_requests: int = Field(default=2, ge=1)
    measured_requests: int = Field(default=12, ge=3, le=10_000)
    load_concurrency: int = Field(default=4, ge=1, le=256)
    temperature: float = Field(default=0.0, ge=0)
    local_files_only: bool = False
    export_perfetto: bool = True
    max_response_bytes: int = Field(default=64 << 20, ge=1024)
    max_server_log_bytes: int = Field(default=1 << 20, ge=1024)
    max_prompt_tokens: int = Field(default=131_072, ge=1, le=1_000_000)

    @model_validator(mode="after")
    def port_range_covers_candidates(self) -> RealProfilerSettings:
        if self.base_port + 64 > 65_535:
            raise ValueError("base_port must leave room for candidate-specific ports")
        return self


class _BudgetMeter:
    def __init__(self, budget: ProfilingBudget, *, hourly_price_usd: float) -> None:
        self.budget = budget
        self.hourly_price_usd = hourly_price_usd

    def remaining_s(self) -> float:
        duration_remaining = self.budget.max_duration_s - self.budget.measured_seconds
        if self.hourly_price_usd == 0:
            return max(0.0, duration_remaining)
        cost_remaining = self.budget.max_cost_usd - self.budget.spent_cost_usd
        return max(0.0, min(duration_remaining, cost_remaining * 3600 / self.hourly_price_usd))

    def timeout(self, requested_s: float) -> float:
        remaining = self.remaining_s()
        if remaining <= 0.001:
            raise RuntimeError("profiling budget exhausted before the next bounded operation")
        return min(requested_s, remaining)

    def measure(self, function: Callable[[], T]) -> T:
        if self.remaining_s() <= 0.001:
            raise RuntimeError("profiling budget exhausted before the next measurement")
        started = time.monotonic()
        result = function()
        elapsed = time.monotonic() - started
        self.budget.reserve(duration_s=elapsed, hourly_price_usd=self.hourly_price_usd)
        return result

    def charge_elapsed(self, elapsed_s: float) -> None:
        self.budget.reserve(duration_s=elapsed_s, hourly_price_usd=self.hourly_price_usd)


def build_real_probe_cases(
    trace: Sequence[TraceRequest],
    *,
    seed: int,
    warmup_requests: int,
    measured_requests: int,
    max_prompt_tokens: int = 131_072,
) -> list[RealProbeCase]:
    """Select deterministic trace coverage and materialize bounded synthetic text prompts."""
    requested = warmup_requests + measured_requests
    if len(trace) < requested:
        raise ValueError(
            f"real profiling needs at least {requested} trace requests, got {len(trace)}"
        )
    rng = random.Random(seed)
    selected_indices = sorted(rng.sample(range(len(trace)), requested))
    cases: list[RealProbeCase] = []
    for index in selected_indices:
        request = trace[index]
        if request.prompt_tokens > max_prompt_tokens:
            raise RuntimeError(
                f"request {request.request_id} asks for {request.prompt_tokens} prompt tokens, "
                f"above profiler materialization limit {max_prompt_tokens}"
            )
        # The hint remains the source trace count. Actual tokenizer counts are always measured.
        prompt = (f"sloforge_{request.request_class} " * request.prompt_tokens).strip()
        cases.append(
            RealProbeCase(
                request_id=request.request_id,
                prompt=prompt,
                prompt_tokens_hint=request.prompt_tokens,
                max_new_tokens=request.output_tokens,
            )
        )
    return cases


def _memory_estimate(
    candidate: BackendCandidate, *, max_batch_tokens: int = 8192
) -> tuple[int, int]:
    bytes_per_parameter = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
        "int8": 1,
        "int4": 1,
    }[candidate.dtype]
    weight_bytes = candidate.model_parameter_count * bytes_per_parameter
    kv_bytes_per_token = max(256, int(math.sqrt(candidate.model_parameter_count) * 32))
    return weight_bytes, kv_bytes_per_token * max_batch_tokens


def _aggregate_itl(timing: OpenAIStreamTiming) -> float:
    timestamps = timing.token_timestamps_ms
    if len(timestamps) > 1 and len(timestamps) == timing.output_tokens:
        return statistics.median(
            timestamps[index] - timestamps[index - 1] for index in range(1, len(timestamps))
        )
    return max(
        0.0,
        (timing.e2e_ms - timing.ttft_ms) / max(1, timing.output_tokens - 1),
    )


def _measurements_for_timing(
    *,
    candidate_id: str,
    sample_index: int,
    warmup: bool,
    prompt_tokens: int,
    timing: OpenAIStreamTiming,
    active_sequences: int,
    seed: int,
    peak_memory_bytes: int | None = None,
) -> list[RawMeasurement]:
    prefix = f"{candidate_id}-real-{sample_index}"
    itl_ms = _aggregate_itl(timing)
    return [
        RawMeasurement(
            measurement_id=f"{prefix}-prefill",
            candidate_id=candidate_id,
            stage="prefill",
            sample_index=sample_index,
            warmup=warmup,
            prompt_tokens=prompt_tokens,
            batch_size=1,
            latency_ms=timing.ttft_ms,
            peak_memory_bytes=peak_memory_bytes,
            seed=seed,
        ),
        RawMeasurement(
            measurement_id=f"{prefix}-decode",
            candidate_id=candidate_id,
            stage="decode",
            sample_index=sample_index,
            warmup=warmup,
            output_tokens=timing.output_tokens,
            batch_size=active_sequences,
            active_sequences=active_sequences,
            latency_ms=itl_ms,
            peak_memory_bytes=peak_memory_bytes,
            seed=seed,
        ),
        RawMeasurement(
            measurement_id=f"{prefix}-load",
            candidate_id=candidate_id,
            stage="load",
            sample_index=sample_index,
            warmup=warmup,
            prompt_tokens=prompt_tokens,
            output_tokens=timing.output_tokens,
            batch_size=active_sequences,
            active_sequences=active_sequences,
            latency_ms=timing.e2e_ms,
            ttft_ms=timing.ttft_ms,
            itl_ms=itl_ms,
            e2e_ms=timing.e2e_ms,
            peak_memory_bytes=peak_memory_bytes,
            seed=seed,
        ),
    ]


def _torch_dtype(torch: Any, dtype: str) -> Any:
    values = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    result = values.get(dtype)
    if result is None:
        raise RuntimeError(
            f"Transformers correctness profiling does not implement requested dtype {dtype!r}"
        )
    return result


def _profile_transformers(
    *,
    candidate: BackendCandidate,
    cases: Sequence[RealProbeCase],
    settings: RealProfilerSettings,
    budget: ProfilingBudget,
    seed: int,
    output_dir: Path,
) -> list[RawMeasurement]:
    adapter = get_engine_adapter("transformers")
    adapter.validate_available()
    torch = ensure_cuda_requested(device=settings.device, index=settings.device_index)
    meter = _BudgetMeter(budget, hourly_price_usd=candidate.hourly_price_usd)
    transformers = importlib.import_module("transformers")
    auto_model: Any = transformers.AutoModelForCausalLM
    auto_tokenizer: Any = transformers.AutoTokenizer

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    load_started = time.monotonic()
    tokenizer: Any = auto_tokenizer.from_pretrained(
        settings.model,
        revision=settings.model_revision,
        local_files_only=settings.local_files_only,
        trust_remote_code=False,
    )
    loaded_model: Any = auto_model.from_pretrained(
        settings.model,
        revision=settings.model_revision,
        local_files_only=settings.local_files_only,
        trust_remote_code=False,
        torch_dtype=_torch_dtype(torch, candidate.dtype),
    )
    model: Any = loaded_model.to(f"cuda:{settings.device_index}")
    model.eval()
    torch.cuda.synchronize(settings.device_index)
    startup_s = time.monotonic() - load_started
    meter.charge_elapsed(startup_s)
    measurements = [
        RawMeasurement(
            measurement_id=f"{candidate.candidate_id}-real-startup-0",
            candidate_id=candidate.candidate_id,
            stage="startup",
            sample_index=0,
            warmup=False,
            latency_ms=startup_s * 1000,
            peak_memory_bytes=int(torch.cuda.max_memory_allocated(settings.device_index)),
            seed=seed,
        )
    ]

    def probe(
        case: RealProbeCase, *, trace_output: Path | None = None
    ) -> tuple[OpenAIStreamTiming, int, int]:
        encoded = tokenizer(case.prompt, return_tensors="pt")
        encoded = {key: value.to(f"cuda:{settings.device_index}") for key, value in encoded.items()}
        prompt_tokens = int(encoded["input_ids"].shape[-1])
        if prompt_tokens + case.max_new_tokens > candidate.max_sequence_length:
            raise RuntimeError(
                f"request {case.request_id} has {prompt_tokens + case.max_new_tokens} tokens, "
                f"above candidate limit {candidate.max_sequence_length}"
            )
        torch.cuda.reset_peak_memory_stats(settings.device_index)

        def generate(max_new_tokens: int) -> Any:
            with torch.inference_mode():
                return model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

        torch.cuda.synchronize(settings.device_index)
        first_started = time.perf_counter_ns()
        one_token = generate(1)
        torch.cuda.synchronize(settings.device_index)
        ttft_ms = (time.perf_counter_ns() - first_started) / 1e6
        torch.cuda.synchronize(settings.device_index)
        full_started = time.perf_counter_ns()
        generated = generate(case.max_new_tokens)
        torch.cuda.synchronize(settings.device_index)
        e2e_ms = (time.perf_counter_ns() - full_started) / 1e6
        if trace_output is not None:
            with torch_perfetto_trace(
                output=trace_output, device=settings.device, index=settings.device_index
            ) as profile:
                _ = generate(case.max_new_tokens)
                profile.step()
        torch.cuda.synchronize(settings.device_index)
        output_tokens = int(generated.shape[-1] - prompt_tokens)
        if output_tokens < 1:
            raise RuntimeError(f"Transformers emitted no tokens for {case.request_id}")
        if int(one_token[0, prompt_tokens]) != int(generated[0, prompt_tokens]):
            raise RuntimeError(
                f"Transformers deterministic correctness check failed for {case.request_id}: "
                "one-token and full-generation prefixes differ"
            )
        interval = max(0.0, (e2e_ms - ttft_ms) / max(output_tokens - 1, 1))
        timing = OpenAIStreamTiming(
            ttft_ms=ttft_ms,
            e2e_ms=max(e2e_ms, ttft_ms),
            output_tokens=output_tokens,
            prompt_tokens=prompt_tokens,
            token_timestamps_ms=tuple(
                ttft_ms + token_index * interval for token_index in range(output_tokens)
            ),
            event_count=output_tokens,
            response_bytes=int(generated.nelement() * generated.element_size()),
            finish_reason=None,
        )
        return timing, prompt_tokens, int(torch.cuda.max_memory_allocated(settings.device_index))

    try:
        for index, case in enumerate(cases):
            trace_path = None
            if settings.export_perfetto and index == settings.warmup_requests:
                trace_path = output_dir / "traces" / f"{candidate.candidate_id}-torch.json"
            timing, actual_prompt_tokens, peak_memory = meter.measure(
                partial(probe, case, trace_output=trace_path)
            )
            measurements.extend(
                _measurements_for_timing(
                    candidate_id=candidate.candidate_id,
                    sample_index=index,
                    warmup=index < settings.warmup_requests,
                    prompt_tokens=actual_prompt_tokens,
                    timing=timing,
                    active_sequences=1,
                    seed=seed,
                    peak_memory_bytes=peak_memory,
                )
            )
    finally:
        torch.cuda.empty_cache()
    return measurements


def _server_payload(settings: RealProfilerSettings, case: RealProbeCase) -> dict[str, object]:
    return {
        "model": settings.model,
        "prompt": case.prompt,
        "max_tokens": case.max_new_tokens,
        "temperature": settings.temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def _profile_server(
    *,
    candidate: BackendCandidate,
    candidate_index: int,
    cases: Sequence[RealProbeCase],
    settings: RealProfilerSettings,
    budget: ProfilingBudget,
    seed: int,
    output_dir: Path,
    nsight_commands: list[dict[str, object]],
) -> list[RawMeasurement]:
    if settings.model_revision is not None:
        raise RuntimeError(
            "server adapter APIs do not expose a revision flag; pass an immutable local model path "
            "instead of silently ignoring model_revision"
        )
    adapter = get_engine_adapter(candidate.runtime)
    port = settings.base_port + candidate_index
    command = adapter.serve_command(
        model=settings.model,
        host=settings.host,
        port=port,
        dtype=candidate.dtype,
        max_model_len=candidate.max_sequence_length,
    )
    nsight_commands.append(
        {
            "candidate_id": candidate.candidate_id,
            "argv": build_nsight_systems_command(
                command,
                output_prefix=output_dir / "nsight" / candidate.candidate_id,
                require_available=False,
            ),
            "executed": False,
        }
    )
    base_url = f"http://{settings.host}:{port}"
    endpoint = f"{base_url}/v1/completions"
    meter = _BudgetMeter(budget, hourly_price_usd=candidate.hourly_price_usd)
    measurements: list[RawMeasurement] = []
    startup_started = time.monotonic()
    with ManagedEngineServer(
        command,
        env=cuda_subprocess_environment(device_index=settings.device_index),
        max_log_bytes=settings.max_server_log_bytes,
        shutdown_timeout_s=settings.shutdown_timeout_s,
    ) as server:
        server.wait_ready(
            base_url=base_url,
            timeout_s=meter.timeout(settings.readiness_timeout_s),
        )
        startup_s = time.monotonic() - startup_started
        meter.charge_elapsed(startup_s)
        measurements.append(
            RawMeasurement(
                measurement_id=f"{candidate.candidate_id}-real-startup-0",
                candidate_id=candidate.candidate_id,
                stage="startup",
                sample_index=0,
                warmup=False,
                latency_ms=startup_s * 1000,
                seed=seed,
            )
        )

        def invoke(case: RealProbeCase, timeout_s: float) -> OpenAIStreamTiming:
            return stream_openai_completion(
                url=endpoint,
                payload=_server_payload(settings, case),
                timeout_s=timeout_s,
                max_response_bytes=settings.max_response_bytes,
            )

        for index, case in enumerate(cases[: settings.warmup_requests]):
            timeout_s = meter.timeout(settings.request_timeout_s)

            timing = meter.measure(partial(invoke, case, timeout_s))
            if timing.prompt_tokens is None:
                raise RuntimeError(
                    f"engine {candidate.runtime!r} omitted usage.prompt_tokens for "
                    f"{case.request_id}; refusing to label a token curve with an estimate"
                )
            measurements.extend(
                _measurements_for_timing(
                    candidate_id=candidate.candidate_id,
                    sample_index=index,
                    warmup=True,
                    prompt_tokens=timing.prompt_tokens,
                    timing=timing,
                    active_sequences=1,
                    seed=seed,
                )
            )

        measured_cases = list(cases[settings.warmup_requests :])
        concurrency = min(
            settings.load_concurrency,
            candidate.max_concurrency,
            len(measured_cases),
        )
        timeout_s = meter.timeout(settings.request_timeout_s)
        load_started = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(invoke, case, timeout_s): (index, case)
                for index, case in enumerate(measured_cases, start=settings.warmup_requests)
            }
            results: list[tuple[int, RealProbeCase, OpenAIStreamTiming]] = []
            for future in concurrent.futures.as_completed(futures):
                index, case = futures[future]
                results.append((index, case, future.result()))
        meter.charge_elapsed(time.monotonic() - load_started)
        for index, case, timing in sorted(results, key=lambda result: result[0]):
            if timing.prompt_tokens is None:
                raise RuntimeError(
                    f"engine {candidate.runtime!r} omitted usage.prompt_tokens for "
                    f"{case.request_id}; refusing to label a token curve with an estimate"
                )
            measurements.extend(
                _measurements_for_timing(
                    candidate_id=candidate.candidate_id,
                    sample_index=index,
                    warmup=False,
                    prompt_tokens=timing.prompt_tokens,
                    timing=timing,
                    active_sequences=concurrency,
                    seed=seed,
                )
            )
    return measurements


def _summaries(measurements: Sequence[RawMeasurement]) -> dict[str, float]:
    non_warm = [sample for sample in measurements if not sample.warmup]
    startup = [sample.latency_ms for sample in non_warm if sample.stage == "startup"]
    loads = [sample for sample in non_warm if sample.stage == "load" and not sample.failed]
    if not startup or not loads:
        raise RuntimeError("real profiler produced insufficient non-warm startup/load samples")
    ttfts = [sample.ttft_ms for sample in loads if sample.ttft_ms is not None]
    itls = [sample.itl_ms for sample in loads if sample.itl_ms is not None]
    e2es = [sample.e2e_ms for sample in loads if sample.e2e_ms is not None]
    total_tokens = sum(sample.output_tokens or 0 for sample in loads)
    maximum_active = max((sample.active_sequences or 1 for sample in loads), default=1)
    load_span_s = max(sum(e2es) / maximum_active / 1000.0, 1e-9)
    return {
        "startup_p95_ms": percentile(startup, 0.95),
        "ttft_p50_ms": percentile(ttfts, 0.50),
        "ttft_p95_ms": percentile(ttfts, 0.95),
        "itl_p99_ms": percentile(itls, 0.99),
        "e2e_p95_ms": percentile(e2es, 0.95),
        "availability": len(loads)
        / max(1, len([sample for sample in non_warm if sample.stage == "load"])),
        "measured_goodput_tokens_s": total_tokens / load_span_s,
    }


def _candidate_with_measured_curves(
    candidate: BackendCandidate, measurements: Sequence[RawMeasurement]
) -> BackendCandidate:
    usable = [sample for sample in measurements if not sample.warmup and not sample.failed]
    startup = [sample.latency_ms for sample in usable if sample.stage == "startup"]
    prefill = [
        sample
        for sample in usable
        if sample.stage == "prefill" and sample.prompt_tokens is not None
    ]
    decode = [sample for sample in usable if sample.stage == "decode"]
    if not startup or not prefill or not decode:
        raise RuntimeError(
            "cannot calibrate candidate metadata without startup, prefill, and decode"
        )
    x_values = [float(sample.prompt_tokens or 0) for sample in prefill]
    y_values = [sample.latency_ms for sample in prefill]
    x_mean = statistics.mean(x_values)
    y_mean = statistics.mean(y_values)
    denominator = sum((value - x_mean) ** 2 for value in x_values)
    slope = (
        sum(
            (x_value - x_mean) * (y_value - y_mean)
            for x_value, y_value in zip(x_values, y_values, strict=True)
        )
        / denominator
        if denominator > 0
        else y_mean / max(x_mean, 1.0)
    )
    slope = max(slope, 1e-9)
    intercept = max(0.0, y_mean - slope * x_mean)
    decode_per_sequence = statistics.median(
        sample.latency_ms / max(1, sample.active_sequences or 1) for sample in decode
    )
    peak_memory = max(
        (sample.peak_memory_bytes or 0 for sample in measurements),
        default=candidate.memory_bytes,
    )
    return candidate.model_copy(
        update={
            "startup_ms": statistics.median(startup),
            "prefill_base_ms": intercept,
            "prefill_ms_per_token": slope,
            "decode_base_ms": 0.0,
            "decode_ms_per_active_sequence": max(decode_per_sequence, 1e-9),
            "failure_rate": len([sample for sample in usable if sample.failed])
            / max(1, len(usable)),
            "memory_bytes": max(candidate.memory_bytes, peak_memory),
        }
    )


def profile_real_candidates(
    *,
    candidates: Sequence[BackendCandidate],
    trace: Sequence[TraceRequest],
    trace_path: Path,
    hardware_path: Path,
    budget: ProfilingBudget,
    seed: int,
    output_dir: Path,
    settings: RealProfilerSettings,
) -> ProfileBundle:
    """Run an explicitly requested CUDA profile; this function never falls back to mocks or CPU."""
    if not candidates:
        raise ValueError("real profiling requires at least one candidate")
    if len(candidates) > 64:
        raise ValueError("real profiling accepts at most 64 candidates per run")
    ensure_cuda_requested(device=settings.device, index=settings.device_index)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = build_real_probe_cases(
        trace,
        seed=seed,
        warmup_requests=settings.warmup_requests,
        measured_requests=settings.measured_requests,
        max_prompt_tokens=settings.max_prompt_tokens,
    )
    measurements: list[RawMeasurement] = []
    profiles: list[CandidateProfile] = []
    nsight_commands: list[dict[str, object]] = []
    for candidate_index, candidate in enumerate(candidates):
        weight_bytes, kv_bytes = _memory_estimate(candidate)
        required_bytes = int((weight_bytes + kv_bytes) * 1.10)
        if required_bytes > candidate.memory_bytes:
            profiles.append(
                CandidateProfile(
                    candidate=candidate,
                    feasible=False,
                    rejection_reason=(
                        f"static memory estimate {required_bytes} exceeds capacity "
                        f"{candidate.memory_bytes}"
                    ),
                    estimated_weight_bytes=weight_bytes,
                    estimated_kv_bytes_per_token=kv_bytes // 8192,
                    raw_measurement_ids=[],
                    summaries={},
                )
            )
            continue
        if candidate.runtime == "transformers":
            candidate_measurements = _profile_transformers(
                candidate=candidate,
                cases=cases,
                settings=settings,
                budget=budget,
                seed=seed,
                output_dir=output_dir,
            )
        else:
            candidate_measurements = _profile_server(
                candidate=candidate,
                candidate_index=candidate_index,
                cases=cases,
                settings=settings,
                budget=budget,
                seed=seed,
                output_dir=output_dir,
                nsight_commands=nsight_commands,
            )
        measured_candidate = _candidate_with_measured_curves(candidate, candidate_measurements)
        profiles.append(
            CandidateProfile(
                candidate=measured_candidate,
                feasible=True,
                estimated_weight_bytes=weight_bytes,
                estimated_kv_bytes_per_token=kv_bytes // 8192,
                raw_measurement_ids=[item.measurement_id for item in candidate_measurements],
                summaries=_summaries(candidate_measurements),
            )
        )
        measurements.extend(candidate_measurements)

    gpu_manifest = gpu_environment(device=settings.device, index=settings.device_index)
    environment = environment_manifest(include_packages=False)
    environment["gpu"] = gpu_manifest.model_dump(mode="json")
    environment["profiling"] = {
        "timing_clock": "time.perf_counter_ns",
        "cuda_synchronization": "explicit for Transformers; SSE arrival timing for servers",
        "transformers_itl_method": "aggregate decode interval (non-streaming correctness baseline)",
        "server_itl_method": "median SSE content-event interval",
        "warmup_requests": settings.warmup_requests,
        "measured_requests": settings.measured_requests,
        "seed": seed,
    }
    workload_sha256 = sha256_file(trace_path)
    hardware_sha256 = sha256_file(hardware_path)
    identity: dict[str, object] = {
        "seed": seed,
        "model": settings.model,
        "trace_sha256": workload_sha256,
        "hardware_sha256": hardware_sha256,
        "candidates": [candidate.candidate_id for candidate in candidates],
    }
    bundle = ProfileBundle(
        profile_id=f"profile-real-{sha256_bytes(canonical_json(identity).encode())[:16]}",
        generated_at=utc_now(),
        seed=seed,
        workload_sha256=workload_sha256,
        hardware_sha256=hardware_sha256,
        budget=budget,
        candidates=profiles,
        raw_measurements=measurements,
        environment=environment,
    )
    write_json(output_dir / "profile.json", bundle.model_dump(exclude={"raw_measurements"}))
    with (output_dir / "measurements.jsonl").open("w", encoding="utf-8") as handle:
        for measurement in measurements:
            handle.write(measurement.model_dump_json(exclude_none=True) + "\n")
    write_json(output_dir / "environment.json", environment)
    write_json(output_dir / "nsight-commands.json", nsight_commands)
    return bundle


_SUPPORTED_ENGINES = frozenset({"transformers", "vllm", "sglang", "tensorrt-llm"})


def _validate_engine_requests(engines: Sequence[str]) -> dict[str, str]:
    unsupported = sorted(set(engines) - _SUPPORTED_ENGINES)
    if unsupported:
        raise ValueError(
            f"unsupported real engines {unsupported}; expected a subset of {sorted(_SUPPORTED_ENGINES)}"
        )
    if not engines:
        raise ValueError("at least one real engine must be requested")
    if len(set(engines)) != len(engines):
        raise ValueError("real engine list contains duplicates")
    package_names = {
        "transformers": "transformers",
        "vllm": "vllm",
        "sglang": "sglang",
        "tensorrt-llm": "tensorrt_llm",
    }
    versions: dict[str, str] = {}
    for engine in engines:
        package = package_names[engine]
        try:
            versions[engine] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"engine {engine!r} was requested but package {package!r} is not installed"
            ) from exc
    return versions


def _required_gpu_fields(hardware: ProbeResult) -> tuple[int, float, str]:
    if hardware.requested_device != "cuda" or hardware.hardware.gpu is None:
        raise RuntimeError("real engines require an explicit CUDA hardware probe")
    gpu = hardware.hardware.gpu
    vram_mib = gpu.get("vram_mib")
    hourly_price = gpu.get("hourly_price_usd")
    gpu_name = gpu.get("name")
    if not isinstance(vram_mib, int) or isinstance(vram_mib, bool) or vram_mib <= 0:
        raise RuntimeError("CUDA hardware probe must include positive gpu.vram_mib")
    if not isinstance(hourly_price, (int, float)) or isinstance(hourly_price, bool):
        raise RuntimeError(
            "CUDA hardware metadata must explicitly include gpu.hourly_price_usd (zero is valid "
            "for owned hardware); no price fallback is permitted"
        )
    if not math.isfinite(float(hourly_price)) or float(hourly_price) < 0:
        raise RuntimeError("gpu.hourly_price_usd must be finite and nonnegative")
    if not isinstance(gpu_name, str) or not gpu_name.strip():
        raise RuntimeError("CUDA hardware probe must include gpu.name")
    return vram_mib, float(hourly_price), gpu_name


def _directory_checksum(directory: Path) -> str:
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    if not files:
        raise RuntimeError(f"resolved model directory contains no files: {directory}")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(directory).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        file_digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                file_digest.update(chunk)
        digest.update(file_digest.digest())
    return digest.hexdigest()


def _load_bounded_json(path: Path, *, max_bytes: int) -> object:
    with path.open("rb") as handle:
        payload = handle.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise RuntimeError(f"JSON metadata exceeds {max_bytes} byte safety limit: {path}")
    return json.loads(payload)


def _safetensors_model_metadata(directory: Path) -> tuple[int, ProfileDType]:
    index_path = directory / "model.safetensors.index.json"
    if index_path.is_file():
        index = _load_bounded_json(index_path, max_bytes=32 << 20)
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise RuntimeError("model.safetensors.index.json has no non-empty weight_map")
        if not all(isinstance(filename, str) for filename in weight_map.values()):
            raise RuntimeError("safetensors weight_map filenames must be strings")
        filenames = sorted(
            {filename for filename in weight_map.values() if isinstance(filename, str)}
        )
        relative_files = [Path(filename) for filename in filenames]
        if any(
            path.is_absolute() or ".." in path.parts or path.suffix != ".safetensors"
            for path in relative_files
        ):
            raise RuntimeError("safetensors weight_map contains an unsafe weight path")
        files = [directory / path for path in relative_files]
    else:
        files = sorted(directory.glob("model*.safetensors"))
    if not files or any(not path.is_file() for path in files):
        raise RuntimeError(
            "resolved model must include complete safetensors weights so parameter count can be "
            "verified without loading untrusted pickle data"
        )
    parameter_count = 0
    dtype_counts: dict[str, int] = {}
    dtype_bytes = {
        "BOOL": 1,
        "U8": 1,
        "I8": 1,
        "F8_E4M3": 1,
        "F8_E5M2": 1,
        "I16": 2,
        "U16": 2,
        "F16": 2,
        "BF16": 2,
        "I32": 4,
        "U32": 4,
        "F32": 4,
        "I64": 8,
        "U64": 8,
        "F64": 8,
    }
    for path in files:
        with path.open("rb") as handle:
            header_size_raw = handle.read(8)
            if len(header_size_raw) != 8:
                raise RuntimeError(f"invalid safetensors header in {path}")
            header_size = int.from_bytes(header_size_raw, "little")
            if header_size <= 0 or header_size > 128 << 20:
                raise RuntimeError(f"unsafe safetensors header size in {path}: {header_size}")
            header_raw = handle.read(header_size)
        header = json.loads(header_raw)
        if not isinstance(header, dict):
            raise RuntimeError(f"safetensors header in {path} is not an object")
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(tensor, dict):
                raise RuntimeError(f"invalid tensor metadata for {name!r} in {path}")
            shape = tensor.get("shape")
            dtype = tensor.get("dtype")
            offsets = tensor.get("data_offsets")
            if (
                not isinstance(shape, list)
                or not all(isinstance(dimension, int) and dimension >= 0 for dimension in shape)
                or not isinstance(dtype, str)
                or dtype not in dtype_bytes
                or not isinstance(offsets, list)
                or len(offsets) != 2
                or not all(isinstance(offset, int) and offset >= 0 for offset in offsets)
                or offsets[1] < offsets[0]
            ):
                raise RuntimeError(
                    f"invalid shape, dtype, or data offsets for tensor {name!r} in {path}"
                )
            elements = math.prod(shape)
            expected_bytes = elements * dtype_bytes[dtype]
            if offsets[1] - offsets[0] != expected_bytes:
                raise RuntimeError(
                    f"tensor {name!r} byte range disagrees with its shape and dtype in {path}"
                )
            if 8 + header_size + offsets[1] > path.stat().st_size:
                raise RuntimeError(f"tensor {name!r} points beyond end of safetensors file {path}")
            parameter_count += elements
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + elements
    if parameter_count <= 0 or not dtype_counts:
        raise RuntimeError("safetensors weights contain no parameters")
    dominant_dtype = max(dtype_counts, key=dtype_counts.__getitem__)
    dtype_map: dict[str, ProfileDType] = {
        "F32": "float32",
        "F16": "float16",
        "BF16": "bfloat16",
    }
    model_dtype = dtype_map.get(dominant_dtype)
    if model_dtype is None:
        raise RuntimeError(
            f"dominant model storage dtype {dominant_dtype!r} needs an explicit profiler adapter"
        )
    return parameter_count, model_dtype


def _required_architecture_metadata(snapshot: Path, *, parameter_count: int) -> ModelArchitecture:
    config = _load_bounded_json(snapshot / "config.json", max_bytes=8 << 20)
    if not isinstance(config, dict):
        raise RuntimeError("model config.json must be an object")
    text_config = config.get("text_config", config)
    if not isinstance(text_config, dict):
        raise RuntimeError("model text_config must be an object")

    def required_int(*keys: str) -> int:
        value = next((text_config[key] for key in keys if key in text_config), None)
        if not isinstance(value, int) or value <= 0:
            raise RuntimeError(
                "model config is missing positive integer architecture field " + "/".join(keys)
            )
        return value

    architectures = config.get("architectures")
    family = (
        architectures[0]
        if isinstance(architectures, list)
        and architectures
        and isinstance(architectures[0], str)
        and architectures[0]
        else text_config.get("model_type") or config.get("model_type")
    )
    if not isinstance(family, str) or not family:
        raise RuntimeError("model config must declare architectures[0] or model_type")
    attention_heads = required_int("num_attention_heads", "n_head")
    kv_heads_value = text_config.get("num_key_value_heads", attention_heads)
    if not isinstance(kv_heads_value, int) or kv_heads_value <= 0:
        raise RuntimeError("model config num_key_value_heads must be a positive integer")
    return ModelArchitecture(
        family=family,
        parameter_count=parameter_count,
        hidden_size=required_int("hidden_size", "d_model", "n_embd"),
        num_layers=required_int("num_hidden_layers", "n_layer"),
        num_attention_heads=attention_heads,
        num_key_value_heads=kv_heads_value,
        vocabulary_size=required_int("vocab_size"),
    )


def _required_license_metadata(snapshot: Path, *, model: str) -> LicenseMetadata:
    readme = snapshot / "README.md"
    if not readme.is_file():
        raise RuntimeError("model snapshot must include README.md license metadata")
    with readme.open("rb") as handle:
        raw = handle.read((512 << 10) + 1)
    if len(raw) > 512 << 10:
        raise RuntimeError("model README.md exceeds the bounded metadata parser limit")
    text = raw.decode("utf-8")
    if not text.startswith("---"):
        raise RuntimeError("model README.md must have YAML front matter with a license field")
    closing = text.find("\n---", 3)
    if closing < 0:
        raise RuntimeError("model README.md YAML front matter is not terminated")
    document = yaml.safe_load(text[3:closing])
    license_id = document.get("license") if isinstance(document, dict) else None
    if not isinstance(license_id, str):
        raise RuntimeError("model card must declare a single SPDX-compatible license identifier")
    normalized = license_id.lower()
    supported = {
        "apache-2.0": ("Apache-2.0", "Apache License 2.0"),
        "mit": ("MIT", "MIT License"),
        "bsd-2-clause": ("BSD-2-Clause", "BSD 2-Clause License"),
        "bsd-3-clause": ("BSD-3-Clause", "BSD 3-Clause License"),
        "mpl-2.0": ("MPL-2.0", "Mozilla Public License 2.0"),
        "gpl-3.0": ("GPL-3.0-only", "GNU General Public License v3.0"),
    }
    identity = supported.get(normalized)
    if identity is None:
        raise RuntimeError(
            f"model card license {license_id!r} is not in SLOForge's reviewed SPDX allowlist"
        )
    repository = model.rpartition("@")[0] if "@" in model else model
    url = None if Path(repository).expanduser().exists() else f"https://huggingface.co/{repository}"
    return LicenseMetadata(
        spdx_id=identity[0],
        name=identity[1],
        url=url,
        redistribution_allowed=True,
        verified_at=datetime.now(UTC),
    )


def _resolve_model_snapshot(
    model: str,
) -> tuple[Path, str, str, int, int, ProfileDType]:
    requested_path = Path(model).expanduser()
    if requested_path.exists():
        if not requested_path.is_dir():
            raise RuntimeError("local model specification must name a directory")
        snapshot = requested_path.resolve()
        revision_hint: str | None = None
    else:
        repository, separator, revision = model.rpartition("@")
        repository_id = repository if separator else model
        revision_hint = revision if separator else None
        if not repository_id or (separator and not revision):
            raise ValueError("remote model must use 'repository' or 'repository@revision' syntax")
        try:
            hub = importlib.import_module("huggingface_hub")
        except ImportError as exc:
            raise RuntimeError(
                "remote model resolution requires huggingface_hub from a GPU profiling extra"
            ) from exc
        snapshot = Path(
            hub.snapshot_download(
                repo_id=repository_id,
                revision=revision_hint,
                local_files_only=False,
                etag_timeout=30,
            )
        ).resolve()
    checksum = _directory_checksum(snapshot)
    resolved_revision = snapshot.name if re.fullmatch(r"[0-9a-f]{40,64}", snapshot.name) else None
    if resolved_revision is None:
        resolved_revision = revision_hint or f"local-{checksum[:16]}"
    config_path = snapshot / "config.json"
    if not config_path.is_file():
        raise RuntimeError(f"resolved model has no config.json: {snapshot}")
    config = _load_bounded_json(config_path, max_bytes=8 << 20)
    if not isinstance(config, dict):
        raise RuntimeError("model config.json must be an object")
    text_config = config.get("text_config", config)
    if not isinstance(text_config, dict):
        raise RuntimeError("model text_config must be an object")
    maximum_sequence_length = text_config.get("max_position_embeddings")
    if not isinstance(maximum_sequence_length, int) or maximum_sequence_length < 128:
        raise RuntimeError("model config must declare max_position_embeddings >= 128")
    parameter_count, dtype = _safetensors_model_metadata(snapshot)
    return snapshot, resolved_revision, checksum, parameter_count, maximum_sequence_length, dtype


def _candidate_catalog_from_gpu(
    *,
    engines: Sequence[str],
    hardware: ProbeResult,
    parameter_count: int,
    maximum_sequence_length: int,
    dtype: ProfileDType,
    load_concurrency: int,
) -> list[BackendCandidate]:
    versions = _validate_engine_requests(engines)
    vram_mib, hourly_price, gpu_name = _required_gpu_fields(hardware)
    safe_gpu_name = re.sub(r"[^a-z0-9]+", "-", gpu_name.lower()).strip("-")
    candidates: list[BackendCandidate] = []
    for engine in engines:
        candidates.append(
            BackendCandidate(
                candidate_id=f"{engine}-{safe_gpu_name}-{dtype}",
                runtime=engine,
                runtime_version=versions[engine],
                hardware_id=hardware.fingerprint,
                dtype=dtype,
                hourly_price_usd=hourly_price,
                # Initial values are never emitted: profile_real_candidates replaces them with
                # values calibrated from the raw samples before constructing the ProfileBundle.
                startup_ms=1.0,
                startup_jitter=0.0,
                prefill_base_ms=0.0,
                prefill_ms_per_token=1.0,
                decode_base_ms=0.0,
                decode_ms_per_active_sequence=1.0,
                max_concurrency=load_concurrency,
                memory_bytes=vram_mib * 1024 * 1024,
                failure_rate=0.0,
                model_parameter_count=parameter_count,
                max_sequence_length=maximum_sequence_length,
            )
        )
    return candidates


def profile_real_engines(
    *,
    model: str,
    engines: Sequence[str],
    hardware: ProbeResult,
    hardware_path: Path,
    trace: Sequence[TraceRequest],
    trace_path: Path,
    budget: ProfilingBudget,
    seed: int,
    output_dir: Path,
) -> ProfileBundle:
    """CLI integration boundary for immutable, provenance-complete real-engine profiling."""
    settings = RealProfilerSettings(model=model)
    _validate_engine_requests(engines)
    _, hourly_price, _ = _required_gpu_fields(hardware)
    resolution_meter = _BudgetMeter(budget, hourly_price_usd=hourly_price)
    resolution_meter.timeout(budget.max_duration_s)
    resolution_started = time.monotonic()
    snapshot, revision, checksum, parameters, max_sequence, dtype = _resolve_model_snapshot(model)
    architecture = _required_architecture_metadata(snapshot, parameter_count=parameters)
    license_metadata = _required_license_metadata(snapshot, model=model)
    resolution_meter.charge_elapsed(time.monotonic() - resolution_started)
    candidates = _candidate_catalog_from_gpu(
        engines=engines,
        hardware=hardware,
        parameter_count=parameters,
        maximum_sequence_length=max_sequence,
        dtype=dtype,
        load_concurrency=settings.load_concurrency,
    )
    # All engines receive the same immutable local snapshot, including server-mode adapters.
    resolved_settings = settings.model_copy(update={"model": str(snapshot), "model_revision": None})
    bundle = profile_real_candidates(
        candidates=candidates,
        trace=trace,
        trace_path=trace_path,
        hardware_path=hardware_path,
        budget=budget,
        seed=seed,
        output_dir=output_dir,
        settings=resolved_settings,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    workload_copy = output_dir / "workload.jsonl"
    hardware_copy = output_dir / "hardware.json"
    if trace_path.resolve() != workload_copy.resolve():
        shutil.copyfile(trace_path, workload_copy)
    if hardware_path.resolve() != hardware_copy.resolve():
        shutil.copyfile(hardware_path, hardware_copy)
    write_json(
        output_dir / "model-metadata.json",
        {
            "model_id": model.partition("@")[0],
            "requested_model": model,
            "revision": revision,
            "checksum_sha256": checksum,
            "model_is_mock": False,
            "architecture": architecture.model_dump(mode="json"),
            "license": license_metadata.model_dump(mode="json"),
            "resolved_snapshot": str(snapshot),
            "parameter_count": parameters,
            "maximum_sequence_length": max_sequence,
            "dtype": dtype,
        },
    )
    return bundle
