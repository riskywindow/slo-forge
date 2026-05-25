from __future__ import annotations

import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import pairwise
from pathlib import Path

import httpx

from sloforge.profiler.core import (
    BackendCandidate,
    CandidateProfile,
    ProfileBundle,
    ProfilingBudget,
    RawMeasurement,
    _memory_estimate,
)
from sloforge.trace import TraceRequest
from sloforge.util import environment_manifest, percentile, sha256_file, utc_now, write_json


class StreamTiming:
    def __init__(
        self,
        *,
        status_code: int,
        ttft_ms: float | None,
        e2e_ms: float,
        itl_samples_ms: list[float],
        chunks: int,
        error: str | None,
    ) -> None:
        self.status_code = status_code
        self.ttft_ms = ttft_ms
        self.e2e_ms = e2e_ms
        self.itl_samples_ms = itl_samples_ms
        self.chunks = chunks
        self.error = error


def _bounded_error_text(response: httpx.Response, *, max_bytes: int = 512) -> str:
    body = bytearray()
    truncated = False
    for chunk in response.iter_bytes():
        remaining = max_bytes - len(body)
        body.extend(chunk[: max(0, remaining)])
        if len(chunk) > remaining or len(body) == max_bytes:
            truncated = True
            break
    rendered = body.decode(errors="replace")
    return f"{rendered}...[truncated]" if truncated else rendered


def _stream_probe(
    endpoint: str, *, prompt_tokens: int, output_tokens: int, timeout_s: float = 20.0
) -> StreamTiming:
    payload = {
        "model": "sloforge-mock",
        "prompt": "latency " * prompt_tokens,
        "max_tokens": output_tokens,
        "stream": True,
    }
    started = time.perf_counter_ns()
    timestamps: list[float] = []
    error: str | None = None
    with httpx.Client(timeout=timeout_s) as client:
        try:
            with client.stream("POST", f"{endpoint}/v1/completions", json=payload) as response:
                status = response.status_code
                if status >= 400:
                    error = _bounded_error_text(response)
                else:
                    for line in response.iter_lines():
                        if not line.startswith("data: ") or line == "data: [DONE]":
                            continue
                        raw = line[6:]
                        try:
                            json.loads(raw)
                        except json.JSONDecodeError as exc:
                            error = f"malformed SSE JSON: {exc}"
                            break
                        timestamps.append((time.perf_counter_ns() - started) / 1e6)
        except httpx.HTTPError as exc:
            status = 599
            error = str(exc)
    e2e = (time.perf_counter_ns() - started) / 1e6
    itls = [right - left for left, right in pairwise(timestamps)]
    return StreamTiming(
        status_code=status,
        ttft_ms=timestamps[0] if timestamps else None,
        e2e_ms=e2e,
        itl_samples_ms=itls,
        chunks=len(timestamps),
        error=error,
    )


def profile_running_mock_candidates(
    *,
    candidates: list[BackendCandidate],
    endpoints: dict[str, str],
    startup_samples_ms: dict[str, list[float]],
    trace: list[TraceRequest],
    trace_path: Path,
    hardware_path: Path,
    budget: ProfilingBudget,
    seed: int,
    output_dir: Path,
) -> ProfileBundle:
    """Profile explicit running mock HTTP servers; every latency sample is wall-clock measured."""
    measurements: list[RawMeasurement] = []
    profiles: list[CandidateProfile] = []
    prompt_grid = [32, 128, 512, 2048]
    for candidate in candidates:
        endpoint = endpoints.get(candidate.candidate_id)
        if endpoint is None:
            raise ValueError(f"missing endpoint for {candidate.candidate_id}")
        health = httpx.get(f"{endpoint}/health", timeout=3.0)
        health.raise_for_status()
        weight_bytes, kv_bytes = _memory_estimate(candidate)
        required = int((weight_bytes + kv_bytes) * 1.10)
        if required > candidate.memory_bytes:
            profiles.append(
                CandidateProfile(
                    candidate=candidate,
                    feasible=False,
                    rejection_reason=f"static memory estimate {required} exceeds capacity {candidate.memory_bytes}",
                    estimated_weight_bytes=weight_bytes,
                    estimated_kv_bytes_per_token=kv_bytes // 8192,
                    raw_measurement_ids=[],
                    summaries={},
                )
            )
            continue
        raw: list[RawMeasurement] = []
        for index, sample in enumerate(startup_samples_ms[candidate.candidate_id]):
            budget.reserve(duration_s=sample / 1000.0, hourly_price_usd=candidate.hourly_price_usd)
            raw.append(
                RawMeasurement(
                    measurement_id=f"{candidate.candidate_id}-startup-{index}",
                    candidate_id=candidate.candidate_id,
                    stage="startup",
                    sample_index=index,
                    warmup=index == 0,
                    latency_ms=sample,
                    seed=seed,
                )
            )
        for prompt in prompt_grid:
            for index in range(5):
                timing = _stream_probe(endpoint, prompt_tokens=prompt, output_tokens=1)
                budget.reserve(
                    duration_s=timing.e2e_ms / 1000.0, hourly_price_usd=candidate.hourly_price_usd
                )
                raw.append(
                    RawMeasurement(
                        measurement_id=f"{candidate.candidate_id}-prefill-{prompt}-{index}",
                        candidate_id=candidate.candidate_id,
                        stage="prefill",
                        sample_index=index,
                        warmup=index == 0,
                        prompt_tokens=prompt,
                        batch_size=1,
                        latency_ms=timing.ttft_ms or timing.e2e_ms,
                        peak_memory_bytes=weight_bytes + prompt * (kv_bytes // 8192),
                        failed=timing.status_code >= 400 or timing.error is not None,
                        seed=seed,
                    )
                )
        for active in (1, 2, 4):
            if active > candidate.max_concurrency:
                continue
            for index in range(4):
                with ThreadPoolExecutor(
                    max_workers=active, thread_name_prefix="sloforge-profile"
                ) as pool:
                    timings = list(
                        pool.map(
                            partial(
                                _concurrent_probe,
                                endpoint,
                                prompt_tokens=128,
                                output_tokens=8,
                            ),
                            range(active),
                        )
                    )
                elapsed = sum(item.e2e_ms for item in timings) / 1000.0
                budget.reserve(duration_s=elapsed, hourly_price_usd=candidate.hourly_price_usd)
                itls = [sample for timing in timings for sample in timing.itl_samples_ms]
                failed = any(
                    timing.status_code >= 400 or timing.error is not None for timing in timings
                )
                latency = statistics.median(itls) if itls else max(item.e2e_ms for item in timings)
                raw.append(
                    RawMeasurement(
                        measurement_id=f"{candidate.candidate_id}-decode-{active}-{index}",
                        candidate_id=candidate.candidate_id,
                        stage="decode",
                        sample_index=index,
                        warmup=index == 0,
                        active_sequences=active,
                        batch_size=active,
                        latency_ms=latency,
                        failed=failed,
                        seed=seed,
                    )
                )
        for index, request in enumerate(trace[:24]):
            timing = _stream_probe(
                endpoint,
                prompt_tokens=min(request.prompt_tokens, 2048),
                output_tokens=min(request.output_tokens, 32),
            )
            budget.reserve(
                duration_s=timing.e2e_ms / 1000.0, hourly_price_usd=candidate.hourly_price_usd
            )
            itl = (
                statistics.median(timing.itl_samples_ms) if timing.itl_samples_ms else timing.e2e_ms
            )
            raw.append(
                RawMeasurement(
                    measurement_id=f"{candidate.candidate_id}-load-{index}",
                    candidate_id=candidate.candidate_id,
                    stage="load",
                    sample_index=index,
                    warmup=False,
                    prompt_tokens=request.prompt_tokens,
                    output_tokens=timing.chunks,
                    active_sequences=1 + index % candidate.max_concurrency,
                    batch_size=1,
                    latency_ms=timing.e2e_ms,
                    ttft_ms=timing.ttft_ms or timing.e2e_ms,
                    itl_ms=itl,
                    e2e_ms=timing.e2e_ms,
                    failed=timing.status_code >= 400 or timing.error is not None,
                    seed=seed,
                )
            )
        usable = [item for item in raw if not item.warmup]
        load = [item for item in usable if item.stage == "load" and not item.failed]
        startup = [item.latency_ms for item in usable if item.stage == "startup"]
        ttfts = [item.ttft_ms for item in load if item.ttft_ms is not None]
        itls = [item.itl_ms for item in load if item.itl_ms is not None]
        e2es = [item.e2e_ms for item in load if item.e2e_ms is not None]
        if not load or not startup or not ttfts or not itls or not e2es:
            raise RuntimeError(
                f"candidate {candidate.candidate_id} produced insufficient successful HTTP measurements"
            )
        duration_s = max((trace[-1].arrival_ms - trace[0].arrival_ms) / 1000.0, 0.001)
        summaries = {
            "startup_p95_ms": percentile(startup, 0.95),
            "ttft_p50_ms": percentile(ttfts, 0.50),
            "ttft_p95_ms": percentile(ttfts, 0.95),
            "itl_p99_ms": percentile(itls, 0.99),
            "e2e_p95_ms": percentile(e2es, 0.95),
            "availability": len(load) / len([item for item in usable if item.stage == "load"]),
            "measured_goodput_tokens_s": sum(item.output_tokens or 0 for item in load) / duration_s,
        }
        profiles.append(
            CandidateProfile(
                candidate=candidate,
                feasible=True,
                estimated_weight_bytes=weight_bytes,
                estimated_kv_bytes_per_token=kv_bytes // 8192,
                raw_measurement_ids=[item.measurement_id for item in raw],
                summaries=summaries,
            )
        )
        measurements.extend(raw)
    profile_id = f"http-profile-{seed}-{int(time.time())}"
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
        for item in measurements:
            handle.write(item.model_dump_json(exclude_none=True) + "\n")
    write_json(output_dir / "environment.json", bundle.environment)
    (output_dir / "workload.jsonl").write_bytes(trace_path.read_bytes())
    (output_dir / "hardware.json").write_bytes(hardware_path.read_bytes())
    return bundle


def _concurrent_probe(
    endpoint: str, _: int, *, prompt_tokens: int, output_tokens: int
) -> StreamTiming:
    return _stream_probe(endpoint, prompt_tokens=prompt_tokens, output_tokens=output_tokens)
