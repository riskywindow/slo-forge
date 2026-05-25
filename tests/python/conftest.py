from __future__ import annotations

import pytest

from sloforge.hardware.probe import run_probe, save_probe
from sloforge.models import fit_service_curves
from sloforge.optimizer import OptimizationRequest, optimize, parse_slo_expression
from sloforge.profiler.core import BackendCandidate, ProfilingBudget, profile_mock_candidates
from sloforge.trace.format import generate_bursty_trace, write_trace


@pytest.fixture(scope="session")
def compiled_inputs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    root = tmp_path_factory.mktemp("compiled-inputs")
    trace = generate_bursty_trace(seed=7, count=36)
    trace_path = root / "workload.jsonl"
    write_trace(trace_path, trace)
    hardware = run_probe(device="cpu", warmups=0, samples=3)
    hardware_path = root / "hardware.json"
    save_probe(hardware_path, hardware)
    candidates = [
        BackendCandidate(
            candidate_id="test-fast",
            runtime="mock",
            runtime_version="1.0.0",
            hardware_id="test-cpu",
            dtype="float32",
            hourly_price_usd=0.8,
            startup_ms=80,
            startup_jitter=0.05,
            prefill_base_ms=2,
            prefill_ms_per_token=0.02,
            decode_base_ms=1,
            decode_ms_per_active_sequence=1.2,
            max_concurrency=4,
            memory_bytes=hardware.hardware.memory_bytes,
            failure_rate=0.0,
            model_parameter_count=1_000_000,
            max_sequence_length=8192,
        ),
        BackendCandidate(
            candidate_id="test-cheap",
            runtime="mock",
            runtime_version="1.0.0",
            hardware_id="test-cpu",
            dtype="float32",
            hourly_price_usd=0.2,
            startup_ms=150,
            startup_jitter=0.1,
            prefill_base_ms=6,
            prefill_ms_per_token=0.05,
            decode_base_ms=2,
            decode_ms_per_active_sequence=2.1,
            max_concurrency=2,
            memory_bytes=hardware.hardware.memory_bytes,
            failure_rate=0.01,
            model_parameter_count=1_000_000,
            max_sequence_length=8192,
        ),
    ]
    profile_dir = root / "profile"
    profile = profile_mock_candidates(
        candidates=candidates,
        trace=trace,
        trace_path=trace_path,
        hardware_path=hardware_path,
        budget=ProfilingBudget(max_duration_s=10, max_cost_usd=1),
        seed=7,
        output_dir=profile_dir,
    )
    models = fit_service_curves(profile, seed=7)
    request = OptimizationRequest(
        constraints=parse_slo_expression("p95_ttft_ms<=10000,p99_itl_ms<=1000,availability>=0.50"),
        max_replicas=3,
        trial_budget=12,
        seed=7,
    )
    optimization = optimize(profile=profile, models=models, trace=trace, request=request)
    return {
        "root": root,
        "trace": trace,
        "trace_path": trace_path,
        "hardware": hardware,
        "hardware_path": hardware_path,
        "profile": profile,
        "profile_dir": profile_dir,
        "models": models,
        "optimization": optimization,
    }
