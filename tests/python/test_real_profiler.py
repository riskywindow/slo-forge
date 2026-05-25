from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from sloforge.hardware.probe import HardwareProbe, ProbeResult
from sloforge.profiler.core import BackendCandidate, ProfileBundle, ProfilingBudget
from sloforge.profiler.gpu_tools import GpuEnvironment, OpenAIStreamTiming
from sloforge.profiler.real import (
    RealProfilerSettings,
    _candidate_catalog_from_gpu,
    _load_bounded_json,
    _resolve_model_snapshot,
    build_real_probe_cases,
    profile_real_candidates,
    profile_real_engines,
)
from sloforge.trace import TraceRequest
from sloforge.trace.format import write_trace


def _trace(count: int = 16) -> list[TraceRequest]:
    return [
        TraceRequest(
            request_id=f"request-{index}",
            arrival_ms=float(index * 10),
            prompt_tokens=4 + index,
            output_tokens=2 + index % 3,
            priority=0 if index % 2 == 0 else 1,
        )
        for index in range(count)
    ]


def _hardware(*, price: float | None = 0.0) -> ProbeResult:
    gpu: dict[str, str | int | float | None] = {
        "index": 0,
        "name": "Test GPU",
        "vram_mib": 16_384,
    }
    if price is not None:
        gpu["hourly_price_usd"] = price
    return ProbeResult(
        captured_at="2026-01-01T00:00:00+00:00",
        requested_device="cuda",
        hardware=HardwareProbe(
            hostname="test",
            os="test",
            architecture="x86_64",
            cpu_model="test",
            logical_cpu_count=8,
            memory_bytes=32 * 1024**3,
            gpu=gpu,
        ),
        benchmarks=[],
        fingerprint="a" * 64,
    )


def _candidate() -> BackendCandidate:
    return BackendCandidate(
        candidate_id="vllm-test",
        runtime="vllm",
        runtime_version="1.0.0",
        hardware_id="hardware-test",
        dtype="float16",
        hourly_price_usd=0,
        startup_ms=1,
        startup_jitter=0,
        prefill_base_ms=0,
        prefill_ms_per_token=1,
        decode_base_ms=0,
        decode_ms_per_active_sequence=1,
        max_concurrency=2,
        memory_bytes=16 * 1024**3,
        failure_rate=0,
        model_parameter_count=1_000_000,
        max_sequence_length=4096,
    )


def test_probe_case_selection_is_seeded_and_preserves_trace_provenance() -> None:
    trace = _trace()
    first = build_real_probe_cases(trace, seed=9, warmup_requests=1, measured_requests=3)
    second = build_real_probe_cases(trace, seed=9, warmup_requests=1, measured_requests=3)
    different = build_real_probe_cases(trace, seed=10, warmup_requests=1, measured_requests=3)
    assert first == second
    assert first != different
    assert all(case.prompt_tokens_hint >= 4 for case in first)
    assert all(case.request_id.startswith("request-") for case in first)


def test_local_snapshot_metadata_uses_weight_headers_without_loading_pickle(
    tmp_path: Path,
) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"max_position_embeddings": 2048}), encoding="utf-8"
    )
    header = json.dumps(
        {"weight": {"dtype": "F16", "shape": [2, 3], "data_offsets": [0, 12]}}
    ).encode()
    (tmp_path / "model.safetensors").write_bytes(
        len(header).to_bytes(8, "little") + header + bytes(12)
    )
    snapshot, revision, checksum, parameters, maximum, dtype = _resolve_model_snapshot(
        str(tmp_path)
    )
    assert snapshot == tmp_path.resolve()
    assert revision.startswith("local-")
    assert len(checksum) == 64
    assert parameters == 6
    assert maximum == 2048
    assert dtype == "float16"


def test_model_metadata_rejects_weight_path_traversal_and_oversized_json(
    tmp_path: Path,
) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"max_position_embeddings": 2048}), encoding="utf-8"
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "../model.safetensors"}}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="unsafe weight path"):
        _resolve_model_snapshot(str(tmp_path))

    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b'"' + b"x" * 32 + b'"')
    with pytest.raises(RuntimeError, match="safety limit"):
        _load_bounded_json(oversized, max_bytes=16)


def test_gpu_candidate_catalog_requires_explicit_price_and_supported_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sloforge.profiler.real.importlib.metadata.version", lambda package: f"1.0-{package}"
    )
    with pytest.raises(RuntimeError, match="hourly_price_usd"):
        _candidate_catalog_from_gpu(
            engines=["vllm"],
            hardware=_hardware(price=None),
            parameter_count=10,
            maximum_sequence_length=2048,
            dtype="float16",
            load_concurrency=4,
        )
    with pytest.raises(ValueError, match="unsupported real engines"):
        _candidate_catalog_from_gpu(
            engines=["unknown"],
            hardware=_hardware(),
            parameter_count=10,
            maximum_sequence_length=2048,
            dtype="float16",
            load_concurrency=4,
        )


class _FakeAdapter:
    def serve_command(self, **kwargs: object) -> list[str]:
        assert kwargs["model"] == "model"
        return ["fake-engine"]


class _FakeServer:
    def __init__(self, *_: object, **__: object) -> None:
        self.ready = False

    def __enter__(self) -> _FakeServer:
        return self

    def __exit__(self, *_: object) -> None:
        self.ready = False

    def wait_ready(self, **_: object) -> str:
        self.ready = True
        time.sleep(0.001)
        return "/health"


def test_real_server_profile_emits_staged_raw_measurements_and_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    trace = _trace(4)
    trace_path = tmp_path / "input.jsonl"
    hardware_path = tmp_path / "hardware-input.json"
    write_trace(trace_path, trace)
    hardware_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr("sloforge.profiler.real.ensure_cuda_requested", lambda **_: object())
    monkeypatch.setattr("sloforge.profiler.real.get_engine_adapter", lambda _: _FakeAdapter())
    monkeypatch.setattr("sloforge.profiler.real.ManagedEngineServer", _FakeServer)
    monkeypatch.setattr(
        "sloforge.profiler.real.build_nsight_systems_command",
        lambda *_, **__: ["nsys", "--", "fake-engine"],
    )
    monkeypatch.setattr(
        "sloforge.profiler.real.stream_openai_completion",
        lambda **_: OpenAIStreamTiming(
            ttft_ms=4,
            e2e_ms=10,
            output_tokens=3,
            prompt_tokens=12,
            token_timestamps_ms=(4, 7, 10),
            event_count=4,
            response_bytes=100,
            finish_reason="length",
        ),
    )
    monkeypatch.setattr(
        "sloforge.profiler.real.gpu_environment",
        lambda **_: GpuEnvironment(
            captured_at="2026-01-01T00:00:00+00:00",
            cuda_visible_devices="0",
            torch_version="test",
            torch_cuda_version="test",
            cudnn_version=None,
            device_count=1,
            selected_device_index=0,
            selected_device_name="Test GPU",
            selected_device_capability=(9, 0),
            selected_device_total_memory_bytes=16 * 1024**3,
            packages={"vllm": "test"},
        ),
    )
    monkeypatch.setattr(
        "sloforge.profiler.real.environment_manifest",
        lambda **_: {"captured_at": "2026-01-01T00:00:00+00:00"},
    )
    output = tmp_path / "profile"
    bundle = profile_real_candidates(
        candidates=[_candidate()],
        trace=trace,
        trace_path=trace_path,
        hardware_path=hardware_path,
        budget=ProfilingBudget(max_duration_s=10, max_cost_usd=0),
        seed=3,
        output_dir=output,
        settings=RealProfilerSettings(
            model="model",
            warmup_requests=1,
            measured_requests=3,
            load_concurrency=2,
            export_perfetto=False,
        ),
    )
    assert isinstance(bundle, ProfileBundle)
    assert len(bundle.raw_measurements) == 13
    assert {measurement.stage for measurement in bundle.raw_measurements} == {
        "startup",
        "prefill",
        "decode",
        "load",
    }
    assert sum(measurement.warmup for measurement in bundle.raw_measurements) == 3
    assert bundle.candidates[0].candidate.prefill_ms_per_token > 0
    assert bundle.candidates[0].candidate.startup_ms > 0
    assert (output / "profile.json").is_file()
    assert len((output / "measurements.jsonl").read_text().splitlines()) == 13
    assert json.loads((output / "nsight-commands.json").read_text())[0]["executed"] is False


def test_cli_wrapper_copies_inputs_and_writes_resolved_model_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["ExampleForCausalLM"],
                "max_position_embeddings": 4096,
                "hidden_size": 128,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "vocab_size": 1024,
            }
        ),
        encoding="utf-8",
    )
    (snapshot / "README.md").write_text("---\nlicense: apache-2.0\n---\n", encoding="utf-8")
    trace_path = tmp_path / "source.jsonl"
    hardware_path = tmp_path / "source-hardware.json"
    write_trace(trace_path, _trace())
    hardware_path.write_text(_hardware().model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(
        "sloforge.profiler.real._resolve_model_snapshot",
        lambda _: (snapshot, "f" * 40, "e" * 64, 1_000_000, 4096, "float16"),
    )
    monkeypatch.setattr("sloforge.profiler.real.importlib.metadata.version", lambda _: "1.0.0")
    captured: dict[str, Any] = {}

    def fake_profile(**kwargs: Any) -> ProfileBundle:
        captured.update(kwargs)
        return ProfileBundle(
            profile_id="real-test",
            generated_at="2026-01-01T00:00:00+00:00",
            seed=kwargs["seed"],
            workload_sha256="a" * 64,
            hardware_sha256="b" * 64,
            budget=kwargs["budget"],
            candidates=[],
            raw_measurements=[],
            environment={},
        )

    monkeypatch.setattr("sloforge.profiler.real.profile_real_candidates", fake_profile)
    output = tmp_path / "result"
    result = profile_real_engines(
        model="Qwen/example@main",
        engines=["transformers", "vllm", "sglang", "tensorrt-llm"],
        hardware=_hardware(),
        hardware_path=hardware_path,
        trace=_trace(),
        trace_path=trace_path,
        budget=ProfilingBudget(max_duration_s=100, max_cost_usd=0),
        seed=7,
        output_dir=output,
    )
    assert result.profile_id == "real-test"
    assert Path(captured["settings"].model) == snapshot
    assert len(captured["candidates"]) == 4
    assert (output / "workload.jsonl").read_bytes() == trace_path.read_bytes()
    assert (output / "hardware.json").read_bytes() == hardware_path.read_bytes()
    metadata = json.loads((output / "model-metadata.json").read_text())
    assert metadata["model_id"] == "Qwen/example"
    assert metadata["revision"] == "f" * 40
    assert metadata["checksum_sha256"] == "e" * 64
    assert metadata["model_is_mock"] is False
    assert metadata["architecture"]["family"] == "ExampleForCausalLM"
    assert metadata["license"]["spdx_id"] == "Apache-2.0"
