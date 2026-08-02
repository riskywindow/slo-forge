from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest
import yaml

from sloforge.compiler import compile_deployment, explain_plan, mock_qwen3_metadata
from sloforge.exporters import ExportContext, export_plan
from sloforge.hardware.probe import ProbeResult
from sloforge.ir import canonical_hash, load_deployment_plan, load_evidence_bundle
from sloforge.models.service_curve import CalibratedModels
from sloforge.optimizer.core import OptimizationResult
from sloforge.profiler.core import ProfileBundle
from sloforge.trace.format import TraceRequest


def test_compiler_links_plan_and_evidence(
    compiled_inputs: Mapping[str, object], tmp_path: Path
) -> None:
    profile = compiled_inputs["profile"]
    models = compiled_inputs["models"]
    optimization = compiled_inputs["optimization"]
    hardware = compiled_inputs["hardware"]
    trace = compiled_inputs["trace"]
    assert isinstance(profile, ProfileBundle)
    assert isinstance(models, CalibratedModels)
    assert isinstance(optimization, OptimizationResult)
    assert isinstance(hardware, ProbeResult)
    assert isinstance(trace, list) and all(isinstance(item, TraceRequest) for item in trace)
    result = compile_deployment(
        optimization=optimization,
        profile=profile,
        models=models,
        hardware=hardware,
        trace=trace,
        trace_path=compiled_inputs["trace_path"],  # type: ignore[arg-type]
        hardware_path=compiled_inputs["hardware_path"],  # type: ignore[arg-type]
        profile_dir=compiled_inputs["profile_dir"],  # type: ignore[arg-type]
        output_path=tmp_path / "plan.json",
        evidence_dir=tmp_path / "evidence",
        model_metadata=mock_qwen3_metadata(
            requested_model="sloforge/test-model",
            parameter_count=1_000_000,
            maximum_sequence_length=8192,
        ),
        repository_root=Path(__file__).resolve().parents[2],
    )
    loaded_plan = load_deployment_plan(result.plan_path)
    loaded_evidence = load_evidence_bundle(result.evidence_path)
    assert loaded_evidence.plan_digest.value == canonical_hash(loaded_plan)
    assert loaded_evidence.measurements
    assert loaded_plan.predicted_metrics["p95_ttft_ms"].point == pytest.approx(
        optimization.selected.predicted.p95_ttft_ms
    )
    assert loaded_plan.predicted_metrics["cost_per_million_tokens"].point == pytest.approx(
        optimization.selected.predicted.cost_per_million_tokens
    )
    assert "dominant predicted bottleneck" in explain_plan(loaded_plan)


@pytest.mark.parametrize("target", ["local", "docker", "kubernetes", "modal", "truss"])
def test_exporters_generate_valid_offline_artifacts(target: str, tmp_path: Path) -> None:
    context = ExportContext(
        plan_id="fixture-plan",
        model_id="Qwen/Qwen3-0.6B",
        model_revision="main",
        engine="transformers",
        dtype="float16",
        accelerator="L4",
        gpu_count=1,
        max_replicas=3,
        concurrency=4,
        regions=["us-west"],
    )
    result = export_plan(
        context=context,
        target=target,  # type: ignore[arg-type]
        output=tmp_path / target,
        repository_root=Path(__file__).resolve().parents[2],
    )
    assert result.files
    assert result.validation


@pytest.mark.parametrize("target", ["modal", "truss"])
def test_mock_cloud_exporters_preserve_valid_python(target: str, tmp_path: Path) -> None:
    context = ExportContext(
        plan_id="mock-plan",
        model_id="sloforge/mock-model",
        model_revision="fixture",
        engine="mock",
        dtype="float32",
        gpu_count=0,
        concurrency=2,
    )
    result = export_plan(
        context=context,
        target=target,  # type: ignore[arg-type]
        output=tmp_path / target,
        repository_root=Path(__file__).resolve().parents[2],
    )
    assert result.validation
    assert result.deployed is False


def test_modal_export_does_not_emit_local_development_region(tmp_path: Path) -> None:
    output = tmp_path / "modal-local"
    export_plan(
        context=ExportContext(
            plan_id="local-plan",
            model_id="sloforge/mock-model",
            model_revision="fixture",
            engine="mock",
            dtype="float32",
            gpu_count=0,
            regions=["local"],
        ),
        target="modal",
        output=output,
        repository_root=Path(__file__).resolve().parents[2],
    )
    assert "region=None" in (output / "app.py").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("engine", "expected_executable", "has_mock_config"),
    [("transformers", "python", False), ("vllm", "vllm", False), ("mock", "mock-backend", True)],
)
def test_docker_export_never_silently_substitutes_mock_engine(
    engine: str, expected_executable: str, has_mock_config: bool, tmp_path: Path
) -> None:
    output = tmp_path / engine
    export_plan(
        context=ExportContext(
            plan_id=f"docker-{engine}",
            model_id="Qwen/Qwen3-0.6B",
            model_revision="main",
            engine=engine,  # type: ignore[arg-type]
            dtype="float16",
        ),
        target="docker",
        output=output,
        repository_root=Path(__file__).resolve().parents[2],
    )
    compose = yaml.safe_load((output / "compose.yaml").read_text(encoding="utf-8"))
    assert compose["services"]["backend"]["command"][0] == expected_executable
    assert (output / "mock-backend.json").exists() is has_mock_config


def test_export_context_rejects_implicit_device_or_invalid_replica_topology() -> None:
    common = {
        "plan_id": "unsafe",
        "model_id": "Qwen/Qwen3-0.6B",
        "model_revision": "main",
        "engine": "vllm",
        "dtype": "float16",
    }
    with pytest.raises(ValueError, match="accelerator"):
        ExportContext.model_validate({**common, "gpu_count": 1})
    with pytest.raises(ValueError, match="min_replicas"):
        ExportContext.model_validate({**common, "min_replicas": 4, "max_replicas": 2})


def test_docker_gpu_export_reserves_exact_gpu_count(tmp_path: Path) -> None:
    output = tmp_path / "docker-gpu"
    export_plan(
        context=ExportContext(
            plan_id="docker-gpu",
            model_id="Qwen/Qwen3-0.6B",
            model_revision="main",
            engine="vllm",
            dtype="float16",
            accelerator="L4",
            gpu_count=2,
        ),
        target="docker",
        output=output,
        repository_root=Path(__file__).resolve().parents[2],
    )
    compose = yaml.safe_load((output / "compose.yaml").read_text(encoding="utf-8"))
    devices = compose["services"]["backend"]["deploy"]["resources"]["reservations"]["devices"]
    assert devices == [{"driver": "nvidia", "count": 2, "capabilities": ["gpu"]}]


def test_local_exports_do_not_publish_an_unauthenticated_gateway_globally(tmp_path: Path) -> None:
    context = ExportContext(
        plan_id="local-bind",
        model_id="sloforge/mock",
        model_revision="v1",
        engine="mock",
        dtype="float32",
    )
    local = tmp_path / "local"
    docker = tmp_path / "docker"
    root = Path(__file__).resolve().parents[2]
    export_plan(context=context, target="local", output=local, repository_root=root)
    export_plan(context=context, target="docker", output=docker, repository_root=root)
    assert json.loads((local / "gateway.json").read_text())["bind"] == "127.0.0.1:8080"
    compose = yaml.safe_load((docker / "compose.yaml").read_text())
    assert compose["services"]["gateway"]["ports"] == ["127.0.0.1:8080:8080"]
