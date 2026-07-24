from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sloforge.helix.characterization.hardware_studies import (
    GPU_BUDGET_ENV,
    EvidenceFile,
    ExercisedArtifactInput,
    ExercisedHardwareArtifact,
    HardwareMetric,
    HardwareOperation,
    HardwareStudyKind,
    MeasurementStatus,
    build_hardware_studies,
    detect_hardware_capabilities,
    write_hardware_studies,
)

BASELINE_HARDWARE = Path("artifacts/branchfabric/manifests/hardware-baseline.json")
BASELINE_SOFTWARE = Path("artifacts/branchfabric/manifests/software-baseline.json")


def _write_json(path: Path, value: object) -> str:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _capable_manifests(tmp_path: Path) -> tuple[Path, Path]:
    hardware = tmp_path / "hardware.json"
    software = tmp_path / "software.json"
    _write_json(
        hardware,
        {
            "schema_version": "sloforge.branchfabric.hardware-manifest/v1",
            "measurement_class": "HARDWARE_BACKED_REAL",
            "gpu": {
                "nvidia": {
                    "available": True,
                    "nvidia_smi": True,
                    "cuda_toolkit": True,
                    "nvlink": True,
                    "hardware_backed_helix_supported": True,
                    "device_count": 2,
                    "devices": [
                        {"model": "Test GPU 0"},
                        {"model": "Test GPU 1"},
                    ],
                }
            },
            "fabric": {
                "multi_gpu": True,
                "multi_node": True,
                "infiniband": True,
                "rdma": True,
            },
        },
    )
    _write_json(
        software,
        {
            "schema_version": "sloforge.branchfabric.software-manifest/v1",
            "measurement_class": "HARDWARE_BACKED_REAL",
            "tools": {"nvcc": "CUDA 13.0", "nccl": "2.28"},
            "packages": {
                "torch": "2.12.0",
                "vllm": "0.20.0",
                "sglang": "0.5.0",
                "nccl": "2.28",
            },
            "profilers": {
                "nvidia_smi": "/usr/bin/nvidia-smi",
                "nsys": "/opt/nvidia/nsys",
                "ncu": "/opt/nvidia/ncu",
                "perfetto": "/opt/perfetto/perfetto",
            },
        },
    )
    return hardware, software


def _artifact(
    tmp_path: Path,
    *,
    hardware: Path,
    software: Path,
    study: HardwareStudyKind,
    operations: tuple[HardwareOperation, ...],
    seed: int = 41,
) -> ExercisedArtifactInput:
    raw = tmp_path / f"{study.value}-samples.jsonl"
    raw.write_text('{"duration_ns":1234}\n', encoding="utf-8")
    raw_payload = raw.read_bytes()
    raw_hash = hashlib.sha256(raw_payload).hexdigest()
    evidence = EvidenceFile(
        path=raw.name,
        sha256=raw_hash,
        size_bytes=len(raw_payload),
        role="raw_samples",
    )
    metrics = tuple(
        HardwareMetric(
            metric=f"{operation.value}/duration",
            operation=operation,
            value=1234.0,
            unit="ns",
            statistic="median",
            sample_count=3,
            evidence_sha256=(raw_hash,),
        )
        for operation in operations
    )
    document = ExercisedHardwareArtifact(
        measurement_class=MeasurementStatus.HARDWARE_BACKED_REAL,
        study=study,
        seed=seed,
        hardware_manifest_sha256=hashlib.sha256(hardware.read_bytes()).hexdigest(),
        software_manifest_sha256=hashlib.sha256(software.read_bytes()).hexdigest(),
        hardware_exercised=True,
        simulated_hardware=False,
        completed=True,
        command_argv=("sloforge", "helix", "characterize", "run"),
        operations_exercised=operations,
        evidence_files=(evidence,),
        metrics=metrics,
        dropped_samples=0,
    )
    artifact = tmp_path / f"{study.value}-artifact.json"
    payload = (document.model_dump_json(indent=2) + "\n").encode()
    artifact.write_bytes(payload)
    return ExercisedArtifactInput(
        study=study,
        path=artifact,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
    )


def test_baseline_machine_reports_unavailable_without_invented_metrics() -> None:
    bundle = build_hardware_studies(
        BASELINE_HARDWARE,
        BASELINE_SOFTWARE,
        seed=41,
        environment={},
    )

    assert not bundle.capabilities.helix_cuda_compatible
    assert bundle.capabilities.gpu_count == 0
    assert not bundle.capabilities.multi_gpu_available
    assert not bundle.capabilities.multi_node_available
    assert bundle.capabilities.pytorch_available is False
    assert bundle.capabilities.vllm_available is False
    assert bundle.capabilities.sglang_available is False
    assert bundle.capabilities.budget.present is False
    assert bundle.capabilities.budget.provisioning_performed is False
    for report in (bundle.gpu, bundle.multi_gpu, bundle.multi_node):
        assert report.status is MeasurementStatus.UNAVAILABLE
        assert report.metrics == ()
        assert report.exercised_artifact is None
        assert report.synthetic_substitution_used is False
        assert report.future_commands
        assert all("--replace" not in command.argv for command in report.future_commands)
        assert all(not command.provisions_resources for command in report.future_commands)
    covered = {
        operation
        for report in (bundle.gpu, bundle.multi_gpu, bundle.multi_node)
        for command in report.future_commands
        for operation in command.covered_operations
    }
    assert covered == set(HardwareOperation)


def test_capabilities_and_profiler_commands_are_manifest_derived(tmp_path: Path) -> None:
    hardware, software = _capable_manifests(tmp_path)
    capabilities = detect_hardware_capabilities(
        hardware,
        software,
        environment={GPU_BUDGET_ENV: "12.50"},
    )
    bundle = build_hardware_studies(
        hardware,
        software,
        seed=73,
        environment={GPU_BUDGET_ENV: "12.50"},
    )

    assert capabilities.helix_cuda_compatible
    assert capabilities.gpu_count == 2
    assert capabilities.gpu_models == ("Test GPU 0", "Test GPU 1")
    assert capabilities.nvlink_available
    assert capabilities.multi_gpu_available
    assert capabilities.multi_node_available
    assert capabilities.nccl_available
    assert capabilities.pytorch_available
    assert capabilities.vllm_available
    assert capabilities.sglang_available
    assert capabilities.budget.ceiling_usd == 12.5
    assert not capabilities.budget.paid_resource_provisioning_authorized
    assert bundle.gpu.status is MeasurementStatus.UNAVAILABLE
    assert bundle.gpu.capabilities_satisfied
    assert bundle.gpu.metrics == ()
    command_ids = {item.command_id for item in bundle.gpu.future_commands}
    assert {"characterize-gpu", "nsight-systems-gpu", "nsight-compute-gpu"}.issubset(command_ids)
    perfetto = next(
        command
        for command in bundle.gpu.future_commands
        if command.command_id == "perfetto-system-gpu"
    )
    assert perfetto.argv[0] == "/opt/perfetto/perfetto"
    assert perfetto.required_files == ("benchmarks/branchfabric/perfetto-system-trace.cfg",)
    base = next(
        command
        for command in bundle.gpu.future_commands
        if command.command_id == "characterize-gpu"
    )
    assert base.argv[:7] == (
        "uv",
        "run",
        "--locked",
        "sloforge",
        "helix",
        "characterize",
        "run",
    )
    assert base.environment == {"SLOFORGE_BRANCHFABRIC_CHARACTERIZATION_ALLOW_GPU": "1"}


def test_complete_hash_verified_gpu_artifact_enables_only_gpu_report(tmp_path: Path) -> None:
    hardware, software = _capable_manifests(tmp_path)
    required_gpu = (
        HardwareOperation.KV_SIZE,
        HardwareOperation.STATE_FORK,
        HardwareOperation.STATE_CONVERT,
        HardwareOperation.H2D,
        HardwareOperation.D2H,
        HardwareOperation.GPU_INTERFERENCE,
        HardwareOperation.MIGRATION,
    )
    artifact = _artifact(
        tmp_path,
        hardware=hardware,
        software=software,
        study=HardwareStudyKind.GPU,
        operations=required_gpu,
    )

    bundle = build_hardware_studies(
        hardware,
        software,
        seed=41,
        exercised_artifacts=(artifact,),
        environment={},
    )

    assert bundle.gpu.status is MeasurementStatus.HARDWARE_BACKED_REAL
    assert bundle.gpu.exercised_artifact is not None
    assert len(bundle.gpu.metrics) == len(required_gpu)
    assert not bundle.gpu.hardware_measurement_missing
    assert bundle.multi_gpu.status is MeasurementStatus.UNAVAILABLE
    assert bundle.multi_gpu.metrics == ()
    assert bundle.multi_node.status is MeasurementStatus.UNAVAILABLE


def test_incomplete_or_hash_mismatched_artifact_fails_closed(tmp_path: Path) -> None:
    hardware, software = _capable_manifests(tmp_path)
    incomplete = _artifact(
        tmp_path,
        hardware=hardware,
        software=software,
        study=HardwareStudyKind.GPU,
        operations=(HardwareOperation.KV_SIZE,),
    )
    bundle = build_hardware_studies(
        hardware,
        software,
        seed=41,
        exercised_artifacts=(incomplete,),
        environment={},
    )
    assert bundle.gpu.status is MeasurementStatus.UNAVAILABLE
    assert bundle.gpu.metrics == ()
    assert bundle.gpu.exercised_artifact is None
    assert "missing operations" in bundle.gpu.reason

    mismatched = incomplete.model_copy(update={"expected_sha256": "0" * 64})
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        build_hardware_studies(
            hardware,
            software,
            seed=41,
            exercised_artifacts=(mismatched,),
            environment={},
        )


def test_budget_validation_and_exclusive_report_write(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="finite non-negative"):
        detect_hardware_capabilities(
            BASELINE_HARDWARE,
            BASELINE_SOFTWARE,
            environment={GPU_BUDGET_ENV: "NaN"},
        )
    bundle = build_hardware_studies(
        BASELINE_HARDWARE,
        BASELINE_SOFTWARE,
        seed=113,
        environment={},
    )
    output = tmp_path / "hardware-studies.json"
    artifact_hash = write_hardware_studies(bundle, output)
    assert artifact_hash == hashlib.sha256(output.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        write_hardware_studies(bundle, output)
