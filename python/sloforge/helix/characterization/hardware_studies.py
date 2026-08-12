"""Fail-closed hardware characterization status and future command generation.

Capability manifests describe what a machine may be able to exercise; they are
not measurements of Helix state operations.  A study is classified
``HARDWARE_BACKED_REAL`` only after a caller supplies a hash-pinned exercised
artifact whose raw evidence files also verify.  Missing, incomplete, simulated,
or capability-incompatible evidence produces ``UNAVAILABLE`` with no metrics.

This module never probes, provisions, or launches hardware.  It only validates
bounded local evidence and emits non-destructive command argv for a future run.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shlex
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_EXERCISED_ARTIFACT_BYTES = 16 * 1024 * 1024
MAX_EVIDENCE_FILE_BYTES = 16 * 1024 * 1024 * 1024
MAX_EVIDENCE_FILES = 256
MAX_METRICS = 4096
MAX_COMMANDS = 128
GPU_BUDGET_ENV: Final[Literal["SLOFORGE_BRANCHFABRIC_CHARACTERIZATION_GPU_BUDGET_USD"]] = (
    "SLOFORGE_BRANCHFABRIC_CHARACTERIZATION_GPU_BUDGET_USD"
)

Identifier = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}$"),
]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class HardwareStudyModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class HardwareStudyKind(StrEnum):
    GPU = "gpu"
    MULTI_GPU = "multi_gpu"
    MULTI_NODE = "multi_node"


class HardwareOperation(StrEnum):
    KV_SIZE = "kv_size"
    STATE_FORK = "state_fork"
    STATE_CONVERT = "state_convert"
    H2D = "host_to_device"
    D2H = "device_to_host"
    P2P = "gpu_peer_to_peer"
    NCCL = "nccl_collective"
    GPU_INTERFERENCE = "gpu_interference"
    MIGRATION = "continuum_migration"


_REQUIRED_OPERATIONS: dict[HardwareStudyKind, frozenset[HardwareOperation]] = {
    HardwareStudyKind.GPU: frozenset(
        {
            HardwareOperation.KV_SIZE,
            HardwareOperation.STATE_FORK,
            HardwareOperation.STATE_CONVERT,
            HardwareOperation.H2D,
            HardwareOperation.D2H,
            HardwareOperation.GPU_INTERFERENCE,
            HardwareOperation.MIGRATION,
        }
    ),
    HardwareStudyKind.MULTI_GPU: frozenset({HardwareOperation.P2P, HardwareOperation.NCCL}),
    HardwareStudyKind.MULTI_NODE: frozenset({HardwareOperation.MIGRATION}),
}


class MeasurementStatus(StrEnum):
    HARDWARE_BACKED_REAL = "HARDWARE_BACKED_REAL"
    UNAVAILABLE = "UNAVAILABLE"


class ManifestEvidence(HardwareStudyModel):
    reference: str = Field(min_length=1, max_length=4096)
    sha256: Sha256
    size_bytes: int = Field(ge=1, le=MAX_MANIFEST_BYTES)
    schema_version: str = Field(min_length=1, max_length=256)
    measurement_class: str = Field(min_length=1, max_length=256)


class BudgetStatus(HardwareStudyModel):
    environment_variable: Literal["SLOFORGE_BRANCHFABRIC_CHARACTERIZATION_GPU_BUDGET_USD"] = (
        GPU_BUDGET_ENV
    )
    present: bool
    ceiling_usd: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    paid_resource_provisioning_authorized: Literal[False] = False
    provisioning_performed: Literal[False] = False
    reason: str = Field(min_length=1, max_length=1024)


class HardwareCapabilities(HardwareStudyModel):
    schema_version: Literal["sloforge.branchfabric.hardware-capabilities/v1"] = (
        "sloforge.branchfabric.hardware-capabilities/v1"
    )
    hardware_manifest: ManifestEvidence
    software_manifest: ManifestEvidence
    cuda_gpu_available: bool
    helix_cuda_compatible: bool
    gpu_count: int = Field(ge=0, le=4096)
    gpu_models: tuple[str, ...] = Field(max_length=4096)
    cuda_toolkit_available: bool
    nvidia_smi_available: bool
    nvlink_available: bool
    multi_gpu_available: bool
    infiniband_available: bool
    rdma_available: bool
    multi_node_available: bool
    pytorch_available: bool
    vllm_available: bool
    sglang_available: bool
    nccl_available: bool
    profiler_tools: dict[str, str]
    budget: BudgetStatus
    detection_policy: tuple[str, ...] = Field(min_length=1)


class FutureCommand(HardwareStudyModel):
    command_id: Identifier
    study: HardwareStudyKind
    purpose: str = Field(min_length=1, max_length=2048)
    covered_operations: tuple[HardwareOperation, ...] = Field(min_length=1)
    argv: tuple[str, ...] = Field(min_length=1, max_length=128)
    shell_rendering: str = Field(min_length=1, max_length=16384)
    environment: dict[str, str]
    timeout_seconds: int = Field(ge=1, le=86_400)
    expected_output: str = Field(min_length=1, max_length=4096)
    required_capabilities: tuple[str, ...]
    required_files: tuple[str, ...] = ()
    currently_runnable: bool
    provisions_resources: Literal[False] = False
    overwrites_existing_output: Literal[False] = False

    @model_validator(mode="after")
    def rendering_matches_argv(self) -> Self:
        if self.shell_rendering != shlex.join(self.argv):
            raise ValueError("shell rendering must be the exact quoted argv")
        if "--replace" in self.argv:
            raise ValueError("future hardware commands cannot overwrite existing output")
        return self


class ExercisedArtifactInput(HardwareStudyModel):
    study: HardwareStudyKind
    path: Path
    expected_sha256: Sha256


class EvidenceFile(HardwareStudyModel):
    path: str = Field(min_length=1, max_length=4096)
    sha256: Sha256
    size_bytes: int = Field(ge=1, le=MAX_EVIDENCE_FILE_BYTES)
    role: Identifier


class HardwareMetric(HardwareStudyModel):
    metric: Identifier
    operation: HardwareOperation
    value: float = Field(allow_inf_nan=False)
    unit: Identifier
    statistic: Identifier
    sample_count: int = Field(ge=1, le=10**9)
    evidence_sha256: tuple[Sha256, ...] = Field(min_length=1, max_length=MAX_EVIDENCE_FILES)


class ExercisedHardwareArtifact(HardwareStudyModel):
    schema_version: Literal["sloforge.branchfabric.exercised-hardware-artifact/v1"] = (
        "sloforge.branchfabric.exercised-hardware-artifact/v1"
    )
    measurement_class: Literal[MeasurementStatus.HARDWARE_BACKED_REAL]
    study: HardwareStudyKind
    seed: Annotated[int, Field(ge=0, le=2**63 - 1)]
    hardware_manifest_sha256: Sha256
    software_manifest_sha256: Sha256
    hardware_exercised: Literal[True]
    simulated_hardware: Literal[False]
    completed: Literal[True]
    command_argv: tuple[str, ...] = Field(min_length=1, max_length=128)
    operations_exercised: tuple[HardwareOperation, ...] = Field(min_length=1, max_length=64)
    evidence_files: tuple[EvidenceFile, ...] = Field(min_length=1, max_length=MAX_EVIDENCE_FILES)
    metrics: tuple[HardwareMetric, ...] = Field(min_length=1, max_length=MAX_METRICS)
    dropped_samples: int = Field(ge=0)

    @model_validator(mode="after")
    def metrics_reference_declared_evidence(self) -> Self:
        if len(set(self.operations_exercised)) != len(self.operations_exercised):
            raise ValueError("operations_exercised must be unique")
        evidence = {item.sha256 for item in self.evidence_files}
        for metric in self.metrics:
            if not set(metric.evidence_sha256).issubset(evidence):
                raise ValueError("metric references undeclared raw evidence")
            if metric.operation not in self.operations_exercised:
                raise ValueError("metric operation was not declared exercised")
        return self


class VerifiedExercisedArtifact(HardwareStudyModel):
    reference: str = Field(min_length=1, max_length=4096)
    sha256: Sha256
    size_bytes: int = Field(ge=1, le=MAX_EXERCISED_ARTIFACT_BYTES)
    document: ExercisedHardwareArtifact
    verified_evidence_files: tuple[EvidenceFile, ...]


class HardwareStudyReport(HardwareStudyModel):
    schema_version: Literal["sloforge.branchfabric.hardware-study-report/v1"] = (
        "sloforge.branchfabric.hardware-study-report/v1"
    )
    study: HardwareStudyKind
    status: MeasurementStatus
    seed: Annotated[int, Field(ge=0, le=2**63 - 1)]
    reason: str = Field(min_length=1, max_length=4096)
    capabilities_satisfied: bool
    required_operations: tuple[HardwareOperation, ...] = Field(min_length=1)
    exercised_artifact: VerifiedExercisedArtifact | None
    metrics: tuple[HardwareMetric, ...] = Field(max_length=MAX_METRICS)
    future_commands: tuple[FutureCommand, ...] = Field(min_length=1, max_length=MAX_COMMANDS)
    synthetic_substitution_used: Literal[False] = False
    hardware_measurement_missing: bool
    limitations: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def measurement_status_is_fail_closed(self) -> Self:
        if self.status is MeasurementStatus.UNAVAILABLE:
            if self.metrics or self.exercised_artifact is not None:
                raise ValueError("unavailable hardware reports cannot contain metrics or artifacts")
            if not self.hardware_measurement_missing:
                raise ValueError("unavailable report must mark hardware measurement missing")
        else:
            if self.exercised_artifact is None or not self.metrics:
                raise ValueError("hardware-backed report requires verified exercised evidence")
            if self.hardware_measurement_missing or not self.capabilities_satisfied:
                raise ValueError("hardware-backed report requires satisfied capabilities")
        return self


class HardwareStudiesBundle(HardwareStudyModel):
    schema_version: Literal["sloforge.branchfabric.hardware-studies/v1"] = (
        "sloforge.branchfabric.hardware-studies/v1"
    )
    seed: Annotated[int, Field(ge=0, le=2**63 - 1)]
    capabilities: HardwareCapabilities
    gpu: HardwareStudyReport
    multi_gpu: HardwareStudyReport
    multi_node: HardwareStudyReport
    provisioning_performed: Literal[False] = False

    @model_validator(mode="after")
    def study_kinds_match_fields(self) -> Self:
        if (
            self.gpu.study is not HardwareStudyKind.GPU
            or self.multi_gpu.study is not HardwareStudyKind.MULTI_GPU
            or self.multi_node.study is not HardwareStudyKind.MULTI_NODE
        ):
            raise ValueError("hardware study report assigned to the wrong bundle field")
        return self


def _read_bounded(path: Path, maximum: int) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"evidence input must be a regular non-symlink file: {path}")
    size = path.stat().st_size
    if size <= 0 or size > maximum:
        raise ValueError(f"evidence input size must be within 1..{maximum} bytes: {path}")
    with path.open("rb") as handle:
        payload = handle.read(maximum + 1)
    if len(payload) > maximum:
        raise ValueError(f"evidence input exceeds {maximum} bytes: {path}")
    return payload


def _sha256_file(path: Path, maximum: int) -> tuple[str, int]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"evidence file must be a regular non-symlink file: {path}")
    size = path.stat().st_size
    if size <= 0 or size > maximum:
        raise ValueError(f"evidence file size must be within 1..{maximum} bytes: {path}")
    digest = hashlib.sha256()
    observed = 0
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            observed += len(block)
            if observed > maximum:
                raise ValueError(f"evidence file grew beyond {maximum} bytes: {path}")
            digest.update(block)
    if observed != size:
        raise ValueError(f"evidence file size changed while hashing: {path}")
    return digest.hexdigest(), observed


def _load_manifest(path: Path) -> tuple[dict[str, object], ManifestEvidence]:
    payload = _read_bounded(path, MAX_MANIFEST_BYTES)
    document = json.loads(payload)
    if not isinstance(document, dict):
        raise ValueError(f"manifest must be a JSON object: {path}")
    schema = document.get("schema_version")
    if not isinstance(schema, str) or not schema:
        raise ValueError(f"manifest schema_version is absent: {path}")
    measurement_class = document.get("measurement_class", "UNDECLARED")
    if not isinstance(measurement_class, str):
        raise ValueError(f"manifest measurement_class must be a string: {path}")
    return document, ManifestEvidence(
        reference=path.as_posix(),
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        schema_version=schema,
        measurement_class=measurement_class,
    )


def _mapping(document: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = document.get(key)
    return value if isinstance(value, dict) else {}


def _available_version(mapping: Mapping[str, object], *keys: str) -> bool:
    for key in keys:
        value = mapping.get(key)
        if value is not None and value != "" and value != "ABSENT" and value is not False:
            return True
    return False


def _budget_status(environment: Mapping[str, str]) -> BudgetStatus:
    raw = environment.get(GPU_BUDGET_ENV)
    if raw is None or not raw.strip():
        return BudgetStatus(
            present=False,
            ceiling_usd=None,
            reason=(
                "GPU budget is absent. Existing accessible hardware may be measured; paid "
                "resource provisioning remains unauthorized and is never performed here."
            ),
        )
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"{GPU_BUDGET_ENV} must be a finite non-negative number") from error
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{GPU_BUDGET_ENV} must be a finite non-negative number")
    return BudgetStatus(
        present=True,
        ceiling_usd=value,
        reason=(
            "A hard budget ceiling was observed, but this module performs no provisioning or "
            "external mutation. The ceiling is recorded for a future authorized executor."
        ),
    )


def detect_hardware_capabilities(
    hardware_manifest: Path,
    software_manifest: Path,
    *,
    environment: Mapping[str, str] | None = None,
) -> HardwareCapabilities:
    """Conservatively interpret preserved manifests without probing the host."""

    hardware, hardware_evidence = _load_manifest(hardware_manifest)
    software, software_evidence = _load_manifest(software_manifest)
    gpu = _mapping(hardware, "gpu")
    nvidia = _mapping(gpu, "nvidia")
    fabric = _mapping(hardware, "fabric")
    tools = _mapping(software, "tools")
    packages = _mapping(software, "packages")
    profilers = _mapping(software, "profilers")
    cuda_gpu = bool(nvidia.get("available", False))
    cuda_toolkit = bool(nvidia.get("cuda_toolkit", False)) or _available_version(tools, "nvcc")
    nvidia_smi = bool(nvidia.get("nvidia_smi", False)) or _available_version(
        profilers, "nvidia_smi"
    )
    compatible = bool(
        hardware_evidence.measurement_class == "HARDWARE_BACKED_REAL"
        and cuda_gpu
        and cuda_toolkit
        and nvidia_smi
        and nvidia.get("hardware_backed_helix_supported", False)
    )
    devices = nvidia.get("devices")
    device_rows = (
        tuple(item for item in devices if isinstance(item, dict))
        if isinstance(devices, list)
        else ()
    )
    declared_count = nvidia.get("device_count")
    if isinstance(declared_count, int) and not isinstance(declared_count, bool):
        gpu_count = max(0, declared_count)
    elif device_rows:
        gpu_count = len(device_rows)
    else:
        gpu_count = 1 if cuda_gpu else 0
    models = tuple(
        str(item.get("model", item.get("name", "unknown NVIDIA GPU"))) for item in device_rows
    )
    if cuda_gpu and not models:
        model = nvidia.get("model")
        models = (str(model) if model else "unreported NVIDIA GPU",)
    profiler_tools = {
        str(key): str(value)
        for key, value in sorted(profilers.items())
        if key in {"nsys", "ncu", "perfetto", "trace_processor_shell", "xctrace"}
        and value not in {None, ""}
    }
    nccl = _available_version(packages, "nccl", "pynccl") or _available_version(tools, "nccl")
    multi_gpu = bool(compatible and gpu_count >= 2 and fabric.get("multi_gpu", False))
    multi_node = bool(fabric.get("multi_node", False))
    selected_environment = os.environ if environment is None else environment
    return HardwareCapabilities(
        hardware_manifest=hardware_evidence,
        software_manifest=software_evidence,
        cuda_gpu_available=cuda_gpu,
        helix_cuda_compatible=compatible,
        gpu_count=gpu_count,
        gpu_models=models,
        cuda_toolkit_available=cuda_toolkit,
        nvidia_smi_available=nvidia_smi,
        nvlink_available=bool(nvidia.get("nvlink", False)),
        multi_gpu_available=multi_gpu,
        infiniband_available=bool(fabric.get("infiniband", False)),
        rdma_available=bool(fabric.get("rdma", False)),
        multi_node_available=multi_node,
        pytorch_available=_available_version(packages, "torch", "pytorch"),
        vllm_available=_available_version(packages, "vllm"),
        sglang_available=_available_version(packages, "sglang"),
        nccl_available=nccl,
        profiler_tools=profiler_tools,
        budget=_budget_status(selected_environment),
        detection_policy=(
            "manifests are consumed as evidence; this module performs no live hardware probe",
            "CUDA compatibility requires real-manifest provenance, GPU availability, CUDA toolkit, nvidia-smi, and explicit Helix support",
            "multi-GPU requires at least two declared GPUs plus Fabric multi_gpu capability",
            "multi-node requires explicit Fabric multi_node capability; it is not inferred from a NIC",
            "package and profiler availability requires a non-null manifest version or path",
        ),
    )


def _base_command(
    study: HardwareStudyKind,
    *,
    seed: int,
    matrix_path: Path,
    output_root: Path,
    timeout_seconds: int,
    currently_runnable: bool,
) -> FutureCommand:
    output = output_root / study.value
    argv = (
        "uv",
        "run",
        "--locked",
        "sloforge",
        "helix",
        "characterize",
        "run",
        "--matrix",
        matrix_path.as_posix(),
        "--output",
        output.as_posix(),
        "--hardware",
        "gpu",
        "--seed",
        str(seed),
        "--max-experiments",
        "100000",
        "--timeout-seconds",
        str(timeout_seconds),
    )
    required = {
        HardwareStudyKind.GPU: ("compatible CUDA GPU",),
        HardwareStudyKind.MULTI_GPU: ("at least two compatible CUDA GPUs", "NCCL"),
        HardwareStudyKind.MULTI_NODE: ("multi-node allocation", "network fabric"),
    }[study]
    return FutureCommand(
        command_id=f"characterize-{study.value}",
        study=study,
        purpose=(
            "Execute the existing bounded characterization matrix; the matrix/runtime must "
            "emit exercised hardware evidence for the covered operation set."
        ),
        covered_operations=tuple(sorted(_REQUIRED_OPERATIONS[study], key=lambda item: item.value)),
        argv=argv,
        shell_rendering=shlex.join(argv),
        environment={"SLOFORGE_BRANCHFABRIC_CHARACTERIZATION_ALLOW_GPU": "1"},
        timeout_seconds=timeout_seconds,
        expected_output=output.as_posix(),
        required_capabilities=required,
        currently_runnable=currently_runnable,
    )


def _profile_commands(
    base: FutureCommand,
    capabilities: HardwareCapabilities,
) -> tuple[FutureCommand, ...]:
    commands: list[FutureCommand] = []
    nsys = capabilities.profiler_tools.get("nsys")
    if nsys is not None:
        output = f"{base.expected_output}/profilers/nsight-systems"
        argv = (
            nsys,
            "profile",
            "--trace=cuda,nvtx,osrt",
            "--sample=none",
            "--force-overwrite=false",
            "--output",
            output,
            *base.argv,
        )
        commands.append(
            FutureCommand(
                command_id=f"nsight-systems-{base.study.value}",
                study=base.study,
                purpose="Capture CUDA, NVTX, and OS-runtime timing around the bounded run.",
                covered_operations=base.covered_operations,
                argv=argv,
                shell_rendering=shlex.join(argv),
                environment=base.environment,
                timeout_seconds=base.timeout_seconds,
                expected_output=f"{output}.nsys-rep",
                required_capabilities=(*base.required_capabilities, "Nsight Systems"),
                currently_runnable=base.currently_runnable,
            )
        )
    ncu = capabilities.profiler_tools.get("ncu")
    if ncu is not None and base.study is HardwareStudyKind.GPU:
        output = f"{base.expected_output}/profilers/nsight-compute"
        argv = (
            ncu,
            "--set",
            "full",
            "--force-overwrite=false",
            "--export",
            output,
            *base.argv,
        )
        commands.append(
            FutureCommand(
                command_id="nsight-compute-gpu",
                study=base.study,
                purpose="Capture GPU kernel, memory-bandwidth, and launch metrics.",
                covered_operations=base.covered_operations,
                argv=argv,
                shell_rendering=shlex.join(argv),
                environment=base.environment,
                timeout_seconds=base.timeout_seconds,
                expected_output=f"{output}.ncu-rep",
                required_capabilities=(*base.required_capabilities, "Nsight Compute"),
                currently_runnable=base.currently_runnable,
            )
        )
    perfetto = capabilities.profiler_tools.get("perfetto")
    if perfetto is not None:
        config = Path("benchmarks/branchfabric/perfetto-system-trace.cfg")
        output = f"{base.expected_output}/profilers/system.perfetto-trace"
        argv = (perfetto, "-c", config.as_posix(), "-o", output)
        commands.append(
            FutureCommand(
                command_id=f"perfetto-system-{base.study.value}",
                study=base.study,
                purpose=(
                    "Capture a system Perfetto trace alongside the Helix Perfetto export; "
                    "run the characterization command during the bounded capture window."
                ),
                covered_operations=base.covered_operations,
                argv=argv,
                shell_rendering=shlex.join(argv),
                environment={},
                timeout_seconds=base.timeout_seconds,
                expected_output=output,
                required_capabilities=("Perfetto executable",),
                required_files=(config.as_posix(),),
                currently_runnable=base.currently_runnable and config.is_file(),
            )
        )
    return tuple(commands)


def generate_future_commands(
    capabilities: HardwareCapabilities,
    *,
    seed: int,
    matrix_path: Path = Path("benchmarks/branchfabric/characterization.yaml"),
    output_root: Path = Path("artifacts/branchfabric/characterization/hardware"),
    timeout_seconds: int = 3600,
) -> dict[HardwareStudyKind, tuple[FutureCommand, ...]]:
    if not 0 <= seed <= 2**63 - 1:
        raise ValueError("seed must fit signed 64-bit")
    if not 1 <= timeout_seconds <= 86_400:
        raise ValueError("timeout_seconds must be within 1..86400")
    runnable = {
        HardwareStudyKind.GPU: capabilities.helix_cuda_compatible,
        HardwareStudyKind.MULTI_GPU: (
            capabilities.multi_gpu_available and capabilities.nccl_available
        ),
        HardwareStudyKind.MULTI_NODE: capabilities.multi_node_available,
    }
    result: dict[HardwareStudyKind, tuple[FutureCommand, ...]] = {}
    for study in HardwareStudyKind:
        base = _base_command(
            study,
            seed=seed,
            matrix_path=matrix_path,
            output_root=output_root,
            timeout_seconds=timeout_seconds,
            currently_runnable=runnable[study],
        )
        result[study] = (base, *_profile_commands(base, capabilities))
    return result


def _verify_exercised_artifact(
    artifact_input: ExercisedArtifactInput,
    *,
    capabilities: HardwareCapabilities,
    seed: int,
) -> VerifiedExercisedArtifact:
    payload = _read_bounded(artifact_input.path, MAX_EXERCISED_ARTIFACT_BYTES)
    digest = hashlib.sha256(payload).hexdigest()
    if digest != artifact_input.expected_sha256:
        raise ValueError(f"exercised artifact hash mismatch: {artifact_input.path}")
    document = ExercisedHardwareArtifact.model_validate_json(payload)
    if document.study is not artifact_input.study:
        raise ValueError("exercised artifact study does not match its supplied slot")
    if document.seed != seed:
        raise ValueError("exercised artifact seed does not match the requested study seed")
    if document.hardware_manifest_sha256 != capabilities.hardware_manifest.sha256:
        raise ValueError("exercised artifact hardware manifest hash mismatch")
    if document.software_manifest_sha256 != capabilities.software_manifest.sha256:
        raise ValueError("exercised artifact software manifest hash mismatch")
    verified: list[EvidenceFile] = []
    for evidence in document.evidence_files:
        path = Path(evidence.path)
        if not path.is_absolute():
            path = artifact_input.path.parent / path
        observed_hash, observed_size = _sha256_file(path, MAX_EVIDENCE_FILE_BYTES)
        if observed_hash != evidence.sha256 or observed_size != evidence.size_bytes:
            raise ValueError(f"raw exercised evidence hash or size mismatch: {path}")
        verified.append(evidence)
    return VerifiedExercisedArtifact(
        reference=artifact_input.path.as_posix(),
        sha256=digest,
        size_bytes=len(payload),
        document=document,
        verified_evidence_files=tuple(verified),
    )


def _capabilities_satisfied(study: HardwareStudyKind, capabilities: HardwareCapabilities) -> bool:
    if study is HardwareStudyKind.GPU:
        return capabilities.helix_cuda_compatible
    if study is HardwareStudyKind.MULTI_GPU:
        return capabilities.multi_gpu_available and capabilities.nccl_available
    return capabilities.multi_node_available


def _unavailable_reason(
    study: HardwareStudyKind,
    *,
    capabilities_satisfied: bool,
    verified: VerifiedExercisedArtifact | None,
) -> str:
    if not capabilities_satisfied:
        return (
            f"Preserved manifests do not establish the capabilities required for {study.value}; "
            "no hardware metrics were generated or inferred."
        )
    if verified is None:
        return (
            f"Capabilities may support {study.value}, but no hash-pinned exercised artifact was "
            "supplied; capability is not measurement."
        )
    missing = sorted(
        operation.value
        for operation in _REQUIRED_OPERATIONS[study]
        if operation not in verified.document.operations_exercised
    )
    return (
        "The exercised artifact is incomplete for this study; missing operations: "
        + ", ".join(missing)
        + ". Partial metrics are excluded from the hardware report."
    )


def _report(
    study: HardwareStudyKind,
    *,
    seed: int,
    capabilities: HardwareCapabilities,
    verified: VerifiedExercisedArtifact | None,
    commands: tuple[FutureCommand, ...],
) -> HardwareStudyReport:
    satisfied = _capabilities_satisfied(study, capabilities)
    complete = bool(
        verified is not None
        and _REQUIRED_OPERATIONS[study].issubset(verified.document.operations_exercised)
    )
    required = tuple(sorted(_REQUIRED_OPERATIONS[study], key=lambda item: item.value))
    if satisfied and complete and verified is not None:
        return HardwareStudyReport(
            study=study,
            status=MeasurementStatus.HARDWARE_BACKED_REAL,
            seed=seed,
            reason=(
                "Capabilities are established and every required operation is backed by a "
                "hash-verified exercised artifact and raw evidence."
            ),
            capabilities_satisfied=True,
            required_operations=required,
            exercised_artifact=verified,
            metrics=verified.document.metrics,
            future_commands=commands,
            hardware_measurement_missing=False,
            limitations=(
                "Metrics describe only the supplied machine, runtime versions, seed, and command.",
                "Artifact validation establishes integrity and provenance, not production representativeness.",
            ),
        )
    return HardwareStudyReport(
        study=study,
        status=MeasurementStatus.UNAVAILABLE,
        seed=seed,
        reason=_unavailable_reason(
            study,
            capabilities_satisfied=satisfied,
            verified=verified,
        ),
        capabilities_satisfied=satisfied,
        required_operations=required,
        exercised_artifact=None,
        metrics=(),
        future_commands=commands,
        hardware_measurement_missing=True,
        limitations=(
            "No synthetic transfer, kernel, topology, bandwidth, or latency value substitutes for missing hardware evidence.",
            "Future commands do not provision resources and do not overwrite an existing output directory.",
        ),
    )


def build_hardware_studies(
    hardware_manifest: Path,
    software_manifest: Path,
    *,
    seed: int,
    exercised_artifacts: Sequence[ExercisedArtifactInput] = (),
    environment: Mapping[str, str] | None = None,
    matrix_path: Path = Path("benchmarks/branchfabric/characterization.yaml"),
    output_root: Path = Path("artifacts/branchfabric/characterization/hardware"),
    command_timeout_seconds: int = 3600,
) -> HardwareStudiesBundle:
    """Build three reports without performing a hardware or cloud operation."""

    if not 0 <= seed <= 2**63 - 1:
        raise ValueError("seed must fit signed 64-bit")
    if len(exercised_artifacts) > len(HardwareStudyKind):
        raise ValueError("at most one exercised artifact per hardware study is supported")
    supplied = {item.study: item for item in exercised_artifacts}
    if len(supplied) != len(exercised_artifacts):
        raise ValueError("exercised artifact study slots must be unique")
    capabilities = detect_hardware_capabilities(
        hardware_manifest,
        software_manifest,
        environment=environment,
    )
    commands = generate_future_commands(
        capabilities,
        seed=seed,
        matrix_path=matrix_path,
        output_root=output_root,
        timeout_seconds=command_timeout_seconds,
    )
    verified = {
        study: _verify_exercised_artifact(item, capabilities=capabilities, seed=seed)
        for study, item in supplied.items()
    }
    reports = {
        study: _report(
            study,
            seed=seed,
            capabilities=capabilities,
            verified=verified.get(study),
            commands=commands[study],
        )
        for study in HardwareStudyKind
    }
    return HardwareStudiesBundle(
        seed=seed,
        capabilities=capabilities,
        gpu=reports[HardwareStudyKind.GPU],
        multi_gpu=reports[HardwareStudyKind.MULTI_GPU],
        multi_node=reports[HardwareStudyKind.MULTI_NODE],
    )


def write_hardware_studies(bundle: HardwareStudiesBundle, output: Path) -> str:
    """Write one report bundle exclusively and return its exact SHA-256."""

    payload = (bundle.model_dump_json(indent=2) + "\n").encode()
    if len(payload) > MAX_EXERCISED_ARTIFACT_BYTES:
        raise ValueError("hardware studies report exceeds the output bound")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "GPU_BUDGET_ENV",
    "BudgetStatus",
    "EvidenceFile",
    "ExercisedArtifactInput",
    "ExercisedHardwareArtifact",
    "FutureCommand",
    "HardwareCapabilities",
    "HardwareMetric",
    "HardwareOperation",
    "HardwareStudiesBundle",
    "HardwareStudyKind",
    "HardwareStudyReport",
    "ManifestEvidence",
    "MeasurementStatus",
    "VerifiedExercisedArtifact",
    "build_hardware_studies",
    "detect_hardware_capabilities",
    "generate_future_commands",
    "write_hardware_studies",
]
