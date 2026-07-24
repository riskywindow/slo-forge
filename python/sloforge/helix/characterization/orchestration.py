"""Append-only, restartable orchestration for bounded characterization stages.

This module owns run control, not characterization logic.  Stage implementations
are imported lazily and invoked through their public Python APIs in a killable
worker process.  Run inputs and the run manifest are immutable.  Attempts,
checkpoints, and status snapshots are append-only; resume never removes or
overwrites evidence left by an interrupted worker.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from enum import Enum, StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sloforge.helix.characterization.matrix import expand_matrix, load_matrix

MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_ARTIFACTS_PER_ATTEMPT = 100_000
MAX_ERROR_BYTES = 16 * 1024
MAX_STAGE_ATTEMPTS = 5
_WORKER_RESULT = "worker-result.json"


class OrchestrationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class CharacterizationStage(StrEnum):
    MATRIX_VALIDATE = "matrix_validate"
    VERTICAL_TRACE = "vertical_trace"
    CONTINUUM_STATE = "continuum_state"
    ENVIRONMENT_STATE = "environment_state"
    METADATA = "metadata"
    INSTRUMENTATION_OVERHEAD = "instrumentation_overhead"
    GPU = "gpu"
    MULTI_GPU = "multi_gpu"
    MULTI_NODE = "multi_node"


CPU_STAGES: tuple[CharacterizationStage, ...] = (
    CharacterizationStage.MATRIX_VALIDATE,
    CharacterizationStage.VERTICAL_TRACE,
    CharacterizationStage.CONTINUUM_STATE,
    CharacterizationStage.ENVIRONMENT_STATE,
    CharacterizationStage.METADATA,
    CharacterizationStage.INSTRUMENTATION_OVERHEAD,
)

DEFAULT_STAGES: tuple[CharacterizationStage, ...] = (
    *CPU_STAGES,
    CharacterizationStage.GPU,
    CharacterizationStage.MULTI_GPU,
    CharacterizationStage.MULTI_NODE,
)

_HARDWARE_STAGES = frozenset(
    {
        CharacterizationStage.GPU,
        CharacterizationStage.MULTI_GPU,
        CharacterizationStage.MULTI_NODE,
    }
)


class StageStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"
    SKIPPED_UNAVAILABLE = "SKIPPED_UNAVAILABLE"


_TERMINAL_STAGE_STATUSES = frozenset(
    {
        StageStatus.SUCCEEDED,
        StageStatus.FAILED,
        StageStatus.TIMED_OUT,
        StageStatus.SKIPPED_UNAVAILABLE,
    }
)
_SUCCESSFUL_STAGE_STATUSES = frozenset({StageStatus.SUCCEEDED, StageStatus.SKIPPED_UNAVAILABLE})


class RunState(StrEnum):
    ACTIVE = "ACTIVE"
    COMPLETE = "COMPLETE"
    COMPLETE_WITH_FAILURES = "COMPLETE_WITH_FAILURES"


class OrchestrationConfig(OrchestrationModel):
    schema_version: Literal["sloforge.branchfabric.orchestration-config/v1"] = (
        "sloforge.branchfabric.orchestration-config/v1"
    )
    seed: Annotated[int, Field(ge=0, le=2**63 - 1)]
    stages: tuple[CharacterizationStage, ...] = DEFAULT_STAGES
    stage_timeout_seconds: float = Field(gt=0.0, le=3600.0, allow_inf_nan=False, default=300.0)
    maximum_attempts: int = Field(default=2, ge=1, le=MAX_STAGE_ATTEMPTS)
    continue_on_error: bool = True
    environment_repetitions: int = Field(default=1, ge=1, le=20)
    environment_warmups: int = Field(default=0, ge=0, le=5)
    metadata_operations_per_thread: int = Field(default=8, ge=1, le=256)
    metadata_repetitions: int = Field(default=3, ge=1, le=20)
    metadata_warmups: int = Field(default=1, ge=0, le=5)
    metadata_thread_counts: tuple[int, ...] = (1, 2)
    overhead_repetitions: int = Field(default=3, ge=1, le=20)

    @model_validator(mode="after")
    def stage_plan_is_bounded(self) -> Self:
        if not self.stages or len(set(self.stages)) != len(self.stages):
            raise ValueError("stages must be non-empty and unique")
        if self.stages[0] is not CharacterizationStage.MATRIX_VALIDATE:
            raise ValueError("matrix_validate must be the first stage")
        if (
            not self.metadata_thread_counts
            or len(set(self.metadata_thread_counts)) != len(self.metadata_thread_counts)
            or tuple(sorted(self.metadata_thread_counts)) != self.metadata_thread_counts
            or self.metadata_thread_counts[0] != 1
            or self.metadata_thread_counts[-1] > 16
        ):
            raise ValueError(
                "metadata_thread_counts must be unique, increasing, begin at 1, and not exceed 16"
            )
        if self.metadata_operations_per_thread * self.metadata_thread_counts[-1] > 4096:
            raise ValueError("metadata operation count exceeds the per-sample bound")
        return self


class HardwareCapabilityEvidence(OrchestrationModel):
    schema_version: Literal["sloforge.branchfabric.hardware-capability-evidence/v1"] = (
        "sloforge.branchfabric.hardware-capability-evidence/v1"
    )
    source_present: bool
    source_reference: str
    preserved_reference: str | None
    source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    compatible_gpu_available: bool
    gpu_count: int = Field(ge=0)
    multi_gpu_available: bool
    multi_node_available: bool
    nvidia_smi_available: bool
    cuda_toolkit_available: bool
    rdma_available: bool
    evidence_class: Literal["HARDWARE_BACKED_REAL", "UNAVAILABLE"]
    reason: str = Field(min_length=1, max_length=2048)


class CharacterizationRunManifest(OrchestrationModel):
    schema_version: Literal["sloforge.branchfabric.characterization-run-manifest/v1"] = (
        "sloforge.branchfabric.characterization-run-manifest/v1"
    )
    run_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    created_at: str
    config: OrchestrationConfig
    matrix_source_reference: str
    preserved_matrix_reference: Literal["inputs/characterization.yaml"] = (
        "inputs/characterization.yaml"
    )
    matrix_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    matrix_id: str
    matrix_case_count: int = Field(ge=1, le=100_000)
    matrix_cases_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sweep_ids: tuple[str, ...] = Field(min_length=1, max_length=4096)
    distribution_claim: Literal[False] = False
    capability_evidence: HardwareCapabilityEvidence
    software_source_reference: str
    preserved_software_reference: str | None
    software_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    immutable: Literal[True] = True


class ArtifactHash(OrchestrationModel):
    relative_path: str = Field(min_length=1, max_length=4096)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class StageError(OrchestrationModel):
    error_type: str = Field(min_length=1, max_length=512)
    message: str = Field(min_length=1, max_length=4096)
    traceback: str | None = Field(default=None, max_length=MAX_ERROR_BYTES)


class StageCheckpoint(OrchestrationModel):
    schema_version: Literal["sloforge.branchfabric.stage-checkpoint/v1"] = (
        "sloforge.branchfabric.stage-checkpoint/v1"
    )
    sequence: int = Field(ge=0)
    run_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    stage: CharacterizationStage
    attempt: int = Field(ge=0, lt=MAX_STAGE_ATTEMPTS)
    status: StageStatus
    stage_seed: Annotated[int, Field(ge=0, le=2**63 - 1)]
    command: tuple[str, ...] = Field(min_length=1, max_length=64)
    timeout_seconds: float = Field(gt=0.0, le=3600.0, allow_inf_nan=False)
    started_at: str
    finished_at: str
    duration_ns: int = Field(ge=0)
    worker_exit_code: int | None
    attempt_reference: str = Field(min_length=1, max_length=4096)
    artifacts: tuple[ArtifactHash, ...] = Field(max_length=MAX_ARTIFACTS_PER_ATTEMPT)
    error: StageError | None
    previous_checkpoint_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    immutable: Literal[True] = True

    @model_validator(mode="after")
    def error_matches_status(self) -> Self:
        if self.status in {StageStatus.FAILED, StageStatus.TIMED_OUT} and self.error is None:
            raise ValueError("failed and timed-out checkpoints require an error")
        if self.status in _SUCCESSFUL_STAGE_STATUSES and self.error is not None:
            raise ValueError("successful and skipped checkpoints cannot contain an error")
        if self.status in {StageStatus.PENDING, StageStatus.RUNNING}:
            raise ValueError("checkpoints must have a terminal attempt status")
        return self


class RunStatusSnapshot(OrchestrationModel):
    schema_version: Literal["sloforge.branchfabric.run-status/v1"] = (
        "sloforge.branchfabric.run-status/v1"
    )
    sequence: int = Field(ge=0)
    run_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    run_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observed_at: str
    run_state: RunState
    stages: dict[CharacterizationStage, StageStatus]
    running_stage: CharacterizationStage | None
    checkpoint_reference: str | None
    checkpoint_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    previous_status_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    immutable: Literal[True] = True


class CharacterizationRunResult(OrchestrationModel):
    schema_version: Literal["sloforge.branchfabric.characterization-run-result/v1"] = (
        "sloforge.branchfabric.characterization-run-result/v1"
    )
    run_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    run_directory: str
    run_state: RunState
    stages: dict[CharacterizationStage, StageStatus]
    checkpoint_count: int = Field(ge=0)
    status_snapshot_count: int = Field(ge=1)
    latest_status_reference: str


_COMMANDS: dict[CharacterizationStage, tuple[str, ...]] = {
    CharacterizationStage.MATRIX_VALIDATE: (
        "python-api",
        "sloforge.helix.characterization.matrix.load_matrix+expand_matrix",
    ),
    CharacterizationStage.VERTICAL_TRACE: (
        "python-api",
        "sloforge.helix.characterization.runner.run_vertical_trace",
    ),
    CharacterizationStage.CONTINUUM_STATE: (
        "python-api",
        "sloforge.continuum.characterization.run_continuum_characterization",
    ),
    CharacterizationStage.ENVIRONMENT_STATE: (
        "python-api",
        "sloforge.helix.characterization.environment_study.run_environment_study",
    ),
    CharacterizationStage.METADATA: (
        "python-api",
        "sloforge.helix.characterization.metadata_study.run_metadata_study",
    ),
    CharacterizationStage.INSTRUMENTATION_OVERHEAD: (
        "python-api",
        "sloforge.continuum.characterization.measure_instrumentation_overhead",
    ),
    CharacterizationStage.GPU: ("not-executed", "compatible GPU unavailable"),
    CharacterizationStage.MULTI_GPU: ("not-executed", "multi-GPU unavailable"),
    CharacterizationStage.MULTI_NODE: ("not-executed", "multi-node unavailable"),
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_bounded(path: Path, maximum: int = MAX_MANIFEST_BYTES) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"input must be a regular non-symlink file: {path}")
    with path.open("rb") as handle:
        payload = handle.read(maximum + 1)
    if len(payload) > maximum:
        raise ValueError(f"input exceeds {maximum} bytes: {path}")
    return payload


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()


def _exclusive_bytes(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    return _sha256_bytes(payload)


def _exclusive_json(path: Path, value: object) -> str:
    payload = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
        + b"\n"
    )
    if len(payload) > MAX_MANIFEST_BYTES:
        raise ValueError(f"JSON record exceeds {MAX_MANIFEST_BYTES} bytes")
    return _exclusive_bytes(path, payload)


def _model_json(path: Path, model: BaseModel) -> str:
    return _exclusive_json(path, model.model_dump(mode="json"))


def _jsonable(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported stage result value {type(value).__name__}")


def _hardware_evidence(source: Path | None, *, run_directory: Path) -> HardwareCapabilityEvidence:
    reference = source.as_posix() if source is not None else "not-provided"
    if source is None or not source.exists():
        return HardwareCapabilityEvidence(
            source_present=False,
            source_reference=reference,
            preserved_reference=None,
            compatible_gpu_available=False,
            gpu_count=0,
            multi_gpu_available=False,
            multi_node_available=False,
            nvidia_smi_available=False,
            cuda_toolkit_available=False,
            rdma_available=False,
            evidence_class="UNAVAILABLE",
            reason="No hardware capability manifest was provided or found; hardware stages fail closed.",
        )
    payload = _read_bounded(source)
    document = json.loads(payload)
    if not isinstance(document, dict):
        raise ValueError("hardware capability manifest must be a JSON object")
    preserved = run_directory / "inputs" / "hardware-capability.json"
    digest = _exclusive_bytes(preserved, payload)
    gpu = document.get("gpu")
    fabric = document.get("fabric")
    nvidia = gpu.get("nvidia") if isinstance(gpu, dict) else None
    nvidia = nvidia if isinstance(nvidia, dict) else {}
    fabric = fabric if isinstance(fabric, dict) else {}
    evidence_class: Literal["HARDWARE_BACKED_REAL", "UNAVAILABLE"] = (
        "HARDWARE_BACKED_REAL"
        if document.get("measurement_class") == "HARDWARE_BACKED_REAL"
        else "UNAVAILABLE"
    )
    compatible = bool(
        evidence_class == "HARDWARE_BACKED_REAL"
        and nvidia.get("available", False)
        and nvidia.get("hardware_backed_helix_supported", False)
    )
    devices = nvidia.get("devices")
    gpu_count = len(devices) if isinstance(devices, list) else (1 if compatible else 0)
    multi_gpu = bool(fabric.get("multi_gpu", False) and gpu_count >= 2)
    multi_node = bool(fabric.get("multi_node", False))
    return HardwareCapabilityEvidence(
        source_present=True,
        source_reference=reference,
        preserved_reference="inputs/hardware-capability.json",
        source_sha256=digest,
        compatible_gpu_available=compatible,
        gpu_count=gpu_count,
        multi_gpu_available=multi_gpu,
        multi_node_available=multi_node,
        nvidia_smi_available=bool(nvidia.get("nvidia_smi", False)),
        cuda_toolkit_available=bool(nvidia.get("cuda_toolkit", False)),
        rdma_available=bool(fabric.get("rdma", False)),
        evidence_class=evidence_class,
        reason=(
            "Capability fields were preserved verbatim and evaluated conservatively; "
            "availability does not imply that a characterization implementation exists."
        ),
    )


def _case_digest(cases: Sequence[BaseModel]) -> str:
    return _sha256_bytes(_canonical([case.model_dump(mode="json") for case in cases]))


def create_characterization_run(
    run_directory: Path,
    *,
    matrix_path: Path,
    config: OrchestrationConfig,
    hardware_capability_manifest: Path | None = Path(
        "artifacts/branchfabric/manifests/hardware-baseline.json"
    ),
    software_manifest: Path | None = Path(
        "artifacts/branchfabric/manifests/software-baseline.json"
    ),
) -> CharacterizationRunManifest:
    """Create immutable run inputs, manifest, and the initial pending status."""

    matrix_payload = _read_bounded(matrix_path)
    matrix = load_matrix(matrix_path)
    cases = expand_matrix(matrix)
    if run_directory.exists():
        raise FileExistsError("characterization run directory already exists")
    run_directory.mkdir(parents=True, exist_ok=False)
    for child in ("inputs", "attempts", "checkpoints", "status"):
        (run_directory / child).mkdir(mode=0o700)
    matrix_digest = _exclusive_bytes(
        run_directory / "inputs" / "characterization.yaml", matrix_payload
    )
    capability = _hardware_evidence(
        hardware_capability_manifest,
        run_directory=run_directory,
    )
    software_reference = software_manifest.as_posix() if software_manifest else "not-provided"
    preserved_software_reference: str | None = None
    software_sha256: str | None = None
    if software_manifest is not None and software_manifest.exists():
        software_payload = _read_bounded(software_manifest)
        software_document = json.loads(software_payload)
        if not isinstance(software_document, dict):
            raise ValueError("software manifest must be a JSON object")
        preserved_software_reference = "inputs/software-baseline.json"
        software_sha256 = _exclusive_bytes(
            run_directory / preserved_software_reference,
            software_payload,
        )
    identity = {
        "schema": "sloforge.branchfabric.characterization-run-identity/v1",
        "config": config.model_dump(mode="json"),
        "matrix_sha256": matrix_digest,
        "matrix_cases_sha256": _case_digest(cases),
        "capability_sha256": capability.source_sha256,
        "software_sha256": software_sha256,
    }
    run_id = _sha256_bytes(_canonical(identity))[:32]
    manifest = CharacterizationRunManifest(
        run_id=run_id,
        created_at=_now(),
        config=config,
        matrix_source_reference=matrix_path.as_posix(),
        matrix_sha256=matrix_digest,
        matrix_id=matrix.matrix_id,
        matrix_case_count=len(cases),
        matrix_cases_sha256=_case_digest(cases),
        sweep_ids=tuple(item.sweep_id for item in matrix.sweeps),
        capability_evidence=capability,
        software_source_reference=software_reference,
        preserved_software_reference=preserved_software_reference,
        software_sha256=software_sha256,
    )
    manifest_digest = _model_json(run_directory / "run-manifest.json", manifest)
    initial_stages = {stage: StageStatus.PENDING for stage in config.stages}
    initial = RunStatusSnapshot(
        sequence=0,
        run_id=run_id,
        run_manifest_sha256=manifest_digest,
        observed_at=_now(),
        run_state=RunState.ACTIVE,
        stages=initial_stages,
        running_stage=None,
        checkpoint_reference=None,
        checkpoint_sha256=None,
        previous_status_sha256=None,
    )
    _model_json(run_directory / "status" / "000000.json", initial)
    return manifest


def load_characterization_run(run_directory: Path) -> CharacterizationRunManifest:
    """Load a run and verify its preserved immutable input hashes."""

    manifest_path = run_directory / "run-manifest.json"
    manifest = CharacterizationRunManifest.model_validate_json(_read_bounded(manifest_path))
    matrix_path = run_directory / manifest.preserved_matrix_reference
    if _sha256_bytes(_read_bounded(matrix_path)) != manifest.matrix_sha256:
        raise ValueError("preserved characterization matrix hash mismatch")
    matrix = load_matrix(matrix_path)
    cases = expand_matrix(matrix)
    if (
        len(cases) != manifest.matrix_case_count
        or _case_digest(cases) != manifest.matrix_cases_sha256
    ):
        raise ValueError("preserved characterization matrix expansion mismatch")
    if (
        matrix.matrix_id != manifest.matrix_id
        or tuple(item.sweep_id for item in matrix.sweeps) != manifest.sweep_ids
    ):
        raise ValueError("preserved characterization matrix identity mismatch")
    capability = manifest.capability_evidence
    if capability.preserved_reference is not None:
        path = run_directory / capability.preserved_reference
        if _sha256_bytes(_read_bounded(path)) != capability.source_sha256:
            raise ValueError("preserved hardware capability evidence hash mismatch")
    if manifest.preserved_software_reference is not None:
        software_path = run_directory / manifest.preserved_software_reference
        if _sha256_bytes(_read_bounded(software_path)) != manifest.software_sha256:
            raise ValueError("preserved software manifest hash mismatch")
    identity = {
        "schema": "sloforge.branchfabric.characterization-run-identity/v1",
        "config": manifest.config.model_dump(mode="json"),
        "matrix_sha256": manifest.matrix_sha256,
        "matrix_cases_sha256": manifest.matrix_cases_sha256,
        "capability_sha256": capability.source_sha256,
        "software_sha256": manifest.software_sha256,
    }
    if _sha256_bytes(_canonical(identity))[:32] != manifest.run_id:
        raise ValueError("immutable run identity does not match the run manifest")
    initial_status = RunStatusSnapshot.model_validate_json(
        _read_bounded(run_directory / "status" / "000000.json")
    )
    if initial_status.run_manifest_sha256 != _sha256_bytes(_read_bounded(manifest_path)):
        raise ValueError("initial status run-manifest hash mismatch")
    return manifest


def _checkpoint_files(run_directory: Path) -> tuple[Path, ...]:
    return tuple(sorted((run_directory / "checkpoints").glob("*.json")))


def _status_files(run_directory: Path) -> tuple[Path, ...]:
    return tuple(sorted((run_directory / "status").glob("*.json")))


def _load_checkpoints(
    run_directory: Path, manifest: CharacterizationRunManifest
) -> tuple[tuple[StageCheckpoint, str, Path], ...]:
    records: list[tuple[StageCheckpoint, str, Path]] = []
    previous: str | None = None
    for expected, path in enumerate(_checkpoint_files(run_directory)):
        payload = _read_bounded(path)
        digest = _sha256_bytes(payload)
        checkpoint = StageCheckpoint.model_validate_json(payload)
        if checkpoint.sequence != expected or checkpoint.run_id != manifest.run_id:
            raise ValueError("checkpoint sequence or run identity mismatch")
        if checkpoint.previous_checkpoint_sha256 != previous:
            raise ValueError("checkpoint hash chain mismatch")
        records.append((checkpoint, digest, path))
        previous = digest
    return tuple(records)


def _load_statuses(
    run_directory: Path, manifest: CharacterizationRunManifest
) -> tuple[tuple[RunStatusSnapshot, str, Path], ...]:
    records: list[tuple[RunStatusSnapshot, str, Path]] = []
    previous: str | None = None
    manifest_digest, _manifest_size = _hash_file(run_directory / "run-manifest.json")
    for expected, path in enumerate(_status_files(run_directory)):
        payload = _read_bounded(path)
        digest = _sha256_bytes(payload)
        status = RunStatusSnapshot.model_validate_json(payload)
        if status.sequence != expected or status.run_id != manifest.run_id:
            raise ValueError("status sequence or run identity mismatch")
        if status.run_manifest_sha256 != manifest_digest:
            raise ValueError("status journal run-manifest hash mismatch")
        if status.previous_status_sha256 != previous:
            raise ValueError("status hash chain mismatch")
        records.append((status, digest, path))
        previous = digest
    if not records:
        raise ValueError("characterization run has no status journal")
    return tuple(records)


def _derived_stage_statuses(
    manifest: CharacterizationRunManifest,
    checkpoints: Sequence[tuple[StageCheckpoint, str, Path]],
) -> dict[CharacterizationStage, StageStatus]:
    statuses = {stage: StageStatus.PENDING for stage in manifest.config.stages}
    for checkpoint, _digest, _path in checkpoints:
        if checkpoint.stage not in statuses:
            raise ValueError("checkpoint names a stage outside the immutable run plan")
        statuses[checkpoint.stage] = checkpoint.status
    return statuses


def _run_state(statuses: Mapping[CharacterizationStage, StageStatus]) -> RunState:
    if any(status in {StageStatus.PENDING, StageStatus.RUNNING} for status in statuses.values()):
        return RunState.ACTIVE
    if any(status in {StageStatus.FAILED, StageStatus.TIMED_OUT} for status in statuses.values()):
        return RunState.COMPLETE_WITH_FAILURES
    return RunState.COMPLETE


def _append_status(
    run_directory: Path,
    manifest: CharacterizationRunManifest,
    *,
    stages: Mapping[CharacterizationStage, StageStatus],
    running_stage: CharacterizationStage | None,
    checkpoint: tuple[Path, str] | None,
) -> tuple[RunStatusSnapshot, str, Path]:
    statuses = _load_statuses(run_directory, manifest)
    previous_status_sha = statuses[-1][1]
    sequence = len(statuses)
    checkpoint_path, checkpoint_sha = checkpoint if checkpoint is not None else (None, None)
    status = RunStatusSnapshot(
        sequence=sequence,
        run_id=manifest.run_id,
        run_manifest_sha256=statuses[0][0].run_manifest_sha256,
        observed_at=_now(),
        run_state=_run_state(stages),
        stages=dict(stages),
        running_stage=running_stage,
        checkpoint_reference=(
            checkpoint_path.relative_to(run_directory).as_posix()
            if checkpoint_path is not None
            else None
        ),
        checkpoint_sha256=checkpoint_sha,
        previous_status_sha256=previous_status_sha,
    )
    path = run_directory / "status" / f"{sequence:06d}.json"
    digest = _model_json(path, status)
    return status, digest, path


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _artifact_hashes(run_directory: Path, attempt_directory: Path) -> tuple[ArtifactHash, ...]:
    artifacts: list[ArtifactHash] = []
    for path in sorted(attempt_directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"stage output cannot contain symlinks: {path}")
        if not path.is_file():
            continue
        digest, size = _hash_file(path)
        artifacts.append(
            ArtifactHash(
                relative_path=path.relative_to(run_directory).as_posix(),
                size_bytes=size,
                sha256=digest,
            )
        )
        if len(artifacts) > MAX_ARTIFACTS_PER_ATTEMPT:
            raise ValueError("stage attempt exceeds artifact count bound")
    return tuple(artifacts)


def _stage_seed(seed: int, _stage: CharacterizationStage) -> int:
    # Keep the user's declared run seed intact across stages and retries.  The
    # existing Helix demo accepts specific explicitly tested seeds; replacing
    # one with a derived pseudo-random value would change the workload itself.
    return seed


def _append_checkpoint(
    run_directory: Path,
    manifest: CharacterizationRunManifest,
    *,
    stage: CharacterizationStage,
    attempt: int,
    status: StageStatus,
    stage_seed: int,
    started_at: str,
    finished_at: str,
    duration_ns: int,
    worker_exit_code: int | None,
    attempt_directory: Path,
    error: StageError | None,
) -> tuple[StageCheckpoint, str, Path]:
    prior = _load_checkpoints(run_directory, manifest)
    sequence = len(prior)
    previous = prior[-1][1] if prior else None
    checkpoint = StageCheckpoint(
        sequence=sequence,
        run_id=manifest.run_id,
        stage=stage,
        attempt=attempt,
        status=status,
        stage_seed=stage_seed,
        command=_COMMANDS[stage],
        timeout_seconds=manifest.config.stage_timeout_seconds,
        started_at=started_at,
        finished_at=finished_at,
        duration_ns=duration_ns,
        worker_exit_code=worker_exit_code,
        attempt_reference=attempt_directory.relative_to(run_directory).as_posix(),
        artifacts=_artifact_hashes(run_directory, attempt_directory),
        error=error,
        previous_checkpoint_sha256=previous,
    )
    path = (
        run_directory / "checkpoints" / f"{sequence:06d}-{stage.value}-attempt-{attempt:03d}.json"
    )
    digest = _model_json(path, checkpoint)
    return checkpoint, digest, path


def _attempt_directories(run_directory: Path, stage: CharacterizationStage) -> dict[int, Path]:
    parent = run_directory / "attempts" / stage.value
    if not parent.exists():
        return {}
    results: dict[int, Path] = {}
    for path in sorted(parent.glob("attempt-*")):
        try:
            attempt = int(path.name.removeprefix("attempt-"))
        except ValueError as error:
            raise ValueError(f"invalid attempt directory {path}") from error
        if attempt in results or path.is_symlink() or not path.is_dir():
            raise ValueError(f"invalid or duplicate attempt directory {path}")
        results[attempt] = path
    return results


def _recover_orphan_attempts(
    run_directory: Path,
    manifest: CharacterizationRunManifest,
    stage: CharacterizationStage,
) -> None:
    checkpoints = _load_checkpoints(run_directory, manifest)
    recorded = {item.attempt for item, _digest, _path in checkpoints if item.stage is stage}
    for attempt, path in _attempt_directories(run_directory, stage).items():
        if attempt in recorded:
            continue
        timestamp = _now()
        error = StageError(
            error_type="InterruptedAttempt",
            message=(
                "Attempt directory exists without an immutable checkpoint; preserved and "
                "classified as interrupted during resume."
            ),
        )
        _checkpoint, digest, checkpoint_path = _append_checkpoint(
            run_directory,
            manifest,
            stage=stage,
            attempt=attempt,
            status=StageStatus.FAILED,
            stage_seed=_stage_seed(manifest.config.seed, stage),
            started_at=timestamp,
            finished_at=timestamp,
            duration_ns=0,
            worker_exit_code=None,
            attempt_directory=path,
            error=error,
        )
        statuses = _derived_stage_statuses(manifest, _load_checkpoints(run_directory, manifest))
        _append_status(
            run_directory,
            manifest,
            stages=statuses,
            running_stage=None,
            checkpoint=(checkpoint_path, digest),
        )


def _reconcile_status_with_checkpoints(
    run_directory: Path, manifest: CharacterizationRunManifest
) -> None:
    """Close the narrow crash window between checkpoint and status publication."""

    checkpoints = _load_checkpoints(run_directory, manifest)
    derived = _derived_stage_statuses(manifest, checkpoints)
    latest = _load_statuses(run_directory, manifest)[-1][0]
    if latest.stages == derived and latest.running_stage is None:
        return
    checkpoint_reference: tuple[Path, str] | None = None
    if checkpoints:
        _checkpoint, digest, path = checkpoints[-1]
        checkpoint_reference = (path, digest)
    _append_status(
        run_directory,
        manifest,
        stages=derived,
        running_stage=None,
        checkpoint=checkpoint_reference,
    )


def _skip_reason(stage: CharacterizationStage, evidence: HardwareCapabilityEvidence) -> str:
    if stage is CharacterizationStage.GPU:
        if not evidence.compatible_gpu_available:
            return "compatible NVIDIA/CUDA Helix GPU unavailable in preserved capability evidence"
        return "GPU capability exists, but no GPU stage implementation is available in this run"
    if stage is CharacterizationStage.MULTI_GPU:
        if not evidence.multi_gpu_available:
            return "multi-GPU capability unavailable in preserved capability evidence"
        return "multi-GPU capability exists, but no multi-GPU stage implementation is available"
    if not evidence.multi_node_available:
        return "multi-node capability unavailable in preserved capability evidence"
    return "multi-node capability exists, but no multi-node stage implementation is available"


def _write_hardware_skip(
    attempt_directory: Path,
    *,
    stage: CharacterizationStage,
    evidence: HardwareCapabilityEvidence,
) -> None:
    _exclusive_json(
        attempt_directory / "skip-evidence.json",
        {
            "schema_version": "sloforge.branchfabric.hardware-stage-skip/v1",
            "stage": stage.value,
            "status": StageStatus.SKIPPED_UNAVAILABLE.value,
            "reason": _skip_reason(stage, evidence),
            "capability_evidence": evidence.model_dump(mode="json"),
            "no_simulated_measurement_substituted": True,
        },
    )


def _worker_outcome(attempt_directory: Path) -> tuple[StageStatus, StageError | None]:
    path = attempt_directory / _WORKER_RESULT
    if not path.is_file():
        return (
            StageStatus.FAILED,
            StageError(
                error_type="MissingWorkerResult",
                message="Stage worker exited without an immutable worker-result.json record.",
            ),
        )
    document = json.loads(_read_bounded(path))
    if not isinstance(document, dict):
        raise ValueError("worker result must be a JSON object")
    status = StageStatus(str(document.get("status")))
    if status is StageStatus.SUCCEEDED or status is StageStatus.SKIPPED_UNAVAILABLE:
        return status, None
    raw_error = document.get("error")
    if not isinstance(raw_error, dict):
        return (
            StageStatus.FAILED,
            StageError(error_type="MalformedWorkerError", message="Worker error record is absent."),
        )
    return StageStatus.FAILED, StageError.model_validate(raw_error)


def _run_worker(
    run_directory: Path,
    *,
    stage: CharacterizationStage,
    attempt_directory: Path,
    stage_seed: int,
    timeout_seconds: float,
) -> tuple[StageStatus, StageError | None, int | None, int]:
    command = (
        sys.executable,
        "-m",
        "sloforge.helix.characterization.orchestration",
        "_worker",
        run_directory.as_posix(),
        stage.value,
        attempt_directory.as_posix(),
        str(stage_seed),
    )
    started_ns = time.perf_counter_ns()
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5.0)
        duration = time.perf_counter_ns() - started_ns
        return (
            StageStatus.TIMED_OUT,
            StageError(
                error_type="StageTimeout",
                message=f"Stage exceeded its {timeout_seconds:g} second bounded timeout.",
            ),
            process.returncode,
            duration,
        )
    duration = time.perf_counter_ns() - started_ns
    status, error = _worker_outcome(attempt_directory)
    if process.returncode != 0 and status in _SUCCESSFUL_STAGE_STATUSES:
        status = StageStatus.FAILED
        error = StageError(
            error_type="WorkerExitError",
            message=f"Stage worker exited with code {process.returncode}.",
        )
    return status, error, process.returncode, duration


def _execute_attempt(
    run_directory: Path,
    manifest: CharacterizationRunManifest,
    stage: CharacterizationStage,
    attempt: int,
) -> StageCheckpoint:
    attempt_directory = run_directory / "attempts" / stage.value / f"attempt-{attempt:03d}"
    attempt_directory.mkdir(parents=True, exist_ok=False)
    checkpoints = _load_checkpoints(run_directory, manifest)
    stage_statuses = _derived_stage_statuses(manifest, checkpoints)
    stage_statuses[stage] = StageStatus.RUNNING
    _append_status(
        run_directory,
        manifest,
        stages=stage_statuses,
        running_stage=stage,
        checkpoint=None,
    )
    stage_seed = _stage_seed(manifest.config.seed, stage)
    started_at = _now()
    if stage in _HARDWARE_STAGES:
        _write_hardware_skip(
            attempt_directory,
            stage=stage,
            evidence=manifest.capability_evidence,
        )
        status = StageStatus.SKIPPED_UNAVAILABLE
        error = None
        worker_exit_code = None
        duration_ns = 0
    else:
        status, error, worker_exit_code, duration_ns = _run_worker(
            run_directory,
            stage=stage,
            attempt_directory=attempt_directory,
            stage_seed=stage_seed,
            timeout_seconds=manifest.config.stage_timeout_seconds,
        )
    checkpoint, digest, checkpoint_path = _append_checkpoint(
        run_directory,
        manifest,
        stage=stage,
        attempt=attempt,
        status=status,
        stage_seed=stage_seed,
        started_at=started_at,
        finished_at=_now(),
        duration_ns=duration_ns,
        worker_exit_code=worker_exit_code,
        attempt_directory=attempt_directory,
        error=error,
    )
    stage_statuses = _derived_stage_statuses(manifest, _load_checkpoints(run_directory, manifest))
    _append_status(
        run_directory,
        manifest,
        stages=stage_statuses,
        running_stage=None,
        checkpoint=(checkpoint_path, digest),
    )
    return checkpoint


def _stage_attempt_count(
    checkpoints: Sequence[tuple[StageCheckpoint, str, Path]], stage: CharacterizationStage
) -> int:
    return sum(1 for item, _digest, _path in checkpoints if item.stage is stage)


def resume_characterization_run(run_directory: Path) -> CharacterizationRunResult:
    """Resume pending/failed stages without modifying any prior attempt or journal record."""

    manifest = load_characterization_run(run_directory)
    for stage in manifest.config.stages:
        _recover_orphan_attempts(run_directory, manifest, stage)
        _reconcile_status_with_checkpoints(run_directory, manifest)
        while True:
            checkpoints = _load_checkpoints(run_directory, manifest)
            statuses = _derived_stage_statuses(manifest, checkpoints)
            status = statuses[stage]
            attempts = _stage_attempt_count(checkpoints, stage)
            if status in _SUCCESSFUL_STAGE_STATUSES:
                break
            if attempts >= manifest.config.maximum_attempts:
                break
            checkpoint = _execute_attempt(run_directory, manifest, stage, attempts)
            if checkpoint.status in _SUCCESSFUL_STAGE_STATUSES:
                break
            if not manifest.config.continue_on_error:
                return characterization_run_result(run_directory)
    return characterization_run_result(run_directory)


def characterization_run_result(run_directory: Path) -> CharacterizationRunResult:
    manifest = load_characterization_run(run_directory)
    checkpoints = _load_checkpoints(run_directory, manifest)
    statuses = _derived_stage_statuses(manifest, checkpoints)
    status_records = _load_statuses(run_directory, manifest)
    latest_path = status_records[-1][2]
    return CharacterizationRunResult(
        run_id=manifest.run_id,
        run_directory=run_directory.as_posix(),
        run_state=_run_state(statuses),
        stages=statuses,
        checkpoint_count=len(checkpoints),
        status_snapshot_count=len(status_records),
        latest_status_reference=latest_path.relative_to(run_directory).as_posix(),
    )


def run_characterization(
    run_directory: Path,
    *,
    matrix_path: Path,
    config: OrchestrationConfig,
    hardware_capability_manifest: Path | None = Path(
        "artifacts/branchfabric/manifests/hardware-baseline.json"
    ),
    software_manifest: Path | None = Path(
        "artifacts/branchfabric/manifests/software-baseline.json"
    ),
) -> CharacterizationRunResult:
    """Create and execute one new bounded characterization run."""

    create_characterization_run(
        run_directory,
        matrix_path=matrix_path,
        config=config,
        hardware_capability_manifest=hardware_capability_manifest,
        software_manifest=software_manifest,
    )
    return resume_characterization_run(run_directory)


def _import(name: str) -> Any:
    return importlib.import_module(name)


def _execute_stage_api(
    run_directory: Path,
    stage: CharacterizationStage,
    attempt_directory: Path,
    stage_seed: int,
) -> object:
    manifest = load_characterization_run(run_directory)
    config = manifest.config
    if stage is CharacterizationStage.MATRIX_VALIDATE:
        matrix = load_matrix(run_directory / manifest.preserved_matrix_reference)
        cases = expand_matrix(matrix)
        result = {
            "schema_version": "sloforge.branchfabric.matrix-validation/v1",
            "matrix_id": matrix.matrix_id,
            "case_count": len(cases),
            "case_digest": _case_digest(cases),
            "distribution_claim": matrix.distribution_claim,
            "run_order_seed": matrix.run_order_seed,
            "stage_seed": stage_seed,
        }
        _exclusive_json(attempt_directory / "matrix-validation.json", result)
        return result
    if stage is CharacterizationStage.VERTICAL_TRACE:
        module = _import("sloforge.helix.characterization.runner")
        if (
            manifest.capability_evidence.preserved_reference is None
            or manifest.preserved_software_reference is None
        ):
            raise FileNotFoundError(
                "vertical trace requires preserved hardware and software manifests"
            )
        hardware_path = run_directory / manifest.capability_evidence.preserved_reference
        software_path = run_directory / manifest.preserved_software_reference
        software_document = json.loads(_read_bounded(software_path))
        hardware_document = json.loads(_read_bounded(hardware_path))
        if not isinstance(software_document, dict) or not isinstance(hardware_document, dict):
            raise ValueError("vertical trace manifests must be JSON objects")
        if not isinstance(software_document.get("host"), dict):
            host = hardware_document.get("host")
            if not isinstance(host, dict):
                raise ValueError("hardware manifest cannot supply missing software host fields")
            software_document = {
                **software_document,
                "host": host,
                "orchestration_derivation": {
                    "schema_version": "sloforge.branchfabric.manifest-derivation/v1",
                    "reason": "runner requires host fields stored in the hardware baseline",
                    "hardware_source_sha256": manifest.capability_evidence.source_sha256,
                    "software_source_sha256": manifest.software_sha256,
                },
            }
            software_path = attempt_directory / "vertical-software-manifest.json"
            _exclusive_json(software_path, software_document)
        result = module.run_vertical_trace(
            attempt_directory / "vertical-trace",
            seed=stage_seed,
            replace=False,
            hardware_baseline=hardware_path,
            software_baseline=software_path,
        )
        return result
    if stage is CharacterizationStage.CONTINUUM_STATE:
        module = _import("sloforge.continuum.characterization")
        result = module.run_continuum_characterization(seed=stage_seed)
        _exclusive_json(
            attempt_directory / "continuum-characterization.json",
            _jsonable(result),
        )
        return result
    if stage is CharacterizationStage.ENVIRONMENT_STATE:
        module = _import("sloforge.helix.characterization.environment_study")
        matrix_module = _import("sloforge.helix.characterization.matrix")
        result = module.run_environment_study(
            attempt_directory / "environment-study",
            seed=stage_seed,
            repetitions=config.environment_repetitions,
            warmups=config.environment_warmups,
            workloads=(matrix_module.EnvironmentSize.TINY,),
            trace_levels=(matrix_module.TraceLevel.FULL,),
            trial_timeout_seconds=min(config.stage_timeout_seconds, 300.0),
        )
        return result
    if stage is CharacterizationStage.METADATA:
        module = _import("sloforge.helix.characterization.metadata_study")
        metadata_config = module.MetadataStudyConfig(
            seed=stage_seed,
            operations_per_thread=config.metadata_operations_per_thread,
            warmup_repetitions=config.metadata_warmups,
            measurement_repetitions=config.metadata_repetitions,
            thread_counts=config.metadata_thread_counts,
            include_software_baselines=True,
            sample_timeout_seconds=min(config.stage_timeout_seconds, 300.0),
        )
        result = module.run_metadata_study(metadata_config)
        module.write_metadata_study(result, attempt_directory / "metadata-study.json")
        return result
    if stage is CharacterizationStage.INSTRUMENTATION_OVERHEAD:
        module = _import("sloforge.continuum.characterization")
        result = module.measure_instrumentation_overhead(
            seed=stage_seed,
            repetitions=config.overhead_repetitions,
        )
        _exclusive_json(attempt_directory / "instrumentation-overhead.json", _jsonable(result))
        return result
    raise ValueError(f"hardware stage {stage.value} must be skipped by the parent orchestrator")


def _worker_entry(arguments: Sequence[str]) -> int:
    if len(arguments) != 5 or arguments[0] != "_worker":
        return 64
    run_directory = Path(arguments[1])
    stage = CharacterizationStage(arguments[2])
    attempt_directory = Path(arguments[3])
    stage_seed = int(arguments[4])
    try:
        result = _execute_stage_api(run_directory, stage, attempt_directory, stage_seed)
    except ModuleNotFoundError as error:
        _exclusive_json(
            attempt_directory / _WORKER_RESULT,
            {
                "schema_version": "sloforge.branchfabric.stage-worker-result/v1",
                "status": StageStatus.SKIPPED_UNAVAILABLE.value,
                "reason": f"optional stage module unavailable: {error.name}",
                "stage": stage.value,
            },
        )
        return 0
    except Exception as error:  # stage boundary intentionally records arbitrary failures
        trace = traceback.format_exc()
        encoded = trace.encode()[:MAX_ERROR_BYTES]
        clipped = encoded.decode(errors="replace")
        _exclusive_json(
            attempt_directory / _WORKER_RESULT,
            {
                "schema_version": "sloforge.branchfabric.stage-worker-result/v1",
                "status": StageStatus.FAILED.value,
                "stage": stage.value,
                "error": StageError(
                    error_type=type(error).__name__,
                    message=str(error)[:4096] or type(error).__name__,
                    traceback=clipped,
                ).model_dump(mode="json"),
            },
        )
        return 1
    _exclusive_json(
        attempt_directory / _WORKER_RESULT,
        {
            "schema_version": "sloforge.branchfabric.stage-worker-result/v1",
            "status": StageStatus.SUCCEEDED.value,
            "stage": stage.value,
            "result_type": type(result).__name__,
        },
    )
    return 0


__all__ = [
    "CPU_STAGES",
    "DEFAULT_STAGES",
    "ArtifactHash",
    "CharacterizationRunManifest",
    "CharacterizationRunResult",
    "CharacterizationStage",
    "HardwareCapabilityEvidence",
    "OrchestrationConfig",
    "RunState",
    "RunStatusSnapshot",
    "StageCheckpoint",
    "StageError",
    "StageStatus",
    "characterization_run_result",
    "create_characterization_run",
    "load_characterization_run",
    "resume_characterization_run",
    "run_characterization",
]


if __name__ == "__main__":
    raise SystemExit(_worker_entry(sys.argv[1:]))
