"""Ephemeral two-A100 Modal entry point for Experiment 004 capacity calibration."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import modal
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

MODAL_SDK_VERSION = "1.5.3"
APP_NAME = "sloforge-branchfabric-capacity-calibration-004"
GPU_REQUEST = "A100-80GB:2"
GPU_COUNT = 2
GPU_FUNCTION_TIMEOUT_SECONDS = 160
GPU_FUNCTION_STARTUP_TIMEOUT_SECONDS = 180
GPU_COORDINATOR_RESERVATION_WALL_SECONDS = 212
MODEL_VOLUME_NAME = "sloforge-model-cache"
RESULTS_VOLUME_NAME = "sloforge-branchfabric-results"
REMOTE_EXPERIMENT_PREFIX = "experiment-004/calibration/modal"
MODEL_ID: Literal["Qwen/Qwen2.5-7B-Instruct"] = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION: Literal["a09a35458c702b33eeacc393d103063234e8bc28"] = (
    "a09a35458c702b33eeacc393d103063234e8bc28"
)
_ATTEMPT = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9-]{7,95}$")]

_SOURCE = Path(__file__).resolve()
_REPOSITORY_CANDIDATE = _SOURCE.parents[2] if len(_SOURCE.parents) > 2 else None
_LOCAL_REPOSITORY_AVAILABLE = bool(
    _REPOSITORY_CANDIDATE is not None and (_REPOSITORY_CANDIDATE / "pyproject.toml").is_file()
)
LOCAL_REPOSITORY_ROOT = (
    _REPOSITORY_CANDIDATE
    if _LOCAL_REPOSITORY_AVAILABLE and _REPOSITORY_CANDIDATE is not None
    else Path("/opt/sloforge")
)
LOCAL_EXPERIMENT_ROOT = (
    LOCAL_REPOSITORY_ROOT / "artifacts/branchfabric/gpu-validation/experiment-004"
)
MODEL_MOUNT = Path("/models")
RESULTS_MOUNT = Path("/results")
_LAUNCH_TOKEN = os.getenv("SLOFORGE_MODAL_PREFLIGHT_TOKEN", "")
_CLOUD_GRAPH_ENABLED = len(_LAUNCH_TOKEN) == 64 and all(
    character in "0123456789abcdef" for character in _LAUNCH_TOKEN
)
_MODAL_CLI_RUN = "modal" in Path(sys.argv[0]).as_posix().lower() and "run" in sys.argv[1:]
if _MODAL_CLI_RUN and not _CLOUD_GRAPH_ENABLED:
    raise RuntimeError("refusing to hydrate capacity calibration without budget preflight")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class CapacityCalibrationConfig(_StrictModel):
    schema_version: Literal["sloforge.branchfabric.capacity-modal-config/v1"] = (
        "sloforge.branchfabric.capacity-modal-config/v1"
    )
    attempt_id: _ATTEMPT
    seed: int = Field(ge=0, lt=1 << 63)
    model: Literal["Qwen/Qwen2.5-7B-Instruct"] = MODEL_ID
    model_revision: Literal["a09a35458c702b33eeacc393d103063234e8bc28"] = MODEL_REVISION
    tokenizer_revision: Literal["a09a35458c702b33eeacc393d103063234e8bc28"] = MODEL_REVISION
    runtime: Literal["vllm"] = "vllm"
    runtime_version: Literal["0.23.0"] = "0.23.0"
    dtype: Literal["bf16"] = "bf16"
    tracing_level: Literal["minimal"] = "minimal"
    serving_prompt_tokens: Literal[256] = 256
    serving_output_tokens: Literal[64] = 64
    gpu_memory_utilization: float = Field(default=0.72, gt=0.0, le=0.85)
    warmup_seconds: float = Field(default=1.0, ge=1.0, le=5.0)
    minimum_measurement_seconds: float = Field(default=10.0, ge=10.0, le=20.0)
    maximum_measurement_seconds: float = Field(default=10.0, ge=10.0, le=20.0)
    maximum_probes: Literal[3] = 3
    rate_grid_rps: tuple[float, ...] = (10.0, 15.0, 17.25, 20.0, 23.5, 25.0, 30.0, 35.0)
    serving_slo_maximum_ttft_seconds: float = Field(default=2.0, ge=2.0, le=2.0)
    worker_initialization_timeout_seconds: float = Field(default=115.0, ge=60.0, le=120.0)
    probe_timeout_seconds: float = Field(default=20.0, ge=15.0, le=25.0)
    tail_drain_seconds: float = Field(default=1.0, ge=1.0, le=2.0)
    producer_queue_capacity: int = Field(default=256, ge=64, le=512)
    controller_timeout_seconds: float = Field(default=150.0, ge=120.0, le=150.0)
    worker_shutdown_timeout_seconds: float = Field(default=10.0, ge=5.0, le=15.0)
    maximum_allocation_wall_seconds: float = Field(default=212.0, ge=212.0, le=212.0)

    @model_validator(mode="after")
    def bounded_calibration(self) -> Self:
        if self.maximum_measurement_seconds < self.minimum_measurement_seconds:
            raise ValueError("maximum measurement duration precedes minimum")
        if tuple(sorted(set(self.rate_grid_rps))) != self.rate_grid_rps:
            raise ValueError("calibration rate grid must be sorted and unique")
        if self.controller_timeout_seconds >= self.maximum_allocation_wall_seconds:
            raise ValueError("controller timeout must leave outer cleanup headroom")
        return self


class RemoteAuthorization(_StrictModel):
    schema_version: Literal["sloforge.branchfabric.capacity-authorization/v1"] = (
        "sloforge.branchfabric.capacity-authorization/v1"
    )
    reservation_id: _ATTEMPT
    config_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    maximum_gpu_seconds: Literal[424] = 424
    budget_usd: float = Field(gt=0.0, allow_inf_nan=False)


def _canonical_bytes(value: Any) -> bytes:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode()


def _write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(_canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inventory_tree(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "relative_path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def _validate_model_snapshot(snapshot: Path) -> dict[str, Any]:
    manifest_path = snapshot / "MODEL_MANIFEST.json"
    inputs_path = snapshot / "BRANCHFABRIC_INPUTS.json"
    if not manifest_path.is_file() or not inputs_path.is_file():
        raise FileNotFoundError("pinned model cache lacks calibration inputs or manifest")
    manifest = json.loads(manifest_path.read_text())
    expected = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
    }
    if not isinstance(manifest, dict) or any(
        manifest.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("cached model identity differs from the pinned calibration model")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("cached model manifest has no immutable file inventory")
    for item in files:
        path = snapshot / str(item["relative_path"])
        if path.is_symlink() or not path.is_file() or path.stat().st_size != int(item["bytes"]):
            raise ValueError(f"cached model file failed identity/size validation: {path}")
    # Full multi-GiB content hashes were already verified when this authorized
    # immutable cache snapshot was created and again by v9. Rehashing it while
    # GPUs are allocated consumed material budget in v9, so calibration binds
    # the existing hash inventory by its own digest and validates every size.
    return {
        "schema_version": "sloforge.branchfabric.capacity-model-cache-validation/v1",
        "model_manifest_sha256": _sha256(manifest_path),
        "model_manifest_file_count": len(files),
        "file_sizes_verified": True,
        "content_hashes_verified_during_gpu_invocation": False,
        "content_hash_basis": "immutable authorized cache manifest previously verified by v9",
    }


def _validate_authorization(
    config: CapacityCalibrationConfig, payload: dict[str, Any]
) -> RemoteAuthorization:
    authorization = RemoteAuthorization.model_validate_json(
        json.dumps(payload, sort_keys=True, separators=(",", ":")), strict=True
    )
    if authorization.config_sha256 != hashlib.sha256(_canonical_bytes(config)).hexdigest():
        raise RuntimeError("capacity authorization does not match the immutable config")
    return authorization


def _validate_local_reservation(config: CapacityCalibrationConfig, reservation_id: str) -> None:
    from sloforge.helix.characterization.gpu_reclamation_methodology import (
        Experiment004GpuHourLedger,
    )

    ledger = Experiment004GpuHourLedger.model_validate_json(
        (LOCAL_EXPERIMENT_ROOT / "gpu-hours.json").read_text(), strict=True
    )
    matches = tuple(item for item in ledger.reservations if item.reservation_id == reservation_id)
    if len(matches) != 1:
        raise RuntimeError("capacity GPU reservation is absent or duplicated")
    reservation = matches[0]
    if (
        reservation.invocation_id != config.attempt_id
        or reservation.maximum_wall_seconds != GPU_COORDINATOR_RESERVATION_WALL_SECONDS
        or reservation.config_sha256 != hashlib.sha256(_canonical_bytes(config)).hexdigest()
    ):
        raise RuntimeError("capacity GPU reservation differs from the Modal call")


model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=False)
results_volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=False)
gpu_image = importlib.import_module("modal_real_gpu_cow").gpu_image
app = modal.App(
    APP_NAME,
    tags={"project": "sloforge", "experiment": "branchfabric-capacity-calibration-004"},
)
_run_calibration_function: Any = None


def run_calibration(
    config_payload: dict[str, Any], authorization_payload: dict[str, Any]
) -> dict[str, Any]:
    config = CapacityCalibrationConfig.model_validate_json(
        json.dumps(config_payload, sort_keys=True, separators=(",", ":")), strict=True
    )
    authorization = _validate_authorization(config, authorization_payload)
    if modal.__version__ != MODAL_SDK_VERSION:
        raise RuntimeError(f"Modal SDK must be {MODAL_SDK_VERSION}, observed {modal.__version__}")
    function_call_id = modal.current_function_call_id()
    if not function_call_id:
        raise RuntimeError("Modal did not expose the capacity FunctionCall ID")
    started_ns = time.monotonic_ns()
    work_root = Path("/tmp/sloforge-branchfabric-capacity") / config.attempt_id
    config_path = work_root.parent / f"{config.attempt_id}.json"
    snapshot = MODEL_MOUNT / "snapshots" / MODEL_REVISION
    _write_new(config_path, config)
    controller: dict[str, Any]
    run_error: dict[str, str] | None = None
    topology: list[Any] = []
    model_cache_validation: dict[str, Any] | None = None
    try:
        model_cache_validation = _validate_model_snapshot(snapshot)
        from sloforge.helix.characterization.gpu_reclamation_instrumentation import (
            capture_topology_commands,
            validate_topology_capture,
        )

        topology = list(capture_topology_commands())
        validate_topology_capture(tuple(topology), expected_gpu_count=2)
        from gpu_capacity_calibration_controller import run_capacity_calibration_controller

        controller = run_capacity_calibration_controller(
            config_path=config_path,
            work_root=work_root,
            worker_path=Path(
                "/opt/sloforge/experiments/branchfabric/gpu_capacity_calibration_worker.py"
            ),
            model_snapshot=snapshot,
        )
    except BaseException as caught:
        work_root.mkdir(parents=True, exist_ok=True)
        run_error = {
            "type": type(caught).__name__,
            "message": str(caught),
            "traceback": traceback.format_exc(),
        }
        controller = {
            "schema_version": "sloforge.branchfabric.capacity-controller-failure/v1",
            "status": "failed",
            "controller_error": run_error,
        }
        _write_new(work_root / "function-failure.json", run_error)
    if model_cache_validation is not None:
        _write_new(work_root / "model-cache-validation.json", model_cache_validation)
    _write_new(
        work_root / "topology/topology-commands.json",
        [item.model_dump(mode="json") for item in topology],
    )
    elapsed_ns = time.monotonic_ns() - started_ns
    calibration_result_path = work_root / "calibration/calibration-result.json"
    calibration_result = (
        json.loads(calibration_result_path.read_text())
        if calibration_result_path.is_file()
        else None
    )
    completion = {
        "schema_version": "sloforge.branchfabric.capacity-function-completion/v1",
        "status": "succeeded" if controller.get("status") == "succeeded" else "failed",
        "attempt_id": config.attempt_id,
        "reservation_id": authorization.reservation_id,
        "function_call_id": function_call_id,
        "requested_gpu": GPU_REQUEST,
        "gpu_count": GPU_COUNT,
        "gpu_allocation_seconds": elapsed_ns / 1e9,
        "gpu_seconds": GPU_COUNT * elapsed_ns / 1e9,
        "calibration_status": (
            calibration_result["status"]
            if controller.get("status") == "succeeded" and calibration_result is not None
            else None
        ),
        "v10_launch_permitted": False,
        "review_required": bool(
            controller.get("status") == "succeeded"
            and calibration_result is not None
            and calibration_result["status"] == "ready"
        ),
        "run_error": run_error,
        "controller": controller,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    _write_new(work_root / "function-completion.json", completion)
    staging = RESULTS_MOUNT / f"{REMOTE_EXPERIMENT_PREFIX}/{config.attempt_id}.staging"
    final = RESULTS_MOUNT / f"{REMOTE_EXPERIMENT_PREFIX}/{config.attempt_id}"
    if staging.exists() or final.exists():
        raise FileExistsError("immutable capacity result prefix already exists")
    staging.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copytree(work_root, staging)
        _write_new(
            staging / "REMOTE_MANIFEST.json",
            {
                "schema_version": "sloforge.branchfabric.capacity-remote-manifest/v1",
                "attempt_id": config.attempt_id,
                "remote_prefix": str(final.relative_to(RESULTS_MOUNT)),
                "artifacts": _inventory_tree(staging),
            },
        )
        os.replace(staging, final)
        results_volume.commit()
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        results_volume.commit()
        raise
    return {
        **completion,
        "remote_prefix": str(final.relative_to(RESULTS_MOUNT)),
        "remote_manifest_sha256": _sha256(final / "REMOTE_MANIFEST.json"),
    }


def main(*, config_path: str) -> None:
    config = CapacityCalibrationConfig.model_validate_json(
        Path(config_path).read_text(), strict=True
    )
    budget_raw = os.environ.get("SLOFORGE_GPU_BUDGET_USD")
    reservation_id = os.environ.get("SLOFORGE_EXP004_RESERVATION_ID")
    if budget_raw is None or not float(budget_raw) > 0.0:
        raise RuntimeError("SLOFORGE_GPU_BUDGET_USD is required and must be positive")
    if reservation_id is None:
        raise RuntimeError("capacity GPU-hour reservation ID is required")
    _validate_local_reservation(config, reservation_id)
    authorization = RemoteAuthorization(
        reservation_id=reservation_id,
        config_sha256=hashlib.sha256(_canonical_bytes(config)).hexdigest(),
        budget_usd=float(budget_raw),
    )
    call = _run_calibration_function.spawn(
        config.model_dump(mode="json"), authorization.model_dump(mode="json")
    )
    result = call.get(timeout=GPU_FUNCTION_TIMEOUT_SECONDS + GPU_FUNCTION_STARTUP_TIMEOUT_SECONDS)
    print(
        _canonical_bytes(
            {
                "result": result,
                "materialized": {
                    "remote_path": f"{REMOTE_EXPERIMENT_PREFIX}/{config.attempt_id}",
                    "volume_name": RESULTS_VOLUME_NAME,
                },
            }
        ).decode(),
        end="",
    )


if _CLOUD_GRAPH_ENABLED:
    _run_calibration_function = app.function(
        image=gpu_image,
        gpu=GPU_REQUEST,
        volumes={
            str(MODEL_MOUNT): model_volume.with_mount_options(read_only=True),
            str(RESULTS_MOUNT): results_volume,
        },
        cpu=16.0,
        memory=64 * 1024,
        timeout=GPU_FUNCTION_TIMEOUT_SECONDS,
        startup_timeout=GPU_FUNCTION_STARTUP_TIMEOUT_SECONDS,
        retries=0,
        max_containers=1,
        buffer_containers=0,
        single_use_containers=True,
    )(run_calibration)
    _local_entrypoint = app.local_entrypoint()(main)


__all__ = [
    "APP_NAME",
    "GPU_REQUEST",
    "CapacityCalibrationConfig",
    "app",
    "main",
    "run_calibration",
]
