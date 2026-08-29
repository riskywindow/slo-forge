"""Budget-gated one-A100 Modal surface for Experiment 004 v11 micro-validation.

Importing this file without the sole coordinator's preflight token does not
hydrate a GPU function.  The app defines no endpoint and persists only the
pre-existing model-cache and BranchFabric-results volumes.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import shutil
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

import modal
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from sloforge.helix.characterization.gpu_reclamation_v11_methodology import (
    Experiment004V11MicroConfig,
    validate_bound_artifact,
)

MODAL_SDK_VERSION = "1.5.3"
APP_NAME = "sloforge-branchfabric-gpu-reclamation-004-v11-micro"
GPU_REQUEST = "A100-80GB:1"
GPU_COUNT = 1
GPU_FUNCTION_TIMEOUT_SECONDS = 525
GPU_FUNCTION_STARTUP_TIMEOUT_SECONDS = 180
POST_CONTROLLER_RESERVE_SECONDS = 10.0
MODEL_VOLUME_NAME = "sloforge-model-cache"
RESULTS_VOLUME_NAME = "sloforge-branchfabric-results"
REMOTE_EXPERIMENT_PREFIX = "experiment-004/v11/micro-validation/modal"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"

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
    raise RuntimeError("refusing to hydrate v11 micro GPU work without budget preflight")

_Attempt = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9-]{7,95}$")]
_Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class V11MicroRemoteAuthorization(_StrictModel):
    schema_version: Literal["sloforge.branchfabric.experiment-004-v11-micro-authorization/v1"] = (
        "sloforge.branchfabric.experiment-004-v11-micro-authorization/v1"
    )
    reservation_id: _Attempt
    config_sha256: _Sha256
    preflight_token_sha256: _Sha256
    requested_gpu: Literal["A100-80GB"] = "A100-80GB"
    gpu_count: Literal[1] = 1
    maximum_gpu_seconds: float = Field(gt=0, le=525.0)
    budget_usd: float = Field(gt=0.0, allow_inf_nan=False)


def _canonical_bytes(value: BaseModel | dict[str, Any]) -> bytes:
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
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


def _validate_authorization(
    config: Experiment004V11MicroConfig,
    payload: dict[str, Any],
) -> V11MicroRemoteAuthorization:
    authorization = V11MicroRemoteAuthorization.model_validate(payload)
    if authorization.config_sha256 != hashlib.sha256(_canonical_bytes(config)).hexdigest():
        raise RuntimeError("v11 micro authorization does not bind the immutable config")
    if authorization.preflight_token_sha256 != hashlib.sha256(_LAUNCH_TOKEN.encode()).hexdigest():
        raise RuntimeError("v11 micro authorization does not bind the coordinator preflight token")
    if not math.isclose(
        authorization.maximum_gpu_seconds,
        float(GPU_FUNCTION_TIMEOUT_SECONDS),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeError("v11 micro authorization does not reserve exactly one 525s A100")
    return authorization


def _validate_local_reservation(
    config: Experiment004V11MicroConfig,
    reservation_id: str,
) -> None:
    from sloforge.helix.characterization.gpu_reclamation_methodology import (
        Experiment004GpuHourLedger,
    )

    ledger_path = LOCAL_EXPERIMENT_ROOT / "gpu-hours.json"
    ledger = Experiment004GpuHourLedger.model_validate_json(
        ledger_path.resolve(strict=True).read_text(), strict=True
    )
    matches = [item for item in ledger.reservations if item.reservation_id == reservation_id]
    if len(matches) != 1:
        raise RuntimeError("v11 micro sole-coordinator reservation is absent or duplicated")
    reservation = matches[0]
    if (
        reservation.invocation_id != config.attempt_id
        or reservation.config_sha256 != hashlib.sha256(_canonical_bytes(config)).hexdigest()
        or reservation.gpu_count != 1
        or reservation.requested_gpu != "A100-80GB"
        or not math.isclose(
            reservation.maximum_wall_seconds,
            float(GPU_FUNCTION_TIMEOUT_SECONDS),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise RuntimeError("v11 micro ledger reservation differs from the Modal request")


def _validate_model_snapshot(snapshot: Path) -> dict[str, Any]:
    manifest_path = snapshot / "MODEL_MANIFEST.json"
    inputs_path = snapshot / "BRANCHFABRIC_INPUTS.json"
    if not manifest_path.is_file() or not inputs_path.is_file():
        raise FileNotFoundError("pinned model snapshot or exact tokenizer inputs are absent")
    manifest = json.loads(manifest_path.read_text())
    expected = {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
    }
    if not isinstance(manifest, dict) or any(
        manifest.get(key) != value for key, value in expected.items()
    ):
        raise RuntimeError("cached model identity differs from the v11 micro config")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError("cached model manifest has no immutable file inventory")
    for item in files:
        if not isinstance(item, dict):
            raise RuntimeError("cached model inventory row is invalid")
        path = snapshot / str(item["relative_path"])
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != int(item["bytes"])
            or _sha256(path) != str(item["sha256"])
        ):
            raise RuntimeError(f"cached model file failed validation: {path}")
    inputs = json.loads(inputs_path.read_text())
    if (
        not isinstance(inputs, dict)
        or len(inputs.get("prefix_token_ids", ())) < 16_384
        or len(inputs.get("divergent_token_ids", ())) < 8
    ):
        raise RuntimeError("cached exact 16K/fanout-8 tokenizer input is incomplete")
    return manifest


def _validate_config_artifacts(
    config: Experiment004V11MicroConfig, *, repository_root: Path
) -> None:
    validate_bound_artifact(
        repository_root,
        reference=config.offline_gate_manifest,
        expected_sha256=config.offline_gate_manifest_sha256,
    )
    validate_bound_artifact(
        repository_root,
        reference=config.budget_authorization,
        expected_sha256=config.budget_authorization_sha256,
    )


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


model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=False)
results_volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=False)
gpu_image = importlib.import_module("modal_real_gpu_cow").gpu_image
if _LOCAL_REPOSITORY_AVAILABLE and hasattr(gpu_image, "add_local_dir"):
    bundled = "/opt/sloforge/artifacts/branchfabric/gpu-validation/experiment-004"
    gate_root = LOCAL_EXPERIMENT_ROOT / "v11/integration-gates"
    if gate_root.is_dir():
        gpu_image = gpu_image.add_local_dir(
            gate_root, f"{bundled}/v11/integration-gates", copy=True
        )
    budget_path = LOCAL_EXPERIMENT_ROOT / "budget-authorization-v11.json"
    if budget_path.is_file():
        gpu_image = gpu_image.add_local_file(
            budget_path, f"{bundled}/budget-authorization-v11.json", copy=True
        )

app = modal.App(
    APP_NAME,
    tags={"project": "sloforge", "experiment": "branchfabric-gpu-validation-004-v11"},
)
_run_micro_function: Any = None
_launch_token_secret: Any = None


def run_micro(
    config_payload: dict[str, Any], authorization_payload: dict[str, Any]
) -> dict[str, Any]:
    entry_ns = time.monotonic_ns()
    config = Experiment004V11MicroConfig.model_validate(config_payload)
    authorization = _validate_authorization(config, authorization_payload)
    _validate_config_artifacts(config, repository_root=Path("/opt/sloforge"))
    if modal.__version__ != MODAL_SDK_VERSION:
        raise RuntimeError(f"Modal SDK must be exactly {MODAL_SDK_VERSION}")
    function_call_id = modal.current_function_call_id()
    if not function_call_id:
        raise RuntimeError("Modal did not expose the v11 micro FunctionCall ID")
    snapshot = MODEL_MOUNT / "snapshots" / MODEL_REVISION
    _validate_model_snapshot(snapshot)
    work_root = Path("/tmp/sloforge-branchfabric-exp004-v11") / config.attempt_id
    config_path = work_root.parent / f"{config.attempt_id}.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    _write_new(config_path, config)
    controller_deadline_ns = entry_ns + round(
        (GPU_FUNCTION_TIMEOUT_SECONDS - POST_CONTROLLER_RESERVE_SECONDS) * 1e9
    )
    controller: dict[str, Any]
    run_error: dict[str, Any] | None = None
    try:
        from gpu_reclamation_controller_v11 import run_one_gpu_micro_controller

        controller = run_one_gpu_micro_controller(
            config_path=config_path,
            work_root=work_root,
            worker_path=Path(
                "/opt/sloforge/experiments/branchfabric/gpu_reclamation_worker_v11.py"
            ),
            model_snapshot=snapshot,
            absolute_deadline_ns=controller_deadline_ns,
        )
        if controller.get("status") != "succeeded":
            raise RuntimeError("v11 one-GPU micro controller failed")
        worker_result = controller.get("worker_result")
        if not isinstance(worker_result, dict) or worker_result.get("status") != "succeeded":
            raise RuntimeError("v11 micro worker result failed closed")
    except BaseException as error:
        run_error = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        work_root.mkdir(parents=True, exist_ok=True)
        _write_new(work_root / "function-failure.json", run_error)
        controller = {
            "schema_version": "sloforge.branchfabric.experiment-004-v11-controller-failure/v1",
            "status": "failed",
            "run_error": run_error,
        }

    controller_interval_seconds = (time.monotonic_ns() - entry_ns) / 1e9
    completion = {
        "schema_version": "sloforge.branchfabric.experiment-004-v11-micro-completion/v1",
        "status": "succeeded" if run_error is None else "failed",
        "attempt_id": config.attempt_id,
        "reservation_id": authorization.reservation_id,
        "function_call_id": function_call_id,
        "requested_gpu": GPU_REQUEST,
        "gpu_count": GPU_COUNT,
        "controller_and_analysis_interval_seconds": controller_interval_seconds,
        "gpu_allocation_seconds_status": "pending-final-function-return",
        "controller": controller,
        "run_error": run_error,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    _write_new(work_root / "function-completion.json", completion)
    staging = RESULTS_MOUNT / f"{REMOTE_EXPERIMENT_PREFIX}/{config.attempt_id}.staging"
    final = RESULTS_MOUNT / f"{REMOTE_EXPERIMENT_PREFIX}/{config.attempt_id}"
    if staging.exists() or final.exists():
        raise FileExistsError("immutable v11 micro result prefix already exists")
    staging.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copytree(work_root, staging)
        _write_new(
            staging / "REMOTE_MANIFEST.json",
            {
                "schema_version": "sloforge.branchfabric.experiment-004-v11-remote-manifest/v1",
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
    final_elapsed_seconds = (time.monotonic_ns() - entry_ns) / 1e9
    return {
        **completion,
        "gpu_allocation_seconds": final_elapsed_seconds,
        "gpu_seconds": final_elapsed_seconds,
        "gpu_hours": final_elapsed_seconds / 3600.0,
        "remote_prefix": str(final.relative_to(RESULTS_MOUNT)),
        "remote_manifest_sha256": _sha256(final / "REMOTE_MANIFEST.json"),
    }


def main(*, config_path: str) -> None:
    config = Experiment004V11MicroConfig.model_validate_json(Path(config_path).read_text())
    _validate_config_artifacts(config, repository_root=LOCAL_REPOSITORY_ROOT)
    budget_raw = os.getenv("SLOFORGE_GPU_BUDGET_USD")
    reservation_id = os.getenv("SLOFORGE_EXP004_RESERVATION_ID")
    if budget_raw is None or not float(budget_raw) > 0:
        raise RuntimeError("SLOFORGE_GPU_BUDGET_USD is required and must be positive")
    if reservation_id is None or not _CLOUD_GRAPH_ENABLED:
        raise RuntimeError("v11 micro requires reservation and preflight-token evidence")
    _validate_local_reservation(config, reservation_id)
    authorization = V11MicroRemoteAuthorization(
        reservation_id=reservation_id,
        config_sha256=hashlib.sha256(_canonical_bytes(config)).hexdigest(),
        preflight_token_sha256=hashlib.sha256(_LAUNCH_TOKEN.encode()).hexdigest(),
        maximum_gpu_seconds=float(GPU_FUNCTION_TIMEOUT_SECONDS),
        budget_usd=float(budget_raw),
    )
    call = _run_micro_function.spawn(
        config.model_dump(mode="json"), authorization.model_dump(mode="json")
    )
    result = call.get(timeout=GPU_FUNCTION_TIMEOUT_SECONDS + GPU_FUNCTION_STARTUP_TIMEOUT_SECONDS)
    print(_canonical_bytes({"result": result}).decode(), end="")


if _CLOUD_GRAPH_ENABLED:
    # The local coordinator token gates graph hydration and must also be
    # available when the module is imported in the remote container. Modal
    # does not forward arbitrary local environment variables, so inject this
    # one-time capability through an ephemeral runtime Secret rather than
    # baking it into the image or trusting an unbound function argument.
    _launch_token_secret = modal.Secret.from_dict({"SLOFORGE_MODAL_PREFLIGHT_TOKEN": _LAUNCH_TOKEN})
    _run_micro_function = app.function(
        image=gpu_image,
        gpu=GPU_REQUEST,
        secrets=[_launch_token_secret],
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
    )(run_micro)
    _local_entrypoint = app.local_entrypoint()(main)


__all__ = [
    "APP_NAME",
    "GPU_REQUEST",
    "V11MicroRemoteAuthorization",
    "app",
    "main",
    "run_micro",
]
