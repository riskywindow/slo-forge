"""Paid, fail-closed Modal surface for one integrated Experiment 004 v11 run.

Importing this module without the sole coordinator's one-time preflight token
does not hydrate a GPU function.  The cloud graph requests exactly two
A100-80GB devices on one host, defines no endpoint, and persists only to the
pre-existing model-cache and BranchFabric-results volumes.

Modal's A100-80GB request cannot select PCIe versus SXM4.  The controller's
physical inventory is therefore retained verbatim and direct comparison with
the frozen v10 timing is allowed only when both names are exactly
``NVIDIA A100-SXM4-80GB``.
"""

from __future__ import annotations

import hashlib
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
    Experiment004V11IntegratedConfig,
    validate_bound_artifact,
)

MODAL_SDK_VERSION = "1.5.3"
APP_NAME = "sloforge-branchfabric-gpu-reclamation-004-v11-integrated"
GPU_REQUEST = "A100-80GB:2"
GPU_COUNT = 2
GPU_FUNCTION_TIMEOUT_SECONDS = 588
GPU_FUNCTION_STARTUP_TIMEOUT_SECONDS = 180
POST_CONTROLLER_RESERVE_SECONDS = 10.0
MAXIMUM_GPU_SECONDS = GPU_COUNT * GPU_FUNCTION_TIMEOUT_SECONDS
MODEL_VOLUME_NAME = "sloforge-model-cache"
RESULTS_VOLUME_NAME = "sloforge-branchfabric-results"
REMOTE_EXPERIMENT_PREFIX = "experiment-004/v11/integrated/modal"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
V10_DIRECT_COMPARISON_GPU_NAME = "NVIDIA A100-SXM4-80GB"
MODAL_CAN_PIN_A100_MEMORY_SIZE_BUT_NOT_INTERCONNECT = True
CUDA_IMAGE = "nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04"
UV_VERSION = "0.10.2"
VLLM_VERSION = "0.23.0"
TORCH_VERSION = "2.11.0"
TRANSFORMERS_VERSION = "5.14.1"
PYDANTIC_VERSION = "2.13.4"
PROCESS_LIFECYCLE_PHASES = (
    "RUNNING",
    "QUIESCE",
    "ENGINE_STOP",
    "WORKER_STOP",
    "CHILD_REAP",
    "PGID_EMPTY",
    "CUDA_RELEASED",
    "FUNCTION_RETURN",
)

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
    raise RuntimeError("refusing to hydrate integrated v11 GPU work without budget preflight")

_Attempt = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9-]{7,95}$")]
_Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class IntegratedV11RemoteAuthorization(_StrictModel):
    schema_version: Literal[
        "sloforge.branchfabric.experiment-004-v11-integrated-authorization/v1"
    ] = "sloforge.branchfabric.experiment-004-v11-integrated-authorization/v1"
    reservation_id: _Attempt
    reservation_commitment_sha256: _Sha256
    config_sha256: _Sha256
    preflight_token_sha256: _Sha256
    requested_gpu: Literal["A100-80GB"] = "A100-80GB"
    gpu_count: Literal[2] = 2
    maximum_wall_seconds: float = Field(gt=0.0, le=588.0, allow_inf_nan=False)
    maximum_gpu_seconds: float = Field(gt=0.0, le=1176.0, allow_inf_nan=False)
    budget_usd: float = Field(gt=0.0, allow_inf_nan=False)


def _canonical_bytes(value: BaseModel | dict[str, Any]) -> bytes:
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode()


def _ledger_file_bytes(value: BaseModel | dict[str, Any]) -> bytes:
    """Match the authoritative Experiment 004 ledger writer byte-for-byte."""

    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    return (json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()


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


def _positive_budget(raw: str | None) -> float:
    try:
        value = float(raw) if raw is not None else math.nan
    except (TypeError, ValueError) as error:
        raise RuntimeError("SLOFORGE_GPU_BUDGET_USD must be finite and positive") from error
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError("SLOFORGE_GPU_BUDGET_USD must be finite and positive")
    return value


def _require_modal_sdk_version() -> None:
    if modal.__version__ != MODAL_SDK_VERSION:
        raise RuntimeError(f"Modal SDK must be exactly {MODAL_SDK_VERSION}")


def _validate_config_artifacts(
    config: Experiment004V11IntegratedConfig,
    *,
    repository_root: Path,
) -> dict[str, Any]:
    bindings = (
        (config.offline_gate_manifest, config.offline_gate_manifest_sha256),
        (config.micro_validation_artifact, config.micro_validation_sha256),
        (config.post_micro_review_manifest, config.post_micro_review_manifest_sha256),
        (config.budget_authorization, config.budget_authorization_sha256),
    )
    for reference, expected_sha256 in bindings:
        validate_bound_artifact(
            repository_root,
            reference=reference,
            expected_sha256=expected_sha256,
        )
    budget_path = validate_bound_artifact(
        repository_root,
        reference=config.budget_authorization,
        expected_sha256=config.budget_authorization_sha256,
    )
    payload = json.loads(budget_path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError("integrated v11 budget authorization is not an object")
    authorized_usd = payload.get("authorized_gpu_budget_usd")
    authorized_seconds = payload.get("authorized_cumulative_gpu_seconds")
    if (
        isinstance(authorized_usd, bool)
        or not isinstance(authorized_usd, (int, float))
        or not math.isfinite(float(authorized_usd))
        or float(authorized_usd) <= 0.0
        or isinstance(authorized_seconds, bool)
        or not isinstance(authorized_seconds, (int, float))
        or float(authorized_seconds) < MAXIMUM_GPU_SECONDS
    ):
        raise RuntimeError("integrated v11 budget authorization is insufficient")
    return payload


def _validate_authorization(
    config: Experiment004V11IntegratedConfig,
    payload: dict[str, Any],
    *,
    authorized_budget_usd: float,
) -> IntegratedV11RemoteAuthorization:
    authorization = IntegratedV11RemoteAuthorization.model_validate(payload)
    if authorization.config_sha256 != hashlib.sha256(_canonical_bytes(config)).hexdigest():
        raise RuntimeError("integrated v11 authorization does not bind the immutable config")
    if authorization.preflight_token_sha256 != hashlib.sha256(_LAUNCH_TOKEN.encode()).hexdigest():
        raise RuntimeError("integrated v11 authorization does not bind the preflight token")
    if not math.isclose(
        authorization.maximum_wall_seconds,
        float(GPU_FUNCTION_TIMEOUT_SECONDS),
        rel_tol=0.0,
        abs_tol=1e-9,
    ) or not math.isclose(
        authorization.maximum_gpu_seconds,
        float(MAXIMUM_GPU_SECONDS),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeError("integrated v11 authorization must reserve 2xA100 for exactly 588s")
    if authorization.budget_usd > authorized_budget_usd:
        raise RuntimeError("integrated v11 launch budget exceeds the bound authorization")
    return authorization


def _validate_local_reservation(
    config: Experiment004V11IntegratedConfig,
    reservation_id: str,
) -> str:
    from sloforge.helix.characterization.gpu_reclamation_methodology import (
        Experiment004GpuHourLedger,
    )

    ledger_path = LOCAL_EXPERIMENT_ROOT / "gpu-hours.json"
    ledger = Experiment004GpuHourLedger.model_validate_json(
        ledger_path.resolve(strict=True).read_text(), strict=True
    )
    matches = [item for item in ledger.reservations if item.reservation_id == reservation_id]
    if len(matches) != 1 or len(ledger.reservations) != 1:
        raise RuntimeError("integrated v11 sole-coordinator reservation is absent or duplicated")
    reservation = matches[0]
    if (
        reservation.invocation_id != config.attempt_id
        or reservation.config_sha256 != hashlib.sha256(_canonical_bytes(config)).hexdigest()
        or reservation.gpu_count != GPU_COUNT
        or reservation.requested_gpu != "A100-80GB"
        or not math.isclose(
            reservation.maximum_wall_seconds,
            float(GPU_FUNCTION_TIMEOUT_SECONDS),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise RuntimeError("integrated v11 ledger reservation differs from the Modal request")
    pre_reservation = Experiment004GpuHourLedger(
        hard_additional_gpu_seconds=ledger.hard_additional_gpu_seconds,
        consumed_additional_gpu_seconds=ledger.consumed_additional_gpu_seconds,
        intervals=ledger.intervals,
        reservations=(),
        conservative_failure_charges=ledger.conservative_failure_charges,
    )
    reconstructed_sha256 = hashlib.sha256(_ledger_file_bytes(pre_reservation)).hexdigest()
    if reconstructed_sha256 != config.ledger_sha256_before_reservation:
        raise RuntimeError(
            "integrated v11 pre-reservation ledger commitment differs from the sealed config"
        )
    return hashlib.sha256(_canonical_bytes(reservation)).hexdigest()


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
        raise RuntimeError("cached model identity differs from integrated v11")
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


def _validate_inventory_rows(value: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(value, (tuple, list)) or len(value) != GPU_COUNT:
        raise RuntimeError("integrated v11 requires exactly two physical GPU inventory rows")
    rows: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            raise RuntimeError("integrated v11 GPU inventory row is invalid")
        required = {
            "index",
            "uuid",
            "name",
            "driver_version",
            "memory_total_mib",
            "memory_used_mib",
            "utilization_percent",
        }
        if set(item) != required:
            raise RuntimeError("integrated v11 GPU inventory fields are incomplete")
        row = dict(item)
        if (
            not str(row["uuid"]).startswith("GPU-")
            or "A100" not in str(row["name"])
            or int(row["memory_total_mib"]) < 79_000
        ):
            raise RuntimeError("integrated v11 inventory contains a non-A100-80GB device")
        rows.append(row)
    if len({str(row["uuid"]) for row in rows}) != GPU_COUNT:
        raise RuntimeError("integrated v11 inventory repeats a physical GPU UUID")
    if len({str(row["name"]) for row in rows}) != 1:
        raise RuntimeError("integrated v11 requires two identical A100-80GB models")
    return rows[0], rows[1]


def _hardware_comparability(controller: dict[str, Any]) -> dict[str, Any]:
    before = _validate_inventory_rows(controller.get("inventory_before"))
    after = _validate_inventory_rows(controller.get("inventory_after"))
    before_identity = tuple(
        (row["uuid"], row["name"], row["driver_version"], row["memory_total_mib"]) for row in before
    )
    after_identity = tuple(
        (row["uuid"], row["name"], row["driver_version"], row["memory_total_mib"]) for row in after
    )
    if before_identity != after_identity:
        raise RuntimeError("integrated v11 physical GPU identity changed during the transaction")
    names = [str(row["name"]) for row in before]
    direct = all(name == V10_DIRECT_COMPARISON_GPU_NAME for name in names)
    return {
        "modal_request": GPU_REQUEST,
        "modal_interconnect_selection_available": False,
        "observed_gpu_names": names,
        "observed_gpu_uuids": [str(row["uuid"]) for row in before],
        "frozen_v10_gpu_name": V10_DIRECT_COMPARISON_GPU_NAME,
        "direct_v10_timing_comparable": direct,
        "timing_comparability_reason": (
            "both physical devices exactly match frozen v10 A100-SXM4-80GB"
            if direct
            else "Modal cannot pin PCIe versus SXM4; exact observed hardware differs from v10"
        ),
        "byte_and_amplification_comparable": True,
    }


def _require_in_function_cleanup(controller: dict[str, Any], *, work_root: Path) -> dict[str, Any]:
    """Reject missing, contradictory, or incomplete controller cleanup evidence."""

    cleanup = controller.get("in_function_cleanup")
    if not isinstance(cleanup, dict):
        raise RuntimeError("integrated v11 controller omitted in-function cleanup evidence")
    artifact = work_root / "in_function_cleanup.json"
    if not artifact.is_file():
        raise RuntimeError("integrated v11 controller omitted in_function_cleanup.json")
    try:
        artifact_cleanup = json.loads(artifact.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("integrated v11 cleanup artifact is unreadable") from error
    if artifact_cleanup != cleanup:
        raise RuntimeError("integrated v11 cleanup artifact contradicts controller evidence")

    lifecycle = cleanup.get("lifecycle")
    phases = (
        [item.get("phase") for item in lifecycle]
        if isinstance(lifecycle, list) and all(isinstance(item, dict) for item in lifecycle)
        else None
    )
    phase_timestamps = (
        [item.get("observed_at_monotonic_ns") for item in lifecycle]
        if isinstance(lifecycle, list) and all(isinstance(item, dict) for item in lifecycle)
        else None
    )
    empty_evidence = (
        "surviving_children",
        "surviving_process_groups",
        "profiler_processes_after",
        "serving_workers_after",
        "rollout_workers_after",
        "resource_tracker_processes_after",
        "zombie_processes_after",
        "leaked_ipc_resources",
        "leaked_threads",
        "compute_processes_after",
        "cleanup_errors",
    )
    affirmative_evidence = (
        "parent_reaped_all_owned_children",
        "owned_ipc_resources_released",
        "pipes_closed",
        "cuda_released",
    )
    owned_children = cleanup.get("owned_children")
    termination_actions = cleanup.get("termination_actions")
    forced_kills = cleanup.get("forced_kills")
    worker_pids = controller.get("worker_pids")
    worker_process_groups = controller.get("worker_process_groups")
    worker_session_ids = controller.get("worker_session_ids")
    owned_pids = (
        {item.get("pid") for item in owned_children}
        if isinstance(owned_children, list)
        and all(isinstance(item, dict) for item in owned_children)
        else set()
    )
    child_rows_complete = bool(
        isinstance(owned_children, list)
        and all(
            {
                "pid",
                "ppid",
                "pgid",
                "sid",
                "role",
                "process_kind",
                "termination_signal",
                "exit_status",
                "reap_timestamp_monotonic_ns",
                "reap_method",
            }
            <= set(item)
            for item in owned_children
        )
    )
    actions_complete = bool(
        isinstance(termination_actions, list)
        and all(
            isinstance(item, dict)
            and item.get("signal") in {"SIGTERM", "SIGKILL"}
            and item.get("forced") is (item.get("signal") == "SIGKILL")
            for item in termination_actions
        )
        and isinstance(forced_kills, list)
        and forced_kills
        == [item for item in termination_actions if item.get("signal") == "SIGKILL"]
        and cleanup.get("forced_kill_required") is bool(forced_kills)
    )
    worker_ownership_complete = bool(
        isinstance(worker_pids, dict)
        and set(worker_pids) == {"serving", "rollout"}
        and all(
            isinstance(pid, int) and pid > 0 and pid in owned_pids for pid in worker_pids.values()
        )
        and isinstance(worker_process_groups, dict)
        and set(worker_process_groups) == set(worker_pids)
        and isinstance(worker_session_ids, dict)
        and set(worker_session_ids) == set(worker_pids)
        and all(
            any(
                item.get("pid") == worker_pids[role]
                and item.get("pgid") == worker_process_groups[role]
                and item.get("sid") == worker_session_ids[role]
                and item.get("role") == role
                and item.get("process_kind") == "worker"
                for item in owned_children
            )
            for role in worker_pids
        )
    )
    subreaper = cleanup.get("child_subreaper")
    subreaper_complete = bool(
        isinstance(subreaper, dict)
        and subreaper.get("error") is None
        and subreaper.get("restored_before_return") is True
        and (subreaper.get("supported") is False or subreaper.get("enabled_for_experiment") is True)
    )
    if (
        cleanup.get("schema_version") != "sloforge.branchfabric.in-function-cleanup/v1"
        or cleanup.get("status") != "PASS"
        or cleanup.get("pass") is not True
        or phases != list(PROCESS_LIFECYCLE_PHASES)
        or not isinstance(phase_timestamps, list)
        or not all(
            isinstance(value, int) and not isinstance(value, bool) for value in phase_timestamps
        )
        or phase_timestamps != sorted(phase_timestamps)
        or cleanup.get("required_lifecycle") != list(PROCESS_LIFECYCLE_PHASES)
        or not all(
            isinstance(cleanup.get(field), int)
            and not isinstance(cleanup.get(field), bool)
            and cleanup.get(field) > 0
            for field in ("parent_pid", "parent_pgid", "parent_sid")
        )
        or any(cleanup.get(field) != [] for field in empty_evidence)
        or any(cleanup.get(field) is not True for field in affirmative_evidence)
        or not child_rows_complete
        or not actions_complete
        or not worker_ownership_complete
        or not subreaper_complete
        or controller.get("compute_processes_after") != cleanup.get("compute_processes_after")
    ):
        raise RuntimeError("integrated v11 in-function cleanup evidence is not PASS-complete")
    return cleanup


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


def _publish_immutable_result(
    *,
    work_root: Path,
    staging: Path,
    final: Path,
    results_root: Path,
    volume: Any,
    attempt_id: str,
) -> str:
    if staging.exists() or final.exists():
        raise FileExistsError("immutable integrated v11 result prefix already exists")
    staging.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copytree(work_root, staging)
        _write_new(
            staging / "REMOTE_MANIFEST.json",
            {
                "schema_version": (
                    "sloforge.branchfabric.experiment-004-v11-integrated-remote-manifest/v1"
                ),
                "attempt_id": attempt_id,
                "remote_prefix": str(final.relative_to(results_root)),
                "artifacts": _inventory_tree(staging),
            },
        )
        os.replace(staging, final)
        volume.commit()
    finally:
        if staging.exists():
            shutil.rmtree(staging)
            volume.commit()
    return _sha256(final / "REMOTE_MANIFEST.json")


model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=False)
results_volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=False)
gpu_image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.12")
    .entrypoint([])
    .apt_install("gcc", "libnuma1", "libgl1", "libglib2.0-0", "procps", "pciutils")
    .uv_pip_install(
        f"torch=={TORCH_VERSION}",
        f"transformers=={TRANSFORMERS_VERSION}",
        f"vllm=={VLLM_VERSION}",
        f"pydantic=={PYDANTIC_VERSION}",
        "psutil==7.2.2",
        "nvidia-ml-py==13.610.43",
        "huggingface-hub==1.26.0",
        "safetensors==0.8.0",
        "httpx==0.28.1",
        "jinja2==3.1.6",
        "jsonschema==4.26.0",
        "pyyaml==6.0.3",
        "rich==13.9.4",
        "typer==0.27.0",
        uv_version=UV_VERSION,
    )
)
if _LOCAL_REPOSITORY_AVAILABLE and hasattr(gpu_image, "add_local_dir"):
    gpu_image = (
        gpu_image.add_local_dir(LOCAL_REPOSITORY_ROOT / "python", "/opt/sloforge/python", copy=True)
        .add_local_dir(
            LOCAL_REPOSITORY_ROOT / "benchmarks/branchfabric",
            "/opt/sloforge/benchmarks/branchfabric",
            copy=True,
        )
        .add_local_dir(
            LOCAL_REPOSITORY_ROOT / "experiments/branchfabric",
            "/opt/sloforge/experiments/branchfabric",
            copy=True,
        )
        .add_local_dir(LOCAL_REPOSITORY_ROOT / "tests", "/opt/sloforge/tests", copy=True)
    )
    bundled = "/opt/sloforge/artifacts/branchfabric/gpu-validation/experiment-004"
    gate_root = LOCAL_EXPERIMENT_ROOT / "v11/integration-gates"
    review_root = LOCAL_EXPERIMENT_ROOT / "v11/reviews"
    final_gate_root = LOCAL_EXPERIMENT_ROOT / "v11-final"
    if gate_root.is_dir():
        gpu_image = gpu_image.add_local_dir(
            gate_root, f"{bundled}/v11/integration-gates", copy=True
        )
    if review_root.is_dir():
        gpu_image = gpu_image.add_local_dir(review_root, f"{bundled}/v11/reviews", copy=True)
    if final_gate_root.is_dir():
        gpu_image = gpu_image.add_local_dir(
            final_gate_root, f"{bundled}/v11-final", copy=True
        )
    for source, destination in (
        (
            LOCAL_EXPERIMENT_ROOT
            / ("v11/runtime/exp004-v11-integrated-s41-b/postflight-cleanup-and-settlement.json"),
            (
                f"{bundled}/v11/runtime/exp004-v11-integrated-s41-b/"
                "postflight-cleanup-and-settlement.json"
            ),
        ),
        (
            LOCAL_EXPERIMENT_ROOT / "v11/micro-validation/status.json",
            f"{bundled}/v11/micro-validation/status.json",
        ),
        (
            LOCAL_EXPERIMENT_ROOT
            / "v11/micro-validation/exp004-v11-micro-s41-d/scientific-outcome.json",
            (f"{bundled}/v11/micro-validation/exp004-v11-micro-s41-d/scientific-outcome.json"),
        ),
        (
            LOCAL_EXPERIMENT_ROOT / "budget-authorization-v11.json",
            f"{bundled}/budget-authorization-v11.json",
        ),
    ):
        if source.is_file():
            gpu_image = gpu_image.add_local_file(source, destination, copy=True)
gpu_image = gpu_image.env(
    {
        "PYTHONPATH": (
            "/opt/sloforge:/opt/sloforge/python:/opt/sloforge/benchmarks/branchfabric:"
            "/opt/sloforge/experiments/branchfabric"
        ),
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "DO_NOT_TRACK": "1",
        "WANDB_MODE": "disabled",
        "WANDB_DISABLED": "true",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "CC": "/usr/bin/gcc",
    }
)

app = modal.App(
    APP_NAME,
    tags={"project": "sloforge", "experiment": "branchfabric-gpu-validation-004-v11"},
)
_run_integrated_function: Any = None
_launch_token_secret: Any = None


def run_integrated(
    config_payload: dict[str, Any], authorization_payload: dict[str, Any]
) -> dict[str, Any]:
    entry_ns = time.monotonic_ns()
    config = Experiment004V11IntegratedConfig.model_validate(config_payload)
    bound_budget = _validate_config_artifacts(config, repository_root=Path("/opt/sloforge"))
    authorization = _validate_authorization(
        config,
        authorization_payload,
        authorized_budget_usd=float(bound_budget["authorized_gpu_budget_usd"]),
    )
    _require_modal_sdk_version()
    function_call_id = modal.current_function_call_id()
    if not function_call_id:
        raise RuntimeError("Modal did not expose the integrated v11 FunctionCall ID")
    snapshot = MODEL_MOUNT / "snapshots" / MODEL_REVISION
    work_root = Path("/tmp/sloforge-branchfabric-exp004-v11-integrated") / config.attempt_id
    config_path = work_root.parent / f"{config.attempt_id}.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    _write_new(config_path, config)
    controller_deadline_ns = entry_ns + round(
        (GPU_FUNCTION_TIMEOUT_SECONDS - POST_CONTROLLER_RESERVE_SECONDS) * 1e9
    )
    controller: dict[str, Any] = {
        "schema_version": (
            "sloforge.branchfabric.experiment-004-v11-integrated-controller-failure/v1"
        ),
        "status": "failed",
    }
    run_error: dict[str, Any] | None = None
    hardware: dict[str, Any] = {
        "modal_request": GPU_REQUEST,
        "modal_interconnect_selection_available": False,
        "observed_gpu_names": [],
        "observed_gpu_uuids": [],
        "frozen_v10_gpu_name": V10_DIRECT_COMPARISON_GPU_NAME,
        "direct_v10_timing_comparable": False,
        "timing_comparability_reason": "controller did not produce validated inventory",
        "byte_and_amplification_comparable": False,
    }
    try:
        _validate_model_snapshot(snapshot)
        from gpu_reclamation_integrated_controller_v11 import run_integrated_v11_controller

        controller = run_integrated_v11_controller(
            config_path=config_path,
            work_root=work_root,
            worker_path=Path(
                "/opt/sloforge/experiments/branchfabric/gpu_reclamation_integrated_worker_v11.py"
            ),
            model_snapshot=snapshot,
            repository_root=Path("/opt/sloforge"),
            absolute_deadline_ns=controller_deadline_ns,
        )
        _require_in_function_cleanup(controller, work_root=work_root)
        if controller.get("status") != "succeeded":
            raise RuntimeError("integrated v11 controller failed closed")
        hardware = _hardware_comparability(controller)
    except BaseException as error:
        run_error = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        work_root.mkdir(parents=True, exist_ok=True)
        _write_new(work_root / "function-failure.json", run_error)
    finally:
        work_root.mkdir(parents=True, exist_ok=True)
        in_function_cleanup = controller.get("in_function_cleanup")
        cleanup_present = isinstance(in_function_cleanup, dict)
        cleanup_pass = bool(cleanup_present and in_function_cleanup.get("pass") is True)
        _write_new(
            work_root / "function-finally-cleanup.json",
            {
                "schema_version": (
                    "sloforge.branchfabric.experiment-004-v11-integrated-function-cleanup/v1"
                ),
                "attempt_id": config.attempt_id,
                "controller_status": controller.get("status"),
                "controller_cleanup_evidence_present": cleanup_present,
                "in_function_cleanup_pass": cleanup_pass,
                "in_function_cleanup_artifact": "in_function_cleanup.json",
                "owned_child_processes_after": (
                    in_function_cleanup.get("surviving_children") if cleanup_present else None
                ),
                "profiler_processes_after": (
                    in_function_cleanup.get("profiler_processes_after") if cleanup_present else None
                ),
                "forced_kills": (
                    in_function_cleanup.get("forced_kills") if cleanup_present else None
                ),
                "provider_cleanup_status": "pending-function-return-and-local-postflight",
                "recorded_at_utc": datetime.now(UTC).isoformat(),
            },
        )

    controller_and_analysis_seconds = (time.monotonic_ns() - entry_ns) / 1e9
    completion = {
        "schema_version": ("sloforge.branchfabric.experiment-004-v11-integrated-completion/v1"),
        "status": "provisional" if run_error is None else "failed",
        "scientific_status": (
            "pending-local-budget-settlement-and-provider-cleanup"
            if run_error is None
            else "invalid"
        ),
        "attempt_id": config.attempt_id,
        "reservation_id": authorization.reservation_id,
        "reservation_commitment_sha256": authorization.reservation_commitment_sha256,
        "config_sha256": authorization.config_sha256,
        "function_call_id": function_call_id,
        "requested_gpu": GPU_REQUEST,
        "gpu_count": GPU_COUNT,
        "controller_and_analysis_interval_seconds": controller_and_analysis_seconds,
        "gpu_allocation_seconds_status": "pending-final-function-return",
        "hardware_comparability": hardware,
        "bound_artifacts": {
            "offline_gate_manifest": config.offline_gate_manifest,
            "offline_gate_manifest_sha256": config.offline_gate_manifest_sha256,
            "micro_validation_artifact": config.micro_validation_artifact,
            "micro_validation_sha256": config.micro_validation_sha256,
            "post_micro_review_manifest": config.post_micro_review_manifest,
            "post_micro_review_manifest_sha256": config.post_micro_review_manifest_sha256,
            "budget_authorization": config.budget_authorization,
            "budget_authorization_sha256": config.budget_authorization_sha256,
        },
        "absolute_deadlines": {
            "function_entry_monotonic_ns": entry_ns,
            "controller_deadline_monotonic_ns": controller_deadline_ns,
            "function_deadline_monotonic_ns": entry_ns
            + GPU_FUNCTION_TIMEOUT_SECONDS * 1_000_000_000,
            "post_controller_reserve_seconds": POST_CONTROLLER_RESERVE_SECONDS,
        },
        "controller": controller,
        "in_function_cleanup": controller.get("in_function_cleanup"),
        "run_error": run_error,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    _write_new(work_root / "function-completion.json", completion)
    staging = RESULTS_MOUNT / f"{REMOTE_EXPERIMENT_PREFIX}/{config.attempt_id}.staging"
    final = RESULTS_MOUNT / f"{REMOTE_EXPERIMENT_PREFIX}/{config.attempt_id}"
    manifest_sha256 = _publish_immutable_result(
        work_root=work_root,
        staging=staging,
        final=final,
        results_root=RESULTS_MOUNT,
        volume=results_volume,
        attempt_id=config.attempt_id,
    )
    shutil.rmtree(work_root)
    final_elapsed_seconds = (time.monotonic_ns() - entry_ns) / 1e9
    return {
        **completion,
        "gpu_allocation_seconds": final_elapsed_seconds,
        "gpu_seconds": GPU_COUNT * final_elapsed_seconds,
        "gpu_hours": GPU_COUNT * final_elapsed_seconds / 3600.0,
        "remote_prefix": str(final.relative_to(RESULTS_MOUNT)),
        "remote_manifest_sha256": manifest_sha256,
    }


def _materialize(attempt_id: str) -> dict[str, str]:
    return {
        "remote_path": f"{REMOTE_EXPERIMENT_PREFIX}/{attempt_id}",
        "volume_name": RESULTS_VOLUME_NAME,
    }


def main(*, config_path: str) -> None:
    _require_modal_sdk_version()
    config = Experiment004V11IntegratedConfig.model_validate_json(Path(config_path).read_text())
    budget_artifact = _validate_config_artifacts(config, repository_root=LOCAL_REPOSITORY_ROOT)
    budget = _positive_budget(os.getenv("SLOFORGE_GPU_BUDGET_USD"))
    if budget > float(budget_artifact["authorized_gpu_budget_usd"]):
        raise RuntimeError("integrated v11 launch budget exceeds the bound authorization")
    reservation_id = os.getenv("SLOFORGE_EXP004_RESERVATION_ID")
    if reservation_id is None or not _CLOUD_GRAPH_ENABLED:
        raise RuntimeError("integrated v11 requires reservation and preflight-token evidence")
    reservation_commitment = _validate_local_reservation(config, reservation_id)
    authorization = IntegratedV11RemoteAuthorization(
        reservation_id=reservation_id,
        reservation_commitment_sha256=reservation_commitment,
        config_sha256=hashlib.sha256(_canonical_bytes(config)).hexdigest(),
        preflight_token_sha256=hashlib.sha256(_LAUNCH_TOKEN.encode()).hexdigest(),
        maximum_wall_seconds=float(GPU_FUNCTION_TIMEOUT_SECONDS),
        maximum_gpu_seconds=float(MAXIMUM_GPU_SECONDS),
        budget_usd=budget,
    )
    call = _run_integrated_function.spawn(
        config.model_dump(mode="json"), authorization.model_dump(mode="json")
    )
    result = call.get(timeout=GPU_FUNCTION_TIMEOUT_SECONDS + GPU_FUNCTION_STARTUP_TIMEOUT_SECONDS)
    print(
        _canonical_bytes(
            {"result": result, "materialized": _materialize(config.attempt_id)}
        ).decode(),
        end="",
    )


if _CLOUD_GRAPH_ENABLED:
    _launch_token_secret = modal.Secret.from_dict({"SLOFORGE_MODAL_PREFLIGHT_TOKEN": _LAUNCH_TOKEN})
    _run_integrated_function = app.function(
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
    )(run_integrated)
    _local_entrypoint = app.local_entrypoint()(main)


__all__ = [
    "APP_NAME",
    "GPU_REQUEST",
    "IntegratedV11RemoteAuthorization",
    "app",
    "main",
    "run_integrated",
]
