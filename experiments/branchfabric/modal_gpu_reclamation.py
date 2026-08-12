"""Modal entrypoint for BranchFabric GPU Validation Experiment 004.

One ephemeral Function requests exactly two A100-80GB devices on one host.
The Function itself stays CUDA-clean and launches one fresh interpreter per
physical GPU through :mod:`gpu_reclamation_controller`.
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
from typing import Annotated, Any, Literal, Self

import modal
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

MODAL_SDK_VERSION = "1.5.3"
APP_NAME = "sloforge-branchfabric-gpu-reclamation-004"
GPU_REQUEST = "A100-80GB:2"
GPU_COUNT = 2
GPU_FUNCTION_TIMEOUT_SECONDS = 340
GPU_FUNCTION_STARTUP_TIMEOUT_SECONDS = 180
GPU_COORDINATOR_RESERVATION_WALL_SECONDS = 340
MODEL_VOLUME_NAME = "sloforge-model-cache"
RESULTS_VOLUME_NAME = "sloforge-branchfabric-results"
REMOTE_EXPERIMENT_PREFIX = "experiment-004/modal"
MODEL_REVISION: Literal["a09a35458c702b33eeacc393d103063234e8bc28"] = (
    "a09a35458c702b33eeacc393d103063234e8bc28"
)
MODEL_ID: Literal["Qwen/Qwen2.5-7B-Instruct"] = "Qwen/Qwen2.5-7B-Instruct"
VLLM_VERSION: Literal["0.23.0"] = "0.23.0"
TORCH_VERSION = "2.11.0"
_ATTEMPT = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9-]{7,95}$")]
_SHA256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]

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
    raise RuntimeError("refusing to hydrate Experiment 004 without budget preflight")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class Experiment004PilotConfig(_StrictModel):
    schema_version: Literal["sloforge.branchfabric.experiment-004-modal-config/v1"] = (
        "sloforge.branchfabric.experiment-004-modal-config/v1"
    )
    attempt_id: _ATTEMPT
    seed: int = Field(ge=0, lt=1 << 63)
    model: Literal["Qwen/Qwen2.5-7B-Instruct"] = MODEL_ID
    model_revision: Literal["a09a35458c702b33eeacc393d103063234e8bc28"] = MODEL_REVISION
    tokenizer_revision: Literal["a09a35458c702b33eeacc393d103063234e8bc28"] = MODEL_REVISION
    runtime: Literal["vllm"] = "vllm"
    runtime_version: Literal["0.23.0"] = VLLM_VERSION
    mode: Literal["PRESERVE_NAIVE"] = "PRESERVE_NAIVE"
    prefix_length: Literal[16384] = 16384
    fanout: Literal[8] = 8
    suffix_length: Literal[256] = 256
    tracing_level: Literal["disabled", "minimal", "full"] = "minimal"
    serving_methodology: Literal["v9-independent-shards", "v10-global-capacity"] = (
        "v9-independent-shards"
    )
    gpu_memory_utilization: float = Field(default=0.72, gt=0.0, le=0.85)
    baseline_seconds: float = Field(default=4.0, ge=2.0, le=15.0)
    serving_spike_seconds: float = Field(default=12.0, ge=5.0, le=30.0)
    temporary_serving_seconds: float = Field(default=3.0, ge=1.0, le=10.0)
    serving_prompt_tokens: int = Field(default=256, ge=16, le=2048)
    serving_output_tokens: int = Field(default=64, ge=1, le=128)
    gpu0_control_request_rate_per_second: float = Field(default=8.0, ge=1.0, le=64.0)
    gpu0_spike_request_rate_per_second: float = Field(default=64.0, ge=4.0, le=128.0)
    gpu1_spike_request_rate_per_second: float = Field(default=32.0, ge=1.0, le=128.0)
    serving_spike_request_rate_per_second: float | None = Field(default=None, ge=1.0, le=128.0)
    gpu0_restore_request_rate_per_second: float = Field(default=8.0, ge=1.0, le=64.0)
    serving_slo_maximum_ttft_seconds: float = Field(default=2.0, gt=0.0, le=10.0)
    serving_slo_maximum_inter_token_latency_seconds: float = Field(default=0.5, gt=0.0, le=5.0)
    serving_slo_stability_window_seconds: float = Field(default=1.0, ge=0.5, le=5.0)
    gpu0_overload_probe_seconds: float = Field(default=3.0, ge=1.0, le=10.0)
    serving_recovery_evaluation_seconds: float = Field(default=1.0, ge=0.25, le=2.0)
    serving_recovery_queue_threshold: int = Field(default=4, ge=0, le=128)
    serving_maximum_pending_requests: int = Field(default=1024, ge=1, le=20_000)
    serving_restore_handoff_lead_requests: int = Field(default=4, ge=2, le=64)
    calibration_attempt_id: _ATTEMPT | None = None
    lambda_1_rps: float | None = Field(default=None, gt=0.0, le=128.0)
    lambda_2_rps: float | None = Field(default=None, gt=0.0, le=256.0)
    lambda_spike_rps: float | None = Field(default=None, gt=0.0, le=128.0)
    calibration_selected_load_artifact: str | None = Field(default=None, min_length=1)
    calibration_selected_load_sha256: _SHA256 | None = None
    agent9_approval_artifact: str | None = Field(default=None, min_length=1)
    agent9_approval_sha256: _SHA256 | None = None
    maximum_wall_seconds: int = Field(default=480, ge=60, le=900)
    initialization_timeout_seconds: int = Field(default=540, ge=60, le=900)
    cleanup_timeout_seconds: int = Field(default=60, ge=10, le=120)

    @model_validator(mode="after")
    def bounded_function_lifetime(self) -> Self:
        if self.serving_methodology == "v9-independent-shards":
            if self.serving_slo_stability_window_seconds > self.temporary_serving_seconds:
                raise ValueError(
                    "serving SLO stability window exceeds the measured serving interval"
                )
            if (
                self.gpu0_spike_request_rate_per_second + self.gpu1_spike_request_rate_per_second
                < 4.0 * self.gpu0_control_request_rate_per_second
            ):
                raise ValueError("serving spike must offer at least 4x the control request rate")
        else:
            required_calibration = (
                self.calibration_attempt_id,
                self.lambda_1_rps,
                self.lambda_2_rps,
                self.lambda_spike_rps,
                self.calibration_selected_load_artifact,
                self.calibration_selected_load_sha256,
                self.agent9_approval_artifact,
                self.agent9_approval_sha256,
            )
            if self.serving_spike_request_rate_per_second is None or any(
                item is None for item in required_calibration
            ):
                raise ValueError("v10 requires its immutable calibrated load and Agent 9 approval")
            assert self.lambda_1_rps is not None
            assert self.lambda_2_rps is not None
            assert self.lambda_spike_rps is not None
            if not math.isclose(
                self.serving_spike_request_rate_per_second,
                self.lambda_spike_rps,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("v10 serving spike differs from the approved calibrated spike")
            spike_over_one = self.lambda_spike_rps / self.lambda_1_rps
            spike_over_two = self.lambda_spike_rps / self.lambda_2_rps
            if not 1.15 <= spike_over_one <= 1.30 or not 0.80 <= spike_over_two <= 0.90:
                raise ValueError("v10 calibrated load does not have the required capacity margins")
            if self.gpu0_control_request_rate_per_second > self.lambda_1_rps:
                raise ValueError("v10 control load exceeds measured one-GPU stable capacity")
            if self.gpu0_restore_request_rate_per_second >= self.lambda_1_rps:
                raise ValueError("v10 restore load must remain below one-GPU stable capacity")
            if self.serving_output_tokens != 64:
                raise ValueError("v10 requires exactly 64 serving output tokens")
            if self.serving_prompt_tokens != 256:
                raise ValueError("v10 requires the calibrated 256-token serving prompt")
            if self.serving_slo_stability_window_seconds < 5.0:
                raise ValueError("v10 requires at least five seconds of serving stability")
            windows = (
                self.serving_slo_stability_window_seconds / self.serving_recovery_evaluation_seconds
            )
            if not math.isclose(windows, round(windows), rel_tol=0.0, abs_tol=1e-9):
                raise ValueError("v10 stability must contain whole evaluation windows")
            if (
                self.serving_spike_request_rate_per_second
                <= self.gpu0_control_request_rate_per_second
            ):
                raise ValueError("v10 spike must exceed the measured GPU0 control rate")
        if (
            self.maximum_wall_seconds
            + self.initialization_timeout_seconds
            + self.cleanup_timeout_seconds
            + 40
            > GPU_FUNCTION_TIMEOUT_SECONDS
        ):
            raise ValueError("pilot inner bounds exceed the Modal Function timeout")
        return self


class RemoteAuthorization(_StrictModel):
    schema_version: Literal["sloforge.branchfabric.experiment-004-authorization/v1"] = (
        "sloforge.branchfabric.experiment-004-authorization/v1"
    )
    reservation_id: _ATTEMPT
    config_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    maximum_gpu_seconds: int = Field(gt=0, le=GPU_COORDINATOR_RESERVATION_WALL_SECONDS * GPU_COUNT)
    budget_usd: float = Field(gt=0.0, allow_inf_nan=False)


def _canonical_bytes(value: Any) -> bytes:
    if isinstance(value, Experiment004PilotConfig):
        # Preserve the exact v1/v9 config digest when newer methodology-only
        # fields remain unset. Explicit v10 fields are still serialized.
        value = value.model_dump(mode="json", exclude_unset=True)
    elif isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")


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
    config: Experiment004PilotConfig, payload: dict[str, Any]
) -> RemoteAuthorization:
    authorization = RemoteAuthorization.model_validate(payload)
    digest = hashlib.sha256(_canonical_bytes(config)).hexdigest()
    if authorization.config_sha256 != digest:
        raise RuntimeError("Modal authorization does not match the immutable pilot config")
    if authorization.maximum_gpu_seconds != GPU_COORDINATOR_RESERVATION_WALL_SECONDS * GPU_COUNT:
        raise RuntimeError("Modal authorization does not reserve the exact bounded GPU interval")
    return authorization


def _validate_local_reservation(config: Experiment004PilotConfig, reservation_id: str) -> None:
    """Require the sole coordinator's immutable ledger reservation before spawn."""

    from sloforge.helix.characterization.gpu_reclamation_methodology import (
        Experiment004GpuHourLedger,
    )

    ledger_path = LOCAL_EXPERIMENT_ROOT / "gpu-hours.json"
    if not ledger_path.is_file():
        raise RuntimeError("Experiment 004 GPU-hour ledger is absent before Modal spawn")
    ledger = Experiment004GpuHourLedger.model_validate_json(ledger_path.read_text(), strict=True)
    matches = [item for item in ledger.reservations if item.reservation_id == reservation_id]
    if len(matches) != 1:
        raise RuntimeError("sole-coordinator GPU reservation is absent or duplicated")
    reservation = matches[0]
    expected_digest = hashlib.sha256(_canonical_bytes(config)).hexdigest()
    if (
        reservation.invocation_id != config.attempt_id
        or reservation.config_sha256 != expected_digest
        or reservation.gpu_count != GPU_COUNT
        or reservation.requested_gpu != "A100-80GB"
        or reservation.maximum_wall_seconds != GPU_COORDINATOR_RESERVATION_WALL_SECONDS
    ):
        raise RuntimeError("sole-coordinator GPU reservation differs from the Modal pilot")


def _validate_model_snapshot(snapshot: Path) -> dict[str, Any]:
    manifest_path = snapshot / "MODEL_MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("pinned Qwen snapshot is absent from sloforge-model-cache")
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError("model manifest must be a JSON object")
    expected = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("cached model identity differs from Experiment 003")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("cached model manifest has no immutable file inventory")
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("cached model inventory has an invalid row")
        path = snapshot / str(item["relative_path"])
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"cached model file is missing or symlinked: {path}")
        if path.stat().st_size != int(item["bytes"]) or _sha256(path) != item["sha256"]:
            raise ValueError(f"cached model hash mismatch: {path}")
    inputs = snapshot / "BRANCHFABRIC_INPUTS.json"
    if not inputs.is_file():
        raise FileNotFoundError("Experiment 003 tokenizer inputs are absent from model cache")
    return manifest


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

# Reuse the exact Experiment 003 dependency image; all installs occur during
# image construction, never while the A100s are allocated.
gpu_image = importlib.import_module("modal_real_gpu_cow").gpu_image

app = modal.App(
    APP_NAME,
    tags={"project": "sloforge", "experiment": "branchfabric-gpu-validation-004"},
)
_run_pilot_function: Any = None


def run_pilot(
    config_payload: dict[str, Any], authorization_payload: dict[str, Any]
) -> dict[str, Any]:
    config = Experiment004PilotConfig.model_validate(config_payload)
    authorization = _validate_authorization(config, authorization_payload)
    if modal.__version__ != MODAL_SDK_VERSION:
        raise RuntimeError(f"Modal SDK must be {MODAL_SDK_VERSION}, observed {modal.__version__}")
    function_call_id = modal.current_function_call_id()
    if not function_call_id:
        raise RuntimeError("Modal did not expose the FunctionCall ID")
    started_ns = time.monotonic_ns()
    work_root = Path("/tmp/sloforge-branchfabric-exp004") / config.attempt_id
    snapshot = MODEL_MOUNT / "snapshots" / MODEL_REVISION
    config_path = work_root.parent / f"{config.attempt_id}.json"
    _write_new(config_path, config)
    topology: tuple[Any, ...] = ()
    controller: dict[str, Any]
    run_error: dict[str, str] | None = None
    pilot_assessment: Any = None
    scientific_validity: Any = None
    movement_accounting: dict[str, Any] | None = None
    serving_recovery: dict[str, Any] | None = None
    try:
        _validate_model_snapshot(snapshot)
        from sloforge.helix.characterization.gpu_reclamation_instrumentation import (
            capture_topology_commands,
            validate_topology_capture,
        )

        topology = capture_topology_commands()
        validate_topology_capture(topology, expected_gpu_count=2)
        from gpu_reclamation_controller import run_two_gpu_controller

        controller = run_two_gpu_controller(
            config_path=config_path,
            work_root=work_root,
            worker_path=Path("/opt/sloforge/experiments/branchfabric/gpu_reclamation_worker.py"),
            model_snapshot=snapshot,
            readiness_timeout_seconds=float(config.initialization_timeout_seconds),
            transaction_timeout_seconds=float(config.maximum_wall_seconds),
        )
        if config.serving_methodology == "v10-global-capacity":
            from sloforge.helix.characterization.gpu_reclamation_v10_postprocess import (
                assess_v10_directory,
            )
            from sloforge.helix.characterization.gpu_reclamation_v10_validity import (
                BudgetEvidence,
                CleanupEvidence,
            )

            scientific_validity, movement_accounting, serving_recovery = assess_v10_directory(
                work_root=work_root,
                config=config.model_dump(mode="json"),
                controller=controller,
                # The remote function cannot observe ledger settlement or its
                # own provider teardown. Those two gates remain explicitly
                # false until the sole local coordinator supplies post-exit
                # evidence and regenerates the canonical validity artifact.
                budget=BudgetEvidence(
                    conservative_gpu_seconds_before=0.0,
                    invocation_gpu_seconds=0.0,
                    conservative_gpu_seconds_after=0.0,
                    hard_ceiling_gpu_seconds=float(authorization.maximum_gpu_seconds),
                    ledger_updated=False,
                ),
                cleanup=CleanupEvidence(
                    zero_active_tasks=False,
                    zero_running_containers=False,
                    zero_endpoints=False,
                    zero_reservations=False,
                    zero_owned_child_processes=False,
                    zero_profilers=False,
                    authorized_persistent_volumes_only=False,
                ),
            )
        else:
            from sloforge.helix.characterization.gpu_reclamation_pilot import (
                assess_pilot_directory,
            )

            pilot_assessment = assess_pilot_directory(
                work_root=work_root,
                config=config.model_dump(mode="json"),
                controller=controller,
            )
    except BaseException as caught:
        # A failed controller preflight is still a charged GPU allocation and
        # must leave an immutable diagnostic bundle.  This catch never turns
        # the failure into valid scientific evidence.
        work_root.mkdir(parents=True, exist_ok=True)
        run_error = {
            "type": type(caught).__name__,
            "message": str(caught),
            "traceback": traceback.format_exc(),
        }
        controller = {
            "schema_version": "sloforge.branchfabric.experiment-004-controller-failure/v1",
            "status": "failed",
            "transaction_error": run_error,
        }
        failure_reason = (
            f"GPU pilot/controller failed before scientific validation: "
            f"{run_error['type']}: {run_error['message']}"
        )
        if config.serving_methodology == "v10-global-capacity":
            from sloforge.helix.characterization.gpu_reclamation_v10_validity import (
                fail_closed_v10_scientific_validity,
            )

            scientific_validity = fail_closed_v10_scientific_validity(failure_reason)
            movement_accounting = {
                "schema_version": (
                    "sloforge.branchfabric.experiment-004-v10-movement-accounting/v1"
                ),
                "status": "invalid",
                "invalid_reason": failure_reason,
            }
            serving_recovery = {
                "schema_version": (
                    "sloforge.branchfabric.experiment-004-v10-serving-recovery/v1"
                ),
                "status": "invalid",
                "invalid_reason": failure_reason,
            }
        else:
            from sloforge.helix.characterization.gpu_reclamation_pilot import PilotAssessment

            pilot_assessment = PilotAssessment(
                pilot_valid=False,
                invalid_reasons=(failure_reason,),
            )
        _write_new(work_root / "function-failure.json", run_error)
    if config.serving_methodology == "v10-global-capacity":
        if scientific_validity is None or movement_accounting is None or serving_recovery is None:
            raise RuntimeError("v10 postprocessor did not produce all required analysis artifacts")
        _write_new(
            work_root / "analysis/scientific-validity.remote-provisional.json",
            scientific_validity,
        )
        _write_new(work_root / "analysis/movement-accounting.json", movement_accounting)
        _write_new(work_root / "analysis/serving-recovery.json", serving_recovery)
        pilot_valid = bool(scientific_validity.scientifically_valid)
        invalid_reasons = tuple(scientific_validity.invalid_reasons)
    else:
        if pilot_assessment is None:
            raise RuntimeError("legacy pilot postprocessor did not produce an assessment")
        _write_new(work_root / "analysis/pilot-assessment.json", pilot_assessment)
        pilot_valid = bool(pilot_assessment.pilot_valid)
        invalid_reasons = tuple(pilot_assessment.invalid_reasons)
    _write_new(
        work_root / "topology/topology-commands.json",
        [item.model_dump(mode="json") for item in topology],
    )
    elapsed_ns = time.monotonic_ns() - started_ns
    v10_provisional = (
        config.serving_methodology == "v10-global-capacity"
        and controller.get("status") == "succeeded"
        and scientific_validity is not None
        and all(
            bool(getattr(scientific_validity, field))
            for field in (
                "state_correctness_pass",
                "gpu0_overload_pass",
                "two_gpu_excess_capacity_pass",
                "queue_drain_pass",
                "slo_restoration_pass",
                "pre_restore_stability_pass",
                "gpu0_active_during_restore_pass",
                "branch_resume_pass",
                "movement_accounting_pass",
            )
        )
    )
    completion_succeeded = controller.get("status") == "succeeded" and pilot_valid
    completion = {
        "schema_version": "sloforge.branchfabric.experiment-004-function-completion/v1",
        "status": (
            "provisional"
            if v10_provisional
            else ("succeeded" if completion_succeeded else "failed")
        ),
        "attempt_id": config.attempt_id,
        "reservation_id": authorization.reservation_id,
        "config_sha256": authorization.config_sha256,
        "function_call_id": function_call_id,
        "requested_gpu": GPU_REQUEST,
        "gpu_count": GPU_COUNT,
        "gpu_allocation_seconds": elapsed_ns / 1e9,
        "gpu_seconds": GPU_COUNT * elapsed_ns / 1e9,
        "gpu_hours": GPU_COUNT * elapsed_ns / 3.6e12,
        "pilot_valid": pilot_valid,
        "scientific_status": (
            "pending-local-finalization"
            if v10_provisional
            else ("valid" if pilot_valid else "invalid")
        ),
        "measurable_gates_pass": v10_provisional,
        "pilot_invalid_reasons": invalid_reasons,
        "run_error": run_error,
        "controller": controller,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    _write_new(work_root / "function-completion.json", completion)
    staging = RESULTS_MOUNT / f"{REMOTE_EXPERIMENT_PREFIX}/{config.attempt_id}.staging"
    final = RESULTS_MOUNT / f"{REMOTE_EXPERIMENT_PREFIX}/{config.attempt_id}"
    if staging.exists() or final.exists():
        raise FileExistsError("immutable Experiment 004 result prefix already exists")
    staging.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copytree(work_root, staging)
        manifest = {
            "schema_version": "sloforge.branchfabric.experiment-004-remote-manifest/v1",
            "attempt_id": config.attempt_id,
            "remote_prefix": str(final.relative_to(RESULTS_MOUNT)),
            "artifacts": _inventory_tree(staging),
        }
        _write_new(staging / "REMOTE_MANIFEST.json", manifest)
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


def _materialize(attempt_id: str) -> dict[str, Any]:
    # Modal 1.5 forbids Volume.reload()/filesystem access from a local
    # entrypoint. Materialization is performed by the sole coordinator with
    # `modal volume get` after this app exits.
    return {
        "remote_path": f"{REMOTE_EXPERIMENT_PREFIX}/{attempt_id}",
        "volume_name": "sloforge-branchfabric-results",
    }


def main(*, config_path: str) -> None:
    config = Experiment004PilotConfig.model_validate(json.loads(Path(config_path).read_text()))
    budget_raw = os.environ.get("SLOFORGE_GPU_BUDGET_USD")
    if budget_raw is None:
        raise RuntimeError("SLOFORGE_GPU_BUDGET_USD is required before Modal invocation")
    budget = float(budget_raw)
    if not budget > 0:
        raise RuntimeError("SLOFORGE_GPU_BUDGET_USD must be positive")
    reservation_id = os.environ.get("SLOFORGE_EXP004_RESERVATION_ID")
    if reservation_id is None:
        raise RuntimeError("Experiment 004 GPU-hour reservation ID is required")
    _validate_local_reservation(config, reservation_id)
    authorization = RemoteAuthorization(
        reservation_id=reservation_id,
        config_sha256=hashlib.sha256(_canonical_bytes(config)).hexdigest(),
        maximum_gpu_seconds=GPU_COORDINATOR_RESERVATION_WALL_SECONDS * GPU_COUNT,
        budget_usd=budget,
    )
    # Preserve the caller's backwards-compatible explicit field set across the
    # local/remote Pydantic boundary so the authorization digest is identical.
    call = _run_pilot_function.spawn(
        json.loads(_canonical_bytes(config)), authorization.model_dump(mode="json")
    )
    result = call.get(timeout=GPU_FUNCTION_TIMEOUT_SECONDS + GPU_FUNCTION_STARTUP_TIMEOUT_SECONDS)
    materialized = _materialize(config.attempt_id)
    print(_canonical_bytes({"result": result, "materialized": materialized}).decode(), end="")


if _CLOUD_GRAPH_ENABLED:
    _run_pilot_function = app.function(
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
    )(run_pilot)
    _local_entrypoint = app.local_entrypoint()(main)


__all__ = [
    "APP_NAME",
    "GPU_REQUEST",
    "Experiment004PilotConfig",
    "app",
    "main",
    "run_pilot",
]
