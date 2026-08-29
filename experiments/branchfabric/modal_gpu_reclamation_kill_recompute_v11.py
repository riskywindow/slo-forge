"""Single-use Modal surface for Experiment 004 KILL_AND_RECOMPUTE.

Only the sole CPU coordinator may hydrate this graph.  The function requests
exactly two A100-80GB devices for at most 660 seconds, invokes the dedicated
kill controller/worker, publishes one immutable attempt prefix, and returns
provisional evidence for local settlement and provider cleanup.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

import modal
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

_BASE_MODAL_SHA256 = "638555b54abb6804ced704f06fc6016c176c4f5f44abe7e32544e3b6dbc32cde"
_METHODOLOGY_SHA256 = "23b8c3b2b6521c0dea1982c903f1a32c62a3e96d72712b7d35b08c82b20d1368"


def _resolve_bootstrap_paths(
    source: Path,
    *,
    remote_repository_root: Path = Path("/opt/sloforge"),
) -> tuple[Path, Path, Path]:
    """Resolve local checkout or shallow Modal-mounted import paths without indexing blindly."""

    resolved_source = source.resolve()
    candidate = resolved_source.parents[2] if len(resolved_source.parents) > 2 else None
    local_repository_available = bool(
        candidate is not None
        and not (candidate / "pyproject.toml").is_symlink()
        and (candidate / "pyproject.toml").is_file()
    )
    repository_root = (
        candidate
        if local_repository_available and candidate is not None
        else remote_repository_root
    )
    base_modal_path = (
        resolved_source.with_name("modal_gpu_reclamation_integrated_v11.py")
        if local_repository_available
        else repository_root / "experiments/branchfabric/modal_gpu_reclamation_integrated_v11.py"
    )
    return (
        repository_root,
        base_modal_path,
        repository_root
        / "python/sloforge/helix/characterization/gpu_reclamation_v11_methodology.py",
    )


def _validate_bootstrap_sources(base_modal_path: Path, methodology_path: Path) -> None:
    expected = (
        (base_modal_path, _BASE_MODAL_SHA256, "base Modal"),
        (methodology_path, _METHODOLOGY_SHA256, "methodology"),
    )
    for path, sha256, label in expected:
        if (
            path.is_symlink()
            or not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != sha256
        ):
            raise RuntimeError(f"kill/recompute {label} source changed before import: {path}")


LOCAL_REPOSITORY_ROOT, _BASE_MODAL_PATH, _METHODOLOGY_PATH = _resolve_bootstrap_paths(
    Path(__file__)
)
_validate_bootstrap_sources(_BASE_MODAL_PATH, _METHODOLOGY_PATH)
import modal_gpu_reclamation_integrated_v11 as base  # noqa: E402

from sloforge.helix.characterization.gpu_reclamation_v11_methodology import (  # noqa: E402
    V11_KILL_RECOMPUTE_RESERVATION_WALL_SECONDS,
    Experiment004V11KillRecomputeConfig,
)

_REMOTE_RUNTIME_BINDINGS = {
    "experiments/branchfabric/gpu_reclamation_kill_recompute_controller_v11.py": (
        "05d790d5d1ac1e23ba98cccc167a24855cc2b5ec5619fc295d7a3556c9e040c1"
    ),
    "experiments/branchfabric/gpu_reclamation_kill_recompute_worker_v11.py": (
        "6266db3fbd3809ec7ab5561f33b8283df93801264068614fd6ec4bd9c7ee2579"
    ),
    "experiments/branchfabric/gpu_reclamation_integrated_controller_v11.py": (
        "c7c07ea0444874a100aa6d6da273058edd4f8cee03548e2c08c62bca5ecae2fc"
    ),
    "experiments/branchfabric/gpu_reclamation_integrated_worker_v11.py": (
        "a9312a017116ffc86ac311b578da67f2afb97198815f30369c8ee3e0159c2002"
    ),
    "experiments/branchfabric/gpu_reclamation_integrated_trigger_v11.py": (
        "9e81ab90ed6d7cb8f7f77e9fd63f2a0b7f3001422c53bcef84eb01344c83d2cc"
    ),
    "experiments/branchfabric/gpu_reclamation_worker_v11.py": (
        "08449e5af89759847add7433054b2383eb9b1659785fb7e1184f25dc5e02de11"
    ),
    "experiments/branchfabric/gpu_reclamation_worker.py": (
        "e6858e445ddbca912877fc67155dd9422c82d6f6fd4dcbf71871831dee15008a"
    ),
    "experiments/branchfabric/gpu_reclamation_v10_serving.py": (
        "d53b007be4fc72d760ae7a83223773a206e0a0eed52137a834ce5b601bcad3ba"
    ),
    "experiments/branchfabric/gpu_capacity_calibration_worker.py": (
        "2c41499a847879b813ce01470a7e32625d772fc07233753404296125c4afc225"
    ),
    "experiments/branchfabric/gpu_reclamation_controller.py": (
        "c91ccd8b3aa4b4c5a9034a5137ccc33257e3471528a96dcc2ca7920c8b7eefc9"
    ),
    "python/sloforge/continuum/adapters/vllm_allocator_epochs.py": (
        "db543a8c4e7c2c7628e78f7dd0c4b04bcab60a4234bc2829787cca21214e35ce"
    ),
    "python/sloforge/continuum/adapters/vllm_reclamation.py": (
        "fa8f59b6e6755d0eb5c6c008d26df48b5fcbb4ad85f2af3b84233808c0ba953f"
    ),
    "python/sloforge/continuum/adapters/vllm_reclamation_v11.py": (
        "697d9d9786fbd1759bc4e4242f5b507a7123d1f6daf1af42c12cffc790dde458"
    ),
    "python/sloforge/continuum/adapters/vllm_reclamation_v11_ownership.py": (
        "0ca553fbe21b431bc315553e792b7b4de523da3c44f84f36e95ff469ba3e6de0"
    ),
    "python/sloforge/continuum/adapters/vllm_reclamation_v11_sync.py": (
        "05dc160d881f0b772d77db4f6b96298fe4290a7de81be9abcc876e83cada2820"
    ),
    "python/sloforge/continuum/adapters/vllm_metadata_0230.py": (
        "779c446cbc1ae302ee76bf9d9717f7037aafa09216758b2d0d6e7b1bd06dcf95"
    ),
    "python/sloforge/continuum/adapters/real_runtime.py": (
        "b7b790e03da5f31c5e68dd1985b8163451d48ea8ee46c805fe20ba9f38aba8f9"
    ),
    "python/sloforge/continuum/adapters/vllm_live.py": (
        "e22243b768c07d35afb6f98a40e979576a165c804b84944dba53d1aa1fca7ad4"
    ),
    "python/sloforge/helix/characterization/gpu_capacity_calibration.py": (
        "7ffb993004fbc9b595cc3c5fc2e503ffd9dd2370befdeb8714d017badc4e0b3d"
    ),
    "python/sloforge/helix/characterization/gpu_reclamation_accounting.py": (
        "0285017813cb07367bb26acb24798bfc2b036ca3955907c82c2c6a1f55ba2a1c"
    ),
    "python/sloforge/helix/characterization/gpu_reclamation_instrumentation.py": (
        "32b3d134496b627354f4813d94f9c827a6ca4d04b1e0092bd20c985419ecaf53"
    ),
    "python/sloforge/helix/characterization/gpu_reclamation_methodology.py": (
        "3a6217ec2b9c8ed4302a347ce5e36a0e1b1a19e8c1b1d38df6f6ea186b5507c3"
    ),
    "experiments/branchfabric/gpu_capacity_calibration_controller.py": (
        "31fb70855e49d3f59493592de84120fc82f6a950e5b5edd0e775d5ba47978aa2"
    ),
    "python/sloforge/continuum/adapters/vllm_reclamation_v11_accounting.py": (
        "fe8c913af0414ec402ce230ffb19e8db9474bfe19b8ad539f80f0419c99b23ec"
    ),
    "python/sloforge/continuum/adapters/vllm_live_trace.py": (
        "1e1cbf417e3996d4f22e27e1097be5c0c6282b5d86cd3090a705337511266bb6"
    ),
    "python/sloforge/continuum/adapters/external.py": (
        "94784ebdf6392f824bfde953f184042cb4027a158a2c8e9a34e1b14b4e7759f4"
    ),
    "python/sloforge/continuum/adapters/sdk.py": (
        "6a1ca84eebf8865ee0e784961bc6e635789352b3c821fdbcafd5a512906bf361"
    ),
    "python/sloforge/continuum/ir/canonical.py": (
        "ba84929cb375e0ac8b391970b5ce863b2680be0e98552b7e049f952bab7046c2"
    ),
    "python/sloforge/helix/characterization/gpu_reclamation.py": (
        "27c38db4bb4b7c1b61a29a54ab33f4e4769725a95754d86f4ce54e9a73a175ce"
    ),
    "python/sloforge/helix/characterization/gpu_reclamation_trace.py": (
        "b9e416500c79a6e3620f6962d8fbdd187b8dfe40260758e9b7a3fe2e3659cc73"
    ),
    "python/sloforge/helix/characterization/trace/__init__.py": (
        "73cf51ae90fadf84a0fee028818ec91cafc0ba8aaa002c185c701ae533dd2542"
    ),
    "python/sloforge/helix/characterization/trace/adapters.py": (
        "84a9e8c190b9016843df89c18af7beb071bbde08097315a45104cdeb6dd889f2"
    ),
    "python/sloforge/helix/characterization/trace/buffer.py": (
        "4f768e042a07dc3acf206fa598589869233ddaa4b0365ed0dfb709cf7230d7f9"
    ),
    "python/sloforge/helix/characterization/trace/canonical.py": (
        "7a765e489858b834424b810e0a0936c3062c301972eb720357c117de02b534c2"
    ),
    "python/sloforge/helix/characterization/trace/conversion.py": (
        "f74e631d511bd62d28221c40f7e0ba9f55e83035dd70ddf9d0fa2ac7e2b70798"
    ),
    "python/sloforge/helix/characterization/trace/io.py": (
        "fecb5332fb35a614c4e6ef9e48b4991afe66a6203cbc346380cb63a0c6a2eba6"
    ),
    "python/sloforge/helix/characterization/trace/manifest.py": (
        "1fa378df8019c5767a865f70e5a7d4262bf85ebf55704aa88cac6848c91781a1"
    ),
    "python/sloforge/helix/characterization/trace/models.py": (
        "da3a5d4cbbe0c418c3ab0506821781f29c3bf74f26741695196a8dfb5ff3b8bf"
    ),
    "python/sloforge/helix/characterization/trace/parquet.py": (
        "4d450f39bc4c86783aaac69daaaf1b7dad1ff80175059e68a8ce2e9fec5a0bd0"
    ),
    "python/sloforge/helix/characterization/trace/perfetto.py": (
        "4edc9cdfc3b2c00b50d3667cefdf2aa33eb43b2b4a4e6fae769b58b56047f67c"
    ),
    "python/sloforge/helix/characterization/gpu_reclamation_v10_authorization.py": (
        "3a06207c4865bf6df5fc38027e66d793726b89e0ca35a5472d9fa1f0d44ad840"
    ),
}


def _validate_remote_runtime_closure(repository_root: Path) -> None:
    resolved_root = repository_root.resolve(strict=True)
    for reference, expected_sha256 in _REMOTE_RUNTIME_BINDINGS.items():
        path = repository_root / reference
        if (
            path.is_symlink()
            or not path.is_file()
            or resolved_root not in path.resolve(strict=True).parents
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256
        ):
            raise RuntimeError(f"kill/recompute remote runtime changed: {reference}")


APP_NAME = "sloforge-branchfabric-gpu-reclamation-004-v11-kill-recompute"
GPU_REQUEST = "A100-80GB:2"
GPU_COUNT = 2
GPU_FUNCTION_TIMEOUT_SECONDS = 660
GPU_FUNCTION_STARTUP_TIMEOUT_SECONDS = 180
FUNCTION_RESULT_PUBLICATION_GRACE_SECONDS = 60
FIRST_FUNCTION_CALL_POLL_TIMEOUT_SECONDS = (
    GPU_FUNCTION_STARTUP_TIMEOUT_SECONDS + GPU_FUNCTION_TIMEOUT_SECONDS
)
FUNCTION_CALL_REPOLL_TIMEOUT_SECONDS = (
    GPU_FUNCTION_TIMEOUT_SECONDS + FUNCTION_RESULT_PUBLICATION_GRACE_SECONDS
)
POST_CONTROLLER_RESERVE_SECONDS = 10.0
MAXIMUM_GPU_SECONDS = 1320.0
REMOTE_EXPERIMENT_PREFIX = "experiment-004/v11/kill-recompute/modal"
_LAUNCH_TOKEN = os.getenv("SLOFORGE_MODAL_PREFLIGHT_TOKEN", "")
_CLOUD_GRAPH_ENABLED = len(_LAUNCH_TOKEN) == 64 and all(
    character in "0123456789abcdef" for character in _LAUNCH_TOKEN
)
_MODAL_CLI_RUN = "modal" in Path(sys.argv[0]).as_posix().lower() and "run" in sys.argv[1:]
if _MODAL_CLI_RUN and not _CLOUD_GRAPH_ENABLED:
    raise RuntimeError("refusing to hydrate kill/recompute without budget preflight")

_Attempt = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9-]{7,95}$")]
_Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class KillRecomputeRemoteAuthorization(_StrictModel):
    schema_version: Literal[
        "sloforge.branchfabric.experiment-004-kill-recompute-authorization/v1"
    ] = "sloforge.branchfabric.experiment-004-kill-recompute-authorization/v1"
    reservation_id: _Attempt
    reservation_commitment_sha256: _Sha256
    config_sha256: _Sha256
    preflight_token_sha256: _Sha256
    requested_gpu: Literal["A100-80GB"] = "A100-80GB"
    gpu_count: Literal[2] = 2
    maximum_wall_seconds: float = Field(ge=660.0, le=660.0, allow_inf_nan=False)
    maximum_gpu_seconds: float = Field(ge=1320.0, le=1320.0, allow_inf_nan=False)
    budget_usd: float = Field(gt=0.0, le=80.0, allow_inf_nan=False)


def _write_new_fsynced_json(path: Path, payload: dict[str, Any]) -> None:
    """Persist local FunctionCall recovery evidence before any blocking poll."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or path.exists() or path.is_symlink():
        raise FileExistsError(f"kill/recompute FunctionCall evidence already exists: {path}")
    encoded = base._canonical_bytes(payload)
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _get_function_call_result_with_repoll(
    call: Any,
    *,
    evidence_root: Path,
    attempt_id: str,
    reservation_id: str,
    reservation_commitment_sha256: str,
    config_sha256: str,
    observed_at_utc: Any = None,
) -> Any:
    """Poll one immutable FunctionCall twice without unwinding the ephemeral App.

    Modal 1.5.3 includes queue/image hydration in ``FunctionCall.get``'s poll
    timeout.  A timeout does not cancel the call, but allowing it to escape the
    local entrypoint disconnects the ephemeral App and interrupts the remote
    Function.  Persist the call ID first, retain the App context after the
    first timeout, and repoll that exact call for one full Function bound plus
    a bounded immutable-publication grace.
    """

    call_id = getattr(call, "object_id", None)
    if not isinstance(call_id, str) or re.fullmatch(r"fc-[0-9A-HJKMNP-TV-Z]{26}", call_id) is None:
        raise ValueError("kill/recompute Modal FunctionCall ID is invalid")
    if (
        re.fullmatch(r"[a-z0-9][a-z0-9-]{7,95}", attempt_id) is None
        or re.fullmatch(r"[a-z0-9][a-z0-9-]{7,95}", reservation_id) is None
        or re.fullmatch(r"[0-9a-f]{64}", reservation_commitment_sha256) is None
        or re.fullmatch(r"[0-9a-f]{64}", config_sha256) is None
    ):
        raise ValueError("kill/recompute FunctionCall identity evidence is invalid")
    utc_now = (
        (lambda: datetime.now(UTC).isoformat()) if observed_at_utc is None else observed_at_utc
    )
    common = {
        "attempt_id": attempt_id,
        "function_call_id": call_id,
        "reservation_id": reservation_id,
        "reservation_commitment_sha256": reservation_commitment_sha256,
        "config_sha256": config_sha256,
        "function_timeout_seconds": GPU_FUNCTION_TIMEOUT_SECONDS,
        "function_startup_timeout_seconds": GPU_FUNCTION_STARTUP_TIMEOUT_SECONDS,
        "first_poll_timeout_seconds": FIRST_FUNCTION_CALL_POLL_TIMEOUT_SECONDS,
        "repoll_timeout_seconds": FUNCTION_CALL_REPOLL_TIMEOUT_SECONDS,
        "publication_grace_seconds": FUNCTION_RESULT_PUBLICATION_GRACE_SECONDS,
    }
    _write_new_fsynced_json(
        evidence_root / "function-call.json",
        {
            "schema_version": ("sloforge.branchfabric.kill-recompute-function-call-capture/v1"),
            "status": "SPAWNED_AND_PERSISTED_BEFORE_FIRST_POLL",
            **common,
            "recorded_at_utc": utc_now(),
        },
    )
    try:
        return call.get(timeout=FIRST_FUNCTION_CALL_POLL_TIMEOUT_SECONDS)
    except TimeoutError as error:
        _write_new_fsynced_json(
            evidence_root / "first-poll-timeout.json",
            {
                "schema_version": ("sloforge.branchfabric.kill-recompute-function-call-timeout/v1"),
                "status": "FIRST_POLL_TIMED_OUT_REPOLLING_SAME_CALL",
                **common,
                "same_function_call_repolled": True,
                "error": {"type": type(error).__name__, "message": str(error)},
                "recorded_at_utc": utc_now(),
            },
        )
    try:
        return call.get(timeout=FUNCTION_CALL_REPOLL_TIMEOUT_SECONDS)
    except TimeoutError as error:
        _write_new_fsynced_json(
            evidence_root / "terminal-poll-timeout.json",
            {
                "schema_version": (
                    "sloforge.branchfabric.kill-recompute-function-call-terminal-timeout/v1"
                ),
                "status": "SECOND_POLL_TIMED_OUT_FAILING_CLOSED",
                **common,
                "same_function_call_repolled": True,
                "error": {"type": type(error).__name__, "message": str(error)},
                "recorded_at_utc": utc_now(),
            },
        )
        raise


def _parse_config(payload: dict[str, Any]) -> Experiment004V11KillRecomputeConfig:
    return Experiment004V11KillRecomputeConfig.model_validate(payload, strict=True)


def _validate_authorization(
    config: Experiment004V11KillRecomputeConfig,
    payload: dict[str, Any],
    *,
    authorized_budget_usd: float,
) -> KillRecomputeRemoteAuthorization:
    from sloforge.helix.characterization.gpu_reclamation_methodology import (
        Experiment004GpuHourLedger,
        reserve_gpu_invocation,
    )
    from sloforge.helix.characterization.gpu_reclamation_v11_methodology import (
        validate_bound_artifact,
    )

    authorization = KillRecomputeRemoteAuthorization.model_validate(payload, strict=True)
    config_sha256 = hashlib.sha256(base._canonical_bytes(config)).hexdigest()
    snapshot_path = validate_bound_artifact(
        Path("/opt/sloforge"),
        reference=config.ledger_snapshot_artifact,
        expected_sha256=config.ledger_snapshot_sha256,
    )
    snapshot = Experiment004GpuHourLedger.model_validate_json(
        snapshot_path.read_text(), strict=True
    )
    reserved, preflight = reserve_gpu_invocation(
        snapshot,
        reservation_id=authorization.reservation_id,
        invocation_id=config.attempt_id,
        maximum_wall_seconds=V11_KILL_RECOMPUTE_RESERVATION_WALL_SECONDS,
        config_sha256=config_sha256,
        gpu_count=GPU_COUNT,
    )
    reconstructed_commitment = hashlib.sha256(
        base._canonical_bytes(reserved.reservations[0])
    ).hexdigest()
    if authorization.config_sha256 != config_sha256:
        raise RuntimeError("kill/recompute authorization does not bind the immutable config")
    if authorization.preflight_token_sha256 != hashlib.sha256(_LAUNCH_TOKEN.encode()).hexdigest():
        raise RuntimeError("kill/recompute authorization does not bind the preflight token")
    if authorization.budget_usd > authorized_budget_usd:
        raise RuntimeError("kill/recompute launch budget exceeds explicit authorization")
    if (
        reconstructed_commitment != authorization.reservation_commitment_sha256
        or preflight.proposed_maximum_gpu_seconds != MAXIMUM_GPU_SECONDS
        or preflight.hard_limit_passed is not True
    ):
        raise RuntimeError("kill/recompute remote reservation reconstruction changed")
    return authorization


def _validate_config_budget(
    config: Experiment004V11KillRecomputeConfig,
    *,
    repository_root: Path,
) -> dict[str, Any]:
    from sloforge.helix.characterization.gpu_reclamation_v11_methodology import (
        validate_bound_artifact,
    )

    budget_path = validate_bound_artifact(
        repository_root,
        reference=config.budget_authorization,
        expected_sha256=config.budget_authorization_sha256,
    )
    payload = json.loads(budget_path.read_text())
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version")
        != "sloforge.branchfabric.experiment-004-budget-authorization/v1"
        or payload.get("authorized_gpu_budget_usd") != 80.0
        or payload.get("authorized_cumulative_gpu_seconds") != 43_200.0
        or "one kill-and-recompute comparison only after integrated v11 passes and freezes"
        not in payload.get("authorized_scope", [])
    ):
        raise RuntimeError("kill/recompute budget artifact does not authorize this arm")
    return payload


def run_kill_recompute(
    config_payload: dict[str, Any], authorization_payload: dict[str, Any]
) -> dict[str, Any]:
    entry_ns = time.monotonic_ns()
    raw_attempt_id = config_payload.get("attempt_id")
    attempt_id = (
        raw_attempt_id
        if isinstance(raw_attempt_id, str)
        and 8 <= len(raw_attempt_id) <= 96
        and raw_attempt_id[0].isalnum()
        and all(
            character.islower() or character.isdigit() or character == "-"
            for character in raw_attempt_id
        )
        else "pre-controller-"
        + hashlib.sha256(base._canonical_bytes(config_payload)).hexdigest()[:16]
    )
    work_root = Path("/tmp/sloforge-branchfabric-exp004-v11-kill-recompute") / attempt_id
    snapshot = base.MODEL_MOUNT / "snapshots" / base.MODEL_REVISION
    function_call_id: str | None = None
    try:
        _validate_remote_runtime_closure(Path("/opt/sloforge"))
        base._require_modal_sdk_version()
        function_call_id = modal.current_function_call_id()
        if not function_call_id:
            raise RuntimeError("Modal did not expose the kill/recompute FunctionCall ID")
        config = _parse_config(config_payload)
        bound_budget = _validate_config_budget(config, repository_root=Path("/opt/sloforge"))
        authorization = _validate_authorization(
            config,
            authorization_payload,
            authorized_budget_usd=float(bound_budget["authorized_gpu_budget_usd"]),
        )
        base._validate_model_snapshot(snapshot)
    except BaseException as error:
        work_root.mkdir(parents=True, exist_ok=True)
        primary_error = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        cleanup = {
            "schema_version": "sloforge.branchfabric.kill-recompute-pre-controller-cleanup/v1",
            "status": "PASS",
            "phase": "PRE_CONTROLLER",
            "attempt_id": attempt_id,
            "function_call_id": function_call_id or "UNAVAILABLE_AT_PRE_CONTROLLER_FAILURE",
            "controller_started": False,
            "controller_owned_children": 0,
            "engines_created": 0,
            "cuda_owning_children": 0,
            "ipc_resources_created": 0,
            "primary_error_preserved": True,
        }
        base._write_new(work_root / "pre-controller-failure.json", primary_error)
        base._write_new(work_root / "in_function_cleanup.json", cleanup)
        completion = {
            "schema_version": "sloforge.branchfabric.kill-recompute-completion/v1",
            "status": "failed",
            "scientific_status": "invalid-pre-controller",
            "attempt_id": attempt_id,
            "reservation_id": authorization_payload.get("reservation_id"),
            "reservation_commitment_sha256": authorization_payload.get(
                "reservation_commitment_sha256"
            ),
            "config_sha256": authorization_payload.get("config_sha256"),
            "function_call_id": function_call_id or "UNAVAILABLE_AT_PRE_CONTROLLER_FAILURE",
            "requested_gpu": GPU_REQUEST,
            "gpu_count": GPU_COUNT,
            "controller_and_analysis_interval_seconds": (time.monotonic_ns() - entry_ns) / 1e9,
            "gpu_allocation_seconds_status": "pending-final-function-return",
            "hardware_comparability": {
                "modal_request": GPU_REQUEST,
                "observed_gpu_names": [],
                "observed_gpu_uuids": [],
                "direct_k_comparison_ready": False,
            },
            "bound_k_evidence": {
                "integrated_k_status": config_payload.get("integrated_k_status_artifact"),
                "integrated_k_status_sha256": config_payload.get("integrated_k_status_sha256"),
                "integrated_k_remote_manifest": config_payload.get("integrated_k_remote_manifest"),
                "integrated_k_remote_manifest_sha256": config_payload.get(
                    "integrated_k_remote_manifest_sha256"
                ),
                "integrated_k_provider_cleanup": config_payload.get(
                    "integrated_k_provider_cleanup_artifact"
                ),
                "integrated_k_provider_cleanup_sha256": config_payload.get(
                    "integrated_k_provider_cleanup_sha256"
                ),
                "optimized_v11_freeze": config_payload.get("optimized_v11_freeze_artifact"),
                "optimized_v11_freeze_sha256": config_payload.get("optimized_v11_freeze_sha256"),
                "optimized_v11_freeze_tag_binding": config_payload.get(
                    "optimized_v11_freeze_tag_binding_artifact"
                ),
                "optimized_v11_freeze_tag_binding_sha256": config_payload.get(
                    "optimized_v11_freeze_tag_binding_sha256"
                ),
                "ledger_snapshot": config_payload.get("ledger_snapshot_artifact"),
                "ledger_snapshot_sha256": config_payload.get("ledger_snapshot_sha256"),
            },
            "absolute_deadlines": {
                "function_entry_monotonic_ns": entry_ns,
                "controller_deadline_monotonic_ns": entry_ns
                + round((GPU_FUNCTION_TIMEOUT_SECONDS - POST_CONTROLLER_RESERVE_SECONDS) * 1e9),
                "function_deadline_monotonic_ns": entry_ns
                + round(GPU_FUNCTION_TIMEOUT_SECONDS * 1e9),
                "post_controller_reserve_seconds": POST_CONTROLLER_RESERVE_SECONDS,
            },
            "controller": None,
            "in_function_cleanup": cleanup,
            "run_error": primary_error,
            "completed_at_utc": datetime.now(UTC).isoformat(),
        }
        base._write_new(work_root / "function-completion.json", completion)
        staging = base.RESULTS_MOUNT / f"{REMOTE_EXPERIMENT_PREFIX}/{attempt_id}.staging"
        final = base.RESULTS_MOUNT / f"{REMOTE_EXPERIMENT_PREFIX}/{attempt_id}"
        manifest_sha256 = base._publish_immutable_result(
            work_root=work_root,
            staging=staging,
            final=final,
            results_root=base.RESULTS_MOUNT,
            volume=base.results_volume,
            attempt_id=attempt_id,
        )
        shutil.rmtree(work_root)
        elapsed = (time.monotonic_ns() - entry_ns) / 1e9
        return {
            **completion,
            "gpu_allocation_seconds": elapsed,
            "gpu_seconds": GPU_COUNT * elapsed,
            "gpu_hours": GPU_COUNT * elapsed / 3600.0,
            "remote_prefix": str(final.relative_to(base.RESULTS_MOUNT)),
            "remote_manifest_sha256": manifest_sha256,
        }
    config_path = work_root.parent / f"{config.attempt_id}.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    base._write_new(config_path, config)
    controller_deadline_ns = entry_ns + round(
        (GPU_FUNCTION_TIMEOUT_SECONDS - POST_CONTROLLER_RESERVE_SECONDS) * 1e9
    )
    controller: dict[str, Any] = {
        "schema_version": "sloforge.branchfabric.kill-recompute-controller-failure/v1",
        "status": "failed",
    }
    run_error: dict[str, Any] | None = None
    hardware: dict[str, Any] = {
        "modal_request": GPU_REQUEST,
        "observed_gpu_names": [],
        "observed_gpu_uuids": [],
        "direct_k_comparison_ready": False,
    }
    try:
        from gpu_reclamation_kill_recompute_controller_v11 import (
            run_kill_recompute_controller,
        )

        controller = run_kill_recompute_controller(
            config_path=config_path,
            work_root=work_root,
            worker_path=Path(
                "/opt/sloforge/experiments/branchfabric/"
                "gpu_reclamation_kill_recompute_worker_v11.py"
            ),
            model_snapshot=snapshot,
            repository_root=Path("/opt/sloforge"),
            absolute_deadline_ns=controller_deadline_ns,
        )
        base._require_in_function_cleanup(controller, work_root=work_root)
        if controller.get("status") != "succeeded":
            raise RuntimeError("kill/recompute controller failed closed")
        hardware = base._hardware_comparability(controller)
        hardware["direct_k_comparison_ready"] = hardware.get("direct_v10_timing_comparable") is True
    except BaseException as error:
        run_error = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        work_root.mkdir(parents=True, exist_ok=True)
        base._write_new(work_root / "function-failure.json", run_error)
    finally:
        work_root.mkdir(parents=True, exist_ok=True)
        cleanup = controller.get("in_function_cleanup")
        cleanup_present = isinstance(cleanup, dict)
        base._write_new(
            work_root / "function-finally-cleanup.json",
            {
                "schema_version": ("sloforge.branchfabric.kill-recompute-function-cleanup/v1"),
                "attempt_id": config.attempt_id,
                "controller_status": controller.get("status"),
                "controller_cleanup_evidence_present": cleanup_present,
                "in_function_cleanup_pass": bool(cleanup_present and cleanup.get("pass") is True),
                "in_function_cleanup_artifact": "in_function_cleanup.json",
                "provider_cleanup_status": "pending-function-return-and-local-postflight",
                "recorded_at_utc": datetime.now(UTC).isoformat(),
            },
        )
    interval_seconds = (time.monotonic_ns() - entry_ns) / 1e9
    completion = {
        "schema_version": "sloforge.branchfabric.kill-recompute-completion/v1",
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
        "controller_and_analysis_interval_seconds": interval_seconds,
        "gpu_allocation_seconds_status": "pending-final-function-return",
        "hardware_comparability": hardware,
        "bound_k_evidence": {
            "integrated_k_status": config.integrated_k_status_artifact,
            "integrated_k_status_sha256": config.integrated_k_status_sha256,
            "integrated_k_remote_manifest": config.integrated_k_remote_manifest,
            "integrated_k_remote_manifest_sha256": (config.integrated_k_remote_manifest_sha256),
            "integrated_k_provider_cleanup": config.integrated_k_provider_cleanup_artifact,
            "integrated_k_provider_cleanup_sha256": (config.integrated_k_provider_cleanup_sha256),
            "optimized_v11_freeze": config.optimized_v11_freeze_artifact,
            "optimized_v11_freeze_sha256": config.optimized_v11_freeze_sha256,
            "optimized_v11_freeze_tag_binding": (config.optimized_v11_freeze_tag_binding_artifact),
            "optimized_v11_freeze_tag_binding_sha256": (
                config.optimized_v11_freeze_tag_binding_sha256
            ),
            "ledger_snapshot": config.ledger_snapshot_artifact,
            "ledger_snapshot_sha256": config.ledger_snapshot_sha256,
        },
        "absolute_deadlines": {
            "function_entry_monotonic_ns": entry_ns,
            "controller_deadline_monotonic_ns": controller_deadline_ns,
            "function_deadline_monotonic_ns": entry_ns + round(GPU_FUNCTION_TIMEOUT_SECONDS * 1e9),
            "post_controller_reserve_seconds": POST_CONTROLLER_RESERVE_SECONDS,
        },
        "controller": controller,
        "in_function_cleanup": controller.get("in_function_cleanup"),
        "run_error": run_error,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    base._write_new(work_root / "function-completion.json", completion)
    staging = base.RESULTS_MOUNT / f"{REMOTE_EXPERIMENT_PREFIX}/{config.attempt_id}.staging"
    final = base.RESULTS_MOUNT / f"{REMOTE_EXPERIMENT_PREFIX}/{config.attempt_id}"
    manifest_sha256 = base._publish_immutable_result(
        work_root=work_root,
        staging=staging,
        final=final,
        results_root=base.RESULTS_MOUNT,
        volume=base.results_volume,
        attempt_id=config.attempt_id,
    )
    shutil.rmtree(work_root)
    final_elapsed_seconds = (time.monotonic_ns() - entry_ns) / 1e9
    return {
        **completion,
        "gpu_allocation_seconds": final_elapsed_seconds,
        "gpu_seconds": GPU_COUNT * final_elapsed_seconds,
        "gpu_hours": GPU_COUNT * final_elapsed_seconds / 3600.0,
        "remote_prefix": str(final.relative_to(base.RESULTS_MOUNT)),
        "remote_manifest_sha256": manifest_sha256,
    }


app = modal.App(
    APP_NAME,
    tags={"project": "sloforge", "experiment": "branchfabric-004-kill-recompute"},
)
gpu_image = base.gpu_image

_run_kill_recompute_function: Any = None
_launch_token_secret: Any = None


def _materialize(attempt_id: str) -> dict[str, str]:
    return {
        "remote_path": f"{REMOTE_EXPERIMENT_PREFIX}/{attempt_id}",
        "volume_name": base.RESULTS_VOLUME_NAME,
    }


def main(*, config_path: str) -> None:
    base._require_modal_sdk_version()
    config = _parse_config(json.loads(Path(config_path).read_text()))
    budget_artifact = _validate_config_budget(config, repository_root=base.LOCAL_REPOSITORY_ROOT)
    budget = base._positive_budget(os.getenv("SLOFORGE_GPU_BUDGET_USD"))
    if budget > float(budget_artifact["authorized_gpu_budget_usd"]):
        raise RuntimeError("kill/recompute requested budget exceeds authorization")
    reservation_id = os.getenv("SLOFORGE_EXP004_RESERVATION_ID")
    reservation_commitment = os.getenv("SLOFORGE_EXP004_RESERVATION_COMMITMENT_SHA256")
    if (
        reservation_id is None
        or reservation_commitment is None
        or len(reservation_commitment) != 64
        or any(character not in "0123456789abcdef" for character in reservation_commitment)
        or not _CLOUD_GRAPH_ENABLED
    ):
        raise RuntimeError("kill/recompute requires reservation and preflight evidence")
    authorization = KillRecomputeRemoteAuthorization(
        reservation_id=reservation_id,
        reservation_commitment_sha256=reservation_commitment,
        config_sha256=hashlib.sha256(base._canonical_bytes(config)).hexdigest(),
        preflight_token_sha256=hashlib.sha256(_LAUNCH_TOKEN.encode()).hexdigest(),
        maximum_wall_seconds=V11_KILL_RECOMPUTE_RESERVATION_WALL_SECONDS,
        maximum_gpu_seconds=MAXIMUM_GPU_SECONDS,
        budget_usd=budget,
    )
    call = _run_kill_recompute_function.spawn(
        config.model_dump(mode="json"), authorization.model_dump(mode="json")
    )
    evidence_root = (
        LOCAL_REPOSITORY_ROOT
        / "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/kill-recompute"
        / "function-calls"
        / config.attempt_id
    )
    result = _get_function_call_result_with_repoll(
        call,
        evidence_root=evidence_root,
        attempt_id=config.attempt_id,
        reservation_id=authorization.reservation_id,
        reservation_commitment_sha256=authorization.reservation_commitment_sha256,
        config_sha256=authorization.config_sha256,
    )
    print(
        base._canonical_bytes(
            {"result": result, "materialized": _materialize(config.attempt_id)}
        ).decode(),
        end="",
    )


if _CLOUD_GRAPH_ENABLED:
    _launch_token_secret = modal.Secret.from_dict({"SLOFORGE_MODAL_PREFLIGHT_TOKEN": _LAUNCH_TOKEN})
    _run_kill_recompute_function = app.function(
        name="run-kill-recompute-v11",
        image=gpu_image,
        gpu=GPU_REQUEST,
        secrets=[_launch_token_secret],
        volumes={
            str(base.MODEL_MOUNT): base.model_volume.with_mount_options(read_only=True),
            str(base.RESULTS_MOUNT): base.results_volume,
        },
        cpu=16.0,
        memory=64 * 1024,
        timeout=GPU_FUNCTION_TIMEOUT_SECONDS,
        startup_timeout=GPU_FUNCTION_STARTUP_TIMEOUT_SECONDS,
        retries=0,
        max_containers=1,
        buffer_containers=0,
        single_use_containers=True,
    )(run_kill_recompute)
    _local_entrypoint = app.local_entrypoint()(main)


__all__ = [
    "APP_NAME",
    "GPU_FUNCTION_TIMEOUT_SECONDS",
    "GPU_REQUEST",
    "KillRecomputeRemoteAuthorization",
    "app",
    "main",
    "run_kill_recompute",
]
