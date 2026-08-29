"""Modal coordinator for bounded BranchFabric real GPU KV-state validation.

The app is ephemeral.  Only the named model and result Volumes persist.  Model
download and token construction run in a CPU-only function before any GPU call;
each GPU input executes exactly one independent or shared-root trial on one
explicit ``A100-80GB`` with automatic application retries disabled.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal, cast

import modal
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

MODAL_SDK_VERSION = "1.5.3"
APP_NAME = "sloforge-branchfabric-real-gpu-cow"
GPU_SKU = "A100-80GB"
GPU_COUNT = 1
GPU_HOUR_LIMIT = 4.0
GPU_HOUR_TARGET = 2.0
MODAL_A100_80GB_USD_PER_HOUR = 2.4984
MODAL_CPU_USD_PER_CORE_SECOND = 0.0000131
MODAL_MEMORY_USD_PER_GIB_SECOND = 0.00000222
BUDGET_RESERVE_FRACTION = 0.15
GPU_FUNCTION_TIMEOUT_SECONDS = 5400
GPU_FUNCTION_STARTUP_TIMEOUT_SECONDS = 1800
GPU_FUNCTION_CPU_CORES = 8.0
GPU_FUNCTION_MEMORY_GIB = 32.0
MODEL_FUNCTION_TIMEOUT_SECONDS = 7200
MODEL_FUNCTION_STARTUP_TIMEOUT_SECONDS = 900
MODEL_FUNCTION_CPU_CORES = 4.0
MODEL_FUNCTION_MEMORY_GIB = 16.0
MODEL_VOLUME_NAME = "sloforge-model-cache"
RESULTS_VOLUME_NAME = "sloforge-branchfabric-results"
REMOTE_EXPERIMENT_PREFIX = "experiment-002/modal"
VLLM_VERSION = "0.23.0"
TORCH_VERSION = "2.11.0"
TRANSFORMERS_VERSION = "5.14.1"
PYDANTIC_VERSION = "2.13.4"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
TOKENIZER_REVISION = MODEL_REVISION
POLICY_EPOCH = "branchfabric-modal-exp002-policy-v1"
ADAPTER_VERSION = "1.0.0"
CUDA_IMAGE = "nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04"
UV_VERSION = "0.10.2"
_SOURCE_PATH = Path(__file__).resolve()
_SOURCE_PARENTS = tuple(_SOURCE_PATH.parents)
_LOCAL_REPOSITORY_CANDIDATE = _SOURCE_PARENTS[2] if len(_SOURCE_PARENTS) > 2 else None
_LOCAL_WORKSPACE_AVAILABLE = bool(
    _LOCAL_REPOSITORY_CANDIDATE is not None
    and (_LOCAL_REPOSITORY_CANDIDATE / "pyproject.toml").is_file()
)
# Modal imports the mounted source as /root/modal_real_gpu_cow.py.  Local-only
# coordinator paths must therefore not assume the repository parent depth while
# the remote function is starting.
LOCAL_REPOSITORY_ROOT = (
    _LOCAL_REPOSITORY_CANDIDATE
    if _LOCAL_WORKSPACE_AVAILABLE and _LOCAL_REPOSITORY_CANDIDATE is not None
    else Path("/opt/sloforge")
)
LOCAL_EXPERIMENT_ROOT = (
    LOCAL_REPOSITORY_ROOT / "artifacts/branchfabric/gpu-validation/experiment-002"
)
MODEL_MOUNT = Path("/models")
RESULTS_MOUNT = Path("/results")
_CONTAINER_IMPORT_UTC = datetime.now(UTC).isoformat()
_CONTAINER_IMPORT_MONOTONIC_NS = time.monotonic_ns()
_TOKEN_CORPUS = (
    "BranchFabric validation measures immutable shared-prefix transformer KV state, "
    "append-only divergent suffix allocation, scheduler readiness, and decode interference. "
    "Every observation is version-pinned, seed-controlled, pointer-free, and tied to physical "
    "runtime block ownership. "
)
_ATTEMPT = Annotated[
    str,
    StringConstraints(pattern=r"^[a-z0-9][a-z0-9-]{7,95}$"),
]
_LAUNCH_TOKEN = os.getenv("SLOFORGE_MODAL_PREFLIGHT_TOKEN", "")
_CLOUD_GRAPH_ENABLED = len(_LAUNCH_TOKEN) == 64 and all(
    character in "0123456789abcdef" for character in _LAUNCH_TOKEN
)
_MODAL_CLI_RUN = "modal" in Path(sys.argv[0]).as_posix().lower() and "run" in sys.argv[1:]
if _MODAL_CLI_RUN and not _CLOUD_GRAPH_ENABLED:
    raise RuntimeError(
        "refusing to hydrate Modal without tools/branchfabric-modal-launch.py budget preflight"
    )


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class ModalBenchmarkConfig(_StrictModel):
    """One exact Modal GPU function input."""

    schema_version: Literal["sloforge.branchfabric.modal-real-gpu-cow-config/v1"] = (
        "sloforge.branchfabric.modal-real-gpu-cow-config/v1"
    )
    attempt_id: _ATTEMPT
    model: Literal["Qwen/Qwen2.5-7B-Instruct"] = "Qwen/Qwen2.5-7B-Instruct"
    model_revision: Literal["a09a35458c702b33eeacc393d103063234e8bc28"] = (
        "a09a35458c702b33eeacc393d103063234e8bc28"
    )
    tokenizer_revision: Literal["a09a35458c702b33eeacc393d103063234e8bc28"] = (
        "a09a35458c702b33eeacc393d103063234e8bc28"
    )
    runtime: Literal["vllm"] = "vllm"
    runtime_version: Literal["0.23.0"] = "0.23.0"
    fanout: Literal[1, 2, 8, 32]
    prefix_length: Literal[2048, 4096, 16384]
    suffix_length: Literal[16, 256]
    seed: int = Field(ge=0, lt=1 << 63)
    baseline_mode: Literal["independent", "shared_root"]
    tracing_level: Literal["disabled", "minimal", "full"] = "full"
    maximum_wall_seconds: int = Field(default=3600, ge=60, le=3600)
    initialization_timeout_seconds: int = Field(default=1200, ge=60, le=1800)
    cleanup_timeout_seconds: int = Field(default=60, ge=10, le=300)
    gpu_memory_utilization: float = Field(default=0.80, gt=0.0, le=0.90)
    gpu_sample_interval_seconds: float = Field(default=0.25, ge=0.05, le=2.0)

    @model_validator(mode="after")
    def bounded_shape(self) -> ModalBenchmarkConfig:
        if self.prefix_length in {2048, 4096}:
            if self.fanout != 2 or self.suffix_length != 16 or self.baseline_mode != "shared_root":
                raise ValueError("smoke requires shared_root, fanout 2, and suffix 16")
        elif self.fanout not in {1, 8, 32} or self.suffix_length != 256:
            raise ValueError("16K trials require fanout 1/8/32 and suffix 256")
        if (
            self.initialization_timeout_seconds
            + self.maximum_wall_seconds
            + self.cleanup_timeout_seconds
            + 300
            > GPU_FUNCTION_TIMEOUT_SECONDS
        ):
            raise ValueError("inner initialization/work/cleanup bounds require 300s outer headroom")
        return self


class CloudBudgetDecision(_StrictModel):
    budget_usd: float = Field(gt=0.0)
    reserve_fraction: float = Field(ge=0.0, lt=1.0)
    spendable_usd: float = Field(gt=0.0)
    already_consumed_gpu_hours: float = Field(ge=0.0)
    proposed_maximum_gpu_hours: float = Field(ge=0.0)
    projected_total_gpu_hours: float = Field(ge=0.0)
    projected_gpu_cost_usd: float = Field(ge=0.0)
    already_accounted_cloud_cost_usd: float = Field(ge=0.0)
    proposed_maximum_cloud_cost_usd: float = Field(ge=0.0)
    projected_total_cloud_cost_usd: float = Field(ge=0.0)
    authorized: Literal[True] = True


class ModalRemoteAuthorization(_StrictModel):
    """Coordinator-issued proof that the paid call has a durable local reservation."""

    schema_version: Literal["sloforge.branchfabric.modal-remote-authorization/v1"] = (
        "sloforge.branchfabric.modal-remote-authorization/v1"
    )
    kind: Literal["model_prepare", "gpu"]
    reservation_id: _ATTEMPT
    config_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    maximum_remote_seconds: int = Field(gt=0, le=24 * 3600)
    budget_usd: float = Field(gt=0.0, allow_inf_nan=False)


class ProviderTerminalAudit(_StrictModel):
    """Explicit operator evidence resolving an accepted-call ambiguity without a call ID."""

    schema_version: Literal["sloforge.branchfabric.modal-provider-terminal-audit/v1"] = (
        "sloforge.branchfabric.modal-provider-terminal-audit/v1"
    )
    attempt_id: _ATTEMPT
    reservation_id: _ATTEMPT
    kind: Literal["model_prepare", "gpu"]
    provider_terminal_state: Literal["not_found", "cancelled", "terminated", "completed"]
    provider_live_or_pending: Literal[False]
    audited_at_utc: datetime
    evidence_reference: Annotated[str, StringConstraints(min_length=1, max_length=4096)]
    auditor: Annotated[str, StringConstraints(min_length=1, max_length=256)]


def _remote_authorization(
    config: ModalBenchmarkConfig,
    *,
    kind: Literal["model_prepare", "gpu"],
    reservation_id: str,
    decision: CloudBudgetDecision,
) -> ModalRemoteAuthorization:
    maximum_seconds = (
        MODEL_FUNCTION_TIMEOUT_SECONDS + MODEL_FUNCTION_STARTUP_TIMEOUT_SECONDS
        if kind == "model_prepare"
        else GPU_FUNCTION_TIMEOUT_SECONDS + GPU_FUNCTION_STARTUP_TIMEOUT_SECONDS
    )
    return ModalRemoteAuthorization(
        kind=kind,
        reservation_id=reservation_id,
        config_sha256=hashlib.sha256(_canonical_bytes(config)).hexdigest(),
        maximum_remote_seconds=maximum_seconds,
        budget_usd=decision.budget_usd,
    )


def _validate_remote_authorization(
    config: ModalBenchmarkConfig,
    payload: dict[str, Any],
    *,
    kind: Literal["model_prepare", "gpu"],
) -> ModalRemoteAuthorization:
    authorization = ModalRemoteAuthorization.model_validate(payload)
    expected = hashlib.sha256(_canonical_bytes(config)).hexdigest()
    expected_seconds = (
        MODEL_FUNCTION_TIMEOUT_SECONDS + MODEL_FUNCTION_STARTUP_TIMEOUT_SECONDS
        if kind == "model_prepare"
        else GPU_FUNCTION_TIMEOUT_SECONDS + GPU_FUNCTION_STARTUP_TIMEOUT_SECONDS
    )
    if (
        authorization.kind != kind
        or authorization.config_sha256 != expected
        or authorization.maximum_remote_seconds != expected_seconds
    ):
        raise RuntimeError("Modal paid call lacks a matching coordinator authorization")
    return authorization


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_bytes(value)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _replace_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    payload = _canonical_bytes(value)
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        with path.open("rb") as handle:
            os.fsync(handle.fileno())


def _inventory(root: Path, *, exclude: frozenset[str] = frozenset()) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in exclude:
            continue
        records.append(
            {"relative_path": relative, "bytes": path.stat().st_size, "sha256": _sha256(path)}
        )
    return records


def _load_consumed_gpu_hours(ledger_path: Path) -> float:
    if not ledger_path.is_file():
        return 0.0
    payload = json.loads(ledger_path.read_text())
    value = payload.get("consumed_gpu_hours")
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("GPU-hour ledger has an invalid consumed_gpu_hours value")
    return float(value)


def _load_accounted_cloud_cost(ledger_path: Path) -> float:
    if not ledger_path.is_file():
        return 0.0
    payload = json.loads(ledger_path.read_text())
    consumed = payload.get("consumed_cloud_cost_usd", 0.0)
    reservations = payload.get("modal_in_flight_reservations", [])
    model_reservations = payload.get("modal_in_flight_model_reservations", [])
    if (
        not isinstance(consumed, (int, float))
        or not math.isfinite(consumed)
        or consumed < 0
        or not isinstance(reservations, list)
        or not isinstance(model_reservations, list)
    ):
        raise ValueError("GPU-hour ledger has invalid cloud-cost accounting")
    reserved = 0.0
    for item in [*reservations, *model_reservations]:
        value = item.get("maximum_cloud_cost_usd") if isinstance(item, dict) else None
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("GPU-hour ledger has an invalid cloud-cost reservation")
        reserved += float(value)
    return float(consumed) + reserved


def require_cloud_budget(
    *,
    maximum_gpu_seconds: int,
    maximum_model_prep_seconds: int = 0,
    ledger_path: Path | None = None,
) -> CloudBudgetDecision:
    """Fail closed before any remote call unless the repository cloud guard is set."""

    raw = os.getenv("SLOFORGE_GPU_BUDGET_USD")
    if raw is None:
        raise RuntimeError("SLOFORGE_GPU_BUDGET_USD is required before any Modal remote call")
    try:
        budget = float(raw)
    except ValueError as error:
        raise RuntimeError("SLOFORGE_GPU_BUDGET_USD must be a finite positive number") from error
    if not math.isfinite(budget) or budget <= 0:
        raise RuntimeError("SLOFORGE_GPU_BUDGET_USD must be a finite positive number")
    resolved_ledger = ledger_path or LOCAL_EXPERIMENT_ROOT / "gpu-hours.json"
    consumed = _load_consumed_gpu_hours(resolved_ledger)
    accounted_cloud_cost = _load_accounted_cloud_cost(resolved_ledger)
    proposed = maximum_gpu_seconds * GPU_COUNT / 3600.0
    projected_hours = consumed + proposed
    if projected_hours > GPU_HOUR_LIMIT:
        raise RuntimeError("projected Modal call would exceed the four GPU-hour hard limit")
    projected_gpu_cost = proposed * MODAL_A100_80GB_USD_PER_HOUR
    gpu_support_cost = maximum_gpu_seconds * (
        GPU_FUNCTION_CPU_CORES * MODAL_CPU_USD_PER_CORE_SECOND
        + GPU_FUNCTION_MEMORY_GIB * MODAL_MEMORY_USD_PER_GIB_SECOND
    )
    model_prep_cost = maximum_model_prep_seconds * (
        MODEL_FUNCTION_CPU_CORES * MODAL_CPU_USD_PER_CORE_SECOND
        + MODEL_FUNCTION_MEMORY_GIB * MODAL_MEMORY_USD_PER_GIB_SECOND
    )
    proposed_cloud_cost = projected_gpu_cost + gpu_support_cost + model_prep_cost
    projected_cloud_cost = accounted_cloud_cost + proposed_cloud_cost
    spendable = budget * (1.0 - BUDGET_RESERVE_FRACTION)
    if projected_cloud_cost > spendable:
        raise RuntimeError(
            "projected Modal compute cost exceeds the cloud budget after the 15% reserve"
        )
    return CloudBudgetDecision(
        budget_usd=budget,
        reserve_fraction=BUDGET_RESERVE_FRACTION,
        spendable_usd=spendable,
        already_consumed_gpu_hours=consumed,
        proposed_maximum_gpu_hours=proposed,
        projected_total_gpu_hours=projected_hours,
        projected_gpu_cost_usd=projected_gpu_cost,
        already_accounted_cloud_cost_usd=accounted_cloud_cost,
        proposed_maximum_cloud_cost_usd=proposed_cloud_cost,
        projected_total_cloud_cost_usd=projected_cloud_cost,
    )


def _assert_modal_version() -> None:
    if modal.__version__ != MODAL_SDK_VERSION:
        raise RuntimeError(
            f"Modal SDK must be exactly {MODAL_SDK_VERSION}, got {modal.__version__}"
        )


model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
results_volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)

model_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "huggingface-hub==1.26.0",
        "transformers==5.14.1",
        "tokenizers==0.22.2",
        "safetensors==0.8.0",
        f"pydantic=={PYDANTIC_VERSION}",
        uv_version=UV_VERSION,
    )
    .env(
        {
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "DO_NOT_TRACK": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)

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
if _LOCAL_WORKSPACE_AVAILABLE:
    gpu_image = (
        gpu_image.add_local_dir(
            LOCAL_REPOSITORY_ROOT / "python",
            "/opt/sloforge/python",
            copy=True,
        )
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
    )
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
    tags={"project": "sloforge", "experiment": "branchfabric-gpu-validation-002"},
)
_prepare_model_function: Any = None
_run_gpu_function: Any = None


def _validate_model_manifest(snapshot: Path) -> dict[str, Any]:
    manifest_path = snapshot / "MODEL_MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"model manifest is absent at {manifest_path}")
    manifest = cast(dict[str, Any], json.loads(manifest_path.read_text()))
    if (
        manifest.get("model_id") != MODEL_ID
        or manifest.get("model_revision") != MODEL_REVISION
        or manifest.get("tokenizer_revision") != TOKENIZER_REVISION
    ):
        raise ValueError("model manifest identity does not match the pinned experiment")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("model manifest has no file inventory")
    for item in files:
        path = snapshot / item["relative_path"]
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"model inventory entry is absent or a symlink: {path}")
        if path.stat().st_size != item["bytes"] or _sha256(path) != item["sha256"]:
            raise ValueError(f"model inventory hash mismatch: {path}")
    return manifest


def prepare_model(
    config_payload: dict[str, Any], authorization_payload: dict[str, Any]
) -> dict[str, Any]:
    """Download and hash the pinned model without allocating a GPU."""

    config = ModalBenchmarkConfig.model_validate(config_payload)
    _validate_remote_authorization(config, authorization_payload, kind="model_prepare")
    del config
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer  # type: ignore[import-not-found]

    snapshots = MODEL_MOUNT / "snapshots"
    target = snapshots / MODEL_REVISION
    if target.is_dir():
        manifest = _validate_model_manifest(target)
        return {
            "status": "already_prepared_and_verified",
            "snapshot": str(target),
            "manifest_sha256": _sha256(target / "MODEL_MANIFEST.json"),
            "files": len(manifest["files"]),
        }
    staging = snapshots / f".{MODEL_REVISION}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=False)
    os.environ["HF_HOME"] = str(MODEL_MOUNT / "hf-home")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(MODEL_MOUNT / "hf-home/hub")
    snapshot_download(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=staging,
        token=None,
        max_workers=8,
    )
    hidden_cache = staging / ".cache"
    if hidden_cache.exists():
        shutil.rmtree(hidden_cache)
    tokenizer = AutoTokenizer.from_pretrained(
        staging,
        revision=TOKENIZER_REVISION,
        trust_remote_code=False,
        local_files_only=True,
    )
    repetitions = 256
    while True:
        token_ids = tokenizer.encode(_TOKEN_CORPUS * repetitions, add_special_tokens=False)
        if len(token_ids) >= 16_384:
            break
        repetitions *= 2
        if repetitions > 65_536:
            raise RuntimeError("deterministic token corpus did not reach 16K tokens")
    prefix_ids = token_ids[:16_384]
    special_ids = set(tokenizer.all_special_ids)
    vocabulary_ids = sorted(set(tokenizer.get_vocab().values()) - special_ids)
    divergent_ids = [token for token in vocabulary_ids if token != prefix_ids[-1]][:32]
    if len(divergent_ids) != 32:
        raise RuntimeError("tokenizer did not expose 32 safe divergent token IDs")
    inputs = {
        "schema_version": "sloforge.branchfabric.modal-token-inputs/v1",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
        "construction": "repeat fixed UTF-8 corpus, tokenize without special tokens, truncate",
        "corpus_sha256": hashlib.sha256(_TOKEN_CORPUS.encode()).hexdigest(),
        "corpus_repetitions": repetitions,
        "prefix_token_ids": prefix_ids,
        "divergent_token_ids": divergent_ids,
    }
    _write_json(staging / "BRANCHFABRIC_INPUTS.json", inputs)
    file_records = _inventory(staging)
    manifest = {
        "schema_version": "sloforge.branchfabric.modal-model-manifest/v1",
        "created_at_utc": _utc_now(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
        "files": file_records,
    }
    _write_json(staging / "MODEL_MANIFEST.json", manifest)
    snapshots.mkdir(parents=True, exist_ok=True)
    os.replace(staging, target)
    model_volume.commit()
    return {
        "status": "prepared",
        "snapshot": str(target),
        "manifest_sha256": _sha256(target / "MODEL_MANIFEST.json"),
        "files": len(file_records),
    }


def _run_command(argv: list[str], *, timeout: float = 30.0) -> dict[str, Any]:
    started = time.monotonic_ns()
    completed = subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return {
        "argv": argv,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "duration_ns": time.monotonic_ns() - started,
    }


def _gpu_preflight() -> dict[str, Any]:
    inventory = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    if inventory["returncode"] != 0:
        raise RuntimeError("nvidia-smi GPU inventory failed")
    rows = [row.strip() for row in inventory["stdout"].splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"expected exactly one visible GPU, observed {len(rows)}")
    values = [value.strip() for value in rows[0].split(",")]
    if len(values) != 7:
        raise RuntimeError("nvidia-smi GPU inventory had an unexpected shape")
    gpu = {
        "index": int(values[0]),
        "uuid": values[1],
        "name": values[2],
        "driver_version": values[3],
        "memory_total_mib": int(values[4]),
        "memory_used_mib": int(values[5]),
        "utilization_percent": int(values[6]),
    }
    if "A100" not in gpu["name"] or gpu["memory_total_mib"] < 80_000:
        raise RuntimeError(f"Modal returned unsupported GPU hardware: {gpu}")
    processes = _run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if processes["returncode"] != 0 or processes["stdout"].strip():
        raise RuntimeError("visible GPU was not exclusive before model initialization")
    full = _run_command(["nvidia-smi"])
    return {"gpu": gpu, "inventory": inventory, "processes": processes, "full": full}


def _gpu_postflight(preflight: dict[str, Any]) -> dict[str, Any]:
    inventory = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    processes = _run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    full = _run_command(["nvidia-smi"])
    record = {"inventory": inventory, "processes": processes, "full": full}
    if inventory["returncode"] != 0 or processes["returncode"] != 0:
        raise RuntimeError(f"nvidia-smi cleanup audit failed: {record}")
    rows = [row.strip() for row in inventory["stdout"].splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"cleanup audit expected one visible GPU: {record}")
    values = [value.strip() for value in rows[0].split(",")]
    if len(values) != 7 or values[1] != preflight["gpu"]["uuid"]:
        raise RuntimeError(f"cleanup audit GPU identity changed: {record}")
    memory_used_mib = int(values[5])
    baseline_mib = int(preflight["gpu"]["memory_used_mib"])
    if memory_used_mib > baseline_mib + 1024:
        raise RuntimeError(f"GPU memory did not recover within 1 GiB of preflight: {record}")
    process_rows = [row for row in processes["stdout"].splitlines() if row.strip()]
    unexpected = []
    permitted = []
    for index, row in enumerate(process_rows):
        pieces = [piece.strip() for piece in row.split(",")]
        try:
            process_uuid = pieces[0]
            int(pieces[1])
            process_name = pieces[2].lower()
            process_memory_mib = int(pieces[3])
        except (IndexError, ValueError):
            unexpected.append(row)
            continue
        # NVIDIA reports host PIDs while Python sees the container namespace. Modal's
        # single-use GPU container can attribute its still-live function CUDA context to
        # either Python or PID 1 (/bin/dumb-init). Permit exactly one such bounded context;
        # all child processes were already terminated and aggregate HBM recovery is checked.
        self_process = "python" in process_name or process_name.endswith("dumb-init")
        if (
            index > 0
            or process_uuid != preflight["gpu"]["uuid"]
            or not self_process
            or process_memory_mib > baseline_mib + 1024
        ):
            unexpected.append(row)
        else:
            permitted.append(row)
    if unexpected:
        raise RuntimeError(f"unexpected compute processes survived cleanup: {unexpected}")
    return {
        **record,
        "memory_used_mib": memory_used_mib,
        "preflight_memory_used_mib": baseline_mib,
        "memory_recovered_within_1_gib": True,
        "unexpected_compute_processes": [],
        "permitted_function_container_compute_processes": permitted,
    }


def _dependency_versions() -> dict[str, Any]:
    names = (
        "torch",
        "transformers",
        "vllm",
        "pydantic",
        "psutil",
        "nvidia-ml-py",
        "huggingface-hub",
        "safetensors",
    )
    import torch  # type: ignore[import-not-found]

    return {
        "packages": {name: importlib.metadata.version(name) for name in names},
        # Modal injects its exact runtime SDK into Functions without installing
        # distribution metadata in the user image. The imported module version is
        # already fail-closed by _assert_modal_version().
        "modal_sdk": modal.__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "torch_cuda_userspace": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "runtime_environment": {
            "cc": os.environ.get("CC"),
            "cc_resolved": shutil.which(os.environ.get("CC", "")),
            "vllm_enable_v1_multiprocessing": os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING"),
            "vllm_use_flashinfer_sampler": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER"),
        },
    }


def _build_runner_input(
    config: ModalBenchmarkConfig,
    *,
    gpu_uuid: str,
    snapshot: Path,
) -> tuple[Any, Any, Any]:
    from gpu_validation_runner import (  # pyright: ignore[reportMissingImports]
        ExperimentProfile,
        ExperimentSpecification,
        GpuSamplingConfiguration,
        TrialPath,
        VllmEngineConfiguration,
        VllmGpuValidationInvocation,
    )

    from sloforge.continuum.adapters.real_runtime import RuntimeModelIdentity
    from sloforge.helix.characterization.trace import TraceLevel

    inputs = json.loads((snapshot / "BRANCHFABRIC_INPUTS.json").read_text())
    prefix = tuple(inputs["prefix_token_ids"][: config.prefix_length])
    divergent = tuple(inputs["divergent_token_ids"][: config.fanout])
    if len(prefix) != config.prefix_length or len(divergent) != config.fanout:
        raise ValueError("prepared tokenizer inputs do not cover the requested configuration")
    identity = RuntimeModelIdentity(
        runtime="vllm",
        runtime_version=VLLM_VERSION,
        adapter_version=ADAPTER_VERSION,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        tokenizer_id=MODEL_ID,
        tokenizer_revision=TOKENIZER_REVISION,
        dtype="bfloat16",
        device="cuda:0",
        policy_epoch=POLICY_EPOCH,
    )
    experiment = ExperimentSpecification(
        profile=(
            ExperimentProfile.SMOKE
            if config.prefix_length in {2048, 4096}
            else ExperimentProfile.CONTEXT_16K
        ),
        seed=config.seed,
        timeout_s=float(config.maximum_wall_seconds),
        cleanup_timeout_s=float(config.cleanup_timeout_seconds),
        prefix_token_ids=prefix,
        divergent_token_ids=divergent,
        fanout=config.fanout,
        suffix_tokens=config.suffix_length,
        expected_identity=identity,
        trace_level=TraceLevel(config.tracing_level),
        gpu_sampling=GpuSamplingConfiguration(
            interval_s=config.gpu_sample_interval_seconds,
            command_timeout_s=2.0,
            maximum_samples=min(100_000, config.maximum_wall_seconds * 20),
            physical_device_selector=gpu_uuid,
        ),
    )
    engine = VllmEngineConfiguration(
        execution_model_path=str(snapshot),
        download_dir="/tmp/sloforge-download",
        max_model_len=max(4096, config.prefix_length + config.suffix_length + 2),
        block_size=16,
        max_num_seqs=max(2, config.fanout),
        gpu_memory_utilization=config.gpu_memory_utilization,
        initialization_timeout_s=float(config.initialization_timeout_seconds),
        maximum_trace_events=262_144,
        enforce_eager=False,
        disable_log_stats=True,
        trust_remote_code=False,
        enable_chunked_prefill=True,
    )
    invocation = VllmGpuValidationInvocation(experiment=experiment, engine=engine)
    path = (
        TrialPath.INDEPENDENT_PREFILL
        if config.baseline_mode == "independent"
        else TrialPath.SHARED_ROOT
    )
    return invocation, path, identity


def _terminate_new_children(initial_pids: set[int]) -> list[dict[str, Any]]:
    import psutil  # type: ignore[import-untyped]

    current = psutil.Process()
    children = [
        child for child in current.children(recursive=True) if child.pid not in initial_pids
    ]
    records: list[dict[str, Any]] = []
    for child in children:
        try:
            records.append({"pid": child.pid, "command": child.cmdline(), "action": "terminate"})
            child.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    _, alive = psutil.wait_procs(children, timeout=10.0)
    for child in alive:
        try:
            records.append({"pid": child.pid, "command": child.cmdline(), "action": "kill"})
            child.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    _, still_alive = psutil.wait_procs(alive, timeout=5.0)
    if still_alive:
        raise RuntimeError(
            f"experiment child processes survived cleanup: {[p.pid for p in still_alive]}"
        )
    return records


def _publish_result(
    *,
    attempt_id: str,
    local_root: Path,
    completion: dict[str, Any],
) -> tuple[str, str]:
    remote_prefix = f"{REMOTE_EXPERIMENT_PREFIX}/{attempt_id}"
    final = RESULTS_MOUNT / remote_prefix
    staging = RESULTS_MOUNT / f"{remote_prefix}.staging"
    if final.exists() or staging.exists():
        raise FileExistsError(f"immutable result prefix already exists: {remote_prefix}")
    _fsync_tree(local_root)
    staging.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copytree(local_root, staging)
        _write_json(staging / "function-completion.json", completion)
        manifest = {
            "schema_version": "sloforge.branchfabric.modal-result-manifest/v1",
            "attempt_id": attempt_id,
            "remote_prefix": remote_prefix,
            "created_at_utc": _utc_now(),
            "artifacts": _inventory(staging),
        }
        _write_json(staging / "REMOTE_MANIFEST.json", manifest)
        os.replace(staging, final)
        results_volume.commit()
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        # Persist removal if a background Volume commit exposed the staging tree.
        results_volume.commit()
        raise
    return remote_prefix, _sha256(final / "REMOTE_MANIFEST.json")


def _legacy_in_process_gpu_configuration(
    config_payload: dict[str, Any], authorization_payload: dict[str, Any]
) -> dict[str, Any]:
    """Execute one exact benchmark path and return a bounded result manifest."""

    import psutil
    import torch  # pyright: ignore[reportMissingImports]

    function_body_started_utc = _utc_now()
    function_body_started_ns = time.monotonic_ns()
    call_id = modal.current_function_call_id()
    if not call_id:
        raise RuntimeError("Modal did not expose the current FunctionCall identifier")
    config = ModalBenchmarkConfig.model_validate(config_payload)
    _validate_remote_authorization(config, authorization_payload, kind="gpu")
    work_root = Path("/tmp/sloforge-branchfabric") / config.attempt_id
    work_root.mkdir(parents=True, exist_ok=False)
    wrapper = work_root / "wrapper"
    wrapper.mkdir()
    os.environ.update(
        {
            "XDG_CACHE_HOME": str(work_root / "cache"),
            "HF_HOME": str(work_root / "hf"),
            "HUGGINGFACE_HUB_CACHE": str(work_root / "hf/hub"),
            "TRANSFORMERS_CACHE": str(work_root / "hf/transformers"),
            "TORCH_HOME": str(work_root / "torch"),
            "TRITON_CACHE_DIR": str(work_root / "triton"),
            "TMPDIR": str(work_root / "tmp"),
        }
    )
    for name in ("cache", "hf", "torch", "triton", "tmp"):
        (work_root / name).mkdir(exist_ok=True)
    initial_children = {child.pid for child in psutil.Process().children(recursive=True)}
    status = "failed"
    error_record: dict[str, Any] | None = None
    model_ready_utc: str | None = None
    engine_initialization_started_utc: str | None = None
    benchmark_started_utc: str | None = None
    benchmark_ended_utc: str | None = None
    model_ready_ns: int | None = None
    engine_initialization_started_ns: int | None = None
    benchmark_started_ns: int | None = None
    benchmark_ended_ns: int | None = None
    preflight: dict[str, Any] | None = None
    summary_payload: dict[str, Any] | None = None
    adapter: Any = None
    try:
        _assert_modal_version()
        preflight = _gpu_preflight()
        versions = _dependency_versions()
        packages = versions["packages"]
        runtime_environment = versions["runtime_environment"]
        if (
            packages["vllm"] != VLLM_VERSION
            or packages["torch"] != TORCH_VERSION
            or packages["transformers"] != TRANSFORMERS_VERSION
            or versions["torch_cuda_userspace"] != "13.0"
            or not versions["cuda_available"]
            or runtime_environment["cc"] != "/usr/bin/gcc"
            or runtime_environment["cc_resolved"] != "/usr/bin/gcc"
            or runtime_environment["vllm_enable_v1_multiprocessing"] != "0"
            or runtime_environment["vllm_use_flashinfer_sampler"] != "0"
        ):
            raise RuntimeError(f"runtime dependency pin mismatch: {versions}")
        if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability(0) != (8, 0):
            raise RuntimeError("Modal GPU is not one visible sm80 device")
        if shutil.disk_usage(work_root).free < 20 * 1024**3:
            raise RuntimeError("Modal ephemeral disk has less than 20 GiB free")
        snapshot = MODEL_MOUNT / "snapshots" / MODEL_REVISION
        model_manifest_path = snapshot / "MODEL_MANIFEST.json"
        _validate_model_manifest(snapshot)
        gpu = preflight["gpu"]
        invocation, path, _ = _build_runner_input(
            config,
            gpu_uuid=gpu["uuid"],
            snapshot=snapshot,
        )
        _write_json(wrapper / "modal-config.json", config)
        _write_json(wrapper / "invocation.json", invocation)
        _write_json(wrapper / "preflight.json", preflight)
        _write_json(wrapper / "dependency-versions.json", versions)
        _write_json(
            wrapper / "model-manifest-reference.json",
            {
                "path": str(model_manifest_path),
                "sha256": _sha256(model_manifest_path),
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "tokenizer_revision": TOKENIZER_REVISION,
            },
        )
        from gpu_validation_runner import (  # pyright: ignore[reportMissingImports]
            build_vllm_adapter_factory,
            run_independent_trial,
            run_shared_trial,
        )

        factory = build_vllm_adapter_factory(invocation)
        engine_initialization_started_ns = time.monotonic_ns()
        engine_initialization_started_utc = _utc_now()
        adapter = factory(path)
        model_ready_ns = time.monotonic_ns()
        model_ready_utc = _utc_now()
        benchmark_started_ns = time.monotonic_ns()
        benchmark_started_utc = _utc_now()
        trial_output = work_root / "runner" / path.value
        if config.baseline_mode == "independent":
            summary = run_independent_trial(
                adapter,
                invocation.experiment,
                trial_output,
            )
        else:
            summary = run_shared_trial(
                adapter,
                invocation.experiment,
                trial_output,
            )
        adapter = None
        benchmark_ended_ns = time.monotonic_ns()
        benchmark_ended_utc = _utc_now()
        summary_payload = summary.model_dump(mode="json")
        _write_json(wrapper / "trial-summary.json", summary_payload)
        status = "succeeded"
    except Exception as error:
        error_record = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        _write_json(wrapper / "failure.json", error_record)
    finally:
        if adapter is not None:
            try:
                adapter.cleanup_runtime(timeout_s=float(config.cleanup_timeout_seconds))
            except Exception as cleanup_error:
                if error_record is None:
                    error_record = {
                        "type": type(cleanup_error).__name__,
                        "message": str(cleanup_error),
                        "traceback": traceback.format_exc(),
                    }
                    status = "failed"
                _write_json(
                    wrapper / "cleanup-failure.json",
                    {
                        "type": type(cleanup_error).__name__,
                        "message": str(cleanup_error),
                        "traceback": traceback.format_exc(),
                    },
                )
        try:
            child_actions = _terminate_new_children(initial_children)
        except Exception as cleanup_error:
            child_actions = [
                {
                    "action": "cleanup_failed",
                    "type": type(cleanup_error).__name__,
                    "message": str(cleanup_error),
                }
            ]
            status = "failed"
            if error_record is None:
                error_record = {
                    "type": type(cleanup_error).__name__,
                    "message": str(cleanup_error),
                    "traceback": traceback.format_exc(),
                }
        try:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception as cuda_cleanup_error:
            child_actions.append(
                {
                    "action": "cuda_allocator_cleanup_failed",
                    "type": type(cuda_cleanup_error).__name__,
                    "message": str(cuda_cleanup_error),
                }
            )
            status = "failed"
        try:
            postflight = (
                _run_command(["nvidia-smi"]) if preflight is None else _gpu_postflight(preflight)
            )
        except Exception as postflight_error:
            postflight = {
                "error_type": type(postflight_error).__name__,
                "error_message": str(postflight_error),
            }
            status = "failed"
        _write_json(wrapper / "postflight.json", postflight)
        _write_json(wrapper / "child-process-cleanup.json", child_actions)
    function_end_ns = time.monotonic_ns()
    function_end_utc = _utc_now()
    completion = {
        "schema_version": "sloforge.branchfabric.modal-function-completion/v1",
        "status": status,
        "function_call_id": call_id,
        "attempt_id": config.attempt_id,
        "gpu_count": GPU_COUNT,
        "requested_gpu": GPU_SKU,
        "actual_gpu": None if preflight is None else preflight["gpu"],
        "container_import_utc": _CONTAINER_IMPORT_UTC,
        "function_start_utc": function_body_started_utc,
        "model_ready_utc": model_ready_utc,
        "engine_initialization_started_utc": engine_initialization_started_utc,
        "benchmark_start_utc": benchmark_started_utc,
        "benchmark_end_utc": benchmark_ended_utc,
        "function_end_utc": function_end_utc,
        "gpu_allocation_lower_bound_seconds": (function_end_ns - _CONTAINER_IMPORT_MONOTONIC_NS)
        / 1_000_000_000,
        "function_body_seconds": (function_end_ns - function_body_started_ns) / 1_000_000_000,
        "model_load_and_warmup_seconds": (
            None
            if model_ready_ns is None or engine_initialization_started_ns is None
            else (model_ready_ns - engine_initialization_started_ns) / 1_000_000_000
        ),
        "preflight_validation_and_engine_ready_seconds": (
            None
            if model_ready_ns is None
            else (model_ready_ns - function_body_started_ns) / 1_000_000_000
        ),
        "model_load_and_warmup_source": (
            "vLLM engine initialization, memory profiling, dummy model execution, "
            "and CUDA graph capture before the adapter factory returns"
        ),
        "benchmark_seconds": (
            None
            if benchmark_started_ns is None or benchmark_ended_ns is None
            else (benchmark_ended_ns - benchmark_started_ns) / 1_000_000_000
        ),
        "summary": summary_payload,
        "error": error_record,
        "function_end_measurement": "immediately before immutable result publication and return",
    }
    remote_prefix, manifest_sha256 = _publish_result(
        attempt_id=config.attempt_id,
        local_root=work_root,
        completion=completion,
    )
    return {
        **completion,
        "remote_prefix": remote_prefix,
        "remote_manifest_sha256": manifest_sha256,
    }


def run_gpu_configuration(
    config_payload: dict[str, Any], authorization_payload: dict[str, Any]
) -> dict[str, Any]:
    """Run one trial behind a CUDA-clean parent/fresh-worker process boundary."""

    function_body_started_utc = _utc_now()
    function_body_started_ns = time.monotonic_ns()
    call_id = modal.current_function_call_id()
    if not call_id:
        raise RuntimeError("Modal did not expose the current FunctionCall identifier")
    config = ModalBenchmarkConfig.model_validate(config_payload)
    _validate_remote_authorization(config, authorization_payload, kind="gpu")
    work_root = Path("/tmp/sloforge-branchfabric") / config.attempt_id
    work_root.mkdir(parents=True, exist_ok=False)
    snapshot = MODEL_MOUNT / "snapshots" / MODEL_REVISION
    _validate_model_manifest(snapshot)

    # This is the only experiment module imported by the Modal GPU Function body.
    # gpu_cow_controller is standard-library-only and audits sys.modules before
    # taking the first driver-visible baseline and immediately before Popen.
    from gpu_cow_controller import run_controller  # pyright: ignore[reportMissingImports]

    worker_path = Path("/opt/sloforge/experiments/branchfabric/gpu_cow_worker.py")
    if not worker_path.is_file():
        raise FileNotFoundError(f"GPU worker source is absent from the Modal image: {worker_path}")
    controller_result = run_controller(
        config_payload=config.model_dump(mode="json"),
        work_root=work_root,
        worker_path=worker_path,
        model_snapshot=snapshot,
    )
    child_manifest = controller_result.get("child_manifest")
    if not isinstance(child_manifest, dict):
        child_manifest = {}
    cleanup_gate = controller_result["cleanup_gate"]
    # Retain a backwards-compatible wrapper postflight while making the lifecycle
    # gate the authoritative source for v6 and later attempts.
    _write_json(
        work_root / "wrapper/postflight.json",
        {
            "schema_version": "sloforge.branchfabric.modal-postflight/v2",
            "memory_recovered_within_1_gib": (
                cleanup_gate.get("process_hbm_recovery") == "verified"
            ),
            "unexpected_compute_processes": (
                []
                if cleanup_gate.get("no_compute_process_remained_in_winning_samples") is True
                else ["driver-visible cleanup gate did not obtain three process-free samples"]
            ),
            "cleanup_gate": cleanup_gate.get("cleanup_gate"),
            "GPU_RUNTIME_LIFECYCLE_CLEAN": cleanup_gate.get("GPU_RUNTIME_LIFECYCLE_CLEAN", False),
            "authoritative_artifact": "lifecycle/cleanup-gate.json",
        },
    )
    function_end_ns = time.monotonic_ns()
    function_end_utc = _utc_now()
    completion = {
        "schema_version": "sloforge.branchfabric.modal-function-completion/v1",
        "status": controller_result["status"],
        "function_call_id": call_id,
        "attempt_id": config.attempt_id,
        "gpu_count": GPU_COUNT,
        "requested_gpu": GPU_SKU,
        "actual_gpu": controller_result.get("actual_gpu"),
        "container_import_utc": _CONTAINER_IMPORT_UTC,
        "function_start_utc": function_body_started_utc,
        "model_ready_utc": child_manifest.get("child_end_utc"),
        "engine_initialization_started_utc": child_manifest.get("child_start_utc"),
        "benchmark_start_utc": None,
        "benchmark_end_utc": None,
        "function_end_utc": function_end_utc,
        "gpu_allocation_lower_bound_seconds": (function_end_ns - _CONTAINER_IMPORT_MONOTONIC_NS)
        / 1_000_000_000,
        "function_body_seconds": (function_end_ns - function_body_started_ns) / 1_000_000_000,
        "model_load_and_warmup_seconds": child_manifest.get("model_load_and_warmup_seconds"),
        "preflight_validation_and_engine_ready_seconds": None,
        "model_load_and_warmup_source": (
            "fresh child vLLM engine initialization, memory profiling, dummy execution, "
            "and CUDA graph capture"
        ),
        "benchmark_seconds": child_manifest.get("benchmark_seconds"),
        "summary": controller_result.get("summary"),
        "error": controller_result.get("error"),
        "GPU_RUNTIME_LIFECYCLE_CLEAN": cleanup_gate.get("GPU_RUNTIME_LIFECYCLE_CLEAN", False),
        "time_to_hbm_recovery_ms": cleanup_gate.get("time_to_hbm_recovery_ms"),
        "child_pid": child_manifest.get("child_pid"),
        "child_pgid": child_manifest.get("child_pgid"),
        "child_return_code": controller_result["controller_manifest"].get("child_return_code"),
        "forced_gpu_process_kill_required": cleanup_gate.get("forced_gpu_process_kill_required"),
        "function_end_measurement": (
            "after exact child/descendant termination and stable driver-visible HBM recovery, "
            "immediately before immutable publication"
        ),
    }
    remote_prefix, manifest_sha256 = _publish_result(
        attempt_id=config.attempt_id,
        local_root=work_root,
        completion=completion,
    )
    return {
        **completion,
        "remote_prefix": remote_prefix,
        "remote_manifest_sha256": manifest_sha256,
    }


def _persist_call_state(
    *, attempt_id: str, kind: str, call_id: str, config: ModalBenchmarkConfig
) -> Path:
    filename = (
        f"{attempt_id}-gpu.json" if kind == "gpu" else f"{attempt_id}-model-prepare-{call_id}.json"
    )
    path = LOCAL_EXPERIMENT_ROOT / "modal/calls" / filename
    if path.exists():
        raise FileExistsError(f"call-state record already exists: {path}")
    _write_json(
        path,
        {
            "schema_version": "sloforge.branchfabric.modal-call-state/v1",
            "recorded_at_utc": _utc_now(),
            "kind": kind,
            "attempt_id": attempt_id,
            "function_call_id": call_id,
            "config": config.model_dump(mode="json"),
        },
    )
    return path


def _materialize_result(
    attempt_id: str, *, expected_manifest_sha256: str | None = None
) -> dict[str, Any]:
    remote_prefix = f"{REMOTE_EXPERIMENT_PREFIX}/{attempt_id}"
    target = LOCAL_EXPERIMENT_ROOT / "modal" / attempt_id
    staging = target.with_name(f".{attempt_id}.staging")
    if target.exists():
        raise FileExistsError(f"local immutable result path already exists: {target}")
    if staging.exists():
        failed_root = LOCAL_EXPERIMENT_ROOT / "modal/failed-materializations"
        failed_root.mkdir(parents=True, exist_ok=True)
        archived = failed_root / f"{attempt_id}-{uuid.uuid4().hex}"
        os.replace(staging, archived)
    entries = list(results_volume.iterdir(remote_prefix, recursive=True))
    if not entries:
        raise FileNotFoundError(f"Modal result prefix is empty: {remote_prefix}")
    staging.mkdir(parents=True, exist_ok=False)
    downloaded: list[dict[str, Any]] = []
    prefix_path = PurePosixPath(remote_prefix)
    for entry in entries:
        entry_type = getattr(entry.type, "name", str(entry.type))
        remote_path = PurePosixPath(entry.path.lstrip("/"))
        try:
            relative = remote_path.relative_to(prefix_path)
        except ValueError as error:
            raise RuntimeError(
                f"Volume returned path outside result prefix: {entry.path}"
            ) from error
        local_path = staging / Path(relative.as_posix())
        if entry_type == "DIRECTORY":
            local_path.mkdir(parents=True, exist_ok=True)
            continue
        if entry_type != "FILE":
            raise RuntimeError(f"unsupported Volume result entry type: {entry_type}")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        size = 0
        with local_path.open("xb") as handle:
            for chunk in results_volume.read_file(entry.path):
                handle.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        downloaded.append(
            {
                "relative_path": relative.as_posix(),
                "bytes": size,
                "sha256": digest.hexdigest(),
            }
        )
    manifest_path = staging / "REMOTE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_sha256 = _sha256(manifest_path)
    if expected_manifest_sha256 is not None and manifest_sha256 != expected_manifest_sha256:
        raise RuntimeError("Modal result manifest does not match the Function return value")
    if (
        manifest.get("schema_version") != "sloforge.branchfabric.modal-result-manifest/v1"
        or manifest.get("attempt_id") != attempt_id
        or manifest.get("remote_prefix") != remote_prefix
    ):
        raise RuntimeError("Modal result manifest identity is invalid")
    expected = {item["relative_path"]: item for item in manifest["artifacts"]}
    observed = {item["relative_path"]: item for item in downloaded}
    observed.pop("REMOTE_MANIFEST.json", None)
    if observed != expected:
        raise RuntimeError("downloaded Modal results do not match REMOTE_MANIFEST.json")
    verification = {
        "schema_version": "sloforge.branchfabric.modal-local-copy-verification/v1",
        "verified_at_utc": _utc_now(),
        "attempt_id": attempt_id,
        "remote_prefix": remote_prefix,
        "remote_manifest_sha256": manifest_sha256,
        "artifacts": sorted(downloaded, key=lambda item: item["relative_path"]),
    }
    _write_json(staging / "LOCAL_COPY_VERIFICATION.json", verification)
    os.replace(staging, target)
    return verification


def _load_gpu_ledger() -> dict[str, Any]:
    ledger_path = LOCAL_EXPERIMENT_ROOT / "gpu-hours.json"
    return cast(
        dict[str, Any],
        json.loads(ledger_path.read_text())
        if ledger_path.is_file()
        else {
            "schema_version": "sloforge.branchfabric.gpu-hours/v1",
            "experiment": "experiment-002",
            "hard_limit_gpu_hours": GPU_HOUR_LIMIT,
            "target_gpu_hours": GPU_HOUR_TARGET,
            "consumed_gpu_hours": 0.0,
            "gpu_active_intervals": [],
            "consumed_cloud_cost_usd": 0.0,
            "cloud_compute_intervals": [],
            "modal_in_flight_reservations": [],
            "modal_in_flight_model_reservations": [],
        },
    )


def _reserve_model_call(config: ModalBenchmarkConfig, decision: CloudBudgetDecision) -> str:
    ledger_path = LOCAL_EXPERIMENT_ROOT / "gpu-hours.json"
    ledger = _load_gpu_ledger()
    reservations = ledger.setdefault("modal_in_flight_model_reservations", [])
    gpu_reservations = ledger.get("modal_in_flight_reservations", [])
    if (
        not isinstance(reservations, list)
        or not isinstance(gpu_reservations, list)
        or reservations
        or gpu_reservations
    ):
        raise RuntimeError("another Modal model or GPU reservation is active")
    reservation_id = f"modal-model-reservation-{uuid.uuid4().hex}"
    reservations.append(
        {
            "reservation_id": reservation_id,
            "attempt_id": config.attempt_id,
            "created_at_utc": _utc_now(),
            "maximum_cloud_cost_usd": decision.proposed_maximum_cloud_cost_usd,
            "maximum_remote_seconds": (
                MODEL_FUNCTION_TIMEOUT_SECONDS + MODEL_FUNCTION_STARTUP_TIMEOUT_SECONDS
            ),
            "config_sha256": hashlib.sha256(_canonical_bytes(config)).hexdigest(),
            "function_call_id": None,
        }
    )
    ledger["updated_at"] = _utc_now()
    _replace_json(ledger_path, ledger)
    return reservation_id


def _attach_model_call_to_reservation(reservation_id: str, call_id: str) -> None:
    ledger_path = LOCAL_EXPERIMENT_ROOT / "gpu-hours.json"
    ledger = _load_gpu_ledger()
    reservations = ledger.get("modal_in_flight_model_reservations", [])
    matches = [item for item in reservations if item.get("reservation_id") == reservation_id]
    if len(matches) != 1 or matches[0].get("function_call_id") is not None:
        raise RuntimeError("model reservation is absent, duplicated, or already attached")
    matches[0]["function_call_id"] = call_id
    matches[0]["attached_at_utc"] = _utc_now()
    ledger["updated_at"] = _utc_now()
    _replace_json(ledger_path, ledger)


def _reconcile_model_reservation(
    *,
    reservation_id: str,
    call_id: str,
    attempt_id: str,
    elapsed_seconds: float,
    status: str,
    maximum_charge: bool,
    provider_terminal_audit_sha256: str | None = None,
) -> None:
    ledger_path = LOCAL_EXPERIMENT_ROOT / "gpu-hours.json"
    ledger = _load_gpu_ledger()
    reservations = ledger.get("modal_in_flight_model_reservations", [])
    matches = [item for item in reservations if item.get("reservation_id") == reservation_id]
    if len(matches) != 1:
        raise RuntimeError("model reservation is absent or duplicated")
    intervals = ledger.setdefault("cloud_compute_intervals", [])
    if not isinstance(intervals, list):
        raise ValueError("cloud_compute_intervals must be a list")
    if not any(item.get("function_call_id") == call_id for item in intervals):
        reservation = matches[0]
        per_second = (
            MODEL_FUNCTION_CPU_CORES * MODAL_CPU_USD_PER_CORE_SECOND
            + MODEL_FUNCTION_MEMORY_GIB * MODAL_MEMORY_USD_PER_GIB_SECOND
        )
        intervals.append(
            {
                "backend": "modal",
                "kind": "model_prepare",
                "function_call_id": call_id,
                "attempt_id": attempt_id,
                "reservation_id": reservation_id,
                "reservation_created_at_utc": reservation.get("created_at_utc"),
                "reserved_maximum_cloud_cost_usd": reservation.get("maximum_cloud_cost_usd"),
                "reserved_maximum_remote_seconds": reservation.get("maximum_remote_seconds"),
                "authorization_config_sha256": reservation.get("config_sha256"),
                "provider_terminal_audit_sha256": provider_terminal_audit_sha256,
                "elapsed_seconds": elapsed_seconds,
                "cost_upper_bound_usd": elapsed_seconds * per_second,
                "maximum_charge": maximum_charge,
                "status": status,
                "recorded_at_utc": _utc_now(),
            }
        )
    ledger["modal_in_flight_model_reservations"] = [
        item for item in reservations if item.get("reservation_id") != reservation_id
    ]
    ledger["consumed_cloud_cost_usd"] = sum(
        float(item["cost_upper_bound_usd"]) for item in intervals
    )
    ledger["updated_at"] = _utc_now()
    _replace_json(ledger_path, ledger)


def _reserve_gpu_call(config: ModalBenchmarkConfig, decision: CloudBudgetDecision) -> str:
    ledger_path = LOCAL_EXPERIMENT_ROOT / "gpu-hours.json"
    ledger = _load_gpu_ledger()
    reservations = ledger.setdefault("modal_in_flight_reservations", [])
    model_reservations = ledger.get("modal_in_flight_model_reservations", [])
    if (
        not isinstance(reservations, list)
        or not isinstance(model_reservations, list)
        or reservations
        or model_reservations
    ):
        raise RuntimeError("another Modal model or GPU reservation is active or ledger is invalid")
    maximum_seconds = GPU_FUNCTION_TIMEOUT_SECONDS + GPU_FUNCTION_STARTUP_TIMEOUT_SECONDS
    maximum_gpu_hours = maximum_seconds * GPU_COUNT / 3600.0
    consumed = float(ledger.get("consumed_gpu_hours", 0.0))
    if consumed + maximum_gpu_hours > GPU_HOUR_LIMIT:
        raise RuntimeError("Modal GPU reservation would exceed the four GPU-hour hard limit")
    reservation_id = f"modal-reservation-{uuid.uuid4().hex}"
    reservations.append(
        {
            "reservation_id": reservation_id,
            "attempt_id": config.attempt_id,
            "created_at_utc": _utc_now(),
            "maximum_gpu_seconds": maximum_seconds,
            "maximum_gpu_hours": maximum_gpu_hours,
            "maximum_cloud_cost_usd": decision.proposed_maximum_cloud_cost_usd,
            "maximum_remote_seconds": maximum_seconds,
            "config_sha256": hashlib.sha256(_canonical_bytes(config)).hexdigest(),
            "function_call_id": None,
        }
    )
    ledger["updated_at"] = _utc_now()
    _replace_json(ledger_path, ledger)
    return reservation_id


def _attach_call_to_reservation(reservation_id: str, call_id: str) -> None:
    ledger_path = LOCAL_EXPERIMENT_ROOT / "gpu-hours.json"
    ledger = _load_gpu_ledger()
    reservations = ledger.get("modal_in_flight_reservations", [])
    matches = [item for item in reservations if item.get("reservation_id") == reservation_id]
    if len(matches) != 1 or matches[0].get("function_call_id") is not None:
        raise RuntimeError("Modal reservation is absent, duplicated, or already attached")
    matches[0]["function_call_id"] = call_id
    matches[0]["attached_at_utc"] = _utc_now()
    ledger["updated_at"] = _utc_now()
    _replace_json(ledger_path, ledger)


def _record_cloud_interval(
    *,
    kind: str,
    call_id: str,
    attempt_id: str,
    elapsed_seconds: float,
    status: str,
    maximum_charge: bool,
) -> None:
    ledger_path = LOCAL_EXPERIMENT_ROOT / "gpu-hours.json"
    ledger = _load_gpu_ledger()
    intervals = ledger.setdefault("cloud_compute_intervals", [])
    if not isinstance(intervals, list):
        raise ValueError("cloud_compute_intervals must be a list")
    if any(item.get("function_call_id") == call_id for item in intervals):
        return
    if kind == "model_prepare":
        per_second = (
            MODEL_FUNCTION_CPU_CORES * MODAL_CPU_USD_PER_CORE_SECOND
            + MODEL_FUNCTION_MEMORY_GIB * MODAL_MEMORY_USD_PER_GIB_SECOND
        )
    else:
        per_second = (
            MODAL_A100_80GB_USD_PER_HOUR / 3600.0
            + GPU_FUNCTION_CPU_CORES * MODAL_CPU_USD_PER_CORE_SECOND
            + GPU_FUNCTION_MEMORY_GIB * MODAL_MEMORY_USD_PER_GIB_SECOND
        )
    cost = elapsed_seconds * per_second
    intervals.append(
        {
            "backend": "modal",
            "kind": kind,
            "function_call_id": call_id,
            "attempt_id": attempt_id,
            "elapsed_seconds": elapsed_seconds,
            "cost_upper_bound_usd": cost,
            "maximum_charge": maximum_charge,
            "status": status,
            "recorded_at_utc": _utc_now(),
        }
    )
    ledger["consumed_cloud_cost_usd"] = sum(
        float(item["cost_upper_bound_usd"]) for item in intervals
    )
    ledger["updated_at"] = _utc_now()
    _replace_json(ledger_path, ledger)


def _record_gpu_hours(
    result: dict[str, Any], *, reservation_id: str, accounted_seconds: float
) -> None:
    ledger_path = LOCAL_EXPERIMENT_ROOT / "gpu-hours.json"
    ledger = _load_gpu_ledger()
    existing_intervals = ledger.get("gpu_active_intervals", [])
    reservations = ledger.get("modal_in_flight_reservations", [])
    if not isinstance(existing_intervals, list) or not isinstance(reservations, list):
        raise ValueError("GPU-hour ledger intervals/reservations must be lists")
    matches = [item for item in reservations if item.get("reservation_id") == reservation_id]
    if len(matches) != 1:
        raise RuntimeError("Modal GPU reservation is absent or duplicated")
    if any(
        item.get("function_call_id") == result["function_call_id"] for item in existing_intervals
    ):
        return
    reservation = matches[0]
    seconds = max(float(result["gpu_allocation_lower_bound_seconds"]), accounted_seconds)
    interval = {
        "backend": "modal",
        "function_call_id": result["function_call_id"],
        "attempt_id": result["attempt_id"],
        "reservation_id": reservation_id,
        "reservation_created_at_utc": reservation.get("created_at_utc"),
        "reserved_maximum_gpu_seconds": reservation.get("maximum_gpu_seconds"),
        "reserved_maximum_gpu_hours": reservation.get("maximum_gpu_hours"),
        "reserved_maximum_cloud_cost_usd": reservation.get("maximum_cloud_cost_usd"),
        "reserved_maximum_remote_seconds": reservation.get("maximum_remote_seconds"),
        "authorization_config_sha256": reservation.get("config_sha256"),
        "provider_terminal_audit_sha256": result.get("provider_terminal_audit_sha256"),
        "gpu_count": result["gpu_count"],
        "requested_gpu": result["requested_gpu"],
        "actual_gpu": result["actual_gpu"],
        "start_utc": result["container_import_utc"],
        "end_utc": result["function_end_utc"],
        "seconds": seconds,
        "gpu_hours": seconds * int(result["gpu_count"]) / 3600.0,
        "status": result["status"],
        "remote_manifest_sha256": result.get("remote_manifest_sha256"),
        "accounting": "upper bound from client elapsed, lower-bounded by container observation",
    }
    intervals = [*existing_intervals, interval]
    consumed = sum(float(item["gpu_hours"]) for item in intervals)
    if consumed > GPU_HOUR_LIMIT:
        raise RuntimeError("Modal result would exceed the four GPU-hour ledger limit")
    cloud_intervals = ledger.setdefault("cloud_compute_intervals", [])
    if not isinstance(cloud_intervals, list):
        raise ValueError("cloud_compute_intervals must be a list")
    if not any(
        item.get("function_call_id") == result["function_call_id"] for item in cloud_intervals
    ):
        per_second = (
            MODAL_A100_80GB_USD_PER_HOUR / 3600.0
            + GPU_FUNCTION_CPU_CORES * MODAL_CPU_USD_PER_CORE_SECOND
            + GPU_FUNCTION_MEMORY_GIB * MODAL_MEMORY_USD_PER_GIB_SECOND
        )
        cloud_intervals.append(
            {
                "backend": "modal",
                "kind": "gpu",
                "function_call_id": result["function_call_id"],
                "attempt_id": result["attempt_id"],
                "reservation_id": reservation_id,
                "reservation_created_at_utc": reservation.get("created_at_utc"),
                "reserved_maximum_cloud_cost_usd": reservation.get("maximum_cloud_cost_usd"),
                "reserved_maximum_remote_seconds": reservation.get("maximum_remote_seconds"),
                "authorization_config_sha256": reservation.get("config_sha256"),
                "provider_terminal_audit_sha256": result.get("provider_terminal_audit_sha256"),
                "elapsed_seconds": seconds,
                "cost_upper_bound_usd": seconds * per_second,
                "maximum_charge": seconds
                == GPU_FUNCTION_TIMEOUT_SECONDS + GPU_FUNCTION_STARTUP_TIMEOUT_SECONDS,
                "status": result["status"],
                "recorded_at_utc": _utc_now(),
            }
        )
    ledger.update(
        {
            "consumed_gpu_hours": consumed,
            "gpu_active_intervals": intervals,
            "modal_in_flight_reservations": [
                item for item in reservations if item.get("reservation_id") != reservation_id
            ],
            "consumed_cloud_cost_usd": sum(
                float(item["cost_upper_bound_usd"]) for item in cloud_intervals
            ),
            "stop_reason": None,
            "updated_at": _utc_now(),
        }
    )
    _replace_json(ledger_path, ledger)


def _charge_failed_gpu_reservation(
    *,
    reservation_id: str,
    call_id: str,
    attempt_id: str,
    reason: str,
    remote_manifest_sha256: str | None = None,
    provider_terminal_audit_sha256: str | None = None,
) -> None:
    maximum_seconds = GPU_FUNCTION_TIMEOUT_SECONDS + GPU_FUNCTION_STARTUP_TIMEOUT_SECONDS
    result = {
        "function_call_id": call_id,
        "attempt_id": attempt_id,
        "gpu_count": GPU_COUNT,
        "requested_gpu": GPU_SKU,
        "actual_gpu": None,
        "container_import_utc": None,
        "function_end_utc": _utc_now(),
        "gpu_allocation_lower_bound_seconds": 0.0,
        "status": f"failed_conservative_charge:{reason}",
        "remote_manifest_sha256": remote_manifest_sha256,
        "provider_terminal_audit_sha256": provider_terminal_audit_sha256,
    }
    _record_gpu_hours(
        result,
        reservation_id=reservation_id,
        accounted_seconds=float(maximum_seconds),
    )


def _load_config(path: str) -> ModalBenchmarkConfig:
    if not path:
        raise ValueError("--config-path is required")
    config_path = Path(path).resolve(strict=True)
    if config_path.stat().st_size > 1024 * 1024:
        raise ValueError("Modal benchmark config exceeds 1 MiB")
    return ModalBenchmarkConfig.model_validate_json(config_path.read_bytes())


def _successful_local_completions() -> list[dict[str, Any]]:
    root = LOCAL_EXPERIMENT_ROOT / "modal"
    completions: list[dict[str, Any]] = []
    if not root.is_dir():
        return completions
    for path in sorted(root.glob("*/function-completion.json")):
        payload = json.loads(path.read_text())
        attempt_root = path.parent
        cleanup_path = attempt_root / "lifecycle/cleanup-gate.json"
        import_audit_path = attempt_root / "lifecycle/forbidden-import-audit.json"
        child_manifest_path = attempt_root / "lifecycle/child-manifest.json"
        if not (
            payload.get("status") == "succeeded"
            and isinstance(payload.get("summary"), dict)
            and cleanup_path.is_file()
            and import_audit_path.is_file()
            and child_manifest_path.is_file()
        ):
            continue
        cleanup = json.loads(cleanup_path.read_text())
        import_audit = json.loads(import_audit_path.read_text())
        child_manifest = json.loads(child_manifest_path.read_text())
        invariants = child_manifest.get("semantic_invariants")
        if (
            cleanup.get("cleanup_gate") == "PASS"
            and cleanup.get("GPU_RUNTIME_LIFECYCLE_CLEAN") is True
            and cleanup.get("forced_gpu_process_kill_required") is False
            and import_audit.get("cuda_clean") is True
            and child_manifest.get("status") == "succeeded"
            and child_manifest.get("final_runtime_assigned_kv_bytes") == 0
            and isinstance(invariants, dict)
            and bool(invariants)
            and all(value is True for value in invariants.values())
        ):
            completions.append(payload)
    return completions


def _assert_adaptive_gate(config: ModalBenchmarkConfig) -> None:
    """Require the hard smoke -> fanout 1 -> fanout 8 -> fanout 32 order."""

    if config.prefix_length in {2048, 4096}:
        return
    completions = _successful_local_completions()
    summaries = [cast(dict[str, Any], item["summary"]) for item in completions]
    smoke_passed = any(
        summary.get("path") == "shared_root"
        and summary.get("fanout") == 2
        and summary.get("prefix_tokens") == 2048
        and summary.get("seed") == config.seed
        for summary in summaries
    )
    if not smoke_passed:
        raise RuntimeError("16K Modal execution requires a successful same-seed semantic smoke")
    if config.fanout == 1:
        return
    fanout_one_paths = {
        summary.get("path")
        for summary in summaries
        if summary.get("fanout") == 1
        and summary.get("prefix_tokens") == 16_384
        and summary.get("seed") == config.seed
    }
    if fanout_one_paths != {"independent_prefill", "shared_root"}:
        raise RuntimeError("fanout 8 requires successful same-seed fanout-1 paired paths")
    if config.fanout == 8:
        return
    fanout_eight_paths = {
        summary.get("path")
        for summary in summaries
        if summary.get("fanout") == 8
        and summary.get("prefix_tokens") == 16_384
        and summary.get("seed") == config.seed
    }
    if fanout_eight_paths != {"independent_prefill", "shared_root"}:
        raise RuntimeError("fanout 32 requires successful same-seed fanout-8 paired paths")
    approval = LOCAL_EXPERIMENT_ROOT / "analysis" / f"fanout-32-approval-seed-{config.seed}.json"
    if not approval.is_file():
        raise RuntimeError("fanout 32 requires an explicit post-fanout-8 HBM/headroom approval")
    payload = json.loads(approval.read_text())
    if (
        payload.get("approved") is not True
        or payload.get("hbm_headroom_safe") is not True
        or payload.get("adapter_invariants_valid") is not True
        or payload.get("gpu_hour_budget_safe") is not True
    ):
        raise RuntimeError("fanout-32 approval does not satisfy every adaptive safety gate")
    evidence = payload.get("fanout_8_evidence")
    if not isinstance(evidence, list) or len(evidence) != 2:
        raise RuntimeError("fanout-32 approval must bind exactly two fanout-8 result manifests")
    evidence_paths: set[str] = set()
    for item in evidence:
        if not isinstance(item, dict):
            raise RuntimeError("fanout-32 evidence entries must be objects")
        attempt_id = item.get("attempt_id")
        expected_hash = item.get("remote_manifest_sha256")
        if not isinstance(attempt_id, str) or not isinstance(expected_hash, str):
            raise RuntimeError("fanout-32 evidence identity/hash is invalid")
        result_root = LOCAL_EXPERIMENT_ROOT / "modal" / attempt_id
        completion = json.loads((result_root / "function-completion.json").read_text())
        verification = json.loads((result_root / "LOCAL_COPY_VERIFICATION.json").read_text())
        summary = completion.get("summary")
        if (
            completion.get("status") != "succeeded"
            or not isinstance(summary, dict)
            or summary.get("fanout") != 8
            or summary.get("prefix_tokens") != 16_384
            or summary.get("seed") != config.seed
            or verification.get("remote_manifest_sha256") != expected_hash
        ):
            raise RuntimeError("fanout-32 approval references invalid or stale fanout-8 evidence")
        evidence_paths.add(str(summary.get("path")))
    if evidence_paths != {"independent_prefill", "shared_root"}:
        raise RuntimeError("fanout-32 evidence must bind independent and shared fanout-8 paths")
    ledger_path = LOCAL_EXPERIMENT_ROOT / "gpu-hours.json"
    if payload.get("gpu_hour_ledger_sha256") != _sha256(ledger_path):
        raise RuntimeError("fanout-32 approval is stale relative to the GPU-hour ledger")


def _fetch_call_logs(call: Any, attempt_id: str, kind: str) -> None:
    call_id = str(getattr(call, "object_id", "unknown-call"))
    retrieval_id = f"{kind}-{call_id}-{uuid.uuid4().hex}"
    root = LOCAL_EXPERIMENT_ROOT / "modal/logs" / attempt_id / retrieval_id
    root.mkdir(parents=True, exist_ok=False)
    for source in ("stdout", "stderr", "system"):
        try:
            entries = call.logs.fetch(source=source)
            lines = []
            for entry in entries:
                timestamp = getattr(entry, "timestamp", None)
                message = getattr(entry, "message", str(entry))
                lines.append(f"{timestamp.isoformat() if timestamp else ''}\t{message}")
            text = "".join(lines)
        except Exception as error:
            text = f"log retrieval failed: {type(error).__name__}: {error}\n"
        path = root / f"{source}.log"
        with path.open("x", encoding="utf-8") as handle:
            handle.write(text)


def _cancel_call(call: Any) -> str | None:
    try:
        # Modal 1.5.3's FunctionCall API defaults this to False.  The provider
        # rejects True for recovered detached calls; cancelling the call still
        # terminates its inputs without retrying, and single-use containers are
        # reclaimed by the platform after the input stops.
        call.cancel(terminate_containers=False)
    except Exception as error:
        return f"{type(error).__name__}: {error}"
    return None


def _require_confirmed_cancellation(
    call: Any, *, context: str, source_error: BaseException
) -> None:
    """Keep the durable reservation intact unless provider cancellation is confirmed."""

    cancel_error = _cancel_call(call)
    if cancel_error is not None:
        raise RuntimeError(
            f"{context}; cancellation was unconfirmed and reservation retained: {cancel_error}"
        ) from source_error


def _recover_or_cancel_call(config: ModalBenchmarkConfig, *, cancel: bool) -> dict[str, Any]:
    ledger = _load_gpu_ledger()
    gpu_reservations = [
        item
        for item in ledger.get("modal_in_flight_reservations", [])
        if item.get("attempt_id") == config.attempt_id
    ]
    model_reservations = [
        item
        for item in ledger.get("modal_in_flight_model_reservations", [])
        if item.get("attempt_id") == config.attempt_id
    ]
    if not gpu_reservations and not model_reservations and not cancel:
        accounted = [
            item
            for item in ledger.get("gpu_active_intervals", [])
            if item.get("attempt_id") == config.attempt_id
            and isinstance(item.get("remote_manifest_sha256"), str)
        ]
        target = LOCAL_EXPERIMENT_ROOT / "modal" / config.attempt_id
        if len(accounted) == 1 and not target.exists():
            verification = _materialize_result(
                config.attempt_id,
                expected_manifest_sha256=str(accounted[0]["remote_manifest_sha256"]),
            )
            succeeded = accounted[0].get("status") == "succeeded"
            return {
                "status": (
                    "accounted_gpu_result_materialized"
                    if succeeded
                    else "accounted_failed_gpu_result_materialized"
                ),
                "function_call_id": accounted[0].get("function_call_id"),
                "local_copy": verification,
                "experiment_succeeded": succeeded,
            }
    if len(gpu_reservations) + len(model_reservations) != 1:
        raise RuntimeError("recovery requires exactly one in-flight reservation for the attempt")
    reservation = (gpu_reservations or model_reservations)[0]
    call_id = reservation.get("function_call_id")
    kind = "gpu" if gpu_reservations else "model_prepare"
    maximum_seconds = (
        GPU_FUNCTION_TIMEOUT_SECONDS + GPU_FUNCTION_STARTUP_TIMEOUT_SECONDS
        if kind == "gpu"
        else MODEL_FUNCTION_TIMEOUT_SECONDS + MODEL_FUNCTION_STARTUP_TIMEOUT_SECONDS
    )
    if not isinstance(call_id, str) or not call_id:
        created = datetime.fromisoformat(str(reservation["created_at_utc"])).astimezone(UTC)
        elapsed = (datetime.now(UTC) - created).total_seconds()
        if kind == "gpu":
            remote_manifest_path = (
                f"{REMOTE_EXPERIMENT_PREFIX}/{config.attempt_id}/REMOTE_MANIFEST.json"
            )
            try:
                manifest_bytes = b"".join(results_volume.read_file(remote_manifest_path))
            except FileNotFoundError:
                return {
                    "status": "ambiguous_spawn_reservation_retained_pending_provider_audit",
                    "kind": kind,
                    "reservation_id": reservation["reservation_id"],
                    "function_call_id": None,
                    "elapsed_seconds": elapsed,
                    "reservation_retained": True,
                    "reason": (
                        "Modal scheduling time is not bounded by Function timeout; absence of "
                        "a final manifest is not terminal evidence"
                    ),
                }
            expected_hash = hashlib.sha256(manifest_bytes).hexdigest()
            synthetic_call_id = f"published-unattached-{reservation['reservation_id']}"
            _charge_failed_gpu_reservation(
                reservation_id=str(reservation["reservation_id"]),
                call_id=synthetic_call_id,
                attempt_id=config.attempt_id,
                reason="unattached_call_published_final_manifest",
                remote_manifest_sha256=expected_hash,
            )
            verification = _materialize_result(
                config.attempt_id,
                expected_manifest_sha256=expected_hash,
            )
            return {
                "status": "unattached_gpu_reservation_charged_and_result_recovered",
                "function_call_id": None,
                "local_copy": verification,
                "experiment_succeeded": False,
            }
        return {
            "status": "ambiguous_spawn_reservation_retained_pending_provider_audit",
            "kind": kind,
            "reservation_id": reservation["reservation_id"],
            "function_call_id": None,
            "elapsed_seconds": elapsed,
            "reservation_retained": True,
            "reason": (
                "the persistent model manifest may predate this call and is not "
                "attempt-terminal evidence"
            ),
        }
    call = modal.FunctionCall.from_id(call_id)
    if cancel:
        cancel_error = _cancel_call(call)
        if cancel_error is not None:
            return {
                "status": "cancellation_unconfirmed_reservation_retained",
                "kind": kind,
                "function_call_id": call_id,
                "cancel_error": cancel_error,
                "reservation_retained": True,
            }
        if kind == "gpu":
            _charge_failed_gpu_reservation(
                reservation_id=str(reservation["reservation_id"]),
                call_id=call_id,
                attempt_id=config.attempt_id,
                reason="explicit_cancel",
            )
        else:
            _reconcile_model_reservation(
                reservation_id=str(reservation["reservation_id"]),
                call_id=call_id,
                attempt_id=config.attempt_id,
                elapsed_seconds=float(maximum_seconds),
                status="explicit_cancel_conservative_charge",
                maximum_charge=True,
            )
        _fetch_call_logs(call, config.attempt_id, f"{kind}-cancel")
        return {
            "status": "cancelled_and_conservatively_charged",
            "kind": kind,
            "function_call_id": call_id,
            "cancel_error": cancel_error,
        }
    try:
        result = call.get(timeout=300)
    except TimeoutError:
        return {
            "status": "still_running_or_pending",
            "kind": kind,
            "function_call_id": call_id,
            "reservation_retained": True,
        }
    except BaseException as recovery_error:
        cancel_error = _cancel_call(call)
        if cancel_error is not None:
            raise RuntimeError(
                f"Modal {kind} recovery and cancellation failed; reservation retained: "
                f"{cancel_error}"
            ) from recovery_error
        if kind == "gpu":
            _charge_failed_gpu_reservation(
                reservation_id=str(reservation["reservation_id"]),
                call_id=call_id,
                attempt_id=config.attempt_id,
                reason=f"recovery:{type(recovery_error).__name__}",
            )
        else:
            _reconcile_model_reservation(
                reservation_id=str(reservation["reservation_id"]),
                call_id=call_id,
                attempt_id=config.attempt_id,
                elapsed_seconds=float(maximum_seconds),
                status=f"recovery_failed:{type(recovery_error).__name__}",
                maximum_charge=True,
            )
        _fetch_call_logs(call, config.attempt_id, f"{kind}-recovery-failed")
        raise RuntimeError(
            f"Modal {kind} recovery failed and was conservatively charged; "
            f"cancel_error={cancel_error}"
        ) from recovery_error
    _fetch_call_logs(call, config.attempt_id, f"{kind}-recovered")
    if kind == "model_prepare":
        _reconcile_model_reservation(
            reservation_id=str(reservation["reservation_id"]),
            call_id=call_id,
            attempt_id=config.attempt_id,
            elapsed_seconds=float(maximum_seconds),
            status=f"recovered:{result['status']}",
            maximum_charge=True,
        )
        return {"status": "model_preparation_recovered", "result": result}
    _record_gpu_hours(
        result,
        reservation_id=str(reservation["reservation_id"]),
        accounted_seconds=float(maximum_seconds),
    )
    verification = _materialize_result(
        config.attempt_id,
        expected_manifest_sha256=str(result["remote_manifest_sha256"]),
    )
    if result.get("status") != "succeeded":
        raise RuntimeError(
            "recovered Modal GPU call completed with a failed experiment status; "
            "artifacts were preserved at "
            f"{LOCAL_EXPERIMENT_ROOT / 'modal' / config.attempt_id}"
        )
    return {
        "status": "gpu_result_recovered",
        "result": result,
        "local_copy": verification,
        "experiment_succeeded": True,
    }


def _cleanup_remote_staging(config: ModalBenchmarkConfig) -> dict[str, Any]:
    ledger = _load_gpu_ledger()
    matching_reservations = [
        {"kind": kind, **item}
        for kind, field in (
            ("gpu", "modal_in_flight_reservations"),
            ("model_prepare", "modal_in_flight_model_reservations"),
        )
        for item in ledger.get(field, [])
        if isinstance(item, dict) and item.get("attempt_id") == config.attempt_id
    ]
    if matching_reservations:
        raise RuntimeError(
            "refusing staging cleanup while the exact attempt has an in-flight reservation"
        )
    remote_staging = f"{REMOTE_EXPERIMENT_PREFIX}/{config.attempt_id}.staging"
    remote_final = f"{REMOTE_EXPERIMENT_PREFIX}/{config.attempt_id}"
    if list(results_volume.iterdir(remote_final, recursive=True)):
        raise RuntimeError("refusing staging cleanup because an immutable final result exists")
    entries = list(results_volume.iterdir(remote_staging, recursive=True))
    if not entries:
        return {"status": "no_remote_staging_present", "remote_staging": remote_staging}
    results_volume.remove_file(remote_staging, recursive=True)
    return {
        "status": "removed_exact_incomplete_remote_staging",
        "remote_staging": remote_staging,
        "entries_removed": len(entries),
    }


def _reconcile_unattached_from_provider_audit(
    config: ModalBenchmarkConfig, *, provider_audit_path: str
) -> dict[str, Any]:
    if not provider_audit_path:
        raise ValueError("--provider-audit-path is required for reconcile-unattached")
    path = Path(provider_audit_path).resolve(strict=True)
    allowed_root = (LOCAL_EXPERIMENT_ROOT / "modal/provider-audits").resolve()
    try:
        path.relative_to(allowed_root)
    except ValueError as error:
        raise ValueError(
            "provider audit must be under Experiment 002 modal/provider-audits"
        ) from error
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise ValueError("provider audit must be one regular file no larger than 1 MiB")
    audit = ProviderTerminalAudit.model_validate_json(path.read_bytes())
    if audit.audited_at_utc.tzinfo is None:
        raise ValueError("provider audit timestamp must include a UTC offset")
    if audit.attempt_id != config.attempt_id:
        raise ValueError("provider audit attempt does not match the configuration")
    ledger = _load_gpu_ledger()
    field = (
        "modal_in_flight_reservations"
        if audit.kind == "gpu"
        else "modal_in_flight_model_reservations"
    )
    reservations = ledger.get(field, [])
    if not isinstance(reservations, list):
        raise ValueError("GPU-hour ledger reservation field is invalid")
    matches = [
        item
        for item in reservations
        if isinstance(item, dict)
        and item.get("attempt_id") == config.attempt_id
        and item.get("reservation_id") == audit.reservation_id
    ]
    if len(matches) != 1 or matches[0].get("function_call_id") is not None:
        raise RuntimeError("provider audit requires exactly one matching unattached reservation")
    reservation_created = datetime.fromisoformat(str(matches[0]["created_at_utc"])).astimezone(UTC)
    audit_time = audit.audited_at_utc.astimezone(UTC)
    if audit_time < reservation_created or audit_time > datetime.now(UTC):
        raise ValueError("provider audit time must be after reservation creation and not future")
    audit_sha256 = _sha256(path)
    synthetic_call_id = f"provider-audited-{audit.reservation_id}"
    if audit.kind == "gpu":
        _charge_failed_gpu_reservation(
            reservation_id=audit.reservation_id,
            call_id=synthetic_call_id,
            attempt_id=config.attempt_id,
            reason=f"provider_terminal_audit:{audit.provider_terminal_state}",
            provider_terminal_audit_sha256=audit_sha256,
        )
    else:
        _reconcile_model_reservation(
            reservation_id=audit.reservation_id,
            call_id=synthetic_call_id,
            attempt_id=config.attempt_id,
            elapsed_seconds=float(
                MODEL_FUNCTION_TIMEOUT_SECONDS + MODEL_FUNCTION_STARTUP_TIMEOUT_SECONDS
            ),
            status=f"provider_terminal_audit:{audit.provider_terminal_state}",
            maximum_charge=True,
            provider_terminal_audit_sha256=audit_sha256,
        )
    return {
        "status": "unattached_reservation_reconciled_from_provider_terminal_audit",
        "kind": audit.kind,
        "attempt_id": config.attempt_id,
        "reservation_id": audit.reservation_id,
        "provider_terminal_state": audit.provider_terminal_state,
        "provider_terminal_audit_path": str(path),
        "provider_terminal_audit_sha256": audit_sha256,
        "conservative_maximum_charge": True,
    }


def main(
    action: str = "validate",
    config_path: str = "",
    remote_manifest_sha256: str = "",
    provider_audit_path: str = "",
) -> None:
    """Validate, prepare, run, or materialize one immutable configuration."""

    _assert_modal_version()
    config = _load_config(config_path)
    if action == "validate":
        print(_canonical_bytes(config).decode(), end="")
        return
    if action not in {
        "prepare-model",
        "run",
        "recover",
        "cancel",
        "materialize",
        "cleanup-staging",
        "reconcile-unattached",
    }:
        raise ValueError("unsupported Modal experiment action")
    if action == "run":
        _assert_adaptive_gate(config)
        gpu_call_record = LOCAL_EXPERIMENT_ROOT / "modal/calls" / f"{config.attempt_id}-gpu.json"
        if gpu_call_record.exists():
            raise FileExistsError(
                f"GPU attempt already has a durable call record: {gpu_call_record}"
            )
    budget = require_cloud_budget(
        maximum_gpu_seconds=0,
        maximum_model_prep_seconds=(
            MODEL_FUNCTION_TIMEOUT_SECONDS + MODEL_FUNCTION_STARTUP_TIMEOUT_SECONDS
            if action in {"prepare-model", "run"}
            else 0
        ),
    )
    print(_canonical_bytes({"cloud_budget": budget.model_dump(mode="json")}).decode(), end="")
    if action in {"recover", "cancel"}:
        print(
            _canonical_bytes(_recover_or_cancel_call(config, cancel=action == "cancel")).decode(),
            end="",
        )
        return
    if action == "cleanup-staging":
        print(_canonical_bytes(_cleanup_remote_staging(config)).decode(), end="")
        return
    if action == "reconcile-unattached":
        print(
            _canonical_bytes(
                _reconcile_unattached_from_provider_audit(
                    config, provider_audit_path=provider_audit_path
                )
            ).decode(),
            end="",
        )
        return
    if action == "materialize":
        if not remote_manifest_sha256:
            raise ValueError("standalone materialization requires --remote-manifest-sha256")
        print(
            _canonical_bytes(
                _materialize_result(
                    config.attempt_id,
                    expected_manifest_sha256=remote_manifest_sha256,
                )
            ).decode(),
            end="",
        )
        return
    model_started = time.monotonic()
    if _prepare_model_function is None:
        raise RuntimeError("Modal graph was not enabled by the budget-first launcher")
    model_reservation_id = _reserve_model_call(config, budget)
    model_authorization = _remote_authorization(
        config,
        kind="model_prepare",
        reservation_id=model_reservation_id,
        decision=budget,
    )
    try:
        model_call = _prepare_model_function.spawn(
            config.model_dump(mode="json"), model_authorization.model_dump(mode="json")
        )
    except BaseException as spawn_error:
        raise RuntimeError(
            "Modal model spawn outcome is ambiguous; unattached reservation retained and "
            "must be recovered or provider-audited before another paid call"
        ) from spawn_error
    try:
        _persist_call_state(
            attempt_id=config.attempt_id,
            kind="model-prepare",
            call_id=model_call.object_id,
            config=config,
        )
        _attach_model_call_to_reservation(model_reservation_id, model_call.object_id)
    except BaseException as persistence_error:
        _require_confirmed_cancellation(
            model_call,
            context="Modal model call-state persistence failed",
            source_error=persistence_error,
        )
        _reconcile_model_reservation(
            reservation_id=model_reservation_id,
            call_id=model_call.object_id,
            attempt_id=config.attempt_id,
            elapsed_seconds=float(
                MODEL_FUNCTION_TIMEOUT_SECONDS + MODEL_FUNCTION_STARTUP_TIMEOUT_SECONDS
            ),
            status="failed_call_state_persistence",
            maximum_charge=True,
        )
        raise
    try:
        model_result = model_call.get(
            timeout=MODEL_FUNCTION_TIMEOUT_SECONDS + MODEL_FUNCTION_STARTUP_TIMEOUT_SECONDS + 300
        )
    except BaseException as model_error:
        _require_confirmed_cancellation(
            model_call,
            context="Modal model call failed",
            source_error=model_error,
        )
        _reconcile_model_reservation(
            reservation_id=model_reservation_id,
            call_id=model_call.object_id,
            attempt_id=config.attempt_id,
            elapsed_seconds=float(
                MODEL_FUNCTION_TIMEOUT_SECONDS + MODEL_FUNCTION_STARTUP_TIMEOUT_SECONDS
            ),
            status="failed_conservative_charge",
            maximum_charge=True,
        )
        raise
    else:
        _reconcile_model_reservation(
            reservation_id=model_reservation_id,
            call_id=model_call.object_id,
            attempt_id=config.attempt_id,
            elapsed_seconds=min(
                time.monotonic() - model_started,
                float(MODEL_FUNCTION_TIMEOUT_SECONDS + MODEL_FUNCTION_STARTUP_TIMEOUT_SECONDS),
            ),
            status=str(model_result["status"]),
            maximum_charge=False,
        )
    finally:
        _fetch_call_logs(model_call, config.attempt_id, "model-prepare")
    print(_canonical_bytes({"model_preparation": model_result}).decode(), end="")
    if model_result.get("status") not in {"prepared", "already_prepared_and_verified"}:
        raise RuntimeError("Modal model preparation returned a non-success status")
    if action == "prepare-model":
        return
    gpu_budget = require_cloud_budget(
        maximum_gpu_seconds=GPU_FUNCTION_TIMEOUT_SECONDS + GPU_FUNCTION_STARTUP_TIMEOUT_SECONDS
    )
    if _run_gpu_function is None:
        raise RuntimeError("Modal GPU graph was not enabled by the budget-first launcher")
    reservation_id = _reserve_gpu_call(config, gpu_budget)
    gpu_authorization = _remote_authorization(
        config,
        kind="gpu",
        reservation_id=reservation_id,
        decision=gpu_budget,
    )
    try:
        gpu_call = _run_gpu_function.spawn(
            config.model_dump(mode="json"), gpu_authorization.model_dump(mode="json")
        )
    except BaseException as spawn_error:
        raise RuntimeError(
            "Modal GPU spawn outcome is ambiguous; unattached reservation retained and must "
            "be recovered or provider-audited before another GPU call"
        ) from spawn_error
    try:
        _persist_call_state(
            attempt_id=config.attempt_id,
            kind="gpu",
            call_id=gpu_call.object_id,
            config=config,
        )
        _attach_call_to_reservation(reservation_id, gpu_call.object_id)
    except BaseException as persistence_error:
        _require_confirmed_cancellation(
            gpu_call,
            context="Modal GPU call-state persistence failed",
            source_error=persistence_error,
        )
        _charge_failed_gpu_reservation(
            reservation_id=reservation_id,
            call_id=gpu_call.object_id,
            attempt_id=config.attempt_id,
            reason="call_state_persistence",
        )
        raise
    gpu_started = time.monotonic()
    try:
        result = gpu_call.get(
            timeout=GPU_FUNCTION_TIMEOUT_SECONDS + GPU_FUNCTION_STARTUP_TIMEOUT_SECONDS + 300
        )
    except BaseException as call_error:
        _require_confirmed_cancellation(
            gpu_call,
            context="Modal GPU call failed",
            source_error=call_error,
        )
        _charge_failed_gpu_reservation(
            reservation_id=reservation_id,
            call_id=gpu_call.object_id,
            attempt_id=config.attempt_id,
            reason=type(call_error).__name__,
        )
        raise
    finally:
        _fetch_call_logs(gpu_call, config.attempt_id, "gpu")
    accounted_seconds = min(
        time.monotonic() - gpu_started,
        float(GPU_FUNCTION_TIMEOUT_SECONDS + GPU_FUNCTION_STARTUP_TIMEOUT_SECONDS),
    )
    _record_gpu_hours(
        result,
        reservation_id=reservation_id,
        accounted_seconds=accounted_seconds,
    )
    _record_cloud_interval(
        kind="gpu",
        call_id=gpu_call.object_id,
        attempt_id=config.attempt_id,
        elapsed_seconds=accounted_seconds,
        status=str(result["status"]),
        maximum_charge=False,
    )
    verification = _materialize_result(
        config.attempt_id,
        expected_manifest_sha256=str(result["remote_manifest_sha256"]),
    )
    if result.get("status") != "succeeded":
        raise RuntimeError(
            "Modal GPU function returned a failed experiment status; artifacts were "
            f"preserved at {LOCAL_EXPERIMENT_ROOT / 'modal' / config.attempt_id}"
        )
    print(_canonical_bytes({"gpu_result": result, "local_copy": verification}).decode(), end="")


if _CLOUD_GRAPH_ENABLED:
    _prepare_model_function = app.function(
        image=model_image,
        volumes={str(MODEL_MOUNT): model_volume},
        cpu=MODEL_FUNCTION_CPU_CORES,
        memory=int(MODEL_FUNCTION_MEMORY_GIB * 1024),
        timeout=MODEL_FUNCTION_TIMEOUT_SECONDS,
        startup_timeout=MODEL_FUNCTION_STARTUP_TIMEOUT_SECONDS,
        retries=0,
        max_containers=1,
        single_use_containers=True,
    )(prepare_model)
    _run_gpu_function = app.function(
        image=gpu_image,
        gpu=GPU_SKU,
        volumes={
            str(MODEL_MOUNT): model_volume.with_mount_options(read_only=True),
            str(RESULTS_MOUNT): results_volume,
        },
        cpu=GPU_FUNCTION_CPU_CORES,
        memory=int(GPU_FUNCTION_MEMORY_GIB * 1024),
        timeout=GPU_FUNCTION_TIMEOUT_SECONDS,
        startup_timeout=GPU_FUNCTION_STARTUP_TIMEOUT_SECONDS,
        retries=0,
        max_containers=1,
        buffer_containers=0,
        single_use_containers=True,
    )(run_gpu_configuration)
    _modal_local_entrypoint = app.local_entrypoint()(main)


__all__ = [
    "APP_NAME",
    "GPU_HOUR_LIMIT",
    "GPU_SKU",
    "MODEL_VOLUME_NAME",
    "RESULTS_VOLUME_NAME",
    "ModalBenchmarkConfig",
    "app",
    "main",
    "prepare_model",
    "require_cloud_budget",
    "run_gpu_configuration",
]
