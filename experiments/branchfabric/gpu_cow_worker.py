"""Fresh-interpreter GPU worker for BranchFabric Experiments 002 and 003.

Only this process imports PyTorch, vLLM, the live runtime adapter, or project
benchmark modules.  The parent controller treats complete interpreter exit as
the authoritative CUDA-context cleanup mechanism.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import threading
import time
import traceback
from collections.abc import Sequence
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

VLLM_VERSION = "0.23.0"
TORCH_VERSION = "2.11.0"
TRANSFORMERS_VERSION = "5.14.1"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
TOKENIZER_REVISION = MODEL_REVISION
POLICY_EPOCH = "branchfabric-modal-exp002-policy-v1"
ADAPTER_VERSION = "1.0.0"


class _NvmlSampler:
    """Bounded low-overhead child-side GPU samples for Experiment 003."""

    def __init__(self, gpu_uuid: str, *, interval_s: float = 0.1) -> None:
        self._gpu_uuid = gpu_uuid
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.rows: list[dict[str, Any]] = []
        self.error: dict[str, str] | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="exp003-nvml", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                raise RuntimeError("bounded Experiment 003 NVML sampler did not stop")

    def _run(self) -> None:
        try:
            import pynvml  # type: ignore[import-not-found]

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByUUID(self._gpu_uuid)
            while not self._stop.is_set() and len(self.rows) < 36_000:
                observed_ns = time.monotonic_ns()
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                row: dict[str, Any] = {
                    "observed_at_monotonic_ns": observed_ns,
                    "gpu_utilization_percent": int(utilization.gpu),
                    "memory_utilization_percent": int(utilization.memory),
                    "memory_used_bytes": int(memory.used),
                }
                for name, counter in (
                    ("pcie_tx_kib_per_second", pynvml.NVML_PCIE_UTIL_TX_BYTES),
                    ("pcie_rx_kib_per_second", pynvml.NVML_PCIE_UTIL_RX_BYTES),
                ):
                    try:
                        row[name] = int(pynvml.nvmlDeviceGetPcieThroughput(handle, counter))
                    except pynvml.NVMLError:
                        row[name] = None
                self.rows.append(row)
                self._stop.wait(self._interval_s)
        except Exception as error:
            self.error = {"type": type(error).__name__, "message": str(error)}
        finally:
            with suppress(Exception):
                pynvml.nvmlShutdown()

    def persist(self, path: Path) -> None:
        _write_json(
            path,
            {
                "schema_version": "sloforge.branchfabric.nvml-samples/v1",
                "gpu_uuid": self._gpu_uuid,
                "interval_seconds": self._interval_s,
                "samples": self.rows,
                "error": self.error,
            },
        )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_bytes(value: Any) -> bytes:
    if hasattr(value, "model_dump"):
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


def _emit_event(path: Path, event: str, **attributes: Any) -> None:
    row = {
        "schema_version": "sloforge.branchfabric.gpu-worker-event/v1",
        "event": event,
        "observed_at_utc": _utc_now(),
        "observed_at_monotonic_ns": time.monotonic_ns(),
        **attributes,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(_canonical_bytes(row))
        handle.flush()
        os.fsync(handle.fileno())


def _emit_historical_event(
    path: Path, event: str, observed_at_monotonic_ns: int, **attributes: Any
) -> None:
    row = {
        "schema_version": "sloforge.branchfabric.gpu-worker-event/v1",
        "event": event,
        "observed_at_utc": None,
        "observed_at_monotonic_ns": observed_at_monotonic_ns,
        "timestamp_source": "runner_phase_timing",
        **attributes,
    }
    with path.open("ab") as handle:
        handle.write(_canonical_bytes(row))
        handle.flush()
        os.fsync(handle.fileno())


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--model-snapshot", required=True, type=Path)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--event-log", required=True, type=Path)
    parser.add_argument("--child-manifest", required=True, type=Path)
    return parser.parse_args(argv)


def _validate_config(config: dict[str, Any]) -> None:
    schema = config.get("schema_version")
    if schema not in {
        "sloforge.branchfabric.modal-real-gpu-cow-config/v1",
        "sloforge.branchfabric.modal-metadata-characterization-config/v1",
    }:
        raise ValueError(f"worker config has unsupported schema_version: {schema!r}")
    expected = {
        "runtime": "vllm",
        "runtime_version": VLLM_VERSION,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"worker config has unsupported {key}: {config.get(key)!r}")
    if config.get("baseline_mode") not in {"independent", "shared_root"}:
        raise ValueError("worker config has unsupported baseline_mode")
    for key in ("fanout", "prefix_length", "suffix_length", "seed"):
        if not isinstance(config.get(key), int) or isinstance(config.get(key), bool):
            raise ValueError(f"worker config has invalid {key}")
    for key in ("fanout", "prefix_length", "suffix_length"):
        if int(config[key]) <= 0:
            raise ValueError(f"worker config requires positive {key}")
    if schema == "sloforge.branchfabric.modal-metadata-characterization-config/v1":
        if config.get("implementation") not in {"baseline", "optimized"}:
            raise ValueError("Experiment 003 requires baseline or optimized implementation")
        if config.get("tracing_level") not in {"disabled", "minimal", "full"}:
            raise ValueError("Experiment 003 has an invalid tracing level")
        if int(config["prefix_length"]) != 16_384 or int(config["suffix_length"]) != 256:
            raise ValueError("Experiment 003 requires an exact 16K/256-token shape")
        if int(config["fanout"]) not in {1, 8, 16, 32}:
            raise ValueError("Experiment 003 fanout must be 1, 8, 16, or 32")


def _validate_model_manifest(snapshot: Path) -> dict[str, Any]:
    import hashlib

    manifest_path = snapshot / "MODEL_MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"model manifest is absent at {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError("model manifest must be a JSON object")
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
        if path.is_symlink() or not path.is_file() or path.stat().st_size != item["bytes"]:
            raise ValueError(f"model inventory entry is invalid: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != item["sha256"]:
            raise ValueError(f"model inventory hash mismatch: {path}")
    return manifest


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dependency_versions(torch: Any) -> dict[str, Any]:
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
    return {
        "packages": {name: importlib.metadata.version(name) for name in names},
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
    config: dict[str, Any], *, gpu_uuid: str, snapshot: Path
) -> tuple[Any, Any]:
    from gpu_validation_runner import (
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
    prefix_length = int(config["prefix_length"])
    fanout = int(config["fanout"])
    prefix = tuple(inputs["prefix_token_ids"][:prefix_length])
    divergent = tuple(inputs["divergent_token_ids"][:fanout])
    if len(prefix) != prefix_length or len(divergent) != fanout:
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
            if prefix_length in {2048, 4096}
            else ExperimentProfile.CONTEXT_16K
        ),
        seed=int(config["seed"]),
        timeout_s=float(config.get("maximum_wall_seconds", 3600)),
        cleanup_timeout_s=float(config.get("cleanup_timeout_seconds", 60)),
        prefix_token_ids=prefix,
        divergent_token_ids=divergent,
        fanout=fanout,
        suffix_tokens=int(config["suffix_length"]),
        expected_identity=identity,
        trace_level=TraceLevel(str(config.get("tracing_level", "full"))),
        gpu_sampling=GpuSamplingConfiguration(
            interval_s=float(config.get("gpu_sample_interval_seconds", 0.25)),
            command_timeout_s=2.0,
            maximum_samples=min(100_000, int(config.get("maximum_wall_seconds", 3600)) * 20),
            physical_device_selector=gpu_uuid,
        ),
    )
    engine = VllmEngineConfiguration(
        execution_model_path=str(snapshot),
        download_dir="/tmp/sloforge-download",
        max_model_len=max(prefix_length + int(config["suffix_length"]) + 2, 4096),
        block_size=16,
        max_num_seqs=max(2, fanout),
        gpu_memory_utilization=float(config.get("gpu_memory_utilization", 0.8)),
        initialization_timeout_s=float(config.get("initialization_timeout_seconds", 1200)),
        maximum_trace_events=262_144,
        enforce_eager=False,
        disable_log_stats=(
            config.get("schema_version")
            != "sloforge.branchfabric.modal-metadata-characterization-config/v1"
        ),
        trust_remote_code=False,
        enable_chunked_prefill=True,
    )
    invocation = VllmGpuValidationInvocation(experiment=experiment, engine=engine)
    path = (
        TrialPath.INDEPENDENT_PREFILL
        if config["baseline_mode"] == "independent"
        else TrialPath.SHARED_ROOT
    )
    return invocation, path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(row) for row in path.read_text().splitlines() if row.strip()]


def _semantic_invariants(
    *, config: dict[str, Any], trial_output: Path, summary: dict[str, Any]
) -> tuple[dict[str, bool], int]:
    physical_rows = _read_jsonl(trial_output / "raw/physical-snapshots.jsonl")
    gpu_rows = _read_jsonl(trial_output / "raw/gpu-memory-states.jsonl")
    physical = {str(row["phase"]): row["snapshot"] for row in physical_rows}
    gpu = {str(row["phase"]): row["state"] for row in gpu_rows}
    assertions = set(summary.get("assertions", []))
    final_phase = "post_root_delete" if config["baseline_mode"] == "shared_root" else "post_destroy"
    final_runtime_assigned = int(gpu[final_phase]["kv_assigned_bytes"])
    identity = summary["identity"]
    common = {
        "real_gpu_execution": identity["device"] == "cuda:0",
        "pinned_runtime_identity": (
            identity["runtime"] == "vllm"
            and identity["runtime_version"] == VLLM_VERSION
            and identity["model_id"] == MODEL_ID
            and identity["model_revision"] == MODEL_REVISION
            and identity["tokenizer_id"] == MODEL_ID
            and identity["tokenizer_revision"] == TOKENIZER_REVISION
            and identity["dtype"] == "bfloat16"
        ),
        "branches_decode_independently": (
            int(summary["decode_tokens"]) > 0
            and (
                "branches_decode_concurrently_under_one_branchpoint" in assertions
                or "independent_branches_decode_concurrently" in assertions
            )
        ),
        "final_runtime_assigned_kv_bytes_zero": final_runtime_assigned == 0,
    }
    if config["baseline_mode"] == "shared_root":
        root = physical["root"]
        fork = physical["fork"]
        divergence = physical["divergence"]
        expected_blocks = (int(config["prefix_length"]) + 15) // 16
        post_branch = physical.get("post_branch_delete")
        post_root = physical["post_root_delete"]
        root_ids = set(root["shared_prefix_block_ids"])
        divergence_sessions = set(divergence["session_ids"])
        divergence_blocks = {block["runtime_block_id"]: block for block in divergence["blocks"]}
        private_owners = {
            owner
            for block_id in divergence["private_suffix_block_ids"]
            for owner in divergence_blocks[block_id]["branch_ids"]
        }
        shared_blocks_cover_all_branches = all(
            set(divergence_blocks[block_id]["branch_ids"]) == divergence_sessions
            and int(divergence_blocks[block_id]["refcount"]) >= int(config["fanout"])
            for block_id in root_ids
        )
        smoke_requires_single_branch_delete = int(config["prefix_length"]) in {2048, 4096}
        common.update(
            {
                "prefix_physical_blocks_exposed": len(root["shared_prefix_block_ids"])
                == expected_blocks,
                "shared_physical_root_demonstrated": (
                    root_ids == set(divergence["shared_prefix_block_ids"])
                    and shared_blocks_cover_all_branches
                    and "all_branches_reference_exact_root_block_ids" in assertions
                    and "runtime_native_refcounts_cover_live_branches" in assertions
                ),
                "zero_prefix_block_duplication_at_fork": (
                    root_ids == set(fork["shared_prefix_block_ids"])
                    and int(fork["physical_assigned_bytes"]) == int(root["physical_assigned_bytes"])
                    and not fork["private_suffix_block_ids"]
                    and "prefix_prefilled_once_before_branch_admission" in assertions
                ),
                "private_suffix_allocation_observed": (
                    bool(divergence["private_suffix_block_ids"])
                    and private_owners == divergence_sessions
                    and "private_suffix_allocation_observed_after_divergence" in assertions
                ),
                "concurrent_decode_asserted": (
                    "branches_decode_concurrently_under_one_branchpoint" in assertions
                ),
                "destroying_one_branch_preserves_other": (
                    (post_branch is None and not smoke_requires_single_branch_delete)
                    or (
                        post_branch is not None
                        and set(post_branch["shared_prefix_block_ids"]) == root_ids
                        and len(post_branch["session_ids"]) == int(config["fanout"]) - 1
                        and "deleting_branch_a_preserves_branch_b_and_shared_root" in assertions
                    )
                ),
                "final_branch_releases_shared_root": (
                    not post_root["shared_prefix_block_ids"]
                    and not post_root["private_suffix_block_ids"]
                    and int(post_root["physical_assigned_bytes"]) == 0
                    and "final_branch_and_root_release_physical_state" in assertions
                ),
                "runtime_native_release_evidence": (
                    "final_root_exact_ids_hash_cleared_and_complete_kv_pool_recovered" in assertions
                ),
            }
        )
    else:
        final = physical[f"suffix_{int(config['suffix_length'])}"]
        common.update(
            {
                "prefix_cache_disabled_or_isolated": (
                    "independent_prefix_cache_hits_absent" in assertions
                    and not final["shared_prefix_block_ids"]
                ),
                "independent_prefix_block_sets_disjoint": (
                    "independent_branch_physical_block_sets_disjoint" in assertions
                ),
                "every_branch_performed_full_prefill": (
                    "every_branch_executes_complete_prefix_prefill" in assertions
                ),
            }
        )
    return common, final_runtime_assigned


def _emit_runner_phase_events(event_log: Path, trial_output: Path) -> None:
    phase_path = trial_output / "raw/phase-timings.jsonl"
    if not phase_path.is_file():
        return
    names = {
        "root_prefill": "prefix_prefill",
        "root_publish": "shared_root_publish",
        "fork_metadata": "branch_fork",
        "concurrent_decode": "decode",
        "independent_session_metadata": "independent_branch_admission",
        "independent_prefill_and_concurrent_decode": "independent_prefill_and_decode",
    }
    for row in _read_jsonl(phase_path):
        phase = str(row.get("phase"))
        if phase not in names:
            continue
        _emit_historical_event(
            event_log,
            names[phase] + "_start",
            int(row["wall_start_monotonic_ns"]),
            runner_phase=phase,
        )
        _emit_historical_event(
            event_log,
            names[phase] + "_end",
            int(row["wall_end_monotonic_ns"]),
            runner_phase=phase,
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    child_started_ns = time.monotonic_ns()
    child_started_utc = _utc_now()
    event_log = args.event_log.resolve()
    manifest_path = args.child_manifest.resolve()
    work_root = args.work_root.resolve()
    wrapper = work_root / "wrapper"
    wrapper.mkdir(parents=True, exist_ok=True)
    config: dict[str, Any] = {}
    status = "failed"
    error_record: dict[str, Any] | None = None
    summary_payload: dict[str, Any] | None = None
    semantic_invariants: dict[str, bool] = {}
    final_runtime_assigned_kv_bytes: int | None = None
    versions: dict[str, Any] | None = None
    torch_diagnostics: dict[str, Any] = {}
    model_load_started_ns: int | None = None
    model_ready_ns: int | None = None
    benchmark_started_ns: int | None = None
    benchmark_ended_ns: int | None = None
    adapter: Any = None
    torch: Any = None
    trial_output: Path | None = None
    nvml_sampler: _NvmlSampler | None = None
    try:
        config = json.loads(args.config.resolve(strict=True).read_text())
        _validate_config(config)
        _emit_event(
            event_log,
            "child_process_started",
            child_pid=os.getpid(),
            child_pgid=os.getpgid(0),
            child_sid=os.getsid(0),
            parent_pid=os.getppid(),
        )
        # Experiment 003 hot compiler/cache state is deliberately outside the
        # trial artifact tree.  The enclosing Modal Function owns /tmp and is
        # single-use, so these bounded per-child directories disappear with
        # the container instead of being copied to the persistent results
        # Volume.  Keep the Experiment 002 layout unchanged for compatibility.
        cache_root = work_root
        if config["schema_version"] == (
            "sloforge.branchfabric.modal-metadata-characterization-config/v1"
        ):
            cache_root = (
                Path("/tmp/sloforge-branchfabric-003-ephemeral")
                / str(config["campaign_id"])
                / str(config["attempt_id"])
            )
            cache_root.mkdir(parents=True, exist_ok=False)
        os.environ.update(
            {
                "XDG_CACHE_HOME": str(cache_root / "cache"),
                "HF_HOME": str(cache_root / "hf"),
                "HUGGINGFACE_HUB_CACHE": str(cache_root / "hf/hub"),
                "TRANSFORMERS_CACHE": str(cache_root / "hf/transformers"),
                "TORCH_HOME": str(cache_root / "torch"),
                "TRITON_CACHE_DIR": str(cache_root / "triton"),
                "TMPDIR": str(cache_root / "tmp"),
            }
        )
        for name in ("cache", "hf", "torch", "triton", "tmp"):
            (cache_root / name).mkdir(exist_ok=True)
        _emit_event(event_log, "cuda_runtime_import_started")
        import torch as imported_torch  # type: ignore[import-not-found]

        torch = imported_torch
        _emit_event(event_log, "cuda_runtime_import_completed", torch_version=torch.__version__)
        versions = _dependency_versions(torch)
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
        snapshot = args.model_snapshot.resolve(strict=True)
        _validate_model_manifest(snapshot)
        _write_json(wrapper / "modal-config.json", config)
        _write_json(wrapper / "dependency-versions.json", versions)
        _write_json(
            wrapper / "model-manifest-reference.json",
            {
                "path": str(snapshot / "MODEL_MANIFEST.json"),
                "sha256": _sha256(snapshot / "MODEL_MANIFEST.json"),
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "tokenizer_revision": TOKENIZER_REVISION,
            },
        )
        invocation, path = _build_runner_input(
            config,
            gpu_uuid=str(args.gpu_uuid),
            snapshot=snapshot,
        )
        _write_json(wrapper / "invocation.json", invocation)
        from gpu_validation_runner import (
            build_vllm_adapter_factory,
            run_independent_trial,
            run_shared_trial,
        )

        experiment_003 = (
            config["schema_version"]
            == "sloforge.branchfabric.modal-metadata-characterization-config/v1"
        )
        factory = build_vllm_adapter_factory(
            invocation,
            metadata_instrumentation_level=(
                str(config["tracing_level"]) if experiment_003 else "disabled"
            ),
            metadata_optimization=(
                "cached_immutable_root_hashes"
                if experiment_003 and config["implementation"] == "optimized"
                else "baseline"
            ),
        )
        model_load_started_ns = time.monotonic_ns()
        _emit_event(event_log, "model_load_started")
        adapter = factory(path)
        model_ready_ns = time.monotonic_ns()
        _emit_event(event_log, "model_load_completed")
        _emit_event(event_log, "kv_pool_allocated")
        benchmark_started_ns = time.monotonic_ns()
        _emit_event(event_log, "benchmark_started", trial_path=path.value)
        trial_output = work_root / "runner" / path.value
        if experiment_003:
            nvml_sampler = _NvmlSampler(str(args.gpu_uuid))
            nvml_sampler.start()
        if experiment_003:
            from metadata_characterization import (
                MetadataTrialConfiguration,
                run_metadata_trial,
            )

            trial_config = MetadataTrialConfiguration(
                attempt_id=str(config["attempt_id"]),
                mode=(
                    "independent_prefill"
                    if config["baseline_mode"] == "independent"
                    else "shared_root"
                ),
                implementation=cast(Literal["baseline", "optimized"], config["implementation"]),
                seed=int(config["seed"]),
                prefix_token_ids=invocation.experiment.prefix_token_ids,
                divergent_token_ids=invocation.experiment.divergent_token_ids,
                suffix_tokens=invocation.experiment.suffix_tokens,
                timeout_s=invocation.experiment.timeout_s,
                trace_level=cast(Literal["disabled", "minimal", "full"], config["tracing_level"]),
            )
            summary_payload = run_metadata_trial(adapter, trial_config, trial_output)
            summary = None
        elif config["baseline_mode"] == "independent":
            summary = run_independent_trial(adapter, invocation.experiment, trial_output)
        else:
            summary = run_shared_trial(adapter, invocation.experiment, trial_output)
        adapter = None
        benchmark_ended_ns = time.monotonic_ns()
        if nvml_sampler is not None:
            nvml_sampler.stop()
            nvml_sampler.persist(trial_output / "raw/nvml-samples.json")
            if nvml_sampler.error is not None:
                raise RuntimeError(f"Experiment 003 NVML sampling failed: {nvml_sampler.error}")
        _emit_event(event_log, "branch_teardown_and_vllm_shutdown_completed")
        _emit_event(event_log, "benchmark_completed", trial_path=path.value)
        if not experiment_003:
            assert summary is not None
            summary_payload = summary.model_dump(mode="json")
        assert summary_payload is not None
        _write_json(wrapper / "trial-summary.json", summary_payload)
        _emit_runner_phase_events(event_log, trial_output)
        if experiment_003:
            semantics = summary_payload.get("semantics", {})
            semantic_invariants = {
                "real_gpu_execution": True,
                "pinned_runtime_and_model": True,
                "runtime_assigned_kv_released": (
                    summary_payload.get("final_runtime_assigned_kv_bytes") == 0
                ),
                **{
                    str(name): value for name, value in semantics.items() if isinstance(value, bool)
                },
            }
            final_runtime_assigned_kv_bytes = int(
                summary_payload["final_runtime_assigned_kv_bytes"]
            )
        else:
            semantic_invariants, final_runtime_assigned_kv_bytes = _semantic_invariants(
                config=config,
                trial_output=trial_output,
                summary=summary_payload,
            )
        if not all(semantic_invariants.values()):
            failed = sorted(key for key, value in semantic_invariants.items() if not value)
            raise RuntimeError("worker semantic invariant failure: " + ", ".join(failed))
        torch.cuda.synchronize()
        gc.collect()
        torch_diagnostics["before_empty_cache"] = {
            "allocated_bytes": int(torch.cuda.memory_allocated()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
        }
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch_diagnostics["after_empty_cache"] = {
            "allocated_bytes": int(torch.cuda.memory_allocated()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
        }
        status = "succeeded"
    except Exception as error:
        error_record = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        _write_json(wrapper / "failure.json", error_record)
    finally:
        if nvml_sampler is not None and nvml_sampler._thread is not None:
            nvml_sampler.stop()
        if adapter is not None:
            try:
                adapter.cleanup_runtime(timeout_s=float(config.get("cleanup_timeout_seconds", 60)))
                _emit_event(event_log, "vllm_shutdown_completed_after_error")
            except Exception as cleanup_error:
                status = "failed"
                cleanup_record = {
                    "type": type(cleanup_error).__name__,
                    "message": str(cleanup_error),
                    "traceback": traceback.format_exc(),
                }
                _write_json(wrapper / "cleanup-failure.json", cleanup_record)
                if error_record is None:
                    error_record = cleanup_record
        if torch is not None:
            try:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch_diagnostics["final_before_process_exit"] = {
                    "allocated_bytes": int(torch.cuda.memory_allocated()),
                    "reserved_bytes": int(torch.cuda.memory_reserved()),
                }
            except Exception as diagnostic_error:
                torch_diagnostics["final_diagnostic_error"] = {
                    "type": type(diagnostic_error).__name__,
                    "message": str(diagnostic_error),
                }
        active_children: list[dict[str, Any]] = []
        try:
            import multiprocessing

            active_children = [
                {"pid": child.pid, "name": child.name, "daemon": child.daemon}
                for child in multiprocessing.active_children()
            ]
        except Exception as child_error:
            active_children = [{"audit_error": str(child_error)}]
        if active_children:
            status = "failed"
            if error_record is None:
                error_record = {
                    "type": "SurvivingChildProcessError",
                    "message": f"worker retained multiprocessing children: {active_children}",
                }
        child_end_ns = time.monotonic_ns()
        manifest = {
            "schema_version": "sloforge.branchfabric.gpu-cow-child-manifest/v1",
            "attempt_id": config.get("attempt_id"),
            "status": status,
            "child_pid": os.getpid(),
            "child_pgid": os.getpgid(0),
            "child_sid": os.getsid(0),
            "parent_pid": os.getppid(),
            "child_start_utc": child_started_utc,
            "child_end_utc": _utc_now(),
            "child_duration_seconds": (child_end_ns - child_started_ns) / 1e9,
            "model_load_start_monotonic_ns": model_load_started_ns,
            "model_ready_monotonic_ns": model_ready_ns,
            "model_load_and_warmup_seconds": (
                None
                if model_load_started_ns is None or model_ready_ns is None
                else (model_ready_ns - model_load_started_ns) / 1e9
            ),
            "benchmark_start_monotonic_ns": benchmark_started_ns,
            "benchmark_end_monotonic_ns": benchmark_ended_ns,
            "benchmark_seconds": (
                None
                if benchmark_started_ns is None or benchmark_ended_ns is None
                else (benchmark_ended_ns - benchmark_started_ns) / 1e9
            ),
            "versions": versions,
            "cuda_visibility": {
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "expected_gpu_uuid": args.gpu_uuid,
            },
            "vllm_process_architecture": {
                "VLLM_ENABLE_V1_MULTIPROCESSING": os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING"),
                "tensor_parallel_size": 1,
                "expected_engine_client": "InprocClient",
                "expected_executor": "UniProcExecutor",
                "supported_shutdown_path": (
                    "VllmLiveStateAdapter.cleanup_runtime -> EngineCoreClient.shutdown"
                ),
            },
            "semantic_invariants": semantic_invariants,
            "final_runtime_assigned_kv_bytes": final_runtime_assigned_kv_bytes,
            "torch_memory_diagnostics": torch_diagnostics,
            "multiprocessing_children_before_exit": active_children,
            "summary": summary_payload,
            "error": error_record,
            "cleanup_mechanism": "complete_child_process_termination",
            "best_effort_allocator_calls_are_diagnostic_only": True,
        }
        _write_json(manifest_path, manifest)
        _emit_event(event_log, "child_manifest_flushed_exit_imminent", status=status)
    return 0 if status == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
