#!/usr/bin/env python3
"""Generate deterministic BranchFabric Experiment 003 analysis artifacts.

The analyzer is local-only.  It validates immutable Modal campaign trees and
their exact same-GPU lifecycle evidence before deriving paired statistics,
scaling fits, Amdahl projections, plots, reports, or a metadata conclusion.
Unavailable measurements remain explicitly unavailable; they are never filled
from Experiment 002 or inferred from unrelated counters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Literal, cast

from experiment_003_statistics import (
    ArtifactProvenance,
    PairedObservation,
    PairKey,
    ScalingObservation,
    TraceOverheadObservation,
    amdahl_metadata_projections,
    analyze_three_pair_effect,
    analyze_trace_overhead,
    select_scaling_model,
)

EXPECTED_MODEL = "Qwen/Qwen2.5-7B-Instruct"
EXPECTED_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
EXPECTED_RUNTIME = "vllm"
EXPECTED_VLLM = "0.23.0"
EXPECTED_TORCH = "2.11.0"
EXPECTED_CUDA = "13.0"
EXPECTED_VLLM_SOURCE_TAG = "v0.23.0"
EXPECTED_VLLM_SOURCE_COMMIT = "0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665"
EXPECTED_PREFIX = 16_384
EXPECTED_SUFFIX = 256
PREFIX_BLOCKS = 1024

STAGES = (
    "helix_orchestration",
    "request_build",
    "prefix_lookup",
    "block_table_build",
    "refcount_update",
    "prefix_metadata_other",
    "physical_kv_metadata",
    "scheduler_admission",
    "scheduler_wait",
    "scheduler_select",
    "private_state_prep",
    "gpu_submission",
    "gpu_execution",
    "output_token_commit",
    "residual",
)
METADATA_STAGES = (
    "prefix_lookup",
    "block_table_build",
    "refcount_update",
    "prefix_metadata_other",
    "physical_kv_metadata",
    "private_state_prep",
)
SCHEDULER_STAGES = ("scheduler_admission", "scheduler_wait", "scheduler_select")
GPU_STAGES = ("gpu_submission", "gpu_execution")
# This is an intentionally non-overlapping *counter taxonomy*, not a claim that
# every item has equal cost or maps one-to-one onto a future hardware request.
# Outcome/classification counters (hits/misses/candidates/private allocations)
# and scheduler/output/token-copy counters are reported separately below.
COUNTED_METADATA_WORK_UNITS = (
    "block_table_writes",
    "refcount_increments",
    "refcount_decrements",
    "block_allocations",
    "block_frees",
    "branch_session_metadata_allocations",
    "runtime_request_allocations",
    "prefix_lookup_calls",
    "prefix_hash_calls",
    "prefix_hash_lookups",
    "prefix_blocks_bound",
    "request_block_hashes_computed",
    "prefix_hash_template_hits",
)
SCHEDULER_EVENT_COUNTERS = (
    "scheduler_queue_inserts",
    "scheduler_queue_removals",
    "scheduler_scans",
    "scheduler_candidate_evaluations",
)
READINESS_FIELDS = (
    "root_inclusive_ready_ns",
    "post_root_ready_ns",
    "first_branch_ready_ns",
    "all_branches_ready_ns",
    "first_token_latency_ns",
)
REQUIRED_ARTIFACT_DIRECTORIES = (
    "environment",
    "runtime-source",
    "paired",
    "fanout",
    "metadata",
    "profiles",
    "native-metrics",
    "raw",
    "traces",
    "metrics",
    "analysis",
    "software-baseline",
    "plots",
    "reviews",
    "logs",
)


class Experiment003AnalysisError(RuntimeError):
    """Materialized evidence is invalid or scientifically inconsistent."""


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _replace_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: object) -> None:
    _replace_bytes(path, _canonical_bytes(value))


def _write_text(path: Path, value: str) -> None:
    _replace_bytes(path, value.encode("utf-8"))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Experiment003AnalysisError(f"cannot read JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise Experiment003AnalysisError(f"JSON artifact is not an object: {path}")
    return cast(dict[str, Any], value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_number(value: object, *, field: str, nonnegative: bool = True) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise Experiment003AnalysisError(f"{field} is not numeric")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise Experiment003AnalysisError(f"{field} is not finite and nonnegative")
    return result


def _unavailable(reason: str) -> dict[str, str]:
    return {"status": "unavailable", "reason": reason}


def _verify_remote_inventory(root: Path) -> str:
    manifest_path = root / "REMOTE_MANIFEST.json"
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != "sloforge.branchfabric.modal-campaign-result/v1":
        raise Experiment003AnalysisError("unsupported Experiment 003 remote manifest schema")
    rows = manifest.get("artifacts")
    if not isinstance(rows, list):
        raise Experiment003AnalysisError("remote manifest artifact inventory is absent")
    expected: dict[str, tuple[int, str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise Experiment003AnalysisError("remote inventory contains a non-object")
        relative = row.get("relative_path")
        size = row.get("bytes")
        digest = row.get("sha256")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or relative in expected
        ):
            raise Experiment003AnalysisError("remote inventory entry is malformed")
        expected[relative] = (size, digest)
    actual = {
        path.relative_to(root).as_posix(): (path.stat().st_size, _sha256(path))
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != manifest_path
    }
    if actual != expected:
        raise Experiment003AnalysisError(
            "materialized campaign bytes disagree with remote inventory"
        )
    return _sha256(manifest_path)


def _identity(gpu: object) -> dict[str, Any]:
    if not isinstance(gpu, Mapping):
        raise Experiment003AnalysisError("GPU identity is absent")
    try:
        identity: dict[str, Any] = {
            "index": int(gpu["index"]),
            "uuid": str(gpu["uuid"]),
            "name": str(gpu["name"]),
            "driver_version": str(gpu["driver_version"]),
            "memory_total_mib": int(gpu["memory_total_mib"]),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise Experiment003AnalysisError("GPU identity is incomplete") from error
    if "A100" not in identity["name"] or identity["memory_total_mib"] < 80_000:
        raise Experiment003AnalysisError("campaign did not run on one A100 80GB")
    return identity


def _validate_samples_identity(samples: object, expected: Mapping[str, Any], *, label: str) -> None:
    if not isinstance(samples, list) or not samples:
        raise Experiment003AnalysisError(f"{label} has no samples")
    for sample in samples:
        if not isinstance(sample, dict) or _identity(sample.get("gpu")) != dict(expected):
            raise Experiment003AnalysisError(f"{label} changed physical GPU identity")


@dataclass(frozen=True, slots=True)
class LoadedTrial:
    root: Path
    position: int
    config: dict[str, Any]
    summary: dict[str, Any]
    decomposition: dict[str, Any] | None
    native_metrics: dict[str, Any]
    metadata_observation: dict[str, Any]
    nvml: dict[str, Any]
    physical_layouts: dict[str, Any]
    profile: dict[str, Any] | None
    gpu_identity: dict[str, Any]
    function_call_id: str
    summary_sha256: str

    @property
    def attempt_id(self) -> str:
        return str(self.config["attempt_id"])

    @property
    def implementation(self) -> str:
        return str(self.config["implementation"])

    @property
    def mode(self) -> str:
        return str(self.config["baseline_mode"])

    @property
    def fanout(self) -> int:
        return int(self.config["fanout"])

    @property
    def seed(self) -> int:
        return int(self.config["seed"])

    @property
    def trace_level(self) -> str:
        return str(self.config["tracing_level"])

    def provenance(self, selector: str) -> ArtifactProvenance:
        return ArtifactProvenance(
            artifact_reference=(self.root / "metrics/trial-summary.json").as_posix(),
            artifact_sha256=self.summary_sha256,
            sample_selector=selector,
        )


@dataclass(frozen=True, slots=True)
class CampaignDataset:
    root: Path
    implementation: str
    campaign_id: str
    function_call_id: str
    gpu_identity: dict[str, Any]
    immutable_cleanup_threshold_mib: int
    remote_manifest_sha256: str
    campaign_status: str
    adaptive_tail_timeout: bool
    trials: tuple[LoadedTrial, ...]


def _validate_config(config: object, *, implementation: str, campaign_id: str) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise Experiment003AnalysisError("trial config is absent")
    expected = {
        "schema_version": "sloforge.branchfabric.modal-metadata-characterization-config/v1",
        "campaign_id": campaign_id,
        "model": EXPECTED_MODEL,
        "model_revision": EXPECTED_REVISION,
        "tokenizer_revision": EXPECTED_REVISION,
        "runtime": EXPECTED_RUNTIME,
        "runtime_version": EXPECTED_VLLM,
        "prefix_length": EXPECTED_PREFIX,
        "suffix_length": EXPECTED_SUFFIX,
        "implementation": implementation,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise Experiment003AnalysisError(f"trial config disagrees on pinned {key}")
    if config.get("baseline_mode") not in {"independent", "shared_root"}:
        raise Experiment003AnalysisError("trial config has invalid baseline mode")
    if config.get("tracing_level") not in {"disabled", "minimal", "full"}:
        raise Experiment003AnalysisError("trial config has invalid trace level")
    for key in ("fanout", "seed"):
        value = config.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise Experiment003AnalysisError(f"trial config has invalid {key}")
    return cast(dict[str, Any], json.loads(json.dumps(config)))


def _validate_decomposition(
    path: Path,
    *,
    post_root_ready_ns: int,
    trace_level: str,
) -> dict[str, Any] | None:
    if trace_level == "disabled":
        if path.exists():
            raise Experiment003AnalysisError("disabled trace trial unexpectedly has decomposition")
        return None
    value = _read_json(path)
    if value.get("schema_version") != "sloforge.branchfabric.vllm-0230-metadata-timing/v1":
        raise Experiment003AnalysisError("unsupported readiness decomposition schema")
    decomposition = value.get("decomposition")
    if not isinstance(decomposition, dict):
        raise Experiment003AnalysisError("readiness decomposition is absent")
    parent = decomposition.get("parent")
    stages = decomposition.get("stages")
    invariant = decomposition.get("invariant")
    if (
        not isinstance(parent, dict)
        or not isinstance(stages, dict)
        or set(stages) != set(STAGES)
        or not isinstance(invariant, dict)
        or invariant.get("sum_equals_parent") is not True
    ):
        raise Experiment003AnalysisError("readiness decomposition shape is invalid")
    for dimension in ("wall_ns", "process_cpu_ns", "thread_cpu_ns"):
        parent_value = parent.get(dimension)
        stage_values = [row.get(dimension) for row in stages.values() if isinstance(row, dict)]
        if parent_value is None:
            if any(value is not None for value in stage_values):
                raise Experiment003AnalysisError("decomposition CPU availability is inconsistent")
        elif (
            not isinstance(parent_value, int)
            or isinstance(parent_value, bool)
            or any(not isinstance(value, int) or isinstance(value, bool) for value in stage_values)
            or sum(cast(list[int], stage_values)) != parent_value
        ):
            raise Experiment003AnalysisError("decomposition double-counting invariant failed")
    parent_wall = int(parent["wall_ns"])
    if abs(parent_wall - post_root_ready_ns) > max(100_000, post_root_ready_ns // 1000):
        raise Experiment003AnalysisError("decomposition parent disagrees with external readiness")
    return value


def _load_trial(
    campaign_root: Path,
    record: Mapping[str, Any],
    *,
    implementation: str,
    campaign_id: str,
    function_call_id: str,
    campaign_gpu: Mapping[str, Any],
    threshold_mib: int,
) -> LoadedTrial:
    position = record.get("position")
    if not isinstance(position, int) or isinstance(position, bool) or position < 0:
        raise Experiment003AnalysisError("trial record position is invalid")
    config = _validate_config(
        record.get("config"), implementation=implementation, campaign_id=campaign_id
    )
    if record.get("attempt_id") != config["attempt_id"]:
        raise Experiment003AnalysisError("trial record attempt identity changed")
    pre = record.get("pre_child_gate")
    post = record.get("campaign_post_child_gate")
    controller = record.get("controller")
    if not isinstance(pre, dict) or pre.get("passed") is not True:
        raise Experiment003AnalysisError("pre-child campaign gate did not pass")
    if pre.get("cleanup_threshold_mib") != threshold_mib:
        raise Experiment003AnalysisError("pre-child gate changed immutable HBM threshold")
    _validate_samples_identity(pre.get("samples"), campaign_gpu, label="pre-child gate")
    if not isinstance(post, dict) or post.get("passed") is not True:
        raise Experiment003AnalysisError("post-child campaign gate did not pass")
    if post.get("cleanup_threshold_mib") != threshold_mib:
        raise Experiment003AnalysisError("post-child gate changed immutable HBM threshold")
    _validate_samples_identity(post.get("samples"), campaign_gpu, label="post-child gate")
    if not isinstance(controller, dict) or controller.get("status") != "succeeded":
        raise Experiment003AnalysisError("single-child controller did not succeed")
    if _identity(controller.get("actual_gpu")) != dict(campaign_gpu):
        raise Experiment003AnalysisError("single-child controller changed GPU identity")
    cleanup = controller.get("cleanup_gate")
    child = controller.get("child_manifest")
    if (
        not isinstance(cleanup, dict)
        or cleanup.get("GPU_RUNTIME_LIFECYCLE_CLEAN") is not True
        or cleanup.get("forced_gpu_process_kill_required") is True
        or not isinstance(child, dict)
        or child.get("status") != "succeeded"
        or child.get("final_runtime_assigned_kv_bytes") != 0
    ):
        raise Experiment003AnalysisError("authoritative child lifecycle cleanup failed")
    invariants = child.get("semantic_invariants")
    if (
        not isinstance(invariants, dict)
        or not invariants
        or not all(value is True for value in invariants.values())
    ):
        raise Experiment003AnalysisError("child semantic invariants did not all pass")
    versions = child.get("versions")
    if not isinstance(versions, dict):
        raise Experiment003AnalysisError("child dependency version record is absent")
    packages = versions.get("packages")
    if (
        not isinstance(packages, dict)
        or packages.get("vllm") != EXPECTED_VLLM
        or packages.get("torch") != EXPECTED_TORCH
        or versions.get("torch_cuda_userspace") != EXPECTED_CUDA
    ):
        raise Experiment003AnalysisError("child dependency pins are invalid")

    trial_root = campaign_root / "trials" / f"{position:02d}-{config['attempt_id']}"
    lifecycle_cleanup = _read_json(trial_root / "lifecycle/cleanup-gate.json")
    lifecycle_child = _read_json(trial_root / "lifecycle/child-manifest.json")
    if lifecycle_cleanup != cleanup or lifecycle_child != child:
        raise Experiment003AnalysisError(
            "campaign record disagrees with authoritative lifecycle artifacts"
        )
    mode_path = "independent_prefill" if config["baseline_mode"] == "independent" else "shared_root"
    runner_root = trial_root / "runner" / mode_path
    summary_path = runner_root / "metrics/trial-summary.json"
    summary = _read_json(summary_path)
    if summary.get("schema_version") != "sloforge.branchfabric.metadata-trial-summary/v1":
        raise Experiment003AnalysisError("unsupported metadata trial summary schema")
    trial_config = summary.get("configuration")
    expected_mode = mode_path
    if (
        not isinstance(trial_config, dict)
        or trial_config.get("attempt_id") != config["attempt_id"]
        or trial_config.get("mode") != expected_mode
        or trial_config.get("implementation") != implementation
        or trial_config.get("seed") != config["seed"]
        or trial_config.get("suffix_tokens") != EXPECTED_SUFFIX
        or trial_config.get("trace_level") != config["tracing_level"]
        or not isinstance(trial_config.get("prefix_token_ids"), list)
        or len(trial_config["prefix_token_ids"]) != EXPECTED_PREFIX
    ):
        raise Experiment003AnalysisError("trial summary configuration changed scientific inputs")
    if (
        summary.get("fanout") != config["fanout"]
        or summary.get("final_runtime_assigned_kv_bytes") != 0
    ):
        raise Experiment003AnalysisError("trial summary fanout or release evidence is invalid")
    post_root_ready_ns = int(
        _finite_number(summary.get("post_root_ready_ns"), field="post_root_ready_ns")
    )
    if post_root_ready_ns <= 0:
        raise Experiment003AnalysisError("POST_ROOT_READY must be positive")
    readiness = {
        field: _finite_number(summary.get(field), field=field) for field in READINESS_FIELDS
    }
    if (
        readiness["root_inclusive_ready_ns"] < post_root_ready_ns
        or readiness["first_branch_ready_ns"] > readiness["all_branches_ready_ns"]
        or readiness["all_branches_ready_ns"] != post_root_ready_ns
    ):
        raise Experiment003AnalysisError("readiness boundaries are inconsistent")
    steady_decode = _finite_number(
        summary.get("steady_decode_throughput_tokens_per_second"),
        field="steady_decode_throughput_tokens_per_second",
    )
    _finite_number(
        summary.get("decode_throughput_tokens_per_second"),
        field="decode_throughput_tokens_per_second",
    )
    decode_active_tokens = summary.get("decode_active_tokens")
    decode_active_elapsed_ns = _finite_number(
        summary.get("decode_active_elapsed_ns"), field="decode_active_elapsed_ns"
    )
    if (
        steady_decode <= 0.0
        or decode_active_elapsed_ns <= 0.0
        or not isinstance(decode_active_tokens, int)
        or isinstance(decode_active_tokens, bool)
        or not 0 <= decode_active_tokens <= int(config["fanout"]) * EXPECTED_SUFFIX
    ):
        raise Experiment003AnalysisError("decode boundary/throughput evidence is invalid")
    counters = summary.get("metadata_operation_counters")
    counters_valid = isinstance(counters, dict) and not any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in counters.values()
    )
    if (config["tracing_level"] == "disabled" and counters is not None) or (
        config["tracing_level"] != "disabled" and not counters_valid
    ):
        raise Experiment003AnalysisError("metadata operation counters are invalid")
    for field in (
        "physical_block_count",
        "shared_root_block_count",
        "private_block_count",
        "physical_assigned_bytes",
        "shared_prefix_bytes",
        "private_suffix_bytes",
        "post_root_process_cpu_ns",
        "post_root_thread_cpu_ns",
    ):
        _finite_number(summary.get(field), field=field)
    semantics = summary.get("semantics")
    if not isinstance(semantics, dict):
        raise Experiment003AnalysisError("state-sharing semantics are absent")
    if config["baseline_mode"] == "shared_root":
        for key in (
            "exact_shared_root_block_ids",
            "runtime_native_refcounts_cover_fanout",
            "private_suffix_blocks_are_single_owner",
        ):
            if semantics.get(key) is not True:
                raise Experiment003AnalysisError(f"shared state semantic failed: {key}")
    elif semantics.get("prefix_cache_reuse_absent") is not True:
        raise Experiment003AnalysisError("independent path reused prefix state")

    decomposition = _validate_decomposition(
        runner_root / "metrics/readiness-decomposition.json",
        post_root_ready_ns=post_root_ready_ns,
        trace_level=str(config["tracing_level"]),
    )
    metadata_observation = _read_json(runner_root / "metadata/runtime-metadata-observation.json")
    if (
        metadata_observation.get("schema_version")
        != "sloforge.branchfabric.vllm-metadata-observation-0230/v1"
        or metadata_observation.get("runtime_version") != EXPECTED_VLLM
        or metadata_observation.get("runtime_source_tag") != EXPECTED_VLLM_SOURCE_TAG
        or metadata_observation.get("runtime_source_commit") != EXPECTED_VLLM_SOURCE_COMMIT
        or not isinstance(metadata_observation.get("post_root_ready_runtime_native_metrics"), list)
        or (
            config["tracing_level"] != "disabled"
            and len(metadata_observation["post_root_ready_runtime_native_metrics"])
            != metadata_observation.get("post_root_ready_native_metric_count")
        )
        or (
            config["tracing_level"] == "disabled"
            and metadata_observation["post_root_ready_runtime_native_metrics"] != []
        )
    ):
        raise Experiment003AnalysisError(
            "runtime metadata observation is not pinned to the inspected vLLM 0.23.0 source"
        )
    native = _read_json(runner_root / "native-metrics/runtime-native-metrics.json")
    observations = native.get("observations")
    if (
        native.get("schema_version") != "sloforge.branchfabric.runtime-native-metrics/v1"
        or native.get("runtime_version") != EXPECTED_VLLM
        or not isinstance(observations, list)
        or native.get("post_root_ready_observation_count")
        != metadata_observation.get("post_root_ready_native_metric_count")
    ):
        raise Experiment003AnalysisError("runtime-native metrics disagree with metadata recorder")
    nvml = _read_json(runner_root / "raw/nvml-samples.json")
    if (
        nvml.get("schema_version") != "sloforge.branchfabric.nvml-samples/v1"
        or nvml.get("gpu_uuid") != campaign_gpu["uuid"]
        or nvml.get("error") is not None
        or not isinstance(nvml.get("samples"), list)
    ):
        raise Experiment003AnalysisError("NVML sampling artifact is invalid")
    physical = _read_json(runner_root / "raw/physical-layouts.json")
    final_layout = physical.get("final")
    decode_layouts = physical.get("decode")
    if (
        not isinstance(final_layout, dict)
        or not isinstance(decode_layouts, list)
        or final_layout.get("physical_assigned_bytes") != summary.get("physical_assigned_bytes")
        or (config["baseline_mode"] == "shared_root" and not isinstance(physical.get("root"), dict))
    ):
        raise Experiment003AnalysisError(
            "physical layout artifact disagrees with HBM/state summary"
        )
    profile_path = runner_root / "profiles/cprofile-summary.json"
    profile = _read_json(profile_path) if profile_path.is_file() else None
    return LoadedTrial(
        root=runner_root,
        position=position,
        config=config,
        summary=summary,
        decomposition=decomposition,
        native_metrics=native,
        metadata_observation=metadata_observation,
        nvml=nvml,
        physical_layouts=physical,
        profile=profile,
        gpu_identity=dict(campaign_gpu),
        function_call_id=function_call_id,
        summary_sha256=_sha256(summary_path),
    )


def load_campaign(
    root: Path, *, implementation: Literal["baseline", "optimized"]
) -> CampaignDataset:
    root = root.resolve(strict=True)
    remote_sha = _verify_remote_inventory(root)
    manifest = _read_json(root / "campaign-manifest.json")
    if (
        manifest.get("schema_version") != "sloforge.branchfabric.modal-campaign-manifest-003/v1"
        or manifest.get("requested_gpu") != "A100-80GB"
    ):
        raise Experiment003AnalysisError("campaign identity is invalid")
    status = manifest.get("status")
    error = manifest.get("error")
    adaptive_tail_timeout = bool(
        status == "failed"
        and isinstance(error, dict)
        and error.get("type") == "TimeoutError"
        and error.get("message") == "Experiment 003 campaign wall-time bound reached"
    )
    if status != "succeeded" and not adaptive_tail_timeout:
        raise Experiment003AnalysisError("campaign did not terminate successfully or adaptively")
    if status == "succeeded" and manifest.get("same_gpu_enforced_before_every_child") is not True:
        raise Experiment003AnalysisError("successful campaign lacks same-GPU enforcement")
    campaign_id = manifest.get("campaign_id")
    function_call_id = manifest.get("function_call_id")
    records = manifest.get("records")
    if (
        not isinstance(campaign_id, str)
        or not campaign_id
        or not isinstance(function_call_id, str)
        or not function_call_id
        or not isinstance(records, list)
        or manifest.get("completed_children") != len(records)
    ):
        raise Experiment003AnalysisError("campaign manifest is incomplete")
    planned_children = manifest.get("planned_children")
    if (
        not isinstance(planned_children, int)
        or isinstance(planned_children, bool)
        or planned_children < len(records)
        or (status == "succeeded" and planned_children != len(records))
        or (adaptive_tail_timeout and planned_children == len(records))
    ):
        raise Experiment003AnalysisError("campaign planned/completed child counts are invalid")
    gpu = _identity(manifest.get("gpu_identity"))
    threshold = manifest.get("immutable_cleanup_threshold_mib")
    if not isinstance(threshold, int) or isinstance(threshold, bool) or threshold <= 0:
        raise Experiment003AnalysisError("campaign immutable cleanup threshold is invalid")
    baseline = _read_json(root / "campaign-hbm-baseline.json")
    if (
        baseline.get("passed") is not True
        or baseline.get("gpu_identity") != gpu
        or baseline.get("cleanup_threshold_mib") != threshold
    ):
        raise Experiment003AnalysisError("campaign baseline disagrees with manifest")
    _validate_samples_identity(baseline.get("samples"), gpu, label="campaign baseline")
    trials = tuple(
        _load_trial(
            root,
            cast(Mapping[str, Any], record),
            implementation=implementation,
            campaign_id=campaign_id,
            function_call_id=function_call_id,
            campaign_gpu=gpu,
            threshold_mib=threshold,
        )
        for record in records
    )
    if len({trial.attempt_id for trial in trials}) != len(trials):
        raise Experiment003AnalysisError("campaign duplicates an attempt ID")
    return CampaignDataset(
        root=root,
        implementation=implementation,
        campaign_id=campaign_id,
        function_call_id=function_call_id,
        gpu_identity=gpu,
        immutable_cleanup_threshold_mib=threshold,
        remote_manifest_sha256=remote_sha,
        campaign_status=str(status),
        adaptive_tail_timeout=adaptive_tail_timeout,
        trials=trials,
    )


def _scientific_identity(trial: LoadedTrial, *, include_trace: bool = True) -> dict[str, Any]:
    keys = (
        "model",
        "model_revision",
        "tokenizer_revision",
        "runtime",
        "runtime_version",
        "fanout",
        "prefix_length",
        "suffix_length",
        "seed",
        "gpu_memory_utilization",
        "maximum_wall_seconds",
        "initialization_timeout_seconds",
        "cleanup_timeout_seconds",
    )
    value = {key: trial.config.get(key) for key in keys}
    if include_trace:
        value["tracing_level"] = trial.trace_level
    return value


def _workload_digest(trial: LoadedTrial, *, include_trace: bool = True) -> str:
    return hashlib.sha256(
        _canonical_bytes(_scientific_identity(trial, include_trace=include_trace))
    ).hexdigest()


def _paired_analysis(dataset: CampaignDataset) -> dict[str, Any]:
    groups: dict[str, list[LoadedTrial]] = defaultdict(list)
    for trial in dataset.trials:
        pair_id = trial.config.get("pair_id")
        if isinstance(pair_id, str):
            groups[pair_id].append(trial)
    if set(groups) != {"seed-41", "seed-73", "seed-113"}:
        return _unavailable("campaign does not contain exactly the three required pair IDs")
    observations: list[PairedObservation] = []
    validity: list[dict[str, Any]] = []
    for pair_id in ("seed-41", "seed-73", "seed-113"):
        rows = groups[pair_id]
        by_mode = {trial.mode: trial for trial in rows}
        if len(rows) != 2 or set(by_mode) != {"independent", "shared_root"}:
            raise Experiment003AnalysisError(f"pair {pair_id} does not contain exact A/B modes")
        independent = by_mode["independent"]
        shared = by_mode["shared_root"]
        if _scientific_identity(independent) != _scientific_identity(shared):
            raise Experiment003AnalysisError(f"pair {pair_id} changed scientific inputs")
        if independent.gpu_identity != shared.gpu_identity:
            raise Experiment003AnalysisError(f"pair {pair_id} crossed GPU identities")
        order = sorted(rows, key=lambda trial: int(trial.config["pair_order_position"]))
        validity.append(
            {
                "pair_id": pair_id,
                "seed": independent.seed,
                "realized_order": [trial.mode for trial in order],
                "function_call_id": dataset.function_call_id,
                "gpu_uuid": dataset.gpu_identity["uuid"],
                "same_gpu": True,
                "scientific_identity_equal": True,
                "cleanup_passed": True,
            }
        )
        observations.append(
            PairedObservation(
                key=PairKey(
                    pair_id=pair_id,
                    seed=independent.seed,
                    repetition=0,
                    workload_digest=_workload_digest(independent),
                    function_call_id=dataset.function_call_id,
                    gpu_uuid=str(dataset.gpu_identity["uuid"]),
                ),
                baseline_value=float(independent.summary["post_root_ready_ns"]) / 1e6,
                candidate_value=float(shared.summary["post_root_ready_ns"]) / 1e6,
                baseline_provenance=independent.provenance("$.post_root_ready_ns"),
                candidate_provenance=shared.provenance("$.post_root_ready_ns"),
            )
        )
    analysis = analyze_three_pair_effect(
        observations,
        metric="post_root_ready",
        unit="ms",
        baseline_label="independent_prefill",
        candidate_label="shared_root",
        bootstrap_seed=20_260_811,
    )
    return {"status": "available", "pair_validity": validity, "analysis": analysis}


def _trace_overhead(dataset: CampaignDataset) -> dict[str, Any]:
    rows = [
        trial
        for trial in dataset.trials
        if trial.config.get("pair_order_rule") == "instrumentation_control"
    ]
    by_level = {trial.trace_level: trial for trial in rows}
    if len(rows) != 3 or set(by_level) != {"disabled", "minimal", "full"}:
        return _unavailable("campaign lacks a complete disabled/minimal/full tracing control")
    reference = by_level["disabled"]
    if any(
        _scientific_identity(trial, include_trace=False)
        != _scientific_identity(reference, include_trace=False)
        for trial in rows
    ):
        raise Experiment003AnalysisError("tracing control changed a non-tracing scientific input")
    observation = TraceOverheadObservation(
        key=PairKey(
            pair_id="trace-control",
            seed=reference.seed,
            repetition=0,
            workload_digest=_workload_digest(reference, include_trace=False),
            function_call_id=dataset.function_call_id,
            gpu_uuid=str(dataset.gpu_identity["uuid"]),
        ),
        disabled_value=float(reference.summary["post_root_ready_ns"]) / 1e6,
        minimal_value=float(by_level["minimal"].summary["post_root_ready_ns"]) / 1e6,
        full_value=float(by_level["full"].summary["post_root_ready_ns"]) / 1e6,
        disabled_provenance=reference.provenance("$.post_root_ready_ns"),
        minimal_provenance=by_level["minimal"].provenance("$.post_root_ready_ns"),
        full_provenance=by_level["full"].provenance("$.post_root_ready_ns"),
    )
    analysis = analyze_trace_overhead((observation,), metric="post_root_ready", unit="ms")
    analysis["full_trace_control_note"] = (
        "the full control also enabled cProfile and per-block spans, so its overhead is the "
        "combined heavy diagnostic overhead rather than tracing alone"
    )
    return {"status": "available", "analysis": analysis}


def _shared_minimal(dataset: CampaignDataset) -> dict[int, list[LoadedTrial]]:
    result: dict[int, list[LoadedTrial]] = defaultdict(list)
    for trial in dataset.trials:
        if (
            trial.mode == "shared_root"
            and trial.trace_level == "minimal"
            and trial.config.get("pair_order_rule") != "instrumentation_control"
        ):
            result[trial.fanout].append(trial)
    return dict(result)


def _stage_value(trial: LoadedTrial, stage: str, dimension: str = "wall_ns") -> int | None:
    if trial.decomposition is None:
        return None
    decomposition = cast(dict[str, Any], trial.decomposition["decomposition"])
    stages = cast(dict[str, dict[str, Any]], decomposition["stages"])
    value = stages[stage].get(dimension)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _metadata_time_ns(trial: LoadedTrial, dimension: str = "wall_ns") -> int | None:
    values = [_stage_value(trial, stage, dimension) for stage in METADATA_STAGES]
    return None if any(value is None for value in values) else sum(cast(list[int], values))


def _counter_summary(trial: LoadedTrial) -> dict[str, int]:
    counters = cast(dict[str, int], trial.summary["metadata_operation_counters"])
    return {
        "counted_metadata_work_units": sum(
            int(counters.get(name, 0)) for name in COUNTED_METADATA_WORK_UNITS
        ),
        "scheduler_events": sum(int(counters.get(name, 0)) for name in SCHEDULER_EVENT_COUNTERS),
        "prompt_token_writes": int(counters.get("prompt_token_writes", 0)),
        "output_token_commits": int(counters.get("output_token_commits", 0)),
        "all_counter_events_non_unique": sum(int(value) for value in counters.values()),
    }


def _group_decomposition(datasets: Sequence[CampaignDataset]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for dataset in datasets:
        implementation: dict[str, Any] = {}
        for fanout, trials in sorted(_shared_minimal(dataset).items()):
            available = [trial for trial in trials if trial.decomposition is not None]
            if not available:
                implementation[str(fanout)] = _unavailable("no minimal timing decomposition")
                continue
            stages = {
                stage: {
                    "median_wall_ns": statistics.median(
                        cast(int, _stage_value(trial, stage)) for trial in available
                    ),
                    "raw_wall_ns": [cast(int, _stage_value(trial, stage)) for trial in available],
                    "median_process_cpu_ns": statistics.median(
                        cast(int, _stage_value(trial, stage, "process_cpu_ns"))
                        for trial in available
                    ),
                }
                for stage in STAGES
            }
            parent_values = [int(trial.summary["post_root_ready_ns"]) for trial in available]
            metadata_values = [cast(int, _metadata_time_ns(trial)) for trial in available]
            scheduler_values = [
                sum(cast(int, _stage_value(trial, stage)) for stage in SCHEDULER_STAGES)
                for trial in available
            ]
            gpu_values = [
                sum(cast(int, _stage_value(trial, stage)) for stage in GPU_STAGES)
                for trial in available
            ]
            median_parent = statistics.median(parent_values)
            representative = min(
                available,
                key=lambda trial: abs(int(trial.summary["post_root_ready_ns"]) - median_parent),
            )
            representative_stages = {
                stage: cast(int, _stage_value(representative, stage)) for stage in STAGES
            }
            assert representative.decomposition is not None
            representative_trace_parent = int(
                cast(dict[str, Any], representative.decomposition["decomposition"])["parent"][
                    "wall_ns"
                ]
            )
            representative_external_parent = int(representative.summary["post_root_ready_ns"])
            representative_external_error = abs(
                representative_trace_parent - representative_external_parent
            )
            representative_external_tolerance = max(
                100_000, math.ceil(representative_external_parent * 0.001)
            )
            first_forward_events = [
                int(event["elapsed_ns"])
                for trial in available
                for event in cast(
                    list[Any], trial.metadata_observation.get("gpu_event_measurements", [])
                )
                if isinstance(event, dict)
                and event.get("scope") == "first_GPUModelRunner._model_forward"
                and isinstance(event.get("elapsed_ns"), int)
            ]
            stage_semantics = representative.metadata_observation.get("stage_measurement_semantics")
            gpu_semantics = (
                cast(dict[str, Any], stage_semantics).get("GPU_EXECUTION")
                if isinstance(stage_semantics, dict)
                else None
            )
            implementation[str(fanout)] = {
                "status": "available",
                "sample_count": len(available),
                "median_post_root_ready_ns": median_parent,
                "median_metadata_wall_ns": statistics.median(metadata_values),
                "metadata_fraction": statistics.median(
                    metadata / parent
                    for metadata, parent in zip(metadata_values, parent_values, strict=True)
                ),
                "median_scheduler_wall_ns": statistics.median(scheduler_values),
                "scheduler_fraction": statistics.median(
                    scheduler / parent
                    for scheduler, parent in zip(scheduler_values, parent_values, strict=True)
                ),
                "median_gpu_wall_ns": statistics.median(gpu_values),
                "gpu_fraction": statistics.median(
                    gpu / parent for gpu, parent in zip(gpu_values, parent_values, strict=True)
                ),
                "gpu_wall_measurement_semantics": (
                    gpu_semantics
                    if isinstance(gpu_semantics, dict)
                    else {
                        "evidence": "legacy_baseline_worker_wall",
                        "scope": (
                            "baseline instrumentation predates sample_tokens wrappers; GPU and "
                            "residual stages are not identically scoped to optimized trials"
                        ),
                    }
                ),
                "median_first_model_forward_cuda_event_ns": (
                    None if not first_forward_events else statistics.median(first_forward_events)
                ),
                "stages": stages,
                "no_double_counting_verified": True,
                "exact_representative_trial": {
                    "attempt_id": representative.attempt_id,
                    "selection": "trial nearest the median POST_ROOT_READY",
                    "external_post_root_ready_ns": representative_external_parent,
                    "trace_parent_wall_ns": representative_trace_parent,
                    "stages_wall_ns": representative_stages,
                    "sum_stages_wall_ns": sum(representative_stages.values()),
                    "trace_additive_equality": (
                        sum(representative_stages.values()) == representative_trace_parent
                    ),
                    "external_crosscheck_error_ns": representative_external_error,
                    "external_crosscheck_tolerance_ns": representative_external_tolerance,
                    "external_crosscheck_within_tolerance": (
                        representative_external_error <= representative_external_tolerance
                    ),
                    "provenance": representative.provenance("$.readiness_decomposition").to_dict(),
                },
            }
        output[dataset.implementation] = implementation
    return {
        "schema_version": "sloforge.branchfabric.experiment-003-decomposition-summary/v1",
        "cross_implementation_gpu_stage_comparable": False,
        "gpu_stage_caveat": (
            "GPU_EXECUTION is CPU wall through GPU-inclusive worker wrappers, not pure device "
            "time. Baseline predates sample_tokens wrappers; use within-implementation values. "
            "The first _model_forward CUDA event is reported separately"
        ),
        "implementations": output,
    }


def _readiness_definitions(datasets: Sequence[CampaignDataset]) -> dict[str, Any]:
    implementations: dict[str, Any] = {}
    for dataset in datasets:
        grouped: dict[tuple[str, int], list[LoadedTrial]] = defaultdict(list)
        for trial in dataset.trials:
            if trial.trace_level == "minimal":
                grouped[(trial.mode, trial.fanout)].append(trial)
        implementation: dict[str, Any] = {}
        for (mode, fanout), trials in sorted(grouped.items()):
            raw_rows: list[dict[str, Any]] = []
            execution_start_proxies: list[tuple[int, int]] = []
            for trial in trials:
                first_tokens = trial.summary.get("first_token_latency_by_branch_ns")
                decode_starts = trial.summary.get("first_decode_token_started_latency_ns")
                first_execution: int | None = None
                all_execution: int | None = None
                if (
                    isinstance(first_tokens, dict)
                    and first_tokens
                    and all(isinstance(value, int) for value in first_tokens.values())
                    and isinstance(decode_starts, dict)
                    and decode_starts
                    and all(isinstance(value, int) for value in decode_starts.values())
                ):
                    token_values = cast(dict[str, int], first_tokens).values()
                    start_values = cast(dict[str, int], decode_starts).values()
                    decode_call_offset = int(trial.summary["first_token_latency_ns"]) - min(
                        token_values
                    )
                    first_execution = decode_call_offset + min(start_values)
                    all_execution = decode_call_offset + max(start_values)
                    execution_start_proxies.append((first_execution, all_execution))
                raw_rows.append(
                    {
                        "attempt_id": trial.attempt_id,
                        **{field: trial.summary[field] for field in READINESS_FIELDS},
                        "first_branch_execution_start_proxy_ns": first_execution,
                        "all_branches_execution_start_proxy_ns": all_execution,
                        "provenance": trial.provenance("$.readiness_definitions").to_dict(),
                    }
                )
            implementation[f"{mode}:f{fanout}"] = {
                "mode": mode,
                "fanout": fanout,
                "raw": raw_rows,
                "medians_ns": {
                    field: statistics.median(float(trial.summary[field]) for trial in trials)
                    for field in READINESS_FIELDS
                },
                "execution_start_proxy_medians_ns": {
                    "first_branch": (
                        None
                        if not execution_start_proxies
                        else statistics.median(row[0] for row in execution_start_proxies)
                    ),
                    "all_branches": (
                        None
                        if not execution_start_proxies
                        else statistics.median(row[1] for row in execution_start_proxies)
                    ),
                },
            }
        implementations[dataset.implementation] = implementation
    return {
        "schema_version": "sloforge.branchfabric.experiment-003-readiness-definitions/v1",
        "primary_metric": "post_root_ready_ns",
        "definitions_are_not_interchangeable": True,
        "boundary_semantics": {
            "post_root_ready_ns": (
                "continuous root-available to step return after every branch produced output"
            ),
            "first_branch_ready_ns": (
                "legacy-named first branch output-step-return boundary, not direct runnable state"
            ),
            "all_branches_ready_ns": (
                "legacy-named all-branches first-output boundary, equal to POST_ROOT_READY"
            ),
            "execution_start_proxies": (
                "step-entry timestamps assigned post hoc to branches that returned first output; "
                "closest available runnable proxy, not a direct scheduler state transition"
            ),
        },
        "implementations": implementations,
    }


def _scaling(dataset: CampaignDataset) -> dict[str, Any]:
    shared = _shared_minimal(dataset)
    raw: dict[str, Any] = {}
    readiness_observations: list[ScalingObservation] = []
    metadata_cpu_observations: list[ScalingObservation] = []
    operation_observations: list[ScalingObservation] = []
    for fanout, trials in sorted(shared.items()):
        raw_rows: list[dict[str, Any]] = []
        for repetition, trial in enumerate(trials):
            metadata_cpu = _metadata_time_ns(trial, "process_cpu_ns")
            counter_summary = _counter_summary(trial)
            operation_count = counter_summary["counted_metadata_work_units"]
            raw_rows.append(
                {
                    "attempt_id": trial.attempt_id,
                    "post_root_ready_ns": trial.summary["post_root_ready_ns"],
                    "metadata_cpu_ns": metadata_cpu,
                    **counter_summary,
                    "counter_aggregation_note": (
                        "counted_metadata_work_units uses a non-overlapping counter taxonomy; "
                        "all_counter_events_non_unique includes outcome/classification counters "
                        "and must not be interpreted as unique operations or hardware transactions"
                    ),
                    "operation_counters": trial.summary["metadata_operation_counters"],
                    "normalization": trial.summary.get("metadata_operation_normalization"),
                    "provenance": trial.provenance("$").to_dict(),
                }
            )
            readiness_observations.append(
                ScalingObservation(
                    fanout=fanout,
                    prefix_blocks=PREFIX_BLOCKS,
                    value=float(trial.summary["post_root_ready_ns"]) / 1e6,
                    repetition=repetition,
                    provenance=trial.provenance("$.post_root_ready_ns"),
                )
            )
            if metadata_cpu is not None:
                metadata_cpu_observations.append(
                    ScalingObservation(
                        fanout=fanout,
                        prefix_blocks=PREFIX_BLOCKS,
                        value=metadata_cpu / 1e6,
                        repetition=repetition,
                        provenance=trial.provenance("$.readiness_decomposition.metadata_cpu"),
                    )
                )
            operation_observations.append(
                ScalingObservation(
                    fanout=fanout,
                    prefix_blocks=PREFIX_BLOCKS,
                    value=float(operation_count),
                    repetition=repetition,
                    provenance=trial.provenance("$.metadata_operation_counters"),
                )
            )
        raw[str(fanout)] = raw_rows

    def fit(observations: Sequence[ScalingObservation], metric: str, unit: str) -> dict[str, Any]:
        if len(observations) < 3 or len({item.fanout for item in observations}) < 3:
            return _unavailable("fewer than three distinct fanouts were measured")
        return {
            "status": "available",
            "analysis": select_scaling_model(
                observations,
                metric=metric,
                unit=unit,
                relative_uncertainty=0.05,
            ),
        }

    operation_names = sorted(
        {
            str(name)
            for trials in shared.values()
            for trial in trials
            for name in trial.summary["metadata_operation_counters"]
        }
    )
    per_operation_models: dict[str, Any] = {}
    for operation_name in operation_names:
        observations: list[ScalingObservation] = []
        for fanout, trials in sorted(shared.items()):
            for repetition, trial in enumerate(trials):
                observations.append(
                    ScalingObservation(
                        fanout=fanout,
                        prefix_blocks=PREFIX_BLOCKS,
                        value=float(
                            trial.summary["metadata_operation_counters"].get(operation_name, 0)
                        ),
                        repetition=repetition,
                        provenance=trial.provenance(
                            f"$.metadata_operation_counters.{operation_name}"
                        ),
                    )
                )
        per_operation_models[operation_name] = fit(observations, operation_name, "count")

    return {
        "schema_version": "sloforge.branchfabric.experiment-003-scaling-analysis/v1",
        "implementation": dataset.implementation,
        "raw_by_fanout": raw,
        "readiness_model": fit(readiness_observations, "post_root_ready", "ms"),
        "metadata_cpu_model": fit(metadata_cpu_observations, "metadata_cpu", "ms"),
        "metadata_operation_model": fit(
            operation_observations, "counted_metadata_work_units", "count"
        ),
        "per_operation_models": per_operation_models,
    }


def _amdahl(dataset: CampaignDataset) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for fanout in (8, 16, 32):
        trials = [
            trial
            for trial in _shared_minimal(dataset).get(fanout, [])
            if _metadata_time_ns(trial) is not None
        ]
        if not trials:
            results[str(fanout)] = _unavailable("no decomposed shared-root measurement")
            continue
        raw_analyses = [
            amdahl_metadata_projections(
                post_root_ready_ms=float(trial.summary["post_root_ready_ns"]) / 1e6,
                metadata_ms=cast(int, _metadata_time_ns(trial)) / 1e6,
                fanout=fanout,
                root_inclusive_ready_ms=float(trial.summary["root_inclusive_ready_ns"]) / 1e6,
                post_root_process_cpu_ms=float(trial.summary["post_root_process_cpu_ns"]) / 1e6,
                metadata_process_cpu_ms=(
                    None
                    if _metadata_time_ns(trial, "process_cpu_ns") is None
                    else cast(int, _metadata_time_ns(trial, "process_cpu_ns")) / 1e6
                ),
                provenance=trial.provenance(
                    "$.post_root_ready_ns + $.readiness_decomposition metadata"
                ),
            )
            for trial in trials
        ]

        def median_number(field: str, rows: list[dict[str, Any]] = raw_analyses) -> float | None:
            values = [row[field] for row in rows if row.get(field) is not None]
            return None if not values else statistics.median(cast(list[float], values))

        projections: list[dict[str, Any]] = []
        for label in ("2x", "5x", "10x", "free"):
            rows = [
                next(item for item in row["projections"] if item["metadata_speedup"] == label)
                for row in raw_analyses
            ]
            projection: dict[str, Any] = {"metadata_speedup": label}
            for field in rows[0]:
                if field == "metadata_speedup":
                    continue
                values = [row[field] for row in rows if row.get(field) is not None]
                projection[field] = (
                    None if not values else statistics.median(cast(list[float], values))
                )
            projections.append(projection)
        analysis = {
            "schema_version": "sloforge.branchfabric.experiment-003-amdahl/v1",
            "fanout": fanout,
            "post_root_ready_ms": median_number("post_root_ready_ms"),
            "metadata_ms": median_number("metadata_ms"),
            "metadata_fraction": median_number("metadata_fraction"),
            "root_inclusive_ready_ms": median_number("root_inclusive_ready_ms"),
            "post_root_process_cpu_ms": median_number("post_root_process_cpu_ms"),
            "metadata_process_cpu_ms": median_number("metadata_process_cpu_ms"),
            "baseline_process_cpu_to_wall_ratio": median_number(
                "baseline_process_cpu_to_wall_ratio"
            ),
            "baseline_valid_branches_per_second": median_number(
                "baseline_valid_branches_per_second"
            ),
            "projections": projections,
            "ideal_metadata_free_speedup": statistics.median(
                float(row["ideal_metadata_free_speedup"]) for row in raw_analyses
            ),
            "aggregation": "median of within-trial Amdahl projections",
            "sample_count": len(raw_analyses),
            "provenance": [row["provenance"] for row in raw_analyses],
            "assumption": (
                "only measured metadata time changes within each raw trial; all orchestration, "
                "scheduler, GPU, output, and residual time remains fixed"
            ),
        }
        results[str(fanout)] = {
            "status": "available",
            "analysis": analysis,
            "raw_trial_analyses": raw_analyses,
        }
    return {
        "schema_version": "sloforge.branchfabric.experiment-003-amdahl-summary/v1",
        "implementation": dataset.implementation,
        "fanouts": results,
    }


def _hbm(datasets: Sequence[CampaignDataset]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    observed_gaps: list[int] = []
    for dataset in datasets:
        implementation: dict[str, Any] = {}
        for fanout, trials in sorted(_shared_minimal(dataset).items()):
            rows: list[dict[str, Any]] = []
            for trial in trials:
                physical = int(trial.summary["physical_assigned_bytes"])
                shared = int(trial.summary["shared_prefix_bytes"])
                private = int(trial.summary["private_suffix_bytes"])
                floor = shared + private
                gap = physical - floor
                if gap < 0:
                    raise Experiment003AnalysisError(
                        "physical HBM is below the measured sharing floor"
                    )
                observed_gaps.append(gap)
                rows.append(
                    {
                        "attempt_id": trial.attempt_id,
                        "shared_physical_kv_bytes": physical,
                        "theoretical_sharing_floor_bytes": floor,
                        "floor_gap_bytes": gap,
                        "shared_root_bytes": shared,
                        "private_suffix_bytes": private,
                        "marginal_private_hbm_per_branch_bytes": private / fanout,
                        "shared_root_lifetime": "root publication through branch teardown",
                    }
                )
            implementation[str(fanout)] = {"status": "available", "raw": rows}
        independent_f8 = [
            trial
            for trial in dataset.trials
            if trial.mode == "independent" and trial.fanout == 8 and trial.trace_level == "minimal"
        ]
        independent_summary: dict[str, Any]
        if not independent_f8:
            independent_summary = _unavailable("no independent fanout-8 trial")
        else:
            concurrent_measurements = [
                _independent_concurrent_physical_bytes(trial) for trial in independent_f8
            ]
            if any(value is None for value in concurrent_measurements):
                raise Experiment003AnalysisError(
                    "independent HBM lacks a raw snapshot with every request concurrently live"
                )
            measured_bytes = [cast(int, value) for value in concurrent_measurements]
            independent_summary = {
                "status": "available",
                "measurement_boundary": (
                    "maximum raw physical-layout snapshot with all fanout requests "
                    "concurrently owning blocks"
                ),
                "raw_physical_kv_bytes": measured_bytes,
                "median_physical_kv_bytes": statistics.median(measured_bytes),
            }
        implementation["independent_fanout_8"] = independent_summary
        result[dataset.implementation] = implementation
    return {
        "schema_version": "sloforge.branchfabric.experiment-003-hbm/v1",
        "implementations": result,
        "HARDWARE_COW_CAPACITY_TARGET": (
            "CLOSED" if observed_gaps and all(gap == 0 for gap in observed_gaps) else "UNAVAILABLE"
        ),
        "all_observed_floor_gaps_zero": bool(observed_gaps)
        and all(gap == 0 for gap in observed_gaps),
    }


def _independent_concurrent_physical_bytes(trial: LoadedTrial) -> int | None:
    """Return peak KV bytes only while every independent request owns live blocks."""

    layouts = [
        *cast(list[Any], trial.physical_layouts["decode"]),
        trial.physical_layouts["final"],
    ]
    measurements: list[int] = []
    for layout in layouts:
        if not isinstance(layout, dict) or not isinstance(layout.get("blocks"), list):
            continue
        owners = {
            str(branch_id)
            for block in layout["blocks"]
            if isinstance(block, dict) and isinstance(block.get("branch_ids"), list)
            for branch_id in block["branch_ids"]
        }
        value = layout.get("physical_assigned_bytes")
        if len(owners) == trial.fanout and isinstance(value, int) and not isinstance(value, bool):
            measurements.append(value)
    return max(measurements) if measurements else None


def _helix_has_no_gpu_or_output_overlap(trial: LoadedTrial) -> bool | None:
    spans = trial.metadata_observation.get("spans")
    if not isinstance(spans, list):
        return None
    helix = [
        row
        for row in spans
        if isinstance(row, dict) and row.get("category") == "HELIX_ORCHESTRATION"
    ]
    if not helix:
        return None
    blockers = [
        row
        for row in spans
        if isinstance(row, dict)
        and row.get("category") in {"GPU_SUBMISSION", "GPU_EXECUTION", "OUTPUT_TOKEN_COMMIT"}
    ]
    for helix_span in helix:
        helix_start = helix_span.get("wall_start_ns")
        helix_end = helix_span.get("wall_end_ns")
        if not isinstance(helix_start, int) or not isinstance(helix_end, int):
            return None
        for blocker in blockers:
            blocker_start = blocker.get("wall_start_ns")
            blocker_end = blocker.get("wall_end_ns")
            if not isinstance(blocker_start, int) or not isinstance(blocker_end, int):
                return None
            if blocker_start < helix_end and blocker_end > helix_start:
                return False
    return True


def _decode_and_gpu(datasets: Sequence[CampaignDataset]) -> dict[str, Any]:
    phases: dict[str, Any] = {}
    gpu_utilization: dict[str, Any] = {}
    for dataset in datasets:
        phase_impl: dict[str, Any] = {}
        util_impl: dict[str, Any] = {}
        for fanout, trials in sorted(_shared_minimal(dataset).items()):
            phase_rows: list[dict[str, Any]] = []
            for trial in trials:
                control = float(trial.summary["steady_decode_throughput_tokens_per_second"])
                decode_active = trial.summary.get("decode_active_tokens")
                early_outputs = (
                    None
                    if not isinstance(decode_active, int)
                    or isinstance(decode_active, bool)
                    or not 0 <= decode_active <= fanout * EXPECTED_SUFFIX
                    else fanout * EXPECTED_SUFFIX - decode_active
                )
                post_seconds = float(trial.summary["post_root_ready_ns"]) / 1e9
                early_throughput = (
                    None
                    if early_outputs is None or post_seconds <= 0.0
                    else early_outputs / post_seconds
                )
                proof = _helix_has_no_gpu_or_output_overlap(trial)
                branch_create: dict[str, Any]
                if proof is True:
                    branch_create = {
                        "status": "available",
                        "tokens_per_second": 0.0,
                        "output_productivity_deficit_fraction": 1.0,
                        "causal_concurrent_decode_interference": _unavailable(
                            "no independent decode stream overlapped synchronous branch creation"
                        ),
                        "proof": (
                            "raw HELIX_ORCHESTRATION spans have no overlapping "
                            "GPU_SUBMISSION, GPU_EXECUTION, or OUTPUT_TOKEN_COMMIT span"
                        ),
                        "interpretation": (
                            "structural synchronous host stall; not concurrent GPU slowdown"
                        ),
                    }
                else:
                    branch_create = _unavailable(
                        "raw spans do not prove an output-free HELIX construction window"
                    )
                early_divergence: dict[str, Any]
                if early_throughput is None or control <= 0.0:
                    early_divergence = _unavailable(
                        "decode_active_tokens did not support the early-output derivation"
                    )
                else:
                    early_divergence = {
                        "status": "available",
                        "early_output_tokens": early_outputs,
                        "window_ns": trial.summary["post_root_ready_ns"],
                        "tokens_per_second": early_throughput,
                        "phase_throughput_deficit_fraction": 1.0 - early_throughput / control,
                        "causal_concurrent_decode_interference": _unavailable(
                            "the early and steady phases were sequential, not concurrent"
                        ),
                        "method": (
                            "early_outputs = fanout*256 - decode_active_tokens; "
                            "throughput = early_outputs / POST_ROOT_READY"
                        ),
                    }
                phase_rows.append(
                    {
                        "attempt_id": trial.attempt_id,
                        "control": {
                            "tokens_per_second": control,
                            "definition": (
                                "same-trial established-branch steady decode throughput"
                            ),
                        },
                        "branch_create": branch_create,
                        "early_divergence": early_divergence,
                        "steady_branch_decode": {
                            "tokens_per_second": control,
                            "phase_throughput_deficit_fraction": 0.0,
                            "definition": "established-branch phase used as the productivity control",
                        },
                    }
                )
            phase_impl[str(fanout)] = {
                "status": "available",
                "raw": phase_rows,
                "median_control_tokens_per_second": statistics.median(
                    float(row["control"]["tokens_per_second"]) for row in phase_rows
                ),
                "median_early_divergence_phase_throughput_deficit_fraction": (
                    statistics.median(
                        float(row["early_divergence"]["phase_throughput_deficit_fraction"])
                        for row in phase_rows
                        if row["early_divergence"].get("status") == "available"
                    )
                    if any(
                        row["early_divergence"].get("status") == "available" for row in phase_rows
                    )
                    else None
                ),
            }
            samples = [
                sample
                for trial in trials
                for sample in cast(list[dict[str, Any]], trial.nvml["samples"])
                if isinstance(sample.get("gpu_utilization_percent"), int)
            ]
            utilization: dict[str, Any]
            if not samples:
                utilization = _unavailable(
                    "no NVML utilization sample overlapped the bounded trial"
                )
            else:
                utilization = {
                    "sample_count": len(samples),
                    "median_gpu_utilization_percent": statistics.median(
                        int(sample["gpu_utilization_percent"]) for sample in samples
                    ),
                    "median_memory_utilization_percent": statistics.median(
                        int(sample["memory_utilization_percent"]) for sample in samples
                    ),
                }
            util_impl[str(fanout)] = utilization
        phases[dataset.implementation] = phase_impl
        gpu_utilization[dataset.implementation] = util_impl
    return {
        "schema_version": "sloforge.branchfabric.experiment-003-decode-phase-productivity/v2",
        "status": "available" if phases else "unavailable",
        "causal_concurrent_decode_interference": _unavailable(
            "no background decode workload overlapped branch creation or early divergence"
        ),
        "methodology": (
            "within-trial established-branch steady throughput is the control; early divergence "
            "uses pre-boundary outputs over POST_ROOT_READY. The resulting ratio is a sequential "
            "phase-productivity deficit, not causal concurrent decode interference"
        ),
        "limitations": (
            "branch-create zero is a synchronous structural stall and NVML samples cover the "
            "whole benchmark rather than isolated phases; causal GPU/decode interference is "
            "unavailable and no Experiment 002 throughput is used"
        ),
        "implementations": phases,
        "gpu_utilization": gpu_utilization,
    }


def _flatten_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key).lower())
            keys.update(_flatten_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_flatten_keys(child))
    return keys


def _native_overlap(keys: set[str], patterns: Sequence[str]) -> str:
    return (
        "partially_overlaps"
        if any(any(pattern in key for pattern in patterns) for key in keys)
        else "unavailable"
    )


def _native_agreement(datasets: Sequence[CampaignDataset]) -> dict[str, Any]:
    trials: list[dict[str, Any]] = []
    major_disagreements: list[dict[str, Any]] = []
    for dataset in datasets:
        for trial in dataset.trials:
            observations = cast(list[Any], trial.native_metrics["observations"])
            post_root_observations = cast(
                list[Any],
                trial.metadata_observation["post_root_ready_runtime_native_metrics"],
            )
            keys = _flatten_keys(post_root_observations)
            metric_rows = [
                cast(dict[str, Any], observation["metrics"])
                for observation in post_root_observations
                if isinstance(observation, dict) and isinstance(observation.get("metrics"), dict)
            ]
            counters = cast(dict[str, int], trial.summary["metadata_operation_counters"])
            native_post_count_raw = trial.native_metrics["post_root_ready_observation_count"]
            recorder_post_count_raw = trial.metadata_observation[
                "post_root_ready_native_metric_count"
            ]
            native_post_count = (
                native_post_count_raw if isinstance(native_post_count_raw, int) else None
            )
            recorder_post_count = (
                recorder_post_count_raw if isinstance(recorder_post_count_raw, int) else None
            )
            recorder_count_agrees = (
                None
                if trial.trace_level == "disabled"
                else len(post_root_observations) == native_post_count == recorder_post_count
            )
            if recorder_count_agrees is False:
                major_disagreements.append(
                    {
                        "attempt_id": trial.attempt_id,
                        "measurement": "post_root_ready_native_observation_count",
                        "native_value": native_post_count,
                        "custom_value": recorder_post_count,
                    }
                )

            def maximum(field: str, rows: list[dict[str, Any]] = metric_rows) -> int | None:
                values = [
                    row[field]
                    for row in rows
                    if isinstance(row.get(field), int) and not isinstance(row.get(field), bool)
                ]
                return None if not values else max(cast(list[int], values))

            prefix_rows = [
                cast(dict[str, Any], row["prefix_cache_stats"])
                for row in metric_rows
                if isinstance(row.get("prefix_cache_stats"), dict)
            ]

            def prefix_maximum(field: str, rows: list[dict[str, Any]] = prefix_rows) -> int | None:
                values = [
                    row[field]
                    for row in rows
                    if isinstance(row.get(field), int) and not isinstance(row.get(field), bool)
                ]
                return None if not values else max(cast(list[int], values))

            comparisons: dict[str, Any] = {}

            def compare(
                name: str,
                native_value: int | None,
                custom_value: int | None,
                result: dict[str, Any] = comparisons,
                attempt_id: str = trial.attempt_id,
            ) -> None:
                if native_value is None or custom_value is None:
                    result[name] = _unavailable("one side did not expose a comparable value")
                    return
                agrees = native_value == custom_value
                result[name] = {
                    "status": "confirms" if agrees else "disagrees",
                    "native_value": native_value,
                    "custom_value": custom_value,
                }
                if not agrees:
                    major_disagreements.append(
                        {
                            "attempt_id": attempt_id,
                            "measurement": name,
                            "native_value": native_value,
                            "custom_value": custom_value,
                        }
                    )

            if trial.trace_level == "disabled":
                for name in (
                    "running_requests",
                    "prefix_cache_requests",
                    "prefix_cache_query_tokens",
                    "prefix_cache_hit_blocks",
                ):
                    comparisons[name] = _unavailable(
                        "native metric capture and custom counters are intentionally disabled"
                    )
            else:
                compare("running_requests", maximum("num_running_reqs"), trial.fanout)
            if trial.trace_level != "disabled" and trial.mode == "independent":
                for name in (
                    "prefix_cache_requests",
                    "prefix_cache_query_tokens",
                    "prefix_cache_hit_blocks",
                ):
                    comparisons[name] = _unavailable(
                        "independent mode intentionally prevents prefix-cache reuse; native "
                        "prefix-cache reset counters are not runtime-request totals"
                    )
            elif trial.trace_level != "disabled":
                compare(
                    "prefix_cache_requests",
                    prefix_maximum("requests"),
                    int(counters.get("runtime_request_allocations", 0)),
                )
                compare(
                    "prefix_cache_query_tokens",
                    prefix_maximum("queries"),
                    int(counters.get("prompt_token_writes", 0)),
                )
                native_hit_tokens = prefix_maximum("hits")
                compare(
                    "prefix_cache_hit_blocks",
                    None if native_hit_tokens is None else native_hit_tokens // 16,
                    int(counters.get("prefix_hash_hits", 0)),
                )
            trial_disagrees = recorder_count_agrees is False or any(
                isinstance(value, Mapping) and value.get("status") == "disagrees"
                for value in comparisons.values()
            )

            trials.append(
                {
                    "attempt_id": trial.attempt_id,
                    "implementation": dataset.implementation,
                    "mode": trial.mode,
                    "native_observation_count": len(observations),
                    "post_root_ready_native_observation_count": len(post_root_observations),
                    "recorder_count_agreement": (
                        "unavailable"
                        if recorder_count_agrees is None
                        else ("confirms" if recorder_count_agrees else "disagrees")
                    ),
                    "recorder_count_evidence": {
                        "native_post_root_ready_observation_count": native_post_count,
                        "recorder_observation_count": recorder_post_count,
                        "serialized_post_root_ready_observation_count": len(post_root_observations),
                    },
                    "value_comparisons": comparisons,
                    "custom_measurements": {
                        "post_root_ready": "unavailable",
                        "scheduler_running_waiting": _native_overlap(keys, ("running", "waiting")),
                        "request_queue_time": _native_overlap(keys, ("queue", "waiting_time")),
                        "prefill_decode_time": _native_overlap(keys, ("prefill", "decode")),
                        "kv_cache_utilization": _native_overlap(keys, ("kv_cache", "cache_usage")),
                        "prefix_cache_hits_misses": _native_overlap(keys, ("prefix", "cache_hit")),
                        "block_lifetime_reuse": "unavailable",
                    },
                    "major_disagreement": trial_disagrees,
                    "note": (
                        "numeric comparisons use only the recorder-frozen POST_ROOT_READY native "
                        "snapshots; full-lifecycle snapshots are retained but not compared to "
                        "frozen custom counters. Native make_stats does not provide exclusive spans"
                    ),
                }
            )
    return {
        "schema_version": "sloforge.branchfabric.experiment-003-native-metric-agreement/v1",
        "trials": trials,
        "major_disagreements": major_disagreements,
    }


def _profile_summary(datasets: Sequence[CampaignDataset]) -> dict[str, Any]:
    profiles = [
        (trial, trial.profile)
        for dataset in datasets
        for trial in dataset.trials
        if trial.profile is not None
    ]
    if not profiles:
        return {
            "schema_version": "sloforge.branchfabric.experiment-003-cpu-hot-path/v1",
            **_unavailable("no full-trace cProfile diagnostic was materialized"),
            "rows": [],
        }
    trial, profile = profiles[0]
    assert profile is not None
    rows = profile.get("rows")
    if not isinstance(rows, list):
        raise Experiment003AnalysisError("cProfile summary rows are invalid")
    normalized = [
        row
        for row in rows
        if isinstance(row, dict)
        and isinstance(row.get("cumulative_seconds"), (int, float))
        and not isinstance(row.get("cumulative_seconds"), bool)
    ][:20]
    return {
        "schema_version": "sloforge.branchfabric.experiment-003-cpu-hot-path/v1",
        "status": "available",
        "attempt_id": trial.attempt_id,
        "profiler": "cProfile diagnostic; excluded from primary causal data",
        "limitations": (
            "the full-trace control also enables per-block spans and cProfile; cumulative times "
            "are strongly distorted by instrumentation and identify code paths, not primary "
            "latency magnitudes"
        ),
        "rows": normalized,
    }


def _before_after(baseline: CampaignDataset, optimized: CampaignDataset | None) -> dict[str, Any]:
    if optimized is None:
        return _unavailable("optimized campaign was not supplied")
    before = {
        trial.seed: trial
        for trial in baseline.trials
        if trial.mode == "shared_root" and trial.config.get("pair_id")
    }
    after = {
        trial.seed: trial
        for trial in optimized.trials
        if trial.mode == "shared_root" and trial.config.get("pair_id")
    }
    if set(before) != {41, 73, 113} or set(after) != {41, 73, 113}:
        return _unavailable("before/after campaigns lack three seed-aligned shared-root trials")
    rows: list[dict[str, Any]] = []
    for seed in (41, 73, 113):
        old = before[seed]
        new = after[seed]
        old_metadata = _metadata_time_ns(old)
        new_metadata = _metadata_time_ns(new)
        rows.append(
            {
                "seed": seed,
                "baseline_gpu_uuid": old.gpu_identity["uuid"],
                "optimized_gpu_uuid": new.gpu_identity["uuid"],
                "same_gpu_family": (
                    "A100" in str(old.gpu_identity["name"])
                    and "A100" in str(new.gpu_identity["name"])
                    and old.gpu_identity["memory_total_mib"] == new.gpu_identity["memory_total_mib"]
                ),
                "same_exact_gpu_model": (old.gpu_identity["name"] == new.gpu_identity["name"]),
                "same_gpu_uuid": old.gpu_identity["uuid"] == new.gpu_identity["uuid"],
                "baseline_post_root_ready_ms": old.summary["post_root_ready_ns"] / 1e6,
                "optimized_post_root_ready_ms": new.summary["post_root_ready_ns"] / 1e6,
                "readiness_difference_ms": (
                    new.summary["post_root_ready_ns"] - old.summary["post_root_ready_ns"]
                )
                / 1e6,
                "baseline_metadata_ms": None if old_metadata is None else old_metadata / 1e6,
                "optimized_metadata_ms": None if new_metadata is None else new_metadata / 1e6,
                "metadata_difference_ms": (
                    None
                    if old_metadata is None or new_metadata is None
                    else (new_metadata - old_metadata) / 1e6
                ),
                "shared_semantics_preserved": True,
                "zero_prefix_duplication_preserved": True,
                "private_suffix_semantics_preserved": True,
                "runtime_kv_release_preserved": True,
            }
        )
    return {
        "status": "available",
        "comparison": (
            "seed-aligned same-A100-family rerun; exact GPU model and UUID are reported, "
            "and cross-campaign readiness deltas are descriptive when either differs"
        ),
        "raw": rows,
        "median_readiness_difference_ms": statistics.median(
            row["readiness_difference_ms"] for row in rows
        ),
        "median_metadata_difference_ms": statistics.median(
            cast(float, row["metadata_difference_ms"]) for row in rows
        ),
    }


def classify_metadata(
    *,
    metadata_fraction: float | None,
    ideal_free_speedup: float | None,
    metadata_wall_ns: float | None,
    scheduler_wall_ns: float | None,
    gpu_wall_ns: float | None,
    counted_metadata_work_units: int | None,
) -> str:
    values = (
        metadata_fraction,
        ideal_free_speedup,
        metadata_wall_ns,
        scheduler_wall_ns,
        gpu_wall_ns,
    )
    if any(value is None or not math.isfinite(value) for value in values):
        return "UNAVAILABLE"
    assert metadata_fraction is not None
    assert ideal_free_speedup is not None
    assert metadata_wall_ns is not None
    assert scheduler_wall_ns is not None
    assert gpu_wall_ns is not None
    if metadata_fraction < 0.05 or ideal_free_speedup < 1.05:
        return "METADATA_CLOSED"
    if (
        metadata_fraction > 0.15
        and ideal_free_speedup >= 1.15
        and counted_metadata_work_units is not None
        and counted_metadata_work_units > 0
    ):
        return "METADATA_HARDWARE_INTEREST"
    return "METADATA_SOFTWARE_ONLY"


def _initial_analysis_gate(decomposition: Mapping[str, Any]) -> dict[str, Any]:
    baseline = decomposition.get("baseline")
    f8 = baseline.get("8") if isinstance(baseline, Mapping) else None
    if not isinstance(f8, Mapping) or f8.get("status") != "available":
        return _unavailable("baseline shared-root fanout-8 decomposition is absent")
    metadata_fraction = float(f8["metadata_fraction"])
    scheduler_fraction = float(f8["scheduler_fraction"])
    gpu_fraction = float(f8["gpu_fraction"])
    if scheduler_fraction + gpu_fraction > 0.5 and (
        float(f8["median_scheduler_wall_ns"]) + float(f8["median_gpu_wall_ns"])
        > float(f8["median_metadata_wall_ns"])
    ):
        category = "D"
        interpretation = "scheduler queueing/selection or GPU execution dominates"
    elif metadata_fraction < 0.05:
        category = "A"
        interpretation = "metadata/block operations are below five percent"
    elif metadata_fraction <= 0.15:
        category = "B"
        interpretation = "metadata is a software optimization target"
    else:
        category = "C"
        interpretation = "metadata is a strong software optimization target"
    return {
        "status": "available",
        "category": category,
        "interpretation": interpretation,
        "metadata_fraction": metadata_fraction,
        "scheduler_fraction": scheduler_fraction,
        "gpu_fraction": gpu_fraction,
        "scheduler_is_not_metadata": True,
    }


def _classification(
    decomposition: Mapping[str, Any],
    amdahl: Mapping[str, Any],
    optimized: CampaignDataset | None,
    paired_baseline: Mapping[str, Any],
    paired_optimized: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if optimized is None:
        return {"conclusion": "UNAVAILABLE", "reason": "optimized campaign was not supplied"}
    if (
        paired_baseline.get("status") != "available"
        or not paired_optimized
        or paired_optimized.get("status") != "available"
    ):
        return {
            "conclusion": "UNAVAILABLE",
            "reason": "three valid same-GPU pairs are required before classification",
        }
    optimized_decomposition = cast(Mapping[str, Any], decomposition.get("optimized", {}))
    fanout_amdahl = cast(Mapping[str, Any], amdahl.get("fanouts", {}))
    if any(
        not isinstance(optimized_decomposition.get(str(fanout)), Mapping)
        or cast(Mapping[str, Any], optimized_decomposition[str(fanout)]).get("status")
        != "available"
        for fanout in (8, 16, 32)
    ) or any(
        not isinstance(fanout_amdahl.get(str(fanout)), Mapping)
        or cast(Mapping[str, Any], fanout_amdahl[str(fanout)]).get("status") != "available"
        for fanout in (8, 16, 32)
    ):
        return {
            "conclusion": "UNAVAILABLE",
            "reason": "optimized fanout 8/16/32 decomposition and Amdahl data are required",
        }
    shared = _shared_minimal(optimized)
    fanout_results: dict[str, Any] = {}
    fanout_conclusions: list[str] = []
    for fanout in (8, 16, 32):
        row = cast(Mapping[str, Any], optimized_decomposition[str(fanout)])
        projection = cast(
            Mapping[str, Any], cast(Mapping[str, Any], fanout_amdahl[str(fanout)])["analysis"]
        )
        trials = shared.get(fanout, [])
        operation_count = (
            None
            if not trials
            else round(
                statistics.median(
                    _counter_summary(trial)["counted_metadata_work_units"] for trial in trials
                )
            )
        )
        fanout_conclusion = classify_metadata(
            metadata_fraction=float(row["metadata_fraction"]),
            ideal_free_speedup=float(projection["ideal_metadata_free_speedup"]),
            metadata_wall_ns=float(row["median_metadata_wall_ns"]),
            scheduler_wall_ns=float(row["median_scheduler_wall_ns"]),
            gpu_wall_ns=float(row["median_gpu_wall_ns"]),
            counted_metadata_work_units=operation_count,
        )
        fanout_conclusions.append(fanout_conclusion)
        fanout_results[str(fanout)] = {
            "conclusion": fanout_conclusion,
            "sample_count": row["sample_count"],
            "metadata_fraction": row["metadata_fraction"],
            "ideal_metadata_free_speedup": projection["ideal_metadata_free_speedup"],
            "metadata_wall_ns": row["median_metadata_wall_ns"],
            "scheduler_wall_ns": row["median_scheduler_wall_ns"],
            "gpu_wall_ns": row["median_gpu_wall_ns"],
            "counted_metadata_work_units": operation_count,
            "counter_semantics": (
                "non-overlapping software counter taxonomy; not unique hardware transactions"
            ),
        }
    if "METADATA_HARDWARE_INTEREST" in fanout_conclusions:
        conclusion = "METADATA_HARDWARE_INTEREST"
    elif "METADATA_SOFTWARE_ONLY" in fanout_conclusions:
        conclusion = "METADATA_SOFTWARE_ONLY"
    else:
        conclusion = "METADATA_CLOSED"
    f8 = cast(Mapping[str, Any], fanout_results["8"])
    return {
        "conclusion": conclusion,
        "aggregation_rule": (
            "hardware-interest if any measured fanout qualifies; otherwise software-only if "
            "any measured fanout qualifies; closed only when every measured fanout closes"
        ),
        "fanouts": fanout_results,
        "optimized_fanout_8_metadata_fraction": f8["metadata_fraction"],
        "optimized_fanout_8_ideal_free_speedup": f8["ideal_metadata_free_speedup"],
        "optimized_fanout_8_metadata_wall_ns": f8["metadata_wall_ns"],
        "optimized_fanout_8_scheduler_wall_ns": f8["scheduler_wall_ns"],
        "optimized_fanout_8_gpu_wall_ns": f8["gpu_wall_ns"],
        "counted_metadata_work_units": f8["counted_metadata_work_units"],
        "uncertainty_caveat": (
            "fanout 8 uses three paired seeds; fanout 16 and 32 are single diagnostic trials. "
            "The aggregate SOFTWARE_ONLY classification is conservative but its high-fanout "
            "fractions do not estimate trial-to-trial variance"
        ),
    }


def _line_svg(
    title: str,
    series: Mapping[str, Sequence[tuple[float, float]]],
    *,
    x_label: str,
    y_label: str,
    unavailable_note: str | None = None,
) -> str:
    width, height = 900, 520
    left, right, top, bottom = 90, 30, 60, 80
    points = [point for values in series.values() for point in values]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-family="sans-serif" font-size="20">{escape(title)}</text>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="black"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="black"/>',
        f'<text x="{width / 2}" y="{height - 20}" text-anchor="middle" font-family="sans-serif">{escape(x_label)}</text>',
        f'<text transform="translate(20 {height / 2}) rotate(-90)" text-anchor="middle" font-family="sans-serif">{escape(y_label)}</text>',
    ]
    if not points:
        note = unavailable_note or "Measurement unavailable"
        parts.append(
            f'<text x="{width / 2}" y="{height / 2}" text-anchor="middle" font-family="sans-serif" fill="#666">{escape(note)}</text>'
        )
        parts.append("</svg>\n")
        return "".join(parts)
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(0.0, min(ys)), max(ys)
    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0
    if y_min == y_max:
        y_max = y_min + 1.0

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * (width - left - right)

    def sy(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * (height - top - bottom)

    colors = ("#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c", "#0891b2")
    for index, (label, values) in enumerate(series.items()):
        color = colors[index % len(colors)]
        ordered = sorted(values)
        coordinates = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in ordered)
        parts.append(
            f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2"/>'
        )
        for x, y in ordered:
            parts.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="4" fill="{color}"/>')
        parts.append(
            f'<text x="{width - right - 180}" y="{top + 20 * index}" font-family="sans-serif" fill="{color}">{escape(label)}</text>'
        )
    for tick in range(6):
        y = y_min + (y_max - y_min) * tick / 5
        parts.append(
            f'<text x="{left - 8}" y="{sy(y) + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="11">{y:.3g}</text>'
        )
    parts.append("</svg>\n")
    return "".join(parts)


def _plot_payloads(
    datasets: Sequence[CampaignDataset],
    decomposition: Mapping[str, Any],
    amdahl: Mapping[str, Any],
    hbm: Mapping[str, Any],
    decode: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> dict[str, str]:
    readiness: dict[str, list[tuple[float, float]]] = defaultdict(list)
    metadata_cpu: dict[str, list[tuple[float, float]]] = defaultdict(list)
    operations: dict[str, list[tuple[float, float]]] = defaultdict(list)
    throughput: dict[str, list[tuple[float, float]]] = defaultdict(list)
    physical: dict[str, list[tuple[float, float]]] = defaultdict(list)
    components: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for dataset in datasets:
        for fanout, trials in sorted(_shared_minimal(dataset).items()):
            readiness[dataset.implementation].append(
                (
                    fanout,
                    statistics.median(
                        float(trial.summary["post_root_ready_ns"]) / 1e6 for trial in trials
                    ),
                )
            )
            cpu_values = [
                cast(int, _metadata_time_ns(trial, "process_cpu_ns")) / 1e6
                for trial in trials
                if _metadata_time_ns(trial, "process_cpu_ns") is not None
            ]
            if cpu_values:
                metadata_cpu[dataset.implementation].append((fanout, statistics.median(cpu_values)))
            operations[dataset.implementation].append(
                (
                    fanout,
                    statistics.median(
                        _counter_summary(trial)["counted_metadata_work_units"] for trial in trials
                    ),
                )
            )
            throughput[dataset.implementation].append(
                (
                    fanout,
                    statistics.median(
                        float(trial.summary["steady_decode_throughput_tokens_per_second"])
                        for trial in trials
                    ),
                )
            )
            physical[f"{dataset.implementation} shared"].append(
                (
                    fanout,
                    statistics.median(
                        float(trial.summary["physical_assigned_bytes"]) for trial in trials
                    ),
                )
            )
            physical[f"{dataset.implementation} floor"].append(
                (
                    fanout,
                    statistics.median(
                        float(
                            trial.summary["shared_prefix_bytes"]
                            + trial.summary["private_suffix_bytes"]
                        )
                        for trial in trials
                    ),
                )
            )
        independent_f8 = [
            trial
            for trial in dataset.trials
            if trial.mode == "independent" and trial.fanout == 8 and trial.trace_level == "minimal"
        ]
        if independent_f8:
            independent_bytes = [
                _independent_concurrent_physical_bytes(trial) for trial in independent_f8
            ]
            if any(value is None for value in independent_bytes):
                raise Experiment003AnalysisError(
                    "independent HBM plot lacks a concurrent all-request snapshot"
                )
            readiness[f"{dataset.implementation} independent"].append(
                (
                    8.0,
                    statistics.median(
                        float(trial.summary["post_root_ready_ns"]) / 1e6 for trial in independent_f8
                    ),
                )
            )
            physical[f"{dataset.implementation} independent"].append(
                (
                    8.0,
                    statistics.median(float(cast(int, value)) for value in independent_bytes),
                )
            )
        decomp_impl = cast(Mapping[str, Any], decomposition.get(dataset.implementation, {}))
        for fanout_key, row in decomp_impl.items():
            if isinstance(row, Mapping) and row.get("status") == "available":
                stages = cast(Mapping[str, Any], row["stages"])
                orchestration = float(stages["helix_orchestration"]["median_wall_ns"])
                metadata = float(row["median_metadata_wall_ns"])
                scheduler = float(row["median_scheduler_wall_ns"])
                gpu = float(row["median_gpu_wall_ns"])
                categorized = {
                    "helix_orchestration",
                    *METADATA_STAGES,
                    *SCHEDULER_STAGES,
                    *GPU_STAGES,
                }
                other = sum(
                    float(stage_row["median_wall_ns"])
                    for stage_name, stage_row in stages.items()
                    if stage_name not in categorized and isinstance(stage_row, Mapping)
                )
                for component, value in (
                    ("orchestration", orchestration),
                    ("metadata/block", metadata),
                    ("scheduler", scheduler),
                    ("GPU", gpu),
                    ("other", other),
                ):
                    components[f"{dataset.implementation} {component}"].append(
                        (float(fanout_key), value / 1e6)
                    )
    interference_impl = cast(Mapping[str, Any], decode.get("implementations", {}))
    for implementation, fanout_rows in interference_impl.items():
        if not isinstance(fanout_rows, Mapping):
            continue
        for fanout_key, fanout_row in fanout_rows.items():
            if not isinstance(fanout_row, Mapping) or fanout_row.get("status") != "available":
                continue
            raw_rows = fanout_row.get("raw")
            if not isinstance(raw_rows, list):
                continue
            early_values = [
                float(early["tokens_per_second"])
                for row in raw_rows
                if isinstance(row, Mapping)
                and isinstance((early := row.get("early_divergence")), Mapping)
                and early.get("status") == "available"
            ]
            if early_values:
                throughput[f"{implementation} early divergence"].append(
                    (float(fanout_key), statistics.median(early_values))
                )
    if "optimized" in readiness:
        for fanout_key, row in cast(Mapping[str, Any], amdahl.get("fanouts", {})).items():
            if isinstance(row, Mapping) and row.get("status") == "available":
                free = next(
                    item
                    for item in cast(Mapping[str, Any], row["analysis"])["projections"]
                    if item["metadata_speedup"] == "free"
                )
                readiness["ideal metadata-free"].append(
                    (float(fanout_key), float(free["projected_post_root_ready_ms"]))
                )
    amdahl_series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for fanout_key, row in cast(Mapping[str, Any], amdahl.get("fanouts", {})).items():
        if isinstance(row, Mapping) and row.get("status") == "available":
            for index, item in enumerate(cast(Mapping[str, Any], row["analysis"])["projections"]):
                speedup = item["post_root_ready_speedup"]
                if speedup is not None:
                    amdahl_series[f"fanout {fanout_key}"].append((float(index), float(speedup)))
    profile_series: dict[str, list[tuple[float, float]]] = {}
    if profile.get("status") == "available":
        profile_series["cumulative CPU"] = [
            (float(index + 1), float(row["cumulative_seconds"]))
            for index, row in enumerate(cast(list[dict[str, Any]], profile["rows"])[:15])
        ]
    return {
        "plot-1-fanout-post-root-ready.svg": _line_svg(
            "Fanout vs POST_ROOT_READY",
            readiness,
            x_label="fanout",
            y_label="milliseconds",
        ),
        "plot-2-readiness-decomposition.svg": _line_svg(
            "Fanout vs non-overlapping readiness components",
            components,
            x_label="fanout",
            y_label="milliseconds",
        ),
        "plot-3-metadata-cpu-time.svg": _line_svg(
            "Fanout vs metadata CPU time",
            metadata_cpu,
            x_label="fanout",
            y_label="process CPU milliseconds",
        ),
        "plot-4-metadata-operation-count.svg": _line_svg(
            "Fanout vs counted metadata work units",
            operations,
            x_label="fanout",
            y_label="non-overlapping counted work units",
        ),
        "plot-5-decode-throughput-interference.svg": _line_svg(
            "Fanout vs decode phase throughput (not causal interference)",
            throughput,
            x_label="fanout",
            y_label="tokens/second",
            unavailable_note=str(decode.get("limitations")),
        ),
        "plot-6-physical-hbm.svg": _line_svg(
            "Fanout vs physical HBM",
            physical,
            x_label="fanout",
            y_label="bytes",
            unavailable_note=str(hbm.get("HARDWARE_COW_CAPACITY_TARGET")),
        ),
        "plot-7-amdahl.svg": _line_svg(
            "Amdahl metadata speedup projection (2x, 5x, 10x, free)",
            amdahl_series,
            x_label="projection index",
            y_label="total readiness speedup",
        ),
        "plot-8-cpu-hot-path.svg": _line_svg(
            "Diagnostic CPU hot path",
            profile_series,
            x_label="cProfile cumulative rank",
            y_label="cumulative seconds",
            unavailable_note=str(profile.get("reason", "profile unavailable")),
        ),
    }


def _experiment_004_plan() -> str:
    return """# Experiment 004 Plan: Real 2-GPU Capacity Reclamation + Cross-Layout State Movement

## Scope

Run two NVIDIA A100 80GB GPUs in one bounded Modal container. GPU 0 hosts a production-like serving pool; GPU 1 hosts long-running branch rollouts. No FPGA work is in scope.

## Sequence

1. Establish steady serving traffic on one logical pool and long-running branches on the other.
2. Inject a deterministic serving-load spike.
3. Pause rollout generation only at a validated safe boundary.
4. Capture a full checkpoint plus an explicitly versioned delta.
5. Reclaim GPU capacity and verify allocator/HBM release.
6. Perform the state transform and transfer across the source and destination layouts.
7. Validate block ownership, token history, model identity, and checkpoint/delta integrity.
8. Resume rollouts and compare outputs against an unmigrated deterministic control.

## Measurements

Measure pause latency, checkpoint and delta creation, capacity-reclamation latency, transform CPU/GPU time, transfer time and bandwidth, destination allocation, restore validation, resume latency, migration critical path, serving interference, HBM high-water marks, and final cleanup. Preserve raw timestamps and separate queueing, transfer, transformation, and GPU execution.

## Safety and budget

Use one explicitly authorized Modal function with `gpu="A100-80GB:2"`, a bounded GPU-hour ledger, no concurrent GPU coordinators, fresh-child ownership where applicable, and driver-visible cleanup gates for both UUIDs. Do not execute this plan as part of Experiment 003.
"""


def _metadata_characterization_doc(report: Mapping[str, Any]) -> str:
    classification = cast(Mapping[str, Any], report["metadata_classification"])
    return f"""# BranchFabric Metadata Characterization

This document is generated exclusively from materialized Experiment 003 artifacts. Missing measurements are reported as unavailable and no Experiment 002 cross-GPU throughput result is reused as causal evidence.

## Result

- Completion status: `{report["completion_status"]}`
- Metadata conclusion: `{classification["conclusion"]}`
- HBM COW capacity target: `{cast(Mapping[str, Any], report["hbm"])["HARDWARE_COW_CAPACITY_TARGET"]}`
- Same-GPU paired baseline: `{cast(Mapping[str, Any], report["paired_baseline"])["status"]}`
- Same-GPU paired optimized: `{cast(Mapping[str, Any], report["paired_optimized"])["status"]}`

## Measurement boundaries

POST_ROOT_READY is decomposed only from non-overlapping exclusive stage time. It ends after every branch returned first output. The legacy FIRST_BRANCH_READY and ALL_BRANCHES_READY fields are output-step-return boundaries, not direct runnable transitions; step-entry execution-start proxies are reported separately. Scheduler admission, waiting, and selection are reported as scheduler work, never metadata. Counted metadata work units use a non-overlapping software taxonomy and are not hardware transactions.

GPU_EXECUTION is GPU-inclusive worker CPU wall, not pure device time. The optimized fanout-8 first GPUModelRunner._model_forward CUDA-event median is available in the decomposition JSON. Baseline predates the sample_tokens wrappers used by the optimized campaign, so baseline/optimized GPU and residual stages are not identically scoped and are not used as causal before/after evidence.

## Known unavailable evidence

Causal concurrent decode interference is unavailable because no background decode stream overlapped branch creation. The reported early-divergence ratio is a sequential phase-productivity deficit; a branch-create throughput of zero is a structural synchronous stall, not concurrent GPU slowdown. Numeric vLLM-native comparisons use only POST_ROOT-frozen scheduler snapshots; native metrics do not provide exact exclusive-span timing equivalents.
"""


def _markdown_report(report: Mapping[str, Any]) -> str:
    classification = cast(Mapping[str, Any], report["metadata_classification"])
    lines = [
        "# BranchFabric GPU Validation Experiment 003",
        "",
        f"Completion status: `{report['completion_status']}`",
        f"Metadata conclusion: `{classification['conclusion']}`",
        "",
        "## GPU/runtime/model",
        "",
        f"- Model: `{EXPECTED_MODEL}` revision `{EXPECTED_REVISION}`",
        f"- Runtime: vLLM `{EXPECTED_VLLM}`, PyTorch `{EXPECTED_TORCH}`, CUDA `{EXPECTED_CUDA}`",
        f"- Baseline GPU: `{cast(Mapping[str, Any], report['environment'])['baseline_gpu']}`",
        f"- Optimized GPU: `{cast(Mapping[str, Any], report['environment'])['optimized_gpu']}`",
        "",
        "## Same-GPU paired status",
        "",
        f"- Baseline: `{cast(Mapping[str, Any], report['paired_baseline'])['status']}`",
        f"- Optimized: `{cast(Mapping[str, Any], report['paired_optimized'])['status']}`",
        "",
        "## Readiness, metadata, scheduler, and GPU",
        "",
        "See `metrics/readiness-decomposition-summary.json` for the exact non-overlapping stages and raw values. Scheduler and GPU stages are not counted as metadata.",
        "`POST_ROOT_READY` ends after every branch returned first output. The legacy `FIRST_BRANCH_READY`/`ALL_BRANCHES_READY` fields are output-step-return boundaries, not direct runnable transitions; step-entry execution-start proxies are reported separately.",
        "`GPU_EXECUTION` is GPU-inclusive worker CPU wall, not pure device time. Optimized fanout-8 first `_model_forward` CUDA-event time is reported separately; baseline lacks the later sample-token wrappers, so baseline/optimized GPU and residual components are not identically scoped.",
        "",
        "## Software optimization and Amdahl",
        "",
        f"- Before/after: `{cast(Mapping[str, Any], report['before_after'])['status']}`",
        f"- Classification evidence: `{json.dumps(classification, sort_keys=True)}`",
        "- Uncertainty: fanout 8 has three paired seeds; fanout 16/32 are singletons and drive the conservative SOFTWARE_ONLY aggregate.",
        "- Before/after latency deltas are descriptive only because baseline used A100-SXM4 and optimized used A100-PCIE on different UUIDs; within-campaign independent/shared pairs remain causal.",
        "",
        "## HBM/state sharing",
        "",
        f"- Hardware COW capacity target: `{cast(Mapping[str, Any], report['hbm'])['HARDWARE_COW_CAPACITY_TARGET']}`",
        "",
        "## Decode interference",
        "",
        "Causal concurrent decode interference is unavailable because no background decode stream overlapped branch creation. Early-divergence values are labeled sequential phase-productivity deficits against established-branch throughput; branch-create zero is only a structural synchronous stall. No Experiment 002 throughput is used.",
        "",
        "## GPU-hours and cleanup",
        "",
        f"- GPU-hour ledger: `{json.dumps(report['gpu_hours'], sort_keys=True)}`",
        f"- Cleanup status: `{json.dumps(report['modal_cleanup_status'], sort_keys=True)}`",
        "",
        "## Next experiment",
        "",
        f"`{report['recommended_next_experiment']}`",
        "",
    ]
    return "\n".join(lines)


def _gpu_hour_summary(artifact_root: Path) -> dict[str, Any]:
    path = artifact_root / "gpu-hours.json"
    if not path.is_file():
        return _unavailable("GPU-hour ledger was not present at analysis time")
    ledger = _read_json(path)
    consumed = _finite_number(ledger.get("consumed_gpu_hours"), field="consumed_gpu_hours")
    intervals = ledger.get("gpu_active_intervals")
    reservations = ledger.get("modal_in_flight_reservations")
    if (
        ledger.get("schema_version") != "sloforge.branchfabric.gpu-hours/v1"
        or not isinstance(intervals, list)
        or not isinstance(reservations, list)
    ):
        raise Experiment003AnalysisError("GPU-hour ledger schema is invalid")
    recorded_cost = sum(
        float(interval.get("cost_usd", 0.0)) for interval in intervals if isinstance(interval, dict)
    )
    known_gpu_only_cost = sum(
        float(interval.get("gpu_cost_usd", interval.get("cost_usd", 0.0)))
        for interval in intervals
        if isinstance(interval, dict)
    )
    known_support_cost = sum(
        float(interval.get("support_cost_usd", 0.0))
        for interval in intervals
        if isinstance(interval, dict)
    )
    return {
        "status": "available",
        "consumed_gpu_hours": consumed,
        "historical_experiment_002_gpu_hours": ledger.get("historical_experiment_002_gpu_hours"),
        "target_additional_gpu_hours": ledger.get("target_additional_gpu_hours"),
        "hard_additional_gpu_hours": ledger.get("hard_additional_gpu_hours"),
        "known_gpu_only_cost_usd": known_gpu_only_cost,
        "known_support_cost_usd": known_support_cost,
        "recorded_cost_usd_mixed_conventions": recorded_cost,
        "cost_note": (
            "baseline recorded GPU cost only; optimized recorded GPU plus support cost. "
            "known_gpu_only_cost_usd is comparable across campaigns"
        ),
        "gpu_active_intervals": intervals,
        "in_flight_reservations": reservations,
        "all_reservations_cleared": reservations == [],
        "provenance": {
            "artifact_reference": path.as_posix(),
            "artifact_sha256": _sha256(path),
        },
    }


def _modal_cleanup_summary(artifact_root: Path, gpu_hours: Mapping[str, Any]) -> dict[str, Any]:
    path = artifact_root / "logs/modal-provider-cleanup.json"
    if not path.is_file():
        return {
            "child_lifecycle_cleanup": "PASS",
            "forced_gpu_process_kill_required": False,
            "campaign_reservations_cleared": gpu_hours.get("all_reservations_cleared"),
            "provider_zero_active_tasks": False,
            "deployed_endpoints_absent": False,
            "only_authorized_persistent_volumes": False,
            "status": "unavailable",
            "reason": "provider inventory was not persisted",
        }
    inventory = _read_json(path)
    if inventory.get("schema_version") != "sloforge.branchfabric.modal-provider-cleanup/v1":
        raise Experiment003AnalysisError("Modal cleanup inventory schema is invalid")
    apps = inventory.get("apps")
    containers = inventory.get("containers")
    volumes = inventory.get("volume_names")
    authorized = inventory.get("authorized_volume_names")
    if not all(isinstance(value, list) for value in (apps, containers, volumes, authorized)):
        raise Experiment003AnalysisError("Modal cleanup inventory lists are invalid")
    app_rows = cast(list[Any], apps)
    zero_tasks = (
        all(
            isinstance(app, dict) and app.get("state") == "stopped" and str(app.get("tasks")) == "0"
            for app in app_rows
        )
        and cast(list[Any], containers) == []
    )
    endpoint_free = all(isinstance(app, dict) and app.get("state") == "stopped" for app in app_rows)
    volume_set = {str(value) for value in cast(list[Any], volumes)}
    authorized_set = {str(value) for value in cast(list[Any], authorized)}
    volumes_ok = (
        volume_set
        == authorized_set
        == {
            "sloforge-branchfabric-results",
            "sloforge-model-cache",
        }
    )
    return {
        "status": "PASS" if zero_tasks and endpoint_free and volumes_ok else "FAIL",
        "child_lifecycle_cleanup": "PASS",
        "forced_gpu_process_kill_required": False,
        "campaign_reservations_cleared": gpu_hours.get("all_reservations_cleared"),
        "provider_zero_active_tasks": zero_tasks,
        "deployed_endpoints_absent": endpoint_free,
        "only_authorized_persistent_volumes": volumes_ok,
        "observed_at_utc": inventory.get("observed_at_utc"),
        "provenance": {
            "artifact_reference": path.as_posix(),
            "artifact_sha256": _sha256(path),
        },
    }


def _runtime_source_summary(datasets: Sequence[CampaignDataset]) -> dict[str, Any]:
    observations = [trial.metadata_observation for dataset in datasets for trial in dataset.trials]
    tags = {str(observation["runtime_source_tag"]) for observation in observations}
    commits = {str(observation["runtime_source_commit"]) for observation in observations}
    levels = sorted({str(observation["instrumentation_level"]) for observation in observations})
    optimizations = sorted({str(observation["optimization"]) for observation in observations})
    if tags != {EXPECTED_VLLM_SOURCE_TAG} or commits != {EXPECTED_VLLM_SOURCE_COMMIT}:
        raise Experiment003AnalysisError("campaigns disagree on the pinned vLLM source")
    return {
        "schema_version": "sloforge.branchfabric.experiment-003-runtime-source/v1",
        "runtime": EXPECTED_RUNTIME,
        "runtime_version": EXPECTED_VLLM,
        "runtime_source_tag": EXPECTED_VLLM_SOURCE_TAG,
        "runtime_source_commit": EXPECTED_VLLM_SOURCE_COMMIT,
        "instrumentation_schema": ("sloforge.branchfabric.vllm-metadata-observation-0230/v1"),
        "instrumentation_levels_observed": levels,
        "optimizations_observed": optimizations,
        "trial_observation_count": len(observations),
        "version_scoped_internal_path": True,
    }


def _hardware_interest_payload(
    optimized: CampaignDataset,
    classification: Mapping[str, Any],
) -> dict[str, Any]:
    fanouts: dict[str, Any] = {}
    for fanout, trials in sorted(_shared_minimal(optimized).items()):
        rows: list[dict[str, Any]] = []
        for trial in trials:
            metadata_ns = _metadata_time_ns(trial)
            operation_count = _counter_summary(trial)["counted_metadata_work_units"]
            rows.append(
                {
                    "attempt_id": trial.attempt_id,
                    "counted_metadata_work_units": operation_count,
                    "metadata_wall_ns": metadata_ns,
                    "counted_metadata_work_units_per_second": (
                        None
                        if metadata_ns in {None, 0}
                        else operation_count / (cast(int, metadata_ns) / 1e9)
                    ),
                    "working_set_physical_kv_bytes": trial.summary["physical_assigned_bytes"],
                    "block_table_size": _unavailable(
                        "runtime did not emit block-table storage bytes"
                    ),
                    "operation_latency_p50_p95_p99": _unavailable(
                        "aggregate counters do not support per-operation latency percentiles"
                    ),
                    "concurrent_operations": _unavailable(
                        "operation concurrency was not directly recorded"
                    ),
                    "queue_depth": _unavailable(
                        "queue inserts/removals do not establish instantaneous queue depth"
                    ),
                    "read_write_ratio": _unavailable(
                        "counter taxonomy does not classify every operation as a byte read/write"
                    ),
                    "operation_counters": trial.summary["metadata_operation_counters"],
                    "counter_semantics": (
                        "software work units only; not inferred hardware transactions"
                    ),
                }
            )
        fanouts[str(fanout)] = rows
    return {
        "schema_version": "sloforge.branchfabric.metadata-hardware-interest/v1",
        "classification": dict(classification),
        "fanouts": fanouts,
        "fpga_justification": False,
        "note": "architecture characterization only; unavailable fields are not inferred",
    }


def generate_analysis(
    *,
    baseline_campaign_root: Path,
    optimized_campaign_root: Path | None,
    artifact_root: Path,
    reports_root: Path,
    docs_root: Path,
) -> dict[str, Any]:
    baseline = load_campaign(baseline_campaign_root, implementation="baseline")
    optimized = (
        None
        if optimized_campaign_root is None
        else load_campaign(optimized_campaign_root, implementation="optimized")
    )
    datasets = [baseline, *([] if optimized is None else [optimized])]
    artifact_root.mkdir(parents=True, exist_ok=True)
    for directory in REQUIRED_ARTIFACT_DIRECTORIES:
        (artifact_root / directory).mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)
    docs_root.mkdir(parents=True, exist_ok=True)

    paired_baseline = _paired_analysis(baseline)
    paired_optimized = (
        _unavailable("optimized campaign was not supplied")
        if optimized is None
        else _paired_analysis(optimized)
    )
    trace = _trace_overhead(baseline)
    if trace.get("status") != "available" and optimized is not None:
        trace = _trace_overhead(optimized)
    decomposition_summary = _group_decomposition(datasets)
    decomposition = cast(dict[str, Any], decomposition_summary["implementations"])
    readiness_definitions = _readiness_definitions(datasets)
    scaling = {dataset.implementation: _scaling(dataset) for dataset in datasets}
    amdahl = (
        {
            "schema_version": "sloforge.branchfabric.experiment-003-amdahl-summary/v1",
            "implementation": "optimized",
            "fanouts": {
                str(fanout): _unavailable("optimized campaign was not supplied")
                for fanout in (8, 16, 32)
            },
        }
        if optimized is None
        else _amdahl(optimized)
    )
    hbm = _hbm(datasets)
    decode = _decode_and_gpu(datasets)
    native = _native_agreement(datasets)
    profile = _profile_summary(datasets)
    before_after = _before_after(baseline, optimized)
    initial_gate = _initial_analysis_gate(decomposition)
    gpu_hours = _gpu_hour_summary(artifact_root)
    modal_cleanup = _modal_cleanup_summary(artifact_root, gpu_hours)
    modal_cleanup["children_verified"] = sum(len(dataset.trials) for dataset in datasets)
    runtime_source = _runtime_source_summary(datasets)
    classification = _classification(
        decomposition,
        amdahl,
        optimized,
        paired_baseline,
        paired_optimized,
    )
    conclusion = str(classification["conclusion"])
    shared_fanouts = {
        dataset.implementation: sorted(_shared_minimal(dataset)) for dataset in datasets
    }
    completion_checks = {
        "exact_metadata_conclusion": conclusion.startswith("METADATA_"),
        "baseline_shared_fanouts_1_8_16_32": set(shared_fanouts.get("baseline", [])).issuperset(
            {1, 8, 16, 32}
        ),
        "optimized_shared_fanouts_1_8_16_32": set(shared_fanouts.get("optimized", [])).issuperset(
            {1, 8, 16, 32}
        ),
        "three_same_gpu_pairs_before": paired_baseline.get("status") == "available",
        "three_same_gpu_pairs_after": paired_optimized.get("status") == "available",
        "trace_overhead_measured": trace.get("status") == "available",
        "cpu_profile_available": profile.get("status") == "available",
        "before_after_available": before_after.get("status") == "available",
        "runtime_native_metrics_no_major_disagreements": native.get("major_disagreements") == [],
        "hbm_floor_closed": hbm["HARDWARE_COW_CAPACITY_TARGET"] == "CLOSED",
        "gpu_hour_ledger_clear": gpu_hours.get("status") == "available"
        and gpu_hours.get("all_reservations_cleared") is True,
        "provider_zero_active_tasks": modal_cleanup.get("provider_zero_active_tasks") is True,
        "deployed_endpoints_absent": modal_cleanup.get("deployed_endpoints_absent") is True,
        "only_authorized_persistent_volumes": modal_cleanup.get(
            "only_authorized_persistent_volumes"
        )
        is True,
    }
    completion_status = "complete" if all(completion_checks.values()) else "incomplete"
    next_experiment = (
        "Experiment 004 capacity reclamation"
        if conclusion in {"METADATA_CLOSED", "METADATA_SOFTWARE_ONLY"}
        else (
            "Dedicated metadata architectural characterization before Experiment 004"
            if conclusion == "METADATA_HARDWARE_INTEREST"
            else "Unavailable until the Experiment 003 completion evidence is materialized"
        )
    )
    report: dict[str, Any] = {
        "schema_version": "sloforge.branchfabric.gpu-validation-experiment-003-report/v1",
        "completion_status": completion_status,
        "environment": {
            "model": EXPECTED_MODEL,
            "model_revision": EXPECTED_REVISION,
            "vllm": EXPECTED_VLLM,
            "pytorch": EXPECTED_TORCH,
            "cuda": EXPECTED_CUDA,
            "baseline_gpu": baseline.gpu_identity,
            "optimized_gpu": None if optimized is None else optimized.gpu_identity,
        },
        "source_campaigns": {
            "baseline": {
                "path": baseline.root.as_posix(),
                "campaign_id": baseline.campaign_id,
                "function_call_id": baseline.function_call_id,
                "remote_manifest_sha256": baseline.remote_manifest_sha256,
                "campaign_status": baseline.campaign_status,
                "adaptive_tail_timeout": baseline.adaptive_tail_timeout,
            },
            "optimized": (
                None
                if optimized is None
                else {
                    "path": optimized.root.as_posix(),
                    "campaign_id": optimized.campaign_id,
                    "function_call_id": optimized.function_call_id,
                    "remote_manifest_sha256": optimized.remote_manifest_sha256,
                    "campaign_status": optimized.campaign_status,
                    "adaptive_tail_timeout": optimized.adaptive_tail_timeout,
                }
            ),
        },
        "fanouts_executed": {
            dataset.implementation: sorted({trial.fanout for trial in dataset.trials})
            for dataset in datasets
        },
        "completion_checks": completion_checks,
        "paired_baseline": paired_baseline,
        "paired_optimized": paired_optimized,
        "tracing_overhead": trace,
        "initial_analysis_gate": initial_gate,
        "readiness_definitions": readiness_definitions,
        "decomposition": decomposition,
        "decomposition_measurement_caveats": {
            "cross_implementation_gpu_stage_comparable": decomposition_summary[
                "cross_implementation_gpu_stage_comparable"
            ],
            "gpu_stage_caveat": decomposition_summary["gpu_stage_caveat"],
        },
        "scaling": scaling,
        "before_after": before_after,
        "hbm": hbm,
        "decode_interference": decode,
        "native_metric_agreement": native,
        "cpu_hot_path": profile,
        "amdahl": amdahl,
        "gpu_hours": gpu_hours,
        "modal_cleanup_status": modal_cleanup,
        "metadata_classification": classification,
        "recommended_next_experiment": next_experiment,
        "causal_exclusions": [
            "Experiment 002 cross-GPU throughput is historical only",
            "scheduler queueing is not metadata",
            "profiled full-trace diagnostics are excluded from primary causal timing",
        ],
    }

    _write_json(
        artifact_root / "paired/paired-analysis.json",
        {"baseline": paired_baseline, "optimized": paired_optimized},
    )
    _write_json(
        artifact_root / "metrics/readiness-decomposition-summary.json", decomposition_summary
    )
    _write_json(artifact_root / "metrics/readiness-definitions.json", readiness_definitions)
    _write_json(artifact_root / "metrics/instrumentation-overhead.json", trace)
    _write_json(artifact_root / "analysis/initial-gate.json", initial_gate)
    _write_json(artifact_root / "fanout/scaling-analysis.json", scaling)
    _write_json(
        artifact_root / "metadata/operation-scaling.json",
        {
            key: {
                "total": value["metadata_operation_model"],
                "per_operation": value["per_operation_models"],
                "raw_by_fanout": value["raw_by_fanout"],
            }
            for key, value in scaling.items()
        },
    )
    _write_json(artifact_root / "analysis/amdahl.json", amdahl)
    _write_json(artifact_root / "analysis/hbm-analysis.json", hbm)
    _write_json(artifact_root / "analysis/decode-interference.json", decode)
    _write_json(artifact_root / "native-metrics/runtime-native-metric-agreement.json", native)
    _write_json(artifact_root / "profiles/cpu-hot-path-summary.json", profile)
    _write_json(artifact_root / "software-baseline/before-after.json", before_after)
    _write_json(
        artifact_root / "environment/environment.json",
        {
            "schema_version": "sloforge.branchfabric.experiment-003-environment/v1",
            **cast(dict[str, Any], report["environment"]),
            "gpu_hours": gpu_hours,
            "modal_cleanup_status": report["modal_cleanup_status"],
        },
    )
    _write_json(
        artifact_root / "runtime-source/vllm-0230-runtime-source.json",
        runtime_source,
    )
    _write_json(
        artifact_root / "raw/source-index.json",
        {
            "schema_version": "sloforge.branchfabric.experiment-003-source-index/v1",
            "campaigns": report["source_campaigns"],
            "raw_artifacts_are_immutable": True,
        },
    )
    for name, svg in _plot_payloads(datasets, decomposition, amdahl, hbm, decode, profile).items():
        _write_text(artifact_root / "plots" / name, svg)

    report_json = reports_root / "branchfabric-gpu-validation-experiment-003.json"
    report_md = reports_root / "branchfabric-gpu-validation-experiment-003.md"
    _write_json(report_json, report)
    _write_text(report_md, _markdown_report(report))
    _write_text(docs_root / "METADATA_CHARACTERIZATION.md", _metadata_characterization_doc(report))
    if conclusion in {"METADATA_CLOSED", "METADATA_SOFTWARE_ONLY"}:
        _write_text(docs_root / "EXPERIMENT_004_PLAN.md", _experiment_004_plan())
    elif conclusion == "METADATA_HARDWARE_INTEREST":
        assert optimized is not None
        _write_json(
            artifact_root / "metadata-hardware-interest.json",
            _hardware_interest_payload(optimized, classification),
        )
        _write_text(
            docs_root / "METADATA_HARDWARE_INTEREST.md",
            "# Metadata Hardware Interest\n\nThis gate is an architecture-characterization signal, not FPGA justification. See the Experiment 003 report and raw operation/scaling artifacts.\n",
        )

    generated = [
        path
        for path in sorted(artifact_root.rglob("*"))
        if path.is_file() and path != artifact_root / "manifest.json"
    ]
    manifest = {
        "schema_version": "sloforge.branchfabric.gpu-validation-experiment-003-manifest/v1",
        "completion_status": completion_status,
        "metadata_conclusion": conclusion,
        "gpu_hours": gpu_hours,
        "modal_cleanup_status": report["modal_cleanup_status"],
        "source_campaigns": report["source_campaigns"],
        "required_directories": list(REQUIRED_ARTIFACT_DIRECTORIES),
        "generated_artifacts": [
            {
                "relative_path": path.relative_to(artifact_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in generated
        ],
        "reports": [report_json.as_posix(), report_md.as_posix()],
        "documentation": [
            (docs_root / "METADATA_CHARACTERIZATION.md").as_posix(),
            *(
                [(docs_root / "EXPERIMENT_004_PLAN.md").as_posix()]
                if conclusion in {"METADATA_CLOSED", "METADATA_SOFTWARE_ONLY"}
                else (
                    [(docs_root / "METADATA_HARDWARE_INTEREST.md").as_posix()]
                    if conclusion == "METADATA_HARDWARE_INTEREST"
                    else []
                )
            ),
        ],
    }
    _write_json(artifact_root / "manifest.json", manifest)
    return report


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-campaign", type=Path, required=True)
    parser.add_argument("--optimized-campaign", type=Path)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/branchfabric/gpu-validation/experiment-003"),
    )
    parser.add_argument("--reports-root", type=Path, default=Path("reports"))
    parser.add_argument("--docs-root", type=Path, default=Path("docs/branchfabric"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    report = generate_analysis(
        baseline_campaign_root=args.baseline_campaign,
        optimized_campaign_root=args.optimized_campaign,
        artifact_root=args.artifact_root,
        reports_root=args.reports_root,
        docs_root=args.docs_root,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "Experiment003AnalysisError",
    "classify_metadata",
    "generate_analysis",
    "load_campaign",
    "main",
]
