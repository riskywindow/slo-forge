#!/usr/bin/env python3
"""Analyze locally materialized BranchFabric GPU Validation Experiment 002 results.

This module is deliberately local-only: it does not import Modal, initialize CUDA, invoke
``nvidia-smi``, or access a remote host. It accepts only results whose Modal manifest and local
copy verification are complete, revalidates their bytes, checks scientific pairing, and emits
deterministic JSON, Markdown, and SVG artifacts.
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
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from html import escape
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal, cast

from gpu_validation_runner import TrialPath, TrialSummary
from pydantic import ValidationError

from sloforge.continuum.adapters.real_runtime import PhysicalKvLayoutSnapshot

ANALYZER_SCHEMA_VERSION = "sloforge.branchfabric.gpu-validation-analysis/v1"
REPORT_SCHEMA_VERSION = "sloforge.branchfabric.gpu-validation-report/v1"
REMOTE_MANIFEST_SCHEMA = "sloforge.branchfabric.modal-result-manifest/v1"
LOCAL_VERIFICATION_SCHEMA = "sloforge.branchfabric.modal-local-copy-verification/v1"
EXPECTED_RUNTIME = "vllm"
EXPECTED_RUNTIME_VERSION = "0.23.0"
EXPECTED_GPU_SKU = "A100-80GB"
EXPECTED_PREFIX_TOKENS = 16_384
_PRIMARY_TRACE_LEVEL = "full"
_TRACE_CONTROL_LEVEL = "disabled"
_FINAL_PROVIDER_AUDIT = Path("environment/final-modal-provider-audit-v2.json")
_LIFECYCLE_REQUIRED_FILES = (
    "controller-manifest.json",
    "child-manifest.json",
    "process-tree-before.json",
    "process-tree-during.json",
    "process-tree-after.json",
    "hbm-baseline.json",
    "hbm-postflight.json",
    "cleanup-gate.json",
    "forbidden-import-audit.json",
    "child-events.jsonl",
)


class AnalysisError(RuntimeError):
    """A materialized result is incomplete, inconsistent, or scientifically unpairable."""


@dataclass(frozen=True, slots=True)
class LifecycleEvidence:
    controller_manifest: dict[str, Any]
    child_manifest: dict[str, Any]
    process_tree_before: dict[str, Any]
    process_tree_during: dict[str, Any]
    process_tree_after: dict[str, Any]
    hbm_baseline: dict[str, Any]
    hbm_postflight: dict[str, Any]
    cleanup_gate: dict[str, Any]
    forbidden_import_audit: dict[str, Any]
    child_events: tuple[dict[str, Any], ...]

    @property
    def time_to_hbm_recovery_ms(self) -> float | None:
        value = self.cleanup_gate.get("time_to_hbm_recovery_ms")
        return (
            float(value)
            if isinstance(value, (int, float)) and not isinstance(value, bool)
            else None
        )


@dataclass(frozen=True, slots=True)
class LoadedTrial:
    attempt_id: str
    root: Path
    config: dict[str, Any]
    completion: dict[str, Any]
    invocation: dict[str, Any]
    runner_configuration: dict[str, Any]
    model_manifest_reference: dict[str, Any]
    dependency_versions: dict[str, Any]
    summary: TrialSummary
    remote_manifest_sha256: str
    gpu_uuid: str
    gpu_name: str
    physical_rows: tuple[tuple[str, PhysicalKvLayoutSnapshot], ...]
    gpu_memory_rows: tuple[dict[str, Any], ...]
    sampler_rows: tuple[dict[str, Any], ...]
    phase_rows: tuple[dict[str, Any], ...]
    release_rows: tuple[dict[str, Any], ...]
    postflight_clean: bool
    lifecycle: LifecycleEvidence | None

    @property
    def lifecycle_clean(self) -> bool | None:
        if self.lifecycle is None:
            return None
        return self.lifecycle.cleanup_gate["GPU_RUNTIME_LIFECYCLE_CLEAN"] is True

    @property
    def scientific_input_sha256(self) -> str:
        experiment = cast(dict[str, Any], self.invocation["experiment"])
        engine = cast(dict[str, Any], self.invocation["engine"])
        identity = cast(dict[str, Any], experiment["expected_identity"])
        payload = {
            "prefix_token_ids": experiment["prefix_token_ids"],
            "divergent_token_ids": experiment["divergent_token_ids"],
            "fanout": experiment["fanout"],
            "suffix_tokens": experiment["suffix_tokens"],
            "seed": experiment["seed"],
            "identity": identity,
            "gpu_sampling": {
                key: value
                for key, value in cast(dict[str, Any], experiment["gpu_sampling"]).items()
                if key != "physical_device_selector"
            },
            "engine": {
                key: engine[key]
                for key in sorted(engine)
                if key not in {"execution_model_path", "download_dir"}
            },
            "model_manifest_sha256": self.model_manifest_reference["sha256"],
            "dependency_versions": self.dependency_versions,
        }
        return hashlib.sha256(_canonical_bytes(payload)).hexdigest()

    @property
    def tracing_level(self) -> str:
        return str(self.config["tracing_level"])

    @property
    def baseline_mode(self) -> str:
        return str(self.config["baseline_mode"])

    @property
    def fanout(self) -> int:
        return int(self.config["fanout"])

    @property
    def prefix_length(self) -> int:
        return int(self.config["prefix_length"])

    @property
    def suffix_length(self) -> int:
        return int(self.config["suffix_length"])

    @property
    def seed(self) -> int:
        return int(self.config["seed"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode()


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


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise AnalysisError(f"cannot read JSON artifact {path}: {error}") from error
    if not isinstance(payload, dict):
        raise AnalysisError(f"JSON artifact is not an object: {path}")
    return cast(dict[str, Any], payload)


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text().splitlines()
    except OSError as error:
        raise AnalysisError(f"cannot read JSONL artifact {path}: {error}") from error
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise AnalysisError(f"invalid JSONL row {path}:{index}: {error}") from error
        if not isinstance(payload, dict):
            raise AnalysisError(f"JSONL row is not an object: {path}:{index}")
        rows.append(cast(dict[str, Any], payload))
    return tuple(rows)


def _driver_memory_mib(sample: Mapping[str, Any]) -> int | None:
    gpu = sample.get("gpu")
    if not isinstance(gpu, dict):
        return None
    value = gpu.get("memory_used_mib")
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        return None
    return value


def _stable_hbm_window(
    samples: Sequence[Mapping[str, Any]], *, threshold_mib: int
) -> tuple[int, int] | None:
    consecutive = 0
    start = 0
    for index, sample in enumerate(samples):
        memory_mib = _driver_memory_mib(sample)
        qualifies = (
            sample.get("error") is None
            and memory_mib is not None
            and memory_mib <= threshold_mib
            and sample.get("compute_processes") == []
        )
        if qualifies:
            if consecutive == 0:
                start = index
            consecutive += 1
            if consecutive >= 3:
                return start, index
        else:
            consecutive = 0
    return None


def _load_lifecycle_evidence(
    root: Path,
    *,
    attempt_id: str,
    gpu_uuid: str,
    completion: Mapping[str, Any],
    wrapper_postflight: Mapping[str, Any],
) -> LifecycleEvidence | None:
    lifecycle_root = root / "lifecycle"
    if not lifecycle_root.exists():
        return None
    if not lifecycle_root.is_dir():
        raise AnalysisError(f"lifecycle artifact root is not a directory: {attempt_id}")
    missing = [name for name in _LIFECYCLE_REQUIRED_FILES if not (lifecycle_root / name).is_file()]
    if missing:
        raise AnalysisError(
            f"authoritative lifecycle evidence is incomplete for {attempt_id}: {missing}"
        )

    controller = _read_json(lifecycle_root / "controller-manifest.json")
    child = _read_json(lifecycle_root / "child-manifest.json")
    before = _read_json(lifecycle_root / "process-tree-before.json")
    during = _read_json(lifecycle_root / "process-tree-during.json")
    after = _read_json(lifecycle_root / "process-tree-after.json")
    baseline = _read_json(lifecycle_root / "hbm-baseline.json")
    hbm_postflight = _read_json(lifecycle_root / "hbm-postflight.json")
    cleanup = _read_json(lifecycle_root / "cleanup-gate.json")
    import_audit = _read_json(lifecycle_root / "forbidden-import-audit.json")
    events = _read_jsonl(lifecycle_root / "child-events.jsonl")
    expected_schemas = (
        (controller, "sloforge.branchfabric.gpu-cow-controller-manifest/v1"),
        (child, "sloforge.branchfabric.gpu-cow-child-manifest/v1"),
        (before, "sloforge.branchfabric.process-tree/v1"),
        (during, "sloforge.branchfabric.process-tree-series/v1"),
        (after, "sloforge.branchfabric.process-tree/v1"),
        (baseline, "sloforge.branchfabric.hbm-baseline/v1"),
        (hbm_postflight, "sloforge.branchfabric.hbm-postflight/v1"),
        (cleanup, "sloforge.branchfabric.gpu-runtime-cleanup-gate/v1"),
        (import_audit, "sloforge.branchfabric.forbidden-gpu-import-audits/v1"),
    )
    if any(payload.get("schema_version") != schema for payload, schema in expected_schemas):
        raise AnalysisError(f"unsupported authoritative lifecycle schema in {attempt_id}")
    if any(
        event.get("schema_version") != "sloforge.branchfabric.gpu-worker-event/v1"
        or not isinstance(event.get("event"), str)
        or not isinstance(event.get("observed_at_monotonic_ns"), int)
        for event in events
    ):
        raise AnalysisError(f"invalid child lifecycle event stream in {attempt_id}")

    child_pid = controller.get("child_pid")
    child_pgid = controller.get("child_pgid")
    if (
        controller.get("attempt_id") != attempt_id
        or child.get("attempt_id") != attempt_id
        or not isinstance(child_pid, int)
        or isinstance(child_pid, bool)
        or child_pid <= 0
        or child_pgid != child_pid
        or child.get("child_pid") != child_pid
        or child.get("child_pgid") != child_pgid
        or during.get("child_pid") != child_pid
        or during.get("child_pgid") != child_pgid
    ):
        raise AnalysisError(f"controller/worker process identity is inconsistent in {attempt_id}")
    after_processes = after.get("processes")
    tracked_processes = during.get("tracked_owned_processes")
    if (
        not isinstance(after_processes, list)
        or not isinstance(tracked_processes, list)
        or not tracked_processes
        or child_pid
        not in {process.get("pid") for process in tracked_processes if isinstance(process, dict)}
    ):
        raise AnalysisError(f"process ownership evidence is incomplete for {attempt_id}")
    after_pids = {process.get("pid") for process in after_processes if isinstance(process, dict)}
    surviving_owned = sorted(
        int(process["pid"])
        for process in tracked_processes
        if isinstance(process, dict)
        and isinstance(process.get("pid"), int)
        and process["pid"] in after_pids
    )
    if surviving_owned:
        raise AnalysisError(
            f"owned PIDs remain in postflight process tree for {attempt_id}: {surviving_owned}"
        )

    baseline_samples = baseline.get("samples")
    baseline_values = baseline.get("baseline_samples_mib")
    baseline_median = baseline.get("baseline_median_mib")
    threshold = baseline.get("cleanup_threshold_mib")
    if (
        not isinstance(baseline_samples, list)
        or len(baseline_samples) < 3
        or not isinstance(baseline_values, list)
        or baseline_values != [_driver_memory_mib(sample) for sample in baseline_samples]
        or any(not isinstance(value, int) or isinstance(value, bool) for value in baseline_values)
        or baseline_median != int(statistics.median(baseline_values))
        or threshold != baseline_median + 1024
        or baseline.get("allowance_mib") != 1024
        or baseline.get("gpu_uuid") != gpu_uuid
    ):
        raise AnalysisError(f"invalid driver-visible HBM baseline in {attempt_id}")
    for sample in baseline_samples:
        gpu = sample.get("gpu") if isinstance(sample, dict) else None
        if not isinstance(gpu, dict) or gpu.get("uuid") != gpu_uuid:
            raise AnalysisError(f"baseline sampled a different GPU in {attempt_id}")

    post_samples = hbm_postflight.get("post_child_exit_samples")
    child_live_samples = hbm_postflight.get("child_live_samples")
    if not isinstance(post_samples, list) or not isinstance(child_live_samples, list):
        raise AnalysisError(f"invalid lifecycle HBM sample series in {attempt_id}")
    for sample in (*child_live_samples, *post_samples):
        if not isinstance(sample, dict):
            raise AnalysisError(f"non-object lifecycle HBM sample in {attempt_id}")
        if sample.get("error") is None:
            gpu = sample.get("gpu")
            if (
                not isinstance(sample.get("observed_at_monotonic_ns"), int)
                or not isinstance(gpu, dict)
                or gpu.get("uuid") != gpu_uuid
                or _driver_memory_mib(sample) is None
            ):
                raise AnalysisError(f"lifecycle HBM sample changed GPU identity in {attempt_id}")
    window = _stable_hbm_window(post_samples, threshold_mib=int(threshold))
    evaluation = cleanup.get("stable_sample_evaluation")
    declared_clean = cleanup.get("GPU_RUNTIME_LIFECYCLE_CLEAN") is True
    if not isinstance(evaluation, dict):
        raise AnalysisError(f"cleanup gate lacks stable-sample evaluation in {attempt_id}")
    expected_window = None if window is None else [window[0], window[1]]
    declared_window = [
        evaluation.get("winning_sample_start_index"),
        evaluation.get("winning_sample_end_index"),
    ]
    if (
        evaluation.get("passed") is not (window is not None)
        or evaluation.get("required_consecutive_samples") != 3
        or (window is not None and declared_window != expected_window)
        or cleanup.get("baseline_median_mib") != baseline_median
        or cleanup.get("cleanup_threshold_mib") != threshold
        or cleanup.get("allowance_mib") != 1024
    ):
        raise AnalysisError(f"cleanup gate disagrees with raw HBM samples in {attempt_id}")

    audits = import_audit.get("audits")
    expected_stages = {
        "controller_entry_before_baseline",
        "immediately_before_child_launch",
        "after_child_exit_and_hbm_postflight",
    }
    audits_clean = (
        isinstance(audits, list)
        and expected_stages.issubset(
            {audit.get("stage") for audit in audits if isinstance(audit, dict)}
        )
        and all(isinstance(audit, dict) and audit.get("cuda_clean") is True for audit in audits)
        and import_audit.get("cuda_clean") is True
    )
    invariants = child.get("semantic_invariants")
    semantic_clean = (
        child.get("status") == "succeeded"
        and isinstance(invariants, dict)
        and bool(invariants)
        and all(value is True for value in invariants.values())
        and child.get("final_runtime_assigned_kv_bytes") == 0
        and child.get("multiprocessing_children_before_exit") == []
    )
    time_to_recovery = cleanup.get("time_to_hbm_recovery_ms")
    clean_facts = (
        cleanup.get("cleanup_gate") == "PASS"
        and cleanup.get("process_hbm_recovery") == "verified"
        and window is not None
        and isinstance(time_to_recovery, (int, float))
        and not isinstance(time_to_recovery, bool)
        and cleanup.get("child_return_code") == 0
        and cleanup.get("child_semantics_valid") is True
        and cleanup.get("no_owned_descendant_remained") is True
        and cleanup.get("no_compute_process_remained_in_winning_samples") is True
        and cleanup.get("forced_gpu_process_kill_required") is False
        and cleanup.get("termination_actions") == []
        and cleanup.get("timed_out") is False
        and controller.get("status") == "succeeded"
        and controller.get("child_return_code") == 0
        and controller.get("parent_cuda_clean") is True
        and controller.get("forced_gpu_process_kill_required") is False
        and controller.get("termination_actions") == []
        and semantic_clean
        and audits_clean
    )
    if declared_clean is not clean_facts:
        raise AnalysisError(
            f"authoritative lifecycle cleanup claim is inconsistent in {attempt_id}"
        )
    if not declared_clean and cleanup.get("cleanup_gate") != "FAIL":
        raise AnalysisError(f"failed lifecycle gate has an invalid classification in {attempt_id}")

    if (
        wrapper_postflight.get("schema_version") != "sloforge.branchfabric.modal-postflight/v2"
        or wrapper_postflight.get("authoritative_artifact") != "lifecycle/cleanup-gate.json"
        or wrapper_postflight.get("GPU_RUNTIME_LIFECYCLE_CLEAN") is not declared_clean
        or wrapper_postflight.get("cleanup_gate") != cleanup.get("cleanup_gate")
        or wrapper_postflight.get("memory_recovered_within_1_gib")
        is not (cleanup.get("process_hbm_recovery") == "verified")
        or (wrapper_postflight.get("unexpected_compute_processes") == [])
        is not (cleanup.get("no_compute_process_remained_in_winning_samples") is True)
        or completion.get("GPU_RUNTIME_LIFECYCLE_CLEAN") is not declared_clean
        or completion.get("child_pid") != child_pid
        or completion.get("child_pgid") != child_pgid
        or completion.get("child_return_code") != controller.get("child_return_code")
    ):
        raise AnalysisError(f"wrapper/completion lifecycle projection is invalid in {attempt_id}")
    return LifecycleEvidence(
        controller_manifest=controller,
        child_manifest=child,
        process_tree_before=before,
        process_tree_during=during,
        process_tree_after=after,
        hbm_baseline=baseline,
        hbm_postflight=hbm_postflight,
        cleanup_gate=cleanup,
        forbidden_import_audit=import_audit,
        child_events=events,
    )


def _inventory_map(records: object, *, source: str) -> dict[str, tuple[int, str]]:
    if not isinstance(records, list):
        raise AnalysisError(f"{source} artifact inventory is not a list")
    result: dict[str, tuple[int, str]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise AnalysisError(f"{source} artifact inventory contains a non-object")
        relative = record.get("relative_path")
        size = record.get("bytes")
        digest = record.get("sha256")
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
        ):
            raise AnalysisError(f"{source} artifact inventory has an invalid entry")
        if relative in result:
            raise AnalysisError(f"{source} artifact inventory duplicates {relative}")
        result[relative] = (size, digest)
    return result


def _verify_materialized_tree(root: Path, attempt_id: str) -> str:
    remote_path = root / "REMOTE_MANIFEST.json"
    local_path = root / "LOCAL_COPY_VERIFICATION.json"
    remote = _read_json(remote_path)
    local = _read_json(local_path)
    remote_digest = _sha256(remote_path)
    if (
        remote.get("schema_version") != REMOTE_MANIFEST_SCHEMA
        or remote.get("attempt_id") != attempt_id
        or remote.get("remote_prefix") != f"experiment-002/modal/{attempt_id}"
    ):
        raise AnalysisError(f"remote manifest identity is invalid for {attempt_id}")
    if (
        local.get("schema_version") != LOCAL_VERIFICATION_SCHEMA
        or local.get("attempt_id") != attempt_id
        or local.get("remote_manifest_sha256") != remote_digest
    ):
        raise AnalysisError(f"local copy verification is invalid for {attempt_id}")
    remote_inventory = _inventory_map(remote.get("artifacts"), source="remote manifest")
    local_inventory = _inventory_map(local.get("artifacts"), source="local verification")
    manifest_record = local_inventory.pop("REMOTE_MANIFEST.json", None)
    if manifest_record != (remote_path.stat().st_size, remote_digest):
        raise AnalysisError(f"local verification does not anchor REMOTE_MANIFEST for {attempt_id}")
    if local_inventory != remote_inventory:
        raise AnalysisError(f"remote and local inventories disagree for {attempt_id}")
    expected_files = set(remote_inventory) | {
        "REMOTE_MANIFEST.json",
        "LOCAL_COPY_VERIFICATION.json",
    }
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        extra = sorted(actual_files - expected_files)
        raise AnalysisError(
            f"materialized tree differs from its inventory for {attempt_id}; "
            f"missing={missing}, extra={extra}"
        )
    for relative, (expected_size, expected_digest) in remote_inventory.items():
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise AnalysisError(f"materialized artifact is absent or a symlink: {path}")
        if path.stat().st_size != expected_size or _sha256(path) != expected_digest:
            raise AnalysisError(f"materialized artifact hash mismatch: {path}")
    return remote_digest


def _verify_runner_manifest(runner_root: Path) -> None:
    manifest_path = runner_root / "artifact-manifest.json"
    manifest = _read_json(manifest_path)
    records = manifest.get("artifacts")
    if not isinstance(records, list) or not records:
        raise AnalysisError(f"runner artifact manifest is empty: {manifest_path}")
    expected: set[str] = {"artifact-manifest.json"}
    for record in records:
        if not isinstance(record, dict):
            raise AnalysisError(f"runner artifact manifest has a non-object: {manifest_path}")
        relative = record.get("relative_path")
        size = record.get("bytes")
        digest = record.get("sha256")
        count = record.get("records")
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
        ):
            raise AnalysisError(f"runner artifact manifest has an invalid entry: {manifest_path}")
        path = runner_root / relative
        expected.add(relative)
        if path.is_symlink() or not path.is_file():
            raise AnalysisError(f"runner artifact is absent or a symlink: {path}")
        if path.stat().st_size != size or _sha256(path) != digest:
            raise AnalysisError(f"runner artifact hash mismatch: {path}")
        if count is not None:
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise AnalysisError(f"runner artifact has an invalid record count: {path}")
            actual_count = sum(1 for line in path.read_text().splitlines() if line.strip())
            if actual_count != count:
                raise AnalysisError(f"runner artifact record count mismatch: {path}")
    actual = {
        path.relative_to(runner_root).as_posix()
        for path in runner_root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if actual != expected:
        raise AnalysisError(
            f"runner artifact manifest coverage mismatch; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _trace_assertion(level: str) -> str:
    if level == _TRACE_CONTROL_LEVEL:
        return "runtime_trace_extraction_disabled_for_overhead_control"
    return f"runtime_trace_level_{level}_explicitly_selected"


def _load_trial(root: Path) -> LoadedTrial:
    attempt_id = root.name
    remote_digest = _verify_materialized_tree(root, attempt_id)
    completion = _read_json(root / "function-completion.json")
    config = _read_json(root / "wrapper/modal-config.json")
    completion_status = completion.get("status")
    if completion_status not in {"succeeded", "failed"} or completion.get("error") is not None:
        raise AnalysisError(f"attempt has no completed benchmark measurement: {attempt_id}")
    if completion.get("attempt_id") != attempt_id or config.get("attempt_id") != attempt_id:
        raise AnalysisError(f"attempt identity changed inside {attempt_id}")
    if config.get("runtime") != EXPECTED_RUNTIME or config.get("runtime_version") != (
        EXPECTED_RUNTIME_VERSION
    ):
        raise AnalysisError(f"unsupported runtime identity in {attempt_id}")
    summary_payload = completion.get("summary")
    if not isinstance(summary_payload, dict):
        raise AnalysisError(f"attempt has no trial summary: {attempt_id}")
    wrapper_summary = _read_json(root / "wrapper/trial-summary.json")
    invocation = _read_json(root / "wrapper/invocation.json")
    model_manifest_reference = _read_json(root / "wrapper/model-manifest-reference.json")
    dependency_versions = _read_json(root / "wrapper/dependency-versions.json")
    expected_path = (
        TrialPath.INDEPENDENT_PREFILL
        if config.get("baseline_mode") == "independent"
        else TrialPath.SHARED_ROOT
    )
    runner_root = root / "runner" / expected_path.value
    _verify_runner_manifest(runner_root)
    runner_summary = _read_json(runner_root / "metrics/trial-summary.json")
    if summary_payload != wrapper_summary or summary_payload != runner_summary:
        raise AnalysisError(f"summary copies are not byte-semantically equal for {attempt_id}")
    try:
        summary = TrialSummary.model_validate_json(_canonical_bytes(summary_payload))
    except ValidationError as error:
        raise AnalysisError(f"invalid TrialSummary in {attempt_id}: {error}") from error
    expected = {
        "path": expected_path.value,
        "fanout": config.get("fanout"),
        "prefix_tokens": config.get("prefix_length"),
        "suffix_tokens": config.get("suffix_length"),
        "seed": config.get("seed"),
    }
    observed = {
        "path": summary.path.value,
        "fanout": summary.fanout,
        "prefix_tokens": summary.prefix_tokens,
        "suffix_tokens": summary.suffix_tokens,
        "seed": summary.seed,
    }
    if observed != expected:
        raise AnalysisError(f"config and trial summary disagree for {attempt_id}")
    level = config.get("tracing_level")
    if level not in {_PRIMARY_TRACE_LEVEL, _TRACE_CONTROL_LEVEL, "minimal"}:
        raise AnalysisError(f"unsupported trace condition in {attempt_id}")
    invocation_experiment = invocation.get("experiment")
    invocation_engine = invocation.get("engine")
    runner_configuration = _read_json(runner_root / "configuration.json")
    if (
        invocation.get("schema_version")
        != "sloforge.branchfabric.vllm-gpu-validation-invocation/v1"
        or not isinstance(invocation_experiment, dict)
        or not isinstance(invocation_engine, dict)
        or invocation_experiment != runner_configuration
    ):
        raise AnalysisError(f"invocation and runner configuration disagree for {attempt_id}")
    prefix_token_ids = invocation_experiment.get("prefix_token_ids")
    divergent_token_ids = invocation_experiment.get("divergent_token_ids")
    if (
        not isinstance(prefix_token_ids, list)
        or len(prefix_token_ids) != summary.prefix_tokens
        or not all(
            isinstance(token, int) and not isinstance(token, bool) and token >= 0
            for token in prefix_token_ids
        )
        or not isinstance(divergent_token_ids, list)
        or len(divergent_token_ids) != summary.fanout
        or len(set(divergent_token_ids)) != summary.fanout
        or invocation_experiment.get("trace_level") != level
    ):
        raise AnalysisError(f"invocation token or trace material is invalid for {attempt_id}")
    if (
        invocation_engine.get("max_num_seqs", 0) < summary.fanout
        or invocation_engine.get("max_model_len", 0)
        < summary.prefix_tokens + summary.suffix_tokens + 2
        or invocation_engine.get("block_size") != 16
    ):
        raise AnalysisError(f"invocation engine does not cover the workload for {attempt_id}")
    identity = summary.identity
    if (
        identity.runtime != config.get("runtime")
        or identity.runtime_version != config.get("runtime_version")
        or identity.model_id != config.get("model")
        or identity.model_revision != config.get("model_revision")
        or identity.tokenizer_revision != config.get("tokenizer_revision")
    ):
        raise AnalysisError(f"runtime/model/tokenizer identity mismatch for {attempt_id}")
    if _trace_assertion(str(level)) not in summary.assertions:
        raise AnalysisError(f"trace condition lacks a matching runner assertion in {attempt_id}")
    if (
        model_manifest_reference.get("model_id") != identity.model_id
        or model_manifest_reference.get("model_revision") != identity.model_revision
        or model_manifest_reference.get("tokenizer_revision") != identity.tokenizer_revision
        or not isinstance(model_manifest_reference.get("sha256"), str)
        or len(str(model_manifest_reference["sha256"])) != 64
    ):
        raise AnalysisError(f"model manifest reference is invalid for {attempt_id}")
    packages = dependency_versions.get("packages")
    if (
        not isinstance(packages, dict)
        or packages.get("vllm") != EXPECTED_RUNTIME_VERSION
        or packages.get("torch") is None
        or packages.get("transformers") is None
        or dependency_versions.get("cuda_available") is not True
    ):
        raise AnalysisError(f"dependency provenance is invalid for {attempt_id}")
    actual_gpu = completion.get("actual_gpu")
    if not isinstance(actual_gpu, dict):
        raise AnalysisError(f"completed attempt lacks actual GPU identity: {attempt_id}")
    gpu_uuid = actual_gpu.get("uuid")
    gpu_name = actual_gpu.get("name")
    memory_total_mib = actual_gpu.get("memory_total_mib")
    if (
        completion.get("requested_gpu") != EXPECTED_GPU_SKU
        or completion.get("gpu_count") != 1
        or not isinstance(gpu_uuid, str)
        or not gpu_uuid
        or not isinstance(gpu_name, str)
        or "A100" not in gpu_name
        or not isinstance(memory_total_mib, int)
        or memory_total_mib < 80_000
    ):
        raise AnalysisError(f"completed attempt did not use one A100 80GB: {attempt_id}")
    postflight = _read_json(root / "wrapper/postflight.json")
    lifecycle = _load_lifecycle_evidence(
        root,
        attempt_id=attempt_id,
        gpu_uuid=gpu_uuid,
        completion=completion,
        wrapper_postflight=postflight,
    )
    postflight_clean = (
        lifecycle.cleanup_gate["GPU_RUNTIME_LIFECYCLE_CLEAN"] is True
        if lifecycle is not None
        else (
            postflight.get("memory_recovered_within_1_gib") is True
            and postflight.get("unexpected_compute_processes") == []
        )
    )
    if completion_status == "succeeded" and not postflight_clean:
        raise AnalysisError(f"successful attempt lacks a clean GPU postflight: {attempt_id}")
    if completion_status == "failed":
        if postflight_clean:
            raise AnalysisError(f"failed measurement has a clean postflight: {attempt_id}")
        if (root / "wrapper/failure.json").exists() or (
            root / "wrapper/cleanup-failure.json"
        ).exists():
            raise AnalysisError(
                f"failed measurement has a benchmark or adapter-cleanup failure: {attempt_id}"
            )
        if lifecycle is None and (
            not isinstance(postflight.get("error_type"), str)
            or not isinstance(postflight.get("error_message"), str)
        ):
            raise AnalysisError(
                f"failed measurement is not attributable to the postflight gate: {attempt_id}"
            )
    physical_rows: list[tuple[str, PhysicalKvLayoutSnapshot]] = []
    for row in _read_jsonl(runner_root / "raw/physical-snapshots.jsonl"):
        phase = row.get("phase")
        snapshot_payload = row.get("snapshot")
        if not isinstance(phase, str) or not isinstance(snapshot_payload, dict):
            raise AnalysisError(f"invalid physical snapshot row in {attempt_id}")
        try:
            snapshot = PhysicalKvLayoutSnapshot.model_validate_json(
                _canonical_bytes(snapshot_payload)
            )
        except ValidationError as error:
            raise AnalysisError(f"invalid physical snapshot in {attempt_id}: {error}") from error
        physical_rows.append((phase, snapshot))
    if not physical_rows:
        raise AnalysisError(f"attempt has no physical KV snapshots: {attempt_id}")
    final_phase = f"suffix_{summary.suffix_tokens}"
    final_matches = [snapshot for phase, snapshot in physical_rows if phase == final_phase]
    if len(final_matches) != 1:
        raise AnalysisError(f"attempt lacks one {final_phase} physical snapshot: {attempt_id}")
    final = final_matches[0]
    if (
        final.physical_assigned_bytes != summary.final_physical_bytes
        or final.shared_prefix_bytes != summary.final_shared_prefix_bytes
        or final.private_suffix_bytes != summary.final_private_suffix_bytes
    ):
        raise AnalysisError(f"final physical snapshot disagrees with summary: {attempt_id}")
    phase_rows = _read_jsonl(runner_root / "raw/phase-timings.jsonl")
    if list(phase_rows) != list(summary.phase_metrics):
        raise AnalysisError(f"raw phase timings disagree with summary: {attempt_id}")
    return LoadedTrial(
        attempt_id=attempt_id,
        root=root,
        config=config,
        completion=completion,
        invocation=invocation,
        runner_configuration=runner_configuration,
        model_manifest_reference=model_manifest_reference,
        dependency_versions=dependency_versions,
        summary=summary,
        remote_manifest_sha256=remote_digest,
        gpu_uuid=gpu_uuid,
        gpu_name=gpu_name,
        physical_rows=tuple(physical_rows),
        gpu_memory_rows=_read_jsonl(runner_root / "raw/gpu-memory-states.jsonl"),
        sampler_rows=_read_jsonl(runner_root / "raw/nvidia-smi-samples.jsonl"),
        phase_rows=phase_rows,
        release_rows=_read_jsonl(runner_root / "raw/block-release-evidence.jsonl"),
        postflight_clean=postflight_clean,
        lifecycle=lifecycle,
    )


def _load_trials(experiment_root: Path) -> tuple[LoadedTrial, ...]:
    modal_root = experiment_root / "modal"
    if not modal_root.is_dir():
        raise AnalysisError(f"Modal artifact root is absent: {modal_root}")
    trial_roots = sorted(
        path
        for path in modal_root.iterdir()
        if path.is_dir() and (path / "function-completion.json").is_file()
    )
    if not trial_roots:
        raise AnalysisError("no locally materialized Modal attempts are present")
    trials: list[LoadedTrial] = []
    for path in trial_roots:
        _verify_materialized_tree(path, path.name)
        completion = _read_json(path / "function-completion.json")
        if isinstance(completion.get("summary"), dict):
            trials.append(_load_trial(path))
    if not trials:
        raise AnalysisError("no materialized Modal attempt completed a benchmark workload")
    attempt_ids = [trial.attempt_id for trial in trials]
    if len(attempt_ids) != len(set(attempt_ids)):
        raise AnalysisError("materialized Modal attempt IDs are not unique")
    return tuple(trials)


def _ledger(experiment_root: Path, trials: Sequence[LoadedTrial]) -> dict[str, Any]:
    ledger = _read_json(experiment_root / "gpu-hours.json")
    if (
        ledger.get("modal_in_flight_reservations") != []
        or ledger.get("modal_in_flight_model_reservations") != []
    ):
        raise AnalysisError("GPU-hour ledger retains an in-flight Modal reservation")
    intervals = ledger.get("gpu_active_intervals")
    if not isinstance(intervals, list):
        raise AnalysisError("GPU-hour ledger has no interval list")
    cloud_intervals = ledger.get("cloud_compute_intervals")
    if not isinstance(cloud_intervals, list):
        raise AnalysisError("GPU-hour ledger has no cloud-compute interval list")

    calls_root = experiment_root / "modal/calls"
    call_paths = sorted(calls_root.glob("*.json")) if calls_root.is_dir() else []
    call_records: list[dict[str, Any]] = []
    for path in call_paths:
        record = _read_json(path)
        if (
            record.get("schema_version") != "sloforge.branchfabric.modal-call-state/v1"
            or not isinstance(record.get("attempt_id"), str)
            or not isinstance(record.get("function_call_id"), str)
            or record.get("kind") not in {"gpu", "model-prepare"}
        ):
            raise AnalysisError(f"invalid Modal call record: {path}")
        call_records.append(record)
    call_keys = {
        (
            str(record["attempt_id"]),
            str(record["function_call_id"]),
            str(record["kind"]).replace("-", "_"),
        )
        for record in call_records
    }
    cloud_keys = {
        (
            str(interval.get("attempt_id")),
            str(interval.get("function_call_id")),
            str(interval.get("kind")),
        )
        for interval in cloud_intervals
        if isinstance(interval, dict)
    }
    if len(call_keys) != len(call_records) or len(cloud_keys) != len(cloud_intervals):
        raise AnalysisError("Modal call records or cloud intervals are not unique")
    if call_keys != cloud_keys:
        raise AnalysisError("Modal call records do not reconcile to cloud-compute intervals")
    by_attempt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for interval in intervals:
        if isinstance(interval, dict) and isinstance(interval.get("attempt_id"), str):
            by_attempt[str(interval["attempt_id"])].append(cast(dict[str, Any], interval))
    for trial in trials:
        matches = by_attempt.get(trial.attempt_id, [])
        if len(matches) != 1:
            raise AnalysisError(f"GPU-hour ledger does not contain exactly one {trial.attempt_id}")
        interval = matches[0]
        if (
            interval.get("status") != trial.completion.get("status")
            or interval.get("function_call_id") != trial.completion.get("function_call_id")
            or interval.get("remote_manifest_sha256") != trial.remote_manifest_sha256
            or interval.get("actual_gpu") != trial.completion.get("actual_gpu")
        ):
            raise AnalysisError(f"GPU-hour ledger provenance mismatch for {trial.attempt_id}")
    consumed = ledger.get("consumed_gpu_hours")
    if not isinstance(consumed, (int, float)) or isinstance(consumed, bool) or consumed < 0:
        raise AnalysisError("GPU-hour ledger total is invalid")
    summed = sum(float(item.get("gpu_hours", 0.0)) for item in intervals if isinstance(item, dict))
    if not math.isclose(float(consumed), summed, rel_tol=0.0, abs_tol=1e-12):
        raise AnalysisError("GPU-hour ledger total does not equal its intervals")
    if float(consumed) > 4.0:
        raise AnalysisError("GPU-hour ledger exceeds the experiment hard limit")
    consumed_cost = ledger.get("consumed_cloud_cost_usd")
    if (
        not isinstance(consumed_cost, (int, float))
        or isinstance(consumed_cost, bool)
        or consumed_cost < 0
    ):
        raise AnalysisError("GPU-hour ledger cloud-cost upper bound is invalid")
    summed_cost = sum(
        float(item.get("cost_upper_bound_usd", 0.0))
        for item in cloud_intervals
        if isinstance(item, dict)
    )
    if not math.isclose(float(consumed_cost), summed_cost, rel_tol=0.0, abs_tol=1e-12):
        raise AnalysisError("GPU-hour ledger cloud-cost total does not equal its intervals")
    return ledger


def _pair_key(trial: LoadedTrial, *, include_trace: bool = True) -> tuple[object, ...]:
    values: tuple[object, ...] = (trial.scientific_input_sha256,)
    return (*values, trial.tracing_level) if include_trace else values


def _median(values: Iterable[float | int]) -> float:
    materialized = [float(value) for value in values]
    if not materialized:
        raise AnalysisError("cannot aggregate an empty metric")
    return float(statistics.median(materialized))


def _distribution(values: Iterable[float | int]) -> dict[str, float | int]:
    materialized = [float(value) for value in values]
    if not materialized:
        raise AnalysisError("cannot summarize an empty distribution")
    return {
        "count": len(materialized),
        "minimum": min(materialized),
        "median": float(statistics.median(materialized)),
        "mean": float(statistics.fmean(materialized)),
        "maximum": max(materialized),
    }


def _phase_metric(trial: LoadedTrial, phase: str, field: str) -> int:
    rows = [row for row in trial.phase_rows if row.get("phase") == phase]
    if len(rows) != 1:
        raise AnalysisError(f"{trial.attempt_id} does not contain exactly one {phase} phase")
    value = rows[0].get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise AnalysisError(f"{trial.attempt_id} has invalid {phase}.{field}")
    return value


def _snapshot(trial: LoadedTrial, phase: str) -> PhysicalKvLayoutSnapshot:
    matches = [snapshot for row_phase, snapshot in trial.physical_rows if row_phase == phase]
    if len(matches) != 1:
        raise AnalysisError(f"{trial.attempt_id} does not contain exactly one {phase} snapshot")
    return matches[0]


def _validate_physical_invariants(trial: LoadedTrial) -> None:
    phases = [phase for phase, _ in trial.physical_rows]
    checkpoints = ["divergence"]
    if trial.suffix_length >= 16:
        checkpoints.append("suffix_16")
    if trial.suffix_length >= 64:
        checkpoints.append("suffix_64")
    if trial.suffix_length >= 256:
        checkpoints.append("suffix_256")
    required = (
        ["root", "fork", *checkpoints] if trial.baseline_mode == "shared_root" else checkpoints
    )
    positions: list[int] = []
    for phase in required:
        if phases.count(phase) != 1:
            raise AnalysisError(f"{trial.attempt_id} does not contain exactly one {phase} phase")
        positions.append(phases.index(phase))
    if positions != sorted(positions):
        raise AnalysisError(f"physical snapshot phases are out of order: {trial.attempt_id}")
    branch_ids = set(trial.summary.branch_ready_latency_ns)
    previous_private: set[str] = set()
    if trial.baseline_mode == "shared_root":
        root = _snapshot(trial, "root")
        root_ids = set(root.shared_prefix_block_ids)
        if not root_ids or root.shared_prefix_bytes != trial.summary.root_physical_bytes:
            raise AnalysisError(f"shared root snapshot is incomplete: {trial.attempt_id}")
        fork = _snapshot(trial, "fork")
        if set(fork.shared_prefix_block_ids) != root_ids or fork.shared_prefix_bytes != (
            root.shared_prefix_bytes
        ):
            raise AnalysisError(f"fork multiplied or changed the shared root: {trial.attempt_id}")
        for phase in checkpoints:
            snapshot = _snapshot(trial, phase)
            if (
                set(snapshot.shared_prefix_block_ids) != root_ids
                or snapshot.shared_prefix_bytes != root.shared_prefix_bytes
            ):
                raise AnalysisError(f"shared root changed at {phase}: {trial.attempt_id}")
            blocks = {block.runtime_block_id: block for block in snapshot.blocks}
            for block_id in root_ids:
                block = blocks.get(block_id)
                if (
                    block is None
                    or set(block.branch_ids) != branch_ids
                    or block.refcount < trial.fanout
                ):
                    raise AnalysisError(
                        f"root block lacks native owners/refcounts at {phase}: {trial.attempt_id}"
                    )
            private = set(snapshot.private_suffix_block_ids)
            if phase == "divergence" and not private:
                raise AnalysisError(f"private divergence was not observed: {trial.attempt_id}")
            if not previous_private.issubset(private):
                raise AnalysisError(f"private allocation was not append-only: {trial.attempt_id}")
            previous_private = private
    else:
        for phase in checkpoints:
            snapshot = _snapshot(trial, phase)
            if snapshot.shared_prefix_block_ids or snapshot.shared_prefix_bytes:
                raise AnalysisError(f"independent path contains shared KV at {phase}")
            owners: dict[str, set[str]] = defaultdict(set)
            for block in snapshot.blocks:
                if len(block.branch_ids) != 1:
                    raise AnalysisError(
                        f"independent branches alias block {block.runtime_block_id} at {phase}"
                    )
                owners[block.runtime_block_id].update(block.branch_ids)
            if any(len(value) != 1 for value in owners.values()):
                raise AnalysisError(f"independent block ownership is not disjoint at {phase}")


def _validate_smoke_release(trial: LoadedTrial) -> None:
    by_phase = {
        str(row.get("phase")): row
        for row in trial.release_rows
        if isinstance(row.get("phase"), str)
    }
    if set(by_phase) != {
        "pre_branch_a_destroy",
        "post_branch_a_destroy",
        "post_root_destroy",
    }:
        raise AnalysisError(f"smoke release evidence phases are incomplete: {trial.attempt_id}")
    branch_release = by_phase["post_branch_a_destroy"]
    root_release = by_phase["post_root_destroy"]
    for label, row in (("branch", branch_release), ("root", root_release)):
        requested = row.get("requested_block_ids")
        blocks = row.get("blocks")
        if not isinstance(requested, list) or not requested or not isinstance(blocks, list):
            raise AnalysisError(f"smoke {label} release has no exact block evidence")
        observed = {
            block.get("runtime_block_id"): block for block in blocks if isinstance(block, dict)
        }
        if set(observed) != set(requested):
            raise AnalysisError(f"smoke {label} release does not cover exact former IDs")
        for block in observed.values():
            if block.get("native_refcount") != 0 or block.get("allocator_available") is not True:
                raise AnalysisError(f"smoke {label} release left a natively owned block")
            if label == "root" and block.get("block_hash_present") is not False:
                raise AnalysisError("final smoke root release retained a block hash")
    if root_release.get("pool_free_block_count") != root_release.get("pool_usable_block_count"):
        raise AnalysisError("final smoke release did not recover the usable KV pool")
    root = _snapshot(trial, "root")
    post_branch = _snapshot(trial, "post_branch_delete")
    post_root = _snapshot(trial, "post_root_delete")
    if set(post_branch.shared_prefix_block_ids) != set(root.shared_prefix_block_ids):
        raise AnalysisError("smoke branch deletion did not preserve its sibling's shared root")
    if post_root.blocks or post_root.physical_assigned_bytes != 0:
        raise AnalysisError("smoke final root deletion left assigned physical state")


def _theoretical_shared_floor(trial: LoadedTrial, *, phase: str | None = None) -> int:
    final = _snapshot(trial, phase or f"suffix_{trial.suffix_length}")
    if trial.baseline_mode != "shared_root" or not final.shared_prefix_block_ids:
        raise AnalysisError("theoretical shared floor requires one shared-root trial")
    blocks = final.blocks
    page_sizes = {block.bytes for block in blocks}
    block_sizes = {block.block_size_tokens for block in blocks}
    if len(page_sizes) != 1 or len(block_sizes) != 1:
        raise AnalysisError(f"{trial.attempt_id} uses nonuniform physical KV pages")
    page_bytes = next(iter(page_sizes))
    block_tokens = next(iter(block_sizes))
    private_floor = 0
    branch_ids = set(trial.summary.branch_ready_latency_ns)
    for branch_id in branch_ids:
        private = [
            block
            for block in blocks
            if block.runtime_block_id in set(final.private_suffix_block_ids)
            and branch_id in block.branch_ids
        ]
        if not private:
            raise AnalysisError(f"{trial.attempt_id} lacks private state for {branch_id}")
        if any(len(block.branch_ids) != 1 for block in private):
            raise AnalysisError(f"{trial.attempt_id} aliases a private suffix page")
        logical_end = max(block.logical_token_end_exclusive for block in private)
        private_tokens = logical_end - trial.prefix_length
        if private_tokens <= 0:
            raise AnalysisError(f"{trial.attempt_id} has invalid private logical coverage")
        private_floor += math.ceil(private_tokens / block_tokens) * page_bytes
    return final.shared_prefix_bytes + private_floor


def _private_per_branch(trial: LoadedTrial) -> dict[str, int]:
    final = _snapshot(trial, f"suffix_{trial.suffix_length}")
    private_ids = set(final.private_suffix_block_ids)
    return {
        branch_id: sum(
            block.bytes
            for block in final.blocks
            if block.runtime_block_id in private_ids and branch_id in block.branch_ids
        )
        for branch_id in trial.summary.branch_ready_latency_ns
    }


def _independent_physical_accounting(trial: LoadedTrial) -> dict[str, Any]:
    if trial.baseline_mode != "independent":
        raise AnalysisError("independent physical accounting requires an independent trial")
    required_assertions = {
        "independent_prefix_cache_hits_absent",
        "independent_branch_physical_block_sets_disjoint",
        "independent_prefix_physical_bytes_scale_with_fanout",
        "every_branch_executes_complete_prefix_prefill",
    }
    if not required_assertions.issubset(trial.summary.assertions):
        raise AnalysisError(
            f"independent terminal estimate lacks baseline assertions: {trial.attempt_id}"
        )
    measured = [
        (phase, snapshot)
        for phase, snapshot in trial.physical_rows
        if phase in {"divergence", "suffix_16", "suffix_64", "suffix_256"}
    ]
    if not measured:
        raise AnalysisError(
            f"independent trial has no physical measurement phases: {trial.attempt_id}"
        )
    peak_phase, peak = max(measured, key=lambda item: item[1].physical_assigned_bytes)
    terminal = _snapshot(trial, f"suffix_{trial.suffix_length}")
    terminal_owners = {branch_id for block in terminal.blocks for branch_id in block.branch_ids}
    terminal_bytes = sum(block.bytes for block in terminal.blocks)
    expected_logical_end = trial.prefix_length + trial.suffix_length
    if (
        len(terminal_owners) != 1
        or not terminal.blocks
        or any(len(block.branch_ids) != 1 for block in terminal.blocks)
        or terminal_bytes != terminal.physical_assigned_bytes
        or max(block.logical_token_end_exclusive for block in terminal.blocks)
        != expected_logical_end
    ):
        raise AnalysisError(
            f"independent terminal snapshot is not one exact complete branch: {trial.attempt_id}"
        )
    estimate = terminal.physical_assigned_bytes * trial.fanout
    if peak.physical_assigned_bytes > estimate:
        raise AnalysisError(
            f"independent observed peak exceeds its full-fanout terminal estimate: "
            f"{trial.attempt_id}"
        )
    return {
        "observed_peak_bytes": peak.physical_assigned_bytes,
        "observed_peak_phase": peak_phase,
        "observed_terminal_survivor_bytes": terminal.physical_assigned_bytes,
        "observed_terminal_survivor_branch_id": next(iter(terminal_owners)),
        "full_fanout_terminal_estimate_bytes": estimate,
        "estimate_minus_observed_peak_bytes": estimate - peak.physical_assigned_bytes,
        "estimate_method": (
            "exact_terminal_survivor_physical_pages_times_fanout; all branches asserted "
            "independent, prefix-cache-free, disjoint, and fully prefilled"
        ),
        "estimate_is_direct_observation": False,
    }


def _aligned_all_live_physical_accounting(
    independent: LoadedTrial, shared: LoadedTrial
) -> dict[str, Any]:
    if independent.fanout != shared.fanout:
        raise AnalysisError("aligned physical accounting requires equal fanout")
    expected_independent = set(independent.summary.branch_ready_latency_ns)
    expected_shared = set(shared.summary.branch_ready_latency_ns)
    ordered_phases = ("divergence", "suffix_16", "suffix_64", "suffix_256")
    common = {
        phase
        for phase in ordered_phases
        if any(row_phase == phase for row_phase, _ in independent.physical_rows)
        and any(row_phase == phase for row_phase, _ in shared.physical_rows)
    }
    qualifying: list[str] = []
    for phase in ordered_phases:
        if phase not in common:
            continue
        independent_snapshot = _snapshot(independent, phase)
        shared_snapshot = _snapshot(shared, phase)
        independent_owners = {
            branch_id for block in independent_snapshot.blocks for branch_id in block.branch_ids
        }
        shared_owners = {
            branch_id for block in shared_snapshot.blocks for branch_id in block.branch_ids
        }
        if independent_owners == expected_independent and shared_owners == expected_shared:
            qualifying.append(phase)
    if not qualifying:
        raise AnalysisError(
            f"paired trials have no aligned all-branches-live physical checkpoint: "
            f"{independent.attempt_id}, {shared.attempt_id}"
        )
    phase = qualifying[-1]
    independent_snapshot = _snapshot(independent, phase)
    shared_snapshot = _snapshot(shared, phase)
    floor = _theoretical_shared_floor(shared, phase=phase)
    return {
        "phase": phase,
        "measurement_kind": "direct_aligned_all_branches_live_physical_snapshots",
        "independent_actual_bytes": independent_snapshot.physical_assigned_bytes,
        "shared_actual_bytes": shared_snapshot.physical_assigned_bytes,
        "sharing_savings_bytes": (
            independent_snapshot.physical_assigned_bytes - shared_snapshot.physical_assigned_bytes
        ),
        "sharing_efficiency": (
            1.0
            - shared_snapshot.physical_assigned_bytes / independent_snapshot.physical_assigned_bytes
        ),
        "theoretical_shared_floor_bytes": floor,
        "physical_amplification": shared_snapshot.physical_assigned_bytes / floor,
        "cow_floor_gap_bytes": shared_snapshot.physical_assigned_bytes - floor,
    }


def _shared_root_inclusive_readiness_ns(trial: LoadedTrial) -> int:
    if trial.baseline_mode != "shared_root":
        raise AnalysisError("root-inclusive shared readiness requires a shared-root trial")
    root_start = _phase_metric(trial, "root_session_create", "wall_start_monotonic_ns")
    fork_start = _phase_metric(trial, "fork_metadata", "wall_start_monotonic_ns")
    fork_duration = _phase_metric(trial, "fork_metadata", "wall_duration_ns")
    decode_start = _phase_metric(trial, "concurrent_decode", "wall_start_monotonic_ns")
    runtime_readiness = trial.summary.all_branches_ready_ns - fork_duration
    if (
        fork_start < root_start
        or fork_duration < 0
        or decode_start < fork_start + fork_duration
        or runtime_readiness < 0
    ):
        raise AnalysisError(f"shared root phases are causally invalid: {trial.attempt_id}")
    return decode_start - root_start + runtime_readiness


def _validate_smoke_and_pairs(trials: Sequence[LoadedTrial]) -> tuple[LoadedTrial, ...]:
    for trial in trials:
        _validate_physical_invariants(trial)
    primary = [trial for trial in trials if trial.tracing_level == _PRIMARY_TRACE_LEVEL]
    smoke = [trial for trial in primary if trial.prefix_length in {2048, 4096}]
    all_contexts = [trial for trial in trials if trial.prefix_length == EXPECTED_PREFIX_TOKENS]
    contexts = [trial for trial in all_contexts if trial.tracing_level == _PRIMARY_TRACE_LEVEL]
    dirty_contexts = [
        trial.attempt_id for trial in all_contexts if trial.lifecycle_clean is not True
    ]
    if dirty_contexts:
        raise AnalysisError(
            "16K artifacts lack an authoritative clean controller/worker lifecycle: "
            + ", ".join(dirty_contexts)
        )
    if contexts and not any(trial.lifecycle_clean is True for trial in smoke):
        raise AnalysisError(
            "16K artifacts exist without an authoritative clean controller/worker smoke"
        )
    for trial in smoke:
        required = {
            "one_physical_shared_root_not_multiplied_by_fanout",
            "deleting_branch_a_preserves_branch_b_and_shared_root",
            "branch_a_former_private_ids_native_refcount_zero_and_allocator_available",
            "final_root_exact_ids_hash_cleared_and_complete_kv_pool_recovered",
        }
        if (
            trial.baseline_mode != "shared_root"
            or trial.fanout != 2
            or trial.suffix_length != 16
            or not required.issubset(trial.summary.assertions)
        ):
            raise AnalysisError(f"semantic smoke proof is incomplete: {trial.attempt_id}")
        _validate_smoke_release(trial)
    groups: dict[tuple[object, ...], dict[str, list[LoadedTrial]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for trial in contexts:
        groups[_pair_key(trial)][trial.baseline_mode].append(trial)
    for key, paths in groups.items():
        if set(paths) != {"independent", "shared_root"}:
            raise AnalysisError(f"16K comparison cell is missing a baseline path: {key}")
        if len(paths["independent"]) != 1 or len(paths["shared_root"]) != 1:
            raise AnalysisError(
                f"duplicate comparison cells are ambiguous without an explicit repetition ID: {key}"
            )
    controls = [trial for trial in trials if trial.tracing_level == _TRACE_CONTROL_LEVEL]
    full_shared_keys = {
        _pair_key(trial, include_trace=False)
        for trial in contexts
        if trial.baseline_mode == "shared_root"
    }
    for control in controls:
        if (
            control.baseline_mode != "shared_root"
            or _pair_key(control, include_trace=False) not in full_shared_keys
        ):
            raise AnalysisError(
                f"trace-disabled trial lacks an identical full shared-root peer: {control.attempt_id}"
            )
    control_keys = [_pair_key(trial, include_trace=False) for trial in controls]
    if len(control_keys) != len(set(control_keys)):
        raise AnalysisError("duplicate trace controls are ambiguous without a repetition ID")
    return tuple(smoke)


def _aggregate_cells(trials: Sequence[LoadedTrial]) -> list[dict[str, Any]]:
    primary = [
        trial
        for trial in trials
        if trial.tracing_level == _PRIMARY_TRACE_LEVEL
        and trial.prefix_length == EXPECTED_PREFIX_TOKENS
    ]
    by_cell: dict[tuple[int, int], dict[str, list[LoadedTrial]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for trial in primary:
        by_cell[(trial.seed, trial.fanout)][trial.baseline_mode].append(trial)
    cells: list[dict[str, Any]] = []
    for (seed, fanout), paths in sorted(by_cell.items()):
        independent = sorted(paths["independent"], key=lambda item: item.attempt_id)
        shared = sorted(paths["shared_root"], key=lambda item: item.attempt_id)
        independent_ready = _median(item.summary.all_branches_ready_ns for item in independent)
        shared_ready = _median(item.summary.all_branches_ready_ns for item in shared)
        shared_root_inclusive_ready = _median(
            _shared_root_inclusive_readiness_ns(item) for item in shared
        )
        independent_accounting = [_independent_physical_accounting(item) for item in independent]
        independent_observed_peak = _median(
            int(item["observed_peak_bytes"]) for item in independent_accounting
        )
        independent_terminal_survivor = _median(
            int(item["observed_terminal_survivor_bytes"]) for item in independent_accounting
        )
        independent_full_fanout_estimate = _median(
            int(item["full_fanout_terminal_estimate_bytes"]) for item in independent_accounting
        )
        aligned_accounting = [
            _aligned_all_live_physical_accounting(left, right)
            for left, right in zip(independent, shared, strict=True)
        ]
        aligned_independent_physical = _median(
            int(item["independent_actual_bytes"]) for item in aligned_accounting
        )
        aligned_shared_physical = _median(
            int(item["shared_actual_bytes"]) for item in aligned_accounting
        )
        aligned_floor = _median(
            int(item["theoretical_shared_floor_bytes"]) for item in aligned_accounting
        )
        shared_physical = _median(item.summary.final_physical_bytes for item in shared)
        shared_root = _median(item.summary.final_shared_prefix_bytes for item in shared)
        shared_private = _median(item.summary.final_private_suffix_bytes for item in shared)
        independent_decode = _median(
            item.summary.decode_throughput_tokens_per_second for item in independent
        )
        shared_decode = _median(item.summary.decode_throughput_tokens_per_second for item in shared)
        floors = [_theoretical_shared_floor(item) for item in shared]
        theoretical_floor = _median(floors)
        state_fractions = [
            _phase_metric(item, "fork_metadata", "wall_duration_ns")
            / item.summary.all_branches_ready_ns
            for item in shared
        ]
        cpu_fractions = [
            _phase_metric(item, "fork_metadata", "process_cpu_duration_ns")
            / item.summary.all_branches_ready_ns
            for item in shared
        ]
        marginal_private = [
            value for item in shared for value in _private_per_branch(item).values()
        ]
        same_gpu = all(
            left.gpu_uuid == right.gpu_uuid for left, right in zip(independent, shared, strict=True)
        )
        same_gpu_variant = all(
            left.gpu_name == right.gpu_name for left, right in zip(independent, shared, strict=True)
        )
        causal_reasons: list[str] = []
        if not same_gpu:
            causal_reasons.append("different physical GPU UUIDs")
        if not same_gpu_variant:
            causal_reasons.append("different A100 model variants")
        causal_metrics = same_gpu and same_gpu_variant
        cells.append(
            {
                "seed": seed,
                "fanout": fanout,
                "repetitions_per_path": len(shared),
                "same_physical_gpu_for_paired_repetitions": same_gpu,
                "same_gpu_model_variant_for_paired_repetitions": same_gpu_variant,
                "gpu_uuids": sorted({item.gpu_uuid for item in (*independent, *shared)}),
                "independent_gpu_uuids": [item.gpu_uuid for item in independent],
                "shared_gpu_uuids": [item.gpu_uuid for item in shared],
                "gpu_model_variants": sorted({item.gpu_name for item in (*independent, *shared)}),
                "causal_pair_metrics_available": causal_metrics,
                "causal_pair_metrics_unavailable_reason": (
                    None
                    if causal_metrics
                    else "independent/shared calls used " + " and ".join(causal_reasons)
                ),
                "independent_attempt_ids": [item.attempt_id for item in independent],
                "shared_attempt_ids": [item.attempt_id for item in shared],
                "all_branches_ready_ns": {
                    "measurement_scope": {
                        "independent": ("root_inclusive_each_branch_performs_full_prefix_prefill"),
                        "shared_root": ("post_root_publish_branch_fork_and_decode_readiness"),
                        "cross_path_delta": (
                            "prefill_work_avoidance_comparison_with_different_measurement_origins"
                        ),
                    },
                    "independent": _distribution(
                        item.summary.all_branches_ready_ns for item in independent
                    ),
                    "shared_root": _distribution(
                        item.summary.all_branches_ready_ns for item in shared
                    ),
                    "theoretical_zero_cost": 0,
                },
                "root_inclusive_all_branches_ready_ns": {
                    "measurement_scope": (
                        "continuous wall time from path runtime/session start through concurrent "
                        "decode start plus measured runtime readiness; shared includes root "
                        "construction, publication, validation, and admission"
                    ),
                    "independent": _distribution(
                        item.summary.all_branches_ready_ns for item in independent
                    ),
                    "shared_root": _distribution(
                        _shared_root_inclusive_readiness_ns(item) for item in shared
                    ),
                },
                "physical_kv_bytes": {
                    "independent_observed_peak": _distribution(
                        int(item["observed_peak_bytes"]) for item in independent_accounting
                    ),
                    "independent_observed_peak_phases": [
                        str(item["observed_peak_phase"]) for item in independent_accounting
                    ],
                    "independent_observed_terminal_survivor": _distribution(
                        int(item["observed_terminal_survivor_bytes"])
                        for item in independent_accounting
                    ),
                    "independent_comparable_full_fanout_terminal_estimate": _distribution(
                        int(item["full_fanout_terminal_estimate_bytes"])
                        for item in independent_accounting
                    ),
                    "independent_estimate_provenance": independent_accounting,
                    "aligned_all_branches_live": {
                        "checkpoint_phases": [str(item["phase"]) for item in aligned_accounting],
                        "independent_actual": _distribution(
                            int(item["independent_actual_bytes"]) for item in aligned_accounting
                        ),
                        "shared_actual": _distribution(
                            int(item["shared_actual_bytes"]) for item in aligned_accounting
                        ),
                        "theoretical_shared_floor": _distribution(
                            int(item["theoretical_shared_floor_bytes"])
                            for item in aligned_accounting
                        ),
                        "pair_provenance": aligned_accounting,
                    },
                    "shared_full_fanout_terminal_actual": _distribution(
                        item.summary.final_physical_bytes for item in shared
                    ),
                    "shared_root": _distribution(
                        item.summary.final_physical_bytes for item in shared
                    ),
                    "theoretical_shared_floor": _distribution(floors),
                    "shared_prefix": shared_root,
                    "shared_private_suffix": shared_private,
                },
                "decode_tokens_per_second": {
                    "cross_path_comparison_available": causal_metrics,
                    "cross_path_comparison_unavailable_reason": (
                        None
                        if causal_metrics
                        else "observational values came from " + " and ".join(causal_reasons)
                    ),
                    "independent": _distribution(
                        item.summary.decode_throughput_tokens_per_second for item in independent
                    ),
                    "shared_root": _distribution(
                        item.summary.decode_throughput_tokens_per_second for item in shared
                    ),
                },
                "derived": {
                    "sharing_efficiency": (
                        None
                        if aligned_independent_physical == 0
                        else 1.0 - aligned_shared_physical / aligned_independent_physical
                    ),
                    "sharing_efficiency_scope": ("direct_aligned_all_branches_live_checkpoint"),
                    "sharing_savings_bytes": (
                        aligned_independent_physical - aligned_shared_physical
                    ),
                    "sharing_savings_scope": ("direct_aligned_all_branches_live_checkpoint"),
                    "observed_peak_savings_bytes": (
                        independent_observed_peak - aligned_shared_physical
                    ),
                    "projected_terminal_sharing_savings_bytes": (
                        independent_full_fanout_estimate - shared_physical
                    ),
                    "projected_terminal_sharing_efficiency": (
                        None
                        if independent_full_fanout_estimate == 0
                        else 1.0 - shared_physical / independent_full_fanout_estimate
                    ),
                    "independent_full_fanout_estimate_minus_observed_peak_bytes": (
                        independent_full_fanout_estimate - independent_observed_peak
                    ),
                    "independent_terminal_survivor_bytes": independent_terminal_survivor,
                    "marginal_private_kv_bytes_per_branch": _distribution(marginal_private),
                    "branch_readiness_overhead_ns_vs_zero_cost": shared_ready,
                    "post_root_branch_readiness_overhead_ns_vs_zero_cost": shared_ready,
                    "independent_minus_shared_readiness_ns": (
                        independent_ready - shared_ready if causal_metrics else None
                    ),
                    "readiness_speedup": (
                        independent_ready / shared_ready
                        if causal_metrics and shared_ready != 0
                        else None
                    ),
                    "root_inclusive_independent_minus_shared_readiness_ns": (
                        independent_ready - shared_root_inclusive_ready if causal_metrics else None
                    ),
                    "root_inclusive_readiness_speedup": (
                        independent_ready / shared_root_inclusive_ready
                        if causal_metrics and shared_root_inclusive_ready != 0
                        else None
                    ),
                    "prefill_work_avoided_tokens": EXPECTED_PREFIX_TOKENS * (fanout - 1),
                    "state_management_fraction": _distribution(state_fractions),
                    "post_root_state_management_fraction": _distribution(state_fractions),
                    "decode_interference_fraction": (
                        None
                        if independent_decode == 0 or not causal_metrics
                        else 1.0 - shared_decode / independent_decode
                    ),
                    "physical_amplification_over_shared_floor": (
                        None if aligned_floor == 0 else aligned_shared_physical / aligned_floor
                    ),
                    "physical_amplification": (
                        None if aligned_floor == 0 else aligned_shared_physical / aligned_floor
                    ),
                    "cow_floor_gap_bytes": aligned_shared_physical - aligned_floor,
                    "terminal_physical_amplification_over_shared_floor": (
                        None if theoretical_floor == 0 else shared_physical / theoretical_floor
                    ),
                    "terminal_cow_floor_gap_bytes": shared_physical - theoretical_floor,
                    "cpu_block_management_fraction_of_readiness": _distribution(cpu_fractions),
                    "post_root_cpu_block_management_fraction": _distribution(cpu_fractions),
                },
            }
        )
    return cells


def _tracing_overhead(trials: Sequence[LoadedTrial]) -> list[dict[str, Any]]:
    full: dict[tuple[object, ...], list[LoadedTrial]] = defaultdict(list)
    disabled: dict[tuple[object, ...], list[LoadedTrial]] = defaultdict(list)
    for trial in trials:
        if trial.baseline_mode != "shared_root" or trial.prefix_length != EXPECTED_PREFIX_TOKENS:
            continue
        key = _pair_key(trial, include_trace=False)
        if trial.tracing_level == _PRIMARY_TRACE_LEVEL:
            full[key].append(trial)
        elif trial.tracing_level == _TRACE_CONTROL_LEVEL:
            disabled[key].append(trial)
    rows: list[dict[str, Any]] = []
    for key in sorted(set(full) & set(disabled), key=str):
        full_trials = sorted(full[key], key=lambda item: item.attempt_id)
        disabled_trials = sorted(disabled[key], key=lambda item: item.attempt_id)
        if len(full_trials) != len(disabled_trials):
            raise AnalysisError(f"trace control repetition count mismatch: {key}")
        for measured, control in zip(full_trials, disabled_trials, strict=True):
            measured_seconds = measured.completion.get("benchmark_seconds")
            control_seconds = control.completion.get("benchmark_seconds")
            if (
                not isinstance(measured_seconds, (int, float))
                or isinstance(measured_seconds, bool)
                or not isinstance(control_seconds, (int, float))
                or isinstance(control_seconds, bool)
                or control_seconds <= 0
            ):
                raise AnalysisError("trace overhead pair lacks valid benchmark durations")
            runtime_control = control.summary.readiness_and_decode_elapsed_ns
            same_gpu = measured.gpu_uuid == control.gpu_uuid
            rows.append(
                {
                    "seed": measured.seed,
                    "fanout": measured.fanout,
                    "full_attempt_id": measured.attempt_id,
                    "disabled_attempt_id": control.attempt_id,
                    "same_physical_gpu": same_gpu,
                    "end_to_end_overhead_fraction": (
                        measured_seconds / control_seconds - 1.0 if same_gpu else None
                    ),
                    "readiness_and_decode_overhead_fraction": (
                        None
                        if runtime_control == 0 or not same_gpu
                        else measured.summary.readiness_and_decode_elapsed_ns / runtime_control
                        - 1.0
                    ),
                    "full_trace_self_wall_time_ns": (
                        measured.summary.trace_live_event_record_wall_time_ns
                    ),
                    "full_trace_layout_diff_wall_time_ns": (
                        measured.summary.trace_live_layout_diff_wall_time_ns
                    ),
                    "availability": (
                        "available_same_physical_gpu"
                        if same_gpu
                        else "unavailable_cross_instance_confounded"
                    ),
                }
            )
    return rows


def _interest_assessment(cells: Sequence[dict[str, Any]]) -> dict[str, Any]:
    signals: list[dict[str, Any]] = []

    def classify(
        *, gate: str, value: float | None, threshold: float, weak: float, fanout: int
    ) -> None:
        if value is None:
            return
        severity: Literal["interest", "weak", "none"]
        if value > threshold:
            severity = "interest"
        elif value >= weak:
            severity = "weak"
        else:
            severity = "none"
        signals.append(
            {
                "gate": gate,
                "fanout": fanout,
                "value": value,
                "interest_threshold": threshold,
                "weak_threshold": weak,
                "classification": severity,
            }
        )

    for cell in cells:
        fanout = int(cell["fanout"])
        derived = cast(dict[str, Any], cell["derived"])
        classify(
            gate="state_management_fraction_of_readiness",
            value=float(derived["state_management_fraction"]["median"]),
            threshold=0.15,
            weak=0.05,
            fanout=fanout,
        )
        interference = derived.get("decode_interference_fraction")
        classify(
            gate="decode_throughput_degradation",
            value=None if interference is None else float(interference),
            threshold=0.15,
            weak=0.05,
            fanout=fanout,
        )
        amplification = derived.get("physical_amplification_over_shared_floor")
        classify(
            gate="physical_allocation_above_shared_floor",
            value=None if amplification is None else float(amplification) - 1.0,
            threshold=0.20,
            weak=0.05,
            fanout=fanout,
        )
        classify(
            gate="cpu_block_management_fraction_of_readiness",
            value=float(derived["cpu_block_management_fraction_of_readiness"]["median"]),
            threshold=0.10,
            weak=0.05,
            fanout=fanout,
        )
    by_fanout = {int(cell["fanout"]): cell for cell in cells}
    comparable_scaling_uuid = (
        8 in by_fanout
        and 32 in by_fanout
        and by_fanout[8]["shared_gpu_uuids"] == by_fanout[32]["shared_gpu_uuids"]
    )
    if comparable_scaling_uuid:
        ready_8 = float(by_fanout[8]["all_branches_ready_ns"]["shared_root"]["median"])
        ready_32 = float(by_fanout[32]["all_branches_ready_ns"]["shared_root"]["median"])
        factor = ready_32 / ready_8 / 4.0 if ready_8 else None
        classify(
            gate="readiness_superlinear_fanout_8_to_32",
            value=factor,
            threshold=1.15,
            weak=1.05,
            fanout=32,
        )
    unavailable = [
        {
            "gate": "memory_copy_contends_with_decode",
            "reason": "copy-engine/CUPTI counters are unavailable",
        },
        {
            "gate": "hbm_bandwidth_reduces_rollout_concurrency",
            "reason": "nvidia-smi memory utilization is not attributable HBM bandwidth",
        },
        {
            "gate": "multi_gib_state_movement_on_critical_path",
            "reason": "allocation deltas do not prove physical byte movement",
        },
    ]
    observed_gates = {str(row["gate"]) for row in signals}
    cell_dependent_gates = {
        "state_management_fraction_of_readiness": (
            "paired 16K independent/shared readiness and state-operation timing are required"
        ),
        "decode_throughput_degradation": (
            "paired 16K independent/shared concurrent-decode trials are required"
        ),
        "physical_allocation_above_shared_floor": (
            "paired 16K physical KV accounting and a theoretical shared floor are required"
        ),
        "cpu_block_management_fraction_of_readiness": (
            "paired 16K readiness and measured CPU block-management time are required"
        ),
    }
    for gate, reason in cell_dependent_gates.items():
        if gate not in observed_gates:
            if gate == "decode_throughput_degradation" and cells:
                confounders = sorted(
                    {
                        str(cell["causal_pair_metrics_unavailable_reason"])
                        for cell in cells
                        if cell.get("causal_pair_metrics_unavailable_reason") is not None
                    }
                )
                reason = (
                    "observed per-path throughput is retained, but causal decode interference "
                    "is unavailable: " + "; ".join(confounders)
                )
            unavailable.append({"gate": gate, "reason": reason})
    if not comparable_scaling_uuid:
        unavailable.append(
            {
                "gate": "readiness_superlinear_fanout_8_to_32",
                "reason": (
                    "valid fanout-8/fanout-32 shared trials on the same physical GPU UUID "
                    "are required"
                ),
            }
        )
    return {
        "signals": signals,
        "interest_signals": [row for row in signals if row["classification"] == "interest"],
        "weak_signals": [row for row in signals if row["classification"] == "weak"],
        "unavailable_gates": unavailable,
        "fpga_justification_claimed": False,
    }


def _scaling(cells: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        grouped[int(cell["fanout"])].append(cell)
    medians: dict[int, dict[str, float]] = {}
    for fanout, items in grouped.items():
        medians[fanout] = {
            "shared_ready_ns": _median(
                item["all_branches_ready_ns"]["shared_root"]["median"] for item in items
            ),
            "shared_physical_bytes": _median(
                item["physical_kv_bytes"]["shared_root"]["median"] for item in items
            ),
            "shared_decode_tps": _median(
                item["decode_tokens_per_second"]["shared_root"]["median"] for item in items
            ),
            "state_fraction": _median(
                item["derived"]["state_management_fraction"]["median"] for item in items
            ),
            "cpu_fraction": _median(
                item["derived"]["cpu_block_management_fraction_of_readiness"]["median"]
                for item in items
            ),
        }
    result: list[dict[str, Any]] = []
    ordered = sorted(medians)
    for previous, current in pairwise(ordered):
        before = medians[previous]
        after = medians[current]
        branches = current - previous
        result.append(
            {
                "from_fanout": previous,
                "to_fanout": current,
                "readiness_ratio": (
                    None
                    if before["shared_ready_ns"] == 0
                    else after["shared_ready_ns"] / before["shared_ready_ns"]
                ),
                "physical_kv_ratio": (
                    None
                    if before["shared_physical_bytes"] == 0
                    else after["shared_physical_bytes"] / before["shared_physical_bytes"]
                ),
                "marginal_physical_kv_bytes_per_added_branch": (
                    (after["shared_physical_bytes"] - before["shared_physical_bytes"]) / branches
                ),
                "decode_throughput_ratio": (
                    None
                    if before["shared_decode_tps"] == 0
                    else after["shared_decode_tps"] / before["shared_decode_tps"]
                ),
                "state_management_fraction_delta": (
                    after["state_fraction"] - before["state_fraction"]
                ),
                "cpu_management_fraction_delta": after["cpu_fraction"] - before["cpu_fraction"],
            }
        )
    return result


def _svg_line_chart(
    *,
    title: str,
    x_labels: Sequence[str],
    y_label: str,
    series: Sequence[tuple[str, Sequence[float], str]],
) -> str:
    width, height = 920, 520
    left, right, top, bottom = 92, 32, 64, 82
    plot_w, plot_h = width - left - right, height - top - bottom
    values = [value for _, rows, _ in series for value in rows if math.isfinite(value)]
    maximum = max(values, default=1.0)
    if maximum <= 0:
        maximum = 1.0
    maximum *= 1.08

    def x_position(index: int) -> float:
        return left + (plot_w / 2 if len(x_labels) == 1 else index * plot_w / (len(x_labels) - 1))

    def y_position(value: float) -> float:
        return top + plot_h - value / maximum * plot_h

    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-family="sans-serif" '
        f'font-size="20">{escape(title)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" '
        f'y2="{top + plot_h}" stroke="#222"/>',
        f'<text transform="translate(22 {top + plot_h / 2}) rotate(-90)" '
        f'text-anchor="middle" font-family="sans-serif" font-size="13">{escape(y_label)}</text>',
    ]
    for tick in range(6):
        value = maximum * tick / 5
        y = y_position(value)
        chunks.extend(
            [
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" '
                'stroke="#e4e4e4"/>',
                f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end" '
                f'font-family="monospace" font-size="11">{value:.3g}</text>',
            ]
        )
    for index, label in enumerate(x_labels):
        x = x_position(index)
        chunks.append(
            f'<text x="{x:.2f}" y="{top + plot_h + 24}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="12">{escape(label)}</text>'
        )
    for series_index, (name, rows, color) in enumerate(series):
        points = " ".join(
            f"{x_position(index):.2f},{y_position(float(value)):.2f}"
            for index, value in enumerate(rows)
        )
        chunks.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>'
        )
        for index, value in enumerate(rows):
            chunks.append(
                f'<circle cx="{x_position(index):.2f}" cy="{y_position(float(value)):.2f}" '
                f'r="4" fill="{color}"/>'
            )
        legend_x = left + series_index * 240
        chunks.extend(
            [
                f'<line x1="{legend_x}" y1="{height - 24}" x2="{legend_x + 24}" '
                f'y2="{height - 24}" stroke="{color}" stroke-width="3"/>',
                f'<text x="{legend_x + 30}" y="{height - 20}" font-family="sans-serif" '
                f'font-size="12">{escape(name)}</text>',
            ]
        )
    chunks.append("</svg>\n")
    return "".join(chunks)


def _time_series_svg(trial: LoadedTrial) -> str:
    rows = [
        row
        for row in trial.sampler_rows
        if row.get("available") is True
        and isinstance(row.get("observed_at_monotonic_ns"), int)
        and isinstance(row.get("memory_used_bytes"), int)
    ]
    if not rows:
        raise AnalysisError(f"{trial.attempt_id} has no available nvidia-smi time-series samples")
    origin = min(int(row["observed_at_monotonic_ns"]) for row in rows)
    times = [(int(row["observed_at_monotonic_ns"]) - origin) / 1e9 for row in rows]
    memory = [int(row["memory_used_bytes"]) / 1024**3 for row in rows]
    utilization = [
        float(row["gpu_utilization_percent"])
        if isinstance(row.get("gpu_utilization_percent"), (int, float))
        else 0.0
        for row in rows
    ]
    duration = max(times[-1], 1e-9)
    memory_max = max(memory) * 1.05 or 1.0
    width, height = 920, 560
    left, right, top = 86, 72, 60
    panel_h, gap = 175, 54
    plot_w = width - left - right

    def x(value: float) -> float:
        return left + value / duration * plot_w

    def path(values: Sequence[float], maximum: float, panel_top: int) -> str:
        points = [
            f"{x(time):.2f},{panel_top + panel_h - value / maximum * panel_h:.2f}"
            for time, value in zip(times, values, strict=True)
        ]
        return " ".join(points)

    markers: list[tuple[float, str]] = []
    for phase, snapshot in trial.physical_rows:
        if phase in {"fork", "divergence"}:
            markers.append(((snapshot.observed_at_monotonic_ns - origin) / 1e9, phase))
    for row in trial.phase_rows:
        if row.get("phase") == "concurrent_decode" and isinstance(
            row.get("wall_start_monotonic_ns"), int
        ):
            markers.append(((int(row["wall_start_monotonic_ns"]) - origin) / 1e9, "decode"))
    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-family="sans-serif" '
        f'font-size="20">GPU state time series — {escape(trial.attempt_id)}</text>',
    ]
    for panel_index, (label, values, maximum, color) in enumerate(
        (
            ("GPU memory used (GiB)", memory, memory_max, "#2563eb"),
            ("GPU utilization (%)", utilization, 100.0, "#dc2626"),
        )
    ):
        panel_top = top + panel_index * (panel_h + gap)
        chunks.extend(
            [
                f'<rect x="{left}" y="{panel_top}" width="{plot_w}" height="{panel_h}" '
                'fill="none" stroke="#222"/>',
                f'<text transform="translate(20 {panel_top + panel_h / 2}) rotate(-90)" '
                f'text-anchor="middle" font-family="sans-serif" font-size="12">{label}</text>',
                f'<polyline points="{path(values, maximum, panel_top)}" fill="none" '
                f'stroke="{color}" stroke-width="2"/>',
            ]
        )
        for marker_time, marker in markers:
            if 0 <= marker_time <= duration:
                marker_x = x(marker_time)
                chunks.append(
                    f'<line x1="{marker_x:.2f}" y1="{panel_top}" x2="{marker_x:.2f}" '
                    f'y2="{panel_top + panel_h}" stroke="#555" stroke-dasharray="4 3"/>'
                )
                if panel_index == 0:
                    chunks.append(
                        f'<text x="{marker_x + 3:.2f}" y="{panel_top + 14}" '
                        f'font-family="sans-serif" font-size="10">{escape(marker)}</text>'
                    )
    chunks.extend(
        [
            f'<text x="{left + plot_w / 2}" y="{height - 18}" text-anchor="middle" '
            'font-family="sans-serif" font-size="12">seconds from first GPU sample</text>',
            "</svg>\n",
        ]
    )
    return "".join(chunks)


def _lifecycle_hbm_svg(trial: LoadedTrial) -> str:
    lifecycle = trial.lifecycle
    if lifecycle is None:
        raise AnalysisError(f"{trial.attempt_id} has no controller/worker lifecycle evidence")
    baseline_samples = cast(list[dict[str, Any]], lifecycle.hbm_baseline["samples"])
    live_samples = cast(list[dict[str, Any]], lifecycle.hbm_postflight["child_live_samples"])
    post_samples = cast(list[dict[str, Any]], lifecycle.hbm_postflight["post_child_exit_samples"])
    all_samples = [
        sample
        for sample in (*baseline_samples, *live_samples, *post_samples)
        if isinstance(sample.get("observed_at_monotonic_ns"), int)
        and _driver_memory_mib(sample) is not None
    ]
    if not all_samples:
        raise AnalysisError(f"{trial.attempt_id} has no lifecycle driver-visible HBM samples")
    all_samples.sort(key=lambda sample: int(sample["observed_at_monotonic_ns"]))
    origin_ns = int(all_samples[0]["observed_at_monotonic_ns"])
    end_ns = int(all_samples[-1]["observed_at_monotonic_ns"])
    duration_s = max((end_ns - origin_ns) / 1e9, 1e-9)
    times_s = [
        (int(sample["observed_at_monotonic_ns"]) - origin_ns) / 1e9 for sample in all_samples
    ]
    memory_mib = [float(cast(int, _driver_memory_mib(sample))) for sample in all_samples]
    threshold_mib = float(lifecycle.cleanup_gate["cleanup_threshold_mib"])
    maximum_mib = max((*memory_mib, threshold_mib), default=1.0) * 1.06
    width, height = 1120, 600
    left, right, top, bottom = 96, 34, 82, 102
    plot_w, plot_h = width - left - right, height - top - bottom

    def x(monotonic_ns: int) -> float:
        elapsed_s = (monotonic_ns - origin_ns) / 1e9
        return left + min(max(elapsed_s / duration_s, 0.0), 1.0) * plot_w

    def y(memory_value_mib: float) -> float:
        return top + plot_h - memory_value_mib / maximum_mib * plot_h

    events_by_name: dict[str, list[int]] = defaultdict(list)
    for event in lifecycle.child_events:
        events_by_name[str(event["event"])].append(int(event["observed_at_monotonic_ns"]))

    def first_event(*names: str) -> int | None:
        values = [timestamp for name in names for timestamp in events_by_name.get(name, [])]
        return min(values) if values else None

    baseline_ns = int(baseline_samples[len(baseline_samples) // 2]["observed_at_monotonic_ns"])
    child_exit_ns: int | None = None
    if post_samples:
        first_post = post_samples[0]
        observed = first_post.get("observed_at_monotonic_ns")
        elapsed_ms = first_post.get("elapsed_since_child_exit_ms")
        if isinstance(observed, int) and isinstance(elapsed_ms, (int, float)):
            child_exit_ns = observed - round(float(elapsed_ms) * 1e6)
    if child_exit_ns is None:
        child_exit_ns = first_event("child_manifest_flushed_exit_imminent")
    recovery_ns: int | None = None
    evaluation = lifecycle.cleanup_gate["stable_sample_evaluation"]
    recovery_index = evaluation.get("winning_sample_start_index")
    if isinstance(recovery_index, int) and 0 <= recovery_index < len(post_samples):
        observed = post_samples[recovery_index].get("observed_at_monotonic_ns")
        recovery_ns = observed if isinstance(observed, int) else None
    teardown_ns = first_event("branch_teardown_and_vllm_shutdown_completed")
    marker_candidates = (
        (baseline_ns, "parent baseline"),
        (first_event("child_process_started"), "child launch"),
        (first_event("model_load_started"), "model load"),
        (first_event("kv_pool_allocated"), "KV pool allocation"),
        (first_event("benchmark_started"), "benchmark start"),
        (first_event("branch_fork_start"), "branch fork"),
        (first_event("decode_start", "independent_prefill_and_decode_start"), "decode"),
        (teardown_ns, "branch teardown"),
        (teardown_ns, "vLLM shutdown"),
        (child_exit_ns, "child exit"),
        (recovery_ns, "HBM recovery"),
    )
    markers = [
        (timestamp, label) for timestamp, label in marker_candidates if timestamp is not None
    ]
    required_labels = {label for _, label in marker_candidates}
    observed_labels = {label for _, label in markers}
    if trial.baseline_mode == "shared_root" and required_labels != observed_labels:
        raise AnalysisError(
            f"{trial.attempt_id} lacks lifecycle plot events: "
            + ", ".join(sorted(required_labels - observed_labels))
        )

    memory_points = " ".join(
        f"{left + time_s / duration_s * plot_w:.2f},{y(value):.2f}"
        for time_s, value in zip(times_s, memory_mib, strict=True)
    )
    recovery_label = (
        "unavailable"
        if lifecycle.time_to_hbm_recovery_ms is None
        else f"{lifecycle.time_to_hbm_recovery_ms:.1f} ms after child exit"
    )
    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="31" text-anchor="middle" font-family="sans-serif" '
        f'font-size="20">Driver-visible HBM lifecycle — {escape(trial.attempt_id)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" '
        f'y2="{top + plot_h}" stroke="#222"/>',
        f'<text transform="translate(25 {top + plot_h / 2}) rotate(-90)" '
        'text-anchor="middle" font-family="sans-serif" font-size="13">'
        "driver-visible GPU memory (MiB)</text>",
        f'<polyline points="{memory_points}" fill="none" stroke="#2563eb" stroke-width="2.5"/>',
        f'<line x1="{left}" y1="{y(threshold_mib):.2f}" x2="{left + plot_w}" '
        f'y2="{y(threshold_mib):.2f}" stroke="#16a34a" stroke-dasharray="7 4"/>',
        f'<text x="{left + 6}" y="{y(threshold_mib) - 6:.2f}" font-family="sans-serif" '
        f'font-size="11" fill="#166534">cleanup threshold {threshold_mib:.0f} MiB</text>',
    ]
    for index, (timestamp, label) in enumerate(markers):
        marker_x = x(timestamp)
        label_y = top + 14 + (index % 4) * 15
        chunks.extend(
            [
                f'<line x1="{marker_x:.2f}" y1="{top}" x2="{marker_x:.2f}" '
                f'y2="{top + plot_h}" stroke="#555" stroke-dasharray="3 3"/>',
                f'<text x="{marker_x + 3:.2f}" y="{label_y}" font-family="sans-serif" '
                f'font-size="10">{escape(label)}</text>',
            ]
        )
    chunks.extend(
        [
            f'<text x="{left + plot_w / 2}" y="{height - 34}" text-anchor="middle" '
            'font-family="sans-serif" font-size="12">seconds from first parent baseline sample</text>',
            f'<text x="{left}" y="{height - 14}" font-family="sans-serif" font-size="11">'
            f"baseline {float(lifecycle.hbm_baseline['baseline_median_mib']):.0f} MiB; "
            f"recovery {recovery_label}</text>",
            "</svg>\n",
        ]
    )
    return "".join(chunks)


def _private_progression_svg(trial: LoadedTrial) -> str:
    phases = [
        phase
        for phase in ("divergence", "suffix_16", "suffix_64", "suffix_256")
        if any(row_phase == phase for row_phase, _ in trial.physical_rows)
    ]
    minimums: list[float] = []
    medians: list[float] = []
    maximums: list[float] = []
    branch_ids = set(trial.summary.branch_ready_latency_ns)
    for phase in phases:
        snapshot = _snapshot(trial, phase)
        private = set(snapshot.private_suffix_block_ids)
        values = [
            sum(
                block.bytes
                for block in snapshot.blocks
                if block.runtime_block_id in private and branch_id in block.branch_ids
            )
            / 1024**2
            for branch_id in branch_ids
        ]
        minimums.append(min(values))
        medians.append(float(statistics.median(values)))
        maximums.append(max(values))
    return _svg_line_chart(
        title=f"Branch progression vs private KV allocation — fanout {trial.fanout}",
        x_labels=phases,
        y_label="private KV per branch (MiB)",
        series=(
            ("minimum", minimums, "#16a34a"),
            ("median", medians, "#2563eb"),
            ("maximum", maximums, "#dc2626"),
        ),
    )


def _plots(
    plots_root: Path, cells: Sequence[dict[str, Any]], trials: Sequence[LoadedTrial]
) -> list[dict[str, Any]]:
    lifecycle_trials = [
        trial
        for trial in trials
        if trial.lifecycle is not None and trial.baseline_mode == "shared_root"
    ]
    lifecycle_representative = (
        max(
            lifecycle_trials,
            key=lambda item: (
                item.lifecycle_clean is True,
                item.prefix_length,
                item.fanout,
                item.attempt_id,
            ),
        )
        if lifecycle_trials
        else None
    )
    if not cells:
        plots_root.mkdir(parents=True, exist_ok=True)
        payloads = (
            {}
            if lifecycle_representative is None
            else {"gpu-runtime-lifecycle.svg": _lifecycle_hbm_svg(lifecycle_representative)}
        )
        records: list[dict[str, Any]] = []
        for name, payload in sorted(payloads.items()):
            path = plots_root / name
            _replace_bytes(path, payload.encode())
            records.append({"path": name, "bytes": path.stat().st_size, "sha256": _sha256(path)})
        _replace_bytes(
            plots_root / "README.md",
            b"# Plots unavailable\n\n"
            b"No paired 16K independent/shared result is materialized. Comparative plots "
            b"would fabricate measurements; an authoritative lifecycle plot is retained when "
            b"controller/worker evidence is available.\n",
        )
        _replace_bytes(
            plots_root / "plot-manifest.json",
            _canonical_bytes(
                {
                    "schema_version": "sloforge.branchfabric.gpu-validation-plots/v1",
                    "plots": records,
                    "representative_lifecycle_attempt": (
                        None
                        if lifecycle_representative is None
                        else lifecycle_representative.attempt_id
                    ),
                    "unavailable_reason": (
                        "no paired 16K independent/shared result is materialized"
                    ),
                }
            ),
        )
        return records
    labels = [str(cell["fanout"]) for cell in cells]
    readiness = _svg_line_chart(
        title=(
            "Fanout vs readiness with explicit measurement origins "
            "(observational; cross-GPU/variant causality may be unavailable)"
        ),
        x_labels=labels,
        y_label="latency (ms)",
        series=(
            (
                "independent (root-inclusive prefill)",
                [cell["all_branches_ready_ns"]["independent"]["median"] / 1e6 for cell in cells],
                "#dc2626",
            ),
            (
                "shared root (root-inclusive end-to-end)",
                [
                    cell["root_inclusive_all_branches_ready_ns"]["shared_root"]["median"] / 1e6
                    for cell in cells
                ],
                "#7c3aed",
            ),
            (
                "shared root (post-root readiness)",
                [cell["all_branches_ready_ns"]["shared_root"]["median"] / 1e6 for cell in cells],
                "#2563eb",
            ),
        ),
    )
    physical = _svg_line_chart(
        title="Fanout vs physical KV state",
        x_labels=labels,
        y_label="physical KV (GiB)",
        series=(
            (
                "independent observed peak",
                [
                    cell["physical_kv_bytes"]["independent_observed_peak"]["median"] / 1024**3
                    for cell in cells
                ],
                "#dc2626",
            ),
            (
                "independent full-fanout terminal estimate",
                [
                    cell["physical_kv_bytes"][
                        "independent_comparable_full_fanout_terminal_estimate"
                    ]["median"]
                    / 1024**3
                    for cell in cells
                ],
                "#f97316",
            ),
            (
                "shared aligned all-live actual",
                [
                    cell["physical_kv_bytes"]["aligned_all_branches_live"]["shared_actual"][
                        "median"
                    ]
                    / 1024**3
                    for cell in cells
                ],
                "#2563eb",
            ),
            (
                "theoretical sharing floor",
                [
                    cell["physical_kv_bytes"]["aligned_all_branches_live"][
                        "theoretical_shared_floor"
                    ]["median"]
                    / 1024**3
                    for cell in cells
                ],
                "#16a34a",
            ),
        ),
    )
    throughput = _svg_line_chart(
        title="Observed active decode throughput (cross-path causality may be unavailable)",
        x_labels=labels,
        y_label="tokens / second",
        series=(
            (
                "independent",
                [cell["decode_tokens_per_second"]["independent"]["median"] for cell in cells],
                "#dc2626",
            ),
            (
                "shared root",
                [cell["decode_tokens_per_second"]["shared_root"]["median"] for cell in cells],
                "#2563eb",
            ),
        ),
    )
    shared_trials = [
        trial
        for trial in trials
        if trial.baseline_mode == "shared_root"
        and trial.tracing_level == _PRIMARY_TRACE_LEVEL
        and trial.prefix_length == EXPECTED_PREFIX_TOKENS
    ]
    representative = max(shared_trials, key=lambda item: (item.fanout, -item.seed, item.attempt_id))
    payloads = {
        "fanout-readiness.svg": readiness,
        "fanout-physical-hbm.svg": physical,
        "gpu-state-timeseries.svg": _time_series_svg(representative),
        "fanout-decode-throughput.svg": throughput,
        "branch-private-kv-progression.svg": _private_progression_svg(representative),
    }
    if lifecycle_representative is None:
        raise AnalysisError("paired 16K results lack an authoritative lifecycle time series")
    payloads["gpu-runtime-lifecycle.svg"] = _lifecycle_hbm_svg(lifecycle_representative)
    plots_root.mkdir(parents=True, exist_ok=True)
    (plots_root / "README.md").unlink(missing_ok=True)
    records: list[dict[str, Any]] = []
    for name, payload in sorted(payloads.items()):
        path = plots_root / name
        _replace_bytes(path, payload.encode())
        records.append({"path": name, "bytes": path.stat().st_size, "sha256": _sha256(path)})
    _replace_bytes(
        plots_root / "plot-manifest.json",
        _canonical_bytes(
            {
                "schema_version": "sloforge.branchfabric.gpu-validation-plots/v1",
                "plots": records,
                "representative_time_series_attempt": representative.attempt_id,
                "representative_lifecycle_attempt": lifecycle_representative.attempt_id,
            }
        ),
    )
    return records


def _next_experiment_classification(
    cells: Sequence[dict[str, Any]],
    *,
    clean_smoke: bool,
) -> dict[str, Any] | None:
    if not clean_smoke:
        return None
    by_fanout = {int(cell["fanout"]): cell for cell in cells}
    if 8 not in by_fanout:
        return None
    cell = by_fanout[8]
    derived = cast(dict[str, Any], cell["derived"])
    state_fraction = float(derived["state_management_fraction"]["median"])
    cpu_fraction = float(derived["cpu_block_management_fraction_of_readiness"]["median"])
    amplification = float(derived["physical_amplification"])
    floor_gap = float(derived["cow_floor_gap_bytes"])
    interference = derived.get("decode_interference_fraction")
    if amplification > 1.05:
        choice = "B"
        name = "GPU ALLOCATION + COW DIVERGENCE CHARACTERIZATION"
        rationale = "shared physical KV is measurably above the exact sharing floor"
    elif cpu_fraction >= 0.10 or (state_fraction >= 0.15 and amplification <= 1.05):
        choice = "E"
        name = "HIGH-FANOUT RUNTIME METADATA CHARACTERIZATION"
        rationale = (
            "fanout-8 exposes a material unresolved runtime-metadata signal while physical KV "
            "remains at the exact sharing floor; paired trace-disabled controls are unavailable"
        )
    elif state_fraction >= 0.05:
        choice = "C"
        name = "REAL TRANSFORM + TRANSFER PIPELINE CHARACTERIZATION"
        rationale = "state-operation time is material without physical amplification"
    elif (
        interference is not None
        and float(interference) < 0.05
        and state_fraction < 0.05
        and cpu_fraction < 0.05
        and amplification <= 1.05
    ):
        choice = "A"
        name = "REAL CAPACITY RECLAMATION + CROSS-LAYOUT MIGRATION"
        rationale = "measured COW management, interference, and physical amplification are low"
    else:
        choice = "E"
        name = "HIGH-FANOUT RUNTIME METADATA CHARACTERIZATION"
        rationale = "the unresolved fanout-8 signal is runtime metadata rather than HBM allocation"
    return {
        "choice": choice,
        "name": name,
        "rationale": rationale,
        "evidence": {
            "fanout": 8,
            "state_management_fraction": state_fraction,
            "cpu_block_management_fraction": cpu_fraction,
            "state_management_fraction_scope": (
                "fork_metadata wall time divided by post-root readiness wall time"
            ),
            "cpu_block_management_fraction_scope": (
                "fork_metadata process CPU time divided by post-root readiness wall time; "
                "not critical-path attribution"
            ),
            "tracing_overhead_availability": "unavailable_no_paired_trace_disabled_control",
            "physical_amplification": amplification,
            "cow_floor_gap_bytes": floor_gap,
            "decode_interference_fraction": interference,
            "decode_interference_availability": (
                "available" if interference is not None else "unavailable_cross_gpu_pair"
            ),
        },
    }


def _recommended_next(
    cells: Sequence[dict[str, Any]],
    *,
    clean_smoke: bool,
    classification: Mapping[str, Any] | None,
) -> str:
    fanouts = {int(cell["fanout"]) for cell in cells}
    if not clean_smoke:
        return (
            "Resolve the authoritative controller/worker cleanup failure, rerun only the 2K "
            "semantic smoke, and require normal child/descendant exit plus three stable "
            "driver-visible HBM samples within 1 GiB of baseline before any 16K allocation."
        )
    if not fanouts:
        return "Run paired 16K fanout-1 independent/shared trials after the valid smoke."
    if 8 not in fanouts:
        return "Run one paired 16K fanout-8 trial and inspect HBM, invariants, and budget."
    if classification is None:
        raise AnalysisError("fanout-8 evidence lacks one next-experiment classification")
    return f"{classification['choice']}. {classification['name']}: {classification['rationale']}."


def _report_markdown(report: Mapping[str, Any]) -> str:
    runtime = cast(Mapping[str, Any], report["selected_runtime"])
    model = cast(Mapping[str, Any], report["selected_model"])
    experiments = cast(Sequence[Mapping[str, Any]], report["experiments_actually_run"])
    cells = cast(Sequence[Mapping[str, Any]], report["measurements"])
    interest = cast(Mapping[str, Any], report["branchfabric_interest"])
    classification = cast(Mapping[str, Any], report.get("next_experiment_classification") or {})
    semantic_smoke = cast(Mapping[str, Any], report["semantic_smoke"])
    smoke_observations = cast(Sequence[Mapping[str, Any]], semantic_smoke["observations"])
    lifecycle_experiments = [
        trial for trial in experiments if trial.get("lifecycle_mode") == "controller_worker"
    ]
    clean_lifecycle_experiments = [
        trial for trial in lifecycle_experiments if trial.get("gpu_runtime_lifecycle_clean") is True
    ]
    context_experiments = [
        trial for trial in experiments if trial.get("prefix_tokens") == EXPECTED_PREFIX_TOKENS
    ]
    lines = [
        "# BranchFabric GPU Validation Experiment 002",
        "",
        "## Outcome",
        "",
        f"Status: `{report['status']}`. GPU-hours consumed: {float(report['gpu_hours']):.6f}.",
        f"Ledger-estimated cloud cost (upper bound): "
        f"`${float(report['estimated_cloud_cost_usd']):.6f}`.",
        f"Real physical shared-root semantics demonstrated: "
        f"`{str(report['real_physical_shared_root_demonstrated']).lower()}`.",
        f"Complete semantic smoke (including postflight HBM recovery): "
        f"`{str(report['semantic_smoke']['passed']).lower()}`.",
        "This report does not claim FPGA justification.",
        "",
        "## Runtime adapter and exposed state",
        "",
        f"Runtime: `{runtime['name']}=={runtime['version']}`. Model: `{model['id']}` at "
        f"revision `{model['revision']}` with dtype `{model['dtype']}`.",
        "The adapter exposes CUDA-backed KV tensors, ordered physical block IDs, native "
        "refcounts, logical ranges, shared/private ownership, allocation epochs, exact-ID "
        "release evidence, KV assigned/reserved bytes, NVML/PyTorch memory, and hardware-backed "
        "state traces. The semantics are immutable shared-root plus append-only private suffixes, "
        "not general CUDA write-fault COW.",
        "",
        "## Experiments actually run",
        "",
        "| Attempt | Path | Fanout | Prefix | Suffix | Seed | GPU UUID | Trace | Lifecycle |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for trial in experiments:
        lines.append(
            f"| {trial['attempt_id']} | {trial['path']} | {trial['fanout']} | "
            f"{trial['prefix_tokens']} | {trial['suffix_tokens']} | {trial['seed']} | "
            f"{trial['gpu_uuid']} | {trial['tracing_level']} | "
            f"{trial.get('lifecycle_mode', 'legacy')} / "
            f"{trial.get('gpu_runtime_lifecycle_clean')} |"
        )
    if not experiments:
        lines.append("| — | — | — | — | — | — | — | — | — |")
    lines.extend(
        [
            "",
            f"The materialized evidence contains {len(experiments)} completed benchmark "
            f"workload(s), {len(smoke_observations)} semantic smoke observation(s), and "
            f"{len(context_experiments)} accepted 16K trial(s). "
            f"{len(lifecycle_experiments)} workload(s) use the controller/worker boundary; "
            f"{len(clean_lifecycle_experiments)} passed its authoritative driver-visible "
            "lifecycle gate. Earlier attempts remain listed as historical evidence.",
            "",
            "## Real GPU semantic smoke observation",
            "",
            "| Shared blocks / bytes | Private blocks / bytes | Root refs | Fork-added KV | "
            "All ready | Decode | Post-decode assigned KV | Post-root-delete assigned KV |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for observation in smoke_observations:
        lines.append(
            f"| {observation['shared_prefix_blocks']} / "
            f"{float(observation['shared_prefix_bytes']) / 1024**2:.2f} MiB | "
            f"{observation['private_suffix_blocks']} / "
            f"{float(observation['private_suffix_bytes']) / 1024**2:.2f} MiB | "
            f"{observation['root_native_refcount_at_divergence']} | "
            f"{float(observation['fork_allocated_additional_physical_bytes']) / 1024**2:.2f} "
            f"MiB | {float(observation['all_branches_ready_ns']) / 1e6:.3f} ms | "
            f"{float(observation['decode_throughput_tokens_per_second']):.3f} tok/s | "
            f"{float(observation['physical_assigned_bytes']) / 1024**2:.2f} MiB | "
            f"{float(observation['post_root_delete_physical_assigned_bytes']) / 1024**2:.2f} "
            "MiB |"
        )
    lines.extend(
        [
            "",
            "The smoke assertions cover exact shared-root physical IDs, zero fork-time prefix "
            "allocation, private divergence, sibling survival after one branch is deleted, and "
            "complete runtime-assigned KV release after the final root is deleted.",
        ]
    )
    lines.extend(
        [
            "",
            "## Paired 16K results",
            "",
            "Physical HBM below means assigned physical KV state, not the pre-reserved vLLM KV "
            "pool or total driver-visible model memory.",
            "",
            "| Fanout | Independent root-inclusive ready ms | Shared root-inclusive ready ms | "
            "Shared post-root ready ms | Aligned all-live phase | Independent aligned actual GiB | "
            "Shared aligned actual GiB | Aligned savings GiB | Aligned efficiency | Independent "
            "observed peak GiB | Independent full-256 projection GiB | Shared full-256 actual GiB | "
            "Projected terminal savings GiB | Projected terminal efficiency | Amplification | "
            "Floor gap MiB | Decode interference | Causal pair |",
            "| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for cell in cells:
        derived = cast(Mapping[str, Any], cell["derived"])
        sharing = derived.get("sharing_efficiency")
        interference = derived.get("decode_interference_fraction")
        lines.append(
            f"| {cell['fanout']} | "
            f"{float(cell['all_branches_ready_ns']['independent']['median']) / 1e6:.3f} | "
            f"{float(cell['root_inclusive_all_branches_ready_ns']['shared_root']['median']) / 1e6:.3f} | "
            f"{float(cell['all_branches_ready_ns']['shared_root']['median']) / 1e6:.3f} | "
            f"{', '.join(cell['physical_kv_bytes']['aligned_all_branches_live']['checkpoint_phases'])} | "
            f"{float(cell['physical_kv_bytes']['aligned_all_branches_live']['independent_actual']['median']) / 1024**3:.4f} | "
            f"{float(cell['physical_kv_bytes']['aligned_all_branches_live']['shared_actual']['median']) / 1024**3:.4f} | "
            f"{float(derived['sharing_savings_bytes']) / 1024**3:.4f} | "
            f"{'unavailable' if sharing is None else f'{float(sharing):.4f}'} | "
            f"{float(cell['physical_kv_bytes']['independent_observed_peak']['median']) / 1024**3:.4f} | "
            f"{float(cell['physical_kv_bytes']['independent_comparable_full_fanout_terminal_estimate']['median']) / 1024**3:.4f} | "
            f"{float(cell['physical_kv_bytes']['shared_full_fanout_terminal_actual']['median']) / 1024**3:.4f} | "
            f"{float(derived['projected_terminal_sharing_savings_bytes']) / 1024**3:.4f} | "
            f"{float(derived['projected_terminal_sharing_efficiency']):.4f} | "
            f"{float(derived['physical_amplification']):.4f} | "
            f"{float(derived['cow_floor_gap_bytes']) / 1024**2:.2f} | "
            f"{'unavailable' if interference is None else f'{float(interference):.4f}'} | "
            f"{cell['causal_pair_metrics_available']} |"
        )
    if not cells:
        lines.append("| — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |")
    lines.extend(
        [
            "",
            "The shared root-inclusive readiness is a continuous wall-time measure from "
            "`root_session_create` start through `concurrent_decode` start plus the measured "
            "runtime readiness remaining after fork. It therefore includes the validation and "
            "instrumentation interval between fork completion and decode admission. The post-root "
            "value retains the fork/admission-through-decode-readiness view. The "
            "independent observed peak is the maximum direct physical snapshot; its comparable "
            "all-branches-live savings/efficiency uses the latest phase where all owners remain "
            "present in both paths. The full-fanout terminal value is explicitly an estimate "
            "from the exact surviving "
            "terminal branch pages times fanout, backed by cache-free, disjoint, full-prefill "
            "assertions. It is not mislabeled as a simultaneous direct observation.",
        ]
    )
    lines.extend(
        [
            "",
            "## BranchFabric follow-up signals",
            "",
        ]
    )
    signals = cast(Sequence[Mapping[str, Any]], interest["signals"])
    if signals:
        for signal in signals:
            lines.append(
                f"- `{signal['gate']}` at fanout {signal['fanout']}: "
                f"{float(signal['value']):.4f} ({signal['classification']})."
            )
    else:
        lines.append("No paired 16K interest gate is available yet.")
    for unavailable in cast(Sequence[Mapping[str, Any]], interest["unavailable_gates"]):
        lines.append(f"- `{unavailable['gate']}` unavailable: {unavailable['reason']}.")
    lines.extend(
        [
            "",
            "All accepted 16K trials used full tracing and no matched trace-disabled control was "
            "run. The state-management fraction is fork wall time divided by post-root readiness; "
            "the CPU fraction is process CPU time divided by wall readiness and is not a "
            "critical-path attribution. These are material diagnostic signals, not proof that "
            "metadata dominates execution.",
        ]
    )
    lines.extend(
        [
            "",
            "## Recommended next GPU experiment",
            "",
            f"Classification: `{classification.get('choice', 'pending')}` — "
            f"{classification.get('name', 'pending')}",
            "",
            str(report["recommended_next_gpu_experiment"]),
            "",
            "## Artifacts and cleanup",
            "",
            "Every input was revalidated against its remote manifest and local-copy hash "
            "inventory. Controller/worker attempts additionally require a CUDA-clean parent, "
            "normal child and descendant termination, no forced GPU-process kill, zero final "
            "runtime-assigned KV bytes, and three consecutive process-free driver samples at "
            "or below baseline plus 1,024 MiB. Legacy in-process evidence remains readable but "
            "cannot authorize a 16K trial. Modal provider cleanup is reported separately from "
            "the stronger in-container process-lifecycle gate. Current provider cleanup status: "
            f"`{report['cleanup']['provider_container_teardown']}`; verified: "
            f"`{str(report['cleanup']['provider_cleanup_verified']).lower()}`. The earlier v5 "
            "provider audit remains historical evidence and is not promoted to current-campaign "
            "proof.",
            "",
        ]
    )
    return "\n".join(lines)


def _execution_attempts(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    modal_root = root / "modal"
    for attempt_root in sorted(modal_root.iterdir()):
        completion_path = attempt_root / "function-completion.json"
        if not attempt_root.is_dir() or not completion_path.is_file():
            continue
        completion = _read_json(completion_path)
        records.append(
            {
                "attempt_id": attempt_root.name,
                "status": completion.get("status"),
                "function_call_id": completion.get("function_call_id"),
                "actual_gpu": completion.get("actual_gpu"),
                "model_ready_utc": completion.get("model_ready_utc"),
                "benchmark_start_utc": completion.get("benchmark_start_utc"),
                "benchmark_end_utc": completion.get("benchmark_end_utc"),
                "function_end_utc": completion.get("function_end_utc"),
                "benchmark_seconds": completion.get("benchmark_seconds"),
                "benchmark_workload_completed": isinstance(completion.get("summary"), dict),
                "error": completion.get("error"),
            }
        )
    return records


def _parse_utc(value: object, *, field: str) -> datetime:
    if not isinstance(value, str):
        raise AnalysisError(f"provider audit {field} is not an ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise AnalysisError(f"provider audit {field} is not an ISO timestamp") from error
    if parsed.utcoffset() is None:
        raise AnalysisError(f"provider audit {field} lacks a UTC offset")
    return parsed


def _provider_cleanup_evidence(
    root: Path, execution_attempts: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    audit_path = root / _FINAL_PROVIDER_AUDIT
    legacy_path = root / "environment/final-modal-provider-audit.json"
    legacy = (
        {
            "path": str(legacy_path.relative_to(root)),
            "sha256": _sha256(legacy_path),
            "status": "historical_v5_scope_only_not_current_campaign_proof",
            "recorded_at_utc": _read_json(legacy_path).get("recorded_at_utc"),
        }
        if legacy_path.is_file()
        else None
    )
    if not audit_path.is_file():
        return {
            "status": "pending_unverified",
            "verified": False,
            "required_artifact": str(_FINAL_PROVIDER_AUDIT),
            "reason": "current-campaign final Modal app audit is not materialized",
            "legacy_provider_audit": legacy,
        }
    audit = _read_json(audit_path)
    expected_attempt_ids = sorted(str(item["attempt_id"]) for item in execution_attempts)
    expected_call_ids = sorted(
        str(item["function_call_id"])
        for item in execution_attempts
        if isinstance(item.get("function_call_id"), str)
    )
    apps = audit.get("experiment_apps")
    source_attempt_ids = audit.get("source_attempt_ids")
    source_call_ids = audit.get("source_function_call_ids")
    if (
        audit.get("schema_version") != "sloforge.branchfabric.modal-provider-final-audit/v2"
        or audit.get("experiment") != "experiment-002"
        or audit.get("modal_app_list_command") != ["modal", "app", "list", "--json"]
        or audit.get("modal_container_list_command") != ["modal", "container", "list", "--json"]
        or not isinstance(source_attempt_ids, list)
        or len(source_attempt_ids) != len(set(source_attempt_ids))
        or sorted(source_attempt_ids) != expected_attempt_ids
        or not isinstance(source_call_ids, list)
        or len(source_call_ids) != len(set(source_call_ids))
        or sorted(source_call_ids) != expected_call_ids
        or audit.get("experiment_app_description") != "sloforge-branchfabric-real-gpu-cow"
        or not isinstance(apps, list)
        or not apps
        or audit.get("experiment_apps_not_stopped_or_with_tasks") != []
        or audit.get("all_experiment_apps_stopped") is not True
        or audit.get("running_containers") != []
        or audit.get("provider_hbm_after_container_teardown_sampled") is not False
        or audit.get("persistent_resources_retained")
        != ["sloforge-model-cache", "sloforge-branchfabric-results"]
    ):
        raise AnalysisError("current-campaign final Modal provider audit is invalid")
    for app in apps:
        if (
            not isinstance(app, dict)
            or not isinstance(app.get("app_id"), str)
            or not app["app_id"]
            or app.get("description") != "sloforge-branchfabric-real-gpu-cow"
            or str(app.get("state")).lower() != "stopped"
            or app.get("task_count") != 0
        ):
            raise AnalysisError("final Modal provider audit contains a live/invalid experiment app")
    recorded = _parse_utc(audit.get("recorded_at_utc"), field="recorded_at_utc")
    function_end_times = [
        _parse_utc(item["function_end_utc"], field="function_end_utc")
        for item in execution_attempts
        if item.get("function_end_utc") is not None
    ]
    if not function_end_times or recorded < max(function_end_times):
        raise AnalysisError("final Modal provider audit predates the current experiment campaign")
    return {
        "status": "verified_all_experiment_apps_stopped_zero_tasks_and_no_running_containers",
        "verified": True,
        "audit_path": str(_FINAL_PROVIDER_AUDIT),
        "audit_sha256": _sha256(audit_path),
        "recorded_at_utc": audit["recorded_at_utc"],
        "source_attempt_ids": expected_attempt_ids,
        "source_function_call_ids": expected_call_ids,
        "experiment_apps": apps,
        "provider_hbm_after_teardown_sampled": False,
        "legacy_provider_audit": legacy,
    }


def _smoke_observations(smoke: Sequence[LoadedTrial]) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for trial in smoke:
        root = _snapshot(trial, "root")
        fork = _snapshot(trial, "fork")
        divergence = _snapshot(trial, "divergence")
        post_branch = _snapshot(trial, "post_branch_delete")
        post_root = _snapshot(trial, "post_root_delete")
        memory = {
            str(row["phase"]): cast(dict[str, Any], row["state"])
            for row in trial.gpu_memory_rows
            if isinstance(row.get("phase"), str) and isinstance(row.get("state"), dict)
        }
        page_sizes = {block.bytes for block in divergence.blocks}
        observations.append(
            {
                "attempt_id": trial.attempt_id,
                "function_status": trial.completion["status"],
                "postflight_clean": trial.postflight_clean,
                "lifecycle_mode": (
                    "controller_worker" if trial.lifecycle is not None else "legacy_in_process"
                ),
                "gpu_runtime_lifecycle_clean": trial.lifecycle_clean,
                "time_to_hbm_recovery_ms": (
                    None if trial.lifecycle is None else trial.lifecycle.time_to_hbm_recovery_ms
                ),
                "fanout": trial.fanout,
                "prefix_tokens": trial.prefix_length,
                "suffix_tokens": trial.suffix_length,
                "gpu_name": trial.gpu_name,
                "gpu_uuid": trial.gpu_uuid,
                "physical_page_bytes": next(iter(page_sizes)) if len(page_sizes) == 1 else None,
                "shared_prefix_blocks": len(divergence.shared_prefix_block_ids),
                "shared_prefix_bytes": divergence.shared_prefix_bytes,
                "root_native_refcount_at_divergence": divergence.root_reference_count,
                "private_suffix_blocks": len(divergence.private_suffix_block_ids),
                "private_suffix_bytes": divergence.private_suffix_bytes,
                "physical_assigned_bytes": divergence.physical_assigned_bytes,
                "fork_allocated_additional_physical_bytes": (
                    fork.physical_assigned_bytes - root.physical_assigned_bytes
                ),
                "post_branch_delete_root_refcount": post_branch.root_reference_count,
                "post_branch_delete_private_suffix_bytes": post_branch.private_suffix_bytes,
                "post_root_delete_physical_assigned_bytes": post_root.physical_assigned_bytes,
                "all_branches_ready_ns": trial.summary.all_branches_ready_ns,
                "decode_throughput_tokens_per_second": (
                    trial.summary.decode_throughput_tokens_per_second
                ),
                "decode_tokens": trial.summary.decode_tokens,
                "kv_memory_states": memory,
            }
        )
    return observations


def analyze(*, experiment_root: Path, report_json: Path, report_md: Path) -> dict[str, Any]:
    """Analyze verified local artifacts and atomically replace derived outputs."""

    root = experiment_root.resolve(strict=True)
    trials = _load_trials(root)
    ledger = _ledger(root, trials)
    smoke = _validate_smoke_and_pairs(trials)
    clean_smoke = any(trial.lifecycle_clean is True for trial in smoke)
    smoke_observations = _smoke_observations(smoke)
    execution_attempts = _execution_attempts(root)
    provider_cleanup = _provider_cleanup_evidence(root, execution_attempts)
    cells = _aggregate_cells(trials)
    trace_overhead = _tracing_overhead(trials)
    interest = _interest_assessment(cells)
    next_classification = _next_experiment_classification(cells, clean_smoke=clean_smoke)
    scaling = _scaling(cells)
    primary_cells = sorted(cells, key=lambda cell: (int(cell["fanout"]), int(cell["seed"])))
    plot_records = _plots(root / "plots", primary_cells, trials)
    fanouts = {int(cell["fanout"]) for cell in cells}
    status = (
        "completed_fanout_32"
        if 32 in fanouts
        else "completed_fanout_8"
        if 8 in fanouts
        else "completed_fanout_1"
        if 1 in fanouts
        else "semantic_smoke_passed"
        if clean_smoke
        else "semantic_smoke_observed_cleanup_failed"
    )
    representative = trials[0].summary.identity
    lifecycle_trials = [trial for trial in trials if trial.lifecycle is not None]
    clean_lifecycle_trials = [trial for trial in lifecycle_trials if trial.lifecycle_clean is True]
    experiments = [
        {
            "attempt_id": trial.attempt_id,
            "path": trial.summary.path.value,
            "fanout": trial.fanout,
            "prefix_tokens": trial.prefix_length,
            "suffix_tokens": trial.suffix_length,
            "seed": trial.seed,
            "tracing_level": trial.tracing_level,
            "gpu_name": trial.gpu_name,
            "gpu_uuid": trial.gpu_uuid,
            "remote_manifest_sha256": trial.remote_manifest_sha256,
            "function_status": trial.completion["status"],
            "postflight_clean": trial.postflight_clean,
            "lifecycle_mode": (
                "controller_worker" if trial.lifecycle is not None else "legacy_in_process"
            ),
            "gpu_runtime_lifecycle_clean": trial.lifecycle_clean,
            "time_to_hbm_recovery_ms": (
                None
                if trial.lifecycle is None or trial.lifecycle_clean is not True
                else trial.lifecycle.time_to_hbm_recovery_ms
            ),
        }
        for trial in sorted(trials, key=lambda item: item.attempt_id)
    ]
    analysis = {
        "schema_version": ANALYZER_SCHEMA_VERSION,
        "status": status,
        "source_attempts": experiments,
        "semantic_smoke_attempt_ids": [item.attempt_id for item in smoke],
        "semantic_smoke_passed": clean_smoke,
        "semantic_smoke_observations": smoke_observations,
        "authoritative_lifecycle_attempt_ids": [
            trial.attempt_id for trial in trials if trial.lifecycle is not None
        ],
        "real_physical_shared_root_demonstrated": bool(smoke),
        "execution_attempts": execution_attempts,
        "paired_cells": primary_cells,
        "scaling": scaling,
        "tracing_overhead": trace_overhead,
        "instrumentation_availability": {
            "nvml_time_series": "available",
            "physical_kv_layout": "available",
            "copy_engine_activity": "unavailable_without_profiler_counters",
            "attributable_hbm_bandwidth": "unavailable_without_CUPTI_or_DCGM",
        },
        "branchfabric_interest": interest,
        "next_experiment_classification": next_classification,
        "provider_cleanup": provider_cleanup,
        "gpu_hours": float(ledger["consumed_gpu_hours"]),
        "estimated_cloud_cost_usd": float(ledger["consumed_cloud_cost_usd"]),
        "plots": plot_records,
    }
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "experiment": "experiment-002",
        "status": status,
        "execution_backend": "modal",
        "selected_runtime": {
            "name": representative.runtime,
            "version": representative.runtime_version,
            "adapter_version": representative.adapter_version,
            "runtime_internals_required": True,
            "upstream_patch_required": False,
        },
        "selected_model": {
            "id": representative.model_id,
            "revision": representative.model_revision,
            "tokenizer_revision": representative.tokenizer_revision,
            "dtype": representative.dtype,
        },
        "experiments_actually_run": experiments,
        "modal_function_attempts": execution_attempts,
        "gpu_hours": float(ledger["consumed_gpu_hours"]),
        "estimated_cloud_cost_usd": float(ledger["consumed_cloud_cost_usd"]),
        "real_physical_shared_root_demonstrated": bool(smoke),
        "gpu_cow_claim": False,
        "shared_root_semantics": (
            "runtime-native immutable physical shared root with append-only private suffixes"
        ),
        "semantic_smoke": {
            "physical_state_observed": bool(smoke),
            "passed": clean_smoke,
            "failure_gate": None if clean_smoke else "driver_visible_hbm_postflight_recovery",
            "attempt_ids": [item.attempt_id for item in smoke],
            "observations": smoke_observations,
        },
        "measurements": primary_cells,
        "scaling": scaling,
        "tracing_overhead": trace_overhead,
        "branchfabric_interest": interest,
        "next_experiment_classification": next_classification,
        "recommended_next_gpu_experiment": _recommended_next(
            primary_cells,
            clean_smoke=clean_smoke,
            classification=next_classification,
        ),
        "cleanup": {
            "all_completed_workloads_passed_postflight": all(
                trial.postflight_clean for trial in trials
            ),
            "authoritative_lifecycle_attempt_ids": [trial.attempt_id for trial in lifecycle_trials],
            "clean_authoritative_lifecycle_attempt_ids": [
                trial.attempt_id for trial in clean_lifecycle_trials
            ],
            "all_accepted_16k_trials_lifecycle_clean": all(
                trial.lifecycle_clean is True
                for trial in trials
                if trial.prefix_length == EXPECTED_PREFIX_TOKENS
            ),
            "modal_ephemeral_function_containers": "single_use",
            "child_process_cleanup": (
                "verified_by_authoritative_controller_worker_artifacts"
                if clean_lifecycle_trials
                else "not_yet_verified_by_authoritative_controller_worker_artifacts"
            ),
            "driver_visible_hbm_postflight": (
                "authoritative_process_lifecycle_passed"
                if clean_smoke
                else "no_authoritative_clean_semantic_smoke"
            ),
            "provider_container_teardown": provider_cleanup["status"],
            "provider_cleanup_verified": provider_cleanup["verified"],
            "provider_cleanup_evidence": provider_cleanup,
            "provider_hbm_after_teardown": (
                "unverified_not_sampled"
                if provider_cleanup.get("provider_hbm_after_teardown_sampled") is not True
                else "sampled"
            ),
            "persistent_resources_allowed": [
                "sloforge-model-cache",
                "sloforge-branchfabric-results",
            ],
            "mew0_cleanup_receipt": "historical_previous_attempt_only",
        },
        "artifacts": {
            "root": str(experiment_root),
            "analysis": str(experiment_root / "analysis/experiment-summary.json"),
            "metrics": str(experiment_root / "metrics/experiment-summary.json"),
            "plots": str(experiment_root / "plots/plot-manifest.json"),
            "gpu_hours": str(experiment_root / "gpu-hours.json"),
        },
    }
    _replace_bytes(root / "analysis/experiment-summary.json", _canonical_bytes(analysis))
    _replace_bytes(root / "metrics/experiment-summary.json", _canonical_bytes(analysis))
    _replace_bytes(report_json, _canonical_bytes(report))
    _replace_bytes(report_md, _report_markdown(report).encode())
    (root / "metrics/UNAVAILABLE.json").unlink(missing_ok=True)
    if any(trial.tracing_level != _TRACE_CONTROL_LEVEL for trial in trials):
        (root / "traces/UNAVAILABLE.json").unlink(missing_ok=True)
    return report


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    parser.add_argument("--report-md", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    report = analyze(
        experiment_root=args.experiment_root,
        report_json=args.report_json,
        report_md=args.report_md,
    )
    print(_canonical_bytes(report).decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
