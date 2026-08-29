"""Real two-role KILL_AND_RECOMPUTE worker for BranchFabric Experiment 004.

The module imports the frozen K serving and retained-engine methodology but
never enters the optimized export/restore pipeline.  GPU1 destroys the live
rollout state, proves allocator-visible release, temporarily serves, then
replays the eight exact token histories through vLLM.  Submitted history,
prefix-cache reuse, and uncached scheduler work are reported separately.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
import time
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode()


def _write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(_canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _public(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "__dataclass_fields__"):
        return {
            field: _public(getattr(value, field))
            for field in value.__dataclass_fields__
            if not field.startswith("_")
        }
    if isinstance(value, Mapping):
        return {str(key): _public(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_public(item) for item in value]
    return value


_KILL_CONTROL_DRIFT_POLICY = "slope>0.10-and-last-quarter-median>first-quarter-median+2"


def _material_persistent_control_drift(
    diagnostic: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute the pre-existing material-and-persistent queue-drift rule."""

    expected_fields = {
        "sample_interval_ns",
        "samples",
        "initial_depth",
        "final_depth",
        "maximum_depth",
        "first_half_mean_depth",
        "second_half_mean_depth",
        "slope_requests_per_second",
        "sustained_positive",
    }
    samples = diagnostic.get("samples")
    interval_ns = diagnostic.get("sample_interval_ns")
    if (
        set(diagnostic) != expected_fields
        or type(interval_ns) is not int
        or interval_ns <= 0
        or not isinstance(samples, (tuple, list))
        or len(samples) < 4
    ):
        raise RuntimeError("kill/recompute control drift diagnostic is malformed")
    timestamps: list[int] = []
    depths: list[int] = []
    for row in samples:
        if not isinstance(row, Mapping) or set(row) != {"timestamp_ns", "total_outstanding"}:
            raise RuntimeError("kill/recompute control drift sample is malformed")
        timestamp_ns = row.get("timestamp_ns")
        depth = row.get("total_outstanding")
        if (
            type(timestamp_ns) is not int
            or timestamp_ns <= 0
            or type(depth) is not int
            or depth < 0
        ):
            raise RuntimeError("kill/recompute control drift sample is malformed")
        timestamps.append(timestamp_ns)
        depths.append(depth)
    if any(right - left != interval_ns for left, right in pairwise(timestamps)):
        raise RuntimeError("kill/recompute control drift sample cadence changed")
    if (
        diagnostic.get("initial_depth") != depths[0]
        or diagnostic.get("final_depth") != depths[-1]
        or diagnostic.get("maximum_depth") != max(depths)
    ):
        raise RuntimeError("kill/recompute control drift summary changed")
    seconds = tuple((timestamp - timestamps[0]) / 1_000_000_000.0 for timestamp in timestamps)
    mean_x = sum(seconds) / len(seconds)
    mean_y = sum(depths) / len(depths)
    denominator = sum((value - mean_x) ** 2 for value in seconds)
    if denominator <= 0.0:
        raise RuntimeError("kill/recompute control drift time span is empty")
    slope = (
        sum(
            (x_value - mean_x) * (y_value - mean_y)
            for x_value, y_value in zip(seconds, depths, strict=True)
        )
        / denominator
    )
    if diagnostic.get("slope_requests_per_second") != slope:
        raise RuntimeError("kill/recompute control drift slope changed")
    midpoint = len(depths) // 2
    first_half_mean = sum(depths[:midpoint]) / len(depths[:midpoint])
    second_half_mean = sum(depths[midpoint:]) / len(depths[midpoint:])
    frozen_sustained_positive = bool(
        slope > 0.0 and depths[-1] > depths[0] and second_half_mean > first_half_mean
    )
    if (
        diagnostic.get("first_half_mean_depth") != first_half_mean
        or diagnostic.get("second_half_mean_depth") != second_half_mean
        or diagnostic.get("sustained_positive") is not frozen_sustained_positive
    ):
        raise RuntimeError("kill/recompute control drift half summary changed")
    quarter_count = max(2, len(depths) // 4)
    first_median = float(statistics.median(depths[:quarter_count]))
    last_median = float(statistics.median(depths[-quarter_count:]))
    persistent = bool(slope > 0.10 and last_median > first_median + 2)
    return {
        "schema_version": "sloforge.branchfabric.kill-recompute-control-drift-policy/v1",
        "policy": _KILL_CONTROL_DRIFT_POLICY,
        "quarter_sample_count": quarter_count,
        "first_quarter_median_depth": first_median,
        "last_quarter_median_depth": last_median,
        "minimum_positive_slope_requests_per_second": 0.10,
        "minimum_material_median_increase": 2,
        "persistent_positive": persistent,
    }


def _kill_control_interval_evidence(
    result: Mapping[str, Any],
    *,
    expected_rate_rps: float,
    expected_duration_seconds: float,
    slo_ttft_seconds: float,
    warmup_seconds: float = 1.0,
    evaluation_seconds: float = 1.0,
    minimum_completion_fraction: float = 0.90,
    _base_evaluator: Any | None = None,
) -> dict[str, Any]:
    """Retain the frozen gate and replace only its over-sensitive drift predicate."""

    if _base_evaluator is None:
        import gpu_reclamation_integrated_worker_v11 as integrated_worker

        _base_evaluator = integrated_worker._control_interval_evidence
    if _base_evaluator is _kill_control_interval_evidence or not callable(_base_evaluator):
        raise RuntimeError("kill/recompute frozen control evaluator is unavailable")
    evidence = _base_evaluator(
        result,
        expected_rate_rps=expected_rate_rps,
        expected_duration_seconds=expected_duration_seconds,
        slo_ttft_seconds=slo_ttft_seconds,
        warmup_seconds=warmup_seconds,
        evaluation_seconds=evaluation_seconds,
        minimum_completion_fraction=minimum_completion_fraction,
    )
    if not isinstance(evidence, dict):
        raise RuntimeError("kill/recompute frozen control evidence is malformed")
    diagnostic = evidence.get("total_outstanding_diagnostic")
    if not isinstance(diagnostic, Mapping):
        raise RuntimeError("kill/recompute control drift diagnostic is absent")
    drift = _material_persistent_control_drift(diagnostic)
    kill_diagnostic = dict(diagnostic)
    kill_diagnostic.update(
        {
            "drift_policy": drift["policy"],
            "quarter_sample_count": drift["quarter_sample_count"],
            "first_quarter_median_depth": drift["first_quarter_median_depth"],
            "last_quarter_median_depth": drift["last_quarter_median_depth"],
            "minimum_positive_slope_requests_per_second": drift[
                "minimum_positive_slope_requests_per_second"
            ],
            "minimum_material_median_increase": drift["minimum_material_median_increase"],
            "sustained_positive": drift["persistent_positive"],
        }
    )
    queue_stable = not bool(drift["persistent_positive"])
    evidence["total_outstanding_diagnostic"] = kill_diagnostic
    evidence["kill_recompute_drift_policy"] = dict(drift)
    evidence["outstanding_queue_non_positive_trend"] = queue_stable
    evidence["passed"] = bool(
        evidence.get("full_arrivals") == evidence.get("expected_full_arrivals")
        and evidence.get("arrivals") == evidence.get("expected_arrivals")
        and evidence.get("unique_control_request_ids") is True
        and evidence.get("unique_scheduled_arrivals") is True
        and evidence.get("scheduled_arrival_cadence_exact") is True
        and evidence.get("monotonic_control_timestamps") is True
        and evidence.get("full_cohort_output_tokens_exact") is True
        and evidence.get("complete_full_request_accounting") is True
        and evidence.get("complete_request_accounting") is True
        and evidence.get("offered_rate_matches_config") is True
        and evidence.get("completion_tracks_offer") is True
        and evidence.get("p95_ttft_below_slo") is True
        and queue_stable
        and evidence.get("waiting_queue_bounded_below_normal_trigger") is True
        and evidence.get("total_outstanding_bounded_below_normal_trigger") is True
    )
    return evidence


def _history_evidence(
    branch_tables: Sequence[Any],
    *,
    prefix_length: int,
    suffix_length: int,
) -> dict[str, Any]:
    """Validate and count the exact histories actually submitted for replay."""

    rows: list[dict[str, Any]] = []
    ordered = sorted(branch_tables, key=lambda table: str(table.logical_branch_id))
    if [str(table.logical_branch_id) for table in ordered] != [f"branch.{i}" for i in range(8)]:
        raise RuntimeError("kill/recompute history does not cover exactly eight logical branches")
    common_prefix: tuple[int, ...] | None = None
    for table in ordered:
        raw_token_ids = tuple(table.token_ids)
        if any(type(item) is not int or item < 0 for item in raw_token_ids):
            raise RuntimeError("kill/recompute branch history contains an invalid token")
        token_ids = tuple(raw_token_ids)
        computed_tokens = table.computed_tokens
        expected_computed_tokens = prefix_length + suffix_length
        uncomputed_tail_count = len(token_ids) - computed_tokens
        if (
            type(computed_tokens) is not int
            or computed_tokens != expected_computed_tokens
            or uncomputed_tail_count != 1
        ):
            raise RuntimeError("kill/recompute branch history differs from its live KV boundary")
        prefix = token_ids[:prefix_length]
        if common_prefix is None:
            common_prefix = prefix
        elif prefix != common_prefix:
            raise RuntimeError("kill/recompute replay histories do not share the exact prefix")
        rows.append(
            {
                "logical_branch_id": str(table.logical_branch_id),
                "prefix_tokens_submitted": prefix_length,
                "private_tokens_submitted": len(token_ids) - prefix_length,
                "total_tokens_submitted": len(token_ids),
                "computed_tokens": computed_tokens,
                "computed_private_tokens": computed_tokens - prefix_length,
                "uncomputed_tail_token_count": uncomputed_tail_count,
                "uncomputed_tail_token_ids": list(token_ids[computed_tokens:]),
                "token_history_sha256": hashlib.sha256(_canonical_bytes(token_ids)).hexdigest(),
                "computed_boundary_sha256": hashlib.sha256(
                    _canonical_bytes(token_ids[:computed_tokens])
                ).hexdigest(),
                "token_ids": list(token_ids),
            }
        )
    submitted_prefix = sum(row["prefix_tokens_submitted"] for row in rows)
    submitted_private = sum(row["private_tokens_submitted"] for row in rows)
    submitted_total = sum(row["total_tokens_submitted"] for row in rows)
    computed_boundary_tokens = sum(row["computed_tokens"] for row in rows)
    live_uncomputed_tail_tokens = sum(row["uncomputed_tail_token_count"] for row in rows)
    unique_recompute = prefix_length + submitted_private
    analytical_expectation = 133_120
    analytical_unique = 18_432
    if (
        submitted_prefix != 131_072
        or computed_boundary_tokens != analytical_expectation
        or live_uncomputed_tail_tokens != 8
        or submitted_private != 2_048 + live_uncomputed_tail_tokens
        or submitted_total != analytical_expectation + live_uncomputed_tail_tokens
        or unique_recompute != analytical_unique + live_uncomputed_tail_tokens
    ):
        raise RuntimeError("kill/recompute exact replay accounting changed")
    return {
        "schema_version": "sloforge.branchfabric.kill-recompute-history/v2",
        "branches": rows,
        "computed_boundary_tokens": computed_boundary_tokens,
        "live_uncomputed_tail_tokens": live_uncomputed_tail_tokens,
        "submitted_prefix_tokens": submitted_prefix,
        "submitted_private_tokens": submitted_private,
        "submitted_replay_tokens": submitted_total,
        "submitted_vs_analytical_delta_tokens": live_uncomputed_tail_tokens,
        "analytical_independent_branch_expectation_tokens": analytical_expectation,
        "unique_submitted_replay_history_tokens": unique_recompute,
        "computed_state_topology_tokens": analytical_unique,
        "lost_private_rollout_work_tokens": 2_048,
        "passed": True,
    }


def _run_kill_calibration_phase(
    *,
    frozen_worker: Any,
    adapter: Any,
    config: dict[str, Any],
    inputs: dict[str, Any],
    role: str,
    physical_gpu_uuid: str,
    barrier_root: Path,
    model_load_started_ns: int,
    model_ready_ns: int,
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    """Reuse K's exact A/B/15 retained-engine command count, then restore it."""

    original_probe_count = frozen_worker._V10_SANITY_GUARD_COUNT
    original_validator = frozen_worker._validate_integrated_transaction_command
    if original_probe_count != 2:
        raise RuntimeError("frozen v10 sanity command count drifted from two")

    def validate_kill_command(**kwargs: Any) -> tuple[dict[str, Any], dict[str, str]]:
        base_config = kwargs.get("base_config")
        command = kwargs.get("command")
        expected_command_fields = {
            "schema_version",
            "effective_config",
            "selection_sha256",
            "authorization_artifact_hash",
            "sanity_result_sha256",
            "issued_at_monotonic_ns",
        }
        if (
            not isinstance(base_config, dict)
            or base_config.get("schema_version")
            != "sloforge.branchfabric.experiment-004-kill-recompute-config/v1"
            or base_config.get("mode") != "KILL_AND_RECOMPUTE"
            or base_config.get("execution_mode") != "integrated-kill-recompute-v11"
            or not isinstance(command, dict)
            or set(command) != expected_command_fields
            or command.get("schema_version")
            != "sloforge.branchfabric.kill-recompute-transaction-command/v1"
        ):
            raise ValueError("kill/recompute transaction command envelope is invalid")
        delegated = dict(command)
        delegated["schema_version"] = "sloforge.branchfabric.integrated-transaction-command/v1"
        return original_validator(
            base_config=base_config,
            command=delegated,
            selected_load_sha256=kwargs["selected_load_sha256"],
        )

    frozen_worker._V10_SANITY_GUARD_COUNT = 3
    frozen_worker._validate_integrated_transaction_command = validate_kill_command
    try:
        return frozen_worker._run_integrated_calibration_phase(
            adapter=adapter,
            config=config,
            inputs=inputs,
            role=role,
            physical_gpu_uuid=physical_gpu_uuid,
            barrier_root=barrier_root,
            model_load_started_ns=model_load_started_ns,
            model_ready_ns=model_ready_ns,
        )
    finally:
        frozen_worker._validate_integrated_transaction_command = original_validator
        frozen_worker._V10_SANITY_GUARD_COUNT = original_probe_count


class _SchedulerReplayCounter:
    """Measure vLLM scheduler work through each replay request's first token."""

    def __init__(self, scheduler: Any, request_ids: Mapping[str, str]) -> None:
        self.scheduler = scheduler
        self.logical_by_request_id = {
            str(request_id): str(logical) for logical, request_id in request_ids.items()
        }
        self._original = scheduler.schedule
        self._active = set(self.logical_by_request_id.values())
        self.scheduled_by_branch = {logical: 0 for logical in self._active}
        self.cached_by_branch: dict[str, int | None] = {logical: None for logical in self._active}
        self.schedule_calls = 0
        self.raw_rows: list[dict[str, Any]] = []
        self.raw_cache_admission_rows: list[dict[str, Any]] = []

    @staticmethod
    def _request_id(value: Any) -> str | None:
        if isinstance(value, Mapping):
            request_id = value.get("request_id")
            req_id = value.get("req_id")
        else:
            request_id = getattr(value, "request_id", None)
            req_id = getattr(value, "req_id", None)
        if request_id is not None and req_id is not None and request_id != req_id:
            raise RuntimeError("vLLM scheduler replay request identities conflict")
        raw = request_id if request_id is not None else req_id
        return str(raw) if isinstance(raw, str) and raw else None

    @staticmethod
    def _integer(value: Any, name: str) -> int | None:
        raw = value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)
        return raw if type(raw) is int and raw >= 0 else None

    def _record_cache_admission(self, output: Any) -> None:
        rows = getattr(output, "scheduled_new_reqs", ())
        if not isinstance(rows, Sequence):
            raise RuntimeError("vLLM scheduler output lacks scheduled_new_reqs evidence")
        for row in rows:
            request_id = self._request_id(row)
            logical = self.logical_by_request_id.get(str(request_id))
            if logical is None:
                raise RuntimeError("vLLM scheduler admitted an unknown replay request")
            if logical not in self._active:
                raise RuntimeError("vLLM scheduler admitted a replay request more than once")
            cached = self._integer(row, "num_cached_tokens")
            if cached is None:
                cached = self._integer(row, "num_computed_tokens")
            if cached is None:
                raise RuntimeError("vLLM scheduler did not expose replay cache-hit tokens")
            prior = self.cached_by_branch[logical]
            if prior is not None:
                raise RuntimeError("vLLM replay cache admission was observed more than once")
            self.raw_cache_admission_rows.append(
                {
                    "logical_branch_id": logical,
                    "runtime_request_id": str(request_id),
                    "runtime_cache_reused_tokens": cached,
                }
            )
            self.cached_by_branch[logical] = cached

    def _wrapped_schedule(self) -> Any:
        output = self._original()
        scheduled = getattr(output, "num_scheduled_tokens", None)
        if not isinstance(scheduled, Mapping):
            raise RuntimeError("vLLM scheduler output lacks per-request scheduled-token evidence")
        self._record_cache_admission(output)
        row: dict[str, int] = {}
        for request_id, token_count in scheduled.items():
            request_key = str(request_id)
            if request_key not in self.logical_by_request_id:
                raise RuntimeError("vLLM scheduler reported an unknown replay request")
            logical = self.logical_by_request_id[request_key]
            if logical not in self._active:
                continue
            if type(token_count) is not int or token_count < 0:
                raise RuntimeError("vLLM scheduler emitted an invalid replay token count")
            self.scheduled_by_branch[logical] += token_count
            row[logical] = token_count
        self.raw_rows.append(row)
        self.schedule_calls += 1
        return output

    def __enter__(self) -> _SchedulerReplayCounter:
        self.scheduler.schedule = self._wrapped_schedule
        return self

    def observe_first_token(self, logical: str) -> None:
        self._active.discard(logical)

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.scheduler.schedule = self._original

    def evidence(self, *, submitted_by_branch: Mapping[str, int]) -> dict[str, Any]:
        if self._active:
            raise RuntimeError("replay scheduler accounting ended before every first token")
        if any(value is None for value in self.cached_by_branch.values()):
            raise RuntimeError("replay scheduler accounting lacks per-branch cache evidence")
        cached = {
            key: int(value) for key, value in self.cached_by_branch.items() if value is not None
        }
        for logical, submitted in submitted_by_branch.items():
            if self.scheduled_by_branch.get(logical, -1) + cached.get(logical, -1) != submitted:
                raise RuntimeError("replay scheduler computed/cache accounting does not conserve")
        if len(self.raw_cache_admission_rows) != len(submitted_by_branch):
            raise RuntimeError("replay scheduler accounting lacks exact cache admissions")
        raw_schedule_rows = [dict(row) for row in self.raw_rows]
        raw_cache_rows = [dict(row) for row in self.raw_cache_admission_rows]
        return {
            "schema_version": "sloforge.branchfabric.vllm-replay-scheduler-accounting/v1",
            "scheduler_schedule_calls": self.schedule_calls,
            "submitted_tokens_by_branch": dict(sorted(submitted_by_branch.items())),
            "runtime_cache_reused_tokens_by_branch": dict(sorted(cached.items())),
            "runtime_uncached_scheduled_tokens_by_branch": dict(
                sorted(self.scheduled_by_branch.items())
            ),
            "runtime_cache_reused_tokens": sum(cached.values()),
            "runtime_uncached_scheduled_tokens": sum(self.scheduled_by_branch.values()),
            "raw_schedule_rows": raw_schedule_rows,
            "raw_schedule_rows_sha256": hashlib.sha256(
                _canonical_bytes(raw_schedule_rows)
            ).hexdigest(),
            "raw_cache_admission_rows": raw_cache_rows,
            "raw_cache_admission_rows_sha256": hashlib.sha256(
                _canonical_bytes(raw_cache_rows)
            ).hexdigest(),
            "accounting_conserves_submitted_history": True,
            "passed": True,
        }


def _prepare_rollouts_with_measurement(
    adapter: Any,
    config: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Run K's exact source build while timing total and private lost work."""

    import torch  # type: ignore[import-not-found]

    prefix = tuple(int(item) for item in inputs["prefix_token_ids"][:16_384])
    divergent = tuple(int(item) for item in inputs["divergent_token_ids"][:8])
    if len(prefix) != 16_384 or len(divergent) != 8:
        raise ValueError("cached Experiment 003 tokenizer inputs are incomplete")
    root_id = f"{config['attempt_id']}-root"
    branches = tuple(f"branch.{index}" for index in range(8))
    source_started_ns = time.monotonic_ns()
    source_start = torch.cuda.Event(enable_timing=True)
    source_end = torch.cuda.Event(enable_timing=True)
    source_start.record()
    adapter.start_session(root_id, token_ids=prefix, seed=int(config["seed"]), timeout_s=30.0)
    adapter.prefill_session(root_id, timeout_s=300.0)
    adapter.pause_at_safe_decode_boundary(root_id, timeout_s=5.0)
    root = adapter.create_shared_root_reference(root_id, prefix_token_count=16_384, timeout_s=30.0)
    for branch_id, token_id in zip(branches, divergent, strict=True):
        adapter.fork_same_policy_session(
            root.root_reference_id,
            branch_id,
            divergent_token_id=token_id,
            seed=int(config["seed"]),
            timeout_s=10.0,
        )
    private_started_ns = time.monotonic_ns()
    private_start = torch.cuda.Event(enable_timing=True)
    private_end = torch.cuda.Event(enable_timing=True)
    private_start.record()
    decode = adapter.run_concurrent_branches(
        branches,
        maximum_new_tokens=256,
        seed=int(config["seed"]),
        timeout_s=300.0,
    )
    private_end.record()
    private_end.synchronize()
    private_completed_ns = time.monotonic_ns()
    source_end.record()
    source_end.synchronize()
    source_completed_ns = time.monotonic_ns()
    if set(decode.completed_branch_ids) != set(branches):
        raise RuntimeError("not all eight rollout branches reached the checkpoint boundary")
    source_wall = (source_completed_ns - source_started_ns) / 1_000_000_000.0
    source_gpu = source_start.elapsed_time(source_end) / 1_000.0
    private_wall = (private_completed_ns - private_started_ns) / 1_000_000_000.0
    private_gpu = private_start.elapsed_time(private_end) / 1_000.0
    if not 0.0 < private_gpu <= private_wall <= source_wall or not 0.0 < source_gpu <= source_wall:
        raise RuntimeError("kill/recompute source-build timing is physically inconsistent")
    return {
        "root_id": root_id,
        "root_reference_id": root.root_reference_id,
        "branches": branches,
        "source_build_measurement": {
            "schema_version": "sloforge.branchfabric.kill-recompute-source-build/v1",
            "started_ns": source_started_ns,
            "completed_ns": source_completed_ns,
            "wall_seconds": source_wall,
            "gpu_seconds": source_gpu,
            "unique_prefix_tokens": 16_384,
            "private_rollout_tokens": 2_048,
            "unique_source_tokens": 18_432,
            "private_rollout_started_ns": private_started_ns,
            "private_rollout_completed_ns": private_completed_ns,
            "lost_private_rollout_wall_seconds": private_wall,
            "lost_private_rollout_gpu_seconds": private_gpu,
            "lost_private_rollout_gpu_time_provenance": (
                "CUDA events enclosing only vLLM run_concurrent_branches for the "
                "eight 256-token private trajectories"
            ),
            "passed": True,
        },
    }


def run_kill_recompute_gpu0(
    engine: Any,
    *,
    adapter: Any,
    inputs: Mapping[str, Any],
    config: Any,
    start_ns: int,
    barriers: Path,
) -> dict[str, Any]:
    """Reuse the exact frozen K serving path while GPU1 recomputes."""

    import gpu_reclamation_integrated_worker_v11 as integrated_worker

    frozen_evaluator = integrated_worker._control_interval_evidence

    def kill_evaluator(result: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
        return _kill_control_interval_evidence(
            result,
            _base_evaluator=frozen_evaluator,
            **kwargs,
        )

    integrated_worker._control_interval_evidence = kill_evaluator
    try:
        result = integrated_worker.run_integrated_v11_gpu0(
            engine,
            adapter=adapter,
            inputs=inputs,
            config=config,
            start_ns=start_ns,
            barriers=barriers,
        )
    finally:
        integrated_worker._control_interval_evidence = frozen_evaluator
    result["schema_version"] = "sloforge.branchfabric.kill-recompute-gpu0-result/v1"
    result["mode"] = "KILL_AND_RECOMPUTE"
    return result


def run_kill_recompute_gpu1(
    serving_engine: Any,
    *,
    adapter: Any,
    prepared: Mapping[str, Any],
    inputs: Mapping[str, Any],
    config: Any,
    physical_gpu_uuid: str,
    barriers: Path,
) -> dict[str, Any]:
    """Discard live state, serve, replay exact histories, and resume branches."""

    import torch  # type: ignore[import-not-found]
    from gpu_capacity_calibration_worker import _runtime_queue_state
    from gpu_reclamation_integrated_worker_v11 import (
        build_live_v10_config,
        validate_reclaim_trigger_timeline,
    )
    from gpu_reclamation_v10_serving import run_v10_gpu1
    from gpu_reclamation_worker import (
        _geometry,
        _runtime_capture_inputs,
        _sampling_params,
        _wait_for,
    )
    from gpu_reclamation_worker_v11 import _run_independent_oracle

    from sloforge.continuum.adapters.vllm_reclamation import build_canonical_capture_plan
    from sloforge.continuum.adapters.vllm_reclamation_v11 import Vllm0230EngineStepBinding
    from sloforge.continuum.adapters.vllm_reclamation_v11_ownership import (
        capture_source_ownership_v11,
        release_source_and_prove_v11,
    )

    trigger = _wait_for(
        barriers / "v10-reclaim-trigger.json", timeout_s=config.maximum_wall_seconds
    )
    trigger_ns = int(trigger["triggered_ns"])
    rollout_admission_stop_ns = time.monotonic_ns()
    branches = tuple(str(item) for item in prepared["branches"])
    _tensors, geometry = _geometry(adapter)
    binding = Vllm0230EngineStepBinding(adapter._view, acquire_timeout_seconds=30.0)
    source_sampling_by_branch = {
        branch: {
            "effective_seed": int(adapter._effective_seed(config.seed, adapter._sessions[branch]))
        }
        for branch in branches
    }
    try:
        memory_before = adapter.inspect_gpu_memory_state(timeout_s=10.0)
        # Ownership capture/release intentionally consumes the production
        # EXPORT_CAPTURE witness type.  No export is performed; the kill-specific
        # gate ID and discard artifact record the actual operation semantics.
        with binding.critical_section(
            "EXPORT_CAPTURE", gate_id=f"{config.attempt_id}:kill-discard"
        ) as discard_gate:
            state_quiescence_ns = time.monotonic_ns()
            runtime_inputs = _runtime_capture_inputs(
                adapter,
                branches,
                parent_logical_branch_id=str(prepared["root_id"]),
            )
            history = _history_evidence(
                runtime_inputs,
                prefix_length=config.prefix_length,
                suffix_length=config.suffix_length,
            )
            plan = build_canonical_capture_plan(
                branches=runtime_inputs,
                block_size_tokens=geometry.block_size_tokens,
                logical_token_bytes=geometry.logical_token_bytes,
                physical_page_bytes=geometry.physical_page_bytes,
                gpu_uuid=physical_gpu_uuid,
                allocation_epoch_by_block=dict(adapter._observer.block_epochs),
            )
            source_epochs = {
                int(binding.source.block_index): int(binding.source.allocation_epoch)
                for binding in plan.capture_evidence.bindings
            }
            ownership_before = capture_source_ownership_v11(
                adapter,
                root_session_id=str(prepared["root_id"]),
                branch_session_ids=branches,
                expected_allocation_epochs=source_epochs,
                expected_source_block_count=config.expected_total_blocks,
                admission_gate=discard_gate,
                timeout_s=60.0,
            )
            state_discard_started_ns = time.monotonic_ns()
            release = release_source_and_prove_v11(
                adapter,
                ownership_before,
                admission_gate=discard_gate,
                timeout_s=60.0,
            )
            release.require_passed()
            state_discard_completed_ns = time.monotonic_ns()
        memory_after_release = adapter.inspect_gpu_memory_state(timeout_s=10.0)
        if int(memory_after_release.kv_assigned_bytes) != 0:
            raise RuntimeError("kill/recompute discard left assigned rollout KV bytes")
        hbm_reclaim_ns = time.monotonic_ns()
        discard = {
            "schema_version": "sloforge.branchfabric.kill-recompute-state-discard/v1",
            "event": "STATE_DISCARDED",
            "attempt_id": config.attempt_id,
            "state_discard_started_ns": state_discard_started_ns,
            "state_discard_completed_ns": state_discard_completed_ns,
            "hbm_reclaim_confirmed_ns": hbm_reclaim_ns,
            "source_block_count": release.source_block_count,
            "kv_assigned_bytes_before": int(memory_before.kv_assigned_bytes),
            "kv_assigned_bytes_after": int(memory_after_release.kv_assigned_bytes),
            "allocator_hbm_agreement": release.allocator_hbm_agreement,
            "export_started": False,
            "checkpoint_materialized": False,
            "post_free_ownership": _public(release),
            "history": history,
            "passed": True,
        }
        _write_new(barriers / "kill-state-discarded.json", discard)

        serving = run_v10_gpu1(
            serving_engine,
            prefix=tuple(int(item) for item in inputs["prefix_token_ids"][:256]),
            params=_sampling_params(max_tokens=config.serving_output_tokens, seed=config.seed),
            config=build_live_v10_config(config),
            barriers=barriers,
            write_new=_write_new,
            runtime_queue_state=lambda: _runtime_queue_state(adapter),
        )
        gpu1_ready = json.loads((barriers / "v10-gpu1-serving-ready.json").read_text())
        gpu1_serving_ready_ns = int(gpu1_ready["observed_ns"])
        if gpu1_serving_ready_ns < hbm_reclaim_ns:
            raise RuntimeError("kill/recompute GPU1 serving preceded HBM reclamation")
        if not adapter._view.manager.reset_prefix_cache():
            raise RuntimeError("kill/recompute GPU1 serving allocation did not drain")
        adapter._allocator_epoch_source.observe_prefix_cache_reset()
        restore = _wait_for(
            barriers / "v10-restore-start.json", timeout_s=config.maximum_wall_seconds
        )
        restore_trigger_ns = int(restore["observed_ns"])
        replay_started_ns = time.monotonic_ns()
        logical_by_external: dict[str, str] = {}
        internal_by_logical: dict[str, str] = {}
        submitted_by_branch: dict[str, int] = {}
        for table in sorted(runtime_inputs, key=lambda item: str(item.logical_branch_id)):
            logical = str(table.logical_branch_id)
            external = f"{logical}@kill-recompute-1"
            token_ids = tuple(int(item) for item in table.token_ids)
            internal = adapter._view.llm_engine.add_request(
                external,
                {"prompt_token_ids": list(token_ids)},
                _sampling_params(
                    max_tokens=config.continuation_tokens,
                    seed=int(source_sampling_by_branch[logical]["effective_seed"]),
                ),
            )
            if not isinstance(internal, str) or not internal:
                raise RuntimeError("kill/recompute replay request lacks internal identity")
            if internal in internal_by_logical.values():
                raise RuntimeError("kill/recompute replay request identity was reused")
            logical_by_external[external] = logical
            internal_by_logical[logical] = internal
            submitted_by_branch[logical] = len(token_ids)

        observed_first: dict[str, int] = {}
        output_tokens: dict[str, tuple[int, ...]] = {branch: () for branch in branches}
        first_token_ns: dict[str, int] = {}
        completed_ns: dict[str, int] = {}
        replay_event_start = torch.cuda.Event(enable_timing=True)
        replay_event_end = torch.cuda.Event(enable_timing=True)
        replay_event_start.record()
        counter = _SchedulerReplayCounter(adapter._view.scheduler, internal_by_logical)
        deadline_ns = time.monotonic_ns() + 180_000_000_000
        all_first_ns: int | None = None
        with counter:
            while len(completed_ns) < len(branches):
                if time.monotonic_ns() >= deadline_ns:
                    raise TimeoutError("kill/recompute replay exceeded its bounded timeout")
                outputs = adapter._view.llm_engine.step()
                observed_ns = time.monotonic_ns()
                for output in outputs:
                    logical = logical_by_external.get(str(getattr(output, "request_id", "")))
                    if logical is None:
                        continue
                    candidates = getattr(output, "outputs", ())
                    tokens = tuple(int(item) for item in getattr(candidates[0], "token_ids", ()))
                    if tokens:
                        output_tokens[logical] = tokens
                        if logical not in observed_first:
                            observed_first[logical] = tokens[0]
                            first_token_ns[logical] = observed_ns
                            counter.observe_first_token(logical)
                            if len(observed_first) == len(branches):
                                replay_event_end.record()
                                all_first_ns = observed_ns
                    if bool(getattr(output, "finished", False)):
                        completed_ns[logical] = observed_ns
                if not outputs:
                    time.sleep(0.001)
        if all_first_ns is None:
            raise RuntimeError("kill/recompute replay never reached all first tokens")
        replay_event_end.synchronize()
        replay_gpu_seconds = replay_event_start.elapsed_time(replay_event_end) / 1_000.0
        replay_first_token_ns = min(first_token_ns.values())
        replay_complete_ns = max(completed_ns.values())
        scheduler_accounting = counter.evidence(submitted_by_branch=submitted_by_branch)
        if sum(submitted_by_branch.values()) != history["submitted_replay_tokens"]:
            raise RuntimeError("kill/recompute submitted histories changed at engine admission")
        if min(len(tokens) for tokens in output_tokens.values()) < config.continuation_tokens:
            raise RuntimeError("kill/recompute did not resume every branch for eight tokens")

        _write_new(
            barriers / "rollout-restore-complete.json",
            {
                "schema_version": "sloforge.branchfabric.v11-rollout-restore-complete/v1",
                "mode": "KILL_AND_RECOMPUTE",
                "restore_trigger_ns": restore_trigger_ns,
                "native_restore_complete_ns": all_first_ns,
                "first_resumed_token_ns": replay_first_token_ns,
                "all_branches_resumed_ns": all_first_ns,
                "continuation_complete_ns": replay_complete_ns,
                "minimum_tokens_per_branch": min(len(value) for value in output_tokens.values()),
                "branch_count": len(branches),
            },
        )
        if not adapter._view.manager.reset_prefix_cache():
            raise RuntimeError("kill/recompute replay requests did not drain before oracle")
        adapter._allocator_epoch_source.observe_prefix_cache_reset()
        expected_first = _run_independent_oracle(
            adapter._view.llm_engine,
            branch_tables=runtime_inputs,
            source_sampling_by_branch=source_sampling_by_branch,
            sampling_params=_sampling_params,
            timeout_seconds=180.0,
        )
        exact = observed_first == expected_first and len(observed_first) == len(branches)
        if not exact:
            raise RuntimeError("kill/recompute first resumed tokens differ from recompute oracle")
        if scheduler_accounting["runtime_uncached_scheduled_tokens"] <= 0:
            raise RuntimeError("kill/recompute runtime reported zero uncached replay work")
        prefill_wall_seconds = (all_first_ns - replay_started_ns) / 1_000_000_000.0
        if prefill_wall_seconds <= 0 or replay_gpu_seconds <= 0:
            raise RuntimeError("kill/recompute replay timing is not positive")
        trigger_timeline = validate_reclaim_trigger_timeline(
            trigger,
            rollout_admission_stop_ns=rollout_admission_stop_ns,
            state_quiescence_ns=state_quiescence_ns,
        )
        return {
            "schema_version": "sloforge.branchfabric.kill-recompute-gpu1-result/v1",
            "status": "succeeded",
            "mode": "KILL_AND_RECOMPUTE",
            "attempt_id": config.attempt_id,
            "physical_gpu_uuid": physical_gpu_uuid,
            "trigger_timeline": trigger_timeline,
            "state_discard": discard,
            "serving": serving,
            "recompute": {
                "schema_version": "sloforge.branchfabric.kill-recompute-measurement/v1",
                "history": history,
                "discarded_source_build_measurement": dict(prepared["source_build_measurement"]),
                "scheduler_accounting": scheduler_accounting,
                "actual_submitted_prefix_tokens": history["submitted_prefix_tokens"],
                "actual_submitted_private_tokens": history["submitted_private_tokens"],
                "actual_submitted_replay_tokens": history["submitted_replay_tokens"],
                "actual_computed_boundary_tokens": history["computed_boundary_tokens"],
                "actual_live_uncomputed_tail_tokens": history["live_uncomputed_tail_tokens"],
                "submitted_vs_analytical_delta_tokens": history[
                    "submitted_vs_analytical_delta_tokens"
                ],
                "actual_runtime_cache_reused_tokens": scheduler_accounting[
                    "runtime_cache_reused_tokens"
                ],
                "actual_runtime_uncached_recompute_tokens": scheduler_accounting[
                    "runtime_uncached_scheduled_tokens"
                ],
                "gpu_computed_tokens_derived_from_scheduler": scheduler_accounting[
                    "runtime_uncached_scheduled_tokens"
                ],
                "gpu_computed_token_provenance": (
                    "vllm SchedulerOutput.num_scheduled_tokens through each replay "
                    "request's first token; CUDA events provide time, not token counts"
                ),
                "analytical_independent_branch_expectation_tokens": 133_120,
                "unique_submitted_replay_history_tokens": history[
                    "unique_submitted_replay_history_tokens"
                ],
                "computed_state_topology_tokens": history["computed_state_topology_tokens"],
                "lost_private_rollout_work_tokens": history["lost_private_rollout_work_tokens"],
                "prefill_wall_seconds": prefill_wall_seconds,
                "prefill_gpu_seconds": replay_gpu_seconds,
                "uncached_prefill_tokens_per_second_wall": (
                    scheduler_accounting["runtime_uncached_scheduled_tokens"] / prefill_wall_seconds
                ),
                "uncached_prefill_tokens_per_gpu_second": (
                    scheduler_accounting["runtime_uncached_scheduled_tokens"] / replay_gpu_seconds
                ),
                "restore_trigger_to_first_resumed_token_seconds": (
                    replay_first_token_ns - restore_trigger_ns
                )
                / 1_000_000_000.0,
                "restore_trigger_to_all_branches_resumed_seconds": (
                    all_first_ns - restore_trigger_ns
                )
                / 1_000_000_000.0,
                "restore_trigger_to_all_branches_complete_seconds": (
                    replay_complete_ns - restore_trigger_ns
                )
                / 1_000_000_000.0,
                "passed": True,
            },
            "timings": {
                "reclaim_trigger_ns": trigger_ns,
                "rollout_admission_stop_ns": rollout_admission_stop_ns,
                "state_quiescence_ns": state_quiescence_ns,
                "state_discard_started_ns": state_discard_started_ns,
                "state_discard_completed_ns": state_discard_completed_ns,
                "hbm_reclaim_confirmed_ns": hbm_reclaim_ns,
                "gpu1_serving_ready_ns": gpu1_serving_ready_ns,
                "restore_trigger_ns": restore_trigger_ns,
                "replay_started_ns": replay_started_ns,
                "first_resumed_token_ns": replay_first_token_ns,
                "all_branches_resumed_ns": all_first_ns,
                "continuation_complete_ns": replay_complete_ns,
            },
            "continuation": {
                "observed_first_tokens": observed_first,
                "independent_recompute_first_tokens": expected_first,
                "exact_matches": len(observed_first),
                "minimum_tokens_per_branch": min(len(value) for value in output_tokens.values()),
                "branch_count": len(branches),
                "passed": True,
            },
            "correctness": {
                "state_discard_pass": True,
                "no_checkpoint_export_pass": True,
                "post_free_ownership_pass": release.passed,
                "allocator_hbm_agreement_pass": release.allocator_hbm_agreement,
                "gpu1_serving_after_release_pass": gpu1_serving_ready_ns >= hbm_reclaim_ns,
                "actual_recompute_accounting_pass": scheduler_accounting["passed"],
                "all_branches_resumed_pass": True,
                "first_token_exact_8_of_8_pass": exact,
            },
        }
    finally:
        binding.close()


def run_worker(
    role: Literal["serving", "rollout"],
    config_payload: dict[str, Any],
    *,
    model_snapshot: Path,
    physical_gpu_uuid: str,
    work_root: Path,
    barrier_root: Path,
) -> dict[str, Any]:
    """Load one retained engine and execute one kill/recompute role."""

    from gpu_reclamation_integrated_worker_v11 import (
        _validate_runtime,
        expanded_runtime_config,
    )

    from sloforge.helix.characterization.gpu_reclamation_v11_methodology import (
        Experiment004V11KillRecomputeConfig,
        validate_bound_artifact,
    )

    config = Experiment004V11KillRecomputeConfig.model_validate(config_payload, strict=True)
    repository_root = Path(__file__).resolve().parents[2]
    for reference, expected in (
        (config.integrated_k_status_artifact, config.integrated_k_status_sha256),
        (config.integrated_k_remote_manifest, config.integrated_k_remote_manifest_sha256),
        (
            config.integrated_k_provider_cleanup_artifact,
            config.integrated_k_provider_cleanup_sha256,
        ),
        (config.optimized_v11_freeze_artifact, config.optimized_v11_freeze_sha256),
        (
            config.optimized_v11_freeze_tag_binding_artifact,
            config.optimized_v11_freeze_tag_binding_sha256,
        ),
        (config.ledger_snapshot_artifact, config.ledger_snapshot_sha256),
        (config.budget_authorization, config.budget_authorization_sha256),
    ):
        validate_bound_artifact(repository_root, reference=reference, expected_sha256=expected)
    os.environ.update(
        {
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "TRITON_CACHE_DIR": str(work_root / "triton"),
        }
    )
    (work_root / "triton").mkdir(parents=True, exist_ok=True)
    _validate_runtime(config, physical_gpu_uuid=physical_gpu_uuid)
    inputs = json.loads((model_snapshot / "BRANCHFABRIC_INPUTS.json").read_text())
    import gpu_reclamation_worker as frozen_worker
    from gpu_capacity_calibration_worker import _CompilationLogCapture
    from gpu_reclamation_worker import (
        _create_adapter,
        _run_measured_v10_transaction,
        _transaction_ready_path,
        _wait_for,
    )

    adapter: Any = None
    try:
        started_ns = time.monotonic_ns()
        _write_new(
            barrier_root / f"{role}.engine-started.json",
            {
                "schema_version": "sloforge.branchfabric.retained-engine-start/v1",
                "role": role,
                "device": "gpu0" if role == "serving" else "gpu1",
                "pid": os.getpid(),
                "physical_gpu_uuid": physical_gpu_uuid,
                "engine_started_ns": started_ns,
            },
        )
        adapter = _create_adapter(
            expanded_runtime_config(config), model_snapshot, physical_gpu_uuid
        )
        model_ready_ns = time.monotonic_ns()
        effective, serving_engine, handoff = _run_kill_calibration_phase(
            frozen_worker=frozen_worker,
            adapter=adapter,
            config=expanded_runtime_config(config),
            inputs=inputs,
            role=role,
            physical_gpu_uuid=physical_gpu_uuid,
            barrier_root=barrier_root,
            model_load_started_ns=started_ns,
            model_ready_ns=model_ready_ns,
        )
        if effective != expanded_runtime_config(config):
            raise RuntimeError("kill/recompute handoff changed its sealed effective config")
        prepared = None
        if role == "rollout":
            prepared = _prepare_rollouts_with_measurement(adapter, effective, inputs)
        _write_new(
            _transaction_ready_path(barrier_root, role),
            {
                **handoff,
                "schema_version": "sloforge.branchfabric.v11-transaction-ready/v1",
                "rollouts_ready": prepared is not None,
                "branch_count": 0 if prepared is None else len(prepared["branches"]),
                "observed_at_monotonic_ns": time.monotonic_ns(),
            },
        )
        start = _wait_for(barrier_root / "start.json", timeout_s=config.maximum_wall_seconds)
        common_start_ns = int(start["start_monotonic_ns"])

        def operation() -> dict[str, Any]:
            if role == "serving":
                return run_kill_recompute_gpu0(
                    serving_engine,
                    adapter=adapter,
                    inputs=inputs,
                    config=config,
                    start_ns=common_start_ns,
                    barriers=barrier_root,
                )
            assert prepared is not None
            return run_kill_recompute_gpu1(
                serving_engine,
                adapter=adapter,
                prepared=prepared,
                inputs=inputs,
                config=config,
                physical_gpu_uuid=physical_gpu_uuid,
                barriers=barrier_root,
            )

        result = _run_measured_v10_transaction(
            role=role,
            work_root=work_root,
            operation=operation,
            compilation_capture_factory=_CompilationLogCapture,
        )
        result.update(
            {
                "retained_engine_handoff": handoff,
                "role": role,
                "pid": os.getpid(),
                "physical_gpu_uuid": physical_gpu_uuid,
                "attempt_id": config.attempt_id,
                "status": "succeeded",
                "mode": "KILL_AND_RECOMPUTE",
            }
        )
        cache_evidence = serving_engine.evidence()
        if cache_evidence["passed"] is not True:
            raise RuntimeError("kill/recompute serving cache-salt evidence failed")
        result["cache_salt_evidence"] = cache_evidence
        return result
    finally:
        if adapter is not None:
            adapter.cleanup_runtime(timeout_s=float(config.cleanup_timeout_seconds))
        try:
            import torch  # type: ignore[import-not-found]

            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("serving", "rollout"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--physical-gpu-uuid", required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--barrier-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    args.work_root.mkdir(parents=True, exist_ok=True)
    try:
        result = run_worker(
            args.role,
            json.loads(args.config.read_text()),
            model_snapshot=args.model_snapshot.resolve(strict=True),
            physical_gpu_uuid=args.physical_gpu_uuid,
            work_root=args.work_root,
            barrier_root=args.barrier_root,
        )
        _write_new(args.work_root / "result.json", result)
        return 0
    except BaseException as error:
        _write_new(
            args.work_root / "failure.json",
            {
                "schema_version": "sloforge.branchfabric.kill-recompute-worker-failure/v1",
                "status": "failed",
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "_SchedulerReplayCounter",
    "_history_evidence",
    "_kill_control_interval_evidence",
    "_material_persistent_control_drift",
    "run_kill_recompute_gpu0",
    "run_kill_recompute_gpu1",
    "run_worker",
]
