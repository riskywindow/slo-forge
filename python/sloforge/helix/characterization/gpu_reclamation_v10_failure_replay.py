"""Deterministic offline replay of the Experiment 004 v10/v6 serving failure.

This module is deliberately independent from the live Modal driver.  It turns
the immutable v6 artifacts into reference invariants that the live controller
must satisfy before another paid attempt is scientifically justified.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

NS_PER_SECOND: Final = 1_000_000_000
AUDIT_SCHEMA: Final = "sloforge.branchfabric.experiment-004-v10-v6-serving-replay/v1"
DELTA_SCHEMA: Final = "sloforge.branchfabric.v10-gpu1-telemetry-delta/v1"
OFFER_STATE_SCHEMA: Final = "sloforge.branchfabric.v10-global-offer-state/v1"
PARTIAL_SCHEMA: Final = "sloforge.branchfabric.v10-gpu0-partial-failure/v1"
_VLLM_RATE = re.compile(
    r"Avg generation throughput: (?P<tokens>[0-9.]+) tokens/s, "
    r"Running: (?P<running>[0-9]+) reqs, Waiting: (?P<waiting>[0-9]+) reqs"
)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected one JSON object in {path}")
    return value


def queue_depths_at(requests: Iterable[Mapping[str, Any]], *, timestamp_ns: int) -> dict[str, int]:
    """Separate queue backlog from admitted work that is already in service."""

    if timestamp_ns < 0:
        raise ValueError("queue timestamp cannot be negative")
    total_outstanding = 0
    waiting_backlog = 0
    active_in_service = 0
    for row in requests:
        scheduled = int(row["scheduled_arrival_ns"])
        admitted_value = row.get("admitted_ns")
        completed_value = row.get("completed_ns")
        admitted = None if admitted_value is None else int(admitted_value)
        completed = None if completed_value is None else int(completed_value)
        if scheduled > timestamp_ns or (completed is not None and completed <= timestamp_ns):
            continue
        total_outstanding += 1
        if admitted is None or admitted > timestamp_ns:
            waiting_backlog += 1
        else:
            active_in_service += 1
    if total_outstanding != waiting_backlog + active_in_service:
        raise AssertionError("queue partition does not conserve outstanding requests")
    return {
        "total_outstanding": total_outstanding,
        "waiting_backlog": waiting_backlog,
        "active_in_service": active_in_service,
    }


def compact_cumulative_snapshots(paths: Sequence[Path]) -> tuple[dict[str, Any], ...]:
    """Convert the historical cumulative v6 files into reference compact deltas.

    This conversion is an offline audit operation.  The resulting delta stream
    defines the live-path contract: a consumer sees each completed request once
    and never needs to reopen or rescan historical snapshots.
    """

    previous: tuple[dict[str, Any], ...] = ()
    prior_watermark = -1
    deltas: list[dict[str, Any]] = []
    for expected_sequence, path in enumerate(paths):
        payload = _read_object(path)
        if payload.get("schema_version") != "sloforge.branchfabric.v10-gpu1-telemetry/v1":
            raise ValueError("v6 GPU1 telemetry schema is unexpected")
        if payload.get("sequence") != expected_sequence:
            raise ValueError("v6 GPU1 telemetry sequence is non-contiguous")
        watermark = int(payload["observed_ns"])
        if watermark <= prior_watermark:
            raise ValueError("v6 GPU1 telemetry watermark is not monotonic")
        current_value = payload.get("requests")
        if not isinstance(current_value, list) or not all(
            isinstance(row, dict) for row in current_value
        ):
            raise ValueError("v6 GPU1 telemetry requests are malformed")
        current = tuple(dict(row) for row in current_value)
        if len(current) < len(previous) or current[: len(previous)] != previous:
            raise ValueError("v6 cumulative GPU1 telemetry is not an immutable prefix")
        delta = current[len(previous) :]
        deltas.append(
            {
                "schema_version": DELTA_SCHEMA,
                "sequence": expected_sequence,
                "observed_ns": watermark,
                "requests": delta,
            }
        )
        previous = current
        prior_watermark = watermark
    return tuple(deltas)


@dataclass
class IncrementalTelemetryCursor:
    """O(1)-state consumer for compact request-completion deltas."""

    next_snapshot_sequence: int = 0
    watermark_ns: int = -1
    last_request_sequence: int = -1
    completed_request_count: int = 0
    inspected_request_rows: int = 0
    maximum_delta_rows: int = 0

    def consume(self, payload: Mapping[str, Any]) -> None:
        if payload.get("schema_version") != DELTA_SCHEMA:
            raise ValueError("GPU1 telemetry delta schema is invalid")
        if payload.get("sequence") != self.next_snapshot_sequence:
            raise ValueError("GPU1 telemetry delta sequence is non-contiguous")
        watermark = int(payload["observed_ns"])
        if watermark <= self.watermark_ns:
            raise ValueError("GPU1 telemetry delta watermark is not monotonic")
        rows = payload.get("requests")
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise ValueError("GPU1 telemetry delta rows are malformed")
        prior = self.last_request_sequence
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("GPU1 telemetry delta contains a non-object row")
            sequence = int(row["sequence"])
            if sequence <= prior:
                raise ValueError("GPU1 completion sequence was replayed or reordered")
            prior = sequence
        self.next_snapshot_sequence += 1
        self.watermark_ns = watermark
        self.last_request_sequence = prior
        self.completed_request_count += len(rows)
        self.inspected_request_rows += len(rows)
        self.maximum_delta_rows = max(self.maximum_delta_rows, len(rows))


def enforce_total_outstanding_bound(*, total_outstanding: int, maximum: int) -> None:
    """Keep the scientific hard bound active in every serving phase."""

    if total_outstanding < 0 or maximum < 1:
        raise ValueError("outstanding-bound inputs are invalid")
    if total_outstanding > maximum:
        raise RuntimeError("v10 total outstanding work exceeded its hard bound")


@dataclass(frozen=True)
class GlobalOfferState:
    observed_ns: int
    last_offered_sequence: int
    next_sequence: int
    stopped: bool
    error: str | None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> GlobalOfferState:
        if value.get("schema_version") != OFFER_STATE_SCHEMA:
            raise ValueError("global offer-state schema is invalid")
        state = cls(
            observed_ns=int(value["observed_ns"]),
            last_offered_sequence=int(value["last_offered_sequence"]),
            next_sequence=int(value["next_sequence"]),
            stopped=value["stopped"],
            error=value.get("error"),
        )
        if state.observed_ns < 0 or state.last_offered_sequence < -1:
            raise ValueError("global offer-state counters are invalid")
        if state.next_sequence != state.last_offered_sequence + 1:
            raise ValueError("global offer-state sequence watermark is inconsistent")
        if not isinstance(state.stopped, bool):
            raise ValueError("global offer-state stop flag is not Boolean")
        if state.error is not None and not isinstance(state.error, str):
            raise ValueError("global offer-state error is malformed")
        return state


def gpu1_admission_allowed(
    *, sequence: int, last_admitted_sequence: int, state: GlobalOfferState
) -> bool:
    """Admit only a newly and globally offered sequence while the clock is live."""

    if sequence < 0 or last_admitted_sequence < -1:
        raise ValueError("GPU1 admission sequence is invalid")
    return (
        not state.stopped
        and state.error is None
        and last_admitted_sequence < sequence <= state.last_offered_sequence
    )


def build_partial_failure_evidence(
    *,
    attempt_id: str,
    started_ns: int,
    failed_ns: int,
    offered_requests: Sequence[Mapping[str, Any]],
    gpu0_requests: Sequence[Mapping[str, Any]],
    queue_state: Mapping[str, int],
    offer_state: Mapping[str, Any],
    error: BaseException,
) -> dict[str, Any]:
    """Build the minimum durable evidence required when GPU0 exits by exception."""

    if not attempt_id or started_ns < 0 or failed_ns < started_ns:
        raise ValueError("partial failure interval is invalid")
    payload = {
        "schema_version": PARTIAL_SCHEMA,
        "attempt_id": attempt_id,
        "started_ns": started_ns,
        "failed_ns": failed_ns,
        "global_offered_requests": [dict(row) for row in offered_requests],
        "gpu0_requests": [dict(row) for row in gpu0_requests],
        "queue_state": dict(queue_state),
        "global_offer_state": dict(offer_state),
        "error": {"type": type(error).__name__, "message": str(error)},
    }
    payload["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return payload


def persist_partial_failure_evidence(path: Path, payload: Mapping[str, Any]) -> None:
    """Create a durable, content-addressed partial record without overwriting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(canonical_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _vllm_rate_rows(path: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _VLLM_RATE.search(line)
        if match is None:
            continue
        token_rate = float(match.group("tokens"))
        rows.append(
            {
                "token_rate_per_second": token_rate,
                "completion_equivalent_rate_per_second": token_rate / 64.0,
                "running_requests": int(match.group("running")),
                "waiting_requests": int(match.group("waiting")),
            }
        )
    return tuple(rows)


def _p95(values: Iterable[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate p95 of an empty cohort")
    position = (len(ordered) - 1) * 0.95
    low = math.floor(position)
    high = math.ceil(position)
    return (
        ordered[low]
        if low == high
        else ordered[low] * (high - position) + ordered[high] * (position - low)
    )


def audit_v6_failure(*, repository_root: Path, seed: int) -> dict[str, Any]:
    """Replay immutable v6 evidence and emit the retry acceptance contract."""

    if seed != 41:
        raise ValueError("the immutable v6 replay requires its recorded seed 41")
    relative_root = Path(
        "artifacts/branchfabric/gpu-validation/experiment-004/raw/modal/exp004-v10-naive-s41-v6"
    )
    root = repository_root / relative_root
    config_path = root / "effective-config.json"
    result_path = root / "serving/result.json"
    failure_path = root / "serving/failure.json"
    ack_path = root / "barriers/v10-serving-enable-ack.json"
    first_path = root / "barriers/v10-gpu1-first-useful.json"
    serving_log = root / "logs/serving.stdout.log"
    rollout_log = root / "logs/rollout.stdout.log"
    snapshot_paths = tuple(sorted((root / "barriers/v10-gpu1-telemetry").glob("*.json")))
    if len(snapshot_paths) != 96:
        raise ValueError("v6 GPU1 telemetry snapshot set is incomplete")

    config = _read_object(config_path)
    result = _read_object(result_path)
    failure = _read_object(failure_path)
    ack = _read_object(ack_path)
    first = _read_object(first_path)
    if config.get("seed") != seed or config.get("attempt_id") != "exp004-v10-naive-s41-v6":
        raise ValueError("v6 effective configuration identity is unexpected")
    if result.get("status") != "failed" or failure.get("type") != "RuntimeError":
        raise ValueError("v6 serving failure evidence is unexpected")

    cumulative = tuple(_read_object(path) for path in snapshot_paths)
    final_rows = tuple(dict(row) for row in cumulative[-1]["requests"])
    deltas = compact_cumulative_snapshots(snapshot_paths)
    cursor = IncrementalTelemetryCursor()
    for delta in deltas:
        cursor.consume(delta)

    checkpoint_ns = 250 * NS_PER_SECOND
    checkpoint_depths = queue_depths_at(final_rows, timestamp_ns=checkpoint_ns)
    recovery_threshold = int(config["serving_recovery_queue_threshold"])
    serving_rates = _vllm_rate_rows(serving_log)
    rollout_rates = _vllm_rate_rows(rollout_log)
    final_gpu0 = serving_rates[-1]
    configured_bound = int(config["serving_overload_queue_abort"])
    # In v6 the external producer queue was full at the same configured
    # 64-request bound while vLLM reported 16 running + 48 waiting.  Therefore
    # 128 is a conservative lower bound; the failed offer itself is excluded.
    minimum_gpu0_outstanding = (
        int(final_gpu0["running_requests"])
        + int(final_gpu0["waiting_requests"])
        + int(config["serving_maximum_pending_requests"])
    )

    spike_start_ns = int(ack["spike_start_ns"])
    spike_rate = float(ack["spike_rate_per_second"])
    cutover = int(ack["enable_cutover_sequence"])
    serving_failed_ns = int(result["ended_ns"])

    def scheduled(sequence: int) -> int:
        return spike_start_ns + math.floor(sequence * NS_PER_SECOND / spike_rate)

    last_scheduled_before_failure = cutover
    while scheduled(last_scheduled_before_failure + 1) < serving_failed_ns:
        last_scheduled_before_failure += 1
    invalid_gpu1_rows = tuple(
        row for row in final_rows if int(row["scheduled_arrival_ns"]) >= serving_failed_ns
    )
    gpu1_complete_start = min(int(row["completed_ns"]) for row in final_rows)
    gpu1_complete_end = max(int(row["completed_ns"]) for row in final_rows)
    gpu1_completion_rate = (len(final_rows) - 1) / (
        (gpu1_complete_end - gpu1_complete_start) / NS_PER_SECOND
    )
    steady_rows = tuple(
        row
        for row in final_rows
        if 240 * NS_PER_SECOND <= int(row["scheduled_arrival_ns"]) < 300 * NS_PER_SECOND
    )
    gpu1_p95_ttft_ms = _p95(
        (int(row["first_token_ns"]) - int(row["scheduled_arrival_ns"])) / 1_000_000
        for row in steady_rows
    )
    cumulative_rows = sum(len(payload["requests"]) for payload in cumulative)
    cumulative_bytes = sum(path.stat().st_size for path in snapshot_paths)
    final_bytes = snapshot_paths[-1].stat().st_size

    input_paths = (
        config_path,
        result_path,
        failure_path,
        ack_path,
        first_path,
        serving_log,
        rollout_log,
        *snapshot_paths,
    )
    provenance = [
        {
            "artifact_reference": path.relative_to(repository_root).as_posix(),
            "artifact_sha256": file_sha256(path),
        }
        for path in input_paths
    ]
    expected_behavior = {
        "recovery_threshold_metric": "waiting_backlog_only",
        "active_in_service_excluded_from_recovery_threshold": True,
        "telemetry_format": "compact_incremental_completion_deltas",
        "telemetry_consumer_complexity": "O(new_rows)_time_O(1)_cursor_state_per_update",
        "hard_bound_metric": "all_globally_offered_not_completed_requests",
        "hard_bound_active_phases": [
            "pre-reclaim",
            "preserve",
            "two-gpu-recovery",
            "restore-handoff",
        ],
        "gpu1_admission_requires": [
            "sequence_at_or_below_last_globally_offered_watermark",
            "global_producer_not_stopped",
            "global_producer_has_no_error",
        ],
        "gpu0_exception_requires_partial_artifact": True,
        "partial_artifact_required_fields": [
            "global_offered_requests",
            "gpu0_requests",
            "queue_state",
            "global_offer_state",
            "error",
        ],
    }
    return {
        "schema_version": AUDIT_SCHEMA,
        "attempt_id": config["attempt_id"],
        "seed": seed,
        "status": "IMPLEMENTATION_BUG_CONFIRMED",
        "lambda_2_regression_supported": False,
        "v6_observations": {
            "gpu1_completion_rate_per_second": gpu1_completion_rate,
            "gpu1_steady_p95_ttft_ms": gpu1_p95_ttft_ms,
            "gpu1_first_useful_ns": int(first["first_token_ns"]),
            "queue_metric_checkpoint_ns": checkpoint_ns,
            "queue_metric_checkpoint": checkpoint_depths,
            "configured_recovery_queue_threshold": recovery_threshold,
            "total_outstanding_threshold_pass": (
                checkpoint_depths["total_outstanding"] <= recovery_threshold
            ),
            "waiting_backlog_threshold_pass": (
                checkpoint_depths["waiting_backlog"] <= recovery_threshold
            ),
            "telemetry_snapshot_count": len(snapshot_paths),
            "telemetry_cumulative_rows_read_by_v6_pattern": cumulative_rows,
            "telemetry_unique_completion_rows": len(final_rows),
            "telemetry_row_rescan_amplification": cumulative_rows / len(final_rows),
            "telemetry_cumulative_bytes": cumulative_bytes,
            "telemetry_final_snapshot_bytes": final_bytes,
            "telemetry_byte_amplification": cumulative_bytes / final_bytes,
            "compact_replay_rows_inspected": cursor.inspected_request_rows,
            "compact_replay_maximum_delta_rows": cursor.maximum_delta_rows,
            "gpu0_final_vllm_state": final_gpu0,
            "gpu1_final_vllm_state": rollout_rates[-1],
            "minimum_gpu0_total_outstanding_at_failure": minimum_gpu0_outstanding,
            "configured_hard_outstanding_bound": configured_bound,
            "hard_outstanding_bound_pass": minimum_gpu0_outstanding <= configured_bound,
            "last_sequence_scheduled_before_gpu0_failure": last_scheduled_before_failure,
            "gpu1_rows_scheduled_after_gpu0_failure": len(invalid_gpu1_rows),
            "gpu1_post_failure_sequence_minimum": min(
                int(row["sequence"]) for row in invalid_gpu1_rows
            ),
            "gpu1_post_failure_sequence_maximum": max(
                int(row["sequence"]) for row in invalid_gpu1_rows
            ),
            "v6_partial_gpu0_evidence_present": "partial_evidence" in result,
        },
        "expected_retry_behavior": expected_behavior,
        "retry_recommendation": (
            "NO_RETRY_UNCHANGED; permit one unchanged-load integrated retry only after "
            "all expected_retry_behavior invariants pass deterministic offline tests"
        ),
        "provenance": provenance,
    }


__all__ = [
    "AUDIT_SCHEMA",
    "DELTA_SCHEMA",
    "OFFER_STATE_SCHEMA",
    "PARTIAL_SCHEMA",
    "GlobalOfferState",
    "IncrementalTelemetryCursor",
    "audit_v6_failure",
    "build_partial_failure_evidence",
    "compact_cumulative_snapshots",
    "enforce_total_outstanding_bound",
    "gpu1_admission_allowed",
    "persist_partial_failure_evidence",
    "queue_depths_at",
]
