"""Success-only publication for the frozen Experiment 004 v10 baseline.

This module is deliberately CPU-only.  It accepts only the canonical, locally
finalized scientific-validity artifact and derives reports, plots, and future
plans from hash-verifiable raw evidence.  It never runs or describes optimized
preservation, kill/recompute, Experiment 005, or hardware as completed work.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
import statistics
from collections import defaultdict
from collections.abc import Iterable
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel

NS_PER_SECOND = 1_000_000_000
P95_MINIMUM_SAMPLES = 20

PLOT_SPECS = (
    ("01-queue-depth-over-time", "Queue depth over time", "line"),
    ("02-p95-ttft-over-time", "p95 TTFT over time", "line"),
    ("03-offered-vs-completed-rate", "Offered vs completed request rate", "line"),
    ("04-gpu-hbm-over-time", "GPU0/GPU1 HBM over time", "line"),
    ("05-reclamation-waterfall", "Reclamation waterfall", "waterfall"),
    ("06-restore-waterfall", "Restore waterfall", "waterfall"),
    ("07-gpu0-throughput-during-restore", "GPU0 throughput during restore phases", "bar"),
    ("08-logical-vs-physical-bytes", "Logical state bytes vs physical bytes touched", "bar"),
    ("09-state-pass-graph", "Measured v10 state-pass graph", "graph"),
    (
        "10-state-movement-amplification-decomposition",
        "State movement amplification decomposition",
        "bar",
    ),
)

REQUIRED_VALIDITY_GATES = (
    "authorization_alid",
    "phase_budget_valid",
    "two_engine_ready",
    "sanity_12rps_stable",
    "sanity_15rps_overload",
    "control_stable",
    "gpu0_overload_pass",
    "bounded_backlog_pass",
    "state_correctness_pass",
    "gpu1_state_release_pass",
    "gpu1_hbm_reclaim_pass",
    "gpu1_useful_capacity_pass",
    "two_gpu_service_gt_offered_pass",
    "queue_drain_pass",
    "slo_restoration_pass",
    "slo_stability_pass",
    "restore_load_reduced_pass",
    "gpu1_serving_drained_pass",
    "gpu0_active_during_restore_pass",
    "branch_resume_pass",
    "movement_accounting_pass",
    "budget_pass",
    "cleanup_pass",
)


def _canonical_bytes(value: Any) -> bytes:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"v10 publication input is not an object: {path}")
    return value


def _raw_ref(path: Path, root: Path, selector: str = "$") -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_file():
        raise ValueError(f"v10 publication source is not a regular file: {path}")
    try:
        reference = resolved.relative_to(root.resolve(strict=True)).as_posix()
    except ValueError:
        reference = str(resolved)
    return {
        "artifact_reference": reference,
        "artifact_sha256": _sha256(resolved),
        "sample_selector": selector,
    }


def _verify_prerequisite_references(validity: dict[str, Any]) -> None:
    prerequisites = validity.get("prerequisites")
    references = (
        prerequisites.get("evidence_references") if isinstance(prerequisites, dict) else None
    )
    if not isinstance(references, list) or len(references) != 7:
        raise ValueError("scientific validity lacks seven immutable prerequisite references")
    for reference in references:
        if not isinstance(reference, str):
            raise ValueError("scientific prerequisite reference is not a string")
        parts = reference.split(":", 2)
        if len(parts) != 3:
            raise ValueError("scientific prerequisite reference is malformed")
        _kind, expected, path_text = parts
        path = Path(path_text)
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise ValueError("scientific prerequisite reference is not an immutable file")
        if _sha256(path) != expected:
            raise ValueError("scientific prerequisite artifact changed after finalization")


def _strict_validity(payload: dict[str, Any]) -> dict[str, Any]:
    from sloforge.helix.characterization.gpu_reclamation_v10_validity import (
        V10ScientificValidity,
    )

    # Strict JSON validation preserves numeric/boolean strictness while allowing
    # JSON arrays to satisfy immutable tuple fields in the canonical schema.
    model = V10ScientificValidity.model_validate_json(_canonical_bytes(payload), strict=True)
    return model.model_dump(mode="json")


def _require_success(path: Path) -> dict[str, Any]:
    raw = _load_object(path)
    validity = _strict_validity(raw)
    if validity.get("scientifically_valid") is not True:
        raise ValueError("v10 success publication refuses non-scientifically-valid evidence")
    if any(validity.get(field) is not True for field in REQUIRED_VALIDITY_GATES):
        raise ValueError("v10 canonical validity is true but a required publication gate is not")
    _verify_prerequisite_references(validity)
    return validity


def _percentile(values: Iterable[float], fraction: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _timeline(validity: dict[str, Any]) -> dict[str, int]:
    raw = validity.get("timeline")
    events = raw.get("events") if isinstance(raw, dict) else None
    if not isinstance(events, list) or not events:
        raise ValueError("scientifically valid v10 lacks its causal timeline")
    result = {str(row["phase"]): int(row["monotonic_timestamp_ns"]) for row in events}
    if len(result) != len(events):
        raise ValueError("v10 timeline contains duplicate phases")
    return result


def _request_rows(
    serving: dict[str, Any], rollout: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    offered = serving.get("global_offered_requests")
    temporary = rollout.get("temporary_serving")
    observed = list(serving.get("requests", ()))
    if isinstance(temporary, dict):
        observed.extend(temporary.get("requests", ()))
    if not isinstance(offered, list) or not offered or not observed:
        raise ValueError("v10 publication lacks global offered/observed serving requests")
    by_id = {str(row["request_id"]): row for row in observed}
    if len(by_id) != len(observed) or {str(row["request_id"]) for row in offered} != set(by_id):
        raise ValueError("v10 plot inputs do not preserve the offered/observed request bijection")
    joined: list[dict[str, Any]] = []
    for row in offered:
        request_id = str(row["request_id"])
        observed_row = by_id[request_id]
        token_times = observed_row.get("token_timestamps_ns")
        if not isinstance(token_times, list) or len(token_times) != 64:
            raise ValueError("v10 plot input request does not contain exactly 64 output tokens")
        joined.append(
            {
                "request_id": request_id,
                "phase": str(row["phase"]),
                "device": str(row["device"]),
                "arrival_ns": int(row["scheduled_arrival_ns"]),
                "offered_ns": int(row["offered_ns"]),
                "service_start_ns": int(observed_row["service_start_ns"]),
                "first_token_ns": int(token_times[0]),
                "token_timestamps_ns": tuple(int(value) for value in token_times),
                "completion_ns": int(observed_row["completed_ns"]),
                "token_count": len(token_times),
            }
        )
    return joined, offered


def _time_bins(rows: list[dict[str, Any]]) -> tuple[int, list[int]]:
    start = min(row["arrival_ns"] for row in rows)
    end = max(row["completion_ns"] for row in rows)
    return start, list(range(start, end + NS_PER_SECOND, NS_PER_SECOND))


def _annotations(timeline: dict[str, int], start_ns: int) -> list[dict[str, Any]]:
    names = (
        ("CONTROL_STABLE", "control"),
        ("LOAD_SPIKE_BEGIN", "spike"),
        ("RECLAIM_TRIGGER", "reclaim"),
        ("GPU1_SERVING_ENABLE", "GPU1 joins"),
        ("SERVING_QUEUE_DRAIN_BEGIN", "queue drain"),
        ("SERVING_SLO_RESTORED", "SLO recovery"),
        ("RESTORE_TRIGGER", "restore"),
    )
    return [
        {"label": label, "time_seconds": (timeline[phase] - start_ns) / NS_PER_SECOND}
        for phase, label in names
    ]


def _queue_plot(rows: list[dict[str, Any]], timeline: dict[str, int]) -> dict[str, Any]:
    start, bins = _time_bins(rows)
    return {
        "x_unit": "seconds_from_first_arrival",
        "y_unit": "requests",
        "series": [
            {
                "name": "waiting_queue_depth",
                "points": [
                    {
                        "x": (timestamp - start) / NS_PER_SECOND,
                        "y": sum(
                            row["arrival_ns"] <= timestamp < row["service_start_ns"] for row in rows
                        ),
                    }
                    for timestamp in bins
                ],
            },
            {
                "name": "in_system_requests",
                "points": [
                    {
                        "x": (timestamp - start) / NS_PER_SECOND,
                        "y": sum(
                            row["arrival_ns"] <= timestamp < row["completion_ns"] for row in rows
                        ),
                    }
                    for timestamp in bins
                ],
            },
        ],
        "queue_depth_semantics": "arrived and not service-started",
        "in_system_semantics": "arrived and not completed; matches recovery trend evidence",
        "annotations": _annotations(timeline, start),
    }


def _ttft_plot(rows: list[dict[str, Any]], timeline: dict[str, int]) -> dict[str, Any]:
    start = min(row["arrival_ns"] for row in rows)
    end = max(row["completion_ns"] for row in rows)
    window_ns = 2 * NS_PER_SECOND
    bins = range(start, end + window_ns, window_ns)
    points = []
    for timestamp in bins:
        values = [
            (row["first_token_ns"] - row["arrival_ns"]) / NS_PER_SECOND
            for row in rows
            if timestamp <= row["arrival_ns"] < timestamp + window_ns
        ]
        points.append(
            {
                "x": (timestamp - start) / NS_PER_SECOND,
                "y": _percentile(values, 0.95) if len(values) >= P95_MINIMUM_SAMPLES else None,
                "sample_count": len(values),
                "percentile_reportable": len(values) >= P95_MINIMUM_SAMPLES,
            }
        )
    return {
        "x_unit": "seconds_from_first_arrival",
        "y_unit": "seconds",
        "series": [{"name": "p95_ttft", "points": points}],
        "reference_lines": [{"label": "SLO", "y": 2.0}],
        "minimum_samples_per_p95": P95_MINIMUM_SAMPLES,
        "window_seconds": window_ns / NS_PER_SECOND,
        "annotations": _annotations(timeline, start),
    }


def _rate_plot(rows: list[dict[str, Any]], timeline: dict[str, int]) -> dict[str, Any]:
    start, bins = _time_bins(rows)
    series = []
    for name, field in (("offered", "arrival_ns"), ("completed", "completion_ns")):
        series.append(
            {
                "name": name,
                "points": [
                    {
                        "x": (timestamp - start) / NS_PER_SECOND,
                        "y": sum(
                            timestamp <= row[field] < timestamp + NS_PER_SECOND for row in rows
                        ),
                    }
                    for timestamp in bins
                ],
            }
        )
    return {
        "x_unit": "seconds_from_first_arrival",
        "y_unit": "requests_per_second",
        "series": series,
        "annotations": _annotations(timeline, start),
    }


def _hbm_plot(telemetry: tuple[tuple[str, dict[str, Any]], ...], origin_ns: int) -> dict[str, Any]:
    series = []
    for role, payload in telemetry:
        sampling = payload.get("sampling")
        samples = sampling.get("samples") if isinstance(sampling, dict) else None
        if not isinstance(samples, list) or not samples:
            raise ValueError(f"v10 {role} HBM telemetry has no samples")
        points = []
        for sample in samples:
            gpu_samples = sample.get("gpu_samples")
            if not isinstance(gpu_samples, list) or len(gpu_samples) != 1:
                raise ValueError("v10 HBM sample is not bound to exactly one GPU")
            points.append(
                {
                    "x": (int(sample["sample_trigger_monotonic_ns"]) - origin_ns) / NS_PER_SECOND,
                    "y": int(gpu_samples[0]["memory_used_bytes"]),
                }
            )
        series.append({"name": role, "points": points})
    return {
        "x_unit": "seconds_from_first_arrival",
        "y_unit": "bytes",
        "series": series,
    }


def _waterfall(
    timeline: dict[str, int], stages: tuple[tuple[str, str, str], ...]
) -> dict[str, Any]:
    origin = min(timeline[start] for _label, start, _end in stages)
    rows = []
    for label, start, end in stages:
        if timeline[end] < timeline[start]:
            raise ValueError(f"v10 waterfall stage ends before it starts: {label}")
        rows.append(
            {
                "label": label,
                "start_seconds": (timeline[start] - origin) / NS_PER_SECOND,
                "duration_seconds": (timeline[end] - timeline[start]) / NS_PER_SECOND,
            }
        )
    return {"x_unit": "seconds", "stages": rows}


def _restore_activity(validity: dict[str, Any]) -> dict[str, Any]:
    activity = validity.get("gpu0_restore_activity")
    stages = activity.get("stage_activity") if isinstance(activity, dict) else None
    if not isinstance(stages, list) or not stages:
        raise ValueError("v10 validity lacks GPU0 restore-stage activity")
    bars = []
    for stage in stages:
        interval = stage["interval"]
        duration = (int(interval["end_ns"]) - int(interval["start_ns"])) / NS_PER_SECOND
        if duration <= 0:
            raise ValueError("v10 restore interference stage has nonpositive duration")
        bars.append(
            {
                "label": str(stage["stage"]),
                "requests_per_second": int(stage["completion_count"]) / duration,
                "tokens_per_second": int(stage["emitted_tokens"]) / duration,
                "completion_count": int(stage["completion_count"]),
                "ttft_sample_count": int(stage["ttft_sample_count"]),
                "eligible": bool(stage["eligible_for_interference"]),
                "sufficient_sample": bool(stage["sufficient_sample"]),
            }
        )
    return {"x_unit": "restore_stage", "y_unit": "rate_per_second", "bars": bars}


def _phase_interference(
    *,
    rows: list[dict[str, Any]],
    validity: dict[str, Any],
    timeline: dict[str, int],
    serving_telemetry: dict[str, Any],
) -> dict[str, Any]:
    control = validity["gpu0_overload"]["control"]["interval"]
    intervals = (
        ("CONTROL", int(control["start_ns"]), int(control["end_ns"])),
        ("GPU1_PRESERVE_D2H", timeline["D2H_BEGIN"], timeline["D2H_END"]),
        (
            "GPU1_PRESERVE_VALIDATION",
            timeline["INTEGRITY_BEGIN"],
            timeline["INTEGRITY_END"],
        ),
        ("GPU1_RESTORE_H2D", timeline["H2D_BEGIN"], timeline["H2D_END"]),
        ("GPU1_IMPORT", timeline["STATE_IMPORT_BEGIN"], timeline["STATE_IMPORT_END"]),
        (
            "GPU1_DESTINATION_VALIDATION",
            timeline["STATE_VALIDATE_BEGIN"],
            timeline["STATE_VALIDATE_END"],
        ),
    )
    telemetry_samples = serving_telemetry.get("sampling", {}).get("samples", ())
    phases = []
    for name, start, end in intervals:
        if end <= start:
            raise ValueError(f"v10 interference interval is not positive: {name}")
        duration = (end - start) / NS_PER_SECOND
        completions = [
            row for row in rows if start <= row["completion_ns"] < end and row["device"] == "gpu0"
        ]
        arrivals = [
            row for row in rows if start <= row["arrival_ns"] < end and row["device"] == "gpu0"
        ]
        ttft = [(row["first_token_ns"] - row["arrival_ns"]) / NS_PER_SECOND for row in arrivals]
        token_times = [
            timestamp
            for row in rows
            if row["device"] == "gpu0"
            for timestamp in row["token_timestamps_ns"]
            if start <= timestamp < end
        ]
        token_latencies = [
            (right - left) / NS_PER_SECOND
            for row in arrivals
            for left, right in zip(
                row["token_timestamps_ns"], row["token_timestamps_ns"][1:], strict=False
            )
        ]
        samples = [
            sample
            for sample in telemetry_samples
            if start <= int(sample["sample_trigger_monotonic_ns"]) < end
        ]
        pcie_rx = [
            int(sample["gpu_samples"][0]["pcie_rx_bytes_per_second"])
            for sample in samples
            if sample["gpu_samples"][0].get("pcie_rx_bytes_per_second") is not None
        ]
        pcie_tx = [
            int(sample["gpu_samples"][0]["pcie_tx_bytes_per_second"])
            for sample in samples
            if sample["gpu_samples"][0].get("pcie_tx_bytes_per_second") is not None
        ]
        gpu_utilization = [
            int(sample["gpu_samples"][0]["gpu_utilization_percent"])
            for sample in samples
            if sample["gpu_samples"][0].get("gpu_utilization_percent") is not None
        ]
        process_rss = [
            int(process["rss_bytes"])
            for sample in samples
            for process in sample["host_sample"].get("processes", ())
            if process.get("rss_bytes") is not None
        ]
        host_pressure = [
            int(sample["host_sample"]["host_memory_total_bytes"])
            - int(sample["host_sample"]["host_memory_available_bytes"])
            for sample in samples
        ]
        cpu_percent = None
        if len(samples) >= 2:
            first, last = samples[0]["host_sample"], samples[-1]["host_sample"]
            busy = (
                int(last["system_cpu_user_ns"])
                + int(last["system_cpu_system_ns"])
                - int(first["system_cpu_user_ns"])
                - int(first["system_cpu_system_ns"])
            )
            idle = int(last["system_cpu_idle_ns"]) - int(first["system_cpu_idle_ns"])
            cpu_percent = 100.0 * busy / (busy + idle) if busy + idle > 0 else None
        throughput_reportable = len(completions) >= 2
        token_rate_reportable = len(token_times) >= 64
        p50_ttft_reportable = len(ttft) >= 2
        p95_ttft_reportable = len(ttft) >= P95_MINIMUM_SAMPLES
        phases.append(
            {
                "phase": name,
                "start_ns": start,
                "end_ns": end,
                "completed_requests": len(completions),
                "requests_per_second": len(completions) / duration,
                "emitted_tokens": len(token_times),
                "tokens_per_second": len(token_times) / duration,
                "ttft_sample_count": len(ttft),
                "p50_ttft_seconds": _percentile(ttft, 0.50),
                "p95_ttft_seconds": (_percentile(ttft, 0.95) if p95_ttft_reportable else None),
                "throughput_interference_reportable": throughput_reportable,
                "token_rate_interference_reportable": token_rate_reportable,
                "p50_ttft_reportable": p50_ttft_reportable,
                "p95_ttft_reportable": p95_ttft_reportable,
                "p50_inter_token_latency_seconds": _percentile(token_latencies, 0.50),
                "resource_sample_count": len(samples),
                "cpu_utilization_percent": cpu_percent,
                "maximum_host_memory_used_bytes": max(host_pressure, default=None),
                "maximum_process_rss_bytes": max(process_rss, default=None),
                "mean_gpu0_utilization_percent": (
                    statistics.fmean(gpu_utilization) if gpu_utilization else None
                ),
                "mean_pcie_rx_bytes_per_second": statistics.fmean(pcie_rx) if pcie_rx else None,
                "mean_pcie_tx_bytes_per_second": statistics.fmean(pcie_tx) if pcie_tx else None,
            }
        )
    baseline = phases[0]
    baseline_throughput = float(cast(float, baseline["requests_per_second"]))
    baseline_p50 = baseline["p50_ttft_seconds"]
    baseline_p95 = baseline["p95_ttft_seconds"]
    baseline_token_rate = float(cast(float, baseline["tokens_per_second"]))
    for phase in phases:
        phase_throughput = float(cast(float, phase["requests_per_second"]))
        phase_token_rate = float(cast(float, phase["tokens_per_second"]))
        phase_p50 = phase["p50_ttft_seconds"]
        phase_p95 = phase["p95_ttft_seconds"]
        phase["throughput_interference"] = (
            1 - phase_throughput / baseline_throughput
            if baseline_throughput > 0
            and baseline["throughput_interference_reportable"]
            and phase["throughput_interference_reportable"]
            else None
        )
        phase["token_rate_interference"] = (
            1 - phase_token_rate / baseline_token_rate
            if baseline_token_rate > 0
            and baseline["token_rate_interference_reportable"]
            and phase["token_rate_interference_reportable"]
            else None
        )
        phase["p50_ttft_increase"] = (
            cast(float, phase_p50) / cast(float, baseline_p50) - 1
            if baseline_p50 not in (None, 0)
            and phase_p50 is not None
            and baseline["p50_ttft_reportable"]
            and phase["p50_ttft_reportable"]
            else None
        )
        phase["p95_ttft_increase"] = (
            cast(float, phase_p95) / cast(float, baseline_p95) - 1
            if baseline_p95 not in (None, 0)
            and phase_p95 is not None
            and baseline["p95_ttft_reportable"]
            and phase["p95_ttft_reportable"]
            else None
        )
    return {
        "minimum_p95_samples": P95_MINIMUM_SAMPLES,
        "interpretation": "descriptive_association_only",
        "causal_interference_claimed": False,
        "phases": phases,
    }


def _state_pass_graph(movement: dict[str, Any]) -> dict[str, Any]:
    report = movement.get("raw_state_movement_report")
    passes = report.get("passes") if isinstance(report, dict) else None
    if not isinstance(passes, list) or not passes:
        raise ValueError("v10 movement artifact lacks measured state passes")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in passes:
        record_id = str(row["record_id"])
        fields = record_id.split(":", 2)
        label = fields[1] if len(fields) == 3 else record_id
        grouped[label].append(row)
    nodes: list[dict[str, Any]] = []
    for label, rows in grouped.items():
        nodes.append(
            {
                "id": label,
                "operation": str(rows[0]["operation"]),
                "source_memory": str(rows[0]["source_memory"]),
                "destination_memory": str(rows[0]["destination_memory"]),
                "logical_segments": sorted({str(row["state_segment"]) for row in rows}),
                "bytes_read": sum(int(row.get("bytes_read", 0)) for row in rows),
                "bytes_written": sum(int(row.get("bytes_written", 0)) for row in rows),
                "temporary_bytes": sum(
                    int(row.get("temporary_allocation_bytes", 0)) for row in rows
                ),
                "start_ns": min(int(row["start_ns"]) for row in rows),
                "end_ns": max(int(row["end_ns"]) for row in rows),
                "required_unavoidable": all(bool(row.get("required_unavoidable")) for row in rows),
            }
        )
    nodes.sort(key=lambda row: (row["start_ns"], row["end_ns"], row["id"]))
    return {
        "nodes": nodes,
        "edges": [
            {
                "source": before["id"],
                "destination": after["id"],
                "kind": "measured_serial_dependency",
                "bytes": after["bytes_read"] + after["bytes_written"],
            }
            for before, after in pairwise(nodes)
        ],
        "node_count": len(nodes),
        "timing_semantics": "one interval per measured operation; shared segment rows deduplicated",
    }


def _pass_breakdown(movement: dict[str, Any]) -> dict[str, Any]:
    report = movement.get("raw_state_movement_report")
    passes = report.get("passes") if isinstance(report, dict) else None
    if not isinstance(passes, list) or not passes:
        raise ValueError("v10 movement artifact lacks state pass records")
    aggregate: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"bytes": 0, "record_count": 0, "intervals": set()}
    )
    for row in passes:
        item = aggregate[str(row["operation"])]
        item["bytes"] += (
            int(row.get("bytes_read", 0))
            + int(row.get("bytes_written", 0))
            + int(row.get("transfer_bytes", 0))
        )
        item["intervals"].add((int(row["start_ns"]), int(row["end_ns"])))
        item["record_count"] += 1
    bars = []
    for name, values in sorted(aggregate.items()):
        intervals = values.pop("intervals")
        values["wall_ns"] = sum(end - start for start, end in intervals)
        values["operation_instance_count"] = len(intervals)
        bars.append({"label": name, **values})
    return {
        "x_unit": "state_pass_operation",
        "y_unit": "bytes_touched_or_moved",
        "wall_time_semantics": (
            "shared underlying operation intervals are deduplicated across logical segments"
        ),
        "bars": bars,
    }


def _amplification_decomposition(accounting: dict[str, Any]) -> dict[str, Any]:
    logical = int(accounting["logical_state_bytes"])
    if logical <= 0:
        raise ValueError("v10 amplification decomposition requires positive logical state")
    surfaces = (
        ("external_movement", int(accounting["external_movement_bytes"]), False),
        ("avoidable_lower_bound", int(accounting["avoidable_movement_bytes"]), True),
        ("critical_path", int(accounting["critical_path_movement_bytes"]), False),
        ("full_physical_touch", int(accounting["full_physical_touch_bytes"]), False),
    )
    return {
        "x_unit": "movement_surface",
        "y_unit": "amplification_x_logical_state",
        "logical_state_bytes": logical,
        "bars": [
            {
                "label": label,
                "value": byte_count / logical,
                "bytes": byte_count,
                "conservative_lower_bound": lower_bound,
            }
            for label, byte_count, lower_bound in surfaces
        ],
    }


def _plot_payloads(
    *,
    validity: dict[str, Any],
    serving: dict[str, Any],
    rollout: dict[str, Any],
    movement: dict[str, Any],
    telemetry: tuple[tuple[str, dict[str, Any]], ...],
    provenance: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    timeline = _timeline(validity)
    rows, _offered = _request_rows(serving, rollout)
    origin = min(row["arrival_ns"] for row in rows)
    accounting = validity.get("movement_accounting")
    if not isinstance(accounting, dict):
        raise ValueError("v10 validity lacks movement accounting")
    data = (
        _queue_plot(rows, timeline),
        _ttft_plot(rows, timeline),
        _rate_plot(rows, timeline),
        _hbm_plot(telemetry, origin),
        _waterfall(
            timeline,
            (
                ("quiesce", "BRANCH_QUIESCE_BEGIN", "BRANCH_QUIESCE_END"),
                ("capture", "STATE_CAPTURE_BEGIN", "STATE_CAPTURE_END"),
                ("transform", "STATE_TRANSFORM_BEGIN", "STATE_TRANSFORM_END"),
                ("D2H", "D2H_BEGIN", "D2H_END"),
                ("integrity", "INTEGRITY_BEGIN", "INTEGRITY_END"),
                ("release", "GPU1_STATE_RELEASE_BEGIN", "GPU1_HBM_RECLAIM_CONFIRMED"),
            ),
        ),
        _waterfall(
            timeline,
            (
                ("H2D", "H2D_BEGIN", "H2D_END"),
                ("import", "STATE_IMPORT_BEGIN", "STATE_IMPORT_END"),
                (
                    "native write",
                    "DESTINATION_NATIVE_WRITE_BEGIN",
                    "DESTINATION_NATIVE_WRITE_END",
                ),
                ("validation", "STATE_VALIDATE_BEGIN", "STATE_VALIDATE_END"),
                ("resume", "BRANCH_RESUME_BEGIN", "ALL_BRANCHES_RESUMED"),
            ),
        ),
        _restore_activity(validity),
        {
            "x_unit": "byte_surface",
            "y_unit": "bytes",
            "bars": [
                {"label": "logical_state", "value": int(accounting["logical_state_bytes"])},
                {
                    "label": "full_physical_touch",
                    "value": int(accounting["full_physical_touch_bytes"]),
                },
                {
                    "label": "external_movement",
                    "value": int(accounting["external_movement_bytes"]),
                },
                {
                    "label": "avoidable_movement",
                    "value": int(accounting["avoidable_movement_bytes"]),
                },
            ],
        },
        _state_pass_graph(movement),
        _amplification_decomposition(accounting),
    )
    return [
        {
            "schema_version": "sloforge.branchfabric.experiment-004-v10-plot/v1",
            "plot_id": plot_id,
            "title": title,
            "kind": kind,
            "scientifically_valid_v10_only": True,
            "data": plot_data,
            "raw_provenance": provenance,
        }
        for (plot_id, title, kind), plot_data in zip(PLOT_SPECS, data, strict=True)
    ]


def _svg(payload: dict[str, Any]) -> bytes:
    width, height = 960, 440
    title = html.escape(str(payload["title"]))
    kind = payload["kind"]
    data = payload["data"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{title}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="28" y="32" font-family="sans-serif" font-size="20" fill="#111">{title}</text>',
        '<rect x="70" y="55" width="850" height="330" fill="none" stroke="#777"/>',
    ]
    if kind == "line":
        series = data.get("series", [])
        points = [
            point
            for item in series
            for point in item.get("points", [])
            if point.get("y") is not None
        ]
        if not points:
            parts.append(
                '<text x="90" y="110" font-family="sans-serif" font-size="14" fill="#555">'
                "No temporal window had enough samples for a reportable percentile.</text>"
            )
            for reference in data.get("reference_lines", []):
                parts.append(
                    '<line x1="70" y1="220" x2="920" y2="220" '
                    'stroke="#9b1c31" stroke-dasharray="7 5"/>'
                )
                parts.append(
                    f'<text x="790" y="212" font-family="sans-serif" font-size="11" '
                    f'fill="#9b1c31">{html.escape(str(reference["label"]))}: '
                    f"{float(reference['y']):g} s</text>"
                )
            parts.append("</svg>\n")
            return "".join(parts).encode()
        xs = [float(point["x"]) for point in points]
        ys = [float(point["y"]) for point in points]
        ys.extend(float(line["y"]) for line in data.get("reference_lines", []))
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(0.0, min(ys)), max(ys)
        if xmax == xmin:
            xmax += 1.0
        if ymax == ymin:
            ymax += 1.0
        colors = ("#0057b8", "#d35400", "#258039")
        for index, item in enumerate(series):
            valid = [point for point in item.get("points", []) if point.get("y") is not None]
            coords = " ".join(
                f"{70 + 850 * (float(point['x']) - xmin) / (xmax - xmin):.2f},"
                f"{385 - 330 * (float(point['y']) - ymin) / (ymax - ymin):.2f}"
                for point in valid
            )
            parts.append(
                f'<polyline fill="none" stroke="{colors[index % len(colors)]}" '
                f'stroke-width="2" points="{coords}"/>'
            )
            parts.append(
                f'<text x="{80 + index * 190}" y="418" font-family="sans-serif" font-size="12" '
                f'fill="{colors[index % len(colors)]}">{html.escape(str(item["name"]))}</text>'
            )
        for reference in data.get("reference_lines", []):
            value = float(reference["y"])
            if ymin <= value <= ymax:
                y = 385 - 330 * (value - ymin) / (ymax - ymin)
                parts.append(
                    f'<line x1="70" y1="{y:.2f}" x2="920" y2="{y:.2f}" '
                    'stroke="#9b1c31" stroke-dasharray="7 5"/>'
                )
                parts.append(
                    f'<text x="825" y="{y - 5:.2f}" font-family="sans-serif" '
                    f'font-size="11" fill="#9b1c31">{html.escape(str(reference["label"]))}</text>'
                )
    elif kind in {"bar", "waterfall"}:
        bars = data.get("bars", data.get("stages", []))
        values = [
            float(
                row.get(
                    "value",
                    row.get("duration_seconds", row.get("bytes", row.get("tokens_per_second", 0))),
                )
            )
            for row in bars
        ]
        maximum = max(values, default=0.0)
        if maximum <= 0:
            maximum = 1.0
        count = max(1, len(bars))
        bar_width = max(4.0, 760 / count)
        for index, (row, value) in enumerate(zip(bars, values, strict=True)):
            x = 85 + index * (820 / count)
            bar_height = 280 * value / maximum
            parts.append(
                f'<rect x="{x:.2f}" y="{365 - bar_height:.2f}" width="{bar_width:.2f}" '
                f'height="{bar_height:.2f}" fill="#0057b8"/>'
            )
            label = html.escape(str(row.get("label", index)))
            parts.append(
                f'<text x="{x:.2f}" y="380" font-family="sans-serif" font-size="9" '
                f'transform="rotate(45 {x:.2f} 380)" fill="#111">{label}</text>'
            )
    else:
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])
        parts.append(
            f'<text x="90" y="95" font-family="sans-serif" font-size="14" fill="#111">'
            f"{len(nodes)} memory nodes · {len(edges)} aggregated measured edges</text>"
        )
        for index, edge in enumerate(edges[:12]):
            label = html.escape(f"{edge['source']} → {edge['destination']} · {edge['bytes']} B")
            parts.append(
                f'<text x="90" y="{125 + index * 20}" font-family="monospace" font-size="11" '
                f'fill="#111">{label}</text>'
            )
    parts.append("</svg>\n")
    return "".join(parts).encode()


def _duration_seconds(timeline: dict[str, int], start: str, end: str) -> float:
    return (timeline[end] - timeline[start]) / NS_PER_SECOND


def _report_payload(
    *,
    validity: dict[str, Any],
    config: dict[str, Any],
    movement: dict[str, Any],
    serving: dict[str, Any],
    rollout: dict[str, Any],
    serving_telemetry: dict[str, Any],
    ledger: dict[str, Any],
    plot_refs: list[dict[str, Any]],
    raw_provenance: list[dict[str, Any]],
) -> dict[str, Any]:
    timeline = _timeline(validity)
    control = validity["gpu0_overload"]["control"]
    overload = validity["gpu0_overload"]["overload"]
    capacity = validity["two_gpu_excess_capacity"]
    accounting = validity["movement_accounting"]
    state = validity["state_correctness"]
    branch = validity["branch_resume"]
    budget = validity["budget"]
    logical_bytes = int(accounting["logical_state_bytes"])
    if logical_bytes <= 0:
        raise ValueError("v10 movement amplification denominator must be positive")
    request_rows, _offered = _request_rows(serving, rollout)
    interference = _phase_interference(
        rows=request_rows,
        validity=validity,
        timeline=timeline,
        serving_telemetry=serving_telemetry,
    )
    cost = sum(
        float(row["accounted_wall_seconds"])
        * int(row["gpu_count"])
        / 3600.0
        * float(row.get("gpu_price_per_hour_usd") or 0.0)
        for row in ledger.get("intervals", [])
    )
    return {
        "schema_version": "sloforge.branchfabric.experiment-004-v10-report/v2",
        "status": "success",
        "scientifically_valid": True,
        "attempt_id": config["attempt_id"],
        "baseline": "NAIVE_PRESERVE_V10_FROZEN",
        "capacity": {
            "lambda_1_rps": config["lambda_1_rps"],
            "lambda_2_rps": config["lambda_2_rps"],
            "lambda_spike_rps": config["lambda_spike_rps"],
            "one_gpu_overload_margin": config["lambda_spike_rps"] / config["lambda_1_rps"] - 1,
            "two_gpu_reserve_margin": 1 - config["lambda_spike_rps"] / config["lambda_2_rps"],
        },
        "control": control,
        "overload": overload,
        "two_gpu_recovery": capacity,
        "queue_drain": validity["queue_drain"],
        "slo_stability": validity["slo_stability"],
        "state": {
            "logical_state_bytes": state["logical_state_bytes"],
            "preserved_state_bytes": state["preserved_state_bytes"],
            "released_source_blocks": state["source_blocks_released"],
        },
        "latencies_seconds": {
            "state_reclamation": _duration_seconds(
                timeline, "RECLAIM_TRIGGER", "GPU1_HBM_RECLAIM_CONFIRMED"
            ),
            "time_to_useful_reclaimed_capacity": _duration_seconds(
                timeline, "RECLAIM_TRIGGER", "GPU1_FIRST_USEFUL_SERVING_REQUEST"
            ),
            "time_to_two_gpu_excess_service": _duration_seconds(
                timeline, "RECLAIM_TRIGGER", "TWO_GPU_SERVICE_STABLE"
            ),
            "time_to_serving_slo_restoration": _duration_seconds(
                timeline, "RECLAIM_TRIGGER", "SERVING_SLO_RESTORED"
            ),
            "rollout_restore": _duration_seconds(
                timeline, "RESTORE_TRIGGER", "FIRST_RESUMED_TOKEN"
            ),
            "all_branches_restore": _duration_seconds(
                timeline, "RESTORE_TRIGGER", "ALL_BRANCHES_RESUMED"
            ),
        },
        "branch_correctness": {
            "exact_first_token_matches": sum(
                branch["expected_first_tokens"].get(key) == value
                for key, value in branch["observed_first_tokens"].items()
            ),
            "branch_count": len(branch["expected_first_tokens"]),
            "minimum_continuation_tokens": min(branch["continuation_token_counts"].values()),
        },
        "movement": {
            "logical_state_bytes": accounting["logical_state_bytes"],
            "full_physical_touch_bytes": accounting["full_physical_touch_bytes"],
            "full_physical_touch_amplification": (
                int(accounting["full_physical_touch_bytes"]) / logical_bytes
            ),
            "external_movement_amplification": (
                int(accounting["external_movement_bytes"]) / logical_bytes
            ),
            "avoidable_movement_amplification": (
                int(accounting["avoidable_movement_bytes"]) / logical_bytes
            ),
            "critical_path_movement_amplification": (
                int(accounting["critical_path_movement_bytes"]) / logical_bytes
            ),
            "duplicate_bytes": accounting["accounting_duplicate_bytes"],
            "dominant_naive_chain": (
                "native read → repack/unpage/stack/concatenate → D2H → integrity/publish "
                "verification → H2D → import verification → native write → destination "
                "readback/validation D2H"
            ),
        },
        "gpu0_restore_activity": validity["gpu0_restore_activity"],
        "serving_interference": interference,
        "budget": {
            **budget,
            "cumulative_gpu_hours": float(ledger["consumed_additional_gpu_seconds"]) / 3600.0,
            "cumulative_estimated_gpu_cost_usd": cost,
        },
        "cleanup": validity["cleanup"],
        "frozen_baseline_ready": True,
        "optimized_preservation_executed": False,
        "kill_and_recompute_executed": False,
        "fpga_justification_claimed": False,
        "plots": plot_refs,
        "raw_provenance": raw_provenance,
        "movement_source": movement["schema_version"],
    }


def _report_markdown(report: dict[str, Any]) -> bytes:
    capacity = report["capacity"]
    latency = report["latencies_seconds"]
    movement = report["movement"]
    branch = report["branch_correctness"]
    control = report["control"]
    overload = report["overload"]
    recovery = report["two_gpu_recovery"]
    budget = report["budget"]
    lines = [
        "# BranchFabric GPU Validation Experiment 004 v10",
        "",
        "Status: **SCIENTIFICALLY VALID NAIVE-PRESERVATION BASELINE**.",
        "",
        "This report freezes the existing naive state path. No optimized-preservation, "
        "kill/recompute, Experiment 005, RTL, HLS, simulator, or FPGA work was executed.",
        "",
        "## Calibrated serving envelope",
        "",
        f"- λ₁: `{capacity['lambda_1_rps']}` requests/s",
        f"- λ₂: `{capacity['lambda_2_rps']}` requests/s",
        f"- λ_spike: `{capacity['lambda_spike_rps']}` requests/s",
        f"- One-GPU overload margin: `{capacity['one_gpu_overload_margin']:.6f}`",
        f"- Two-GPU reserve margin: `{capacity['two_gpu_reserve_margin']:.6f}`",
        f"- One-GPU unstable point: `{capacity['lambda_spike_rps']}` requests/s",
        "",
        "## Control and bounded overload",
        "",
        f"- Control completions: `{control['completed_requests']}`",
        f"- Control p95 TTFT: `{control['ttft']['p95']}` ns",
        f"- Overload completions: `{overload['completed_requests']}`",
        f"- Overload p95 TTFT: `{overload['ttft']['p95']}` ns",
        "",
        "## Capacity reclamation",
        "",
        f"- State reclamation: `{latency['state_reclamation']:.6f}` s",
        f"- First useful GPU1 capacity: `{latency['time_to_useful_reclaimed_capacity']:.6f}` s",
        f"- Stable excess two-GPU service: `{latency['time_to_two_gpu_excess_service']:.6f}` s",
        f"- Stable serving SLO restoration: `{latency['time_to_serving_slo_restoration']:.6f}` s",
        f"- Two-GPU offered rate: `{recovery['offered_rate_per_second']}` requests/s",
        f"- Two-GPU completed rate: `{recovery['completed_rate_per_second']}` requests/s",
        "",
        "## Restore and correctness",
        "",
        f"- First resumed token: `{latency['rollout_restore']:.6f}` s after restore trigger",
        f"- All branches restored: `{latency['all_branches_restore']:.6f}` s",
        f"- Exact first-token matches: `{branch['exact_first_token_matches']}/{branch['branch_count']}`",
        f"- Minimum continuation length: `{branch['minimum_continuation_tokens']}` tokens",
        "",
        "## Frozen naive movement baseline",
        "",
        f"- Logical state: `{movement['logical_state_bytes']}` bytes",
        f"- Full physical touch: `{movement['full_physical_touch_bytes']}` bytes "
        f"(`{movement['full_physical_touch_amplification']:.6f}x`)",
        f"- External movement: `{movement['external_movement_amplification']:.6f}x`",
        f"- Avoidable movement: `{movement['avoidable_movement_amplification']:.6f}x`",
        f"- Critical-path movement: `{movement['critical_path_movement_amplification']:.6f}x`",
        f"- Dominant chain: {movement['dominant_naive_chain']}",
        "",
        "## Serving interference and budget",
        "",
        "GPU0 restore-phase measurements, sample eligibility, CPU/host pressure, and PCIe "
        "observations are recorded in the JSON report. Unsupported p95 values remain null.",
        "",
        f"- Cumulative Experiment 004 GPU use: `{budget['cumulative_gpu_hours']:.6f}` hours",
        f"- Cumulative estimated GPU cost: `${budget['cumulative_estimated_gpu_cost_usd']:.6f}`",
        "- Modal cleanup gate: `PASS`",
        "",
        "The frozen baseline is ready for the separately planned A/B/C campaign.",
        "",
    ]
    return "\n".join(lines).encode()


def _optimization_plan(report: dict[str, Any], movement: dict[str, Any]) -> bytes:
    nodes = _state_pass_graph(movement)["nodes"]

    def estimate(*needles: str) -> tuple[int, int, float]:
        selected = [
            row for row in nodes if any(needle in str(row["id"]).lower() for needle in needles)
        ]
        intervals = {(int(row["start_ns"]), int(row["end_ns"])) for row in selected}
        byte_count = sum(
            int(row["bytes_read"]) + int(row["bytes_written"]) + int(row["temporary_bytes"])
            for row in selected
        )
        wall_seconds = sum(end - start for start, end in intervals) / NS_PER_SECOND
        return len(selected), byte_count, wall_seconds

    candidates = (
        (
            1,
            "Fused capture transforms",
            ("capture-unpage", "capture-stack", "capture-concatenate"),
            "medium",
            "medium",
        ),
        (
            2,
            "Direct native-to-transport conversion",
            ("capture-native-axis", "capture-unpage", "capture-stack", "capture-concatenate"),
            "high",
            "high",
        ),
        (
            3,
            "Eliminate complete canonical intermediates",
            ("canonical", "concatenate", "stack"),
            "medium",
            "high",
        ),
        (4, "Pinned double buffering", ("capture-d2h", "restore-canonical-h2d"), "low", "medium"),
        (5, "Asynchronous D2H/H2D", ("capture-d2h", "restore-canonical-h2d"), "medium", "medium"),
        (
            6,
            "Overlap transformation and copy",
            (
                "capture-unpage",
                "capture-stack",
                "capture-concatenate",
                "capture-d2h",
                "restore-canonical-h2d",
            ),
            "medium",
            "high",
        ),
        (
            7,
            "Combine integrity scans",
            ("integrity", "publish", "import-validation", "transport-layout"),
            "high",
            "medium",
        ),
        (8, "Eliminate unnecessary full-page zero-fill", ("zero-fill",), "medium", "low"),
        (
            9,
            "Fused restore transform/native write",
            ("restore-destination-conversion", "restore-destination-write"),
            "high",
            "high",
        ),
        (
            10,
            "Device-side integrity proof",
            ("integrity", "publish", "import-validation"),
            "high",
            "high",
        ),
        (
            11,
            "Avoid full destination D2H recapture",
            (
                "validation-native-read",
                "validation-canonical-transform",
                "validation-d2h",
                "validation-host-compare",
            ),
            "medium",
            "medium",
        ),
        (
            12,
            "Validation sampling if semantics permit",
            (
                "validation-native-read",
                "validation-canonical-transform",
                "validation-d2h",
                "validation-host-compare",
            ),
            "very high",
            "low",
        ),
    )
    rows = []
    for rank, name, needles, risk, complexity in candidates:
        pass_count, byte_count, wall_seconds = estimate(*needles)
        rows.append(
            f"| {rank} | {name} | {pass_count} | {byte_count} | "
            f"{wall_seconds:.6f} | {risk} | {complexity} |"
        )
    logical = int(report["movement"]["logical_state_bytes"])
    return (
        "# Experiment 004 v11 Software Optimization Plan\n\n"
        "Status: **PLAN ONLY — NOT IMPLEMENTED OR EXECUTED**.\n\n"
        "Every estimate below is computed from measured v10 pass labels, byte touches, and "
        "deduplicated operation intervals. Bytes and time are removable upper bounds, not promised "
        "speedups; zero means v10 exposed no separately attributable pass for that family.\n\n"
        f"Frozen logical-state denominator: `{logical}` bytes.\n\n"
        "| Rank | Candidate | Measured passes targeted | Physical bytes potentially removed | "
        "Critical-path seconds potentially removed | Semantic risk | Complexity |\n"
        "|---:|---|---:|---:|---:|---|---|\n"
        + "\n".join(rows)
        + "\n\nCorrectness gates remain fresh destination allocation, full integrity until a proven "
        "equivalent replaces it, eight branches, at least eight continuation tokens, and 8/8 "
        "independent first-token matches. No optimization is authorized by this plan.\n"
    ).encode()


def _campaign_plan(report: dict[str, Any]) -> bytes:
    capacity = report["capacity"]
    return (
        "# Experiment 004 Baseline Campaign Plan\n\n"
        "Status: **PLAN ONLY — NO CAMPAIGN ARM WAS EXECUTED BY THIS GENERATOR**.\n\n"
        "## Arms\n\n"
        "A. `KILL_AND_RECOMPUTE`\n\n"
        "B. `NAIVE_PRESERVE` — the frozen scientifically valid v10 baseline\n\n"
        "C. `OPTIMIZED_PRESERVE` — future v11 software path\n\n"
        "## Fixed experimental controls\n\n"
        f"- λ₁: `{capacity['lambda_1_rps']}` requests/s\n"
        f"- λ₂: `{capacity['lambda_2_rps']}` requests/s\n"
        f"- λ_spike: `{capacity['lambda_spike_rps']}` requests/s\n"
        "- Hardware: the same exact two-A100-80GB topology and runtime pins as v10\n"
        "- State: the same 16,384-token shared prefix, eight branches, and at least 256-token "
        "divergent suffix\n"
        "- Seeds: `41, 73, 113, 149, 197`; arm order is SHA-256-seeded and position-balanced\n"
        "- Checks: identical output length, trigger, recovery, restore, first-token, continuation, "
        "fresh-allocation, and cleanup gates\n"
        "- Accounting: identical logical denominator and complete physical pass graph for every arm\n\n"
        "Analyze paired per-seed differences; do not report unsupported tail percentiles. A failed "
        "arm remains in the audit trail and is never silently retried or replaced.\n\n"
        "This plan does not authorize execution, Experiment 005, or hardware implementation.\n"
    ).encode()


def _atomic_replace(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(contents)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def publish_v10_success(
    *,
    repository_root: Path,
    run_root: Path,
    experiment_root: Path,
) -> dict[str, Any]:
    """Publish the complete v10 success bundle, refusing every invalid run."""

    repository_root = repository_root.resolve(strict=True)
    run_root = run_root.resolve(strict=True)
    validity_path = run_root / "analysis/scientific-validity.json"
    validity = _require_success(validity_path)
    experiment_root = experiment_root.resolve(strict=True)
    try:
        experiment_root.relative_to(repository_root)
    except ValueError as error:
        raise ValueError("v10 experiment publication root must be inside the repository") from error
    source_paths = {
        "validity": validity_path,
        "serving": run_root / "serving/result.json",
        "rollout": run_root / "rollout/result.json",
        "movement": run_root / "analysis/movement-accounting.json",
        "recovery": run_root / "analysis/serving-recovery.json",
        "config": run_root / "effective-config.json",
        "serving_telemetry": run_root / "serving/telemetry/resource-sampling.json",
        "rollout_telemetry": run_root / "rollout/telemetry/resource-sampling.json",
        "topology": run_root / "topology/topology-commands.json",
        "gpu_phase_ledger": run_root / "analysis/gpu-phase-ledger.json",
        "authorization": experiment_root / "v10-authorization.json",
        "phase_budget": experiment_root / "v10-phase-budget.json",
        "ledger": experiment_root / "gpu-hours.json",
    }
    for path in source_paths.values():
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"successful v10 publication source is absent: {path}")
    serving = _load_object(source_paths["serving"])
    rollout = _load_object(source_paths["rollout"])
    movement = _load_object(source_paths["movement"])
    recovery = _load_object(source_paths["recovery"])
    config = _load_object(source_paths["config"])
    ledger = _load_object(source_paths["ledger"])
    authorization = _load_object(source_paths["authorization"])
    topology = json.loads(source_paths["topology"].read_text())
    if not isinstance(topology, list) or not topology:
        raise ValueError("v10 success publication lacks GPU topology commands")
    if serving.get("status") != "succeeded" or rollout.get("status") != "succeeded":
        raise ValueError("scientifically valid publication requires two successful raw workers")
    if (
        recovery.get("schema_version")
        != "sloforge.branchfabric.experiment-004-v10-serving-recovery/v1"
    ):
        raise ValueError("v10 serving recovery artifact has an unexpected schema")
    if config.get("serving_methodology") != "v10-global-capacity":
        raise ValueError("v10 success publication received a non-v10 effective config")
    raw_provenance = [_raw_ref(path, repository_root) for path in source_paths.values()]
    telemetry = (
        ("gpu0", _load_object(source_paths["serving_telemetry"])),
        ("gpu1", _load_object(source_paths["rollout_telemetry"])),
    )
    plots = _plot_payloads(
        validity=validity,
        serving=serving,
        rollout=rollout,
        movement=movement,
        telemetry=telemetry,
        provenance=raw_provenance,
    )
    if len(plots) != 10 or [plot["plot_id"] for plot in plots] != [row[0] for row in PLOT_SPECS]:
        raise RuntimeError("v10 success publication did not produce exactly ten required plots")

    files: dict[Path, bytes] = {}
    plot_refs: list[dict[str, Any]] = []
    for plot in plots:
        json_path = experiment_root / "plots" / f"{plot['plot_id']}.json"
        svg_path = experiment_root / "plots" / f"{plot['plot_id']}.svg"
        json_bytes = _canonical_bytes(plot)
        svg_bytes = _svg(plot)
        files[json_path] = json_bytes
        files[svg_path] = svg_bytes
        plot_refs.append(
            {
                "plot_id": plot["plot_id"],
                "json": {
                    "path": json_path.relative_to(repository_root).as_posix(),
                    "sha256": hashlib.sha256(json_bytes).hexdigest(),
                },
                "svg": {
                    "path": svg_path.relative_to(repository_root).as_posix(),
                    "sha256": hashlib.sha256(svg_bytes).hexdigest(),
                },
            }
        )
    report = _report_payload(
        validity=validity,
        config=config,
        movement=movement,
        serving=serving,
        rollout=rollout,
        serving_telemetry=telemetry[0][1],
        ledger=ledger,
        plot_refs=plot_refs,
        raw_provenance=raw_provenance,
    )
    report_json_path = (
        repository_root / "reports/branchfabric-gpu-validation-experiment-004-v10-final.json"
    )
    report_md_path = (
        repository_root / "reports/branchfabric-gpu-validation-experiment-004-v10-final.md"
    )
    optimization_path = (
        repository_root / "docs/branchfabric/EXPERIMENT_004_V11_OPTIMIZATION_PLAN.md"
    )
    campaign_path = repository_root / "docs/branchfabric/EXPERIMENT_004_BASELINE_CAMPAIGN_PLAN.md"
    files[report_json_path] = _canonical_bytes(report)
    files[report_md_path] = _report_markdown(report)
    files[optimization_path] = _optimization_plan(report, movement)
    files[campaign_path] = _campaign_plan(report)
    files[experiment_root / "scientific-validity.json"] = _canonical_bytes(validity)
    files[experiment_root / "serving-recovery.json"] = _canonical_bytes(recovery)
    files[experiment_root / "movement-accounting.json"] = _canonical_bytes(movement)
    files[experiment_root / "branch-resume-validation.json"] = _canonical_bytes(
        {
            "schema_version": (
                "sloforge.branchfabric.experiment-004-v10-branch-resume-validation/v1"
            ),
            "attempt_id": config["attempt_id"],
            "passed": validity["branch_resume_pass"],
            "evidence": validity["branch_resume"],
            "raw_provenance": _raw_ref(source_paths["rollout"], repository_root),
        }
    )
    files[experiment_root / "state-passes/v10-state-pass-graph.json"] = _canonical_bytes(
        _state_pass_graph(movement)
    )
    result_hashes = {
        name: _sha256(path)
        for name, path in source_paths.items()
        if name
        in {
            "serving",
            "rollout",
            "movement",
            "recovery",
            "validity",
            "topology",
            "gpu_phase_ledger",
        }
    }
    baseline_body = {
        "schema_version": "sloforge.branchfabric.experiment-004-naive-baseline/v1",
        "status": "FROZEN",
        "immutable": True,
        "experiment": "BranchFabric GPU Validation Experiment 004",
        "experiment_version": "v10",
        "attempt_id": config["attempt_id"],
        "repository_commit": authorization["code_commit"],
        "model": config["model"],
        "model_revision": config["model_revision"],
        "runtime": config["runtime"],
        "runtime_version": config["runtime_version"],
        "gpu_topology": topology,
        "load_parameters": {
            "lambda_1_rps": config["lambda_1_rps"],
            "lambda_spike_rps": config["lambda_spike_rps"],
            "lambda_2_rps": config["lambda_2_rps"],
            "control_rps": config["gpu0_control_request_rate_per_second"],
            "restore_rps": config["gpu0_restore_request_rate_per_second"],
        },
        "state_parameters": {
            "prefix_length": config["prefix_length"],
            "fanout": config["fanout"],
            "suffix_length": config["suffix_length"],
            "logical_state_bytes": report["movement"]["logical_state_bytes"],
        },
        "transaction_configuration_sha256": _sha256(source_paths["config"]),
        "movement_graph_sha256": hashlib.sha256(
            _canonical_bytes(_state_pass_graph(movement))
        ).hexdigest(),
        "scientific_validity_sha256": _sha256(validity_path),
        "result_artifact_hashes": result_hashes,
        "recommended_git_tag": "branchfabric-exp004-naive-baseline-v10",
        "git_tag_created": False,
        "hash_convention": "sha256(canonical JSON body excluding artifact_hash)",
    }
    baseline_record = {
        **baseline_body,
        "artifact_hash": hashlib.sha256(_canonical_bytes(baseline_body)).hexdigest(),
    }
    files[experiment_root / "baseline/branchfabric-exp004-naive-baseline-v10.json"] = (
        _canonical_bytes(baseline_record)
    )

    manifest_path = experiment_root / "manifest.json"
    previous_manifest = None
    if manifest_path.is_file():
        previous_bytes = manifest_path.read_bytes()
        previous_digest = hashlib.sha256(previous_bytes).hexdigest()
        previous_payload = _load_object(manifest_path)
        if (
            previous_payload.get("schema_version")
            == "sloforge.branchfabric.experiment-004-v10-success-manifest/v1"
            and previous_payload.get("attempt_id") == config["attempt_id"]
        ):
            raise FileExistsError("this v10 attempt already has a committed success publication")
        previous_path = experiment_root / "analysis/manifests" / f"pre-v10-{previous_digest}.json"
        if previous_path.is_file() and previous_path.read_bytes() != previous_bytes:
            raise ValueError("archived Experiment 004 manifest has a hash collision")
        files[previous_path] = previous_bytes
        previous_manifest = {
            "artifact_reference": previous_path.relative_to(repository_root).as_posix(),
            "artifact_sha256": previous_digest,
            "sample_selector": "$",
        }
    generated = [
        {
            "path": path.relative_to(repository_root).as_posix(),
            "bytes": len(contents),
            "sha256": hashlib.sha256(contents).hexdigest(),
        }
        for path, contents in sorted(files.items(), key=lambda item: str(item[0]))
    ]
    manifest = {
        "schema_version": "sloforge.branchfabric.experiment-004-v10-success-manifest/v1",
        "attempt_id": config["attempt_id"],
        "scientifically_valid": True,
        "frozen_baseline": "NAIVE_PRESERVE_V10",
        "generated_files_excluding_manifest": generated,
        "raw_provenance": raw_provenance,
        "superseded_manifest": previous_manifest,
        "optimized_preservation_executed": False,
        "kill_and_recompute_executed": False,
        "experiment_005_executed": False,
        "fpga_justification_claimed": False,
    }
    files[manifest_path] = _canonical_bytes(manifest)
    for path, contents in sorted(files.items(), key=lambda item: str(item[0])):
        if path == manifest_path:
            continue
        _atomic_replace(path, contents)
    # The manifest is the publication commit marker and is replaced only after
    # every content-addressed output reached its final path.
    _atomic_replace(manifest_path, files[manifest_path])
    return manifest


__all__ = ["PLOT_SPECS", "publish_v10_success"]
