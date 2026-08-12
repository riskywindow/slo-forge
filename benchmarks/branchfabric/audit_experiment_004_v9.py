#!/usr/bin/env python3
"""Reconstruct and audit Experiment 004 v9 state movement without a GPU.

The audit is intentionally pinned to the immutable v9 artifacts.  It fails
closed if their hashes or the recorded movement totals change.  The additional
restore-import verification is reconstructed from the executed source path and
bounded by the raw phase/allocation timestamps; it is never presented as a
record emitted by v9.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = ROOT / "artifacts/branchfabric/gpu-validation/experiment-004"
V9_ROOT = EXPERIMENT_ROOT / "raw/modal/exp004-pilot-naive-s41-v9"
ROLLOUT_PATH = V9_ROOT / "rollout/result.json"
TELEMETRY_PATH = V9_ROOT / "rollout/telemetry/cuda-and-host-operations.json"
RESOURCE_PATH = V9_ROOT / "rollout/telemetry/resource-sampling.json"
EXPECTED_SHA256 = {
    ROLLOUT_PATH: "5bc1167139165cc36792186ae607e7a0d971aa8f614f784f5cf93ac5c2d7ae2e",
    TELEMETRY_PATH: "cb0841111a2f80bb6d0227e80445a7b0464444aa807a1a1720a9a29bde9c81f4",
}


LABEL_AUDIT: dict[str, dict[str, Any]] = {
    "capture-source-native-read": dict(
        stage="capture",
        processor="GPU",
        real=True,
        bookkeeping=False,
        required=True,
        artifact=False,
        software="SOFTWARE_FUSIBLE",
        streaming="STREAMING_FUSIBLE",
        counted_elsewhere=False,
        explanation="Mandatory native-pool read; the standalone gathered/contiguous buffer is fusible.",
    ),
    "capture-native-axis-contiguous": dict(
        stage="capture",
        processor="GPU",
        real=True,
        bookkeeping=False,
        required=False,
        artifact=True,
        software="SOFTWARE_FUSIBLE",
        streaming="STREAMING_FUSIBLE",
        counted_elsewhere=False,
        explanation="Second full materialization caused by movedim(...).contiguous().",
    ),
    "capture-unpage-valid-tokens": dict(
        stage="transform",
        processor="GPU",
        real=True,
        bookkeeping=False,
        required=False,
        artifact=True,
        software="SOFTWARE_FUSIBLE",
        streaming="STREAMING_FUSIBLE",
        counted_elsewhere=False,
        explanation="Valid-token selection is required; its standalone output buffer is not.",
    ),
    "capture-stack-layers": dict(
        stage="transform",
        processor="GPU",
        real=True,
        bookkeeping=False,
        required=False,
        artifact=True,
        software="SOFTWARE_FUSIBLE",
        streaming="STREAMING_FUSIBLE",
        counted_elsewhere=False,
        explanation="Per-page layer-stack materialization.",
    ),
    "capture-concatenate-pages": dict(
        stage="transform",
        processor="GPU",
        real=True,
        bookkeeping=False,
        required=False,
        artifact=True,
        software="SOFTWARE_FUSIBLE",
        streaming="STREAMING_FUSIBLE",
        counted_elsewhere=False,
        explanation="Final flat canonical concatenation can feed a chunked D2H stream.",
    ),
    "capture-d2h": dict(
        stage="d2h",
        processor="GPU+CPU",
        real=True,
        bookkeeping=False,
        required=True,
        artifact=False,
        software="CANNOT_FUSE",
        streaming="STREAMING_FUSIBLE",
        counted_elsewhere="endpoint bytes are also represented by the D2H link edge",
        explanation="One required synchronous checkpoint transfer to pinned host memory.",
    ),
    "capture-pinned-transport-lifetime": dict(
        stage="checkpoint_lifetime",
        processor="CPU",
        real=False,
        bookkeeping=True,
        required=True,
        artifact=False,
        software="CANNOT_FUSE",
        streaming="CANNOT_FUSE",
        counted_elsewhere=False,
        explanation="Allocation-capacity bookkeeping only; persistent host checkpoint storage is required by naive preservation.",
    ),
    "capture-integrity-manifest": dict(
        stage="integrity",
        processor="CPU",
        real=True,
        bookkeeping=False,
        required=True,
        artifact=True,
        software="SOFTWARE_FUSIBLE",
        streaming="STREAMING_FUSIBLE",
        counted_elsewhere=False,
        explanation="Integrity is required; payload.numpy().tobytes() creates an avoidable full pageable copy.",
    ),
    "capture-integrity-hash-reads": dict(
        stage="integrity",
        processor="CPU",
        real=True,
        bookkeeping=False,
        required=True,
        artifact=True,
        software="SOFTWARE_FUSIBLE",
        streaming="STREAMING_FUSIBLE",
        counted_elsewhere=False,
        explanation="Aggregate and per-page SHA passes are real but can share one streaming traversal.",
    ),
    "transport-publish-validation": dict(
        stage="publish",
        processor="CPU",
        real=True,
        bookkeeping=False,
        required=False,
        artifact=True,
        software="SOFTWARE_FUSIBLE",
        streaming="SOFTWARE_FUSIBLE",
        counted_elsewhere=False,
        explanation="Immediate re-verification of the unchanged payload after manifest construction.",
    ),
    "transport-publish-hash-reads": dict(
        stage="publish",
        processor="CPU",
        real=True,
        bookkeeping=False,
        required=False,
        artifact=True,
        software="SOFTWARE_FUSIBLE",
        streaming="SOFTWARE_FUSIBLE",
        counted_elsewhere=False,
        explanation="Aggregate/page hash reads belonging to redundant publish verification.",
    ),
    "restore-transport-validation": dict(
        stage="restore_integrity",
        processor="CPU",
        real=True,
        bookkeeping=False,
        required=True,
        artifact=True,
        software="SOFTWARE_FUSIBLE",
        streaming="STREAMING_FUSIBLE",
        counted_elsewhere=False,
        explanation="Pre-use verification barrier is mandatory; full tobytes materialization is not.",
    ),
    "restore-transport-hash-reads": dict(
        stage="restore_integrity",
        processor="CPU",
        real=True,
        bookkeeping=False,
        required=True,
        artifact=True,
        software="SOFTWARE_FUSIBLE",
        streaming="STREAMING_FUSIBLE",
        counted_elsewhere=False,
        explanation="Aggregate/page hashes can share a streaming traversal.",
    ),
    "restore-h2d": dict(
        stage="h2d",
        processor="CPU+GPU",
        real=True,
        bookkeeping=False,
        required=True,
        artifact=False,
        software="CANNOT_FUSE",
        streaming="STREAMING_FUSIBLE",
        counted_elsewhere="endpoint bytes are also represented by the H2D link edge",
        explanation="Required host-to-device checkpoint transfer.",
    ),
    "restore-import-validation": dict(
        stage="state_import",
        processor="CPU",
        real=True,
        bookkeeping=False,
        required=False,
        artifact=True,
        software="SOFTWARE_FUSIBLE",
        streaming="SOFTWARE_FUSIBLE",
        counted_elsewhere=False,
        explanation="Unrecorded second verification inside import_group(); unchanged payload was already verified immediately before H2D.",
    ),
    "restore-import-hash-reads": dict(
        stage="state_import",
        processor="CPU",
        real=True,
        bookkeeping=False,
        required=False,
        artifact=True,
        software="SOFTWARE_FUSIBLE",
        streaming="SOFTWARE_FUSIBLE",
        counted_elsewhere=False,
        explanation="Unrecorded aggregate/page hash reads from the redundant import_group() verification.",
    ),
    "restore-zero-native-pages": dict(
        stage="state_import",
        processor="GPU",
        real=True,
        bookkeeping=False,
        required=False,
        artifact=True,
        software="SOFTWARE_FUSIBLE",
        streaming="SOFTWARE_FUSIBLE",
        counted_elsewhere=False,
        explanation="All v9 pages are full, so zero fill is overwritten completely.",
    ),
    "restore-overlay-valid-tokens": dict(
        stage="state_import",
        processor="GPU",
        real=True,
        bookkeeping=False,
        required=False,
        artifact=True,
        software="SOFTWARE_FUSIBLE",
        streaming="STREAMING_FUSIBLE",
        counted_elsewhere=False,
        explanation="Data placement is required; standalone overlay materialization is fusible.",
    ),
    "restore-native-axis-contiguous": dict(
        stage="state_import",
        processor="GPU",
        real=True,
        bookkeeping=False,
        required=False,
        artifact=True,
        software="SOFTWARE_FUSIBLE",
        streaming="STREAMING_FUSIBLE",
        counted_elsewhere=False,
        explanation="Native-axis contiguous materialization.",
    ),
    "restore-stack-native-pages": dict(
        stage="state_import",
        processor="GPU",
        real=True,
        bookkeeping=False,
        required=False,
        artifact=True,
        software="SOFTWARE_FUSIBLE",
        streaming="STREAMING_FUSIBLE",
        counted_elsewhere=False,
        explanation="Per-layer native page stack materialization.",
    ),
    "restore-destination-native-write": dict(
        stage="state_import",
        processor="GPU",
        real=True,
        bookkeeping=False,
        required=True,
        artifact=False,
        software="SOFTWARE_FUSIBLE",
        streaming="STREAMING_FUSIBLE",
        counted_elsewhere=False,
        explanation="Fresh destination write is mandatory; its source intermediate can be fused.",
    ),
    "validation-destination-native-read": dict(
        stage="validation",
        processor="GPU",
        real=True,
        bookkeeping=False,
        required=True,
        artifact=False,
        software="CANNOT_FUSE",
        streaming="STREAMING_FUSIBLE",
        counted_elsewhere=False,
        explanation="Post-write readback is the current exact-integrity barrier.",
    ),
    "validation-native-axis-contiguous": dict(
        stage="validation",
        processor="GPU",
        real=True,
        bookkeeping=False,
        required=False,
        artifact=True,
        software="SOFTWARE_FUSIBLE",
        streaming="STREAMING_FUSIBLE",
        counted_elsewhere=False,
        explanation="Validation gather contiguous materialization.",
    ),
    "validation-unpage-valid-tokens": dict(
        stage="validation",
        processor="GPU",
        real=True,
        bookkeeping=False,
        required=False,
        artifact=True,
        software="SOFTWARE_FUSIBLE",
        streaming="STREAMING_FUSIBLE",
        counted_elsewhere=False,
        explanation="Validation valid-token materialization.",
    ),
    "validation-stack-layers": dict(
        stage="validation",
        processor="GPU",
        real=True,
        bookkeeping=False,
        required=False,
        artifact=True,
        software="SOFTWARE_FUSIBLE",
        streaming="STREAMING_FUSIBLE",
        counted_elsewhere=False,
        explanation="Validation layer-stack materialization.",
    ),
    "validation-concatenate-pages": dict(
        stage="validation",
        processor="GPU",
        real=True,
        bookkeeping=False,
        required=False,
        artifact=True,
        software="SOFTWARE_FUSIBLE",
        streaming="STREAMING_FUSIBLE",
        counted_elsewhere=False,
        explanation="Validation canonical concatenation can stream to comparison or D2H.",
    ),
    "validation-d2h": dict(
        stage="validation",
        processor="GPU+CPU",
        real=True,
        bookkeeping=False,
        required=False,
        artifact=True,
        software="SOFTWARE_FUSIBLE",
        streaming="STREAMING_FUSIBLE",
        counted_elsewhere="endpoint bytes are also represented by the D2H link edge",
        explanation="Real full destination recapture; exact validation is required, but bulk host readback is an implementation choice.",
    ),
    "validation-expected-page-concatenation": dict(
        stage="validation",
        processor="CPU",
        real=True,
        bookkeeping=False,
        required=False,
        artifact=True,
        software="SOFTWARE_FUSIBLE",
        streaming="STREAMING_FUSIBLE",
        counted_elsewhere=False,
        explanation="Expected host bytes can be viewed/hashed without a second concatenation.",
    ),
    "validation-host-tensor-compare": dict(
        stage="validation",
        processor="CPU",
        real=True,
        bookkeeping=False,
        required=False,
        artifact=True,
        software="SOFTWARE_FUSIBLE",
        streaming="STREAMING_FUSIBLE",
        counted_elsewhere=False,
        explanation="Real compare; current host placement is removable with device-side proof.",
    ),
    "validation-recapture-host-lifetime": dict(
        stage="validation",
        processor="CPU",
        real=False,
        bookkeeping=True,
        required=False,
        artifact=True,
        software="SOFTWARE_FUSIBLE",
        streaming="STREAMING_FUSIBLE",
        counted_elsewhere=False,
        explanation="Allocation-lifetime bookkeeping for validation buffers, not movement.",
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _atomic_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(_canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _label(record_id: str) -> str:
    return record_id.split(":", 2)[1]


def _audit_record(record: dict[str, Any], *, inferred: bool = False) -> dict[str, Any]:
    label = _label(record["record_id"])
    classification = LABEL_AUDIT[label]
    return {
        "record_id": record["record_id"],
        "record_origin": "code_inferred_missing_v9_ledger" if inferred else "raw_v9_state_pass",
        "stage": classification["stage"],
        "operation": record["operation"],
        "logical_segment": record["state_segment"],
        "logical_offset_bytes": record["logical_offset_bytes"],
        "logical_bytes": record["logical_bytes"],
        "source_memory": record["source_memory"],
        "destination_memory": record["destination_memory"],
        "bytes_read": record["bytes_read"],
        "bytes_written": record["bytes_written"],
        "link_transfer_bytes": record["transfer_bytes"],
        "link_transfer_direction": record["transfer_direction"],
        "checksum_work_bytes": record["checksum_bytes"],
        "temporary_bytes": record["temporary_allocation_bytes"],
        "processor": classification["processor"],
        "start_ns": record["start_ns"],
        "end_ns": record["end_ns"],
        "real_physical_pass": classification["real"],
        "semantic_bookkeeping": classification["bookkeeping"],
        "counted_elsewhere": classification["counted_elsewhere"],
        "required": classification["required"],
        "implementation_artifact": classification["artifact"],
        "software_fusion": classification["software"],
        "streaming_hardware_fusion": classification["streaming"],
        "critical_path": "reclamation"
        if classification["stage"] in {"capture", "transform", "d2h", "integrity", "publish"}
        else (None if classification["stage"] == "checkpoint_lifetime" else "restore"),
        "explanation": classification["explanation"],
    }


def _missing_import_records(rollout: dict[str, Any]) -> list[dict[str, Any]]:
    segments = rollout["movement_report"]["logical_segments"]
    phases = {row["phase"]: row["monotonic_timestamp_ns"] for row in rollout["phase_events"]}
    first_allocation = next(
        row for row in rollout["restore_stages"] if row["stage"] == "destination_allocation_subset"
    )
    start_ns = int(phases["STATE_IMPORT_BEGIN"])
    end_ns = int(first_allocation["measured_start_ns"])
    records: list[dict[str, Any]] = []
    sequence = 10_000
    for label, operation, read_multiplier, write_multiplier, checksum_multiplier, temporary in (
        ("restore-import-validation", "validate", 1, 1, 0, True),
        ("restore-import-hash-reads", "validate", 2, 0, 2, False),
    ):
        for segment in segments:
            sequence += 1
            logical = int(segment["logical_bytes"])
            records.append(
                {
                    "record_id": f"inferred-{sequence}:{label}:{segment['segment_id']}",
                    "state_segment": segment["segment_id"],
                    "operation": operation,
                    "source_memory": "pinned_host_transport"
                    if write_multiplier
                    else "pageable_host_buffer",
                    "destination_memory": "pageable_host_buffer" if write_multiplier else "none",
                    "bytes_read": logical * read_multiplier,
                    "bytes_written": logical * write_multiplier,
                    "transfer_bytes": 0,
                    "transfer_direction": "none",
                    "checksum_bytes": logical * checksum_multiplier,
                    "temporary_allocation_bytes": logical if temporary else 0,
                    "logical_offset_bytes": 0,
                    "logical_bytes": logical,
                    "start_ns": start_ns,
                    "end_ns": end_ns,
                }
            )
    return records


def _grouped_passes(records: list[dict[str, Any]], logical: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[_label(record["record_id"])].append(record)
    result: list[dict[str, Any]] = []
    for label in LABEL_AUDIT:
        rows = grouped[label]
        if not rows:
            raise ValueError(f"audit label {label!r} is absent")
        classification = LABEL_AUDIT[label]
        result.append(
            {
                "label": label,
                "stage": classification["stage"],
                "operation": rows[0]["operation"],
                "record_count": len(rows),
                "logical_bytes": sum(row["logical_bytes"] for row in rows),
                "bytes_read": sum(row["bytes_read"] for row in rows),
                "bytes_written": sum(row["bytes_written"] for row in rows),
                "link_transfer_bytes": sum(row["transfer_bytes"] for row in rows),
                "checksum_work_bytes": sum(row["checksum_bytes"] for row in rows),
                "temporary_allocation_bytes": sum(
                    row["temporary_allocation_bytes"] for row in rows
                ),
                "full_state_memory_passes": (
                    sum(row["bytes_read"] + row["bytes_written"] for row in rows) / logical
                ),
                "wall_ns": max(row["end_ns"] for row in rows)
                - min(row["start_ns"] for row in rows),
                "start_ns": min(row["start_ns"] for row in rows),
                "end_ns": max(row["end_ns"] for row in rows),
                "processor": classification["processor"],
                "required": classification["required"],
                "implementation_artifact": classification["artifact"],
                "software_fusion": classification["software"],
                "streaming_hardware_fusion": classification["streaming"],
                "explanation": classification["explanation"],
            }
        )
    return result


def _interval_union_ns(rows: list[dict[str, Any]]) -> int:
    intervals = sorted((int(row["start_ns"]), int(row["end_ns"])) for row in rows)
    if not intervals:
        return 0
    total = 0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def _chain(
    name: str,
    labels: list[str],
    grouped: list[dict[str, Any]],
    logical: int,
    *,
    barriers: list[str],
    optional: list[str],
    buffers: list[str],
) -> dict[str, Any]:
    by_label = {row["label"]: row for row in grouped}
    rows = [by_label[label] for label in labels]
    return {
        "name": name,
        "operations": labels,
        "logical_bytes": logical,
        "real_physical_endpoint_bytes": sum(
            row["bytes_read"] + row["bytes_written"] for row in rows
        ),
        "external_link_bytes": sum(row["link_transfer_bytes"] for row in rows),
        "full_physical_bytes_read_written_moved": sum(
            row["bytes_read"] + row["bytes_written"] + row["link_transfer_bytes"] for row in rows
        ),
        "full_state_endpoint_passes": sum(row["bytes_read"] + row["bytes_written"] for row in rows)
        / logical,
        # Several labels deliberately describe the same fused implementation
        # interval.  Interval union prevents those semantic labels from
        # multiplying measured wall time.
        "wall_time_accounting": "union of raw operation intervals; CPU/GPU transfer intervals appear in both processor views",
        "cpu_wall_ns": _interval_union_ns([row for row in rows if "CPU" in row["processor"]]),
        "gpu_wall_ns": _interval_union_ns([row for row in rows if "GPU" in row["processor"]]),
        "temporary_allocation_bytes": sum(row["temporary_allocation_bytes"] for row in rows),
        "mandatory_barriers": barriers,
        "optional_barriers": optional,
        "potentially_removable_buffers": buffers,
        "adjacent_pair_classification": [
            {
                "left": left,
                "right": right,
                "classification": (
                    "CANNOT_FUSE"
                    if left
                    in {
                        "restore-transport-hash-reads",
                        "restore-destination-native-write",
                    }
                    else (
                        "STREAMING_FUSIBLE"
                        if "d2h" in left or "d2h" in right or "h2d" in right
                        else "SOFTWARE_FUSIBLE"
                    )
                ),
            }
            for left, right in pairwise(labels)
        ],
    }


def build_audit() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    for path, expected in EXPECTED_SHA256.items():
        observed = _sha256(path)
        if observed != expected:
            raise ValueError(f"immutable v9 artifact hash changed: {path}: {observed}")
    rollout = json.loads(ROLLOUT_PATH.read_text())
    telemetry = json.loads(TELEMETRY_PATH.read_text())
    resources = json.loads(RESOURCE_PATH.read_text())
    report = rollout["movement_report"]
    accounting = report["accounting"]
    logical = int(accounting["logical_state_bytes"])
    if logical != 1_056_964_608 or len(report["passes"]) != 252:
        raise ValueError("v9 movement identity differs from the audited trial")
    if set(_label(row["record_id"]) for row in report["passes"]) != set(LABEL_AUDIT) - {
        "restore-import-validation",
        "restore-import-hash-reads",
    }:
        raise ValueError("v9 movement labels differ from the audited implementation")

    inferred = _missing_import_records(rollout)
    combined = [*report["passes"], *inferred]
    per_record = [
        *(_audit_record(row) for row in report["passes"]),
        *(_audit_record(row, inferred=True) for row in inferred),
    ]
    grouped = _grouped_passes(combined, logical)
    recorded_endpoint = int(
        accounting["physical_bytes_read"]
        + accounting["physical_bytes_written"]
        + accounting["host_intermediate_bytes"]
        + accounting["gpu_intermediate_bytes"]
    )
    link = int(accounting["d2h_bytes"] + accounting["h2d_bytes"])
    missing_endpoint = sum(row["bytes_read"] + row["bytes_written"] for row in inferred)
    corrected_endpoint = recorded_endpoint + missing_endpoint
    corrected_full = corrected_endpoint + link
    conservative_avoidable = (
        sum(
            row["bytes_read"] + row["bytes_written"]
            for row in report["passes"]
            if not row["required_unavoidable"]
        )
        + missing_endpoint
    )
    if (
        recorded_endpoint,
        link,
        missing_endpoint,
        corrected_endpoint,
        corrected_full,
        conservative_avoidable,
    ) != (
        53_905_195_008,
        3_170_893_824,
        4_227_858_432,
        58_133_053_440,
        61_303_947_264,
        32_765_902_848,
    ):
        raise ValueError("audited v9 movement formulas no longer reconcile")

    cuda_d2h = [row for row in telemetry["cuda_operations"] if row["kind"] == "d2h"]
    physical_transfers = [
        {
            "operation_id": row["operation_id"],
            "bytes": row["bytes"],
            "copy_sizes_bytes": row["copy_sizes_bytes"],
            "cpu_start_monotonic_ns": row["cpu_start_monotonic_ns"],
            "cpu_end_monotonic_ns": row["cpu_end_monotonic_ns"],
            "cuda_event_elapsed_ns": row["cuda_event_elapsed_ns"],
            "real_physical_transfer": True,
            "accounting_duplicate": False,
            "purpose": "checkpoint_capture"
            if row["operation_id"] == "capture-canonical-d2h"
            else "destination_validation_recapture",
        }
        for row in cuda_d2h
    ]
    unique_d2h = sum(row["bytes"] for row in cuda_d2h)
    if unique_d2h != 2_113_929_216 or sorted(row["bytes"] for row in cuda_d2h) != [
        102_760_448,
        954_204_160,
        1_056_964_608,
    ]:
        raise ValueError("raw CUDA D2H evidence changed")
    d2h = {
        "schema_version": "sloforge.branchfabric.v9-d2h-audit/v1",
        "attempt_id": "exp004-pilot-naive-s41-v9",
        "logical_bytes": logical,
        "logical_layout": {
            "page_count": 1152,
            "page_bytes": 917_504,
            "shared_root_pages": 1024,
            "shared_root_bytes": 939_524_096,
            "private_pages": 128,
            "private_bytes": 117_440_512,
            "private_branches": 8,
            "k_and_v_already_included": True,
            "formula": "1152 pages * 16 tokens/page * 28 layers * 2 K/V * 4 heads * 128 dimension * 2 bf16 bytes",
        },
        "physical_transfers": physical_transfers,
        "unique_real_d2h_bytes": unique_d2h,
        "duplicate_accounting_bytes": 0,
        "required_bytes": logical,
        "avoidable_bytes": logical,
        "current_implementation_required_bytes": unique_d2h,
        "explanation": "One full checkpoint capture plus one full destination-validation recapture. The validation transfer is split into disjoint 954,204,160-byte and 102,760,448-byte subsets. K and V are already in the logical denominator; no tracer event is duplicated.",
        "provenance": {
            "rollout_result": {
                "path": str(ROLLOUT_PATH.relative_to(ROOT)),
                "sha256": EXPECTED_SHA256[ROLLOUT_PATH],
            },
            "cuda_telemetry": {
                "path": str(TELEMETRY_PATH.relative_to(ROOT)),
                "sha256": EXPECTED_SHA256[TELEMETRY_PATH],
            },
        },
    }

    formulas = {
        "schema_version": "sloforge.branchfabric.v9-movement-formulas/v1",
        "logical_state_bytes": logical,
        "legacy_reported": {
            "numerator_bytes": int(accounting["amplification_numerator_bytes"]),
            "amplification": float(accounting["state_movement_amplification"]),
            "formula": "51L memory endpoint touches + 3L external link traversals",
            "tracer_duplicate_bytes": 0,
            "cross_metric_overlap_bytes": link,
            "omitted_real_endpoint_bytes": missing_endpoint,
            "canonical": False,
        },
        "full_physical_touch": {
            "definition": "all real physical bytes read, written, or traversing an external link",
            "memory_endpoint_bytes": corrected_endpoint,
            "external_link_bytes": link,
            "numerator_bytes": corrected_full,
            "amplification": corrected_full / logical,
        },
        "memory_touch_only": {
            "definition": "all real memory endpoint reads and writes; link traversal is reported separately",
            "numerator_bytes": corrected_endpoint,
            "amplification": corrected_endpoint / logical,
        },
        "external_movement": {
            "d2h_bytes": int(accounting["d2h_bytes"]),
            "h2d_bytes": int(accounting["h2d_bytes"]),
            "network_bytes": 0,
            "storage_bytes": 0,
            "numerator_bytes": link,
            "amplification": link / logical,
        },
        "avoidable_movement": {
            "definition": "conservative whole-pass lower bound: raw false-required endpoint touches plus proven redundant omitted import verification",
            "numerator_bytes": conservative_avoidable,
            "amplification": conservative_avoidable / logical,
            "lower_bound": True,
        },
        "critical_path_movement": {
            "reclamation_memory_endpoint_bytes": 21_139_292_160,
            "reclamation_external_link_bytes": logical,
            "restore_memory_endpoint_bytes": 36_993_761_280,
            "restore_external_link_bytes": logical * 2,
            "memory_endpoint_numerator_bytes": corrected_endpoint,
            "full_read_write_move_numerator_bytes": corrected_full,
            "memory_endpoint_amplification": corrected_endpoint / logical,
            "full_read_write_move_amplification": corrected_full / logical,
        },
        "temporary_memory": {
            "reported_host_allocation_sum_bytes": int(
                accounting["host_temporary_allocation_bytes"]
            ),
            "missing_import_verify_allocation_bytes": logical,
            "corrected_host_allocation_sum_bytes": int(
                accounting["host_temporary_allocation_bytes"]
            )
            + logical,
            "measured_peak_host_temporary_bytes": int(
                accounting["peak_host_temporary_allocation_bytes"]
            ),
            "peak_changes_after_correction": False,
            "gpu_allocation_sum_bytes": int(accounting["gpu_temporary_allocation_bytes"]),
        },
    }

    capture_labels = [
        "capture-source-native-read",
        "capture-native-axis-contiguous",
        "capture-unpage-valid-tokens",
        "capture-stack-layers",
        "capture-concatenate-pages",
        "capture-d2h",
        "capture-integrity-manifest",
        "capture-integrity-hash-reads",
        "transport-publish-validation",
        "transport-publish-hash-reads",
    ]
    restore_labels = [
        "restore-transport-validation",
        "restore-transport-hash-reads",
        "restore-h2d",
        "restore-import-validation",
        "restore-import-hash-reads",
        "restore-zero-native-pages",
        "restore-overlay-valid-tokens",
        "restore-native-axis-contiguous",
        "restore-stack-native-pages",
        "restore-destination-native-write",
        "validation-destination-native-read",
        "validation-native-axis-contiguous",
        "validation-unpage-valid-tokens",
        "validation-stack-layers",
        "validation-concatenate-pages",
        "validation-d2h",
        "validation-expected-page-concatenation",
        "validation-host-tensor-compare",
    ]
    chains = [
        _chain(
            "capture_reclamation",
            capture_labels,
            grouped,
            logical,
            barriers=[
                "branch quiescence before source read",
                "D2H and integrity publication before source release",
                "release confirmation before GPU1 serving enable",
            ],
            optional=[
                "publish re-verification of unchanged payload",
                "per-materialization synchronization within packing",
            ],
            buffers=[
                "source gather/contiguous",
                "valid-token pages",
                "layer stacks",
                "flat concatenation",
                "integrity and publish tobytes copies",
            ],
        ),
        _chain(
            "restore_and_validation",
            restore_labels,
            grouped,
            logical,
            barriers=[
                "transport integrity before use",
                "H2D before native conversion",
                "fresh allocation before native write",
                "native write before exact destination readback",
                "first-subset validation before shared-root publication",
                "all subset validation before scheduler admission",
            ],
            optional=[
                "second import verification of already-verified immutable payload",
                "host readback boundary if exact device proof is substituted",
            ],
            buffers=[
                "duplicate import tobytes",
                "zero pages",
                "native contiguous/stack",
                "validation gather/unpage/stack/concat",
                "validation recapture and expected concatenation",
            ],
        ),
    ]

    rss_rows = [
        {
            "sample_trigger_monotonic_ns": row["sample_trigger_monotonic_ns"],
            "rss_bytes": row["host_sample"]["processes"][0]["rss_bytes"],
            "cpu_user_ns": row["host_sample"]["processes"][0]["cpu_user_ns"],
            "gpu_utilization_percent": row["gpu_samples"][0]["gpu_utilization_percent"],
        }
        for row in resources["sampling"]["samples"]
        if 140_900_000_000 <= row["sample_trigger_monotonic_ns"] <= 142_700_000_000
    ]
    audit = {
        "schema_version": "sloforge.branchfabric.v9-movement-audit/v1",
        "attempt_id": "exp004-pilot-naive-s41-v9",
        "offline_only": True,
        "conclusion": "The raw 54x ledger is duplicate-free internally but noncanonical and incomplete: it mixes 51x endpoint touches with 3x link traversal while omitting a real redundant 4x restore-import verification. Corrected memory touch is 55x; explicit full read/write/link work is 58x.",
        "logical_state_bytes": logical,
        "recorded_state_pass_count": len(report["passes"]),
        "inferred_missing_state_pass_count": len(inferred),
        "audited_state_pass_count": len(combined),
        "recorded_movement_edge_count": int(accounting["movement_edge_count"]),
        "audited_movement_edge_count": int(accounting["movement_edge_count"]) + 36,
        "duplicate_event_ids": [],
        "d2h_accounting": d2h,
        "movement_metrics": formulas,
        "temporary_memory_attribution": formulas["temporary_memory"],
        "grouped_physical_passes": grouped,
        "physical_state_pass_graph": per_record,
        "fusible_chains": chains,
        "missing_import_verification": {
            "code_path": "python/sloforge/continuum/adapters/vllm_reclamation.py:1153",
            "implementation": "CanonicalKvTransportState.verify at lines 519-538",
            "start_ns": inferred[0]["start_ns"],
            "end_ns": inferred[0]["end_ns"],
            "bounded_wall_ns": inferred[0]["end_ns"] - inferred[0]["start_ns"],
            "endpoint_bytes": missing_endpoint,
            "checksum_work_bytes": logical * 2,
            "temporary_allocation_bytes": logical,
            "resource_samples": rss_rows,
        },
        "semantic_trace_bookkeeping": {
            "source": "rollout.result.json minimal_trace_events and phase_events",
            "used_in_movement_numerator": False,
            "explanation": "Repeated logical/physical byte annotations on paired BranchWorkloadTrace and StateOperationTrace phase markers are semantic provenance, not physical passes.",
        },
        "independent_review": {
            "reviewers": [
                "state-pass graph",
                "movement formula",
                "D2H",
                "independent reproduction",
            ],
            "material_agreement": True,
            "agreed_memory_touch_amplification": 55.0,
            "agreed_external_movement_amplification": 3.0,
            "agreed_d2h_duplicate_accounting_bytes": 0,
        },
        "provenance": {
            "rollout_result": {
                "path": str(ROLLOUT_PATH.relative_to(ROOT)),
                "sha256": EXPECTED_SHA256[ROLLOUT_PATH],
                "sample_selector": "$.movement_report",
            },
            "cuda_and_host_operations": {
                "path": str(TELEMETRY_PATH.relative_to(ROOT)),
                "sha256": EXPECTED_SHA256[TELEMETRY_PATH],
                "sample_selector": "$",
            },
            "resource_sampling": {
                "path": str(RESOURCE_PATH.relative_to(ROOT)),
                "sha256": _sha256(RESOURCE_PATH),
                "sample_selector": "$.sampling.samples[*]",
            },
            "source_paths": [
                "experiments/branchfabric/gpu_reclamation_worker.py",
                "python/sloforge/continuum/adapters/vllm_reclamation.py",
                "python/sloforge/helix/characterization/gpu_reclamation_accounting.py",
                "python/sloforge/helix/characterization/gpu_reclamation_pilot.py",
            ],
        },
    }

    markdown = render_markdown(audit)
    return audit, d2h, formulas, markdown


def render_markdown(audit: dict[str, Any]) -> str:
    formulas = audit["movement_metrics"]
    lines = [
        "# v9 Movement Amplification Audit",
        "",
        "This audit is entirely offline and is pinned to immutable v9 raw-artifact hashes.",
        "",
        "## Result",
        "",
        audit["conclusion"],
        "",
        "| Metric | Bytes | Amplification |",
        "|---|---:|---:|",
        f"| Legacy composite (noncanonical) | {formulas['legacy_reported']['numerator_bytes']:,} | {formulas['legacy_reported']['amplification']:.1f}\u00d7 |",
        f"| Corrected memory endpoint touches | {formulas['memory_touch_only']['numerator_bytes']:,} | {formulas['memory_touch_only']['amplification']:.1f}\u00d7 |",
        f"| External D2H + H2D movement | {formulas['external_movement']['numerator_bytes']:,} | {formulas['external_movement']['amplification']:.1f}\u00d7 |",
        f"| Full read + write + link work | {formulas['full_physical_touch']['numerator_bytes']:,} | {formulas['full_physical_touch']['amplification']:.1f}\u00d7 |",
        f"| Conservative avoidable endpoint work | {formulas['avoidable_movement']['numerator_bytes']:,} | {formulas['avoidable_movement']['amplification']:.1f}\u00d7 |",
        f"| Critical-path read + write + link work | {formulas['critical_path_movement']['full_read_write_move_numerator_bytes']:,} | {formulas['critical_path_movement']['full_read_write_move_amplification']:.1f}\u00d7 |",
        "",
        "The legacy numerator has no duplicate event IDs. Its problem is dimensional: it combines endpoint reads/writes and link traversal into one score, then misses a second restore-time verification. The corrected memory-only metric is 55\u00d7. If the requested full-touch definition literally includes bytes read, bytes written, and bytes moved over PCIe, the result is 58\u00d7; external movement remains separately reported as 3\u00d7.",
        "",
        "## Exact 2\u00d7 D2H explanation",
        "",
        "The 2,113,929,216 D2H bytes are three real synchronous CUDA copies: 1,056,964,608 bytes for checkpoint capture, 954,204,160 bytes for validation of the shared root plus branch 0, and 102,760,448 bytes for validation of branches 1-7. The validation subsets are disjoint and sum to one logical state. K and V are already included in the 1,056,964,608-byte denominator. Duplicate D2H accounting is zero.",
        "",
        "## Missing physical pass",
        "",
        "`Vllm0230RestoreStager.import_group()` calls `state.verify()` after the worker already completed and recorded a full restore verification. The immutable payload is unchanged. This unrecorded verification adds one pinned-host read, one pageable-host write, one aggregate-hash read, and one set of page-hash reads: 4\u00d7 logical bytes of real endpoint work, 2\u00d7 checksum work, and one logical payload of transient allocation. The raw import-to-first-allocation gap and RSS drop independently corroborate the source-path reconstruction.",
        "",
        "## Physical pass groups",
        "",
        "| Label | Stage | Processor | R | W | Link | Temp | Required | Artifact | Fusion |",
        "|---|---|---|---:|---:|---:|---:|---|---|---|",
    ]
    for row in audit["grouped_physical_passes"]:
        lines.append(
            f"| `{row['label']}` | {row['stage']} | {row['processor']} | {row['bytes_read']:,} | {row['bytes_written']:,} | {row['link_transfer_bytes']:,} | {row['temporary_allocation_bytes']:,} | {str(row['required']).lower()} | {str(row['implementation_artifact']).lower()} | {row['software_fusion']} |"
        )
    lines.extend(
        [
            "",
            "Allocation-lifetime records are semantic bookkeeping and do not contribute movement bytes. The JSON audit contains the required classification for every recorded and inferred segment-level pass.",
            "",
            "## Fusible chains",
            "",
            "Capture: `READ_NATIVE → CONTIGUOUS → UNPAGE → STACK → CONCAT → D2H → MATERIALIZE/HASH → PUBLISH_VERIFY`. The GPU packing steps are software-fusible; canonical output can be streamed to D2H; hash generation can consume that stream; publish re-verification is removable.",
            "",
            "Restore: `VERIFY → H2D → DUPLICATE_VERIFY → ZERO/OVERLAY → CONTIGUOUS → STACK → NATIVE_WRITE → READBACK → REPACK → D2H → EXPECTED_CONCAT → COMPARE`. The duplicate verify is removable. Full v9 pages make zero-fill removable. Packing and native write are software-fusible. The write/readback barrier is mandatory for the current proof, while device-side exact proof could remove validation D2H and both host validation buffers.",
            "",
            "No fusion or preservation optimization was implemented in this task.",
            "",
            "## Provenance",
            "",
            f"- `{audit['provenance']['rollout_result']['path']}` (`{audit['provenance']['rollout_result']['sha256']}`)",
            f"- `{audit['provenance']['cuda_and_host_operations']['path']}` (`{audit['provenance']['cuda_and_host_operations']['sha256']}`)",
            "- `experiments/branchfabric/gpu_reclamation_worker.py`",
            "- `python/sloforge/continuum/adapters/vllm_reclamation.py`",
            "- `python/sloforge/helix/characterization/gpu_reclamation_accounting.py`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify committed outputs only")
    args = parser.parse_args()
    audit, d2h, formulas, markdown = build_audit()
    outputs: dict[Path, bytes] = {
        EXPERIMENT_ROOT / "analysis/v9-movement-audit.json": _canonical_bytes(audit),
        EXPERIMENT_ROOT / "analysis/v9-d2h-audit.json": _canonical_bytes(d2h),
        EXPERIMENT_ROOT / "analysis/movement-formulas.json": _canonical_bytes(formulas),
        ROOT / "docs/branchfabric/V9_MOVEMENT_AMPLIFICATION_AUDIT.md": markdown.encode(),
    }
    if args.check:
        for path, expected in outputs.items():
            if not path.is_file() or path.read_bytes() != expected:
                raise ValueError(f"offline audit output is missing or stale: {path}")
    else:
        _atomic_write(EXPERIMENT_ROOT / "analysis/v9-movement-audit.json", audit)
        _atomic_write(EXPERIMENT_ROOT / "analysis/v9-d2h-audit.json", d2h)
        _atomic_write(EXPERIMENT_ROOT / "analysis/movement-formulas.json", formulas)
        _atomic_write_text(ROOT / "docs/branchfabric/V9_MOVEMENT_AMPLIFICATION_AUDIT.md", markdown)
    print(
        json.dumps(
            {
                "status": "ok",
                "check": args.check,
                "outputs": [str(path.relative_to(ROOT)) for path in outputs],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
