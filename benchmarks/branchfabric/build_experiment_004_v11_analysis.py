#!/usr/bin/env python3
"""Build the offline Experiment 004 v11 pass audit from frozen v10 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from itertools import pairwise
from pathlib import Path
from typing import Any

PRIMARY_CLASSIFICATION = {
    "capture-source-native-read": "SEMANTICALLY_REQUIRED",
    "capture-native-axis-contiguous": "SOFTWARE_IMPLEMENTATION_ARTIFACT",
    "capture-concatenate-pages": "SOFTWARE_IMPLEMENTATION_ARTIFACT",
    "capture-stack-layers": "SOFTWARE_IMPLEMENTATION_ARTIFACT",
    "capture-unpage-valid-tokens": "SOFTWARE_IMPLEMENTATION_ARTIFACT",
    "capture-d2h": "SEMANTICALLY_REQUIRED",
    "capture-pinned-transport-lifetime": "SEMANTICALLY_REQUIRED",
    "capture-integrity-hash-reads": "CORRECTNESS_REQUIRED_BUT_REIMPLEMENTABLE",
    "capture-integrity-manifest": "CORRECTNESS_REQUIRED_BUT_REIMPLEMENTABLE",
    "transport-publish-hash-reads": "SOFTWARE_IMPLEMENTATION_ARTIFACT",
    "transport-publish-validation": "SOFTWARE_IMPLEMENTATION_ARTIFACT",
    "restore-transport-hash-reads": "CORRECTNESS_REQUIRED_BUT_REIMPLEMENTABLE",
    "restore-transport-validation": "CORRECTNESS_REQUIRED_BUT_REIMPLEMENTABLE",
    "restore-h2d": "SEMANTICALLY_REQUIRED",
    "restore-import-hash-reads": "SOFTWARE_IMPLEMENTATION_ARTIFACT",
    "restore-import-validation": "SOFTWARE_IMPLEMENTATION_ARTIFACT",
    "restore-native-axis-contiguous": "SOFTWARE_IMPLEMENTATION_ARTIFACT",
    "restore-overlay-valid-tokens": "SOFTWARE_IMPLEMENTATION_ARTIFACT",
    "restore-stack-native-pages": "SOFTWARE_IMPLEMENTATION_ARTIFACT",
    "restore-zero-native-pages": "SOFTWARE_IMPLEMENTATION_ARTIFACT",
    "restore-destination-native-write": "SEMANTICALLY_REQUIRED",
    "validation-destination-native-read": "CORRECTNESS_REQUIRED_BUT_REIMPLEMENTABLE",
    "validation-native-axis-contiguous": "SOFTWARE_IMPLEMENTATION_ARTIFACT",
    "validation-concatenate-pages": "SOFTWARE_IMPLEMENTATION_ARTIFACT",
    "validation-stack-layers": "SOFTWARE_IMPLEMENTATION_ARTIFACT",
    "validation-unpage-valid-tokens": "SOFTWARE_IMPLEMENTATION_ARTIFACT",
    "validation-d2h": "SOFTWARE_IMPLEMENTATION_ARTIFACT",
    "validation-recapture-host-lifetime": "SOFTWARE_IMPLEMENTATION_ARTIFACT",
    "validation-expected-page-concatenation": "SOFTWARE_IMPLEMENTATION_ARTIFACT",
    "validation-host-tensor-compare": "CORRECTNESS_REQUIRED_BUT_REIMPLEMENTABLE",
}

EXPECTED_EXECUTION_COMMIT = "c5fa2f169e8a343ef7a86ebc814b3a23a7de6b39"
EXPECTED_ANALYSIS_COMMIT = "1c51853e10809686d4368037153927f25e834117"
EXPECTED_SOURCE_HASHES = {
    "baseline": "8f5aadff659bef6b60881d08cb7e48d6fd9f992362899733a7f016c80380b03e",
    "movement": "ad31263ad118a2107cb9d8806b06052866144a63e7e79b2af27240ca5f5939f3",
    "graph": "a70d1f554201bab696084e055310bbbb6a85745740845def93a4cd8a87e52dfa",
    "telemetry": "edf6958f6ad95009693223cc451eddc4abb75d2c91054b9ba2fd265c347875c7",
}

REPLACEMENT_CONTRACT = {
    "CORRECTNESS_REQUIRED_BUT_REIMPLEMENTABLE": (
        "retain the invariant with independently tested streamed or logical-domain evidence"
    ),
    "SOFTWARE_IMPLEMENTATION_ARTIFACT": (
        "remove only through bitwise-equivalent bounded conversion or validation"
    ),
}

SERIAL_PREFIX = [
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
    "restore-transport-validation",
    "restore-transport-hash-reads",
    "restore-h2d",
    "restore-import-validation",
    "restore-import-hash-reads",
]

SUBSET_SERIAL = [
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

CORRECTNESS_PURPOSE = {
    "capture-source-native-read": "preserve every logically referenced native KV byte",
    "capture-d2h": "move the canonical checkpoint to reclaimable host ownership",
    "capture-pinned-transport-lifetime": "retain checkpoint bytes while GPU1 is reclaimed",
    "capture-integrity-hash-reads": "bind every page and the complete payload to SHA-256",
    "capture-integrity-manifest": "publish logical ranges, ownership, identity, and SHA-256 evidence",
    "transport-publish-hash-reads": "recompute payload/page SHA-256 before publication",
    "transport-publish-validation": "reject malformed or corrupted transport before source release",
    "restore-transport-hash-reads": "recompute payload/page SHA-256 before restore",
    "restore-transport-validation": "reject corrupted transport before destination admission",
    "restore-h2d": "make canonical bytes available to the destination GPU",
    "restore-import-hash-reads": "repeat transport verification inside the import boundary",
    "restore-import-validation": "repeat manifest and payload validation before allocation",
    "restore-destination-native-write": "populate fresh vLLM native allocations",
    "restore-zero-native-pages": (
        "initialize unreadable page bytes; frozen v10 pages were all full and then overwritten"
    ),
    "validation-destination-native-read": "observe restored logical bytes before visibility",
    "validation-d2h": "return restored bytes to the host for exact comparison",
    "validation-host-tensor-compare": "prove restored logical bytes exactly equal the checkpoint",
}


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_frozen_provenance(
    root: Path,
    *,
    baseline_path: Path,
    graph_path: Path,
    movement_path: Path,
    telemetry_path: Path,
) -> dict[str, Any]:
    paths = {
        "baseline": baseline_path,
        "movement": movement_path,
        "graph": graph_path,
        "telemetry": telemetry_path,
    }
    actual_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if actual_hashes != EXPECTED_SOURCE_HASHES:
        raise RuntimeError("frozen v10 source artifact hash mismatch")
    baseline = json.loads(baseline_path.read_text())
    stored_artifact_hash = baseline.get("artifact_hash")
    unsealed = dict(baseline)
    unsealed.pop("artifact_hash", None)
    if stored_artifact_hash != hashlib.sha256(_canonical_bytes(unsealed)).hexdigest():
        raise RuntimeError("frozen v10 baseline self-hash mismatch")
    if (
        baseline.get("repository_commit") != EXPECTED_EXECUTION_COMMIT
        or baseline.get("analysis_commit") != EXPECTED_ANALYSIS_COMMIT
    ):
        raise RuntimeError("frozen v10 baseline commit identity mismatch")
    for commit in (EXPECTED_EXECUTION_COMMIT, EXPECTED_ANALYSIS_COMMIT):
        subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            cwd=root,
            check=True,
            capture_output=True,
        )
    subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            EXPECTED_EXECUTION_COMMIT,
            EXPECTED_ANALYSIS_COMMIT,
        ],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return {
        "verified": True,
        "source_sha256": actual_hashes,
        "baseline_self_hash_verified": True,
        "commit_objects_verified": True,
        "execution_is_analysis_ancestor": True,
    }


def _memory_tier(memory: str) -> str:
    if memory in {"source_gpu_native_paged", "destination_gpu_native_paged"}:
        return "GPU_HBM_NATIVE_POOL"
    if memory.startswith("gpu_"):
        return "GPU_HBM_TEMPORARY"
    if memory == "pinned_host_transport":
        return "HOST_PINNED"
    if memory == "pageable_host_buffer":
        return "HOST_PAGEABLE"
    return "NONE"


def _representation(memory: str) -> str:
    return {
        "source_gpu_native_paged": "VLLM_NATIVE_PAGED_LAYOUT",
        "destination_gpu_native_paged": "VLLM_NATIVE_PAGED_LAYOUT",
        "gpu_transform_buffer": "NAIVE_GPU_TRANSFORM_INTERMEDIATE",
        "gpu_transport_buffer": "CONTINUUM_CANONICAL_TRANSPORT_LAYOUT",
        "pinned_host_transport": "CONTINUUM_CANONICAL_TRANSPORT_LAYOUT",
        "pageable_host_buffer": "NAIVE_HOST_VERIFICATION_INTERMEDIATE",
        "none": "NONE",
    }[memory]


def _secondary_tags(pass_id: str) -> list[str]:
    tags: list[str] = []
    if pass_id not in {
        "capture-pinned-transport-lifetime",
        "restore-destination-native-write",
    }:
        tags.append("POTENTIALLY_STREAMABLE")
    if any(
        token in pass_id
        for token in ("axis", "concatenate", "stack", "unpage", "overlay", "hash", "validation")
    ):
        tags.append("POTENTIALLY_FUSIBLE")
    return tags


def build(root: Path) -> dict[str, Any]:
    experiment_root = root / "artifacts/branchfabric/gpu-validation/experiment-004"
    baseline_path = experiment_root / "baseline/branchfabric-exp004-naive-baseline-v10.json"
    graph_path = experiment_root / "state-passes/v10-state-pass-graph.json"
    movement_path = experiment_root / "movement-accounting.json"
    telemetry_path = (
        experiment_root
        / "raw/modal/exp004-v10-naive-s41-v7/rollout/telemetry/cuda-and-host-operations.json"
    )
    provenance = _verify_frozen_provenance(
        root,
        baseline_path=baseline_path,
        graph_path=graph_path,
        movement_path=movement_path,
        telemetry_path=telemetry_path,
    )
    source_graph = json.loads(graph_path.read_text())
    movement = json.loads(movement_path.read_text())
    telemetry = json.loads(telemetry_path.read_text())
    records = movement["state_pass_records"]
    if len(source_graph.get("nodes", ())) != 30 or len(records) != 270:
        raise RuntimeError("frozen v10 graph or segment-record cardinality changed")
    cuda_by_id = {item["operation_id"]: item for item in telemetry["cuda_operations"]}
    by_stage = {
        stage: [item for item in records if item["stage"] == stage]
        for stage in PRIMARY_CLASSIFICATION
    }

    def pass_row(
        pass_id: str, selected: list[dict[str, Any]], *, execution_id: str
    ) -> dict[str, Any]:
        if not selected:
            raise RuntimeError(f"frozen graph node lacks StatePassRecord evidence: {pass_id}")
        source_memories = {str(item["source_memory"]) for item in selected}
        destination_memories = {str(item["destination_memory"]) for item in selected}
        operations = {str(item["operation"]) for item in selected}
        if len(source_memories) != 1 or len(destination_memories) != 1 or len(operations) != 1:
            raise RuntimeError(f"pass group is not representation-homogeneous: {execution_id}")
        source_memory = next(iter(source_memories))
        destination_memory = next(iter(destination_memories))
        operation = next(iter(operations))
        intervals = sorted({(int(item["start_ns"]), int(item["end_ns"])) for item in selected})
        allocation_lifetime = pass_id in {
            "capture-pinned-transport-lifetime",
            "validation-recapture-host-lifetime",
        }
        cuda_ids = (
            []
            if allocation_lifetime
            else sorted({item for record in selected for item in record["cuda_operation_ids"]})
        )
        cuda_operations = [cuda_by_id[item] for item in cuda_ids]
        processors = sorted({record["processor"] for record in selected})
        primary = PRIMARY_CLASSIFICATION[pass_id]
        bytes_read = sum(int(item["bytes_read"]) for item in selected)
        bytes_written = sum(int(item["bytes_written"]) for item in selected)
        transfer_bytes = sum(int(item["transfer_bytes"]) for item in selected)
        logical_bytes = sum(int(item["logical_bytes"]) for item in selected)
        logical_segments = sorted({str(item["state_segment"]) for item in selected})
        return {
            "pass_id": execution_id,
            "aggregate_pass_id": pass_id,
            "stage": pass_id.split("-", 1)[0],
            "operation": operation,
            "source_representation": _representation(source_memory),
            "destination_representation": _representation(destination_memory),
            "source_memory_tier": _memory_tier(source_memory),
            "destination_memory_tier": _memory_tier(destination_memory),
            "logical_segments": logical_segments,
            "logical_bytes": logical_bytes,
            "bytes_read": bytes_read,
            "bytes_written": bytes_written,
            "transfer_bytes": transfer_bytes,
            "physical_touch_bytes": bytes_read + bytes_written + transfer_bytes,
            "temporary_bytes": sum(int(item["temporary_allocation_bytes"]) for item in selected),
            "wall_time_ns": sum(end - start for start, end in intervals),
            "measured_intervals": [
                {"start_ns": start, "end_ns": end, "duration_ns": end - start}
                for start, end in intervals
            ],
            "timing_shared": any(
                any(
                    other != pass_id
                    and {(int(item["start_ns"]), int(item["end_ns"])) for item in by_stage[other]}
                    & set(intervals)
                    for other in PRIMARY_CLASSIFICATION
                )
                for _interval in intervals
            ),
            "cpu_time_ns": None,
            "cpu_time_semantics": "process CPU time was not instrumented per v10 pass",
            "cuda_time_ns": (
                sum(int(item["cuda_event_elapsed_ns"]) for item in cuda_operations)
                if cuda_operations
                else None
            ),
            "cuda_operation_ids": cuda_ids,
            "synchronization": {
                "explicit_wait_ns": sum(
                    int(item["synchronization_wait_ns"]) for item in cuda_operations
                ),
                "stream_ids": sorted({item["stream_id"] for item in cuda_operations}),
                "semantics": (
                    "synchronous default-stream operation; CUDA event recorder waited for completion"
                    if cuda_operations
                    else "CPU pass or allocation lifetime; no CUDA synchronization attributed"
                ),
            },
            "processors": processors,
            "coverage": (
                "FULL_STATE"
                if logical_bytes == movement["logical_state_bytes"]
                else "PARTIAL_STATE"
            ),
            "correctness_purpose": CORRECTNESS_PURPOSE.get(
                pass_id,
                "naive layout materialization; no independent semantic invariant",
            ),
            "primary_classification": primary,
            "secondary_tags": _secondary_tags(pass_id),
            "required_or_avoidable": (
                "REQUIRED" if primary == "SEMANTICALLY_REQUIRED" else "REIMPLEMENTABLE_OR_AVOIDABLE"
            ),
            "fusibility": selected[0]["fusibility_classification"],
            "raw_record_ids": sorted(item["record_id"] for item in selected),
            "raw_evidence": {
                "graph": str(graph_path.relative_to(root)),
                "movement": str(movement_path.relative_to(root)),
                "telemetry": str(telemetry_path.relative_to(root)),
            },
        }

    passes = [
        pass_row(stage, by_stage[stage], execution_id=stage) for stage in PRIMARY_CLASSIFICATION
    ]

    execution_passes: list[dict[str, Any]] = []
    for stage in PRIMARY_CLASSIFICATION:
        interval_groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for record in by_stage[stage]:
            interval_groups.setdefault((int(record["start_ns"]), int(record["end_ns"])), []).append(
                record
            )
        ordered = sorted(interval_groups.items())
        for index, ((_start_ns, _end_ns), selected) in enumerate(ordered):
            execution_id = stage if len(ordered) == 1 else f"{stage}:subset-{index:02d}"
            execution_passes.append(pass_row(stage, selected, execution_id=execution_id))

    timing_groups: dict[tuple[tuple[int, int], ...], list[dict[str, Any]]] = {}
    for item in execution_passes:
        signature = tuple(
            (interval["start_ns"], interval["end_ns"]) for interval in item["measured_intervals"]
        )
        timing_groups.setdefault(signature, []).append(item)
    for timing_group_index, (_signature, members) in enumerate(
        sorted(timing_groups.items()), start=1
    ):
        latency_members = [
            item
            for item in members
            if item["aggregate_pass_id"]
            not in {
                "capture-pinned-transport-lifetime",
                "validation-recapture-host-lifetime",
            }
        ]
        owner = latency_members[0] if latency_members else None
        for item in members:
            item["timing_group_id"] = f"measured-operation-{timing_group_index:02d}"
            item["timing_owner_pass_id"] = owner["pass_id"] if owner is not None else None
            item["timing_role"] = (
                "OWNER"
                if item is owner
                else "SHARED_ACCOUNTING_SUBPASS"
                if owner is not None
                else "OWNERSHIP_LIFETIME_ONLY"
            )
            item["critical_path_wall_time_ns"] = item["wall_time_ns"] if item is owner else 0
            item["critical_path_cuda_time_ns"] = item["cuda_time_ns"] if item is owner else 0

    expected_subset_bytes = (954_204_160, 102_760_448)
    split_stages = (*SUBSET_SERIAL, "validation-recapture-host-lifetime")
    for stage in split_stages:
        rows = [item for item in execution_passes if item["aggregate_pass_id"] == stage]
        if len(rows) != 2 or tuple(item["logical_bytes"] for item in rows) != expected_subset_bytes:
            raise RuntimeError(f"frozen v10 subset partition changed for {stage}")
    subset_derivation = {
        "validated": True,
        "subset_00_logical_bytes": expected_subset_bytes[0],
        "subset_01_logical_bytes": expected_subset_bytes[1],
        "sum_logical_bytes": sum(expected_subset_bytes),
        "expected_logical_state_bytes": movement["logical_state_bytes"],
        "ordering": "root+branch0 write/validate completes before remaining private pages",
    }
    if sum(expected_subset_bytes) != movement["logical_state_bytes"]:
        raise RuntimeError("frozen v10 restore subsets do not conserve logical bytes")

    exact_edges: list[dict[str, Any]] = []
    execution_by_id = {item["pass_id"]: item for item in execution_passes}

    def edge(source: str, destination: str, kind: str = "serial_data_dependency") -> None:
        if (
            kind == "serial_data_dependency"
            and execution_by_id[source]["timing_group_id"]
            == execution_by_id[destination]["timing_group_id"]
        ):
            kind = "semantic_order_within_shared_measured_operation"
        exact_edges.append(
            {
                "source": source,
                "destination": destination,
                "kind": kind,
                "contributes_to_latency_path": kind
                in {"serial_data_dependency", "control_dependency_with_intervening_allocation"},
            }
        )

    for source, destination in pairwise(SERIAL_PREFIX):
        edge(source, destination)
    edge("capture-d2h", "capture-pinned-transport-lifetime", "ownership_lifetime_begins")
    edge(
        "capture-pinned-transport-lifetime",
        "restore-transport-validation",
        "checkpoint_must_remain_live_not_serial_latency",
    )
    previous = SERIAL_PREFIX[-1]
    for subset_index in range(2):
        suffix = f":subset-{subset_index:02d}"
        subset_nodes = [f"{stage}{suffix}" for stage in SUBSET_SERIAL]
        edge(
            previous,
            subset_nodes[0],
            "control_dependency_with_intervening_allocation"
            if subset_index
            else "serial_data_dependency",
        )
        for source, destination in pairwise(subset_nodes):
            edge(source, destination)
        lifetime = f"validation-recapture-host-lifetime{suffix}"
        d2h = f"validation-d2h{suffix}"
        compare = f"validation-host-tensor-compare{suffix}"
        edge(d2h, lifetime, "ownership_lifetime_begins")
        edge(lifetime, compare, "buffer_must_remain_live_not_serial_latency")
        previous = compare

    incoming: dict[str, list[str]] = {item["pass_id"]: [] for item in execution_passes}
    outgoing: dict[str, list[str]] = {item["pass_id"]: [] for item in execution_passes}
    for item in exact_edges:
        incoming[item["destination"]].append(item["source"])
        outgoing[item["source"]].append(item["destination"])
        if item["contributes_to_latency_path"]:
            source_end = max(
                interval["end_ns"]
                for interval in execution_by_id[item["source"]]["measured_intervals"]
            )
            destination_start = min(
                interval["start_ns"]
                for interval in execution_by_id[item["destination"]]["measured_intervals"]
            )
            if source_end > destination_start:
                raise RuntimeError(
                    "latency-contributing DAG edge is not timestamp-monotone: "
                    f"{item['source']} -> {item['destination']}"
                )
    for item in execution_passes:
        item["dependency_edges"] = {
            "predecessors": incoming[item["pass_id"]],
            "successors": outgoing[item["pass_id"]],
        }

    candidates = [
        item for item in passes if item["primary_classification"] != "SEMANTICALLY_REQUIRED"
    ]
    candidates.sort(
        key=lambda item: (-item["physical_touch_bytes"], -item["wall_time_ns"], item["pass_id"])
    )
    ranking = [
        {
            "rank": rank,
            "pass_id": item["pass_id"],
            "primary_classification": item["primary_classification"],
            "physical_touch_bytes": item["physical_touch_bytes"],
            "physical_touch_amplification": item["physical_touch_bytes"]
            / movement["logical_state_bytes"],
            "measured_interval_union_ns": item["wall_time_ns"],
            "disposition": (
                "REIMPLEMENT_WITH_EQUIVALENT_PROOF"
                if item["primary_classification"] == "CORRECTNESS_REQUIRED_BUT_REIMPLEMENTABLE"
                else "REMOVE_BY_BITWISE_EQUIVALENT_PIPELINE"
            ),
            "replacement_contract": REPLACEMENT_CONTRACT[item["primary_classification"]],
            "overlap_warning": "interval may be shared by multiple ledger passes; do not sum as latency",
        }
        for rank, item in enumerate(candidates, 1)
    ]
    output = {
        "schema_version": "sloforge.branchfabric.experiment-004-v11-naive-pass-graph/v1",
        "status": "OFFLINE_RECONSTRUCTION_COMPLETE",
        "provenance_verification": provenance,
        "frozen_baseline": {
            "execution_commit": "c5fa2f169e8a343ef7a86ebc814b3a23a7de6b39",
            "analysis_commit": "1c51853e10809686d4368037153927f25e834117",
            "attempt_id": "exp004-v10-naive-s41-v7",
            "logical_state_bytes": movement["logical_state_bytes"],
            "full_physical_touch_bytes": movement["full_physical_touch_bytes"],
            "full_physical_touch_amplification": movement["amplification"]["full_physical_touch"],
            "external_movement_bytes": movement["external_movement_bytes"],
            "avoidable_movement_bytes": movement["avoidable_movement_bytes"],
        },
        "classification_vocabulary": [
            "SEMANTICALLY_REQUIRED",
            "CORRECTNESS_REQUIRED_BUT_REIMPLEMENTABLE",
            "SOFTWARE_IMPLEMENTATION_ARTIFACT",
            "RUNTIME_API_ARTIFACT",
            "MEASUREMENT_ONLY",
            "POTENTIALLY_STREAMABLE",
            "POTENTIALLY_FUSIBLE",
            "UNKNOWN",
        ],
        "aggregate_full_state_pass_count": len(passes),
        "raw_segment_record_count": len(records),
        "passes": passes,
        "execution_pass_count": len(execution_passes),
        "execution_passes": execution_passes,
        "restore_subset_derivation": subset_derivation,
        "edges": exact_edges,
        "optimization_candidate_ranking": ranking,
        "digest_convention": (
            "sha256 of canonical compact sorted-key JSON plus newline before dag_sha256 is added"
        ),
        "measurement_limits": [
            "v10 did not collect process CPU time per state pass",
            "several ledger passes intentionally share one measured operation interval",
            "CUDA event time is attributed once per aggregate node and must not be summed across raw segments",
            "allocation lifetime edges express ownership overlap and are not serial latency",
        ],
    }
    if (
        sum(item["physical_touch_bytes"] for item in passes)
        != movement["full_physical_touch_bytes"]
    ):
        raise RuntimeError("aggregate pass graph does not conserve frozen physical touch bytes")
    output["dag_sha256"] = hashlib.sha256(_canonical_bytes(output)).hexdigest()
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/branchfabric/gpu-validation/experiment-004/v11-analysis/naive-pass-graph.json"
        ),
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else args.repository_root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build(args.repository_root), indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
