from __future__ import annotations

import hashlib
from typing import Any

from sloforge.helix.characterization.gpu_reclamation import ExperimentPhase
from sloforge.helix.characterization.gpu_reclamation_accounting import (
    LogicalStateSegment,
    MemoryDomain,
    StatePassOperation,
    StatePassRecord,
    StateSegmentKind,
    TransferDirection,
    build_state_movement_report,
)
from sloforge.helix.characterization.gpu_reclamation_pilot import (
    assess_pilot_directory,
    assess_pilot_payloads,
)

GPU0 = "GPU-serving-1"
GPU1 = "GPU-rollout-2"
NS = 1_000_000_000


def _token_hash(tokens: tuple[int, ...]) -> str:
    return hashlib.sha256(b"".join(item.to_bytes(8, "little") for item in tokens)).hexdigest()


def _manifest() -> dict[str, Any]:
    branches = tuple(f"branch.{index}" for index in range(8))
    pages = [
        {
            "logical_page_id": "page.shared",
            "logical_token_start": 0,
            "valid_tokens": 1,
            "payload_offset_bytes": 0,
            "payload_bytes": 4,
            "content_sha256": "a" * 64,
            "branch_ids": branches,
            "shared_root": True,
        }
    ]
    for index, branch in enumerate(branches):
        pages.append(
            {
                "logical_page_id": f"page.private.{index}",
                "logical_token_start": 1,
                "valid_tokens": 1,
                "payload_offset_bytes": 4 * (index + 1),
                "payload_bytes": 4,
                "content_sha256": "a" * 64,
                "branch_ids": (branch,),
                "shared_root": False,
            }
        )
    return {
        "schema_version": "sloforge.continuum.vllm-kv-branch-group/v1",
        "runtime": "vllm",
        "runtime_version": "0.23.0",
        "adapter_version": "0.1.0",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "model_revision": "a09a35458c702b33eeacc393d103063234e8bc28",
        "tokenizer_id": "Qwen/Qwen2.5-7B-Instruct",
        "tokenizer_revision": "a09a35458c702b33eeacc393d103063234e8bc28",
        "dtype": "bfloat16",
        "policy_epoch": "policy-1",
        "block_size_tokens": 1,
        "layer_names": ("layer.0",),
        "kv_heads": 1,
        "head_size": 1,
        "element_size_bytes": 2,
        "canonical_layout": "page,layer,token,kv,head,dim",
        "pages": tuple(pages),
        "branches": tuple(
            {
                "logical_branch_id": branch,
                "parent_logical_branch_id": "root",
                "token_ids": (99, index),
                "token_history_sha256": _token_hash((99, index)),
                "computed_tokens": 2,
                "logical_page_ids": ("page.shared", f"page.private.{index}"),
            }
            for index, branch in enumerate(branches)
        ),
        "logical_state_bytes": 36,
        "physical_source_bytes": 36,
        "payload_sha256": "b" * 64,
    }


def _movement() -> dict[str, Any]:
    segments = [
        LogicalStateSegment(
            segment_id="shared",
            branch_group_id="group",
            kind=StateSegmentKind.SHARED_KV,
            logical_bytes=4,
        )
    ]
    segments.extend(
        LogicalStateSegment(
            segment_id=f"private.{index}",
            branch_group_id="group",
            kind=StateSegmentKind.PRIVATE_KV,
            logical_bytes=4,
            branch_id=f"branch.{index}",
        )
        for index in range(8)
    )
    passes: list[StatePassRecord] = []
    operations = (
        (
            StatePassOperation.READ,
            MemoryDomain.SOURCE_GPU_NATIVE_PAGED,
            MemoryDomain.GPU_TRANSFORM_BUFFER,
            TransferDirection.NONE,
        ),
        (
            StatePassOperation.D2H,
            MemoryDomain.GPU_TRANSFORM_BUFFER,
            MemoryDomain.PINNED_HOST_TRANSPORT,
            TransferDirection.D2H,
        ),
        (
            StatePassOperation.H2D,
            MemoryDomain.PINNED_HOST_TRANSPORT,
            MemoryDomain.GPU_TRANSPORT_BUFFER,
            TransferDirection.H2D,
        ),
        (
            StatePassOperation.WRITE,
            MemoryDomain.GPU_TRANSPORT_BUFFER,
            MemoryDomain.DESTINATION_GPU_NATIVE_PAGED,
            TransferDirection.NONE,
        ),
        (
            StatePassOperation.D2H,
            MemoryDomain.DESTINATION_GPU_NATIVE_PAGED,
            MemoryDomain.PINNED_HOST_TRANSPORT,
            TransferDirection.D2H,
        ),
    )
    for clock, (operation, source, destination, direction) in enumerate(operations):
        for segment in segments:
            passes.append(
                StatePassRecord(
                    record_id=f"pass.{len(passes)}",
                    state_segment=segment.segment_id,
                    branch_group="group",
                    operation=operation,
                    source_memory=source,
                    destination_memory=destination,
                    bytes_read=4,
                    bytes_written=4,
                    transfer_direction=direction,
                    transfer_bytes=4 if direction is not TransferDirection.NONE else 0,
                    logical_bytes=4,
                    start_ns=clock,
                    end_ns=clock + 1,
                    device=GPU1,
                    required_unavoidable=operation is StatePassOperation.WRITE,
                )
            )
    return build_state_movement_report(logical_segments=segments, passes=passes).model_dump(
        mode="json"
    )


def _critical_rows(names: tuple[str, ...], start: int) -> list[dict[str, Any]]:
    result = []
    cursor = start
    for name in names:
        result.append({"stage": name, "start_ns": cursor, "end_ns": cursor + 1, "duration_ns": 1})
        cursor += 1
    return result


def _serving_row(
    *, request_id: str, phase: str, arrival: int, gpu1: bool = False
) -> dict[str, Any]:
    admitted = max(arrival + 10_000_000, 7 * NS) if gpu1 else arrival + 10_000_000
    first = admitted + 100_000_000
    timestamps = [first + index * 10_000_000 for index in range(16)]
    return {
        "request_id": request_id,
        "phase": phase,
        "scheduled_arrival_ns": arrival,
        "admitted_ns": admitted,
        "first_token_ns": first,
        "completed_ns": timestamps[-1],
        "token_timestamps_ns": timestamps,
        "output_token_ids": list(range(16)),
    }


def _serving_results() -> tuple[dict[str, Any], dict[str, Any]]:
    gpu0_rows: list[dict[str, Any]] = []
    sequence = 0
    for second in range(4):
        gpu0_rows.append(
            _serving_row(
                request_id=f"gpu0-serving-{sequence:05d}",
                phase="control",
                arrival=second * NS,
            )
        )
        sequence += 1
    for index in range(48):
        gpu0_rows.append(
            _serving_row(
                request_id=f"gpu0-serving-{sequence:05d}",
                phase="spike",
                arrival=4 * NS + index * 250_000_000,
            )
        )
        sequence += 1
    for index in range(3):
        gpu0_rows.append(
            _serving_row(
                request_id=f"gpu0-serving-{sequence:05d}",
                phase="restore-interference",
                arrival=16 * NS + index * NS,
            )
        )
        sequence += 1
    gpu1_rows = [
        _serving_row(
            request_id=f"gpu1-serving-{index:05d}",
            phase="spike",
            arrival=4 * NS + index * 500_000_000,
            gpu1=True,
        )
        for index in range(24)
    ]
    serving = {
        "status": "succeeded",
        "role": "serving",
        "physical_gpu_uuid": GPU0,
        "start_ns": 0,
        "end_ns": 19 * NS + 300_000_000,
        "requests": gpu0_rows,
        "resource_telemetry": {"status": "succeeded"},
    }
    temporary = {
        "start_ns": 4 * NS,
        "end_ns": 16 * NS,
        "requests": gpu1_rows,
    }
    return serving, temporary


def _payloads() -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], tuple[dict[str, Any], ...]
]:
    config = {
        "attempt_id": "pilot-fixture-004",
        "seed": 41,
        "prefix_length": 16_384,
        "fanout": 8,
        "suffix_length": 256,
        "serving_prompt_tokens": 256,
        "serving_output_tokens": 16,
        "serving_slo_maximum_ttft_seconds": 2.0,
        "serving_slo_maximum_inter_token_latency_seconds": 0.5,
        "serving_slo_stability_window_seconds": 1.0,
    }
    serving, temporary = _serving_results()
    raw_phases = (
        ExperimentPhase.HELIX_RECLAIM_TRIGGER,
        ExperimentPhase.ROLLOUT_ADMISSION_STOP,
        ExperimentPhase.BRANCH_QUIESCE,
        ExperimentPhase.STATE_CAPTURE_BEGIN,
        ExperimentPhase.STATE_CAPTURE_END,
        ExperimentPhase.STATE_TRANSFORM_BEGIN,
        ExperimentPhase.STATE_TRANSFORM_END,
        ExperimentPhase.D2H_BEGIN,
        ExperimentPhase.D2H_END,
        ExperimentPhase.INTEGRITY_BEGIN,
        ExperimentPhase.INTEGRITY_END,
        ExperimentPhase.STATE_PUBLISH,
        ExperimentPhase.GPU_STATE_RELEASE_BEGIN,
        ExperimentPhase.GPU_STATE_RELEASE_END,
        ExperimentPhase.HBM_RECLAIM_CONFIRMED,
        ExperimentPhase.SERVING_SECONDARY_ENABLE,
        ExperimentPhase.GPU1_FIRST_SERVING_REQUEST,
        ExperimentPhase.ROLLOUT_RESTORE_TRIGGER,
        ExperimentPhase.H2D_BEGIN,
        ExperimentPhase.H2D_END,
        ExperimentPhase.STATE_IMPORT_BEGIN,
        ExperimentPhase.STATE_VALIDATE_BEGIN,
        ExperimentPhase.STATE_VALIDATE_END,
        ExperimentPhase.STATE_IMPORT_END,
        ExperimentPhase.BRANCH_RESUME_BEGIN,
        ExperimentPhase.FIRST_RESUMED_TOKEN,
        ExperimentPhase.ROLLOUT_RESUME_COMPLETE,
    )
    phase_times: dict[ExperimentPhase, int] = {}
    cursor = 4 * NS
    for phase in raw_phases:
        if phase is ExperimentPhase.GPU1_FIRST_SERVING_REQUEST:
            cursor = 7 * NS + 100_000_000
        elif phase is ExperimentPhase.ROLLOUT_RESTORE_TRIGGER:
            cursor = 16 * NS + 50_000_000
        elif phase is ExperimentPhase.ROLLOUT_RESUME_COMPLETE:
            cursor = 17 * NS + 500_000_000
        else:
            cursor += 1_000_000
        phase_times[phase] = cursor
    manifest = _manifest()
    source_bindings = tuple(
        {
            "logical_page_id": page["logical_page_id"],
            "source": {"gpu_uuid": GPU1, "block_index": index, "allocation_epoch": 1},
        }
        for index, page in enumerate(manifest["pages"])
    )
    expected = {f"branch.{index}": 100 + index for index in range(8)}
    rollout = {
        "status": "succeeded",
        "role": "rollout",
        "physical_gpu_uuid": GPU1,
        "trigger_ns": 4 * NS,
        "logical_state_bytes": 36,
        "kv_logical_state_bytes": 36,
        "logical_state_accounting_scope": (
            "gpu-resident KV payload; required host-resident transport metadata is reported "
            "separately and excluded from movement amplification"
        ),
        "host_resident_transport_metadata_bytes": 1234,
        "helix_environment_state_bytes": 0,
        "helix_environment_state_scope": (
            "bounded model-only greedy rollout has no external environment or trajectory payload"
        ),
        "shared_bytes": 4,
        "private_bytes": 32,
        "physical_source_bytes": 36,
        "branch_count": 8,
        "transport_manifest": manifest,
        "source_capture_evidence": {"bindings": source_bindings},
        "source_memory_before_release": {
            "kv_assigned_bytes": 36,
            "kv_pool_reserved_bytes": 100,
        },
        "source_memory_after_release": {
            "kv_assigned_bytes": 0,
            "kv_pool_reserved_bytes": 100,
        },
        "source_block_release_summary": {
            "all_native_refcounts_zero": True,
            "all_allocator_available": True,
            "all_hashes_cleared": True,
            "allocation_epochs_match_capture": True,
            "full_free_pool_recovered": True,
        },
        "source_allocations": tuple((index, 1) for index in range(9)),
        "destination_allocations": tuple((index, 2) for index in range(9)),
        "destination_page_map": {
            page["logical_page_id"]: index for index, page in enumerate(manifest["pages"])
        },
        "reclamation_stages": _critical_rows(
            (
                "admission_stop",
                "branch_quiesce",
                "final_state_capture",
                "delta_extraction",
                "source_layout_read",
                "state_transform",
                "device_to_host",
                "integrity_generation",
                "transport_publish",
                "runtime_state_release",
                "capacity_reclaim_confirmation",
                "serving_secondary_enable",
                "first_useful_serving_request",
            ),
            4 * NS,
        ),
        "restore_stages": _critical_rows(
            (
                "destination_request_construction",
                "transport_layout_read_validation",
                "host_to_device",
                "destination_allocation_subset",
                "destination_native_write_subset",
                "destination_validation_subset",
                "scheduler_first_forward_and_token",
            ),
            16 * NS + 50_000_000,
        ),
        "phase_events": tuple(
            {
                "phase": phase.value,
                "monotonic_timestamp_ns": phase_times[phase],
                "logical_bytes": 36,
                "physical_bytes": 36,
            }
            for phase in raw_phases
        ),
        "temporary_serving": temporary,
        "first_serving_ns": 7 * NS + 100_000_000,
        "first_resumed_tokens": expected,
        "expected_first_tokens": expected,
        "continuation_token_counts": {f"{branch}@restore-1": 8 for branch in expected},
        "sampling_semantics_by_branch": {
            branch: {
                "effective_seed": 1000 + index,
                "temperature": 0.0,
                "ignore_eos": True,
            }
            for index, branch in enumerate(expected)
        },
        "movement_report": _movement(),
        "operation_telemetry": {
            "status": "succeeded",
            "cuda_summary": {
                "copy_count": 3,
                "bytes_by_kind": {"d2h": 72, "h2d": 36},
                "copy_count_by_kind": {"d2h": 2, "h2d": 1, "d2d": 0},
                "copy_sizes_bytes_by_kind": {
                    "d2h": [36, 36],
                    "h2d": [36],
                    "d2d": [],
                },
            },
            "host_allocation_summary": {
                "allocated_bytes_by_kind": {"pageable": 0, "pinned": 0},
                "peak_live_bytes_by_kind": {"pageable": 0, "pinned": 0},
                "active_bytes_by_kind": {"pageable": 0, "pinned": 0},
            },
        },
        "minimal_trace_events": tuple(
            {"attributes": {"experiment_phase": phase.value}}
            for phase in raw_phases
            for _ in range(2)
        ),
        "transport_integrity_valid": True,
        "source_destination_fresh": True,
        "all_branches_resumed": True,
        "warm_pool_driver_hbm_released": False,
        "resource_telemetry": {"status": "succeeded"},
    }
    controller = {
        "status": "succeeded",
        "inventory_before": (
            {"uuid": GPU0, "name": "NVIDIA A100-SXM4-80GB"},
            {"uuid": GPU1, "name": "NVIDIA A100-SXM4-80GB"},
        ),
        "worker_returncodes": {"serving": 0, "rollout": 0},
        "compute_processes_after": (),
        "cuda_clean_import_audits": tuple({"cuda_clean": True} for _ in range(3)),
    }
    readiness = (
        {"role": "serving", "model_ready_ns": NS, "rollouts_ready": False},
        {"role": "rollout", "model_ready_ns": 2 * NS, "rollouts_ready": True},
    )
    return config, controller, serving, rollout, readiness


def test_valid_raw_pilot_instantiates_contract_and_derives_slo_event() -> None:
    config, controller, serving, rollout, readiness = _payloads()
    assessment = assess_pilot_payloads(
        config=config,
        controller=controller,
        serving=serving,
        rollout=rollout,
        readiness=readiness,
    )

    assert assessment.pilot_valid
    assert assessment.evidence is not None
    assert assessment.meaningful_spike is not None
    assert assessment.meaningful_spike.queue_growth_confirmed
    assert assessment.meaningful_spike.measured_slo_departure_confirmed
    assert assessment.slo_restoration is not None
    assert assessment.slo_restoration.status == "restored"
    assert [item.phase for item in assessment.derived_phase_events] == [
        ExperimentPhase.SERVING_SLO_RESTORED
    ]
    assert len(assessment.derived_trace_events) == 2
    assert assessment.restore_interference_overlap is not None
    assert assessment.restore_interference_overlap.confirmed


def test_non_restored_slo_never_creates_restored_event() -> None:
    config, controller, serving, rollout, readiness = _payloads()
    config["serving_slo_maximum_ttft_seconds"] = 0.05
    assessment = assess_pilot_payloads(
        config=config,
        controller=controller,
        serving=serving,
        rollout=rollout,
        readiness=readiness,
    )

    assert not assessment.pilot_valid
    assert assessment.slo_restoration is not None
    assert assessment.slo_restoration.status == "not-restored"
    assert assessment.derived_phase_events == ()
    assert assessment.derived_trace_events == ()
    assert any("SLO was not restored" in reason for reason in assessment.invalid_reasons)


def test_repeated_validation_markers_form_one_lifecycle_envelope() -> None:
    config, controller, serving, rollout, readiness = _payloads()
    events = list(rollout["phase_events"])
    end_index = next(
        index
        for index, event in enumerate(events)
        if event["phase"] == ExperimentPhase.STATE_VALIDATE_END.value
    )
    repeated_begin = dict(events[end_index - 1])
    repeated_begin["monotonic_timestamp_ns"] = events[end_index]["monotonic_timestamp_ns"] + 1
    repeated_end = dict(events[end_index])
    repeated_end["monotonic_timestamp_ns"] += 2
    events[end_index + 1 : end_index + 1] = [repeated_begin, repeated_end]
    rollout["phase_events"] = tuple(events)
    rollout["minimal_trace_events"] += (
        {"attributes": {"experiment_phase": ExperimentPhase.STATE_VALIDATE_BEGIN.value}},
        {"attributes": {"experiment_phase": ExperimentPhase.STATE_VALIDATE_BEGIN.value}},
        {"attributes": {"experiment_phase": ExperimentPhase.STATE_VALIDATE_END.value}},
        {"attributes": {"experiment_phase": ExperimentPhase.STATE_VALIDATE_END.value}},
    )

    assessment = assess_pilot_payloads(
        config=config,
        controller=controller,
        serving=serving,
        rollout=rollout,
        readiness=readiness,
    )

    assert assessment.pilot_valid


def test_missing_restore_interference_overlap_invalidates_pilot() -> None:
    config, controller, serving, rollout, readiness = _payloads()
    serving["requests"] = tuple(
        row for row in serving["requests"] if row["phase"] != "restore-interference"
    )

    assessment = assess_pilot_payloads(
        config=config,
        controller=controller,
        serving=serving,
        rollout=rollout,
        readiness=readiness,
    )

    assert not assessment.pilot_valid
    assert any("restore-interference" in reason for reason in assessment.invalid_reasons)


def test_directory_loader_preserves_structured_invalid_assessment(tmp_path) -> None:
    assessment = assess_pilot_directory(
        work_root=tmp_path,
        config={"attempt_id": "missing-fixture"},
        controller={},
    )

    assert not assessment.pilot_valid
    assert assessment.evidence is None
    assert assessment.invalid_reasons[0].startswith("raw pilot evidence could not be loaded")
