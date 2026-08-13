from __future__ import annotations

import json
from pathlib import Path

import pytest

from sloforge.helix.characterization.gpu_reclamation_v10_failure_replay import (
    OFFER_STATE_SCHEMA,
    GlobalOfferState,
    audit_v6_failure,
    build_partial_failure_evidence,
    enforce_total_outstanding_bound,
    gpu1_admission_allowed,
    persist_partial_failure_evidence,
    queue_depths_at,
)

ROOT = Path(__file__).parents[2]


@pytest.fixture(scope="module")
def v6_audit() -> dict[str, object]:
    return audit_v6_failure(repository_root=ROOT, seed=41)


def test_v6_replay_is_bound_to_raw_evidence_and_does_not_claim_lambda2_regression(
    v6_audit: dict[str, object],
) -> None:
    assert v6_audit["attempt_id"] == "exp004-v10-naive-s41-v6"
    assert v6_audit["seed"] == 41
    assert v6_audit["status"] == "IMPLEMENTATION_BUG_CONFIRMED"
    assert v6_audit["lambda_2_regression_supported"] is False
    provenance = v6_audit["provenance"]
    assert isinstance(provenance, list)
    assert len(provenance) == 103
    assert all(len(row["artifact_sha256"]) == 64 for row in provenance)
    with pytest.raises(ValueError, match="seed 41"):
        audit_v6_failure(repository_root=ROOT, seed=42)


def test_recovery_threshold_uses_waiting_backlog_not_healthy_inflight_work(
    v6_audit: dict[str, object],
) -> None:
    observations = v6_audit["v6_observations"]
    assert isinstance(observations, dict)
    assert observations["queue_metric_checkpoint"] == {
        "total_outstanding": 8,
        "waiting_backlog": 0,
        "active_in_service": 8,
    }
    assert observations["configured_recovery_queue_threshold"] == 4
    assert observations["total_outstanding_threshold_pass"] is False
    assert observations["waiting_backlog_threshold_pass"] is True

    rows = (
        {
            "scheduled_arrival_ns": 1,
            "admitted_ns": 2,
            "completed_ns": 10,
        },
        {
            "scheduled_arrival_ns": 3,
            "admitted_ns": 8,
            "completed_ns": 12,
        },
    )
    assert queue_depths_at(rows, timestamp_ns=5) == {
        "total_outstanding": 2,
        "waiting_backlog": 1,
        "active_in_service": 1,
    }


def test_compact_telemetry_contract_consumes_each_completion_exactly_once(
    v6_audit: dict[str, object],
) -> None:
    observations = v6_audit["v6_observations"]
    assert isinstance(observations, dict)
    assert observations["telemetry_snapshot_count"] == 96
    assert observations["telemetry_unique_completion_rows"] == 734
    assert observations["telemetry_cumulative_rows_read_by_v6_pattern"] == 35_192
    assert observations["telemetry_row_rescan_amplification"] == pytest.approx(
        47.94550408719346
    )
    assert observations["compact_replay_rows_inspected"] == 734
    assert observations["compact_replay_maximum_delta_rows"] == 9
    assert observations["telemetry_cumulative_bytes"] == 52_342_107
    assert observations["telemetry_final_snapshot_bytes"] == 1_091_744


def test_hard_total_outstanding_bound_remains_live_after_gpu1_is_useful(
    v6_audit: dict[str, object],
) -> None:
    observations = v6_audit["v6_observations"]
    assert isinstance(observations, dict)
    assert observations["gpu0_final_vllm_state"]["running_requests"] == 16
    assert observations["gpu0_final_vllm_state"]["waiting_requests"] == 48
    assert observations["minimum_gpu0_total_outstanding_at_failure"] == 128
    assert observations["configured_hard_outstanding_bound"] == 64
    assert observations["hard_outstanding_bound_pass"] is False
    enforce_total_outstanding_bound(total_outstanding=64, maximum=64)
    with pytest.raises(RuntimeError, match="total outstanding"):
        enforce_total_outstanding_bound(total_outstanding=65, maximum=64)


def test_gpu1_admission_is_bounded_by_global_watermark_and_stop_state(
    v6_audit: dict[str, object],
) -> None:
    observations = v6_audit["v6_observations"]
    assert isinstance(observations, dict)
    assert observations["last_sequence_scheduled_before_gpu0_failure"] == 1566
    assert observations["gpu1_rows_scheduled_after_gpu0_failure"] == 15
    assert observations["gpu1_post_failure_sequence_minimum"] == 1568
    assert observations["gpu1_post_failure_sequence_maximum"] == 1596

    running = GlobalOfferState.from_mapping(
        {
            "schema_version": OFFER_STATE_SCHEMA,
            "observed_ns": 100,
            "last_offered_sequence": 1566,
            "next_sequence": 1567,
            "stopped": False,
            "error": None,
        }
    )
    assert gpu1_admission_allowed(
        sequence=1566, last_admitted_sequence=1564, state=running
    )
    assert not gpu1_admission_allowed(
        sequence=1568, last_admitted_sequence=1566, state=running
    )
    stopped = GlobalOfferState(
        observed_ns=101,
        last_offered_sequence=1566,
        next_sequence=1567,
        stopped=True,
        error="bounded GPU0 queue overflow",
    )
    assert not gpu1_admission_allowed(
        sequence=1566, last_admitted_sequence=1564, state=stopped
    )


def test_gpu0_failure_persists_complete_partial_evidence_before_propagation(
    tmp_path: Path, v6_audit: dict[str, object]
) -> None:
    observations = v6_audit["v6_observations"]
    assert isinstance(observations, dict)
    assert observations["v6_partial_gpu0_evidence_present"] is False

    path = tmp_path / "serving" / "partial-failure.json"
    try:
        raise RuntimeError("synthetic bounded queue overflow")
    except RuntimeError as error:
        payload = build_partial_failure_evidence(
            attempt_id="exp004-v10-offline-replay",
            started_ns=10,
            failed_ns=20,
            offered_requests=({"request_id": "offered-1", "sequence": 1},),
            gpu0_requests=(
                {
                    "request_id": "offered-1",
                    "sequence": 1,
                    "completed_ns": None,
                },
            ),
            queue_state={"external": 1, "running": 1, "waiting": 0},
            offer_state={
                "schema_version": OFFER_STATE_SCHEMA,
                "observed_ns": 20,
                "last_offered_sequence": 1,
                "next_sequence": 2,
                "stopped": True,
                "error": "RuntimeError",
            },
            error=error,
        )
        persist_partial_failure_evidence(path, payload)

    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted == payload
    assert persisted["error"] == {
        "type": "RuntimeError",
        "message": "synthetic bounded queue overflow",
    }
    assert persisted["global_offered_requests"][0]["request_id"] == "offered-1"
    assert persisted["gpu0_requests"][0]["completed_ns"] is None
    assert len(persisted["evidence_sha256"]) == 64
    with pytest.raises(FileExistsError):
        persist_partial_failure_evidence(path, payload)


def test_retry_contract_requires_all_five_controller_corrections(
    v6_audit: dict[str, object],
) -> None:
    expected = v6_audit["expected_retry_behavior"]
    assert expected == {
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
    assert str(v6_audit["retry_recommendation"]).startswith("NO_RETRY_UNCHANGED")
