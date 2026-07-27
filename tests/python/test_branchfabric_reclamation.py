from __future__ import annotations

from pathlib import Path

import pytest

from sloforge.continuum.transaction import CoordinatorConflict, DurableCoordinator, StaleOwner
from sloforge.helix.characterization.reclamation import _bounded_map, run_trial


def test_causal_cpu_reference_reclamation_is_transactional(tmp_path: Path) -> None:
    result = run_trial(
        seed=41,
        trace_level="full",
        baseline="optimized_bounded_parallel",
        work_root=tmp_path / "trial",
        repetition=0,
    )

    assert result["evidence_class"] == "CPU_REFERENCE_LOCAL_TRANSACTION"
    assert result["branch_count"] == 8
    assert result["admissions_stopped"]
    assert result["slo_restored"]
    assert result["work_lost_ticks"] == 0
    assert result["reclaimed_capacity_units"] == 8
    assert result["physical_gpu_capacity_reclaimed"] == 0
    assert result["stale_source_rejected"]
    assert result["corrupt_state_rejected"]
    assert result["all_transactions_completed"]
    assert not result["hidden_fallback"]
    assert 1 < result["maximum_concurrency"] <= 4
    assert result["maximum_queue_occupancy"] <= 8
    assert result["transferred_state_bytes"] > 0
    assert result["logical_transfer_bytes_requested"] > result["transferred_state_bytes"]
    assert result["transfer_reuse_saved_bytes"] == (
        result["logical_transfer_bytes_requested"] - result["transferred_state_bytes"]
    )
    assert result["sharing_saved_bytes"] > 0
    for operation in (
        "fork",
        "cow",
        "append",
        "checkpoint",
        "delta",
        "transform",
        "transfer",
        "commit",
        "reclaim",
    ):
        assert result["operation_stats_ns"][operation]["p50"] > 0
        assert result["candidate_operation_fraction_of_total_wall"][operation] > 0


def test_progress_cas_rejects_regression_and_stale_owner(tmp_path: Path) -> None:
    with DurableCoordinator(tmp_path / "coordinator.sqlite") as coordinator:
        lease = coordinator.create_lease(
            session_id="branch.a",
            owner_runtime="runtime.a",
            expiration_ms=1000,
            initial_token_index=3,
        )
        updated = coordinator.record_committed_progress(
            session_id="branch.a",
            owner_runtime="runtime.a",
            owner_epoch=lease.owner_epoch,
            fencing_token=lease.fencing_token,
            state_version=7,
            token_index=5,
            now_ms=10,
        )
        assert updated.last_committed_state_version == 7
        assert updated.last_committed_token_index == 5
        assert (
            coordinator.record_committed_progress(
                session_id="branch.a",
                owner_runtime="runtime.a",
                owner_epoch=lease.owner_epoch,
                fencing_token=lease.fencing_token,
                state_version=7,
                token_index=5,
                now_ms=11,
            )
            == updated
        )
        with pytest.raises(CoordinatorConflict, match="cannot regress"):
            coordinator.record_committed_progress(
                session_id="branch.a",
                owner_runtime="runtime.a",
                owner_epoch=lease.owner_epoch,
                fencing_token=lease.fencing_token,
                state_version=6,
                token_index=5,
                now_ms=12,
            )
        with pytest.raises(StaleOwner, match="stale owner"):
            coordinator.record_committed_progress(
                session_id="branch.a",
                owner_runtime="runtime.a",
                owner_epoch=lease.owner_epoch,
                fencing_token=lease.fencing_token + 1,
                state_version=8,
                token_index=6,
                now_ms=13,
            )


def test_reclamation_requires_controlled_seed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="one of"):
        run_trial(
            seed=7,
            trace_level="disabled",
            baseline="existing_serial",
            work_root=tmp_path / "invalid",
            repetition=0,
        )


def test_bounded_map_has_an_aggregate_deadline() -> None:
    release = __import__("threading").Event()

    def blocked(_concurrency: int, _occupancy: int) -> int:
        release.wait(timeout=0.2)
        return 1

    try:
        with pytest.raises(TimeoutError, match="deadline expired"):
            _bounded_map((blocked,), 1, timeout_seconds=0.01)
    finally:
        release.set()
