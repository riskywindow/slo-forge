from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from sloforge.helix.characterization.gpu_reclamation_methodology import (
    ArtifactSampleRef,
    Experiment004GpuHourLedger,
    GpuActiveInterval,
    charge_failed_gpu_reservation,
    reserve_gpu_invocation,
    settle_gpu_invocation,
)


def _artifact(name: str) -> ArtifactSampleRef:
    return ArtifactSampleRef(
        artifact_reference=f"artifacts/{name}.json",
        artifact_sha256="a" * 64,
        sample_selector="$",
    )


def test_one_gpu_micro_reservation_counts_device_seconds_once() -> None:
    ledger = Experiment004GpuHourLedger(
        hard_additional_gpu_seconds=21_600.0,
        consumed_additional_gpu_seconds=0.0,
    )

    reserved, preflight = reserve_gpu_invocation(
        ledger,
        reservation_id="reservation-v11-micro",
        invocation_id="exp004-v11-micro-s41-v1",
        maximum_wall_seconds=525.0,
        config_sha256="b" * 64,
        gpu_count=1,
    )

    assert reserved.reservations[0].gpu_count == 1
    assert reserved.committed_additional_gpu_seconds == 525.0
    assert preflight.gpu_count == 1
    assert preflight.proposed_maximum_gpu_seconds == 525.0
    assert preflight.projected_committed_gpu_seconds == 525.0
    assert preflight.remaining_hard_budget_after_gpu_seconds == 21_075.0


def test_one_gpu_micro_settlement_preserves_cardinality_and_cost() -> None:
    reserved, _ = reserve_gpu_invocation(
        Experiment004GpuHourLedger(consumed_additional_gpu_seconds=0.0),
        reservation_id="reservation-v11-settle",
        invocation_id="exp004-v11-micro-s41-v1",
        maximum_wall_seconds=525.0,
        config_sha256="b" * 64,
        gpu_count=1,
    )

    settled = settle_gpu_invocation(
        reserved,
        reservation_id="reservation-v11-settle",
        function_call_id="fc-v11-micro",
        actual_gpu_models=("NVIDIA A100-SXM4-80GB",),
        gpu_uuids=("GPU-v11-micro-a100",),
        client_elapsed_seconds=300.0,
        remote_observed_allocation_seconds=299.0,
        raw_manifest=_artifact("v11-micro"),
        gpu_price_per_hour_usd=2.4984,
    )

    assert not settled.reservations
    assert settled.consumed_additional_gpu_seconds == 300.0
    assert settled.intervals[0].gpu_count == 1
    assert settled.intervals[0].gpu_seconds == 300.0
    assert settled.intervals[0].gpu_cost_usd == pytest.approx(300.0 / 3_600.0 * 2.4984)


def test_one_gpu_failure_charge_uses_one_device_second_per_wall_second() -> None:
    reserved, _ = reserve_gpu_invocation(
        Experiment004GpuHourLedger(consumed_additional_gpu_seconds=0.0),
        reservation_id="reservation-v11-failed",
        invocation_id="exp004-v11-micro-failed",
        maximum_wall_seconds=525.0,
        config_sha256="b" * 64,
        gpu_count=1,
    )

    charged = charge_failed_gpu_reservation(
        reserved,
        reservation_id="reservation-v11-failed",
        failure_stage="v11-micro-controller",
        failure_evidence=_artifact("v11-micro-failure"),
    )

    assert not charged.reservations
    assert charged.consumed_additional_gpu_seconds == 525.0
    assert charged.conservative_failure_charges[0].gpu_count == 1
    assert charged.conservative_failure_charges[0].conservative_gpu_seconds == 525.0


def test_settlement_rejects_gpu_inventory_cardinality_substitution() -> None:
    reserved, _ = reserve_gpu_invocation(
        Experiment004GpuHourLedger(consumed_additional_gpu_seconds=0.0),
        reservation_id="reservation-v11-cardinality",
        invocation_id="exp004-v11-micro-s41-v1",
        maximum_wall_seconds=525.0,
        config_sha256="b" * 64,
        gpu_count=1,
    )

    with pytest.raises(ValidationError, match="model count does not match"):
        settle_gpu_invocation(
            reserved,
            reservation_id="reservation-v11-cardinality",
            function_call_id="fc-v11-substituted",
            actual_gpu_models=(
                "NVIDIA A100-SXM4-80GB",
                "NVIDIA A100-SXM4-80GB",
            ),
            gpu_uuids=("GPU-v11-a", "GPU-v11-b"),
            client_elapsed_seconds=1.0,
            remote_observed_allocation_seconds=1.0,
            raw_manifest=_artifact("v11-cardinality"),
        )


def test_default_two_gpu_reservation_serialization_is_unchanged() -> None:
    reserved, preflight = reserve_gpu_invocation(
        Experiment004GpuHourLedger(consumed_additional_gpu_seconds=0.0),
        reservation_id="reservation-v10-default",
        invocation_id="exp004-v10-default",
        maximum_wall_seconds=10.0,
        config_sha256="c" * 64,
    )

    reservation_payload = reserved.reservations[0].model_dump(mode="json")
    assert reservation_payload == {
        "reservation_id": "reservation-v10-default",
        "invocation_id": "exp004-v10-default",
        "requested_gpu": "A100-80GB",
        "gpu_count": 2,
        "maximum_wall_seconds": 10.0,
        "config_sha256": "c" * 64,
    }
    assert preflight.gpu_count == 2
    assert preflight.proposed_maximum_gpu_seconds == 20.0


def test_persisted_v10_ledger_still_round_trips_strictly() -> None:
    path = Path("artifacts/branchfabric/gpu-validation/experiment-004/gpu-hours.json")
    source = json.loads(path.read_text())
    ledger = Experiment004GpuHourLedger.model_validate_json(path.read_text(), strict=True)

    assert ledger.model_dump(mode="json") == source
    legacy = [
        interval
        for interval in ledger.intervals
        if interval.invocation_id == "exp004-pilot-naive-s41-v1"
    ]
    assert len(legacy) == 1
    assert legacy[0].gpu_count == 2


@pytest.mark.parametrize("gpu_count", [0, 3])
def test_reservation_rejects_unsupported_gpu_count(gpu_count: int) -> None:
    with pytest.raises(ValidationError):
        reserve_gpu_invocation(
            Experiment004GpuHourLedger(consumed_additional_gpu_seconds=0.0),
            reservation_id="reservation-invalid-count",
            invocation_id="exp004-v11-invalid-count",
            maximum_wall_seconds=1.0,
            config_sha256="d" * 64,
            gpu_count=gpu_count,  # type: ignore[arg-type]
        )


def test_active_interval_rejects_non_a100_single_gpu() -> None:
    with pytest.raises(ValidationError, match="only A100-80GB"):
        GpuActiveInterval(
            invocation_id="exp004-v11-wrong-sku",
            function_call_id="fc-v11-wrong-sku",
            actual_gpu_models=("NVIDIA H100 80GB HBM3",),
            gpu_uuids=("GPU-v11-wrong-sku",),
            gpu_count=1,
            accounted_wall_seconds=1.0,
            raw_manifest=_artifact("v11-wrong-sku"),
        )
