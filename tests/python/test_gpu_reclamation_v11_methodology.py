from __future__ import annotations

import pytest
from pydantic import ValidationError

from sloforge.helix.characterization.gpu_reclamation_v11_methodology import (
    Experiment004V11IntegratedConfig,
    Experiment004V11MicroConfig,
    projected_gpu_seconds,
)


def _sha() -> str:
    return "a" * 64


def _micro() -> dict[str, object]:
    return {
        "attempt_id": "exp004-v11-micro-s41-v1",
        "seed": 41,
        "offline_gate_manifest": "artifacts/offline.json",
        "offline_gate_manifest_sha256": _sha(),
        "budget_authorization": "artifacts/budget.json",
        "budget_authorization_sha256": _sha(),
        "ledger_sha256_before_reservation": _sha(),
    }


def test_micro_contract_is_exact_one_gpu_16k_fanout8_topology() -> None:
    config = Experiment004V11MicroConfig.model_validate(_micro())
    assert config.gpu_count == 1
    assert config.prefix_length == 16_384
    assert config.fanout == 8
    assert config.suffix_length == 256
    assert config.expected_shared_blocks == 1_024
    assert config.expected_private_blocks == 128
    assert config.expected_total_blocks == 1_152
    assert config.maximum_chunk_bytes == 28 * 1024 * 1024
    assert config.buffer_count == 2


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gpu_count", 2),
        ("prefix_length", 8_192),
        ("fanout", 4),
        ("suffix_length", 128),
        ("tracing_level", "full"),
        ("maximum_chunk_bytes", 1),
        ("mode", "PRESERVE_NAIVE"),
    ],
)
def test_micro_contract_rejects_methodology_drift(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        Experiment004V11MicroConfig.model_validate(_micro() | {field: value})


def test_integrated_contract_reuses_frozen_v10_rates() -> None:
    config = Experiment004V11IntegratedConfig(
        attempt_id="exp004-v11-integrated-s41-v1",
        seed=41,
        offline_gate_manifest="artifacts/offline.json",
        offline_gate_manifest_sha256=_sha(),
        micro_validation_artifact="artifacts/micro.json",
        micro_validation_sha256=_sha(),
        post_micro_review_manifest="artifacts/reviews.json",
        post_micro_review_manifest_sha256=_sha(),
        budget_authorization="artifacts/budget.json",
        budget_authorization_sha256=_sha(),
        ledger_sha256_before_reservation=_sha(),
    )
    assert (config.lambda_1_rps, config.lambda_spike_rps, config.lambda_2_rps) == (
        12.0,
        15.0,
        20.0,
    )
    assert config.serving_stability_seconds == 5.0
    assert config.overload_queue_trigger == 20
    assert config.overload_queue_abort == 64


def test_budget_projection_includes_count_wall_and_explicit_margin() -> None:
    assert projected_gpu_seconds(gpu_count=1, predicted_wall_seconds=525, margin=0.15) == 603.75
    assert projected_gpu_seconds(
        gpu_count=2, predicted_wall_seconds=588, margin=0.15
    ) == pytest.approx(1352.4)
    with pytest.raises(ValueError, match="one or two"):
        projected_gpu_seconds(gpu_count=3, predicted_wall_seconds=1, margin=0)
