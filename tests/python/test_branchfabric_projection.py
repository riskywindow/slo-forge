from pathlib import Path

import pytest

from sloforge.helix.characterization.matrix import DivergencePattern, EvidenceClass
from sloforge.helix.characterization.projection import (
    calibrate_state,
    common_suffix_tokens,
    project_attention_cow,
    project_state_composition,
)

CAPSULE = Path(
    "artifacts/branchfabric/baseline/helix-demo-seed-41/capture/"
    "coding-failure-capture.continuum.json"
)


def test_calibration_preserves_source_and_fixture_sizes() -> None:
    calibration = calibrate_state(CAPSULE)

    assert calibration.evidence_class is EvidenceClass.SYNTHETIC
    assert calibration.observed_tokens == 8
    assert calibration.attention_bytes_per_token == 256
    assert calibration.component_bytes("state/attention-kv") == 2048
    assert calibration.component_bytes("state/token-history") == 115
    assert len(calibration.source_sha256) == 64

    projection = project_state_composition(calibration, tokens=1024)
    assert projection.attention_kv_bytes == 256 * 1024
    assert projection.observed_token_history_bytes == 115
    assert projection.total_without_token_history_extrapolation_bytes == 256 * 1024 + 497


@pytest.mark.parametrize("page_size", [4096, 16384, 65536, 262144, 1048576, 2097152])
def test_cow_projection_is_conservative_and_accounted(page_size: int) -> None:
    calibration = calibrate_state(CAPSULE)
    projection = project_attention_cow(
        calibration,
        branch_fanout=32,
        prefix_tokens=4097,
        suffix_tokens=256,
        divergence_pattern=DivergencePattern.MID,
        page_size_bytes=page_size,
    )

    assert projection.common_suffix_tokens == 128
    assert projection.divergent_suffix_tokens == 128
    assert projection.metadata_bytes == 0
    assert projection.internal_fragmentation_bytes >= 0
    assert -1 <= projection.sharing_efficiency <= 1
    assert projection.physical_amplification > 0


def test_controlled_divergence_mapping_and_validation() -> None:
    assert common_suffix_tokens(DivergencePattern.IMMEDIATE, 16) == 0
    assert common_suffix_tokens(DivergencePattern.EARLY, 16) == 2
    assert common_suffix_tokens(DivergencePattern.MID, 16) == 8
    assert common_suffix_tokens(DivergencePattern.LATE, 16) == 14
    assert common_suffix_tokens(DivergencePattern.HIGHLY_SHARED, 16) == 15
    with pytest.raises(ValueError, match="power of two"):
        project_attention_cow(
            calibrate_state(CAPSULE),
            branch_fanout=2,
            prefix_tokens=8,
            suffix_tokens=1,
            divergence_pattern=DivergencePattern.IMMEDIATE,
            page_size_bytes=5000,
        )
