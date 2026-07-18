from __future__ import annotations

from pathlib import Path

import pytest

from sloforge.helix.characterization import EvidenceClass, TraceLevel, expand_matrix, load_matrix

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "benchmarks/branchfabric/characterization.yaml"


def test_characterization_matrix_is_deterministic_bounded_and_covers_required_sweeps() -> None:
    matrix = load_matrix(MATRIX)
    first = expand_matrix(matrix)
    second = expand_matrix(matrix)

    assert first == second
    assert len(first) < 100_000
    assert len({item.experiment_id for item in first}) == len(first)
    assert {item.spec.branch_fanout for item in first} >= {1, 2, 4, 8, 16, 32, 64, 128}
    assert {item.spec.prefix_tokens for item in first} >= {
        1024,
        4096,
        8192,
        16384,
        32768,
        65536,
        131072,
    }
    assert {item.spec.suffix_tokens for item in first} >= {16, 64, 256, 1024, 4096}
    assert {item.spec.page_size_bytes for item in first} >= {
        4096,
        16384,
        65536,
        262144,
        1048576,
        2097152,
    }
    assert {item.spec.trace_level for item in first} == set(TraceLevel)
    assert all(item.spec.evidence_class is not EvidenceClass.HARDWARE_BACKED_REAL for item in first)
    assert matrix.distribution_claim is False


def test_matrix_rejects_unknown_fields_and_simulated_hardware_mislabel(tmp_path: Path) -> None:
    source = MATRIX.read_text()
    unknown = tmp_path / "unknown.yaml"
    unknown.write_text(source.replace("matrix_id:", "unknown_field: true\nmatrix_id:", 1))
    with pytest.raises(ValueError, match="Extra inputs"):
        load_matrix(unknown)

    dishonest = tmp_path / "dishonest.yaml"
    dishonest.write_text(
        source.replace("evidence_class: SYNTHETIC", "evidence_class: HARDWARE_BACKED_REAL", 1)
    )
    with pytest.raises(ValueError, match="simulated GPU state"):
        load_matrix(dishonest)
