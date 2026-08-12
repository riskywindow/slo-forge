from pathlib import Path

import pytest

from sloforge.helix.characterization.runner import (
    _hardware_manifest,
    _prepare_output,
    _software_manifest,
    run_continuum_trace,
)
from sloforge.helix.characterization.trace import iter_jsonl

HARDWARE = Path("artifacts/branchfabric/manifests/hardware-baseline.json")
SOFTWARE = Path("artifacts/branchfabric/manifests/software-baseline.json")


def test_output_replacement_requires_recognized_marker(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    (output / "foreign.txt").write_text("preserve")
    with pytest.raises(FileExistsError, match=r"vertical-run\.json"):
        _prepare_output(output, replace=True)
    assert (output / "foreign.txt").read_text() == "preserve"


def test_empty_output_is_accepted(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    _prepare_output(output, replace=False)
    assert output.is_dir()


def test_baseline_manifests_convert_to_trace_manifests() -> None:
    hardware = _hardware_manifest(HARDWARE)
    software = _software_manifest(SOFTWARE)
    assert hardware.cpu_model == "Apple M4 Pro"
    assert hardware.logical_cpu_count == 12
    assert software.python_version
    assert any(component.name == "pydantic" for component in software.components)


def test_continuum_trace_preserves_real_and_simulated_timing_classes(tmp_path: Path) -> None:
    result = run_continuum_trace(tmp_path / "continuum", seed=41)
    events = tuple(iter_jsonl(tmp_path / "continuum" / "state-operation-trace-v1.jsonl"))
    assert result.state_event_count == len(events) >= 18
    assert result.dropped_events == 0
    assert result.filtered_events == 0
    assert result.timing_measurement_counts["HARDWARE_BACKED_REAL"] > 0
    assert result.timing_measurement_counts["SIMULATED_HARDWARE"] > 0
    assert any(event.dependency_event_ids for event in events)
    assert result.migration_transaction_id
