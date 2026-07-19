from pathlib import Path

import pytest

from sloforge.helix.characterization.runner import (
    _hardware_manifest,
    _prepare_output,
    _software_manifest,
)

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
