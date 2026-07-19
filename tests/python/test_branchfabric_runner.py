from pathlib import Path

import pytest

from sloforge.helix.characterization.runner import _prepare_output


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
