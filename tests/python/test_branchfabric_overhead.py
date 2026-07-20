from pathlib import Path

import pytest

from sloforge.helix.characterization.overhead import run_instrumentation_overhead_study


def test_overhead_study_validates_bounds_before_running(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unique"):
        run_instrumentation_overhead_study(
            tmp_path / "duplicate",
            seeds=(41, 41),
            repetitions=1,
            warmups_per_level=0,
            run_order_seed=7,
        )
    with pytest.raises(ValueError, match=r"1\.\.20"):
        run_instrumentation_overhead_study(
            tmp_path / "repetitions",
            seeds=(41,),
            repetitions=0,
            warmups_per_level=0,
            run_order_seed=7,
        )


def test_overhead_study_refuses_nonempty_output(tmp_path: Path) -> None:
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "owned.txt").write_text("preserve")
    with pytest.raises(FileExistsError, match="empty"):
        run_instrumentation_overhead_study(
            output,
            seeds=(41,),
            repetitions=1,
            warmups_per_level=0,
            run_order_seed=7,
        )
    assert (output / "owned.txt").read_text() == "preserve"
