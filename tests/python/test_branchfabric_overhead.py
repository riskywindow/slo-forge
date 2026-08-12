from pathlib import Path

import pytest

from sloforge.helix.characterization import overhead
from sloforge.helix.characterization.matrix import EvidenceClass, TraceLevel
from sloforge.helix.characterization.overhead import (
    OverheadTrial,
    run_instrumentation_overhead_study,
)


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


def test_overhead_series_do_not_mix_trace_levels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    durations = {
        TraceLevel.DISABLED: 100,
        TraceLevel.MINIMAL: 110,
        TraceLevel.FULL: 130,
    }

    def fake_trial(
        output: Path,
        *,
        level: TraceLevel,
        seed: int,
        repetition: int,
        warmup: bool,
        order_index: int,
    ) -> OverheadTrial:
        duration = durations[level]
        return OverheadTrial(
            workload_evidence_class=EvidenceClass.SYNTHETIC,
            timing_measurement_class=EvidenceClass.HARDWARE_BACKED_REAL,
            trace_level=level,
            seed=seed,
            repetition=repetition,
            warmup=warmup,
            order_index=order_index,
            artifact_path=output.as_posix(),
            source_commit="0" * 40,
            wall_time_ns=duration,
            trace_persistence_time_ns=0,
            end_to_end_wall_time_ns=duration,
            cpu_time_ns=duration,
            cpu_core_equivalents=1.0,
            branch_event_count=0,
            state_event_count=0,
            canonical_event_count=0,
            canonical_events_dropped=0,
            trace_persistence_bytes=0,
            semantic_digest="0" * 64,
            workload_artifact_bytes=0,
            workload_storage_bytes_per_second=0.0,
            resource_sample_count=0,
            resource_samples_dropped=0,
        )

    monkeypatch.setattr(overhead, "_run_trial", fake_trial)
    result = run_instrumentation_overhead_study(
        tmp_path / "study",
        seeds=(41,),
        repetitions=1,
        warmups_per_level=0,
        run_order_seed=7,
    )
    disabled_summary = result.wall_time_statistics["disabled"]["summary"]
    minimal_summary = result.wall_time_statistics["minimal"]["summary"]
    full_summary = result.wall_time_statistics["full"]["summary"]
    assert isinstance(disabled_summary, dict)
    assert isinstance(minimal_summary, dict)
    assert isinstance(full_summary, dict)
    assert disabled_summary["median"] == 100
    assert minimal_summary["median"] == 110
    assert full_summary["median"] == 130
    assert result.wall_time_paired_effects["minimal"]["mean_difference"] == 10
    assert result.wall_time_paired_effects["full"]["mean_difference"] == 30


def test_overhead_warmups_precede_measurements_and_remain_level_specific(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: list[tuple[TraceLevel, bool, int]] = []

    def fake_trial(
        output: Path,
        *,
        level: TraceLevel,
        seed: int,
        repetition: int,
        warmup: bool,
        order_index: int,
    ) -> OverheadTrial:
        observed.append((level, warmup, order_index))
        return OverheadTrial(
            workload_evidence_class=EvidenceClass.SYNTHETIC,
            timing_measurement_class=EvidenceClass.HARDWARE_BACKED_REAL,
            trace_level=level,
            seed=seed,
            repetition=repetition,
            warmup=warmup,
            order_index=order_index,
            artifact_path=output.as_posix(),
            source_commit="0" * 40,
            wall_time_ns=100 + order_index,
            trace_persistence_time_ns=0,
            end_to_end_wall_time_ns=100 + order_index,
            cpu_time_ns=100 + order_index,
            cpu_core_equivalents=1.0,
            branch_event_count=0,
            state_event_count=0,
            canonical_event_count=0,
            canonical_events_dropped=0,
            trace_persistence_bytes=0,
            semantic_digest="0" * 64,
            workload_artifact_bytes=0,
            workload_storage_bytes_per_second=0.0,
            resource_sample_count=0,
            resource_samples_dropped=0,
        )

    monkeypatch.setattr(overhead, "_run_trial", fake_trial)
    result = run_instrumentation_overhead_study(
        tmp_path / "warmup-study",
        seeds=(41,),
        repetitions=2,
        warmups_per_level=1,
        run_order_seed=7,
    )
    assert all(warmup for _level, warmup, _index in observed[:3])
    assert all(not warmup for _level, warmup, _index in observed[3:])
    for level in TraceLevel:
        series = result.wall_time_statistics[level.value]["series"]
        assert isinstance(series, dict)
        assert len(series["warmup_samples"]) == 1
        assert series["provenance"]["sample_count"] == 3
