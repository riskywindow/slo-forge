from __future__ import annotations

from pathlib import Path

import pytest

from sloforge.helix.characterization.environment_study import (
    FIXTURE_SPECS,
    EnvironmentOperation,
    EnvironmentStudyArtifact,
    StateCategory,
    build_environment_fixture,
    run_environment_study,
)
from sloforge.helix.characterization.matrix import EnvironmentSize, EvidenceClass, TraceLevel


@pytest.mark.parametrize("workload", tuple(FIXTURE_SPECS))
def test_controlled_environment_fixtures_are_deterministic_bounded_and_authenticated(
    tmp_path: Path,
    workload: EnvironmentSize,
) -> None:
    first, first_services = build_environment_fixture(
        tmp_path / "first",
        workload=workload,
        seed=41,
    )
    second, second_services = build_environment_fixture(
        tmp_path / "second",
        workload=workload,
        seed=41,
    )

    assert first == second
    assert first_services == second_services
    assert first.workload_evidence_class is EvidenceClass.SYNTHETIC
    assert not first.distribution_claim
    assert first.payload_bytes <= 4 * 1024 * 1024
    assert len(first.files) <= 512
    assert len({item.relative_path for item in first.files}) == len(first.files)
    assert all(len(item.sha256) == 64 for item in first.files)
    assert all(
        (tmp_path / "first" / item.relative_path).stat().st_size == item.size_bytes
        for item in first.files
    )


def test_environment_study_preserves_raw_trace_levels_without_changing_semantics(
    tmp_path: Path,
) -> None:
    result = run_environment_study(
        tmp_path / "study",
        seed=73,
        repetitions=1,
        warmups=1,
        workloads=(EnvironmentSize.TINY,),
        trace_levels=(TraceLevel.DISABLED, TraceLevel.MINIMAL, TraceLevel.FULL),
    )

    assert len(result.trials) == 6
    measured = [trial for trial in result.trials if not trial.warmup]
    assert {trial.trace_level for trial in measured} == set(TraceLevel)
    assert len({trial.semantic_result_sha256 for trial in measured}) == 1
    assert len({trial.base_capsule_id for trial in measured}) == 1
    assert len({trial.checkpoint_capsule_id for trial in measured}) == 1
    expected_operations = tuple(EnvironmentOperation)
    for trial in result.trials:
        assert tuple(item.operation for item in trial.operations) == expected_operations
        assert all(item.duration_ns >= 0 for item in trial.operations)
        assert all(item.process_cpu_ns >= 0 for item in trial.operations)
        assert trial.workload_evidence_class is EvidenceClass.SYNTHETIC
        assert trial.timing_evidence_class is EvidenceClass.HARDWARE_BACKED_REAL
        assert not trial.physical_cow_measured
        assert not trial.process_startup_measured

    by_level = {trial.trace_level: trial for trial in measured}
    assert all(item.observation is None for item in by_level[TraceLevel.DISABLED].operations)
    assert not by_level[TraceLevel.DISABLED].base_capsule_files
    assert not by_level[TraceLevel.DISABLED].base_cas_objects
    assert not by_level[TraceLevel.DISABLED].checkpoint_capsule_files
    assert not by_level[TraceLevel.DISABLED].checkpoint_cas_objects
    assert all(item.observation is not None for item in by_level[TraceLevel.MINIMAL].operations)
    assert not by_level[TraceLevel.MINIMAL].base_capsule_files
    assert not by_level[TraceLevel.MINIMAL].base_cas_objects
    assert not by_level[TraceLevel.MINIMAL].checkpoint_capsule_files
    assert not by_level[TraceLevel.MINIMAL].checkpoint_cas_objects
    assert by_level[TraceLevel.FULL].base_capsule_files
    assert by_level[TraceLevel.FULL].base_cas_objects
    assert by_level[TraceLevel.FULL].checkpoint_capsule_files
    assert by_level[TraceLevel.FULL].checkpoint_cas_objects
    assert (
        by_level[TraceLevel.DISABLED].trace_detail_bytes
        < by_level[TraceLevel.MINIMAL].trace_detail_bytes
    )
    assert (
        by_level[TraceLevel.MINIMAL].trace_detail_bytes
        < by_level[TraceLevel.FULL].trace_detail_bytes
    )

    artifact_path = tmp_path / "study" / "environment-study.json"
    reloaded = EnvironmentStudyArtifact.model_validate_json(artifact_path.read_text())
    assert reloaded == result
    assert not (tmp_path / "study" / ".working").exists()


def test_service_state_is_separate_from_filesystem_and_process_memory_claims(
    tmp_path: Path,
) -> None:
    result = run_environment_study(
        tmp_path / "service-study",
        seed=113,
        repetitions=1,
        warmups=0,
        workloads=(EnvironmentSize.SERVICE_HEAVY,),
        trace_levels=(TraceLevel.FULL,),
    )
    trial = result.trials[0]
    composition = trial.state_composition

    assert composition.filesystem_state_bytes > 0
    assert composition.database_state_bytes > 0
    assert composition.service_state_bytes > 0
    assert composition.process_memory_bytes == 0
    assert composition.process_reconstruction_metadata_bytes > 0
    assert (
        composition.service_recipe_metadata_bytes
        >= composition.process_reconstruction_metadata_bytes
    )
    assert composition.logical_payload_bytes == (
        composition.filesystem_state_bytes
        + composition.database_state_bytes
        + composition.service_state_bytes
    )
    assert {item.category for item in trial.base_capsule_files} >= {
        StateCategory.FILESYSTEM,
        StateCategory.DATABASE,
        StateCategory.SERVICE,
    }
    assert all(
        item.content_sha256 == item.content_sha256.lower()
        for item in trial.base_cas_objects + trial.checkpoint_cas_objects
    )
    reconstruct = next(
        item
        for item in trial.operations
        if item.operation is EnvironmentOperation.SERVICE_RECONSTRUCT
    )
    assert reconstruct.observation is not None
    assert reconstruct.observation.file_count == 2


def test_environment_study_rejects_unbounded_or_destructive_invocations(tmp_path: Path) -> None:
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "user-owned.txt").write_text("preserve me\n")
    with pytest.raises(FileExistsError):
        run_environment_study(occupied, seed=1, repetitions=1, warmups=0)
    assert (occupied / "user-owned.txt").read_text() == "preserve me\n"

    with pytest.raises(ValueError, match="repetitions"):
        run_environment_study(tmp_path / "too-many", seed=1, repetitions=101)
    with pytest.raises(ValueError, match="unsupported"):
        build_environment_fixture(
            tmp_path / "none",
            workload=EnvironmentSize.NONE,
            seed=1,
        )
