from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from sloforge.helix.characterization.orchestration import (
    CharacterizationRunManifest,
    CharacterizationStage,
    OrchestrationConfig,
    RunState,
    StageCheckpoint,
    StageStatus,
    create_characterization_run,
    load_characterization_run,
    resume_characterization_run,
)

MATRIX = Path("benchmarks/branchfabric/characterization.yaml")


def _skip_only_config() -> OrchestrationConfig:
    return OrchestrationConfig(
        seed=41,
        stages=(
            CharacterizationStage.MATRIX_VALIDATE,
            CharacterizationStage.GPU,
            CharacterizationStage.MULTI_GPU,
            CharacterizationStage.MULTI_NODE,
        ),
        stage_timeout_seconds=20.0,
        maximum_attempts=1,
    )


def _checkpoints(run_directory: Path) -> list[StageCheckpoint]:
    return [
        StageCheckpoint.model_validate_json(path.read_bytes())
        for path in sorted((run_directory / "checkpoints").glob("*.json"))
    ]


def test_create_preserves_immutable_inputs_and_refuses_existing_run(tmp_path: Path) -> None:
    run_directory = tmp_path / "run"
    manifest = create_characterization_run(
        run_directory,
        matrix_path=MATRIX,
        config=_skip_only_config(),
        hardware_capability_manifest=None,
    )

    assert isinstance(manifest, CharacterizationRunManifest)
    assert manifest.matrix_id == "helix-characterization-cpu-v1"
    assert manifest.matrix_case_count == 855
    assert manifest.distribution_claim is False
    assert manifest.capability_evidence.evidence_class == "UNAVAILABLE"
    assert not manifest.capability_evidence.compatible_gpu_available
    preserved = run_directory / "inputs" / "characterization.yaml"
    assert hashlib.sha256(preserved.read_bytes()).hexdigest() == manifest.matrix_sha256
    assert load_characterization_run(run_directory) == manifest
    initial_status = json.loads((run_directory / "status" / "000000.json").read_bytes())
    assert initial_status["run_state"] == "ACTIVE"
    assert set(initial_status["stages"].values()) == {"PENDING"}

    with pytest.raises(FileExistsError, match="already exists"):
        create_characterization_run(
            run_directory,
            matrix_path=MATRIX,
            config=_skip_only_config(),
            hardware_capability_manifest=None,
        )


def test_resume_hashes_artifacts_and_explicitly_skips_unavailable_hardware(
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "run"
    create_characterization_run(
        run_directory,
        matrix_path=MATRIX,
        config=_skip_only_config(),
        hardware_capability_manifest=None,
    )
    result = resume_characterization_run(run_directory)

    assert result.run_state is RunState.COMPLETE
    assert result.stages == {
        CharacterizationStage.MATRIX_VALIDATE: StageStatus.SUCCEEDED,
        CharacterizationStage.GPU: StageStatus.SKIPPED_UNAVAILABLE,
        CharacterizationStage.MULTI_GPU: StageStatus.SKIPPED_UNAVAILABLE,
        CharacterizationStage.MULTI_NODE: StageStatus.SKIPPED_UNAVAILABLE,
    }
    checkpoints = _checkpoints(run_directory)
    assert [checkpoint.sequence for checkpoint in checkpoints] == list(range(4))
    assert all(checkpoint.artifacts for checkpoint in checkpoints)
    for checkpoint in checkpoints:
        for artifact in checkpoint.artifacts:
            payload = (run_directory / artifact.relative_path).read_bytes()
            assert len(payload) == artifact.size_bytes
            assert hashlib.sha256(payload).hexdigest() == artifact.sha256
    matrix_result = json.loads(
        (
            run_directory
            / "attempts"
            / "matrix_validate"
            / "attempt-000"
            / "matrix-validation.json"
        ).read_bytes()
    )
    assert matrix_result["evaluated_case_count"] == 855
    assert matrix_result["hardware_executed_case_count"] == 0
    for stage in (
        CharacterizationStage.GPU,
        CharacterizationStage.MULTI_GPU,
        CharacterizationStage.MULTI_NODE,
    ):
        skip = json.loads(
            (
                run_directory / "attempts" / stage.value / "attempt-000" / "skip-evidence.json"
            ).read_bytes()
        )
        assert skip["no_simulated_measurement_substituted"] is True
        assert skip["capability_evidence"]["source_present"] is False

    checkpoint_count = result.checkpoint_count
    status_count = result.status_snapshot_count
    resumed = resume_characterization_run(run_directory)
    assert resumed.checkpoint_count == checkpoint_count
    assert resumed.status_snapshot_count == status_count


def test_resume_records_orphan_then_retries_with_the_same_stage_seed(tmp_path: Path) -> None:
    run_directory = tmp_path / "run"
    config = OrchestrationConfig(
        seed=73,
        stages=(CharacterizationStage.MATRIX_VALIDATE,),
        stage_timeout_seconds=20.0,
        maximum_attempts=2,
    )
    create_characterization_run(
        run_directory,
        matrix_path=MATRIX,
        config=config,
        hardware_capability_manifest=None,
    )
    orphan = run_directory / "attempts" / "matrix_validate" / "attempt-000"
    orphan.mkdir(parents=True)
    (orphan / "partial-evidence.json").write_text('{"partial":true}\n', encoding="utf-8")

    result = resume_characterization_run(run_directory)
    checkpoints = _checkpoints(run_directory)

    assert result.run_state is RunState.COMPLETE
    assert result.stages[CharacterizationStage.MATRIX_VALIDATE] is StageStatus.SUCCEEDED
    assert [checkpoint.attempt for checkpoint in checkpoints] == [0, 1]
    assert [checkpoint.status for checkpoint in checkpoints] == [
        StageStatus.FAILED,
        StageStatus.SUCCEEDED,
    ]
    assert checkpoints[0].error is not None
    assert checkpoints[0].error.error_type == "InterruptedAttempt"
    assert checkpoints[0].stage_seed == checkpoints[1].stage_seed
    assert checkpoints[0].artifacts[0].relative_path.endswith("partial-evidence.json")


def test_timeout_is_terminal_and_preserves_attempt_directory(tmp_path: Path) -> None:
    run_directory = tmp_path / "run"
    config = OrchestrationConfig(
        seed=113,
        stages=(CharacterizationStage.MATRIX_VALIDATE,),
        stage_timeout_seconds=0.000001,
        maximum_attempts=1,
    )
    create_characterization_run(
        run_directory,
        matrix_path=MATRIX,
        config=config,
        hardware_capability_manifest=None,
    )

    result = resume_characterization_run(run_directory)
    checkpoint = _checkpoints(run_directory)[0]
    assert result.run_state is RunState.COMPLETE_WITH_FAILURES
    assert checkpoint.status is StageStatus.TIMED_OUT
    assert checkpoint.error is not None and checkpoint.error.error_type == "StageTimeout"
    assert (run_directory / checkpoint.attempt_reference).is_dir()


def test_tampered_preserved_input_fails_closed(tmp_path: Path) -> None:
    run_directory = tmp_path / "run"
    create_characterization_run(
        run_directory,
        matrix_path=MATRIX,
        config=_skip_only_config(),
        hardware_capability_manifest=None,
    )
    preserved = run_directory / "inputs" / "characterization.yaml"
    preserved.write_bytes(preserved.read_bytes() + b"\n# changed\n")

    with pytest.raises(ValueError, match="matrix hash mismatch"):
        load_characterization_run(run_directory)


def test_orchestration_configuration_bounds() -> None:
    with pytest.raises(ValidationError, match="first stage"):
        OrchestrationConfig(seed=1, stages=(CharacterizationStage.GPU,))
    with pytest.raises(ValidationError, match="non-empty and unique"):
        OrchestrationConfig(
            seed=1,
            stages=(
                CharacterizationStage.MATRIX_VALIDATE,
                CharacterizationStage.MATRIX_VALIDATE,
            ),
        )
    with pytest.raises(ValidationError, match="metadata_thread_counts"):
        OrchestrationConfig(seed=1, metadata_thread_counts=(1, 4, 2))
