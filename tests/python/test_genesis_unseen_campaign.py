from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from sloforge.genesis.evaluation_campaigns.unseen import (
    H1AttemptEvidence,
    H1CampaignConfiguration,
    H1CampaignReport,
    run_h1_unseen_campaign,
    validate_h1_unseen_campaign,
)
from sloforge.synthbench import GrammarConfiguration


@pytest.fixture(scope="module")
def unseen_campaign(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, H1CampaignReport]:
    output = tmp_path_factory.mktemp("genesis-h1") / "campaign"
    report = run_h1_unseen_campaign(
        H1CampaignConfiguration(
            grammar=GrammarConfiguration(
                seed=81_337,
                count=2,
                minimum_blocks=4,
                maximum_blocks=5,
                public_cases_per_task=2,
                hidden_cases_per_task=3,
            ),
            synthesis_seeds=(10_003, 20_011),
            request_timeout_seconds=5.0,
            maximum_attempt_seconds=40.0,
        ),
        output,
    )
    return output, report


def test_h1_campaign_runs_randomized_zero_day_tasks_across_seeds(
    unseen_campaign: tuple[Path, H1CampaignReport],
) -> None:
    output, report = unseen_campaign

    assert report.execution_scope == "cpu_only"
    assert report.hardware_validation_status == "not_exercised"
    assert report.selection_is_seed_deterministic
    assert report.aggregate.task_count == 2
    assert report.aggregate.attempt_count == 4
    assert report.aggregate.valid_system_rate == 1.0
    assert report.aggregate.unsupported_rate == 0.0
    assert report.aggregate.exact_public_case_rate == 1.0
    assert report.aggregate.exact_hidden_case_rate == 1.0
    assert report.aggregate.public_case_count == 8
    assert report.aggregate.hidden_case_count == 12
    assert report.aggregate.median_time_to_first_compiling_runtime_ns > 0
    assert report.aggregate.median_time_to_first_correct_runtime_ns > 0
    assert report.aggregate.human_authored_model_specific_lines == 0
    assert report.aggregate.valid_system_interval.sample_unit == "unseen_task"
    assert report.aggregate.valid_system_interval.lower <= 1.0
    assert report.aggregate.valid_system_interval.upper == 1.0
    assert report.aggregate.unsupported_task_interval.lower == 0.0
    assert report.aggregate.unsupported_task_interval.upper > 0.0
    assert len({task.task_generation_seed for task in report.tasks}) == 2
    assert (output / "report.json").is_file()

    for task in report.tasks:
        assert task.synthesis_seeds == (10_003, 20_011)
        assert task.valid_system
        assert task.compiling_runtime_found
        assert task.correct_runtime_found
        assert not task.unsupported
        assert task.time_to_first_compiling_runtime_ns is not None
        assert task.time_to_first_correct_runtime_ns is not None
        assert task.correct_seed_interval.sample_unit == "synthesis_seed"
        assert task.correct_seed_interval.successes == 2
        assert task.human_authored_model_specific_lines == 0
        for raw_reference in task.attempts:
            raw_path = Path(raw_reference.path)
            assert raw_path.is_file()
            attempt = H1AttemptEvidence.model_validate_json(raw_path.read_bytes(), strict=True)
            assert attempt.correct
            assert attempt.special_case_audit is not None
            assert attempt.special_case_audit.passed
            assert attempt.human_authored_model_specific_lines == 0
            assert attempt.submitted_model_specific_adapter_path is None
            assert len(attempt.public_cases) == 2
            assert len(attempt.hidden_cases) == 3
            for case in (*attempt.public_cases, *attempt.hidden_cases):
                execution = Path(case.execution_evidence.path)
                request = execution.parent.parent / f"{execution.parent.name}-request.json"
                assert execution.is_file()
                assert request.is_file()
                assert b"expected_tokens" not in request.read_bytes()

    validate_h1_unseen_campaign(report)


def test_h1_validator_rejects_forged_summary_and_raw_evidence(
    unseen_campaign: tuple[Path, H1CampaignReport],
) -> None:
    _, report = unseen_campaign
    forged_aggregate = report.aggregate.model_copy(update={"valid_system_rate": 0.0})
    forged_report = report.model_copy(update={"aggregate": forged_aggregate})
    with pytest.raises(ValueError, match="aggregate is not derived"):
        validate_h1_unseen_campaign(forged_report)

    raw_path = Path(report.tasks[0].attempts[0].path)
    original = raw_path.read_bytes()
    try:
        raw_path.write_bytes(original + b" ")
        with pytest.raises(ValueError, match="raw attempt artifact"):
            validate_h1_unseen_campaign(report)
    finally:
        raw_path.write_bytes(original)

    attempt = H1AttemptEvidence.model_validate_json(original, strict=True)
    execution_path = Path(attempt.hidden_cases[0].execution_evidence.path)
    execution_original = execution_path.read_bytes()
    try:
        execution_path.write_bytes(execution_original + b" ")
        with pytest.raises(ValueError, match="sandbox execution evidence"):
            validate_h1_unseen_campaign(report)
    finally:
        execution_path.write_bytes(execution_original)
    validate_h1_unseen_campaign(report)


def test_h1_configuration_requires_multiple_distinct_synthesis_seeds() -> None:
    grammar = GrammarConfiguration(seed=7, count=1)
    with pytest.raises(ValidationError, match="at least two synthesis seeds"):
        H1CampaignConfiguration(grammar=grammar, synthesis_seeds=(3,))
    with pytest.raises(ValidationError, match="must be distinct"):
        H1CampaignConfiguration(grammar=grammar, synthesis_seeds=(3, 3))
