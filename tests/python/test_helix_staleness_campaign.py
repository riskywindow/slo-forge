from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from sloforge.helix.evaluation.staleness_campaign import (
    CASE_ORDER,
    BatchAttemptEvidence,
    BatchDisposition,
    CaseObservation,
    RawPolicyEvidence,
    RawStalenessEvidence,
    StalenessCampaign,
    StalenessCase,
    TrainerAttemptEvidence,
    TrainerDisposition,
    run_staleness_campaign,
    validate_staleness_campaign,
)
from sloforge.helix.staleness import (
    LogProbabilitySource,
    ReasonCode,
    StalenessDisposition,
    TrainingEligibility,
)


def _artifact_path(output: Path, observation: CaseObservation, kind: str) -> Path:
    artifact = next(item for item in observation.raw_artifacts if item.artifact_kind == kind)
    path = output / artifact.path
    assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact.sha256
    return path


def _files(output: Path) -> dict[str, bytes]:
    return {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in sorted(item for item in output.rglob("*") if item.is_file())
    }


def _digest_document(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def test_h4_campaign_executes_complete_fail_closed_matrix(tmp_path: Path) -> None:
    output = tmp_path / "h4"
    campaign = run_staleness_campaign(output, seeds=(7, 11))

    assert tuple((item.seed, item.case) for item in campaign.observations) == tuple(
        (seed, case) for seed in campaign.seeds for case in CASE_ORDER
    )
    assert campaign.invalid_case_count == 8
    assert campaign.invalid_sample_acceptance_count == 0
    assert campaign.invalid_sample_acceptance_rate == 0.0
    assert campaign.staleness_eligible_case_count == 8
    assert campaign.trained_case_count == 4
    assert campaign.finite_training_case_count == 4
    assert campaign.evaluated_policy_outcome_count == 4

    first_seed = {item.case: item for item in campaign.observations if item.seed == 7}
    expected = {
        StalenessCase.CURRENT_ON_POLICY: (
            StalenessDisposition.BOUNDED_ACCEPT,
            TrainingEligibility.ELIGIBLE,
            BatchDisposition.ACCEPTED,
            TrainerDisposition.TRAINED,
        ),
        StalenessCase.BOUNDED_STALE: (
            StalenessDisposition.PRIORITY_REDUCTION,
            TrainingEligibility.ELIGIBLE_WITH_CORRECTION,
            BatchDisposition.ACCEPTED,
            TrainerDisposition.REFERENCE_ADAPTER_REJECTED,
        ),
        StalenessCase.HARD_REJECTED_STALE: (
            StalenessDisposition.HARD_REJECT,
            TrainingEligibility.INELIGIBLE,
            BatchDisposition.STALE_REJECTED,
            TrainerDisposition.BATCH_REJECTED,
        ),
        StalenessCase.MISSING_BEHAVIOR_LOGPROB: (
            StalenessDisposition.RECOMPUTE_LOGPROBS,
            TrainingEligibility.INELIGIBLE,
            BatchDisposition.RECOMPUTE_REQUIRED,
            TrainerDisposition.BATCH_REJECTED,
        ),
        StalenessCase.SEGMENTED_MIXED_POLICY: (
            StalenessDisposition.BOUNDED_ACCEPT,
            TrainingEligibility.ELIGIBLE_WITH_CORRECTION,
            BatchDisposition.STRICT_POLICY_MIXING_REJECTED,
            TrainerDisposition.BATCH_REJECTED,
        ),
        StalenessCase.RECOMPUTED_LOGPROB: (
            StalenessDisposition.BOUNDED_ACCEPT,
            TrainingEligibility.ELIGIBLE,
            BatchDisposition.ACCEPTED,
            TrainerDisposition.TRAINED,
        ),
        StalenessCase.INCOMPATIBLE_MODEL_STATE: (
            StalenessDisposition.HARD_REJECT,
            TrainingEligibility.INELIGIBLE,
            BatchDisposition.INCOMPATIBLE_STATE_REJECTED,
            TrainerDisposition.BATCH_REJECTED,
        ),
    }
    for case, dispositions in expected.items():
        observed = first_seed[case]
        assert (
            observed.staleness_disposition,
            observed.training_eligibility,
            observed.batch_disposition,
            observed.trainer_disposition,
        ) == dispositions
        assert not observed.invalid_sample_accepted

    parsed = StalenessCampaign.model_validate_json(
        (output / "campaign.json").read_text(encoding="utf-8"), strict=True
    )
    assert parsed == campaign


def test_h4_raw_evidence_preserves_invalid_and_recomputed_inputs(tmp_path: Path) -> None:
    output = tmp_path / "h4"
    campaign = run_staleness_campaign(output, seeds=(13, 17))
    cases = {item.case: item for item in campaign.observations if item.seed == 13}

    missing = cases[StalenessCase.MISSING_BEHAVIOR_LOGPROB]
    missing_staleness = RawStalenessEvidence.model_validate_json(
        _artifact_path(output, missing, "staleness").read_text(encoding="utf-8"),
        strict=True,
    )
    missing_record = missing_staleness.request.trajectory.log_probabilities[0]
    assert missing_record.behavior_source is LogProbabilitySource.MISSING
    assert missing_record.behavior_log_probability is None
    assert missing_staleness.portable_lag_reports == ()
    assert missing_staleness.portable_report_omission_reason is not None
    assert ReasonCode.MISSING_BEHAVIOR_LOGPROB in {
        reason.code for reason in missing_staleness.report.reasons
    }
    missing_batch = BatchAttemptEvidence.model_validate_json(
        _artifact_path(output, missing, "batch").read_text(encoding="utf-8"),
        strict=True,
    )
    assert missing_batch.candidates[0].training_sample is None
    assert missing_batch.manifest is None

    recomputed = cases[StalenessCase.RECOMPUTED_LOGPROB]
    recomputed_staleness = RawStalenessEvidence.model_validate_json(
        _artifact_path(output, recomputed, "staleness").read_text(encoding="utf-8"),
        strict=True,
    )
    recomputed_record = recomputed_staleness.request.trajectory.log_probabilities[0]
    assert recomputed_record.behavior_source is LogProbabilitySource.RECOMPUTED
    assert recomputed_record.behavior_recomputation is not None
    assert recomputed_record.behavior_evidence_digest == (
        recomputed_record.behavior_recomputation.result_digest
    )
    recomputed_trainer = TrainerAttemptEvidence.model_validate_json(
        _artifact_path(output, recomputed, "trainer").read_text(encoding="utf-8"),
        strict=True,
    )
    assert recomputed_trainer.result is not None
    assert all(
        math.isfinite(value)
        for metric in recomputed_trainer.result.metrics
        for value in (metric.objective, metric.policy_kl)
    )

    incompatible = cases[StalenessCase.INCOMPATIBLE_MODEL_STATE]
    incompatible_staleness = RawStalenessEvidence.model_validate_json(
        _artifact_path(output, incompatible, "staleness").read_text(encoding="utf-8"),
        strict=True,
    )
    assert ReasonCode.INCOMPATIBLE_TRANSITION in {
        reason.code for reason in incompatible_staleness.report.reasons
    }
    assert incompatible_staleness.portable_lag_reports == ()

    bounded = cases[StalenessCase.BOUNDED_STALE]
    bounded_policies = RawPolicyEvidence.model_validate_json(
        _artifact_path(output, bounded, "policy_epochs").read_bytes(), strict=True
    )
    bounded_staleness = RawStalenessEvidence.model_validate_json(
        _artifact_path(output, bounded, "staleness").read_bytes(), strict=True
    )
    distance = bounded_staleness.request.distance_evidence[0]
    expected_l2 = math.sqrt(
        sum(
            (left - right) ** 2
            for left, right in zip(
                bounded_policies.reference_policies[0].logits,
                bounded_policies.reference_policies[1].logits,
                strict=True,
            )
        )
    )
    assert distance.parameter_delta_l2 == expected_l2
    assert distance.sample_count == len(bounded_policies.reference_policies[0].actions)


def test_h4_preserves_async_adapter_rejections_and_policy_outcomes(tmp_path: Path) -> None:
    output = tmp_path / "h4"
    campaign = run_staleness_campaign(output, seeds=(19, 23))
    cases = {item.case: item for item in campaign.observations if item.seed == 19}

    bounded = cases[StalenessCase.BOUNDED_STALE]
    bounded_batch = BatchAttemptEvidence.model_validate_json(
        _artifact_path(output, bounded, "batch").read_text(encoding="utf-8"),
        strict=True,
    )
    bounded_trainer = TrainerAttemptEvidence.model_validate_json(
        _artifact_path(output, bounded, "trainer").read_text(encoding="utf-8"),
        strict=True,
    )
    assert bounded_batch.manifest is not None
    assert bounded_batch.manifest.behavior_policy_epoch_id.endswith("@0")
    assert bounded_batch.manifest.learner_policy_epoch_id.endswith("@1")
    assert bounded_trainer.base_policy is not None
    assert bounded_trainer.base_policy.policy_epoch_id.endswith("@1")
    assert bounded_trainer.result is None
    assert "mixed behavior policy epochs" in (bounded_trainer.rejection_reason or "")

    segmented = cases[StalenessCase.SEGMENTED_MIXED_POLICY]
    segmented_batch = BatchAttemptEvidence.model_validate_json(
        _artifact_path(output, segmented, "batch").read_text(encoding="utf-8"),
        strict=True,
    )
    assert segmented_batch.manifest is None
    assert {
        candidate.training_sample.policy_epoch_id
        for candidate in segmented_batch.candidates
        if candidate.training_sample is not None
    } == {
        f"h4-19-{StalenessCase.SEGMENTED_MIXED_POLICY.value}@0",
        f"h4-19-{StalenessCase.SEGMENTED_MIXED_POLICY.value}@1",
    }
    assert "silently mixed behavior policy epochs" in (segmented_batch.rejection_reason or "")

    for case in (StalenessCase.CURRENT_ON_POLICY, StalenessCase.RECOMPUTED_LOGPROB):
        observation = cases[case]
        assert observation.training_stability is not None
        outcome = observation.evaluated_policy_outcome
        assert outcome is not None
        assert outcome.candidate_action_probability > outcome.base_action_probability
        assert outcome.candidate_success_rate >= outcome.base_success_rate
        policy_evidence = RawPolicyEvidence.model_validate_json(
            _artifact_path(output, observation, "policy_epochs").read_text(encoding="utf-8"),
            strict=True,
        )
        assert policy_evidence.candidate_epoch is not None
        assert policy_evidence.candidate_epoch.parent_epoch == 1


def test_h4_campaign_is_deterministic_and_rejects_tampering(tmp_path: Path) -> None:
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"
    first = run_staleness_campaign(first_output, seeds=(29, 31))
    second = run_staleness_campaign(second_output, seeds=(29, 31))
    assert first == second
    assert _files(first_output) == _files(second_output)

    document = json.loads((first_output / "campaign.json").read_text(encoding="utf-8"))
    document["invalid_sample_acceptance_count"] = 1
    with pytest.raises(ValidationError, match="aggregate measurements"):
        StalenessCampaign.model_validate_json(json.dumps(document), strict=True)

    with pytest.raises(ValueError, match="unique seeds"):
        run_staleness_campaign(tmp_path / "duplicate", seeds=(37, 37))
    with pytest.raises(FileExistsError, match="absent or empty"):
        run_staleness_campaign(first_output, seeds=(37, 41))


def test_h4_campaign_reopen_validation_rejects_raw_tampering(tmp_path: Path) -> None:
    output = tmp_path / "h4"
    campaign = run_staleness_campaign(output, seeds=(43, 47))
    assert validate_staleness_campaign(output) == campaign

    raw = output / campaign.observations[0].raw_artifacts[0].path
    raw.write_bytes(raw.read_bytes() + b" ")
    with pytest.raises(ValueError, match="raw artifact digest mismatch"):
        validate_staleness_campaign(output)


def test_h4_campaign_reopen_rejects_resigned_summary_forgery(tmp_path: Path) -> None:
    output = tmp_path / "h4"
    run_staleness_campaign(output, seeds=(53, 59))
    path = output / "campaign.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    observation = next(
        item for item in document["observations"] if item["training_stability"] is not None
    )
    observation["training_stability"]["maximum_policy_kl"] = 99.0
    observation["case_id"] = _digest_document(
        {key: value for key, value in observation.items() if key != "case_id"}
    )
    document["campaign_id"] = _digest_document(
        {key: value for key, value in document.items() if key != "campaign_id"}
    )
    path.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="training stability"):
        validate_staleness_campaign(output)


def test_h4_campaign_reopen_rejects_oversized_manifest(tmp_path: Path) -> None:
    output = tmp_path / "h4"
    run_staleness_campaign(output, seeds=(61, 67))
    (output / "campaign.json").write_bytes(b" " * (8 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match="byte limit"):
        validate_staleness_campaign(output)
