from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from sloforge.continuum.adapters import (
    ModelContract,
    ReferenceHeadMajorAdapter,
    ReferenceTokenMajorAdapter,
)
from sloforge.helix.evaluation.preservation import (
    PreservationCampaign,
    PreservationCompatibilityError,
    PreservationError,
    PreservationRawEvidence,
    PreservationStrategy,
    StrategyObservation,
    run_preservation_campaign,
    validate_preservation_campaign,
)
from sloforge.helix.rollouts import ReferenceTrajectory

_SCENARIO = Path("scenarios/helix/resource/rollout-preservation-campaign.json")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _by_strategy(
    campaign: PreservationCampaign, seed: int
) -> dict[PreservationStrategy, StrategyObservation]:
    return {item.strategy: item for item in campaign.observations if item.seed == seed}


def test_preservation_campaign_executes_complete_multi_seed_state_matrix(
    tmp_path: Path,
) -> None:
    output = tmp_path / "campaign"
    campaign = run_preservation_campaign(output, scenario_path=_SCENARIO, seeds=(11, 29))

    assert len(campaign.observations) == 2 * len(PreservationStrategy)
    assert campaign.tick_basis == "deterministic_reference_accounting_ticks"
    assert "declared accounting weights" in campaign.tick_accounting_note
    assert campaign.source_revision_kind == "fixture"
    assert "not workspace Git HEAD" in campaign.limitations[-1]
    sealed_scenario = output / campaign.scenario_path
    assert hashlib.sha256(sealed_scenario.read_bytes()).hexdigest() == campaign.scenario_sha256
    for seed in campaign.seeds:
        rows = _by_strategy(campaign, seed)
        terminated = rows[PreservationStrategy.TERMINATE_RESTART]
        environment = rows[PreservationStrategy.ENVIRONMENT_ONLY]
        model = rows[PreservationStrategy.MODEL_ONLY]
        joint = rows[PreservationStrategy.JOINT]

        assert (terminated.lost_tokens, terminated.lost_tool_work_units) == (4, 1)
        assert (environment.lost_tokens, environment.lost_tool_work_units) == (4, 0)
        assert (model.lost_tokens, model.lost_tool_work_units) == (0, 1)
        assert (joint.lost_tokens, joint.lost_tool_work_units) == (0, 0)
        assert terminated.total_state_bytes == 0
        assert environment.model_state_bytes == 0 < environment.environment_state_bytes
        assert model.environment_state_bytes == 0 < model.model_state_bytes
        assert joint.model_state_bytes > 0 and joint.environment_state_bytes > 0
        assert all(item.resume_time_ms == item.resume_ticks * 5 for item in rows.values())

        for observation in rows.values():
            evidence_path = output / observation.evidence_path
            assert hashlib.sha256(evidence_path.read_bytes()).hexdigest() == (
                observation.evidence_sha256
            )
            evidence = PreservationRawEvidence.model_validate_json(
                evidence_path.read_bytes(), strict=True
            )
            assert (
                ReferenceTrajectory.model_validate_json(
                    json.dumps(evidence.rollout_restore.prefix_trajectory), strict=True
                ).trajectory_id
                == evidence.rollout_restore.prefix_trajectory_id
            )
            assert (
                ReferenceTrajectory.model_validate_json(
                    json.dumps(evidence.rollout_restore.post_restore_trajectory), strict=True
                ).trajectory_id
                == evidence.rollout_restore.post_restore_trajectory_id
            )
            assert evidence.evidence_id == observation.evidence_id
            assert evidence.tick_accounting_note == campaign.tick_accounting_note
            assert evidence.source_revision_kind == "fixture"
            assert evidence.source_revision == campaign.source_revision
            assert evidence.metrics.model_dump(mode="json") == {
                "lost_tokens": observation.lost_tokens,
                "lost_tool_work_units": observation.lost_tool_work_units,
                "resume_ticks": observation.resume_ticks,
                "resume_time_ms": observation.resume_time_ms,
                "serving_restoration_ticks": observation.serving_restoration_ticks,
                "model_state_bytes": observation.model_state_bytes,
                "environment_state_bytes": observation.environment_state_bytes,
                "total_state_bytes": observation.total_state_bytes,
                "cost_microunits": observation.cost_microunits,
            }
            assert evidence.model_restore.continuation_validation_passed
            assert (
                evidence.model_restore.expected_probe_token_id
                == evidence.model_restore.observed_probe_token_id
            )
            if observation.strategy.preserves_model:
                checkpoint = evidence.model_restore.checkpoint_document
                assert checkpoint is not None
                capsule = cast(dict[str, object], checkpoint["capsule"])
                identity = cast(dict[str, object], capsule["identity"])
                assert identity["git_commit"] == campaign.source_revision
                assert evidence.model_restore.resume_journal
            else:
                assert evidence.model_restore.replayed_token_count == 4
            if observation.strategy.preserves_environment:
                assert evidence.environment_restore.prefix_tool_work_preserved
                assert evidence.environment_restore.checkpoint_state_bytes > 0
            else:
                assert not evidence.environment_restore.prefix_tool_work_preserved

    assert (
        PreservationCampaign.model_validate_json(
            (output / "campaign.json").read_bytes(), strict=True
        )
        == campaign
    )
    assert validate_preservation_campaign(output) == campaign
    assert validate_preservation_campaign(output, scenario_path=_SCENARIO) == campaign
    assert not any(
        path.name.startswith(("work-", ".preservation-stage-")) for path in output.iterdir()
    )


def test_preservation_campaign_is_byte_deterministic_for_exact_seed_replay(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    left = run_preservation_campaign(first, scenario_path=_SCENARIO, seeds=(7, 13))
    right = run_preservation_campaign(second, scenario_path=_SCENARIO, seeds=(7, 13))

    assert left == right
    left_files = {
        path.relative_to(first): path.read_bytes() for path in first.rglob("*") if path.is_file()
    }
    right_files = {
        path.relative_to(second): path.read_bytes() for path in second.rglob("*") if path.is_file()
    }
    assert left_files == right_files


def test_preservation_campaign_rejects_incompatible_model_state_and_publishes_nothing(
    tmp_path: Path,
) -> None:
    source_model = ReferenceTokenMajorAdapter().config.model
    incompatible = replace(source_model, model_hash="f" * 64)

    def incompatible_destination(_source_model: ModelContract) -> ReferenceHeadMajorAdapter:
        return ReferenceHeadMajorAdapter(model=incompatible)

    output = tmp_path / "rejected"
    with pytest.raises(PreservationCompatibilityError, match="rejected incompatible"):
        run_preservation_campaign(
            output,
            scenario_path=_SCENARIO,
            seeds=(3, 5),
            destination_factory=incompatible_destination,
        )
    assert output.is_dir()
    assert not any(output.iterdir())


def test_preservation_campaign_enforces_seed_and_output_bounds(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="two to maximum_seeds"):
        run_preservation_campaign(
            tmp_path / "one-seed",
            scenario_path=_SCENARIO,
            seeds=(1,),
        )
    output = tmp_path / "nonempty"
    output.mkdir()
    (output / "keep.txt").write_text("user data\n")
    with pytest.raises(FileExistsError, match="absent or empty"):
        run_preservation_campaign(output, scenario_path=_SCENARIO, seeds=(1, 2))
    assert (output / "keep.txt").read_text() == "user data\n"


def test_preservation_campaign_validator_rejects_scenario_and_evidence_tampering(
    tmp_path: Path,
) -> None:
    output = tmp_path / "campaign"
    campaign = run_preservation_campaign(output, scenario_path=_SCENARIO, seeds=(17, 23))
    mismatched_scenario = tmp_path / "mismatched-scenario.json"
    scenario_document = json.loads(_SCENARIO.read_text())
    scenario_document["initial_tool_content"] = "different initial state\n"
    mismatched_scenario.write_text(json.dumps(scenario_document))

    with pytest.raises(PreservationError, match="external scenario differs"):
        validate_preservation_campaign(output, scenario_path=mismatched_scenario)

    sealed_scenario_path = output / campaign.scenario_path
    sealed_scenario_payload = sealed_scenario_path.read_bytes()
    sealed_scenario_path.write_bytes(sealed_scenario_payload + b" ")
    with pytest.raises(PreservationError, match="sealed preservation scenario digest"):
        validate_preservation_campaign(output)
    sealed_scenario_path.write_bytes(sealed_scenario_payload)

    evidence_path = output / campaign.observations[0].evidence_path
    evidence_path.write_bytes(evidence_path.read_bytes() + b" ")
    with pytest.raises(PreservationError, match="evidence digest mismatch"):
        validate_preservation_campaign(output)


def test_preservation_campaign_validator_rejects_missing_raw_artifact(tmp_path: Path) -> None:
    output = tmp_path / "campaign"
    campaign = run_preservation_campaign(output, scenario_path=_SCENARIO, seeds=(19, 31))
    (output / campaign.observations[-1].evidence_path).unlink()

    with pytest.raises(FileNotFoundError):
        validate_preservation_campaign(output)


def test_preservation_campaign_validator_rejects_resigned_observation_metrics(
    tmp_path: Path,
) -> None:
    output = tmp_path / "campaign"
    run_preservation_campaign(output, scenario_path=_SCENARIO, seeds=(37, 41))
    campaign_path = output / "campaign.json"
    document = json.loads(campaign_path.read_bytes())
    observation = document["observations"][0]
    observation["lost_tokens"] += 1
    observation["observation_id"] = _digest(
        {key: value for key, value in observation.items() if key != "observation_id"}
    )
    aggregate = document["aggregates"][0]
    aggregate["mean_lost_tokens"] += 1 / len(document["seeds"])
    document["campaign_id"] = _digest(
        {key: value for key, value in document.items() if key != "campaign_id"}
    )
    payload = _canonical_bytes(document) + b"\n"
    campaign_path.write_bytes(payload)

    PreservationCampaign.model_validate_json(payload, strict=True)
    with pytest.raises(PreservationError, match="metrics disagree"):
        validate_preservation_campaign(output, scenario_path=_SCENARIO)


def test_raw_preservation_models_reject_resigned_restore_claim_forgeries(
    tmp_path: Path,
) -> None:
    output = tmp_path / "campaign"
    campaign = run_preservation_campaign(output, scenario_path=_SCENARIO, seeds=(43, 47))
    rows = _by_strategy(campaign, 43)

    model_document = json.loads(
        (output / rows[PreservationStrategy.MODEL_ONLY].evidence_path).read_bytes()
    )
    model_document["model_restore"]["restored_continuation_hash"] = "f" * 64
    model_document["evidence_id"] = _digest(
        {key: value for key, value in model_document.items() if key != "evidence_id"}
    )
    with pytest.raises(ValueError, match="continuation validation claim"):
        PreservationRawEvidence.model_validate_json(_canonical_bytes(model_document), strict=True)

    environment_document = json.loads(
        (output / rows[PreservationStrategy.ENVIRONMENT_ONLY].evidence_path).read_bytes()
    )
    environment_document["environment_restore"]["restored_prefix_content_hash"] = "e" * 64
    environment_document["evidence_id"] = _digest(
        {key: value for key, value in environment_document.items() if key != "evidence_id"}
    )
    with pytest.raises(ValueError, match="preservation claim"):
        PreservationRawEvidence.model_validate_json(
            _canonical_bytes(environment_document), strict=True
        )

    manifest_document = json.loads(
        (output / rows[PreservationStrategy.ENVIRONMENT_ONLY].evidence_path).read_bytes()
    )
    manifest_document["environment_restore"]["selected_capsule_id"] = "d" * 64
    manifest_document["environment_restore"]["selected_capsule_manifest"]["capsule_id"] = "d" * 64
    manifest_document["evidence_id"] = _digest(
        {key: value for key, value in manifest_document.items() if key != "evidence_id"}
    )
    with pytest.raises(ValueError, match="manifest content identity"):
        PreservationRawEvidence.model_validate_json(
            _canonical_bytes(manifest_document), strict=True
        )


def test_raw_model_restore_rejects_resigned_checkpoint_and_journal_forgeries(
    tmp_path: Path,
) -> None:
    output = tmp_path / "campaign"
    campaign = run_preservation_campaign(output, scenario_path=_SCENARIO, seeds=(53, 59))
    observation = _by_strategy(campaign, 53)[PreservationStrategy.MODEL_ONLY]
    evidence_payload = (output / observation.evidence_path).read_bytes()

    watermark_document = json.loads(evidence_payload)
    watermark_document["model_restore"]["checkpoint_watermark"] += 1
    watermark_document["model_restore"]["restored_watermark"] += 1
    watermark_document["evidence_id"] = _digest(
        {key: value for key, value in watermark_document.items() if key != "evidence_id"}
    )
    with pytest.raises(ValueError, match="checkpoint watermark claim"):
        PreservationRawEvidence.model_validate_json(
            _canonical_bytes(watermark_document), strict=True
        )

    identity_document = json.loads(evidence_payload)
    identity_document["model_restore"]["checkpoint_capsule_id"] = "b" * 64
    identity_document["evidence_id"] = _digest(
        {key: value for key, value in identity_document.items() if key != "evidence_id"}
    )
    with pytest.raises(ValueError, match="checkpoint capsule claim"):
        PreservationRawEvidence.model_validate_json(
            _canonical_bytes(identity_document), strict=True
        )

    manifest_id_document = json.loads(evidence_payload)
    manifest_id_document["model_restore"]["checkpoint_manifest_id"] = "e" * 64
    manifest_id_document["evidence_id"] = _digest(
        {key: value for key, value in manifest_id_document.items() if key != "evidence_id"}
    )
    with pytest.raises(ValueError, match="checkpoint manifest claim"):
        PreservationRawEvidence.model_validate_json(
            _canonical_bytes(manifest_id_document), strict=True
        )

    restored_watermark_document = json.loads(evidence_payload)
    restored_watermark_document["model_restore"]["restored_watermark"] += 1
    restored_watermark_document["evidence_id"] = _digest(
        {key: value for key, value in restored_watermark_document.items() if key != "evidence_id"}
    )
    with pytest.raises(ValueError, match="resume changed the committed watermark"):
        PreservationRawEvidence.model_validate_json(
            _canonical_bytes(restored_watermark_document), strict=True
        )

    ancestry_document = json.loads(evidence_payload)
    ancestry_document["model_restore"]["checkpoint_document"]["ancestry"]["digest"] = "c" * 64
    ancestry_document["evidence_id"] = _digest(
        {key: value for key, value in ancestry_document.items() if key != "evidence_id"}
    )
    with pytest.raises(ValueError, match="ancestry digest"):
        PreservationRawEvidence.model_validate_json(
            _canonical_bytes(ancestry_document), strict=True
        )

    journal_document = json.loads(evidence_payload)
    journal_document["model_restore"]["resume_journal"][0]["transaction_id"] = "d" * 64
    journal_document["evidence_id"] = _digest(
        {key: value for key, value in journal_document.items() if key != "evidence_id"}
    )
    with pytest.raises(ValueError, match="journal contains a different transaction"):
        PreservationRawEvidence.model_validate_json(_canonical_bytes(journal_document), strict=True)

    sequence_document = json.loads(evidence_payload)
    sequence_document["model_restore"]["resume_journal"][1]["sequence"] += 1
    sequence_document["evidence_id"] = _digest(
        {key: value for key, value in sequence_document.items() if key != "evidence_id"}
    )
    with pytest.raises(ValueError, match="journal sequence is not contiguous"):
        PreservationRawEvidence.model_validate_json(
            _canonical_bytes(sequence_document), strict=True
        )

    terminal_document = json.loads(evidence_payload)
    terminal_document["model_restore"]["resume_journal"][-1]["to_phase"] = "SOURCE_DRAINING"
    terminal_document["evidence_id"] = _digest(
        {key: value for key, value in terminal_document.items() if key != "evidence_id"}
    )
    with pytest.raises(ValueError, match="journal does not terminate in COMPLETED"):
        PreservationRawEvidence.model_validate_json(
            _canonical_bytes(terminal_document), strict=True
        )
