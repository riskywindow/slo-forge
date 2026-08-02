from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from sloforge.genesis.capsule import CapsuleValidationReport, Digest, VerificationLevel
from sloforge.genesis.evaluation_campaigns.evolution import (
    AdaptationStrategy,
    EvolutionCampaignValidationError,
    RuntimeEvidenceMode,
    run_evolution_campaign,
    validate_evolution_campaign,
)
from sloforge.genesis.evolution import (
    CapsuleReference,
    ChallengerSpec,
    GateObservation,
    GateStage,
    TransitionCategory,
    TransitionCompatibility,
)
from sloforge.genesis.ir import canonical_json


def _capsule(tmp_path: Path, name: str, character: str) -> CapsuleReference:
    path = tmp_path / name
    path.mkdir()
    return CapsuleReference(
        capsule_id=name,
        capsule_digest=character * 64,
        genome_hash=("f" if character != "f" else "e") * 64,
        path=str(path.resolve()),
    )


def _capsule_validator(
    champion: CapsuleReference, challenger: CapsuleReference
) -> Callable[[Path], CapsuleValidationReport]:
    by_path = {
        Path(champion.path).resolve(): champion,
        Path(challenger.path).resolve(): challenger,
    }

    def validate(path: Path) -> CapsuleValidationReport:
        reference = by_path[path.resolve()]
        return CapsuleValidationReport(
            capsule_digest=Digest(value=reference.capsule_digest),
            candidate_genome_hash=Digest(value=reference.genome_hash),
            promotion_verification_level=VerificationLevel.PROPERTY,
            integrity_valid=True,
            contract_compatible=True,
            evidence_complete=True,
            promotion_eligible=True,
            checked_at=datetime(2026, 8, 2, tzinfo=UTC),
            issues=(),
            local_evolution_eligible=True,
            external_production_eligible=False,
        )

    return validate


def _transition_validator(
    _champion: CapsuleReference,
    _challenger: CapsuleReference,
    category: TransitionCategory,
) -> TransitionCompatibility:
    assert category is TransitionCategory.REQUEST_BOUNDARY_SWAP
    return TransitionCompatibility(
        compatible=True,
        behavior="pin_existing_streams",
        reason="test fixture validates request-boundary stream pinning",
        champion_recovery_digest="a" * 64,
        challenger_recovery_digest="b" * 64,
    )


def _gate_collector(**values: object) -> GateObservation:
    output = values["output_directory"]
    assert isinstance(output, Path)
    output.mkdir(parents=True)
    stage = values["stage"]
    assert isinstance(stage, GateStage)
    seed = values["seed"]
    sample_count = values["sample_count"]
    candidate_id = values["candidate_id"]
    challenger = values["challenger"]
    observed_at_ms = values["observed_at_ms"]
    assert isinstance(seed, int)
    assert isinstance(sample_count, int)
    assert isinstance(candidate_id, str)
    assert isinstance(challenger, CapsuleReference)
    assert isinstance(observed_at_ms, int)
    payload = (
        canonical_json(
            {
                "candidate_id": candidate_id,
                "capsule_digest": challenger.capsule_digest,
                "sample_count": sample_count,
                "seed": seed,
                "stage": stage.value,
            }
        )
        + b"\n"
    )
    path = output / "gate-evidence.json"
    path.write_bytes(payload)
    return GateObservation(
        event_id=f"h7-{seed}-{stage.value}-evidence",
        candidate_id=candidate_id,
        capsule_digest=challenger.capsule_digest,
        evidence_digest=hashlib.sha256(payload).hexdigest(),
        stage=stage,
        verification_level=VerificationLevel.PROPERTY,
        observed_at_ms=observed_at_ms,
        deterministic_seed=seed,
        sample_count=sample_count,
        error_rate=0.0,
        p95_ttft_ratio=1.0,
        p99_tpot_ratio=1.0,
        quality_regression=0.0,
        interrupted_streams=0,
    )


def _gate_validator_factory(
    root: Path,
) -> Callable[[GateObservation, ChallengerSpec, CapsuleReference], bool]:
    def validate(
        observation: GateObservation,
        challenger: ChallengerSpec,
        _champion: CapsuleReference,
    ) -> bool:
        path = root / observation.stage.value / "gate-evidence.json"
        if not path.is_file():
            return False
        payload = path.read_bytes()
        document = json.loads(payload)
        return hashlib.sha256(payload).hexdigest() == observation.evidence_digest and document == {
            "candidate_id": challenger.candidate_id,
            "capsule_digest": challenger.capsule.capsule_digest,
            "sample_count": observation.sample_count,
            "seed": observation.deterministic_seed,
            "stage": observation.stage.value,
        }

    return validate


def _run(
    tmp_path: Path,
) -> tuple[Path, tuple[CapsuleReference, CapsuleReference]]:
    champion = _capsule(tmp_path, "champion", "a")
    challenger = _capsule(tmp_path, "challenger", "b")
    output = tmp_path / "campaign"
    report = run_evolution_campaign(
        output,
        champion=champion,
        challenger=challenger,
        capsule_validator=_capsule_validator(champion, challenger),
        seeds=(73129, 73130, 73131),
        rollback_seeds=(73131,),
        runtime_evidence_mode=RuntimeEvidenceMode.TEST_FIXTURE,
        gate_collector=_gate_collector,
        gate_validator_factory=_gate_validator_factory,
        transition_validator=_transition_validator,
    )
    return Path(report.report_path), (champion, challenger)


def test_campaign_exercises_controller_and_recomputes_all_metrics(tmp_path: Path) -> None:
    report_path, references = _run(tmp_path)
    champion, challenger = references
    with pytest.raises(
        EvolutionCampaignValidationError, match="test-fixture runtime evidence is not publishable"
    ):
        validate_evolution_campaign(
            report_path,
            capsule_validator=_capsule_validator(champion, challenger),
            transition_validator=_transition_validator,
        )
    report = validate_evolution_campaign(
        report_path,
        capsule_validator=_capsule_validator(champion, challenger),
        transition_validator=_transition_validator,
        allow_test_fixture=True,
        test_gate_validator_factory=_gate_validator_factory,
    )

    by_strategy = {item.strategy: item for item in report.aggregates}
    assert by_strategy[AdaptationStrategy.NO_ADAPTATION].restoration_rate == 0.0
    assert by_strategy[AdaptationStrategy.PHYSICAL_REPLAN_ONLY].restoration_rate == 0.0
    assert by_strategy[AdaptationStrategy.THRESHOLD_CONTROLLER].restoration_rate == 1.0
    genesis = by_strategy[AdaptationStrategy.GENESIS_EVOLUTION]
    assert genesis.restoration_rate == 1.0
    assert genesis.mean_logical_time_to_slo_restoration_ticks is not None
    assert genesis.mean_logical_time_to_slo_restoration_ticks < 10.0
    assert genesis.total_dropped_requests == 0
    assert genesis.total_interrupted_streams == 0
    assert genesis.total_rollbacks == 1
    assert genesis.total_invalid_challengers == 3
    assert genesis.total_local_runtime_request_observations == 15
    assert report.actual_hardware_experiments == 0
    assert all(
        row.active_stream_preserved
        for row in report.run_metrics
        if row.strategy is AdaptationStrategy.GENESIS_EVOLUTION
    )


def test_validator_rejects_raw_event_tampering(tmp_path: Path) -> None:
    report_path, references = _run(tmp_path)
    champion, challenger = references
    report = json.loads(report_path.read_text(encoding="utf-8"))
    raw_path = Path(report["raw_events_path"])
    payload = raw_path.read_text(encoding="utf-8")
    raw_path.write_text(payload.replace('"dropped_requests":10', '"dropped_requests":0', 1))

    with pytest.raises(EvolutionCampaignValidationError, match="raw event trace digest changed"):
        validate_evolution_campaign(
            report_path,
            capsule_validator=_capsule_validator(champion, challenger),
            transition_validator=_transition_validator,
            allow_test_fixture=True,
            test_gate_validator_factory=_gate_validator_factory,
        )


def test_validator_rejects_controller_snapshot_tampering(tmp_path: Path) -> None:
    report_path, references = _run(tmp_path)
    champion, challenger = references
    report = json.loads(report_path.read_text(encoding="utf-8"))
    snapshot_path = Path(report["genesis_evidence"][0]["controller_snapshot_path"])
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot["phase"] = "idle"
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")

    with pytest.raises(
        EvolutionCampaignValidationError, match="controller snapshot digest changed"
    ):
        validate_evolution_campaign(
            report_path,
            capsule_validator=_capsule_validator(champion, challenger),
            transition_validator=_transition_validator,
            allow_test_fixture=True,
            test_gate_validator_factory=_gate_validator_factory,
        )


def test_validator_recomputes_genesis_trace_after_attacker_rehashes(tmp_path: Path) -> None:
    report_path, references = _run(tmp_path)
    champion, challenger = references
    report = json.loads(report_path.read_text(encoding="utf-8"))
    raw_path = Path(report["raw_events_path"])
    rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
    promoted = next(
        row for row in rows if row["strategy"] == "genesis_evolution" and row["action"] == "promote"
    )
    promoted["logical_time_tick"] = 1
    payload = b"".join(canonical_json(row) + b"\n" for row in rows)
    raw_path.write_bytes(payload)
    report["raw_events_sha256"] = hashlib.sha256(payload).hexdigest()
    report_path.write_bytes(canonical_json(report) + b"\n")

    with pytest.raises(
        EvolutionCampaignValidationError,
        match="Genesis event trace does not derive from controller evidence",
    ):
        validate_evolution_campaign(
            report_path,
            capsule_validator=_capsule_validator(champion, challenger),
            transition_validator=_transition_validator,
            allow_test_fixture=True,
            test_gate_validator_factory=_gate_validator_factory,
        )
