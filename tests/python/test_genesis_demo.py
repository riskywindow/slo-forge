from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from sloforge.genesis.demo import run_genesis_demo
from sloforge.genesis.evaluation import run_genesis_evaluation
from sloforge.genesis.evolution import EvolutionSnapshot, local_gate_evidence_validator
from sloforge.genesis.ir import canonical_json


def test_cpu_genesis_demo_is_artifact_backed_and_cross_layer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = run_genesis_demo(
        Path("demo"),
        seed=73131,
        runtime_seed=73129,
        observed_at=datetime(2026, 8, 2, 12, 0, tzinfo=UTC),
    )

    assert result.runtime_differential_passed
    assert result.cross_layer_accepted
    assert result.rejected_candidate_ids
    assert result.minimized_counterexample_ids
    assert result.learned_constraint_ids
    assert not result.capsule_promotion_eligible
    assert result.capsule_local_evolution_eligible
    assert not result.capsule_external_production_eligible
    assert result.redteam_finding_count == result.redteam_replayed_count
    assert result.kernel_candidate_count == 2
    assert result.kernel_speedup_claim_count == 0
    assert result.kernel_measurement_scope == "cpu_generated_runtime_end_to_end_serving"
    assert not result.kernel_causal_attribution
    assert result.evolution_promoted
    assert result.active_stream_preserved
    assert result.physical_degradation_triggered
    assert result.hardware_backed is False
    report = json.loads(Path(result.report_path).read_text(encoding="utf-8"))
    assert report["accepted_genome_hash"] == result.accepted_genome_hash
    timeline = json.loads((tmp_path / "demo/evolution/timeline.json").read_text(encoding="utf-8"))
    assert timeline["source"] == "controller_audit_records_and_artifact_bound_synthesis"
    assert any(item["action"] == "promote" for item in timeline["events"])
    actions = [item["action"] for item in timeline["events"]]
    assert actions.index("begin_evolution") < actions.index("synthesize_challenger")
    assert actions.index("synthesize_challenger") < actions.index("register_challenger")
    assert actions.index("register_challenger") < actions.index("begin_shadow")
    synthesis_event = next(
        item for item in timeline["events"] if item["action"] == "synthesize_challenger"
    )
    assert (
        hashlib.sha256(Path(synthesis_event["artifact_path"]).read_bytes()).hexdigest()
        == synthesis_event["artifact_sha256"]
    )
    promoted = json.loads(
        (tmp_path / "demo/evolution/promoted-snapshot.json").read_text(encoding="utf-8")
    )
    challenger = promoted["challengers"][0]
    for stage in ("shadow", "canary"):
        artifact_path = tmp_path / f"demo/evolution/runtime-gates/{stage}/gate-evidence.json"
        payload = artifact_path.read_bytes()
        artifact = json.loads(payload)
        observation = challenger[f"{stage}_observation"]
        assert observation["evidence_digest"] == hashlib.sha256(payload).hexdigest()
        assert artifact["comparison"]["mismatches"] == []
        assert artifact["champion_sandbox"]["termination"] == "success"
        assert artifact["challenger_sandbox"]["termination"] == "success"
        assert artifact["trace_request_count"] == observation["sample_count"]

    champion_manifest = json.loads((tmp_path / "demo/run/run_manifest.json").read_text())
    challenger_manifest = json.loads(
        (tmp_path / "demo/evolution/challenger-run/run_manifest.json").read_text()
    )
    assert (champion_manifest["seed"], champion_manifest["runtime_seed"]) == (73131, 73129)
    assert (challenger_manifest["seed"], challenger_manifest["runtime_seed"]) == (
        73132,
        73129,
    )

    snapshot = EvolutionSnapshot.model_validate_json(
        (tmp_path / "demo/evolution/promoted-snapshot.json").read_bytes(), strict=True
    )
    assert snapshot.previous_champion is not None
    challenger_record = snapshot.challengers[0]
    observation = challenger_record.shadow_observation
    assert observation is not None
    evidence_root = tmp_path / "demo/evolution/runtime-gates"
    validator = local_gate_evidence_validator(evidence_root)
    assert validator(observation, challenger_record.spec, snapshot.previous_champion)

    stage_root = evidence_root / "shadow"
    artifact_path = stage_root / "gate-evidence.json"
    artifact_payload = artifact_path.read_bytes()
    artifact = json.loads(artifact_payload)

    forged_champion = snapshot.previous_champion.model_copy(update={"capsule_digest": "0" * 64})
    assert not validator(observation, challenger_record.spec, forged_champion)

    forged_artifact = dict(artifact)
    forged_artifact["challenger_runtime_bundle_digest"] = "0" * 64
    forged_artifact_payload = canonical_json(forged_artifact) + b"\n"
    artifact_path.write_bytes(forged_artifact_payload)
    forged_observation = observation.model_copy(
        update={"evidence_digest": hashlib.sha256(forged_artifact_payload).hexdigest()}
    )
    assert not validator(forged_observation, challenger_record.spec, snapshot.previous_champion)
    artifact_path.write_bytes(artifact_payload)

    trace_path = stage_root / "trace.json"
    trace_payload = trace_path.read_bytes()
    forged_trace = json.loads(trace_payload)
    forged_trace["requests"][0]["text"] += "-forged"
    forged_trace_payload = canonical_json(forged_trace) + b"\n"
    trace_path.write_bytes(forged_trace_payload)
    forged_artifact = dict(artifact)
    forged_artifact["trace_sha256"] = hashlib.sha256(forged_trace_payload).hexdigest()
    forged_artifact_payload = canonical_json(forged_artifact) + b"\n"
    artifact_path.write_bytes(forged_artifact_payload)
    forged_observation = observation.model_copy(
        update={"evidence_digest": hashlib.sha256(forged_artifact_payload).hexdigest()}
    )
    assert not validator(forged_observation, challenger_record.spec, snapshot.previous_champion)
    trace_path.write_bytes(trace_payload)
    artifact_path.write_bytes(artifact_payload)

    raw_paths = (
        stage_root / "champion/runtime-observation.json",
        stage_root / "challenger/runtime-observation.json",
    )
    raw_payloads = tuple(path.read_bytes() for path in raw_paths)
    forged_raw_documents = []
    forged_raw_payloads = []
    for raw_payload in raw_payloads:
        document = json.loads(raw_payload)
        document["cases"][0]["token_ids"] = [999_999]
        document["cases"][0]["token_count"] = 1
        forged_raw_documents.append(document)
        forged_raw_payloads.append(canonical_json(document) + b"\n")
    for path, payload in zip(raw_paths, forged_raw_payloads, strict=True):
        path.write_bytes(payload)
    forged_artifact = dict(artifact)
    forged_artifact["champion_observation"] = forged_raw_documents[0]
    forged_artifact["challenger_observation"] = forged_raw_documents[1]
    forged_artifact["champion_observation_sha256"] = hashlib.sha256(
        forged_raw_payloads[0]
    ).hexdigest()
    forged_artifact["challenger_observation_sha256"] = hashlib.sha256(
        forged_raw_payloads[1]
    ).hexdigest()
    forged_artifact_payload = canonical_json(forged_artifact) + b"\n"
    artifact_path.write_bytes(forged_artifact_payload)
    forged_observation = observation.model_copy(
        update={"evidence_digest": hashlib.sha256(forged_artifact_payload).hexdigest()}
    )
    assert not validator(forged_observation, challenger_record.spec, snapshot.previous_champion)


def test_demo_reset_rejects_symlink_output(tmp_path: Path) -> None:
    target = tmp_path / "existing-target"
    target.mkdir()
    marker = target / "preserve.txt"
    marker.write_text("preserve", encoding="utf-8")
    output = tmp_path / "linked-output"
    output.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinked output"):
        run_genesis_demo(output, seed=73129, reset=True)
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_demo_and_evaluation_reject_symlink_output_without_reset(tmp_path: Path) -> None:
    target = tmp_path / "empty-target"
    target.mkdir()
    output = tmp_path / "linked-output"
    output.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinked output"):
        run_genesis_demo(output, seed=73129)
    with pytest.raises(ValueError, match="symlinked evaluation"):
        run_genesis_evaluation(output, seed=73129, count=2)
    assert not any(target.iterdir())
