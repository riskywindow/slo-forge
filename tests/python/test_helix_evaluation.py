from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from sloforge.helix.evaluation import (
    EvaluationRun,
    run_reference_evaluation,
    validate_reference_evaluation,
)
from sloforge.helix.scheduler import SchedulerPolicy


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_canonical_bytes(value) + b"\n")


def _reseal_evaluation(document: dict[str, object]) -> None:
    identity = {key: value for key, value in document.items() if key != "evaluation_id"}
    document["evaluation_id"] = hashlib.sha256(_canonical_bytes(identity)).hexdigest()


def test_reference_evaluation_rejects_invalid_seed_matrix(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="two to 32 unique seeds"):
        run_reference_evaluation(
            tmp_path / "evaluation",
            reports=tmp_path / "reports",
            workload=Path("scenarios/helix/resource/cpu-learning-aware.json"),
            seeds=(41,),
        )


def test_reference_evaluation_refuses_nonempty_output(tmp_path: Path) -> None:
    output = tmp_path / "evaluation"
    output.mkdir()
    (output / "unrelated.txt").write_text("preserve me")
    with pytest.raises(FileExistsError, match="absent or empty"):
        run_reference_evaluation(
            output,
            reports=tmp_path / "reports",
            workload=Path("scenarios/helix/resource/cpu-learning-aware.json"),
            seeds=(41, 73),
        )


def test_reference_evaluation_is_multi_seed_provenance_sealed_and_honest(
    tmp_path: Path,
) -> None:
    output = tmp_path / "evaluation"
    reports = tmp_path / "reports"
    run = run_reference_evaluation(
        output,
        reports=reports,
        workload=Path("scenarios/helix/resource/cpu-learning-aware.json"),
        seeds=(41, 73),
    )

    assert len(run.scheduler_comparisons) == 2 * len(SchedulerPolicy)
    assert {(item.seed, item.policy) for item in run.scheduler_comparisons} == {
        (seed, policy) for seed in run.seeds for policy in SchedulerPolicy
    }
    assert all(item.serving_predicted_slo_feasible for item in run.scheduler_comparisons)
    assert run.success_rate_delta.method == "two_sided_student_t_mean"
    assert run.success_rate_delta.sample_count == 2
    assert len(run.seed_observations) == 2
    assert all((output / item.path).is_file() for item in run.raw_artifacts)
    assert all(
        hashlib.sha256((output / item.path).read_bytes()).hexdigest() == item.sha256
        for item in run.raw_artifacts
    )
    assert EvaluationRun.model_validate_json((output / "evaluation.json").read_bytes()) == run
    assert validate_reference_evaluation(output) == run

    hypotheses = {item.hypothesis_id: item for item in run.hypotheses}
    assert hypotheses["H3"].status == "not_supported"
    assert hypotheses["H4"].status == "supported_within_scope"
    assert hypotheses["H6"].status == "supported_within_scope"
    assert hypotheses["H8"].status == "partial"
    assert hypotheses["H9"].status == "inconclusive"
    assert all(hypotheses[item].artifact_paths for item in ("H3", "H4", "H6", "H8", "H9"))
    report = (reports / "helix-evaluation.md").read_text()
    assert "Student-t 95% interval" in report
    assert "modeled predictions, not measurements" in report
    assert "Raw provenance" in report

    payload = run.model_dump(mode="python")
    payload["limitations"] = (*run.limitations, "tampered claim")
    with pytest.raises(ValidationError, match="evaluation identifier is invalid"):
        EvaluationRun.model_validate(payload)

    evaluation_path = output / "evaluation.json"
    original_evaluation = evaluation_path.read_bytes()
    hypothesis_document = json.loads(original_evaluation)
    hypothesis_document["hypotheses"][8]["status"] = "partial"
    hypothesis_document["hypotheses"][8]["observations"][0] = (
        "re-signed but unsupported hypothesis narrative"
    )
    _reseal_evaluation(hypothesis_document)
    _write_json(evaluation_path, hypothesis_document)
    with pytest.raises(ValueError, match="hypothesis summaries disagree"):
        validate_reference_evaluation(output)
    evaluation_path.write_bytes(original_evaluation)

    summary_artifact = next(
        item for item in run.raw_artifacts if item.artifact_kind == "flagship_summary"
    )
    summary_path = output / summary_artifact.path
    original_summary = summary_path.read_bytes()
    summary_path.write_bytes(original_summary + b" ")
    with pytest.raises(ValueError, match="raw evaluation artifact digest mismatch"):
        validate_reference_evaluation(output)
    summary_path.write_bytes(original_summary)

    summary_document = json.loads(original_summary)
    summary_document["replay"]["joint"]["exact_identity_verified"] = False
    _write_json(summary_path, summary_document)
    changed_digest = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    evaluation_document = json.loads(evaluation_path.read_bytes())
    for artifact in evaluation_document["raw_artifacts"]:
        if artifact["path"] == summary_artifact.path:
            artifact["sha256"] = changed_digest
    for observation in evaluation_document["seed_observations"]:
        if observation["seed"] == summary_artifact.seed:
            observation["summary_artifact_sha256"] = changed_digest
    _reseal_evaluation(evaluation_document)
    _write_json(evaluation_path, evaluation_document)
    with pytest.raises(ValueError, match="aggregate counts disagree"):
        validate_reference_evaluation(output)
