from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from sloforge.helix.characterization.orchestration import (
    CharacterizationStage,
    OrchestrationConfig,
    RunState,
    create_characterization_run,
)
from sloforge.helix.characterization.runner import run_continuum_trace
from sloforge.helix.characterization.workflow import (
    analyze_cow,
    analyze_workload,
    derive_requirements,
    write_characterization_report,
)

VERTICAL = Path("artifacts/branchfabric/characterization/first-vertical-seed-41-rerun")
MATRIX = Path("benchmarks/branchfabric/characterization.yaml")
HARDWARE = Path("artifacts/branchfabric/manifests/hardware-baseline.json")
SOFTWARE = Path("artifacts/branchfabric/manifests/software-baseline.json")


def _completed_evidence_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    run = tmp_path / "run"
    create_characterization_run(
        run,
        matrix_path=MATRIX,
        config=OrchestrationConfig(
            seed=41,
            stages=(CharacterizationStage.MATRIX_VALIDATE,),
            maximum_attempts=1,
            stage_timeout_seconds=30,
        ),
        hardware_capability_manifest=HARDWARE,
        software_manifest=SOFTWARE,
    )
    vertical_attempt = run / "attempts" / "vertical_trace" / "attempt-000"
    vertical = vertical_attempt / "vertical-trace"
    vertical.mkdir(parents=True)
    for name in (
        "trace-manifest-v1.json",
        "branch-workload-trace-v1.jsonl",
        "state-operation-trace-v1.jsonl",
        "sharing-analysis.json",
    ):
        shutil.copyfile(VERTICAL / name, vertical / name)
    (vertical_attempt / "worker-result.json").write_text(
        json.dumps({"status": "SUCCEEDED"}), encoding="utf-8"
    )

    continuum_attempt = run / "attempts" / "continuum_state" / "attempt-000"
    continuum_attempt.mkdir(parents=True)
    run_continuum_trace(
        continuum_attempt / "continuum-trace",
        seed=41,
        hardware_baseline=run / "inputs" / "hardware-capability.json",
        software_baseline=run / "inputs" / "software-baseline.json",
    )
    (continuum_attempt / "worker-result.json").write_text(
        json.dumps({"status": "SUCCEEDED"}), encoding="utf-8"
    )

    metadata_attempt = run / "attempts" / "metadata" / "attempt-000"
    metadata_attempt.mkdir(parents=True)
    (metadata_attempt / "metadata-study.json").write_text(
        json.dumps({"summaries": []}), encoding="utf-8"
    )
    (metadata_attempt / "worker-result.json").write_text(
        json.dumps({"status": "SUCCEEDED"}), encoding="utf-8"
    )
    monkeypatch.setattr(
        "sloforge.helix.characterization.workflow.characterization_run_result",
        lambda _run: SimpleNamespace(run_state=RunState.COMPLETE),
    )
    return run


def test_workload_analysis_preserves_trace_evidence(tmp_path: Path) -> None:
    trace = VERTICAL / "branch-workload-trace-v1.jsonl"
    result = analyze_workload(
        trace=trace,
        output=tmp_path / "workload",
        seed=41,
        max_events=1000,
        replace=False,
    )
    artifact = tmp_path / "workload" / "workload-analysis.json"
    document = json.loads(artifact.read_text())
    assert result["artifact_reference"] == trace.as_posix()
    assert document["branch_fanout"]["p50"] == 4
    assert document["branch_lifetime_ns"]["sample_count"] == 4
    assert document["workload_provenance"] == ["SYNTHETIC"]


def test_cow_analysis_is_a_controlled_projection_not_a_distribution(tmp_path: Path) -> None:
    trace = VERTICAL / "state-operation-trace-v1.jsonl"
    result = analyze_cow(
        trace=trace,
        output=tmp_path / "cow.json",
        page_sizes=(4096, 65536),
        seed=41,
        max_events=1000,
        replace=False,
    )
    assert result["projection_count"] == 8 * 6 * 2
    assert result["distribution_claim"] is False
    recommendation = result["recommendation"]
    assert isinstance(recommendation, dict)
    assert recommendation["universal_page_size_supported"] is False
    with pytest.raises(FileExistsError):
        analyze_cow(
            trace=trace,
            output=tmp_path / "cow.json",
            page_sizes=(4096,),
            seed=41,
            max_events=1000,
            replace=False,
        )


def test_requirements_and_report_compile_only_bound_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _completed_evidence_run(tmp_path, monkeypatch)
    result = derive_requirements(
        run=run,
        output=tmp_path / "requirements",
        replace=False,
    )
    requirements_path = tmp_path / "requirements" / "branchfabric_requirements.json"
    requirements = json.loads(requirements_path.read_text(encoding="utf-8"))

    assert result["trace_corpus_hash"] == requirements["trace_corpus_hash"]
    assert requirements["state"]["branch_fanout"]["p50"]["value"] == 4
    assert requirements["state"]["divergence_rate"]["p50"]["availability"] == "UNKNOWN"
    assert requirements["bandwidth_targets"][0]["mean"]["availability"] == "UNAVAILABLE"
    assert requirements["recommended_isa"] == []
    assert len(requirements["not_justified_operations"]) == 7
    cow_isa = next(
        item
        for item in requirements["unresolved_isa_operations"]
        if item["operation"] == "STATE_COW"
    )
    assert cow_isa["classification"]["availability"] == "UNKNOWN"
    assert cow_isa["expected_end_to_end_speedup"]["availability"] == "AVAILABLE"
    assert cow_isa["expected_end_to_end_speedup"]["value"] >= 1.0
    assert sum(item["included_in_trace_corpus"] for item in requirements["verified_artifacts"]) == 3

    report_result = write_characterization_report(
        run=run,
        output=tmp_path / "report",
        replace=False,
    )
    report = (tmp_path / "report" / "CHARACTERIZATION_REPORT.md").read_text(encoding="utf-8")
    assert report_result["trace_corpus_hash"] == requirements["trace_corpus_hash"]
    assert "No hardware primitive is promoted from this run alone" in report
    assert "GPU status: unavailable" in report
    with pytest.raises(FileExistsError):
        write_characterization_report(run=run, output=tmp_path / "report", replace=False)


def test_requirements_reject_trace_changed_after_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _completed_evidence_run(tmp_path, monkeypatch)
    trace = (
        run
        / "attempts"
        / "vertical_trace"
        / "attempt-000"
        / "vertical-trace"
        / "state-operation-trace-v1.jsonl"
    )
    trace.write_bytes(b" " + trace.read_bytes())

    with pytest.raises(ValueError, match="immutable manifest"):
        derive_requirements(run=run, output=tmp_path / "requirements", replace=False)
