from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from sloforge.continuum.benchmarking import (
    EvaluationRequest,
    run_evaluation,
    run_evaluation_campaign,
)
from sloforge.continuum.reports import generate_reports


def _request(output: Path) -> EvaluationRequest:
    return EvaluationRequest(
        output_dir=output,
        seeds=(11, 22, 33),
        git_commit="7e51ea7f7338755d23f889820558a4e046d6c42e",
        initial_output_tokens=8,
        delta_rounds=(1,),
        resumed_tokens=2,
        converter_repetitions=3,
    )


def test_reports_are_static_scoped_and_derived_from_validated_raw_artifacts(
    tmp_path: Path,
) -> None:
    root = tmp_path / "campaign"
    campaign = run_evaluation_campaign(_request(root))
    references = campaign.reports.model_dump(mode="python").values()
    for value in references:
        path = root / value["path"]
        assert path.is_file()
        assert path.stat().st_size == value["size_bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == value["sha256"]

    evaluation = (root / campaign.reports.evaluation_markdown.path).read_text()
    html = (root / campaign.reports.evaluation_html.path).read_text()
    compatibility = (root / campaign.reports.compatibility_markdown.path).read_text()
    faults = (root / campaign.reports.fault_tolerance_markdown.path).read_text()
    adapters = (root / campaign.reports.runtime_adapters_markdown.path).read_text()

    assert "Observed host timings" in evaluation
    assert "synthetic_protocol" in evaluation
    assert "Pre-copy pause ms" in evaluation
    assert "H4 | pass" in evaluation
    assert "No GPU, RDMA" in evaluation
    assert "No GPU, RDMA, cloud, or multi-node result is claimed" in html
    assert "Unsafe direct-reuse acceptances: **0**" in compatibility
    assert "does not claim an executed changed-weight migration" in compatibility
    assert "ROLLED_BACK" in faults
    assert "exactly-once acceptance at the SLOForge gateway" in faults
    assert "continuum-reference-token-major" in adapters
    assert "Public API discovery does not constitute migration validation" in adapters


def test_report_generation_rejects_tampered_seed_artifact(tmp_path: Path) -> None:
    root = tmp_path / "tamper"
    evaluation = run_evaluation(_request(root))
    victim = root / evaluation.per_seed[0].flagship_artifact.path
    victim.write_bytes(victim.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="integrity failed"):
        generate_reports(evaluation, root=root)


def test_report_generation_recomputes_summary_aggregates_from_raw_seed_files(
    tmp_path: Path,
) -> None:
    root = tmp_path / "aggregate-tamper"
    evaluation = run_evaluation(_request(root))
    first = evaluation.confidence_intervals[0]
    altered = first.model_copy(
        update={
            "mean": first.mean + 500.0,
            "lower": first.mean + 499.0,
            "upper": first.mean + 501.0,
        }
    )
    tampered = evaluation.model_copy(
        update={"confidence_intervals": (altered, *evaluation.confidence_intervals[1:])}
    )

    with pytest.raises(ValueError, match="confidence interval summary differs"):
        generate_reports(tampered, root=root)
