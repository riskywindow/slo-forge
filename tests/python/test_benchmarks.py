from __future__ import annotations

import json
from pathlib import Path

import pytest

from sloforge.benchmarks import CPU_OUTPUTS, benchmark_gpu, finalize_cpu_report


def test_finalize_cpu_requires_verified_manifest(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for name in CPU_OUTPUTS:
        (source / name).write_text("{}\n", encoding="utf-8")
    (source / "report-manifest.json").write_text(
        json.dumps({"schema_version": "sloforge.report/v1", "verified_artifact_count": 0}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not attest"):
        finalize_cpu_report(source=source, output=tmp_path / "output")


def test_gpu_status_never_invents_measurements(tmp_path: Path) -> None:
    result = benchmark_gpu(output=tmp_path / "gpu")
    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["measurements"] == []
    assert payload["status"] in {"unavailable", "ready-not-executed"}
    assert "No GPU performance numbers" in (tmp_path / "gpu" / "evaluation.md").read_text()
