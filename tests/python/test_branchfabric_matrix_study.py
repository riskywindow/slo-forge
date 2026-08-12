from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sloforge.helix.characterization.matrix_study import (
    evaluate_controlled_matrix,
    write_controlled_matrix_study,
)

MATRIX = Path("benchmarks/branchfabric/characterization.yaml")
CALIBRATION = Path(
    "artifacts/branchfabric/baseline/helix-demo-seed-41/capture/"
    "coding-failure-capture.continuum.json"
)


def test_controlled_matrix_study_evaluates_every_declared_cell(tmp_path: Path) -> None:
    study = evaluate_controlled_matrix(MATRIX, calibration_artifact=CALIBRATION)
    assert study.case_count == 855
    assert study.evaluated_case_count == 855
    assert study.hardware_executed_case_count == 0
    assert study.distribution_claim is False
    assert len(study.cases) == 855
    assert {cell.execution_class for cell in study.cases} == {"ANALYTICAL_SYNTHETIC_STATE_MODEL"}
    assert all(not cell.distribution_claim for cell in study.cases)

    output = tmp_path / "matrix-study.json"
    digest = write_controlled_matrix_study(study, output)
    assert digest == hashlib.sha256(output.read_bytes()).hexdigest()
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["case_count"] == document["evaluated_case_count"]
