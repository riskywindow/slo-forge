from __future__ import annotations

from pathlib import Path

import pytest

from sloforge.reports import ArtifactIndex, build_artifact_index


def test_artifact_index_detects_tampering(tmp_path: Path) -> None:
    artifact = tmp_path / "raw.json"
    artifact.write_text('{"measurement": 1}\n', encoding="utf-8")
    index_path = tmp_path / "index.json"
    build_artifact_index(
        repository_root=tmp_path,
        artifacts={"raw": (artifact, "application/json")},
        output=index_path,
    )
    index = ArtifactIndex.model_validate_json(index_path.read_text(encoding="utf-8"))
    assert index.path_for("raw", repository_root=tmp_path) == artifact
    artifact.write_text('{"measurement": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        index.path_for("raw", repository_root=tmp_path)
