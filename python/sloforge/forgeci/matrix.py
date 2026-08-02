"""Offline-by-default ForgeCI benchmark matrix orchestration."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from sloforge.forgeci.models import BenchmarkMatrix, MatrixRunRecord
from sloforge.forgeci.runner import run_case


def _git(*arguments: str, cwd: Path, timeout: float = 60.0) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"git {' '.join(arguments)} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def run_matrix(
    matrix: BenchmarkMatrix,
    *,
    output_directory: Path,
    repository_base: Path,
    allow_network: bool = False,
) -> MatrixRunRecord:
    """Execute a matrix in isolated clones; external repositories require opt-in."""

    started_at = datetime.now(UTC)
    output_directory.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix="sloforge-forgeci-matrix-"))
    runs = []
    warnings: list[str] = []
    try:
        for index, case in enumerate(matrix.cases):
            source_path = Path(case.repository)
            if not source_path.is_absolute():
                source_path = (repository_base / source_path).resolve()
            if source_path.exists():
                source = str(source_path)
            elif not allow_network:
                raise ValueError(
                    f"case {case.case_id} repository is not local; network execution is disabled"
                )
            else:
                source = case.repository
            checkout = temporary_root / f"case-{index:04d}"
            _git("clone", "--no-hardlinks", "--quiet", source, str(checkout), cwd=temporary_root)
            revision = _git("rev-parse", "--verify", f"{case.revision}^{{commit}}", cwd=checkout)
            _git("checkout", "--detach", "--force", revision, cwd=checkout)
            resolved = case.model_copy(update={"repository": source, "revision": revision})
            run = run_case(
                resolved,
                checkout=checkout,
                output_directory=output_directory / "runs",
            )
            runs.append(run)
            warnings.extend(f"{case.case_id}: {warning}" for warning in run.warnings)
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)
    record = MatrixRunRecord(
        matrix_id=matrix.matrix_id,
        started_at=started_at,
        completed_at=datetime.now(UTC),
        success=bool(runs) and all(run.success for run in runs),
        runs=tuple(runs),
        artifact_directory=str(output_directory),
        warnings=tuple(warnings),
    )
    (output_directory / "matrix-run.json").write_text(
        record.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return record
