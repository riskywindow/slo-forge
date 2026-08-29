#!/usr/bin/env python3
"""Build, seal, or verify the local Experiment 004 v10 authorization."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_PYTHON = _ROOT / "python"
if str(_PYTHON) not in sys.path:
    sys.path.insert(0, str(_PYTHON))

from sloforge.helix.characterization.gpu_reclamation_v10_authorization import (  # noqa: E402
    authorization_subject_sha256,
    build_authorization_subject,
    canonical_json_bytes,
    file_sha256,
    seal_authorization,
    verify_authorization_file,
)

_EXPERIMENT_ROOT = _ROOT / "artifacts/branchfabric/gpu-validation/experiment-004"
_CALIBRATION_ROOT = (
    _EXPERIMENT_ROOT / "raw/modal/exp004-v10-integrated-s41-v3/calibration"
)
_DEFAULT_SUBJECT = _EXPERIMENT_ROOT / "reviews/v10-authorization-subject.json"
_DEFAULT_AUTHORIZATION = _EXPERIMENT_ROOT / "v10-authorization.json"
_SOURCE_ARTIFACTS = (
    ("selected_load", _CALIBRATION_ROOT / "selected-load.json"),
    ("calibration_result", _CALIBRATION_ROOT / "calibration-result.json"),
    (
        "lambda_1_plan",
        _CALIBRATION_ROOT / "one-gpu/capacity-04-gpu0-only/plan.json",
    ),
    (
        "lambda_1_raw",
        _CALIBRATION_ROOT / "one-gpu/capacity-04-gpu0-only/raw.json",
    ),
    (
        "lambda_1_result",
        _CALIBRATION_ROOT / "one-gpu/capacity-04-gpu0-only/result.json",
    ),
    (
        "lambda_spike_plan",
        _CALIBRATION_ROOT / "one-gpu/capacity-02-gpu0-only/plan.json",
    ),
    (
        "lambda_spike_raw",
        _CALIBRATION_ROOT / "one-gpu/capacity-02-gpu0-only/raw.json",
    ),
    (
        "lambda_spike_result",
        _CALIBRATION_ROOT / "one-gpu/capacity-02-gpu0-only/result.json",
    ),
    (
        "lambda_2_plan",
        _CALIBRATION_ROOT / "two-gpu/capacity-05-two-gpu-round-robin/plan.json",
    ),
    (
        "lambda_2_raw",
        _CALIBRATION_ROOT / "two-gpu/capacity-05-two-gpu-round-robin/raw.json",
    ),
    (
        "lambda_2_result",
        _CALIBRATION_ROOT / "two-gpu/capacity-05-two-gpu-round-robin/result.json",
    ),
)


def _relative(path: Path) -> str:
    return path.relative_to(_ROOT).as_posix()


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=10.0,
    )
    commit = completed.stdout.strip()
    if completed.returncode != 0 or len(commit) != 40:
        raise RuntimeError(f"cannot resolve repository commit: {completed.stderr.strip()}")
    return commit


def _source_rows() -> list[dict[str, str]]:
    rows = []
    for role, path in _SOURCE_ARTIFACTS:
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"source calibration artifact is absent or unsafe: {path}")
        rows.append(
            {
                "role": role,
                "artifact_reference": _relative(path),
                "artifact_sha256": file_sha256(path),
            }
        )
    return rows


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read JSON object {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"expected one JSON object in {path}")
    return value


def _write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(canonical_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _review_decision(path: Path) -> dict[str, Any]:
    review = _read_object(path)
    return {
        **review,
        "artifact_reference": _relative(path.resolve(strict=True)),
        "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _build_subject(output: Path) -> dict[str, Any]:
    subject = build_authorization_subject(
        code_commit=_git_commit(), source_calibration_artifact_hashes=_source_rows()
    )
    payload = {
        **subject,
        "authorization_subject_sha256": authorization_subject_sha256(subject),
    }
    _write_new(output, payload)
    return payload


def _seal(subject_path: Path, review_paths: list[Path], output: Path) -> dict[str, Any]:
    subject = _read_object(subject_path)
    expected_subject_hash = subject.pop("authorization_subject_sha256", None)
    actual_subject_hash = authorization_subject_sha256(subject)
    if expected_subject_hash != actual_subject_hash:
        raise RuntimeError("authorization subject hash does not match its canonical content")
    authorization = seal_authorization(
        subject=subject,
        reviews=[_review_decision(path) for path in review_paths],
    )
    _write_new(output, authorization)
    return authorization


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subject_parser = subparsers.add_parser("subject")
    subject_parser.add_argument("--output", type=Path, default=_DEFAULT_SUBJECT)

    seal_parser = subparsers.add_parser("seal")
    seal_parser.add_argument("--subject", type=Path, default=_DEFAULT_SUBJECT)
    seal_parser.add_argument("--review", type=Path, action="append", required=True)
    seal_parser.add_argument("--output", type=Path, default=_DEFAULT_AUTHORIZATION)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--authorization", type=Path, default=_DEFAULT_AUTHORIZATION)
    verify_parser.add_argument("--expected-artifact-hash")

    arguments = parser.parse_args()
    if arguments.command == "subject":
        payload = _build_subject(arguments.output.resolve())
    elif arguments.command == "seal":
        if len(arguments.review) != 3:
            parser.error("seal requires exactly three --review arguments")
        payload = _seal(
            arguments.subject.resolve(),
            [path.resolve() for path in arguments.review],
            arguments.output.resolve(),
        )
    else:
        payload = verify_authorization_file(
            arguments.authorization.resolve(),
            repository_root=_ROOT,
            expected_artifact_hash=arguments.expected_artifact_hash,
        )
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
