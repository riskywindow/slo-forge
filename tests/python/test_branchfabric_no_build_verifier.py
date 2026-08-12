from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sloforge.helix.characterization.no_build_verifier import (
    NoBuildVerificationError,
    verify_no_build,
)


def _write(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True).encode()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _fixture(root: Path) -> None:
    evidence = root / "raw.json"
    evidence_hash = _write(evidence, {"samples": [1, 2, 3]})
    candidate = {
        "candidate": "state_copy",
        "evidence": [{"path": "raw.json", "sha256": evidence_hash}],
    }
    _write(
        root / "artifacts/branchfabric/gates/branchfabric_gate_input.json",
        {"phase": "final", "candidates": [candidate]},
    )
    _write(
        root / "artifacts/branchfabric/gates/branchfabric_gate_result.json",
        {
            "phase": "final",
            "baseline_commit": "a" * 40,
            "outcome": "FAIL_NO_BUILD",
            "required_action": "TERMINATE_HARDWARE_PATH",
            "hardware_implementation_allowed": False,
            "functional_model_or_cycle_simulator_allowed": False,
            "passing_candidates": [],
            "candidates": [
                {
                    "disposition": "NOT_JUSTIFIED",
                    "passes_all_mandatory_and_one_value_gate": False,
                }
            ],
        },
    )
    _write(
        root / "artifacts/branchfabric/execution/ledgers/spend-ledger.json",
        {"total_authorized": 0.0, "spent": 0.0},
    )
    _write(
        root / "artifacts/branchfabric/execution/ledgers/cleanup-ledger.json",
        {
            "entries": [
                {
                    "billable_resources_remaining": 0,
                    "sloforge_experiment_processes_remaining": 0,
                    "fault_configuration_active": False,
                }
            ]
        },
    )
    for relative in (
        "BRANCHFABRIC_GATE_REPORT.md",
        "BRANCHFABRIC_NO_BUILD_REPORT.md",
        "BRANCHFABRIC_FINAL_REPORT.md",
    ):
        path = root / relative
        path.write_text("FAIL_NO_BUILD\n", encoding="utf-8")
    future = root / "docs/branchfabric/FUTURE_MEASUREMENT_PLAN.md"
    future.parent.mkdir(parents=True, exist_ok=True)
    future.write_text("future measurement plan\n", encoding="utf-8")


def test_verified_no_build_requires_failed_gate_and_absent_hardware(tmp_path: Path) -> None:
    _fixture(tmp_path)
    result = verify_no_build(tmp_path)
    assert result["status"] == "PASS_VERIFIED_NO_BUILD"
    assert result["target_hardware_implemented"] is False

    forbidden = tmp_path / "hardware/branchfabric"
    forbidden.mkdir(parents=True)
    with pytest.raises(NoBuildVerificationError, match="forbidden no-build path"):
        verify_no_build(tmp_path)


def test_verified_no_build_rejects_mutated_raw_evidence(tmp_path: Path) -> None:
    _fixture(tmp_path)
    (tmp_path / "raw.json").write_text("{}", encoding="utf-8")
    with pytest.raises(NoBuildVerificationError, match="hash mismatch"):
        verify_no_build(tmp_path)
