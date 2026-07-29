"""Fail-closed verification for the BranchFabric no-build completion branch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


class NoBuildVerificationError(RuntimeError):
    """The repository does not satisfy the BranchFabric no-build contract."""


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise NoBuildVerificationError(f"cannot read required JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise NoBuildVerificationError(f"required JSON artifact is not an object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NoBuildVerificationError(message)


def verify_no_build(repository_root: Path) -> dict[str, Any]:
    """Verify evidence bindings, final disposition, reports, and hardware absence."""

    root = repository_root.resolve()
    gate_input_path = root / "artifacts/branchfabric/gates/branchfabric_gate_input.json"
    gate_result_path = root / "artifacts/branchfabric/gates/branchfabric_gate_result.json"
    gate_input = _read_object(gate_input_path)
    gate_result = _read_object(gate_result_path)
    _require(gate_input.get("phase") == "final", "gate input is not final")
    _require(gate_result.get("phase") == "final", "gate result is not final")
    _require(gate_result.get("outcome") == "FAIL_NO_BUILD", "gate did not fail no-build")
    _require(
        gate_result.get("required_action") == "TERMINATE_HARDWARE_PATH",
        "gate does not terminate the hardware path",
    )
    _require(not gate_result.get("hardware_implementation_allowed"), "hardware is allowed")
    _require(
        not gate_result.get("functional_model_or_cycle_simulator_allowed"),
        "hardware model or simulator is allowed despite the failed gate",
    )
    _require(gate_result.get("passing_candidates") == [], "gate has passing candidates")
    candidates = gate_result.get("candidates")
    _require(isinstance(candidates, list) and candidates, "gate has no candidate audit")
    _require(
        all(
            isinstance(candidate, dict)
            and candidate.get("disposition") == "NOT_JUSTIFIED"
            and not candidate.get("passes_all_mandatory_and_one_value_gate")
            for candidate in candidates
        ),
        "a gate candidate is not explicitly rejected",
    )

    verified_evidence: dict[str, str] = {}
    for candidate in gate_input.get("candidates", []):
        _require(isinstance(candidate, dict), "gate input candidate is malformed")
        for evidence in candidate.get("evidence", []):
            _require(isinstance(evidence, dict), "gate evidence binding is malformed")
            relative = evidence.get("path")
            expected = evidence.get("sha256")
            _require(isinstance(relative, str), "gate evidence path is missing")
            _require(isinstance(expected, str), f"gate evidence hash is missing: {relative}")
            evidence_path = root / relative
            _require(evidence_path.is_file(), f"gate evidence is absent: {relative}")
            observed = _sha256(evidence_path)
            _require(observed == expected, f"gate evidence hash mismatch: {relative}")
            verified_evidence[relative] = observed

    for forbidden in (
        "hardware/branchfabric",
        "sim/branchfabric",
        "crates/sloforge-branchfabric-model",
        "crates/sloforge-branchfabric-driver",
    ):
        _require(not (root / forbidden).exists(), f"forbidden no-build path exists: {forbidden}")

    spend = _read_object(root / "artifacts/branchfabric/execution/ledgers/spend-ledger.json")
    cleanup = _read_object(root / "artifacts/branchfabric/execution/ledgers/cleanup-ledger.json")
    _require(float(spend.get("total_authorized", -1)) == 0.0, "authorized spend is nonzero")
    _require(float(spend.get("spent", -1)) == 0.0, "BranchFabric spend is nonzero")
    cleanup_entries = cleanup.get("entries")
    _require(isinstance(cleanup_entries, list) and cleanup_entries, "cleanup ledger is empty")
    _require(
        all(
            isinstance(entry, dict)
            and entry.get("billable_resources_remaining") == 0
            and entry.get("sloforge_experiment_processes_remaining") == 0
            and not entry.get("fault_configuration_active")
            for entry in cleanup_entries
        ),
        "cleanup ledger reports a remaining resource, process, or fault",
    )

    report_paths = (
        "BRANCHFABRIC_GATE_REPORT.md",
        "BRANCHFABRIC_NO_BUILD_REPORT.md",
        "BRANCHFABRIC_FINAL_REPORT.md",
        "docs/branchfabric/FUTURE_MEASUREMENT_PLAN.md",
    )
    report_hashes: dict[str, str] = {}
    for relative in report_paths:
        report_path = root / relative
        _require(report_path.is_file(), f"required no-build report is absent: {relative}")
        text = report_path.read_text(encoding="utf-8")
        _require(
            "FAIL_NO_BUILD" in text or relative.endswith("FUTURE_MEASUREMENT_PLAN.md"),
            f"report does not state the no-build result: {relative}",
        )
        report_hashes[relative] = _sha256(report_path)

    return {
        "schema_version": "sloforge.branchfabric.no-build-verification/v1",
        "baseline_commit": gate_result.get("baseline_commit"),
        "gate_outcome": gate_result.get("outcome"),
        "required_action": gate_result.get("required_action"),
        "passing_candidates": [],
        "target_hardware_implemented": False,
        "functional_model_or_cycle_simulator_implemented": False,
        "verified_gate_input_sha256": _sha256(gate_input_path),
        "verified_gate_result_sha256": _sha256(gate_result_path),
        "verified_evidence": dict(sorted(verified_evidence.items())),
        "verified_reports": report_hashes,
        "billable_resources_remaining": 0,
        "experiment_processes_remaining": 0,
        "fault_configuration_active": False,
        "status": "PASS_VERIFIED_NO_BUILD",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = verify_no_build(arguments.repository_root)
    output = arguments.output
    if not output.is_absolute():
        output = arguments.repository_root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
