#!/usr/bin/env python3
"""Finalize Experiment 004 v10 only after ledger settlement and Modal cleanup.

The paid remote function cannot observe its own provider teardown or the local
GPU-hour ledger settlement. This CPU-only finalizer requires both as immutable
inputs and writes the canonical scientific-validity artifact fail closed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_EXPERIMENT_ROOT = _ROOT / "artifacts/branchfabric/gpu-validation/experiment-004"
sys.path.insert(0, str(_ROOT / "python"))
sys.path.insert(0, str(_ROOT / "experiments/branchfabric"))


def _canonical_bytes(value: Any) -> bytes:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode()


def _write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(_canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"finalizer input is not an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _regular_file(path: Path) -> Path:
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_file():
        raise ValueError(f"v10 prerequisite artifact is not an immutable regular file: {path}")
    return resolved


def _repo_artifact(reference: object) -> Path:
    if not isinstance(reference, str):
        raise ValueError("v10 prerequisite artifact reference is not a string")
    relative = Path(reference)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("v10 prerequisite artifact reference is unsafe")
    return _regular_file(_ROOT / relative)


def _binding(kind: str, path: Path) -> str:
    resolved = _regular_file(path)
    return f"{kind}:{_sha256(resolved)}:{resolved}"


def _write_gpu_phase_ledger(
    *,
    run_root: Path,
    scientific: Any,
    settlement: dict[str, Any],
    invocation_gpu_seconds: float,
    conservative_gpu_seconds_before: float,
    conservative_gpu_seconds_after: float,
    hard_ceiling_gpu_seconds: float,
) -> Path:
    """Materialize before/after accounting for every v10 GPU-active phase.

    The two GPUs remain allocated throughout, so each measured phase consumes
    twice its wall interval.  The enclosing invocation remains the canonical
    conservative ledger charge; phase totals are diagnostic and may overlap
    where serving and state movement execute concurrently.
    """

    from sloforge.helix.characterization.gpu_reclamation_v10_budget import (
        PHASE_ORDER,
        load_v10_phase_budget,
    )
    from sloforge.helix.characterization.gpu_reclamation_v10_validity import V10Phase

    if scientific.timeline is None:
        raise ValueError("scientifically valid v10 lacks its phase timeline")
    timeline = scientific.timeline
    phase_budget = load_v10_phase_budget(
        _EXPERIMENT_ROOT / "v10-phase-budget.json", repository_root=_ROOT
    )
    selected_bounds = {row.phase: row.selected_upper_bound_seconds for row in phase_budget.phases}
    readiness = _load_object(run_root / "readiness/both-engines-ready.json")
    engine_rows = readiness.get("engine_evidence")
    if not isinstance(engine_rows, list) or len(engine_rows) != 2:
        raise ValueError("v10 phase ledger lacks exact two-engine readiness evidence")
    completion = _load_object(run_root / "function-completion.json")
    deadlines = completion.get("absolute_deadlines")
    if not isinstance(deadlines, dict):
        raise ValueError("v10 phase ledger lacks function-entry deadline evidence")
    allocation_start_ns = int(deadlines["function_entry_monotonic_ns"])
    actual_gpu_seconds = settlement.get("actual_gpu_seconds")
    if not isinstance(actual_gpu_seconds, (int, float)) or isinstance(actual_gpu_seconds, bool):
        raise ValueError("v10 budget settlement lacks actual GPU seconds")
    actual_wall_seconds = float(actual_gpu_seconds) / 2.0
    allocation_end_ns = allocation_start_ns + round(actual_wall_seconds * 1e9)
    if not math.isclose(
        float(completion["gpu_seconds"]),
        float(actual_gpu_seconds),
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError("phase ledger actual time differs from function completion")

    sanity_subphases: list[dict[str, Any]] = []
    for rate in (12, 15):
        plan = _load_object(run_root / f"sanity/{rate}-rps/plan.json")
        sanity_subphases.append(
            {
                "name": f"sanity_{rate}rps",
                "before_monotonic_ns": int(plan["warmup_start_ns"]),
                "after_monotonic_ns": int(plan["measurement_end_ns"]),
                "source_evidence": f"sanity/{rate}-rps/plan.json",
            }
        )
    reset_rows = json.loads((run_root / "calibration/final-request-state-reset.json").read_text())
    if (
        not isinstance(reset_rows, list)
        or len(reset_rows) != 2
        or any(not isinstance(row, dict) for row in reset_rows)
    ):
        raise ValueError("v10 phase ledger lacks the exact final reset pair")
    readiness_sanity_end_ns = max(int(row["ended_ns"]) for row in reset_rows)
    rollout_ready = _load_object(run_root / "barriers/rollout.transaction-ready.json")
    rollout_preparation_end_ns = int(rollout_ready["observed_at_monotonic_ns"])
    serving_result = _load_object(run_root / "serving/result.json")

    def event(phase: V10Phase) -> int:
        return timeline.timestamp(phase)

    measured = (
        (
            "two_engine_readiness_sanity_check",
            allocation_start_ns,
            readiness_sanity_end_ns,
            (
                "readiness/both-engines-ready.json + sanity/* + "
                "calibration/final-request-state-reset.json"
            ),
            sanity_subphases,
        ),
        (
            "rollout_state_preparation",
            readiness_sanity_end_ns,
            rollout_preparation_end_ns,
            "barriers/rollout.transaction-ready.json#observed_at_monotonic_ns",
            (),
        ),
        (
            "control_interval",
            int(serving_result["start_ns"]),
            event(V10Phase.LOAD_SPIKE_BEGIN),
            "analysis:control",
            (),
        ),
        (
            "overload_interval",
            event(V10Phase.LOAD_SPIKE_BEGIN),
            event(V10Phase.RECLAIM_TRIGGER),
            "analysis:overload",
            (),
        ),
        (
            "branch_quiesce",
            event(V10Phase.BRANCH_QUIESCE_BEGIN),
            event(V10Phase.BRANCH_QUIESCE_END),
            "timeline",
            (),
        ),
        (
            "state_capture",
            event(V10Phase.STATE_CAPTURE_BEGIN),
            event(V10Phase.STATE_CAPTURE_END),
            "timeline",
            (),
        ),
        (
            "transformation",
            event(V10Phase.STATE_TRANSFORM_BEGIN),
            event(V10Phase.STATE_TRANSFORM_END),
            "timeline",
            (),
        ),
        ("d2h", event(V10Phase.D2H_BEGIN), event(V10Phase.D2H_END), "timeline", ()),
        (
            "validation_publish",
            event(V10Phase.INTEGRITY_BEGIN),
            event(V10Phase.STATE_PUBLISH),
            "timeline",
            (),
        ),
        (
            "gpu1_hbm_release",
            event(V10Phase.GPU1_STATE_RELEASE_BEGIN),
            event(V10Phase.GPU1_HBM_RECLAIM_CONFIRMED),
            "timeline",
            (),
        ),
        (
            "serving_recovery",
            event(V10Phase.GPU1_SERVING_ENABLE),
            event(V10Phase.SERVING_SLO_RESTORED),
            "timeline",
            (),
        ),
        (
            "slo_stability_window",
            event(V10Phase.SERVING_SLO_STABILITY_BEGIN),
            event(V10Phase.SERVING_SLO_STABILITY_END),
            "timeline",
            (),
        ),
        (
            "restore_preparation",
            event(V10Phase.RESTORE_TRIGGER),
            event(V10Phase.H2D_BEGIN),
            "timeline",
            (),
        ),
        ("h2d", event(V10Phase.H2D_BEGIN), event(V10Phase.H2D_END), "timeline", ()),
        (
            "state_import",
            event(V10Phase.STATE_IMPORT_BEGIN),
            event(V10Phase.STATE_IMPORT_END),
            "timeline",
            (),
        ),
        (
            "destination_write",
            event(V10Phase.DESTINATION_NATIVE_WRITE_BEGIN),
            event(V10Phase.DESTINATION_NATIVE_WRITE_END),
            "timeline",
            (),
        ),
        (
            "state_validation",
            event(V10Phase.STATE_VALIDATE_BEGIN),
            event(V10Phase.STATE_VALIDATE_END),
            "timeline",
            (),
        ),
        (
            "branch_continuation",
            event(V10Phase.BRANCH_RESUME_BEGIN),
            event(V10Phase.ROLLOUT_CONTINUATION_COMPLETE),
            "timeline",
            (),
        ),
        (
            "cleanup",
            event(V10Phase.ROLLOUT_CONTINUATION_COMPLETE),
            allocation_end_ns,
            "function-completion.json#gpu_allocation_seconds",
            (),
        ),
    )
    if tuple(row[0] for row in measured) != PHASE_ORDER[1:]:
        raise ValueError("v10 measured phase ledger differs from phase-budget order")

    conservative_invocation_wall_seconds = invocation_gpu_seconds / 2.0
    conservative_overhead_wall_seconds = max(
        0.0, conservative_invocation_wall_seconds - actual_wall_seconds
    )
    conservative_overhead_gpu_seconds = 2.0 * conservative_overhead_wall_seconds
    if not math.isclose(
        conservative_gpu_seconds_before
        + conservative_overhead_gpu_seconds
        + float(actual_gpu_seconds),
        conservative_gpu_seconds_after,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError("phase ledger cannot reconcile actual and conservative invocation time")
    conservative_price = float(settlement["invocation_conservative_cost_usd"]) / (
        invocation_gpu_seconds / 3_600.0
    )
    actual_price = float(settlement["actual_gpu_cost_usd"]) / (float(actual_gpu_seconds) / 3_600.0)
    if not math.isclose(conservative_price, actual_price, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("phase ledger actual and conservative GPU prices differ")

    phases = [
        {
            "sequence": 0,
            "phase": "modal_function_startup",
            "clock_domain": "local-client-envelope",
            "before_monotonic_ns": None,
            "after_monotonic_ns": None,
            "actual_wall_seconds": None,
            "actual_A100_seconds": None,
            "actual_measurement_status": "not-separately-observable-from-provider-startup",
            "conservative_wall_seconds": conservative_overhead_wall_seconds,
            "conservative_A100_seconds": conservative_overhead_gpu_seconds,
            "conservative_A100_seconds_before": conservative_gpu_seconds_before,
            "conservative_A100_seconds_after": (
                conservative_gpu_seconds_before + conservative_overhead_gpu_seconds
            ),
            "remaining_hard_budget_A100_seconds_before": (
                hard_ceiling_gpu_seconds - conservative_gpu_seconds_before
            ),
            "remaining_hard_budget_A100_seconds_after": (
                hard_ceiling_gpu_seconds
                - conservative_gpu_seconds_before
                - conservative_overhead_gpu_seconds
            ),
            "actual_cost_usd": None,
            "conservative_cost_usd": (
                conservative_overhead_gpu_seconds / 3_600.0 * conservative_price
            ),
            "selected_upper_bound_seconds": selected_bounds["modal_function_startup"],
            "source_evidence": (
                "analysis/budget-settlement.json#client_elapsed_seconds-minus-"
                "function-completion.gpu_allocation_seconds"
            ),
            "gpu_count": 2,
            "subphases": [],
        }
    ]
    for sequence, (name, start_ns, end_ns, source, subphases) in enumerate(measured, start=1):
        start = int(start_ns)
        end = int(end_ns)
        if end < start or start < allocation_start_ns or end > allocation_end_ns:
            raise ValueError(f"v10 GPU-active phase timestamps are outside allocation: {name}")
        wall_seconds = (end - start) / 1e9
        actual_before = 2.0 * (start - allocation_start_ns) / 1e9
        actual_after = 2.0 * (end - allocation_start_ns) / 1e9
        conservative_before = (
            conservative_gpu_seconds_before + conservative_overhead_gpu_seconds + actual_before
        )
        conservative_after = (
            conservative_gpu_seconds_before + conservative_overhead_gpu_seconds + actual_after
        )
        phases.append(
            {
                "sequence": sequence,
                "phase": name,
                "clock_domain": "remote-function-monotonic",
                "before_monotonic_ns": start,
                "after_monotonic_ns": end,
                "actual_wall_seconds": wall_seconds,
                "actual_A100_seconds": 2.0 * wall_seconds,
                "actual_measurement_status": "measured-from-hash-bound-live-timestamps",
                "actual_invocation_A100_seconds_before": actual_before,
                "actual_invocation_A100_seconds_after": actual_after,
                "conservative_A100_seconds_before": conservative_before,
                "conservative_A100_seconds_after": conservative_after,
                "remaining_hard_budget_A100_seconds_before": (
                    hard_ceiling_gpu_seconds - conservative_before
                ),
                "remaining_hard_budget_A100_seconds_after": (
                    hard_ceiling_gpu_seconds - conservative_after
                ),
                "actual_cost_usd": 2.0 * wall_seconds / 3_600.0 * actual_price,
                "conservative_cost_usd": (2.0 * wall_seconds / 3_600.0 * conservative_price),
                "selected_upper_bound_seconds": selected_bounds[name],
                "within_selected_phase_bound": (wall_seconds <= selected_bounds[name] + 1e-9),
                "gpu_count": 2,
                "source_evidence": source,
                "subphases": list(subphases),
            }
        )
    if tuple(row["phase"] for row in phases) != PHASE_ORDER:
        raise ValueError("v10 phase ledger does not contain the exact budget phase set")
    payload = {
        "schema_version": "sloforge.branchfabric.experiment-004-v10-gpu-phase-ledger/v1",
        "attempt_id": settlement["attempt_id"],
        "allocation_semantics": "both A100s remain allocated throughout",
        "phase_accounting_semantics": (
            "measured remote phase intervals are non-additive when serving/state work "
            "overlaps; provider startup is conservative-only and never presented as "
            "measured GPU time"
        ),
        "phases": phases,
        "actual_invocation_gpu_seconds": float(actual_gpu_seconds),
        "actual_invocation_gpu_hours": float(actual_gpu_seconds) / 3_600.0,
        "actual_invocation_gpu_cost_usd": settlement["actual_gpu_cost_usd"],
        "conservative_invocation_gpu_seconds": invocation_gpu_seconds,
        "conservative_invocation_gpu_hours": invocation_gpu_seconds / 3_600.0,
        "conservative_gpu_seconds_before": conservative_gpu_seconds_before,
        "conservative_gpu_seconds_after": conservative_gpu_seconds_after,
        "remaining_hard_budget_gpu_seconds": (
            hard_ceiling_gpu_seconds - conservative_gpu_seconds_after
        ),
        "invocation_conservative_cost_usd": settlement["invocation_conservative_cost_usd"],
        "cumulative_conservative_gpu_cost_usd": settlement["cumulative_conservative_gpu_cost_usd"],
    }
    path = run_root / "analysis/gpu-phase-ledger.json"
    _write_new(path, payload)
    return path


def _validate_manifest_member(*, manifest_path: Path, root: Path, member: Path) -> None:
    manifest = _load_object(manifest_path)
    if manifest.get(
        "schema_version"
    ) != "sloforge.branchfabric.capacity-remote-manifest/v1" or not isinstance(
        manifest.get("artifacts"), list
    ):
        raise ValueError("capacity manifest identity is invalid during v10 finalization")
    relative = _regular_file(member).relative_to(root.resolve(strict=True)).as_posix()
    matches = tuple(
        row
        for row in manifest["artifacts"]
        if isinstance(row, dict) and row.get("relative_path") == relative
    )
    if len(matches) != 1 or set(matches[0]) != {"relative_path", "bytes", "sha256"}:
        raise ValueError(f"capacity manifest does not bind prerequisite artifact: {relative}")
    if matches[0]["bytes"] != member.stat().st_size or matches[0]["sha256"] != _sha256(member):
        raise ValueError(f"capacity prerequisite differs from its immutable manifest: {relative}")


def _validate_calibration_binding(config: dict[str, Any]) -> None:
    """Reuse the sole launcher's full raw-probe and independent-review audit."""

    launcher_path = _ROOT / "tools/branchfabric-experiment-004-launch.py"
    specification = importlib.util.spec_from_file_location(
        "sloforge_exp004_v10_finalizer_launcher_audit", launcher_path
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load the v10 launcher's calibration validator")
    launcher = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(launcher)
    launcher._validate_v10_calibration_binding(config)


def _integrated_manifest_entries(*, run_root: Path, manifest: dict[str, Any]) -> dict[str, Path]:
    rows = manifest.get("artifacts")
    if not isinstance(rows, list) or not rows:
        raise ValueError("integrated run manifest has no artifact inventory")
    entries: dict[str, Path] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"relative_path", "sha256"}:
            raise ValueError("integrated run manifest artifact row differs from its contract")
        relative = Path(str(row["relative_path"]))
        key = relative.as_posix()
        digest = row["sha256"]
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or key in entries
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("integrated run manifest contains an unsafe artifact identity")
        path = _regular_file(run_root / relative)
        if not path.is_relative_to(run_root) or _sha256(path) != digest:
            raise ValueError(f"integrated run artifact differs from its manifest: {key}")
        entries[key] = path
    return entries


def _integrated_probe_result(
    *,
    run_root: Path,
    entries: dict[str, Path],
    topology: str,
    probe_id: str,
) -> Any:
    from sloforge.helix.characterization.gpu_capacity_calibration import (
        CapacityProbePlan,
        CapacityProbeRaw,
        CapacityProbeResult,
        evaluate_probe,
    )

    prefix = f"calibration/{topology}/{probe_id}"
    required = {
        f"{prefix}/plan.json",
        f"{prefix}/raw.json",
        f"{prefix}/result.json",
        f"{prefix}/request-state-reset.json",
        f"{prefix}/worker-shards.json",
    }
    if not required.issubset(entries):
        raise ValueError(f"integrated manifest omits raw selected probe evidence: {probe_id}")
    plan = CapacityProbePlan.model_validate_json(
        entries[f"{prefix}/plan.json"].read_text(), strict=True
    )
    raw = CapacityProbeRaw.model_validate_json(
        entries[f"{prefix}/raw.json"].read_text(), strict=True
    )
    result = CapacityProbeResult.model_validate_json(
        entries[f"{prefix}/result.json"].read_text(), strict=True
    )
    if raw.plan != plan or evaluate_probe(raw) != result:
        raise ValueError(f"integrated selected probe differs from raw recomputation: {probe_id}")
    return result


def _build_integrated_prerequisite_evidence(
    *,
    base_config: dict[str, Any],
    base_config_path: Path,
    run_root: Path,
    budget: Any,
    cleanup: Any,
    settlement_path: Path,
    cleanup_path: Path,
) -> tuple[dict[str, Any], Any]:
    """Validate preauthorized guards and transaction from one retained allocation."""

    from gpu_capacity_calibration_controller import (
        _validate_readiness_payloads,
        _validate_reset_payloads,
    )
    from gpu_reclamation_controller import (
        _assess_sanity_guard,
        _validate_measured_transaction_compilation_payloads,
    )
    from gpu_reclamation_worker import _validate_integrated_transaction_command

    from sloforge.helix.characterization.gpu_capacity_calibration import CapacityProbeResult
    from sloforge.helix.characterization.gpu_reclamation_v10_authorization import (
        verify_authorization_file,
    )
    from sloforge.helix.characterization.gpu_reclamation_v10_budget import (
        load_v10_phase_budget,
        phase_budget_sha256,
    )
    from sloforge.helix.characterization.gpu_reclamation_v10_validity import V10PrerequisiteEvidence

    manifest_path = _regular_file(run_root / "integrated-run-manifest.json")
    manifest = _load_object(manifest_path)
    required_manifest_fields = {
        "schema_version",
        "attempt_id",
        "base_config_sha256",
        "selected_load_sha256",
        "authorization_artifact_hash",
        "authorization_verified_before_workers",
        "live_review_exchange_performed",
        "sanity_result_sha256",
        "worker_identity_by_device",
        "engine_reload_count",
        "single_worker_pair",
        "sanity_and_transaction_same_allocation",
        "broad_capacity_calibration_performed",
        "measured_transaction_compilation_free",
        "measured_transaction_compilation_by_role",
        "artifacts",
    }
    if set(manifest) != required_manifest_fields or any(
        (
            manifest.get("schema_version") != "sloforge.branchfabric.integrated-run-manifest/v2",
            manifest.get("attempt_id") != base_config.get("attempt_id"),
            manifest.get("base_config_sha256") != _sha256(base_config_path),
            manifest.get("engine_reload_count") != 0,
            manifest.get("measured_transaction_compilation_free") is not True,
            manifest.get("single_worker_pair") is not True,
            manifest.get("sanity_and_transaction_same_allocation") is not True,
            manifest.get("broad_capacity_calibration_performed") is not False,
            manifest.get("live_review_exchange_performed") is not False,
            manifest.get("authorization_verified_before_workers") is not True,
        )
    ):
        raise ValueError(
            "integrated v10 manifest does not prove the preauthorized retained allocation"
        )
    entries = _integrated_manifest_entries(run_root=run_root, manifest=manifest)
    required = {
        "base-config.json",
        "effective-config.json",
        "authorization/verified.json",
        "readiness/both-engines-ready.json",
        "sanity/sanity-result.json",
        "sanity/12-rps/result.json",
        "sanity/12-rps/assessment.json",
        "sanity/15-rps/result.json",
        "sanity/15-rps/assessment.json",
        "calibration/selected-load.json",
        "calibration/final-request-state-reset.json",
        "barriers/transaction.command.json",
        "barriers/serving.transaction-ready.json",
        "barriers/rollout.transaction-ready.json",
        "serving/result.json",
        "rollout/result.json",
    }
    if not required.issubset(entries):
        raise ValueError("integrated v10 manifest omits authorization, sanity, or handoff evidence")
    if entries["base-config.json"].read_bytes() != _canonical_bytes(base_config):
        raise ValueError("integrated copied config differs from the launched config")

    authorization_path = _repo_artifact(base_config.get("authorization_artifact"))
    authorization = verify_authorization_file(
        authorization_path,
        repository_root=_ROOT,
        expected_artifact_hash=str(base_config.get("authorization_artifact_hash")),
    )
    phase_budget_path = _repo_artifact(base_config.get("phase_budget_artifact"))
    phase_budget = load_v10_phase_budget(phase_budget_path, repository_root=_ROOT)
    if phase_budget_sha256(phase_budget_path) != base_config.get("phase_budget_sha256"):
        raise ValueError("phase budget differs from launched hash")
    remote_authorization = _load_object(entries["authorization/verified.json"])
    if (
        remote_authorization.get("status") != "APPROVED"
        or remote_authorization.get("artifact_hash") != authorization["artifact_hash"]
        or remote_authorization.get("source_evidence_hashes_verified") is not True
        or remote_authorization.get("remote_review_exchange_permitted") is not False
        or manifest.get("authorization_artifact_hash") != authorization["artifact_hash"]
    ):
        raise ValueError("remote controller did not verify the sealed authorization")

    controller = _load_object(run_root / "controller-result.json")
    inventory = controller.get("inventory_before")
    if (
        controller.get("status") != "succeeded"
        or controller.get("sanity_status") != "passed"
        or controller.get("sanity_guard_count") != 2
        or controller.get("controller_error") is not None
        or controller.get("cleanup_error") is not None
        or not isinstance(inventory, list)
        or len(inventory) != 2
        or len({str(row.get("uuid")) for row in inventory}) != 2
        or any("A100" not in str(row.get("name")) for row in inventory)
    ):
        raise ValueError("integrated v10 controller/readiness topology is invalid")
    readiness_path = entries["readiness/both-engines-ready.json"]
    readiness = _load_object(readiness_path)
    engines = readiness.get("engine_evidence")
    if (
        readiness.get("BOTH_ENGINES_READY") is not True
        or not isinstance(engines, list)
        or len(engines) != 2
    ):
        raise ValueError("integrated two-engine readiness is invalid")
    _validate_readiness_payloads(
        payloads=(engines[0], engines[1]), expected_inventory=(inventory[0], inventory[1])
    )
    compilation_by_role = _validate_measured_transaction_compilation_payloads(
        (
            _load_object(entries["serving/result.json"]),
            _load_object(entries["rollout/result.json"]),
        )
    )
    if (
        controller.get("measured_transaction_compilation_free") is not True
        or controller.get("measured_transaction_compilation_by_role") != compilation_by_role
        or manifest.get("measured_transaction_compilation_by_role") != compilation_by_role
    ):
        raise ValueError("integrated v10 no-compilation evidence differs across artifacts")

    assessments = []
    for rate in (12, 15):
        result_path = entries[f"sanity/{rate}-rps/result.json"]
        result = CapacityProbeResult.model_validate_json(result_path.read_text(), strict=True)
        assessment = _assess_sanity_guard(result, expected_rate_rps=float(rate))
        if (
            assessment["passed"] is not True
            or _load_object(entries[f"sanity/{rate}-rps/assessment.json"]) != assessment
        ):
            raise ValueError(f"{rate}-rps sanity assessment failed recomputation")
        assessments.append((result_path, assessment))
    sanity_result_path = entries["sanity/sanity-result.json"]
    sanity_result = _load_object(sanity_result_path)
    if (
        sanity_result.get("status") != "PASSED"
        or sanity_result.get("guard_count") != 2
        or sanity_result.get("adaptive_capacity_search_performed") is not False
        or sanity_result.get("two_gpu_capacity_probe_performed") is not False
        or manifest.get("sanity_result_sha256") != _sha256(sanity_result_path)
    ):
        raise ValueError("v10 sanity aggregate does not prove the exact two guards")

    resets = json.loads(entries["calibration/final-request-state-reset.json"].read_text())
    if not isinstance(resets, list):
        raise ValueError("final request reset is not a two-worker array")
    final_resets = _validate_reset_payloads(
        payloads=tuple(resets), expected_probe_id="final-calibration-reset"
    )
    continuity = manifest.get("worker_identity_by_device")
    if not isinstance(continuity, dict) or set(continuity) != {"gpu0", "gpu1"}:
        raise ValueError("integrated v10 manifest lacks exact engine identities")
    for payload in (*final_resets, *engines):
        identity = continuity.get(str(payload.get("device")))
        if not isinstance(identity, dict) or any(
            payload.get(key) != identity.get(key)
            for key in ("pid", "engine_nonce", "physical_gpu_uuid")
        ):
            raise ValueError("engine identity changed across readiness, guards, and transaction")

    selection_path = entries["calibration/selected-load.json"]
    if _sha256(selection_path) != manifest.get("selected_load_sha256") or _sha256(
        selection_path
    ) != base_config.get("calibration_selected_load_sha256"):
        raise ValueError("selected load differs from authorization")
    transaction = _load_object(entries["barriers/transaction.command.json"])
    effective_config, hashes = _validate_integrated_transaction_command(
        base_config=base_config, command=transaction, selected_load_sha256=_sha256(selection_path)
    )
    if (
        hashes
        != {
            "authorization_artifact_hash": authorization["artifact_hash"],
            "sanity_result_sha256": _sha256(sanity_result_path),
        }
        or _load_object(entries["effective-config.json"]) != effective_config
    ):
        raise ValueError("transaction command differs from pre-GPU evidence")
    if not budget.passed or not cleanup.passed:
        raise ValueError("v10 prerequisite construction requires passing budget and cleanup")
    evidence = V10PrerequisiteEvidence(
        authorization_valid=True,
        phase_budget_valid=phase_budget.status == "EVIDENCE_DERIVED",
        two_engine_ready=True,
        sanity_12rps_stable=bool(assessments[0][1]["passed"]),
        sanity_15rps_overload=bool(assessments[1][1]["passed"]),
        budget_pass=True,
        cleanup_pass=True,
        evidence_references=(
            _binding("authorization", authorization_path),
            _binding("phase-budget", phase_budget_path),
            _binding("engine-readiness", readiness_path),
            _binding("sanity-12rps", assessments[0][0]),
            _binding("sanity-15rps", assessments[1][0]),
            _binding("budget-settlement", settlement_path),
            _binding("modal-cleanup", cleanup_path),
        ),
    )
    return effective_config, evidence


def _build_prerequisite_evidence(
    *,
    config: dict[str, Any],
    budget: Any,
    cleanup: Any,
    settlement_path: Path,
    cleanup_path: Path,
    run_root: Path | None = None,
    config_path: Path | None = None,
) -> Any:
    """Bind every pre/postflight gate to immutable, content-addressed evidence."""

    if run_root is not None and (run_root / "integrated-run-manifest.json").is_file():
        if config_path is None:
            raise ValueError("integrated v10 prerequisite binding requires the base config path")
        return _build_integrated_prerequisite_evidence(
            base_config=config,
            base_config_path=config_path,
            run_root=run_root,
            budget=budget,
            cleanup=cleanup,
            settlement_path=settlement_path,
            cleanup_path=cleanup_path,
        )
    raise ValueError(
        "v10 finalization requires the preauthorized integrated-run-manifest/v2; "
        "legacy live-review calibration runs are diagnostic only"
    )


def main() -> int:
    from sloforge.helix.characterization.gpu_reclamation_methodology import (
        Experiment004GpuHourLedger,
    )
    from sloforge.helix.characterization.gpu_reclamation_v10_postprocess import (
        assess_v10_directory,
    )
    from sloforge.helix.characterization.gpu_reclamation_v10_validity import (
        BudgetEvidence,
        CleanupEvidence,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--gpu-hours-path", required=True, type=Path)
    parser.add_argument("--modal-cleanup-evidence", required=True, type=Path)
    args = parser.parse_args()
    config_path = args.config_path.resolve(strict=True)
    config = _load_object(config_path)
    run_root = args.run_root.resolve(strict=True)
    controller = _load_object(run_root / "controller-result.json")
    ledger = Experiment004GpuHourLedger.model_validate_json(
        args.gpu_hours_path.resolve(strict=True).read_text(), strict=True
    )
    settlement_path = run_root / "analysis/budget-settlement.json"
    settlement = _load_object(settlement_path)
    ledger_digest = hashlib.sha256(
        args.gpu_hours_path.resolve(strict=True).read_bytes()
    ).hexdigest()
    if (
        settlement.get("attempt_id") != config["attempt_id"]
        or settlement.get("gpu_hours_sha256") != ledger_digest
    ):
        raise ValueError("budget settlement is not bound to the current GPU-hour ledger")
    required_settlement_fields = {
        "schema_version",
        "attempt_id",
        "budget",
        "client_elapsed_seconds",
        "actual_gpu_seconds",
        "actual_gpu_hours",
        "actual_gpu_cost_usd",
        "conservative_gpu_seconds",
        "conservative_gpu_hours",
        "invocation_conservative_cost_usd",
        "cumulative_conservative_gpu_seconds",
        "cumulative_conservative_gpu_hours",
        "cumulative_conservative_gpu_cost_usd",
        "remaining_hard_budget_gpu_seconds",
        "gpu_hours_sha256",
    }
    if set(settlement) != required_settlement_fields:
        raise ValueError("budget settlement lacks actual/conservative/cost/remaining accounting")
    matching = tuple(
        interval
        for interval in ledger.intervals
        if interval.invocation_id == str(config["attempt_id"])
    )
    if len(matching) != 1 or ledger.reservations:
        raise ValueError("v10 finalizer requires one settled invocation and zero reservations")
    invocation_gpu_seconds = matching[0].gpu_seconds
    before = ledger.consumed_additional_gpu_seconds - invocation_gpu_seconds
    budget = BudgetEvidence(
        conservative_gpu_seconds_before=before,
        invocation_gpu_seconds=invocation_gpu_seconds,
        conservative_gpu_seconds_after=ledger.consumed_additional_gpu_seconds,
        hard_ceiling_gpu_seconds=ledger.hard_additional_gpu_seconds,
        ledger_updated=True,
    )
    if settlement.get("budget") != budget.model_dump(mode="json"):
        raise ValueError("budget settlement artifact differs from the settled ledger")
    if (
        not math.isclose(
            float(settlement["conservative_gpu_seconds"]),
            invocation_gpu_seconds,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or not math.isclose(
            float(settlement["cumulative_conservative_gpu_seconds"]),
            ledger.consumed_additional_gpu_seconds,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or not math.isclose(
            float(settlement["remaining_hard_budget_gpu_seconds"]),
            ledger.hard_additional_gpu_seconds - ledger.committed_additional_gpu_seconds,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise ValueError("budget settlement totals disagree with the authoritative ledger")
    cleanup_path = args.modal_cleanup_evidence.resolve(strict=True)
    cleanup_payload = _load_object(cleanup_path)
    expected_volumes = {"sloforge-model-cache", "sloforge-branchfabric-results"}
    empty_fields = (
        "active_tasks",
        "running_containers",
        "endpoints",
        "reservations",
        "owned_child_processes",
        "profilers",
    )
    if (
        cleanup_payload.get("schema_version")
        != "sloforge.branchfabric.experiment-004-modal-cleanup-evidence/v1"
        or cleanup_payload.get("attempt_id") != config["attempt_id"]
        or cleanup_payload.get("observed_after_function_call_id") != matching[0].function_call_id
        or not isinstance(cleanup_payload.get("raw_provider_queries"), list)
        or not cleanup_payload["raw_provider_queries"]
        or any(cleanup_payload.get(field) != [] for field in empty_fields)
        or set(cleanup_payload.get("persistent_volumes", ())) != expected_volumes
    ):
        raise ValueError("Modal cleanup evidence is incomplete or does not prove zero resources")
    cleanup = CleanupEvidence(
        zero_active_tasks=True,
        zero_running_containers=True,
        zero_endpoints=True,
        zero_reservations=True,
        zero_owned_child_processes=True,
        zero_profilers=True,
        authorized_persistent_volumes_only=True,
    )
    prerequisite_result = _build_prerequisite_evidence(
        config=config,
        budget=budget,
        cleanup=cleanup,
        settlement_path=settlement_path,
        cleanup_path=cleanup_path,
        run_root=run_root,
        config_path=config_path,
    )
    if isinstance(prerequisite_result, tuple):
        config, prerequisites = prerequisite_result
    else:
        prerequisites = prerequisite_result
    if config.get("serving_methodology") != "v10-global-capacity":
        raise ValueError("v10 finalizer refuses a non-v10 serving methodology")
    scientific, movement, recovery = assess_v10_directory(
        work_root=run_root,
        config=config,
        controller=controller,
        budget=budget,
        cleanup=cleanup,
        prerequisites=prerequisites,
    )
    if scientific.scientifically_valid:
        _write_gpu_phase_ledger(
            run_root=run_root,
            scientific=scientific,
            settlement=settlement,
            invocation_gpu_seconds=invocation_gpu_seconds,
            conservative_gpu_seconds_before=before,
            conservative_gpu_seconds_after=ledger.consumed_additional_gpu_seconds,
            hard_ceiling_gpu_seconds=ledger.hard_additional_gpu_seconds,
        )
    for name, payload in (
        ("movement-accounting.json", movement),
        ("serving-recovery.json", recovery),
    ):
        existing = run_root / "analysis" / name
        if existing.is_file() and existing.read_bytes() != _canonical_bytes(payload):
            raise ValueError(f"local recomputation differs from remote provisional {name}")
        if not existing.is_file():
            _write_new(existing, payload)
    _write_new(run_root / "analysis/scientific-validity.json", scientific)
    print(_canonical_bytes(scientific).decode(), end="")
    return 0 if scientific.scientifically_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
