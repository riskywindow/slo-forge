from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
FINALIZER = ROOT / "tools/branchfabric-experiment-004-finalize-v10.py"


def _load(name: str):
    specification = importlib.util.spec_from_file_location(name, FINALIZER)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prerequisite_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = _load("exp004_v10_finalizer_prerequisites")
    experiment_root = tmp_path / "artifacts/branchfabric/gpu-validation/experiment-004"
    calibration_root = experiment_root / "calibration/raw/modal/capacity-fixture"
    engine_evidence = [{"device": "gpu0"}, {"device": "gpu1"}]
    controller = {
        "schema_version": "sloforge.branchfabric.capacity-controller/v1",
        "status": "succeeded",
        "both_engines_ready": True,
        "calibration_status": "ready",
        "controller_error": None,
        "cleanup_error": None,
        "inventory_before": [{"uuid": "GPU-0"}, {"uuid": "GPU-1"}],
        "worker_ready_evidence": engine_evidence,
        "probe_count": 1,
    }
    readiness = {
        "schema_version": "sloforge.branchfabric.both-engines-ready/v1",
        "BOTH_ENGINES_READY": True,
        "engine_evidence": engine_evidence,
    }
    selection = {"lambda_1_rps": 10.0, "lambda_2_rps": 20.0, "lambda_spike_rps": 12.0}
    calibration_result = {
        "schema_version": "sloforge.branchfabric.capacity-calibration-result/v1",
        "status": "ready",
        "review_required": True,
        "v10_launch_permitted": False,
        "selection": selection,
        "probe_results": [{"probe_id": "fixture"}],
    }
    members = {
        "controller-result.json": controller,
        "readiness/both-engines-ready.json": readiness,
        "calibration/calibration-result.json": calibration_result,
        "calibration/selected-load.json": selection,
    }
    for relative, value in members.items():
        _write(calibration_root / relative, value)
    _write(
        calibration_root / "REMOTE_MANIFEST.json",
        {
            "schema_version": "sloforge.branchfabric.capacity-remote-manifest/v1",
            "artifacts": [
                {
                    "relative_path": relative,
                    "bytes": (calibration_root / relative).stat().st_size,
                    "sha256": _sha(calibration_root / relative),
                }
                for relative in members
            ],
        },
    )
    selected_path = experiment_root / "calibration/selected-load.json"
    review_path = experiment_root / "calibration/reviews/agent9-selected-load-approval.json"
    settlement_path = tmp_path / "run/analysis/budget-settlement.json"
    cleanup_path = tmp_path / "cleanup.json"
    for path, value in (
        (selected_path, selection),
        (review_path, {"verdict": "approved"}),
        (settlement_path, {"budget": "passed"}),
        (cleanup_path, {"cleanup": "passed"}),
    ):
        _write(path, value)
    config = {
        "calibration_attempt_id": "capacity-fixture",
        "calibration_selected_load_artifact": str(selected_path.relative_to(tmp_path)),
        "agent9_approval_artifact": str(review_path.relative_to(tmp_path)),
    }
    readiness_calls: list[object] = []
    monkeypatch.setitem(
        sys.modules,
        "gpu_capacity_calibration_controller",
        SimpleNamespace(
            _validate_readiness_payloads=lambda **kwargs: readiness_calls.append(kwargs)
        ),
    )
    monkeypatch.setattr(module, "_ROOT", tmp_path)
    monkeypatch.setattr(module, "_EXPERIMENT_ROOT", experiment_root)
    monkeypatch.setattr(module, "_validate_calibration_binding", lambda payload: None)
    evidence = module._build_prerequisite_evidence(
        config=config,
        budget=SimpleNamespace(passed=True),
        cleanup=SimpleNamespace(passed=True),
        settlement_path=settlement_path,
        cleanup_path=cleanup_path,
    )
    return module, evidence, readiness_calls, calibration_root


def test_prerequisite_builder_hash_binds_every_required_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load("exp004_v10_finalizer_rejects_legacy_prerequisites")
    with pytest.raises(ValueError, match="preauthorized integrated-run-manifest/v2"):
        module._build_prerequisite_evidence(
            config={"serving_methodology": "v10-global-capacity"},
            budget=SimpleNamespace(passed=True),
            cleanup=SimpleNamespace(passed=True),
            settlement_path=tmp_path / "budget.json",
            cleanup_path=tmp_path / "cleanup.json",
        )


def test_budget_authorization_preserves_and_supersedes_historical_blocker_digest() -> None:
    experiment_root = ROOT / "artifacts/branchfabric/gpu-validation/experiment-004"
    authorization = json.loads((experiment_root / "budget-authorization-v10.json").read_text())
    ledger_path = experiment_root / "gpu-hours.json"
    ledger = json.loads(ledger_path.read_text())
    outcome = json.loads((experiment_root / "calibration/calibration-outcome.json").read_text())
    blocker = json.loads((experiment_root / "calibration/calibration-blocker.json").read_text())

    assert authorization["authorized_gpu_budget_usd"] == 40.0
    assert authorization["authorized_cumulative_a100_hours"] == 3.0
    assert authorization["authorized_cumulative_gpu_seconds"] == 10_800.0
    authorized_interval_count = authorization["ledger_mutation"]["settled_interval_count_after"]
    authorized_intervals = ledger["intervals"][:authorized_interval_count]
    authorized_intervals_digest = hashlib.sha256(
        (json.dumps(authorized_intervals, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    assert authorization["authorized_settled_intervals_sha256"] == authorized_intervals_digest
    if len(ledger["intervals"]) == authorized_interval_count and not ledger["reservations"]:
        assert authorization["authorized_ledger_sha256"] == _sha(ledger_path)
    assert authorization["previous_ledger_sha256"] == outcome["ledger"]["artifact_sha256"]
    assert authorization["previous_ledger_sha256"] == blocker["budget"]["ledger"]["artifact_sha256"]
    assert authorization["ledger_mutation"]["settled_intervals_changed"] is False


def test_prerequisite_builder_rejects_readiness_changed_after_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load("exp004_v10_finalizer_manifest_tamper")
    run_root = tmp_path / "run"
    member = run_root / "base-config.json"
    _write(member, {"attempt_id": "fixture"})
    _write(
        run_root / "integrated-run-manifest.json",
        {
            "schema_version": "sloforge.branchfabric.integrated-run-manifest/v2",
            "artifacts": [{"relative_path": "base-config.json", "sha256": _sha(member)}],
        },
    )
    member.write_text(member.read_text() + " ")
    manifest = json.loads((run_root / "integrated-run-manifest.json").read_text())
    with pytest.raises(ValueError, match="differs from its manifest"):
        module._integrated_manifest_entries(run_root=run_root, manifest=manifest)


def test_integrated_prerequisite_builder_binds_same_run_manifest_and_engine_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load("exp004_v10_finalizer_integrated_prerequisites")
    run_root = tmp_path / "run"
    config_path = tmp_path / "base-config.json"
    attempt_id = "exp004-integrated-fixture"
    authorization_path = tmp_path / "authorization.json"
    phase_budget_path = tmp_path / "phase-budget.json"
    _write(authorization_path, {"status": "APPROVED"})
    _write(phase_budget_path, {"status": "EVIDENCE_DERIVED"})
    base_config = {
        "attempt_id": attempt_id,
        "execution_mode": "integrated-calibration-v10",
        "authorization_artifact": "authorization.json",
        "authorization_artifact_hash": "a" * 64,
        "phase_budget_artifact": "phase-budget.json",
        "phase_budget_sha256": _sha(phase_budget_path),
    }
    _write(config_path, base_config)
    selection = {
        "schema_version": "sloforge.branchfabric.capacity-selection/v1",
        "lambda_1_rps": 12.0,
        "lambda_2_rps": 20.0,
        "lambda_spike_rps": 15.0,
    }
    base_config["calibration_selected_load_sha256"] = "pending"
    identities = {
        "gpu0": {
            "pid": 101,
            "engine_nonce": "a" * 64,
            "physical_gpu_uuid": "GPU-fixture-0",
        },
        "gpu1": {
            "pid": 102,
            "engine_nonce": "b" * 64,
            "physical_gpu_uuid": "GPU-fixture-1",
        },
    }
    engines = [{"device": device, **identity} for device, identity in identities.items()]
    compilation_by_role = {
        role: {
            "schema_version": (
                "sloforge.branchfabric.measured-transaction-compilation-observation/v1"
            ),
            "source": "bounded-python-logging-handler",
            "role": role,
            "interval_start_ns": 10,
            "interval_end_ns": 20,
            "capture_buffer_valid": True,
            "events": [],
            "no_deferred_compilation_event": True,
            "passed": True,
        }
        for role in ("serving", "rollout")
    }
    effective = dict(base_config)
    _write(
        run_root / "controller-result.json",
        {
            "schema_version": "sloforge.branchfabric.experiment-004-integrated-controller/v1",
            "status": "succeeded",
            "sanity_status": "passed",
            "sanity_guard_count": 2,
            "controller_error": None,
            "cleanup_error": None,
            "inventory_before": [
                {"uuid": "GPU-fixture-0", "name": "NVIDIA A100 80GB PCIe"},
                {"uuid": "GPU-fixture-1", "name": "NVIDIA A100 80GB PCIe"},
            ],
            "measured_transaction_compilation_free": True,
            "measured_transaction_compilation_by_role": compilation_by_role,
        },
    )
    artifacts: dict[str, object] = {
        "base-config.json": base_config,
        "effective-config.json": effective,
        "authorization/verified.json": {
            "status": "APPROVED",
            "artifact_hash": "a" * 64,
            "source_evidence_hashes_verified": True,
            "remote_review_exchange_permitted": False,
        },
        "readiness/both-engines-ready.json": {
            "schema_version": "sloforge.branchfabric.both-engines-ready/v1",
            "BOTH_ENGINES_READY": True,
            "attempt_id": attempt_id,
            "engine_evidence": engines,
        },
        "sanity/12-rps/result.json": {"rate": 12},
        "sanity/12-rps/assessment.json": {"passed": True, "rate": 12},
        "sanity/15-rps/result.json": {"rate": 15},
        "sanity/15-rps/assessment.json": {"passed": True, "rate": 15},
        "sanity/sanity-result.json": {
            "status": "PASSED",
            "guard_count": 2,
            "adaptive_capacity_search_performed": False,
            "two_gpu_capacity_probe_performed": False,
        },
        "calibration/selected-load.json": selection,
        "calibration/final-request-state-reset.json": engines,
        "barriers/transaction.command.json": {"fixture": "transaction"},
        "barriers/serving.transaction-ready.json": {
            "device": "gpu0",
            "passed": True,
            **identities["gpu0"],
        },
        "barriers/rollout.transaction-ready.json": {
            "device": "gpu1",
            "passed": True,
            **identities["gpu1"],
        },
        "serving/result.json": {
            "status": "succeeded",
            "role": "serving",
            "measured_transaction_compilation_observation": compilation_by_role["serving"],
        },
        "rollout/result.json": {
            "status": "succeeded",
            "role": "rollout",
            "measured_transaction_compilation_observation": compilation_by_role["rollout"],
        },
    }
    for relative, payload in artifacts.items():
        _write(run_root / relative, payload)
    selected_sha = _sha(run_root / "calibration/selected-load.json")
    base_config["calibration_selected_load_sha256"] = selected_sha
    effective["calibration_selected_load_sha256"] = selected_sha
    _write(config_path, base_config)
    _write(run_root / "base-config.json", base_config)
    _write(run_root / "effective-config.json", effective)
    sanity_sha = _sha(run_root / "sanity/sanity-result.json")
    command_hashes = {
        "authorization_artifact_hash": "a" * 64,
        "sanity_result_sha256": sanity_sha,
    }
    _write(
        run_root / "integrated-run-manifest.json",
        {
            "schema_version": "sloforge.branchfabric.integrated-run-manifest/v2",
            "attempt_id": attempt_id,
            "base_config_sha256": _sha(config_path),
            "selected_load_sha256": selected_sha,
            "authorization_artifact_hash": "a" * 64,
            "authorization_verified_before_workers": True,
            "live_review_exchange_performed": False,
            "sanity_result_sha256": sanity_sha,
            "worker_identity_by_device": identities,
            "engine_reload_count": 0,
            "single_worker_pair": True,
            "sanity_and_transaction_same_allocation": True,
            "broad_capacity_calibration_performed": False,
            "measured_transaction_compilation_free": True,
            "measured_transaction_compilation_by_role": compilation_by_role,
            "artifacts": [
                {"relative_path": relative, "sha256": _sha(run_root / relative)}
                for relative in artifacts
            ],
        },
    )
    monkeypatch.setitem(
        sys.modules,
        "gpu_capacity_calibration_controller",
        SimpleNamespace(
            _validate_readiness_payloads=lambda **kwargs: kwargs["payloads"],
            _validate_reset_payloads=lambda **kwargs: kwargs["payloads"],
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "gpu_reclamation_controller",
        SimpleNamespace(
            _assess_sanity_guard=lambda result, expected_rate_rps: {
                "passed": True,
                "rate": int(expected_rate_rps),
            },
            _validate_measured_transaction_compilation_payloads=lambda payloads: {
                str(payload["role"]): payload["measured_transaction_compilation_observation"]
                for payload in payloads
            },
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "gpu_reclamation_worker",
        SimpleNamespace(
            _validate_integrated_transaction_command=lambda **kwargs: (
                effective,
                command_hashes,
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "sloforge.helix.characterization.gpu_capacity_calibration",
        SimpleNamespace(
            CapacityProbeResult=SimpleNamespace(
                model_validate_json=lambda raw, strict: json.loads(raw)
            )
        ),
    )
    import sloforge.helix.characterization.gpu_reclamation_v10_authorization as authorization
    import sloforge.helix.characterization.gpu_reclamation_v10_budget as phase_budget

    monkeypatch.setattr(
        authorization,
        "verify_authorization_file",
        lambda *args, **kwargs: {"artifact_hash": "a" * 64},
    )
    monkeypatch.setattr(
        phase_budget,
        "load_v10_phase_budget",
        lambda *args, **kwargs: SimpleNamespace(status="EVIDENCE_DERIVED"),
    )
    monkeypatch.setattr(phase_budget, "phase_budget_sha256", lambda path: _sha(path))
    monkeypatch.setattr(module, "_ROOT", tmp_path)
    settlement_path = tmp_path / "budget-settlement.json"
    cleanup_path = tmp_path / "cleanup.json"
    _write(settlement_path, {"passed": True})
    _write(cleanup_path, {"passed": True})

    observed_config, evidence = module._build_integrated_prerequisite_evidence(
        base_config=base_config,
        base_config_path=config_path,
        run_root=run_root,
        budget=SimpleNamespace(passed=True),
        cleanup=SimpleNamespace(passed=True),
        settlement_path=settlement_path,
        cleanup_path=cleanup_path,
    )

    assert observed_config == effective
    assert evidence.authorization_valid
    assert evidence.phase_budget_valid
    assert evidence.two_engine_ready
    assert evidence.sanity_12rps_stable and evidence.sanity_15rps_overload
    assert {item.split(":", 1)[0] for item in evidence.evidence_references} == {
        "authorization",
        "phase-budget",
        "engine-readiness",
        "sanity-12rps",
        "sanity-15rps",
        "budget-settlement",
        "modal-cleanup",
    }


def test_finalizer_passes_constructed_prerequisites_to_canonical_assessor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sloforge.helix.characterization.gpu_reclamation_v10_validity import (
        V10PrerequisiteEvidence,
    )

    module = _load("exp004_v10_finalizer_end_to_end")
    attempt_id = "exp004-v10-fixture"
    config_path = tmp_path / "config.json"
    run_root = tmp_path / "run"
    ledger_path = tmp_path / "gpu-hours.json"
    cleanup_path = tmp_path / "cleanup.json"
    _write(config_path, {"attempt_id": attempt_id, "serving_methodology": "v10-global-capacity"})
    _write(run_root / "controller-result.json", {"status": "succeeded"})
    raw_manifest = tmp_path / "raw-manifest.json"
    _write(raw_manifest, {"fixture": True})
    ledger = {
        "schema_version": "sloforge.branchfabric.experiment-004-gpu-hours/v1",
        "historical_through_experiment_003_gpu_hours": 1.669146,
        "target_additional_gpu_seconds": 3600.0,
        "hard_additional_gpu_seconds": 10800.0,
        "consumed_additional_gpu_seconds": 20.0,
        "intervals": [
            {
                "invocation_id": attempt_id,
                "function_call_id": "fc-fixture",
                "requested_gpu": "A100-80GB",
                "gpu_count": 2,
                "accounted_wall_seconds": 10.0,
                "actual_gpu_models": ["NVIDIA A100 80GB PCIe", "NVIDIA A100 80GB PCIe"],
                "gpu_uuids": ["GPU-fixture-0", "GPU-fixture-1"],
                "gpu_price_per_hour_usd": 2.4984,
                "raw_manifest": {
                    "artifact_reference": str(raw_manifest),
                    "artifact_sha256": _sha(raw_manifest),
                    "sample_selector": "$",
                },
            }
        ],
        "reservations": [],
    }
    _write(ledger_path, ledger)
    settlement = {
        "schema_version": "sloforge.branchfabric.experiment-004-v10-budget-settlement/v1",
        "attempt_id": attempt_id,
        "budget": {
            "conservative_gpu_seconds_before": 0.0,
            "invocation_gpu_seconds": 20.0,
            "conservative_gpu_seconds_after": 20.0,
            "hard_ceiling_gpu_seconds": 10800.0,
            "ledger_updated": True,
        },
        "client_elapsed_seconds": 10.0,
        "actual_gpu_seconds": 20.0,
        "actual_gpu_hours": 20.0 / 3600.0,
        "actual_gpu_cost_usd": 20.0 / 3600.0 * 2.4984,
        "conservative_gpu_seconds": 20.0,
        "conservative_gpu_hours": 20.0 / 3600.0,
        "invocation_conservative_cost_usd": 20.0 / 3600.0 * 2.4984,
        "cumulative_conservative_gpu_seconds": 20.0,
        "cumulative_conservative_gpu_hours": 20.0 / 3600.0,
        "cumulative_conservative_gpu_cost_usd": 20.0 / 3600.0 * 2.4984,
        "remaining_hard_budget_gpu_seconds": 10780.0,
        "gpu_hours_sha256": _sha(ledger_path),
    }
    _write(run_root / "analysis/budget-settlement.json", settlement)
    _write(
        cleanup_path,
        {
            "schema_version": "sloforge.branchfabric.experiment-004-modal-cleanup-evidence/v1",
            "attempt_id": attempt_id,
            "observed_after_function_call_id": "fc-fixture",
            "raw_provider_queries": [{"query": "fixture"}],
            "active_tasks": [],
            "running_containers": [],
            "endpoints": [],
            "reservations": [],
            "owned_child_processes": [],
            "profilers": [],
            "persistent_volumes": ["sloforge-model-cache", "sloforge-branchfabric-results"],
        },
    )
    kinds = (
        "authorization",
        "phase-budget",
        "engine-readiness",
        "sanity-12rps",
        "sanity-15rps",
        "budget-settlement",
        "modal-cleanup",
    )
    prerequisites = V10PrerequisiteEvidence(
        authorization_valid=True,
        phase_budget_valid=True,
        two_engine_ready=True,
        sanity_12rps_stable=True,
        sanity_15rps_overload=True,
        budget_pass=True,
        cleanup_pass=True,
        evidence_references=tuple(
            f"{kind}:{index:064x}:fixture/{kind}.json" for index, kind in enumerate(kinds, start=1)
        ),
    )
    monkeypatch.setattr(module, "_build_prerequisite_evidence", lambda **kwargs: prerequisites)
    monkeypatch.setattr(
        module,
        "_write_gpu_phase_ledger",
        lambda **kwargs: run_root / "analysis/gpu-phase-ledger.json",
    )
    captured: dict[str, object] = {}

    class Scientific:
        scientifically_valid = True

        def model_dump(self, *, mode: str):
            assert mode == "json"
            return {"scientifically_valid": True}

    def assess(**kwargs):
        captured.update(kwargs)
        return Scientific(), {"movement": "fixture"}, {"recovery": "fixture"}

    import sloforge.helix.characterization.gpu_reclamation_v10_postprocess as postprocess

    monkeypatch.setattr(postprocess, "assess_v10_directory", assess)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(FINALIZER),
            "--config-path",
            str(config_path),
            "--run-root",
            str(run_root),
            "--gpu-hours-path",
            str(ledger_path),
            "--modal-cleanup-evidence",
            str(cleanup_path),
        ],
    )

    assert module.main() == 0
    assert captured["prerequisites"] is prerequisites


def test_gpu_phase_ledger_records_before_after_actual_and_conservative_totals(
    tmp_path: Path,
) -> None:
    from sloforge.helix.characterization.gpu_reclamation_v10_validity import V10Phase

    module = _load("exp004_v10_phase_ledger")
    run_root = tmp_path / "run"
    _write(
        run_root / "readiness/both-engines-ready.json",
        {"engine_evidence": [{"engine_started_ns": 1}, {"engine_started_ns": 2}]},
    )
    _write(
        run_root / "sanity/12-rps/plan.json",
        {"warmup_start_ns": 10_000_000_000, "measurement_end_ns": 20_000_000_000},
    )
    _write(
        run_root / "sanity/15-rps/plan.json",
        {"warmup_start_ns": 21_000_000_000, "measurement_end_ns": 30_000_000_000},
    )
    _write(
        run_root / "calibration/final-request-state-reset.json",
        [{"ended_ns": 30_000_000_000}, {"ended_ns": 30_000_000_001}],
    )
    _write(
        run_root / "barriers/rollout.transaction-ready.json",
        {"observed_at_monotonic_ns": 34_000_000_000},
    )
    _write(run_root / "serving/result.json", {"start_ns": 35_000_000_000})
    _write(
        run_root / "function-completion.json",
        {
            "gpu_seconds": 210.0,
            "absolute_deadlines": {"function_entry_monotonic_ns": 0},
        },
    )
    second = 1_000_000_000
    timestamps = {
        V10Phase.CONTROL_STABLE: 40 * second,
        V10Phase.LOAD_SPIKE_BEGIN: 40 * second,
        V10Phase.RECLAIM_TRIGGER: 60 * second,
        V10Phase.BRANCH_QUIESCE_BEGIN: 61 * second,
        V10Phase.BRANCH_QUIESCE_END: 62 * second,
        V10Phase.STATE_CAPTURE_BEGIN: 62 * second,
        V10Phase.STATE_CAPTURE_END: 63 * second,
        V10Phase.STATE_TRANSFORM_BEGIN: 63 * second,
        V10Phase.STATE_TRANSFORM_END: 64 * second,
        V10Phase.D2H_BEGIN: 64 * second,
        V10Phase.D2H_END: 65 * second,
        V10Phase.INTEGRITY_BEGIN: 65 * second,
        V10Phase.STATE_PUBLISH: 67 * second,
        V10Phase.GPU1_STATE_RELEASE_BEGIN: 67 * second,
        V10Phase.GPU1_HBM_RECLAIM_CONFIRMED: 69 * second,
        V10Phase.GPU1_SERVING_ENABLE: 70 * second,
        V10Phase.SERVING_SLO_RESTORED: 80 * second,
        V10Phase.SERVING_SLO_STABILITY_BEGIN: 75 * second,
        V10Phase.SERVING_SLO_STABILITY_END: 80 * second,
        V10Phase.RESTORE_TRIGGER: 90 * second,
        V10Phase.H2D_BEGIN: 91 * second,
        V10Phase.H2D_END: 92 * second,
        V10Phase.STATE_IMPORT_BEGIN: 93 * second,
        V10Phase.STATE_IMPORT_END: 98 * second,
        V10Phase.DESTINATION_NATIVE_WRITE_BEGIN: 94 * second,
        V10Phase.DESTINATION_NATIVE_WRITE_END: 95 * second,
        V10Phase.STATE_VALIDATE_BEGIN: 95 * second,
        V10Phase.STATE_VALIDATE_END: 96 * second,
        V10Phase.BRANCH_RESUME_BEGIN: 96 * second,
        V10Phase.ROLLOUT_CONTINUATION_COMPLETE: 100 * second,
    }
    scientific = SimpleNamespace(
        timeline=SimpleNamespace(timestamp=lambda phase: timestamps[phase])
    )
    path = module._write_gpu_phase_ledger(
        run_root=run_root,
        scientific=scientific,
        settlement={
            "attempt_id": "exp004-v10-fixture",
            "actual_gpu_seconds": 210.0,
            "actual_gpu_cost_usd": 210.0 / 3_600.0 * 2.4984,
            "invocation_conservative_cost_usd": 212.0 / 3_600.0 * 2.4984,
            "cumulative_conservative_gpu_cost_usd": 1.0,
        },
        invocation_gpu_seconds=212.0,
        conservative_gpu_seconds_before=100.0,
        conservative_gpu_seconds_after=312.0,
        hard_ceiling_gpu_seconds=10_800.0,
    )
    payload = json.loads(path.read_text())
    assert len(payload["phases"]) == 20
    assert payload["phases"][0]["phase"] == "modal_function_startup"
    assert payload["phases"][0]["actual_A100_seconds"] is None
    assert all(
        row["after_monotonic_ns"] >= row["before_monotonic_ns"]
        and row["actual_A100_seconds"] == 2 * row["actual_wall_seconds"]
        for row in payload["phases"][1:]
    )
    assert payload["actual_invocation_gpu_seconds"] == 210.0
    assert payload["conservative_invocation_gpu_seconds"] == 212.0
    assert payload["remaining_hard_budget_gpu_seconds"] == 10_488.0
