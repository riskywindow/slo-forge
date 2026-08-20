from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from sloforge.helix.characterization.gpu_reclamation_methodology import (
    Experiment004GpuHourLedger,
)
from sloforge.helix.characterization.gpu_reclamation_v11_methodology import (
    Experiment004V11IntegratedConfig,
    canonical_json_bytes,
)

ROOT = Path(__file__).resolve().parents[2]
COORDINATOR = ROOT / "tools/branchfabric-experiment-004-v11-final.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("exp004_v11_final_coordinator", COORDINATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _base_ledger(module: ModuleType) -> Experiment004GpuHourLedger:
    return Experiment004GpuHourLedger(
        hard_additional_gpu_seconds=21_600.0,
        consumed_additional_gpu_seconds=0.0,
    )


def _config(
    module: ModuleType,
    *,
    ledger_sha256: str,
    budget_sha256: str = "4" * 64,
) -> Experiment004V11IntegratedConfig:
    return Experiment004V11IntegratedConfig(
        attempt_id=module.ATTEMPT_ID,
        seed=41,
        offline_gate_manifest="artifacts/offline.json",
        offline_gate_manifest_sha256="1" * 64,
        micro_validation_artifact="artifacts/micro.json",
        micro_validation_sha256="2" * 64,
        post_micro_review_manifest="artifacts/reviews.json",
        post_micro_review_manifest_sha256="3" * 64,
        budget_authorization="artifacts/budget.json",
        budget_authorization_sha256=budget_sha256,
        ledger_sha256_before_reservation=ledger_sha256,
    )


def _write_ledger(module: ModuleType, path: Path, ledger: Experiment004GpuHourLedger) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(module._ledger_bytes(ledger))


def _inventory(index: int) -> dict[str, Any]:
    return {
        "index": index,
        "uuid": f"GPU-00000000-0000-0000-0000-00000000000{index}",
        "name": "NVIDIA A100-SXM4-80GB",
        "driver_version": "580.95.05",
        "memory_total_mib": 81_920,
        "memory_used_mib": 4,
        "utilization_percent": 0,
    }


def _remote_envelope(
    module: ModuleType,
    *,
    config: Experiment004V11IntegratedConfig,
    reservation_id: str,
    commitment: str,
) -> dict[str, Any]:
    cleanup = {"status": "PASS", "pass": True}
    inventory = [_inventory(0), _inventory(1)]
    controller = {
        "status": "succeeded",
        "inventory_before": inventory,
        "inventory_after": inventory,
        "in_function_cleanup": cleanup,
    }
    entry = 1_000_000_000
    remote = {
        "schema_version": "sloforge.branchfabric.experiment-004-v11-integrated-completion/v1",
        "status": "provisional",
        "scientific_status": "pending-local-budget-settlement-and-provider-cleanup",
        "attempt_id": config.attempt_id,
        "reservation_id": reservation_id,
        "reservation_commitment_sha256": commitment,
        "config_sha256": _sha(canonical_json_bytes(config)),
        "function_call_id": "fc-test-v11-c",
        "requested_gpu": "A100-80GB:2",
        "gpu_count": 2,
        "controller_and_analysis_interval_seconds": 4.0,
        "gpu_allocation_seconds_status": "pending-final-function-return",
        "hardware_comparability": {"direct_v10_timing_comparable": True},
        "bound_artifacts": {
            "offline_gate_manifest": config.offline_gate_manifest,
            "offline_gate_manifest_sha256": config.offline_gate_manifest_sha256,
            "micro_validation_artifact": config.micro_validation_artifact,
            "micro_validation_sha256": config.micro_validation_sha256,
            "post_micro_review_manifest": config.post_micro_review_manifest,
            "post_micro_review_manifest_sha256": config.post_micro_review_manifest_sha256,
            "budget_authorization": config.budget_authorization,
            "budget_authorization_sha256": config.budget_authorization_sha256,
        },
        "absolute_deadlines": {
            "function_entry_monotonic_ns": entry,
            "controller_deadline_monotonic_ns": entry + 578_000_000_000,
            "function_deadline_monotonic_ns": entry + 588_000_000_000,
            "post_controller_reserve_seconds": 10.0,
        },
        "controller": controller,
        "in_function_cleanup": cleanup,
        "run_error": None,
        "completed_at_utc": "2026-08-19T21:00:00+00:00",
        "gpu_allocation_seconds": 5.0,
        "gpu_seconds": 10.0,
        "gpu_hours": 10.0 / 3600.0,
        "remote_prefix": f"{module.REMOTE_PREFIX}/{config.attempt_id}",
        "remote_manifest_sha256": "0" * 64,
    }
    return {
        "result": remote,
        "materialized": {
            "remote_path": remote["remote_prefix"],
            "volume_name": module.RESULTS_VOLUME,
        },
    }


def _materialize_bundle(module: ModuleType, root: Path, envelope: dict[str, Any]) -> Path:
    remote = envelope["result"]
    final_only = {
        "gpu_allocation_seconds",
        "gpu_seconds",
        "gpu_hours",
        "remote_prefix",
        "remote_manifest_sha256",
    }
    completion = {key: value for key, value in remote.items() if key not in final_only}
    (root / "in_function_cleanup.json").parent.mkdir(parents=True, exist_ok=True)
    (root / "function-completion.json").write_text(
        json.dumps(completion, sort_keys=True, separators=(",", ":")) + "\n"
    )
    (root / "in_function_cleanup.json").write_text(
        json.dumps(remote["in_function_cleanup"], sort_keys=True, separators=(",", ":")) + "\n"
    )
    artifacts = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            artifacts.append(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha(path.read_bytes()),
                }
            )
    manifest = {
        "schema_version": (
            "sloforge.branchfabric.experiment-004-v11-integrated-remote-manifest/v1"
        ),
        "attempt_id": remote["attempt_id"],
        "remote_prefix": remote["remote_prefix"],
        "artifacts": artifacts,
    }
    manifest_path = root / "REMOTE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n")
    remote["remote_manifest_sha256"] = _sha(manifest_path.read_bytes())
    return manifest_path


def test_sealed_config_requires_fresh_attempt_and_current_ledger(tmp_path: Path) -> None:
    module = _module()
    ledger_path = tmp_path / "gpu-hours.json"
    _write_ledger(module, ledger_path, _base_ledger(module))
    config = _config(module, ledger_sha256=_sha(ledger_path.read_bytes()))
    config_path = tmp_path / "config.json"
    config_path.write_bytes(canonical_json_bytes(config))

    loaded, seal = module._load_sealed_config(
        config_path,
        repository_root=tmp_path,
        ledger_path=ledger_path,
        seal_verifier=lambda candidate, root: {
            "status": "PASS",
            "attempt_id": candidate.attempt_id,
            "root": str(root),
        },
    )
    assert loaded == config
    assert seal["status"] == "PASS"

    ledger_path.write_bytes(ledger_path.read_bytes() + b"\n")
    with pytest.raises(RuntimeError, match="current pre-reservation ledger"):
        module._load_sealed_config(
            config_path,
            repository_root=tmp_path,
            ledger_path=ledger_path,
            seal_verifier=lambda candidate, root: {"status": "PASS"},
        )


def test_atomic_reservation_is_exact_two_a100_for_588_seconds(tmp_path: Path) -> None:
    module = _module()
    ledger_path = tmp_path / "gpu-hours.json"
    _write_ledger(module, ledger_path, _base_ledger(module))
    config = _config(module, ledger_sha256=_sha(ledger_path.read_bytes()))

    updated, preflight, commitment = module._reserve_locked(
        config,
        ledger_path=ledger_path,
        reservation_id="exp004-v11-c-reservation-test",
    )
    assert len(updated.reservations) == 1
    reservation = updated.reservations[0]
    assert reservation.invocation_id == module.ATTEMPT_ID
    assert reservation.requested_gpu == "A100-80GB"
    assert reservation.gpu_count == 2
    assert reservation.maximum_wall_seconds == 588.0
    assert reservation.maximum_gpu_seconds == 1_176.0
    assert preflight["proposed_maximum_gpu_seconds"] == 1_176.0
    assert commitment == _sha(canonical_json_bytes(reservation))
    assert module._load_ledger(ledger_path) == updated
    assert ledger_path.read_bytes() == module._ledger_bytes(updated)


def test_modal_invocation_has_one_time_token_and_bounded_exact_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    observed: dict[str, Any] = {}

    def runner(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        observed.update({"command": command, **kwargs})
        return subprocess.CompletedProcess(command, 0, stdout="{}\n", stderr="")

    monkeypatch.setenv("UNRELATED_ENVIRONMENT", "preserved")
    config_path = tmp_path / "config.json"
    config_path.write_text("{}")
    completed, elapsed = module._invoke_modal(
        config_path,
        reservation_id="exp004-v11-c-reservation-test",
        budget_usd=40.0,
        runner=runner,
    )
    assert completed.returncode == 0
    assert elapsed >= 0.0
    assert observed["command"] == [
        "modal",
        "run",
        str(module._APP),
        "--config-path",
        str(config_path),
    ]
    assert observed["timeout"] == 828.0
    environment = observed["env"]
    assert environment["SLOFORGE_EXP004_RESERVATION_ID"] == ("exp004-v11-c-reservation-test")
    assert environment["SLOFORGE_GPU_BUDGET_USD"] == "40"
    assert len(environment["SLOFORGE_MODAL_PREFLIGHT_TOKEN"]) == 64
    int(environment["SLOFORGE_MODAL_PREFLIGHT_TOKEN"], 16)
    assert environment["UNRELATED_ENVIRONMENT"] == "preserved"


def test_exact_result_manifest_and_measured_settlement(tmp_path: Path) -> None:
    module = _module()
    ledger_path = tmp_path / "gpu-hours.json"
    _write_ledger(module, ledger_path, _base_ledger(module))
    config = _config(module, ledger_sha256=_sha(ledger_path.read_bytes()))
    reservation_id = "exp004-v11-c-reservation-test"
    _updated, _preflight, commitment = module._reserve_locked(
        config, ledger_path=ledger_path, reservation_id=reservation_id
    )
    envelope = _remote_envelope(
        module,
        config=config,
        reservation_id=reservation_id,
        commitment=commitment,
    )
    local_root = tmp_path / "raw" / config.attempt_id
    manifest = _materialize_bundle(module, local_root, envelope)

    parsed = module._parse_exact_result(
        "Modal log\n" + json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n"
    )
    remote, materialized, inventory = module._validate_remote_result(
        parsed,
        config=config,
        reservation_id=reservation_id,
        reservation_commitment_sha256=commitment,
    )
    assert (
        module._verify_downloaded_manifest(local_root, remote=remote, materialized=materialized)
        == manifest
    )
    settled, accounted_wall = module._settle_verified(
        ledger_path=ledger_path,
        reservation_id=reservation_id,
        remote=remote,
        inventory=inventory,
        manifest_path=manifest,
        coordinator_elapsed_seconds=5.5,
    )
    assert accounted_wall == 5.0
    assert settled.consumed_additional_gpu_seconds == 10.0
    assert settled.reservations == ()
    assert settled.intervals[-1].function_call_id == "fc-test-v11-c"


def test_gpu_identity_allows_dynamic_utilization_and_memory_samples() -> None:
    module = _module()
    before = [_inventory(0), _inventory(1)]
    after = [dict(row) for row in before]
    after[0].update({"memory_used_mib": 4096, "utilization_percent": 14})
    after[1].update({"memory_used_mib": 61440, "utilization_percent": 100})

    inventory = module._exact_inventory(
        {"controller": {"inventory_before": before, "inventory_after": after}}
    )

    assert [row["uuid"] for row in inventory] == [row["uuid"] for row in before]


def test_download_uses_exact_immutable_modal_prefix(tmp_path: Path) -> None:
    module = _module()
    local_root = tmp_path / "raw" / module.ATTEMPT_ID
    materialized = {
        "remote_path": f"{module.REMOTE_PREFIX}/{module.ATTEMPT_ID}",
        "volume_name": module.RESULTS_VOLUME,
    }
    observed: dict[str, Any] = {}

    def runner(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        observed.update({"command": command, **kwargs})
        local_root.mkdir(parents=True)
        return subprocess.CompletedProcess(command, 0, stdout="downloaded\n", stderr="")

    module._download_result(materialized, local_root=local_root, runner=runner)
    assert observed["command"] == [
        "modal",
        "volume",
        "get",
        module.RESULTS_VOLUME,
        f"{module.REMOTE_PREFIX}/{module.ATTEMPT_ID}",
        str(local_root.parent),
    ]
    assert observed["timeout"] == 120.0
    with pytest.raises(FileExistsError, match="immutable integrated result"):
        module._download_result(materialized, local_root=local_root, runner=runner)


def test_manifest_tamper_and_duplicate_result_are_rejected(tmp_path: Path) -> None:
    module = _module()
    ledger_path = tmp_path / "gpu-hours.json"
    _write_ledger(module, ledger_path, _base_ledger(module))
    config = _config(module, ledger_sha256=_sha(ledger_path.read_bytes()))
    reservation_id = "exp004-v11-c-reservation-test"
    _updated, _preflight, commitment = module._reserve_locked(
        config, ledger_path=ledger_path, reservation_id=reservation_id
    )
    envelope = _remote_envelope(
        module, config=config, reservation_id=reservation_id, commitment=commitment
    )
    local_root = tmp_path / "raw" / config.attempt_id
    _materialize_bundle(module, local_root, envelope)
    line = json.dumps(envelope, sort_keys=True, separators=(",", ":"))
    with pytest.raises(ValueError, match="2 exact"):
        module._parse_exact_result(f"{line}\n{line}\n")

    (local_root / "function-completion.json").write_text("tampered\n")
    with pytest.raises(ValueError, match="failed integrity"):
        module._verify_downloaded_manifest(
            local_root,
            remote=envelope["result"],
            materialized=envelope["materialized"],
        )


def test_unverifiable_failure_charges_full_reservation_without_fabricated_actuals(
    tmp_path: Path,
) -> None:
    module = _module()
    ledger_path = tmp_path / "gpu-hours.json"
    _write_ledger(module, ledger_path, _base_ledger(module))
    config = _config(module, ledger_sha256=_sha(ledger_path.read_bytes()))
    reservation_id = "exp004-v11-c-reservation-test"
    module._reserve_locked(config, ledger_path=ledger_path, reservation_id=reservation_id)
    completed = subprocess.CompletedProcess(
        ["modal", "run"], 1, stdout="remote stdout", stderr="remote stderr"
    )
    result = module._charge_active_failure(
        ledger_path=ledger_path,
        reservation_id=reservation_id,
        attempt_id=config.attempt_id,
        failure_root=tmp_path / "failures",
        stage="result-validation",
        error=ValueError("bad remote identity"),
        client_elapsed_seconds=7.0,
        completed=completed,
    )
    ledger = module._load_ledger(ledger_path)
    assert result["status"] == "CONSERVATIVELY_CHARGED"
    assert result["charged_gpu_seconds"] == 1_176.0
    assert ledger.reservations == ()
    assert ledger.intervals == ()
    assert ledger.consumed_additional_gpu_seconds == 1_176.0
    charge = ledger.conservative_failure_charges[-1]
    assert charge.conservative_gpu_seconds == 1_176.0
    evidence = json.loads(Path(result["failure_evidence"]).read_text())
    assert evidence["actual_gpu_seconds"] is None
    assert evidence["accounting_policy"] == (
        "charge-full-preflight-bound-without-fabricated-measurement"
    )


def test_provider_cleanup_inputs_remain_pending_and_do_not_launder_function_cleanup(
    tmp_path: Path,
) -> None:
    module = _module()
    ledger_path = tmp_path / "gpu-hours.json"
    _write_ledger(module, ledger_path, _base_ledger(module))
    config = _config(module, ledger_sha256=_sha(ledger_path.read_bytes()))
    envelope = _remote_envelope(
        module,
        config=config,
        reservation_id="exp004-v11-c-reservation-test",
        commitment="a" * 64,
    )
    local_root = tmp_path / "raw" / config.attempt_id
    manifest = _materialize_bundle(module, local_root, envelope)
    output = tmp_path / "provider-cleanup-inputs.json"
    payload = module._emit_provider_cleanup_inputs(
        output,
        remote=envelope["result"],
        reservation_id="exp004-v11-c-reservation-test",
        manifest_path=manifest,
        ledger_path=ledger_path,
    )
    assert payload["status"] == "PENDING_EXPLICIT_PROVIDER_POST_RETURN_AUDIT"
    assert payload["in_function_cleanup_status"] == "PASS"
    assert payload["provider_cleanup_pass"] is None
    assert set(payload["required_zero_counts"]) == {
        "active_apps",
        "running_tasks",
        "running_containers",
        "endpoints",
        "provider_reservations",
        "owned_child_processes",
        "profilers",
    }
    assert json.loads(output.read_text()) == payload


def test_budget_is_hash_bound_and_cannot_exceed_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    budget_path = tmp_path / "artifacts/budget.json"
    budget_path.parent.mkdir(parents=True)
    budget_path.write_text(json.dumps({"authorized_gpu_budget_usd": 40.0}) + "\n")
    config = _config(
        module,
        ledger_sha256="5" * 64,
        budget_sha256=_sha(budget_path.read_bytes()),
    )
    monkeypatch.setenv("SLOFORGE_GPU_BUDGET_USD", "40")
    assert module._require_launch_budget(config, tmp_path) == 40.0
    monkeypatch.setenv("SLOFORGE_GPU_BUDGET_USD", "40.01")
    with pytest.raises(RuntimeError, match="without exceeding authorization"):
        module._require_launch_budget(config, tmp_path)


def test_coordinator_source_never_invokes_gpu_during_import() -> None:
    source = COORDINATOR.read_text()
    assert source.count('"modal", "run"') == 1
    assert "subprocess.run(" not in source
    assert "cuda" not in source.lower()
    assert "torch" not in source.lower()
    assert "gpu_count=2" in source
    assert "maximum_wall_seconds=FUNCTION_WALL_SECONDS" in source
    assert os.access(COORDINATOR, os.R_OK)
