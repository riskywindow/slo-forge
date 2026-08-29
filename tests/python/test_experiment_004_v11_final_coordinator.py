from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from sloforge.helix.characterization.gpu_reclamation_methodology import (
    Experiment004GpuHourLedger,
)
from sloforge.helix.characterization.gpu_reclamation_v11_methodology import (
    Experiment004V11IntegratedConfig,
    Experiment004V11TargetedSourceIdentityConfig,
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


def _target_config(
    module: ModuleType,
    *,
    ledger_sha256: str,
    budget_sha256: str = "4" * 64,
) -> Experiment004V11TargetedSourceIdentityConfig:
    base = _config(
        module,
        ledger_sha256=ledger_sha256,
        budget_sha256=budget_sha256,
    ).model_dump(mode="python")
    base.update(
        {
            "schema_version": (
                "sloforge.branchfabric.experiment-004-v11-targeted-source-identity-config/v1"
            ),
            "execution_mode": "targeted-source-identity-v11",
            "attempt_id": module.TARGETED_ATTEMPT_ID,
            "terminal_phase": "SOURCE_ALLOCATION_IDENTITY_GATE",
            "full_export_authorized": False,
            "source_release_authorized": False,
            "maximum_wall_seconds": 300.0,
        }
    )
    return Experiment004V11TargetedSourceIdentityConfig.model_validate(base, strict=True)


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


def _pre_worker_failure_envelope(
    module: ModuleType,
    *,
    config: Experiment004V11IntegratedConfig,
    reservation_id: str,
    commitment: str,
) -> dict[str, Any]:
    envelope = _remote_envelope(
        module,
        config=config,
        reservation_id=reservation_id,
        commitment=commitment,
    )
    raw = json.loads(
        (
            ROOT / "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/"
            "raw/exp004-v11-integrated-s41-h/function-completion.json"
        ).read_text()
    )
    remote = envelope["result"]
    remote.update(raw)
    remote["controller"]["attempt_id"] = config.attempt_id
    remote.update(
        {
            "attempt_id": config.attempt_id,
            "reservation_id": reservation_id,
            "reservation_commitment_sha256": commitment,
            "config_sha256": _sha(canonical_json_bytes(config)),
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
            "gpu_allocation_seconds": 30.0,
            "gpu_seconds": 60.0,
            "gpu_hours": 60.0 / 3600.0,
            "remote_prefix": f"{module.REMOTE_PREFIX}/{config.attempt_id}",
            "remote_manifest_sha256": "0" * 64,
        }
    )
    envelope["materialized"]["remote_path"] = remote["remote_prefix"]
    return envelope


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


def test_targeted_mode_reserves_and_validates_exact_300_second_stop_path(
    tmp_path: Path,
) -> None:
    module = _module()
    ledger_path = tmp_path / "gpu-hours.json"
    _write_ledger(module, ledger_path, _base_ledger(module))
    config = _target_config(module, ledger_sha256=_sha(ledger_path.read_bytes()))
    config_path = tmp_path / "targeted-config.json"
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

    reservation_id = "exp004-v11-target-reservation-test"
    updated, preflight, commitment = module._reserve_locked(
        config,
        ledger_path=ledger_path,
        reservation_id=reservation_id,
    )
    reservation = updated.reservations[0]
    assert reservation.maximum_wall_seconds == 300.0
    assert reservation.maximum_gpu_seconds == 600.0
    assert preflight["proposed_maximum_gpu_seconds"] == 600.0

    envelope = _remote_envelope(
        module,
        config=config,
        reservation_id=reservation_id,
        commitment=commitment,
    )
    remote = envelope["result"]
    remote.update(
        {
            "schema_version": (
                "sloforge.branchfabric.experiment-004-v11-targeted-source-identity-completion/v1"
            ),
            "scientific_status": "targeted-identity-pass-pending-cleanup",
            "absolute_deadlines": {
                "function_entry_monotonic_ns": 1_000_000_000,
                "controller_deadline_monotonic_ns": 291_000_000_000,
                "function_deadline_monotonic_ns": 301_000_000_000,
                "post_controller_reserve_seconds": 10.0,
            },
        }
    )
    validated, _materialized, _inventory_rows = module._validate_remote_result(
        envelope,
        config=config,
        reservation_id=reservation_id,
        reservation_commitment_sha256=commitment,
    )
    assert validated["schema_version"].endswith("targeted-source-identity-completion/v1")


def test_coordinator_rejects_consumed_targeted_attempt_a(tmp_path: Path) -> None:
    module = _module()
    ledger_path = tmp_path / "gpu-hours.json"
    _write_ledger(module, ledger_path, _base_ledger(module))
    config = _target_config(module, ledger_sha256=_sha(ledger_path.read_bytes())).model_copy(
        update={"attempt_id": "exp004-v11-targeted-identity-s41-a"}
    )
    config_path = tmp_path / "targeted-attempt-a.json"
    config_path.write_bytes(canonical_json_bytes(config))

    with pytest.raises(ValueError, match="only the next exact v11 attempt"):
        module._load_sealed_config(
            config_path,
            repository_root=tmp_path,
            ledger_path=ledger_path,
            seal_verifier=lambda candidate, root: {"status": "PASS"},
        )


def test_coordinator_rejects_consumed_integrated_attempt_e(tmp_path: Path) -> None:
    module = _module()
    ledger_path = tmp_path / "gpu-hours.json"
    _write_ledger(module, ledger_path, _base_ledger(module))
    config = _config(module, ledger_sha256=_sha(ledger_path.read_bytes())).model_copy(
        update={"attempt_id": "exp004-v11-integrated-s41-e"}
    )
    config_path = tmp_path / "consumed-attempt-e.json"
    config_path.write_bytes(canonical_json_bytes(config))

    with pytest.raises(ValueError, match="only the next exact v11 attempt"):
        module._load_sealed_config(
            config_path,
            repository_root=tmp_path,
            ledger_path=ledger_path,
            seal_verifier=lambda candidate, root: {"status": "PASS"},
        )


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
        "/usr/bin/caffeinate",
        "-dimsu",
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


def test_modal_image_prebuild_retries_without_gpu_reservation_and_cleans_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    calls: list[list[str]] = []
    deploy_count = 0

    def runner(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal deploy_count
        calls.append(command)
        if "deploy" in command:
            deploy_count += 1
            return subprocess.CompletedProcess(
                command,
                1 if deploy_count == 1 else 0,
                stdout="Building image im-ExactCache\n",
                stderr="transient" if deploy_count == 1 else "",
            )
        if command[:3] == ["modal", "app", "stop"]:
            return subprocess.CompletedProcess(command, 0, stdout="stopped\n", stderr="")
        if command == ["modal", "app", "list", "--json"]:
            return subprocess.CompletedProcess(command, 0, stdout="[]\n", stderr="")
        if command == ["modal", "container", "list", "--json"]:
            return subprocess.CompletedProcess(command, 0, stdout="[]\n", stderr="")
        raise AssertionError(command)

    monkeypatch.setenv("UNRELATED_ENVIRONMENT", "preserved")
    result = module._prebuild_modal_image(budget_usd=80.0, runner=runner)

    assert result["status"] == "PASS"
    assert result["gpu_reservation_created"] is False
    assert result["gpu_function_invoked"] is False
    assert result["attempt_count"] == 2
    assert result["provider_zero_before_gpu_reservation"] is True
    assert all(item["provider_cleanup"]["provider_zero"] for item in result["attempts"])
    assert result["attempts"][1]["image_ids"] == ["im-ExactCache"]
    assert sum("deploy" in command for command in calls) == 2
    assert sum(command[:3] == ["modal", "app", "stop"] for command in calls) == 2
    assert sum(command == ["modal", "container", "list", "--json"] for command in calls) == 2
    deploy = next(command for command in calls if "deploy" in command)
    assert deploy[:3] == ["/usr/bin/caffeinate", "-dimsu", "modal"]


def test_prebuild_cleanup_accepts_only_stopped_zero_task_apps() -> None:
    module = _module()

    def runner(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if command[:3] == ["modal", "app", "stop"]:
            return subprocess.CompletedProcess(command, 0, stdout="stopped\n", stderr="")
        if command == ["modal", "app", "list", "--json"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps([{"state": "stopped", "tasks": "0"}]) + "\n",
                stderr="",
            )
        if command == ["modal", "container", "list", "--json"]:
            return subprocess.CompletedProcess(command, 0, stdout="[]\n", stderr="")
        raise AssertionError(command)

    result = module._stop_modal_app_and_require_provider_zero(runner=runner)
    assert result["provider_zero"] is True
    assert result["app_list_observations"][-1]["stopped_or_disabled_apps_only"] is True


@pytest.mark.parametrize(
    "row",
    (
        {"state": "deployed", "tasks": "0"},
        {"state": "stopped", "tasks": "1"},
        {"state": "stopped", "tasks": False},
        {"state": "stopped"},
    ),
)
def test_prebuild_cleanup_rejects_active_or_malformed_apps(row: dict[str, Any]) -> None:
    module = _module()

    def runner(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if command[:3] == ["modal", "app", "stop"]:
            return subprocess.CompletedProcess(command, 0, stdout="stopped\n", stderr="")
        if command == ["modal", "app", "list", "--json"]:
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps([row]), stderr="")
        if command == ["modal", "container", "list", "--json"]:
            return subprocess.CompletedProcess(command, 0, stdout="[]\n", stderr="")
        raise AssertionError(command)

    with pytest.raises(RuntimeError, match="provider zero"):
        module._stop_modal_app_and_require_provider_zero(runner=runner)


def test_bounded_runner_reaps_timeout_process_group() -> None:
    module = _module()
    script = """
import os
import signal
import subprocess
import sys
import time
subprocess.Popen([
    sys.executable,
    "-c",
    "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)",
])
print(os.getpid(), flush=True)
time.sleep(30)
"""
    result = module._run_bounded_process_group(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=0.1,
    )
    pgid = int(result.stdout.splitlines()[0])
    assert result.returncode == 124
    assert "SLOFORGE_PROCESS_GROUP_TIMEOUT_CLEANUP" in result.stderr
    assert module._process_group_exists(pgid) is False


def test_repeated_seal_projection_ignores_only_verification_clock() -> None:
    module = _module()
    before = {
        "status": "PASS",
        "attempt_id": module.ATTEMPT_ID,
        "verified_at_monotonic_ns": 100,
        "binding": {"sha256": "a" * 64},
    }
    after = before | {"verified_at_monotonic_ns": 200}
    assert module._deterministic_seal_projection(before) == (
        module._deterministic_seal_projection(after)
    )
    tampered = after | {"binding": {"sha256": "b" * 64}}
    assert module._deterministic_seal_projection(before) != (
        module._deterministic_seal_projection(tampered)
    )


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


def test_exact_pre_worker_failure_downloads_then_charges_conservative_bound(
    tmp_path: Path,
) -> None:
    module = _module()
    ledger_path = tmp_path / "gpu-hours.json"
    _write_ledger(module, ledger_path, _base_ledger(module))
    config = _config(module, ledger_sha256=_sha(ledger_path.read_bytes()))
    reservation_id = "exp004-v11-h-pre-worker-test"
    _updated, _preflight, commitment = module._reserve_locked(
        config,
        ledger_path=ledger_path,
        reservation_id=reservation_id,
    )
    envelope = _pre_worker_failure_envelope(
        module,
        config=config,
        reservation_id=reservation_id,
        commitment=commitment,
    )
    local_root = tmp_path / "raw" / config.attempt_id
    manifest = _materialize_bundle(module, local_root, envelope)

    remote, materialized, inventory = module._validate_remote_result(
        envelope,
        config=config,
        reservation_id=reservation_id,
        reservation_commitment_sha256=commitment,
    )
    assert inventory is None
    assert (
        module._verify_downloaded_manifest(local_root, remote=remote, materialized=materialized)
        == manifest
    )
    result = module._charge_active_failure(
        ledger_path=ledger_path,
        reservation_id=reservation_id,
        attempt_id=config.attempt_id,
        failure_root=tmp_path / "failures",
        stage="verified-pre-worker-failure-conservative-settlement",
        error=RuntimeError(
            "FileNotFoundError: [Errno 2] No such file or directory: "
            "'/opt/sloforge/artifacts/branchfabric/gpu-validation/experiment-004/raw'"
        ),
        client_elapsed_seconds=31.0,
        completed=None,
        verified_remote=remote,
        verified_manifest_path=manifest,
    )
    assert result["status"] == "CONSERVATIVELY_CHARGED"
    ledger = module._load_ledger(ledger_path)
    assert ledger.reservations == ()
    assert ledger.intervals == ()
    assert ledger.consumed_additional_gpu_seconds == 1_176.0
    evidence = json.loads(Path(result["failure_evidence"]).read_text())
    assert evidence["actual_gpu_seconds"] is None
    assert evidence["error"] == {
        "type": "RuntimeError",
        "message": (
            "FileNotFoundError: [Errno 2] No such file or directory: "
            "'/opt/sloforge/artifacts/branchfabric/gpu-validation/experiment-004/raw'"
        ),
    }
    assert evidence["verified_remote_failure"] == {
        "status": "failed",
        "scientific_status": "invalid",
        "function_call_id": "fc-01M122M4JYMDD27K30ZJSR1P56",
        "cleanup_scope": "PRE_WORKER_PREFLIGHT",
        "failure_stage": "SEALED_EVIDENCE",
        "controller_error": {
            "message": (
                "[Errno 2] No such file or directory: "
                "'/opt/sloforge/artifacts/branchfabric/gpu-validation/experiment-004/raw'"
            ),
            "type": "FileNotFoundError",
        },
        "remote_manifest": str(manifest),
        "remote_manifest_sha256": _sha(manifest.read_bytes()),
        "remote_reported_allocation_seconds_diagnostic_only": 30.0,
        "remote_reported_gpu_seconds_diagnostic_only": 60.0,
    }


def test_zero_compute_pre_worker_failure_remains_conservative_and_inventory_free(
    tmp_path: Path,
) -> None:
    module = _module()
    ledger_path = tmp_path / "gpu-hours.json"
    _write_ledger(module, ledger_path, _base_ledger(module))
    config = _config(module, ledger_sha256=_sha(ledger_path.read_bytes()))
    reservation_id = "exp004-v11-i-zero-compute-pre-worker-test"
    _updated, _preflight, commitment = module._reserve_locked(
        config,
        ledger_path=ledger_path,
        reservation_id=reservation_id,
    )
    envelope = _pre_worker_failure_envelope(
        module,
        config=config,
        reservation_id=reservation_id,
        commitment=commitment,
    )
    envelope["result"]["controller"]["failure_stage"] = "ZERO_COMPUTE_BEFORE_WORKERS"

    _remote, _materialized, inventory = module._validate_remote_result(
        envelope,
        config=config,
        reservation_id=reservation_id,
        reservation_commitment_sha256=commitment,
    )

    assert inventory is None


def test_pre_worker_failure_may_finish_cleanup_after_operation_deadline(
    tmp_path: Path,
) -> None:
    module = _module()
    ledger_path = tmp_path / "gpu-hours.json"
    _write_ledger(module, ledger_path, _base_ledger(module))
    config = _config(module, ledger_sha256=_sha(ledger_path.read_bytes()))
    reservation_id = "exp004-v11-i-cleanup-reserve-test"
    _updated, _preflight, commitment = module._reserve_locked(
        config,
        ledger_path=ledger_path,
        reservation_id=reservation_id,
    )
    envelope = _pre_worker_failure_envelope(
        module,
        config=config,
        reservation_id=reservation_id,
        commitment=commitment,
    )
    controller = envelope["result"]["controller"]
    controller["ended_ns"] = controller["operation_deadline_ns"] + 1

    _remote, _materialized, inventory = module._validate_remote_result(
        envelope,
        config=config,
        reservation_id=reservation_id,
        reservation_commitment_sha256=commitment,
    )

    assert inventory is None


def test_verified_post_worker_failure_never_reaches_measured_settlement(
    tmp_path: Path,
) -> None:
    module = _module()
    ledger_path = tmp_path / "gpu-hours.json"
    _write_ledger(module, ledger_path, _base_ledger(module))
    config = _config(module, ledger_sha256=_sha(ledger_path.read_bytes()))
    reservation_id = "exp004-v11-i-post-worker-failure-test"
    _updated, _preflight, commitment = module._reserve_locked(
        config,
        ledger_path=ledger_path,
        reservation_id=reservation_id,
    )
    envelope = _remote_envelope(
        module,
        config=config,
        reservation_id=reservation_id,
        commitment=commitment,
    )
    remote = envelope["result"]
    remote.update(
        {
            "status": "failed",
            "scientific_status": "invalid",
            "run_error": {
                "type": "RuntimeError",
                "message": "post-worker failure",
                "traceback": "verified post-worker traceback",
            },
        }
    )
    local_root = tmp_path / "raw" / config.attempt_id
    manifest = _materialize_bundle(module, local_root, envelope)
    validated, _materialized, inventory = module._validate_remote_result(
        envelope,
        config=config,
        reservation_id=reservation_id,
        reservation_commitment_sha256=commitment,
    )
    assert inventory is not None

    charge = module._conservatively_settle_verified_failure(
        ledger_path=ledger_path,
        reservation_id=reservation_id,
        attempt_id=config.attempt_id,
        failure_root=tmp_path / "failures",
        remote=validated,
        inventory=inventory,
        manifest_path=manifest,
        client_elapsed_seconds=5.0,
        completed=subprocess.CompletedProcess(
            ["modal", "run"], 0, stdout="verified failure", stderr=""
        ),
    )

    ledger = module._load_ledger(ledger_path)
    assert charge["status"] == "CONSERVATIVELY_CHARGED"
    assert ledger.intervals == ()
    assert ledger.reservations == ()
    evidence = json.loads(Path(charge["failure_evidence"]).read_text())
    assert evidence["failure_stage"] == ("verified-post-worker-failure-conservative-settlement")
    main_source = COORDINATOR.read_text().split("def main()", maxsplit=1)[1]
    assert main_source.index('if remote["status"] == "failed":') < main_source.index(
        'stage = "measured-settlement"'
    )


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("controller", "inventory_before"), [_inventory(0)]),
        (("controller", "worker_results"), [{"role": "serving"}]),
        (("controller", "sealed_evidence"), {"status": "PASS"}),
        (("controller", "controller_error"), None),
        (("controller", "cleanup_scope"), "UNKNOWN"),
        (("controller", "failure_stage"), "UNKNOWN"),
        (("controller", "cuda_clean_import_audits"), []),
        (("controller", "unexpected"), True),
        (("in_function_cleanup", "pass"), False),
        (("in_function_cleanup", "owned_children"), [{"pid": 99}]),
        (("hardware_comparability", "observed_gpu_names"), ["unexpected"]),
    ),
)
def test_pre_worker_failure_rejects_inventory_worker_cleanup_and_schema_tamper(
    tmp_path: Path,
    path: tuple[str, str],
    value: Any,
) -> None:
    module = _module()
    ledger_path = tmp_path / "gpu-hours.json"
    _write_ledger(module, ledger_path, _base_ledger(module))
    config = _config(module, ledger_sha256=_sha(ledger_path.read_bytes()))
    reservation_id = "exp004-v11-h-pre-worker-test"
    _updated, _preflight, commitment = module._reserve_locked(
        config,
        ledger_path=ledger_path,
        reservation_id=reservation_id,
    )
    envelope = _pre_worker_failure_envelope(
        module,
        config=config,
        reservation_id=reservation_id,
        commitment=commitment,
    )
    envelope["result"][path[0]][path[1]] = value
    if path[0] == "in_function_cleanup":
        envelope["result"]["controller"]["in_function_cleanup"] = envelope["result"][
            "in_function_cleanup"
        ]
    with pytest.raises(ValueError):
        module._validate_remote_result(
            envelope,
            config=config,
            reservation_id=reservation_id,
            reservation_commitment_sha256=commitment,
        )


def test_pre_worker_failure_rejects_truncated_lifecycle_and_cleanup_mismatch(
    tmp_path: Path,
) -> None:
    module = _module()
    ledger_path = tmp_path / "gpu-hours.json"
    _write_ledger(module, ledger_path, _base_ledger(module))
    config = _config(module, ledger_sha256=_sha(ledger_path.read_bytes()))
    reservation_id = "exp004-v11-h-pre-worker-test"
    _updated, _preflight, commitment = module._reserve_locked(
        config,
        ledger_path=ledger_path,
        reservation_id=reservation_id,
    )
    envelope = _pre_worker_failure_envelope(
        module,
        config=config,
        reservation_id=reservation_id,
        commitment=commitment,
    )
    envelope["result"]["in_function_cleanup"]["lifecycle"] = envelope["result"][
        "in_function_cleanup"
    ]["lifecycle"][:-1]
    with pytest.raises(ValueError, match="cleanup differs"):
        module._validate_remote_result(
            envelope,
            config=config,
            reservation_id=reservation_id,
            reservation_commitment_sha256=commitment,
        )

    envelope = _pre_worker_failure_envelope(
        module,
        config=config,
        reservation_id=reservation_id,
        commitment=commitment,
    )
    envelope["result"]["controller"]["in_function_cleanup"] = {
        **envelope["result"]["in_function_cleanup"],
        "pass": False,
    }
    with pytest.raises(ValueError, match="cleanup differs"):
        module._validate_remote_result(
            envelope,
            config=config,
            reservation_id=reservation_id,
            reservation_commitment_sha256=commitment,
        )


@pytest.mark.parametrize(
    "mutate",
    (
        lambda value: value["result"]["in_function_cleanup"].update(parent_pid=True),
        lambda value: value["result"]["in_function_cleanup"].update(
            initial_threads=[False], final_threads=[False]
        ),
        lambda value: value["result"]["in_function_cleanup"]["child_subreaper"].update(
            supported="yes"
        ),
        lambda value: value["result"]["controller"].update(operation_deadline_ns=1),
        lambda value: value["result"]["controller"].update(
            ended_ns=value["result"]["controller"]["cleanup_deadline_ns"] + 1
        ),
        lambda value: value["result"].update(
            run_error={"type": "MadeUp", "message": "unrelated", "traceback": "unrelated"}
        ),
        lambda value: value["result"]["hardware_comparability"].update(
            modal_interconnect_selection_available=True
        ),
    ),
)
def test_pre_worker_failure_rejects_exact_audit_type_order_and_linkage_tamper(
    tmp_path: Path,
    mutate: Any,
) -> None:
    module = _module()
    ledger_path = tmp_path / "gpu-hours.json"
    _write_ledger(module, ledger_path, _base_ledger(module))
    config = _config(module, ledger_sha256=_sha(ledger_path.read_bytes()))
    reservation_id = "exp004-v11-h-pre-worker-test"
    _updated, _preflight, commitment = module._reserve_locked(
        config,
        ledger_path=ledger_path,
        reservation_id=reservation_id,
    )
    envelope = _pre_worker_failure_envelope(
        module,
        config=config,
        reservation_id=reservation_id,
        commitment=commitment,
    )
    mutate(envelope)
    if envelope["result"]["in_function_cleanup"] != envelope["result"]["controller"].get(
        "in_function_cleanup"
    ):
        envelope["result"]["controller"]["in_function_cleanup"] = envelope["result"][
            "in_function_cleanup"
        ]
    with pytest.raises(ValueError):
        module._validate_remote_result(
            envelope,
            config=config,
            reservation_id=reservation_id,
            reservation_commitment_sha256=commitment,
        )


def test_provisional_result_cannot_omit_inventory(tmp_path: Path) -> None:
    module = _module()
    ledger_path = tmp_path / "gpu-hours.json"
    _write_ledger(module, ledger_path, _base_ledger(module))
    config = _config(module, ledger_sha256=_sha(ledger_path.read_bytes()))
    reservation_id = "exp004-v11-h-provisional-test"
    _updated, _preflight, commitment = module._reserve_locked(
        config,
        ledger_path=ledger_path,
        reservation_id=reservation_id,
    )
    envelope = _remote_envelope(
        module,
        config=config,
        reservation_id=reservation_id,
        commitment=commitment,
    )
    envelope["result"]["controller"]["inventory_before"] = []
    envelope["result"]["controller"]["inventory_after"] = []
    with pytest.raises(ValueError, match="exactly two GPU inventory"):
        module._validate_remote_result(
            envelope,
            config=config,
            reservation_id=reservation_id,
            reservation_commitment_sha256=commitment,
        )


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

    dangling = tmp_path / "raw" / "dangling-attempt"
    dangling.symlink_to(tmp_path / "absent", target_is_directory=True)
    with pytest.raises(FileExistsError, match="immutable integrated result"):
        module._download_result(materialized, local_root=dangling, runner=runner)

    linked = tmp_path / "linked-attempt"
    linked.symlink_to(local_root, target_is_directory=True)
    with pytest.raises(ValueError, match="artifact root is invalid"):
        module._verify_downloaded_manifest(
            linked,
            remote={"remote_manifest_sha256": "0" * 64},
            materialized=materialized,
        )


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


def test_targeted_provider_cleanup_input_path_is_attempt_scoped() -> None:
    module = _module()
    source = COORDINATOR.read_text()
    initial = (
        module._EXPERIMENT_ROOT
        / "v11-final/targeted-repro"
        / "provider-cleanup-audit-inputs-exp004-v11-targeted-identity-s41-a.json"
    )
    retry = (
        module._EXPERIMENT_ROOT
        / "v11-final/targeted-repro"
        / f"provider-cleanup-audit-inputs-{module.TARGETED_ATTEMPT_ID}.json"
    )

    assert module.TARGETED_ATTEMPT_ID == "exp004-v11-targeted-identity-s41-b"
    assert initial != retry
    assert 'f"provider-cleanup-audit-inputs-{config.attempt_id}.json"' in source


def test_image_closure_retry_attempt_and_provider_input_are_attempt_scoped() -> None:
    module = _module()
    source = COORDINATOR.read_text()
    integrated = (
        module._EXPERIMENT_ROOT
        / "v11-final/integrated"
        / f"provider-cleanup-audit-inputs-{module.ATTEMPT_ID}.json"
    )
    consumed_attempt_e = (
        module._EXPERIMENT_ROOT
        / "v11-final/integrated"
        / "provider-cleanup-audit-inputs-exp004-v11-integrated-s41-e.json"
    )

    assert module.ATTEMPT_ID == "exp004-v11-integrated-s41-k"
    assert integrated != consumed_attempt_e
    assert source.count('f"provider-cleanup-audit-inputs-{config.attempt_id}.json"') == 2
    assert 'f"exp004-v11-{attempt_suffix}-{secrets.token_hex(10)}"' in source


def test_attempt_k_preimport_closure_is_exact_and_precedes_controller_import(
    tmp_path: Path,
) -> None:
    module = _module()
    config = _config(module, ledger_sha256="5" * 64)

    verified = module._require_retained_capacity_preimport_closure(config, ROOT)
    assert verified == {
        label: {"artifact": reference, "sha256": digest}
        for label, (reference, digest) in module._RETAINED_CAPACITY_PREIMPORT_BINDINGS.items()
    }

    for reference, _digest in module._RETAINED_CAPACITY_PREIMPORT_BINDINGS.values():
        source = ROOT / reference
        destination = tmp_path / reference
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    controller_reference, _digest = module._RETAINED_CAPACITY_PREIMPORT_BINDINGS[
        "integrated_controller"
    ]
    with (tmp_path / controller_reference).open("ab") as handle:
        handle.write(
            b'\n# unreviewed rebinding\nglobals()["_classify_targeted_serving_sanity"] = None\n'
        )
    with pytest.raises(ValueError, match="pre-import artifact hash mismatch"):
        module._require_retained_capacity_preimport_closure(config, tmp_path)

    verifier_source = ast.get_source_segment(
        COORDINATOR.read_text(),
        next(
            node
            for node in ast.parse(COORDINATOR.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == "_default_seal_verifier"
        ),
    )
    assert verifier_source is not None
    assert verifier_source.index("_require_retained_capacity_preimport_closure(") < (
        verifier_source.index("from gpu_reclamation_integrated_controller_v11 import")
    )


@pytest.mark.parametrize(
    "collision",
    ("provider-input", "conservative-failure", "status", "provider-cleanup"),
)
def test_attempt_i_rejects_terminal_output_collision_before_reservation(
    tmp_path: Path, collision: str
) -> None:
    module = _module()
    config = _config(module, ledger_sha256="5" * 64)
    provider_inputs = tmp_path / f"provider-cleanup-audit-inputs-{config.attempt_id}.json"
    failure_root = tmp_path / "failures"
    failure_output = failure_root / f"{config.attempt_id}-conservative-charge.json"
    status_output = tmp_path / "status-attempt-k.json"
    cleanup_output = tmp_path / "provider-cleanup-attempt-k.json"
    targets = {
        "provider-input": provider_inputs,
        "conservative-failure": failure_output,
        "status": status_output,
        "provider-cleanup": cleanup_output,
    }
    target = targets[collision]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("stale\n")

    with pytest.raises(FileExistsError, match="before reservation"):
        module._require_integrated_pre_reservation_outputs_fresh(
            config,
            provider_inputs_path=provider_inputs,
            failure_root=failure_root,
            terminal_output_paths=(status_output, cleanup_output),
        )


def test_attempt_i_collision_guard_precedes_prebuild_and_paid_reservation() -> None:
    source = COORDINATOR.read_text()
    main_source = source[source.index("def main()") :]

    assert main_source.index("_require_local_result_output_fresh(") < main_source.index(
        "_reserve_locked("
    )
    guard = main_source.index("_require_integrated_pre_reservation_outputs_fresh(")
    remote_guard = main_source.index("_require_remote_attempt_prefix_fresh(")
    assert guard < main_source.index("_prebuild_modal_image(")
    assert guard < main_source.index("_reserve_locked(")
    assert remote_guard < main_source.index("_prebuild_modal_image(")
    assert remote_guard < main_source.index("_reserve_locked(")
    assert main_source.index("_prebuild_modal_image(") < main_source.index("_reserve_locked(")
    assert main_source.count("_load_sealed_config(") >= 2


def test_attempt_k_rechecks_remote_and_local_freshness_after_prebuild(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    config_path = tmp_path / "config.json"
    config_path.write_text("{}\n")
    config = _config(module, ledger_sha256="5" * 64)
    seal = {"verified": True}
    calls = {"local": 0, "terminal": 0, "remote": 0, "reserve": 0, "invoke": 0}

    monkeypatch.setattr(module, "_ROOT", tmp_path)
    monkeypatch.setattr(module, "_EXPERIMENT_ROOT", tmp_path / "experiment")
    monkeypatch.setattr(module, "_LOCK", tmp_path / "gpu-hours.lock")
    monkeypatch.setattr(module, "_LEDGER", tmp_path / "gpu-hours.json")
    monkeypatch.setattr(module, "_LOCAL_RESULT_PARENT", tmp_path / "integrated/raw")
    monkeypatch.setattr(module, "_FAILURE_ROOT", tmp_path / "integrated/failures")
    monkeypatch.setattr(module, "_PROVIDER_INPUTS_PARENT", tmp_path / "integrated")
    monkeypatch.setattr(module, "_PREBUILD_ROOT", tmp_path / "prebuild")
    monkeypatch.setattr(
        module,
        "_load_sealed_config",
        lambda *args, **kwargs: (config, seal),
    )
    monkeypatch.setattr(module, "_require_launch_budget", lambda *args, **kwargs: 80.0)

    def local_fresh(_path: Path) -> None:
        calls["local"] += 1

    def terminal_fresh(*args: Any, **kwargs: Any) -> None:
        calls["terminal"] += 1

    def remote_fresh(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls["remote"] += 1
        if calls["remote"] == 2:
            raise FileExistsError(
                "fresh integrated attempt remote prefix exists before reservation: "
                f"{module.REMOTE_PREFIX}/{config.attempt_id}.staging/worker"
            )
        return {
            "schema_version": "sloforge.branchfabric.remote-prefix-freshness/v1",
            "status": "PASS",
            "volume_name": module.RESULTS_VOLUME,
            "listed_prefix": module.REMOTE_PREFIX,
            "attempt_id": config.attempt_id,
            "forbidden_paths": [],
            "observed_entries": [],
            "collision_count": 0,
        }

    def reserve(*args: Any, **kwargs: Any) -> None:
        calls["reserve"] += 1
        raise AssertionError("reservation must not be reached")

    def invoke(*args: Any, **kwargs: Any) -> None:
        calls["invoke"] += 1
        raise AssertionError("Modal invocation must not be reached")

    monkeypatch.setattr(module, "_require_local_result_output_fresh", local_fresh)
    monkeypatch.setattr(
        module,
        "_require_integrated_pre_reservation_outputs_fresh",
        terminal_fresh,
    )
    monkeypatch.setattr(module, "_require_remote_attempt_prefix_fresh", remote_fresh)
    monkeypatch.setattr(
        module,
        "_prebuild_modal_image",
        lambda **kwargs: {"schema_version": "prebuild-test/v1", "status": "PASS"},
    )
    monkeypatch.setattr(module, "_reserve_locked", reserve)
    monkeypatch.setattr(module, "_invoke_modal", invoke)
    monkeypatch.setattr(sys, "argv", [str(COORDINATOR), "--config-path", str(config_path)])

    with pytest.raises(FileExistsError, match=r"\.staging/worker"):
        module.main()

    assert calls == {"local": 2, "terminal": 2, "remote": 2, "reserve": 0, "invoke": 0}


def test_attempt_i_remote_prefix_freshness_is_fail_closed() -> None:
    module = _module()
    config = _config(module, ledger_sha256="5" * 64)
    prior = {
        "filename": f"{module.REMOTE_PREFIX}/exp004-v11-integrated-s41-h",
        "type": "dir",
        "created_modified": "2026-08-27 00:00:00",
        "size": "0 B",
    }

    def runner(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert command == [
            "modal",
            "volume",
            "ls",
            module.RESULTS_VOLUME,
            module.REMOTE_PREFIX,
            "--json",
        ]
        assert kwargs["timeout"] == 60.0
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps([prior]), stderr="")

    evidence = module._require_remote_attempt_prefix_fresh(config, runner=runner)
    assert evidence["status"] == "PASS"
    assert evidence["collision_count"] == 0
    assert evidence["observed_entries"] == [prior]


@pytest.mark.parametrize(
    "suffix",
    (
        "",
        "/child",
        ".staging",
        ".staging/child",
        ".inflight",
        ".inflight/child",
    ),
)
def test_attempt_i_remote_prefix_freshness_rejects_all_publication_collisions(
    suffix: str,
) -> None:
    module = _module()
    config = _config(module, ledger_sha256="5" * 64)
    collision = {
        "filename": f"{module.REMOTE_PREFIX}/{config.attempt_id}{suffix}",
        "type": "dir",
        "created_modified": "2026-08-27 00:00:00",
        "size": "0 B",
    }

    def runner(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps([collision]), stderr="")

    with pytest.raises(FileExistsError, match="remote prefix exists before reservation"):
        module._require_remote_attempt_prefix_fresh(config, runner=runner)


@pytest.mark.parametrize(
    "completed",
    (
        subprocess.CompletedProcess(["modal"], 1, stdout="", stderr="failed"),
        subprocess.CompletedProcess(["modal"], 0, stdout="not-json", stderr=""),
        subprocess.CompletedProcess(["modal"], 0, stdout='[{"filename":"x"}]', stderr=""),
        subprocess.CompletedProcess(
            ["modal"],
            0,
            stdout=('[{"Filename":"x","Type":"dir","Created/Modified":"now","Size":"0 B"}]'),
            stderr="",
        ),
    ),
)
def test_attempt_i_remote_prefix_freshness_rejects_unverifiable_inventory(
    completed: subprocess.CompletedProcess[str],
) -> None:
    module = _module()
    config = _config(module, ledger_sha256="5" * 64)

    def runner(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return completed

    with pytest.raises((RuntimeError, ValueError)):
        module._require_remote_attempt_prefix_fresh(config, runner=runner)


@pytest.mark.parametrize(
    "filenames",
    (
        ("../escape",),
        ("/experiment-004/v11/integrated/modal/absolute",),
        ("neighbor",),
        ("experiment-004/v11/integrated/modal/../escape",),
        ("experiment-004/v11/integrated/modal/trailing/",),
        (
            "experiment-004/v11/integrated/modal/existing",
            "experiment-004/v11/integrated/modal/existing/",
        ),
    ),
)
def test_attempt_i_remote_prefix_freshness_rejects_unsafe_or_duplicate_paths(
    filenames: tuple[str, ...],
) -> None:
    module = _module()
    config = _config(module, ledger_sha256="5" * 64)
    rows = [
        {
            "filename": filename,
            "type": "dir",
            "created_modified": "2026-08-27 00:00:00",
            "size": "0 B",
        }
        for filename in filenames
    ]

    def runner(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(rows), stderr="")

    with pytest.raises(ValueError, match="inventory paths are invalid"):
        module._require_remote_attempt_prefix_fresh(config, runner=runner)


@pytest.mark.parametrize(
    "suffix",
    (
        "-neighbor",
        ".staging-neighbor",
        ".staging.0123456789abcdef",
        ".inflight-neighbor",
        ".inflight.worker-0",
    ),
)
def test_attempt_i_remote_prefix_freshness_allows_prefix_neighbors(suffix: str) -> None:
    module = _module()
    config = _config(module, ledger_sha256="5" * 64)
    row = {
        "filename": f"{module.REMOTE_PREFIX}/{config.attempt_id}{suffix}",
        "type": "dir",
        "created_modified": "2026-08-27 00:00:00",
        "size": "0 B",
    }

    def runner(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps([row]), stderr="")

    evidence = module._require_remote_attempt_prefix_fresh(config, runner=runner)
    assert evidence["collision_count"] == 0


def test_attempt_i_dangling_local_result_symlink_is_a_pre_reservation_collision(
    tmp_path: Path,
) -> None:
    module = _module()
    local_root = tmp_path / "exp004-v11-integrated-s41-k"
    local_root.symlink_to(tmp_path / "missing-target", target_is_directory=True)

    with pytest.raises(FileExistsError, match="local immutable result"):
        module._require_local_result_output_fresh(local_root)


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
    assert source.count('        "run",\n') == 1
    assert "subprocess.run(" not in source
    tree = ast.parse(source)
    imported_roots = {
        alias.name.split(".", maxsplit=1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".", maxsplit=1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert imported_roots.isdisjoint({"torch", "vllm", "cupy", "pynvml"})
    assert "torch" not in source.lower()
    assert "gpu_count=2" in source
    assert "maximum_wall_seconds=wall_seconds" in source
    assert "TARGETED_FUNCTION_WALL_SECONDS" in source
    assert os.access(COORDINATOR, os.R_OK)
