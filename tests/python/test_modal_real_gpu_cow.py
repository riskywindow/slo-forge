from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "sloforge_modal_real_gpu_cow",
    _ROOT / "experiments/branchfabric/modal_real_gpu_cow.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODAL_EXPERIMENT = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODAL_EXPERIMENT
_SPEC.loader.exec_module(_MODAL_EXPERIMENT)

_LAUNCHER_SPEC = importlib.util.spec_from_file_location(
    "sloforge_branchfabric_modal_launcher",
    _ROOT / "tools/branchfabric-modal-launch.py",
)
assert _LAUNCHER_SPEC is not None and _LAUNCHER_SPEC.loader is not None
_MODAL_LAUNCHER = importlib.util.module_from_spec(_LAUNCHER_SPEC)
sys.modules[_LAUNCHER_SPEC.name] = _MODAL_LAUNCHER
_LAUNCHER_SPEC.loader.exec_module(_MODAL_LAUNCHER)

GPU_HOUR_LIMIT = _MODAL_EXPERIMENT.GPU_HOUR_LIMIT
GPU_SKU = _MODAL_EXPERIMENT.GPU_SKU
GPU_FUNCTION_STARTUP_TIMEOUT_SECONDS = _MODAL_EXPERIMENT.GPU_FUNCTION_STARTUP_TIMEOUT_SECONDS
GPU_FUNCTION_TIMEOUT_SECONDS = _MODAL_EXPERIMENT.GPU_FUNCTION_TIMEOUT_SECONDS
MODAL_A100_80GB_USD_PER_HOUR = _MODAL_EXPERIMENT.MODAL_A100_80GB_USD_PER_HOUR
MODAL_SDK_VERSION = _MODAL_EXPERIMENT.MODAL_SDK_VERSION
ModalBenchmarkConfig = _MODAL_EXPERIMENT.ModalBenchmarkConfig
require_cloud_budget = _MODAL_EXPERIMENT.require_cloud_budget
_assert_adaptive_gate = _MODAL_EXPERIMENT._assert_adaptive_gate
_attach_call_to_reservation = _MODAL_EXPERIMENT._attach_call_to_reservation
_attach_model_call_to_reservation = _MODAL_EXPERIMENT._attach_model_call_to_reservation
_charge_failed_gpu_reservation = _MODAL_EXPERIMENT._charge_failed_gpu_reservation
_cleanup_remote_staging = _MODAL_EXPERIMENT._cleanup_remote_staging
_fetch_call_logs = _MODAL_EXPERIMENT._fetch_call_logs
_recover_or_cancel_call = _MODAL_EXPERIMENT._recover_or_cancel_call
_reconcile_unattached_from_provider_audit = (
    _MODAL_EXPERIMENT._reconcile_unattached_from_provider_audit
)
_require_confirmed_cancellation = _MODAL_EXPERIMENT._require_confirmed_cancellation
_reconcile_model_reservation = _MODAL_EXPERIMENT._reconcile_model_reservation
_reserve_gpu_call = _MODAL_EXPERIMENT._reserve_gpu_call
_reserve_model_call = _MODAL_EXPERIMENT._reserve_model_call


def _smoke_payload() -> dict[str, object]:
    return {
        "attempt_id": "modal-smoke-test-v1",
        "fanout": 2,
        "prefix_length": 2048,
        "suffix_length": 16,
        "seed": 41,
        "baseline_mode": "shared_root",
    }


def test_modal_config_is_exact_hardware_and_runtime_scoped() -> None:
    config = ModalBenchmarkConfig.model_validate(_smoke_payload())
    assert GPU_SKU == "A100-80GB"
    assert MODAL_SDK_VERSION == "1.5.3"
    assert _MODAL_EXPERIMENT.PYDANTIC_VERSION == "2.13.4"
    assert config.runtime == "vllm"
    assert config.runtime_version == "0.23.0"
    assert config.model_revision == "a09a35458c702b33eeacc393d103063234e8bc28"
    assert _MODAL_EXPERIMENT._run_gpu_function is None
    assert not hasattr(_MODAL_EXPERIMENT.run_gpu_configuration, "spawn")
    source = (_ROOT / "experiments/branchfabric/modal_real_gpu_cow.py").read_text()
    assert '.apt_install("gcc",' in source
    assert '"CC": "/usr/bin/gcc"' in source
    assert '"VLLM_USE_FLASHINFER_SAMPLER": "0"' in source


def test_dependency_manifest_uses_injected_modal_module_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = SimpleNamespace(
        version=SimpleNamespace(cuda="13.0"),
        cuda=SimpleNamespace(is_available=lambda: True),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        _MODAL_EXPERIMENT.importlib.metadata,
        "version",
        lambda name: f"test-{name}",
    )
    versions = _MODAL_EXPERIMENT._dependency_versions()
    assert versions["modal_sdk"] == MODAL_SDK_VERSION
    assert "modal" not in versions["packages"]


def test_modal_gpu_entrypoint_delegates_without_importing_cuda_runtime() -> None:
    source = inspect.getsource(_MODAL_EXPERIMENT.run_gpu_configuration)
    assert "from gpu_cow_controller import run_controller" in source
    assert "import torch" not in source
    assert "import vllm" not in source
    assert "torch.cuda" not in source


def test_smoke_and_16k_shapes_fail_closed() -> None:
    with pytest.raises(ValidationError, match="smoke requires"):
        ModalBenchmarkConfig.model_validate({**_smoke_payload(), "baseline_mode": "independent"})
    with pytest.raises(ValidationError, match="16K trials require"):
        ModalBenchmarkConfig.model_validate(
            {
                **_smoke_payload(),
                "attempt_id": "modal-16k-invalid-v1",
                "prefix_length": 16384,
                "fanout": 2,
                "suffix_length": 256,
            }
        )


def test_two_k_smoke_runner_uses_supported_minimum_model_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    benchmark_root = _ROOT / "benchmarks/branchfabric"
    monkeypatch.syspath_prepend(str(benchmark_root))
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "BRANCHFABRIC_INPUTS.json").write_text(
        json.dumps(
            {
                "prefix_token_ids": [17] * 2048,
                "divergent_token_ids": list(range(100, 132)),
            }
        )
    )
    config = ModalBenchmarkConfig.model_validate(_smoke_payload())
    invocation, path, _ = _MODAL_EXPERIMENT._build_runner_input(
        config,
        gpu_uuid="GPU-test-sm80",
        snapshot=snapshot,
    )
    assert invocation.engine.max_model_len == 4096
    assert path.value == "shared_root"


def test_cloud_budget_is_mandatory_and_reserves_fifteen_percent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = tmp_path / "gpu-hours.json"
    ledger.write_text(json.dumps({"consumed_gpu_hours": 0.0}))
    monkeypatch.delenv("SLOFORGE_GPU_BUDGET_USD", raising=False)
    with pytest.raises(RuntimeError, match="is required"):
        require_cloud_budget(maximum_gpu_seconds=60, ledger_path=ledger)
    monkeypatch.setenv("SLOFORGE_GPU_BUDGET_USD", "1")
    with pytest.raises(RuntimeError, match="15% reserve"):
        require_cloud_budget(maximum_gpu_seconds=3600, ledger_path=ledger)
    monkeypatch.setenv("SLOFORGE_GPU_BUDGET_USD", "20")
    decision = require_cloud_budget(maximum_gpu_seconds=3600, ledger_path=ledger)
    assert decision.proposed_maximum_gpu_hours == 1.0
    assert decision.projected_gpu_cost_usd == MODAL_A100_80GB_USD_PER_HOUR


def test_cloud_budget_never_overrides_gpu_hour_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = tmp_path / "gpu-hours.json"
    ledger.write_text(json.dumps({"consumed_gpu_hours": GPU_HOUR_LIMIT - 0.25}))
    monkeypatch.setenv("SLOFORGE_GPU_BUDGET_USD", "1000")
    with pytest.raises(RuntimeError, match="four GPU-hour"):
        require_cloud_budget(maximum_gpu_seconds=1800, ledger_path=ledger)


def test_checked_in_modal_configs_validate() -> None:
    configs = sorted((_ROOT / "experiments/branchfabric/configs").glob("modal-*.json"))
    assert len(configs) >= 5
    attempts = {
        ModalBenchmarkConfig.model_validate_json(path.read_bytes()).attempt_id for path in configs
    }
    assert len(attempts) == len(configs)


def test_failed_gpu_call_is_conservatively_charged_and_clears_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_MODAL_EXPERIMENT, "LOCAL_EXPERIMENT_ROOT", tmp_path)
    monkeypatch.setenv("SLOFORGE_GPU_BUDGET_USD", "1000")
    ledger = tmp_path / "gpu-hours.json"
    ledger.write_text(
        json.dumps(
            {
                "consumed_gpu_hours": 0.0,
                "gpu_active_intervals": [],
                "consumed_cloud_cost_usd": 0.0,
                "cloud_compute_intervals": [],
                "modal_in_flight_reservations": [],
            }
        )
    )
    config = ModalBenchmarkConfig.model_validate(_smoke_payload())
    decision = require_cloud_budget(
        maximum_gpu_seconds=GPU_FUNCTION_TIMEOUT_SECONDS + GPU_FUNCTION_STARTUP_TIMEOUT_SECONDS,
        ledger_path=ledger,
    )
    reservation = _reserve_gpu_call(config, decision)
    _attach_call_to_reservation(reservation, "fc-test-failure")
    _charge_failed_gpu_reservation(
        reservation_id=reservation,
        call_id="fc-test-failure",
        attempt_id=config.attempt_id,
        reason="TimeoutError",
        remote_manifest_sha256="b" * 64,
    )
    payload = json.loads(ledger.read_text())
    assert payload["modal_in_flight_reservations"] == []
    assert payload["consumed_gpu_hours"] == 2.0
    assert payload["gpu_active_intervals"][0]["status"].endswith("TimeoutError")
    assert payload["gpu_active_intervals"][0]["reservation_id"] == reservation
    assert len(payload["gpu_active_intervals"][0]["authorization_config_sha256"]) == 64
    assert payload["gpu_active_intervals"][0]["reserved_maximum_gpu_hours"] == 2.0
    assert payload["gpu_active_intervals"][0]["remote_manifest_sha256"] == "b" * 64
    assert payload["consumed_cloud_cost_usd"] > 0


def test_adaptive_gate_requires_smoke_then_paired_lower_fanout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_MODAL_EXPERIMENT, "LOCAL_EXPERIMENT_ROOT", tmp_path)
    fanout_one = ModalBenchmarkConfig.model_validate(
        {
            **_smoke_payload(),
            "attempt_id": "modal-16k-f1-test-v1",
            "prefix_length": 16384,
            "suffix_length": 256,
            "fanout": 1,
            "baseline_mode": "shared_root",
        }
    )
    with pytest.raises(RuntimeError, match="semantic smoke"):
        _assert_adaptive_gate(fanout_one)

    def write_completion(attempt: str, summary: dict[str, object]) -> None:
        root = tmp_path / "modal" / attempt
        (root / "lifecycle").mkdir(parents=True)
        (root / "function-completion.json").write_text(
            json.dumps({"status": "succeeded", "summary": summary})
        )
        (root / "lifecycle/cleanup-gate.json").write_text(
            json.dumps(
                {
                    "cleanup_gate": "PASS",
                    "GPU_RUNTIME_LIFECYCLE_CLEAN": True,
                    "forced_gpu_process_kill_required": False,
                }
            )
        )
        (root / "lifecycle/forbidden-import-audit.json").write_text(
            json.dumps({"cuda_clean": True})
        )
        (root / "lifecycle/child-manifest.json").write_text(
            json.dumps(
                {
                    "status": "succeeded",
                    "final_runtime_assigned_kv_bytes": 0,
                    "semantic_invariants": {"test_fixture": True},
                }
            )
        )

    write_completion(
        "modal-smoke-gate-v1",
        {"path": "shared_root", "fanout": 2, "prefix_tokens": 2048, "seed": 41},
    )
    _assert_adaptive_gate(fanout_one)
    fanout_eight = fanout_one.model_copy(update={"attempt_id": "modal-16k-f8-test-v1", "fanout": 8})
    with pytest.raises(RuntimeError, match="fanout-1 paired"):
        _assert_adaptive_gate(fanout_eight)
    for path in ("independent_prefill", "shared_root"):
        write_completion(
            f"modal-f1-{path}-v1",
            {"path": path, "fanout": 1, "prefix_tokens": 16384, "seed": 41},
        )
    _assert_adaptive_gate(fanout_eight)


def test_model_preparation_is_reserved_before_call_and_reconciled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_MODAL_EXPERIMENT, "LOCAL_EXPERIMENT_ROOT", tmp_path)
    monkeypatch.setenv("SLOFORGE_GPU_BUDGET_USD", "1000")
    ledger = tmp_path / "gpu-hours.json"
    ledger.write_text(
        json.dumps(
            {
                "consumed_gpu_hours": 0.0,
                "gpu_active_intervals": [],
                "consumed_cloud_cost_usd": 0.0,
                "cloud_compute_intervals": [],
                "modal_in_flight_reservations": [],
                "modal_in_flight_model_reservations": [],
            }
        )
    )
    config = ModalBenchmarkConfig.model_validate(_smoke_payload())
    decision = require_cloud_budget(
        maximum_gpu_seconds=0,
        maximum_model_prep_seconds=8100,
        ledger_path=ledger,
    )
    reservation = _reserve_model_call(config, decision)
    _attach_model_call_to_reservation(reservation, "fc-model-test")
    _reconcile_model_reservation(
        reservation_id=reservation,
        call_id="fc-model-test",
        attempt_id=config.attempt_id,
        elapsed_seconds=7.0,
        status="prepared",
        maximum_charge=False,
    )
    payload = json.loads(ledger.read_text())
    assert payload["modal_in_flight_model_reservations"] == []
    assert payload["cloud_compute_intervals"][0]["kind"] == "model_prepare"
    assert payload["cloud_compute_intervals"][0]["reservation_id"] == reservation
    assert len(payload["cloud_compute_intervals"][0]["authorization_config_sha256"]) == 64
    assert payload["consumed_cloud_cost_usd"] > 0


def test_budget_first_launcher_never_starts_modal_when_guard_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def forbidden_run(*args: object, **kwargs: object) -> None:
        nonlocal called
        called = True

    monkeypatch.delenv("SLOFORGE_GPU_BUDGET_USD", raising=False)
    monkeypatch.setattr(_MODAL_LAUNCHER.subprocess, "run", forbidden_run)
    with pytest.raises(RuntimeError, match="required before invoking Modal"):
        _MODAL_LAUNCHER.main(
            [
                "--action",
                "run",
                "--config-path",
                str(_ROOT / "experiments/branchfabric/configs/modal-smoke-shared-s41-v1.json"),
            ]
        )
    assert called is False


def test_budget_first_launcher_projects_full_cost_before_modal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = tmp_path / "gpu-hours.json"
    ledger.write_text(
        json.dumps(
            {
                "consumed_gpu_hours": 0.0,
                "consumed_cloud_cost_usd": 0.0,
                "modal_in_flight_reservations": [],
                "modal_in_flight_model_reservations": [],
            }
        )
    )
    monkeypatch.setattr(_MODAL_LAUNCHER, "_LEDGER", ledger)
    called = False

    def fake_run(*args: object, **kwargs: object) -> SimpleNamespace:
        nonlocal called
        called = True
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        assert len(environment["SLOFORGE_MODAL_PREFLIGHT_TOKEN"]) == 64
        command = args[0]
        assert isinstance(command, list)
        assert "--detach" in command
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(_MODAL_LAUNCHER.subprocess, "run", fake_run)
    arguments = [
        "--action",
        "run",
        "--config-path",
        str(_ROOT / "experiments/branchfabric/configs/modal-smoke-shared-s41-v1.json"),
    ]
    monkeypatch.setenv("SLOFORGE_GPU_BUDGET_USD", "1")
    with pytest.raises(RuntimeError, match="15% reserve"):
        _MODAL_LAUNCHER.main(arguments)
    assert called is False
    monkeypatch.setenv("SLOFORGE_GPU_BUDGET_USD", "1000")
    assert _MODAL_LAUNCHER.main(arguments) == 0
    assert called is True


def test_budget_first_launcher_counts_in_flight_gpu_hours(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = tmp_path / "gpu-hours.json"
    ledger.write_text(
        json.dumps(
            {
                "consumed_gpu_hours": 1.0,
                "consumed_cloud_cost_usd": 0.0,
                "modal_in_flight_reservations": [
                    {"maximum_gpu_hours": 2.0, "maximum_cloud_cost_usd": 10.0}
                ],
                "modal_in_flight_model_reservations": [],
            }
        )
    )
    monkeypatch.setattr(_MODAL_LAUNCHER, "_LEDGER", ledger)
    monkeypatch.setenv("SLOFORGE_GPU_BUDGET_USD", "1000")
    monkeypatch.setattr(
        _MODAL_LAUNCHER.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("Modal must not start"),
    )
    with pytest.raises(RuntimeError, match="four GPU-hour"):
        _MODAL_LAUNCHER.main(
            [
                "--action",
                "run",
                "--config-path",
                str(_ROOT / "experiments/branchfabric/configs/modal-smoke-shared-s41-v1.json"),
            ]
        )


def test_remote_staging_cleanup_refuses_active_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_MODAL_EXPERIMENT, "LOCAL_EXPERIMENT_ROOT", tmp_path)
    config = ModalBenchmarkConfig.model_validate(_smoke_payload())
    (tmp_path / "gpu-hours.json").write_text(
        json.dumps(
            {
                "modal_in_flight_reservations": [
                    {"attempt_id": config.attempt_id, "reservation_id": "modal-reservation-x"}
                ],
                "modal_in_flight_model_reservations": [],
            }
        )
    )
    with pytest.raises(RuntimeError, match="in-flight reservation"):
        _cleanup_remote_staging(config)


def test_accounted_failed_result_materialization_is_not_reported_successful(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_MODAL_EXPERIMENT, "LOCAL_EXPERIMENT_ROOT", tmp_path)
    config = ModalBenchmarkConfig.model_validate(_smoke_payload())
    (tmp_path / "gpu-hours.json").write_text(
        json.dumps(
            {
                "gpu_active_intervals": [
                    {
                        "attempt_id": config.attempt_id,
                        "function_call_id": "fc-failed",
                        "remote_manifest_sha256": "a" * 64,
                        "status": "failed",
                    }
                ],
                "modal_in_flight_reservations": [],
                "modal_in_flight_model_reservations": [],
            }
        )
    )
    monkeypatch.setattr(
        _MODAL_EXPERIMENT,
        "_materialize_result",
        lambda *args, **kwargs: {"verified_at_utc": datetime.now(UTC).isoformat()},
    )
    result = _recover_or_cancel_call(config, cancel=False)
    assert result["status"] == "accounted_failed_gpu_result_materialized"
    assert result["experiment_succeeded"] is False


def test_log_retrieval_uses_unique_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_MODAL_EXPERIMENT, "LOCAL_EXPERIMENT_ROOT", tmp_path)

    class Logs:
        def fetch(self, *, source: str) -> list[object]:
            return []

    call = SimpleNamespace(object_id="fc-log-test", logs=Logs())
    _fetch_call_logs(call, "modal-smoke-test-v1", "gpu")
    _fetch_call_logs(call, "modal-smoke-test-v1", "gpu")
    roots = list((tmp_path / "modal/logs/modal-smoke-test-v1").iterdir())
    assert len(roots) == 2
    assert roots[0].name != roots[1].name


def test_unconfirmed_cancellation_fails_before_reservation_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _MODAL_EXPERIMENT,
        "_cancel_call",
        lambda call: "ConnectionError: provider unreachable",
    )
    source = TimeoutError("client wait expired")
    with pytest.raises(RuntimeError, match="reservation retained") as caught:
        _require_confirmed_cancellation(
            object(), context="Modal GPU call failed", source_error=source
        )
    assert caught.value.__cause__ is source


def test_recovered_call_cancellation_uses_provider_supported_mode() -> None:
    observed: list[bool] = []

    class Call:
        def cancel(self, *, terminate_containers: bool) -> None:
            observed.append(terminate_containers)

    assert _MODAL_EXPERIMENT._cancel_call(Call()) is None
    assert observed == [False]


def test_provider_terminal_audit_reconciles_only_exact_unattached_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_MODAL_EXPERIMENT, "LOCAL_EXPERIMENT_ROOT", tmp_path)
    monkeypatch.setenv("SLOFORGE_GPU_BUDGET_USD", "1000")
    ledger = tmp_path / "gpu-hours.json"
    ledger.write_text(
        json.dumps(
            {
                "consumed_gpu_hours": 0.0,
                "gpu_active_intervals": [],
                "consumed_cloud_cost_usd": 0.0,
                "cloud_compute_intervals": [],
                "modal_in_flight_reservations": [],
                "modal_in_flight_model_reservations": [],
            }
        )
    )
    config = ModalBenchmarkConfig.model_validate(_smoke_payload())
    decision = require_cloud_budget(
        maximum_gpu_seconds=GPU_FUNCTION_TIMEOUT_SECONDS + GPU_FUNCTION_STARTUP_TIMEOUT_SECONDS,
        ledger_path=ledger,
    )
    reservation = _reserve_gpu_call(config, decision)
    audit_root = tmp_path / "modal/provider-audits"
    audit_root.mkdir(parents=True)
    audit = audit_root / "provider-terminal.json"
    audit.write_text(
        json.dumps(
            {
                "schema_version": "sloforge.branchfabric.modal-provider-terminal-audit/v1",
                "attempt_id": config.attempt_id,
                "reservation_id": reservation,
                "kind": "gpu",
                "provider_terminal_state": "not_found",
                "provider_live_or_pending": False,
                "audited_at_utc": datetime.now(UTC).isoformat(),
                "evidence_reference": "Modal dashboard call search exported locally",
                "auditor": "test-operator",
            }
        )
    )
    result = _reconcile_unattached_from_provider_audit(config, provider_audit_path=str(audit))
    assert result["status"] == "unattached_reservation_reconciled_from_provider_terminal_audit"
    payload = json.loads(ledger.read_text())
    assert payload["modal_in_flight_reservations"] == []
    assert (
        payload["gpu_active_intervals"][0]["provider_terminal_audit_sha256"]
        == result["provider_terminal_audit_sha256"]
    )
    assert payload["gpu_active_intervals"][0]["gpu_hours"] == 2.0


def test_unattached_model_reservation_is_not_cleared_by_global_model_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_MODAL_EXPERIMENT, "LOCAL_EXPERIMENT_ROOT", tmp_path)
    config = ModalBenchmarkConfig.model_validate(_smoke_payload())
    (tmp_path / "gpu-hours.json").write_text(
        json.dumps(
            {
                "modal_in_flight_reservations": [],
                "modal_in_flight_model_reservations": [
                    {
                        "attempt_id": config.attempt_id,
                        "reservation_id": "modal-model-reservation-0123456789abcdef",
                        "function_call_id": None,
                        "created_at_utc": datetime.now(UTC).isoformat(),
                    }
                ],
            }
        )
    )
    result = _recover_or_cancel_call(config, cancel=False)
    assert result["status"] == "ambiguous_spawn_reservation_retained_pending_provider_audit"
    assert result["reservation_retained"] is True
    ledger = json.loads((tmp_path / "gpu-hours.json").read_text())
    assert len(ledger["modal_in_flight_model_reservations"]) == 1
