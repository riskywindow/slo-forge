from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from sloforge.helix.characterization.gpu_capacity_calibration import (
    NS_PER_SECOND,
    CapacityProbeRaw,
    CapacityRequestObservation,
    ProbeTopology,
    RequestTerminalState,
    build_probe_plan,
    choose_spike,
    evaluate_probe,
)
from sloforge.helix.characterization.gpu_reclamation_methodology import (
    Experiment004GpuHourLedger,
    GpuInvocationReservation,
)

ROOT = Path(__file__).parents[2]
CONTROLLER = ROOT / "experiments/branchfabric/gpu_reclamation_controller.py"
MODAL_APP = ROOT / "experiments/branchfabric/modal_gpu_reclamation.py"
LAUNCHER = ROOT / "tools/branchfabric-experiment-004-launch.py"
PILOT_CONFIG = (
    ROOT / "artifacts/branchfabric/gpu-validation/experiment-004/raw/pilot-naive-config.json"
)


def _load(path: Path, name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _pilot_payload() -> dict[str, object]:
    payload = json.loads(PILOT_CONFIG.read_text())
    assert isinstance(payload, dict)
    payload["maximum_wall_seconds"] = 150
    payload["initialization_timeout_seconds"] = 120
    payload["cleanup_timeout_seconds"] = 30
    return payload


def _v10_payload() -> dict[str, object]:
    return {
        **_pilot_payload(),
        "serving_methodology": "v10-global-capacity",
        "serving_spike_request_rate_per_second": 15.0,
        "gpu0_overload_probe_seconds": 3.0,
        "serving_recovery_evaluation_seconds": 1.0,
        "serving_recovery_queue_threshold": 4,
        "serving_maximum_pending_requests": 1024,
        "serving_restore_handoff_lead_requests": 4,
        "serving_slo_stability_window_seconds": 5.0,
        "calibration_attempt_id": "exp004-capacity-fixture",
        "lambda_1_rps": 12.0,
        "lambda_2_rps": 18.0,
        "lambda_spike_rps": 15.0,
        "calibration_selected_load_artifact": (
            "artifacts/branchfabric/gpu-validation/experiment-004/calibration/selected-load.json"
        ),
        "calibration_selected_load_sha256": "1" * 64,
        "agent9_approval_artifact": (
            "artifacts/branchfabric/gpu-validation/experiment-004/calibration/reviews/"
            "agent9-selected-load-approval.json"
        ),
        "agent9_approval_sha256": "2" * 64,
    }


def _capacity_raw(
    *, topology: ProbeTopology, rate: float, capacity: float, probe_id: str
) -> CapacityProbeRaw:
    plan = build_probe_plan(
        probe_id=probe_id,
        seed=41,
        topology=topology,
        configured_rate_rps=rate,
        start_ns=NS_PER_SECOND,
    )
    per_gpu_capacity = capacity if topology == ProbeTopology.GPU0_ONLY else capacity / 2
    service_ns = round(NS_PER_SECOND / per_gpu_capacity)
    next_available = {"gpu0": 0, "gpu1": 0}
    observations = []
    for arrival in plan.arrivals:
        completed = max(
            arrival.scheduled_arrival_ns + service_ns,
            next_available[arrival.assigned_device] + service_ns,
        )
        next_available[arrival.assigned_device] = completed
        observations.append(
            CapacityRequestObservation(
                request_id=arrival.request_id,
                global_sequence=arrival.global_sequence,
                device=arrival.assigned_device,
                scheduled_arrival_ns=arrival.scheduled_arrival_ns,
                offered_ns=arrival.scheduled_arrival_ns,
                enqueued_ns=arrival.scheduled_arrival_ns,
                admitted_ns=arrival.scheduled_arrival_ns,
                first_token_ns=completed - service_ns // 2,
                completed_ns=completed,
                terminated_ns=completed,
                requested_output_tokens=64,
                emitted_tokens=64,
                terminal_state=RequestTerminalState.COMPLETED,
            )
        )
    return CapacityProbeRaw(
        plan=plan,
        observations=tuple(observations),
        tail_drain_end_ns=plan.measurement_end_ns + 5 * NS_PER_SECOND,
        probe_end_ns=plan.measurement_end_ns + 5 * NS_PER_SECOND,
        tail_drain_seconds=5.0,
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def _artifact_ref(path: Path) -> dict[str, object]:
    return {
        "artifact_reference": str(path),
        "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sample_selector": "$",
    }


def test_controller_import_is_cpu_only_in_a_fresh_interpreter() -> None:
    code = f"""
import importlib.util, json, sys
path = {str(CONTROLLER)!r}
before = set(sys.modules)
spec = importlib.util.spec_from_file_location('exp004_controller_fresh', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
prefixes = module.FORBIDDEN_GPU_MODULE_PREFIXES
loaded = sorted(name for name in set(sys.modules) - before if any(name == prefix or name.startswith(prefix + '.') for prefix in prefixes))
print(json.dumps({{'loaded': loaded, 'audit': module.cuda_clean_import_audit('fixture')}}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    evidence = json.loads(completed.stdout)
    assert evidence["loaded"] == []
    assert evidence["audit"]["cuda_clean"] is True


def test_inventory_requires_exactly_two_distinct_a100_80gb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _load(CONTROLLER, "exp004_controller_inventory")

    def completed(stdout: str) -> SimpleNamespace:
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    valid = (
        "0, GPU-first-1, NVIDIA A100 80GB PCIe, 580.95.05, 81920, 0, 0\n"
        "1, GPU-second-2, NVIDIA A100-SXM4-80GB, 580.95.05, 81920, 0, 0\n"
    )
    monkeypatch.setattr(controller.subprocess, "run", lambda *_args, **_kwargs: completed(valid))
    rows = controller._inventory()
    assert tuple(row["uuid"] for row in rows) == ("GPU-first-1", "GPU-second-2")

    one = "0, GPU-first-1, NVIDIA A100 80GB PCIe, 580.95.05, 81920, 0, 0\n"
    monkeypatch.setattr(controller.subprocess, "run", lambda *_args, **_kwargs: completed(one))
    with pytest.raises(RuntimeError, match="exactly two visible GPUs"):
        controller._inventory()

    substituted = valid.replace("NVIDIA A100-SXM4-80GB", "NVIDIA H100 80GB HBM3")
    monkeypatch.setattr(
        controller.subprocess, "run", lambda *_args, **_kwargs: completed(substituted)
    )
    with pytest.raises(RuntimeError, match="requires A100-80GB"):
        controller._inventory()


def test_exclusive_postflight_rejects_any_compute_process_not_only_worker_leaders() -> None:
    controller = _load(CONTROLLER, "exp004_controller_postflight")
    escaped = (
        {
            "gpu_uuid": "GPU-first-1",
            "pid": 987654,
            "process_name": "escaped-worker-descendant",
            "used_gpu_memory_mib": 128,
        },
    )
    with pytest.raises(RuntimeError, match="worker postflight cleanup"):
        controller._require_no_compute_processes(escaped, phase="worker postflight cleanup")


class _FakeVolume:
    def with_mount_options(self, *, read_only: bool) -> tuple[str, bool]:
        return ("mount", read_only)

    def commit(self) -> None:
        return None

    def reload(self) -> None:
        return None


class _FakeVolumeType:
    @staticmethod
    def from_name(_name: str, *, create_if_missing: bool) -> _FakeVolume:
        assert create_if_missing is False
        return _FakeVolume()


class _FakeApp:
    def __init__(self, calls: list[dict[str, object]]) -> None:
        self.calls = calls

    def function(self, **kwargs: object):  # type: ignore[no-untyped-def]
        self.calls.append(kwargs)

        def decorate(function):  # type: ignore[no-untyped-def]
            return function

        return decorate

    def local_entrypoint(self):  # type: ignore[no-untyped-def]
        return lambda function: function


def test_modal_graph_requests_exact_two_a100s_and_strict_bounded_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    function_calls: list[dict[str, object]] = []
    fake_modal = ModuleType("modal")
    fake_modal.__version__ = "1.5.3"  # type: ignore[attr-defined]
    fake_modal.Volume = _FakeVolumeType  # type: ignore[attr-defined]
    fake_modal.App = lambda *_args, **_kwargs: _FakeApp(function_calls)  # type: ignore[attr-defined]
    fake_modal.current_function_call_id = lambda: "fixture-call"  # type: ignore[attr-defined]
    fake_image = ModuleType("modal_real_gpu_cow")
    fake_image.gpu_image = object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    monkeypatch.setitem(sys.modules, "modal_real_gpu_cow", fake_image)
    monkeypatch.setenv("SLOFORGE_MODAL_PREFLIGHT_TOKEN", "a" * 64)

    module = _load(MODAL_APP, "exp004_modal_exact_gpu")
    assert module.GPU_REQUEST == "A100-80GB:2"
    assert module.GPU_COUNT == 2
    assert len(function_calls) == 1
    assert function_calls[0]["gpu"] == "A100-80GB:2"
    assert function_calls[0]["max_containers"] == 1
    assert function_calls[0]["single_use_containers"] is True

    config = module.Experiment004PilotConfig.model_validate(_pilot_payload(), strict=True)
    assert config.fanout == 8
    assert config.serving_slo_maximum_ttft_seconds == 2.0
    assert config.serving_slo_maximum_inter_token_latency_seconds == 0.5
    assert config.serving_slo_stability_window_seconds == 1.0
    for tracing_level in ("disabled", "minimal", "full"):
        assert (
            module.Experiment004PilotConfig.model_validate(
                {**_pilot_payload(), "tracing_level": tracing_level}, strict=True
            ).tracing_level
            == tracing_level
        )
    with pytest.raises(ValueError):
        module.Experiment004PilotConfig.model_validate(
            {**_pilot_payload(), "fanout": 16}, strict=True
        )
    with pytest.raises(ValueError, match="stability window"):
        module.Experiment004PilotConfig.model_validate(
            {
                **_pilot_payload(),
                "temporary_serving_seconds": 1.0,
                "serving_slo_stability_window_seconds": 2.0,
            },
            strict=True,
        )
    v10 = module.Experiment004PilotConfig.model_validate(_v10_payload(), strict=True)
    assert v10.serving_methodology == "v10-global-capacity"
    assert v10.serving_spike_request_rate_per_second == 15.0
    with pytest.raises(ValueError, match="five seconds"):
        module.Experiment004PilotConfig.model_validate(
            {**_v10_payload(), "serving_slo_stability_window_seconds": 4.0}, strict=True
        )

    config_digest = hashlib.sha256(module._canonical_bytes(config)).hexdigest()
    (tmp_path / "gpu-hours.json").write_text(
        json.dumps(
            {
                "schema_version": "sloforge.branchfabric.experiment-004-gpu-hours/v1",
                "historical_through_experiment_003_gpu_hours": 1.669146,
                "target_additional_gpu_seconds": 3600.0,
                "hard_additional_gpu_seconds": 7200.0,
                "consumed_additional_gpu_seconds": 0.0,
                "intervals": [],
                "reservations": [
                    {
                        "reservation_id": "exp004-fixture-reservation",
                        "invocation_id": config.attempt_id,
                        "requested_gpu": "A100-80GB",
                        "gpu_count": 2,
                        "maximum_wall_seconds": 340.0,
                        "config_sha256": config_digest,
                    }
                ],
            }
        )
    )
    monkeypatch.setattr(module, "LOCAL_EXPERIMENT_ROOT", tmp_path)
    module._validate_local_reservation(config, "exp004-fixture-reservation")
    with pytest.raises(RuntimeError, match="absent or duplicated"):
        module._validate_local_reservation(config, "nonexistent-reservation")


def test_launcher_validates_complete_config_and_declared_slo_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _load(LAUNCHER, "exp004_launcher_config")
    validated_bindings: list[dict[str, object]] = []
    monkeypatch.setattr(
        launcher,
        "_validate_v10_calibration_binding",
        lambda payload: validated_bindings.append(payload),
    )
    valid = tmp_path / "valid.json"
    valid.write_text(json.dumps(_pilot_payload()))
    assert launcher._validate_config(valid) == _pilot_payload()

    missing_slo = dict(_pilot_payload())
    missing_slo.pop("serving_slo_maximum_ttft_seconds")
    invalid = tmp_path / "missing.json"
    invalid.write_text(json.dumps(missing_slo))
    with pytest.raises(ValueError, match="fields differ from the exact contract"):
        launcher._validate_config(invalid)

    invalid.write_text(
        json.dumps({**_pilot_payload(), "serving_slo_stability_window_seconds": 4.0})
    )
    with pytest.raises(ValueError, match="stability window exceeds"):
        launcher._validate_config(invalid)

    v10 = tmp_path / "v10.json"
    v10.write_text(json.dumps(_v10_payload()))
    assert launcher._validate_config(v10) == _v10_payload()
    assert validated_bindings == [_v10_payload()]
    invalid.write_text(json.dumps({**_v10_payload(), "serving_spike_request_rate_per_second": 8.0}))
    with pytest.raises(ValueError, match="spike must exceed"):
        launcher._validate_config(invalid)


def test_v10_launch_binding_requires_hashed_selection_agent9_and_raw_probes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _load(LAUNCHER, "exp004_launcher_calibration_binding")
    experiment_root = tmp_path / "artifacts/branchfabric/gpu-validation/experiment-004"
    attempt_id = "exp004-capacity-fixture"
    raws = (
        _capacity_raw(
            topology=ProbeTopology.GPU0_ONLY,
            rate=20.0,
            capacity=20.0,
            probe_id="capacity-one-20",
        ),
        _capacity_raw(
            topology=ProbeTopology.GPU0_ONLY,
            rate=25.0,
            capacity=20.0,
            probe_id="capacity-one-25",
        ),
        _capacity_raw(
            topology=ProbeTopology.TWO_GPU_ROUND_ROBIN,
            rate=30.0,
            capacity=40.0,
            probe_id="capacity-two-30",
        ),
    )
    results = tuple(evaluate_probe(raw) for raw in raws)
    selection = choose_spike(results)
    assert selection is not None
    entries_by_topology: dict[str, list[dict[str, object]]] = {
        "one-gpu": [],
        "two-gpu": [],
    }
    raw_root = experiment_root / f"calibration/raw/modal/{attempt_id}/calibration"
    for raw, result in zip(raws, results, strict=True):
        topology_dir = "one-gpu" if raw.plan.topology == ProbeTopology.GPU0_ONLY else "two-gpu"
        probe_root = raw_root / topology_dir / raw.plan.probe_id
        plan_path = probe_root / "plan.json"
        raw_path = probe_root / "raw.json"
        result_path = probe_root / "result.json"
        _write_json(plan_path, raw.plan)
        _write_json(raw_path, raw)
        _write_json(result_path, result)
        entries_by_topology[topology_dir].append(
            {
                "probe_id": raw.plan.probe_id,
                "plan": _artifact_ref(plan_path),
                "raw": _artifact_ref(raw_path),
                "result": _artifact_ref(result_path),
            }
        )
    for topology_dir, entries in entries_by_topology.items():
        _write_json(
            experiment_root / f"calibration/{topology_dir}/{attempt_id}.index.json",
            {
                "schema_version": "sloforge.branchfabric.capacity-artifact-index/v1",
                "attempt_id": attempt_id,
                "topology": topology_dir,
                "entries": entries,
            },
        )
    raw_selection_path = raw_root / "selected-load.json"
    _write_json(raw_selection_path, selection)
    selection_path = experiment_root / "calibration/selected-load.json"
    _write_json(
        selection_path,
        {**selection.model_dump(mode="json"), "raw_provenance": _artifact_ref(raw_selection_path)},
    )
    approval_path = experiment_root / "calibration/reviews/agent9-selected-load-approval.json"
    approval = {
        "schema_version": "sloforge.branchfabric.capacity-agent9-approval/v1",
        "reviewer": "agent9",
        "verdict": "approved",
        "calibration_attempt_id": attempt_id,
        "selected_load_artifact": launcher._SELECTED_LOAD_REFERENCE,
        "selected_load_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        "lambda_1_rps": selection.lambda_1_rps,
        "lambda_2_rps": selection.lambda_2_rps,
        "lambda_spike_rps": selection.lambda_spike_rps,
        "one_gpu_unsustainable_probe_id": selection.one_gpu_unsustainable_probe_id,
        "one_gpu_sustainable_probe_id": selection.one_gpu_sustainable_probe_id,
        "two_gpu_sustainable_probe_id": selection.two_gpu_sustainable_probe_id,
    }
    _write_json(approval_path, approval)
    payload = {
        **_v10_payload(),
        "calibration_attempt_id": attempt_id,
        "lambda_1_rps": selection.lambda_1_rps,
        "lambda_2_rps": selection.lambda_2_rps,
        "lambda_spike_rps": selection.lambda_spike_rps,
        "serving_spike_request_rate_per_second": selection.lambda_spike_rps,
        "calibration_selected_load_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        "agent9_approval_sha256": hashlib.sha256(approval_path.read_bytes()).hexdigest(),
    }
    monkeypatch.setattr(launcher, "_ROOT", tmp_path)
    monkeypatch.setattr(launcher, "_EXPERIMENT_ROOT", experiment_root)
    launcher._validate_v10_calibration_binding(payload)

    rejected = dict(payload)
    rejected["lambda_spike_rps"] = 24.0
    with pytest.raises(ValueError, match="differ from selected-load"):
        launcher._validate_v10_calibration_binding(rejected)
    approval["verdict"] = "rejected"
    _write_json(approval_path, approval)
    rejected = dict(payload)
    rejected["agent9_approval_sha256"] = hashlib.sha256(approval_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="does not exactly approve"):
        launcher._validate_v10_calibration_binding(rejected)


def test_v10_config_cannot_validate_before_selection_and_approval_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _load(LAUNCHER, "exp004_launcher_missing_calibration_gate")
    experiment_root = tmp_path / "artifacts/branchfabric/gpu-validation/experiment-004"
    experiment_root.mkdir(parents=True)
    config_path = tmp_path / "v10.json"
    config_path.write_text(json.dumps(_v10_payload()))
    monkeypatch.setattr(launcher, "_ROOT", tmp_path)
    monkeypatch.setattr(launcher, "_EXPERIMENT_ROOT", experiment_root)

    with pytest.raises(FileNotFoundError):
        launcher._validate_config(config_path)


def test_unset_budget_fails_before_lock_or_ledger_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _load(LAUNCHER, "exp004_launcher_no_budget")
    ledger = tmp_path / "gpu-hours.json"
    lock = tmp_path / "gpu-hours.lock"
    monkeypatch.setattr(launcher, "_LEDGER", ledger)
    monkeypatch.setattr(launcher, "_LOCK", lock)
    monkeypatch.delenv("SLOFORGE_GPU_BUDGET_USD", raising=False)
    monkeypatch.setattr(sys, "argv", [str(LAUNCHER), "--config-path", str(PILOT_CONFIG)])
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("Modal subprocess must not be invoked"),
    )

    with pytest.raises(RuntimeError, match="SLOFORGE_GPU_BUDGET_USD"):
        launcher.main()
    assert not ledger.exists()
    assert not lock.exists()


def test_launcher_extracts_one_result_amid_modal_cli_logs() -> None:
    launcher = _load(LAUNCHER, "exp004_launcher_output")
    payload = {"result": {"status": "succeeded"}, "materialized": {"local_path": "/tmp/x"}}
    stdout = "Initialized\n" + json.dumps(payload) + "\nApp completed\n"
    assert launcher._parse_modal_result(stdout) == payload
    with pytest.raises(ValueError, match="contained 0"):
        launcher._parse_modal_result("only logs\n")
    with pytest.raises(ValueError, match="contained 2"):
        launcher._parse_modal_result(json.dumps(payload) + "\n" + json.dumps(payload))


def test_pilot_settlement_validates_exact_remote_download_and_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _load(LAUNCHER, "exp004_launcher_settlement_envelope")
    experiment_root = tmp_path / "experiment-004"
    ledger_path = experiment_root / "gpu-hours.json"
    config = _v10_payload()
    config_digest = hashlib.sha256(launcher._canonical_bytes(config)).hexdigest()
    reservation_id = "exp004-0123456789abcdef0123"
    ledger = Experiment004GpuHourLedger(
        consumed_additional_gpu_seconds=0.0,
        reservations=(
            GpuInvocationReservation(
                reservation_id=reservation_id,
                invocation_id=str(config["attempt_id"]),
                maximum_wall_seconds=340.0,
                config_sha256=config_digest,
            ),
        ),
    )
    _write_json(ledger_path, ledger)
    monkeypatch.setattr(launcher, "_LEDGER", ledger_path)
    monkeypatch.setattr(launcher, "_EXPERIMENT_ROOT", experiment_root)
    inventory = [
        {
            "uuid": "GPU-a100-fixture-0",
            "name": "NVIDIA A100-SXM4-80GB",
            "memory_total_mib": 81_920,
        },
        {
            "uuid": "GPU-a100-fixture-1",
            "name": "NVIDIA A100 80GB PCIe",
            "memory_total_mib": 81_920,
        },
    ]
    completion = {
        "schema_version": "sloforge.branchfabric.experiment-004-function-completion/v1",
        "status": "provisional",
        "attempt_id": config["attempt_id"],
        "reservation_id": reservation_id,
        "config_sha256": config_digest,
        "function_call_id": "fc-v10-fixture",
        "requested_gpu": "A100-80GB:2",
        "gpu_count": 2,
        "gpu_allocation_seconds": 100.0,
        "gpu_seconds": 200.0,
        "gpu_hours": 200.0 / 3600.0,
        "pilot_valid": False,
        "scientific_status": "pending-local-finalization",
        "measurable_gates_pass": True,
        "pilot_invalid_reasons": ["budget and provider cleanup are pending"],
        "run_error": None,
        "controller": {"status": "succeeded", "inventory_before": inventory},
        "completed_at_utc": "2026-08-12T00:00:00+00:00",
    }
    remote_path = f"experiment-004/modal/{config['attempt_id']}"
    local_root = experiment_root / f"raw/modal/{config['attempt_id']}"
    completion_path = local_root / "function-completion.json"
    _write_json(completion_path, completion)
    manifest = {
        "schema_version": "sloforge.branchfabric.experiment-004-remote-manifest/v1",
        "attempt_id": config["attempt_id"],
        "remote_prefix": remote_path,
        "artifacts": [
            {
                "relative_path": "function-completion.json",
                "bytes": completion_path.stat().st_size,
                "sha256": hashlib.sha256(completion_path.read_bytes()).hexdigest(),
            }
        ],
    }
    manifest_path = local_root / "REMOTE_MANIFEST.json"
    _write_json(manifest_path, manifest)
    remote = {
        **completion,
        "remote_prefix": remote_path,
        "remote_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    envelope = {
        "result": remote,
        "materialized": {
            "remote_path": remote_path,
            "volume_name": "sloforge-branchfabric-results",
        },
    }
    _remote, validated_inventory, reservation = launcher._validate_remote_identity(
        reservation_id, envelope, config
    )
    assert tuple(item["uuid"] for item in validated_inventory) == (
        "GPU-a100-fixture-0",
        "GPU-a100-fixture-1",
    )
    launcher._validate_downloaded_manifest(
        local_root,
        remote=remote,
        materialized=envelope["materialized"],
        expected_attempt_id=reservation.invocation_id,
    )

    tampered = {**remote, "function_call_id": "fc-substituted"}
    with pytest.raises(ValueError, match="function completion differs"):
        launcher._validate_downloaded_manifest(
            local_root,
            remote=tampered,
            materialized=envelope["materialized"],
            expected_attempt_id=reservation.invocation_id,
        )
    substituted_inventory = [*inventory]
    substituted_inventory[1] = {**inventory[1], "name": "NVIDIA H100 80GB HBM3"}
    substituted = {
        **remote,
        "controller": {"status": "succeeded", "inventory_before": substituted_inventory},
    }
    with pytest.raises(RuntimeError, match="two distinct A100-80GB"):
        launcher._validate_remote_identity(
            reservation_id,
            {**envelope, "result": substituted},
            config,
        )


def test_pilot_terminal_failure_preserves_full_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _load(LAUNCHER, "exp004_launcher_terminal_failure")
    experiment_root = tmp_path / "experiment-004"
    ledger = experiment_root / "gpu-hours.json"
    ledger.parent.mkdir(parents=True)
    ledger.write_text('{"sentinel":"reservation-open"}\n')
    monkeypatch.setattr(launcher, "_EXPERIMENT_ROOT", experiment_root)

    launcher._record_terminal_failure(
        reservation_id="exp004-reservation-fixture",
        attempt_id="exp004-v10-fixture",
        stage="result-validation-settlement",
        error=ValueError("manifest tamper"),
        elapsed_seconds=12.0,
    )
    failure = json.loads(
        (experiment_root / "logs/exp004-v10-fixture-terminal-failure.json").read_text()
    )
    assert failure["conservatively_committed_gpu_seconds"] == 680.0
    assert not failure["reservation_cleared"]
    assert ledger.read_text() == '{"sentinel":"reservation-open"}\n'


def test_sole_coordinator_reservation_is_explicit_and_second_reservation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _load(LAUNCHER, "exp004_launcher_reservation")
    ledger = tmp_path / "gpu-hours.json"
    monkeypatch.setattr(launcher, "_LEDGER", ledger)
    config = _pilot_payload()

    reservation_id, preflight = launcher._reserve(config)
    document = json.loads(ledger.read_text())
    assert reservation_id.startswith("exp004-")
    assert document["reservations"] == [
        {
            "config_sha256": document["reservations"][0]["config_sha256"],
            "gpu_count": 2,
            "invocation_id": config["attempt_id"],
            "maximum_wall_seconds": 340.0,
            "requested_gpu": "A100-80GB",
            "reservation_id": reservation_id,
        }
    ]
    assert preflight["gpu_count"] == 2
    assert preflight["proposed_maximum_gpu_seconds"] == 680.0
    with pytest.raises(ValueError, match="another GPU invocation reservation is already active"):
        launcher._reserve(config)
    assert json.loads(ledger.read_text()) == document


def _write_peer_fixture(path: Path) -> None:
    path.write_text(
        """\
import argparse, json, os
parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
parser.add_argument('--expected-gpu-uuid', action='append', dest='uuids', required=True)
args = parser.parse_args()
payload = {
    'schema_version': 'sloforge.branchfabric.experiment-004-cuda-peer-probe/v1',
    'expected_physical_gpu_uuids': args.uuids,
    'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
    'torch_cuda_version': 'fixture-no-cuda',
    'matrix': {
        'schema_version': 'sloforge.branchfabric.cuda-peer-access/v1',
        'observed_at_monotonic_ns': 1,
        'device_count': 2,
        'device_names': ['fixture-a100-0', 'fixture-a100-1'],
        'access': [
            {'source_logical_device': 0, 'destination_logical_device': 1, 'can_access_peer': True},
            {'source_logical_device': 1, 'destination_logical_device': 0, 'can_access_peer': True},
        ],
    },
}
with open(args.output, 'x') as handle:
    json.dump(payload, handle)
"""
    )


def _write_orphaning_worker_fixture(path: Path) -> None:
    path.write_text(
        """\
import json, os, subprocess, sys, time
def value(flag):
    return sys.argv[sys.argv.index(flag) + 1]
role = value('--role')
barrier = value('--barrier-root')
grandchild = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
with open(os.path.join(barrier, role + '.ready.json'), 'x') as handle:
    json.dump({'role': role, 'pid': os.getpid(), 'grandchild_pid': grandchild.pid}, handle)
"""
    )


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_timeout_cleanup_kills_descendants_after_worker_leaders_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = _load(CONTROLLER, "exp004_controller_cleanup")
    worker = tmp_path / "gpu_reclamation_worker.py"
    peer = tmp_path / "gpu_peer_probe.py"
    _write_orphaning_worker_fixture(worker)
    _write_peer_fixture(peer)
    config = tmp_path / "config.json"
    config.write_text("{}")
    model = tmp_path / "model"
    model.mkdir()
    inventory = (
        {
            "index": 0,
            "uuid": "GPU-first-1",
            "name": "NVIDIA A100 80GB PCIe",
            "driver_version": "fixture",
            "memory_total_mib": 81_920,
            "memory_used_mib": 0,
            "utilization_percent": 0,
        },
        {
            "index": 1,
            "uuid": "GPU-second-2",
            "name": "NVIDIA A100 80GB PCIe",
            "driver_version": "fixture",
            "memory_total_mib": 81_920,
            "memory_used_mib": 0,
            "utilization_percent": 0,
        },
    )
    monkeypatch.setattr(controller, "_inventory", lambda: inventory)
    monkeypatch.setattr(controller, "_compute_processes", lambda: ())
    # This fixture shares pytest's interpreter with tensor adapter tests; the
    # production controller instead starts in a fresh Modal Function process.
    monkeypatch.setattr(
        controller,
        "cuda_clean_import_audit",
        lambda stage: {"stage": stage, "loaded_forbidden_modules": [], "cuda_clean": True},
    )
    work = tmp_path / "work"

    result = controller.run_two_gpu_controller(
        config_path=config,
        work_root=work,
        worker_path=worker,
        model_snapshot=model,
        readiness_timeout_seconds=2.0,
        transaction_timeout_seconds=0.2,
    )
    assert result["status"] == "failed"
    assert result["transaction_error"]["type"] == "RuntimeError"
    assert "workers exited while waiting" in result["transaction_error"]["message"]
    assert result["cleanup_error"] is None
    assert result["compute_processes_after"] == ()

    ready = [
        json.loads(path.read_text()) for path in sorted((work / "barriers").glob("*.ready.json"))
    ]
    pids = [int(row[key]) for row in ready for key in ("pid", "grandchild_pid")]
    deadline = time.monotonic() + 2.0
    while any(_pid_exists(pid) for pid in pids) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert all(not _pid_exists(pid) for pid in pids)


def test_peer_probe_fixture_is_bounded_exact_pair_and_parent_cuda_clean(
    tmp_path: Path,
) -> None:
    controller = _load(CONTROLLER, "exp004_controller_peer")
    peer = tmp_path / "gpu_peer_probe.py"
    _write_peer_fixture(peer)
    (tmp_path / "logs").mkdir()
    actions: list[dict[str, object]] = []
    before = set(sys.modules)
    result = controller._run_peer_probe(
        peer_probe_path=peer,
        work_root=tmp_path,
        gpu_uuids=("GPU-first-1", "GPU-second-2"),
        cleanup_actions=actions,
    )
    newly_loaded = set(sys.modules) - before
    assert not {
        name
        for name in newly_loaded
        if name == "torch" or name.startswith(("torch.", "vllm.", "triton."))
    }
    assert result["matrix"]["device_count"] == 2
    assert result["cuda_visible_devices"] == "GPU-first-1,GPU-second-2"
    assert actions == []
