from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
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
INTEGRATED_CONFIG = (
    ROOT / "artifacts/branchfabric/gpu-validation/experiment-004/raw/"
    "exp004-v10-integrated-s41-v1-config.json"
)
INTEGRATED_CONFIG_V2 = (
    ROOT / "artifacts/branchfabric/gpu-validation/experiment-004/raw/"
    "exp004-v10-integrated-s41-v2-config.json"
)
INTEGRATED_CONFIG_V3 = (
    ROOT / "artifacts/branchfabric/gpu-validation/experiment-004/raw/"
    "exp004-v10-integrated-s41-v3-config.json"
)
INTEGRATED_CONFIG_V4 = (
    ROOT / "artifacts/branchfabric/gpu-validation/experiment-004/raw/"
    "exp004-v10-naive-s41-v4-config.json"
)
INTEGRATED_CONFIG_V5 = (
    ROOT / "artifacts/branchfabric/gpu-validation/experiment-004/raw/"
    "exp004-v10-naive-s41-v5-config.json"
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
        "lambda_2_rps": 20.0,
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
        "two_gpu_headroom_adr_artifact": None,
        "two_gpu_headroom_adr_sha256": None,
    }


def _deadline_evidence(*, controller_seconds: float, elapsed_seconds: float) -> dict[str, object]:
    entry_ns = 1_000_000_000
    return {
        "function_entry_monotonic_ns": entry_ns,
        "function_deadline_monotonic_ns": entry_ns + 588_000_000_000,
        "controller_deadline_monotonic_ns": entry_ns + round(controller_seconds * 1e9),
        "post_controller_reserve_seconds": 0,
        "remaining_function_seconds_at_completion": 588.0 - elapsed_seconds,
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


def test_integrated_controller_strictly_decodes_observation_enum_from_json() -> None:
    controller = _load(CONTROLLER, "exp004_controller_observation_json")
    raw = _capacity_raw(
        topology=ProbeTopology.GPU0_ONLY,
        rate=10.0,
        capacity=20.0,
        probe_id="strict-json-enum",
    )
    row = raw.observations[0].model_dump(mode="json")
    round_tripped = json.loads(json.dumps({"observations": [row]}))["observations"][0]
    assert round_tripped["terminal_state"] == "completed"
    with pytest.raises(ValueError, match="RequestTerminalState"):
        CapacityRequestObservation.model_validate(round_tripped, strict=True)

    observation = controller._validate_capacity_observation_json(round_tripped)
    assert observation == raw.observations[0]
    assert observation.terminal_state is RequestTerminalState.COMPLETED
    with pytest.raises(ValueError):
        controller._validate_capacity_observation_json(
            {**round_tripped, "global_sequence": str(round_tripped["global_sequence"])}
        )


def test_integrated_controller_requires_both_transaction_compilation_observations() -> None:
    controller = _load(CONTROLLER, "exp004_controller_transaction_compilation")

    def payload(role: str) -> dict[str, object]:
        return {
            "status": "succeeded",
            "role": role,
            "measured_transaction_compilation_observation": {
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
            },
        }

    serving = payload("serving")
    rollout = payload("rollout")
    assert set(
        controller._validate_measured_transaction_compilation_payloads((serving, rollout))
    ) == {"serving", "rollout"}
    bad_rollout = json.loads(json.dumps(rollout))
    bad_rollout["measured_transaction_compilation_observation"]["events"] = [
        "capturing CUDA graphs"
    ]
    bad_rollout["measured_transaction_compilation_observation"]["passed"] = False
    with pytest.raises(RuntimeError, match="rollout measured transaction compilation gate"):
        controller._validate_measured_transaction_compilation_payloads((serving, bad_rollout))


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
    adr_payload = {
        **_v10_payload(),
        "lambda_2_rps": 18.0,
        "two_gpu_headroom_adr_artifact": (
            "artifacts/branchfabric/gpu-validation/experiment-004/calibration/reviews/"
            "agent13-two-gpu-headroom-adr.json"
        ),
        "two_gpu_headroom_adr_sha256": "3" * 64,
    }
    assert (
        module.Experiment004PilotConfig.model_validate(
            adr_payload, strict=True
        ).two_gpu_headroom_adr_sha256
        == "3" * 64
    )
    with pytest.raises(ValueError, match="requires its hash-bound headroom ADR"):
        module.Experiment004PilotConfig.model_validate(
            {**_v10_payload(), "lambda_2_rps": 18.0}, strict=True
        )
    with pytest.raises(ValueError, match="must not carry an exception ADR"):
        module.Experiment004PilotConfig.model_validate(
            {
                **_v10_payload(),
                "two_gpu_headroom_adr_artifact": adr_payload["two_gpu_headroom_adr_artifact"],
                "two_gpu_headroom_adr_sha256": "3" * 64,
            },
            strict=True,
        )
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
                "hard_additional_gpu_seconds": 12600.0,
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


def test_integrated_json_config_uses_only_the_evidence_derived_v10_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _load(LAUNCHER, "exp004_integrated_launcher_config")
    monkeypatch.setattr(launcher, "_verify_authorized_code_commit", lambda *_args: None)
    payload = launcher._validate_config(INTEGRATED_CONFIG_V5)
    assert payload["attempt_id"] == "exp004-v10-naive-s41-v5"
    assert payload["execution_mode"] == "integrated-calibration-v10"
    assert payload["serving_spike_request_rate_per_second"] == 15.0
    assert launcher._reservation_wall_seconds(payload) == 588.0
    assert launcher._reservation_wall_seconds(payload) * launcher._GPU_COUNT == 1176.0
    changed_warmup = tmp_path / "changed-warmup.json"
    _write_json(changed_warmup, {**payload, "warmup_seconds": 1.5})
    with pytest.raises(ValueError, match="preauthorized v10 contract"):
        launcher._validate_config(changed_warmup)
    for stale in (
        INTEGRATED_CONFIG,
        INTEGRATED_CONFIG_V2,
        INTEGRATED_CONFIG_V3,
        INTEGRATED_CONFIG_V4,
    ):
        with pytest.raises(ValueError):
            launcher._validate_config(stale)

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
    module = _load(MODAL_APP, "exp004_modal_integrated_json")
    config = module.Experiment004PilotConfig.model_validate(payload, strict=True)
    assert config.sanity_guard_measurement_seconds == 3.0
    assert config.lambda_1_rps == 12.0
    assert config.lambda_spike_rps == 15.0
    assert config.lambda_2_rps == 20.0
    assert config.serving_overload_queue_trigger == 20
    assert config.serving_overload_queue_abort == 25
    assert config.warmup_seconds == 1.0
    assert not hasattr(config, "rate_grid_rps")
    assert config.initialization_timeout_seconds == 160
    assert config.disable_log_stats is False
    assert config.maximum_wall_seconds == 578.0
    assert config.controller_timeout_seconds == 578.0
    assert module.INTEGRATED_POST_CONTROLLER_RESERVE_SECONDS == 10.0
    assert module.INTEGRATED_CONTROLLER_DEADLINE_SECONDS == 578.0
    function_deadline_ns, controller_deadline_ns = module._absolute_function_deadlines(
        function_entry_ns=1_000_000_000,
        config=config,
    )
    assert function_deadline_ns == 589_000_000_000
    assert controller_deadline_ns == 579_000_000_000
    assert function_deadline_ns - controller_deadline_ns == 10_000_000_000
    with pytest.raises(ValueError):
        module.Experiment004PilotConfig.model_validate(
            {**payload, "maximum_wall_seconds": 578.0001}, strict=True
        )
    with pytest.raises(ValueError):
        module.Experiment004PilotConfig.model_validate(
            {**payload, "controller_timeout_seconds": 578.0001}, strict=True
        )
    with pytest.raises(ValueError, match="25-request scientific abort"):
        module.Experiment004PilotConfig.model_validate(
            {**payload, "serving_overload_queue_abort": 64}, strict=True
        )
    with pytest.raises(ValueError, match="predeclared 1s control warmup"):
        module.Experiment004PilotConfig.model_validate(
            {**payload, "warmup_seconds": 1.5}, strict=True
        )
    module._require_integrated_controller_success({"status": "succeeded"})
    with pytest.raises(
        RuntimeError,
        match=(
            r"integrated controller failed: ValidationError: .*terminal_state.*RequestTerminalState"
        ),
    ):
        module._require_integrated_controller_success(
            {
                "status": "failed",
                "controller_error": {
                    "type": "ValidationError",
                    "message": (
                        "terminal_state: Input should be an instance of RequestTerminalState"
                    ),
                },
            }
        )


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
            rate=35.0,
            capacity=40.0,
            probe_id="capacity-two-35",
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


def test_headroom_exception_requires_hash_bound_strong_stability_adr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _load(LAUNCHER, "exp004_launcher_headroom_adr")
    experiment_root = tmp_path / "artifacts/branchfabric/gpu-validation/experiment-004"
    attempt_id = "exp004-capacity-fixture"
    result = evaluate_probe(
        _capacity_raw(
            topology=ProbeTopology.TWO_GPU_ROUND_ROBIN,
            rate=30.0,
            capacity=40.0,
            probe_id="capacity-two-30",
        )
    )
    result_path = (
        experiment_root
        / f"calibration/raw/modal/{attempt_id}/calibration/two-gpu/capacity-two-30/result.json"
    )
    _write_json(result_path, result)
    selection_path = experiment_root / "calibration/selected-load.json"
    _write_json(selection_path, {"fixture": "selected-load"})
    selection = SimpleNamespace(
        lambda_spike_rps=25.0,
        lambda_2_rps=30.0,
        two_gpu_sustainable_probe_id="capacity-two-30",
    )
    strong = {
        "measurement_seconds": 10.0,
        "p95_ttft_seconds": result.p95_ttft_seconds,
        "completed_rate_rps": result.completed_rate_rps,
        "observed_offered_rate_rps": result.observed_offered_rate_rps,
        "queue_slope_requests_per_second": result.queue_slope_requests_per_second,
        "maximum_queue_depth": result.maximum_queue_depth,
        "all_measurement_requests_completed": result.all_measurement_requests_completed,
    }
    adr_path = experiment_root / "calibration/reviews/agent13-two-gpu-headroom-adr.json"
    adr = {
        "schema_version": "sloforge.branchfabric.capacity-headroom-adr/v1",
        "reviewer": "agent13",
        "verdict": "approved",
        "calibration_attempt_id": attempt_id,
        "selected_load_artifact": launcher._SELECTED_LOAD_REFERENCE,
        "selected_load_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        "spike_over_lambda_2": 25.0 / 30.0,
        "two_gpu_sustainable_probe_id": "capacity-two-30",
        "two_gpu_result_provenance": _artifact_ref(result_path),
        "strong_stability": strong,
        "justification": (
            "The measured two-GPU point completed every request with sub-second p95 TTFT, "
            "no persistent queue drift, and completion throughput at least equal to arrivals."
        ),
    }
    _write_json(adr_path, adr)
    payload = {
        "calibration_attempt_id": attempt_id,
        "two_gpu_headroom_adr_artifact": launcher._HEADROOM_ADR_REFERENCE,
        "two_gpu_headroom_adr_sha256": hashlib.sha256(adr_path.read_bytes()).hexdigest(),
    }
    monkeypatch.setattr(launcher, "_ROOT", tmp_path)
    monkeypatch.setattr(launcher, "_EXPERIMENT_ROOT", experiment_root)

    launcher._validate_headroom_adr(
        payload=payload,
        selection=selection,
        selection_path=selection_path,
    )
    adr["strong_stability"]["p95_ttft_seconds"] = 1.5
    _write_json(adr_path, adr)
    payload["two_gpu_headroom_adr_sha256"] = hashlib.sha256(adr_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="does not prove strong"):
        launcher._validate_headroom_adr(
            payload=payload,
            selection=selection,
            selection_path=selection_path,
        )


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
        "absolute_deadlines": _deadline_evidence(
            controller_seconds=float(config["maximum_wall_seconds"]),
            elapsed_seconds=100.0,
        ),
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
    extended_deadline = {
        **remote,
        "absolute_deadlines": {
            **remote["absolute_deadlines"],
            "controller_deadline_monotonic_ns": (
                remote["absolute_deadlines"]["controller_deadline_monotonic_ns"] + 1_000_000_000
            ),
        },
    }
    with pytest.raises(ValueError, match="absolute deadline evidence is invalid"):
        launcher._validate_remote_identity(
            reservation_id,
            {**envelope, "result": extended_deadline},
            config,
        )


def test_pilot_terminal_failure_clears_and_conservatively_charges_full_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _load(LAUNCHER, "exp004_launcher_terminal_failure")
    experiment_root = tmp_path / "experiment-004"
    ledger_path = experiment_root / "gpu-hours.json"
    reservation_id = "exp004-reservation-fixture"
    attempt_id = "exp004-v10-fixture"
    _write_json(
        ledger_path,
        Experiment004GpuHourLedger(
            consumed_additional_gpu_seconds=0.0,
            reservations=(
                GpuInvocationReservation(
                    reservation_id=reservation_id,
                    invocation_id=attempt_id,
                    maximum_wall_seconds=340.0,
                    config_sha256="c" * 64,
                ),
            ),
        ),
    )
    monkeypatch.setattr(launcher, "_EXPERIMENT_ROOT", experiment_root)
    monkeypatch.setattr(launcher, "_LEDGER", ledger_path)

    failure = launcher._record_terminal_failure(
        reservation_id=reservation_id,
        attempt_id=attempt_id,
        stage="result-validation-settlement",
        error=ValueError("manifest tamper"),
        elapsed_seconds=12.0,
    )
    assert failure == json.loads(
        (experiment_root / f"logs/{attempt_id}-terminal-failure.json").read_text()
    )
    assert failure["actual_gpu_seconds"] is None
    assert failure["conservatively_accounted_gpu_seconds"] == 680.0
    assert failure["reservation_cleared"] is True
    assert failure["remaining_hard_budget_gpu_seconds"] == 10_120.0
    settled = Experiment004GpuHourLedger.model_validate_json(ledger_path.read_text(), strict=True)
    assert not settled.reservations
    assert not settled.intervals
    assert settled.consumed_additional_gpu_seconds == 680.0
    assert settled.conservative_failure_charges[0].reservation_id == reservation_id
    evidence = Path(str(failure["charge_evidence_reference"]))
    assert hashlib.sha256(evidence.read_bytes()).hexdigest() == failure["charge_evidence_sha256"]


def test_model_validation_failure_bundle_is_hash_validated_settled_and_preserved_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sloforge.helix.characterization.gpu_reclamation_v10_validity import (
        fail_closed_v10_scientific_validity,
    )

    launcher = _load(LAUNCHER, "exp004_launcher_failed_bundle_settlement")
    experiment_root = tmp_path / "experiment-004"
    ledger_path = experiment_root / "gpu-hours.json"
    config = _v10_payload()
    config_digest = hashlib.sha256(launcher._canonical_bytes(config)).hexdigest()
    reservation_id = "exp004-failed0123456789012"
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
    reason = (
        "GPU pilot/controller failed before scientific validation: "
        "FileNotFoundError: cached model fixture is absent"
    )
    validity = fail_closed_v10_scientific_validity(reason)
    run_error = {
        "type": "FileNotFoundError",
        "message": "cached model fixture is absent",
        "traceback": "fixture trace",
    }
    completion = {
        "schema_version": "sloforge.branchfabric.experiment-004-function-completion/v1",
        "status": "failed",
        "attempt_id": config["attempt_id"],
        "reservation_id": reservation_id,
        "config_sha256": config_digest,
        "function_call_id": "fc-v10-failed-fixture",
        "requested_gpu": "A100-80GB:2",
        "gpu_count": 2,
        "gpu_allocation_seconds": 30.0,
        "gpu_seconds": 60.0,
        "gpu_hours": 60.0 / 3600.0,
        "pilot_valid": False,
        "scientific_status": "invalid",
        "measurable_gates_pass": False,
        "pilot_invalid_reasons": [reason],
        "run_error": run_error,
        "controller": {
            "schema_version": "sloforge.branchfabric.experiment-004-controller-failure/v1",
            "status": "failed",
            "transaction_error": run_error,
        },
        "absolute_deadlines": _deadline_evidence(
            controller_seconds=float(config["maximum_wall_seconds"]),
            elapsed_seconds=30.0,
        ),
        "completed_at_utc": "2026-08-12T00:00:00+00:00",
    }
    source = tmp_path / "remote-bundle"
    _write_json(source / "function-completion.json", completion)
    _write_json(
        source / "analysis/scientific-validity.remote-provisional.json",
        validity,
    )
    _write_json(
        source / "topology/topology-commands.json",
        [
            {
                "command": {"name": "gpu_inventory"},
                "status": "success",
                "returncode": 0,
                "stdout": (
                    "0, GPU-fixture-a100-0, NVIDIA A100-SXM4-80GB, 0000:01:00.0, "
                    "81920, 575.57.08\n"
                    "1, GPU-fixture-a100-1, NVIDIA A100-SXM4-80GB, 0000:02:00.0, "
                    "81920, 575.57.08\n"
                ),
            }
        ],
    )
    remote_path = f"experiment-004/modal/{config['attempt_id']}"
    artifacts = [
        {
            "relative_path": path.relative_to(source).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(source.rglob("*"))
        if path.is_file()
    ]
    _write_json(
        source / "REMOTE_MANIFEST.json",
        {
            "schema_version": "sloforge.branchfabric.experiment-004-remote-manifest/v1",
            "attempt_id": config["attempt_id"],
            "remote_prefix": remote_path,
            "artifacts": artifacts,
        },
    )
    remote = {
        **completion,
        "remote_prefix": remote_path,
        "remote_manifest_sha256": hashlib.sha256(
            (source / "REMOTE_MANIFEST.json").read_bytes()
        ).hexdigest(),
    }
    envelope = {
        "result": remote,
        "materialized": {
            "remote_path": remote_path,
            "volume_name": "sloforge-branchfabric-results",
        },
    }

    def materialize(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        target = experiment_root / f"raw/modal/{config['attempt_id']}"
        shutil.copytree(source, target)
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(launcher.subprocess, "run", materialize)
    local_root = launcher._settle(reservation_id, envelope, 20.0, config)

    settled = Experiment004GpuHourLedger.model_validate_json(ledger_path.read_text(), strict=True)
    assert not settled.reservations
    assert settled.consumed_additional_gpu_seconds == 60.0
    assert settled.intervals[-1].actual_gpu_models == (
        "NVIDIA A100-SXM4-80GB",
        "NVIDIA A100-SXM4-80GB",
    )
    canonical = json.loads((local_root / "analysis/scientific-validity.json").read_text())
    assert canonical["scientifically_valid"] is False
    settlement = json.loads((local_root / "analysis/failed-run-settlement.json").read_text())
    assert settlement["status"] == "settled-invalid"
    assert settlement["classification"] == "runtime-failure"
    assert settlement["reservation_cleared"] is True
    assert settlement["remote_observed_allocation_seconds"] == 30.0


def test_failed_bundle_scientific_status_mismatch_is_rejected_before_settlement(
    tmp_path: Path,
) -> None:
    from sloforge.helix.characterization.gpu_reclamation_v10_validity import (
        fail_closed_v10_scientific_validity,
    )

    launcher = _load(LAUNCHER, "exp004_launcher_failed_bundle_invalidity")
    local_root = tmp_path / "bundle"
    _write_json(
        local_root / "analysis/scientific-validity.remote-provisional.json",
        fail_closed_v10_scientific_validity("raw reason"),
    )
    with pytest.raises(ValueError, match="does not preserve its scientific invalidity"):
        launcher._validate_downloaded_invalid_outcome(
            local_root=local_root,
            remote={
                "status": "failed",
                "pilot_invalid_reasons": ["substituted reason"],
            },
        )


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


def test_integrated_controller_rejects_an_expired_absolute_deadline(tmp_path: Path) -> None:
    controller = _load(CONTROLLER, "exp004_controller_absolute_deadline")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"execution_mode": "integrated-calibration-v10"}))
    worker = tmp_path / "gpu_reclamation_worker.py"
    worker.write_text("raise SystemExit(0)\n")
    model = tmp_path / "model"
    model.mkdir()

    with pytest.raises(ValueError, match="future absolute deadline"):
        controller.run_integrated_two_gpu_controller(
            config_path=config,
            work_root=tmp_path / "work",
            worker_path=worker,
            model_snapshot=model,
            repository_root=tmp_path,
            readiness_timeout_seconds=1.0,
            absolute_deadline_ns=time.monotonic_ns() - 1,
        )


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
