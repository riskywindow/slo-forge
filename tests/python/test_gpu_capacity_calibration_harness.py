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
    CapacityProbeRaw,
    CapacityRequestObservation,
    ProbeTopology,
    RequestTerminalState,
    SpikeSelection,
    build_probe_plan,
    evaluate_probe,
)
from sloforge.helix.characterization.gpu_reclamation_methodology import (
    Experiment004GpuHourLedger,
    GpuInvocationReservation,
)

ROOT = Path(__file__).parents[2]
WORKER = ROOT / "experiments/branchfabric/gpu_capacity_calibration_worker.py"
CONTROLLER = ROOT / "experiments/branchfabric/gpu_capacity_calibration_controller.py"
MODAL_APP = ROOT / "experiments/branchfabric/modal_gpu_capacity_calibration.py"
LAUNCHER = ROOT / "tools/branchfabric-experiment-004-calibrate.py"
FAILURE_CONVERTER = ROOT / "tools/branchfabric-experiment-004-convert-capacity-failure.py"
V2_FAILURE_CONVERTER = ROOT / "tools/branchfabric-experiment-004-convert-capacity-v2-failure.py"
CONFIG = (
    ROOT
    / "artifacts/branchfabric/gpu-validation/experiment-004/calibration/calibration-config-v1.json"
)
CONFIG_V2 = (
    ROOT
    / "artifacts/branchfabric/gpu-validation/experiment-004/calibration/calibration-config-v2.json"
)
PLAN_V2 = (
    ROOT
    / "artifacts/branchfabric/gpu-validation/experiment-004/calibration/calibration-plan-v2.json"
)
CALIBRATION_BLOCKER = CONFIG.parent / "calibration-blocker.json"
CALIBRATION_OUTCOME = CONFIG.parent / "calibration-outcome.json"
CONVERSION_PROVENANCE_CORRECTION = (
    CONFIG.parent / "logs/capacity-failure-charge-conversion-provenance-correction.json"
)


def _load(path: Path, name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def test_capacity_controller_import_is_cuda_clean_in_fresh_interpreter() -> None:
    code = f"""
import importlib.util, json, sys
sys.path.insert(0, {str(CONTROLLER.parent)!r})
before = set(sys.modules)
spec = importlib.util.spec_from_file_location('capacity_controller_fixture', {str(CONTROLLER)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
prefixes = ('torch', 'vllm', 'triton', 'cupy', 'pynvml', 'cuda', 'numba.cuda')
loaded = sorted(name for name in set(sys.modules) - before if any(name == prefix or name.startswith(prefix + '.') for prefix in prefixes))
print(json.dumps(loaded))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=10.0,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == []


def test_controller_serializes_actual_nested_pydantic_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(CONTROLLER.parent))
    controller = _load(CONTROLLER, "capacity_controller_pydantic_serialization")
    plan = build_probe_plan(
        probe_id="capacity-serialization-fixture",
        seed=41,
        topology=ProbeTopology.GPU0_ONLY,
        configured_rate_rps=10.0,
        start_ns=1_000_000_000,
        warmup_seconds=1.0,
        measurement_seconds=10.0,
    )
    observations = tuple(
        CapacityRequestObservation(
            request_id=item.request_id,
            global_sequence=item.global_sequence,
            device=item.assigned_device,
            scheduled_arrival_ns=item.scheduled_arrival_ns,
            offered_ns=item.scheduled_arrival_ns,
            enqueued_ns=item.scheduled_arrival_ns,
            admitted_ns=item.scheduled_arrival_ns,
            first_token_ns=item.scheduled_arrival_ns + 1,
            completed_ns=item.scheduled_arrival_ns + 2,
            terminated_ns=item.scheduled_arrival_ns + 2,
            requested_output_tokens=64,
            emitted_tokens=64,
            terminal_state=RequestTerminalState.COMPLETED,
        )
        for item in plan.arrivals
    )
    raw = CapacityProbeRaw(
        plan=plan,
        observations=observations,
        tail_drain_end_ns=plan.measurement_end_ns + 1_000_000_000,
        probe_end_ns=plan.measurement_end_ns + 1_000_000_000,
        tail_drain_seconds=1.0,
    )
    result = evaluate_probe(raw)
    selection = SpikeSelection(
        lambda_1_rps=10.0,
        lambda_2_rps=15.0,
        lambda_spike_rps=12.0,
        spike_over_lambda_1=1.2,
        spike_over_lambda_2=0.8,
        one_gpu_unsustainable_probe_id="capacity-one-unsustainable",
        one_gpu_sustainable_probe_id="capacity-one-sustainable",
        two_gpu_sustainable_probe_id="capacity-two-sustainable",
    )
    payloads = {
        "command.json": {"schema_version": "fixture/v1", "plan": plan},
        "plan.json": plan,
        "raw.json": raw,
        "result.json": result,
        "calibration-result.json": {
            "probe_results": (result,),
            "selection": selection,
        },
        "selected-load.json": selection,
    }

    for filename, payload in payloads.items():
        path = tmp_path / filename
        controller._write_new(path, payload)
        decoded = json.loads(path.read_text())
        assert decoded is not None
    assert json.loads((tmp_path / "command.json").read_text())["plan"]["topology"] == ("gpu0-only")
    assert (
        json.loads((tmp_path / "calibration-result.json").read_text())["probe_results"][0][
            "verdict"
        ]
        == "sustainable"
    )


def _worker_completion(*, device: str, uuid: str, status: str = "succeeded") -> dict:
    return {
        "schema_version": "sloforge.branchfabric.capacity-worker-completion/v1",
        "status": status,
        "device": device,
        "physical_gpu_uuid": uuid,
        "started_ns": 1,
        "ended_ns": 2,
        "completed_at_utc": "2026-08-12T00:00:00+00:00",
        "runtime_evidence": {},
        "error": None if status == "succeeded" else {"type": "RuntimeError"},
    }


def test_controller_rejects_failed_worker_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(CONTROLLER.parent))
    controller = _load(CONTROLLER, "capacity_controller_failed_worker")
    paths = (tmp_path / "gpu0.json", tmp_path / "gpu1.json")
    paths[0].write_text(json.dumps(_worker_completion(device="gpu0", uuid="GPU-0")))
    paths[1].write_text(
        json.dumps(_worker_completion(device="gpu1", uuid="GPU-1", status="failed"))
    )

    with pytest.raises(RuntimeError, match="gpu1 worker completion failed"):
        controller._validate_worker_completions(
            result_paths=paths,
            expected_inventory=({"uuid": "GPU-0"}, {"uuid": "GPU-1"}),
            returncodes={"gpu0": 0, "gpu1": 1},
        )


def test_controller_rejects_nonzero_worker_exit_after_success_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(CONTROLLER.parent))
    controller = _load(CONTROLLER, "capacity_controller_nonzero_exit")
    paths = (tmp_path / "gpu0.json", tmp_path / "gpu1.json")
    paths[0].write_text(json.dumps(_worker_completion(device="gpu0", uuid="GPU-0")))
    paths[1].write_text(json.dumps(_worker_completion(device="gpu1", uuid="GPU-1")))

    with pytest.raises(RuntimeError, match="return codes were invalid"):
        controller._validate_worker_completions(
            result_paths=paths,
            expected_inventory=({"uuid": "GPU-0"}, {"uuid": "GPU-1"}),
            returncodes={"gpu0": 0, "gpu1": 9},
        )


def test_persistent_worker_executes_only_its_global_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _load(WORKER, "capacity_worker_fixture")
    fake_reclamation = ModuleType("gpu_reclamation_worker")
    fake_reclamation._sampling_params = lambda **_kwargs: object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gpu_reclamation_worker", fake_reclamation)
    monkeypatch.setenv("SLOFORGE_EXP004_PHYSICAL_GPU_UUID", "GPU-fixture-0")

    class Engine:
        def __init__(self) -> None:
            self.active: list[str] = []

        def add_request(self, request_id: str, _prompt: object, _params: object) -> None:
            self.active.append(request_id)

        def step(self):  # type: ignore[no-untyped-def]
            active, self.active = self.active, []
            return tuple(
                SimpleNamespace(
                    request_id=request_id,
                    outputs=(SimpleNamespace(token_ids=tuple(range(64))),),
                    finished=True,
                )
                for request_id in active
            )

        def abort_request(self, request_ids: list[str]) -> None:
            self.active = [item for item in self.active if item not in request_ids]

    plan = build_probe_plan(
        probe_id="capacity-route-fixture",
        seed=41,
        topology=ProbeTopology.TWO_GPU_ROUND_ROBIN,
        configured_rate_rps=10.0,
        start_ns=time.monotonic_ns() + 50_000_000,
        warmup_seconds=0.05,
        measurement_seconds=0.2,
    )
    result = worker._execute_probe(
        Engine(),
        plan_payload=plan.model_dump(mode="json"),
        device="gpu1",
        prefix=(1, 2, 3),
        output_tokens=64,
        seed=41,
        probe_timeout_seconds=2.0,
        tail_drain_seconds=0.05,
        producer_queue_capacity=8,
    )

    assert result["tail_drain_end_ns"] == plan.measurement_end_ns + 50_000_000
    assert result["tail_drain_end_ns"] <= result["probe_end_ns"]
    assert result["probe_end_ns"] <= result["tail_drain_end_ns"] + 1_000_000_000
    assert result["observations"]
    assert all(row["device"] == "gpu1" for row in result["observations"])
    assert all(row["global_sequence"] % 2 == 1 for row in result["observations"])
    assert all(row["emitted_tokens"] == 64 for row in result["observations"])
    assert all(
        row["scheduled_arrival_ns"] <= row["offered_ns"] <= row["enqueued_ns"]
        for row in result["observations"]
    )


class _FakeVolume:
    def with_mount_options(self, *, read_only: bool) -> tuple[str, bool]:
        return ("mount", read_only)

    def commit(self) -> None:
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
        return lambda function: function

    def local_entrypoint(self):  # type: ignore[no-untyped-def]
        return lambda function: function


def test_modal_capacity_graph_is_exactly_two_a100s_and_160_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    fake_modal = ModuleType("modal")
    fake_modal.__version__ = "1.5.3"  # type: ignore[attr-defined]
    fake_modal.Volume = _FakeVolumeType  # type: ignore[attr-defined]
    fake_modal.App = lambda *_args, **_kwargs: _FakeApp(calls)  # type: ignore[attr-defined]
    fake_modal.current_function_call_id = lambda: "fixture-call"  # type: ignore[attr-defined]
    fake_image = ModuleType("modal_real_gpu_cow")
    fake_image.gpu_image = object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    monkeypatch.setitem(sys.modules, "modal_real_gpu_cow", fake_image)
    monkeypatch.setenv("SLOFORGE_MODAL_PREFLIGHT_TOKEN", "a" * 64)

    module = _load(MODAL_APP, "capacity_modal_fixture")
    config = module.CapacityCalibrationConfig.model_validate_json(CONFIG.read_text(), strict=True)

    assert module.GPU_REQUEST == "A100-80GB:2"
    assert module.GPU_COORDINATOR_RESERVATION_WALL_SECONDS == 212
    assert config.maximum_allocation_wall_seconds == 212.0
    assert len(calls) == 1
    assert calls[0]["gpu"] == "A100-80GB:2"
    assert calls[0]["timeout"] == 160
    assert calls[0]["max_containers"] == 1
    assert calls[0]["single_use_containers"] is True
    assert calls[0]["retries"] == 0
    v1_client_minus_remote_seconds = 133.33001966698794 - 85.441581288
    assert 160 + v1_client_minus_remote_seconds == pytest.approx(207.88843837898794)
    assert 160 + v1_client_minus_remote_seconds < 212
    assert 2 * 212 / 1616.2042090820032 <= 0.2625
    assert 1616.2042090820032 - 2 * 212 - 680 > 0


def test_capacity_launcher_reserves_exactly_424_a100_seconds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _load(LAUNCHER, "capacity_launcher_fixture")
    ledger_path = tmp_path / "gpu-hours.json"
    ledger_path.write_text(
        Experiment004GpuHourLedger(consumed_additional_gpu_seconds=0.0).model_dump_json()
    )
    monkeypatch.setattr(launcher, "_LEDGER", ledger_path)
    payload = launcher._validate_config(CONFIG)

    reservation_id, preflight = launcher._reserve(payload)
    ledger = json.loads(ledger_path.read_text())

    assert reservation_id.startswith("exp004-")
    assert preflight["maximum_wall_seconds"] == 212.0
    assert preflight["proposed_maximum_gpu_seconds"] == 424.0
    assert ledger["reservations"][0]["maximum_wall_seconds"] == 212.0
    assert ledger["reservations"][0]["gpu_count"] == 2
    assert (
        ledger["reservations"][0]["config_sha256"]
        == hashlib.sha256(launcher._canonical_bytes(payload)).hexdigest()
    )


def test_v2_inputs_preserve_v1_and_leave_positive_v10_headroom() -> None:
    launcher = _load(LAUNCHER, "capacity_launcher_v2_contract")
    v1 = launcher._validate_config(CONFIG)
    v2 = launcher._validate_config(CONFIG_V2)
    plan = json.loads(PLAN_V2.read_text())

    assert v1["attempt_id"] == "exp004-capacity-s41-v1"
    assert v2["attempt_id"] == "exp004-capacity-s41-v2"
    intentionally_changed = {"attempt_id", "controller_timeout_seconds"}
    assert {key: value for key, value in v1.items() if key not in intentionally_changed} == {
        key: value for key, value in v2.items() if key not in intentionally_changed
    }
    assert v1["controller_timeout_seconds"] == 125.0
    assert v2["controller_timeout_seconds"] == 150.0
    budget = plan["budget"]
    assert budget["settled_gpu_seconds_before_calibration"] + budget[
        "retained_v1_reservation_gpu_seconds"
    ] + budget["v2_maximum_reservation_gpu_seconds"] + budget[
        "v10_maximum_reservation_gpu_seconds"
    ] == pytest.approx(budget["total_conservative_after_all_reservations_gpu_seconds"])
    assert budget["hard_ceiling_headroom_after_all_reservations_gpu_seconds"] == pytest.approx(
        88.2042090820032
    )
    assert budget["hard_ceiling_headroom_after_all_reservations_gpu_seconds"] > 0


def test_v1_failure_conversion_validates_evidence_and_charges_full_reservation() -> None:
    converter = _load(FAILURE_CONVERTER, "capacity_failure_converter_fixture")
    config = json.loads(CONFIG.read_text())
    config_sha = hashlib.sha256(converter._canonical_bytes(config)).hexdigest()
    reservation = GpuInvocationReservation(
        reservation_id="exp004-acada550ab4cc05f17df",
        invocation_id="exp004-capacity-s41-v1",
        maximum_wall_seconds=212.0,
        config_sha256=config_sha,
    )
    ledger = Experiment004GpuHourLedger(
        consumed_additional_gpu_seconds=0.0,
        reservations=(reservation,),
    )

    updated, record = converter._prepare_failure_charge(
        experiment_root=CONFIG.parents[1], ledger=ledger
    )

    assert updated.consumed_additional_gpu_seconds == 424.0
    assert updated.reservations == ()
    assert updated.intervals[-1].accounted_wall_seconds == 212.0
    assert updated.intervals[-1].function_call_id == "fc-01KZVP7S0E823VKE1BKXER2B8N"
    assert record["downloaded_inventory_verified"] is True
    assert record["accounted_gpu_seconds"] == 424.0


def test_v1_failure_conversion_rejects_downloaded_artifact_tamper(tmp_path: Path) -> None:
    converter = _load(FAILURE_CONVERTER, "capacity_failure_converter_tamper")
    source = CONFIG.parents[1]
    fixture_root = tmp_path / "experiment-004"
    shutil.copytree(source / "calibration", fixture_root / "calibration")
    config = json.loads((fixture_root / "calibration/calibration-config-v1.json").read_text())
    config_sha = hashlib.sha256(converter._canonical_bytes(config)).hexdigest()
    ledger = Experiment004GpuHourLedger(
        consumed_additional_gpu_seconds=0.0,
        reservations=(
            GpuInvocationReservation(
                reservation_id="exp004-acada550ab4cc05f17df",
                invocation_id="exp004-capacity-s41-v1",
                maximum_wall_seconds=212.0,
                config_sha256=config_sha,
            ),
        ),
    )
    completion = (
        fixture_root / "calibration/raw/modal/exp004-capacity-s41-v1/function-completion.json"
    )
    completion.write_text("tampered\n")

    with pytest.raises(ValueError, match="failed integrity"):
        converter._prepare_failure_charge(experiment_root=fixture_root, ledger=ledger)


def test_v2_failure_conversion_validates_timeout_and_charges_full_reservation() -> None:
    converter = _load(V2_FAILURE_CONVERTER, "capacity_v2_failure_converter_fixture")
    config = json.loads(CONFIG_V2.read_text())
    config_sha = hashlib.sha256(converter._canonical_bytes(config)).hexdigest()
    ledger = Experiment004GpuHourLedger(
        consumed_additional_gpu_seconds=0.0,
        reservations=(
            GpuInvocationReservation(
                reservation_id="exp004-e04866e0a14e78d4b522",
                invocation_id="exp004-capacity-s41-v2",
                maximum_wall_seconds=212.0,
                config_sha256=config_sha,
            ),
        ),
    )

    updated, record = converter._prepare_failure_charge(
        experiment_root=CONFIG.parents[1], ledger=ledger
    )

    assert updated.consumed_additional_gpu_seconds == 424.0
    assert updated.reservations == ()
    assert updated.intervals[-1].accounted_wall_seconds == 212.0
    assert updated.intervals[-1].function_call_id == "fc-01KZVPV1V8K5GGJV55GW4XNNRN"
    assert record["failure_classification"] == "engine-readiness-timeout-before-first-probe"
    assert record["downloaded_manifest_artifact_count"] == 9
    assert record["clean_compute_postflight_verified"] is True


def test_v2_failure_conversion_rejects_controller_tamper(tmp_path: Path) -> None:
    converter = _load(V2_FAILURE_CONVERTER, "capacity_v2_failure_converter_tamper")
    source = CONFIG.parents[1]
    fixture_root = tmp_path / "experiment-004"
    shutil.copytree(source / "calibration", fixture_root / "calibration")
    config = json.loads((fixture_root / "calibration/calibration-config-v2.json").read_text())
    config_sha = hashlib.sha256(converter._canonical_bytes(config)).hexdigest()
    ledger = Experiment004GpuHourLedger(
        consumed_additional_gpu_seconds=0.0,
        reservations=(
            GpuInvocationReservation(
                reservation_id="exp004-e04866e0a14e78d4b522",
                invocation_id="exp004-capacity-s41-v2",
                maximum_wall_seconds=212.0,
                config_sha256=config_sha,
            ),
        ),
    )
    controller = (
        fixture_root / "calibration/raw/modal/exp004-capacity-s41-v2/controller-result.json"
    )
    controller.write_text("tampered\n")

    with pytest.raises(ValueError, match="failed integrity"):
        converter._prepare_failure_charge(experiment_root=fixture_root, ledger=ledger)


def test_terminal_calibration_blocker_and_conversion_provenance_are_bound() -> None:
    converter = _load(V2_FAILURE_CONVERTER, "capacity_conversion_digest_fixture")
    blocker = json.loads(CALIBRATION_BLOCKER.read_text())
    outcome = json.loads(CALIBRATION_OUTCOME.read_text())
    correction = json.loads(CONVERSION_PROVENANCE_CORRECTION.read_text())
    ledger_path = CONFIG.parents[1] / "gpu-hours.json"
    ledger = Experiment004GpuHourLedger.model_validate_json(ledger_path.read_text(), strict=True)

    assert blocker["status"] == "valid-blocker"
    assert blocker["capacity_probe_count"] == 0
    assert blocker["lambda_1_rps"] is None
    assert blocker["lambda_2_rps"] is None
    assert blocker["lambda_spike_rps"] is None
    assert blocker["v10_launch_permitted"] is False
    assert blocker["budget"]["budget_terminal"] is True
    assert ledger.reservations == ()
    assert ledger.consumed_additional_gpu_seconds == pytest.approx(6431.795790917997)
    assert ledger.intervals[-1].invocation_id == "exp004-capacity-s41-v2"
    assert ledger.intervals[-1].accounted_wall_seconds == 212.0

    for row, config_path in zip(correction["records"], (CONFIG, CONFIG_V2), strict=True):
        assert row["config_file_sha256"] == hashlib.sha256(config_path.read_bytes()).hexdigest()
        canonical = hashlib.sha256(
            converter._canonical_bytes(json.loads(config_path.read_text()))
        ).hexdigest()
        assert row["canonical_config_sha256"] == canonical
        conversion_path = ROOT / row["original_conversion_record"]["artifact_reference"]
        assert (
            row["original_conversion_record"]["artifact_sha256"]
            == hashlib.sha256(conversion_path.read_bytes()).hexdigest()
        )

    assert outcome["status"] == "blocked"
    assert outcome["v10_launch_permitted"] is False
    assert (
        outcome["blocker"]["artifact_sha256"]
        == hashlib.sha256(CALIBRATION_BLOCKER.read_bytes()).hexdigest()
    )
    assert (
        outcome["provenance_correction"]["artifact_sha256"]
        == hashlib.sha256(CONVERSION_PROVENANCE_CORRECTION.read_bytes()).hexdigest()
    )
    assert (
        outcome["ledger"]["artifact_sha256"] == hashlib.sha256(ledger_path.read_bytes()).hexdigest()
    )


def test_capacity_launcher_requires_budget_before_any_modal_or_ledger_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _load(LAUNCHER, "capacity_launcher_no_budget")
    ledger = tmp_path / "gpu-hours.json"
    lock = tmp_path / "gpu-hours.lock"
    monkeypatch.setattr(launcher, "_LEDGER", ledger)
    monkeypatch.setattr(launcher, "_LOCK", lock)
    monkeypatch.delenv("SLOFORGE_GPU_BUDGET_USD", raising=False)
    monkeypatch.setattr(sys, "argv", [str(LAUNCHER), "--config-path", str(CONFIG)])
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("Modal must not be invoked without budget"),
    )

    with pytest.raises(RuntimeError, match="SLOFORGE_GPU_BUDGET_USD"):
        launcher.main()
    assert not ledger.exists()
    assert not lock.exists()


def test_downloaded_raw_calibration_is_indexed_with_hash_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _load(LAUNCHER, "capacity_launcher_index")
    experiment_root = tmp_path / "experiment-004"
    local_root = experiment_root / "calibration/raw/modal/capacity-fixture"
    probe_root = local_root / "calibration/one-gpu/capacity-probe-00"
    probe_root.mkdir(parents=True)
    for name, payload in (
        ("plan.json", {"kind": "plan"}),
        ("raw.json", {"kind": "raw"}),
        ("result.json", {"kind": "result"}),
    ):
        (probe_root / name).write_text(json.dumps(payload))
    (local_root / "calibration/selected-load.json").write_text(
        json.dumps({"schema_version": "fixture-selection/v1", "lambda_spike_rps": 25.0})
    )
    monkeypatch.setattr(launcher, "_EXPERIMENT_ROOT", experiment_root)

    launcher._materialize_calibration_index(local_root, attempt_id="capacity-fixture")

    index_path = experiment_root / "calibration/one-gpu/capacity-fixture.index.json"
    index = json.loads(index_path.read_text())
    assert index["entries"][0]["raw"]["artifact_reference"].endswith("raw.json")
    assert (
        index["entries"][0]["raw"]["artifact_sha256"]
        == hashlib.sha256((probe_root / "raw.json").read_bytes()).hexdigest()
    )
    selection = json.loads((experiment_root / "calibration/selected-load.json").read_text())
    assert selection["lambda_spike_rps"] == 25.0
    assert selection["raw_provenance"]["artifact_sha256"]


def test_downloaded_manifest_tamper_is_rejected_before_settlement(tmp_path: Path) -> None:
    launcher = _load(LAUNCHER, "capacity_launcher_manifest_tamper")
    local_root = tmp_path / "exp004-capacity-fixture"
    artifact = local_root / "calibration/one-gpu/probe-00/result.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b'{"verdict":"sustainable"}\n')
    remote_path = "experiment-004/calibration/modal/exp004-capacity-fixture"
    manifest = {
        "schema_version": "sloforge.branchfabric.capacity-remote-manifest/v1",
        "attempt_id": "exp004-capacity-fixture",
        "remote_prefix": remote_path,
        "artifacts": [
            {
                "relative_path": artifact.relative_to(local_root).as_posix(),
                "bytes": artifact.stat().st_size,
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }
        ],
    }
    manifest_path = local_root / "REMOTE_MANIFEST.json"
    manifest_path.write_bytes(launcher._canonical_bytes(manifest))
    remote = {"remote_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
    materialized = {"remote_path": remote_path}

    launcher._validate_downloaded_manifest(
        local_root,
        remote=remote,
        materialized=materialized,
        expected_attempt_id="exp004-capacity-fixture",
    )
    artifact.write_bytes(b'{"verdict":"tampered"}\n')
    with pytest.raises(ValueError, match="failed integrity"):
        launcher._validate_downloaded_manifest(
            local_root,
            remote=remote,
            materialized=materialized,
            expected_attempt_id="exp004-capacity-fixture",
        )


def test_terminal_failure_retains_full_conservative_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _load(LAUNCHER, "capacity_launcher_terminal_failure")
    experiment_root = tmp_path / "experiment-004"
    ledger = experiment_root / "gpu-hours.json"
    ledger.parent.mkdir(parents=True)
    ledger.write_text('{"sentinel":"reservation-still-open"}\n')
    monkeypatch.setattr(launcher, "_EXPERIMENT_ROOT", experiment_root)

    launcher._record_terminal_failure(
        reservation_id="exp004-reservation-fixture",
        attempt_id="exp004-capacity-fixture",
        stage="result-validation-settlement",
        error=ValueError("manifest tamper"),
        elapsed_seconds=18.0,
    )

    failure = json.loads(
        (
            experiment_root / "calibration/logs/exp004-capacity-fixture-terminal-failure.json"
        ).read_text()
    )
    assert failure["conservatively_committed_gpu_seconds"] == 424.0
    assert not failure["reservation_cleared"]
    assert ledger.read_text() == '{"sentinel":"reservation-still-open"}\n'


def test_capacity_config_is_immutable_and_rejects_v9_transaction_fields() -> None:
    launcher = _load(LAUNCHER, "capacity_launcher_contract")
    payload = launcher._validate_config(CONFIG)
    invalid = dict(payload)
    invalid["gpu0_spike_request_rate_per_second"] = 64.0
    path = CONFIG.parent / ".capacity-invalid-fixture.json"
    try:
        path.write_text(json.dumps(invalid))
        with pytest.raises(ValueError, match="fields differ"):
            launcher._validate_config(path)
    finally:
        if path.exists():
            os.unlink(path)
