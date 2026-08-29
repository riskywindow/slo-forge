from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_CAMPAIGN_PATH = _ROOT / "experiments/branchfabric/gpu_validation_campaign.py"
_SPEC = importlib.util.spec_from_file_location(
    "sloforge_gpu_validation_campaign_tests",
    _CAMPAIGN_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
_CAMPAIGN = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _CAMPAIGN
_SPEC.loader.exec_module(_CAMPAIGN)


def _gpu_sample(
    memory_used_mib: int,
    *,
    uuid: str = "GPU-campaign-test",
    name: str = "NVIDIA A100 80GB PCIe",
    driver_version: str = "test-driver",
    processes: list[dict[str, object]] | None = None,
) -> dict[str, Any]:
    return {
        "observed_at_utc": "2026-08-11T00:00:00+00:00",
        "observed_at_monotonic_ns": 1,
        "gpu": {
            "index": 0,
            "uuid": uuid,
            "name": name,
            "driver_version": driver_version,
            "memory_total_mib": 81_920,
            "memory_used_mib": memory_used_mib,
            "utilization_percent": 0,
        },
        "compute_processes": [] if processes is None else processes,
        "commands": {},
    }


def _identity() -> dict[str, Any]:
    return {
        "index": 0,
        "uuid": "GPU-campaign-test",
        "name": "NVIDIA A100 80GB PCIe",
        "driver_version": "test-driver",
        "memory_total_mib": 81_920,
    }


def _successful_controller_result(
    config_payload: dict[str, Any],
    *,
    controller_threshold_mib: int = 4096,
) -> dict[str, Any]:
    identity = _identity()
    return {
        "status": "succeeded",
        "actual_gpu": identity,
        "summary": {
            "attempt_id": config_payload["attempt_id"],
            "path": config_payload["baseline_mode"],
        },
        "child_manifest": {
            "status": "succeeded",
            "gpu_uuid": identity["uuid"],
            "semantic_invariants": {"fixture": True},
            "final_runtime_assigned_kv_bytes": 0,
        },
        "cleanup_gate": {
            "GPU_RUNTIME_LIFECYCLE_CLEAN": True,
            "cleanup_gate": "PASS",
            "forced_gpu_process_kill_required": False,
            "cleanup_threshold_mib": controller_threshold_mib,
        },
        "controller_manifest": {
            "status": "succeeded",
            "attempt_id": config_payload["attempt_id"],
            "cleanup_threshold_mib": controller_threshold_mib,
        },
        "error": None,
    }


def _config(**overrides: Any):
    values = {
        "campaign_id": "exp003-local-campaign",
        "randomized_order_seed": 20260811,
        "baseline_sample_interval_seconds": 0.0,
        "cleanup_sample_interval_seconds": 0.0,
        "cleanup_timeout_seconds": 1.0,
        "maximum_cleanup_samples": 6,
    }
    values.update(overrides)
    return _CAMPAIGN.CampaignConfig(**values)


def test_campaign_module_is_standard_library_only() -> None:
    tree = ast.parse(_CAMPAIGN_PATH.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
            imported_roots.add((node.module or "").partition(".")[0])
    assert imported_roots <= sys.stdlib_module_names


def test_primary_pair_plan_is_deterministic_and_records_order_provenance() -> None:
    config = _config()
    first = _CAMPAIGN.plan_primary_pairs(config)
    second = _CAMPAIGN.plan_primary_pairs(config)

    assert first == second
    assert [pair.seed for pair in first] == [41, 73, 113]
    assert first[0].realized_order == ("independent", "shared_root")
    assert first[1].realized_order == ("shared_root", "independent")
    assert set(first[2].realized_order) == {"independent", "shared_root"}
    assert first[2].order_seed == 20260811
    assert first[2].to_dict()["order_algorithm"] == _CAMPAIGN.ORDER_ALGORITHM
    json.dumps(config.to_dict())
    json.dumps([pair.to_dict() for pair in first])


@pytest.mark.parametrize(
    "overrides",
    [
        {"seeds": (41, 41, 113)},
        {"seeds": (41, 73)},
        {"tracing_level": "verbose"},
        {"campaign_id": "../escape"},
        {"maximum_cleanup_samples": 2},
    ],
)
def test_campaign_config_fails_closed(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        _config(**overrides)


def test_campaign_runs_six_children_serially_with_one_immutable_gpu_gate(
    tmp_path: Path,
) -> None:
    events: list[tuple[str, Any]] = []
    sample_count = 0

    def sampler(expected_uuid: str | None) -> dict[str, Any]:
        nonlocal sample_count
        sample_count += 1
        events.append(("sample", expected_uuid))
        baseline = [12, 4, 8]
        memory_mib = baseline[sample_count - 1] if sample_count <= 3 else 1000
        return _gpu_sample(memory_mib)

    payloads: list[dict[str, Any]] = []
    roots: list[Path] = []

    def controller_runner(**kwargs: Any) -> dict[str, Any]:
        payload = kwargs["config_payload"]
        trial_root = kwargs["work_root"]
        assert not trial_root.exists()
        trial_root.mkdir()
        payloads.append(dict(payload))
        roots.append(trial_root)
        events.append(("runner", payload["attempt_id"]))
        return _successful_controller_result(payload)

    config = _config()
    work_root = tmp_path / "campaign"
    manifest = _CAMPAIGN.run_campaign(
        config=config,
        base_config_payload={
            "schema_version": "sloforge.branchfabric.modal-real-gpu-cow-config/v1",
            "runtime": "vllm",
        },
        work_root=work_root,
        worker_path=tmp_path / "worker.py",
        model_snapshot=tmp_path / "model",
        controller_runner=controller_runner,
        gpu_sampler=sampler,
        sleep=lambda _: None,
    )

    expected_modes = [
        mode for pair in _CAMPAIGN.plan_primary_pairs(config) for mode in pair.realized_order
    ]
    assert manifest["status"] == "succeeded"
    assert manifest["planned_children"] == 6
    assert manifest["started_children"] == 6
    assert manifest["completed_children"] == 6
    assert manifest["gpu_identity"] == _identity()
    assert manifest["campaign_hbm_baseline"]["baseline_median_mib"] == 8
    assert manifest["authoritative_cleanup_threshold_mib"] == 1032
    assert [payload["baseline_mode"] for payload in payloads] == expected_modes
    assert [payload["seed"] for payload in payloads] == [41, 41, 73, 73, 113, 113]
    assert all(payload["tracing_level"] == "minimal" for payload in payloads)
    assert len(roots) == len(set(roots)) == 6

    runner_positions = [index for index, event in enumerate(events) if event[0] == "runner"]
    assert runner_positions == [6, 13, 20, 27, 34, 41]
    assert events[0] == ("sample", None)
    assert all(event[1] == "GPU-campaign-test" for event in events[1:] if event[0] == "sample")

    for pair in manifest["pair_manifests"]:
        assert pair["status"] == "succeeded"
        assert pair["same_gpu_identity_verified"] is True
        assert pair["authoritative_cleanup_threshold_mib"] == 1032
        assert len(pair["trials"]) == 2
        for trial in pair["trials"]:
            assert trial["pre_child_gate"]["cleanup_threshold_mib"] == 1032
            assert trial["campaign_postflight_gate"]["cleanup_threshold_mib"] == 1032
            assert trial["campaign_postflight_gate"]["passed"] is True

    on_disk = json.loads((work_root / "campaign-manifest.json").read_text())
    assert on_disk == manifest
    assert (work_root / "campaign-plan.json").read_bytes().endswith(b"\n")
    assert len(list((work_root / "pairs").glob("*/pair-manifest.json"))) == 3
    json.dumps(manifest)


@pytest.mark.parametrize(
    ("changed_field", "changed_value"),
    [("uuid", "GPU-drift"), ("driver_version", "different-driver")],
)
def test_campaign_aborts_before_next_child_on_gpu_identity_drift(
    tmp_path: Path,
    changed_field: str,
    changed_value: str,
) -> None:
    sample_count = 0
    controller_calls = 0

    def sampler(expected_uuid: str | None) -> dict[str, Any]:
        nonlocal sample_count
        sample_count += 1
        values: dict[str, Any] = {}
        # 3 campaign baseline + 3 pre + 3 post samples precede child two.
        if sample_count == 10:
            values[changed_field] = changed_value
        return _gpu_sample(8, **values)

    def controller_runner(**kwargs: Any) -> dict[str, Any]:
        nonlocal controller_calls
        controller_calls += 1
        return _successful_controller_result(kwargs["config_payload"])

    manifest = _CAMPAIGN.run_campaign(
        config=_config(),
        base_config_payload={},
        work_root=tmp_path / "campaign",
        worker_path=tmp_path / "worker.py",
        model_snapshot=tmp_path / "model",
        controller_runner=controller_runner,
        gpu_sampler=sampler,
        sleep=lambda _: None,
    )

    assert manifest["status"] == "failed"
    assert manifest["error"]["stage"] == "pre_child_gate"
    assert manifest["started_children"] == 1
    assert manifest["completed_children"] == 1
    assert controller_calls == 1
    pair = manifest["pair_manifests"][0]
    assert pair["status"] == "failed"
    assert pair["same_gpu_identity_verified"] is False
    assert [trial["status"] for trial in pair["trials"]] == ["succeeded", "failed"]


def test_campaign_cleanup_gate_cannot_ratchet_to_a_later_child_baseline(
    tmp_path: Path,
) -> None:
    sample_count = 0
    controller_calls = 0

    def sampler(expected_uuid: str | None) -> dict[str, Any]:
        nonlocal sample_count
        sample_count += 1
        if sample_count <= 3:
            return _gpu_sample(100)
        if sample_count <= 6:
            return _gpu_sample(1100)
        return _gpu_sample(1500)

    def controller_runner(**kwargs: Any) -> dict[str, Any]:
        nonlocal controller_calls
        controller_calls += 1
        # A locally recalculated 1100 + 1024 threshold would incorrectly pass
        # the 1500 MiB postflight samples.
        return _successful_controller_result(
            kwargs["config_payload"],
            controller_threshold_mib=2124,
        )

    manifest = _CAMPAIGN.run_campaign(
        config=_config(maximum_cleanup_samples=4),
        base_config_payload={},
        work_root=tmp_path / "campaign",
        worker_path=tmp_path / "worker.py",
        model_snapshot=tmp_path / "model",
        controller_runner=controller_runner,
        gpu_sampler=sampler,
        sleep=lambda _: None,
    )

    assert manifest["status"] == "failed"
    assert manifest["authoritative_cleanup_threshold_mib"] == 1124
    assert manifest["error"]["stage"] == "campaign_postflight_gate"
    assert manifest["started_children"] == 1
    assert manifest["completed_children"] == 0
    assert controller_calls == 1
    trial = manifest["pair_manifests"][0]["trials"][0]
    assert trial["pre_child_gate"]["cleanup_threshold_mib"] == 1124
    assert trial["controller"]["cleanup_gate"]["cleanup_threshold_mib"] == 2124
    assert trial["campaign_postflight_gate"]["passed"] is False
    assert [
        sample["gpu"]["memory_used_mib"] for sample in trial["campaign_postflight_gate"]["samples"]
    ] == [
        1500,
        1500,
        1500,
        1500,
    ]


def test_campaign_aborts_when_single_child_controller_cleanup_fails(
    tmp_path: Path,
) -> None:
    controller_calls = 0

    def controller_runner(**kwargs: Any) -> dict[str, Any]:
        nonlocal controller_calls
        controller_calls += 1
        result = _successful_controller_result(kwargs["config_payload"])
        result["status"] = "failed"
        result["cleanup_gate"]["GPU_RUNTIME_LIFECYCLE_CLEAN"] = False
        return result

    manifest = _CAMPAIGN.run_campaign(
        config=_config(),
        base_config_payload={},
        work_root=tmp_path / "campaign",
        worker_path=tmp_path / "worker.py",
        model_snapshot=tmp_path / "model",
        controller_runner=controller_runner,
        gpu_sampler=lambda expected_uuid: _gpu_sample(8),
        sleep=lambda _: None,
    )

    assert manifest["status"] == "failed"
    assert manifest["error"]["stage"] == "child_controller"
    assert manifest["started_children"] == 1
    assert manifest["completed_children"] == 0
    assert controller_calls == 1
