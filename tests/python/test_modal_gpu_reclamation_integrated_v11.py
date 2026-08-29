from __future__ import annotations

import ast
import hashlib
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest
from pydantic import ValidationError

from sloforge.helix.characterization.gpu_reclamation_methodology import (
    Experiment004GpuHourLedger,
    reserve_gpu_invocation,
)
from sloforge.helix.characterization.gpu_reclamation_v11_methodology import (
    Experiment004V11IntegratedConfig,
    Experiment004V11TargetedSourceIdentityConfig,
    validate_bound_artifact,
)

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "experiments/branchfabric"
LAUNCHER = EXPERIMENTS / "modal_gpu_reclamation_integrated_v11.py"


def _module() -> ModuleType:
    sys.path.insert(0, str(EXPERIMENTS))
    try:
        import modal_gpu_reclamation_integrated_v11

        return modal_gpu_reclamation_integrated_v11
    finally:
        sys.path.remove(str(EXPERIMENTS))


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _config(**updates: object) -> Experiment004V11IntegratedConfig:
    payload: dict[str, object] = {
        "attempt_id": "exp004-v11-integrated-s41-a",
        "seed": 41,
        "offline_gate_manifest": "artifacts/offline.json",
        "offline_gate_manifest_sha256": "1" * 64,
        "micro_validation_artifact": "artifacts/micro.json",
        "micro_validation_sha256": "2" * 64,
        "post_micro_review_manifest": "artifacts/reviews.json",
        "post_micro_review_manifest_sha256": "3" * 64,
        "budget_authorization": "artifacts/budget.json",
        "budget_authorization_sha256": "4" * 64,
        "ledger_sha256_before_reservation": "5" * 64,
    }
    payload.update(updates)
    return Experiment004V11IntegratedConfig.model_validate(payload)


def _targeted_config(**updates: object) -> Experiment004V11TargetedSourceIdentityConfig:
    payload = _config().model_dump(mode="json")
    payload.update(
        {
            "schema_version": (
                "sloforge.branchfabric.experiment-004-v11-targeted-source-identity-config/v1"
            ),
            "execution_mode": "targeted-source-identity-v11",
            "attempt_id": "exp004-v11-targeted-identity-s41-a",
            "terminal_phase": "SOURCE_ALLOCATION_IDENTITY_GATE",
            "full_export_authorized": False,
            "source_release_authorized": False,
            "maximum_wall_seconds": 300.0,
        }
    )
    payload.update(updates)
    return Experiment004V11TargetedSourceIdentityConfig.model_validate(payload)


def _row(
    index: int,
    uuid: str,
    *,
    name: str = "NVIDIA A100-SXM4-80GB",
) -> dict[str, object]:
    return {
        "index": index,
        "uuid": uuid,
        "name": name,
        "driver_version": "580.95.05",
        "memory_total_mib": 81_920,
        "memory_used_mib": 4,
        "utilization_percent": 0,
    }


def test_modal_surface_is_exact_two_a100_588s_and_preflight_gated() -> None:
    source = LAUNCHER.read_text()
    tree = ast.parse(source)
    constants = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"GPU_REQUEST", "GPU_COUNT", "GPU_FUNCTION_TIMEOUT_SECONDS"}
    }
    assert constants == {
        "GPU_REQUEST": "A100-80GB:2",
        "GPU_COUNT": 2,
        "GPU_FUNCTION_TIMEOUT_SECONDS": 588,
    }
    assert "SLOFORGE_GPU_BUDGET_USD" in source
    assert "SLOFORGE_EXP004_RESERVATION_ID" in source
    assert "SLOFORGE_MODAL_PREFLIGHT_TOKEN" in source
    assert "modal.Secret.from_dict" in source
    assert "secrets=[_launch_token_secret]" in source
    assert "retries=0" in source
    assert "max_containers=1" in source
    assert "single_use_containers=True" in source
    assert "create_if_missing=False" in source
    assert "gpu_reclamation_integrated_controller_v11" in source
    assert "gpu_reclamation_integrated_worker_v11.py" in source
    assert "modal_gpu_reclamation.py" not in source
    assert "modal_real_gpu_cow" not in source
    assert "importlib.import_module" not in source
    assert 'modal.Image.from_registry(CUDA_IMAGE, add_python="3.12")' in source
    assert 'f"vllm=={VLLM_VERSION}"' in source
    assert 'f"torch=={TORCH_VERSION}"' in source
    assert (
        '.add_local_dir(LOCAL_REPOSITORY_ROOT / "tests", "/opt/sloforge/tests", copy=True)'
        in source
    )
    assert 'final_gate_root = LOCAL_EXPERIMENT_ROOT / "v11-final"' in source
    assert '"integrated/authorization/ledger-snapshot-before-attempt-f.json"' in source
    assert 'raise FileNotFoundError("Attempt-F immutable ledger snapshot is absent")' in source
    assert 'f"{bundled}/v11-final"' in source
    assert '.add_local_file(LOCAL_EXPERIMENT_ROOT / "gpu-hours.json"' not in source
    assert "exp004-v11-integrated-s41-b/postflight-cleanup-and-settlement.json" in source
    assert "exp004-v11-integrated-s41-a/postflight-cleanup-and-settlement.json" not in source
    assert source.count("create_if_missing=False") == 2
    assert "create_if_missing=True" not in source
    assert source.count("validate_bound_artifact(") >= 2
    assert 'controller.get("in_function_cleanup")' in source
    assert '"in_function_cleanup_artifact": "in_function_cleanup.json"' in source
    assert '"launcher_owned_child_processes": 0' not in source
    assert '"launcher_profiler_count": 0' not in source
    for field in (
        "offline_gate_manifest",
        "micro_validation_artifact",
        "post_micro_review_manifest",
        "budget_authorization",
    ):
        assert f"config.{field}" in source


def test_targeted_modal_surface_is_distinct_bounded_and_single_use() -> None:
    module = _module()
    config = _targeted_config()
    assert module._authorized_wall_seconds(config) == 300.0
    source = LAUNCHER.read_text()
    assert "TARGETED_GPU_FUNCTION_TIMEOUT_SECONDS = 300" in source
    assert "_run_targeted_identity_function = app.function(" in source
    targeted_definition = source[source.index("_run_targeted_identity_function = app.function(") :]
    assert "timeout=TARGETED_GPU_FUNCTION_TIMEOUT_SECONDS" in targeted_definition
    assert "retries=0" in targeted_definition
    assert "max_containers=1" in targeted_definition
    assert "single_use_containers=True" in targeted_definition


def test_targeted_remote_authorization_and_ledger_reservation_are_exact_600_gpu_seconds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    base_ledger = Experiment004GpuHourLedger(
        hard_additional_gpu_seconds=21_600.0,
        consumed_additional_gpu_seconds=0.0,
    )
    pre_reservation_bytes = module._ledger_file_bytes(base_ledger)
    config = _targeted_config(ledger_sha256_before_reservation=_sha_bytes(pre_reservation_bytes))
    config_sha256 = _sha_bytes(module._canonical_bytes(config))
    ledger, _preflight = reserve_gpu_invocation(
        base_ledger,
        reservation_id="exp004-v11-targeted-reservation",
        invocation_id=config.attempt_id,
        maximum_wall_seconds=300.0,
        config_sha256=config_sha256,
        gpu_count=2,
    )
    experiment_root = tmp_path / "experiment-004"
    experiment_root.mkdir()
    (experiment_root / "gpu-hours.json").write_bytes(module._ledger_file_bytes(ledger))
    monkeypatch.setattr(module, "LOCAL_EXPERIMENT_ROOT", experiment_root)
    assert len(module._validate_local_reservation(config, "exp004-v11-targeted-reservation")) == 64
    payload = {
        "reservation_id": "exp004-v11-targeted-reservation",
        "reservation_commitment_sha256": "a" * 64,
        "config_sha256": config_sha256,
        "preflight_token_sha256": _sha_bytes(module._LAUNCH_TOKEN.encode()),
        "requested_gpu": "A100-80GB",
        "gpu_count": 2,
        "maximum_wall_seconds": 300.0,
        "maximum_gpu_seconds": 600.0,
        "budget_usd": 40.0,
    }
    authorization = module._validate_authorization(config, payload, authorized_budget_usd=40.0)
    assert authorization.maximum_gpu_seconds == 600.0
    with pytest.raises(RuntimeError, match="exactly 300s"):
        module._validate_authorization(
            config,
            payload | {"maximum_gpu_seconds": 1_176.0},
            authorized_budget_usd=40.0,
        )


def test_image_layout_materializes_all_content_addressed_test_bindings(
    tmp_path: Path,
) -> None:
    source = LAUNCHER.read_text()
    assert (
        '.add_local_dir(LOCAL_REPOSITORY_ROOT / "tests", "/opt/sloforge/tests", copy=True)'
        in source
    )
    remote_root = tmp_path / "opt/sloforge"
    references = (
        "tests/python/test_gpu_reclamation_integrated_worker_v11.py",
        "tests/python/test_gpu_reclamation_integrated_controller_v11.py",
        "tests/python/test_modal_gpu_reclamation_integrated_v11.py",
    )
    for reference in references:
        local = ROOT / reference
        remote = remote_root / reference
        remote.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local, remote)
        expected = hashlib.sha256(local.read_bytes()).hexdigest()
        assert validate_bound_artifact(
            remote_root,
            reference=reference,
            expected_sha256=expected,
        ) == remote.resolve(strict=True)


def test_image_layout_materializes_all_replacement_evidence_bindings(
    tmp_path: Path,
) -> None:
    sys.path.insert(0, str(EXPERIMENTS))
    try:
        import gpu_reclamation_integrated_controller_v11 as controller
    finally:
        sys.path.remove(str(EXPERIMENTS))

    source = LAUNCHER.read_text()
    cleanup_reference = controller.REPLACEMENT_EVIDENCE_BINDINGS["prior_attempt_cleanup"]
    assert "postflight-cleanup-and-settlement.json" in source
    assert "gpu_image.add_local_file(source, destination, copy=True)" in source

    remote_root = tmp_path / "opt/sloforge"
    references = (
        *controller.INTEGRATED_PRELAUNCH_BINDINGS.values(),
        *controller.REPLACEMENT_EVIDENCE_BINDINGS.values(),
    )
    assert cleanup_reference in references
    for reference in references:
        local = ROOT / reference
        assert local.is_file(), reference
        remote = remote_root / reference
        remote.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local, remote)
        expected = hashlib.sha256(local.read_bytes()).hexdigest()
        assert validate_bound_artifact(
            remote_root,
            reference=reference,
            expected_sha256=expected,
        ) == remote.resolve(strict=True)


def test_image_layout_bundles_exact_frozen_v10_control_replay_closure(
    tmp_path: Path,
) -> None:
    module = _module()
    sys.path.insert(0, str(EXPERIMENTS))
    try:
        import gpu_reclamation_integrated_controller_v11 as controller
    finally:
        sys.path.remove(str(EXPERIMENTS))

    expected = tuple(controller.CONTROL_GATE_FROZEN_V10_BINDINGS.values())
    assert expected == module.FROZEN_V10_CONTROL_REPLAY_IMAGE_BINDINGS
    assert len(expected) == 3
    source = LAUNCHER.read_text()
    assert "for reference, expected_sha256 in FROZEN_V10_CONTROL_REPLAY_IMAGE_BINDINGS" in source
    assert 'f"/opt/sloforge/{reference}"' in source

    for reference, expected_sha256 in expected:
        local = ROOT / reference
        assert (
            module._require_image_file_binding(
                local,
                expected_sha256=expected_sha256,
            )
            == local
        )
        remote = tmp_path / "opt/sloforge" / reference
        remote.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local, remote)
        assert validate_bound_artifact(
            tmp_path / "opt/sloforge",
            reference=reference,
            expected_sha256=expected_sha256,
        ) == remote.resolve(strict=True)


def test_image_binding_fails_closed_on_absence_symlink_and_hash_drift(
    tmp_path: Path,
) -> None:
    module = _module()
    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError, match="absent or symlinked"):
        module._require_image_file_binding(missing, expected_sha256="0" * 64)

    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}\n")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        module._require_image_file_binding(artifact, expected_sha256="0" * 64)

    link = tmp_path / "link.json"
    link.symlink_to(artifact)
    with pytest.raises(FileNotFoundError, match="absent or symlinked"):
        module._require_image_file_binding(
            link,
            expected_sha256=_sha_bytes(artifact.read_bytes()),
        )


def test_remote_authorization_binds_config_token_and_full_device_seconds() -> None:
    module = _module()
    config = _config()
    payload = {
        "reservation_id": "exp004-v11-integrated-reservation",
        "reservation_commitment_sha256": "a" * 64,
        "config_sha256": _sha_bytes(module._canonical_bytes(config)),
        "preflight_token_sha256": _sha_bytes(module._LAUNCH_TOKEN.encode()),
        "requested_gpu": "A100-80GB",
        "gpu_count": 2,
        "maximum_wall_seconds": 588.0,
        "maximum_gpu_seconds": 1176.0,
        "budget_usd": 40.0,
    }
    authorization = module._validate_authorization(config, payload, authorized_budget_usd=40.0)
    assert authorization.maximum_gpu_seconds == 2 * authorization.maximum_wall_seconds

    with pytest.raises(RuntimeError, match="immutable config"):
        module._validate_authorization(
            config,
            {**payload, "config_sha256": "b" * 64},
            authorized_budget_usd=40.0,
        )
    with pytest.raises(RuntimeError, match="exactly 588s"):
        module._validate_authorization(
            config,
            {**payload, "maximum_gpu_seconds": 588.0},
            authorized_budget_usd=40.0,
        )
    with pytest.raises(RuntimeError, match="exceeds"):
        module._validate_authorization(
            config,
            {**payload, "budget_usd": 40.01},
            authorized_budget_usd=40.0,
        )
    with pytest.raises(ValidationError):
        module.IntegratedV11RemoteAuthorization.model_validate({**payload, "gpu_count": 1})


def test_local_ledger_reservation_is_atomic_and_exact_two_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    base_ledger = Experiment004GpuHourLedger(
        hard_additional_gpu_seconds=21_600.0,
        consumed_additional_gpu_seconds=0.0,
    )
    pre_reservation_bytes = module._ledger_file_bytes(base_ledger)
    pre_reservation_sha256 = _sha_bytes(pre_reservation_bytes)
    config = _config(ledger_sha256_before_reservation=pre_reservation_sha256)
    config_sha256 = _sha_bytes(module._canonical_bytes(config))
    ledger, _preflight = reserve_gpu_invocation(
        base_ledger,
        reservation_id="exp004-v11-integrated-reservation",
        invocation_id=config.attempt_id,
        maximum_wall_seconds=588.0,
        config_sha256=config_sha256,
        gpu_count=2,
    )
    experiment_root = tmp_path / "experiment-004"
    experiment_root.mkdir()
    (experiment_root / "gpu-hours.json").write_bytes(module._ledger_file_bytes(ledger))
    monkeypatch.setattr(module, "LOCAL_EXPERIMENT_ROOT", experiment_root)

    commitment = module._validate_local_reservation(config, "exp004-v11-integrated-reservation")
    assert len(commitment) == 64
    assert ledger.committed_additional_gpu_seconds == 1_176.0

    drifted_config = _config(ledger_sha256_before_reservation="0" * 64)
    drifted_ledger, _ = reserve_gpu_invocation(
        base_ledger,
        reservation_id="exp004-v11-integrated-reservation",
        invocation_id=drifted_config.attempt_id,
        maximum_wall_seconds=588.0,
        config_sha256=_sha_bytes(module._canonical_bytes(drifted_config)),
        gpu_count=2,
    )
    (experiment_root / "gpu-hours.json").write_bytes(module._ledger_file_bytes(drifted_ledger))
    with pytest.raises(RuntimeError, match="pre-reservation ledger commitment"):
        module._validate_local_reservation(drifted_config, "exp004-v11-integrated-reservation")

    wrong, _ = reserve_gpu_invocation(
        base_ledger,
        reservation_id="exp004-v11-integrated-reservation",
        invocation_id=config.attempt_id,
        maximum_wall_seconds=587.0,
        config_sha256=config_sha256,
        gpu_count=2,
    )
    (experiment_root / "gpu-hours.json").write_bytes(module._ledger_file_bytes(wrong))
    with pytest.raises(RuntimeError, match="differs"):
        module._validate_local_reservation(config, "exp004-v11-integrated-reservation")


def test_real_authoritative_ledger_serialization_round_trips_exactly() -> None:
    module = _module()
    path = ROOT / "artifacts/branchfabric/gpu-validation/experiment-004/gpu-hours.json"
    ledger = Experiment004GpuHourLedger.model_validate_json(path.read_text(), strict=True)
    assert module._ledger_file_bytes(ledger) == path.read_bytes()


def test_hardware_comparability_requires_two_stable_identical_sxm4_devices() -> None:
    module = _module()
    sxm = [_row(0, "GPU-a"), _row(1, "GPU-b")]
    comparable = module._hardware_comparability({"inventory_before": sxm, "inventory_after": sxm})
    assert comparable["direct_v10_timing_comparable"] is True
    assert comparable["byte_and_amplification_comparable"] is True

    pcie = [
        _row(0, "GPU-a", name="NVIDIA A100 80GB PCIe"),
        _row(1, "GPU-b", name="NVIDIA A100 80GB PCIe"),
    ]
    confounded = module._hardware_comparability({"inventory_before": pcie, "inventory_after": pcie})
    assert confounded["direct_v10_timing_comparable"] is False
    assert "cannot pin" in confounded["timing_comparability_reason"]

    with pytest.raises(RuntimeError, match="exactly two"):
        module._hardware_comparability({"inventory_before": sxm[:1], "inventory_after": sxm[:1]})
    with pytest.raises(RuntimeError, match="identical"):
        module._hardware_comparability(
            {
                "inventory_before": [sxm[0], pcie[1]],
                "inventory_after": [sxm[0], pcie[1]],
            }
        )
    changed = [dict(sxm[0]), dict(sxm[1])]
    changed[1]["driver_version"] = "different"
    with pytest.raises(RuntimeError, match="identity changed"):
        module._hardware_comparability({"inventory_before": sxm, "inventory_after": changed})


def _cleanup_evidence(module: ModuleType) -> dict[str, object]:
    return {
        "schema_version": "sloforge.branchfabric.in-function-cleanup/v1",
        "status": "PASS",
        "pass": True,
        "lifecycle": [
            {"phase": phase, "observed_at_monotonic_ns": index}
            for index, phase in enumerate(module.PROCESS_LIFECYCLE_PHASES)
        ],
        "required_lifecycle": list(module.PROCESS_LIFECYCLE_PHASES),
        "parent_pid": 10,
        "parent_pgid": 10,
        "parent_sid": 10,
        "child_subreaper": {
            "supported": False,
            "enabled_for_experiment": False,
            "restored_before_return": True,
            "error": None,
        },
        "owned_children": [
            {
                "pid": pid,
                "ppid": 10,
                "pgid": pid,
                "sid": pid,
                "role": role,
                "process_kind": "worker",
                "termination_signal": None,
                "exit_status": 0,
                "reap_timestamp_monotonic_ns": 20 + index,
                "reap_method": "popen-waitpid",
            }
            for index, (role, pid) in enumerate((("serving", 11), ("rollout", 12)))
        ],
        "termination_actions": [],
        "forced_kills": [],
        "forced_kill_required": False,
        "surviving_children": [],
        "surviving_process_groups": [],
        "profiler_processes_after": [],
        "serving_workers_after": [],
        "rollout_workers_after": [],
        "resource_tracker_processes_after": [],
        "zombie_processes_after": [],
        "leaked_ipc_resources": [],
        "leaked_threads": [],
        "compute_processes_after": [],
        "cleanup_errors": [],
        "parent_reaped_all_owned_children": True,
        "owned_ipc_resources_released": True,
        "pipes_closed": True,
        "cuda_released": True,
    }


def _controller_with_cleanup(cleanup: dict[str, object]) -> dict[str, object]:
    return {
        "status": "succeeded",
        "worker_pids": {"serving": 11, "rollout": 12},
        "worker_process_groups": {"serving": 11, "rollout": 12},
        "worker_session_ids": {"serving": 11, "rollout": 12},
        "compute_processes_after": [],
        "in_function_cleanup": cleanup,
    }


def _pre_worker_controller_with_cleanup(
    cleanup: dict[str, object],
) -> dict[str, object]:
    return {
        "status": "failed",
        "cleanup_scope": "PRE_WORKER_PREFLIGHT",
        "failure_stage": "SEALED_EVIDENCE",
        "sealed_evidence": None,
        "inventory_before": [],
        "inventory_after": [],
        "stable_physical_gpu_identity": False,
        "worker_pids": {},
        "worker_process_groups": {},
        "worker_session_ids": {},
        "worker_returncodes": {},
        "engine_start_evidence": [],
        "readiness_evidence": [],
        "readiness_deadline_ns": None,
        "sanity_guard_pair": None,
        "worker_results": [],
        "controller_error": {"type": "FileNotFoundError", "message": "missing snapshot"},
        "cleanup_error": None,
        "cleanup_actions": [],
        "cuda_clean_import_audits": [{"cuda_clean": True}],
        "compute_processes_after": [],
        "in_function_cleanup": cleanup,
    }


def test_launcher_requires_matching_pass_complete_in_function_cleanup(
    tmp_path: Path,
) -> None:
    module = _module()
    cleanup = _cleanup_evidence(module)
    artifact = tmp_path / "in_function_cleanup.json"
    artifact.write_text(json.dumps(cleanup))
    assert (
        module._require_in_function_cleanup(
            _controller_with_cleanup(cleanup),
            work_root=tmp_path,
        )
        == cleanup
    )

    for field, invalid in (
        ("pass", False),
        ("surviving_children", [{"pid": 7}]),
        ("profiler_processes_after", [{"pid": 8}]),
        ("zombie_processes_after", [{"pid": 9}]),
        ("parent_reaped_all_owned_children", False),
        ("cuda_released", False),
    ):
        rejected = {**cleanup, field: invalid}
        artifact.write_text(json.dumps(rejected))
        with pytest.raises(RuntimeError, match="not PASS-complete"):
            module._require_in_function_cleanup(
                _controller_with_cleanup(rejected),
                work_root=tmp_path,
            )

    artifact.write_text(json.dumps(cleanup))
    for field, invalid in (
        ("worker_pids", {"serving": 11}),
        ("worker_process_groups", {"serving": 11, "rollout": 99}),
        ("worker_session_ids", {"serving": 99, "rollout": 12}),
        ("compute_processes_after", [{"pid": 13}]),
        ("cleanup_scope", "PRE_WORKER_PREFLIGHT"),
        ("cleanup_scope", "UNKNOWN"),
    ):
        controller = _controller_with_cleanup(cleanup)
        controller[field] = invalid
        with pytest.raises(RuntimeError, match="not PASS-complete"):
            module._require_in_function_cleanup(controller, work_root=tmp_path)


def test_launcher_accepts_only_exact_pre_worker_cleanup_scope(tmp_path: Path) -> None:
    module = _module()
    cleanup = _cleanup_evidence(module)
    cleanup["owned_children"] = []
    artifact = tmp_path / "in_function_cleanup.json"
    artifact.write_text(json.dumps(cleanup))
    controller = _pre_worker_controller_with_cleanup(cleanup)

    assert module._require_in_function_cleanup(controller, work_root=tmp_path) == cleanup

    mutations = (
        lambda value: value.update(worker_pids={"serving": 11}),
        lambda value: value.update(inventory_before=[{"uuid": "GPU-unexpected"}]),
        lambda value: value.update(engine_start_evidence=[{"pid": 11}]),
        lambda value: value.update(cleanup_scope="UNKNOWN"),
        lambda value: value.update(controller_error=None),
    )
    for mutate in mutations:
        rejected_controller = _pre_worker_controller_with_cleanup(cleanup)
        mutate(rejected_controller)
        with pytest.raises(RuntimeError, match="not PASS-complete"):
            module._require_in_function_cleanup(
                rejected_controller,
                work_root=tmp_path,
            )

    rejected_cleanup = {**cleanup, "owned_children": [{"pid": 99}]}
    artifact.write_text(json.dumps(rejected_cleanup))
    rejected_controller = _pre_worker_controller_with_cleanup(rejected_cleanup)
    with pytest.raises(RuntimeError, match="not PASS-complete"):
        module._require_in_function_cleanup(rejected_controller, work_root=tmp_path)


def test_launcher_canonicalizes_tuple_cleanup_evidence_before_comparison(
    tmp_path: Path,
) -> None:
    module = _module()
    cleanup = _cleanup_evidence(module)
    cleanup["required_lifecycle"] = tuple(module.PROCESS_LIFECYCLE_PHASES)
    cleanup["termination_actions"] = ()
    cleanup["forced_kills"] = ()
    cleanup["surviving_children"] = ()
    artifact = tmp_path / "in_function_cleanup.json"
    artifact.write_text(json.dumps(cleanup))
    controller = _controller_with_cleanup(cleanup)
    controller["compute_processes_after"] = ()

    observed = module._require_in_function_cleanup(
        controller,
        work_root=tmp_path,
    )

    assert observed["required_lifecycle"] == list(module.PROCESS_LIFECYCLE_PHASES)
    assert observed["termination_actions"] == []


def test_launcher_rejects_missing_or_contradictory_cleanup_artifact(tmp_path: Path) -> None:
    module = _module()
    cleanup = _cleanup_evidence(module)
    with pytest.raises(RuntimeError, match="omitted in-function"):
        module._require_in_function_cleanup({"status": "succeeded"}, work_root=tmp_path)
    with pytest.raises(RuntimeError, match=r"omitted in_function_cleanup\.json"):
        module._require_in_function_cleanup(
            _controller_with_cleanup(cleanup),
            work_root=tmp_path,
        )

    artifact = tmp_path / "in_function_cleanup.json"
    artifact.write_text(json.dumps({**cleanup, "status": "FAIL"}))
    with pytest.raises(RuntimeError, match="contradicts"):
        module._require_in_function_cleanup(
            _controller_with_cleanup(cleanup),
            work_root=tmp_path,
        )


def test_all_four_bound_artifacts_and_budget_authorization_are_verified(
    tmp_path: Path,
) -> None:
    module = _module()
    paths = {
        "offline": tmp_path / "artifacts/offline.json",
        "micro": tmp_path / "artifacts/micro.json",
        "reviews": tmp_path / "artifacts/reviews.json",
        "budget": tmp_path / "artifacts/budget.json",
    }
    for key, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, object] = (
            {
                "authorized_gpu_budget_usd": 40.0,
                "authorized_cumulative_gpu_seconds": 21_600.0,
            }
            if key == "budget"
            else {"status": "PASS", "kind": key}
        )
        path.write_text(json.dumps(payload))
    relative = {key: str(path.relative_to(tmp_path)) for key, path in paths.items()}
    digest = {key: _sha_bytes(path.read_bytes()) for key, path in paths.items()}
    config = _config(
        offline_gate_manifest=relative["offline"],
        offline_gate_manifest_sha256=digest["offline"],
        micro_validation_artifact=relative["micro"],
        micro_validation_sha256=digest["micro"],
        post_micro_review_manifest=relative["reviews"],
        post_micro_review_manifest_sha256=digest["reviews"],
        budget_authorization=relative["budget"],
        budget_authorization_sha256=digest["budget"],
    )
    budget = module._validate_config_artifacts(config, repository_root=tmp_path)
    assert budget["authorized_gpu_budget_usd"] == 40.0

    paths["micro"].write_text('{"status":"DRIFT"}')
    with pytest.raises(ValueError, match="hash mismatch"):
        module._validate_config_artifacts(config, repository_root=tmp_path)


def test_immutable_result_publish_has_manifest_and_refuses_overwrite(tmp_path: Path) -> None:
    module = _module()

    class Volume:
        def __init__(self) -> None:
            self.commits = 0

        def commit(self) -> None:
            self.commits += 1

    work_root = tmp_path / "work"
    results_root = tmp_path / "results"
    staging = results_root / "attempt.staging"
    final = results_root / "attempt"
    work_root.mkdir()
    (work_root / "result.json").write_text('{"status":"provisional"}')
    volume = Volume()

    manifest_sha256 = module._publish_immutable_result(
        work_root=work_root,
        staging=staging,
        final=final,
        results_root=results_root,
        volume=volume,
        attempt_id="exp004-v11-integrated-s41-a",
    )
    assert manifest_sha256 == _sha_bytes((final / "REMOTE_MANIFEST.json").read_bytes())
    manifest = json.loads((final / "REMOTE_MANIFEST.json").read_text())
    assert manifest["remote_prefix"] == "attempt"
    assert manifest["artifacts"] == [
        {
            "relative_path": "result.json",
            "bytes": 24,
            "sha256": _sha_bytes((final / "result.json").read_bytes()),
        }
    ]
    assert not staging.exists()
    assert volume.commits == 1
    with pytest.raises(FileExistsError, match="already exists"):
        module._publish_immutable_result(
            work_root=work_root,
            staging=staging,
            final=final,
            results_root=results_root,
            volume=volume,
            attempt_id="exp004-v11-integrated-s41-a",
        )


@pytest.mark.parametrize("raw", [None, "", "0", "-1", "nan", "inf", "not-a-number"])
def test_paid_launch_budget_must_be_finite_and_positive(raw: str | None) -> None:
    module = _module()
    with pytest.raises(RuntimeError, match="finite and positive"):
        module._positive_budget(raw)


def test_modal_sdk_version_is_checked_before_local_spawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    monkeypatch.setattr(module.modal, "__version__", "1.5.2")
    with pytest.raises(RuntimeError, match=r"must be exactly 1\.5\.3"):
        module._require_modal_sdk_version()
    source = LAUNCHER.read_text()
    main = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    assert "_require_modal_sdk_version()" in ast.get_source_segment(source, main)


def test_integrated_and_targeted_gpu_functions_have_distinct_modal_tags() -> None:
    source = LAUNCHER.read_text()
    assert 'name="run-integrated-v11"' in source
    assert 'name="run-targeted-source-identity-v11"' in source
