"""CUDA-clean two-process controller for Experiment 004 capacity calibration."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from gpu_reclamation_controller import (
    _compute_processes,
    _inventory,
    _require_no_compute_processes,
    _terminate_group,
    cuda_clean_import_audit,
)


def _canonical_bytes(value: Any) -> bytes:
    def json_default(item: Any) -> Any:
        model_dump = getattr(item, "model_dump", None)
        if callable(model_dump):
            return model_dump(mode="json")
        raise TypeError(f"Object of type {type(item).__name__} is not JSON serializable")

    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=json_default,
        )
        + "\n"
    ).encode()


def _write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(_canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())


def _wait_for_paths(
    paths: tuple[Path, ...],
    *,
    deadline_ns: int,
    phase: str,
    children: tuple[tuple[str, subprocess.Popen[bytes]], ...],
) -> None:
    while not all(path.is_file() for path in paths):
        failures = tuple(
            (device, process.returncode)
            for device, process in children
            if process.poll() not in {None, 0}
        )
        if failures:
            raise RuntimeError(f"capacity workers failed during {phase}: {failures}")
        if time.monotonic_ns() >= deadline_ns:
            missing = tuple(str(path) for path in paths if not path.is_file())
            raise TimeoutError(f"capacity calibration timed out during {phase}: {missing}")
        time.sleep(0.02)


def _validate_worker_completions(
    *,
    result_paths: tuple[Path, ...],
    expected_inventory: tuple[dict[str, Any], ...],
    returncodes: dict[str, int | None],
) -> tuple[dict[str, Any], ...]:
    """Require two identity-bound successful workers and clean process exits."""

    expected_devices = ("gpu0", "gpu1")
    if len(result_paths) != 2 or len(expected_inventory) != 2:
        raise RuntimeError("capacity completion validation requires exactly two workers")
    payloads: list[dict[str, Any]] = []
    for path, device, gpu in zip(result_paths, expected_devices, expected_inventory, strict=True):
        payload = json.loads(path.read_text())
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version")
            != "sloforge.branchfabric.capacity-worker-completion/v1"
            or payload.get("device") != device
            or payload.get("physical_gpu_uuid") != gpu.get("uuid")
            or payload.get("status") != "succeeded"
            or payload.get("error") is not None
        ):
            raise RuntimeError(f"capacity {device} worker completion failed identity or status")
        payloads.append(payload)
    if set(returncodes) != set(expected_devices) or set(returncodes.values()) != {0}:
        raise RuntimeError(f"capacity worker return codes were invalid: {returncodes}")
    return tuple(payloads)


def _validate_readiness_payloads(
    *,
    payloads: tuple[dict[str, Any], ...],
    expected_inventory: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Require two complete, identity-bound EngineReadinessEvidence objects."""

    if len(payloads) != 2 or len(expected_inventory) != 2:
        raise RuntimeError("BOTH_ENGINES_READY requires exactly two readiness objects")
    expected_devices = ("gpu0", "gpu1")
    validated: list[dict[str, Any]] = []
    for payload, device, gpu in zip(payloads, expected_devices, expected_inventory, strict=True):
        runtime = payload.get("runtime_evidence")
        compilation = payload.get("compilation_observation")
        probe = payload.get("probe")
        hbm = payload.get("hbm")
        state = payload.get("runtime_state")
        required_sizes = payload.get("required_cudagraph_batch_sizes")
        observed_sizes = payload.get("observed_cudagraph_capture_sizes")
        timestamps = tuple(
            payload.get(field)
            for field in (
                "engine_started_ns",
                "model_ready_ns",
                "compile_warmup_started_ns",
                "compile_warmup_ended_ns",
            )
        )
        if (
            payload.get("schema_version") != "sloforge.branchfabric.engine-readiness-evidence/v1"
            or payload.get("device") != device
            or payload.get("physical_gpu_uuid") != gpu.get("uuid")
            or payload.get("passed") is not True
            or any(isinstance(value, bool) or not isinstance(value, int) for value in timestamps)
            or not timestamps[0] <= timestamps[1] <= timestamps[2] < timestamps[3]
            or tuple(required_sizes or ()) != (1, 2, 4, 8, 16)
            or not set(required_sizes or ()).issubset(set(observed_sizes or ()))
            or payload.get("cudagraph_coverage_pass") is not True
            or payload.get("warmup_request_count") != 31
            or payload.get("warmup_output_target_verified") is not True
            or payload.get("verification_request_count") != 31
            or payload.get("verification_output_target_verified") is not True
        ):
            raise RuntimeError(f"capacity {device} readiness identity/warmup evidence failed")
        if (
            not isinstance(runtime, dict)
            or runtime.get("packages", {}).get("vllm") != "0.23.0"
            or runtime.get("packages", {}).get("torch") != "2.11.0"
            or "A100" not in str(runtime.get("gpu_name", ""))
            or not runtime.get("torch_cuda")
        ):
            raise RuntimeError(f"capacity {device} runtime evidence differs from its pins")
        if (
            not isinstance(compilation, dict)
            or compilation.get("engine_initialization_completed_before_warmup") is not True
            or compilation.get("no_active_compilation_event") is not True
            or tuple(compilation.get("measured_probe_compilation_events", ()))
        ):
            raise RuntimeError(f"capacity {device} measured probe observed compilation")
        if (
            not isinstance(probe, dict)
            or probe.get("requested_output_tokens") != 64
            or probe.get("emitted_output_tokens") != 64
            or probe.get("output_target_verified") is not True
            or probe.get("ttft_sane") is not True
            or not isinstance(probe.get("ttft_seconds"), (int, float))
            or isinstance(probe.get("ttft_seconds"), bool)
            or not 0.0 < float(probe["ttft_seconds"]) <= 2.0
            or not isinstance(probe.get("request_throughput_rps"), (int, float))
            or not math.isfinite(float(probe["request_throughput_rps"]))
            or float(probe["request_throughput_rps"]) <= 0.0
            or not isinstance(probe.get("output_token_throughput_tps"), (int, float))
            or not math.isfinite(float(probe["output_token_throughput_tps"]))
            or float(probe["output_token_throughput_tps"]) <= 0.0
        ):
            raise RuntimeError(f"capacity {device} readiness probe failed")
        hbm_samples = hbm.get("samples", ()) if isinstance(hbm, dict) else ()
        hbm_fields = (
            "nvml_device_used_bytes",
            "torch_allocated_bytes",
            "torch_reserved_bytes",
            "kv_pool_reserved_bytes",
            "kv_assigned_bytes",
        )
        hbm_recomputed_stable = len(hbm_samples) >= 3
        if hbm_recomputed_stable:
            for field in hbm_fields:
                values = tuple(sample.get(field) for sample in hbm_samples)
                if any(
                    isinstance(value, bool) or not isinstance(value, int) or value < 0
                    for value in values
                ):
                    hbm_recomputed_stable = False
                    break
                tolerance = 16 * 1024 * 1024 if field == "nvml_device_used_bytes" else 0
                if max(values) - min(values) > tolerance:
                    hbm_recomputed_stable = False
                    break
        if (
            not isinstance(hbm, dict)
            or hbm.get("stable") is not True
            or hbm.get("nvml_stability_tolerance_bytes") != 16 * 1024 * 1024
            or not all(isinstance(sample, dict) for sample in hbm_samples)
            or not hbm_recomputed_stable
        ):
            raise RuntimeError(f"capacity {device} HBM stability evidence failed")
        zero_fields = (
            "request_count",
            "running_requests",
            "waiting_requests",
            "skipped_waiting_requests",
            "queue_depth",
        )
        if (
            not isinstance(state, dict)
            or any(state.get(field) != 0 for field in zero_fields)
            or state.get("prefix_cache_reset") is not True
            or state.get("health") != "healthy"
            or state.get("metrics_available") is not True
            or payload.get("queue_empty_pass") is not True
        ):
            raise RuntimeError(f"capacity {device} idle/health/metrics evidence failed")
        validated.append(payload)
    return validated[0], validated[1]


def _validate_reset_payloads(
    *, payloads: tuple[dict[str, Any], ...], expected_probe_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(payloads) != 2:
        raise RuntimeError("capacity reset evidence requires exactly two worker shards")
    zero_fields = (
        "request_count",
        "running_requests",
        "waiting_requests",
        "skipped_waiting_requests",
        "queue_depth",
    )
    for payload in payloads:
        before = payload.get("state_before")
        after = payload.get("state_after")
        duration = payload.get("duration_seconds")
        if (
            payload.get("schema_version") != "sloforge.branchfabric.capacity-request-reset/v1"
            or payload.get("probe_id") != expected_probe_id
            or payload.get("engine_reloaded") is not False
            or payload.get("prefix_cache_reset") is not True
            or payload.get("passed") is not True
            or not isinstance(before, dict)
            or not isinstance(after, dict)
            or any(before.get(field) != 0 for field in zero_fields)
            or any(after.get(field) != 0 for field in zero_fields)
            or not isinstance(duration, (int, float))
            or isinstance(duration, bool)
            or not math.isfinite(float(duration))
            or float(duration) < 0.0
        ):
            raise RuntimeError("capacity request-state reset evidence failed")
    return payloads[0], payloads[1]


def run_capacity_calibration_controller(
    *,
    config_path: Path,
    work_root: Path,
    worker_path: Path,
    model_snapshot: Path,
) -> dict[str, Any]:
    """Load two engines once and conduct exactly the bounded adaptive search."""

    from sloforge.helix.characterization.gpu_capacity_calibration import (
        CapacityProbeRaw,
        CapacityRequestObservation,
        ProbeTopology,
        build_probe_plan,
        evaluate_probe,
        recommend_next_probe,
    )

    config = json.loads(config_path.read_text())
    if not worker_path.is_file() or not model_snapshot.is_dir():
        raise FileNotFoundError("calibration worker or model snapshot is absent")
    work_root.mkdir(parents=True, exist_ok=False)
    command_root = work_root / "commands"
    log_root = work_root / "logs"
    command_root.mkdir()
    log_root.mkdir()
    before_audit = cuda_clean_import_audit("capacity-before-inventory")
    if not before_audit["cuda_clean"]:
        raise RuntimeError("capacity controller imported a CUDA-owning module")
    inventory_before = _inventory()
    _require_no_compute_processes(_compute_processes(), phase="capacity worker preflight")

    children: list[tuple[str, subprocess.Popen[bytes], Any, Any]] = []
    cleanup_actions: list[dict[str, Any]] = []
    controller_error: BaseException | None = None
    results: list[Any] = []
    final_action: Any = None
    both_engines_ready = False
    started_ns = time.monotonic_ns()
    overall_deadline_ns = started_ns + round(float(config["controller_timeout_seconds"]) * 1e9)
    try:
        for device, gpu in zip(("gpu0", "gpu1"), inventory_before, strict=True):
            device_root = work_root / device
            stdout_handle = (log_root / f"{device}.stdout.log").open("xb")
            stderr_handle = (log_root / f"{device}.stderr.log").open("xb")
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu["index"])
            environment["SLOFORGE_EXP004_PHYSICAL_GPU_UUID"] = str(gpu["uuid"])
            command = [
                sys.executable,
                str(worker_path),
                "--device",
                device,
                "--config",
                str(config_path),
                "--work-root",
                str(device_root),
                "--command-root",
                str(command_root),
                "--model-snapshot",
                str(model_snapshot),
                "--physical-gpu-uuid",
                str(gpu["uuid"]),
            ]
            try:
                child = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    env=environment,
                    start_new_session=True,
                )
            except BaseException:
                stdout_handle.close()
                stderr_handle.close()
                raise
            children.append((device, child, stdout_handle, stderr_handle))

        child_processes = tuple((device, process) for device, process, *_ in children)
        ready_paths = tuple(command_root / f"{device}.ready.json" for device, *_ in children)
        _wait_for_paths(
            ready_paths,
            deadline_ns=min(
                overall_deadline_ns,
                started_ns + round(float(config["worker_initialization_timeout_seconds"]) * 1e9),
            ),
            phase="engine readiness",
            children=child_processes,
        )
        ready_payloads = _validate_readiness_payloads(
            payloads=tuple(json.loads(path.read_text()) for path in ready_paths),
            expected_inventory=tuple(inventory_before),
        )
        both_engines_ready = True
        _write_new(
            work_root / "readiness/both-engines-ready.json",
            {
                "schema_version": "sloforge.branchfabric.both-engines-ready/v1",
                "BOTH_ENGINES_READY": True,
                "observed_at_monotonic_ns": time.monotonic_ns(),
                "engine_evidence": ready_payloads,
            },
        )

        for probe_index in range(int(config["maximum_probes"])):
            action = recommend_next_probe(
                tuple(results),
                maximum_probes=int(config["maximum_probes"]),
                rate_grid_rps=tuple(float(item) for item in config["rate_grid_rps"]),
            )
            if action.status != "probe":
                final_action = action
                break
            assert action.topology is not None and action.rate_rps is not None
            start_ns = time.monotonic_ns() + 250_000_000
            plan = build_probe_plan(
                probe_id=f"capacity-{probe_index:02d}-{action.topology.value}",
                seed=int(config["seed"]),
                topology=ProbeTopology(action.topology),
                configured_rate_rps=float(action.rate_rps),
                start_ns=start_ns,
                warmup_seconds=float(config["warmup_seconds"]),
                measurement_seconds=float(config["maximum_measurement_seconds"]),
            )
            _write_new(
                command_root / f"probe-{probe_index:02d}.json",
                {
                    "schema_version": "sloforge.branchfabric.capacity-probe-command/v1",
                    "reason": action.reason,
                    "plan": plan,
                },
            )
            worker_paths = tuple(
                command_root / f"probe-{probe_index:02d}.{device}.json" for device, *_ in children
            )
            _wait_for_paths(
                worker_paths,
                deadline_ns=min(
                    overall_deadline_ns,
                    time.monotonic_ns() + round(float(config["probe_timeout_seconds"]) * 1e9),
                ),
                phase=f"probe {probe_index}",
                children=child_processes,
            )
            worker_payloads = tuple(json.loads(path.read_text()) for path in worker_paths)
            reset_paths = tuple(
                command_root / f"probe-{probe_index:02d}.{device}.reset.json"
                for device, *_ in children
            )
            _wait_for_paths(
                reset_paths,
                deadline_ns=min(
                    overall_deadline_ns,
                    time.monotonic_ns() + round(float(config["probe_timeout_seconds"]) * 1e9),
                ),
                phase=f"probe {probe_index} request-state reset",
                children=child_processes,
            )
            reset_payloads = _validate_reset_payloads(
                payloads=tuple(json.loads(path.read_text()) for path in reset_paths),
                expected_probe_id=plan.probe_id,
            )
            expected_tail_end_ns = plan.measurement_end_ns + round(
                float(config["tail_drain_seconds"]) * 1e9
            )
            if any(
                int(payload["tail_drain_end_ns"]) != expected_tail_end_ns
                for payload in worker_payloads
            ):
                raise RuntimeError("capacity workers disagreed with the fixed probe tail")
            probe_end_ns = max(int(payload["probe_end_ns"]) for payload in worker_payloads)
            if not expected_tail_end_ns <= probe_end_ns <= expected_tail_end_ns + 1_000_000_000:
                raise RuntimeError("capacity worker exceeded the abort-finalization allowance")
            observations = tuple(
                CapacityRequestObservation.model_validate_json(
                    json.dumps(row, sort_keys=True, separators=(",", ":")), strict=True
                )
                for payload in worker_payloads
                for row in payload["observations"]
            )
            raw = CapacityProbeRaw(
                plan=plan,
                observations=observations,
                tail_drain_end_ns=expected_tail_end_ns,
                probe_end_ns=probe_end_ns,
                tail_drain_seconds=float(config["tail_drain_seconds"]),
            )
            result = evaluate_probe(
                raw,
                slo_ttft_seconds=float(config["serving_slo_maximum_ttft_seconds"]),
            )
            topology_root = (
                work_root / "calibration/one-gpu"
                if result.topology == ProbeTopology.GPU0_ONLY
                else work_root / "calibration/two-gpu"
            )
            probe_root = topology_root / result.probe_id
            _write_new(probe_root / "plan.json", plan)
            _write_new(probe_root / "raw.json", raw)
            _write_new(probe_root / "result.json", result)
            _write_new(probe_root / "request-state-reset.json", reset_payloads)
            results.append(result)
        if final_action is None:
            final_action = recommend_next_probe(
                tuple(results),
                maximum_probes=int(config["maximum_probes"]),
                rate_grid_rps=tuple(float(item) for item in config["rate_grid_rps"]),
            )
        _write_new(
            command_root / "shutdown.json",
            {
                "schema_version": "sloforge.branchfabric.capacity-worker-shutdown/v1",
                "issued_at_monotonic_ns": time.monotonic_ns(),
            },
        )
        worker_results = tuple(work_root / device / "result.json" for device, *_ in children)
        _wait_for_paths(
            worker_results,
            deadline_ns=overall_deadline_ns,
            phase="worker shutdown",
            children=child_processes,
        )
        shutdown_timeout_seconds = float(config["worker_shutdown_timeout_seconds"])
        for _device, child, *_ in children:
            remaining_seconds = (overall_deadline_ns - time.monotonic_ns()) / 1e9
            child.wait(timeout=max(0.001, min(shutdown_timeout_seconds, remaining_seconds)))
        worker_completions = _validate_worker_completions(
            result_paths=worker_results,
            expected_inventory=tuple(inventory_before),
            returncodes={device: child.returncode for device, child, *_ in children},
        )
    except BaseException as caught:
        controller_error = caught
        if not (command_root / "shutdown.json").exists():
            _write_new(
                command_root / "shutdown.json",
                {
                    "schema_version": "sloforge.branchfabric.capacity-worker-shutdown/v1",
                    "issued_at_monotonic_ns": time.monotonic_ns(),
                    "controller_error": type(caught).__name__,
                },
            )
    finally:
        for _device, child, stdout_handle, stderr_handle in children:
            try:
                _terminate_group(child, cleanup_actions)
            finally:
                stdout_handle.close()
                stderr_handle.close()

    processes_after = _compute_processes()
    cleanup_error: BaseException | None = None
    try:
        _require_no_compute_processes(processes_after, phase="capacity worker postflight")
    except BaseException as caught:
        cleanup_error = caught
    status = "succeeded" if controller_error is None and cleanup_error is None else "failed"
    if status == "succeeded" and final_action is not None:
        if final_action.selection is not None:
            _write_new(work_root / "calibration/selected-load.json", final_action.selection)
        _write_new(
            work_root / "calibration/calibration-result.json",
            {
                "schema_version": "sloforge.branchfabric.capacity-calibration-result/v1",
                "status": final_action.status,
                "reason": final_action.reason,
                "probe_results": results,
                "selection": final_action.selection,
                "v10_launch_permitted": False,
                "review_required": final_action.status == "ready",
            },
        )
    result_payload = {
        "schema_version": "sloforge.branchfabric.capacity-controller/v1",
        "status": status,
        "inventory_before": inventory_before,
        "worker_ready_evidence": (locals().get("ready_payloads", ())),
        "both_engines_ready": both_engines_ready,
        "worker_completion_evidence": (locals().get("worker_completions", ())),
        "worker_pids": {device: child.pid for device, child, *_ in children},
        "worker_returncodes": {device: child.returncode for device, child, *_ in children},
        "probe_count": len(results),
        "calibration_status": (
            None if status != "succeeded" or final_action is None else final_action.status
        ),
        "cleanup_actions": cleanup_actions,
        "compute_processes_after": processes_after,
        "cuda_clean_import_audits": (
            before_audit,
            cuda_clean_import_audit("capacity-after-worker-cleanup"),
        ),
        "controller_error": (
            None
            if controller_error is None
            else {"type": type(controller_error).__name__, "message": str(controller_error)}
        ),
        "cleanup_error": (
            None
            if cleanup_error is None
            else {"type": type(cleanup_error).__name__, "message": str(cleanup_error)}
        ),
        "started_ns": started_ns,
        "ended_ns": time.monotonic_ns(),
    }
    _write_new(work_root / "controller-result.json", result_payload)
    return result_payload


__all__ = [
    "_validate_readiness_payloads",
    "_validate_reset_payloads",
    "_validate_worker_completions",
    "run_capacity_calibration_controller",
]
