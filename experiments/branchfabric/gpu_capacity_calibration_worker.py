#!/usr/bin/env python3
"""Persistent single-GPU worker for Experiment 004 capacity calibration.

The process loads exactly one vLLM engine, then executes immutable probe plans
issued by the CUDA-clean controller.  It never chooses rates or routing.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import queue
import threading
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _canonical_bytes(value: Any) -> bytes:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode()


def _write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(_canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())


def _wait_for(path: Path, *, deadline_ns: int) -> dict[str, Any]:
    while not path.is_file():
        if time.monotonic_ns() >= deadline_ns:
            raise TimeoutError(f"timed out waiting for {path.name}")
        time.sleep(0.02)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _sleep_until(timestamp_ns: int) -> None:
    while True:
        remaining = timestamp_ns - time.monotonic_ns()
        if remaining <= 0:
            return
        time.sleep(min(remaining / 1e9, 0.02))


def _execute_probe(
    engine: Any,
    *,
    plan_payload: dict[str, Any],
    device: str,
    prefix: tuple[int, ...],
    output_tokens: int,
    seed: int,
    probe_timeout_seconds: float,
    tail_drain_seconds: float,
    producer_queue_capacity: int,
) -> dict[str, Any]:
    from gpu_reclamation_worker import _sampling_params

    from sloforge.helix.characterization.gpu_capacity_calibration import CapacityProbePlan

    plan = CapacityProbePlan.model_validate_json(
        json.dumps(plan_payload, sort_keys=True, separators=(",", ":")), strict=True
    )
    assigned = tuple(item for item in plan.arrivals if item.assigned_device == device)
    tail_drain_end_ns = plan.measurement_end_ns + round(tail_drain_seconds * 1e9)
    if not assigned:
        return {
            "schema_version": "sloforge.branchfabric.capacity-worker-probe/v1",
            "probe_id": plan.probe_id,
            "device": device,
            "physical_gpu_uuid": os.environ["SLOFORGE_EXP004_PHYSICAL_GPU_UUID"],
            "observations": (),
            "tail_drain_end_ns": tail_drain_end_ns,
            "probe_end_ns": tail_drain_end_ns,
            "completed_at_monotonic_ns": time.monotonic_ns(),
        }
    if producer_queue_capacity <= 0:
        raise ValueError("producer queue capacity must be positive")
    admission_queue: queue.Queue[Any] = queue.Queue(maxsize=producer_queue_capacity)
    observations: dict[str, dict[str, Any]] = {}
    observations_lock = threading.Lock()
    producer_errors: list[BaseException] = []
    producer_done = threading.Event()
    active: set[str] = set()
    params = _sampling_params(max_tokens=output_tokens, seed=seed)
    deadline_ns = time.monotonic_ns() + round(probe_timeout_seconds * 1e9)

    def produce() -> None:
        try:
            for arrival in assigned:
                _sleep_until(arrival.scheduled_arrival_ns)
                offered_ns = time.monotonic_ns()
                row = {
                    "request_id": arrival.request_id,
                    "global_sequence": arrival.global_sequence,
                    "device": device,
                    "scheduled_arrival_ns": arrival.scheduled_arrival_ns,
                    "offered_ns": offered_ns,
                    "enqueued_ns": None,
                    "admitted_ns": None,
                    "first_token_ns": None,
                    "completed_ns": None,
                    "terminated_ns": tail_drain_end_ns,
                    "requested_output_tokens": output_tokens,
                    "emitted_tokens": 0,
                    "terminal_state": "not-admitted",
                }
                with observations_lock:
                    observations[arrival.request_id] = row
                while time.monotonic_ns() < tail_drain_end_ns:
                    enqueued_ns = time.monotonic_ns()
                    try:
                        admission_queue.put_nowait((arrival, enqueued_ns))
                    except queue.Full:
                        time.sleep(0.01)
                        continue
                    with observations_lock:
                        row["enqueued_ns"] = enqueued_ns
                    break
        except BaseException as caught:
            producer_errors.append(caught)
        finally:
            producer_done.set()

    producer = threading.Thread(
        target=produce,
        name=f"capacity-producer-{plan.probe_id}-{device}",
        daemon=False,
    )
    producer.start()

    while time.monotonic_ns() < tail_drain_end_ns:
        now = time.monotonic_ns()
        if now >= deadline_ns:
            raise TimeoutError(f"probe {plan.probe_id} exceeded its absolute worker bound")
        while True:
            try:
                arrival, enqueued_ns = admission_queue.get_nowait()
            except queue.Empty:
                break
            engine.add_request(
                arrival.request_id,
                {"prompt_token_ids": list(prefix)},
                params,
            )
            admitted_ns = time.monotonic_ns()
            with observations_lock:
                row = observations[arrival.request_id]
                row["enqueued_ns"] = enqueued_ns
                row["admitted_ns"] = admitted_ns
                row["terminal_state"] = "aborted"
            active.add(arrival.request_id)
        if active:
            outputs = engine.step()
            observed_ns = time.monotonic_ns()
            for output in outputs:
                request_id = str(getattr(output, "request_id", ""))
                with observations_lock:
                    output_row = observations.get(request_id)
                    if output_row is None:
                        continue
                    candidates = getattr(output, "outputs", ())
                    token_ids = (
                        tuple(int(item) for item in getattr(candidates[0], "token_ids", ()))
                        if candidates
                        else ()
                    )
                    if len(token_ids) > int(output_row["emitted_tokens"]):
                        output_row["emitted_tokens"] = len(token_ids)
                        if output_row["first_token_ns"] is None:
                            output_row["first_token_ns"] = observed_ns
                    if bool(getattr(output, "finished", False)):
                        output_row["completed_ns"] = observed_ns
                        output_row["terminated_ns"] = observed_ns
                        output_row["terminal_state"] = "completed"
                        active.discard(request_id)
        else:
            time.sleep(0.001)

    producer.join(timeout=1.0)
    if producer.is_alive():
        raise TimeoutError("bounded capacity producer survived the fixed probe tail")
    if producer_errors:
        raise RuntimeError(f"capacity producer failed: {producer_errors[0]}")
    if active:
        engine.abort_request(sorted(active))
    terminal_ns = time.monotonic_ns()
    if terminal_ns > tail_drain_end_ns + 1_000_000_000:
        raise TimeoutError("capacity abort finalization exceeded its one-second allowance")
    with observations_lock:
        for request_id in active:
            observations[request_id]["terminated_ns"] = terminal_ns
            observations[request_id]["terminal_state"] = "aborted"
        while True:
            try:
                pending, enqueued_ns = admission_queue.get_nowait()
            except queue.Empty:
                break
            observations[pending.request_id]["enqueued_ns"] = enqueued_ns
            observations[pending.request_id]["terminated_ns"] = terminal_ns
            observations[pending.request_id]["terminal_state"] = "not-admitted"
        for row in observations.values():
            if row["terminal_state"] != "completed":
                row["terminated_ns"] = terminal_ns

    if set(observations) != {item.request_id for item in assigned}:
        raise RuntimeError("capacity producer omitted planned request accounting")
    rows = tuple(observations[item.request_id] for item in assigned)
    if any(
        row["terminal_state"] == "completed" and row["emitted_tokens"] != output_tokens
        for row in rows
    ):
        raise RuntimeError("completed calibration request violated the 64-token sampling target")
    return {
        "schema_version": "sloforge.branchfabric.capacity-worker-probe/v1",
        "probe_id": plan.probe_id,
        "device": device,
        "physical_gpu_uuid": os.environ["SLOFORGE_EXP004_PHYSICAL_GPU_UUID"],
        "observations": rows,
        "tail_drain_end_ns": tail_drain_end_ns,
        "probe_end_ns": terminal_ns,
        "completed_at_monotonic_ns": terminal_ns,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("gpu0", "gpu1"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--command-root", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--physical-gpu-uuid", required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.work_root.mkdir(parents=True, exist_ok=False)
    started_ns = time.monotonic_ns()
    adapter: Any | None = None
    error: dict[str, str] | None = None
    try:
        from gpu_reclamation_worker import _create_adapter

        adapter_config = {
            **config,
            "initialization_timeout_seconds": config["worker_initialization_timeout_seconds"],
        }
        adapter = _create_adapter(
            adapter_config,
            args.model_snapshot.resolve(strict=True),
            args.physical_gpu_uuid,
        )
        inputs = json.loads((args.model_snapshot / "BRANCHFABRIC_INPUTS.json").read_text())
        prefix = tuple(
            int(item) for item in inputs["prefix_token_ids"][: int(config["serving_prompt_tokens"])]
        )
        import torch  # type: ignore[import-not-found]

        packages = {
            name: importlib.metadata.version(name) for name in ("vllm", "torch", "transformers")
        }
        gpu_name = torch.cuda.get_device_name(0)
        if packages["vllm"] != "0.23.0" or packages["torch"] != "2.11.0":
            raise RuntimeError(f"calibration runtime pins differ: {packages}")
        if "A100" not in gpu_name or torch.version.cuda is None:
            raise RuntimeError("calibration worker lacks the required A100/CUDA runtime")
        runtime_evidence = {
            "packages": packages,
            "torch_cuda": torch.version.cuda,
            "gpu_name": gpu_name,
        }
        model_ready_ns = time.monotonic_ns()
        _write_new(
            args.command_root / f"{args.device}.ready.json",
            {
                "schema_version": "sloforge.branchfabric.capacity-worker-ready/v1",
                "device": args.device,
                "pid": os.getpid(),
                "physical_gpu_uuid": args.physical_gpu_uuid,
                "started_ns": started_ns,
                "model_ready_ns": model_ready_ns,
                "runtime_evidence": runtime_evidence,
            },
        )
        overall_deadline_ns = started_ns + round(float(config["controller_timeout_seconds"]) * 1e9)
        for probe_index in range(int(config["maximum_probes"])):
            command_path = args.command_root / f"probe-{probe_index:02d}.json"
            shutdown_path = args.command_root / "shutdown.json"
            while not command_path.is_file() and not shutdown_path.is_file():
                if time.monotonic_ns() >= overall_deadline_ns:
                    raise TimeoutError("calibration worker exceeded its overall deadline")
                time.sleep(0.02)
            if shutdown_path.is_file():
                break
            command = json.loads(command_path.read_text())
            result = _execute_probe(
                adapter._view.llm_engine,
                plan_payload=command["plan"],
                device=args.device,
                prefix=prefix,
                output_tokens=int(config["serving_output_tokens"]),
                seed=int(config["seed"]),
                probe_timeout_seconds=float(config["probe_timeout_seconds"]),
                tail_drain_seconds=float(config["tail_drain_seconds"]),
                producer_queue_capacity=int(config["producer_queue_capacity"]),
            )
            _write_new(args.command_root / f"probe-{probe_index:02d}.{args.device}.json", result)
            if not adapter._view.manager.reset_prefix_cache():
                raise RuntimeError("calibration worker prefix cache did not drain between probes")
        else:
            _wait_for(args.command_root / "shutdown.json", deadline_ns=overall_deadline_ns)
    except BaseException as caught:
        error = {
            "type": type(caught).__name__,
            "message": str(caught),
            "traceback": traceback.format_exc(),
        }
    finally:
        if adapter is not None:
            try:
                adapter.cleanup_runtime(timeout_s=float(config["worker_shutdown_timeout_seconds"]))
            except BaseException as caught:
                if error is None:
                    error = {
                        "type": type(caught).__name__,
                        "message": str(caught),
                        "traceback": traceback.format_exc(),
                    }
    result = {
        "schema_version": "sloforge.branchfabric.capacity-worker-completion/v1",
        "status": "succeeded" if error is None else "failed",
        "device": args.device,
        "physical_gpu_uuid": args.physical_gpu_uuid,
        "started_ns": started_ns,
        "ended_ns": time.monotonic_ns(),
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "runtime_evidence": locals().get("runtime_evidence"),
        "error": error,
    }
    _write_new(args.work_root / "result.json", result)
    return 0 if error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
