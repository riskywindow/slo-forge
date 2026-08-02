from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Literal, cast

import httpx

from sloforge.compiler import compile_deployment, explain_plan, mock_qwen3_metadata
from sloforge.controller import ControllerConfig, evaluate_controllers
from sloforge.controller.core import DecisionRecord
from sloforge.exporters import ExportContext, export_plan
from sloforge.faults import execute_scenario, load_scenario
from sloforge.hardware.probe import run_probe, save_probe
from sloforge.ir import (
    ArtifactDigest,
    BenchmarkResult,
    DeploymentPlan,
    MetricEstimate,
    save_evidence_bundle,
)
from sloforge.models import fit_service_curves
from sloforge.optimizer import (
    OptimizationRequest,
    evaluate_configuration,
    optimize,
    parse_slo_expression,
)
from sloforge.profiler.core import BackendCandidate, ProfilingBudget
from sloforge.profiler.http_mock import _stream_probe, profile_running_mock_candidates
from sloforge.reports import build_artifact_index, generate_report
from sloforge.runtime import replay_gateway, replay_simulator
from sloforge.trace.format import generate_bursty_trace, write_trace
from sloforge.util import percentile, sha256_file, write_json


class ManagedProcess:
    def __init__(self, *, name: str, command: list[str], cwd: Path, log_dir: Path) -> None:
        self.name = name
        self.command = command
        log_dir.mkdir(parents=True, exist_ok=True)
        self.stdout_path = log_dir / f"{name}.stdout.log"
        self.stderr_path = log_dir / f"{name}.stderr.log"
        self._stdout = self.stdout_path.open("w", encoding="utf-8")
        self._stderr = self.stderr_path.open("w", encoding="utf-8")
        self._stopped = False
        try:
            self.process = subprocess.Popen(
                command,
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=self._stdout,
                stderr=self._stderr,
                text=True,
                start_new_session=True,
            )
        except BaseException:
            self._stdout.close()
            self._stderr.close()
            raise

    def stop(self, *, timeout_s: float = 8.0) -> None:
        if self._stopped:
            return
        self._stopped = True
        try:
            if self.process.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self.process.pid, signal.SIGTERM)
                try:
                    self.process.wait(timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait(timeout=3)
        finally:
            self._stdout.close()
            self._stderr.close()
        if self.process.returncode not in (0, -signal.SIGTERM):
            stderr = self.stderr_path.read_text(encoding="utf-8")[-2000:]
            raise RuntimeError(f"{self.name} exited with {self.process.returncode}: {stderr}")

    def __enter__(self) -> ManagedProcess:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        self.stop()


def _port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _wait_ready(url: str, process: ManagedProcess, *, timeout_s: float = 15.0) -> float:
    started = time.perf_counter_ns()
    deadline = time.monotonic() + timeout_s
    last_error = "not attempted"
    while time.monotonic() < deadline:
        if process.process.poll() is not None:
            raise RuntimeError(
                f"{process.name} exited before readiness with {process.process.returncode}"
            )
        try:
            response = httpx.get(url, timeout=0.5)
            if response.status_code == 200:
                return (time.perf_counter_ns() - started) / 1e6
            last_error = f"HTTP {response.status_code}"
        except httpx.HTTPError as exc:
            last_error = str(exc)
        time.sleep(0.05)
    raise TimeoutError(f"{process.name} did not become ready at {url}: {last_error}")


def _prepare_directory(path: Path, *, reset: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not reset:
            raise FileExistsError(f"output directory is not empty: {path}; pass --reset")
        resolved = path.resolve()
        if resolved.name not in {"demo", "cpu-demo"}:
            raise ValueError(f"refusing to reset non-demo directory: {resolved}")
        shutil.rmtree(resolved)
    path.mkdir(parents=True, exist_ok=True)


def _candidate_catalog(memory_bytes: int) -> list[BackendCandidate]:
    common: dict[str, object] = {
        "runtime": "mock",
        "runtime_version": "1.0.0",
        "hardware_id": "local-cpu",
        "dtype": "float32",
        "memory_bytes": memory_bytes,
        "model_parameter_count": 600_000_000,
        "max_sequence_length": 32768,
    }
    return [
        BackendCandidate.model_validate(
            {
                **common,
                "candidate_id": "mock-balanced",
                "hourly_price_usd": 1.10,
                "startup_ms": 180,
                "startup_jitter": 0.12,
                "prefill_base_ms": 8.0,
                "prefill_ms_per_token": 0.080,
                "decode_base_ms": 1.0,
                "decode_ms_per_active_sequence": 2.6,
                "max_concurrency": 4,
                "failure_rate": 0.008,
            }
        ),
        BackendCandidate.model_validate(
            {
                **common,
                "candidate_id": "mock-fast",
                "hourly_price_usd": 2.20,
                "startup_ms": 420,
                "startup_jitter": 0.18,
                "prefill_base_ms": 4.0,
                "prefill_ms_per_token": 0.042,
                "decode_base_ms": 0.8,
                "decode_ms_per_active_sequence": 1.7,
                "max_concurrency": 6,
                "failure_rate": 0.004,
            }
        ),
        BackendCandidate.model_validate(
            {
                **common,
                "candidate_id": "mock-economy",
                "hourly_price_usd": 0.45,
                "startup_ms": 75,
                "startup_jitter": 0.08,
                "prefill_base_ms": 12.0,
                "prefill_ms_per_token": 0.115,
                "decode_base_ms": 1.4,
                "decode_ms_per_active_sequence": 3.8,
                "max_concurrency": 2,
                "failure_rate": 0.015,
            }
        ),
    ]


def _mock_config(candidate: BackendCandidate, port: int, seed: int) -> dict[str, object]:
    return {
        "name": candidate.candidate_id,
        "bind": f"127.0.0.1:{port}",
        "seed": seed,
        "startup_ms": int(candidate.startup_ms),
        "startup_jitter_ms": int(candidate.startup_ms * candidate.startup_jitter),
        "startup_every_n_requests": 0,
        "prefill_base_ms": int(candidate.prefill_base_ms),
        "prefill_per_token_us": int(candidate.prefill_ms_per_token * 1000),
        "decode_per_token_ms": max(
            1, int(candidate.decode_base_ms + candidate.decode_ms_per_active_sequence)
        ),
        "max_concurrency": candidate.max_concurrency,
        "max_output_tokens": 256,
        "price_per_hour_usd": candidate.hourly_price_usd,
        "failure_rate": candidate.failure_rate,
    }


def _actual_cold_samples(endpoint: str, candidate: BackendCandidate) -> list[float]:
    samples: list[float] = []
    for delay in (
        30,
        int(candidate.startup_ms * 0.65),
        int(candidate.startup_ms),
        int(candidate.startup_ms * 1.2),
    ):
        response = httpx.post(
            f"{endpoint}/admin/fault",
            json={"fault": "cold_start", "next_delay_ms": max(1, delay)},
            timeout=3,
        )
        response.raise_for_status()
        timing = _stream_probe(endpoint, prompt_tokens=16, output_tokens=1)
        if timing.status_code >= 400 or timing.error:
            raise RuntimeError(
                f"cold-start probe failed for {candidate.candidate_id}: {timing.error}"
            )
        samples.append(timing.e2e_ms)
    return samples


def _benchmark(
    *,
    benchmark_id: str,
    command: tuple[str, ...],
    raw_path: Path,
    seed: int,
    metrics: dict[str, tuple[float, str]],
    when: datetime,
    repository_root: Path,
) -> BenchmarkResult:
    return BenchmarkResult(
        benchmark_id=benchmark_id,
        command=command,
        raw_result_uri=str(raw_path.resolve().relative_to(repository_root.resolve())),
        raw_result_digest=ArtifactDigest(value=sha256_file(raw_path)),
        seed=seed,
        started_at=when,
        completed_at=when,
        metrics={
            name: MetricEstimate(
                point=value,
                lower=value,
                upper=value,
                confidence=0.95,
                unit=unit,
                sample_count=1,
                measurement_ids=(f"{benchmark_id}-{name}",),
            )
            for name, (value, unit) in metrics.items()
        },
    )


def _portable_arguments(
    command: list[str] | tuple[str, ...], repository_root: Path
) -> tuple[str, ...]:
    rendered: list[str] = []
    root = repository_root.resolve()
    for argument in command:
        path = Path(argument)
        rendered_argument = argument
        if path.is_absolute():
            try:
                rendered_argument = str(path.resolve().relative_to(root))
            except ValueError:
                rendered_argument = argument
        rendered.append(rendered_argument)
    return tuple(rendered)


def _display_command(command: list[str] | tuple[str, ...], repository_root: Path) -> str:
    return " ".join(_portable_arguments(command, repository_root))


def _controller_simulator_actions(
    decisions: list[DecisionRecord], plan: DeploymentPlan
) -> tuple[list[dict[str, object]], str]:
    active = [f"replica-{index}" for index in range(plan.replica_topology.initial_replicas)]
    next_replica = len(active)
    current_concurrency = plan.batching.maximum_active_sequences
    actions: list[dict[str, object]] = []
    routing = {
        "round_robin": "round_robin",
        "least_outstanding": "least_outstanding",
        "earliest_finish": "earliest_finish",
        "slo_slack": "slo_slack_aware",
    }[plan.routing.kind.value]
    for decision in decisions:
        at_ms = max(0, int(decision.observed.window_start_ms))
        target = decision.chosen.target_replicas
        while len(active) < target:
            replica_id = f"controller-replica-{next_replica}"
            next_replica += 1
            active.append(replica_id)
            actions.append(
                {
                    "at_ms": at_ms,
                    "action": {
                        "type": "add_replica",
                        "replica": {
                            "id": replica_id,
                            "initially_warm": False,
                            "max_queue": plan.admission.queue_capacity,
                            "max_active_sequences": decision.chosen.target_concurrency,
                            "service_rate_multiplier": 1.0,
                            "hourly_price_usd": plan.hardware.hourly_price_usd,
                            "canary": decision.canary,
                        },
                    },
                }
            )
        while len(active) > target:
            replica_id = active.pop()
            actions.append(
                {
                    "at_ms": at_ms,
                    "action": {"type": "remove_replica", "replica_id": replica_id},
                }
            )
        if decision.chosen.target_concurrency != current_concurrency:
            current_concurrency = decision.chosen.target_concurrency
            actions.extend(
                {
                    "at_ms": at_ms,
                    "action": {
                        "type": "capacity_loss",
                        "replica_id": replica_id,
                        "max_active_sequences": current_concurrency,
                    },
                }
                for replica_id in active
            )
        if decision.chosen.routing_policy == "slo_slack":
            routing = "slo_slack_aware"
    actions.sort(key=lambda item: cast(int, item["at_ms"]))
    return actions, routing


def run_demo(*, repository_root: Path, artifact_dir: Path, report_dir: Path, reset: bool) -> None:
    if os.getenv("SLOFORGE_GPU_BUDGET_USD"):
        print("CPU demo ignores SLOFORGE_GPU_BUDGET_USD and creates no cloud resources.")
    _prepare_directory(artifact_dir, reset=reset)
    _prepare_directory(report_dir, reset=reset)
    transcript: list[str] = []
    seed = 41
    trace = generate_bursty_trace(seed=seed, count=120)
    trace_path = artifact_dir / "workloads" / "coding-agent.jsonl"
    write_trace(trace_path, trace)
    hardware = run_probe(device="cpu", warmups=1, samples=5)
    hardware_path = artifact_dir / "hardware" / "local-cpu.json"
    save_probe(hardware_path, hardware)
    candidates = _candidate_catalog(hardware.hardware.memory_bytes)
    gateway_binary = repository_root / "target" / "debug" / "sloforge-gateway"
    if not gateway_binary.exists():
        command = ["cargo", "build", "-q", "-p", "sloforge-gateway", "-p", "sloforge-sim"]
        transcript.append(" ".join(command))
        subprocess.run(command, cwd=repository_root, check=True, timeout=240)
    runtime_dir = artifact_dir / "runtime"
    log_dir = runtime_dir / "logs"
    ports = [_port() for _ in candidates]
    backend_processes: list[ManagedProcess] = []
    endpoints: dict[str, str] = {}
    startup_samples: dict[str, list[float]] = {}
    gateway_process: ManagedProcess | None = None
    try:
        for index, (candidate, port) in enumerate(zip(candidates, ports, strict=True)):
            config_path = runtime_dir / f"{candidate.candidate_id}.json"
            write_json(config_path, _mock_config(candidate, port, seed + index))
            command = [str(gateway_binary), "mock-backend", "--config", str(config_path)]
            transcript.append(_display_command(command, repository_root))
            process = ManagedProcess(
                name=candidate.candidate_id, command=command, cwd=repository_root, log_dir=log_dir
            )
            backend_processes.append(process)
            endpoint = f"http://127.0.0.1:{port}"
            ready_ms = _wait_ready(f"{endpoint}/health", process)
            endpoints[candidate.candidate_id] = endpoint
            startup_samples[candidate.candidate_id] = [
                ready_ms,
                *_actual_cold_samples(endpoint, candidate),
            ]
        profile_dir = artifact_dir / "profiles" / "qwen3-cpu-mocks"
        profile = profile_running_mock_candidates(
            candidates=candidates,
            endpoints=endpoints,
            startup_samples_ms=startup_samples,
            trace=trace,
            trace_path=trace_path,
            hardware_path=hardware_path,
            budget=ProfilingBudget(max_duration_s=180.0, max_cost_usd=0.20),
            seed=seed,
            output_dir=profile_dir,
        )
        models = fit_service_curves(profile, seed=seed)
        models_path = artifact_dir / "models" / "service-curves.json"
        write_json(models_path, models.model_dump())
        optimization_request = OptimizationRequest(
            constraints=parse_slo_expression("p95_ttft_ms<=250,p99_itl_ms<=45,availability>=0.70"),
            objective="cost_per_million_tokens",
            direction="minimize",
            max_replicas=4,
            trial_budget=24,
            uncertainty_safety=0.5,
            seed=seed,
        )
        optimization = optimize(
            profile=profile, models=models, trace=trace, request=optimization_request
        )
        optimization_path = artifact_dir / "optimization" / "result.json"
        write_json(optimization_path, optimization.model_dump())
        default_configuration = next(
            item.configuration
            for item in optimization.evaluated_candidates
            if item.configuration.backend_candidate_id == "mock-balanced"
            and item.configuration.replicas == 1
            and item.configuration.concurrency == 1
            and item.configuration.max_batched_tokens == 2048
            and not item.configuration.chunked_prefill
            and item.configuration.routing_policy == "round_robin"
        )
        manual_configuration = next(
            item.configuration
            for item in optimization.evaluated_candidates
            if item.configuration.backend_candidate_id == "mock-fast"
            and item.configuration.replicas == 1
            and item.configuration.concurrency == 4
            and item.configuration.max_batched_tokens == 2048
            and item.configuration.chunked_prefill
            and item.configuration.routing_policy == "round_robin"
        )
        peak_second = max(
            {int(item.arrival_ms // 1000) for item in trace},
            key=lambda second: sum(int(item.arrival_ms // 1000) == second for item in trace),
        )
        burst = [item for item in trace if int(item.arrival_ms // 1000) == peak_second]
        regimes = {
            "steady": [
                item.model_copy(update={"arrival_ms": float(index * 100)})
                for index, item in enumerate(trace)
            ],
            "bursty": burst,
            "short-prompts": [item for item in trace if item.request_class == "interactive"],
            "long-prompts": [item for item in trace if item.request_class == "long-context"],
            "mixed": trace,
        }
        serving_baselines: list[dict[str, object]] = []
        for baseline_name, configuration in (
            ("documented-engine-default", default_configuration),
            ("manual-static", manual_configuration),
            ("sloforge-plan", optimization.selected.configuration),
        ):
            for regime_name, regime_trace in regimes.items():
                evaluation = evaluate_configuration(
                    configuration=configuration,
                    profile=profile,
                    models=models,
                    trace=regime_trace,
                    request=optimization.request,
                )
                serving_baselines.append(
                    {
                        "baseline": baseline_name,
                        "regime": regime_name,
                        "configuration": configuration.model_dump(),
                        "fidelity": "calibrated_prediction",
                        "metrics": evaluation.predicted.model_dump(),
                        "uncertainty": evaluation.uncertainty.model_dump(),
                        "feasible": evaluation.feasible,
                        "rejection_reasons": evaluation.rejection_reasons,
                    }
                )
        serving_baselines_path = artifact_dir / "benchmarks" / "serving-baselines.json"
        write_json(
            serving_baselines_path,
            {
                "schema_version": "sloforge.serving-baselines/v1",
                "source_profile_id": profile.profile_id,
                "measurement_claim": (
                    "These are calibrated configuration predictions; raw backend measurements "
                    "are referenced by the profile and are not relabeled as configuration trials."
                ),
                "results": serving_baselines,
            },
        )
        compiled = compile_deployment(
            optimization=optimization,
            profile=profile,
            models=models,
            hardware=hardware,
            trace=trace,
            trace_path=trace_path,
            hardware_path=hardware_path,
            profile_dir=profile_dir,
            output_path=artifact_dir / "plans" / "qwen3-coding-agent.json",
            evidence_dir=artifact_dir / "evidence",
            model_metadata=mock_qwen3_metadata(requested_model="Qwen/Qwen3-0.6B"),
            repository_root=repository_root,
        )
        selected_configuration = optimization.selected.configuration
        selected_candidate = next(
            candidate
            for candidate in candidates
            if candidate.candidate_id == selected_configuration.backend_candidate_id
        )
        selected_endpoints = [endpoints[selected_candidate.candidate_id]]
        # The running validation must exercise the compiled replica topology, not route across
        # every candidate that happened to be profiled. Start distinct selected-backend processes
        # so replica capacity and crash isolation are real rather than duplicated endpoint labels.
        for replica_index in range(1, selected_configuration.replicas):
            replica_port = _port()
            replica_name = f"{selected_candidate.candidate_id}-replica-{replica_index}"
            replica_config = runtime_dir / f"{replica_name}.json"
            write_json(
                replica_config,
                _mock_config(selected_candidate, replica_port, seed + 100 + replica_index),
            )
            replica_command = [
                str(gateway_binary),
                "mock-backend",
                "--config",
                str(replica_config),
            ]
            transcript.append(_display_command(replica_command, repository_root))
            replica_process = ManagedProcess(
                name=replica_name,
                command=replica_command,
                cwd=repository_root,
                log_dir=log_dir,
            )
            backend_processes.append(replica_process)
            replica_endpoint = f"http://127.0.0.1:{replica_port}"
            _wait_ready(f"{replica_endpoint}/health", replica_process)
            selected_endpoints.append(replica_endpoint)
        gateway_port = _port()
        gateway_trace = runtime_dir / "gateway-trace.json"
        gateway_config = {
            "bind": f"127.0.0.1:{gateway_port}",
            "backends": [
                {
                    "name": f"{selected_candidate.candidate_id}-replica-{replica_index}",
                    "base_url": endpoint,
                    "capacity": selected_configuration.concurrency,
                    "estimated_service_ms": selected_candidate.prefill_base_ms
                    + selected_candidate.decode_base_ms * 16,
                    "price_per_hour_usd": selected_candidate.hourly_price_usd,
                    "health_path": "/health",
                    "weight": 1,
                }
                for replica_index, endpoint in enumerate(selected_endpoints)
            ],
            "routing_policy": {
                "round_robin": "round_robin",
                "least_outstanding": "least_outstanding",
                "earliest_finish": "estimated_earliest_finish",
                "slo_slack": "slo_slack_aware",
            }[selected_configuration.routing_policy],
            "admission_capacity": compiled.plan.admission.queue_capacity,
            "stream_buffer_bytes": 262144,
            "max_request_bytes": 1048576,
            "queue_timeout_ms": 800,
            "request_timeout_ms": 15000,
            "connect_timeout_ms": 1000,
            "health_interval_ms": 200,
            "health_timeout_ms": 150,
            "retry_attempts": 1,
            "breaker_failures": 2,
            "breaker_cooldown_ms": 500,
            "trace_output": str(gateway_trace.resolve().relative_to(repository_root.resolve())),
            "max_trace_events": 100000,
            "provenance": {"plan_id": compiled.plan.metadata.uid, "profile_id": profile.profile_id},
        }
        gateway_config_path = runtime_dir / "gateway.json"
        write_json(gateway_config_path, gateway_config)
        command = [str(gateway_binary), "serve", "--config", str(gateway_config_path)]
        transcript.append(_display_command(command, repository_root))
        gateway_process = ManagedProcess(
            name="gateway", command=command, cwd=repository_root, log_dir=log_dir
        )
        gateway_url = f"http://127.0.0.1:{gateway_port}"
        _wait_ready(f"{gateway_url}/ready", gateway_process)
        gateway_replay_path = artifact_dir / "gateway" / "replay.json"
        gateway_replay = asyncio.run(
            replay_gateway(
                gateway_url=gateway_url,
                backend_urls=selected_endpoints,
                trace=trace,
                # Preserve the workload arrival process used by optimization. Compressing time
                # here would silently turn runtime validation into a different, 5.6x-heavier SLO.
                time_scale=1.0,
                output_path=gateway_replay_path,
            )
        )
        metrics_response = httpx.get(f"{gateway_url}/metrics", timeout=3)
        metrics_response.raise_for_status()
        gateway_metrics_path = artifact_dir / "gateway" / "metrics.prom"
        gateway_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        gateway_metrics_path.write_text(metrics_response.text, encoding="utf-8")
        gateway_process.stop()
        gateway_process = None
        if not gateway_trace.exists():
            raise RuntimeError(
                "gateway did not emit its configured Chrome trace during graceful shutdown"
            )
    finally:
        cleanup_errors: list[Exception] = []
        if gateway_process is not None:
            try:
                gateway_process.stop()
            except Exception as exc:
                cleanup_errors.append(exc)
        for process in reversed(backend_processes):
            try:
                process.stop()
            except Exception as exc:
                cleanup_errors.append(exc)
        if cleanup_errors:
            active_error = sys.exception()
            if active_error is None:
                raise ExceptionGroup(
                    "one or more demo processes failed during cleanup", cleanup_errors
                )
            for cleanup_error in cleanup_errors:
                active_error.add_note(f"cleanup failure: {cleanup_error}")
    simulator_output = artifact_dir / "simulator" / "replay.json"
    simulator_output.parent.mkdir(parents=True, exist_ok=True)
    simulator = replay_simulator(
        plan=compiled.plan,
        models=models,
        trace=trace,
        output_path=simulator_output,
        chrome_trace_path=artifact_dir / "simulator" / "trace.json",
        repository_root=repository_root,
        seed=seed,
        inject_required_faults=True,
    )
    transcript.append(_display_command(simulator.command, repository_root))
    selected_model = next(
        item
        for item in models.candidates
        if item.candidate_id == optimization.selected.configuration.backend_candidate_id
    )
    prompt_p95 = percentile([float(item.prompt_tokens) for item in trace], 0.95)
    prefill_points = selected_model.prefill.points
    prefill_slope = 0.0
    if len(prefill_points) > 1:
        left, right = prefill_points[-2:]
        prefill_slope = max(0.0, (right.median_ms - left.median_ms) / max(right.x - left.x, 1e-9))
    calibrated_ttft, _, _ = selected_model.prefill.predict(prompt_p95)
    controller = evaluate_controllers(
        trace,
        ControllerConfig(
            max_replicas=4,
            initial_replicas=compiled.plan.replica_topology.initial_replicas,
            initial_concurrency=compiled.plan.batching.maximum_active_sequences,
            max_concurrency=max(8, compiled.plan.batching.maximum_active_sequences),
            target_p95_ttft_ms=250.0,
            hourly_replica_cost_usd=compiled.plan.hardware.hourly_price_usd,
            calibrated_base_ttft_ms=calibrated_ttft,
            calibrated_prefill_ms_per_token=prefill_slope,
            profiled_prompt_p95_tokens=prompt_p95,
            calibrated_cold_start_p95_ms=selected_model.startup_p95_ms,
        ),
    )
    predictive_actions, predictive_routing = _controller_simulator_actions(
        controller.predictive_decisions, compiled.plan
    )
    reactive_actions, reactive_routing = _controller_simulator_actions(
        controller.reactive_decisions, compiled.plan
    )
    predictive_twin_path = artifact_dir / "controller" / "predictive-twin.json"
    predictive_twin = replay_simulator(
        plan=compiled.plan,
        models=models,
        trace=trace,
        output_path=predictive_twin_path,
        chrome_trace_path=artifact_dir / "controller" / "predictive-twin-trace.json",
        repository_root=repository_root,
        seed=seed,
        inject_required_faults=False,
        control_actions=predictive_actions,
        routing_policy_override=predictive_routing,
    )
    reactive_twin_path = artifact_dir / "controller" / "reactive-twin.json"
    reactive_twin = replay_simulator(
        plan=compiled.plan,
        models=models,
        trace=trace,
        output_path=reactive_twin_path,
        chrome_trace_path=artifact_dir / "controller" / "reactive-twin-trace.json",
        repository_root=repository_root,
        seed=seed,
        inject_required_faults=False,
        control_actions=reactive_actions,
        routing_policy_override=reactive_routing,
    )
    transcript.extend(
        (
            _display_command(predictive_twin.command, repository_root),
            _display_command(reactive_twin.command, repository_root),
        )
    )
    controller = controller.model_copy(
        update={
            "predictive": controller.predictive.model_copy(
                update={
                    "slo_violations": int(predictive_twin.summary["deadline_miss_count"]),
                    "estimated_cost_usd": float(predictive_twin.summary["cost_usd"]),
                }
            ),
            "reactive": controller.reactive.model_copy(
                update={
                    "slo_violations": int(reactive_twin.summary["deadline_miss_count"]),
                    "estimated_cost_usd": float(reactive_twin.summary["cost_usd"]),
                }
            ),
        }
    )
    controller_path = artifact_dir / "controller" / "evaluation.json"
    write_json(controller_path, controller.model_dump())
    chaos = execute_scenario(load_scenario(repository_root / "scenarios" / "cpu-demo.yaml"))
    chaos_path = artifact_dir / "chaos" / "result.json"
    write_json(chaos_path, chaos.model_dump())
    export_root = artifact_dir / "exports"
    plan = compiled.plan
    export_context = ExportContext(
        plan_id=plan.metadata.uid,
        model_id=plan.model.model_id,
        model_revision=plan.model.revision,
        engine=("tensorrt-llm" if plan.engine.runtime == "tensorrt_llm" else plan.engine.runtime),
        dtype=cast(
            Literal["float32", "float16", "bfloat16", "int8", "int4"],
            plan.engine.dtype.value,
        ),
        accelerator=plan.hardware.gpus[0].product if plan.hardware.gpus else None,
        gpu_count=len(plan.hardware.gpus),
        cpu_cores=float(plan.hardware.cpu.physical_cores),
        memory_gib=max(1, plan.hardware.system_memory_bytes // (1024**3)),
        min_replicas=plan.replica_topology.minimum_replicas,
        max_replicas=plan.replica_topology.maximum_replicas,
        concurrency=plan.batching.maximum_active_sequences,
        max_batched_tokens=plan.batching.maximum_batched_tokens,
        max_sequence_length=plan.model.maximum_sequence_length,
        regions=list(plan.replica_topology.regions),
        scaledown_window_s=int(plan.autoscaling.scale_down_cooldown_seconds),
        enable_memory_snapshot=False,
        estimated_service_ms=plan.predicted_metrics["p95_e2e_ms"].point,
        hourly_price_usd=plan.hardware.hourly_price_usd,
    )
    for target in ("local", "docker", "kubernetes", "modal", "truss"):
        result = export_plan(
            context=export_context,
            target=target,
            output=export_root / target,
            repository_root=repository_root,
        )
        transcript.append(f"sloforge export --target {target} # {len(result.files)} files")
    transcript_path = artifact_dir / "command-transcript.txt"
    transcript_path.write_text("\n".join(transcript) + "\n", encoding="utf-8")
    when = datetime.now(UTC)
    benchmark_results = (
        _benchmark(
            benchmark_id="cpu-rust-simulator",
            command=tuple(simulator.command),
            raw_path=simulator_output,
            seed=seed,
            metrics={
                "p95_ttft_ms": (float(simulator.summary["p95_ttft_ms"]), "ms"),
                "slo_attainment": (float(simulator.summary["slo_attainment"]), "ratio"),
            },
            when=when,
            repository_root=repository_root,
        ),
        _benchmark(
            benchmark_id="cpu-rust-gateway-replay",
            command=_portable_arguments(
                ("sloforge", "replay", "--gateway", str(gateway_config_path)), repository_root
            ),
            raw_path=gateway_replay_path,
            seed=seed,
            metrics={
                "p95_ttft_ms": (float(gateway_replay.summary["p95_ttft_ms"]), "ms"),
                "availability": (float(gateway_replay.summary["availability"]), "ratio"),
            },
            when=when,
            repository_root=repository_root,
        ),
        _benchmark(
            benchmark_id="predictive-controller-comparison",
            command=("sloforge", "benchmark", "controller"),
            raw_path=controller_path,
            seed=seed,
            metrics={
                "predictive_deadline_misses": (
                    float(controller.predictive.slo_violations),
                    "requests",
                ),
                "reactive_deadline_misses": (
                    float(controller.reactive.slo_violations),
                    "requests",
                ),
            },
            when=when,
            repository_root=repository_root,
        ),
        _benchmark(
            benchmark_id="fault-diagnosis",
            command=("sloforge", "chaos", "--scenario", "scenarios/cpu-demo.yaml"),
            raw_path=chaos_path,
            seed=seed,
            metrics={"accuracy": (chaos.diagnosis_accuracy, "ratio")},
            when=when,
            repository_root=repository_root,
        ),
    )
    evidence_artifacts = {
        **compiled.evidence.artifact_hashes,
        "optimization": ArtifactDigest(value=sha256_file(optimization_path)),
        "models": ArtifactDigest(value=sha256_file(models_path)),
        "simulator_replay": ArtifactDigest(value=sha256_file(simulator_output)),
        "simulator_scenario": ArtifactDigest(
            value=sha256_file(simulator_output.with_suffix(".scenario.json"))
        ),
        "simulator_raw": ArtifactDigest(
            value=sha256_file(simulator_output.with_suffix(".raw.json"))
        ),
        "simulator_trace": ArtifactDigest(
            value=sha256_file(artifact_dir / "simulator" / "trace.json")
        ),
        "gateway_replay": ArtifactDigest(value=sha256_file(gateway_replay_path)),
        "controller": ArtifactDigest(value=sha256_file(controller_path)),
        "predictive_controller_twin": ArtifactDigest(value=sha256_file(predictive_twin_path)),
        "predictive_controller_scenario": ArtifactDigest(
            value=sha256_file(predictive_twin_path.with_suffix(".scenario.json"))
        ),
        "predictive_controller_raw": ArtifactDigest(
            value=sha256_file(predictive_twin_path.with_suffix(".raw.json"))
        ),
        "predictive_controller_trace": ArtifactDigest(
            value=sha256_file(artifact_dir / "controller" / "predictive-twin-trace.json")
        ),
        "reactive_controller_twin": ArtifactDigest(value=sha256_file(reactive_twin_path)),
        "reactive_controller_scenario": ArtifactDigest(
            value=sha256_file(reactive_twin_path.with_suffix(".scenario.json"))
        ),
        "reactive_controller_raw": ArtifactDigest(
            value=sha256_file(reactive_twin_path.with_suffix(".raw.json"))
        ),
        "reactive_controller_trace": ArtifactDigest(
            value=sha256_file(artifact_dir / "controller" / "reactive-twin-trace.json")
        ),
        "chaos": ArtifactDigest(value=sha256_file(chaos_path)),
        "command_transcript": ArtifactDigest(value=sha256_file(transcript_path)),
        "serving_baselines": ArtifactDigest(value=sha256_file(serving_baselines_path)),
    }
    evidence = compiled.evidence.model_copy(
        update={"benchmark_results": benchmark_results, "artifact_hashes": evidence_artifacts}
    )
    save_evidence_bundle(compiled.evidence_path, evidence)
    artifacts = {
        "plan": (compiled.plan_path, "application/json"),
        "evidence": (compiled.evidence_path, "application/json"),
        "optimization": (optimization_path, "application/json"),
        "models": (models_path, "application/json"),
        "controller": (controller_path, "application/json"),
        "predictive_controller_twin": (predictive_twin_path, "application/json"),
        "predictive_controller_scenario": (
            predictive_twin_path.with_suffix(".scenario.json"),
            "application/json",
        ),
        "predictive_controller_raw": (
            predictive_twin_path.with_suffix(".raw.json"),
            "application/json",
        ),
        "predictive_controller_trace": (
            artifact_dir / "controller" / "predictive-twin-trace.json",
            "application/json",
        ),
        "reactive_controller_twin": (reactive_twin_path, "application/json"),
        "reactive_controller_scenario": (
            reactive_twin_path.with_suffix(".scenario.json"),
            "application/json",
        ),
        "reactive_controller_raw": (
            reactive_twin_path.with_suffix(".raw.json"),
            "application/json",
        ),
        "reactive_controller_trace": (
            artifact_dir / "controller" / "reactive-twin-trace.json",
            "application/json",
        ),
        "chaos": (chaos_path, "application/json"),
        "replay": (simulator_output, "application/json"),
        "simulator_scenario": (
            simulator_output.with_suffix(".scenario.json"),
            "application/json",
        ),
        "simulator_raw": (simulator_output.with_suffix(".raw.json"), "application/json"),
        "simulator_trace": (
            artifact_dir / "simulator" / "trace.json",
            "application/json",
        ),
        "gateway_replay": (gateway_replay_path, "application/json"),
        "gateway_metrics": (gateway_metrics_path, "text/plain"),
        "gateway_trace": (gateway_trace, "application/json"),
        "profile": (profile_dir / "profile.json", "application/json"),
        "raw_measurements": (profile_dir / "measurements.jsonl", "application/x-ndjson"),
        "hardware": (hardware_path, "application/json"),
        "workload": (trace_path, "application/x-ndjson"),
        "command_transcript": (transcript_path, "text/plain"),
        "serving_baselines": (serving_baselines_path, "application/json"),
    }
    artifact_index_path = artifact_dir / "evidence" / "artifact-index.json"
    build_artifact_index(
        repository_root=repository_root,
        artifacts=artifacts,
        output=artifact_index_path,
    )
    report = generate_report(
        evidence_path=compiled.evidence_path,
        artifact_index_path=artifact_index_path,
        output_dir=report_dir,
        repository_root=repository_root,
        plan_explanation=explain_plan(plan),
    )
    print(
        json.dumps(
            {
                "plan": str(compiled.plan_path),
                "evidence": str(compiled.evidence_path),
                "report": report.outputs["evaluation.md"],
                "selected_config": optimization.selected.configuration.config_id,
                "gateway_requests": gateway_replay.summary["request_count"],
                "simulated_requests": simulator.summary["request_count"],
                "diagnosis_accuracy": chaos.diagnosis_accuracy,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete deterministic CPU SLOForge demo")
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/demo"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/demo"))
    parser.add_argument("--reset", action="store_true")
    arguments = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[2]
    run_demo(
        repository_root=repository_root,
        artifact_dir=(repository_root / arguments.artifact_dir).resolve(),
        report_dir=(repository_root / arguments.report_dir).resolve(),
        reset=arguments.reset,
    )


if __name__ == "__main__":
    main()
