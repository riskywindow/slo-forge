"""CPU-only ServingSynthBench runner with raw measured evidence and local baselines."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import platform
import random
import statistics
import sys
import time
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path
from types import ModuleType
from typing import cast

from sloforge.genesis.compiler import initialize_genesis_run
from sloforge.genesis.frontend import inspect_reference_package
from sloforge.genesis.ir import canonical_json
from sloforge.genesis.sandbox import SandboxLimits, SandboxRequest, execute_sandboxed

from .grammar import load_hidden_cases, load_task, load_workload, verify_public_package
from .integrity import audit_raw_samples
from .models import (
    AggregateMetrics,
    BaselineKind,
    BaselineStatus,
    BaselineSummary,
    CpuRunConfiguration,
    HiddenCaseResult,
    HiddenEvaluationSummary,
    RawCpuSample,
    SynthBenchReport,
    TaskDescriptor,
    TaskRunReport,
    WorkloadRequest,
)
from .special_case import audit_special_casing


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _seed(base: int, *parts: object) -> int:
    payload = "\0".join((str(base), *(str(part) for part in parts))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _environment_fingerprint() -> str:
    value = {
        "implementation": platform.python_implementation(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "timer": "time.perf_counter_ns",
    }
    return _sha(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def _load_module(path: Path, identity: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(identity, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot import generated reference module: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _callable(module: ModuleType, name: str) -> Callable[..., object]:
    value = getattr(module, name, None)
    if value is None or not callable(value):
        raise ValueError(f"generated reference entry point is missing: {name}")
    return cast(Callable[..., object], value)


class _Executor:
    def __init__(
        self,
        task_directory: Path,
        descriptor: TaskDescriptor,
        *,
        identity_suffix: str = "reference",
    ) -> None:
        self.descriptor = descriptor
        module = _load_module(
            task_directory / descriptor.public_package_path / "reference.py",
            f"synthbench_{descriptor.task_id}_{identity_suffix}",
        )
        self.load_model = _callable(module, "load_model")
        self.allocate_state = _callable(module, "allocate_state")
        self.prefill = _callable(module, "prefill")
        self.decode_step = _callable(module, "decode_step")
        self.sample = _callable(module, "custom_sampler")
        self.model = self.load_model(seed=descriptor.seed)

    def execute(
        self,
        request: WorkloadRequest,
        *,
        deadline_ns: int,
    ) -> tuple[tuple[int, ...], int, int, tuple[int, ...], tuple[str, str] | None]:
        start = time.perf_counter_ns()
        state = self.allocate_state(
            request_id=request.request_id,
            prompt_tokens=request.prompt_tokens,
            seed=request.seed,
        )
        state = self.prefill(
            model=self.model,
            prompt_tokens=request.prompt_tokens,
            state=state,
            seed=request.seed,
        )
        previous = request.prompt_tokens[-1]
        tokens: list[int] = []
        token_times: list[int] = []
        for position in range(request.maximum_new_tokens):
            if time.perf_counter_ns() > deadline_ns:
                raise TimeoutError("ServingSynthBench CPU request exceeded run timeout")
            result = self.decode_step(
                model=self.model,
                previous_token=previous,
                state=state,
                position=position,
                seed=request.seed,
            )
            if not isinstance(result, dict) or set(result) != {"logits", "state"}:
                raise TypeError("generated decode_step violated its reference contract")
            state = result["state"]
            logits = result["logits"]
            sample_seed = _seed(request.seed, "sample", position)
            token = self.sample(logits=logits, seed=sample_seed)
            if not isinstance(token, int) or isinstance(token, bool):
                raise TypeError("generated sampler returned a non-integer token")
            tokens.append(token)
            token_times.append(time.perf_counter_ns())
            previous = token
        end = time.perf_counter_ns()
        ttft = max(1, token_times[0] - start)
        intervals = tuple(max(1, right - left) for left, right in pairwise(token_times))
        return tuple(tokens), max(1, end - start), ttft, intervals, None


class _GenesisExecutor:
    """Execute generated runtime requests only through the strict local sandbox."""

    def __init__(self, config_path: Path, sandbox_root: Path, *, seed: int) -> None:
        self.config_path = config_path.resolve(strict=True)
        self.runtime_root = self.config_path.parent
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.package_root = Path(str(config["reference_package_root"])).resolve(strict=True)
        self.repository_python = Path(__file__).resolve().parents[2]
        self.sandbox_root = sandbox_root.resolve()
        self.seed = seed
        self.call_index = 0

    def execute(
        self,
        request: WorkloadRequest,
        *,
        deadline_ns: int,
    ) -> tuple[tuple[int, ...], int, int, tuple[int, ...], tuple[str, str] | None]:
        remaining_seconds = (deadline_ns - time.perf_counter_ns()) / 1e9
        if remaining_seconds <= 0:
            raise TimeoutError("ServingSynthBench CPU request exceeded run timeout")
        call_id = self.call_index
        self.call_index += 1
        request_path = self.sandbox_root / "requests" / f"request-{call_id:05d}.json"
        request_path.parent.mkdir(parents=True, exist_ok=True)
        request_path.write_bytes(
            canonical_json(
                {
                    "request_id": request.request_id,
                    "prompt_tokens": request.prompt_tokens,
                    "maximum_new_tokens": request.maximum_new_tokens,
                    "seed": request.seed,
                }
            )
            + b"\n"
        )
        output = self.sandbox_root / "outputs" / f"request-{call_id:05d}"
        bounded_timeout = min(30.0, remaining_seconds)
        result = execute_sandboxed(
            SandboxRequest(
                argv=(
                    sys.executable,
                    "-m",
                    "sloforge.synthbench.runtime_runner",
                    "--config",
                    str(self.config_path),
                    "--request",
                    str(request_path),
                    "--seed",
                    str(self.seed),
                    "--timeout-seconds",
                    str(bounded_timeout),
                ),
                working_directory=self.runtime_root,
                read_only_paths=(
                    self.runtime_root,
                    self.package_root,
                    request_path,
                    self.repository_python,
                    Path(sys.prefix),
                    Path(sys.base_prefix),
                ),
                artifact_output_directory=output,
                seed=self.seed,
                limits=SandboxLimits(
                    wall_time_seconds=bounded_timeout,
                    cpu_time_seconds=max(1, min(30, int(bounded_timeout) + 1)),
                    memory_bytes=2 * 1024 * 1024 * 1024,
                    process_count=1,
                    output_bytes=64 * 1024,
                    artifact_bytes=1024 * 1024,
                    artifact_entries=16,
                    open_files=64,
                ),
            )
        )
        if not result.succeeded:
            raise RuntimeError(
                "sandboxed generated runtime failed: "
                f"{result.termination.value}: {result.stderr.strip()}"
            )
        runner_path = Path(__file__).with_name("runtime_runner.py").resolve(strict=True)
        execution_record = {
            "schema_version": "sloforge.synthbench.sandbox-execution/v1",
            "request_id": request.request_id,
            "request_seed": request.seed,
            "run_seed": self.seed,
            "config_sha256": _sha(self.config_path.read_bytes()),
            "request_path": str(request_path),
            "request_sha256": _sha(request_path.read_bytes()),
            "runner_sha256": _sha(runner_path.read_bytes()),
            "stdout_sha256": _sha(result.stdout.encode()),
            "stderr_sha256": _sha(result.stderr.encode()),
            "termination": result.termination.value,
            "return_code": result.return_code,
            "sandbox_backend": result.capabilities.backend.value,
            "capabilities": result.capabilities.model_dump(mode="json"),
            "sanitized_environment_names": result.sanitized_environment_names,
            "process_group_cleaned": result.process_group_cleaned,
            "output_bytes": result.output_bytes,
        }
        evidence_path = output / "sandbox-execution.json"
        evidence_payload = canonical_json(execution_record) + b"\n"
        evidence_path.write_bytes(evidence_payload)
        execution_evidence = (str(evidence_path.resolve()), _sha(evidence_payload))
        document = json.loads(result.stdout)
        tokens = tuple(int(token) for token in document["tokens"])
        intervals = tuple(int(value) for value in document["inter_token_ns"])
        return (
            tokens,
            int(document["latency_ns"]),
            int(document["ttft_ns"]),
            intervals,
            execution_evidence,
        )


_MEASURED_BASELINES = (
    BaselineKind.PYTHON_EAGER_REFERENCE,
    BaselineKind.GENERIC_RUNTIME,
    BaselineKind.TUNED_STATIC_SLOFORGE,
    BaselineKind.POLICY_ONLY,
    BaselineKind.WITHOUT_AUTOPSY,
    BaselineKind.WITHOUT_COUNTEREXAMPLE_LEARNING,
    BaselineKind.WITHOUT_LINEAGE,
    BaselineKind.GENESIS_FULL,
    BaselineKind.DETERMINISTIC_SINGLE_SHOT,
)

_UNAVAILABLE_REASONS = {
    BaselineKind.PYTORCH_EAGER_REFERENCE: "task grammar intentionally emits a dependency-free Python reference, not a torch.nn.Module",
    BaselineKind.TORCH_COMPILE: "task has no torch.export or torch.nn.Module surface in CPU smoke mode",
    BaselineKind.PHYSICAL_PLAN_ONLY: "CPU smoke task declares no GPU fabric topology",
    BaselineKind.KERNEL_ONLY: "CPU smoke task declares no generated GPU kernel target",
}


def _candidate_orders(
    baseline: BaselineKind, requests: tuple[WorkloadRequest, ...]
) -> tuple[tuple[WorkloadRequest, ...], ...]:
    fifo = requests
    shortest = tuple(sorted(requests, key=lambda item: (len(item.prompt_tokens), item.request_id)))
    deadline = tuple(
        sorted(requests, key=lambda item: (item.deadline_ms, -item.priority, item.request_id))
    )
    priority = tuple(
        sorted(requests, key=lambda item: (-item.priority, item.arrival_offset_ms, item.request_id))
    )
    prefix = tuple(
        sorted(
            requests,
            key=lambda item: (item.prompt_tokens[:2], len(item.prompt_tokens), item.request_id),
        )
    )
    if baseline in {
        BaselineKind.PYTHON_EAGER_REFERENCE,
        BaselineKind.GENERIC_RUNTIME,
        BaselineKind.DETERMINISTIC_SINGLE_SHOT,
    }:
        return (fifo,)
    if baseline is BaselineKind.TUNED_STATIC_SLOFORGE:
        return (fifo, shortest, deadline)
    if baseline is BaselineKind.POLICY_ONLY:
        return (deadline, priority)
    if baseline is BaselineKind.GENESIS_FULL:
        return (deadline, priority, shortest, prefix)
    return (fifo, shortest)


def _order_score(order: tuple[WorkloadRequest, ...]) -> float:
    return sum(
        (index + 1) * (len(request.prompt_tokens) + request.maximum_new_tokens)
        for index, request in enumerate(order)
    )


def _selected_order(
    baseline: BaselineKind, requests: tuple[WorkloadRequest, ...]
) -> tuple[tuple[WorkloadRequest, ...], int]:
    candidates = _candidate_orders(baseline, requests)
    selected = min(
        candidates,
        key=lambda order: (_order_score(order), tuple(item.request_id for item in order)),
    )
    return selected, len(candidates)


def _write_samples(path: Path, samples: tuple[RawCpuSample, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite raw benchmark samples: {path}")
    path.write_bytes(b"".join(canonical_json(sample) + b"\n" for sample in samples))


def _percentile(values: list[int], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(probability * len(ordered)) - 1))
    return float(ordered[index])


def _summary_from_artifact(
    baseline: BaselineKind,
    path: Path,
    *,
    candidate_count: int,
) -> BaselineSummary:
    samples = tuple(
        RawCpuSample.model_validate_json(line, strict=True)
        for line in path.read_bytes().splitlines()
        if line
    )
    latencies = [sample.latency_ns for sample in samples]
    ttfts = [sample.ttft_ns for sample in samples]
    intervals = [value for sample in samples for value in sample.inter_token_ns]
    surface = samples[0].execution_surface
    return BaselineSummary(
        baseline=baseline,
        status=BaselineStatus.MEASURED,
        reason=(
            "direct dependency-free eager reference execution; summary derived from complete "
            "raw CPU samples"
            if baseline is BaselineKind.PYTHON_EAGER_REFERENCE
            else "actual generated bounded Genesis runtime execution; CPU smoke validates the "
            "zero-day runtime path but does not represent optimized search or a speedup claim"
            if baseline is BaselineKind.GENESIS_FULL
            else f"CPU smoke request-order surrogate for {baseline.value}; uses the same "
            "dependency-free reference executor and does not represent an independent engine, "
            "kernel, or full ablation implementation"
        ),
        raw_samples_path=str(path.resolve()),
        raw_samples_sha256=_sha(path.read_bytes()),
        execution_surface=surface,
        sample_count=len(samples),
        valid_request_rate=(sum(sample.exact_match for sample in samples) / len(samples)),
        median_latency_ns=float(statistics.median(latencies)),
        p95_latency_ns=_percentile(latencies, 0.95),
        median_ttft_ns=float(statistics.median(ttfts)),
        median_inter_token_ns=float(statistics.median(intervals)) if intervals else 0.0,
        measured_cpu_seconds=sum(latencies) / 1_000_000_000,
        candidate_count=candidate_count,
        human_authored_model_specific_lines=0,
    )


def _unavailable(baseline: BaselineKind, reason: str) -> BaselineSummary:
    return BaselineSummary(
        baseline=baseline,
        status=BaselineStatus.NOT_APPLICABLE,
        reason=reason,
        raw_samples_path=None,
        raw_samples_sha256=None,
        execution_surface="not_applicable",
        sample_count=0,
        valid_request_rate=0.0,
        median_latency_ns=0.0,
        p95_latency_ns=0.0,
        median_ttft_ns=0.0,
        median_inter_token_ns=0.0,
        measured_cpu_seconds=0.0,
        candidate_count=0,
        human_authored_model_specific_lines=0,
    )


def _hidden_summary_from_artifact(baseline: BaselineKind, path: Path) -> HiddenEvaluationSummary:
    results = tuple(
        HiddenCaseResult.model_validate_json(line, strict=True)
        for line in path.read_bytes().splitlines()
        if line
    )
    exact = sum(result.exact_match for result in results)
    return HiddenEvaluationSummary(
        baseline=baseline,
        status=BaselineStatus.MEASURED,
        evidence_path=str(path.resolve()),
        evidence_sha256=_sha(path.read_bytes()),
        case_count=len(results),
        exact_case_rate=exact / len(results) if results else 0.0,
        escaped_regressions=len(results) - exact,
    )


def _unavailable_hidden(baseline: BaselineKind) -> HiddenEvaluationSummary:
    return HiddenEvaluationSummary(
        baseline=baseline,
        status=BaselineStatus.NOT_APPLICABLE,
        evidence_path=None,
        evidence_sha256=None,
        case_count=0,
        exact_case_rate=0.0,
        escaped_regressions=0,
    )


def _decode_measurement_order(values: tuple[str, ...]) -> tuple[tuple[int, BaselineKind], ...]:
    decoded: list[tuple[int, BaselineKind]] = []
    for value in values:
        try:
            repetition_text, baseline_text = value.split(":", 1)
            decoded.append((int(repetition_text), BaselineKind(baseline_text)))
        except (TypeError, ValueError) as error:
            raise ValueError("invalid SynthBench measurement run order") from error
    return tuple(decoded)


def _validate_sandbox_execution_evidence(
    path_value: str | None,
    digest: str | None,
    *,
    request_id: str,
    request_seed: int,
    run_seed: int,
    config_sha256: str,
) -> None:
    if path_value is None or digest is None:
        raise ValueError("generated execution is missing sandbox evidence")
    path = Path(path_value)
    if not path.is_file() or _sha(path.read_bytes()) != digest:
        raise ValueError("generated sandbox execution evidence is unavailable or changed")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("sandbox execution evidence must be an object")
    if (
        document.get("schema_version") != "sloforge.synthbench.sandbox-execution/v1"
        or document.get("request_id") != request_id
        or document.get("request_seed") != request_seed
        or document.get("run_seed") != run_seed
        or document.get("config_sha256") != config_sha256
        or document.get("termination") != "success"
        or document.get("return_code") != 0
        or document.get("sandbox_backend") in {None, "none"}
        or document.get("process_group_cleaned") is not True
    ):
        raise ValueError("sandbox execution evidence does not match the raw sample")
    request_path_value = document.get("request_path")
    request_digest = document.get("request_sha256")
    if not isinstance(request_path_value, str) or not isinstance(request_digest, str):
        raise ValueError("sandbox evidence omits its oracle-free request identity")
    request_path = Path(request_path_value)
    if not request_path.is_file() or _sha(request_path.read_bytes()) != request_digest:
        raise ValueError("sandbox request artifact is unavailable or changed")
    request_document = json.loads(request_path.read_text(encoding="utf-8"))
    if not isinstance(request_document, dict) or set(request_document) != {
        "request_id",
        "prompt_tokens",
        "maximum_new_tokens",
        "seed",
    }:
        raise ValueError("sandbox request contains undeclared or evaluator-only fields")
    runner_path = Path(__file__).with_name("runtime_runner.py").resolve(strict=True)
    if document.get("runner_sha256") != _sha(runner_path.read_bytes()):
        raise ValueError("sandbox execution used an unrecognized trusted runner")


def _aggregate(reports: tuple[TaskRunReport, ...]) -> AggregateMetrics:
    measured = [
        baseline
        for report in reports
        for baseline in report.baselines
        if baseline.status is BaselineStatus.MEASURED
    ]
    unavailable = [
        baseline
        for report in reports
        for baseline in report.baselines
        if baseline.status is BaselineStatus.NOT_APPLICABLE
    ]
    genesis = [
        (report, baseline)
        for report in reports
        for baseline in report.baselines
        if baseline.status is BaselineStatus.MEASURED
        and baseline.baseline is BaselineKind.GENESIS_FULL
    ]
    sample_count = sum(baseline.sample_count for _, baseline in genesis)
    exact_count = sum(
        round(baseline.valid_request_rate * baseline.sample_count) for _, baseline in genesis
    )
    return AggregateMetrics(
        measured_baseline_runs=len(measured),
        unavailable_baseline_runs=len(unavailable),
        valid_system_rate=(
            sum(
                baseline.valid_request_rate == 1.0
                and next(
                    hidden.exact_case_rate
                    for hidden in report.hidden_evaluations
                    if hidden.baseline is BaselineKind.GENESIS_FULL
                )
                == 1.0
                for report, baseline in genesis
            )
            / len(genesis)
            if genesis
            else 0.0
        ),
        exact_request_rate=exact_count / sample_count if sample_count else 0.0,
        measured_cpu_seconds=sum(baseline.measured_cpu_seconds for baseline in measured),
        task_count=len({report.task_id for report in reports}),
        distinct_task_seeds=len({report.task_generation_seed for report in reports}),
    )


def validate_cpu_benchmark_report(report: SynthBenchReport) -> None:
    """Reopen raw artifacts and independently reconstruct every report summary."""

    expected_baselines = set(BaselineKind)
    if {item.run_seed for item in report.tasks} != set(report.run_seeds):
        raise ValueError("report task seeds do not match declared run seeds")
    task_seed_pairs = [(item.task_id, item.run_seed) for item in report.tasks]
    if len(task_seed_pairs) != len(set(task_seed_pairs)):
        raise ValueError("report repeats a task/run-seed pair")
    for task_id in {item.task_id for item in report.tasks}:
        variants = tuple(item for item in report.tasks if item.task_id == task_id)
        if {item.run_seed for item in variants} != set(report.run_seeds):
            raise ValueError("report omits a task/run-seed execution")
        identities = {
            (
                item.task_hash,
                item.public_package_hash,
                item.reference_package_hash,
                item.task_generation_seed,
            )
            for item in variants
        }
        if len(identities) != 1:
            raise ValueError("task identity changes across run seeds")
    for task in report.tasks:
        if {item.baseline for item in task.baselines} != expected_baselines:
            raise ValueError("report baseline set is incomplete")
        if {item.baseline for item in task.hidden_evaluations} != expected_baselines:
            raise ValueError("report hidden-evaluation set is incomplete")
        manifest_path = Path(task.genesis_runtime_manifest_path)
        if (
            not manifest_path.is_file()
            or _sha(manifest_path.read_bytes()) != task.genesis_runtime_manifest_sha256
        ):
            raise ValueError("generated Genesis runtime manifest is unavailable or changed")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            not isinstance(manifest, dict)
            or manifest.get("package_hash") != task.reference_package_hash
        ):
            raise ValueError("generated runtime is bound to another public package")
        runtime_config_path = manifest_path.parent / "runtime_config.json"
        if not runtime_config_path.is_file():
            raise ValueError("generated runtime configuration is unavailable")
        runtime_config_sha256 = _sha(runtime_config_path.read_bytes())
        all_samples: list[RawCpuSample] = []
        for summary in task.baselines:
            if summary.status is not BaselineStatus.MEASURED:
                continue
            if summary.raw_samples_path is None or summary.raw_samples_sha256 is None:
                raise ValueError("measured baseline is missing raw evidence")
            path = Path(summary.raw_samples_path)
            if not path.is_file() or _sha(path.read_bytes()) != summary.raw_samples_sha256:
                raise ValueError("SynthBench raw samples are unavailable or changed")
            reconstructed = _summary_from_artifact(
                summary.baseline, path, candidate_count=summary.candidate_count
            )
            if reconstructed != summary:
                raise ValueError("baseline summary is not derived from its raw samples")
            loaded_samples = tuple(
                RawCpuSample.model_validate_json(line, strict=True)
                for line in path.read_bytes().splitlines()
                if line
            )
            for sample in loaded_samples:
                if sample.execution_surface == "genesis_generated_runtime":
                    _validate_sandbox_execution_evidence(
                        sample.execution_evidence_path,
                        sample.execution_evidence_sha256,
                        request_id=sample.request_id,
                        request_seed=sample.request_seed,
                        run_seed=sample.run_seed,
                        config_sha256=runtime_config_sha256,
                    )
            all_samples.extend(loaded_samples)
        for hidden in task.hidden_evaluations:
            if hidden.status is not BaselineStatus.MEASURED:
                continue
            if hidden.evidence_path is None or hidden.evidence_sha256 is None:
                raise ValueError("measured hidden evaluation is missing evidence")
            path = Path(hidden.evidence_path)
            if not path.is_file() or _sha(path.read_bytes()) != hidden.evidence_sha256:
                raise ValueError("SynthBench hidden evidence is unavailable or changed")
            if _hidden_summary_from_artifact(hidden.baseline, path) != hidden:
                raise ValueError("hidden summary is not derived from its evidence")
            hidden_results = tuple(
                HiddenCaseResult.model_validate_json(line, strict=True)
                for line in path.read_bytes().splitlines()
                if line
            )
            for result in hidden_results:
                if result.execution_surface == "genesis_generated_runtime":
                    _validate_sandbox_execution_evidence(
                        result.execution_evidence_path,
                        result.execution_evidence_sha256,
                        request_id=result.request_id,
                        request_seed=result.request_seed,
                        run_seed=task.run_seed,
                        config_sha256=runtime_config_sha256,
                    )
        rebuilt_integrity = audit_raw_samples(
            tuple(all_samples),
            measured_baselines=_MEASURED_BASELINES,
            workload_fingerprint=task.integrity.workload_fingerprint,
            environment_fingerprint=task.integrity.environment_fingerprint,
            task_hash=task.task_hash,
            run_seed=task.run_seed,
            repetitions=task.repetitions,
            request_ids=task.integrity.expected_request_ids,
            measurement_run_order=_decode_measurement_order(task.measurement_run_order),
        )
        if rebuilt_integrity != task.integrity or not rebuilt_integrity.passed:
            raise ValueError("report integrity result is not independently reproducible")
    if _aggregate(report.tasks) != report.metrics:
        raise ValueError("aggregate metrics are not derived from task evidence")


def _run_task(
    task_directory: Path,
    descriptor: TaskDescriptor,
    *,
    run_seed: int,
    configuration: CpuRunConfiguration,
    output_directory: Path,
) -> TaskRunReport:
    verify_public_package(task_directory, descriptor)
    workload_path = task_directory / descriptor.workload_path
    workload = load_workload(task_directory, descriptor)
    hidden_cases = load_hidden_cases(task_directory, descriptor)
    workload_fingerprint = _sha(workload_path.read_bytes())
    environment = _environment_fingerprint()
    task_hash = _sha(canonical_json(descriptor))
    runtime_root = output_directory / "generated-runtimes" / descriptor.task_id / f"seed-{run_seed}"
    inspection = inspect_reference_package(task_directory / descriptor.public_package_path)
    initialized = initialize_genesis_run(
        task_directory / descriptor.public_package_path,
        inspection,
        runtime_root,
        seed=run_seed,
        workload_contract_hash=workload_fingerprint,
    )
    runtime_manifest = initialized.runtime.output_directory / "artifact_manifest.json"
    special_case_audit = audit_special_casing(
        initialized.runtime.output_directory / "runtime.py", descriptor, hidden_cases
    )
    if not special_case_audit.passed:
        raise ValueError("generated Genesis runtime contains task-specific special casing")
    executors: dict[BaselineKind, _Executor | _GenesisExecutor] = {
        baseline: (
            _GenesisExecutor(
                initialized.runtime.output_directory / "runtime_config.json",
                output_directory / "sandbox" / descriptor.task_id / f"seed-{run_seed}" / "public",
                seed=run_seed,
            )
            if baseline is BaselineKind.GENESIS_FULL
            else _Executor(
                task_directory,
                descriptor,
                identity_suffix=f"{run_seed}_{baseline.value}",
            )
        )
        for baseline in _MEASURED_BASELINES
    }
    deadline_ns = time.perf_counter_ns() + int(configuration.maximum_runtime_seconds * 1e9)
    for baseline in _MEASURED_BASELINES:
        executor = executors[baseline]
        selected, _ = _selected_order(baseline, workload)
        for _ in range(configuration.warmup_count):
            for request in selected:
                executor.execute(request, deadline_ns=deadline_ns)
    run_order = [
        (repetition, baseline)
        for repetition in range(configuration.repetitions)
        for baseline in _MEASURED_BASELINES
    ]
    random.Random(_seed(run_seed, descriptor.task_id, "run-order")).shuffle(run_order)
    samples_by_baseline: dict[BaselineKind, list[RawCpuSample]] = {
        baseline: [] for baseline in _MEASURED_BASELINES
    }
    for measurement_order_ordinal, (repetition, baseline) in enumerate(run_order):
        executor = executors[baseline]
        selected, _ = _selected_order(baseline, workload)
        for request in selected:
            observed, latency, ttft, intervals, execution_evidence = executor.execute(
                request, deadline_ns=deadline_ns
            )
            baseline_samples = samples_by_baseline[baseline]
            baseline_samples.append(
                RawCpuSample(
                    task_id=descriptor.task_id,
                    task_hash=task_hash,
                    workload_fingerprint=workload_fingerprint,
                    environment_fingerprint=environment,
                    baseline=baseline,
                    run_seed=run_seed,
                    repetition=repetition,
                    measurement_order_ordinal=measurement_order_ordinal,
                    execution_ordinal=len(baseline_samples),
                    request_id=request.request_id,
                    request_seed=request.seed,
                    latency_ns=latency,
                    ttft_ns=ttft,
                    inter_token_ns=intervals,
                    observed_tokens=observed,
                    expected_tokens=request.expected_tokens,
                    exact_match=observed == request.expected_tokens,
                    execution_surface=(
                        "python_eager_reference"
                        if baseline is BaselineKind.PYTHON_EAGER_REFERENCE
                        else "genesis_generated_runtime"
                        if baseline is BaselineKind.GENESIS_FULL
                        else "reference_order_surrogate"
                    ),
                    execution_evidence_path=(
                        execution_evidence[0] if execution_evidence is not None else None
                    ),
                    execution_evidence_sha256=(
                        execution_evidence[1] if execution_evidence is not None else None
                    ),
                )
            )
    summaries: list[BaselineSummary] = []
    hidden_summaries: list[HiddenEvaluationSummary] = []
    all_samples: list[RawCpuSample] = []
    for baseline in BaselineKind:
        if baseline in _UNAVAILABLE_REASONS:
            summaries.append(_unavailable(baseline, _UNAVAILABLE_REASONS[baseline]))
            hidden_summaries.append(_unavailable_hidden(baseline))
            continue
        persisted_samples = tuple(samples_by_baseline[baseline])
        path = (
            output_directory
            / "raw"
            / descriptor.task_id
            / f"seed-{run_seed}"
            / f"{baseline.value}.jsonl"
        )
        _write_samples(path, persisted_samples)
        _, candidate_count = _selected_order(baseline, workload)
        if baseline is BaselineKind.GENESIS_FULL:
            candidate_count = 1
        summaries.append(_summary_from_artifact(baseline, path, candidate_count=candidate_count))
        all_samples.extend(persisted_samples)
        hidden_requests = tuple(case.request for case in hidden_cases)
        selected_hidden, _ = _selected_order(baseline, hidden_requests)
        case_by_request = {case.request.request_id: case for case in hidden_cases}
        hidden_results: list[HiddenCaseResult] = []
        hidden_executor: _Executor | _GenesisExecutor = (
            _GenesisExecutor(
                initialized.runtime.output_directory / "runtime_config.json",
                output_directory / "sandbox" / descriptor.task_id / f"seed-{run_seed}" / "hidden",
                seed=run_seed,
            )
            if baseline is BaselineKind.GENESIS_FULL
            else _Executor(
                task_directory,
                descriptor,
                identity_suffix=f"{run_seed}_{baseline.value}_hidden",
            )
        )
        for request in selected_hidden:
            observed, _, _, _, execution_evidence = hidden_executor.execute(
                request, deadline_ns=deadline_ns
            )
            hidden_results.append(
                HiddenCaseResult(
                    case_id=case_by_request[request.request_id].case_id,
                    baseline=baseline,
                    request_id=request.request_id,
                    request_seed=request.seed,
                    observed_tokens=observed,
                    expected_tokens=request.expected_tokens,
                    exact_match=observed == request.expected_tokens,
                    execution_surface=(
                        "python_eager_reference"
                        if baseline is BaselineKind.PYTHON_EAGER_REFERENCE
                        else "genesis_generated_runtime"
                        if baseline is BaselineKind.GENESIS_FULL
                        else "reference_order_surrogate"
                    ),
                    execution_evidence_path=(
                        execution_evidence[0] if execution_evidence is not None else None
                    ),
                    execution_evidence_sha256=(
                        execution_evidence[1] if execution_evidence is not None else None
                    ),
                )
            )
        hidden_path = (
            output_directory
            / "hidden-evidence"
            / descriptor.task_id
            / f"seed-{run_seed}"
            / f"{baseline.value}.jsonl"
        )
        hidden_path.parent.mkdir(parents=True, exist_ok=True)
        if hidden_path.exists():
            raise FileExistsError(f"refusing to overwrite hidden evidence: {hidden_path}")
        hidden_path.write_bytes(
            b"".join(canonical_json(result) + b"\n" for result in hidden_results)
        )
        hidden_summaries.append(_hidden_summary_from_artifact(baseline, hidden_path))
    integrity = audit_raw_samples(
        tuple(all_samples),
        measured_baselines=_MEASURED_BASELINES,
        workload_fingerprint=workload_fingerprint,
        environment_fingerprint=environment,
        task_hash=task_hash,
        run_seed=run_seed,
        repetitions=configuration.repetitions,
        request_ids=tuple(request.request_id for request in workload),
        measurement_run_order=tuple(run_order),
    )
    return TaskRunReport(
        task_id=descriptor.task_id,
        task_hash=task_hash,
        public_package_hash=descriptor.public_package_hash,
        reference_package_hash=inspection.package_hash,
        task_generation_seed=descriptor.seed,
        run_seed=run_seed,
        warmup_count=configuration.warmup_count,
        repetitions=configuration.repetitions,
        measurement_run_order=tuple(
            f"{repetition}:{baseline.value}" for repetition, baseline in run_order
        ),
        baselines=tuple(summaries),
        hidden_evaluations=tuple(hidden_summaries),
        integrity=integrity,
        genesis_runtime_manifest_path=str(runtime_manifest.resolve()),
        genesis_runtime_manifest_sha256=_sha(runtime_manifest.read_bytes()),
    )


def run_cpu_benchmark(
    task_directories: tuple[Path, ...],
    configuration: CpuRunConfiguration,
    output_directory: Path,
) -> SynthBenchReport:
    """Run bounded CPU tasks and derive the report only from written raw samples."""

    selected = task_directories[: configuration.maximum_tasks]
    reports = tuple(
        _run_task(
            task_directory,
            load_task(task_directory / "task.json"),
            run_seed=run_seed,
            configuration=configuration,
            output_directory=output_directory,
        )
        for task_directory in selected
        for run_seed in configuration.seeds
    )
    metrics = _aggregate(reports)
    result = SynthBenchReport(
        run_seeds=configuration.seeds,
        tasks=reports,
        metrics=metrics,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    report_path = output_directory / "report.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite synthbench report: {report_path}")
    report_path.write_bytes(canonical_json(result) + b"\n")
    persisted = SynthBenchReport.model_validate_json(report_path.read_bytes(), strict=True)
    validate_cpu_benchmark_report(persisted)
    return persisted
