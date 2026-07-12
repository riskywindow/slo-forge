"""CPU-only ServingSynthBench runner with raw measured evidence and local baselines."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import random
import statistics
import sys
import time
from pathlib import Path

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

_MAXIMUM_REPORT_ARTIFACT_BYTES = 32 * 1024 * 1024
_MAXIMUM_RUNTIME_ARTIFACT_BYTES = 8 * 1024 * 1024


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _contained_regular_file(
    path_value: str | Path,
    artifact_root: Path,
    *,
    label: str,
    maximum_bytes: int = _MAXIMUM_REPORT_ARTIFACT_BYTES,
) -> Path:
    root = artifact_root.resolve(strict=True)
    unresolved = Path(path_value)
    if not unresolved.is_absolute():
        raise ValueError(f"{label} path must be absolute")
    if unresolved.is_symlink():
        raise ValueError(f"{label} cannot be a symlink")
    try:
        lexical_relative = unresolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes the benchmark artifact root") from error
    cursor = root
    for part in lexical_relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"{label} cannot traverse a symlink")
    path = unresolved.resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes the benchmark artifact root") from error
    if not path.is_file() or path.stat().st_size > maximum_bytes:
        raise ValueError(f"{label} is not a bounded regular file")
    return path


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


def _persist_execution_evidence(
    *,
    result_stdout: str,
    result_stderr: str,
    result_termination: str,
    result_return_code: int | None,
    result_backend: str,
    result_capabilities: dict[str, object],
    sanitized_environment_names: tuple[str, ...],
    process_group_cleaned: bool,
    output_bytes: int,
    output: Path,
    request_path: Path,
    request: WorkloadRequest,
    run_seed: int,
    execution_surface: str,
    subject_sha256: str,
    runner_path: Path,
) -> tuple[str, str]:
    stdout_path = output / "runner-stdout.json"
    stderr_path = output / "runner-stderr.txt"
    stdout_payload = result_stdout.encode()
    stderr_payload = result_stderr.encode()
    stdout_path.write_bytes(stdout_payload)
    stderr_path.write_bytes(stderr_payload)
    execution_record = {
        "schema_version": "sloforge.synthbench.sandbox-execution/v2",
        "execution_surface": execution_surface,
        "subject_sha256": subject_sha256,
        "request_id": request.request_id,
        "request_seed": request.seed,
        "run_seed": run_seed,
        "request_path": str(request_path.resolve()),
        "request_sha256": _sha(request_path.read_bytes()),
        "runner_sha256": _sha(runner_path.read_bytes()),
        "stdout_sha256": _sha(stdout_payload),
        "stderr_sha256": _sha(stderr_payload),
        "stdout_path": str(stdout_path.resolve()),
        "stderr_path": str(stderr_path.resolve()),
        "termination": result_termination,
        "return_code": result_return_code,
        "sandbox_backend": result_backend,
        "capabilities": result_capabilities,
        "sanitized_environment_names": sanitized_environment_names,
        "process_group_cleaned": process_group_cleaned,
        "output_bytes": output_bytes,
    }
    evidence_path = output / "sandbox-execution.json"
    evidence_payload = canonical_json(execution_record) + b"\n"
    evidence_path.write_bytes(evidence_payload)
    return str(evidence_path.resolve()), _sha(evidence_payload)


class _Executor:
    def __init__(
        self,
        task_directory: Path,
        descriptor: TaskDescriptor,
        *,
        sandbox_root: Path,
        run_seed: int,
        execution_surface: str,
    ) -> None:
        self.descriptor = descriptor
        self.reference_root = (task_directory / descriptor.public_package_path).resolve(strict=True)
        self.repository_python = Path(__file__).resolve().parents[2]
        self.sandbox_root = sandbox_root.resolve()
        self.run_seed = run_seed
        self.execution_surface = execution_surface
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
                    "sloforge.synthbench.reference_runner",
                    "--reference-package",
                    str(self.reference_root),
                    "--request",
                    str(request_path),
                    "--model-seed",
                    str(self.descriptor.seed),
                    "--timeout-seconds",
                    str(bounded_timeout),
                ),
                working_directory=self.reference_root,
                read_only_paths=(
                    self.reference_root,
                    request_path,
                    self.repository_python,
                    Path(sys.prefix),
                    Path(sys.base_prefix),
                ),
                artifact_output_directory=output,
                seed=self.run_seed,
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
                "sandboxed reference runtime failed: "
                f"{result.termination.value}: {result.stderr.strip()}"
            )
        evidence = _persist_execution_evidence(
            result_stdout=result.stdout,
            result_stderr=result.stderr,
            result_termination=result.termination.value,
            result_return_code=result.return_code,
            result_backend=result.capabilities.backend.value,
            result_capabilities=result.capabilities.model_dump(mode="json"),
            sanitized_environment_names=result.sanitized_environment_names,
            process_group_cleaned=result.process_group_cleaned,
            output_bytes=result.output_bytes,
            output=output,
            request_path=request_path,
            request=request,
            run_seed=self.run_seed,
            execution_surface=self.execution_surface,
            subject_sha256=self.descriptor.public_package_hash,
            runner_path=Path(__file__).with_name("reference_runner.py").resolve(strict=True),
        )
        document = json.loads(result.stdout)
        return (
            tuple(int(token) for token in document["tokens"]),
            int(document["latency_ns"]),
            int(document["ttft_ns"]),
            tuple(int(value) for value in document["inter_token_ns"]),
            evidence,
        )


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
        execution_evidence = _persist_execution_evidence(
            result_stdout=result.stdout,
            result_stderr=result.stderr,
            result_termination=result.termination.value,
            result_return_code=result.return_code,
            result_backend=result.capabilities.backend.value,
            result_capabilities=result.capabilities.model_dump(mode="json"),
            sanitized_environment_names=result.sanitized_environment_names,
            process_group_cleaned=result.process_group_cleaned,
            output_bytes=result.output_bytes,
            output=output,
            request_path=request_path,
            request=request,
            run_seed=self.seed,
            execution_surface="genesis_generated_runtime",
            subject_sha256=_sha(self.config_path.read_bytes()),
            runner_path=runner_path,
        )
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

_EXECUTED_BASELINES = (
    BaselineKind.PYTHON_EAGER_REFERENCE,
    BaselineKind.GENESIS_FULL,
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
        status=(
            BaselineStatus.MEASURED
            if baseline in {BaselineKind.PYTHON_EAGER_REFERENCE, BaselineKind.GENESIS_FULL}
            else BaselineStatus.SURROGATE
        ),
        reason=(
            "direct dependency-free eager reference execution; summary derived from complete "
            "raw CPU samples"
            if baseline is BaselineKind.PYTHON_EAGER_REFERENCE
            else "actual generated bounded Genesis runtime execution; CPU smoke validates the "
            "zero-day runtime path but does not represent optimized search or a speedup claim"
            if baseline is BaselineKind.GENESIS_FULL
            else f"CPU smoke request-order surrogate for {baseline.value}; uses bound "
            "eager-reference observations without another timed execution and does not "
            "represent an independent engine, kernel, or full ablation implementation"
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
        observed_request_wall_seconds=(
            sum(latencies) / 1_000_000_000 if baseline in _EXECUTED_BASELINES else 0.0
        ),
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
        observed_request_wall_seconds=0.0,
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
        status=(
            BaselineStatus.MEASURED
            if baseline in {BaselineKind.PYTHON_EAGER_REFERENCE, BaselineKind.GENESIS_FULL}
            else BaselineStatus.SURROGATE
        ),
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


def _measurement_order(
    *, run_seed: int, task_id: str, repetitions: int
) -> tuple[tuple[int, BaselineKind], ...]:
    order = [
        (repetition, baseline)
        for repetition in range(repetitions)
        for baseline in _MEASURED_BASELINES
    ]
    random.Random(_seed(run_seed, task_id, "run-order")).shuffle(order)
    return tuple(order)


def _validate_sandbox_execution_evidence(
    path_value: str | None,
    digest: str | None,
    *,
    expected_request: WorkloadRequest,
    run_seed: int,
    expected_execution_surface: str,
    expected_subject_sha256: str,
    observed_tokens: tuple[int, ...],
    latency_ns: int | None,
    ttft_ns: int | None,
    inter_token_ns: tuple[int, ...] | None,
    artifact_root: Path,
) -> None:
    if path_value is None or digest is None:
        raise ValueError("generated execution is missing sandbox evidence")
    path = _contained_regular_file(
        path_value, artifact_root, label="generated sandbox execution evidence"
    )
    if _sha(path.read_bytes()) != digest:
        raise ValueError("generated sandbox execution evidence is unavailable or changed")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("sandbox execution evidence must be an object")
    if (
        document.get("schema_version") != "sloforge.synthbench.sandbox-execution/v2"
        or document.get("execution_surface") != expected_execution_surface
        or document.get("subject_sha256") != expected_subject_sha256
        or document.get("request_id") != expected_request.request_id
        or document.get("request_seed") != expected_request.seed
        or document.get("run_seed") != run_seed
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
    request_path = _contained_regular_file(
        request_path_value,
        artifact_root,
        label="sandbox request artifact",
        maximum_bytes=1024 * 1024,
    )
    if _sha(request_path.read_bytes()) != request_digest:
        raise ValueError("sandbox request artifact is unavailable or changed")
    request_document = json.loads(request_path.read_text(encoding="utf-8"))
    if not isinstance(request_document, dict) or set(request_document) != {
        "request_id",
        "prompt_tokens",
        "maximum_new_tokens",
        "seed",
    }:
        raise ValueError("sandbox request contains undeclared or evaluator-only fields")
    if request_document != {
        "request_id": expected_request.request_id,
        "prompt_tokens": list(expected_request.prompt_tokens),
        "maximum_new_tokens": expected_request.maximum_new_tokens,
        "seed": expected_request.seed,
    }:
        raise ValueError("sandbox request does not match the benchmark task input")
    stdout_path_value = document.get("stdout_path")
    stdout_digest = document.get("stdout_sha256")
    if not isinstance(stdout_path_value, str) or not isinstance(stdout_digest, str):
        raise ValueError("sandbox evidence omits the retained runner response")
    stdout_path = _contained_regular_file(
        stdout_path_value,
        artifact_root,
        label="sandbox runner response",
        maximum_bytes=1024 * 1024,
    )
    if _sha(stdout_path.read_bytes()) != stdout_digest:
        raise ValueError("sandbox runner response is unavailable or changed")
    stderr_path_value = document.get("stderr_path")
    stderr_digest = document.get("stderr_sha256")
    if not isinstance(stderr_path_value, str) or not isinstance(stderr_digest, str):
        raise ValueError("sandbox evidence omits the retained runner diagnostics")
    stderr_path = _contained_regular_file(
        stderr_path_value,
        artifact_root,
        label="sandbox runner diagnostics",
        maximum_bytes=1024 * 1024,
    )
    if _sha(stderr_path.read_bytes()) != stderr_digest:
        raise ValueError("sandbox runner diagnostics are unavailable or changed")
    response = json.loads(stdout_path.read_text(encoding="utf-8"))
    if not isinstance(response, dict) or set(response) != {
        "tokens",
        "latency_ns",
        "ttft_ns",
        "inter_token_ns",
    }:
        raise ValueError("sandbox runner response has an invalid schema")
    if tuple(response["tokens"]) != observed_tokens:
        raise ValueError("raw observed tokens do not match the sandbox runner response")
    if latency_ns is not None and response["latency_ns"] != latency_ns:
        raise ValueError("raw latency does not match the sandbox runner response")
    if ttft_ns is not None and response["ttft_ns"] != ttft_ns:
        raise ValueError("raw TTFT does not match the sandbox runner response")
    if inter_token_ns is not None and tuple(response["inter_token_ns"]) != inter_token_ns:
        raise ValueError("raw inter-token timings do not match the sandbox runner response")
    runner_path = (
        Path(__file__)
        .with_name(
            "runtime_runner.py"
            if expected_execution_surface == "genesis_generated_runtime"
            else "reference_runner.py"
        )
        .resolve(strict=True)
    )
    if document.get("runner_sha256") != _sha(runner_path.read_bytes()):
        raise ValueError("sandbox execution used an unrecognized trusted runner")


def _validate_runtime_manifest(path: Path, *, artifact_root: Path) -> None:
    document = json.loads(path.read_text(encoding="utf-8"))
    artifacts = document.get("artifacts") if isinstance(document, dict) else None
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("generated runtime manifest has no artifact identities")
    for item in artifacts:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not isinstance(item[1], str)
        ):
            raise ValueError("generated runtime manifest contains an invalid artifact identity")
        relative = Path(item[0])
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError("generated runtime manifest artifact path is not relative")
        artifact = _contained_regular_file(
            path.parent / relative,
            artifact_root,
            label="generated runtime artifact",
            maximum_bytes=_MAXIMUM_RUNTIME_ARTIFACT_BYTES,
        )
        if _sha(artifact.read_bytes()) != item[1]:
            raise ValueError("generated runtime artifact is unavailable or changed")


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
    surrogates = [
        baseline
        for report in reports
        for baseline in report.baselines
        if baseline.status is BaselineStatus.SURROGATE
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
        surrogate_baseline_runs=len(surrogates),
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
        observed_request_wall_seconds=sum(
            baseline.observed_request_wall_seconds for baseline in (*measured, *surrogates)
        ),
        task_count=len({report.task_id for report in reports}),
        distinct_task_seeds=len({report.task_generation_seed for report in reports}),
    )


def validate_cpu_benchmark_report(report: SynthBenchReport, *, artifact_root: Path) -> None:
    """Reopen raw artifacts and independently reconstruct every report summary."""

    if artifact_root.is_symlink() or not artifact_root.is_dir():
        raise ValueError("benchmark artifact root must be a regular non-symlinked directory")
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
        descriptor_path = Path(task.task_descriptor_path)
        if (
            descriptor_path.is_symlink()
            or not descriptor_path.is_file()
            or _sha(descriptor_path.read_bytes()) != task.task_descriptor_sha256
        ):
            raise ValueError("SynthBench task descriptor is unavailable or changed")
        descriptor = load_task(descriptor_path)
        task_directory = descriptor_path.parent
        if (
            descriptor.task_id != task.task_id
            or _sha(canonical_json(descriptor)) != task.task_hash
            or descriptor.public_package_hash != task.public_package_hash
            or descriptor.seed != task.task_generation_seed
        ):
            raise ValueError("report task identity does not match its descriptor")
        verify_public_package(task_directory, descriptor)
        workload = load_workload(task_directory, descriptor)
        workload_by_id = {request.request_id: request for request in workload}
        hidden_cases = load_hidden_cases(task_directory, descriptor)
        hidden_by_id = {case.case_id: case for case in hidden_cases}
        manifest_path = _contained_regular_file(
            task.genesis_runtime_manifest_path,
            artifact_root,
            label="generated Genesis runtime manifest",
            maximum_bytes=_MAXIMUM_RUNTIME_ARTIFACT_BYTES,
        )
        if _sha(manifest_path.read_bytes()) != task.genesis_runtime_manifest_sha256:
            raise ValueError("generated Genesis runtime manifest is unavailable or changed")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            not isinstance(manifest, dict)
            or manifest.get("package_hash") != task.reference_package_hash
        ):
            raise ValueError("generated runtime is bound to another public package")
        _validate_runtime_manifest(manifest_path, artifact_root=artifact_root)
        runtime_config_path = _contained_regular_file(
            manifest_path.parent / "runtime_config.json",
            artifact_root,
            label="generated runtime configuration",
            maximum_bytes=_MAXIMUM_RUNTIME_ARTIFACT_BYTES,
        )
        runtime_config_sha256 = _sha(runtime_config_path.read_bytes())
        all_samples: list[RawCpuSample] = []
        for summary in task.baselines:
            if summary.status not in {BaselineStatus.MEASURED, BaselineStatus.SURROGATE}:
                continue
            if summary.raw_samples_path is None or summary.raw_samples_sha256 is None:
                raise ValueError("measured baseline is missing raw evidence")
            path = _contained_regular_file(
                summary.raw_samples_path, artifact_root, label="SynthBench raw samples"
            )
            if _sha(path.read_bytes()) != summary.raw_samples_sha256:
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
                expected_request = workload_by_id.get(sample.request_id)
                if expected_request is None:
                    raise ValueError("raw sample request is absent from the bound workload")
                if (
                    sample.request_seed != expected_request.seed
                    or sample.expected_tokens != expected_request.expected_tokens
                    or sample.exact_match != (sample.observed_tokens == sample.expected_tokens)
                ):
                    raise ValueError("raw sample is not bound to the workload oracle")
                _validate_sandbox_execution_evidence(
                    sample.execution_evidence_path,
                    sample.execution_evidence_sha256,
                    expected_request=expected_request,
                    run_seed=sample.run_seed,
                    expected_execution_surface=(
                        "python_eager_reference"
                        if sample.execution_surface == "reference_order_surrogate"
                        else sample.execution_surface
                    ),
                    expected_subject_sha256=(
                        runtime_config_sha256
                        if sample.execution_surface == "genesis_generated_runtime"
                        else task.public_package_hash
                    ),
                    observed_tokens=sample.observed_tokens,
                    latency_ns=sample.latency_ns,
                    ttft_ns=sample.ttft_ns,
                    inter_token_ns=sample.inter_token_ns,
                    artifact_root=artifact_root,
                )
            all_samples.extend(loaded_samples)
        for hidden in task.hidden_evaluations:
            if hidden.status not in {BaselineStatus.MEASURED, BaselineStatus.SURROGATE}:
                continue
            if hidden.evidence_path is None or hidden.evidence_sha256 is None:
                raise ValueError("measured hidden evaluation is missing evidence")
            path = _contained_regular_file(
                hidden.evidence_path, artifact_root, label="SynthBench hidden evidence"
            )
            if _sha(path.read_bytes()) != hidden.evidence_sha256:
                raise ValueError("SynthBench hidden evidence is unavailable or changed")
            if _hidden_summary_from_artifact(hidden.baseline, path) != hidden:
                raise ValueError("hidden summary is not derived from its evidence")
            hidden_results = tuple(
                HiddenCaseResult.model_validate_json(line, strict=True)
                for line in path.read_bytes().splitlines()
                if line
            )
            hidden_requests = tuple(case.request for case in hidden_cases)
            expected_hidden_order, _ = _selected_order(hidden.baseline, hidden_requests)
            case_by_request = {case.request.request_id: case.case_id for case in hidden_cases}
            expected_case_ids = tuple(
                case_by_request[request.request_id] for request in expected_hidden_order
            )
            if tuple(result.case_id for result in hidden_results) != expected_case_ids:
                raise ValueError("hidden evidence is incomplete, duplicated, or reordered")
            for result in hidden_results:
                expected_case = hidden_by_id.get(result.case_id)
                if expected_case is None or expected_case.request.request_id != result.request_id:
                    raise ValueError("hidden result is absent from the bound hidden corpus")
                expected_request = expected_case.request
                if (
                    result.baseline is not hidden.baseline
                    or result.execution_surface
                    != (
                        "python_eager_reference"
                        if hidden.baseline is BaselineKind.PYTHON_EAGER_REFERENCE
                        else "genesis_generated_runtime"
                        if hidden.baseline is BaselineKind.GENESIS_FULL
                        else "reference_order_surrogate"
                    )
                    or result.request_seed != expected_request.seed
                    or result.expected_tokens != expected_request.expected_tokens
                    or result.exact_match != (result.observed_tokens == result.expected_tokens)
                ):
                    raise ValueError("hidden result is not bound to its evaluator oracle")
                _validate_sandbox_execution_evidence(
                    result.execution_evidence_path,
                    result.execution_evidence_sha256,
                    expected_request=expected_request,
                    run_seed=task.run_seed,
                    expected_execution_surface=(
                        "python_eager_reference"
                        if result.execution_surface == "reference_order_surrogate"
                        else result.execution_surface
                    ),
                    expected_subject_sha256=(
                        runtime_config_sha256
                        if result.execution_surface == "genesis_generated_runtime"
                        else task.public_package_hash
                    ),
                    observed_tokens=result.observed_tokens,
                    latency_ns=None,
                    ttft_ns=None,
                    inter_token_ns=None,
                    artifact_root=artifact_root,
                )
        decoded_order = _decode_measurement_order(task.measurement_run_order)
        expected_order = _measurement_order(
            run_seed=task.run_seed,
            task_id=task.task_id,
            repetitions=task.repetitions,
        )
        if decoded_order != expected_order:
            raise ValueError("measurement run order does not replay from its declared seed")
        for ordinal, (_repetition, baseline) in enumerate(expected_order):
            expected_requests, _ = _selected_order(baseline, workload)
            block = sorted(
                (sample for sample in all_samples if sample.measurement_order_ordinal == ordinal),
                key=lambda sample: sample.execution_ordinal,
            )
            if tuple(sample.request_id for sample in block) != tuple(
                request.request_id for request in expected_requests
            ):
                raise ValueError("measurement request order differs from the declared policy")
        rebuilt_integrity = audit_raw_samples(
            tuple(all_samples),
            measured_baselines=_MEASURED_BASELINES,
            workload_fingerprint=task.integrity.workload_fingerprint,
            environment_fingerprint=task.integrity.environment_fingerprint,
            task_hash=task.task_hash,
            run_seed=task.run_seed,
            repetitions=task.repetitions,
            request_ids=task.integrity.expected_request_ids,
            measurement_run_order=decoded_order,
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
                sandbox_root=(
                    output_directory
                    / "sandbox"
                    / descriptor.task_id
                    / f"seed-{run_seed}"
                    / "public"
                    / baseline.value
                ),
                run_seed=run_seed,
                execution_surface=(
                    "python_eager_reference"
                    if baseline is BaselineKind.PYTHON_EAGER_REFERENCE
                    else "reference_order_surrogate"
                ),
            )
        )
        for baseline in _EXECUTED_BASELINES
    }
    deadline_ns = time.perf_counter_ns() + int(configuration.maximum_runtime_seconds * 1e9)
    for baseline in _EXECUTED_BASELINES:
        executor = executors[baseline]
        selected, _ = _selected_order(baseline, workload)
        for _ in range(configuration.warmup_count):
            for request in selected:
                executor.execute(request, deadline_ns=deadline_ns)
    run_order = _measurement_order(
        run_seed=run_seed,
        task_id=descriptor.task_id,
        repetitions=configuration.repetitions,
    )
    samples_by_baseline: dict[BaselineKind, list[RawCpuSample]] = {
        baseline: [] for baseline in _MEASURED_BASELINES
    }
    for measurement_order_ordinal, (repetition, baseline) in enumerate(run_order):
        if baseline not in _EXECUTED_BASELINES:
            continue
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
    eager_by_request = {
        (sample.repetition, sample.request_id): sample
        for sample in samples_by_baseline[BaselineKind.PYTHON_EAGER_REFERENCE]
    }
    for measurement_order_ordinal, (repetition, baseline) in enumerate(run_order):
        if baseline in _EXECUTED_BASELINES:
            continue
        selected, _ = _selected_order(baseline, workload)
        baseline_samples = samples_by_baseline[baseline]
        for request in selected:
            reference = eager_by_request[(repetition, request.request_id)]
            baseline_samples.append(
                RawCpuSample(
                    task_id=reference.task_id,
                    task_hash=reference.task_hash,
                    workload_fingerprint=reference.workload_fingerprint,
                    environment_fingerprint=reference.environment_fingerprint,
                    baseline=baseline,
                    run_seed=reference.run_seed,
                    repetition=repetition,
                    measurement_order_ordinal=measurement_order_ordinal,
                    execution_ordinal=len(baseline_samples),
                    request_id=reference.request_id,
                    request_seed=reference.request_seed,
                    latency_ns=reference.latency_ns,
                    ttft_ns=reference.ttft_ns,
                    inter_token_ns=reference.inter_token_ns,
                    observed_tokens=reference.observed_tokens,
                    expected_tokens=reference.expected_tokens,
                    exact_match=reference.exact_match,
                    execution_surface="reference_order_surrogate",
                    execution_evidence_path=reference.execution_evidence_path,
                    execution_evidence_sha256=reference.execution_evidence_sha256,
                    source="replayed_cpu_reference_observation",
                )
            )
    summaries: list[BaselineSummary] = []
    hidden_summaries: list[HiddenEvaluationSummary] = []
    all_samples: list[RawCpuSample] = []
    hidden_reference_by_request: dict[str, HiddenCaseResult] = {}
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
        if baseline in _EXECUTED_BASELINES:
            hidden_executor: _Executor | _GenesisExecutor = (
                _GenesisExecutor(
                    initialized.runtime.output_directory / "runtime_config.json",
                    output_directory
                    / "sandbox"
                    / descriptor.task_id
                    / f"seed-{run_seed}"
                    / "hidden",
                    seed=run_seed,
                )
                if baseline is BaselineKind.GENESIS_FULL
                else _Executor(
                    task_directory,
                    descriptor,
                    sandbox_root=(
                        output_directory
                        / "sandbox"
                        / descriptor.task_id
                        / f"seed-{run_seed}"
                        / "hidden"
                        / baseline.value
                    ),
                    run_seed=run_seed,
                    execution_surface="python_eager_reference",
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
                        ),
                        execution_evidence_path=(
                            execution_evidence[0] if execution_evidence is not None else None
                        ),
                        execution_evidence_sha256=(
                            execution_evidence[1] if execution_evidence is not None else None
                        ),
                    )
                )
            if baseline is BaselineKind.PYTHON_EAGER_REFERENCE:
                hidden_reference_by_request = {item.request_id: item for item in hidden_results}
        else:
            for request in selected_hidden:
                hidden_reference = hidden_reference_by_request[request.request_id]
                hidden_results.append(
                    HiddenCaseResult(
                        case_id=case_by_request[request.request_id].case_id,
                        baseline=baseline,
                        request_id=request.request_id,
                        request_seed=hidden_reference.request_seed,
                        observed_tokens=hidden_reference.observed_tokens,
                        expected_tokens=hidden_reference.expected_tokens,
                        exact_match=hidden_reference.exact_match,
                        execution_surface="reference_order_surrogate",
                        execution_evidence_path=hidden_reference.execution_evidence_path,
                        execution_evidence_sha256=hidden_reference.execution_evidence_sha256,
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
        task_descriptor_path=str((task_directory / "task.json").resolve()),
        task_descriptor_sha256=_sha((task_directory / "task.json").read_bytes()),
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
    validate_cpu_benchmark_report(persisted, artifact_root=output_directory)
    return persisted
