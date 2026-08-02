"""CPU-only ServingSynthBench runner with raw measured evidence and local baselines."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import platform
import random
import statistics
import time
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path
from types import ModuleType
from typing import cast

from sloforge.genesis.ir import canonical_json

from .grammar import load_hidden_cases, load_task, load_workload
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
    def __init__(self, task_directory: Path, descriptor: TaskDescriptor) -> None:
        self.descriptor = descriptor
        module = _load_module(
            task_directory / descriptor.public_package_path / "reference.py",
            f"synthbench_{descriptor.task_id}",
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
    ) -> tuple[tuple[int, ...], int, int, tuple[int, ...]]:
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
        return tuple(tokens), max(1, end - start), ttft, intervals


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
    return BaselineSummary(
        baseline=baseline,
        status=BaselineStatus.MEASURED,
        reason="summary derived from complete raw measured CPU samples",
        raw_samples_path=str(path.resolve()),
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
        case_count=len(results),
        exact_case_rate=exact / len(results) if results else 0.0,
        escaped_regressions=len(results) - exact,
    )


def _unavailable_hidden(baseline: BaselineKind) -> HiddenEvaluationSummary:
    return HiddenEvaluationSummary(
        baseline=baseline,
        status=BaselineStatus.NOT_APPLICABLE,
        evidence_path=None,
        case_count=0,
        exact_case_rate=0.0,
        escaped_regressions=0,
    )


def _run_task(
    task_directory: Path,
    descriptor: TaskDescriptor,
    *,
    run_seed: int,
    configuration: CpuRunConfiguration,
    output_directory: Path,
) -> TaskRunReport:
    workload_path = task_directory / descriptor.workload_path
    workload = load_workload(task_directory, descriptor)
    hidden_cases = load_hidden_cases(task_directory, descriptor)
    workload_fingerprint = _sha(workload_path.read_bytes())
    environment = _environment_fingerprint()
    task_hash = _sha(canonical_json(descriptor))
    executor = _Executor(task_directory, descriptor)
    deadline_ns = time.perf_counter_ns() + int(configuration.maximum_runtime_seconds * 1e9)
    for baseline in _MEASURED_BASELINES:
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
    for repetition, baseline in run_order:
        selected, _ = _selected_order(baseline, workload)
        for request in selected:
            observed, latency, ttft, intervals = executor.execute(request, deadline_ns=deadline_ns)
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
                    execution_ordinal=len(baseline_samples),
                    request_id=request.request_id,
                    latency_ns=latency,
                    ttft_ns=ttft,
                    inter_token_ns=intervals,
                    observed_tokens=observed,
                    expected_tokens=request.expected_tokens,
                    exact_match=observed == request.expected_tokens,
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
        summaries.append(_summary_from_artifact(baseline, path, candidate_count=candidate_count))
        all_samples.extend(persisted_samples)
        hidden_requests = tuple(case.request for case in hidden_cases)
        selected_hidden, _ = _selected_order(baseline, hidden_requests)
        case_by_request = {case.request.request_id: case for case in hidden_cases}
        hidden_results: list[HiddenCaseResult] = []
        for request in selected_hidden:
            observed, _, _, _ = executor.execute(request, deadline_ns=deadline_ns)
            hidden_results.append(
                HiddenCaseResult(
                    case_id=case_by_request[request.request_id].case_id,
                    baseline=baseline,
                    observed_tokens=observed,
                    expected_tokens=request.expected_tokens,
                    exact_match=observed == request.expected_tokens,
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
        repetitions=configuration.repetitions,
        request_ids=tuple(request.request_id for request in workload),
    )
    return TaskRunReport(
        task_id=descriptor.task_id,
        task_hash=task_hash,
        run_seed=run_seed,
        warmup_count=configuration.warmup_count,
        repetitions=configuration.repetitions,
        measurement_run_order=tuple(
            f"{repetition}:{baseline.value}" for repetition, baseline in run_order
        ),
        baselines=tuple(summaries),
        hidden_evaluations=tuple(hidden_summaries),
        integrity=integrity,
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
    sample_count = sum(baseline.sample_count for baseline in measured)
    exact_count = sum(
        round(baseline.valid_request_rate * baseline.sample_count) for baseline in measured
    )
    metrics = AggregateMetrics(
        measured_baseline_runs=len(measured),
        unavailable_baseline_runs=len(unavailable),
        valid_system_rate=(
            sum(
                baseline.valid_request_rate == 1.0
                and next(
                    hidden.exact_case_rate
                    for hidden in report.hidden_evaluations
                    if hidden.baseline is baseline.baseline
                )
                == 1.0
                for report in reports
                for baseline in report.baselines
                if baseline.status is BaselineStatus.MEASURED
            )
            / len(measured)
            if measured
            else 0.0
        ),
        exact_request_rate=exact_count / sample_count if sample_count else 0.0,
        measured_cpu_seconds=sum(baseline.measured_cpu_seconds for baseline in measured),
        task_count=len(selected),
        distinct_task_seeds=len({load_task(path / "task.json").seed for path in selected}),
    )
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
    return SynthBenchReport.model_validate_json(report_path.read_bytes(), strict=True)
