"""Deterministic CPU-only H1 campaign for randomized unseen-model synthesis.

The campaign deliberately treats a generated ServingSynthBench task as the
statistical unit.  Synthesis seeds are repeated observations within a task;
they are not promoted to independent samples when constructing the aggregate
Wilson interval.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
import tempfile
import time
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from sloforge.genesis.compiler import initialize_genesis_run
from sloforge.genesis.frontend import InspectionResult, inspect_reference_package
from sloforge.genesis.frontend.models import DiagnosticSeverity
from sloforge.genesis.ir import canonical_json
from sloforge.genesis.sandbox import SandboxLimits, SandboxRequest, execute_sandboxed
from sloforge.synthbench import (
    GrammarConfiguration,
    HiddenCase,
    SpecialCaseAudit,
    TaskDescriptor,
    WorkloadRequest,
    audit_special_casing,
    generate_tasks,
    load_hidden_cases,
    load_task,
    load_workload,
    verify_public_package,
)

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]
PositiveNs = Annotated[int, Field(gt=0)]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class _CampaignModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
        allow_inf_nan=False,
    )


class H1AttemptStatus(StrEnum):
    INSPECTION_FAILED = "inspection_failed"
    UNSUPPORTED = "unsupported"
    COMPILE_FAILED = "compile_failed"
    HARD_CODING_REJECTED = "hard_coding_rejected"
    CORRECTNESS_FAILED = "correctness_failed"
    CORRECT = "correct"


class H1CampaignConfiguration(_CampaignModel):
    grammar: GrammarConfiguration
    synthesis_seeds: tuple[NonNegativeInt, ...]
    request_timeout_seconds: Annotated[float, Field(gt=0.0, le=30.0)] = 10.0
    maximum_attempt_seconds: Annotated[float, Field(gt=0.0, le=300.0)] = 120.0

    @model_validator(mode="after")
    def multiple_distinct_synthesis_seeds(self) -> Self:
        if len(self.synthesis_seeds) < 2:
            raise ValueError("H1 campaign requires at least two synthesis seeds")
        if len(self.synthesis_seeds) != len(set(self.synthesis_seeds)):
            raise ValueError("H1 synthesis seeds must be distinct")
        return self


class ArtifactReference(_CampaignModel):
    path: NonEmpty
    sha256: Sha256


class WilsonInterval(_CampaignModel):
    successes: NonNegativeInt
    trials: PositiveInt
    confidence: Probability = 0.95
    lower: Probability
    upper: Probability
    sample_unit: Literal["unseen_task", "synthesis_seed"]

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if self.successes > self.trials:
            raise ValueError("Wilson successes cannot exceed trials")
        if self.confidence != 0.95:
            raise ValueError("H1 campaign reports pre-registered 95% Wilson intervals")
        if self.lower > self.upper:
            raise ValueError("Wilson interval is reversed")
        estimate = self.successes / self.trials
        if not self.lower <= estimate <= self.upper:
            raise ValueError("Wilson interval does not contain its observed proportion")
        return self


class H1CaseEvidence(_CampaignModel):
    phase: Literal["public", "hidden"]
    case_id: NonEmpty
    request_id: NonEmpty
    request_seed: NonNegativeInt
    expected_tokens: tuple[NonNegativeInt, ...]
    observed_tokens: tuple[NonNegativeInt, ...]
    exact_match: bool
    execution_evidence: ArtifactReference

    @model_validator(mode="after")
    def exact_flag_is_derived(self) -> Self:
        if self.exact_match != (self.expected_tokens == self.observed_tokens):
            raise ValueError("case exact-match flag is not derived from tokens")
        return self


class H1AttemptEvidence(_CampaignModel):
    schema_version: Literal["sloforge.genesis.h1-attempt/v1"] = "sloforge.genesis.h1-attempt/v1"
    task_id: NonEmpty
    task_generation_seed: NonNegativeInt
    synthesis_seed: NonNegativeInt
    attempt_ordinal: NonNegativeInt
    task_descriptor: ArtifactReference
    public_package_hash: Sha256
    reference_package_hash: Sha256 | None
    inspection: ArtifactReference | None
    unsupported_diagnostic_ids: tuple[NonEmpty, ...]
    runtime_manifest: ArtifactReference | None
    runtime_config: ArtifactReference | None
    special_case_audit: SpecialCaseAudit | None
    public_cases: tuple[H1CaseEvidence, ...]
    hidden_cases: tuple[H1CaseEvidence, ...]
    compile_latency_ns: PositiveNs | None
    correct_latency_ns: PositiveNs | None
    total_latency_ns: PositiveNs
    status: H1AttemptStatus
    failure: NonEmpty | None
    submitted_model_specific_adapter_path: None = None
    human_authored_model_specific_lines: Literal[0] = 0
    execution_scope: Literal["cpu_only_sandboxed"] = "cpu_only_sandboxed"
    hardware_evidence: Literal[False] = False

    @model_validator(mode="after")
    def status_matches_evidence(self) -> Self:
        compiled = self.runtime_manifest is not None and self.runtime_config is not None
        if (self.inspection is None) != (self.reference_package_hash is None):
            raise ValueError("reference package identity must match inspection evidence")
        if compiled and self.inspection is None:
            raise ValueError("generated runtime must follow a persisted zero-day inspection")
        if (self.runtime_manifest is None) != (self.runtime_config is None):
            raise ValueError("runtime manifest and configuration must appear together")
        if compiled != (self.compile_latency_ns is not None):
            raise ValueError("compile latency must match generated-runtime evidence")
        if self.status is H1AttemptStatus.CORRECT:
            if not compiled or self.correct_latency_ns is None or self.failure is not None:
                raise ValueError("correct attempt is missing successful runtime evidence")
            if self.special_case_audit is None or not self.special_case_audit.passed:
                raise ValueError("correct attempt did not pass the hard-coding audit")
            if not self.public_cases or not self.hidden_cases:
                raise ValueError("correct attempt must exercise public and hidden cases")
            if not all(case.exact_match for case in (*self.public_cases, *self.hidden_cases)):
                raise ValueError("correct attempt contains a failing case")
        elif self.failure is None:
            raise ValueError("non-correct attempt must record its failure")
        if self.status is H1AttemptStatus.INSPECTION_FAILED and self.inspection is not None:
            raise ValueError("inspection-failed attempt unexpectedly contains inspection evidence")
        if self.status is H1AttemptStatus.UNSUPPORTED and (self.inspection is None or compiled):
            raise ValueError("unsupported attempt must stop after zero-day inspection")
        if self.status is H1AttemptStatus.COMPILE_FAILED and (self.inspection is None or compiled):
            raise ValueError("compile-failed attempt must not claim a generated runtime")
        if self.status is H1AttemptStatus.HARD_CODING_REJECTED and (
            not compiled or self.special_case_audit is None or self.special_case_audit.passed
        ):
            raise ValueError("hard-coding rejection must carry a failing audit")
        if self.status is H1AttemptStatus.CORRECTNESS_FAILED and (
            not compiled or self.special_case_audit is None or not self.special_case_audit.passed
        ):
            raise ValueError("correctness failure must follow compilation and source audit")
        if (
            self.correct_latency_ns is not None
            and self.compile_latency_ns is not None
            and self.correct_latency_ns < self.compile_latency_ns
        ):
            raise ValueError("correctness cannot precede compilation")
        if self.compile_latency_ns is not None and self.total_latency_ns < self.compile_latency_ns:
            raise ValueError("total attempt time cannot precede compilation")
        if self.correct_latency_ns is not None and self.total_latency_ns < self.correct_latency_ns:
            raise ValueError("total attempt time cannot precede correctness")
        if self.status is H1AttemptStatus.UNSUPPORTED and not self.unsupported_diagnostic_ids:
            raise ValueError("unsupported attempt must identify unsupported diagnostics")
        return self

    @property
    def compiled(self) -> bool:
        return self.runtime_manifest is not None

    @property
    def correct(self) -> bool:
        return self.status is H1AttemptStatus.CORRECT

    @property
    def unsupported(self) -> bool:
        return self.status is H1AttemptStatus.UNSUPPORTED


class H1TaskReport(_CampaignModel):
    task_id: NonEmpty
    task_generation_seed: NonNegativeInt
    task_descriptor: ArtifactReference
    attempts: tuple[ArtifactReference, ...]
    synthesis_seeds: tuple[NonNegativeInt, ...]
    compiling_runtime_found: bool
    correct_runtime_found: bool
    valid_system: bool
    unsupported: bool
    time_to_first_compiling_runtime_ns: PositiveNs | None
    time_to_first_correct_runtime_ns: PositiveNs | None
    public_case_count: NonNegativeInt
    hidden_case_count: NonNegativeInt
    exact_public_case_rate: Probability
    exact_hidden_case_rate: Probability
    human_authored_model_specific_lines: Literal[0] = 0
    compiling_seed_interval: WilsonInterval
    correct_seed_interval: WilsonInterval
    unsupported_seed_interval: WilsonInterval


class H1AggregateMetrics(_CampaignModel):
    task_count: PositiveInt
    attempt_count: PositiveInt
    compiling_task_count: NonNegativeInt
    correct_task_count: NonNegativeInt
    valid_system_count: NonNegativeInt
    unsupported_task_count: NonNegativeInt
    compiling_task_rate: Probability
    correct_task_rate: Probability
    valid_system_rate: Probability
    unsupported_rate: Probability
    compiling_task_interval: WilsonInterval
    correct_task_interval: WilsonInterval
    valid_system_interval: WilsonInterval
    unsupported_task_interval: WilsonInterval
    median_time_to_first_compiling_runtime_ns: NonNegativeInt
    median_time_to_first_correct_runtime_ns: NonNegativeInt
    public_case_count: NonNegativeInt
    hidden_case_count: NonNegativeInt
    exact_public_case_rate: Probability
    exact_hidden_case_rate: Probability
    human_authored_model_specific_lines: Literal[0] = 0
    measured_campaign_wall_ns: PositiveNs


class H1CampaignReport(_CampaignModel):
    schema_version: Literal["sloforge.genesis.h1-campaign/v1"] = "sloforge.genesis.h1-campaign/v1"
    configuration: H1CampaignConfiguration
    tasks: tuple[H1TaskReport, ...]
    aggregate: H1AggregateMetrics
    task_success_definition: Literal[
        "every_declared_synthesis_seed_compiles_and_passes_public_and_hidden_cases"
    ] = "every_declared_synthesis_seed_compiles_and_passes_public_and_hidden_cases"
    confidence_interval_unit: Literal["unseen_task"] = "unseen_task"
    selection_is_seed_deterministic: Literal[True] = True
    timings_source: Literal["host_monotonic_clock"] = "host_monotonic_clock"
    execution_scope: Literal["cpu_only"] = "cpu_only"
    hardware_validation_status: Literal["not_exercised"] = "not_exercised"
    hardware_validation_reason: Literal[
        "campaign intentionally performs no GPU or distributed hardware execution"
    ] = "campaign intentionally performs no GPU or distributed hardware execution"
    report_source: Literal["independently_reconstructed_from_raw_attempts"] = (
        "independently_reconstructed_from_raw_attempts"
    )

    @model_validator(mode="after")
    def complete_task_set(self) -> Self:
        if len(self.tasks) != self.configuration.grammar.count:
            raise ValueError("H1 report task count differs from its grammar configuration")
        identifiers = [task.task_id for task in self.tasks]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("H1 report repeats a generated task")
        return self


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _artifact(path: Path) -> ArtifactReference:
    canonical = path.resolve(strict=True)
    return ArtifactReference(path=str(canonical), sha256=_sha(canonical.read_bytes()))


def _write_new(path: Path, value: BaseModel | dict[str, object]) -> ArtifactReference:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite H1 campaign artifact: {path}")
    path.write_bytes(canonical_json(value) + b"\n")
    return _artifact(path)


def _wilson(successes: int, trials: int, *, sample_unit: str) -> WilsonInterval:
    if trials <= 0:
        raise ValueError("Wilson interval requires a positive task count")
    z = 1.959963984540054
    proportion = successes / trials
    z_squared = z * z
    denominator = 1.0 + z_squared / trials
    center = (proportion + z_squared / (2.0 * trials)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / trials + z_squared / (4.0 * trials * trials))
        / denominator
    )
    return WilsonInterval(
        successes=successes,
        trials=trials,
        lower=0.0 if successes == 0 else max(0.0, center - margin),
        upper=1.0 if successes == trials else min(1.0, center + margin),
        sample_unit=sample_unit,  # type: ignore[arg-type]
    )


def _failure(error: BaseException) -> str:
    message = str(error).strip() or "no diagnostic message"
    return f"{type(error).__name__}: {message}"


def _validate_runtime_manifest(path: Path) -> None:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or set(document) != {
        "artifacts",
        "package_hash",
        "runtime_id",
        "schema_version",
    }:
        raise ValueError("generated runtime manifest has an invalid schema")
    artifacts = document["artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("generated runtime manifest has no artifacts")
    for item in artifacts:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not isinstance(item[1], str)
        ):
            raise ValueError("generated runtime manifest contains an invalid artifact identity")
        artifact = path.parent / item[0]
        if (
            not artifact.is_file()
            or artifact.is_symlink()
            or _sha(artifact.read_bytes()) != item[1]
        ):
            raise ValueError("generated runtime artifact is unavailable or changed")


def _execute_case(
    *,
    request: WorkloadRequest,
    phase: Literal["public", "hidden"],
    case_id: str,
    runtime_config: Path,
    runtime_manifest: Path,
    package_root: Path,
    output: Path,
    synthesis_seed: int,
    timeout_seconds: float,
) -> H1CaseEvidence:
    request_path = output.parent / f"{output.name}-request.json"
    _write_new(
        request_path,
        {
            "maximum_new_tokens": request.maximum_new_tokens,
            "prompt_tokens": list(request.prompt_tokens),
            "request_id": request.request_id,
            "seed": request.seed,
        },
    )
    runner_path = Path(__file__).resolve().parents[2] / "synthbench" / "runtime_runner.py"
    repository_python = Path(__file__).resolve().parents[3]
    remaining = timeout_seconds
    result = execute_sandboxed(
        SandboxRequest(
            argv=(
                sys.executable,
                "-m",
                "sloforge.synthbench.runtime_runner",
                "--config",
                str(runtime_config.resolve(strict=True)),
                "--request",
                str(request_path.resolve(strict=True)),
                "--seed",
                str(synthesis_seed),
                "--timeout-seconds",
                str(remaining),
            ),
            working_directory=runtime_config.parent.resolve(strict=True),
            read_only_paths=(
                runtime_config.parent.resolve(strict=True),
                package_root.resolve(strict=True),
                request_path.resolve(strict=True),
                repository_python.resolve(strict=True),
                Path(sys.prefix).resolve(strict=True),
                Path(sys.base_prefix).resolve(strict=True),
            ),
            artifact_output_directory=output,
            seed=synthesis_seed,
            limits=SandboxLimits(
                wall_time_seconds=remaining,
                cpu_time_seconds=max(1, min(30, math.ceil(remaining))),
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
            f"sandboxed generated runtime failed as {result.termination.value}: "
            f"{result.stderr.strip()}"
        )
    stdout_path = output / "runner-stdout.json"
    stderr_path = output / "runner-stderr.txt"
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    response = json.loads(result.stdout)
    if not isinstance(response, dict) or set(response) != {
        "inter_token_ns",
        "latency_ns",
        "tokens",
        "ttft_ns",
    }:
        raise ValueError("generated runtime runner returned an invalid response")
    observed = tuple(int(token) for token in response["tokens"])
    record = {
        "capabilities": result.capabilities.model_dump(mode="json"),
        "case_id": case_id,
        "config_sha256": _sha(runtime_config.read_bytes()),
        "manifest_sha256": _sha(runtime_manifest.read_bytes()),
        "phase": phase,
        "process_group_cleaned": result.process_group_cleaned,
        "request_id": request.request_id,
        "request_path": str(request_path.resolve(strict=True)),
        "request_seed": request.seed,
        "request_sha256": _sha(request_path.read_bytes()),
        "return_code": result.return_code,
        "run_seed": synthesis_seed,
        "runner_sha256": _sha(runner_path.read_bytes()),
        "sanitized_environment_names": list(result.sanitized_environment_names),
        "sandbox_backend": result.capabilities.backend.value,
        "schema_version": "sloforge.genesis.h1-execution/v1",
        "stderr_path": str(stderr_path.resolve(strict=True)),
        "stderr_sha256": _sha(stderr_path.read_bytes()),
        "stdout_path": str(stdout_path.resolve(strict=True)),
        "stdout_sha256": _sha(stdout_path.read_bytes()),
        "termination": result.termination.value,
    }
    evidence = _write_new(output / "execution.json", record)
    return H1CaseEvidence(
        phase=phase,
        case_id=case_id,
        request_id=request.request_id,
        request_seed=request.seed,
        expected_tokens=request.expected_tokens,
        observed_tokens=observed,
        exact_match=observed == request.expected_tokens,
        execution_evidence=evidence,
    )


def _attempt(
    *,
    task_directory: Path,
    descriptor: TaskDescriptor,
    synthesis_seed: int,
    attempt_ordinal: int,
    output: Path,
    configuration: H1CampaignConfiguration,
) -> H1AttemptEvidence:
    started = time.perf_counter_ns()
    descriptor_reference = _artifact(task_directory / "task.json")
    inspection_reference: ArtifactReference | None = None
    runtime_manifest_reference: ArtifactReference | None = None
    runtime_config_reference: ArtifactReference | None = None
    audit: SpecialCaseAudit | None = None
    public_results: list[H1CaseEvidence] = []
    hidden_results: list[H1CaseEvidence] = []
    unsupported_ids: tuple[str, ...] = ()
    compile_latency: int | None = None
    correct_latency: int | None = None
    status = H1AttemptStatus.INSPECTION_FAILED
    failure: str | None = None
    inspection: InspectionResult | None = None
    package = task_directory / descriptor.public_package_path
    inspection_path = output / "inspection.json"
    try:
        inspection = inspect_reference_package(package)
        inspection_reference = _write_new(inspection_path, inspection)
    except Exception as error:  # evaluation failures are retained as raw evidence
        failure = _failure(error)
    if inspection is not None:
        unsupported_ids = tuple(
            sorted(
                diagnostic.diagnostic_id
                for diagnostic in inspection.diagnostics
                if diagnostic.severity is DiagnosticSeverity.UNSUPPORTED
            )
        )
        if unsupported_ids:
            status = H1AttemptStatus.UNSUPPORTED
            failure = "zero-day inspection reported unsupported behavior"
        else:
            try:
                workload_hash = _sha((task_directory / descriptor.workload_path).read_bytes())
                initialized = initialize_genesis_run(
                    package,
                    inspection,
                    output / "initialized",
                    seed=synthesis_seed,
                    workload_contract_hash=workload_hash,
                )
                runtime_manifest = initialized.runtime.output_directory / "artifact_manifest.json"
                runtime_config = initialized.runtime.output_directory / "runtime_config.json"
                _validate_runtime_manifest(runtime_manifest)
                runtime_manifest_reference = _artifact(runtime_manifest)
                runtime_config_reference = _artifact(runtime_config)
                compile_latency = max(1, time.perf_counter_ns() - started)
            except Exception as error:  # retained compiler rejection
                status = H1AttemptStatus.COMPILE_FAILED
                failure = _failure(error)
            else:
                hidden = load_hidden_cases(task_directory, descriptor)
                audit = audit_special_casing(
                    runtime_manifest.parent / "runtime.py", descriptor, hidden
                )
                if not audit.passed:
                    status = H1AttemptStatus.HARD_CODING_REJECTED
                    failure = "generated runtime failed the task-specific hard-coding audit"
                else:
                    deadline = started + int(configuration.maximum_attempt_seconds * 1e9)
                    try:
                        for index, request in enumerate(load_workload(task_directory, descriptor)):
                            remaining = (deadline - time.perf_counter_ns()) / 1e9
                            if remaining <= 0.0:
                                raise TimeoutError("H1 attempt exceeded its wall-clock bound")
                            public_results.append(
                                _execute_case(
                                    request=request,
                                    phase="public",
                                    case_id=f"public-{index:03d}",
                                    runtime_config=runtime_config,
                                    runtime_manifest=runtime_manifest,
                                    package_root=package,
                                    output=output / "sandbox" / "public" / f"case-{index:03d}",
                                    synthesis_seed=synthesis_seed,
                                    timeout_seconds=min(
                                        configuration.request_timeout_seconds, remaining
                                    ),
                                )
                            )
                        for index, hidden_case in enumerate(hidden):
                            remaining = (deadline - time.perf_counter_ns()) / 1e9
                            if remaining <= 0.0:
                                raise TimeoutError("H1 attempt exceeded its wall-clock bound")
                            hidden_results.append(
                                _execute_case(
                                    request=hidden_case.request,
                                    phase="hidden",
                                    case_id=hidden_case.case_id,
                                    runtime_config=runtime_config,
                                    runtime_manifest=runtime_manifest,
                                    package_root=package,
                                    output=output / "sandbox" / "hidden" / f"case-{index:03d}",
                                    synthesis_seed=synthesis_seed,
                                    timeout_seconds=min(
                                        configuration.request_timeout_seconds, remaining
                                    ),
                                )
                            )
                    except Exception as error:  # retain partial correctness evidence
                        status = H1AttemptStatus.CORRECTNESS_FAILED
                        failure = _failure(error)
                    else:
                        exact = all(case.exact_match for case in (*public_results, *hidden_results))
                        if exact and public_results and hidden_results:
                            status = H1AttemptStatus.CORRECT
                            failure = None
                            correct_latency = max(1, time.perf_counter_ns() - started)
                        else:
                            status = H1AttemptStatus.CORRECTNESS_FAILED
                            failure = "generated runtime disagreed with a committed task oracle"
    total = max(1, time.perf_counter_ns() - started)
    return H1AttemptEvidence(
        task_id=descriptor.task_id,
        task_generation_seed=descriptor.seed,
        synthesis_seed=synthesis_seed,
        attempt_ordinal=attempt_ordinal,
        task_descriptor=descriptor_reference,
        public_package_hash=descriptor.public_package_hash,
        reference_package_hash=inspection.package_hash if inspection is not None else None,
        inspection=inspection_reference,
        unsupported_diagnostic_ids=unsupported_ids,
        runtime_manifest=runtime_manifest_reference,
        runtime_config=runtime_config_reference,
        special_case_audit=audit,
        public_cases=tuple(public_results),
        hidden_cases=tuple(hidden_results),
        compile_latency_ns=compile_latency,
        correct_latency_ns=correct_latency,
        total_latency_ns=total,
        status=status,
        failure=failure,
    )


def _task_report(
    descriptor: TaskDescriptor,
    descriptor_path: Path,
    attempts: tuple[tuple[ArtifactReference, H1AttemptEvidence], ...],
    synthesis_seeds: tuple[int, ...],
) -> H1TaskReport:
    evidence = tuple(item[1] for item in attempts)
    elapsed = 0
    first_compile: int | None = None
    first_correct: int | None = None
    for attempt in evidence:
        if first_compile is None and attempt.compile_latency_ns is not None:
            first_compile = elapsed + attempt.compile_latency_ns
        if first_correct is None and attempt.correct_latency_ns is not None:
            first_correct = elapsed + attempt.correct_latency_ns
        elapsed += attempt.total_latency_ns
    public = tuple(case for attempt in evidence for case in attempt.public_cases)
    hidden = tuple(case for attempt in evidence for case in attempt.hidden_cases)
    compiled_count = sum(attempt.compiled for attempt in evidence)
    correct_count = sum(attempt.correct for attempt in evidence)
    unsupported_count = sum(attempt.unsupported for attempt in evidence)
    trials = len(evidence)
    return H1TaskReport(
        task_id=descriptor.task_id,
        task_generation_seed=descriptor.seed,
        task_descriptor=_artifact(descriptor_path),
        attempts=tuple(item[0] for item in attempts),
        synthesis_seeds=synthesis_seeds,
        compiling_runtime_found=compiled_count > 0,
        correct_runtime_found=correct_count > 0,
        valid_system=correct_count == trials,
        unsupported=unsupported_count > 0,
        time_to_first_compiling_runtime_ns=first_compile,
        time_to_first_correct_runtime_ns=first_correct,
        public_case_count=len(public),
        hidden_case_count=len(hidden),
        exact_public_case_rate=(
            sum(case.exact_match for case in public) / len(public) if public else 0
        ),
        exact_hidden_case_rate=(
            sum(case.exact_match for case in hidden) / len(hidden) if hidden else 0
        ),
        compiling_seed_interval=_wilson(compiled_count, trials, sample_unit="synthesis_seed"),
        correct_seed_interval=_wilson(correct_count, trials, sample_unit="synthesis_seed"),
        unsupported_seed_interval=_wilson(unsupported_count, trials, sample_unit="synthesis_seed"),
    )


def _median_optional(values: tuple[int | None, ...]) -> int:
    measured = [value for value in values if value is not None]
    return int(statistics.median(measured)) if measured else 0


def _aggregate(tasks: tuple[H1TaskReport, ...], measured_wall_ns: int) -> H1AggregateMetrics:
    count = len(tasks)
    if count <= 0:
        raise ValueError("H1 campaign cannot aggregate an empty task set")
    compiling = sum(task.compiling_runtime_found for task in tasks)
    correct = sum(task.correct_runtime_found for task in tasks)
    valid = sum(task.valid_system for task in tasks)
    unsupported = sum(task.unsupported for task in tasks)
    public_count = sum(task.public_case_count for task in tasks)
    hidden_count = sum(task.hidden_case_count for task in tasks)
    exact_public = sum(task.exact_public_case_rate * task.public_case_count for task in tasks)
    exact_hidden = sum(task.exact_hidden_case_rate * task.hidden_case_count for task in tasks)
    return H1AggregateMetrics(
        task_count=count,
        attempt_count=sum(len(task.attempts) for task in tasks),
        compiling_task_count=compiling,
        correct_task_count=correct,
        valid_system_count=valid,
        unsupported_task_count=unsupported,
        compiling_task_rate=compiling / count,
        correct_task_rate=correct / count,
        valid_system_rate=valid / count,
        unsupported_rate=unsupported / count,
        compiling_task_interval=_wilson(compiling, count, sample_unit="unseen_task"),
        correct_task_interval=_wilson(correct, count, sample_unit="unseen_task"),
        valid_system_interval=_wilson(valid, count, sample_unit="unseen_task"),
        unsupported_task_interval=_wilson(unsupported, count, sample_unit="unseen_task"),
        median_time_to_first_compiling_runtime_ns=_median_optional(
            tuple(task.time_to_first_compiling_runtime_ns for task in tasks)
        ),
        median_time_to_first_correct_runtime_ns=_median_optional(
            tuple(task.time_to_first_correct_runtime_ns for task in tasks)
        ),
        public_case_count=public_count,
        hidden_case_count=hidden_count,
        exact_public_case_rate=exact_public / public_count if public_count else 0.0,
        exact_hidden_case_rate=exact_hidden / hidden_count if hidden_count else 0.0,
        measured_campaign_wall_ns=max(1, measured_wall_ns),
    )


def _load_attempt(reference: ArtifactReference) -> H1AttemptEvidence:
    path = Path(reference.path)
    if not path.is_file() or path.is_symlink() or _sha(path.read_bytes()) != reference.sha256:
        raise ValueError("H1 raw attempt artifact is unavailable or changed")
    return H1AttemptEvidence.model_validate_json(path.read_bytes(), strict=True)


def _validate_execution(
    case: H1CaseEvidence,
    *,
    request: WorkloadRequest,
    runtime_config: ArtifactReference,
    runtime_manifest: ArtifactReference,
    synthesis_seed: int,
) -> None:
    reference = case.execution_evidence
    path = Path(reference.path)
    if not path.is_file() or path.is_symlink() or _sha(path.read_bytes()) != reference.sha256:
        raise ValueError("H1 sandbox execution evidence is unavailable or changed")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema_version") != (
        "sloforge.genesis.h1-execution/v1"
    ):
        raise ValueError("H1 sandbox execution evidence has an invalid schema")
    expected_identity = {
        "case_id": case.case_id,
        "config_sha256": runtime_config.sha256,
        "manifest_sha256": runtime_manifest.sha256,
        "phase": case.phase,
        "request_id": request.request_id,
        "request_seed": request.seed,
        "run_seed": synthesis_seed,
        "termination": "success",
        "return_code": 0,
    }
    if any(document.get(key) != value for key, value in expected_identity.items()):
        raise ValueError("H1 sandbox evidence identity does not match its case")
    if document.get("sandbox_backend") in {None, "none"}:
        raise ValueError("H1 execution was not protected by a kernel sandbox")
    if document.get("process_group_cleaned") is not True:
        raise ValueError("H1 sandbox did not prove child-process cleanup")
    request_path_value = document.get("request_path")
    stdout_path_value = document.get("stdout_path")
    if not isinstance(request_path_value, str) or not isinstance(stdout_path_value, str):
        raise ValueError("H1 sandbox evidence omits retained inputs or output")
    request_path = Path(request_path_value)
    stdout_path = Path(stdout_path_value)
    if (
        not request_path.is_file()
        or _sha(request_path.read_bytes()) != document.get("request_sha256")
        or not stdout_path.is_file()
        or _sha(stdout_path.read_bytes()) != document.get("stdout_sha256")
    ):
        raise ValueError("H1 retained sandbox request or output changed")
    request_document = json.loads(request_path.read_text(encoding="utf-8"))
    if request_document != {
        "maximum_new_tokens": request.maximum_new_tokens,
        "prompt_tokens": list(request.prompt_tokens),
        "request_id": request.request_id,
        "seed": request.seed,
    }:
        raise ValueError("H1 sandbox request is not bound to its task input")
    response = json.loads(stdout_path.read_text(encoding="utf-8"))
    if not isinstance(response, dict) or set(response) != {
        "inter_token_ns",
        "latency_ns",
        "tokens",
        "ttft_ns",
    }:
        raise ValueError("H1 retained sandbox response has an invalid schema")
    if tuple(response["tokens"]) != case.observed_tokens:
        raise ValueError("H1 case tokens differ from the sandbox response")
    runner = Path(__file__).resolve().parents[2] / "synthbench" / "runtime_runner.py"
    if document.get("runner_sha256") != _sha(runner.read_bytes()):
        raise ValueError("H1 sandbox evidence used an unrecognized trusted runner")
    if case.request_seed != request.seed or case.expected_tokens != request.expected_tokens:
        raise ValueError("H1 case is not bound to its task oracle")


def _validate_attempt(
    attempt: H1AttemptEvidence,
    *,
    descriptor: TaskDescriptor,
    task_directory: Path,
    attempt_ordinal: int,
    synthesis_seed: int,
) -> None:
    if (
        attempt.task_id != descriptor.task_id
        or attempt.task_generation_seed != descriptor.seed
        or attempt.public_package_hash != descriptor.public_package_hash
        or attempt.attempt_ordinal != attempt_ordinal
        or attempt.synthesis_seed != synthesis_seed
    ):
        raise ValueError("H1 attempt identity differs from its task or configured seed")
    if attempt.task_descriptor != _artifact(task_directory / "task.json"):
        raise ValueError("H1 attempt does not bind the current task descriptor")
    package = task_directory / descriptor.public_package_path
    if attempt.inspection is not None:
        inspection_path = Path(attempt.inspection.path)
        if (
            not inspection_path.is_file()
            or inspection_path.is_symlink()
            or _sha(inspection_path.read_bytes()) != attempt.inspection.sha256
        ):
            raise ValueError("H1 inspection evidence is unavailable or changed")
        persisted = InspectionResult.model_validate_json(inspection_path.read_bytes(), strict=True)
        recomputed = inspect_reference_package(package)
        if persisted != recomputed:
            raise ValueError("H1 inspection is not independently reproducible")
        if attempt.reference_package_hash != recomputed.package_hash:
            raise ValueError("H1 reference package hash differs from inspection")
        unsupported_ids = tuple(
            sorted(
                diagnostic.diagnostic_id
                for diagnostic in recomputed.diagnostics
                if diagnostic.severity is DiagnosticSeverity.UNSUPPORTED
            )
        )
        if unsupported_ids != attempt.unsupported_diagnostic_ids:
            raise ValueError("H1 unsupported diagnostics differ from inspection")
    if attempt.runtime_manifest is None or attempt.runtime_config is None:
        if attempt.public_cases or attempt.hidden_cases:
            raise ValueError("H1 uncompiled attempt contains runtime execution evidence")
        return
    manifest_path = Path(attempt.runtime_manifest.path)
    config_path = Path(attempt.runtime_config.path)
    if (
        not manifest_path.is_file()
        or _sha(manifest_path.read_bytes()) != attempt.runtime_manifest.sha256
        or not config_path.is_file()
        or _sha(config_path.read_bytes()) != attempt.runtime_config.sha256
    ):
        raise ValueError("H1 generated-runtime identity is unavailable or changed")
    _validate_runtime_manifest(manifest_path)
    manifest_document = json.loads(manifest_path.read_text(encoding="utf-8"))
    config_document = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        not isinstance(config_document, dict)
        or not isinstance(manifest_document, dict)
        or config_document.get("generation_seed") != synthesis_seed
        or config_document.get("package_hash") != attempt.reference_package_hash
        or manifest_document.get("package_hash") != attempt.reference_package_hash
    ):
        raise ValueError("H1 runtime configuration differs from task or synthesis seed")
    hidden_cases = load_hidden_cases(task_directory, descriptor)
    expected_audit = audit_special_casing(
        manifest_path.parent / "runtime.py", descriptor, hidden_cases
    )
    if expected_audit != attempt.special_case_audit:
        raise ValueError("H1 hard-coding audit is not independently reproducible")
    public = load_workload(task_directory, descriptor)
    public_by_id = {request.request_id: request for request in public}
    hidden_by_case = {case.case_id: case for case in hidden_cases}
    if len(attempt.public_cases) > len(public) or len(attempt.hidden_cases) > len(hidden_cases):
        raise ValueError("H1 attempt contains excess correctness cases")
    for case in attempt.public_cases:
        request = public_by_id.get(case.request_id)
        if request is None or case.phase != "public":
            raise ValueError("H1 public result is absent from the public workload")
        _validate_execution(
            case,
            request=request,
            runtime_config=attempt.runtime_config,
            runtime_manifest=attempt.runtime_manifest,
            synthesis_seed=synthesis_seed,
        )
    for case in attempt.hidden_cases:
        hidden_case: HiddenCase | None = hidden_by_case.get(case.case_id)
        if (
            hidden_case is None
            or case.phase != "hidden"
            or hidden_case.request.request_id != case.request_id
        ):
            raise ValueError("H1 hidden result is absent from the committed hidden corpus")
        _validate_execution(
            case,
            request=hidden_case.request,
            runtime_config=attempt.runtime_config,
            runtime_manifest=attempt.runtime_manifest,
            synthesis_seed=synthesis_seed,
        )
    if attempt.correct and (
        len(attempt.public_cases) != len(public) or len(attempt.hidden_cases) != len(hidden_cases)
    ):
        raise ValueError("H1 correct attempt omitted public or hidden cases")


def validate_h1_unseen_campaign(report: H1CampaignReport) -> None:
    """Independently regenerate tasks and reconstruct all task/aggregate claims."""

    with tempfile.TemporaryDirectory(prefix="sloforge-h1-validator-") as temporary:
        regenerated_root = Path(temporary) / "tasks"
        regenerated = generate_tasks(report.configuration.grammar, regenerated_root)
        expected_descriptors = {descriptor.task_id: descriptor for descriptor in regenerated}
        if set(expected_descriptors) != {task.task_id for task in report.tasks}:
            raise ValueError("H1 task set is not generated by its declared grammar seed")
        rebuilt_tasks: list[H1TaskReport] = []
        for task in report.tasks:
            descriptor_path = Path(task.task_descriptor.path)
            if (
                not descriptor_path.is_file()
                or descriptor_path.is_symlink()
                or _sha(descriptor_path.read_bytes()) != task.task_descriptor.sha256
            ):
                raise ValueError("H1 task descriptor is unavailable or changed")
            descriptor = load_task(descriptor_path)
            expected = expected_descriptors[task.task_id]
            if descriptor != expected:
                raise ValueError("H1 task descriptor differs from deterministic grammar output")
            task_directory = descriptor_path.parent
            verify_public_package(task_directory, descriptor)
            load_hidden_cases(task_directory, descriptor)
            if task.synthesis_seeds != report.configuration.synthesis_seeds:
                raise ValueError("H1 task omits a configured synthesis seed")
            if len(task.attempts) != len(task.synthesis_seeds):
                raise ValueError("H1 task attempt count differs from synthesis seeds")
            attempts: list[tuple[ArtifactReference, H1AttemptEvidence]] = []
            for ordinal, (reference, synthesis_seed) in enumerate(
                zip(task.attempts, task.synthesis_seeds, strict=True)
            ):
                attempt = _load_attempt(reference)
                _validate_attempt(
                    attempt,
                    descriptor=descriptor,
                    task_directory=task_directory,
                    attempt_ordinal=ordinal,
                    synthesis_seed=synthesis_seed,
                )
                attempts.append((reference, attempt))
            rebuilt = _task_report(
                descriptor,
                descriptor_path,
                tuple(attempts),
                report.configuration.synthesis_seeds,
            )
            if rebuilt != task:
                raise ValueError("H1 task report is not derived from raw attempt evidence")
            rebuilt_tasks.append(rebuilt)
        rebuilt_aggregate = _aggregate(
            tuple(rebuilt_tasks), report.aggregate.measured_campaign_wall_ns
        )
        if rebuilt_aggregate != report.aggregate:
            raise ValueError("H1 aggregate is not derived from task-level evidence")


def run_h1_unseen_campaign(
    configuration: H1CampaignConfiguration, output_directory: Path
) -> H1CampaignReport:
    """Generate held-out tasks, synthesize for every seed, and validate raw evidence."""

    output = output_directory.absolute()
    if output.exists() and (output.is_symlink() or not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"H1 campaign output must be absent or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter_ns()
    descriptors = generate_tasks(configuration.grammar, output / "tasks")
    tasks: list[H1TaskReport] = []
    for descriptor in descriptors:
        task_directory = output / "tasks" / descriptor.task_id
        verify_public_package(task_directory, descriptor)
        attempt_artifacts: list[tuple[ArtifactReference, H1AttemptEvidence]] = []
        for ordinal, synthesis_seed in enumerate(configuration.synthesis_seeds):
            attempt_root = (
                output / "attempt-artifacts" / descriptor.task_id / f"seed-{synthesis_seed}"
            )
            evidence = _attempt(
                task_directory=task_directory,
                descriptor=descriptor,
                synthesis_seed=synthesis_seed,
                attempt_ordinal=ordinal,
                output=attempt_root,
                configuration=configuration,
            )
            reference = _write_new(
                output / "raw" / descriptor.task_id / f"seed-{synthesis_seed}.json",
                evidence,
            )
            attempt_artifacts.append((reference, evidence))
        tasks.append(
            _task_report(
                descriptor,
                task_directory / "task.json",
                tuple(attempt_artifacts),
                configuration.synthesis_seeds,
            )
        )
    report = H1CampaignReport(
        configuration=configuration,
        tasks=tuple(tasks),
        aggregate=_aggregate(tuple(tasks), max(1, time.perf_counter_ns() - started)),
    )
    report_path = output / "report.json"
    _write_new(report_path, report)
    persisted = H1CampaignReport.model_validate_json(report_path.read_bytes(), strict=True)
    validate_h1_unseen_campaign(persisted)
    return persisted


__all__ = [
    "ArtifactReference",
    "H1AggregateMetrics",
    "H1AttemptEvidence",
    "H1AttemptStatus",
    "H1CampaignConfiguration",
    "H1CampaignReport",
    "H1CaseEvidence",
    "H1TaskReport",
    "WilsonInterval",
    "run_h1_unseen_campaign",
    "validate_h1_unseen_campaign",
]
