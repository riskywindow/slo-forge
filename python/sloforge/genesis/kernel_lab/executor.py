"""Sandbox-only execution and independent correctness comparison."""

from __future__ import annotations

import json
import shutil
import sys
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from ..sandbox import SandboxLimits, SandboxRequest, SandboxTermination, execute_sandboxed
from .generator import validate_generated_source
from .models import (
    CaseResult,
    CorrectnessEvidence,
    CorrectnessMismatch,
    KernelCandidate,
    KernelCase,
    LabStatus,
)


class _RunnerOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    case_results: tuple[CaseResult, ...]
    samples: tuple[dict[str, object], ...]


def _write_bundle(
    root: Path,
    candidate: KernelCandidate,
    source: str,
    configuration: dict[str, object],
) -> tuple[Path, Path, Path]:
    source_root = root / "source"
    source_root.mkdir(parents=True, exist_ok=True)
    candidate_path = source_root / "candidate.py"
    candidate_path.write_text(source, encoding="utf-8")
    runner_source = Path(__file__).with_name("sandbox_runner.py")
    runner_path = source_root / "sandbox_runner.py"
    shutil.copyfile(runner_source, runner_path)
    config_path = source_root / "config.json"
    config_path.write_text(
        json.dumps(configuration, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    return source_root, candidate_path, config_path


def _sandbox_request(
    *,
    source_root: Path,
    candidate_path: Path,
    config_path: Path,
    output_root: Path,
    seed: int,
    mode: str,
    wall_time_seconds: float,
) -> tuple[SandboxRequest, Path]:
    result_path = output_root / "runner-output.json"
    request = SandboxRequest(
        argv=(
            sys.executable,
            str(source_root / "sandbox_runner.py"),
            str(candidate_path),
            str(config_path),
            str(result_path),
            mode,
        ),
        working_directory=source_root,
        read_only_paths=(source_root, Path(sys.prefix)),
        artifact_output_directory=output_root,
        seed=seed,
        limits=SandboxLimits(
            wall_time_seconds=wall_time_seconds,
            cpu_time_seconds=max(1, int(wall_time_seconds)),
            memory_bytes=2 * 1024 * 1024 * 1024,
            process_count=4,
            output_bytes=64 * 1024,
            artifact_bytes=16 * 1024 * 1024,
            artifact_entries=128,
            open_files=32,
        ),
        require_network_isolation=True,
        require_filesystem_isolation=True,
    )
    return request, result_path


def execute_correctness(
    candidate: KernelCandidate,
    source: str,
    cases: tuple[KernelCase, ...],
    *,
    output_root: Path,
    seed: int,
    timeout_seconds: float = 10.0,
) -> CorrectnessEvidence:
    if not cases or len(cases) > 2_048:
        raise ValueError("correctness case count must be in [1, 2048]")
    diagnostics = validate_generated_source(candidate, source)
    counts = Counter(case.category for case in cases)
    randomized_count = counts["random"] + counts["stride"]
    assumptions = (
        "every symmetric int8 previous-state value is enumerated at zero activation",
        "randomized values are search data and not a universal proof",
        "generated source executes only in the fail-closed local sandbox",
    )
    if diagnostics:
        static_mismatches = tuple(
            CorrectnessMismatch(
                case_id="static-source",
                category="source_allowlist",
                expected="restricted generated source",
                observed=diagnostic,
            )
            for diagnostic in diagnostics
        )
        return CorrectnessEvidence(
            status=LabStatus.FAILED,
            candidate=candidate,
            deterministic_seed=seed,
            source_allowlist_valid=False,
            cases_executed=0,
            edge_cases=counts["edge"],
            randomized_cases=randomized_count,
            stride_cases=counts["stride"],
            noncontiguous_cases=counts["noncontiguous"],
            nonfinite_cases=counts["nonfinite"],
            alias_cases=counts["alias"],
            mismatches=static_mismatches,
            sandbox_termination="not_run_static_rejection",
            sandbox_backend="not_run",
            assumptions=assumptions,
        )
    configuration: dict[str, object] = {
        "mode": "correctness",
        "cases": [case.model_dump(mode="json") for case in cases],
    }
    source_root, candidate_path, config_path = _write_bundle(
        output_root, candidate, source, configuration
    )
    sandbox_output = output_root / "artifacts"
    request, result_path = _sandbox_request(
        source_root=source_root,
        candidate_path=candidate_path,
        config_path=config_path,
        output_root=sandbox_output,
        seed=seed,
        mode="correctness",
        wall_time_seconds=timeout_seconds,
    )
    result = execute_sandboxed(request)
    if not result.succeeded:
        status = (
            LabStatus.UNAVAILABLE
            if result.termination is SandboxTermination.POLICY_UNAVAILABLE
            else LabStatus.FAILED
        )
        return CorrectnessEvidence(
            status=status,
            candidate=candidate,
            deterministic_seed=seed,
            source_allowlist_valid=True,
            cases_executed=0,
            edge_cases=counts["edge"],
            randomized_cases=randomized_count,
            stride_cases=counts["stride"],
            noncontiguous_cases=counts["noncontiguous"],
            nonfinite_cases=counts["nonfinite"],
            alias_cases=counts["alias"],
            mismatches=(),
            sandbox_termination=result.termination.value,
            sandbox_backend=result.capabilities.backend.value,
            assumptions=assumptions,
        )
    try:
        payload = _RunnerOutput.model_validate_json(result_path.read_bytes(), strict=True)
    except (OSError, ValueError) as error:
        return CorrectnessEvidence(
            status=LabStatus.FAILED,
            candidate=candidate,
            deterministic_seed=seed,
            source_allowlist_valid=True,
            cases_executed=0,
            edge_cases=counts["edge"],
            randomized_cases=randomized_count,
            stride_cases=counts["stride"],
            noncontiguous_cases=counts["noncontiguous"],
            nonfinite_cases=counts["nonfinite"],
            alias_cases=counts["alias"],
            mismatches=(
                CorrectnessMismatch(
                    case_id="runner-output",
                    category="sandbox_protocol",
                    expected="valid bounded runner output",
                    observed=type(error).__name__,
                ),
            ),
            sandbox_termination=result.termination.value,
            sandbox_backend=result.capabilities.backend.value,
            assumptions=assumptions,
        )
    observed = {item.case_id: item for item in payload.case_results}
    mismatches: list[CorrectnessMismatch] = []
    for case in cases:
        item = observed.get(case.case_id)
        expected_storage = list(case.previous_storage)
        if case.output_alias_previous and case.expected is not None:
            for index, value in enumerate(case.expected):
                expected_storage[case.previous_offset + index * case.previous_stride] = value
        if item is None:
            mismatches.append(
                CorrectnessMismatch(
                    case_id=case.case_id,
                    category=case.category,
                    expected="one result",
                    observed="missing result",
                )
            )
        elif item.observed != case.expected or item.error_type != case.expected_error:
            mismatches.append(
                CorrectnessMismatch(
                    case_id=case.case_id,
                    category=case.category,
                    expected=f"values={case.expected!r}, error={case.expected_error!r}",
                    observed=f"values={item.observed!r}, error={item.error_type!r}",
                )
            )
        elif item.previous_storage_after != tuple(expected_storage):
            mismatches.append(
                CorrectnessMismatch(
                    case_id=case.case_id,
                    category="alias_atomicity",
                    expected=f"previous_storage_after={tuple(expected_storage)!r}",
                    observed=f"previous_storage_after={item.previous_storage_after!r}",
                )
            )
    if len(observed) != len(cases):
        extra = sorted(set(observed) - {case.case_id for case in cases})
        for case_id in extra:
            mismatches.append(
                CorrectnessMismatch(
                    case_id=case_id,
                    category="sandbox_protocol",
                    expected="no undeclared result",
                    observed="unexpected result",
                )
            )
    return CorrectnessEvidence(
        status=LabStatus.PASSED if not mismatches else LabStatus.FAILED,
        candidate=candidate,
        deterministic_seed=seed,
        source_allowlist_valid=True,
        cases_executed=len(payload.case_results),
        edge_cases=counts["edge"],
        randomized_cases=randomized_count,
        stride_cases=counts["stride"],
        noncontiguous_cases=counts["noncontiguous"],
        nonfinite_cases=counts["nonfinite"],
        alias_cases=counts["alias"],
        mismatches=tuple(mismatches),
        sandbox_termination=result.termination.value,
        sandbox_backend=result.capabilities.backend.value,
        assumptions=assumptions,
    )


def execute_benchmark_payload(
    candidate: KernelCandidate,
    source: str,
    configuration: dict[str, object],
    *,
    output_root: Path,
    seed: int,
    timeout_seconds: float,
) -> tuple[SandboxTermination, str, tuple[dict[str, object], ...]]:
    diagnostics = validate_generated_source(candidate, source)
    if diagnostics:
        return SandboxTermination.SANDBOX_VIOLATION, "not_run", ()
    source_root, candidate_path, config_path = _write_bundle(
        output_root, candidate, source, configuration
    )
    request, result_path = _sandbox_request(
        source_root=source_root,
        candidate_path=candidate_path,
        config_path=config_path,
        output_root=output_root / "artifacts",
        seed=seed,
        mode="benchmark",
        wall_time_seconds=timeout_seconds,
    )
    result = execute_sandboxed(request)
    if not result.succeeded:
        return result.termination, result.capabilities.backend.value, ()
    try:
        raw = json.loads(result_path.read_text(encoding="utf-8"))
        samples = tuple(raw["samples"])
    except (OSError, ValueError, KeyError, TypeError):
        return SandboxTermination.SANDBOX_VIOLATION, result.capabilities.backend.value, ()
    return result.termination, result.capabilities.backend.value, samples
