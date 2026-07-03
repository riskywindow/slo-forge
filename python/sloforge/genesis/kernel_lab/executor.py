"""Sandbox-only execution and independent correctness comparison."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
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
) -> tuple[Path, Path, Path, Path]:
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
    return source_root, candidate_path, config_path, runner_path


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
    canonical_source = source_root.resolve(strict=True)
    canonical_candidate = candidate_path.resolve(strict=True)
    canonical_config = config_path.resolve(strict=True)
    canonical_output = output_root.resolve()
    runner = (canonical_source / "sandbox_runner.py").resolve(strict=True)
    result_path = canonical_output / "runner-output.json"
    request = SandboxRequest(
        argv=(
            sys.executable,
            str(runner),
            str(canonical_candidate),
            str(canonical_config),
            str(result_path),
            mode,
        ),
        working_directory=canonical_source,
        read_only_paths=(canonical_source, Path(sys.prefix), Path(sys.base_prefix)),
        artifact_output_directory=canonical_output,
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
    source_root, candidate_path, config_path, runner_path = _write_bundle(
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
        candidate_source_path=str(candidate_path.resolve()),
        candidate_source_sha256=hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
        cases_config_path=str(config_path.resolve()),
        cases_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        runner_path=str(runner_path.resolve()),
        runner_sha256=hashlib.sha256(runner_path.read_bytes()).hexdigest(),
        runner_output_path=str(result_path.resolve()),
        runner_output_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
    )


def validate_correctness_evidence(evidence: CorrectnessEvidence) -> None:
    """Reopen and independently replay passing generated-kernel correctness evidence."""

    if evidence.status is not LabStatus.PASSED:
        return
    identities = (
        (evidence.candidate_source_path, evidence.candidate_source_sha256),
        (evidence.cases_config_path, evidence.cases_config_sha256),
        (evidence.runner_path, evidence.runner_sha256),
        (evidence.runner_output_path, evidence.runner_output_sha256),
    )
    paths: list[Path] = []
    for path_value, digest in identities:
        if path_value is None or digest is None:
            raise ValueError("passing kernel correctness evidence omits raw artifact identity")
        path = Path(path_value)
        if path.is_symlink() or not path.is_file():
            raise ValueError("kernel correctness raw artifact is unavailable")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("kernel correctness raw artifact changed after execution")
        paths.append(path)
    source_path, config_path, runner_path, result_path = paths
    source = source_path.read_text(encoding="utf-8")
    if hashlib.sha256(source.encode()).hexdigest() != evidence.candidate.source_sha256:
        raise ValueError("kernel correctness source is not bound to its candidate")
    trusted_runner = Path(__file__).with_name("sandbox_runner.py")
    if runner_path.read_bytes() != trusted_runner.read_bytes():
        raise ValueError("kernel correctness used an unrecognized sandbox runner")
    configuration = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(configuration, dict) or configuration.get("mode") != "correctness":
        raise ValueError("kernel correctness configuration has an invalid mode")
    cases_value = configuration.get("cases")
    if not isinstance(cases_value, list):
        raise ValueError("kernel correctness configuration omits cases")
    cases = tuple(
        KernelCase.model_validate_json(json.dumps(item), strict=True) for item in cases_value
    )
    _RunnerOutput.model_validate_json(result_path.read_bytes(), strict=True)
    with tempfile.TemporaryDirectory(prefix="sloforge-kernel-replay-") as temporary:
        replay = execute_correctness(
            evidence.candidate,
            source,
            cases,
            output_root=Path(temporary),
            seed=evidence.deterministic_seed,
        )
    semantic_fields = (
        "status",
        "candidate",
        "deterministic_seed",
        "source_allowlist_valid",
        "cases_executed",
        "edge_cases",
        "randomized_cases",
        "stride_cases",
        "noncontiguous_cases",
        "nonfinite_cases",
        "alias_cases",
        "mismatches",
        "sandbox_termination",
        "sandbox_backend",
        "assumptions",
    )
    if any(getattr(replay, field) != getattr(evidence, field) for field in semantic_fields):
        raise ValueError("kernel correctness evidence does not match independent replay")


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
    source_root, candidate_path, config_path, _runner_path = _write_bundle(
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
