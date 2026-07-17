"""Deterministic, hash-bound reward execution in the existing strict sandbox."""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Collection, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sloforge.genesis.sandbox import (
    SandboxLimits,
    SandboxRequest,
    SandboxResult,
    SandboxTermination,
    execute_sandboxed,
)
from sloforge.helix.security import (
    sanitize_tool_output,
    validate_tool_output_evidence,
)


class _RewardModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _tree_hash(root: Path) -> str:
    root = root.resolve(strict=True)
    digest = sha256(b"sloforge.helix.reward-tree/v1\0")
    entries = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            target = path.resolve(strict=True)
            if not target.is_relative_to(root):
                raise ValueError(f"reward source contains escaping symlink: {path}")
            continue
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        entries += 1
        if entries > 10_000:
            raise ValueError("reward source file count exceeds 10,000")
        relative = path.relative_to(root).as_posix()
        payload = path.read_bytes()
        if len(payload) > 8 * 1024 * 1024:
            raise ValueError(f"reward source file exceeds 8 MiB: {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(payload)).encode("ascii"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return digest.hexdigest()


class VerifierCommand(_RewardModel):
    verifier_id: Annotated[str, Field(min_length=1, max_length=160)]
    argv: Annotated[tuple[str, ...], Field(min_length=1, max_length=64)]
    source_version: Annotated[str, Field(min_length=1, max_length=160)]
    expected_return_code: Annotated[int, Field(ge=0, le=255)] = 0
    score_on_pass: float = 1.0
    score_on_fail: float = -1.0

    @field_validator("argv")
    @classmethod
    def validate_argv(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item or len(item) > 4096 for item in value):
            raise ValueError("verifier argv entries must be bounded and non-empty")
        return value


class HiddenCase(_RewardModel):
    """A black-box case whose expected output never enters the policy sandbox."""

    case_id: Annotated[str, Field(min_length=1, max_length=160)]
    runner: Annotated[str, Field(min_length=1, max_length=512)]
    arguments: Annotated[tuple[str, ...], Field(max_length=64)]
    expected_stdout: Annotated[str, Field(max_length=65_536)]
    score_on_pass: float = 1.0
    score_on_fail: float = -1.0


class RewardComponentResult(_RewardModel):
    component_id: str
    source_type: Literal["deterministic_command", "hidden_black_box"]
    source_version: str
    input_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    output_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    score: float
    passed: bool
    confidence: Annotated[float, Field(ge=1.0, le=1.0)] = 1.0
    deterministic: Literal[True] = True
    termination: SandboxTermination
    return_code: int | None
    duration_seconds: Annotated[float, Field(ge=0.0)]
    stdout_excerpt: Annotated[str, Field(max_length=4096)]
    stderr_excerpt: Annotated[str, Field(max_length=4096)]
    stdout_security_evidence_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    stderr_security_evidence_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    output_redaction_verified: Literal[True] = True

    @model_validator(mode="after")
    def validate_finite_score(self) -> Self:
        if not math.isfinite(self.score):
            raise ValueError("reward scores must be finite")
        return self


class RewardRun(_RewardModel):
    schema_version: str = "sloforge.helix.deterministic-reward/v1"
    reward_id: Annotated[str, Field(min_length=1, max_length=160)]
    tenant_id: Annotated[str, Field(min_length=1, max_length=160)] = "default"
    trajectory_id: Annotated[str, Field(min_length=1, max_length=160)]
    policy_epoch_id: Annotated[str, Field(min_length=1, max_length=160)]
    environment_hash_before: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    environment_hash_after: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    verifier_spec_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    evaluator_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    evaluator_trusted: bool
    source_artifact_verified: Literal[True] = True
    components: Annotated[tuple[RewardComponentResult, ...], Field(min_length=1, max_length=1_024)]
    total_score: float
    immutable_source: bool
    hidden_expected_values_exposed: Literal[False] = False

    @model_validator(mode="after")
    def validate_total(self) -> Self:
        if abs(sum(item.score for item in self.components) - self.total_score) > 1e-12:
            raise ValueError("total reward must preserve and sum its components")
        if self.immutable_source != (self.environment_hash_before == self.environment_hash_after):
            raise ValueError("immutable-source flag disagrees with environment hashes")
        return self


class DeterministicRewardWorker:
    def __init__(
        self,
        *,
        wall_time_seconds: float = 10.0,
        output_bytes: int = 256 * 1024,
        memory_bytes: int = 1024 * 1024 * 1024,
        trusted_evaluator_digests: Collection[str] | None = None,
    ) -> None:
        self.limits = SandboxLimits(
            wall_time_seconds=wall_time_seconds,
            cpu_time_seconds=max(1, math.ceil(wall_time_seconds)),
            memory_bytes=memory_bytes,
            process_count=16,
            output_bytes=output_bytes,
            artifact_bytes=1024 * 1024,
            artifact_entries=128,
            open_files=64,
        )
        if trusted_evaluator_digests is not None and any(
            len(item) != 64 or any(character not in "0123456789abcdef" for character in item)
            for item in trusted_evaluator_digests
        ):
            raise ValueError("trusted evaluator identities must be lowercase SHA-256 digests")
        self._trusted_evaluator_digests = (
            None if trusted_evaluator_digests is None else frozenset(trusted_evaluator_digests)
        )
        self._submissions: set[str] = set()

    @staticmethod
    def _input_hash(identifier: str, argv: tuple[str, ...]) -> str:
        payload = json.dumps(
            {"identifier": identifier, "argv": argv},
            sort_keys=True,
            separators=(",", ":"),
        )
        return sha256(payload.encode("utf-8")).hexdigest()

    def _execute(
        self,
        *,
        source: Path,
        output: Path,
        argv: tuple[str, ...],
        seed: int,
    ) -> SandboxResult:
        return execute_sandboxed(
            SandboxRequest(
                argv=argv,
                working_directory=source,
                read_only_paths=(source,),
                artifact_output_directory=output,
                seed=seed,
                limits=self.limits,
                require_network_isolation=True,
                require_filesystem_isolation=True,
            )
        )

    def verify(
        self,
        *,
        reward_id: str,
        trajectory_id: str,
        policy_epoch_id: str,
        tenant_id: str = "default",
        source: Path,
        evidence_directory: Path,
        commands: tuple[VerifierCommand, ...],
        hidden_cases: tuple[HiddenCase, ...],
        seed: int,
        secret_values: Sequence[str] = (),
    ) -> RewardRun:
        if not commands and not hidden_cases:
            raise ValueError("deterministic reward requires at least one verifier")
        source = source.resolve(strict=True)
        if not source.is_dir():
            raise ValueError("reward source must be a directory")
        before = _tree_hash(source)
        spec_payload = _specification_payload(commands, hidden_cases)
        spec_hash = sha256(spec_payload.encode("utf-8")).hexdigest()
        evaluator_digest = compute_evaluator_digest(
            source=source,
            commands=commands,
            hidden_cases=hidden_cases,
        )
        evaluator_trusted = self._trusted_evaluator_digests is not None
        if (
            self._trusted_evaluator_digests is not None
            and evaluator_digest not in self._trusted_evaluator_digests
        ):
            raise PermissionError("reward evaluator is not present in the trusted authority set")
        submission = sha256(
            f"helix-reward-submission/v1\0{tenant_id}\0{reward_id}\0{trajectory_id}".encode()
        ).hexdigest()
        if submission in self._submissions:
            raise ValueError("duplicate reward submission")
        self._submissions.add(submission)
        components: list[RewardComponentResult] = []
        for index, command in enumerate(commands):
            argv = tuple(sys.executable if item == "{python}" else item for item in command.argv)
            result = self._execute(
                source=source,
                output=evidence_directory / f"command-{index:04d}",
                argv=argv,
                seed=seed + index,
            )
            passed = (
                result.termination is SandboxTermination.SUCCESS
                and result.return_code == command.expected_return_code
            )
            output_payload = f"{result.return_code}\0{result.stdout}\0{result.stderr}".encode()
            stdout, stdout_security, stdout_redaction = sanitize_tool_output(
                result.stdout.encode("utf-8"), secret_values=secret_values, max_bytes=4096
            )
            stderr, stderr_security, stderr_redaction = sanitize_tool_output(
                result.stderr.encode("utf-8"), secret_values=secret_values, max_bytes=4096
            )
            if (
                not validate_tool_output_evidence(
                    stdout_security,
                    raw_output=result.stdout.encode("utf-8"),
                    sanitized_output=stdout,
                    redaction_evidence=stdout_redaction,
                    checked_at_ms=0,
                ).passed
                or not validate_tool_output_evidence(
                    stderr_security,
                    raw_output=result.stderr.encode("utf-8"),
                    sanitized_output=stderr,
                    redaction_evidence=stderr_redaction,
                    checked_at_ms=0,
                ).passed
            ):
                raise ValueError("reward output security evidence failed validation")
            components.append(
                RewardComponentResult(
                    component_id=command.verifier_id,
                    source_type="deterministic_command",
                    source_version=command.source_version,
                    input_hash=self._input_hash(command.verifier_id, argv),
                    output_hash=sha256(output_payload).hexdigest(),
                    score=command.score_on_pass if passed else command.score_on_fail,
                    passed=passed,
                    termination=result.termination,
                    return_code=result.return_code,
                    duration_seconds=result.duration_seconds,
                    stdout_excerpt=stdout,
                    stderr_excerpt=stderr,
                    stdout_security_evidence_sha256=stdout_security.evidence_digest,
                    stderr_security_evidence_sha256=stderr_security.evidence_digest,
                )
            )
        for index, case in enumerate(hidden_cases):
            runner = (source / case.runner).resolve(strict=True)
            if not runner.is_relative_to(source) or not runner.is_file():
                raise ValueError(f"hidden case runner escapes source: {case.runner}")
            argv = (sys.executable, str(runner), *case.arguments)
            result = self._execute(
                source=source,
                output=evidence_directory / f"hidden-{index:04d}",
                argv=argv,
                seed=seed + len(commands) + index,
            )
            passed = (
                result.termination is SandboxTermination.SUCCESS
                and result.return_code == 0
                and result.stdout.strip() == case.expected_stdout.strip()
            )
            output_payload = f"{result.return_code}\0{result.stdout}\0{result.stderr}".encode()
            stdout, stdout_security, stdout_redaction = sanitize_tool_output(
                result.stdout.encode("utf-8"), secret_values=secret_values, max_bytes=4096
            )
            stderr, stderr_security, stderr_redaction = sanitize_tool_output(
                result.stderr.encode("utf-8"), secret_values=secret_values, max_bytes=4096
            )
            if (
                not validate_tool_output_evidence(
                    stdout_security,
                    raw_output=result.stdout.encode("utf-8"),
                    sanitized_output=stdout,
                    redaction_evidence=stdout_redaction,
                    checked_at_ms=0,
                ).passed
                or not validate_tool_output_evidence(
                    stderr_security,
                    raw_output=result.stderr.encode("utf-8"),
                    sanitized_output=stderr,
                    redaction_evidence=stderr_redaction,
                    checked_at_ms=0,
                ).passed
            ):
                raise ValueError("reward output security evidence failed validation")
            components.append(
                RewardComponentResult(
                    component_id=case.case_id,
                    source_type="hidden_black_box",
                    source_version="hidden-case/v1",
                    input_hash=self._input_hash(case.case_id, argv),
                    output_hash=sha256(output_payload).hexdigest(),
                    score=case.score_on_pass if passed else case.score_on_fail,
                    passed=passed,
                    termination=result.termination,
                    return_code=result.return_code,
                    duration_seconds=result.duration_seconds,
                    stdout_excerpt=stdout,
                    stderr_excerpt=stderr,
                    stdout_security_evidence_sha256=stdout_security.evidence_digest,
                    stderr_security_evidence_sha256=stderr_security.evidence_digest,
                )
            )
        after = _tree_hash(source)
        if before != after:
            raise ValueError("reward source artifact changed during verification")
        if evaluator_digest != compute_evaluator_digest(
            source=source, commands=commands, hidden_cases=hidden_cases
        ):
            raise ValueError("reward evaluator artifact changed during verification")
        return RewardRun(
            reward_id=reward_id,
            tenant_id=tenant_id,
            trajectory_id=trajectory_id,
            policy_epoch_id=policy_epoch_id,
            environment_hash_before=before,
            environment_hash_after=after,
            verifier_spec_hash=spec_hash,
            evaluator_sha256=evaluator_digest,
            evaluator_trusted=evaluator_trusted,
            components=tuple(components),
            total_score=sum(item.score for item in components),
            immutable_source=before == after,
        )


def _specification_payload(
    commands: tuple[VerifierCommand, ...], hidden_cases: tuple[HiddenCase, ...]
) -> str:
    return json.dumps(
        {
            "commands": [item.model_dump(mode="json") for item in commands],
            "hidden": [
                {
                    "case_id": item.case_id,
                    "runner": item.runner,
                    "arguments": item.arguments,
                    "expected_hash": sha256(item.expected_stdout.encode()).hexdigest(),
                    "score_on_pass": item.score_on_pass,
                    "score_on_fail": item.score_on_fail,
                }
                for item in hidden_cases
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def compute_evaluator_digest(
    *,
    source: Path,
    commands: tuple[VerifierCommand, ...],
    hidden_cases: tuple[HiddenCase, ...],
) -> str:
    """Bind a reward authority identity to its spec and exact local verifier bytes."""

    root = source.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("reward source must be a directory")
    verifier_files: set[Path] = set()
    for command in commands:
        for argument in command.argv:
            if argument == "{python}":
                continue
            candidate = Path(argument)
            if not candidate.is_absolute():
                candidate = root / candidate
            try:
                resolved = candidate.resolve(strict=True)
            except OSError:
                continue
            if resolved.is_relative_to(root) and resolved.is_file():
                verifier_files.add(resolved)
    for case in hidden_cases:
        runner = (root / case.runner).resolve(strict=True)
        if not runner.is_relative_to(root) or not runner.is_file():
            raise ValueError(f"hidden case runner escapes source: {case.runner}")
        verifier_files.add(runner)
    digest = sha256(b"sloforge.helix.reward-evaluator/v1\0")
    digest.update(_specification_payload(commands, hidden_cases).encode("utf-8"))
    digest.update(b"\0")
    for path in sorted(verifier_files):
        relative = path.relative_to(root).as_posix()
        payload = path.read_bytes()
        if len(payload) > 8 * 1024 * 1024:
            raise ValueError("reward evaluator artifact exceeds 8 MiB")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256(payload).digest())
        digest.update(b"\0")
    return digest.hexdigest()
