"""Capability-gated learned reward execution with fail-closed provenance checks."""

from __future__ import annotations

import json
import math
import sys
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sloforge.genesis.sandbox import (
    SandboxLimits,
    SandboxRequest,
    SandboxTermination,
    execute_sandboxed,
)


class _LearnedRewardModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
        allow_inf_nan=False,
    )


class LearnedRewardSpec(_LearnedRewardModel):
    reward_source_id: Annotated[str, Field(min_length=1, max_length=160)]
    source_version: Annotated[str, Field(min_length=1, max_length=160)]
    reward_policy_epoch_id: Annotated[str, Field(min_length=1, max_length=160)]
    runner: Annotated[str, Field(min_length=1, max_length=512)]
    model_artifact: Annotated[str, Field(min_length=1, max_length=512)]
    calibration_artifact: Annotated[str, Field(min_length=1, max_length=512)]
    known_limitations: Annotated[tuple[str, ...], Field(max_length=64)] = ()


class LearnedRewardComponent(_LearnedRewardModel):
    name: Annotated[str, Field(min_length=1, max_length=160)]
    score: float
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]


class LearnedRewardOutput(_LearnedRewardModel):
    score: float
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    components: Annotated[tuple[LearnedRewardComponent, ...], Field(min_length=1, max_length=256)]

    @field_validator("components", mode="before")
    @classmethod
    def decode_json_components(cls, value: object) -> object:
        # JSON has arrays rather than tuples. Convert only the outer container;
        # component records still receive strict typed validation.
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_component_total(self) -> Self:
        if abs(sum(component.score for component in self.components) - self.score) > 1e-9:
            raise ValueError("learned reward score must preserve its component sum")
        return self


class LearnedRewardEvidence(_LearnedRewardModel):
    schema_version: Literal["sloforge.helix.learned-reward/v1"] = "sloforge.helix.learned-reward/v1"
    reward_id: Annotated[str, Field(min_length=1, max_length=160)]
    trajectory_id: Annotated[str, Field(min_length=1, max_length=160)]
    behavior_policy_epoch_id: Annotated[str, Field(min_length=1, max_length=160)]
    reward_policy_epoch_id: Annotated[str, Field(min_length=1, max_length=160)]
    reward_source_id: Annotated[str, Field(min_length=1, max_length=160)]
    source_version: Annotated[str, Field(min_length=1, max_length=160)]
    input_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    runner_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    model_artifact_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    calibration_artifact_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    output_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    score: float
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    components: Annotated[tuple[LearnedRewardComponent, ...], Field(min_length=1, max_length=256)]
    deterministic: Literal[False] = False
    authority_verified: Literal[True] = True
    source_immutable: Literal[True] = True
    sandbox_termination: Literal[SandboxTermination.SUCCESS] = SandboxTermination.SUCCESS
    known_limitations: Annotated[tuple[str, ...], Field(max_length=64)] = ()


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"learned reward artifact must be a regular file: {path}")
    return sha256(path.read_bytes()).hexdigest()


def _resolve_file(root: Path, relative: str) -> Path:
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"learned reward artifact escapes its authority root: {relative}")
    return path


class LearnedRewardWorker:
    """Run a pre-authorized learned scorer without exposing network or credentials."""

    def __init__(
        self,
        *,
        trusted_runner_digests: frozenset[str],
        trusted_model_digests: frozenset[str],
        trusted_calibration_digests: frozenset[str],
        wall_time_seconds: float = 10.0,
    ) -> None:
        for label, values in (
            ("runner", trusted_runner_digests),
            ("model", trusted_model_digests),
            ("calibration", trusted_calibration_digests),
        ):
            if not values:
                raise ValueError(f"learned reward {label} authority set must not be empty")
            if any(
                len(item) != 64 or any(c not in "0123456789abcdef" for c in item) for item in values
            ):
                raise ValueError(f"learned reward {label} authority contains an invalid SHA-256")
        self._trusted_runner_digests = trusted_runner_digests
        self._trusted_model_digests = trusted_model_digests
        self._trusted_calibration_digests = trusted_calibration_digests
        self._limits = SandboxLimits(
            wall_time_seconds=wall_time_seconds,
            cpu_time_seconds=max(1, math.ceil(wall_time_seconds)),
            memory_bytes=1024 * 1024 * 1024,
            process_count=8,
            output_bytes=256 * 1024,
            artifact_bytes=1024 * 1024,
            artifact_entries=32,
            open_files=64,
        )
        self._submissions: set[str] = set()

    def evaluate(
        self,
        *,
        reward_id: str,
        trajectory_id: str,
        behavior_policy_epoch_id: str,
        authority_root: Path,
        input_artifact: str,
        spec: LearnedRewardSpec,
        evidence_directory: Path,
        seed: int,
    ) -> LearnedRewardEvidence:
        root = authority_root.resolve(strict=True)
        runner = _resolve_file(root, spec.runner)
        model = _resolve_file(root, spec.model_artifact)
        calibration = _resolve_file(root, spec.calibration_artifact)
        input_path = _resolve_file(root, input_artifact)
        runner_hash = _sha256_file(runner)
        model_hash = _sha256_file(model)
        calibration_hash = _sha256_file(calibration)
        if runner_hash not in self._trusted_runner_digests:
            raise PermissionError("learned reward runner is outside the trusted authority set")
        if model_hash not in self._trusted_model_digests:
            raise PermissionError("learned reward model is outside the trusted authority set")
        if calibration_hash not in self._trusted_calibration_digests:
            raise PermissionError("learned reward calibration is outside the trusted authority set")
        submission = sha256(
            f"helix-learned-reward/v1\0{reward_id}\0{trajectory_id}".encode()
        ).hexdigest()
        if submission in self._submissions:
            raise ValueError("duplicate learned reward submission")
        self._submissions.add(submission)
        before = {path: _sha256_file(path) for path in (runner, model, calibration, input_path)}
        result = execute_sandboxed(
            SandboxRequest(
                argv=(
                    sys.executable,
                    str(runner),
                    "--input",
                    str(input_path),
                    "--model",
                    str(model),
                    "--calibration",
                    str(calibration),
                    "--seed",
                    str(seed),
                ),
                working_directory=root,
                read_only_paths=(root,),
                artifact_output_directory=evidence_directory,
                seed=seed,
                limits=self._limits,
                require_network_isolation=True,
                require_filesystem_isolation=True,
            )
        )
        if result.termination is not SandboxTermination.SUCCESS or result.return_code != 0:
            raise RuntimeError(
                f"learned reward execution failed closed: {result.termination.value}"
            )
        try:
            parsed = LearnedRewardOutput.model_validate(json.loads(result.stdout))
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError("learned reward output failed strict schema validation") from error
        after = {path: _sha256_file(path) for path in before}
        if before != after:
            raise ValueError("learned reward mutated an immutable authority artifact")
        output_bytes = result.stdout.encode("utf-8")
        return LearnedRewardEvidence(
            reward_id=reward_id,
            trajectory_id=trajectory_id,
            behavior_policy_epoch_id=behavior_policy_epoch_id,
            reward_policy_epoch_id=spec.reward_policy_epoch_id,
            reward_source_id=spec.reward_source_id,
            source_version=spec.source_version,
            input_sha256=before[input_path],
            runner_sha256=runner_hash,
            model_artifact_sha256=model_hash,
            calibration_artifact_sha256=calibration_hash,
            output_sha256=sha256(output_bytes).hexdigest(),
            score=parsed.score,
            confidence=parsed.confidence,
            components=parsed.components,
            known_limitations=spec.known_limitations,
        )
