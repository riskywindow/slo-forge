"""Typed request, capability, and result records for hostile-code execution."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
PositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(gt=0.0)]


class SandboxModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
        allow_inf_nan=False,
    )


class IsolationStatus(StrEnum):
    ENFORCED = "enforced"
    BEST_EFFORT = "best_effort"
    UNAVAILABLE = "unavailable"


class SandboxBackend(StrEnum):
    MACOS_SANDBOX_EXEC = "macos_sandbox_exec"
    LINUX_BUBBLEWRAP = "linux_bubblewrap"
    NONE = "none"


class SandboxCapabilities(SandboxModel):
    backend: SandboxBackend
    network_isolation: IsolationStatus
    filesystem_read_isolation: IsolationStatus
    filesystem_write_isolation: IsolationStatus
    environment_sanitization: IsolationStatus
    cpu_limit: IsolationStatus
    memory_limit: IsolationStatus
    process_limit: IsolationStatus
    output_limit: IsolationStatus
    child_cleanup: IsolationStatus
    limitations: tuple[NonEmptyString, ...] = ()


class SandboxLimits(SandboxModel):
    wall_time_seconds: PositiveFloat = 30.0
    cpu_time_seconds: PositiveInt = 15
    memory_bytes: Annotated[int, Field(ge=64 * 1024 * 1024)] = 1024 * 1024 * 1024
    process_count: Annotated[int, Field(ge=1, le=256)] = 16
    output_bytes: PositiveInt = 1024 * 1024
    artifact_bytes: PositiveInt = 256 * 1024 * 1024
    artifact_entries: PositiveInt = 4096
    open_files: Annotated[int, Field(ge=16, le=4096)] = 64


class EnvironmentVariable(SandboxModel):
    name: NonEmptyString
    value: str

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not value.replace("_", "").isalnum() or not (
            value[0].isalpha() or value[0] == "_"
        ):
            raise ValueError("environment variable name is invalid")
        return value


class SandboxRequest(SandboxModel):
    argv: tuple[NonEmptyString, ...]
    working_directory: Path
    read_only_paths: tuple[Path, ...]
    artifact_output_directory: Path
    environment: tuple[EnvironmentVariable, ...] = ()
    seed: int
    limits: SandboxLimits = Field(default_factory=SandboxLimits)
    require_network_isolation: bool = True
    require_filesystem_isolation: bool = True

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        if not self.argv:
            raise ValueError("argv must not be empty")
        if len({item.name for item in self.environment}) != len(self.environment):
            raise ValueError("environment variable names must be unique")
        if not self.read_only_paths:
            raise ValueError("at least one explicit read-only input path is required")
        return self


class SandboxTermination(StrEnum):
    SUCCESS = "success"
    NONZERO_EXIT = "nonzero_exit"
    TIMEOUT = "timeout"
    OUTPUT_LIMIT = "output_limit"
    POLICY_UNAVAILABLE = "policy_unavailable"
    SETUP_ERROR = "setup_error"
    SANDBOX_VIOLATION = "sandbox_violation"
    SIGNAL = "signal"


class SandboxResult(SandboxModel):
    termination: SandboxTermination
    return_code: int | None
    stdout: str
    stderr: str
    duration_seconds: Annotated[float, Field(ge=0.0)]
    output_bytes: Annotated[int, Field(ge=0)]
    capabilities: SandboxCapabilities
    sanitized_environment_names: tuple[NonEmptyString, ...]
    process_group_cleaned: bool
    artifact_output_directory: Path

    @property
    def succeeded(self) -> bool:
        return self.termination is SandboxTermination.SUCCESS
