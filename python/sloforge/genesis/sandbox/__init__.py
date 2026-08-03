"""Capability-aware hostile generated-code sandbox."""

from .executor import detect_capabilities, execute_sandboxed, interpreter_read_roots
from .models import (
    EnvironmentVariable,
    IsolationStatus,
    SandboxBackend,
    SandboxCapabilities,
    SandboxLimits,
    SandboxRequest,
    SandboxResult,
    SandboxTermination,
)

__all__ = [
    "EnvironmentVariable",
    "IsolationStatus",
    "SandboxBackend",
    "SandboxCapabilities",
    "SandboxLimits",
    "SandboxRequest",
    "SandboxResult",
    "SandboxTermination",
    "detect_capabilities",
    "execute_sandboxed",
    "interpreter_read_roots",
]
