"""Fail-closed local sandbox for untrusted generated programs.

The executor uses macOS Sandbox profiles or Linux bubblewrap when available.
It never invokes a shell, never inherits the host environment, and kills the
whole process group on timeout or output overflow.
"""

from __future__ import annotations

import json
import math
import os
import platform
import resource
import selectors
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from .models import (
    IsolationStatus,
    SandboxBackend,
    SandboxCapabilities,
    SandboxRequest,
    SandboxResult,
    SandboxTermination,
)

_BLOCKED_ENVIRONMENT_MARKERS = (
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "AWS_",
    "AZURE_",
    "GOOGLE_",
    "GCP_",
    "OPENAI_",
    "ANTHROPIC_",
    "SSH_",
    "KUBECONFIG",
)
_ALLOWED_CALLER_ENVIRONMENT = frozenset({"LANG", "LC_ALL", "TZ"})


def detect_capabilities() -> SandboxCapabilities:
    system = platform.system()
    if system == "Darwin" and Path("/usr/bin/sandbox-exec").is_file():
        return SandboxCapabilities(
            backend=SandboxBackend.MACOS_SANDBOX_EXEC,
            network_isolation=IsolationStatus.ENFORCED,
            filesystem_read_isolation=IsolationStatus.ENFORCED,
            filesystem_write_isolation=IsolationStatus.ENFORCED,
            environment_sanitization=IsolationStatus.ENFORCED,
            cpu_limit=IsolationStatus.ENFORCED,
            memory_limit=IsolationStatus.BEST_EFFORT,
            process_limit=IsolationStatus.ENFORCED,
            output_limit=IsolationStatus.ENFORCED,
            child_cleanup=IsolationStatus.ENFORCED,
            limitations=(
                "sandbox-exec is deprecated by Apple and must be revalidated after OS updates",
                "RLIMIT_AS behavior varies across macOS runtime implementations",
            ),
        )
    if system == "Linux" and shutil.which("bwrap") is not None:
        return SandboxCapabilities(
            backend=SandboxBackend.LINUX_BUBBLEWRAP,
            network_isolation=IsolationStatus.ENFORCED,
            filesystem_read_isolation=IsolationStatus.ENFORCED,
            filesystem_write_isolation=IsolationStatus.ENFORCED,
            environment_sanitization=IsolationStatus.ENFORCED,
            cpu_limit=IsolationStatus.ENFORCED,
            memory_limit=IsolationStatus.BEST_EFFORT,
            process_limit=IsolationStatus.BEST_EFFORT,
            output_limit=IsolationStatus.ENFORCED,
            child_cleanup=IsolationStatus.ENFORCED,
            limitations=(
                "bubblewrap availability does not guarantee user namespaces are enabled",
                "RLIMIT_NPROC is user-scoped rather than a cgroup process counter",
            ),
        )
    return SandboxCapabilities(
        backend=SandboxBackend.NONE,
        network_isolation=IsolationStatus.UNAVAILABLE,
        filesystem_read_isolation=IsolationStatus.UNAVAILABLE,
        filesystem_write_isolation=IsolationStatus.UNAVAILABLE,
        environment_sanitization=IsolationStatus.ENFORCED,
        cpu_limit=IsolationStatus.ENFORCED,
        memory_limit=IsolationStatus.BEST_EFFORT,
        process_limit=IsolationStatus.BEST_EFFORT,
        output_limit=IsolationStatus.ENFORCED,
        child_cleanup=IsolationStatus.ENFORCED,
        limitations=("no supported kernel sandbox backend was found; strict execution fails closed",),
    )


def _is_credential_name(name: str) -> bool:
    upper = name.upper()
    return any(marker in upper for marker in _BLOCKED_ENVIRONMENT_MARKERS)


def _sanitized_environment(request: SandboxRequest) -> dict[str, str]:
    environment = {
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TZ": "UTC",
        "PYTHONHASHSEED": str(request.seed),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "HOME": str(request.artifact_output_directory / "home"),
        "TMPDIR": str(request.artifact_output_directory / "tmp"),
    }
    for item in request.environment:
        if _is_credential_name(item.name):
            raise ValueError(f"credential-like environment variable is forbidden: {item.name}")
        if item.name not in _ALLOWED_CALLER_ENVIRONMENT:
            raise ValueError(f"environment variable is not allowlisted: {item.name}")
        environment[item.name] = item.value
    return environment


def _canonical_paths(request: SandboxRequest) -> tuple[Path, tuple[Path, ...], Path]:
    output = request.artifact_output_directory
    if output.exists() and output.is_symlink():
        raise ValueError("artifact output directory must not be a symlink")
    output.mkdir(parents=True, exist_ok=True)
    output = output.resolve(strict=True)
    (output / "tmp").mkdir(mode=0o700, exist_ok=True)
    (output / "home").mkdir(mode=0o700, exist_ok=True)
    working = request.working_directory.resolve(strict=True)
    if not working.is_dir():
        raise ValueError("working_directory must be a directory")
    read_only: list[Path] = []
    for item in request.read_only_paths:
        if item.is_symlink():
            raise ValueError(f"read-only input must not be a symlink: {item}")
        read_only.append(item.resolve(strict=True))
    if not any(working == item or working.is_relative_to(item) for item in read_only):
        raise ValueError("working_directory must be within an explicit read-only input")
    if any(output == item or output.is_relative_to(item) for item in read_only):
        raise ValueError("artifact output must not be nested in a read-only input")
    return working, tuple(dict.fromkeys(read_only)), output


def _sandbox_string(path: Path) -> str:
    return json.dumps(str(path))


def _macos_profile(read_only: tuple[Path, ...], output: Path, executable: Path) -> str:
    runtime_roots = tuple(
        dict.fromkeys(
            path
            for path in (
                Path("/System"),
                Path("/usr/lib"),
                Path("/usr/share"),
                Path("/private/etc/localtime"),
                Path(sys.base_prefix).resolve(),
                Path(sys.prefix).resolve(),
                executable.absolute(),
                executable.absolute().parent,
                executable.resolve(),
                *read_only,
            )
            if path.exists()
        )
    )
    reads = "\n".join(f"(allow file-read* (subpath {_sandbox_string(path)}))" for path in runtime_roots)
    common = Path(os.path.commonpath([*(str(item) for item in read_only), str(output)]))
    protected_roots = tuple(
        dict.fromkeys(
            path
            for path in (Path.home().resolve(), common)
            if path != Path("/") and path.exists()
        )
    )
    read_denials = "\n".join(
        f"(deny file-read* (subpath {_sandbox_string(path)}))" for path in protected_roots
    )
    return "\n".join(
        (
            "(version 1)",
            "(allow default)",
            "(deny network*)",
            "(deny process-fork)",
            "(deny file-write*)",
            read_denials,
            reads,
            f"(allow file-read* (subpath {_sandbox_string(output)}))",
            f"(allow file-write* (subpath {_sandbox_string(output)}))",
            "(allow file-read* file-write* (literal \"/dev/null\"))",
            "(allow file-read* (literal \"/dev/urandom\"))",
        )
    )


def _bubblewrap_command(
    argv: tuple[str, ...], read_only: tuple[Path, ...], output: Path, working: Path
) -> list[str]:
    command = [
        "bwrap",
        "--die-with-parent",
        "--new-session",
        "--unshare-all",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
    ]
    runtime_roots = [Path(item) for item in ("/usr", "/bin", "/lib", "/lib64", "/etc")]
    for path in tuple(runtime_roots) + read_only:
        if path.exists():
            command.extend(("--ro-bind", str(path), str(path)))
    command.extend(("--bind", str(output), str(output), "--chdir", str(working), "--"))
    command.extend(argv)
    return command


def _build_command(
    request: SandboxRequest,
    capabilities: SandboxCapabilities,
    working: Path,
    read_only: tuple[Path, ...],
    output: Path,
) -> list[str]:
    argv = list(request.argv)
    executable = Path(argv[0])
    if not executable.is_absolute():
        resolved = shutil.which(argv[0])
        if resolved is None:
            raise ValueError(f"executable was not found: {argv[0]}")
        executable = Path(resolved)
        argv[0] = resolved
    if capabilities.backend is SandboxBackend.MACOS_SANDBOX_EXEC:
        profile = _macos_profile(read_only, output, executable)
        return ["/usr/bin/sandbox-exec", "-p", profile, *argv]
    if capabilities.backend is SandboxBackend.LINUX_BUBBLEWRAP:
        return _bubblewrap_command(tuple(argv), read_only, output, working)
    return argv


def _limit_resources(request: SandboxRequest) -> None:
    limits = request.limits

    def bounded_value(kind: int, value: int) -> int:
        _, hard = resource.getrlimit(kind)
        return min(value, hard)

    def apply_required_limit(kind: int, value: int) -> None:
        bounded = bounded_value(kind, value)
        resource.setrlimit(kind, (bounded, bounded))

    def apply_best_effort_limit(kind: int, value: int) -> None:
        try:
            bounded = bounded_value(kind, value)
            resource.setrlimit(kind, (bounded, bounded))
        except (OSError, ValueError):
            return

    apply_required_limit(resource.RLIMIT_CPU, limits.cpu_time_seconds)
    apply_best_effort_limit(resource.RLIMIT_AS, limits.memory_bytes)
    # macOS RLIMIT_NPROC is per-user; lowering it below the host's current
    # process count prevents sandbox-exec itself from starting its target.
    # The macOS policy denies process-fork outright, which is stronger.
    if platform.system() != "Darwin":
        apply_best_effort_limit(resource.RLIMIT_NPROC, limits.process_count)
    apply_required_limit(resource.RLIMIT_FSIZE, limits.artifact_bytes)
    apply_required_limit(resource.RLIMIT_NOFILE, limits.open_files)


def _failure_result(
    request: SandboxRequest,
    capabilities: SandboxCapabilities,
    termination: SandboxTermination,
    message: str,
    duration: float,
) -> SandboxResult:
    return SandboxResult(
        termination=termination,
        return_code=None,
        stdout="",
        stderr=message,
        duration_seconds=duration,
        output_bytes=len(message.encode("utf-8")),
        capabilities=capabilities,
        sanitized_environment_names=(),
        process_group_cleaned=True,
        artifact_output_directory=request.artifact_output_directory,
    )


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def _validate_artifact_tree(output: Path, maximum_bytes: int, maximum_entries: int) -> str | None:
    pending = [output]
    observed_bytes = 0
    observed_entries = 0
    while pending:
        directory = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as exc:
            return f"artifact tree cannot be inspected: {exc}"
        for entry in entries:
            observed_entries += 1
            if observed_entries > maximum_entries:
                return "artifact tree exceeds the configured entry limit"
            try:
                if entry.is_symlink():
                    return "artifact tree contains a symlink"
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))
                    continue
                if not entry.is_file(follow_symlinks=False):
                    return "artifact tree contains a non-regular file"
                observed_bytes += entry.stat(follow_symlinks=False).st_size
            except OSError as exc:
                return f"artifact tree changed during inspection: {exc}"
            if observed_bytes > maximum_bytes:
                return "artifact tree exceeds the configured byte limit"
    return None


def _bounded_output(
    process: subprocess.Popen[bytes], maximum_bytes: int, deadline: float
) -> tuple[bytes, bytes, int, SandboxTermination | None]:
    selector = selectors.DefaultSelector()
    assert process.stdout is not None
    assert process.stderr is not None
    streams = {process.stdout.fileno(): "stdout", process.stderr.fileno(): "stderr"}
    for file_descriptor, label in streams.items():
        os.set_blocking(file_descriptor, False)
        selector.register(file_descriptor, selectors.EVENT_READ, label)
    stdout = bytearray()
    stderr = bytearray()
    observed = 0
    termination: SandboxTermination | None = None
    while selector.get_map():
        remaining = deadline - time.monotonic()
        if remaining <= 0 and termination is None:
            termination = SandboxTermination.TIMEOUT
            _kill_process_group(process)
        events = selector.select(timeout=max(0.0, min(0.05, remaining)))
        for key, _ in events:
            try:
                chunk = os.read(key.fd, 65536)
            except BlockingIOError:
                continue
            if not chunk:
                selector.unregister(key.fileobj)
                continue
            observed += len(chunk)
            if termination is None and observed > maximum_bytes:
                termination = SandboxTermination.OUTPUT_LIMIT
                _kill_process_group(process)
            if len(stdout) + len(stderr) < maximum_bytes:
                capacity = maximum_bytes - len(stdout) - len(stderr)
                if key.data == "stdout":
                    stdout.extend(chunk[:capacity])
                else:
                    stderr.extend(chunk[:capacity])
        if process.poll() is not None and not events:
            for key in list(selector.get_map().values()):
                try:
                    chunk = os.read(key.fd, 65536)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(key.fileobj)
    selector.close()
    return bytes(stdout), bytes(stderr), observed, termination


def execute_sandboxed(request: SandboxRequest) -> SandboxResult:
    """Execute untrusted code or return a fail-closed policy result."""

    started = time.monotonic()
    capabilities = detect_capabilities()
    strict_unavailable = (
        request.require_network_isolation
        and capabilities.network_isolation is not IsolationStatus.ENFORCED
    ) or (
        request.require_filesystem_isolation
        and (
            capabilities.filesystem_read_isolation is not IsolationStatus.ENFORCED
            or capabilities.filesystem_write_isolation is not IsolationStatus.ENFORCED
        )
    )
    if strict_unavailable:
        return _failure_result(
            request,
            capabilities,
            SandboxTermination.POLICY_UNAVAILABLE,
            "required kernel isolation is unavailable; generated code was not executed",
            time.monotonic() - started,
        )
    try:
        working, read_only, output = _canonical_paths(request)
        environment = _sanitized_environment(request)
        command = _build_command(request, capabilities, working, read_only, output)
    except (OSError, ValueError) as exc:
        return _failure_result(
            request,
            capabilities,
            SandboxTermination.SETUP_ERROR,
            str(exc),
            time.monotonic() - started,
        )
    try:
        process = subprocess.Popen(
            command,
            cwd=working,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            preexec_fn=lambda: _limit_resources(request),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return _failure_result(
            request,
            capabilities,
            SandboxTermination.SETUP_ERROR,
            f"sandbox process failed to start: {exc}",
            time.monotonic() - started,
        )
    deadline = started + request.limits.wall_time_seconds
    stdout, stderr, observed, forced = _bounded_output(
        process, request.limits.output_bytes, deadline
    )
    try:
        return_code = process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        _kill_process_group(process)
        return_code = process.wait(timeout=1.0)
    if forced is not None:
        termination = forced
    elif return_code == 0:
        termination = SandboxTermination.SUCCESS
    elif return_code < 0:
        termination = SandboxTermination.SIGNAL
    else:
        termination = SandboxTermination.NONZERO_EXIT
    artifact_violation = _validate_artifact_tree(
        output, request.limits.artifact_bytes, request.limits.artifact_entries
    )
    if artifact_violation is not None:
        termination = SandboxTermination.SANDBOX_VIOLATION
        stderr = stderr + (b"\n" if stderr else b"") + artifact_violation.encode("utf-8")
    return SandboxResult(
        termination=termination,
        return_code=return_code,
        stdout=stdout.decode("utf-8", errors="replace"),
        stderr=stderr.decode("utf-8", errors="replace"),
        duration_seconds=max(0.0, time.monotonic() - started),
        output_bytes=observed,
        capabilities=capabilities,
        sanitized_environment_names=tuple(sorted(environment)),
        process_group_cleaned=process.poll() is not None,
        artifact_output_directory=output,
    )


def minimum_memory_for_python() -> int:
    """Conservative helper for callers constructing Python sandbox limits."""

    return max(512 * 1024 * 1024, math.ceil(1.5 * 1024 * 1024 * 1024))
