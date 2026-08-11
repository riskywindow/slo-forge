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
_PIPE_DRAIN_GRACE_SECONDS = 0.25


def interpreter_read_roots() -> tuple[Path, ...]:
    """Return canonical interpreter roots safe for strict sandbox allowlists.

    macOS commonly spells temporary virtual environments below ``/var`` even
    though that component is a symlink to ``/private/var``. Callers must not
    pass that unresolved spelling into the symlink-rejecting trust boundary.
    """

    return tuple(
        dict.fromkeys(
            (
                Path(sys.prefix).resolve(strict=True),
                Path(sys.base_prefix).resolve(strict=True),
            )
        )
    )


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
                "memory is bounded by a process-group RSS watchdog; use an outer container for kernel-enforced memory isolation",
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
            process_limit=IsolationStatus.UNAVAILABLE,
            output_limit=IsolationStatus.ENFORCED,
            child_cleanup=IsolationStatus.ENFORCED,
            limitations=(
                "bubblewrap availability does not guarantee user namespaces are enabled",
                "process-count isolation requires an outer cgroup because RLIMIT_NPROC is user-scoped",
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
        child_cleanup=IsolationStatus.UNAVAILABLE,
        limitations=(
            "no supported kernel sandbox backend was found; strict execution fails closed",
            "a non-strict caller cannot prevent a generated process from escaping its process group",
        ),
    )


def _is_credential_name(name: str) -> bool:
    upper = name.upper()
    return any(marker in upper for marker in _BLOCKED_ENVIRONMENT_MARKERS)


def _sanitized_environment(request: SandboxRequest, output: Path) -> dict[str, str]:
    environment = {
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TZ": "UTC",
        "PYTHONHASHSEED": str(request.seed),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "SLOFORGE_GENESIS_SANDBOX_LAUNCH": "sandbox-executor-v1",
        "HOME": str(output / "home"),
        "TMPDIR": str(output / "tmp"),
    }
    for item in request.environment:
        if _is_credential_name(item.name):
            raise ValueError(f"credential-like environment variable is forbidden: {item.name}")
        if item.name not in _ALLOWED_CALLER_ENVIRONMENT:
            raise ValueError(f"environment variable is not allowlisted: {item.name}")
        environment[item.name] = item.value
    return environment


def _reject_symlink_components(path: Path, *, label: str) -> None:
    """Reject existing symlinks in a caller-controlled path spelling."""

    absolute = path if path.is_absolute() else Path.cwd() / path
    cursor = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        cursor = cursor / component
        if cursor.is_symlink():
            raise ValueError(f"{label} must not contain symlink components: {cursor}")
        if not cursor.exists():
            break


def _canonical_paths(request: SandboxRequest) -> tuple[Path, tuple[Path, ...], Path]:
    _reject_symlink_components(request.working_directory, label="working_directory")
    working = request.working_directory.resolve(strict=True)
    if not working.is_dir():
        raise ValueError("working_directory must be a directory")
    read_only: list[Path] = []
    for item in request.read_only_paths:
        _reject_symlink_components(item, label="read-only input")
        read_only.append(item.resolve(strict=True))
    if not any(working == item or working.is_relative_to(item) for item in read_only):
        raise ValueError("working_directory must be within an explicit read-only input")

    requested_output = request.artifact_output_directory
    _reject_symlink_components(requested_output, label="artifact output directory")
    output = requested_output.resolve(strict=False)
    if any(output == item or output.is_relative_to(item) for item in read_only):
        raise ValueError("artifact output must not be nested in a read-only input")
    if requested_output.exists():
        if not requested_output.is_dir():
            raise ValueError("artifact output path must be a directory")
        if any(requested_output.iterdir()):
            raise ValueError("artifact output directory must be empty")
    else:
        requested_output.mkdir(mode=0o700, parents=True)
    _reject_symlink_components(requested_output, label="artifact output directory")
    canonical_output = requested_output.resolve(strict=True)
    if canonical_output != output:
        raise ValueError("artifact output path changed during sandbox setup")
    output = canonical_output
    if not output.is_dir() or any(output.iterdir()):
        raise ValueError("artifact output directory changed during sandbox setup")
    (output / "tmp").mkdir(mode=0o700)
    (output / "home").mkdir(mode=0o700)
    return working, tuple(dict.fromkeys(read_only)), output


def _sandbox_string(path: Path) -> str:
    return json.dumps(str(path))


def _macos_firmlink_alias(path: Path) -> Path | None:
    spelling = str(path)
    for public in ("/var", "/tmp"):
        private = f"/private{public}"
        if spelling == public or spelling.startswith(f"{public}/"):
            return Path(f"/private{spelling}")
        if spelling == private or spelling.startswith(f"{private}/"):
            return Path(spelling.removeprefix("/private"))
    return None


def _macos_profile(read_only: tuple[Path, ...], output: Path, executable: Path) -> str:
    runtime_roots = list(
        dict.fromkeys(
            path
            for path in (
                Path("/System"),
                Path("/Library/Frameworks"),
                Path("/Library/Preferences"),
                Path("/usr/lib"),
                Path("/usr/share"),
                Path("/private/etc/localtime"),
                Path("/private/var/db"),
                Path("/private/var/select"),
                *interpreter_read_roots(),
                executable.absolute(),
                executable.absolute().parent,
                executable.resolve(),
                *read_only,
            )
            if path.exists()
        )
    )
    # macOS presents /var and /tmp through /private firmlinks. Sandbox filters
    # may receive either spelling even when pathlib.resolve() preserves the
    # public spelling, so both must be explicit.
    for path in tuple(runtime_roots):
        alias = _macos_firmlink_alias(path)
        if alias is not None:
            runtime_roots.append(alias)
    runtime_roots = list(dict.fromkeys(runtime_roots))

    read_exemptions = [
        *(f"    (require-not (subpath {_sandbox_string(path)}))" for path in runtime_roots),
    ]
    ancestors = {Path("/")}
    for path in runtime_roots:
        ancestors.update(path.parents)
    ancestors.update({Path("/dev/null"), Path("/dev/urandom")})
    read_exemptions.extend(
        f"    (require-not (literal {_sandbox_string(path)}))"
        for path in sorted(ancestors, key=str)
    )

    write_roots = [output]
    output_alias = _macos_firmlink_alias(output)
    if output_alias is not None:
        write_roots.append(output_alias)
    write_exemptions = [
        *(f"    (require-not (subpath {_sandbox_string(path)}))" for path in write_roots),
        '    (require-not (literal "/dev/null"))',
    ]
    return "\n".join(
        (
            "(version 1)",
            "(allow default)",
            "(deny network*)",
            "(deny process-fork)",
            "(deny file-write*",
            "  (require-all",
            *write_exemptions,
            "  )",
            ")",
            "(deny file-read*",
            "  (require-all",
            *read_exemptions,
            "  )",
            ")",
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
    runtime_roots = [
        *(Path(item) for item in ("/usr", "/bin", "/lib", "/lib64", "/etc")),
        *interpreter_read_roots(),
    ]
    for path in dict.fromkeys((*runtime_roots, *read_only)):
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
    # RLIMIT_AS/DATA on macOS counts large interpreter mappings and can abort a
    # small Python workload before its first instruction. The capability report
    # therefore declares memory isolation unavailable on macOS; callers that
    # require it must add an outer VM/container instead of receiving a false
    # assurance. Linux retains the address-space limit.
    if platform.system() != "Darwin":
        apply_best_effort_limit(resource.RLIMIT_AS, limits.memory_bytes)
    # RLIMIT_NPROC is per host UID, not per sandbox. Lowering it in the wrapper
    # can prevent bubblewrap from creating its namespace whenever the calling
    # UID already owns more processes than the request's bound. macOS denies
    # process-fork in its sandbox profile; Linux reports process-count isolation
    # unavailable and requires an outer cgroup for that boundary.
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


def _process_group_rss_bytes(process_group: int) -> int | None:
    """Read aggregate RSS without trusting or invoking the generated process."""

    ps = Path("/bin/ps")
    if not ps.is_file():
        return None
    try:
        completed = subprocess.run(
            (str(ps), "-axo", "pgid=,rss="),
            check=False,
            capture_output=True,
            text=True,
            timeout=0.25,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    total_kib = 0
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) != 2:
            continue
        try:
            pgid, rss_kib = (int(item) for item in fields)
        except ValueError:
            continue
        if pgid == process_group:
            total_kib += rss_kib
    return total_kib * 1024


def _bounded_output(
    process: subprocess.Popen[bytes],
    maximum_bytes: int,
    deadline: float,
    *,
    memory_bytes: int,
    artifact_output: Path,
    artifact_bytes: int,
    artifact_entries: int,
) -> tuple[bytes, bytes, int, SandboxTermination | None, str | None]:
    selector = selectors.DefaultSelector()
    assert process.stdout is not None
    assert process.stderr is not None
    streams = {process.stdout.fileno(): "stdout", process.stderr.fileno(): "stderr"}
    pipe_objects = {
        process.stdout.fileno(): process.stdout,
        process.stderr.fileno(): process.stderr,
    }
    for file_descriptor, label in streams.items():
        os.set_blocking(file_descriptor, False)
        selector.register(file_descriptor, selectors.EVENT_READ, label)
    stdout = bytearray()
    stderr = bytearray()
    observed = 0
    termination: SandboxTermination | None = None
    violation: str | None = None
    next_resource_check = time.monotonic()
    forced_at: float | None = None
    while selector.get_map():
        now = time.monotonic()
        remaining = deadline - now
        if remaining <= 0 and termination is None:
            termination = SandboxTermination.TIMEOUT
            _kill_process_group(process)
        if termination is not None and forced_at is None:
            forced_at = now
        if termination is None and now >= next_resource_check:
            next_resource_check = now + 0.05
            rss_bytes = _process_group_rss_bytes(process.pid)
            if rss_bytes is not None and rss_bytes > memory_bytes:
                termination = SandboxTermination.SANDBOX_VIOLATION
                violation = "process group exceeds the configured memory limit"
                _kill_process_group(process)
            if termination is None:
                artifact_violation = _validate_artifact_tree(
                    artifact_output, artifact_bytes, artifact_entries
                )
                if artifact_violation is not None:
                    termination = SandboxTermination.SANDBOX_VIOLATION
                    violation = artifact_violation
                    _kill_process_group(process)
        if termination is not None and forced_at is None:
            forced_at = now
        if forced_at is not None and now - forced_at >= _PIPE_DRAIN_GRACE_SECONDS:
            for key in list(selector.get_map().values()):
                selector.unregister(key.fd)
                pipe_objects[key.fd].close()
            break
        drain_remaining = (
            _PIPE_DRAIN_GRACE_SECONDS - (now - forced_at) if forced_at is not None else remaining
        )
        events = selector.select(timeout=max(0.0, min(0.05, drain_remaining)))
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
    return bytes(stdout), bytes(stderr), observed, termination, violation


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
        environment = _sanitized_environment(request, output)
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
    stdout, stderr, observed, forced, runtime_violation = _bounded_output(
        process,
        request.limits.output_bytes,
        deadline,
        memory_bytes=request.limits.memory_bytes,
        artifact_output=output,
        artifact_bytes=request.limits.artifact_bytes,
        artifact_entries=request.limits.artifact_entries,
    )
    try:
        return_code = process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        _kill_process_group(process)
        return_code = process.wait(timeout=1.0)
    # Reap same-session children even when the generated parent exited cleanly.
    # Kernel sandboxes additionally prevent detached descendants; the NONE
    # backend reports that guarantee as unavailable.
    _kill_process_group(process)
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
        runtime_violation = runtime_violation or artifact_violation
    if runtime_violation is not None:
        stderr = stderr + (b"\n" if stderr else b"") + runtime_violation.encode("utf-8")
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
