from __future__ import annotations

import codecs
import importlib
import importlib.metadata
import json
import os
import shutil
import signal
import subprocess
import threading
import time
from collections import deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from pathlib import Path
from types import TracebackType
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field

from sloforge.util import bounded_run, utc_now


class SSEProtocolError(RuntimeError):
    """The backend returned an invalid or unsafe SSE stream."""


class SSEEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: str
    event: str | None = None
    event_id: str | None = None
    retry_ms: int | None = Field(default=None, ge=0)


class OpenAIStreamTiming(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ttft_ms: float = Field(ge=0)
    e2e_ms: float = Field(ge=0)
    output_tokens: int = Field(ge=1)
    prompt_tokens: int | None = Field(default=None, ge=1)
    token_timestamps_ms: tuple[float, ...]
    event_count: int = Field(ge=1)
    response_bytes: int = Field(ge=1)
    finish_reason: str | None = None


class GpuEnvironment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    captured_at: str
    requested_device: Literal["cuda"] = "cuda"
    cuda_visible_devices: str | None
    torch_version: str
    torch_cuda_version: str | None
    cudnn_version: int | None
    device_count: int = Field(ge=1)
    selected_device_index: int = Field(ge=0)
    selected_device_name: str
    selected_device_capability: tuple[int, int]
    selected_device_total_memory_bytes: int = Field(gt=0)
    driver_version: str | None = None
    packages: dict[str, str]


class _SSEDecoder:
    def __init__(self, *, max_event_bytes: int, max_stream_bytes: int) -> None:
        if max_event_bytes < 1 or max_stream_bytes < max_event_bytes:
            raise ValueError("SSE limits require 0 < max_event_bytes <= max_stream_bytes")
        self._utf8 = codecs.getincrementaldecoder("utf-8")(errors="strict")
        self._text = ""
        self._data_lines: list[str] = []
        self._event: str | None = None
        self._event_id: str | None = None
        self._retry_ms: int | None = None
        self._event_bytes = 0
        self._stream_bytes = 0
        self._max_event_bytes = max_event_bytes
        self._max_stream_bytes = max_stream_bytes

    @property
    def stream_bytes(self) -> int:
        return self._stream_bytes

    def feed(self, chunk: bytes) -> list[SSEEvent]:
        self._stream_bytes += len(chunk)
        if self._stream_bytes > self._max_stream_bytes:
            raise SSEProtocolError(
                f"SSE stream exceeded {self._max_stream_bytes} byte response limit"
            )
        try:
            self._text += self._utf8.decode(chunk, final=False)
        except UnicodeDecodeError as exc:
            raise SSEProtocolError("SSE stream is not valid UTF-8") from exc
        events: list[SSEEvent] = []
        while True:
            newline = self._text.find("\n")
            if newline < 0:
                break
            line = self._text[:newline]
            self._text = self._text[newline + 1 :]
            if line.endswith("\r"):
                line = line[:-1]
            event = self._consume_line(line)
            if event is not None:
                events.append(event)
        return events

    def finish(self) -> list[SSEEvent]:
        try:
            self._text += self._utf8.decode(b"", final=True)
        except UnicodeDecodeError as exc:
            raise SSEProtocolError("SSE stream ended in an incomplete UTF-8 sequence") from exc
        events: list[SSEEvent] = []
        if self._text:
            event = self._consume_line(self._text.removesuffix("\r"))
            self._text = ""
            if event is not None:
                events.append(event)
        event = self._dispatch()
        if event is not None:
            events.append(event)
        return events

    def _consume_line(self, line: str) -> SSEEvent | None:
        self._event_bytes += len(line.encode("utf-8")) + 1
        if self._event_bytes > self._max_event_bytes:
            raise SSEProtocolError(f"SSE event exceeded {self._max_event_bytes} byte limit")
        if line == "":
            return self._dispatch()
        if line.startswith(":"):
            return None
        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if field == "data":
            self._data_lines.append(value)
        elif field == "event":
            self._event = value
        elif field == "id" and "\x00" not in value:
            self._event_id = value
        elif field == "retry" and value.isdecimal():
            self._retry_ms = int(value)
        return None

    def _dispatch(self) -> SSEEvent | None:
        if not self._data_lines:
            self._reset_event()
            return None
        result = SSEEvent(
            data="\n".join(self._data_lines),
            event=self._event,
            event_id=self._event_id,
            retry_ms=self._retry_ms,
        )
        self._reset_event()
        return result

    def _reset_event(self) -> None:
        self._data_lines = []
        self._event = None
        self._retry_ms = None
        self._event_bytes = 0


def iter_sse_events(
    chunks: Iterable[bytes], *, max_event_bytes: int = 1 << 20, max_stream_bytes: int = 64 << 20
) -> Iterator[SSEEvent]:
    """Incrementally parse SSE without assuming network chunks align to lines."""
    decoder = _SSEDecoder(max_event_bytes=max_event_bytes, max_stream_bytes=max_stream_bytes)
    for chunk in chunks:
        if chunk:
            yield from decoder.feed(chunk)
    yield from decoder.finish()


def _choice_text(choice: object) -> str:
    if not isinstance(choice, dict):
        return ""
    text = choice.get("text")
    if isinstance(text, str):
        return text
    delta = choice.get("delta")
    if isinstance(delta, dict):
        content = delta.get("content")
        return content if isinstance(content, str) else ""
    return ""


def stream_openai_completion(
    *,
    url: str,
    payload: Mapping[str, object],
    timeout_s: float,
    max_response_bytes: int = 64 << 20,
    max_events: int = 131_072,
    require_done: bool = True,
) -> OpenAIStreamTiming:
    """Measure one OpenAI-compatible SSE response without retrying partial output."""
    if timeout_s <= 0:
        raise ValueError("stream timeout must be positive")
    if max_events < 1:
        raise ValueError("max_events must be positive")
    if max_response_bytes < 1:
        raise ValueError("max_response_bytes must be positive")
    body = dict(payload)
    if body.get("stream") is not True:
        raise ValueError("SSE timing requires payload stream=true")
    timeout = httpx.Timeout(timeout_s, connect=min(timeout_s, 10.0))
    started_ns = time.perf_counter_ns()
    token_timestamps: list[float] = []
    completion_tokens: int | None = None
    prompt_tokens: int | None = None
    finish_reason: str | None = None
    done = False
    event_count = 0
    response_bytes = 0
    with (
        httpx.Client(timeout=timeout, follow_redirects=False) as client,
        client.stream("POST", url, json=body) as response,
    ):
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").partition(";")[0].strip().lower()
        if content_type != "text/event-stream":
            raise SSEProtocolError("backend response content-type is not text/event-stream")

        def counted_chunks() -> Iterator[bytes]:
            nonlocal response_bytes
            for chunk in response.iter_bytes():
                response_bytes += len(chunk)
                yield chunk

        for event in iter_sse_events(
            counted_chunks(),
            max_event_bytes=min(1 << 20, max_response_bytes),
            max_stream_bytes=max_response_bytes,
        ):
            event_count += 1
            if event_count > max_events:
                raise SSEProtocolError(f"SSE stream exceeded {max_events} event limit")
            if event.data.strip() == "[DONE]":
                done = True
                break
            try:
                document = json.loads(event.data)
            except json.JSONDecodeError as exc:
                raise SSEProtocolError("SSE data field is not valid JSON") from exc
            if not isinstance(document, dict):
                raise SSEProtocolError("SSE JSON event must be an object")
            error = document.get("error")
            if error is not None:
                raise SSEProtocolError("backend emitted an error event")
            choices = document.get("choices", [])
            if not isinstance(choices, list):
                raise SSEProtocolError("SSE choices field must be a list")
            emitted_text = "".join(_choice_text(choice) for choice in choices)
            for choice in choices:
                if isinstance(choice, dict) and choice.get("finish_reason") is not None:
                    finish_reason = str(choice["finish_reason"])
            if emitted_text:
                token_timestamps.append((time.perf_counter_ns() - started_ns) / 1e6)
            usage = document.get("usage")
            if isinstance(usage, dict):
                if isinstance(usage.get("completion_tokens"), int) and not isinstance(
                    usage.get("completion_tokens"), bool
                ):
                    completion_tokens = usage["completion_tokens"]
                if isinstance(usage.get("prompt_tokens"), int) and not isinstance(
                    usage.get("prompt_tokens"), bool
                ):
                    prompt_tokens = usage["prompt_tokens"]
    if require_done and not done:
        raise SSEProtocolError("SSE stream ended before the [DONE] sentinel")
    if not token_timestamps:
        raise SSEProtocolError("SSE stream completed without emitting output text")
    elapsed_ms = (time.perf_counter_ns() - started_ns) / 1e6
    output_tokens = completion_tokens if completion_tokens is not None else len(token_timestamps)
    if output_tokens < 1:
        raise SSEProtocolError("backend reported a non-positive completion token count")
    return OpenAIStreamTiming(
        ttft_ms=token_timestamps[0],
        e2e_ms=max(elapsed_ms, token_timestamps[-1]),
        output_tokens=output_tokens,
        prompt_tokens=prompt_tokens,
        token_timestamps_ms=tuple(token_timestamps),
        event_count=event_count,
        response_bytes=response_bytes,
        finish_reason=finish_reason,
    )


class ManagedEngineServer:
    """Own an engine process group and retain only a bounded tail of its output."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        cwd: Path | None = None,
        max_log_bytes: int = 1 << 20,
        shutdown_timeout_s: float = 10.0,
    ) -> None:
        if not command or any("\x00" in part for part in command):
            raise ValueError("engine command must be non-empty and contain no NUL bytes")
        if max_log_bytes < 1024:
            raise ValueError("max_log_bytes must be at least 1024")
        if shutdown_timeout_s <= 0:
            raise ValueError("shutdown_timeout_s must be positive")
        self.command = tuple(command)
        self.env = dict(env) if env is not None else None
        self.cwd = cwd
        self.max_log_bytes = max_log_bytes
        self.shutdown_timeout_s = shutdown_timeout_s
        self._process: subprocess.Popen[bytes] | None = None
        self._reader: threading.Thread | None = None
        self._chunks: deque[bytes] = deque()
        self._log_size = 0
        self._lock = threading.Lock()
        self._stopped = False

    @property
    def process(self) -> subprocess.Popen[bytes]:
        if self._process is None:
            raise RuntimeError("engine server has not been started")
        return self._process

    def start(self) -> ManagedEngineServer:
        if self._process is not None or self._stopped:
            raise RuntimeError("engine server can only be started once")
        kwargs: dict[str, Any] = {"start_new_session": os.name == "posix"}
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        self._process = subprocess.Popen(
            list(self.command),
            cwd=self.cwd,
            env=self.env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            close_fds=True,
            **kwargs,
        )
        self._reader = threading.Thread(target=self._drain_output, daemon=True)
        self._reader.start()
        return self

    def _drain_output(self) -> None:
        stream = self.process.stdout
        if stream is None:
            return
        buffered_stream: Any = stream
        while True:
            chunk = buffered_stream.read1(4096)
            if not chunk:
                return
            with self._lock:
                if len(chunk) >= self.max_log_bytes:
                    self._chunks.clear()
                    retained = chunk[-self.max_log_bytes :]
                    self._chunks.append(retained)
                    self._log_size = len(retained)
                    continue
                self._chunks.append(chunk)
                self._log_size += len(chunk)
                while self._log_size > self.max_log_bytes and self._chunks:
                    removed = self._chunks.popleft()
                    self._log_size -= len(removed)

    def log_tail(self) -> str:
        with self._lock:
            content = b"".join(self._chunks)
        return content[-self.max_log_bytes :].decode("utf-8", errors="replace")

    def wait_ready(
        self,
        *,
        base_url: str,
        timeout_s: float,
        paths: Sequence[str] = ("/health", "/v1/models"),
        poll_interval_s: float = 0.1,
    ) -> str:
        if timeout_s <= 0 or poll_interval_s <= 0:
            raise ValueError("readiness timeout and poll interval must be positive")
        if not paths:
            raise ValueError("at least one readiness path is required")
        deadline = time.monotonic() + timeout_s
        last_detail = "no readiness response"
        while True:
            return_code = self.process.poll()
            if return_code is not None:
                raise RuntimeError(f"engine server exited before readiness with code {return_code}")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"engine server did not become ready within {timeout_s:.3f}s ({last_detail})"
                )
            per_request_timeout = min(1.0, remaining)
            for path in paths:
                try:
                    with httpx.stream(
                        "GET",
                        f"{base_url.rstrip('/')}/{path.lstrip('/')}",
                        timeout=per_request_timeout,
                        follow_redirects=False,
                    ) as response:
                        if 200 <= response.status_code < 300:
                            return path
                        last_detail = f"{path} returned HTTP {response.status_code}"
                except httpx.HTTPError as exc:
                    last_detail = type(exc).__name__
            time.sleep(min(poll_interval_s, max(0.0, deadline - time.monotonic())))

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        process = self._process
        if process is None:
            return
        if os.name == "posix":
            # The launcher may have exited after daemonizing children. Signal the process group
            # even when the leader is already gone so those children do not become orphans.
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
        elif process.poll() is None:
            with suppress(ProcessLookupError):
                process.terminate()
        if process.poll() is None:
            try:
                process.wait(timeout=self.shutdown_timeout_s)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                process.wait(timeout=self.shutdown_timeout_s)
        if self._reader is not None:
            self._reader.join(timeout=1.0)
            if self._reader.is_alive() and os.name == "posix":
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                self._reader.join(timeout=1.0)
        if process.stdout is not None:
            process.stdout.close()

    def __enter__(self) -> ManagedEngineServer:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        self.stop()


def ensure_cuda_requested(*, device: str, index: int = 0) -> Any:
    """Validate an explicit CUDA request and return torch; never choose another device."""
    if device != "cuda":
        raise RuntimeError("real GPU profiling requires device='cuda'; CPU fallback is forbidden")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.strip() in {"", "-1"}:
        raise RuntimeError("CUDA was requested but CUDA_VISIBLE_DEVICES hides all GPUs")
    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:
        raise RuntimeError("CUDA was requested but PyTorch is not installed") from exc
    if not bool(torch.cuda.is_available()):
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    count = int(torch.cuda.device_count())
    if index < 0 or index >= count:
        raise RuntimeError(f"CUDA device index {index} is unavailable; detected {count} device(s)")
    return torch


def cuda_subprocess_environment(*, device_index: int) -> dict[str, str]:
    """Pin a child to the same logical CUDA device while preserving required runtime paths."""
    if device_index < 0:
        raise ValueError("CUDA device index cannot be negative")
    environment = dict(os.environ)
    visible = environment.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        selected = str(device_index)
    else:
        identifiers = [
            identifier.strip() for identifier in visible.split(",") if identifier.strip()
        ]
        if device_index >= len(identifiers):
            raise RuntimeError(
                f"logical CUDA index {device_index} is outside CUDA_VISIBLE_DEVICES={visible!r}"
            )
        selected = identifiers[device_index]
    environment["CUDA_VISIBLE_DEVICES"] = selected
    return environment


def gpu_environment(*, device: str, index: int = 0) -> GpuEnvironment:
    torch = ensure_cuda_requested(device=device, index=index)
    properties = torch.cuda.get_device_properties(index)
    packages: dict[str, str] = {}
    for package in ("torch", "transformers", "vllm", "sglang", "triton"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    driver_version: str | None = None
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is not None:
        completed = bounded_run(
            [nvidia_smi, "--query-gpu=driver_version", "--format=csv,noheader", f"--id={index}"],
            timeout_s=5.0,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            driver_version = completed.stdout.splitlines()[0].strip()
    cudnn_value = torch.backends.cudnn.version()
    capability = torch.cuda.get_device_capability(index)
    return GpuEnvironment(
        captured_at=utc_now(),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        torch_version=str(torch.__version__),
        torch_cuda_version=torch.version.cuda,
        cudnn_version=int(cudnn_value) if cudnn_value is not None else None,
        device_count=int(torch.cuda.device_count()),
        selected_device_index=index,
        selected_device_name=str(properties.name),
        selected_device_capability=(int(capability[0]), int(capability[1])),
        selected_device_total_memory_bytes=int(properties.total_memory),
        driver_version=driver_version,
        packages=packages,
    )


@contextmanager
def torch_perfetto_trace(*, output: Path, device: str, index: int = 0) -> Iterator[Any]:
    """Capture one caller-defined PyTorch region as a Perfetto-compatible Chrome trace."""
    torch = ensure_cuda_requested(device=device, index=index)
    output.parent.mkdir(parents=True, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as profile:
        yield profile
    profile.export_chrome_trace(str(output))
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("torch.profiler did not produce a non-empty Chrome trace")


def nsight_systems_available() -> bool:
    return shutil.which("nsys") is not None


def build_nsight_systems_command(
    command: Sequence[str],
    *,
    output_prefix: Path,
    require_available: bool = True,
) -> list[str]:
    """Generate, but do not execute, a reproducible Nsight Systems capture command."""
    if not command or any("\x00" in part for part in command):
        raise ValueError("profiled command must be non-empty and contain no NUL bytes")
    executable = shutil.which("nsys")
    if executable is None:
        if require_available:
            raise RuntimeError("Nsight Systems command requested but 'nsys' is not installed")
        executable = "nsys"
    return [
        executable,
        "profile",
        "--force-overwrite=false",
        "--sample=none",
        "--stats=true",
        "--trace=cuda,nvtx,osrt",
        "--output",
        str(output_prefix),
        "--",
        *command,
    ]
