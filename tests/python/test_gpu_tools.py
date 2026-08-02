from __future__ import annotations

import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from sloforge.profiler.gpu_tools import (
    ManagedEngineServer,
    SSEProtocolError,
    build_nsight_systems_command,
    cuda_subprocess_environment,
    ensure_cuda_requested,
    iter_sse_events,
    stream_openai_completion,
)


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def test_sse_parser_handles_split_utf8_crlf_and_multiline_data() -> None:
    chunks = [
        b': keepalive\r\nid: 7\r\ndata: {"text":"caf',
        "é".encode()[:1],
        "é".encode()[1:] + b'"}\r\n',
        b"data: second\r\n\r\ndata: [DONE]\n\n",
    ]
    events = list(iter_sse_events(chunks))
    assert events[0].event_id == "7"
    assert events[0].data == '{"text":"café"}\nsecond'
    assert events[1].data == "[DONE]"


def test_sse_parser_enforces_event_and_stream_bounds() -> None:
    with pytest.raises(SSEProtocolError, match="event exceeded"):
        list(iter_sse_events([b"data: " + b"x" * 40 + b"\n\n"], max_event_bytes=16))
    with pytest.raises(SSEProtocolError, match="response limit"):
        list(
            iter_sse_events(
                [b"data: x\n\n", b"data: y\n\n"],
                max_event_bytes=16,
                max_stream_bytes=17,
            )
        )


class _SSEHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        del format, args

    def do_POST(self) -> None:
        content_length = int(self.headers.get("content-length", "0"))
        request = json.loads(self.rfile.read(content_length))
        assert request["stream"] is True
        chunks = [
            b'data: {"choices":[{"text":"a"}]}\n\n',
            b'data: {"choices":[{"text":"b","finish_reason":"stop"}],'
            b'"usage":{"prompt_tokens":5,"completion_tokens":2}}\n\n',
            b"data: [DONE]\n\n",
        ]
        body_length = sum(len(chunk) for chunk in chunks)
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(body_length))
        self.end_headers()
        for chunk in chunks:
            self.wfile.write(chunk)
            self.wfile.flush()


def test_openai_stream_timing_parses_usage_and_done() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SSEHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = stream_openai_completion(
            url=f"http://127.0.0.1:{server.server_port}/v1/completions",
            payload={"prompt": "hello", "stream": True},
            timeout_s=2,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert result.output_tokens == 2
    assert result.prompt_tokens == 5
    assert len(result.token_timestamps_ms) == 2
    assert result.finish_reason == "stop"
    assert result.e2e_ms >= result.ttft_ms
    assert result.response_bytes > 0


def test_managed_server_waits_for_readiness_and_stops_process_group() -> None:
    port = _free_port()
    script = """
from http.server import BaseHTTPRequestHandler, HTTPServer
import sys
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        return
    def do_GET(self):
        self.send_response(200 if self.path == '/health' else 404)
        self.end_headers()
print('readying', flush=True)
HTTPServer(('127.0.0.1', int(sys.argv[1])), Handler).serve_forever()
"""
    server = ManagedEngineServer(
        [sys.executable, "-u", "-c", script, str(port)],
        max_log_bytes=1024,
        shutdown_timeout_s=2,
    )
    with server:
        assert server.wait_ready(base_url=f"http://127.0.0.1:{port}", timeout_s=3) == "/health"
        assert "readying" in server.log_tail()
        assert server.process.poll() is None
    assert server.process.poll() is not None


def test_managed_server_bounds_log_tail() -> None:
    command = [
        sys.executable,
        "-u",
        "-c",
        "import sys,time; sys.stdout.write('x'*8192); sys.stdout.flush(); time.sleep(2)",
    ]
    server = ManagedEngineServer(command, max_log_bytes=1024, shutdown_timeout_s=1)
    with server:
        time.sleep(0.1)
        assert len(server.log_tail().encode()) <= 1024


def test_cuda_requires_explicit_device_before_import_or_fallback() -> None:
    with pytest.raises(RuntimeError, match="device='cuda'"):
        ensure_cuda_requested(device="cpu")


def test_cuda_subprocess_environment_preserves_logical_device_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-first,GPU-second")
    environment = cuda_subprocess_environment(device_index=1)
    assert environment["CUDA_VISIBLE_DEVICES"] == "GPU-second"
    with pytest.raises(RuntimeError, match="outside CUDA_VISIBLE_DEVICES"):
        cuda_subprocess_environment(device_index=2)


def test_nsight_command_generation_is_offline_and_quoted_as_argv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("sloforge.profiler.gpu_tools.shutil.which", lambda _: None)
    output = tmp_path / "capture name"
    command = build_nsight_systems_command(
        ["python", "script with spaces.py"], output_prefix=output, require_available=False
    )
    assert command[0] == "nsys"
    assert command[-2:] == ["python", "script with spaces.py"]
    assert str(output) in command
    assert not output.parent.joinpath("capture name.nsys-rep").exists()
