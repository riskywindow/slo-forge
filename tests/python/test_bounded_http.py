from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from sloforge.cli.common import load_yaml_or_json
from sloforge.profiler.http_mock import _bounded_error_text as bounded_sync_error
from sloforge.runtime.gateway_replay import (
    MAX_REPLAY_REQUESTS,
    replay_gateway,
)
from sloforge.runtime.gateway_replay import (
    _bounded_error_text as bounded_async_error,
)
from sloforge.trace import TraceRequest


def test_profiler_error_body_is_bounded() -> None:
    rendered = bounded_sync_error(httpx.Response(500, content=b"sensitive" * 1_024))
    assert rendered.endswith("...[truncated]")
    assert len(rendered.encode()) <= 512 + len("...[truncated]")


@pytest.mark.asyncio
async def test_gateway_replay_error_body_is_bounded() -> None:
    rendered = await bounded_async_error(httpx.Response(500, content=b"sensitive" * 1_024))
    assert rendered.endswith("...[truncated]")
    assert len(rendered.encode()) <= 512 + len("...[truncated]")


@pytest.mark.asyncio
async def test_gateway_replay_rejects_empty_and_unbounded_workloads() -> None:
    with pytest.raises(ValueError, match="at least one request"):
        await replay_gateway(
            gateway_url="http://127.0.0.1:1",
            backend_urls=["http://127.0.0.1:2"],
            trace=[],
            time_scale=1.0,
            output_path=Path("unused.json"),
        )
    request = TraceRequest(
        request_id="bounded",
        arrival_ms=0,
        prompt_tokens=1,
        output_tokens=1,
    )
    with pytest.raises(ValueError, match=str(MAX_REPLAY_REQUESTS)):
        await replay_gateway(
            gateway_url="http://127.0.0.1:1",
            backend_urls=["http://127.0.0.1:2"],
            trace=[request] * (MAX_REPLAY_REQUESTS + 1),
            time_scale=1.0,
            output_path=Path("unused.json"),
        )


def test_control_document_loader_is_bounded_and_rejects_yaml_references(
    tmp_path: Path,
) -> None:
    alias = tmp_path / "counterfactual.yaml"
    alias.write_text("scenario_id: &id repair\ncopy: *id\n", encoding="utf-8")
    with pytest.raises(ValueError, match="anchors or aliases"):
        load_yaml_or_json(alias)
    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b" " * (4 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match="exceeds 4 MiB"):
        load_yaml_or_json(oversized)
