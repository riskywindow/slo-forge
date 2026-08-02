from __future__ import annotations

import httpx
import pytest

from sloforge.profiler.http_mock import _bounded_error_text as bounded_sync_error
from sloforge.runtime.gateway_replay import _bounded_error_text as bounded_async_error


def test_profiler_error_body_is_bounded() -> None:
    rendered = bounded_sync_error(httpx.Response(500, content=b"sensitive" * 1_024))
    assert rendered.endswith("...[truncated]")
    assert len(rendered.encode()) <= 512 + len("...[truncated]")


@pytest.mark.asyncio
async def test_gateway_replay_error_body_is_bounded() -> None:
    rendered = await bounded_async_error(httpx.Response(500, content=b"sensitive" * 1_024))
    assert rendered.endswith("...[truncated]")
    assert len(rendered.encode()) <= 512 + len("...[truncated]")
