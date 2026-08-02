from __future__ import annotations

import asyncio
import json
import statistics
import time
from itertools import pairwise
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict, Field

from sloforge.trace import TraceRequest
from sloforge.util import percentile, utc_now, write_json

MAX_REPLAY_REQUESTS = 100_000
MAX_REPLAY_WORKERS = 128
MAX_REPLAY_QUEUE = 256


class GatewayRequestResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    status_code: int
    ttft_ms: float | None
    e2e_ms: float
    itl_ms: list[float]
    chunks: int = Field(ge=0)
    error: str | None = None


class InjectedGatewayFault(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fault: str
    backend_url: str
    scheduled_s: float
    applied_at: str
    response_status: int


class GatewayReplayResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "sloforge.gateway-replay/v1"
    generated_at: str
    time_scale: float
    results: list[GatewayRequestResult]
    faults: list[InjectedGatewayFault]
    summary: dict[str, float | int]


async def _bounded_error_text(response: httpx.Response, *, max_bytes: int = 512) -> str:
    body = bytearray()
    truncated = False
    async for chunk in response.aiter_bytes():
        remaining = max_bytes - len(body)
        body.extend(chunk[: max(0, remaining)])
        if len(chunk) > remaining or len(body) == max_bytes:
            truncated = True
            break
    rendered = body.decode(errors="replace")
    return f"{rendered}...[truncated]" if truncated else rendered


async def _one_request(
    client: httpx.AsyncClient,
    gateway_url: str,
    request: TraceRequest,
    *,
    delay_s: float,
) -> GatewayRequestResult:
    await asyncio.sleep(max(0.0, delay_s))
    payload = {
        "model": "sloforge-mock",
        "prompt": "replay " * min(request.prompt_tokens, 2048),
        "max_tokens": min(request.output_tokens, 32),
        "stream": True,
        "sloforge": {
            "deadline_ms": int(request.deadline_ms) if request.deadline_ms is not None else None,
            "priority": request.priority,
            "request_class": request.request_class,
        },
    }
    started = time.perf_counter_ns()
    timestamps: list[float] = []
    error: str | None = None
    status = 599
    try:
        async with client.stream("POST", f"{gateway_url}/v1/completions", json=payload) as response:
            status = response.status_code
            if status >= 400:
                error = await _bounded_error_text(response)
            else:
                async for line in response.aiter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    try:
                        json.loads(line[6:])
                    except json.JSONDecodeError as exc:
                        error = f"malformed SSE: {exc}"
                        break
                    timestamps.append((time.perf_counter_ns() - started) / 1e6)
    except httpx.HTTPError as exc:
        error = str(exc)
    e2e = (time.perf_counter_ns() - started) / 1e6
    itls = [right - left for left, right in pairwise(timestamps)]
    return GatewayRequestResult(
        request_id=request.request_id,
        status_code=status,
        ttft_ms=timestamps[0] if timestamps else None,
        e2e_ms=e2e,
        itl_ms=itls,
        chunks=len(timestamps),
        error=error,
    )


async def _inject_faults(
    *,
    backend_urls: list[str],
    duration_s: float,
    client: httpx.AsyncClient,
) -> list[InjectedGatewayFault]:
    if not backend_urls:
        raise ValueError("fault injection requires at least one backend URL")
    schedule: list[tuple[float, str, dict[str, object]]] = [
        (duration_s * 0.20, backend_urls[0], {"fault": "slowdown", "multiplier": 3.0}),
        (
            duration_s * 0.38,
            backend_urls[1 % len(backend_urls)],
            {"fault": "crash", "enabled": True},
        ),
        (
            duration_s * 0.52,
            backend_urls[1 % len(backend_urls)],
            {"fault": "crash", "enabled": False},
        ),
        (
            duration_s * 0.64,
            backend_urls[2 % len(backend_urls)],
            {"fault": "cold_start", "next_delay_ms": 350},
        ),
        (duration_s * 0.80, backend_urls[0], {"fault": "clear"}),
    ]
    injected: list[InjectedGatewayFault] = []
    previous = 0.0
    for at_s, backend, payload in schedule:
        await asyncio.sleep(max(0.0, at_s - previous))
        previous = at_s
        response = await client.post(f"{backend}/admin/fault", json=payload)
        injected.append(
            InjectedGatewayFault(
                fault=str(payload["fault"]),
                backend_url=backend,
                scheduled_s=at_s,
                applied_at=utc_now(),
                response_status=response.status_code,
            )
        )
    return injected


async def replay_gateway(
    *,
    gateway_url: str,
    backend_urls: list[str],
    trace: list[TraceRequest],
    time_scale: float,
    output_path: Path,
) -> GatewayReplayResult:
    if not 0 < time_scale <= 10:
        raise ValueError("time_scale must be in (0, 10]")
    if not trace:
        raise ValueError("gateway replay requires at least one request")
    if len(trace) > MAX_REPLAY_REQUESTS:
        raise ValueError(f"gateway replay is limited to {MAX_REPLAY_REQUESTS} requests")
    if not backend_urls or len(backend_urls) > 256:
        raise ValueError("gateway replay requires between 1 and 256 backend URLs")
    base_arrival = trace[0].arrival_ms
    delays = [(item.arrival_ms - base_arrival) / 1000.0 * time_scale for item in trace]
    duration_s = max(delays[-1] + 0.5, 1.0)
    limits = httpx.Limits(max_connections=128, max_keepalive_connections=32)
    async with httpx.AsyncClient(timeout=30.0, limits=limits) as client:
        worker_count = min(MAX_REPLAY_WORKERS, len(trace))
        queue: asyncio.Queue[tuple[int, TraceRequest, float] | None] = asyncio.Queue(
            maxsize=MAX_REPLAY_QUEUE
        )
        ordered_results: list[GatewayRequestResult | None] = [None] * len(trace)
        replay_started = time.monotonic()

        async def produce() -> None:
            for index, (item, delay) in enumerate(zip(trace, delays, strict=True)):
                await queue.put((index, item, delay))
            for _ in range(worker_count):
                await queue.put(None)

        async def consume() -> None:
            while True:
                work = await queue.get()
                try:
                    if work is None:
                        return
                    index, item, delay = work
                    remaining_delay = delay - (time.monotonic() - replay_started)
                    ordered_results[index] = await _one_request(
                        client,
                        gateway_url,
                        item,
                        delay_s=max(0.0, remaining_delay),
                    )
                finally:
                    queue.task_done()

        async with asyncio.TaskGroup() as task_group:
            task_group.create_task(produce())
            workers = [task_group.create_task(consume()) for _ in range(worker_count)]
            fault_task = task_group.create_task(
                _inject_faults(backend_urls=backend_urls, duration_s=duration_s, client=client)
            )
        if any(item is None for item in ordered_results) or any(
            not worker.done() for worker in workers
        ):
            raise RuntimeError("bounded gateway replay workers did not complete")
        results = [item for item in ordered_results if item is not None]
        faults = fault_task.result()
    successes = [
        item
        for item in results
        if item.status_code < 400 and item.error is None and item.chunks > 0
    ]
    ttfts = [item.ttft_ms for item in successes if item.ttft_ms is not None]
    itls = [sample for item in successes for sample in item.itl_ms]
    e2es = [item.e2e_ms for item in successes]
    if not successes or not ttfts or not itls:
        raise RuntimeError("gateway replay produced no successful streaming measurements")
    summary: dict[str, float | int] = {
        "request_count": len(results),
        "successful_count": len(successes),
        "failed_count": len(results) - len(successes),
        "p50_ttft_ms": percentile(ttfts, 0.50),
        "p95_ttft_ms": percentile(ttfts, 0.95),
        "p99_itl_ms": percentile(itls, 0.99),
        "p95_e2e_ms": percentile(e2es, 0.95),
        "availability": len(successes) / len(results),
        "mean_chunks": statistics.mean(item.chunks for item in successes),
    }
    replay = GatewayReplayResult(
        generated_at=utc_now(),
        time_scale=time_scale,
        results=results,
        faults=faults,
        summary=summary,
    )
    write_json(output_path, replay.model_dump())
    return replay
