from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sloforge.util import percentile

MAX_TRACE_BYTES = 512 * 1024 * 1024
MAX_TRACE_LINE_BYTES = 1024 * 1024
MAX_TRACE_RECORDS = 1_000_000


class TraceRequest(BaseModel):
    """One open-loop request arrival in the canonical JSONL trace format."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1, max_length=256)
    arrival_ms: float = Field(ge=0)
    prompt_tokens: int = Field(ge=1)
    output_tokens: int = Field(ge=1)
    priority: Literal[0, 1, 2, 3] = 1
    request_class: str = Field(default="interactive", min_length=1, max_length=128)
    deadline_ms: float | None = Field(default=None, gt=0)
    adapter_id: str | None = None
    prefix_group: str | None = None
    cancelled_at_ms: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def cancellation_follows_arrival(self) -> TraceRequest:
        if self.cancelled_at_ms is not None and self.cancelled_at_ms < self.arrival_ms:
            raise ValueError("cancelled_at_ms cannot precede arrival_ms")
        return self


class TraceSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_count: int
    duration_ms: float
    mean_arrival_rate_rps: float
    peak_one_second_rps: int
    prompt_p50: float
    prompt_p95: float
    output_p50: float
    output_p95: float
    priorities: dict[str, int]
    request_classes: dict[str, int]


def load_trace(path: Path) -> list[TraceRequest]:
    if path.stat().st_size > MAX_TRACE_BYTES:
        raise ValueError(f"{path}: trace exceeds {MAX_TRACE_BYTES} byte safety limit")
    requests: list[TraceRequest] = []
    with path.open("rb") as handle:
        line_number = 0
        while raw_line := handle.readline(MAX_TRACE_LINE_BYTES + 1):
            line_number += 1
            if len(raw_line) > MAX_TRACE_LINE_BYTES:
                raise ValueError(
                    f"{path}:{line_number}: trace record exceeds {MAX_TRACE_LINE_BYTES} bytes"
                )
            if len(requests) >= MAX_TRACE_RECORDS:
                raise ValueError(f"{path}: trace exceeds {MAX_TRACE_RECORDS} record safety limit")
            try:
                line = raw_line.decode("utf-8").strip()
            except UnicodeDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: trace record is not UTF-8") from exc
            if not line:
                continue
            try:
                requests.append(TraceRequest.model_validate_json(line))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid trace record: {exc}") from exc
    if not requests:
        raise ValueError(f"{path}: trace contains no requests")
    return requests


def validate_trace(requests: list[TraceRequest]) -> TraceSummary:
    if not requests:
        raise ValueError("trace contains no requests")
    seen: set[str] = set()
    previous = -math.inf
    per_second: Counter[int] = Counter()
    for request in requests:
        if request.request_id in seen:
            raise ValueError(f"duplicate request_id: {request.request_id}")
        if request.arrival_ms < previous:
            raise ValueError(
                f"trace arrivals must be nondecreasing: {request.request_id} arrived out of order"
            )
        seen.add(request.request_id)
        previous = request.arrival_ms
        per_second[int(request.arrival_ms // 1000)] += 1
    duration_ms = max(requests[-1].arrival_ms - requests[0].arrival_ms, 1.0)
    prompts = [float(item.prompt_tokens) for item in requests]
    outputs = [float(item.output_tokens) for item in requests]
    return TraceSummary(
        request_count=len(requests),
        duration_ms=duration_ms,
        mean_arrival_rate_rps=len(requests) / (duration_ms / 1000.0),
        peak_one_second_rps=max(per_second.values()),
        prompt_p50=percentile(prompts, 0.50),
        prompt_p95=percentile(prompts, 0.95),
        output_p50=percentile(outputs, 0.50),
        output_p95=percentile(outputs, 0.95),
        priorities={
            str(key): value for key, value in sorted(Counter(r.priority for r in requests).items())
        },
        request_classes=dict(sorted(Counter(r.request_class for r in requests).items())),
    )


def generate_bursty_trace(*, seed: int, count: int = 180) -> list[TraceRequest]:
    """Generate a deterministic mixed workload with burst and quiet phases."""
    if count < 8:
        raise ValueError("count must be at least eight to represent a mixed workload")
    rng = random.Random(seed)
    requests: list[TraceRequest] = []
    arrival_ms = 0.0
    for index in range(count):
        phase = (index // max(1, count // 6)) % 3
        rate_rps = (5.0, 23.0, 9.0)[phase]
        arrival_ms += rng.expovariate(rate_rps) * 1000.0
        is_long = rng.random() < 0.28
        if is_long:
            prompt = max(256, int(rng.lognormvariate(math.log(1150), 0.42)))
            output = max(48, int(rng.lognormvariate(math.log(180), 0.35)))
            request_class = "long-context"
            priority: Literal[0, 1, 2, 3] = 2
            deadline = 6500.0
        else:
            prompt = max(16, int(rng.lognormvariate(math.log(105), 0.50)))
            output = max(12, int(rng.lognormvariate(math.log(55), 0.45)))
            request_class = "interactive"
            priority = 0 if rng.random() < 0.25 else 1
            deadline = 1800.0
        requests.append(
            TraceRequest(
                request_id=f"req-{seed:04d}-{index:05d}",
                arrival_ms=round(arrival_ms, 3),
                prompt_tokens=min(prompt, 8192),
                output_tokens=min(output, 1024),
                priority=priority,
                request_class=request_class,
                deadline_ms=deadline,
                adapter_id="code" if index % 7 == 0 else None,
                prefix_group="repository-context" if is_long and index % 3 == 0 else None,
            )
        )
    return requests


def write_trace(path: Path, requests: list[TraceRequest]) -> None:
    validate_trace(requests)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(request.model_dump_json(exclude_none=True) + "\n" for request in requests),
        encoding="utf-8",
    )
