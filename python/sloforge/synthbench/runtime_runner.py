"""Trusted bounded subprocess harness for one generated-runtime benchmark request."""

from __future__ import annotations

import argparse
import json
import time
from itertools import pairwise
from pathlib import Path

from sloforge.genesis.runtime import RuntimeRequest, load_generated_runtime


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=float, required=True)
    arguments = parser.parse_args()
    if not 0.0 < arguments.timeout_seconds <= 30.0:
        raise ValueError("runtime benchmark timeout is outside its bounded domain")
    request_document = json.loads(arguments.request.read_text(encoding="utf-8"))
    if not isinstance(request_document, dict):
        raise TypeError("runtime benchmark request must be an object")
    runtime = load_generated_runtime(arguments.config, seed=arguments.seed)
    runtime.start()
    try:
        started = time.perf_counter_ns()
        handle = runtime.submit(
            RuntimeRequest(
                request_id=str(request_document["request_id"]),
                prompt_tokens=tuple(int(token) for token in request_document["prompt_tokens"]),
                maximum_new_tokens=int(request_document["maximum_new_tokens"]),
                seed=int(request_document["seed"]),
                timeout_seconds=arguments.timeout_seconds,
            )
        )
        tokens: list[int] = []
        token_times: list[int] = []
        for event in handle.events(arguments.timeout_seconds):
            if event.token_id is not None:
                tokens.append(event.token_id)
                token_times.append(time.perf_counter_ns())
            elif event.kind.value in {"error", "cancelled", "timed_out"}:
                raise RuntimeError(f"generated runtime terminated as {event.kind.value}")
        ended = time.perf_counter_ns()
    finally:
        runtime.shutdown()
    if not token_times:
        raise RuntimeError("generated runtime emitted no tokens")
    print(
        json.dumps(
            {
                "tokens": tokens,
                "latency_ns": max(1, ended - started),
                "ttft_ns": max(1, token_times[0] - started),
                "inter_token_ns": [max(1, right - left) for left, right in pairwise(token_times)],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
