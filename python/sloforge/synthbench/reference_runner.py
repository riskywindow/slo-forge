"""Trusted bounded subprocess harness for one reference benchmark request."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from itertools import pairwise
from pathlib import Path

from sloforge.genesis.frontend import load_reference_package
from sloforge.genesis.runtime.adapter import ReferenceRuntimeAdapter


def _seed(base: int, *parts: object) -> int:
    payload = "\0".join((str(base), *(str(part) for part in parts))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-package", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=float, required=True)
    arguments = parser.parse_args()
    if not 0 <= arguments.model_seed < 1 << 64:
        raise ValueError("reference model seed is outside the unsigned 64-bit domain")
    if not 0.0 < arguments.timeout_seconds <= 30.0:
        raise ValueError("reference benchmark timeout is outside its bounded domain")
    request_document = json.loads(arguments.request.read_text(encoding="utf-8"))
    if not isinstance(request_document, dict) or set(request_document) != {
        "request_id",
        "prompt_tokens",
        "maximum_new_tokens",
        "seed",
    }:
        raise TypeError("reference benchmark request has an invalid oracle-free schema")
    package = load_reference_package(arguments.reference_package)
    adapter = ReferenceRuntimeAdapter(
        reference_path=package.resolve(package.manifest.reference_module),
        tokenizer_path=package.resolve(package.manifest.tokenizer_module),
        entry_points=package.manifest.entry_points,
        identity="synthbench_reference",
        seed=arguments.model_seed,
    )
    prompt_tokens = tuple(int(token) for token in request_document["prompt_tokens"])
    request_seed = int(request_document["seed"])
    maximum_new_tokens = int(request_document["maximum_new_tokens"])
    deadline = time.perf_counter() + arguments.timeout_seconds
    started = time.perf_counter_ns()
    state = adapter.allocate_state(str(request_document["request_id"]), prompt_tokens, request_seed)
    state = adapter.prefill(prompt_tokens, state, request_seed)
    previous = prompt_tokens[-1]
    tokens: list[int] = []
    token_times: list[int] = []
    for position in range(maximum_new_tokens):
        if time.perf_counter() > deadline:
            raise TimeoutError("reference benchmark request exceeded its deadline")
        result = adapter.decode_step(previous, state, position, request_seed)
        state = result.state
        token = adapter.sample(result.logits, _seed(request_seed, "sample", position))
        tokens.append(token)
        token_times.append(time.perf_counter_ns())
        previous = token
    ended = time.perf_counter_ns()
    if not token_times:
        raise RuntimeError("reference benchmark emitted no tokens")
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
