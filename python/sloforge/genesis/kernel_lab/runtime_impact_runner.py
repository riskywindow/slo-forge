"""Sandboxed runner for paired generated-runtime serving impact and semantic replay."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol, cast

from sloforge.genesis.frontend import load_reference_package
from sloforge.genesis.runtime.adapter import ReferenceRuntimeAdapter


class _Event(Protocol):
    token_id: int | None


class _Handle(Protocol):
    def events(self, timeout_seconds: float) -> Iterator[_Event]: ...


class _Runtime(Protocol):
    def submit_text(
        self,
        *,
        request_id: str,
        text: str,
        maximum_new_tokens: int,
        seed: int,
        timeout_seconds: float,
        batching_eligible: bool,
    ) -> _Handle: ...


class _Application(Protocol):
    runtime: _Runtime

    def start(self) -> None: ...

    def shutdown(self) -> None: ...


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return value


def _unsigned(value: object, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < 1 << 64:
        raise ValueError(f"{field} must be an unsigned 64-bit integer")
    return value


def _positive(value: object, *, field: str, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= maximum:
        raise ValueError(f"{field} must be an integer in [1, {maximum}]")
    return value


def _finite_float(value: object, *, field: str, maximum: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not 0.0 < result <= maximum:
        raise ValueError(f"{field} must be in (0, {maximum}]")
    return result


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _load_runtime(bundle: Path, identity: str, generation_seed: int) -> _Application:
    runtime_path = bundle / "runtime.py"
    specification = importlib.util.spec_from_file_location(identity, runtime_path)
    if specification is None or specification.loader is None:
        raise RuntimeError("generated runtime module could not be loaded")
    module = importlib.util.module_from_spec(specification)
    sys.path.insert(0, str(bundle))
    try:
        specification.loader.exec_module(module)
    finally:
        sys.path.remove(str(bundle))
    application = getattr(module, "application", None)
    if not callable(application):
        raise RuntimeError("generated runtime has no callable application entry point")
    return cast("_Application", application(seed=generation_seed))


def _runtime_seed(
    generation_seed: int,
    request_seed: int,
    request_id: str,
    phase: str,
    position: int,
) -> int:
    identity = (
        f"{generation_seed}\0{request_seed}\0{request_id}\0{phase}\0{position}"
    ).encode()
    return int.from_bytes(hashlib.sha256(identity).digest()[:8], "big", signed=False)


def _run_trace(
    bundle: Path,
    alternative: str,
    generation_seed: int,
    requests: tuple[dict[str, object], ...],
    timeout_seconds: float,
    identity_index: int,
) -> tuple[int, tuple[dict[str, object], ...]]:
    application = _load_runtime(
        bundle,
        f"genesis_kernel_runtime_{alternative}_{identity_index}",
        generation_seed,
    )
    handles: list[tuple[str, _Handle]] = []
    observations: list[dict[str, object]] = []
    application.start()
    try:
        started = time.perf_counter_ns()
        # All requests are admitted before their handles are drained. This
        # exercises the generated runtime's bounded queue and batching path.
        for request in requests:
            request_id = str(request["request_id"])
            handle = application.runtime.submit_text(
                request_id=request_id,
                text=str(request["text"]),
                maximum_new_tokens=_positive(
                    request["maximum_new_tokens"], field="maximum_new_tokens", maximum=16
                ),
                seed=_unsigned(request["seed"], field="request seed"),
                timeout_seconds=timeout_seconds,
                batching_eligible=bool(request["batching_eligible"]),
            )
            handles.append((request_id, handle))
        for request_id, handle in handles:
            tokens = [
                int(event.token_id)
                for event in handle.events(timeout_seconds)
                if event.token_id is not None
            ]
            observations.append({"request_id": request_id, "token_ids": tokens})
        duration = max(1, time.perf_counter_ns() - started)
    finally:
        application.shutdown()
    return duration, tuple(observations)


def _semantic_replay(
    package_root: Path,
    alternative: str,
    generation_seed: int,
    requests: tuple[dict[str, object], ...],
) -> tuple[dict[str, object], ...]:
    package = load_reference_package(package_root)
    adapter = ReferenceRuntimeAdapter(
        reference_path=package.resolve(package.manifest.reference_module),
        tokenizer_path=package.resolve(package.manifest.tokenizer_module),
        entry_points=package.manifest.entry_points,
        identity=f"kernel_impact_semantic_{alternative}_{package.package_hash}",
        seed=generation_seed,
    )
    observations: list[dict[str, object]] = []
    for request in requests:
        request_id = str(request["request_id"])
        request_seed = _unsigned(request["seed"], field="request seed")
        prompt = adapter.tokenize(str(request["text"]))
        state = adapter.allocate_state(
            request_id,
            prompt,
            _runtime_seed(generation_seed, request_seed, request_id, "allocate", 0),
        )
        state = adapter.prefill(
            prompt,
            state,
            _runtime_seed(generation_seed, request_seed, request_id, "prefill", 0),
        )
        previous = prompt[-1]
        tokens: list[int] = []
        maximum_new_tokens = _positive(
            request["maximum_new_tokens"], field="maximum_new_tokens", maximum=16
        )
        for position in range(maximum_new_tokens):
            result = adapter.decode_step(
                previous,
                state,
                position,
                _runtime_seed(generation_seed, request_seed, request_id, "decode", position),
            )
            state = result.state
            token = adapter.sample(
                result.logits,
                _runtime_seed(generation_seed, request_seed, request_id, "sample", position),
            )
            tokens.append(token)
            previous = token
        if not isinstance(state, dict) or set(state) != {
            "kv_window",
            "recurrent_state",
            "quantized_state",
            "expert_loads",
            "speculative_state",
            "prompt_length",
        }:
            raise TypeError("HybridDecoder final state is outside its declared typed contract")
        observations.append(
            {
                "request_id": request_id,
                "token_ids": tokens,
                "final_state": state,
            }
        )
    return tuple(observations)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("measure", "replay"), required=True)
    arguments = parser.parse_args(argv)
    config = _read_object(arguments.config)
    generation_seed = _unsigned(config.get("runtime_generation_seed"), field="generation seed")
    trial_order_seed = _unsigned(config.get("trial_order_seed"), field="trial-order seed")
    repetitions = _positive(config.get("repetitions"), field="repetitions", maximum=1_000)
    if repetitions < 7:
        raise ValueError("runtime impact requires at least seven paired trials")
    warmup_count = _positive(config.get("warmup_count"), field="warmup_count", maximum=100)
    timeout_seconds = _finite_float(
        config.get("request_timeout_seconds"), field="request timeout", maximum=30.0
    )
    trace = _read_object(Path(str(config["trace_path"])))
    raw_requests = trace.get("requests")
    if not isinstance(raw_requests, list) or not 2 <= len(raw_requests) <= 32:
        raise ValueError("runtime-impact trace must contain 2 to 32 requests")
    requests: tuple[dict[str, object], ...] = tuple(
        request for request in raw_requests if isinstance(request, dict)
    )
    if len(requests) != len(raw_requests):
        raise TypeError("every runtime-impact trace request must be an object")
    bundles = {
        "reference": Path(str(config["reference_bundle"])),
        "candidate": Path(str(config["candidate_bundle"])),
    }
    packages = {
        "reference": Path(str(config["reference_package"])),
        "candidate": Path(str(config["candidate_package"])),
    }
    semantics = {
        alternative: _semantic_replay(
            packages[alternative], alternative, generation_seed, requests
        )
        for alternative in ("reference", "candidate")
    }
    samples: list[dict[str, object]] = []
    observations: list[dict[str, object]] = []
    identity_index = 0
    if arguments.mode == "measure":
        for _ in range(warmup_count):
            for alternative in ("reference", "candidate"):
                _run_trace(
                    bundles[alternative],
                    alternative,
                    generation_seed,
                    requests,
                    timeout_seconds,
                    identity_index,
                )
                identity_index += 1
        generator = random.Random(trial_order_seed)
        order_index = 0
        for trial_index in range(repetitions):
            alternatives = ["reference", "candidate"]
            generator.shuffle(alternatives)
            for alternative in alternatives:
                duration_ns, output = _run_trace(
                    bundles[alternative],
                    alternative,
                    generation_seed,
                    requests,
                    timeout_seconds,
                    identity_index,
                )
                identity_index += 1
                encoded_output = _canonical(output)
                samples.append(
                    {
                        "alternative": alternative,
                        "trial_index": trial_index,
                        "order_index": order_index,
                        "duration_ns": duration_ns,
                        "request_count": len(requests),
                        "emitted_token_count": sum(
                            len(cast("list[object]", item["token_ids"])) for item in output
                        ),
                        "output_sha256": hashlib.sha256(encoded_output).hexdigest(),
                    }
                )
                observations.append(
                    {
                        "alternative": alternative,
                        "trial_index": trial_index,
                        "requests": output,
                    }
                )
                order_index += 1
    else:
        for alternative in ("reference", "candidate"):
            _duration_ns, output = _run_trace(
                bundles[alternative],
                alternative,
                generation_seed,
                requests,
                timeout_seconds,
                identity_index,
            )
            identity_index += 1
            observations.append(
                {"alternative": alternative, "trial_index": 0, "requests": output}
            )
    result = {
        "schema_version": "sloforge.genesis.kernel-runtime-runner/v1",
        "mode": arguments.mode,
        "samples": samples,
        "runtime_observations": observations,
        "semantics": semantics,
    }
    with arguments.output.open("x", encoding="utf-8") as handle:
        handle.write(_canonical(result).decode())
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
