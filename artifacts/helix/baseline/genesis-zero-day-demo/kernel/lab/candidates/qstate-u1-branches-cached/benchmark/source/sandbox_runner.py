"""Trusted bounded runner copied beside, but independent from, generated source."""

from __future__ import annotations

import gc
import importlib.util
import json
import random
import sys
import time
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_candidate(path: Path) -> ModuleType:
    specification = importlib.util.spec_from_file_location("generated_candidate", path)
    if specification is None or specification.loader is None:
        raise RuntimeError("candidate module cannot be loaded")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _decode_case(case: dict[str, Any]) -> tuple[list[int], list[float]]:
    previous = [int(value) for value in case["previous_storage"]]
    activations = [float(value) for value in case["activation_storage"]]
    return previous, activations


def _reference_value(previous: int, activation: float) -> int:
    import math

    if not -127 <= previous <= 127:
        raise ValueError("previous state outside symmetric int8 domain")
    if not math.isfinite(activation):
        raise ValueError("activation must be finite")
    combined = previous * 0.625 + activation * 31.0
    if math.isinf(combined):
        return 127 if combined > 0 else -127
    return max(-127, min(127, round(combined)))


def _reference(case: dict[str, Any]) -> list[int]:
    previous, activations = _decode_case(case)
    return [
        _reference_value(
            previous[case["previous_offset"] + index * case["previous_stride"]],
            activations[case["activation_offset"] + index * case["activation_stride"]],
        )
        for index in range(case["count"])
    ]


def _candidate(function: Callable[..., list[int]], case: dict[str, Any]) -> list[int]:
    previous, activations = _decode_case(case)
    return function(
        previous,
        activations,
        case["count"],
        case["previous_offset"],
        case["previous_stride"],
        case["activation_offset"],
        case["activation_stride"],
        case["output_alias_previous"],
    )


def _correctness(function: Callable[..., list[int]], cases: list[dict[str, Any]]) -> dict[str, Any]:
    results = []
    for case in cases:
        previous, activations = _decode_case(case)
        try:
            observed = function(
                previous,
                activations,
                case["count"],
                case["previous_offset"],
                case["previous_stride"],
                case["activation_offset"],
                case["activation_stride"],
                case["output_alias_previous"],
            )
            error_type = None
        except (ArithmeticError, IndexError, TypeError, ValueError) as error:
            observed = None
            error_type = type(error).__name__
        results.append(
            {
                "case_id": case["case_id"],
                "observed": observed,
                "error_type": error_type,
                "previous_storage_after": previous,
            }
        )
    return {"case_results": results, "samples": []}


def _micro_workload(
    implementation: Callable[[dict[str, Any]], list[int]],
    case: dict[str, Any],
    iterations: int,
) -> int:
    checksum = 0
    for _ in range(iterations):
        values = implementation(case)
        checksum ^= values[0]
    return checksum


def _token_workload(
    implementation: Callable[[dict[str, Any]], list[int]],
    case: dict[str, Any],
    iterations: int,
    token_steps: int,
) -> int:
    initial_previous, initial_activations = _decode_case(case)
    previous = [
        initial_previous[case["previous_offset"] + index * case["previous_stride"]]
        for index in range(case["count"])
    ]
    activations = [
        initial_activations[case["activation_offset"] + index * case["activation_stride"]]
        for index in range(case["count"])
    ]
    contiguous = {
        **case,
        "previous_offset": 0,
        "previous_stride": 1,
        "activation_offset": 0,
        "activation_stride": 1,
        "output_alias_previous": False,
    }
    checksum = 0
    for iteration in range(iterations):
        state = previous
        for step in range(token_steps):
            contiguous["previous_storage"] = state
            contiguous["activation_storage"] = [repr(value) for value in activations]
            state = implementation(contiguous)
            checksum ^= state[(iteration + step) % len(state)]
    return checksum


def _benchmark(function: Callable[..., list[int]], config: dict[str, Any]) -> dict[str, Any]:
    regimes = config["regimes"]
    samples: list[dict[str, Any]] = []
    implementations: dict[str, Callable[[dict[str, Any]], list[int]]] = {
        "reference": _reference,
        "candidate": lambda case: _candidate(function, case),
    }
    gc.disable()
    try:
        for regime_index, regime in enumerate(regimes):
            case = regime["case"]
            iterations = regime["iterations"]
            token_steps = regime["token_steps"]
            for alternative in ("reference", "candidate"):
                for _ in range(config["warmup_count"]):
                    if token_steps > 0:
                        _token_workload(implementations[alternative], case, iterations, token_steps)
                    else:
                        _micro_workload(implementations[alternative], case, iterations)
            order = ["reference", "candidate"] * config["repetitions"]
            random.Random(config["seed"] + regime_index).shuffle(order)
            trial_counts = {"reference": 0, "candidate": 0}
            for order_index, alternative in enumerate(order):
                started = time.perf_counter_ns()
                if token_steps > 0:
                    _token_workload(implementations[alternative], case, iterations, token_steps)
                else:
                    _micro_workload(implementations[alternative], case, iterations)
                duration = time.perf_counter_ns() - started
                samples.append(
                    {
                        "regime": regime["regime"],
                        "alternative": alternative,
                        "order_index": order_index,
                        "trial_index": trial_counts[alternative],
                        "iterations": iterations,
                        "duration_ns": max(1, duration),
                    }
                )
                trial_counts[alternative] += 1
    finally:
        gc.enable()
    return {"case_results": [], "samples": samples}


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit("usage: sandbox_runner.py candidate.py config.json output.json")
    candidate_path = Path(sys.argv[1])
    config = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    output_path = Path(sys.argv[3])
    # The fourth argument is an exact mode token kept separate from untrusted source.
    mode = sys.argv[4]
    module = _load_candidate(candidate_path)
    function = module.quantized_recurrent_state_update
    payload = (
        _correctness(function, config["cases"])
        if mode == "correctness"
        else _benchmark(function, config)
    )
    output_path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
