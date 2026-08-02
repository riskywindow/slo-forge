"""Reproducible local CPU benchmark retaining every raw sample."""

from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import time
from collections.abc import Callable
from pathlib import Path

from quantized_state_update import (
    quantized_recurrent_state_update,
    reference_quantized_state_update,
)


def _reference_vector(previous: list[int], activation: list[float]) -> list[int]:
    return [
        reference_quantized_state_update(left, right)
        for left, right in zip(previous, activation, strict=True)
    ]


def _measure(function: Callable[[], list[int]], iterations: int) -> tuple[int, int]:
    checksum = 0
    start = time.perf_counter_ns()
    for _ in range(iterations):
        result = function()
        checksum += sum(result)
    elapsed = time.perf_counter_ns() - start
    return elapsed, checksum


def run(seed: int, repetitions: int, warmup: int, iterations: int, count: int) -> dict[str, object]:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if repetitions < 7 or warmup < 1 or iterations < 1 or not 1 <= count <= 128:
        raise ValueError("require repetitions>=7, warmup>=1, iterations>=1, count in [1,128]")
    generator = random.Random(seed)
    previous = [generator.randint(-127, 127) for _ in range(count)]
    activation = [generator.uniform(-8.0, 8.0) for _ in range(count)]

    def reference() -> list[int]:
        return _reference_vector(previous, activation)

    def candidate() -> list[int]:
        return quantized_recurrent_state_update(previous, activation, count)

    expected = reference()
    if candidate() != expected:
        raise RuntimeError("candidate failed pre-benchmark differential check")
    for _ in range(warmup):
        reference()
        candidate()

    samples: list[dict[str, int | str]] = []
    order_generator = random.Random(seed ^ 0x5A17)
    functions = {"reference": reference, "candidate": candidate}
    for repetition in range(repetitions):
        order = ["reference", "candidate"]
        order_generator.shuffle(order)
        for ordinal, name in enumerate(order):
            elapsed, checksum = _measure(functions[name], iterations)
            samples.append(
                {
                    "repetition": repetition,
                    "ordinal": ordinal,
                    "implementation": name,
                    "elapsed_ns": elapsed,
                    "nanoseconds_per_iteration": elapsed // iterations,
                    "checksum": checksum,
                }
            )
    reference_samples = [
        int(item["nanoseconds_per_iteration"])
        for item in samples
        if item["implementation"] == "reference"
    ]
    candidate_samples = [
        int(item["nanoseconds_per_iteration"])
        for item in samples
        if item["implementation"] == "candidate"
    ]
    reference_median = float(statistics.median(reference_samples))
    candidate_median = float(statistics.median(candidate_samples))
    return {
        "schema_version": "hybrid-quantized-state-update-benchmark/v1",
        "source": "measured_cpu_perf_counter_ns",
        "seed": seed,
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "timer": "time.perf_counter_ns",
        },
        "configuration": {
            "count": count,
            "warmup": warmup,
            "repetitions": repetitions,
            "iterations_per_sample": iterations,
            "output_alias_previous": False,
            "previous_stride": 1,
            "activation_stride": 1,
        },
        "correctness": {"exact_match_before_measurement": True},
        "raw_samples": samples,
        "summary": {
            "reference_median_ns_per_iteration": reference_median,
            "candidate_median_ns_per_iteration": candidate_median,
            "candidate_point_improvement_percent": (
                (reference_median - candidate_median) / reference_median * 100.0
            ),
            "acceptance": "not_evaluated_single_machine_point_estimate_only",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--repetitions", type=int, default=11)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=2_000)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = run(
        arguments.seed,
        arguments.repetitions,
        arguments.warmup,
        arguments.iterations,
        arguments.count,
    )
    payload = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if arguments.output is not None:
        if arguments.output.exists():
            raise FileExistsError(f"refusing to overwrite {arguments.output}")
        arguments.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
