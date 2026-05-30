"""Deterministic delta minimization for upstream-ready ForgeCI reproducers."""

from __future__ import annotations

from collections.abc import Callable

from sloforge.forgeci.models import (
    BenchmarkInput,
    BenchmarkSpec,
    HardwareRequirement,
    MinimalReproducer,
    MinimizationStep,
)

RegressionPredicate = Callable[[BenchmarkSpec], bool]


def minimize_benchmark(
    benchmark: BenchmarkSpec,
    *,
    good_commit: str,
    bad_commit: str,
    hardware: HardwareRequirement,
    preserves_regression: RegressionPredicate,
    expected_regression: str,
    confidence_interval: str,
    artifact_references: tuple[str, ...],
) -> MinimalReproducer:
    """Minimize request shape and runtime arguments while preserving the predicate."""

    current = benchmark
    steps: list[MinimizationStep] = []

    def attempt(field: str, value: int) -> None:
        nonlocal current
        before_input = current.input
        candidate_input = before_input.model_copy(update={field: value})
        candidate = current.model_copy(update={"input": candidate_input})
        preserved = preserves_regression(candidate)
        steps.append(
            MinimizationStep(
                field=field,
                before=str(getattr(before_input, field)),
                after=str(value),
                preserved_regression=preserved,
            )
        )
        if preserved:
            current = candidate

    for field in ("prompt_tokens", "output_tokens", "concurrency"):
        original = int(getattr(current.input, field))
        low = 1
        high = original
        while low < high:
            midpoint = (low + high) // 2
            before = current
            attempt(field, midpoint)
            if current is not before:
                high = midpoint
            else:
                low = midpoint + 1

    arguments = list(current.input.runtime_arguments)
    index = 0
    while index < len(arguments):
        candidate_arguments = tuple(arguments[:index] + arguments[index + 1 :])
        candidate_input = current.input.model_copy(
            update={"runtime_arguments": candidate_arguments}
        )
        candidate = current.model_copy(update={"input": candidate_input})
        preserved = preserves_regression(candidate)
        steps.append(
            MinimizationStep(
                field="runtime_arguments",
                before=" ".join(arguments),
                after=" ".join(candidate_arguments) or "<empty>",
                preserved_regression=preserved,
            )
        )
        if preserved:
            current = candidate
            arguments = list(candidate_arguments)
        else:
            index += 1

    command = " ".join([current.command.executable, *current.command.arguments])
    return MinimalReproducer(
        good_commit=good_commit,
        bad_commit=bad_commit,
        benchmark=current,
        steps=tuple(steps),
        reproduction_commands=(
            f"git checkout --detach {bad_commit}",
            command,
        ),
        expected_regression=expected_regression,
        confidence_interval=confidence_interval,
        hardware=hardware,
        artifact_references=artifact_references,
    )


def reproducer_environment(benchmark: BenchmarkSpec) -> tuple[tuple[str, str], ...]:
    """Return the portable input environment used by the runner."""

    value: BenchmarkInput = benchmark.input
    return (
        ("SLOFORGE_FORGECI_PROMPT_TOKENS", str(value.prompt_tokens)),
        ("SLOFORGE_FORGECI_OUTPUT_TOKENS", str(value.output_tokens)),
        ("SLOFORGE_FORGECI_CONCURRENCY", str(value.concurrency)),
    )
