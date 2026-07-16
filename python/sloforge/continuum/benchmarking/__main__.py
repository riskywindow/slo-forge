"""Standalone CPU evaluation entry point used by recorded reproduction commands."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .evaluation import run_evaluation_campaign
from .models import EvaluationRequest


def _positive_csv(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("values must be non-negative integers")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the deterministic Continuum CPU campaign")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=_positive_csv, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--initial-output-tokens", type=int, default=16)
    parser.add_argument("--delta-rounds", type=_positive_csv, default=(3, 2))
    parser.add_argument("--resumed-tokens", type=int, default=3)
    parser.add_argument("--converter-repetitions", type=int, default=5)
    parser.add_argument("--reset", action="store_true")
    arguments = parser.parse_args()
    if arguments.output.exists() and arguments.reset:
        resolved = arguments.output.resolve()
        if resolved in {Path("/"), Path.cwd().resolve(), Path.cwd().resolve().parent}:
            parser.error("refusing to reset a broad output directory")
        shutil.rmtree(arguments.output)
    result = run_evaluation_campaign(
        EvaluationRequest(
            output_dir=arguments.output,
            seeds=arguments.seeds,
            git_commit=arguments.git_commit,
            initial_output_tokens=arguments.initial_output_tokens,
            delta_rounds=arguments.delta_rounds,
            resumed_tokens=arguments.resumed_tokens,
            converter_repetitions=arguments.converter_repetitions,
        )
    )
    print(arguments.output / result.summary_artifact.path)


if __name__ == "__main__":
    main()
