"""CPU-only ForgeCI regression detection and bisection demonstration."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from sloforge.forgeci import run_fixture_evaluation


def run_forgeci_demo(*, output_directory: Path, report_path: Path, reset: bool) -> None:
    if output_directory.exists():
        if not reset:
            raise FileExistsError(f"output already exists: {output_directory}")
        resolved = output_directory.resolve()
        if resolved in {Path("/").resolve(), Path.home().resolve()}:
            raise ValueError("refusing to reset a broad directory")
        shutil.rmtree(output_directory)
    evaluation = run_fixture_evaluation(output_directory, report_path)
    if not evaluation.bisection_correct:
        raise RuntimeError("ForgeCI did not identify the fixture's first regressing commit")
    print(evaluation.model_dump_json(indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/forgeci/demo"))
    parser.add_argument("--report", type=Path, default=Path("reports/forgeci-evaluation.md"))
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    run_forgeci_demo(output_directory=args.output, report_path=args.report, reset=args.reset)


if __name__ == "__main__":
    main()
