"""Artifact-preserving Git bisection for inference performance regressions."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from sloforge.forgeci.models import (
    BisectResult,
    BisectStep,
    ComparisonClassification,
    MatrixCase,
)
from sloforge.forgeci.runner import compare_runs, run_case, write_comparison


def _git(repository: Path, *arguments: str, timeout: float = 30.0) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"git {' '.join(arguments)} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def _resolve(repository: Path, revision: str) -> str:
    return _git(repository, "rev-parse", "--verify", f"{revision}^{{commit}}")


def _candidate_commits(repository: Path, good: str, bad: str) -> list[str]:
    ancestry = _git(repository, "merge-base", "--is-ancestor", good, bad)
    if ancestry:
        raise ValueError("good commit must be an ancestor of bad commit")
    output = _git(repository, "rev-list", "--ancestry-path", "--reverse", f"{good}..{bad}")
    commits = [good, *output.splitlines()]
    if not commits or commits[-1] != bad:
        raise ValueError("bad commit is not reachable from good commit")
    return commits


def _checkout(repository: Path, revision: str) -> None:
    _git(repository, "checkout", "--detach", "--force", revision)


def bisect_regression(
    *,
    repository: Path,
    good_revision: str,
    bad_revision: str,
    case: MatrixCase,
    output_directory: Path,
    maximum_inconclusive_retries: int = 2,
) -> BisectResult:
    """Find the first likely regression without modifying the source checkout."""

    if maximum_inconclusive_retries < 0:
        raise ValueError("maximum_inconclusive_retries must be non-negative")
    source = repository.resolve()
    good = _resolve(source, good_revision)
    bad = _resolve(source, bad_revision)
    commits = _candidate_commits(source, good, bad)
    output_directory.mkdir(parents=True, exist_ok=True)
    clone_path = Path(tempfile.mkdtemp(prefix="sloforge-forgeci-bisect-"))
    steps: list[BisectStep] = []
    inconclusive: list[str] = []
    caveats: list[str] = []
    try:
        _git(clone_path.parent, "clone", "--no-hardlinks", "--quiet", str(source), str(clone_path))
        _checkout(clone_path, good)
        good_case = case.model_copy(update={"revision": good, "repository": str(source)})
        baseline = run_case(
            good_case,
            checkout=clone_path,
            output_directory=output_directory / "runs",
        )
        if not baseline.success:
            raise RuntimeError("known-good benchmark failed")

        cache: dict[str, ComparisonClassification] = {good: ComparisonClassification.UNCHANGED}

        def classify(commit: str) -> ComparisonClassification:
            if commit in cache:
                return cache[commit]
            _checkout(clone_path, commit)
            classification = ComparisonClassification.INCONCLUSIVE
            repetitions = case.benchmark.repetitions
            for attempt in range(maximum_inconclusive_retries + 1):
                candidate_case = case.model_copy(
                    update={"revision": commit, "repository": str(source)}
                )
                candidate = run_case(
                    candidate_case,
                    checkout=clone_path,
                    output_directory=output_directory / "runs",
                    repetitions=repetitions,
                )
                comparison = compare_runs(baseline, candidate, case.benchmark)
                comparison_path = output_directory / "comparisons" / f"{commit}-{attempt}.json"
                comparison_path.parent.mkdir(parents=True, exist_ok=True)
                write_comparison(comparison, comparison_path)
                classification = comparison.classification
                steps.append(
                    BisectStep(
                        commit=commit,
                        classification=classification,
                        comparison_artifact=str(comparison_path),
                        repetitions=repetitions,
                        attempt=attempt,
                    )
                )
                if classification not in {
                    ComparisonClassification.INCONCLUSIVE,
                    ComparisonClassification.FLAKY,
                    ComparisonClassification.FAILED,
                }:
                    break
                repetitions = min(repetitions * 2, case.benchmark.maximum_repetitions)
                if repetitions == steps[-1].repetitions:
                    break
            cache[commit] = classification
            if classification in {
                ComparisonClassification.INCONCLUSIVE,
                ComparisonClassification.FLAKY,
                ComparisonClassification.FAILED,
            }:
                inconclusive.append(commit)
            return classification

        bad_classification = classify(bad)
        if bad_classification != ComparisonClassification.REGRESSION:
            caveats.append("provided bad revision was not classified as a regression")
            first_bad: str | None = None
        else:
            low = 0
            high = len(commits) - 1
            while high - low > 1:
                midpoint = (low + high) // 2
                outcome = classify(commits[midpoint])
                if outcome == ComparisonClassification.REGRESSION:
                    high = midpoint
                elif outcome in {
                    ComparisonClassification.UNCHANGED,
                    ComparisonClassification.IMPROVEMENT,
                }:
                    low = midpoint
                else:
                    left = next(
                        (
                            index
                            for index in range(midpoint - 1, low, -1)
                            if classify(commits[index])
                            not in {
                                ComparisonClassification.INCONCLUSIVE,
                                ComparisonClassification.FLAKY,
                                ComparisonClassification.FAILED,
                            }
                        ),
                        low,
                    )
                    right = next(
                        (
                            index
                            for index in range(midpoint + 1, high)
                            if classify(commits[index])
                            not in {
                                ComparisonClassification.INCONCLUSIVE,
                                ComparisonClassification.FLAKY,
                                ComparisonClassification.FAILED,
                            }
                        ),
                        high,
                    )
                    if classify(commits[right]) == ComparisonClassification.REGRESSION:
                        high = right
                    else:
                        low = left
                    caveats.append(f"commit {commits[midpoint]} was statistically inconclusive")
            first_bad = commits[high]

        confidence = 0.0
        if first_bad is not None:
            relevant = [
                step
                for step in steps
                if step.commit == first_bad
                and step.classification == ComparisonClassification.REGRESSION
            ]
            if relevant:
                evidence = json.loads(Path(relevant[-1].comparison_artifact).read_text())
                corrected_alphas = [
                    float(metric["corrected_significance_level"])
                    for metric in evidence["metrics"]
                    if metric["classification"] == ComparisonClassification.REGRESSION.value
                ]
                if corrected_alphas:
                    confidence = min(1.0 - value for value in corrected_alphas)
                    confidence *= 0.9 ** len(set(inconclusive))
        result = BisectResult(
            repository=str(source),
            good_commit=good,
            bad_commit=bad,
            first_regressing_commit=first_bad,
            confidence=confidence,
            steps=tuple(steps),
            inconclusive_commits=tuple(dict.fromkeys(inconclusive)),
            artifact_directory=str(output_directory),
            caveats=tuple(caveats),
        )
        result_path = output_directory / "bisect-result.json"
        result_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        digest = hashlib.sha256(result_path.read_bytes()).hexdigest()
        (output_directory / "bisect-result.sha256").write_text(
            f"{digest}  bisect-result.json\n", encoding="utf-8"
        )
        return result
    finally:
        shutil.rmtree(clone_path, ignore_errors=True)
